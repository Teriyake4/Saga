#include "saga/backend/cuda.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace saga;
namespace {
std::uint32_t parse(const char* s, std::uint32_t fallback) {
    if (!s) return fallback;
    char* end{}; const auto v = std::strtoul(s, &end, 10);
    if (!end || *end || v == 0 || v > 10000) throw std::invalid_argument("--iterations must be in 1..10000");
    return static_cast<std::uint32_t>(v);
}
std::uint64_t extent(std::uint64_t rows, std::uint64_t stride, std::uint64_t row_bytes) {
    return (rows - 1) * stride + row_bytes;
}
std::uint64_t padded(std::uint64_t v) { return (v + 255u) & ~std::uint64_t(255u); }
std::uint64_t pinned_padded(std::uint64_t v) { return (v + 4095u) & ~std::uint64_t(4095u); }
struct Fixture {
    backend::cuda::ProjectionRequest request;
    backend::KernelRequest kernel_request;
    std::uint64_t weight_bytes{}, activation_bytes{}, output_bytes{};
    std::vector<__nv_bfloat16> weights, activations;
};
Fixture make_fixture(const backend::cuda::ProjectionRequest& r) {
    Fixture f; f.request = r; f.kernel_request = backend::cuda::make_bf16_projection_request(r);
    const auto n = r.shape.outputs, k = r.shape.inputs, t = r.shape.tokens;
    f.weight_bytes = extent(n, r.strides.weight_row_bytes, k * 2);
    f.activation_bytes = extent(t, r.strides.activation_row_bytes, k * 2);
    f.output_bytes = extent(t, r.strides.output_row_bytes, n * 4);
    f.weights.resize(static_cast<std::size_t>(f.weight_bytes / 2));
    f.activations.resize(static_cast<std::size_t>(f.activation_bytes / 2));
    for (std::uint64_t row=0; row<n; ++row) for (std::uint64_t p=0; p<k; ++p) {
        const int code = static_cast<int>((row * 7 + p * 3) % 17) - 8;
        f.weights[row * r.strides.weight_row_bytes / 2 + p] = __float2bfloat16(code * 0.0625f);
    }
    for (std::uint64_t row=0; row<t; ++row) for (std::uint64_t p=0; p<k; ++p) {
        const int code = static_cast<int>((row * 5 + p * 2) % 13) - 6;
        f.activations[row * r.strides.activation_row_bytes / 2 + p] = __float2bfloat16(code * 0.125f);
    }
    return f;
}
void add_record(std::vector<memory::AllocationRecord>& records, std::string id,
                memory::Location location, memory::Lifetime lifetime, std::uint64_t logical_bytes,
                std::uint64_t alignment = 256, std::uint64_t page_size = 1) {
    records.push_back({std::move(id), "projection-test", location, lifetime,
        memory::AllocationSource::project, true, alignment, page_size, logical_bytes, {}, {}});
}
}
int main(int argc, char** argv) try {
    if (argc > 2) throw std::invalid_argument("usage: saga_cuda_projection_test [iterations]");
    const auto iterations = parse(argc > 1 ? argv[1] : nullptr, 30);
    backend::cuda::Executor executor;
    backend::KernelRegistry registry; backend::cuda::register_bf16_projection_kernels(registry);

    // Odd dimensions and tails exercise the baseline independently of WMMA's tile packing.
    const backend::cuda::ProjectionRequest base_request{{19,23,3},{48,50,80},backend::Phase::decode,2};
    const backend::cuda::ProjectionRequest wmma_request{{32,32,32},{96,96,192},backend::Phase::decode,32};
    auto zero_shape = backend::cuda::make_bf16_projection_request(base_request);
    zero_shape.shape.outputs = 0;
    if (!backend::validate_request(zero_shape)) throw std::runtime_error("zero projection dimension passed host request validation");
    auto zero_alignment = backend::cuda::make_bf16_projection_request(base_request);
    zero_alignment.physical.alignment = 0;
    if (!backend::validate_request(zero_alignment)) throw std::runtime_error("zero projection alignment passed host request validation");
    const auto baseline = make_fixture(base_request);
    const auto tiled = make_fixture(wmma_request);
    auto prepared_base = registry.resolve(baseline.kernel_request, executor.device().facts.caps);
    auto prepared_wmma = registry.resolve(tiled.kernel_request, executor.device().facts.caps);
    if (prepared_base.implementation_id().find("baseline") == std::string::npos ||
        prepared_wmma.implementation_id().find("wmma") == std::string::npos)
        throw std::runtime_error("registry did not select the requested guarded projection variants");

    auto unsupported_wmma = tiled.kernel_request;
    unsupported_wmma.shape.outputs = 31;
    auto boundary_selection = registry.resolve(unsupported_wmma, executor.device().facts.caps);
    if (boundary_selection.implementation_id().find("baseline") == std::string::npos)
        throw std::runtime_error("WMMA tile boundary did not fall back to baseline");

    const auto weight_bytes = std::max(baseline.weight_bytes, tiled.weight_bytes);
    const auto activation_bytes = std::max(baseline.activation_bytes, tiled.activation_bytes);
    const auto output_bytes = std::max(baseline.output_bytes, tiled.output_bytes);
    std::vector<memory::AllocationRecord> records;
    add_record(records,"weights",memory::Location::device,memory::Lifetime::persistent,weight_bytes);
    add_record(records,"input",memory::Location::device,memory::Lifetime::persistent,activation_bytes);
    add_record(records,"output",memory::Location::device,memory::Lifetime::persistent,output_bytes);
    add_record(records,"staging-host",memory::Location::host,memory::Lifetime::bringup,activation_bytes);
    add_record(records,"staging-pinned",memory::Location::pinned,memory::Lifetime::bringup,weight_bytes,256,4096);
    records.push_back({"runtime","cuda-runtime",memory::Location::device,memory::Lifetime::bringup,
        memory::AllocationSource::runtime_overhead,true,1,1,{},(1ull<<30),{}});
    memory::StaticCandidate candidate("cuda-projection-test",memory::GraphMode::disabled,std::move(records));
    memory::Budgets budgets{}; for (auto& b: budgets) b={UINT64_MAX,UINT64_MAX,0};
    auto grant=candidate.grant(budgets);
    bool rejected_domain=false;
    try { (void)executor.allocate(grant,"weights",padded(weight_bytes),backend::cuda::BufferDomain::host,256); }
    catch(const std::invalid_argument&) { rejected_domain=true; }
    bool rejected_alignment=false;
    try { (void)executor.allocate(grant,"weights",padded(weight_bytes),backend::cuda::BufferDomain::device,128); }
    catch(const std::invalid_argument&) { rejected_alignment=true; }
    const auto untouched=grant.report();
    const auto untouched_weights=std::find_if(untouched.allocations.begin(),untouched.allocations.end(),
        [](const auto& allocation){return allocation.id=="weights";});
    if(!rejected_domain || !rejected_alignment || untouched_weights==untouched.allocations.end() ||
       untouched_weights->measured || untouched_weights->live_bytes!=0)
        throw std::runtime_error("domain/alignment mismatch was not rejected before physical allocation");
    auto weights=executor.allocate(grant,"weights",padded(weight_bytes),backend::cuda::BufferDomain::device,256);
    auto input=executor.allocate(grant,"input",padded(activation_bytes),backend::cuda::BufferDomain::device,256);
    auto output=executor.allocate(grant,"output",padded(output_bytes),backend::cuda::BufferDomain::device,256);
    nlohmann::json baseline_result, wmma_result;
    backend::cuda::KernelRuntimeInfo baseline_attributes, wmma_attributes;
    {
        auto pinned=executor.allocate(grant,"staging-pinned",pinned_padded(weight_bytes),backend::cuda::BufferDomain::pinned,256);
        auto host=executor.allocate(grant,"staging-host",padded(activation_bytes),backend::cuda::BufferDomain::host,256);
        auto check_variant=[&](const backend::PreparedKernel& kernel, const Fixture& fixture) {
            std::memcpy(pinned.address(),fixture.weights.data(),static_cast<std::size_t>(fixture.weight_bytes));
            std::memcpy(host.address(),fixture.activations.data(),static_cast<std::size_t>(fixture.activation_bytes));
            executor.copy_to_device(weights,0,pinned,0,fixture.weight_bytes);
            executor.copy_to_device(input,0,host,0,fixture.activation_bytes);
            executor.launch(kernel,fixture.request,weights,input,output);
            executor.synchronize();
            std::vector<float> actual(static_cast<std::size_t>(fixture.output_bytes/sizeof(float)));
            executor.copy_to_host({reinterpret_cast<std::byte*>(actual.data()),actual.size()*sizeof(float)},output);
            float max_error=0;
            for(std::uint64_t row=0;row<fixture.request.shape.tokens;++row)
                for(std::uint64_t col=0;col<fixture.request.shape.outputs;++col) {
                    float expected=0;
                    for(std::uint64_t p=0;p<fixture.request.shape.inputs;++p)
                        expected=std::fma(__bfloat162float(fixture.weights[col*fixture.request.strides.weight_row_bytes/2+p]),
                            __bfloat162float(fixture.activations[row*fixture.request.strides.activation_row_bytes/2+p]),expected);
                    max_error=std::max(max_error,std::abs(actual[row*fixture.request.strides.output_row_bytes/4+col] - expected));
                }
            if(max_error>0.02f) throw std::runtime_error("projection output differs from independent host reference");
            std::vector<float> ms; for(int rep=0;rep<5;++rep)
                ms.push_back(executor.time_launch(kernel,fixture.request,weights,input,output,iterations));
            return nlohmann::json{{"variant",kernel.implementation_id()},
                {"request",{{"operation","projection"},{"weights","bf16"},{"activation","bf16"},
                  {"output","fp32"},{"numerics","fp32_accumulate"},{"layout","contiguous"},
                  {"phase","decode"},{"outputs",fixture.request.shape.outputs},
                  {"inputs",fixture.request.shape.inputs},{"tokens",fixture.request.shape.tokens},
                  {"weight_row_stride_bytes",fixture.request.strides.weight_row_bytes},
                  {"activation_row_stride_bytes",fixture.request.strides.activation_row_bytes},
                  {"output_row_stride_bytes",fixture.request.strides.output_row_bytes},
                  {"alignment",fixture.request.alignment}}},
                {"max_abs_error",max_error},{"event_ms_per_launch",ms}};
        };
        baseline_result=check_variant(prepared_base,baseline);
        auto short_output=output.subview(0,4);
        bool rejected_extent=false;
        try { executor.launch(prepared_base,baseline.request,weights,input,short_output); }
        catch(const std::invalid_argument&) { rejected_extent=true; }
        if(!rejected_extent) throw std::runtime_error("undersized output was not rejected");
        executor.synchronize();
        bool rejected_subview=false;
        try { (void)output.subview(output.bytes()-2,4); }
        catch(const std::invalid_argument&) { rejected_subview=true; }
        if(!rejected_subview) throw std::runtime_error("out-of-range subview was not rejected");
        const auto last_byte = output.subview(output.bytes()-1,1);
        std::array<std::byte,1> last_byte_result{};
        executor.copy_to_host(std::span<std::byte>(last_byte_result),last_byte);
        const auto one_byte_view = output.subview(output.bytes()-1,1);
        bool rejected_view_overrun=false;
        try {
            std::array<std::byte,2> overrun{};
            executor.copy_to_host(std::span<std::byte>(overrun),one_byte_view);
        } catch(const std::invalid_argument&) { rejected_view_overrun=true; }
        if(!rejected_view_overrun) throw std::runtime_error("operation extending beyond subview was not rejected");
        bool rejected_alias=false;
        try { executor.launch(prepared_base,baseline.request,weights,weights,output); }
        catch(const std::invalid_argument&) { rejected_alias=true; }
        if(!rejected_alias) throw std::runtime_error("aliased projection buffers were not rejected");
        executor.synchronize();
        wmma_result=check_variant(prepared_wmma,tiled);
        baseline_attributes=backend::cuda::query_kernel_runtime_info(prepared_base.implementation_id());
        wmma_attributes=backend::cuda::query_kernel_runtime_info(prepared_wmma.implementation_id());
        if(baseline_attributes.binary_version!=89 || wmma_attributes.binary_version!=89 ||
           baseline_attributes.active_blocks_per_multiprocessor<=0 || wmma_attributes.active_blocks_per_multiprocessor<=0)
            throw std::runtime_error("projection kernels are not available as SM89 kernels with nonzero queried occupancy");
        const auto tail_view=output.subview(output.bytes()-8,8);
        std::array<std::byte,4> tail{};
        executor.copy_to_host(std::span<std::byte>(tail),tail_view,4);
    }

    bool wrong_thread_rejected=false;
    std::thread wrong_thread([&] {
        try { executor.synchronize(); }
        catch(const std::logic_error&) { wrong_thread_rejected=true; }
    });
    wrong_thread.join();
    if(!wrong_thread_rejected) throw std::runtime_error("non-owner thread use was not rejected");

    grant.mark_warmed(); executor.synchronize();
    grant.observe_overhead("runtime",executor.observe_runtime_overhead());
    const auto ledger_report=grant.report();
    for(const auto id : {"staging-host","staging-pinned"}) {
        const auto line=std::find_if(ledger_report.allocations.begin(),ledger_report.allocations.end(),
            [&](const auto& a){return a.id==id;});
        const auto expected_peak = id==std::string_view("staging-host") ? padded(activation_bytes) : pinned_padded(weight_bytes);
        if(line==ledger_report.allocations.end() || line->live_bytes!=0 || line->measured_peak!=expected_peak)
            throw std::runtime_error("transient host/pinned staging lifetime was not measured and released");
    }
    auto reconciled=std::move(grant).reconcile(); (void)reconciled;
    const auto facts=executor.device();
    const auto attributes_json=[](const backend::cuda::KernelRuntimeInfo& a) {
        return nlohmann::json{{"registers_per_thread",a.registers_per_thread},
            {"static_shared_bytes",a.static_shared_bytes},{"local_bytes_per_thread",a.local_bytes_per_thread},
            {"max_threads_per_block",a.max_threads_per_block},{"binary_version",a.binary_version},
            {"ptx_version",a.ptx_version},{"active_blocks_per_multiprocessor",a.active_blocks_per_multiprocessor}};
    };
    baseline_result["kernel_attributes"] = attributes_json(baseline_attributes);
    wmma_result["kernel_attributes"] = attributes_json(wmma_attributes);
    nlohmann::json result{{"status","physical_projection_test_passed"},{"qualification","not_claimed"},
        {"cuda_compiler",backend::cuda::cuda_compiler_version()},
        {"readiness_approval",false},{"ledger_reconciliation","passed"},
        {"device",{{"name",facts.facts.product_identity},{"architecture",facts.facts.caps.architecture},
          {"architecture_version",facts.facts.caps.architecture_version},{"identity",facts.facts.caps.device_identity},
          {"ordinal",facts.ordinal},{"driver",facts.driver_version},{"runtime",facts.runtime_version}}},
        {"iterations_per_batch",iterations},{"timing_batches_per_variant",5},
        {"variants",nlohmann::json::array({baseline_result,wmma_result})},
        {"ledger",{{"profile",candidate.profile_identity()},{"live_bytes",ledger_report.live_bytes},
          {"allocations",nlohmann::json::array()}}}};
    for(const auto& allocation:ledger_report.allocations) result["ledger"]["allocations"].push_back({
        {"id",allocation.id},{"cap_bytes",allocation.cap},{"live_bytes",allocation.live_bytes},
        {"measured_peak_bytes",allocation.measured_peak},{"measured",allocation.measured}});
    std::cout << result.dump() << '\n'; return 0;
} catch(const std::exception& e) { std::cerr << "projection test failed: " << e.what() << '\n'; return 1; }
