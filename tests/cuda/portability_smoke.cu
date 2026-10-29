#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "../../src/backends/cuda/resource_owners.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <type_traits>

using saga::cuda_internal::DeviceBuffer;

namespace {
constexpr int warp_size = 32;
constexpr int matrix_elements = 16 * 16;
constexpr int expected_sms = 76;

// Each lane issues one aligned 16-byte transaction. There is deliberately no
// synchronous-copy fallback: an older code image cannot pass this probe.
__global__ void cp_async_smoke(const uint4* source, uint4* destination, int* compiled_arch) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) *compiled_arch = __CUDA_ARCH__;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) uint4 staging[warp_size];
    const unsigned lane = threadIdx.x;
    const unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(&staging[lane]));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 "cp.async.commit_group;\n"
                 "cp.async.wait_group 0;\n"
                 : : "r"(shared_address), "l"(source + lane) : "memory");
    __syncthreads();
    destination[lane] = staging[lane];
#endif
}

__global__ void bf16_wmma_smoke(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                float* destination, int* compiled_arch) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) *compiled_arch = __CUDA_ARCH__;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> left;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> right;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    // All 32 lanes execute every WMMA operation with identical parameters.
    wmma::load_matrix_sync(left, a, 16);
    wmma::load_matrix_sync(right, b, 16);
    wmma::fill_fragment(accumulator, 0.5f);
    wmma::mma_sync(accumulator, left, right, accumulator);
    wmma::store_matrix_sync(destination, accumulator, 16, wmma::mem_row_major);
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(result) +
                                 " (" + std::to_string(static_cast<int>(result)) + "): " +
                                 cudaGetErrorString(result));
    }
}

// Escape bytes outside printable ASCII too, so even unexpected driver-provided
// names cannot break the single JSON object emitted on stdout.
std::string json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

struct Check {
    const char* name;
    const char* status = "not_run";
    std::string detail;
};

struct KernelResources {
    const char* name;
    bool available = false;
    cudaFuncAttributes attributes{};
    int active_blocks_per_sm = 0;
};

struct Report {
    int device = 0;
    int device_count = -1;
    int driver_version = 0;
    int runtime_version = 0;
    bool properties_available = false;
    cudaDeviceProp properties{};
    bool memory_available = false;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    int cooperative_attribute = -1;
    int cp_async_arch = 0;
    int bf16_arch = 0;
    saga::cuda_internal::CleanupReport cleanup;
    std::string error;
    KernelResources cp_async{"cp_async_smoke"};
    KernelResources bf16{"bf16_wmma_smoke"};
    std::vector<Check> checks{
        {"arguments"}, {"driver_runtime_versions"}, {"device_selection"},
        {"device_properties"}, {"original_rtx_4080"}, {"compute_capability_sm89"},
        {"sm_count_76"}, {"memory_query"}, {"cooperative_support_query"},
        {"cp_async_resources"}, {"bf16_wmma_resources"}, {"allocations_alignment"},
        {"resource_initialization"}, {"cp_async_launch_and_sync"}, {"cp_async_result"},
        {"bf16_wmma_launch_and_sync"}, {"bf16_wmma_result"},
        {"resource_cleanup"}, {"allocation_cleanup"}
    };
};

template <typename Function>
void step(Report& report, std::string_view name, Function&& function) {
    for (auto& check : report.checks) {
        if (name != check.name) continue;
        try {
            function();
            check.status = "pass";
        } catch (const std::exception& error) {
            check.status = "fail";
            check.detail = error.what();
            throw;
        }
        return;
    }
    throw std::logic_error("unknown probe check");
}



template <typename Kernel>
void query_resources(Kernel kernel, KernelResources& result) {
    cuda_check(cudaFuncGetAttributes(&result.attributes, kernel), "cudaFuncGetAttributes");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &result.active_blocks_per_sm, kernel, warp_size, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    result.available = true;
    require(result.attributes.binaryVersion == 89, "loaded kernel binary is not SM89");
    require(result.attributes.maxThreadsPerBlock >= warp_size, "kernel cannot launch one warp");
    require(result.active_blocks_per_sm > 0, "kernel has no resident one-warp block capacity");
}

void run_probe(Report& report, int argc, char** argv) {
    step(report, "arguments", [&] {
        require(argc == 1 || (argc == 3 && std::string_view(argv[1]) == "--device"),
                "usage: saga_cuda_portability_smoke [--device nonnegative-ordinal]");
        if (argc == 3) {
            const std::string_view argument(argv[2]);
            const auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), report.device);
            require(parsed.ec == std::errc{} && parsed.ptr == argument.data() + argument.size() &&
                        report.device >= 0, "device ordinal must be a nonnegative integer");
        }
    });
    step(report, "driver_runtime_versions", [&] {
        cuda_check(cudaDriverGetVersion(&report.driver_version), "cudaDriverGetVersion");
        cuda_check(cudaRuntimeGetVersion(&report.runtime_version), "cudaRuntimeGetVersion");
    });
    std::optional<saga::cuda_internal::DeviceGuard> device_guard;
    step(report, "device_selection", [&] {
        cuda_check(cudaGetDeviceCount(&report.device_count), "cudaGetDeviceCount");
        require(report.device < report.device_count, "selected CUDA device is unavailable");
        device_guard.emplace(report.device, report.cleanup);
    });
    step(report, "device_properties", [&] {
        cuda_check(cudaGetDeviceProperties(&report.properties, report.device), "cudaGetDeviceProperties");
        report.properties_available = true;
    });
    step(report, "original_rtx_4080", [&] {
        const std::string_view name(report.properties.name);
        require(name == "NVIDIA GeForce RTX 4080" || name == "GeForce RTX 4080",
                "requires original desktop RTX 4080; Super, laptop and other devices are rejected");
    });
    step(report, "compute_capability_sm89", [&] {
        require(report.properties.major == 8 && report.properties.minor == 9,
                "requires compute capability 8.9");
        require(report.properties.warpSize == warp_size, "requires 32-lane warps");
    });
    step(report, "sm_count_76", [&] {
        require(report.properties.multiProcessorCount == expected_sms, "original RTX 4080 must report 76 SMs");
    });
    step(report, "memory_query", [&] {
        cuda_check(cudaFree(nullptr), "cudaFree(nullptr) context initialization");
        cuda_check(cudaMemGetInfo(&report.free_bytes, &report.total_bytes), "cudaMemGetInfo");
        report.memory_available = true;
        require(report.free_bytes <= report.total_bytes && report.total_bytes > 0,
                "inconsistent free/total device memory");
    });
    step(report, "cooperative_support_query", [&] {
        cuda_check(cudaDeviceGetAttribute(&report.cooperative_attribute, cudaDevAttrCooperativeLaunch,
                                         report.device), "cudaDeviceGetAttribute(cooperativeLaunch)");
        require(report.cooperative_attribute == report.properties.cooperativeLaunch,
                "cooperative-launch property and attribute disagree");
    });
    step(report, "cp_async_resources", [&] { query_resources(cp_async_smoke, report.cp_async); });
    step(report, "bf16_wmma_resources", [&] { query_resources(bf16_wmma_smoke, report.bf16); });


    std::array<uint4, warp_size> source{};
    std::array<uint4, warp_size> copied{};
    std::array<__nv_bfloat16, matrix_elements> a{};
    std::array<__nv_bfloat16, matrix_elements> b{};
    std::array<float, matrix_elements> expected{};
    std::array<float, matrix_elements> actual{};
    for (unsigned lane = 0; lane < warp_size; ++lane) {
        source[lane] = make_uint4(0x12340001u + lane, 0xabcd0001u + lane * 7u,
                                 0x80000001u + lane * 13u, 0xffff0001u - lane);
    }
    for (int row = 0; row < 16; ++row) {
        for (int column = 0; column < 16; ++column) {
            const int index = row * 16 + column;
            const float value = static_cast<float>((row * 5 + column * 3) % 17 - 8);
            a[index] = __float2bfloat16_rn(row == column ? 1.0f : 0.0f);
            b[index] = __float2bfloat16_rn(value);
            expected[index] = value + 0.5f;
        }
    }

    DeviceBuffer<uint4> input(report.cleanup);
    DeviceBuffer<uint4> output(report.cleanup);
    DeviceBuffer<__nv_bfloat16> left(report.cleanup);
    DeviceBuffer<__nv_bfloat16> right(report.cleanup);
    DeviceBuffer<float> product(report.cleanup);
    DeviceBuffer<int> arch(report.cleanup);
    saga::cuda_internal::Event completion(report.cleanup);
    saga::cuda_internal::Stream stream(report.cleanup);
    step(report, "resource_initialization", [&] {
        require(stream.get() != nullptr && completion.get() != nullptr,
                "nonblocking stream or completion event was not initialized");
    });
    step(report, "allocations_alignment", [&] {
        input.allocate(warp_size);
        output.allocate(warp_size);
        left.allocate(matrix_elements);
        right.allocate(matrix_elements);
        product.allocate(matrix_elements);
        arch.allocate(1);
        require(reinterpret_cast<std::uintptr_t>(input.get()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(output.get()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(left.get()) % 32 == 0 &&
                    reinterpret_cast<std::uintptr_t>(right.get()) % 32 == 0 &&
                    reinterpret_cast<std::uintptr_t>(product.get()) % 32 == 0,
                "cp.async/WMMA device buffer alignment requirements not met");
    });
    step(report, "cp_async_launch_and_sync", [&] {
        cuda_check(cudaMemcpyAsync(input.get(), source.data(), sizeof(source), cudaMemcpyHostToDevice, stream.get()),
                   "cudaMemcpyAsync(cp.async input H2D)");
        cuda_check(cudaMemsetAsync(output.get(), 0, sizeof(copied), stream.get()), "cudaMemsetAsync(cp.async output)");
        cuda_check(cudaMemsetAsync(arch.get(), 0, sizeof(int), stream.get()), "cudaMemsetAsync(cp.async arch)");
        cp_async_smoke<<<1, warp_size, 0, stream.get()>>>(input.get(), output.get(), arch.get());
        cuda_check(cudaGetLastError(), "cp_async_smoke launch");
        completion.record(stream.get());
        completion.synchronize();
    });
    step(report, "cp_async_result", [&] {
        cuda_check(cudaMemcpyAsync(copied.data(), output.get(), sizeof(copied), cudaMemcpyDeviceToHost, stream.get()),
                   "cudaMemcpyAsync(cp.async output D2H)");
        cuda_check(cudaMemcpyAsync(&report.cp_async_arch, arch.get(), sizeof(int), cudaMemcpyDeviceToHost, stream.get()),
                   "cudaMemcpyAsync(cp.async arch D2H)");
        completion.record(stream.get());
        completion.synchronize();
        require(report.cp_async_arch == 890, "cp.async kernel was not compiled for __CUDA_ARCH__=890");
        for (int lane = 0; lane < warp_size; ++lane) {
            const auto& x = source[lane];
            const auto& y = copied[lane];
            require(x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w,
                    "cp.async data mismatch at lane " + std::to_string(lane));
        }
    });
    step(report, "bf16_wmma_launch_and_sync", [&] {
        cuda_check(cudaMemcpyAsync(left.get(), a.data(), sizeof(a), cudaMemcpyHostToDevice, stream.get()),
                   "cudaMemcpyAsync(WMMA A H2D)");
        cuda_check(cudaMemcpyAsync(right.get(), b.data(), sizeof(b), cudaMemcpyHostToDevice, stream.get()),
                   "cudaMemcpyAsync(WMMA B H2D)");
        cuda_check(cudaMemsetAsync(product.get(), 0, sizeof(actual), stream.get()),
                   "cudaMemsetAsync(WMMA output)");
        cuda_check(cudaMemsetAsync(arch.get(), 0, sizeof(int), stream.get()), "cudaMemsetAsync(WMMA arch)");
        bf16_wmma_smoke<<<1, warp_size, 0, stream.get()>>>(left.get(), right.get(), product.get(), arch.get());
        cuda_check(cudaGetLastError(), "bf16_wmma_smoke launch");
        completion.record(stream.get()); completion.synchronize();
    });
    step(report, "bf16_wmma_result", [&] {
        cuda_check(cudaMemcpyAsync(actual.data(), product.get(), sizeof(actual), cudaMemcpyDeviceToHost, stream.get()),
                   "cudaMemcpyAsync(WMMA output D2H)");
        cuda_check(cudaMemcpyAsync(&report.bf16_arch, arch.get(), sizeof(int), cudaMemcpyDeviceToHost, stream.get()),
                   "cudaMemcpyAsync(WMMA arch D2H)");
        completion.record(stream.get());
        completion.synchronize();
        require(report.bf16_arch == 890, "WMMA kernel was not compiled for __CUDA_ARCH__=890");
        for (int i = 0; i < matrix_elements; ++i) {
            require(actual[i] == expected[i], "BF16 WMMA mismatch at row " + std::to_string(i / 16) +
                    ", column " + std::to_string(i % 16) + ": expected " +
                    std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
        }
    });
}

void write_resources(const KernelResources& resource, const Report& report) {
    std::cout << "{\"name\":" << json_string(resource.name) << ",\"queried\":" << resource.available;
    if (resource.available) {
        const auto& a = resource.attributes;
        const auto resident_grid = static_cast<long long>(resource.active_blocks_per_sm) *
                                   report.properties.multiProcessorCount;
        std::cout << ",\"binary_version\":" << a.binaryVersion << ",\"ptx_version\":" << a.ptxVersion
                  << ",\"registers_per_thread\":" << a.numRegs << ",\"static_shared_bytes\":" << a.sharedSizeBytes
                  << ",\"local_bytes_per_thread\":" << a.localSizeBytes
                  << ",\"max_threads_per_block\":" << a.maxThreadsPerBlock
                  << ",\"launch_blocks\":1,\"threads_per_block\":32,\"dynamic_shared_bytes\":0"
                  << ",\"active_blocks_per_sm\":" << resource.active_blocks_per_sm
                  << ",\"resident_warps_per_sm\":" << resource.active_blocks_per_sm
                  << ",\"resident_grid_block_bound\":" << resident_grid
                  << ",\"cooperative_grid_block_bound\":";
        if (report.cooperative_attribute == 1) std::cout << resident_grid;
        else std::cout << "null";
    }
    std::cout << '}';
}

void write_report(const Report& report) {
    std::cout << std::boolalpha;
    std::cout << "{\"schema_version\":1,\"probe\":\"saga_cuda_portability_smoke\",\"passed\":"
              << report.error.empty() << ",\"error\":" << json_string(report.error)
              << ",\"toolchain\":{\"nvcc_major\":" << __CUDACC_VER_MAJOR__
              << ",\"nvcc_minor\":" << __CUDACC_VER_MINOR__ << ",\"nvcc_build\":" << __CUDACC_VER_BUILD__
              << ",\"cuda_headers_version\":" << CUDART_VERSION << ",\"cplusplus\":" << __cplusplus;
#ifdef _MSC_VER
    std::cout << ",\"host_compiler\":\"MSVC\",\"host_compiler_version\":" << _MSC_FULL_VER;
#elif defined(__clang__)
    std::cout << ",\"host_compiler\":\"Clang\",\"host_compiler_version\":" << json_string(__clang_version__);
#elif defined(__GNUC__)
    std::cout << ",\"host_compiler\":\"GCC\",\"host_compiler_version\":" << json_string(__VERSION__);
#else
    std::cout << ",\"host_compiler\":\"unknown\"";
#endif
#ifdef _WIN32
    std::cout << ",\"os\":\"Windows\"";
#elif defined(__linux__)
    std::cout << ",\"os\":\"Linux\"";
#else
    std::cout << ",\"os\":\"other\"";
#endif
    std::cout << "},\"driver_version\":" << report.driver_version
              << ",\"runtime_version\":" << report.runtime_version
              << ",\"device_ordinal\":" << report.device << ",\"device_count\":" << report.device_count
              << ",\"device\":";
    if (report.properties_available) {
        const auto& p = report.properties;
        std::cout << "{\"name\":" << json_string(p.name) << ",\"compute_major\":" << p.major
                  << ",\"compute_minor\":" << p.minor << ",\"sm_count\":" << p.multiProcessorCount
                  << ",\"expected_sm_count\":76,\"warp_size\":" << p.warpSize
                  << ",\"total_global_mem_bytes\":" << p.totalGlobalMem
                  << ",\"cooperative_launch_property\":" << (p.cooperativeLaunch != 0)
                  << ",\"cooperative_launch_attribute\":" << report.cooperative_attribute << '}';
    } else std::cout << "null";
    std::cout << ",\"memory_before_allocations\":";
    if (report.memory_available) {
        std::cout << "{\"free_bytes\":" << report.free_bytes << ",\"total_bytes\":" << report.total_bytes << '}';
    } else std::cout << "null";
    std::cout << ",\"expected_compile_arch\":890,\"cp_async_compiled_arch\":" << report.cp_async_arch
              << ",\"bf16_wmma_compiled_arch\":" << report.bf16_arch
              << ",\"cooperative_launch_exercised\":false,\"kernels\":[";
    write_resources(report.cp_async, report);
    std::cout << ',';
    write_resources(report.bf16, report);
    std::cout << "],\"checks\":[";
    bool first = true;
    for (const auto& check : report.checks) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"name\":" << json_string(check.name) << ",\"status\":" << json_string(check.status)
                  << ",\"detail\":" << json_string(check.detail) << '}';
    }
    std::cout << "]}\n";
}
} // namespace

int main(int argc, char** argv) {
    Report report;
    try {
        run_probe(report, argc, argv);
    } catch (const std::exception& error) {
        report.error = error.what();
    }
    try {
        step(report, "resource_cleanup", [&] {
            cuda_check(report.cleanup.first_error, "CUDA resource cleanup before allocation cleanup");
        });
    } catch (const std::exception& error) {
        if (!report.error.empty()) report.error += "; ";
        report.error += error.what();
    }
    try {
        step(report, "allocation_cleanup", [&] {
            cuda_check(report.cleanup.first_error, "CUDA resource cleanup (stream, event, buffers, device restoration)");
        });
    } catch (const std::exception& error) {
        if (!report.error.empty()) report.error += "; ";
        report.error += error.what();
    }
    write_report(report);
    return report.error.empty() && std::cout.good() ? 0 : 1;
}
