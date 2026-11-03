#include "saga/backend/cuda.h"

#include "control.h"

#include <chrono>
#include <cstddef>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif


namespace {
void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}
void noop_kernel(const void*, const saga::backend::LaunchContext&) {}

saga::memory::StaticCandidate candidate() {
    using namespace saga::memory;
    std::vector<AllocationRecord> records;
    for (const char* id : {"a", "b", "c"}) {
        AllocationRecord r; r.id=id; r.owner="test"; r.location=Location::device;
        r.lifetime=Lifetime::execution; r.alignment=256; r.page_size=1; r.logical_bytes=256;
        records.push_back(r);
    }
    AllocationRecord pinned; pinned.id="pinned"; pinned.owner="test"; pinned.location=Location::pinned;
    pinned.lifetime=Lifetime::execution; pinned.alignment=256; pinned.page_size=1; pinned.logical_bytes=256;
    records.push_back(pinned);
    AllocationRecord runtime; runtime.id="runtime"; runtime.owner="test"; runtime.location=Location::host;
    runtime.source=AllocationSource::runtime_overhead; runtime.overhead_bound=256;
    records.push_back(runtime);
    return StaticCandidate("fake", GraphMode::disabled, std::move(records));
}
saga::memory::Budgets budgets() {
    saga::memory::Budgets b{};
    for (auto& d : b) { d.configured=1<<20; d.available=1<<20; }
    return b;
}
saga::backend::cuda::Buffer allocate(saga::backend::cuda::Executor& e,
                                      saga::memory::BringupGrant& g, const char* id,
                                      saga::backend::cuda::BufferDomain domain) {
    return e.allocate(g, id, 256, domain, 256);
}
}

int main() {
    using namespace saga::backend::cuda;
    using namespace saga::memory;
    try {
        ExecutorOptions options; options.fence_timeout=std::chrono::milliseconds(12);
        {
            Executor e(0, options);
            require(e.healthy(), "new executor must be healthy");
            fake::not_ready_queries=2;
            require(e.drain(), "event query must keep polling through NotReady");
            require(fake::not_ready_queries==0, "query poll count was not consumed");
            std::thread wrong_thread([&] { require(!e.drain(), "wrong-thread drain must fail"); });
            wrong_thread.join();
            require(!e.healthy(), "wrong-thread drain must poison");
        }

        {
            Executor e(0, options);
            fake::not_ready_queries=-1;
            require(!e.drain(), "NotReady through the deadline must time out");
            require(!e.healthy(), "timeout must poison the executor");
            const auto report=e.failure_report();
            require(report && report->reason.find("timed out")!=std::string::npos,
                    "timeout reason must be observable");
            fake::not_ready_queries=0;
        }

        {
            auto plan=candidate(); auto grant=plan.grant(budgets()); grant.observe_overhead("runtime", 0);
            const auto frees_before=fake::frees;
            {
            Executor e(0, options);
            auto a=allocate(e,grant,"a",BufferDomain::device);
            auto b=a.subview(0, 128);
            auto c=a.subview(128, 128);
            saga::backend::PreparedKernel kernel(noop_kernel, std::make_shared<int>(0), {}, {});
            e.launch(kernel, ProjectionRequest{}, a,b,c);
            a={}; b={}; c={};
            fake::fail_query=true;
            bool threw=false;
            try { e.synchronize(); } catch (const FenceFailure&) { threw=true; }
            require(threw, "failed event query must throw FenceFailure");
            const auto report=e.failure_report();
            require(report && report->unreclaimed_bytes==256 && report->unreclaimed_allocations==1,
                    "poison report must deduplicate subviews by backing identity");
            require(!e.healthy() && !e.drain(), "poisoned executor must reject drain/work");
            const auto before=grant.report().live_bytes[2];
            require(before==256, "pending leases must remain charged after caller release");
            }
            require(grant.report().live_bytes[2]==256,
                    "quarantined leases must remain charged after poisoned executor destruction");
            require(fake::frees==frees_before, "poisoned device backing must not be physically freed");
        }
        fake::fail_query=false;

        {
            auto plan=candidate(); auto grant=plan.grant(budgets()); grant.observe_overhead("runtime", 0);
            const auto frees_before=fake::frees;
            Executor e(0, options);
            auto d=allocate(e,grant,"a",BufferDomain::device);
            auto h=allocate(e,grant,"pinned",BufferDomain::pinned);
            fake::fail_copy=true;
            bool copy_failed=false;
            try { e.copy_to_device(d,0,h,0,16); } catch (const std::runtime_error&) { copy_failed=true; }
            require(copy_failed && e.healthy(), "copy submission error with successful fence must unwind safely");
            fake::fail_copy=false;
            e.copy_to_device(d,0,h,0,16);
            std::vector<std::byte> result(16);
            e.copy_to_host(result,d,0);
            require(e.healthy(), "successful buffer copies must remain healthy");
            d={}; h={};
            require(grant.report().live_bytes[2]==0 && grant.report().live_bytes[1]==0,
                    "successful fences must release transient leases");
            require(fake::frees==frees_before+2, "successful buffer backings must be physically freed");
        }

        {
            auto plan=candidate(); auto grant=plan.grant(budgets()); grant.observe_overhead("runtime", 0);
            Executor owner(0, options);
            Executor foreign(0, options);
            auto device=allocate(owner,grant,"a",BufferDomain::device);
            auto host=allocate(owner,grant,"pinned",BufferDomain::pinned);
            bool rejected=false;
            try { foreign.copy_to_device(device, 0, host, 0, 16); }
            catch (const std::invalid_argument&) { rejected=true; }
            require(rejected && foreign.healthy(), "foreign executor buffers must be rejected before submission");
            auto out_of_range=device.subview(256, 0);
            bool range_rejected=false;
            try { foreign.copy_to_device(out_of_range, 0, host, 0, 1); }
            catch (const std::invalid_argument&) { range_rejected=true; }
            require(range_rejected, "out-of-range/foreign buffer views must be rejected");
        }

        {
            const auto destroyed_before=fake::event_destroys;
            fake::event_creates=0; fake::fail_event_create_at=2;
            bool ctor_failed=false;
            try { Executor failed(0, options); } catch (const std::runtime_error&) { ctor_failed=true; }
            require(ctor_failed, "injected partial constructor failure must throw");
            require(fake::event_destroys>destroyed_before, "partial constructor must destroy initialized events");
            fake::fail_event_create_at=0;
        }

        {
            Executor e(0, options);
            fake::fail_record=true;
            bool threw=false;
            try { e.synchronize(); } catch (const FenceFailure&) { threw=true; }
            require(threw && !e.healthy(), "event-record failure must poison and throw");
            fake::fail_record=false;
        }
        {
            auto plan=candidate(); auto grant=plan.grant(budgets()); grant.observe_overhead("runtime", 0);
            Executor e(0, options);
            auto a=allocate(e,grant,"a",BufferDomain::device);
            saga::backend::PreparedKernel kernel(noop_kernel, std::make_shared<int>(0), {}, {});
            e.launch(kernel, ProjectionRequest{}, a,a,a);
            a={};
            fake::fail_set_device=true;
            bool threw=false;
            try { e.synchronize(); } catch (const FenceFailure&) { threw=true; }
            require(threw && !e.healthy(), "device-selection failure with pending work must poison synchronize");
            const auto report=e.failure_report();
            require(report && report->unreclaimed_bytes==256 && report->unreclaimed_allocations==1,
                    "device-selection failure must report pending retained allocation");
            fake::fail_set_device=false;
        }
        std::cout << "cuda executor fake-runtime tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cuda executor fake-runtime test failure: " << e.what() << '\n';
        return 1;
    }
}
