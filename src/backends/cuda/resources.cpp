#include "saga/backend/cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <memory>
#include <limits>
#include <stdexcept>
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <string>
#include <thread>
#include <vector>
#include <chrono>

namespace saga::backend::cuda {
namespace {
void check(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorName(e) + ": " + cudaGetErrorString(e));
}
[[noreturn]] void fail_stop() noexcept { std::terminate(); }
struct BackingLife { std::atomic<bool> alive{true}; };
struct Backing {
    void* pointer{}; std::uint64_t bytes{}; BufferDomain domain{}; int ordinal{};
    std::shared_ptr<BackingLife> life{std::make_shared<BackingLife>()};
    ~Backing() noexcept {
        life->alive.store(false, std::memory_order_release);
        if (!pointer) return;
        if (domain == BufferDomain::host) {
#if defined(_WIN32)
            _aligned_free(pointer);
#else
            std::free(pointer);
#endif
            return;
        }
        if (domain == BufferDomain::pinned) {
            if (cudaFreeHost(pointer) != cudaSuccess) fail_stop();
            return;
        }
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(ordinal) != cudaSuccess) fail_stop();
        const auto result = cudaFree(pointer);
        const auto restore = cudaSetDevice(previous);
        if (result != cudaSuccess || restore != cudaSuccess) fail_stop();
    }
};
}
struct Executor::Impl {
    int ordinal{};
    cudaStream_t stream{};
    cudaEvent_t start{}, stop{}, fence_event{};
    ExecutorOptions options;
    std::optional<FenceFailureReport> failure;
    DeviceInfo info;
    std::thread::id owner{std::this_thread::get_id()};
    struct Registered { void* address; std::uint64_t bytes; BufferDomain domain; std::weak_ptr<BackingLife> life; };
    std::vector<Registered> allocations;
    std::vector<Buffer> pending_owners;
    Impl* quarantine_next{};
    inline static std::atomic<Impl*> quarantine_head{};
    void quarantine() noexcept {
        auto* head = quarantine_head.load(std::memory_order_relaxed);
        do { quarantine_next = head; }
        while (!quarantine_head.compare_exchange_weak(head, this, std::memory_order_release, std::memory_order_relaxed));
    }
    void record_failure(const char* reason) noexcept {
        poisoned = true;
        auto& report = *failure;
        try { report.reason.assign(reason); } catch (...) {}
        report.unreclaimed_bytes = 0;
        report.unreclaimed_allocations = 0;
        for (std::size_t i = 0; i < pending_owners.size(); ++i) {
            const auto& buffer = pending_owners[i];
            const auto* token = buffer.backing_lifetime_.get();
            bool duplicate = false;
            for (std::size_t j = 0; j < i; ++j)
                if (pending_owners[j].backing_lifetime_.get() == token) { duplicate = true; break; }
            if (duplicate) continue;
            const auto bytes = buffer.ledger_bytes();
            if (bytes <= UINT64_MAX - report.unreclaimed_bytes) report.unreclaimed_bytes += bytes;
            ++report.unreclaimed_allocations;
        }
    }
    void require_owned(const Buffer& b, std::uint64_t offset, std::uint64_t bytes) const {
        if (!b.address() || offset > b.bytes() || bytes > b.bytes() - offset || !b.ledger_bytes())
            throw std::invalid_argument("buffer has no live ledger lease or its requested range is invalid");
        const auto view = reinterpret_cast<std::uintptr_t>(b.address());
        if (b.bytes() > UINTPTR_MAX - view || offset > UINTPTR_MAX - view)
            throw std::overflow_error("buffer view or offset overflows uintptr_t");
        const auto begin = view + offset;
        if (bytes > UINTPTR_MAX - begin) throw std::overflow_error("requested buffer range overflows uintptr_t");
        for (const auto& item : allocations) {
            if (item.domain != b.domain()) continue;
            const auto life = item.life.lock();
            if (!life || !life->alive.load(std::memory_order_acquire)) continue;
            const auto base = reinterpret_cast<std::uintptr_t>(item.address);
            if (view >= base && view - base <= item.bytes && b.bytes() <= item.bytes - (view - base) &&
                begin >= base && begin - base <= item.bytes && bytes <= item.bytes - (begin - base)) return;
        }
        throw std::invalid_argument("buffer view/range is not within a live allocation owned by this CUDA executor");
    }
    std::uint64_t runtime_baseline{};
    bool poisoned{};
    explicit Impl(int device, ExecutorOptions opts) : ordinal(device), options(opts) {
        if (options.fence_timeout.count() <= 0) throw std::invalid_argument("fence timeout must be positive");
        check(cudaSetDevice(ordinal), "cudaSetDevice");
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, ordinal), "cudaGetDeviceProperties");
        info.ordinal = ordinal;
        info.facts.caps.kind = DeviceKind::accelerator;
        info.facts.caps.architecture = prop.major == 8 && prop.minor == 9 ? "sm89" : "sm" + std::to_string(prop.major) + std::to_string(prop.minor);
        info.facts.caps.architecture_version = static_cast<std::uint32_t>(prop.major * 10 + prop.minor);
        constexpr char hex[] = "0123456789abcdef";
        info.facts.caps.device_identity.clear();
        for (const auto byte : prop.uuid.bytes) {
            const auto value = static_cast<unsigned char>(byte);
            info.facts.caps.device_identity.push_back(hex[value >> 4]);
            info.facts.caps.device_identity.push_back(hex[value & 15]);
        }
        info.facts.product_identity = prop.name;
        info.facts.warp_size = prop.warpSize;
        info.facts.multiprocessor_count = prop.multiProcessorCount;
        std::size_t free_bytes{}, total_bytes{}; check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        info.facts.device_memory_budget_bytes = free_bytes;
        runtime_baseline = total_bytes - free_bytes;
        failure.emplace();
        failure->reason.reserve(128);
        check(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
        check(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
        try {
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
            check(cudaEventCreate(&start), "cudaEventCreate(start)");
            check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
            check(cudaEventCreate(&fence_event), "cudaEventCreate(fence)");

        } catch (...) {
            if ((start || stop || fence_event || stream) && cudaSetDevice(ordinal) != cudaSuccess) fail_stop();
            if (fence_event && cudaEventDestroy(fence_event) != cudaSuccess) fail_stop();
            if (start && cudaEventDestroy(start) != cudaSuccess) fail_stop();
            if (stop && cudaEventDestroy(stop) != cudaSuccess) fail_stop();
            if (stream && cudaStreamDestroy(stream) != cudaSuccess) fail_stop();
            throw;
        }
    }
    void current() {
        if (std::this_thread::get_id() != owner) throw std::logic_error("CUDA executor used from a non-owner thread");
        if (poisoned) throw FenceFailure("CUDA executor is poisoned after a failed physical fence");
        const auto result = cudaSetDevice(ordinal);
        if (result != cudaSuccess) {
            record_failure(cudaGetErrorString(result));
            throw FenceFailure(failure->reason);
        }
    }
    void fence() {
        if (poisoned) throw FenceFailure("CUDA executor is poisoned after a failed physical fence");
        const auto recorded = cudaEventRecord(fence_event, stream);
        auto result = recorded;
        const auto deadline = std::chrono::steady_clock::now() + options.fence_timeout;
        bool timed_out = false;
        if (result == cudaSuccess) {
            for (;;) {
                result = cudaEventQuery(fence_event);
                if (result != cudaErrorNotReady) break;
                if (std::chrono::steady_clock::now() >= deadline) { timed_out = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (result != cudaSuccess || timed_out) {
            record_failure(recorded != cudaSuccess ? cudaGetErrorString(recorded) :
                timed_out ? "fence event query timed out" : cudaGetErrorString(result));
            throw FenceFailure(failure->reason);
        }
        pending_owners.clear();
    }
    ~Impl() noexcept {
        if (!stream || poisoned) return;
        if (cudaSetDevice(ordinal) != cudaSuccess) return;
        if (fence_event) (void)cudaEventDestroy(fence_event);
        if (start) (void)cudaEventDestroy(start);
        if (stop) (void)cudaEventDestroy(stop);
        (void)cudaStreamDestroy(stream);
    }
};
Executor::Executor(int ordinal) : Executor(ordinal, ExecutorOptions{}) {}
Executor::Executor(int ordinal, ExecutorOptions options) : impl_(std::make_unique<Impl>(ordinal, options)) {}
Executor::~Executor() noexcept {
    if (!impl_) return;
    if (!impl_->poisoned) (void)drain();
    if (impl_->poisoned) {
        auto* abandoned = impl_.release();
        abandoned->quarantine();
    }
}
Executor::Executor(Executor&& other) noexcept : impl_(std::move(other.impl_)) {}
Executor& Executor::operator=(Executor&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            if (!impl_->poisoned) (void)drain();
            if (impl_->poisoned) {
                auto* abandoned = impl_.release();
                abandoned->quarantine();
            }
        }
        impl_ = std::move(other.impl_);
    }
    return *this;
}
const DeviceInfo& Executor::device() const noexcept { return impl_->info; }
bool Executor::drain() noexcept {
    if (!impl_ || impl_->poisoned) return false;
    try { impl_->current(); impl_->fence(); return true; }
    catch (...) {
        if (!impl_->poisoned) impl_->record_failure("drain failed before completion could be verified");
        return false;
    }
}
bool Executor::healthy() const noexcept { return impl_ && !impl_->poisoned; }
std::optional<FenceFailureReport> Executor::failure_report() const {
    return impl_ && impl_->poisoned ? impl_->failure : std::nullopt;
}
Buffer Executor::allocate(memory::BringupGrant& grant, const std::string& id, std::uint64_t reserve,
                          BufferDomain domain, std::uint64_t alignment) {
    const auto contract = grant.allocation_contract(id);
    const auto declared_domain = contract.location == memory::Location::host ? BufferDomain::host :
        contract.location == memory::Location::pinned ? BufferDomain::pinned : BufferDomain::device;
    if (domain != declared_domain)
        throw std::invalid_argument("requested buffer domain does not match granted allocation location: " + id);
    if (!alignment || (alignment & (alignment - 1))) throw std::invalid_argument("allocation alignment must be a power of two");
    if (alignment < contract.alignment)
        throw std::invalid_argument("requested allocation alignment is below the granted base alignment: " + id);
    impl_->current();
    if (domain != BufferDomain::host && domain != BufferDomain::pinned && domain != BufferDomain::device)
        throw std::invalid_argument("unknown buffer memory domain");
    if (reserve % alignment != 0) throw std::invalid_argument("ledger reservation must already be padded to requested alignment");
    void* address = nullptr;
    std::uint64_t actual_bytes = 0;
    std::shared_ptr<BackingLife> life;
    auto lease = grant.allocate(id, reserve, [&](std::uint64_t cap) {
        if (!cap || cap > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("allocation cap is zero or exceeds size_t");
        auto backing = std::make_unique<Backing>(); backing->bytes = cap; backing->domain = domain; backing->ordinal = impl_->ordinal;
        if (domain == BufferDomain::host) {
            const auto host_alignment = std::max<std::uint64_t>(alignment, alignof(void*));
            if (cap > std::numeric_limits<std::size_t>::max() - (host_alignment - 1))
                throw std::overflow_error("host allocation padding overflows size_t");
            const auto padded = (static_cast<std::size_t>(cap) + host_alignment - 1) & ~(static_cast<std::size_t>(host_alignment) - 1);
#if defined(_WIN32)
            backing->pointer = _aligned_malloc(padded, static_cast<std::size_t>(host_alignment));
            if (!backing->pointer) throw std::bad_alloc();
#else
            if (posix_memalign(&backing->pointer, static_cast<std::size_t>(host_alignment), padded) != 0) throw std::bad_alloc();
#endif
            backing->bytes = cap;
        } else if (domain == BufferDomain::pinned) check(cudaHostAlloc(&backing->pointer, static_cast<std::size_t>(cap), cudaHostAllocPortable), "cudaHostAlloc");
        else check(cudaMalloc(&backing->pointer, static_cast<std::size_t>(cap)), "cudaMalloc");
        if (domain != BufferDomain::host && reinterpret_cast<std::uintptr_t>(backing->pointer) % alignment)
            throw std::runtime_error("CUDA allocator did not satisfy requested alignment");
        address = backing->pointer; actual_bytes = backing->bytes; life = backing->life;
        return memory::MemoryBacking{memory::OwnedBacking(std::move(backing)), actual_bytes};
    });
    Buffer result(address, actual_bytes, domain, domain == BufferDomain::device ? impl_->ordinal : -1,
                  std::move(lease), life);
    std::erase_if(impl_->allocations, [](const auto& item) {
        const auto live = item.life.lock();
        return !live || !live->alive.load(std::memory_order_acquire);
    });
    impl_->allocations.push_back({address, actual_bytes, domain, life});
    return result;
}
void Executor::copy_to_device(const Buffer& dst, std::uint64_t offset, std::span<const std::byte> src) {
    impl_->current();
    if (dst.domain() != BufferDomain::device || dst.device_ordinal() != impl_->ordinal) throw std::invalid_argument("invalid ledger-backed device copy destination/domain");
    impl_->require_owned(dst, offset, src.size());
    impl_->pending_owners.push_back(dst);
    try {
        check(cudaMemcpyAsync(static_cast<std::byte*>(dst.address()) + offset, src.data(), src.size(), cudaMemcpyHostToDevice, impl_->stream), "cudaMemcpyAsync(H2D)");
        impl_->fence();
    } catch (...) {
        if (impl_->poisoned) fail_stop(); // borrowed host memory may still be read by the device
        try { impl_->fence(); } catch (...) { fail_stop(); }
        throw;
    }
}
void Executor::copy_to_device(const Buffer& dst, std::uint64_t dst_offset, const Buffer& src,
                              std::uint64_t src_offset, std::uint64_t bytes) {
    impl_->current();
    if (dst.domain() != BufferDomain::device || src.domain() == BufferDomain::device || dst.device_ordinal() != impl_->ordinal)
        throw std::invalid_argument("buffer copy requires host/pinned source and this executor's device destination");
    impl_->require_owned(dst, dst_offset, bytes); impl_->require_owned(src, src_offset, bytes);
    impl_->pending_owners.reserve(impl_->pending_owners.size() + 2);
    impl_->pending_owners.push_back(dst);
    impl_->pending_owners.push_back(src);
    try {
        check(cudaMemcpyAsync(static_cast<std::byte*>(dst.address()) + dst_offset,
            static_cast<const std::byte*>(src.address()) + src_offset, static_cast<std::size_t>(bytes),
            cudaMemcpyHostToDevice, impl_->stream), "cudaMemcpyAsync(buffer H2D)");
        impl_->fence();
    } catch (...) { impl_->fence(); throw; }
}
void Executor::copy_to_host(std::span<std::byte> dst, const Buffer& src, std::uint64_t offset) {
    impl_->current();
    if (src.domain() != BufferDomain::device || src.device_ordinal() != impl_->ordinal) throw std::invalid_argument("invalid ledger-backed device copy source/domain");
    impl_->require_owned(src, offset, dst.size());
    impl_->pending_owners.push_back(src);
    try {
        check(cudaMemcpyAsync(dst.data(), static_cast<const std::byte*>(src.address()) + offset, dst.size(), cudaMemcpyDeviceToHost, impl_->stream), "cudaMemcpyAsync(D2H)");
        impl_->fence();
    } catch (...) {
        if (impl_->poisoned) fail_stop(); // borrowed host destination may still be written by the device
        try { impl_->fence(); } catch (...) { fail_stop(); }
        throw;
    }
}
void Executor::copy_to_host(const Buffer& dst, std::uint64_t dst_offset, const Buffer& src,
                            std::uint64_t src_offset, std::uint64_t bytes) {
    impl_->current();
    if (src.domain() != BufferDomain::device || dst.domain() == BufferDomain::device || src.device_ordinal() != impl_->ordinal)
        throw std::invalid_argument("buffer copy requires this executor's device source and host/pinned destination");
    impl_->require_owned(dst, dst_offset, bytes); impl_->require_owned(src, src_offset, bytes);
    impl_->pending_owners.reserve(impl_->pending_owners.size() + 2);
    impl_->pending_owners.push_back(dst);
    impl_->pending_owners.push_back(src);
    try {
        check(cudaMemcpyAsync(static_cast<std::byte*>(dst.address()) + dst_offset,
            static_cast<const std::byte*>(src.address()) + src_offset, static_cast<std::size_t>(bytes),
            cudaMemcpyDeviceToHost, impl_->stream), "cudaMemcpyAsync(buffer D2H)");
        impl_->fence();
    } catch (...) { impl_->fence(); throw; }
}
void Executor::launch(const PreparedKernel& kernel, const ProjectionRequest& req, const Buffer& w,
                      const Buffer& x, const Buffer& y) {
    impl_->current();
    for (const auto* b : {&w, &x, &y}) {
        if (b->domain() != BufferDomain::device || b->device_ordinal() != impl_->ordinal) throw std::invalid_argument("projection requires device buffers on this executor's device");
        impl_->require_owned(*b, 0, b->bytes());
    }
    const BufferView views[]{{w.address(), w.bytes()}, {x.address(), x.bytes()}, {y.address(), y.bytes()}};
    auto r = make_bf16_projection_request(req);
    if (kernel.bindings() != r) throw std::invalid_argument("prepared projection request differs from launch request");
    impl_->pending_owners.reserve(impl_->pending_owners.size() + 3);
    impl_->pending_owners.push_back(w);
    impl_->pending_owners.push_back(x);
    impl_->pending_owners.push_back(y);
    try {
        kernel.launch({views, {}, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(impl_->stream))});
        check(cudaPeekAtLastError(), "projection kernel launch");
    } catch (...) { impl_->fence(); throw; }
}
void Executor::synchronize() { impl_->current(); impl_->fence(); }
float Executor::time_launch(const PreparedKernel& kernel, const ProjectionRequest& req, const Buffer& w,
                            const Buffer& x, const Buffer& y, std::uint32_t reps) {
    impl_->current(); if (!reps || reps > 10000) throw std::invalid_argument("timing repetitions must be in 1..10000");
    try {
        check(cudaEventRecord(impl_->start, impl_->stream), "cudaEventRecord(start)");
        for (std::uint32_t i = 0; i < reps; ++i) launch(kernel, req, w, x, y);
        check(cudaEventRecord(impl_->stop, impl_->stream), "cudaEventRecord(stop)");
        impl_->fence();
        float ms{}; check(cudaEventElapsedTime(&ms, impl_->start, impl_->stop), "cudaEventElapsedTime");
        return ms / reps;
    } catch (...) {
        // Events, vector growth, launch validation, and launch errors can all fail after earlier
        // repetitions have entered the stream. Only a successful stream fence permits lease release.
        impl_->fence();
        throw;
    }
}
std::uint64_t Executor::observe_runtime_overhead() const {
    impl_->current();
    std::size_t free_bytes{}, total_bytes{}; check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo(overhead)");
    const auto used = total_bytes - free_bytes;
    std::uint64_t project_bytes = 0;
    for (const auto& allocation : impl_->allocations) {
        const auto life = allocation.life.lock();
        if (allocation.domain == BufferDomain::device && life && life->alive.load(std::memory_order_acquire)) {
            if (allocation.bytes > UINT64_MAX - project_bytes) throw std::overflow_error("tracked CUDA allocation total overflows");
            project_bytes += allocation.bytes;
        }
    }
    const auto delta = used > impl_->runtime_baseline ? used - impl_->runtime_baseline : 0;
    return delta > project_bytes ? delta - project_bytes : 0;
}
} // namespace saga::backend::cuda
