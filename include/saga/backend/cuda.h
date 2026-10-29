#pragma once

#include "saga/backend/device_profile.h"
#include "saga/memory/ledger.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>

namespace saga::backend::cuda {

enum class BufferDomain : std::uint8_t { host, pinned, device };
struct DeviceInfo {
    DeviceFacts facts;
    int ordinal{};
    int driver_version{};
    int runtime_version{};
};
struct KernelRuntimeInfo {
    std::string implementation_id;
    int registers_per_thread{};
    std::uint64_t static_shared_bytes{};
    std::uint64_t local_bytes_per_thread{};
    int max_threads_per_block{};
    int binary_version{};
    int ptx_version{};
    int active_blocks_per_multiprocessor{};
};
struct ProjectionShape { std::uint64_t outputs{}, inputs{}, tokens{}; };
struct ProjectionStrides { std::uint64_t weight_row_bytes{}, activation_row_bytes{}, output_row_bytes{}; };
struct ProjectionRequest {
    ProjectionShape shape;
    ProjectionStrides strides;
    backend::Phase phase{backend::Phase::decode};
    std::uint64_t alignment{1};
};
class Executor;
class Buffer final {
public:
    Buffer() noexcept = default;
    void* address() const noexcept { return address_; }
    std::uint64_t bytes() const noexcept { return bytes_; }
    BufferDomain domain() const noexcept { return domain_; }
    int device_ordinal() const noexcept { return device_ordinal_; }
    std::uint64_t ledger_bytes() const noexcept { return lease_.bytes(); }
    Buffer subview(std::uint64_t offset, std::uint64_t bytes) const {
        if (!address_ || offset > bytes_ || bytes > bytes_ - offset)
            throw std::invalid_argument("buffer subview exceeds its parent range");
        const auto base = reinterpret_cast<std::uintptr_t>(address_);
        if (offset > UINTPTR_MAX - base) throw std::overflow_error("buffer subview address overflows uintptr_t");
        Buffer result = *this;
        result.address_ = reinterpret_cast<void*>(base + offset);
        result.bytes_ = bytes;
        return result;
    }
private:
    friend class Executor;
    Buffer(void* address, std::uint64_t bytes, BufferDomain domain, int device_ordinal,
           memory::AllocationLease lease, std::shared_ptr<const void> backing_lifetime)
        : address_(address), bytes_(bytes), domain_(domain), device_ordinal_(device_ordinal),
          lease_(std::move(lease)), backing_lifetime_(std::move(backing_lifetime)) {}
    void* address_{};
    std::uint64_t bytes_{};
    BufferDomain domain_{BufferDomain::device};
    int device_ordinal_{-1};
    memory::AllocationLease lease_;
    std::shared_ptr<const void> backing_lifetime_;
};

// The executor is single-thread-affine. launch() success means submitted, not physically complete;
// operand leases remain retained until synchronize()/a later fence. Every Buffer passed here is a
// whole allocation view: its full address/byte range must belong to this executor. Copies and timing
// calls fence before returning. Span copy sources/destinations must remain valid until return. A
// failed physical fence terminates rather than releasing potentially in-flight memory.
class Executor final {
public:
    explicit Executor(int device_ordinal = 0);
    ~Executor() noexcept;
    Executor(Executor&&) noexcept;
    Executor& operator=(Executor&&) noexcept;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    const DeviceInfo& device() const noexcept;
    Buffer allocate(memory::BringupGrant&, const std::string& ledger_id,
                    std::uint64_t reservation, BufferDomain, std::uint64_t alignment = 256);
    void copy_to_device(const Buffer& destination, std::uint64_t offset,
                        std::span<const std::byte> source);
    void copy_to_device(const Buffer& destination, std::uint64_t destination_offset,
                        const Buffer& source, std::uint64_t source_offset, std::uint64_t bytes);
    void copy_to_host(std::span<std::byte> destination, const Buffer& source,
                      std::uint64_t offset = 0);
    void copy_to_host(const Buffer& destination, std::uint64_t destination_offset,
                      const Buffer& source, std::uint64_t source_offset, std::uint64_t bytes);
    void launch(const PreparedKernel&, const ProjectionRequest&, const Buffer& weights,
                const Buffer& activations, const Buffer& output);
    void synchronize();
    float time_launch(const PreparedKernel&, const ProjectionRequest&, const Buffer& weights,
                      const Buffer& activations, const Buffer& output, std::uint32_t repetitions);
    std::uint64_t observe_runtime_overhead() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Registers real projection-only implementations. Phase is accepted only as a label for the
// identical stateless matrix operation; it does not implement prefill/decode/verify semantics.
void register_bf16_projection_kernels(KernelRegistry&);
KernelRuntimeInfo query_kernel_runtime_info(std::string_view implementation_id);
std::string cuda_compiler_version();
KernelRequest make_bf16_projection_request(const ProjectionRequest&, std::string_view implementation_id = {});

} // namespace saga::backend::cuda
