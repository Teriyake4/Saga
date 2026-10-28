#pragma once

#include "saga/kv/catalog.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace saga::backend {

enum class Operation : std::uint8_t { projection, gate_up_swiglu, attention, kv_append, recurrent_update };
enum class Phase : std::uint8_t { prefill, decode, verify };
enum class WeightEncoding : std::uint8_t { none, fp16, bf16, gsq3_group128_v1 };
enum class ActivationFormat : std::uint8_t { fp16, bf16, fp32 };
enum class Layout : std::uint8_t { contiguous, gsq3_canonical_v1, gsq3_tiled_v1, paged_kv_v1 };
enum class NumericalContract : std::uint8_t { fp32_accumulate, bf16_intermediate_rounding, fp16_intermediate_rounding };
enum class DeviceKind : std::uint8_t { cpu, accelerator };

struct DeviceCaps {
    DeviceKind kind{DeviceKind::cpu};
    std::string architecture{"host"};
    std::uint32_t architecture_version{0};
    std::uint64_t features{0};
    std::string device_identity;
};

struct Shape {
    std::uint64_t outputs{1};
    std::uint64_t inputs{1};
    std::uint64_t tokens{1};
    bool operator==(const Shape&) const = default;
};

// Strides are in bytes, not elements. Alignment is the guaranteed base alignment.
struct PhysicalContract {
    std::uint64_t weight_row_stride{1};
    std::uint64_t activation_row_stride{1};
    std::uint64_t output_row_stride{1};
    std::uint64_t alignment{1};
    std::uint32_t quantization_group{0}; // gsq3_group128_v1 requires exactly 128
    bool operator==(const PhysicalContract&) const = default;
};

struct KernelRequest {
    Operation operation{Operation::projection};
    Phase phase{Phase::decode};
    WeightEncoding weights{WeightEncoding::none};
    ActivationFormat activation{ActivationFormat::bf16};
    Layout layout{Layout::contiguous};
    NumericalContract numerics{NumericalContract::fp32_accumulate};
    std::optional<kv::Pair> kv_pair;
    Shape shape;
    PhysicalContract physical;
    std::uint64_t workspace_available{0};
    std::uint64_t workspace_alignment{1}; // always a power of two; compared only for nonzero requirements
    bool operator==(const KernelRequest&) const = default;
};

struct DimensionRange {
    std::uint64_t minimum{1};
    std::uint64_t maximum{1};
    std::uint64_t multiple{1};
};
struct ShapeConstraint {
    DimensionRange outputs;
    DimensionRange inputs;
    DimensionRange tokens;
};
struct WorkspaceRequirement {
    std::uint64_t bytes{0};
    std::uint64_t alignment{1};
    bool operator==(const WorkspaceRequirement&) const = default;
};
struct DeviceRequirement {
    DeviceKind kind{DeviceKind::cpu};
    std::string architecture{"host"};
    std::uint32_t minimum_version{0};
    std::uint32_t maximum_version{0};
    std::uint64_t required_features{0};
};

// One descriptor is one explicit format/numerical contract, not a claim of universal support.
struct KernelDescriptor {
    std::string id;
    Operation operation{Operation::projection};
    Phase phase{Phase::decode};
    WeightEncoding weights{WeightEncoding::none};
    ActivationFormat activation{ActivationFormat::bf16};
    Layout layout{Layout::contiguous};
    NumericalContract numerics{NumericalContract::fp32_accumulate};
    std::optional<kv::Pair> kv_pair;
    ShapeConstraint shape;
    PhysicalContract required_physical; // stride fields are required multiples
    WorkspaceRequirement workspace;
    DeviceRequirement device;
    int default_priority{0};
};

// Rejects malformed requests/descriptors before comparing guards, including unknown enums.
// Returns the first diagnostic. supports() may narrow, never widen the explicit contract.
std::optional<std::string> incompatibility(const KernelDescriptor&, const KernelRequest&, const DeviceCaps&);
std::string describe(const KernelRequest&, const DeviceCaps&);

struct BufferView {
    void* address{nullptr};
    std::uint64_t bytes{0};
};
struct LaunchContext {
    std::span<const BufferView> bindings;
    BufferView workspace;
    std::uint64_t execution_lane{0}; // interpreted by the backend, never a public CUDA type
};
struct KernelPreparationContext {
    const DeviceCaps& device;
    // Deliberately address-free: preparation cannot acquire unledgered device buffers.
};

using LaunchFunction = void (*)(const void* metadata, const LaunchContext&);
class IKernelImplementation;
class KernelRegistry;

class PreparedKernel {
public:
    PreparedKernel(LaunchFunction entry, std::shared_ptr<const void> metadata,
                   KernelRequest bindings, WorkspaceRequirement workspace);
    // Checks the actual workspace view before calling the nonvirtual entry. A zero-byte
    // requirement permits a null workspace; any supplied address must satisfy alignment.
    void launch(const LaunchContext& context) const;
    const KernelRequest& bindings() const noexcept { return bindings_; }
    WorkspaceRequirement workspace() const noexcept { return workspace_; }
    const std::string& implementation_id() const noexcept { return implementation_id_; }

private:
    friend class KernelRegistry;
    LaunchFunction entry_;
    // Registry bundles metadata with the implementation owner, including destruction ordering.
    std::shared_ptr<const void> metadata_;
    KernelRequest bindings_;
    WorkspaceRequirement workspace_;
    std::string implementation_id_;

};

class IKernelImplementation {
public:
    virtual ~IKernelImplementation() = default;
    virtual KernelDescriptor descriptor() const = 0;
    virtual bool supports(const KernelRequest&, const DeviceCaps&) const noexcept = 0;
    virtual PreparedKernel prepare(const KernelRequest&, KernelPreparationContext&) const = 0;
};

// Preferences only rank compatible candidates. Unknown/incompatible IDs cannot force support.
struct PairTuningProfile {
    std::vector<std::string> preferred_implementations;
};

class KernelRegistry {
public:
    // Malformed descriptors/requests throw std::invalid_argument before implementation callbacks.
    void add(std::shared_ptr<const IKernelImplementation> implementation);
    PreparedKernel resolve(const KernelRequest&, const DeviceCaps&,
                           const PairTuningProfile* preferences = nullptr) const;
private:
    struct Entry {
        std::shared_ptr<const IKernelImplementation> implementation;
        KernelDescriptor descriptor;
    };
    std::vector<Entry> entries_;
};

// A bounded, native ordered call list; not a tensor-graph interpreter or runtime stub.
struct LogicalWeightBinding {
    std::string logical_name;
    std::string physical_identity;
    WeightEncoding encoding{WeightEncoding::none};
    Shape shape;
    std::uint64_t byte_offset{0};
    std::uint64_t byte_extent{0};
};
struct ModelSpec {
    std::string model_identity;
    std::uint64_t maximum_context_tokens{0};
    kv::Geometry main_attention;
    std::vector<LogicalWeightBinding> weights;
};
struct ModelProgram {
    Phase phase{Phase::decode};
    std::vector<KernelRequest> calls;
};

} // namespace saga::backend
