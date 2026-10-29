#pragma once

#include "saga/backend/contracts.h"
#include "saga/backend/device_profile.h"
#include "saga/memory/planner.h"
#include "saga/memory/ledger.h"
#include "saga/runtime/completion.h"
#include "saga/runtime/sequence_slots.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

#include <optional>
#include <string>
#include <vector>

namespace saga::runtime {

class ExecutionPlan;

// Identities and sizes are declarations; validation does not verify component payload bytes.
struct ArtifactComponent {
    std::string identity;
    std::uint64_t byte_size{};
};
enum class Capability : std::uint8_t { mtp, vision };
struct ArtifactMetadata {
    std::string identity;
    std::vector<ArtifactComponent> components;
    std::vector<Capability> required_capabilities;
    std::vector<Capability> optional_capabilities;
};
struct RuntimeOptions {
    std::uint64_t context_tokens{};
    bool request_mtp{false};
    bool request_vision{false};
    memory::GraphMode graphs{memory::GraphMode::disabled};
};
struct ModelCapabilities {
    bool mtp{false};
    bool vision{false};
};

// Structural metadata validation only. It does not inspect payload bytes, packing,
// numerical correctness, device qualification, allocation fit, or readiness.
void validate_model_contract(const ArtifactMetadata&, const backend::ModelSpec&,
                             const backend::ModelProgram&, const RuntimeOptions&,
                             ModelCapabilities);

enum class FrontendPolicy : std::uint8_t { caller_token_ids, vocabulary_ids };
struct FrontendContract {
    std::string identity;
    FrontendPolicy policy{FrontendPolicy::caller_token_ids};
    std::uint32_t vocabulary_size{};
    std::vector<std::uint32_t> special_token_ids;
};
enum class StateKind : std::uint8_t { key_value, recurrent, convolution, auxiliary };
enum class StateDType : std::uint8_t { u8, i8, f16, bf16, f32, i32, i64 };
enum class StateLifetime : std::uint8_t { sequence, request, step };
struct StateSchema {
    std::string id;
    StateKind kind{StateKind::key_value};
    StateDType dtype{StateDType::u8};
    std::vector<std::uint64_t> dimensions;
    memory::Location location{memory::Location::device};
    StateLifetime lifetime{StateLifetime::sequence};
    std::string resource_id;
};
struct ModelInspection {
    ArtifactMetadata artifact;
    backend::ModelSpec spec;
    FrontendContract frontend;
    std::vector<StateSchema> state;
    ModelCapabilities capabilities;
};
struct ModelDescription {
    ModelInspection inspection;
    memory::Profile resources;
    std::vector<backend::ModelProgram> programs;
};

struct StepRow {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint64_t> positions;
    std::uint64_t context_tokens{};
    std::uint32_t chunk_tokens{};
};
struct PrefillRequest {
    SequenceHandle sequence;
    std::vector<StepRow> rows;
};
struct DecodeRequest {
    SequenceHandle sequence;
    std::vector<StepRow> rows;
};
struct VerifyRequest {
    SequenceHandle sequence;
    std::vector<StepRow> rows;
};
void validate_step_request(const PrefillRequest&, const FrontendContract&, std::uint64_t maximum_context_tokens);
void validate_step_request(const DecodeRequest&, const FrontendContract&, std::uint64_t maximum_context_tokens);
void validate_step_request(const VerifyRequest&, const FrontendContract&, std::uint64_t maximum_context_tokens);
struct StepRowResult {
    std::uint64_t consumed_frontier{};
    std::vector<float> logits;
    std::vector<std::byte> data;
};
inline constexpr std::size_t maximum_step_rows = 1;
inline constexpr std::size_t maximum_step_logits = 1U << 20;
inline constexpr std::size_t maximum_step_result_bytes = 16U << 20;

struct StepResultBounds {
    explicit StepResultBounds(std::uint64_t frontier,
                              std::optional<std::size_t> logits_per_row = std::nullopt)
        : maximum_frontier(frontier), expected_logits_per_row(logits_per_row) {}
    std::uint64_t maximum_frontier;
    std::optional<std::size_t> expected_logits_per_row;
};
struct OwningStepResult { std::vector<StepRowResult> rows; };

namespace detail { struct StepResultState; }
class StepSource;
class PendingStep {
public:
    PendingStep() noexcept = default;
    bool valid() const noexcept { return completion_.valid(); }
    CompletionStatus status() const { return completion_.status(); }
    bool is_ready() const { return completion_.is_ready(); }
    Outcome wait() const { return completion_.wait(); }
    Outcome wait_for(std::chrono::steady_clock::duration timeout) const { return completion_.wait_for(timeout); }
    OwningStepResult result() const;
private:
    friend class StepSource;
    PendingStep(Completion completion, std::shared_ptr<detail::StepResultState> result) noexcept;
    Completion completion_;
    std::shared_ptr<detail::StepResultState> result_;
};
class StepSource {
public:
    StepSource(ResourceLease retained, StepResultBounds bounds);
    StepSource(const StepSource&) = delete;
    StepSource& operator=(const StepSource&) = delete;
    StepSource(StepSource&&) noexcept = default;
    StepSource& operator=(StepSource&&) noexcept = default;
    PendingStep pending() const;
    bool succeed(OwningStepResult result) noexcept;
    bool fail(std::exception_ptr error = {}) noexcept;
    bool cancel() noexcept;
private:
    CompletionSource completion_;
    std::shared_ptr<detail::StepResultState> result_;
    StepResultBounds bounds_;
};

class IExecutionSession {
public:
    virtual ~IExecutionSession() = default;
    // Input/output allocations in resources must survive physical settlement and release of
    // every PendingStep consumer; cancellation/abandonment is not a physical-work fence.
    // Callers serialize calls and destruction and drain physical work before teardown.
    virtual PendingStep prefill(PrefillRequest request, ResourceLease resources) = 0;
    virtual PendingStep decode(DecodeRequest request, ResourceLease resources) = 0;
    virtual std::optional<PendingStep> verify(VerifyRequest, ResourceLease) {
        return std::nullopt;
    }
    // Startup-only sessions may perform asynchronous physical preparation; successful
    // completion is a physical-success proof, not merely logical settlement. drain() must
    // block until all physical work is settled; if it cannot prove that fence, the backend must
    // fail-stop rather than return and allow owners to unwind.
    virtual Completion warmup(memory::BringupGrant&) = 0;
    virtual Completion capture(memory::BringupGrant&) = 0;
    virtual std::vector<std::pair<std::string, std::uint64_t>> overhead_measurements(memory::BringupGrant&) = 0;
    // Blocking physical barrier. Must not return until all submitted work is physically settled.
    virtual void drain() noexcept = 0;
};
class IModelRuntime {
public:
    virtual ~IModelRuntime() = default;
    virtual SequenceHandle create_sequence(std::uint64_t context_tokens) = 0;
    virtual PendingStep prefill(PrefillRequest, ResourceLease) = 0;
    virtual PendingStep decode(DecodeRequest, ResourceLease) = 0;
    virtual std::optional<PendingStep> verify(VerifyRequest, ResourceLease) { return std::nullopt; }
    // Invalidates the logical handle immediately; backing is reclaimed only after this
    // completion and all dependent physical work settle.
    virtual Completion release_sequence(SequenceHandle) = 0;
};
class IGpuBackend {
public:
    virtual ~IGpuBackend() = default;
    virtual backend::DeviceFacts device_facts() const = 0;
    // Preparation only. The backend must consume this exact plan and grant, and copy borrowed
    // inputs needed after return; arbitrary descriptions cannot be instantiated. If construction
    // schedules physical work and then throws before returning a session, the backend retains the
    // partial session/resources, drains to a physical fence, and releases them only after that
    // fence. The coordinator cannot drain a session it was never given; returning/throwing before
    // the backend can prove settlement is a backend contract violation (fail-stop instead).
    virtual std::unique_ptr<IExecutionSession> instantiate(const ExecutionPlan&,
                                                            memory::BringupGrant&) = 0;
};
class IModelDefinition {
public:
    virtual ~IModelDefinition() = default;
    virtual ModelInspection inspect(const ArtifactMetadata&) const = 0;
    virtual std::vector<backend::ModelProgram> describe_programs(
        const ModelInspection&, const RuntimeOptions&) const = 0;
    virtual memory::Profile describe_resources(const ModelInspection&, const RuntimeOptions&) const = 0;
    // Factories consume borrowed arguments synchronously and copy anything retained later.
    // Runtime destruction precedes session destruction after physical work drains. If factory
    // construction starts asynchronous work and then throws before returning a runtime, it must
    // retain and physically drain that work itself before throwing.
    virtual std::unique_ptr<IModelRuntime> create_runtime(const ModelDescription&,
                                                           IExecutionSession&) const = 0;
};
void validate_model_description(const ModelDescription&, const RuntimeOptions&);

// Startup-only externally serialized lookup. Returned owners outlive registry destruction.
class ModelRegistry {
public:
    void add(std::string id, std::shared_ptr<const IModelDefinition> value);
    std::shared_ptr<const IModelDefinition> find(const std::string& id) const;
private:
    struct Entry { std::string id; std::shared_ptr<const IModelDefinition> value; };
    std::vector<Entry> entries_;
};
class BackendRegistry {
public:
    void add(std::string id, std::shared_ptr<IGpuBackend> value);
    std::shared_ptr<IGpuBackend> find(const std::string& id) const;
private:
    struct Entry { std::string id; std::shared_ptr<IGpuBackend> value; };
    std::vector<Entry> entries_;
};

} // namespace saga::runtime
