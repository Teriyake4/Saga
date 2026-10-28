#pragma once

#include "saga/backend/contracts.h"
#include "saga/memory/ledger.h"
#include "saga/runtime/completion.h"
#include "saga/runtime/sequence_slots.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace saga::runtime {

struct ArtifactComponent {
    std::string identity;
    std::uint64_t byte_size{};
};
struct ArtifactMetadata {
    std::string identity;
    std::vector<ArtifactComponent> components;
};
struct RuntimeOptions {
    std::uint64_t context_tokens{};
    bool request_mtp{false};
    bool request_vision{false};
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

struct ModelDescription {
    backend::ModelSpec spec;
    backend::ModelProgram program;
    ModelCapabilities capabilities;
};

class IExecutionSession {
public:
    virtual ~IExecutionSession() = default;
    // Enqueued work must retain every accessed resource until its physical fence and
    // consumer release; Completion abandonment/cancellation is not a physical fence.
    virtual Completion execute(const backend::ModelProgram&, ResourceLease resources) = 0;
};
class IModelRuntime {
public:
    virtual ~IModelRuntime() = default;
    virtual SequenceHandle create_sequence(std::uint64_t context_tokens) = 0;
    virtual Completion release_sequence(SequenceHandle) = 0;
};
class IGpuBackend {
public:
    virtual ~IGpuBackend() = default;
    virtual backend::DeviceCaps capabilities() const = 0;
    // Instantiation is preparation-only and must consume allocations owned by the
    // supplied bring-up grant. A grant never constitutes serving approval.
    virtual std::unique_ptr<IExecutionSession> instantiate(const ModelDescription&,
                                                            memory::BringupGrant&) = 0;
};
class IModelDefinition {
public:
    virtual ~IModelDefinition() = default;
    virtual ModelDescription describe(const ArtifactMetadata&, const RuntimeOptions&) const = 0;
    // Owner-thread and destruction requirements are implementation-defined; callers
    // must serialize operations/destruction unless an implementation states otherwise.
    virtual std::unique_ptr<IModelRuntime> create_runtime(const ModelDescription&,
                                                           IExecutionSession&) const = 0;
};

// Startup-only, externally serialized, non-reentrant registration/lookup. Returned
// shared owners remain valid after registry destruction. No fallback or auto-registration.
class ModelRegistry {
public:
    void add(std::string id, std::shared_ptr<const IModelDefinition> definition);
    std::shared_ptr<const IModelDefinition> find(const std::string& id) const;
private:
    struct Entry { std::string id; std::shared_ptr<const IModelDefinition> value; };
    std::vector<Entry> entries_;
};
class BackendRegistry {
public:
    void add(std::string id, std::shared_ptr<const IGpuBackend> backend);
    std::shared_ptr<const IGpuBackend> find(const std::string& id) const;
private:
    struct Entry { std::string id; std::shared_ptr<const IGpuBackend> value; };
    std::vector<Entry> entries_;
};

} // namespace saga::runtime
