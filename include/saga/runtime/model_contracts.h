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
    // execute runs only the immutable program prebound at backend instantiation.
    // Enqueued work retains resources until physical settlement AND consumer release;
    // cancellation/abandonment is not a fence. Caller serializes calls/destruction and
    // drains physical work before destruction. Session outlives its runtime.
    virtual Completion execute(ResourceLease resources) = 0;
};
class IModelRuntime {
public:
    virtual ~IModelRuntime() = default;
    virtual SequenceHandle create_sequence(std::uint64_t context_tokens) = 0;
    // Invalidates the logical handle immediately; physical sequence storage is reclaimed
    // only after this completion and all dependent work have physically settled.
    virtual Completion release_sequence(SequenceHandle) = 0;
};
class IGpuBackend {
public:
    virtual ~IGpuBackend() = default;
    virtual backend::DeviceCaps capabilities() const = 0;
    // Preparation/calibration only, never serving approval. Must consume allocations
    // owned by the grant, validate and resolve the description, and copy every needed
    // borrowed input before returning the session with that immutable program prebound.
    virtual std::unique_ptr<IExecutionSession> instantiate(const ModelDescription&,
                                                            memory::BringupGrant&) = 0;
};
class IModelDefinition {
public:
    virtual ~IModelDefinition() = default;
    virtual ModelDescription describe(const ArtifactMetadata&, const RuntimeOptions&) const = 0;
    // Factories consume borrowed metadata/options/description synchronously and must
    // copy anything needed after return. Caller serializes operations/destruction;
    // runtime destruction precedes session destruction, after physical work drains.
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
    void add(std::string id, std::shared_ptr<IGpuBackend> backend);
    std::shared_ptr<IGpuBackend> find(const std::string& id) const;
private:
    struct Entry { std::string id; std::shared_ptr<IGpuBackend> value; };
    std::vector<Entry> entries_;
};

} // namespace saga::runtime
