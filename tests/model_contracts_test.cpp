#include "saga/runtime/model_contracts.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace saga;
void check(bool ok, const char* message) { if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); } }
template<class F> void rejects(F&& f) { try { f(); } catch (const std::exception&) { return; } check(false, "expected rejection"); }
struct Model final : runtime::IModelDefinition {
    runtime::ModelDescription describe(const runtime::ArtifactMetadata&, const runtime::RuntimeOptions&) const override { return {}; }
    std::unique_ptr<runtime::IModelRuntime> create_runtime(const runtime::ModelDescription&, runtime::IExecutionSession&) const override { return {}; }
};
struct Backend final : runtime::IGpuBackend {
    backend::DeviceCaps capabilities() const override { return {}; }
    std::unique_ptr<runtime::IExecutionSession> instantiate(const runtime::ModelDescription&, memory::BringupGrant&) override { return {}; }
};
}
int main() {
    runtime::ArtifactMetadata artifact{"artifact", {{"weights", 64}}};
    backend::ModelSpec spec{"synthetic", 128, {1, 2, 4}, {{"w", "weights", backend::WeightEncoding::fp16, {2, 3, 1}, 8, 16}}};
    backend::ModelProgram program{};
    runtime::RuntimeOptions options{32, false, false};
    runtime::validate_model_contract(artifact, spec, program, options, {});
    auto invalid = spec; invalid.weights[0].byte_offset = UINT64_MAX;
    rejects([&] { runtime::validate_model_contract(artifact, invalid, program, options, {}); });
    invalid = spec; invalid.weights.push_back(invalid.weights[0]);
    rejects([&] { runtime::validate_model_contract(artifact, invalid, program, options, {}); });
    invalid = spec; invalid.weights[0].encoding = static_cast<backend::WeightEncoding>(255);
    rejects([&] { runtime::validate_model_contract(artifact, invalid, program, options, {}); });
    auto requested = options; requested.request_mtp = true;
    rejects([&] { runtime::validate_model_contract(artifact, spec, program, requested, {}); });

    std::shared_ptr<const runtime::IModelDefinition> retained;
    { runtime::ModelRegistry models; auto model = std::make_shared<Model>();
      models.add("synthetic", model); retained = models.find("synthetic");
      rejects([&] { models.add("synthetic", model); }); rejects([&] { models.add("", model); });
      rejects([&] { models.add("null", {}); }); rejects([&] { (void)models.find("missing"); }); }
    check(static_cast<bool>(retained), "definition survives registry destruction");
    runtime::BackendRegistry backends; auto backend = std::make_shared<Backend>();
    backends.add("test-backend", backend); check(backends.find("test-backend") == backend, "backend selection");
    rejects([&] { backends.add("test-backend", backend); }); rejects([&] { (void)backends.find("missing"); });
    return 0;
}
