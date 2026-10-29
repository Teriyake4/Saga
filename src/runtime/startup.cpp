#include "saga/runtime/startup.h"

#include "saga/memory/checked.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace saga::runtime {

ExecutionPlan::ExecutionPlan(ModelDescription description, RuntimeOptions options,
    backend::DeviceFacts facts, backend::DeviceProfile device_profile,
    std::optional<backend::PairProfile> pair_profile,
    std::vector<backend::PreparedProgram> programs, memory::StaticCandidate candidate,
    std::string identity, std::vector<std::string> context_attempt_diagnostics)
    : description_(std::move(description)), options_(options), facts_(std::move(facts)),
      device_profile_(std::move(device_profile)), pair_profile_(std::move(pair_profile)),
      programs_(std::move(programs)), candidate_(std::move(candidate)), identity_(std::move(identity)),
      context_attempt_diagnostics_(std::move(context_attempt_diagnostics)) {}

namespace {
void require_success(const Completion& completion, const char* stage) {
    if (!completion.valid()) throw std::runtime_error(std::string(stage) + " returned no completion proof");
    const auto outcome = completion.wait();
    if (outcome.status != CompletionStatus::success) {
        if (outcome.error) std::rethrow_exception(outcome.error);
        throw std::runtime_error(std::string(stage) + " did not physically complete successfully");
    }
}
struct CanonicalWriter {
    std::string bytes{"saga-plan-canonical-v2"};
    void add(std::string_view tag, std::string_view value) {
        bytes += std::to_string(tag.size()) + ":" + std::string(tag);
        bytes += std::to_string(value.size()) + ":" + std::string(value);
    }
    template<class T> void number(std::string_view tag, T value) { add(tag, std::to_string(value)); }
    void boolean(std::string_view tag, bool value) { add(tag, value ? "1" : "0"); }
    void optional_number(std::string_view tag, const std::optional<std::uint64_t>& value) {
        boolean(std::string(tag) + ".present", value.has_value());
        if (value) number(tag, *value);
    }
};
void serialize_request(CanonicalWriter& w, const backend::KernelRequest& r) {
    w.number("request.operation", static_cast<unsigned>(r.operation));
    w.number("request.phase", static_cast<unsigned>(r.phase));
    w.number("request.weights", static_cast<unsigned>(r.weights));
    w.number("request.activation", static_cast<unsigned>(r.activation));
    w.number("request.layout", static_cast<unsigned>(r.layout));
    w.number("request.numerics", static_cast<unsigned>(r.numerics));
    w.boolean("request.kv.present", r.kv_pair.has_value());
    if (r.kv_pair) { w.number("request.kv.key", static_cast<unsigned>(r.kv_pair->key));
        w.number("request.kv.value", static_cast<unsigned>(r.kv_pair->value)); }
    w.number("request.shape.outputs", r.shape.outputs); w.number("request.shape.inputs", r.shape.inputs);
    w.number("request.shape.tokens", r.shape.tokens);
    w.number("request.physical.weight_stride", r.physical.weight_row_stride);
    w.number("request.physical.activation_stride", r.physical.activation_row_stride);
    w.number("request.physical.output_stride", r.physical.output_row_stride);
    w.number("request.physical.alignment", r.physical.alignment);
    w.number("request.physical.group", r.physical.quantization_group);
    w.number("request.workspace.available", r.workspace_available);
    w.number("request.workspace.alignment", r.workspace_alignment);
}
void serialize_artifact(CanonicalWriter& w, const ArtifactMetadata& a) {
    w.add("artifact.identity", a.identity); w.number("artifact.components.count", a.components.size());
    for (const auto& c : a.components) { w.add("artifact.component.identity", c.identity); w.number("artifact.component.bytes", c.byte_size); }
    w.number("artifact.required.count", a.required_capabilities.size());
    for (auto c : a.required_capabilities) w.number("artifact.required", static_cast<unsigned>(c));
    w.number("artifact.optional.count", a.optional_capabilities.size());
    for (auto c : a.optional_capabilities) w.number("artifact.optional", static_cast<unsigned>(c));
}
std::string canonical_identity(const ModelDescription& d, const RuntimeOptions& o,
    const backend::DeviceFacts& f, const backend::DeviceProfile& p,
    const std::optional<backend::PairProfile>& pair,
    const std::vector<backend::PreparedProgram>& programs, const memory::StaticCandidate& candidate,
    std::string_view model_id, std::string_view backend_id, const memory::Budgets& budgets,
    bool explicit_context, const std::vector<std::uint64_t>& contexts) {
    CanonicalWriter w;
    w.add("registry.model_id", model_id); w.add("registry.backend_id", backend_id);
    serialize_artifact(w, d.inspection.artifact);
    w.add("model.identity", d.inspection.spec.model_identity);
    w.number("model.max_context", d.inspection.spec.maximum_context_tokens);
    w.number("model.geometry.layers", d.inspection.spec.main_attention.layers);
    w.number("model.geometry.heads", d.inspection.spec.main_attention.heads);
    w.number("model.geometry.channels", d.inspection.spec.main_attention.head_channels);
    w.number("model.weights.count", d.inspection.spec.weights.size());
    for (const auto& x : d.inspection.spec.weights) {
        w.add("weight.logical", x.logical_name); w.add("weight.physical", x.physical_identity);
        w.number("weight.encoding", static_cast<unsigned>(x.encoding)); w.number("weight.outputs", x.shape.outputs);
        w.number("weight.inputs", x.shape.inputs); w.number("weight.tokens", x.shape.tokens);
        w.number("weight.offset", x.byte_offset); w.number("weight.extent", x.byte_extent); w.add("weight.component", x.component_identity);
    }
    w.add("frontend.identity", d.inspection.frontend.identity);
    w.number("frontend.policy", static_cast<unsigned>(d.inspection.frontend.policy));
    w.number("frontend.vocabulary", d.inspection.frontend.vocabulary_size);
    w.number("frontend.special.count", d.inspection.frontend.special_token_ids.size());
    for (auto x : d.inspection.frontend.special_token_ids) w.number("frontend.special", x);
    w.number("state.count", d.inspection.state.size());
    for (const auto& s : d.inspection.state) {
        w.add("state.id", s.id); w.number("state.kind", static_cast<unsigned>(s.kind));
        w.number("state.dtype", static_cast<unsigned>(s.dtype)); w.number("state.dimensions.count", s.dimensions.size()); for (auto n : s.dimensions) w.number("state.dimension", n);
        w.number("state.location", static_cast<unsigned>(s.location)); w.number("state.lifetime", static_cast<unsigned>(s.lifetime)); w.add("state.resource", s.resource_id);
    }
    w.boolean("capability.mtp", d.inspection.capabilities.mtp); w.boolean("capability.vision", d.inspection.capabilities.vision);
    w.number("options.context", o.context_tokens); w.boolean("options.context.explicit", explicit_context);
    w.boolean("options.mtp", o.request_mtp); w.boolean("options.vision", o.request_vision);
    w.number("options.graphs", static_cast<unsigned>(o.graphs));
    w.number("context.candidates.count", contexts.size());
    for (const auto context : contexts) w.number("context.candidate", context);
    w.add("device.profile.identity", p.identity()); w.number("device.profile.kind", static_cast<unsigned>(p.kind()));
    w.add("device.profile.arch", p.architecture()); w.number("device.profile.version", p.architecture_version());
    w.add("device.profile.physical", p.physical_device_identity()); w.add("device.profile.product", p.product_identity());
    w.boolean("device.profile.warp.present", p.warp_size().has_value()); if (p.warp_size()) w.number("device.profile.warp", *p.warp_size());
    w.boolean("device.profile.sm.present", p.multiprocessor_count().has_value()); if (p.multiprocessor_count()) w.number("device.profile.sm", *p.multiprocessor_count());
    w.optional_number("device.profile.memory", p.minimum_device_memory_budget_bytes());
    w.optional_number("device.profile.workspace", p.minimum_workspace_budget_bytes());
    w.number("device.profile.preferences.count", p.preferred_implementations().size());
    for (const auto& id : p.preferred_implementations()) w.add("device.profile.preference", id);
    w.number("facts.kind", static_cast<unsigned>(f.caps.kind)); w.add("facts.arch", f.caps.architecture);
    w.number("facts.version", f.caps.architecture_version); w.number("facts.features", f.caps.features);
    w.add("facts.device", f.caps.device_identity); w.add("facts.product", f.product_identity);
    w.boolean("facts.warp.present", f.warp_size.has_value()); if (f.warp_size) w.number("facts.warp", *f.warp_size);
    w.boolean("facts.sm.present", f.multiprocessor_count.has_value()); if (f.multiprocessor_count) w.number("facts.sm", *f.multiprocessor_count);
    w.optional_number("facts.memory", f.device_memory_budget_bytes); w.optional_number("facts.workspace", f.workspace_budget_bytes);
    w.boolean("pair.present", pair.has_value()); if (pair) {
        w.add("pair.model", pair->model_identity()); w.add("pair.device_profile", pair->device_profile_identity());
        w.add("pair.physical", pair->physical_device_identity()); w.number("pair.preferences.count", pair->preferred_implementations().size());
        for (const auto& id : pair->preferred_implementations()) w.add("pair.preference", id);
    }
    w.add("resources.identity", d.resources.identity); w.number("resources.max_context", d.resources.model_max_context);
    w.boolean("resources.draft", d.resources.draft_enabled); w.boolean("resources.vision", d.resources.vision_enabled);
    w.number("resources.graphs", static_cast<unsigned>(d.resources.graphs));
    const auto cache = [&](std::string_view name, const std::optional<memory::CacheSpec>& c) {
        w.boolean(std::string(name) + ".present", c.has_value());
        if (!c) return;
        w.boolean(std::string(name) + ".geometry.present", c->geometry.has_value());
        if (c->geometry) { w.number(std::string(name) + ".layers", c->geometry->layers); w.number(std::string(name) + ".heads", c->geometry->heads); w.number(std::string(name) + ".channels", c->geometry->head_channels); }
        w.boolean(std::string(name) + ".codecs.present", c->codecs.has_value());
        if (c->codecs) { w.number(std::string(name) + ".key", static_cast<unsigned>(c->codecs->key)); w.number(std::string(name) + ".value", static_cast<unsigned>(c->codecs->value)); }
        w.boolean(std::string(name) + ".layout.present", c->layout.has_value());
        if (c->layout) { w.number(std::string(name) + ".tokens_per_page", c->layout->tokens_per_page); w.number(std::string(name) + ".plane_alignment", c->layout->plane_alignment); w.number(std::string(name) + ".table_bytes", c->layout->table_bytes_per_page); }
        w.optional_number(std::string(name) + ".verification", c->verification_tokens);
        w.optional_number(std::string(name) + ".mtp_alignment", c->mtp_alignment);
        w.number(std::string(name) + ".allocation_alignment", c->allocation_alignment); w.number(std::string(name) + ".page_size", c->page_size);
    };
    cache("resources.main_cache", d.resources.main_cache); cache("resources.draft_cache", d.resources.draft_cache);
    w.number("resources.terms.count", d.resources.terms.size());
    for (const auto& t : d.resources.terms) {
        w.add("term.id", t.id); w.add("term.owner", t.owner); w.number("term.category", static_cast<unsigned>(t.category));
        w.number("term.location", static_cast<unsigned>(t.location)); w.number("term.lifetime", static_cast<unsigned>(t.lifetime));
        w.optional_number("term.bytes", t.bytes); w.optional_number("term.bytes_per_context", t.bytes_per_context);
        w.optional_number("term.overhead", t.overhead_bound); w.number("term.alignment", t.alignment);
        w.number("term.page", t.page_size); w.add("term.physical", t.physical_identity);
    }
    for (const auto& b : budgets) { w.number("budget.configured", b.configured); w.number("budget.available", b.available); w.number("budget.headroom", b.headroom); }
    w.number("programs.count", d.programs.size());
    for (const auto& program : d.programs) {
        w.number("program.phase", static_cast<unsigned>(program.phase));
        w.number("program.calls.count", program.calls.size());
        for (const auto& r : program.calls) serialize_request(w, r);
        w.number("program.alternatives.count", program.alternatives.size());
        for (const auto& alt : program.alternatives) { w.add("alternative.identity", alt.identity); w.number("alternative.calls.count", alt.calls.size()); for (const auto& r : alt.calls) serialize_request(w, r); }
    }
    w.number("selected.programs.count", programs.size());
    for (const auto& program : programs) {
        w.add("selected.alternative", program.selected_alternative_identity()); w.number("selected.kernels.count", program.kernels().size());
        for (const auto& k : program.kernels()) { w.add("selected.kernel", k.implementation_id()); serialize_request(w, k.bindings());
            w.number("selected.workspace.bytes", k.workspace().bytes); w.number("selected.workspace.alignment", k.workspace().alignment); }
    }
    w.number("allocations.count", candidate.unique_allocations().size());
    for (const auto& line : candidate.unique_allocations()) {
        const auto& a = line.allocation; w.add("allocation.id", a.id); w.add("allocation.owner", a.owner);
        w.number("allocation.location", static_cast<unsigned>(a.location)); w.number("allocation.lifetime", static_cast<unsigned>(a.lifetime));
        w.number("allocation.source", static_cast<unsigned>(a.source)); w.boolean("allocation.enabled", a.enabled);
        w.number("allocation.alignment", a.alignment); w.number("allocation.page", a.page_size);
        w.optional_number("allocation.bytes", a.logical_bytes); w.optional_number("allocation.overhead", a.overhead_bound);
        w.add("allocation.sharing", a.sharing_identity);
    }
    return std::move(w.bytes);
}
}

StartupCoordinator::StartupCoordinator(const ModelRegistry& models, const BackendRegistry& backends)
    : models_(models), backends_(backends) {}

ApprovedRuntime StartupCoordinator::start(const StartupRequest& request) const {
    if (!request.kernels) throw std::invalid_argument("startup requires a kernel registry");
    auto model = models_.find(request.model_id);
    auto backend = backends_.find(request.backend_id);
    auto facts = backend->device_facts();
    if (facts.caps.kind == backend::DeviceKind::cpu && facts.caps.architecture != "host")
        throw std::invalid_argument("CPU backend must report the explicit host architecture");
    if (!request.device_profile.matches(facts)) throw std::invalid_argument("observed device facts do not match selected profile");
    auto inspection = model->inspect(request.artifact);
    const auto same_artifact = [](const ArtifactMetadata& a, const ArtifactMetadata& b) {
        if (a.identity != b.identity || a.components.size() != b.components.size() ||
            a.required_capabilities != b.required_capabilities || a.optional_capabilities != b.optional_capabilities) return false;
        for (std::size_t i = 0; i < a.components.size(); ++i)
            if (a.components[i].identity != b.components[i].identity || a.components[i].byte_size != b.components[i].byte_size) return false;
        return true;
    };
    if (!same_artifact(inspection.artifact, request.artifact))
        throw std::invalid_argument("model inspection changed caller-supplied artifact metadata");
    if (request.graphs != memory::GraphMode::disabled && request.graphs != memory::GraphMode::enabled)
        throw std::invalid_argument("invalid startup graph mode");
    if (request.options.graphs != request.graphs)
        throw std::invalid_argument("runtime options graph mode conflicts with startup graph selection");
    if (request.options.context_tokens == 0 && (request.auto_contexts.empty() || request.auto_contexts.size() > 256))
        throw std::invalid_argument("auto context candidates must be finite and bounded to 1..256");
    const auto candidates = request.options.context_tokens ? std::vector<std::uint64_t>{request.options.context_tokens}
                                                            : request.auto_contexts;
    if (candidates.empty()) throw std::invalid_argument("startup requires explicit context or finite auto candidates");
    std::vector<std::uint64_t> ordered = candidates;
    std::sort(ordered.begin(), ordered.end(), std::greater<>());
    if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
        throw std::invalid_argument("context candidates must be unique");
    for (const auto context : ordered)
        if (context == 0 || context > inspection.spec.maximum_context_tokens)
            throw std::invalid_argument("context candidate is outside model limits");
    if (request.pair_profile && (request.pair_profile->model_identity() != inspection.spec.model_identity ||
        request.pair_profile->device_profile_identity() != request.device_profile.identity() ||
        request.pair_profile->physical_device_identity() != facts.caps.device_identity))
        throw std::invalid_argument("pair profile does not match inspected model and observed device");
    std::string infeasible;
    std::vector<std::string> context_diagnostics;
    for (const auto context : ordered) {
        RuntimeOptions options = request.options;
        options.context_tokens = context;
        options.graphs = request.graphs;
        auto programs = model->describe_programs(inspection, options);
        auto resources = model->describe_resources(inspection, options);
        resources.budgets = request.budgets;
        if (resources.graphs != options.graphs)
            throw std::invalid_argument("model resource graph mode does not match effective runtime options");
        ModelDescription description{inspection, std::move(resources), std::move(programs)};
        validate_model_description(description, options);
        auto evaluation = memory::evaluate(description.resources, context);
        if (!evaluation.preparation_eligible) {
            const auto reason = evaluation.unresolved.empty() ? std::string("static candidate is infeasible") : evaluation.unresolved.front();
            std::string attempt = "context=" + std::to_string(context) + " statically rejected: ";
            if (evaluation.unresolved.empty()) attempt += reason;
            else for (std::size_t i = 0; i < evaluation.unresolved.size(); ++i) {
                if (i) attempt += ", ";
                attempt += evaluation.unresolved[i];
            }
            context_diagnostics.push_back(attempt);
            infeasible += (infeasible.empty() ? "" : "; ") + attempt;
            if (request.options.context_tokens) throw std::runtime_error("explicit context is infeasible: " + reason);
            continue;
        }
        auto candidate = std::move(*evaluation.candidate);
        auto prepared = std::vector<backend::PreparedProgram>{};
        prepared.reserve(description.programs.size());
        for (const auto& program : description.programs)
            prepared.push_back(backend::resolve_program(description.inspection.spec.model_identity, program,
                facts, request.device_profile, *request.kernels,
                request.pair_profile ? &*request.pair_profile : nullptr));

        // The backend's selected workspace must be covered by a known declared term in each domain.
        for (const auto& program : prepared) for (const auto& kernel : program.kernels()) {
            const auto requirement = kernel.workspace();
            if (!requirement.bytes) continue;
            const bool covered = std::any_of(description.resources.terms.begin(), description.resources.terms.end(),
                [&](const memory::Term& term) {
                    if (term.category != memory::Category::kernel_workspace ||
                        term.location != (facts.caps.kind == backend::DeviceKind::cpu ? memory::Location::host : memory::Location::device)) return false;
                    const auto declared = term.bytes.has_value() || term.bytes_per_context.has_value();
                    if (!declared || term.alignment < requirement.alignment) return false;
                    const auto raw = memory::checked_add(term.bytes.value_or(0),
                        memory::checked_multiply(term.bytes_per_context.value_or(0), context));
                    const auto padded = memory::round_up(raw, std::max(term.alignment, term.page_size));
                    return padded >= requirement.bytes;
                });
            if (!covered) throw std::invalid_argument("selected kernel workspace is not covered by known kernel_workspace metadata");
        }
        context_diagnostics.push_back("context=" + std::to_string(context) + " selected: statically eligible");
        auto identity = canonical_identity(description, options, facts, request.device_profile,
            request.pair_profile, prepared, candidate, request.model_id, request.backend_id,
            request.budgets, request.options.context_tokens != 0, ordered);
        candidate = candidate.with_profile_identity(identity);
        auto plan = std::shared_ptr<const ExecutionPlan>(new ExecutionPlan(std::move(description), options,
            facts, request.device_profile, request.pair_profile, std::move(prepared),
            std::move(candidate), std::move(identity), std::move(context_diagnostics)));
        auto grant = plan->candidate().grant(request.budgets);
        std::unique_ptr<IExecutionSession> session;
        std::unique_ptr<IModelRuntime> runtime;
        try {
            session = backend->instantiate(*plan, grant);
            if (!session) throw std::runtime_error("backend returned null execution session");
            runtime = model->create_runtime(plan->description(), *session);
            if (!runtime) throw std::runtime_error("model returned null runtime");
            // Warmup/capture failures are propagated. They are not capacity evidence; automatic
            // calibration retry is intentionally not implemented until a typed capacity result exists.
            require_success(session->warmup(grant), "warmup");
            grant.mark_warmed();
            if (request.graphs == memory::GraphMode::enabled) {
                require_success(session->capture(grant), "graph capture");
                grant.mark_graph_captured();
            }
            // Settle all submitted startup work before taking final high-water observations.
            session->drain();
            for (const auto& [id, bytes] : session->overhead_measurements(grant)) grant.observe_overhead(id, bytes);
            auto approval = std::move(grant).reconcile().approve();
            return ApprovedRuntime(std::move(model), std::move(backend), std::move(session),
                                   std::move(runtime), std::move(plan), std::move(approval));
        } catch (...) {
            if (session) session->drain();
            runtime.reset();
            session.reset();
            grant.abort();
            throw;
        }
    }
    throw std::runtime_error("no auto context candidate is statically feasible: " + infeasible);
}

ApprovedRuntime::ApprovedRuntime(std::shared_ptr<const IModelDefinition> model, std::shared_ptr<IGpuBackend> backend,
    std::unique_ptr<IExecutionSession> session, std::unique_ptr<IModelRuntime> runtime,
    std::shared_ptr<const ExecutionPlan> plan, memory::ReadinessApproval approval)
    : model_(std::move(model)), backend_(std::move(backend)), session_(std::move(session)),
      runtime_(std::move(runtime)), plan_(std::move(plan)), approval_(std::move(approval)) {}
ApprovedRuntime::ApprovedRuntime(ApprovedRuntime&& other) noexcept
    : model_(std::move(other.model_)), backend_(std::move(other.backend_)),
      session_(std::move(other.session_)), runtime_(std::move(other.runtime_)),
      plan_(std::move(other.plan_)), approval_(std::move(other.approval_)) {
    other.approval_.reset();
}
ApprovedRuntime& ApprovedRuntime::operator=(ApprovedRuntime&& other) noexcept {
    if (this != &other) {
        reset();
        model_ = std::move(other.model_);
        backend_ = std::move(other.backend_);
        session_ = std::move(other.session_);
        runtime_ = std::move(other.runtime_);
        plan_ = std::move(other.plan_);
        approval_ = std::move(other.approval_);
        other.approval_.reset();
    }
    return *this;
}
ApprovedRuntime::~ApprovedRuntime() noexcept { reset(); }
const ExecutionPlan& ApprovedRuntime::plan() const {
    if (!plan_) throw std::logic_error("approved runtime is empty");
    return *plan_;
}
const memory::ReadinessApproval& ApprovedRuntime::approval() const {
    if (!approval_) throw std::logic_error("approved runtime is empty");
    return *approval_;
}
void ApprovedRuntime::reset() noexcept {
    if (session_) session_->drain();
    runtime_.reset(); session_.reset(); approval_.reset(); plan_.reset(); model_.reset(); backend_.reset();
}
IModelRuntime& ApprovedRuntime::runtime() const {
    if (!runtime_) throw std::logic_error("approved runtime is empty");
    return *runtime_;
}

} // namespace saga::runtime
