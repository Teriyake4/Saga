#include "saga/memory/planner.h"
#include "saga/memory/checked.h"

#include <algorithm>

#include <stdexcept>


namespace saga::memory {
namespace {

bool valid_location(Location value) {
  return value == Location::host || value == Location::pinned || value == Location::device;
}
bool valid_lifetime(Lifetime value) {
  return value == Lifetime::persistent || value == Lifetime::bringup || value == Lifetime::execution;
}
bool valid_graph_mode(GraphMode value) { return value == GraphMode::disabled || value == GraphMode::enabled; }

bool contains(const std::vector<Term>& terms, Category category) {
  return std::any_of(terms.begin(), terms.end(), [category](const Term& term) {
    return term.category == category;
  });
}

Location placeholder_location(Category category) {
  switch (category) {
    case Category::host_mappings: case Category::staging: case Category::request_buffers:
    case Category::media_workspace: return Location::host;
    case Category::pinned_transfer: case Category::pinned_control: return Location::pinned;
    default: return Location::device;
  }
}

std::string category_name(Category category) {
  switch (category) {
    case Category::text_weights: return "text_weights";
    case Category::draft_weights: return "draft_weights";
    case Category::vision_weights: return "vision_weights";
    case Category::main_kv: return "main_kv";
    case Category::draft_kv: return "draft_kv";
    case Category::fp32_recurrence: return "fp32_recurrence";
    case Category::convolution: return "convolution";
    case Category::continuation: return "continuation";
    case Category::provisional: return "provisional";
    case Category::rollback_records: return "rollback_records";
    case Category::activations: return "activations";
    case Category::kernel_workspace: return "kernel_workspace";
    case Category::logits: return "logits";
    case Category::distributions: return "distributions";
    case Category::acceptance: return "acceptance";
    case Category::media_weights: return "media_weights";
    case Category::media_workspace: return "media_workspace";
    case Category::media_features: return "media_features";
    case Category::host_mappings: return "host_mappings";
    case Category::staging: return "staging";
    case Category::request_buffers: return "request_buffers";
    case Category::pinned_transfer: return "pinned_transfer";
    case Category::pinned_control: return "pinned_control";
    case Category::runtime_overhead: return "runtime_overhead";
    case Category::graph_overhead: return "graph_overhead";
  }
  throw std::invalid_argument("invalid resource category");
}

std::uint64_t term_bytes(const Term& term, std::uint64_t context) {
  const auto fixed = term.bytes.value_or(0);
  const auto variable = term.bytes_per_context
      ? checked_multiply(*term.bytes_per_context, context) : 0;
  return checked_add(fixed, variable);
}

std::string candidate_identity(const Profile& profile, std::uint64_t context) {
  return profile.identity + ":context=" + std::to_string(context);
}

void validate_cache(const CacheSpec& spec, bool mtp_enabled) {
  if (!is_power_of_two(spec.allocation_alignment) || !is_power_of_two(spec.page_size))
    throw std::invalid_argument("cache allocation alignment/page size must be powers of two");
  if (mtp_enabled && (!spec.verification_tokens || !spec.mtp_alignment))
    throw std::invalid_argument("MTP cache verification reserve and alignment must be explicit");
  if (spec.mtp_alignment && !is_power_of_two(*spec.mtp_alignment))
    throw std::invalid_argument("MTP alignment must be a power of two");
  if (spec.codecs) {
    (void)kv::descriptor(spec.codecs->key);
    (void)kv::descriptor(spec.codecs->value);
  }
  if (spec.layout) {
    if (spec.layout->tokens_per_page == 0 || !is_power_of_two(spec.layout->plane_alignment))
      throw std::invalid_argument("invalid KV page layout");
  }
  if (spec.geometry && (spec.geometry->layers == 0 || spec.geometry->heads == 0 || spec.geometry->head_channels == 0))
    throw std::invalid_argument("KV geometry fields must be positive");
}

} // namespace

Evaluation evaluate(const Profile& profile, std::uint64_t context) {
  if (profile.identity.empty() || profile.model_max_context == 0)
    throw std::invalid_argument("profile identity and model maximum are required");
  if (context == 0 || context > profile.model_max_context)
    throw std::invalid_argument("context outside model limits");
  if (!valid_graph_mode(profile.graphs)) throw std::invalid_argument("invalid graph mode");


  std::vector<std::string> ids;
  for (const auto& term : profile.terms) {
    if (term.id.empty() || term.owner.empty()) throw std::invalid_argument("allocation term needs ID and owner");
    if (std::find(ids.begin(), ids.end(), term.id) != ids.end())
      throw std::invalid_argument("duplicate allocation term ID: " + term.id);
    ids.push_back(term.id);
    if (!valid_location(term.location) || !valid_lifetime(term.lifetime))
      throw std::invalid_argument("invalid allocation location or lifetime: " + term.id);
    if (!is_power_of_two(term.alignment) || !is_power_of_two(term.page_size))
      throw std::invalid_argument("invalid allocation alignment/page size: " + term.id);
    (void)category_name(term.category);
    if (term.category == Category::main_kv || term.category == Category::draft_kv)
      throw std::invalid_argument("cache categories are specified only by CacheSpec: " + term.id);
    if (term.category == Category::graph_overhead && profile.graphs == GraphMode::disabled)
      throw std::invalid_argument("graph overhead supplied for graph-disabled profile: " + term.id);
    const bool overhead_category = term.category == Category::runtime_overhead ||
                                  term.category == Category::graph_overhead;
    if (!overhead_category && term.overhead_bound)
      throw std::invalid_argument("project allocation cannot specify an overhead bound: " + term.id);
  }
  if (profile.main_cache) validate_cache(*profile.main_cache, profile.draft_enabled);
  if (profile.draft_cache) validate_cache(*profile.draft_cache, profile.draft_enabled);

  Evaluation result;
  result.context = context;
  std::vector<AllocationRecord> records;
  auto add_cache = [&](const CacheSpec& spec, std::uint64_t capacity, bool is_draft) {
    if (!spec.geometry || !spec.codecs || !spec.layout) {
      const auto id = std::string(is_draft ? "draft" : "main") + "-kv";
      result.unresolved.push_back(id + " cache metadata is incomplete");
      records.push_back({"unresolved-" + id, "unresolved:" + id, Location::device, Lifetime::persistent,
        AllocationSource::project, true, 1, 1, std::nullopt, std::nullopt, {}});
      return;
    }
    const auto reserve = profile.draft_enabled ? *spec.verification_tokens : spec.verification_tokens.value_or(0);
    const auto unrounded = checked_add(capacity, reserve);
    const auto rounded_capacity = spec.mtp_alignment
        ? round_up(unrounded, *spec.mtp_alignment) : unrounded;
    const auto cache = kv::main_cache_bytes(*spec.geometry, *spec.codecs, rounded_capacity, *spec.layout);
    const auto bytes = round_up(cache.total_bytes, std::max(spec.allocation_alignment, spec.page_size));
    records.push_back({is_draft ? "draft-kv" : "main-kv", is_draft ? "draft-kv" : "main-kv",
      Location::device, Lifetime::persistent, AllocationSource::project, true,
      spec.allocation_alignment, spec.page_size, bytes, std::nullopt, {}});
    if (is_draft) {
      result.draft_cache_reserve_tokens = reserve;
      result.draft_cache_capacity = rounded_capacity;
      result.draft_cache_bytes = bytes;
    } else {
      result.main_cache_reserve_tokens = reserve;
      result.main_cache_capacity = rounded_capacity;
      result.main_cache_bytes = bytes;
    }
  };
  if (profile.main_cache) add_cache(*profile.main_cache, context, false);
  else {
    result.unresolved.push_back("main cache specification omitted");
    records.push_back({"unresolved-main-kv", "unresolved:main-kv", Location::device, Lifetime::persistent,
      AllocationSource::project, true, 1, 1, std::nullopt, std::nullopt, {}});
  }
  if (profile.draft_enabled) {
    if (profile.draft_cache) add_cache(*profile.draft_cache, context, true);
    else {
      result.unresolved.push_back("draft cache specification omitted");
      records.push_back({"unresolved-draft-kv", "unresolved:draft-kv", Location::device, Lifetime::persistent,
        AllocationSource::project, true, 1, 1, std::nullopt, std::nullopt, {}});
    }
  } else if (profile.draft_cache) {
    throw std::invalid_argument("draft cache supplied while draft profile is disabled");
  }

  std::vector<Category> required{
    Category::text_weights, Category::fp32_recurrence, Category::convolution, Category::continuation,
    Category::provisional, Category::rollback_records, Category::activations, Category::kernel_workspace,
    Category::logits, Category::distributions, Category::acceptance, Category::host_mappings,
    Category::staging, Category::request_buffers, Category::pinned_transfer, Category::pinned_control,
    Category::main_kv, Category::runtime_overhead
  };
  if (profile.draft_enabled) required.insert(required.end(), {Category::draft_weights, Category::draft_kv});
  if (profile.vision_enabled) {
    required.insert(required.end(), {Category::vision_weights, Category::media_weights,
      Category::media_workspace, Category::media_features});
  }
  if (profile.graphs == GraphMode::enabled) required.push_back(Category::graph_overhead);
  for (const auto category : required) {
    const bool supplied = category == Category::main_kv ? profile.main_cache.has_value() :
      (category == Category::draft_kv ? profile.draft_cache.has_value() : contains(profile.terms, category));
    if (!supplied) {
      const auto name = category_name(category);
      result.unresolved.push_back("required category omitted: " + name);
      if (category != Category::runtime_overhead && category != Category::graph_overhead)
        records.push_back({"unresolved-" + name, "unresolved:" + name, placeholder_location(category),
          Lifetime::persistent, AllocationSource::project, true, 1, 1, std::nullopt, std::nullopt, {}});
    }
  }

  for (const auto& term : profile.terms) {
    const bool runtime = term.category == Category::runtime_overhead;
    const bool graph = term.category == Category::graph_overhead;
    const auto known = term.bytes.has_value() || term.bytes_per_context.has_value();
    const auto logical = known ? std::optional<std::uint64_t>(term_bytes(term, context)) : std::nullopt;
    const auto bound = runtime || graph ? term.overhead_bound : std::nullopt;
    if (!known && !runtime && !graph) result.unresolved.push_back("unknown project bytes: " + term.id);
    if ((runtime || graph) && !bound) result.unresolved.push_back("unbounded overhead: " + term.id);
    if (graph) result.unresolved.push_back("graph overhead requires capture calibration: " + term.id);
    const auto source = runtime ? AllocationSource::runtime_overhead :
      (graph ? AllocationSource::graph_overhead : AllocationSource::project);
    records.push_back({term.id, term.owner, term.location, term.lifetime, source, true,
      term.alignment, term.page_size, logical, bound, term.physical_identity});
  }
  if (!contains(profile.terms, Category::runtime_overhead))
    records.push_back({"runtime-unresolved", "runtime", Location::device, Lifetime::bringup,
      AllocationSource::runtime_overhead, true, 1, 1, std::nullopt, std::nullopt, {}});
  if (profile.graphs == GraphMode::enabled && !contains(profile.terms, Category::graph_overhead))
    records.push_back({"graph-unresolved", "graph", Location::device, Lifetime::bringup,
      AllocationSource::graph_overhead, true, 1, 1, std::nullopt, std::nullopt, {}});

  StaticCandidate candidate(candidate_identity(profile, context), profile.graphs, std::move(records));
  result.known_bytes = candidate.known_reserved_bytes();
  for (const auto& line : candidate.unique_allocations()) {
    if (line.allocation.source == AllocationSource::project && !line.padded_bytes)
      result.unresolved.push_back("unknown project allocation: " + line.allocation.id);
    if (line.allocation.source != AllocationSource::project && !line.bringup_cap)
      result.unresolved.push_back("unbounded overhead allocation: " + line.allocation.id);
  }
  result.allocations.reserve(candidate.unique_allocations().size());
  for (const auto& line : candidate.unique_allocations()) result.allocations.push_back(line.allocation);
  result.budget_fits = true;
  for (std::size_t i = 0; i < profile.budgets.size(); ++i) {
    const auto& budget = profile.budgets[i];
    const auto limit = std::min(budget.configured, budget.available);
    if (budget.configured == 0 || budget.available == 0 || budget.headroom > limit ||
        result.known_bytes[i] > limit - budget.headroom) {
      result.budget_fits = false;
      result.unresolved.push_back("budget/availability/headroom exceeded in domain " + std::to_string(i));
    }
  }
  result.candidate.emplace(std::move(candidate));
  if (result.budget_fits) {
    try {
      (void)result.candidate->grant(profile.budgets);
      result.preparation_eligible = true;
    } catch (const std::runtime_error& error) {
      result.unresolved.push_back(std::string("ledger rejects preparation: ") + error.what());
    }
  }
  return result;
}

Selection prepare(const Profile& profile, const ContextRequest& request) {
  if (request.max_candidates == 0 || request.max_candidates > 256)
    throw std::invalid_argument("candidate bound must be 1..256");
  Selection selection;
  if (request.explicit_context) {
    auto attempt = evaluate(profile, *request.explicit_context);
    if (!attempt.preparation_eligible)
      throw std::runtime_error("explicit context is infeasible: " +
        (attempt.unresolved.empty() ? std::string("budget does not fit") : attempt.unresolved.front()));
    selection.selected = attempt;
    selection.attempts.push_back(std::move(attempt));
    return selection;
  }
  if (request.auto_contexts.empty()) throw std::invalid_argument("auto context candidates must be explicit");
  if (request.auto_contexts.size() > request.max_candidates)
    throw std::invalid_argument("auto context candidate bound exceeded");
  auto contexts = request.auto_contexts;
  std::sort(contexts.begin(), contexts.end(), std::greater<>());
  if (std::adjacent_find(contexts.begin(), contexts.end()) != contexts.end())
    throw std::invalid_argument("auto context candidates must be unique");
  for (const auto context : contexts) {
    auto attempt = evaluate(profile, context);
    if (!selection.selected && attempt.preparation_eligible) selection.selected = attempt;
    selection.attempts.push_back(std::move(attempt));
  }
  return selection;
}

} // namespace saga::memory
