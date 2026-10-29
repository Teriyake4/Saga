#include "saga/memory/planner.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace saga::memory;
namespace kv = saga::kv;

namespace {
int failures = 0;
void check(bool condition, const std::string& message) {
  if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class F> void throws(F&& fn, const std::string& message) {
  try { fn(); check(false, message); } catch (const std::exception&) {}
}

Term make_term(std::string id, Category category, Location location = Location::device,
               std::optional<std::uint64_t> bytes = 64,
               std::optional<std::uint64_t> bound = std::nullopt) {
  return {std::move(id), "owner", category, location, Lifetime::persistent, bytes,
          std::nullopt, bound, 1, 1, {}};
}
CacheSpec main_cache() {
  return {kv::Geometry{16,4,256}, kv::Pair{kv::Codec::int8_g64,kv::Codec::int8_g64},
          kv::CacheLayout{1,1,0}, std::nullopt, std::nullopt, 1, 1};
}
CacheSpec draft_cache() {
  return {kv::Geometry{1,1,64}, kv::Pair{kv::Codec::fp16,kv::Codec::fp16},
          kv::CacheLayout{1,1,0}, 2, 4, 1, 1};
}
Profile complete_profile() {
  Profile p;
  p.identity = "planner-fixture";
  p.model_max_context = 8192;
  for (auto& b : p.budgets) b = {1ULL << 40,1ULL << 40,0};
  p.main_cache = main_cache();
  const std::vector<std::pair<Category,Location>> categories{
    {Category::text_weights,Location::device}, {Category::fp32_recurrence,Location::device},
    {Category::convolution,Location::device}, {Category::continuation,Location::device},
    {Category::provisional,Location::device}, {Category::rollback_records,Location::device},
    {Category::activations,Location::device}, {Category::kernel_workspace,Location::device},
    {Category::logits,Location::device}, {Category::distributions,Location::device},
    {Category::acceptance,Location::device}, {Category::host_mappings,Location::host},
    {Category::staging,Location::host}, {Category::request_buffers,Location::host},
    {Category::pinned_transfer,Location::pinned}, {Category::pinned_control,Location::pinned}
  };
  int n = 0;
  for (const auto& [category, location] : categories)
    p.terms.push_back(make_term("term-" + std::to_string(n++), category, location));
  p.terms.push_back(make_term("runtime",Category::runtime_overhead,Location::device,std::nullopt,8192));
  return p;
}

void sizing_and_selection_tests() {
  auto p = complete_profile();
  auto exact = evaluate(p,8192);
  check(exact.main_cache_bytes == 264ULL * 1024 * 1024, "8K INT8-G64 main cache fixture");
  check(exact.main_cache_capacity == 8192 && exact.draft_cache_capacity == 0,
        "main cache capacity reported separately");
  check(exact.unresolved.empty() && exact.preparation_eligible, "complete profile is preparation eligible");
  check(exact.known_bytes[static_cast<std::size_t>(Location::host)] > 0 &&
        exact.known_bytes[static_cast<std::size_t>(Location::pinned)] > 0 &&
        exact.known_bytes[static_cast<std::size_t>(Location::device)] > 0,
        "ordinary host, pinned, and device terms are budgeted independently");
  check(exact.candidate.has_value(), "evaluation owns its exact StaticCandidate");

  p.draft_enabled = true;
  p.main_cache->verification_tokens = 2;
  p.main_cache->mtp_alignment = 4;
  p.draft_cache = draft_cache();
  p.terms.push_back(make_term("draft-weights",Category::draft_weights));
  auto with_draft = evaluate(p,8192);
  check(with_draft.main_cache_reserve_tokens == 2 && with_draft.draft_cache_reserve_tokens == 2 &&
        with_draft.main_cache_capacity == 8196 && with_draft.draft_cache_capacity == 8196,
        "MTP reserve and four-token alignment reported separately on each independent cache");
  check(with_draft.draft_cache_bytes == 8196ULL * 256,
        "draft cache uses its inspected one-layer geometry, not main-layer geometry");

  auto request = ContextRequest{};
  request.auto_contexts = {4096,8192};
  auto selected = prepare(complete_profile(),request);
  check(selected.attempts.size() == 2 && selected.attempts[0].context == 8192 &&
        selected.attempts[1].context == 4096, "auto attempts are deterministic descending and complete");
  check(selected.selected && selected.selected->context == 8192, "auto chooses largest eligible context");

  auto explicit_zero = complete_profile();
  explicit_zero.terms.back().bytes = 0;
  explicit_zero.terms.back().overhead_bound = 0;
  check(evaluate(explicit_zero,1024).preparation_eligible,
        "explicit known-zero runtime estimate and bound are retained as evidence");

  auto too_small = complete_profile();
  too_small.budgets[static_cast<std::size_t>(Location::device)].configured = 1;
  throws([&] { (void)prepare(too_small,ContextRequest{8192,{},256}); },
         "explicit infeasible context throws instead of shrinking");
  request.auto_contexts = {8192,4096};
  auto rejected = prepare(too_small,request);
  check(!rejected.selected && rejected.attempts.size() == 2 &&
        !rejected.attempts[0].budget_fits && !rejected.attempts[1].budget_fits,
        "auto returns every rejected attempt with budget diagnostics");
  request.auto_contexts = {8192,4096,2048}; request.max_candidates = 2;
  throws([&] { (void)prepare(complete_profile(),request); }, "auto candidate bound enforced");
}

void enabled_component_tests() {
  auto p = complete_profile();
  p.draft_enabled = true;
  p.vision_enabled = true;
  p.main_cache->verification_tokens = 3;
  p.main_cache->mtp_alignment = 8;
  p.draft_cache = draft_cache();
  p.terms.push_back(make_term("draft-weights-enabled",Category::draft_weights,Location::device,1024));
  p.terms.push_back(make_term("vision-weights",Category::vision_weights,Location::device,2048));
  p.terms.push_back(make_term("media-weights",Category::media_weights,Location::device,512));
  p.terms.push_back(make_term("media-workspace",Category::media_workspace,Location::host,4096));
  p.terms.push_back(make_term("media-features",Category::media_features,Location::device,1024));
  auto evaluation = evaluate(p,4096);
  check(evaluation.preparation_eligible && evaluation.draft_cache_bytes > 0,
        "complete enabled MTP and vision metadata is planned");
  check(evaluation.main_cache_capacity == 4104 && evaluation.draft_cache_capacity == 4100,
        "MTP verification reserve/alignment is explicit per cache");
  check(evaluation.known_bytes[static_cast<std::size_t>(Location::host)] >= 4096,
        "CPU media workspace contributes to host memory");

  auto incomplete = p;
  incomplete.terms.erase(std::remove_if(incomplete.terms.begin(),incomplete.terms.end(),[](const Term& t) {
    return t.category == Category::media_features;
  }),incomplete.terms.end());
  auto rejected = evaluate(incomplete,4096);
  check(!rejected.preparation_eligible && rejected.candidate->has_unresolved_terms(),
        "enabled vision category omission creates an unresolved candidate allocation");

  auto page_rounded = complete_profile();
  page_rounded.main_cache->allocation_alignment = 256;
  page_rounded.main_cache->page_size = 4096;
  auto paged = evaluate(page_rounded,1024);
  check(paged.main_cache_bytes % 4096 == 0 && paged.main_cache_bytes >= 1024ULL * 4096,
        "cache allocation reports physical page rounding");
}

void missing_and_unknown_tests() {
  auto base = complete_profile();
  const std::vector<Category> mandatory{
    Category::text_weights,Category::fp32_recurrence,Category::convolution,Category::continuation,
    Category::provisional,Category::rollback_records,Category::activations,Category::kernel_workspace,
    Category::logits,Category::distributions,Category::acceptance,Category::host_mappings,
    Category::staging,Category::request_buffers,Category::pinned_transfer,Category::pinned_control,
    Category::runtime_overhead
  };
  for (const auto missing : mandatory) {
    auto p = base;
    p.terms.erase(std::remove_if(p.terms.begin(),p.terms.end(),[missing](const Term& t) {
      return t.category == missing;
    }),p.terms.end());
    auto e = evaluate(p,1024);
    check(!e.preparation_eligible, "omitted mandatory category blocks: " + std::to_string(static_cast<int>(missing)));
    check(std::any_of(e.allocations.begin(),e.allocations.end(),[&](const AllocationRecord& a) {
      if (missing == Category::runtime_overhead) return a.source == AllocationSource::runtime_overhead && !a.overhead_bound;
      return a.source == AllocationSource::project && !a.logical_bytes && a.id.find("unresolved-") == 0;
    }), "omitted mandatory category materialized as unknown candidate record");
  }
  for (const auto unknown : mandatory) {
    if (unknown == Category::runtime_overhead) continue;
    auto p = base;
    auto it = std::find_if(p.terms.begin(),p.terms.end(),[unknown](const Term& t) { return t.category == unknown; });
    if (it == p.terms.end()) continue;
    it->bytes.reset(); it->bytes_per_context.reset();
    auto e = evaluate(p,1024);
    check(!e.preparation_eligible, "unknown mandatory bytes block: " + std::to_string(static_cast<int>(unknown)));
    check(e.candidate && e.candidate->has_unresolved_terms(), "unknown allocation remains on the owned candidate");
  }
  auto optional_metadata = complete_profile();
  optional_metadata.main_cache->geometry.reset();
  auto malformed = evaluate(optional_metadata,1024);
  check(!malformed.preparation_eligible && malformed.candidate->has_unresolved_terms(),
        "incomplete cache metadata is represented unresolved");
  optional_metadata.terms.push_back(make_term("manual-main-cache",Category::main_kv));
  throws([&] { (void)evaluate(optional_metadata,1024); }, "manual cache category marker rejected");

  auto bounded_project = complete_profile();
  bounded_project.terms.front().overhead_bound = 4096;
  throws([&] { (void)evaluate(bounded_project,1024); },
         "known project allocation cannot silently discard overhead bound");
  bounded_project.terms.front().bytes.reset();
  throws([&] { (void)evaluate(bounded_project,1024); },
         "unknown project allocation cannot use overhead bound as substitute");
}

void budgets_sharing_overflow_tests() {
  auto p = complete_profile();
  p.budgets[static_cast<std::size_t>(Location::host)] = {100,100,50};
  auto host_limited = evaluate(p,1024);
  check(!host_limited.budget_fits, "host headroom participates in domain budget");
  p.budgets = {};
  for (auto& b : p.budgets) b = {1ULL << 40,1ULL << 40,0};
  p.budgets[static_cast<std::size_t>(Location::pinned)].available = 1;
  check(!evaluate(p,1024).budget_fits, "pinned transfer budget is separate from device budget");

  auto shared = complete_profile();
  auto a = make_term("weight-a",Category::text_weights,Location::device,4096);
  auto b = make_term("weight-b",Category::draft_weights,Location::device,4096);
  a.physical_identity = "exact-backing"; b.physical_identity = "exact-backing";
  shared.terms.push_back(a); shared.terms.push_back(b);
  const auto once = evaluate(shared,1024).known_bytes[static_cast<std::size_t>(Location::device)];
  shared.terms.back().bytes = 4097;
  throws([&] { (void)evaluate(shared,1024); }, "conflicting physical alias metadata rejected");
  shared.terms.back().bytes = 4096;
  shared.terms.back().physical_identity.clear();
  const auto twice = evaluate(shared,1024).known_bytes[static_cast<std::size_t>(Location::device)];
  check(twice > once, "unidentified backing is conservatively counted separately");

  auto overflow = complete_profile();
  auto& scaled = overflow.terms.front();
  scaled.bytes.reset(); scaled.bytes_per_context = UINT64_MAX;
  throws([&] { (void)evaluate(overflow,2); }, "context-scaled allocation overflow is checked");
  auto cache_overflow = complete_profile();
  cache_overflow.main_cache->geometry = kv::Geometry{UINT64_MAX,UINT64_MAX,UINT64_MAX};
  throws([&] { (void)evaluate(cache_overflow,1024); }, "KV geometry overflow is checked");
  auto bad_alignment = complete_profile();
  bad_alignment.terms.front().alignment = 3;
  throws([&] { (void)evaluate(bad_alignment,1024); }, "invalid term alignment rejected");
  auto bad_mtp = complete_profile();
  bad_mtp.draft_enabled = true; bad_mtp.draft_cache = draft_cache();
  bad_mtp.main_cache->verification_tokens.reset();
  throws([&] { (void)evaluate(bad_mtp,1024); }, "MTP reserve cannot default to zero");
}

MemoryBacking backing(std::uint64_t bytes) {
  auto storage = std::make_unique<std::vector<std::byte>>(static_cast<std::size_t>(bytes));
  return {OwnedBacking(std::move(storage)),bytes};
}

void graph_and_grant_tests() {
  auto p = complete_profile();
  p.graphs = GraphMode::enabled;
  p.terms.push_back(make_term("graph-bound",Category::graph_overhead,Location::device,std::nullopt,4096));
  auto e = evaluate(p,1024);
  check(e.preparation_eligible && !e.unresolved.empty(),
        "bounded graph candidate permits bring-up but remains unresolved for readiness");
  auto bounded_graph_grant = e.candidate->grant(p.budgets);
  bounded_graph_grant.abort();
  p = complete_profile();
  e = evaluate(p,1024);
  auto grant = e.candidate->grant(p.budgets);
  std::vector<AllocationLease> leases;
  for (const auto& allocation : e.candidate->unique_allocations()) {
    if (allocation.allocation.source == AllocationSource::project) {
      const auto bytes = allocation.padded_bytes.value_or(0);
      leases.push_back(grant.allocate(allocation.allocation.id,bytes,[bytes](std::uint64_t) { return backing(bytes); }));
    }
  }
  grant.mark_warmed();
  for (const auto& allocation : e.candidate->unique_allocations())
    if (allocation.allocation.source != AllocationSource::project)
      grant.observe_overhead(allocation.allocation.id,0);
  auto reconciled = std::move(grant).reconcile();
  auto approval = std::move(reconciled).approve();
  check(approval.profile_identity() == e.candidate->profile_identity(),
        "grant/reconcile uses the exact planner candidate identity");

  auto unknown_bound = complete_profile();
  unknown_bound.terms.back().overhead_bound.reset();
  auto unbounded = evaluate(unknown_bound,1024);
  check(!unbounded.preparation_eligible, "missing runtime overhead bound blocks preparation");

  auto exhausted = complete_profile();
  auto exhausted_eval = evaluate(exhausted,1024);
  auto exhausted_grant = exhausted_eval.candidate->grant(exhausted.budgets);
  throws([&] { exhausted_grant.observe_overhead("runtime",8193); },
         "runtime overhead exceeding its explicit cap fails the grant");
  exhausted_grant.abort();
}
}

int main() {
  sizing_and_selection_tests();
  enabled_component_tests();
  missing_and_unknown_tests();
  budgets_sharing_overflow_tests();
  graph_and_grant_tests();
  if (failures != 0) std::cerr << failures << " resource planner assertion(s) failed\n";
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
