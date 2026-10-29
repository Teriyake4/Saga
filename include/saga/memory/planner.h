#pragma once

#include "saga/kv/catalog.h"
#include "saga/memory/ledger.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace saga::memory {

enum class Category : std::uint8_t {
  text_weights, draft_weights, vision_weights, main_kv, draft_kv,
  fp32_recurrence, convolution, continuation, provisional, rollback_records,
  activations, kernel_workspace, logits, distributions, acceptance,
  media_weights, media_workspace, media_features, host_mappings, staging,
  request_buffers, pinned_transfer, pinned_control, runtime_overhead, graph_overhead
};

struct Term {
  std::string id;
  std::string owner;
  Category category{Category::text_weights};
  Location location{Location::device};
  Lifetime lifetime{Lifetime::persistent};
  std::optional<std::uint64_t> bytes;
  std::optional<std::uint64_t> bytes_per_context;
  std::optional<std::uint64_t> overhead_bound;
  std::uint64_t alignment{1};
  std::uint64_t page_size{1};
  std::string physical_identity;
};

struct CacheSpec {
  std::optional<kv::Geometry> geometry;
  std::optional<kv::Pair> codecs;
  std::optional<kv::CacheLayout> layout;
  std::optional<std::uint64_t> verification_tokens;
  std::optional<std::uint64_t> mtp_alignment;
  std::uint64_t allocation_alignment{1};
  std::uint64_t page_size{1};
};

struct Profile {
  std::string identity;
  std::uint64_t model_max_context{0};
  bool draft_enabled{false};
  bool vision_enabled{false};
  GraphMode graphs{GraphMode::disabled};
  std::optional<CacheSpec> main_cache;
  std::optional<CacheSpec> draft_cache;
  std::vector<Term> terms;
  Budgets budgets{};
};

struct Evaluation {
  std::uint64_t context{0};
  std::uint64_t main_cache_reserve_tokens{0};
  std::uint64_t draft_cache_reserve_tokens{0};
  std::uint64_t main_cache_capacity{0};
  std::uint64_t draft_cache_capacity{0};
  std::uint64_t main_cache_bytes{0};
  std::uint64_t draft_cache_bytes{0};
  DomainBytes known_bytes{};
  std::vector<std::string> unresolved;
  std::vector<AllocationRecord> allocations;
  std::optional<StaticCandidate> candidate;
  bool budget_fits{false};
  bool preparation_eligible{false};
};

struct ContextRequest {
  std::optional<std::uint64_t> explicit_context;
  std::vector<std::uint64_t> auto_contexts;
  std::uint32_t max_candidates{256};
};

struct Selection {
  std::optional<Evaluation> selected;
  std::vector<Evaluation> attempts;
};

Evaluation evaluate(const Profile& profile, std::uint64_t user_context);
Selection prepare(const Profile& profile, const ContextRequest& request);

} // namespace saga::memory
