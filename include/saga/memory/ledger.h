#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace saga::memory {

enum class Location : std::uint8_t { host, pinned, device };
enum class AllocationSource : std::uint8_t { project, runtime_overhead, graph_overhead };
enum class Lifetime : std::uint8_t { persistent, bringup, execution };
enum class GraphMode : std::uint8_t { disabled, enabled };

struct DomainBudget {
    std::uint64_t configured{0};
    std::uint64_t available{0};
    std::uint64_t headroom{0};
};
using Budgets = std::array<DomainBudget, 3>; // host, pinned, device; pinned is not device memory
using DomainBytes = std::array<std::uint64_t, 3>;

struct AllocationRecord {
    std::string id;
    std::string owner;
    Location location{Location::device};
    Lifetime lifetime{Lifetime::persistent};
    AllocationSource source{AllocationSource::project};
    bool enabled{true};
    std::uint64_t alignment{1};
    std::uint64_t page_size{1};
    std::optional<std::uint64_t> logical_bytes;
    // External/runtime and graph terms always need an explicit enforced bound, even if estimated.
    std::optional<std::uint64_t> overhead_bound;
    std::string sharing_identity; // empty = unique; nonempty = identical physical backing, not a hint
};

struct LedgerLine {
    AllocationRecord allocation;
    std::optional<std::uint64_t> padded_bytes;
    std::optional<std::uint64_t> bringup_cap;
};

class BringupGrant;
class StaticCandidate {
public:
    StaticCandidate(std::string profile_identity, GraphMode graphs, std::vector<AllocationRecord> records);
    StaticCandidate(const StaticCandidate&) = default;
    StaticCandidate& operator=(const StaticCandidate&) = default;
    StaticCandidate(StaticCandidate&&) noexcept;
    StaticCandidate& operator=(StaticCandidate&&) noexcept;
    const std::string& profile_identity() const noexcept { return profile_identity_; }
    GraphMode graphs() const noexcept { return graphs_; }
    const std::vector<LedgerLine>& unique_allocations() const noexcept { return lines_; }
    bool has_unresolved_terms() const noexcept;
    DomainBytes known_reserved_bytes() const; // reporting only; not a fit/serving approval
    BringupGrant grant(const Budgets&) const;

private:
    friend class BringupGrant;
    bool valid_{true};
    std::string profile_identity_;
    GraphMode graphs_;
    std::vector<LedgerLine> lines_;
    std::vector<std::pair<std::string, std::size_t>> aliases_;
};

namespace detail {
struct GrantState;
struct AllocationControl;
}

// Exclusively owns a typed unique_ptr, including its deleter. No shared-owner constructor or
// raw-owner accessor: consumers retain AllocationLease, never a separate backing owner.
class OwnedBacking {
public:
    OwnedBacking() noexcept = default;
    OwnedBacking(const OwnedBacking&) = delete;
    OwnedBacking& operator=(const OwnedBacking&) = delete;
    OwnedBacking(OwnedBacking&&) noexcept = default;
    OwnedBacking& operator=(OwnedBacking&&) noexcept = default;

    template<class T, class Deleter>
        requires (!std::is_reference_v<Deleter> && std::is_nothrow_move_constructible_v<Deleter>)
    explicit OwnedBacking(std::unique_ptr<T, Deleter> owner)
        : owner_(nullptr, &destroy<std::unique_ptr<T, Deleter>>) {
        // Allocate the erasure box before the noexcept ownership transfer. If allocation
        // throws, the parameter still owns and releases the backing with its original deleter.
        if (owner) owner_.reset(new std::unique_ptr<T, Deleter>(std::move(owner)));
    }

    explicit operator bool() const noexcept { return static_cast<bool>(owner_); }
    void reset() noexcept { owner_.reset(); }

private:
    template<class Owner> static void destroy(void* box) noexcept { delete static_cast<Owner*>(box); }
    std::unique_ptr<void, void (*)(void*) noexcept> owner_{nullptr, nullptr};
};

struct MemoryBacking {
    // The unique owner/deleter must release the actual storage, not merely a shared handle to it.
    // Deleters must not throw. Factory-local RAII must also unwind on exceptions.
    OwnedBacking owner;
    std::uint64_t actual_bytes{0};
};

// Independent lease objects may be copied/released on different threads, including final
// releases concurrent with coordinator operations. The same lease object needs external locking.
class AllocationLease {
public:
    AllocationLease() = default;
    std::uint64_t bytes() const noexcept;
private:
    friend class BringupGrant;
    explicit AllocationLease(std::shared_ptr<detail::AllocationControl> control);
    std::shared_ptr<detail::AllocationControl> control_;
};

struct AllocationMeasurement {
    std::string id;
    std::uint64_t cap{0};
    std::uint64_t live_bytes{0};
    std::uint64_t measured_peak{0};
    bool measured{false};
};
struct GrantReport {
    DomainBytes live_bytes{};
    DomainBytes measured_peaks{}; // sum of per-allocation peaks: conservative overlap accounting
    std::vector<AllocationMeasurement> allocations;
};

class ReadinessApproval;
class ReconciledResources {
public:
    ReconciledResources(const ReconciledResources&) = delete;
    ReconciledResources& operator=(const ReconciledResources&) = delete;
    ReconciledResources(ReconciledResources&&) noexcept = default;
    ReconciledResources& operator=(ReconciledResources&&) noexcept = default;
    ReadinessApproval approve() &&;
private:
    friend class BringupGrant;
    ReconciledResources(std::shared_ptr<detail::GrantState>,
                        std::vector<std::shared_ptr<detail::AllocationControl>>);
    std::shared_ptr<detail::GrantState> state_;
    std::vector<std::shared_ptr<detail::AllocationControl>> retained_;
};

// Proof is scoped to exactly one immutable profile; it retains persistent physical owners.
class ReadinessApproval {
public:
    ReadinessApproval(const ReadinessApproval&) = delete;
    ReadinessApproval& operator=(const ReadinessApproval&) = delete;
    ReadinessApproval(ReadinessApproval&&) noexcept = default;
    ReadinessApproval& operator=(ReadinessApproval&&) noexcept = default;
    const std::string& profile_identity() const;
    GraphMode graphs() const;
    GrantReport report() const;
private:
    friend class ReconciledResources;
    ReadinessApproval(std::shared_ptr<detail::GrantState>,
                      std::vector<std::shared_ptr<detail::AllocationControl>>);
    std::shared_ptr<detail::GrantState> state_;
    std::vector<std::shared_ptr<detail::AllocationControl>> retained_;
};

// Single bring-up coordinator, not a concurrent allocator. Coordinator/proof operations and
// moves must be externally serialized, but independent leases may release on other threads.
// Factories/deleters run without ledger locks. Same-thread reentrancy is supported, except
// warmup, capture and reconciliation require no active factory or backing-release transaction.
// Cross-thread deleters must not call coordinator operations without external serialization.
// There is no serving conversion.
class BringupGrant {
public:
    BringupGrant(const BringupGrant&) = delete;
    BringupGrant& operator=(const BringupGrant&) = delete;
    BringupGrant(BringupGrant&&) noexcept = default;
    BringupGrant& operator=(BringupGrant&&) noexcept = default;

    // Reserve before invoking factory. It must never allocate more than the supplied cap and
    // must report real padded bytes, not payload bytes. Exceptions permanently fail this grant;
    // previous leases remain accounted until their last copy unwinds. Aliases share the same cap.
    // Factory results transfer exclusive backing ownership; consumers retain only leases.
    // The original state is pinned across callbacks. On return, this grant must still reference
    // that state: moving it away/replacing it rejects the result and fails only the original
    // state (moving away and back is allowed). Do not destroy this grant during an operation.
    AllocationLease allocate(const std::string& id, std::uint64_t reservation,
                             const std::function<MemoryBacking(std::uint64_t cap)>& factory);
    // Called at monitored high-water observations, including zero. Over-bound observations latch
    // failure. Real backends must stop/unwind at this limit, not merely report it after serving.
    void observe_overhead(const std::string& id, std::uint64_t actual_bytes);
    void mark_warmed();
    void mark_graph_captured();
    void abort() noexcept;
    GrantReport report() const;
    ReconciledResources reconcile() &&;

private:
    friend class StaticCandidate;
    explicit BringupGrant(const StaticCandidate&, const Budgets&);
    std::shared_ptr<detail::GrantState> state_;
};

} // namespace saga::memory
