#include "saga/memory/ledger.h"
#include "saga/memory/checked.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace saga::memory {
namespace {
std::size_t domain(Location location) {
    const auto index = static_cast<std::size_t>(location);
    if (index >= 3) throw std::invalid_argument("invalid allocation location");
    return index;
}
std::uint64_t padded(const AllocationRecord& record, std::uint64_t bytes) {
    return round_up(bytes, std::max(record.alignment, record.page_size));
}
bool same_backing(const AllocationRecord& a, const AllocationRecord& b) {
    return a.location == b.location && a.lifetime == b.lifetime && a.source == b.source &&
           a.alignment == b.alignment && a.page_size == b.page_size &&
           a.logical_bytes == b.logical_bytes && a.overhead_bound == b.overhead_bound;
}
} // namespace

StaticCandidate::StaticCandidate(std::string profile_identity, GraphMode graphs,
                                 std::vector<AllocationRecord> records)
    : profile_identity_(std::move(profile_identity)), graphs_(graphs) {
    if (profile_identity_.empty()) throw std::invalid_argument("profile identity is required");
    if (graphs != GraphMode::disabled && graphs != GraphMode::enabled)
        throw std::invalid_argument("invalid graph mode");
    bool runtime_term = false, graph_term = false;
    std::vector<std::string> ids;
    for (auto& record : records) {
        if (record.id.empty() || record.owner.empty()) throw std::invalid_argument("allocation needs ID and owner");
        if (std::find(ids.begin(), ids.end(), record.id) != ids.end())
            throw std::invalid_argument("duplicate allocation ID: " + record.id);
        ids.push_back(record.id);
        domain(record.location);
        if (!is_power_of_two(record.alignment) || !is_power_of_two(record.page_size))
            throw std::invalid_argument("invalid allocation alignment/page size: " + record.id);
        if (record.lifetime != Lifetime::persistent && record.lifetime != Lifetime::bringup &&
            record.lifetime != Lifetime::execution) throw std::invalid_argument("invalid lifetime");
        if (record.source != AllocationSource::project && record.source != AllocationSource::runtime_overhead &&
            record.source != AllocationSource::graph_overhead) throw std::invalid_argument("invalid allocation source");
        if (!record.enabled) continue;
        if (record.source == AllocationSource::graph_overhead) {
            if (graphs == GraphMode::disabled)
                throw std::invalid_argument("graph-disabled profile contains enabled graph allocation: " + record.id);
            graph_term = true;
        }
        runtime_term |= record.source == AllocationSource::runtime_overhead;
        if (record.source == AllocationSource::project && record.overhead_bound)
            throw std::invalid_argument("project allocation cannot use an overhead allowance: " + record.id);
        std::size_t index = lines_.size();
        if (!record.sharing_identity.empty()) {
            for (std::size_t i = 0; i < lines_.size(); ++i) {
                if (lines_[i].allocation.sharing_identity == record.sharing_identity) {
                    if (!same_backing(lines_[i].allocation, record))
                        throw std::invalid_argument("inconsistent physical sharing identity: " + record.sharing_identity);
                    index = i;
                    break;
                }
            }
        }
        aliases_.emplace_back(record.id, index);
        if (index != lines_.size()) continue;
        const auto logical = record.logical_bytes;
        const auto cap = record.source == AllocationSource::project ? logical : record.overhead_bound;
        LedgerLine line{std::move(record), std::nullopt, std::nullopt};
        if (logical) line.padded_bytes = padded(line.allocation, *logical);
        if (cap) line.bringup_cap = padded(line.allocation, *cap);
        if (line.padded_bytes && line.bringup_cap && *line.padded_bytes > *line.bringup_cap)
            throw std::invalid_argument("allocation estimate exceeds bring-up cap: " + line.allocation.id);
        lines_.push_back(std::move(line));
    }
    if (!runtime_term) throw std::invalid_argument("profile must explicitly reserve runtime overhead (including zero)");
    if (graphs == GraphMode::enabled && !graph_term)
        throw std::invalid_argument("graph-enabled profile needs an explicit graph overhead term");
}

StaticCandidate::StaticCandidate(StaticCandidate&& other) noexcept
    : valid_(std::exchange(other.valid_, false)), profile_identity_(std::move(other.profile_identity_)),
      graphs_(other.graphs_), lines_(std::move(other.lines_)), aliases_(std::move(other.aliases_)) {}

StaticCandidate& StaticCandidate::operator=(StaticCandidate&& other) noexcept {
    if (this != &other) {
        valid_ = std::exchange(other.valid_, false);
        profile_identity_ = std::move(other.profile_identity_);
        graphs_ = other.graphs_;
        lines_ = std::move(other.lines_);
        aliases_ = std::move(other.aliases_);
    }
    return *this;
}

StaticCandidate StaticCandidate::with_profile_identity(std::string identity) const {
    if (!valid_) throw std::logic_error("cannot reidentify a moved-from static candidate");
    if (identity.empty()) throw std::invalid_argument("profile identity is required");
    auto copy = *this;
    copy.profile_identity_ = std::move(identity);
    return copy;
}

bool StaticCandidate::has_unresolved_terms() const noexcept {
    for (const auto& line : lines_)
        if (!line.padded_bytes || line.allocation.source != AllocationSource::project) return true;
    return false;
}

DomainBytes StaticCandidate::known_reserved_bytes() const {
    DomainBytes totals{};
    for (const auto& line : lines_) if (line.bringup_cap) {
        auto& total = totals[domain(line.allocation.location)];
        total = checked_add(total, *line.bringup_cap);
    }
    return totals;
}

namespace detail {
struct Observation {
    std::uint64_t live{0};
    std::uint64_t pending{0};
    std::uint64_t peak{0};
    bool measured{false};
    // Entries exist only until their own release transaction completes, not for grant history.
    std::map<const AllocationControl*, std::weak_ptr<AllocationControl>> owners;
};
struct GrantState {
    mutable std::mutex mutex;
    StaticCandidate candidate;
    Budgets budgets;
    std::vector<Observation> observations;
    std::size_t active_factories{0}; // counts transactions even when their reservation is zero
    std::size_t active_releases{0};
    bool failed{false};
    bool warmed{false};
    bool captured{false};
    bool reconciled{false};
    bool approved{false};
    GrantState(const StaticCandidate& plan, const Budgets& limits)
        : candidate(plan), budgets(limits), observations(plan.unique_allocations().size()) {}
};
struct AllocationControl {
    std::shared_ptr<GrantState> state;
    std::size_t index;
    std::uint64_t bytes{0};
    OwnedBacking owner;
    bool registered{false};
    AllocationControl(std::shared_ptr<GrantState> grant, std::size_t allocation_index)
        : state(std::move(grant)), index(allocation_index) {}
    ~AllocationControl() {
        if (!registered) return;
        {
            std::lock_guard lock(state->mutex);
            ++state->active_releases;
        }
        // Keep bytes charged, and the registry entry present, until the user deleter returns.
        // Its weak_ptr is already expired here; reconciliation must never treat that as proof.
        owner.reset();
        {
            std::lock_guard lock(state->mutex);
            auto& observation = state->observations[index];
            observation.live -= bytes;
            observation.owners.erase(this);
            --state->active_releases;
        }
    }
};
} // namespace detail

namespace {
detail::GrantState& require_state(const std::shared_ptr<detail::GrantState>& state) {
    if (!state) throw std::logic_error("moved-from resource state");
    return *state;
}
void require_open(detail::GrantState& state) {
    if (state.failed) throw std::logic_error("bring-up grant failed; unwind resources and replan");
    if (state.reconciled) throw std::logic_error("bring-up grant already reconciled");
}
void require_quiescent(const detail::GrantState& state) {
    if (state.active_factories != 0 || state.active_releases != 0)
        throw std::logic_error("active allocation/release transaction blocks readiness transition");
}
// Caller holds state.mutex.
GrantReport report_state(const detail::GrantState& state) {
    GrantReport result;
    const auto& lines = state.candidate.unique_allocations();
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto& observation = state.observations[i];
        const auto& line = lines[i];
        const auto index = domain(line.allocation.location);
        result.live_bytes[index] = checked_add(result.live_bytes[index], observation.live);
        result.measured_peaks[index] = checked_add(result.measured_peaks[index], observation.peak);
        result.allocations.push_back({line.allocation.id, *line.bringup_cap, observation.live,
                                      observation.peak, observation.measured});
    }
    return result;
}
} // namespace

BringupGrant StaticCandidate::grant(const Budgets& budgets) const { return BringupGrant(*this, budgets); }

BringupGrant::BringupGrant(const StaticCandidate& candidate, const Budgets& budgets) {
    if (!candidate.valid_) throw std::logic_error("moved-from static candidate");
    for (const auto& line : candidate.lines_) {
        if (line.allocation.source == AllocationSource::project && !line.padded_bytes)
            throw std::runtime_error("unknown enabled project allocation blocks bring-up: " + line.allocation.id);
        if (!line.bringup_cap)
            throw std::runtime_error("unbounded runtime/graph overhead blocks bring-up: " + line.allocation.id);
        if (!line.padded_bytes && *line.bringup_cap == 0)
            throw std::runtime_error("unreserved unknown runtime/graph overhead blocks bring-up: " + line.allocation.id);
    }
    const auto totals = candidate.known_reserved_bytes();
    for (std::size_t i = 0; i < budgets.size(); ++i) {
        const auto limit = std::min(budgets[i].configured, budgets[i].available);
        if (budgets[i].headroom > limit || totals[i] > limit - budgets[i].headroom)
            throw std::runtime_error("bring-up reservation exceeds budget/availability minus headroom in domain " + std::to_string(i));
    }
    state_ = std::make_shared<detail::GrantState>(candidate, budgets);
}

AllocationLease::AllocationLease(std::shared_ptr<detail::AllocationControl> control) : control_(std::move(control)) {}
std::uint64_t AllocationLease::bytes() const noexcept { return control_ ? control_->bytes : 0; }

AllocationLease BringupGrant::allocate(const std::string& id, std::uint64_t reservation,
                                     const std::function<MemoryBacking(std::uint64_t)>& factory) {
    const auto original = state_; // callbacks may move/replace the initiating grant
    auto& state = require_state(original);
    std::size_t index;
    {
        std::lock_guard lock(state.mutex);
        require_open(state);
        const auto alias = std::find_if(state.candidate.aliases_.begin(), state.candidate.aliases_.end(),
                                        [&](const auto& item) { return item.first == id; });
        if (alias == state.candidate.aliases_.end()) throw std::invalid_argument("allocation not granted: " + id);
        index = alias->second;
        const auto& line = state.candidate.lines_[index];
        if (line.allocation.source != AllocationSource::project)
            throw std::invalid_argument("external overhead must be monitored, not project-allocated: " + id);
        auto& observation = state.observations[index];
        const auto cap = *line.bringup_cap;
        if (reservation > cap || observation.live > cap - reservation ||
            observation.pending > cap - reservation - observation.live) {
            state.failed = true;
            throw std::runtime_error("actual allocation cap exhausted before allocation: " + id);
        }
        observation.pending += reservation;
        ++state.active_factories;
        state.warmed = false;
        state.captured = false;
    }
    // Declare cleanup owners outside the locked commit scope: no user deleter runs under it,
    // even if control/registry allocation or validation throws.
    MemoryBacking backing;
    std::shared_ptr<detail::AllocationControl> control;
    try {
        backing = factory(reservation);
        std::lock_guard lock(state.mutex);
        auto& observation = state.observations[index];
        observation.peak = std::max(observation.peak, checked_add(observation.live, backing.actual_bytes));
        observation.measured = true;
        if (backing.actual_bytes > reservation || (backing.actual_bytes != 0 && !backing.owner))
            throw std::runtime_error("allocator violated granted physical byte cap/ownership: " + id);
        if (state_ != original) throw std::logic_error("grant state changed during allocation factory");
        require_open(state); // a reentrant factory may have failed/aborted the grant
        control = std::make_shared<detail::AllocationControl>(original, index);
        observation.owners.emplace(control.get(), control);
        control->bytes = backing.actual_bytes;
        control->owner = std::move(backing.owner);
        control->registered = true;
        observation.live += control->bytes;
        observation.pending -= reservation;
        --state.active_factories;
        return AllocationLease(std::move(control));
    } catch (...) {
        {
            std::lock_guard lock(state.mutex);
            state.failed = true;
        }
        // Keep the transaction active through rejected-result destruction and its callbacks.
        backing.owner.reset();
        control.reset();
        {
            std::lock_guard lock(state.mutex);
            state.observations[index].pending -= reservation;
            --state.active_factories;
        }
        throw;
    }
}

void BringupGrant::observe_overhead(const std::string& id, std::uint64_t actual_bytes) {
    auto& state = require_state(state_);
    std::lock_guard lock(state.mutex);
    // Monitoring remains available during failed-grant unwind, without reopening allocation.
    if (state.reconciled) throw std::logic_error("bring-up grant already reconciled");
    const auto alias = std::find_if(state.candidate.aliases_.begin(), state.candidate.aliases_.end(),
                                    [&](const auto& item) { return item.first == id; });
    if (alias == state.candidate.aliases_.end()) throw std::invalid_argument("overhead not granted: " + id);
    const auto index = alias->second;
    const auto& line = state.candidate.lines_[index];
    if (line.allocation.source == AllocationSource::project)
        throw std::invalid_argument("project measurements must come from physical allocation owners: " + id);
    auto& observation = state.observations[index];
    observation.live = actual_bytes;
    observation.peak = std::max(observation.peak, actual_bytes);
    observation.measured = true;
    if (actual_bytes > *line.bringup_cap) {
        state.failed = true;
        throw std::runtime_error("runtime/graph overhead exceeded abort bound: " + id);
    }
}

void BringupGrant::mark_warmed() {
    auto& state = require_state(state_);
    std::lock_guard lock(state.mutex);
    require_open(state);
    require_quiescent(state);
    state.warmed = true;
    state.captured = false;
    for (std::size_t i = 0; i < state.observations.size(); ++i)
        if (state.candidate.unique_allocations()[i].allocation.source != AllocationSource::project)
            state.observations[i].measured = false;
}
void BringupGrant::mark_graph_captured() {
    auto& state = require_state(state_);
    std::lock_guard lock(state.mutex);
    require_open(state);
    require_quiescent(state);
    if (state.candidate.graphs() == GraphMode::disabled)
        throw std::logic_error("cannot capture a graph-disabled profile");
    if (!state.warmed) throw std::logic_error("graph capture requires selected-profile warmup");
    state.captured = true;
    for (std::size_t i = 0; i < state.observations.size(); ++i)
        if (state.candidate.unique_allocations()[i].allocation.source != AllocationSource::project)
            state.observations[i].measured = false;
}
void BringupGrant::abort() noexcept {
    if (state_) {
        std::lock_guard lock(state_->mutex);
        state_->failed = true;
    }
}
GrantReport BringupGrant::report() const {
    auto& state = require_state(state_);
    std::lock_guard lock(state.mutex);
    return report_state(state);
}

ReconciledResources BringupGrant::reconcile() && {
    const auto original = state_;
    auto& state = require_state(original);
    // On failure these pins must unwind after the mutex, since they may be last owners.
    std::vector<std::shared_ptr<detail::AllocationControl>> retained;
    std::vector<std::shared_ptr<detail::AllocationControl>> temporary;
    std::lock_guard lock(state.mutex);
    require_open(state);
    require_quiescent(state);
    if (!state.warmed) throw std::logic_error("readiness requires selected-profile warmup");
    if (state.candidate.graphs() == GraphMode::enabled && !state.captured)
        throw std::logic_error("readiness requires selected-profile graph capture");
    // Reserve before locking any weak owner, so vector growth cannot destroy a last-owner
    // temporary under the mutex. Registry size is bounded by live/releasing controls.
    std::size_t owner_count = 0;
    for (const auto& observation : state.observations) owner_count += observation.owners.size();
    retained.reserve(owner_count);
    temporary.reserve(owner_count);
    const auto& lines = state.candidate.unique_allocations();
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        const auto& observation = state.observations[i];
        if (observation.pending != 0 || !observation.measured)
            throw std::logic_error("unresolved allocation/overhead measurement: " + line.allocation.id);
        if (observation.peak > *line.bringup_cap)
            throw std::runtime_error("measured peak exceeds grant: " + line.allocation.id);
        if (line.allocation.source == AllocationSource::project) {
            if (observation.peak != *line.padded_bytes)
                throw std::logic_error("physical allocation does not match planned padded bytes: " + line.allocation.id);
            std::uint64_t covered = 0;
            auto& pins = line.allocation.lifetime == Lifetime::persistent ? retained : temporary;
            for (const auto& [identity, owner] : observation.owners) {
                auto live = owner.lock();
                // Expiry can precede the destructor acquiring this mutex. Reject that window,
                // including zero-byte owners, instead of relying on the release counter alone.
                if (!live) throw std::logic_error("allocation release in progress: " + line.allocation.id);
                pins.push_back(std::move(live));
                covered = checked_add(covered, pins.back()->bytes);
            }
            if (covered != observation.live)
                throw std::logic_error("physical allocation owner coverage missing: " + line.allocation.id);
            if (line.allocation.lifetime == Lifetime::persistent && covered != *line.padded_bytes)
                throw std::logic_error("persistent allocation owner missing: " + line.allocation.id);
        }
    }
    const auto measured = report_state(state).measured_peaks;
    for (std::size_t i = 0; i < measured.size(); ++i) {
        const auto limit = std::min(state.budgets[i].configured, state.budgets[i].available) - state.budgets[i].headroom;
        if (measured[i] > limit) throw std::runtime_error("measured high-water total exceeds usable budget");
    }
    state.reconciled = true;
    return ReconciledResources(std::move(state_), std::move(retained));
}

ReconciledResources::ReconciledResources(std::shared_ptr<detail::GrantState> state,
                                       std::vector<std::shared_ptr<detail::AllocationControl>> retained)
    : state_(std::move(state)), retained_(std::move(retained)) {}
ReadinessApproval ReconciledResources::approve() && {
    const auto original = state_;
    auto& state = require_state(original);
    std::lock_guard lock(state.mutex);
    if (!state.reconciled || state.failed || state.approved)
        throw std::logic_error("readiness requires a unique successful reconciliation");
    state.approved = true;
    return ReadinessApproval(std::move(state_), std::move(retained_));
}
ReadinessApproval::ReadinessApproval(std::shared_ptr<detail::GrantState> state,
                                   std::vector<std::shared_ptr<detail::AllocationControl>> retained)
    : state_(std::move(state)), retained_(std::move(retained)) {}
const std::string& ReadinessApproval::profile_identity() const { return require_state(state_).candidate.profile_identity(); }
GraphMode ReadinessApproval::graphs() const { return require_state(state_).candidate.graphs(); }
GrantReport ReadinessApproval::report() const {
    auto& state = require_state(state_);
    std::lock_guard lock(state.mutex);
    return report_state(state);
}

} // namespace saga::memory
