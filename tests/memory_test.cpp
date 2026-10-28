#include "saga/memory/ledger.h"
#include "saga/kv/catalog.h"

#include <atomic>
#include <barrier>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace saga::memory;

static_assert(!std::is_copy_constructible_v<OwnedBacking> && !std::is_copy_assignable_v<OwnedBacking>);
static_assert(!std::is_copy_constructible_v<MemoryBacking> && !std::is_copy_assignable_v<MemoryBacking>);
static_assert(std::is_nothrow_move_constructible_v<OwnedBacking> && std::is_nothrow_move_assignable_v<OwnedBacking>);
static_assert(std::is_nothrow_move_constructible_v<MemoryBacking>);
static_assert(!std::is_constructible_v<OwnedBacking, std::shared_ptr<void>>);
static_assert(!std::is_constructible_v<OwnedBacking, std::shared_ptr<std::vector<std::byte>>>);
static_assert(!std::is_constructible_v<MemoryBacking, std::shared_ptr<void>, std::uint64_t>);
static_assert(!std::is_constructible_v<std::shared_ptr<void>, OwnedBacking>);

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void throws_with(F&& function, const std::string& fragment) {
    try { function(); }
    catch (const std::exception& error) {
        check(std::string(error.what()).find(fragment) != std::string::npos,
              "wrong exception: " + std::string(error.what()) + "; wanted " + fragment);
        return;
    }
    throw std::runtime_error("expected exception containing: " + fragment);
}
AllocationRecord project(std::string id, std::uint64_t bytes, Location location = Location::device) {
    AllocationRecord result;
    result.id = std::move(id);
    result.owner = "synthetic-component";
    result.location = location;
    result.logical_bytes = bytes;
    return result;
}
AllocationRecord overhead(std::string id = "runtime", std::uint64_t bound = 64,
                          AllocationSource source = AllocationSource::runtime_overhead) {
    auto result = project(std::move(id), 0);
    result.source = source;
    result.logical_bytes.reset();
    result.overhead_bound = bound;
    return result;
}
std::vector<AllocationRecord> plain_records() {
    return {project("weights", 256), project("host", 128, Location::host),
            project("pinned", 64, Location::pinned), overhead()};
}
Budgets budgets(std::uint64_t bytes = 4096) {
    return {DomainBudget{bytes, bytes, 64}, DomainBudget{bytes, bytes, 64}, DomainBudget{bytes, bytes, 64}};
}
StaticCandidate plain_candidate() { return {"synthetic-eager-8k", GraphMode::disabled, plain_records()}; }
MemoryBacking storage(std::uint64_t cap) {
    return {OwnedBacking(std::make_unique<std::vector<std::byte>>(static_cast<std::size_t>(cap))), cap};
}
struct StorageCounters {
    std::atomic<int> live{0};
    std::atomic<int> releases{0};
};
MemoryBacking tracked_storage(std::uint64_t cap, const std::shared_ptr<StorageCounters>& counters) {
    auto buffer = std::make_unique<std::vector<std::byte>>(static_cast<std::size_t>(cap));
    auto deleter = [counters](std::vector<std::byte>* pointer) noexcept {
        delete pointer;
        --counters->live;
        ++counters->releases;
    };
    std::unique_ptr<std::vector<std::byte>, decltype(deleter)> owner(buffer.release(), std::move(deleter));
    ++counters->live;
    return {OwnedBacking(std::move(owner)), cap};
}
MemoryBacking callback_storage(std::uint64_t cap, std::function<void()> callback,
                               std::exception_ptr& error) {
    auto buffer = std::make_unique<std::vector<std::byte>>(static_cast<std::size_t>(cap));
    auto deleter = [callback = std::move(callback), &error](std::vector<std::byte>* pointer) noexcept {
        try { callback(); }
        catch (...) { error = std::current_exception(); }
        delete pointer;
    };
    std::unique_ptr<std::vector<std::byte>, decltype(deleter)> owner(buffer.release(), std::move(deleter));
    return {OwnedBacking(std::move(owner)), cap};
}
AllocationLease allocate(BringupGrant& grant, const std::string& id, std::uint64_t cap) {
    return grant.allocate(id, cap, storage);
}
std::vector<AllocationLease> allocate_plain(BringupGrant& grant) {
    std::vector<AllocationLease> leases;
    leases.push_back(allocate(grant, "weights", 256));
    leases.push_back(allocate(grant, "host", 128));
    leases.push_back(allocate(grant, "pinned", 64));
    return leases;
}
void warm_and_measure(BringupGrant& grant) {
    grant.mark_warmed();
    grant.observe_overhead("runtime", 32);
}

void candidate_tests() {
    check(!std::is_convertible_v<StaticCandidate, ReadinessApproval>, "candidate convertible to approval");
    check(!std::is_convertible_v<BringupGrant, ReadinessApproval>, "grant convertible to approval");
    check(!std::is_default_constructible_v<ReadinessApproval> && !std::is_copy_constructible_v<BringupGrant>,
          "resource proof can be forged/copied");
    const auto candidate = plain_candidate();
    check(candidate.has_unresolved_terms(), "unmeasured runtime overhead reported resolved");
    const auto totals = candidate.known_reserved_bytes();
    check(totals == DomainBytes{128, 64, 320}, "separate host/pinned/device domains");
    auto limits = budgets();
    limits[2] = {1000, 383, 64};
    throws_with([&] { candidate.grant(limits); }, "budget/availability minus headroom");
    limits[2] = {383, 1000, 64};
    throws_with([&] { candidate.grant(limits); }, "budget/availability minus headroom");
    limits[2] = {384, 384, 64};
    auto exact = candidate.grant(limits);
    check(exact.report().live_bytes == DomainBytes{}, "grant is not an actual allocation");
    limits[0] = {10, 10, 11};
    throws_with([&] { candidate.grant(limits); }, "headroom");

    auto records = plain_records();
    records[0].logical_bytes.reset();
    StaticCandidate unknown("unknown-mtp", GraphMode::disabled, records);
    check(unknown.has_unresolved_terms(), "unknown term resolved");
    throws_with([&] { unknown.grant(budgets()); }, "unknown enabled project allocation");
    records[0].overhead_bound = 1024;
    throws_with([&] { StaticCandidate("hidden-project", GraphMode::disabled, records); }, "cannot use an overhead allowance");
    records[0].overhead_bound.reset();
    records[0].enabled = false;
    auto disabled_unknown = StaticCandidate("component-disabled", GraphMode::disabled, records).grant(budgets());
    throws_with([&] { allocate(disabled_unknown, "weights", 1); }, "not granted");
    records = plain_records();
    records.back().overhead_bound.reset();
    StaticCandidate unbounded("unbounded", GraphMode::disabled, records);
    throws_with([&] { unbounded.grant(budgets()); }, "unbounded runtime/graph");
    auto zero_unknown = overhead("runtime", 0);
    StaticCandidate unreserved("unreserved", GraphMode::disabled, {zero_unknown});
    throws_with([&] { unreserved.grant(budgets()); }, "unreserved unknown");
    zero_unknown.logical_bytes = 0; // explicit known zero still needs actual post-warm evidence
    auto known_zero = StaticCandidate("known-zero", GraphMode::disabled, {zero_unknown}).grant(budgets());
    known_zero.mark_warmed();
    throws_with([&] { std::move(known_zero).reconcile(); }, "unresolved");
    known_zero.observe_overhead("runtime", 0);
    auto zero_ready = std::move(std::move(known_zero).reconcile()).approve();
    check(zero_ready.report().measured_peaks == DomainBytes{}, "explicit zero measurement");
    records.pop_back();
    throws_with([&] { StaticCandidate("implicit-runtime-zero", GraphMode::disabled, records); }, "explicitly reserve runtime");
    records.push_back(overhead("runtime", 32));
    records.back().logical_bytes = 33;
    throws_with([&] { StaticCandidate("bad-estimate", GraphMode::disabled, records); }, "estimate exceeds");

    auto aligned = project("aligned", 257);
    aligned.alignment = 256;
    aligned.page_size = 1024;
    StaticCandidate rounded("rounded", GraphMode::disabled, {aligned, overhead("runtime", 0)});
    check(rounded.unique_allocations().front().padded_bytes == 1024, "alignment and page rounding");
    check(rounded.known_reserved_bytes()[2] == 1024, "padded bytes not charged");
    auto bad = aligned;
    bad.alignment = 3;
    throws_with([&] { StaticCandidate("bad", GraphMode::disabled, {bad, overhead()}); }, "alignment/page");
    bad = aligned;
    bad.page_size = 0;
    throws_with([&] { StaticCandidate("bad", GraphMode::disabled, {bad, overhead()}); }, "alignment/page");
    bad = aligned;
    bad.logical_bytes = std::numeric_limits<std::uint64_t>::max();
    throws_with([&] { StaticCandidate("overflow", GraphMode::disabled, {bad, overhead()}); }, "overflow");
    auto huge = project("huge", std::numeric_limits<std::uint64_t>::max());
    StaticCandidate overflow_total("overflow-total", GraphMode::disabled, {huge, overhead()});
    throws_with([&] { overflow_total.known_reserved_bytes(); }, "overflow");
    throws_with([&] { overflow_total.grant(budgets()); }, "overflow");
    throws_with([&] { StaticCandidate("duplicate-id", GraphMode::disabled, {aligned, aligned, overhead()}); }, "duplicate allocation ID");
    throws_with([&] { StaticCandidate("", GraphMode::disabled, plain_records()); }, "profile identity");

    const auto cache8 = saga::kv::main_cache_bytes({}, {saga::kv::Codec::int8_g64, saga::kv::Codec::int8_g64}, 8192).total_bytes;
    const auto cache16 = saga::kv::main_cache_bytes({}, {saga::kv::Codec::int8_g64, saga::kv::Codec::int8_g64}, 16384).total_bytes;
    auto first = plain_records();
    first.push_back(project("main-kv", cache8));
    auto second = first;
    second.back().logical_bytes = cache16;
    second.push_back(project("mtp-kv", cache8 / 16));
    check(StaticCandidate("16k-mtp", GraphMode::disabled, second).known_reserved_bytes()[2] ==
              StaticCandidate("8k", GraphMode::disabled, first).known_reserved_bytes()[2] + cache8 + cache8 / 16,
          "context/components do not change authoritative ledger");
}

void sharing_tests() {
    auto text = project("embedding", 256);
    text.sharing_identity = "artifact/tensor-42/packing-v1";
    auto draft = text;
    draft.id = "draft-embedding";
    draft.owner = "draft";
    StaticCandidate shared("shared", GraphMode::disabled, {text, draft, overhead()});
    check(shared.unique_allocations().size() == 2 && shared.known_reserved_bytes()[2] == 320,
          "shared physical text/draft weights counted twice");
    const std::vector<std::function<void(AllocationRecord&)>> mismatches{
        [](auto& a) { a.logical_bytes = 257; }, [](auto& a) { a.alignment = 2; },
        [](auto& a) { a.page_size = 2; }, [](auto& a) { a.location = Location::host; },
        [](auto& a) { a.lifetime = Lifetime::execution; }, [](auto& a) { a.logical_bytes.reset(); },
        [](auto& a) { a.source = AllocationSource::runtime_overhead; a.overhead_bound = 256; }
    };
    for (const auto& mutate : mismatches) {
        auto changed = draft;
        mutate(changed);
        throws_with([&] { StaticCandidate("bad-share", GraphMode::disabled, {text, changed, overhead()}); }, "sharing identity");
    }
    auto unique = draft;
    unique.sharing_identity.clear();
    check(StaticCandidate("not-shared", GraphMode::disabled, {text, unique, overhead()}).known_reserved_bytes()[2] == 576,
          "empty sharing identity deduplicated unrelated storage");
    auto grant = shared.grant(budgets());
    auto alias_lease = allocate(grant, "draft-embedding", 256);
    bool called = false;
    throws_with([&] { grant.allocate("embedding", 1, [&](auto) { called = true; return storage(1); }); }, "cap exhausted");
    check(!called && grant.report().live_bytes[2] == 256, "alias bypassed shared actual allocation cap");
    throws_with([&] { grant.mark_warmed(); }, "failed");
}

void readiness_tests() {
    auto grant = plain_candidate().grant(budgets());
    auto leases = allocate_plain(grant);
    check(leases.front().bytes() == 256, "lease physical bytes");
    throws_with([&] { std::move(grant).reconcile(); }, "warmup");
    grant.mark_warmed();
    throws_with([&] { std::move(grant).reconcile(); }, "unresolved allocation/overhead");
    grant.observe_overhead("runtime", 32);
    throws_with([&] { grant.mark_graph_captured(); }, "graph-disabled");
    auto reconciled = std::move(grant).reconcile();
    throws_with([&] { grant.report(); }, "moved-from");
    leases.clear(); // readiness proof must retain persistent owners independently of local handles
    auto approval = std::move(reconciled).approve();
    check(approval.profile_identity() == "synthetic-eager-8k" && approval.graphs() == GraphMode::disabled,
          "profile identity lost");
    check(approval.report().live_bytes == DomainBytes{128, 64, 288}, "approval did not retain actual owners");
    throws_with([&] { std::move(reconciled).approve(); }, "moved-from");

    auto undersized = plain_candidate().grant(budgets());
    auto partial = allocate(undersized, "weights", 128);
    auto host = allocate(undersized, "host", 128);
    auto pinned = allocate(undersized, "pinned", 64);
    warm_and_measure(undersized);
    throws_with([&] { std::move(undersized).reconcile(); }, "does not match planned padded bytes");
    partial = {};
    auto missing = allocate(undersized, "weights", 256);
    warm_and_measure(undersized);
    missing = {};
    throws_with([&] { std::move(undersized).reconcile(); }, "persistent allocation owner missing");

    auto records = plain_records();
    records[0].lifetime = Lifetime::bringup;
    auto temporary = StaticCandidate("temporary", GraphMode::disabled, records).grant(budgets());
    auto temporary_leases = allocate_plain(temporary);
    temporary_leases.erase(temporary_leases.begin());
    warm_and_measure(temporary);
    auto ready_temporary = std::move(std::move(temporary).reconcile()).approve();
    check(ready_temporary.report().measured_peaks[2] == 288 && ready_temporary.report().live_bytes[2] == 32,
          "released bring-up allocation high-water lost");

    auto stale = plain_candidate().grant(budgets());
    stale.mark_warmed();
    stale.observe_overhead("runtime", 32);
    auto stale_leases = allocate_plain(stale);
    throws_with([&] { std::move(stale).reconcile(); }, "warmup");
    stale.mark_warmed();
    throws_with([&] { std::move(stale).reconcile(); }, "unresolved");
    stale.observe_overhead("runtime", 32);
    auto ready_stale = std::move(std::move(stale).reconcile()).approve();
    check(ready_stale.report().measured_peaks[2] == 288, "post-warm reconciliation");
}

void ownership_tests() {
    auto moved_counters = std::make_shared<StorageCounters>();
    {
        auto backing = tracked_storage(32, moved_counters);
        OwnedBacking moved(std::move(backing.owner));
        check(!backing.owner && moved, "backing move did not transfer exclusive ownership");
        OwnedBacking assigned;
        assigned = std::move(moved);
        check(!moved && assigned && moved_counters->live == 1, "backing move assignment lost ownership");
        assigned.reset();
        check(moved_counters->live == 0 && moved_counters->releases == 1, "unique backing reset did not release once");
    }
    check(moved_counters->releases == 1, "moved-from backing double released");
    OwnedBacking empty(std::unique_ptr<std::vector<std::byte>>{});
    check(!empty, "null unique_ptr became a nonempty backing");

    const StaticCandidate candidate("exclusive-backing", GraphMode::disabled,
                                    {project("weights", 256), overhead()});
    auto counters = std::make_shared<StorageCounters>();
    {
        auto grant = candidate.grant(budgets());
        auto lease = grant.allocate("weights", 256, [&](auto cap) { return tracked_storage(cap, counters); });
        auto consumer = lease; // this is the only supported shared retention of physical storage
        lease = {};
        check(counters->live == 1 && grant.report().live_bytes[2] == 256,
              "retained lease lost physical storage or live byte accounting");
        bool factory_called = false;
        throws_with([&] {
            grant.allocate("weights", 256, [&](auto cap) { factory_called = true; return storage(cap); });
        }, "cap exhausted");
        check(!factory_called && counters->live == 1, "retained lease allowed another cap allocation");
        consumer = {};
        check(counters->live == 0 && counters->releases == 1 && grant.report().live_bytes[2] == 0,
              "last lease did not release physical storage and accounting together");
    }

    counters = std::make_shared<StorageCounters>();
    MemoryBacking factory_result;
    {
        auto grant = candidate.grant(budgets());
        auto lease = grant.allocate("weights", 256, [&](auto cap) {
            factory_result = tracked_storage(cap, counters);
            return std::move(factory_result);
        });
        check(!factory_result.owner && counters->live == 1, "factory retained a backing owner after transfer");
        auto consumer = lease;
        lease = {};
        consumer = {};
        check(counters->live == 0 && counters->releases == 1 && grant.report().live_bytes[2] == 0,
              "cap became reusable before physical storage release");
        auto replacement = grant.allocate("weights", 256, [&](auto cap) {
            check(counters->live == 0, "new cap allocation overlaps released accounting");
            return tracked_storage(cap, counters);
        });
        consumer = replacement;
        replacement = {};
        warm_and_measure(grant);
        auto reconciled = std::move(grant).reconcile();
        consumer = {};
        check(counters->live == 1, "reconciliation did not retain the unique physical owner");
        auto approval = std::move(reconciled).approve();
        check(counters->live == 1 && approval.report().live_bytes[2] == 288,
              "readiness undercounted retained physical storage");
    }
    check(counters->live == 0 && counters->releases == 2 && !factory_result.owner,
          "readiness/moved factory result leaked or double released backing");
}

void failure_tests() {
    auto grant = plain_candidate().grant(budgets());
    auto prior_owner = std::make_shared<StorageCounters>();
    auto first = grant.allocate("weights", 256, [&](auto cap) { return tracked_storage(cap, prior_owner); });
    auto partial_owner = std::make_shared<StorageCounters>();
    throws_with([&] {
        grant.allocate("host", 128, [&](auto cap) -> MemoryBacking {
            auto partial = tracked_storage(cap / 2, partial_owner);
            throw std::runtime_error("synthetic partial allocator failure");
        });
    }, "partial allocator failure");
    check(partial_owner->live == 0 && partial_owner->releases == 1 && prior_owner->live == 1 &&
          prior_owner->releases == 0, "partial failure RAII");
    check(grant.report().live_bytes == DomainBytes{0, 0, 256}, "partial failure lost earlier live allocation accounting");
    throws_with([&] { allocate(grant, "pinned", 64); }, "failed");
    throws_with([&] { std::move(grant).reconcile(); }, "failed");
    first = {};
    check(prior_owner->live == 0 && prior_owner->releases == 1 && grant.report().live_bytes == DomainBytes{},
          "failed grant unwind accounting");
    grant.abort();
    grant.abort();

    auto over_cap = plain_candidate().grant(budgets());
    bool called = false;
    throws_with([&] { over_cap.allocate("weights", 257, [&](auto cap) { called = true; return storage(cap); }); }, "before allocation");
    check(!called, "allocator ran before cap check");

    auto liar = plain_candidate().grant(budgets());
    auto rejected_owner = std::make_shared<StorageCounters>();
    throws_with([&] {
        liar.allocate("weights", 256, [&](auto cap) { return tracked_storage(cap + 1, rejected_owner); });
    }, "violated granted");
    check(rejected_owner->live == 0 && rejected_owner->releases == 1, "rejected backing was not released exactly once");
    check(liar.report().allocations.front().measured_peak == 257 && liar.report().live_bytes[2] == 0,
          "allocator violation peak/unwind lost");
    throws_with([&] { liar.mark_warmed(); }, "failed");
    auto ownerless = plain_candidate().grant(budgets());
    throws_with([&] { ownerless.allocate("weights", 256, [](auto cap) { return MemoryBacking{{}, cap}; }); }, "ownership");

    auto aborted = plain_candidate().grant(budgets());
    auto aborted_owner = std::make_shared<StorageCounters>();
    throws_with([&] {
        aborted.allocate("weights", 256, [&](auto cap) {
            auto backing = tracked_storage(cap, aborted_owner);
            aborted.abort();
            return backing;
        });
    }, "failed");
    check(aborted_owner->live == 0 && aborted_owner->releases == 1 && aborted.report().live_bytes == DomainBytes{},
          "post-factory rejection leaked backing or accounting");

    auto external = plain_candidate().grant(budgets());
    auto external_leases = allocate_plain(external);
    external.mark_warmed();
    external.observe_overhead("runtime", 64);
    external.observe_overhead("runtime", 0);
    check(external.report().allocations.back().measured_peak == 64, "external high-water replaced by latest observation");
    throws_with([&] { external.observe_overhead("runtime", 65); }, "exceeded abort bound");
    check(external.report().allocations.back().measured_peak == 65, "overhead failure peak not retained");
    external.observe_overhead("runtime", 0); // cleanup observations cannot clear the failed state
    check(external.report().allocations.back().live_bytes == 0 &&
          external.report().allocations.back().measured_peak == 65, "failed-overhead unwind accounting");
    throws_with([&] { std::move(external).reconcile(); }, "failed");

    auto misuse = plain_candidate().grant(budgets());
    throws_with([&] { allocate(misuse, "undisclosed", 1); }, "not granted");
    throws_with([&] { allocate(misuse, "runtime", 1); }, "must be monitored");
    throws_with([&] { misuse.observe_overhead("weights", 256); }, "physical allocation owners");
    throws_with([&] { misuse.observe_overhead("unknown", 1); }, "not granted");

    // Pending bytes are reserved before entering an allocator, including reentrant attempts.
    auto pending = plain_candidate().grant(budgets());
    throws_with([&] {
        pending.allocate("weights", 256, [&](auto cap) {
            allocate(pending, "weights", 1);
            return storage(cap);
        });
    }, "cap exhausted");
    check(pending.report().live_bytes[2] == 0, "pending reservation became live on failure");
}

void candidate_move_tests() {
    auto source = plain_candidate();
    auto moved = std::move(source);
    throws_with([&] { source.grant(budgets()); }, "moved-from static candidate");
    auto invalid_copy = source;
    throws_with([&] { invalid_copy.grant(budgets()); }, "moved-from static candidate");
    auto destination = plain_candidate();
    destination = std::move(moved);
    throws_with([&] { moved.grant(budgets()); }, "moved-from static candidate");
    auto grant = destination.grant(budgets());
    check(grant.report().allocations.size() == 4, "candidate move lost plan");
    source = destination;
    auto restored = source.grant(budgets());
    check(restored.report().allocations.size() == 4, "copy assignment did not restore candidate");
    destination = std::move(invalid_copy);
    throws_with([&] { destination.grant(budgets()); }, "moved-from static candidate");
}

void factory_transaction_tests() {
    const StaticCandidate candidate("transactions", GraphMode::disabled,
                                    {project("weights", 256), overhead()});
    auto counters = std::make_shared<StorageCounters>();
    auto grant = candidate.grant(budgets());
    std::optional<BringupGrant> moved;
    throws_with([&] {
        grant.allocate("weights", 256, [&](auto cap) {
            moved.emplace(std::move(grant));
            check(moved->report().live_bytes[2] == 0, "pending bytes reported as live");
            return tracked_storage(cap, counters);
        });
    }, "state changed");
    throws_with([&] { grant.report(); }, "moved-from");
    throws_with([&] { moved->mark_warmed(); }, "failed");
    check(counters->live == 0 && counters->releases == 1 && moved->report().live_bytes[2] == 0,
          "moving grant in factory leaked backing/accounting");

    grant = candidate.grant(budgets());
    throws_with([&] {
        grant.allocate("weights", 256, [&](auto cap) {
            // No external handle retains the old state: allocate itself must pin it.
            grant = candidate.grant(budgets());
            return tracked_storage(cap, counters);
        });
    }, "state changed");
    check(counters->live == 0 && counters->releases == 2, "replaced grant leaked result");
    auto replacement = allocate(grant, "weights", 256);
    replacement = {};
    throws_with([&] {
        grant.allocate("weights", 256, [&](auto cap) -> MemoryBacking {
            grant = candidate.grant(budgets());
            auto partial = tracked_storage(cap, counters);
            throw std::runtime_error("replaced factory threw");
        });
    }, "replaced factory threw");
    replacement = allocate(grant, "weights", 256); // replacement must not inherit old failure
    replacement = {};

    throws_with([&] {
        grant.allocate("weights", 256, [&](auto cap) {
            { auto discarded = std::move(grant); }
            return storage(cap);
        });
    }, "state changed");
    grant = plain_candidate().grant(budgets());
    AllocationLease nested;
    auto lease = grant.allocate("weights", 256, [&](auto cap) {
        auto transfer = std::move(grant);
        grant = std::move(transfer); // returning to the identical state is supported
        nested = allocate(grant, "host", 128);
        check(grant.report().live_bytes[0] == 128, "reentrant report/nested allocation failed");
        throws_with([&] { grant.mark_warmed(); }, "active allocation/release");
        return storage(cap);
    });
    auto pinned = allocate(grant, "pinned", 64);
    warm_and_measure(grant);
    auto approval = std::move(std::move(grant).reconcile()).approve();
    check(approval.report().live_bytes == DomainBytes{128, 64, 288}, "round-trip grant move lost state");

    // Rejected backing destruction may itself reenter the coordinator. It must see failure,
    // run without the mutex, and remain covered by the still-active allocation transaction.
    auto failed = candidate.grant(budgets());
    std::exception_ptr cleanup_error;
    bool released = false;
    throws_with([&] {
        failed.allocate("weights", 256, [&](auto cap) {
            return callback_storage(cap + 1, [&] {
                released = true;
                check(failed.report().live_bytes[2] == 0, "rejected backing committed live bytes");
                throws_with([&] { failed.mark_warmed(); }, "failed");
                throws_with([&] { std::move(failed).reconcile(); }, "failed");
            }, cleanup_error);
        });
    }, "violated granted");
    if (cleanup_error) std::rethrow_exception(cleanup_error);
    check(released, "rejected backing deleter not run");
}

void zero_transaction_tests() {
    auto runtime = overhead("runtime", 0);
    runtime.logical_bytes = 0;
    auto graph = overhead("graphs", 0, AllocationSource::graph_overhead);
    graph.logical_bytes = 0;
    const StaticCandidate candidate("zero-transactions", GraphMode::enabled,
                                    {project("zero", 0), runtime, graph});
    auto grant = candidate.grant(budgets());
    auto counters = std::make_shared<StorageCounters>();
    auto first = grant.allocate("zero", 0, [&](auto cap) { return tracked_storage(cap, counters); });
    grant.mark_warmed();
    grant.mark_graph_captured();
    grant.observe_overhead("runtime", 0);
    grant.observe_overhead("graphs", 0);
    AllocationLease nested;
    auto second = grant.allocate("zero", 0, [&](auto cap) {
        nested = allocate(grant, "zero", 0);
        throws_with([&] { grant.mark_warmed(); }, "active allocation/release");
        throws_with([&] { grant.mark_graph_captured(); }, "active allocation/release");
        throws_with([&] { std::move(grant).reconcile(); }, "active allocation/release");
        check(grant.report().live_bytes == DomainBytes{}, "zero-byte transaction became live bytes");
        return tracked_storage(cap, counters);
    });
    throws_with([&] { std::move(grant).reconcile(); }, "warmup");
    grant.mark_warmed();
    grant.mark_graph_captured();
    grant.observe_overhead("runtime", 0);
    grant.observe_overhead("graphs", 0);
    {
        auto reconciled = std::move(grant).reconcile();
        first = {};
        second = {};
        nested = {};
        check(counters->live == 2, "zero-byte physical owners were not retained");
        auto approval = std::move(reconciled).approve();
        check(approval.report().live_bytes == DomainBytes{}, "zero approval accounting");
    }
    check(counters->live == 0 && counters->releases == 2, "zero-byte retained owners leaked");
}

void deleter_transaction_tests() {
    // Both nonzero and zero-byte last-owner deleters must block readiness transitions.
    for (const auto bytes : {std::uint64_t{256}, std::uint64_t{0}}) {
        const StaticCandidate candidate("release-transactions", GraphMode::disabled,
                                        {project("weights", bytes), overhead()});
        auto grant = candidate.grant(budgets());
        std::exception_ptr error;
        bool released = false;
        auto lease = grant.allocate("weights", bytes, [&](auto cap) {
            return callback_storage(cap, [&] {
                released = true;
                check(grant.report().allocations[0].live_bytes == bytes,
                      "accounting dropped before physical deleter returned");
                throws_with([&] { std::move(grant).reconcile(); }, "active allocation/release");
                throws_with([&] { grant.mark_warmed(); }, "active allocation/release");
                throws_with([&] { grant.mark_graph_captured(); }, "active allocation/release");
                grant.observe_overhead("runtime", 0);
            }, error);
        });
        warm_and_measure(grant);
        lease = {};
        if (error) std::rethrow_exception(error);
        check(released && grant.report().live_bytes[2] == 0, "release transaction failed to complete");
        if (bytes != 0)
            throws_with([&] { std::move(grant).reconcile(); }, "persistent allocation owner missing");
        auto fresh = allocate(grant, "weights", bytes);
        warm_and_measure(grant);
        auto approval = std::move(std::move(grant).reconcile()).approve();
        check(approval.report().allocations[0].live_bytes == bytes, "post-release allocation/reconcile failed");
    }

    auto grant = plain_candidate().grant(budgets());
    std::exception_ptr error;
    auto lease = grant.allocate("weights", 256, [&](auto cap) {
        return callback_storage(cap, [&] {
            bool called = false;
            throws_with([&] {
                grant.allocate("weights", 1, [&](auto size) { called = true; return storage(size); });
            }, "cap exhausted");
            check(!called, "release freed cap before physical deleter finished");
        }, error);
    });
    lease = {};
    if (error) std::rethrow_exception(error);
    check(grant.report().live_bytes[2] == 0, "failed reentrant allocation lost release accounting");
    throws_with([&] { grant.mark_warmed(); }, "failed");
}

void concurrent_release_tests() {
    const StaticCandidate candidate("concurrent-release", GraphMode::disabled,
                                    {project("weights", 256), overhead()});
    auto grant = candidate.grant(budgets());
    std::latch entered(1), resume(1);
    std::exception_ptr error;
    auto lease = grant.allocate("weights", 256, [&](auto cap) {
        return callback_storage(cap, [&] {
            entered.count_down();
            resume.wait(); // no coordinator calls on the release thread
        }, error);
    });
    warm_and_measure(grant);
    std::jthread worker([owned = std::move(lease)]() mutable { owned = {}; });
    entered.wait();
    try {
        check(grant.report().live_bytes[2] == 288, "concurrent in-flight deleter lost live bytes");
        throws_with([&] { std::move(grant).reconcile(); }, "active allocation/release");
        throws_with([&] { grant.mark_warmed(); }, "active allocation/release");
    } catch (...) {
        resume.count_down();
        worker.join();
        throw;
    }
    resume.count_down();
    worker.join();
    if (error) std::rethrow_exception(error);
    check(grant.report().live_bytes[2] == 32, "concurrent release did not retire accounting");
    throws_with([&] { std::move(grant).reconcile(); }, "persistent allocation owner missing");

    // Race independent final releases against weak-owner retention. A successful proof must
    // retain ALL physical owners; observing live counters alone cannot establish that proof.
    constexpr int threads = 4, per_thread = 8, count = threads * per_thread;
    const StaticCandidate split("split-owners", GraphMode::disabled,
                                {project("weights", count), overhead()});
    for (int iteration = 0; iteration < 40; ++iteration) {
        auto racing = split.grant(budgets());
        auto counters = std::make_shared<StorageCounters>();
        std::vector<std::vector<AllocationLease>> owners(threads);
        for (auto& group : owners)
            for (int i = 0; i < per_thread; ++i)
                group.push_back(racing.allocate("weights", 1, [&](auto cap) { return tracked_storage(cap, counters); }));
        warm_and_measure(racing);
        std::barrier start(threads + 1);
        std::atomic<int> done{0};
        std::vector<std::jthread> workers;
        for (auto& group : owners) {
            workers.emplace_back([owned = std::move(group), &start, &done]() mutable {
                start.arrive_and_wait();
                for (auto& item : owned) { item = {}; std::this_thread::yield(); }
                ++done;
            });
        }
        start.arrive_and_wait();
        std::optional<ReadinessApproval> approval;
        try {
            approval.emplace(std::move(std::move(racing).reconcile()).approve());
        } catch (const std::logic_error& failure) {
            const std::string message = failure.what();
            check(message.find("transaction") != std::string::npos ||
                  message.find("release in progress") != std::string::npos ||
                  message.find("owner missing") != std::string::npos,
                  "unexpected concurrent reconciliation failure: " + message);
        }
        do {
            const auto report = approval ? approval->report() : racing.report();
            check(report.allocations[0].live_bytes <= count && report.measured_peaks[2] == count + 32 &&
                  report.live_bytes[2] == report.allocations[0].live_bytes + 32,
                  "inconsistent snapshot during independent releases");
        } while (done.load() != threads);
        workers.clear(); // join before inspecting physical-owner counters
        if (approval) {
            check(counters->live == count && counters->releases == 0,
                  "reconciliation approved bytes without retaining actual owners");
            approval.reset();
        } else {
            check(counters->live == 0 && racing.report().allocations[0].live_bytes == 0,
                  "concurrent final releases leaked storage/accounting");
        }
        check(counters->live == 0 && counters->releases == count, "independent owners not released exactly once");
    }

    // Reconciliation pins only persistent controls. Execution releases must remain safe when
    // they overlap reports from a reconciled/approved state, not just a BringupGrant.
    auto temporary_record = project("scratch", count);
    temporary_record.lifetime = Lifetime::execution;
    auto temporary = StaticCandidate("temporary-release", GraphMode::disabled,
                                    {project("weights", 256), temporary_record, overhead()}).grant(budgets());
    auto persistent = allocate(temporary, "weights", 256);
    std::vector<AllocationLease> scratch;
    for (int i = 0; i < count; ++i) scratch.push_back(allocate(temporary, "scratch", 1));
    warm_and_measure(temporary);
    auto approval = std::move(std::move(temporary).reconcile()).approve();
    std::atomic<bool> done{false};
    std::jthread releaser([owned = std::move(scratch), &done]() mutable {
        for (auto& item : owned) { item = {}; std::this_thread::yield(); }
        done = true;
    });
    do {
        const auto report = approval.report();
        check(report.allocations[0].live_bytes == 256 && report.allocations[1].live_bytes <= count,
              "approval report raced with execution release");
    } while (!done.load());
    releaser.join();
    check(approval.report().live_bytes[2] == 288, "approval retained execution owners");
}

void owner_churn_tests() {
    const StaticCandidate candidate("owner-churn", GraphMode::disabled,
                                    {project("weights", 1), overhead()});
    auto grant = candidate.grant(budgets());
    // Exercise immediate registry retirement, including zero-byte controls. The registry has
    // no history API; this stresses bounded-history behavior without adding production hooks.
    for (int i = 0; i < 20000; ++i) {
        auto zero = allocate(grant, "weights", 0);
        auto one = allocate(grant, "weights", 1);
    }
    check(grant.report().allocations[0].live_bytes == 0, "churn leaked live bytes");
    auto final = allocate(grant, "weights", 1);
    warm_and_measure(grant);
    auto approval = std::move(std::move(grant).reconcile()).approve();
    final = {};
    check(approval.report().allocations[0].live_bytes == 1, "churn lost final retained owner");
}

void graph_tests() {
    auto records = plain_records();
    records.push_back(overhead("graphs", 32, AllocationSource::graph_overhead));
    throws_with([&] { StaticCandidate("eager", GraphMode::disabled, records); }, "graph-disabled");
    throws_with([&] { StaticCandidate("missing-graphs", GraphMode::enabled, plain_records()); }, "explicit graph overhead");
    records.back().enabled = false;
    auto eager = StaticCandidate("eager", GraphMode::disabled, records).grant(budgets());
    auto eager_leases = allocate_plain(eager);
    warm_and_measure(eager);
    auto eager_ready = std::move(std::move(eager).reconcile()).approve();
    check(eager_ready.graphs() == GraphMode::disabled, "eager falsely approves graphs");

    records.back().enabled = true;
    records.back().overhead_bound.reset();
    auto unbounded = StaticCandidate("graphs-unbounded", GraphMode::enabled, records);
    throws_with([&] { unbounded.grant(budgets()); }, "unbounded runtime/graph");
    records.back().overhead_bound = 32;
    auto candidate = StaticCandidate("synthetic-captured-8k", GraphMode::enabled, records);
    check(candidate.has_unresolved_terms(), "unmeasured graphs considered ready");
    auto grant = candidate.grant(budgets());
    auto leases = allocate_plain(grant);
    throws_with([&] { grant.mark_graph_captured(); }, "warmup");
    warm_and_measure(grant);
    grant.observe_overhead("graphs", 16);
    throws_with([&] { std::move(grant).reconcile(); }, "graph capture");
    grant.mark_graph_captured();
    throws_with([&] { std::move(grant).reconcile(); }, "unresolved");
    grant.observe_overhead("runtime", 32);
    throws_with([&] { std::move(grant).reconcile(); }, "unresolved");
    grant.observe_overhead("graphs", 24);
    auto ready = std::move(std::move(grant).reconcile()).approve();
    check(ready.graphs() == GraphMode::enabled && ready.profile_identity() == "synthetic-captured-8k",
          "graph profile approval");
    check(ready.report().measured_peaks[2] == 312, "graph high-water not charged");

    auto failed = candidate.grant(budgets());
    auto failed_leases = allocate_plain(failed);
    failed.mark_warmed();
    failed.mark_graph_captured();
    throws_with([&] { failed.observe_overhead("graphs", 33); }, "abort bound");
    throws_with([&] { std::move(failed).reconcile(); }, "failed");
}
} // namespace

int main() {
    try {
        candidate_tests();
        sharing_tests();
        readiness_tests();
        ownership_tests();
        failure_tests();
        candidate_move_tests();
        factory_transaction_tests();
        zero_transaction_tests();
        deleter_transaction_tests();
        concurrent_release_tests();
        owner_churn_tests();
        graph_tests();
        std::cout << "memory_test: passed (synthetic allocation evidence only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "memory_test: " << error.what() << '\n';
        return 1;
    }
}
