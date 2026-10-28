#include "saga/runtime/sequence_slots.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace {
using namespace saga::runtime;

void check(bool value, const char* message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
template<class F> void throws(F&& f, const char* message) {
    try { f(); } catch (const std::exception&) { return; }
    check(false, message);
}

void test_reuse_pending_and_stale() {
    SequenceSlots pool(1);
    auto first = pool.acquire();
    CompletionSource fence_source;
    auto fence = fence_source.completion();
    auto token = pool.begin_release(first);
    check(pool.counts().retiring == 1, "release immediately retires handle");
    throws([&] { (void)pool.begin_release(first); }, "stale handle rejected");
    check(pool.finish_release(std::move(token), fence) == RetirementResult::pending,
          "pending fence remains retiring");
    throws([&] { (void)pool.acquire(); }, "retiring slot unavailable");
    fence_source.succeed();
    check(pool.finish_release(std::move(token), fence) == RetirementResult::reusable,
          "successful fence permits reuse");
    throws([&] { (void)pool.finish_release(std::move(token), fence); },
           "duplicate finish rejected");
    auto second = pool.acquire();
    check(second.slot == first.slot && second.generation > first.generation,
          "reused slot gets a new generation");
    throws([&] { (void)pool.begin_release(first); }, "old generation rejected");
}

void test_failures_and_invalid() {
    for (auto status : {CompletionStatus::failure, CompletionStatus::cancelled,
                        CompletionStatus::abandoned}) {
        SequenceSlots pool(1);
        auto handle = pool.acquire();
        auto token = pool.begin_release(handle);
        throws([&] { (void)pool.finish_release(std::move(token), Completion{}); },
               "invalid completion rejected");
        check(pool.counts().retiring == 1 && token.valid(), "invalid fence preserves reservation");
        CompletionSource source;
        auto fence = source.completion();
        if (status == CompletionStatus::failure) source.fail();
        else if (status == CompletionStatus::cancelled) source.cancel();
        else source = CompletionSource{};
        check(pool.finish_release(std::move(token), fence) == RetirementResult::quarantined,
              "non-success completion quarantines");
        check(pool.counts().quarantined == 1, "quarantined slot counted");
        throws([&] { (void)pool.acquire(); }, "quarantine never reused");
    }
}

void test_cross_pool_moved_and_abandoned_token() {
    SequenceSlots a(1), b(1);
    auto h = a.acquire();
    auto moved = a.begin_release(h);
    SequenceSlots::RetirementToken token(std::move(moved));
    CompletionSource source;
    auto fence = source.completion();
    throws([&] { (void)b.finish_release(std::move(token), fence); }, "cross-pool token rejected");
    check(token.valid(), "cross-pool rejection preserves token");
    token = SequenceSlots::RetirementToken{};
    check(a.counts().retiring == 1, "abandoned token leaves reservation retiring");
    throws([&] { (void)a.acquire(); }, "abandoned token does not recycle slot");
}

void test_exhaustion_and_shutdown() {
    throws([] { SequenceSlots pool(0); }, "zero capacity rejected");
    SequenceSlots::RetirementToken orphan;
    {
        SequenceSlots pool(1);
        auto h = pool.acquire();
        throws([&] { (void)pool.acquire(); }, "bounded pool reports exhaustion");
        orphan = pool.begin_release(h);
    }
    check(orphan.valid(), "pool shutdown does not implicitly reclaim token reservation");
}

void test_full_accounting_transitions() {
    SequenceSlots pool(3);
    auto counts = pool.counts();
    check(counts.active == 0 && counts.retiring == 0 && counts.quarantined == 0 && counts.free == 3,
          "initial counts account for every slot");
    auto a = pool.acquire();
    auto b = pool.acquire();
    check(pool.counts().active == 2 && pool.counts().free == 1 &&
          pool.counts().retiring == 0 && pool.counts().quarantined == 0,
          "acquisition moves free to active");
    auto pending_token = pool.begin_release(a);
    check(pool.counts().active == 1 && pool.counts().retiring == 1 &&
          pool.counts().free == 1 && pool.counts().quarantined == 0,
          "begin release moves active to retiring");
    CompletionSource pending_source;
    auto pending_fence = pending_source.completion();
    check(pool.finish_release(std::move(pending_token), pending_fence) == RetirementResult::pending,
          "pending remains pending");
    check(pool.counts().active == 1 && pool.counts().retiring == 1 &&
          pool.counts().free == 1 && pool.counts().quarantined == 0,
          "pending leaves counts unchanged");
    pending_source.succeed();
    check(pool.finish_release(std::move(pending_token), pending_fence) == RetirementResult::reusable,
          "success frees slot");
    check(pool.counts().active == 1 && pool.counts().retiring == 0 &&
          pool.counts().free == 2 && pool.counts().quarantined == 0,
          "success moves retiring to free");
    auto quarantine_token = pool.begin_release(b);
    CompletionSource failed_source;
    auto failed_fence = failed_source.completion();
    failed_source.fail();
    check(pool.finish_release(std::move(quarantine_token), failed_fence) == RetirementResult::quarantined,
          "failure quarantines slot");
    counts = pool.counts();
    check(counts.active == 0 && counts.retiring == 0 && counts.free == 2 && counts.quarantined == 1,
          "failure moves retiring to quarantined");
}

void test_wrong_thread_rejected_without_changes() {
    SequenceSlots pool(2);
    auto active = pool.acquire();
    auto retiring = pool.begin_release(active);
    CompletionSource source;
    auto fence = source.completion();
    bool acquire_rejected = false, begin_rejected = false;
    bool finish_rejected = false, counts_rejected = false;
    std::thread wrong_thread([&] {
        try { (void)pool.acquire(); } catch (const std::logic_error&) { acquire_rejected = true; }
        try { (void)pool.begin_release(active); } catch (const std::logic_error&) { begin_rejected = true; }
        try { (void)pool.finish_release(std::move(retiring), fence); }
        catch (const std::logic_error&) { finish_rejected = true; }
        try { (void)pool.counts(); } catch (const std::logic_error&) { counts_rejected = true; }
    });
    wrong_thread.join();
    check(acquire_rejected && begin_rejected && finish_rejected && counts_rejected,
          "all operations reject non-owner thread");
    auto counts = pool.counts();
    check(counts.active == 0 && counts.retiring == 1 && counts.quarantined == 0 && counts.free == 1,
          "wrong-thread operations leave pool counts unchanged");
    check(retiring.valid(), "wrong-thread finish preserves token");
}

void test_token_moves_and_overwrite() {
    SequenceSlots pool(2);
    auto first_handle = pool.acquire();
    auto second_handle = pool.acquire();
    auto first = pool.begin_release(first_handle);
    auto second = pool.begin_release(second_handle);
    SequenceSlots::RetirementToken moved(std::move(first));
    check(!first.valid() && moved.valid(), "move construction transfers token ownership");
    moved = std::move(second);
    check(!second.valid() && moved.valid(), "move assignment transfers replacement token");
    check(pool.counts().retiring == 2, "overwriting live token does not recycle its slot");
    CompletionSource source;
    auto fence = source.completion();
    source.succeed();
    check(pool.finish_release(std::move(moved), fence) == RetirementResult::reusable,
          "moved token can finish exactly once");
    throws([&] { (void)pool.finish_release(std::move(moved), fence); },
           "moved and finished token cannot finish twice");
    auto counts = pool.counts();
    check(counts.retiring == 1 && counts.free == 1 && counts.active == 0 && counts.quarantined == 0,
          "overwritten token slot remains retiring after other token finishes");
    const auto available = pool.acquire();
    check(available.slot == second_handle.slot && available.slot != first_handle.slot,
          "only the successfully retired slot is reusable");
    counts = pool.counts();
    check(counts.active == 1 && counts.retiring == 1 && counts.free == 0 && counts.quarantined == 0,
          "overwritten token reservation remains accounted as retiring");
}
}

int main() {
    static_assert(!std::is_copy_constructible_v<SequenceSlots::RetirementToken>);
    test_reuse_pending_and_stale();
    test_failures_and_invalid();
    test_cross_pool_moved_and_abandoned_token();
    test_exhaustion_and_shutdown();
    test_full_accounting_transitions();
    test_wrong_thread_rejected_without_changes();
    test_token_moves_and_overwrite();
}
