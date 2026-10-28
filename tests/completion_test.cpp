#include "saga/runtime/completion.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace saga::runtime;

static_assert(!std::is_constructible_v<ResourceLease, int*>);
static_assert(!std::is_constructible_v<ResourceLease, std::reference_wrapper<int>>);
static_assert(std::is_constructible_v<ResourceLease, std::unique_ptr<int>>);

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

struct Gate {
    void wait() {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return open; });
    }
    void release() {
        {
            std::lock_guard lock(mutex);
            open = true;
        }
        condition.notify_all();
    }
    std::mutex mutex;
    std::condition_variable condition;
    bool open{false};
};

void test_status_and_settle_once() {
    CompletionSource source;
    auto completion = source.completion();
    check(completion.status() == CompletionStatus::pending, "starts pending");
    check(!source.settle(CompletionStatus::pending), "rejects pending settlement");
    check(!source.settle(CompletionStatus::abandoned), "rejects explicit abandonment");
    check(!source.settle(static_cast<CompletionStatus>(99)), "rejects unknown status");
    check(!source.settle(CompletionStatus::success,
                         std::make_exception_ptr(std::runtime_error("unused"))),
          "rejects exception on success");
    check(completion.status() == CompletionStatus::pending,
          "rejected settlement does not consume pending state");
    check(source.succeed(), "success settles");
    check(!source.cancel(), "first settlement wins");
    completion.result();
    check(completion.status() == CompletionStatus::success, "success status visible");
}

void test_failure_and_invalid() {
    Completion invalid;
    bool threw = false;
    try { (void)invalid.status(); } catch (const std::logic_error&) { threw = true; }
    check(threw, "invalid consumer rejected");
    CompletionSource source;
    auto completion = source.completion();
    const auto error = std::make_exception_ptr(std::runtime_error("stored"));
    check(source.fail(error), "failure settles");
    threw = false;
    try { completion.result(); } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "stored";
    }
    check(threw, "result rethrows stored exception");
}

void test_move_and_abandon_wakes() {
    CompletionSource moved;
    Completion completion;
    {
        CompletionSource original;
        completion = original.completion();
        moved = std::move(original);
        check(!original.valid(), "moved-from source invalid");
    }
    moved = CompletionSource{};
    check(completion.wait().status == CompletionStatus::abandoned,
          "producer move assignment abandons old producer");
    check(moved.valid(), "move assignment transfers replacement producer");
    moved.cancel();
}

void test_timeout_and_concurrent_waiters() {
    CompletionSource source;
    auto completion = source.completion();
    const auto timed = completion.wait_for(std::chrono::milliseconds(0));
    check(timed.status == CompletionStatus::pending, "timeout preserves pending status");
    check(completion.wait_for(std::chrono::steady_clock::duration::min()).status ==
              CompletionStatus::pending,
          "minimum timeout polls pending without overflow");
    check(source.succeed(), "settle for extreme timeout checks");
    check(completion.wait_for(std::chrono::steady_clock::duration::max()).status ==
              CompletionStatus::success,
          "maximum timeout returns settled outcome without overflow");

    CompletionSource ready;
    auto ready_completion = ready.completion();
    ready.cancel();
    check(ready_completion.wait_for(std::chrono::steady_clock::duration::min()).status ==
              CompletionStatus::cancelled,
          "minimum timeout polls settled outcome");

    CompletionSource source_for_waiters;
    completion = source_for_waiters.completion();

    constexpr int count = 8;
    std::vector<std::thread> waiters;
    std::atomic<int> observed{0};
    Gate started;
    for (int i = 0; i < count; ++i) {
        waiters.emplace_back([&] {
            observed.fetch_add(1);
            started.wait();
            if (completion.wait().status == CompletionStatus::success) observed.fetch_add(1);
        });
    }
    while (observed.load() != count) std::this_thread::yield();
    started.release();
    check(source_for_waiters.succeed(), "settlement with waiters");
    for (auto& waiter : waiters) waiter.join();
    check(observed.load() == count * 2, "all concurrent waiters wake");
}

void test_competing_settlement() {
    CompletionSource source;
    auto completion = source.completion();
    std::atomic<int> winners{0};
    std::thread success([&] { if (source.succeed()) winners.fetch_add(1); });
    std::thread cancelled([&] { if (source.cancel()) winners.fetch_add(1); });
    success.join();
    cancelled.join();
    check(winners.load() == 1, "exactly one competing settlement wins");
    check(completion.status() == CompletionStatus::success ||
          completion.status() == CompletionStatus::cancelled, "winning outcome retained");
}

void test_resource_retention() {
    auto destroyed = std::make_shared<std::atomic<int>>(0);
    struct Token {
        explicit Token(std::shared_ptr<std::atomic<int>> value) : count(std::move(value)) {}
        Token(Token&& other) noexcept : count(std::move(other.count)) {}
        Token(const Token&) = delete;
        ~Token() noexcept { if (count) count->fetch_add(1); }
        std::shared_ptr<std::atomic<int>> count;
    };
    CompletionSource source(ResourceLease(Token{destroyed}));
    auto completion = source.completion();
    source = CompletionSource{};
    check(completion.wait().status == CompletionStatus::abandoned, "drop producer abandons");
    check(destroyed->load() == 0, "consumer retains resource after producer ends");
    completion = Completion{};
    check(destroyed->load() == 1, "last consumer releases resource");
}

void test_ledger_resource_retention_on_worker() {
    using namespace saga::memory;
    auto released = std::make_shared<std::atomic<int>>(0);
    AllocationRecord record;
    record.id = "tracked";
    record.owner = "completion-test";
    record.location = Location::host;
    record.logical_bytes = 32;
    AllocationRecord overhead;
    overhead.id = "runtime";
    overhead.owner = "completion-test";
    overhead.source = AllocationSource::runtime_overhead;
    overhead.overhead_bound = 64;
    StaticCandidate candidate("completion-test", GraphMode::disabled, {record, overhead});
    const Budgets limits{DomainBudget{4096, 4096, 0},
                         DomainBudget{4096, 4096, 0},
                         DomainBudget{4096, 4096, 0}};
    auto grant = candidate.grant(limits);
    auto allocation = grant.allocate("tracked", 32, [&](std::uint64_t cap) {
        auto deleter = [released](int* pointer) noexcept {
            delete pointer;
            released->fetch_add(1);
        };
        std::unique_ptr<int, decltype(deleter)> owner(new int(7), deleter);
        return MemoryBacking{OwnedBacking(std::move(owner)), cap};
    });
    ResourceLease retained = ResourceLease::retain(allocation);
    allocation = {};

    Gate gate;
    CompletionSource source(std::move(retained));
    Completion completion = source.completion();
    std::thread worker([&] {
        gate.wait();
        source.succeed();
    });

    completion = Completion{};
    check(grant.report().live_bytes[0] == 32 && released->load() == 0,
          "early consumer drop retains ledger allocation while worker owns producer");
    Completion result_consumer = source.completion();
    gate.release();
    worker.join();
    check(result_consumer.status() == CompletionStatus::success,
          "worker settles after deterministic gate release");
    check(grant.report().live_bytes[0] == 32 && released->load() == 0,
          "result consumer retains ledger allocation after worker completion");
    source = CompletionSource{};
    check(grant.report().live_bytes[0] == 32 && released->load() == 0,
          "result consumer retains allocation after source destruction");
    result_consumer = Completion{};
    check(grant.report().live_bytes[0] == 0 && released->load() == 1,
          "last result release frees backing and ledger accounting once");
}

void test_result_resource_and_allocation_bridge() {
    auto destroyed = std::make_shared<std::atomic<int>>(0);
    struct Token {
        explicit Token(std::shared_ptr<std::atomic<int>> value) : count(std::move(value)) {}
        Token(Token&& other) noexcept : count(std::move(other.count)) {}
        Token(const Token&) = delete;
        ~Token() noexcept { if (count) count->fetch_add(1); }
        std::shared_ptr<std::atomic<int>> count;
    };
    CompletionSource source(ResourceLease(Token{destroyed}));
    auto completion = source.completion();
    source.succeed();
    source = CompletionSource{};
    completion.result();
    check(destroyed->load() == 0, "successful result retains backing");
    completion = Completion{};
    check(destroyed->load() == 1, "successful backing released with last result consumer");

    saga::memory::AllocationLease allocation;
    ResourceLease lease = ResourceLease::retain(allocation);
    check(static_cast<bool>(lease), "allocation lease can be retained by value");
}

void test_move_assignment_abandons_old_and_wakes() {
    CompletionSource old_source;
    auto old_completion = old_source.completion();
    CompletionSource replacement;
    auto replacement_completion = replacement.completion();
    old_source = std::move(replacement);
    check(old_completion.wait().status == CompletionStatus::abandoned,
          "source move assignment abandons previous state");
    check(replacement_completion.status() == CompletionStatus::pending,
          "source move assignment transfers new pending state");
    old_source.succeed();
    check(replacement_completion.status() == CompletionStatus::success,
          "transferred state settles normally");
}

} // namespace

int main() {
    test_status_and_settle_once();
    test_failure_and_invalid();
    test_move_and_abandon_wakes();
    test_timeout_and_concurrent_waiters();
    test_competing_settlement();
    test_resource_retention();
    test_ledger_resource_retention_on_worker();
    test_result_resource_and_allocation_bridge();
    test_move_assignment_abandons_old_and_wakes();
    return 0;
}
