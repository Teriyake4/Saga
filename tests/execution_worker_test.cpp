#include "saga/runtime/execution_worker.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace saga::runtime;
namespace {
void check(bool ok, const char* message) {
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
struct Gate {
    void arrive_and_wait() {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return open; });
    }
    void await_entered() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return entered; });
    }
    void release() { { std::lock_guard lock(mutex); open = true; } changed.notify_all(); }
    std::mutex mutex;
    std::condition_variable changed;
    bool entered{false};
    bool open{false};
};
struct Counted {
    explicit Counted(std::shared_ptr<std::atomic<int>> count) : count(std::move(count)) {}
    Counted(Counted&&) noexcept = default;
    Counted(const Counted&) = delete;
    ~Counted() noexcept { if (count) ++*count; }
    std::shared_ptr<std::atomic<int>> count;
};

void test_cleanup_only_and_slot_reuse() {
    ExecutionWorker worker(1, 1);
    std::atomic<int> ran{0};
    auto old = worker.reserve_cleanup([&] { ++ran; });
    auto completion = old.submit();
    check(completion.wait().status == CompletionStatus::success, "cleanup-only queue runs without dummy task");
    auto newer = worker.reserve_cleanup([&] { ++ran; });
    check(static_cast<bool>(newer), "completed cleanup slot can be reserved again");
    // The old ticket was consumed, and its record identity cannot schedule the new slot.
    worker.shutdown();
    check(ran.load() == 2, "both reservation callbacks execute on worker");
}

void test_fifo_full_cancel_and_lease() {
    ExecutionWorker worker(1, 4);
    Gate gate;
    std::atomic<int> order{0};
    auto releases = std::make_shared<std::atomic<int>>(0);
    auto first = worker.submit([&](std::stop_token token) {
        check(token.stop_possible(), "running callback receives token");
        gate.arrive_and_wait();
        order = 1;
    }, ResourceLease(Counted(releases)));
    check(first.status == SubmissionStatus::accepted, "first accepted");
    gate.await_entered();
    auto queued = worker.submit([&](std::stop_token) { check(order.load() == 1, "FIFO ordering"); order = 2; });
    check(queued.status == SubmissionStatus::accepted, "queued task accepted");
    check(worker.submit([](std::stop_token) {}).status == SubmissionStatus::full, "full explicit");
    worker.request_stop();
    check(first.completion.status() == CompletionStatus::pending, "running completion remains pending on stop");
    check(releases->load() == 0, "running task lease retained");
    gate.release();
    worker.shutdown();
    check(first.completion.status() == CompletionStatus::success, "fenced running task settles success");
    check(queued.completion.status() == CompletionStatus::cancelled, "queued task cancelled");
    check(order.load() == 1, "queued task not run after stop");
}

void test_cleanup_capacity_dismiss_move_and_reuse() {
    ExecutionWorker worker(1, 2);
    Gate gate;
    std::vector<std::thread::id> owners;
    std::thread::id worker_thread;
    std::atomic<int> sequence{0}, rollback_count{0};
    auto running = worker.submit([&](std::stop_token) { worker_thread = std::this_thread::get_id(); gate.arrive_and_wait(); });
    gate.await_entered();
    auto one = worker.reserve_cleanup([&] { owners.push_back(std::this_thread::get_id()); });
    auto two = worker.reserve_cleanup([&] {
        owners.push_back(std::this_thread::get_id());
        ++rollback_count;
        sequence.store(1);
    });
    check(one && two, "cleanup slots reserve independently");
    check(!worker.reserve_cleanup([] {}), "cleanup pool full explicit");
    auto queued = worker.submit([&](std::stop_token) {
        check(sequence.load() == 1, "scheduled cleanup precedes next ordinary callback");
        sequence.store(2);
    });
    check(queued.status == SubmissionStatus::accepted, "ordinary task queues behind running task");
    one.dismiss();
    check(running.completion.status() == CompletionStatus::pending, "cleanup waits behind running unit");
    // Move assignment schedules displaced rollback before releasing the running task.
    two = std::move(one);
    gate.release();
    check(running.completion.wait().status == CompletionStatus::success, "running unit done");
    check(queued.completion.wait().status == CompletionStatus::success, "queued ordinary callback runs");
    worker.shutdown();
    check(rollback_count.load() == 1 && owners.size() == 1, "dismiss is no-op; displaced rollback runs exactly once");
    check(owners.front() == worker_thread, "rollback capture executes and is released on worker");
    check(sequence.load() == 2, "scheduled cleanup runs before next ordinary task");
}

void test_autodestructor_while_task_queue_full_and_held_ticket() {
    auto worker = std::make_unique<ExecutionWorker>(1, 1);
    Gate gate;
    std::atomic<int> cleaned{0};
    auto running = worker->submit([&](std::stop_token) { gate.arrive_and_wait(); });
    gate.await_entered();
    auto queued = worker->submit([](std::stop_token) {});
    check(queued.status == SubmissionStatus::accepted, "ordinary queue full");
    auto ticket = worker->reserve_cleanup([&] { ++cleaned; });
    check(static_cast<bool>(ticket), "cleanup reserved before work concludes");
    worker->request_stop();
    gate.release();
    worker.reset();
    check(cleaned.load() == 1, "destructor performs cleanup despite full task queue and held ticket");
    auto completion = ticket.submit();
    check(completion.valid() && completion.status() == CompletionStatus::success,
          "held ticket completion survives facade destruction");
    check(queued.completion.status() == CompletionStatus::cancelled, "shutdown cancels pending task");
}

void test_failure_and_reentrant_stop_notification() {
    {
        ExecutionWorker worker(3, 2);
        Gate entered;
        std::atomic<int> cleanup{0};
        auto running = worker.submit([&](std::stop_token) { entered.arrive_and_wait(); });
        entered.await_entered();
        auto reservation = worker.reserve_cleanup([&] { ++cleanup; });
        auto queued = worker.submit([](std::stop_token) {});
        auto failure = worker.submit([](std::stop_token) { throw std::runtime_error("ordinary failure"); });
        entered.release();
        check(failure.completion.wait().status == CompletionStatus::failure, "callback exception fails completion");
        worker.shutdown();
        check(!worker.healthy(), "ordinary error latches failed state");
        check(queued.completion.status() == CompletionStatus::success, "FIFO task preceding failure runs");
        check(cleanup.load() == 1, "failure drains reserved cleanup");
        check(running.completion.status() == CompletionStatus::success, "running callback settles after fence");
        (void)reservation;
    }
    {
        ExecutionWorker worker(1, 1);
        Gate registered;
        std::atomic<int> notified{0};
        auto running = worker.submit([&](std::stop_token token) {
            std::stop_callback callback(token, [&] {
                check(worker.healthy(), "stop callback can query health");
                worker.request_stop();
                check(!worker.shutdown(), "shutdown refuses stop-notification self-wait");
                ++notified;
            });
            registered.arrive_and_wait();
        });
        registered.await_entered();
        std::thread stopper([&] { worker.request_stop(); });
        stopper.join();
        registered.release();
        worker.shutdown();
        check(notified.load() == 1, "reentrant stop callback runs without state-lock deadlock");
        check(running.completion.status() == CompletionStatus::success, "running callback fenced before settlement");
    }
}

void test_resource_destructor_reenters_worker() {
    ExecutionWorker worker(1, 1);
    Gate gate;
    std::atomic<bool> released{false};
    struct Probe {
        Probe(ExecutionWorker* owner, std::atomic<bool>* flag) : worker(owner), released(flag) {}
        Probe(const Probe&) = delete;
        Probe(Probe&& other) noexcept : worker(other.worker), released(other.released) {
            other.worker = nullptr;
            other.released = nullptr;
        }
        ~Probe() noexcept {
            if (!worker) return;
            (void)worker->healthy();
            worker->request_stop();
            released->store(true);
        }
        ExecutionWorker* worker;
        std::atomic<bool>* released;
    };
    auto submission = worker.submit([&](std::stop_token) { gate.arrive_and_wait(); },
                                    ResourceLease(Probe{&worker, &released}));
    check(submission.status == SubmissionStatus::accepted, "reentrant-resource task accepted");
    gate.await_entered();
    auto completion = std::move(submission.completion);
    completion = Completion{};
    gate.release();
    worker.shutdown();
    check(released.load(), "resource destructor reenters after task fence without worker-lock deadlock");
}

void test_cleanup_failure_and_self_join_and_concurrent_shutdown() {
    ExecutionWorker worker(2, 2);
    std::atomic<bool> self_rejected{false};
    auto self = worker.submit([&](std::stop_token) {
        self_rejected = !worker.shutdown();
    });
    auto bad_cleanup = worker.reserve_cleanup([] { throw std::runtime_error("cleanup failed"); });
    check(static_cast<bool>(bad_cleanup), "cleanup reservation accepted");
    check(self.completion.wait().status == CompletionStatus::success, "self-join rejection doesn't fail task");
    std::thread first([&] { worker.shutdown(); });
    std::thread second([&] { worker.shutdown(); });
    first.join();
    second.join();
    check(self_rejected.load(), "worker self-join rejected");
    check(!worker.healthy(), "cleanup exception latches unhealthy");
}
}

int main() {
    test_cleanup_only_and_slot_reuse();
    test_fifo_full_cancel_and_lease();
    test_cleanup_capacity_dismiss_move_and_reuse();
    test_autodestructor_while_task_queue_full_and_held_ticket();
    test_failure_and_reentrant_stop_notification();
    test_resource_destructor_reenters_worker();
    test_cleanup_failure_and_self_join_and_concurrent_shutdown();
}
