#include "saga/runtime/admission.h"
#include "saga/runtime/execution_worker.h"
#include "saga/runtime/output_buffer.h"
#include "saga/runtime/sequence_slots.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <utility>

namespace {
using namespace saga::runtime;
using Clock = AdmissionCoordinator::Clock;

void check(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

Clock::time_point at(int seconds = 0) {
    return Clock::time_point{} + std::chrono::seconds(seconds);
}

struct Gate {
    void arrive_and_wait() {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        check(changed.wait_for(lock, std::chrono::seconds(5), [&] { return open; }),
              "worker callback gate released within bound");
    }

    bool wait_until_entered() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
    }

    void release() {
        {
            std::lock_guard lock(mutex);
            open = true;
        }
        changed.notify_all();
    }

    std::mutex mutex;
    std::condition_variable changed;
    bool entered{};
    bool open{};
};

struct RetainedProbe {
    explicit RetainedProbe(std::shared_ptr<std::atomic<int>> count) : count(std::move(count)) {}
    RetainedProbe(RetainedProbe&&) noexcept = default;
    RetainedProbe(const RetainedProbe&) = delete;
    ~RetainedProbe() noexcept { if (count) count->fetch_add(1); }
    std::shared_ptr<std::atomic<int>> count;
};

void test_worker_cleanup_fence_controls_sequence_reuse() {
    ExecutionWorker worker(1, 1);
    SequenceSlots slots(1);
    Gate work_gate;
    std::atomic<bool> cleanup_ran{false};

    auto work = worker.submit([&](std::stop_token) { work_gate.arrive_and_wait(); });
    check(work.status == SubmissionStatus::accepted, "gated work accepted");
    check(work_gate.wait_until_entered(), "worker entered gated work within bound");

    auto handle = slots.acquire();
    auto retirement = slots.begin_release(handle);
    auto cleanup = worker.reserve_cleanup([&] { cleanup_ran.store(true); });
    check(static_cast<bool>(cleanup), "reserved cleanup ticket acquired");
    Completion physical_cleanup = cleanup.submit();

    check(slots.finish_release(std::move(retirement), physical_cleanup) == RetirementResult::pending,
          "sequence remains retiring while worker cleanup is fenced");
    check(slots.counts().retiring == 1 && slots.counts().free == 0,
          "pending physical cleanup does not free sequence slot");
    bool unavailable = false;
    try {
        (void)slots.acquire();
    } catch (const std::runtime_error&) {
        unavailable = true;
    }
    check(unavailable, "retiring sequence slot cannot be reused");
    check(!cleanup_ran.load(), "reserved cleanup cannot run ahead of gated work");

    work_gate.release();
    check(work.completion.wait_for(std::chrono::seconds(5)).status == CompletionStatus::success,
          "gated worker operation completes");
    check(physical_cleanup.wait_for(std::chrono::seconds(5)).status == CompletionStatus::success,
          "reserved physical cleanup completes successfully");
    check(cleanup_ran.load(), "physical cleanup callback ran");
    check(slots.finish_release(std::move(retirement), physical_cleanup) == RetirementResult::reusable,
          "successful physical cleanup permits retirement completion");
    const auto next = slots.acquire();
    check(next.slot == handle.slot && next.generation > handle.generation,
          "reused slot carries a newer generation");
    check(worker.shutdown(), "worker shutdown joins after cleanup completion");
}

void test_admission_worker_output_fence_backpressure_and_close() {
    AdmissionCoordinator admission({1, 2, 16});
    OutputBuffer output(4, 2);
    Gate worker_gate;
    auto destroyed = std::make_shared<std::atomic<int>>(0);
    auto admitted = admission.admit("request", ResourceLease(RetainedProbe(destroyed)),
                                    at(20), std::chrono::seconds(5), at());
    check(admitted.status == AdmissionStatus::accepted, "request admitted");
    AdmissionTicket ticket = admitted.ticket;
    auto active = admission.take_next(at());
    const auto active_id = admission.start(active, at());

    ExecutionWorker worker(1, 1);
    auto submitted = worker.submit([&](std::stop_token) { worker_gate.arrive_and_wait(); }, active.resource());
    check(submitted.status == SubmissionStatus::accepted, "admitted work submitted to worker");
    admission.retire(active_id, submitted.completion);
    check(worker_gate.wait_until_entered(), "worker is inside physical operation");

    check(output.try_publish_committed("ab", submitted.completion) == OutputPublishResult::commit_not_ready,
          "pending worker fence cannot publish bytes");
    check(active.cancel(), "active cancellation is requested");
    admission.close();
    admission.poll(at());
    check(active.completion().status() == CompletionStatus::pending && admission.has_active(),
          "active cancellation and close do not settle before the worker fence");
    check(ticket.phase() == AdmissionPhase::active && destroyed->load() == 0,
          "active request and retained resource survive pending physical work");

    worker_gate.release();
    check(submitted.completion.wait_for(std::chrono::seconds(5)).status == CompletionStatus::success,
          "worker completion reports physical fence success");
    check(output.try_publish_committed("ab", submitted.completion) == OutputPublishResult::published,
          "successful worker fence authorizes committed bytes");
    check(output.try_publish_committed("cd", submitted.completion) == OutputPublishResult::published,
          "second committed chunk fills bounded output");
    check(output.try_publish_committed("e", submitted.completion) == OutputPublishResult::backpressure,
          "bounded output applies backpressure");

    admission.poll(at());
    check(active.completion().status() == CompletionStatus::cancelled && !admission.has_active(),
          "cancelled request settles only after successful worker fence");
    check(destroyed->load() == 0, "request resource remains retained by live ticket/completion handles");
    check(output.close(OutputTerminal::cancelled), "terminal close succeeds after production ends");
    check(output.try_read().bytes == "ab", "first queued bytes drain after terminal close");
    check(output.try_read().bytes == "cd", "second queued bytes drain before terminal close");
    const auto end = output.try_read();
    check(end.status == OutputReadStatus::closed && output.terminal() == OutputTerminal::cancelled,
          "terminal close is observed only after queued bytes drain");
    check(output.try_publish_committed("x", submitted.completion) == OutputPublishResult::closed,
          "closed output rejects later publication");

    check(worker.shutdown(), "worker shutdown joins after request fence");
    check(destroyed->load() == 0, "shutdown does not release resource retained by request handles");
    submitted.completion = Completion{};
    active = AdmissionTicket{};
    ticket = AdmissionTicket{};
    admitted.ticket = AdmissionTicket{};
    check(destroyed->load() == 1, "retained request resource releases after final request handles drop");
}
}

int main() {
    test_worker_cleanup_fence_controls_sequence_reuse();
    test_admission_worker_output_fence_backpressure_and_close();
}
