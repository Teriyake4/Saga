#include "saga/runtime/admission.h"
#include "saga/runtime/execution_worker.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace saga::runtime;
using Clock = AdmissionCoordinator::Clock;
void check(bool value, const char* message) { if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); } }
template<class F> void throws(F&& f, const char* message) {
    try { f(); } catch (const std::exception&) { return; }
    check(false, message);
}
Clock::time_point at(int seconds = 0) { return Clock::time_point{} + std::chrono::seconds(seconds); }
AdmissionResult add(AdmissionCoordinator& c, std::string_view v, int deadline = 20) {
    return c.admit(v, {}, at(deadline), std::chrono::seconds(5), at());
}
struct Gate {
    void wait_in() { std::unique_lock lock(m); in = true; cv.notify_all(); cv.wait(lock, [&]{ return open; }); }
    void entered() { std::unique_lock lock(m); cv.wait(lock, [&]{ return in; }); }
    void release() { { std::lock_guard lock(m); open = true; } cv.notify_all(); }
    std::mutex m; std::condition_variable cv; bool in{}, open{};
};
void test_ready_slot_fifo_queue_and_queue_zero() {
    AdmissionCoordinator c({1, 4, 16});
    auto a = add(c, "a"), b = add(c, "b");
    check(a.status == AdmissionStatus::accepted && b.status == AdmissionStatus::accepted, "ready slot plus one waiter");
    check(add(c, "c").status == AdmissionStatus::full, "bounded wait queue");
    auto first = c.take_next(at()); check(first.payload() == "a", "FIFO first");
    auto h = c.start(first, at());
    throws([&]{ (void)c.start(first, at()); }, "duplicate start rejected");
    CompletionSource f; c.retire(h, f.completion()); c.poll(at());
    check(c.has_active() && first.phase() == AdmissionPhase::active, "pending fence keeps active");
    f.succeed(); c.poll(at()); check(first.completion().status() == CompletionStatus::success, "success after terminal fence");
    auto second = c.take_next(at()); check(second.payload() == "b", "FIFO second");
    AdmissionCoordinator zero({0, 2, 16});
    check(add(zero, "idle").status == AdmissionStatus::accepted, "zero queue permits idle dispatch slot");
    check(add(zero, "wait").status == AdmissionStatus::full, "zero queue rejects waiting");
}
void test_rejections_empty_payload_and_quota_rollback() {
    AdmissionCoordinator c({0, 1, 4});
    check(c.admit(std::string_view{}, {}, at(), std::chrono::seconds(1), at()).status == AdmissionStatus::expired,
          "expired request rejected before reservation");
    check(c.admit("x", {}, at(5), Clock::duration::zero(), at()).status == AdmissionStatus::invalid_timeout,
          "nonpositive execution timeout rejected explicitly");
    check(c.admit("12345", {}, at(5), std::chrono::seconds(1), at()).status == AdmissionStatus::oversize,
          "oversize rejected before reservation");
    auto empty = c.admit(std::string_view{}, {}, at(5), std::chrono::seconds(1), at());
    check(empty.status == AdmissionStatus::accepted && empty.ticket.payload().empty(), "default empty view normalized safely");
    auto extra = c.admit("x", {}, at(5), std::chrono::seconds(1), at());
    check(extra.status == AdmissionStatus::full, "one ready slot counted despite zero wait capacity");
    auto ready = c.take_next(at());
    check(ready.payload().empty() && static_cast<bool>(ready.resource()), "read-only payload and producer lease retain normalized payload");
}
void test_deadline_saturation_edges() {

    AdmissionCoordinator c({1, 2, 8});
    const auto minimum = Clock::time_point::min();
    const auto maximum = Clock::time_point::max();
    auto near_max = c.admit("m", {}, maximum - Clock::duration(1), Clock::duration::max(),
                            maximum - Clock::duration(2));
    check(near_max.status == AdmissionStatus::accepted, "near-maximum injected time accepted");
    auto active = c.take_next(maximum - Clock::duration(2));
    auto id = c.start(active, maximum - Clock::duration(2));
    CompletionSource fence; c.retire(id, fence.completion());
    c.poll(maximum - Clock::duration(1));
    check(!active.stop_token().stop_requested(), "saturated deadline does not expire before maximum time");
    c.poll(maximum);
    check(active.stop_token().stop_requested() && c.has_active(), "saturated execution deadline expires at maximum time but keeps pending fence");
    fence.succeed(); c.poll(maximum);
    check(active.completion().status() == CompletionStatus::cancelled, "maximum time timeout waits for terminal fence");
    auto earliest = c.admit("e", {}, maximum, Clock::duration::max(), minimum);
    check(earliest.status == AdmissionStatus::accepted, "minimum time point does not overflow deadline arithmetic");
    auto early_active = c.take_next(minimum);
    auto early_id = c.start(early_active, minimum);
    CompletionSource early_fence; c.retire(early_id, early_fence.completion());
    c.poll(minimum + Clock::duration(1));
    check(!early_active.stop_token().stop_requested(), "maximum timeout from minimum start does not expire immediately");
    c.poll(Clock::time_point{});
    check(early_active.stop_token().stop_requested(), "maximum timeout from minimum start expires at saturated deadline");
    early_fence.succeed(); c.poll(Clock::time_point{});
}
void test_quota_completion_lifetime_normalization_and_oversize() {
    AdmissionCoordinator c({1, 1, 4});
    std::string oversized_capacity("abc"); oversized_capacity.reserve(1024 * 1024);
    auto r = c.admit(oversized_capacity, {}, at(5), std::chrono::seconds(1), at());
    check(r.status == AdmissionStatus::accepted, "capacity is normalized and charges size");
    auto ticket = r.ticket; auto active = c.take_next(at()); auto h = c.start(active, at());
    CompletionSource fence; c.retire(h, fence.completion()); fence.succeed(); c.poll(at());
    Completion held = ticket.completion(); ticket = {}; active = {}; r.ticket = {};
    check(add(c, "x").status == AdmissionStatus::full, "completion alone retains byte and request quota");
    held = Completion{};
    check(add(c, "x").status == AdmissionStatus::accepted, "last completion releases quota");
    check(c.admit("12345", {}, at(5), std::chrono::seconds(1), at()).status == AdmissionStatus::oversize, "oversize rejected");
}
void test_cancel_deadline_active_and_close() {
    AdmissionCoordinator c({2, 5, 16}); auto a = add(c, "a"), b = add(c, "b");
    check(b.ticket.cancel(), "cancel token requests stop");
    check(b.ticket.stop_token().stop_requested(), "persistent token observes cancellation");
    c.poll(at()); check(b.ticket.completion().status() == CompletionStatus::cancelled, "queued cancel settles immediately");
    auto active = c.take_next(at()); auto h = c.start(active, at());
    check(active.cancel(), "active cancel accepted"); c.poll(at());
    check(c.has_active() && active.completion().status() == CompletionStatus::pending, "active waits for fence");
    CompletionSource f; c.retire(h, f.completion()); f.succeed(); c.poll(at());
    check(active.completion().status() == CompletionStatus::cancelled, "active cancellation settles after fence");
    auto timed = add(c, "t"); auto timed_active = c.take_next(at()); auto timed_id = c.start(timed_active, at());
    std::atomic<bool> callback_reentered{false};
    std::stop_callback callback(timed_active.stop_token(), [&] {
        (void)timed_active.phase();
        (void)timed_active.cancel();
        callback_reentered = true;
    });
    CompletionSource timeout_fence; c.retire(timed_id, timeout_fence.completion());
    c.poll(at(6)); check(timed_active.stop_token().stop_requested() && c.has_active(), "execution timeout requests stop but retains slot");
    check(callback_reentered.load(), "stop callback can safely reenter ticket APIs");
    timeout_fence.succeed(); c.poll(at(6));
    check(timed.ticket.completion().status() == CompletionStatus::cancelled, "timeout settles only after fence");
    auto exp = c.admit("e", {}, at(1), std::chrono::seconds(1), at());
    c.poll(at(1)); check(exp.ticket.completion().status() == CompletionStatus::cancelled, "queue deadline uses injected clock");
    auto running = add(c, "r"); auto work = c.take_next(at()); auto id = c.start(work, at());
    c.close(); check(work.completion().status() == CompletionStatus::pending, "close preserves submitted work pending");
    CompletionSource g; c.retire(id, g.completion()); g.succeed(); c.poll(at());
    check(work.completion().status() == CompletionStatus::cancelled, "close requests active cancellation");
    check(add(c, "n").status == AdmissionStatus::stopping, "closed admission rejects");
    (void)a; (void)running;
}
void test_ids_foreign_invalid_and_quarantine() {
    AdmissionCoordinator a({1, 2, 8}), b({1, 2, 8}); auto ticket = add(a, "x").ticket;
    auto active = a.take_next(at()); auto id = a.start(active, at());
    throws([&]{ (void)b.start(active, at()); }, "foreign ticket rejected");
    throws([&]{ b.retire(id, Completion{}); }, "invalid foreign fence rejected");
    CompletionSource abandoned; Completion fence = abandoned.completion(); abandoned = CompletionSource{};
    a.retire(id, fence); a.poll(at());
    check(!a.healthy() && active.completion().status() == CompletionStatus::failure, "abandoned fence quarantines unhealthy");
    check(add(a, "z").status == AdmissionStatus::failed, "unhealthy rejects new work");
    throws([&]{ a.retire(id, fence); }, "stale active id rejected");
    AdmissionCoordinator cancelled({0, 1, 8});
    auto cancelled_ticket = add(cancelled, "c");
    auto cancelled_active = cancelled.take_next(at()); auto cancelled_id = cancelled.start(cancelled_active, at());
    CompletionSource cancelled_fence; cancelled.retire(cancelled_id, cancelled_fence.completion());
    cancelled_fence.cancel(); cancelled.poll(at());
    check(cancelled_ticket.ticket.completion().status() == CompletionStatus::cancelled,
          "cancelled physical fence never becomes request success");
    (void)ticket;
}
void test_settlement_requests_stop_callbacks() {
    {
        AdmissionCoordinator c({2, 4, 16});
        auto queued = add(c, "close");
        std::atomic<bool> reentered{false};
        std::stop_callback callback(queued.ticket.stop_token(), [&] {
            (void)queued.ticket.phase();
            (void)queued.ticket.payload();
            reentered.store(true);
        });
        c.close();
        check(reentered.load() && queued.ticket.stop_token().stop_requested(),
              "close settlement requests stop and runs reentrant ticket callback");
        check(queued.ticket.completion().status() == CompletionStatus::cancelled,
              "close settles queued ticket cancelled");
    }
    {
        AdmissionCoordinator c({2, 4, 16});
        auto active_ticket = add(c, "active");
        auto queued = add(c, "queued");
        auto active = c.take_next(at()); auto id = c.start(active, at());
        std::atomic<bool> reentered{false};
        std::stop_callback callback(queued.ticket.stop_token(), [&] {
            (void)queued.ticket.phase();
            (void)queued.ticket.completion();
            reentered.store(true);
        });
        CompletionSource failed; c.retire(id, failed.completion()); failed.fail(); c.poll(at());
        check(reentered.load() && queued.ticket.stop_token().stop_requested(),
              "failed fence cancellation requests queued stop source");
        check(queued.ticket.completion().status() == CompletionStatus::cancelled,
              "failed active fence settles waiting requests cancelled");
        check(active_ticket.ticket.completion().status() == CompletionStatus::failure,
              "failed physical fence fails active request");
    }
    {
        AdmissionCoordinator c({2, 4, 16});
        auto polled = c.admit("poll", {}, at(1), std::chrono::seconds(1), at());
        std::atomic<bool> poll_callback{false};
        std::stop_callback poll_stop(polled.ticket.stop_token(), [&] {
            (void)polled.ticket.phase(); poll_callback.store(true);
        });
        c.poll(at(1));
        check(poll_callback.load() && polled.ticket.completion().status() == CompletionStatus::cancelled,
              "queue deadline poll cancels and notifies ticket");

        auto taken = c.admit("take", {}, at(1), std::chrono::seconds(1), at());
        std::atomic<bool> take_callback{false};
        std::stop_callback take_stop(taken.ticket.stop_token(), [&] {
            (void)taken.ticket.payload(); take_callback.store(true);
        });
        check(!c.take_next(at(1)).valid(), "take_next skips expired queued work");
        check(take_callback.load() && taken.ticket.completion().status() == CompletionStatus::cancelled,
              "take_next deadline cancellation requests stop and settles");
    }
}
void test_cross_thread_cancel_and_execution_worker_lease() {
    AdmissionCoordinator c({1, 2, 8}); auto result = add(c, "payload"); auto active = c.take_next(at()); auto id = c.start(active, at());
    CompletionSource f; c.retire(id, f.completion());
    std::thread cancel([ticket = active]() mutable { check(ticket.cancel(), "concurrent cancel"); });
    cancel.join(); c.poll(at()); check(c.has_active(), "cancel does not clear pending fence"); f.succeed(); c.poll(at());

    AdmissionTicket retained;
    Completion completion;
    Gate gate;
    {
        auto owner = std::make_unique<AdmissionCoordinator>(AdmissionLimits{1, 1, 32});
        auto admitted = add(*owner, "worker"); retained = admitted.ticket;
        auto item = owner->take_next(at()); auto handle = owner->start(item, at());
        ExecutionWorker worker(1, 1);
        auto submission = worker.submit([&](std::stop_token token) { check(token.stop_possible(), "worker stop token"); gate.wait_in(); }, item.resource());
        owner->retire(handle, submission.completion); completion = admitted.ticket.completion();
        gate.entered(); owner.reset();
        check(completion.status() == CompletionStatus::abandoned, "coordinator destruction abandons notification, not success");
        check(add(c, "x").status == AdmissionStatus::accepted, "unrelated quota remains available");
        gate.release(); worker.shutdown();
    }
    retained = {};
}
void test_owner_thread() {
    AdmissionCoordinator c({1, 1, 4}); bool query_rejected = false, close_rejected = false;
    std::thread t([&]{
        try { (void)c.queued(); } catch (const std::logic_error&) { query_rejected = true; }
        try { c.close(); } catch (const std::logic_error&) { close_rejected = true; }
    });
    t.join(); check(query_rejected && close_rejected && c.healthy(), "owner operations enforce constructing thread without mutation");
}
}
int main() {
    test_ready_slot_fifo_queue_and_queue_zero();
    test_rejections_empty_payload_and_quota_rollback();
    test_deadline_saturation_edges();
    test_quota_completion_lifetime_normalization_and_oversize();
    test_cancel_deadline_active_and_close();
    test_ids_foreign_invalid_and_quarantine();
    test_settlement_requests_stop_callbacks();
    test_cross_thread_cancel_and_execution_worker_lease();
    test_owner_thread();
}
