#include "saga/runtime/execution_worker.h"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace saga::runtime::detail {
struct Task {
    std::function<void(std::stop_token)> callback;
    CompletionSource source;
};
struct CleanupRecord {
    std::function<void()> callback;
    CompletionSource source;
    bool scheduled{false};
    bool completed{false};
    bool run_callback{true};
};
struct ExecutionWorkerState {
    ExecutionWorkerState(std::size_t tasks, std::size_t cleanups)
        : task_capacity(tasks), task_queue(tasks), cleanup_slots(cleanups) {}
    std::mutex mutex;
    std::condition_variable changed;
    std::mutex join_mutex;
    const std::size_t task_capacity;
    std::vector<std::unique_ptr<Task>> task_queue;
    std::vector<std::shared_ptr<CleanupRecord>> cleanup_slots;
    std::size_t task_head{0};
    std::size_t task_count{0};
    bool stopping{false};
    bool failed{false};
    bool joined{false};
    std::stop_source stop;
    std::thread::id worker_id;
};
} // namespace saga::runtime::detail

namespace saga::runtime {
namespace {
thread_local bool in_stop_notification = false;

void notify_stop(const std::shared_ptr<detail::ExecutionWorkerState>& state) noexcept {
    const bool prior = in_stop_notification;
    in_stop_notification = true;
    state->stop.request_stop();
    in_stop_notification = prior;
}

void close_admission(const std::shared_ptr<detail::ExecutionWorkerState>& state) noexcept {
    {
        std::lock_guard lock(state->mutex);
        state->stopping = true;
        for (auto& record : state->cleanup_slots)
            if (record && !record->scheduled) record->scheduled = true;
    }
    state->changed.notify_all();
    notify_stop(state);
}

void worker_loop(const std::shared_ptr<detail::ExecutionWorkerState>& state) noexcept {
    {
        std::lock_guard lock(state->mutex);
        state->worker_id = std::this_thread::get_id();
    }
    for (;;) {
        std::unique_ptr<detail::Task> task;
        std::shared_ptr<detail::CleanupRecord> cleanup;
        bool run_cleanup = true;
        bool cancel_task = false;
        {
            std::unique_lock lock(state->mutex);
            state->changed.wait(lock, [&] {
                if (state->stopping && state->task_count) return true;
                for (const auto& item : state->cleanup_slots)
                    if (item && item->scheduled && !item->completed) return true;
                if (state->task_count) return true;
                if (state->stopping) return true;
                return false;
            });
            if (state->stopping && state->task_count) {
                task = std::move(state->task_queue[state->task_head]);
                state->task_head = (state->task_head + 1) % state->task_capacity;
                --state->task_count;
                cancel_task = true;
            } else {
                for (auto& item : state->cleanup_slots) {
                    if (item && item->scheduled && !item->completed) {
                        cleanup = item;
                        run_cleanup = item->run_callback;
                        break;
                    }
                }
                if (!cleanup && state->task_count) {
                    task = std::move(state->task_queue[state->task_head]);
                    state->task_head = (state->task_head + 1) % state->task_capacity;
                    --state->task_count;
                }
                if (!cleanup && !task && state->stopping) {
                    state->worker_id = {};
                    lock.unlock();
                    state->changed.notify_all();
                    return;
                }
            }
        }
        if (task) {
            if (cancel_task || state->stop.stop_requested()) {
                task->callback = {};
                task->source.cancel();
            } else {
                try {
                    task->callback(state->stop.get_token());
                    task->callback = {};
                    task->source.succeed();
                } catch (...) {
                    const auto error = std::current_exception();
                    task->callback = {};
                    {
                        std::lock_guard lock(state->mutex);
                        state->failed = true;
                    }
                    close_admission(state);
                    task->source.fail(error);
                }
            }
        }
        if (cleanup) {
            std::exception_ptr error;
            if (run_cleanup) {
                try { cleanup->callback(); }
                catch (...) { error = std::current_exception(); }
            }
            cleanup->callback = {};
            if (error) {
                {
                    std::lock_guard lock(state->mutex);
                    state->failed = true;
                }
                close_admission(state);
            }
            {
                std::lock_guard lock(state->mutex);
                cleanup->completed = true;
                for (auto& slot : state->cleanup_slots)
                    if (slot == cleanup) { slot.reset(); break; }
            }
            if (error) cleanup->source.fail(error);
            else cleanup->source.succeed();
            state->changed.notify_all();
        }
    }
}
} // namespace

ExecutionWorker::ExecutionWorker(std::size_t task_capacity, std::size_t cleanup_capacity)
    : state_(nullptr) {
    if (!task_capacity || !cleanup_capacity) throw std::invalid_argument("worker capacities must be nonzero");
    state_ = std::make_shared<detail::ExecutionWorkerState>(task_capacity, cleanup_capacity);
    thread_ = std::thread([state = state_] { worker_loop(state); });
}
ExecutionWorker::~ExecutionWorker() noexcept {
    if (!shutdown()) std::terminate();
}

Submission ExecutionWorker::submit(std::function<void(std::stop_token)> callback, ResourceLease resource) {
    if (!callback) return {SubmissionStatus::failed, {}};
    auto task = std::make_unique<detail::Task>();
    task->callback = std::move(callback);
    task->source = CompletionSource(std::move(resource));
    auto completion = task->source.completion();
    {
        std::lock_guard lock(state_->mutex);
        if (state_->failed) return {SubmissionStatus::failed, {}};
        if (state_->stopping) return {SubmissionStatus::stopping, {}};
        if (state_->task_count == state_->task_capacity) return {SubmissionStatus::full, {}};
        const auto index = (state_->task_head + state_->task_count) % state_->task_capacity;
        state_->task_queue[index] = std::move(task);
        ++state_->task_count;
    }
    state_->changed.notify_one();
    return {SubmissionStatus::accepted, std::move(completion)};
}

ExecutionWorker::CleanupTicket ExecutionWorker::reserve_cleanup(std::function<void()> cleanup, ResourceLease resource) {
    if (!cleanup) return {};
    auto record = std::make_shared<detail::CleanupRecord>();
    record->callback = std::move(cleanup);
    record->source = CompletionSource(std::move(resource));
    std::lock_guard lock(state_->mutex);
    if (state_->stopping || state_->failed) return {};
    for (auto& slot : state_->cleanup_slots) {
        if (!slot) {
            slot = record;
            return CleanupTicket(state_, std::move(record));
        }
    }
    return {};
}

void ExecutionWorker::request_stop() noexcept { close_admission(state_); }
bool ExecutionWorker::healthy() const noexcept {
    std::lock_guard lock(state_->mutex);
    return !state_->failed;
}
bool ExecutionWorker::shutdown() noexcept {
    if (in_stop_notification) return false;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->worker_id == std::this_thread::get_id()) return false;
    }
    close_admission(state_);
    std::lock_guard join_lock(state_->join_mutex);
    if (state_->joined) return true;
    if (thread_.joinable()) thread_.join();
    state_->joined = true;
    return true;
}

ExecutionWorker::CleanupTicket::CleanupTicket(std::shared_ptr<detail::ExecutionWorkerState> state,
                                               std::shared_ptr<detail::CleanupRecord> record) noexcept
    : state_(std::move(state)), record_(std::move(record)) {}
ExecutionWorker::CleanupTicket::CleanupTicket(CleanupTicket&& other) noexcept
    : state_(std::move(other.state_)), record_(std::move(other.record_)) {}
ExecutionWorker::CleanupTicket& ExecutionWorker::CleanupTicket::operator=(CleanupTicket&& other) noexcept {
    if (this != &other) { schedule(); state_ = std::move(other.state_); record_ = std::move(other.record_); }
    return *this;
}
ExecutionWorker::CleanupTicket::~CleanupTicket() noexcept { schedule(); }
ExecutionWorker::CleanupTicket::operator bool() const noexcept { return static_cast<bool>(record_); }
void ExecutionWorker::CleanupTicket::schedule() noexcept {
    if (!record_) return;
    {
        std::lock_guard lock(state_->mutex);
        if (!record_->scheduled && !record_->completed) record_->scheduled = true;
    }
    state_->changed.notify_one();
    record_.reset();
    state_.reset();
}
Completion ExecutionWorker::CleanupTicket::submit() {
    if (!record_) return {};
    auto result = record_->source.completion();
    schedule();
    return result;
}
void ExecutionWorker::CleanupTicket::dismiss() noexcept {
    if (!record_) return;
    {
        std::lock_guard lock(state_->mutex);
        if (!record_->scheduled && !record_->completed) {
            record_->run_callback = false;
            record_->scheduled = true;
        }
    }
    state_->changed.notify_one();
    record_.reset();
    state_.reset();
}

} // namespace saga::runtime
