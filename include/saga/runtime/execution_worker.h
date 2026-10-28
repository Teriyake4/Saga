#pragma once

#include "saga/runtime/completion.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <stop_token>
#include <thread>

namespace saga::runtime {

enum class SubmissionStatus { accepted, full, stopping, failed };
struct Submission {
    SubmissionStatus status{SubmissionStatus::failed};
    Completion completion;
};

namespace detail { struct ExecutionWorkerState; struct CleanupRecord; }

// One native worker executes bounded synchronous units. Callbacks must fence all external work
// before returning; this class contains no device-specific behavior. shutdown() is external-owner
// only and joins; callbacks may request_stop/query state but must never wait for shutdown().
// Destroying the facade from its own callback is unsupported. Concurrent shutdown calls serialize.
// Callback exceptions conservatively fail-stop the worker because external state may be unknown.
// Cleanup reservations are bounded records allocated at reservation time. Shutdown schedules even
// tickets still held by callers; their Completion remains usable after shutdown and wrapper destruction.
// dismiss() schedules a worker-thread no-op release, not the rollback callback.
class ExecutionWorker {
public:
    class CleanupTicket {
    public:
        CleanupTicket() noexcept = default;
        CleanupTicket(const CleanupTicket&) = delete;
        CleanupTicket& operator=(const CleanupTicket&) = delete;
        CleanupTicket(CleanupTicket&&) noexcept;
        CleanupTicket& operator=(CleanupTicket&&) noexcept;
        ~CleanupTicket() noexcept;
        // Enqueue the reserved cleanup and return its completion. Also valid after shutdown; by then
        // shutdown has scheduled and settled the reservation.
        Completion submit();
        // Explicitly schedule a no-op release on the worker.
        void dismiss() noexcept;
        explicit operator bool() const noexcept;
    private:
        friend class ExecutionWorker;
        CleanupTicket(std::shared_ptr<detail::ExecutionWorkerState>,
                      std::shared_ptr<detail::CleanupRecord>) noexcept;
        void schedule() noexcept;
        std::shared_ptr<detail::ExecutionWorkerState> state_;
        std::shared_ptr<detail::CleanupRecord> record_;
    };

    ExecutionWorker(std::size_t task_capacity, std::size_t cleanup_capacity);
    ExecutionWorker(const ExecutionWorker&) = delete;
    ExecutionWorker& operator=(const ExecutionWorker&) = delete;
    ~ExecutionWorker() noexcept;

    Submission submit(std::function<void(std::stop_token)> callback, ResourceLease resource = {});
    // Reserve cleanup callback, its retained resource, and its completion before starting work.
    // Returns empty when the bounded reservation pool is full or admission has closed.
    CleanupTicket reserve_cleanup(std::function<void()> cleanup, ResourceLease resource = {});
    void request_stop() noexcept;
    bool healthy() const noexcept;
    // Returns false without blocking on the worker thread or from a stop notification callback;
    // returns true after an external caller has joined (including repeated calls).
    bool shutdown() noexcept;

private:
    std::shared_ptr<detail::ExecutionWorkerState> state_;
    std::thread thread_;
};

} // namespace saga::runtime
