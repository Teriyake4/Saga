#include "saga/runtime/completion.h"

#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace saga::runtime::detail {
struct CompletionState {
    explicit CompletionState(ResourceLease retained) : resource(std::move(retained)) {}

    mutable std::mutex mutex;
    std::condition_variable changed;
    Outcome outcome;
    ResourceLease resource;
};
} // namespace saga::runtime::detail

namespace saga::runtime {
namespace {

std::shared_ptr<detail::CompletionState> require_state(
    const std::shared_ptr<detail::CompletionState>& state) {
    if (!state) throw std::logic_error("invalid completion handle");
    return state;
}

Outcome snapshot(const std::shared_ptr<detail::CompletionState>& state) {
    std::lock_guard lock(state->mutex);
    return state->outcome;
}

} // namespace

Completion::Completion(std::shared_ptr<detail::CompletionState> state) noexcept
    : state_(std::move(state)) {}

CompletionStatus Completion::status() const {
    return snapshot(require_state(state_)).status;
}

bool Completion::is_ready() const { return status() != CompletionStatus::pending; }

Outcome Completion::wait() const {
    auto state = require_state(state_);
    std::unique_lock lock(state->mutex);
    state->changed.wait(lock, [&] { return state->outcome.status != CompletionStatus::pending; });
    return state->outcome;
}

Outcome Completion::wait_until(std::chrono::steady_clock::time_point deadline) const {
    auto state = require_state(state_);
    std::unique_lock lock(state->mutex);
    state->changed.wait_until(lock, deadline,
                              [&] { return state->outcome.status != CompletionStatus::pending; });
    return state->outcome;
}

Outcome Completion::wait_for(std::chrono::steady_clock::duration timeout) const {
    if (timeout <= std::chrono::steady_clock::duration::zero()) return snapshot(require_state(state_));

    const auto now = std::chrono::steady_clock::now();
    const auto latest = std::chrono::steady_clock::time_point::max();
    if (now > latest - timeout) return wait_until(latest);
    return wait_until(now + timeout);
}

void Completion::result() const {
    const Outcome value = wait();
    switch (value.status) {
    case CompletionStatus::success:
        return;
    case CompletionStatus::failure:
        if (value.error) std::rethrow_exception(value.error);
        throw std::runtime_error("completion failed without an exception");
    case CompletionStatus::cancelled:
        throw std::runtime_error("completion was cancelled");
    case CompletionStatus::abandoned:
        throw std::runtime_error("completion producer was abandoned");
    case CompletionStatus::pending:
        throw std::logic_error("pending completion returned from wait");
    }
    throw std::logic_error("invalid completion outcome");
}

CompletionSource::CompletionSource() : CompletionSource(ResourceLease{}) {}

CompletionSource::CompletionSource(ResourceLease resource)
    : state_(std::make_shared<detail::CompletionState>(std::move(resource))) {}

CompletionSource::CompletionSource(CompletionSource&& other) noexcept
    : state_(std::move(other.state_)) {}

CompletionSource& CompletionSource::operator=(CompletionSource&& other) noexcept {
    if (this != &other) {
        abandon();
        state_ = std::move(other.state_);
    }
    return *this;
}

CompletionSource::~CompletionSource() noexcept { abandon(); }

Completion CompletionSource::completion() const {
    if (!state_) throw std::logic_error("invalid completion source");
    return Completion(state_);
}

bool CompletionSource::settle(CompletionStatus status, std::exception_ptr error) noexcept {
    if (!state_ || (status != CompletionStatus::success &&
                    status != CompletionStatus::failure &&
                    status != CompletionStatus::cancelled)) return false;
    if (status != CompletionStatus::failure && error) return false;
    auto state = state_;
    {
        std::lock_guard lock(state->mutex);
        if (state->outcome.status != CompletionStatus::pending) return false;
        state->outcome = Outcome{status, std::move(error)};
    }
    state->changed.notify_all();
    return true;
}

void CompletionSource::abandon() noexcept {
    auto state = std::move(state_);
    if (!state) return;
    {
        std::lock_guard lock(state->mutex);
        if (state->outcome.status != CompletionStatus::pending) return;
        state->outcome = Outcome{CompletionStatus::abandoned, {}};
    }
    state->changed.notify_all();
}

} // namespace saga::runtime
