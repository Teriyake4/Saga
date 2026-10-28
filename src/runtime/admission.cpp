#include "saga/runtime/admission.h"

#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace saga::runtime::detail {
struct BudgetState {
    explicit BudgetState(AdmissionLimits value) : limits(value) {}
    std::mutex mutex;
    AdmissionLimits limits;
    std::size_t requests{};
    std::size_t bytes{};
};
struct PayloadResources {
    PayloadResources(std::string value, ResourceLease caller, std::shared_ptr<BudgetState> budget_state)
        : payload(std::move(value)), resource(std::move(caller)), budget(std::move(budget_state)), charged(payload.size()) {}
    ~PayloadResources() {
        std::lock_guard lock(budget->mutex);
        --budget->requests;
        budget->bytes -= charged;
    }
    const std::string payload;
    const ResourceLease resource;
    const std::shared_ptr<BudgetState> budget;
    const std::size_t charged;
};
struct AdmissionTicketState {
    std::atomic<AdmissionPhase> phase{AdmissionPhase::queued};
    std::stop_source stop;
    Completion completion;
    std::shared_ptr<const PayloadResources> payload;
};
struct AdmissionRecord {
    explicit AdmissionRecord(std::shared_ptr<AdmissionTicketState> value)
        : ticket(std::move(value)), producer(ResourceLease(ticket->payload)) {}
    std::shared_ptr<AdmissionTicketState> ticket;
    CompletionSource producer;
    std::chrono::steady_clock::time_point deadline{};
    std::chrono::steady_clock::duration timeout{};
    std::chrono::steady_clock::time_point started{};
    Completion fence;
    std::uint64_t generation{};
    bool started_work{};
};
struct AdmissionState {
    AdmissionState(AdmissionLimits l, std::uint64_t identity)
        : limits(l), budget(std::make_shared<BudgetState>(l)), pool_id(identity), owner(std::this_thread::get_id()) {}
    AdmissionLimits limits;
    std::shared_ptr<BudgetState> budget;
    const std::uint64_t pool_id;
    const std::thread::id owner;
    std::deque<std::shared_ptr<AdmissionRecord>> queue;
    std::shared_ptr<AdmissionRecord> active;
    std::uint64_t generation{};
    bool closed{}, healthy{true};
    std::atomic<bool> healthy_snapshot{true};
    std::atomic<std::size_t> queued_snapshot{};
    std::atomic<bool> active_snapshot{};
};
}

namespace saga::runtime {
namespace {
std::atomic<std::uint64_t> next_pool_id{1};
std::uint64_t allocate_pool_id() {
    auto value = next_pool_id.load(std::memory_order_relaxed);
    for (;;) {
        if (!value || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("admission pool identity exhausted");
        if (next_pool_id.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
}
void require_owner(const std::shared_ptr<detail::AdmissionState>& s) {
    if (std::this_thread::get_id() != s->owner) throw std::logic_error("admission owner thread required");
}
void phase_settle(const std::shared_ptr<detail::AdmissionRecord>& r, CompletionStatus status) {
    if (status == CompletionStatus::cancelled) r->ticket->stop.request_stop();
    r->ticket->phase.store(AdmissionPhase::settled, std::memory_order_release);
    r->producer.settle(status);
}
AdmissionCoordinator::Clock::time_point saturated_deadline(
    AdmissionCoordinator::Clock::time_point start, AdmissionCoordinator::Clock::duration duration) {
    const auto latest_start = AdmissionCoordinator::Clock::time_point::max() - duration;
    return start > latest_start ? AdmissionCoordinator::Clock::time_point::max() : start + duration;
}
}

AdmissionTicket::AdmissionTicket(std::shared_ptr<detail::AdmissionTicketState> s) noexcept : state_(std::move(s)) {}
bool AdmissionTicket::valid() const noexcept { return static_cast<bool>(state_); }
bool AdmissionTicket::cancel() const noexcept {
    return state_ && state_->phase.load(std::memory_order_acquire) != AdmissionPhase::settled && state_->stop.request_stop();
}
AdmissionPhase AdmissionTicket::phase() const {
    if (!state_) throw std::logic_error("invalid admission ticket");
    return state_->phase.load(std::memory_order_acquire);
}
Completion AdmissionTicket::completion() const {
    if (!state_) throw std::logic_error("invalid admission ticket");
    return state_->completion;
}
std::stop_token AdmissionTicket::stop_token() const noexcept { return state_ ? state_->stop.get_token() : std::stop_token{}; }
std::string_view AdmissionTicket::payload() const {
    if (!state_) throw std::logic_error("invalid admission ticket");
    return state_->payload->payload;
}
ResourceLease AdmissionTicket::resource() const {
    if (!state_) throw std::logic_error("invalid admission ticket");
    return ResourceLease(state_->payload);
}

AdmissionCoordinator::AdmissionCoordinator(AdmissionLimits limits) {
    if (!limits.outstanding_capacity || !limits.byte_capacity) throw std::invalid_argument("zero admission budget");
    state_ = std::make_shared<detail::AdmissionState>(limits, allocate_pool_id());
}
AdmissionCoordinator::~AdmissionCoordinator() { close(); }
AdmissionResult AdmissionCoordinator::admit(std::string_view payload, ResourceLease resource,
    Clock::time_point queue_deadline, Clock::duration execution_timeout, Clock::time_point now) {
    auto s = state_; require_owner(s);
    if (payload.size() > s->limits.byte_capacity) return {AdmissionStatus::oversize, {}};
    if (s->closed) return {AdmissionStatus::stopping, {}};
    if (!s->healthy) return {AdmissionStatus::failed, {}};
    if (execution_timeout <= Clock::duration::zero()) return {AdmissionStatus::invalid_timeout, {}};
    if (queue_deadline <= now) return {AdmissionStatus::expired, {}};
    {
        std::lock_guard lock(s->budget->mutex);
        if (s->budget->requests >= s->limits.outstanding_capacity ||
            payload.size() > s->limits.byte_capacity - s->budget->bytes)
            return {AdmissionStatus::full, {}};
        ++s->budget->requests;
        s->budget->bytes += payload.size();
    }
    struct Rollback {
        std::shared_ptr<detail::BudgetState> budget;
        std::size_t bytes;
        bool armed{true};
        ~Rollback() { if (armed) { std::lock_guard lock(budget->mutex); --budget->requests; budget->bytes -= bytes; } }
    } rollback{s->budget, payload.size()};
    std::string normalized;
    if (!payload.empty()) normalized.assign(payload.data(), payload.size());
    auto payload_state = std::make_shared<const detail::PayloadResources>(std::move(normalized), std::move(resource), s->budget);
    rollback.armed = false;
    auto ticket_state = std::make_shared<detail::AdmissionTicketState>();
    ticket_state->payload = payload_state;
    auto rec = std::make_shared<detail::AdmissionRecord>(ticket_state);
    ticket_state->completion = rec->producer.completion();
    rec->deadline = queue_deadline; rec->timeout = execution_timeout;
    const bool queue_full = s->active
        ? s->queue.size() >= s->limits.queue_capacity
        : !s->queue.empty() && s->queue.size() - 1 >= s->limits.queue_capacity;
    if (queue_full) return {AdmissionStatus::full, {}};
    s->queue.push_back(rec);
    s->queued_snapshot.store(s->queue.size(), std::memory_order_release);
    return {AdmissionStatus::accepted, AdmissionTicket(std::move(ticket_state))};
}
AdmissionTicket AdmissionCoordinator::take_next(Clock::time_point now) {
    auto s = state_; require_owner(s);
    if (s->closed || !s->healthy || s->active) return {};
    while (!s->queue.empty()) {
        auto r = s->queue.front(); s->queue.pop_front();
        s->queued_snapshot.store(s->queue.size(), std::memory_order_release);
        if (r->ticket->stop.stop_requested() || now >= r->deadline) {
            phase_settle(r, CompletionStatus::cancelled);
            continue;
        }
        r->ticket->phase.store(AdmissionPhase::active, std::memory_order_release);
        s->active = r; s->active_snapshot.store(true, std::memory_order_release);
        return AdmissionTicket(r->ticket);
    }
    return {};
}
ActiveRequest AdmissionCoordinator::start(const AdmissionTicket& ticket, Clock::time_point now) {
    auto s = state_; require_owner(s);
    if (!ticket.state_ || !s->active || ticket.state_ != s->active->ticket || s->active->started_work ||
        ticket.phase() != AdmissionPhase::active) throw std::invalid_argument("invalid active ticket");
    if (s->generation == std::numeric_limits<std::uint64_t>::max()) {
        s->healthy = false; s->healthy_snapshot.store(false, std::memory_order_release);
        throw std::overflow_error("admission generation exhausted");
    }
    s->active->started_work = true; s->active->started = now; s->active->generation = ++s->generation;
    return {s->pool_id, s->generation};
}
void AdmissionCoordinator::retire(ActiveRequest active, const Completion& fence) {
    auto s = state_; require_owner(s);
    if (!s->active || !s->active->started_work || active.pool_id != s->pool_id ||
        active.generation != s->active->generation || !fence.valid() || s->active->fence.valid())
        throw std::invalid_argument("invalid active request or fence");
    s->active->fence = fence;
}
void AdmissionCoordinator::poll(Clock::time_point now) {
    auto s = state_; require_owner(s);
    for (auto it = s->queue.begin(); it != s->queue.end();) {
        auto r = *it;
        if (r->ticket->stop.stop_requested() || now >= r->deadline) {
            phase_settle(r, CompletionStatus::cancelled); it = s->queue.erase(it);
        } else ++it;
    }
    s->queued_snapshot.store(s->queue.size(), std::memory_order_release);
    if (!s->active) return;
    auto r = s->active;
    if (!r->started_work) {
        if (r->ticket->stop.stop_requested()) { phase_settle(r, CompletionStatus::cancelled); s->active.reset(); s->active_snapshot.store(false, std::memory_order_release); }
        return;
    }
    if (r->ticket->stop.stop_requested() || now >= saturated_deadline(r->started, r->timeout))
        r->ticket->stop.request_stop();
    if (!r->fence.valid() || r->fence.status() == CompletionStatus::pending) return;
    auto outcome = r->fence.status();
    if (outcome == CompletionStatus::success || outcome == CompletionStatus::cancelled) {
        phase_settle(r, outcome == CompletionStatus::cancelled || r->ticket->stop.stop_requested()
            ? CompletionStatus::cancelled : CompletionStatus::success);
    } else {
        s->healthy = false; s->healthy_snapshot.store(false, std::memory_order_release);
        phase_settle(r, CompletionStatus::failure);
        for (auto& queued : s->queue) phase_settle(queued, CompletionStatus::cancelled);
        s->queue.clear(); s->queued_snapshot.store(0, std::memory_order_release);
    }
    s->active.reset(); s->active_snapshot.store(false, std::memory_order_release);
}
void AdmissionCoordinator::close() {
    auto s = state_; require_owner(s);
    s->closed = true;
    for (auto& r : s->queue) phase_settle(r, CompletionStatus::cancelled);
    s->queue.clear(); s->queued_snapshot.store(0, std::memory_order_release);
    if (s->active) s->active->ticket->stop.request_stop();
}
bool AdmissionCoordinator::healthy() const noexcept { return state_->healthy_snapshot.load(std::memory_order_acquire); }
std::size_t AdmissionCoordinator::queued() const { require_owner(state_); return state_->queue.size(); }
bool AdmissionCoordinator::has_active() const { require_owner(state_); return static_cast<bool>(state_->active); }
} // namespace saga::runtime
