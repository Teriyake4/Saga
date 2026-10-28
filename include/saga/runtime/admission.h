#pragma once

#include "saga/runtime/completion.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string_view>
#include <thread>

namespace saga::runtime {

namespace detail { struct AdmissionState; struct AdmissionTicketState; }

enum class AdmissionStatus { accepted, full, stopping, failed, oversize, expired, invalid_timeout };
enum class AdmissionPhase { queued, active, settled };
struct AdmissionLimits {
    std::size_t queue_capacity{};
    std::size_t outstanding_capacity{};
    std::size_t byte_capacity{};
};

class AdmissionTicket {
public:
    AdmissionTicket() noexcept = default;
    bool valid() const noexcept;
    bool cancel() const noexcept;
    AdmissionPhase phase() const;
    Completion completion() const;
    std::stop_token stop_token() const noexcept;
    std::string_view payload() const;
    ResourceLease resource() const;
private:
    friend class AdmissionCoordinator;
    explicit AdmissionTicket(std::shared_ptr<detail::AdmissionTicketState>) noexcept;
    std::shared_ptr<detail::AdmissionTicketState> state_;
};

struct AdmissionResult { AdmissionStatus status{AdmissionStatus::failed}; AdmissionTicket ticket; };
struct ActiveRequest {
    std::uint64_t pool_id{};
    std::uint64_t generation{};
    explicit operator bool() const noexcept { return pool_id != 0 && generation != 0; }
    friend bool operator==(const ActiveRequest&, const ActiveRequest&) = default;
};

// All owner operations, including close and destruction, are externally serialized on the
// constructing thread. Ticket cancel/read/token calls are thread-safe; stop callbacks may reenter
// those calls but must not invoke owner operations. queued() includes the ready-slot entry: while
// idle, queue_capacity N permits N waiting entries plus one ready entry; while active it permits
// N waiting entries. Payload size is charged by normalized string size; caller resource size is
// caller-budgeted separately. Execution code must retain ticket.resource() (or ticket completion)
// through its physical fence. Destruction abandons pending completion notification; it does not
// assert that submitted work has physically completed.
class AdmissionCoordinator {
public:
    using Clock = std::chrono::steady_clock;
    explicit AdmissionCoordinator(AdmissionLimits limits);
    AdmissionCoordinator(const AdmissionCoordinator&) = delete;
    AdmissionCoordinator& operator=(const AdmissionCoordinator&) = delete;
    ~AdmissionCoordinator();
    AdmissionResult admit(std::string_view payload, ResourceLease resource,
                          Clock::time_point queue_deadline, Clock::duration execution_timeout,
                          Clock::time_point now);
    AdmissionTicket take_next(Clock::time_point now);
    ActiveRequest start(const AdmissionTicket& ticket, Clock::time_point now);
    void retire(ActiveRequest active, const Completion& fence);
    void poll(Clock::time_point now);
    void close();
    bool healthy() const noexcept;
    std::size_t queued() const;
    bool has_active() const;
private:
    std::shared_ptr<detail::AdmissionState> state_;
};

} // namespace saga::runtime
