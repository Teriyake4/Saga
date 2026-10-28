#include "saga/runtime/sequence_slots.h"

#include <atomic>
#include <limits>
#include <stdexcept>

namespace saga::runtime {
namespace {
std::atomic<std::uint64_t> next_pool_id{1};

// Exhaustion is intentionally not forced in tests: doing so would require a production seam.
std::uint64_t allocate_pool_id() {
    auto value = next_pool_id.load(std::memory_order_relaxed);
    for (;;) {
        if (value == 0 || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("sequence pool identity space exhausted");
        if (next_pool_id.compare_exchange_weak(value, value + 1,
                                               std::memory_order_relaxed))
            return value;
    }
}
}

struct SequenceSlots::State {
    enum class Status { free, active, retiring, quarantined };
    struct Slot { Status status{Status::free}; std::uint64_t generation{1}; };
    State(std::size_t size, std::uint64_t identity)
        : slots(size), id(identity), owner(std::this_thread::get_id()) {}
    void check_owner() const {
        if (std::this_thread::get_id() != owner)
            throw std::logic_error("sequence pool used from non-owner thread");
        if (shutdown) throw std::logic_error("sequence pool is shut down");
    }
    std::vector<Slot> slots;
    std::uint64_t id;
    std::thread::id owner;
    bool shutdown{false};
};

SequenceSlots::RetirementToken::RetirementToken(std::shared_ptr<State> state,
        std::size_t slot, std::uint64_t generation) noexcept
    : state_(std::move(state)), slot_(slot), generation_(generation) {}

SequenceSlots::SequenceSlots(std::size_t capacity)
    : state_(std::make_shared<State>(capacity, allocate_pool_id())) {
    if (capacity == 0) throw std::invalid_argument("sequence pool capacity must be positive");
}

SequenceSlots::~SequenceSlots() {
    if (state_) state_->shutdown = true;
}

SequenceHandle SequenceSlots::acquire() {
    state_->check_owner();
    for (std::size_t i = 0; i < state_->slots.size(); ++i) {
        auto& slot = state_->slots[i];
        if (slot.status == State::Status::free) {
            slot.status = State::Status::active;
            return {state_->id, i, slot.generation};
        }
    }
    throw std::runtime_error("sequence slot pool exhausted");
}

SequenceSlots::RetirementToken SequenceSlots::begin_release(SequenceHandle handle) {
    state_->check_owner();
    if (handle.pool_id != state_->id || handle.slot >= state_->slots.size())
        throw std::invalid_argument("foreign or invalid sequence handle");
    auto& slot = state_->slots[handle.slot];
    if (slot.status != State::Status::active || slot.generation != handle.generation)
        throw std::invalid_argument("stale or inactive sequence handle");
    slot.status = State::Status::retiring;
    return RetirementToken(state_, handle.slot, handle.generation);
}

RetirementResult SequenceSlots::finish_release(RetirementToken&& token,
                                                const Completion& fence) {
    state_->check_owner();
    if (!token.state_) throw std::invalid_argument("invalid or already-finished retirement token");
    if (token.state_.get() != state_.get())
        throw std::invalid_argument("retirement token belongs to another pool");
    if (!fence.valid()) throw std::invalid_argument("invalid completion fence");
    auto& slot = state_->slots.at(token.slot_);
    if (slot.status != State::Status::retiring || slot.generation != token.generation_)
        throw std::invalid_argument("stale retirement token");
    const auto status = fence.status();
    if (status == CompletionStatus::pending) return RetirementResult::pending;
    token.state_.reset();
    // Generation exhaustion is likewise unreachable through the public API in practical tests.
    if (status != CompletionStatus::success || slot.generation == std::numeric_limits<std::uint64_t>::max()) {
        slot.status = State::Status::quarantined;
        return RetirementResult::quarantined;
    }
    ++slot.generation;
    slot.status = State::Status::free;
    return RetirementResult::reusable;
}

SequenceSlotCounts SequenceSlots::counts() const {
    state_->check_owner();
    SequenceSlotCounts result;
    for (const auto& slot : state_->slots) {
        switch (slot.status) {
        case State::Status::active: ++result.active; break;
        case State::Status::retiring: ++result.retiring; break;
        case State::Status::quarantined: ++result.quarantined; break;
        case State::Status::free: ++result.free; break;
        }
    }
    return result;
}

} // namespace saga::runtime
