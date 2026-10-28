#pragma once

#include "saga/runtime/completion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace saga::runtime {

struct SequenceHandle {
    std::uint64_t pool_id{};
    std::size_t slot{};
    std::uint64_t generation{};
    explicit operator bool() const noexcept { return pool_id != 0 && generation != 0; }
    friend bool operator==(const SequenceHandle&, const SequenceHandle&) = default;
};

struct SequenceSlotCounts {
    std::size_t active{};
    std::size_t retiring{};
    std::size_t quarantined{};
    std::size_t free{};
};

enum class RetirementResult { pending, reusable, quarantined };

// A bounded logical reservation pool only; it owns no physical execution resources.
// All operations, including destruction, must be externally serialized; operational
// calls are restricted to the constructing thread. Tokens may outlive the pool and
// retain only its logical state; token validity means it still owns a retirement
// record, not that the pool is operational. Destroying or overwriting a token does
// not imply a fence and leaves its reservation retiring until pool shutdown.
class SequenceSlots {
    struct State;
public:
    class RetirementToken {
    public:
        RetirementToken() noexcept = default;
        RetirementToken(const RetirementToken&) = delete;
        RetirementToken& operator=(const RetirementToken&) = delete;
        RetirementToken(RetirementToken&&) noexcept = default;
        RetirementToken& operator=(RetirementToken&&) noexcept = default;
        ~RetirementToken() = default;
        bool valid() const noexcept { return static_cast<bool>(state_); }
    private:
        friend class SequenceSlots;
        RetirementToken(std::shared_ptr<State> state, std::size_t slot,
                        std::uint64_t generation) noexcept;
        std::shared_ptr<State> state_;
        std::size_t slot_{};
        std::uint64_t generation_{};
    };

    explicit SequenceSlots(std::size_t capacity);
    SequenceSlots(const SequenceSlots&) = delete;
    SequenceSlots& operator=(const SequenceSlots&) = delete;
    SequenceSlots(SequenceSlots&&) = delete;
    SequenceSlots& operator=(SequenceSlots&&) = delete;
    ~SequenceSlots();

    SequenceHandle acquire();
    RetirementToken begin_release(SequenceHandle handle);
    RetirementResult finish_release(RetirementToken&& token, const Completion& fence);
    SequenceSlotCounts counts() const;

private:
    std::shared_ptr<State> state_;
};

} // namespace saga::runtime
