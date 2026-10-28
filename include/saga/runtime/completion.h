#pragma once

#include "saga/memory/ledger.h"

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace saga::runtime {

enum class CompletionStatus { pending, success, failure, cancelled, abandoned };

struct Outcome {
    CompletionStatus status{CompletionStatus::pending};
    std::exception_ptr error;
};

namespace detail {
template<class T> struct IsReferenceWrapper : std::false_type {};
template<class T> struct IsReferenceWrapper<std::reference_wrapper<T>> : std::true_type {};
}

// A lease owns an immutable type-erased RAII value. Its destructor must not throw.
// Raw pointers and reference_wrapper values are rejected because they do not transfer ownership.
class ResourceLease {
public:
    ResourceLease() noexcept = default;

    template<class T>
        requires (!std::is_same_v<std::remove_cvref_t<T>, ResourceLease> &&
                  !std::is_pointer_v<std::remove_cvref_t<T>> &&
                  !detail::IsReferenceWrapper<std::remove_cvref_t<T>>::value &&
                  std::is_nothrow_destructible_v<std::remove_cvref_t<T>>) 
    explicit ResourceLease(T&& value)
        : owner_(std::make_shared<const Holder<std::remove_cvref_t<T>>>(std::forward<T>(value))) {}

    explicit operator bool() const noexcept { return static_cast<bool>(owner_); }

    static ResourceLease retain(memory::AllocationLease lease) {
        return ResourceLease(std::move(lease));
    }

private:
    struct HolderBase {
        virtual ~HolderBase() noexcept = default;
    };

    template<class T>
    struct Holder final : HolderBase {
        template<class U>
        explicit Holder(U&& value) : value_(std::forward<U>(value)) {}
        T value_;
    };

    std::shared_ptr<const HolderBase> owner_;
};

namespace detail {
struct CompletionState;
}

class CompletionSource;

class Completion {
public:
    Completion() noexcept = default;

    bool valid() const noexcept { return static_cast<bool>(state_); }
    CompletionStatus status() const;
    bool is_ready() const;
    Outcome wait() const;
    Outcome wait_until(std::chrono::steady_clock::time_point deadline) const;
    Outcome wait_for(std::chrono::steady_clock::duration timeout) const;

    // Returns only on success; rethrows the stored exception on failure and throws
    // std::runtime_error for cancelled/abandoned outcomes.
    void result() const;

private:
    friend class CompletionSource;
    explicit Completion(std::shared_ptr<detail::CompletionState> state) noexcept;
    std::shared_ptr<detail::CompletionState> state_;
};

class CompletionSource {
public:
    // Multiple threads may compete to settle a source; exactly one valid settlement wins.
    // Moving, assigning, or destroying the same source object must be externally serialized.
    // Abandonment only notifies consumers; it is not a physical-work fence. The executing
    // backend must retain its producer until all contributing work has actually completed.
    CompletionSource();
    explicit CompletionSource(ResourceLease resource);
    CompletionSource(const CompletionSource&) = delete;
    CompletionSource& operator=(const CompletionSource&) = delete;
    CompletionSource(CompletionSource&& other) noexcept;
    CompletionSource& operator=(CompletionSource&& other) noexcept;
    ~CompletionSource() noexcept;

    bool valid() const noexcept { return static_cast<bool>(state_); }
    Completion completion() const;
    bool settle(CompletionStatus status, std::exception_ptr error = {}) noexcept;
    bool succeed() noexcept { return settle(CompletionStatus::success); }
    bool fail(std::exception_ptr error = {}) noexcept { return settle(CompletionStatus::failure, error); }
    bool cancel() noexcept { return settle(CompletionStatus::cancelled); }

private:
    void abandon() noexcept;
    std::shared_ptr<detail::CompletionState> state_;
};

} // namespace saga::runtime
