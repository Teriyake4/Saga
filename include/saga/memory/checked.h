#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace saga::memory {

inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("byte count addition overflow");
    return a + b;
}

inline std::uint64_t checked_multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("byte count multiplication overflow");
    return a * b;
}

inline bool is_power_of_two(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

inline std::uint64_t ceil_divide(std::uint64_t value, std::uint64_t divisor) {
    if (divisor == 0) throw std::invalid_argument("zero block/page size");
    return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

inline std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
    if (!is_power_of_two(alignment))
        throw std::invalid_argument("alignment/page size must be a power of two");
    return checked_multiply(ceil_divide(value, alignment), alignment);
}

} // namespace saga::memory
