#pragma once

#include <cstddef>
#include <string_view>

namespace saga::platform::detail {
inline constexpr std::size_t max_argument_bytes = 32767;
inline constexpr std::size_t max_total_argument_bytes = 1024 * 1024;
inline constexpr int max_argument_count = 4096;

[[nodiscard]] bool valid_utf8(std::string_view value) noexcept;
void validate_name(std::string_view name, std::size_t max_bytes);
void validate_value(std::string_view value, std::size_t max_bytes);
}
