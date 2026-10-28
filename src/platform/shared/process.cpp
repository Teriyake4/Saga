#include "saga/platform/process.hpp"
#include "process_internal.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace saga::platform::detail {
bool valid_utf8(std::string_view value) noexcept {
    std::size_t index = 0;
    while (index < value.size()) {
        const auto first = static_cast<unsigned char>(value[index]);
        if (first < 0x80) {
            ++index;
            continue;
        }

        std::size_t count = 0;
        unsigned int codepoint = 0;
        if ((first & 0xe0) == 0xc0) {
            count = 2;
            codepoint = first & 0x1f;
            if (codepoint < 2) return false;
        } else if ((first & 0xf0) == 0xe0) {
            count = 3;
            codepoint = first & 0x0f;
        } else if ((first & 0xf8) == 0xf0) {
            count = 4;
            codepoint = first & 0x07;
            if (codepoint > 4) return false;
        } else {
            return false;
        }

        if (count > value.size() - index) return false;
        for (std::size_t offset = 1; offset < count; ++offset) {
            const auto next = static_cast<unsigned char>(value[index + offset]);
            if ((next & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if ((count == 3 && codepoint < 0x800) ||
            (count == 4 && codepoint < 0x10000) || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
        index += count;
    }
    return true;
}

void validate_name(std::string_view name, std::size_t max_bytes) {
    if (name.empty() || name.size() > max_argument_bytes ||
        name.find('\0') != name.npos || name.find('=') != name.npos ||
        !valid_utf8(name)) {
        throw std::invalid_argument("invalid environment variable name");
    }
    if (max_bytes == 0 || max_bytes > max_total_argument_bytes) {
        throw std::invalid_argument("environment byte limit must be in 1..1MiB");
    }
}

void validate_value(std::string_view value, std::size_t max_bytes) {
    if (value.size() > max_bytes) {
        throw std::length_error("environment value exceeds byte limit");
    }
    if (!valid_utf8(value)) {
        throw std::runtime_error("environment value is not valid UTF-8");
    }
}
}

namespace saga::platform {
std::vector<std::string> collect_arguments(int argc, char* const argv[]) {
    if (argc < 0 || argc > detail::max_argument_count || (argc > 0 && !argv)) {
        throw std::invalid_argument("invalid argv");
    }

    std::vector<std::string> result;
    result.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0);
    std::size_t total_bytes = 0;
    for (int index = 1; index < argc; ++index) {
        if (!argv[index]) throw std::invalid_argument("null argv entry");
        std::size_t length = 0;
        while (length <= detail::max_argument_bytes && argv[index][length] != '\0') ++length;
        if (length > detail::max_argument_bytes) {
            throw std::length_error("argument exceeds byte limit");
        }
        if (length > detail::max_total_argument_bytes - total_bytes) {
            throw std::length_error("total arguments exceed byte limit");
        }
        std::string_view argument(argv[index], length);
        if (!detail::valid_utf8(argument)) {
            throw std::invalid_argument("argument is not valid UTF-8");
        }
        total_bytes += length;
        result.emplace_back(argument);
    }
    return result;
}
}
