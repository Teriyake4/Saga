#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "saga/platform/process.hpp"
#include "../shared/process_internal.h"
#include <windows.h>

#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace saga::platform {
namespace {
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> stop_flag{false};
std::mutex registration_mutex;
bool active = false;

BOOL WINAPI shutdown_handler(DWORD event) noexcept {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    stop_flag.store(true, std::memory_order_relaxed);
    return TRUE;
}

std::string to_utf8(const wchar_t* value, int length) {
    if (length == 0) return {};
    const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, length,
                                          nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) throw std::invalid_argument("invalid UTF-16 argument");
    std::string result(static_cast<std::size_t>(bytes), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, length,
                            result.data(), bytes, nullptr, nullptr) != bytes) {
        throw std::invalid_argument("UTF-16 conversion failed");
    }
    return result;
}

std::wstring to_wide(std::string_view value) {
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("environment name is too long");
    }
    const int length = static_cast<int>(value.size());
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), length,
                                          nullptr, 0);
    if (count <= 0) throw std::invalid_argument("invalid UTF-8 environment name");
    std::wstring result(static_cast<std::size_t>(count), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), length,
                            result.data(), count) != count) {
        throw std::invalid_argument("UTF-8 conversion failed");
    }
    return result;
}
}

std::vector<std::string> collect_arguments(int argc, wchar_t* const argv[]) {
    if (argc < 0 || argc > detail::max_argument_count || (argc > 0 && !argv)) {
        throw std::invalid_argument("invalid argv");
    }
    std::vector<std::string> result;
    result.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0);
    std::size_t total_bytes = 0;
    for (int index = 1; index < argc; ++index) {
        if (!argv[index]) throw std::invalid_argument("null argv entry");
        std::size_t length = 0;
        while (length <= detail::max_argument_bytes && argv[index][length] != L'\0') ++length;
        if (length > detail::max_argument_bytes) throw std::length_error("argument exceeds length limit");
        std::string argument = to_utf8(argv[index], static_cast<int>(length));
        if (argument.size() > detail::max_argument_bytes) throw std::length_error("argument exceeds byte limit");
        if (argument.size() > detail::max_total_argument_bytes - total_bytes) {
            throw std::length_error("total arguments exceed byte limit");
        }
        total_bytes += argument.size();
        result.push_back(std::move(argument));
    }
    return result;
}

std::optional<std::string> read_environment(std::string_view name, std::size_t max_bytes) {
    detail::validate_name(name, max_bytes);
    const std::wstring key = to_wide(name);
    for (int attempt = 0; attempt < 3; ++attempt) {
        SetLastError(ERROR_SUCCESS);
        const DWORD needed = GetEnvironmentVariableW(key.c_str(), nullptr, 0);
        if (needed == 0) {
            const DWORD error = GetLastError();
            if (error == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
            if (error == ERROR_SUCCESS) return std::string{};
            throw std::runtime_error("could not read environment variable");
        }
        if (needed > 32769) throw std::length_error("environment value exceeds UTF-16 bound");
        std::wstring value(needed, L'\0');
        SetLastError(ERROR_SUCCESS);
        const DWORD got = GetEnvironmentVariableW(key.c_str(), value.data(), needed);
        if (got == 0) {
            const DWORD error = GetLastError();
            if (error == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
            if (error == ERROR_SUCCESS) return std::string{};
            continue;
        }
        if (got >= needed) continue;
        std::string result = to_utf8(value.data(), static_cast<int>(got));
        detail::validate_value(result, max_bytes);
        return result;
    }
    throw std::runtime_error("environment variable changed during read");
}

ShutdownSignals::ShutdownSignals() {
    std::lock_guard<std::mutex> lock(registration_mutex);
    if (active) throw std::logic_error("shutdown signals already registered");
    stop_flag.store(false, std::memory_order_relaxed);
    if (!SetConsoleCtrlHandler(shutdown_handler, TRUE)) {
        throw std::runtime_error("could not register console handler");
    }
    active = true;
    active_ = true;
}

ShutdownSignals::~ShutdownSignals() {
    if (!active_) return;
    std::lock_guard<std::mutex> lock(registration_mutex);
    SetConsoleCtrlHandler(shutdown_handler, FALSE);
    active = false;
}

bool ShutdownSignals::requested() const noexcept {
    return stop_flag.load(std::memory_order_relaxed);
}
}
