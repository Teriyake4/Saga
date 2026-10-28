#include "saga/platform/process.hpp"
#include "../shared/process_internal.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace saga::platform {
namespace {
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> stop_flag{false};
std::mutex registration_mutex;
bool active = false;
struct sigaction previous_int {};
struct sigaction previous_term {};

void shutdown_handler(int) noexcept {
    stop_flag.store(true, std::memory_order_relaxed);
}
}

std::vector<std::string> collect_arguments(int, wchar_t* const[]) {
    throw std::invalid_argument("wide argv is supported only on Windows");
}

std::optional<std::string> read_environment(std::string_view name, std::size_t max_bytes) {
    detail::validate_name(name, max_bytes);
    std::string key(name);
    // POSIX does not provide a bounded snapshot API; this assumes the process environment
    // is immutable while this read occurs.
    const char* value = std::getenv(key.c_str());
    if (!value) return std::nullopt;

    std::size_t length = 0;
    while (length <= max_bytes && value[length] != '\0') ++length;
    if (length > max_bytes) throw std::length_error("environment value exceeds byte limit");
    std::string result(value, length);
    detail::validate_value(result, max_bytes);
    return result;
}

ShutdownSignals::ShutdownSignals() {
    std::lock_guard<std::mutex> lock(registration_mutex);
    if (active) throw std::logic_error("shutdown signals already registered");
    stop_flag.store(false, std::memory_order_relaxed);

    struct sigaction action {};
    action.sa_handler = shutdown_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    if (sigaction(SIGINT, &action, &previous_int) != 0) {
        throw std::runtime_error("could not register SIGINT");
    }
    if (sigaction(SIGTERM, &action, &previous_term) != 0) {
        sigaction(SIGINT, &previous_int, nullptr);
        throw std::runtime_error("could not register SIGTERM");
    }
    active = true;
    active_ = true;
}

ShutdownSignals::~ShutdownSignals() {
    if (!active_) return;
    std::lock_guard<std::mutex> lock(registration_mutex);
    sigaction(SIGINT, &previous_int, nullptr);
    sigaction(SIGTERM, &previous_term, nullptr);
    active = false;
}

bool ShutdownSignals::requested() const noexcept {
    return stop_flag.load(std::memory_order_relaxed);
}
}
