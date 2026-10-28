#include "saga/core/log.h"

#include <json.hpp>
#include <chrono>
#include <iostream>
#include <mutex>

namespace saga {
void log_event(std::string_view level, std::string_view event, std::string_view detail) {
    static std::mutex mutex;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const nlohmann::json record{
        {"timestamp_ms", std::chrono::duration_cast<std::chrono::milliseconds>(now).count()},
        {"level", level}, {"event", event}, {"detail", detail}};
    std::lock_guard lock(mutex);
    std::cerr << record.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
}
} // namespace saga
