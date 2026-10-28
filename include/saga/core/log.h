#pragma once

#include <string_view>

namespace saga {
// Never pass request bodies, authentication values or generated text to operational logging.
void log_event(std::string_view level, std::string_view event, std::string_view detail = {});
} // namespace saga
