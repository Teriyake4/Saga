#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace saga::platform {
// Excludes argv[0]. UTF-8 is strict; Windows startup uses the native UTF-16 overload.
[[nodiscard]] std::vector<std::string> collect_arguments(int argc, char* const argv[]);
[[nodiscard]] std::vector<std::string> collect_arguments(int argc, wchar_t* const argv[]);
// Values are bounded by UTF-8 bytes and never logged; absent and empty are distinct.
[[nodiscard]] std::optional<std::string> read_environment(std::string_view utf8_name, std::size_t max_bytes = 4096);

// One registration per process. Native callbacks touch process-lifetime atomic state
// only; normal application threads perform cleanup after observing requested().
class ShutdownSignals {
public:
    ShutdownSignals();
    ~ShutdownSignals();
    ShutdownSignals(const ShutdownSignals&) = delete;
    ShutdownSignals& operator=(const ShutdownSignals&) = delete;
    ShutdownSignals(ShutdownSignals&&) = delete;
    ShutdownSignals& operator=(ShutdownSignals&&) = delete;
    [[nodiscard]] bool requested() const noexcept;
private:
    bool active_ = false;
};
}
