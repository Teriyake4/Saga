#include "saga/api/server.h"
#include "saga/core/config.h"
#include "saga/core/log.h"
#include "saga/platform/process.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
int execute(const std::vector<std::string_view>& arguments) {
    auto command = saga::parse_command_line(arguments);
    if (command.help) { std::cout << saga::usage(); return 0; }
    if (command.version) { std::cout << "saga-server 0.1.0-dev (host foundation; no inference)\n"; return 0; }
    if (command.check_config) {
        std::cout << "Static configuration valid; capability support, fit and readiness are NOT validated.\n";
        return 0;
    }

    saga::platform::ShutdownSignals shutdown;
    // One byte beyond the transport grammar's bound lets it explain a 4097-byte
    // invalid token without logging its value; larger environment values fail early.
    auto bearer_key = saga::platform::read_environment(command.config.server.api_key_env, 4097);
    saga::api::Server server(command.config, bearer_key.value_or(""));
    const auto port = server.bind();
    saga::log_event("info", "listener_bound", command.config.server.host + ":" + std::to_string(port));
    saga::log_event("warning", "model_unavailable", "host foundation only; no model loaded; readiness remains false");
    if (!saga::is_loopback_binding(command.config.server.host))
        saga::log_event("warning", "remote_plaintext_http", "use a trusted TLS reverse proxy with absolute request/header deadlines");
    std::jthread signals([&](std::stop_token token) {
        while (!token.stop_requested()) {
            if (shutdown.requested()) { server.stop(); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const bool success = server.run();
    signals.request_stop();
    signals.join();
    saga::log_event(success ? "info" : "error", "listener_stopped");
    return success ? 0 : 1;
}

template <typename Character>
int native_main(int argc, Character* argv[]) {
    try {
        const auto storage = saga::platform::collect_arguments(argc, argv);
        const std::vector<std::string_view> arguments(storage.begin(), storage.end());
        return execute(arguments);
    } catch (const std::exception& error) {
        saga::log_event("error", "startup_failed", error.what());
        return 1;
    }
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) { return native_main(argc, argv); }
#else
int main(int argc, char* argv[]) { return native_main(argc, argv); }
#endif
