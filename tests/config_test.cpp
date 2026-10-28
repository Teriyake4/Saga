#include "saga/core/config.h"
#include "saga/api/server.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& operation) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}
saga::CommandLine cli(std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> values(args);
    return saga::parse_command_line(values);
}

class TemporaryConfig {
public:
    TemporaryConfig() {
        const auto root = std::filesystem::temp_directory_path();
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned i = 0; i < 100; ++i) {
            directory_ = root / ("saga-config-" + std::to_string(stamp) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(directory_)) return;
        }
        throw std::runtime_error("cannot create temporary config directory");
    }
    ~TemporaryConfig() {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }
    std::filesystem::path path() const { return directory_ / "config.toml"; }
    std::string write(std::string_view contents) const {
        std::ofstream file(path(), std::ios::binary | std::ios::trunc);
        file << contents;
        file.close();
        check(!file.fail(), "write temporary config");
        const auto utf8 = path().u8string();
        return std::string(utf8.begin(), utf8.end());
    }
private:
    std::filesystem::path directory_;
};

void merged_config_tests() {
    TemporaryConfig file;
    auto path = file.write("[context]\nmax_tokens=16384\n");
    rejects([&] { (void)saga::load_config(file.path()); });
    check(cli({"--config", path, "--kv-capacity", "auto"}).config.max_tokens.tokens == 16384,
          "validate context after merging TOML with CLI, not before");
    rejects([&] { (void)cli({"--config", path}); });
    path = file.write("[kv]\nkey_format='fp16'\n");
    rejects([&] { (void)saga::load_config(file.path()); });
    check(cli({"--cache-type-v", "fp16", "--config", path}).config.kv_pair ==
          saga::kv::Pair{saga::kv::Codec::fp16, saga::kv::Codec::fp16}, "CLI repairs TOML/default pair");
    rejects([&] { (void)cli({"--config", path, "--cache-type-v", "q4_0"}); });
    path = file.write("[server]\nport=17\n[context]\nmax_tokens=0\n"
                      "[speculation]\nmode='unknown'\nmax_draft_tokens=4\n");
    rejects([&] { (void)saga::load_config(file.path()); });
    const auto merged = cli({"--port", "23", "--context", "4096", "--speculation", "off",
                             "--max-draft-tokens", "2", "--config", path}).config;
    check(merged.server.port == 23 && merged.max_tokens.tokens == 4096 &&
          merged.kv_capacity_tokens.tokens == 8192 && merged.speculation_mode == "off" &&
          merged.max_draft_tokens == 2, "CLI wins regardless of --config position; defaults retained");
    path = file.write("[server]\nport=17\n");
    check(cli({"--config", path, "--port", "24"}).config.server.port == 24, "CLI after config wins");
    check(saga::load_config(file.path()).server.port == 17, "standalone load still works");
    for (const auto input : {"[server]\nport='17'\n", "[server]\nport=-1\n",
                             "[server]\nunknown=1\n", "[server", "[server]\nhttp_workers=0\n"}) {
        path = file.write(input);
        rejects([&] { (void)cli({"--config", path, "--port", "24"}); });
    }
    for (const auto pair : {saga::kv::Pair{static_cast<saga::kv::Codec>(255), static_cast<saga::kv::Codec>(255)},
                            saga::kv::Pair{static_cast<saga::kv::Codec>(255), saga::kv::Codec::fp16},
                            saga::kv::Pair{saga::kv::Codec::fp16, static_cast<saga::kv::Codec>(255)}}) {
        saga::Config config;
        config.kv_pair = pair;
        rejects([&] { saga::validate(config); });
    }
    for (const auto& entry : saga::kv::catalog()) {
        saga::Config config;
        config.kv_pair = {entry.id, entry.id};
        saga::validate(config);
    }
}

saga::Config listener_config(std::uint16_t port = 0) {
    saga::Config config;
    config.server.port = port;
    config.server.http_workers = 1;
    return config;
}
bool port_available(std::uint16_t port) {
    // The vendor enables SO_REUSEPORT: another vendor listener is not a leak probe.
    // No connections are made in these lifecycle tests, so there is no TIME_WAIT.
    const auto socket = ::socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
    check(socket != INVALID_SOCKET, "create port probe");
#else
    check(socket >= 0, "create port probe");
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    const bool available = ::bind(socket, reinterpret_cast<const sockaddr*>(&address),
                                      static_cast<int>(sizeof(address))) == 0;
#ifdef _WIN32
    ::closesocket(socket);
#else
    ::close(socket);
#endif
    return available;
}
void check_released(std::uint16_t port) {
    check(port_available(port), "listener handle was closed, not merely reusable via SO_REUSEPORT");
    saga::api::Server replacement(listener_config(port));
    check(replacement.bind() == port, "released listener can be rebound in the same process");
}
void lifecycle_tests() {
    struct StartupFailure {};
    for (unsigned i = 0; i < 8; ++i) {
        std::uint16_t port = 0;
        {
            saga::api::Server server(listener_config());
            port = server.bind();
            check(!port_available(port), "bound listener really owns the port");
            bool rebind_rejected = false;
            try { (void)server.bind(); }
            catch (const std::logic_error&) { rebind_rejected = true; }
            check(rebind_rejected && !port_available(port), "double bind preserves the original listener");
        }
        check_released(port);
        {
            saga::api::Server server(listener_config());
            port = server.bind();
            server.stop();
        }
        check_released(port);
        {
            saga::api::Server server(listener_config());
            port = server.bind();
            server.stop();
            server.stop();
            check(server.run() && !server.is_running(), "stop-before-run succeeds without starting");
            check_released(port); // must be released before server destruction
            bool rerun_rejected = false;
            try { (void)server.run(); }
            catch (const std::logic_error&) { rerun_rejected = true; }
            check(rerun_rejected, "cancelled run still consumes the single run");
        }
        {
            saga::api::Server server(listener_config());
            server.stop();
            port = server.bind();
            check(server.run(), "stop-before-bind cancellation is sticky");
            check_released(port);
        }
        try {
            saga::api::Server server(listener_config());
            port = server.bind();
            throw StartupFailure{}; // simulate a failure in post-bind application startup
        } catch (const StartupFailure&) {}
        check_released(port);

        saga::api::Server server(listener_config());
        port = server.bind();
        bool success = false;
        std::exception_ptr failure;
        std::jthread runner([&] {
            try { success = server.run(); }
            catch (...) { failure = std::current_exception(); }
        });
        // Alternate a startup cancellation race with a confirmed running listener.
        bool running = false;
        if (i % 2) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!(running = server.is_running()) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::jthread other_stop([&] { server.stop(); });
        server.stop();
        other_stop.join();
        runner.join();
        if (failure) std::rethrow_exception(failure);
        check(success && !server.is_running(), "concurrent stop/startup race shuts down successfully");
        check(!(i % 2) || running, "listener reached running state");
        check_released(port);
    }
}

void key_tests() {
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~+/-";
    for (const auto& key : {alphabet, alphabet + "=", alphabet + "===", std::string(4096, 'a'),
                            "a" + std::string(4095, '=')}) {
        saga::api::Server valid(listener_config(), key);
    }
    for (const auto& key : std::vector<std::string>{"secret%41", "secret%", "space key", "tab\tkey",
             "line\r\nkey", "colon:key", "quote\"key", std::string("nul\0key", 7),
             std::string(1, '\x7f'), std::string(1, '\xff'), std::string(4097, 'a'),
             "=secret", "====", "secret=middle", "secret==middle", "secret=+", "secret=/",
             "a" + std::string(4096, '=')}) {
        try {
            saga::api::Server invalid(listener_config(), key);
            throw std::runtime_error("unsupported key accepted");
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            check(message.find("[A-Za-z0-9._~+/-]") != std::string::npos &&
                  message.find("optional '=' padding") != std::string::npos &&
                  message.find("4096") != std::string::npos, "key grammar and bound are explicit in startup error");
            check(message.find("decod") == std::string::npos, "key error must not blame header decoding");
        }
    }
    auto remote = listener_config();
    remote.server.host = "0.0.0.0";
    rejects([&] { saga::api::Server invalid(remote); });
}
}

int main() {
    try {
        const auto defaults = saga::parse_config("");
        check(defaults.server.host == "127.0.0.1" && defaults.server.port == 8080, "safe network defaults");
        check(defaults.artifact.empty() && !defaults.vision_enabled && defaults.speculation_mode == "off", "no enabled mock capabilities");
        const auto c = saga::parse_config(R"(
[server]
port = 0
http_workers = 2
[context]
max_tokens = 8192
kv_capacity_tokens = "auto"
[kv]
key_format = "f16"
value_format = "fp16"
)");
        check(c.server.port == 0 && c.server.http_workers == 2, "server settings");
        check(!c.kv_capacity_tokens.tokens && c.max_tokens.tokens == 8192, "auto is unresolved");
        check(c.kv_pair.key == saga::kv::Codec::fp16 && c.kv_pair.value == saga::kv::Codec::fp16, "f16 alias normalization");
        const auto bf = saga::parse_config("[kv]\nkey_format='bf16'\nvalue_format='bf16'\n");
        check(bf.kv_pair.key != c.kv_pair.key, "BF16 is not FP16");
        const auto overridden = cli({"--context", "16384", "--kv-capacity", "auto", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1", "--port", "0"});
        check(overridden.config.server.port == 0 && overridden.config.max_tokens.tokens == 16384, "CLI override values");
        check(overridden.config.kv_pair.value == saga::kv::Codec::q5_1, "curated mixed pair");
        check(cli({"--help"}).help && cli({"--version"}).version && cli({"--check-config"}).check_config, "CLI modes");
        for (const std::string& invalid : std::vector<std::string>{
                 "bad = 1", "server = 1", "[server]\nport = -1", "[server]\nport = 65536",
                 "[server]\nport = '8080'", "[server]\nport = 1.0", "[server]\nhttp_workers = 0",
                 "[server]\nmax_pending_connections = 0", "[server]\nmax_active_requests = 2",
                 "[server]\nmax_body_bytes = 0", "[server]\nio_timeout_seconds = 0", "[server]\napi_key_env='a=b'",
                 "[server]\nunknown=1", "[context]\nmax_tokens=0", "[context]\nmax_tokens=262145",
                 "[context]\nmax_tokens='8192'", "[context]\nmax_tokens=8192\nkv_capacity_tokens=4096",
                 "[context]\nprefill_chunk_tokens = 0", "[context]\nmax_tokens=true", "[model]\nruntime='unknown'",
                 "[backend]\ntype='cpu'", "[speculation]\nmode='ngram'", "[speculation]\nmax_draft_tokens=4",
                 "[vision]\nenabled='yes'", "[vision]\ncpu_threads=0", "[vision]\ndevice='auto'",
                 "[kv]\nkey_format='Q3_K'", "[kv]\nkey_format='fp16'\nvalue_format='q4_0'", "[server", "port=\"secret\""})
            rejects([&] { (void)saga::parse_config(invalid); });
        rejects([] { (void)saga::parse_config(std::string(65537, ' ')); });
        rejects([] { (void)cli({"--unknown"}); });
        rejects([] { (void)cli({"--port"}); });
        rejects([] { (void)cli({"--port", "-1"}); });
        rejects([] { (void)cli({"--port", "1x"}); });
        rejects([] { (void)cli({"--port", "65536"}); });
        rejects([] { (void)cli({"--context", "18446744073709551616"}); });
        rejects([] { (void)cli({"--config", "one", "--config", "two"}); });
        check(saga::is_loopback_binding("::1") && saga::is_loopback_binding("127.0.0.1"), "literal loopback");
        check(!saga::is_loopback_binding("localhost") && !saga::is_loopback_binding("0.0.0.0"), "do not trust DNS names as loopback");
        merged_config_tests();
        key_tests();
        lifecycle_tests();
        std::cout << "PASS: strict TOML, merged CLI validation, bearer alphabet and in-process listener lifecycle\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
