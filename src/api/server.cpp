#include "saga/api/server.h"
#include "saga/core/log.h"

#include <httplib.h>
#include <json.hpp>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <functional>
#include <stdexcept>
#include <utility>

namespace saga::api {
namespace {
void error(httplib::Response& response, int status, std::string_view code,
           std::string_view message, std::string_view type = "server_error") {
    response.status = status;
    response.set_content(nlohmann::json{{"error", {{"message", message}, {"type", type},
        {"param", nullptr}, {"code", code}}}}.dump(), "application/json");
}
bool ascii_equal(std::string_view left, std::string_view right) noexcept {
    if (left.size() != right.size()) return false;
    const auto lower = [](unsigned char c) {
        return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
    };
    for (std::size_t i = 0; i < left.size(); ++i)
        if (lower(left[i]) != lower(right[i])) return false;
    return true;
}
bool valid_bearer_token(std::string_view token) noexcept {
    if (token.empty() || token.size() > 4096) return false;
    const auto padding = token.find('=');
    const auto value = token.substr(0, padding);
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
               (c >= '0' && c <= '9') || std::string_view("._~+/-").find(c) != std::string_view::npos;
    })) return false;
    return padding == std::string_view::npos || token.find_first_not_of('=', padding) == std::string_view::npos;
}
bool same_key(std::string_view input, std::string_view expected) noexcept {
    if (input.size() != expected.size()) return false;
    unsigned difference = 0;
    for (std::size_t i = 0; i < input.size(); ++i)
        difference |= static_cast<unsigned char>(input[i]) ^ static_cast<unsigned char>(expected[i]);
    return difference == 0;
}
bool bearer_matches(std::string_view authorization, std::string_view expected) noexcept {
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() <= scheme.size() ||
        !ascii_equal(authorization.substr(0, scheme.size()), scheme) ||
        authorization[scheme.size()] != ' ') return false;
    const auto token = authorization.find_first_not_of(' ', scheme.size());
    return token != std::string_view::npos && same_key(authorization.substr(token), expected);
}
// stop() and the vendor destructor do not close a bound, not-yet-running listener.
// Use the protected socket handle only while the listener thread is quiescent.
class OwnedListener final : public httplib::Server {
public:
    ~OwnedListener() override { close_unused_listener(); }

    void close_unused_listener() noexcept {
        const auto socket = svr_sock_.exchange(INVALID_SOCKET);
        if (socket == INVALID_SOCKET) return;
#ifdef _WIN32
        ::closesocket(socket);
#else
        ::close(socket);
#endif
    }

    bool run_owned() {
        try {
            const bool success = listen_after_bind();
            // On normal return the vendor has closed the listener, including on
            // accept failure (which leaves a stale handle). Never close it twice.
            svr_sock_.store(INVALID_SOCKET);
            return success;
        } catch (...) {
            close_unused_listener();
            throw;
        }
    }
};

// TaskQueue callbacks run on the listener thread, so transport stop never races
// startup, accept-error cleanup, or another caller of transport stop().
class InterruptibleQueue final : public httplib::TaskQueue {
public:
    InterruptibleQueue(std::size_t workers, std::size_t pending, httplib::Server& http,
                       const std::atomic<bool>& stop_requested)
        : http_(http), stop_requested_(stop_requested), pool_(workers, pending) {}
    ~InterruptibleQueue() override { shutdown(); }
    bool enqueue(std::function<void()> task) override {
        if (cancelled()) return false;
        return pool_.enqueue(std::move(task));
    }
    void on_idle() override { (void)cancelled(); }
    void shutdown() override {
        if (!joined_) {
            pool_.shutdown();
            joined_ = true;
        }
    }
private:
    bool cancelled() {
        if (!stop_requested_.load()) return false;
        http_.stop();
        return true;
    }
    httplib::Server& http_;
    const std::atomic<bool>& stop_requested_;
    httplib::ThreadPool pool_;
    bool joined_{false};
};

bool reject_framing(const httplib::Request& request, httplib::Response& response) {
    const auto lengths = request.get_header_value_count("Content-Length");
    const auto encodings = request.get_header_value_count("Transfer-Encoding");
    // Reject all duplicate lengths, even matching ones, rather than normalize a
    // potentially ambiguous representation. Only the supported chunked TE is accepted.
    bool invalid = lengths > 1 || encodings > 1 || (lengths && encodings);
    if (lengths == 1) {
        const auto length = request.get_header_value("Content-Length");
        std::uint64_t value = 0;
        const auto [end, status] = std::from_chars(length.data(), length.data() + length.size(), value);
        invalid |= length.empty() || status != std::errc{} || end != length.data() + length.size();
    }
    if (encodings == 1)
        invalid |= !ascii_equal(request.get_header_value("Transfer-Encoding"), "chunked");
    if (!invalid) return false;
    error(response, 400, "invalid_request_framing", "Ambiguous or unsupported HTTP request framing", "invalid_request_error");
    response.set_header("Connection", "close");
    return true;
}
} // namespace

struct Server::Impl {
    Config config;
    std::string bearer_key;
    OwnedListener http;
    std::atomic<bool> stop_requested{false};
    std::atomic<std::uint64_t> requests{0};
    std::uint16_t bound_port{0};
    bool bound{false};
    bool started{false}; // lifecycle methods bind/run are owned by the main thread

    Impl(Config options, std::string key) : config(std::move(options)), bearer_key(std::move(key)) {
        validate(config);
        if (!is_loopback_binding(config.server.host) && bearer_key.empty())
            throw std::invalid_argument("remote binding requires a bearer key from server.api_key_env");
        if (!bearer_key.empty() && !valid_bearer_token(bearer_key))
            throw std::invalid_argument("bearer key must be an RFC 6750 b64token: one or more ASCII bytes from "
                                        "[A-Za-z0-9._~+/-] followed by optional '=' padding (max 4096 bytes)");
        if (!config.artifact.empty())
            throw std::invalid_argument("model.artifact loading is not implemented in this foundation increment");
        http.new_task_queue = [this] {
            return new InterruptibleQueue(config.server.http_workers,
                                          config.server.max_pending_connections, http, stop_requested);
        };
        http.set_payload_max_length(static_cast<std::size_t>(config.server.max_body_bytes));
        http.set_read_timeout(config.server.io_timeout_seconds);
        http.set_write_timeout(config.server.io_timeout_seconds);
        http.set_keep_alive_timeout(1);
        // This vendor does not end its read loop for a response's Connection: close.
        // One request per socket ensures an unread rejected body is never routed.
        http.set_keep_alive_max_count(1);
        http.set_idle_interval(0, 100000);
        http.set_default_headers({{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
        // Expect handling precedes pre-routing; reject before sending 100 Continue.
        http.set_expect_100_continue_handler([](const httplib::Request& request, httplib::Response& response) {
            return reject_framing(request, response) ? 400 : 100;
        });
        http.set_pre_routing_handler([this](const httplib::Request& request, httplib::Response& response) {
            // In this vendor pre-routing precedes every content reader/body read.
            if (reject_framing(request, response)) return httplib::Server::HandlerResponse::Handled;
            requests.fetch_add(1, std::memory_order_relaxed);
            // Liveness discloses no configuration. All other routes require auth when a key is configured.
            if (request.path != "/healthz" && !bearer_key.empty()) {
                const auto values = request.get_header_value_count("Authorization");
                const auto authorization = request.get_header_value("Authorization");
                if (values != 1 || !bearer_matches(authorization, bearer_key)) {
                    error(response, 401, "invalid_api_key", "Missing or invalid bearer authentication", "authentication_error");
                    response.set_header("WWW-Authenticate", "Bearer");
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        http.Get("/healthz", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(R"({"status":"ok"})", "application/json");
        });
        http.Get("/readyz", [](const httplib::Request&, httplib::Response& response) {
            error(response, 503, "model_not_ready", "No model is loaded; inference is not implemented in this increment");
        });
        http.Get("/v1/models", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(R"({"object":"list","data":[]})", "application/json");
        });
        http.Post("/v1/chat/completions", [](const httplib::Request&, httplib::Response& response) {
            error(response, 503, "model_not_ready", "No model is loaded; generation and streaming are unavailable");
            response.set_header("Retry-After", "5");
        });
        http.Get("/metrics", [this](const httplib::Request&, httplib::Response& response) {
            response.set_content("# HELP saga_engine_ready Reconciled model serving readiness.\n"
                "# TYPE saga_engine_ready gauge\nsaga_engine_ready 0\n"
                "# HELP saga_http_requests_total Requests reaching the routing layer.\n"
                "# TYPE saga_http_requests_total counter\nsaga_http_requests_total " +
                std::to_string(requests.load(std::memory_order_relaxed)) + "\n", "text/plain; version=0.0.4");
        });
        http.set_error_handler([](const httplib::Request&, httplib::Response& response) {
            if (!response.body.empty()) return;
            if (response.status == 413) error(response, 413, "request_too_large", "Request body exceeds configured limit", "invalid_request_error");
            else if (response.status == 404) error(response, 404, "not_found", "Unknown endpoint", "invalid_request_error");
            else error(response, response.status, "http_error", "HTTP request rejected", "invalid_request_error");
        });
        http.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr) {
            error(response, 500, "internal_error", "Internal request handling error");
            log_event("error", "http_exception");
        });
    }
};

Server::Server(Config config, std::string key) : impl_(std::make_unique<Impl>(std::move(config), std::move(key))) {}
Server::~Server() = default;

std::uint16_t Server::bind() {
    if (impl_->bound || impl_->started) throw std::logic_error("server can be bound only once");
    const auto& s = impl_->config.server;
    try {
        if (s.port == 0) {
            const int port = impl_->http.bind_to_any_port(s.host);
            if (port <= 0 || port > 65535) throw std::runtime_error("HTTP listener bind failed");
            impl_->bound_port = static_cast<std::uint16_t>(port);
        } else {
            if (!impl_->http.bind_to_port(s.host, s.port)) throw std::runtime_error("HTTP listener bind failed");
            impl_->bound_port = s.port;
        }
        impl_->bound = true;
        return impl_->bound_port;
    } catch (...) {
        impl_->http.close_unused_listener();
        throw;
    }
}

bool Server::run() {
    if (!impl_->bound || impl_->started) throw std::logic_error("server must be bound and run only once");
    impl_->started = true;
    if (impl_->stop_requested.load()) {
        impl_->http.close_unused_listener();
        return true;
    }
    return impl_->http.run_owned();
}

void Server::stop() { impl_->stop_requested.store(true); }
bool Server::is_running() const { return impl_->http.is_running(); }
} // namespace saga::api
