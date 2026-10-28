#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>

SOCKET saga_http_transport_test_ws_accept(SOCKET socket,
                                          struct sockaddr *address,
                                          LPINT address_length,
                                          LPCONDITIONPROC condition,
                                          DWORD callback_data);
#define WSAAccept saga_http_transport_test_ws_accept
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int saga_http_transport_test_accept(int socket, struct sockaddr *address,
                                    socklen_t *address_length);
#define accept saga_http_transport_test_accept
#if defined(SOCK_CLOEXEC)
int saga_http_transport_test_accept4(int socket, struct sockaddr *address,
                                     socklen_t *address_length, int flags);
#define accept4 saga_http_transport_test_accept4
#endif
#endif

#include <httplib.h>

#if defined(_WIN32)
#undef WSAAccept
#else
#undef accept
#if defined(SOCK_CLOEXEC)
#undef accept4
#endif
#endif

namespace {

std::atomic<unsigned> accept_attempts{0};

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

void check(bool condition, std::string_view message) {
  if (!condition) { fail(message); }
}

void parse_header_into(httplib::Request &request, const std::string &line) {
  bool callback_called = false;
  const auto parsed = httplib::detail::parse_header(
      line.data(), line.data() + line.size(),
      [&](const std::string &key, const std::string &value) {
        request.headers.emplace(key, value);
        callback_called = true;
      });
  check(parsed, "parse_header rejected a syntactically valid field");
  check(callback_called, "parse_header did not emit the parsed field");
}

void test_header_values_remain_raw() {
  httplib::Request request;
  parse_header_into(request, "Authorization: Bearer alpha%2Fbeta%2Bgamma");
  parse_header_into(request, "Proxy-Authorization: Basic user%3Asecret");
  parse_header_into(request, "Content-Length: 12%2F");
  parse_header_into(request, "Transfer-Encoding: chunked%2F");

  check(request.get_header_value("Authorization") ==
            "Bearer alpha%2Fbeta%2Bgamma",
        "Authorization header value was URI-decoded");
  check(request.get_header_value("Proxy-Authorization") ==
            "Basic user%3Asecret",
        "Proxy-Authorization header value was URI-decoded");
  check(request.get_header_value("Content-Length") == "12%2F",
        "Content-Length header value was URI-decoded");
  check(request.get_header_value("Transfer-Encoding") == "chunked%2F",
        "Transfer-Encoding header value was URI-decoded");
}

struct ChunkRead {
  httplib::detail::ReadContentResult result;
  std::string body;
  httplib::Headers trailers;
};

ChunkRead read_chunked(std::string_view wire, size_t payload_limit,
                       std::string declared_trailers = {}) {
  httplib::detail::BufferStream stream;
  const std::string input(wire);
  const auto written = stream.write(input.data(), input.size());
  check(written == static_cast<ssize_t>(input.size()),
        "BufferStream did not accept the complete test message");

  httplib::Request request;
  if (!declared_trailers.empty()) {
    request.headers.emplace("Trailer", std::move(declared_trailers));
  }

  std::string body;
  const auto result = httplib::detail::read_content_chunked(
      stream, request, payload_limit,
      [&](const char *data, size_t length, size_t, size_t) {
        body.append(data, length);
        return true;
      });
  return {result, std::move(body), std::move(request.trailers)};
}

void expect_chunk_result(std::string_view wire, size_t payload_limit,
                         httplib::detail::ReadContentResult expected,
                         std::string_view message) {
  const auto actual = read_chunked(wire, payload_limit);
  check(actual.result == expected, message);
}

void test_chunked_content() {
  using httplib::detail::ReadContentResult;

  const auto plain = read_chunked("3\r\nabc\r\n0\r\n\r\n", 3);
  check(plain.result == ReadContentResult::Success,
        "valid chunked body did not succeed");
  check(plain.body == "abc", "valid chunked body payload differed");

  const auto extensions = read_chunked(
      "3 ; foo = \"bar\\\"baz\"\r\nabc\r\n0\r\n\r\n", 3);
  check(extensions.result == ReadContentResult::Success,
        "valid token and quoted chunk extensions were rejected");
  check(extensions.body == "abc", "extended chunk payload differed");

  const auto with_trailer = read_chunked(
      "3;checksum=ok\r\nabc\r\n0\r\nX-Checksum: done%2F\r\n\r\n", 3,
      "X-Checksum");
  check(with_trailer.result == ReadContentResult::Success,
        "valid chunk extension and declared trailer did not succeed");
  check(with_trailer.body == "abc", "trailer-bearing payload differed");
  const auto trailer = with_trailer.trailers.find("X-Checksum");
  check(trailer != with_trailer.trailers.end(),
        "declared trailer was not retained");
  check(trailer->second == "done%2F", "trailer value was URI-decoded");

  const std::vector<std::string> malformed_sizes = {
      "+0\r\n\r\n",
      "+1\r\nx\r\n0\r\n\r\n",
      "-1\r\nx\r\n0\r\n\r\n",
      "0x0\r\n\r\n",
      "0garbage\r\n\r\n",
      " 1\r\nx\r\n0\r\n\r\n",
      "1x\r\nx\r\n0\r\n\r\n",
      "1;=bad\r\nx\r\n0\r\n\r\n",
      "1;foo=\r\nx\r\n0\r\n\r\n",
      "1;foo=\"unterminated\r\nx\r\n0\r\n\r\n",
      "1;foo=\"dangling\\\r\nx\r\n0\r\n\r\n",
      "1\nx\r\n0\r\n\r\n",
  };
  for (const auto &wire : malformed_sizes) {
    expect_chunk_result(wire, 16, ReadContentResult::Error,
                        "malformed chunk size or extension was accepted");
  }

  char max_hex[2 * sizeof(size_t) + 1]{};
  const auto converted = std::to_chars(
      max_hex, max_hex + sizeof(max_hex),
      (std::numeric_limits<size_t>::max)(), 16);
  check(converted.ec == std::errc{},
        "could not format size_t maximum for overflow regression case");
  std::string overflowing_size(max_hex, converted.ptr);
  overflowing_size.push_back('0');
  expect_chunk_result(overflowing_size + "\r\n", 16,
                      ReadContentResult::Error,
                      "chunk size arithmetic overflow was accepted");

  const auto exact_limit = read_chunked("3\r\nabc\r\n0\r\n\r\n", 3);
  check(exact_limit.result == ReadContentResult::Success,
        "payload exactly at the limit was rejected");
  const auto over_limit = read_chunked("4\r\nabcd\r\n0\r\n\r\n", 3);
  check(over_limit.result == ReadContentResult::PayloadTooLarge,
        "payload over the limit did not report PayloadTooLarge");
  check(over_limit.body.empty(),
        "over-limit first chunk was delivered to the receiver");
  const auto cumulative_over_limit =
      read_chunked("2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n", 3);
  check(cumulative_over_limit.result == ReadContentResult::PayloadTooLarge,
        "cumulative payload over the limit did not report PayloadTooLarge");
  check(cumulative_over_limit.body == "ab",
        "payload prefix before the oversized chunk differed");

  const std::vector<std::string> truncated_messages = {
      "3\r\nab", "1\r\nx", "1\r\nxZZ0\r\n\r\n", "0\r\n",
      "0\r\nX-Checksum: done\r\n"};
  for (const auto &wire : truncated_messages) {
    expect_chunk_result(wire, 16, ReadContentResult::Error,
                        "invalid chunk terminator or truncated data succeeded");
  }

  expect_chunk_result("0\r\nNot-A-Header\r\n\r\n", 16,
                      ReadContentResult::Error,
                      "malformed trailer field was accepted");

  static_assert(CPPHTTPLIB_HEADER_MAX_COUNT > 0,
                "trailer boundary test requires a positive header limit");
  constexpr size_t max_trailers =
      static_cast<size_t>(CPPHTTPLIB_HEADER_MAX_COUNT);
  std::string over_limit_trailers = "0\r\n";
  for (size_t i = 0; i <= max_trailers; ++i) {
    over_limit_trailers += "X-Unlisted-" + std::to_string(i) + ": v\r\n";
  }
  over_limit_trailers += "\r\n";
  expect_chunk_result(over_limit_trailers, 0, ReadContentResult::Error,
                      "too many valid undeclared trailers were accepted");

  std::string at_limit_trailers = "0\r\nX-Keep: retained\r\n";
  for (size_t i = 1; i < max_trailers; ++i) {
    at_limit_trailers += "X-Unlisted-" + std::to_string(i) + ": v\r\n";
  }
  at_limit_trailers += "\r\n";
  const auto at_limit = read_chunked(at_limit_trailers, 0, "X-Keep");
  check(at_limit.result == ReadContentResult::Success,
        "exactly the allowed number of mixed trailers was rejected");
  const auto kept_trailer = at_limit.trailers.find("X-Keep");
  check(kept_trailer != at_limit.trailers.end() &&
            kept_trailer->second == "retained",
        "declared trailer was not preserved at the count boundary");

  const char nul_size_bytes[] = "1\0\r\nx\r\n0\r\n\r\n";
  const std::string nul_size_wire(nul_size_bytes, sizeof(nul_size_bytes) - 1);
  expect_chunk_result(nul_size_wire, 16, ReadContentResult::Error,
                      "embedded NUL in chunk-size line was accepted");

  const char nul_separator_bytes[] = "1\r\nx\r\0\n0\r\n\r\n";
  const std::string nul_separator_wire(nul_separator_bytes,
                                       sizeof(nul_separator_bytes) - 1);
  expect_chunk_result(nul_separator_wire, 16, ReadContentResult::Error,
                      "embedded NUL in chunk-data separator was accepted");

  const char nul_final_trailer_bytes[] =
      "0\r\nX-Checksum: done\r\n\0\r\n";
  const std::string nul_final_trailer_wire(
      nul_final_trailer_bytes, sizeof(nul_final_trailer_bytes) - 1);
  expect_chunk_result(nul_final_trailer_wire, 16, ReadContentResult::Error,
                      "embedded NUL in final trailer line was accepted");
}

#if defined(_WIN32)
using test_socket_t = SOCKET;

class SocketRuntime {
public:
  SocketRuntime() {
    WSADATA data{};
    check(WSAStartup(MAKEWORD(2, 2), &data) == 0,
          "WSAStartup failed for EMFILE listener regression test");
  }
  ~SocketRuntime() { WSACleanup(); }
};

void close_test_socket(test_socket_t socket) { closesocket(socket); }
#else
using test_socket_t = int;

void close_test_socket(test_socket_t socket) { ::close(socket); }
#endif

class PendingLoopbackClient {
public:
  explicit PendingLoopbackClient(int port) {
    socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
#if defined(_WIN32)
    check(socket_ != INVALID_SOCKET,
          "could not create loopback client socket");
#else
    check(socket_ >= 0, "could not create loopback client socket");
#endif

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    const auto result = ::connect(
        socket_, reinterpret_cast<struct sockaddr *>(&address),
        static_cast<socklen_t>(sizeof(address)));
#if defined(_WIN32)
    if (result == SOCKET_ERROR) {
#else
    if (result != 0) {
#endif
      close_test_socket(socket_);
#if defined(_WIN32)
      socket_ = INVALID_SOCKET;
#else
      socket_ = -1;
#endif
      fail("could not connect pending loopback client");
    }
  }

  ~PendingLoopbackClient() {
#if defined(_WIN32)
    if (socket_ != INVALID_SOCKET) { close_test_socket(socket_); }
#else
    if (socket_ >= 0) { close_test_socket(socket_); }
#endif
  }

  PendingLoopbackClient(const PendingLoopbackClient &) = delete;
  PendingLoopbackClient &operator=(const PendingLoopbackClient &) = delete;

private:
  test_socket_t socket_{};
};

class CancellationObservingQueue final : public httplib::TaskQueue {
public:
  CancellationObservingQueue(httplib::Server &server,
                             std::atomic<unsigned> &idle_calls,
                             std::atomic<bool> &stop_claimed,
                             std::atomic<bool> &stopped_from_idle)
      : server_(server), idle_calls_(idle_calls), stop_claimed_(stop_claimed),
        stopped_from_idle_(stopped_from_idle) {}

  bool enqueue(std::function<void()>) override { return false; }
  void shutdown() override {}

  void on_idle() override {
    idle_calls_.fetch_add(1, std::memory_order_relaxed);
    if (accept_attempts.load(std::memory_order_relaxed) != 0) {
      bool expected = false;
      if (stop_claimed_.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel)) {
        server_.stop();
        stopped_from_idle_.store(true, std::memory_order_release);
      }
    }
  }

private:
  httplib::Server &server_;
  std::atomic<unsigned> &idle_calls_;
  std::atomic<bool> &stop_claimed_;
  std::atomic<bool> &stopped_from_idle_;
};

bool wait_until(const std::function<bool()> &predicate,
                std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return predicate();
}

void test_emfile_accept_cancellation() {
#if defined(_WIN32)
  SocketRuntime socket_runtime;
  (void)socket_runtime;
#endif

  accept_attempts.store(0, std::memory_order_relaxed);
  std::atomic<unsigned> idle_calls{0};
  std::atomic<bool> stop_claimed{false};
  std::atomic<bool> stopped_from_idle{false};
  std::atomic<bool> listener_finished{false};
  std::atomic<bool> listen_result{false};
  std::exception_ptr listener_exception;

  httplib::Server server;
  const auto port = server.bind_to_any_port("127.0.0.1");
  check(port > 0, "could not bind loopback listener for EMFILE regression test");
  server.set_idle_interval(0, 100000);
  server.new_task_queue = [&] {
    return new CancellationObservingQueue(server, idle_calls, stop_claimed,
                                          stopped_from_idle);
  };
  PendingLoopbackClient pending_client(port);

  std::thread listener([&] {
    try {
      listen_result.store(server.listen_after_bind(), std::memory_order_release);
    } catch (...) {
      listener_exception = std::current_exception();
    }
    listener_finished.store(true, std::memory_order_release);
  });


  const bool callback_stopped_listener = wait_until(
      [&] { return stopped_from_idle.load(std::memory_order_acquire) ||
                   listener_finished.load(std::memory_order_acquire); },
      std::chrono::seconds(3));
  if (!callback_stopped_listener &&
      !listener_finished.load(std::memory_order_acquire)) {
    // A bounded fallback prevents a regressed accept loop from hanging this
    // test forever; the normal stop path is the listener-thread on_idle callback.
    bool expected = false;
    if (stop_claimed.compare_exchange_strong(expected, true,
                                             std::memory_order_acq_rel)) {
      server.stop();
    }
  }
  listener.join();

  if (listener_exception) { std::rethrow_exception(listener_exception); }
  check(accept_attempts.load(std::memory_order_relaxed) > 0,
        "synthetic accept wrapper was not reached");
  check(idle_calls.load(std::memory_order_relaxed) > 0,
        "TaskQueue::on_idle was never called");
  check(stopped_from_idle.load(std::memory_order_acquire),
        "on_idle did not cancel the listener after the EMFILE accept");
  check(listen_result.load(std::memory_order_acquire),
        "listen_after_bind reported failure after normal cancellation");
}

} // namespace

#if defined(_WIN32)
SOCKET saga_http_transport_test_ws_accept(SOCKET, struct sockaddr *, LPINT,
                                          LPCONDITIONPROC, DWORD) {
  accept_attempts.fetch_add(1, std::memory_order_relaxed);
  WSASetLastError(WSAEMFILE);
  return INVALID_SOCKET;
}
#else
int saga_http_transport_test_accept(int, struct sockaddr *, socklen_t *) {
  accept_attempts.fetch_add(1, std::memory_order_relaxed);
  errno = EMFILE;
  return -1;
}
#if defined(SOCK_CLOEXEC)
int saga_http_transport_test_accept4(int, struct sockaddr *, socklen_t *, int) {
  accept_attempts.fetch_add(1, std::memory_order_relaxed);
  errno = EMFILE;
  return -1;
}
#endif
#endif

int main() {
  int failures = 0;
  const auto run = [&failures](std::string_view name, auto test) {
    try {
      test();
    } catch (const std::exception &error) {
      std::cerr << name << " failed: " << error.what() << '\n';
      ++failures;
    }
  };

  run("raw header values", test_header_values_remain_raw);
  run("EMFILE accept cancellation", test_emfile_accept_cancellation);
  run("chunked content", test_chunked_content);

  if (failures != 0) {
    std::cerr << failures << " HTTP transport regression group(s) failed\n";
    return 1;
  }
  std::cout << "HTTP transport hardening regressions passed\n";
  return 0;
}
