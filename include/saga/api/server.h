#pragma once

#include "saga/core/config.h"

#include <cstdint>
#include <memory>
#include <string>

namespace saga::api {

// Model-free M1 transport shell. There is intentionally no readiness setter or fake backend.
class Server {
public:
    // Nonempty keys must be RFC 6750 b64tokens (max 4096 ASCII bytes):
    // one or more [A-Za-z0-9._~+/-] bytes followed only by optional '=' padding.
    explicit Server(Config config, std::string bearer_key = {});
    ~Server(); // releases an unused bound listener; must not overlap run()
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // bind/run/destruction have one owner; only stop/is_running may overlap run.
    std::uint16_t bind(); // port 0 selects an OS-assigned port
    bool run();          // blocks; bound server may be run once, including after stop()
    void stop();         // sticky cancellation; ordinary thread only, not a signal handler
    bool is_running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace saga::api
