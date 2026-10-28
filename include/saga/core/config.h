#pragma once

#include "saga/kv/catalog.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>


namespace saga {

struct Capacity {
    std::optional<std::uint64_t> tokens{8192}; // nullopt means auto, not an allocation approval
    bool operator==(const Capacity&) const = default;
};

struct ServerConfig {
    std::string host{"127.0.0.1"};
    std::uint16_t port{8080};
    std::uint32_t http_workers{4};
    std::uint32_t max_pending_connections{32};
    std::uint32_t max_active_requests{1};
    std::uint32_t max_queued_requests{16};
    std::uint64_t max_body_bytes{1024 * 1024};
    std::uint32_t io_timeout_seconds{2};
    std::string api_key_env{"SAGA_API_KEY"};
};

struct Config {
    ServerConfig server;
    std::string model_runtime{"qwen3_8_27b"};
    std::string artifact;
    std::string public_name{"qwen3.8-27b-gsq3"};
    std::string backend_type{"cuda"};
    std::uint32_t device{0};
    std::string tuning_profile{"auto"};
    Capacity max_tokens;
    Capacity kv_capacity_tokens;
    std::uint32_t prefill_chunk_tokens{512};
    kv::Pair kv_pair;
    std::string speculation_mode{"off"};
    std::uint32_t max_draft_tokens{3};
    bool vision_enabled{false};
    std::string vision_device{"cpu"};
    std::uint32_t vision_cpu_threads{4};
    std::uint32_t max_images_per_request{4};

    Config();
};

struct CommandLine {
    Config config;
    bool help{false};
    bool version{false};
    bool check_config{false};
};

// Standalone parsing/loading includes semantic validation against defaults.
Config parse_config(std::string_view toml);
Config load_config(const std::filesystem::path& path);
// Merge order is defaults, typed TOML, explicit model-config INI fields, then CLI overrides;
// validate semantic constraints once after the complete merge.
CommandLine parse_command_line(std::span<const std::string_view> arguments);
void validate(const Config&);
bool is_loopback_binding(std::string_view host) noexcept;
std::string usage();

} // namespace saga
