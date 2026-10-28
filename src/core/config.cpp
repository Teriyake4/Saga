#include "saga/core/config.h"
#include "saga/core/model_config.h"

#include <toml.hpp>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace saga {
namespace {
[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("configuration: " + message);
}
void known_keys(const toml::table& table, std::initializer_list<std::string_view> allowed,
                std::string_view prefix) {
    for (const auto& [key, value] : table) {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), key.str()) == allowed.end())
            invalid("unknown field " + std::string(prefix) + std::string(key.str()));
    }
}
const toml::table* section(const toml::table& root, std::string_view key) {
    const auto* node = root.get(key);
    if (!node) return nullptr;
    const auto* table = node->as_table();
    if (!table) invalid(std::string(key) + " must be a table");
    return table;
}
void text(const toml::table& t, std::string_view key, std::string& output) {
    if (const auto* n = t.get(key)) {
        if (!n->is_string()) invalid(std::string(key) + " must be a string");
        output = *n->value<std::string>();
    }
}
void boolean(const toml::table& t, std::string_view key, bool& output) {
    if (const auto* n = t.get(key)) {
        if (!n->is_boolean()) invalid(std::string(key) + " must be a boolean");
        output = *n->value<bool>();
    }
}
template <typename T>
void number(const toml::table& t, std::string_view key, T& output) {
    if (const auto* n = t.get(key)) {
        const auto value = n->value<std::int64_t>();
        if (!n->is_integer() || !value || *value < 0 ||
            static_cast<std::uint64_t>(*value) > std::numeric_limits<T>::max())
            invalid(std::string(key) + " must be an in-range unsigned integer");
        output = static_cast<T>(*value);
    }
}
void capacity(const toml::table& t, std::string_view key, Capacity& output) {
    if (const auto* n = t.get(key)) {
        if (n->is_string() && n->value<std::string>() == "auto") output.tokens.reset();
        else if (n->is_integer()) {
            const auto value = *n->value<std::int64_t>();
            if (value < 0) invalid(std::string(key) + " must be an unsigned integer or auto");
            output.tokens = static_cast<std::uint64_t>(value);
        } else invalid(std::string(key) + " must be a positive integer or auto");
    }
}
template <typename T> T unsigned_argument(std::string_view value, std::string_view option) {
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        value.empty() || parsed > std::numeric_limits<T>::max())
        invalid(std::string(option) + " requires an in-range unsigned integer");
    return static_cast<T>(parsed);
}
bool required_pair(kv::Pair pair) {
    const auto known = [](kv::Codec codec) {
        const auto catalog = kv::catalog();
        return std::any_of(catalog.begin(), catalog.end(),
                           [codec](const auto& entry) { return entry.id == codec; });
    };
    if (!known(pair.key) || !known(pair.value)) return false;
    if (pair.key == pair.value) return true;
    return (pair.key == kv::Codec::q8_0 && (pair.value == kv::Codec::q4_0 ||
             pair.value == kv::Codec::q5_0 || pair.value == kv::Codec::q5_1)) ||
           (pair.key == kv::Codec::int8_g64 && pair.value == kv::Codec::int4_g64);
}
void safe_label(std::string_view name, std::string_view value) {
    if (value.empty() || value.size() > 256 ||
        std::any_of(value.begin(), value.end(), [](unsigned char c) { return c <= 32 || c == 127; }))
        invalid(std::string(name) + " must be a nonempty label without whitespace/control bytes (max 256)");
}
} // namespace

Config::Config() : kv_pair{kv::Codec::int8_g64, kv::Codec::int8_g64} {}

bool is_loopback_binding(std::string_view host) noexcept {
    return host == "127.0.0.1" || host == "::1";
}

void validate(const Config& c) {
    safe_label("server.host", c.server.host);
    if (c.server.http_workers == 0 || c.server.http_workers > 64)
        invalid("server.http_workers must be 1..64");
    if (c.server.max_pending_connections == 0 || c.server.max_pending_connections > 4096)
        invalid("server.max_pending_connections must be 1..4096 (zero would disable the bound)");
    if (c.server.max_active_requests != 1) invalid("only one active generation request is supported");
    if (c.server.max_queued_requests > 4096) invalid("server.max_queued_requests must be 0..4096");
    if (c.server.max_body_bytes == 0 || c.server.max_body_bytes > 64 * 1024 * 1024)
        invalid("server.max_body_bytes must be 1..67108864");
    if (c.server.io_timeout_seconds == 0 || c.server.io_timeout_seconds > 30)
        invalid("server.io_timeout_seconds must be 1..30");
    safe_label("server.api_key_env", c.server.api_key_env);
    if (!std::all_of(c.server.api_key_env.begin(), c.server.api_key_env.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '_';
    })) invalid("server.api_key_env must name an environment variable");
    if (c.model_runtime != "qwen3_8_27b") invalid("unsupported model.runtime");
    safe_label("model.public_name", c.public_name);
    if (std::any_of(c.artifact.begin(), c.artifact.end(), [](unsigned char ch) {
        return ch == 0 || ch < 0x20 || ch == 0x7f;
    })) invalid("model.artifact contains a control byte");
    if (c.backend_type != "cuda") invalid("only the cuda production backend is planned; no CPU text backend");
    if (c.tuning_profile != "auto" && c.tuning_profile != "none") invalid("tuning.profile must be auto or none");
    for (const auto& cap : {c.max_tokens, c.kv_capacity_tokens})
        if (cap.tokens && (*cap.tokens == 0 || *cap.tokens > 262144))
            invalid("context capacity must be 1..262144 or auto (not a fit guarantee)");
    if (c.max_tokens.tokens && c.kv_capacity_tokens.tokens &&
        *c.max_tokens.tokens > *c.kv_capacity_tokens.tokens)
        invalid("context.max_tokens exceeds explicit context.kv_capacity_tokens");
    if (c.prefill_chunk_tokens == 0 || c.prefill_chunk_tokens > 8192)
        invalid("context.prefill_chunk_tokens must be 1..8192");
    if (!required_pair(c.kv_pair)) invalid("K/V pair is outside the required same-codec/curated mixed catalog");
    if (c.speculation_mode != "off" && c.speculation_mode != "mtp") invalid("speculation.mode must be off or mtp");
    if (c.max_draft_tokens == 0 || c.max_draft_tokens > 3) invalid("speculation.max_draft_tokens must be 1..3");
    if (c.vision_device != "cpu" && c.vision_device != "cuda") invalid("vision.device must be cpu or cuda");
    if (c.vision_cpu_threads == 0 || c.vision_cpu_threads > 256) invalid("vision.cpu_threads must be 1..256");
    if (c.max_images_per_request == 0 || c.max_images_per_request > 16) invalid("vision.max_images_per_request must be 1..16");
}

namespace {
// Decode structure/types first; CLI overrides may repair semantic conflicts in the file.
Config parse_config_fields(std::string_view input) {
    if (input.size() > 64 * 1024) invalid("TOML file exceeds 64 KiB");
    toml::table root;
    try { root = toml::parse(input); }
    catch (const toml::parse_error&) { invalid("invalid TOML syntax (configuration contents omitted)"); }
    known_keys(root, {"server", "model", "backend", "tuning", "context", "kv", "speculation", "vision"}, "");
    Config c;
    if (const auto* t = section(root, "server")) {
        known_keys(*t, {"host", "port", "http_workers", "max_pending_connections", "max_active_requests",
                       "max_queued_requests", "max_body_bytes", "io_timeout_seconds", "api_key_env"}, "server.");
        text(*t, "host", c.server.host); number(*t, "port", c.server.port);
        number(*t, "http_workers", c.server.http_workers);
        number(*t, "max_pending_connections", c.server.max_pending_connections);
        number(*t, "max_active_requests", c.server.max_active_requests);
        number(*t, "max_queued_requests", c.server.max_queued_requests);
        number(*t, "max_body_bytes", c.server.max_body_bytes);
        number(*t, "io_timeout_seconds", c.server.io_timeout_seconds);
        text(*t, "api_key_env", c.server.api_key_env);
    }
    if (const auto* t = section(root, "model")) {
        known_keys(*t, {"runtime", "artifact", "public_name"}, "model.");
        text(*t, "runtime", c.model_runtime); text(*t, "artifact", c.artifact); text(*t, "public_name", c.public_name);
    }
    if (const auto* t = section(root, "backend")) {
        known_keys(*t, {"type", "device"}, "backend.");
        text(*t, "type", c.backend_type); number(*t, "device", c.device);
    }
    if (const auto* t = section(root, "tuning")) {
        known_keys(*t, {"profile"}, "tuning."); text(*t, "profile", c.tuning_profile);
    }
    if (const auto* t = section(root, "context")) {
        known_keys(*t, {"max_tokens", "kv_capacity_tokens", "prefill_chunk_tokens"}, "context.");
        capacity(*t, "max_tokens", c.max_tokens); capacity(*t, "kv_capacity_tokens", c.kv_capacity_tokens);
        number(*t, "prefill_chunk_tokens", c.prefill_chunk_tokens);
    }
    if (const auto* t = section(root, "kv")) {
        known_keys(*t, {"key_format", "value_format"}, "kv.");
        std::string k(kv::descriptor(c.kv_pair.key).name), v(kv::descriptor(c.kv_pair.value).name);
        text(*t, "key_format", k); text(*t, "value_format", v);
        c.kv_pair = {kv::parse_codec(k), kv::parse_codec(v)};
    }
    if (const auto* t = section(root, "speculation")) {
        known_keys(*t, {"mode", "max_draft_tokens"}, "speculation.");
        text(*t, "mode", c.speculation_mode); number(*t, "max_draft_tokens", c.max_draft_tokens);
    }
    if (const auto* t = section(root, "vision")) {
        known_keys(*t, {"enabled", "device", "cpu_threads", "max_images_per_request"}, "vision.");
        boolean(*t, "enabled", c.vision_enabled); text(*t, "device", c.vision_device);
        number(*t, "cpu_threads", c.vision_cpu_threads); number(*t, "max_images_per_request", c.max_images_per_request);
    }
    return c;
}

Config load_config_fields(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) invalid("cannot open TOML file");
    std::string contents(64 * 1024 + 1, '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (input.bad()) invalid("failed to read TOML file");
    contents.resize(static_cast<std::size_t>(input.gcount()));
    return parse_config_fields(contents);
}
} // namespace

Config parse_config(std::string_view input) {
    auto config = parse_config_fields(input);
    validate(config);
    return config;
}

Config load_config(const std::filesystem::path& path) {
    auto config = load_config_fields(path);
    validate(config);
    return config;
}

CommandLine parse_command_line(std::span<const std::string_view> args) {
    CommandLine result;
    std::vector<std::pair<std::string_view, std::string_view>> overrides;
    std::optional<std::string_view> path;
    std::optional<std::string_view> model_config_path;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto option = args[i];
        if (option == "--help") result.help = true;
        else if (option == "--version") result.version = true;
        else if (option == "--check-config") result.check_config = true;

        else {
            constexpr std::string_view options[]{"--config", "--model-config", "--host", "--port", "--cache-type-k", "--cache-type-v",
                "--context", "--kv-capacity", "--prefill-chunk", "--speculation", "--max-draft-tokens"};
            if (std::find(std::begin(options), std::end(options), option) == std::end(options))
                invalid("unknown command-line option");
            if (++i >= args.size()) invalid(std::string(option) + " requires a value");
            if (option == "--config") {
                if (path) invalid("--config may be specified only once");
                path = args[i];
            } else if (option == "--model-config") {
                if (model_config_path) invalid("--model-config may be specified only once");
                model_config_path = args[i];
            } else overrides.emplace_back(option, args[i]);
        }
    }
    if (result.help || result.version) return result;

    if (path) {
        const std::u8string utf8(path->begin(), path->end());
        result.config = load_config_fields(std::filesystem::path(utf8));
    }
    auto& c = result.config;
    if (model_config_path) {
        const std::u8string utf8(model_config_path->begin(), model_config_path->end());
        load_model_config(std::filesystem::path(utf8), c);
    }
    for (const auto& [option, value] : overrides) {
        if (option == "--host") c.server.host = value;
        else if (option == "--port") c.server.port = unsigned_argument<std::uint16_t>(value, option);
        else if (option == "--cache-type-k") c.kv_pair.key = kv::parse_codec(value);
        else if (option == "--cache-type-v") c.kv_pair.value = kv::parse_codec(value);
        else if (option == "--prefill-chunk") c.prefill_chunk_tokens = unsigned_argument<std::uint32_t>(value, option);
        else if (option == "--speculation") c.speculation_mode = value;
        else if (option == "--max-draft-tokens") c.max_draft_tokens = unsigned_argument<std::uint32_t>(value, option);
        else if (option == "--context" || option == "--kv-capacity") {
            auto& destination = option == "--context" ? c.max_tokens : c.kv_capacity_tokens;
            if (value == "auto") destination.tokens.reset();
            else destination.tokens = unsigned_argument<std::uint64_t>(value, option);
        }
    }
    validate(c);
    return result;
}

std::string usage() {
    return "Saga host foundation (no inference yet)\n"
        "Usage: saga-server [--config FILE] [--model-config FILE] [--host IP] [--port 0..65535]\n"
        "  --check-config             Validate static settings only, then exit\n"
        "  --model-config FILE        Apply one strict [model] INI profile\n"
        "  --cache-type-k/v CODEC     Select catalog metadata (not codec execution)\n"
        "  --context N|auto           Total request capacity candidate\n"
        "  --kv-capacity N|auto       Main KV capacity candidate\n"
        "  --prefill-chunk N          Bounded prompt chunk candidate\n"
        "  --speculation off|mtp      Capability request (not implemented yet)\n"
        "  --max-draft-tokens 1..3    Maximum draft width candidate\n"
        "  --help | --version\n"
        "Merge order: defaults, typed TOML, explicit model INI fields, then CLI overrides; validate once.\n"
        "Port 0 selects a free port.\n";
}
} // namespace saga
