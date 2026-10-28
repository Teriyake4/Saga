#include "saga/core/model_config.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <system_error>

namespace saga {
namespace {
[[noreturn]] void bad() { throw std::invalid_argument("configuration: invalid model config file"); }
std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return s;
}
bool valid_utf8(std::string_view s) {
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { ++i; continue; }
        unsigned n = 0;
        if (c >= 0xC2 && c <= 0xDF) n = 2;
        else if (c >= 0xE0 && c <= 0xEF) n = 3;
        else if (c >= 0xF0 && c <= 0xF4) n = 4;
        else return false;
        if (i + n > s.size()) return false;
        std::uint32_t cp = c & ((1u << (7 - n)) - 1);
        for (unsigned j = 1; j < n; ++j) {
            const auto d = static_cast<unsigned char>(s[i + j]);
            if ((d & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3F);
        }
        if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) ||
            cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF) ||
            (cp >= 0x80 && cp <= 0x9F)) return false;
        i += n;
    }
    return true;
}
std::uint64_t unsigned_value(std::string_view v) {
    std::uint64_t n{};
    const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
    if (v.empty() || ec != std::errc{} || p != v.data() + v.size()) bad();
    return n;
}
struct Patch { std::map<std::string, std::string> fields; };
std::filesystem::path path_from_utf8(std::string_view value) {
    std::u8string encoded;
    encoded.reserve(value.size());
    for (const unsigned char byte : value) encoded.push_back(static_cast<char8_t>(byte));
    return std::filesystem::path(encoded);
}
std::string path_to_utf8(const std::filesystem::path& p) {
    const auto encoded = p.generic_u8string();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}
void apply(Config& c, const Patch& patch, const std::filesystem::path& source) {
    for (const auto& [key, value] : patch.fields) {
        if (key == "model") {
            auto p = path_from_utf8(value);
            const bool drive_prefix = value.size() >= 2 &&
                ((value[0] >= 'A' && value[0] <= 'Z') || (value[0] >= 'a' && value[0] <= 'z')) && value[1] == ':';
            const bool drive_absolute = drive_prefix && value.size() >= 3 && (value[2] == '/' || value[2] == '\\');
            const bool unc = value.size() >= 2 && (value[0] == '/' || value[0] == '\\') && value[0] == value[1];
#ifdef _WIN32
            const bool root_relative = !value.empty() && (value[0] == '/' || value[0] == '\\') && !unc;
            if ((drive_prefix && !drive_absolute) || root_relative) bad();
            if (!p.is_absolute() && !drive_absolute && !unc) {
                std::error_code ec;
                const auto absolute_source = std::filesystem::absolute(source, ec);
                if (ec) bad();
                p = absolute_source.parent_path() / p;
            }
#else
            (void)drive_prefix;
            (void)drive_absolute;
            (void)unc;
            if (!p.is_absolute()) {
                std::error_code ec;
                const auto absolute_source = std::filesystem::absolute(source, ec);
                if (ec) bad();
                p = absolute_source.parent_path() / p;
            }
#endif
            c.artifact = path_to_utf8(p.lexically_normal());
        } else if (key == "alias") c.public_name = value;
        else if (key == "runtime") c.model_runtime = value;
        else if (key == "backend") c.backend_type = value;
        else if (key == "tuning") c.tuning_profile = value;
        else if (key == "speculation") c.speculation_mode = value;
        else if (key == "vision-device") c.vision_device = value;
        else if (key == "ctx-size") {
            if (value == "auto") c.max_tokens.tokens.reset();
            else c.max_tokens.tokens = unsigned_value(value);
        } else if (key == "kv-capacity") {
            if (value == "auto") c.kv_capacity_tokens.tokens.reset();
            else c.kv_capacity_tokens.tokens = unsigned_value(value);
        } else if (key == "cache-type-k" || key == "cache-type-v") {
            try {
                if (key == "cache-type-k") c.kv_pair.key = kv::parse_codec(value);
                else c.kv_pair.value = kv::parse_codec(value);
            } catch (const std::exception&) { bad(); }
        } else if (key == "device" || key == "prefill-chunk" || key == "max-draft-tokens" ||
                   key == "vision-cpu-threads" || key == "max-images-per-request") {
            const auto n = unsigned_value(value);
            if (n > std::numeric_limits<std::uint32_t>::max()) bad();
            if (key == "device") c.device = static_cast<std::uint32_t>(n);
            else if (key == "prefill-chunk") c.prefill_chunk_tokens = static_cast<std::uint32_t>(n);
            else if (key == "max-draft-tokens") c.max_draft_tokens = static_cast<std::uint32_t>(n);
            else if (key == "vision-cpu-threads") c.vision_cpu_threads = static_cast<std::uint32_t>(n);
            else c.max_images_per_request = static_cast<std::uint32_t>(n);
        } else if (key == "vision-enabled") {
            if (value == "true" || value == "1" || value == "on") c.vision_enabled = true;
            else if (value == "false" || value == "0" || value == "off") c.vision_enabled = false;
            else bad();
        } else bad();
    }
}
}

void apply_model_config(std::string_view input, const std::filesystem::path& source, Config& base) {
    if (input.size() > 64 * 1024 || input.find('\0') != input.npos || !valid_utf8(input)) bad();
    if (input.size() >= 3 && static_cast<unsigned char>(input[0]) == 0xef &&
        static_cast<unsigned char>(input[1]) == 0xbb && static_cast<unsigned char>(input[2]) == 0xbf) input.remove_prefix(3);
    Patch patch;
    bool saw_model_section = false;
    std::size_t pos = 0;
    while (pos <= input.size()) {
        const auto end = input.find('\n', pos);
        auto line = input.substr(pos, end == input.npos ? input.size() - pos : end - pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        for (unsigned char ch : line) if ((ch < 0x20 && ch != '\t') || ch == 0x7f) bad();
        auto t = trim(line);
        if (!t.empty() && t.front() != ';' && t.front() != '#') {
            if (t.front() == '[') {
                if (t != "[model]" || saw_model_section) bad();
                saw_model_section = true;
            } else {
                if (!saw_model_section) bad();
                bool quoted = false;
                std::size_t comment = t.size();
                for (std::size_t i = 0; i < t.size(); ++i) {
                    if (t[i] == '"') quoted = !quoted;
                    if (!quoted && (t[i] == ';' || t[i] == '#') &&
                        (i == 0 || t[i - 1] == ' ' || t[i - 1] == '\t')) { comment = i; break; }
                }
                if (quoted) bad();
                t = trim(t.substr(0, comment));
                const auto eq = t.find('=');
                if (eq == t.npos) bad();
                const auto key = trim(t.substr(0, eq));
                auto value = trim(t.substr(eq + 1));
                if (key.empty() || value.empty() || std::any_of(key.begin(), key.end(), [](unsigned char c) { return c <= 32 || c == 127; })) bad();
                const bool is_quoted = value.front() == '"';
                if (is_quoted) {
                    if (value.size() < 2 || value.back() != '"') bad();
                    value.remove_prefix(1);
                    value.remove_suffix(1);
                    if (value.find('"') != value.npos) bad();
                    const bool string_field = key == "model" || key == "alias" || key == "runtime" ||
                        key == "backend" || key == "tuning" || key == "speculation" ||
                        key == "vision-device" || key == "cache-type-k" || key == "cache-type-v";
                    if (!string_field) bad();
                } else if (value.find('"') != value.npos) bad();
                if (value.empty()) bad();
                if (!patch.fields.emplace(std::string(key), std::string(value)).second) bad();
            }
        }
        if (end == input.npos) break;
        pos = end + 1;
    }
    if (!saw_model_section) bad();
    Config candidate = base;
    apply(candidate, patch, source);
    base = std::move(candidate);
}

void load_model_config(const std::filesystem::path& path, Config& base) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("configuration: cannot open model config file");
    std::string contents(64 * 1024 + 1, '\0');
    in.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    contents.resize(static_cast<std::size_t>(in.gcount()));
    if (in.bad()) throw std::invalid_argument("configuration: failed to read model config file");
    apply_model_config(contents, path, base);
}

Config parse_model_config(std::string_view contents, const std::filesystem::path& source) {
    Config config;
    apply_model_config(contents, source, config);
    validate(config);
    return config;
}
Config load_model_config(const std::filesystem::path& path) {
    Config config;
    load_model_config(path, config);
    validate(config);
    return config;
}
} // namespace saga
