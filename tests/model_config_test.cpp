#include "saga/core/config.h"
#include "saga/core/model_config.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}
saga::CommandLine cli(std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> values(args);
    return saga::parse_command_line(values);
}
struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("saga-model-config-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("cannot create temp directory");
    }
    ~TempDir() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.generic_u8string();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}
void write(const std::filesystem::path& path, std::string_view text) {
    std::ofstream out(path, std::ios::binary);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) throw std::runtime_error("cannot write fixture");
}
}

int main() {
    try {
        saga::Config defaults;
        const std::string bom_ini = "\xef\xbb\xbf[model]\r\nmodel=weights\\Qwen.gguf\r\nalias=東京\r\nctx-size=auto\r\n";
        saga::apply_model_config(bom_ini, "C:/profile/model.ini", defaults);
        check(!defaults.max_tokens.tokens, "ctx-size accepts auto");
        check(defaults.public_name == "東京" && defaults.max_draft_tokens == 3, "explicit fields patch defaults only");
        check(defaults.artifact.find("weights/Qwen.gguf") != std::string::npos || defaults.artifact.find("weights\\Qwen.gguf") != std::string::npos,
              "backslash path stays literal and resolves relative to config");

        auto typed = saga::parse_model_config("[model]\nctx-size=4096\nkv-capacity=auto\n", "profile.ini");
        check(typed.max_tokens.tokens == 4096 && !typed.kv_capacity_tokens.tokens, "typed numeric and auto fields");
        check(typed.public_name == saga::Config{}.public_name && typed.max_draft_tokens == 3, "no absent-field replacement");

        TempDir dir;
        const auto toml = dir.path / "server.toml";
        write(toml, "[model]\npublic_name=\"toml-name\"\n[context]\nmax_tokens=16384\nkv_capacity_tokens=4096\n");
        const auto ini = dir.path / std::filesystem::path(u8"single model 模型.ini");
        const auto toml_argument = utf8_path(toml);
        const auto ini_argument = utf8_path(ini);
        write(ini, "[model]\nctx-size=2048\n");
        const auto merged = cli({"--config", toml_argument, "--model-config", ini_argument});
        check(merged.config.max_tokens.tokens == 2048 && merged.config.kv_capacity_tokens.tokens == 4096 &&
              merged.config.public_name == "toml-name",
              "INI patches TOML fields and preserves absent values while repairing a semantic conflict");
        write(ini, "[model]\nctx-size=auto\n");
        const auto auto_merge = cli({"--config", toml_argument, "--model-config", ini_argument, "--context", "2048"});
        check(auto_merge.config.max_tokens.tokens == 2048, "CLI follows INI and TOML");
        write(ini, "[model]\nctx-size=8192\n");
        rejects([&] { (void)cli({"--config", toml_argument, "--model-config", ini_argument}); });
        write(ini, "[model]\nctx-size=8192\n");
        const auto final_repair = cli({"--config", toml_argument, "--model-config", ini_argument, "--context", "auto"});
        check(!final_repair.config.max_tokens.tokens, "CLI can repair INI/TOML semantic conflict");

        for (const std::string bad : {
                 "", "[other]\nmodel=x\n", "[model]\n[model]\n", "[model]\nmodel=x\nmodel=y\n",
                 "[model]\nunknown=x\n", "[model]\nctx-size=no\n", "[model]\nctx-size=-1\n",
                 "[model]\nctx-size=\"4096\"\n", "[model]\nkv-capacity=on\n", "[model]\nvision-enabled=yes\n",
                 "[model]\ncache-type-k=not-a-codec\n", "[model\nmodel=x\n", "model=x\n"}) {
            rejects([&] { auto c = saga::Config{}; saga::apply_model_config(bad, "x.ini", c); });
        }
        rejects([] { auto c = saga::Config{}; saga::apply_model_config(std::string(65537, ' '), "x.ini", c); });
        std::string nul = "[model]\nmodel=x"; nul.push_back('\0'); nul += "y\n";
        rejects([&] { auto c = saga::Config{}; saga::apply_model_config(nul, "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\xff\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\xf8\x90\x80\x80\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\xf5\x80\x80\x80\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\xc2\x85\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\"\xc2\x9f\"\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nmodel=\"\"\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\"a\"\"b\"\n", "x.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\"\"\n", "x.ini", c); });

        auto quoted = saga::parse_model_config("[model]\nalias=\"a#b;c\"\nmodel=\"relative\\\\weights.gguf\" ; trailing comment\n", "C:/base/model.ini");
        check(quoted.public_name == "a#b;c" && quoted.artifact.find("relative") != std::string::npos,
              "quote-aware comments and literal quoted backslashes");
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nalias=\"unterminated\n", "x.ini", c); });
        auto hash_value = saga::parse_model_config("[model]\nalias=x#not-comment\n", "x.ini");
        check(hash_value.public_name == "x#not-comment", "inline marker without preceding whitespace remains value data");

        const auto relative_ini = dir.path / "relative.ini";
        write(relative_ini, "[model]\nmodel=artifacts/model.gguf\n");
        saga::Config relative;
        saga::load_model_config(relative_ini, relative);
        const auto expected_artifact = std::filesystem::absolute(relative_ini).parent_path() / "artifacts/model.gguf";
        const auto expected_u8 = expected_artifact.lexically_normal().generic_u8string();
        const std::string expected_utf8(reinterpret_cast<const char*>(expected_u8.data()), expected_u8.size());
        check(relative.artifact == expected_utf8, "relative artifact resolves against absolute INI directory");

        saga::Config atomic;
        atomic.public_name = "before";
        atomic.max_tokens.tokens = 10;
        rejects([&] { saga::apply_model_config("[model]\nalias=after\nctx-size=bad\n", relative_ini, atomic); });
        check(atomic.public_name == "before" && atomic.max_tokens.tokens == 10,
              "failed patch leaves base Config unchanged");
        auto drive = saga::parse_model_config("[model]\nmodel=C:/models/a.gguf\n", "C:/cfg/model.ini");
        check(drive.artifact.find("C:/models/a.gguf") != std::string::npos, "absolute drive path preserved");
        auto unc = saga::parse_model_config("[model]\nmodel=\\\\server\\share\\a.gguf\n", "C:/cfg/model.ini");
        check(unc.artifact.find("server") != std::string::npos && unc.artifact.find("share") != std::string::npos, "UNC path preserved");
#ifdef _WIN32
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nmodel=C:weights.gguf\n", "C:/cfg/model.ini", c); });
        rejects([] { auto c = saga::Config{}; saga::apply_model_config("[model]\nmodel=\\weights.gguf\n", "C:/cfg/model.ini", c); });
#endif

        rejects([] { (void)cli({"--model-config", "one.ini", "--model-config", "two.ini"}); });
        rejects([] { (void)cli({"--models-preset", "x"}); });
        rejects([] { (void)cli({"--model", "x"}); });
        rejects([] { (void)cli({"--list-models"}); });
        const auto old = cli({"--port", "0", "--context", "auto"});
        check(old.config.server.port == 0 && !old.config.max_tokens.tokens, "existing CLI semantics preserved");
        std::cout << "PASS: single-profile model config patch parsing and merged validation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
