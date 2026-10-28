#pragma once

#include "saga/core/config.h"

#include <filesystem>
#include <string_view>

namespace saga {

// Parse a single [model] patch and apply only explicitly supplied fields to base.
// Structural/type errors are raised here; callers run validate() once after all merges.
void apply_model_config(std::string_view contents, const std::filesystem::path& source, Config& base);
// Load/apply without semantic validation, for the normal TOML -> INI -> CLI merge.
void load_model_config(const std::filesystem::path& path, Config& base);
// Standalone convenience: start from defaults, apply the file, then validate once.
Config parse_model_config(std::string_view contents, const std::filesystem::path& source);
Config load_model_config(const std::filesystem::path& path);

} // namespace saga
