# Single-model configuration

Saga accepts one model profile per INI file. It is not a model catalog: there is no model selector, router, hot loading, or offline model listing. A running Saga process retains its existing single-model startup and readiness behavior; validating configuration is not loading a model or reporting readiness.

```sh
saga-server --config config/server.example.toml --model-config config/model.example.ini
saga-server --config config/server.example.toml --model-config config/model.example.ini --check-config
```

The INI file contains exactly one `[model]` section. Only the explicitly specified fields patch the current `Config`; omitted values are preserved. Merge order is defaults, typed TOML, explicitly supplied typed INI fields, CLI overrides, then one semantic validation. Structural, syntax, type, unknown-key, and duplicate-key errors are fatal before semantic validation. This allows a later INI or CLI value to repair a semantic conflict from TOML while malformed values remain errors.

Existing TOML `[model]` settings (`runtime`, `artifact`, and `public_name`) remain supported as a compatibility fallback. The INI file is preferred for model-specific configuration and patches only its stated fields; it does not replace other TOML settings. Server and security settings remain in TOML.

Supported keys are `model` (artifact path), `alias` (public name), `runtime`, `backend`, `device`, `tuning`, `ctx-size` (`auto` or unsigned integer), `kv-capacity` (`auto` or unsigned integer), `cache-type-k`, `cache-type-v`, `prefill-chunk`, `speculation`, `max-draft-tokens`, `vision-enabled`, `vision-device`, `vision-cpu-threads`, and `max-images-per-request`. Unknown keys or any section other than `[model]` are rejected. This is not a general llama.cpp options file; server/security settings remain in TOML.

The planned runtime model-input container is GGUF, with support limited to an explicitly verified Saga architecture/encoding profile—not arbitrary GGUF models or generic ggml execution. This is a design decision only: no GGUF reader/converter/model loading is implemented, and configuration does not validate extensions or support. See [artifact-format.md](artifact-format.md) for representation, packing, and unresolved-profile constraints.

Input is UTF-8 (optional BOM), limited to 64 KiB, and accepts CRLF. Keys and values are trimmed around `=`. `;` and `#` start an inline comment only outside quotes and when preceded by whitespace (or at the start of a trimmed line). Values can be unquoted or enclosed in a matching pair of double quotes. Quotes are delimiters only: no escapes, shell execution, environment expansion, or C-style backslash escapes are supported. Backslashes inside quoted and unquoted values are literal. NUL, malformed UTF-8, and control characters are rejected. Errors do not echo configuration values.

Relative artifact paths resolve against the absolute INI file's directory, never the process working directory (so a relative INI filename is anchored to the launch directory once at configuration load). On Windows, absolute drive and UNC paths are preserved; drive-relative paths such as `C:weights.gguf` and ambiguous root-relative paths such as `\weights.gguf` are rejected. Artifact remains a string in `Config`; there is no runtime suffix validation or supported loader yet, so existing startup continues to reject nonempty artifacts. The INI profile changes configuration only; it does not imply inference readiness or a loaded model in `/v1/models`.
