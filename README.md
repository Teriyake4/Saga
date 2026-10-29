# Saga

A specialized C++20/CUDA inference engine under development. The first target is
**original RTX 4080 16 GB + Qwen3.8-27B GSQ3**, with native Windows as the active
build/qualification target. Linux host development/testing is available; Windows remains the production target.
Architecture, complete scope and blocking acceptance requirements
are in [PROPOSAL.md](PROPOSAL.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Current implementation slice: host-testable M2 contracts, planning and startup composition.** M2.1–M2.7/.9 host slices and M2.8 profile policy are implemented; final macOS/Linux host matrices passed 25/25 CTest groups in all ten configurations after the final test-only early-notification regression. See [docs/m2-evidence.md](docs/m2-evidence.md) for configurations, repeat tests, logs and limitations. Real compiled SM89 algorithms/tuning, physical backend qualification, native Windows and original-4080 evidence remain outstanding. No model loader, tokenizer execution, text inference, sampler, MTP or vision runtime exists. This is not a feature-complete or performance-qualified engine. The server never fabricates readiness, models, completions or token streams. See [docs/status.md](docs/status.md), [docs/m2-evidence.md](docs/m2-evidence.md), [docs/m1-evidence.md](docs/m1-evidence.md), and [docs/HANDOFF.md](docs/HANDOFF.md).

## Build and test without a GPU

Requirements: CMake >=3.28, a C++20 compiler, a host build tool, and Python >=3.9
when tests are enabled. Host dependency headers/licenses are vendored and hashed;
configuration/builds do not access the network. CUDA is off by default.

Native Windows with Visual Studio 2022 C++ build tools and its Windows SDK, from
an x64 Visual Studio developer PowerShell prompt:

```powershell
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug --parallel 4
ctest --preset windows-msvc-debug
```

Use `windows-msvc-release` for Release. These existing Visual Studio generator
presets remain available and use native x64 MSVC with a consistent dynamic CRT;
no WSL is involved. Native Windows Ninja host presets are also available from a
Visual Studio x64 developer prompt with Ninja installed and on `PATH`:

```powershell
cmake --preset windows-ninja-debug
cmake --build --preset windows-ninja-debug --parallel 4
ctest --preset windows-ninja-debug

cmake --preset windows-ninja-release
cmake --build --preset windows-ninja-release --parallel 4
ctest --preset windows-ninja-release
```

They explicitly select `cl.exe`; do not run them from a plain shell and treat a
failed MSVC environment setup as a qualified build. These are host build/test
paths, not Windows qualification evidence by themselves. To build, test, install and retain evidence:

```powershell
.\tools\windows\verify.ps1 -Configuration Debug
.\tools\windows\verify.ps1 -Configuration Release
```

See [docs/windows-build.md](docs/windows-build.md) for native Windows paths and
[docs/linux-build.md](docs/linux-build.md) for Linux host development/testing.
Windows Debug/Release, macOS development-regression, and Linux host jobs are
configured in `.github/workflows/host.yml`; configuration is **not** evidence CI
has run. macOS arm64 and the documented OrbStack Linux VM host runs have been
verified locally. Neither is inference-backend or production qualification. For
macOS host tests, use `host-debug` with the same configure/build/ctest sequence;
this is not a Metal/CPU inference backend.

On macOS/Linux, `host-ninja-debug` and `host-ninja-release` provide explicit
Ninja Debug/Release configure, build, and CTest paths; the existing Unix Makefiles
presets remain available. For example:

```sh
cmake --preset host-ninja-debug
cmake --build --preset host-ninja-debug --parallel 4
ctest --preset host-ninja-debug

cmake --preset host-ninja-release
cmake --build --preset host-ninja-release --parallel 4
ctest --preset host-ninja-release
```

Named product-only Release presets are `product-ninja-release` on macOS/Linux and
`windows-product-ninja-release` on native Windows. They set
`SAGA_BUILD_TESTS=OFF` and `SAGA_ENABLE_CUDA=OFF`, so product configuration does
not discover Python or create test targets. For example, from the applicable
host environment:

```sh
cmake --preset product-ninja-release
cmake --build --preset product-ninja-release --parallel 4
```

```powershell
cmake --preset windows-product-ninja-release
cmake --build --preset windows-product-ninja-release --parallel 4
```

The Windows command requires the same x64 VS developer environment and installed
Ninja as the Windows host presets. Product-only means building the server/product
targets without tests or CUDA; it is not a qualification claim. Additional Unix
presets are `host-release` (checks remain enabled in NDEBUG), `host-sanitized`
(ASan+UBSan), and `host-thread-sanitized` (ThreadSanitizer). Run thread
sanitization separately from ASan/UBSan; CMake rejects combining them.
`SAGA_WARNINGS_AS_ERRORS=ON` is the default for Saga sources. Third-party headers
are marked system includes. All host tests are finite and use no weights. A built
server's `--check-config` validates static configuration only: it does not load or
validate model artifacts, establish capability support, allocate resources, or
report runtime readiness.

## Run the model-free server

```sh
./build/host-debug/saga-server --config config/server.example.toml
```

On Windows, use `build/windows-msvc-debug/Debug/saga-server.exe` instead. Ctrl+C
requests orderly shutdown; native Windows also handles Ctrl+Break via the native
platform adapter. Configuration precedence is defaults → TOML → optional
single-model INI → command-line overrides, with cross-field validation after
merging. TOML/INI syntax, field names and types remain strict. `--help` lists the initial startup
flags; `--check-config` validates static settings only, not capability support,
actual allocations or readiness. Port 0 requests an OS-selected port, reported
as a structured `listener_bound` log on stderr.

| Endpoint | Present behavior |
|---|---|
| `GET /healthz` | HTTP 200, process liveness only |
| `GET /readyz` | HTTP 503, `model_not_ready` |
| `GET /v1/models` | HTTP 200, OpenAI list shape with an empty `data` array |
| `POST /v1/chat/completions` | HTTP 503 JSON before any SSE headers; no generation |
| `GET /metrics` | HTTP routing counter and `saga_engine_ready 0` |

These are transport/operational endpoints, **not the M5/M6 correct API text
checkpoint**. A nonempty `model.artifact` fails startup explicitly until loading
is implemented. Capability/context/codec settings are canonicalized metadata
candidates only; selecting `mtp`, an image option or a codec does not make it
implemented. The full-release 8K/MTP3/CPU-vision profile remains unqualified.

### One model configuration per instance

Use one INI file per model/instance, separate from the server's TOML settings:

```ini
[model]
model = C:\Models\qwen3_8_27b_gsq3.gguf
alias = qwen3.8-27b-gsq3
runtime = qwen3_8_27b
ctx-size = 8192
kv-capacity = auto
cache-type-k = int8_g64
cache-type-v = int8_g64
speculation = off
```

```powershell
.\build\windows-msvc-debug\Debug\saga-server.exe --config config/server.example.toml --model-config config/model.example.ini --check-config
```

There is exactly one `[model]` section—no wildcard sections, model catalog,
selector, auto-routing or hot loading. Each instance receives its own file;
the filename need not be `model.ini`. Only explicit INI fields override the
TOML/defaults, and explicit CLI flags win last. Relative artifact paths are
anchored to the INI directory; Windows drive/UNC paths and Unicode are preserved.
The INI follows familiar llama.cpp option names but is intentionally not a
full llama.cpp router/options file. See [docs/model-config.md](docs/model-config.md).

Static validation does not load the artifact or enforce a filename extension. A
nonempty artifact still produces an explicit unsupported-loading error at normal
startup in this increment; no model entry or completion is fabricated. GGUF is
the planned runtime input container, not an implemented loader or a promise to
support arbitrary GGUF models. Saga's GSQ3 profile is not GGUF Q3_K/Q3_K_M/IQ3;
see [docs/artifact-format.md](docs/artifact-format.md).

### Limits and authentication

- Bind `127.0.0.1` by default. Only literal `127.0.0.1`/`::1` count as loopback for
  the authentication policy; a DNS name is not assumed safe.
- Set a secret in the environment named by `server.api_key_env` (default
  `SAGA_API_KEY`). Its value is not stored in TOML, startup flags or logs. A
  non-loopback bind fails without a key. Nonempty keys must be RFC 6750 tokens:
  1–4096 ASCII bytes, `[A-Za-z0-9._~+/-]+` followed by optional `=` padding.
  Unsupported keys fail startup without logging their contents. When configured,
  all routes except liveness require `Authorization: Bearer ...`; the scheme is
  case-insensitive, but token bytes and padding are compared exactly.
- Bound worker count, pending transport connections, body bytes, header count
  (64), header/URI length and per-I/O idle timeouts. Generation queue settings
  are reserved for the later scheduler; transport queue saturation currently
  closes the connection rather than emitting a generation-admission response.
- One request per connection; HTTP keep-alive/pipelining is disabled so unread
  rejected bodies cannot become subsequent requests. This does not prohibit a
  future streaming response on that connection. Duplicate `Content-Length`
  (even identical values), TE+CL, invalid lengths and unsupported transfer codings
  are rejected before body reads or `100 Continue`. Valid chunked bodies are
  supported with strict chunk/extension/trailer parsing and bounded trailer counts.
  Headers, including authorization and framing values, are not URI-decoded.
- No TLS is built in. Do not expose this development server directly to the
  Internet. Use a trusted TLS reverse proxy with absolute request/header/body
  deadlines and connection/body limits. Per-I/O idle timeouts do not stop
  continuous trickle traffic and are not an absolute shutdown-time guarantee.
- No remote model/image fetch, request-body logging or permissive CORS is enabled.
  Images are unavailable, not decoded by an unbounded fallback.

## What is implemented beneath the shell

- Startup-only kernel descriptor/support/prepare resolution, compatibility guards,
  optional profile ranking and nonvirtual prepared launchers. Invalid enums and
  contradictory GSQ3 group sizes fail before selection. Workspace-free kernels
  do not require an unused allocation alignment. These are host contracts,
  **not CUDA inference kernels**.
- Separate typed allocation candidate, bounded bring-up grant, reconciliation and
  approval. Checked arithmetic, exact sharing identities, exclusive backing
  ownership and lease-retained accounting; conservative sum-of-peaks accounting.
  Allocation factories pin their original grant state, readiness requires no active
  factory/release transaction, and reconciliation retains verified persistent owners.
  Independent leases may release on other threads; coordinator/proof operations
  and moves remain externally serialized. User factories/deleters run outside
  ledger locks, and expired allocation-owner history is retired immediately.
- Metadata/checked byte calculation for all eleven required KV formats and curated
  mixed pairs. No append/read/attention implementations or codec availability claim.
- Host-testable M2 contracts provide structural metadata/program checks, owning
  registries, bounded resource/context planning, typed step contracts, profile-based
  guarded kernel selection, and synthetic startup/grant/reconcile/approval composition.
  See [docs/m2-evidence.md](docs/m2-evidence.md). This does not load GGUF, register a
  production model/backend, compile SM89 inference algorithms, or implement inference.
- A separately selected `saga_platform` target: `src/platform/shared/` for common
  checks, `src/platform/windows/` for native Windows mapping/process APIs, and
  `src/platform/linux/` for Linux selection reusing `src/platform/posix/`.
  Linux host development/testing is documented in
  [docs/linux-build.md](docs/linux-build.md); macOS also uses the POSIX implementation
  for development tests. Mapping covers
  Unicode native paths, empty/unaligned windows and checked >4-GiB file ranges;
  native process adapters own UTF-8 arguments, environment reads and shutdown
  notification. See [docs/platform.md](docs/platform.md).
- A separate `saga_runtime` library: resource-retaining completions, a bounded single
  execution worker with reserved cleanup, generation-tagged sequence retirement,
  bounded single-active-request admission/cancellation/deadlines, and a bounded
  committed-byte output ring. Failed fences quarantine slots or fail-stop admission;
  cancellation does not authorize early reclamation. Cross-component tests exercise
  cleanup/slot reuse and request/worker/output lifetimes. See
  [docs/runtime-references.md](docs/runtime-references.md) for contracts and limits.
- Offline dependency verification and explicit blocked reference/quality manifests.

These runtime primitives are tested libraries, **not wired into HTTP generation**.
Production model/backend sessions and liveness-based scratch reuse remain future work.
Finite static auto-context planning now exists in the host-testable M2 planner; it
selects the highest statically eligible candidate and does not retry smaller contexts
after subsequent preparation/warmup/capture/reconciliation failures. This runtime-
history note predates the current M2 slice and added no generation integration.
Historical M1 host validation and current M2 targeted/full-matrix evidence have distinct
scopes: see [docs/m1-evidence.md](docs/m1-evidence.md) and
[docs/m2-evidence.md](docs/m2-evidence.md). Native Windows/MSVC execution remains
unverified.

## Real 4080 CUDA portability gate

The optional CUDA target implements an instruction/result probe, not inference.
It exercises aligned `cp.async`, BF16 WMMA, compiled SM89 identity and actual-kernel
occupancy queries. It rejects other GPUs, including 4080 Super.

Use a native Windows x64 MSVC developer prompt with Ninja, an original desktop
RTX 4080 and a supported installed CUDA toolkit:

```powershell
.\tools\windows\cuda-smoke.ps1 -Device 0
```

The script uses `cuda-windows`, retains verbose compilation/device-link evidence
and checks both the runtime exit code and JSON result. Initial build floor is
CUDA >=12.8; the exact MSVC/toolkit/driver matrix is not yet qualified. Linux CUDA portability and reference reproduction remain deferred workstreams.
See [docs/cuda-portability.md](docs/cuda-portability.md) for evidence requirements.
**This CUDA translation unit has not been compiled, device-linked or run here.**
The native Windows original-4080 run is the active portability gate. The original
cross-platform requirement still needs a separate Linux run later; host tests do
not satisfy either device gate.

## Reference sources and reproducibility

Pinned NInfer, NInfer-4080, Strata, Switchyard and llama.cpp sources were retrieved
and inspected; exact revisions/file hashes and actual failed probes are recorded
in [docs/source-lock.md](docs/source-lock.md). No reference was built/benchmarked,
and no inference code has yet been imported. Switchyard reuse is blocked by
unresolved license evidence. The nearest reusable-kernel reference is the 4080
fork; actual activation dispatch, represented values and licenses must still be
matched before adaptation and parity claims.

```sh
python3 -B tools/verify_dependencies.py
python3 -B tools/verify_sources.py --structure-only
python3 -B tools/verify_sources.py
```

The last command intentionally exits **1 (blocked)** while mandatory artifacts,
reference reproduction, independent fixtures, policies and GPU evidence remain
unresolved. That is not a broken build or a passing qualification gate. Optional
`--online` explicitly re-fetches bounded source/metadata URLs and checks hashes;
it never downloads weights or executes retrieved code. Network use/rate limits
are separate from host builds. See [docs/quality-policy.md](docs/quality-policy.md)
for preregistration requirements before affected quantization or optimization.
