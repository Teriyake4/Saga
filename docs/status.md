# Saga implementation status

Snapshot: **28 October 2026**. The user approved starting implementation; this is
the M0/M1/M2 foundation plus hardening, single-model configuration, modular
platform work and neutral runtime primitives, not completion of those milestones.
The continuation entry point is [HANDOFF.md](HANDOFF.md). Native Windows is now the
active build/qualification target; Linux work is deferred at the user's request.
The full feature scope and 95% throughput / 110% uncached-TTFT thresholds remain
unchanged, and no Windows-only result implies eventual cross-platform completion.

## M0 / M1 / M2 milestone gates

| Milestone | Status | Implemented evidence | Outstanding exit-gate work |
|---|---|---|---|
| **M0 - sources and reference baseline** | **Partial / blocked** | Pinned reference-source and model-metadata retrieval, hashes/licenses where available, reference/quality manifests and policy scaffolding | Actual original-4080 environment, reference build/run and observed precision dispatch, verified component/weight/tokenizer bytes, frozen matched suite, independent fixtures and approved quality budgets; Switchyard reuse remains license-blocked |
| **M1 - foundation and server shell** | **Partial; host shell implemented** | Model-free server, configuration/security hardening, platform adapters, host build/install scripts and presets; macOS host validation | Native Windows/MSVC Debug/Release and installed-package evidence; real original-4080 CUDA compile/device-link/run (probe source exists only); full platform/device ownership and qualification gates. Linux qualification remains deferred |
| **M2 - contracts and resource planning** | **Partial; neutral groundwork implemented** | Kernel contracts/selection, allocation ledger and candidate/grant/reconcile/approval types, runtime primitives, minimal model/backend/session contracts and static registries, synthetic composition tests | Complete metadata-driven component calculator, explicit/auto context planning, integrated startup/readiness and failure orchestration, concrete model/state/frontend and generation-step contracts, production sessions/physical fences, and remaining integration qualification/tuning |

The latest implementation slice passed **18/18 host test groups** on macOS in
Debug, Release, ASan/UBSan and TSan, plus 20 repeated TSan `model_contracts` runs.
These results do not close Windows, CUDA, model-quality, fit or performance gates.
No milestone percentage or full M0/M1/M2 completion is claimed.

## Runtime model format decision (planned)

**GGUF is the required runtime input container**, not a proprietary `.engine` or
`.ninfer` model file. This is a documentation requirement, not implemented loader
support. The initial architecture/weight encodings remain narrowly scoped; GSQ3
is not interchangeable with GGUF Q3_K/IQ3 recipes. The exact verified GSQ/Q4 GGUF
profile and source/specification pins remain M0/M3 work. See
[artifact-format.md](artifact-format.md).

GGUF does not dictate the GPU execution layout: bounded one-time or cached native
repacking may preserve the declared represented values, with all transient and
steady-state memory accounted for. It does not establish NInfer parity or waive
any existing performance/quality gate. No GGUF reader/writer/converter, universal
GGUF-model support or new model/device implementation is supplied by this decision.

## Implemented and locally exercised

| Area | Present implementation | Evidence/limits |
|---|---|---|
| Build | C++20 CMake targets, separate native-MSVC Debug/Release presets with dynamic CRT, optional Windows CUDA probe, install/evidence scripts | Windows-priority CI plus macOS host regression configured; only macOS executed here, Linux jobs deferred |
| Server shell | Strict TOML + optional one-model INI + CLI merge, structured errors/logs, liveness, unavailable readiness/generation, empty model list and truthful metrics | Real-process HTTP/config tests; no model catalog/router or M5/M6 correct text API claim |
| Bounds/security | Fixed HTTP workers/pending connection bound, header-count/body limits, strict framing/chunk/trailer parsing, I/O idle timeouts, localhost default, environment bearer auth | Raw-socket/idle/disconnect/restart tests; one request per connection; no TLS or absolute trickle-client deadline; trusted reverse proxy required for remote exposure |
| Kernel contracts | Startup descriptor/support/prepare, well-formedness/compatibility-before-ranking, optional profile preferences, nonvirtual prepared launchers and actual workspace checks | Synthetic specialization guard/fallback/lifetime tests, invalid enum/GSQ3-group tests and zero-workspace alignment tests; no real CUDA inference implementation |
| Resource planning | Checked per-domain ledger, sharing identities, exclusive backing ownership, synchronized lease-release accounting, separate candidate/grant/reconcile/approval types and overhead caps | Fake allocation/measurement tests, including reentrant transactions, verified owner coverage, concurrent releases and failed bring-up; not actual driver measurements or model fit |
| KV metadata | Eleven canonical formats, alias normalization, exact block sizes and checked main-cache page/layout sizing | All proposal main-cache byte fixtures; no quantizer/attention/verification execution support |
| Platform | Separate `saga_platform` target selects shared + native Windows or reusable POSIX sources; mapping, Unicode argv/environment and RAII shutdown adapters | macOS tests include >4-GiB windows, Unicode/environment bounds and prior signal restoration; Windows runtime unverified, Linux qualification deferred |
| Runtime primitives | Resource-retaining completions; bounded execution worker/reserved cleanup; generation-tagged sequence retirement; bounded single-active admission and committed-byte output | Unit and cross-component tests; owner-thread contracts enforced where specified; no HTTP generation integration or device synchronization |
| Model/backend contracts | Structural artifact/model/program validation, distinct component/tensor identities, exact tied-weight aliases, neutral session/runtime interfaces and independent owning static registries | Synthetic grant-backed composition and lifetime tests; no production registrations, typed generation-step results, startup orchestration or readiness integration |
| CUDA probe | Real `cp.async` and BF16 WMMA result checks, original-4080 identity, SM89 image/occupancy queries and JSON evidence | Source only; **not compiled, device-linked or run** |
| Source locks | Five pinned reference commits, selected implementation/build/license hashes, original/GSQ model metadata and actual failed probes | Partial source verification; no reference binary, weights or matched benchmark |

The M2 layer is intentionally incomplete: neutral model/backend registry and
session interfaces exist, but production registrations/session implementations,
complete model/state/frontend schemas, automatic context search and liveness-based
scratch optimization are not present. Completion, logical deferred retirement and reserved worker cleanup now
exist as neutral primitives; a future backend must supply physical-work fences.
Memory accounting conservatively sums unique peaks. Real backends must allocate
through the granted contract, supply truthful measurements and retain leases;
the host calculator does not intercept driver allocations or prove fit itself.

## Current neutral model-contract increment

- Added `include/saga/runtime/model_contracts.h`, its implementation and the
  `model_contracts` test group as one bounded M2.1/M2.2 slice. Existing model types
  and kernel guards are reused; M2 remains partial.
- Logical bindings now distinguish artifact-component identity from physical tensor
  identity. Exact aliases are accepted; conflicting aliases, overlapping distinct
  tensors, invalid ranges/enums/dimensions/context and unavailable requested
  capabilities are rejected. Bounds and overlap checks avoid addition overflow.
- `backend::validate_request()` shares structural kernel checks with both registry
  resolution and model validation. No payload packing or numerical quality is proven.
- Independent model/backend registries retain selected owners and have no default
  registrations or fallback. Sessions bind their program at instantiation rather
  than receiving a generic call list on each execution. Factory inputs are borrowed
  synchronously; callers must drain physical work and destroy runtime before session.
- Test-only implementations exercise registry-selected instantiation, grant-backed
  allocation, constructor-failure unwinding, sequence lifetime, copied descriptions,
  and resource retention after producer/runtime/session destruction until the last
  completion consumer releases its handle. This is not startup/readiness orchestration.
- GPT6 Luna implemented the slice; two independent Luna reviewers checked API/lifetime
  and metadata/validation paths, followed by fixes and re-reviews with no concrete
  blockers. This is not a proof of correctness or native Windows qualification.
- Final macOS arm64 builds and **18/18 CTest groups passed** in Debug, Release,
  ASan/UBSan and TSan; `model_contracts` also passed 20 repeated TSan runs.
  Prior installation evidence below was not rerun for this slice.
- Stop after this one step pending user approval. No planner, startup orchestrator,
  controller, HTTP generation, model inspection, kernels or GPU tests were added.

## Prior model/device-neutral runtime increment

- Added `saga_runtime`: `completion`, `execution_worker`, `sequence_slots`,
  `admission` and `output_buffer`, with six new CTest groups including integration.
- Completion consumers and producer records retain resource leases. Success is
  the only readable result; cancellation/abandonment are not physical fences.
- The worker has separately bounded ordinary tasks and pre-reserved cleanup;
  shutdown cancels queued ordinary work, drains cleanup and joins. User callbacks
  and stop notifications execute outside queue locks. Callback failures fail-stop.
- Sequence retirement invalidates handles immediately; a pending fence prevents
  reuse and only success permits reuse. Other terminal outcomes quarantine slots.
- Admission owns one active request and bounded queued/retained payloads. Queued
  cancellation notifies stop subscribers; active cancellation waits for its fence.
  Copied ticket/completion/resource handles keep payload quota charged until release.
- Output copies only bytes authorized by a successful commit completion, with fixed
  byte/chunk capacities, nonblocking backpressure and drain-before-terminal reads.
  It does not determine model commit semantics, encode SSE or implement generation.
- Steps used GPT6 Luna implementation and separate review passes, followed by fixes
  and re-review. Final review found no concrete blockers; this is not a correctness
  proof or Windows qualification. A parent-found test teardown race was also fixed.
- No production HTTP generation wiring, model inspection, kernels or device tuning
  was added. The model-free shell keeps its existing truthful 503/empty-list behavior.

**Final validation on macOS arm64:** all **17/17 CTest groups passed** in Debug,
Release, ASan/UBSan and TSan. The five runtime unit groups each passed 20 repeated
TSan runs; `runtime_integration` passed 50 repeated TSan runs. Release installation
into `build/neutral-runtime-install` and installed TOML plus TOML/INI static checks
passed. Vendored dependency/patch hashes and source structure remain passing.
Windows, PowerShell, NVCC and GPU execution remain unverified. Commands and API
caveats are preserved in [HANDOFF.md](HANDOFF.md).

## Prior Windows-first / single-model increment

- Added `--model-config FILE`: exactly one `[model]` INI profile, no catalog,
  wildcard sections, model selection, listing or hot reload. Explicit fields
  patch typed TOML/defaults; CLI overrides win, followed by final validation.
  INI cannot override server/network/auth settings. Unknown/duplicate keys,
  malformed UTF-8/control characters, ambiguous Windows paths and malformed
  values fail; failed patches leave the previous configuration unchanged.
- Relative artifacts anchor to the absolute INI directory. BOM/CRLF, literal
  Windows backslashes, quoted comments, Unicode drive/UNC paths and omitted-field
  preservation have dedicated tests. Static checks never claim model support.
- `platform/shared`, `platform/windows` and deferred `platform/linux` selections
  isolate OS behavior; POSIX bodies are reused for macOS development and later
  Linux work. Startup no longer embeds Win32 environment/signal conversion logic.
- Windows scripts configure/build/test/install Debug/Release, require both installed
  example configurations, retain exit codes and logs, and never treat a skipped
  package check as success. The separate CUDA script checks exit code plus JSON.
- The pinned llama.cpp server README was fetched and its SHA-256 independently
  rechecked (`bench/reference/model-config-reference.json`). Familiar option names
  are reused, not its multi-model router or parser implementation.

**Prior validation on macOS arm64:** all **11/11 CTest groups passed** in Debug,
Release, ASan/UBSan and ThreadSanitizer configurations. The new groups are
`model_config` and `platform_process`; existing HTTP framing/auth/listener and
memory/concurrency hardening checks remain enabled and passing. Release install
and both installed TOML/INI static checks passed from a directory containing
spaces and Unicode. Vendored dependency/patch hashes still match.

No PowerShell/MSVC/Windows/NVCC runtime was available. Windows scripts were reviewed
but not parsed or executed with PowerShell, Windows sources were not compiled,
and no hosted CI/GPU/model pass is claimed. Windows CUDA and application execution
remain gates for the next native validation environment. See
[windows-build.md](windows-build.md), [model-config.md](model-config.md) and
[platform.md](platform.md).

## Foundation hardening (preserved)

- Allocation factories pin their original grant state. Leaving the initiating grant
  moved/replaced rejects the result and fails only the original state.
- Active factory/release transactions, including zero-byte calls, block readiness
  transitions. Reconciliation pins real owners and verifies live-byte coverage;
  expired weak ownership cannot masquerade as retained persistent storage.
- Per-grant synchronization supports independent lease releases concurrent with
  reports/reconciliation. Coordinator/proof operations and moves remain externally
  serialized. User factories and backing deleters execute outside ledger locks.
- Controls retire their registry entries after physical release instead of retaining
  expired allocation history. Moved-from candidates cannot mint grants.
- Listener RAII covers bound-but-unused destruction, pre-run cancellation and
  startup unwinding. The listener thread owns transport stop; accept retries service
  cancellation, including a deterministic injected-descriptor-exhaustion test.
- Framing rejects duplicate lengths, TE+CL, invalid literal lengths and unsupported
  transfer codings before body reads or `100 Continue`. Valid chunked requests have
  checked size/extension/trailer syntax and total trailer-count bounds. One request
  per connection prevents unread rejected bodies becoming subsequent requests.
- Authorization values remain literal HTTP header bytes. Bearer schemes are
  case-insensitive with one or more SP; tokens/padding compare exactly. Configured
  keys must follow bounded RFC 6750 token syntax, with no secret in startup errors.
- CLI semantic validation runs after the full merge; malformed TOML types/fields
  still fail. Unknown codec IDs fail public validation.
- The HTTP dependency patch, upstream identity and installed identity are separately
  recorded in `third_party/lock.json`. The pinned upstream plus recorded patch was
  applied in scratch storage and reproduced the installed header byte-for-byte.

This prior hardening did not complete M2. The subsequent runtime increment adds
completion tokens and reserved cleanup, but production sessions and validated tensor
bindings remain absent. Neither increment establishes full HTTP conformance or
production remote-exposure qualification.

## Prior foundation-hardening validation

Environment: macOS arm64 Darwin 27.0.0, AppleClang 21.0.0, CMake 4.3.0,
Xcode Make, Python 3.14.9 selected by CMake. The system Python 3.9.6 also ran
source/dependency tooling. No `nvcc` was available.

- `cmake --build --preset host-debug` and `ctest --preset host-debug`: **9/9 pass**.
- `cmake --build --preset host-release` and `ctest --preset host-release`: **9/9 pass**.
- `cmake --build --preset host-sanitized` and `ctest --preset host-sanitized`:
  **9/9 pass**, AddressSanitizer + UndefinedBehaviorSanitizer.
- `cmake --build --preset host-thread-sanitized` and `ctest --preset host-thread-sanitized`:
  **9/9 pass**, ThreadSanitizer, run separately from ASan/UBSan.
- ThreadSanitizer memory, configuration/listener lifecycle, HTTP transport and
  process-shell groups each passed **10 consecutive repeated runs**.
- Release installation and static validation using the installed server/config
  passed. The recorded HTTP patch reproduced the vendored header byte-for-byte.
- CTest groups: contracts, memory, platform, configuration/in-process listener
  lifecycle, isolated HTTP parser/accept-retry probes, real-process HTTP shell,
  vendored dependency/patch hashes, source-manifest structure and the source
  verifier's **24 regression tests**. Runtime checks remain active with NDEBUG.
- Source online/offline audit: **expected exit 1, blocked**, not a model readiness
  failure in the host tests and not a passed M0 gate. Structural validation exits
  0 explicitly `structure-valid-not-qualified`.
- No Linux/native-Windows build execution, hosted CI execution, CUDA compilation,
  model generation, BF16 evaluation, quantization evaluation or throughput test.

On this sandboxed Mac, `/usr/bin/make` and compiler wrappers attempted unwritable
Xcode cache access. The initial build also rejected deprecated C++20 `u8path`;
that source was fixed to use a native `char8_t` filesystem path. Successful local
configure used explicit real tool paths, without weakening warnings:

```sh
cmake --preset host-debug \
  -DCMAKE_CXX_COMPILER=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  -DCMAKE_MAKE_PROGRAM=/Applications/Xcode.app/Contents/Developer/usr/bin/make
```

The same overrides were used with `host-release`, `host-sanitized` and
`host-thread-sanitized`. These are
local environment workarounds, not production Linux/Windows toolchain pins.

## Reference findings affecting the next increments

See [source-lock.md](source-lock.md) for URLs, revisions, hashes and license texts.
The retrieval snapshot contains **104 inputs, 96 HTTP-200 body hashes and eight
HTTP-404 probes**; all five pinned commit responses matched the requested SHA.
Transient GitHub API 403 rate-limit responses remain in retrieval history. The
final online audit reverified 96 hashes; the eight probe paths remained 404.

- The 4080 fork has shape-dependent **A8 dispatch, including small verification
  widths**. Source flags alone do not establish per-site A16 parity.
- Its source-scale conversion handles BF16-to-FP16 rounding/subnormal cases.
  Repacking and represented-value changes must not be conflated.
- Switchyard root/alternative license probes returned 404. This does not prove
  no license exists anywhere, but reuse permission is unresolved: **do not import
  its code** until established. Its public fusion descriptions can inform design.
- Original/GSQ configs and tokenizer/template configuration metadata are hashed;
  full weights and tokenizer execution remain unverified. The GSQ card omits MTP.
- Future Flash-Next uses a separately retrieved Qwen Community License 1.0;
  it is out of scope. Do not transfer dense-model license assumptions to it.
- Pinned host HTTP/JSON/TOML headers and licenses are independently verified in
  `third_party/lock.json`; no inference-engine source has been copied yet.

## Blocking gates and next work

1. **M0.1 / M1.6:** obtain physical original-4080 native Windows/MSVC
   build/device-link/run evidence and exact hardware/toolchain/environment data
   first. Linux has its own deferred qualification gate; no Windows result can
   stand in for it.
2. **M0.2 / M0.5:** build and run the pinned 4080 reference; record actual phase/site
   precision, rounding, sampling/head/KV/graph settings and any A16 patch hash.
   Freeze mandatory matched cases/OS scope before tuning; no current suite is
   frozen and an empty suite is not a pass.
3. **M0.3 / M3:** verify actual original/GSQ/MTP/vision/tokenizer inventories,
   checksums, packing/scale contracts, shared bindings and component compatibility.
   Pin GGUF specification/reader inputs and verify an exact GSQ/Q4 GGUF mapping;
   the format decision is not encoding interoperability or byte-level verification.
   Retrieved metadata/publisher claims do not satisfy byte-level verification.
4. **M0.6–M0.7:** produce independent encoding/layer/state fixtures and original-BF16
   baseline evidence; review dataset splits, metrics, budgets, confidence and
   sample sizes before affected conversion/optimization. Null budgets stay blocked.
5. **M2:** build bounded context planning and metadata-driven startup orchestration
   on the neutral contracts/registries and ownership primitives. Complete concrete
   model/state/frontend and generation-step contracts only from verified inputs.
   Keep synthetic fixtures test-only. Proceed to M3's narrow GGUF converter/reader
   and exact frontend only after the model/device work boundary is
   approved and source inputs are sufficiently verified. No silent fake backend or
   arbitrary-model importer.

M0/M1/M2 remain **partial**. No optimized plain-text checkpoint, full-feature
candidate or performance-qualified release is claimed. All eleven cache formats,
native MTP and both vision placements remain full-release requirements. Missing
measurements, quality policy and profile support cannot be waived for speed.
