# Saga implementation status

> **Current snapshot, 29 October 2026:** M0/M1/M2 code work and easy-new-model scaffolding are authorized and present in the working tree; current macOS/Linux host validation passed 25/25 CTest groups in ten configurations, including sanitizers. The retained Windows report records Ninja/MSVC Debug and Release 26/26 CTest groups each, packaging checks, and physical original-4080 CUDA smoke 19/19. Linux ARM64 Clang 20/CUDA 12.8 compilation reportedly produced device-only SM89 and host objects, not a device link or run. Windows rerun and physical Linux GPU testing are later tasks. These reports do not close acceptance gates.

Historical snapshots/counts remain exactly historical: 25/25 in ten macOS/Linux host M2 configurations, 26/26 per Windows configuration and 19/19 Windows smoke checks, and earlier M1/M0 counts are not new current-tree results. M0 has no reference build/run/dispatch, frozen suite, original-BF16 baseline, or approved quality budgets. Full M0/M1/M2 gates remain open. Preserve the 95% throughput / 110% uncached-TTFT requirements; do not reinterpret status as passing. See [implementation snapshot](implementation-2026-10-29.md), [HANDOFF](HANDOFF.md), and [operator provenance](operator-provenance.md).

## M0 / M1 / M2 milestone gates

| Milestone | Status | Implemented evidence | Outstanding exit-gate work |
|---|---|---|---|
| **M0 - sources and reference baseline** | **Partial / blocked** | Pinned sources and tokenizer/template bytes; frontend input index/candidates; artifact inventory, reference capture scaffolding, layer/state and pending target-vector fixtures, independent scalar oracle | Narrow source-only Q3-linear/SwiGLU route trace patch exists, but no complete phase/site instrumentation, observed dispatch, reference build/run, or frozen suite; original-BF16 baselines and quality budgets remain open; source-acquisition receipt is not runtime evidence; adoption decisions recorded in `operator-provenance.md`; Switchyard reuse remains license-blocked |
| **M1 - foundation and server shell** | **Implementation present; exit blocked/unqualified** | Historical Windows Ninja/MSVC Debug/Release 26/26 and physical original-4080 smoke 19/19 reports; Linux verifier timeout fix and native Linux smoke runner are present. See dated snapshots below. | Current-tree Windows rerun and physical Linux CUDA compile/device-link/run are pending; ARM64 device-only/host-object compilation is not device link/run or qualification. VS2022-generator/hosted-CI status remains unestablished |
| **M2 - contracts, planning and CUDA executor** | **Code present; full gate blocked** | Prior host contracts/planner/startup and historical 25/25 matrices; current tree adds BF16 baseline and SM89 WMMA CUDA path plus ledger-backed Executor and CUDA tests/driver | No claim of current rerun. Executor is not an `IGpuBackend` serving integration; no production registrations, full model execution, measured tuning, physical Linux run, or M0 quality/reference gates |

Historical M2 validation: 25/25 CTest groups in ten macOS/Linux configurations; focused groups passed 20 repeats on each TSan host. These historical figures and logs remain preserved in [m2-evidence.md](m2-evidence.md); they are not validation of the newly changed tree. Current-tree host validation also passed 25/25 groups in all ten configurations; the final Linux evidence runner passed install/config/source-stability checks. See [the current snapshot](implementation-2026-10-29.md) for logs and explicit Python skips. CUDA validation remains compile-only, not device-link/run. Model quality/fit/performance and physical Linux CUDA remain open; M1 cross-platform exit remains blocked.

## Runtime model format decision (planned)

**GGUF is the required runtime input container**, not a proprietary `.engine` or
`.ninfer` model file. This is a documentation requirement, not implemented loader
support. The initial architecture/weight encodings remain narrowly scoped; GSQ3
is not interchangeable with GGUF Q3_K/IQ3 recipes. The exact verified GSQ/Q4 GGUF
profile remains M0/M3 work. The specification/selected reader sources are now
content-verified at a pinned revision; this is not a compiled/implemented reader. See
[artifact-format.md](artifact-format.md).

GGUF does not dictate the GPU execution layout: bounded one-time or cached native
repacking may preserve the declared represented values, with all transient and
steady-state memory accounted for. It does not establish NInfer parity or waive
any existing performance/quality gate. No GGUF reader/writer/converter, universal
GGUF-model support or new model/device implementation is supplied by this decision.

## Implemented and locally exercised

| Area | Present implementation | Evidence/limits |
|---|---|---|
| Build | C++20 CMake targets, host presets, Linux verifier and CUDA smoke tooling | Historical host/Windows counts preserved; current tree not rerun. Linux ARM64 SM89 device-only and host-object compile reported, no device link/run |
| Server shell | Strict TOML + optional one-model INI + CLI merge, structured errors/logs, liveness, unavailable readiness/generation, empty model list and truthful metrics | Real-process HTTP/config tests; no model catalog/router or M5/M6 correct text API claim |
| Bounds/security | Fixed HTTP workers/pending connection bound, header-count/body limits, strict framing/chunk/trailer parsing, I/O idle timeouts, localhost default, environment bearer auth | Raw-socket/idle/disconnect/restart tests; one request per connection; no TLS or absolute trickle-client deadline; trusted reverse proxy required for remote exposure |
| Kernel contracts | Startup descriptor/support/prepare, well-formedness/compatibility-before-ranking, optional profile preferences, nonvirtual prepared launchers and actual workspace checks | Synthetic specialization guard/fallback/lifetime tests, invalid enum/GSQ3-group tests and zero-workspace alignment tests; no real CUDA inference implementation |
| Resource planning | Checked per-domain ledger, sharing identities, exclusive backing ownership, synchronized lease-release accounting, separate candidate/grant/reconcile/approval types and overhead caps | Fake allocation/measurement tests, including reentrant transactions, verified owner coverage, concurrent releases and failed bring-up; not actual driver measurements or model fit |
| KV metadata | Eleven canonical formats, alias normalization, exact block sizes and checked main-cache page/layout sizing | All proposal main-cache byte fixtures; no quantizer/attention/verification execution support |
| Platform | Separate `saga_platform` target selects shared + native Windows or reusable POSIX sources; mapping, Unicode argv/environment and RAII shutdown adapters | macOS tests include >4-GiB windows, Unicode/environment bounds and prior signal restoration; OrbStack ARM64 host tests pass, but Windows runtime, Linux x86_64 and Linux CUDA remain unqualified |
| Runtime primitives | Resource-retaining completions; bounded execution worker/reserved cleanup; generation-tagged sequence retirement; bounded single-active admission and committed-byte output | Unit and cross-component tests; owner-thread contracts enforced where specified; no HTTP generation integration or device synchronization |
| Model/backend contracts | Structural artifact/model/program validation, distinct component/tensor identities, exact tied-weight aliases, independent owning registries, typed bounded model-step requests/results and host startup orchestration | Synthetic grant-backed composition, failure and lifetime tests; no production model/backend registrations or HTTP readiness/generation integration |
| CUDA / M2 executor | Registered BF16 baseline and WMMA SM89 implementation, CUDA target/tests and ledger-backed Executor | Reported Clang 20 Linux ARM64 CUDA 12.8 device-only SM89 + host-object compilation only; no device link/run, NVCC, Windows retest, GPU correctness/performance, or serving backend |
| Source locks | Five original reference pins plus the GGUF specification/selected ggml reader-source pin, actual source/build/license hashes and original/GSQ metadata | 114 inputs, 105 recorded-verified; new optional commit API 403 remains unverified; no reference binary, verified weights or matched benchmark |

The M2 host contract/planning/startup slice is implemented, but remains synthetic
and is not an inference backend. Production registrations/sessions and real model
inventory are absent. The planner evaluates explicit or finite auto-context candidates
from supplied metadata; auto candidates are bounded and descending, and duplicate
entries are rejected rather than silently deduplicated. It picks the highest statically
eligible candidate only: a later statically eligible kernel, preparation, warmup,
capture, or reconciliation failure does not trigger lower-context retry. Unknown
enabled allocations remain explicitly unresolved, never fabricated.
Runtime cancellation/abandonment/failure is not a physical fence. A backend must
implement `drain()` as a physical barrier on every failure/throw path; a backend
factory that throws after submitting work retains responsibility to drain it.
Memory accounting conservatively sums unique peaks; it is not driver interception or
proof of real model fit. Plan identity is canonical length-prefixed serialization,
not cryptographic attestation. Shared alias caps are shared accounting constraints,
not allocation memoization. Full caveats and exact tests are in [m2-evidence.md](m2-evidence.md).

## Preserved M0 evidence continuation (historical context)

The earlier user authorization covered M0 work with GPT6 Luna implementers and parallel independent
reviewers; it is historical, not the current boundary. Three reviewed host-side slices add GGUF source provenance, fail-closed
source/local artifact byte auditing, and small independent storage/projection
fixtures. Original source/evidence history is preserved; the optional new commit
API rate limit is recorded, not hidden. No weights or upstream executables were
run/downloaded, and no numerical quality or hardware result was invented.

The complete M0 gates remain open. The independent scalar fixture does not fill
the complete model/layer/state inventory, and checksum matching does not establish
component compatibility. Hardware/reference execution, verified model inputs,
actual GGUF mapping, operator-adoption decisions and reviewed quality/suite freezes
remain outstanding. See [m0-evidence.md](m0-evidence.md) for exact progress and
commands. The M0 audit intentionally returns exit 1 (blocked).

## Prior neutral model-contract increment

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
- This was a separately approved one-step increment; subsequent M0 evidence work
  is recorded above. No planner, startup orchestrator, controller, HTTP generation,
  model inspection, kernels or GPU tests were added by the contract step.

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
The initial retrieval snapshot contained **104 inputs, 96 HTTP-200 body hashes and
eight HTTP-404 probes**; all five original pinned commit responses matched the SHA.
The later GGUF append brings the inventory to 114 inputs/105 recorded-verified,
with its optional pinned commit API still an unverified 403.
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

## M1.1–M1.6 evidence and current boundary

M1 source slices are implemented. Native Windows Ninja/MSVC Debug/Release,
packaging and physical original-4080 CUDA smoke passed; see
[Windows / RTX 4080 evidence](windows-4080-evidence.md). Its full both-OS exit
remains unqualified pending native Linux original-4080 CUDA evidence.
M2 host-testable implementation is tracked in [m2-evidence.md](m2-evidence.md);
this does not authorize M3+ inference integration. See [m1-evidence.md](m1-evidence.md)
and [linux-build.md](linux-build.md) for historical M1 evidence and the Linux
workflow. Neither these host tests nor the small GPU smoke qualify model support,
production backend fences, fit or performance.

## Blocking gates and next work

1. **M0.1 / M1.6:** retain the completed native Windows original-4080 evidence
   and obtain the separate physical CUDA build/device-link/run and environment
   evidence on native Linux. Complete reference-run power/load conditions when
   reference work is authorized. OrbStack ARM64 host evidence is not a CUDA run;
   neither OS's result can stand in for the other.
2. **M0.2 / M0.5:** build and run the pinned 4080 reference; record actual phase/site
   precision, rounding, sampling/head/KV/graph settings and any A16 patch hash.
   Freeze mandatory matched cases/OS scope before tuning; no current suite is
   frozen and an empty suite is not a pass.
3. **M0.3 / M3:** verify actual original/GSQ/MTP/vision/tokenizer inventories,
   checksums, packing/scale contracts, shared bindings and component compatibility.
   Use the pinned GGUF specification/selected reader sources, pin any further
   implementation inputs and verify an exact GSQ/Q4 GGUF mapping;
   the format decision is not encoding interoperability or byte-level verification.
   Retrieved metadata/publisher claims do not satisfy byte-level verification.
4. **M0.6–M0.7:** produce independent encoding/layer/state fixtures and original-BF16
   baseline evidence; review dataset splits, metrics, budgets, confidence and
   sample sizes before affected conversion/optimization. Null budgets stay blocked.
5. **M2 remaining:** use [m2-evidence.md](m2-evidence.md) for the exact host slice
   status. Do not infer production readiness from synthetic tests. Keep production
   registrations absent and proceed to M3 only with a separately authorized scope
   and sufficiently verified source inputs. No silent fake backend or arbitrary-model
   importer.

M0/M1/M2 remain **partial at their full gates**. No optimized plain-text checkpoint, full-feature
candidate or performance-qualified release is claimed. All eleven cache formats,
native MTP and both vision placements remain full-release requirements. Missing
measurements, quality policy and profile support cannot be waived for speed.
