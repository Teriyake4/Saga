# Implementation and evidence snapshot — 2026-10-29

This is a dated integration and bounded host-validation snapshot, not milestone sign-off. New code received independent GPT-6 Luna implementation/review/fix/re-review passes. The current host results below are separate from historical evidence and do not change device, model, quality or performance gates. Native Windows retesting and physical original RTX 4080 execution remain later work, as does native Linux GPU execution.

## Present in the working tree

- **M0:** artifact inventory, frontend/reference capture tooling, tokenizer/template checks, independent layer/state oracle and partial/pending target-vector fixtures. A small independent target-vector GDN case accepts `alpha = exp(g)`, `beta`, and q/k after optional normalization, with a small 2x2 recurrence formula. It is a hand-sized oracle/fixture, not a compiled target kernel, full layer/state fixture, or full-model validation. See the M0 quality evaluator docs as well as the fixture evidence. Ten pinned frontend files (five per model) are hash-verified. The `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` pairs are byte-identical; the model and generation configs differ. Tokenizer/template identity does not imply model compatibility. Tokenizer token JSON reports 248,077 entries while config reports vocabulary size 248,320; reconcile this explicitly rather than normalizing it away. Frontend input-index candidates are 128, 2048, and 7680 tokens plus a 512-token reserve.
- **M0 still blocked:** suite remains **UNFROZEN**. `bench/reference/dispatch-audit.patch` provides narrow, source-only Q3-linear/SwiGLU route tracing; phase is unknown and graph-capture selection is not graph replay. There is no complete phase/site instrumentation or observed dispatch, reference build/run, original-BF16 baseline, or approved quality budget. `bench/reference/reference-source-acquisition.json` records acquisition of a 10.6 MB source archive with SHA/sourceDigest, which is source provenance only—not runtime evidence. The fixtures remain partial.
- **M1:** Linux verifier timeout handling fix and native Linux CUDA smoke runner are present. Existing Windows report from 2026-10-29 records Ninja/MSVC Debug and Release at 26/26 CTest groups each, packaging/config checks, and physical original-4080 smoke at 19/19. Treat these as reported historical evidence; raw Windows bundle is unavailable locally and no rerun was performed against this tree.
- **M2/CUDA:** registered BF16 baseline and WMMA SM89 path, CUDA tests/driver, and a ledger-backed Executor are present. The Executor is not an `IGpuBackend` serving integration. Reported compilation on Linux ARM64 with Clang 20 and locally available CUDA 12.8 wheels reached device-only SM89 and host-object compilation. This was **not** device-link, NVCC, Windows compilation, device execution, correctness qualification, tuning, or performance measurement.
- **New model scaffold:** `tools/new_model.py` and `docs/adding-models.md` offer generated scaffolding. Generated code is compile/fail-closed scaffolding, not model support or a registered runnable model. It does not implement weights, tokenizer, kernels, or serving.

## Resource and acceptance constraints

The OrbStack Linux VM has no GPU and 8 GiB RAM. The Mac had about 19 GiB free disk at this snapshot and cannot hold/run the full BF16 model. No large artifact download or full BF16 baseline was attempted. Windows raw evidence bundle was not present locally, so its detailed report remains referenced as a historical record rather than independently re-audited here.

## Current host validation

All configurations used image decoding ON and CUDA OFF. Configure/builds passed with parallelism 2; full CTest was rerun after the final test-only portability fixes. No final validation command timed out.

| Host / configuration | Result |
|---|---|
| macOS `host-debug`, `host-release`, `host-sanitized`, `host-thread-sanitized` | 25/25 CTest groups passed in each |
| OrbStack Linux `linux-gcc-debug`, `linux-gcc-release` | 25/25 CTest groups passed in each |
| OrbStack Linux `linux-clang-debug`, `linux-clang-release`, `linux-clang-sanitized`, `linux-clang-thread-sanitized` | 25/25 CTest groups passed in each |
| macOS Python 3.9.6 bench suite, normal and `-O` | 124 total: 119 executed, 5 explicit skips in each mode |
| macOS Python 3.9.6 tooling suite, normal and `-O` | 45 total: 38 executed, 7 explicit skips in each mode |
| Optional pinned frontend environment | 11/11 frontend tests passed; actual offline comparison/rendering passed |
| Dependency hashes / source manifest structure / whitespace | Passed; structure-valid is not qualification |

The full offline M0 audit was also run and returned exit 1, `blocked`, with 18 unresolved evidence gates; its report is `build/m012-validation-logs/source-audit-final.json`. This is an expected blocked qualification result, not a passing model/reference audit.

CTest group success includes intentional internal Python skips. On macOS, the bench skips cover optional frontend dependencies and native Windows artifact opening; tooling skips cover PowerShell availability and the Linux `/proc` regression. That Linux regression separately passed 30 repeated executions in OrbStack. Optional frontend tests ran in the pinned local environment rather than becoming mandatory host-build dependencies.

Logs are retained at `build/m012-validation-logs/`, with final CTest filenames ending in `-ctest-final-portable.log`. The complete Linux verifier also passed configure/build/25-group CTest/install/TOML and INI checks/dependency verification with equal scoped before/after snapshots: `build/linux-evidence/20261029T190231Z-7af2c531abd0/evidence.json`. These snapshots have the existing bounded M1 scope, not complete or authenticated repository provenance.

Earlier failures are preserved: a Python 3.14 JSON-depth test exposed reliance on runtime recursion limits (fixed with an explicit depth bound); Linux lacked the optional `patch` executable (now an explicit applicability-test skip); and a `/proc` process-exit race in the timeout regression was fixed before the successful final verifier run. Earlier evidence directories were not overwritten.

CUDA compile-only logs, exact commands, wheel/tool hashes and source hashes are retained in `build/cuda-compile-evidence/README.md`. The final `resources.cpp` host compile, `projection.cu` host and SM89 device compiles, and `projection_test.cu` host compile all exited 0 with unchanged source hashes and no header shim. Clang 20.1.2 emitted a CUDA 12.8 partial-support warning. The local wheel toolkit lacks `fatbinary`/`nvlink`, so no combined CUDA executable or device link was produced. No CUDA execution, event timings, correctness, tuning or Windows result is claimed.

The historical M2 matrix, Windows host matrix (26/26 per configuration), and Windows CUDA smoke (19/19) remain separate, dated evidence in their original documents.

## Consolidated Windows developer-shell commands (future rerun)

From a PowerShell 7.4+ x64 Visual Studio developer shell with a CUDA-supported MSVC toolchain and Ninja, on the native Windows target:

```powershell
pwsh -NoProfile -File .\tools\windows\verify.ps1 -Configuration Debug -ConfigurePreset windows-ninja-debug -EnableImageDecoder
pwsh -NoProfile -File .\tools\windows\verify.ps1 -Configuration Release -ConfigurePreset windows-ninja-release -EnableImageDecoder
pwsh -NoProfile -File .\tools\windows\cuda-smoke.ps1 -Device 0
python .\tools\m2_cuda.py
```

The first two commands produce host evidence; the third requires the physical original desktop RTX 4080 (Device 0) and checks the existing portability smoke. `tools/m2_cuda.py` is a distinct M2 CUDA test driver and is not a substitute for physical device evidence. Retain fresh reports and record toolchain/device identity. These are recommended commands, **not results from this snapshot**.

## M0 evidence map and remaining work

- [Artifact inventory](m0-artifact-inventory.md), [frontend index](m0-frontend.md), and [reference capture](m0-reference-run.md): current candidate inputs and capture plumbing, not a frozen acceptance suite or reference execution.
- [M0 evidence ledger](m0-evidence.md): verify actual model/component bytes and exact supported GGUF mapping; resolve tokenizer/config size discrepancy; complete independent layer/state/frontend fixtures; build/run pinned reference and observe dispatch; freeze matched cases and native OS scope; obtain original-BF16 baselines; review quality policy/budgets and provenance. [M0 quality evaluation](m0-quality-evaluation.md) documents the bounded paired token-NLL analyzer; it does not execute a model or approve quality.
- [Operator provenance](operator-provenance.md): independent implementation and independent oracle policy; per-operator adoption decision.
- [M1 evidence](m1-evidence.md): current-tree rerun and physical native Linux CUDA link/run remain open despite historical Windows reported results and limited Linux compilation.
- [M2 evidence](m2-evidence.md): distinguish registered kernels/Executor implementation from production backend, physical fences, measured tuning, and serving integration.

No milestone gate has been changed to pass by this snapshot. The 95% matched decode-throughput and 110% uncached-TTFT limits, quality budgets, exact-profile requirements, and cross-platform/device evidence remain blocking requirements.
