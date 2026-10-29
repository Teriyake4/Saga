# Saga continuation handoff

> **Windows follow-up, 29 October 2026:** native Windows Ninja/MSVC Debug and
> Release passed 26/26 CTest groups each; product install/config checks and all 19
> physical original-4080 CUDA smoke checks passed. Windows-specific fixes and
> retained evidence are in `docs/windows-4080-evidence.md`. The Windows-pending
> statements and Windows next action below describe the historical 28 October
> handoff, not the latest result. Linux physical-CUDA and M0 model/reference/
> quality gates remain open. No M3+ implementation was added.

Historical handoff: **28 October 2026**. Start with `docs/status.md`, this file, and
`docs/m2-evidence.md`; consult `IMPLEMENTATION_PLAN.md` and `PROPOSAL.md` for
requirements and complete release scope.

## Current scope and state

The user explicitly authorized finishing the host-testable M2 implementation,
superseding stale no-M2 restrictions in earlier handoffs/plan notes. M2.1–M2.7/.9
host slices and M2.8 profile policy are implemented and dual reviewed. Final host
matrices passed 25/25 CTest groups in ten configurations after rebuilding with the
final test-only early-notification regression: macOS Debug, Release, ASan+UBSan,
TSan; Linux GCC Debug/Release; Linux Clang Debug/Release/ASan+UBSan/TSan. Image
decoding was ON and CUDA OFF. Four focused groups (`startup`, `step`,
`model_contracts`, `runtime_integration`) each passed 20 repeats on macOS and Linux
TSan (80 executions per OS). No command timed out. Product-only Release build/install
and TOML+INI installed-config checks passed on macOS/Linux earlier, with production
source unchanged. Exact environment, logs and reproducible commands are in
`docs/m2-evidence.md`.

This is not M2 production qualification. Native Windows/MSVC and physical original
RTX 4080 evidence remain outstanding. M0 reference/artifact/quality gates remain
blocked. M3 loaders/tokenizers/inference/CUDA operators and M6 HTTP generation are
out of scope; no production model/backend registrations exist. Server readiness
remains HTTP 503, `/v1/models` remains empty, and chat completions remain HTTP 503
before SSE. Never fabricate model or generation success.

Auto-context selection is a finite, descending static search only. After selecting
the highest statically eligible candidate, kernel/preparation/warmup/capture/
reconciliation failure propagates; it does not retry lower contexts. Unknown enabled
allocations remain explicit unresolved terms and are never fabricated. Runtime
failure, cancellation, and abandonment are not physical fences. Backends must make
`drain()` a physical barrier on every failure path, including factory exceptions
after submitted work. Plan identity is canonical length-prefixed serialization,
not cryptographic attestation. Shared alias caps are shared caps, not allocation
memoization. Real compiled SM89 algorithms and measured tuning remain M2.8/M4/M8
work, not present functionality.

## Code and contract map

- `include/saga/runtime/model_contracts.h`, `src/runtime/model_contracts.cpp`:
  structural artifact/model/program validation, model/runtime/backend/session
  contracts and owning startup-only registries. Synthetic implementations only.
- `include/saga/memory/planner.h`, `src/memory/planner.cpp`:
  category/domain allocation terms, cache metadata, explicit/automatic context
  evaluation, unresolved-term reporting and candidate/grant foundation.
- `include/saga/runtime/startup.h`, `src/runtime/startup.cpp`:
  immutable execution plan, `StartupRequest`, synchronous `StartupCoordinator`,
  startup selection/preparation and `ApprovedRuntime` ownership/teardown.
- `include/saga/backend/device_profile.h`, `src/backend/device_profile.cpp`:
  observed-facts-based device profile, original-4080 named constraints, optional
  pair preferences, guarded program resolution and startup-prepared call list.
- `include/saga/runtime/model_contracts.h` and `tests/step_test.cpp`:
  finite frontend/request validation and typed owning `PendingStep` result contract.
- Tests: `tests/contracts_test.cpp`, `tests/model_contracts_test.cpp`,
  `tests/memory_test.cpp`, `tests/resource_planner_test.cpp`, `tests/startup_test.cpp`,
  `tests/device_profile_test.cpp`, `tests/step_test.cpp`, and
  `tests/runtime_integration_test.cpp`.
- Runtime primitives and caller obligations: `docs/runtime-references.md`.
- Exact M2 evidence and outstanding gates: `docs/m2-evidence.md`.

Important ownership rules: registry operations and startup are externally serialized;
backend/model factory inputs needed later must be copied. Runtime is destroyed before
its borrowed session. Failed/cancelled/abandoned work cannot authorize resource
reclamation; retain and physically drain work before teardown. Startup coordinator,
approval, grant and returned owner are single-owner/thread-serialized. Keep cleanup
capacity reserved before work that can require rollback. No HTTP generation wiring
or production session is implied by these host contracts.

## Evidence and current validation boundary

Older M1/runtime/model-contract histories and counts remain historical evidence.
The M2 full-matrix results in `docs/m2-evidence.md` are from the final source state,
including the test-only early-notification regression; historical M1 evidence is
not rewritten as current validation.

Previous dirty and untracked user work is intentional/provenance-bearing. Preserve
it. Do not reset, amend, delete generated evidence, commit or create branches unless
asked. The Linux verifier's known timeout descendant-cleanup issue remains unresolved;
final Linux test results and retained-log scope are recorded in `docs/m2-evidence.md`.

## Next actions

1. On native Windows, obtain evidence from a PowerShell 7.4+ x64 VS2022 developer
   shell (enable the optional image decoder for both host configurations), then run
   the physical original RTX 4080 smoke on Device 0:

   ```powershell
   pwsh -NoProfile -File .\tools\windows\verify.ps1 -Configuration Debug -EnableImageDecoder
   pwsh -NoProfile -File .\tools\windows\verify.ps1 -Configuration Release -EnableImageDecoder
   pwsh -NoProfile -File .\tools\windows\cuda-smoke.ps1 -Device 0
   ```

   Retain/inspect reports. These Windows and original-4080 gates remain pending;
   host synthetic kernels do not satisfy them. Linux host results do not qualify
   Linux x86_64/CUDA or native Windows.
2. M0 still needs the actual reference build/dispatch, verified model/tokenizer
   inventories, exact GSQ/Q4 profile, frozen matched cases, complete independent
   fixtures and reviewed quality budgets/baselines. Preserve historical source-lock
   retrieval records rather than turning retrieval into execution evidence.
3. M2.8 real SM89 algorithm implementation and measured tuning remain open; M4/M8
   correctness, quality and performance gates remain later work. Do not start M3
   loaders/tokenizers, inference, CUDA ops, HTTP generation, or production
   registration under this M2 host-testable authorization.

## Provenance and reference pointers

- `docs/source-lock.md` and `bench/reference/source-lock.json`: pinned NInfer,
  NInfer-4080, Strata, Switchyard, llama.cpp and GGUF/ggml source paths, revisions,
  licenses and hash/retrieval evidence. Retrieval is not a reference build/run.
- `docs/m0-evidence.md`: M0 gate-by-gate artifact, reference, quality and hardware
  blockers; `docs/m1-evidence.md` and `docs/linux-build.md`: host/platform evidence.
- `docs/runtime-references.md`: historical runtime contracts plus current M2
  planner/profile/startup/step extensions and caller obligations.
- `docs/model-config.md` and `bench/reference/model-config-reference.json`:
  familiar config-option reference, not parser/router reuse.
- `docs/quality-policy.md`: preregistration requirements; unresolved budgets remain
  blockers. `docs/cuda-portability.md`: physical CUDA procedure/evidence requirements.

## Product/runtime facts that must remain truthful

GGUF is the required future input container, not an implemented loader or arbitrary
GGUF support. The optional image decoder is a bounded standalone adapter, not
server vision. Capability/configuration metadata does not establish support. The
server reports `/readyz` 503 and no completions. Host test success does not establish
physical fences, model fit, quality, CUDA correctness, performance, or production
readiness. See `docs/status.md`, `docs/m1-evidence.md`, `docs/m0-evidence.md`,
`docs/runtime-references.md`, `docs/artifact-format.md`, and `docs/cuda-portability.md`.
