# M2 implementation and evidence ledger

Historical host snapshot: **28 October 2026**. The prior M2 host implementation was authorized and implemented; ten host matrix configurations and specified repeat tests passed then. Those counts and logs remain historical and are not current-tree validation. The 29 October tree adds registered BF16 baseline/SM89 WMMA code and a ledger-backed Executor; reported Linux ARM64 Clang 20/CUDA 12.8 compilation reached device-only SM89 and host objects only, with no device link/run. The Executor is not an `IGpuBackend` serving integration. Current host validation passed 25/25 CTest groups in all ten macOS/Linux configurations, and the fresh Linux evidence runner passed install/config/scoped-source stability checks. Exact logs and skips are in the current snapshot; physical/production gates remain open. See [implementation snapshot](implementation-2026-10-29.md) and [operator provenance](operator-provenance.md).

## Per-task implementation and evidence

| Task | Current source / focused tests | Implementation and limit | Evidence state |
|---|---|---|---|
| M2.1 model/program contract | `include/saga/runtime/model_contracts.h`, `src/runtime/model_contracts.cpp`; `tests/model_contracts_test.cpp` | Structural artifact/model/program validation, component/tensor identities, exact alias boundaries, capability/program checks. Not payload verification or a complete production schema. | Implemented, dual reviewed; targeted Debug/Release model-contract test verified. |
| M2.2 registries, session and startup kernel interfaces | `include/saga/backend/contracts.h`, `src/backend/contracts.cpp`; `include/saga/backend/device_profile.h`; `tests/contracts_test.cpp`, `tests/device_profile_test.cpp`, `tests/model_contracts_test.cpp` | Independent owning registries and startup-prepared launch interfaces. Synthetic registrations only; no production model/backend/session or compiled SM89 algorithm. | Implemented host contract/profile slice, dual reviewed; targeted Debug/Release contracts/profile/model-contract tests verified. |
| M2.3 startup composition | `include/saga/runtime/startup.h`, `src/runtime/startup.cpp`; `tests/startup_test.cpp` | Inspect/describe, resolve profile/program, plan, grant, instantiate, warm/capture, reconcile and approval ownership. Backend failure/throw paths require physical `drain()` barrier. | Implemented host composition, dual reviewed; targeted Debug/Release startup tests verified. |
| M2.4 lifetime primitives | `include/saga/runtime/`, `src/runtime/`; `docs/runtime-references.md` | Completions, leases, worker/cleanup, sequence retirement, admission and output buffer. Failed/cancelled/abandoned outcome is not a physical fence; not wired to HTTP. | Historical targeted/full evidence remains labeled in `docs/HANDOFF.md` and `docs/runtime-references.md`; not a claim of device synchronization. |
| M2.5 allocation ledger | `include/saga/memory/ledger.h`, `src/memory/ledger.cpp`, `include/saga/memory/planner.h`, `src/memory/planner.cpp`; `tests/memory_test.cpp`, `tests/resource_planner_test.cpp` | Checked domain budgets, metadata terms, cache sizing, sharing identities, alignment/page rounding, grant/reconciliation types. Accounting is metadata-based and conservative, not actual driver interception or model-fit proof. | Implemented, dual reviewed; targeted Debug/Release planner/memory tests verified. |
| M2.6 context planning | `include/saga/memory/planner.h`, `src/memory/planner.cpp`; `tests/resource_planner_test.cpp`; startup path in `src/runtime/startup.cpp`, tested in `tests/startup_test.cpp` | Explicit context fails if infeasible. Auto candidates are finite, bounded and descending; duplicate entries are rejected rather than silently deduplicated; highest statically eligible candidate is chosen. Unknown enabled allocations remain unresolved, never invented. No lower-context retry after statically eligible kernel, preparation, warmup, capture or reconcile failure. | Implemented, dual reviewed; targeted Debug/Release planner/startup tests verified. |
| M2.7 fake backend/profile integration | `src/backend/device_profile.cpp`, `src/runtime/startup.cpp`; `tests/device_profile_test.cpp`, `tests/startup_test.cpp` | Synthetic tests exercise capability/layout/numeric/workspace guards, profile preference ranking without bypass, fallback/error paths, startup failure unwinding and ownership. | Implemented, dual reviewed; targeted Debug/Release profile/startup tests verified. |
| M2.8 profile policy | `include/saga/backend/device_profile.h`, `src/backend/device_profile.cpp`; `tests/device_profile_test.cpp` | Device and optional pair profiles constrain/rank already-valid implementations; profiles are not hardware attestation. | Policy implemented and targeted tests verified. Real compiled SM89 algorithms, tile/pipeline variants and measured tuning remain outstanding (M2.8/M4/M8). |
| M2.9 transitions/failure paths | `src/memory/ledger.cpp`, `src/runtime/startup.cpp`; `tests/resource_planner_test.cpp`, `tests/startup_test.cpp` | Candidate/grant/approval, unresolved terms/caps, partial initialization, warmup/capture/reconciliation and drain ordering covered with synthetic backend. | Implemented host slice, dual reviewed; targeted Debug/Release tests verified. Physical fences and real device measurements remain open. |
| Model step/request/result contracts | `include/saga/runtime/model_contracts.h`, `src/runtime/model_contracts.cpp`; `tests/step_test.cpp`, startup integration in `tests/startup_test.cpp` | Typed bounded owning step results and finite frontend/request validation; not tokenizer execution, model math, sampler, MTP or inference. | Implemented, dual reviewed; targeted Debug/Release step/model-contract tests verified. |

## Final host validation

All ten configurations below passed **25/25 CTest groups**, rebuilt after the final
test-only early-notification regression. Image decoding was ON and CUDA OFF in each
configuration. Toolchains: AppleClang 21.0.0, arm64-darwin27, macOS; Ubuntu 25.04
aarch64, GCC 14.2.0 or Clang 20.1.2, CMake 3.31.6, Ninja 1.12.1.

| Host / compiler | Configure preset | Result / log |
|---|---|---|
| macOS / AppleClang | `host-debug` | 25/25; `build/m2-validation-logs/host-debug-ctest-final.log` |
| macOS / AppleClang | `host-release` | 25/25; `build/m2-validation-logs/host-release-ctest-final.log` |
| macOS / AppleClang | `host-sanitized` (ASan+UBSan) | 25/25; `build/m2-validation-logs/host-sanitized-ctest-final.log` |
| macOS / AppleClang | `host-thread-sanitized` (TSan) | 25/25; `build/m2-validation-logs/host-thread-sanitized-ctest-final.log` |
| Linux / GCC 14.2.0 | `linux-gcc-debug` | 25/25; `build/linux-gcc-debug/Testing/Temporary/LastTest.log` |
| Linux / GCC 14.2.0 | `linux-gcc-release` | 25/25; `build/linux-gcc-release/Testing/Temporary/LastTest.log` |
| Linux / Clang 20.1.2 | `linux-clang-debug` | 25/25; `build/linux-clang-debug/Testing/Temporary/LastTest.log` |
| Linux / Clang 20.1.2 | `linux-clang-release` | 25/25; `build/linux-clang-release/Testing/Temporary/LastTest.log` |
| Linux / Clang 20.1.2 | `linux-clang-sanitized` (ASan+UBSan) | 25/25; `build/linux-clang-sanitized/Testing/Temporary/LastTest.log` |
| Linux / Clang 20.1.2 | `linux-clang-thread-sanitized` (TSan) | 25/25; retained full stdout: `build/m2-validation-logs/linux-clang-thread-sanitized-ctest-final.log`; repeat log: `build/m2-validation-logs/linux-clang-thread-sanitized-repeat-final.log` |

The four focused CTest groups `startup`, `step`, `model_contracts`, and
`runtime_integration` each passed `--repeat until-fail:20` on macOS TSan and Linux
TSan: 80 successful executions per OS. No command timed out.

The direct bounded CMake/CTest commands (run each row on its named host) are:

| Preset | Configure | Build | Full test |
|---|---|---|---|
| `host-debug` | `cmake --preset host-debug -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset host-debug --parallel 2` | `ctest --preset host-debug` |
| `host-release` | `cmake --preset host-release -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset host-release --parallel 2` | `ctest --preset host-release` |
| `host-sanitized` | `cmake --preset host-sanitized -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset host-sanitized --parallel 2` | `ctest --preset host-sanitized` |
| `host-thread-sanitized` | `cmake --preset host-thread-sanitized -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset host-thread-sanitized --parallel 2` | `ctest --preset host-thread-sanitized` |
| `linux-gcc-debug` | `cmake --preset linux-gcc-debug -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset linux-gcc-debug --parallel 2` | `ctest --preset linux-gcc-debug` |
| `linux-gcc-release` | `cmake --preset linux-gcc-release -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset linux-gcc-release --parallel 2` | `ctest --preset linux-gcc-release` |
| `linux-clang-debug` | `cmake --preset linux-clang-debug -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset linux-clang-debug --parallel 2` | `ctest --preset linux-clang-debug` |
| `linux-clang-release` | `cmake --preset linux-clang-release -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset linux-clang-release --parallel 2` | `ctest --preset linux-clang-release` |
| `linux-clang-sanitized` | `cmake --preset linux-clang-sanitized -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset linux-clang-sanitized --parallel 2` | `ctest --preset linux-clang-sanitized` |
| `linux-clang-thread-sanitized` | `cmake --preset linux-clang-thread-sanitized -DSAGA_BUILD_TESTS=ON -DSAGA_ENABLE_CUDA=OFF -DSAGA_ENABLE_IMAGE_DECODER=ON` | `cmake --build --preset linux-clang-thread-sanitized --parallel 2` | `ctest --preset linux-clang-thread-sanitized` |

The repeat test invocations on both TSan hosts were:

```sh
ctest --preset host-thread-sanitized -R '^(startup|step|model_contracts|runtime_integration)$' --repeat until-fail:20
ctest --preset linux-clang-thread-sanitized -R '^(startup|step|model_contracts|runtime_integration)$' --repeat until-fail:20
```

Additional checks passed: `git diff --check`, `python3 -B tools/verify_dependencies.py`,
and `python3 -B tools/verify_sources.py --structure-only`. The full source audit remains
a separate blocked qualification gate as documented in `docs/source-lock.md`.

Product-only Release build/install and installed configuration checks for TOML and
TOML+INI passed earlier on both platforms, with production source unchanged. Actual
build/install trees were `build/m2-macos-product` and `build/m2-macos-product-install`
on macOS, and `build/m2-linux-product` and `build/m2-linux-product-install` on Linux.
The installed TOML and TOML+INI config checks passed; Linux retained
`build/m2-validation-logs/linux-product-installed-config-checks.log`. These product
build/config checks are not M2 or inference qualification; the paths are the actual
runs, not a claim that a product preset produced them.

Validation also passed `git diff --check`, dependency verification, source-structure
checks, and the full CTest matrices. The Linux verifier has a known timeout descendant-
cleanup issue; it is preserved, not fixed or newly qualified here. Final Linux CTest
results came from direct bounded preset commands. The Linux TSan full-suite evidence
is the retained stdout log above, not the overwritten `LastTest.log`; the other Linux
configurations have their 25-test `Testing/Temporary/LastTest.log` files. No new
verifier snapshot or provenance claim is made. Historical M1 evidence and counts in
`docs/m1-evidence.md` remain historical.

## Behavioral and integration limitations

- Auto context planning is a static finite descending search; runtime failure after
  selecting a statically eligible candidate propagates and does not fall back to a
  smaller context.
- Unknown enabled component/overhead terms remain unresolved. A generic overhead
  cap cannot substitute for unknown engine allocations; no missing values are
  fabricated.
- Plan identity is a canonical length-prefixed serialization of inputs, not a
  cryptographic hash, authentication token, or hardware qualification.
- Exact physical allocation aliases are counted once only when identity and
  metadata agree. Shared alias caps constrain grant accounting; they do not imply
  factory memoization or shared allocation creation.
- A backend factory can throw after submitting asynchronous work. The backend must
  retain work/resources and drain to a physical barrier before returning/throwing;
  startup cannot infer a fence from failure, cancellation, timeout, or abandonment.
- `ApprovedRuntime` is host-side startup ownership/approval, not server readiness.
  No production registrations are present and the HTTP server remains not ready
  (`/readyz` 503; chat completions 503 before SSE).
- No GGUF loader, tokenizer, inference operators, CUDA execution, HTTP generation,
  model fit, quality result or performance qualification is included.

## Pending qualification gates

1. Current-tree Windows/MSVC Debug/Release validation remains outstanding; the reported 29 October 26/26 runs per configuration are preserved historical evidence, not a rerun. Physical original-4080 smoke is likewise reported 19/19 in that report.
2. Native Linux physical CUDA device-link/run remains outstanding. The reported Linux ARM64 Clang 20 device-only SM89 and host-object compile is not a link or run. The CUDA Executor/kernel additions require current-tree build/tests and correctness evidence.
3. M0 model/reference evidence, actual component inventories and resource sizes, physical device measurements, production session/backend implementation and measured tuning remain open. The new Executor is ledger-backed but not an `IGpuBackend` serving integration. M0/M1/M2 full gates remain unqualified.
4. M3+ loaders/tokenizers/inference/CUDA operations and HTTP generation are outside
   the current scope; do not represent this document as authorization for them.
