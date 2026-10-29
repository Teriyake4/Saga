# M2 CUDA projection device test

`tools/m2_cuda.py` is a bounded native-device evidence runner for the M2 CUDA
projection test. It configures the existing `cuda-linux` preset on Linux or
`cuda-windows` on Windows, enables only the CUDA test target, builds
`saga_cuda_projection_test` verbosely, runs the test, and validates its JSON
report. It is not a product, serving-readiness, or performance qualification.

## Prerequisites

- Run on native Linux or native Windows; macOS and cross-compilation are rejected.
- CUDA Toolkit/NVCC, CMake, Ninja, and a compatible host compiler must be on PATH.
- On Windows, use an x64 Visual Studio developer shell so CMake/NVCC can use MSVC.
- The runner requires an original GeForce RTX 4080 (SM89); a simulator, mock, or
  report fixture is not device evidence.

## Commands

From the Saga source root, run one command on each native host when evidence for
both platforms is required. The default is 30 iterations per timing batch; each
subprocess is bounded by the default 1200-second timeout. Each stdout and stderr
stream is independently capped at 16 MiB by default; use `--max-log-bytes N` to
change the per-stream limit.

Linux:

```sh
python3 -B tools/m2_cuda.py --iterations 30 --timeout-seconds 1200
```

Windows, from the x64 Visual Studio developer shell:

```powershell
python -B tools/m2_cuda.py --iterations 30 --timeout-seconds 1200
```

Use `--iterations N` to override the bounded test iteration count (1–10000),
`--timeout-seconds N` to adjust the per-command limit, or `--source-root PATH`
for a different checkout. Each run gets a unique directory under
`build/m2-cuda-evidence/<UTC stamp>-<id>/`, with its build tree in `binary/`.

## Retained evidence and gates

The run directory contains command JSON records with exact argv, working
directory, timeout, exit status (or timeout marker), and stdout/stderr paths;
tool version logs; verbose configure/build logs; copied CMake cache and generated
compiler facts; before/after source snapshots; `runtime.json` and
`runtime.stderr.log`; and the built executable's SHA-256 and size. The runner
compares scoped source snapshots and fails if relevant source changed. On Linux,
timeouts terminate the command process group. On Windows, timeout cleanup uses
`taskkill /T /F` for the command process tree, then a bounded direct-process kill
fallback. Failed tree cleanup is recorded as unresolved and forces failure. Output
that exceeds the per-stream cap is truncated at the cap, recorded as
`log_limit_exceeded`, and forces failure; truncated runtime JSON is never validated.

A successful run requires the test's physical-pass status, the literal
`qualification: not_claimed`, `readiness_approval: false`, passed ledger
reconciliation, an original RTX 4080 identity and SM89 architecture, baseline and
WMMA variant IDs, finite nonnegative errors no greater than the test's `0.02`
threshold, queried kernel resources with SM89 binary versions and positive
occupancy, finite timing values (at least five samples for each variant), and
allocation live/peak values within reported
caps. The runner itself never changes readiness or makes qualification claims.

The runner has not been executed against hardware here. Host-only contract tests
exercise rejection paths and do not count as device evidence. Execute the native
commands above on the target hosts and review the retained evidence before making
any hardware-specific statement.
