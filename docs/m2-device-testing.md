# Native CUDA projection and session device tests

`tools/m2_cuda.py` configures the platform CUDA preset, builds and device-links
`saga_cuda_projection_test` and `saga_cuda_session_test_device`, runs both, and
validates their separate JSON reports. Work/acceptance is tracked in
[CUDA-01 / #16](https://github.com/Teriyake4/Saga/issues/16); Windows and Linux
portability evidence remains separately tracked in [#2](https://github.com/Teriyake4/Saga/issues/2)
and [#4](https://github.com/Teriyake4/Saga/issues/4).

Neither test establishes production model support, HTTP readiness, full-model
quality, tuned performance or complete device-failure qualification. The
session test uses a synthetic model definition with real registered BF16
projection kernels and the production backend/session lifecycle.

## Prerequisites and bounded commands

- Native **Linux x86_64** or native **Windows x64/MSVC** and an original physical
  GeForce RTX 4080 (SM89, not Super or laptop).
- CUDA Toolkit/NVCC 12.8 or newer, CMake, Ninja and compatible host compiler on PATH.
- On Windows, an x64 Visual Studio developer shell and Python.
- macOS, GPU-less OrbStack, ARM64 compilation and WSL are not substitutes for
  either native physical acceptance environment. A host platform check alone
  cannot authenticate native hardware/OS provenance; record it independently.

From the exact checkout to be tested:

```sh
# Native Linux x86_64
python3 -B tools/m2_cuda.py --iterations 30 --timeout-seconds 1200
```

```powershell
# Native Windows x64 Visual Studio developer shell
python -B tools/m2_cuda.py --iterations 30 --timeout-seconds 1200
```

Each subprocess has a 1200-second default timeout; this is not a total-run
timeout. Builds use parallelism 2. Iterations are bounded to 1..10000; this is a
test repetition control, not a performance qualification setting. Each stdout
and stderr stream is independently capped at 16 MiB by default. A fresh unique
directory is created under `build/m2-cuda-evidence/<UTC stamp>-<id>/`.

Before transfer/execution, record `git rev-parse HEAD`, dirty-tree status and a
source snapshot/diff identity including untracked source files. Preserve OS,
CPU/architecture, available RAM/disk, GPU name/UUID, driver, NVCC/MSVC or host
compiler, runtime and power/background-load conditions. Do not infer the tested
tree from a commit alone when local changes are present.

## Retained artifacts and interpretation

The run directory contains:

- `evidence.json`, command records with exact argv/cwd/timeouts/exit status and
  stdout/stderr paths, verbose build/device-link logs, copied CMake cache and
  generated compiler facts;
- before/after scoped source snapshots and separate executable sizes/SHA-256s;
- projection `runtime.json` / `runtime.stderr.log`;
- session `session-runtime.json` / `session-runtime.stderr.log`.

Source snapshots cover relevant code/tests/tools but remain bounded change
checks, not authenticated repository provenance. Both binaries must report the
same GPU UUID/ordinal/driver/runtime. Failed commands, malformed reports,
truncated logs, unresolved descendant cleanup and source changes force failure.
Timeout cleanup targets process groups on Linux and process trees on Windows;
process termination is a safety boundary, never evidence of a successful fence.

Projection validation checks exact variant/request contracts, finite errors and
timings, queried SM89 kernel resources, measured allocation identities/caps and
consistent host/pinned/device totals. Output values themselves are explicitly
checked for finiteness before the C++ error reduction.

Session validation requires the exact exercised check set: startup-selected real
projection, independent scalar answers, owning bounded outputs, consumer lease
retention, callback failures before/after submission with safe drain, wrong-thread
and unselected-preparation rejection, post-submit cancellation/fencing and repeated
healthy startup/teardown. It validates the measured persistent/bringup allocation
states and runtime overhead. Its fixture-level startup approval is not server
readiness; reports deliberately withhold serving/qualification claims.

The session report explicitly records `device_loss_injection: not_run`. Host fake
runtime tests exercise failed query/record/device-selection/timeout quarantine;
they do **not** prove actual device-loss behavior. A passing device fixture alone
therefore cannot close every #16 failure-path acceptance requirement. Do not reset
or deliberately hang the user's GPU to manufacture that evidence.

Native Windows and native Linux results are independent gates. Retain applicable
Compute Sanitizer results separately (tool/version, invocation, exit status and raw
logs); host ASan/UBSan/TSan are not substitutes for device checks. Missing physical
build/run, sanitizer or failure-class evidence remains explicit rather than being
promoted from host tests or historical smoke reports.

## Host regression tests

With a host preset configured and built, the new tests are bounded CTests:

```sh
ctest --preset host-debug -R '^(cuda_executor|cuda_session|startup|memory)$'
python3 -B -m unittest tests.tooling.test_m2_cuda
python3 -B -O -m unittest tests.tooling.test_m2_cuda
```

These use fake CUDA/report fixtures and require no model or GPU. They cannot
establish native CUDA compilation, device-link, execution or qualification.
