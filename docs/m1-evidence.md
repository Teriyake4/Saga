# M1.1–M1.6 implementation and evidence

> **29 October integration note:** the Windows report records Ninja/MSVC Debug and
> Release at 26/26 CTest groups each, packaging checks, and physical original-4080
> CUDA smoke at 19/19; see [Windows evidence](windows-4080-evidence.md). Raw report
> bundle is not locally available and these runs were not repeated against the
> current tree. Linux verifier timeout fix and CUDA smoke runner are present. Linux
> ARM64 Clang 20/CUDA 12.8 device-only SM89 and host-object compilation reportedly
> succeeded, but there was no device link/run. Current host validation passed 25/25
> CTest groups in ten macOS/Linux configurations; the Linux verifier also passed
> install/config/scoped-source stability checks. See [current validation](implementation-2026-10-29.md).
> Current Windows retesting, native Linux GPU, VS2022-generator and hosted CI remain
> outstanding. Full gates are not closed.

Historical snapshot: **28 October 2026**. The 28 October implementation and evidence tables below remain a historical snapshot; their Windows-pending language and counts are not overwritten by the 29 October report. Later authorization includes M0/M1/M2 code work and easy-new-model scaffolding. No historical count is a current-tree rerun or M1 exit qualification.

## Source slices

| Slice | Implemented scope | Evidence and limits |
|---|---|---|
| M1.1 — host/product builds | Host Ninja Debug/Release presets and product-only Ninja paths; existing Visual Studio presets preserved. Linux presets and bounded fail-closed runner added; Ubuntu 24.04 x86_64 GCC/Clang Debug/Release and ASan/UBSan CI configured. | macOS `host-ninja-debug` and `host-ninja-release` each passed 21/21 groups with image decoding ON. Linux OrbStack ARM64 host matrix and product Release evidence: see below. Ubuntu 24.04 x86_64 CI is configured, not run here. Ninja 1.13.0 on macOS is not a qualified toolchain pin. |
| M1.2 — configuration, logging, dependencies | Strict TOML + optional one-model INI + CLI precedence; structured errors/logging and bounded dependencies. Includes optional PNG/JPEG-to-RGB decoder, default OFF, stb_image v2.30 commit `f7b2fcb360fc7c611e5b27f6aeb456e40b559210`, MIT terms. | Static checks do not load models or establish readiness. Decoder is standalone, not server/model/vision endpoint support; see [image-decoding.md](image-decoding.md). |
| M1.3 — platform RAII and ownership | File/mapping offsets, Unicode paths/arguments/environment, shutdown/thread lifecycle adapters; internal CUDA device/buffer/event/stream owners in `src/backends/cuda/resource_owners.h`. | Host and fake-runtime tests only. Windows helper/native behavior and physical CUDA ownership are not natively validated; failed synchronization is not a completion fence. |
| M1.4 — server shell lifecycle | Model-free health/unavailable-readiness behavior, startup failure and orderly shutdown; occupied/fixed-port restart, SIGINT/SIGTERM and private Windows Ctrl+C/Ctrl+Break coverage. | macOS host tests passed; Windows-specific control helper was not run natively. No model readiness or generation. |
| M1.5 — CI, packaging and evidence | Windows Debug/Release CI/devshell setup, installed-package/config checks, bounded evidence snapshots of eligible files under relevant roots (including dirty/untracked source), strict report validation. Windows Debug CI enables `-EnableImageDecoder`. | Workflow/configuration is not a CI run. Snapshot scope and exclusions are detailed in [windows-build.md](windows-build.md#host-ci); it covers listed roots/files and named build/cache exclusions, not arbitrary `.gitignore` rules. It is bounded change evidence, not complete or authenticated provenance. PowerShell 7.5.4 parsed wrapper syntax and macOS rejection gates only, not native Windows execution. |
| M1.6 — SM89 CUDA portability probe | Original-4080 portability probe; report requires `__CUDA_ARCH__` 890 and binary architecture 89, with 19 named checks. | **Historical 28 October state:** source-only; not compiled/device-linked/run. **Reported 29 October Windows evidence:** physical original-4080 smoke passed 19/19 checks (see [Windows evidence](windows-4080-evidence.md)); it was not rerun against the current tree and the raw bundle is unavailable locally. Linux ARM64 Clang 20/CUDA 12.8 work reached device-only SM89 and host-object compilation only—not device link/run. Current-tree CUDA/device validation and native Linux physical-GPU evidence remain pending. Host fake-runtime tests do not validate this probe. |

## Validation actually run

**Linux host, Ubuntu 25.04 aarch64 in OrbStack**, kernel 7.2.12, GCC 14.2.0,
Clang 20.1.2, CMake 3.31.6 and Ninja 1.12.1 (2026-10-28):

| Run | Options | Result and retained evidence |
|---|---|---|
| `linux-gcc-debug` | image decoding ON | Full runner stages passed; unchanged scoped sources; 21/21 CTest groups. `build/linux-evidence/20261028T231828Z-191cf8714484/evidence.json` |
| `linux-gcc-release` | image decoding OFF | Full runner stages passed; unchanged scoped sources; 20/20. `build/linux-evidence/20261028T231828Z-92f9b2664606/evidence.json` |
| `linux-clang-debug` | image decoding OFF | Full runner stages passed; unchanged scoped sources; 20/20. `build/linux-evidence/20261028T231828Z-2c416f33f4aa/evidence.json` |
| `linux-clang-release` | image decoding OFF | Full runner stages passed; unchanged scoped sources; 20/20. `build/linux-evidence/20261028T231828Z-db299d6681d5/evidence.json` |
| `linux-clang-sanitized` | ASan/UBSan, image decoding OFF | Full runner stages passed; unchanged scoped sources; 20/20. `build/linux-evidence/20261028T231909Z-e2b416ed613f/evidence.json` |
| `linux-clang-thread-sanitized` via the full bounded runner | image decoding OFF | Full runner stages passed; unchanged scoped sources; 20/20. `build/linux-evidence/20261028T231909Z-c56d902e86f2/evidence.json` |

All six fresh full-runner checks passed, including dependency hashes and unchanged
scoped-source snapshots. The 22 tooling tests include checks that `safe_build_root`
rejects all in-source output locations except the permitted `build/` subtree. This
is bounded change evidence, not complete/authenticated provenance.
The Linux product Release build (tests/CUDA/image decoding OFF) passed build/install
to `build/linux-product-install` and both installed static config checks; no Python
discovery, CTest files/tests, or decoder archive were present. The `tools/linux/Dockerfile`
was not built/tested. Initial Clang configure failures due to automatic C++20 module
scanning without `clang-scan-deps` are retained at `20261028T230935Z-{971be7d83dfe,
e24d09d6c672,f5f94a421915}` and `20261028T231018Z-6ff9eff4d17a`; after setting
`CMAKE_CXX_SCAN_FOR_MODULES OFF` (disabled scanning before FindThreads because Saga uses no C++ modules),
all reruns passed. This change was implemented and independently reviewed.

Tooling unit tests passed 22/22 on both Mac and VM. The Mac host-debug final rebuild
passed 21/21 with image decoding ON. Earlier broader Mac M1 results remain historical
as labeled; other Mac configurations were not rerun for this evidence increment.

On **macOS arm64, AppleClang 21, CUDA OFF**, each of `host-debug`, `host-release`,
`host-sanitized`, and `host-thread-sanitized` was configured with
`-DSAGA_ENABLE_IMAGE_DECODER=ON`, built and tested: **21/21 CTest groups passed in
each configuration** (prior evidence; not rerun in this increment). This is newer than
the historical 18-group suite and earlier labels in `docs/m0-evidence.md`.

The following focused TSan groups passed ten repetitions each:

```sh
ctest --test-dir build/host-thread-sanitized -R '^(cuda_resource_owners|image_decoder|platform_process|api_shell)$' --repeat until-fail:10
```

A fresh final product-only Ninja build with tests OFF, CUDA OFF and image decoding
OFF built and installed to `build/m1-final-product-install`; both installed TOML-only
and TOML+INI static checks passed. Verified: no Python discovery, no CTest
registration, no decoder archive. A separate image-enabled host Release install to
`build/m1 enabled install` passed both static checks; the decoder archive, public
header and stb license were present. Six Python tooling tests passed normally and
with `-O`; that earlier six-test result is superseded for the current tooling suite
by 22/22 on both Mac and VM. This does not reassert or replace the historical 57-test
M0 suite.

These are host/product checks only. No hosted CI run is evidenced. No native
Windows/MSVC build or install, CUDA device compile/device-link/run, model/reference
execution, readiness, performance result, or full release qualification was
performed. Linux host evidence is ARM64 under OrbStack; it does not establish Linux
x86_64 or native Windows support. M1's full exit remains **blocked/unqualified**;
no toolchain pin is qualified.

## Native next steps (remaining work)

From a **PowerShell 7.4+ x64 Visual Studio developer shell** on native Windows, run
both host configurations with the optional decoder enabled, then the CUDA smoke on
the physical original desktop RTX 4080 16 GB (Device 0):

```powershell
pwsh -NoProfile -File .\tools\windows\verify.ps1 -Configuration Debug -EnableImageDecoder
pwsh -NoProfile -File .\tools\windows\verify.ps1 -Configuration Release -EnableImageDecoder
pwsh -NoProfile -File .\tools\windows\cuda-smoke.ps1 -Device 0
```

Retain a fresh report/artifacts and inspect compiler, toolkit, driver, device identity and results. The Windows report already recorded above is historical and is not a rerun of the current tree. Do not substitute CI setup, host tests (including Linux OrbStack), a fake runtime, a different 4080 SKU, or a source-level architecture check for native execution. Linux host development/testing is authorized; Linux
native original-4080 CUDA evidence remains outstanding. M0 external
artifact/reference/baseline gates remain separately blocked as documented in
[m0-evidence.md](m0-evidence.md). See [linux-build.md](linux-build.md) for the Linux
host workflow and evidence.
