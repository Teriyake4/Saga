# Windows / RTX 4080 evidence

Status: **bounded native Windows host and CUDA smoke evidence**, 2026-10-29.
This records Windows/RTX 4080 testing and fixes only. It does not implement or
close general M0 requirements. The historical M0 status, prior evidence and
limitations remain in [m0-evidence.md](m0-evidence.md); prior Windows procedure
and earlier failure history remain in [windows-build.md](windows-build.md) and
its linked records. Earlier CUDA evidence (19 checks) is retained; the final
run below is the current run.

## Host and toolchain

Commands were run natively on Windows via unsandboxed Git Bash, launching
portable PowerShell 7.5.4 with the Visual Studio developer environment.
PowerShell was downloaded to `build/native-tools`; publisher SHA-256:
`b40d192ae95ba6ccc4cc362ff4e1b18ca6fb5055bebbcd3920684e12701fa8f6`.

- Windows 11 Pro x64, version `10.0.26200`; Ryzen 7 7800X3D; OS-reported RAM
  `33444892672` bytes.
- Original NVIDIA GeForce RTX 4080, UUID
  `GPU-a656d0c9-68da-006c-4d83-fd29755126d3`; 76 SM, warp 32, CC 8.9;
  CUDA total memory `17170956288` bytes.
- Driver `610.88`; CUDA `13.3.73`; MSVC `19.51.36248.0`, toolset
  `14.51.36231` from VS Build Tools 2026 `18.8.0`; Windows SDK `10.0.26100.0`.
- CMake `4.4.0`; Python `3.13.5`; PowerShell `7.5.4` portable; Visual
  Studio-bundled Ninja `1.13.2`. These are observed versions, not a broadly
  qualified toolchain matrix.

Only Visual Studio 2026 was installed. Existing Windows Ninja presets were used;
Visual Studio 2022-generator presets and hosted CI were not run.

## Reproduction and results

From a native x64 Visual Studio developer environment with PowerShell 7.4+:

```powershell
.\tools\windows\verify.ps1 -Configuration Debug -ConfigurePreset windows-ninja-debug -EnableImageDecoder
.\tools\windows\verify.ps1 -Configuration Release -ConfigurePreset windows-ninja-release -EnableImageDecoder
.\tools\windows\cuda-smoke.ps1 -Device 0
```

The tested working tree is based on revision
`c29ba38d1c97790779fe0b441e727a667b65edb4`, with the uncommitted Windows fixes
listed below. No commit or branch was created.

Both fresh host wrapper runs passed **26/26 CTest groups**:

- Debug: `build/windows-evidence/Debug-20261029-002418-c2224c11`
- Release: `build/windows-evidence/Release-20261029-002632-687c75ea`

Each retains `verify.log`, transcript, CMake cache, compiler facts and scoped
source snapshots, plus installed files under paths containing spaces. The final
Debug, Release and CUDA scoped source-file inventories agree. These bounded
snapshots exclude `bench/` and `.gitattributes`; they are not complete source-tree
or authenticated provenance records.

The CUDA smoke passed all **19 required checks**, exited 0, and reported compile
architecture 890 and binary architecture 89. `build/cuda-windows-evidence/20261029-002632-8492e902`
contains build/device-link logs, raw runtime JSON and stderr, and matching
source snapshots. Final report: `cplusplus=202002`. CUDA executable SHA-256:
`51345730264b855fabac79985ce244508c34cc1c89f5dabfeaa1a794531eaae6`.

The `windows-product-ninja-release` product preset (tests, CUDA and image decoding disabled)
built and installed successfully, including path-with-spaces and TOML/INI
self-checks. Its transcript is `build/native-tools/product-release.log`; no
dedicated wrapper snapshot is claimed for this run.

## Windows fixes exercised

- Made optional device-profile literals unsigned and added the direct mutex
  include in model contracts; renamed a startup-test member that shadowed a name.
- Corrected CUDA's `_MSVC_LANG` report. Used explicit CRLF bytes in the API test
  to avoid Windows text-mode CRCRLF. Segregated symlink-privilege and FIFO tests
  on Windows; directory rejection still runs.
- Marked byte-pinned headers, patches and licenses `-text` in `.gitattributes`.
  Windows `autocrlf` filtering was validated against all 17 pinned inputs,
  preserving the two genuine-CRLF files.
- Hardened verifier cache guards against mislabeled single-config/test-disabled
  builds, with six AST regression tests. Real negative wrapper runs also rejected
  Debug with `windows-ninja-release` and the tests-disabled product preset before
  build/CTest, both with expected exit 1. Their logs are retained under
  `build/windows-evidence/Debug-20261029-003315-ab42e01f` and
  `build/windows-evidence/Release-20261029-003315-a3bd19c3` respectively.

No validators or hash manifests were loosened. Windows symlink creation lacked
privilege (`WinError 1314`) and FIFO is unsupported; those cases are explicit
skips, not passes. The standalone audit/oracle suite ran 59 tests with three
explicit skips (two symlink-privilege checks and FIFO); directory rejection still
ran. Earlier failures and historical evidence remain preserved.

## Limits

This is not a full M0/M1 exit, hosted-CI result, Linux qualification, model or
reference run, benchmark, larger M0 implementation, GPU performance result, or
production-readiness claim. No model/reference/benchmark work was done. Power
and memory telemetry is only a current observation (display-disabled query:
11.3 W); there was no controlled GPU load, power-policy control, or thermal
benchmark. The smoke test establishes only its named build/runtime checks on
this host. See [CUDA portability](cuda-portability.md) and
[Windows build and evidence](windows-build.md) for procedure and scope.
