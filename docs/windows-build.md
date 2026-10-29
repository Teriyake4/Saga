# Native Windows build and evidence

Native Windows Ninja/MSVC Debug/Release, product packaging and the physical RTX
4080 CUDA smoke passed on 29 October 2026. See
[Windows / RTX 4080 evidence](windows-4080-evidence.md) for exact versions, retained
logs and limitations; this is not a hosted-CI or both-OS qualification claim.

## Host builds

The retained `windows-msvc-debug` and `windows-msvc-release` presets use the
Visual Studio 17 2022 generator and dynamic CRT. Windows Ninja alternatives are
`windows-ninja-debug` and `windows-ninja-release`; both require CMake >=3.28,
Ninja installed and on `PATH`, a compatible MSVC C++ toolset, the Windows
SDK, and an x64 Visual Studio developer PowerShell prompt that exposes `cl.exe`
and the MSVC development environment. The presets explicitly select `cl.exe`;
Ninja does not itself provide a compiler. These paths are Windows-only and do not
qualify a build unless actually run on native Windows.

```powershell
cmake --preset windows-ninja-debug
cmake --build --preset windows-ninja-debug --parallel 4
ctest --preset windows-ninja-debug

cmake --preset windows-ninja-release
cmake --build --preset windows-ninja-release --parallel 4
ctest --preset windows-ninja-release
```

`windows-product-ninja-release` is the named product-only Release path. It
explicitly disables tests and CUDA, and therefore has no Python discovery or
CTest targets:

```powershell
cmake --preset windows-product-ninja-release
cmake --build --preset windows-product-ninja-release --parallel 4
```

The `windows-msvc-*` Visual Studio presets remain the standard host CI/evidence
path; Ninja presets do not replace them.

## Host CI

The required host matrix runs native x64 MSVC on `windows-2022` in separate Debug
and Release configurations. The generator is Visual Studio 17 2022; the presets
set `CMAKE_MSVC_RUNTIME_LIBRARY` to the dynamic CRT (`/MD` and `/MDd`), matching
the existing MSVC DLL-oriented default. No toolset or SDK version is declared a
qualified pin by these files. CI runs the repository CTest suite, installs to a
path containing spaces, checks the installed executable's config-validation
mode, and uploads timestamped command logs/cache evidence. Linux is not a required
job; Unix host presets and macOS ASan/UBSan and TSan regression jobs remain.

From a native x64 Visual Studio developer PowerShell prompt:

```powershell
.\tools\windows\verify.ps1 -Configuration Debug
.\tools\windows\verify.ps1 -Configuration Release -ConfigurePreset windows-msvc-release
```

The script accepts `-SourceRoot`, `-BuildRoot`, `-Configuration`, and
`-ConfigurePreset`. By default the preset follows the chosen Debug/Release
configuration. An explicit Ninja preset can use the installed compatible MSVC
version without requiring the VS2022 generator (the recorded native runs used
VS Build Tools 2026). After configure, the script rejects disabled/missing host
tests and mismatched single-config build types; multi-config generators must
contain the requested configuration. Product-only presets require their separate
build/install procedure, not this host verifier. It pushes the source root before invoking CMake so preset
lookup works from any caller directory, and restores the original directory on
exit. It records command lines, native process exit codes, OS/PowerShell/tool
locations, selected Visual Studio/SDK facts and selected CMake cache fields
without dumping PATH or the full environment. Transcript output and native
command output are kept in separate files. Inspect the CTest log and install
contents; a successful configure alone is not a build/test pass. Evidence directories use a timestamp plus random suffix and retain the full CMake cache, compiler/tool version output, and before/after source snapshots. Snapshots hash SHA-256 and byte size for files in `include/`, `src/`, `apps/`, `tests/`, `cmake/`, `tools/`, `third_party/`, plus root `CMakeLists.txt`, `CMakePresets.json`, and the server/model example configs; named build/cache directories are omitted. Arbitrary `.gitignore` rules are not consulted, so other ignored files within the listed roots can be included if eligible. Symlinks in scope fail snapshotting rather than being followed. Files over 32 MiB, credential-like filenames (`.env*`, `credential*`, `secret*`, `id_rsa`, `id_ed25519`), and common weight/key suffixes are excluded. Snapshots include uncommitted and untracked eligible files and record Git revision/dirty status when Git is available. The build fails if those scoped file hashes change during the run. This is a bounded change-detection aid, not an authenticated provenance statement; files outside the scope, excluded files, toolchain state, and malicious snapshot replacement are not covered. The installed
server self-check is invoked as
`saga-server.exe --config <installed share/saga/server.example.toml> --check-config`.
The single-model INI must also be installed: the script checks it using
`--model-config <installed share/saga/model.example.ini> --check-config` together
with the installed TOML file. Missing either example configuration fails the
packaging check rather than silently skipping it. CTest continues
to run the existing API shell tests, dependency hashes, source-manifest checks
and unit tests. No model weights, SDK, toolkit or package downloads are
performed.

## Optional Linux host procedure

Linux host development/testing is available, but Windows remains the production
build/qualification target. Linux CI is configured but is not required hosted
qualification, and CI execution has not been verified here. For Linux prerequisites,
presets, the evidence runner, and actual host results, see
[Linux host build and test](linux-build.md).

## Optional CUDA smoke on native Windows

CUDA is an independent hardware qualification, not part of host CI. Use a native
Windows x64 MSVC developer environment, a supported locally installed CUDA
Toolkit (the source currently enforces >=12.8), and an **original desktop RTX
4080**. The `cuda-windows` preset uses Ninja and architecture 89, but does not
pin or claim validation of a particular MSVC/CUDA combination. Check NVIDIA's
current host compiler/toolkit compatibility table for the installed versions.

```powershell
.\tools\windows\cuda-smoke.ps1 -Device 0
```

The optional script requires native Windows, PowerShell 7.4+, and an x64 MSVC developer environment. PowerShell 7.4 native stdout redirection preserves the runtime JSON bytes; older PowerShell is rejected. The script pushes the source root before preset use and restores the original working directory even if cleanup fails. It records NVCC/CMake/compiler facts and CUDA cache, keeps native command output separate from its transcript,
retains the verbose build/device-link output and raw runtime JSON, checks native
exit codes, and fails if any stage fails. Device/runtime time limits remain
unchanged; bound execution externally if required by the hardware lab. Runtime
stdout JSON and stderr are retained separately; the script requires exit code zero and validates the full schema, literal `passed=true`, Windows OS, requested device ordinal, original desktop RTX 4080, CC 8.9/76 SM/warp 32, compile architecture 890 and binary architecture 89, and each of the 19 named required checks exactly once with status `pass`. Missing, duplicate, unknown, failed, or not-run checks fail validation. Optional `nvidia-smi` collection is not synthesized: unavailable machine telemetry remains explicitly unavailable, and power/display measurements may be recorded manually. Inspect the individual checks and build log; the script does not turn missing CUDA or another GPU into a pass.
See [CUDA portability](cuda-portability.md) for exact hardware assertions and
the scope and limitations of the probe. No GPU evidence is claimed by this
procedure or by host CI.

## Provenance and review notes

The preset and workflow choices here are repository-local configuration, not
copied implementation. A cross-platform packaging reference was considered at
Strata revision `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`; no source code was
copied from it. The project has not established any Strata-derived license or
compatibility claim. No external reference content was needed to define these
scripts.

Parent integration should independently verify that all targets/tests and the
`--check-config` executable interface exist in the integrated CMake/application
sources. Static review found the MSVC flags in the current CMake configuration
(`/W4 /permissive- /utf-8 /Zc:__cplusplus /WX`) and Windows socket link
(`ws2_32`), while actual compilation remains unverified here. Review native
Windows portability in `src/platform/windows/`, `src/platform/shared/`, `src/api/server.cpp`,
and test sources before treating this tooling as evidence that all sources
compile. Historical note: the earlier review environment was macOS and did not have
PowerShell/MSVC/NVCC/native Windows; none of the Windows scripts or builds had
been run at that point. In the later M1 continuation, PowerShell 7.5.4 was installed
project-locally and parsed the Windows wrapper scripts plus their negative macOS
platform gates. This was syntax/gate-checking on macOS only: no native Windows
PowerShell, MSVC build, CI job, NVCC/device-link, or GPU runtime was exercised.
The bounded evidence snapshots described above include eligible dirty and untracked
files within listed roots, but exclude named build/cache directories, files over
32 MiB, credential-like filenames, and common weight/key suffixes; they are not a
complete or authenticated snapshot of the working tree.
