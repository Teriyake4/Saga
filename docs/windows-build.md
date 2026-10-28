# Native Windows build and evidence

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
configuration. It pushes the source root before invoking CMake so preset
lookup works from any caller directory, and restores the original directory on
exit. It records command lines, native process exit codes, OS/PowerShell/tool
locations, selected Visual Studio/SDK facts and selected CMake cache fields
without dumping PATH or the full environment. Transcript output and native
command output are kept in separate files. Inspect the CTest log and install
contents; a successful configure alone is not a build/test pass. The installed
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

Linux is deliberately not part of required hosted jobs. On a configured Linux
host with CMake >=3.28 and Python >=3.9, the retained Unix presets can be run as
`cmake --preset host-debug`, `cmake --build --preset host-debug`, and
`ctest --preset host-debug`; analogous presets include `host-release`,
`host-sanitized`, and `host-thread-sanitized`. Sanitizers are compiler/runtime
specific and run as separate configurations.

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

The optional script pushes the source root before preset use and restores the
original working directory. It records NVCC/CMake/tool locations and CUDA
compiler/cache, keeps native command output separate from its transcript,
retains the verbose build/device-link output and raw runtime JSON, checks native
exit codes, and fails if any stage fails. Device/runtime time limits remain
unchanged; bound execution externally if required by the hardware lab. Runtime
stdout JSON and stderr are retained separately; the script requires both exit
code zero and JSON `passed=true`. Inspect individual checks and the build log; the script does not turn missing CUDA or another GPU into a pass.
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
compile. This environment is macOS and PowerShell/MSVC/NVCC/native Windows are
unavailable; none of the Windows scripts or builds have been run here.
