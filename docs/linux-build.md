# Linux host build and test (development only)

Linux host development/testing is supported by the existing Linux CMake presets
and `tools/linux/verify.py`. **Windows remains the production build and
qualification target.** Linux VM results are host build/test evidence only; they
do not establish Windows behavior, CUDA support, model compatibility, inference
correctness, performance, or production readiness. The OrbStack VM has no GPU.
Do not run CUDA, download models, or use these results for GPU/performance
qualification. No M2+ or other Linux production claim is made here.

## OrbStack VM setup on macOS

This procedure assumes the repository is available at the same project path in
the existing OrbStack Linux VM and that `orb` can run commands in that VM. Inspect
the guest before installing packages:

```sh
orb uname -s
orb uname -m
orb cat /etc/os-release
```

The verified VM on 2026-10-28 was Ubuntu 25.04, aarch64, kernel 7.2.12, with
g++ 14.2.0, Clang 20.1.2, CMake 3.31.6, and Ninja 1.12.1. Requirements are
CMake >=3.28, Python >=3.9, Ninja, and a C++20-capable GCC or Clang toolchain.
This is a recorded host, not a pinned or generally qualified toolchain. Install prerequisites only
if needed; package installation requires network access inside the VM and root
privileges. The verified setup used `orb -u root apt-get` to install CMake, Ninja,
Clang, `libclang-rt-dev`, and Git; g++ and Python were already present. A fresh
Ubuntu host should also install the standard compiler/build essentials:

```sh
orb -u root apt-get update
orb -u root apt-get install -y build-essential clang libclang-rt-dev cmake ninja-build python3 git
```

The repository's optional `tools/linux/Dockerfile` is an Ubuntu 24.04 toolchain
recipe, not an immutable image pin and not a container build verified by this
evidence.

## Evidence-producing host runs

From a macOS terminal with the current working directory set to the project root,
run the verifier through `orb` as shown below. The same commands can be run
inside the Linux guest with the leading `orb` omitted. Its default source root is
the repository containing the script; it accepts `--source-root`, `--build-root`,
and `--timeout-seconds` (default 1200) overrides. `--parallel` is required and
must be positive. The optional image decoder flag enables that bounded adapter
only; it does not provide inference image support.

```sh
orb python3 -B tools/linux/verify.py --preset linux-gcc-debug --parallel 2 --enable-image-decoder
```

Available verifier presets are `linux-gcc-debug`, `linux-gcc-release`,
`linux-clang-debug`, `linux-clang-release`, `linux-clang-sanitized`, and
`linux-clang-thread-sanitized`. Example
invocations (omit `--enable-image-decoder` for image-decoder-off runs):

```sh
orb python3 -B tools/linux/verify.py --preset linux-gcc-release --parallel 2
orb python3 -B tools/linux/verify.py --preset linux-clang-debug --parallel 2
orb python3 -B tools/linux/verify.py --preset linux-clang-release --parallel 2
orb python3 -B tools/linux/verify.py --preset linux-clang-sanitized --parallel 2
orb python3 -B tools/linux/verify.py --preset linux-clang-thread-sanitized --parallel 2
```

The verifier configures, builds, runs CTest, installs, runs static configuration
checks against both installed TOML and installed TOML+INI, verifies dependency
hashes, and compares scoped source snapshots before/after. Each evidence directory
contains `evidence.json`. Snapshots are bounded change-detection aids, not
complete/authenticated provenance; see the verifier output and
[Windows evidence description](windows-build.md#host-ci) for the shared snapshot
scope and limitations. The runner does not download tools or dependencies. Its
safety check permits an in-source build root only under the source tree's `build/`
subtree; an explicit build root outside the source tree is also permitted. Roots
such as `docs/` or `.git/` are rejected before run/evidence creation. A rejected
initial platform-gate attempt may therefore leave no evidence directory. The
safety tests for this restriction were independently reviewed.

Run TSan separately from ASan/UBSan. The TSan verifier invocation is a manual
local development run, not part of the configured Linux CI matrix. Its successful
run retained verifier evidence as listed below. The runner's
initial Clang attempts failed because automatic C++20 module scanning needed a
missing `clang-scan-deps`. The CMake module-scan setting was then disabled before
`FindThreads`, independently reviewed, and the reruns below passed. Preserve this
failure history; it is not evidence of a C++ portability defect.

## Recorded local results (2026-10-28)

The following runs used the Ubuntu 25.04 aarch64 OrbStack VM described above.
All six fresh runs passed every runner stage, including build, tests, install,
both installed static checks, dependency hashes, and equal scoped before/after
source hashes. The final source was frozen and reviewed before these runs:

| Preset | Image decoder | CTest groups | Evidence directory |
|---|---:|---:|---|
| `linux-gcc-debug` | On | 21/21 | `build/linux-evidence/20261028T231828Z-191cf8714484` |
| `linux-gcc-release` | Off | 20/20 | `build/linux-evidence/20261028T231828Z-92f9b2664606` |
| `linux-clang-debug` | Off | 20/20 | `build/linux-evidence/20261028T231828Z-2c416f33f4aa` |
| `linux-clang-release` | Off | 20/20 | `build/linux-evidence/20261028T231828Z-db299d6681d5` |
| `linux-clang-sanitized` (ASan+UBSan) | Off | 20/20 | `build/linux-evidence/20261028T231909Z-e2b416ed613f` |
| `linux-clang-thread-sanitized` (manual, TSan) | Off | 20/20 | `build/linux-evidence/20261028T231909Z-c56d902e86f2` |

The product-only `linux-product-release` preset also configured, built, and
installed with tests, CUDA, and image decoder disabled. Both installed static
configuration checks passed. As intended for this preset, it did not discover
Python or create CTest files/tests or a decoder archive. To reproduce the
configure/build/install steps with an explicit project-root-relative prefix:

```sh
orb cmake --preset linux-product-release -DCMAKE_INSTALL_PREFIX=build/linux-product-install
orb cmake --build --preset linux-product-release --parallel 2
orb cmake --install build/linux-product-release --prefix build/linux-product-install
```

Then run both static checks against the installed TOML, and the installed TOML
plus INI, using the installed executable:

```sh
orb build/linux-product-install/bin/saga-server --config build/linux-product-install/share/saga/server.example.toml --check-config
orb build/linux-product-install/bin/saga-server --config build/linux-product-install/share/saga/server.example.toml --model-config build/linux-product-install/share/saga/model.example.ini --check-config
```

After the final source freeze and review, macOS `host-debug` passed 21/21 groups.
The 22 tooling unit tests passed on both macOS and the VM.

These are specific local host runs, not CI execution or a supported-distribution
matrix. Linux CI is configured in `.github/workflows/host.yml` for Ubuntu 24.04
x86_64 GCC/Clang Debug/Release and ASan+UBSan; image decoder is enabled only for
GCC Debug. It has not been executed as part of this evidence. TSan is manual and
is not in that CI matrix. Windows remains the production target; no Linux GPU is
available in the VM, and no CUDA, models, or performance qualification was done.

## Other Linux presets and limits

`CMakePresets.json` also defines general Unix `host-*` presets, the Linux product
preset `linux-product-release`, and `cuda-linux`. Host presets build host code;
`linux-product-release` disables tests, CUDA, and image decoding. `cuda-linux` is
not part of this procedure and does not imply that CUDA can be built or run on
this VM. Use the native Windows procedures in [windows-build.md](windows-build.md)
for the production target and CUDA portability gate.
