# Original RTX 4080 CUDA portability probe (M1.6)

## Evidence status

**Native Windows/MSVC compile, device-link and original-4080 run passed on
29 October 2026:** all 19 required checks, CUDA 13.3.73 and MSVC 19.51.36248.0.
Exact environment, logs and raw report are recorded in
[Windows / RTX 4080 evidence](windows-4080-evidence.md). The original macOS
source-only implementation history remains historical. Native Linux original-4080
evidence is still missing, so M1.6's **both-OS** exit remains blocked. Host-only
builds, source review, mocks, cross-compilation and WSL cannot replace that run.

`tests/cuda/portability_smoke.cu` is a standalone CUDA Runtime executable. It does
not load a model, import inference kernels, require external math/JSON libraries,
benchmark throughput, or qualify serving readiness.

Host mock tests for the resource-owner layer remain separate from the physical
Windows probe; they do not exercise this `.cu` translation unit or a CUDA device.
The passing probe qualifies only its named smoke paths, not inference, model fit,
performance or serving readiness.

## What actually runs

The default device is CUDA ordinal 0; `--device N` selects another nonnegative
ordinal. There is deliberately **no other-device override**. Qualification requires:

- Exact runtime name `NVIDIA GeForce RTX 4080` or `GeForce RTX 4080`, compute
  capability 8.9, 32-lane warps, and **76 SMs**. RTX 4080 Super, laptop variants,
  other GPUs, and incompatible kernel images fail rather than skip. This checks
  driver-reported identity/properties, not tamper-resistant hardware attestation.
- A real `cp.async.ca.shared.global` test: one block of 32 threads, each copying a
  different aligned 16-byte `uint4` from global memory into an explicitly
  16-byte-aligned shared array. Inline PTX commits and waits for each lane's async
  group; a block barrier precedes the shared-to-global result read. All 128
  32-bit words are compared exactly on the host. PTX is guarded by
  `__CUDA_ARCH__ >= 800`; there is **no synchronous fallback**.
- A real BF16 Tensor Core WMMA test: exactly one 32-lane warp, `m16n16k16`,
  row-major BF16 inputs and FP32 accumulation/output. A is the identity;
  `B[row,column] = (5*row + 3*column) % 17 - 8`. The accumulator starts at 0.5,
  so every expected output is exactly `B[row,column] + 0.5`. All inputs and
  expected outputs are exactly representable; all 256 outputs require exact
  equality, with no tolerance or self-reference to a GPU result. The varying,
  nonsymmetric B catches layout/transposition errors. Device allocations satisfy
  WMMA's 32-byte base alignment; leading dimension 16 gives a 32-byte BF16 stride
  and a 64-byte FP32 stride. Every lane executes the same WMMA operations.
- Both kernels report their compiled `__CUDA_ARCH__`; each must return **890**.
  `cudaFuncGetAttributes` must report `binaryVersion == 89` for each loaded
  kernel. PTX version is recorded separately, not treated as physical GPU identity.
- All allocations, host/device copies, output initialization, launches,
  `cudaGetLastError`, event synchronization, attempted stream draining, and
  device frees are checked. Every copy, memset, and kernel launch uses one
  explicit nonblocking stream; a successful completion-event synchronization
  establishes the host-read boundary. This includes both BF16 H2D inputs,
  avoiding default-stream ordering assumptions. Noncopyable RAII buffers free
  successful partial allocations on errors. Destructors retain the first cleanup
  error without throwing; cleanup failure also causes a nonzero exit. Results
  are read back only after successful event synchronization.
- Lifetime order is deliberate: the report outlives `run_probe`; the optional
  device guard is declared before probe resources and emplaced within the named
  device-selection check. Host arrays and device buffers precede the event, and
  the event precedes the stream, so reverse destruction attempts to synchronize
  and destroy the stream before destroying the event, buffers, or host arrays;
  device restoration is last. A successful stream synchronization drains
  submitted work before those dependents are destroyed. If synchronization
  fails, the failure is retained and reported, but it is not a successful
  physical-work fence and does not guarantee dependent storage is safe to
  destroy. This standalone probe is fail-stop on such errors and is not a
  serving-safe fence or recovery mechanism. `resource_cleanup` records
  restoration/resource errors before the final compatibility check
  `allocation_cleanup`, which covers all resource cleanup, not allocations alone.

Checks fail fast; later checks remain `not_run`, never assumed to have passed.
The `resource_cleanup` and compatibility `allocation_cleanup` checks still run
on failure (and succeed vacuously if no resources were created). A rejected
device therefore does not execute the kernels. Cleanup checks report retained
errors but cannot establish that asynchronous work has stopped when stream
synchronization fails; the process exits nonzero and must not be treated as
recoverable or safe for continued serving. The report declares 19 named checks,
each with a corresponding execution step.

## Resource queries and their limits

For **each actual smoke kernel**, using the actual 32-thread block and zero
*dynamic* shared bytes, the probe checks `cudaFuncGetAttributes` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`. JSON records registers/thread,
static shared bytes, local bytes/thread, maximum block threads, native binary/PTX
versions, active blocks/SM, and resident warps/SM. The kernels must support one
warp and at least one resident block. Static shared usage is already included by
the occupancy API; it is not passed a second time as dynamic shared memory.

Both `cudaDeviceProp::cooperativeLaunch` and
`cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch)` are recorded and required
to agree. **Unsupported cooperative launch is not a failure:** no cooperative
algorithm is used, and no cooperative kernel is launched. When support is true,
`active_blocks_per_sm * observed_sm_count` is reported as the occupancy-derived
cooperative grid block bound for that particular smoke kernel/configuration;
otherwise its cooperative bound is `null`. `cooperative_launch_exercised` is
always `false`. These queries do not test grid synchronization or a cooperative
launch, and do not establish that a future cooperative algorithm is portable.

**These are tiny instruction/correctness smoke kernels, not representative
inference workloads.** Their occupancy bounds, register counts, shared-memory
usage, and correctness results say nothing about eventual attention/GEMM/codec
kernel resource bounds, model fit, latency, bandwidth, throughput, or quality.
Those kernels/configurations must be queried and measured independently.

## Optional build integration

The root `CMakeLists.txt` now integrates `SAGA_ENABLE_CUDA` and the optional
`saga_cuda_portability_smoke` target. The default macOS/host-only build remains
free of CUDA discovery and language initialization. The target configuration
is equivalent to the following, with additional Linux/native-Windows checks
and an initial CUDA compiler floor of 12.8:

```cmake
option(SAGA_ENABLE_CUDA "Build the original RTX 4080 portability probe" OFF)
if(SAGA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(saga_cuda_portability_smoke
        tests/cuda/portability_smoke.cu)
    set_target_properties(saga_cuda_portability_smoke PROPERTIES
        CUDA_ARCHITECTURES 89
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared)
    target_link_libraries(saga_cuda_portability_smoke PRIVATE CUDA::cudart)
endif()
```

Use CUDA >=12.8 and a supported host compiler with SM89/CUDA C++20 support;
verify the exact toolkit/compiler support matrix. Saga requires CMake >=3.28.
These are build requirements, not yet qualified production toolchain pins. No cuBLAS, NVML, CUDA Driver API
link library, POSIX APIs, or third-party dependencies are needed. The CUDA
Runtime shared library must be discoverable when running (including CUDA's DLL
on Windows). Retain the verbose build log to prove separate device compilation,
device link, host compiler, flags, and architecture; runtime JSON alone does
not prove which build/link commands were executed. Do not silently fall back to
host-only when `SAGA_ENABLE_CUDA=ON` but CUDA is unavailable.

Example Linux commands from the repository root (the `cuda-linux` Ninja preset
provides equivalent settings):

```sh
cmake -S . -B build-cuda -DSAGA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target saga_cuda_portability_smoke --verbose
./build-cuda/saga_cuda_portability_smoke --device 0 > cuda-portability-linux.json
```

For native Windows, use an x64 MSVC developer prompt with the chosen CUDA toolkit
on PATH and, for example, the Ninja generator:

```powershell
cmake -S . -B build-cuda-win -G Ninja -DSAGA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-win --target saga_cuda_portability_smoke --verbose
.\build-cuda-win\saga_cuda_portability_smoke.exe --device 0 > cuda-portability-windows.json
$LASTEXITCODE
```

These are procedures, **not commands run or passes claimed by this change**.
For another generator, executable paths/configuration selection may differ.
PowerShell redirection encoding varies by version; retain valid UTF-8 JSON when
moving the report into an evidence archive. Check the process exit code before
running another command (`$LASTEXITCODE` on PowerShell; the shell's process
status on Linux). Do not treat the existence of a redirected file as success.

## JSON and acceptance evidence

Normal success and handled failures emit one JSON object on stdout; success
returns 0, and any failed check/CUDA operation/cleanup or stdout write failure
returns nonzero. Driver strings and error text are JSON escaped. Non-ASCII bytes
are escaped individually as `\u00XX`, preserving JSON validity without assuming
the encoding of driver messages. Fatal process/driver termination cannot be
promised to produce a complete report.

The schema records:

- NVCC major/minor/build, CUDA header version, host compiler/version, OS family,
  and `__cplusplus` (MSVC may report its legacy macro value without `/Zc:__cplusplus`;
  the retained C++20 compile flags are authoritative).
- Driver/runtime API versions in CUDA's integer format (for example, 12040 means
  12.4), selected ordinal/device count, device name, compute capability, actual
  and expected SM count, warp size, and `totalGlobalMem`.
- `cudaMemGetInfo` free/total bytes after context initialization but before probe
  allocations. Unqueried memory/device fields are `null`; version/compiled-arch
  zero and count/attribute -1 denote not obtained, not valid measurements.
  Display/OS/runtime reservations can reduce visible memory; no exact 16-GiB
  equality or inference-capacity assertion is made.
- Per-kernel compile architecture and resources, cooperative support/query,
  whether cooperative launch was exercised, and individual named checks with
  `pass`, `fail`, or `not_run` and failure details. Overall `passed` must be true
  **and** the exit code zero; neither alone replaces inspection of retained logs.

For M1.6, retain separate **native Linux and native Windows/MSVC** evidence on a
physical original RTX 4080: source revision/hash, CMake configuration/cache,
`cmake --version`, `nvcc --version`, host compiler version, full verbose
compile/device-link logs and process exit codes, and unedited runtime JSON.
Record OS version, GPU identity/UUID, host CPU/RAM, driver/toolkit installation,
display/background GPU load, power policy, and memory conditions externally
(e.g. `nvidia-smi` plus platform tools); the probe is not a full environment
inventory. A passing run demonstrates only these instruction, launch/result,
and resource-query paths. Future inference/cooperative paths and broader M1
server acceptance remain separate gates.
