# M0 pinned NInfer-4080 build/run capture

This is a reproducibility aid for a later native Windows/RTX 4080 or Linux/CUDA
host. It does **not** claim that the reference has been built, run, or measured.
The archive is expected at
`.cache/m0-reference/ninfer-4080-2469b5752672ad597055af61fe28e8e7ced20056` and
must be checked against revision `2469b5752672ad597055af61fe28e8e7ced20056`
of `yalding/ninfer-4080`. The manifest in `bench/reference/manifest.json` remains
unchanged and still lists this reference as `not-built`.

## What source inspection established

The pinned CMake project requires CMake 3.28, C++20, CUDA compiler 12.8 or newer,
and exactly `CMAKE_CUDA_ARCHITECTURES=89`. Apps default ON; the build target names
are `ninfer` and `ninfer-serve` (plus `ninfer-perplexity`). The upstream run
examples are `scripts/run-ninfer-4080.sh` and `.bat`; they launch `ninfer-serve`
with artifact paths and numerous options. They are not a frozen test workload or
safe generic commands. Review `bench/reference/reference-build-plan.json` and
freeze the selected executable, artifact/configuration, prompts, arguments,
working directory, and runtime environment on the target host before executing.

No model or other large download is part of this flow. Configure vcpkg/dependency
availability and CUDA/driver/toolchain separately; record actual versions rather
than assuming the upstream requirements establish them. Do not run the upstream
artifact/model downloader as part of build capture.

## Harness input and invocation

`tools/m0_reference.py` takes a JSON recipe containing exact argv arrays; it does
not interpret shell syntax. Paths are resolved from the invoking directory, so
run it from the Saga repository root. Every command needs an explicit bounded
timeout. An example *template* (not yet an authorized/frozen workload) is:

```json
{
  "repository": "yalding/ninfer-4080",
  "revision": "2469b5752672ad597055af61fe28e8e7ced20056",
  "source_root": "<DISPOSABLE_PATCHED_SOURCE_COPY>",
  "mode": "build",
  "source_sha256": "<REVIEWED_64_HEX_SOURCE_TREE_HASH>",
  "source_review": {"reviewer": "<REVIEWER_ID>", "reviewed_at": "<DATE>", "basis": "<ARCHIVE_PROVENANCE>"},
  "trace_inventory_review": {"reviewer": "<REVIEWER_ID>", "reviewed_at": "<DATE>", "basis": "<INSTRUMENTATION_AND_SITE_INVENTORY_REVIEW>"},
  "build_dir": "<ABSOLUTE_BUILD_DIR_OUTSIDE_SOURCE>",
  "environment": {"NINFER_M0_Q3_DISPATCH_TRACE": "1"},
  "commands": [
    {"role": "configure", "argv": ["cmake", "-S", "<DISPOSABLE_PATCHED_SOURCE_COPY>", "-B", "<ABSOLUTE_BUILD_DIR_OUTSIDE_SOURCE>", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DNINFER_BUILD_APPS=ON", "-DBUILD_TESTING=OFF", "-DNINFER_BUILD_BENCHMARKS=OFF"], "timeout_seconds": 1800},
    {"role": "build", "argv": ["cmake", "--build", "<ABSOLUTE_BUILD_DIR_OUTSIDE_SOURCE>", "--config", "Release", "--target", "ninfer", "ninfer-serve", "--parallel", "<BOUNDED_JOB_COUNT>"], "timeout_seconds": 7200},
    {"role": "run", "argv": ["<ABSOLUTE_BUILT_BINARY>", "<FROZEN_RUN_ARGUMENTS>"], "cwd": "<RUN_DIRECTORY>", "timeout_seconds": 600}
  ],
  "patches": ["bench/reference/dispatch-audit.patch"],
  "binary": "<ABSOLUTE_BUILT_BINARY_INSIDE_BUILD_DIR>",
  "binary_sha256": null,
  "required_site_phases": [{"site": "<reviewed-site>", "phase": "<phase>"}]
}
```

The angle-bracket values are documentation placeholders, not values to pass to
the runner. `mode: "build"` requires a configure command using CMake `-S` for
the pinned source and `-B` for the declared external build directory, followed by
a CMake `--build` command and one run command. The declared binary must be inside
the external build directory and absent before configuration; its fresh post-build
path/hash/size snapshot is recorded and checked again before/after execution. A
build-mode `binary_sha256` may be null, or may assert an expected hash to compare
to the freshly observed result.

For a binary built elsewhere, use `mode: "reviewed-prebuilt"`, omit build commands,
and provide `prebuilt_provenance` with exactly `reviewer`, `reviewed_at`, `basis`,
`source_revision`, and matching `binary_sha256`. The output status is
`prebuilt-run-captured-incomplete-dispatch`, not a verified build. **Review fields
are operator assertions, not independently authenticated provenance; a matching
binary hash identifies bytes and does not prove how they were built.**

Both modes require reviewed source provenance, trace-inventory review, exact argv,
finite timeouts, and a reviewed nonempty array of `{"site": ..., "phase": ...}`
objects. The runner validates the entire recipe before starting any command. Source
trees containing symlinks, evidence output inside the source tree, and pre-existing
evidence directories are rejected. All run stdout/stderr is drained in bounded
chunks to a capped log; hitting the cap terminates the process tree and invalidates
the run. `capture.json` is written even when recipe validation, command execution,
or trace parsing fails (except when refusing an existing output directory). Each
command records exact argv, cwd, exit, timeout, duration, and log hash. The raw
run trace must be emitted in the captured run's stdout; an external trace path is
not accepted. Run `argv[0]` is checked against the declared executable and hash
immediately before launch. Overrides are stored verbatim: do not put secrets in
the recipe. The runner records inherited environment variable names but does not
sanitize ambient inherited values.

```sh
python3 -B tools/m0_reference.py --config <RECIPE.json> --output <EVIDENCE_DIR>
```

Exit zero means only that the selected build/run path and declared trace inventory passed harness checks; both output statuses explicitly remain incomplete for dispatch. A nonzero status means capture is invalid/incomplete. The runner never
rewrites source or the bench manifest. The JSON source-tree digest is a canonical SHA-256 over sorted relative paths
and file hashes. For the opt-in patch workflow, the `source_root` must be a
disposable copy of the pinned source with the patch applied, and `source_sha256`
must hash that exact patched tree. `patches` records the patch file hash; the
build plan separately retains hashes of the pristine upstream files and upstream
license. `source_review` is still an operator assertion. The downloaded pinned
cache is not modified.

## Opt-in Q3 dispatch producer and coverage limits

The optional `bench/reference/dispatch-audit.patch` instruments the actual Q3
linear dispatcher selection immediately before the chosen launch, including the
A8 named launch branches, and Q3 linear SwiGLU launch branches (including its
chunked projection path). It is intentionally narrow. The patch is source-only,
not applied to the cached source, and has not been compiled or run on hardware.
Its SHA-256 and the exact pristine source-file/license hashes inspected for patch
context are recorded in `bench/reference/reference-build-plan.json`.

Apply it only to a disposable working copy of the exact pinned source and retain
the upstream `LICENSE` and attribution:

```sh
patch --dry-run --batch -p1 -i /path/to/Saga/bench/reference/dispatch-audit.patch
patch --batch -p1 -i /path/to/Saga/bench/reference/dispatch-audit.patch
```

The producer is disabled unless the process environment contains
`NINFER_M0_Q3_DISPATCH_TRACE=1`. It emits `M0DISPATCH` JSONL on stderr at the
selected branch with `site`, `route`, kernel launch function, `A16`/`A8`, and
`m/n/k`. The value `phase: "unknown"` is deliberate: these dispatcher call sites
have no trustworthy prefill/decode/verification phase identifier. The harness
requires exact site/phase inventory matches, so this producer's unknown phase
cannot satisfy a required `prefill`, `decode`, or `verification` record.

`launch-selection` means the C++ dispatch selected that launcher immediately
before invoking it; it does not prove successful kernel completion or device
execution. The helper calls `cudaStreamIsCapturing` only when tracing is enabled.
An active capture is marked `graph-record-selection`, distinct from an ordinary
`launch-selection`; there is no graph-replay execution event. Capture-query errors
and invalidated capture states are explicit non-qualifying events.

The parser accepts these selection records (`selected: true`, positive
`record_id`, nonempty route/kernel) as well as the legacy `executed: true`
`launch`/`graph-record` record format. Different geometries and unique producer
record IDs may select different routes. Boolean values are not accepted as integer
geometry. Contradictory route choices for the same event identity, malformed
records, absent/partial coverage, or malformed unterminated final trace lines fail
and persist a failure in `capture.json`. A valid record with `phase: "unknown"`
only covers an inventory that explicitly asks for `unknown`; it never implies
phase mapping. The captured command log retains bounded raw run output, including
trace lines.

This patch does **not** cover Q3 pair/other frontends, non-Q3 linear formats,
linear-add, attention-input projections, GDN, non-Q3 SwiGLU, sampling, or other
operators; it does not identify model layer identity or phase. It is a useful
narrow selection trace, not comprehensive M0 dispatch evidence. No build,
compile, model run, timing result, or hardware qualification is claimed. The
upstream Apache-2.0 license is preserved; the newly added helper also carries an
Apache-2.0 SPDX notice.
