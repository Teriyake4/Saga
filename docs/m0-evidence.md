# M0 evidence and completion checklist

Status: **partial / blocked**, 28 October 2026. The user authorized M0 work using
GPT6 Luna implementation plus independent review/fix/re-review steps. Independent
slices may run in parallel with disjoint write scopes. This is not authorization
to claim missing measurements, approve arbitrary quality budgets, or complete
M1/M2 by changing their gates.

## What this continuation actually established

1. **GGUF source provenance (part of M0.2/M0.8).**
   `ggml-org/ggml` is pinned at
   `facdc44a67da17df4433adcc468b424f48e9ded2`. Nine required source/specification,
   API/header, build and license files were fetched, hashed, then independently
   re-fetched with matching hashes. The original 104 source/evidence records and
   historical attempts were preserved. The combined inventory has 114 entries:
   105 recorded-verified inputs and nine unverified observations (the original
   eight 404 probes plus the new optional commit API's rate-limited 403).
   A pinned raw-source URL/content hash is not a successful commit API response,
   signed-commit authentication, built reader, or verified GSQ/GGUF mapping.
2. **Evidence audit hardening (part of M0.3/M0.8).**
   The verifier checks final source URL identity and one-to-one retrieval-evidence
   IDs/required fields. Optional local artifact verification recomputes actual
   file size and SHA-256 rather than trusting a `verified` label. It performs no
   discovery, download or model execution. Byte matching does not establish
   component/frontend compatibility or model quality.
3. **Independent scalar fixtures (part of M0.6).**
   `bench/oracles/q3_scalar.py` and `bench/fixtures/` provide hand-auditable
   signed-int3 group-128 packing, exact finite-FP16 represented values and small
   projection/dot expectations. Tests use literal answers, a separate positional
   polynomial, all code positions, and every finite binary16 pattern checked
   against the standard library. This is the declared reference storage contract,
   not verification of publisher GSQ codes/scales, a GGUF profile, complete Qwen
   layer/GDN/state math, or an original-BF16 quality baseline.

All slices received separate Luna reviews and focused fixes/re-reviews. No model
weights were downloaded and no upstream executables were run. Retrieved source
text was not imported into the production runtime; only upstream license text was
preserved as source data. The server and its not-ready behavior are unchanged.

## Gate-by-gate status

| Gate | Present evidence | Remaining requirement |
|---|---|---|
| M0.1 | macOS arm64 development history plus bounded Linux host build/test evidence from Ubuntu 25.04 aarch64 OrbStack (2026-10-28); Linux host facts are recorded in [m1-evidence.md](m1-evidence.md) | Actual original RTX 4080 target hardware, OS/toolchain/driver, VRAM, CPU/RAM and power/display/runtime conditions; native Windows evidence remains unavailable |
| M0.2 | NInfer-4080, llama.cpp KV and GGUF specification/selected reader-source pins | Build/run the pinned reference, preserve reproducible commands/binary/patch identities, observe actual dispatch; verify the exact GSQ/Q4 GGUF mapping |
| M0.3 | Pinned metadata, structural binding checks and opt-in local byte verifier | Full original/GSQ/MTP/vision/tokenizer byte inventories, checksums and independent component/frontend compatibility; settle the GGUF component/profile layout |
| M0.4 | Existing license/source inventory; no inference code imported; Switchyard reuse forbidden pending license evidence | Per-initial-operator adaptation versus independent-implementation decision and any required per-file attribution review |
| M0.5 | Explicit candidate case inventory and fail-closed frozen-suite checks | Nonempty reviewed mandatory suite, exact rendered token IDs/configurations, native OS scope, observed precision/rounding/head/KV/graph/sampling settings and reference measurements |
| M0.6 | Independent storage and small tensor/projection fixtures | Complete target layer, recurrent/state, continuation and frontend fixtures with independent provenance; current partial fixtures do not satisfy the complete inventory |
| M0.7 | Quality schema/policy and explicit null/unfrozen budgets | Original-BF16 baseline execution, pinned datasets/disjoint splits, reviewed metrics/budgets/confidence/sample-size policy and required baseline results |
| M0.8 | Expanded source locks, license bytes, audit tooling and tests | Remaining artifact/dataset/converter/build/toolchain identities and reproducible execution evidence |

No whole M0 checkbox is closed by this continuation. The quality manifest's
complete independent-fixture inventory remains unresolved; the small scalar
fixture is not used to clear the broader layer/state evidence gate. Mandatory
cases, quality budgets and results remain unfrozen/null rather than fabricated.

## Local artifact byte verification

The opt-in layout is:

```text
ARTIFACT_ROOT/
  original-bf16-text/<manifest files[].path>
  gsq-text/<manifest files[].path>
  original-mtp/<manifest files[].path>
  original-vision/<manifest files[].path>
  tokenizer/<manifest files[].path>
  template/<manifest files[].path>
  reference-container/<manifest files[].path>
```

Declare the actual relative file paths, sizes and expected hashes in the reviewed
reference manifest. Pass the actual local root using `--artifact-root DIRECTORY`.
There is no automatic scan, remote fetch, manifest rewrite, compatibility approval
or inference. With no root, artifact byte claims cannot become verified merely
because status/hash metadata says so. The explicit root and stable hierarchy are
trusted inputs, not a guarantee against malicious concurrent filesystem changes.

```sh
python3 -B tools/verify_sources.py --help
python3 -B tools/verify_sources.py --structure-only
python3 -B -m unittest discover -s bench/tests -v
python3 -O -B -m unittest discover -s bench/tests -v
```

Final local validation: **57/57 Python tests passed**, both normally and with
`-O`; all **18/18 CTest groups passed** in the existing macOS Debug, Release,
ASan/UBSan and TSan presets. No CUDA, Windows, upstream reference or model run
was performed. The default audit remained exit 1 with all external gates blocked.
`git diff --check` passed.

The structure-only command checks schemas/consistency without accessing artifacts
or the network. The default audit intentionally exits **1 (blocked)** while M0
inputs are missing. Invalid evidence structure exits **2**. Source availability,
byte checks, mathematical tests and release qualification are distinct results.

## Inputs needed to finish

- Access to the original RTX 4080 target environment, or its collected, reviewable
  native hardware/toolchain and reference-run evidence. Native Windows remains the
  active production target; Linux host development/testing is now authorized, but
  OrbStack on ARM is not native Windows, x86_64 Linux, or CUDA/4080 qualification.
  If the reference cannot run natively on Windows, explicitly resolve/freeze the
  permitted reference OS scope rather than claiming cross-OS parity.
- Actual pinned model/component/tokenizer files on storage with sufficient room,
  and an independent original-BF16 execution environment (CPU/larger machine is
  permitted). This Mac had about 27 GB free at inspection; no large download was
  started or assumed feasible.
- Reference dispatch/results and independent baseline evidence needed to propose
  and review a real performance-suite freeze and quality preregistration. Existing
  performance thresholds remain unchanged; unresolved quality budgets cannot be
  filled from those performance thresholds.

See `docs/source-lock.md`, `docs/quality-policy.md`, `docs/artifact-format.md` and
`IMPLEMENTATION_PLAN.md` for the authoritative provenance and acceptance rules.
