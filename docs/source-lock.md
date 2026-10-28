# M0 source lock and retrieval evidence

Status: **partially source-verified; M0 blocked; no inference code imported**.
Inspection date: 2026-10-27. Authority: `PROPOSAL.md` section 16 and
`IMPLEMENTATION_PLAN.md` M0. Source inspection is not build, dispatch, numerical,
model-quality, memory, or performance qualification.

## Evidence inventory

- `bench/reference/source-lock.json`: exact requested revisions/paths, HTTPS URLs,
  required/optional distinction, byte counts and SHA-256 of successfully retrieved
  response bodies. Unretrieved inputs have **null**, not invented, hashes.
- `bench/reference/retrieval-evidence.json`: actual HTTP outcomes, successful body
  hashes, final URLs, bounded HTTP error bodies and historical attempts.
- `bench/reference/model-metadata.json`: Hugging Face discovery revisions and file
  listings. Listed weight files were **not downloaded or hashed**.
- `bench/reference/licenses/`: unchanged retrieved license texts, each tied to its
  source-lock entry and independently checked against its retrieved byte hash.
- `bench/reference/manifest.json` and `bench/quality/manifest.json`: blocked
  artifact, comparison, hardware, dispatch and quality inventories. Neither is a
  result set or a frozen mandatory benchmark suite.

The initial snapshot contains 104 URL inputs: 96 HTTP-200 body hashes and eight
HTTP-404 probes. All five GitHub commit APIs ultimately returned the exact
requested commit SHA. This verifies accessible pinned source content, not every
file in each tree, a signed commit, or an independently reproduced binary.
Raw-source byte hashes are local retrieval evidence; a model card's weight hash
is only a **publisher claim** until independently checked on the actual bytes.

## Pinned engines and licensing

| Repository | Exact revision | Retrieved and inspected |
|---|---|---|
| `Neroued/ninfer` | `64d10fbf9f8e6bfc79ab0570b94674c5b9997929` | README, Apache-2.0 LICENSE, root CMake, architecture/model references, selected graph/planning/verification sources |
| `yalding/ninfer-4080` | `2469b5752672ad597055af61fe28e8e7ced20056` | README, Apache-2.0 LICENSE, root CMake, artifact card, Q3 dispatch/storage/A8/SwiGLU, selected GDN/KV sources and converter/verifier sources |
| `Niko1221/Strata` | `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` | README, MIT LICENSE, root CMake, execution/benchmark documents and selected prefill/expert-cache/speculation sources |
| `zero734kr/switchyard` | `b4bd35679da0ec5de08027637e7b94b4578a2cf8` | README, root `pyproject.toml`, kernel methodology, guards and three selected fusion sources; license unresolved |
| `ggml-org/llama.cpp` | `e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf` | README, MIT LICENSE, root CMake, cache CLI parser, block definitions/codebook, scalar quantizers/dequantizers and dependent headers |

Examples of actual retrieved SHA-256 values (the JSON lock has the complete list):

| Pinned file | SHA-256 |
|---|---|
| NInfer-4080 `README.md` | `df0bc51498227a66aa98a0ce1630d4d709fdcc4dfa0b48df50628be2fc142a69` |
| NInfer-4080 `src/ops/linear/q3/q3_dispatch.cpp` | `49a34b256ec826813a0323172513013b14b677105836a8d5c781390bdb677f41` |
| NInfer-4080 `src/ops/linear/q3/q3_rowsplit_storage.cuh` | `acd422880252865b88c4b2ae97139c953f750695ec2e0f76f73ca36d164130db` |
| NInfer-4080 `tools/convert/qwen3_8_27b/gsq3_source.py` | `12db75b4dfcfefb5f80662b3e69ce8053cf82276e473c29cd040e75d86b585fa` |
| llama.cpp `ggml/src/ggml-common.h` | `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c` |
| llama.cpp `ggml/src/ggml-quants.c` | `5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab` |

### Actual failures and limits

- First NInfer-4080 commit API probe: **HTTP 403, `rate limit exceeded`**.
  A later request succeeded with the requested SHA; both facts are recorded.
- Switchyard root `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `COPYING`, and `NOTICE`:
  **HTTP 404**. This is not a conclusion about every possible license location,
  but permission to adapt its code has not been established. Required `LICENSE`
  stays unresolved and blocks reuse; public visibility is not a license.
- Switchyard `CMakeLists.txt`, `setup.py`, and `kernels/pyproject.toml` probes:
  **HTTP 404**. Root `pyproject.toml` succeeded; it declares Python >=3.12,
  hatchling and PyYAML, not a CMake/CUDA engine build.
- Initial Hugging Face `/resolve/` requests were stopped by this verifier's
  overly strict redirect policy, **not by remote HTTP unavailability**. Historical
  errors retain that distinction. After narrowly allowing same-file,
  same-revision Hugging Face metadata-cache redirects, the pinned metadata
  requests succeeded. Weight/CDN redirects remain disallowed.
- A host-specific `allow_hosts` request was attempted for GitHub/raw/Hugging Face;
  the terminal reported that unrestricted network access was already enabled for
  this thread and required omitting the redundant option. Subsequent requests
  still used only explicit HTTPS metadata/source endpoints.

## What the source actually establishes

### Build declarations, not tested toolchain pins

NInfer root CMake requires CMake 3.28/C++20 and restricts its architecture to
`120a`; it is not an SM89 build recipe. The 4080 fork declares CMake 3.28/C++20,
restricts architecture to `89`, and checks CUDA compiler >=12.8. Its published
artifact/container instructions discuss a CUDA 13.1-or-newer driver. These are
upstream declarations with different meanings, **not locally tested Saga
compiler/toolkit/driver pins**. Strata's root CMake declares CMake 3.24/C++20 and
optional backends; its larger-workload/offload paths are not a matched dense
GSQ3 comparison. Switchyard is Python/Dynamo/vLLM orchestration plus guarded
Triton experiments, not demonstrated 4080 parity. No reference was built or run.
No inspected upstream build commands or downloaded Python modules were executed.

### 4080 Q3 dispatch and packing

At the locked fork revision:

- `q3_dispatch.cpp:46-55` / `q3_dispatch.h:15-27` select A8 only with
  `AllowA8`, positive dimensions, exact unpadded K and K divisible by 64.
  Tall A8 requires T >=129 and N divisible by 128. Small-T A8 covers T=2..16
  with N divisible by 32 and K divisible by 512. Other eligible paths resolve
  A16 launches. These are source predicates, **not observed site coverage**.
- `q3_dispatch.cpp:63-91` validates group-128/FP16-scale row-split inputs and
  dispatches activation quantization and the selected A8 launcher before the
  A16 fallback. `select_q3_launch()` alone therefore does not prove A16 execution.
- `q3_linear_swiglu.cu:77-151` separately branches through the same A8 predicate,
  small-T/fused GEMV/tall paths and a chunked FP32 projection-plane epilogue.
  The latter rounds `silu(gate) * up` to BF16 at the epilogue. Full phase/site
  accumulation and intermediate-rounding parity remains unresolved.
- `q3_rowsplit_storage.cuh` describes 128 signed three-bit codes in 48
  little-endian bytes per group, a separate FP16 scale plane and two's-complement
  sign extension. `gsq3_source.py` describes the publisher's shifted unsigned
  grid and BF16 multipliers, a code-plane sign transform and an audited
  BF16-to-FP16 scale conversion. The converter and verifier explicitly handle
  subnormal rounding. **Do not call all represented scales bitwise lossless.**
- The retrieved artifact card reports 218 rounded scale words and publishes a
  weight-file checksum. Those counts/checksum and its performance/quality tables
  are author-reported, not independently verified Saga conversion or results.
  Its command example enables `--ngram chain`, an optimized draft head and a
  different KV recipe. It cannot silently satisfy a matched native-MTP,
  ngram-OFF, same-head/same-KV A16 gate.

No A16 control patch, patch hash, runtime trace, reference binary hash, benchmark
run command or native reference result has been frozen. Obtain evidence per
phase and operation site before freezing a comparable case; patch only if needed
and record exact patch bytes/hash and build/run commands.

### llama.cpp numerical-source provenance

Lock `common/arg.cpp`, `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`,
`ggml/src/ggml-quants.h`, `ggml/src/ggml-impl.h`, and `ggml/include/ggml.h`
together. The common header contains `block_q8_0`, `block_q5_0`, `block_q5_1`,
`block_q4_0`, `block_q4_1`, `block_iq4_nl`, their size assertions and
`kvalues_iq4nl`. The scalar C file contains corresponding `quantize_row_*_ref`
and `dequantize_row_*` routines. Record any further transitive source used by a
future independent fixture. CLI acceptance does not qualify a CUDA K/V pair.
No decoder or codebook has been copied into Saga inference code in this work.
Custom `int8_g64`/`int4_g64` are separate proposal-owned contracts, not aliases
for these GGML block-32 formats. The quality manifest enumerates every required
same-codec/mixed pair without claiming implemented or qualified support.

## Hugging Face metadata: accessible, not weight-verified

| Repository | Discovered revision, then used for pinned metadata requests |
|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` | `b5ce0b76f60020a875dee4f6ec9d934cca4121e4` |
| `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| `roofkid/Qwen3.8-27B-GSQ3-NInfer` | `d648daf360dea5a9d0731071b5990da5293f525a` |

Official dense config SHA-256:
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
GSQ config SHA-256:
`3c44b7ff4556e7779226d04448408aa727b8a2a0f893a2a2f2869aaaa6cbd384`.
The retrieved configs declare `qwen3_5`, 64 text layers, hidden size 5120,
intermediate size 17408, 24 attention heads, four KV heads, head dimension 256,
FP32 recurrent state and vocabulary size 248320. These are metadata declarations,
not verified tensor geometry, allocation fit, or implemented model math.

The GSQ card/config state three-bit group-128 body weights and four-bit group-64
embedding/head weights. The card explicitly says MTP modules are not included;
a config declaring an MTP layer does **not** establish that those weights exist.
Vision is described as preserved but not covered by its text-only calibration.
Matching original MTP/vision tensor inventory, checksums, vocabulary ties,
requantization and end-to-end quality remain blocked. Template and tokenizer
*configuration* bytes were hashed; full `tokenizer.json`/vocab/merges bytes,
rendered prompt IDs and frontend execution were not verified.

Original dense LICENSE is Apache-2.0; the GSQ card declares Apache-2.0, but its
listing has no standalone LICENSE. The retrieved Flash-Next LICENSE is **Qwen
Community License 1.0**, not Apache-2.0; it requires separate review and is a
future, out-of-scope integration. Do not transfer dense-model license assumptions
to it. The roofkid card/artifact manifest is metadata, not a verified weight file.
No model weights, safetensors, containers or `.ninfer` artifacts were downloaded.

## License preservation and host dependencies

For any later imported/adapted file, first record original URL, commit, path,
content hash, license, copyright/NOTICE requirements, modifications/patch hashes
and decision to adapt versus independently implement. Preserve applicable full
license text and notices; repository-level licensing alone is not a complete
transitive attribution audit. No imported inference source or adaptation decision
is approved by this snapshot. Switchyard reuse is specifically blocked pending
license evidence.

Host dependencies are vendored with their licenses: HTTP `cpp-httplib` v0.28.0
plus an explicit Saga hardening patch, unmodified JSON `nlohmann/json` v3.11.3,
and unmodified TOML `toml++` v3.4.0. `third_party/lock.json` records exact URLs,
installed byte counts and SHA-256 hashes, plus the pristine upstream identity and
separately hashed patch for cpp-httplib; `python3 tools/verify_dependencies.py`
checks installed files and patch integrity offline. The patch preserves raw HTTP
header values, strictly parses bounded chunk/trailer framing, and services accept
retry cancellation using native platform errors. Reproduction instructions and
transport limits are in `third_party/README.md`. The HTTP pin was
advanced from the initially considered v0.18.3 before integration to obtain an
explicit header-count bound. See `third_party/README.md` for license and transport
limitations. Host builds do not fetch or execute remote dependency scripts.
macOS host validation is recorded in `docs/status.md`; no production
Linux/MSVC/CUDA toolchain qualification is inferred from vendoring or source pins.

## Reproduce the checks

From the repository root, with Python 3 (standard library only):

```sh
python3 -B tools/verify_sources.py --structure-only
python3 -B tools/verify_sources.py --output bench/reference/verification-offline.json
python3 -B tools/verify_sources.py --online --output bench/reference/verification-online.json
python3 -B -m unittest discover -s bench/tests -v
```

Default mode is offline: it cross-checks expected hashes/byte counts against the
recorded retrieval evidence and preserved license bytes, **not live remote
availability**. `--online` explicitly re-fetches only bounded allowlisted HTTPS
source/metadata endpoints and checks against existing expectations; it never
updates hashes to make a mismatch pass. Network use is opt-in and is not part of
the build. GitHub/Hugging Face APIs can be rate-limited or change discovery
metadata; optional discovery failures do not erase pinned-file evidence.

Reports are canonical sorted-key JSON without wall-clock timings; unchanged
recorded inputs give byte-identical offline reports. Exit 0 means only the
requested check succeeded, never performance/model qualification. Exit 1 means
required source/artifact/baseline/hardware inputs are unresolved; currently this
is the **expected** readiness outcome. Exit 2 means invalid schema/consistency or
malformed input. Structure-only exit 0 explicitly says `structure-valid-not-qualified`.
Schemas use JSON Schema draft 2020-12; the stdlib checker implements only the
subset used in these files. External full-schema validation may be added, but no
third-party validator is required. Preserve snapshots and failures for review;
do not silently replace pins, remove mandatory inputs or freeze candidate cases.
