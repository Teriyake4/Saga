# Quality qualification policy (M0)

**Status: blocked / unfrozen / not qualified.** This policy defines evidence and
preregistration rules; it does not supply invented budgets or passing results.
The machine-readable inventory is `bench/quality/manifest.json`, with
`bench/quality/manifest.schema.json`. The authoritative milestone requirements
are `IMPLEMENTATION_PLAN.md` sections 2 and 4 and `PROPOSAL.md` section 12.

## Keep three comparisons separate

1. **Implementation correctness:** identical represented weights, tokenizer and
   prompt tokens versus an independent scalar/tensor/layer/state oracle. Record
   operator-specific tolerances before optimization. Test exact code/value
   preservation, record copying and rollback exactly where the contract requires
   it; use declared dtype/rounding tolerances for approximate arithmetic.
2. **Quantization/system quality:** GSQ3 text versus the matching original BF16
   checkpoint; each K/V pair versus two-byte K/V with the same represented text
   weights; imported/requantized MTP and vision versus their source precision;
   complete image-to-answer behavior versus the original multimodal model.
3. **Performance:** matched, pinned reference execution on the same original
   RTX 4080. A plausible answer, a publisher's evaluation, source inspection or
   host test cannot establish correctness, quality or competitive performance.

Original BF16 execution can use CPU or a larger reference machine. It need not
fit on the 4080. Pin that execution environment, independent harness and baseline
precision; do not substitute the quantized reference engine as the sole oracle.
No original-BF16 execution or full-weight checksum verification occurred here.

## Preregister before affected tuning

For each comparison, freeze all `freeze_required_fields` in the quality manifest:

- Baseline checkpoint revision, **actual file checksums**, precision and execution
  environment; relevant component and frontend binding evidence.
- Dataset revisions/checksums and exact calibration/held-out record IDs. Prove
  split disjointness and prevent prompt, near-duplicate and tuning leakage.
- Tasks, prompts, rendered token IDs, tokenizer/template hashes and context lengths.
- Metrics, directions, regression budgets and their baseline evidence/reviewer
  approval. Numeric/operator tolerances are not task-quality budgets.
- Confidence method/level, seeds, sample sizes, stopping/noise and multiple-test
  policy, evaluation harness revision and a reviewable dated freeze record.

Every unresolved value stays JSON `null` with a blocker, not zero, an empty
passing result, an arbitrary tolerance, or a guessed model size. Budgets must be
derived from baseline evidence and review **before** the affected conversion or
optimization. Do not tune against held-out outcomes, widen a budget after seeing
a regression, or remove/relabel a frozen case to obtain a pass. Amendments need
an explicit new preregistration with rationale and preserved prior results.

## Required quality gates

| Inventory ID | Milestones | Required evidence; currently missing |
|---|---|---|
| `gsq-vs-original-bf16` | M0/M3/M5 | Original BF16 execution; held-out distribution/perplexity and task quality; separate represented-weight implementation error |
| `kv-vs-two-byte` | M7a/M7b/M9 | Every required pair: append/reconstruction, context-dependent logit/task error, long-context retrieval and feasible measured profiles |
| `mtp-component-vs-source` | M3/M9 | Compatible original MTP/vocabulary provenance, component error, draft acceptance and net committed-token benefit |
| `vision-component-vs-source` | M3/M10 | Source-precision features/logits, CPU/GPU placement, grid/patch/position and expanded-token alignment |
| `exact-stochastic-mtp` | M9 | Enumerable small-vocabulary target distributions and multiround statistics under frozen sampling/stopping rules |
| `multimodal-vs-original-bf16` | M10 | Held-out complete image-to-answer tasks/logits, image-plus-MTP continuations; feature parity alone is insufficient |

Stochastic fixtures must include zero probabilities, restricted proposal support,
forced rejection at each depth, all-accepted/bonus/short rounds, processors and
stopping boundaries. Freeze confidence and sample design before optimization;
no statistical passing thresholds are invented in this increment.

The quality manifest enumerates all 11 required same-codec pairs and the four
curated mixed pairs (`q8_0/q4_0`, `q8_0/q5_0`, `q8_0/q5_1`,
`int8_g64/int4_g64`). This is a **requirements inventory**, not supported kernel
registration. Each row remains unqualified with null geometry, capacity, backend,
placement and evidence. Main and native-MTP cache geometry and enabled
append/prefill/decode/verify paths need separate evidence. Custom group-64 and
pinned GGML block-32 contracts must not be conflated. No arbitrary Cartesian
product of K/V formats is implied; no context fit is inferred from bit width.

## Partial independent scalar fixtures

`bench/oracles/q3_scalar.py`, `bench/fixtures/q3_scalar_known_answers.json` and
`bench/tests/test_independent_oracles.py` now exercise the locked reference's
signed-int3 group-128 storage contract, exact finite-FP16 represented values and
small scalar projections. Literal known answers, a separate positional-polynomial
construction and all finite binary16 patterns provide independent cross-checks.
They do not verify publisher GSQ codes/BF16 scale conversion, a GGUF mapping,
complete Qwen layer/GDN/state behavior, frontend bindings or original-BF16 quality.
The full `independent_fixture_inventory` remains unresolved; a nonempty toy fixture
must not be used to clear all model/layer/state evidence requirements. See
[../bench/fixtures/README.md](../bench/fixtures/README.md).

## Frozen mandatory performance suite versus proposals

`bench/reference/manifest.json` currently has:

- `suite.status = "unfrozen"`, unresolved native OS scope and no freeze record.
- `mandatory_cases = []`: a **blocker**, not an empty suite that passed.
- Separately named `candidate_cases`: proposed plain/native-MTP cases at 128,
  2048 and near-8K prompt lengths, targeting 512 controlled output tokens. Near-8K
  exact length/output reserve and every execution configuration remain unresolved.
- No reference build, runtime trace, raw measurements or comparison results.
  Prose/code/reasoning/multi-turn and feasible larger contexts are proposals too.

Before promoting any case, freeze the complete comparable suite and its digest,
review record, exact prompt IDs, model/component checksums and the reference
manifest's `freeze_required_fields`. Those include actual phase/site A16/A8
routes and rounding/accumulation, K/V numerical contracts in both pools,
sampling processors/defaults/order, seeds/stops, full/restricted draft head and
vocabulary, actual verify widths, graph profiles and chunk sizes. Use prefix
reuse OFF with empty request state; use ngram OFF for native-MTP comparisons.
A flag, an upstream command example or an author-reported dispatch claim is not
a runtime trace. Pin any necessary A16-control patch and its hash first.

Freeze timing definitions, warmup, at least five measured short-test repetitions,
noise policy and raw-record schema. Record physical GPU SKU (not 4080 Super),
VRAM, CPU/RAM, OS, compiler/toolkit/driver, clocks/thermals/power/display and
background GPU conditions. Compare on the same physical hardware and OS.
Report equal-configuration, quality-matched-system and best-system separately.
Neither unmatched activation/KV/head choices nor DFlash2 replace a native-MTP
mandatory case. Cold startup, retained-prefix and image TTFT are separate results.

The **existing plan's performance thresholds**, not new quality budgets, are
per frozen comparable case: median committed decode throughput >=0.95 times
reference and median uncached TTFT <=1.10 times reference. Both must pass every
mandatory case; averages cannot hide a miss. Missing/unmatched results block
M8 plain performance and M11 plain/native-MTP release qualification. If a native
Windows reference is unavailable, explicitly freeze Linux-only performance
scope; own-engine feature/quality/memory/lifecycle tests remain mandatory on
Linux and native Windows/MSVC. WSL is not native Windows evidence.

## Validation and reporting

`python3 -B tools/verify_sources.py --structure-only` validates JSON schemas and
consistency, rejecting candidate cases labeled mandatory/frozen, duplicate IDs,
unfrozen suites with mandatory rows, incomplete frozen configurations and
incomplete frozen quality preregistration. It does not run a model or approve a
budget. Source retrieval observations must match locked IDs one-to-one and preserve
final URL identity. Optional `--artifact-root DIRECTORY` streams only declared local
files under `DIRECTORY/<artifact-id>/<files[].path>` to check actual sizes/hashes;
without a root, a declared verified artifact cannot rely on metadata claims alone.
There is no scan, download or compatibility approval. The root and stable hierarchy
are trusted inputs. Default and optional online audits additionally fail on unverified
required source/artifact inputs and unresolved suite/hardware/quality evidence.
The verifier deliberately reports `qualification = "not-evaluated"`; it is not a
numerical evaluator or release-qualification tool.

Keep raw measurements, artifact/build/harness hashes, operator fixtures and
per-case summaries when they eventually exist. Missing mandatory measurements,
unresolved affected budgets, component compatibility, dispatch or native hardware
block the affected gate. Integration can proceed only as an explicitly
unqualified candidate, not as a feature-complete or performance-qualified release.
