# M0 paired token-NLL evaluator

`tools/m0_quality.py` is a bounded, standard-library-only JSONL analyzer for a
future original-BF16 baseline versus variant run. It does not invoke a model,
verify checkpoint bytes against an authenticated source, prove execution
precision, or qualify complete model quality. Any provenance is explicitly a
**supplied assertion**, not physical verification. Running it is not evidence
that original BF16 has been executed.

## Input protocol

The evaluator requires a standalone preregistration JSON and its exact lowercase
SHA-256 (`--prereg-sha256`). It hashes and validates that file before opening
any JSONL evaluation records. No generated template, baseline, dataset, budget,
threshold, or approval is supplied. Preregistration must include:

- `freeze_record` (nonempty review/freeze record).
- `baseline`: `checkpoint_revision`, nonempty `files` entries containing
  `path` and lowercase SHA-256 `sha256`, explicit BF16 `precision`,
  `harness_revision`, and declared `environment`.
- `dataset`: revision and SHA-256; unique, disjoint `calibration_ids` and
  `heldout_ids`; `calibration_content_sha256`, exact SHA-256 hashes of
  whitespace-normalized calibration content.
- `metrics`: `primary: token_weighted_mean_nll`, `direction: lower`;
  `confidence`: `method: paired_percentile_bootstrap`, declared `level` and
  `replicates` (at most 100,000); unique nonnegative `seeds`; positive
  `sample_count` exactly equal to the held-out ID count; and
  `stopping_rule: fixed-full-heldout` (the only supported stopping design).
- Explicit `budget_approval` with `approved: true`, reviewer, date and
  baseline evidence, plus finite nonnegative
  `regression_budget.max_upper_nll_regression`.

Every JSONL line is an object with exactly `id`, `content`, `context_tokens`,
`token_count`, `nll_sum`, and `seed`. Baseline and variant must have one row per
held-out ID, exact matching IDs/normalized content/context/token count/seed,
unique normalized content within each file, and no normalized-content hash
present in calibration. The evaluator sums NLLs and token counts to report
token-weighted mean NLL and perplexity. It bootstraps paired per-row NLL
differences using a deterministic PRNG seeded by the sum of preregistered
seeds; confidence interval is the declared percentile interval. The budget
comparison uses the interval upper endpoint. This is a simple declared
bootstrap procedure, not an assertion of statistical adequacy for every task.

A complete but unapproved/missing budget must not produce a budget comparison;
invalid approval/budget data is rejected. A preregistration missing required
fields returns `not_evaluated` without opening records. Missing held-out rows,
invalid pairs, duplicate keys, JSON non-finite constants, booleans in numeric
fields, and capped input excess fail closed. Limits: 1 MB preregistration,
1 MB per JSONL line, 256 MB per JSONL file, 100,000 records and 100,000
bootstrap replicates.

Example invocation (all paths and hash must be supplied by the operator):

```sh
python3 tools/m0_quality.py \
  --prereg preregistration.json --prereg-sha256 <sha256> \
  --baseline-jsonl original-bf16.jsonl --variant-jsonl candidate.jsonl \
  --output result.json
```

The output path must not already exist. `inventory_validation` means inputs
conformed to this tool's checks; it is not quality validation. Even an
in-budget result remains `qualification: unqualified` and `full_model_quality:
not evaluated`. Actual pinned original-BF16 model execution, source/checkpoint
provenance verification, independent harness review, and all other frozen
quality gates remain necessary. No budgets or quality thresholds are invented
by this tool.

Run the tests with `python3 -B -m unittest bench.tests.test_m0_quality` from
the repository root.
