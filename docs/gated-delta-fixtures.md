# Gated-delta recurrent fixture (limited source-derived coverage)

`bench/oracles/gated_delta.py` is an independent, stdlib-only CPU implementation
of one recurrent GDN transition, with literal answers in
`bench/fixtures/gated_delta_known_answers.json`. It does not copy or import
reference code. Provenance is pinned to NInfer-4080 revision
`2469b5752672ad597055af61fe28e8e7ced20056`; per-file SHA-256, relevant symbols
and line ranges, model-card identity, and license identity are in the JSON.
The repository-level source lock records the revision, Apache-2.0 license and
model-card hash. The four CUDA/header source bytes were locally hashed for this
fixture but are not individually enumerated as source-lock inputs; the JSON
states that distinction.

## Verified formula boundary

The inspected `recurrent.cuh:104-121` computes, for each value row `r`,
`p_r = S[r,:] dot k`, `delta_r = beta * (v_r - exp(g) * p_r)`, then
`S'[r,c] = exp(g) * S[r,c] + delta_r * k_c`. The fixture spells
`alpha = exp(g)` and accepts `alpha` directly. Its `transition` inputs are
`S` shaped `[Dv,Dk]`, `k` length `Dk`, `v` length `Dv`, nonnegative `alpha`,
`beta` in `[0,1]`, `q` length `Dk`, and output scale. It returns updated state,
per-value-row deltas, and `output[r] = scale * (S'[r,:] dot q)`.
`run_recurrent_sequence` applies each token's state update before publishing
that token's output (`recurrent.cuh:616-639`).

The reference has an optional Q/K L2-normalization branch with epsilon `1e-6`
(`recurrent.cuh:32,63-77,124-141`). The fixture does not pick the template
setting: callers provide the exact key/query vectors at that operator boundary.
The source gate kernel forms `g = -exp(A_log[h]) * softplus(a+dt_bias[h])` and
`beta = sigmoid(b)` (`gdn_gating.cuh:15-27`). This fixture accepts already
transformed `alpha=exp(g)` and `beta`; it does not invent those parameters or
reimplement gate transforms. Matrix layout/head mapping details in
`common.cuh` and `recurrent.cuh` establish a value-by-key state, but this tiny
test uses dimensions 2x2, not production geometry.

The hand answer is independently inspectable. Start with identity state. Token
1 uses `k=(1,1)`, `v=(2,0)`, `alpha=1/2`, `beta=1/2`, producing
`delta=(3/4,-1/4)` and `S1=((5/4,3/4),(-1/4,1/4))`. Token 2 uses the distinct
`k=(2,-1)`, `v=(1,3)`, `alpha=1/4`, `beta=1/2`, producing
`delta=(9/32,51/32)` and
`S2=((7/8,-3/32),(25/8,-49/32))`. Queries/scales yield literal outputs
`(11/8,1/8)` and `(25/128,51/128)`. These are rational calculations, not
oracle-generated expected values. Oracle arithmetic is binary64 and tests use
`1e-12` absolute and relative tolerance.

`GatedDeltaState` supplies mutable lifecycle behavior specifically for this
small recurrence: step, immutable checkpoint, rollback, reset and deep clone.
Tests compare tokenwise and grouped calls, and branch isolated clones. They do
not claim actual production allocator/snapshot slot semantics.

## Additional inspected operator and remaining boundary

`causal_conv1d.cuh:73-100,234-258` declares a depthwise width-4 causal filter:
taps apply to three prior history values oldest-first and then the current
sample; decode shifts three history slots and applies SiLU. This fixture does
**not** implement or qualify that convolution operator. Although the source
establishes its tap/history ordering, a separate fixture with literal expected
answers is still required.

The model card at the same revision is metadata about the NInfer `.ninfer`
artifact and its intended Qwen3.8-27B profile (including the reported 320 text
matrices); it is not original model source or a numerical reference for every
state tensor. No weights were downloaded or executed here. This work does not
establish original-BF16 parity, model state serialization/layout across the
whole runtime, exact compiled precision/rounding, complete gate-to-projection
compatibility, Qwen profile/frontend compatibility, or M0 completion. The
source-lock's previous evidence and remaining gates remain applicable.

Run the focused tests from repository root:

```sh
python3 -B -m unittest discover -s bench/tests -p 'test_gated_delta_oracles.py' -v
python3 -O -B -m unittest discover -s bench/tests -p 'test_gated_delta_oracles.py' -v
```
