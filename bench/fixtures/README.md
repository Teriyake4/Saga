# Independent Q3 scalar mathematical fixture

This fixture pins only the declared storage arithmetic: one group of 128 signed three-bit two's-complement codes in 48 bytes, with code zero starting at the least-significant bit and bytes ordered in increasing little-endian stream order. A finite binary16 scale is interpreted as its exact represented rational; the tiny projection is an exact scalar dot product over those represented values.

The fixture is deliberately **not** a production quantizer/converter, GGUF schema or source mapping, nor an equivalence claim to Saga's model path. It does not cover a complete Qwen layer, GDN/state, tokenizer, GGUF source compatibility, publisher BF16 scale quality/conversion, or BF16 quality fixtures. Nonfinite FP16 values are excluded because they do not have finite `Fraction` representations; this says nothing about any production policy. M0.6 remains partial.

## Provenance and independence

The mathematical contract is pinned to `yalding/ninfer-4080` revision `2469b5752672ad597055af61fe28e8e7ced20056`, file `src/ops/linear/q3/q3_rowsplit_storage.cuh`, at:

`https://raw.githubusercontent.com/yalding/ninfer-4080/2469b5752672ad597055af61fe28e8e7ced20056/src/ops/linear/q3/q3_rowsplit_storage.cuh`

The pinned SHA-256 is `acd422880252865b88c4b2ae97139c953f750695ec2e0f76f73ca36d164130db`, matching `bench/reference/source-lock.json` and the Q3 contract in `docs/source-lock.md`. The Python oracle is independently written scalar arithmetic; no upstream source is copied, imported, or executed. Known answers are hand-auditable: the repeated eight-code cycle has the literal three-byte word `ac8f68` repeated 16 times. Independently, its integer code stream is `sum((code % 8) * 8**index)` over the eight codes, which is `0x688fac`; emitting that integer low byte first gives `ac 8f 68`. Sparse boundary bytes are listed explicitly, and the projection's dot is `-4*1 + 3*2 - 1*3 + 2*4 = 7`.

## Rerun

From the repository root, run the focused test only:

```sh
python3 -B -m unittest discover -s bench/tests -p 'test_independent_oracles.py' -v
```

The JSON fixture is the hand-authored known-answer source; tests compare independent literal expectations to the scalar oracle, rather than relying on pack/unpack round trips as the sole oracle.

## Independent small layer/state fixture (limited mathematical conventions)

`layer_state_known_answers.json` and `oracles/layer_state.py` add stdlib-only,
small fixtures for RMSNorm, one-pair RoPE, SwiGLU, single-head causal attention,
a scalar gated-delta-style recurrence, and causal depthwise FIR history. These
are deliberately **declared fixture formulas**, not assertions of the locked
Qwen3.8-27B implementation:

- RMSNorm: `x_i / sqrt(mean(x*x) + epsilon) * weight_i`.
- RoPE: adjacent pair rotation by `position / theta` (one pair only).
- SwiGLU: `silu(gate) * up`, with `silu(z)=z/(1+exp(-z))`.
- Attention: stable softmax of causal `q·k/sqrt(d)` over positions through the
  current token, followed by the weighted value sum; no masks beyond causality.
- Recurrence: scalar `s' = s + beta*(v-k*s)` and output `k*s'`.
- Convolution: per-channel causal FIR with kernel index zero on the current
  token; state contains prior samples, newest first.

Known-answer outputs are literal JSON values chosen independently of the oracle.
The nonzero attention case uses logits `0` and `ln(2)`, hence softmax weights
`1/3` and `2/3` and weighted value `7`; the recurrence uses two distinct keys
and beta values. Python binary64 is used; comparisons allow absolute and relative
tolerance `1e-12`. Transcendental results therefore are not bitwise targets.
`ScalarRecurrenceState` provides a small mutable toy API with `step`, `reset`,
`checkpoint`, `rollback`, and independent `clone`; tests cover continuation,
chunk-grouped versus tokenwise calls, restore/reset, and branch isolation. The
convolution fixture passes immutable history explicitly and is not a state API.

The source lock pins upstream NInfer-4080 references and identifies selected
SwiGLU/GDN sources, but this fixture work did not verify their precise formulas,
parameterization, normalization/epsilon, rotary convention/frequency schedule,
attention head mapping/masks, convolution layout, state update/reset semantics,
or target rounding/accumulation against those sources or an executable. Inputs
reject booleans/non-numbers and nonfinite values; derived overflow/nonfinite
outputs raise `ValueError`. RMSNorm and SiLU use scaled/stable forms where needed.
In particular the scalar recurrence object is only a pedagogical state fixture,
not Qwen GDN. No complete Qwen weights/profile/frontend, original BF16 comparison, or
BF16 qualification is claimed. M0.6 remains partial; see `docs/m0-evidence.md`
and `docs/source-lock.md` for unresolved evidence gates.
