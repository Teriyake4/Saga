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
