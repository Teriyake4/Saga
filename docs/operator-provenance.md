# Operator provenance and implementation decisions

**Decision snapshot: 29 October 2026.** This records provenance policy for the newly authored CUDA operator work. It is not a statement that M0 is complete or that any operator is qualified.

## Decision

The newly added BF16 baseline and SM89 WMMA kernels are independently written for Saga. Their correctness oracles must also be independent: test vectors and expected values must not be derived from the implementation under test. Do not infer that code or numerical expectations were imported from cached upstream sources. No cached upstream artifact is attribution evidence.

For the initial operators, the adoption decision is **independent implementation unless a later, explicit per-operator attribution/adaptation approval is recorded**. Any future adaptation must identify the exact upstream repository/revision/file and symbols, applicable license/notice, patch/adaptation history, and review decision before importing code. Switchyard remains reuse-blocked pending license resolution. Public descriptions or cached source do not grant reuse permission.

## Required per-operator record

For each operator, retain:

- Saga source path(s), implementation author/provenance, algorithm and numerical contract.
- Whether independent implementation or explicitly approved adaptation; if adapted, upstream revision, file/symbol ranges, license and notices.
- Independent oracle/fixture origin and generation procedure; never use the implementation to generate its own expected answers.
- Supported shapes, dtypes, accumulation/rounding, architecture and compile flags.
- Separate compile, device-link, device-run, correctness, quality and performance evidence. Do not collapse these into a single “tested” status.

Current CUDA compilation evidence is limited to reported Linux ARM64 Clang 20 device-only SM89 and host-object compilation using local CUDA 12.8 wheels. It does not establish device linking, execution, numerical correctness, performance, Windows support, or an attribution relationship to upstream code. Current M0 suite remains UNFROZEN. A narrow source-only Q3-linear/SwiGLU route tracing patch exists, but it lacks complete phase/site instrumentation and there is no observed dispatch or reference build/run, BF16 model baseline, or approved quality budget. The source-acquisition receipt records archive bytes/hash/sourceDigest only; it is not runtime evidence. See [m0-evidence.md](m0-evidence.md), [m2-evidence.md](m2-evidence.md), and [implementation snapshot](implementation-2026-10-29.md).
