# Inference Engine Implementation Plan

**Status:** Linux host development/testing and host-testable M2 implementation are authorized; native Windows remains the active production target. M2.1–M2.7/M2.9 host slices and M2.8 profile/selection policy are implemented. Final macOS/Linux host matrices passed 25/25 in ten configurations after the final test-only early-notification regression; see [docs/m2-evidence.md](docs/m2-evidence.md). Native Windows/MSVC, physical original-4080, model/reference evidence and real production algorithms remain pending. Earlier no-M2 instructions are historical and superseded. See [docs/status.md](docs/status.md).
**Date:** 27 October 2026.  
**Design basis:** [PROPOSAL.md](PROPOSAL.md), including independent model runtimes, GPU backends, architecture kernels, and optional card/model–GPU tuning.

This plan turns the architecture into reviewable implementation increments. It targets a server for **Qwen3.8-27B GSQ3 on an original RTX 4080 16 GB**, on Linux and native Windows. Each milestone has dependencies, concrete outputs, and an exit gate. Checkboxes represent unfinished milestone tasks, not verified capabilities. Some named paths now contain partial implementations; [docs/status.md](docs/status.md) records the actual evidence and remaining work. No checkbox is closed merely because a scaffold or metadata entry exists.

**Current sequencing update:** native Windows is the active production target; Linux host development/testing is authorized and host results are recorded in [docs/m1-evidence.md](docs/m1-evidence.md). OrbStack ARM64 host results do not qualify Linux x86_64, CUDA/model runtime, or production performance. Both-OS requirements remain cross-platform gates. A Windows-only result cannot satisfy Linux or matched-reference performance claims. Each instance selects one model via an optional `--model-config FILE` containing one `[model]` INI section; there is no multi-model registry/router/selector in that configuration. See [docs/model-config.md](docs/model-config.md) and [docs/windows-build.md](docs/windows-build.md).

**Section 2 of this plan is authoritative for performance acceptance:** at least 95% of matched NInfer-4080 median committed decode throughput and no more than 110% of matched median uncached TTFT, per frozen case. These are blocking qualification requirements, not predicted or measured results. A feature-complete candidate is distinct from a performance-qualified release.

## 1. Release scope and invariants

The first feature-complete candidate must provide:

- One long-running C++20 server; OpenAI-compatible Chat Completions with streaming and nonstreaming output.
- `qwen3_8_27b` model integration; `qwen3_8_flash_next` names reserved for later, without a registered implementation.
- CUDA backend qualified for RTX 4080; backend-neutral model definitions and state semantics. Reuse compatible SM89 kernel families and compiled variants with `Rtx4080Profile` card tuning. Allow dedicated guarded card/model–GPU kernels when measurements justify a different algorithm; do not require a backend/kernel subclass per SKU or duplicate the server, scheduler, frontend or model runtime per card.
- GPU-resident GSQ3 text body, compatible vocabulary endpoints, native MTP with an off switch, and image input.
- CPU or GPU placement of the complete image encoder and merger/projector; CPU is the initial conservative memory profile.
- GGUF is the required runtime model-input container with a finite supported architecture/encoding profile; this does not imply arbitrary GGUF weight support or generic ggml execution. See [docs/artifact-format.md](docs/artifact-format.md); exact GSQ/Q4 mapping remains an M0/M3 gate.
- Explicit or automatic startup context capacity and required selectable K/V catalog: `f32`, `fp16`, `bf16` (`f16` is an alias for `fp16`), `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`, `iq4_nl`, `int8_g64`, and `int4_g64`. Require all same-codec K/V pairs plus curated mixed pairs `q8_0/q4_0`, `q8_0/q5_0`, `q8_0/q5_1`, and `int8_g64/int4_g64` (K/V order); not the full Cartesian product. These KV numerical contracts are independent of GGUF model-weight support.
- One active generation request with bounded admission, CPU media work, output buffering, cancellation, and deadlines.
- Measured resource accounting and readiness only after validation, allocation, warmup, and selected graph preparation.

Preserve these architectural invariants throughout implementation:

| Invariant | Enforcement |
|---|---|
| Model and GPU are independently registered | Model describes typed operations; backend resolves physical execution |
| Pair tuning is optional | Valid backend defaults/decompositions remain usable; tuning cannot waive support checks |
| Specialization does not fork the engine | Shared architecture variants by default; card tuning selects parameters/variants, while justified dedicated algorithms register as guarded candidates |
| Hot path is specialized | Kernel class descriptor/support/prepare calls occur at startup only; prebound concrete launchers or captured graphs, with no per-token virtual dispatch, registry search or generic graph interpretation |
| State has one execution owner | GPU worker orders mutation, settlement and publication |
| Async buffers survive their consumers | Completion tokens plus leases cover CPU jobs, copies and CUDA execution |
| Quantization is explicit | No silent dtype, codec, scale-policy or weight-representation substitution |
| Context is feasible, not merely configurable | Allocation plan includes main/draft caches and verification reserves |
| Output is committed before publication | No rejected draft or partially settled state reaches a client |

Excluded: Blackwell implementation, additional production GPUs, Flash-Next runtime, general text-weight CPU offload, multi-GPU, distributed serving, hot swapping, interactive CLI/UI, video/audio, tools/structured output/logprobs, prefix retention, DFlash2, and a general graph compiler. Test/benchmark/conversion executables remain development tools. CPU vision support does not imply a CPU text-generation backend.

## 2. Acceptance profiles and performance contract

### Functional profiles

| Profile | Context | Main KV | Speculation | Vision | Purpose |
|---|---|---|---|---|---|
| Reference | Explicit 8K initially | BF16 or FP16, declared per path | Off | Disabled | Independent math/state qualification |
| First combined profile | Explicit 8K | `int8_g64/int8_g64` | MTP up to 3 drafts | CPU | Conservative integration candidate; retain this default while expanding the catalog |
| Catalog coverage | Measured feasible capacity per pair | Every required same-codec and curated mixed pair | Off and MTP | Qualified placement | Required full-catalog feature coverage |
| Memory variants | 16K, 32K, auto candidates | Qualified catalog pairs | Off and MTP | CPU | Establish supported capacity matrix |
| GPU vision variant | Any measured feasible capacity | Qualified pair | Off and MTP | CUDA | Qualify alternate placement |

The capacities are experiments, not fit guarantees. Unsupported or infeasible combinations must return an explicit error. Publish availability by phase, head geometry, placement, capacity and backend; distinguish registered codec metadata, implemented execution, and qualified profiles. Every catalog codec must have a measured feasible, qualified same-codec 4080 profile, and every required mixed pair must have one too. Validate the selected pair for both main and MTP caches, which initially inherit the same selection. A missing required profile blocks the full-catalog feature-complete candidate; it cannot be silently dropped. This does not require every context/placement combination or every possible mixed pair. All required capabilities must also work together in the first combined profile. Reduce an automatic capacity only through the documented planner; do not shrink an explicit request silently.

### Performance qualification

Freeze the mandatory matched comparison suite in M0 before performance tuning, including comparable plain and native-MTP cases. Section 2 governs M8's plain-text checkpoint and M11's release qualification:

1. Use the same physical 4080, OS, driver, power policy, background GPU conditions, tokenizer, rendered prompt tokens and generation limits. Freeze sampling processors, their defaults and application order, RNG/seeds and stop rules—not merely temperature/top-p labels.
2. Match represented text/MTP weights, K/V formats, activation precision and speculative mode. Record **actual dispatch per phase and operation site**: A16 versus A8, intermediate rounding and accumulation. A configured A16 label is insufficient if a phase dispatches A8. Pin a patched A16 reference build and patch hash if necessary to obtain a comparable A16 path; an unresolved mismatch blocks the affected case. Artifact containers and lossless physical packing may differ.
3. Separate plain decode and native MTP. Freeze ngram drafting **OFF** for matched native-MTP cases, draft count, actual target verification width (including bonus/short rounds), and full versus optimized draft-head/vocabulary support. Match any head/vocabulary restriction; a restricted reference head is not silently equivalent to a full-vocabulary engine. DFlash2 headline figures are outside this claim.
4. Freeze graph on/off, prepared graph profiles and prefill/verification chunk sizes. Prefix reuse is explicitly **OFF** for uncached measurements, with empty request state; warmed kernels/graphs do not imply a cached prompt. Separate cold GGUF loading/repacking and startup, retained-prefix experiments and image processing from uncached TTFT; file parsing/repacking must not recur per token or be hidden in request measurements.
5. **BLOCKING, per frozen case:** median committed decode throughput must be **≥ 0.95 × reference** and median uncached TTFT must be **≤ 1.10 × reference**. Both must pass for **every** predeclared comparable plain and native-MTP case at M11. A miss, missing result or unmatched case blocks performance qualification; averages cannot compensate, and the suite cannot be shrunk or cases relabeled post hoc to obtain a pass.
6. Run at least five measured repetitions after warmup for stable short cases; freeze timing boundaries, repetition/noise policy and raw-record schema. Record spread and repeat materially noisy comparisons. Collect enough requests before reporting tail-latency conclusions.

Start with text prompt lengths 128, 2K and a near-8K case that leaves room for generation; add 16K/32K contexts when feasible. Include 512-token controlled generation and naturally stopping prose, code and reasoning requests. State the actual token counts in every result. Benchmark CPU image TTFT separately; image/decode tests still check the complete feature combination.

Report **equal-configuration**, **quality-matched**, and **best-system** comparisons separately. Different codecs, represented weights, activation paths, draft heads or sampling rules cannot satisfy an equal-configuration gate merely by receiving a quality-matched or best-system label. Such additional results do not replace a mandatory frozen case.

If the pinned reference cannot run natively on Windows, explicitly freeze and report a **Linux-only performance-qualification scope**, using the matched Linux suite. Never claim Windows reference parity or call WSL native Windows. Our own quality, memory, lifecycle, API and feature qualification remains required on both Linux and native Windows; own-engine cross-OS measurements do not establish Windows reference parity.

### Quality policy and visible checkpoints

M0 must plan `bench/quality/manifest.json` and `docs/quality-policy.md` as source-controlled deliverables. Before each affected optimization or quantization choice, preregister dataset revisions/hashes, disjoint calibration and held-out evaluation splits, tasks/context lengths, baseline checkpoint/precision, metrics, acceptance budgets, confidence method/level, seeds and sample sizes. Include numeric/distribution error, task quality and long-context retrieval where relevant. Choose budgets from baseline evidence and review; this plan invents neither numerical quality thresholds nor passing results. No unresolved quality budget or missing required result is allowed at the feature gate. Preserve separate kernel-error and quantization-quality evidence.

| Visible checkpoint | Required evidence and status |
|---|---|
| **Correct API text — M5/M6** | Eager text math/state and original-BF16-versus-GSQ quality gates, lifecycle and real-client/SSE tests on both OSes. Not the full feature candidate or a speed claim. |
| **Performance-qualified plain text — M8** | Every frozen comparable plain case passes both Section 2 thresholds, plus eager/graph state and reconciled-memory gates. A failure blocks this checkpoint; integration may continue explicitly **performance-unqualified**. |
| **Full features — M11** | All catalog, native-MTP, CPU/GPU vision, end-to-end quality, memory, lifecycle and both-OS gates produce a **feature-complete candidate**. It is a **performance-qualified release only after** both thresholds pass all frozen comparable plain and native-MTP cases in the declared OS scope. |

## 3. Milestone dependencies

| Milestone | Depends on | Exit deliverable |
|---|---|---|
| M0 — Freeze inputs and baseline | Available test hardware/model sources | Reproducible reference suite and source lock |
| M1 — Portable foundation | Initial M0 compiler/dependency decisions | Both-OS server and representative real-4080 CUDA portability smoke |
| M2 — Contracts and resource planning | M1; M0 component metadata | Fake-backend qualification and complete allocation calculator |
| M3 — Artifact and frontend | M0, M1, M2 schemas | Verified artifact reader/converter and exact prompt tokens |
| M4 — Correct CUDA operators | M2, M3 fixtures | Independently qualified text operators |
| M5 — Eager text engine | M2–M4 | Correct end-to-end prefill/decode |
| M6 — Server lifecycle/API | M1, M5 | Reliable text serving through real clients |
| M7a — Core KV codecs and context | M4, M5 | Two-byte references and custom group-64 pair qualification |
| M7b — Broad KV catalog | M7a | All required same-codec and curated mixed profiles |
| M8 — 4080 kernels and graphs | M5, core M7a; need not wait for M7b | Blocking plain-text performance checkpoint and reconciled graph readiness |
| M9 — Native MTP and replay | M3, M5, M7a; M8 for graph optimization | Correct speculative serving; extend validation across M7b catalog |
| M10 — Images and CPU offload | M3, M6; M9 for combined gate | CPU/GPU vision and image+MTP qualification |
| M11 — Release integration | M6, all M7a/M7b, M8 functional/memory work, M9–M10; performance qualification gated separately | Feature-complete both-OS candidate; performance-qualified release only on full blocking suite pass |

M6 and M7a can progress independently after M5; M8 may start after core M7a while M7b expands the catalog. M11 waits for **all** M7 coverage, including M9 verification on both caches. Vision operator work can begin after M3, but its combined MTP release gate waits for M9. M0 hardware measurements and graph/runtime calibration remain necessary even when model-free work proceeds elsewhere. M8 performance failure permits explicitly unqualified integration, not a qualified checkpoint or release. Do not add calendar estimates until baseline access and reference compatibility are established.

## 4. Detailed implementation milestones

### M0 — Freeze source inputs and reproduce the reference

- [ ] **M0.1** Record GPU SKU, available VRAM, host CPU/RAM, OS, driver, CUDA toolkit, compiler and power/display conditions. Distinguish original 4080 from 4080 Super.
- [ ] **M0.2** Pin PROPOSAL.md's inspected NInfer-4080 revision `2469b5752672ad597055af61fe28e8e7ced20056`; preserve required patches, patch hashes and exact build/run commands. Pin llama.cpp commit `e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf` for the GGML block-32 KV numeric contracts. Separately pin the GGUF specification revision and any actual reader/converter implementation sources used for the runtime profile; the existing llama.cpp KV-source lock is not GGUF-loader evidence. Resolve the supported GSQ/Q4 payload mapping/profile before implementation.
- [ ] **M0.3** Pin GSQ text, the original BF16 text/multimodal quality-baseline checkpoint, original-model MTP/vision, tokenizer and template revisions with checksums. Confirm cross-component geometry, shared vocabulary bindings and compatibility. Define the verified GGUF runtime profile and whether compatible MTP/vision inputs are in the main GGUF or checksummed companion GGUF files; file count/layout remains unresolved. The GSQ source alone omits MTP.
- [ ] **M0.4** Decide whether to adapt attributed reference kernels or independently implement each initial operator. Preserve applicable licenses/notices and record provenance per imported file.
- [ ] **M0.5** Freeze Section 2's mandatory plain/native-MTP case IDs and OS scope, prompts/rendered token IDs, seeds, output/timing rules, sampling processors/defaults/order, exact K/V pairs, phase/site A16/A8 dispatch, rounding/accumulation, graph/chunk settings, prefix reuse OFF, ngram OFF, actual verify widths and full/optimized draft-head/vocabulary support. Verify runtime dispatch rather than trusting flags. Pin a patched A16 reference if needed; record unmatched cases as blockers, not removals. Measure plain and MTP separately.
- [ ] **M0.6** Create independent small tensor/layer/state fixtures; use CPU or a larger reference environment where a full original BF16 model is needed. A reference engine alone is not the sole mathematical oracle.
- [ ] **M0.7** Establish the quality-policy/manifest schemas and original-BF16 baselines. Pin datasets and disjoint calibration/evaluation splits; approve metrics, budgets, confidence and sample-size policies before affected M3/M5 GSQ, M7 KV, M9 component/sampling and M10 multimodal work. Later additions must be preregistered before their affected optimization, never tuned on held-out results.
- [ ] **M0.8** Record source locks for reference engines/patches, llama.cpp KV codec source paths/symbols, the pinned GGUF specification and any reader/converter sources, model/component/tokenizer/template artifacts, dependencies, converter/kernel builds, licenses, dataset manifests and exact commands. Store content hashes and retrieval instructions; distinguish source-inspected contracts from measured results. Do not alter historical retrieval evidence to imply that the current KV lock verified GGUF support.

**Outputs (planned):** `bench/reference/manifest.json`, `bench/quality/manifest.json`, `bench/workloads/`, `docs/source-lock.md`, `docs/quality-policy.md`, reference results and independent fixture provenance. Do not fill unmeasured entries with fabricated results or invented quality thresholds.

**Exit gate:** another developer can reconstruct baseline dispatch and identify every source, weight/template, workload and applicable quality policy. Missing component sources, unresolved affected quality policy or unmatchable reference precision are explicit blockers for the affected comparisons, not implied successes.

**M0 continuation (2026-10-28):** GGUF specification/selected reader sources are pinned and content-verified at `ggml-org/ggml` revision `facdc44a67da17df4433adcc468b424f48e9ded2`; the optional commit API remains rate-limited/unverified. Source evidence/redirect identity and opt-in local artifact hashing are checked; independent signed-int3/FP16 storage and small projection fixtures exist. These are partial M0.2/M0.3/M0.6/M0.8 deliverables, not complete model/profile, layer/state, hardware/reference or original-BF16 quality evidence. GPT6 Luna implementation and independent review/fix/re-review slices were used. All M0 checkboxes remain open where full gates are unmet; see [docs/m0-evidence.md](docs/m0-evidence.md) for the gate-by-gate evidence and required external inputs.

### M1 — Portable foundation and server shell

- [ ] **M1.1** Establish CMake/Ninja presets, C++20, SM89 CUDA compilation and tested Linux/MSVC toolchains. Linux host GCC/Clang presets and Ubuntu 24.04 x86_64 CI are configured; CI has not yet run, and native Windows and CUDA qualification remain outstanding. Enable tests/benchmarks separately from the product build.
- [ ] **M1.2** Create the `apps/server/` entry point, server TOML, one-profile model INI (`--model-config`), startup flags, structured errors and operational logging. Preserve defaults → typed TOML → explicit INI fields → CLI → semantic validation, with no server/security override through INI and no model catalog. Select/pin HTTP and image-decoding dependencies behind narrow adapters.
- [ ] **M1.3** Implement platform RAII for files/mapping, large offsets, environment/argument conversion, shutdown and thread lifecycle. Select shared plus native sources through `src/platform/shared/`, `src/platform/windows/` and enabled `src/platform/linux/`; Linux host selection uses POSIX-backed implementations. Host tests do not qualify Linux CUDA or production runtime. CUDA device/buffer/event/stream owners stay in `src/backends/cuda/`.
- [ ] **M1.4** Add `/healthz`, initially unavailable `/readyz`, clean startup-failure unwinding and orderly shutdown. Do not fabricate model readiness.
- [ ] **M1.5** Establish native-Windows Debug/Release host CI and installed-package/config evidence, with macOS host regression coverage. Linux host GCC/Clang CI for Ubuntu 24.04 x86_64 is now configured (not yet run); local OrbStack ARM64 host evidence is recorded separately. Native Linux/Windows device and production qualification remain outstanding. Establish an explicit real-4080 test procedure; host CI, cross-compilation and WSL do not qualify native-Windows CUDA execution.
- [ ] **M1.6** Compile, device-link and run representative SM89 CUDA portability smoke kernels on a real original 4080 under both Linux and native Windows/MSVC. Exercise `cp.async`, BF16 MMA, kernel launch and result checks; verify occupancy queries and cooperative-launch support/resource bounds if those paths are used. Record toolchain flags, architecture targets and runtime evidence before relying on imported CUDA assumptions.

**Exit gate:** model-free server builds/runs on Linux and native Windows, handles malformed configuration, and shuts down without leaking live worker threads or resources; representative real-4080 CUDA portability smoke passes on both OSes. Host-only CI cannot satisfy this gate.

### M2 — Separate model/backend contracts and resource ownership

- [x] **M2.1 host slice** Define `ModelSpec`, `ModelProgram`, typed tensor/operation/codec contracts, logical weight bindings, numerical permissions and capability descriptors. The implemented structural host contract does not verify artifact payloads or constitute a complete production artifact schema; see [docs/m2-evidence.md](docs/m2-evidence.md).
- [x] **M2.2 host slice** Implement independently registered `IModelDefinition`, `IModelRuntime`, `IGpuBackend`, `IExecutionSession` and optional pair profiles. Add startup-only `IKernelImplementation` with `descriptor()`, `supports(request, caps)` and `prepare(request, resources)` overrides: descriptors name operation/phase, geometry, formats/layouts, numeric contract, architecture requirements and workspace. `prepare` returns a concrete/prebound launcher and resource requirements. Use concrete contract types; avoid an arbitrary graph IR. Synthetic implementations only; no production registration or compiled inference algorithms.
- [x] **M2.3 host slice** Implement inspect → describe program → static candidate → bounded bring-up grant → instantiate → warm/(selected capture) → reconcile → readiness approval. Keep candidate, grant and approval states distinct. Model headers remain CUDA-neutral; kernel-class interfaces resolve only at startup, with no per-token virtual dispatch. Host synthetic composition is implemented; physical fence obligations remain backend responsibilities.
- [x] **M2.4 host slice** Implement completion/status/lease primitives, generation-tagged sequence handles, deferred reclamation and an execution-worker cleanup queue. Reserve transaction cleanup capacity before speculative work begins. **Host foundation implemented/reviewed (2026-10-28):** `saga_runtime` supplies these neutral primitives plus bounded admission and output buffering; see `docs/runtime-references.md`. Physical device fences, backend integration and native Windows qualification remain outstanding.
- [x] **M2.5 host slice** Build the allocation ledger: unique text/draft weights, main/draft KV, FP32 recurrence, rollback/records, activations, distributions/logits, media, ordinary host/pinned memory and runtime/graph overhead. Include alignment, page rounding and lifetime overlap. Static candidates may name unresolved terms and are not approved fit results. The implemented planner evaluates supplied metadata, not actual model inventory or device fit.
- [x] **M2.6 host slice** Implement explicit and automatic context planning. Explicit infeasible settings fail; auto evaluates a bounded list of candidate capacities after selected components are reserved. A bounded bring-up grant requires every enabled engine allocation to be known and capped, external/runtime overhead conservatively reserved, and enforced stop/unwind on exceeding grant/headroom. Unknown enabled engine allocations cannot be hidden in an overhead allowance. Grant only preparation, not serving. Graph-disabled eager profiles reserve no graph resources; graph-enabled candidates retain unresolved graph overhead until M8 calibration, with a conservative enforced capture bound for bring-up and no premature graph-readiness approval. Auto candidates are finite and descending; duplicate entries are rejected rather than silently deduplicated; selection is static eligibility only, with no lower-context retry after eligible kernel/preparation/warmup/capture/reconcile failure.
- [x] **M2.7 host slice** Qualify with a fake backend and synthetic model definitions: incompatible layouts fail, optional profiles cannot bypass checks, defaults work without pair tuning, and backend registration requires no scheduler edits. Add a missing-kernel test that registers/replaces one `IKernelImplementation` without forking a model/backend. A synthetic specialized candidate rejects incompatible device capability, encoding/layout, shape/token width, dtype, alignment/strides, workspace and numerical contract; forced pair preferences do not bypass guards. Test available qualified default/decomposition selection and clear startup failure when none exists. These are integration tests, not a second production model/backend.
- [ ] **M2.8** Compose shared SM89 implementations and compiled tile/pipeline variants with `Rtx4080Profile` card limits/tuning and optional model–GPU preferences. Keep tuning parameters/variant selection in profiles and algorithms in registered kernel implementations. Dedicated card-specific algorithms are allowed as guarded candidates with measured justification; sharing must not impose a slower universal algorithm. Filter support/numeric/resource contracts before ranking and bind the selected nonvirtual launcher at startup. SM89 kernels are reuse candidates for a future 4090, but only 4080 is a qualification target now; no device qualification has been established. Future 5080 architecture/compiler/layout assessment and implementation are outside this release. Do not build mandatory per-SKU inheritance trees or duplicate model/API/scheduler logic.

  **Policy-complete host slice (not M2.8 completion):** device/pair profile constraints and guarded ranking/selection are implemented and targeted-tested; see [docs/m2-evidence.md](docs/m2-evidence.md). Real compiled SM89 algorithms, tile/pipeline variants, measured tuning and physical 4080 qualification remain outstanding.
- [x] **M2.9 host slice** Test candidate/grant/approval transitions, unresolved terms, external-overhead reserve exhaustion, partial-init failures and selected graph-profile failures. Only warmed/(if enabled, captured), measured and reconciled resources within budget/headroom can receive readiness approval; never serve from a bring-up grant. Synthetic host tests cover these transitions; physical device and production qualification remain open.

**Exit gate:** static planning works from metadata without loading the full model and labels unresolved candidates honestly. Bounded preparation is possible without circular dependence on measurements, but readiness approval requires no unresolved enabled terms and reconciled high-water marks. M5 supplies eager reconciliation; M8 resolves selected graph overhead. Fake asynchronous operations prove buffers survive execution and result consumption, including cancellation/shutdown.

**Historical neutral-contract evidence (2026-10-28):** The initial M2.1/M2.2 contract increment and its 18-group macOS Debug/Release/ASan+UBSan/TSan result predate the current M2 authorization and remain valid historical evidence, not the current complete slice. Current planner/startup/profile/step implementation and only targeted verified results are cataloged in `docs/m2-evidence.md`; do not rewrite historical counts as current full-matrix evidence.

### M3 — Artifact conversion, binding and exact frontend

- [ ] **M3.1** Define the finite supported GGUF runtime profile: pinned specification/revision, verified format magic/version, supported architecture/geometry, required/optional tensor mappings, standard payload types, namespaced metadata semantics, components, source identities, shared-parent bindings, checksums, bounded offsets and alignment requirements. Exact GSQ/Q4 mapping is unresolved until verified; do not invent type IDs or claim arbitrary GGUF support.
- [ ] **M3.2** Implement a bounded GGUF reader for the supported profile and offline conversion from the pinned GSQ3/Q4 reference sources. Add independent scalar decoders/fixtures; verify payload code interpretation, group axis, scale bits/rounding, padding and per-head projection ordering. Reject unknown required tensor types and unsupported mappings.
- [ ] **M3.3** Convert compatible MTP and vision components into the supported GGUF profile; select and record their actual encodings after inspecting the sources. Decide whether they reside in the main file or checksummed companion GGUF files. Any value-changing conversion/requantization requires separate preregistered evaluation (M9 for MTP and M10 for vision). Deduplicate shared embeddings/head weights.
- [ ] **M3.4** Implement bounded one-time C++ reading/materialization with format magic/version, overflow, truncation, checksum, bounds and alignment checks, plus atomic GGUF/sidecar output. Do not trust the `.gguf` suffix as format validation. Generate optional content-addressed native backend packing caches; preserve represented values for a lossless repack, never imply a new quantization, and do not parse/repack per token. Account for load/repack overlap, conversion workspace, and steady-state allocations in the ledger.
- [ ] **M3.5** Integrate tokenizer, pinned chat template, special tokens, thinking modes and incremental decoding. Use independently rendered token fixtures.
- [ ] **M3.6** Generate exact component-byte inventories and first candidate 8K/16K/32K/auto allocation reports, marking still-unmeasured execution terms unresolved.
- [ ] **M3.7** Qualify GSQ represented weights against the original BF16 checkpoint on preregistered held-out layer/logit/distribution and task fixtures under M0.7's policy, using a larger/CPU reference environment where needed. Exact decoding/repacking proves representation fidelity, not GSQ-versus-original quality; carry end-to-end validation into M5.

**Exit gate:** a finite GGUF runtime profile and reader accept only verified architecture/encoding/tensor mappings; packed represented values match their declared conversion contract; malformed inputs, unknown required tensor types and unsupported mappings fail safely; frontend token IDs match fixtures; conversion-stage GSQ-versus-original quality evidence meets approved budgets. Missing MTP/vision fails only when that capability is requested (not a waiver of their M11 requirements). Repack/load peak and steady-state allocations are accounted for. No model weights need expansion to full-model BF16 on the 4080.

### M4 — Qualified baseline text operators

- [ ] **M4.1** Implement BF16/FP16 activation paths with explicit accumulation/rounding contracts: norms, Q3/Q4 projections, residuals, SwiGLU, Q/K norms, partial RoPE and vocabulary head.
- [ ] **M4.2** Implement full-attention GQA, causal append and basic paged main KV. Begin with a correct two-byte cache and bounded attention scratch; prepare explicit typed append/read/attention contracts for M7a/M7b without assuming all codecs use custom group-64 scales.
- [ ] **M4.3** Implement GDN causal convolution, gating and one-token FP32 recurrence, including grouped heads and continuation history.
- [ ] **M4.4** Add bounded prefill projection/GDN paths. A simple path may establish correctness first, but chunk-boundary states must be visible and testable.
- [ ] **M4.5** Implement valid-vocabulary masking, greedy selection and documented stochastic sampling order/RNG. Keep model probabilities and token decoding separate.
- [ ] **M4.6** Define operator tolerances before tuning using independent fixtures/oracles; exercise real projection shapes, tails, scales, gates, tiny norms and long recurrent chains.

**Exit gate:** outputs and persistent states pass independent numeric qualification. Run device-memory/race checks where applicable. Generic library GEMM is only a bounded temporary baseline for formats it actually supports.

### M5 — Eager text runtime and worker lifecycle

- [ ] **M5.1** Implement `Qwen38_27BDefinition`/runtime with exact layer pattern and prebound calls into the CUDA session. Batch-shaped interfaces accept only one active sequence initially.
- [ ] **M5.2** Connect prefill, one-token decode, selected/consumed frontiers, RNG, output ledger and full continuation state.
- [ ] **M5.3** Instantiate eager text under M2's bounded bring-up grant, with graph capture disabled and no graph reservation for that profile. Reconcile warmed engine allocations, measured external/runtime overhead and safety headroom before issuing eager readiness approval. Run an 8K candidate with two-byte KV if it fits, otherwise stop/diagnose without silently changing configuration. This does not approve an unresolved graph-enabled profile; M8 must capture/reconcile that separately.
- [ ] **M5.4** Implement bounded prompt chunks, admission, queue/execution deadlines, cancel-at-boundary behavior and completion-aware release/reuse.
- [ ] **M5.5** Compare layer outputs/logits, tokenwise/chunked prefill, one-shot/continuation, reset, cancellation and reused slots. Repeat across several prompt lengths/chunk sizes.
- [ ] **M5.6** Complete end-to-end GSQ-versus-original-BF16 held-out text quality validation (including task metrics and long-context behavior) under the frozen manifest/policy. Independently test runtime error against the same represented GSQ weights; do not substitute kernel agreement or coherent prose for original-model quality.

**Exit gate:** coherent generation, independent state/logit evidence, original-model quality budgets and reconciled eager readiness pass; no per-token allocation, virtual kernel dispatch or registry search. Together with M6 this is the **correct API text checkpoint**, not the complete feature candidate or performance qualification.

### M6 — OpenAI server integration

- [ ] **M6.1** Implement `/v1/models`, `/v1/chat/completions`, `/metrics`, readiness after completed preparation, and supported request/response/error schemas.
- [ ] **M6.2** Implement SSE event identity/order, role/content/final events, requested usage and `[DONE]`; match streamed assembly to nonstreamed output.
- [ ] **M6.3** Implement output token limits, total context checks, stop/EOS/UTF-8 buffering, reasoning parsing and token-ID-based usage.
- [ ] **M6.4** Bound queue/body/output memory; disconnects signal cancellation without socket work blocking the GPU worker. Reject unsupported options and overload before streaming headers where possible.
- [ ] **M6.5** Verify one real OpenAI client plus raw HTTP/SSE tests on both OSes. Document supported fields/extensions and localhost/auth behavior.

**Exit gate / correct API text checkpoint (M5/M6):** request success, error, deadline, slow-reader, disconnect, cancellation and shutdown paths leave valid engine state on both OSes, with M5 quality and eager-memory approval intact. This is not a full-feature or performance-qualified release. Media input remains explicitly unavailable until M10.

### M7a — Core KV codecs and context

- [ ] **M7a.1** Implement PROPOSAL.md Section 6's precise custom `int8_g64`/`int4_g64` scale/code rules, zero handling, nibble order, padding, nonfinite rejection and layout/version identities, alongside independent `fp16`/`bf16` references. Canonicalize `f16` to `fp16`; do not alias BF16. Keep scalar decoders and independent encoder/quantization fixtures separate from optimized kernels.
- [ ] **M7a.2** Bind append/read and prefill/decode attention for required core pairs. Declare support across all enabled phases, including verify; M9 must validate real MTP execution before its readiness/release gate. Main and MTP caches initially inherit the selected K/V pair, but validate their geometry and resource plans separately.
- [ ] **M7a.3** Qualify `bf16/bf16`, `fp16/fp16`, `int8_g64/int8_g64`, `int4_g64/int4_g64` and `int8_g64/int4_g64`; do not infer arbitrary mixed support from individual codecs. Retain the first combined 8K `int8_g64/int8_g64`, MTP3, CPU-vision profile; the expanded catalog does not change that default.
- [ ] **M7a.4** Validate exact/auto capacities, lower request ceilings, output/context boundaries, page tails and capacity exhaustion. Preregister and measure long-context retrieval, logits/distributions and held-out task quality against two-byte KV with the same represented text weights, separating additional KV error from GSQ error. Freeze metrics/budgets, confidence and sample sizes before tuning codecs.
- [ ] **M7a.5** Verify custom-codec allocation fixtures: main 8K `int8_g64/int8_g64` = 264 MiB, `int8_g64/int4_g64` = 200 MiB, `int4_g64/int4_g64` = 136 MiB before paging/alignment, per PROPOSAL.md Section 6. Preserve these values; GGML block metadata uses different formulas.

**Core exit gate:** core pair math, context, quality and memory checks pass; M8 may start without waiting for M7b. Publish support/status explicitly. Quantizing KV does not change FP32 GDN state, text weights or activation permissions. Changing physical profile requires restart. Core completion alone is not full-catalog feature completion.

### M7b — Required llama.cpp-like KV catalog

- [ ] **M7b.1** Add `f32` and pinned GGML block-32 KV codecs `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`, `iq4_nl`, completing the catalog with M7a's `fp16`, `bf16`, `int8_g64`, `int4_g64`. Cross-reference PROPOSAL.md Section 6 and llama.cpp commit `e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf` in descriptor/version metadata and fixtures. These KV formats are numerical cache contracts, not a model-weight container policy or proof of GGUF model-type support.
- [ ] **M7b.2** Implement independent scalar pinned decoders and encoding oracles/fixtures for block layout/size, scale and offset storage, sign/high-bit/nibble ordering, IQ4 lookup, quantization rounding/ties, zeros, saturation, tails/padding and dtype conversions. Reject nonfinite cache inputs before encoding for every catalog codec. Derive each finite-input numerical contract from the pinned source; compare encoder bytes and decoded represented values to the pinned implementation. Do **not** apply custom group-64 symmetric ranges, zero-group scale, scale clamping or nibble rules to GGML codecs. Record any qualified execution rounding separately from storage fidelity.
- [ ] **M7b.3** Implement and qualify each codec's same-codec K/V pair, plus required mixed K/V `q8_0/q4_0`, `q8_0/q5_0`, `q8_0/q5_1` and core `int8_g64/int4_g64`. Do not implement the entire Cartesian product merely because settings are independent. Each required pair must support append, prefill attention, decode attention and enabled verification; qualify tails, causal masks, attention logits and persistent stored values across tokenwise/chunked paths. Extend M7a.4's preregistered two-byte-KV quality comparisons to every required pair.
- [ ] **M7b.4** Publish a machine-readable availability matrix enumerating phase, main/MTP head geometry, placement, capacity and backend, with **registered / implemented / qualified** distinct. Map `--cache-type-k` / `--cache-type-v` and TOML `kv.key_format` / `kv.value_format` to the same canonicalized settings. Test startup validation of both cache pools and all enabled phases. Unsupported/infeasible requests error with the missing combination; no silent pair/precision substitution, context shrink or fallback. Every codec needs a feasible qualified same-codec 4080 profile and every required mixed pair needs one; an absent profile blocks the full-catalog candidate, even if metadata is registered.
- [ ] **M7b.5** Verify main-cache 8K byte fixtures below from pinned per-block sizes, plus mixed `q8_0/q4_0` = 208 MiB, `q8_0/q5_0` = 224 MiB and `q8_0/q5_1` = 232 MiB; count draft cache, page tables, alignment, partial blocks and verification reserve separately. Cross-check planner calculations against actual buffer owners and measured high-water marks.

| Same-codec main K/V at 8,192 tokens | MiB before paging/alignment |
|---|---:|
| `f32/f32` | 1,024 |
| `fp16/fp16` or `bf16/bf16` | 512 |
| `q8_0/q8_0` | 272 |
| `q5_0/q5_0` | 176 |
| `q5_1/q5_1` | 192 |
| `q4_0/q4_0` | 144 |
| `q4_1/q4_1` | 160 |
| `iq4_nl/iq4_nl` | 144 |

**Full M7 exit gate:** every required same-codec and curated mixed pair has independent numeric, held-out quality, memory and feasible-profile evidence on 4080, with both-OS coverage and real enabled MTP verification completed through M9. Publish exact availability, not universal context/placement claims. M11 requires all M7a/M7b work; neither unimplemented registered codecs nor dropped profiles satisfy catalog coverage.

### M8 — 4080 specialization, chunked prefill and CUDA Graphs

- [ ] **M8.1** Add shared SM89 backend Q3 staged GEMV, small-T verification GEMM and tall prefill GEMM as `IKernelImplementation` registrations; benchmark exact model N/K/T values with guarded fallback paths. Startup descriptor/support/prepare resolution binds concrete launchers; no per-SKU inheritance tree or per-token virtual selection.
- [ ] **M8.2** Implement qualified gate/up+SwiGLU and selected projection/norm/residual fusion. Preserve the frozen reference rounding contract; do not conflate the reference projects' different intermediate rounding.
- [ ] **M8.3** Optimize GDN chunked prefill and attention load/dequantization/GQA reuse. Derive cooperative-launch bounds from the actual compiled kernels and device resources; keep a valid fallback.
- [ ] **M8.4** Capture, instantiate and upload plain-decode graphs with stable addresses and context resource tiers for **only the selected startup K/V pair** and enabled profiles, not every catalog codec. Use a bounded capture grant; measure graph preparation/rebuild overhead and warmed/captured peak, reconcile the ledger and safety headroom, then issue graph-enabled readiness approval. Unresolved overhead or bound exhaustion blocks approval; eager approval is not graph approval.
- [ ] **M8.5** Compose common SM89 kernels with `Rtx4080Profile` card tuning and optional `qwen3_8_27b × rtx4080` preferences. Measure tile dimensions, staging/pipeline depth, attention partitions and phase crossover points on the actual 4080; record choices rather than assuming one variant fits every compatible card. Register any justified dedicated algorithm as a guarded candidate, preserving the same model/runtime and numerical contracts. Version tuning keys by architecture/card, kernel build, actual geometry/phase, numeric contract, selected pair and resource profile. Removing optional pair tuning still selects valid qualified defaults; profiles cannot bypass support checks. A future 4090 must remeasure launch/cooperative-residency limits and pass its own correctness, memory, lifecycle and performance qualification despite shared SM89 compatibility. A future 5080 needs architecture-specific compiler/instruction/layout/algorithm assessment and potentially new implementations; neither future GPU is implemented or qualified here.
- [ ] **M8.6** Profile full HTTP requests and GPU work separately; accept only end-to-end wins with correctness/state tests intact.

**Exit gate / performance-qualified plain-text checkpoint:** **BLOCKING** Section 2 thresholds (≥95% median committed decode throughput and ≤110% median uncached TTFT) pass on **every frozen comparable plain case**. Missing/unmatched cases block this checkpoint too. Eager/graph output and persistent-state parity, held-out quality and reconciled memory/readiness must pass. Document any shortfall and measured dominant cost; do not substitute a speculative headline. Work may proceed to feature integration explicitly performance-unqualified, but neither this checkpoint nor M11 release performance qualification may be claimed until their full suites pass.

### M9 — Native MTP, exact acceptance and ReplaySSM

- [ ] **M9.1** Bind MTP stem/block/head sharing/private KV and shifted token/hidden/position alignment, including prompt chunk boundaries. Reserve widths 1–3 and a plain path; record actual verify input width separately. Main and draft caches inherit the selected K/V pair initially; validate both geometries and all append/prefill/decode/verify contracts across M7a/M7b's required profiles, not just the default pair.
- [ ] **M9.2** Implement `DraftBatch`, `VerifyBatch`, distribution accessors and `CommitDecision`. Stochastic rejection requires access to the actual proposal distribution, not only the selected token's probability.
- [ ] **M9.3** Implement greedy verification and distribution-preserving stochastic acceptance/correction with transaction-local RNG, output-budget/EOS/stop clamping and explicit pending correction/bonus tokens.
- [ ] **M9.4** Start with snapshot/accepted-prefix replay as an oracle. Exercise rejection at every position, all accepted, zero visible output, cancellation and async commit/abort failures.
- [ ] **M9.5** Implement raw-input ReplaySSM and convolution-column records. Fold the same finite-precision transition as verify; compare persistent state directly at every accepted length and across long sequences of rounds.
- [ ] **M9.6** Capture selected startup-pair draft/verify/settlement profiles with short-round handling and context-boundary reserves, using M8's bounded capture/reconciliation approval. Do not capture every codec pair in one process. Record actual proposal, target, replay and total round times.
- [ ] **M9.7** Measure fixed widths first. Then add a small cost-based controller that selects only prepared widths/off, excludes first-capture distortion and rechecks stale measurements. Use frozen settings/controller policy for matched cases; ngram stays OFF and draft-head/vocabulary support matches the reference.
- [ ] **M9.8** Qualify any MTP component requantization or optimized head/vocabulary variant against its original BF16 component/full head, separating proposal acceptance/cost changes from target output quality. Preregister held-out metrics, budgets, confidence, seeds and sample sizes before optimizing; update component/source locks and use M10 for end-to-end multimodal effects. An unmatched head variant is supplementary, not a replacement matched-performance case.
- [ ] **M9.9** Add independent small-vocabulary stochastic tests for target/proposal distributions, acceptance and normalized rejection correction. Cover zero proposal probabilities (`q = 0`, including target mass outside proposal support), restricted proposal-vocabulary subsets, masked tokens, all-accepted rounds and rejection at every depth. Preregister statistical tolerance, confidence method/level, seed set, sample sizes and any multiple-comparison rule before observing outcomes. Compare empirical committed-token distributions to the target and RNG/transaction behavior to the independent oracle; a plausible rollout or only selected-token probabilities is insufficient.

**Exit gate:** exact record/restore invariants, all required main/draft codec profiles, component-quality budgets and preregistered statistical sampling checks pass. Compare matched native-MTP configurations against native-MTP reference results under Section 2; report committed tokens/round, acceptance by depth, net tokens/s and peak memory. This milestone does not waive M11's blocking all-case performance gate. Four-input raw recurrent/conv records should match the derived 6.8203125 MiB fixture before other buffers, not be mistaken for total speculative memory.

### M10 — Images and CPU mmproj offload

- [ ] **M10.1** Implement bounded embedded image data-URL validation, decode, model resize/normalize/patch logic, grid metadata and expanded token/position construction. Reject remote URLs/video/audio in v0.
- [ ] **M10.2** Implement the model vision program using CPU reference operators with explicit precision; vectorize hot paths only after qualification. Use bounded workers and ordinary host weight RAM.
- [ ] **M10.3** Implement a CUDA vision path for parity and alternate placement, using the same model equations and backend contracts. GPU placement must pass its own memory plan.
- [ ] **M10.4** Add `vision.device = cpu|cuda`; CPU placement covers the full encoder and merger/projector. Transfer only required projected features through bounded reusable pinned staging.
- [ ] **M10.5** Retain feature/position leases through required prefill/replay; cancellation cannot reclaim in-flight transfer data. Count image-expanded tokens before admission.
- [ ] **M10.6** Compare CPU/GPU features, final logits and image tasks; qualify image+MTP positions/frontiers. Measure preprocessing, encoding, transfer, prefill, TTFT, host/pinned RAM and VRAM independently.
- [ ] **M10.7** Qualify any vision-component requantization and the complete GSQ + selected KV + MTP + vision pipeline against the **original BF16 multimodal model**, not merely our CPU/GPU feature parity. Preregister disjoint datasets/splits, image/text task metrics, budgets, confidence, seeds and sample sizes before affected optimizations. Test CPU and CUDA placement, text-only controls, image-expanded long contexts and image+MTP outputs on both OSes; preserve component and end-to-end quality evidence.

**Exit gate:** the initial 8K `int8_g64/int8_g64`, MTP up to 3, CPU-vision combined profile fits with headroom and passes original-model end-to-end quality, correctness and lifecycle tests on Linux/native Windows; alternate CUDA placement passes its own plan/quality gates. CPU/GPU agreement alone cannot qualify multimodal quality. CPU offload promises placement, not faster image TTFT; report observed latency.

### M11 — Release qualification and packaging

- [ ] **M11.1** Finalize the supported context × K/V pair × MTP width/off × vision placement matrix and precise rejection rules on both OSes. Require feasible qualified 4080 profiles for **every catalog same-codec pair and every required curated mixed pair**, validating main and MTP caches; distinguish registered/implemented/qualified. Cover boundaries without requiring every arbitrary combination or Cartesian mixed pair. Do not drop a required codec/pair to call the catalog complete.
- [ ] **M11.2** Run integrated text/image, cancellation, slow-client, deadline, shutdown/restart and repeated sequence-reuse tests. Publish actual peak memory and stability observations.
- [ ] **M11.3 — BLOCKING PERFORMANCE** Run the full M0-frozen equal-configuration suite against the pinned fork/build/patches. **Both** ≥95% reference median committed decode throughput **and** ≤110% reference median uncached TTFT must pass **every** predeclared comparable plain and native-MTP case. A miss, missing result or unmatched case blocks performance-qualified release; never shrink the suite post hoc or replace it with aggregates, best-system or quality-matched results. If reference native Windows is unavailable, label qualification **Linux only**, with no Windows reference-parity claim; both-OS own-engine gates remain mandatory.
- [ ] **M11.4** Package the server, example configuration, dependency notices, converter instructions, artifact checksums and supported-toolchain matrix for both OSes.
- [ ] **M11.5** Document readiness diagnostics, context auto-resolution, unsupported capability errors, reasoning/usage semantics and recovery behavior after fatal device errors.
- [ ] **M11.6** Reconcile PROPOSAL.md and release documentation claims with the evidence and Section 2's authoritative gates; keep unsupported contexts and unmeasured performance labeled accurately. Publish feature-complete and performance-qualified status separately without weakening thresholds.
- [ ] **M11.7 — BLOCKING FEATURE QUALITY** Close every preregistered budget/result in `bench/quality/manifest.json` and `docs/quality-policy.md`: M3/M5 GSQ versus original BF16, M7 KV versus two-byte, M9 component requantization and stochastic distribution checks, and M10 end-to-end multimodal versus original. Require quality, memory, lifecycle, platform and feature coverage on Linux and native Windows; no unresolved budget or required catalog profile may pass this gate.

**Exit gate / full-feature checkpoint:** all required features/catalog profiles pass quality, memory, lifecycle and platform gates on both OSes and work together in the initial combined 4080 profile, producing a **feature-complete candidate**. A **performance-qualified release** additionally requires M11.3's **BLOCKING** pass of both Section 2 thresholds on all frozen comparable plain/native-MTP cases in the declared OS scope. Misses or missing/unmatched cases block qualification, not merely a parity claim. Continued integration or a feature-complete candidate must remain explicitly performance-unqualified until resolved; no quality/resource gate is waived for speed.

## 5. Decisions to settle early

| Decision | Deadline | Required evidence |
|---|---|---|
| Exact GSQ/MTP/vision inputs and component encodings | M0/M3 | Checksums, geometry, represented-value/quality verification |
| Attributed kernel adaptation versus independent implementation | M0 | Provenance/license record and maintenance cost |
| Numerical tolerances, actual phase/site precision and reference rounding | M0 baseline freeze; before each M4/M8 optimization | Independent fixtures/oracles, dispatch traces, accumulation/rounding and pinned A16 patches if needed |
| Held-out quality budgets and statistical policy | Before affected M3/M5/M7/M9/M10 work; all closed at M11 | Pinned datasets/disjoint splits, original/two-byte baselines, metrics, confidence, seeds and sample sizes in quality manifest/policy |
| HTTP/image dependency and Linux/MSVC toolchain pins | M1 | Both-OS build/smoke results |
| Artifact schema, completion semantics and backend-neutral ABI | M2/M3 | Contract and lifecycle tests; versioning policy |
| Kernel-class and architecture/card composition | M2; measured tuning M8 | Startup-only descriptor/support/prepare, shared SM89 families/compiled variants, measured card tuning and justified guarded dedicated algorithms; no support bypass or per-card engine copies |
| Candidate / bring-up grant / serving approval | M2; eager reconciliation M5, graph reconciliation M8 | Known/capped enabled allocations, enforced overhead bounds, warmed/captured reconciled memory, no unresolved readiness |
| Required codec/placement/context availability matrix | Core M7a, catalog M7b; finalized M11 | Pinned codec contracts, actual memory, held-out quality and feasible profiles for all required pairs |
| Initial combined context, cache and MTP width | Frozen candidate now; qualify M11 | 8K, `int8_g64/int8_g64`, MTP up to 3, CPU vision; complete-profile fit and measured performance, no silent default change |
| Performance suite and qualification OS scope | Freeze M0; plain checkpoint M8, full suite M11 | Both Section 2 thresholds on every frozen comparable case; Linux-only scope if no native-Windows reference |

Do not introduce a dynamic plugin ABI in v0. Independently registered statically linked modules provide the required modularity without DLL compatibility overhead. New operation contracts can extend the backend later without implementing future model/GPU combinations now.

## 6. Working and verification rules

Each implementation increment should name its milestone/task IDs and include the changed behavior, validation and unresolved limitations. Run the relevant existing gates after kernel/math/layout/state changes. Avoid broad benchmark repetition for documentation or low-impact plumbing changes.

Keep three result sets distinct: mathematical/operator correctness; model/KV quantization quality; and system performance. Preserve fixtures and expected tolerances independently of optimized implementations. Pin independent scalar decoders/encoding oracles to PROPOSAL.md Section 6 and the source lock; GGML block-32 and custom group-64 codecs must not share an assumed numeric policy. If an optimization changes numeric boundaries, qualify that as a deliberate execution variant rather than loosening tolerances until it passes. Preregister quality budgets and statistical design before the affected work; do not tune on held-out evaluation or leave required budgets unresolved at feature completion.

Benchmark records must contain commit/build/kernel IDs, model/artifact/template hashes, OS/driver/toolkit, hardware conditions, context/KV/MTP/vision options, timing definitions, memory and completion counts. Include Section 2's actual per-phase/site dispatch and rounding, sampling processor defaults/order, graph/chunk configuration, uncached prefix-reuse OFF, native-MTP ngram OFF, head/vocabulary support and actual verify widths. Preserve raw measurements and derived per-case summaries, including misses/missing/unmatched cases; never remove frozen cases to obtain qualification. Large weights and benchmark dumps do not belong in source control; commit their manifests/checksums and retrieval instructions.

Instrument resource ownership and phase boundaries from the start. GPU allocations, graph overhead and retained mappings must reconcile with the plan. Distinguish unresolved static candidates, enforced bounded preparation grants and final readiness approvals; graph-disabled eager profiles reserve no graph resources, while unresolved selected graph overhead blocks graph readiness. Test failure after partial initialization or grant exhaustion, and ensure the process cannot serve with incomplete, unapproved or corrupted state.

## 7. Historical first increment and current execution scope

The first-increment instructions below are retained as historical sequencing, not
current authorization. The current user authorization covers M1 implementation and
qualification; the M1 source slices are implemented with independent
review/fix/re-review, while native Windows/4080 evidence remains pending. No
M2/M3/M4+ implementation or integration is authorized; see the current-boundary
subsection and `docs/m1-evidence.md`. M0 evidence work and its blockers remain
preserved in `docs/m0-evidence.md`.

The original plan was to begin with **M0.1–M0.4, M0.7–M0.8 and M1.1–M1.6**; freeze M0.5/M0.6 baseline cases/fixtures before affected tuning. Then establish **M2.1–M2.3 and M2.6–M2.9** contract/planning tests before building the full model loop. The first reviewable change should contain:

- Build presets and one server executable that starts without a model and reports not ready (M1.1–M1.4).
- Config/error/logging types, initial model/backend registration and startup-only kernel descriptor/support/prepare schemas, with architecture kernels separate from card/pair tuning (M2.1–M2.3, M2.8).
- Source-lock, artifact-inventory and `bench/quality/manifest.json` / `docs/quality-policy.md` schemas (M0.7–M0.8), including the pinned llama.cpp codec contract and M7a/M7b catalog IDs. Mark unresolved component sizes, budgets and measurements honestly; none may be called qualified.
- Candidate/grant/approval state tests, conservative overhead enforcement, optional-profile support checks and one missing-kernel replacement test (M2.6–M2.9).
- Linux/native-Windows server build evidence and **representative real-4080 CUDA compile/device-link/run smoke** (M1.5–M1.6). Record unavailable hardware tests as blocked, never as host-CI qualification.
- The first reproducible NInfer-4080 baseline and frozen comparable case manifest with actual dispatch/rounding and sampling/MTP/graph settings (M0.2, M0.5–M0.6); missing reference evidence remains a qualification blocker.

Do not start by writing every optimized kernel or importing the entire reference engine. Establish a runnable foundation, a matched performance baseline, and explicit model/backend contracts so subsequent speedups can be measured without sacrificing modularity.

### Current execution boundary and handoff (2026-10-28)

Linux host development/testing is authorized; native Windows remains the active
production target. M1 source slices and evidence limitations are detailed in
`docs/m1-evidence.md`. Linux OrbStack ARM64 host/product checks passed, but M1 exit
remains blocked pending native Windows/MSVC Debug/Release and real original-4080
CUDA evidence on both Linux and Windows. Ubuntu 24.04 x86_64 CI is configured but
not run; no CUDA device compile/link/run or model/reference/performance result exists.

Host-testable M2 implementation is authorized and recorded in `docs/m2-evidence.md`;
this does not authorize M3+ inference integration or close M0/M1/M2 qualification.
M0 remains blocked; see `docs/m0-evidence.md`. Native Windows/MSVC and physical
original-4080 evidence remain pending. See `docs/status.md` and `docs/HANDOFF.md`.
