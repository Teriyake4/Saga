# Runtime artifact format policy

## Decision

GGUF is Saga's required **runtime model-input container**. This decision does not mean Saga supports arbitrary GGUF models or tensor encodings, runs a generic ggml interpreter, or produces files universally consumable by llama.cpp. Runtime support is a finite, explicitly identified architecture/encoding/profile with checked tensor mappings and backend bindings. Unknown architectures, required tensor types, encodings, or unsupported mappings must fail before readiness.

The current target profile is Qwen3.8-27B with GSQ3 body weights and Q4 embedding/head weights. The exact supported GGUF profile and GSQ/Q4 payload mapping are unresolved M0/M3 gates; there is no GGUF reader, writer, converter, or model support implemented yet.

## Representation and compatibility

- GGUF is the container. It does not define the numerical equivalence of every payload a Saga profile may support.
- Saga GSQ3 is not GGUF `Q3_K`, `Q3_K_M`, `IQ3`, or raw three-bit packing. The actual represented weights—including code interpretation and scale dtype/rounding—must match for a matched comparison. Similar names, bit widths, or metadata labels do not establish equivalence.
- Common GGUF `Q3_K`/`IQ3` availability is not promised merely because GGUF is the container. Exact GSQ/Q4 data may require an explicit profile using namespaced metadata and verified standard typed code/scale tensors. Verify that mapping against the pinned specification and actual reader; do not invent tensor-type IDs, mislabel payloads as a common ggml type, or claim universal llama.cpp interoperability.
- Use standard architecture and tensor metadata where compatible. Reject unknown required tensor types and unsupported mappings; do not silently substitute a type or encoding.
- Safetensors and reference checkpoint formats are **offline conversion inputs only**, never alternative runtime formats.

## Components and sidecars

The target includes compatible text, MTP, and vision/projector components under the same validated profile. Each component must have checksums, explicit identities, required/optional status, and logical/shared-tensor bindings. It is unresolved whether MTP and vision reside in the main GGUF or one or more companion GGUF files; do not assume a file count or layout before M0/M3 defines and verifies it. Missing required components fail when their capability is requested.

Tokenizer/template/configuration data and manifest/checksum sidecars are allowed. They are data only; artifacts must never execute embedded code.

## Loading, packing, and memory

GGUF remains the runtime input regardless of backend layout. A backend may perform a one-time lossless repack to its selected native layout, and may reuse a content-addressed native packing cache. Cache identity and validation must cover artifact/component hashes, profile and encoding (including numerical conversion/rounding), packing schema/version, and relevant backend/architecture requirements. A cache hit is still subject to the selected kernel's support and compatibility guards; cached data cannot override them. This is policy only, not a defined or implemented cache schema. It does not require an on-disk GPU packing format. Do not parse or repack model files per token. A lossless repack preserves the declared represented values; any scale conversion or other value-changing operation must be identified and separately evaluated.

The allocation ledger must cover both peak load/repack overlap and steady state. Include mapped/staged GGUF bytes, source-format buffers, conversion workspace, native GPU packing, and any retained cache materialization, with host, pinned-host, and device memory reported separately. Avoid retaining duplicate native layouts unless their bytes and lifetimes are explicitly budgeted. No full-model BF16 expansion is required. Measure cold artifact loading/repacking separately from warm startup and request latency; uncached generation must not hide repeated parsing or packing.

The performance gate remains the matched, per-case requirement: at least 95% of reference median committed decode throughput and no more than 110% of reference median uncached TTFT. A GGUF container or a lossless physical repack does not establish parity.

## Verification and source pinning

Before implementation, M0/M3 must pin the GGUF specification revision and any reader/converter source actually used, then verify the exact profile and standard payload mapping with independent fixtures. The current llama.cpp source lock covers selected GGML **KV-cache numerical contracts**; it is not evidence of a GGUF model loader or GSQ interoperability.

Design reference (currently unpinned; not retrieved or verified in this change): [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). The presence of standard integer/float tensor types in that specification alone does not define or qualify a Saga GSQ profile.
