# M0 actual local artifact inventory

`tools/m0_artifacts.py` creates a reviewable, offline inventory of files explicitly supplied by a user. It uses only Python's standard library. It never downloads, discovers, imports, or executes model artifacts, and it does not rewrite a source manifest or expected hashes. A successful SHA-256 calculation establishes only the bytes currently present at that path. Compatibility and qualification are always reported as **not assessed**.

## Invoking it

Pass a trusted artifact root and either explicit root-relative file names or a JSON manifest. The manifest is an input list, not an expected-hash file; its minimal form is:

```json
{
  "files": [
    {"path": "original-bf16-text/model.safetensors", "category": "weights"},
    {"path": "tokenizer/tokenizer.json", "category": "tokenizer"},
    {"path": "template/chat_template.jinja", "category": "template"},
    {"path": "original-bf16-text/config.json", "category": "config"}
  ]
}
```

Example after placing the actual, independently pinned files under `/data/m0`:

```sh
python3 -B tools/m0_artifacts.py \
  --root /data/m0 \
  --manifest /data/m0/inventory-input.json \
  --output /data/m0/inventory-review.json
```

Alternatively, repeat `--file` for every file. Paths must be canonical and relative to the explicit root; symlink/reparse-point components and root escapes are rejected. On POSIX this uses directory-relative `O_NOFOLLOW` opens. On Windows it uses `CreateFileW` handles opened with `FILE_FLAG_OPEN_REPARSE_POINT`; opened directory handles remain held while descending, and handles deny write/delete sharing during inspection. Platforms without the required secure open mechanism fail closed rather than silently weakening traversal checks. A colon is rejected in paths on every platform to prevent Windows drive/alternate-stream syntax. There is no directory scan. Keep the report separate from inputs, preserve the original manifest and hashes, and review the report alongside the pinned component/revision records. Exit 0 means all supplied files were hashed and supplied safetensors files were structurally valid; 1 means one or more files were unavailable or a safetensors structure was invalid; 2 means invalid invocation/manifest. A report is still emitted for per-file errors when possible.

Safetensors inspection is offline and bounded: it limits the header to 64 MiB and JSON nesting to 128 levels (counting only structural brackets outside escaped strings), checks duplicate JSON keys, tensor count/rank, JSON integer types (booleans are not integers here), supported byte-aligned dtype sizes, shapes, ranges, overlap, truncation, gaps and unclaimed trailing payload bytes. Sub-byte dtypes are rejected because their packing convention is not pinned here. Tensor names are sorted in the JSON report; names, dtypes, shapes, offsets and byte lengths are recorded. Hashing is streamed with no arbitrary cap on weight-file size; streaming limits memory use, not total I/O time. The input manifest is capped at 4 MiB. Hashing and safetensors parsing use one no-follow-opened file descriptor and verify metadata plus a second hash pass, so detected path replacement or in-place mutation fails the record rather than receiving a valid hash/structure status. Other listed files (for example config, tokenizer, and template files) are inventoried by size and SHA-256 only; their formats and meaning are not interpreted.

## Review limits and later checks

This is a local artifact evidence aid, not completion of M0.3/M0.8 and not an artifact compatibility verifier. A structurally valid safetensors file does not show that names, shapes, dtypes, or represented values match a model architecture, component revision, or engine binding. Hashes do not establish provenance unless reviewed against the independently pinned source/revision and expected values. Missing weights remain missing; no expected hash is inferred or filled.

After obtaining actual pinned model/component/tokenizer/template files, inventory each explicit file and independently review the output against the pinned publisher/revision and source lock. Still-required checks include binding tensor names/shapes/dtypes to the selected model/component geometry, matching embedding/output projection dimensions to tokenizer vocabulary and special-token IDs, checking tokenizer/config/template semantics and rendered token IDs, and verifying converted/quantized values against the original representation. These require authoritative model metadata and independent fixtures or execution; they are not performed here. Byte hashes, even when matched, cannot establish compatibility, quality, runtime support, or qualification.

This is conceptually complementary to `tools/verify_sources.py` and the local artifact policy in `docs/m0-evidence.md`: both avoid downloads and execution and distinguish byte evidence from qualification. This inventory tool does not modify or replace the global source manifests, verifier, GGUF runtime policy, or production sources.

## Pinned-source tensor contract and bounded next slice

### Evidence and provenance status

This is a source-derived contract, not local weight verification. The original review checkout contains the documented frontend cache `.cache/m0-frontend/{original,gsq}/` and pinned reference cache `.cache/m0-reference/ninfer-4080-2469b5752672ad597055af61fe28e8e7ced20056/`; neither was copied into this issue worktree or supplied as an operator weight root. `docs/m0-frontend.md` records that only frontend JSON/template inputs were acquired, not weights.

On 2026-11-06, the parent recomputed and matched the archive SHA-256 and canonical source-tree digest against the existing receipt, and rechecked that the current cached files match 10 frontend-lock entries, 22 locked reference-source entries, and 8 preserved license files identified by `local_path` in `bench/reference/source-lock.json`. The existing [`frontend-lock.json`](../bench/reference/frontend-lock.json), [`source-lock.json`](../bench/reference/source-lock.json), and [`reference-source-acquisition.json`](../bench/reference/reference-source-acquisition.json) remain the evidence records. That acquisition record lists archive SHA-256 `4dbdc09000e0358df70a4dbdd824edc1cb853fa48634a4877554d7f1841379ba` and canonical source-tree SHA-256 `35b71e27437fa63c201fbdb387b6bc81199ea02f229cc22193bae9e05be1b804`. This is a cache recheck, not new retrieval, build, conversion, or model execution. Matching preserved license-file hashes do not establish a license audit or permission for absent component files; Switchyard remains license-unresolved and is not reused.

Evidence classes: frontend config/source bytes are lock-matched; converter and target behavior below is source-derived; a selected experiment profile remains proposed; original/GSQ weight payloads, physical aliases, GGUF files/mapping, and artifact-specific permissions remain unavailable or unresolved. Hashes for these configs and reference-source files establish the recorded files, not the existence or identity of model weight payloads.

Pinned converter sources at [`yalding/ninfer-4080@2469b5752672ad597055af61fe28e8e7ced20056`](https://github.com/yalding/ninfer-4080/tree/2469b5752672ad597055af61fe28e8e7ced20056): [`recipe_gsq3.py`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/tools/convert/qwen3_8_27b/recipe_gsq3.py), [`gsq3_source.py`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/tools/convert/qwen3_8_27b/gsq3_source.py), [`convert_gsq3.py`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/tools/convert/qwen3_8_27b/convert_gsq3.py), [`verify_gsq3.py`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/tools/convert/qwen3_8_27b/verify_gsq3.py), and family [`qwen3_6_27b/recipe.py`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/tools/convert/qwen3_6_27b/recipe.py). Pinned target references: [`config.h`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/targets/qwen3_6_27b/impl/config.h), [`bindings.cpp`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/targets/qwen3_6_27b/impl/load/bindings.cpp), [`storage_layouts.cpp`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/artifact/storage_layouts.cpp), and [`text_context_impl.h`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/targets/qwen3_6/impl/runtime/text_context_impl.h). These links identify inspected source, not an endorsed runtime dependency.

### Geometry and block tensors

Config geometry agrees with the pinned 27B target geometry (`config.h:13-42,51-69`; original and GSQ cached `config.json`): hidden width D=5120, MLP intermediate I=17408, 64 layers; full attention at 3, 7, …, 63 (16 layers), GDN/linear attention in the other 48. Full attention has Q=24 heads, KV=4 heads, head dimension 256. GDN has Q/K=16 heads and V=48 heads, each dimension 128; value/key-head ratio is 3. These are geometry declarations, not proof of payload availability.

Selected source-to-internal mappings below are defined by the pinned `recipe_gsq3.py` and family recipe; HF names and internal NInfer package names are deliberately distinguished. All quantized source matrix shapes are logical `[N,K]`.

| Block / role | HF source checkpoint tensor(s) | Source logical shape | Internal package tensor / expected shape |
|---|---|---:|---|
| Layer 3 attention Q and output gate | `model.language_model.layers.3.self_attn.q_proj.weight` | `[12288,5120]` | Head-wise row split: query half and gate half; see interleave below |
| Attention query/K | `...self_attn.k_proj.weight` | `[1024,5120]` | `text/layers/3/attention/query_key` `[7168,5120]` |
| Attention gate/V | `...self_attn.v_proj.weight` | `[1024,5120]` | `text/layers/3/attention/gate_value` `[7168,5120]` |
| Attention output | `...self_attn.o_proj.weight` | `[5120,6144]` | `text/layers/3/attention/output` `[5120,6144]` |
| Layer 0 GDN Q/K/V input | `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` | `[10240,5120]` | `query_key` `[4096,5120]`; first 4096 rows. Remaining 6144 rows feed `value_z` |
| GDN Z input | `...linear_attn.in_proj_z.weight` | `[6144,5120]` | `text/layers/0/gdn/value_z` `[12288,5120]`; concatenate after QKV rows 4096:10240 |
| GDN output | `...linear_attn.out_proj.weight` | `[5120,6144]` | `text/layers/0/gdn/output` `[5120,6144]` |
| Optional MLP gate/up | `...mlp.gate_proj.weight`, then `...mlp.up_proj.weight` | each `[17408,5120]` | `mlp/gate_up` `[34816,5120]`, gate then up |
| Optional MLP down | `...mlp.down_proj.weight` | `[5120,17408]` | `mlp/down` `[5120,17408]` |
| Endpoints, not needed for subblock diagnostic | `model.language_model.embed_tokens.weight`; `lm_head.weight` | each `[248320,5120]` | distinct logical `text/token_embedding` and `text/output_head`; physical sharing unknown |

Q projection row interleave is 24 consecutive 512-row head blocks; within each head, `[0,256)` is query and `[256,512)` is gate. The converter selects each half head-by-head in head order, then concatenates query before K, and gate before V (`recipe_gsq3.py:_qproj_rows,_attention_indices,_build_text_core_recipes`; verifier-local inverse `verify_gsq3.py:_qproj_part,packed_object_plan`). The reference does not thereby prove the HF source payload bytes or a GGUF mapping.

The converter's source prefix is `model.language_model.layers.{i}.`. The quantized tensors use physical safetensors entries formed by replacing terminal `.weight` with suffixes—not appending after it. For example: `model.language_model.layers.3.self_attn.q_proj.weight_packed`, `.weight_scale`, and `.weight_shape`; corresponding source prefix is `...self_attn.q_proj`. The three entries are I32 `[N,K*bits/32]`, BF16 `[N,K/group_size]`, and I64 `[2]` logical `[N,K]`, respectively (`gsq3_source.py:read_metadata`).

### Packed representation, scale conversion, direct tensors

GSQ config describes static symmetric groupwise integer weights: body 3-bit/group 128 for `linear_attn`, `self_attn`, `mlp`; endpoints 4-bit/group 64 for embedding and `lm_head`. The converter expects dense LSB-first packed codes and BF16 per-group multipliers. Its adapter's representation rule is an **expectation derived from pinned source code**, not independent publisher-byte proof: unsigned offset-binary `u` represents `(u-2^(bits-1))*scale`; conversion XORs the sign bit of each code to yield signed two's-complement row-split codes (3-bit mask `24 49 92` hex, Q4 mask `88` hex). Codes are selected/concatenated without decode or requantization (`gsq3_source.py:1-25,42-51,233-254`; `recipe_gsq3.py:206-230,392-427`).

The NInfer target format is `Q3G128_F16S` or `Q4G64_F16S`; it is not a standard GGUF type assertion. The converter changes source BF16 scales via FP32 intermediate to FP16. It requires finite values, magnitude <=65504, error <=2^-25; rounded words must remain FP16 subnormal. Normal-range accepted values are exact; accepted subnormal rounding or underflow to zero can change represented values (`gsq3_source.py:256-315`). Actual per-tensor rounded counts/error await actual source shards and verifier execution. Reference `row_split_geometry`/`storage_layouts.cpp:148-170` derives K padding to 128 and plane alignment 256 for its row-split encoding. That is a source-derived reference contract only; actual artifact offsets, lengths, padding and aliases are unknown. Final binding still requires physical parent hash, byte range, encoding, axes, scale location/dtype/axis, padding and layout.

Selected direct tensors, separate from packed triples:

| Role | HF source tensor | Source shape | Reference/internal expectation |
|---|---|---:|---|
| Layer 3 attention input norm | `model.language_model.layers.3.input_layernorm.weight` | `[5120]` | BF16 reference tensor; source dtype to verify |
| Layer 3 Q/K norms | `...self_attn.q_norm.weight`; `...self_attn.k_norm.weight` | each `[256]` | BF16 reference tensors; source dtype to verify |
| Layer 0 input norm | `model.language_model.layers.0.input_layernorm.weight` | `[5120]` | BF16 reference tensor; source dtype to verify |
| Layer 0 GDN control | `...linear_attn.A_log`; `...linear_attn.dt_bias` | each `[48]` | FP32 reference tensors; source dtype to verify |
| Layer 0 control projections | `...linear_attn.in_proj_a.weight`; `...linear_attn.in_proj_b.weight` | each `[48,5120]` | BF16 reference tensors; source dtype to verify |
| Layer 0 convolution | `...linear_attn.conv1d.weight` | `[10240,1,4]` | slice singleton axis, reshape `[10240,4]`, transpose to `[4,10240]`; BF16 reference tensor |
| Layer 0 GDN output norm | `...linear_attn.norm.weight` | `[128]` | BF16 reference tensor; source dtype to verify |
| Outside first subblock | `...post_attention_layernorm.weight`; `model.language_model.norm.weight` | `[5120]` each | post-attention and final norm; separate later integration |

Direct name/shape transforms are in pinned `qwen3_6_27b/recipe.py:60-159`; generic materialization operations (slice/reshape/transpose/concat/cast) are in `qwen3_6/common/recipe.py:110-160,330-380`. Converter recipes pass norm weights through without adding one. Reference RMSNorm API documents `gain = unit_offset ? 1+weight : weight` (`include/ninfer/ops/rmsnorm.h:9-23`); ordinary text attention/input/post/final calls pass `true` (`text_context_impl.h:825-844,1005-1008,1206-1209`). GDN control projection kernel applies `1+norm_weight` (`bf16_gdn_norm_gating_proj_27.cu:14-38`), while GDN output uses `gated_rmsnorm` with non-unit-offset path and z gate (`text_context_impl.h:982-984`; `ops/wrapper/rmsnorm.cpp:91-98`). These are reference operator facts, not final model-semantic authority.

### Unresolved semantic and component contracts

Both cached configs say `output_gate_type="swish"`; the pinned text path calls `sigmoid_mul`, whose public contract is `x*sigmoid(gate)` (`include/ninfer/ops/sigmoid_mul.h:9-20`; `text_context_impl.h:876`). The key is not consumed in the inspected pinned target/converter path. Preserve the discrepancy; do not choose semantics from the label. Both configs set `max_position_embeddings=262144`; reference `config.h:104` sets `kNativeContext=1048576`. Preserve that discrepancy; a bounded context-8 diagnostic makes no global-context claim. Config vocab is 248320 while the locked tokenizer evidence reports vocab 248077. Do not describe 248320 as a verified padded artifact binding. Config `tie_word_embeddings=false` and distinct logical endpoint recipe objects do not resolve physical sharing.

The reference converter is an NInfer package converter, **not a GGUF converter** and not a production runtime path. It intends to source GSQ quantized text body/endpoints and vision candidate tensors from the GSQ root, and MTP direct tensors from the official root: converter selects `official_reader` only when direct source names are all `mtp.*`, otherwise `gsq_reader` (`convert_gsq3.py:214-225,283-320`; recipe inventory `recipe_gsq3.py:326-338,464-473`). GSQ ignore patterns mean unquantized, not automatically omitted. This intent is not payload availability, conversion success, or proof of equivalence. In particular, the GSQ-root vision candidate is not the original-root vision quality oracle; original vision bytes are still missing. Whether MTP/vision are parent or companion GGUF files remains unresolved under #11. No GGUF mappings/type IDs are asserted.

### Minimal input handoff and resource boundary

Smallest handoff is an operator-authorized root and exact file list; then source/index and relevant safetensors shard headers, tensor names/shapes/dtypes/byte offsets/lengths, provenance hashes and license/notice. Only then request the selected layer-3 and layer-0 quantized triples plus direct tensors from the exact shards (or provenance-preserving subset), including physical parent offsets/ranges. Header reports are not payload verification. No full-model weights are needed for this block diagnostic; original-BF16 block tensors are optional for a quantization differential. Full original baseline, MTP and vision remain release gates. The source shard/header absence is an external artifact blocker, not a tooling project.

Logical packed code+scale estimates only: body `N*K*(3/8+2/128)` bytes; Q4 `N*K*(4/8+2/64)` bytes. Layer-3 attention packed projections/output: 40,960,000 B; layer-0 GDN packed input/output: 45,056,000 B; optional MLP layer: 104,448,000 B; each endpoint: 675,430,400 B. Exclude headers, alignment, direct tensors, file/resource metadata, staging, conversion temporaries, native packed/cache copies, and allocator/runtime overhead. These are not artifact extents or fit evidence.

State estimates follow the pinned [linear-attention state payload](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/core/linear_attention_state.cpp#L95-L109) and [target state configuration](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/targets/qwen3_6/impl/runtime/layouts_impl.h#L131-L142): recurrent FP32 `128*128*48` = 3 MiB per GDN layer/slot; convolution BF16 `10240*3` = 60 KiB per GDN layer/slot; all 48 layers total 146.8125 MiB/slot before extra snapshots/rollback. Full-attention BF16 KV is 4096 B/token/layer; only 16 layers use it. The pinned reference GDN chunk length is 64 ([`common.h:7-8`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/ops/linear_attention/gated_delta_net/common.h#L7-L8)); top-level prefill alignment is 128 ([`layouts_impl.h:671-676`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/targets/qwen3_6/impl/runtime/layouts_impl.h#L671-L676), [`config.h:99-104`](https://github.com/yalding/ninfer-4080/blob/2469b5752672ad597055af61fe28e8e7ced20056/src/targets/qwen3_6_27b/impl/config.h#L99-L104)); a tiny path does not cover chunk workspace.

Before any experiment inventory/cap each domain: mapped and resident source bytes; load/repack source+destination overlap; conversion temporaries; native packing/cache retention; pageable and pinned host buffers; device weights; per-call activations, staging and selected-kernel workspace lifetimes; runtime overhead. Separately count sequence slots, snapshots/rollback, positions, token frontiers, RNG and delayed reclamation. No numeric caps are selected here, so this document does not authorize a run. Reuse the existing memory planner/ledger; model runtime owns logical equations/state transitions, backend/session owns physical execution/lifetime—no duplicate semantics.

A future separately approved one-subblock diagnostic may use B=1, context 8, four-token prefill then two single-token decode steps, BF16 activations/KV, FP32 recurrence/control, no MLP/head/graphs/MTP/vision. It requires verified source bindings and scale audit, resolution of gate/norm/RoPE/GDN-control semantics, an independent oracle with reviewed tolerances, a small BF16 input fixture plus identical initial state, and both final-output **and persistent-state comparisons** against a sequential independent reference: attention KV, positions/frontier; GDN recurrence and convolution state (plus positions/frontier). Also require commit/abort/retry/reset and physical-lifetime/cancellation tests; cancellation is not a fence. T<=6 is operator scope only, not the top-level 128-token launch; there is no existing real-block command claimed. Serving still requires #11 verified GGUF profile/reader and offline packaging; safetensors is conversion input, not runtime format.
