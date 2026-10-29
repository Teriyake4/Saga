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
