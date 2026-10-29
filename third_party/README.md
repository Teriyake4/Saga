# Pinned host dependencies

Vendored single headers keep configuration and builds offline. JSON and TOML
headers are unmodified; cpp-httplib has the local hardening patch described below.
Each file's retrieval URL, version, installed byte count and SHA-256 is in
`lock.json`. Patched files additionally record the pristine upstream identity and
referenced, independently hashed patch files. Run
`python3 tools/verify_dependencies.py` before accepting an update.

| Dependency | Version | License | Purpose |
|---|---|---|---|
| cpp-httplib | v0.28.0 + Saga hardening patch | MIT (`cpp-httplib/LICENSE`) | Cross-platform HTTP transport behind `saga::api::Server` |
| nlohmann/json | v3.11.3 | MIT (`json/LICENSE`) | Structured protocol errors and operational logs |
| toml++ | v3.4.0 | MIT (`tomlplusplus/LICENSE`) | Strict TOML configuration |
| stb_image | v2.30 at `f7b2fcb360fc7c611e5b27f6aeb456e40b559210` | MIT or Unlicense (`stb-image/LICENSE`; Saga selects MIT) | Optional PNG/JPEG decoder adapter |

The HTTP pin was advanced from the initially considered v0.18.3 to v0.28.0
before build integration because it supports an explicit header-count bound.
This is a verified version pin, not a claim that it is the latest release or
free of vulnerabilities. TLS/compression integrations are disabled; remote
exposure requires a trusted TLS reverse proxy with header/body limits and
absolute request deadlines. Per-I/O idle timeouts alone do not defeat a client
that continually trickles data.

## Local cpp-httplib patch

`patches/cpp-httplib-v0.28.0-saga-hardening.patch` is a unified diff against the
pristine v0.28.0 header at the URL in `lock.json`. The original copyright and MIT
license are preserved. The patch:

- Preserves header values as HTTP field bytes instead of URI-decoding them.
- Parses complete hexadecimal chunk sizes with checked `size_t` arithmetic,
  validates chunk extensions, and requires exact CRLF delimiters/termination.
- Rejects malformed trailers and counts every trailer against the configured
  bound, including undeclared or prohibited fields that are ignored.
- Services task-queue cancellation on accept retry paths, including descriptor
  exhaustion, without concurrent listener-socket closure. Windows retries use
  native Winsock errors rather than assuming the CRT `errno` was updated.

To reproduce, retrieve the original header into a scratch checkout layout,
verify its `upstream_sha256`/`upstream_bytes`, apply the recorded patch with
`patch -p1` from that scratch root, and compare the result with the installed
`sha256`/`bytes`. Applying the patch to the already-patched vendored file is not
an update procedure. Do not silently replace upstream hashes with patched hashes.

`tests/http_transport_test.cpp` tests the raw parser, chunk/trailer boundaries,
and deterministic accept-retry cancellation. The server adapter separately rejects
ambiguous framing before body reads/Expect handling, restricts bearer-token syntax,
and uses one request per connection. Neither the patch nor these tests establish
full HTTP conformance or production remote-exposure qualification.

Builds do not download dependencies or execute remote configuration scripts.
No inference-engine source code has been copied into these dependencies.
The optional image decoder is isolated in `saga_image_decoder`, disabled by default,
and is not part of the text-only server. Its exact upstream commit URL, file sizes,
and SHA-256 pins are in `lock.json`; use the MIT alternative in the included
stb_image dual-license text. No inference-engine source code has been copied into
these dependencies.
