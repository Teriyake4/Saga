# Optional bounded image decoding

The `SAGA_ENABLE_IMAGE_DECODER` CMake option defaults to `OFF`. The text-only product
therefore does not compile or link the decoder or stb_image. When explicitly enabled,
`saga_image_decoder` provides `saga::media::ImageDecoder` through
a narrow header that exposes only spans, limits, and RGB byte vectors; no vendor type
leaks into public headers. This adapter is standalone: it does not connect to a model,
runtime, server endpoint, or vision API.

Only the complete PNG signature and JPEG SOI signatures are accepted. Output is 8-bit
interleaved RGB; 16-bit PNG is rejected, while lower-bit-depth PNG is expanded by
stb_image's 8-bit decoder. Files, remote URLs, HDR, GIF, and all other formats are
unsupported. Defaults cap encoded data
at 16 MiB, width and height at 8192 each, pixels at 32 Mi, returned RGB at 96 MiB, and
decoder working allocations at 128 MiB. Callers may reduce or raise these limits.

The encoded length is checked before decoder calls. stb_image allocations/reallocations
are routed through per-thread, per-call accounting. The budget includes each allocation's
`size_t` bookkeeping header (and therefore its allocator request); it excludes allocator
metadata, fragmentation, stack use, and the caller-owned encoded span. The returned
`std::vector` is separately bounded by the output-byte cap and is not included in the
working-allocation budget. Both budgets are distinct; neither should be interpreted as
a hard process-RSS limit. Decoder-side working-budget exhaustion, unsupported formats,
and malformed/truncated data are reported as `std::runtime_error`; temporary decoder
allocations are released on all exits. Allocation exceptions from the returned
`std::vector` are not translated: `std::bad_alloc` or `std::length_error` may propagate
if output storage cannot be obtained. The output cap constrains its requested element
count, not allocator overhead or capacity over-allocation.

## Test fixture provenance

Both embedded pixel arrays describe solid red samples. `tests/fixtures/red1.png` is a
69-byte 1x1, 8-bit truecolor PNG generated with Python 3.9.6's standard-library
`struct` and `zlib`: filter byte `0`, then RGB `(255, 0, 0)`, zlib-compressed in an
IDAT chunk, with CRC-32 chunk checksums. SHA-256:
`b1ff9c8ea3a780bad09b346c423d2d0e46815926879b18e841d928376a946640`.

`tests/fixtures/red8.ppm` is the 203-byte binary P6 8x8 image whose 64 pixels are
`(255, 0, 0)`; SHA-256:
`50d9620314d2c6ce3b617a6343254fdc174c8cc8ecb13d81f87b6a94f8437f0c`. The 634-byte
JPEG array in `tests/jpeg_fixture.inc` was generated from it with libjpeg-turbo 3.2.0
using `cjpeg -quality 100 -sample 1x1`; SHA-256 of `tests/fixtures/red8.jpg`:
`1c886e18997f57230d91fe97a442ed135f36c3384ff06d36709d0ba2c2187d0b`. PPM/JPEG files
are provenance material only; the C++ test embeds the JPEG bytes and has no encoder or
file-I/O runtime dependency. JPEG checks use a tolerance around the known red source
RGB to allow lossy quantization while detecting channel-order errors.

The vendored dependency is stb_image v2.30, pinned to commit
`f7b2fcb360fc7c611e5b27f6aeb456e40b559210`. The lock records full upstream URLs, byte
sizes and SHA-256 values; `third_party/stb-image/LICENSE` includes MIT and Unlicense
alternatives, and Saga uses the MIT terms. Run `python3 tools/verify_dependencies.py`
to verify the pins. The adapter must remain isolated from text-only targets.
