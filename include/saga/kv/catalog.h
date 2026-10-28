#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace saga::kv {

enum class Codec : std::uint8_t {
    f32, fp16, bf16, q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl, int8_g64, int4_g64
};
enum class NumericalFamily : std::uint8_t { ieee, ggml_linear, ggml_nonlinear, symmetric_g64 };
enum class ScalePolicy : std::uint8_t { none, ggml_fp16, ggml_fp16_scale_min, normal_fp16_clamped };
enum class Packing : std::uint8_t { scalar, canonical_ggml_block, codes_and_separate_scale_plane };

// Metadata only: registration does not claim any append/attention implementation or qualification.
struct CodecDescriptor {
    Codec id;
    std::string_view name;
    std::string_view numerical_version;
    NumericalFamily family;
    ScalePolicy scale_policy;
    Packing packing;
    std::uint32_t block_elements;
    std::uint32_t block_bytes;
    std::uint32_t metadata_bytes;
    // Stored signed codes for Q8/G64; stored unsigned indices for Q4/Q5/IQ4.
    // GGML index offsets/reconstruction are fixed by numerical_version, not by G64 rules.
    std::int32_t minimum_code;
    std::int32_t maximum_code;
    bool rejects_nonfinite;
    bool masks_zero_padded_lanes;
};

// Applies only to the engine-owned G64 formats. This is a numerical contract, not a codec.
struct Group64Numerics {
    float minimum_scale{0x1p-14f};
    float maximum_scale{65504.0f};
    float zero_group_scale{1.0f};
    bool scale_and_codes_nearest_even{true};
    bool quantize_using_stored_scale{true};
    bool int4_lower_channel_in_low_nibble{true};
    bool int4_signed_twos_complement{true};
    bool little_endian_fp16_scales{true};
};
inline constexpr Group64Numerics group64_numerics{};

struct Pair {
    Codec key{Codec::fp16};
    Codec value{Codec::fp16};
    bool operator==(const Pair&) const = default;
};

std::span<const CodecDescriptor> catalog() noexcept;
const CodecDescriptor& descriptor(Codec codec);
Codec parse_codec(std::string_view name); // f16 normalizes to fp16

struct Geometry {
    std::uint64_t layers{16};
    std::uint64_t heads{4};
    std::uint64_t head_channels{256};
};

struct CacheLayout {
    std::uint64_t tokens_per_page{1};
    std::uint64_t plane_alignment{1};
    std::uint64_t table_bytes_per_page{0};
};

struct CacheBytes {
    std::uint64_t key_bytes_per_token;
    std::uint64_t value_bytes_per_token;
    std::uint64_t rounded_capacity_tokens;
    std::uint64_t canonical_block_bytes;
    std::uint64_t layout_padding_bytes;
    std::uint64_t table_bytes;
    std::uint64_t total_bytes;
};

// Blocks restart at every head/token/plane. Partial blocks use zero padded, masked lanes.
// G64's separate scale plane is aligned per page; GGML metadata stays inside each block.
// No draft KV, attention scratch, implementation availability or fit claim is included.
CacheBytes main_cache_bytes(Geometry geometry, Pair codecs, std::uint64_t capacity_tokens,
                           CacheLayout layout = {});

} // namespace saga::kv
