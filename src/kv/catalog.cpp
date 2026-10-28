#include "saga/kv/catalog.h"
#include "saga/memory/checked.h"

#include <array>
#include <stdexcept>
#include <string>

namespace saga::kv {
namespace {
constexpr std::string_view ggml_version = "llama.cpp/e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf";
constexpr std::array entries{
    CodecDescriptor{Codec::f32, "f32", "ieee754-binary32", NumericalFamily::ieee, ScalePolicy::none, Packing::scalar, 1, 4, 0, 0, 0, true, true},
    CodecDescriptor{Codec::fp16, "fp16", "ieee754-binary16", NumericalFamily::ieee, ScalePolicy::none, Packing::scalar, 1, 2, 0, 0, 0, true, true},
    CodecDescriptor{Codec::bf16, "bf16", "bfloat16-v1", NumericalFamily::ieee, ScalePolicy::none, Packing::scalar, 1, 2, 0, 0, 0, true, true},
    CodecDescriptor{Codec::q8_0, "q8_0", ggml_version, NumericalFamily::ggml_linear, ScalePolicy::ggml_fp16, Packing::canonical_ggml_block, 32, 34, 2, -128, 127, true, true},
    CodecDescriptor{Codec::q5_0, "q5_0", ggml_version, NumericalFamily::ggml_linear, ScalePolicy::ggml_fp16, Packing::canonical_ggml_block, 32, 22, 2, 0, 31, true, true},
    CodecDescriptor{Codec::q5_1, "q5_1", ggml_version, NumericalFamily::ggml_linear, ScalePolicy::ggml_fp16_scale_min, Packing::canonical_ggml_block, 32, 24, 4, 0, 31, true, true},
    CodecDescriptor{Codec::q4_0, "q4_0", ggml_version, NumericalFamily::ggml_linear, ScalePolicy::ggml_fp16, Packing::canonical_ggml_block, 32, 18, 2, 0, 15, true, true},
    CodecDescriptor{Codec::q4_1, "q4_1", ggml_version, NumericalFamily::ggml_linear, ScalePolicy::ggml_fp16_scale_min, Packing::canonical_ggml_block, 32, 20, 4, 0, 15, true, true},
    CodecDescriptor{Codec::iq4_nl, "iq4_nl", ggml_version, NumericalFamily::ggml_nonlinear, ScalePolicy::ggml_fp16, Packing::canonical_ggml_block, 32, 18, 2, 0, 15, true, true},
    CodecDescriptor{Codec::int8_g64, "int8_g64", "saga-g64-rne-normal-fp16-v1", NumericalFamily::symmetric_g64, ScalePolicy::normal_fp16_clamped, Packing::codes_and_separate_scale_plane, 64, 66, 2, -127, 127, true, true},
    CodecDescriptor{Codec::int4_g64, "int4_g64", "saga-g64-rne-normal-fp16-v1", NumericalFamily::symmetric_g64, ScalePolicy::normal_fp16_clamped, Packing::codes_and_separate_scale_plane, 64, 34, 2, -7, 7, true, true}
};

std::uint64_t page_plane_bytes(const CodecDescriptor& codec, std::uint64_t blocks,
                              std::uint64_t alignment) {
    using namespace memory;
    if (codec.packing == Packing::codes_and_separate_scale_plane) {
        return checked_add(round_up(checked_multiply(blocks, codec.block_bytes - codec.metadata_bytes), alignment),
                           round_up(checked_multiply(blocks, codec.metadata_bytes), alignment));
    }
    return round_up(checked_multiply(blocks, codec.block_bytes), alignment);
}
} // namespace

std::span<const CodecDescriptor> catalog() noexcept { return entries; }

const CodecDescriptor& descriptor(Codec codec) {
    for (const auto& entry : entries) if (entry.id == codec) return entry;
    throw std::invalid_argument("unknown KV codec ID");
}

Codec parse_codec(std::string_view name) {
    if (name == "f16") name = "fp16";
    for (const auto& entry : entries) if (entry.name == name) return entry.id;
    throw std::invalid_argument("unknown KV codec: " + std::string(name));
}

CacheBytes main_cache_bytes(Geometry geometry, Pair codecs, std::uint64_t capacity_tokens,
                           CacheLayout layout) {
    using namespace memory;
    if (geometry.layers == 0 || geometry.heads == 0 || geometry.head_channels == 0 ||
        capacity_tokens == 0 || layout.tokens_per_page == 0)
        throw std::invalid_argument("KV geometry, capacity and page tokens must be positive");
    if (!is_power_of_two(layout.plane_alignment))
        throw std::invalid_argument("KV plane alignment must be a power of two");
    const auto& key = descriptor(codecs.key);
    const auto& value = descriptor(codecs.value);
    const auto head_count = checked_multiply(geometry.layers, geometry.heads);
    const auto key_blocks = checked_multiply(head_count, ceil_divide(geometry.head_channels, key.block_elements));
    const auto value_blocks = checked_multiply(head_count, ceil_divide(geometry.head_channels, value.block_elements));
    const auto key_bytes = checked_multiply(key_blocks, key.block_bytes);
    const auto value_bytes = checked_multiply(value_blocks, value.block_bytes);
    const auto pages = ceil_divide(capacity_tokens, layout.tokens_per_page);
    const auto rounded_tokens = checked_multiply(pages, layout.tokens_per_page);
    const auto canonical = checked_multiply(rounded_tokens, checked_add(key_bytes, value_bytes));
    const auto per_page = checked_add(
        page_plane_bytes(key, checked_multiply(key_blocks, layout.tokens_per_page), layout.plane_alignment),
        page_plane_bytes(value, checked_multiply(value_blocks, layout.tokens_per_page), layout.plane_alignment));
    const auto storage = checked_multiply(pages, per_page);
    const auto tables = checked_multiply(pages, layout.table_bytes_per_page);
    return {key_bytes, value_bytes, rounded_tokens, canonical, storage - canonical, tables,
            checked_add(storage, tables)};
}

} // namespace saga::kv
