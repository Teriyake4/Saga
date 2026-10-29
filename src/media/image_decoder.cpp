#include "saga/media/image_decoder.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {
struct AllocationState {
    std::size_t used{};
    std::size_t limit{};
    bool exceeded{};
};
thread_local AllocationState* active_state = nullptr;

struct alignas(std::max_align_t) AllocationHeader { std::size_t charged; };
void* bounded_alloc(std::size_t n) noexcept {
    auto* s = active_state;
    if (!s || n > std::numeric_limits<std::size_t>::max() - sizeof(AllocationHeader)) return nullptr;
    const std::size_t charge = n + sizeof(AllocationHeader);
    if (charge > s->limit - s->used) { s->exceeded = true; return nullptr; }
    auto* h = static_cast<AllocationHeader*>(std::malloc(charge));
    if (!h) return nullptr;
    h->charged = charge;
    s->used += charge;
    return h + 1;
}
void bounded_free(void* p) noexcept {
    if (!p) return;
    auto* h = static_cast<AllocationHeader*>(p) - 1;
    if (active_state && active_state->used >= h->charged) active_state->used -= h->charged;
    std::free(h);
}
void* bounded_realloc(void* p, std::size_t n) noexcept {
    if (!p) return bounded_alloc(n);
    if (n == 0) { bounded_free(p); return nullptr; }
    auto* old = static_cast<AllocationHeader*>(p) - 1;
    auto* s = active_state;
    if (!s || n > std::numeric_limits<std::size_t>::max() - sizeof(AllocationHeader)) return nullptr;
    const auto charge = n + sizeof(AllocationHeader);
    const auto prior = old->charged;
    if (charge > prior && charge - prior > s->limit - s->used) { s->exceeded = true; return nullptr; }
    auto* next = static_cast<AllocationHeader*>(std::realloc(old, charge));
    if (!next) return nullptr;
    next->charged = charge;
    s->used = s->used - prior + charge;
    return next + 1;
}
struct DecodeScope {
    AllocationState state;
    AllocationState* previous;
    explicit DecodeScope(std::size_t limit) : state{0, limit, false}, previous(active_state) { active_state = &state; }
    ~DecodeScope() { active_state = previous; }
};
#define STBI_MALLOC(sz) bounded_alloc(sz)
#define STBI_REALLOC(p, newsz) bounded_realloc(p, newsz)
#define STBI_FREE(p) bounded_free(p)
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
} // namespace

namespace saga::media {
DecodedImage ImageDecoder::decode_rgb(std::span<const std::uint8_t> encoded,
                                      const ImageDecodeLimits& limits) {
    if (encoded.empty() || encoded.size() > limits.max_encoded_bytes ||
        encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("encoded image exceeds limit or is empty");
    constexpr std::array<std::uint8_t, 8> png_signature{
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    const bool png = encoded.size() >= png_signature.size() &&
                     std::equal(png_signature.begin(), png_signature.end(), encoded.begin());
    const bool jpeg = encoded.size() >= 3 && encoded[0] == 0xff && encoded[1] == 0xd8 && encoded[2] == 0xff;
    if (!png && !jpeg) throw std::runtime_error("unsupported image format (PNG/JPEG only)");
    if (png && encoded.size() > 24 && encoded[24] == 16)
        throw std::runtime_error("16-bit PNG is unsupported");

    DecodeScope scope(limits.max_working_bytes);
    int width = 0, height = 0, channels = 0;
    if (!stbi_info_from_memory(encoded.data(), static_cast<int>(encoded.size()),
                              &width, &height, &channels))
        throw std::runtime_error(scope.state.exceeded ? "image decode working-memory limit exceeded" : "malformed or truncated image");
    if (width <= 0 || height <= 0 || static_cast<std::uint32_t>(width) > limits.max_width ||
        static_cast<std::uint32_t>(height) > limits.max_height)
        throw std::runtime_error("image dimensions exceed limit");
    const auto w = static_cast<std::size_t>(width), h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h || w * h > limits.max_pixels ||
        w * h > limits.max_output_bytes / 3)
        throw std::runtime_error("image pixel/output limit exceeded");
    auto* pixels = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()),
                                         &width, &height, &channels, 3);
    struct PixelGuard { stbi_uc* p; ~PixelGuard() { stbi_image_free(p); } } guard{pixels};
    if (!pixels) throw std::runtime_error(scope.state.exceeded ? "image decode working-memory limit exceeded" : "malformed or truncated image");
    const auto bytes = w * h * 3;
    DecodedImage result;
    result.width = static_cast<std::uint32_t>(width);
    result.height = static_cast<std::uint32_t>(height);
    result.rgb.assign(pixels, pixels + bytes);
    return result;
}
} // namespace saga::media
