#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace saga::media {

struct ImageDecodeLimits {
    std::size_t max_encoded_bytes{16U * 1024U * 1024U};
    std::uint32_t max_width{8192};
    std::uint32_t max_height{8192};
    std::size_t max_pixels{32U * 1024U * 1024U};
    std::size_t max_output_bytes{96U * 1024U * 1024U};
    std::size_t max_working_bytes{128U * 1024U * 1024U};
};

struct DecodedImage {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> rgb;
};

class ImageDecoder final {
public:
    static DecodedImage decode_rgb(std::span<const std::uint8_t> encoded,
                                   const ImageDecodeLimits& limits = {});
};

} // namespace saga::media
