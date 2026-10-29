#include "saga/media/image_decoder.h"

#include <array>
#include <atomic>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using saga::media::ImageDecodeLimits;
using saga::media::ImageDecoder;

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Function>
void expect_failure(Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected decode rejection");
}

constexpr std::array<unsigned char, 69> png{{
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
    0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
    0x00, 0x03, 0x01, 0x01, 0x00, 0xc9, 0xfe, 0x92, 0xef, 0x00, 0x00, 0x00,
    0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
}};
constexpr std::array<unsigned char, 634> jpeg{{
#include "jpeg_fixture.inc"
}};
constexpr std::array<unsigned char, 4> unsupported{{1, 2, 3, 4}};

void check_red_jpeg(const saga::media::DecodedImage& image) {
    check(image.width == 8 && image.height == 8 && image.rgb.size() == 8U * 8U * 3U,
          "JPEG dimensions/output size");
    for (std::size_t i = 0; i < image.rgb.size(); i += 3) {
        check(image.rgb[i] >= 220, "JPEG red channel/order");
        check(image.rgb[i + 1] <= 35, "JPEG green channel/order");
        check(image.rgb[i + 2] <= 35, "JPEG blue channel/order");
    }
}
} // namespace

int main() {
    try {
        const auto decoded_png = ImageDecoder::decode_rgb(png);
        check(decoded_png.width == 1 && decoded_png.height == 1 && decoded_png.rgb.size() == 3,
              "PNG dimensions/output size");
        check(decoded_png.rgb[0] == 255 && decoded_png.rgb[1] == 0 && decoded_png.rgb[2] == 0,
              "PNG exact RGB pixels");

        const auto decoded_jpeg = ImageDecoder::decode_rgb(jpeg);
        check_red_jpeg(decoded_jpeg);

        expect_failure([] {
            ImageDecoder::decode_rgb(std::span<const unsigned char>(png.data(), png.size() - 10));
        });
        expect_failure([] {
            ImageDecoder::decode_rgb(std::span<const unsigned char>(jpeg.data(), jpeg.size() - 20));
        });
        expect_failure([] {
            ImageDecoder::decode_rgb(std::span<const unsigned char>(jpeg.data(), 2));
        });
        expect_failure([] { ImageDecoder::decode_rgb(unsupported); });

        ImageDecodeLimits limits;
        limits.max_encoded_bytes = 4;
        expect_failure([&] { ImageDecoder::decode_rgb(png, limits); });
        limits = {};
        limits.max_width = 0;
        expect_failure([&] { ImageDecoder::decode_rgb(png, limits); });
        limits = {};
        limits.max_pixels = 0;
        expect_failure([&] { ImageDecoder::decode_rgb(png, limits); });
        limits = {};
        limits.max_output_bytes = 2;
        expect_failure([&] { ImageDecoder::decode_rgb(png, limits); });
        limits = {};
        limits.max_working_bytes = 1;
        expect_failure([&] { ImageDecoder::decode_rgb(png, limits); });
        check(ImageDecoder::decode_rgb(png).rgb == decoded_png.rgb, "recovery after rejected inputs");

        std::atomic<bool> concurrent_ok{true};
        std::vector<std::thread> threads;
        for (int thread_index = 0; thread_index < 8; ++thread_index) {
            threads.emplace_back([&, thread_index] {
                try {
                    for (int iteration = 0; iteration < 30; ++iteration) {
                        if ((thread_index + iteration) % 2 == 0) {
                            if (ImageDecoder::decode_rgb(png).rgb != decoded_png.rgb) {
                                concurrent_ok = false;
                            }
                        } else {
                            check_red_jpeg(ImageDecoder::decode_rgb(jpeg));
                        }
                    }
                } catch (...) {
                    concurrent_ok = false;
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        check(concurrent_ok, "concurrent mixed PNG/JPEG decode");
        std::cout << "PASS image decoder\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
