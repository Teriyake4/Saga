#include "saga/platform/file_mapping.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#endif

namespace {
namespace fs = std::filesystem;
using saga::platform::ReadOnlyFileMapping;

static_assert(!std::is_copy_constructible_v<ReadOnlyFileMapping>);
static_assert(!std::is_copy_assignable_v<ReadOnlyFileMapping>);
static_assert(std::is_nothrow_move_constructible_v<ReadOnlyFileMapping>);
static_assert(std::is_nothrow_move_assignable_v<ReadOnlyFileMapping>);

void check(bool condition, std::string_view description) {
    if (!condition) throw std::runtime_error(std::string(description));
}

template <typename Exception, typename Function>
void expect_throw(Function&& function, std::string_view description) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string(description));
}

class TempDirectory {
public:
    explicit TempDirectory(const fs::path& parent) {
        check(fs::is_directory(parent), "temporary parent must be an existing directory");
        std::random_device random;
        for (unsigned attempt = 0; attempt < 100; ++attempt) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            auto candidate = parent / ("saga-platform-" + std::to_string(stamp) + "-" +
                                       std::to_string(random()) + "-" + std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(candidate, error)) {
                path_ = std::move(candidate);
                return;
            }
            if (error && error != std::errc::file_exists) {
                throw fs::filesystem_error("create test directory", candidate, error);
            }
        }
        throw std::runtime_error("could not create a unique test directory");
    }
    ~TempDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
        if (error) std::cerr << "WARNING: temporary directory cleanup failed: " << error.message() << '\n';
    }
    const fs::path& path() const noexcept { return path_; }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
private:
    fs::path path_;
};

void write_file(const fs::path& path, std::string_view content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    check(stream.is_open(), "open test file for writing");
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.close();
    check(!stream.fail(), "write and close test file");
}

bool equals(std::span<const std::byte> bytes, std::string_view text) {
    if (bytes.size() != text.size()) return false;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (std::to_integer<unsigned char>(bytes[i]) != static_cast<unsigned char>(text[i])) return false;
    }
    return true;
}

void check_closed(const ReadOnlyFileMapping& mapping) {
    check(!mapping.is_open() && mapping.file_size() == 0 && mapping.offset() == 0 &&
          mapping.size() == 0 && mapping.bytes().empty(), "closed/moved-from state");
}

void ordinary_and_bounds(const fs::path& directory) {
    const auto path = directory / "ordinary.bin";
    const std::string content("abc\0def\n0123456789", 18);
    write_file(path, content);
    ReadOnlyFileMapping mapping(path);
    check(mapping.is_open(), "ordinary file is open");
    check(mapping.file_size() == content.size() && mapping.size() == content.size() &&
          mapping.offset() == 0, "ordinary file metadata");
    check(equals(mapping.bytes(), content), "ordinary binary content");
    check(equals(mapping.subspan(4, 3), "def"), "bounded subspan");
    check(mapping.subspan(content.size(), 0).empty(), "empty subspan at EOF");
    constexpr auto huge = std::numeric_limits<std::uint64_t>::max();
    expect_throw<std::out_of_range>([&] { (void)mapping.subspan(huge, 0); }, "reject huge subspan offset");
    expect_throw<std::out_of_range>([&] { (void)mapping.subspan(1, huge); }, "reject overflowing subspan length");
    expect_throw<std::out_of_range>([&] { (void)mapping.subspan(content.size(), 1); }, "reject read past EOF");
    expect_throw<std::out_of_range>([&] { (void)mapping.subspan(content.size() + 1, 0); }, "reject empty range past EOF");

    ReadOnlyFileMapping window(path, 2, 7);
    check(window.offset() == 2 && window.size() == 7 && window.file_size() == content.size(), "window metadata");
    check(equals(window.bytes(), std::string_view(content).substr(2, 7)), "unaligned window content");
    expect_throw<std::out_of_range>([&] { (void)window.subspan(0, 8); }, "subspan cannot escape view into file");
    ReadOnlyFileMapping suffix(path, 8);
    check(equals(suffix.bytes(), std::string_view(content).substr(8)), "default length maps through EOF");
    ReadOnlyFileMapping empty_at_end(path, content.size(), 0);
    check(empty_at_end.is_open() && empty_at_end.size() == 0 && empty_at_end.offset() == content.size(),
          "zero-length view at EOF");
    ReadOnlyFileMapping empty_inside(path, 3, 0);
    check(empty_inside.bytes().empty() && empty_inside.subspan(0, 0).empty(), "zero-length interior view");

    // Repeated constructor failures exercise cleanup after the native file opens.
    for (int i = 0; i < 128; ++i) {
        expect_throw<std::out_of_range>([&] { ReadOnlyFileMapping bad(path, huge, 1); }, "reject huge file offset");
        expect_throw<std::out_of_range>([&] { ReadOnlyFileMapping bad(path, 1, huge); }, "reject overflowing file range");
    }
    expect_throw<std::out_of_range>([&] { ReadOnlyFileMapping bad(path, content.size(), 1); }, "reject mapping beyond EOF");
    expect_throw<std::out_of_range>([&] { ReadOnlyFileMapping bad(path, content.size() + 1, 0); }, "reject empty mapping past EOF");
}

void empty_and_moves(const fs::path& directory) {
    const auto empty_path = directory / "empty.bin";
    const auto data_path = directory / "move.bin";
    write_file(empty_path, "");
    write_file(data_path, "move content");
    ReadOnlyFileMapping closed;
    check_closed(closed);
    check(closed.subspan(0, 0).empty(), "closed empty subspan");
    expect_throw<std::out_of_range>([&] { (void)closed.subspan(0, 1); }, "closed nonempty subspan fails");
    ReadOnlyFileMapping empty(empty_path);
    check(empty.is_open() && empty.file_size() == 0 && empty.bytes().empty(), "empty file is valid");
    expect_throw<std::out_of_range>([&] { ReadOnlyFileMapping bad(empty_path, 0, 1); }, "empty file nonempty mapping fails");

    ReadOnlyFileMapping source(data_path, 1, 6);
    const auto borrowed = source.bytes();
    ReadOnlyFileMapping moved(std::move(source));
    check_closed(source);
    check(moved.is_open() && moved.offset() == 1 && moved.file_size() == 12 &&
          equals(moved.bytes(), "ove co"), "move construction transfers metadata and data");
    check(borrowed.data() == moved.bytes().data() && equals(borrowed, "ove co"), "move preserves borrowed address");
    ReadOnlyFileMapping destination(data_path);
    destination = std::move(moved);
    check_closed(moved);
    check(equals(destination.bytes(), "ove co"), "move assignment replaces an existing mapping");
    auto* self = &destination;
    destination = std::move(*self);
    check(equals(destination.bytes(), "ove co"), "self move assignment is harmless");
    destination = std::move(empty);
    check_closed(empty);
    check(destination.is_open() && destination.bytes().empty(), "move empty file over nonempty mapping");
    ReadOnlyFileMapping moved_empty(std::move(destination));
    check_closed(destination);
    check(moved_empty.is_open() && moved_empty.bytes().empty(), "move construct empty file mapping");
    moved_empty.reset();
    moved_empty.reset();
    check_closed(moved_empty);
    destination = std::move(closed);
    check_closed(destination);
    source = ReadOnlyFileMapping(data_path);
    check(equals(source.bytes(), "move content"), "moved-from object can be reused");
    source.reset();
    check_closed(source);
}

void errors_and_unicode(const fs::path& directory) {
    const auto missing = directory / "not-present.bin";
    bool caught = false;
    try {
        ReadOnlyFileMapping mapping(missing);
    } catch (const fs::filesystem_error& error) {
        caught = true;
        check(error.path1() == missing && error.code().value() != 0, "OS diagnostic includes path and error code");
        check(std::string_view(error.what()).find(
#ifdef _WIN32
                  "CreateFileW"
#else
                  "open"
#endif
              ) != std::string_view::npos, "OS diagnostic includes failing operation");
    }
    check(caught, "missing file must fail");
    for (int i = 0; i < 128; ++i) {
        expect_throw<fs::filesystem_error>([&] { ReadOnlyFileMapping bad(directory); },
                                           "reject directory and release native handle");
    }
    const auto unicode = directory / fs::path(u8"模型-é-🌌.bin");
    write_file(unicode, "unicode path content");
    {
        ReadOnlyFileMapping mapping(unicode);
        check(equals(mapping.bytes(), "unicode path content"), "UTF-8 path uses native file APIs");
    }
    // In particular on Windows, this also verifies that no native handles survive destruction.
    check(fs::remove(unicode), "remove Unicode file after destruction");
    auto native = (directory / "embedded").native();
    native.push_back(fs::path::value_type{});
    native += fs::path("suffix").native();
    expect_throw<fs::filesystem_error>([&] { ReadOnlyFileMapping bad{fs::path(native)}; }, "reject embedded NUL path");
}

bool make_sparse(const fs::path& path, std::uint64_t size) {
    write_file(path, "");
#ifdef _WIN32
    // resize_file alone can allocate 4 GiB on Windows. Require sparse support first.
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("open Windows sparse test file");
    DWORD returned = 0;
    const bool sparse = DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
                                        &returned, nullptr) != 0;
    const auto native_error = GetLastError();
    CloseHandle(file);
    if (!sparse) {
        std::cout << "SKIP: filesystem cannot create sparse file (Windows error " << native_error << ")\n";
        return false;
    }
#endif
    std::error_code error;
    fs::resize_file(path, size, error);
    if (error) {
        std::cout << "SKIP: filesystem cannot resize sparse file: " << error.message() << '\n';
        return false;
    }
    return true;
}

void large_sparse_window(const fs::path& directory) {
    static_assert(std::numeric_limits<std::streamoff>::digits >= 63);
    constexpr std::uint64_t boundary = std::uint64_t{1} << 32;
    constexpr std::uint64_t high_offset = boundary + 123;
    constexpr std::uint64_t file_size = high_offset + 64;
    const auto path = directory / "large-sparse.bin";
    if (!make_sparse(path, file_size)) return;
    {
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        check(stream.is_open(), "open sparse file for marker writes");
        stream.seekp(static_cast<std::streamoff>(boundary - 4));
        stream.write("BOUNDARY", 8);
        stream.seekp(static_cast<std::streamoff>(high_offset));
        stream.write("HIGH", 4);
        stream.close();
        check(!stream.fail(), "write markers across and beyond 4 GiB");
    }
    {
        ReadOnlyFileMapping cross(path, boundary - 4, 8);
        check(cross.file_size() == file_size && cross.offset() == boundary - 4 &&
              equals(cross.bytes(), "BOUNDARY"), "window crossing 32-bit boundary");
        ReadOnlyFileMapping high(path, high_offset, 4);
        check(high.offset() > boundary && equals(high.bytes(), "HIGH"), "unaligned window beyond 4 GiB");
        ReadOnlyFileMapping suffix(path, high_offset);
        check(suffix.size() == 64 && equals(suffix.subspan(0, 4), "HIGH"), "64-bit EOF subtraction");
        check(suffix.bytes().back() == std::byte{0}, "sparse hole reads as zero");
        ReadOnlyFileMapping end(path, file_size, 0);
        check(end.offset() == file_size && end.is_open() && end.bytes().empty(), "64-bit empty EOF window");
        expect_throw<std::out_of_range>([&] { (void)high.subspan(boundary, 1); }, "subspan offset is not truncated to 32 bits");
        expect_throw<std::out_of_range>([&] { ReadOnlyFileMapping bad(path, high_offset, 65); }, "64-bit file bounds");
    }
    check(fs::remove(path), "release sparse mappings and native handles");
}

} // namespace

// Optional argument: existing native-encoded temporary parent directory. All test
// files live in a unique child, removed on both success and exception unwinding.
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc > 2) throw std::runtime_error("usage: saga_platform_test [temporary-parent-directory]");
        const TempDirectory temporary(argc == 2 ? fs::path(argv[1]) : fs::temp_directory_path());
        ordinary_and_bounds(temporary.path());
        empty_and_moves(temporary.path());
        errors_and_unicode(temporary.path());
        large_sparse_window(temporary.path());
        std::cout << "PASS: read-only file mapping tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
