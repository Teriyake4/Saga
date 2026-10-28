#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

namespace saga::platform {

// A read-only file view. The file must not be modified or truncated while mapped.
// Paths use filesystem's native encoding; construct paths from u8 strings for UTF-8
// input (not locale-dependent narrow strings on Windows).
class ReadOnlyFileMapping {
public:
    ReadOnlyFileMapping() noexcept = default;

    // A missing length maps through EOF. An empty range (including an empty file)
    // is valid and owns no native mapping. Offsets need not be page-aligned.
    // OS failures throw filesystem_error with the path and native error code;
    // invalid ranges throw out_of_range, unrepresentable views throw length_error.
    explicit ReadOnlyFileMapping(const std::filesystem::path& path,
                                 std::uint64_t offset = 0,
                                 std::optional<std::uint64_t> length = std::nullopt);
    ~ReadOnlyFileMapping();

    ReadOnlyFileMapping(const ReadOnlyFileMapping&) = delete;
    ReadOnlyFileMapping& operator=(const ReadOnlyFileMapping&) = delete;
    ReadOnlyFileMapping(ReadOnlyFileMapping&& other) noexcept;
    ReadOnlyFileMapping& operator=(ReadOnlyFileMapping&& other) noexcept;

    void reset() noexcept;
    [[nodiscard]] bool is_open() const noexcept { return is_open_; }
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::uint64_t offset() const noexcept { return offset_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

    // Offset is relative to this view, not the file. Checks before narrowing or
    // pointer arithmetic, including empty ranges. Throws out_of_range on failure.
    [[nodiscard]] std::span<const std::byte> subspan(std::uint64_t offset,
                                                  std::uint64_t length) const;

    // Spans borrow the mapping: reset/destruction/move-assignment of the owning
    // object invalidates them. A move transfers ownership, preserving the view.
private:
    void* mapping_ = nullptr;
    std::size_t mapped_size_ = 0;
    std::size_t data_offset_ = 0;
    std::size_t size_ = 0;
    std::uint64_t file_size_ = 0;
    std::uint64_t offset_ = 0;
    bool is_open_ = false;
};

} // namespace saga::platform
