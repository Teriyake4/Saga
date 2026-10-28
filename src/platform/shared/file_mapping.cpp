#include "saga/platform/file_mapping.hpp"
#include "native_file_mapping.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace saga::platform {

ReadOnlyFileMapping::ReadOnlyFileMapping(const std::filesystem::path& path,
                                         std::uint64_t offset,
                                         std::optional<std::uint64_t> length) {
    if (path.native().find(std::filesystem::path::value_type{}) !=
        std::filesystem::path::string_type::npos) {
        throw std::filesystem::filesystem_error(
            "file mapping path contains NUL", path,
            std::make_error_code(std::errc::invalid_argument));
    }

    detail::NativeFile file(path);
    file_size_ = file.size();
    if (offset > file_size_) {
        throw std::out_of_range("file mapping offset exceeds file size");
    }
    const auto available = file_size_ - offset;
    const auto requested = length.value_or(available);
    if (requested > available) {
        throw std::out_of_range("file mapping range exceeds file size");
    }
    offset_ = offset;
    if (requested == 0) {
        is_open_ = true;
        return;
    }

    const auto granularity = file.granularity();
    if (granularity == 0) {
        throw std::runtime_error("file mapping allocation granularity is zero");
    }
    const auto aligned_offset = offset - offset % granularity;
    const auto delta = offset - aligned_offset;
    constexpr auto max_difference =
        static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    constexpr auto max_size =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (delta > max_difference || requested > max_difference - delta ||
        delta > max_size || requested > max_size - delta) {
        throw std::length_error("file mapping range is too large for this address space");
    }

    const auto mapped_size = static_cast<std::size_t>(delta + requested);
    void* const view = file.map(aligned_offset, mapped_size);
    mapping_ = view;
    mapped_size_ = mapped_size;
    data_offset_ = static_cast<std::size_t>(delta);
    size_ = static_cast<std::size_t>(requested);
    is_open_ = true;
}

ReadOnlyFileMapping::~ReadOnlyFileMapping() {
    reset();
}

ReadOnlyFileMapping::ReadOnlyFileMapping(ReadOnlyFileMapping&& other) noexcept {
    *this = std::move(other);
}

ReadOnlyFileMapping& ReadOnlyFileMapping::operator=(ReadOnlyFileMapping&& other) noexcept {
    if (this != &other) {
        reset();
        mapping_ = std::exchange(other.mapping_, nullptr);
        mapped_size_ = std::exchange(other.mapped_size_, 0);
        data_offset_ = std::exchange(other.data_offset_, 0);
        size_ = std::exchange(other.size_, 0);
        file_size_ = std::exchange(other.file_size_, 0);
        offset_ = std::exchange(other.offset_, 0);
        is_open_ = std::exchange(other.is_open_, false);
    }
    return *this;
}

void ReadOnlyFileMapping::reset() noexcept {
    if (mapping_ != nullptr) {
        detail::unmap(mapping_, mapped_size_);
    }
    mapping_ = nullptr;
    mapped_size_ = data_offset_ = size_ = 0;
    file_size_ = offset_ = 0;
    is_open_ = false;
}

std::span<const std::byte> ReadOnlyFileMapping::bytes() const noexcept {
    if (size_ == 0) {
        return {};
    }
    return {static_cast<const std::byte*>(mapping_) + data_offset_, size_};
}

std::span<const std::byte> ReadOnlyFileMapping::subspan(std::uint64_t offset,
                                                        std::uint64_t length) const {
    if (offset > size_ || length > size_ - offset) {
        throw std::out_of_range("file mapping subspan exceeds mapped range");
    }
    if (size_ == 0) {
        return {};
    }
    return bytes().subspan(static_cast<std::size_t>(offset),
                           static_cast<std::size_t>(length));
}

} // namespace saga::platform
