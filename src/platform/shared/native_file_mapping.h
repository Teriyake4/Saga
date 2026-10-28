#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace saga::platform::detail {

class NativeFile {
public:
    explicit NativeFile(const std::filesystem::path& path);
    ~NativeFile();

    NativeFile(const NativeFile&) = delete;
    NativeFile& operator=(const NativeFile&) = delete;

    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] std::uint64_t granularity() const;
    [[nodiscard]] void* map(std::uint64_t aligned_offset, std::size_t mapped_size);

private:
    std::filesystem::path path_;
#ifdef _WIN32
    void* file_ = nullptr;
#else
    int file_ = -1;
#endif
    std::uint64_t size_ = 0;
};

void unmap(void* address, std::size_t mapped_size) noexcept;

} // namespace saga::platform::detail
