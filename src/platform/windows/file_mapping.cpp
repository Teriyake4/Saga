#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../shared/native_file_mapping.h"

#include <windows.h>

#include <system_error>
#include <utility>

namespace saga::platform::detail {

namespace {

[[noreturn]] void throw_path_error(const char* operation,
                                   const std::filesystem::path& path,
                                   DWORD error) {
    throw std::filesystem::filesystem_error(
        operation, path,
        std::error_code(static_cast<int>(error), std::system_category()));
}

class Handle {
public:
    explicit Handle(HANDLE value) noexcept : value_(value) {}
    ~Handle() { if (value_ != INVALID_HANDLE_VALUE && value_ != nullptr) CloseHandle(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    HANDLE get() const noexcept { return value_; }
    HANDLE release() noexcept { return std::exchange(value_, INVALID_HANDLE_VALUE); }
private:
    HANDLE value_;
};

} // namespace

NativeFile::NativeFile(const std::filesystem::path& path) : path_(path) {
    Handle file(CreateFileW(path_.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) {
        throw_path_error("CreateFileW", path_, GetLastError());
    }
    if (GetFileType(file.get()) != FILE_TYPE_DISK) {
        throw std::filesystem::filesystem_error(
            "file mapping requires a regular file", path_,
            std::make_error_code(std::errc::invalid_argument));
    }
    LARGE_INTEGER native_size{};
    if (!GetFileSizeEx(file.get(), &native_size)) {
        throw_path_error("GetFileSizeEx", path_, GetLastError());
    }
    if (native_size.QuadPart < 0) {
        throw std::filesystem::filesystem_error(
            "file mapping requires a regular file", path_,
            std::make_error_code(std::errc::invalid_argument));
    }

    size_ = static_cast<std::uint64_t>(native_size.QuadPart);
    file_ = file.release();
}

NativeFile::~NativeFile() {
    if (file_ != nullptr && file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(static_cast<HANDLE>(file_));
    }
}

std::uint64_t NativeFile::size() const noexcept {
    return size_;
}

std::uint64_t NativeFile::granularity() const {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwAllocationGranularity;
}

void* NativeFile::map(std::uint64_t offset, std::size_t mapped_size) {
    Handle section(CreateFileMappingW(static_cast<HANDLE>(file_), nullptr,
                                      PAGE_READONLY, 0, 0, nullptr));
    if (section.get() == nullptr) {
        throw_path_error("CreateFileMappingW", path_, GetLastError());
    }
    void* const view = MapViewOfFile(
        section.get(), FILE_MAP_READ, static_cast<DWORD>(offset >> 32),
        static_cast<DWORD>(offset & 0xffffffffULL), mapped_size);
    if (view == nullptr) {
        const DWORD error = GetLastError();
        throw_path_error("MapViewOfFile", path_, error);
    }
    return view;
}

void unmap(void* address, std::size_t) noexcept {
    UnmapViewOfFile(address);
}

} // namespace saga::platform::detail
