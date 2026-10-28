#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "../shared/native_file_mapping.h"

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <system_error>
#include <utility>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace saga::platform::detail {

static_assert(std::numeric_limits<off_t>::digits >= 63, "64-bit file offsets required");

namespace {

[[noreturn]] void throw_path_error(const char* operation,
                                   const std::filesystem::path& path,
                                   int error) {
    throw std::filesystem::filesystem_error(
        operation, path, std::error_code(error, std::generic_category()));
}

class Descriptor {
public:
    explicit Descriptor(int value) noexcept : value_(value) {}
    ~Descriptor() { if (value_ >= 0) ::close(value_); }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const noexcept { return value_; }
    int release() noexcept { return std::exchange(value_, -1); }
private:
    int value_;
};

} // namespace

NativeFile::NativeFile(const std::filesystem::path& path) : path_(path) {
    int descriptor;
    do {
        descriptor = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) {
        throw_path_error("open", path_, errno);
    }
    Descriptor owned(descriptor);

    struct stat status{};
    int result;
    do {
        result = ::fstat(owned.get(), &status);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
        throw_path_error("fstat", path_, errno);
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        throw std::filesystem::filesystem_error(
            "file mapping requires a regular file", path_,
            std::make_error_code(std::errc::invalid_argument));
    }

    size_ = static_cast<std::uint64_t>(status.st_size);
    file_ = owned.release();
}

NativeFile::~NativeFile() {
    if (file_ >= 0) {
        ::close(file_);
    }
}

std::uint64_t NativeFile::size() const noexcept {
    return size_;
}

std::uint64_t NativeFile::granularity() const {
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::runtime_error("sysconf(_SC_PAGESIZE) returned no usable page size");
    }
    return static_cast<std::uint64_t>(page_size);
}

void* NativeFile::map(std::uint64_t offset, std::size_t mapped_size) {
    void* const view = ::mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE,
                              file_, static_cast<off_t>(offset));
    if (view == MAP_FAILED) {
        const int error = errno;
        throw_path_error("mmap", path_, error);
    }
    return view;
}

void unmap(void* address, std::size_t mapped_size) noexcept {
    ::munmap(address, mapped_size);
}

} // namespace saga::platform::detail
