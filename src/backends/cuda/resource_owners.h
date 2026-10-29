#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace saga::cuda_internal {

// Internal smoke-probe owners only; this is not a general device-affine wrapper.
// Use all owners on the same host thread under one enclosing DeviceGuard. The guard
// and CleanupReport must outlive every resource; do not change the selected device
// while any owner/resource is live. Declare dependent buffers/events before Stream,
// so reverse destruction synchronizes the stream before those dependents are released.
struct CleanupReport {
    cudaError_t first_error = cudaSuccess;

    void retain(cudaError_t result) noexcept {
        if (first_error == cudaSuccess && result != cudaSuccess) {
            first_error = result;
        }
    }
};

class DeviceGuard {
public:
    DeviceGuard(int device, CleanupReport& cleanup) : cleanup_(cleanup) {
        check(cudaGetDevice(&previous_device_), "cudaGetDevice");
        const cudaError_t result = cudaSetDevice(device);
        if (result != cudaSuccess) {
            cleanup_.retain(cudaSetDevice(previous_device_));
            check(result, "cudaSetDevice");
        }
        active_ = true;
    }

    ~DeviceGuard() noexcept {
        if (active_) {
            cleanup_.retain(cudaSetDevice(previous_device_));
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    static void check(cudaError_t result, const char* operation) {
        if (result == cudaSuccess) {
            return;
        }
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(result) +
                                 " (" + std::to_string(static_cast<int>(result)) + "): " +
                                 cudaGetErrorString(result));
    }

private:
    CleanupReport& cleanup_;
    int previous_device_ = 0;
    bool active_ = false;
};

class Stream {
public:
    explicit Stream(CleanupReport& cleanup) : cleanup_(cleanup) {
        const cudaError_t result = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (result != cudaSuccess) {
            if (stream_ != nullptr) {
                cleanup_.retain(cudaStreamDestroy(stream_));
                stream_ = nullptr;
            }
            DeviceGuard::check(result, "cudaStreamCreateWithFlags");
        }
    }

    // Stream destruction is not a physical-work fence. Attempt synchronization first.
    // If it fails, retain the error; queued work may still be pending, so this is not
    // evidence of a successful physical fence and dependents must not be treated safe.
    ~Stream() noexcept {
        if (stream_ != nullptr) {
            cleanup_.retain(cudaStreamSynchronize(stream_));
            cleanup_.retain(cudaStreamDestroy(stream_));
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t get() const noexcept { return stream_; }

    void synchronize() {
        DeviceGuard::check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    CleanupReport& cleanup_;
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    explicit Event(CleanupReport& cleanup) : cleanup_(cleanup) {
        const cudaError_t result = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
        if (result != cudaSuccess) {
            if (event_ != nullptr) {
                cleanup_.retain(cudaEventDestroy(event_));
                event_ = nullptr;
            }
            DeviceGuard::check(result, "cudaEventCreateWithFlags");
        }
    }

    ~Event() noexcept {
        if (event_ != nullptr) {
            cleanup_.retain(cudaEventDestroy(event_));
        }
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

    void record(cudaStream_t stream) {
        DeviceGuard::check(cudaEventRecord(event_, stream), "cudaEventRecord");
    }

    void synchronize() {
        DeviceGuard::check(cudaEventSynchronize(event_), "cudaEventSynchronize");
    }

private:
    CleanupReport& cleanup_;
    cudaEvent_t event_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(CleanupReport& cleanup) noexcept : cleanup_(cleanup) {}

    ~DeviceBuffer() noexcept {
        if (pointer_ != nullptr) {
            cleanup_.retain(cudaFree(pointer_));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        if (pointer_ != nullptr) {
            throw std::logic_error("device buffer already allocated");
        }
        if (count == 0) {
            throw std::invalid_argument("device buffer allocation size must be nonzero");
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("device buffer allocation size overflow");
        }
        DeviceGuard::check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
                           "cudaMalloc");
    }

    T* get() const noexcept { return pointer_; }

private:
    CleanupReport& cleanup_;
    T* pointer_ = nullptr;
};

} // namespace saga::cuda_internal
