#include "resource_owners.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
struct RuntimeState {
    int current_device = 3;
    int malloc_calls = 0;
    int free_calls = 0;
    bool pending_work = false;
    bool fail_get_device = false;
    bool fail_set_device = false;
    bool fail_restore_device = false;
    bool fail_stream_create = false;
    bool partial_stream_create = false;
    bool fail_stream_sync = false;
    bool fail_stream_destroy = false;
    bool fail_event_create = false;
    bool partial_event_create = false;
    bool fail_event_destroy = false;
    bool fail_malloc = false;
    bool partial_malloc = false;
    bool fail_free = false;
    std::vector<std::string> actions;
} state;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void clear_actions() {
    state.actions.clear();
}

std::size_t action_index(const std::string& name) {
    for (std::size_t i = 0; i < state.actions.size(); ++i) {
        if (state.actions[i] == name) {
            return i;
        }
    }
    throw std::runtime_error("missing runtime action: " + name);
}

template <typename Exception, typename Function>
std::string expect_exception(Function&& function, const std::string& description) {
    try {
        function();
    } catch (const Exception& error) {
        return error.what();
    } catch (...) {
        throw std::runtime_error(description + ": unexpected exception type");
    }
    throw std::runtime_error(description + ": expected exception was not thrown");
}
} // namespace

cudaError_t cudaGetDevice(int* device) {
    state.actions.emplace_back("get_device");
    if (state.fail_get_device) return cudaErrorDevice;
    *device = state.current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    state.actions.emplace_back(device == 3 ? "restore_device" : "set_device");
    if (device == 3 && state.fail_restore_device) return cudaErrorDevice;
    if (device != 3 && state.fail_set_device) return cudaErrorDevice;
    state.current_device = device;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    state.actions.emplace_back("stream_create");
    if (state.partial_stream_create) *stream = std::malloc(1);
    if (state.fail_stream_create) return cudaErrorStream;
    *stream = std::malloc(1);
    return *stream == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    state.actions.emplace_back("stream_destroy");
    std::free(stream);
    return state.fail_stream_destroy ? cudaErrorStream : cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) {
    state.actions.emplace_back("stream_sync");
    if (state.fail_stream_sync) return cudaErrorStream;
    state.pending_work = false;
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) {
    state.actions.emplace_back("event_create");
    if (state.partial_event_create) *event = std::malloc(1);
    if (state.fail_event_create) return cudaErrorEvent;
    *event = std::malloc(1);
    return *event == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    state.actions.emplace_back("event_destroy");
    std::free(event);
    return state.fail_event_destroy ? cudaErrorEvent : cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) {
    state.actions.emplace_back("event_record");
    state.pending_work = true;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t) {
    state.actions.emplace_back("event_sync");
    state.pending_work = false;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** pointer, std::size_t bytes) {
    ++state.malloc_calls;
    state.actions.emplace_back("malloc");
    if (bytes == 0) {
        *pointer = nullptr;
        return cudaErrorMemoryAllocation;
    }
    if (state.partial_malloc) *pointer = std::malloc(bytes);
    if (state.fail_malloc) return cudaErrorMemoryAllocation;
    *pointer = std::malloc(bytes);
    return *pointer == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

cudaError_t cudaFree(void* pointer) {
    ++state.free_calls;
    state.actions.emplace_back("free");
    std::free(pointer);
    return state.fail_free ? cudaErrorUnknown : cudaSuccess;
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorStream: return "cudaErrorStream";
    case cudaErrorEvent: return "cudaErrorEvent";
    case cudaErrorDevice: return "cudaErrorDevice";
    default: return "cudaErrorUnknown";
    }
}

const char* cudaGetErrorString(cudaError_t) { return "fake runtime failure"; }

int main() {
    using saga::cuda_internal::CleanupReport;
    using saga::cuda_internal::DeviceBuffer;
    using saga::cuda_internal::DeviceGuard;
    using saga::cuda_internal::Event;
    using saga::cuda_internal::Stream;

    static_assert(!std::is_copy_constructible_v<DeviceGuard>);
    static_assert(!std::is_copy_assignable_v<DeviceGuard>);
    static_assert(!std::is_copy_constructible_v<DeviceBuffer<int>>);
    static_assert(!std::is_copy_assignable_v<DeviceBuffer<int>>);
    static_assert(!std::is_copy_constructible_v<Stream>);
    static_assert(!std::is_copy_assignable_v<Stream>);
    static_assert(!std::is_copy_constructible_v<Event>);
    static_assert(!std::is_copy_assignable_v<Event>);

    {
        CleanupReport cleanup;
        DeviceGuard guard(7, cleanup);
        check(state.current_device == 7, "guard selects requested device");
        {
            DeviceBuffer<int> buffer(cleanup);
            buffer.allocate(4);
        }
        check(state.current_device == 7, "buffer cleanup occurs on selected device");
    }
    check(state.current_device == 3, "guard restores prior device");
    check(state.free_calls == 1, "allocated buffer freed once");

    {
        CleanupReport cleanup;
        const int calls_before = state.malloc_calls;
        DeviceBuffer<int> buffer(cleanup);
        const auto message = expect_exception<std::invalid_argument>([&] {
            buffer.allocate(0);
        }, "zero-size allocation");
        check(message.find("nonzero") != std::string::npos, "zero-size error is descriptive");
        check(state.malloc_calls == calls_before, "zero-size rejected before cudaMalloc");
    }

    {
        CleanupReport cleanup;
        const int calls_before = state.malloc_calls;
        DeviceBuffer<int> buffer(cleanup);
        const auto message = expect_exception<std::overflow_error>([&] {
            buffer.allocate(std::numeric_limits<std::size_t>::max());
        }, "overflow allocation");
        check(message.find("overflow") != std::string::npos, "overflow message is descriptive");
        check(state.malloc_calls == calls_before, "overflow rejected before cudaMalloc");
    }

    {
        CleanupReport cleanup;
        state.fail_malloc = true;
        const auto message = expect_exception<std::runtime_error>([&] {
            DeviceBuffer<int> buffer(cleanup);
            buffer.allocate(2);
        }, "malloc failure");
        state.fail_malloc = false;
        check(message.find("cudaMalloc") != std::string::npos, "allocation error names operation");
        check(state.free_calls == 1, "failed allocation does not free null pointer");
    }

    {
        CleanupReport cleanup;
        state.fail_malloc = true;
        state.partial_malloc = true;
        const int frees_before = state.free_calls;
        const auto message = expect_exception<std::runtime_error>([&] {
            DeviceBuffer<int> buffer(cleanup);
            buffer.allocate(2);
        }, "partial malloc failure");
        state.fail_malloc = false;
        state.partial_malloc = false;
        check(message.find("cudaMalloc") != std::string::npos, "partial allocation reports original failure");
        check(state.free_calls == frees_before + 1, "partial allocation is released during unwinding");
    }

    {
        CleanupReport cleanup;
        state.fail_get_device = true;
        const auto message = expect_exception<std::runtime_error>([&] { DeviceGuard guard(7, cleanup); },
                                                                  "device query failure");
        state.fail_get_device = false;
        check(message.find("cudaGetDevice") != std::string::npos, "device query error names operation");
    }

    {
        CleanupReport cleanup;
        state.fail_set_device = true;
        const auto message = expect_exception<std::runtime_error>([&] { DeviceGuard guard(7, cleanup); },
                                                                  "device selection failure");
        state.fail_set_device = false;
        check(message.find("cudaSetDevice") != std::string::npos, "device selection error names operation");
        check(state.current_device == 3, "failed selection preserves previous device");
    }

    {
        CleanupReport cleanup;
        state.fail_stream_create = true;
        state.partial_stream_create = true;
        clear_actions();
        const auto message = expect_exception<std::runtime_error>([&] { Stream stream(cleanup); },
                                                                  "partial stream construction");
        state.fail_stream_create = false;
        state.partial_stream_create = false;
        check(message.find("cudaStreamCreateWithFlags") != std::string::npos,
              "stream construction error names operation");
        check(state.actions.size() == 2 && state.actions[0] == "stream_create" &&
                  state.actions[1] == "stream_destroy",
              "partially-created stream is destroyed");
    }

    {
        CleanupReport cleanup;
        state.fail_event_create = true;
        state.partial_event_create = true;
        const auto message = expect_exception<std::runtime_error>([&] { Event event(cleanup); },
                                                                  "partial event construction");
        state.fail_event_create = false;
        state.partial_event_create = false;
        check(message.find("cudaEventCreateWithFlags") != std::string::npos,
              "event construction error names operation");
        check(state.actions.back() == "event_destroy", "partially-created event is destroyed");
    }

    {
        CleanupReport cleanup;
        clear_actions();
        const int frees_before = state.free_calls;
        try {
            // Declaration order is deliberate: on successful synchronization, the
            // stream drains before event and buffer storage die during unwinding.
            DeviceBuffer<int> buffer(cleanup);
            buffer.allocate(2);
            Event event(cleanup);
            Stream stream(cleanup);
            event.record(stream.get());
            event.synchronize();
            throw std::runtime_error("exercise owner unwinding");
        } catch (const std::runtime_error& error) {
            check(std::string(error.what()) == "exercise owner unwinding",
                  "original exception propagates through owner cleanup");
        }
        check(state.free_calls == frees_before + 1, "buffer freed during exception unwinding");
        check(action_index("stream_sync") < action_index("stream_destroy"),
              "stream is fenced before its destruction");
        check(action_index("stream_destroy") < action_index("event_destroy"),
              "stream drains before event destruction");
        check(action_index("event_destroy") < action_index("free"),
              "stream drains before device buffer free");
    }

    {
        CleanupReport cleanup;
        state.fail_stream_sync = true;
        state.fail_stream_destroy = true;
        state.fail_event_destroy = true;
        state.pending_work = true;
        state.fail_free = true;
        {
            DeviceBuffer<int> buffer(cleanup);
            buffer.allocate(1);
            Event event(cleanup);
            Stream stream(cleanup);
        }
        state.fail_stream_sync = false;
        state.fail_stream_destroy = false;
        state.fail_event_destroy = false;
        state.fail_free = false;
        check(cleanup.first_error == cudaErrorStream, "stream fence retains first cleanup error");
        check(action_index("stream_sync") < action_index("stream_destroy"),
              "stream destroy follows attempted fence even on fence error");
        check(state.pending_work, "failed synchronization does not claim pending work was fenced");
        state.pending_work = false;
    }

    {
        CleanupReport cleanup;
        state.fail_event_destroy = true;
        { Event event(cleanup); }
        state.fail_event_destroy = false;
        check(cleanup.first_error == cudaErrorEvent, "event cleanup error is retained");
    }

    {
        CleanupReport cleanup;
        state.fail_free = true;
        { DeviceBuffer<int> buffer(cleanup); buffer.allocate(1); }
        state.fail_free = false;
        check(cleanup.first_error == cudaErrorUnknown, "buffer cleanup error is retained");
    }

    {
        CleanupReport cleanup;
        state.fail_restore_device = true;
        { DeviceGuard guard(8, cleanup); }
        state.fail_restore_device = false;
        check(cleanup.first_error == cudaErrorDevice, "device restoration error is retained");
        cleanup.retain(cudaErrorUnknown);
        check(cleanup.first_error == cudaErrorDevice, "first cleanup error is never overwritten");
    }

    {
        CleanupReport cleanup;
        state.fail_stream_sync = true;
        state.fail_stream_destroy = true;
        {
            Stream stream(cleanup);
        }
        state.fail_stream_sync = false;
        state.fail_stream_destroy = false;
        check(cleanup.first_error == cudaErrorStream, "stream sync error precedes destroy error");
    }

    return 0;
}
