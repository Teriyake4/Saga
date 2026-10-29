#pragma once

#include <cstddef>

typedef int cudaError_t;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorUnknown = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorStream = 3;
constexpr cudaError_t cudaErrorEvent = 4;
constexpr cudaError_t cudaErrorDevice = 5;

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDisableTiming = 2;

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
// The test double deterministically rejects zero-byte allocations.
cudaError_t cudaMalloc(void** pointer, std::size_t bytes);
cudaError_t cudaFree(void* pointer);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);
