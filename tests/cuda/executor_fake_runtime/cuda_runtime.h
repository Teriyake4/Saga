#pragma once
#include <cstddef>
typedef int cudaError_t;
constexpr cudaError_t cudaSuccess=0, cudaErrorNotReady=34, cudaErrorUnknown=999;
using cudaStream_t=void*; using cudaEvent_t=void*;
constexpr unsigned cudaStreamNonBlocking=1, cudaHostAllocPortable=1;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct cudaUUID_t { char bytes[16]{}; };
struct cudaDeviceProp { int major=8, minor=9, warpSize=32, multiProcessorCount=1; cudaUUID_t uuid{}; char name[128]="fake"; };
cudaError_t cudaSetDevice(int); cudaError_t cudaGetDevice(int*); cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);
cudaError_t cudaMemGetInfo(std::size_t*,std::size_t*); cudaError_t cudaDriverGetVersion(int*); cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned); cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t*); cudaError_t cudaEventDestroy(cudaEvent_t); cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t); cudaError_t cudaEventQuery(cudaEvent_t); cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);
cudaError_t cudaHostAlloc(void**,std::size_t,unsigned); cudaError_t cudaFreeHost(void*); cudaError_t cudaMalloc(void**,std::size_t); cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t); cudaError_t cudaPeekAtLastError();
const char* cudaGetErrorName(cudaError_t); const char* cudaGetErrorString(cudaError_t);
