#include "saga/backend/cuda.h"
#include "cuda_runtime.h"
#include "control.h"

#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace fake {
bool fail_query=false;
int event_queries=0;
int launch_checks=0;
bool fail_set_device=false;
bool fail_copy=false;
int not_ready_queries=0;
bool fail_record=false;
int fail_event_create_at=0;
int event_creates=0;
int event_destroys=0;
int frees=0;
int allocations=0;
int current_device=0;
}

namespace saga::backend::cuda {
backend::KernelRequest make_bf16_projection_request(const ProjectionRequest& request, std::string_view) {
    if (!request.shape.outputs && !request.shape.inputs && !request.shape.tokens) return {};
    backend::KernelRequest result;
    result.phase = request.phase;
    result.shape = {request.shape.outputs, request.shape.inputs, request.shape.tokens};
    result.physical = {request.strides.weight_row_bytes, request.strides.activation_row_bytes,
                       request.strides.output_row_bytes, request.alignment, 0};
    return result;
}
}

static int handle_counter=0;
cudaError_t cudaSetDevice(int d){ if(fake::fail_set_device)return cudaErrorUnknown; fake::current_device=d; return cudaSuccess; }
cudaError_t cudaGetDevice(int* d){ *d=fake::current_device; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int ordinal){
    *p=cudaDeviceProp{};
    if (ordinal != 0) { p->uuid.bytes[0]=1; p->name[0]='g'; }
    return cudaSuccess;
}
cudaError_t cudaMemGetInfo(std::size_t* f,std::size_t* t){ *f=1<<20; *t=2<<20; return cudaSuccess; }
cudaError_t cudaDriverGetVersion(int* v){*v=1;return cudaSuccess;}
cudaError_t cudaRuntimeGetVersion(int* v){*v=1;return cudaSuccess;}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* p,unsigned){*p=reinterpret_cast<void*>(++handle_counter);return cudaSuccess;}
cudaError_t cudaStreamDestroy(cudaStream_t){return cudaSuccess;}
cudaError_t cudaEventCreate(cudaEvent_t* p){ ++fake::event_creates; if(fake::fail_event_create_at==fake::event_creates)return cudaErrorUnknown; *p=reinterpret_cast<void*>(++handle_counter);return cudaSuccess; }
cudaError_t cudaEventDestroy(cudaEvent_t){++fake::event_destroys;return cudaSuccess;}
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){return fake::fail_record?cudaErrorUnknown:cudaSuccess;}
cudaError_t cudaEventQuery(cudaEvent_t){++fake::event_queries; if(fake::fail_query)return cudaErrorUnknown; if(fake::not_ready_queries<0)return cudaErrorNotReady; if(fake::not_ready_queries>0){--fake::not_ready_queries;return cudaErrorNotReady;}return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t){*ms=1;return cudaSuccess;}
cudaError_t cudaHostAlloc(void** p,std::size_t n,unsigned){
#if defined(_WIN32)
    *p=_aligned_malloc(n,256); if(!*p)return cudaErrorUnknown;
#else
    if(posix_memalign(p,256,n))return cudaErrorUnknown;
#endif
    ++fake::allocations;return cudaSuccess;}
cudaError_t cudaFreeHost(void* p){++fake::frees;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
    return cudaSuccess;}
cudaError_t cudaMalloc(void** p,std::size_t n){
#if defined(_WIN32)
    *p=_aligned_malloc(n,256); if(!*p)return cudaErrorUnknown;
#else
    if(posix_memalign(p,256,n))return cudaErrorUnknown;
#endif
    ++fake::allocations;return cudaSuccess;}
cudaError_t cudaFree(void* p){++fake::frees;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
    return cudaSuccess;}
cudaError_t cudaMemcpyAsync(void* d,const void* s,std::size_t n,cudaMemcpyKind,cudaStream_t){if(fake::fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
cudaError_t cudaPeekAtLastError(){++fake::launch_checks;return cudaSuccess;}
const char* cudaGetErrorName(cudaError_t){return "fake";}
const char* cudaGetErrorString(cudaError_t){return "fake runtime failure";}
