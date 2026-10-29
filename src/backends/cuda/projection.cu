#include "saga/backend/cuda.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <string>

namespace saga::backend::cuda {
namespace {
using namespace nvcuda;
constexpr const char* baseline_id = "sm89.bf16_projection.baseline";
constexpr const char* wmma_id = "sm89.bf16_projection.wmma";
const char* phase_suffix(Phase phase) {
    switch (phase) {
    case Phase::prefill: return "prefill";
    case Phase::decode: return "decode";
    case Phase::verify: return "verify";
    }
    return "invalid";
}
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* what) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error(std::string(what) + " size overflows uint64");
    return a * b;
}
std::uint64_t extent(std::uint64_t rows, std::uint64_t stride, std::uint64_t row_bytes,
                     const char* what) {
    if (stride < row_bytes) throw std::invalid_argument(std::string(what) + " row stride is smaller than logical row");
    const auto prefix = checked_mul(rows - 1, stride, what);
    if (row_bytes > UINT64_MAX - prefix) throw std::overflow_error(std::string(what) + " extent overflows uint64");
    return prefix + row_bytes;
}
struct Metadata {
    ProjectionRequest request;
    int ordinal;
    bool wmma;
};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
#error "Saga CUDA projection requires an SM89 device compilation target"
#endif

__global__ void baseline_kernel(const __nv_bfloat16* w, const __nv_bfloat16* x, float* y,
                                unsigned long long n, unsigned long long k, unsigned long long t,
                                unsigned long long ws, unsigned long long xs, unsigned long long ys) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long total = n * t;
    if (i >= total) return;
    const auto token = i / n, output = i % n;
    float sum = 0.0f;
    for (unsigned long long p = 0; p < k; ++p)
        sum = fmaf(__bfloat162float(w[output * ws + p]), __bfloat162float(x[token * xs + p]), sum);
    y[token * ys + output] = sum;
}

__global__ void wmma_kernel(const __nv_bfloat16* w, const __nv_bfloat16* x, float* y,
                            unsigned long long k,
                            unsigned long long ws, unsigned long long xs, unsigned long long ys) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned tile_n = blockIdx.x * 16, tile_t = blockIdx.y * 16;
    const unsigned lane = threadIdx.x & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (unsigned long long p = 0; p < k; p += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, w + tile_n * ws + p, static_cast<unsigned>(ws));
        wmma::load_matrix_sync(b, x + tile_t * xs + p, static_cast<unsigned>(xs));
        wmma::mma_sync(acc, a, b, acc);
    }
    __shared__ __align__(32) float tile[16 * 16];
    wmma::store_matrix_sync(tile, acc, 16, wmma::mem_row_major);
    __syncthreads();
    for (unsigned q = lane; q < 256; q += 32) {
        const unsigned r = q / 16, c = q % 16;
        y[(tile_t + c) * ys + tile_n + r] = tile[r * 16 + c];
    }
#else
    (void)w; (void)x; (void)y; (void)k; (void)ws; (void)xs; (void)ys;
#endif
}

void validate_view(void* ptr, std::uint64_t bytes, std::uint64_t required, const char* label,
                   int ordinal, std::uint64_t alignment) {
    if (!ptr) throw std::invalid_argument(std::string(label) + " pointer is null");
    if (bytes < required) throw std::invalid_argument(std::string(label) + " buffer extent is undersized");
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignment)
        throw std::invalid_argument(std::string(label) + " pointer violates declared alignment");
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    if (required > UINTPTR_MAX - address) throw std::overflow_error(std::string(label) + " address extent overflows uintptr_t");
    int current_device = -1;
    const auto current_status = cudaGetDevice(&current_device);
    if (current_status != cudaSuccess || (ordinal >= 0 && current_device != ordinal))
        throw std::invalid_argument(std::string(label) + " is not on the active CUDA device");
    cudaDeviceProp properties{};
    const auto property_status = cudaGetDeviceProperties(&properties, current_device);
    if (property_status != cudaSuccess || properties.major != 8 || properties.minor != 9)
        throw std::invalid_argument("BF16 projection requires an SM89 device");
    cudaPointerAttributes attr{};
    const auto status = cudaPointerGetAttributes(&attr, ptr);
    if (status != cudaSuccess) throw std::invalid_argument(std::string(label) + " is not a CUDA device pointer: " + cudaGetErrorString(status));
#if CUDART_VERSION >= 10000
    if (attr.type != cudaMemoryTypeDevice || attr.device != current_device || (ordinal >= 0 && attr.device != ordinal))
#else
    if (attr.memoryType != cudaMemoryTypeDevice || attr.device != current_device)
#endif
        throw std::invalid_argument(std::string(label) + " belongs to a different device or memory domain");
    CUdeviceptr allocation_base{};
    std::size_t allocation_bytes{};
    const auto range_status = cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                                                    static_cast<CUdeviceptr>(address));
    if (range_status != CUDA_SUCCESS) {
        const char* message = nullptr;
        (void)cuGetErrorString(range_status, &message);
        throw std::invalid_argument(std::string(label) + " is not within a CUDA allocation: " +
                                    (message ? message : "unknown CUDA driver error"));
    }
    const auto base = static_cast<std::uintptr_t>(allocation_base);
    if (address < base || address - base > allocation_bytes || required > allocation_bytes - (address - base))
        throw std::invalid_argument(std::string(label) + " extent exceeds its CUDA allocation");
}
void entry(const void* raw, const LaunchContext& ctx) {
    const auto& m = *static_cast<const Metadata*>(raw);
    if (ctx.bindings.size() != 3) throw std::invalid_argument("BF16 projection requires bindings [weights, activations, output]");
    const auto n = m.request.shape.outputs, k = m.request.shape.inputs, t = m.request.shape.tokens;
    if (!n || !k || !t) throw std::invalid_argument("projection dimensions must all be nonzero");
    if (!m.request.alignment || (m.request.alignment & (m.request.alignment - 1)))
        throw std::invalid_argument("projection alignment must be a nonzero power of two");
    const auto ws = m.request.strides.weight_row_bytes / 2;
    const auto xs = m.request.strides.activation_row_bytes / 2;
    const auto ys = m.request.strides.output_row_bytes / 4;
    if (m.request.strides.weight_row_bytes % 2 || m.request.strides.activation_row_bytes % 2 || m.request.strides.output_row_bytes % 4)
        throw std::invalid_argument("projection byte stride is not element aligned");
    const auto wb = extent(n, m.request.strides.weight_row_bytes, checked_mul(k, 2, "weights"), "weights");
    const auto xb = extent(t, m.request.strides.activation_row_bytes, checked_mul(k, 2, "activations"), "activations");
    const auto yb = extent(t, m.request.strides.output_row_bytes, checked_mul(n, 4, "output"), "output");
    const auto w = ctx.bindings[0], x = ctx.bindings[1], y = ctx.bindings[2];
    const auto input_alignment = m.wmma ? std::max<std::uint64_t>(m.request.alignment, 32) : m.request.alignment;
    validate_view(w.address, w.bytes, wb, "weights", m.ordinal, input_alignment);
    validate_view(x.address, x.bytes, xb, "activations", m.ordinal, input_alignment);
    validate_view(y.address, y.bytes, yb, "output", m.ordinal, m.wmma ? 32 : alignof(float));
    const auto overlaps = [](const BufferView& a, std::uint64_t ae, const BufferView& b, std::uint64_t be) {
        const auto ap = reinterpret_cast<std::uintptr_t>(a.address), bp = reinterpret_cast<std::uintptr_t>(b.address);
        if (ae > UINTPTR_MAX - ap || be > UINTPTR_MAX - bp) throw std::overflow_error("buffer address range overflows uintptr_t");
        return ap < bp + be && bp < ap + ae;
    };
    if (overlaps(w, wb, x, xb) || overlaps(w, wb, y, yb) || overlaps(x, xb, y, yb))
        throw std::invalid_argument("projection input/output buffers must not alias");
    if (ctx.execution_lane == 0) throw std::invalid_argument("CUDA projection requires an explicit executor stream lane");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(ctx.execution_lane));
    if (m.wmma) {
        if (ws > UINT_MAX || xs > UINT_MAX || ys > UINT_MAX)
            throw std::invalid_argument("WMMA leading dimension exceeds supported CUDA range");
        wmma_kernel<<<dim3((n + 15) / 16, (t + 15) / 16), 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.address), static_cast<const __nv_bfloat16*>(x.address),
            static_cast<float*>(y.address), k, ws, xs, ys);
    } else {
        const auto count = checked_mul(n, t, "projection grid");
        if (count > UINT_MAX * 256ULL) throw std::invalid_argument("projection exceeds CUDA grid bounds");
        baseline_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.address), static_cast<const __nv_bfloat16*>(x.address),
            static_cast<float*>(y.address), n, k, t, ws, xs, ys);
    }
    const auto launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) throw std::runtime_error(std::string("projection launch failed: ") + cudaGetErrorString(launch_error));
}

class Projection final : public IKernelImplementation {
public:
    Projection(bool tiled, Phase phase) : tiled_(tiled), phase_(phase) {}
    KernelDescriptor descriptor() const override {
        KernelDescriptor d;
        d.id = std::string(tiled_ ? wmma_id : baseline_id) + "." + phase_suffix(phase_);
        d.operation = Operation::projection; d.phase = phase_; d.weights = WeightEncoding::bf16;
        d.activation = ActivationFormat::bf16; d.layout = Layout::contiguous;
        d.numerics = NumericalContract::fp32_accumulate;
        d.shape = tiled_ ? ShapeConstraint{{16, 65536, 16}, {16, 65536, 16}, {16, 65536, 16}}
                         : ShapeConstraint{{1, 65536, 1}, {1, 65536, 1}, {1, 65536, 1}};
        d.required_physical = {tiled_ ? 32u : 2u, tiled_ ? 32u : 2u, tiled_ ? 64u : 4u, tiled_ ? 32u : 2u, 0};
        d.device = {DeviceKind::accelerator, "sm89", 89, 89, 0};
        d.default_priority = tiled_ ? 10 : 0;
        return d;
    }
    bool supports(const KernelRequest& r, const DeviceCaps& c) const noexcept override {
        return c.kind == DeviceKind::accelerator && c.architecture == "sm89" && c.architecture_version == 89 &&
            r.operation == Operation::projection && r.phase == phase_ && r.weights == WeightEncoding::bf16 &&
            r.activation == ActivationFormat::bf16 && r.layout == Layout::contiguous &&
            r.numerics == NumericalContract::fp32_accumulate && !r.kv_pair &&
            (!tiled_ || (r.shape.outputs % 16 == 0 && r.shape.inputs % 16 == 0 && r.shape.tokens % 16 == 0 &&
             r.physical.weight_row_stride % 32 == 0 && r.physical.activation_row_stride % 32 == 0 &&
             r.physical.output_row_stride % 64 == 0 && r.physical.alignment >= 32));
    }
    PreparedKernel prepare(const KernelRequest& r, KernelPreparationContext& c) const override {
        if (!supports(r, c.device)) throw KernelUnavailable("unsupported BF16 projection contract");
        auto meta = std::make_shared<const Metadata>(Metadata{{{r.shape.outputs, r.shape.inputs, r.shape.tokens},
            {r.physical.weight_row_stride, r.physical.activation_row_stride, r.physical.output_row_stride},
            r.phase, r.physical.alignment}, -1, tiled_});
        return PreparedKernel(&entry, std::move(meta), r, {});
    }
private: bool tiled_; Phase phase_;
};
}

void register_bf16_projection_kernels(KernelRegistry& registry) {
    for (auto phase : {Phase::prefill, Phase::decode, Phase::verify}) {
        registry.add(std::make_shared<Projection>(false, phase));
        registry.add(std::make_shared<Projection>(true, phase));
    }
}
KernelRuntimeInfo query_kernel_runtime_info(std::string_view id) {
    cudaFuncAttributes attributes{};
    int active_blocks = 0;
    if (id.starts_with(baseline_id) && id.size() > std::char_traits<char>::length(baseline_id) &&
        id[std::char_traits<char>::length(baseline_id)] == '.') {
        const auto a = cudaFuncGetAttributes(&attributes, baseline_kernel);
        if (a != cudaSuccess) throw std::runtime_error(std::string("cudaFuncGetAttributes(baseline): ") + cudaGetErrorString(a));
        const auto o = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, baseline_kernel, 256, 0);
        if (o != cudaSuccess) throw std::runtime_error(std::string("cudaOccupancyMaxActiveBlocksPerMultiprocessor(baseline): ") + cudaGetErrorString(o));
    } else if (id.starts_with(wmma_id) && id.size() > std::char_traits<char>::length(wmma_id) &&
               id[std::char_traits<char>::length(wmma_id)] == '.') {
        const auto a = cudaFuncGetAttributes(&attributes, wmma_kernel);
        if (a != cudaSuccess) throw std::runtime_error(std::string("cudaFuncGetAttributes(WMMA): ") + cudaGetErrorString(a));
        const auto o = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, wmma_kernel, 32, 0);
        if (o != cudaSuccess) throw std::runtime_error(std::string("cudaOccupancyMaxActiveBlocksPerMultiprocessor(WMMA): ") + cudaGetErrorString(o));
    } else throw std::invalid_argument("unknown CUDA projection implementation ID");
    return {std::string(id), attributes.numRegs, attributes.sharedSizeBytes,
        attributes.localSizeBytes, attributes.maxThreadsPerBlock,
        attributes.binaryVersion, attributes.ptxVersion, active_blocks};
}
std::string cuda_compiler_version() {
#if defined(__CUDACC_VER_MAJOR__)
    return std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) +
        "." + std::to_string(__CUDACC_VER_BUILD__);
#else
    return "unknown";
#endif
}
KernelRequest make_bf16_projection_request(const ProjectionRequest& p, std::string_view) {
    KernelRequest r; r.operation = Operation::projection; r.phase = p.phase;
    r.weights = WeightEncoding::bf16; r.activation = ActivationFormat::bf16;
    r.layout = Layout::contiguous; r.numerics = NumericalContract::fp32_accumulate;
    r.shape = {p.shape.outputs, p.shape.inputs, p.shape.tokens};
    r.physical = {p.strides.weight_row_bytes, p.strides.activation_row_bytes,
                  p.strides.output_row_bytes, p.alignment, 0};
    return r;
}
} // namespace saga::backend::cuda
