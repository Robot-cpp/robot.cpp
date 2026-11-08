// Candidate099: per-lane temporary scores retained in registers; original64
// physical threads; retain the accepted F32 arithmetic and other launches.
#include "ops.h"
#include "common.cuh"
#include "convert.cuh"
#include <utility>

namespace robotcpp::bitvla {
namespace {
template <typename T> static __device__ __forceinline__ float t2f32(T value) { return (float)value; }
struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    float score_scale;
};


template<bool scale_scores>
static __device__ __forceinline__ float softmax_score_input(float input, float scale) {
    if constexpr (!scale_scores) return input;
    const float input_value=input;
    float scaled;
    // Same instruction/operand order as accepted score_scale_cuda, not FMA contraction of a different expression.
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(scaled) : "f"(scale), "f"(input_value), "f"(0.0f));
    return scaled;
}

template <bool use_shared, int ncols_template, int block_size_template, typename T, bool scale_scores = false>
static __global__ void softmax_f32_kernel(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid  = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    //TODO: noncontigous inputs/outputs
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_softmax_f32_kernel[];
    float * buf_iw = data_softmax_f32_kernel; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = sinks ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = softmax_score_input<scale_scores>(x[col], p.score_scale)*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = block_reduce<block_reduce_method::MAX, block_size_template>(max_val, buf_iw);

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = block_reduce<block_reduce_method::SUM, block_size_template>(tmp, buf_iw);

    if (sinks) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}

// Sixteen virtual 64-thread groups preserve the original 1024 logical-thread
// reduction tree. Never sum virtual per-thread values before the warp tree.
template <block_reduce_method method>
static __device__ __forceinline__ float reduce_logical1024(const float (&values)[16], float * shared_vals) {
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int quarter = 0; quarter < 16; ++quarter) {
        const float partial = block_reduce_policy<method, float>::reduce(values[quarter]);
        if (lane == 0) shared_vals[warp + quarter * 2] = partial;
    }
    __syncthreads();
    // Same32 ordered logical-warp partials; each virtual lane retains its XOR tree.
    return block_reduce_policy<method, float>::reduce(shared_vals[lane]);
}

static bool softmax_public_parameters_selected(int64_t width) { return WARP_SIZE == 32 && width > 512 && width < 1024; }

template <typename T, bool scale_scores = false>
static __global__ void softmax_logical1024_fold16_public_parameters_f32_kernel(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int tid = threadIdx.x, ncols = p.ncols;
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    x += int64_t(rowx) * ncols;
    dst += int64_t(rowx) * ncols;
    extern __shared__ float data_softmax_logical1024[];
    float * buf_iw = data_softmax_logical1024;
    // A logical lane consumes only the values it wrote; retain these in registers.
    float vals[16];
    float maximum[16] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    #pragma unroll
    for (int half = 0; half < 16; ++half) {
        const int col = tid + half * 64;
        if (col < ncols) {
            float val;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(val)
                : "f"(softmax_score_input<scale_scores>(x[col], p.score_scale)), "f"(p.scale), "f"(0.0f));
            vals[half] = val;
            maximum[half] = max(maximum[half], val);
        }
    }
    const float max_val = reduce_logical1024<block_reduce_method::MAX>(maximum, buf_iw);
    // Finish all readers of the max partials before the shared scratch is reused.
    __syncthreads();
    float partial_sum[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int half = 0; half < 16; ++half) {
        const int col = tid + half * 64;
        if (col < ncols) {
            const float val = expf(vals[half] - max_val);
            partial_sum[half] += val;
            vals[half] = val;
        }
    }
    float tmp = reduce_logical1024<block_reduce_method::SUM>(partial_sum, buf_iw);
    const float inv_sum = 1.0f / tmp;
    #pragma unroll
    for (int half = 0; half < 16; ++half) {
        const int col = tid + half * 64;
        if (col < ncols) {
            const float value = vals[half] * inv_sum;
            dst[col] = value;
        }
    }
}

template <typename T, bool scale_scores = false>
static __global__ void softmax_logical1024_fold16_regvals_f32_kernel(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int tid = threadIdx.x, ncols = p.ncols;
    const int64_t i03 = blockIdx.z, i02 = blockIdx.y, i01 = blockIdx.x;
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    const int64_t i11 = i01, i12 = i02 % p.ne12, i13 = i03 % p.ne13;
    x += int64_t(rowx) * ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst += int64_t(rowx) * ncols;
    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);
    extern __shared__ float data_softmax_logical1024[];
    float * buf_iw = data_softmax_logical1024;
    // A logical lane consumes only the values it wrote; retain these in registers.
    float vals[16];
    float maximum[16] = {sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY, sinks ? sinks[i02] : -INFINITY};
    #pragma unroll
    for (int half = 0; half < 16; ++half) {
        const int col = tid + half * 64;
        if (col < ncols) {
            const float val = softmax_score_input<scale_scores>(x[col], p.score_scale)*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);
            vals[half] = val;
            maximum[half] = max(maximum[half], val);
        }
    }
    const float max_val = reduce_logical1024<block_reduce_method::MAX>(maximum, buf_iw);
    // Finish all readers of the max partials before the shared scratch is reused.
    __syncthreads();
    float partial_sum[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int half = 0; half < 16; ++half) {
        const int col = tid + half * 64;
        if (col < ncols) {
            const float val = expf(vals[half] - max_val);
            partial_sum[half] += val;
            vals[half] = val;
        }
    }
    float tmp = reduce_logical1024<block_reduce_method::SUM>(partial_sum, buf_iw);
    if (sinks) tmp += expf(sinks[i02] - max_val);
    const float inv_sum = 1.0f / tmp;
    #pragma unroll
    for (int half = 0; half < 16; ++half) {
        const int col = tid + half * 64;
        if (col < ncols) {
            const float value = vals[half] * inv_sum;
            dst[col] = value;
        }
    }
}

template<bool scale_scores, int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((softmax_f32_kernel<true, ncols, block, T, scale_scores>), smpbo);
            softmax_f32_kernel<true, ncols, block, T, scale_scores><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT((softmax_f32_kernel<true, 0, 0, T, scale_scores>), smpbo);
    softmax_f32_kernel<true, 0, 0, T, scale_scores><<<block_nums, block_dims, nbytes_shared, stream>>>(x, mask, sinks, dst, p);
}

} // namespace

template<bool scale_scores>
static void softmax_dispatch(ggml_tensor * dst, void * stream, void * userdata) {
    const auto * input = dst->src[0];
    const int64_t width = input->ne[0];
    GGML_ASSERT(ggml_is_contiguous(input) && width <= 4096);
    int threads = WARP_SIZE;
    while (threads < width && threads < 1024) threads *= 2;
    const dim3 block(threads,1,1), grid(input->ne[1],input->ne[2],input->ne[3]);
    const size_t shared_bytes = (GGML_PAD(width,WARP_SIZE)+WARP_SIZE)*sizeof(float);
    soft_max_params params{};
    params.nheads = input->ne[2];
    params.n_head_log2 = 1u << (uint32_t)floorf(log2f((float)input->ne[2]));
    params.ncols = width;
    params.nrows_x = ggml_nrows(input);
    params.nrows_y = input->ne[1];
    params.ne00 = width;params.ne01 = input->ne[1];params.ne02 = input->ne[2];params.ne03 = input->ne[3];
    params.nb11 = params.nb12 = params.nb13 = 1;
    params.ne12 = params.ne13 = 1;
    params.score_scale = scale_scores ? *static_cast<const float *>(userdata) : 1.0f;
    params.scale = 1.0f;params.max_bias = 0.0f;params.m0 = params.m1 = 1.0f;
    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo);
    // Only the original generic1024 logical-thread range uses the folded
    // physical block. Power-of-two and all other widths retain the old kernel.
    if (softmax_public_parameters_selected(width)) {
        CUDA_SET_SHARED_MEMORY_LIMIT((softmax_logical1024_fold16_public_parameters_f32_kernel<float, scale_scores>),
            ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo);
        softmax_logical1024_fold16_public_parameters_f32_kernel<float, scale_scores><<<grid,dim3(64,1,1),WARP_SIZE*sizeof(float),static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(input->data),static_cast<const float *>(nullptr),nullptr,
            static_cast<float *>(dst->data),params);
        return;
    }
    if (WARP_SIZE == 32 && width > 512 && width < 1024) {
        CUDA_SET_SHARED_MEMORY_LIMIT((softmax_logical1024_fold16_regvals_f32_kernel<float, scale_scores>),
            ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo);
        softmax_logical1024_fold16_regvals_f32_kernel<float, scale_scores><<<grid,dim3(64,1,1),WARP_SIZE*sizeof(float),static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(input->data),static_cast<const float *>(nullptr),nullptr,
            static_cast<float *>(dst->data),params);
        return;
    }
    launch_soft_max_kernels<scale_scores,32,64,128,256,512,1024,2048,4096>(
        static_cast<const float *>(input->data),static_cast<const float *>(nullptr),nullptr,
        static_cast<float *>(dst->data),params,static_cast<cudaStream_t>(stream),block,grid,shared_bytes);
}
void softmax_f32_cuda(ggml_tensor * dst, void * stream, void * userdata) { softmax_dispatch<false>(dst,stream,userdata); }
void softmax_score_f32_cuda(ggml_tensor * dst, void * stream, void * userdata) { softmax_dispatch<true>(dst,stream,userdata); }
} // namespace robotcpp::bitvla
