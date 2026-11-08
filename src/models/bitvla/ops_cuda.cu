#include "ops.h"
#include "ggml-backend.h"

#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace robotcpp::bitvla {
namespace {
template <bool value_layout>
__global__ void bias_attention_layout_f32_kernel(const float * input, const float * bias, float * output,
        int64_t width, int64_t tokens, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t head = i / (width * tokens);
    const int64_t token = value_layout ? i % tokens : (i / width) % tokens;
    const int64_t col = value_layout ? (i / tokens) % width : i % width;
    const int64_t channels = count / tokens, channel = head * width + col;
    float sum;
    // Original ggml ADD is built with -use_fast_math: preserve its FTZ addition.
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(sum) : "f"(input[token * channels + channel]), "f"(bias[channel]));
    output[i] = sum;
}

__global__ void contiguous_layout_f32_kernel(const char * input, float * output,
        int64_t ne0, int64_t ne1, int64_t ne2, size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    const int64_t row = blockIdx.x;
    const int64_t i1 = row % ne1, i2 = (row / ne1) % ne2, i3 = row / (ne1 * ne2);
    input += i1 * nb1 + i2 * nb2 + i3 * nb3;
    output += row * ne0;
    for (int64_t col = threadIdx.x; col < ne0; col += blockDim.x) {
        const float value = *reinterpret_cast<const float *>(input + col * nb0);
        output[col] = value;
    }
}

__device__ __forceinline__ float context_f32_value(const char * input,int token,int col,int dim,size_t nb0,size_t nb1,size_t nb2) {
    const float value=*reinterpret_cast<const float *>(input+size_t(token)*nb2+size_t(col%dim)*nb0+size_t(col/dim)*nb1);
    return value;
}
__device__ __forceinline__ float activation_scale_exact_f32(float maximum) {
    return __double2float_rn(__ddiv_rn(127.0, fmax(static_cast<double>(maximum), 1e-5)));
}
__global__ void context_layout_i2s_encode_kernel(const char * input, int8_t * output, int logical, int padded, int dim, size_t nb0, size_t nb1, size_t nb2) {
    __shared__ float maxima[8], scale;
    __shared__ int sums[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, token = blockIdx.x;
    float maximum = 0;
    for (int i = threadIdx.x; i < logical; i += blockDim.x) maximum = fmaxf(maximum, fabsf(context_f32_value(input,token,i,dim,nb0,nb1,nb2)));
    for (int step = 16; step; step /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, step));
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        maximum = 0;
        for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, maxima[i]);
        scale = activation_scale_exact_f32(maximum);
    }
    __syncthreads();
    auto * row = output + token * (padded + 8);
    int sum = 0;
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        const int value = i < logical ? max(-128, min(127, __float2int_rn(__fmul_rn(context_f32_value(input,token,i,dim,nb0,nb1,nb2), scale)))) : 0;
        row[i] = value;
        sum += value;
    }
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int i = 0; i < 8; ++i) total += sums[i];
        *reinterpret_cast<float *>(row + padded) = scale;
        *reinterpret_cast<int *>(row + padded + 4) = total;
    }
}


template <bool packed_attention>
__global__ void rope_f32_kernel(const float * input, const float * cosine, const float * sine,
                                      float * output, int width, int row_width, int tokens) {
    const int j = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= row_width) return;
    const int col = j % width, half = width / 2;
    const int64_t i = int64_t(blockIdx.y) * row_width + j;
    const int64_t cache = int64_t(blockIdx.y) * width + col;
    const float rotated = col < half ? -input[i + half] : input[i - half];
    float left, right, result;
    // Keep the original two multiply rounds and the final add round; no FMA.
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(left) : "f"(input[i]), "f"(cosine[cache]));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(right) : "f"(rotated), "f"(sine[cache]));
    left = left;
    right = right;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    const int64_t destination = packed_attention
        ? (int64_t(j / width) * tokens + blockIdx.y) * width + col : i;
    output[destination] = result;
}

__global__ void scale_attention_f32_kernel(const float * input, float * output, int64_t count, float scale) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float f32_value = input[i];
        // ggml's SCALE is built with -use_fast_math: preserve its FTZ FMA.
        float scaled;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(scaled)
            : "f"(scale), "f"(f32_value), "f"(0.0f));
        output[i] = scaled;
    }
}
__global__ void gated_relu_square_f32_kernel(const float * gate, const float * up, float * output, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = fmaxf(gate[i], 0.0f);
        float squared, product;
        // Preserve both original F32 fast-math multiplications.
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(squared) : "f"(value), "f"(value));
        const float f32_value = squared;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(f32_value), "f"(up[i]));
        output[i] = product;
    }
}
// Host RGB bytes -> original exact lookup values in CHW layout.
__global__ void vision_rgb_u8_f32_kernel(const uint8_t * rgb, const float * lookup, float * output, int64_t plane) {
    const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < plane) {
        output[p] = lookup[rgb[3*p]];
        output[plane+p] = lookup[256+rgb[3*p+1]];
        output[2*plane+p] = lookup[512+rgb[3*p+2]];
    }
}

// Two RGB pixels; memcpy2 only reinterprets their six input bytes.
__global__ void vision_rgb_u16_pair_f32_232_kernel(const uint8_t * rgb, const float * lookup, float * output, int64_t plane) {
    const int64_t first = 2 * (int64_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (first >= plane) return;
    uint16_t a,b,c;
    memcpy(&a,rgb+3*first,2);
    memcpy(&b,rgb+3*first+2,2);
    memcpy(&c,rgb+3*first+4,2);
    output[first] = lookup[a & 255];
    output[plane+first] = lookup[256+(a >> 8)];
    output[2*plane+first] = lookup[512+(b & 255)];
    output[first+1] = lookup[b >> 8];
    output[plane+first+1] = lookup[256+(c & 255)];
    output[2*plane+first+1] = lookup[512+(c >> 8)];
}

__host__ __forceinline__ bool vision_rgb_u16_pair_f32_232_select(const ggml_tensor * dst) {
    if (!dst || !dst->src[0] || !dst->src[1]) return false;
    const auto * rgb=dst->src[0]; const auto * table=dst->src[1];
    const int64_t side=dst->ne[0];
    if (side<=0 || side>1024 || side%2 || dst->ne[1]!=side || dst->ne[2]!=3 || dst->ne[3]!=1 || dst->type!=GGML_TYPE_F32) return false;
    const int64_t plane=side*side;
    if (rgb->type!=GGML_TYPE_I8 || rgb->ne[0]!=3*plane || rgb->ne[1]!=1 || rgb->ne[2]!=1 || rgb->ne[3]!=1) return false;
    if (table->type!=GGML_TYPE_F32 || table->ne[0]!=256 || table->ne[1]!=3 || table->ne[2]!=1 || table->ne[3]!=1) return false;
    if (!ggml_is_contiguous(rgb) || !ggml_is_contiguous(table) || !ggml_is_contiguous(dst)) return false;
    const uintptr_t limit=std::numeric_limits<uintptr_t>::max(), out=reinterpret_cast<uintptr_t>(dst->data);
    const uintptr_t r=reinterpret_cast<uintptr_t>(rgb->data), l=reinterpret_cast<uintptr_t>(table->data);
    const size_t bytes=size_t(plane)*12, rgb_bytes=size_t(plane)*3;
    if (!out || !r || !l || out%4 || r%2 || l%4 || out>limit-bytes || r>limit-rgb_bytes || l>limit-3072) return false;
    if (!(r+rgb_bytes<=out || out+bytes<=r) || !(l+3072<=out || out+bytes<=l)) return false;
    return true;
}

__global__ void build_rope_cache(float * cosine, float * sine, size_t count, int width) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int dim = i % (width / 2), position = i / width;
    const float angle = position * (1.0f / powf(500000.0f, 2.0f * dim / width));
    cosine[i] = cosf(angle);
    sine[i] = sinf(angle);
}

// One thread owns a row, matching the CPU's left-to-right double accumulation.
__global__ void input_rms_cpu_order(const float * input, float * output, int64_t width, int64_t rows, float eps) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float * x = input + row * width;
    double sum = 0.0;
    for (int64_t col = 0; col < width; ++col) {
        sum = __dadd_rn(sum, double(__fmul_rn(x[col], x[col])));
    }
    const float mean = __double2float_rn(__ddiv_rn(sum, double(width)));
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    for (int64_t col = 0; col < width; ++col) output[row * width + col] = __fmul_rn(x[col], scale);
}
} // namespace

void rms_norm_cpu_order_cuda(ggml_tensor * dst, void * stream, void * userdata) {
    const int64_t rows = ggml_nrows(dst);
    input_rms_cpu_order<<<(rows + 31) / 32, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<float *>(dst->data),
        dst->ne[0], rows, *static_cast<const float *>(userdata));
}

template <bool value_layout>
void launch_bias_attention_layout(ggml_tensor * dst, void * stream) {
    const auto * input = dst->src[0];
    const int64_t count = ggml_nelements(input), width = input->ne[0] / dst->ne[2];
    bias_attention_layout_f32_kernel<value_layout><<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(input->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<float *>(dst->data), width, input->ne[1], count);
}

void bias_query_key_layout_cuda(ggml_tensor * dst, void * stream, void *) {
    launch_bias_attention_layout<false>(dst, stream);
}

void bias_value_layout_cuda(ggml_tensor * dst, void * stream, void *) {
    launch_bias_attention_layout<true>(dst, stream);
}

void contiguous_layout_f32_cuda(ggml_tensor * dst, void * stream, void *) {
    const auto * input = dst->src[0];
    contiguous_layout_f32_kernel<<<ggml_nrows(dst), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const char *>(input->data), static_cast<float *>(dst->data),
        input->ne[0], input->ne[1], input->ne[2], input->nb[0], input->nb[1], input->nb[2], input->nb[3]);
}

template <bool packed_attention>
void launch_rope_f32(ggml_tensor * dst, void * stream) {
    const auto * input = dst->src[0];
    const int width = static_cast<int>(input->ne[0]);
    const int row_width = static_cast<int>(input->ne[0] * input->ne[1]);
    const int tokens = static_cast<int>(input->ne[2]);
    const dim3 blocks((row_width + 255) / 256, tokens);
    rope_f32_kernel<packed_attention><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(input->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<const float *>(dst->src[2]->data), static_cast<float *>(dst->data), width, row_width, tokens);
}

void rope_f32_cuda(ggml_tensor * dst, void * stream, void *) {
    launch_rope_f32<false>(dst, stream);
}

void rope_attention_layout_cuda(ggml_tensor * dst, void * stream, void *) {
    launch_rope_f32<true>(dst, stream);
}

void score_scale_cuda(ggml_tensor * dst, void * stream, void * userdata) {
    const int64_t count = ggml_nelements(dst);
    scale_attention_f32_kernel<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<float *>(dst->data),
        count, *static_cast<const float *>(userdata));
}

void vision_rgb_u8_f32_cuda(ggml_tensor * dst, void * stream, void *) {
    if (vision_rgb_u16_pair_f32_232_select(dst)) {
        const int64_t pair_plane=dst->ne[0]*dst->ne[1];
        vision_rgb_u16_pair_f32_232_kernel<<<(pair_plane+255)/256,128,0,static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t *>(dst->src[0]->data),static_cast<const float *>(dst->src[1]->data),
            static_cast<float *>(dst->data),pair_plane);
        return;
    }
    const int64_t plane = dst->ne[0] * dst->ne[1];
    vision_rgb_u8_f32_kernel<<<(plane + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<float *>(dst->data), plane);
}

void gated_relu_square_f32_cuda(ggml_tensor * dst, void * stream, void *) {
    const int64_t count = ggml_nelements(dst);
    gated_relu_square_f32_kernel<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<float *>(dst->data), count);
}

bool rope_cache_cuda(ggml_tensor * cosine, ggml_tensor * sine) {
    const size_t count = ggml_nelements(cosine);
    build_rope_cache<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<float *>(cosine->data), static_cast<float *>(sine->data), count, cosine->ne[0]);
    const cudaError_t result = cudaStreamSynchronize(cudaStreamPerThread);
    if (result != cudaSuccess) std::fprintf(stderr, "BitVLA CUDA RoPE cache: %s\n", cudaGetErrorString(result));
    return result == cudaSuccess;
}

static bool encode_disjoint_span(const void * x,size_t nx,const void * y,size_t ny) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(x),b=reinterpret_cast<uintptr_t>(y),limit=~uintptr_t(0);
    if(!a || !b || !nx || !ny || nx>limit-a || ny>limit-b)return false;
    return a+nx<=b || b+ny<=a;
}
void contiguous_layout_i2s_encode_cuda(ggml_tensor * dst,void * stream,void *) {
    const auto * input=dst->src[0];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8 && input->ne[0]>0 && input->ne[0]<=1152 && input->ne[1]>0 && input->ne[1]<=1152);
    GGML_ASSERT(input->ne[0]*input->ne[1]==1152 && input->ne[2]>0 && input->ne[2]<=2147483647/1160 && input->ne[3]==1 && dst->ne[0]==1160 && dst->ne[1]==input->ne[2] && dst->ne[2]==1 && dst->ne[3]==1 && ggml_is_contiguous(dst));
    GGML_ASSERT(input->nb[0]==4 && input->nb[2]==size_t(input->ne[0])*4 && input->nb[1]==input->nb[2]*size_t(input->ne[2]));
    GGML_ASSERT(reinterpret_cast<uintptr_t>(input->data)%4==0 && reinterpret_cast<uintptr_t>(dst->data)%4==0);
    const uint64_t bytes=uint64_t(1152)*uint64_t(input->ne[2])*4;
    GGML_ASSERT(bytes<=uint64_t(~uintptr_t(0)) && encode_disjoint_span(input->data,size_t(bytes),dst->data,ggml_nbytes(dst)));
    context_layout_i2s_encode_kernel<<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const char *>(input->data),static_cast<int8_t *>(dst->data),int(dst->ne[0]-8),int(dst->ne[0]-8),int(input->ne[0]),input->nb[0],input->nb[1],input->nb[2]);
}
__global__ void f32_i2s_encode4304_kernel(const float * input,int8_t * output,int logical,int padded) {
    __shared__ float maxima[8], scale;
    __shared__ int sums[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, token = blockIdx.x;
    float maximum = 0;
    for (int i = threadIdx.x; i < logical; i += blockDim.x) maximum = fmaxf(maximum, fabsf(input[token * logical + i]));
    for (int step = 16; step; step /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, step));
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        maximum = 0;
        for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, maxima[i]);
        scale = activation_scale_exact_f32(maximum);
    }
    __syncthreads();
    auto * row = output + token * (padded + 8);
    int sum = 0;
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        const int value = i < logical ? max(-128, min(127, __float2int_rn(__fmul_rn(input[token * logical + i], scale)))) : 0;
        row[i] = value;
        sum += value;
    }
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int i = 0; i < 8; ++i) total += sums[i];
        *reinterpret_cast<float *>(row + padded) = scale;
        *reinterpret_cast<int *>(row + padded + 4) = total;
    }
}

void f32_i2s_encode_cuda(ggml_tensor * dst,void * stream,void *) {
    const auto * input=dst->src[0];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8 && ggml_is_contiguous(input) && ggml_is_contiguous(dst));
    GGML_ASSERT(input->ne[0]==4304 && input->ne[1]>0 && input->ne[1]<=2147483647/4360 && input->ne[2]==1 && input->ne[3]==1);
    GGML_ASSERT(dst->ne[0]==4360 && dst->ne[1]==input->ne[1] && dst->ne[2]==1 && dst->ne[3]==1);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(input->data)%4==0 && reinterpret_cast<uintptr_t>(dst->data)%4==0);
    const uint64_t bytes=uint64_t(4304)*uint64_t(input->ne[1])*4;
    GGML_ASSERT(bytes<=uint64_t(~uintptr_t(0)) && encode_disjoint_span(input->data,size_t(bytes),dst->data,ggml_nbytes(dst)));
    f32_i2s_encode4304_kernel<<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(input->data),static_cast<int8_t *>(dst->data),4304,4352);
}

template<int slots>
__global__ void cpu_order_rms_i2s_encode_kernel(const float * input,int8_t * output,int logical,int padded,const float * weight,float eps) {
    __shared__ float rms_scale;
    if(threadIdx.x==0) {
        const int64_t width=logical;const float * x=input+int64_t(blockIdx.x)*width;
    double sum = 0.0;
    for (int64_t col = 0; col < width; ++col) {
        sum = __dadd_rn(sum, double(__fmul_rn(x[col], x[col])));
    }
    const float mean = __double2float_rn(__ddiv_rn(sum, double(width)));
    rms_scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    __syncthreads();

    __shared__ float maxima[8], scale;
    __shared__ int sums[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, token = blockIdx.x;
    float values[slots];
    float maximum = 0;
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int i = threadIdx.x + slot * blockDim.x;
        values[slot] = 0.0f;
        if (i < logical) {
            const float normalized=__fmul_rn(input[token * logical + i],rms_scale);
            float product;asm("mul.rn.ftz.f32 %0, %1, %2;":"=f"(product):"f"(normalized),"f"(weight[i]));
            values[slot] = product;
            maximum = fmaxf(maximum, fabsf(values[slot]));
        }
    }
    for (int step = 16; step; step /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, step));
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        maximum = 0;
        for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, maxima[i]);
        scale = activation_scale_exact_f32(maximum);
    }
    __syncthreads();
    auto * row = output + token * (padded + 8);
    int sum = 0;
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int i = threadIdx.x + slot * blockDim.x;
        if (i < padded) {
            const int value = i < logical ? max(-128, min(127, __float2int_rn(__fmul_rn(values[slot], scale)))) : 0;
            row[i] = value;
            sum += value;
        }
    }
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int i = 0; i < 8; ++i) total += sums[i];
        *reinterpret_cast<float *>(row + padded) = scale;
        *reinterpret_cast<int *>(row + padded + 4) = total;
    }
}
void cpu_order_rms_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && weight->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8);
    GGML_ASSERT(input->ne[0]==2560 && input->ne[1]>0 && input->ne[1]<=INT32_MAX/2568 && input->ne[2]==1 && input->ne[3]==1);
    GGML_ASSERT(weight->ne[0]==2560 && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1);
    GGML_ASSERT(dst->ne[0]==2568 && dst->ne[1]==input->ne[1] && dst->ne[2]==1 && dst->ne[3]==1);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(weight) && ggml_is_contiguous(dst));
    GGML_ASSERT(reinterpret_cast<uintptr_t>(input->data)%4==0 && reinterpret_cast<uintptr_t>(weight->data)%4==0 && reinterpret_cast<uintptr_t>(dst->data)%4==0);
    GGML_ASSERT(encode_disjoint_span(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)) && encode_disjoint_span(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst)));
    cpu_order_rms_i2s_encode_kernel<10><<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(input->data),static_cast<int8_t *>(dst->data),2560,2560,static_cast<const float *>(weight->data),*static_cast<const float *>(userdata));
}

} // namespace robotcpp::bitvla
