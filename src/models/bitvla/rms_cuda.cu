// Keep the original ggml CUDA RMS reduction and fast-math compilation isolated
// in this translation unit. Custom activation intermediates and stores remain F32.
#include "ops.h"
#include "common.cuh"
#include "unary.cuh"

namespace robotcpp::bitvla {
namespace {
template <int block_size, bool affine = false>
__global__ void rms_norm_f32_kernel(const float * x, float * dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight) {
    const int nrows = gridDim.x, nchannels = gridDim.y;
    const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x;
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += block_size) {
        const float value = scale * x[col];
        const float normalized=value;
        if constexpr (affine) {
            float product;
            // Original ggml bin_bcast MUL is built with-use_fast_math.
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            dst[col]=product;
        } else dst[col]=normalized;
    }
}
// Retain each original F32 input in its logical lane through the unchanged
// 1024-thread reduction. No second global x read for the affine width2560.
template <int block_size, bool affine = false, int slots = 3>
__global__ void rms_norm_cached_input_f32_kernel(const float * x, float * dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight) {
    const int nrows = gridDim.x, nchannels = gridDim.y;
    const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x;
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;
    float values[slots];
    float tmp = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int col = tid + slot * block_size;
        if (col < ncols) {
            const float xi = x[col];
            values[slot] = xi;
            tmp += xi * xi;
        }
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int col = tid + slot * block_size;
        if (col >= ncols) continue;
        const float value = scale * values[slot];
        const float normalized=value;
        if constexpr (affine) {
            float product;
            // Original ggml bin_bcast MUL is built with-use_fast_math.
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            dst[col]=product;
        } else dst[col]=normalized;
    }
}
// Four original logical lanes per physical thread. Each logical lane keeps
// its original 1024-stride accumulation and original warp XOR tree. The 32
// original warp sums occupy the same shared indices as block_reduce<1024>.
__global__ void rms_affine_logical1024_fold4_cached_f32_kernel(const float * x, float * dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight) {
    const int nrows=gridDim.x,nchannels=gridDim.y,tid=threadIdx.x;
    const int row=blockIdx.x,channel=blockIdx.y,sample=blockIdx.z;
    x += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels+channel)*nrows+row)*ncols;
    float values[4][3], partial[4];
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        const int original_tid=tid+logical*256;
        float tmp=0.0f;
        #pragma unroll
        for (int slot=0;slot<3;++slot) {
            const int col=original_tid+slot*1024;
            if(col<ncols) {
                const float xi=x[col];values[logical][slot]=xi;
                tmp += xi*xi;
            }
        }
        partial[logical]=tmp;
    }
    extern __shared__ float s_sum[];
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        const float sum=warp_reduce_sum(partial[logical]);
        if((tid&31)==0)s_sum[tid/32+logical*8]=sum;
    }
    __syncthreads();
    const float tmp=warp_reduce_sum(s_sum[tid&31]);
    const float mean=tmp/ncols;
    const float scale=rsqrtf(mean+eps);
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        #pragma unroll
        for(int slot=0;slot<3;++slot) {
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float value=scale*values[logical][slot];
            const float normalized=value;
            float product;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            dst[col]=product;
        }
    }
}
__host__ __forceinline__ bool rms_fold_disjoint(const void * input,size_t input_bytes,const void * output,size_t output_bytes) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(input),b=reinterpret_cast<uintptr_t>(output),top=~uintptr_t(0);
    return input && output && input_bytes && output_bytes && input_bytes<=top-a && output_bytes<=top-b &&
        (a+input_bytes<=b || b+output_bytes<=a);
}

// Gate generation feeds the original cached RMS reduction, lane for lane.
template <int block_size, bool affine = true, int slots = 7>
__global__ void gated_rms_affine_original_tree_f32_kernel(const float * x, const float * up, float * dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight) {
    const int nrows = gridDim.x, nchannels = gridDim.y;
    const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x;
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    up += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;
    float values[slots];
    float tmp = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int col = tid + slot * block_size;
        if (col < ncols) {
            float value,squared,product;
            // Preserve fmax and the two original F32 FTZ multiplications.
            asm("max.f32 %0, %1, %2;" : "=f"(value) : "f"(x[col]), "f"(0.0f));
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(squared) : "f"(value), "f"(value));
            const float squared_f32=squared;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(squared_f32), "f"(up[col]));
            const float xi=product;
            values[slot] = xi;
            tmp += xi * xi;
        }
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int col = tid + slot * block_size;
        if (col >= ncols) continue;
        const float value = scale * values[slot];
        const float normalized=value;
        if constexpr (affine) {
            float product;
            // Original ggml bin_bcast MUL is built with-use_fast_math.
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            dst[col]=product;
        } else dst[col]=normalized;
    }
}

template <int block_size>
__global__ void layer_norm_affine_f32_kernel(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight, const float * bias) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float normalized = (x[col] - mean) * inv_std;
        float product, added;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(added) : "f"(product), "f"(bias[col]));
        dst[col] = added;
    }
}

// Map exactly 1024 logical lanes onto 256 physical threads.
// Each logical lane keeps the original input loop and float2 sum/square-FMA.
__global__ void layer_norm_affine_logical1024_fold4_f32_kernel(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight, const float * bias) {
    const int nrows=gridDim.x,nchannels=gridDim.y;
    const int row=blockIdx.x,channel=blockIdx.y,sample=blockIdx.z,tid=threadIdx.x;
    x+=sample*stride_sample+channel*stride_channel+row*stride_row;
    dst+=((sample*nchannels+channel)*nrows+row)*ncols;
    float2 partial[4];
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        partial[logical]=make_float2(0.0f,0.0f);
        for(int col=tid+logical*256;col<ncols;col+=1024){
            const float xi=x[col];
            partial[logical].x+=xi;
            partial[logical].y+=xi*xi;
        }
    }
    extern __shared__ float2 s_sum2[];
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        const float2 sum=warp_reduce_sum(partial[logical]);
        if((tid%WARP_SIZE)==0)s_sum2[tid/WARP_SIZE+logical*8]=sum;
    }
    __syncthreads();
    const float2 mean_var=warp_reduce_sum(s_sum2[tid%WARP_SIZE]);
    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        for(int col=tid+logical*256;col<ncols;col+=1024){
            const float normalized = (x[col] - mean) * inv_std;
            float product, added;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(added) : "f"(product), "f"(bias[col]));
            dst[col] = added;
        }
    }
}

__global__ void layer_norm_affine_logical1024_fold4_cached1152_f32_kernel(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * weight, const float * bias) {
    const int nrows=gridDim.x,nchannels=gridDim.y;
    const int row=blockIdx.x,channel=blockIdx.y,sample=blockIdx.z,tid=threadIdx.x;
    x+=sample*stride_sample+channel*stride_channel+row*stride_row;
    dst+=((sample*nchannels+channel)*nrows+row)*ncols;
    float values[4],tail_value=0.0f;
    float2 partial[4];
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        partial[logical]=make_float2(0.0f,0.0f);
        #pragma unroll
        for(int slot=0;slot<2;++slot){
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float xi=x[col];
            if(slot==0)values[logical]=xi;else tail_value=xi;
            partial[logical].x+=xi;
            partial[logical].y+=xi*xi;
        }
    }
    extern __shared__ float2 s_sum2[];
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        const float2 sum=warp_reduce_sum(partial[logical]);
        if((tid%WARP_SIZE)==0)s_sum2[tid/WARP_SIZE+logical*8]=sum;
    }
    __syncthreads();
    const float2 mean_var=warp_reduce_sum(s_sum2[tid%WARP_SIZE]);
    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        #pragma unroll
        for(int slot=0;slot<2;++slot){
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float xi=slot==0?values[logical]:tail_value;
            const float normalized = (xi - mean) * inv_std;
            float product, added;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(added) : "f"(product), "f"(bias[col]));
            dst[col] = added;
        }
    }
}

// Keep the RMS TU flags, but force the encoder's original non-FTZ instructions.
__device__ __forceinline__ float rms_encode_max_noftz(float a,float b) {float v;asm("max.f32 %0,%1,%2;":"=f"(v):"f"(a),"f"(b));return v;}
__device__ __forceinline__ float rms_encode_abs_noftz(float a) {float v;asm("abs.f32 %0,%1;":"=f"(v):"f"(a));return v;}
__device__ __forceinline__ float rms_encode_scale_exact(float maximum) {
    return __double2float_rn(__ddiv_rn(127.0,fmax(static_cast<double>(maximum),1e-5)));
}
__global__ void rms_affine_i2s_encode2560_kernel(const float * x,int8_t * output,
        const int ncols,const int64_t stride_row,const int64_t stride_channel,
        const int64_t stride_sample,const float eps,const float * weight) {
    const int nrows=gridDim.x,nchannels=gridDim.y,tid=threadIdx.x;
    const int row=blockIdx.x,channel=blockIdx.y,sample=blockIdx.z;
    x += sample*stride_sample + channel*stride_channel + row*stride_row;
    float values[4][3], partial[4];
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        const int original_tid=tid+logical*256;
        float tmp=0.0f;
        #pragma unroll
        for (int slot=0;slot<3;++slot) {
            const int col=original_tid+slot*1024;
            if(col<ncols) {
                const float xi=x[col];values[logical][slot]=xi;
                tmp += xi*xi;
            }
        }
        partial[logical]=tmp;
    }
    extern __shared__ float s_sum[];
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        const float sum=warp_reduce_sum(partial[logical]);
        if((tid&31)==0)s_sum[tid/32+logical*8]=sum;
    }
    __syncthreads();
    const float tmp=warp_reduce_sum(s_sum[tid&31]);
    const float mean=tmp/ncols;
    const float scale=rsqrtf(mean+eps);
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        #pragma unroll
        for(int slot=0;slot<3;++slot) {
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float value=scale*values[logical][slot];
            const float normalized=value;
            float product;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            values[logical][slot]=product;
        }
    }

    // The RMS shared32 scratch has no remaining readers or writers here.
    // Encoder scratch is separate so its first warp cannot race the RMS readers.
    __shared__ float maxima[8],activation_scale;__shared__ int sums[8];
    const int lane=tid%32,warp=tid/32,token=blockIdx.x;
    float maximum=0.0f;
    #pragma unroll
    for(int slot=0;slot<10;++slot)maximum=rms_encode_max_noftz(maximum,rms_encode_abs_noftz(values[slot%4][slot/4]));
    for(int step=16;step;step/=2)maximum=rms_encode_max_noftz(maximum,__shfl_down_sync(0xffffffff,maximum,step));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(tid==0){maximum=0.0f;for(int i=0;i<8;++i)maximum=rms_encode_max_noftz(maximum,maxima[i]);activation_scale=rms_encode_scale_exact(maximum);}
    __syncthreads();
    auto * encoded=output+int64_t(token)*(ncols+8);int sum=0;
    #pragma unroll
    for(int slot=0;slot<10;++slot){
        const int i=tid+slot*256;float scaled;int rounded;
        asm("mul.rn.f32 %0,%1,%2;":"=f"(scaled):"f"(values[slot%4][slot/4]),"f"(activation_scale));
        asm("cvt.rni.s32.f32 %0,%1;":"=r"(rounded):"f"(scaled));
        const int value=max(-128,min(127,rounded));encoded[i]=value;sum+=value;
    }
    for(int step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0)sums[warp]=sum;
    __syncthreads();
    if(tid==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];*reinterpret_cast<float *>(encoded+ncols)=activation_scale;*reinterpret_cast<int *>(encoded+ncols+4)=total;}
}

__global__ void permuted_rms_affine_i2s_encode2560_kernel(const char * x,int8_t * output,
        const int ncols,const int64_t stride_row,const int64_t stride_channel,
        const int64_t stride_sample,const float eps,const float * weight,const int dim,const size_t nb0,const size_t nb1) {
    const int nrows=gridDim.x,nchannels=gridDim.y,tid=threadIdx.x;
    const int row=blockIdx.x,channel=blockIdx.y,sample=blockIdx.z;
    x += sample*stride_sample + channel*stride_channel + row*stride_row;
    float values[4][3], partial[4];
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        const int original_tid=tid+logical*256;
        float tmp=0.0f;
        #pragma unroll
        for (int slot=0;slot<3;++slot) {
            const int col=original_tid+slot*1024;
            if(col<ncols) {
                const float xi=*reinterpret_cast<const float *>(x+size_t(col%dim)*nb0+size_t(col/dim)*nb1);values[logical][slot]=xi;
                tmp += xi*xi;
            }
        }
        partial[logical]=tmp;
    }
    extern __shared__ float s_sum[];
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        const float sum=warp_reduce_sum(partial[logical]);
        if((tid&31)==0)s_sum[tid/32+logical*8]=sum;
    }
    __syncthreads();
    const float tmp=warp_reduce_sum(s_sum[tid&31]);
    const float mean=tmp/ncols;
    const float scale=rsqrtf(mean+eps);
    #pragma unroll
    for (int logical=0;logical<4;++logical) {
        #pragma unroll
        for(int slot=0;slot<3;++slot) {
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float value=scale*values[logical][slot];
            const float normalized=value;
            float product;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            values[logical][slot]=product;
        }
    }

    // The RMS shared32 scratch has no remaining readers or writers here.
    // Encoder scratch is separate so its first warp cannot race the RMS readers.
    __shared__ float maxima[8],activation_scale;__shared__ int sums[8];
    const int lane=tid%32,warp=tid/32,token=blockIdx.x;
    float maximum=0.0f;
    #pragma unroll
    for(int slot=0;slot<10;++slot)maximum=rms_encode_max_noftz(maximum,rms_encode_abs_noftz(values[slot%4][slot/4]));
    for(int step=16;step;step/=2)maximum=rms_encode_max_noftz(maximum,__shfl_down_sync(0xffffffff,maximum,step));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(tid==0){maximum=0.0f;for(int i=0;i<8;++i)maximum=rms_encode_max_noftz(maximum,maxima[i]);activation_scale=rms_encode_scale_exact(maximum);}
    __syncthreads();
    auto * encoded=output+int64_t(token)*(ncols+8);int sum=0;
    #pragma unroll
    for(int slot=0;slot<10;++slot){
        const int i=tid+slot*256;float scaled;int rounded;
        asm("mul.rn.f32 %0,%1,%2;":"=f"(scaled):"f"(values[slot%4][slot/4]),"f"(activation_scale));
        asm("cvt.rni.s32.f32 %0,%1;":"=r"(rounded):"f"(scaled));
        const int value=max(-128,min(127,rounded));encoded[i]=value;sum+=value;
    }
    for(int step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0)sums[warp]=sum;
    __syncthreads();
    if(tid==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];*reinterpret_cast<float *>(encoded+ncols)=activation_scale;*reinterpret_cast<int *>(encoded+ncols+4)=total;}
}

template <int block_size, bool affine = true, int slots = 7>
__global__ void gated_rms_affine_i2s_encode6912_kernel(const float * x,const float * up,int8_t * output,
        const int ncols,const int64_t stride_row,const int64_t stride_channel,
        const int64_t stride_sample,const float eps,const float * weight) {
    const int nrows = gridDim.x, nchannels = gridDim.y;
    const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x;
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    up += sample * stride_sample + channel * stride_channel + row * stride_row;
    __shared__ float normalized_values[6912];
    float * dst=normalized_values;
    float values[slots];
    float tmp = 0.0f;
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int col = tid + slot * block_size;
        if (col < ncols) {
            float value,squared,product;
            // Preserve fmax and the two original F32 FTZ multiplications.
            asm("max.f32 %0, %1, %2;" : "=f"(value) : "f"(x[col]), "f"(0.0f));
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(squared) : "f"(value), "f"(value));
            const float squared_f32=squared;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(squared_f32), "f"(up[col]));
            const float xi=product;
            values[slot] = xi;
            tmp += xi * xi;
        }
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    #pragma unroll
    for (int slot = 0; slot < slots; ++slot) {
        const int col = tid + slot * block_size;
        if (col >= ncols) continue;
        const float value = scale * values[slot];
        const float normalized=value;
        if constexpr (affine) {
            float product;
            // Original ggml bin_bcast MUL is built with-use_fast_math.
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            dst[col]=product;
        } else dst[col]=normalized;
    }

    // Every original1024 thread participates in every CTA barrier below.
    // Keep the accepted sequence of F32 stages in immutable shared values.
    __shared__ float maxima[8],activation_scale;__shared__ int sums[8];
    const int lane=tid%32,warp=tid/32;
    __syncthreads();
    if(tid<256) {
        float maximum=0.0f;
        #pragma unroll
        for(int slot=0;slot<27;++slot)maximum=rms_encode_max_noftz(maximum,rms_encode_abs_noftz(dst[tid+slot*256]));
        for(int step=16;step;step/=2)maximum=rms_encode_max_noftz(maximum,__shfl_down_sync(0xffffffff,maximum,step));
        if(lane==0)maxima[warp]=maximum;
    }
    __syncthreads();
    if(tid==0){float maximum=0.0f;for(int i=0;i<8;++i)maximum=rms_encode_max_noftz(maximum,maxima[i]);activation_scale=rms_encode_scale_exact(maximum);}
    __syncthreads();
    auto * encoded=output+int64_t(row)*(ncols+8);
    if(tid<256) {
        int sum=0;
        #pragma unroll
        for(int slot=0;slot<27;++slot){
            const int i=tid+slot*256;float scaled;int rounded_int;
            asm("mul.rn.f32 %0,%1,%2;":"=f"(scaled):"f"(dst[i]),"f"(activation_scale));
            asm("cvt.rni.s32.f32 %0,%1;":"=r"(rounded_int):"f"(scaled));
            const int value=max(-128,min(127,rounded_int));encoded[i]=value;sum+=value;
        }
        for(int step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
        if(lane==0)sums[warp]=sum;
    }
    __syncthreads();
    if(tid==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];*reinterpret_cast<float *>(encoded+ncols)=activation_scale;*reinterpret_cast<int *>(encoded+ncols+4)=total;}
}

__global__ void layer_norm_affine_i2s_encode1152_kernel(const float * x,int8_t * output,const int ncols,
        const int64_t stride_row,const int64_t stride_channel,const int64_t stride_sample,
        const float eps,const float * weight,const float * bias) {
    const int nrows=gridDim.x,nchannels=gridDim.y;
    const int row=blockIdx.x,channel=blockIdx.y,sample=blockIdx.z,tid=threadIdx.x;
    x+=sample*stride_sample+channel*stride_channel+row*stride_row;
    float values[4],tail_value=0.0f;
    float2 partial[4];
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        partial[logical]=make_float2(0.0f,0.0f);
        #pragma unroll
        for(int slot=0;slot<2;++slot){
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float xi=x[col];
            if(slot==0)values[logical]=xi;else tail_value=xi;
            partial[logical].x+=xi;
            partial[logical].y+=xi*xi;
        }
    }
    extern __shared__ float2 s_sum2[];
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        const float2 sum=warp_reduce_sum(partial[logical]);
        if((tid%WARP_SIZE)==0)s_sum2[tid/WARP_SIZE+logical*8]=sum;
    }
    __syncthreads();
    const float2 mean_var=warp_reduce_sum(s_sum2[tid%WARP_SIZE]);
    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);
    #pragma unroll
    for(int logical=0;logical<4;++logical){
        #pragma unroll
        for(int slot=0;slot<2;++slot){
            const int col=tid+logical*256+slot*1024;
            if(col>=ncols)continue;
            const float xi=slot==0?values[logical]:tail_value;
            const float normalized = (xi - mean) * inv_std;
            float product, added;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(product) : "f"(normalized), "f"(weight[col]));
            asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(added) : "f"(product), "f"(bias[col]));
            if(slot==0)values[logical]=added;else tail_value=added;
        }
    }

    __shared__ float maxima[8],activation_scale;__shared__ int sums[8];
    const int lane=tid%32,warp=tid/32;float maximum=0.0f;
    #pragma unroll
    for(int slot=0;slot<5;++slot)if(tid+slot*256<ncols)maximum=rms_encode_max_noftz(maximum,rms_encode_abs_noftz(slot<4?values[slot]:tail_value));
    for(int step=16;step;step/=2)maximum=rms_encode_max_noftz(maximum,__shfl_down_sync(0xffffffff,maximum,step));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(tid==0){maximum=0.0f;for(int i=0;i<8;++i)maximum=rms_encode_max_noftz(maximum,maxima[i]);activation_scale=rms_encode_scale_exact(maximum);}
    __syncthreads();
    auto * encoded=output+int64_t(row)*(ncols+8);int sum=0;
    #pragma unroll
    for(int slot=0;slot<5;++slot){
        const int i=tid+slot*256;if(i>=ncols)continue;float scaled;int rounded;
        asm("mul.rn.f32 %0,%1,%2;":"=f"(scaled):"f"(slot<4?values[slot]:tail_value),"f"(activation_scale));
        asm("cvt.rni.s32.f32 %0,%1;":"=r"(rounded):"f"(scaled));
        const int value=max(-128,min(127,rounded));encoded[i]=value;sum+=value;
    }
    for(int step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0)sums[warp]=sum;
    __syncthreads();
    if(tid==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];*reinterpret_cast<float *>(encoded+ncols)=activation_scale;*reinterpret_cast<int *>(encoded+ncols+4)=total;}
}

__global__ void gelu_f32_i2s_encode4304_kernel(const float * input,int8_t * output) {
    constexpr int logical=4304,padded=4352,slots=17;
    __shared__ float maxima[8],scale;__shared__ int sums[8];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,token=blockIdx.x;
    float values[slots];float maximum=0.0f;
    #pragma unroll
    for(int slot=0;slot<slots;++slot) {
        const int i=threadIdx.x+slot*blockDim.x;values[slot]=0.0f;
        if(i<logical) {
            const float activated=ggml_cuda_op_gelu_single(input[token*logical+i]);
            values[slot]=activated;
            maximum=rms_encode_max_noftz(maximum,rms_encode_abs_noftz(values[slot]));
        }
    }
    for(int step=16;step;step/=2)maximum=rms_encode_max_noftz(maximum,__shfl_down_sync(0xffffffff,maximum,step));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(threadIdx.x==0){maximum=0.0f;for(int i=0;i<8;++i)maximum=rms_encode_max_noftz(maximum,maxima[i]);scale=rms_encode_scale_exact(maximum);}
    __syncthreads();
    auto * row=output+token*(padded+8);int sum=0;
    #pragma unroll
    for(int slot=0;slot<slots;++slot) {
        const int i=threadIdx.x+slot*blockDim.x;int value=0;
        if(i<logical){float scaled;asm("mul.rn.f32 %0,%1,%2;":"=f"(scaled):"f"(values[slot]),"f"(scale));value=max(-128,min(127,__float2int_rn(scaled)));}
        row[i]=value;sum+=value;
    }
    for(int step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0)sums[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];*reinterpret_cast<float *>(row+padded)=scale;*reinterpret_cast<int *>(row+padded+4)=total;}
}
__host__ __forceinline__ bool gelu_encode_inputs_readonly(const ggml_tensor * dst) {
    if(!dst || !dst->src[0])return false;
    const auto * input=dst->src[0];
    if(input->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_I8 || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst))return false;
    if(input->ne[0]!=4304 || input->ne[1]<=0 || input->ne[1]>2147483647/4360 || input->ne[2]!=1 || input->ne[3]!=1)return false;
    if(dst->ne[0]!=4360 || dst->ne[1]!=input->ne[1] || dst->ne[2]!=1 || dst->ne[3]!=1)return false;
    if(reinterpret_cast<uintptr_t>(input->data)%4 || reinterpret_cast<uintptr_t>(dst->data)%4)return false;
    const uint64_t bytes=uint64_t(4304)*uint64_t(input->ne[1])*4;
    return bytes<=uint64_t(~uintptr_t(0)) && rms_fold_disjoint(input->data,size_t(bytes),dst->data,ggml_nbytes(dst));
}

__global__ void bias_gelu_f32_i2s_encode4304_kernel(const float * input,int8_t * output,const float * bias) {
    constexpr int logical=4304,padded=4352,slots=17;
    __shared__ float maxima[8],scale;__shared__ int sums[8];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,token=blockIdx.x;
    float values[slots];float maximum=0.0f;
    #pragma unroll
    for(int slot=0;slot<slots;++slot) {
        const int i=threadIdx.x+slot*blockDim.x;values[slot]=0.0f;
        if(i<logical) {
            float added;asm("add.rn.ftz.f32 %0,%1,%2;":"=f"(added):"f"(input[token*logical+i]),"f"(bias[i]));
            const float before_gelu=added;
            const float activated=ggml_cuda_op_gelu_single(before_gelu);
            values[slot]=activated;
            maximum=rms_encode_max_noftz(maximum,rms_encode_abs_noftz(values[slot]));
        }
    }
    for(int step=16;step;step/=2)maximum=rms_encode_max_noftz(maximum,__shfl_down_sync(0xffffffff,maximum,step));
    if(lane==0)maxima[warp]=maximum;
    __syncthreads();
    if(threadIdx.x==0){maximum=0.0f;for(int i=0;i<8;++i)maximum=rms_encode_max_noftz(maximum,maxima[i]);scale=rms_encode_scale_exact(maximum);}
    __syncthreads();
    auto * row=output+token*(padded+8);int sum=0;
    #pragma unroll
    for(int slot=0;slot<slots;++slot) {
        const int i=threadIdx.x+slot*blockDim.x;int value=0;
        if(i<logical){float scaled;asm("mul.rn.f32 %0,%1,%2;":"=f"(scaled):"f"(values[slot]),"f"(scale));value=max(-128,min(127,__float2int_rn(scaled)));}
        row[i]=value;sum+=value;
    }
    for(int step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0)sums[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];*reinterpret_cast<float *>(row+padded)=scale;*reinterpret_cast<int *>(row+padded+4)=total;}
}
__host__ __forceinline__ bool bias_gelu_encode_inputs_readonly(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1])return false;
    const auto * input=dst->src[0];const auto * bias=dst->src[1];
    if(bias->type!=GGML_TYPE_F32 || !ggml_is_contiguous(bias) || bias->ne[0]!=4304 || bias->ne[1]!=1 || bias->ne[2]!=1 || bias->ne[3]!=1)return false;
    if(input->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_I8 || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst))return false;
    if(input->ne[0]!=4304 || input->ne[1]<=0 || input->ne[1]>2147483647/4360 || input->ne[2]!=1 || input->ne[3]!=1)return false;
    if(dst->ne[0]!=4360 || dst->ne[1]!=input->ne[1] || dst->ne[2]!=1 || dst->ne[3]!=1)return false;
    if(reinterpret_cast<uintptr_t>(bias->data)%4 || reinterpret_cast<uintptr_t>(input->data)%4 || reinterpret_cast<uintptr_t>(dst->data)%4)return false;
    const uint64_t bytes=uint64_t(4304)*uint64_t(input->ne[1])*4;
    return rms_fold_disjoint(bias->data,ggml_nbytes(bias),dst->data,ggml_nbytes(dst)) && bytes<=uint64_t(~uintptr_t(0)) && rms_fold_disjoint(input->data,size_t(bytes),dst->data,ggml_nbytes(dst));
}

} // namespace

template<bool affine>
static void rms_norm_dispatch(ggml_tensor * dst, void * stream, void * userdata) {
    const auto * input = dst->src[0];
    const dim3 grid(input->ne[1], input->ne[2], input->ne[3]);
    const int width = static_cast<int>(input->ne[0]);
    const float eps = *static_cast<const float *>(userdata);
    const auto * x = static_cast<const float *>(input->data);
    const float * weight=affine?static_cast<const float *>(dst->src[1]->data):nullptr;
    auto * y = static_cast<float *>(dst->data);
    const int64_t s1 = input->nb[1] / sizeof(float), s2 = input->nb[2] / sizeof(float), s3 = input->nb[3] / sizeof(float);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    if constexpr (affine) {
        if (width == 2560) {
            const auto * affine_weight=dst->src[1];
            if (ggml_is_contiguous(input) && ggml_is_contiguous(dst) && ggml_is_contiguous(affine_weight) &&
                input->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 && affine_weight->type==GGML_TYPE_F32 &&
                rms_fold_disjoint(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)) &&
                rms_fold_disjoint(affine_weight->data,ggml_nbytes(affine_weight),dst->data,ggml_nbytes(dst))) {
                rms_affine_logical1024_fold4_cached_f32_kernel<<<grid,256,32*sizeof(float),cuda_stream>>>(x,y,width,s1,s2,s3,eps,weight);
            } else {
            rms_norm_cached_input_f32_kernel<1024,true,3><<<grid,1024,32*sizeof(float),cuda_stream>>>(x,y,width,s1,s2,s3,eps,weight);
            }
            return;
        }
        if (width == 6912) {
            rms_norm_cached_input_f32_kernel<1024,true,7><<<grid,1024,32*sizeof(float),cuda_stream>>>(x,y,width,s1,s2,s3,eps,weight);
            return;
        }
    }
    if (width < 1024) {
        rms_norm_f32_kernel<256,affine><<<grid,256,32*sizeof(float),cuda_stream>>>(x,y,width,s1,s2,s3,eps,weight);
    } else {
        rms_norm_f32_kernel<1024,affine><<<grid,1024,32*sizeof(float),cuda_stream>>>(x,y,width,s1,s2,s3,eps,weight);
    }
}
void rms_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && weight->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8);
    GGML_ASSERT(input->ne[0]==2560 && input->ne[1]>0 && input->ne[1]<=INT32_MAX/2568 && input->ne[2]==1 && input->ne[3]==1);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(weight) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[0]==2568 && dst->ne[1]==input->ne[1] && dst->ne[2]==1 && dst->ne[3]==1);
    GGML_ASSERT(weight->ne[0]==2560 && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1);
    GGML_ASSERT(rms_fold_disjoint(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)) && rms_fold_disjoint(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst)));
    rms_affine_i2s_encode2560_kernel<<<dim3(input->ne[1],1,1),256,32*sizeof(float),static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(input->data),static_cast<int8_t *>(dst->data),static_cast<int>(input->ne[0]),input->nb[1]/4,input->nb[2]/4,input->nb[3]/4,*static_cast<const float *>(userdata),static_cast<const float *>(weight->data));
}

void rms_norm_f32_cuda(ggml_tensor * dst, void * stream, void * userdata) { rms_norm_dispatch<false>(dst,stream,userdata); }
void rms_affine_f32_cuda(ggml_tensor * dst, void * stream, void * userdata) { rms_norm_dispatch<true>(dst,stream,userdata); }
void gated_rms_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata) {
    const auto * input=dst->src[0];const auto * up=dst->src[1];const auto * weight=dst->src[2];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && up->type==GGML_TYPE_F32 && weight->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8);
    GGML_ASSERT(ggml_are_same_shape(input,up) && ggml_is_contiguous(input) && ggml_is_contiguous(up) && ggml_is_contiguous(weight) && ggml_is_contiguous(dst));
    GGML_ASSERT(input->ne[0]==6912 && input->ne[1]>0 && input->ne[1]<=INT32_MAX/6920 && input->ne[2]==1 && input->ne[3]==1);
    GGML_ASSERT(weight->ne[0]==6912 && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1);
    GGML_ASSERT(dst->ne[0]==6920 && dst->ne[1]==input->ne[1] && dst->ne[2]==1 && dst->ne[3]==1);
    GGML_ASSERT(rms_fold_disjoint(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)) && rms_fold_disjoint(up->data,ggml_nbytes(up),dst->data,ggml_nbytes(dst)) && rms_fold_disjoint(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst)));
    gated_rms_affine_i2s_encode6912_kernel<1024,true,7><<<dim3(input->ne[1],1,1),1024,32*sizeof(float),static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(input->data),static_cast<const float *>(up->data),static_cast<int8_t *>(dst->data),6912,input->nb[1]/4,input->nb[2]/4,input->nb[3]/4,*static_cast<const float *>(userdata),static_cast<const float *>(weight->data));
}

void gated_rms_affine_f32_cuda(ggml_tensor * dst, void * stream, void * userdata) {
    const auto * gate=dst->src[0];const auto * up=dst->src[1];
    GGML_ASSERT(gate->ne[0]==6912 && ggml_are_same_shape(gate,up));
    GGML_ASSERT(ggml_is_contiguous(gate) && ggml_is_contiguous(up));
    const dim3 grid(gate->ne[1],gate->ne[2],gate->ne[3]);
    gated_rms_affine_original_tree_f32_kernel<1024,true,7><<<grid,1024,32*sizeof(float),static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(gate->data),static_cast<const float *>(up->data),static_cast<float *>(dst->data),
        6912,gate->nb[1]/4,gate->nb[2]/4,gate->nb[3]/4,*static_cast<const float *>(userdata),
        static_cast<const float *>(dst->src[2]->data));
}

void layer_norm_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];const auto * bias=dst->src[2];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && weight->type==GGML_TYPE_F32 && bias->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(weight) && ggml_is_contiguous(bias) && ggml_is_contiguous(dst));
    GGML_ASSERT(input->ne[0]==1152 && input->ne[1]>0 && input->ne[1]<=INT32_MAX/1160 && input->ne[2]==1 && input->ne[3]==1);
    GGML_ASSERT(weight->ne[0]==1152 && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1 && bias->ne[0]==1152 && bias->ne[1]==1 && bias->ne[2]==1 && bias->ne[3]==1);
    GGML_ASSERT(dst->ne[0]==1160 && dst->ne[1]==input->ne[1] && dst->ne[2]==1 && dst->ne[3]==1);
    GGML_ASSERT(rms_fold_disjoint(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)) && rms_fold_disjoint(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst)) && rms_fold_disjoint(bias->data,ggml_nbytes(bias),dst->data,ggml_nbytes(dst)));
    layer_norm_affine_i2s_encode1152_kernel<<<dim3(input->ne[1],1,1),256,32*sizeof(float2),static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(input->data),static_cast<int8_t *>(dst->data),1152,input->nb[1]/4,input->nb[2]/4,input->nb[3]/4,*static_cast<const float *>(userdata),static_cast<const float *>(weight->data),static_cast<const float *>(bias->data));
}

void layer_norm_affine_f32_cuda(ggml_tensor * dst, void * stream, void * userdata) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];const auto * bias=dst->src[2];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && weight->type==GGML_TYPE_F32 && bias->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(weight) && ggml_is_contiguous(bias) && ggml_is_contiguous(dst));
    GGML_ASSERT(rms_fold_disjoint(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)));
    GGML_ASSERT(rms_fold_disjoint(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst)));
    GGML_ASSERT(rms_fold_disjoint(bias->data,ggml_nbytes(bias),dst->data,ggml_nbytes(dst)));
    const dim3 grid(input->ne[1],input->ne[2],input->ne[3]);const int width=static_cast<int>(input->ne[0]);
    const auto * x=static_cast<const float *>(input->data);auto * y=static_cast<float *>(dst->data);
    const auto * w=static_cast<const float *>(weight->data);const auto * b=static_cast<const float *>(bias->data);
    const float eps=*static_cast<const float *>(userdata);const auto cuda_stream=static_cast<cudaStream_t>(stream);
    if(width==1152) {
        layer_norm_affine_logical1024_fold4_cached1152_f32_kernel<<<grid,256,32*sizeof(float2),cuda_stream>>>(x,y,width,input->nb[1]/4,input->nb[2]/4,input->nb[3]/4,eps,w,b);
    } else if(width<1024) {
        layer_norm_affine_f32_kernel<WARP_SIZE><<<grid,WARP_SIZE,0,cuda_stream>>>(x,y,width,input->nb[1]/4,input->nb[2]/4,input->nb[3]/4,eps,w,b);
    } else {
        layer_norm_affine_f32_kernel<1024><<<grid,1024,32*sizeof(float2),cuda_stream>>>(x,y,width,input->nb[1]/4,input->nb[2]/4,input->nb[3]/4,eps,w,b);
    }
}

void permuted_rms_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];
    GGML_ASSERT(input->type==GGML_TYPE_F32 && weight->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_I8);
    GGML_ASSERT(input->ne[0]>0 && input->ne[0]<=2560 && 2560%input->ne[0]==0 && input->ne[1]==2560/input->ne[0]);
    GGML_ASSERT(input->ne[2]>0 && input->ne[2]<=INT32_MAX/2568 && input->ne[3]==1);
    GGML_ASSERT(input->nb[0]==4 && input->nb[2]==uint64_t(input->ne[0])*4 && input->nb[1]==uint64_t(input->ne[0])*uint64_t(input->ne[2])*4);
    GGML_ASSERT(weight->ne[0]==2560 && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1 && ggml_is_contiguous(weight));
    GGML_ASSERT(dst->ne[0]==2568 && dst->ne[1]==input->ne[2] && dst->ne[2]==1 && dst->ne[3]==1 && ggml_is_contiguous(dst));
    GGML_ASSERT(reinterpret_cast<uintptr_t>(input->data)%4==0 && reinterpret_cast<uintptr_t>(weight->data)%4==0 && reinterpret_cast<uintptr_t>(dst->data)%4==0);
    const uint64_t bytes=uint64_t(2560)*uint64_t(input->ne[2])*4;
    GGML_ASSERT(bytes<=uint64_t(~uintptr_t(0)) && rms_fold_disjoint(input->data,size_t(bytes),dst->data,ggml_nbytes(dst)) && rms_fold_disjoint(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst)));
    permuted_rms_affine_i2s_encode2560_kernel<<<dim3(input->ne[2],1,1),256,32*sizeof(float),static_cast<cudaStream_t>(stream)>>>(static_cast<const char *>(input->data),static_cast<int8_t *>(dst->data),2560,input->nb[2],input->nb[3],input->nb[3],*static_cast<const float *>(userdata),static_cast<const float *>(weight->data),int(input->ne[0]),input->nb[0],input->nb[1]);
}

void gelu_f32_i2s_encode_cuda(ggml_tensor * dst,void * stream,void *) {
    GGML_ASSERT(gelu_encode_inputs_readonly(dst));
    gelu_f32_i2s_encode4304_kernel<<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(dst->src[0]->data),static_cast<int8_t *>(dst->data));
}

void bias_gelu_f32_i2s_encode_cuda(ggml_tensor * dst,void * stream,void *) {
    GGML_ASSERT(bias_gelu_encode_inputs_readonly(dst));
    bias_gelu_f32_i2s_encode4304_kernel<<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(dst->src[0]->data),static_cast<int8_t *>(dst->data),static_cast<const float *>(dst->src[1]->data));
}

} // namespace robotcpp::bitvla
