#include "i2s.h"
#include <cuda_runtime.h>
#include "mma.cuh"
#include <cstdint>

namespace robotcpp::bitvla {
namespace {
// Exact RN division on a domain with no subnormal input or output.
// Outside this checked domain keep the original IEEE RN intrinsic unchanged.
__device__ __forceinline__ float epilogue_div_rn_normal_guarded(float numerator, float denominator) {
    const uint32_t a = __float_as_uint(numerator) & 0x7fffffffu;
    const uint32_t b = __float_as_uint(denominator);
    if (b >= 0x00800000u && b <= 0x4b800000u &&
        (a == 0 || (a >= 0x3f800000u && a < 0x7f800000u))) {
        float result;
        asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
        return result;
    }
    return __fdiv_rn(numerator, denominator);
}
// RN conversion of any INT32 is a finite integer in[-2^31,2^31].
// RN subtraction of two such values stays an integer in[-2^32,2^32].
// Hence the original numerator guard (zero or finiteabs>=1) is always true.
// Keep the original denominator guard and both exact RN division instructions.
__device__ __forceinline__ float epilogue_div_rn_int32_difference_guarded(int accumulator, int sum, float denominator) {
    const float numerator=__fsub_rn(__int2float_rn(accumulator),__int2float_rn(sum));
    const uint32_t d=__float_as_uint(denominator);
    if(d>=0x00800000u && d<=0x4b800000u) {
        float result;
        asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
        return result;
    }
    return __fdiv_rn(numerator,denominator);
}
// Compute the I2S activation scale with the original FP64 division,
// then round its result to F32.
__device__ __forceinline__ float activation_scale_exact_f32(float maximum) {
    return __double2float_rn(__ddiv_rn(127.0, fmax(static_cast<double>(maximum), 1e-5)));
}
__global__ void encode_f32_scale_exact(const float * input, int8_t * output, int logical, int padded) {
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

// Retain each original F32 value across row maximum/scale and RN quantization.
template<int slots>
__global__ void encode_cached_f32_scale_exact(const float * input, int8_t * output, int logical, int padded) {
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
            values[slot] = input[token * logical + i];
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

// Packed weight codes 0,1,2 multiply signed INT8 activations in one MMA pass.
template<bool matrix_format, bool row_aligned, bool optimized_route>
__global__ void multiply(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint8_t a[128 * shared_stride];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2], next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    const uint32_t codes = (packed >> (6 - 2 * group)) & 0x03030303u;
                    ggml_cuda_memcpy_1<4>(a + local_row * shared_stride + group * 32 + byte, &codes);
                }
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        } else {
            for (int index = thread; index < 128 * 8; index += 512) {
                const int local_row = index / 8, byte = (index % 8) * 4;
                const int row = first_row + local_row;
                uint32_t packed = 0, following = 0;
                int quarter = 0;
                if constexpr (matrix_format && !row_aligned) {
                    const int col = base + byte;
                    const int64_t index = int64_t(row) * logical + col;
                    const int64_t address = (index / 128) * 32 + index % 32;
                    quarter = (index % 128) / 32;
                    if (row < rows && col < logical) {
                        ggml_cuda_memcpy_1<4>(&packed, weights + address);
                        // A row fragment spans at most two consecutive packed groups.
                        if (quarter && address + 32 < int64_t(rows) * logical / 4)
                            ggml_cuda_memcpy_1<4>(&following, weights + address + 32);
                    }
                } else {
                    constexpr int header = matrix_format ? 0 : 4;
                    if (row < rows) ggml_cuda_memcpy_1<4>(&packed, weights + row * stride + header + base / 4 + byte);
                }
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    uint32_t codes;
                    if constexpr (matrix_format && !row_aligned) {
                        const int position = quarter + group;
                        const uint32_t raw = position < 4 ? packed : following;
                        codes = (raw >> (6 - 2 * (position % 4))) & 0x03030303u;
                    } else {
                        codes = (packed >> (6 - 2 * group)) & 0x03030303u;
                    }
                    ggml_cuda_memcpy_1<4>(a + local_row * shared_stride + group * 32 + byte, &codes);
                }
            }
            for (int index = thread; index < 32 * (tile / 4); index += 512) {
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                const int token = first_token + local_token;
                uint32_t values = 0;
                if (token < tokens) ggml_cuda_memcpy_1<4>(&values, input + token * (k + 8) + base + col);
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &values);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            load_ldmatrix(at, reinterpret_cast<const int *>(a + wm * 16 * shared_stride + offset), shared_stride / 4);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const auto * q = input + token * (k + 8);
                const int sum = *reinterpret_cast<const int *>(q + k + 4);
                const float activation_scale = *reinterpret_cast<const float *>(q + k);
                const auto * scale = matrix_format ? weights + int64_t(rows) * logical / 4 : weights + row * stride;
                const float weight_scale = *reinterpret_cast<const float *>(scale);
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

// Cross-row packed format: same decoding, one K128 tile register lookahead.
template<bool optimized_route>
__global__ void multiply_crossrow_prefetch(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    constexpr bool matrix_format = true, row_aligned = false;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint8_t a[128 * shared_stride];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Prefetch both packed groups required by the original cross-row decoder.
    // Original aligned/legacy kernels remain separate and unchanged.
    constexpr bool prefetch = true;
    uint32_t next_weights[2], next_following[2], next_input[2];
    int next_quarter[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            next_following[item] = 0;
            const int col_weight = base + byte;
            const int64_t matrix_index = int64_t(row) * logical + col_weight;
            const int64_t address = (matrix_index / 128) * 32 + matrix_index % 32;
            next_quarter[item] = (matrix_index % 128) / 32;
            if (row < rows && col_weight < logical) {
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + address);
                if (next_quarter[item] && address + 32 < int64_t(rows) * logical / 4)
                    ggml_cuda_memcpy_1<4>(&next_following[item], weights + address + 32);
            }
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    const int position = next_quarter[item] + group;
                    const uint32_t raw = position < 4 ? packed : next_following[item];
                    const uint32_t codes = (raw >> (6 - 2 * (position % 4))) & 0x03030303u;
                    ggml_cuda_memcpy_1<4>(a + local_row * shared_stride + group * 32 + byte, &codes);
                }
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        } else {
            for (int index = thread; index < 128 * 8; index += 512) {
                const int local_row = index / 8, byte = (index % 8) * 4;
                const int row = first_row + local_row;
                uint32_t packed = 0, following = 0;
                int quarter = 0;
                if constexpr (matrix_format && !row_aligned) {
                    const int col = base + byte;
                    const int64_t index = int64_t(row) * logical + col;
                    const int64_t address = (index / 128) * 32 + index % 32;
                    quarter = (index % 128) / 32;
                    if (row < rows && col < logical) {
                        ggml_cuda_memcpy_1<4>(&packed, weights + address);
                        // A row fragment spans at most two consecutive packed groups.
                        if (quarter && address + 32 < int64_t(rows) * logical / 4)
                            ggml_cuda_memcpy_1<4>(&following, weights + address + 32);
                    }
                } else {
                    constexpr int header = matrix_format ? 0 : 4;
                    if (row < rows) ggml_cuda_memcpy_1<4>(&packed, weights + row * stride + header + base / 4 + byte);
                }
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    uint32_t codes;
                    if constexpr (matrix_format && !row_aligned) {
                        const int position = quarter + group;
                        const uint32_t raw = position < 4 ? packed : following;
                        codes = (raw >> (6 - 2 * (position % 4))) & 0x03030303u;
                    } else {
                        codes = (packed >> (6 - 2 * group)) & 0x03030303u;
                    }
                    ggml_cuda_memcpy_1<4>(a + local_row * shared_stride + group * 32 + byte, &codes);
                }
            }
            for (int index = thread; index < 32 * (tile / 4); index += 512) {
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                const int token = first_token + local_token;
                uint32_t values = 0;
                if (token < tokens) ggml_cuda_memcpy_1<4>(&values, input + token * (k + 8) + base + col);
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &values);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            load_ldmatrix(at, reinterpret_cast<const int *>(a + wm * 16 * shared_stride + offset), shared_stride / 4);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const auto * q = input + token * (k + 8);
                const int sum = *reinterpret_cast<const int *>(q + k + 4);
                const float activation_scale = *reinterpret_cast<const float *>(q + k);
                const auto * scale = matrix_format ? weights + int64_t(rows) * logical / 4 : weights + row * stride;
                const float weight_scale = *reinterpret_cast<const float *>(scale);
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

// Native aligned: retain packed weights in shared memory, unpack MMA registers.
template<bool optimized_route>
__global__ void multiply_packed_shared(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint32_t packed_a[128 * 8];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2], next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                // Four-column planes match the measured LDSM A fragment mapping.
                packed_a[((byte / 16) * 128 + local_row) * 4 + ((byte / 4) % 4)] = packed;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        // A fragment's four packed words serve all four K32 quarters.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = wm * 16 + (l % 2) * 8 + threadIdx.x / 4;
            const int col = threadIdx.x % 4 + (l / 2) * 4;
            packed_fragment[l] = packed_a[((col / 4) * 128 + row) * 4 + col % 4];
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const auto * q = input + token * (k + 8);
                const int sum = *reinterpret_cast<const int *>(q + k + 4);
                const float activation_scale = *reinterpret_cast<const float *>(q + k);
                const auto * scale = matrix_format ? weights + int64_t(rows) * logical / 4 : weights + row * stride;
                const float weight_scale = *reinterpret_cast<const float *>(scale);
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ void multiply_vision_attnout_packed_input8(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint32_t packed_a[128 * 8];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2];
    __align__(16) uint64_t next_input8[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
        }
        if (thread < 256) {
            const int token = first_token + thread / 8, col = (thread % 8) * 16;
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                next_input8[item] = 0;
                if (token < tokens)
                    ggml_cuda_memcpy_1<8>(&next_input8[item], input + token * (k + 8) + base + col + item * 8);
            }
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                // Four-column planes match the measured LDSM A fragment mapping.
                packed_a[((byte / 16) * 128 + local_row) * 4 + ((byte / 4) % 4)] = packed;
            }
            if (thread < 256) {
                const int local_token = thread / 8, col = (thread % 8) * 16;
                ggml_cuda_memcpy_1<16>(activations + local_token * shared_stride + col, next_input8);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        // A fragment's four packed words serve all four K32 quarters.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = wm * 16 + (l % 2) * 8 + threadIdx.x / 4;
            const int col = threadIdx.x % 4 + (l / 2) * 4;
            packed_fragment[l] = packed_a[((col / 4) * 128 + row) * 4 + col % 4];
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // Same accepted134 disjoint span guard; cache native matrix scale and
    // the two original token headers per N tile before its output stores.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ void multiply_packed_cached_epilogue(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint32_t packed_a[128 * 8];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2], next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                // Four-column planes match the measured LDSM A fragment mapping.
                packed_a[((byte / 16) * 128 + local_row) * 4 + ((byte / 4) % 4)] = packed;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        // A fragment's four packed words serve all four K32 quarters.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = wm * 16 + (l % 2) * 8 + threadIdx.x / 4;
            const int col = threadIdx.x % 4 + (l / 2) * 4;
            packed_fragment[l] = packed_a[((col / 4) * 128 + row) * 4 + col % 4];
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // Same accepted134 disjoint span guard; cache native matrix scale and
    // the two original token headers per N tile before its output stores.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}


template<bool optimized_route>
__global__ __launch_bounds__(256, 4) void multiply_direct_prefetch_warp4(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4], next_input[4];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = thread + item * 256;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const auto * q = input + token * (k + 8);
                const int sum = *reinterpret_cast<const int *>(q + k + 4);
                const float activation_scale = *reinterpret_cast<const float *>(q + k);
                const auto * scale = matrix_format ? weights + int64_t(rows) * logical / 4 : weights + row * stride;
                const float weight_scale = *reinterpret_cast<const float *>(scale);
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256, 4) void multiply_direct_cached_epilogue_warp4(const uint8_t * weights, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical) {
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4], next_input[4];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = thread + item * 256;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool value_layout>
__global__ void multiply_packed_attention_layout_f32(const uint8_t * weights,const int8_t * input,float * output,
        int rows,int tokens,int k,int stride,int logical,const float * bias,int heads) {
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint32_t packed_a[128 * 8];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2], next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                // Four-column planes match the measured LDSM A fragment mapping.
                packed_a[((byte / 16) * 128 + local_row) * 4 + ((byte / 4) % 4)] = packed;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        // A fragment's four packed words serve all four K32 quarters.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = wm * 16 + (l % 2) * 8 + threadIdx.x / 4;
            const int col = threadIdx.x % 4 + (l / 2) * 4;
            packed_fragment[l] = packed_a[((col / 4) * 128 + row) * 4 + col % 4];
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // Same accepted134 disjoint span guard; cache native matrix scale and
    // the two original token headers per N tile before its output stores.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                float added;
                // Preserve the accepted bias_attention_layout operand order and F32 result.
                asm("add.rn.ftz.f32 %0, %1, %2;":"=f"(added):"f"(value),"f"(bias[row]));
                const int dim=rows/heads,head=row/dim,col=row%dim;
                const int destination=value_layout?(head*dim+col)*tokens+token:(head*tokens+token)*dim+col;
                output[destination]=added;
            }
        }
    }
}

__global__ void multiply_packed_attention_qkv(const uint8_t * query_weights,const int8_t * input,float * output,
        int rows,int tokens,int k,int stride,int logical,const float * query_bias,int heads,
        const uint8_t * key_weights,const uint8_t * value_weights,const float * key_bias,const float * value_bias) {
    const int family=blockIdx.z;const bool value_layout=family==2;
    const uint8_t * weights=family==0?query_weights:family==1?key_weights:value_weights;
    const float * bias=family==0?query_bias:family==1?key_bias:value_bias;
    output+=int64_t(family)*rows*tokens;

    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint32_t packed_a[128 * 8];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2], next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                // Four-column planes match the measured LDSM A fragment mapping.
                packed_a[((byte / 16) * 128 + local_row) * 4 + ((byte / 4) % 4)] = packed;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        // A fragment's four packed words serve all four K32 quarters.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = wm * 16 + (l % 2) * 8 + threadIdx.x / 4;
            const int col = threadIdx.x % 4 + (l / 2) * 4;
            packed_fragment[l] = packed_a[((col / 4) * 128 + row) * 4 + col % 4];
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // Same accepted134 disjoint span guard; cache native matrix scale and
    // the two original token headers per N tile before its output stores.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                float added;
                // Preserve the accepted bias_attention_layout operand order and F32 result.
                asm("add.rn.ftz.f32 %0, %1, %2;":"=f"(added):"f"(value),"f"(bias[row]));
                const int dim=rows/heads,head=row/dim,col=row%dim;
                const int destination=value_layout?(head*dim+col)*tokens+token:(head*tokens+token)*dim+col;
                output[destination]=added;
            }
        }
    }
}


template<bool optimized_route>
__global__ __launch_bounds__(256, 4) void multiply_qkv_composite_warp4(const uint8_t * weights0,const int8_t * input,float * output,
        int tokens,int k,int stride,int logical,const uint8_t * weights1,const uint8_t * weights2) {
    const int family=blockIdx.x<20?0:(blockIdx.x<25?1:2);
    const int row_tile=blockIdx.x-(family==0?0:(family==1?20:25));
    const int rows=family==0?2560:640;
    const uint8_t * weights=family==0?weights0:(family==1?weights1:weights2);
    output+=int64_t(family==0?0:(family==1?2560:3200))*tokens;

    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = row_tile * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4], next_input[4];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = thread + item * 256;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256, 4) void multiply_llm_ffn_pair_f32_n48(const uint8_t * weights0, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical,const uint8_t * weights1) {
    const uint8_t * weights=blockIdx.z?weights1:weights0;
    output+=int64_t(blockIdx.z)*rows*tokens;

    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[48 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 48;
    ggml_cuda_mma::tile<16, 8, int> accumulators[6];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4];
    alignas(16) uint64_t next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
        }
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int token = first_token + thread / (tile / 16);
            const int col = (thread % (tile / 16)) * 16 + item * 8;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<8>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            const int local_token = thread / (tile / 16), col = (thread % (tile / 16)) * 16;
            ggml_cuda_memcpy_1<16>(activations + local_token * shared_stride + col, next_input);
            if (thread < 128) {
                #pragma unroll
                for (int item = 0; item < 2; ++item) {
                    const int token = first_token + 32 + thread / (tile / 16);
                    const int col = (thread % (tile / 16)) * 16 + item * 8;
                    next_input[item] = 0;
                    if (token < tokens)
                        ggml_cuda_memcpy_1<8>(&next_input[item], input + token * (k + 8) + base + col);
                }
                const int extra_token = 32 + thread / (tile / 16), extra_col = (thread % (tile / 16)) * 16;
                ggml_cuda_memcpy_1<16>(activations + extra_token * shared_stride + extra_col, next_input);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 6; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 6; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256, 4) void multiply_llm_ffn_pair_input8(const uint8_t * weights0, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical,const uint8_t * weights1) {
    const uint8_t * weights=blockIdx.z?weights1:weights0;
    output+=int64_t(blockIdx.z)*rows*tokens;

    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4];
    alignas(16) uint64_t next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
        }
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int token = first_token + thread / (tile / 16);
            const int col = (thread % (tile / 16)) * 16 + item * 8;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<8>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            const int local_token = thread / (tile / 16), col = (thread % (tile / 16)) * 16;
            ggml_cuda_memcpy_1<16>(activations + local_token * shared_stride + col, next_input);
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256, 4) void multiply_direct_pair_warp4(const uint8_t * weights0, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical,const uint8_t * weights1) {
    const uint8_t * weights=blockIdx.z?weights1:weights0;
    output+=int64_t(blockIdx.z)*rows*tokens;

    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4], next_input[4];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = thread + item * 256;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256,4) void multiply_llm_ffndown_input8(const uint8_t * weights, const int8_t * input, float * output,
                         int tokens) {
    constexpr int rows=2560,k=6912,stride=1728,logical=6912;
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4];
    __align__(16) uint64_t next_input8[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
        }
        const int token = first_token + thread / 8;
        const int col = (thread % 8) * 16;
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            next_input8[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<8>(&next_input8[item], input + token * (k + 8) + base + col + item * 8);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    #pragma unroll 1
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            const int local_token = thread / 8, col = (thread % 8) * 16;
            ggml_cuda_memcpy_1<16>(activations + local_token * shared_stride + col, next_input8);
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256,4) void multiply_llm_ffndown_fixed_dimensions(const uint8_t * weights,const int8_t * input,float * output,int tokens) {
    constexpr int rows=2560,k=6912,stride=1728,logical=6912;
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4], next_input[4];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    #pragma unroll 1
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = thread + item * 256;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

template<bool optimized_route>
__global__ __launch_bounds__(256,4) void multiply_llm_attnout_fixed_dimensions(const uint8_t * weights,const int8_t * input,float * output,int tokens) {
    constexpr int rows=2560,k=2560,stride=640,logical=2560;
    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[4];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_fragment[4], next_input[4];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int wi = thread + item * 256;
            const int row = first_row + wm * 16 + (item % 2) * 8 + threadIdx.x / 4;
            const int word = threadIdx.x % 4 + (item / 2) * 4;
            next_fragment[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_fragment[item], weights + row * stride + base / 4 + word * 4);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    #pragma unroll 1
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int index = thread + item * 256;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // Preserve the current operand before loading the next K tile.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) packed_fragment[l] = next_fragment[l];
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // The host requires the entire read-only input/weight spans to be disjoint
    // from output. Cache one matrix scale and the two token headers per N tile.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

__host__ __forceinline__ bool i2s_epilogue_disjoint_span(const void * input, size_t bytes, const void * output, size_t out_bytes) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(input),b=reinterpret_cast<uintptr_t>(output),top=~uintptr_t(0);
    return input && output && bytes && out_bytes && bytes<=top-a && out_bytes<=top-b &&
        (a+bytes<=b || b+out_bytes<=a);
}
__host__ __forceinline__ bool i2s_epilogue_headers_readonly(const ggml_tensor * dst) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];
    return input && weight && input->type==GGML_TYPE_I8 && weight->type==GGML_TYPE_I2_S && dst->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(input) && ggml_is_contiguous(weight) && ggml_is_contiguous(dst) &&
        i2s_epilogue_disjoint_span(input->data,ggml_nbytes(input),dst->data,ggml_nbytes(dst)) &&
        i2s_epilogue_disjoint_span(weight->data,ggml_nbytes(weight),dst->data,ggml_nbytes(dst));
}



__host__ __forceinline__ bool i2s_attention_layout_inputs_readonly(const ggml_tensor * dst,bool value_layout) {
    if(!dst || !dst->src[0] || !dst->src[1] || !dst->src[2])return false;
    const auto * input=dst->src[0];const auto * weight=dst->src[1];const auto * bias=dst->src[2];
    const int64_t k=weight->ne[0],m=weight->ne[1],n=input->ne[1],heads=dst->ne[2];
    if(k<=0 || k%128 || k>INT32_MAX-8 || m<=0 || m>INT32_MAX-127 || n<=0 || n>INT32_MAX-127 || (m>=2048 && n>=512))return false;
    if(heads<=0 || heads>m || m%heads || n>INT32_MAX/(k+8) || n>INT32_MAX/m || m>INT32_MAX/(k/4))return false;
    if(input->ne[0]!=k+8 || input->ne[2]!=1 || input->ne[3]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1)return false;
    if(dst->ne[0]!=(value_layout?n:m/heads) || dst->ne[1]!=(value_layout?m/heads:n) || dst->ne[3]!=1)return false;
    return bias->type==GGML_TYPE_F32 && bias->ne[0]==m && bias->ne[1]==1 && bias->ne[2]==1 && bias->ne[3]==1 &&
        i2s_epilogue_headers_readonly(dst) && ggml_is_contiguous(bias) &&
        i2s_epilogue_disjoint_span(bias->data,ggml_nbytes(bias),dst->data,ggml_nbytes(dst));
}

__host__ __forceinline__ bool i2s_attention_qkv_inputs_readonly(const ggml_tensor * dst,int heads) {
    if(!dst || dst->type!=GGML_TYPE_F32 || !ggml_is_contiguous(dst))return false;
    for(int s=0;s<7;++s)if(!dst->src[s])return false;
    const auto * encoded=dst->src[0];const auto * weight=dst->src[1];
    const int64_t k=weight->ne[0],m=weight->ne[1],n=encoded->ne[1];
    if(k<=0 || k%128 || k>INT32_MAX-8 || m<=0 || m>INT32_MAX-127 || n<=0 || n>INT32_MAX-127 || heads<=0 || heads>m || m%heads || n>INT32_MAX/(3*m) || n>65535LL*32)return false;
    if(dst->ne[0]!=3*m*n || dst->ne[1]!=1 || dst->ne[2]!=1 || dst->ne[3]!=1 || reinterpret_cast<uintptr_t>(dst->data)%4)return false;
    const uintptr_t start=reinterpret_cast<uintptr_t>(dst->data),limit=~uintptr_t(0);const uint64_t bytes=ggml_nbytes(dst);
    if(!start || bytes>limit-start)return false;
    for(int family=0;family<3;++family) {
        const auto * w=dst->src[1+family];if(w->ne[0]!=k || w->ne[1]!=m)return false;
        ggml_tensor projection=*dst;projection.ne[0]=family==2?n:m/heads;projection.ne[1]=family==2?m/heads:n;projection.ne[2]=heads;projection.ne[3]=1;
        projection.nb[0]=sizeof(float);projection.nb[1]=projection.ne[0]*sizeof(float);projection.nb[2]=projection.nb[1]*projection.ne[1];projection.nb[3]=projection.nb[2]*heads;
        projection.data=reinterpret_cast<void *>(start+uint64_t(family)*uint64_t(m)*uint64_t(n)*4);
        for(auto & source:projection.src)source=nullptr;
        projection.src[0]=dst->src[0];projection.src[1]=dst->src[1+family];projection.src[2]=dst->src[4+family];
        if(!i2s_attention_layout_inputs_readonly(&projection,family==2))return false;
    }
    for(int s=0;s<7;++s)if(reinterpret_cast<uintptr_t>(dst->src[s]->data)%4 || !i2s_epilogue_disjoint_span(dst->src[s]->data,ggml_nbytes(dst->src[s]),dst->data,ggml_nbytes(dst)))return false;
    return true;
}

__host__ __forceinline__ bool i2s_qkv_inputs_readonly(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1] || !dst->src[2] || !dst->src[3])return false;
    const auto * encoded=dst->src[0];const int64_t n=encoded->ne[1];
    if(n<512 || n>INT32_MAX/3840 || encoded->ne[0]!=2568 || encoded->ne[2]!=1 || encoded->ne[3]!=1)return false;
    if(dst->ne[0]!=3840*n || dst->ne[1]!=1 || dst->ne[2]!=1 || dst->ne[3]!=1 || !i2s_epilogue_headers_readonly(dst))return false;
    for(int p=0;p<3;++p) {
        const auto * w=dst->src[p+1];const int64_t rows=p==0?2560:640;
        if(w->type!=GGML_TYPE_I2_S || w->ne[0]!=2560 || w->ne[1]!=rows || w->ne[2]!=1 || w->ne[3]!=1 || !ggml_is_contiguous(w))return false;
        if(!i2s_epilogue_disjoint_span(w->data,ggml_nbytes(w),dst->data,ggml_nbytes(dst)))return false;
    }
    return true;
}

__host__ __forceinline__ bool i2s_kv_pair_selected_shape(int64_t k,int64_t m,int64_t n) {
    return k==2560 && m==640 && n>=512;
}

__host__ __forceinline__ bool i2s_packed_pair_inputs_readonly(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1] || !dst->src[2])return false;
    const auto * encoded=dst->src[0];const auto * w0=dst->src[1];const auto * w1=dst->src[2];
    const int64_t k=w0->ne[0],m=dst->ne[0],n=dst->ne[1];
    if(k<=0 || k%128 || k>INT32_MAX-8 || m<=0 || m>INT32_MAX-127 || n<=0 || n>65535*32 || (m>=2048 && n>=512))return false;
    if(n>INT32_MAX/(k+8) || n>(INT32_MAX/2)/m || m>INT32_MAX/(k/4))return false;
    if(encoded->ne[0]!=k+8 || encoded->ne[1]!=n || encoded->ne[2]!=1 || encoded->ne[3]!=1)return false;
    if(w0->ne[1]!=m || w0->ne[2]!=1 || w0->ne[3]!=1 || w1->ne[0]!=k || w1->ne[1]!=m || w1->ne[2]!=1 || w1->ne[3]!=1 || dst->ne[2]!=2 || dst->ne[3]!=1)return false;
    return w1->type==GGML_TYPE_I2_S && i2s_epilogue_headers_readonly(dst) && ggml_is_contiguous(w1) &&
        i2s_epilogue_disjoint_span(w1->data,ggml_nbytes(w1),dst->data,ggml_nbytes(dst));
}

__host__ __forceinline__ bool i2s_pair_inputs_readonly(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1] || !dst->src[2])return false;
    const auto * encoded=dst->src[0];const auto * w0=dst->src[1];const auto * w1=dst->src[2];
    const int64_t k=w0->ne[0],m=dst->ne[0],n=dst->ne[1];
    if(k<=0 || k%128 || k>INT32_MAX-8 || m<2048 || m>INT32_MAX-127 || n<512 || n>INT32_MAX-31)return false;
    if(n>INT32_MAX/(k+8) || n>(INT32_MAX/2)/m || m>INT32_MAX/(k/4))return false;
    if(encoded->ne[0]!=k+8 || encoded->ne[1]!=n || encoded->ne[2]!=1 || encoded->ne[3]!=1)return false;
    if(w0->ne[1]!=m || w0->ne[2]!=1 || w0->ne[3]!=1 || w1->ne[0]!=k || w1->ne[1]!=m || w1->ne[2]!=1 || w1->ne[3]!=1 || dst->ne[2]!=2 || dst->ne[3]!=1)return false;
    return w1->type==GGML_TYPE_I2_S && i2s_epilogue_headers_readonly(dst) && ggml_is_contiguous(w1) &&
        i2s_epilogue_disjoint_span(w1->data,ggml_nbytes(w1),dst->data,ggml_nbytes(dst));
}

template<bool optimized_route>
__global__ void multiply_packed_pair_cached_epilogue(const uint8_t * weights0, const int8_t * input, float * output,
                         int rows, int tokens, int k, int stride, int logical,const uint8_t * weights1) {
    const uint8_t * weights=blockIdx.z?weights1:weights0;
    output+=int64_t(blockIdx.z)*rows*tokens;

    constexpr bool matrix_format = true, row_aligned = true;
    using namespace ggml_cuda_mma;
    constexpr int tile = 128;
    constexpr int shared_stride = tile + 16;
    __shared__ __align__(32) uint32_t packed_a[128 * 8];
    __shared__ __align__(32) int8_t activations[32 * shared_stride];
    const int warp = threadIdx.y, thread = warp * 32 + threadIdx.x;
    const int wm = warp / 2, wn = warp % 2;
    const int first_row = blockIdx.x * 128, first_token = blockIdx.y * 32;
    ggml_cuda_mma::tile<16, 8, int> accumulators[2];
    // Only native row-aligned weights use the register lookahead. Other
    // formats retain the original loading path below, including cross-row tails.
    constexpr bool prefetch = matrix_format && row_aligned;
    uint32_t next_weights[2], next_input[2];
    const auto load_next = [&](int base) {
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int wi = thread + item * 512;
            const int row = first_row + wi / 8;
            const int byte = (wi % 8) * 4;
            next_weights[item] = 0;
            if (row < rows)
                ggml_cuda_memcpy_1<4>(&next_weights[item], weights + row * stride + base / 4 + byte);
            const int token = first_token + wi / (tile / 4);
            const int col = (wi % (tile / 4)) * 4;
            next_input[item] = 0;
            if (token < tokens)
                ggml_cuda_memcpy_1<4>(&next_input[item], input + token * (k + 8) + base + col);
        }
    };
    if constexpr (prefetch) {
        if (k > 0) load_next(0);
    }
    for (int base = 0; base < k; base += tile) {
        if constexpr (prefetch) {
            #pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int index = thread + item * 512;
                const int local_row = index / 8, byte = (index % 8) * 4;
                const uint32_t packed = next_weights[item];
                // Four-column planes match the measured LDSM A fragment mapping.
                packed_a[((byte / 16) * 128 + local_row) * 4 + ((byte / 4) % 4)] = packed;
                const int local_token = index / (tile / 4), col = (index % (tile / 4)) * 4;
                ggml_cuda_memcpy_1<4>(activations + local_token * shared_stride + col, &next_input[item]);
            }
        }
        __syncthreads();
        // The current shared tile is complete. Load the next tile into registers
        // before the current MMA work; it is consumed only after the final barrier.
        if constexpr (prefetch) {
            if (base + tile < k) load_next(base + tile);
        }
        // A fragment's four packed words serve all four K32 quarters.
        uint32_t packed_fragment[4];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = wm * 16 + (l % 2) * 8 + threadIdx.x / 4;
            const int col = threadIdx.x % 4 + (l / 2) * 4;
            packed_fragment[l] = packed_a[((col / 4) * 128 + row) * 4 + col % 4];
        }
        #pragma unroll
        for (int offset = 0; offset < tile; offset += 32) {
            ggml_cuda_mma::tile<16, 8, int> at;
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                at.x[l] = (packed_fragment[l] >> (6 - 2 * (offset / 32))) & 0x03030303u;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                ggml_cuda_mma::tile<8, 8, int> bt;
                const int index = (wn * 16 + n * 8) * shared_stride + offset;
                load_ldmatrix(bt, reinterpret_cast<const int *>(activations + index), shared_stride / 4);
                mma(accumulators[n], at, bt);
            }
        }
        __syncthreads();
    }
    // Same accepted134 disjoint span guard; cache native matrix scale and
    // the two original token headers per N tile before its output stores.
    const float weight_scale = *reinterpret_cast<const float *>(weights + int64_t(rows) * logical / 4);
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int token_sums[2] = {0,0};
        float token_scales[2] = {1.0f,1.0f};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(j);
            if (token < tokens) {
                const auto * q = input + token * (k + 8);
                token_sums[j] = *reinterpret_cast<const int *>(q + k + 4);
                token_scales[j] = *reinterpret_cast<const float *>(q + k);
            }
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = first_row + wm * 16 + accumulators[n].get_i(l);
            const int token = first_token + wn * 16 + n * 8 + accumulators[n].get_j(l);
            if (row < rows && token < tokens) {
                const int sum = token_sums[l % 2];
                const float activation_scale = token_scales[l % 2];
                const float value = __fmul_rn(epilogue_div_rn_int32_difference_guarded(accumulators[n].x[l], sum, activation_scale), weight_scale);
                output[token * rows + row] = value;
            }
        }
    }
}

__host__ __forceinline__ bool i2s_vision_ffn_first_register_eligible(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1])return false;
    const auto * e=dst->src[0];const auto * w=dst->src[1];
    return e->ne[0]==1160 && e->ne[1]==256 && e->ne[2]==1 && e->ne[3]==1 &&
        w->ne[0]==1152 && w->ne[1]==4304 && w->ne[2]==1 && w->ne[3]==1 &&
        dst->ne[0]==4304 && dst->ne[1]==256 && dst->ne[2]==1 && dst->ne[3]==1 &&
        i2s_epilogue_headers_readonly(dst);
}

__host__ __forceinline__ bool i2s_vision_attnout_packed_input8_eligible(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1])return false;
    const auto * e=dst->src[0];const auto * w=dst->src[1];
    return e->ne[0]==1160 && e->ne[1]==256 && e->ne[2]==1 && e->ne[3]==1 &&
        w->ne[0]==1152 && w->ne[1]==1152 && w->ne[2]==1 && w->ne[3]==1 &&
        dst->ne[0]==1152 && dst->ne[1]==256 && dst->ne[2]==1 && dst->ne[3]==1 &&
        reinterpret_cast<uintptr_t>(e->data)%8==0 && reinterpret_cast<uintptr_t>(w->data)%4==0 &&
        reinterpret_cast<uintptr_t>(dst->data)%4==0 && i2s_epilogue_headers_readonly(dst);
}

__host__ __forceinline__ bool i2s_llm_ffndown_fixed_dimensions_eligible(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1])return false;
    const auto * e=dst->src[0];const auto * w=dst->src[1];const int64_t n=e->ne[1];
    if(n<512 || n>INT32_MAX/6920 || e->ne[0]!=6920 || e->ne[2]!=1 || e->ne[3]!=1 || w->ne[0]!=6912 || w->ne[1]!=2560 || w->ne[2]!=1 || w->ne[3]!=1 || dst->ne[0]!=2560 || dst->ne[1]!=n || dst->ne[2]!=1 || dst->ne[3]!=1)return false;
    if(reinterpret_cast<uintptr_t>(e->data)%4 || reinterpret_cast<uintptr_t>(w->data)%4 || reinterpret_cast<uintptr_t>(dst->data)%4)return false;
    return i2s_epilogue_headers_readonly(dst);
}
__host__ __forceinline__ bool i2s_llm_ffndown_input8_eligible(const ggml_tensor * dst) {
    return i2s_llm_ffndown_fixed_dimensions_eligible(dst) &&
        reinterpret_cast<uintptr_t>(dst->src[0]->data)%8==0;
}


__host__ __forceinline__ bool i2s_llm_attnout_fixed_dimensions_eligible(const ggml_tensor * dst) {
    if(!dst || !dst->src[0] || !dst->src[1])return false;
    const auto * e=dst->src[0];const auto * w=dst->src[1];const int64_t n=e->ne[1];
    if(n<512 || n>INT32_MAX/2568 || e->ne[0]!=2568 || e->ne[2]!=1 || e->ne[3]!=1 || w->ne[0]!=2560 || w->ne[1]!=2560 || w->ne[2]!=1 || w->ne[3]!=1 || dst->ne[0]!=2560 || dst->ne[1]!=n || dst->ne[2]!=1 || dst->ne[3]!=1)return false;
    if(reinterpret_cast<uintptr_t>(e->data)%4 || reinterpret_cast<uintptr_t>(w->data)%4 || reinterpret_cast<uintptr_t>(dst->data)%4)return false;
    return i2s_epilogue_headers_readonly(dst);
}

template<bool matrix_format, bool row_aligned, bool optimized_route>
void launch_multiply(ggml_tensor * dst, void * stream) {
    const auto * weight = dst->src[1];
    const int logical = matrix_format ? weight->ne[0] : dst->src[0]->ne[0] - 8;
    const int stride = matrix_format ? logical / 4 : weight->nb[1];
    if constexpr (matrix_format && row_aligned) {
    if (i2s_vision_ffn_first_register_eligible(dst)) {
    multiply_direct_cached_epilogue_warp4<optimized_route><<<dim3((dst->ne[0]+127)/128,(dst->ne[1]+31)/32),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data),static_cast<const int8_t *>(dst->src[0]->data),static_cast<float *>(dst->data),dst->ne[0],dst->ne[1],dst->src[0]->ne[0]-8,stride,logical);
    return;
    }
    if(i2s_llm_ffndown_input8_eligible(dst)) {
    multiply_llm_ffndown_input8<optimized_route><<<dim3(20,(dst->ne[1]+31)/32),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data),static_cast<const int8_t *>(dst->src[0]->data),static_cast<float *>(dst->data),dst->ne[1]);
    return;
    }
    if(i2s_llm_ffndown_fixed_dimensions_eligible(dst)) {
    multiply_llm_ffndown_fixed_dimensions<optimized_route><<<dim3(20,(dst->ne[1]+31)/32),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data),static_cast<const int8_t *>(dst->src[0]->data),static_cast<float *>(dst->data),dst->ne[1]);
    return;
    }
    if(i2s_llm_attnout_fixed_dimensions_eligible(dst)) {
    multiply_llm_attnout_fixed_dimensions<optimized_route><<<dim3(20,(dst->ne[1]+31)/32),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data),static_cast<const int8_t *>(dst->src[0]->data),static_cast<float *>(dst->data),dst->ne[1]);
    return;
    }
    if (dst->ne[0] >= 2048 && dst->ne[1] >= 512) {
    if (i2s_epilogue_headers_readonly(dst)) {
    multiply_direct_cached_epilogue_warp4<optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    } else {
    multiply_direct_prefetch_warp4<optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    }
    } else {
    if (i2s_vision_attnout_packed_input8_eligible(dst)) {
    multiply_vision_attnout_packed_input8<optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 16), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    return;
    }
    if (i2s_epilogue_headers_readonly(dst)) {
    multiply_packed_cached_epilogue<optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 16), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    } else {
    multiply_packed_shared<optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 16), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    }
    }
    } else if constexpr (matrix_format && !row_aligned) {
    multiply_crossrow_prefetch<optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 16), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    } else {
    multiply<matrix_format, row_aligned, optimized_route><<<dim3((dst->ne[0] + 127) / 128, (dst->ne[1] + 31) / 32), dim3(32, 16), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const int8_t *>(dst->src[0]->data),
        static_cast<float *>(dst->data), dst->ne[0], dst->ne[1], dst->src[0]->ne[0] - 8, stride, logical);
    }
}
} // namespace

void i2s_encode_cuda(ggml_tensor * dst, void * stream, void *) {
    const int logical = dst->src[0]->ne[0], padded = dst->ne[0] - 8;
    if (padded == logical && (logical == 2560 || logical == 6912)) {
        if (logical == 2560)
            encode_cached_f32_scale_exact<10><<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(dst->src[0]->data),static_cast<int8_t *>(dst->data),logical,padded);
        else
            encode_cached_f32_scale_exact<27><<<dst->ne[1],256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(dst->src[0]->data),static_cast<int8_t *>(dst->data),logical,padded);
        return;
    }
    encode_f32_scale_exact<<<dst->ne[1], 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<int8_t *>(dst->data),
        dst->src[0]->ne[0], dst->ne[0] - 8);
}
void i2s_matmul_cuda(ggml_tensor * dst, void * stream, void * optimized_route) {
    const auto * weight = dst->src[1];
    if (weight->type == GGML_TYPE_I2_S) {
        if (weight->ne[0] % 128 == 0) {
            if (optimized_route) launch_multiply<true, true, true>(dst, stream);
            else launch_multiply<true, true, false>(dst, stream);
        } else {
            if (optimized_route) launch_multiply<true, false, true>(dst, stream);
            else launch_multiply<true, false, false>(dst, stream);
        }
    } else {
        if (optimized_route) launch_multiply<false, true, true>(dst, stream);
        else launch_multiply<false, true, false>(dst, stream);
    }
}
template<bool value_layout>
void i2s_matmul_attention_layout_cuda(ggml_tensor * dst,void * stream) {
    GGML_ASSERT(i2s_attention_layout_inputs_readonly(dst,value_layout));
    const auto * input=dst->src[0];const auto * weight=dst->src[1];const auto * bias=dst->src[2];
    const int m=weight->ne[1],n=input->ne[1],k=weight->ne[0],heads=dst->ne[2];
    multiply_packed_attention_layout_f32<value_layout><<<dim3((m+127)/128,(n+31)/32),dim3(32,16),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(weight->data),static_cast<const int8_t *>(input->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const float *>(bias->data),heads);
}
void i2s_matmul_query_key_layout_cuda(ggml_tensor * dst,void * stream,void *){i2s_matmul_attention_layout_cuda<false>(dst,stream);}
void i2s_matmul_value_layout_cuda(ggml_tensor * dst,void * stream,void *){i2s_matmul_attention_layout_cuda<true>(dst,stream);}

void i2s_matmul_packed_pair_cuda(ggml_tensor * dst,void * stream,void *) {
    GGML_ASSERT(i2s_packed_pair_inputs_readonly(dst));
    const auto * input=dst->src[0];const auto * w0=dst->src[1];const auto * w1=dst->src[2];
    const int m=dst->ne[0],n=dst->ne[1],k=w0->ne[0];
    // The complete existing packed-pair span/layout/int guard above still applies.
    // The existing selected-pair API and its M>=2048 contract are unchanged.
    if(i2s_kv_pair_selected_shape(k,m,n)) {
        multiply_direct_pair_warp4<true><<<dim3((m+127)/128,(n+31)/32,2),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t *>(w0->data),static_cast<const int8_t *>(input->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const uint8_t *>(w1->data));
        return;
    }
    multiply_packed_pair_cached_epilogue<true><<<dim3((m+127)/128,(n+31)/32,2),dim3(32,16),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(w0->data),static_cast<const int8_t *>(input->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const uint8_t *>(w1->data));
}

void i2s_matmul_qkv_cuda(ggml_tensor * dst,void * stream,void *) {
    GGML_ASSERT(i2s_qkv_inputs_readonly(dst));
    const int n=dst->src[0]->ne[1];
    multiply_qkv_composite_warp4<true><<<dim3(30,(n+31)/32),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(dst->src[1]->data),static_cast<const int8_t *>(dst->src[0]->data),static_cast<float *>(dst->data),n,2560,640,2560,
        static_cast<const uint8_t *>(dst->src[2]->data),static_cast<const uint8_t *>(dst->src[3]->data));
}

static bool i2s_llm_ffn_pair_f32_n48_eligible(const ggml_tensor * dst) {
    if (!i2s_pair_inputs_readonly(dst)) return false;
    const auto * input = dst->src[0];
    const auto * weights0 = dst->src[1];
    const auto * weights1 = dst->src[2];
    return weights0->ne[0] == 2560 && weights0->ne[1] == 6912 &&
        (reinterpret_cast<uintptr_t>(input->data) & 7) == 0 &&
        (reinterpret_cast<uintptr_t>(weights0->data) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(weights1->data) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(dst->data) & 3) == 0;
}

static bool i2s_llm_ffn_pair_input8_eligible(const ggml_tensor * dst) {
    if (!i2s_pair_inputs_readonly(dst)) return false;
    const auto * input = dst->src[0];
    const auto * weights0 = dst->src[1];
    const auto * weights1 = dst->src[2];
    return weights0->ne[0] == 2560 && weights0->ne[1] == 6912 &&
        (reinterpret_cast<uintptr_t>(input->data) & 7) == 0 &&
        (reinterpret_cast<uintptr_t>(weights0->data) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(weights1->data) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(dst->data) & 3) == 0;
}

void i2s_matmul_pair_cuda(ggml_tensor * dst,void * stream,void *) {
    GGML_ASSERT(i2s_pair_inputs_readonly(dst));
    const auto * input=dst->src[0];const auto * w0=dst->src[1];const auto * w1=dst->src[2];
    const int m=dst->ne[0],n=dst->ne[1],k=w0->ne[0];
    if (i2s_llm_ffn_pair_f32_n48_eligible(dst)) {
        multiply_llm_ffn_pair_f32_n48<true><<<dim3((m+127)/128,(n+47)/48,2),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t *>(w0->data),static_cast<const int8_t *>(input->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const uint8_t *>(w1->data));
        return;
    }
    if (i2s_llm_ffn_pair_input8_eligible(dst)) {
        multiply_llm_ffn_pair_input8<true><<<dim3((m+127)/128,(n+31)/32,2),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t *>(w0->data),static_cast<const int8_t *>(input->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const uint8_t *>(w1->data));
        return;
    }
    multiply_direct_pair_warp4<true><<<dim3((m+127)/128,(n+31)/32,2),dim3(32,8),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(w0->data),static_cast<const int8_t *>(input->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const uint8_t *>(w1->data));
}

void i2s_matmul_attention_qkv_cuda(ggml_tensor * dst,void * stream,void * userdata) {
    GGML_ASSERT(userdata);const int heads=*static_cast<const int *>(userdata);GGML_ASSERT(i2s_attention_qkv_inputs_readonly(dst,heads));
    const auto * encoded=dst->src[0];const auto * q=dst->src[1];const int k=q->ne[0],m=q->ne[1],n=encoded->ne[1];
    multiply_packed_attention_qkv<<<dim3((m+127)/128,(n+31)/32,3),dim3(32,16),0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t *>(q->data),static_cast<const int8_t *>(encoded->data),static_cast<float *>(dst->data),m,n,k,k/4,k,static_cast<const float *>(dst->src[4]->data),heads,
        static_cast<const uint8_t *>(dst->src[2]->data),static_cast<const uint8_t *>(dst->src[3]->data),static_cast<const float *>(dst->src[5]->data),static_cast<const float *>(dst->src[6]->data));
}

} // namespace robotcpp::bitvla
