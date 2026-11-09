#include "i2s.h"
#ifdef BITVLA_USE_CPU_SME
#include "i2s_sme.h"
#include <memory>
#include <sys/sysctl.h>
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
#include "scratch_003.h"
#endif
#endif
#include "ops.h"
#include <climits>
#include <vector>
#include "legacy_i2s_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || (defined(__APPLE__) && defined(__aarch64__))
#include <arm_neon.h>
#endif

namespace robotcpp::bitvla {
namespace {
// Stable marker passed through the existing custom-op userdata.
bool optimized_route_marker;

#if defined(__APPLE__) && defined(__aarch64__)
// Preserve the legacy double floor/scale and magic-bias rounding. Integer-bit
// absmax is order-independent for finite inputs; nonfinite rows retain the
// original ordered scan. Sum the clipped I8 bytes, not intermediate integers.
void quantize_row_i8_s_neon(const float * x, int8_t * dst, int64_t n, float * act_scales, int32_t * act_sums) {
#pragma clang fp contract(off)
    const auto overlaps = [](const void * a, size_t na, const void * b, size_t nb) {
        const auto aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
        return aa < bb + nb && bb < aa + na;
    };
    if (n < 16 || n > INT_MAX / 128 || overlaps(x, size_t(n) * sizeof(float), dst, size_t(n)) ||
        overlaps(x, size_t(n) * sizeof(float), act_scales, sizeof(float)) ||
        overlaps(x, size_t(n) * sizeof(float), act_sums, sizeof(int32_t)) ||
        overlaps(dst, size_t(n), act_scales, sizeof(float)) || overlaps(dst, size_t(n), act_sums, sizeof(int32_t)) ||
        overlaps(act_scales, sizeof(float), act_sums, sizeof(int32_t))) {
        legacy_cpu::quantize_row_i8_s(x, dst, n, act_scales, act_sums);
        return;
    }
    const auto absolute_mask = vdupq_n_u32(0x7fffffffu);
    auto maximum0 = vdupq_n_u32(0), maximum1 = maximum0;
    auto maximum2 = maximum0, maximum3 = maximum0;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        maximum0 = vmaxq_u32(maximum0, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i)), absolute_mask));
        maximum1 = vmaxq_u32(maximum1, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i + 4)), absolute_mask));
        maximum2 = vmaxq_u32(maximum2, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i + 8)), absolute_mask));
        maximum3 = vmaxq_u32(maximum3, vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x + i + 12)), absolute_mask));
    }
    uint32_t maximum_bits = vmaxvq_u32(vmaxq_u32(vmaxq_u32(maximum0, maximum1), vmaxq_u32(maximum2, maximum3)));
    for (; i < n; ++i) {
        uint32_t value;
        std::memcpy(&value, x + i, sizeof(value));
        maximum_bits = std::max(maximum_bits, value & 0x7fffffffu);
    }
    if (maximum_bits >= 0x7f800000u) {
        legacy_cpu::quantize_row_i8_s(x, dst, n, act_scales, act_sums);
        return;
    }
    float maximum_float;
    std::memcpy(&maximum_float, &maximum_bits, sizeof(maximum_float));
#if defined(BITVLA_USE_CPU_SME) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // Match CUDA101's exact-scale guard on the LLM encoder widths. The
    // maximum is already F32; its low bits certify the BF16 denominator.
    float scale;
    if ((n == 2560 || n == 6912) && (maximum_bits & 0xffffu) == 0 && maximum_float > 1e-5f) {
        scale = 127.0f / maximum_float;
    } else {
        const double maximum = std::max(0.00001, static_cast<double>(maximum_float));
        scale                = 127 / maximum;
    }
#else
    const double maximum = std::max(0.00001, static_cast<double>(maximum_float));
    const float scale    = 127 / maximum;
#endif
    *act_scales              = scale;
    const auto rounding_bias = vdupq_n_f32(12582912.f);
    const auto mantissa_mask = vdupq_n_u32(0x007fffffu);
    // nearest_int = masked_bits -0x400000; clip in that biased integer
    // domain. Its low byte is exactly the clipped signed I8 result.
    const auto lower     = vdupq_n_u32(0x00400000u - 128u);
    const auto upper     = vdupq_n_u32(0x00400000u + 127u);
    const auto quantize4 = [&](const float * p) {
        const auto scaled  = vmulq_n_f32(vld1q_f32(p), scale);
        const auto rounded = vaddq_f32(scaled, rounding_bias);
        const auto bits    = vandq_u32(vreinterpretq_u32_f32(rounded), mantissa_mask);
        return vreinterpretq_s32_u32(vmaxq_u32(lower, vminq_u32(upper, bits)));
    };
    auto partial_sums = vdupq_n_s32(0);
    i                 = 0;
    for (; i + 16 <= n; i += 16) {
        const auto low    = vmovn_high_s32(vmovn_s32(quantize4(x + i)), quantize4(x + i + 4));
        const auto high   = vmovn_high_s32(vmovn_s32(quantize4(x + i + 8)), quantize4(x + i + 12));
        const auto packed = vmovn_high_s16(vmovn_s16(low), high);
        vst1q_s8(dst + i, packed);
        partial_sums = vpadalq_s16(partial_sums, vpaddlq_s8(packed));
    }
    int32_t sum = vaddvq_s32(partial_sums);
    for (; i < n; ++i) {
        const int value = std::max(-128, std::min(127, legacy_cpu::nearest_int(x[i] * scale)));
        dst[i]          = static_cast<int8_t>(value);
        sum += value;
    }
    *act_sums = sum;
}
#endif

void encode(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * input = dst->src[0];
    const int logical = input->ne[0], padded = dst->ne[0] - 8;
#if defined(__APPLE__) && defined(__aarch64__)
    // Tensor metadata stays fixed during this callback; do not cache data values.
    const int64_t tokens      = input->ne[1];
    const auto * input_data   = static_cast<const char *>(input->data);
    auto * output_data        = static_cast<char *>(dst->data);
    const size_t input_stride = input->nb[1], output_stride = dst->nb[1];
    for (int token = ith; token < tokens; token += nth) {
        const auto * x = reinterpret_cast<const float *>(input_data + token * input_stride);
        auto * q       = reinterpret_cast<int8_t *>(output_data + token * output_stride);
        float scale;
        int32_t sum;
        quantize_row_i8_s_neon(x, q, logical, &scale, &sum);
        std::fill(q + logical, q + padded, 0);
        std::memcpy(q + padded, &scale, sizeof(scale));
        std::memcpy(q + padded + 4, &sum, sizeof(sum));
    }
#else
    for (int token = ith; token < input->ne[1]; token += nth) {
        const auto * x = reinterpret_cast<const float *>(static_cast<const char *>(input->data) + token * input->nb[1]);
        auto * q       = reinterpret_cast<int8_t *>(static_cast<char *>(dst->data) + token * dst->nb[1]);
        float scale;
        int32_t sum;
        legacy_cpu::quantize_row_i8_s(x, q, logical, &scale, &sum);
        std::fill(q + logical, q + padded, 0);
        std::memcpy(q + padded, &scale, sizeof(scale));
        std::memcpy(q + padded + 4, &sum, sizeof(sum));
    }
#endif
}

float dot(const uint8_t * packed, const int8_t * x, int k) {
#ifdef __AVX2__
    float result;
    legacy_cpu::ggml_vec_dot_i2_i8_s(k, &result, 0, packed, 0, x, 0, 1);
    return result;
#else
    int32_t result = 0;
    int i          = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);
    for (; i + 128 <= k; i += 128) {
        const auto low     = vld1q_u8(packed + i / 4);
        const auto high    = vld1q_u8(packed + i / 4 + 16);
        const auto consume = [&](uint8x16_t a, uint8x16_t b, int offset) {
            sum0 = vdotq_s32(sum0, vreinterpretq_s8_u8(vandq_u8(a, mask)), vld1q_s8(x + i + offset));
            sum1 = vdotq_s32(sum1, vreinterpretq_s8_u8(vandq_u8(b, mask)), vld1q_s8(x + i + offset + 16));
        };
        // Each packed byte holds the codes at offsets 0, 32, 64 and 96.
        consume(vshrq_n_u8(low, 6), vshrq_n_u8(high, 6), 0);
        consume(vshrq_n_u8(low, 4), vshrq_n_u8(high, 4), 32);
        consume(vshrq_n_u8(low, 2), vshrq_n_u8(high, 2), 64);
        consume(low, high, 96);
    }
    result = vaddvq_s32(vaddq_s32(sum0, sum1));
#endif
    for (; i < k; ++i) {
        const int code = (packed[(i / 128) * 32 + i % 32] >> (6 - 2 * ((i % 128) / 32))) & 3;
        result += code * x[i];
    }
    return result;
#endif
}

// The original format packs the entire matrix, so an unaligned row may share
// a 128-value group with its neighbours. Read those codes by global index.
float matrix_dot(const uint8_t * packed, const int8_t * x, int row, int k) {
    if (k % 128 == 0)
        return dot(packed + int64_t(row) * k / 4, x, k);
    int i          = 0;
    int32_t result = 0;
#ifdef __AVX2__
    const uint64_t base = uint64_t(row) * k;
    __m256i sum         = _mm256_setzero_si256();
    const __m256i mask = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    const auto codes16 = [packed](uint64_t index) {
        const int shift = 6 - 2 * ((index % 128) / 32);
        auto codes      = _mm_loadu_si128(reinterpret_cast<const __m128i *>(packed + (index / 128) * 32 + index % 32));
        return _mm_and_si128(_mm_srl_epi16(codes, _mm_cvtsi32_si128(shift)), _mm_set1_epi8(3));
    };
    const auto codes32 = [packed, &codes16, mask](uint64_t index) {
        if (index % 32 == 0) {
            const int shift = 6 - 2 * ((index % 128) / 32);
            auto codes      = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed + (index / 128) * 32));
            return _mm256_and_si256(_mm256_srl_epi16(codes, _mm_cvtsi32_si128(shift)), mask);
        }
        return _mm256_set_m128i(codes16(index + 16), codes16(index));
    };
    const int phase = base % 128, quarter = phase / 32;
    const bool split = phase % 32 != 0;
    // Preserve the original I16 wrapping and its 4096-value flush boundary.
    for (int first = 0; k % 16 == 0 && first < k; first += 4096) {
        __m256i accumulator = _mm256_setzero_si256();
        const int end       = std::min(k, first + 4096);
        for (i = first; i + 128 <= end; i += 128) {
            const auto * group = packed + ((base + i) / 128) * 32;
            const auto a       = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(group));
            const auto b =
                phase ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(group + 32)) : _mm256_setzero_si256();
            const auto consume = [&](int offset) {
                const int q        = quarter + offset;
                const auto current = q < 4 ? a : b;
                const int shift    = 6 - 2 * (q % 4);
                __m256i codes;
                if (!split) {
                    codes = _mm256_srl_epi16(current, _mm_cvtsi32_si128(shift));
                } else {
                    const auto next      = q + 1 < 4 ? a : b;
                    const int next_shift = 6 - 2 * ((q + 1) % 4);
                    const auto low  = _mm_srl_epi16(_mm256_extracti128_si256(current, 1), _mm_cvtsi32_si128(shift));
                    const auto high = _mm_srl_epi16(_mm256_castsi256_si128(next), _mm_cvtsi32_si128(next_shift));
                    codes           = _mm256_set_m128i(high, low);
                }
                const auto values  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i + offset * 32));
                const auto partial = _mm256_maddubs_epi16(_mm256_and_si256(codes, mask), values);
                accumulator        = _mm256_add_epi16(accumulator, partial);
            };
            consume(0);
            consume(1);
            consume(2);
            consume(3);
        }
        for (; i + 32 <= end; i += 32) {
            const auto codes  = codes32(base + i);
            const auto values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i));
            accumulator       = _mm256_add_epi16(accumulator, _mm256_maddubs_epi16(codes, values));
        }
        if (i < end) {
            const auto codes   = codes16(base + i);
            const auto values  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
            const auto partial = _mm_maddubs_epi16(codes, values);
            accumulator        = _mm256_add_epi16(accumulator, _mm256_set_m128i(_mm_setzero_si128(), partial));
            i += 16;
        }
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(accumulator, ones));
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), sum);
    for (int lane : lanes)
        result += lane;
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint64_t base = uint64_t(row) * k;
    // A 16-value span never crosses a packed 32-code quarter once aligned.
    for (; i < k && (base + i) % 16 != 0; ++i) {
        const uint64_t index = base + i;
        const int code       = (packed[(index / 128) * 32 + index % 32] >> (6 - 2 * ((index % 128) / 32))) & 3;
        result += code * x[i];
    }
    int32x4_t sum         = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);
    for (; i + 16 <= k; i += 16) {
        const uint64_t index = base + i;
        const int shift      = 6 - 2 * ((index % 128) / 32);
        const auto bytes     = vld1q_u8(packed + (index / 128) * 32 + index % 32);
        const auto codes     = vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-shift)), mask);
        sum                  = vdotq_s32(sum, vreinterpretq_s8_u8(codes), vld1q_s8(x + i));
    }
    result += vaddvq_s32(sum);
#endif
    for (; i < k; ++i) {
        const int64_t index = int64_t(row) * k + i;
        const int code      = (packed[(index / 128) * 32 + index % 32] >> (6 - 2 * ((index % 128) / 32))) & 3;
        result += code * x[i];
    }
    return static_cast<float>(result);
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
// Decode one packed weight row once for eight independent activation tokens.
// Packing remains the native matrix-wide 128-code / 32-byte format.
void dot_eight_tokens(const uint8_t * packed, const int8_t * input, size_t token_stride, int k, float * products) {
    int32x4_t sums[8];
    for (int token = 0; token < 8; ++token)
        sums[token] = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);
    for (int i = 0; i < k; i += 128) {
        const auto low     = vld1q_u8(packed + i / 4);
        const auto high    = vld1q_u8(packed + i / 4 + 16);
        const auto consume = [&](uint8x16_t a, uint8x16_t b, int offset) {
            const auto codes0 = vreinterpretq_s8_u8(vandq_u8(a, mask));
            const auto codes1 = vreinterpretq_s8_u8(vandq_u8(b, mask));
#pragma clang loop unroll(full)
            for (int token = 0; token < 8; ++token) {
                const auto * x = input + token * token_stride + i + offset;
                sums[token]    = vdotq_s32(sums[token], codes0, vld1q_s8(x));
                sums[token]    = vdotq_s32(sums[token], codes1, vld1q_s8(x + 16));
            }
        };
        consume(vshrq_n_u8(low, 6), vshrq_n_u8(high, 6), 0);
        consume(vshrq_n_u8(low, 4), vshrq_n_u8(high, 4), 32);
        consume(vshrq_n_u8(low, 2), vshrq_n_u8(high, 2), 64);
        consume(low, high, 96);
    }
    for (int token = 0; token < 8; ++token)
        products[token] = static_cast<float>(vaddvq_s32(sums[token]));
}

void matmul_native_blocked(ggml_tensor * dst, int ith, int nth, const uint8_t * matrix, int k, float weight_scale) {
    const auto * input = dst->src[0];
    const int64_t rows = dst->ne[0], tokens = dst->ne[1];
    constexpr int64_t row_block = 8, token_block = 256;
    const int64_t row_blocks   = (rows + row_block - 1) / row_block;
    const int64_t token_blocks = (tokens + token_block - 1) / token_block;
    auto * output              = static_cast<float *>(dst->data);
    for (int64_t block = ith; block < row_blocks * token_blocks; block += nth) {
        const int64_t first_row   = (block % row_blocks) * row_block;
        const int64_t first_token = (block / row_blocks) * token_block;
        const int64_t row_end     = std::min(rows, first_row + row_block);
        const int64_t token_end   = std::min(tokens, first_token + token_block);
        for (int64_t row = first_row; row < row_end; ++row) {
            const auto * packed = matrix + row * k / 4;
            int64_t token       = first_token;
            for (; token + 8 <= token_end; token += 8) {
                const auto * q =
                    reinterpret_cast<const int8_t *>(static_cast<const char *>(input->data) + token * input->nb[1]);
                float products[8];
                dot_eight_tokens(packed, q, input->nb[1], k, products);
                for (int t = 0; t < 8; ++t) {
                    const auto * encoded = q + t * input->nb[1];
                    float activation_scale;
                    int32_t sum;
                    std::memcpy(&activation_scale, encoded + k, sizeof(activation_scale));
                    std::memcpy(&sum, encoded + k + 4, sizeof(sum));
                    const float value                = ((products[t] - sum) / activation_scale) * weight_scale;
                    output[(token + t) * rows + row] = value;
                }
            }
            for (; token < token_end; ++token) {
                const auto * q =
                    reinterpret_cast<const int8_t *>(static_cast<const char *>(input->data) + token * input->nb[1]);
                float activation_scale;
                int32_t sum;
                std::memcpy(&activation_scale, q + k, sizeof(activation_scale));
                std::memcpy(&sum, q + k + 4, sizeof(sum));
                const float product        = dot(packed, q, k);
                const float value          = ((product - sum) / activation_scale) * weight_scale;
                output[token * rows + row] = value;
            }
        }
    }
}
#endif

#ifdef BITVLA_USE_CPU_SME
int integer_sme_words() {
    static const int words = [] {
        for (const char * feature : {"hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME2"}) {
            int available = 0;
            size_t size   = sizeof(available);
            if (sysctlbyname(feature, &available, &size, nullptr, 0) != 0 || available != 1)
                return 0;
        }
        const auto width = cpu_sme_words();
        return width >= 1 && width <= 64 ? static_cast<int>(width) : 0;
    }();
    return words;
}

template <bool four_rows = false, class Load>
void pack_sme_rows(int8_t * output, int rows, int k, int words, Load load) {
    const int pair   = 2 * words;
    const int padded = (rows + pair - 1) / pair * pair;
    for (int first = 0; first < padded; first += words) {
        auto * block    = output + size_t(first) * k;
        const int count = std::clamp(rows - first, 0, words);
        if (count < words)
            std::memset(block, 0, size_t(words) * k);
        int row = 0;
        if constexpr (four_rows) {
            if (words % 4 == 0) {
                for (; row + 4 <= count; row += 4) {
                    for (int col = 0; col < k; col += 16) {
                        const auto w0 = load(first + row, col);
                        const auto w1 = load(first + row + 1, col);
                        const auto w2 = load(first + row + 2, col);
                        const auto w3 = load(first + row + 3, col);
                        const auto a = vtrn1q_u32(w0, w1), b = vtrn2q_u32(w0, w1);
                        const auto c = vtrn1q_u32(w2, w3), d = vtrn2q_u32(w2, w3);
                        // Each K group gets four adjacent rows in one store.
                        uint32x4_t low_ac, low_bd;
                        __asm__("zip1 %0.2d, %1.2d, %2.2d" : "=w"(low_ac) : "w"(a), "w"(c));
                        __asm__("zip1 %0.2d, %1.2d, %2.2d" : "=w"(low_bd) : "w"(b), "w"(d));
                        const uint32x4_t groups[] = {
                            low_ac,
                            low_bd,
                            vcombine_u32(vget_high_u32(a), vget_high_u32(c)),
                            vcombine_u32(vget_high_u32(b), vget_high_u32(d)),
                        };
                        auto * group = block + size_t(col) * words + row * 4;
                        for (int j = 0; j < 4; ++j)
                            vst1q_s8(group + j * words * 4, vreinterpretq_s8_u32(groups[j]));
                    }
                }
            }
        }
        for (; row < count; ++row) {
            for (int col = 0; col < k; col += 16) {
                const auto lanes        = load(first + row, col);
                const uint32_t chunks[] = {vgetq_lane_u32(lanes, 0), vgetq_lane_u32(lanes, 1), vgetq_lane_u32(lanes, 2),
                                           vgetq_lane_u32(lanes, 3)};
                auto * group            = block + size_t(col) * words + row * 4;
                for (int j = 0; j < 4; ++j)
                    std::memcpy(group + j * words * 4, &chunks[j], 4);
            }
        }
    }
}

#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
// C006: private KV packing only; same four-row byte transpose/store layout.
template <int Shift>
void store_kv_quarter_c006(int8_t * group, uint8x16_t p0, uint8x16_t p1, uint8x16_t p2, uint8x16_t p3) {
    const auto mask   = vdupq_n_u8(3);
    const auto decode = [&](uint8x16_t x) {
        if constexpr (Shift == 0)
            return vreinterpretq_u32_u8(vandq_u8(x, mask));
        else
            return vreinterpretq_u32_u8(vandq_u8(vshrq_n_u8(x, Shift), mask));
    };
    const auto w0 = decode(p0), w1 = decode(p1), w2 = decode(p2), w3 = decode(p3);
    const auto a = vtrn1q_u32(w0, w1), b = vtrn2q_u32(w0, w1), c = vtrn1q_u32(w2, w3), d = vtrn2q_u32(w2, w3);
    uint32x4_t low_ac, low_bd;
    __asm__("zip1 %0.2d, %1.2d, %2.2d" : "=w"(low_ac) : "w"(a), "w"(c));
    __asm__("zip1 %0.2d, %1.2d, %2.2d" : "=w"(low_bd) : "w"(b), "w"(d));
    const uint32x4_t groups[] = {low_ac, low_bd, vcombine_u32(vget_high_u32(a), vget_high_u32(c)),
                                 vcombine_u32(vget_high_u32(b), vget_high_u32(d))};
    for (int j = 0; j < 4; ++j)
        vst1q_s8(group + j * 64, vreinterpretq_s8_u32(groups[j]));
}
void pack_kv_groups_c006(int8_t * output, const uint8_t * matrix, int first, int rows) {
    constexpr int k = 2560, words = 16;
    // Exceptional helper tails use the exact old packer; actual eight-worker
    // KV bands are 64 or 96 rows and take the grouped decoder below.
    if (rows % 4) {
        pack_sme_rows<true>(output, rows, k, words, [&](int row, int col) {
            const size_t index = size_t(first + row) * k + col;
            const int shift    = 6 - 2 * ((index % 128) / 32);
            const auto bytes   = vld1q_u8(matrix + (index / 128) * 32 + index % 32);
            return vreinterpretq_u32_u8(vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-shift)), vdupq_n_u8(3)));
        });
        return;
    }
    const int padded = (rows + 31) / 32 * 32;
    for (int block_first = 0; block_first < padded; block_first += words) {
        auto * block    = output + size_t(block_first) * k;
        const int count = std::clamp(rows - block_first, 0, words);
        if (count < words)
            std::memset(block, 0, size_t(words) * k);
        for (int row = 0; row < count; row += 4)
            for (int col = 0; col < k; col += 128) {
                uint8x16_t low[4], high[4];
                for (int r = 0; r < 4; ++r) {
                    const auto * packed = matrix + size_t(first + block_first + row + r) * (k / 4) + col / 4;
                    low[r]              = vld1q_u8(packed);
                    high[r]             = vld1q_u8(packed + 16);
                }
                auto * group = block + size_t(col) * words + row * 4;
                store_kv_quarter_c006<6>(group, low[0], low[1], low[2], low[3]);
                store_kv_quarter_c006<6>(group + 16 * words, high[0], high[1], high[2], high[3]);
                store_kv_quarter_c006<4>(group + 32 * words, low[0], low[1], low[2], low[3]);
                store_kv_quarter_c006<4>(group + 48 * words, high[0], high[1], high[2], high[3]);
                store_kv_quarter_c006<2>(group + 64 * words, low[0], low[1], low[2], low[3]);
                store_kv_quarter_c006<2>(group + 80 * words, high[0], high[1], high[2], high[3]);
                store_kv_quarter_c006<0>(group + 96 * words, low[0], low[1], low[2], low[3]);
                store_kv_quarter_c006<0>(group + 112 * words, high[0], high[1], high[2], high[3]);
            }
    }
}
#endif

bool valid_sme_scales(const int8_t * scales, size_t stride, int64_t tokens) {
    const auto mask     = vdupq_n_u32(0x7fffffffU);
    const auto infinity = vdupq_n_u32(0x7f800000U);
    const auto valid4   = [&](const int8_t * group) {
        uint32_t bits[4];
        for (int lane = 0; lane < 4; ++lane)
            std::memcpy(&bits[lane], group + lane * stride, sizeof(bits[lane]));
        const auto absolute = vandq_u32(vld1q_u32(bits), mask);
        return vandq_u32(vcgtq_u32(absolute, vdupq_n_u32(0)), vcltq_u32(absolute, infinity));
    };
    int64_t token = 0;
    for (; token + 4 <= tokens; token += 4)
        if (vminvq_u32(valid4(scales + token * stride)) == 0)
            return false;
    for (; token < tokens; ++token) {
        float scale;
        std::memcpy(&scale, scales + token * stride, sizeof(scale));
        if (!std::isfinite(scale) || scale == 0.f)
            return false;
    }
    return true;
}

struct ContiguousSmeOutput {
    void write4(float * target, int, int, float32x4_t value) const { vst1q_f32(target, value); }
    void write1(float * target, int, int, float value) const { *target = value; }
};

#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
struct AttentionSmeOutput {
    float * output;
    const float * bias;
    int width, tokens;
    bool value_layout;
    void write4(float *, int row, int token, float32x4_t value) const {
#pragma clang fp contract(off)
        // Constructor guards finite bias and normal weight scale. Keep this
        // F32 addition separate from the existing restoration multiply.
        value          = vaddq_f32(value, vld1q_f32(bias + row));
        const int head = row / width, col = row % width;
        if (!value_layout)
            vst1q_f32(output + (head * tokens + token) * width + col, value);
        else {
            auto * target      = output + (head * width + col) * tokens + token;
            target[0]          = vgetq_lane_f32(value, 0);
            target[tokens]     = vgetq_lane_f32(value, 1);
            target[2 * tokens] = vgetq_lane_f32(value, 2);
            target[3 * tokens] = vgetq_lane_f32(value, 3);
        }
    }
    void write1(float *, int row, int token, float value) const {
#pragma clang fp contract(off)
        const int head = row / width, col = row % width;
        const int index = value_layout ? row * tokens + token : (head * tokens + token) * width + col;
        output[index]   = value + bias[row];
    }
};
#endif

template <class Output = ContiguousSmeOutput>
bool matmul_sme(ggml_tensor * dst, int ith, int nth, int k, int logical, int rows, float weight_scale,
                Output store = {}) {
#pragma clang fp contract(off)
    const auto * encoded = dst->src[0];
    const auto * weight  = dst->src[1];
    const int64_t n      = dst->ne[1];
    const bool language =
        logical == k && ((k == 2560 && (rows == 6912 || rows == 2560 || rows == 640)) || (k == 6912 && rows == 2560));
    const bool vision = n == 256 && ((logical == 1152 && k == 1152 && (rows == 1152 || rows == 4304)) ||
                                     (logical == 4304 && k == 4352 && rows == 1152));
    if ((!language && !vision) || weight->ne[1] != rows || n < 32 || n > INT_MAX / std::max(k + 8, rows) ||
        n != encoded->ne[1] || dst->ne[2] != 1 || dst->ne[3] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
        encoded->ne[2] != 1 || encoded->ne[3] != 1 || !ggml_is_contiguous(weight) || !std::isfinite(weight_scale) ||
        weight_scale == 0.f)
        return false;
    const int words = integer_sme_words();
    if (!words)
        return false;

    const auto * activations = static_cast<const int8_t *>(encoded->data);
    const size_t stride      = encoded->nb[1];
    // All workers make the same decision so their output ranges never overlap.
    if (!valid_sme_scales(activations + k, stride, n))
        return false;

    const int pair      = 2 * words;
    const int row_pairs = (rows + pair - 1) / pair;
    const int first     = row_pairs * ith / nth * pair;
    const int end       = std::min(rows, row_pairs * (ith + 1) / nth * pair);
    if (first >= end)
        return true;
    const int local_rows  = end - first;
    const int padded_rows = (local_rows + pair - 1) / pair * pair;
    const int panel       = 2 * pair;
    // One weight band per worker and a bounded token panel, private to this call.
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    const bool reuse_ffn =
        words == 16 && nth == 8 && logical == k && ((k == 2560 && rows == 6912) || (k == 6912 && rows == 2560));
    SmeScratchLease003 scratch(reuse_ffn, size_t(padded_rows) * k, size_t(panel) * k, size_t(local_rows) * panel);
    const auto & packed_w = scratch.weights();
    const auto & packed_a = scratch.activations();
    const auto & products = scratch.products();
#else
    std::unique_ptr<int8_t[]> packed_w(new int8_t[size_t(padded_rows) * k]);
    std::unique_ptr<int8_t[]> packed_a(new int8_t[size_t(panel) * k]);
    std::unique_ptr<int32_t[]> products(new int32_t[size_t(local_rows) * panel]);
#endif
    const auto * matrix    = static_cast<const uint8_t *>(weight->data);
    const auto mask        = vdupq_n_u8(3);
    const auto load_weight = [&](int row, int col) {
        // ViT FFN has 48 extra columns in the SME scratch buffer. The stored
        // matrix still has 4304 columns per row and shares blocks across rows.
        if (col >= logical)
            return vdupq_n_u32(0);
        const size_t index = size_t(first + row) * logical + col;
        const int shift    = 6 - 2 * ((index % 128) / 32);
        const auto bytes   = vld1q_u8(matrix + (index / 128) * 32 + index % 32);
        return vreinterpretq_u32_u8(vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-shift)), mask));
    };
    if (logical == k)
        pack_sme_rows<true>(packed_w.get(), local_rows, k, words, load_weight);
    else
        pack_sme_rows(packed_w.get(), local_rows, k, words, load_weight);
    auto * output      = static_cast<float *>(dst->data);
    const auto weights = vdupq_n_f32(weight_scale);
    for (int start = 0; start < n; start += panel) {
        const int tokens = std::min(panel, static_cast<int>(n) - start);
        pack_sme_rows<true>(packed_a.get(), tokens, k, words, [&](int row, int col) {
            return vreinterpretq_u32_s8(vld1q_s8(activations + (start + row) * stride + col));
        });
        cpu_sme_i8(packed_w.get(), packed_a.get(), products.get(), local_rows, tokens, k / 4, words, 0, 1);
        for (int token = 0; token < tokens; ++token) {
            const auto * q = activations + (start + token) * stride;
            float scale;
            int32_t sum;
            std::memcpy(&scale, q + k, sizeof(scale));
            std::memcpy(&sum, q + k + 4, sizeof(sum));
            const auto * dot  = products.get() + size_t(token) * local_rows;
            auto * restored   = output + size_t(start + token) * rows + first;
            const auto sums   = vdupq_n_f32(static_cast<float>(sum));
            const auto scales = vdupq_n_f32(scale);
            int row           = 0;
            for (; row + 4 <= local_rows; row += 4) {
                const auto values = vcvtq_f32_s32(vld1q_s32(dot + row));
                store.write4(restored + row, first + row, start + token,
                             vmulq_f32(vdivq_f32(vsubq_f32(values, sums), scales), weights));
            }
            for (; row < local_rows; ++row) {
                const float product = static_cast<float>(dot[row]);
                store.write1(restored + row, first + row, start + token, ((product - sum) / scale) * weight_scale);
            }
        }
    }
    return true;
}
#endif

void matmul(ggml_tensor * dst, int ith, int nth, void * optimized_route) {
    const auto * input  = dst->src[0];
    const auto * weight = dst->src[1];
    const int k = input->ne[0] - 8, rows = dst->ne[0];
    const bool matrix_format = weight->type == GGML_TYPE_I2_S;
    const int logical        = matrix_format ? weight->ne[0] : k;
    const bool cross_row     = matrix_format && logical % 128 != 0;
    const auto * matrix      = static_cast<const uint8_t *>(weight->data);
    float matrix_scale       = 0;
    if (matrix_format)
        std::memcpy(&matrix_scale, matrix + int64_t(rows) * logical / 4, sizeof(matrix_scale));
#ifdef BITVLA_USE_CPU_SME
    // The scheduler already selected the CPU for this node.
    if (matrix_format && matmul_sme(dst, ith, nth, k, logical, rows, matrix_scale))
        return;
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Keep the existing NaN payload propagation for nonfinite weight scales.
    if (matrix_format && !cross_row && k > 0 && logical == k && std::isfinite(matrix_scale) && rows / 8 >= nth &&
        dst->ne[1] >= 8 && dst->ne[2] == 1 && dst->ne[3] == 1) {
        matmul_native_blocked(dst, ith, nth, matrix, k, matrix_scale);
        return;
    }
#endif
    auto * output = static_cast<float *>(dst->data);
    for (int64_t index = ith; index < ggml_nelements(dst); index += nth) {
        const int token = index / rows, row = index % rows;
        const auto * q =
            reinterpret_cast<const int8_t *>(static_cast<const char *>(input->data) + token * input->nb[1]);
        float activation_scale, weight_scale;
        int32_t sum;
        std::memcpy(&activation_scale, q + k, sizeof(activation_scale));
        std::memcpy(&sum, q + k + 4, sizeof(sum));
        const auto * packed = matrix_format ? matrix + int64_t(row) * logical / 4 : matrix + row * weight->nb[1];
        weight_scale        = matrix_scale;
        if (!matrix_format) {
            std::memcpy(&weight_scale, packed, sizeof(weight_scale));
            packed += 4;
        }
        const float product = cross_row ? matrix_dot(matrix, q, row, logical) : dot(packed, q, k);
        const float value   = ((product - sum) / activation_scale) * weight_scale;
        output[index]       = value;
    }
}
// Keep the unrounded original matmul, then call the original CPU bias/layout body.
int value_layout_marker;
void matmul_attention_layout(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0)
        return;
    GGML_UNUSED(nth);
    const auto * encoded = dst->src[0];
    const auto * weight  = dst->src[1];
    const int64_t m = weight->ne[1], n = encoded->ne[1];
    std::vector<float> temporary(static_cast<size_t>(m * n));
    ggml_tensor product = *dst;
    product.ne[0]       = m;
    product.ne[1]       = n;
    product.ne[2] = product.ne[3] = 1;
    product.nb[0]                 = sizeof(float);
    product.nb[1]                 = m * sizeof(float);
    product.nb[2] = product.nb[3] = m * n * sizeof(float);
    product.data                  = temporary.data();
    product.src[2]                = nullptr;
    matmul(&product, 0, 1, nullptr);
    ggml_tensor layout = *dst;
    layout.src[0]      = &product;
    layout.src[1]      = dst->src[2];
    layout.src[2]      = nullptr;
    bias_attention_layout_cpu_existing(&layout, 0, 1, userdata != nullptr);
}
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
void matmul_attention_layout_sme(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * encoded = dst->src[0];
    const auto * weight  = dst->src[1];
    const int k = int(weight->ne[0]), rows = int(weight->ne[1]), tokens = int(encoded->ne[1]);
    float scale;
    std::memcpy(&scale, static_cast<const char *>(weight->data) + size_t(k) * rows / 4, sizeof(scale));
    ggml_tensor product = *dst;
    product.ne[0]       = rows;
    product.ne[1]       = tokens;
    product.ne[2] = product.ne[3] = 1;
    product.nb[0]                 = 4;
    product.nb[1]                 = size_t(rows) * 4;
    product.nb[2] = product.nb[3] = size_t(rows) * tokens * 4;
    product.src[2]                = nullptr;
    AttentionSmeOutput store{static_cast<float *>(dst->data), static_cast<const float *>(dst->src[2]->data),
                             rows / int(dst->ne[2]), tokens, userdata != nullptr};
    if (matmul_sme(&product, ith, nth, k, k, rows, scale, store))
        return;
    // Uniform runtime scale rejection retains the original compatibility body.
    // Its one-worker fallback is exceptional; the guarded normal SME route
    // above retains the original row_pairs ownership across all workers.
    matmul_attention_layout(dst, ith, nth, userdata);
}
#endif
void matmul_attention_qkv(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0)
        return;
    GGML_UNUSED(nth);
    GGML_ASSERT(userdata);
    const int heads = *static_cast<const int *>(userdata);
    const int64_t m = dst->src[1]->ne[1], n = dst->src[0]->ne[1], dim = m / heads;
    for (int family = 0; family < 3; ++family) {
        ggml_tensor projection = *dst;
        projection.ne[0]       = family == 2 ? n : dim;
        projection.ne[1]       = family == 2 ? dim : n;
        projection.ne[2]       = heads;
        projection.ne[3]       = 1;
        projection.nb[0]       = sizeof(float);
        projection.nb[1]       = projection.ne[0] * sizeof(float);
        projection.nb[2]       = projection.nb[1] * projection.ne[1];
        projection.nb[3]       = projection.nb[2] * heads;
        projection.data        = static_cast<float *>(dst->data) + int64_t(family) * m * n;
        for (auto & source : projection.src)
            source = nullptr;
        projection.src[0] = dst->src[0];
        projection.src[1] = dst->src[1 + family];
        projection.src[2] = dst->src[4 + family];
        matmul_attention_layout(&projection, 0, 1, family == 2 ? &value_layout_marker : nullptr);
    }
}

#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
// CPU adaptation of CUDA176: share the current input panel across the three
// original SME projections. Keep every projection's row band and F32 writer.
bool matmul_attention_qkv_sme_body(ggml_tensor * dst, int ith, int nth) {
#pragma clang fp contract(off)
    constexpr int k = 1152, rows = 1152, tokens_total = 256, heads = 16;
    const auto * encoded = dst->src[0];
    const int words      = integer_sme_words();
    if (words != 16 || nth != 8 || encoded->ne[0] != k + 8 || encoded->ne[1] != tokens_total ||
        !ggml_is_contiguous(encoded))
        return false;
    float weight_scales[3];
    for (int family = 0; family < 3; ++family) {
        const auto * weight = dst->src[1 + family];
        std::memcpy(&weight_scales[family], static_cast<const char *>(weight->data) + size_t(k) * rows / 4, 4);
        if (!std::isnormal(weight_scales[family]))
            return false;
    }
    const auto * activations = static_cast<const int8_t *>(encoded->data);
    const size_t stride      = encoded->nb[1];
    // The decision is uniform before any worker writes output.
    if (!valid_sme_scales(activations + k, stride, tokens_total))
        return false;
    const int pair = 2 * words, row_pairs = (rows + pair - 1) / pair;
    const int first = row_pairs * ith / nth * pair, end = std::min(rows, row_pairs * (ith + 1) / nth * pair);
    if (first >= end)
        return true;
    const int local_rows = end - first, padded_rows = (local_rows + pair - 1) / pair * pair, panel = 2 * pair;
    std::unique_ptr<int8_t[]> packed_w[3];
    for (auto & buffer : packed_w)
        buffer.reset(new int8_t[size_t(padded_rows) * k]);
    std::unique_ptr<int8_t[]> packed_a(new int8_t[size_t(panel) * k]);
    std::unique_ptr<int32_t[]> products(new int32_t[size_t(local_rows) * panel]);
    const auto mask = vdupq_n_u8(3);
    for (int family = 0; family < 3; ++family) {
        const auto * matrix    = static_cast<const uint8_t *>(dst->src[1 + family]->data);
        const auto load_weight = [&](int row, int col) {
            const size_t index = size_t(first + row) * k + col;
            const int shift    = 6 - 2 * ((index % 128) / 32);
            const auto bytes   = vld1q_u8(matrix + (index / 128) * 32 + index % 32);
            return vreinterpretq_u32_u8(vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-shift)), mask));
        };
        pack_sme_rows<true>(packed_w[family].get(), local_rows, k, words, load_weight);
    }
    for (int start = 0; start < tokens_total; start += panel) {
        const int tokens = std::min(panel, tokens_total - start);
        pack_sme_rows<true>(packed_a.get(), tokens, k, words, [&](int row, int col) {
            return vreinterpretq_u32_s8(vld1q_s8(activations + (start + row) * stride + col));
        });
        for (int family = 0; family < 3; ++family) {
            auto * output = static_cast<float *>(dst->data) + size_t(family) * rows * tokens_total;
            AttentionSmeOutput store{output, static_cast<const float *>(dst->src[4 + family]->data), rows / heads,
                                     tokens_total, family == 2};
            const float weight_scale = weight_scales[family];
            const auto weights       = vdupq_n_f32(weight_scale);
            cpu_sme_i8(packed_w[family].get(), packed_a.get(), products.get(), local_rows, tokens, k / 4, words, 0, 1);
            for (int token = 0; token < tokens; ++token) {
                const auto * q = activations + (start + token) * stride;
                float scale;
                int32_t sum;
                std::memcpy(&scale, q + k, sizeof(scale));
                std::memcpy(&sum, q + k + 4, sizeof(sum));
                const auto * dot  = products.get() + size_t(token) * local_rows;
                auto * restored   = output + size_t(start + token) * rows + first;
                const auto sums   = vdupq_n_f32(static_cast<float>(sum));
                const auto scales = vdupq_n_f32(scale);
                int row           = 0;
                for (; row + 4 <= local_rows; row += 4) {
                    const auto values = vcvtq_f32_s32(vld1q_s32(dot + row));
                    store.write4(restored + row, first + row, start + token,
                                 vmulq_f32(vdivq_f32(vsubq_f32(values, sums), scales), weights));
                }
                for (; row < local_rows; ++row) {
                    const float product = static_cast<float>(dot[row]);
                    store.write1(restored + row, first + row, start + token, ((product - sum) / scale) * weight_scale);
                }
            }
        }
    }
    return true;
}
void matmul_attention_qkv_sme(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_ASSERT(userdata && *static_cast<const int *>(userdata) == 16);
    if (matmul_attention_qkv_sme_body(dst, ith, nth))
        return;
    const int64_t m = dst->src[1]->ne[1], n = dst->src[0]->ne[1], dim = m / 16;
    for (int family = 0; family < 3; ++family) {
        ggml_tensor projection = *dst;
        projection.ne[0]       = family == 2 ? n : dim;
        projection.ne[1]       = family == 2 ? dim : n;
        projection.ne[2]       = 16;
        projection.ne[3]       = 1;
        projection.nb[0]       = 4;
        projection.nb[1]       = projection.ne[0] * 4;
        projection.nb[2]       = projection.nb[1] * projection.ne[1];
        projection.nb[3]       = projection.nb[2] * 16;
        projection.data        = static_cast<float *>(dst->data) + int64_t(family) * m * n;
        for (auto & source : projection.src)
            source = nullptr;
        projection.src[0] = dst->src[0];
        projection.src[1] = dst->src[1 + family];
        projection.src[2] = dst->src[4 + family];
        // Use the existing per-projection callback, including its original
        // exceptional scale compatibility fallback; no new serial normal path.
        matmul_attention_layout_sme(&projection, ith, nth, family == 2 ? &value_layout_marker : nullptr);
    }
}
#endif

#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
// Pair only the existing LLM K/V shape; keep full-context inputs. Each branch keeps its
// original row band, token panels, integer leaf and F32 restoration order.
bool matmul_packed_pair_sme_body(ggml_tensor * dst, int ith, int nth) {
#pragma clang fp contract(off)
    const auto * encoded = dst->src[0];
    const int k = 2560, rows = 640;
    const int64_t n = encoded->ne[1];
    if (dst->ne[0] != rows || dst->ne[1] != n || dst->ne[2] != 2 || dst->ne[3] != 1 || n < 32 ||
        n > (INT_MAX / 2) / rows || encoded->ne[0] != k + 8 || !ggml_is_contiguous(encoded))
        return false;
    const int words = integer_sme_words();
    if (!words)
        return false;
    float weight_scales[2];
    for (int branch = 0; branch < 2; ++branch) {
        const auto * weight = dst->src[branch + 1];
        if (weight->type != GGML_TYPE_I2_S || weight->ne[0] != k || weight->ne[1] != rows ||
            !ggml_is_contiguous(weight))
            return false;
        std::memcpy(&weight_scales[branch], static_cast<const char *>(weight->data) + size_t(k) * rows / 4, 4);
        if (!std::isnormal(weight_scales[branch]))
            return false;
    }
    const auto * activations = static_cast<const int8_t *>(encoded->data);
    const size_t stride      = encoded->nb[1];
    // The decision is uniform across all workers, before allocating/writing.
    if (!valid_sme_scales(activations + k, stride, n))
        return false;
    const int pair = 2 * words, row_pairs = (rows + pair - 1) / pair;
    const int first = row_pairs * ith / nth * pair, end = std::min(rows, row_pairs * (ith + 1) / nth * pair);
    if (first >= end)
        return true;
    const int local_rows = end - first, padded_rows = (local_rows + pair - 1) / pair * pair, panel = 2 * pair;
    std::unique_ptr<int8_t[]> packed_w[2];
    for (auto & buffer : packed_w)
        buffer.reset(new int8_t[size_t(padded_rows) * k]);
    std::unique_ptr<int8_t[]> packed_a(new int8_t[size_t(panel) * k]);
    std::unique_ptr<int32_t[]> products(new int32_t[size_t(local_rows) * panel]);
    const auto mask = vdupq_n_u8(3);
    for (int branch = 0; branch < 2; ++branch) {
        const auto * matrix = static_cast<const uint8_t *>(dst->src[branch + 1]->data);
        if (words == 16 && nth == 8)
            pack_kv_groups_c006(packed_w[branch].get(), matrix, first, local_rows);
        else {
            pack_sme_rows<true>(packed_w[branch].get(), local_rows, k, words, [&](int row, int col) {
                const size_t index = size_t(first + row) * k + col;
                const int shift    = 6 - 2 * ((index % 128) / 32);
                const auto bytes   = vld1q_u8(matrix + (index / 128) * 32 + index % 32);
                return vreinterpretq_u32_u8(vandq_u8(vshlq_u8(bytes, vdupq_n_s8(-shift)), mask));
            });
        }
    }
    const size_t plane = size_t(rows) * n;
    auto * output      = static_cast<float *>(dst->data);
    for (int start = 0; start < n; start += panel) {
        const int tokens = std::min(panel, int(n) - start);
        pack_sme_rows<true>(packed_a.get(), tokens, k, words, [&](int row, int col) {
            return vreinterpretq_u32_s8(vld1q_s8(activations + (start + row) * stride + col));
        });
        for (int branch = 0; branch < 2; ++branch) {
            cpu_sme_i8(packed_w[branch].get(), packed_a.get(), products.get(), local_rows, tokens, k / 4, words, 0, 1);
            const auto weights = vdupq_n_f32(weight_scales[branch]);
            for (int token = 0; token < tokens; ++token) {
                const auto * q = activations + (start + token) * stride;
                float scale;
                int32_t sum;
                std::memcpy(&scale, q + k, 4);
                std::memcpy(&sum, q + k + 4, 4);
                const auto * dot = products.get() + size_t(token) * local_rows;
                auto * restored  = output + size_t(branch) * plane + size_t(start + token) * rows + first;
                const auto sums = vdupq_n_f32(float(sum)), scales = vdupq_n_f32(scale);
                int row = 0;
                for (; row + 4 <= local_rows; row += 4) {
                    const auto values = vcvtq_f32_s32(vld1q_s32(dot + row));
                    vst1q_f32(restored + row, vmulq_f32(vdivq_f32(vsubq_f32(values, sums), scales), weights));
                }
                for (; row < local_rows; ++row)
                    restored[row] = ((float(dot[row]) - sum) / scale) * weight_scales[branch];
            }
        }
    }
    return true;
}
void matmul_packed_pair_sme(ggml_tensor * dst, int ith, int nth, void *) {
    if (matmul_packed_pair_sme_body(dst, ith, nth))
        return;
    // Exceptional scales retain the exact original parallel per-ith kernels.
    // No extra synchronization or serial fallback is introduced.
    const int64_t plane = dst->ne[0] * dst->ne[1];
    for (int branch = 0; branch < 2; ++branch) {
        ggml_tensor projection = *dst;
        projection.ne[2] = projection.ne[3] = 1;
        projection.nb[2] = projection.nb[3] = plane * sizeof(float);
        projection.data                     = static_cast<float *>(dst->data) + int64_t(branch) * plane;
        projection.src[1]                   = dst->src[branch + 1];
        projection.src[2]                   = nullptr;
        matmul(&projection, ith, nth, &optimized_route_marker);
    }
}
#endif

void matmul_pair(ggml_tensor * dst, int ith, int nth, void *) {
    if (ith != 0)
        return;
    GGML_UNUSED(nth);
    const int64_t plane = dst->ne[0] * dst->ne[1];
    for (int p = 0; p < 2; ++p) {
        ggml_tensor projection = *dst;
        projection.ne[2] = projection.ne[3] = 1;
        projection.nb[2] = projection.nb[3] = plane * sizeof(float);
        projection.data                     = static_cast<float *>(dst->data) + int64_t(p) * plane;
        projection.src[1]                   = dst->src[p + 1];
        projection.src[2]                   = nullptr;
        matmul(&projection, 0, 1, &optimized_route_marker);
    }
}
void matmul_qkv(ggml_tensor * dst, int ith, int nth, void *) {
    if (ith != 0)
        return;
    GGML_UNUSED(nth);
    const int64_t tokens = dst->src[0]->ne[1];
    int64_t offset       = 0;
    for (int p = 0; p < 3; ++p) {
        const int64_t rows     = p == 0 ? 2560 : 640;
        ggml_tensor projection = *dst;
        projection.ne[0]       = rows;
        projection.ne[1]       = tokens;
        projection.ne[2] = projection.ne[3] = 1;
        projection.nb[0]                    = sizeof(float);
        projection.nb[1]                    = rows * sizeof(float);
        projection.nb[2] = projection.nb[3] = rows * tokens * sizeof(float);
        projection.data                     = static_cast<float *>(dst->data) + offset;
        projection.src[1]                   = dst->src[p + 1];
        projection.src[2] = projection.src[3] = nullptr;
        matmul(&projection, 0, 1, &optimized_route_marker);
        offset += rows * tokens;
    }
}
} // namespace

void i2s_encode_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    encode(dst, ith, nth, userdata);
}

ggml_tensor * i2s_linear_preencoded_attention_qkv(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * query,
                                                  ggml_tensor * key, ggml_tensor * value, ggml_tensor * query_bias,
                                                  ggml_tensor * key_bias, ggml_tensor * value_bias, int heads) {
    ggml_tensor * weights[] = {query, key, value};
    ggml_tensor * biases[]  = {query_bias, key_bias, value_bias};
    for (int family = 0; family < 3; ++family) {
        auto * weight = weights[family];
        auto * bias   = biases[family];

        if (!weight || !encoded || !bias || weight->type != GGML_TYPE_I2_S || encoded->type != GGML_TYPE_I8 ||
            bias->type != GGML_TYPE_F32)
            return nullptr;
        const int64_t k = weight->ne[0], m = weight->ne[1], n = encoded->ne[1];
        if (k <= 0 || k % 128 || k > INT_MAX - 8 || m <= 0 || m > INT_MAX - 127 || n <= 0 || n > INT_MAX - 127 ||
            (m >= 2048 && n >= 512))
            return nullptr;
        if (heads <= 0 || heads > m || m % heads || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            encoded->ne[0] != k + 8 || encoded->ne[2] != 1 || encoded->ne[3] != 1)
            return nullptr;
        if (bias->ne[0] != m || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1)
            return nullptr;
        if (n > INT_MAX / (k + 8) || n > INT_MAX / m || m > INT_MAX / (k / 4))
            return nullptr;
        if (!ggml_is_contiguous(weight) || !ggml_is_contiguous(encoded) || !ggml_is_contiguous(bias))
            return nullptr;
        if (weight->ne[0] != query->ne[0] || weight->ne[1] != query->ne[1])
            return nullptr;
    }
    const int64_t m = query->ne[1], n = encoded->ne[1];
    if (n > 2147483647 / (3 * m) || n > 65535LL * 32)
        return nullptr;
    auto * head_count        = static_cast<int *>(ggml_new_buffer(ctx, sizeof(int)));
    *head_count              = heads;
    ggml_tensor * sources[]  = {encoded, query, key, value, query_bias, key_bias, value_bias};
    ggml_custom_op_t cpu_fun = matmul_attention_qkv;
    int tasks                = 1;
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
    if (query->ne[0] != 1152 || m != 1152 || n != 256 || heads != 16 || integer_sme_words() != 16)
        return nullptr;
    for (int family = 0; family < 3; ++family) {
        if (!weights[family]->data || !biases[family]->data)
            return nullptr;
        float scale;
        std::memcpy(&scale, static_cast<const char *>(weights[family]->data) + size_t(1152) * 1152 / 4, 4);
        if (!std::isnormal(scale))
            return nullptr;
        for (int row = 0; row < 1152; ++row)
            if (!std::isfinite(static_cast<const float *>(biases[family]->data)[row]))
                return nullptr;
    }
    cpu_fun = matmul_attention_qkv_sme;
    tasks   = GGML_N_TASKS_MAX;
#endif
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, 3 * m * n, 1, 1, 1, sources, 7, cpu_fun, tasks, head_count);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, i2s_matmul_attention_qkv_cuda);
#endif
    return result;
}

ggml_tensor * i2s_linear_preencoded_attention_layout(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * encoded,
                                                     ggml_tensor * bias, int heads, bool value_layout) {
    if (!weight || !encoded || !bias || weight->type != GGML_TYPE_I2_S || encoded->type != GGML_TYPE_I8 ||
        bias->type != GGML_TYPE_F32)
        return nullptr;
    const int64_t k = weight->ne[0], m = weight->ne[1], n = encoded->ne[1];
    if (k <= 0 || k % 128 || k > INT_MAX - 8 || m <= 0 || m > INT_MAX - 127 || n <= 0 || n > INT_MAX - 127 ||
        (m >= 2048 && n >= 512))
        return nullptr;
    if (heads <= 0 || heads > m || m % heads || weight->ne[2] != 1 || weight->ne[3] != 1 || encoded->ne[0] != k + 8 ||
        encoded->ne[2] != 1 || encoded->ne[3] != 1)
        return nullptr;
    if (bias->ne[0] != m || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1)
        return nullptr;
    if (n > INT_MAX / (k + 8) || n > INT_MAX / m || m > INT_MAX / (k / 4))
        return nullptr;
    if (!ggml_is_contiguous(weight) || !ggml_is_contiguous(encoded) || !ggml_is_contiguous(bias))
        return nullptr;
    ggml_tensor * sources[]  = {encoded, weight, bias};
    const int64_t dim        = m / heads;
    ggml_custom_op_t cpu_fun = matmul_attention_layout;
    int tasks                = 1;
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
    // This standalone CPU port currently supports only the frozen ViT shape.
    // Other shapes keep their existing separate projection/bias/layout graph.
    const int words = integer_sme_words();
    if (k != 1152 || m != 1152 || n != 256 || heads != 16 || !weight->data || !bias->data || words <= 0 || words % 2)
        return nullptr;
    float scale;
    std::memcpy(&scale, static_cast<const char *>(weight->data) + size_t(k) * m / 4, sizeof(scale));
    if (!std::isnormal(scale))
        return nullptr;
    const auto * values = static_cast<const float *>(bias->data);
    for (int64_t row = 0; row < m; ++row)
        if (!std::isfinite(values[row]))
            return nullptr;
    cpu_fun = matmul_attention_layout_sme;
    tasks   = GGML_N_TASKS_MAX;
#endif
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, value_layout ? n : dim, value_layout ? dim : n, heads, 1,
                                   sources, 3, cpu_fun, tasks, value_layout ? &value_layout_marker : nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, value_layout ? i2s_matmul_value_layout_cuda : i2s_matmul_query_key_layout_cuda);
#endif
    return result;
}

ggml_tensor * i2s_linear_preencoded_packed_pair(ggml_context * ctx, ggml_tensor * weight0, ggml_tensor * weight1,
                                                ggml_tensor * encoded) {
    if (!weight0 || !weight1 || !encoded || weight0->type != GGML_TYPE_I2_S || weight1->type != GGML_TYPE_I2_S ||
        encoded->type != GGML_TYPE_I8)
        return nullptr;
    const int64_t k = weight0->ne[0], m = weight0->ne[1], n = encoded->ne[1];
    if (k <= 0 || k % 128 || k > INT_MAX - 8 || m <= 0 || m > INT_MAX - 127 || n <= 0 || n > 65535 * 32 ||
        (m >= 2048 && n >= 512))
        return nullptr;
    if (n > INT_MAX / (k + 8) || n > (INT_MAX / 2) / m || m > INT_MAX / (k / 4))
        return nullptr;
    if (weight0->ne[2] != 1 || weight0->ne[3] != 1 || weight1->ne[0] != k || weight1->ne[1] != m ||
        weight1->ne[2] != 1 || weight1->ne[3] != 1)
        return nullptr;
    if (encoded->ne[0] != k + 8 || encoded->ne[2] != 1 || encoded->ne[3] != 1)
        return nullptr;
    if (!ggml_is_contiguous(weight0) || !ggml_is_contiguous(weight1) || !ggml_is_contiguous(encoded))
        return nullptr;
    ggml_tensor * sources[]  = {encoded, weight0, weight1};
    ggml_custom_op_t cpu_fun = matmul_pair;
    int tasks                = 1;
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
    if (k != 2560 || m != 640 || n < 32 || !weight0->data || !weight1->data || !integer_sme_words())
        return nullptr;
    for (const auto * weight : {weight0, weight1}) {
        float scale;
        std::memcpy(&scale, static_cast<const char *>(weight->data) + size_t(k) * m / 4, 4);
        if (!std::isnormal(scale))
            return nullptr;
    }
    cpu_fun = matmul_packed_pair_sme;
    tasks   = GGML_N_TASKS_MAX;
#endif
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, m, n, 2, 1, sources, 3, cpu_fun, tasks, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, i2s_matmul_packed_pair_cuda);
#endif
    return result;
}

ggml_tensor * i2s_linear_preencoded_qkv(ggml_context * ctx, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value,
                                        ggml_tensor * encoded) {
    if (!query || !key || !value || !encoded || encoded->type != GGML_TYPE_I8)
        return nullptr;
    const int64_t n = encoded->ne[1];
    if (n < 512 || n > INT_MAX / 3840 || encoded->ne[0] != 2568 || encoded->ne[2] != 1 || encoded->ne[3] != 1 ||
        !ggml_is_contiguous(encoded))
        return nullptr;
    ggml_tensor * weights[] = {query, key, value};
    for (int p = 0; p < 3; ++p) {
        auto * w           = weights[p];
        const int64_t rows = p == 0 ? 2560 : 640;
        if (w->type != GGML_TYPE_I2_S || w->ne[0] != 2560 || w->ne[1] != rows || w->ne[2] != 1 || w->ne[3] != 1 ||
            !ggml_is_contiguous(w))
            return nullptr;
    }
    ggml_tensor * sources[] = {encoded, query, key, value};
    auto * result           = ggml_custom_4d(ctx, GGML_TYPE_F32, 3840 * n, 1, 1, 1, sources, 4, matmul_qkv, 1, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, i2s_matmul_qkv_cuda);
#endif
    return result;
}

ggml_tensor * i2s_linear_preencoded_pair(ggml_context * ctx, ggml_tensor * weight0, ggml_tensor * weight1,
                                         ggml_tensor * encoded) {
    if (!weight0 || !weight1 || !encoded || weight0->type != GGML_TYPE_I2_S || weight1->type != GGML_TYPE_I2_S ||
        encoded->type != GGML_TYPE_I8)
        return nullptr;
    const int64_t k = weight0->ne[0], m = weight0->ne[1], n = encoded->ne[1];
    if (k <= 0 || k % 128 || k > INT_MAX - 8 || m < 2048 || m > INT_MAX - 127 || n < 512 || n > INT_MAX - 31)
        return nullptr;
    if (n > INT_MAX / (k + 8) || n > (INT_MAX / 2) / m || m > INT_MAX / (k / 4))
        return nullptr;
    if (weight0->ne[2] != 1 || weight0->ne[3] != 1 || weight1->ne[0] != k || weight1->ne[1] != m ||
        weight1->ne[2] != 1 || weight1->ne[3] != 1)
        return nullptr;
    if (encoded->ne[0] != k + 8 || encoded->ne[2] != 1 || encoded->ne[3] != 1)
        return nullptr;
    if (!ggml_is_contiguous(weight0) || !ggml_is_contiguous(weight1) || !ggml_is_contiguous(encoded))
        return nullptr;
    ggml_tensor * sources[] = {encoded, weight0, weight1};
    auto * result           = ggml_custom_4d(ctx, GGML_TYPE_F32, m, n, 2, 1, sources, 3, matmul_pair, 1, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, i2s_matmul_pair_cuda);
#endif
    return result;
}

ggml_tensor * i2s_linear_preencoded(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * encoded,
                                    bool optimized_route) {
    GGML_ASSERT(weight && weight->type == GGML_TYPE_I2_S && weight->ne[0] > 0 && weight->ne[0] <= INT32_MAX - 127);
    const int k = (weight->ne[0] + 127) / 128 * 128;
    GGML_ASSERT(encoded && encoded->type == GGML_TYPE_I8 && ggml_is_contiguous(encoded));
    GGML_ASSERT(encoded->ne[0] == k + 8 && encoded->ne[1] > 0 && encoded->ne[2] == 1 && encoded->ne[3] == 1);
    ggml_tensor * sources[] = {encoded, weight};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, weight->ne[1], encoded->ne[1], 1, 1, sources, 2, matmul,
                                   GGML_N_TASKS_MAX, optimized_route ? &optimized_route_marker : nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, i2s_matmul_cuda);
#endif
    return result;
}

ggml_tensor * i2s_encode(ggml_context * ctx, ggml_tensor * input, int padded_k) {
    ggml_tensor * sources[] = {input};
    auto * encoded          = ggml_custom_4d(ctx, GGML_TYPE_I8, padded_k + 8, input->ne[1], 1, 1, sources, 1, encode,
                                             GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(encoded, i2s_encode_cuda);
#endif
    return encoded;
}

ggml_tensor * i2s_linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input, bool optimized_route) {
    const int k = weight->type == GGML_TYPE_I2_S ? (weight->ne[0] + 127) / 128 * 128 : (weight->ne[0] - 4) * 4;
    return i2s_linear_encoded(ctx, weight, input, i2s_encode(ctx, input, k), optimized_route);
}

ggml_tensor * i2s_linear_encoded(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input, ggml_tensor * encoded,
                                 bool optimized_route) {
    const int k = weight->type == GGML_TYPE_I2_S ? (weight->ne[0] + 127) / 128 * 128 : (weight->ne[0] - 4) * 4;
    GGML_ASSERT(encoded && encoded->type == GGML_TYPE_I8 && encoded->src[0] == input);
    GGML_ASSERT(encoded->ne[0] == k + 8 && encoded->ne[1] == input->ne[1]);
    ggml_tensor * sources[] = {encoded, weight};
    auto * result           = ggml_custom_4d(ctx, GGML_TYPE_F32, weight->ne[1], input->ne[1], 1, 1, sources, 2, matmul,
                                             GGML_N_TASKS_MAX, optimized_route ? &optimized_route_marker : nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, i2s_matmul_cuda);
#endif
    return result;
}
} // namespace robotcpp::bitvla
