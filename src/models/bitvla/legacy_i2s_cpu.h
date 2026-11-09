#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Original CPU functions: BitVLA.cpp 2756ed8 + llama.cpp 5a7b0161.
// Only namespace/dependencies and selection of the original AVX2 branch are adapted.
namespace robotcpp::bitvla::legacy_cpu {
#define MAX(a, b) ((a) > (b) ? (a) : (b))
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_row_i8_s(const float * x, void * y, int64_t n, float * act_scales, int32_t * act_sums) {
    // void quantize_row_i8_s(const float * x, void * y, int64_t n, float* act_scales) {
    int8_t * dst = (int8_t *)y;
    double min   = 0.00001;
    double max   = min;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, (double)fabs((double)x[i]));
    }
    float s       = 127 / max;
    act_scales[0] = s;
    int32_t sum   = 0;
    for (int i = 0; i < n; ++i) {
        int v = nearest_int(x[i] * s);
        if (v > 127)
            v = 127;
        if (v < -128)
            v = -128;
        sum += v;
        dst[i] = (int8_t)(v);
    }
    act_sums[0] = sum;
}
#undef MAX

#ifdef __AVX2__
constexpr int QK_I2_S = 128;
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64  = _mm_add_epi32(hi64, sum128);
    const __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by,
                          int nrc) {
    const uint8_t * x = (uint8_t *)vx;
    const int8_t * y  = (int8_t *)vy;

    const int nb          = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num      = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i = 0; i < group32_num; i++) {
        __m256i accu32 = _mm256_setzero_si256();
        for (int j = 0; j < 32; j++) {
            // 128 index
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i *)(x + i * 32 * 32 + j * 32));
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

            // each 32 index
            xq8_3 = _mm256_and_si256(xq8_3, mask);
            xq8_2 = _mm256_and_si256(xq8_2, mask);
            xq8_1 = _mm256_and_si256(xq8_1, mask);
            xq8_0 = _mm256_and_si256(xq8_0, mask);

            // each 32 index
            __m256i yq8_0 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 96));

            // 128 index accumulation add
            // split into 32 accumulation block
            // each block each 128 index accumulated 4index
            // each index maximum 256
            // each block maximum 4 * 256
            // each block accumulation maximum 127 * 256
            // each 32 group index (128 index in one group) needs cast to int32
            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
            accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++) {
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
            // 128 index
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i *)(x + group32_num * 32 * 32 + j * 32));
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

            // each 32 index
            xq8_3 = _mm256_and_si256(xq8_3, mask);
            xq8_2 = _mm256_and_si256(xq8_2, mask);
            xq8_1 = _mm256_and_si256(xq8_1, mask);
            xq8_0 = _mm256_and_si256(xq8_0, mask);

            // each 32 index
            __m256i yq8_0 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 96));

            // 128 index accumulation add
            // split into 32 accumulation block
            // each block each 128 index accumulated 4index
            // each index maximum 256
            // each block maximum 4 * 256
            // each block accumulation maximum 127 * 256
            // each 32 group index (128 index in one group) needs cast to int32
            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s       = (float)sumi;
}
#endif
} // namespace robotcpp::bitvla::legacy_cpu
