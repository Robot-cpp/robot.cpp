#include <arm_sme.h>
#include <cstddef>
#include <cstdint>
static uint64_t words_streaming() __arm_streaming {
    return svcntsw();
}
extern "C" uint64_t cpu_sme_words() {
    return words_streaming();
}

// Four I32 output tiles. Each SMOPA accumulates four signed-byte products
// per output element. The matrix codes remain0..3; no new quantization.
extern "C" __arm_new("za") void cpu_sme_i8_streaming(const int8_t * w, const int8_t * a, int32_t * c, int M, int N,
                                                     int Kgroups, int words, int ith, int nth) __arm_streaming {
    if (svcntsw() != static_cast<uint64_t>(words))
        __builtin_trap();
    const int bytes = words * 4, mpairs = (M + 2 * words - 1) / (2 * words), npairs = (N + 2 * words - 1) / (2 * words);
    const svbool_t all = svptrue_b8();
    for (int task = ith; task < mpairs * npairs; task += nth) {
        const int mb = 2 * (task / npairs), nb = 2 * (task % npairs);
        svzero_za();
        for (int k = 0; k < Kgroups; ++k) {
            const svint8_t w0 = svld1_s8(all, w + (size_t(mb) * Kgroups + k) * bytes);
            const svint8_t w1 = svld1_s8(all, w + (size_t(mb + 1) * Kgroups + k) * bytes);
            const svint8_t a0 = svld1_s8(all, a + (size_t(nb) * Kgroups + k) * bytes);
            const svint8_t a1 = svld1_s8(all, a + (size_t(nb + 1) * Kgroups + k) * bytes);
            svmopa_za32_s8_m(0, all, all, w0, a0);
            svmopa_za32_s8_m(1, all, all, w1, a0);
            svmopa_za32_s8_m(2, all, all, w0, a1);
            svmopa_za32_s8_m(3, all, all, w1, a1);
        }
        const int m0 = mb * words, m1 = m0 + words, n0 = nb * words, n1 = n0 + words;
        const svbool_t p0 = svwhilelt_b32(uint64_t(m0), uint64_t(M));
        const svbool_t p1 = svwhilelt_b32(uint64_t(m1), uint64_t(M));
        for (int col = 0; col < words; ++col) {
            if (n0 + col < N) {
                svst1_ver_za32(0, col, p0, c + size_t(n0 + col) * M + m0);
                if (m1 < M)
                    svst1_ver_za32(1, col, p1, c + size_t(n0 + col) * M + m1);
            }
            if (n1 + col < N) {
                svst1_ver_za32(2, col, p0, c + size_t(n1 + col) * M + m0);
                if (m1 < M)
                    svst1_ver_za32(3, col, p1, c + size_t(n1 + col) * M + m1);
            }
        }
    }
}

__arm_new("za") __attribute__((visibility("hidden"),
                               noinline)) void cpu_sme_i8_lookahead085(const int8_t * w, const int8_t * a, int32_t * c,
                                                                       int M, int N, int Kgroups, int words, int ith,
                                                                       int nth) __arm_streaming {
    if (svcntsw() != static_cast<uint64_t>(words))
        __builtin_trap();
    const int bytes = words * 4, mpairs = (M + 2 * words - 1) / (2 * words), npairs = (N + 2 * words - 1) / (2 * words);
    const svbool_t all = svptrue_b8();
    for (int task = ith; task < mpairs * npairs; task += nth) {
        const int mb = 2 * (task / npairs), nb = 2 * (task % npairs);
        svzero_za();
        // One K-group lookahead; no read beyond the original final group.
        svint8_t next_w0 = svld1_s8(all, w + size_t(mb) * Kgroups * bytes);
        svint8_t next_w1 = svld1_s8(all, w + size_t(mb + 1) * Kgroups * bytes);
        svint8_t next_a0 = svld1_s8(all, a + size_t(nb) * Kgroups * bytes);
        svint8_t next_a1 = svld1_s8(all, a + size_t(nb + 1) * Kgroups * bytes);
        for (int k = 0; k < Kgroups; ++k) {
            const svint8_t w0 = next_w0, w1 = next_w1, a0 = next_a0, a1 = next_a1;
            if (k + 1 < Kgroups) {
                next_w0 = svld1_s8(all, w + (size_t(mb) * Kgroups + k + 1) * bytes);
                next_w1 = svld1_s8(all, w + (size_t(mb + 1) * Kgroups + k + 1) * bytes);
                next_a0 = svld1_s8(all, a + (size_t(nb) * Kgroups + k + 1) * bytes);
                next_a1 = svld1_s8(all, a + (size_t(nb + 1) * Kgroups + k + 1) * bytes);
            }
            svmopa_za32_s8_m(0, all, all, w0, a0);
            svmopa_za32_s8_m(1, all, all, w1, a0);
            svmopa_za32_s8_m(2, all, all, w0, a1);
            svmopa_za32_s8_m(3, all, all, w1, a1);
        }
        const int m0 = mb * words, m1 = m0 + words, n0 = nb * words, n1 = n0 + words;
        const svbool_t p0 = svwhilelt_b32(uint64_t(m0), uint64_t(M));
        const svbool_t p1 = svwhilelt_b32(uint64_t(m1), uint64_t(M));
        for (int col = 0; col < words; ++col) {
            if (n0 + col < N) {
                svst1_ver_za32(0, col, p0, c + size_t(n0 + col) * M + m0);
                if (m1 < M)
                    svst1_ver_za32(1, col, p1, c + size_t(n0 + col) * M + m1);
            }
            if (n1 + col < N) {
                svst1_ver_za32(2, col, p0, c + size_t(n1 + col) * M + m0);
                if (m1 < M)
                    svst1_ver_za32(3, col, p1, c + size_t(n1 + col) * M + m1);
            }
        }
    }
}

// The public probe boundary is non-streaming. The SME-enabled translation
// unit performs the ABI mode switch, so the baseline CPU caller needs noSME.
extern "C" void cpu_sme_i8(const int8_t * w, const int8_t * a, int32_t * c, int M, int N, int Kgroups, int words,
                           int ith, int nth) {
    // CUDA085 cross-row method: the unchanged caller has already decoded the
    // Vision4304/padded4352 W band. Overlap only next decodedK-group loads.
    // Keep rejected044 LLM routes absent and all other caller shapes original.
    if (words == 16 && Kgroups == 1088 && (M == 128 || M == 160) && N == 64 && ith == 0 && nth == 1)
        cpu_sme_i8_lookahead085(w, a, c, M, N, Kgroups, words, ith, nth);
    else
        cpu_sme_i8_streaming(w, a, c, M, N, Kgroups, words, ith, nth);
}
