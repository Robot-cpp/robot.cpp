#ifndef PI0_ENGINE_H
#define PI0_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define PI0_API __declspec(dllexport)
#        else
#            define PI0_API __declspec(dllimport)
#        endif
#    else
#        define PI0_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define PI0_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct pi0_context;

struct pi0_params {
    const char * vit_path;
    const char * mmproj_path;
    const char * llm_path;
    const char * tokenizer_path;
    const char * state_path;
    const char * action_decoder_path;
    int n_threads;
    int64_t noise_seed;
    int verbosity;
};

struct pi0_image_view {
    const char * name;
    const uint8_t * data;
    int width;
    int height;
    int channels;
    int stride_bytes;
};

struct pi0_result {
    float * actions;
    int chunk_size;
    int action_dim;
};

struct pi0_stage_timings {
    double preprocess_ms;
    double prefix_ms;
    double state_ms;
    double denoise_ms;
    double output_ms;
    double total_ms;
};

PI0_API struct pi0_params pi0_default_params(void);
PI0_API struct pi0_context * pi0_init(struct pi0_params params);
PI0_API struct pi0_result pi0_predict_raw_rgb(
    struct pi0_context * ctx,
    const struct pi0_image_view * images,
    size_t image_count,
    const float * state,
    size_t state_dim,
    const char * task);
PI0_API struct pi0_stage_timings pi0_get_last_stage_timings(const struct pi0_context * ctx);
PI0_API void pi0_reset(struct pi0_context * ctx);
PI0_API void pi0_free(struct pi0_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // PI0_ENGINE_H
