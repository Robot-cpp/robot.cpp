#ifndef VLACPP_H
#define VLACPP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(VLACPP_BUILD_SHARED)
#    define VLACPP_API __declspec(dllexport)
#  else
#    define VLACPP_API
#  endif
#else
#  define VLACPP_API __attribute__((visibility("default")))
#endif

typedef struct vlacpp_model vlacpp_model;
typedef struct vlacpp_context vlacpp_context;

typedef enum vlacpp_status {
    VLACPP_STATUS_OK = 0,
    VLACPP_STATUS_INVALID_ARGUMENT = 1,
    VLACPP_STATUS_IO_ERROR = 2,
    VLACPP_STATUS_PARSE_ERROR = 3,
    VLACPP_STATUS_UNSUPPORTED = 4,
    VLACPP_STATUS_RUNTIME_ERROR = 5,
} vlacpp_status;

typedef enum vlacpp_backend {
    VLACPP_BACKEND_CPU = 0,
    VLACPP_BACKEND_CUDA = 1,
} vlacpp_backend;

typedef struct vlacpp_model_params {
    vlacpp_backend backend;
    int32_t n_threads;
} vlacpp_model_params;

typedef struct vlacpp_context_params {
    uint32_t seed;
    int32_t flow_steps;
} vlacpp_context_params;

typedef struct vlacpp_image_view {
    const char * name;
    const uint8_t * data;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t stride_bytes;
} vlacpp_image_view;

typedef struct vlacpp_observation {
    const vlacpp_image_view * images;
    size_t image_count;
    const float * state;
    size_t state_count;
    const char * prompt;
    const int32_t * prompt_tokens;
    size_t prompt_token_count;
    const float * noise;
    size_t noise_count;
} vlacpp_observation;

typedef struct vlacpp_action_chunk {
    float * data;
    int32_t horizon;
    int32_t action_dim;
} vlacpp_action_chunk;

typedef struct vlacpp_openpi_graph_info {
    int32_t action_width;
    int32_t vision_width;
    int32_t vision_patch_height;
    int32_t vision_patch_width;
    int32_t vision_layers;
    int32_t language_width;
    int32_t language_q_out;
    int32_t language_kv_out;
    int32_t language_mlp_width;
    int32_t language_layers;
    int32_t action_expert_width;
    int32_t action_expert_q_out;
    int32_t action_expert_kv_out;
    int32_t action_expert_mlp_width;
    int32_t action_expert_layers;
    int32_t full_weights_present;
} vlacpp_openpi_graph_info;

VLACPP_API vlacpp_model_params vlacpp_default_model_params(void);
VLACPP_API vlacpp_context_params vlacpp_default_context_params(void);

VLACPP_API vlacpp_status vlacpp_load_model(
    const char * path,
    const vlacpp_model_params * params,
    vlacpp_model ** out_model);

VLACPP_API void vlacpp_free_model(vlacpp_model * model);
VLACPP_API const char * vlacpp_model_capability(vlacpp_model * model);
VLACPP_API vlacpp_status vlacpp_model_openpi_graph_info(
    vlacpp_model * model,
    vlacpp_openpi_graph_info * out_info);

VLACPP_API vlacpp_status vlacpp_create_context(
    vlacpp_model * model,
    const vlacpp_context_params * params,
    vlacpp_context ** out_context);

VLACPP_API void vlacpp_free_context(vlacpp_context * context);
VLACPP_API vlacpp_status vlacpp_reset_cache(vlacpp_context * context);

VLACPP_API vlacpp_status vlacpp_infer_actions(
    vlacpp_context * context,
    const vlacpp_observation * observation,
    vlacpp_action_chunk * out_actions);

VLACPP_API void vlacpp_free_action_chunk(vlacpp_action_chunk * actions);
VLACPP_API const char * vlacpp_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
