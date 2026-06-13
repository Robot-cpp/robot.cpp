/**
 * SmolVLA State Projector (proprio.cpp style with ggml graph building)
 * 
 * Architecture: normalize (mean_std) → Linear(32, 960)
 * Uses ggml compute graph for efficient inference.
 */

#ifndef SMOLVLA_STATE_PROJ_H
#define SMOLVLA_STATE_PROJ_H

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "models/ggml_backend.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define SMOLVLA_STATE_PROJ_API __declspec(dllexport)
#        else
#            define SMOLVLA_STATE_PROJ_API __declspec(dllimport)
#        endif
#    else
#        define SMOLVLA_STATE_PROJ_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define SMOLVLA_STATE_PROJ_API
#endif

// ============================================================================
// SmolVLA State Projector Context
// ============================================================================
struct smolvla_state_proj {
    // GGUF/GGML contexts
    struct gguf_context * ctx_gguf = nullptr;
    struct ggml_context * ctx_data = nullptr;

    // Backend
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<ggml_backend_t> backends;
    backend_buft_policy buft_policy;
    ggml_backend_buffer_t params_buffer = nullptr;

    // Model parameters
    int state_dim = 6;         // Actual input dimension (from dataset)
    int max_state_dim = 32;    // Padded input dimension
    int hidden_size = 960;     // Output dimension

    // Normalization statistics (MEAN_STD mode)
    std::vector<float> norm_mean;  // [state_dim]
    std::vector<float> norm_std;   // [state_dim]
    bool has_norm_stats = false;

    // Weight tensors (single Linear layer)
    // weight: [hidden_size, max_state_dim] = [960, 32]
    // bias: [hidden_size] = [960]
    struct ggml_tensor * weight = nullptr;
    struct ggml_tensor * bias = nullptr;

    // Compute buffer
    std::vector<uint8_t> buf_compute_meta;
};


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load state projector from GGUF file
 * @param fname Path to state-proj GGUF file
 * @param verbosity Log verbosity level (0=quiet, 1=info, 2=debug)
 * @return State projector context or NULL on failure
 */
SMOLVLA_STATE_PROJ_API struct smolvla_state_proj * smolvla_state_proj_load(const char * fname, int verbosity);

/**
 * Encode state through the projector: normalize → Linear
 * @param ctx State projector context
 * @param n_threads Number of CPU threads
 * @param state Input state array [state_dim] (raw, unnormalized)
 * @param state_dim Actual state dimension (e.g., 6)
 * @param output Output embedding array [hidden_size]
 * @return true on success
 */
SMOLVLA_STATE_PROJ_API bool smolvla_state_proj_encode(
    struct smolvla_state_proj * ctx,
    int n_threads,
    const float * state,
    int state_dim,
    float * output
);

/**
 * Free state projector context
 */
SMOLVLA_STATE_PROJ_API void smolvla_state_proj_free(struct smolvla_state_proj * ctx);

/**
 * Get hidden size (output dimension)
 */
SMOLVLA_STATE_PROJ_API int smolvla_state_proj_hidden_size(const struct smolvla_state_proj * ctx);

/**
 * Get max state dim (input dimension)
 */
SMOLVLA_STATE_PROJ_API int smolvla_state_proj_max_state_dim(const struct smolvla_state_proj * ctx);

#ifdef __cplusplus
}
#endif

#endif // SMOLVLA_STATE_PROJ_H
