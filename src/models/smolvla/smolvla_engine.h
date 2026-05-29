#ifndef SMOLVLA_ENGINE_H
#define SMOLVLA_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define SMOLVLA_API __declspec(dllexport)
#        else
#            define SMOLVLA_API __declspec(dllimport)
#        endif
#    else
#        define SMOLVLA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define SMOLVLA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct smolvla_context;

enum smolvla_noise_mode {
    SMOLVLA_NOISE_MODE_GAUSSIAN = 0,
    SMOLVLA_NOISE_MODE_DEBUG_SIN = 1,
};

// ============================================================================
// Initialization parameters
// ============================================================================
struct smolvla_params {
    const char * vlm_path;            // VLM GGUF path (smolvla-vlm-f16.gguf)
    const char * mmproj_path;         // Vision GGUF path (mmproj-smolvla-f16.gguf, includes connector weight)
    const char * state_proj_path;     // State projector GGUF path (state-proj-smolvla-f16.gguf)
    const char * action_expert_path;  // Action expert GGUF path (action-expert-smolvla-f16.gguf)
    const char * task;                // Language instruction (e.g., "grab the block.")
    int n_threads;                    // Number of CPU threads (0 = auto)
    int action_dim;                   // Action dimension (default 6)
    int chunk_size;                   // Action chunk size (default 50)
    int num_steps;                    // Denoising steps (default 10)
    int n_batch;                      // Batch size for token eval (default 512)
    int n_ctx;                        // Context size (default 2048)
    int noise_mode;                   // 0=gaussian (default), 1=debug_sin
    int64_t noise_seed;               // <0 = nondeterministic, >=0 = reproducible RNG seed
    int verbosity;                    // 0=quiet, 1=info, 2=debug
};

// ============================================================================
// Prediction result
// ============================================================================
struct smolvla_result {
    float * actions;                  // Pointer to action array [chunk_size * action_dim]
    int chunk_size;                   // Number of action steps
    int action_dim;                   // Action dimension per step
};

struct smolvla_stage_timings {
    double vision_ms;                 // Image encode + vision model forward
    double state_proj_ms;             // State normalization + state projection
    double vlm_ms;                    // Prefix/token prep + VLM forward
    double kv_extract_ms;             // Extract VLM KV cache for Phase 2
    double phase2_ms;                 // Full denoising loop + final unnormalization
    double total_ms;                  // End-to-end predict(...) time inside C++
};

// ============================================================================
// API functions
// ============================================================================

/**
 * Get default parameters.
 */
SMOLVLA_API struct smolvla_params smolvla_default_params(void);

/**
 * Initialize the SmolVLA engine.
 * Loads all GGUF models. This is the expensive operation — call once.
 *
 * @param params Initialization parameters
 * @return Engine context, or NULL on failure
 */
SMOLVLA_API struct smolvla_context * smolvla_init(struct smolvla_params params);

/**
 * Run one prediction: image + state + task → actions.
 *
 * SmolVLA uses Flow Matching with denoising iterations:
 * 1. Encode image + state + task → VLM hidden states (KV cache)
 * 2. Denoise random noise → actions (10 steps by default)
 *
 * @param ctx        Engine context (from smolvla_init)
 * @param image_path Path to image file (JPEG/PNG, will be resized to 512x512)
 * @param state      Proprio/state array [state_dim], or NULL if no state
 * @param state_dim  Number of state dimensions (typically 6)
 * @return Prediction result with action array [chunk_size * action_dim].
 *         The result is valid until the next call to smolvla_predict() or smolvla_free().
 *         Returns result with actions=NULL on failure.
 */
SMOLVLA_API struct smolvla_result smolvla_predict(
    struct smolvla_context * ctx,
    const char * image_path,
    const float * state,
    int state_dim);

/**
 * Run one prediction with raw image bytes (for pybind11/C API use).
 *
 * @param ctx         Engine context
 * @param image_bytes Raw image file bytes (JPEG/PNG encoded)
 * @param image_len   Length of image_bytes
 * @param state       State array, or NULL
 * @param state_dim   Number of state dimensions
 * @return Prediction result
 */
SMOLVLA_API struct smolvla_result smolvla_predict_bytes(
    struct smolvla_context * ctx,
    const unsigned char * image_bytes,
    int image_len,
    const float * state,
    int state_dim);

/**
 * Return the stage timings captured by the most recent successful or failed
 * predict()/predict_bytes() call on this context.
 */
SMOLVLA_API struct smolvla_stage_timings smolvla_get_last_stage_timings(
    const struct smolvla_context * ctx);

/**
 * Free the engine context and all associated resources.
 */
SMOLVLA_API void smolvla_free(struct smolvla_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // SMOLVLA_ENGINE_H
