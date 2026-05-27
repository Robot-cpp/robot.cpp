/**
 * SmolVLA Action Expert — 16-layer transformer for flow-matching denoising
 *
 * Architecture per denoising step:
 *   embed_suffix(noisy_actions, timestep) → [1, chunk_size, expert_hidden]
 *   16 layers alternating self-attn (even) / cross-attn (odd)
 *   final_norm → action_out_proj → velocity [1, chunk_size, max_action_dim]
 *
 * Uses ggml compute graphs (same pattern as state_proj.cpp).
 */

#ifndef SMOLVLA_ACTION_EXPERT_H
#define SMOLVLA_ACTION_EXPERT_H

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <string>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define SMOLVLA_EXPERT_API __declspec(dllexport)
#        else
#            define SMOLVLA_EXPERT_API __declspec(dllimport)
#        endif
#    else
#        define SMOLVLA_EXPERT_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define SMOLVLA_EXPERT_API
#endif

// ============================================================================
// Per-layer tensors
// ============================================================================
struct smolvla_expert_layer {
    // Attention norm (RMSNorm)
    struct ggml_tensor * attn_norm;       // [hidden_size]

    // Attention projections
    struct ggml_tensor * attn_q;          // [hidden_size, n_q_heads * head_dim]
    struct ggml_tensor * attn_k;          // [hidden_size, n_kv_heads * head_dim] (self) or [vlm_kv_dim, n_kv_heads * head_dim] (cross)
    struct ggml_tensor * attn_v;          // same dimensions as attn_k
    struct ggml_tensor * attn_output;     // [n_q_heads * head_dim, hidden_size]

    // FFN norm (RMSNorm)
    struct ggml_tensor * ffn_norm;        // [hidden_size]

    // FFN (SwiGLU)
    struct ggml_tensor * ffn_gate;        // [hidden_size, intermediate_size]
    struct ggml_tensor * ffn_up;          // [hidden_size, intermediate_size]
    struct ggml_tensor * ffn_down;        // [intermediate_size, hidden_size]

    bool is_cross_attn;                   // true for odd layers
};

// ============================================================================
// Action Expert Context
// ============================================================================
struct smolvla_action_expert {
    // GGUF/GGML contexts
    struct gguf_context * ctx_gguf = nullptr;
    struct ggml_context * ctx_data = nullptr;

    // Backend
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<ggml_backend_t> backends;
    struct buft_policy_t {
        ggml_backend_buffer_type_t model_buft = nullptr;
        ggml_backend_buffer_type_t runtime_buft = nullptr;
        ggml_backend_buffer_type_t host_buft = nullptr;
    } buft_policy;
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    // ----- Configuration (from GGUF metadata) -----
    int hidden_size       = 720;   // expert hidden dim
    int intermediate_size = 2048;  // FFN intermediate
    int num_layers        = 16;
    int max_action_dim    = 32;    // padded action dim
    int chunk_size        = 50;
    int num_steps         = 10;    // denoising steps
    int self_attn_every_n = 2;     // even layers = self-attn
    float min_period      = 0.004f;
    float max_period      = 4.0f;

    // Derived head info (from weight shapes)
    int head_dim   = 64;
    int n_q_heads  = 0;   // hidden_q_dim / head_dim  (usually 15)
    int n_kv_heads = 0;   // kv_dim / head_dim        (usually 5)
    int vlm_kv_dim = 0;   // VLM n_kv_heads * head_dim for cross-attn input (usually 320)

    // ----- Model tensors -----
    std::vector<smolvla_expert_layer> layers;

    // Action projections
    struct ggml_tensor * action_in_proj_w  = nullptr;  // [max_action_dim, hidden_size]
    struct ggml_tensor * action_in_proj_b  = nullptr;  // [hidden_size]
    struct ggml_tensor * action_out_proj_w = nullptr;  // [hidden_size, max_action_dim]
    struct ggml_tensor * action_out_proj_b = nullptr;  // [max_action_dim]

    // Time MLP: Linear(hidden*2, hidden) → SiLU → Linear(hidden, hidden)
    struct ggml_tensor * time_mlp_in_w  = nullptr;     // [hidden_size*2, hidden_size]
    struct ggml_tensor * time_mlp_in_b  = nullptr;     // [hidden_size]
    struct ggml_tensor * time_mlp_out_w = nullptr;     // [hidden_size, hidden_size]
    struct ggml_tensor * time_mlp_out_b = nullptr;     // [hidden_size]

    // Final RMSNorm
    struct ggml_tensor * final_norm = nullptr;         // [hidden_size]

    // ----- Unnormalization stats -----
    std::vector<float> action_mean;  // [action_dim]
    std::vector<float> action_std;   // [action_dim]
    int action_dim = 0;              // actual action dim (from stats length, e.g. 6)

    // Precomputed sinusoidal embeddings for the fixed denoising schedule.
    std::vector<float> precomputed_timesteps;   // [num_steps]
    std::vector<float> precomputed_time_emb_2d; // [num_steps, hidden_size, chunk_size]

    // ----- Compute resources -----
    std::vector<uint8_t> buf_compute_meta;
    int transformer_runtime_prefix_seq_len = -1;
    int graph_n_threads = 0;
    int verbosity = 0;

    struct prefix_kv_runtime_t {
        int prefix_seq_len = -1;
        struct ggml_context * ctx_data = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<struct ggml_tensor *> k_layers;
        std::vector<struct ggml_tensor *> v_layers;
    } prefix_kv_runtime;

    // Cached Phase 2 attention metadata for repeated denoising steps.
    int cached_prefix_seq_len = -1;
    std::vector<uint8_t> cached_prefix_valid_mask;
    std::vector<int32_t> cached_self_position_ids;
    std::vector<int32_t> cached_cross_position_ids;
    std::vector<float> cached_self_attention_mask_f32;
    std::vector<float> cached_cross_attention_mask_f32;
};

// ============================================================================
// API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load action expert from GGUF file.
 */
SMOLVLA_EXPERT_API struct smolvla_action_expert * smolvla_action_expert_load(
    const char * fname, int verbosity);

/**
 * Free action expert context.
 */
SMOLVLA_EXPERT_API void smolvla_action_expert_free(struct smolvla_action_expert * ctx);

/**
 * Get expert hidden size.
 */
SMOLVLA_EXPERT_API int smolvla_action_expert_hidden_size(const struct smolvla_action_expert * ctx);

/**
 * Compute embed_suffix: noisy_actions + timestep → suffix_emb.
 *
 * @param ctx        Action expert context
 * @param n_threads  Number of CPU threads
 * @param noisy_actions  Input [chunk_size * max_action_dim] flat array
 * @param timestep       Scalar time value (1.0, 0.9, ..., 0.1)
 * @param output         Output [chunk_size * hidden_size] flat array
 * @return true on success
 */
SMOLVLA_EXPERT_API bool smolvla_action_expert_embed_suffix(
    struct smolvla_action_expert * ctx,
    int n_threads,
    const float * noisy_actions,
    float timestep,
    float * output
);

SMOLVLA_EXPERT_API bool smolvla_action_expert_eval_transformer_project_velocity(
    struct smolvla_action_expert * ctx,
    const float * hidden_in,
    const uint8_t * prefix_valid_mask,
    int prefix_seq_len,
    float * velocity_out
);

SMOLVLA_EXPERT_API bool smolvla_action_expert_prepare_prefix_kv_from_backend(
    struct smolvla_action_expert * ctx,
    int prefix_seq_len,
    struct ggml_tensor * const * prefix_k_layers,
    struct ggml_tensor * const * prefix_v_layers,
    bool prefix_v_trans
);

SMOLVLA_EXPERT_API bool smolvla_action_expert_init_fixed_prefix_runtime(
    struct smolvla_action_expert * ctx,
    int prefix_seq_len
);

#ifdef __cplusplus
}
#endif

#endif // SMOLVLA_ACTION_EXPERT_H
