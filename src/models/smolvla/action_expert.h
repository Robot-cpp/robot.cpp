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
#include "models/ggml_backend.h"

#ifdef LLAMA_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef LLAMA_BUILD
#define SMOLVLA_EXPERT_API __declspec(dllexport)
#else
#define SMOLVLA_EXPERT_API __declspec(dllimport)
#endif
#else
#define SMOLVLA_EXPERT_API __attribute__((visibility("default")))
#endif
#else
#define SMOLVLA_EXPERT_API
#endif

// ============================================================================
// Per-layer tensors
// ============================================================================
struct smolvla_expert_layer {
    // Attention norm (RMSNorm)
    struct ggml_tensor * attn_norm; // [hidden_size]

    // Attention projections
    struct ggml_tensor * attn_q; // [hidden_size, n_q_heads * head_dim]
    struct ggml_tensor *
        attn_k; // [hidden_size, n_kv_heads * head_dim] (self) or [llm_kv_dim, n_kv_heads * head_dim] (cross)
    struct ggml_tensor * attn_v;      // same dimensions as attn_k
    struct ggml_tensor * attn_output; // [n_q_heads * head_dim, hidden_size]

    // FFN norm (RMSNorm)
    struct ggml_tensor * ffn_norm; // [hidden_size]

    // FFN (SwiGLU)
    struct ggml_tensor * ffn_gate; // [hidden_size, intermediate_size]
    struct ggml_tensor * ffn_up;   // [hidden_size, intermediate_size]
    struct ggml_tensor * ffn_down; // [intermediate_size, hidden_size]

    bool is_cross_attn; // true for odd layers
};

// ============================================================================
// Action Expert Context
// ============================================================================
struct smolvla_action_expert {
    // GGUF/GGML contexts
    struct gguf_context * ctx_gguf = nullptr;
    struct ggml_context * ctx_data = nullptr;

    // Backend
    ggml_backend_t backend_cpu        = nullptr;
    ggml_backend_sched_t sched        = nullptr;
    ggml_backend_sched_t prefix_sched = nullptr;
    std::vector<ggml_backend_t> backends;
    backend_buft_policy buft_policy;
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    // ----- Configuration (from GGUF metadata) -----
    int hidden_size       = 720;  // expert hidden dim
    int intermediate_size = 2048; // FFN intermediate
    int num_layers        = 16;
    int max_action_dim    = 32; // padded action dim
    int chunk_size        = 50;
    int num_steps         = 10; // denoising steps
    int self_attn_every_n = 2;  // even layers = self-attn
    float min_period      = 0.004f;
    float max_period      = 4.0f;

    // Derived head info (from weight shapes)
    int head_dim   = 64;
    int n_q_heads  = 0; // hidden_q_dim / head_dim  (usually 15)
    int n_kv_heads = 0; // kv_dim / head_dim        (usually 5)
    int llm_kv_dim = 0; // LLM n_kv_heads * head_dim for cross-attn input (usually 320)

    // ----- Model tensors -----
    std::vector<smolvla_expert_layer> layers;

    // Action projections
    struct ggml_tensor * action_in_proj_w  = nullptr; // [max_action_dim, hidden_size]
    struct ggml_tensor * action_in_proj_b  = nullptr; // [hidden_size]
    struct ggml_tensor * action_out_proj_w = nullptr; // [hidden_size, max_action_dim]
    struct ggml_tensor * action_out_proj_b = nullptr; // [max_action_dim]

    // Time MLP: Linear(hidden*2, hidden) → SiLU → Linear(hidden, hidden)
    struct ggml_tensor * time_mlp_in_w  = nullptr; // [hidden_size*2, hidden_size]
    struct ggml_tensor * time_mlp_in_b  = nullptr; // [hidden_size]
    struct ggml_tensor * time_mlp_out_w = nullptr; // [hidden_size, hidden_size]
    struct ggml_tensor * time_mlp_out_b = nullptr; // [hidden_size]

    // Final RMSNorm
    struct ggml_tensor * final_norm = nullptr; // [hidden_size]

    // ----- Unnormalization stats -----
    std::vector<float> action_mean; // [action_dim]
    std::vector<float> action_std;  // [action_dim]
    int action_dim = 0;             // actual action dim (from stats length, e.g. 6)

    // Precomputed sinusoidal embeddings for the fixed denoising schedule.
    std::vector<float> precomputed_time_emb_2d; // [num_steps, hidden_size, chunk_size]

    struct time_embedding_runtime_t {
        bool ready                     = false;
        struct ggml_context * ctx_data = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        struct ggml_tensor * table     = nullptr; // [hidden_size * chunk_size, num_steps]
    } time_embedding_runtime;

    // ----- Compute resources -----
    std::vector<uint8_t> buf_compute_meta;
    int graph_n_threads = 0;
    int verbosity       = 0;

    struct prefix_kv_runtime_t {
        int prefix_seq_len             = -1;
        struct ggml_context * ctx_data = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        std::vector<struct ggml_tensor *> k_layers;
        std::vector<struct ggml_tensor *> v_layers;
    } prefix_kv_runtime;

    struct attention_runtime_t {
        int prefix_seq_len              = -1;
        bool prepared                   = false;
        struct ggml_context * ctx_data  = nullptr;
        ggml_backend_buffer_t buffer    = nullptr;
        struct ggml_tensor * self_pos   = nullptr;
        struct ggml_tensor * cross_pos  = nullptr;
        struct ggml_tensor * self_mask  = nullptr;
        struct ggml_tensor * cross_mask = nullptr;
    } attention_runtime;

    struct denoise_runtime_t {
        int prefix_seq_len = -1;
        bool ready         = false;
        std::vector<uint8_t> meta_buf;
        struct ggml_context * ctx_data = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        std::vector<struct ggml_tensor *> step_ids;
        struct ggml_context * ctx_graph  = nullptr;
        struct ggml_cgraph * graph       = nullptr;
        struct ggml_tensor * inp_actions = nullptr;
        struct ggml_tensor * out         = nullptr;
    } denoise_runtime;
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
SMOLVLA_EXPERT_API struct smolvla_action_expert * smolvla_action_expert_load(const char * fname, int verbosity);

/**
 * Free action expert context.
 */
SMOLVLA_EXPERT_API void smolvla_action_expert_free(struct smolvla_action_expert * ctx);

/**
 * Get expert hidden size.
 */
SMOLVLA_EXPERT_API int smolvla_action_expert_hidden_size(const struct smolvla_action_expert * ctx);

SMOLVLA_EXPERT_API bool smolvla_action_expert_eval_denoise_graph(struct smolvla_action_expert * ctx, int n_threads,
                                                                 const float * noisy_actions, float * actions_out);

SMOLVLA_EXPERT_API bool smolvla_action_expert_prepare_attention_cache(struct smolvla_action_expert * ctx,
                                                                      const uint8_t * prefix_valid_mask,
                                                                      int prefix_seq_len);

SMOLVLA_EXPERT_API bool
smolvla_action_expert_prepare_prefix_kv_from_backend(struct smolvla_action_expert * ctx, int prefix_seq_len,
                                                     struct ggml_tensor * const * prefix_k_layers,
                                                     struct ggml_tensor * const * prefix_v_layers, bool prefix_v_trans);

SMOLVLA_EXPERT_API bool smolvla_action_expert_init_fixed_prefix_runtime(struct smolvla_action_expert * ctx,
                                                                        int prefix_seq_len);

#ifdef __cplusplus
}
#endif

#endif // SMOLVLA_ACTION_EXPERT_H
