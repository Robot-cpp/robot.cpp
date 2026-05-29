// smolvla_engine.cpp — SmolVLA inference engine (stub implementation)
//
// SmolVLA Architecture:
//   Phase 1: image + state + task → VLM → KV Cache (once)
//   Phase 2: Denoising loop with Action Expert (10 steps)

#include "smolvla_engine.h"
#include "state_proj.h"
#include "vision.h"
#include "action_expert.h"
#include "llama.h"
#include "ggml.h"
#include "smolvla_llama_compat.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <climits>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
#include <string>

// ============================================================================
// Internal context structure
// ============================================================================
struct smolvla_context {
    // Models
    llama_model * vlm;
    llama_context * ctx_llama;
    smolvla_vision_ctx * vision;
    smolvla_state_proj * state_proj;
    smolvla_action_expert * expert;
    ggml_tensor * tok_embd;

    // Configuration
    int action_dim;
    int chunk_size;
    int num_steps;
    int n_threads;
    int n_batch;
    int n_ctx;
    int n_embd;
    int noise_mode;
    int64_t noise_seed;
    int verbosity;
    std::mt19937_64 noise_rng;

    // Result buffer
    std::vector<float> action_buffer;
    smolvla_stage_timings last_timings;

    // State embedding buffer (from state_proj)
    std::vector<float> state_emb;

    // Vision embedding buffer
    std::vector<float> vision_emb;
    // LLM debug hidden buffer (last token)
    std::vector<float> llm_last_hidden;

    // Task string
    std::string task;
    // Number of synthetic empty cameras to match SmolVLA prefix construction.
    int empty_cameras;
    // Cached embeddings for deterministic padding paths.
    std::vector<float> empty_cam_emb_cache;
    bool empty_cam_emb_cache_ready;
    std::vector<float> lang_pad_emb_cache;
    llama_token lang_pad_emb_cache_id;
    bool lang_pad_emb_cache_ready;

    // VLM KV metadata extracted after Phase 1
    int fixed_prefix_seq_len;
    int vlm_kv_seq_len;    // number of prefix tokens in KV cache
    int vlm_kv_dim;        // n_kv_heads * head_dim = 320
    std::vector<bool> prefix_valid_mask;

};

using smolvla_clock = std::chrono::steady_clock;

// TODO: 这里的SMOLVLA_LANG_MAX_LEN 48 和SMOLVLA_LANG_PAD_ID pad_id=2应该也要改成从gguf里读的配置，而不是写死的了
static constexpr int SMOLVLA_LANG_MAX_LEN = 48;
static constexpr llama_token SMOLVLA_LANG_PAD_ID = 2;

static double smolvla_elapsed_ms(
    const smolvla_clock::time_point & start,
    const smolvla_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::vector<llama_token> smolvla_tokenize(struct llama_model * model, const std::string & text, bool add_special, bool parse_special) {
    std::vector<llama_token> result;
    if (!model) {
        return result;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    if (n_tokens <= 0) {
        return result;
    }
    result.resize(n_tokens);
    const int n_check = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), result.data(), (int32_t) result.size(), add_special, parse_special);
    if (n_check < 0) {
        result.clear();
    }
    return result;
}

static std::vector<llama_token> smolvla_pad_tokens_right(
    const std::vector<llama_token> & tokens,
    int target_len,
    llama_token pad_id) {
    if (target_len <= 0) {
        return tokens;
    }
    std::vector<llama_token> out = tokens;
    if ((int) out.size() > target_len) {
        out.resize(target_len);
        return out;
    }
    out.resize(target_len, pad_id);
    return out;
}

static bool smolvla_env_has_value(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0';
}

static bool smolvla_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }

    if (strcmp(value, "0") == 0 ||
        strcmp(value, "false") == 0 ||
        strcmp(value, "FALSE") == 0 ||
        strcmp(value, "off") == 0 ||
        strcmp(value, "OFF") == 0 ||
        strcmp(value, "no") == 0 ||
        strcmp(value, "NO") == 0) {
        return false;
    }

    return true;
}

static void smolvla_log_matmul_shape(
    const char * scope,
    const char * name,
    const ggml_tensor * weight,
    int tokens) {
    if (!weight) {
        return;
    }

    const long long in_features = (long long) weight->ne[0];
    const long long out_features = (long long) weight->ne[1];
    fprintf(stderr,
            "[SmolVLA][shape][%s] %s: W[%lld,%lld] x X[%lld,%d] -> Y[%lld,%d]\n",
            scope,
            name,
            out_features,
            in_features,
            in_features,
            tokens,
            out_features,
            tokens);
}

static void smolvla_log_shape_debug(const smolvla_context * ctx) {
    if (!ctx || !smolvla_env_flag_enabled("SMOLVLA_SHAPE_DEBUG")) {
        return;
    }

    if (ctx->expert) {
        const int action_tokens = ctx->expert->chunk_size;
        const int prefix_tokens = ctx->fixed_prefix_seq_len > 0 ? ctx->fixed_prefix_seq_len : 0;
        const int self_tokens = prefix_tokens + action_tokens;
        fprintf(stderr,
                "[SmolVLA][shape][action] hidden=%d intermediate=%d chunk=%d prefix=%d self_kv=%d heads(q=%d kv=%d head_dim=%d)\n",
                ctx->expert->hidden_size,
                ctx->expert->intermediate_size,
                action_tokens,
                prefix_tokens,
                self_tokens,
                ctx->expert->n_q_heads,
                ctx->expert->n_kv_heads,
                ctx->expert->head_dim);
        smolvla_log_matmul_shape("action", "action_in_proj", ctx->expert->action_in_proj_w, action_tokens);
        smolvla_log_matmul_shape("action", "time_mlp_in", ctx->expert->time_mlp_in_w, action_tokens);
        smolvla_log_matmul_shape("action", "time_mlp_out", ctx->expert->time_mlp_out_w, action_tokens);
        if (!ctx->expert->layers.empty()) {
            const auto & self_layer = ctx->expert->layers.front();
            smolvla_log_matmul_shape("action", "self_attn_q", self_layer.attn_q, action_tokens);
            smolvla_log_matmul_shape("action", "self_attn_k", self_layer.attn_k, action_tokens);
            smolvla_log_matmul_shape("action", "self_attn_v", self_layer.attn_v, action_tokens);
            smolvla_log_matmul_shape("action", "self_attn_out", self_layer.attn_output, action_tokens);
            smolvla_log_matmul_shape("action", "self_ffn_gate", self_layer.ffn_gate, action_tokens);
            smolvla_log_matmul_shape("action", "self_ffn_up", self_layer.ffn_up, action_tokens);
            smolvla_log_matmul_shape("action", "self_ffn_down", self_layer.ffn_down, action_tokens);
        }
        for (const auto & layer : ctx->expert->layers) {
            if (!layer.is_cross_attn) {
                continue;
            }
            smolvla_log_matmul_shape("action", "cross_attn_q", layer.attn_q, action_tokens);
            smolvla_log_matmul_shape("action", "cross_attn_k", layer.attn_k, prefix_tokens);
            smolvla_log_matmul_shape("action", "cross_attn_v", layer.attn_v, prefix_tokens);
            smolvla_log_matmul_shape("action", "cross_attn_out", layer.attn_output, action_tokens);
            smolvla_log_matmul_shape("action", "cross_ffn_gate", layer.ffn_gate, action_tokens);
            smolvla_log_matmul_shape("action", "cross_ffn_up", layer.ffn_up, action_tokens);
            smolvla_log_matmul_shape("action", "cross_ffn_down", layer.ffn_down, action_tokens);
            break;
        }
    }

    if (ctx->vlm) {
        const int llama_tokens = ctx->fixed_prefix_seq_len > 0 ? ctx->fixed_prefix_seq_len : SMOLVLA_LANG_MAX_LEN;
        ggml_tensor * q = llama_get_model_tensor(ctx->vlm, "blk.0.attn_q.weight");
        ggml_tensor * k = llama_get_model_tensor(ctx->vlm, "blk.0.attn_k.weight");
        ggml_tensor * v = llama_get_model_tensor(ctx->vlm, "blk.0.attn_v.weight");
        ggml_tensor * out = llama_get_model_tensor(ctx->vlm, "blk.0.attn_output.weight");
        ggml_tensor * gate = llama_get_model_tensor(ctx->vlm, "blk.0.ffn_gate.weight");
        ggml_tensor * up = llama_get_model_tensor(ctx->vlm, "blk.0.ffn_up.weight");
        ggml_tensor * down = llama_get_model_tensor(ctx->vlm, "blk.0.ffn_down.weight");

        fprintf(stderr,
                "[SmolVLA][shape][llama] tokens=%d n_embd=%d\n",
                llama_tokens,
                ctx->n_embd);
        smolvla_log_matmul_shape("llama", "attn_q", q, llama_tokens);
        smolvla_log_matmul_shape("llama", "attn_k", k, llama_tokens);
        smolvla_log_matmul_shape("llama", "attn_v", v, llama_tokens);
        smolvla_log_matmul_shape("llama", "attn_out", out, llama_tokens);
        smolvla_log_matmul_shape("llama", "ffn_gate", gate, llama_tokens);
        smolvla_log_matmul_shape("llama", "ffn_up", up, llama_tokens);
        smolvla_log_matmul_shape("llama", "ffn_down", down, llama_tokens);
    }
}

static const char * smolvla_noise_mode_name(int noise_mode) {
    switch (noise_mode) {
        case SMOLVLA_NOISE_MODE_GAUSSIAN:
            return "gaussian";
        case SMOLVLA_NOISE_MODE_DEBUG_SIN:
            return "debug_sin";
        default:
            return "unknown";
    }
}

static bool dump_m6_step(
    const smolvla_context * ctx,
    int step_idx,
    const float * x_t_in,
    const float * v_t,
    const float * x_t_out,
    int chunk,
    int action_dim) {
    const char * dump_dir_env = std::getenv("SMOLVLA_M6_DUMP_DIR");
    if (!dump_dir_env || dump_dir_env[0] == '\0') {
        return true;
    }

    std::filesystem::path dump_dir(dump_dir_env);
    std::error_code ec;
    std::filesystem::create_directories(dump_dir, ec);
    if (ec) {
        fprintf(stderr, "[M6 dump] failed to create %s: %s\n", dump_dir_env, ec.message().c_str());
        return false;
    }

    std::ofstream meta(dump_dir / "meta.txt", std::ios::binary);
    if (!meta) {
        fprintf(stderr, "[M6 dump] failed to open meta.txt\n");
        return false;
    }
    meta << "num_steps " << ctx->num_steps << "\n";
    meta << "chunk_size " << chunk << "\n";
    meta << "action_dim " << action_dim << "\n";
    meta.close();

    char xin_name[64];
    char vt_name[64];
    char xout_name[64];
    snprintf(xin_name, sizeof(xin_name), "step%02d_x_t_in.bin", step_idx);
    snprintf(vt_name, sizeof(vt_name), "step%02d_v_t.bin", step_idx);
    snprintf(xout_name, sizeof(xout_name), "step%02d_x_t_out.bin", step_idx);

    std::ofstream xin_f(dump_dir / xin_name, std::ios::binary);
    std::ofstream vt_f(dump_dir / vt_name, std::ios::binary);
    std::ofstream xout_f(dump_dir / xout_name, std::ios::binary);
    if (!xin_f || !vt_f || !xout_f) {
        fprintf(stderr, "[M6 dump] failed to open dump files under %s\n", dump_dir_env);
        return false;
    }

    xin_f.write(reinterpret_cast<const char *>(x_t_in), (size_t) chunk * action_dim * sizeof(float));
    vt_f.write(reinterpret_cast<const char *>(v_t), (size_t) chunk * action_dim * sizeof(float));
    xout_f.write(reinterpret_cast<const char *>(x_t_out), (size_t) chunk * action_dim * sizeof(float));
    return true;
}

static bool dump_m7_actions(
    const smolvla_context * ctx,
    const float * actions,
    int chunk,
    int action_dim) {
    const char * dump_dir_env = std::getenv("SMOLVLA_M7_DUMP_DIR");
    if (!dump_dir_env || dump_dir_env[0] == '\0') {
        return true;
    }

    std::filesystem::path dump_dir(dump_dir_env);
    std::error_code ec;
    std::filesystem::create_directories(dump_dir, ec);
    if (ec) {
        fprintf(stderr, "[M7 dump] failed to create %s: %s\n", dump_dir_env, ec.message().c_str());
        return false;
    }

    std::ofstream meta(dump_dir / "meta.txt", std::ios::binary);
    if (!meta) {
        fprintf(stderr, "[M7 dump] failed to open meta.txt\n");
        return false;
    }
    meta << "chunk_size " << chunk << "\n";
    meta << "action_dim " << action_dim << "\n";
    meta.close();

    std::ofstream fout(dump_dir / "final_actions.bin", std::ios::binary);
    if (!fout) {
        fprintf(stderr, "[M7 dump] failed to open final_actions.bin\n");
        return false;
    }

    fout.write(reinterpret_cast<const char *>(actions), (size_t) chunk * action_dim * sizeof(float));
    return true;
}

// Extract VLM KV cache from llama.cpp after Phase 1 and prepare the
// expert-side prefix runtime directly from backend tensors.
static bool extract_vlm_kv_cache(smolvla_context * ctx) {
    const int n_layers = ctx->expert->num_layers;
    const int kv_dim   = ctx->expert->n_kv_heads * ctx->expert->head_dim;  // 5*64 = 320
    const int seq_len  = ctx->fixed_prefix_seq_len;
    const bool v_trans = llama_kv_cache_is_v_trans(ctx->ctx_llama);

    if (seq_len <= 0) {
        fprintf(stderr, "[KV extract] Error: no tokens in KV cache\n");
        return false;
    }
    if (ctx->fixed_prefix_seq_len > 0 && seq_len != ctx->fixed_prefix_seq_len) {
        fprintf(stderr, "[KV extract] Error: unexpected seq_len=%d, expected fixed prefix len=%d\n",
                seq_len, ctx->fixed_prefix_seq_len);
        return false;
    }
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[KV extract] seq_len=%d, kv_dim=%d, v_trans=%d\n",
                seq_len, kv_dim, v_trans);
    }

    ctx->vlm_kv_seq_len = seq_len;
    ctx->vlm_kv_dim     = kv_dim;
    std::vector<struct ggml_tensor *> prefix_k_tensors(n_layers, nullptr);
    std::vector<struct ggml_tensor *> prefix_v_tensors(n_layers, nullptr);
    for (int il = 0; il < n_layers; ++il) {
        struct ggml_tensor * k_tensor = llama_kv_cache_k_tensor(ctx->ctx_llama, il);
        struct ggml_tensor * v_tensor = llama_kv_cache_v_tensor(ctx->ctx_llama, il);
        if (!k_tensor || !v_tensor) {
            fprintf(stderr, "[KV extract] Error: missing KV tensor for layer %d\n", il);
            return false;
        }
        prefix_k_tensors[il] = k_tensor;
        prefix_v_tensors[il] = v_tensor;
    }

    if (!smolvla_action_expert_prepare_prefix_kv_from_backend(
            ctx->expert,
            seq_len,
            prefix_k_tensors.data(),
            prefix_v_tensors.data(),
            v_trans)) {
        fprintf(stderr, "[KV extract] Error: failed to prepare prefix KV runtime from backend tensors\n");
        return false;
    }

    return true;
}

static bool lookup_token_embeddings(
    struct ggml_tensor * tok_embd,
    const llama_token * tokens,
    int n_tokens,
    int n_embd,
    float * output) {
    if (!tok_embd || !tokens || !output || n_tokens <= 0 || n_embd <= 0) {
        return false;
    }

    if (tok_embd->ne[0] != n_embd) {
        fprintf(stderr, "[SmolVLA] Error: token_embd dim mismatch, ne[0]=%lld vs n_embd=%d\n",
                (long long) tok_embd->ne[0], n_embd);
        return false;
    }

    const size_t row_stride = (size_t) tok_embd->nb[1];
    const int64_t n_vocab = tok_embd->ne[1];

    if (tok_embd->type == GGML_TYPE_F32) {
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                fprintf(stderr, "[SmolVLA] Error: token id out of range: %d (n_vocab=%lld)\n",
                        (int) tokens[i], (long long) n_vocab);
                return false;
            }
            const size_t offset = (size_t) tokens[i] * row_stride;
            ggml_backend_tensor_get(tok_embd, output + i * n_embd, offset, n_embd * sizeof(float));
        }
        return true;
    }

    if (tok_embd->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> row_buf(n_embd);
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                fprintf(stderr, "[SmolVLA] Error: token id out of range: %d (n_vocab=%lld)\n",
                        (int) tokens[i], (long long) n_vocab);
                return false;
            }
            const size_t offset = (size_t) tokens[i] * row_stride;
            ggml_backend_tensor_get(tok_embd, row_buf.data(), offset, n_embd * sizeof(ggml_fp16_t));
            for (int j = 0; j < n_embd; j++) {
                output[i * n_embd + j] = ggml_fp16_to_fp32(row_buf[j]);
            }
        }
        return true;
    }

    fprintf(stderr, "[SmolVLA] Error: unsupported token_embd type %d\n", tok_embd->type);
    return false;
}

// ============================================================================
// API Implementation
// ============================================================================

struct smolvla_params smolvla_default_params(void) {
    struct smolvla_params params;
    memset(&params, 0, sizeof(params));

    params.vlm_path           = nullptr;
    params.mmproj_path        = nullptr;
    params.state_proj_path    = nullptr;
    params.action_expert_path = nullptr;
    params.task               = "grab the block.";
    params.n_threads          = 0;  // auto
    params.action_dim         = 6;
    params.chunk_size         = 50;
    params.num_steps          = 10;
    params.n_batch            = 512;
    params.n_ctx              = 2048;
    params.noise_mode         = SMOLVLA_NOISE_MODE_GAUSSIAN;
    params.noise_seed         = -1;
    params.verbosity          = 1;

    return params;
}

static struct smolvla_result smolvla_empty_result(void) {
    struct smolvla_result result;
    result.actions = nullptr;
    result.chunk_size = 0;
    result.action_dim = 0;
    return result;
}

static struct smolvla_stage_timings smolvla_empty_stage_timings(void) {
    struct smolvla_stage_timings timings = {};
    return timings;
}

static void smolvla_reset_last_stage_timings(struct smolvla_context * ctx) {
    if (ctx) {
        ctx->last_timings = smolvla_empty_stage_timings();
    }
}

static struct smolvla_result smolvla_finish_predict(
    struct smolvla_context * ctx,
    struct smolvla_result result,
    const smolvla_clock::time_point & start_time) {
    if (ctx) {
        ctx->last_timings.total_ms = smolvla_elapsed_ms(start_time, smolvla_clock::now());
    }
    return result;
}

struct smolvla_context * smolvla_init(struct smolvla_params params) {
    if (params.verbosity >= 1) {
        fprintf(stderr, "\n");
        fprintf(stderr, "===========================================\n");
        fprintf(stderr, "  SmolVLA Engine Initialization\n");
        fprintf(stderr, "===========================================\n");
        fprintf(stderr, "VLM:           %s\n", params.vlm_path ? params.vlm_path : "(none)");
        fprintf(stderr, "Vision:        %s\n", params.mmproj_path ? params.mmproj_path : "(none)");
        fprintf(stderr, "State Proj:    %s\n", params.state_proj_path ? params.state_proj_path : "(none)");
        fprintf(stderr, "Action Expert: %s\n", params.action_expert_path ? params.action_expert_path : "(none)");
        fprintf(stderr, "Task:          %s\n", params.task ? params.task : "(none)");
        fprintf(stderr, "Threads:       %d\n", params.n_threads);
        fprintf(stderr, "Action dim:    %d\n", params.action_dim);
        fprintf(stderr, "Chunk size:    %d\n", params.chunk_size);
        fprintf(stderr, "Denoise steps: %d\n", params.num_steps);
        fprintf(stderr, "Noise mode:    %s\n", smolvla_noise_mode_name(params.noise_mode));
        fprintf(stderr, "Noise seed:    %s\n", params.noise_seed >= 0 ? "(set)" : "auto");
        if (params.noise_seed >= 0) {
            fprintf(stderr, "Noise seed v:  %lld\n", (long long) params.noise_seed);
        }
        fprintf(stderr, "===========================================\n\n");
    }

    // Allocate context
    smolvla_context * ctx = new smolvla_context();
    ctx->vlm       = nullptr;
    ctx->ctx_llama = nullptr;
    ctx->tok_embd  = nullptr;
    ctx->expert    = nullptr;
    ctx->action_dim = params.action_dim;
    ctx->chunk_size = params.chunk_size;
    ctx->num_steps  = params.num_steps;
    ctx->n_threads  = params.n_threads > 0 ? params.n_threads : 4;
    ctx->n_batch    = params.n_batch > 0 ? params.n_batch : 512;
    ctx->n_ctx      = params.n_ctx > 0 ? params.n_ctx : 2048;
    ctx->n_embd     = 0;
    ctx->noise_mode = params.noise_mode;
    if (ctx->noise_mode != SMOLVLA_NOISE_MODE_GAUSSIAN &&
        ctx->noise_mode != SMOLVLA_NOISE_MODE_DEBUG_SIN) {
        fprintf(stderr, "[SmolVLA] Warning: invalid noise_mode=%d, falling back to gaussian\n", ctx->noise_mode);
        ctx->noise_mode = SMOLVLA_NOISE_MODE_GAUSSIAN;
    }
    ctx->noise_seed = params.noise_seed;
    ctx->verbosity  = params.verbosity;
    ctx->task       = params.task ? params.task : "";
    ctx->empty_cameras = 2; // TODO: 这里写死了2个空摄像头，之后应该改成gguf里读
    ctx->fixed_prefix_seq_len = -1;
    ctx->empty_cam_emb_cache_ready = false;
    ctx->lang_pad_emb_cache_ready = false;
    ctx->lang_pad_emb_cache_id = -1;

    if (ctx->noise_seed >= 0) {
        ctx->noise_rng.seed((uint64_t) ctx->noise_seed);
    } else {
        std::random_device rd;
        std::seed_seq seq{
            rd(), rd(), rd(), rd(),
            (unsigned) std::chrono::high_resolution_clock::now().time_since_epoch().count()
        };
        ctx->noise_rng.seed(seq);
    }

    // Allocate action buffer
    ctx->action_buffer.resize(ctx->chunk_size * ctx->action_dim, 0.0f);
    ctx->last_timings = smolvla_empty_stage_timings();

    // Load State Projector
    ctx->state_proj = nullptr;
    if (params.state_proj_path) {
        ctx->state_proj = smolvla_state_proj_load(params.state_proj_path, params.verbosity);
        if (!ctx->state_proj) {
            fprintf(stderr, "[SmolVLA] Warning: Failed to load state_proj, continuing without\n");
        } else {
            int hidden_size = smolvla_state_proj_hidden_size(ctx->state_proj);
            ctx->state_emb.resize(hidden_size, 0.0f);
        }
    }

    // Load Vision model
    ctx->vision = nullptr;
    if (params.mmproj_path) {
        ctx->vision = smolvla_vision_load(params.mmproj_path, params.verbosity);
        if (!ctx->vision) {
            fprintf(stderr, "[SmolVLA] Warning: Failed to load vision model, continuing without\n");
        } else {
            int embd_size = smolvla_vision_embd_size(ctx->vision);
            ctx->vision_emb.resize(embd_size, 0.0f);

            if (ctx->empty_cameras > 0) {
                const int n_tokens_per_cam = smolvla_vision_n_tokens(ctx->vision);
                const int proj_dim = ctx->vision->proj_dim;
                const size_t one_cam_size = (size_t) n_tokens_per_cam * proj_dim;

                ctx->empty_cam_emb_cache = smolvla_vision_encode_constant(ctx->vision, -1.0f, ctx->n_threads);
                if (ctx->empty_cam_emb_cache.size() != one_cam_size) {
                    if (!ctx->empty_cam_emb_cache.empty()) {
                        fprintf(stderr, "[SmolVLA] Warning: empty camera embedding size mismatch during init\n");
                    }
                    ctx->empty_cam_emb_cache.clear();
                    ctx->empty_cam_emb_cache_ready = false;
                } else {
                    ctx->empty_cam_emb_cache_ready = true;
                }
            }
        }
    }

    // Load VLM model via llama.cpp (reuses standard llama runtime)
    if (params.vlm_path) {
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        ctx->vlm = llama_load_model_from_file(params.vlm_path, model_params);
        if (!ctx->vlm) {
            fprintf(stderr, "[SmolVLA] Error: failed to load VLM from %s\n", params.vlm_path);
            smolvla_free(ctx);
            return nullptr;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = ctx->n_ctx;
        cparams.n_batch = ctx->n_batch;
        cparams.n_seq_max = 3;
        cparams.n_threads = ctx->n_threads;
        cparams.n_threads_batch = ctx->n_threads;
        cparams.embeddings = true;
        cparams.kv_unified = true;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        // F32 K cache for better precision matching Python (V cache must stay F16 without flash_attn)
        cparams.type_k = GGML_TYPE_F32;
        //TODO: Python apply_rope uses default max_wavelength=10000 (not config's 100000)
        // to match the python implementation, we set the same default here
        cparams.rope_freq_base = 10000.0f;

        ctx->ctx_llama = llama_new_context_with_model(ctx->vlm, cparams);
        if (!ctx->ctx_llama) {
            fprintf(stderr, "[SmolVLA] Error: failed to create llama context\n");
            smolvla_free(ctx);
            return nullptr;
        }

        ctx->n_embd = llama_n_embd(ctx->vlm);
        ctx->tok_embd = llama_get_model_tensor(ctx->vlm, "token_embd.weight");
        if (!ctx->tok_embd) {
            fprintf(stderr, "[SmolVLA] Error: failed to get token_embd.weight\n");
            smolvla_free(ctx);
            return nullptr;
        }
        if (ctx->verbosity >= 1) {
            fprintf(stderr, "[SmolVLA] VLM loaded: n_embd=%d, token_embd(type=%d shape=[%d, %d])\n",
                    ctx->n_embd, ctx->tok_embd->type, (int) ctx->tok_embd->ne[0], (int) ctx->tok_embd->ne[1]);
        }
    }

    // Load Action Expert
    if (params.action_expert_path) {
        ctx->expert = smolvla_action_expert_load(params.action_expert_path, params.verbosity);
        if (!ctx->expert) {
            fprintf(stderr, "[SmolVLA] Warning: Failed to load action expert, continuing without\n");
        }
    }

    if (ctx->vision && ctx->expert) {
        const int n_vis = smolvla_vision_n_tokens(ctx->vision) * (1 + std::max(0, ctx->empty_cameras));
        const int n_state = !ctx->state_emb.empty() ? 1 : 0;
        ctx->fixed_prefix_seq_len = n_vis + SMOLVLA_LANG_MAX_LEN + n_state;
        if (!smolvla_action_expert_init_fixed_prefix_runtime(ctx->expert, ctx->fixed_prefix_seq_len)) {
            fprintf(stderr, "[SmolVLA] Error: failed to initialize fixed prefix runtime (seq_len=%d)\n",
                    ctx->fixed_prefix_seq_len);
            smolvla_free(ctx);
            return nullptr;
        }
        if (ctx->verbosity >= 1) {
            fprintf(stderr, "[SmolVLA] Fixed prefix runtime initialized: vision=%d lang=%d state=%d total=%d\n",
                    n_vis, SMOLVLA_LANG_MAX_LEN, n_state, ctx->fixed_prefix_seq_len);
        }
    }

    smolvla_log_shape_debug(ctx);

    if (params.verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] Engine initialized%s\n",
                ctx->expert ? "" : " (no action expert — Phase 2 will be stub)");
        fprintf(stderr, "[SmolVLA] Action buffer: %d x %d = %d floats\n",
                ctx->chunk_size, ctx->action_dim, ctx->chunk_size * ctx->action_dim);
    }

    return ctx;
}

// 主相机+空相机的视觉嵌入准备：如果有空相机则用constant encoding填充
static bool smolvla_prepare_vision_embedding(
    struct smolvla_context * ctx,
    const std::vector<float> & main_cam_emb
) {
    if (!ctx || !ctx->vision || main_cam_emb.empty()) {
        if (ctx) {
            ctx->vision_emb.clear();
        }
        return false;
    }

    const int n_tokens_per_cam = smolvla_vision_n_tokens(ctx->vision);
    const int proj_dim = ctx->vision->proj_dim;
    const size_t one_cam_size = (size_t) n_tokens_per_cam * proj_dim;
    // TODO: 这里现在是写死了一个主摄像头
    if (main_cam_emb.size() != one_cam_size) {
        fprintf(stderr, "[SmolVLA] Error: vision output size mismatch (%zu vs %zu)\n",
                main_cam_emb.size(), one_cam_size);
        ctx->vision_emb.clear();
        return false;
    }

    ctx->vision_emb.clear();
    ctx->vision_emb.reserve((size_t) (1 + std::max(0, ctx->empty_cameras)) * one_cam_size);
    ctx->vision_emb.insert(ctx->vision_emb.end(), main_cam_emb.begin(), main_cam_emb.end());

    for (int i = 0; i < ctx->empty_cameras; ++i) {
        if (!ctx->empty_cam_emb_cache_ready) {
            ctx->empty_cam_emb_cache = smolvla_vision_encode_constant(ctx->vision, -1.0f, ctx->n_threads);
            if (ctx->empty_cam_emb_cache.size() != one_cam_size) {
                fprintf(stderr, "[SmolVLA] Error: empty camera embedding size mismatch\n");
                ctx->vision_emb.clear();
                return false;
            }
            ctx->empty_cam_emb_cache_ready = true;
        }
        ctx->vision_emb.insert(ctx->vision_emb.end(), ctx->empty_cam_emb_cache.begin(), ctx->empty_cam_emb_cache.end());
    }

    // Match SmolVLA Python: img_emb = img_emb * sqrt(hidden_size)
    const float vision_scale = std::sqrt((float) proj_dim);
    for (float & v : ctx->vision_emb) {
        v *= vision_scale;
    }

    if (ctx->verbosity >= 1) {
        const int n_tokens = n_tokens_per_cam * (1 + std::max(0, ctx->empty_cameras));
        fprintf(stdout, "[SmolVLA] image padding: main=1 empty=%d tokens_per_cam=%d\n",
                ctx->empty_cameras, n_tokens_per_cam);
        fprintf(stdout, "[SmolVLA] vision output shape: [%d, %d]\n", n_tokens, proj_dim);
        fprintf(stdout, "[SmolVLA] vision output (first 10): [");
        for (int i = 0; i < 10 && i < (int) ctx->vision_emb.size(); i++) {
            fprintf(stdout, "%.6f%s", ctx->vision_emb[i], i < 9 ? ", " : "");
        }
        fprintf(stdout, "]\n");
        fprintf(stdout, "[SmolVLA] vision output (last 10): [");
        const int n = (int) ctx->vision_emb.size();
        for (int i = n - 10; i < n; i++) {
            if (i >= 0) {
                fprintf(stdout, "%.6f%s", ctx->vision_emb[i], i < n - 1 ? ", " : "");
            }
        }
        fprintf(stdout, "]\n");
    }

    return true;
}

// 产生noise：如果是debug模式就产生sin波，否则产生高斯噪声
static void smolvla_fill_initial_noise(
    struct smolvla_context * ctx,
    float * out,
    int chunk,
    int padded_action_dim
) {
    if (ctx->noise_mode == SMOLVLA_NOISE_MODE_DEBUG_SIN) {
        for (int i = 0; i < chunk; ++i) {
            for (int j = 0; j < padded_action_dim; ++j) {
                out[i * padded_action_dim + j] = sinf((i * padded_action_dim + j) * 0.1f);
            }
        }
        return;
    }

    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < chunk * padded_action_dim; ++i) {
        out[i] = dist(ctx->noise_rng);
    }
}

static struct smolvla_result smolvla_predict_impl(
    struct smolvla_context * ctx,
    const char * image_label,
    const float * state,
    int state_dim)
{
    struct smolvla_result result = smolvla_empty_result();

    if (!ctx) {
        fprintf(stderr, "[SmolVLA] Error: null context\n");
        return result;
    }

    const bool debug_trace = smolvla_env_flag_enabled("SMOLVLA_DEBUG");
    const bool llm_debug = debug_trace;
    const bool m6_debug = debug_trace || smolvla_env_has_value("SMOLVLA_M6_DUMP_DIR");
    const bool m7_debug = debug_trace || smolvla_env_has_value("SMOLVLA_M7_DUMP_DIR");

    if (ctx->verbosity >= 1) {
        fprintf(stdout, "\n");
        fprintf(stdout, "===========================================\n");
        fprintf(stdout, "  SmolVLA Prediction\n");
        fprintf(stdout, "===========================================\n");
        fprintf(stdout, "Image: %s\n", image_label ? image_label : "(none)");
        fprintf(stdout, "State dim: %d\n", state_dim);
        if (state && state_dim > 0) {
            fprintf(stdout, "State: [");
            for (int i = 0; i < state_dim && i < 6; i++) {
                fprintf(stdout, "%.4f%s", state[i], i < state_dim - 1 ? ", " : "");
            }
            if (state_dim > 6) fprintf(stdout, ", ...");
            fprintf(stdout, "]\n");
        }
        fprintf(stdout, "Task: %s\n", ctx->task.c_str());
        fprintf(stdout, "===========================================\n\n");
    }

    if (ctx->vision_emb.empty()) {
        fprintf(stderr, "[SmolVLA] Error: missing vision embedding for prediction\n");
        return result;
    }

    // Step 1.3: Run state_proj (normalize + Linear)
    if (ctx->state_proj && state && state_dim > 0) {
        const auto t_state0 = smolvla_clock::now();
        //TODO: fix thread = 4, multi-threading seems to cause some slowdown here, need to investigate
        bool success = smolvla_state_proj_encode(ctx->state_proj, 4, state, state_dim, ctx->state_emb.data());
        ctx->last_timings.state_proj_ms = smolvla_elapsed_ms(t_state0, smolvla_clock::now());
        if (success) {
            if (ctx->verbosity >= 1) {
                fprintf(stdout, "[SmolVLA] state_proj output (first 10): [");
                for (int i = 0; i < 10 && i < (int)ctx->state_emb.size(); i++) {
                    fprintf(stdout, "%.6f%s", ctx->state_emb[i], i < 9 ? ", " : "");
                }
                fprintf(stdout, "]\n");
                fprintf(stdout, "[SmolVLA] state_proj output (last 10): [");
                int n = (int)ctx->state_emb.size();
                for (int i = n - 10; i < n; i++) {
                    if (i >= 0) fprintf(stdout, "%.6f%s", ctx->state_emb[i], i < n - 1 ? ", " : "");
                }
                fprintf(stdout, "]\n");
            }
        } else {
            fprintf(stderr, "[SmolVLA] Warning: state_proj encode failed\n");
        }
    }

    if (ctx->ctx_llama && !ctx->vision_emb.empty() && ctx->tok_embd) {
        const auto t_vlm0 = smolvla_clock::now();
        // ====1.3 Tokenize task and embed
        // Match SmolVLA preprocessor behavior: ensure trailing newline before tokenization.
        std::string task_text = ctx->task;
        if (task_text.empty() || task_text.back() != '\n') {
            task_text.push_back('\n');
        }

        std::vector<llama_token> task_tokens_raw = smolvla_tokenize(ctx->vlm, task_text, true, true);
        if (task_tokens_raw.empty()) {
            fprintf(stderr, "[SmolVLA] Error: task tokenization failed\n");
            return result;
        }
        // Match SmolVLA config: tokenizer_max_length=48, right padding.
        // TODO: 这里的SMOLVLA_LANG_MAX_LEN 48 和SMOLVLA_LANG_PAD_ID pad_id=2应该也要改成从gguf里读的配置，而不是写死的了
        std::vector<llama_token> task_tokens = smolvla_pad_tokens_right(task_tokens_raw, SMOLVLA_LANG_MAX_LEN, SMOLVLA_LANG_PAD_ID);

        if (ctx->verbosity >= 1) {
            fprintf(stdout, "[SmolVLA] task tokens(raw): n=%d, first 10=[", (int) task_tokens_raw.size());
            for (int i = 0; i < 10 && i < (int) task_tokens_raw.size(); ++i) {
                fprintf(stdout, "%d%s", (int) task_tokens_raw[i], i < 9 && i + 1 < (int) task_tokens_raw.size() ? ", " : "");
            }
            fprintf(stdout, "]\n");
            fprintf(stdout, "[SmolVLA] task tokens(padded): n=%d, first 10=[", (int) task_tokens.size());
            for (int i = 0; i < 10 && i < (int) task_tokens.size(); ++i) {
                fprintf(stdout, "%d%s", (int) task_tokens[i], i < 9 && i + 1 < (int) task_tokens.size() ? ", " : "");
            }
            fprintf(stdout, "]\n");
        }

        std::vector<float> lang_embeds((size_t) task_tokens.size() * ctx->n_embd, 0.0f);
        const int n_raw = std::min((int) task_tokens_raw.size(), (int) task_tokens.size());
        if (n_raw > 0) {
            if (!lookup_token_embeddings(ctx->tok_embd, task_tokens.data(), n_raw, ctx->n_embd, lang_embeds.data())) {
                fprintf(stderr, "[SmolVLA] Error: language token embedding lookup failed\n");
                return result;
            }
        }
        if (n_raw < (int) task_tokens.size()) {
            if (!ctx->lang_pad_emb_cache_ready || ctx->lang_pad_emb_cache_id != SMOLVLA_LANG_PAD_ID) {
                ctx->lang_pad_emb_cache.resize(ctx->n_embd);
                // TODO: 感觉其实这里也可以扔到load阶段去算，然后在predict的时候直接拼padding就行了
                if (!lookup_token_embeddings(ctx->tok_embd, &SMOLVLA_LANG_PAD_ID, 1, ctx->n_embd, ctx->lang_pad_emb_cache.data())) {
                    fprintf(stderr, "[SmolVLA] Error: pad token embedding lookup failed\n");
                    return result;
                }
                ctx->lang_pad_emb_cache_id = SMOLVLA_LANG_PAD_ID;
                ctx->lang_pad_emb_cache_ready = true;
            }
            for (int i = n_raw; i < (int) task_tokens.size(); ++i) {
                memcpy(lang_embeds.data() + (size_t) i * ctx->n_embd,
                       ctx->lang_pad_emb_cache.data(),
                       (size_t) ctx->n_embd * sizeof(float));
            }
        }
        // Match SmolVLA Python: lang_emb = lang_emb * sqrt(lang_emb_dim).
        const float lang_scale = std::sqrt((float) ctx->n_embd);
        for (float & v : lang_embeds) {
            v *= lang_scale;
        }

        if (llm_debug) {
            fprintf(stdout, "\n=== Language Embedding Output ===\n");
            fprintf(stdout, "Shape: [%d, %d]\n", (int) task_tokens.size(), ctx->n_embd);
            fprintf(stdout, "First 10: [");
            for (int i = 0; i < 10 && i < (int) lang_embeds.size(); ++i) fprintf(stdout, "%.6f%s", lang_embeds[i], i < 9 ? ", " : "");
            fprintf(stdout, "]\n");
            fprintf(stdout, "Last 10: [");
            for (int i = std::max(0, (int) lang_embeds.size() - 10); i < (int) lang_embeds.size(); ++i) {
                fprintf(stdout, "%.6f%s", lang_embeds[i], i < (int) lang_embeds.size() - 1 ? ", " : "");
            }
            fprintf(stdout, "]\n");
        }

        //=====1.4 Concatenate embeddings → prefix
        const int n_vis = (int) ctx->vision_emb.size() / ctx->n_embd;
        const int n_task = (int) task_tokens.size();
        const int n_state = (!ctx->state_emb.empty() && (int) ctx->state_emb.size() == ctx->n_embd) ? 1 : 0;
        const int n_total = n_vis + n_task + n_state;
        if (n_vis <= 0 || n_total <= 0) {
            fprintf(stderr, "[SmolVLA] Error: invalid merged token count\n");
            return result;
        }

        // 从这里开始，主要是处理pad，position id，attention mask相关设置
        // Build pad_mask to identify valid (non-padded) tokens.
        // Matches Python: pad_mask = [cam1_valid] + [empty_cam_pad] + ... + [lang_valid+pad] + [state]
        // TODO: 这里相机设置还是写死了，之后要fix
        const int n_vis_per_cam = n_vis / (1 + ctx->empty_cameras);
        std::vector<bool> pad_mask(n_total, false);
        for (int i = 0; i < n_vis_per_cam; ++i) pad_mask[i] = true; //vis
        for (int i = 0; i < n_raw; ++i) pad_mask[n_vis + i] = true; //lang 
        if (n_state == 1) pad_mask[n_total - 1] = true; //state
        ctx->prefix_valid_mask = pad_mask; 

        // Compute position_ids matching Python: cumsum(pad_mask) - 1
        std::vector<int> position_ids(n_total);
        int pos_cumsum = 0;
        for (int i = 0; i < n_total; ++i) {
            if (pad_mask[i]) pos_cumsum++;
            position_ids[i] = pos_cumsum - 1;
        }
        if (ctx->verbosity >= 1) {
            int n_valid = pos_cumsum;
            fprintf(stdout, "[SmolVLA] pad_mask: %d valid / %d total, position range [0, %d]\n",
                    n_valid, n_total, position_ids[n_total - 1]);
        }

        std::vector<float> merged_embeds((size_t) n_total * ctx->n_embd, 0.0f);
        int offset = 0;
        memcpy(merged_embeds.data() + (size_t) offset * ctx->n_embd, ctx->vision_emb.data(), (size_t) n_vis * ctx->n_embd * sizeof(float));
        offset += n_vis;
        memcpy(merged_embeds.data() + (size_t) offset * ctx->n_embd,
               lang_embeds.data(),
               (size_t) n_task * ctx->n_embd * sizeof(float));
        offset += n_task;
        if (n_state == 1) {
            memcpy(merged_embeds.data() + (size_t) offset * ctx->n_embd, ctx->state_emb.data(), (size_t) ctx->n_embd * sizeof(float));
        }
        
        if (llm_debug) {
            fprintf(stdout, "\n=== LLM Input Embeddings (Merged) ===\n");
            fprintf(stdout, "Shape: [%d, %d]\n", n_total, ctx->n_embd);
            fprintf(stdout, "First 10: [");
            for (int i = 0; i < 10 && i < (int) merged_embeds.size(); ++i) fprintf(stdout, "%.6f%s", merged_embeds[i], i < 9 ? ", " : "");
            fprintf(stdout, "]\n");
            fprintf(stdout, "Last 10: [");
            for (int i = std::max(0, (int) merged_embeds.size() - 10); i < (int) merged_embeds.size(); ++i) {
                fprintf(stdout, "%.6f%s", merged_embeds[i], i < (int) merged_embeds.size() - 1 ? ", " : "");
            }
            fprintf(stdout, "]\n");
        }

        llama_kv_cache_clear(ctx->ctx_llama);
        llama_set_embeddings(ctx->ctx_llama, llm_debug);
        // Prefix-LM attention mask via seq_ids, with padding mask:
        //   valid lang+vision tokens:  seq_id={0,1} — bidirectional among valid prefix
        //   padded tokens:        seq_id={2}   - 谁也看不到，但它也看不到谁
        //   state token:          seq_id={1}   — can see valid prefix, prefix can't see it
        llama_set_causal_attn(ctx->ctx_llama, false);
        llama_batch batch = llama_batch_init(n_total, ctx->n_embd, 2);
        batch.n_tokens = n_total;
        memcpy(batch.embd, merged_embeds.data(), (size_t) n_total * ctx->n_embd * sizeof(float));
        int pad_position = 0;
        for (int i = 0; i < n_total; ++i) {
            batch.logits[i] = llm_debug ? 1 : 0;
            if (!pad_mask[i]) {
                // padded tokens: seq_id={2}, invisible to valid tokens
                batch.pos[i] = pad_position++;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 2;
            } else if (i < n_total - 1) {
                // valid prefix tokens: seq_id={0,1}
                batch.pos[i] = position_ids[i];
                batch.n_seq_id[i] = 2;
                batch.seq_id[i][0] = 0;
                batch.seq_id[i][1] = 1;
            } else {
                // state token: seq_id={1}
                batch.pos[i] = position_ids[i];
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 1;
            }
        }

        //======1.5 Run VLM forward → KV Cache
        auto t_llm0 = std::chrono::high_resolution_clock::now();
        if (llama_decode(ctx->ctx_llama, batch)) {
            fprintf(stderr, "[SmolVLA] Error: llama_decode failed on merged embeddings\n");
            llama_batch_free(batch);
            llama_set_causal_attn(ctx->ctx_llama, true);
            llama_set_embeddings(ctx->ctx_llama, false);
            return result;
        }
        auto t_llm1 = std::chrono::high_resolution_clock::now();
        if (ctx->verbosity >= 1) {
            fprintf(stderr, "[SmolVLA] llm merged decode: %.2f ms (%d tokens)\n",
                    std::chrono::duration<double, std::milli>(t_llm1 - t_llm0).count(), n_total);
        }

        if (llm_debug) {
            float * h_first = llama_get_embeddings_ith(ctx->ctx_llama, 0);
            float * h_last = llama_get_embeddings_ith(ctx->ctx_llama, n_total - 1);
            if (!h_first || !h_last) {
                fprintf(stderr, "[SmolVLA] Error: failed to fetch token hidden states\n");
                llama_batch_free(batch);
                llama_set_embeddings(ctx->ctx_llama, false);
                return result;
            }

            ctx->llm_last_hidden.assign(h_last, h_last + ctx->n_embd);

            fprintf(stdout, "\n=== LLM Output (Flatten) ===\n");
            fprintf(stdout, "Shape: [%d, %d]\n", n_total, ctx->n_embd);
            fprintf(stdout, "First 10: [");
            for (int i = 0; i < 10 && i < ctx->n_embd; ++i) {
                fprintf(stdout, "%.6f%s", h_first[i], i < 9 ? ", " : "");
            }
            fprintf(stdout, "]\n");
            fprintf(stdout, "Last 10: [");
            const int tail_start = std::max(0, ctx->n_embd - 10);
            for (int i = tail_start; i < ctx->n_embd; ++i) {
                fprintf(stdout, "%.6f%s", h_last[i], i < ctx->n_embd - 1 ? ", " : "");
            }
            fprintf(stdout, "]\n");
            fprintf(stdout, "\n--- Per-Row LLM Final Output ---\n");
            for (int i = 0; i < n_total; ++i) {
                float * h_i = llama_get_embeddings_ith(ctx->ctx_llama, i);
                if (!h_i) {
                    fprintf(stdout, "  Row %d: <missing>\n", i);
                    continue;
                }
                fprintf(stdout, "  Row %d: [", i);
                for (int j = 0; j < ctx->n_embd; ++j) {
                    fprintf(stdout, "%.6f%s", h_i[j], j < ctx->n_embd - 1 ? ", " : "");
                }
                fprintf(stdout, "]\n");
            }
        } else {
            ctx->llm_last_hidden.clear();
        }

        llama_batch_free(batch);
        llama_set_causal_attn(ctx->ctx_llama, true);
        llama_set_embeddings(ctx->ctx_llama, false);
        ctx->last_timings.vlm_ms = smolvla_elapsed_ms(t_vlm0, smolvla_clock::now());
    }

    // Extract VLM KV cache for Action Expert
    const auto t_kv0 = smolvla_clock::now();
    if (!extract_vlm_kv_cache(ctx)) {
        fprintf(stderr, "[SmolVLA] FATAL: KV cache extraction failed\n");
        return result;
    }
    ctx->last_timings.kv_extract_ms = smolvla_elapsed_ms(t_kv0, smolvla_clock::now());

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] Phase 2: Denoising loop (%d steps)\n", ctx->num_steps);
    }

    const auto t_phase20 = smolvla_clock::now();
    {
        const int chunk = ctx->expert->chunk_size;
        const int padded_action_dim = ctx->expert->max_action_dim;
        const int hidden = ctx->expert->hidden_size;
        const float dt = -1.0f / (float) ctx->num_steps;

        std::vector<uint8_t> prefix_valid_mask_u8(ctx->prefix_valid_mask.size(), 1);
        for (size_t i = 0; i < ctx->prefix_valid_mask.size(); ++i) {
            prefix_valid_mask_u8[i] = (uint8_t) (ctx->prefix_valid_mask[i] ? 1 : 0);
        }
        if (!smolvla_action_expert_prepare_attention_cache(
                ctx->expert,
                prefix_valid_mask_u8.data(),
                ctx->vlm_kv_seq_len)) {
            fprintf(stderr, "[SmolVLA] FATAL: action attention cache preparation failed\n");
            return result;
        }

        std::vector<float> x_t((size_t) chunk * padded_action_dim);
        smolvla_fill_initial_noise(ctx, x_t.data(), chunk, padded_action_dim);

        std::vector<float> suffix_emb((size_t) chunk * hidden);
        std::vector<float> v_t((size_t) chunk * padded_action_dim);
        std::vector<float> x_t_in((size_t) chunk * padded_action_dim);
        for (int step = 0; step < ctx->num_steps; ++step) {
            const float timestep = 1.0f + step * dt;
            x_t_in = x_t;

            if (!smolvla_action_expert_embed_suffix(
                    ctx->expert, ctx->n_threads, x_t.data(), timestep, suffix_emb.data())) {
                fprintf(stderr, "[SmolVLA] FATAL: M6 embed_suffix failed at step %d\n", step);
                return result;
            }
            if (!smolvla_action_expert_eval_transformer_project_velocity(
                           ctx->expert,
                           suffix_emb.data(),
                           v_t.data())) {
                fprintf(stderr, "[SmolVLA] FATAL: M6 transformer+velocity failed at step %d\n", step);
                return result;
            }

            for (size_t i = 0; i < x_t.size(); ++i) {
                x_t[i] += dt * v_t[i];
            }

            if (m6_debug) {
                fprintf(stdout, "[M6 step %d] v_t[0,0,:10]:", step);
                for (int j = 0; j < 10; ++j) fprintf(stdout, " %.6f", v_t[j]);
                fprintf(stdout, "\n");
                fprintf(stdout, "[M6 step %d] x_t[0,0,:10]:", step);
                for (int j = 0; j < 10; ++j) fprintf(stdout, " %.6f", x_t[j]);
                fprintf(stdout, "\n");

                if (!dump_m6_step(
                        ctx,
                        step,
                        x_t_in.data(),
                        v_t.data(),
                        x_t.data(),
                        chunk,
                        padded_action_dim)) {
                    fprintf(stderr, "[SmolVLA] FATAL: M6 step dump failed at step %d\n", step);
                    return result;
                }
            }
        }
        // 找到真正的action dim，这里做了一些防御，防止越界
        const int stats_dim = std::min((int) ctx->expert->action_mean.size(), (int) ctx->expert->action_std.size());
        if (stats_dim <= 0) {
            fprintf(stderr, "[SmolVLA] FATAL: M7 missing action unnormalization stats in action expert GGUF\n");
            return result;
        }

        const int final_action_dim = std::min({ctx->action_dim, stats_dim, padded_action_dim});
        if (final_action_dim <= 0) {
            fprintf(stderr, "[SmolVLA] FATAL: M7 invalid final action dim (requested=%d, stats=%d, padded=%d)\n",
                    ctx->action_dim, stats_dim, padded_action_dim);
            return result;
        }

        std::fill(ctx->action_buffer.begin(), ctx->action_buffer.end(), 0.0f);
        for (int t = 0; t < chunk; ++t) {
            const float * raw_row = x_t.data() + (size_t) t * padded_action_dim;
            float * out_row = ctx->action_buffer.data() + (size_t) t * ctx->action_dim;
            for (int d = 0; d < final_action_dim; ++d) {
                out_row[d] = raw_row[d] * ctx->expert->action_std[d] + ctx->expert->action_mean[d];
            }
        }

        if (m7_debug) {
            fprintf(stdout, "[M7] final_actions[0,0,:6]:");
            for (int j = 0; j < std::min(final_action_dim, 6); ++j) {
                fprintf(stdout, " %.6f", ctx->action_buffer[j]);
            }
            fprintf(stdout, "\n");

            if (!dump_m7_actions(ctx, ctx->action_buffer.data(), chunk, final_action_dim)) {
                fprintf(stderr, "[SmolVLA] FATAL: M7 final action dump failed\n");
                return result;
            }
        }

        result.action_dim = final_action_dim;
    }
    ctx->last_timings.phase2_ms = smolvla_elapsed_ms(t_phase20, smolvla_clock::now());

    result.actions    = ctx->action_buffer.data();
    result.chunk_size = ctx->chunk_size;
    if (result.action_dim == 0) {
        result.action_dim = ctx->action_dim;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] Prediction complete (M7 final actions ready)\n");
    }

    return result;
}

struct smolvla_result smolvla_predict(
    struct smolvla_context * ctx,
    const char * image_path,
    const float * state,
    int state_dim)
{
    if (!ctx) {
        return smolvla_empty_result();
    }

    const auto t_total0 = smolvla_clock::now();
    smolvla_reset_last_stage_timings(ctx);

    ctx->vision_emb.clear();
    if (!ctx->vision || !image_path) {
        fprintf(stderr, "[SmolVLA] Error: predict requires a valid image path and vision model\n");
        return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
    }

    const auto t_vision0 = smolvla_clock::now();
    std::vector<float> main_cam_emb = smolvla_vision_encode_file(ctx->vision, image_path, ctx->n_threads);
    ctx->last_timings.vision_ms = smolvla_elapsed_ms(t_vision0, smolvla_clock::now());
    if (!smolvla_prepare_vision_embedding(ctx, main_cam_emb)) {
        fprintf(stderr, "[SmolVLA] Error: vision encode failed for image path input\n");
        return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
    }

    return smolvla_finish_predict(ctx, smolvla_predict_impl(ctx, image_path, state, state_dim), t_total0);
}

struct smolvla_result smolvla_predict_bytes(
    struct smolvla_context * ctx,
    const unsigned char * image_bytes,
    int image_len,
    const float * state,
    int state_dim)
{
    if (!ctx) {
        return smolvla_empty_result();
    }

    const auto t_total0 = smolvla_clock::now();
    smolvla_reset_last_stage_timings(ctx);

    ctx->vision_emb.clear();
    if (!ctx->vision || !image_bytes || image_len <= 0) {
        fprintf(stderr, "[SmolVLA] Error: predict_bytes requires non-empty image bytes and vision model\n");
        return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] predict_bytes: image_len=%d\n", image_len);
    }

    const auto t_vision0 = smolvla_clock::now();
    std::vector<float> main_cam_emb = smolvla_vision_encode_bytes(ctx->vision, image_bytes, image_len, ctx->n_threads);
    ctx->last_timings.vision_ms = smolvla_elapsed_ms(t_vision0, smolvla_clock::now());
    if (!smolvla_prepare_vision_embedding(ctx, main_cam_emb)) {
        fprintf(stderr, "[SmolVLA] Error: vision encode failed for bytes input\n");
        return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
    }

    return smolvla_finish_predict(ctx, smolvla_predict_impl(ctx, "(bytes)", state, state_dim), t_total0);
}

struct smolvla_stage_timings smolvla_get_last_stage_timings(const struct smolvla_context * ctx) {
    if (!ctx) {
        return smolvla_empty_stage_timings();
    }
    return ctx->last_timings;
}

void smolvla_free(struct smolvla_context * ctx) {
    if (!ctx) return;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] Freeing engine context\n");
    }

    // Free state_proj
    if (ctx->state_proj) {
        smolvla_state_proj_free(ctx->state_proj);
    }

    // Free vision
    if (ctx->vision) {
        smolvla_vision_free(ctx->vision);
    }

    // Free action expert
    if (ctx->expert) {
        smolvla_action_expert_free(ctx->expert);
    }

    if (ctx->ctx_llama) 
        llama_free(ctx->ctx_llama);
    if (ctx->vlm) {
        llama_free_model(ctx->vlm);
        llama_backend_free();
    }

    delete ctx;
}
