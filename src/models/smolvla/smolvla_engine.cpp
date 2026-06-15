// smolvla_engine.cpp — SmolVLA inference engine (stub implementation)
//
// SmolVLA Architecture:
//   Phase 1: image + state + task → LLM → KV Cache (once)
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
    llama_model * llm;
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
    int lang_max_len;
    llama_token lang_pad_id;
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

    std::vector<float> lang_pad_emb_cache;
    llama_token lang_pad_emb_cache_id;
    bool lang_pad_emb_cache_ready;

    // LLM KV metadata extracted after Phase 1
    int fixed_prefix_seq_len;
    int llm_kv_seq_len;    // number of prefix tokens in KV cache
    int llm_kv_dim;        // n_kv_heads * head_dim = 320
    std::vector<bool> prefix_valid_mask;

};

using smolvla_clock = std::chrono::steady_clock;

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

static bool smolvla_read_meta_int(const llama_model * model, const char * key, int * out) {
    char buf[32];
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) < 0) {
        fprintf(stderr, "[SmolVLA] Error: missing required GGUF metadata %s\n", key);
        return false;
    }
    *out = std::atoi(buf);
    if (*out <= 0) {
        fprintf(stderr, "[SmolVLA] Error: invalid GGUF metadata %s=%s\n", key, buf);
        return false;
    }
    return true;
}

static bool smolvla_load_language_config(smolvla_context * ctx) {
    if (!smolvla_read_meta_int(ctx->llm, "smolvla.tokenizer_max_length", &ctx->lang_max_len)) {
        return false;
    }
    ctx->lang_pad_id = llama_vocab_pad(llama_model_get_vocab(ctx->llm));
    if (ctx->lang_pad_id == LLAMA_TOKEN_NULL) {
        fprintf(stderr, "[SmolVLA] Error: missing required GGUF metadata tokenizer.ggml.padding_token_id\n");
        return false;
    }
    return true;
}

static bool smolvla_env_has_value(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0';
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

// Extract LLM KV cache from llama.cpp after Phase 1 and prepare the
// expert-side prefix runtime directly from backend tensors.
static bool extract_llm_kv_cache(smolvla_context * ctx) {
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

    ctx->llm_kv_seq_len = seq_len;
    ctx->llm_kv_dim     = kv_dim;
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

    if (tok_embd->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> row_buf(n_embd);
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                fprintf(stderr, "[SmolVLA] Error: token id out of range: %d (n_vocab=%lld)\n",
                        (int) tokens[i], (long long) n_vocab);
                return false;
            }
            const size_t offset = (size_t) tokens[i] * row_stride;
            ggml_backend_tensor_get(tok_embd, row_buf.data(), offset, n_embd * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(row_buf.data(), output + i * n_embd, n_embd);
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

    params.llm_path           = nullptr;
    params.mmproj_path        = nullptr;
    params.state_proj_path    = nullptr;
    params.action_expert_path = nullptr;
    params.n_threads          = 0;  // auto
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
        fprintf(stderr, "LLM:           %s\n", params.llm_path ? params.llm_path : "(none)");
        fprintf(stderr, "Vision:        %s\n", params.mmproj_path ? params.mmproj_path : "(none)");
        fprintf(stderr, "State Proj:    %s\n", params.state_proj_path ? params.state_proj_path : "(none)");
        fprintf(stderr, "Action Expert: %s\n", params.action_expert_path ? params.action_expert_path : "(none)");
        fprintf(stderr, "Threads:       %d\n", params.n_threads);
        fprintf(stderr, "Noise mode:    %s\n", smolvla_noise_mode_name(params.noise_mode));
        fprintf(stderr, "Noise seed:    %s\n", params.noise_seed >= 0 ? "(set)" : "auto");
        if (params.noise_seed >= 0) {
            fprintf(stderr, "Noise seed v:  %lld\n", (long long) params.noise_seed);
        }
        fprintf(stderr, "===========================================\n\n");
    }

    // Allocate context
    smolvla_context * ctx = new smolvla_context();
    ctx->llm       = nullptr;
    ctx->ctx_llama = nullptr;
    ctx->tok_embd  = nullptr;
    ctx->expert    = nullptr;
    ctx->action_dim = 0;
    ctx->chunk_size = 0;
    ctx->num_steps  = 0;
    ctx->n_threads  = params.n_threads > 0 ? params.n_threads : 4;
    ctx->n_batch    = params.n_batch > 0 ? params.n_batch : 512;
    ctx->n_ctx      = params.n_ctx > 0 ? params.n_ctx : 2048;
    ctx->n_embd     = 0;
    ctx->lang_max_len = 0;
    ctx->lang_pad_id = LLAMA_TOKEN_NULL;
    ctx->noise_mode = params.noise_mode;
    if (ctx->noise_mode != SMOLVLA_NOISE_MODE_GAUSSIAN &&
        ctx->noise_mode != SMOLVLA_NOISE_MODE_DEBUG_SIN) {
        fprintf(stderr, "[SmolVLA] Warning: invalid noise_mode=%d, falling back to gaussian\n", ctx->noise_mode);
        ctx->noise_mode = SMOLVLA_NOISE_MODE_GAUSSIAN;
    }
    ctx->noise_seed = params.noise_seed;
    ctx->verbosity  = params.verbosity;
    ctx->fixed_prefix_seq_len = -1;
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
        }
    }

    // Load LLM model via llama.cpp (reuses standard llama runtime)
    if (params.llm_path) {
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        ctx->llm = llama_load_model_from_file(params.llm_path, model_params);
        if (!ctx->llm) {
            fprintf(stderr, "[SmolVLA] Error: failed to load LLM from %s\n", params.llm_path);
            smolvla_free(ctx);
            return nullptr;
        }
        if (!smolvla_load_language_config(ctx)) {
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

        ctx->ctx_llama = llama_new_context_with_model(ctx->llm, cparams);
        if (!ctx->ctx_llama) {
            fprintf(stderr, "[SmolVLA] Error: failed to create llama context\n");
            smolvla_free(ctx);
            return nullptr;
        }

        ctx->n_embd = llama_n_embd(ctx->llm);
        ctx->tok_embd = llama_get_model_tensor(ctx->llm, "token_embd.weight");
        if (!ctx->tok_embd) {
            fprintf(stderr, "[SmolVLA] Error: failed to get token_embd.weight\n");
            smolvla_free(ctx);
            return nullptr;
        }
        if (ctx->verbosity >= 1) {
            fprintf(stderr, "[SmolVLA] LLM loaded: n_embd=%d, lang_max_len=%d, lang_pad_id=%d, token_embd(type=%d shape=[%d, %d])\n",
                    ctx->n_embd, ctx->lang_max_len, (int) ctx->lang_pad_id,
                    ctx->tok_embd->type, (int) ctx->tok_embd->ne[0], (int) ctx->tok_embd->ne[1]);
        }
    }

    // Load Action Expert
    if (params.action_expert_path) {
        ctx->expert = smolvla_action_expert_load(params.action_expert_path, params.verbosity);
        if (!ctx->expert) {
            fprintf(stderr, "[SmolVLA] Warning: Failed to load action expert, continuing without\n");
        } else {
            ctx->action_dim = ctx->expert->action_dim;
            ctx->chunk_size = ctx->expert->chunk_size;
            ctx->num_steps = ctx->expert->num_steps;
            if (ctx->action_dim <= 0 || ctx->chunk_size <= 0 || ctx->num_steps <= 0) {
                fprintf(stderr,
                        "[SmolVLA] Error: invalid action expert shape from GGUF "
                        "(action_dim=%d chunk_size=%d num_steps=%d)\n",
                        ctx->action_dim, ctx->chunk_size, ctx->num_steps);
                smolvla_free(ctx);
                return nullptr;
            }
            ctx->action_buffer.resize((size_t) ctx->chunk_size * (size_t) ctx->action_dim, 0.0f);
        }
    }

    if (ctx->vision && ctx->expert) {
        const int n_vis = smolvla_vision_n_tokens(ctx->vision);
        const int n_state = !ctx->state_emb.empty() ? 1 : 0;
        ctx->fixed_prefix_seq_len = n_vis + ctx->lang_max_len + n_state;
        if (!smolvla_action_expert_init_fixed_prefix_runtime(ctx->expert, ctx->fixed_prefix_seq_len)) {
            fprintf(stderr, "[SmolVLA] Error: failed to initialize fixed prefix runtime (seq_len=%d)\n",
                    ctx->fixed_prefix_seq_len);
            smolvla_free(ctx);
            return nullptr;
        }
        if (ctx->verbosity >= 1) {
            fprintf(stderr, "[SmolVLA] Fixed prefix runtime initialized: vision=%d lang=%d state=%d total=%d\n",
                    n_vis, ctx->lang_max_len, n_state, ctx->fixed_prefix_seq_len);
        }
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] Engine initialized%s\n",
                ctx->expert ? "" : " (no action expert — Phase 2 will be stub)");
        fprintf(stderr, "[SmolVLA] Action buffer: %d x %d = %d floats\n",
                ctx->chunk_size, ctx->action_dim, ctx->chunk_size * ctx->action_dim);
    }

    return ctx;
}

// Prepare camera vision embeddings for prefix assembly.
static bool smolvla_prepare_vision_embeddings(
    struct smolvla_context * ctx,
    const std::vector<std::vector<float>> & camera_embs
) {
    if (!ctx || !ctx->vision || camera_embs.empty()) {
        if (ctx) {
            ctx->vision_emb.clear();
        }
        return false;
    }

    const int n_tokens_per_cam = smolvla_vision_n_tokens(ctx->vision);
    const int proj_dim = ctx->vision->proj_dim;
    const size_t one_cam_size = (size_t) n_tokens_per_cam * proj_dim;
    for (size_t i = 0; i < camera_embs.size(); ++i) {
        if (camera_embs[i].size() != one_cam_size) {
            fprintf(stderr, "[SmolVLA] Error: vision output size mismatch for camera %zu (%zu vs %zu)\n",
                    i, camera_embs[i].size(), one_cam_size);
            ctx->vision_emb.clear();
            return false;
        }
    }

    ctx->vision_emb.clear();
    ctx->vision_emb.reserve(one_cam_size * camera_embs.size());
    for (const std::vector<float> & camera_emb : camera_embs) {
        ctx->vision_emb.insert(ctx->vision_emb.end(), camera_emb.begin(), camera_emb.end());
    }

    // Match SmolVLA Python: img_emb = img_emb * sqrt(hidden_size)
    const float vision_scale = std::sqrt((float) proj_dim);
    for (float & v : ctx->vision_emb) {
        v *= vision_scale;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stdout, "[SmolVLA] vision tokens: cameras=%zu tokens_per_camera=%d total=%zu\n",
                camera_embs.size(),
                n_tokens_per_cam,
                camera_embs.size() * (size_t) n_tokens_per_cam);
        fprintf(stdout, "[SmolVLA] vision output shape: [%zu, %d]\n",
                camera_embs.size() * (size_t) n_tokens_per_cam,
                proj_dim);
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

static bool smolvla_prepare_vision_embedding(
    struct smolvla_context * ctx,
    const std::vector<float> & main_cam_emb
) {
    std::vector<std::vector<float>> camera_embs;
    camera_embs.push_back(main_cam_emb);
    return smolvla_prepare_vision_embeddings(ctx, camera_embs);
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
    int state_dim,
    const char * task)
{
    struct smolvla_result result = smolvla_empty_result();

    if (!ctx) {
        fprintf(stderr, "[SmolVLA] Error: null context\n");
        return result;
    }

    const bool m7_debug = smolvla_env_has_value("SMOLVLA_M7_DUMP_DIR");

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
        fprintf(stdout, "Task: %s\n", task ? task : "");
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
        const auto t_llm_start = smolvla_clock::now();
        // ====1.3 Tokenize task and embed
        // Match SmolVLA preprocessor behavior: ensure trailing newline before tokenization.
        std::string task_text = task && task[0] != '\0' ? task : "grab the block.";
        if (task_text.empty() || task_text.back() != '\n') {
            task_text.push_back('\n');
        }

        std::vector<llama_token> task_tokens_raw = smolvla_tokenize(ctx->llm, task_text, true, true);
        if (task_tokens_raw.empty()) {
            fprintf(stderr, "[SmolVLA] Error: task tokenization failed\n");
            return result;
        }
        std::vector<llama_token> task_tokens = smolvla_pad_tokens_right(task_tokens_raw, ctx->lang_max_len, ctx->lang_pad_id);

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
            if (!ctx->lang_pad_emb_cache_ready || ctx->lang_pad_emb_cache_id != ctx->lang_pad_id) {
                ctx->lang_pad_emb_cache.resize(ctx->n_embd);
                // TODO: 感觉其实这里也可以扔到load阶段去算，然后在predict的时候直接拼padding就行了
                if (!lookup_token_embeddings(ctx->tok_embd, &ctx->lang_pad_id, 1, ctx->n_embd, ctx->lang_pad_emb_cache.data())) {
                    fprintf(stderr, "[SmolVLA] Error: pad token embedding lookup failed\n");
                    return result;
                }
                ctx->lang_pad_emb_cache_id = ctx->lang_pad_id;
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

        //=====1.4 Concatenate embeddings → prefix
        const int n_vis = (int) ctx->vision_emb.size() / ctx->n_embd;
        const int n_task = (int) task_tokens.size();
        const int n_state = (!ctx->state_emb.empty() && (int) ctx->state_emb.size() == ctx->n_embd) ? 1 : 0;
        const int n_total = n_vis + n_task + n_state;
        if (n_vis <= 0 || n_total <= 0) {
            fprintf(stderr, "[SmolVLA] Error: invalid merged token count\n");
            return result;
        }
        if (n_total > ctx->n_ctx) {
            fprintf(stderr, "[SmolVLA] Error: prefix length %d exceeds n_ctx=%d\n", n_total, ctx->n_ctx);
            return result;
        }
        ctx->fixed_prefix_seq_len = n_total;
        if (ctx->expert &&
            !smolvla_action_expert_init_fixed_prefix_runtime(ctx->expert, ctx->fixed_prefix_seq_len)) {
            fprintf(stderr, "[SmolVLA] Error: failed to prepare fixed prefix runtime (seq_len=%d)\n",
                    ctx->fixed_prefix_seq_len);
            return result;
        }

        // 从这里开始，主要是处理pad，position id，attention mask相关设置
        // Build pad_mask to identify valid (non-padded) tokens.
        // Layout: [real_camera_tokens] + [language raw + language pad] + [state].
        std::vector<bool> pad_mask(n_total, false);
        for (int i = 0; i < n_vis; ++i) pad_mask[i] = true; // vision
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
        
        llama_kv_cache_clear(ctx->ctx_llama);
        llama_set_embeddings(ctx->ctx_llama, false);
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
            batch.logits[i] = 0;
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

        //======1.5 Run LLM forward → KV Cache
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

        llama_batch_free(batch);
        llama_set_causal_attn(ctx->ctx_llama, true);
        llama_set_embeddings(ctx->ctx_llama, false);
        llama_synchronize(ctx->ctx_llama);
        ctx->last_timings.llm_ms = smolvla_elapsed_ms(t_llm_start, smolvla_clock::now());
    }

    if (!ctx->expert) {
        fprintf(stderr, "[SmolVLA] FATAL: missing action expert for prediction\n");
        return result;
    }

    // Extract LLM KV cache for Action Expert
    const auto t_kv0 = smolvla_clock::now();
    if (!extract_llm_kv_cache(ctx)) {
        fprintf(stderr, "[SmolVLA] FATAL: KV cache extraction failed\n");
        return result;
    }
    ctx->last_timings.kv_extract_ms = smolvla_elapsed_ms(t_kv0, smolvla_clock::now());

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] Phase 2: Denoising loop (%d steps)\n", ctx->num_steps);
    }

    const auto t_phase20 = smolvla_clock::now();
    {
        const int chunk = ctx->chunk_size;
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
                ctx->llm_kv_seq_len)) {
            fprintf(stderr, "[SmolVLA] FATAL: action attention cache preparation failed\n");
            return result;
        }

        std::vector<float> x_t((size_t) chunk * padded_action_dim);
        smolvla_fill_initial_noise(ctx, x_t.data(), chunk, padded_action_dim);

        std::vector<float> suffix_emb((size_t) chunk * hidden);
        std::vector<float> v_t((size_t) chunk * padded_action_dim);
        for (int step = 0; step < ctx->num_steps; ++step) {
            const float timestep = 1.0f + step * dt;

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
    int state_dim,
    const char * task)
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

    return smolvla_finish_predict(ctx, smolvla_predict_impl(ctx, image_path, state, state_dim, task), t_total0);
}

struct smolvla_result smolvla_predict_bytes(
    struct smolvla_context * ctx,
    const unsigned char * image_bytes,
    int image_len,
    const float * state,
    int state_dim,
    const char * task)
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

    return smolvla_finish_predict(ctx, smolvla_predict_impl(ctx, "(bytes)", state, state_dim, task), t_total0);
}

struct smolvla_result smolvla_predict_raw_rgb(
    struct smolvla_context * ctx,
    const unsigned char * rgb,
    int width,
    int height,
    int channels,
    int stride_bytes,
    const float * state,
    int state_dim,
    const char * task)
{
    smolvla_image_view image{};
    image.name = "image";
    image.data = rgb;
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.stride_bytes = stride_bytes;
    return smolvla_predict_raw_rgb_batch(ctx, &image, 1, state, state_dim, task);
}

struct smolvla_result smolvla_predict_raw_rgb_batch(
    struct smolvla_context * ctx,
    const struct smolvla_image_view * images,
    size_t image_count,
    const float * state,
    int state_dim,
    const char * task)
{
    if (!ctx) {
        return smolvla_empty_result();
    }

    const auto t_total0 = smolvla_clock::now();
    smolvla_reset_last_stage_timings(ctx);

    ctx->vision_emb.clear();
    if (!ctx->vision || !images || image_count == 0) {
        fprintf(stderr,
            "[SmolVLA] Error: predict_raw_rgb_batch requires at least one RGB/HWC/uint8 image and vision model "
            "(images=%p image_count=%zu)\n",
            (const void *) images, image_count);
        return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "[SmolVLA] predict_raw_rgb_batch: image_count=%zu\n", image_count);
    }

    const auto t_vision0 = smolvla_clock::now();
    std::vector<std::vector<float>> camera_embs;
    camera_embs.reserve(image_count);
    for (size_t i = 0; i < image_count; ++i) {
        const smolvla_image_view & image = images[i];
        if (!image.data || image.width <= 0 || image.height <= 0 || image.channels != 3) {
            fprintf(stderr,
                "[SmolVLA] Error: image %zu must be RGB/HWC/uint8 "
                "(name=%s data=%p width=%d height=%d channels=%d)\n",
                i,
                image.name ? image.name : "(unnamed)",
                (const void *) image.data,
                image.width,
                image.height,
                image.channels);
            ctx->last_timings.vision_ms = smolvla_elapsed_ms(t_vision0, smolvla_clock::now());
            return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
        }

        int stride_bytes = image.stride_bytes;
        if (stride_bytes <= 0) {
            stride_bytes = image.width * image.channels;
        }
        if (stride_bytes < image.width * image.channels) {
            fprintf(stderr,
                "[SmolVLA] Error: image %zu invalid stride_bytes=%d for width=%d channels=%d\n",
                i, stride_bytes, image.width, image.channels);
            ctx->last_timings.vision_ms = smolvla_elapsed_ms(t_vision0, smolvla_clock::now());
            return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
        }

        if (ctx->verbosity >= 1) {
            fprintf(stderr,
                "[SmolVLA] image[%zu]: name=%s width=%d height=%d channels=%d stride_bytes=%d\n",
                i,
                image.name ? image.name : "(unnamed)",
                image.width,
                image.height,
                image.channels,
                stride_bytes);
        }

        camera_embs.push_back(smolvla_vision_encode_raw(
            ctx->vision,
            image.data,
            image.width,
            image.height,
            image.channels,
            stride_bytes,
            ctx->n_threads));
    }
    ctx->last_timings.vision_ms = smolvla_elapsed_ms(t_vision0, smolvla_clock::now());
    if (!smolvla_prepare_vision_embeddings(ctx, camera_embs)) {
        fprintf(stderr, "[SmolVLA] Error: vision encode failed for raw RGB batch input\n");
        return smolvla_finish_predict(ctx, smolvla_empty_result(), t_total0);
    }

    return smolvla_finish_predict(ctx, smolvla_predict_impl(ctx, "(raw-rgb-batch)", state, state_dim, task), t_total0);
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
    if (ctx->llm) {
        llama_free_model(ctx->llm);
        llama_backend_free();
    }

    delete ctx;
}
