/**
 * SmolVLA Action Expert — GGUF loader + tensor management
 *
 * Follows the same load pattern as state_proj.cpp:
 *   1. Open GGUF, read metadata
 *   2. Allocate backend buffer, load tensors
 *   3. Resolve named tensor pointers into struct fields
 */

#include "action_expert.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "models/ggml_backend.h"
#include "models/gguf_loader.h"
#include "smolvla_compat.h"

#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <climits>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>

// ============================================================================
// Logging
// ============================================================================

#define LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)

// ============================================================================
// Forward declarations (for use in load)
// ============================================================================
static ggml_cgraph * embed_suffix_build_graph(smolvla_action_expert * ctx);
static void precompute_time_embedding_cache(smolvla_action_expert * ctx);
static struct ggml_tensor * build_project_velocity_op(
    smolvla_action_expert * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * hidden_in);
static int build_self_attn_mask_and_positions(
    const uint8_t * prefix_valid_mask,
    int prefix_seq_len,
    int suffix_len,
    int32_t * position_ids,
    std::vector<float> & attention_mask_f32);
static void build_cross_attn_mask_and_positions(
    const uint8_t * prefix_valid_mask,
    int prefix_seq_len,
    int suffix_len,
    int32_t * position_ids,
    std::vector<float> & attention_mask_f32);

// ============================================================================
// Helpers (shared pattern with state_proj.cpp)
// ============================================================================

static std::string fmt(const char * f, ...) {
    va_list ap, ap2;
    va_start(ap, f);
    va_copy(ap2, ap);
    int n = vsnprintf(nullptr, 0, f, ap);
    GGML_ASSERT(n >= 0 && n < INT_MAX);
    std::vector<char> buf(n + 1);
    vsnprintf(buf.data(), n + 1, f, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), n);
}

static bool smolvla_action_expert_use_accel_backend() {
    return robotcpp_backend_use_accel_from_env(false);
}

static void smolvla_free_model_buffers(std::vector<ggml_backend_buffer_t> & bufs) {
    for (ggml_backend_buffer_t buf : bufs) {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
    }
    bufs.clear();
}

static void smolvla_free_model_contexts(std::vector<ggml_context *> & ctxs) {
    for (ggml_context * ctx : ctxs) {
        if (ctx) {
            ggml_free(ctx);
        }
    }
    ctxs.clear();
}

static size_t smolvla_graph_meta_size(size_t max_nodes) {
    return max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false);
}

struct smolvla_transformer_graph_state {
    std::vector<uint8_t> meta_buf;
    struct ggml_context * ctx_graph = nullptr;
    struct ggml_cgraph * graph = nullptr;

    struct ggml_tensor * inp_hidden = nullptr;
    std::vector<struct ggml_tensor *> inp_prefix_k;
    std::vector<struct ggml_tensor *> inp_prefix_v;
    struct ggml_tensor * out = nullptr;
};

static void clear_transformer_graph(smolvla_transformer_graph_state & graph_state) {
    if (graph_state.ctx_graph) {
        ggml_free(graph_state.ctx_graph);
    }
    graph_state = smolvla_transformer_graph_state();
}

static void smolvla_free_scheduler_backends(std::vector<ggml_backend_t> & backends) {
    for (ggml_backend_t backend : backends) {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
    backends.clear();
}

static void clear_prefix_kv_runtime(smolvla_action_expert * ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->prefix_kv_runtime.buffer) {
        ggml_backend_buffer_free(ctx->prefix_kv_runtime.buffer);
    }
    if (ctx->prefix_kv_runtime.ctx_data) {
        ggml_free(ctx->prefix_kv_runtime.ctx_data);
    }
    ctx->prefix_kv_runtime = smolvla_action_expert::prefix_kv_runtime_t();
}

static void clear_attention_runtime(smolvla_action_expert * ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->attention_runtime.buffer) {
        ggml_backend_buffer_free(ctx->attention_runtime.buffer);
    }
    if (ctx->attention_runtime.ctx_data) {
        ggml_free(ctx->attention_runtime.ctx_data);
    }
    ctx->attention_runtime = smolvla_action_expert::attention_runtime_t();
}

static bool init_attention_runtime(smolvla_action_expert * ctx);

static bool ensure_prefix_kv_runtime(
    smolvla_action_expert * ctx,
    int prefix_seq_len) {
    if (!ctx || prefix_seq_len <= 0) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (ctx->prefix_kv_runtime.prefix_seq_len == prefix_seq_len &&
        ctx->prefix_kv_runtime.ctx_data != nullptr &&
        ctx->prefix_kv_runtime.buffer != nullptr) {
        return true;
    }

    clear_prefix_kv_runtime(ctx);

    const size_t tensor_count = 2 * (size_t) ctx->num_layers + 1;
    const size_t ctx_size = ggml_tensor_overhead() * tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx->prefix_kv_runtime.ctx_data = ggml_init(params);
    if (!ctx->prefix_kv_runtime.ctx_data) {
        LOG_ERR("%s: failed to init prefix KV ctx\n", __func__);
        clear_prefix_kv_runtime(ctx);
        return false;
    }

    ctx->prefix_kv_runtime.k_layers.resize(ctx->num_layers, nullptr);
    ctx->prefix_kv_runtime.v_layers.resize(ctx->num_layers, nullptr);
    for (int layer_idx = 0; layer_idx < ctx->num_layers; ++layer_idx) {
        const bool is_cross = ctx->layers[layer_idx].is_cross_attn;
        const int prefix_dim = is_cross ? ctx->llm_kv_dim : ctx->n_kv_heads * ctx->head_dim;

        ctx->prefix_kv_runtime.k_layers[layer_idx] = ggml_new_tensor_2d(
            ctx->prefix_kv_runtime.ctx_data, GGML_TYPE_F32, prefix_dim, prefix_seq_len);
        ctx->prefix_kv_runtime.v_layers[layer_idx] = ggml_new_tensor_2d(
            ctx->prefix_kv_runtime.ctx_data, GGML_TYPE_F32, prefix_dim, prefix_seq_len);

        if (!ctx->prefix_kv_runtime.k_layers[layer_idx] || !ctx->prefix_kv_runtime.v_layers[layer_idx]) {
            LOG_ERR("%s: failed to create prefix KV tensors for layer %d\n", __func__, layer_idx);
            clear_prefix_kv_runtime(ctx);
            return false;
        }

        ggml_format_name(ctx->prefix_kv_runtime.k_layers[layer_idx], "prefix_k_%d", layer_idx);
        ggml_format_name(ctx->prefix_kv_runtime.v_layers[layer_idx], "prefix_v_%d", layer_idx);
    }

    ctx->prefix_kv_runtime.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx->prefix_kv_runtime.ctx_data, ctx->buft_policy.runtime_buft);
    if (!ctx->prefix_kv_runtime.buffer) {
        LOG_ERR("%s: failed to allocate prefix KV backend buffer\n", __func__);
        clear_prefix_kv_runtime(ctx);
        return false;
    }

    ctx->prefix_kv_runtime.prefix_seq_len = prefix_seq_len;
    return true;
}

class smolvla_action_expert_loader : public gguf_loader {
public:
    smolvla_action_expert_loader(
        smolvla_action_expert * ctx,
        int verbosity)
        : ctx_(ctx),
          verbosity_(verbosity) {
    }

protected:
    bool parse_metadata(gguf_context * gguf) override {
        ctx_->hidden_size       = (int) this->u32_or(gguf, "smolvla.expert.hidden_size", 720);
        ctx_->intermediate_size = (int) this->u32_or(gguf, "smolvla.expert.intermediate_size", 2048);
        ctx_->num_layers        = (int) this->u32_or(gguf, "smolvla.expert.num_layers", 16);
        ctx_->head_dim          = (int) this->require_u32(gguf, "smolvla.expert.head_dim");
        ctx_->n_q_heads         = (int) this->require_u32(gguf, "smolvla.expert.num_attention_heads");
        ctx_->n_kv_heads        = (int) this->require_u32(gguf, "smolvla.expert.num_key_value_heads");
        ctx_->max_action_dim    = (int) this->require_u32(gguf, "smolvla.action_dim");
        ctx_->chunk_size        = (int) this->require_u32(gguf, "smolvla.chunk_size");
        ctx_->num_steps         = (int) this->require_u32(gguf, "smolvla.num_steps");
        ctx_->self_attn_every_n = (int) this->u32_or(gguf, "smolvla.self_attn_every_n_layers", 2);
        ctx_->min_period        = this->f32_or(gguf, "smolvla.min_period", 0.004f);
        ctx_->max_period        = this->f32_or(gguf, "smolvla.max_period", 4.0f);

        is_cross_.assign(ctx_->num_layers, false);
        const int cross_n = this->arr_n(gguf, "smolvla.expert.cross_attn_layers");
        const int32_t * cross_data = static_cast<const int32_t *>(
            this->arr_data(gguf, "smolvla.expert.cross_attn_layers"));
        if (cross_n >= 0 && cross_data) {
            for (int i = 0; i < cross_n; ++i) {
                const int layer_id = (int) cross_data[i];
                if (layer_id >= 0 && layer_id < ctx_->num_layers) {
                    is_cross_[layer_id] = true;
                }
            }
        } else {
            for (int i = 0; i < ctx_->num_layers; ++i) {
                if (ctx_->self_attn_every_n > 0 && i % ctx_->self_attn_every_n != 0) {
                    is_cross_[i] = true;
                }
            }
        }

        if (verbosity_ >= 1) {
            LOG_INF("%s: expert config: hidden=%d, intermediate=%d, layers=%d\n",
                    __func__, ctx_->hidden_size, ctx_->intermediate_size, ctx_->num_layers);
            LOG_INF("%s: action_dim=%d, chunk_size=%d, num_steps=%d\n",
                    __func__, ctx_->max_action_dim, ctx_->chunk_size, ctx_->num_steps);
            LOG_INF("%s: self_attn_every_n=%d, min_period=%.4f, max_period=%.1f\n",
                    __func__, ctx_->self_attn_every_n, ctx_->min_period, ctx_->max_period);
            std::string cross_str;
            for (int i = 0; i < ctx_->num_layers; ++i) {
                if (is_cross_[i]) {
                    if (!cross_str.empty()) cross_str += ",";
                    cross_str += std::to_string(i);
                }
            }
            LOG_INF("%s: cross_attn_layers: [%s]\n", __func__, cross_str.c_str());
        }

        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        ctx_->action_in_proj_w  = this->require_tensor(ctx_data, "smolvla.action_in_proj.weight");
        ctx_->action_in_proj_b  = this->require_tensor(ctx_data, "smolvla.action_in_proj.bias");
        ctx_->action_out_proj_w = this->require_tensor(ctx_data, "smolvla.action_out_proj.weight");
        ctx_->action_out_proj_b = this->require_tensor(ctx_data, "smolvla.action_out_proj.bias");

        ctx_->time_mlp_in_w  = this->require_tensor(ctx_data, "smolvla.time_mlp.0.weight");
        ctx_->time_mlp_in_b  = this->require_tensor(ctx_data, "smolvla.time_mlp.0.bias");
        ctx_->time_mlp_out_w = this->require_tensor(ctx_data, "smolvla.time_mlp.2.weight");
        ctx_->time_mlp_out_b = this->require_tensor(ctx_data, "smolvla.time_mlp.2.bias");

        ctx_->final_norm = this->require_tensor(ctx_data, "smolvla.lm_expert.norm.weight");

        ctx_->layers.resize(ctx_->num_layers);
        for (int i = 0; i < ctx_->num_layers; ++i) {
            smolvla_expert_layer & L = ctx_->layers[i];
            const std::string pfx = fmt("smolvla.expert.blk.%d.", i);

            L.attn_norm    = this->require_tensor(ctx_data, pfx + "attn_norm.weight");
            L.attn_q       = this->require_tensor(ctx_data, pfx + "attn_q.weight");
            L.attn_k       = this->require_tensor(ctx_data, pfx + "attn_k.weight");
            L.attn_v       = this->require_tensor(ctx_data, pfx + "attn_v.weight");
            L.attn_output  = this->require_tensor(ctx_data, pfx + "attn_output.weight");
            L.ffn_norm     = this->require_tensor(ctx_data, pfx + "ffn_norm.weight");
            L.ffn_gate     = this->require_tensor(ctx_data, pfx + "ffn_gate.weight");
            L.ffn_up       = this->require_tensor(ctx_data, pfx + "ffn_up.weight");
            L.ffn_down     = this->require_tensor(ctx_data, pfx + "ffn_down.weight");
            L.is_cross_attn = i < (int) is_cross_.size() && is_cross_[i];
        }

        for (int i = 0; i < ctx_->num_layers; ++i) {
            if (ctx_->layers[i].is_cross_attn) {
                ctx_->llm_kv_dim = (int) ctx_->layers[i].attn_k->ne[0];
                break;
            }
        }

        ctx_->action_mean = this->read_f32_tensor(ctx_data, "smolvla.unnorm.action_mean");
        ctx_->action_std  = this->read_f32_tensor(ctx_data, "smolvla.unnorm.action_std");
        ctx_->action_dim  = (int) ctx_->action_mean.size();

        return true;
    }
private:
    smolvla_action_expert * ctx_;
    int verbosity_;
    std::vector<bool> is_cross_;
};

// ============================================================================
// Load Action Expert from GGUF
// ============================================================================

struct smolvla_action_expert * smolvla_action_expert_load(const char * fname, int verbosity) {
    auto * ctx = new smolvla_action_expert();
    ctx->verbosity = verbosity;

    ctx->sched = nullptr;
    backend_scheduler_config scheduler_config;
    scheduler_config.max_nodes = 4096;
    scheduler_config.parallel = false;
    scheduler_config.op_offload = true;
    backend_loader backend;
    if (!backend.load(
            ctx->backend_cpu,
            ctx->backends,
            ctx->sched,
            ctx->buft_policy,
            smolvla_action_expert_use_accel_backend(),
            scheduler_config,
            verbosity)) {
        LOG_ERR("%s: failed to initialize action backend: %s\n", __func__, backend.error().c_str());
        smolvla_action_expert_free(ctx);
        return nullptr;
    }

    gguf_load_result loaded;
    {
        smolvla_action_expert_loader loader(ctx, verbosity);
        if (!loader.load(fname, ctx->buft_policy.model_buft, loaded, verbosity)) {
            LOG_ERR("%s: failed to load action expert tensors: %s\n", __func__, loader.error().c_str());
            smolvla_action_expert_free(ctx);
            return nullptr;
        }
    }
    ctx->ctx_gguf = loaded.gguf;
    ctx->ctx_data = loaded.ctx_data;
    ctx->ctxs.push_back(ctx->ctx_data);
    ctx->bufs.push_back(loaded.model_buffer);

    precompute_time_embedding_cache(ctx);
    
    // init attention runtime ( position ids and masks)
    if (!init_attention_runtime(ctx)) {
        LOG_ERR("%s: failed to init attention runtime\n", __func__);
        smolvla_action_expert_free(ctx);
        return nullptr;
    }

    if (verbosity >= 1) {
        LOG_INF("%s: head_dim=%d, n_q_heads=%d, n_kv_heads=%d, llm_kv_dim=%d\n",
                __func__, ctx->head_dim, ctx->n_q_heads, ctx->n_kv_heads, ctx->llm_kv_dim);
        LOG_INF("%s: action_dim=%d (from unnorm stats)\n", __func__, ctx->action_dim);
        LOG_INF("%s: action_mean: [", __func__);
        for (int i = 0; i < ctx->action_dim; i++) {
            fprintf(stderr, "%.4f%s", ctx->action_mean[i], i < ctx->action_dim - 1 ? ", " : "");
        }
        fprintf(stderr, "]\n");
        LOG_INF("%s: action_std:  [", __func__);
        for (int i = 0; i < ctx->action_dim; i++) {
            fprintf(stderr, "%.4f%s", ctx->action_std[i], i < ctx->action_dim - 1 ? ", " : "");
        }
        fprintf(stderr, "]\n");

        // Print a few tensor shapes for verification
        LOG_INF("%s: action_in_proj:  [%lld, %lld] (in=%lld → out=%lld)\n", __func__,
                (long long) ctx->action_in_proj_w->ne[0], (long long) ctx->action_in_proj_w->ne[1],
                (long long) ctx->action_in_proj_w->ne[0], (long long) ctx->action_in_proj_w->ne[1]);
        LOG_INF("%s: action_out_proj: [%lld, %lld]\n", __func__,
                (long long) ctx->action_out_proj_w->ne[0], (long long) ctx->action_out_proj_w->ne[1]);
        LOG_INF("%s: time_mlp_in:     [%lld, %lld]\n", __func__,
                (long long) ctx->time_mlp_in_w->ne[0], (long long) ctx->time_mlp_in_w->ne[1]);
        LOG_INF("%s: layer 0 (self):  q=[%lld,%lld] k=[%lld,%lld]\n", __func__,
                (long long) ctx->layers[0].attn_q->ne[0], (long long) ctx->layers[0].attn_q->ne[1],
                (long long) ctx->layers[0].attn_k->ne[0], (long long) ctx->layers[0].attn_k->ne[1]);
        if (ctx->num_layers > 1) {
            LOG_INF("%s: layer 1 (cross): q=[%lld,%lld] k=[%lld,%lld]\n", __func__,
                    (long long) ctx->layers[1].attn_q->ne[0], (long long) ctx->layers[1].attn_q->ne[1],
                    (long long) ctx->layers[1].attn_k->ne[0], (long long) ctx->layers[1].attn_k->ne[1]);
        }
        LOG_INF("%s: action expert loaded successfully\n", __func__);
        LOG_INF("%s: precomputed %d fixed timestep embeddings\n", __func__, (int) ctx->precomputed_timesteps.size());
    }

    ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());

    {
        ggml_cgraph * gf = embed_suffix_build_graph(ctx);
        if (!gf || !ggml_backend_sched_reserve(ctx->sched, gf)) {
            LOG_ERR("%s: failed to reserve embed_suffix graph\n", __func__);
            smolvla_action_expert_free(ctx);
            return nullptr;
        }
    }

    return ctx;
}

// ============================================================================
// Sinusoidal Time Embedding (CPU)
// ============================================================================
// Python:
//   fraction = linspace(0, 1, dimension//2)  in float64
//   period = min_period * (max_period / min_period) ** fraction
//   scaling = 1/period * 2π
//   sin_input = scaling * time
//   emb = [sin(sin_input), cos(sin_input)]

static void compute_sinusoidal_time_emb(
    float timestep,
    int dimension,
    float min_period,
    float max_period,
    float * output  // [dimension]
) {
    const int half = dimension / 2;
    const double ratio = (double) max_period / (double) min_period;
    const double two_pi = 2.0 * M_PI;

    for (int i = 0; i < half; i++) {
        double fraction = (double) i / (double) (half - 1);  // linspace(0, 1, half)
        double period = (double) min_period * pow(ratio, fraction);
        double scaling = (1.0 / period) * two_pi;
        double angle = scaling * (double) timestep;
        output[i]        = (float) sin(angle);
        output[i + half] = (float) cos(angle);
    }
}

static void precompute_time_embedding_cache(smolvla_action_expert * ctx) {
    if (!ctx || ctx->num_steps <= 0 || ctx->hidden_size <= 0 || ctx->chunk_size <= 0) {
        return;
    }

    ctx->precomputed_timesteps.resize(ctx->num_steps);
    ctx->precomputed_time_emb_2d.resize((size_t) ctx->num_steps * ctx->hidden_size * ctx->chunk_size);

    const float dt = -1.0f / (float) ctx->num_steps;
    std::vector<float> time_emb_1d(ctx->hidden_size);

    for (int step = 0; step < ctx->num_steps; ++step) {
        const float timestep = 1.0f + step * dt;
        ctx->precomputed_timesteps[step] = timestep;

        compute_sinusoidal_time_emb(
            timestep,
            ctx->hidden_size,
            ctx->min_period,
            ctx->max_period,
            time_emb_1d.data());

        float * dst = ctx->precomputed_time_emb_2d.data() + (size_t) step * ctx->hidden_size * ctx->chunk_size;
        for (int j = 0; j < ctx->chunk_size; ++j) {
            memcpy(dst + (size_t) j * ctx->hidden_size, time_emb_1d.data(), ctx->hidden_size * sizeof(float));
        }
    }
}

static const float * resolve_time_emb_2d(
    smolvla_action_expert * ctx,
    float timestep,
    std::vector<float> & time_emb_2d_fallback) {
    for (size_t step = 0; step < ctx->precomputed_timesteps.size(); ++step) {
        // hit cache
        if (fabsf(ctx->precomputed_timesteps[step] - timestep) <= 1e-6f) {
            return ctx->precomputed_time_emb_2d.data() + step * ctx->hidden_size * ctx->chunk_size;
        }
    }
    // cache miss, compute on the fly 
    std::vector<float> time_emb_1d(ctx->hidden_size);
    compute_sinusoidal_time_emb(
        timestep, ctx->hidden_size, ctx->min_period, ctx->max_period, time_emb_1d.data());

    time_emb_2d_fallback.resize((size_t) ctx->hidden_size * ctx->chunk_size);
    for (int j = 0; j < ctx->chunk_size; ++j) {
        memcpy(time_emb_2d_fallback.data() + (size_t) j * ctx->hidden_size,
               time_emb_1d.data(),
               (size_t) ctx->hidden_size * sizeof(float));
    }

    return time_emb_2d_fallback.data();
}

// ============================================================================
// Build embed_suffix compute graph
// ============================================================================
// Inputs:
//   noisy_actions: [max_action_dim, chunk_size]  (ggml 2D)
//   time_emb:      [hidden_size, chunk_size]      (pre-expanded)
// Output:
//   suffix_emb:    [hidden_size, chunk_size]
//
// Graph:
//   action_emb = mul_mat(action_in_proj_w, noisy_actions) + bias  → [hidden, chunk]
//   concat = cat(action_emb, time_emb, dim=0)                     → [hidden*2, chunk]
//   x = mul_mat(time_mlp_in_w, concat) + bias                     → [hidden, chunk]
//   x = silu(x)
//   x = mul_mat(time_mlp_out_w, x) + bias                         → [hidden, chunk]

static struct ggml_tensor * build_embed_suffix_op(
    smolvla_action_expert * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * noisy_actions,
    struct ggml_tensor * time_emb) {
    struct ggml_tensor * action_emb = ggml_mul_mat(ctx0, ctx->action_in_proj_w, noisy_actions);
    action_emb = ggml_add(ctx0, action_emb, ctx->action_in_proj_b);

    struct ggml_tensor * concat = ggml_concat(ctx0, action_emb, time_emb, 0);

    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ctx->time_mlp_in_w, concat);
    cur = ggml_add(ctx0, cur, ctx->time_mlp_in_b);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, ctx->time_mlp_out_w, cur);
    cur = ggml_add(ctx0, cur, ctx->time_mlp_out_b);
    return cur;
}

static ggml_cgraph * embed_suffix_build_graph(smolvla_action_expert * ctx) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Inputs
    struct ggml_tensor * noisy_actions = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, ctx->max_action_dim, ctx->chunk_size);
    ggml_set_name(noisy_actions, "noisy_actions");
    ggml_set_input(noisy_actions);

    struct ggml_tensor * time_emb = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, ctx->hidden_size, ctx->chunk_size);
    ggml_set_name(time_emb, "time_emb");
    ggml_set_input(time_emb);

    struct ggml_tensor * cur = build_embed_suffix_op(ctx, ctx0, noisy_actions, time_emb);

    ggml_set_name(cur, "suffix_emb");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

// ============================================================================
// embed_suffix execution
// ============================================================================

bool smolvla_action_expert_embed_suffix(
    struct smolvla_action_expert * ctx,
    int n_threads,
    const float * noisy_actions,
    float timestep,
    float * output
) {
    if (!ctx || !noisy_actions || !output) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }
    ctx->graph_n_threads = n_threads; //action部分线程数

    std::vector<float> time_emb_2d_fallback;
    const float * time_emb_2d_ptr = resolve_time_emb_2d(ctx, timestep, time_emb_2d_fallback);

    const auto t_build_start = std::chrono::high_resolution_clock::now();
    ggml_cgraph * gf = embed_suffix_build_graph(ctx);
    if (!ctx->sched) {
        LOG_ERR("%s: scheduler is required\n", __func__);
        return false;
    }
    ggml_backend_sched_reset(ctx->sched);
    if (!gf || !ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        LOG_ERR("%s: failed to allocate embed_suffix graph\n", __func__);
        return false;
    }
    const auto t_build_end = std::chrono::high_resolution_clock::now();

    // Set inputs
    struct ggml_tensor * inp_actions = ggml_graph_get_tensor(gf, "noisy_actions");
    struct ggml_tensor * inp_time    = ggml_graph_get_tensor(gf, "time_emb");
    if (!inp_actions || !inp_time) {
        LOG_ERR("%s: failed to get input tensors\n", __func__);
        return false;
    }

    // noisy_actions layout: [max_action_dim, chunk_size] in ggml
    // Input from caller is [chunk_size * max_action_dim] row-major (each row = one action)
    // ggml expects column-major: ne[0]=max_action_dim consecutive, then ne[1]=chunk_size
    // Row-major [chunk_size][max_action_dim] == column-major [max_action_dim, chunk_size] ✓
    ggml_backend_tensor_set(inp_actions, noisy_actions, 0,
                            ctx->max_action_dim * ctx->chunk_size * sizeof(float));
    ggml_backend_tensor_set(inp_time, time_emb_2d_ptr, 0,
                            ctx->hidden_size * ctx->chunk_size * sizeof(float));

    // Compute
    set_backend_threads(ctx->backends, n_threads);
    const auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_backend_sched_graph_compute(ctx->sched, gf);
    const auto t_compute_end = std::chrono::high_resolution_clock::now();

    if (ctx->verbosity >= 1) {
        LOG_INF("%s: graph build+alloc: %.2f ms, compute: %.2f ms\n", __func__,
                std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count(),
                std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());
    }

    // Read output
    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "suffix_emb");
    if (!out) {
        LOG_ERR("%s: failed to get output tensor\n", __func__);
        return false;
    }
    ggml_backend_tensor_get(out, output, 0, ctx->hidden_size * ctx->chunk_size * sizeof(float));

    return true;
}
static bool attention_runtime_is_ready(
    const smolvla_action_expert * ctx,
    int prefix_seq_len) {
    return ctx &&
        ctx->attention_runtime.prepared &&
        ctx->attention_runtime.prefix_seq_len == prefix_seq_len;
}

static bool init_attention_runtime(
    smolvla_action_expert * ctx) {
    if (!ctx) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (ctx->attention_runtime.ctx_data) {
        return true;
    }

    clear_attention_runtime(ctx);

    const size_t tensor_count = 4 + 1;
    const size_t ctx_size = ggml_tensor_overhead() * tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx->attention_runtime.ctx_data = ggml_init(params);
    if (!ctx->attention_runtime.ctx_data) {
        LOG_ERR("%s: failed to init attention runtime ctx\n", __func__);
        clear_attention_runtime(ctx);
        return false;
    }

    ctx->attention_runtime.prepared = false;
    return true;
}

static bool create_attention_runtime_tensors(
    smolvla_action_expert * ctx,
    int prefix_seq_len) {
    if (!ctx || prefix_seq_len <= 0) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (ctx->attention_runtime.prefix_seq_len == prefix_seq_len &&
        ctx->attention_runtime.ctx_data != nullptr &&
        ctx->attention_runtime.buffer != nullptr &&
        ctx->attention_runtime.self_pos != nullptr &&
        ctx->attention_runtime.cross_pos != nullptr &&
        ctx->attention_runtime.self_mask != nullptr &&
        ctx->attention_runtime.cross_mask != nullptr) {
        ctx->attention_runtime.prepared = false;
        return true;
    }

    clear_attention_runtime(ctx);

    const size_t tensor_count = 4 + 1;
    const size_t ctx_size = ggml_tensor_overhead() * tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx->attention_runtime.ctx_data = ggml_init(params);
    if (!ctx->attention_runtime.ctx_data) {
        LOG_ERR("%s: failed to init attention runtime ctx\n", __func__);
        clear_attention_runtime(ctx);
        return false;
    }

    const int q_pad = GGML_PAD(ctx->chunk_size, GGML_KQ_MASK_PAD);
    ctx->attention_runtime.self_pos = ggml_new_tensor_1d(
        ctx->attention_runtime.ctx_data, GGML_TYPE_I32, ctx->chunk_size);
    ctx->attention_runtime.cross_pos = ggml_new_tensor_1d(
        ctx->attention_runtime.ctx_data, GGML_TYPE_I32, ctx->chunk_size);
    ctx->attention_runtime.self_mask = ggml_new_tensor_2d(
        ctx->attention_runtime.ctx_data, GGML_TYPE_F32, prefix_seq_len + ctx->chunk_size, q_pad);
    ctx->attention_runtime.cross_mask = ggml_new_tensor_2d(
        ctx->attention_runtime.ctx_data, GGML_TYPE_F32, prefix_seq_len, q_pad);
    if (!ctx->attention_runtime.self_pos ||
        !ctx->attention_runtime.cross_pos ||
        !ctx->attention_runtime.self_mask ||
        !ctx->attention_runtime.cross_mask) {
        LOG_ERR("%s: failed to create attention runtime tensors\n", __func__);
        clear_attention_runtime(ctx);
        return false;
    }

    ggml_set_name(ctx->attention_runtime.self_pos, "attention_self_pos");
    ggml_set_name(ctx->attention_runtime.cross_pos, "attention_cross_pos");
    ggml_set_name(ctx->attention_runtime.self_mask, "attention_self_mask");
    ggml_set_name(ctx->attention_runtime.cross_mask, "attention_cross_mask");

    ctx->attention_runtime.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx->attention_runtime.ctx_data, ctx->buft_policy.runtime_buft);
    if (!ctx->attention_runtime.buffer) {
        LOG_ERR("%s: failed to allocate attention runtime buffer\n", __func__);
        clear_attention_runtime(ctx);
        return false;
    }

    ctx->attention_runtime.prefix_seq_len = prefix_seq_len;
    ctx->attention_runtime.prepared = false;
    return true;
}

bool smolvla_action_expert_prepare_attention_cache(
    smolvla_action_expert * ctx,
    const uint8_t * prefix_valid_mask,
    int prefix_seq_len) {
    if (!ctx || !prefix_valid_mask || prefix_seq_len <= 0) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (attention_runtime_is_ready(ctx, prefix_seq_len)) {
        return true;
    }
    if (ctx->attention_runtime.prefix_seq_len != prefix_seq_len ||
        !ctx->attention_runtime.self_pos ||
        !ctx->attention_runtime.cross_pos ||
        !ctx->attention_runtime.self_mask ||
        !ctx->attention_runtime.cross_mask) {
        if (!create_attention_runtime_tensors(ctx, prefix_seq_len)) {
            LOG_ERR("%s: failed to create attention runtime tensors\n", __func__);
            return false;
        }
    }

    std::vector<int32_t> self_position_ids((size_t) ctx->chunk_size);
    std::vector<int32_t> cross_position_ids((size_t) ctx->chunk_size);

    std::vector<float> self_attention_mask_f32;
    std::vector<float> cross_attention_mask_f32;

    build_self_attn_mask_and_positions(
        prefix_valid_mask,
        prefix_seq_len,
        ctx->chunk_size,
        self_position_ids.data(),
        self_attention_mask_f32);
    build_cross_attn_mask_and_positions(
        prefix_valid_mask,
        prefix_seq_len,
        ctx->chunk_size,
        cross_position_ids.data(),
        cross_attention_mask_f32);

    ggml_backend_tensor_set(
        ctx->attention_runtime.self_pos,
        self_position_ids.data(),
        0,
        self_position_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        ctx->attention_runtime.cross_pos,
        cross_position_ids.data(),
        0,
        cross_position_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        ctx->attention_runtime.self_mask,
        self_attention_mask_f32.data(),
        0,
        self_attention_mask_f32.size() * sizeof(float));
    ggml_backend_tensor_set(
        ctx->attention_runtime.cross_mask,
        cross_attention_mask_f32.data(),
        0,
        cross_attention_mask_f32.size() * sizeof(float));
    ctx->attention_runtime.prepared = true;

    return true;
}

static struct ggml_tensor * rms_norm_weighted(
    struct ggml_context * ctx0,
    struct ggml_tensor * x,
    struct ggml_tensor * weight,
    float eps) {
    struct ggml_tensor * y = ggml_rms_norm(ctx0, x, eps);
    y = ggml_mul(ctx0, y, weight);
    return y;
}

static struct ggml_tensor * build_self_attn_layer_op(
    smolvla_action_expert * ctx,
    int layer_idx,
    int prefix_seq_len,
    struct ggml_context * ctx0,
    struct ggml_tensor * hidden_in,
    struct ggml_tensor * prefix_k_in,
    struct ggml_tensor * prefix_v_in,
    struct ggml_tensor * pos_in,
    struct ggml_tensor * mask_in) {
    const auto & layer = ctx->layers[layer_idx];
    const int chunk = ctx->chunk_size;
    const int head_dim = ctx->head_dim;
    const int q_dim = ctx->n_q_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float attn_scale = 1.0f / sqrtf((float) head_dim);

    struct ggml_tensor * norm1 = rms_norm_weighted(ctx0, hidden_in, layer.attn_norm, rms_eps);

    struct ggml_tensor * q = ggml_mul_mat(ctx0, layer.attn_q, norm1);
    q = ggml_reshape_3d(ctx0, q, head_dim, ctx->n_q_heads, chunk);
    q = ggml_rope(ctx0, q, pos_in, head_dim, 0);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    struct ggml_tensor * k_suffix = ggml_mul_mat(ctx0, layer.attn_k, norm1);
    k_suffix = ggml_reshape_3d(ctx0, k_suffix, head_dim, ctx->n_kv_heads, chunk);
    k_suffix = ggml_rope(ctx0, k_suffix, pos_in, head_dim, 0);

    struct ggml_tensor * v_suffix = ggml_mul_mat(ctx0, layer.attn_v, norm1);
    v_suffix = ggml_reshape_3d(ctx0, v_suffix, head_dim, ctx->n_kv_heads, chunk);

    struct ggml_tensor * prefix_k_3d = ggml_reshape_3d(ctx0, prefix_k_in, head_dim, ctx->n_kv_heads, prefix_seq_len);
    struct ggml_tensor * prefix_v_3d = ggml_reshape_3d(ctx0, prefix_v_in, head_dim, ctx->n_kv_heads, prefix_seq_len);

    struct ggml_tensor * k_all = ggml_concat(ctx0, prefix_k_3d, k_suffix, 2);
    struct ggml_tensor * v_all = ggml_concat(ctx0, prefix_v_3d, v_suffix, 2);
    struct ggml_tensor * k = ggml_permute(ctx0, k_all, 0, 2, 1, 3);
    struct ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, v_all, 1, 2, 0, 3));

    struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx0, kq, mask_in, attn_scale, 0.0f);
    struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
    struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    struct ggml_tensor * attn = ggml_cont_2d(ctx0, kqv_merged, q_dim, chunk);

    struct ggml_tensor * attn_proj = ggml_mul_mat(ctx0, layer.attn_output, attn);
    struct ggml_tensor * residual1 = ggml_add(ctx0, attn_proj, hidden_in);

    struct ggml_tensor * norm2 = rms_norm_weighted(ctx0, residual1, layer.ffn_norm, rms_eps);
    struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, norm2);
    gate = ggml_silu(ctx0, gate);
    struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, norm2);
    struct ggml_tensor * swiglu = ggml_mul(ctx0, gate, up);
    struct ggml_tensor * down = ggml_mul_mat(ctx0, layer.ffn_down, swiglu);
    return ggml_add(ctx0, down, residual1);
}

static struct ggml_tensor * build_cross_attn_layer_op(
    smolvla_action_expert * ctx,
    int layer_idx,
    int prefix_seq_len,
    struct ggml_context * ctx0,
    struct ggml_tensor * hidden_in,
    struct ggml_tensor * prefix_k_in,
    struct ggml_tensor * prefix_v_in,
    struct ggml_tensor * pos_in,
    struct ggml_tensor * mask_in) {
    const auto & layer = ctx->layers[layer_idx];
    const int chunk = ctx->chunk_size;
    const int head_dim = ctx->head_dim;
    const int q_dim = ctx->n_q_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float attn_scale = 1.0f / sqrtf((float) head_dim);

    struct ggml_tensor * norm1 = rms_norm_weighted(ctx0, hidden_in, layer.attn_norm, rms_eps);

    struct ggml_tensor * q = ggml_mul_mat(ctx0, layer.attn_q, norm1);
    q = ggml_reshape_3d(ctx0, q, head_dim, ctx->n_q_heads, chunk);
    q = ggml_rope(ctx0, q, pos_in, head_dim, 0);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    struct ggml_tensor * k = ggml_mul_mat(ctx0, layer.attn_k, prefix_k_in);
    k = ggml_reshape_3d(ctx0, k, head_dim, ctx->n_kv_heads, prefix_seq_len);

    struct ggml_tensor * v = ggml_mul_mat(ctx0, layer.attn_v, prefix_v_in);
    v = ggml_reshape_3d(ctx0, v, head_dim, ctx->n_kv_heads, prefix_seq_len);
    struct ggml_tensor * k_view = ggml_permute(ctx0, k, 0, 2, 1, 3);
    struct ggml_tensor * v_view = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

    struct ggml_tensor * kq = ggml_mul_mat(ctx0, k_view, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx0, kq, mask_in, attn_scale, 0.0f);
    struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v_view, kq);
    struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    struct ggml_tensor * attn = ggml_cont_2d(ctx0, kqv_merged, q_dim, chunk);

    struct ggml_tensor * attn_proj = ggml_mul_mat(ctx0, layer.attn_output, attn);
    struct ggml_tensor * residual1 = ggml_add(ctx0, attn_proj, hidden_in);

    struct ggml_tensor * norm2 = rms_norm_weighted(ctx0, residual1, layer.ffn_norm, rms_eps);
    struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, norm2);
    gate = ggml_silu(ctx0, gate);
    struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, norm2);
    struct ggml_tensor * swiglu = ggml_mul(ctx0, gate, up);
    struct ggml_tensor * down = ggml_mul_mat(ctx0, layer.ffn_down, swiglu);
    return ggml_add(ctx0, down, residual1);
}

static struct ggml_tensor * build_transformer_stack_op(
    smolvla_action_expert * ctx,
    int prefix_seq_len,
    struct ggml_context * ctx0,
    struct ggml_tensor * hidden_in,
    struct ggml_tensor * self_pos,
    struct ggml_tensor * cross_pos,
    struct ggml_tensor * self_mask,
    struct ggml_tensor * cross_mask) {
    if (!ctx || !ctx0 || !hidden_in || !self_pos || !cross_pos || !self_mask || !cross_mask) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return nullptr;
    }

    if (ctx->prefix_kv_runtime.k_layers.size() != (size_t) ctx->num_layers ||
        ctx->prefix_kv_runtime.v_layers.size() != (size_t) ctx->num_layers) {
        LOG_ERR("%s: prefix KV runtime is not prepared\n", __func__);
        return nullptr;
    }

    struct ggml_tensor * cur = hidden_in;
    for (int layer_idx = 0; layer_idx < ctx->num_layers; ++layer_idx) {
        struct ggml_tensor * prefix_k = ctx->prefix_kv_runtime.k_layers[layer_idx];
        struct ggml_tensor * prefix_v = ctx->prefix_kv_runtime.v_layers[layer_idx];
        if (!prefix_k || !prefix_v) {
            LOG_ERR("%s: missing prepared prefix KV tensors for layer %d\n", __func__, layer_idx);
            return nullptr;
        }

        cur = ctx->layers[layer_idx].is_cross_attn
            ? build_cross_attn_layer_op(
                ctx,
                layer_idx,
                prefix_seq_len,
                ctx0,
                cur,
                prefix_k,
                prefix_v,
                cross_pos,
                cross_mask)
            : build_self_attn_layer_op(
                ctx,
                layer_idx,
                prefix_seq_len,
                ctx0,
                cur,
                prefix_k,
                prefix_v,
                self_pos,
                self_mask);
    }

    return cur;
}

static bool transformer_velocity_build_graph(
    smolvla_action_expert * ctx,
    int prefix_seq_len,
    smolvla_transformer_graph_state & graph_state) {
    const int chunk = ctx->chunk_size;
    const int hidden = ctx->hidden_size;
    const int q_pad = GGML_PAD(chunk, GGML_KQ_MASK_PAD);
    const size_t graph_nodes = 4096;

    graph_state.meta_buf.resize(smolvla_graph_meta_size(graph_nodes));
    struct ggml_init_params params = {
        /*.mem_size   =*/ graph_state.meta_buf.size(),
        /*.mem_buffer =*/ graph_state.meta_buf.data(),
        /*.no_alloc   =*/ true,
    };

    graph_state.ctx_graph = ggml_init(params);
    if (!graph_state.ctx_graph) {
        LOG_ERR("%s: failed to init transformer graph ctx\n", __func__);
        return false;
    }

    graph_state.inp_hidden = ggml_new_tensor_2d(graph_state.ctx_graph, GGML_TYPE_F32, hidden, chunk);

    ggml_set_input(graph_state.inp_hidden);

    graph_state.inp_prefix_k = ctx->prefix_kv_runtime.k_layers;
    graph_state.inp_prefix_v = ctx->prefix_kv_runtime.v_layers;

    struct ggml_tensor * cur = build_transformer_stack_op(
        ctx,
        prefix_seq_len,
        graph_state.ctx_graph,
        graph_state.inp_hidden,
        ctx->attention_runtime.self_pos,
        ctx->attention_runtime.cross_pos,
        ctx->attention_runtime.self_mask,
        ctx->attention_runtime.cross_mask);
    if (!cur) {
        clear_transformer_graph(graph_state);
        return false;
    }

    graph_state.out = cur;
    struct ggml_tensor * velocity_out = build_project_velocity_op(ctx, graph_state.ctx_graph, graph_state.out);
    if (!velocity_out) {
        clear_transformer_graph(graph_state);
        return false;
    }

    graph_state.out = velocity_out;
    ggml_set_name(graph_state.out, "transformer_velocity_out");
    ggml_set_output(graph_state.out);

    graph_state.graph = ggml_new_graph_custom(graph_state.ctx_graph, 4096, false);
    ggml_build_forward_expand(graph_state.graph, graph_state.out);
    return true;
}

static bool reserve_transformer_velocity_graph(
    smolvla_action_expert * ctx,
    int prefix_seq_len) {
    if (!ctx || prefix_seq_len <= 0) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (ctx->transformer_runtime_prefix_seq_len == prefix_seq_len) {
        return true;
    }

    smolvla_transformer_graph_state graph_state;
    if (!transformer_velocity_build_graph(ctx, prefix_seq_len, graph_state)) {
        LOG_ERR("%s: failed to build transformer+velocity reserve graph\n", __func__);
        return false;
    }

    const bool reserve_ok = ctx->sched && ggml_backend_sched_reserve(ctx->sched, graph_state.graph);
    if (!reserve_ok) {
        LOG_ERR("%s: failed to reserve transformer+velocity graph\n", __func__);
        clear_transformer_graph(graph_state);
        return false;
    }

    if (ctx->verbosity >= 1) {
        LOG_INF("%s: transformer+velocity graph nodes=%d splits=%d backends=%zu\n",
                __func__,
                ggml_graph_n_nodes(graph_state.graph),
                ggml_backend_sched_get_n_splits(ctx->sched),
                ctx->backends.size());
    }

    ctx->transformer_runtime_prefix_seq_len = prefix_seq_len;
    clear_transformer_graph(graph_state);
    return true;
}

bool smolvla_action_expert_init_fixed_prefix_runtime(
    struct smolvla_action_expert * ctx,
    int prefix_seq_len
) {
    if (!ctx || prefix_seq_len <= 0) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (!ensure_prefix_kv_runtime(ctx, prefix_seq_len)) {
        LOG_ERR("%s: failed to init prefix KV runtime\n", __func__);
        return false;
    }

    if (!create_attention_runtime_tensors(ctx, prefix_seq_len)) {
        LOG_ERR("%s: failed to create attention runtime tensors\n", __func__);
        return false;
    }

    if (!reserve_transformer_velocity_graph(ctx, prefix_seq_len)) {
        LOG_ERR("%s: failed to reserve transformer+velocity graph\n", __func__);
        return false;
    }

    return true;
}

static struct ggml_tensor * build_project_velocity_op(
    smolvla_action_expert * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * hidden_in) {
    const float rms_eps = 1e-5f;
    struct ggml_tensor * norm = rms_norm_weighted(ctx0, hidden_in, ctx->final_norm, rms_eps);
    struct ggml_tensor * vel = ggml_mul_mat(ctx0, ctx->action_out_proj_w, norm);
    vel = ggml_add(ctx0, vel, ctx->action_out_proj_b);
    return vel;
}

static int build_self_attn_mask_and_positions(
    const uint8_t * prefix_valid_mask,
    int prefix_seq_len,
    int suffix_len,
    int32_t * position_ids,
    std::vector<float> & attention_mask_f32) {
    int prefix_valid_count = 0;
    for (int i = 0; i < prefix_seq_len; ++i) {
        prefix_valid_count += prefix_valid_mask[i] != 0;
    }

    for (int i = 0; i < suffix_len; ++i) {
        position_ids[i] = prefix_valid_count + i;
    }

    const int total_kv_len = prefix_seq_len + suffix_len;
    const int q_pad = GGML_PAD(suffix_len, GGML_KQ_MASK_PAD);
    attention_mask_f32.assign((size_t) total_kv_len * q_pad, -INFINITY);

    std::vector<float> prefix_mask_f32((size_t) prefix_seq_len);
    for (int kj = 0; kj < prefix_seq_len; ++kj) {
        prefix_mask_f32[kj] = prefix_valid_mask[kj] ? 0.0f : -INFINITY;
    }

    for (int qi = 0; qi < suffix_len; ++qi) {
        float * row = attention_mask_f32.data() + (size_t) qi * total_kv_len;
        memcpy(row, prefix_mask_f32.data(), (size_t) prefix_seq_len * sizeof(float));
        std::fill_n(row + prefix_seq_len, qi + 1, 0.0f);
    }

    return prefix_valid_count;
}

static void build_cross_attn_mask_and_positions(
    const uint8_t * prefix_valid_mask,
    int prefix_seq_len,
    int suffix_len,
    int32_t * position_ids,
    std::vector<float> & attention_mask_f32) {
    for (int i = 0; i < suffix_len; ++i) {
        position_ids[i] = i;
    }

    const int q_pad = GGML_PAD(suffix_len, GGML_KQ_MASK_PAD);
    attention_mask_f32.assign((size_t) prefix_seq_len * q_pad, -INFINITY);

    std::vector<float> prefix_mask_f32((size_t) prefix_seq_len);
    for (int kj = 0; kj < prefix_seq_len; ++kj) {
        prefix_mask_f32[kj] = prefix_valid_mask[kj] ? 0.0f : -INFINITY;
    }

    for (int qi = 0; qi < suffix_len; ++qi) {
        float * row = attention_mask_f32.data() + (size_t) qi * prefix_seq_len;
        memcpy(row, prefix_mask_f32.data(), (size_t) prefix_seq_len * sizeof(float));
    }
}

bool smolvla_action_expert_eval_transformer_project_velocity(
    struct smolvla_action_expert * ctx,
    const float * hidden_in,
    float * velocity_out
) {
    if (!ctx || !hidden_in || !velocity_out) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    const int prefix_seq_len = ctx->prefix_kv_runtime.prefix_seq_len;
    if (!attention_runtime_is_ready(ctx, prefix_seq_len)) {
        LOG_ERR("%s: attention cache is not prepared\n", __func__);
        return false;
    }
    const auto t_build_start = std::chrono::high_resolution_clock::now();
    smolvla_transformer_graph_state graph_state;
    if (!transformer_velocity_build_graph(ctx, prefix_seq_len, graph_state)) {
        LOG_ERR("%s: failed to build transformer+velocity graph\n", __func__);
        return false;
    }

    if (!ctx->sched) {
        LOG_ERR("%s: scheduler is required\n", __func__);
        clear_transformer_graph(graph_state);
        return false;
    }
    ggml_backend_sched_reset(ctx->sched);
    const bool alloc_ok = ggml_backend_sched_alloc_graph(ctx->sched, graph_state.graph);
    if (!alloc_ok) {
        LOG_ERR("%s: failed to allocate transformer+velocity graph\n", __func__);
        clear_transformer_graph(graph_state);
        return false;
    }
    const auto t_build_end = std::chrono::high_resolution_clock::now();

    ggml_backend_tensor_set(graph_state.inp_hidden, hidden_in, 0, (size_t) ctx->chunk_size * ctx->hidden_size * sizeof(float));

    set_backend_threads(ctx->backends, ctx->graph_n_threads);

    const auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_backend_sched_graph_compute(ctx->sched, graph_state.graph);
    const auto t_compute_end = std::chrono::high_resolution_clock::now();

    ggml_backend_tensor_get(
        graph_state.out,
        velocity_out,
        0,
        (size_t) ctx->chunk_size * ctx->max_action_dim * sizeof(float));

    if (ctx->verbosity >= 1) {
        LOG_INF("%s: graph build+alloc: %.2f ms, compute: %.2f ms\n", __func__,
                std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count(),
                std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());
    }

    clear_transformer_graph(graph_state);
    return true;
}

bool smolvla_action_expert_prepare_prefix_kv_from_backend(
    struct smolvla_action_expert * ctx,
    int prefix_seq_len,
    struct ggml_tensor * const * prefix_k_layers,
    struct ggml_tensor * const * prefix_v_layers,
    bool prefix_v_trans
) {
    if (!ctx || prefix_seq_len <= 0 || !prefix_k_layers || !prefix_v_layers) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    if (!ensure_prefix_kv_runtime(ctx, prefix_seq_len)) {
        LOG_ERR("%s: failed to ensure prefix KV runtime\n", __func__);
        return false;
    }

    const auto t_build_start = std::chrono::high_resolution_clock::now();
    const int graph_nodes = ctx->num_layers * 8 + 16;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        LOG_ERR("%s: failed to init backend-prefix graph ctx\n", __func__);
        return false;
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, graph_nodes, false);
    if (!gf || !ctx->sched) {
        LOG_ERR("%s: failed to create backend-prefix graph state\n", __func__);
        ggml_free(ctx0);
        return false;
    }

    bool ok = true;
    for (int layer_idx = 0; layer_idx < ctx->num_layers; ++layer_idx) {
        struct ggml_tensor * src_k = prefix_k_layers[layer_idx];
        struct ggml_tensor * src_v = prefix_v_layers[layer_idx];
        struct ggml_tensor * dst_k = ctx->prefix_kv_runtime.k_layers[layer_idx];
        struct ggml_tensor * dst_v = ctx->prefix_kv_runtime.v_layers[layer_idx];
        if (!src_k || !src_v || !dst_k || !dst_v) {
            LOG_ERR("%s: missing source/destination tensor for layer %d\n", __func__, layer_idx);
            ok = false;
            break;
        }

        const int64_t prefix_dim = dst_k->ne[0];
        const size_t k_row_size = ggml_row_size(src_k->type, prefix_dim);
        const int64_t src_k_elems = ggml_nelements(src_k);
        const int64_t src_v_elems = ggml_nelements(src_v);
        const int64_t src_v_kv_size = prefix_dim > 0 ? src_v_elems / prefix_dim : 0;

        if (src_k_elems < prefix_dim * (int64_t) prefix_seq_len) {
            LOG_ERR("%s: incompatible K layout for layer %d\n", __func__, layer_idx);
            ok = false;
            break;
        }

        struct ggml_tensor * k_view = ggml_view_2d(ctx0, src_k, prefix_dim, prefix_seq_len, k_row_size, 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_view, dst_k));

        struct ggml_tensor * v_src = nullptr;
        if (prefix_v_trans) {
            if (src_v_kv_size < prefix_seq_len || src_v_elems < prefix_dim * (int64_t) prefix_seq_len) {
                LOG_ERR("%s: incompatible transposed V layout for layer %d\n", __func__, layer_idx);
                ok = false;
                break;
            }
            struct ggml_tensor * v_view = ggml_view_2d(
                ctx0, src_v, prefix_seq_len, prefix_dim,
                (size_t) src_v_kv_size * ggml_element_size(src_v), 0);
            v_src = ggml_transpose(ctx0, v_view);
        } else {
            if (src_v_elems < prefix_dim * (int64_t) prefix_seq_len) {
                LOG_ERR("%s: incompatible V layout for layer %d\n", __func__, layer_idx);
                ok = false;
                break;
            }
            v_src = ggml_view_2d(ctx0, src_v, prefix_dim, prefix_seq_len, ggml_row_size(src_v->type, prefix_dim), 0);
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, v_src, dst_v));
    }

    if (ok) {
        ggml_backend_sched_reset(ctx->sched);
    }
    if (ok && !ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        LOG_ERR("%s: failed to allocate backend-prefix graph\n", __func__);
        ok = false;
    }
    const auto t_build_end = std::chrono::high_resolution_clock::now();
    if (ok) {
        set_backend_threads(ctx->backends, ctx->graph_n_threads);
        const auto t_compute_start = std::chrono::high_resolution_clock::now();
        ggml_backend_sched_graph_compute(ctx->sched, gf);
        const auto t_compute_end = std::chrono::high_resolution_clock::now();
        if (ctx->verbosity >= 1) {
            LOG_INF("%s: graph build+alloc: %.2f ms, compute: %.2f ms\n", __func__,
                    std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count(),
                    std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());
        }
    }
    ggml_free(ctx0);
    return ok;
}

// ============================================================================
// Free
// ============================================================================

void smolvla_action_expert_free(struct smolvla_action_expert * ctx) {
    if (!ctx) return;

    clear_attention_runtime(ctx);
    clear_prefix_kv_runtime(ctx);
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
    }
    smolvla_free_model_buffers(ctx->bufs);
    smolvla_free_model_contexts(ctx->ctxs);
    smolvla_free_scheduler_backends(ctx->backends);
    ctx->backend_cpu = nullptr;
    if (ctx->ctx_gguf) {
        gguf_free(ctx->ctx_gguf);
    }

    delete ctx;
}

// ============================================================================
// Getters
// ============================================================================

int smolvla_action_expert_hidden_size(const struct smolvla_action_expert * ctx) {
    return ctx ? ctx->hidden_size : 0;
}
