/**
 * SmolVLA State Projector Implementation (proprio.cpp style)
 * 
 * Uses ggml compute graph for: normalize (mean_std) → Linear(32, 960)
 */

#include "state_proj.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "models/gguf_loader.h"
#include "smolvla_compat.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>

// ============================================================================
// Logging (similar to bitvla)
// ============================================================================

#define LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)

// ============================================================================
// Build State Projector Compute Graph
// ============================================================================

static ggml_cgraph * state_proj_build_graph(smolvla_state_proj * ctx) {
    // Graph memory allocation
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Input tensor: [max_state_dim] (padded and normalized)
    struct ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ctx->max_state_dim);
    ggml_set_name(inp, "state_input");
    ggml_set_input(inp);

    // Cast bias to F32 if needed
    struct ggml_tensor * bias_f32 = ctx->bias;
    if (ctx->bias->type != GGML_TYPE_F32) {
        bias_f32 = ggml_cast(ctx0, ctx->bias, GGML_TYPE_F32);
    }

    // Linear: out = inp @ weight.T + bias
    // weight is [hidden_size, max_state_dim], mul_mat handles transpose
    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ctx->weight, inp);
    cur = ggml_add(ctx0, cur, bias_f32);

    // Output
    ggml_set_name(cur, "state_output");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

// ============================================================================
// Normalize State (MEAN_STD mode)
// ============================================================================

static void normalize_state_mean_std(
    const float * input,
    float * output,
    int dim,
    int max_dim,
    const std::vector<float> & mean,
    const std::vector<float> & std
) {
    const float eps = 1e-8f;

    // Normalize actual dimensions
    for (int i = 0; i < dim && i < (int)mean.size(); i++) {
        float denom = std[i] + eps;
        output[i] = (input[i] - mean[i]) / denom;
    }

    // Zero-pad remaining dimensions
    for (int i = dim; i < max_dim; i++) {
        output[i] = 0.0f;
    }
}

// ============================================================================
// Load State Projector from GGUF
// ============================================================================

class smolvla_state_proj_loader : public gguf_loader {
public:
    smolvla_state_proj_loader(
        smolvla_state_proj * ctx,
        int verbosity)
        : ctx_(ctx),
          verbosity_(verbosity) {
    }

protected:
    bool parse_metadata(gguf_context * gguf) override {
        ctx_->state_dim = (int) this->u32_or(gguf, "smolvla.state_dim", ctx_->state_dim);
        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        ctx_->weight = this->require_tensor(ctx_data, "smolvla.state_proj.weight");
        ctx_->bias = this->require_tensor(ctx_data, "smolvla.state_proj.bias");

        ctx_->max_state_dim = (int) ctx_->weight->ne[0];
        ctx_->hidden_size = (int) ctx_->weight->ne[1];

        ctx_->norm_mean = this->read_f32_tensor(ctx_data, "smolvla.norm.observation_state_mean");
        ctx_->norm_std = this->read_f32_tensor(ctx_data, "smolvla.norm.observation_state_std");
        ctx_->has_norm_stats =
            !ctx_->norm_mean.empty() &&
            !ctx_->norm_std.empty() &&
            ctx_->norm_mean.size() == ctx_->norm_std.size();

        if (verbosity_ >= 1) {
            LOG_INF("%s: state_dim = %d, max_state_dim = %d, hidden_size = %d\n",
                    __func__, ctx_->state_dim, ctx_->max_state_dim, ctx_->hidden_size);
        }

        return true;
    }

private:
    smolvla_state_proj * ctx_;
    int verbosity_;
};

struct smolvla_state_proj * smolvla_state_proj_load(const char * fname, int verbosity) {
    // Create context
    smolvla_state_proj * ctx = new smolvla_state_proj();

    // Initialize backend
#ifdef GGML_USE_CUDA
    ctx->backend = ggml_backend_cuda_init(0);
    if (ctx->backend && verbosity >= 1) {
        LOG_INF("%s: state_proj using CUDA backend\n", __func__);
    }
#elif defined(GGML_USE_METAL)
    ctx->backend = ggml_backend_metal_init();
    if (ctx->backend && verbosity >= 1) {
        LOG_INF("%s: state_proj using Metal backend\n", __func__);
    }
#endif

    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
        if (verbosity >= 1) {
            LOG_INF("%s: state_proj using CPU backend\n", __func__);
        }
    }

    gguf_load_result loaded;
    {
        smolvla_state_proj_loader loader(ctx, verbosity);
        if (!loader.load(fname, ggml_backend_get_default_buffer_type(ctx->backend), loaded, verbosity)) {
            LOG_ERR("%s: failed to load state_proj tensors: %s\n", __func__, loader.error().c_str());
            smolvla_state_proj_free(ctx);
            return nullptr;
        }
    }
    ctx->ctx_gguf = loaded.gguf;
    ctx->ctx_data = loaded.ctx_data;
    ctx->params_buffer = loaded.model_buffer;

    if (ctx->has_norm_stats) {
        if (verbosity >= 1) {
            LOG_INF("%s: loaded norm stats (mean_std), dim=%zu\n", __func__, ctx->norm_mean.size());
            LOG_INF("%s: norm_mean[:6]: ", __func__);
            for (size_t i = 0; i < 6 && i < ctx->norm_mean.size(); i++) {
                fprintf(stderr, "%.4f ", ctx->norm_mean[i]);
            }
            fprintf(stderr, "\n");
            LOG_INF("%s: norm_std[:6]: ", __func__);
            for (size_t i = 0; i < 6 && i < ctx->norm_std.size(); i++) {
                fprintf(stderr, "%.4f ", ctx->norm_std[i]);
            }
            fprintf(stderr, "\n");
        }
    } else {
        if (verbosity >= 1) {
            LOG_INF("%s: no norm stats found, normalization will be skipped\n", __func__);
        }
    }

    // Initialize compute allocator
    ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    ctx->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));

    // Pre-build graph to reserve memory
    ggml_cgraph * gf = state_proj_build_graph(ctx);
    ggml_gallocr_reserve(ctx->compute_alloc, gf);
    size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(ctx->compute_alloc, 0);
    
    if (verbosity >= 1) {
        LOG_INF("%s: compute allocated memory: %.2f KB\n", __func__, compute_memory_buffer_size / 1024.0);
        LOG_INF("%s: state_proj loaded successfully: Linear(%d, %d)\n", 
                __func__, ctx->max_state_dim, ctx->hidden_size);
    }

    return ctx;
}

// ============================================================================
// Encode State
// ============================================================================

bool smolvla_state_proj_encode(
    struct smolvla_state_proj * ctx,
    int n_threads,
    const float * state,
    int state_dim,
    float * output
) {
    if (!ctx || !state || !output) {
        LOG_ERR("%s: invalid arguments\n", __func__);
        return false;
    }

    // Normalize and pad state
    std::vector<float> normalized_state(ctx->max_state_dim, 0.0f);
    
    if (ctx->has_norm_stats) {
        normalize_state_mean_std(
            state,
            normalized_state.data(),
            state_dim,
            ctx->max_state_dim,
            ctx->norm_mean,
            ctx->norm_std
        );
    } else {
        // Just pad without normalization
        memcpy(normalized_state.data(), state, state_dim * sizeof(float));
    }

    // Build compute graph
    auto t_build_start = std::chrono::high_resolution_clock::now();
    ggml_cgraph * gf = state_proj_build_graph(ctx);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);
    auto t_build_end = std::chrono::high_resolution_clock::now();

    // Set input data
    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "state_input");
    if (!inp) {
        LOG_ERR("%s: failed to get input tensor\n", __func__);
        return false;
    }

    ggml_backend_tensor_set(inp, normalized_state.data(), 0, ctx->max_state_dim * sizeof(float));

    // Execute graph
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }

    auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(ctx->backend, gf);
    auto t_compute_end = std::chrono::high_resolution_clock::now();

    LOG_INF("%s: graph build+alloc: %.2f ms, compute: %.2f ms\n", __func__,
            std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count(),
            std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());

    // Get output
    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "state_output");
    if (!out) {
        LOG_ERR("%s: failed to get output tensor\n", __func__);
        return false;
    }

    // Copy output to user buffer
    ggml_backend_tensor_get(out, output, 0, ctx->hidden_size * sizeof(float));

    return true;
}

// ============================================================================
// Free State Projector
// ============================================================================

void smolvla_state_proj_free(struct smolvla_state_proj * ctx) {
    if (!ctx) return;

    if (ctx->compute_alloc) {
        ggml_gallocr_free(ctx->compute_alloc);
    }
    if (ctx->params_buffer) {
        ggml_backend_buffer_free(ctx->params_buffer);
    }
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    if (ctx->ctx_data) {
        ggml_free(ctx->ctx_data);
    }
    if (ctx->ctx_gguf) {
        gguf_free(ctx->ctx_gguf);
    }

    delete ctx;
}

// ============================================================================
// Getters
// ============================================================================

int smolvla_state_proj_hidden_size(const struct smolvla_state_proj * ctx) {
    return ctx ? ctx->hidden_size : 0;
}

int smolvla_state_proj_max_state_dim(const struct smolvla_state_proj * ctx) {
    return ctx ? ctx->max_state_dim : 0;
}
