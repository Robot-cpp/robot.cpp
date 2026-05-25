/**
 * SmolVLA State Projector Implementation (proprio.cpp style)
 * 
 * Uses ggml compute graph for: normalize (mean_std) → Linear(32, 960)
 */

#include "state_proj.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
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
#include <cstdarg>
#include <fstream>
#include <sstream>
#include <climits>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// ============================================================================
// Logging (similar to bitvla)
// ============================================================================

#define LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)

// ============================================================================
// Helper functions (from proprio.cpp)
// ============================================================================

static int get_key_idx(const gguf_context * ctx, const char * key) {
    int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        LOG_ERR("%s: key '%s' not found in GGUF file\n", __func__, key);
    }
    return idx;
}

static uint32_t get_u32(const gguf_context * ctx, const char * key) {
    int idx = get_key_idx(ctx, key);
    if (idx < 0) return 0;
    return gguf_get_val_u32(ctx, idx);
}

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }
    return cur;
}

// Read float array from tensor data (must be called AFTER weights are loaded to backend)
static std::vector<float> read_tensor_f32_from_backend(struct ggml_context * ctx, ggml_backend_buffer_t buffer, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) return {};
    
    int64_t n = ggml_nelements(t);
    std::vector<float> result(n);
    
    // Read from backend buffer
    size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw_data(nbytes);
    ggml_backend_tensor_get(t, raw_data.data(), 0, nbytes);
    
    if (t->type == GGML_TYPE_F32) {
        memcpy(result.data(), raw_data.data(), n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)raw_data.data();
        for (int64_t i = 0; i < n; i++) {
            result[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        LOG_ERR("%s: unsupported tensor type %d for %s\n", __func__, t->type, name);
        return {};
    }
    
    return result;
}

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

struct smolvla_state_proj * smolvla_state_proj_load(const char * fname, int verbosity) {
    struct ggml_context * meta = nullptr;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(fname, params);
    if (!gguf_ctx) {
        LOG_ERR("%s: failed to load state_proj GGUF from %s\n", __func__, fname);
        return nullptr;
    }

    if (verbosity >= 1) {
        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        const int n_kv = gguf_get_n_kv(gguf_ctx);
        LOG_INF("%s: loaded state_proj GGUF with %d tensors, %d kv pairs\n", __func__, n_tensors, n_kv);
    }

    // Create context
    smolvla_state_proj * ctx = new smolvla_state_proj();
    ctx->ctx_gguf = gguf_ctx;

    // Read metadata
    int idx_state_dim = gguf_find_key(gguf_ctx, "smolvla.state_dim");
    if (idx_state_dim >= 0) {
        ctx->state_dim = gguf_get_val_u32(gguf_ctx, idx_state_dim);
    }

    // Get weight shape to determine dimensions
    struct ggml_tensor * weight_meta = ggml_get_tensor(meta, "smolvla.state_proj.weight");
    if (weight_meta) {
        ctx->max_state_dim = (int)weight_meta->ne[0];
        ctx->hidden_size = (int)weight_meta->ne[1];
    }

    if (verbosity >= 1) {
        LOG_INF("%s: state_dim = %d, max_state_dim = %d, hidden_size = %d\n", 
                __func__, ctx->state_dim, ctx->max_state_dim, ctx->hidden_size);
    }

    // NOTE: norm stats will be loaded AFTER weights are loaded to backend

    // Calculate model size
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    size_t model_size = 0;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * cur = ggml_get_tensor(meta, name);
        model_size += ggml_nbytes(cur);
        if (verbosity >= 2) {
            LOG_INF("%s:   tensor[%d]: %s, size=%zu\n", __func__, i, name, ggml_nbytes(cur));
        }
    }

    if (verbosity >= 1) {
        LOG_INF("%s: model size = %.2f KB\n", __func__, model_size / 1024.0);
    }

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

    // Create data context for tensors
    size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 1);
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx->ctx_data = ggml_init(ggml_params);
    if (!ctx->ctx_data) {
        LOG_ERR("%s: failed to create ggml context for state_proj data\n", __func__);
        smolvla_state_proj_free(ctx);
        gguf_free(gguf_ctx);
        return nullptr;
    }

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        LOG_ERR("cannot open model file for loading tensors\n");
        smolvla_state_proj_free(ctx);
        gguf_free(gguf_ctx);
        return nullptr;
    }

    // Add tensors to context
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta, name);
        struct ggml_tensor * cur = ggml_dup_tensor(ctx->ctx_data, meta_tensor);
        ggml_set_name(cur, name);
    }

    // Allocate buffer and load weights
    std::vector<uint8_t> read_buf;
    ctx->params_buffer = ggml_backend_alloc_ctx_tensors(ctx->ctx_data, ctx->backend);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx->ctx_data, name);
        const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
        fin.seekg(offset, std::ios::beg);
        if (!fin) {
            LOG_ERR("%s: failed to seek for tensor %s\n", __func__, name);
            smolvla_state_proj_free(ctx);
            gguf_free(gguf_ctx);
            return nullptr;
        }
        int num_bytes = ggml_nbytes(cur);
        if (ggml_backend_buffer_is_host(ctx->params_buffer)) {
            fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
        } else {
            read_buf.resize(num_bytes);
            fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
            ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
        }
    }
    fin.close();

    // Get weight tensors
    ctx->weight = get_tensor(ctx->ctx_data, "smolvla.state_proj.weight");
    ctx->bias = get_tensor(ctx->ctx_data, "smolvla.state_proj.bias");

    // Load normalization statistics from tensor data (AFTER weights are loaded)
    ctx->norm_mean = read_tensor_f32_from_backend(ctx->ctx_data, ctx->params_buffer, "smolvla.norm.observation_state_mean");
    ctx->norm_std = read_tensor_f32_from_backend(ctx->ctx_data, ctx->params_buffer, "smolvla.norm.observation_state_std");

    if (!ctx->norm_mean.empty() && !ctx->norm_std.empty() &&
        ctx->norm_mean.size() == ctx->norm_std.size()) {
        ctx->has_norm_stats = true;
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

    // Free meta context
    ggml_free(meta);

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
