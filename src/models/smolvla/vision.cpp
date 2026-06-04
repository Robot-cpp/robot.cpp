/**
 * SmolVLA Vision (standalone in vision.cpp)
 *
 * Implements:
 *   image -> SigLIP ViT-12 (from mmproj GGUF) -> [1024, 768]
 *         -> SmolVLA connector (reshape pool + linear) -> [64, 960]
 *
 * No dependency on clip.cpp runtime/inference path.
 */

#include "vision.h"

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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Keep image loading isolated from clip.cpp.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(nullptr, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data());
}

static bool smolvla_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "False") != 0 &&
           strcmp(value, "FALSE") != 0;
}

static bool smolvla_vision_build_each_run_enabled() {
    bool build_each_run = true;

    const char * legacy = std::getenv("SMOLVLA_VISION_BUILD_EACH_RUN");
    if (legacy) {
        build_each_run = smolvla_env_flag_enabled("SMOLVLA_VISION_BUILD_EACH_RUN");
    }

    if (smolvla_env_flag_enabled("SMOLVLA_VISION_PERSISTENT_GRAPH")) {
        build_each_run = false;
    }

    return build_each_run;
}

static ggml_tensor * get_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }
    return t;
}

static uint32_t get_u32_or(gguf_context * gctx, const char * key, uint32_t fallback) {
    const int idx = gguf_find_key(gctx, key);
    return idx >= 0 ? gguf_get_val_u32(gctx, idx) : fallback;
}

static float get_f32_or(gguf_context * gctx, const char * key, float fallback) {
    const int idx = gguf_find_key(gctx, key);
    return idx >= 0 ? gguf_get_val_f32(gctx, idx) : fallback;
}

static void get_f32_arr3_or(gguf_context * gctx, const char * key, float out[3], const float fallback[3]) {
    const int idx = gguf_find_key(gctx, key);
    if (idx < 0 || gguf_get_arr_n(gctx, idx) < 3) {
        out[0] = fallback[0]; out[1] = fallback[1]; out[2] = fallback[2];
        return;
    }
    out[0] = gguf_get_arr_data(gctx, idx) ? ((const float *) gguf_get_arr_data(gctx, idx))[0] : fallback[0];
    out[1] = gguf_get_arr_data(gctx, idx) ? ((const float *) gguf_get_arr_data(gctx, idx))[1] : fallback[1];
    out[2] = gguf_get_arr_data(gctx, idx) ? ((const float *) gguf_get_arr_data(gctx, idx))[2] : fallback[2];
}

static const char * smolvla_buft_name(ggml_backend_buffer_type_t buft) {
    return buft ? ggml_backend_buft_name(buft) : "(null)";
}

static const char * smolvla_backend_name(ggml_backend_t backend) {
    return backend ? ggml_backend_name(backend) : "(null)";
}

static const char * smolvla_backend_device_name(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev ? ggml_backend_dev_name(dev) : "(null)";
}

static const char * smolvla_backend_reg_name(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? ggml_backend_reg_name(reg) : "(null)";
}

static void smolvla_vision_log_backend_plan(const smolvla_vision_ctx * ctx, int verbosity) {
    if (!ctx || verbosity < 1) {
        return;
    }

    for (size_t i = 0; i < ctx->backends.size(); ++i) {
        ggml_backend_t backend = ctx->backends[i];
        LOG_INF("%s: backend[%zu]: name=%s device=%s reg=%s default_buft=%s cpu=%d\n",
            __func__,
            i,
            smolvla_backend_name(backend),
            smolvla_backend_device_name(backend),
            smolvla_backend_reg_name(backend),
            smolvla_buft_name(ggml_backend_get_default_buffer_type(backend)),
            ggml_backend_is_cpu(backend) ? 1 : 0);
    }
}

static ggml_backend_buffer_type_t smolvla_vision_default_host_buffer_type() {
#if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)
    const char * target_reg_name =
#if defined(GGML_USE_CUDA)
        "CUDA";
#else
        "Metal";
#endif
// TODO: not tested yet, need a CUDA/Metal device with host buffer support
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_GPU_FULL) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg || strcmp(ggml_backend_reg_name(reg), target_reg_name) != 0) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft) {
            return buft;
        }
    }
#endif

    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t smolvla_vision_default_model_buffer_type() {
#ifdef GGML_USE_CUDA
    // TODO: do not tested yet, need a CUDA device with host buffer support
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_GPU_FULL) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg || strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        return buft ? buft : smolvla_vision_default_host_buffer_type();
    }
#elif defined(GGML_USE_METAL)
    // TODO: do not tested yet, need a Metal device with host buffer support 
    ggml_backend_buffer_type_t buft = ggml_backend_metal_buffer_type();
    if (buft) {
        return buft;
    }
#endif

    return smolvla_vision_default_host_buffer_type();
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

static void smolvla_free_scheduler_backends(std::vector<ggml_backend_t> & backends) {
    for (ggml_backend_t backend : backends) {
        if (backend) {
            ggml_backend_free(backend);
        }
    }
    backends.clear();
}

static bool smolvla_vision_init_backends(
    smolvla_vision_ctx * ctx,
    int verbosity
) {
    if (!ctx) {
        return false;
    }

    ctx->backend_cpu = nullptr;
    ctx->backends.clear();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend) {
            ctx->backends.push_back(backend);
        }
    }

    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        smolvla_free_scheduler_backends(ctx->backends);
        return false;
    }
    ctx->backends.push_back(ctx->backend_cpu);

    if (verbosity >= 1) {
        LOG_INF("%s: enabled %zu backend(s)\n", __func__, ctx->backends.size());
        smolvla_vision_log_backend_plan(ctx, verbosity);
    }

    return true;
}

static bool smolvla_vision_init_scheduler(
    smolvla_vision_ctx * ctx,
    int verbosity
) {
    if (!ctx || ctx->backends.empty() || !ctx->backend_cpu) {
        return false;
    }

    std::vector<ggml_backend_buffer_type_t> backend_buft;
    backend_buft.reserve(ctx->backends.size());
    const ggml_backend_buffer_type_t host_buft = smolvla_vision_default_host_buffer_type();
    for (ggml_backend_t backend : ctx->backends) {
        backend_buft.push_back(
            ggml_backend_is_cpu(backend)
                ? host_buft
                : ggml_backend_get_default_buffer_type(backend));
    }

    ctx->sched = ggml_backend_sched_new(
        ctx->backends.data(),
        backend_buft.data(),
        ctx->backends.size(),
        GGML_DEFAULT_GRAPH_SIZE,
        false,
        false);

    if (!ctx->sched) {
        smolvla_free_scheduler_backends(ctx->backends);
        ctx->backend_cpu = nullptr;
        return false;
    }

    if (verbosity >= 1) {
        LOG_INF("%s: host_buft=%s\n", __func__, smolvla_buft_name(host_buft));
        for (size_t i = 0; i < ctx->backends.size(); ++i) {
            LOG_INF("%s: scheduler backend_buft[%zu]: backend=%s buft=%s\n",
                __func__,
                i,
                smolvla_backend_name(ctx->backends[i]),
                smolvla_buft_name(backend_buft[i]));
        }
        LOG_INF("%s: enabled scheduler with %zu backend(s)\n", __func__, ctx->backends.size());
    }

    return true;
}

static void smolvla_vision_set_backend_threads(smolvla_vision_ctx * ctx, int n_threads) {
    if (!ctx || n_threads <= 0) {
        return;
    }

    auto apply_n_threads = [n_threads](ggml_backend_t backend) {
        if (!backend || !ggml_backend_is_cpu(backend)) {
            return;
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg) {
            return;
        }

        auto set_n_threads_fn = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads_fn) {
            set_n_threads_fn(backend, n_threads);
        }
    };

    if (!ctx->backends.empty()) {
        for (ggml_backend_t backend : ctx->backends) {
            apply_n_threads(backend);
        }
        return;
    }

    apply_n_threads(ctx->backend_cpu);
}

static void preprocess_loaded_image(
    const unsigned char * data,
    int w,
    int h,
    int target_size,
    const float mean[3],
    const float stdv[3],
    std::vector<float> & out_nchw
) {
    (void) mean;
    (void) stdv;

    // Match SmolVLA python resize_with_pad + img*2-1:
    // 1) ratio = max(w/target, h/target)
    // 2) bilinear resize to (floor(h/ratio), floor(w/ratio)), align_corners=False
    // 3) pad on LEFT and TOP with 0 (in [0,1] space)
    // 4) convert [0,1] -> [-1,1]
    const float ratio = std::max((float) w / target_size, (float) h / target_size);
    const int rw = std::max(1, (int) std::floor((float) w / ratio));
    const int rh = std::max(1, (int) std::floor((float) h / ratio));

    std::vector<float> resized(rw * rh * 3, 0.0f);
    const float scale_x = (float) w / rw;
    const float scale_y = (float) h / rh;
    for (int oy = 0; oy < rh; ++oy) {
        const float sy = (oy + 0.5f) * scale_y - 0.5f;
        const int y0f = (int) floorf(sy);
        const int y1f = y0f + 1;
        const int y0 = std::max(0, std::min(h - 1, y0f));
        const int y1 = std::max(0, std::min(h - 1, y1f));
        const float wy = sy - (float) y0f;
        for (int ox = 0; ox < rw; ++ox) {
            const float sx = (ox + 0.5f) * scale_x - 0.5f;
            const int x0f = (int) floorf(sx);
            const int x1f = x0f + 1;
            const int x0 = std::max(0, std::min(w - 1, x0f));
            const int x1 = std::max(0, std::min(w - 1, x1f));
            const float wx = sx - (float) x0f;
            for (int ch = 0; ch < 3; ++ch) {
                const float p00 = data[(y0 * w + x0) * 3 + ch] / 255.0f;
                const float p01 = data[(y0 * w + x1) * 3 + ch] / 255.0f;
                const float p10 = data[(y1 * w + x0) * 3 + ch] / 255.0f;
                const float p11 = data[(y1 * w + x1) * 3 + ch] / 255.0f;
                const float p0 = p00 + wx * (p01 - p00);
                const float p1 = p10 + wx * (p11 - p10);
                resized[(oy * rw + ox) * 3 + ch] = p0 + wy * (p1 - p0);
            }
        }
    }

    const int pad_w = std::max(0, target_size - rw);
    const int pad_h = std::max(0, target_size - rh);
    out_nchw.assign(3 * target_size * target_size, -1.0f); // padded 0 -> after *2-1 => -1
    for (int y = 0; y < rh; ++y) {
        for (int x = 0; x < rw; ++x) {
            const int tx = x + pad_w; // left pad
            const int ty = y + pad_h; // top pad
            for (int ch = 0; ch < 3; ++ch) {
                float v01 = resized[(y * rw + x) * 3 + ch];
                // SmolVLA python path does explicit img = img * 2 - 1.
                // Do not reuse CLIP mean/std normalization here.
                float v = v01 * 2.0f - 1.0f;
                out_nchw[ch * target_size * target_size + ty * target_size + tx] = v;
            }
        }
    }
}

static bool load_image_and_preprocess(
    const char * image_path,
    int target_size,
    const float mean[3],
    const float stdv[3],
    std::vector<float> & out_nchw
) {
    int w = 0, h = 0, c = 0;
    unsigned char * data = stbi_load(image_path, &w, &h, &c, 3);
    if (!data) {
        LOG_ERR("%s: failed to load image: %s\n", __func__, image_path);
        return false;
    }

    preprocess_loaded_image(data, w, h, target_size, mean, stdv, out_nchw);
    stbi_image_free(data);
    return true;
}

static bool load_image_and_preprocess_bytes(
    const unsigned char * image_bytes,
    int image_len,
    int target_size,
    const float mean[3],
    const float stdv[3],
    std::vector<float> & out_nchw
) {
    if (!image_bytes || image_len <= 0) {
        LOG_ERR("%s: invalid args\n", __func__);
        return false;
    }

    int w = 0, h = 0, c = 0;
    unsigned char * data = stbi_load_from_memory(image_bytes, image_len, &w, &h, &c, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes (len=%d)\n", __func__, image_len);
        return false;
    }

    preprocess_loaded_image(data, w, h, target_size, mean, stdv, out_nchw);
    stbi_image_free(data);
    return true;
}

static ggml_tensor * connector_build_op(
    smolvla_vision_ctx * ctx,
    ggml_context * ctx0,
    ggml_tensor * vit_out
) {
    const int hs = ctx->hidden_size;
    const int n_patch = ctx->num_patches;
    const int out_tok = ctx->output_tokens;
    const int in_feat = ctx->connector_in_features;    // 12288
    const int scale = (int) std::sqrt((double) (n_patch / out_tok)); // 4
    const int in_side = (int) std::sqrt((double) n_patch);           // 32
    const int out_side = in_side / scale;                            // 8
    GGML_ASSERT(in_side * in_side == n_patch);
    GGML_ASSERT(out_side * out_side == out_tok);

    // Pure-operator pixel_shuffle equivalent for SmolVLMConnector:
    // vit [hs, 1024] where patch index = y*32 + x
    // -> reshape to [hs*4, ox=8, py=4, oy=8], permute to [hs*4, py, ox, oy]
    // -> flatten to [in_feat=12288, out_tok=64]
    ggml_tensor * x = ggml_reshape_4d(ctx0, vit_out, hs * scale, out_side, scale, out_side);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(ctx0, x, in_feat, out_tok, 1);

    return ggml_mul_mat(ctx0, ctx->connector_w, x);
}

// Build a single persistent vision graph: SigLIP ViT + connector.
static ggml_cgraph * vision_build_graph(smolvla_vision_ctx * ctx) {
    const int image_size = ctx->image_size;
    const int patch = ctx->patch_size;
    const int n_patches = ctx->num_patches;
    const int hs = ctx->hidden_size;
    const int n_head = ctx->n_heads;
    const int d_head = hs / n_head;
    const float eps = ctx->eps;

    ggml_init_params ip = {
        ctx->buf_compute_meta.size(),
        ctx->buf_compute_meta.data(),
        true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, 1);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    ggml_tensor * patch_w = ctx->patch_embd_w->type == GGML_TYPE_F32
        ? ctx->patch_embd_w
        : ggml_cast(ctx0, ctx->patch_embd_w, GGML_TYPE_F32);
    ggml_tensor * x = ggml_conv_2d(ctx0, patch_w, inp, patch, patch, 0, 0, 1, 1);
    x = ggml_reshape_3d(ctx0, x, n_patches, hs, 1);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
    x = ggml_add(ctx0, x, ctx->patch_embd_b);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor * pos = ggml_get_rows(ctx0, ctx->pos_embd, positions);
    x = ggml_add(ctx0, x, pos);

    for (int i = 0; i < ctx->n_layers; ++i) {
        const auto & l = ctx->layers[i];
        ggml_tensor * residual = x;

        // LN1
        ggml_tensor * cur = ggml_norm(ctx0, x, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, l.ln1_w), l.ln1_b);

        // Self-attention
        ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, l.q_w, cur), l.q_b);
        ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, l.k_w, cur), l.k_b);
        ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, l.v_w, cur), l.v_b);
        Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrtf((float) d_head));
        Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, n_patches, 1);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(ctx0, Q, d_head, n_patches, n_head);
        K = ggml_reshape_4d(ctx0, K, d_head, n_head, n_patches, 1);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(ctx0, K, d_head, n_patches, n_head);
        V = ggml_reshape_4d(ctx0, V, d_head, n_head, n_patches, 1);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
        V = ggml_reshape_3d(ctx0, V, n_patches, d_head, n_head);
        ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_soft_max_inplace(ctx0, KQ);
        ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_reshape_4d(ctx0, KQV, d_head, n_patches, n_head, 1);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_3d(ctx0, KQV, hs, n_patches, 1);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.o_w, cur), l.o_b);

        x = ggml_add(ctx0, residual, cur);

        // LN2 + MLP + residual
        residual = x;
        cur = ggml_norm(ctx0, x, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, l.ln2_w), l.ln2_b);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn1_w, cur), l.ffn1_b);
        cur = ggml_gelu_inplace(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn2_w, cur), l.ffn2_b);
        x = ggml_add(ctx0, residual, cur);
    }

    x = ggml_norm(ctx0, x, eps);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, ctx->post_ln_w), ctx->post_ln_b);

    ggml_set_name(x, "vision_output");

    ggml_tensor * y = connector_build_op(ctx, ctx0, x);
    ggml_set_name(y, "connector_output");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_free(ctx0);
    return gf;
}

smolvla_vision_ctx * smolvla_vision_load(const char * mmproj_path, int verbosity) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * gctx = gguf_init_from_file(mmproj_path, params);
    if (!gctx) {
        LOG_ERR("%s: failed to load vision GGUF: %s\n", __func__, mmproj_path);
        return nullptr;
    }

    auto * ctx = new smolvla_vision_ctx();
    memset(ctx->image_mean, 0, sizeof(ctx->image_mean));
    memset(ctx->image_std, 0, sizeof(ctx->image_std));
    ctx->ctx_gguf = gctx;
    ctx->ctx_data = nullptr;
    ctx->backend_cpu = nullptr;
    ctx->sched = nullptr;
    ctx->graph = nullptr;
    ctx->graph_inp_raw = nullptr;
    ctx->graph_positions = nullptr;
    ctx->graph_vision_output = nullptr;
    ctx->graph_connector_output = nullptr;
    ctx->patch_embd_w = ctx->patch_embd_b = ctx->pos_embd = ctx->post_ln_w = ctx->post_ln_b = nullptr;
    ctx->connector_w = nullptr;

    ctx->image_size = (int) get_u32_or(gctx, "smolvla.vision.image_size", 512);
    ctx->patch_size = (int) get_u32_or(gctx, "smolvla.vision.patch_size", 16);
    ctx->hidden_size = (int) get_u32_or(gctx, "smolvla.vision.hidden_size", 768);
    ctx->intermediate_size = (int) get_u32_or(gctx, "smolvla.vision.intermediate_size", 3072);
    ctx->n_heads = (int) get_u32_or(gctx, "smolvla.vision.num_heads", 12);
    ctx->n_layers = (int) get_u32_or(gctx, "smolvla.vision.num_layers", 12);
    ctx->eps = get_f32_or(gctx, "smolvla.vision.layer_norm_eps", 1e-6f);
    ctx->num_patches = (ctx->image_size / ctx->patch_size) * (ctx->image_size / ctx->patch_size);
    const float default_mean[3] = {0.5f, 0.5f, 0.5f};
    const float default_std[3] = {0.5f, 0.5f, 0.5f};
    get_f32_arr3_or(gctx, "smolvla.vision.image_mean", ctx->image_mean, default_mean);
    get_f32_arr3_or(gctx, "smolvla.vision.image_std", ctx->image_std, default_std);

    const ggml_backend_buffer_type_t model_buft = smolvla_vision_default_model_buffer_type();
    if (verbosity >= 1) {
        LOG_INF("%s: resolved model_buft=%s\n", __func__, smolvla_buft_name(model_buft));
    }

    if (!smolvla_vision_init_backends(ctx, verbosity)) {
        LOG_ERR("%s: failed to initialize vision backends\n", __func__);
        smolvla_vision_free(ctx);
        ggml_free(meta);
        return nullptr;
    }

    const int n_tensors = gguf_get_n_tensors(gctx);
    const size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 1);
    ggml_init_params gparams = { ctx_size, nullptr, true };
    ctx->ctx_data = ggml_init(gparams);
    if (!ctx->ctx_data) {
        LOG_ERR("%s: failed to create ctx_data\n", __func__);
        smolvla_vision_free(ctx);
        ggml_free(meta);
        return nullptr;
    }
    ctx->ctxs.push_back(ctx->ctx_data);

    // Duplicate tensor metadata into ctx_data.
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        ggml_tensor * t = ggml_dup_tensor(ctx->ctx_data, mt);
        ggml_set_name(t, name);
    }

    // Allocate backend buffer and load tensor bytes.
    // TODO: model_buft still assumes the whole model lives on a single target device.
    ggml_backend_buffer_t model_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx->ctx_data, model_buft);
    if (!model_buf) {
        LOG_ERR("%s: failed to allocate model buffer\n", __func__);
        smolvla_vision_free(ctx);
        ggml_free(meta);
        return nullptr;
    }

    if (verbosity >= 1) {
        LOG_INF("%s: model_buf=%s is_host=%d\n",
            __func__,
            ggml_backend_buffer_name(model_buf),
            ggml_backend_buffer_is_host(model_buf) ? 1 : 0);
    }
    
    ctx->bufs.push_back(model_buf);
    std::ifstream fin(mmproj_path, std::ios::binary);
    if (!fin) {
        LOG_ERR("%s: failed to open %s\n", __func__, mmproj_path);
        smolvla_vision_free(ctx);
        ggml_free(meta);
        return nullptr;
    }
    std::vector<uint8_t> tmp;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(ctx->ctx_data, name);
        const size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, i);
        fin.seekg(off, std::ios::beg);
        const int nbytes = (int) ggml_nbytes(t);
        if (ggml_backend_buffer_is_host(model_buf)) {
            fin.read((char *) t->data, nbytes);
        } else {
            tmp.resize(nbytes);
            fin.read((char *) tmp.data(), nbytes);
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
    }
    fin.close();
    ggml_free(meta);

    // Bind tensors.
    try {
        ctx->patch_embd_w = get_tensor(ctx->ctx_data, "v.patch_embd.weight");
        ctx->patch_embd_b = get_tensor(ctx->ctx_data, "v.patch_embd.bias");
        ctx->pos_embd = get_tensor(ctx->ctx_data, "v.position_embd.weight");
        ctx->post_ln_w = get_tensor(ctx->ctx_data, "v.post_ln.weight");
        ctx->post_ln_b = get_tensor(ctx->ctx_data, "v.post_ln.bias");
        ctx->layers.resize(ctx->n_layers);
        for (int i = 0; i < ctx->n_layers; ++i) {
            auto & l = ctx->layers[i];
            l.ln1_w = get_tensor(ctx->ctx_data, format("v.blk.%d.ln1.weight", i));
            l.ln1_b = get_tensor(ctx->ctx_data, format("v.blk.%d.ln1.bias", i));
            l.q_w = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_q.weight", i));
            l.q_b = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_q.bias", i));
            l.k_w = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_k.weight", i));
            l.k_b = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_k.bias", i));
            l.v_w = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_v.weight", i));
            l.v_b = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_v.bias", i));
            l.o_w = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_out.weight", i));
            l.o_b = get_tensor(ctx->ctx_data, format("v.blk.%d.attn_out.bias", i));
            l.ln2_w = get_tensor(ctx->ctx_data, format("v.blk.%d.ln2.weight", i));
            l.ln2_b = get_tensor(ctx->ctx_data, format("v.blk.%d.ln2.bias", i));
            l.ffn1_w = get_tensor(ctx->ctx_data, format("v.blk.%d.ffn_down.weight", i));
            l.ffn1_b = get_tensor(ctx->ctx_data, format("v.blk.%d.ffn_down.bias", i));
            l.ffn2_w = get_tensor(ctx->ctx_data, format("v.blk.%d.ffn_up.weight", i));
            l.ffn2_b = get_tensor(ctx->ctx_data, format("v.blk.%d.ffn_up.bias", i));
        }
        ggml_tensor * conn = get_tensor(ctx->ctx_data, "mm.0.weight");
        // Converter writes numpy shape [out_features, in_features] = [960, 12288].
        // In ggml tensor dims this appears as ne0=in_features, ne1=out_features.
        const int64_t in = conn->ne[0];
        const int64_t out = conn->ne[1];
        ctx->connector_out_features = (int) out;
        ctx->connector_in_features = (int) in;
        ctx->pool_num = 16;
        ctx->output_tokens = ctx->num_patches / ctx->pool_num;
        ctx->connector_w = conn;
        ctx->proj_dim = ctx->connector_out_features;
    } catch (const std::exception & e) {
        LOG_ERR("%s: tensor bind failed: %s", __func__, e.what());
        smolvla_vision_free(ctx);
        return nullptr;
    }

    if (!smolvla_vision_init_scheduler(ctx, verbosity)) {
        LOG_ERR("%s: failed to initialize vision scheduler\n", __func__);
        smolvla_vision_free(ctx);
        return nullptr;
    }

    ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    const bool build_each_run = smolvla_vision_build_each_run_enabled();
    ggml_cgraph * reserve_graph = vision_build_graph(ctx);
    const bool reserve_ok = reserve_graph && ggml_backend_sched_reserve(ctx->sched, reserve_graph);
    if (!reserve_ok) {
        LOG_ERR("%s: failed to reserve merged vision graph\n", __func__);
        smolvla_vision_free(ctx);
        return nullptr;
    }

    if (!build_each_run) {
        ctx->graph = reserve_graph;
        ggml_backend_sched_reset(ctx->sched);
        const bool alloc_ok = ggml_backend_sched_alloc_graph(ctx->sched, ctx->graph);
        if (!alloc_ok) {
            LOG_ERR("%s: failed to allocate merged vision graph\n", __func__);
            smolvla_vision_free(ctx);
            return nullptr;
        }

        ctx->graph_inp_raw = ggml_graph_get_tensor(ctx->graph, "inp_raw");
        ctx->graph_positions = ggml_graph_get_tensor(ctx->graph, "positions");
        ctx->graph_vision_output = ggml_graph_get_tensor(ctx->graph, "vision_output");
        ctx->graph_connector_output = ggml_graph_get_tensor(ctx->graph, "connector_output");
        if (!ctx->graph_inp_raw || !ctx->graph_positions || !ctx->graph_connector_output) {
            LOG_ERR("%s: failed to bind merged vision graph tensors\n", __func__);
            smolvla_vision_free(ctx);
            return nullptr;
        }
    }

    ctx->patch_positions.resize(ctx->num_patches);
    for (int i = 0; i < ctx->num_patches; ++i) {
        ctx->patch_positions[i] = i;
    }
    if (ctx->graph_positions) {
        ggml_backend_tensor_set(ctx->graph_positions, ctx->patch_positions.data(), 0,
                                ctx->patch_positions.size() * sizeof(int32_t));
    }

    if (verbosity >= 1) {
        LOG_INF("%s: vision graph mode: %s\n", __func__, build_each_run ? "build-each-run" : "persistent");
        LOG_INF("%s: standalone SigLIP loaded: image=%d patch=%d hidden=%d layers=%d patches=%d\n",
            __func__, ctx->image_size, ctx->patch_size, ctx->hidden_size, ctx->n_layers, ctx->num_patches);
        LOG_INF("%s: connector loaded: in=%d out=%d tokens=%d\n",
            __func__, ctx->connector_in_features, ctx->connector_out_features, ctx->output_tokens);
    }
    return ctx;
}

void smolvla_vision_free(smolvla_vision_ctx * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    smolvla_free_model_buffers(ctx->bufs);
    smolvla_free_model_contexts(ctx->ctxs);
    smolvla_free_scheduler_backends(ctx->backends);
    ctx->backend_cpu = nullptr;
    if (ctx->ctx_gguf) gguf_free(ctx->ctx_gguf);
    delete ctx;
}

static std::vector<float> smolvla_vision_encode_preprocessed(
    smolvla_vision_ctx * ctx,
    const std::vector<float> & inp_nchw,
    int n_threads
) {
    std::vector<float> result;
    if (!ctx || inp_nchw.empty()) {
        LOG_ERR("%s: invalid args\n", __func__);
        return result;
    }
    if (!ctx->sched) {
        LOG_ERR("%s: scheduler is required\n", __func__);
        return result;
    }
    const bool build_each_run = smolvla_vision_build_each_run_enabled();
    if (!build_each_run && (!ctx->graph || !ctx->graph_inp_raw || !ctx->graph_connector_output)) {
        LOG_ERR("%s: merged vision graph is not initialized\n", __func__);
        return result;
    }

    ggml_cgraph * graph = ctx->graph;
    ggml_tensor * inp_raw = ctx->graph_inp_raw;
    ggml_tensor * positions = ctx->graph_positions;
    ggml_tensor * connector_out = ctx->graph_connector_output;

    double build_alloc_ms = 0.0;
    if (build_each_run) {
        auto t_build_start = std::chrono::high_resolution_clock::now();
        graph = vision_build_graph(ctx);
        ggml_backend_sched_reset(ctx->sched);
        const bool alloc_ok = graph && ggml_backend_sched_alloc_graph(ctx->sched, graph);
        if (!alloc_ok) {
            LOG_ERR("%s: failed to build+alloc merged vision graph for this run\n", __func__);
            return result;
        }
        auto t_build_end = std::chrono::high_resolution_clock::now();
        build_alloc_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();

        inp_raw = ggml_graph_get_tensor(graph, "inp_raw");
        positions = ggml_graph_get_tensor(graph, "positions");
        connector_out = ggml_graph_get_tensor(graph, "connector_output");
        if (!inp_raw || !positions || !connector_out) {
            LOG_ERR("%s: failed to bind per-run merged vision graph tensors\n", __func__);
            return result;
        }
    }

    ggml_backend_tensor_set(inp_raw, inp_nchw.data(), 0, inp_nchw.size() * sizeof(float));
    ggml_backend_tensor_set(positions, ctx->patch_positions.data(), 0,
                            ctx->patch_positions.size() * sizeof(int32_t));

    smolvla_vision_set_backend_threads(ctx, n_threads);
    auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_backend_sched_graph_compute(ctx->sched, graph);
    auto t_compute_end = std::chrono::high_resolution_clock::now();

    if (build_each_run) {
        LOG_INF("%s: merged vision graph build+alloc: %.2f ms, compute: %.2f ms\n", __func__,
                build_alloc_ms,
                std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());
    } else {
        LOG_INF("%s: merged vision graph compute: %.2f ms\n", __func__,
                std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());
    }

    result.resize(ctx->output_tokens * ctx->proj_dim);
    ggml_backend_tensor_get(connector_out, result.data(), 0, result.size() * sizeof(float));

    return result;
}

std::vector<float> smolvla_vision_encode_file(smolvla_vision_ctx * ctx, const char * image_path, int n_threads) {
    if (!ctx || !image_path) {
        LOG_ERR("%s: invalid args\n", __func__);
        return {};
    }

    std::vector<float> inp_nchw;
    if (!load_image_and_preprocess(image_path, ctx->image_size, ctx->image_mean, ctx->image_std, inp_nchw)) {
        return {};
    }

    return smolvla_vision_encode_preprocessed(ctx, inp_nchw, n_threads);
}

std::vector<float> smolvla_vision_encode_bytes(
    smolvla_vision_ctx * ctx,
    const unsigned char * image_bytes,
    int image_len,
    int n_threads
) {
    if (!ctx || !image_bytes || image_len <= 0) {
        LOG_ERR("%s: invalid args\n", __func__);
        return {};
    }

    std::vector<float> inp_nchw;
    if (!load_image_and_preprocess_bytes(
            image_bytes,
            image_len,
            ctx->image_size,
            ctx->image_mean,
            ctx->image_std,
            inp_nchw)) {
        return {};
    }

    return smolvla_vision_encode_preprocessed(ctx, inp_nchw, n_threads);
}

std::vector<float> smolvla_vision_encode_constant(smolvla_vision_ctx * ctx, float pixel_value, int n_threads) {
    if (!ctx) {
        LOG_ERR("%s: invalid args\n", __func__);
        return {};
    }

    std::vector<float> inp_nchw(3 * ctx->image_size * ctx->image_size, pixel_value);
    return smolvla_vision_encode_preprocessed(ctx, inp_nchw, n_threads);
}

std::vector<float> smolvla_vision_encode_raw(
    smolvla_vision_ctx * ctx,
    const uint8_t * data,
    int width,
    int height,
    int channels,
    int stride_bytes,
    int n_threads
) {
    if (!ctx || !data || width <= 0 || height <= 0 || channels != 3) {
        LOG_ERR("%s: invalid raw RGB args (data=%p width=%d height=%d channels=%d)\n",
                __func__, (const void *) data, width, height, channels);
        return {};
    }

    const int tight_stride = width * channels;
    if (stride_bytes <= 0) {
        stride_bytes = tight_stride;
    }
    if (stride_bytes < tight_stride) {
        LOG_ERR("%s: invalid stride_bytes=%d for width=%d channels=%d\n",
                __func__, stride_bytes, width, channels);
        return {};
    }

    std::vector<uint8_t> tight;
    const uint8_t * packed = data;
    if (stride_bytes != tight_stride) {
        tight.resize((size_t) tight_stride * (size_t) height);
        for (int y = 0; y < height; ++y) {
            const uint8_t * src = data + (size_t) y * (size_t) stride_bytes;
            uint8_t * dst = tight.data() + (size_t) y * (size_t) tight_stride;
            std::memcpy(dst, src, (size_t) tight_stride);
        }
        packed = tight.data();
    }

    std::vector<float> inp_nchw;
    preprocess_loaded_image(packed, width, height, ctx->image_size, ctx->image_mean, ctx->image_std, inp_nchw);
    return smolvla_vision_encode_preprocessed(ctx, inp_nchw, n_threads);
}

int smolvla_vision_embd_size(const smolvla_vision_ctx * ctx) {
    if (!ctx) return 0;
    return ctx->output_tokens * ctx->proj_dim;
}

int smolvla_vision_n_tokens(const smolvla_vision_ctx * ctx) {
    return ctx ? ctx->output_tokens : 0;
}
