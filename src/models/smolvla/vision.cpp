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
#include "models/ggml_backend.h"
#include "models/gguf_loader.h"
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
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

static bool gguf_string_array(gguf_context * gguf, const char * key, std::vector<std::string> & out) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        return false;
    }
    if (gguf_get_kv_type(gguf, idx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("invalid GGUF string-array metadata type: ") + key);
    }
    const size_t count = gguf_get_arr_n(gguf, idx);
    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back(gguf_get_arr_str(gguf, idx, i));
    }
    return true;
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

static ggml_backend_t smolvla_vision_first_non_cpu_backend(const smolvla_vision_ctx * ctx) {
    if (!ctx) {
        return nullptr;
    }
    for (ggml_backend_t backend : ctx->backends) {
        if (backend && !ggml_backend_is_cpu(backend)) {
            return backend;
        }
    }
    return nullptr;
}

static bool smolvla_vision_backend_is_accel(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void smolvla_vision_bind_preprocess_tensors_to_backend(
    ggml_backend_sched_t sched,
    ggml_cgraph * graph,
    ggml_backend_t backend
) {
    if (!sched || !graph || !backend) {
        return;
    }
    const char * names[] = {
        "inp_raw_hwc_f32",
        "raw_scaled",
        "raw_planar",
        "raw_resized",
        "raw_padded",
        "inp_raw_preprocessed",
    };
    for (const char * name : names) {
        if (ggml_tensor * tensor = ggml_graph_get_tensor(graph, name)) {
            ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
        }
    }
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

static ggml_tensor * vision_build_vit_connector_graph(
    smolvla_vision_ctx * ctx,
    ggml_context * ctx0,
    ggml_tensor * inp
) {
    const int patch = ctx->patch_size;
    const int n_patches = ctx->num_patches;
    const int hs = ctx->hidden_size;
    const int n_head = ctx->n_heads;
    const int d_head = hs / n_head;
    const float eps = ctx->eps;

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
    return y;
}

// Build graph for an already-preprocessed image tensor: ViT + connector.
static ggml_cgraph * vision_build_preprocessed_graph(smolvla_vision_ctx * ctx) {
    const int image_size = ctx->image_size;

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

    ggml_tensor * y = vision_build_vit_connector_graph(ctx, ctx0, inp);
    ggml_build_forward_expand(gf, y);
    ggml_free(ctx0);
    return gf;
}

// Build the complete raw-RGB vision graph: preprocess + ViT + connector.
static ggml_cgraph * vision_build_graph(
    smolvla_vision_ctx * ctx,
    int width,
    int height,
    int resized_width,
    int resized_height,
    int pad_width,
    int pad_height
) {
    ggml_init_params ip = {
        ctx->buf_compute_meta.size(),
        ctx->buf_compute_meta.data(),
        true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 3, width, height, 1);
    ggml_set_name(raw, "inp_raw_hwc_f32");
    ggml_set_input(raw);

    ggml_tensor * x = ggml_scale(ctx0, raw, 1.0f / 255.0f);
    ggml_set_name(x, "raw_scaled");
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 2, 0, 1, 3));
    ggml_set_name(x, "raw_planar");
    x = ggml_interpolate(
        ctx0,
        x,
        resized_width,
        resized_height,
        3,
        1,
        GGML_SCALE_MODE_BILINEAR);
    ggml_set_name(x, "raw_resized");
    x = ggml_pad_ext(ctx0, x, pad_width, 0, pad_height, 0, 0, 0, 0, 0);
    ggml_set_name(x, "raw_padded");
    x = ggml_scale_bias(ctx0, x, 2.0f, -1.0f);
    ggml_set_name(x, "inp_raw_preprocessed");

    ggml_tensor * y = vision_build_vit_connector_graph(ctx, ctx0, x);
    ggml_build_forward_expand(gf, y);
    ggml_free(ctx0);
    return gf;
}

class smolvla_vision_loader : public gguf_loader {
public:
    smolvla_vision_loader(
        smolvla_vision_ctx * ctx,
        int verbosity)
        : ctx_(ctx),
          verbosity_(verbosity) {
    }

protected:
    bool parse_metadata(gguf_context * gguf) override {
        ctx_->image_size        = (int) this->u32_or(gguf, "smolvla.vision.image_size", 512);
        ctx_->patch_size        = (int) this->u32_or(gguf, "smolvla.vision.patch_size", 16);
        ctx_->hidden_size       = (int) this->u32_or(gguf, "smolvla.vision.hidden_size", 768);
        ctx_->intermediate_size = (int) this->u32_or(gguf, "smolvla.vision.intermediate_size", 3072);
        ctx_->n_heads           = (int) this->u32_or(gguf, "smolvla.vision.num_heads", 12);
        ctx_->n_layers          = (int) this->u32_or(gguf, "smolvla.vision.num_layers", 12);
        ctx_->eps               = this->f32_or(gguf, "smolvla.vision.layer_norm_eps", 1e-6f);
        ctx_->num_patches       = (ctx_->image_size / ctx_->patch_size) * (ctx_->image_size / ctx_->patch_size);

        const float default_mean[3] = {0.5f, 0.5f, 0.5f};
        const float default_std[3] = {0.5f, 0.5f, 0.5f};
        this->f32_arr3_or(gguf, "smolvla.vision.image_mean", ctx_->image_mean, default_mean);
        this->f32_arr3_or(gguf, "smolvla.vision.image_std", ctx_->image_std, default_std);

        if (!gguf_string_array(gguf, "smolvla.image_keys", ctx_->image_keys) || ctx_->image_keys.empty()) {
            this->set_error("missing required GGUF metadata: smolvla.image_keys");
            return false;
        }

        if (verbosity_ >= 1) {
            LOG_INF("%s: vision config: image=%d patch=%d hidden=%d layers=%d heads=%d\n",
                    __func__,
                    ctx_->image_size,
                    ctx_->patch_size,
                    ctx_->hidden_size,
                    ctx_->n_layers,
                    ctx_->n_heads);
            for (size_t i = 0; i < ctx_->image_keys.size(); ++i) {
                LOG_INF("%s: image_key[%zu]=%s\n", __func__, i, ctx_->image_keys[i].c_str());
            }
        }

        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        ctx_->patch_embd_w = this->require_tensor(ctx_data, "v.patch_embd.weight");
        ctx_->patch_embd_b = this->require_tensor(ctx_data, "v.patch_embd.bias");
        ctx_->pos_embd = this->require_tensor(ctx_data, "v.position_embd.weight");
        ctx_->post_ln_w = this->require_tensor(ctx_data, "v.post_ln.weight");
        ctx_->post_ln_b = this->require_tensor(ctx_data, "v.post_ln.bias");

        ctx_->layers.resize(ctx_->n_layers);
        for (int i = 0; i < ctx_->n_layers; ++i) {
            auto & l = ctx_->layers[i];
            l.ln1_w = this->require_tensor(ctx_data, format("v.blk.%d.ln1.weight", i));
            l.ln1_b = this->require_tensor(ctx_data, format("v.blk.%d.ln1.bias", i));
            l.q_w = this->require_tensor(ctx_data, format("v.blk.%d.attn_q.weight", i));
            l.q_b = this->require_tensor(ctx_data, format("v.blk.%d.attn_q.bias", i));
            l.k_w = this->require_tensor(ctx_data, format("v.blk.%d.attn_k.weight", i));
            l.k_b = this->require_tensor(ctx_data, format("v.blk.%d.attn_k.bias", i));
            l.v_w = this->require_tensor(ctx_data, format("v.blk.%d.attn_v.weight", i));
            l.v_b = this->require_tensor(ctx_data, format("v.blk.%d.attn_v.bias", i));
            l.o_w = this->require_tensor(ctx_data, format("v.blk.%d.attn_out.weight", i));
            l.o_b = this->require_tensor(ctx_data, format("v.blk.%d.attn_out.bias", i));
            l.ln2_w = this->require_tensor(ctx_data, format("v.blk.%d.ln2.weight", i));
            l.ln2_b = this->require_tensor(ctx_data, format("v.blk.%d.ln2.bias", i));
            l.ffn1_w = this->require_tensor(ctx_data, format("v.blk.%d.ffn_down.weight", i));
            l.ffn1_b = this->require_tensor(ctx_data, format("v.blk.%d.ffn_down.bias", i));
            l.ffn2_w = this->require_tensor(ctx_data, format("v.blk.%d.ffn_up.weight", i));
            l.ffn2_b = this->require_tensor(ctx_data, format("v.blk.%d.ffn_up.bias", i));
        }

        ggml_tensor * conn = this->require_tensor(ctx_data, "mm.0.weight");
        // Converter writes numpy shape [out_features, in_features] = [960, 12288].
        // In ggml tensor dims this appears as ne0=in_features, ne1=out_features.
        ctx_->connector_in_features = (int) conn->ne[0];
        ctx_->connector_out_features = (int) conn->ne[1];
        ctx_->pool_num = 16;
        ctx_->output_tokens = ctx_->num_patches / ctx_->pool_num;
        ctx_->connector_w = conn;
        ctx_->proj_dim = ctx_->connector_out_features;

        return true;
    }

private:
    smolvla_vision_ctx * ctx_;
    int verbosity_;
};

smolvla_vision_ctx * smolvla_vision_load(const char * mmproj_path, int verbosity) {
    auto * ctx = new smolvla_vision_ctx();
    ctx->verbosity = verbosity;
    memset(ctx->image_mean, 0, sizeof(ctx->image_mean));
    memset(ctx->image_std, 0, sizeof(ctx->image_std));
    ctx->ctx_gguf = nullptr;
    ctx->ctx_data = nullptr;
    ctx->backend_cpu = nullptr;
    ctx->sched = nullptr;
    ctx->graph_raw_vision = nullptr;
    ctx->graph_raw_input = nullptr;
    ctx->graph_raw_positions = nullptr;
    ctx->graph_raw_connector_output = nullptr;
    ctx->graph_raw_width = 0;
    ctx->graph_raw_height = 0;
    ctx->patch_embd_w = ctx->patch_embd_b = ctx->pos_embd = ctx->post_ln_w = ctx->post_ln_b = nullptr;
    ctx->connector_w = nullptr;

    backend_scheduler_config scheduler_config;
    scheduler_config.max_nodes = GGML_DEFAULT_GRAPH_SIZE;
    scheduler_config.parallel = false;
    scheduler_config.op_offload = false;
    backend_loader backend;
    if (!backend.load(
            ctx->backend_cpu,
            ctx->backends,
            ctx->sched,
            ctx->buft_policy,
            robotcpp_backend_use_accel_from_env(true),
            scheduler_config,
            verbosity)) {
        LOG_ERR("%s: failed to initialize vision backend: %s\n", __func__, backend.error().c_str());
        smolvla_vision_free(ctx);
        return nullptr;
    }
    smolvla_vision_log_backend_plan(ctx, verbosity);

    gguf_load_result loaded;
    {
        smolvla_vision_loader loader(ctx, verbosity);
        if (!loader.load(mmproj_path, ctx->buft_policy.model_buft, loaded, verbosity)) {
            LOG_ERR("%s: failed to load vision tensors: %s\n", __func__, loader.error().c_str());
            smolvla_vision_free(ctx);
            return nullptr;
        }
    }
    ctx->ctx_gguf = loaded.gguf;
    ctx->ctx_data = loaded.ctx_data;
    ctx->ctxs.push_back(ctx->ctx_data);
    ctx->bufs.push_back(loaded.model_buffer);

    if (verbosity >= 1) {
        LOG_INF("%s: model_buf=%s is_host=%d\n",
            __func__,
            ggml_backend_buffer_name(loaded.model_buffer),
            ggml_backend_buffer_is_host(loaded.model_buffer) ? 1 : 0);
    }

    ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());

    ctx->patch_positions.resize(ctx->num_patches);
    for (int i = 0; i < ctx->num_patches; ++i) {
        ctx->patch_positions[i] = i;
    }

    if (verbosity >= 1) {
        LOG_INF("%s: preprocessed vision graph mode: build-each-run\n", __func__);
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
    ggml_cgraph * graph = nullptr;
    ggml_tensor * inp_raw = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * connector_out = nullptr;
    double build_alloc_ms = 0.0;

    auto t_build_start = std::chrono::high_resolution_clock::now();
    graph = vision_build_preprocessed_graph(ctx);
    ggml_backend_sched_reset(ctx->sched);
    const bool reserve_ok = graph && ggml_backend_sched_reserve(ctx->sched, graph);
    const bool alloc_ok = reserve_ok && ggml_backend_sched_alloc_graph(ctx->sched, graph);
    if (!alloc_ok) {
        LOG_ERR("%s: failed to build+alloc preprocessed vision graph\n", __func__);
        return result;
    }
    auto t_build_end = std::chrono::high_resolution_clock::now();
    build_alloc_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();

    inp_raw = ggml_graph_get_tensor(graph, "inp_raw");
    positions = ggml_graph_get_tensor(graph, "positions");
    connector_out = ggml_graph_get_tensor(graph, "connector_output");
    if (!inp_raw || !positions || !connector_out) {
        LOG_ERR("%s: failed to bind preprocessed vision graph tensors\n", __func__);
        return result;
    }

    ggml_backend_tensor_set(inp_raw, inp_nchw.data(), 0, inp_nchw.size() * sizeof(float));
    ggml_backend_tensor_set(positions, ctx->patch_positions.data(), 0,
                            ctx->patch_positions.size() * sizeof(int32_t));

    set_backend_threads(ctx->backends, n_threads);
    auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_backend_sched_graph_compute(ctx->sched, graph);
    auto t_compute_end = std::chrono::high_resolution_clock::now();

    if (ctx->verbosity >= 1) {
        LOG_INF("%s: preprocessed vision graph build+alloc: %.2f ms, compute: %.2f ms\n", __func__,
                build_alloc_ms,
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

static std::vector<float> smolvla_vision_encode_raw_graph(
    smolvla_vision_ctx * ctx,
    const uint8_t * data,
    int width,
    int height,
    int channels,
    int stride_bytes,
    int n_threads
) {
    std::vector<float> result;
    if (!ctx || !data || width <= 0 || height <= 0 || channels != 3) {
        LOG_ERR("%s: invalid raw RGB args (data=%p width=%d height=%d channels=%d)\n",
                __func__, (const void *) data, width, height, channels);
        return result;
    }
    if (!ctx->sched) {
        LOG_ERR("%s: scheduler is required\n", __func__);
        return result;
    }

    const int tight_stride = width * channels;
    if (stride_bytes <= 0) {
        stride_bytes = tight_stride;
    }
    if (stride_bytes < tight_stride) {
        LOG_ERR("%s: invalid stride_bytes=%d for width=%d channels=%d\n",
                __func__, stride_bytes, width, channels);
        return result;
    }

    const float ratio = std::max((float) width / ctx->image_size, (float) height / ctx->image_size);
    const int resized_width = std::max(1, (int) std::floor((float) width / ratio));
    const int resized_height = std::max(1, (int) std::floor((float) height / ratio));
    const int pad_width = std::max(0, ctx->image_size - resized_width);
    const int pad_height = std::max(0, ctx->image_size - resized_height);

    std::vector<float> raw_hwc((size_t) width * (size_t) height * (size_t) channels);
    for (int y = 0; y < height; ++y) {
        const uint8_t * src = data + (size_t) y * (size_t) stride_bytes;
        float * dst = raw_hwc.data() + (size_t) y * (size_t) tight_stride;
        for (int x = 0; x < tight_stride; ++x) {
            dst[x] = (float) src[x];
        }
    }

    double build_alloc_ms = 0.0;
    if (!ctx->graph_raw_vision || ctx->graph_raw_width != width || ctx->graph_raw_height != height) {
        auto t_build_start = std::chrono::high_resolution_clock::now();
        ggml_backend_t preprocess_backend = smolvla_vision_first_non_cpu_backend(ctx);
        if (smolvla_vision_backend_is_accel(preprocess_backend)) {
            preprocess_backend = nullptr;
        }
        ggml_backend_sched_reset(ctx->sched);

        ggml_cgraph * graph = vision_build_graph(
            ctx,
            width,
            height,
            resized_width,
            resized_height,
            pad_width,
            pad_height);
        smolvla_vision_bind_preprocess_tensors_to_backend(ctx->sched, graph, preprocess_backend);
        const bool alloc_ok = graph && ggml_backend_sched_alloc_graph(ctx->sched, graph);
        if (!alloc_ok) {
            LOG_ERR("%s: failed to build+alloc raw vision graph\n", __func__);
            return result;
        }

        ctx->graph_raw_vision = graph;
        ctx->graph_raw_width = width;
        ctx->graph_raw_height = height;
        ctx->graph_raw_input = ggml_graph_get_tensor(graph, "inp_raw_hwc_f32");
        ctx->graph_raw_positions = ggml_graph_get_tensor(graph, "positions");
        ctx->graph_raw_connector_output = ggml_graph_get_tensor(graph, "connector_output");
        if (!ctx->graph_raw_input || !ctx->graph_raw_positions || !ctx->graph_raw_connector_output) {
            LOG_ERR("%s: failed to bind raw vision graph tensors\n", __func__);
            ctx->graph_raw_vision = nullptr;
            return result;
        }
        auto t_build_end = std::chrono::high_resolution_clock::now();
        build_alloc_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();
    }

    ggml_backend_tensor_set(ctx->graph_raw_input, raw_hwc.data(), 0, raw_hwc.size() * sizeof(float));
    ggml_backend_tensor_set(ctx->graph_raw_positions, ctx->patch_positions.data(), 0,
                            ctx->patch_positions.size() * sizeof(int32_t));

    set_backend_threads(ctx->backends, n_threads);
    auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_backend_sched_graph_compute(ctx->sched, ctx->graph_raw_vision);
    auto t_compute_end = std::chrono::high_resolution_clock::now();

    if (ctx->verbosity >= 1) {
        LOG_INF("%s: raw graph input=%dx%d resized=%dx%d pad=(%d,%d) build+alloc: %.2f ms, compute: %.2f ms\n",
                __func__,
                width,
                height,
                resized_width,
                resized_height,
                pad_width,
                pad_height,
                build_alloc_ms,
                std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count());
    }

    result.resize(ctx->output_tokens * ctx->proj_dim);
    ggml_backend_tensor_get(ctx->graph_raw_connector_output, result.data(), 0, result.size() * sizeof(float));
    return result;
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

    return smolvla_vision_encode_raw_graph(
        ctx, data, width, height, channels, stride_bytes, n_threads);
}

int smolvla_vision_embd_size(const smolvla_vision_ctx * ctx) {
    if (!ctx) return 0;
    return ctx->output_tokens * ctx->proj_dim;
}

int smolvla_vision_n_tokens(const smolvla_vision_ctx * ctx) {
    return ctx ? ctx->output_tokens : 0;
}
