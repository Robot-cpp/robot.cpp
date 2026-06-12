#include "models/pi0/vlm.h"

#include "models/ggml_runtime.h"
#include "models/pi0/load.h"
#include "models/pi0/pi0_context.h"

#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace vlacpp {

namespace {

std::string clean_prompt(const std::string & prompt) {
    size_t begin = 0;
    size_t end = prompt.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(prompt[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(prompt[end - 1]))) {
        --end;
    }

    std::string cleaned;
    cleaned.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        char ch = prompt[i];
        cleaned.push_back(ch == '_' || ch == '\n' ? ' ' : ch);
    }
    return cleaned;
}

bool tokenize_with_vocab(
    const llama_vocab * vocab,
    const std::string & text,
    bool add_special,
    std::vector<int32_t> & out) {
    int32_t count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        false);
    if (count < 0) {
        count = -count;
    }
    if (count <= 0) {
        return true;
    }
    const size_t offset = out.size();
    out.resize(offset + static_cast<size_t>(count));
    const int32_t written = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        reinterpret_cast<llama_token *>(out.data() + offset),
        count,
        add_special,
        false);
    if (written < 0) {
        out.resize(offset);
        return false;
    }
    out.resize(offset + static_cast<size_t>(written));
    return true;
}

} // namespace

Pi0Tokenizer::Pi0Tokenizer(const std::string & tokenizer_path) {
    if (tokenizer_path.empty()) {
        throw std::invalid_argument("missing pi0 tokenizer GGUF path");
    }
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap = true;
    model_.reset(llama_model_load_from_file(tokenizer_path.c_str(), params));
    if (model_ == nullptr) {
        throw std::runtime_error("failed to load pi0 tokenizer GGUF: " + tokenizer_path);
    }
    vocab_ = llama_model_get_vocab(model_.get());
    if (vocab_ == nullptr) {
        throw std::runtime_error("pi0 tokenizer GGUF is missing llama vocabulary: " + tokenizer_path);
    }
}

Pi0Tokenizer::~Pi0Tokenizer() = default;

void Pi0Tokenizer::LlamaModelDeleter::operator()(llama_model * model) const {
    if (model != nullptr) {
        llama_model_free(model);
    }
}

bool Pi0Tokenizer::available() const {
    return vocab_ != nullptr;
}

bool Pi0Tokenizer::tokenize_prompt(const std::string & prompt, int max_tokens, std::vector<int32_t> & out) const {
    out.clear();
    if (!available() || max_tokens <= 0) {
        return false;
    }
    const std::string cleaned = clean_prompt(prompt) + "\n";
    if (!tokenize_with_vocab(vocab_, cleaned, true, out)) {
        out.clear();
        return false;
    }
    if (static_cast<int>(out.size()) > max_tokens) {
        out.resize(static_cast<size_t>(max_tokens));
    }
    return true;
}

namespace {

void pi0_prefill_language_prefix_layers_batch(
    const Pi0Context & ctx,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out,
    bool need_output) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int64_t width = pi0.llm.width;
    const int64_t q_out = pi0.llm.q_out;
    const int64_t kv_out = pi0.llm.kv_out;
    const int64_t mlp_width = pi0.llm.mlp_width;
    const int layers = pi0.llm.layers;
    if (batch <= 0 || width <= 0 || q_out <= 0 || kv_out <= 0 || mlp_width <= 0 || layers <= 0 ||
        heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_out != static_cast<int64_t>(heads) * head_dim ||
        kv_out != static_cast<int64_t>(kv_heads) * head_dim ||
        tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width) ||
        positions.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("language prefix fused prefill input has incompatible shape");
    }

    std::vector<int32_t> pos_host(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        pos_host[static_cast<size_t>(i)] = static_cast<int32_t>(positions[static_cast<size_t>(i)]);
    }
    const size_t context_size = 256 * 1024 * 1024;
    const GgmlRunner & runner = *ctx.components.llm.runner;
    GgmlContext gctx(runner.init_params(context_size, &ctx, static_cast<uint64_t>(batch)));

    ggml_tensor * hidden = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, hidden, tokens.data(), tokens.size() * sizeof(float));
    runner.set_input(inputs, pos, pos_host.data(), pos_host.size() * sizeof(int32_t));

    std::vector<ggml_tensor *> k_outputs;
    std::vector<ggml_tensor *> v_outputs;
    k_outputs.reserve(static_cast<size_t>(layers));
    v_outputs.reserve(static_cast<size_t>(layers));

    for (int layer = 0; layer < layers; ++layer) {
        const Pi0TransformerLayerWeights & layer_w = ctx.weights.llm_layers[static_cast<size_t>(layer)];
        const Tensor * q_w = layer_w.q;
        const Tensor * k_w = layer_w.k;
        const Tensor * v_w = layer_w.v;
        const Tensor * out_w = layer_w.out;
        const Tensor * gate_w = layer_w.gate;
        const Tensor * up_w = layer_w.up;
        const Tensor * down_w = layer_w.down;
        if (q_w == nullptr || k_w == nullptr || v_w == nullptr || out_w == nullptr ||
            gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix fused prefill tensor");
        }

        if (layer_w.input_norm_scale == nullptr || layer_w.post_norm_scale == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix norm scale tensor");
        }
        ggml_tensor * input_scale_tensor = pi0_weight(ctx, *layer_w.input_norm_scale, 1);
        ggml_tensor * post_scale_tensor = pi0_weight(ctx, *layer_w.post_norm_scale, 1);

        ggml_tensor * normed = ggml_mul(gctx, ggml_rms_norm(gctx, hidden, 1.0e-6f), input_scale_tensor);
        ggml_tensor * q = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *q_w, 2), normed), q_out, batch);
        ggml_tensor * k = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *k_w, 2), normed), kv_out, batch);
        ggml_tensor * v = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *v_w, 2), normed), kv_out, batch);
        q = ggml_reshape_3d(gctx, q, head_dim, heads, batch);
        k = ggml_reshape_3d(gctx, k, head_dim, kv_heads, batch);
        v = ggml_reshape_3d(gctx, v, head_dim, kv_heads, batch);
        ggml_tensor * q_rot = ggml_rope_ext(
            gctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_rot = ggml_rope_ext(
            gctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * q_perm = ggml_permute(gctx, q_rot, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(gctx, k_rot, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor * v_perm = ggml_permute(gctx, v, 0, 2, 1, 3);
        v_perm = ggml_cont(gctx, ggml_transpose(gctx, v_perm));
        ggml_tensor * scores = ggml_mul_mat(gctx, k_perm, q_perm);
        scores = ggml_soft_max_ext(gctx, scores, nullptr, scale, 0.0f);
        ggml_tensor * values = ggml_mul_mat(gctx, v_perm, scores);
        ggml_tensor * attn_values = ggml_permute(gctx, values, 0, 2, 1, 3);
        attn_values = ggml_cont_2d(gctx, attn_values, static_cast<int64_t>(head_dim) * heads, batch);
        ggml_tensor * attn_out = ggml_mul_mat(gctx, pi0_weight(ctx, *out_w, 2), attn_values);
        ggml_tensor * first_residual = ggml_add(gctx, hidden, attn_out);

        ggml_tensor * post_norm = ggml_mul(gctx, ggml_rms_norm(gctx, first_residual, 1.0e-6f), post_scale_tensor);
        ggml_tensor * gate = ggml_gelu(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *gate_w, 2), post_norm));
        ggml_tensor * up = ggml_mul_mat(gctx, pi0_weight(ctx, *up_w, 2), post_norm);
        ggml_tensor * mlp_out = ggml_mul_mat(gctx, pi0_weight(ctx, *down_w, 2), ggml_mul(gctx, gate, up));
        hidden = ggml_add(gctx, first_residual, mlp_out);

        k_outputs.push_back(ggml_cont_3d(gctx, k_rot, head_dim, kv_heads, batch));
        v_outputs.push_back(ggml_cont_3d(gctx, v, head_dim, kv_heads, batch));
    }

    ggml_tensor * y = nullptr;
    if (need_output) {
        if (ctx.weights.llm_norm_scale == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix final norm scale tensor");
        }
        ggml_tensor * final_scale_tensor = pi0_weight(ctx, *ctx.weights.llm_norm_scale, 1);
        y = ggml_mul(gctx, ggml_rms_norm(gctx, hidden, 1.0e-6f), final_scale_tensor);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
    if (need_output) {
        ggml_build_forward_expand(graph, y);
    }
    for (size_t i = 0; i < k_outputs.size(); ++i) {
        ggml_build_forward_expand(graph, k_outputs[i]);
        ggml_build_forward_expand(graph, v_outputs[i]);
    }
    runner.compute(gctx, graph, inputs, "ggml language prefix fused prefill graph compute failed");

    if (need_output) {
        out.resize(static_cast<size_t>(batch) * static_cast<size_t>(width));
        runner.get_output(y, out.data(), out.size() * sizeof(float));
    } else {
        out.clear();
    }
    ctx.prefix_kv.k_layers.resize(static_cast<size_t>(layers), nullptr);
    ctx.prefix_kv.v_layers.resize(static_cast<size_t>(layers), nullptr);
    for (int layer = 0; layer < layers; ++layer) {
        ctx.prefix_kv.k_layers[static_cast<size_t>(layer)] = runner.materialize_device_f32_3d_from_backend(
            gctx,
            &ctx.prefix_kv.k_layers[static_cast<size_t>(layer)],
            k_outputs[static_cast<size_t>(layer)],
            head_dim,
            kv_heads,
            batch);
        ctx.prefix_kv.v_layers[static_cast<size_t>(layer)] = runner.materialize_device_f32_3d_from_backend(
            gctx,
            &ctx.prefix_kv.v_layers[static_cast<size_t>(layer)],
            v_outputs[static_cast<size_t>(layer)],
            head_dim,
            kv_heads,
            batch);
    }
}

} // namespace

bool pi0_has_language_layer(const Pi0Context & ctx, int layer) {
    return layer >= 0 &&
        static_cast<size_t>(layer) < ctx.weights.llm_layers.size() &&
        ctx.weights.llm_layers[static_cast<size_t>(layer)].gate != nullptr &&
        ctx.weights.llm_layers[static_cast<size_t>(layer)].q != nullptr;
}

void pi0_prefill_language_prefix_batch(
    const Pi0Context & ctx,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out,
    uint64_t generation,
    bool need_output) {
    const size_t width = static_cast<size_t>(pi0_config(ctx.config).llm.width);
    if (batch <= 0 || width == 0 || tokens.size() != static_cast<size_t>(batch) * width ||
        positions.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("language prefix input has incompatible shape");
    }

    pi0_prefill_language_prefix_layers_batch(ctx, tokens, positions, batch, heads, kv_heads, head_dim, out, need_output);
    ctx.prefix_kv.token_count = static_cast<size_t>(batch);
    ctx.prefix_kv.generation = generation;
}

namespace {

std::vector<float> images_hwc_to_planar_batch(const std::vector<ImageTensor> & images) {
    const ImageTensor & first = images.front();
    const size_t pixels = static_cast<size_t>(first.width) * static_cast<size_t>(first.height);
    std::vector<float> planar(images.size() * pixels * 3);
    for (size_t i = 0; i < images.size(); ++i) {
        const ImageTensor & image = images[i];
        float * dst = planar.data() + i * pixels * 3;
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x);
                const size_t src = pixel * 3;
                dst[pixel] = image.data[src];
                dst[pixels + pixel] = image.data[src + 1];
                dst[2 * pixels + pixel] = image.data[src + 2];
            }
        }
    }
    return planar;
}

} // namespace

bool pi0_has_vision_encoder(const Pi0Context & ctx) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    return pi0.vision.width > 0 &&
        pi0.vision.layers > 0 &&
        pi0.vision.heads > 0 &&
        pi0.vision.norm_epsilon > 0.0f &&
        ctx.weights.vit_patch_w != nullptr &&
        ctx.weights.vit_patch_b != nullptr &&
        ctx.weights.vit_pos != nullptr &&
        ctx.weights.vit_post_norm_w != nullptr &&
        ctx.weights.vit_post_norm_b != nullptr;
}

void pi0_encode_vision(
    const Pi0Context & ctx,
    const std::vector<ImageTensor> & images,
    std::vector<float> & out_tokens,
    int & out_token_count) {
    out_tokens.clear();
    out_token_count = 0;
    if (!pi0_has_vision_encoder(ctx) || images.empty()) {
        return;
    }

    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int64_t width = pi0.vision.width;
    const int64_t layers = pi0.vision.layers;
    const int64_t patch_h = pi0.vision.patch_height;
    const int64_t patch_w = pi0.vision.patch_width;
    const int64_t heads = pi0.vision.heads;
    if (width <= 0 || layers <= 0 || patch_h <= 0 || patch_w <= 0 || heads <= 0 || width % heads != 0 ||
        pi0.vision.norm_epsilon <= 0.0f) {
        throw std::invalid_argument("pi0 ViT metadata has incompatible dimensions");
    }
    const int64_t head_dim = width / heads;

    const Tensor & patch_weight = pi0_tensor(ctx.weights.vit_patch_w, "embeddings.patch_embedding.weight");
    const Tensor & patch_bias = pi0_tensor(ctx.weights.vit_patch_b, "embeddings.patch_embedding.bias");
    const Tensor & position = pi0_tensor(ctx.weights.vit_pos, "embeddings.position_embedding.weight");

    const ImageTensor & first_image = images.front();
    if (first_image.width <= 0 || first_image.height <= 0 || first_image.channels != 3 ||
        first_image.width % patch_w != 0 || first_image.height % patch_h != 0) {
        throw std::invalid_argument("pi0 ViT image has incompatible shape");
    }
    const int64_t patches_x = first_image.width / patch_w;
    const int64_t patches_y = first_image.height / patch_h;
    const int64_t patches = patches_x * patches_y;
    const size_t expected = static_cast<size_t>(first_image.width) * static_cast<size_t>(first_image.height) * 3;
    for (const ImageTensor & image : images) {
        if (image.width <= 0 || image.height <= 0 || image.channels != 3 ||
            image.width != first_image.width || image.height != first_image.height) {
            throw std::invalid_argument("pi0 ViT image has incompatible shape");
        }
        if (image.data.size() != expected) {
            throw std::invalid_argument("pi0 ViT image data size mismatch");
        }
    }

    const int64_t batch = static_cast<int64_t>(images.size());
    const GgmlRunner & runner = *ctx.components.vit.runner;
    GgmlContext gctx(runner.init_params(ggml_graph_context_size(0), &ctx, static_cast<uint64_t>(patches * batch)));
    auto norm = [&](ggml_tensor * input, const Tensor & weight, const Tensor & bias) {
        ggml_tensor * out = ggml_norm(gctx, input, pi0.vision.norm_epsilon);
        out = ggml_mul(gctx, out, pi0_weight(ctx, weight, 1));
        return ggml_add(gctx, out, pi0_weight(ctx, bias, 1));
    };
    auto linear = [&](ggml_tensor * input, const Tensor & weight, const Tensor & bias) {
        return ggml_add(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, weight, 2), input), pi0_weight(ctx, bias, 1));
    };
    auto attention = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        ggml_tensor * q_perm = ggml_permute(gctx, q, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(gctx, k, 0, 2, 1, 3);
        ggml_tensor * v_perm = ggml_cont(gctx, ggml_permute(gctx, v, 1, 2, 0, 3));
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor * scores = ggml_mul_mat(gctx, k_perm, q_perm);
        scores = ggml_soft_max_ext(gctx, scores, nullptr, scale, 0.0f);
        ggml_tensor * values = ggml_mul_mat(gctx, v_perm, scores);
        ggml_tensor * y = ggml_permute(gctx, values, 0, 2, 1, 3);
        return ggml_cont_3d(gctx, y, width, patches, batch);
    };

    ggml_tensor * input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, first_image.width, first_image.height, 3, batch);
    std::vector<GgmlInput> inputs;
    const std::vector<float> planar_images = images_hwc_to_planar_batch(images);
    runner.set_input(inputs, input, planar_images.data(), planar_images.size() * sizeof(float));

    ggml_tensor * hidden = ggml_conv_2d(
        gctx,
        pi0_weight(ctx, patch_weight, 4),
        input,
        patch_w,
        patch_h,
        0,
        0,
        1,
        1);
    hidden = ggml_reshape_4d(gctx, hidden, patches, width, batch, 1);
    hidden = ggml_cont_3d(gctx, ggml_permute(gctx, hidden, 1, 0, 2, 3), width, patches, batch);
    hidden = ggml_add(gctx, hidden, pi0_weight(ctx, patch_bias, 1));
    hidden = ggml_add(gctx, hidden, pi0_weight(ctx, position, 2, GGML_TYPE_F32));

    for (int layer = 0; layer < layers; ++layer) {
        const Pi0VisionLayerWeights & layer_w = ctx.weights.vit_layers[static_cast<size_t>(layer)];
        ggml_tensor * residual = hidden;
        ggml_tensor * cur = norm(
            hidden,
            pi0_tensor(layer_w.norm1_w, "ViT layer_norm1.weight"),
            pi0_tensor(layer_w.norm1_b, "ViT layer_norm1.bias"));

        ggml_tensor * q = linear(
            cur,
            pi0_tensor(layer_w.q_w, "ViT q_proj.weight"),
            pi0_tensor(layer_w.q_b, "ViT q_proj.bias"));
        ggml_tensor * k = linear(
            cur,
            pi0_tensor(layer_w.k_w, "ViT k_proj.weight"),
            pi0_tensor(layer_w.k_b, "ViT k_proj.bias"));
        ggml_tensor * v = linear(
            cur,
            pi0_tensor(layer_w.v_w, "ViT v_proj.weight"),
            pi0_tensor(layer_w.v_b, "ViT v_proj.bias"));
        q = ggml_reshape_4d(gctx, q, head_dim, heads, patches, batch);
        k = ggml_reshape_4d(gctx, k, head_dim, heads, patches, batch);
        v = ggml_reshape_4d(gctx, v, head_dim, heads, patches, batch);
        cur = attention(q, k, v);
        cur = linear(
            cur,
            pi0_tensor(layer_w.out_w, "ViT out_proj.weight"),
            pi0_tensor(layer_w.out_b, "ViT out_proj.bias"));
        hidden = ggml_add(gctx, residual, cur);

        residual = hidden;
        cur = norm(
            hidden,
            pi0_tensor(layer_w.norm2_w, "ViT layer_norm2.weight"),
            pi0_tensor(layer_w.norm2_b, "ViT layer_norm2.bias"));
        cur = linear(
            cur,
            pi0_tensor(layer_w.fc1_w, "ViT fc1.weight"),
            pi0_tensor(layer_w.fc1_b, "ViT fc1.bias"));
        cur = ggml_gelu(gctx, cur);
        cur = linear(
            cur,
            pi0_tensor(layer_w.fc2_w, "ViT fc2.weight"),
            pi0_tensor(layer_w.fc2_b, "ViT fc2.bias"));
        hidden = ggml_add(gctx, residual, cur);
    }

    hidden = norm(
        hidden,
        pi0_tensor(ctx.weights.vit_post_norm_w, "post_layernorm.weight"),
        pi0_tensor(ctx.weights.vit_post_norm_b, "post_layernorm.bias"));

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, hidden);
    runner.compute(gctx, graph, inputs, "ggml pi0 ViT graph compute failed");

    out_tokens.resize(images.size() * static_cast<size_t>(patches) * static_cast<size_t>(width));
    runner.get_output(hidden, out_tokens.data(), out_tokens.size() * sizeof(float));
    out_token_count = static_cast<int>(images.size() * static_cast<size_t>(patches));
}

namespace {

bool has_text_embeddings(const Pi0Context & ctx) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const Tensor * embeddings = ctx.weights.lm_head;
    return embeddings != nullptr &&
        embeddings->shape.size() == 2 &&
        embeddings->shape[0] == pi0.llm.width &&
        embeddings->shape[1] > 0;
}

} // namespace

bool pi0_has_merger(const Pi0Context & ctx) {
    return ctx.weights.merger_w != nullptr && ctx.weights.merger_b != nullptr;
}

bool pi0_has_vision_prefix(const Pi0Context & ctx) {
    return pi0_has_vision_encoder(ctx) && pi0_has_merger(ctx);
}

bool pi0_has_language_prefix(const Pi0Context & ctx) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    return pi0.llm.layers > 0 &&
        pi0.llm.width > 0 &&
        pi0.llm.q_out > 0 &&
        pi0.llm.kv_out > 0 &&
        pi0_has_language_layer(ctx, 0);
}

void pi0_project_vision_tokens(
    const Pi0Context & ctx,
    const std::vector<float> & vision_tokens,
    int token_count,
    std::vector<float> & out) {
    const Tensor * weight = ctx.weights.merger_w;
    const Tensor * bias = ctx.weights.merger_b;
    if (weight == nullptr || bias == nullptr) {
        throw std::invalid_argument("missing pi0 vision projector tensors");
    }
    if (token_count <= 0) {
        throw std::invalid_argument("pi0 vision projector requires tokens");
    }

    pi0_linear_batch(
        ctx,
        *weight,
        *bias,
        vision_tokens,
        token_count,
        out,
        "ggml pi0 vision projector graph compute failed");
}

void pi0_embed_prompt(
    const Pi0Context & ctx,
    const std::string & prompt,
    std::vector<float> & out,
    int & token_count) {
    out.clear();
    token_count = 0;
    if (prompt.empty()) {
        return;
    }
    if (!has_text_embeddings(ctx)) {
        throw std::invalid_argument("missing pi0 LLM token embedding tensor");
    }
    std::vector<int32_t> tokens;
    if (ctx.tokenizer.tokenize_prompt(prompt, ctx.config.common.max_token_len, tokens)) {
        pi0_embed_prompt_tokens(ctx, tokens, out, token_count);
        return;
    }
    throw std::runtime_error("pi0 prompt text requires a llama.cpp tokenizer sidecar or explicit prompt token ids");
}

void pi0_embed_prompt_tokens(
    const Pi0Context & ctx,
    const std::vector<int32_t> & tokens,
    std::vector<float> & out,
    int & token_count) {
    out.clear();
    token_count = 0;
    if (tokens.empty()) {
        return;
    }
    if (!has_text_embeddings(ctx)) {
        throw std::invalid_argument("missing pi0 LLM token embedding tensor");
    }
    const Tensor * embeddings = ctx.weights.lm_head;
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int width = pi0.llm.width;
    const float scale = std::sqrt(static_cast<float>(width));
    const int64_t vocab = embeddings->shape[1];
    const size_t count = std::min(tokens.size(), static_cast<size_t>(ctx.config.common.max_token_len));
    std::vector<int32_t> token_ids(count);
    for (size_t i = 0; i < count; ++i) {
        const int64_t token = tokens[i];
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument("pi0 prompt token id is outside the embedding vocabulary");
        }
        token_ids[i] = static_cast<int32_t>(token);
    }

    const size_t tensor_bytes =
        token_ids.size() * sizeof(int32_t) +
        token_ids.size() * static_cast<size_t>(width) * sizeof(float);
    const GgmlRunner & runner = *ctx.components.llm.runner;
    GgmlContext gctx(runner.init_params(ggml_graph_context_size(tensor_bytes), &ctx, count));

    ggml_tensor * ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, static_cast<int64_t>(count));
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, ids, token_ids.data(), token_ids.size() * sizeof(int32_t));

    ggml_tensor * rows = ggml_get_rows(gctx, pi0_weight(ctx, *embeddings, 2), ids);
    ggml_tensor * y = ggml_scale(gctx, rows, scale);

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(gctx, graph, inputs, "ggml pi0 prompt embedding lookup failed");

    out.resize(count * static_cast<size_t>(width));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
    token_count = static_cast<int>(count);
}

void pi0_prefill_prefix_from_embeddings(
    const Pi0Context & ctx,
    KvCache & cache,
    const std::vector<float> & embeddings,
    int token_count) {
    if (token_count <= 0) {
        cache.reset();
        cache.prefix_valid = true;
        return;
    }
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int width = pi0.llm.width;
    if (!pi0_has_language_prefix(ctx) ||
        width <= 0 ||
        pi0.llm.kv_out <= 0 ||
        pi0.llm.q_out % pi0.llm.kv_out != 0 ||
        embeddings.size() != static_cast<size_t>(token_count) * static_cast<size_t>(width)) {
        throw std::invalid_argument("pi0 prefix embeddings have incompatible shape");
    }

    const int head_dim = pi0.llm.kv_out;
    const int heads = pi0.llm.q_out / head_dim;
    const int kv_heads = pi0.llm.kv_out / head_dim;
    std::vector<int> positions(static_cast<size_t>(token_count), 0);
    for (int i = 0; i < token_count; ++i) {
        positions[static_cast<size_t>(i)] = i;
    }

    std::vector<float> hidden;
    pi0_prefill_language_prefix_batch(
        ctx,
        embeddings,
        positions,
        token_count,
        heads,
        kv_heads,
        head_dim,
        hidden,
        cache.prefix_generation,
        false);
    cache.token_count = static_cast<size_t>(token_count);
    cache.prefix_valid = true;
}

void pi0_prefill_prefix(const Pi0Context & ctx, KvCache & cache, const ObservationData & observation) {
    if (cache.prefix_valid) {
        cache.reset();
        ctx.prefix_kv.reset();
    }
    if (pi0_has_vision_prefix(ctx) && pi0_has_language_prefix(ctx) && !observation.images.empty()) {
        std::vector<float> embeddings;
        int token_count = 0;
        pi0_encode_vision(ctx, observation.images, embeddings, token_count);
        std::vector<float> projected;
        pi0_project_vision_tokens(ctx, embeddings, token_count, projected);
        embeddings = std::move(projected);
        std::vector<float> prompt_embeddings;
        int prompt_tokens = 0;
        if (!observation.prompt_tokens.empty()) {
            pi0_embed_prompt_tokens(ctx, observation.prompt_tokens, prompt_embeddings, prompt_tokens);
        } else {
            pi0_embed_prompt(ctx, observation.prompt, prompt_embeddings, prompt_tokens);
        }
        if (prompt_tokens > 0) {
            embeddings.insert(embeddings.end(), prompt_embeddings.begin(), prompt_embeddings.end());
            token_count += prompt_tokens;
        }
        pi0_prefill_prefix_from_embeddings(ctx, cache, embeddings, token_count);
        return;
    }
    ctx.prefix_kv.reset();
    cache.token_count = 0;
    cache.prefix_valid = true;
}

} // namespace vlacpp
