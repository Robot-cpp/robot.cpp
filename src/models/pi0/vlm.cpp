#include "models/pi0/vlm.h"

#include "models/pi0/load.h"
#include "models/pi0/component_runtime.h"
#include "models/pi0/pi0_context.h"

#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace robotcpp::pi0 {

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

ggml_backend_dev_t pi0_cpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return dev;
        }
    }
    return nullptr;
}

} // namespace

Pi0Tokenizer::Pi0Tokenizer(const std::string & tokenizer_path) {
    if (tokenizer_path.empty()) {
        throw std::invalid_argument("missing pi0 tokenizer GGUF path");
    }
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap = true;
    params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu_devices[] = {pi0_cpu_device(), nullptr};
    if (cpu_devices[0] != nullptr) {
        params.devices = cpu_devices;
        params.main_gpu = 0;
    }
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
    bool need_output,
    ggml_tensor * runtime_prefix_embeddings = nullptr,
    int runtime_prefix_count = 0,
    ggml_tensor * runtime_suffix_embeddings = nullptr,
    int runtime_suffix_count = 0) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int64_t width = pi0.llm.width;
    const int64_t q_out = pi0.llm.q_out;
    const int64_t kv_out = pi0.llm.kv_out;
    const int64_t mlp_width = pi0.llm.mlp_width;
    const int layers = pi0.llm.layers;
    const int host_token_count = batch - runtime_prefix_count - runtime_suffix_count;
    if (batch <= 0 || width <= 0 || q_out <= 0 || kv_out <= 0 || mlp_width <= 0 || layers <= 0 ||
        runtime_prefix_count < 0 || runtime_suffix_count < 0 || host_token_count < 0 ||
        heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_out != static_cast<int64_t>(heads) * head_dim ||
        kv_out != static_cast<int64_t>(kv_heads) * head_dim ||
        tokens.size() != static_cast<size_t>(host_token_count) * static_cast<size_t>(width) ||
        positions.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("language prefix fused prefill input has incompatible shape");
    }
    auto validate_runtime_embeddings = [width](ggml_tensor * tensor, int count) {
        return count == 0 ||
            (tensor != nullptr &&
                ggml_n_dims(tensor) == 2 &&
                tensor->type == GGML_TYPE_F32 &&
                tensor->ne[0] == width &&
                tensor->ne[1] == count);
    };
    if (!validate_runtime_embeddings(runtime_prefix_embeddings, runtime_prefix_count) ||
        !validate_runtime_embeddings(runtime_suffix_embeddings, runtime_suffix_count)) {
        throw std::invalid_argument("language prefix runtime embeddings have incompatible shape");
    }

    std::vector<int32_t> pos_host(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        pos_host[static_cast<size_t>(i)] = static_cast<int32_t>(positions[static_cast<size_t>(i)]);
    }
    const size_t context_size = 256 * 1024 * 1024;
    const Pi0ComponentRuntime & runtime = ctx.components.llm.runtime;
    Pi0GraphContext gctx(pi0_graph_init_params(context_size));

    ggml_tensor * host_input = nullptr;
    if (host_token_count > 0) {
        host_input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, width, host_token_count);
        ggml_set_input(host_input);
    }
    ggml_tensor * hidden = nullptr;
    auto append_hidden = [&](ggml_tensor * value) {
        hidden = hidden == nullptr ? value : ggml_concat(gctx, hidden, value, 1);
    };
    auto view_runtime_embeddings = [&](ggml_tensor * value, int count) {
        return ggml_view_2d(
            gctx,
            value,
            width,
            count,
            ggml_row_size(value->type, width),
            0);
    };
    if (runtime_prefix_count > 0) {
        append_hidden(view_runtime_embeddings(runtime_prefix_embeddings, runtime_prefix_count));
    }
    if (runtime_suffix_count > 0) {
        append_hidden(view_runtime_embeddings(runtime_suffix_embeddings, runtime_suffix_count));
    }
    if (host_input != nullptr) {
        append_hidden(host_input);
    }
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, batch);
    ggml_set_input(pos);

    std::vector<ggml_tensor *> k_outputs;
    std::vector<ggml_tensor *> v_outputs;
    k_outputs.reserve(static_cast<size_t>(layers));
    v_outputs.reserve(static_cast<size_t>(layers));

    for (int layer = 0; layer < layers; ++layer) {
        const Pi0TransformerLayerWeights & layer_w = ctx.weights.llm_layers[static_cast<size_t>(layer)];
        ggml_tensor * q_w = layer_w.q;
        ggml_tensor * k_w = layer_w.k;
        ggml_tensor * v_w = layer_w.v;
        ggml_tensor * out_w = layer_w.out;
        ggml_tensor * gate_w = layer_w.gate;
        ggml_tensor * up_w = layer_w.up;
        ggml_tensor * down_w = layer_w.down;
        if (q_w == nullptr || k_w == nullptr || v_w == nullptr || out_w == nullptr ||
            gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix fused prefill tensor");
        }

        if (layer_w.input_norm_scale == nullptr || layer_w.post_norm_scale == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix norm scale tensor");
        }
        ggml_tensor * input_scale_tensor = pi0_f32_weight(gctx, ctx, layer_w.input_norm_scale, 1);
        ggml_tensor * post_scale_tensor = pi0_f32_weight(gctx, ctx, layer_w.post_norm_scale, 1);

        ggml_tensor * normed = ggml_mul(gctx, ggml_rms_norm(gctx, hidden, 1.0e-6f), input_scale_tensor);
        ggml_tensor * q = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, q_w, 2), normed), q_out, batch);
        ggml_tensor * k = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, k_w, 2), normed), kv_out, batch);
        ggml_tensor * v = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, v_w, 2), normed), kv_out, batch);
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
        ggml_tensor * attn_values = ggml_flash_attn_ext(gctx, q_perm, k_perm, v_perm, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_values, GGML_PREC_F32);
        attn_values = ggml_reshape_2d(gctx, attn_values, static_cast<int64_t>(head_dim) * heads, batch);
        ggml_tensor * attn_out = ggml_mul_mat(gctx, pi0_weight(ctx, out_w, 2), attn_values);
        ggml_tensor * first_residual = ggml_add(gctx, hidden, attn_out);

        ggml_tensor * post_norm = ggml_mul(gctx, ggml_rms_norm(gctx, first_residual, 1.0e-6f), post_scale_tensor);
        ggml_tensor * gate = ggml_gelu(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, gate_w, 2), post_norm));
        ggml_tensor * up = ggml_mul_mat(gctx, pi0_weight(ctx, up_w, 2), post_norm);
        ggml_tensor * mlp_out = ggml_mul_mat(gctx, pi0_weight(ctx, down_w, 2), ggml_mul(gctx, gate, up));
        hidden = ggml_add(gctx, first_residual, mlp_out);

        k_outputs.push_back(ggml_cont_3d(gctx, k_rot, head_dim, kv_heads, batch));
        v_outputs.push_back(ggml_cont_3d(gctx, v, head_dim, kv_heads, batch));
    }

    ggml_tensor * y = nullptr;
    if (need_output) {
        if (ctx.weights.llm_norm_scale == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix final norm scale tensor");
        }
        ggml_tensor * final_scale_tensor = pi0_f32_weight(gctx, ctx, ctx.weights.llm_norm_scale, 1);
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
    ggml_backend_sched_reset(runtime.sched);
    if (!ggml_backend_sched_alloc_graph(runtime.sched, graph)) {
        throw std::runtime_error("pi0 language prefix graph allocation failed");
    }
    if (host_input != nullptr) {
        ggml_backend_tensor_set(host_input, tokens.data(), 0, tokens.size() * sizeof(float));
    }
    ggml_backend_tensor_set(pos, pos_host.data(), 0, pos_host.size() * sizeof(int32_t));
    set_backend_threads(runtime.backends, runtime.n_threads);
    if (ggml_backend_sched_graph_compute(runtime.sched, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml language prefix fused prefill graph compute failed");
    }

    if (need_output) {
        out.resize(static_cast<size_t>(batch) * static_cast<size_t>(width));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    } else {
        out.clear();
    }
    ctx.prefix_kv.k_layers.resize(static_cast<size_t>(layers), nullptr);
    ctx.prefix_kv.v_layers.resize(static_cast<size_t>(layers), nullptr);
    for (int layer = 0; layer < layers; ++layer) {
        ctx.prefix_kv.k_layers[static_cast<size_t>(layer)] = pi0_persist_backend_f32(
            runtime,
            &ctx.prefix_kv.k_layers[static_cast<size_t>(layer)],
            k_outputs[static_cast<size_t>(layer)],
            3,
            head_dim,
            kv_heads,
            batch);
        ctx.prefix_kv.v_layers[static_cast<size_t>(layer)] = pi0_persist_backend_f32(
            runtime,
            &ctx.prefix_kv.v_layers[static_cast<size_t>(layer)],
            v_outputs[static_cast<size_t>(layer)],
            3,
            head_dim,
            kv_heads,
            batch);
    }
}

} // namespace

static bool pi0_has_language_layer(const Pi0Context & ctx, int layer) {
    return layer >= 0 &&
        static_cast<size_t>(layer) < ctx.weights.llm_layers.size() &&
        ctx.weights.llm_layers[static_cast<size_t>(layer)].gate != nullptr &&
        ctx.weights.llm_layers[static_cast<size_t>(layer)].q != nullptr;
}

namespace {

std::vector<float> images_hwc_to_planar_batch(const std::vector<Pi0ImageTensor> & images) {
    const Pi0ImageTensor & first = images.front();
    const size_t pixels = static_cast<size_t>(first.width) * static_cast<size_t>(first.height);
    std::vector<float> planar(images.size() * pixels * 3);
    for (size_t i = 0; i < images.size(); ++i) {
        const Pi0ImageTensor & image = images[i];
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

static bool pi0_has_vision_encoder(const Pi0Context & ctx) {
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

namespace {

ggml_tensor * pi0_encode_vision_prefix(
    const Pi0Context & ctx,
    const std::vector<Pi0ImageTensor> & images,
    const Pi0ComponentRuntime & target_runtime,
    ggml_tensor ** target_slot,
    int & out_token_count) {
    out_token_count = 0;
    if (!pi0_has_vision_encoder(ctx) || images.empty()) {
        return nullptr;
    }
    if (target_slot == nullptr) {
        throw std::invalid_argument("pi0 vision runtime output requires a persistent tensor slot");
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

    ggml_tensor * patch_weight = pi0_tensor(ctx.weights.vit_patch_w, "embeddings.patch_embedding.weight");
    ggml_tensor * patch_bias = pi0_tensor(ctx.weights.vit_patch_b, "embeddings.patch_embedding.bias");
    ggml_tensor * position = pi0_tensor(ctx.weights.vit_pos, "embeddings.position_embedding.weight");

    const Pi0ImageTensor & first_image = images.front();
    if (first_image.width <= 0 || first_image.height <= 0 || first_image.channels != 3 ||
        first_image.width % patch_w != 0 || first_image.height % patch_h != 0) {
        throw std::invalid_argument("pi0 ViT image has incompatible shape");
    }
    const int64_t patches_x = first_image.width / patch_w;
    const int64_t patches_y = first_image.height / patch_h;
    const int64_t patches = patches_x * patches_y;
    const size_t expected = static_cast<size_t>(first_image.width) * static_cast<size_t>(first_image.height) * 3;
    for (const Pi0ImageTensor & image : images) {
        if (image.width <= 0 || image.height <= 0 || image.channels != 3 ||
            image.width != first_image.width || image.height != first_image.height) {
            throw std::invalid_argument("pi0 ViT image has incompatible shape");
        }
        if (image.data.size() != expected) {
            throw std::invalid_argument("pi0 ViT image data size mismatch");
        }
    }

    const int64_t batch = static_cast<int64_t>(images.size());
    const int64_t vision_token_count = batch * patches;
    const Pi0ComponentRuntime & runtime = ctx.components.vit.runtime;
    Pi0GraphContext gctx(pi0_graph_init_params(pi0_graph_context_size(0)));
    auto norm = [&](ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) {
        ggml_tensor * out = ggml_norm(gctx, input, pi0.vision.norm_epsilon);
        out = ggml_mul(gctx, out, pi0_f32_weight(gctx, ctx, weight, 1));
        return ggml_add(gctx, out, pi0_f32_weight(gctx, ctx, bias, 1));
    };
    auto linear = [&](ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias) {
        return ggml_add(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, weight, 2), input), pi0_f32_weight(gctx, ctx, bias, 1));
    };
    auto attention = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        ggml_tensor * q_perm = ggml_permute(gctx, q, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(gctx, k, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor * v_perm = ggml_permute(gctx, v, 0, 2, 1, 3);
        ggml_tensor * y = ggml_flash_attn_ext(gctx, q_perm, k_perm, v_perm, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
        return ggml_reshape_3d(gctx, y, width, patches, batch);
    };

    ggml_tensor * input = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, first_image.width, first_image.height, 3, batch);
    const std::vector<float> planar_images = images_hwc_to_planar_batch(images);
    ggml_set_input(input);

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
    hidden = ggml_add(gctx, hidden, pi0_f32_weight(gctx, ctx, patch_bias, 1));
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

    ggml_tensor * output = ggml_reshape_2d(gctx, hidden, width, vision_token_count);
    ggml_tensor * weight = ctx.weights.merger_w;
    ggml_tensor * bias = ctx.weights.merger_b;
    if (weight == nullptr || bias == nullptr) {
        throw std::invalid_argument("missing pi0 vision projector tensors");
    }
    if (ggml_n_dims(weight) != 2 ||
        weight->ne[0] != width ||
        ggml_n_dims(bias) != 1 ||
        bias->ne[0] != weight->ne[1]) {
        throw std::invalid_argument("pi0 vision projector has incompatible shape");
    }
    output = ggml_add(
        gctx,
        ggml_mul_mat(gctx, pi0_weight(ctx, weight, 2), output),
        pi0_f32_weight(gctx, ctx, bias, 1));
    const int64_t output_width = weight->ne[1];

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_reset(runtime.sched);
    if (!ggml_backend_sched_alloc_graph(runtime.sched, graph)) {
        throw std::runtime_error("pi0 ViT graph allocation failed");
    }
    ggml_backend_tensor_set(input, planar_images.data(), 0, planar_images.size() * sizeof(float));
    set_backend_threads(runtime.backends, runtime.n_threads);
    if (ggml_backend_sched_graph_compute(runtime.sched, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml pi0 ViT graph compute failed");
    }

    out_token_count = static_cast<int>(vision_token_count);
    return pi0_persist_backend_f32(
        target_runtime,
        target_slot,
        output,
        2,
        output_width,
        vision_token_count,
        1);
}

} // namespace

namespace {

bool has_text_embeddings(const Pi0Context & ctx) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    ggml_tensor * embeddings = ctx.weights.lm_head;
    return embeddings != nullptr &&
        ggml_n_dims(embeddings) == 2 &&
        embeddings->ne[0] == pi0.llm.width &&
        embeddings->ne[1] > 0;
}

} // namespace

static bool pi0_has_merger(const Pi0Context & ctx) {
    return ctx.weights.merger_w != nullptr && ctx.weights.merger_b != nullptr;
}

static bool pi0_has_vision_prefix(const Pi0Context & ctx) {
    return pi0_has_vision_encoder(ctx) && pi0_has_merger(ctx);
}

static bool pi0_has_language_prefix(const Pi0Context & ctx) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    return pi0.llm.layers > 0 &&
        pi0.llm.width > 0 &&
        pi0.llm.q_out > 0 &&
        pi0.llm.kv_out > 0 &&
        pi0_has_language_layer(ctx, 0);
}

static void pi0_embed_prompt_tokens(
    const Pi0Context & ctx,
    const std::vector<int32_t> & tokens,
    std::vector<float> & out,
    int & token_count);

static void pi0_embed_prompt(
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

static void pi0_embed_prompt_tokens(
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
    ggml_tensor * embeddings = ctx.weights.lm_head;
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int width = pi0.llm.width;
    const float scale = std::sqrt(static_cast<float>(width));
    const int64_t vocab = embeddings->ne[1];
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
    const Pi0ComponentRuntime & runtime = ctx.components.llm.runtime;
    Pi0GraphContext gctx(pi0_graph_init_params(pi0_graph_context_size(tensor_bytes)));

    ggml_tensor * ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, static_cast<int64_t>(count));
    ggml_set_input(ids);

    ggml_tensor * rows = ggml_get_rows(gctx, pi0_weight(ctx, embeddings, 2), ids);
    ggml_tensor * y = ggml_scale(gctx, rows, scale);

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_sched_reset(runtime.sched);
    if (!ggml_backend_sched_alloc_graph(runtime.sched, graph)) {
        throw std::runtime_error("pi0 prompt embedding graph allocation failed");
    }
    ggml_backend_tensor_set(ids, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    set_backend_threads(runtime.backends, runtime.n_threads);
    if (ggml_backend_sched_graph_compute(runtime.sched, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml pi0 prompt embedding lookup failed");
    }

    out.resize(count * static_cast<size_t>(width));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    token_count = static_cast<int>(count);
}

static void pi0_prefill_prefix_from_runtime_embeddings(
    const Pi0Context & ctx,
    ggml_tensor * runtime_embeddings,
    int runtime_token_count,
    const std::vector<float> & host_embeddings,
    int host_token_count) {
    const int token_count = runtime_token_count + host_token_count;
    if (token_count <= 0) {
        return;
    }
    const Pi0Config & pi0 = pi0_config(ctx.config);
    const int width = pi0.llm.width;
    if (!pi0_has_language_prefix(ctx) ||
        width <= 0 ||
        pi0.llm.kv_out <= 0 ||
        pi0.llm.q_out % pi0.llm.kv_out != 0 ||
        runtime_token_count < 0 ||
        host_token_count < 0 ||
        host_embeddings.size() != static_cast<size_t>(host_token_count) * static_cast<size_t>(width) ||
        (runtime_token_count > 0 &&
            (runtime_embeddings == nullptr ||
                ggml_n_dims(runtime_embeddings) != 2 ||
                runtime_embeddings->type != GGML_TYPE_F32 ||
                runtime_embeddings->ne[0] != width ||
                runtime_embeddings->ne[1] != runtime_token_count))) {
        throw std::invalid_argument("pi0 prefix runtime embeddings have incompatible shape");
    }

    const int head_dim = pi0.llm.kv_out;
    const int heads = pi0.llm.q_out / head_dim;
    const int kv_heads = pi0.llm.kv_out / head_dim;
    std::vector<int> positions(static_cast<size_t>(token_count), 0);
    for (int i = 0; i < token_count; ++i) {
        positions[static_cast<size_t>(i)] = i;
    }

    std::vector<float> hidden;
    ggml_tensor * runtime_prompt_embeddings = nullptr;
    if (host_token_count > 0) {
        runtime_prompt_embeddings = pi0_persist_host_f32_2d(
            ctx.components.llm.runtime,
            &ctx.prefix_kv.prompt_embeddings,
            host_embeddings.data(),
            width,
            host_token_count);
    }
    const std::vector<float> empty_host_embeddings;
    pi0_prefill_language_prefix_layers_batch(
        ctx,
        empty_host_embeddings,
        positions,
        token_count,
        heads,
        kv_heads,
        head_dim,
        hidden,
        false,
        runtime_embeddings,
        runtime_token_count,
        runtime_prompt_embeddings,
        host_token_count);
    ctx.prefix_kv.token_count = static_cast<size_t>(token_count);
}

void pi0_prefill_prefix(const Pi0Context & ctx, const Pi0Observation & observation) {
    ctx.prefix_kv.reset();
    if (pi0_has_vision_prefix(ctx) && pi0_has_language_prefix(ctx) && !observation.images.empty()) {
        int vision_tokens = 0;
        ggml_tensor * vision_prefix_embeddings = pi0_encode_vision_prefix(
            ctx,
            observation.images,
            ctx.components.llm.runtime,
            &ctx.prefix_kv.vision_prefix_embeddings,
            vision_tokens);
        std::vector<float> prompt_embeddings;
        int prompt_tokens = 0;
        if (!observation.prompt_tokens.empty()) {
            pi0_embed_prompt_tokens(ctx, observation.prompt_tokens, prompt_embeddings, prompt_tokens);
        } else {
            pi0_embed_prompt(ctx, observation.prompt, prompt_embeddings, prompt_tokens);
        }
        pi0_prefill_prefix_from_runtime_embeddings(
            ctx,
            vision_prefix_embeddings,
            vision_tokens,
            prompt_embeddings,
            prompt_tokens);
        return;
    }
}

} // namespace robotcpp::pi0
