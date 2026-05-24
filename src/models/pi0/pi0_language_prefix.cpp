#include "models/pi0/pi0_language_prefix.h"

#include "models/ggml_runtime.h"
#include "models/ggml_ops.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <stdexcept>

namespace vlacpp {
namespace {

void run_norm(
    const Tensor & norm_w,
    const std::vector<float> & tokens,
    int batch,
    int64_t width,
    const BackendConfig & backend,
    std::vector<float> & out) {
    require_ggml_vector_1d(norm_w, width, "language prefix norm");
    if (batch <= 0 || tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("language prefix norm input has incompatible shape");
    }

    const size_t tensor_bytes =
        (norm_w.data.size() + tokens.size() + static_cast<size_t>(width) * static_cast<size_t>(batch)) *
        sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    std::vector<float> scale_host(static_cast<size_t>(width));
    for (int64_t i = 0; i < width; ++i) {
        scale_host[static_cast<size_t>(i)] = 1.0f + norm_w.data[static_cast<size_t>(i)];
    }
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, tokens.data(), tokens.size() * sizeof(float));
    runner.set_input(inputs, scale, scale_host.data(), scale_host.size() * sizeof(float));

    ggml_tensor * y = build_ggml_rms_norm(ctx, x, scale, 1.0e-6f);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml language prefix norm graph compute failed");

    out.resize(static_cast<size_t>(batch) * static_cast<size_t>(width));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
}

} // namespace

Pi0LanguagePrefix::Pi0LanguagePrefix(
    const ModelConfig & config,
    const BackendConfig & backend,
    const TensorMap & tensors)
    : config_(config), backend_(backend), tensors_(tensors) {}

bool Pi0LanguagePrefix::has_layer(int layer) const {
    return find_tensor(layer_prefix(layer) + "mlp.gate_proj.weight") != nullptr &&
        find_tensor(layer_prefix(layer) + "self_attn.q_proj.weight") != nullptr;
}

void Pi0LanguagePrefix::prefill_batch(
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<PrefixLayerKv> & cache,
    std::vector<float> & out,
    uint64_t generation,
    bool need_output) const {
    const size_t width = static_cast<size_t>(config_.openpi_language_width);
    if (batch <= 0 || width == 0 || tokens.size() != static_cast<size_t>(batch) * width ||
        positions.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("language prefix input has incompatible shape");
    }

    if (prefill_layers_batch(tokens, positions, batch, heads, kv_heads, head_dim, cache, out, generation, need_output)) {
        return;
    }

    std::vector<float> hidden = tokens;
    cache.clear();
    cache.reserve(static_cast<size_t>(config_.openpi_language_layers));
    for (int layer = 0; layer < config_.openpi_language_layers; ++layer) {
        PrefixLayerKv layer_cache;
        if (prefill_layer_batch(layer, hidden, positions, batch, heads, kv_heads, head_dim, layer_cache, hidden)) {
            cache.push_back(std::move(layer_cache));
            continue;
        }

        std::vector<float> normed;
        norm_batch(layer, "input_layernorm.weight", hidden, batch, normed);

        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        qkv_batch(layer, normed, batch, q, k, v);

        rope_batch(q, positions, batch, heads, head_dim, q);
        rope_batch(k, positions, batch, kv_heads, head_dim, layer_cache.k);
        layer_cache.v = v;
        cache.push_back(layer_cache);

        std::vector<float> attn_values;
        self_attention_batch(q, cache.back().k, cache.back().v, batch, heads, kv_heads, head_dim, attn_values);

        std::vector<float> attn_out;
        attention_out_batch(layer, attn_values, batch, attn_out);

        std::vector<float> first_residual(hidden.size(), 0.0f);
        for (size_t i = 0; i < first_residual.size(); ++i) {
            first_residual[i] = hidden[i] + attn_out[i];
        }

        std::vector<float> post_norm;
        norm_batch(layer, "post_attention_layernorm.weight", first_residual, batch, post_norm);

        std::vector<float> mlp_out;
        mlp_batch(layer, post_norm, batch, mlp_out);

        hidden.resize(first_residual.size());
        for (size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] = first_residual[i] + mlp_out[i];
        }
    }
    final_norm_batch(hidden, batch, out);
}

bool Pi0LanguagePrefix::prefill_layers_batch(
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<PrefixLayerKv> & cache,
    std::vector<float> & out,
    uint64_t generation,
    bool need_output) const {
    if (backend_.backend != VLACPP_BACKEND_CUDA) {
        return false;
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t q_out = config_.openpi_language_q_out;
    const int64_t kv_out = config_.openpi_language_kv_out;
    const int64_t mlp_width = config_.openpi_language_mlp_width;
    const int layers = config_.openpi_language_layers;
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
    GgmlRunner runner(backend_);
    GgmlContext ctx(runner.init_params(context_size, this, static_cast<uint64_t>(batch)));

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, hidden, tokens.data(), tokens.size() * sizeof(float));
    runner.set_input(inputs, pos, pos_host.data(), pos_host.size() * sizeof(int32_t));

    std::vector<ggml_tensor *> k_outputs;
    std::vector<ggml_tensor *> v_outputs;
    k_outputs.reserve(static_cast<size_t>(layers));
    v_outputs.reserve(static_cast<size_t>(layers));

    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = layer_prefix(layer);
        const Tensor * input_norm_w = find_tensor(prefix + "input_layernorm.weight");
        const Tensor * post_norm_w = find_tensor(prefix + "post_attention_layernorm.weight");
        const Tensor * q_w = find_tensor(prefix + "self_attn.q_proj.weight");
        const Tensor * k_w = find_tensor(prefix + "self_attn.k_proj.weight");
        const Tensor * v_w = find_tensor(prefix + "self_attn.v_proj.weight");
        const Tensor * out_w = find_tensor(prefix + "self_attn.o_proj.weight");
        const Tensor * gate_w = find_tensor(prefix + "mlp.gate_proj.weight");
        const Tensor * up_w = find_tensor(prefix + "mlp.up_proj.weight");
        const Tensor * down_w = find_tensor(prefix + "mlp.down_proj.weight");
        if (input_norm_w == nullptr || post_norm_w == nullptr ||
            q_w == nullptr || k_w == nullptr || v_w == nullptr || out_w == nullptr ||
            gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix fused prefill tensor");
        }
        require_ggml_vector_1d(*input_norm_w, width, "language prefix input norm");
        require_ggml_vector_1d(*post_norm_w, width, "language prefix post attention norm");
        require_ggml_weight_2d(*q_w, width, q_out, "language prefix q_proj");
        require_ggml_weight_2d(*k_w, width, kv_out, "language prefix k_proj");
        require_ggml_weight_2d(*v_w, width, kv_out, "language prefix v_proj");
        require_ggml_weight_2d(*out_w, q_out, width, "language prefix o_proj");
        require_ggml_weight_2d(*gate_w, width, mlp_width, "language prefix gate_proj");
        require_ggml_weight_2d(*up_w, width, mlp_width, "language prefix up_proj");
        require_ggml_weight_2d(*down_w, mlp_width, width, "language prefix down_proj");

        ggml_tensor * input_scale_tensor = runner.new_weight_1d_plus_one(ctx, *input_norm_w);
        ggml_tensor * post_scale_tensor = runner.new_weight_1d_plus_one(ctx, *post_norm_w);

        ggml_tensor * normed = build_ggml_rms_norm(ctx, hidden, input_scale_tensor, 1.0e-6f);
        ggml_tensor * q = ggml_cont_2d(ctx, build_ggml_matmul(ctx, runner, *q_w, normed), q_out, batch);
        ggml_tensor * k = ggml_cont_2d(ctx, build_ggml_matmul(ctx, runner, *k_w, normed), kv_out, batch);
        ggml_tensor * v = ggml_cont_2d(ctx, build_ggml_matmul(ctx, runner, *v_w, normed), kv_out, batch);
        q = ggml_reshape_3d(ctx, q, head_dim, heads, batch);
        k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, batch);
        v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, batch);
        ggml_tensor * q_rot = ggml_rope_ext(
            ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_rot = ggml_rope_ext(
            ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_attn = k_rot;
        ggml_tensor * v_attn = v;
        ggml_tensor * q_perm = ggml_permute(ctx, q_rot, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(ctx, k_attn, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor * v_perm = ggml_permute(ctx, v_attn, 0, 2, 1, 3);
        v_perm = ggml_cont(ctx, ggml_transpose(ctx, v_perm));
        ggml_tensor * scores = ggml_mul_mat(ctx, k_perm, q_perm);
        scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
        ggml_tensor * values = ggml_mul_mat(ctx, v_perm, scores);
        ggml_tensor * attn_values = ggml_permute(ctx, values, 0, 2, 1, 3);
        attn_values = ggml_cont_2d(ctx, attn_values, static_cast<int64_t>(head_dim) * heads, batch);
        ggml_tensor * attn_out = build_ggml_matmul(ctx, runner, *out_w, attn_values);
        ggml_tensor * first_residual = ggml_add(ctx, hidden, attn_out);

        ggml_tensor * post_norm = build_ggml_rms_norm(ctx, first_residual, post_scale_tensor, 1.0e-6f);
        ggml_tensor * mlp_out = build_ggml_gated_mlp(ctx, runner, *gate_w, *up_w, *down_w, post_norm);
        hidden = ggml_add(ctx, first_residual, mlp_out);

        k_outputs.push_back(ggml_cont_3d(ctx, k_rot, head_dim, kv_heads, batch));
        v_outputs.push_back(ggml_cont_3d(ctx, v, head_dim, kv_heads, batch));
    }

    ggml_tensor * y = nullptr;
    if (need_output) {
        const Tensor * final_norm_w = find_tensor("paligemma_with_expert.paligemma.model.language_model.norm.weight");
        if (final_norm_w == nullptr) {
            throw std::invalid_argument("missing pi0 language prefix final norm tensor");
        }
        require_ggml_vector_1d(*final_norm_w, width, "language prefix final norm");
        ggml_tensor * final_scale_tensor = runner.new_weight_1d_plus_one(ctx, *final_norm_w);
        y = build_ggml_rms_norm(ctx, hidden, final_scale_tensor, 1.0e-6f);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    if (need_output) {
        ggml_build_forward_expand(graph, y);
    }
    for (size_t i = 0; i < k_outputs.size(); ++i) {
        ggml_build_forward_expand(graph, k_outputs[i]);
        ggml_build_forward_expand(graph, v_outputs[i]);
    }
    runner.compute(ctx, graph, inputs, "ggml language prefix fused prefill graph compute failed");

    if (need_output) {
        out.resize(static_cast<size_t>(batch) * static_cast<size_t>(width));
        runner.get_output(y, out.data(), out.size() * sizeof(float));
    } else {
        out.clear();
    }
    cache.resize(static_cast<size_t>(layers));
    const size_t kv_size = static_cast<size_t>(batch) * static_cast<size_t>(kv_out);
    for (int layer = 0; layer < layers; ++layer) {
        PrefixLayerKv & layer_cache = cache[static_cast<size_t>(layer)];
        layer_cache.generation = 0;
        layer_cache.k_size = kv_size;
        layer_cache.v_size = kv_size;
        layer_cache.device_cached = true;
        layer_cache.k.clear();
        layer_cache.v.clear();
        runner.new_cached_f32_3d_from_backend(
            ctx,
            &layer_cache.k,
            layer_cache.generation,
            k_outputs[static_cast<size_t>(layer)],
            head_dim,
            kv_heads,
            batch);
        runner.new_cached_f32_3d_from_backend(
            ctx,
            &layer_cache.v,
            layer_cache.generation,
            v_outputs[static_cast<size_t>(layer)],
            head_dim,
            kv_heads,
            batch);
    }
    return true;
}

bool Pi0LanguagePrefix::prefill_layer_batch(
    int layer,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    PrefixLayerKv & layer_cache,
    std::vector<float> & out) const {
    if (backend_.backend != VLACPP_BACKEND_CUDA) {
        return false;
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t q_out = config_.openpi_language_q_out;
    const int64_t kv_out = config_.openpi_language_kv_out;
    const int64_t mlp_width = config_.openpi_language_mlp_width;
    if (batch <= 0 || width <= 0 || q_out <= 0 || kv_out <= 0 || mlp_width <= 0 ||
        heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_out != static_cast<int64_t>(heads) * head_dim ||
        kv_out != static_cast<int64_t>(kv_heads) * head_dim ||
        tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width) ||
        positions.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("language prefix fused layer input has incompatible shape");
    }

    const std::string prefix = layer_prefix(layer);
    const Tensor * input_norm_w = find_tensor(prefix + "input_layernorm.weight");
    const Tensor * post_norm_w = find_tensor(prefix + "post_attention_layernorm.weight");
    const Tensor * q_w = find_tensor(prefix + "self_attn.q_proj.weight");
    const Tensor * k_w = find_tensor(prefix + "self_attn.k_proj.weight");
    const Tensor * v_w = find_tensor(prefix + "self_attn.v_proj.weight");
    const Tensor * out_w = find_tensor(prefix + "self_attn.o_proj.weight");
    const Tensor * gate_w = find_tensor(prefix + "mlp.gate_proj.weight");
    const Tensor * up_w = find_tensor(prefix + "mlp.up_proj.weight");
    const Tensor * down_w = find_tensor(prefix + "mlp.down_proj.weight");
    if (input_norm_w == nullptr || post_norm_w == nullptr ||
        q_w == nullptr || k_w == nullptr || v_w == nullptr || out_w == nullptr ||
        gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix fused layer tensor");
    }
    require_ggml_vector_1d(*input_norm_w, width, "language prefix input norm");
    require_ggml_vector_1d(*post_norm_w, width, "language prefix post attention norm");
    require_ggml_weight_2d(*q_w, width, q_out, "language prefix q_proj");
    require_ggml_weight_2d(*k_w, width, kv_out, "language prefix k_proj");
    require_ggml_weight_2d(*v_w, width, kv_out, "language prefix v_proj");
    require_ggml_weight_2d(*out_w, q_out, width, "language prefix o_proj");
    require_ggml_weight_2d(*gate_w, width, mlp_width, "language prefix gate_proj");
    require_ggml_weight_2d(*up_w, width, mlp_width, "language prefix up_proj");
    require_ggml_weight_2d(*down_w, mlp_width, width, "language prefix down_proj");

    std::vector<float> input_scale(static_cast<size_t>(width));
    std::vector<float> post_scale(static_cast<size_t>(width));
    for (int64_t i = 0; i < width; ++i) {
        input_scale[static_cast<size_t>(i)] = 1.0f + input_norm_w->data[static_cast<size_t>(i)];
        post_scale[static_cast<size_t>(i)] = 1.0f + post_norm_w->data[static_cast<size_t>(i)];
    }
    std::vector<int32_t> pos_host(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        pos_host[static_cast<size_t>(i)] = static_cast<int32_t>(positions[static_cast<size_t>(i)]);
    }

    const size_t context_size = 96 * 1024 * 1024;
    GgmlRunner runner(backend_);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * input_scale_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    ggml_tensor * post_scale_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, tokens.data(), tokens.size() * sizeof(float));
    runner.set_input(inputs, input_scale_tensor, input_scale.data(), input_scale.size() * sizeof(float));
    runner.set_input(inputs, post_scale_tensor, post_scale.data(), post_scale.size() * sizeof(float));
    runner.set_input(inputs, pos, pos_host.data(), pos_host.size() * sizeof(int32_t));

    ggml_tensor * normed = build_ggml_rms_norm(ctx, x, input_scale_tensor, 1.0e-6f);
    ggml_tensor * q = build_ggml_matmul(ctx, runner, *q_w, normed);
    ggml_tensor * k = build_ggml_matmul(ctx, runner, *k_w, normed);
    ggml_tensor * v = build_ggml_matmul(ctx, runner, *v_w, normed);
    q = ggml_cont_2d(ctx, q, q_out, batch);
    k = ggml_cont_2d(ctx, k, kv_out, batch);
    v = ggml_cont_2d(ctx, v, kv_out, batch);
    q = ggml_reshape_3d(ctx, q, head_dim, heads, batch);
    k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, batch);
    v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, batch);
    ggml_tensor * q_rot = ggml_rope_ext(
        ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rot = ggml_rope_ext(
        ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_attn = k_rot;
    ggml_tensor * v_attn = v;
    ggml_tensor * q_perm = ggml_permute(ctx, q_rot, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx, k_attn, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx, v_attn, 0, 2, 1, 3);
    v_perm = ggml_cont(ctx, ggml_transpose(ctx, v_perm));
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * scores = ggml_mul_mat(ctx, k_perm, q_perm);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
    ggml_tensor * values = ggml_mul_mat(ctx, v_perm, scores);
    ggml_tensor * attn_values = ggml_permute(ctx, values, 0, 2, 1, 3);
    attn_values = ggml_cont_2d(ctx, attn_values, static_cast<int64_t>(head_dim) * heads, batch);
    ggml_tensor * attn_out = build_ggml_matmul(ctx, runner, *out_w, attn_values);
    ggml_tensor * first_residual = ggml_add(ctx, x, attn_out);

    ggml_tensor * post_norm = build_ggml_rms_norm(ctx, first_residual, post_scale_tensor, 1.0e-6f);
    ggml_tensor * mlp_out = build_ggml_gated_mlp(ctx, runner, *gate_w, *up_w, *down_w, post_norm);
    ggml_tensor * y = ggml_add(ctx, first_residual, mlp_out);
    ggml_tensor * k_cache = ggml_cont_3d(ctx, k_rot, head_dim, kv_heads, batch);
    ggml_tensor * v_cache = ggml_cont_3d(ctx, v, head_dim, kv_heads, batch);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, y);
    ggml_build_forward_expand(graph, k_cache);
    ggml_build_forward_expand(graph, v_cache);
    runner.compute(ctx, graph, inputs, "ggml language prefix fused layer graph compute failed");

    out.resize(static_cast<size_t>(batch) * static_cast<size_t>(width));
    layer_cache.k.resize(static_cast<size_t>(batch) * static_cast<size_t>(kv_out));
    layer_cache.v.resize(static_cast<size_t>(batch) * static_cast<size_t>(kv_out));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
    runner.get_output(k_cache, layer_cache.k.data(), layer_cache.k.size() * sizeof(float));
    runner.get_output(v_cache, layer_cache.v.data(), layer_cache.v.size() * sizeof(float));
    return true;
}

void Pi0LanguagePrefix::norm_batch(
    int layer,
    const char * weight_name,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    const Tensor * norm_w = find_tensor(layer_prefix(layer) + weight_name);
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix norm tensor");
    }
    run_norm(*norm_w, tokens, batch, config_.openpi_language_width, backend_, out);
}

void Pi0LanguagePrefix::final_norm_batch(const std::vector<float> & tokens, int batch, std::vector<float> & out) const {
    const Tensor * norm_w = find_tensor("paligemma_with_expert.paligemma.model.language_model.norm.weight");
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix final norm tensor");
    }
    run_norm(*norm_w, tokens, batch, config_.openpi_language_width, backend_, out);
}

void Pi0LanguagePrefix::qkv_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & q,
    std::vector<float> & k,
    std::vector<float> & v) const {
    const std::string prefix = layer_prefix(layer) + "self_attn.";
    const Tensor * q_w = find_tensor(prefix + "q_proj.weight");
    const Tensor * k_w = find_tensor(prefix + "k_proj.weight");
    const Tensor * v_w = find_tensor(prefix + "v_proj.weight");
    if (q_w == nullptr || k_w == nullptr || v_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix QKV tensors");
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t q_out = config_.openpi_language_q_out;
    const int64_t kv_out = config_.openpi_language_kv_out;
    require_ggml_weight_2d(*q_w, width, q_out, "language prefix q_proj");
    require_ggml_weight_2d(*k_w, width, kv_out, "language prefix k_proj");
    require_ggml_weight_2d(*v_w, width, kv_out, "language prefix v_proj");
    if (batch <= 0 || tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("language prefix QKV input has incompatible shape");
    }

    run_ggml_qkv_batch(
        *q_w,
        *k_w,
        *v_w,
        tokens,
        batch,
        backend_,
        q,
        k,
        v,
        "ggml language prefix QKV graph compute failed");
}

void Pi0LanguagePrefix::rope_batch(
    const std::vector<float> & values,
    const std::vector<int> & positions,
    int tokens,
    int heads,
    int head_dim,
    std::vector<float> & out) const {
    if (tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        positions.size() != static_cast<size_t>(tokens)) {
        throw std::invalid_argument("invalid pi0 language prefix RoPE dimensions");
    }
    const size_t value_size =
        static_cast<size_t>(tokens) * static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
    if (values.size() != value_size) {
        throw std::invalid_argument("language prefix RoPE input has incompatible shape");
    }

    const size_t tensor_bytes = (values.size() + positions.size()) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend_);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    std::vector<int32_t> pos_host(static_cast<size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
        pos_host[static_cast<size_t>(i)] = static_cast<int32_t>(positions[static_cast<size_t>(i)]);
    }
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, values.data(), values.size() * sizeof(float));
    runner.set_input(inputs, pos, pos_host.data(), pos_host.size() * sizeof(int32_t));

    ggml_tensor * y = ggml_rope_ext(
        ctx, x, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml language prefix RoPE graph compute failed");

    out.resize(value_size);
    runner.get_output(y, out.data(), out.size() * sizeof(float));
}

void Pi0LanguagePrefix::self_attention_batch(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int tokens,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    if (tokens <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 || heads % kv_heads != 0) {
        throw std::invalid_argument("invalid pi0 language prefix attention dimensions");
    }
    const size_t q_size = static_cast<size_t>(tokens) * static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
    const size_t kv_size = static_cast<size_t>(tokens) * static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
    if (q.size() != q_size || k.size() != kv_size || v.size() != kv_size) {
        throw std::invalid_argument("language prefix attention inputs must have matching Q/K/V shape");
    }

    const size_t tensor_bytes = (q_size * 2 + kv_size * 2) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes * 2);
    GgmlRunner runner(backend_);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, tokens);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, kv_heads, tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, kv_heads, tokens);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, q_cur, q.data(), q.size() * sizeof(float));
    runner.set_input(inputs, k_cur, k.data(), k.size() * sizeof(float));
    runner.set_input(inputs, v_cur, v.data(), v.size() * sizeof(float));
    ggml_tensor * y = build_ggml_attention(
        ctx,
        q_cur,
        k_cur,
        v_cur,
        nullptr,
        heads,
        kv_heads,
        head_dim,
        tokens,
        tokens);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml language prefix attention graph compute failed");

    out.resize(q_size);
    runner.get_output(y, out.data(), out.size() * sizeof(float));
}

void Pi0LanguagePrefix::attention_out_batch(
    int layer,
    const std::vector<float> & values,
    int batch,
    std::vector<float> & out) const {
    const Tensor * out_w = find_tensor(layer_prefix(layer) + "self_attn.o_proj.weight");
    if (out_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix attention output tensor");
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t q_out = config_.openpi_language_q_out;
    require_ggml_weight_2d(*out_w, q_out, width, "language prefix o_proj");
    if (batch <= 0 || values.size() != static_cast<size_t>(batch) * static_cast<size_t>(q_out)) {
        throw std::invalid_argument("language prefix attention output input has incompatible shape");
    }

    run_ggml_matmul_batch(
        *out_w,
        values,
        batch,
        backend_,
        out,
        "ggml language prefix attention output graph compute failed");
}

void Pi0LanguagePrefix::mlp_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    const std::string prefix = layer_prefix(layer);
    const Tensor * gate_w = find_tensor(prefix + "mlp.gate_proj.weight");
    const Tensor * up_w = find_tensor(prefix + "mlp.up_proj.weight");
    const Tensor * down_w = find_tensor(prefix + "mlp.down_proj.weight");
    if (gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
        throw std::invalid_argument("missing pi0 language prefix MLP tensors");
    }
    const int64_t width = config_.openpi_language_width;
    const int64_t hidden = config_.openpi_language_mlp_width;
    require_ggml_weight_2d(*gate_w, width, hidden, "language prefix gate_proj");
    require_ggml_weight_2d(*up_w, width, hidden, "language prefix up_proj");
    require_ggml_weight_2d(*down_w, hidden, width, "language prefix down_proj");
    if (batch <= 0 || tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("language prefix token batch has incompatible shape");
    }

    run_ggml_gated_mlp_batch(
        *gate_w,
        *up_w,
        *down_w,
        tokens,
        batch,
        backend_,
        out,
        "ggml language prefix MLP graph compute failed");
}

const Tensor * Pi0LanguagePrefix::find_tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    it = tensors_.find("model." + name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::string Pi0LanguagePrefix::layer_prefix(int layer) const {
    if (layer < 0 || layer >= config_.openpi_language_layers) {
        throw std::invalid_argument("language prefix layer index is out of range");
    }
    return "paligemma_with_expert.paligemma.model.language_model.layers." + std::to_string(layer) + ".";
}

} // namespace vlacpp
