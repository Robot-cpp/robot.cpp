#include "models/pi0/pi0_action_expert.h"

#include "models/ggml_runtime.h"
#include "models/ggml_ops.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <stdexcept>

namespace vlacpp {

Pi0ActionExpert::Pi0ActionExpert(
    const ModelConfig & config,
    const BackendConfig & backend,
    const TensorMap & tensors)
    : config_(config), backend_(backend), tensors_(tensors) {}

bool Pi0ActionExpert::has_layer(int layer) const {
    return find_tensor(layer_prefix(layer) + "mlp.gate_proj.weight") != nullptr &&
        find_tensor(layer_prefix(layer) + "mlp.up_proj.weight") != nullptr &&
        find_tensor(layer_prefix(layer) + "mlp.down_proj.weight") != nullptr;
}

void Pi0ActionExpert::input_norm_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    norm_batch(layer, "input_layernorm.weight", tokens, batch, out);
}

void Pi0ActionExpert::post_attention_norm_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    norm_batch(layer, "post_attention_layernorm.weight", tokens, batch, out);
}

void Pi0ActionExpert::final_norm_batch(
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    const Tensor * norm_w = find_tensor("paligemma_with_expert.gemma_expert.model.norm.weight");
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert final norm tensor");
    }
    norm_tensor_batch(*norm_w, tokens, batch, out);
}

void Pi0ActionExpert::norm_batch(
    int layer,
    const char * weight_name,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    const Tensor * norm_w = find_tensor(layer_prefix(layer) + weight_name);
    if (norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert norm tensor");
    }
    norm_tensor_batch(*norm_w, tokens, batch, out);
}

void Pi0ActionExpert::norm_tensor_batch(
    const Tensor & norm_w,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    if (batch <= 0 || config_.openpi_action_expert_width <= 0) {
        throw std::invalid_argument("invalid pi0 action expert norm dimensions");
    }
    const int64_t width = config_.openpi_action_expert_width;
    require_ggml_vector_1d(norm_w, width, "action expert norm");
    if (tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("action expert norm input has incompatible shape");
    }

    const size_t tensor_bytes =
        (norm_w.data.size() + tokens.size() + static_cast<size_t>(width) * static_cast<size_t>(batch)) *
        sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend_);
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
    runner.compute(ctx, graph, inputs, "ggml action expert norm graph compute failed");

    out.resize(static_cast<size_t>(batch) * static_cast<size_t>(width));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
}

void Pi0ActionExpert::qkv_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & q,
    std::vector<float> & k,
    std::vector<float> & v) const {
    if (batch <= 0 ||
        config_.openpi_action_expert_width <= 0 ||
        config_.openpi_action_expert_q_out <= 0 ||
        config_.openpi_action_expert_kv_out <= 0) {
        throw std::invalid_argument("invalid pi0 action expert QKV dimensions");
    }
    const std::string prefix = layer_prefix(layer) + "self_attn.";
    const Tensor * q_w = find_tensor(prefix + "q_proj.weight");
    const Tensor * k_w = find_tensor(prefix + "k_proj.weight");
    const Tensor * v_w = find_tensor(prefix + "v_proj.weight");
    if (q_w == nullptr || k_w == nullptr || v_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert QKV tensors");
    }
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t q_out = config_.openpi_action_expert_q_out;
    const int64_t kv_out = config_.openpi_action_expert_kv_out;
    require_ggml_weight_2d(*q_w, width, q_out, "action expert q_proj");
    require_ggml_weight_2d(*k_w, width, kv_out, "action expert k_proj");
    require_ggml_weight_2d(*v_w, width, kv_out, "action expert v_proj");
    if (tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("action expert QKV input has incompatible shape");
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
        "ggml action expert QKV graph compute failed");
}

void Pi0ActionExpert::self_attention_batch(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int tokens,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    self_attention_masked_batch(q, k, v, {}, tokens, heads, kv_heads, head_dim, out);
}

void Pi0ActionExpert::self_attention_masked_batch(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    const std::vector<float> & attention_mask,
    int tokens,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    attention_masked_batch(q, k, v, attention_mask, tokens, tokens, heads, kv_heads, head_dim, out);
}

void Pi0ActionExpert::attention_masked_batch(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    const std::vector<float> & attention_mask,
    int q_tokens,
    int kv_tokens,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    if (q_tokens <= 0 || kv_tokens <= 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 || heads % kv_heads != 0) {
        throw std::invalid_argument("invalid pi0 action expert attention dimensions");
    }
    const size_t q_size =
        static_cast<size_t>(q_tokens) * static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
    const size_t kv_size =
        static_cast<size_t>(kv_tokens) * static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
    if (q.size() != q_size || k.size() != kv_size || v.size() != kv_size) {
        throw std::invalid_argument("action expert attention inputs must have matching Q/K/V shape");
    }
    if (!attention_mask.empty() &&
        attention_mask.size() != static_cast<size_t>(q_tokens) * static_cast<size_t>(kv_tokens)) {
        throw std::invalid_argument("action expert attention mask has incompatible shape");
    }

    const size_t tensor_bytes = (q_size * 2 + kv_size * 2 + attention_mask.size()) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes * 2);
    GgmlRunner runner(backend_);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * q_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, q_tokens);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, kv_heads, kv_tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, kv_heads, kv_tokens);
    ggml_tensor * kq_mask = nullptr;
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, q_cur, q.data(), q.size() * sizeof(float));
    runner.set_input(inputs, k_cur, k.data(), k.size() * sizeof(float));
    runner.set_input(inputs, v_cur, v.data(), v.size() * sizeof(float));
    if (!attention_mask.empty()) {
        kq_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv_tokens, q_tokens, 1);
        runner.set_input(inputs, kq_mask, attention_mask.data(), attention_mask.size() * sizeof(float));
    }
    ggml_tensor * y = build_ggml_attention(
        ctx,
        q_cur,
        k_cur,
        v_cur,
        kq_mask,
        heads,
        kv_heads,
        head_dim,
        q_tokens,
        kv_tokens);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml action expert attention graph compute failed");

    out.resize(q_size);
    runner.get_output(y, out.data(), out.size() * sizeof(float));
}

void Pi0ActionExpert::rope_batch(
    const std::vector<float> & values,
    const std::vector<int> & positions,
    int tokens,
    int heads,
    int head_dim,
    std::vector<float> & out) const {
    if (tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        positions.size() != static_cast<size_t>(tokens)) {
        throw std::invalid_argument("invalid pi0 action expert RoPE dimensions");
    }
    const size_t value_size =
        static_cast<size_t>(tokens) * static_cast<size_t>(heads) * static_cast<size_t>(head_dim);
    if (values.size() != value_size) {
        throw std::invalid_argument("action expert RoPE input has incompatible shape");
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
        ctx,
        x,
        pos,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        8192,
        10000.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml action expert RoPE graph compute failed");

    out.resize(value_size);
    runner.get_output(y, out.data(), out.size() * sizeof(float));
}

void Pi0ActionExpert::attention_out_batch(
    int layer,
    const std::vector<float> & values,
    int batch,
    std::vector<float> & out) const {
    if (batch <= 0 ||
        config_.openpi_action_expert_width <= 0 ||
        config_.openpi_action_expert_q_out <= 0) {
        throw std::invalid_argument("invalid pi0 action expert attention output dimensions");
    }
    const Tensor * out_w = find_tensor(layer_prefix(layer) + "self_attn.o_proj.weight");
    if (out_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert attention output tensor");
    }
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t q_out = config_.openpi_action_expert_q_out;
    require_ggml_weight_2d(*out_w, q_out, width, "action expert o_proj");
    if (values.size() != static_cast<size_t>(batch) * static_cast<size_t>(q_out)) {
        throw std::invalid_argument("action expert attention output input has incompatible shape");
    }

    run_ggml_matmul_batch(
        *out_w,
        values,
        batch,
        backend_,
        out,
        "ggml action expert attention output graph compute failed");
}

void Pi0ActionExpert::mlp_batch(
    int layer,
    const std::vector<float> & tokens,
    int batch,
    std::vector<float> & out) const {
    if (batch <= 0 ||
        config_.openpi_action_expert_width <= 0 ||
        config_.openpi_action_expert_mlp_width <= 0) {
        throw std::invalid_argument("invalid pi0 action expert MLP dimensions");
    }
    const std::string prefix = layer_prefix(layer);
    const Tensor * gate_w = find_tensor(prefix + "mlp.gate_proj.weight");
    const Tensor * up_w = find_tensor(prefix + "mlp.up_proj.weight");
    const Tensor * down_w = find_tensor(prefix + "mlp.down_proj.weight");
    if (gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert MLP tensors");
    }
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t hidden = config_.openpi_action_expert_mlp_width;
    require_ggml_weight_2d(*gate_w, width, hidden, "action expert gate_proj");
    require_ggml_weight_2d(*up_w, width, hidden, "action expert up_proj");
    require_ggml_weight_2d(*down_w, hidden, width, "action expert down_proj");
    if (tokens.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("action expert token batch has incompatible shape");
    }

    run_ggml_gated_mlp_batch(
        *gate_w,
        *up_w,
        *down_w,
        tokens,
        batch,
        backend_,
        out,
        "ggml action expert MLP graph compute failed");
}

void Pi0ActionExpert::block_batch(
    int layer,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    block_masked_batch(layer, tokens, positions, {}, batch, heads, kv_heads, head_dim, out);
}

void Pi0ActionExpert::block_masked_batch(
    int layer,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    const std::vector<float> & attention_mask,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    const size_t width = static_cast<size_t>(config_.openpi_action_expert_width);
    if (batch <= 0 || width == 0 || tokens.size() != static_cast<size_t>(batch) * width) {
        throw std::invalid_argument("action expert block input has incompatible shape");
    }

    std::vector<float> normed;
    input_norm_batch(layer, tokens, batch, normed);

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    qkv_batch(layer, normed, batch, q, k, v);

    std::vector<float> q_rot;
    std::vector<float> k_rot;
    rope_batch(q, positions, batch, heads, head_dim, q_rot);
    rope_batch(k, positions, batch, kv_heads, head_dim, k_rot);

    std::vector<float> attn_values;
    self_attention_masked_batch(q_rot, k_rot, v, attention_mask, batch, heads, kv_heads, head_dim, attn_values);

    std::vector<float> attn_out;
    attention_out_batch(layer, attn_values, batch, attn_out);

    std::vector<float> first_residual(tokens.size(), 0.0f);
    for (size_t i = 0; i < first_residual.size(); ++i) {
        first_residual[i] = tokens[i] + attn_out[i];
    }

    std::vector<float> post_norm;
    post_attention_norm_batch(layer, first_residual, batch, post_norm);

    std::vector<float> mlp_out;
    mlp_batch(layer, post_norm, batch, mlp_out);

    out.resize(first_residual.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = first_residual[i] + mlp_out[i];
    }
}

void Pi0ActionExpert::block_prefix_batch(
    int layer,
    const std::vector<float> & tokens,
    const std::vector<int> & positions,
    const std::vector<float> & prefix_k,
    const std::vector<float> & prefix_v,
    const std::vector<float> & attention_mask,
    int prefix_tokens,
    int batch,
    int heads,
    int kv_heads,
    int head_dim,
    std::vector<float> & out) const {
    if (prefix_tokens < 0) {
        throw std::invalid_argument("action expert prefix token count is out of range");
    }
    const size_t prefix_size =
        static_cast<size_t>(prefix_tokens) * static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
    if (prefix_k.size() != prefix_size || prefix_v.size() != prefix_size) {
        throw std::invalid_argument("action expert prefix KV tensors have incompatible shape");
    }
    const size_t kv_token_count = static_cast<size_t>(prefix_tokens) + static_cast<size_t>(batch);
    if (!attention_mask.empty() &&
        attention_mask.size() != static_cast<size_t>(batch) * kv_token_count) {
        throw std::invalid_argument("action expert prefix attention mask has incompatible shape");
    }

    const size_t width = static_cast<size_t>(config_.openpi_action_expert_width);
    if (batch <= 0 || width == 0 || tokens.size() != static_cast<size_t>(batch) * width) {
        throw std::invalid_argument("action expert block input has incompatible shape");
    }

    std::vector<float> normed;
    input_norm_batch(layer, tokens, batch, normed);

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    qkv_batch(layer, normed, batch, q, k, v);

    std::vector<float> q_rot;
    std::vector<float> k_rot;
    rope_batch(q, positions, batch, heads, head_dim, q_rot);
    rope_batch(k, positions, batch, kv_heads, head_dim, k_rot);

    std::vector<float> k_all;
    std::vector<float> v_all;
    k_all.reserve(prefix_k.size() + k_rot.size());
    v_all.reserve(prefix_v.size() + v.size());
    k_all.insert(k_all.end(), prefix_k.begin(), prefix_k.end());
    k_all.insert(k_all.end(), k_rot.begin(), k_rot.end());
    v_all.insert(v_all.end(), prefix_v.begin(), prefix_v.end());
    v_all.insert(v_all.end(), v.begin(), v.end());

    std::vector<float> attn_values;
    attention_masked_batch(
        q_rot,
        k_all,
        v_all,
        attention_mask,
        batch,
        static_cast<int>(kv_token_count),
        heads,
        kv_heads,
        head_dim,
        attn_values);

    std::vector<float> attn_out;
    attention_out_batch(layer, attn_values, batch, attn_out);

    std::vector<float> first_residual(tokens.size(), 0.0f);
    for (size_t i = 0; i < first_residual.size(); ++i) {
        first_residual[i] = tokens[i] + attn_out[i];
    }

    std::vector<float> post_norm;
    post_attention_norm_batch(layer, first_residual, batch, post_norm);

    std::vector<float> mlp_out;
    mlp_batch(layer, post_norm, batch, mlp_out);

    out.resize(first_residual.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = first_residual[i] + mlp_out[i];
    }
}

const Tensor * Pi0ActionExpert::find_tensor(const std::string & name) const {
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

std::string Pi0ActionExpert::layer_prefix(int layer) const {
    if (layer < 0 || layer >= config_.openpi_action_expert_layers) {
        throw std::invalid_argument("action expert layer index is out of range");
    }
    return "paligemma_with_expert.gemma_expert.model.layers." + std::to_string(layer) + ".";
}

} // namespace vlacpp
