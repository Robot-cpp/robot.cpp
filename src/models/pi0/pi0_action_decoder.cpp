#include "models/pi0/pi0_action_decoder.h"

#include "models/ggml_runtime.h"
#include "models/ggml_ops.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vlacpp {
namespace {

std::vector<float> posemb_sincos(float time, size_t width) {
    std::vector<float> result(width, 0.0f);
    const size_t half = width / 2;
    if (half == 0) {
        return result;
    }
    constexpr float min_period = 4.0e-3f;
    constexpr float max_period = 4.0f;
    constexpr float pi = 3.14159265358979323846f;
    for (size_t i = 0; i < half; ++i) {
        const float fraction = half == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(half - 1);
        const float period = min_period * std::pow(max_period / min_period, fraction);
        const float angle = time / period * 2.0f * pi;
        result[i] = std::sin(angle);
        result[half + i] = std::cos(angle);
    }
    return result;
}

} // namespace

Pi0ActionDecoder::Pi0ActionDecoder(
    const ModelConfig & config,
    const BackendConfig & backend,
    const TensorMap & tensors)
    : config_(config), backend_(backend), tensors_(tensors), action_expert_(config, backend, tensors) {}

bool Pi0ActionDecoder::has_pi0_action_head() const {
    return find_tensor("vlacpp.openpi.action_in_proj.weight") != nullptr;
}

bool Pi0ActionDecoder::has_pi0_action_expert() const {
    return has_pi0_action_head() &&
        config_.openpi_action_expert_layers > 0 &&
        config_.openpi_action_expert_width > 0 &&
        config_.openpi_action_expert_q_out > 0 &&
        config_.openpi_action_expert_kv_out > 0 &&
        action_expert_.has_layer(0);
}

void Pi0ActionDecoder::state_context(const std::vector<float> & state, std::vector<float> & out) const {
    const Tensor * state_w = find_tensor("vlacpp.openpi.state_proj.weight");
    const Tensor * state_b = find_tensor("vlacpp.openpi.state_proj.bias");
    if (state_w == nullptr || state_b == nullptr) {
        out.clear();
        return;
    }
    run_ggml_linear_batch(
        *state_w,
        *state_b,
        state,
        1,
        backend_,
        out,
        "ggml pi0 state projection graph compute failed");
}

void Pi0ActionDecoder::velocity_batch(
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    const std::vector<PrefixLayerKv> & prefix_layers,
    size_t prefix_tokens,
    std::vector<float> & out) const {
    const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
    const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
    const Tensor & out_w = *find_tensor("vlacpp.openpi.action_out_proj.weight");
    const Tensor & out_b = *find_tensor("vlacpp.openpi.action_out_proj.bias");

    const int batch = config_.action_horizon;
    const size_t width = static_cast<size_t>(in_w.shape[1]);

    if (has_pi0_action_expert()) {
        if (velocity_expert_batch_cuda(time, actions, state_context, prefix_layers, prefix_tokens, out)) {
            return;
        }
        std::vector<float> suffix_tokens;
        suffix_embeddings(time, actions, state_context, suffix_tokens);
        const size_t suffix_count = suffix_tokens.size() / width;
        if (suffix_tokens.size() != suffix_count * width || suffix_count < static_cast<size_t>(batch)) {
            throw std::invalid_argument("pi0 suffix tokens have incompatible shape");
        }
        if (config_.openpi_action_expert_q_out % config_.openpi_action_expert_kv_out != 0) {
            throw std::invalid_argument("pi0 action expert Q/KV widths are incompatible");
        }
        const int head_dim = config_.openpi_action_expert_kv_out;
        const int heads = config_.openpi_action_expert_q_out / head_dim;
        const int kv_heads = config_.openpi_action_expert_kv_out / head_dim;
        std::vector<int> positions(suffix_count, 0);
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = static_cast<int>(prefix_tokens + i);
        }
        const size_t prefix_kv_width = static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
        const bool has_prefix_kv =
            prefix_tokens > 0 &&
            prefix_layers.size() >= static_cast<size_t>(config_.openpi_action_expert_layers) &&
            !prefix_layers.empty() &&
            prefix_layers[0].k.size() == prefix_tokens * prefix_kv_width &&
            prefix_layers[0].v.size() == prefix_tokens * prefix_kv_width;
        const size_t kv_tokens = suffix_count + (has_prefix_kv ? prefix_tokens : 0);
        std::vector<float> attention_mask;
        if (!state_context.empty()) {
            attention_mask.assign(suffix_count * kv_tokens, 0.0f);
            const size_t suffix_key_offset = has_prefix_kv ? prefix_tokens : 0;
            for (size_t key = suffix_key_offset + 1; key < kv_tokens; ++key) {
                attention_mask[key] = -INFINITY;
            }
        }
        std::vector<float> hidden = suffix_tokens;
        for (int layer = 0; layer < config_.openpi_action_expert_layers; ++layer) {
            std::vector<float> next;
            if (has_prefix_kv) {
                action_expert_.block_prefix_batch(
                    layer,
                    hidden,
                    positions,
                    prefix_layers[static_cast<size_t>(layer)].k,
                    prefix_layers[static_cast<size_t>(layer)].v,
                    attention_mask,
                    static_cast<int>(prefix_tokens),
                    static_cast<int>(suffix_count),
                    heads,
                    kv_heads,
                    head_dim,
                    next);
            } else {
                action_expert_.block_masked_batch(
                    layer,
                    hidden,
                    positions,
                    attention_mask,
                    static_cast<int>(suffix_count),
                    heads,
                    kv_heads,
                    head_dim,
                    next);
            }
            hidden.swap(next);
        }
        action_expert_.final_norm_batch(hidden, static_cast<int>(suffix_count), suffix_tokens);

        const size_t action_offset = state_context.empty() ? 0 : width;
        std::vector<float> action_expert_tokens(
            suffix_tokens.begin() + static_cast<std::ptrdiff_t>(action_offset),
            suffix_tokens.begin() + static_cast<std::ptrdiff_t>(action_offset + static_cast<size_t>(batch) * width));
        run_ggml_linear_batch(
            out_w,
            out_b,
            action_expert_tokens,
            batch,
            backend_,
            out,
            "ggml pi0 action output projection graph compute failed");
        return;
    }

    std::vector<float> suffix_tokens;
    suffix_embeddings(time, actions, state_context, suffix_tokens);
    const size_t suffix_count = suffix_tokens.size() / width;
    if (suffix_tokens.size() != suffix_count * width || suffix_count < static_cast<size_t>(batch)) {
        throw std::invalid_argument("pi0 suffix tokens have incompatible shape");
    }

    const size_t action_offset = state_context.empty() ? 0 : width;
    std::vector<float> action_expert_tokens(
        suffix_tokens.begin() + static_cast<std::ptrdiff_t>(action_offset),
        suffix_tokens.begin() + static_cast<std::ptrdiff_t>(action_offset + static_cast<size_t>(batch) * width));
    run_ggml_linear_batch(
        out_w,
        out_b,
        action_expert_tokens,
        batch,
        backend_,
        out,
        "ggml pi0 action output projection graph compute failed");
}

void Pi0ActionDecoder::suffix_embeddings(
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out) const {
    const Tensor & in_w = *find_tensor("vlacpp.openpi.action_in_proj.weight");
    const Tensor & in_b = *find_tensor("vlacpp.openpi.action_in_proj.bias");
    const Tensor & time_in_w = *find_tensor("vlacpp.openpi.action_time_mlp_in.weight");
    const Tensor & time_in_b = *find_tensor("vlacpp.openpi.action_time_mlp_in.bias");
    const Tensor & time_out_w = *find_tensor("vlacpp.openpi.action_time_mlp_out.weight");
    const Tensor & time_out_b = *find_tensor("vlacpp.openpi.action_time_mlp_out.bias");
    const int batch = config_.action_horizon;
    const size_t width = static_cast<size_t>(in_w.shape[1]);

    std::vector<float> action_tokens;
    run_ggml_linear_batch(
        in_w,
        in_b,
        actions,
        batch,
        backend_,
        action_tokens,
        "ggml pi0 action input projection graph compute failed");

    const std::vector<float> time_emb = posemb_sincos(time, width);
    std::vector<float> action_time(static_cast<size_t>(batch) * width * 2, 0.0f);
    for (int row = 0; row < batch; ++row) {
        const size_t src = static_cast<size_t>(row) * width;
        const size_t dst = static_cast<size_t>(row) * width * 2;
        std::copy(
            action_tokens.begin() + static_cast<std::ptrdiff_t>(src),
            action_tokens.begin() + static_cast<std::ptrdiff_t>(src + width),
            action_time.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy(
            time_emb.begin(),
            time_emb.end(),
            action_time.begin() + static_cast<std::ptrdiff_t>(dst + width));
    }
    std::vector<float> hidden;
    run_ggml_linear_batch(
        time_in_w,
        time_in_b,
        action_time,
        batch,
        backend_,
        hidden,
        "ggml pi0 action time input projection graph compute failed",
        true);
    std::vector<float> action_expert_tokens;
    run_ggml_linear_batch(
        time_out_w,
        time_out_b,
        hidden,
        batch,
        backend_,
        action_expert_tokens,
        "ggml pi0 action time output projection graph compute failed");

    out.clear();
    out.reserve(action_expert_tokens.size() + state_context.size());
    if (!state_context.empty()) {
        if (state_context.size() != width) {
            throw std::invalid_argument("state token width does not match action expert width");
        }
        out.insert(out.end(), state_context.begin(), state_context.end());
    }
    out.insert(out.end(), action_expert_tokens.begin(), action_expert_tokens.end());
}

bool Pi0ActionDecoder::velocity_expert_batch_cuda(
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    const std::vector<PrefixLayerKv> & prefix_layers,
    size_t prefix_tokens,
    std::vector<float> & out) const {
    if (backend_.backend != VLACPP_BACKEND_CUDA || !has_pi0_action_expert()) {
        return false;
    }
    if (config_.openpi_action_expert_q_out % config_.openpi_action_expert_kv_out != 0) {
        throw std::invalid_argument("pi0 action expert Q/KV widths are incompatible");
    }
    const int layers = config_.openpi_action_expert_layers;
    const int64_t width = config_.openpi_action_expert_width;
    const int64_t q_out = config_.openpi_action_expert_q_out;
    const int64_t kv_out = config_.openpi_action_expert_kv_out;
    const int64_t mlp_width = config_.openpi_action_expert_mlp_width;
    const int batch = config_.action_horizon;
    const int action_dim = config_.action_dim;
    const int head_dim = config_.openpi_action_expert_kv_out;
    const int heads = config_.openpi_action_expert_q_out / head_dim;
    const int kv_heads = config_.openpi_action_expert_kv_out / head_dim;
    const bool has_state_context = !state_context.empty();
    const int suffix_count = batch + (has_state_context ? 1 : 0);
    const size_t prefix_kv_width = static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
    const bool has_prefix_kv =
        prefix_tokens > 0 &&
        prefix_layers.size() >= static_cast<size_t>(layers) &&
        !prefix_layers.empty() &&
        ((prefix_layers[0].k.size() == prefix_tokens * prefix_kv_width &&
          prefix_layers[0].v.size() == prefix_tokens * prefix_kv_width) ||
         (prefix_layers[0].device_cached &&
          prefix_layers[0].k_size == prefix_tokens * prefix_kv_width &&
          prefix_layers[0].v_size == prefix_tokens * prefix_kv_width));
    if (!has_prefix_kv) {
        prefix_tokens = 0;
    }
    const int kv_tokens = suffix_count + static_cast<int>(prefix_tokens);
    if (suffix_count <= 0 || batch <= 0 || action_dim <= 0 || layers <= 0 ||
        width <= 0 || q_out <= 0 || kv_out <= 0 || mlp_width <= 0 ||
        heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_out != static_cast<int64_t>(heads) * head_dim ||
        kv_out != static_cast<int64_t>(kv_heads) * head_dim ||
        actions.size() != static_cast<size_t>(batch) * static_cast<size_t>(action_dim) ||
        (has_state_context && state_context.size() != static_cast<size_t>(width))) {
        throw std::invalid_argument("pi0 action expert fused cuda input has incompatible shape");
    }
    const Tensor * in_w = find_tensor("vlacpp.openpi.action_in_proj.weight");
    const Tensor * in_b = find_tensor("vlacpp.openpi.action_in_proj.bias");
    const Tensor * time_in_w = find_tensor("vlacpp.openpi.action_time_mlp_in.weight");
    const Tensor * time_in_b = find_tensor("vlacpp.openpi.action_time_mlp_in.bias");
    const Tensor * time_out_w = find_tensor("vlacpp.openpi.action_time_mlp_out.weight");
    const Tensor * time_out_b = find_tensor("vlacpp.openpi.action_time_mlp_out.bias");
    const Tensor * out_w = find_tensor("vlacpp.openpi.action_out_proj.weight");
    const Tensor * out_b = find_tensor("vlacpp.openpi.action_out_proj.bias");
    if (in_w == nullptr || in_b == nullptr ||
        time_in_w == nullptr || time_in_b == nullptr || time_out_w == nullptr || time_out_b == nullptr ||
        out_w == nullptr || out_b == nullptr) {
        throw std::invalid_argument("missing pi0 action projection tensors");
    }
    require_ggml_weight_2d(*in_w, action_dim, width, "action input projection");
    require_ggml_vector_1d(*in_b, width, "action input projection bias");
    require_ggml_weight_2d(*time_in_w, 2 * width, width, "action time input projection");
    require_ggml_vector_1d(*time_in_b, width, "action time input projection bias");
    require_ggml_weight_2d(*time_out_w, width, width, "action time output projection");
    require_ggml_vector_1d(*time_out_b, width, "action time output projection bias");
    require_ggml_weight_2d(*out_w, width, action_dim, "action output projection");
    require_ggml_vector_1d(*out_b, action_dim, "action output projection bias");

    auto find_action_tensor = [this](const std::string & name) -> const Tensor * {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            return &it->second;
        }
        it = tensors_.find("model." + name);
        if (it != tensors_.end()) {
            return &it->second;
        }
        return nullptr;
    };
    auto layer_prefix = [](int layer) {
        return "paligemma_with_expert.gemma_expert.model.layers." + std::to_string(layer) + ".";
    };

    std::vector<int32_t> pos_host(static_cast<size_t>(suffix_count));
    for (int i = 0; i < suffix_count; ++i) {
        pos_host[static_cast<size_t>(i)] = static_cast<int32_t>(prefix_tokens + static_cast<size_t>(i));
    }
    const std::vector<float> time_emb = posemb_sincos(time, static_cast<size_t>(width));
    std::vector<float> time_batch(static_cast<size_t>(batch) * static_cast<size_t>(width));
    for (int row = 0; row < batch; ++row) {
        std::copy(
            time_emb.begin(),
            time_emb.end(),
            time_batch.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(row) * static_cast<size_t>(width)));
    }
    std::vector<float> attention_mask;
    if (has_state_context) {
        attention_mask.assign(static_cast<size_t>(suffix_count) * static_cast<size_t>(kv_tokens), 0.0f);
        const size_t suffix_key_offset = prefix_tokens;
        for (size_t key = suffix_key_offset + 1; key < static_cast<size_t>(kv_tokens); ++key) {
            attention_mask[key] = -INFINITY;
        }
    }
    const size_t context_size = 256 * 1024 * 1024;
    GgmlRunner runner(backend_);
    GgmlContext ctx(runner.init_params(context_size, this, static_cast<uint64_t>(prefix_tokens)));

    ggml_tensor * action_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, action_dim, batch);
    ggml_tensor * time_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, suffix_count);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, action_input, actions.data(), actions.size() * sizeof(float));
    runner.set_input(inputs, time_input, time_batch.data(), time_batch.size() * sizeof(float));
    runner.set_input(inputs, pos, pos_host.data(), pos_host.size() * sizeof(int32_t));

    ggml_tensor * kq_mask = nullptr;
    if (!attention_mask.empty()) {
        kq_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv_tokens, suffix_count, 1);
        runner.set_input(inputs, kq_mask, attention_mask.data(), attention_mask.size() * sizeof(float));
    }

    ggml_tensor * action_tokens = build_ggml_linear(ctx, runner, *in_w, *in_b, action_input);
    ggml_tensor * action_time = ggml_concat(ctx, action_tokens, time_input, 0);
    ggml_tensor * time_hidden = ggml_silu(ctx, build_ggml_linear(ctx, runner, *time_in_w, *time_in_b, action_time));
    ggml_tensor * hidden = build_ggml_linear(ctx, runner, *time_out_w, *time_out_b, time_hidden);
    if (has_state_context) {
        ggml_tensor * state_token = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        runner.set_input(inputs, state_token, state_context.data(), state_context.size() * sizeof(float));
        hidden = ggml_concat(ctx, state_token, hidden, 1);
    }

    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = layer_prefix(layer);
        const Tensor * input_norm_w = find_action_tensor(prefix + "input_layernorm.weight");
        const Tensor * post_norm_w = find_action_tensor(prefix + "post_attention_layernorm.weight");
        const Tensor * q_w = find_action_tensor(prefix + "self_attn.q_proj.weight");
        const Tensor * k_w = find_action_tensor(prefix + "self_attn.k_proj.weight");
        const Tensor * v_w = find_action_tensor(prefix + "self_attn.v_proj.weight");
        const Tensor * attn_out_w = find_action_tensor(prefix + "self_attn.o_proj.weight");
        const Tensor * gate_w = find_action_tensor(prefix + "mlp.gate_proj.weight");
        const Tensor * up_w = find_action_tensor(prefix + "mlp.up_proj.weight");
        const Tensor * down_w = find_action_tensor(prefix + "mlp.down_proj.weight");
        if (input_norm_w == nullptr || post_norm_w == nullptr ||
            q_w == nullptr || k_w == nullptr || v_w == nullptr || attn_out_w == nullptr ||
            gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
            throw std::invalid_argument("missing pi0 action expert fused cuda tensor");
        }
        require_ggml_vector_1d(*input_norm_w, width, "action expert input norm");
        require_ggml_vector_1d(*post_norm_w, width, "action expert post norm");
        require_ggml_weight_2d(*q_w, width, q_out, "action expert q_proj");
        require_ggml_weight_2d(*k_w, width, kv_out, "action expert k_proj");
        require_ggml_weight_2d(*v_w, width, kv_out, "action expert v_proj");
        require_ggml_weight_2d(*attn_out_w, q_out, width, "action expert o_proj");
        require_ggml_weight_2d(*gate_w, width, mlp_width, "action expert gate_proj");
        require_ggml_weight_2d(*up_w, width, mlp_width, "action expert up_proj");
        require_ggml_weight_2d(*down_w, mlp_width, width, "action expert down_proj");

        ggml_tensor * input_scale_tensor = runner.new_weight_1d_plus_one(ctx, *input_norm_w);
        ggml_tensor * post_scale_tensor = runner.new_weight_1d_plus_one(ctx, *post_norm_w);

        ggml_tensor * normed = build_ggml_rms_norm(ctx, hidden, input_scale_tensor, 1.0e-6f);
        ggml_tensor * q = ggml_cont_2d(ctx, build_ggml_matmul(ctx, runner, *q_w, normed), q_out, suffix_count);
        ggml_tensor * k = ggml_cont_2d(ctx, build_ggml_matmul(ctx, runner, *k_w, normed), kv_out, suffix_count);
        ggml_tensor * v = ggml_cont_2d(ctx, build_ggml_matmul(ctx, runner, *v_w, normed), kv_out, suffix_count);
        q = ggml_reshape_3d(ctx, q, head_dim, heads, suffix_count);
        k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, suffix_count);
        v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, suffix_count);
        ggml_tensor * q_rot = ggml_rope_ext(
            ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_rot = ggml_rope_ext(
            ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * k_all = k_rot;
        ggml_tensor * v_all = v;
        if (prefix_tokens > 0) {
            const PrefixLayerKv & layer_prefix_kv = prefix_layers[static_cast<size_t>(layer)];
            const size_t prefix_size = prefix_tokens * static_cast<size_t>(kv_out);
            const bool layer_has_host_kv = layer_prefix_kv.k.size() == prefix_size && layer_prefix_kv.v.size() == prefix_size;
            const bool layer_has_device_kv =
                layer_prefix_kv.device_cached &&
                layer_prefix_kv.k_size == prefix_size &&
                layer_prefix_kv.v_size == prefix_size;
            if (!layer_has_host_kv && !layer_has_device_kv) {
                throw std::invalid_argument("pi0 action expert fused cuda prefix KV has incompatible shape");
            }
            ggml_tensor * prefix_k = runner.new_cached_f32_3d(
                ctx,
                &layer_prefix_kv.k,
                layer_prefix_kv.generation,
                layer_has_host_kv ? layer_prefix_kv.k.data() : nullptr,
                head_dim,
                kv_heads,
                static_cast<int64_t>(prefix_tokens));
            ggml_tensor * prefix_v = runner.new_cached_f32_3d(
                ctx,
                &layer_prefix_kv.v,
                layer_prefix_kv.generation,
                layer_has_host_kv ? layer_prefix_kv.v.data() : nullptr,
                head_dim,
                kv_heads,
                static_cast<int64_t>(prefix_tokens));
            k_all = ggml_concat(ctx, prefix_k, k_rot, 2);
            v_all = ggml_concat(ctx, prefix_v, v, 2);
        }
        ggml_tensor * q_perm = ggml_permute(ctx, q_rot, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(ctx, k_all, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor * v_perm = ggml_permute(ctx, v_all, 0, 2, 1, 3);
        v_perm = ggml_cont(ctx, ggml_transpose(ctx, v_perm));
        ggml_tensor * scores = ggml_mul_mat(ctx, k_perm, q_perm);
        scores = ggml_soft_max_ext(ctx, scores, kq_mask, scale, 0.0f);
        ggml_tensor * values = ggml_mul_mat(ctx, v_perm, scores);
        ggml_tensor * attn_values = ggml_permute(ctx, values, 0, 2, 1, 3);
        attn_values = ggml_cont_2d(ctx, attn_values, static_cast<int64_t>(head_dim) * heads, suffix_count);
        ggml_tensor * attn_out = build_ggml_matmul(ctx, runner, *attn_out_w, attn_values);
        ggml_tensor * first_residual = ggml_add(ctx, hidden, attn_out);

        ggml_tensor * post_norm = build_ggml_rms_norm(ctx, first_residual, post_scale_tensor, 1.0e-6f);
        ggml_tensor * mlp_out = build_ggml_gated_mlp(ctx, runner, *gate_w, *up_w, *down_w, post_norm);
        hidden = ggml_add(ctx, first_residual, mlp_out);
    }

    const Tensor * final_norm_w = find_action_tensor("paligemma_with_expert.gemma_expert.model.norm.weight");
    if (final_norm_w == nullptr) {
        throw std::invalid_argument("missing pi0 action expert final norm tensor");
    }
    require_ggml_vector_1d(*final_norm_w, width, "action expert final norm");
    ggml_tensor * final_scale_tensor = runner.new_weight_1d_plus_one(ctx, *final_norm_w);
    hidden = build_ggml_rms_norm(ctx, hidden, final_scale_tensor, 1.0e-6f);

    const size_t action_offset = suffix_count == batch ? 0 : static_cast<size_t>(width) * sizeof(float);
    ggml_tensor * output_tokens = ggml_view_2d(ctx, hidden, width, batch, hidden->nb[1], action_offset);
    ggml_tensor * y = build_ggml_linear(ctx, runner, *out_w, *out_b, output_tokens);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, "ggml action decoder fused cuda graph compute failed");

    out.resize(static_cast<size_t>(batch) * static_cast<size_t>(action_dim));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
    return true;
}

const Tensor * Pi0ActionDecoder::find_tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

} // namespace vlacpp
