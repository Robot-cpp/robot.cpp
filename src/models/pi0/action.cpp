#include "models/pi0/action.h"

#include "models/ggml_runtime.h"
#include "models/pi0/load.h"
#include "sampling/flow.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

bool pi0_velocity_expert_batch_device(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out);
bool pi0_has_action_expert(const Pi0Context & ctx);
void pi0_action_head_batch(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out);
void pi0_suffix_embeddings(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out);

bool pi0_has_action_head(const Pi0Context & ctx) {
    return ctx.weights.action_in_w != nullptr;
}

bool pi0_has_action_expert(const Pi0Context & ctx) {
    const Pi0Config & pi0 = pi0_config(ctx.config);
    return pi0_has_action_head(ctx) &&
        pi0.action.expert_layers > 0 &&
        pi0.action.expert_width > 0 &&
        pi0.action.expert_q_out > 0 &&
        pi0.action.expert_kv_out > 0 &&
        !ctx.weights.action_layers.empty() &&
        ctx.weights.action_layers[0].q != nullptr &&
        ctx.weights.action_layers[0].k != nullptr &&
        ctx.weights.action_layers[0].gate != nullptr;
}

void pi0_state_context(const Pi0Context & ctx, const std::vector<float> & state, std::vector<float> & out) {
    const Tensor * state_w = ctx.weights.state_w;
    const Tensor * state_b = ctx.weights.state_b;
    if (state_w == nullptr || state_b == nullptr) {
        out.clear();
        return;
    }
    pi0_linear_batch(
        ctx,
        *state_w,
        *state_b,
        state,
        1,
        out,
        "ggml pi0 state projection graph compute failed");
}

void pi0_velocity_batch(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out) {
    if (pi0_has_action_expert(ctx)) {
        if (!pi0_velocity_expert_batch_device(ctx, time, actions, state_context, out)) {
            throw std::invalid_argument("pi0 action expert requires fused ggml graph execution");
        }
        return;
    }
    pi0_action_head_batch(ctx, time, actions, state_context, out);
}

void pi0_action_head_batch(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out) {
    const Tensor & in_w = *ctx.weights.action_in_w;
    const Tensor & out_w = *ctx.weights.action_out_w;
    const Tensor & out_b = *ctx.weights.action_out_b;

    const int batch = ctx.config.common.action_horizon;
    const size_t width = static_cast<size_t>(in_w.shape[1]);

    std::vector<float> suffix_tokens;
    pi0_suffix_embeddings(ctx, time, actions, state_context, suffix_tokens);
    const size_t suffix_count = suffix_tokens.size() / width;
    if (suffix_tokens.size() != suffix_count * width || suffix_count < static_cast<size_t>(batch)) {
        throw std::invalid_argument("pi0 suffix tokens have incompatible shape");
    }

    const size_t action_offset = state_context.empty() ? 0 : width;
    std::vector<float> action_expert_tokens(
        suffix_tokens.begin() + static_cast<std::ptrdiff_t>(action_offset),
        suffix_tokens.begin() + static_cast<std::ptrdiff_t>(action_offset + static_cast<size_t>(batch) * width));
    pi0_linear_batch(
        ctx,
        out_w,
        out_b,
        action_expert_tokens,
        batch,
        out,
        "ggml pi0 action output projection graph compute failed");
}

void pi0_suffix_embeddings(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out) {
    const Tensor & in_w = *ctx.weights.action_in_w;
    const Tensor & in_b = *ctx.weights.action_in_b;
    const Tensor & time_in_w = *ctx.weights.action_time_in_w;
    const Tensor & time_in_b = *ctx.weights.action_time_in_b;
    const Tensor & time_out_w = *ctx.weights.action_time_out_w;
    const Tensor & time_out_b = *ctx.weights.action_time_out_b;
    const int batch = ctx.config.common.action_horizon;
    const size_t width = static_cast<size_t>(in_w.shape[1]);

    std::vector<float> action_tokens;
    pi0_linear_batch(
        ctx,
        in_w,
        in_b,
        actions,
        batch,
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
    pi0_linear_batch(
        ctx,
        time_in_w,
        time_in_b,
        action_time,
        batch,
        hidden,
        "ggml pi0 action time input projection graph compute failed",
        true);
    std::vector<float> action_expert_tokens;
    pi0_linear_batch(
        ctx,
        time_out_w,
        time_out_b,
        hidden,
        batch,
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

bool pi0_velocity_expert_batch_device(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out) {
    if (!pi0_has_action_expert(ctx)) {
        return false;
    }
    const Pi0Config & pi0 = pi0_config(ctx.config);
    if (pi0.action.expert_q_out % pi0.action.expert_kv_out != 0) {
        throw std::invalid_argument("pi0 action expert Q/KV widths are incompatible");
    }
    const int layers = pi0.action.expert_layers;
    const int64_t width = pi0.action.expert_width;
    const int64_t q_out = pi0.action.expert_q_out;
    const int64_t kv_out = pi0.action.expert_kv_out;
    const int64_t mlp_width = pi0.action.expert_mlp_width;
    const int batch = ctx.config.common.action_horizon;
    const int action_dim = ctx.config.common.action_dim;
    const int head_dim = pi0.action.expert_kv_out;
    const int heads = pi0.action.expert_q_out / head_dim;
    const int kv_heads = pi0.action.expert_kv_out / head_dim;
    const bool has_state_context = !state_context.empty();
    const int suffix_count = batch + (has_state_context ? 1 : 0);
    size_t prefix_tokens = ctx.prefix_kv.token_count;
    const bool has_prefix_kv = prefix_tokens > 0 &&
        ctx.prefix_kv.k_layers.size() >= static_cast<size_t>(layers) &&
        ctx.prefix_kv.v_layers.size() >= static_cast<size_t>(layers);
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
        throw std::invalid_argument("pi0 action expert fused graph input has incompatible shape");
    }
    const Tensor * in_w = ctx.weights.action_in_w;
    const Tensor * in_b = ctx.weights.action_in_b;
    const Tensor * time_in_w = ctx.weights.action_time_in_w;
    const Tensor * time_in_b = ctx.weights.action_time_in_b;
    const Tensor * time_out_w = ctx.weights.action_time_out_w;
    const Tensor * time_out_b = ctx.weights.action_time_out_b;
    const Tensor * out_w = ctx.weights.action_out_w;
    const Tensor * out_b = ctx.weights.action_out_b;
    if (in_w == nullptr || in_b == nullptr ||
        time_in_w == nullptr || time_in_b == nullptr || time_out_w == nullptr || time_out_b == nullptr ||
        out_w == nullptr || out_b == nullptr) {
        throw std::invalid_argument("missing pi0 action projection tensors");
    }

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
    const GgmlRunner & runner = *ctx.components.action_decoder.runner;
    GgmlContext gctx(runner.init_params(context_size, &ctx, static_cast<uint64_t>(prefix_tokens)));

    ggml_tensor * action_input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, action_dim, batch);
    ggml_tensor * time_input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, width, batch);
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, suffix_count);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, action_input, actions.data(), actions.size() * sizeof(float));
    runner.set_input(inputs, time_input, time_batch.data(), time_batch.size() * sizeof(float));
    runner.set_input(inputs, pos, pos_host.data(), pos_host.size() * sizeof(int32_t));

    ggml_tensor * kq_mask = nullptr;
    if (!attention_mask.empty()) {
        kq_mask = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, kv_tokens, suffix_count, 1);
        runner.set_input(inputs, kq_mask, attention_mask.data(), attention_mask.size() * sizeof(float));
    }

    ggml_tensor * action_tokens = ggml_add(
        gctx,
        ggml_mul_mat(gctx, pi0_weight(ctx, *in_w, 2), action_input),
        pi0_weight(ctx, *in_b, 1));
    ggml_tensor * action_time = ggml_concat(gctx, action_tokens, time_input, 0);
    ggml_tensor * time_hidden = ggml_silu(gctx, ggml_add(
        gctx,
        ggml_mul_mat(gctx, pi0_weight(ctx, *time_in_w, 2), action_time),
        pi0_weight(ctx, *time_in_b, 1)));
    ggml_tensor * hidden = ggml_add(
        gctx,
        ggml_mul_mat(gctx, pi0_weight(ctx, *time_out_w, 2), time_hidden),
        pi0_weight(ctx, *time_out_b, 1));
    if (has_state_context) {
        ggml_tensor * state_token = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, width, 1);
        runner.set_input(inputs, state_token, state_context.data(), state_context.size() * sizeof(float));
        hidden = ggml_concat(gctx, state_token, hidden, 1);
    }

    for (int layer = 0; layer < layers; ++layer) {
        const Pi0TransformerLayerWeights & layer_w = ctx.weights.action_layers[static_cast<size_t>(layer)];
        const Tensor * q_w = layer_w.q;
        const Tensor * k_w = layer_w.k;
        const Tensor * v_w = layer_w.v;
        const Tensor * attn_out_w = layer_w.out;
        const Tensor * gate_w = layer_w.gate;
        const Tensor * up_w = layer_w.up;
        const Tensor * down_w = layer_w.down;
        if (q_w == nullptr || k_w == nullptr || v_w == nullptr || attn_out_w == nullptr ||
            gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
            throw std::invalid_argument("missing pi0 action expert fused graph tensor");
        }

        if (layer_w.input_norm_scale == nullptr || layer_w.post_norm_scale == nullptr) {
            throw std::invalid_argument("missing pi0 action expert norm scale tensor");
        }
        ggml_tensor * input_scale_tensor = pi0_weight(ctx, *layer_w.input_norm_scale, 1);
        ggml_tensor * post_scale_tensor = pi0_weight(ctx, *layer_w.post_norm_scale, 1);

        ggml_tensor * normed = ggml_mul(gctx, ggml_rms_norm(gctx, hidden, 1.0e-6f), input_scale_tensor);
        ggml_tensor * q = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *q_w, 2), normed), q_out, suffix_count);
        ggml_tensor * k = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *k_w, 2), normed), kv_out, suffix_count);
        ggml_tensor * v = ggml_cont_2d(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *v_w, 2), normed), kv_out, suffix_count);
        q = ggml_reshape_3d(gctx, q, head_dim, heads, suffix_count);
        k = ggml_reshape_3d(gctx, k, head_dim, kv_heads, suffix_count);
        v = ggml_reshape_3d(gctx, v, head_dim, kv_heads, suffix_count);
        ggml_tensor * q_rot = ggml_rope_ext(
            gctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k_rot = ggml_rope_ext(
            gctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * k_all = k_rot;
        ggml_tensor * v_all = v;
        if (prefix_tokens > 0) {
            ggml_tensor * prefix_k = ctx.prefix_kv.k_layers[static_cast<size_t>(layer)];
            ggml_tensor * prefix_v = ctx.prefix_kv.v_layers[static_cast<size_t>(layer)];
            if (prefix_k == nullptr || prefix_v == nullptr ||
                prefix_k->ne[0] != head_dim ||
                prefix_k->ne[1] != kv_heads ||
                prefix_k->ne[2] != static_cast<int64_t>(prefix_tokens) ||
                prefix_v->ne[0] != head_dim ||
                prefix_v->ne[1] != kv_heads ||
                prefix_v->ne[2] != static_cast<int64_t>(prefix_tokens)) {
                throw std::invalid_argument("pi0 action expert fused graph prefix KV has incompatible shape");
            }
            k_all = ggml_concat(gctx, prefix_k, k_rot, 2);
            v_all = ggml_concat(gctx, prefix_v, v, 2);
        }
        ggml_tensor * q_perm = ggml_permute(gctx, q_rot, 0, 2, 1, 3);
        ggml_tensor * k_perm = ggml_permute(gctx, k_all, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor * v_perm = ggml_permute(gctx, v_all, 0, 2, 1, 3);
        v_perm = ggml_cont(gctx, ggml_transpose(gctx, v_perm));
        ggml_tensor * scores = ggml_mul_mat(gctx, k_perm, q_perm);
        scores = ggml_soft_max_ext(gctx, scores, kq_mask, scale, 0.0f);
        ggml_tensor * values = ggml_mul_mat(gctx, v_perm, scores);
        ggml_tensor * attn_values = ggml_permute(gctx, values, 0, 2, 1, 3);
        attn_values = ggml_cont_2d(gctx, attn_values, static_cast<int64_t>(head_dim) * heads, suffix_count);
        ggml_tensor * attn_out = ggml_mul_mat(gctx, pi0_weight(ctx, *attn_out_w, 2), attn_values);
        ggml_tensor * first_residual = ggml_add(gctx, hidden, attn_out);

        ggml_tensor * post_norm = ggml_mul(gctx, ggml_rms_norm(gctx, first_residual, 1.0e-6f), post_scale_tensor);
        ggml_tensor * gate = ggml_gelu(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, *gate_w, 2), post_norm));
        ggml_tensor * up = ggml_mul_mat(gctx, pi0_weight(ctx, *up_w, 2), post_norm);
        ggml_tensor * mlp_out = ggml_mul_mat(gctx, pi0_weight(ctx, *down_w, 2), ggml_mul(gctx, gate, up));
        hidden = ggml_add(gctx, first_residual, mlp_out);
    }

    if (ctx.weights.action_norm_scale == nullptr) {
        throw std::invalid_argument("missing pi0 action expert final norm scale tensor");
    }
    ggml_tensor * final_scale_tensor = pi0_weight(ctx, *ctx.weights.action_norm_scale, 1);
    hidden = ggml_mul(gctx, ggml_rms_norm(gctx, hidden, 1.0e-6f), final_scale_tensor);

    const size_t action_offset = suffix_count == batch ? 0 : static_cast<size_t>(width) * sizeof(float);
    ggml_tensor * output_tokens = ggml_view_2d(gctx, hidden, width, batch, hidden->nb[1], action_offset);
    ggml_tensor * y = ggml_add(
        gctx,
        ggml_mul_mat(gctx, pi0_weight(ctx, *out_w, 2), output_tokens),
        pi0_weight(ctx, *out_b, 1));
    ggml_cgraph * graph = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, y);
    runner.compute(gctx, graph, inputs, "ggml action decoder fused graph compute failed");

    out.resize(static_cast<size_t>(batch) * static_cast<size_t>(action_dim));
    runner.get_output(y, out.data(), out.size() * sizeof(float));
    return true;
}

Pi0Sampler::Pi0Sampler(const Pi0Context & ctx)
    : ctx_(ctx) {}

void Pi0Sampler::sample_actions(
    RuntimeConfig & runtime,
    const std::vector<float> & state_context,
    const KvCache & cache,
    const std::vector<float> & initial_noise,
    std::vector<float> & out_actions) const {
    sample_flow_euler(
        runtime.flow_steps,
        runtime.rng,
        ctx_.config.common.action_horizon,
        ctx_.config.common.action_dim,
        initial_noise.empty() ? nullptr : &initial_noise,
        [&](float time, const std::vector<float> & x, std::vector<float> & v) {
            std::vector<float> action_velocity;
            pi0_velocity_batch(
                ctx_,
                time,
                x,
                state_context,
                action_velocity);
            if (action_velocity.size() != v.size()) {
                throw std::invalid_argument("pi0 action velocity has incompatible shape");
            }
            std::copy(action_velocity.begin(), action_velocity.end(), v.begin());
        },
        out_actions);

    denormalize_actions(out_actions);
}

void Pi0Sampler::denormalize_actions(std::vector<float> & actions) const {
    if (ctx_.config.common.action_mean.size() != static_cast<size_t>(ctx_.config.common.action_dim) ||
        ctx_.config.common.action_std.size() != static_cast<size_t>(ctx_.config.common.action_dim)) {
        return;
    }
    for (size_t i = 0; i < actions.size(); ++i) {
        const size_t col = i % static_cast<size_t>(ctx_.config.common.action_dim);
        actions[i] = actions[i] * ctx_.config.common.action_std[col] + ctx_.config.common.action_mean[col];
    }
}

} // namespace vlacpp
