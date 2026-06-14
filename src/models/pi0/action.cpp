#include "models/pi0/action.h"

#include "models/pi0/component_runtime.h"
#include "models/pi0/load.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace robotcpp::pi0 {

namespace {

void fill_posemb_sincos(float time, float * result, size_t width) {
    std::fill(result, result + width, 0.0f);
    const size_t half = width / 2;
    if (half == 0) {
        return;
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
}

template <typename VelocityFn>
void sample_flow_euler(
    int steps,
    std::mt19937 & rng,
    int horizon,
    int action_dim,
    const std::vector<float> * initial_noise,
    VelocityFn velocity,
    std::vector<float> & out) {
    const int n_steps = std::max(1, steps);
    const size_t n = static_cast<size_t>(horizon) * static_cast<size_t>(action_dim);
    out.resize(n);

    if (initial_noise != nullptr && initial_noise->size() == n) {
        out = *initial_noise;
    } else {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float & value : out) {
            value = normal(rng);
        }
    }

    std::vector<float> v(n, 0.0f);
    float time = 1.0f;
    const float dt = -1.0f / static_cast<float>(n_steps);
    for (int i = 0; i < n_steps; ++i) {
        velocity(time, out, v);
        if (v.size() != n) {
            throw std::invalid_argument("pi0 action velocity has incompatible shape");
        }
        for (size_t j = 0; j < n; ++j) {
            out[j] += dt * v[j];
        }
        time += dt;
    }
}

} // namespace

void pi0_state_context(const Pi0Context & ctx, const std::vector<float> & state, std::vector<float> & out) {
    ggml_tensor * state_w = ctx.weights.state_w;
    ggml_tensor * state_b = ctx.weights.state_b;
    if (state_w == nullptr || state_b == nullptr) {
        out.clear();
        return;
    }
    pi0_linear_batch(
        ctx,
        ctx.components.state.runtime,
        state_w,
        state_b,
        state,
        1,
        out,
        "ggml pi0 state projection graph compute failed");
}

namespace {

class Pi0ActionVelocityGraph {
public:
    Pi0ActionVelocityGraph(const Pi0Context & ctx, const std::vector<float> & state_context)
        : ctx_(ctx),
          state_context_(state_context),
          runtime_(ctx.components.action_decoder.runtime),
          gctx_(pi0_graph_init_params(256 * 1024 * 1024)) {
        init_shape();
        build_graph();
    }

    void eval(float time, const std::vector<float> & actions, std::vector<float> & out) {
        if (actions.size() != static_cast<size_t>(batch_) * static_cast<size_t>(action_dim_)) {
            throw std::invalid_argument("pi0 action expert fused graph input has incompatible shape");
        }

        fill_time_batch(time);
        ggml_backend_tensor_set(action_input_, actions.data(), 0, actions.size() * sizeof(float));
        ggml_backend_tensor_set(time_input_, time_batch_.data(), 0, time_batch_.size() * sizeof(float));
        set_backend_threads(runtime_.backends, runtime_.n_threads);
        if (ggml_backend_sched_graph_compute(runtime_.sched, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml action decoder fused graph compute failed");
        }

        out.resize(static_cast<size_t>(batch_) * static_cast<size_t>(action_dim_));
        ggml_backend_tensor_get(y_, out.data(), 0, out.size() * sizeof(float));
    }

private:
    void init_shape() {
        const Pi0Config & pi0 = pi0_config(ctx_.config);
        if (ctx_.weights.action_in_w == nullptr ||
            pi0.action.expert_layers <= 0 ||
            pi0.action.expert_width <= 0 ||
            pi0.action.expert_q_out <= 0 ||
            pi0.action.expert_kv_out <= 0 ||
            ctx_.weights.action_layers.empty() ||
            ctx_.weights.action_layers[0].q == nullptr ||
            ctx_.weights.action_layers[0].k == nullptr ||
            ctx_.weights.action_layers[0].gate == nullptr) {
            throw std::invalid_argument("pi0 action expert requires fused ggml graph execution");
        }
        if (pi0.action.expert_q_out % pi0.action.expert_kv_out != 0) {
            throw std::invalid_argument("pi0 action expert Q/KV widths are incompatible");
        }

        layers_ = pi0.action.expert_layers;
        width_ = pi0.action.expert_width;
        q_out_ = pi0.action.expert_q_out;
        kv_out_ = pi0.action.expert_kv_out;
        mlp_width_ = pi0.action.expert_mlp_width;
        batch_ = ctx_.config.common.action_horizon;
        action_dim_ = ctx_.config.common.action_dim;
        head_dim_ = pi0.action.expert_kv_out;
        heads_ = pi0.action.expert_q_out / head_dim_;
        kv_heads_ = pi0.action.expert_kv_out / head_dim_;
        has_state_context_ = !state_context_.empty();
        suffix_count_ = batch_ + (has_state_context_ ? 1 : 0);
        prefix_tokens_ = ctx_.prefix_kv.token_count;
        const bool has_prefix_kv = prefix_tokens_ > 0 &&
            ctx_.prefix_kv.k_layers.size() >= static_cast<size_t>(layers_) &&
            ctx_.prefix_kv.v_layers.size() >= static_cast<size_t>(layers_);
        if (!has_prefix_kv) {
            prefix_tokens_ = 0;
        }
        kv_tokens_ = suffix_count_ + static_cast<int>(prefix_tokens_);

        if (suffix_count_ <= 0 || batch_ <= 0 || action_dim_ <= 0 || layers_ <= 0 ||
            width_ <= 0 || q_out_ <= 0 || kv_out_ <= 0 || mlp_width_ <= 0 ||
            heads_ <= 0 || kv_heads_ <= 0 || head_dim_ <= 0 ||
            q_out_ != static_cast<int64_t>(heads_) * head_dim_ ||
            kv_out_ != static_cast<int64_t>(kv_heads_) * head_dim_ ||
            (has_state_context_ && state_context_.size() != static_cast<size_t>(width_))) {
            throw std::invalid_argument("pi0 action expert fused graph input has incompatible shape");
        }

        pos_host_.resize(static_cast<size_t>(suffix_count_));
        for (int i = 0; i < suffix_count_; ++i) {
            pos_host_[static_cast<size_t>(i)] = static_cast<int32_t>(prefix_tokens_ + static_cast<size_t>(i));
        }
        if (has_state_context_) {
            attention_mask_.assign(static_cast<size_t>(suffix_count_) * static_cast<size_t>(kv_tokens_), 0.0f);
            const size_t suffix_key_offset = prefix_tokens_;
            for (size_t key = suffix_key_offset + 1; key < static_cast<size_t>(kv_tokens_); ++key) {
                attention_mask_[key] = -INFINITY;
            }
            attention_mask_f16_.resize(attention_mask_.size());
            ggml_fp32_to_fp16_row(
                attention_mask_.data(),
                attention_mask_f16_.data(),
                static_cast<int64_t>(attention_mask_.size()));
        }

        time_embedding_.resize(static_cast<size_t>(width_));
        time_batch_.resize(static_cast<size_t>(batch_) * static_cast<size_t>(width_));
    }

    void build_graph() {
        ggml_tensor * in_w = ctx_.weights.action_in_w;
        ggml_tensor * in_b = ctx_.weights.action_in_b;
        ggml_tensor * time_in_w = ctx_.weights.action_time_in_w;
        ggml_tensor * time_in_b = ctx_.weights.action_time_in_b;
        ggml_tensor * time_out_w = ctx_.weights.action_time_out_w;
        ggml_tensor * time_out_b = ctx_.weights.action_time_out_b;
        ggml_tensor * out_w = ctx_.weights.action_out_w;
        ggml_tensor * out_b = ctx_.weights.action_out_b;
        if (in_w == nullptr || in_b == nullptr ||
            time_in_w == nullptr || time_in_b == nullptr || time_out_w == nullptr || time_out_b == nullptr ||
            out_w == nullptr || out_b == nullptr) {
            throw std::invalid_argument("missing pi0 action projection tensors");
        }

        action_input_ = ggml_new_tensor_2d(gctx_, GGML_TYPE_F32, action_dim_, batch_);
        time_input_ = ggml_new_tensor_2d(gctx_, GGML_TYPE_F32, width_, batch_);
        pos_ = ggml_new_tensor_1d(gctx_, GGML_TYPE_I32, suffix_count_);
        ggml_set_input(action_input_);
        ggml_set_input(time_input_);
        ggml_set_input(pos_);

        if (!attention_mask_.empty()) {
            kq_mask_ = ggml_new_tensor_3d(gctx_, GGML_TYPE_F16, kv_tokens_, suffix_count_, 1);
            ggml_set_input(kq_mask_);
        }

        ggml_tensor * action_tokens = ggml_add(
            gctx_,
            ggml_mul_mat(gctx_, pi0_weight(ctx_, in_w, 2), action_input_),
            pi0_f32_weight(gctx_, ctx_, in_b, 1));
        ggml_tensor * action_time = ggml_concat(gctx_, action_tokens, time_input_, 0);
        ggml_tensor * time_hidden = ggml_silu(gctx_, ggml_add(
            gctx_,
            ggml_mul_mat(gctx_, pi0_weight(ctx_, time_in_w, 2), action_time),
            pi0_f32_weight(gctx_, ctx_, time_in_b, 1)));
        ggml_tensor * hidden = ggml_add(
            gctx_,
            ggml_mul_mat(gctx_, pi0_weight(ctx_, time_out_w, 2), time_hidden),
            pi0_f32_weight(gctx_, ctx_, time_out_b, 1));
        if (has_state_context_) {
            state_token_ = ggml_new_tensor_2d(gctx_, GGML_TYPE_F32, width_, 1);
            ggml_set_input(state_token_);
            hidden = ggml_concat(gctx_, state_token_, hidden, 1);
        }

        for (int layer = 0; layer < layers_; ++layer) {
            const Pi0TransformerLayerWeights & layer_w = ctx_.weights.action_layers[static_cast<size_t>(layer)];
            ggml_tensor * q_w = layer_w.q;
            ggml_tensor * k_w = layer_w.k;
            ggml_tensor * v_w = layer_w.v;
            ggml_tensor * attn_out_w = layer_w.out;
            ggml_tensor * gate_w = layer_w.gate;
            ggml_tensor * up_w = layer_w.up;
            ggml_tensor * down_w = layer_w.down;
            if (q_w == nullptr || k_w == nullptr || v_w == nullptr || attn_out_w == nullptr ||
                gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
                throw std::invalid_argument("missing pi0 action expert fused graph tensor");
            }

            if (layer_w.input_norm_scale == nullptr || layer_w.post_norm_scale == nullptr) {
                throw std::invalid_argument("missing pi0 action expert norm scale tensor");
            }
            ggml_tensor * input_scale_tensor = pi0_f32_weight(gctx_, ctx_, layer_w.input_norm_scale, 1);
            ggml_tensor * post_scale_tensor = pi0_f32_weight(gctx_, ctx_, layer_w.post_norm_scale, 1);

            ggml_tensor * normed = ggml_mul(gctx_, ggml_rms_norm(gctx_, hidden, 1.0e-6f), input_scale_tensor);
            ggml_tensor * q = ggml_cont_2d(gctx_, ggml_mul_mat(gctx_, pi0_weight(ctx_, q_w, 2), normed), q_out_, suffix_count_);
            ggml_tensor * k = ggml_cont_2d(gctx_, ggml_mul_mat(gctx_, pi0_weight(ctx_, k_w, 2), normed), kv_out_, suffix_count_);
            ggml_tensor * v = ggml_cont_2d(gctx_, ggml_mul_mat(gctx_, pi0_weight(ctx_, v_w, 2), normed), kv_out_, suffix_count_);
            q = ggml_reshape_3d(gctx_, q, head_dim_, heads_, suffix_count_);
            k = ggml_reshape_3d(gctx_, k, head_dim_, kv_heads_, suffix_count_);
            v = ggml_reshape_3d(gctx_, v, head_dim_, kv_heads_, suffix_count_);
            ggml_tensor * q_rot = ggml_rope_ext(
                gctx_, q, pos_, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * k_rot = ggml_rope_ext(
                gctx_, k, pos_, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * k_all = k_rot;
            ggml_tensor * v_all = v;
            if (prefix_tokens_ > 0) {
                ggml_tensor * prefix_k = ctx_.prefix_kv.k_layers[static_cast<size_t>(layer)];
                ggml_tensor * prefix_v = ctx_.prefix_kv.v_layers[static_cast<size_t>(layer)];
                if (prefix_k == nullptr || prefix_v == nullptr ||
                    prefix_k->ne[0] != head_dim_ ||
                    prefix_k->ne[1] != kv_heads_ ||
                    prefix_k->ne[2] != static_cast<int64_t>(prefix_tokens_) ||
                    prefix_v->ne[0] != head_dim_ ||
                    prefix_v->ne[1] != kv_heads_ ||
                    prefix_v->ne[2] != static_cast<int64_t>(prefix_tokens_)) {
                    throw std::invalid_argument("pi0 action expert fused graph prefix KV has incompatible shape");
                }
                k_all = ggml_concat(gctx_, prefix_k, k_rot, 2);
                v_all = ggml_concat(gctx_, prefix_v, v, 2);
            }
            ggml_tensor * q_perm = ggml_permute(gctx_, q_rot, 0, 2, 1, 3);
            ggml_tensor * k_perm = ggml_permute(gctx_, k_all, 0, 2, 1, 3);
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            ggml_tensor * v_perm = ggml_permute(gctx_, v_all, 0, 2, 1, 3);
            ggml_tensor * attn_values = ggml_flash_attn_ext(gctx_, q_perm, k_perm, v_perm, kq_mask_, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_values, GGML_PREC_F32);
            attn_values = ggml_reshape_2d(gctx_, attn_values, static_cast<int64_t>(head_dim_) * heads_, suffix_count_);
            ggml_tensor * attn_out = ggml_mul_mat(gctx_, pi0_weight(ctx_, attn_out_w, 2), attn_values);
            ggml_tensor * first_residual = ggml_add(gctx_, hidden, attn_out);

            ggml_tensor * post_norm = ggml_mul(gctx_, ggml_rms_norm(gctx_, first_residual, 1.0e-6f), post_scale_tensor);
            ggml_tensor * gate = ggml_gelu(gctx_, ggml_mul_mat(gctx_, pi0_weight(ctx_, gate_w, 2), post_norm));
            ggml_tensor * up = ggml_mul_mat(gctx_, pi0_weight(ctx_, up_w, 2), post_norm);
            ggml_tensor * mlp_out = ggml_mul_mat(gctx_, pi0_weight(ctx_, down_w, 2), ggml_mul(gctx_, gate, up));
            hidden = ggml_add(gctx_, first_residual, mlp_out);
        }

        if (ctx_.weights.action_norm_scale == nullptr) {
            throw std::invalid_argument("missing pi0 action expert final norm scale tensor");
        }
        ggml_tensor * final_scale_tensor = pi0_f32_weight(gctx_, ctx_, ctx_.weights.action_norm_scale, 1);
        hidden = ggml_mul(gctx_, ggml_rms_norm(gctx_, hidden, 1.0e-6f), final_scale_tensor);

        const size_t action_offset = suffix_count_ == batch_ ? 0 : static_cast<size_t>(width_) * sizeof(float);
        ggml_tensor * output_tokens = ggml_view_2d(gctx_, hidden, width_, batch_, hidden->nb[1], action_offset);
        y_ = ggml_add(
            gctx_,
            ggml_mul_mat(gctx_, pi0_weight(ctx_, out_w, 2), output_tokens),
            pi0_f32_weight(gctx_, ctx_, out_b, 1));

        graph_ = ggml_new_graph_custom(gctx_, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(graph_, y_);
        ggml_backend_sched_reset(runtime_.sched);
        if (!ggml_backend_sched_alloc_graph(runtime_.sched, graph_)) {
            throw std::runtime_error("pi0 action decoder graph allocation failed");
        }

        ggml_backend_tensor_set(pos_, pos_host_.data(), 0, pos_host_.size() * sizeof(int32_t));
        if (kq_mask_ != nullptr) {
            ggml_backend_tensor_set(
                kq_mask_,
                attention_mask_f16_.data(),
                0,
                attention_mask_f16_.size() * sizeof(ggml_fp16_t));
        }
        if (state_token_ != nullptr) {
            ggml_backend_tensor_set(state_token_, state_context_.data(), 0, state_context_.size() * sizeof(float));
        }
    }

    void fill_time_batch(float time) {
        fill_posemb_sincos(time, time_embedding_.data(), time_embedding_.size());
        for (int row = 0; row < batch_; ++row) {
            std::copy(
                time_embedding_.begin(),
                time_embedding_.end(),
                time_batch_.begin() +
                    static_cast<std::ptrdiff_t>(static_cast<size_t>(row) * static_cast<size_t>(width_)));
        }
    }

    const Pi0Context & ctx_;
    const std::vector<float> & state_context_;
    const Pi0ComponentRuntime & runtime_;
    Pi0GraphContext gctx_;
    int layers_ = 0;
    int64_t width_ = 0;
    int64_t q_out_ = 0;
    int64_t kv_out_ = 0;
    int64_t mlp_width_ = 0;
    int batch_ = 0;
    int action_dim_ = 0;
    int head_dim_ = 0;
    int heads_ = 0;
    int kv_heads_ = 0;
    int suffix_count_ = 0;
    int kv_tokens_ = 0;
    size_t prefix_tokens_ = 0;
    bool has_state_context_ = false;
    std::vector<int32_t> pos_host_;
    std::vector<float> time_embedding_;
    std::vector<float> time_batch_;
    std::vector<float> attention_mask_;
    std::vector<ggml_fp16_t> attention_mask_f16_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * action_input_ = nullptr;
    ggml_tensor * time_input_ = nullptr;
    ggml_tensor * pos_ = nullptr;
    ggml_tensor * kq_mask_ = nullptr;
    ggml_tensor * state_token_ = nullptr;
    ggml_tensor * y_ = nullptr;
};

} // namespace

void denormalize_actions(const Pi0Context & ctx, std::vector<float> & actions) {
    if (ctx.config.common.action_mean.size() != static_cast<size_t>(ctx.config.common.action_dim) ||
        ctx.config.common.action_std.size() != static_cast<size_t>(ctx.config.common.action_dim)) {
        return;
    }
    for (size_t i = 0; i < actions.size(); ++i) {
        const size_t col = i % static_cast<size_t>(ctx.config.common.action_dim);
        actions[i] = actions[i] * ctx.config.common.action_std[col] + ctx.config.common.action_mean[col];
    }
}

void pi0_sample_actions(
    const Pi0Context & ctx,
    Pi0RuntimeConfig & runtime,
    const std::vector<float> & state_context,
    const std::vector<float> & initial_noise,
    std::vector<float> & out_actions) {
    Pi0ActionVelocityGraph velocity_graph(ctx, state_context);
    sample_flow_euler(
        runtime.flow_steps,
        runtime.rng,
        ctx.config.common.action_horizon,
        ctx.config.common.action_dim,
        initial_noise.empty() ? nullptr : &initial_noise,
        [&](float time, const std::vector<float> & x, std::vector<float> & v) {
            velocity_graph.eval(time, x, v);
        },
        out_actions);

    denormalize_actions(ctx, out_actions);
}

} // namespace robotcpp::pi0
