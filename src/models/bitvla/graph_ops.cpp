#include "graph_ops.h"
#include "component.h"
#include "ops.h"
#include "i2s.h"
#include <cmath>

namespace robotcpp::bitvla::graph_ops {

ggml_tensor * LayerWeights::weight(const char * name) const {
    return runtime.tensor(prefix + name + ".weight");
}
ggml_tensor * LayerWeights::bias(const char * name) const {
    return runtime.tensor(prefix + name + ".bias");
}

// Optional, CUDA-only LLM FlashAttention trial. Existing GGML kernels may use
// internal F16 operands; this variant is evaluated by measured error and SR.
static ggml_tensor * llm_flash_attention_encoded238(ggml_context * ctx, ggml_tensor * query, ggml_tensor * key,
                                                    ggml_tensor * value, int heads, int kv_heads, ggml_tensor * weight,
                                                    float eps) {
#ifndef BITVLA_USE_CUDA
    GGML_UNUSED(ctx);
    GGML_UNUSED(query);
    GGML_UNUSED(key);
    GGML_UNUSED(value);
    GGML_UNUSED(heads);
    GGML_UNUSED(kv_heads);
    GGML_UNUSED(weight);
    GGML_UNUSED(eps);
    return nullptr;
#else
    if (!query || !key || !value || !weight || heads != 20 || kv_heads != 5)
        return nullptr;
    const int64_t tokens = key->ne[1], rows = query->ne[1];
    if (tokens <= 0 || (rows != tokens && rows != 56) || query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F32 ||
        value->type != GGML_TYPE_F32 || query->ne[0] != 128 || query->ne[2] != 20 || query->ne[3] != 1 ||
        key->ne[0] != 128 || key->ne[2] != 5 || key->ne[3] != 1 || value->ne[0] != 640 || value->ne[1] != tokens ||
        value->ne[2] != 1 || value->ne[3] != 1 || weight->type != GGML_TYPE_F32 || weight->ne[0] != 2560 ||
        weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1 || !ggml_is_contiguous(query) ||
        !ggml_is_contiguous(key) || !ggml_is_contiguous(value) || !ggml_is_contiguous(weight))
        return nullptr;
    // Source V[d+h*128+t*640] becomes FA V[d,t,h] through standard views.
    auto * fa_value   = ggml_permute(ctx, ggml_reshape_3d(ctx, value, 128, 5, tokens), 0, 2, 1, 3);
    const float scale = 1.0f / std::sqrt(128.0f);
    auto * attention  = ggml_flash_attn_ext(ctx, query, key, fa_value, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
    ggml_set_name(attention, "bitvla.fa238.llm_attention");
    // FA is contiguous [128,20,Q]; the accepted consumer requires physical
    // [128,Q,20]. A standard lossless layout copy preserves its original
    // stride contract and all RMS/affine/I2S operations.
    auto * original_layout = ggml_permute(ctx, ggml_cont(ctx, ggml_permute(ctx, attention, 0, 2, 1, 3)), 0, 2, 1, 3);
    return permuted_rms_affine_i2s_encode(ctx, original_layout, weight, eps);
#endif
}

QKV language_qkv(GraphBuildContext & graph, ggml_tensor * input, const LayerWeights & weights, int layer, float eps,
                 const LanguageShape & shape, ggml_tensor * cosine, ggml_tensor * sine, bool & query_action_rows) {
    auto * ctx          = graph.runtime.ctx;
    const auto & policy = graph.runtime.policy();
    auto weight         = [&](const char * name) { return weights.weight(name); };
    auto * cur          = input;
    query_action_rows   = false;
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
    if (layer == 0 && shape.layers > 1 && shape.width == 2560 && shape.heads == 20 && shape.kv_heads == 5 &&
        graph.runtime.is_cpu() && std::isfinite(eps) && eps >= 0.0f && policy.llm_shared_projection_encoding &&
        weight("attn_q")->type == GGML_TYPE_I2_S && weight("attn_k")->type == GGML_TYPE_I2_S &&
        weight("attn_v")->type == GGML_TYPE_I2_S && weight("attn_q")->ne[0] == 2560 &&
        weight("attn_k")->ne[0] == 2560 && weight("attn_v")->ne[0] == 2560 && weight("attn_q")->ne[1] == 2560 &&
        weight("attn_k")->ne[1] == 640 && weight("attn_v")->ne[1] == 640 && weight("attn_q")->ne[2] == 1 &&
        weight("attn_q")->ne[3] == 1 && weight("attn_k")->ne[2] == 1 && weight("attn_k")->ne[3] == 1 &&
        weight("attn_v")->ne[2] == 1 && weight("attn_v")->ne[3] == 1 && ggml_is_contiguous(weight("attn_q")) &&
        ggml_is_contiguous(weight("attn_k")) && ggml_is_contiguous(weight("attn_v"))) {
        auto * encoded = rms_affine_i2s_encode(ctx, input, weight("attn_norm"), eps);
        if (encoded) {
            ggml_set_name(encoded, "bitvla.first.rms_i2s155");
            auto * q       = i2s_linear_preencoded(ctx, weight("attn_q"), encoded, true);
            auto * pair    = i2s_linear_preencoded_packed_pair(ctx, weight("attn_k"), weight("attn_v"), encoded);
            ggml_tensor *k = nullptr, *v = nullptr;
            if (pair) {
                k = ggml_view_2d(ctx, pair, pair->ne[0], pair->ne[1], pair->nb[1], 0);
                v = ggml_view_2d(ctx, pair, pair->ne[0], pair->ne[1], pair->nb[1], pair->nb[2]);
            } else {
                k = i2s_linear_preencoded(ctx, weight("attn_k"), encoded, true);
                v = i2s_linear_preencoded(ctx, weight("attn_v"), encoded, true);
            }
            return {q, k, v};
        }
    }
#endif
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *qkv_encoded = nullptr;
    if (layer == 0 && policy.llm_fused_rms_i2s && weight("attn_q")->type == GGML_TYPE_I2_S &&
        weight("attn_k")->type == GGML_TYPE_I2_S && weight("attn_v")->type == GGML_TYPE_I2_S &&
        weight("attn_q")->ne[0] == 2560 && weight("attn_k")->ne[0] == 2560 && weight("attn_v")->ne[0] == 2560)
        qkv_encoded = cpu_order_rms_i2s_encode(ctx, cur, weight("attn_norm"), eps);
    if (layer > 0 && policy.llm_fused_rms_i2s && weight("attn_q")->type == GGML_TYPE_I2_S &&
        weight("attn_k")->type == GGML_TYPE_I2_S && weight("attn_v")->type == GGML_TYPE_I2_S &&
        weight("attn_q")->ne[0] == 2560 && weight("attn_k")->ne[0] == 2560 && weight("attn_v")->ne[0] == 2560)
        qkv_encoded = rms_affine_i2s_encode(ctx, cur, weight("attn_norm"), eps);
    if (qkv_encoded) {
        auto * composite =
            i2s_linear_preencoded_qkv(ctx, weight("attn_q"), weight("attn_k"), weight("attn_v"), qkv_encoded);
        if (composite) {
            const int64_t n = qkv_encoded->ne[1];
            q               = ggml_view_2d(ctx, composite, 2560, n, 2560 * sizeof(float), 0);
            k = ggml_view_2d(ctx, composite, 640, n, 640 * sizeof(float), int64_t(2560) * n * sizeof(float));
            v = ggml_view_2d(ctx, composite, 640, n, 640 * sizeof(float), int64_t(3200) * n * sizeof(float));
        } else {
            q                = i2s_linear_preencoded(ctx, weight("attn_q"), qkv_encoded, true);
            auto * packed_kv = i2s_linear_preencoded_packed_pair(ctx, weight("attn_k"), weight("attn_v"), qkv_encoded);
            if (packed_kv) {
                k = ggml_view_2d(ctx, packed_kv, packed_kv->ne[0], packed_kv->ne[1], packed_kv->nb[1], 0);
                v = ggml_view_2d(ctx, packed_kv, packed_kv->ne[0], packed_kv->ne[1], packed_kv->nb[1],
                                 packed_kv->nb[2]);
            } else {
                k = i2s_linear_preencoded(ctx, weight("attn_k"), qkv_encoded, true);
                v = i2s_linear_preencoded(ctx, weight("attn_v"), qkv_encoded, true);
            }
        }
    } else {
        if (layer == 0 && policy.llm_first_rms_cpu_order) {
            cur = ggml_mul(ctx, rms_norm_cpu_order(ctx, cur, eps), weight("attn_norm"));
        } else {
            cur = rms_norm(ctx, cur, weight("attn_norm"), eps);
        }
        cur = linear_input(ctx, weight("attn_q"), cur);
        if (policy.llm_shared_projection_encoding) {
            auto * encoded = shared_linear_input(ctx, cur, {weight("attn_q"), weight("attn_k"), weight("attn_v")});
            // CPU only: K/V still need every row of the same shared encoding.
            // Match the existing last-layer crop contract before reducing Q.
            if (graph.runtime.is_cpu() && encoded && policy.llm_last_layer_action_rows && layer == shape.layers - 1 &&
                shape.layers > 1 && weight("attn_output")->type == GGML_TYPE_I2_S &&
                weight("ffn_gate")->type == GGML_TYPE_I2_S && weight("ffn_up")->type == GGML_TYPE_I2_S &&
                weight("ffn_down")->type == GGML_TYPE_I2_S && weight("attn_q")->ne[0] == shape.width &&
                weight("attn_q")->ne[1] == shape.width && input->type == GGML_TYPE_F32 && input->ne[0] == shape.width &&
                input->ne[1] == shape.tokens && input->ne[2] == 1 && input->ne[3] == 1 && ggml_is_contiguous(input) &&
                ggml_is_contiguous(encoded) && encoded->ne[1] == shape.tokens && encoded->ne[2] == 1 &&
                encoded->ne[3] == 1 && ggml_is_contiguous(cosine) && ggml_is_contiguous(sine) &&
                cosine->ne[2] == shape.tokens && sine->ne[2] == shape.tokens && shape.action_start > 0 &&
                int64_t(shape.action_start - 1) + 56 <= shape.tokens) {
                auto * query_encoded = ggml_view_2d(ctx, encoded, encoded->ne[0], 56, encoded->nb[1],
                                                    int64_t(shape.action_start - 1) * encoded->nb[1]);
                ggml_set_name(query_encoded, "bitvla.last.encoded_query56");
                // This entry accepts an encoded view without weakening the
                // separate exact-F32-input contract of linear_shared().
                q = i2s_linear_preencoded(ctx, weight("attn_q"), query_encoded, true);
                ggml_set_name(q, "bitvla.last.query56");
                query_action_rows = true;
            } else {
                q = linear_shared(ctx, weight("attn_q"), cur, encoded);
            }
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
            auto * packed_kv = graph.runtime.is_cpu() && encoded
                                   ? i2s_linear_preencoded_packed_pair(ctx, weight("attn_k"), weight("attn_v"), encoded)
                                   : nullptr;
            if (packed_kv) {
                k = ggml_view_2d(ctx, packed_kv, packed_kv->ne[0], packed_kv->ne[1], packed_kv->nb[1], 0);
                v = ggml_view_2d(ctx, packed_kv, packed_kv->ne[0], packed_kv->ne[1], packed_kv->nb[1],
                                 packed_kv->nb[2]);
            } else {
#endif
                k = linear_shared(ctx, weight("attn_k"), cur, encoded);
                v = linear_shared(ctx, weight("attn_v"), cur, encoded);
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
            }
#endif
        } else {
            q = linear(ctx, weight("attn_q"), cur);
            k = linear(ctx, weight("attn_k"), cur);
            v = linear(ctx, weight("attn_v"), cur);
        }
    }
    return {q, k, v};
}

AttentionResult language_attention(GraphBuildContext & graph, QKV projections, ggml_tensor * residual,
                                   const LayerWeights & weights, const LanguageShape & shape, int layer,
                                   ggml_tensor * cosine, ggml_tensor * sine, bool action_rows_only,
                                   bool query_action_rows) {
    auto * ctx          = graph.runtime.ctx;
    const auto & policy = graph.runtime.policy();
    auto weight         = [&](const char * name) { return weights.weight(name); };
    const int width = shape.width, heads = shape.heads, kv_heads = shape.kv_heads;
    const int layers = shape.layers, tokens = shape.tokens, action_start = shape.action_start;
    const float eps   = shape.eps;
    auto * q          = projections.q;
    auto * k          = projections.k;
    auto * v          = projections.v;
    ggml_tensor * cur = nullptr;
    // Last-layer queries outside the original56 action rows cannot affect
    // those rows: keys/values and their positions remain the full context.
    if (query_action_rows) {
        GGML_ASSERT(graph.runtime.is_cpu() && policy.llm_last_layer_action_rows && layer == layers - 1 && layers > 1 &&
                    q->type == GGML_TYPE_F32 && q->ne[0] == width && q->ne[1] == 56 && q->ne[2] == 1 && q->ne[3] == 1 &&
                    k->ne[1] == tokens && v->ne[1] == tokens && residual->ne[1] == tokens && action_start > 0 &&
                    int64_t(action_start - 1) + 56 <= tokens);
        auto * query_cosine = ggml_view_3d(ctx, cosine, width / heads, 1, 56, cosine->nb[1], cosine->nb[2],
                                           int64_t(action_start - 1) * cosine->nb[2]);
        auto * query_sine   = ggml_view_3d(ctx, sine, width / heads, 1, 56, sine->nb[1], sine->nb[2],
                                           int64_t(action_start - 1) * sine->nb[2]);
        ggml_set_name(query_cosine, "bitvla.last.cos56");
        ggml_set_name(query_sine, "bitvla.last.sin56");
        q = rope_attention_layout(ctx, ggml_reshape_3d(ctx, q, width / heads, heads, 56), query_cosine, query_sine);
        residual = ggml_view_2d(ctx, residual, width, 56, residual->nb[1], int64_t(action_start - 1) * residual->nb[1]);
        action_rows_only = true;
    } else if (layer == layers - 1 && policy.llm_last_layer_action_rows && layers > 1 &&
               weight("attn_output")->type == GGML_TYPE_I2_S && weight("ffn_gate")->type == GGML_TYPE_I2_S &&
               weight("ffn_up")->type == GGML_TYPE_I2_S && weight("ffn_down")->type == GGML_TYPE_I2_S &&
               q->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 && q->ne[0] == width && q->ne[1] == tokens &&
               q->ne[2] == 1 && q->ne[3] == 1 && residual->ne[0] == width && residual->ne[1] == tokens &&
               residual->ne[2] == 1 && residual->ne[3] == 1 && k->ne[1] == tokens && v->ne[1] == tokens &&
               ggml_is_contiguous(q) && ggml_is_contiguous(residual) && ggml_is_contiguous(cosine) &&
               ggml_is_contiguous(sine) && cosine->ne[2] == tokens && sine->ne[2] == tokens && action_start > 0 &&
               int64_t(action_start - 1) + 56 <= tokens) {
        auto * query_rows   = ggml_view_2d(ctx, q, width, 56, q->nb[1], (action_start - 1) * q->nb[1]);
        auto * query_cosine = ggml_view_3d(ctx, cosine, width / heads, 1, 56, cosine->nb[1], cosine->nb[2],
                                           (action_start - 1) * cosine->nb[2]);
        auto * query_sine =
            ggml_view_3d(ctx, sine, width / heads, 1, 56, sine->nb[1], sine->nb[2], (action_start - 1) * sine->nb[2]);
        ggml_set_name(query_rows, "bitvla.last.query56");
        ggml_set_name(query_cosine, "bitvla.last.cos56");
        ggml_set_name(query_sine, "bitvla.last.sin56");
        q        = rope_attention_layout(ctx, ggml_reshape_3d(ctx, query_rows, width / heads, heads, 56), query_cosine,
                                         query_sine);
        residual = ggml_view_2d(ctx, residual, width, 56, residual->nb[1], (action_start - 1) * residual->nb[1]);
        action_rows_only = true;
    } else {
        q = rope_attention_layout(ctx, ggml_reshape_3d(ctx, q, width / heads, heads, tokens), cosine, sine);
    }
    k = rope_attention_layout(ctx, ggml_reshape_3d(ctx, k, width / heads, kv_heads, tokens), cosine, sine);
    ggml_tensor * attention_encoded = nullptr;
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
    if (graph.runtime.is_cpu() && std::isfinite(eps) && eps >= 0.0f && width == 2560 && heads == 20 && kv_heads == 5 &&
        tokens > 0 && (layer < layers - 1 || action_rows_only) && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 && q->ne[0] == 128 && q->ne[2] == 20 && q->ne[3] == 1 &&
        q->ne[1] == (action_rows_only ? 56 : tokens) && k->ne[0] == 128 && k->ne[1] == tokens && k->ne[2] == 5 &&
        k->ne[3] == 1 && v->ne[0] == 640 && v->ne[1] == tokens && v->ne[2] == 1 && v->ne[3] == 1 &&
        ggml_is_contiguous(q) && ggml_is_contiguous(k) && ggml_is_contiguous(v) &&
        weight("attn_output")->type == GGML_TYPE_I2_S && weight("attn_output")->ne[0] == 2560 &&
        weight("attn_output")->ne[1] == 2560 && weight("attn_output")->ne[2] == 1 &&
        weight("attn_output")->ne[3] == 1 && ggml_is_contiguous(weight("attn_output"))) {
        // Same original KQ/softmax/PV graph; compose only its output layout/RMS/encoding.
        attention_encoded = attention_rms_i2s_encode(ctx, q, k, v, heads, kv_heads, true, weight("attn_sub_norm"), eps);
        if (attention_encoded)
            ggml_set_name(attention_encoded, "bitvla.attention.layout_rms_i2s154");
    }
#endif
    if (!attention_encoded && policy.llm_fused_attention_rms_i2s && width == 2560 &&
        (layer < layers - 1 || action_rows_only) && weight("attn_output")->type == GGML_TYPE_I2_S &&
        weight("attn_output")->ne[0] == 2560) {
        if (policy.llm_flash_attention)
            attention_encoded =
                llm_flash_attention_encoded238(ctx, q, k, v, heads, kv_heads, weight("attn_sub_norm"), eps);
        if (!attention_encoded)
            attention_encoded =
                attention_rms_i2s_encode(ctx, q, k, v, heads, kv_heads, true, weight("attn_sub_norm"), eps);
    }
    if (attention_encoded)
        cur = i2s_linear_preencoded(ctx, weight("attn_output"), attention_encoded, true);
    else {
        cur = attention(ctx, q, k, v, heads, kv_heads, true);
        // If the query guard fell back, retain the accepted127 tail-only view.
        // All key/value context rows remain available to attention.
        if (!action_rows_only && policy.llm_last_layer_action_rows && layers > 1 && layer == layers - 1 &&
            weight("attn_output")->type == GGML_TYPE_I2_S && weight("ffn_gate")->type == GGML_TYPE_I2_S &&
            weight("ffn_up")->type == GGML_TYPE_I2_S && weight("ffn_down")->type == GGML_TYPE_I2_S &&
            cur->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) &&
            ggml_is_contiguous(residual) && cur->ne[0] == width && residual->ne[0] == width && cur->ne[1] == tokens &&
            residual->ne[1] == tokens && cur->ne[2] == 1 && cur->ne[3] == 1 && residual->ne[2] == 1 &&
            residual->ne[3] == 1 && action_start > 0 && int64_t(action_start - 1) + 56 <= tokens) {
            cur      = ggml_view_2d(ctx, cur, width, 56, cur->nb[1], (action_start - 1) * cur->nb[1]);
            residual = ggml_view_2d(ctx, residual, width, 56, residual->nb[1], (action_start - 1) * residual->nb[1]);
            action_rows_only = true;
        }
        ggml_tensor * post_attention_encoded = nullptr;
        if (policy.llm_fused_rms_i2s && weight("attn_output")->type == GGML_TYPE_I2_S &&
            weight("attn_output")->ne[0] == 2560)
            post_attention_encoded = rms_affine_i2s_encode(ctx, cur, weight("attn_sub_norm"), eps);
        if (post_attention_encoded)
            cur = i2s_linear_preencoded(ctx, weight("attn_output"), post_attention_encoded, true);
        else {
            cur = rms_norm(ctx, cur, weight("attn_sub_norm"), eps);
            cur = linear(ctx, weight("attn_output"), linear_input(ctx, weight("attn_output"), cur));
        }
    }
    return {cur, residual, action_rows_only};
}

ggml_tensor * language_ffn(GraphBuildContext & graph, ggml_tensor * input, const LayerWeights & weights, int layer,
                           float eps) {
    auto * ctx          = graph.runtime.ctx;
    const auto & policy = graph.runtime.policy();
    auto weight         = [&](const char * name) { return weights.weight(name); };
    auto * cur          = input;
    ggml_tensor *gate = nullptr, *up = nullptr, *ffn_encoded = nullptr;
    bool cpu_ffn_rms = false;
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
    if (graph.runtime.is_cpu() && std::isfinite(eps) && eps >= 0.0f && cur->type == GGML_TYPE_F32 &&
        cur->ne[0] == 2560 && cur->ne[1] > 0 && cur->ne[2] == 1 && cur->ne[3] == 1 && ggml_is_contiguous(cur) &&
        weight("ffn_gate")->type == GGML_TYPE_I2_S && weight("ffn_up")->type == GGML_TYPE_I2_S &&
        weight("ffn_gate")->ne[0] == 2560 && weight("ffn_gate")->ne[1] == 6912 && weight("ffn_gate")->ne[2] == 1 &&
        weight("ffn_gate")->ne[3] == 1 && weight("ffn_up")->ne[0] == 2560 && weight("ffn_up")->ne[1] == 6912 &&
        weight("ffn_up")->ne[2] == 1 && weight("ffn_up")->ne[3] == 1 && ggml_is_contiguous(weight("ffn_gate")) &&
        ggml_is_contiguous(weight("ffn_up"))) {
        // CUDA144 only: compose the accepted RMS and encoder, without FFN pairing.
        ffn_encoded = rms_affine_i2s_encode(ctx, cur, weight("ffn_norm"), eps);
        cpu_ffn_rms = ffn_encoded != nullptr;
        if (ffn_encoded)
            ggml_set_name(ffn_encoded, "bitvla.pre_ffn.rms_i2s144");
    }
#endif
    if (!ffn_encoded && policy.llm_fused_rms_i2s && weight("ffn_gate")->type == GGML_TYPE_I2_S &&
        weight("ffn_up")->type == GGML_TYPE_I2_S && weight("ffn_gate")->ne[0] == 2560 &&
        weight("ffn_up")->ne[0] == 2560)
        ffn_encoded = rms_affine_i2s_encode(ctx, cur, weight("ffn_norm"), eps);
    if (ffn_encoded) {
        // Keep the original independent CPU projections; rejected166 is not enabled.
        auto * paired =
            cpu_ffn_rms ? nullptr : i2s_linear_preencoded_pair(ctx, weight("ffn_gate"), weight("ffn_up"), ffn_encoded);
        if (paired) {
            gate = ggml_view_2d(ctx, paired, paired->ne[0], paired->ne[1], paired->nb[1], 0);
            up   = ggml_view_2d(ctx, paired, paired->ne[0], paired->ne[1], paired->nb[1], paired->nb[2]);
        } else {
            gate = i2s_linear_preencoded(ctx, weight("ffn_gate"), ffn_encoded, true);
            up   = i2s_linear_preencoded(ctx, weight("ffn_up"), ffn_encoded, true);
        }
    } else {
        cur = rms_norm(ctx, cur, weight("ffn_norm"), eps);
        cur = linear_input(ctx, weight("ffn_gate"), cur);
        if (policy.llm_shared_projection_encoding) {
            auto * encoded = shared_linear_input(ctx, cur, {weight("ffn_gate"), weight("ffn_up")});
            gate           = linear_shared(ctx, weight("ffn_gate"), cur, encoded);
            up             = linear_shared(ctx, weight("ffn_up"), cur, encoded);
        } else {
            gate = linear(ctx, weight("ffn_gate"), cur);
            up   = linear(ctx, weight("ffn_up"), cur);
        }
    }
    ggml_tensor * down_encoded = nullptr;
    if (policy.llm_fused_gated_rms_i2s && weight("ffn_down")->type == GGML_TYPE_I2_S &&
        weight("ffn_down")->ne[0] == 6912)
        down_encoded = gated_rms_affine_i2s_encode(ctx, gate, up, weight("ffn_sub_norm"), eps);
    if (down_encoded)
        cur = i2s_linear_preencoded(ctx, weight("ffn_down"), down_encoded, true);
    else {
        if (policy.llm_fused_gated_rms)
            cur = gated_rms_affine_f32(ctx, gate, up, weight("ffn_sub_norm"), eps);
        else {
            cur = gated_relu_square_f32(ctx, gate, up);
            cur = rms_norm(ctx, cur, weight("ffn_sub_norm"), eps);
        }
        cur = linear(ctx, weight("ffn_down"), linear_input(ctx, weight("ffn_down"), cur));
    }
    return cur;
}

ggml_tensor * vision_patch_descriptor(GraphBuildContext & graph, ggml_tensor * kernel) {
    // This is the existing CPU im2col capability constraint, not a policy flag.
    return graph.runtime.is_cpu() && kernel->type == GGML_TYPE_BF16
               ? ggml_cast(graph.runtime.ctx, kernel, GGML_TYPE_F16)
               : kernel;
}

QKV vision_qkv(GraphBuildContext & graph, ggml_tensor * input, const LayerWeights & weights, int layer, int width,
               int heads, float eps) {
    auto * ctx          = graph.runtime.ctx;
    const auto & policy = graph.runtime.policy();
    auto weight         = [&](const char * name) { return weights.weight(name); };
    auto bias           = [&](const char * name) { return weights.bias(name); };
    auto * cur          = input;
    bool packed_qkv     = false;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *ln1_encoded = nullptr;
    if (policy.vision_fused_layer_norm_i2s && weight("attn_q")->type == GGML_TYPE_I2_S &&
        weight("attn_k")->type == GGML_TYPE_I2_S && weight("attn_v")->type == GGML_TYPE_I2_S &&
        weight("attn_q")->ne[0] == 1152 && weight("attn_k")->ne[0] == 1152 && weight("attn_v")->ne[0] == 1152)
        ln1_encoded = layer_norm_affine_i2s_encode(ctx, cur, weight("ln1"), bias("ln1"), eps);
    if (ln1_encoded) {
        if (weight("attn_out")->type == GGML_TYPE_I2_S && weight("attn_out")->ne[0] == 1152 && width == 1152) {
            auto * combined = i2s_linear_preencoded_attention_qkv(ctx, ln1_encoded, weight("attn_q"), weight("attn_k"),
                                                                  weight("attn_v"), bias("attn_q"), bias("attn_k"),
                                                                  bias("attn_v"), heads);
            if (combined) {
                const int64_t n = ln1_encoded->ne[1], m = weight("attn_q")->ne[1], dim = m / heads,
                              plane = m * n * sizeof(float);
                q = ggml_view_3d(ctx, combined, dim, n, heads, dim * sizeof(float), dim * n * sizeof(float), 0);
                k = ggml_view_3d(ctx, combined, dim, n, heads, dim * sizeof(float), dim * n * sizeof(float), plane);
                v = ggml_view_3d(ctx, combined, n, dim, heads, n * sizeof(float), n * dim * sizeof(float), 2 * plane);
                packed_qkv = true;
            } else {
                q = i2s_linear_preencoded_attention_layout(ctx, weight("attn_q"), ln1_encoded, bias("attn_q"), heads,
                                                           false);
                k = i2s_linear_preencoded_attention_layout(ctx, weight("attn_k"), ln1_encoded, bias("attn_k"), heads,
                                                           false);
                v = i2s_linear_preencoded_attention_layout(ctx, weight("attn_v"), ln1_encoded, bias("attn_v"), heads,
                                                           true);
                packed_qkv = q && k && v;
            }
        }
        if (!packed_qkv) {
            q = ggml_add(ctx, i2s_linear_preencoded(ctx, weight("attn_q"), ln1_encoded), bias("attn_q"));
            k = ggml_add(ctx, i2s_linear_preencoded(ctx, weight("attn_k"), ln1_encoded), bias("attn_k"));
            v = ggml_add(ctx, i2s_linear_preencoded(ctx, weight("attn_v"), ln1_encoded), bias("attn_v"));
        }
    } else {
        cur = policy.vision_fused_layer_norm_affine ? layer_norm_affine_f32(ctx, cur, weight("ln1"), bias("ln1"), eps)
                                                    : layer_norm(ctx, cur, weight("ln1"), bias("ln1"), eps);
        cur = linear_input(ctx, weight("attn_q"), cur);
        auto * qkv_input = shared_linear_input(ctx, cur, {weight("attn_q"), weight("attn_k"), weight("attn_v")});
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
        if (graph.runtime.is_cpu() && qkv_input) {
            if (width == 1152 && heads == 16 && qkv_input->ne[1] == 256 && weight("attn_out")->type == GGML_TYPE_I2_S &&
                weight("attn_out")->ne[0] == 1152) {
                auto * combined = i2s_linear_preencoded_attention_qkv(
                    ctx, qkv_input, weight("attn_q"), weight("attn_k"), weight("attn_v"), bias("attn_q"),
                    bias("attn_k"), bias("attn_v"), heads);
                if (combined) {
                    constexpr int64_t dim = 72, n = 256, plane = 1152 * n * sizeof(float);
                    q = ggml_view_3d(ctx, combined, dim, n, heads, dim * sizeof(float), dim * n * sizeof(float), 0);
                    k = ggml_view_3d(ctx, combined, dim, n, heads, dim * sizeof(float), dim * n * sizeof(float), plane);
                    v = ggml_view_3d(ctx, combined, n, dim, heads, n * sizeof(float), n * dim * sizeof(float),
                                     2 * plane);
                    packed_qkv = true;
                }
            }
            if (!packed_qkv) {
                q = i2s_linear_preencoded_attention_layout(ctx, weight("attn_q"), qkv_input, bias("attn_q"), heads,
                                                           false);
                k = i2s_linear_preencoded_attention_layout(ctx, weight("attn_k"), qkv_input, bias("attn_k"), heads,
                                                           false);
                v = i2s_linear_preencoded_attention_layout(ctx, weight("attn_v"), qkv_input, bias("attn_v"), heads,
                                                           true);
                packed_qkv = q && k && v;
            }
        }
        if (!packed_qkv) {
#endif
            q = linear_shared(ctx, weight("attn_q"), cur, qkv_input, bias("attn_q"));
            k = linear_shared(ctx, weight("attn_k"), cur, qkv_input, bias("attn_k"));
            v = linear_shared(ctx, weight("attn_v"), cur, qkv_input, bias("attn_v"));
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
        }
#endif
    }
    return {q, k, v, packed_qkv};
}

ggml_tensor * vision_attention(GraphBuildContext & graph, QKV projections, const LayerWeights & weights, int layer,
                               int width, int heads) {
    auto * ctx            = graph.runtime.ctx;
    const auto & policy   = graph.runtime.policy();
    auto weight           = [&](const char * name) { return weights.weight(name); };
    auto bias             = [&](const char * name) { return weights.bias(name); };
    auto * q              = projections.q;
    auto * k              = projections.k;
    auto * v              = projections.v;
    const bool packed_qkv = projections.packed;
    ggml_tensor * cur     = nullptr;
    if (policy.vision_fused_attention_i2s && weight("attn_out")->type == GGML_TYPE_I2_S &&
        weight("attn_out")->ne[0] == 1152 && width == 1152) {
        auto * encoded = packed_qkv ? attention_packed_i2s_encode(ctx, q, k, v, heads, true)
                                    : attention_i2s_encode(ctx, q, k, v, heads, heads, true);
        cur            = ggml_add(ctx, i2s_linear_preencoded(ctx, weight("attn_out"), encoded), bias("attn_out"));
    } else {
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) &&          \
    !defined(GGML_USE_METAL)
        cur = packed_qkv && graph.runtime.is_cpu() ? attention_packed_f32(ctx, q, k, v, heads, true)
                                                   : attention(ctx, q, k, v, heads, heads, true);
#else
        cur = attention(ctx, q, k, v, heads, heads, true);
#endif
        cur = linear_input(ctx, weight("attn_out"), cur);
        cur = linear(ctx, weight("attn_out"), cur, bias("attn_out"));
    }
    return cur;
}

ggml_tensor * vision_ffn(GraphBuildContext & graph, ggml_tensor * input, const LayerWeights & weights, int layer,
                         float eps) {
    auto * ctx                = graph.runtime.ctx;
    const auto & policy       = graph.runtime.policy();
    auto weight               = [&](const char * name) { return weights.weight(name); };
    auto bias                 = [&](const char * name) { return weights.bias(name); };
    auto * cur                = input;
    ggml_tensor * ln2_encoded = nullptr;
    if (policy.vision_fused_layer_norm_i2s && weight("ffn_down")->type == GGML_TYPE_I2_S &&
        weight("ffn_down")->ne[0] == 1152)
        ln2_encoded = layer_norm_affine_i2s_encode(ctx, cur, weight("ln2"), bias("ln2"), eps);
    if (ln2_encoded)
        cur = ggml_add(ctx, i2s_linear_preencoded(ctx, weight("ffn_down"), ln2_encoded), bias("ffn_down"));
    else {
        cur = policy.vision_fused_layer_norm_affine ? layer_norm_affine_f32(ctx, cur, weight("ln2"), bias("ln2"), eps)
                                                    : layer_norm(ctx, cur, weight("ln2"), bias("ln2"), eps);
        cur = linear(ctx, weight("ffn_down"), linear_input(ctx, weight("ffn_down"), cur), bias("ffn_down"));
    }
    if (policy.vision_fused_gelu_i2s && weight("ffn_up")->type == GGML_TYPE_I2_S && weight("ffn_up")->ne[0] == 4304 &&
        cur->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) && cur->ne[0] == 4304 && cur->ne[1] > 0 &&
        cur->ne[1] <= 2147483647 / 4360 && cur->ne[2] == 1 && cur->ne[3] == 1) {
        auto * encoded = bias_gelu_f32_i2s_encode(ctx, cur);
        if (!encoded)
            encoded = gelu_f32_i2s_encode(ctx, cur);
        if (!encoded) {
            // Keep the original GELU node and its input; fuse only F32 input handling and I2S encoding.
            auto * activation = ggml_gelu(ctx, cur);
            encoded           = f32_i2s_encode(ctx, activation);
            GGML_ASSERT(encoded);
        }
        cur = ggml_add(ctx, i2s_linear_preencoded(ctx, weight("ffn_up"), encoded), bias("ffn_up"));
    } else {
        cur = ggml_gelu(ctx, cur);
        cur = linear(ctx, weight("ffn_up"), linear_input(ctx, weight("ffn_up"), cur), bias("ffn_up"));
    }
    return cur;
}

} // namespace robotcpp::bitvla::graph_ops
