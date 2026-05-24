#include "models/pi0/pi0_action_expert.h"
#include "models/pi0/pi0_action_decoder.h"
#include "models/pi0/pi0_language_prefix.h"
#include "models/pi0/pi0_vlm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

float gelu(float x) {
    constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
    constexpr float coef = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * x * (1.0f + coef * x * x)));
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

std::vector<float> linear(
    const std::vector<float> & weight,
    const std::vector<float> & input,
    int batch,
    int in,
    int out) {
    std::vector<float> result(static_cast<size_t>(batch) * static_cast<size_t>(out), 0.0f);
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out; ++col) {
            float value = 0.0f;
            for (int i = 0; i < in; ++i) {
                value += weight[static_cast<size_t>(col) * static_cast<size_t>(in) + static_cast<size_t>(i)] *
                    input[static_cast<size_t>(row) * static_cast<size_t>(in) + static_cast<size_t>(i)];
            }
            result[static_cast<size_t>(row) * static_cast<size_t>(out) + static_cast<size_t>(col)] = value;
        }
    }
    return result;
}

std::vector<float> linear_add(
    const std::vector<float> & weight,
    const std::vector<float> & bias,
    const std::vector<float> & input,
    int batch,
    int in,
    int out) {
    std::vector<float> result = linear(weight, input, batch, in, out);
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out; ++col) {
            result[static_cast<size_t>(row) * static_cast<size_t>(out) + static_cast<size_t>(col)] +=
                bias[static_cast<size_t>(col)];
        }
    }
    return result;
}

std::vector<float> posemb_sincos(float time, size_t width) {
    std::vector<float> result(width, 0.0f);
    const size_t half = width / 2;
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

std::vector<float> rms_norm(
    const std::vector<float> & weight,
    const std::vector<float> & input,
    int batch,
    int width) {
    std::vector<float> result(input.size(), 0.0f);
    for (int row = 0; row < batch; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < width; ++col) {
            const float value = input[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)];
            sum += value * value;
        }
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(width) + 1.0e-6f);
        for (int col = 0; col < width; ++col) {
            const size_t index = static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col);
            result[index] = input[index] * inv * (1.0f + weight[static_cast<size_t>(col)]);
        }
    }
    return result;
}

std::vector<float> attention_ref(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    const std::vector<float> & mask,
    int q_tokens,
    int kv_tokens,
    int heads,
    int kv_heads,
    int head_dim) {
    const int repeat = heads / kv_heads;
    std::vector<float> result(q.size(), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < heads; ++head) {
        const int kv_head = head / repeat;
        for (int tq = 0; tq < q_tokens; ++tq) {
            std::vector<float> scores(static_cast<size_t>(kv_tokens), 0.0f);
            float max_score = -INFINITY;
            for (int tk = 0; tk < kv_tokens; ++tk) {
                float score = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim) {
                    const size_t q_index =
                        (static_cast<size_t>(tq) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    const size_t k_index =
                        (static_cast<size_t>(tk) * static_cast<size_t>(kv_heads) + static_cast<size_t>(kv_head)) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    score += q[q_index] * k[k_index];
                }
                scores[static_cast<size_t>(tk)] =
                    score * scale + (mask.empty() ? 0.0f : mask[static_cast<size_t>(tq) * kv_tokens + tk]);
                max_score = std::max(max_score, scores[static_cast<size_t>(tk)]);
            }
            float denom = 0.0f;
            for (float & score : scores) {
                score = std::exp(score - max_score);
                denom += score;
            }
            for (int dim = 0; dim < head_dim; ++dim) {
                float value = 0.0f;
                for (int tk = 0; tk < kv_tokens; ++tk) {
                    const size_t v_index =
                        (static_cast<size_t>(tk) * static_cast<size_t>(kv_heads) + static_cast<size_t>(kv_head)) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(dim);
                    value += scores[static_cast<size_t>(tk)] / denom * v[v_index];
                }
                const size_t out_index =
                    (static_cast<size_t>(tq) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                        static_cast<size_t>(head_dim) +
                    static_cast<size_t>(dim);
                result[out_index] = value;
            }
        }
    }
    return result;
}

std::vector<float> self_attention(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    const std::vector<float> & mask,
    int tokens,
    int heads,
    int kv_heads,
    int head_dim) {
    return attention_ref(q, k, v, mask, tokens, tokens, heads, kv_heads, head_dim);
}

std::vector<float> self_attention(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int tokens,
    int heads,
    int kv_heads,
    int head_dim) {
    return self_attention(q, k, v, {}, tokens, heads, kv_heads, head_dim);
}

std::vector<float> rope_neox(
    const std::vector<float> & values,
    const std::vector<int> & positions,
    int tokens,
    int heads,
    int head_dim) {
    std::vector<float> result(values.size(), 0.0f);
    const int half = head_dim / 2;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = 0; dim < half; ++dim) {
                const float theta =
                    static_cast<float>(positions[static_cast<size_t>(token)]) *
                    std::pow(10000.0f, -2.0f * static_cast<float>(dim) / static_cast<float>(head_dim));
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const size_t lo =
                    (static_cast<size_t>(token) * static_cast<size_t>(heads) + static_cast<size_t>(head)) *
                        static_cast<size_t>(head_dim) +
                    static_cast<size_t>(dim);
                const size_t hi = lo + static_cast<size_t>(half);
                result[lo] = values[lo] * c - values[hi] * s;
                result[hi] = values[hi] * c + values[lo] * s;
            }
        }
    }
    return result;
}

void require_close(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        std::cerr << "size mismatch\n";
        std::exit(1);
    }
    float max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
    }
    if (max_abs > 5.0e-5f) {
        std::cerr << "action expert MLP mismatch: max_abs=" << max_abs << "\n";
        std::exit(1);
    }
}

vlacpp::Tensor tensor(std::vector<int64_t> shape, std::vector<float> data) {
    vlacpp::Tensor result;
    result.shape = std::move(shape);
    result.data = std::move(data);
    return result;
}

} // namespace

int main() {
    vlacpp::ModelConfig config;
    config.openpi_action_expert_width = 2;
    config.openpi_action_expert_q_out = 4;
    config.openpi_action_expert_kv_out = 2;
    config.openpi_action_expert_mlp_width = 3;
    config.openpi_action_expert_layers = 1;
    config.openpi_language_width = 2;
    config.openpi_language_q_out = 4;
    config.openpi_language_kv_out = 2;
    config.openpi_language_mlp_width = 3;
    config.openpi_language_layers = 1;
    config.openpi_vision_width = 3;
    vlacpp::BackendConfig backend;
    backend.n_threads = 1;
    vlacpp::TensorMap tensors;
    const std::string layer_prefix = "model.paligemma_with_expert.gemma_expert.model.layers.0.";
    const std::string mlp_prefix = layer_prefix + "mlp.";
    const std::string language_prefix = "model.paligemma_with_expert.paligemma.model.language_model.layers.0.";
    const std::string language_mlp_prefix = language_prefix + "mlp.";
    tensors["model.paligemma_with_expert.gemma_expert.model.norm.weight"] = tensor({2}, {0.1f, -0.15f});
    tensors["model.paligemma_with_expert.paligemma.model.language_model.norm.weight"] = tensor({2}, {-0.05f, 0.12f});
    tensors["vlacpp.openpi.vision_projector.weight"] = tensor({3, 2}, {0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f});
    tensors["vlacpp.openpi.vision_projector.bias"] = tensor({2}, {0.05f, -0.08f});
    tensors[layer_prefix + "input_layernorm.weight"] = tensor({2}, {-0.1f, 0.25f});
    tensors[layer_prefix + "post_attention_layernorm.weight"] = tensor({2}, {0.05f, -0.2f});
    tensors[layer_prefix + "self_attn.q_proj.weight"] =
        tensor({2, 4}, {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f});
    tensors[layer_prefix + "self_attn.k_proj.weight"] = tensor({2, 2}, {0.2f, 0.1f, -0.3f, 0.5f});
    tensors[layer_prefix + "self_attn.v_proj.weight"] = tensor({2, 2}, {-0.4f, 0.6f, 0.8f, -0.2f});
    tensors[layer_prefix + "self_attn.o_proj.weight"] =
        tensor({4, 2}, {0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f, 0.6f, -0.7f});
    tensors[mlp_prefix + "gate_proj.weight"] = tensor({2, 3}, {0.2f, -0.1f, -0.3f, 0.4f, 0.1f, 0.5f});
    tensors[mlp_prefix + "up_proj.weight"] = tensor({2, 3}, {0.6f, 0.2f, -0.2f, 0.3f, 0.4f, -0.5f});
    tensors[mlp_prefix + "down_proj.weight"] = tensor({3, 2}, {0.3f, -0.2f, 0.1f, -0.4f, 0.2f, 0.5f});
    tensors[language_prefix + "input_layernorm.weight"] = tensors[layer_prefix + "input_layernorm.weight"];
    tensors[language_prefix + "post_attention_layernorm.weight"] = tensors[layer_prefix + "post_attention_layernorm.weight"];
    tensors[language_prefix + "self_attn.q_proj.weight"] = tensors[layer_prefix + "self_attn.q_proj.weight"];
    tensors[language_prefix + "self_attn.k_proj.weight"] = tensors[layer_prefix + "self_attn.k_proj.weight"];
    tensors[language_prefix + "self_attn.v_proj.weight"] = tensors[layer_prefix + "self_attn.v_proj.weight"];
    tensors[language_prefix + "self_attn.o_proj.weight"] = tensors[layer_prefix + "self_attn.o_proj.weight"];
    tensors[language_mlp_prefix + "gate_proj.weight"] = tensors[mlp_prefix + "gate_proj.weight"];
    tensors[language_mlp_prefix + "up_proj.weight"] = tensors[mlp_prefix + "up_proj.weight"];
    tensors[language_mlp_prefix + "down_proj.weight"] = tensors[mlp_prefix + "down_proj.weight"];

    const std::vector<float> input = {1.0f, -2.0f, 0.5f, 0.25f};
    vlacpp::Pi0ActionExpert expert(config, backend, tensors);
    if (!expert.has_layer(0)) {
        std::cerr << "expected action expert layer\n";
        return 1;
    }
    std::vector<float> actual;
    expert.mlp_batch(0, input, 2, actual);

    std::vector<float> norm_actual;
    expert.input_norm_batch(0, input, 2, norm_actual);
    require_close(norm_actual, rms_norm(tensors[layer_prefix + "input_layernorm.weight"].data, input, 2, 2));

    expert.post_attention_norm_batch(0, input, 2, norm_actual);
    require_close(norm_actual, rms_norm(tensors[layer_prefix + "post_attention_layernorm.weight"].data, input, 2, 2));

    expert.final_norm_batch(input, 2, norm_actual);
    require_close(
        norm_actual,
        rms_norm(tensors["model.paligemma_with_expert.gemma_expert.model.norm.weight"].data, input, 2, 2));

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    expert.qkv_batch(0, input, 2, q, k, v);
    require_close(q, linear(tensors[layer_prefix + "self_attn.q_proj.weight"].data, input, 2, 2, 4));
    require_close(k, linear(tensors[layer_prefix + "self_attn.k_proj.weight"].data, input, 2, 2, 2));
    require_close(v, linear(tensors[layer_prefix + "self_attn.v_proj.weight"].data, input, 2, 2, 2));

    const std::vector<float> attention_q = {
        0.2f, -0.1f, 0.5f, 0.3f,
        -0.4f, 0.6f, 0.1f, -0.2f,
        0.7f, 0.0f, -0.3f, 0.4f,
    };
    const std::vector<float> attention_k = {
        -0.2f, 0.4f, 0.3f, -0.5f,
        0.6f, -0.1f, 0.2f, 0.7f,
        -0.3f, 0.8f, -0.4f, 0.1f,
    };
    const std::vector<float> attention_v = {
        0.9f, -0.6f, 0.2f, 0.0f,
        -0.5f, 0.3f, 0.8f, -0.7f,
        0.1f, 0.4f, -0.2f, 0.5f,
    };
    std::vector<float> attention_core;
    expert.self_attention_batch(attention_q, attention_k, attention_v, 3, 2, 2, 2, attention_core);
    require_close(attention_core, self_attention(attention_q, attention_k, attention_v, 3, 2, 2, 2));
    const std::vector<float> attention_mask = {
        0.0f, -INFINITY, -INFINITY,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    expert.self_attention_masked_batch(attention_q, attention_k, attention_v, attention_mask, 3, 2, 2, 2, attention_core);
    require_close(
        attention_core,
        self_attention(attention_q, attention_k, attention_v, attention_mask, 3, 2, 2, 2));

    std::vector<float> rope_actual;
    const std::vector<int> positions = {0, 1, 3};
    expert.rope_batch(attention_q, positions, 3, 2, 2, rope_actual);
    require_close(rope_actual, rope_neox(attention_q, positions, 3, 2, 2));

    const std::vector<float> gqa_k = {
        -0.2f, 0.4f,
        0.6f, -0.1f,
        -0.3f, 0.8f,
    };
    const std::vector<float> gqa_v = {
        0.9f, -0.6f,
        -0.5f, 0.3f,
        0.1f, 0.4f,
    };
    expert.self_attention_batch(attention_q, gqa_k, gqa_v, 3, 2, 1, 2, attention_core);
    require_close(attention_core, self_attention(attention_q, gqa_k, gqa_v, 3, 2, 1, 2));

    const std::vector<float> prefix_q = {
        0.2f, -0.1f, 0.5f, 0.3f,
        -0.4f, 0.6f, 0.1f, -0.2f,
    };
    const std::vector<float> prefix_k = {
        -0.2f, 0.4f,
        0.6f, -0.1f,
        -0.3f, 0.8f,
        0.2f, 0.7f,
    };
    const std::vector<float> prefix_v = {
        0.9f, -0.6f,
        -0.5f, 0.3f,
        0.1f, 0.4f,
        0.8f, -0.7f,
    };
    const std::vector<float> prefix_mask = {
        0.0f, 0.0f, -INFINITY, -INFINITY,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    expert.attention_masked_batch(prefix_q, prefix_k, prefix_v, prefix_mask, 2, 4, 2, 1, 2, attention_core);
    require_close(
        attention_core,
        attention_ref(prefix_q, prefix_k, prefix_v, prefix_mask, 2, 4, 2, 1, 2));

    std::vector<float> attention_out;
    const std::vector<float> attention_values = {0.25f, -0.5f, 0.75f, 1.0f, -1.0f, 0.5f, 0.0f, 0.2f};
    expert.attention_out_batch(0, attention_values, 2, attention_out);
    require_close(
        attention_out,
        linear(tensors[layer_prefix + "self_attn.o_proj.weight"].data, attention_values, 2, 4, 2));

    std::vector<float> gate = linear(tensors[mlp_prefix + "gate_proj.weight"].data, input, 2, 2, 3);
    std::vector<float> up = linear(tensors[mlp_prefix + "up_proj.weight"].data, input, 2, 2, 3);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = gelu(gate[i]) * up[i];
    }
    const std::vector<float> expected = linear(tensors[mlp_prefix + "down_proj.weight"].data, gate, 2, 3, 2);
    require_close(actual, expected);

    std::vector<float> block_actual;
    const std::vector<int> block_positions = {0, 2};
    expert.block_batch(0, input, block_positions, 2, 2, 1, 2, block_actual);

    std::vector<float> block_norm =
        rms_norm(tensors[layer_prefix + "input_layernorm.weight"].data, input, 2, 2);
    std::vector<float> block_q = linear(tensors[layer_prefix + "self_attn.q_proj.weight"].data, block_norm, 2, 2, 4);
    std::vector<float> block_k = linear(tensors[layer_prefix + "self_attn.k_proj.weight"].data, block_norm, 2, 2, 2);
    std::vector<float> block_v = linear(tensors[layer_prefix + "self_attn.v_proj.weight"].data, block_norm, 2, 2, 2);
    block_q = rope_neox(block_q, block_positions, 2, 2, 2);
    block_k = rope_neox(block_k, block_positions, 2, 1, 2);
    std::vector<float> block_attn = self_attention(block_q, block_k, block_v, 2, 2, 1, 2);
    std::vector<float> block_attention_out =
        linear(tensors[layer_prefix + "self_attn.o_proj.weight"].data, block_attn, 2, 4, 2);
    for (size_t i = 0; i < block_attention_out.size(); ++i) {
        block_attention_out[i] += input[i];
    }
    std::vector<float> block_post =
        rms_norm(tensors[layer_prefix + "post_attention_layernorm.weight"].data, block_attention_out, 2, 2);
    std::vector<float> block_gate = linear(tensors[mlp_prefix + "gate_proj.weight"].data, block_post, 2, 2, 3);
    std::vector<float> block_up = linear(tensors[mlp_prefix + "up_proj.weight"].data, block_post, 2, 2, 3);
    for (size_t i = 0; i < block_gate.size(); ++i) {
        block_gate[i] = gelu(block_gate[i]) * block_up[i];
    }
    std::vector<float> block_expected = linear(tensors[mlp_prefix + "down_proj.weight"].data, block_gate, 2, 3, 2);
    for (size_t i = 0; i < block_expected.size(); ++i) {
        block_expected[i] += block_attention_out[i];
    }
    require_close(block_actual, block_expected);

    const std::vector<float> prefix_kv_k = {0.15f, -0.25f};
    const std::vector<float> prefix_kv_v = {0.35f, 0.45f};
    const std::vector<float> prefix_block_mask = {
        0.0f, 0.0f, -INFINITY,
        0.0f, 0.0f, 0.0f,
    };
    std::vector<float> prefix_block_actual;
    expert.block_prefix_batch(
        0,
        input,
        block_positions,
        prefix_kv_k,
        prefix_kv_v,
        prefix_block_mask,
        1,
        2,
        2,
        1,
        2,
        prefix_block_actual);
    std::vector<float> prefix_k_all = prefix_kv_k;
    prefix_k_all.insert(prefix_k_all.end(), block_k.begin(), block_k.end());
    std::vector<float> prefix_v_all = prefix_kv_v;
    prefix_v_all.insert(prefix_v_all.end(), block_v.begin(), block_v.end());
    std::vector<float> prefix_block_attn =
        attention_ref(block_q, prefix_k_all, prefix_v_all, prefix_block_mask, 2, 3, 2, 1, 2);
    std::vector<float> prefix_block_attention_out =
        linear(tensors[layer_prefix + "self_attn.o_proj.weight"].data, prefix_block_attn, 2, 4, 2);
    for (size_t i = 0; i < prefix_block_attention_out.size(); ++i) {
        prefix_block_attention_out[i] += input[i];
    }
    std::vector<float> prefix_block_post =
        rms_norm(tensors[layer_prefix + "post_attention_layernorm.weight"].data, prefix_block_attention_out, 2, 2);
    std::vector<float> prefix_block_gate =
        linear(tensors[mlp_prefix + "gate_proj.weight"].data, prefix_block_post, 2, 2, 3);
    std::vector<float> prefix_block_up =
        linear(tensors[mlp_prefix + "up_proj.weight"].data, prefix_block_post, 2, 2, 3);
    for (size_t i = 0; i < prefix_block_gate.size(); ++i) {
        prefix_block_gate[i] = gelu(prefix_block_gate[i]) * prefix_block_up[i];
    }
    std::vector<float> prefix_block_expected =
        linear(tensors[mlp_prefix + "down_proj.weight"].data, prefix_block_gate, 2, 3, 2);
    for (size_t i = 0; i < prefix_block_expected.size(); ++i) {
        prefix_block_expected[i] += prefix_block_attention_out[i];
    }
    require_close(prefix_block_actual, prefix_block_expected);

    vlacpp::Pi0LanguagePrefix language(config, backend, tensors);
    if (!language.has_layer(0)) {
        std::cerr << "expected language prefix layer\n";
        return 1;
    }
    std::vector<vlacpp::PrefixLayerKv> language_cache;
    std::vector<float> language_actual;
    language.prefill_batch(input, block_positions, 2, 2, 1, 2, language_cache, language_actual);
    if (language_cache.size() != 1) {
        std::cerr << "expected one language prefix cache layer\n";
        return 1;
    }
    require_close(language_cache[0].k, block_k);
    require_close(language_cache[0].v, block_v);
    std::vector<float> language_expected =
        rms_norm(tensors["model.paligemma_with_expert.paligemma.model.language_model.norm.weight"].data,
                 block_expected,
                 2,
                 2);
    require_close(language_actual, language_expected);

    vlacpp::Pi0Vlm vlm(config, backend, tensors);
    if (!vlm.has_vision_projector()) {
        std::cerr << "expected vision projector\n";
        return 1;
    }
    const std::vector<float> vision_tokens = {0.2f, -0.4f, 0.6f, 0.1f, 0.3f, -0.2f};
    std::vector<float> projected;
    vlm.project_vision_tokens(vision_tokens, 2, projected);
    require_close(
        projected,
        linear_add(tensors["vlacpp.openpi.vision_projector.weight"].data,
                   tensors["vlacpp.openpi.vision_projector.bias"].data,
                   vision_tokens,
                   2,
                   3,
                   2));
    if (!vlm.has_language_prefix()) {
        std::cerr << "expected vlm language prefix\n";
        return 1;
    }
    vlacpp::KvCache vlm_cache;
    vlm.prefill_prefix_from_embeddings(vlm_cache, input, 2);
    if (!vlm_cache.prefix_valid || vlm_cache.token_count != 2 || vlm_cache.prefix_layers.size() != 1) {
        std::cerr << "expected vlm prefix cache\n";
        return 1;
    }
    std::vector<vlacpp::PrefixLayerKv> vlm_expected_cache;
    std::vector<float> vlm_expected_out;
    language.prefill_batch(input, {0, 1}, 2, 2, 1, 2, vlm_expected_cache, vlm_expected_out);
    require_close(vlm_cache.prefix_layers[0].k, vlm_expected_cache[0].k);
    require_close(vlm_cache.prefix_layers[0].v, vlm_expected_cache[0].v);

    tensors["vlacpp.openpi.action_in_proj.weight"] = tensor({1, 2}, {0.4f, -0.25f});
    tensors["vlacpp.openpi.action_in_proj.bias"] = tensor({2}, {0.05f, -0.1f});
    tensors["vlacpp.openpi.action_time_mlp_in.weight"] =
        tensor({4, 2}, {0.3f, -0.2f, 0.1f, 0.4f, -0.5f, 0.2f, 0.6f, -0.1f});
    tensors["vlacpp.openpi.action_time_mlp_in.bias"] = tensor({2}, {0.02f, -0.03f});
    tensors["vlacpp.openpi.action_time_mlp_out.weight"] = tensor({2, 2}, {0.5f, -0.4f, 0.3f, 0.2f});
    tensors["vlacpp.openpi.action_time_mlp_out.bias"] = tensor({2}, {0.01f, -0.02f});
    tensors["vlacpp.openpi.action_out_proj.weight"] = tensor({2, 1}, {0.7f, -0.6f});
    tensors["vlacpp.openpi.action_out_proj.bias"] = tensor({1}, {0.08f});

    config.action_horizon = 2;
    config.action_dim = 1;
    vlacpp::Pi0ActionDecoder decoder(config, backend, tensors);
    std::vector<float> velocity;
    const std::vector<float> actions = {0.2f, -0.1f};
    const std::vector<float> state_context = {0.12f, -0.05f};
    decoder.velocity_batch(0.25f, actions, state_context, {}, 5, velocity);

    std::vector<float> action_tokens =
        linear_add(tensors["vlacpp.openpi.action_in_proj.weight"].data,
                   tensors["vlacpp.openpi.action_in_proj.bias"].data,
                   actions,
                   2,
                   1,
                   2);
    const std::vector<float> time_emb = posemb_sincos(0.25f, 2);
    std::vector<float> action_time(8, 0.0f);
    for (int row = 0; row < 2; ++row) {
        std::copy(
            action_tokens.begin() + static_cast<std::ptrdiff_t>(row * 2),
            action_tokens.begin() + static_cast<std::ptrdiff_t>(row * 2 + 2),
            action_time.begin() + static_cast<std::ptrdiff_t>(row * 4));
        std::copy(time_emb.begin(), time_emb.end(), action_time.begin() + static_cast<std::ptrdiff_t>(row * 4 + 2));
    }
    std::vector<float> decoder_hidden =
        linear_add(tensors["vlacpp.openpi.action_time_mlp_in.weight"].data,
                   tensors["vlacpp.openpi.action_time_mlp_in.bias"].data,
                   action_time,
                   2,
                   4,
                   2);
    for (float & value : decoder_hidden) {
        value = silu(value);
    }
    std::vector<float> decoder_suffix =
        linear_add(tensors["vlacpp.openpi.action_time_mlp_out.weight"].data,
                   tensors["vlacpp.openpi.action_time_mlp_out.bias"].data,
                   decoder_hidden,
                   2,
                   2,
                   2);
    decoder_suffix.insert(decoder_suffix.begin(), state_context.begin(), state_context.end());
    std::vector<float> decoder_block;
    const std::vector<int> decoder_positions = {5, 6, 7};
    const std::vector<float> decoder_mask = {
        0.0f, -INFINITY, -INFINITY,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    expert.block_masked_batch(0, decoder_suffix, decoder_positions, decoder_mask, 3, 2, 1, 2, decoder_block);
    std::vector<float> decoder_norm =
        rms_norm(tensors["model.paligemma_with_expert.gemma_expert.model.norm.weight"].data, decoder_block, 3, 2);
    decoder_norm.erase(decoder_norm.begin(), decoder_norm.begin() + 2);
    std::vector<float> decoder_expected =
        linear_add(tensors["vlacpp.openpi.action_out_proj.weight"].data,
                   tensors["vlacpp.openpi.action_out_proj.bias"].data,
                   decoder_norm,
                   2,
                   2,
                   1);
    require_close(velocity, decoder_expected);

    const std::vector<vlacpp::PrefixLayerKv> prefix_layers = {{{0.15f, -0.25f}, {0.35f, 0.45f}}};
    decoder.velocity_batch(0.25f, actions, state_context, prefix_layers, 1, velocity);
    const std::vector<int> decoder_prefix_positions = {1, 2, 3};
    const std::vector<float> decoder_prefix_mask = {
        0.0f, 0.0f, -INFINITY, -INFINITY,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    expert.block_prefix_batch(
        0,
        decoder_suffix,
        decoder_prefix_positions,
        prefix_layers[0].k,
        prefix_layers[0].v,
        decoder_prefix_mask,
        1,
        3,
        2,
        1,
        2,
        decoder_block);
    decoder_norm =
        rms_norm(tensors["model.paligemma_with_expert.gemma_expert.model.norm.weight"].data, decoder_block, 3, 2);
    decoder_norm.erase(decoder_norm.begin(), decoder_norm.begin() + 2);
    decoder_expected =
        linear_add(tensors["vlacpp.openpi.action_out_proj.weight"].data,
                   tensors["vlacpp.openpi.action_out_proj.bias"].data,
                   decoder_norm,
                   2,
                   2,
                   1);
    require_close(velocity, decoder_expected);
    return 0;
}
