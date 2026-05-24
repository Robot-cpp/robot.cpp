#include "models/ggml_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vlacpp {

void require_ggml_weight_2d(const Tensor & tensor, int64_t ne0, int64_t ne1, const char * name) {
    if (tensor.shape.size() != 2 ||
        tensor.shape[0] != ne0 ||
        tensor.shape[1] != ne1 ||
        tensor.data.size() != static_cast<size_t>(ne0 * ne1)) {
        throw std::invalid_argument(std::string(name) + " has incompatible ggml shape");
    }
}

void require_ggml_vector_1d(const Tensor & tensor, int64_t ne0, const char * name) {
    if (tensor.shape.size() != 1 ||
        tensor.shape[0] != ne0 ||
        tensor.data.size() != static_cast<size_t>(ne0)) {
        throw std::invalid_argument(std::string(name) + " has incompatible ggml shape");
    }
}

size_t ggml_graph_context_size(size_t tensor_bytes) {
    return std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
}

ggml_tensor * build_ggml_matmul(
    ggml_context * ctx,
    const GgmlRunner & runner,
    const Tensor & weight,
    ggml_tensor * input) {
    return ggml_mul_mat(ctx, runner.new_weight_2d(ctx, weight), input);
}

ggml_tensor * build_ggml_linear(
    ggml_context * ctx,
    const GgmlRunner & runner,
    const Tensor & weight,
    const Tensor & bias,
    ggml_tensor * input) {
    return ggml_add(
        ctx,
        build_ggml_matmul(ctx, runner, weight, input),
        runner.new_weight_1d(ctx, bias));
}

ggml_tensor * build_ggml_rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * scale,
    float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, input, eps), scale);
}

ggml_tensor * build_ggml_attention(
    ggml_context * ctx,
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    ggml_tensor * kq_mask,
    int heads,
    int kv_heads,
    int head_dim,
    int q_tokens,
    int kv_tokens) {
    if (heads <= 0 || kv_heads <= 0 || head_dim <= 0 || q_tokens <= 0 || kv_tokens <= 0 || heads % kv_heads != 0) {
        throw std::invalid_argument("attention has incompatible ggml dimensions");
    }
    if (kv_heads != heads) {
        ggml_tensor * kv_repeat_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, heads, kv_tokens);
        k = ggml_repeat(ctx, k, kv_repeat_shape);
        v = ggml_repeat(ctx, v, kv_repeat_shape);
    }

    ggml_tensor * q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * scores = ggml_mul_mat(ctx, k_perm, q_perm);
    scores = ggml_soft_max_ext(ctx, scores, kq_mask, scale, 0.0f);
    ggml_tensor * values = ggml_mul_mat(ctx, v_perm, scores);
    ggml_tensor * y = ggml_permute(ctx, values, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, y, static_cast<int64_t>(head_dim) * heads, q_tokens);
}

ggml_tensor * build_ggml_gated_mlp(
    ggml_context * ctx,
    const GgmlRunner & runner,
    const Tensor & gate,
    const Tensor & up,
    const Tensor & down,
    ggml_tensor * input) {
    ggml_tensor * gate_out = ggml_gelu(ctx, build_ggml_matmul(ctx, runner, gate, input));
    ggml_tensor * up_out = build_ggml_matmul(ctx, runner, up, input);
    return build_ggml_matmul(ctx, runner, down, ggml_mul(ctx, gate_out, up_out));
}

void run_ggml_linear_batch(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & output,
    const char * label,
    bool silu) {
    if (weight.shape.size() != 2 || bias.shape.size() != 1) {
        throw std::invalid_argument("linear expects rank-2 weight and rank-1 bias");
    }
    const int64_t ne0 = weight.shape[0];
    const int64_t ne1 = weight.shape[1];
    if (ne0 <= 0 || ne1 <= 0 || batch <= 0 ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(ne0)) {
        throw std::invalid_argument("linear input has incompatible ggml shape");
    }
    require_ggml_weight_2d(weight, ne0, ne1, "linear weight");
    require_ggml_vector_1d(bias, ne1, "linear bias");

    const size_t tensor_bytes =
        (weight.data.size() + bias.data.size() + input.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(ne1) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, input.data(), input.size() * sizeof(float));

    ggml_tensor * y = build_ggml_linear(ctx, runner, weight, bias, x);
    if (silu) {
        y = ggml_silu(ctx, y);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, label != nullptr ? label : "ggml linear graph compute failed");

    output.resize(static_cast<size_t>(batch) * static_cast<size_t>(ne1));
    runner.get_output(y, output.data(), output.size() * sizeof(float));
}

void run_ggml_matmul_batch(
    const Tensor & weight,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & output,
    const char * label) {
    if (weight.shape.size() != 2) {
        throw std::invalid_argument("matmul expects rank-2 weight");
    }
    const int64_t ne0 = weight.shape[0];
    const int64_t ne1 = weight.shape[1];
    if (ne0 <= 0 || ne1 <= 0 || batch <= 0 ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(ne0)) {
        throw std::invalid_argument("matmul input has incompatible ggml shape");
    }
    require_ggml_weight_2d(weight, ne0, ne1, "matmul weight");

    const size_t tensor_bytes =
        (weight.data.size() + input.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(ne1) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, input.data(), input.size() * sizeof(float));

    ggml_tensor * y = build_ggml_matmul(ctx, runner, weight, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, label != nullptr ? label : "ggml matmul graph compute failed");

    output.resize(static_cast<size_t>(batch) * static_cast<size_t>(ne1));
    runner.get_output(y, output.data(), output.size() * sizeof(float));
}

void run_ggml_qkv_batch(
    const Tensor & q_weight,
    const Tensor & k_weight,
    const Tensor & v_weight,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & q,
    std::vector<float> & k,
    std::vector<float> & v,
    const char * label) {
    if (q_weight.shape.size() != 2 || k_weight.shape.size() != 2 || v_weight.shape.size() != 2) {
        throw std::invalid_argument("QKV projection expects rank-2 weights");
    }
    const int64_t width = q_weight.shape[0];
    const int64_t q_out = q_weight.shape[1];
    const int64_t kv_out = k_weight.shape[1];
    if (width <= 0 || q_out <= 0 || kv_out <= 0 || batch <= 0 ||
        k_weight.shape[0] != width ||
        v_weight.shape[0] != width ||
        v_weight.shape[1] != kv_out ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("QKV projection input has incompatible ggml shape");
    }
    require_ggml_weight_2d(q_weight, width, q_out, "Q projection");
    require_ggml_weight_2d(k_weight, width, kv_out, "K projection");
    require_ggml_weight_2d(v_weight, width, kv_out, "V projection");

    const size_t tensor_bytes =
        (q_weight.data.size() + k_weight.data.size() + v_weight.data.size() + input.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(q_out + 2 * kv_out) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, input.data(), input.size() * sizeof(float));

    ggml_tensor * q_out_tensor = build_ggml_matmul(ctx, runner, q_weight, x);
    ggml_tensor * k_out_tensor = build_ggml_matmul(ctx, runner, k_weight, x);
    ggml_tensor * v_out_tensor = build_ggml_matmul(ctx, runner, v_weight, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, q_out_tensor);
    ggml_build_forward_expand(graph, k_out_tensor);
    ggml_build_forward_expand(graph, v_out_tensor);
    runner.compute(ctx, graph, inputs, label != nullptr ? label : "ggml QKV projection graph compute failed");

    q.resize(static_cast<size_t>(batch) * static_cast<size_t>(q_out));
    k.resize(static_cast<size_t>(batch) * static_cast<size_t>(kv_out));
    v.resize(static_cast<size_t>(batch) * static_cast<size_t>(kv_out));
    runner.get_output(q_out_tensor, q.data(), q.size() * sizeof(float));
    runner.get_output(k_out_tensor, k.data(), k.size() * sizeof(float));
    runner.get_output(v_out_tensor, v.data(), v.size() * sizeof(float));
}

void run_ggml_gated_mlp_batch(
    const Tensor & gate,
    const Tensor & up,
    const Tensor & down,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & output,
    const char * label) {
    if (gate.shape.size() != 2 || up.shape.size() != 2 || down.shape.size() != 2) {
        throw std::invalid_argument("gated MLP expects rank-2 weights");
    }
    const int64_t width = gate.shape[0];
    const int64_t hidden = gate.shape[1];
    if (width <= 0 || hidden <= 0 || batch <= 0 ||
        up.shape[0] != width ||
        up.shape[1] != hidden ||
        down.shape[0] != hidden ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(width)) {
        throw std::invalid_argument("gated MLP input has incompatible ggml shape");
    }
    const int64_t out_width = down.shape[1];
    require_ggml_weight_2d(gate, width, hidden, "gated MLP gate");
    require_ggml_weight_2d(up, width, hidden, "gated MLP up");
    require_ggml_weight_2d(down, hidden, out_width, "gated MLP down");

    const size_t tensor_bytes =
        (gate.data.size() + up.data.size() + down.data.size() + input.size()) * sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(hidden + out_width) * sizeof(float);
    const size_t context_size = ggml_graph_context_size(tensor_bytes);
    GgmlRunner runner(backend);
    GgmlContext ctx(runner.init_params(context_size));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, input.data(), input.size() * sizeof(float));

    ggml_tensor * y = build_ggml_gated_mlp(ctx, runner, gate, up, down, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(ctx, graph, inputs, label != nullptr ? label : "ggml gated MLP graph compute failed");

    output.resize(static_cast<size_t>(batch) * static_cast<size_t>(out_width));
    runner.get_output(y, output.data(), output.size() * sizeof(float));
}

} // namespace vlacpp
