#pragma once

#include "models/ggml_runtime.h"
#include "models/model.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vlacpp {

void require_ggml_weight_2d(const Tensor & tensor, int64_t ne0, int64_t ne1, const char * name);
void require_ggml_vector_1d(const Tensor & tensor, int64_t ne0, const char * name);
size_t ggml_graph_context_size(size_t tensor_bytes);
ggml_tensor * build_ggml_matmul(
    ggml_context * ctx,
    const GgmlRunner & runner,
    const Tensor & weight,
    ggml_tensor * input);
ggml_tensor * build_ggml_linear(
    ggml_context * ctx,
    const GgmlRunner & runner,
    const Tensor & weight,
    const Tensor & bias,
    ggml_tensor * input);
ggml_tensor * build_ggml_rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * scale,
    float eps);
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
    int kv_tokens);
ggml_tensor * build_ggml_gated_mlp(
    ggml_context * ctx,
    const GgmlRunner & runner,
    const Tensor & gate,
    const Tensor & up,
    const Tensor & down,
    ggml_tensor * input);
void run_ggml_linear_batch(
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & output,
    const char * label,
    bool silu = false);
void run_ggml_matmul_batch(
    const Tensor & weight,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & output,
    const char * label);
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
    const char * label);
void run_ggml_gated_mlp_batch(
    const Tensor & gate,
    const Tensor & up,
    const Tensor & down,
    const std::vector<float> & input,
    int batch,
    const BackendConfig & backend,
    std::vector<float> & output,
    const char * label);

} // namespace vlacpp
