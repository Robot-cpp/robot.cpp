#pragma once
#include <initializer_list>
#include <cstddef>

#include "ggml.h"
#include <vector>

namespace robotcpp::bitvla {

// Custom activation operators retain F32 values. Each I2S BitLinear input is
// encoded as signed 8-bit values with a separate absmax scale for every token.
ggml_tensor * score_scale(ggml_context * ctx, ggml_tensor * input, float scale);
ggml_tensor * vision_rgb_u8_f32(ggml_context * ctx, ggml_tensor * rgb, ggml_tensor * lookup, int size);
ggml_tensor * gated_relu_square_f32(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up);
ggml_tensor * gated_rms_affine_i2s_encode(ggml_context * ctx,ggml_tensor * gate,ggml_tensor * up,ggml_tensor * weight,float eps);
ggml_tensor * gated_rms_affine_f32(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * weight, float eps);
ggml_tensor * bias_gelu_f32_i2s_encode(ggml_context * ctx,ggml_tensor * added);
ggml_tensor * gelu_f32_i2s_encode(ggml_context * ctx,ggml_tensor * input);
ggml_tensor * f32_i2s_encode(ggml_context * ctx,ggml_tensor * input);
ggml_tensor * contiguous_layout_i2s_encode(ggml_context * ctx,ggml_tensor * input);
// Reuse the original CPU bias/layout implementation without changing its body.
void bias_attention_layout_cpu_existing(ggml_tensor * dst,int ith,int nth,bool value_layout);
// Inputs already have the exact original contiguous Q/K/V GEMM layouts.
ggml_tensor * attention_packed_i2s_encode(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,bool eager);
// Already packed Q/K/V, retaining the ordinary F32 attention output contract.
ggml_tensor * attention_packed_f32(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,bool eager);
ggml_tensor * attention_i2s_encode(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,int kv_heads,bool eager);
ggml_tensor * contiguous_layout_f32(ggml_context * ctx, ggml_tensor * input);
ggml_tensor * activation_quant(ggml_context * ctx, ggml_tensor * tensor);
ggml_tensor * linear_input(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * tensor);
// Share one immutable encoding among projections of the same tensor in this graph.
// Unsupported/mixed weight formats retain each projection's original path.
ggml_tensor * shared_linear_input(ggml_context * ctx, ggml_tensor * input,
                                   std::initializer_list<ggml_tensor *> weights);
ggml_tensor * linear_shared(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
                             ggml_tensor * encoded, ggml_tensor * bias = nullptr);
void initialize_rope_cache(ggml_tensor * cosine, ggml_tensor * sine);
ggml_tensor * rope_attention_layout(ggml_context * ctx, ggml_tensor * input, ggml_tensor * cosine, ggml_tensor * sine);
ggml_tensor * rope(ggml_context * ctx, ggml_tensor * tensor, ggml_tensor * cosine, ggml_tensor * sine);
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
                      ggml_tensor * bias = nullptr);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight,
                          ggml_tensor * bias, float eps);
ggml_tensor * layer_norm_affine_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,ggml_tensor * bias,float eps);
ggml_tensor * layer_norm_affine_f32(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias, float eps);
ggml_tensor * softmax_f32(ggml_context * ctx, ggml_tensor * input);
ggml_tensor * softmax_score_f32(ggml_context * ctx, ggml_tensor * input, float scale);
ggml_tensor * rms_norm_f32(ggml_context * ctx, ggml_tensor * input, float eps);
ggml_tensor * permuted_rms_affine_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,float eps);
ggml_tensor * attention_rms_i2s_encode(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,int kv_heads,bool eager,ggml_tensor * weight,float eps);
ggml_tensor * cpu_order_rms_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,float eps);
ggml_tensor * rms_affine_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,float eps);
ggml_tensor * rms_affine_f32(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, float eps);
ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, float eps);
ggml_tensor * rms_norm_cpu_order(ggml_context * ctx, ggml_tensor * input, float eps);
ggml_tensor * bias_attention_layout(ggml_context * ctx, ggml_tensor * input,
                                    ggml_tensor * bias, int heads, bool value_layout);
ggml_tensor * attention(ggml_context * ctx, ggml_tensor * query, ggml_tensor * key,
                         ggml_tensor * value, int heads, int kv_heads, bool eager);

#ifdef BITVLA_USE_CUDA
void layer_norm_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void layer_norm_affine_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void softmax_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void softmax_score_f32_cuda(ggml_tensor * dst, void * stream, void *);
void rms_norm_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void gated_rms_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void gated_rms_affine_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void permuted_rms_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void rms_affine_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void rms_affine_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void bias_query_key_layout_cuda(ggml_tensor * dst, void * stream, void * userdata);
void bias_value_layout_cuda(ggml_tensor * dst, void * stream, void * userdata);
void contiguous_layout_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void contiguous_layout_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void bias_gelu_f32_i2s_encode_cuda(ggml_tensor * dst,void * stream,void *);
void gelu_f32_i2s_encode_cuda(ggml_tensor * dst,void * stream,void *);
void f32_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void rope_attention_layout_cuda(ggml_tensor * dst, void * stream, void * userdata);
void rope_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void score_scale_cuda(ggml_tensor * dst, void * stream, void * userdata);
void vision_rgb_u8_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
void gated_relu_square_f32_cuda(ggml_tensor * dst, void * stream, void * userdata);
bool rope_cache_cuda(ggml_tensor * cosine, ggml_tensor * sine);
void cpu_order_rms_i2s_encode_cuda(ggml_tensor * dst,void * stream,void * userdata);
void rms_norm_cpu_order_cuda(ggml_tensor * dst, void * stream, void * userdata);
#endif

} // namespace robotcpp::bitvla
