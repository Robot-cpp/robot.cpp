#pragma once
#include "ggml.h"

namespace robotcpp::bitvla {
ggml_tensor * i2s_encode(ggml_context * ctx, ggml_tensor * input, int padded_k);
ggml_tensor * i2s_linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input, bool optimized_route = false);
// Graph-local reuse only: encoded must be produced from this exact input.
ggml_tensor * i2s_linear_encoded(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input, ggml_tensor * encoded,
                                 bool optimized_route = false);
// Explicit encoded payload entry: no assertion about a separate F32 input.
// The existing i2s_linear_encoded exact-input contract remains unchanged.
ggml_tensor * i2s_linear_preencoded(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * encoded,
                                    bool optimized_route = false);
// Optional native aligned packed matmul -> F32 bias addition -> attention layout.
// Three actual dependencies; unsupported layouts return nullptr for the old graph.
ggml_tensor * i2s_linear_preencoded_attention_qkv(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * query,
                                                  ggml_tensor * key, ggml_tensor * value, ggml_tensor * query_bias,
                                                  ggml_tensor * key_bias, ggml_tensor * value_bias, int heads);
ggml_tensor * i2s_linear_preencoded_attention_layout(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * encoded,
                                                     ggml_tensor * bias, int heads, bool value_layout);
// Optional selected paired projection; returns a true F32 [M,N,2] parent.
// Caller uses ordinary same-type contiguous F32 views at offsets0 and nb[2].
ggml_tensor * i2s_linear_preencoded_pair(ggml_context * ctx, ggml_tensor * weight0, ggml_tensor * weight1,
                                         ggml_tensor * encoded);
// Optional original packed selector pair, used by the full-context K/V projections.
ggml_tensor * i2s_linear_preencoded_packed_pair(ggml_context * ctx, ggml_tensor * weight0, ggml_tensor * weight1,
                                                ggml_tensor * encoded);

// Optional exact full-context Q/K/V: true flat F32 parent, four actual dependencies.
// Standard F32 views cover Q[2560,N],K[640,N],V[640,N] consecutive spans.
ggml_tensor * i2s_linear_preencoded_qkv(ggml_context * ctx, ggml_tensor * query, ggml_tensor * key, ggml_tensor * value,
                                        ggml_tensor * encoded);

// Shared implementation for custom CPU producers; calls the original encoder.
void i2s_encode_cpu(ggml_tensor * dst, int ith, int nth, void * userdata);
#ifdef BITVLA_USE_CUDA
void i2s_matmul_qkv_cuda(ggml_tensor * dst, void * stream, void *);
void i2s_encode_cuda(ggml_tensor * dst, void * stream, void *);
void i2s_matmul_cuda(ggml_tensor * dst, void * stream, void *);
void i2s_matmul_pair_cuda(ggml_tensor * dst, void * stream, void *);
void i2s_matmul_packed_pair_cuda(ggml_tensor * dst, void * stream, void *);
void i2s_matmul_attention_qkv_cuda(ggml_tensor * dst, void * stream, void * userdata);
void i2s_matmul_query_key_layout_cuda(ggml_tensor * dst, void * stream, void *);
void i2s_matmul_value_layout_cuda(ggml_tensor * dst, void * stream, void *);
#endif
} // namespace robotcpp::bitvla
