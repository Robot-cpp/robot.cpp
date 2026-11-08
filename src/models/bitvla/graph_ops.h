#pragma once

#include "ggml.h"
#include <string>

namespace robotcpp::bitvla { class component; }

namespace robotcpp::bitvla::graph_ops {

struct GraphBuildContext { component & runtime; };

// Lazy lookup preserves the original guard/fallback tensor access order.
struct LayerWeights {
    component & runtime;
    std::string prefix;
    ggml_tensor * weight(const char * name) const;
    ggml_tensor * bias(const char * name) const;
};

struct QKV { ggml_tensor * q; ggml_tensor * k; ggml_tensor * v; bool packed = false; };
struct LanguageShape { int width, heads, kv_heads, layers, tokens, action_start; float eps; };
struct AttentionResult { ggml_tensor * output; ggml_tensor * residual; bool action_rows_only; };

QKV language_qkv(GraphBuildContext &, ggml_tensor * input, const LayerWeights &, int layer, float eps,
    const LanguageShape &, ggml_tensor * cosine, ggml_tensor * sine, bool & query_action_rows);
AttentionResult language_attention(GraphBuildContext &, QKV, ggml_tensor * residual,
    const LayerWeights &, const LanguageShape &, int layer, ggml_tensor * cosine,
    ggml_tensor * sine, bool action_rows_only, bool query_action_rows = false);
ggml_tensor * language_ffn(GraphBuildContext &, ggml_tensor * input, const LayerWeights &, int layer, float eps);
ggml_tensor * vision_patch_descriptor(GraphBuildContext &, ggml_tensor * kernel);
QKV vision_qkv(GraphBuildContext &, ggml_tensor * input, const LayerWeights &, int layer, int width, int heads, float eps);
ggml_tensor * vision_attention(GraphBuildContext &, QKV, const LayerWeights &, int layer, int width, int heads);
ggml_tensor * vision_ffn(GraphBuildContext &, ggml_tensor * input, const LayerWeights &, int layer, float eps);

} // namespace robotcpp::bitvla::graph_ops
