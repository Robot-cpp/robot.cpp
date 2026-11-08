#include "component.h"
#include "ops.h"
#include "graph_ops.h"
#include <cmath>

namespace robotcpp::bitvla {
void build_language(component & model, int tokens, int action_start) {
    auto * ctx = model.begin_graph();
    graph_ops::GraphBuildContext graph{model};
    const int width = model.integer("bitnet.embedding_length");
    const int heads = model.integer("bitnet.attention.head_count");
    const int kv_heads = model.integer("bitnet.attention.head_count_kv");
    const int layers = model.integer("bitnet.block_count");
    const float eps = model.scalar("bitnet.attention.layer_norm_rms_epsilon");
    model.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
    model.register_language_input();
    auto * cosine = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width / heads, 1, tokens);
    auto * sine = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width / heads, 1, tokens);
    model.register_rope_inputs(cosine, sine);
    auto * cur = model.input;
    bool action_rows_only = false;
    const graph_ops::LanguageShape shape{width, heads, kv_heads, layers, tokens, action_start, eps};
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        graph_ops::LayerWeights weights{model, prefix};
        bool query_action_rows = false;
        const auto projections = graph_ops::language_qkv(graph, cur, weights, layer, eps,
            shape, cosine, sine, query_action_rows);
        const auto attended = graph_ops::language_attention(graph, projections, cur, weights,
            shape, layer, cosine, sine, action_rows_only, query_action_rows);
        action_rows_only = attended.action_rows_only;
        cur = ggml_add(ctx, attended.residual, attended.output);
        auto * residual = cur;
        cur = graph_ops::language_ffn(graph, cur, weights, layer, eps);
        cur = ggml_add(ctx, residual, cur);
    }
    cur = rms_norm(ctx, cur, model.tensor("output_norm.weight"), eps);
    // predict_action uses the hidden state immediately before each action label.
    if (!action_rows_only) cur = ggml_view_2d(ctx, cur, width, 56, cur->nb[1], (action_start - 1) * cur->nb[1]);
    model.finish_language_graph(ggml_cont(ctx, cur), cosine, sine, tokens);
}

} // namespace robotcpp::bitvla
