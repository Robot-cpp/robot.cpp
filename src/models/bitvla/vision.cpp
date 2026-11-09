#include "component.h"
#include "ops.h"
#include "graph_ops.h"
#include <cstring>
#include <cstdlib>

namespace robotcpp::bitvla {
void build_vision(component & model) {
    auto * ctx = model.begin_graph();
    graph_ops::GraphBuildContext graph{model};
    const int size   = model.integer("clip.vision.image_size");
    const int patch  = model.integer("clip.vision.patch_size");
    const int width  = model.integer("clip.vision.embedding_length");
    const int heads  = model.integer("clip.vision.attention.head_count");
    const int layers = model.integer("clip.vision.block_count");
    const float eps  = model.scalar("clip.vision.attention.layer_norm_epsilon");
    const int tokens = (size / patch) * (size / patch);
    model.create_vision_input(size);
    auto * kernel = model.tensor("v.patch_embd.weight");
    // CPU im2col requires an F16 kernel descriptor; it reads only its shape.
    auto * patch_shape = graph_ops::vision_patch_descriptor(graph, kernel);
    auto * patches     = ggml_im2col(ctx, patch_shape, model.input, patch, patch, 0, 0, 1, 1, true, GGML_TYPE_F16);
    patches            = ggml_reshape_2d(ctx, patches, patch * patch * 3, tokens);
    // ggml's BF16 matmul accepts F32 activations, not the F16 im2col output.
    if (kernel->type == GGML_TYPE_BF16)
        patches = ggml_cast(ctx, patches, GGML_TYPE_F32);
    auto * cur = linear(ctx, ggml_reshape_2d(ctx, kernel, patch * patch * 3, width), patches);
    cur        = ggml_add(ctx, cur, model.tensor("v.patch_embd.bias"));
    cur        = ggml_add(ctx, cur, ggml_cast(ctx, model.tensor("v.position_embd.weight"), GGML_TYPE_F32));
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "v.blk." + std::to_string(layer) + ".";
        graph_ops::LayerWeights weights{model, prefix};
        auto * residual        = cur;
        const auto projections = graph_ops::vision_qkv(graph, cur, weights, layer, width, heads, eps);
        cur                    = graph_ops::vision_attention(graph, projections, weights, layer, width, heads);
        cur                    = ggml_add(ctx, residual, cur);
        residual               = cur;
        cur                    = graph_ops::vision_ffn(graph, cur, weights, layer, eps);
        cur                    = ggml_add(ctx, residual, cur);
    }
    // Official vision_feature_layer=-1 selects encoder hidden_states[-1],
    // before the optional SigLIP pooling head and post-layer norm.
    cur = linear(ctx, model.tensor("mm.0.weight"), cur, model.tensor("mm.0.bias"));
    cur = ggml_gelu_erf(ctx, cur);
    cur = linear(ctx, model.tensor("mm.2.weight"), cur, model.tensor("mm.2.bias"));
    model.finish_vision_graph(cur);
}

void encode_vision_device(component & model, const uint8_t * rgb, int stride) {
    model.compute_vision(rgb, stride);
}

std::vector<float> encode_vision(component & model, const uint8_t * rgb, int stride) {
    encode_vision_device(model, rgb, stride);
    return model.read(model.output);
}
} // namespace robotcpp::bitvla
