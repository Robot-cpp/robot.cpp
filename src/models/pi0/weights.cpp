#include "models/pi0/weights.h"

#include "models/pi0/load.h"

#include <stdexcept>
#include <algorithm>
#include <string>

namespace robotcpp::pi0 {

namespace {

ggml_tensor * bind_pi0_tensor(ggml_context * ctx, const std::string & name) {
    if (ctx == nullptr) {
        return nullptr;
    }
    return ggml_get_tensor(ctx, name.c_str());
}

Pi0TransformerLayerWeights build_transformer_layer(ggml_context * ctx, const std::string & prefix) {
    Pi0TransformerLayerWeights out;
    out.input_norm_scale = bind_pi0_tensor(ctx, prefix + "input_layernorm.scale");
    out.post_norm_scale = bind_pi0_tensor(ctx, prefix + "post_attention_layernorm.scale");
    out.q = bind_pi0_tensor(ctx, prefix + "self_attn.q_proj.weight");
    out.k = bind_pi0_tensor(ctx, prefix + "self_attn.k_proj.weight");
    out.v = bind_pi0_tensor(ctx, prefix + "self_attn.v_proj.weight");
    out.out = bind_pi0_tensor(ctx, prefix + "self_attn.o_proj.weight");
    out.gate = bind_pi0_tensor(ctx, prefix + "mlp.gate_proj.weight");
    out.up = bind_pi0_tensor(ctx, prefix + "mlp.up_proj.weight");
    out.down = bind_pi0_tensor(ctx, prefix + "mlp.down_proj.weight");
    return out;
}

} // namespace

Pi0Weights build_pi0_weights(
    const Pi0ModelConfig & config,
    ggml_context * vit,
    ggml_context * mmproj,
    ggml_context * llm,
    ggml_context * state,
    ggml_context * action_decoder) {
    Pi0Weights out;
    const Pi0Config & pi0 = pi0_config(config);
    out.state_w = bind_pi0_tensor(state, pi0_state_tensor(config, "weight"));
    out.state_b = bind_pi0_tensor(state, pi0_state_tensor(config, "bias"));
    out.action_in_w = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_in_proj.weight"));
    out.action_in_b = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_in_proj.bias"));
    out.action_time_in_w = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_in.weight"));
    out.action_time_in_b = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_in.bias"));
    out.action_time_out_w = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_out.weight"));
    out.action_time_out_b = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_out.bias"));
    out.action_out_w = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_out_proj.weight"));
    out.action_out_b = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_out_proj.bias"));
    out.action_norm_scale = bind_pi0_tensor(action_decoder, pi0_action_decoder_tensor(config, "norm.scale"));
    out.action_layers.reserve(static_cast<size_t>(std::max(0, pi0.action.expert_layers)));
    for (int layer = 0; layer < pi0.action.expert_layers; ++layer) {
        out.action_layers.push_back(build_transformer_layer(action_decoder, pi0_action_decoder_layer_prefix(config, layer)));
    }

    out.llm_norm_scale = bind_pi0_tensor(llm, pi0_llm_tensor(config, "norm.scale"));
    out.lm_head = bind_pi0_tensor(llm, pi0_lm_head(config));
    out.llm_layers.reserve(static_cast<size_t>(std::max(0, pi0.llm.layers)));
    for (int layer = 0; layer < pi0.llm.layers; ++layer) {
        out.llm_layers.push_back(build_transformer_layer(llm, pi0_llm_layer_prefix(config, layer)));
    }

    out.merger_w = bind_pi0_tensor(mmproj, pi0_merger_tensor(config, "weight"));
    out.merger_b = bind_pi0_tensor(mmproj, pi0_merger_tensor(config, "bias"));

    out.vit_patch_w = bind_pi0_tensor(vit, pi0_vit_tensor(config, "embeddings.patch_embedding.weight"));
    out.vit_patch_b = bind_pi0_tensor(vit, pi0_vit_tensor(config, "embeddings.patch_embedding.bias"));
    out.vit_pos = bind_pi0_tensor(vit, pi0_vit_tensor(config, "embeddings.position_embedding.weight"));
    out.vit_post_norm_w = bind_pi0_tensor(vit, pi0_vit_tensor(config, "post_layernorm.weight"));
    out.vit_post_norm_b = bind_pi0_tensor(vit, pi0_vit_tensor(config, "post_layernorm.bias"));
    out.vit_layers.reserve(static_cast<size_t>(std::max(0, pi0.vision.layers)));
    for (int layer = 0; layer < pi0.vision.layers; ++layer) {
        const std::string prefix = pi0_vit_layer_prefix(config, layer);
        Pi0VisionLayerWeights current;
        current.norm1_w = bind_pi0_tensor(vit, prefix + "layer_norm1.weight");
        current.norm1_b = bind_pi0_tensor(vit, prefix + "layer_norm1.bias");
        current.q_w = bind_pi0_tensor(vit, prefix + "self_attn.q_proj.weight");
        current.q_b = bind_pi0_tensor(vit, prefix + "self_attn.q_proj.bias");
        current.k_w = bind_pi0_tensor(vit, prefix + "self_attn.k_proj.weight");
        current.k_b = bind_pi0_tensor(vit, prefix + "self_attn.k_proj.bias");
        current.v_w = bind_pi0_tensor(vit, prefix + "self_attn.v_proj.weight");
        current.v_b = bind_pi0_tensor(vit, prefix + "self_attn.v_proj.bias");
        current.out_w = bind_pi0_tensor(vit, prefix + "self_attn.out_proj.weight");
        current.out_b = bind_pi0_tensor(vit, prefix + "self_attn.out_proj.bias");
        current.norm2_w = bind_pi0_tensor(vit, prefix + "layer_norm2.weight");
        current.norm2_b = bind_pi0_tensor(vit, prefix + "layer_norm2.bias");
        current.fc1_w = bind_pi0_tensor(vit, prefix + "mlp.fc1.weight");
        current.fc1_b = bind_pi0_tensor(vit, prefix + "mlp.fc1.bias");
        current.fc2_w = bind_pi0_tensor(vit, prefix + "mlp.fc2.weight");
        current.fc2_b = bind_pi0_tensor(vit, prefix + "mlp.fc2.bias");
        out.vit_layers.push_back(current);
    }
    return out;
}

} // namespace robotcpp::pi0
