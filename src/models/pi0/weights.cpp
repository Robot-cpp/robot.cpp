#include "models/pi0/weights.h"

#include "models/pi0/load.h"

#include <algorithm>
#include <string>

namespace vlacpp {

namespace {

Pi0TransformerLayerWeights build_transformer_layer(const TensorMap & tensors, const std::string & prefix) {
    Pi0TransformerLayerWeights out;
    out.input_norm_scale = find_tensor(tensors, prefix + "input_layernorm.scale");
    out.post_norm_scale = find_tensor(tensors, prefix + "post_attention_layernorm.scale");
    out.q = find_tensor(tensors, prefix + "self_attn.q_proj.weight");
    out.k = find_tensor(tensors, prefix + "self_attn.k_proj.weight");
    out.v = find_tensor(tensors, prefix + "self_attn.v_proj.weight");
    out.out = find_tensor(tensors, prefix + "self_attn.o_proj.weight");
    out.gate = find_tensor(tensors, prefix + "mlp.gate_proj.weight");
    out.up = find_tensor(tensors, prefix + "mlp.up_proj.weight");
    out.down = find_tensor(tensors, prefix + "mlp.down_proj.weight");
    return out;
}

} // namespace

Pi0Weights build_pi0_weights(
    const ModelConfig & config,
    const TensorMap & vit,
    const TensorMap & mmproj,
    const TensorMap & llm,
    const TensorMap & state,
    const TensorMap & action_decoder) {
    Pi0Weights out;
    const Pi0Config & pi0 = pi0_config(config);
    out.state_w = find_tensor(state, pi0_state_tensor(config, "weight"));
    out.state_b = find_tensor(state, pi0_state_tensor(config, "bias"));
    out.action_in_w = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_in_proj.weight"));
    out.action_in_b = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_in_proj.bias"));
    out.action_time_in_w = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_in.weight"));
    out.action_time_in_b = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_in.bias"));
    out.action_time_out_w = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_out.weight"));
    out.action_time_out_b = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_time_mlp_out.bias"));
    out.action_out_w = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_out_proj.weight"));
    out.action_out_b = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "action_out_proj.bias"));
    out.action_norm_scale = find_tensor(action_decoder, pi0_action_decoder_tensor(config, "norm.scale"));
    out.action_layers.reserve(static_cast<size_t>(std::max(0, pi0.action.expert_layers)));
    for (int layer = 0; layer < pi0.action.expert_layers; ++layer) {
        out.action_layers.push_back(build_transformer_layer(action_decoder, pi0_action_decoder_layer_prefix(config, layer)));
    }

    out.llm_norm_scale = find_tensor(llm, pi0_llm_tensor(config, "norm.scale"));
    out.lm_head = find_tensor(llm, pi0_lm_head(config));
    out.llm_layers.reserve(static_cast<size_t>(std::max(0, pi0.llm.layers)));
    for (int layer = 0; layer < pi0.llm.layers; ++layer) {
        out.llm_layers.push_back(build_transformer_layer(llm, pi0_llm_layer_prefix(config, layer)));
    }

    out.merger_w = find_tensor(mmproj, pi0_merger_tensor(config, "weight"));
    out.merger_b = find_tensor(mmproj, pi0_merger_tensor(config, "bias"));

    out.vit_patch_w = find_tensor(vit, pi0_vit_tensor(config, "embeddings.patch_embedding.weight"));
    out.vit_patch_b = find_tensor(vit, pi0_vit_tensor(config, "embeddings.patch_embedding.bias"));
    out.vit_pos = find_tensor(vit, pi0_vit_tensor(config, "embeddings.position_embedding.weight"));
    out.vit_post_norm_w = find_tensor(vit, pi0_vit_tensor(config, "post_layernorm.weight"));
    out.vit_post_norm_b = find_tensor(vit, pi0_vit_tensor(config, "post_layernorm.bias"));
    out.vit_layers.reserve(static_cast<size_t>(std::max(0, pi0.vision.layers)));
    for (int layer = 0; layer < pi0.vision.layers; ++layer) {
        const std::string prefix = pi0_vit_layer_prefix(config, layer);
        Pi0VisionLayerWeights current;
        current.norm1_w = find_tensor(vit, prefix + "layer_norm1.weight");
        current.norm1_b = find_tensor(vit, prefix + "layer_norm1.bias");
        current.q_w = find_tensor(vit, prefix + "self_attn.q_proj.weight");
        current.q_b = find_tensor(vit, prefix + "self_attn.q_proj.bias");
        current.k_w = find_tensor(vit, prefix + "self_attn.k_proj.weight");
        current.k_b = find_tensor(vit, prefix + "self_attn.k_proj.bias");
        current.v_w = find_tensor(vit, prefix + "self_attn.v_proj.weight");
        current.v_b = find_tensor(vit, prefix + "self_attn.v_proj.bias");
        current.out_w = find_tensor(vit, prefix + "self_attn.out_proj.weight");
        current.out_b = find_tensor(vit, prefix + "self_attn.out_proj.bias");
        current.norm2_w = find_tensor(vit, prefix + "layer_norm2.weight");
        current.norm2_b = find_tensor(vit, prefix + "layer_norm2.bias");
        current.fc1_w = find_tensor(vit, prefix + "mlp.fc1.weight");
        current.fc1_b = find_tensor(vit, prefix + "mlp.fc1.bias");
        current.fc2_w = find_tensor(vit, prefix + "mlp.fc2.weight");
        current.fc2_b = find_tensor(vit, prefix + "mlp.fc2.bias");
        out.vit_layers.push_back(current);
    }
    return out;
}

} // namespace vlacpp
