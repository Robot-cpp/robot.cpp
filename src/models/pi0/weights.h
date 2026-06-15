#pragma once

#include "models/pi0/types.h"

#include "ggml.h"

#include <vector>

namespace robotcpp::pi0 {

struct Pi0TransformerLayerWeights {
    ggml_tensor * input_norm_scale = nullptr;
    ggml_tensor * post_norm_scale = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
};

struct Pi0VisionLayerWeights {
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * norm1_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * out_w = nullptr;
    ggml_tensor * out_b = nullptr;
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * norm2_b = nullptr;
    ggml_tensor * fc1_w = nullptr;
    ggml_tensor * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr;
    ggml_tensor * fc2_b = nullptr;
};

struct Pi0Weights {
    ggml_tensor * state_w = nullptr;
    ggml_tensor * state_b = nullptr;
    ggml_tensor * action_in_w = nullptr;
    ggml_tensor * action_in_b = nullptr;
    ggml_tensor * action_time_in_w = nullptr;
    ggml_tensor * action_time_in_b = nullptr;
    ggml_tensor * action_time_out_w = nullptr;
    ggml_tensor * action_time_out_b = nullptr;
    ggml_tensor * action_out_w = nullptr;
    ggml_tensor * action_out_b = nullptr;
    ggml_tensor * action_norm_scale = nullptr;
    std::vector<Pi0TransformerLayerWeights> action_layers;

    ggml_tensor * llm_norm_scale = nullptr;
    ggml_tensor * lm_head = nullptr;
    std::vector<Pi0TransformerLayerWeights> llm_layers;

    ggml_tensor * merger_w = nullptr;
    ggml_tensor * merger_b = nullptr;

    ggml_tensor * vit_patch_w = nullptr;
    ggml_tensor * vit_patch_b = nullptr;
    ggml_tensor * vit_pos = nullptr;
    ggml_tensor * vit_post_norm_w = nullptr;
    ggml_tensor * vit_post_norm_b = nullptr;
    std::vector<Pi0VisionLayerWeights> vit_layers;
};

Pi0Weights build_pi0_weights(
    const Pi0ModelConfig & config,
    ggml_context * vit,
    ggml_context * mmproj,
    ggml_context * llm,
    ggml_context * state,
    ggml_context * action_decoder);

} // namespace robotcpp::pi0
