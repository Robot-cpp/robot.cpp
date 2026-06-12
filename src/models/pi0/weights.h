#pragma once

#include "core/types.h"

#include <vector>

namespace vlacpp {

struct Pi0TransformerLayerWeights {
    const Tensor * input_norm_scale = nullptr;
    const Tensor * post_norm_scale = nullptr;
    const Tensor * q = nullptr;
    const Tensor * k = nullptr;
    const Tensor * v = nullptr;
    const Tensor * out = nullptr;
    const Tensor * gate = nullptr;
    const Tensor * up = nullptr;
    const Tensor * down = nullptr;
};

struct Pi0VisionLayerWeights {
    const Tensor * norm1_w = nullptr;
    const Tensor * norm1_b = nullptr;
    const Tensor * q_w = nullptr;
    const Tensor * q_b = nullptr;
    const Tensor * k_w = nullptr;
    const Tensor * k_b = nullptr;
    const Tensor * v_w = nullptr;
    const Tensor * v_b = nullptr;
    const Tensor * out_w = nullptr;
    const Tensor * out_b = nullptr;
    const Tensor * norm2_w = nullptr;
    const Tensor * norm2_b = nullptr;
    const Tensor * fc1_w = nullptr;
    const Tensor * fc1_b = nullptr;
    const Tensor * fc2_w = nullptr;
    const Tensor * fc2_b = nullptr;
};

struct Pi0Weights {
    const Tensor * state_w = nullptr;
    const Tensor * state_b = nullptr;
    const Tensor * action_in_w = nullptr;
    const Tensor * action_in_b = nullptr;
    const Tensor * action_time_in_w = nullptr;
    const Tensor * action_time_in_b = nullptr;
    const Tensor * action_time_out_w = nullptr;
    const Tensor * action_time_out_b = nullptr;
    const Tensor * action_out_w = nullptr;
    const Tensor * action_out_b = nullptr;
    const Tensor * action_norm_scale = nullptr;
    std::vector<Pi0TransformerLayerWeights> action_layers;

    const Tensor * llm_norm_scale = nullptr;
    const Tensor * lm_head = nullptr;
    std::vector<Pi0TransformerLayerWeights> llm_layers;

    const Tensor * merger_w = nullptr;
    const Tensor * merger_b = nullptr;

    const Tensor * vit_patch_w = nullptr;
    const Tensor * vit_patch_b = nullptr;
    const Tensor * vit_pos = nullptr;
    const Tensor * vit_post_norm_w = nullptr;
    const Tensor * vit_post_norm_b = nullptr;
    std::vector<Pi0VisionLayerWeights> vit_layers;
};

Pi0Weights build_pi0_weights(
    const ModelConfig & config,
    const TensorMap & vit,
    const TensorMap & mmproj,
    const TensorMap & llm,
    const TensorMap & state,
    const TensorMap & action_decoder);

} // namespace vlacpp
