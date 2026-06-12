#pragma once

#include "core/types.h"

#include <any>
#include <stdexcept>

namespace vlacpp {

struct Pi0VisionConfig {
    VLAComponentConfig component = {"vit", "openpi-vit", "pi0.vit.", {}};
    int width = 0;
    int patch_height = 0;
    int patch_width = 0;
    int layers = 0;
    int heads = 0;
    float norm_epsilon = 0.0f;
};

struct Pi0LlmConfig {
    VLAComponentConfig component = {"llm", "gemma", "pi0.llm.", {}};
    int width = 0;
    int q_out = 0;
    int kv_out = 0;
    int mlp_width = 0;
    int layers = 0;
};

struct Pi0MmprojConfig {
    VLAComponentConfig component = {"mmproj", "openpi-mmproj", "pi0.merger.", {}};
};

struct Pi0ActionConfig {
    VLAComponentConfig component = {"action_decoder", "pi0-action-decoder", "pi0.action_decoder.", {}};
    int width = 0;
    int expert_width = 0;
    int expert_q_out = 0;
    int expert_kv_out = 0;
    int expert_mlp_width = 0;
    int expert_layers = 0;
};

struct Pi0StateConfig {
    VLAComponentConfig component = {"state", "openpi-state-proj", "pi0.action_decoder.state_proj.", {}};
};

struct Pi0Config {
    Pi0VisionConfig vision;
    Pi0MmprojConfig mmproj;
    Pi0LlmConfig llm;
    VLAComponentConfig tokenizer = {"tokenizer", "sentencepiece", "", {}};
    Pi0ActionConfig action;
    Pi0StateConfig state;
};

inline Pi0Config & ensure_pi0_config(ModelConfig & config) {
    Pi0Config * pi0 = std::any_cast<Pi0Config>(&config.specific);
    if (pi0 == nullptr) {
        config.specific = Pi0Config{};
        pi0 = std::any_cast<Pi0Config>(&config.specific);
    }
    return *pi0;
}

inline const Pi0Config & pi0_config(const ModelConfig & config) {
    const Pi0Config * pi0 = std::any_cast<Pi0Config>(&config.specific);
    if (pi0 == nullptr) {
        throw std::invalid_argument("model config is not pi0");
    }
    return *pi0;
}

} // namespace vlacpp
