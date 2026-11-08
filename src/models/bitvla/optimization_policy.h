#pragma once

#include "models/ggml_backend.h"

namespace robotcpp::bitvla {

// Selection only. Operators retain their dtype/shape/layout checks and fallbacks.
struct OptimizationPolicy {
    bool vision_rgb_lookup_in_graph = false;
    bool vision_fused_layer_norm_i2s = false;
    bool vision_fused_layer_norm_affine = false;
    bool vision_fused_attention_i2s = false;
    bool vision_fused_gelu_i2s = false;
    bool llm_first_rms_cpu_order = false;
    bool llm_fused_rms_i2s = false;
    bool llm_shared_projection_encoding = false;
    bool llm_last_layer_action_rows = false;
    bool llm_flash_attention = false;
    bool llm_fused_attention_rms_i2s = false;
    bool llm_fused_gated_rms_i2s = false;
    bool llm_fused_gated_rms = false;
};

OptimizationPolicy make_optimization_policy(backend_mode backend);

} // namespace robotcpp::bitvla
