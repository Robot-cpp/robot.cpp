#include "optimization_policy.h"

namespace robotcpp::bitvla {

OptimizationPolicy make_optimization_policy(backend_mode backend) {
    OptimizationPolicy policy;
    switch (backend) {
    case backend_mode::cuda:
        // Capability guards remain at their call sites; enabling a policy
        // does not assert kernel availability.
        policy.vision_rgb_lookup_in_graph = true;
        // Includes the existing guarded joint QKV/layout projection attempts.
        policy.vision_fused_layer_norm_i2s = true;
        policy.vision_fused_layer_norm_affine = true;
        policy.vision_fused_attention_i2s = true;
        policy.vision_fused_gelu_i2s = true;
        policy.llm_first_rms_cpu_order = true;
        // Includes the existing guarded joint QKV/KV and Gate/Up projections.
        policy.llm_fused_rms_i2s = true;
        policy.llm_shared_projection_encoding = true;
        // Both the query crop and its existing attention-output fallback crop.
        policy.llm_last_layer_action_rows = true;
        policy.llm_flash_attention = true;
        policy.llm_fused_attention_rms_i2s = true;
        policy.llm_fused_gated_rms_i2s = true;
        policy.llm_fused_gated_rms = true;
        break;
    case backend_mode::cpu:
        policy.llm_shared_projection_encoding = true;
        // Reuse CUDA's guarded last-layer query56 and attention-output56 paths.
        policy.llm_last_layer_action_rows = true;
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
        policy.llm_fused_gated_rms = true;
        policy.llm_fused_gated_rms_i2s = true;
#endif
        break;
    case backend_mode::metal:
        // Preserve Clean241's existing paths. Shared, unconditional operations
        // remain shared; these flags do not describe every operation in a graph.
        break;
    }
    return policy;
}

} // namespace robotcpp::bitvla
