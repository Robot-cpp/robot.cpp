#pragma once

#include "models/starvla/normalization.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::starvla {

struct PIPolicyConfig {
    std::string backbone_arch;
    std::string bundle_uuid;
    std::string text_filename;
    std::string mmproj_filename;

    int qwen_hidden_dim = 0;
    int qwen_input_embedding_dim = 0;
    int qwen_layer_count = 0;
    int qwen_vocab_size = 0;
    std::string cot_template;
    std::vector<int32_t> qwen_hidden_tuple_indices;
    int image_count = 0;
    std::vector<std::string> image_names;
    int image_framework_inference_pre_resize_width = 0;
    int image_framework_inference_pre_resize_height = 0;
    int image_processor_min_pixels = 0;
    int image_processor_max_pixels = 0;
    int image_patch_size = 0;
    int image_spatial_merge_size = 0;
    int image_min_token_count = 0;
    int image_max_token_count = 0;

    int dit_width = 0;
    int block_count = 0;
    int attention_head_count = 0;
    int attention_head_dim = 0;
    int cross_attention_dim = 0;
    int feed_forward_dim = 0;
    int mlp_hidden_dim = 0;
    int state_dim = 0;
    int action_dim = 0;
    int horizon = 0;
    int state_token_count = 0;
    int future_token_count = 0;
    int action_position_count = 0;
    int timestep_projection_dim = 0;
    int num_inference_timesteps = 0;
    float ada_norm_epsilon = 0.0f;
    float euler_dt = 0.0f;
    std::vector<int32_t> timestep_ids;
    NormalizationConfig normalization;
};

class PIPolicy {
  public:
    ~PIPolicy();

    PIPolicy(const PIPolicy &) = delete;
    PIPolicy & operator=(const PIPolicy &) = delete;

    static std::unique_ptr<PIPolicy> load(const std::string & path, int n_threads,
                                          int verbosity, std::string & error);

    // qwen_hidden_states is layer-major
    // [block_count, token_count, qwen_hidden_dim]. The legacy released
    // implementation did not forward the Qwen attention mask into the policy
    // head. state is either omitted (the official Bridge deployment path) or
    // one token [state_dim], and initial_noise is token-major
    // [horizon, action_dim].
    bool evaluate(const float * qwen_hidden_states, size_t hidden_element_count,
                  const float * state, size_t state_element_count,
                  const float * initial_noise, size_t noise_element_count,
                  std::vector<float> & normalized_actions, std::string & error);
    bool unnormalize(const std::vector<float> & normalized_actions,
                     const std::string & profile_key, std::vector<float> & actions,
                     std::string & error) const;

    const PIPolicyConfig & config() const;
    const char * backend_name() const;
    size_t graph_build_count() const;

  private:
    struct Impl;

    explicit PIPolicy(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
