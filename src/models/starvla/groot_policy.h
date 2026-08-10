#pragma once

#include "models/starvla/normalization.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::starvla {

struct GR00TPolicyConfig {
    std::string backbone_arch;
    std::string bundle_uuid;
    std::string text_filename;
    std::string mmproj_filename;
    int qwen_hidden_dim = 0;
    int qwen_input_embedding_dim = 0;
    int qwen_vocab_size = 0;
    std::string cot_template;
    int image_count = 0;
    std::vector<std::string> image_names;
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
    int output_dim = 0;
    int mlp_hidden_dim = 0;
    int action_dim = 0;
    int horizon = 0;
    int future_token_count = 0;
    int action_position_count = 0;
    int no_state_sequence_length = 0;
    int timestep_projection_dim = 0;
    float ada_norm_epsilon = 0.0f;
    float output_norm_epsilon = 0.0f;
    float euler_dt = 0.0f;
    std::vector<int32_t> timestep_ids;
    NormalizationConfig normalization;
};

class GR00TPolicy {
  public:
    ~GR00TPolicy();

    GR00TPolicy(const GR00TPolicy &) = delete;
    GR00TPolicy & operator=(const GR00TPolicy &) = delete;

    static std::unique_ptr<GR00TPolicy> load(const std::string & path, int n_threads, int verbosity,
                                             std::string & error);

    // qwen_hidden_states is token-major [token_count, qwen_hidden_dim]. The mask
    // follows torch SDPA semantics: non-zero entries participate in attention.
    // initial_noise is token-major [horizon, action_dim].
    bool evaluate(const float * qwen_hidden_states, size_t hidden_element_count,
                  const uint8_t * qwen_attention_mask, size_t mask_element_count,
                  const float * initial_noise, size_t noise_element_count,
                  std::vector<float> & normalized_actions, std::string & error);
    bool unnormalize(const std::vector<float> & normalized_actions, const std::string & profile_key,
                     std::vector<float> & actions, std::string & error) const;

    const GR00TPolicyConfig & config() const;
    const char * backend_name() const;

  private:
    struct Impl;

    explicit GR00TPolicy(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
