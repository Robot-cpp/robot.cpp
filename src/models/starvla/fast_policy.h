#pragma once

#include "models/starvla/fast_codec.h"
#include "models/starvla/normalization.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::starvla {

struct FastPolicyConfig {
    std::string backbone_arch;
    std::string bundle_uuid;
    std::string text_filename;
    std::string mmproj_filename;

    int qwen_hidden_dim          = 0;
    int qwen_input_embedding_dim = 0;
    int qwen_vocab_size          = 0;
    int qwen_layer_count         = 0;

    std::string cot_template;
    int action_dim = 0;
    int horizon    = 0;

    int image_count = 0;
    std::vector<std::string> image_names;
    int image_processor_min_pixels = 0;
    int image_processor_max_pixels = 0;
    int image_patch_size           = 0;
    int image_spatial_merge_size   = 0;
    int image_min_token_count      = 0;
    int image_max_token_count      = 0;

    size_t generation_max_length = 0;
    std::vector<int32_t> generation_eos_token_ids;
    int generation_top_k                = 0;
    float generation_repetition_penalty = 0.0f;

    NormalizationConfig normalization;
};

class FastPolicy {
  public:
    ~FastPolicy();

    FastPolicy(const FastPolicy &)             = delete;
    FastPolicy & operator=(const FastPolicy &) = delete;

    static std::unique_ptr<FastPolicy> load(const std::string & path, int verbosity, std::string & error);

    bool decode_generated(const std::vector<int32_t> & full_sequence, std::vector<float> & normalized_actions,
                          std::string & error) const;

    bool unnormalize(const std::vector<float> & normalized_actions, const std::string & profile_key,
                     std::vector<float> & actions, std::string & error) const;

    const FastPolicyConfig & config() const;
    const char * backend_name() const;

  private:
    struct Impl;

    explicit FastPolicy(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
