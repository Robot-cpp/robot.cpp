#pragma once

#include "models/starvla/normalization.h"
#include "models/starvla/oft_prompt.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::starvla {

struct OFTPolicyConfig {
    std::string backbone_arch;
    std::string bundle_uuid;
    std::string text_filename;
    std::string mmproj_filename;
    int input_dim = 0;
    int input_embedding_dim = 0;
    int vocab_size = 0;
    int hidden_dim = 0;
    int block_count = 0;
    int action_dim = 0;
    int horizon = 0;
    float layer_norm_epsilon = 0.0f;
    OFTPromptConfig prompt;
    int action_token_id = 0;
    int image_count = 0;
    std::vector<std::string> image_names;
    int image_processor_min_pixels = 0;
    int image_processor_max_pixels = 0;
    int image_patch_size = 0;
    int image_spatial_merge_size = 0;
    int image_min_token_count = 0;
    int image_max_token_count = 0;
    NormalizationConfig normalization;
};

class OFTPolicy {
  public:
    ~OFTPolicy();

    OFTPolicy(const OFTPolicy &) = delete;
    OFTPolicy & operator=(const OFTPolicy &) = delete;

    static std::unique_ptr<OFTPolicy> load(const std::string & path, int n_threads, int verbosity,
                                           std::string & error);

    bool evaluate(const float * action_queries, size_t element_count, std::vector<float> & normalized_actions,
                  std::string & error);
    bool unnormalize(const std::vector<float> & normalized_actions, const std::string & profile_key,
                     std::vector<float> & actions, std::string & error) const;

    const OFTPolicyConfig & config() const;
    const char * backend_name() const;

  private:
    struct Impl;

    explicit OFTPolicy(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace robotcpp::starvla
