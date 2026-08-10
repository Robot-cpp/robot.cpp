#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace robotcpp::starvla {

struct NormalizationProfile {
    std::string key;
    std::vector<float> action_q01;
    std::vector<float> action_q99;
    std::vector<uint8_t> action_mask;
};

struct NormalizationConfig {
    bool clip_actions = false;
    float binary_threshold = 0.5f;
    std::string binary_comparison;
    std::vector<int32_t> continuous_dimensions;
    std::vector<int32_t> binary_dimensions;
    std::vector<NormalizationProfile> profiles;
};

bool validate_normalization_config(const NormalizationConfig & config, int action_dim, std::string & error);

const NormalizationProfile * resolve_normalization_profile(const NormalizationConfig & config,
                                                           const std::string & profile_key, std::string & error);

bool denormalize_actions(const NormalizationConfig & config, const std::string & profile_key,
                         const std::vector<float> & normalized, int horizon, int action_dim,
                         std::vector<float> & actions, std::string & error);

} // namespace robotcpp::starvla
