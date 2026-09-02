#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace robotcpp::starvla {

struct OFTPromptConfig {
    int horizon = 0;
    std::string action_token;
    std::string action_suffix;
    bool cot_enabled = false;
    std::string cot_template;
    int state_bins      = 0;
    float state_bin_min = 0.0f;
    float state_bin_max = 0.0f;
    bool state_clip     = false;
};

bool validate_oft_prompt_config(const OFTPromptConfig & config, std::string & error);

bool build_oft_instruction(const OFTPromptConfig & config, const std::string & task, const std::vector<float> & state,
                           std::string & instruction, std::string & error);

std::string build_qwen_media_content(size_t image_count, const std::string & instruction, const char * media_marker);

bool find_last_token_positions(const std::vector<int32_t> & token_ids, int32_t token_id, size_t count,
                               std::vector<size_t> & positions, std::string & error);

} // namespace robotcpp::starvla
