#include "models/starvla/oft_prompt.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace robotcpp::starvla {
namespace {

constexpr const char * kInstructionPlaceholder = "{instruction}";
constexpr const char * kMtmdMediaMarker = "<__media__>";

std::string repeat(const std::string & value, int count) {
    std::string result;
    result.reserve(value.size() * static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        result += value;
    }
    return result;
}

void replace_all(std::string & value, const std::string & needle, const std::string & replacement) {
    size_t offset = 0;
    while ((offset = value.find(needle, offset)) != std::string::npos) {
        value.replace(offset, needle.size(), replacement);
        offset += replacement.size();
    }
}

bool discretize_state(const OFTPromptConfig & config, const std::vector<float> & state,
                      std::string & output, std::string & error) {
    if (state.empty()) {
        output.clear();
        return true;
    }

    std::ostringstream stream;
    const double minimum = static_cast<double>(config.state_bin_min);
    const double maximum = static_cast<double>(config.state_bin_max);
    const double step = (maximum - minimum) / static_cast<double>(config.state_bins);
    for (size_t i = 0; i < state.size(); ++i) {
        double value = static_cast<double>(state[i]);
        if (!std::isfinite(value)) {
            error = "StarVLA OFT state contains a non-finite value";
            return false;
        }
        if (config.state_clip) {
            value = std::max(minimum, std::min(maximum, value));
        }

        // Matches numpy.digitize(value, linspace(min, max, bins + 1)[:-1]) - 1.
        int bin = -1;
        for (int edge = 0; edge < config.state_bins; ++edge) {
            const double boundary = minimum + step * static_cast<double>(edge);
            if (value >= boundary) {
                bin = edge;
            } else {
                break;
            }
        }
        if (i != 0) {
            stream << ' ';
        }
        stream << bin;
    }
    output = stream.str();
    return true;
}

} // namespace

bool validate_oft_prompt_config(const OFTPromptConfig & config, std::string & error) {
    error.clear();
    if (config.horizon <= 0 || config.action_token.empty()) {
        error = "StarVLA OFT prompt has an invalid horizon or action token";
        return false;
    }
    const std::string expected_suffix = " Please predict the next " + std::to_string(config.horizon) +
                                        " robot actions: <action>" +
                                        repeat(config.action_token, config.horizon) + "<action>.";
    if (config.action_suffix != expected_suffix) {
        error = "StarVLA OFT action suffix does not match its horizon/token contract";
        return false;
    }
    if (config.cot_enabled && config.cot_template.find(kInstructionPlaceholder) == std::string::npos) {
        error = "StarVLA OFT CoT template is missing {instruction}";
        return false;
    }
    if (config.state_bins <= 0 ||
        !std::isfinite(config.state_bin_min) || !std::isfinite(config.state_bin_max) ||
        config.state_bin_max <= config.state_bin_min) {
        error = "StarVLA OFT state prompt metadata is incompatible";
        return false;
    }
    return true;
}

bool build_oft_instruction(const OFTPromptConfig & config, const std::string & task,
                           const std::vector<float> & state, std::string & instruction,
                           std::string & error) {
    instruction.clear();
    if (!validate_oft_prompt_config(config, error)) {
        return false;
    }
    if (task.find(kMtmdMediaMarker) != std::string::npos) {
        error = "StarVLA OFT task contains the reserved mtmd media marker";
        return false;
    }

    instruction = task;
    if (!state.empty()) {
        std::string state_text;
        if (!discretize_state(config, state, state_text, error)) {
            instruction.clear();
            return false;
        }
        instruction += " [STATE] " + state_text + " [ACTION]";
    }
    instruction += config.action_suffix;

    if (config.cot_enabled) {
        std::string wrapped = config.cot_template;
        replace_all(wrapped, kInstructionPlaceholder, instruction);
        instruction = std::move(wrapped);
    }
    return true;
}

std::string build_qwen_media_content(size_t image_count, const std::string & instruction,
                                     const char * media_marker) {
    const std::string marker = media_marker == nullptr ? std::string() : std::string(media_marker);
    std::string content;
    content.reserve(marker.size() * image_count + instruction.size());
    for (size_t i = 0; i < image_count; ++i) {
        content += marker;
    }
    content += instruction;
    return content;
}

bool find_last_token_positions(const std::vector<int32_t> & token_ids, int32_t token_id,
                               size_t count, std::vector<size_t> & positions,
                               std::string & error) {
    positions.clear();
    error.clear();
    if (count == 0) {
        error = "StarVLA OFT action-token count must be positive";
        return false;
    }
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (token_ids[i] == token_id) {
            positions.push_back(i);
        }
    }
    if (positions.size() < count) {
        error = "StarVLA OFT prompt contains fewer action tokens than its horizon";
        positions.clear();
        return false;
    }
    positions.erase(positions.begin(), positions.end() - static_cast<std::ptrdiff_t>(count));
    return true;
}

} // namespace robotcpp::starvla
