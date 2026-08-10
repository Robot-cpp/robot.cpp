#include "models/starvla/oft_prompt.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

robotcpp::starvla::OFTPromptConfig official_config() {
    robotcpp::starvla::OFTPromptConfig config;
    config.horizon = 16;
    config.action_token = "\xF0\x9F\x94\x8D";
    config.action_suffix = " Please predict the next 16 robot actions: <action>";
    for (int i = 0; i < config.horizon; ++i) {
        config.action_suffix += config.action_token;
    }
    config.action_suffix += "<action>.";
    config.cot_enabled = true;
    config.cot_template = "Your task is {instruction}. Locate {instruction}.";
    config.state_bins = 256;
    config.state_bin_min = -1.0f;
    config.state_bin_max = 1.0f;
    config.state_clip = false;
    return config;
}

} // namespace

int main() {
    using namespace robotcpp::starvla;

    OFTPromptConfig config = official_config();
    std::string error;
    require(validate_oft_prompt_config(config, error), "official prompt config must validate");

    config.cot_enabled = false;
    std::string instruction;
    require(build_oft_instruction(config, "grab", {}, instruction, error),
            "prompt without state must build");
    require(instruction == "grab" + config.action_suffix, "action suffix placement must match StarVLA");
    require(!build_oft_instruction(config, "grab <__media__> now", {}, instruction, error),
            "reserved mtmd media markers in tasks must be rejected");
    require(error.find("reserved mtmd media marker") != std::string::npos,
            "media marker rejection must explain the contract violation");

    require(build_oft_instruction(config, "grab", {-1.1f, -1.0f, 0.0f, 1.0f, 1.1f},
                                  instruction, error),
            "state prompt must build");
    require(instruction == "grab [STATE] -1 0 128 255 255 [ACTION]" + config.action_suffix,
            "state discretization must match numpy.digitize");

    config.cot_enabled = true;
    require(build_oft_instruction(config, "grab", {}, instruction, error), "CoT prompt must build");
    const std::string unwrapped = "grab" + config.action_suffix;
    require(instruction == "Your task is " + unwrapped + ". Locate " + unwrapped + ".",
            "Python str.replace semantics must replace every instruction placeholder");

    const std::string content = build_qwen_media_content(2, instruction, "<__media__>");
    require(content == "<__media__><__media__>" + instruction,
            "image markers must precede text without separators");

    std::vector<int32_t> ids = {1, 9, 2, 9, 9, 3, 9};
    std::vector<size_t> positions;
    require(find_last_token_positions(ids, 9, 3, positions, error), "action tokens must be found");
    require(positions == std::vector<size_t>({3, 4, 6}),
            "last action positions must remain in temporal order");
    require(!find_last_token_positions(ids, 9, 5, positions, error),
            "insufficient action tokens must fail");
    require(!error.empty(), "insufficient action tokens must return an error");

    std::cout << "starvla OFT prompt tests passed\n";
    return 0;
}
