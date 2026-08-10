#include "models/starvla/groot_prompt.h"

#include <utility>

namespace robotcpp::starvla {
namespace {

constexpr const char * kInstructionPlaceholder = "{instruction}";
constexpr const char * kMtmdMediaMarker        = "<__media__>";

bool contains_nul(const std::string & value) {
    return value.find('\0') != std::string::npos;
}

void replace_all(std::string & value, const std::string & needle, const std::string & replacement) {
    size_t offset = 0;
    while ((offset = value.find(needle, offset)) != std::string::npos) {
        value.replace(offset, needle.size(), replacement);
        offset += replacement.size();
    }
}

bool build_instruction(const char * framework, const std::string & cot_template, const std::string & task,
                       std::string & instruction, std::string & error) {
    instruction.clear();
    error.clear();

    if (task.empty()) {
        error = std::string("StarVLA ") + framework + " task must not be empty";
        return false;
    }
    if (contains_nul(task) || contains_nul(cot_template)) {
        error = std::string("StarVLA ") + framework + " prompt contains an embedded NUL byte";
        return false;
    }
    if (task.find(kMtmdMediaMarker) != std::string::npos || cot_template.find(kMtmdMediaMarker) != std::string::npos) {
        error = std::string("StarVLA ") + framework + " prompt contains the reserved mtmd media marker";
        return false;
    }
    if (cot_template.find(kInstructionPlaceholder) == std::string::npos) {
        error = std::string("StarVLA ") + framework + " CoT template is missing {instruction}";
        return false;
    }

    std::string wrapped = cot_template;
    replace_all(wrapped, kInstructionPlaceholder, task);
    instruction = std::move(wrapped);
    return true;
}

} // namespace

bool build_groot_instruction(const std::string & cot_template, const std::string & task, std::string & instruction,
                             std::string & error) {
    return build_instruction("GR00T", cot_template, task, instruction, error);
}

bool build_pi_v3_instruction(const std::string & cot_template, const std::string & task, std::string & instruction,
                             std::string & error) {
    return build_instruction("PI_v3", cot_template, task, instruction, error);
}

bool build_fast_instruction(const std::string & cot_template, const std::string & task, std::string & instruction,
                            std::string & error) {
    return build_instruction("FAST", cot_template, task, instruction, error);
}

} // namespace robotcpp::starvla
