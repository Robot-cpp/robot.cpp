#include "models/starvla/groot_prompt.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void require_rejected(const std::string & cot_template, const std::string & task,
                      const char * message) {
    std::string instruction = "stale instruction";
    std::string error = "stale error";
    require(!robotcpp::starvla::build_groot_instruction(cot_template, task, instruction, error),
            message);
    require(instruction.empty(), "rejected prompts must clear their instruction output");
    require(!error.empty(), "rejected prompts must explain the contract violation");
}

} // namespace

int main() {
    using robotcpp::starvla::build_groot_instruction;
    using robotcpp::starvla::build_pi_v3_instruction;

    const std::string official_template =
        "Your task is {instruction}. To identify the key objects for your task. "
        "Locate their bounding boxes in [x1,y1,x2,y2] format.";

    std::string instruction;
    std::string error = "stale error";
    require(build_groot_instruction(official_template, "grab the block", instruction, error),
            "official GR00T prompt must build");
    require(instruction ==
                "Your task is grab the block. To identify the key objects for your task. "
                "Locate their bounding boxes in [x1,y1,x2,y2] format.",
            "official GR00T prompt must preserve the checkpoint template exactly");
    require(error.empty(), "successful prompt construction must clear stale errors");

    require(build_pi_v3_instruction(official_template, "grab the block", instruction, error),
            "official PI_v3 prompt must build");
    require(instruction ==
                "Your task is grab the block. To identify the key objects for your task. "
                "Locate their bounding boxes in [x1,y1,x2,y2] format.",
            "PI_v3 must use the same pinned CoT replacement contract");
    require(!build_pi_v3_instruction(official_template, "", instruction, error),
            "empty PI_v3 tasks must be rejected");
    require(error.find("PI_v3") != std::string::npos,
            "PI_v3 prompt errors must identify the active framework");

    require(build_groot_instruction(official_template, "grab the block.", instruction, error),
            "punctuated tasks must build");
    require(instruction ==
                "Your task is grab the block.. To identify the key objects for your task. "
                "Locate their bounding boxes in [x1,y1,x2,y2] format.",
            "task punctuation must not be normalized");

    require(build_groot_instruction("First {instruction}; then {instruction}.", "pick", instruction,
                                    error),
            "templates with repeated placeholders must build");
    require(instruction == "First pick; then pick.",
            "every instruction placeholder must be replaced");

    require_rejected(official_template, "", "empty tasks must be rejected");
    require_rejected(official_template, std::string("grab\0now", 8),
                     "embedded NUL bytes in tasks must be rejected");
    require_rejected(std::string("Use {instruction}\0now", 21), "grab",
                     "embedded NUL bytes in templates must be rejected");
    require_rejected(official_template, "grab <__media__> now",
                     "reserved mtmd media markers in tasks must be rejected");
    require_rejected("<__media__>{instruction}", "grab",
                     "reserved mtmd media markers in templates must be rejected");
    require_rejected("Your task is ready.", "grab",
                     "templates without an instruction placeholder must be rejected");
    require_rejected("Your task is { instruction }.", "grab",
                     "lookalike placeholders must not satisfy the template contract");

    std::cout << "starvla GR00T prompt tests passed\n";
    return 0;
}
