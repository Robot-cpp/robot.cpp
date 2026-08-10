#pragma once

#include <string>

namespace robotcpp::starvla {

bool build_groot_instruction(const std::string & cot_template, const std::string & task,
                             std::string & instruction, std::string & error);

bool build_pi_v3_instruction(const std::string & cot_template, const std::string & task,
                             std::string & instruction, std::string & error);

bool build_fast_instruction(const std::string & cot_template,
                            const std::string & task,
                            std::string & instruction, std::string & error);

} // namespace robotcpp::starvla
