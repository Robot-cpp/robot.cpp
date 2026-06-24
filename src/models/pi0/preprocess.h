#pragma once

#include "models/pi0/types.h"

#include <string>

namespace robotcpp::pi0 {

bool validate_and_preprocess_pi0(const Pi0ModelConfig & config, const Pi0RawObservation & raw, Pi0Observation & out);

} // namespace robotcpp::pi0
