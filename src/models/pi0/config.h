#pragma once

#include "models/pi0/types.h"

namespace robotcpp::pi0 {

inline Pi0Config & ensure_pi0_config(Pi0ModelConfig & config) {
    return config.pi0;
}

inline const Pi0Config & pi0_config(const Pi0ModelConfig & config) {
    return config.pi0;
}

} // namespace robotcpp::pi0
