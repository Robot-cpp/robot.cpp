#pragma once

#include "models/pi0/pi0_context.h"
#include "models/pi0/types.h"

#include <vector>

namespace robotcpp::pi0 {

void pi0_state_context(const Pi0Context & ctx, const std::vector<float> & state, std::vector<float> & out);
void pi0_sample_actions(
    const Pi0Context & ctx,
    Pi0RuntimeConfig & runtime,
    const std::vector<float> & state_context,
    const std::vector<float> & initial_noise,
    std::vector<float> & out_actions);

} // namespace robotcpp::pi0
