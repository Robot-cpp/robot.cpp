#pragma once

#include "models/pi0/pi0_context.h"
#include "models/pi0/types.h"

#include <vector>

namespace robotcpp::pi0 {

class Pi0Sampler {
public:
    explicit Pi0Sampler(const Pi0Context & ctx);

    void sample_actions(
        Pi0RuntimeConfig & runtime,
        const std::vector<float> & state_context,
        const Pi0KvCache & cache,
        const std::vector<float> & initial_noise,
        std::vector<float> & out_actions) const;

private:
    void denormalize_actions(std::vector<float> & actions) const;

    const Pi0Context & ctx_;
};

bool pi0_has_action_head(const Pi0Context & ctx);
void pi0_state_context(const Pi0Context & ctx, const std::vector<float> & state, std::vector<float> & out);
void pi0_velocity_batch(
    const Pi0Context & ctx,
    float time,
    const std::vector<float> & actions,
    const std::vector<float> & state_context,
    std::vector<float> & out);

} // namespace robotcpp::pi0
