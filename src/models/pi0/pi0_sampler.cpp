#include "models/pi0/pi0_sampler.h"

#include "models/pi0/pi0_action_decoder.h"
#include "sampling/flow.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace vlacpp {

Pi0Sampler::Pi0Sampler(const ModelConfig & config, const Pi0ActionDecoder & action_decoder)
    : config_(config),
      action_decoder_(action_decoder) {}

void Pi0Sampler::sample_actions(
    RuntimeConfig & runtime,
    const std::vector<float> & state_context,
    const KvCache & cache,
    const std::vector<float> & initial_noise,
    std::vector<float> & out_actions) const {
    sample_flow_euler(
        runtime.flow_steps,
        runtime.rng,
        config_.action_horizon,
        config_.action_dim,
        initial_noise.empty() ? nullptr : &initial_noise,
        [&](float time, const std::vector<float> & x, std::vector<float> & v) {
            std::vector<float> action_velocity;
            action_decoder_.velocity_batch(
                time,
                x,
                state_context,
                cache.prefix_layers,
                cache.token_count,
                action_velocity);
            if (action_velocity.size() != v.size()) {
                throw std::invalid_argument("pi0 action velocity has incompatible shape");
            }
            std::copy(action_velocity.begin(), action_velocity.end(), v.begin());
        },
        out_actions);

    denormalize_actions(out_actions);
}

void Pi0Sampler::denormalize_actions(std::vector<float> & actions) const {
    if (config_.action_mean.size() != static_cast<size_t>(config_.action_dim) ||
        config_.action_std.size() != static_cast<size_t>(config_.action_dim)) {
        return;
    }
    for (size_t i = 0; i < actions.size(); ++i) {
        const size_t col = i % static_cast<size_t>(config_.action_dim);
        actions[i] = actions[i] * config_.action_std[col] + config_.action_mean[col];
    }
}

} // namespace vlacpp
