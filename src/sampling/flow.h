#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace robotcpp::sampling {

using VelocityFn = std::function<void(float time, const std::vector<float> & x, std::vector<float> & v)>;

void sample_flow_euler(
    int steps,
    std::mt19937 & rng,
    int horizon,
    int action_dim,
    const std::vector<float> * initial_noise,
    const VelocityFn & velocity,
    std::vector<float> & out);

} // namespace robotcpp::sampling
