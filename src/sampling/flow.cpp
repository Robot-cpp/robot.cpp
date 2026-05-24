#include "sampling/flow.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace vlacpp {

void sample_flow_euler(
    int steps,
    std::mt19937 & rng,
    int horizon,
    int action_dim,
    const std::vector<float> * initial_noise,
    const VelocityFn & velocity,
    std::vector<float> & out) {
    const int n_steps = std::max(1, steps);
    const size_t n = static_cast<size_t>(horizon) * action_dim;
    out.resize(n);

    if (initial_noise != nullptr && initial_noise->size() == n) {
        out = *initial_noise;
    } else {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float & value : out) {
            value = normal(rng);
        }
    }

    std::vector<float> v(n, 0.0f);
    float time = 1.0f;
    const float dt = -1.0f / static_cast<float>(n_steps);
    for (int i = 0; i < n_steps; ++i) {
        velocity(time, out, v);
        for (size_t j = 0; j < n; ++j) {
            out[j] += dt * v[j];
        }
        time += dt;
    }
}

} // namespace vlacpp
