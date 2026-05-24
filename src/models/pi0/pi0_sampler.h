#pragma once

#include "models/model.h"

#include <vector>

namespace vlacpp {

class Pi0ActionDecoder;

class Pi0Sampler {
public:
    Pi0Sampler(const ModelConfig & config, const Pi0ActionDecoder & action_decoder);

    void sample_actions(
        RuntimeConfig & runtime,
        const std::vector<float> & state_context,
        const KvCache & cache,
        const std::vector<float> & initial_noise,
        std::vector<float> & out_actions) const;

private:
    void denormalize_actions(std::vector<float> & actions) const;

    const ModelConfig & config_;
    const Pi0ActionDecoder & action_decoder_;
};

} // namespace vlacpp
