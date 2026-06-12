#pragma once

#include "vlacpp.h"

#include <any>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace vlacpp {

struct ComponentRuntimeConfig {
    std::string backend = "inherit";
    std::string data_type = "preserve";
    int n_threads = 0;
};

struct VLAComponentConfig {
    std::string role;
    std::string architecture;
    std::string tensor_prefix;
    ComponentRuntimeConfig runtime;
};

struct CommonModelConfig {
    std::string model_type;
    int image_width = 224;
    int image_height = 224;
    int state_dim = 0;
    int action_dim = 0;
    int action_horizon = 0;
    int max_token_len = 250;
    std::vector<std::string> image_keys;
    std::vector<float> state_mean;
    std::vector<float> state_std;
    std::vector<float> action_mean;
    std::vector<float> action_std;
};

struct ModelConfig {
    std::string component_role;
    CommonModelConfig common;
    std::any specific;
};

struct Tensor {
    // GGUF/ggml dimension order: shape[0] is ne0, shape[1] is ne1, etc.
    std::vector<int64_t> shape;
    std::vector<float> data;
    std::string data_type = "fp32";
};

using TensorMap = std::map<std::string, Tensor>;

struct ImageTensor {
    std::string name;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;
};

struct ObservationData {
    std::vector<ImageTensor> images;
    std::vector<float> state;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    std::vector<float> noise;
};

struct KvCache {
    bool prefix_valid = false;
    size_t token_count = 0;
    uint64_t prefix_generation = 0;

    void reset() {
        prefix_valid = false;
        token_count = 0;
        ++prefix_generation;
    }
};

struct BackendConfig {
    vlacpp_backend backend = VLACPP_BACKEND_CPU;
    int n_threads = 0;
    std::map<std::string, std::string> component_dtype_overrides;
};

struct RuntimeConfig {
    uint32_t seed = 0;
    int flow_steps = 10;
    std::mt19937 rng;
    vlacpp_infer_timings last_timings = {};
};

class RuntimeModel {
public:
    virtual ~RuntimeModel() = default;
    virtual const ModelConfig & config() const = 0;
    virtual vlacpp_status reset_cache(KvCache & cache) = 0;
    virtual vlacpp_status infer(
        KvCache & cache,
        RuntimeConfig & runtime,
        const ObservationData & observation,
        std::vector<float> & out_actions) = 0;
};

} // namespace vlacpp
