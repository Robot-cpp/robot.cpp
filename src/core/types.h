#pragma once

#include "vlacpp.h"

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vlacpp {

struct ModelConfig {
    std::string source_path;
    std::string model_type = "pi0";
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
    int openpi_action_width = 0;
    int openpi_vision_width = 0;
    int openpi_vision_patch_height = 0;
    int openpi_vision_patch_width = 0;
    int openpi_vision_layers = 0;
    int openpi_language_width = 0;
    int openpi_language_q_out = 0;
    int openpi_language_kv_out = 0;
    int openpi_language_mlp_width = 0;
    int openpi_language_layers = 0;
    int openpi_action_expert_width = 0;
    int openpi_action_expert_q_out = 0;
    int openpi_action_expert_kv_out = 0;
    int openpi_action_expert_mlp_width = 0;
    int openpi_action_expert_layers = 0;
    bool openpi_full_weights_present = false;
};

struct Tensor {
    // GGUF/ggml dimension order: shape[0] is ne0, shape[1] is ne1, etc.
    std::vector<int64_t> shape;
    std::vector<float> data;
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

struct PrefixLayerKv {
    std::vector<float> k;
    std::vector<float> v;
    uint64_t generation = 0;
    size_t k_size = 0;
    size_t v_size = 0;
    bool device_cached = false;
};

struct KvCache {
    bool prefix_valid = false;
    size_t token_count = 0;
    uint64_t prefix_generation = 0;
    std::vector<PrefixLayerKv> prefix_layers;

    void reset() {
        prefix_valid = false;
        token_count = 0;
        ++prefix_generation;
        prefix_layers.clear();
    }
};

struct BackendConfig {
    vlacpp_backend backend = VLACPP_BACKEND_CPU;
    int n_threads = 0;
};

struct RuntimeConfig {
    uint32_t seed = 0;
    int flow_steps = 10;
    std::mt19937 rng;
};

class Model {
public:
    virtual ~Model() = default;
    virtual const ModelConfig & config() const = 0;
    virtual const char * capability() const = 0;
    virtual vlacpp_status reset_cache(KvCache & cache) = 0;
    virtual vlacpp_status infer(
        KvCache & cache,
        RuntimeConfig & runtime,
        const ObservationData & observation,
        std::vector<float> & out_actions) = 0;
};

} // namespace vlacpp
