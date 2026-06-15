#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace robotcpp::pi0 {

struct Pi0InferTimings {
    double preprocess_ms = 0.0;
    double prefix_ms = 0.0;
    double state_ms = 0.0;
    double denoise_ms = 0.0;
    double output_ms = 0.0;
    double total_ms = 0.0;
};

struct Pi0RawImageView {
    const char * name = nullptr;
    const std::uint8_t * data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int stride_bytes = 0;
};

struct Pi0RawObservation {
    const Pi0RawImageView * images = nullptr;
    std::size_t image_count = 0;
    const float * state = nullptr;
    std::size_t state_count = 0;
    const char * prompt = nullptr;
    const std::int32_t * prompt_tokens = nullptr;
    std::size_t prompt_token_count = 0;
    const float * noise = nullptr;
    std::size_t noise_count = 0;
};

struct Pi0ComponentRuntimeConfig {
    std::string backend = "inherit";
    std::string data_type = "preserve";
    int n_threads = 0;
};

struct Pi0ComponentConfig {
    std::string role;
    std::string architecture;
    std::string tensor_prefix;
    Pi0ComponentRuntimeConfig runtime;
};

struct Pi0VisionConfig {
    Pi0ComponentConfig component = {"vit", "pi0-vit", "pi0.vit.", {}};
    int width = 0;
    int patch_height = 0;
    int patch_width = 0;
    int layers = 0;
    int heads = 0;
    float norm_epsilon = 0.0f;
};

struct Pi0LlmConfig {
    Pi0ComponentConfig component = {"llm", "gemma", "pi0.llm.", {}};
    int width = 0;
    int q_out = 0;
    int kv_out = 0;
    int mlp_width = 0;
    int layers = 0;
};

struct Pi0MmprojConfig {
    Pi0ComponentConfig component = {"mmproj", "pi0-mmproj", "pi0.merger.", {}};
};

struct Pi0ActionConfig {
    Pi0ComponentConfig component = {"action_decoder", "pi0-action-decoder", "pi0.action_decoder.", {}};
    int width = 0;
    int expert_width = 0;
    int expert_q_out = 0;
    int expert_kv_out = 0;
    int expert_mlp_width = 0;
    int expert_layers = 0;
};

struct Pi0StateConfig {
    Pi0ComponentConfig component = {"state", "pi0-state-proj", "pi0.action_decoder.state_proj.", {}};
};

struct Pi0Config {
    Pi0VisionConfig vision;
    Pi0MmprojConfig mmproj;
    Pi0LlmConfig llm;
    Pi0ComponentConfig tokenizer = {"tokenizer", "sentencepiece", "", {}};
    Pi0ActionConfig action;
    Pi0StateConfig state;
};

struct Pi0CommonConfig {
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

struct Pi0ModelConfig {
    std::string component_role;
    Pi0CommonConfig common;
    Pi0Config pi0;
};

inline bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

struct Pi0ImageTensor {
    std::string name;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;
};

struct Pi0Observation {
    std::vector<Pi0ImageTensor> images;
    std::vector<float> state;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    std::vector<float> noise;
};

struct Pi0BackendConfig {
    bool use_accel = true;
    int n_threads = 0;
};

struct Pi0RuntimeConfig {
    uint32_t seed = 0;
    int flow_steps = 10;
    std::mt19937 rng;
    Pi0InferTimings last_timings = {};
};

} // namespace robotcpp::pi0
