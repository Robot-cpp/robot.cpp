#pragma once

#include "models/gguf_loader.h"
#include "models/pi0/backend.h"
#include "models/pi0/config.h"

#include <string>
#include <utility>

namespace robotcpp::pi0 {

struct Pi0ComponentPaths {
    std::string vit;
    std::string mmproj;
    std::string llm;
    std::string tokenizer;
    std::string state;
    std::string action_decoder;
};

void free_pi0_loaded_component(gguf_load_result & component);

struct Pi0ComponentRuntime {
    gguf_load_result loaded;
    Pi0ComponentBackend backend;

    Pi0ComponentRuntime() = default;
    ~Pi0ComponentRuntime() {
        free_pi0_loaded_component(loaded);
    }

    Pi0ComponentRuntime(const Pi0ComponentRuntime &) = delete;
    Pi0ComponentRuntime & operator=(const Pi0ComponentRuntime &) = delete;

    Pi0ComponentRuntime(Pi0ComponentRuntime && other) noexcept
        : loaded(other.loaded),
          backend(std::move(other.backend)) {
        other.loaded = {};
    }

    Pi0ComponentRuntime & operator=(Pi0ComponentRuntime && other) noexcept {
        if (this != &other) {
            free_pi0_loaded_component(loaded);
            loaded = other.loaded;
            backend = std::move(other.backend);
            other.loaded = {};
        }
        return *this;
    }
};

struct Pi0Components {
    Pi0ComponentRuntime vit;
    Pi0ComponentRuntime mmproj;
    Pi0ComponentRuntime llm;
    Pi0ComponentRuntime state;
    Pi0ComponentRuntime action_decoder;
};

bool should_load_pi0_tensor(const Pi0ModelConfig & config, const std::string & name);

std::string pi0_action_decoder_tensor(const Pi0ModelConfig & config, const std::string & suffix);
std::string pi0_action_decoder_layer_prefix(const Pi0ModelConfig & config, int layer);
std::string pi0_state_tensor(const Pi0ModelConfig & config, const std::string & suffix);
std::string pi0_llm_tensor(const Pi0ModelConfig & config, const std::string & suffix);
std::string pi0_llm_layer_prefix(const Pi0ModelConfig & config, int layer);
std::string pi0_lm_head(const Pi0ModelConfig & config);
std::string pi0_merger_tensor(const Pi0ModelConfig & config, const std::string & suffix);
std::string pi0_vit_tensor(const Pi0ModelConfig & config, const std::string & suffix);
std::string pi0_vit_layer_prefix(const Pi0ModelConfig & config, int layer);

bool load_pi0_components(
    const Pi0ComponentPaths & paths,
    const Pi0BackendConfig & backend,
    Pi0ModelConfig & out_config,
    Pi0Components & out_components,
    int verbosity);
bool validate_pi0_model_config(Pi0ModelConfig & config, const Pi0Components & components);

} // namespace robotcpp::pi0
