#pragma once

#include "models/gguf_loader.h"
#include "models/pi0/component_runtime.h"
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

struct Pi0LoadedComponent {
    gguf_load_result loaded;
    Pi0ComponentRuntime runtime;

    Pi0LoadedComponent() = default;
    ~Pi0LoadedComponent() {
        free_pi0_loaded_component(loaded);
    }

    Pi0LoadedComponent(const Pi0LoadedComponent &) = delete;
    Pi0LoadedComponent & operator=(const Pi0LoadedComponent &) = delete;

    Pi0LoadedComponent(Pi0LoadedComponent && other) noexcept
        : loaded(other.loaded),
          runtime(std::move(other.runtime)) {
        other.loaded = {};
    }

    Pi0LoadedComponent & operator=(Pi0LoadedComponent && other) noexcept {
        if (this != &other) {
            free_pi0_loaded_component(loaded);
            loaded = other.loaded;
            runtime = std::move(other.runtime);
            other.loaded = {};
        }
        return *this;
    }
};

struct Pi0Components {
    Pi0LoadedComponent vit;
    Pi0LoadedComponent mmproj;
    Pi0LoadedComponent llm;
    Pi0LoadedComponent state;
    Pi0LoadedComponent action_decoder;
};

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

} // namespace robotcpp::pi0
