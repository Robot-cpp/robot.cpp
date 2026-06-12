#pragma once

#include "models/pi0/config.h"
#include "models/model.h"

#include <memory>
#include <string>

namespace vlacpp {

struct Pi0ComponentTensors {
    TensorMap vit;
    TensorMap mmproj;
    TensorMap llm;
    TensorMap state;
    TensorMap action_decoder;
};

bool should_load_pi0_tensor(const ModelConfig & config, const std::string & name);

std::string pi0_action_decoder_tensor(const ModelConfig & config, const std::string & suffix);
std::string pi0_action_decoder_layer_prefix(const ModelConfig & config, int layer);
std::string pi0_state_tensor(const ModelConfig & config, const std::string & suffix);
std::string pi0_llm_tensor(const ModelConfig & config, const std::string & suffix);
std::string pi0_llm_layer_prefix(const ModelConfig & config, int layer);
std::string pi0_lm_head(const ModelConfig & config);
std::string pi0_merger_tensor(const ModelConfig & config, const std::string & suffix);
std::string pi0_vit_tensor(const ModelConfig & config, const std::string & suffix);
std::string pi0_vit_layer_prefix(const ModelConfig & config, int layer);

vlacpp_status load_pi0_model_from_artifacts(
    const ModelArtifacts & artifacts,
    const BackendConfig & backend,
    std::unique_ptr<RuntimeModel> & out);
vlacpp_status validate_pi0_model_config(ModelConfig & config, const Pi0ComponentTensors & tensors);
std::unique_ptr<RuntimeModel> make_pi0_model(
    ModelConfig config,
    BackendConfig backend,
    std::string tokenizer_path,
    Pi0ComponentTensors tensors);

} // namespace vlacpp
