#pragma once

#include "core/gguf.h"
#include "models/model.h"

#include <map>
#include <string>

namespace vlacpp {

using ArtifactPathMap = std::map<std::string, std::string>;

vlacpp_status build_artifact_path_map(const ModelArtifacts & artifacts, ArtifactPathMap & out);
vlacpp_status require_artifact_path(const ArtifactPathMap & artifacts, const char * role, std::string & out);
vlacpp_status load_component_gguf_model(
    const std::string & path,
    const char * model_type,
    const char * role,
    GgufTensorLoadFilter filter,
    GgufMetadataHandler metadata_handler,
    ModelConfig & config,
    TensorMap & tensors);
vlacpp_status require_matching_common_config(
    const ModelConfig & base,
    const ModelConfig & current,
    const char * role,
    const char * model_type);
bool is_component_dtype(const std::string & value);

} // namespace vlacpp
