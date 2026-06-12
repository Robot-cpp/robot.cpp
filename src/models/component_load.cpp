#include "models/component_load.h"

#include "core/error.h"

#include <fstream>

namespace vlacpp {

namespace {

vlacpp_status require_path(const std::string & path, const char * role) {
    if (path.empty()) {
        return fail(VLACPP_STATUS_INVALID_ARGUMENT, std::string("missing model artifact path: ") + role);
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(VLACPP_STATUS_IO_ERROR, std::string("failed to open model artifact ") + role + ": " + path);
    }
    return VLACPP_STATUS_OK;
}

} // namespace

vlacpp_status build_artifact_path_map(const ModelArtifacts & artifacts, ArtifactPathMap & out) {
    out.clear();
    if (artifacts.empty()) {
        return fail(VLACPP_STATUS_INVALID_ARGUMENT, "model artifacts are required");
    }
    for (const ModelArtifact & artifact : artifacts) {
        if (artifact.role.empty() || artifact.path.empty()) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "model artifact role and path are required");
        }
        if (!out.emplace(artifact.role, artifact.path).second) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "duplicate model artifact role: " + artifact.role);
        }
    }
    return VLACPP_STATUS_OK;
}

vlacpp_status require_artifact_path(const ArtifactPathMap & artifacts, const char * role, std::string & out) {
    auto it = artifacts.find(role);
    if (it == artifacts.end()) {
        return fail(VLACPP_STATUS_INVALID_ARGUMENT, std::string("missing model artifact role: ") + role);
    }
    out = it->second;
    return VLACPP_STATUS_OK;
}

vlacpp_status load_component_gguf_model(
    const std::string & path,
    const char * model_type,
    const char * role,
    GgufTensorLoadFilter filter,
    GgufMetadataHandler metadata_handler,
    ModelConfig & config,
    TensorMap & tensors) {
    vlacpp_status status = require_path(path, role);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = load_gguf_model_file(path, config, tensors, filter, metadata_handler);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    if (config.common.model_type != model_type) {
        return fail(
            VLACPP_STATUS_UNSUPPORTED,
            std::string("unsupported component model_type: expected ") + model_type + ", got " + config.common.model_type);
    }
    if (config.component_role != role) {
        const std::string actual = config.component_role.empty() ? "<missing>" : config.component_role;
        return fail(
            VLACPP_STATUS_PARSE_ERROR,
            std::string("component role mismatch: expected ") + role + ", got " + actual);
    }
    return VLACPP_STATUS_OK;
}

vlacpp_status require_matching_common_config(
    const ModelConfig & base,
    const ModelConfig & current,
    const char * role,
    const char * model_type) {
    const CommonModelConfig & lhs = base.common;
    const CommonModelConfig & rhs = current.common;
    if (lhs.model_type != rhs.model_type ||
        lhs.image_width != rhs.image_width ||
        lhs.image_height != rhs.image_height ||
        lhs.state_dim != rhs.state_dim ||
        lhs.action_dim != rhs.action_dim ||
        lhs.action_horizon != rhs.action_horizon ||
        lhs.max_token_len != rhs.max_token_len ||
        lhs.image_keys != rhs.image_keys ||
        lhs.state_mean != rhs.state_mean ||
        lhs.state_std != rhs.state_std ||
        lhs.action_mean != rhs.action_mean ||
        lhs.action_std != rhs.action_std) {
        return fail(VLACPP_STATUS_PARSE_ERROR, std::string(model_type) + " component config mismatch: " + role);
    }
    return VLACPP_STATUS_OK;
}

bool is_component_dtype(const std::string & value) {
    return value == "preserve" || value == "fp32" || value == "f16" || value == "bf16";
}

} // namespace vlacpp
