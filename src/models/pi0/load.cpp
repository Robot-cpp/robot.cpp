#include "models/pi0/load.h"

#include "core/error.h"
#include "models/component_load.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace vlacpp {

namespace {

std::string with_prefix(const std::string & prefix, const std::string & suffix) {
    if (suffix.empty()) {
        return prefix;
    }
    if (prefix.empty() || prefix.back() == '.') {
        return prefix + suffix;
    }
    return prefix + "." + suffix;
}

bool parse_pi0_gguf_metadata(
    GgufMetadataReader & reader,
    const std::string & key,
    uint32_t type,
    ModelConfig & config) {
    Pi0Config & pi0 = ensure_pi0_config(config);
    struct ComponentMetadata {
        const char * prefix;
        VLAComponentConfig * component;
    };
    const ComponentMetadata components[] = {
        {"vlacpp.component.vit.", &pi0.vision.component},
        {"vlacpp.component.llm.", &pi0.llm.component},
        {"vlacpp.component.mmproj.", &pi0.mmproj.component},
        {"vlacpp.component.tokenizer.", &pi0.tokenizer},
        {"vlacpp.component.action_decoder.", &pi0.action.component},
        {"vlacpp.component.state.", &pi0.state.component},
    };
    for (const ComponentMetadata & item : components) {
        if (starts_with(key, item.prefix)) {
            return read_gguf_component_metadata(reader, key, type, item.prefix, *item.component);
        }
    }
    if (type == kGgufTypeInt32) {
        struct IntMetadata {
            const char * key;
            int * value;
        };
        const IntMetadata ints[] = {
            {"vlacpp.pi0.action_decoder.width", &pi0.action.width},
            {"vlacpp.pi0.vit.width", &pi0.vision.width},
            {"vlacpp.pi0.vit.patch_height", &pi0.vision.patch_height},
            {"vlacpp.pi0.vit.patch_width", &pi0.vision.patch_width},
            {"vlacpp.pi0.vit.layers", &pi0.vision.layers},
            {"vlacpp.pi0.vit.heads", &pi0.vision.heads},
            {"vlacpp.pi0.llm.width", &pi0.llm.width},
            {"vlacpp.pi0.llm.q_out", &pi0.llm.q_out},
            {"vlacpp.pi0.llm.kv_out", &pi0.llm.kv_out},
            {"vlacpp.pi0.llm.mlp_width", &pi0.llm.mlp_width},
            {"vlacpp.pi0.llm.layers", &pi0.llm.layers},
            {"vlacpp.pi0.action_decoder.expert_width", &pi0.action.expert_width},
            {"vlacpp.pi0.action_decoder.q_out", &pi0.action.expert_q_out},
            {"vlacpp.pi0.action_decoder.kv_out", &pi0.action.expert_kv_out},
            {"vlacpp.pi0.action_decoder.mlp_width", &pi0.action.expert_mlp_width},
            {"vlacpp.pi0.action_decoder.layers", &pi0.action.expert_layers},
        };
        for (const IntMetadata & item : ints) {
            if (key == item.key) {
                return reader.read_scalar(*item.value);
            }
        }
    }
    if (key == "vlacpp.pi0.vit.norm_epsilon" && type == kGgufTypeFloat32) {
        return reader.read_scalar(pi0.vision.norm_epsilon);
    }
    return skip_gguf_metadata_value(reader, type);
}

} // namespace

bool should_load_pi0_tensor(const ModelConfig & config, const std::string & name) {
    const Pi0Config & pi0 = pi0_config(config);
    return starts_with(name, pi0.action.component.tensor_prefix) ||
        starts_with(name, pi0.state.component.tensor_prefix) ||
        starts_with(name, pi0.mmproj.component.tensor_prefix) ||
        starts_with(name, pi0.vision.component.tensor_prefix) ||
        starts_with(name, pi0.llm.component.tensor_prefix);
}

std::string pi0_action_decoder_tensor(const ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).action.component.tensor_prefix, suffix);
}

std::string pi0_action_decoder_layer_prefix(const ModelConfig & config, int layer) {
    return pi0_action_decoder_tensor(config, "layers." + std::to_string(layer) + ".");
}

std::string pi0_state_tensor(const ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).state.component.tensor_prefix, suffix);
}

std::string pi0_llm_tensor(const ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).llm.component.tensor_prefix, suffix);
}

std::string pi0_llm_layer_prefix(const ModelConfig & config, int layer) {
    return pi0_llm_tensor(config, "layers." + std::to_string(layer) + ".");
}

std::string pi0_lm_head(const ModelConfig & config) {
    return pi0_llm_tensor(config, "lm_head.weight");
}

std::string pi0_merger_tensor(const ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).mmproj.component.tensor_prefix, suffix);
}

std::string pi0_vit_tensor(const ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).vision.component.tensor_prefix, suffix);
}

std::string pi0_vit_layer_prefix(const ModelConfig & config, int layer) {
    return pi0_vit_tensor(config, "encoder.layers." + std::to_string(layer) + ".");
}

namespace {

bool has_shape(const TensorMap & tensors, const std::string & name, std::initializer_list<int64_t> expected) {
    const Tensor * tensor = find_tensor(tensors, name);
    if (tensor == nullptr || tensor->shape.size() != expected.size()) {
        return false;
    }
    size_t i = 0;
    for (const int64_t dim : expected) {
        if (tensor->shape[i++] != dim) {
            return false;
        }
    }
    return true;
}

bool has_transformer_stack(
    const TensorMap & tensors,
    const std::string & final_norm_scale_name,
    const std::string & layers_prefix,
    int layers,
    int64_t width,
    int64_t q_out,
    int64_t kv_out,
    int64_t mlp) {
    if (!has_shape(tensors, final_norm_scale_name, {width})) {
        return false;
    }
    for (int layer = 0; layer < layers; ++layer) {
        const std::string base = layers_prefix + std::to_string(layer) + ".";
        if (!has_shape(tensors, base + "input_layernorm.scale", {width}) ||
            !has_shape(tensors, base + "post_attention_layernorm.scale", {width}) ||
            !has_shape(tensors, base + "self_attn.q_proj.weight", {width, q_out}) ||
            !has_shape(tensors, base + "self_attn.k_proj.weight", {width, kv_out}) ||
            !has_shape(tensors, base + "self_attn.v_proj.weight", {width, kv_out}) ||
            !has_shape(tensors, base + "self_attn.o_proj.weight", {q_out, width}) ||
            !has_shape(tensors, base + "mlp.gate_proj.weight", {width, mlp}) ||
            !has_shape(tensors, base + "mlp.up_proj.weight", {width, mlp}) ||
            !has_shape(tensors, base + "mlp.down_proj.weight", {mlp, width})) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace {

bool has_pi0_action_head_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const Tensor * in_w = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_in_proj.weight"));
    const Tensor * in_b = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_in_proj.bias"));
    const Tensor * time_in_w = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_time_mlp_in.weight"));
    const Tensor * time_in_b = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_time_mlp_in.bias"));
    const Tensor * time_out_w = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_time_mlp_out.weight"));
    const Tensor * time_out_b = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_time_mlp_out.bias"));
    const Tensor * out_w = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_out_proj.weight"));
    const Tensor * out_b = find_tensor(tensors, pi0_action_decoder_tensor(config, "action_out_proj.bias"));
    if (in_w == nullptr || in_b == nullptr ||
        time_in_w == nullptr || time_in_b == nullptr ||
        time_out_w == nullptr || time_out_b == nullptr ||
        out_w == nullptr || out_b == nullptr) {
        return false;
    }

    if (in_w->shape.size() != 2 || in_w->shape[0] != config.common.action_dim || in_b->shape.size() != 1) {
        return false;
    }
    const int64_t width = in_w->shape[1];
    return in_b->shape[0] == width &&
        time_in_w->shape.size() == 2 && time_in_w->shape[0] == 2 * width && time_in_w->shape[1] == width &&
        time_in_b->shape.size() == 1 && time_in_b->shape[0] == width &&
        time_out_w->shape.size() == 2 && time_out_w->shape[0] == width && time_out_w->shape[1] == width &&
        time_out_b->shape.size() == 1 && time_out_b->shape[0] == width &&
        out_w->shape.size() == 2 && out_w->shape[0] == width && out_w->shape[1] == config.common.action_dim &&
        out_b->shape.size() == 1 && out_b->shape[0] == config.common.action_dim;
}

bool has_valid_pi0_merger_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const Pi0Config & pi0 = pi0_config(config);
    const Tensor * weight = find_tensor(tensors, pi0_merger_tensor(config, "weight"));
    const Tensor * bias = find_tensor(tensors, pi0_merger_tensor(config, "bias"));
    if (weight == nullptr || bias == nullptr) {
        return false;
    }
    if (pi0.vision.width <= 0 ||
        pi0.llm.width <= 0 ||
        weight->shape.size() != 2 ||
        bias->shape.size() != 1) {
        return false;
    }
    return weight->shape[0] == pi0.vision.width &&
        weight->shape[1] == pi0.llm.width &&
        bias->shape[0] == pi0.llm.width;
}

bool has_valid_pi0_vit_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const Pi0Config & pi0 = pi0_config(config);
    if (pi0.vision.layers <= 0 ||
        pi0.vision.width <= 0 ||
        pi0.vision.patch_height <= 0 ||
        pi0.vision.patch_width <= 0 ||
        pi0.vision.heads <= 0 ||
        pi0.vision.norm_epsilon <= 0.0f) {
        return false;
    }
    const int64_t width = pi0.vision.width;
    const int64_t patch_h = pi0.vision.patch_height;
    const int64_t patch_w = pi0.vision.patch_width;
    const int64_t patches = (config.common.image_width / patch_w) * (config.common.image_height / patch_h);
    if (!has_shape(tensors, pi0_vit_tensor(config, "embeddings.patch_embedding.weight"), {patch_w, patch_h, 3, width}) ||
        !has_shape(tensors, pi0_vit_tensor(config, "embeddings.patch_embedding.bias"), {width}) ||
        !has_shape(tensors, pi0_vit_tensor(config, "embeddings.position_embedding.weight"), {width, patches}) ||
        !has_shape(tensors, pi0_vit_tensor(config, "post_layernorm.weight"), {width}) ||
        !has_shape(tensors, pi0_vit_tensor(config, "post_layernorm.bias"), {width})) {
        return false;
    }
    const int last = pi0.vision.layers - 1;
    for (int layer : {0, last}) {
        const std::string base = pi0_vit_layer_prefix(config, layer);
        if (!has_shape(tensors, base + "layer_norm1.weight", {width}) ||
            !has_shape(tensors, base + "layer_norm1.bias", {width}) ||
            !has_shape(tensors, base + "self_attn.q_proj.weight", {width, width}) ||
            !has_shape(tensors, base + "self_attn.q_proj.bias", {width}) ||
            !has_shape(tensors, base + "self_attn.out_proj.weight", {width, width}) ||
            !has_shape(tensors, base + "self_attn.out_proj.bias", {width}) ||
            !has_shape(tensors, base + "layer_norm2.weight", {width}) ||
            !has_shape(tensors, base + "layer_norm2.bias", {width})) {
            return false;
        }
    }
    return true;
}

bool has_valid_pi0_action_expert_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const Pi0Config & pi0 = pi0_config(config);
    if (pi0.action.expert_layers <= 0 ||
        pi0.action.expert_width <= 0 ||
        pi0.action.expert_q_out <= 0 ||
        pi0.action.expert_kv_out <= 0 ||
        pi0.action.expert_mlp_width <= 0) {
        return false;
    }
    const int64_t width = pi0.action.expert_width;
    const int64_t q_out = pi0.action.expert_q_out;
    const int64_t kv_out = pi0.action.expert_kv_out;
    const int64_t mlp = pi0.action.expert_mlp_width;
    return has_transformer_stack(
        tensors,
        pi0_action_decoder_tensor(config, "norm.scale"),
        pi0_action_decoder_tensor(config, "layers."),
        pi0.action.expert_layers,
        width,
        q_out,
        kv_out,
        mlp);
}

bool has_valid_pi0_language_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const Pi0Config & pi0 = pi0_config(config);
    if (pi0.llm.layers <= 0 ||
        pi0.llm.width <= 0 ||
        pi0.llm.q_out <= 0 ||
        pi0.llm.kv_out <= 0 ||
        pi0.llm.mlp_width <= 0) {
        return false;
    }
    const int64_t width = pi0.llm.width;
    const int64_t q_out = pi0.llm.q_out;
    const int64_t kv_out = pi0.llm.kv_out;
    const int64_t mlp = pi0.llm.mlp_width;
    const Tensor * lm_head = find_tensor(tensors, pi0_lm_head(config));
    if (lm_head == nullptr ||
        lm_head->shape.size() != 2 ||
        lm_head->shape[0] != width ||
        lm_head->shape[1] <= 0) {
        return false;
    }
    return has_transformer_stack(
        tensors,
        pi0_llm_tensor(config, "norm.scale"),
        pi0_llm_tensor(config, "layers."),
        pi0.llm.layers,
        width,
        q_out,
        kv_out,
        mlp);
}

} // namespace

namespace {

void merge_pi0_role_config(ModelConfig & base, const ModelConfig & current, const char * role) {
    Pi0Config & dst = ensure_pi0_config(base);
    const Pi0Config & src = pi0_config(current);
    const std::string role_name(role);
    if (role_name == "vit") {
        dst.vision = src.vision;
    } else if (role_name == "mmproj") {
        dst.mmproj = src.mmproj;
    } else if (role_name == "llm") {
        dst.llm = src.llm;
    } else if (role_name == "tokenizer") {
        dst.tokenizer = src.tokenizer;
    } else if (role_name == "state") {
        dst.state = src.state;
    } else if (role_name == "action_decoder") {
        dst.action = src.action;
    }
}

struct Pi0ComponentDtype {
    const char * role;
    const VLAComponentConfig * component;
};

vlacpp_status validate_pi0_component_dtypes(const ModelConfig & config) {
    const Pi0Config & pi0 = pi0_config(config);
    const Pi0ComponentDtype components[] = {
        {"vit", &pi0.vision.component},
        {"mmproj", &pi0.mmproj.component},
        {"llm", &pi0.llm.component},
        {"state", &pi0.state.component},
        {"action_decoder", &pi0.action.component},
    };
    for (const Pi0ComponentDtype & item : components) {
        if (!is_component_dtype(item.component->runtime.data_type)) {
            return fail(
                VLACPP_STATUS_PARSE_ERROR,
                std::string("unsupported pi0 component dtype for ") + item.role + ": " + item.component->runtime.data_type);
        }
    }
    return VLACPP_STATUS_OK;
}

VLAComponentConfig * pi0_component_by_role(Pi0Config & pi0, const std::string & role) {
    if (role == "vit") {
        return &pi0.vision.component;
    }
    if (role == "mmproj") {
        return &pi0.mmproj.component;
    }
    if (role == "llm") {
        return &pi0.llm.component;
    }
    if (role == "state") {
        return &pi0.state.component;
    }
    if (role == "action_decoder") {
        return &pi0.action.component;
    }
    return nullptr;
}

vlacpp_status apply_component_dtype_overrides(ModelConfig & config, const BackendConfig & backend) {
    Pi0Config & pi0 = ensure_pi0_config(config);
    for (const auto & item : backend.component_dtype_overrides) {
        VLAComponentConfig * component = pi0_component_by_role(pi0, item.first);
        if (component == nullptr) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "unsupported pi0 dtype override role: " + item.first);
        }
        if (!is_component_dtype(item.second)) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "unsupported dtype override for " + item.first + ": " + item.second);
        }
        component->runtime.data_type = item.second;
    }
    return VLACPP_STATUS_OK;
}

vlacpp_status load_and_merge_component(
    const std::string & path,
    const char * role,
    ModelConfig & config,
    bool & have_config,
    TensorMap & out_tensors) {
    ModelConfig component_config;
    TensorMap component_tensors;
    vlacpp_status status = load_component_gguf_model(
        path,
        "pi0",
        role,
        should_load_pi0_tensor,
        parse_pi0_gguf_metadata,
        component_config,
        component_tensors);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    if (!have_config) {
        config = component_config;
        have_config = true;
    } else {
        status = require_matching_common_config(config, component_config, role, "pi0");
        if (status != VLACPP_STATUS_OK) {
            return status;
        }
    }
    merge_pi0_role_config(config, component_config, role);
    out_tensors = std::move(component_tensors);
    return VLACPP_STATUS_OK;
}

struct Pi0ComponentPaths {
    std::string vit;
    std::string mmproj;
    std::string llm;
    std::string tokenizer;
    std::string state;
    std::string action_decoder;
};

vlacpp_status require_pi0_component_paths(const ModelArtifacts & artifacts, Pi0ComponentPaths & out) {
    ArtifactPathMap paths;
    vlacpp_status status = build_artifact_path_map(artifacts, paths);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = require_artifact_path(paths, "vit", out.vit);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = require_artifact_path(paths, "mmproj", out.mmproj);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = require_artifact_path(paths, "llm", out.llm);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = require_artifact_path(paths, "tokenizer", out.tokenizer);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = require_artifact_path(paths, "state", out.state);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    return require_artifact_path(paths, "action_decoder", out.action_decoder);
}

} // namespace

vlacpp_status load_pi0_model_from_artifacts(
    const ModelArtifacts & artifacts,
    const BackendConfig & backend,
    std::unique_ptr<RuntimeModel> & out) {
    Pi0ComponentPaths paths;
    vlacpp_status status = require_pi0_component_paths(artifacts, paths);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }

    ModelConfig config;
    bool have_config = false;
    Pi0ComponentTensors tensors;
    status = load_and_merge_component(paths.vit, "vit", config, have_config, tensors.vit);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = load_and_merge_component(paths.mmproj, "mmproj", config, have_config, tensors.mmproj);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = load_and_merge_component(paths.llm, "llm", config, have_config, tensors.llm);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    TensorMap tokenizer_tensors;
    status = load_and_merge_component(paths.tokenizer, "tokenizer", config, have_config, tokenizer_tensors);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = load_and_merge_component(paths.state, "state", config, have_config, tensors.state);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = load_and_merge_component(paths.action_decoder, "action_decoder", config, have_config, tensors.action_decoder);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }

    config.component_role.clear();
    status = validate_pi0_component_dtypes(config);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = apply_component_dtype_overrides(config, backend);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    status = validate_pi0_model_config(config, tensors);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    out = make_pi0_model(std::move(config), backend, paths.tokenizer, std::move(tensors));
    return VLACPP_STATUS_OK;
}

vlacpp_status validate_pi0_model_config(ModelConfig & config, const Pi0ComponentTensors & tensors) {
    Pi0Config & pi0 = ensure_pi0_config(config);
    if (!has_valid_pi0_merger_tensors(config, tensors.mmproj)) {
        return fail(
            VLACPP_STATUS_PARSE_ERROR,
            "pi0 merger tensors must use ggml ne order [vision_width, language_width]");
    }
    if (config.common.state_dim > 0) {
        const Tensor * state_w = find_tensor(tensors.state, pi0_state_tensor(config, "weight"));
        const Tensor * state_b = find_tensor(tensors.state, pi0_state_tensor(config, "bias"));
        if (state_w == nullptr || state_b == nullptr ||
            state_w->shape.size() != 2 ||
            state_w->shape[0] != config.common.state_dim ||
            state_b->shape.size() != 1 ||
            state_b->shape[0] != state_w->shape[1] ||
            (pi0.action.width > 0 && state_w->shape[1] != pi0.action.width)) {
            return fail(
                VLACPP_STATUS_PARSE_ERROR,
                "pi0 state component requires state_proj tensors with ggml ne order [state_dim, action_width]");
        }
    }
    if (!has_pi0_action_head_tensors(config, tensors.action_decoder)) {
        return fail(
            VLACPP_STATUS_PARSE_ERROR,
            "pi0 model requires mapped OpenPI action decoder tensors");
    }
    const bool required_weights_present =
        pi0.vision.layers > 0 &&
        pi0.vision.width > 0 &&
        pi0.vision.patch_height > 0 &&
        pi0.vision.patch_width > 0 &&
        pi0.vision.heads > 0 &&
        pi0.vision.norm_epsilon > 0.0f &&
        pi0.llm.layers > 0 &&
        pi0.action.expert_layers > 0 &&
        has_valid_pi0_vit_tensors(config, tensors.vit) &&
        has_valid_pi0_merger_tensors(config, tensors.mmproj) &&
        has_valid_pi0_language_tensors(config, tensors.llm) &&
        has_valid_pi0_action_expert_tensors(config, tensors.action_decoder);
    if (!required_weights_present) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "pi0 component model requires full vit, mmproj, llm, and action decoder weights");
    }
    return VLACPP_STATUS_OK;
}

} // namespace vlacpp
