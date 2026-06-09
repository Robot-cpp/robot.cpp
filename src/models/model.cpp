#include "models/model.h"

#include "core/error.h"
#include "core/gguf.h"
#include "core/json.h"

#include <fstream>
#include <memory>
#include <string>

namespace vlacpp {
namespace {

bool has_action_head_tensors(const ModelConfig & config, const TensorMap & tensors) {
    const char * required[] = {
        "vlacpp.openpi.action_in_proj.weight",
        "vlacpp.openpi.action_in_proj.bias",
        "vlacpp.openpi.action_time_mlp_in.weight",
        "vlacpp.openpi.action_time_mlp_in.bias",
        "vlacpp.openpi.action_time_mlp_out.weight",
        "vlacpp.openpi.action_time_mlp_out.bias",
        "vlacpp.openpi.action_out_proj.weight",
        "vlacpp.openpi.action_out_proj.bias",
    };
    for (const char * name : required) {
        if (tensors.find(name) == tensors.end()) {
            return false;
        }
    }

    const Tensor & in_w = tensors.at("vlacpp.openpi.action_in_proj.weight");
    const Tensor & in_b = tensors.at("vlacpp.openpi.action_in_proj.bias");
    const Tensor & time_in_w = tensors.at("vlacpp.openpi.action_time_mlp_in.weight");
    const Tensor & time_in_b = tensors.at("vlacpp.openpi.action_time_mlp_in.bias");
    const Tensor & time_out_w = tensors.at("vlacpp.openpi.action_time_mlp_out.weight");
    const Tensor & time_out_b = tensors.at("vlacpp.openpi.action_time_mlp_out.bias");
    const Tensor & out_w = tensors.at("vlacpp.openpi.action_out_proj.weight");
    const Tensor & out_b = tensors.at("vlacpp.openpi.action_out_proj.bias");

    if (in_w.shape.size() != 2 || in_w.shape[0] != config.action_dim || in_b.shape.size() != 1) {
        return false;
    }
    const int64_t width = in_w.shape[1];
    return in_b.shape[0] == width &&
        time_in_w.shape.size() == 2 && time_in_w.shape[0] == 2 * width && time_in_w.shape[1] == width &&
        time_in_b.shape.size() == 1 && time_in_b.shape[0] == width &&
        time_out_w.shape.size() == 2 && time_out_w.shape[0] == width && time_out_w.shape[1] == width &&
        time_out_b.shape.size() == 1 && time_out_b.shape[0] == width &&
        out_w.shape.size() == 2 && out_w.shape[0] == width && out_w.shape[1] == config.action_dim &&
        out_b.shape.size() == 1 && out_b.shape[0] == config.action_dim;
}

bool has_valid_vision_projector_tensors(const ModelConfig & config, const TensorMap & tensors) {
    auto weight = tensors.find("vlacpp.openpi.vision_projector.weight");
    auto bias = tensors.find("vlacpp.openpi.vision_projector.bias");
    if (weight == tensors.end() && bias == tensors.end()) {
        return true;
    }
    if (weight == tensors.end() || bias == tensors.end()) {
        return false;
    }
    const int64_t ne0 = config.openpi_vision_width > 0 ? config.openpi_vision_width : weight->second.shape[0];
    const int64_t ne1 = config.openpi_language_width > 0 ? config.openpi_language_width : weight->second.shape[1];
    return weight->second.shape.size() == 2 &&
        weight->second.shape[0] == ne0 &&
        weight->second.shape[1] == ne1 &&
        weight->second.data.size() == static_cast<size_t>(ne0 * ne1) &&
        bias->second.shape.size() == 1 &&
        bias->second.shape[0] == ne1 &&
        bias->second.data.size() == static_cast<size_t>(ne1);
}

bool has_tensor_name_or_model_prefix(const TensorMap & tensors, const std::string & name) {
    return tensors.find(name) != tensors.end() || tensors.find("model." + name) != tensors.end();
}

const Tensor * find_tensor_name_or_model_prefix(const TensorMap & tensors, const std::string & name) {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        return &it->second;
    }
    it = tensors.find("model." + name);
    if (it != tensors.end()) {
        return &it->second;
    }
    return nullptr;
}

bool has_tensor_shape(
    const TensorMap & tensors,
    const std::string & name,
    std::initializer_list<int64_t> expected) {
    const Tensor * tensor = find_tensor_name_or_model_prefix(tensors, name);
    if (tensor == nullptr || tensor->shape.size() != expected.size()) {
        return false;
    }
    size_t i = 0;
    int64_t count = 1;
    for (const int64_t dim : expected) {
        if (tensor->shape[i++] != dim) {
            return false;
        }
        count *= dim;
    }
    return tensor->data.size() == static_cast<size_t>(count);
}

bool has_valid_action_expert_tensors(const ModelConfig & config, const TensorMap & tensors) {
    if (config.openpi_action_expert_layers <= 0 ||
        config.openpi_action_expert_width <= 0 ||
        config.openpi_action_expert_q_out <= 0 ||
        config.openpi_action_expert_kv_out <= 0 ||
        config.openpi_action_expert_mlp_width <= 0) {
        return false;
    }
    const int64_t width = config.openpi_action_expert_width;
    const int64_t q_out = config.openpi_action_expert_q_out;
    const int64_t kv_out = config.openpi_action_expert_kv_out;
    const int64_t mlp = config.openpi_action_expert_mlp_width;
    const std::string prefix = "paligemma_with_expert.gemma_expert.model.layers.";
    if (!has_tensor_shape(tensors, "paligemma_with_expert.gemma_expert.model.norm.weight", {width})) {
        return false;
    }
    for (int layer = 0; layer < config.openpi_action_expert_layers; ++layer) {
        const std::string base = prefix + std::to_string(layer) + ".";
        if (!has_tensor_shape(tensors, base + "input_layernorm.weight", {width}) ||
            !has_tensor_shape(tensors, base + "post_attention_layernorm.weight", {width}) ||
            !has_tensor_shape(tensors, base + "self_attn.q_proj.weight", {width, q_out}) ||
            !has_tensor_shape(tensors, base + "self_attn.k_proj.weight", {width, kv_out}) ||
            !has_tensor_shape(tensors, base + "self_attn.v_proj.weight", {width, kv_out}) ||
            !has_tensor_shape(tensors, base + "self_attn.o_proj.weight", {q_out, width}) ||
            !has_tensor_shape(tensors, base + "mlp.gate_proj.weight", {width, mlp}) ||
            !has_tensor_shape(tensors, base + "mlp.up_proj.weight", {width, mlp}) ||
            !has_tensor_shape(tensors, base + "mlp.down_proj.weight", {mlp, width})) {
            return false;
        }
    }
    return true;
}

bool has_valid_language_tensors(const ModelConfig & config, const TensorMap & tensors) {
    if (config.openpi_language_layers <= 0 ||
        config.openpi_language_width <= 0 ||
        config.openpi_language_q_out <= 0 ||
        config.openpi_language_kv_out <= 0 ||
        config.openpi_language_mlp_width <= 0) {
        return false;
    }
    const int64_t width = config.openpi_language_width;
    const int64_t q_out = config.openpi_language_q_out;
    const int64_t kv_out = config.openpi_language_kv_out;
    const int64_t mlp = config.openpi_language_mlp_width;
    const std::string prefix = "paligemma_with_expert.paligemma.model.language_model.layers.";
    if (!has_tensor_shape(tensors, "paligemma_with_expert.paligemma.model.language_model.norm.weight", {width})) {
        return false;
    }
    for (int layer = 0; layer < config.openpi_language_layers; ++layer) {
        const std::string base = prefix + std::to_string(layer) + ".";
        if (!has_tensor_shape(tensors, base + "input_layernorm.weight", {width}) ||
            !has_tensor_shape(tensors, base + "post_attention_layernorm.weight", {width}) ||
            !has_tensor_shape(tensors, base + "self_attn.q_proj.weight", {width, q_out}) ||
            !has_tensor_shape(tensors, base + "self_attn.k_proj.weight", {width, kv_out}) ||
            !has_tensor_shape(tensors, base + "self_attn.v_proj.weight", {width, kv_out}) ||
            !has_tensor_shape(tensors, base + "self_attn.o_proj.weight", {q_out, width}) ||
            !has_tensor_shape(tensors, base + "mlp.gate_proj.weight", {width, mlp}) ||
            !has_tensor_shape(tensors, base + "mlp.up_proj.weight", {width, mlp}) ||
            !has_tensor_shape(tensors, base + "mlp.down_proj.weight", {mlp, width})) {
            return false;
        }
    }
    return true;
}

bool has_full_openpi_weight_tensors(const ModelConfig & config, const TensorMap & tensors) {
    if (config.openpi_vision_layers <= 0 ||
        config.openpi_language_layers <= 0 ||
        config.openpi_action_expert_layers <= 0 ||
        !has_action_head_tensors(config, tensors) ||
        !has_valid_vision_projector_tensors(config, tensors) ||
        !has_valid_language_tensors(config, tensors) ||
        !has_valid_action_expert_tensors(config, tensors)) {
        return false;
    }
    const int last_vision = config.openpi_vision_layers - 1;
    const int last_language = config.openpi_language_layers - 1;
    const std::string vision_prefix =
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.";
    const std::string language_prefix =
        "paligemma_with_expert.paligemma.model.language_model.layers.";
    return
        has_tensor_name_or_model_prefix(
            tensors,
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight") &&
        has_tensor_name_or_model_prefix(tensors, vision_prefix + "0.self_attn.q_proj.weight") &&
        has_tensor_name_or_model_prefix(tensors, vision_prefix + std::to_string(last_vision) + ".self_attn.q_proj.weight") &&
        has_tensor_name_or_model_prefix(tensors, language_prefix + "0.self_attn.q_proj.weight") &&
        has_tensor_name_or_model_prefix(tensors, language_prefix + std::to_string(last_language) + ".self_attn.q_proj.weight");
}

vlacpp_status validate_pi0_tensors(const ModelConfig & config, const TensorMap & tensors) {
    if (!has_valid_vision_projector_tensors(config, tensors)) {
        return fail(
            VLACPP_STATUS_PARSE_ERROR,
            "pi0 vision projector tensors must use ggml ne order [vision_width, language_width]");
    }

    if (has_action_head_tensors(config, tensors)) {
        return VLACPP_STATUS_OK;
    }
    return fail(
        VLACPP_STATUS_PARSE_ERROR,
        "pi0 model requires mapped OpenPI action-head tensors");
}

} // namespace

std::unique_ptr<RuntimeModel> make_pi0_model(ModelConfig config, BackendConfig backend, TensorMap tensors);

vlacpp_status load_model_from_path(
    const std::string & path,
    const BackendConfig & backend,
    std::unique_ptr<RuntimeModel> & out) {
    ModelConfig config;
    TensorMap tensors;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(VLACPP_STATUS_IO_ERROR, "failed to open model file: " + path);
    }
    char magic[4] = {};
    file.read(magic, sizeof(magic));
    const bool is_gguf = file && magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';

    vlacpp_status status = is_gguf ? load_gguf_model_file(path, config, tensors) : load_config_file(path, config);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    config.source_path = path;

    if (config.model_type == "pi0") {
        status = validate_pi0_tensors(config, tensors);
        if (status != VLACPP_STATUS_OK) {
            return status;
        }
        config.openpi_full_weights_present = has_full_openpi_weight_tensors(config, tensors);
        out = make_pi0_model(std::move(config), backend, std::move(tensors));
        return VLACPP_STATUS_OK;
    }

    return fail(VLACPP_STATUS_UNSUPPORTED, "unsupported model_type: " + config.model_type);
}

} // namespace vlacpp
