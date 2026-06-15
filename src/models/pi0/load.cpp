#include "models/pi0/load.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "models/gguf_loader.h"

#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::pi0 {

static bool should_load_pi0_tensor(const Pi0ModelConfig & config, const std::string & name);
static bool validate_pi0_model_config(Pi0ModelConfig & config, const Pi0Components & components);

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

bool is_component_dtype(const std::string & value) {
    return value == "preserve" || value == "fp32" || value == "f16" || value == "bf16";
}

bool pi0_load_error_at(const char * func, const std::string & message) {
    std::fprintf(stderr, "[Pi0] Error: %s: %s\n", func, message.c_str());
    return false;
}

#define pi0_load_error(message) pi0_load_error_at(__func__, (message))

bool require_path(const std::string & path, const char * role) {
    if (path.empty()) {
        return pi0_load_error(std::string("missing pi0 component path: ") + role);
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return pi0_load_error(std::string("failed to open pi0 component ") + role + ": " + path);
    }
    return true;
}

bool gguf_string(gguf_context * gguf, const char * key, std::string & out) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        return false;
    }
    if (gguf_get_kv_type(gguf, idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("invalid GGUF string metadata type: ") + key);
    }
    out = gguf_get_val_str(gguf, idx);
    return true;
}

bool gguf_i32(gguf_context * gguf, const char * key, int & out) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        return false;
    }
    const gguf_type type = gguf_get_kv_type(gguf, idx);
    if (type == GGUF_TYPE_INT32) {
        out = gguf_get_val_i32(gguf, idx);
        return true;
    }
    if (type == GGUF_TYPE_UINT32) {
        out = static_cast<int>(gguf_get_val_u32(gguf, idx));
        return true;
    }
    throw std::runtime_error(std::string("invalid GGUF integer metadata type: ") + key);
}

bool gguf_f32(gguf_context * gguf, const char * key, float & out) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        return false;
    }
    if (gguf_get_kv_type(gguf, idx) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(std::string("invalid GGUF f32 metadata type: ") + key);
    }
    out = gguf_get_val_f32(gguf, idx);
    return true;
}

bool gguf_string_array(gguf_context * gguf, const char * key, std::vector<std::string> & out) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        return false;
    }
    if (gguf_get_kv_type(gguf, idx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("invalid GGUF string-array metadata type: ") + key);
    }
    const size_t count = gguf_get_arr_n(gguf, idx);
    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back(gguf_get_arr_str(gguf, idx, i));
    }
    return true;
}

bool gguf_f32_array(gguf_context * gguf, const char * key, std::vector<float> & out) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        return false;
    }
    if (gguf_get_kv_type(gguf, idx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, idx) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(std::string("invalid GGUF f32-array metadata type: ") + key);
    }
    const size_t count = gguf_get_arr_n(gguf, idx);
    const float * values = static_cast<const float *>(gguf_get_arr_data(gguf, idx));
    if (values == nullptr && count != 0) {
        throw std::runtime_error(std::string("missing GGUF array data: ") + key);
    }
    out.assign(values, values + count);
    return true;
}

bool parse_component_metadata(
    gguf_context * gguf,
    const char * prefix,
    Pi0ComponentConfig & component) {
    std::string key = std::string(prefix) + "architecture";
    gguf_string(gguf, key.c_str(), component.architecture);
    key = std::string(prefix) + "prefix";
    gguf_string(gguf, key.c_str(), component.tensor_prefix);
    key = std::string(prefix) + "backend";
    gguf_string(gguf, key.c_str(), component.runtime.backend);
    key = std::string(prefix) + "dtype";
    gguf_string(gguf, key.c_str(), component.runtime.data_type);
    key = std::string(prefix) + "n_threads";
    gguf_i32(gguf, key.c_str(), component.runtime.n_threads);
    return true;
}

bool parse_common_metadata(gguf_context * gguf, Pi0ModelConfig & config) {
    gguf_string(gguf, "pi0.model_type", config.common.model_type);
    gguf_string(gguf, "pi0.component.role", config.component_role);
    gguf_i32(gguf, "pi0.image_width", config.common.image_width);
    gguf_i32(gguf, "pi0.image_height", config.common.image_height);
    gguf_i32(gguf, "pi0.state_dim", config.common.state_dim);
    gguf_i32(gguf, "pi0.action_dim", config.common.action_dim);
    gguf_i32(gguf, "pi0.action_horizon", config.common.action_horizon);
    gguf_i32(gguf, "pi0.max_token_len", config.common.max_token_len);
    gguf_string_array(gguf, "pi0.image_keys", config.common.image_keys);
    gguf_f32_array(gguf, "pi0.state_mean", config.common.state_mean);
    gguf_f32_array(gguf, "pi0.state_std", config.common.state_std);
    gguf_f32_array(gguf, "pi0.action_mean", config.common.action_mean);
    gguf_f32_array(gguf, "pi0.action_std", config.common.action_std);
    return true;
}

bool validate_common_metadata(const Pi0ModelConfig & config) {
    if (config.common.action_dim <= 0 || config.common.action_horizon <= 0) {
        return pi0_load_error("model config requires positive action_dim and action_horizon");
    }
    if (config.common.state_dim < 0) {
        return pi0_load_error("state_dim must be non-negative");
    }
    if ((!config.common.state_mean.empty() || !config.common.state_std.empty()) &&
        (config.common.state_mean.size() != static_cast<size_t>(config.common.state_dim) ||
            config.common.state_std.size() != static_cast<size_t>(config.common.state_dim))) {
        return pi0_load_error("state_mean and state_std must both match state_dim");
    }
    if ((!config.common.action_mean.empty() || !config.common.action_std.empty()) &&
        (config.common.action_mean.size() != static_cast<size_t>(config.common.action_dim) ||
            config.common.action_std.size() != static_cast<size_t>(config.common.action_dim))) {
        return pi0_load_error("action_mean and action_std must both match action_dim");
    }
    return true;
}

bool parse_pi0_gguf_metadata(gguf_context * gguf, Pi0ModelConfig & config) {
    parse_common_metadata(gguf, config);
    Pi0Config & pi0 = ensure_pi0_config(config);
    parse_component_metadata(gguf, "pi0.component.vit.", pi0.vision.component);
    parse_component_metadata(gguf, "pi0.component.llm.", pi0.llm.component);
    parse_component_metadata(gguf, "pi0.component.mmproj.", pi0.mmproj.component);
    parse_component_metadata(gguf, "pi0.component.tokenizer.", pi0.tokenizer);
    parse_component_metadata(gguf, "pi0.component.action_decoder.", pi0.action.component);
    parse_component_metadata(gguf, "pi0.component.state.", pi0.state.component);

    gguf_i32(gguf, "pi0.action_decoder.width", pi0.action.width);
    gguf_i32(gguf, "pi0.vit.width", pi0.vision.width);
    gguf_i32(gguf, "pi0.vit.patch_height", pi0.vision.patch_height);
    gguf_i32(gguf, "pi0.vit.patch_width", pi0.vision.patch_width);
    gguf_i32(gguf, "pi0.vit.layers", pi0.vision.layers);
    gguf_i32(gguf, "pi0.vit.heads", pi0.vision.heads);
    gguf_i32(gguf, "pi0.llm.width", pi0.llm.width);
    gguf_i32(gguf, "pi0.llm.q_out", pi0.llm.q_out);
    gguf_i32(gguf, "pi0.llm.kv_out", pi0.llm.kv_out);
    gguf_i32(gguf, "pi0.llm.mlp_width", pi0.llm.mlp_width);
    gguf_i32(gguf, "pi0.llm.layers", pi0.llm.layers);
    gguf_i32(gguf, "pi0.action_decoder.expert_width", pi0.action.expert_width);
    gguf_i32(gguf, "pi0.action_decoder.q_out", pi0.action.expert_q_out);
    gguf_i32(gguf, "pi0.action_decoder.kv_out", pi0.action.expert_kv_out);
    gguf_i32(gguf, "pi0.action_decoder.mlp_width", pi0.action.expert_mlp_width);
    gguf_i32(gguf, "pi0.action_decoder.layers", pi0.action.expert_layers);
    gguf_f32(gguf, "pi0.vit.norm_epsilon", pi0.vision.norm_epsilon);
    return validate_common_metadata(config);
}

class Pi0ComponentLoader final : public gguf_loader {
public:
    explicit Pi0ComponentLoader(Pi0ModelConfig & config)
        : config_(config) {
    }

protected:
    bool parse_metadata(gguf_context * gguf) override {
        try {
            return parse_pi0_gguf_metadata(gguf, config_);
        } catch (const std::exception & error) {
            set_error(error.what());
            return false;
        }
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx_data); cur != nullptr; cur = ggml_get_next_tensor(ctx_data, cur)) {
            const std::string name = ggml_get_name(cur);
            if (!should_load_pi0_tensor(config_, name)) {
                continue;
            }
            if (cur->type != GGML_TYPE_F32 && cur->type != GGML_TYPE_F16 && cur->type != GGML_TYPE_BF16) {
                set_error("unsupported pi0 tensor type: " + name);
                return false;
            }
        }
        return true;
    }

private:
    Pi0ModelConfig & config_;
};

} // namespace

static bool should_load_pi0_tensor(const Pi0ModelConfig & config, const std::string & name) {
    const Pi0Config & pi0 = pi0_config(config);
    return starts_with(name, pi0.action.component.tensor_prefix) ||
        starts_with(name, pi0.state.component.tensor_prefix) ||
        starts_with(name, pi0.mmproj.component.tensor_prefix) ||
        starts_with(name, pi0.vision.component.tensor_prefix) ||
        starts_with(name, pi0.llm.component.tensor_prefix);
}

std::string pi0_action_decoder_tensor(const Pi0ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).action.component.tensor_prefix, suffix);
}

std::string pi0_action_decoder_layer_prefix(const Pi0ModelConfig & config, int layer) {
    return pi0_action_decoder_tensor(config, "layers." + std::to_string(layer) + ".");
}

std::string pi0_state_tensor(const Pi0ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).state.component.tensor_prefix, suffix);
}

std::string pi0_llm_tensor(const Pi0ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).llm.component.tensor_prefix, suffix);
}

std::string pi0_llm_layer_prefix(const Pi0ModelConfig & config, int layer) {
    return pi0_llm_tensor(config, "layers." + std::to_string(layer) + ".");
}

std::string pi0_lm_head(const Pi0ModelConfig & config) {
    return pi0_llm_tensor(config, "lm_head.weight");
}

std::string pi0_merger_tensor(const Pi0ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).mmproj.component.tensor_prefix, suffix);
}

std::string pi0_vit_tensor(const Pi0ModelConfig & config, const std::string & suffix) {
    return with_prefix(pi0_config(config).vision.component.tensor_prefix, suffix);
}

std::string pi0_vit_layer_prefix(const Pi0ModelConfig & config, int layer) {
    return pi0_vit_tensor(config, "encoder.layers." + std::to_string(layer) + ".");
}

namespace {

ggml_tensor * find_pi0_tensor(ggml_context * ctx, const std::string & name) {
    return ctx != nullptr ? ggml_get_tensor(ctx, name.c_str()) : nullptr;
}

bool has_shape(ggml_context * ctx, const std::string & name, std::initializer_list<int64_t> expected) {
    ggml_tensor * tensor = find_pi0_tensor(ctx, name);
    if (tensor == nullptr || static_cast<size_t>(ggml_n_dims(tensor)) != expected.size()) {
        return false;
    }
    size_t i = 0;
    for (const int64_t dim : expected) {
        if (tensor->ne[i++] != dim) {
            return false;
        }
    }
    return true;
}

bool has_transformer_stack(
    ggml_context * ctx,
    const std::string & final_norm_scale_name,
    const std::string & layers_prefix,
    int layers,
    int64_t width,
    int64_t q_out,
    int64_t kv_out,
    int64_t mlp) {
    if (!has_shape(ctx, final_norm_scale_name, {width})) {
        return false;
    }
    for (int layer = 0; layer < layers; ++layer) {
        const std::string base = layers_prefix + std::to_string(layer) + ".";
        if (!has_shape(ctx, base + "input_layernorm.scale", {width}) ||
            !has_shape(ctx, base + "post_attention_layernorm.scale", {width}) ||
            !has_shape(ctx, base + "self_attn.q_proj.weight", {width, q_out}) ||
            !has_shape(ctx, base + "self_attn.k_proj.weight", {width, kv_out}) ||
            !has_shape(ctx, base + "self_attn.v_proj.weight", {width, kv_out}) ||
            !has_shape(ctx, base + "self_attn.o_proj.weight", {q_out, width}) ||
            !has_shape(ctx, base + "mlp.gate_proj.weight", {width, mlp}) ||
            !has_shape(ctx, base + "mlp.up_proj.weight", {width, mlp}) ||
            !has_shape(ctx, base + "mlp.down_proj.weight", {mlp, width})) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace {

bool has_valid_pi0_action_projection_tensors(const Pi0ModelConfig & config, ggml_context * ctx) {
    ggml_tensor * in_w = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_in_proj.weight"));
    ggml_tensor * in_b = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_in_proj.bias"));
    ggml_tensor * time_in_w = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_time_mlp_in.weight"));
    ggml_tensor * time_in_b = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_time_mlp_in.bias"));
    ggml_tensor * time_out_w = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_time_mlp_out.weight"));
    ggml_tensor * time_out_b = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_time_mlp_out.bias"));
    ggml_tensor * out_w = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_out_proj.weight"));
    ggml_tensor * out_b = find_pi0_tensor(ctx, pi0_action_decoder_tensor(config, "action_out_proj.bias"));
    if (in_w == nullptr || in_b == nullptr ||
        time_in_w == nullptr || time_in_b == nullptr ||
        time_out_w == nullptr || time_out_b == nullptr ||
        out_w == nullptr || out_b == nullptr) {
        return false;
    }

    if (ggml_n_dims(in_w) != 2 || in_w->ne[0] != config.common.action_dim || ggml_n_dims(in_b) != 1) {
        return false;
    }
    const int64_t width = in_w->ne[1];
    return in_b->ne[0] == width &&
        ggml_n_dims(time_in_w) == 2 && time_in_w->ne[0] == 2 * width && time_in_w->ne[1] == width &&
        ggml_n_dims(time_in_b) == 1 && time_in_b->ne[0] == width &&
        ggml_n_dims(time_out_w) == 2 && time_out_w->ne[0] == width && time_out_w->ne[1] == width &&
        ggml_n_dims(time_out_b) == 1 && time_out_b->ne[0] == width &&
        ggml_n_dims(out_w) == 2 && out_w->ne[0] == width && out_w->ne[1] == config.common.action_dim &&
        ggml_n_dims(out_b) == 1 && out_b->ne[0] == config.common.action_dim;
}

bool has_valid_pi0_merger_tensors(const Pi0ModelConfig & config, ggml_context * ctx) {
    const Pi0Config & pi0 = pi0_config(config);
    ggml_tensor * weight = find_pi0_tensor(ctx, pi0_merger_tensor(config, "weight"));
    ggml_tensor * bias = find_pi0_tensor(ctx, pi0_merger_tensor(config, "bias"));
    if (weight == nullptr || bias == nullptr) {
        return false;
    }
    if (pi0.vision.width <= 0 ||
        pi0.llm.width <= 0 ||
        ggml_n_dims(weight) != 2 ||
        ggml_n_dims(bias) != 1) {
        return false;
    }
    return weight->ne[0] == pi0.vision.width &&
        weight->ne[1] == pi0.llm.width &&
        bias->ne[0] == pi0.llm.width;
}

bool has_valid_pi0_vit_tensors(const Pi0ModelConfig & config, ggml_context * ctx) {
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
    if (!has_shape(ctx, pi0_vit_tensor(config, "embeddings.patch_embedding.weight"), {patch_w, patch_h, 3, width}) ||
        !has_shape(ctx, pi0_vit_tensor(config, "embeddings.patch_embedding.bias"), {width}) ||
        !has_shape(ctx, pi0_vit_tensor(config, "embeddings.position_embedding.weight"), {width, patches}) ||
        !has_shape(ctx, pi0_vit_tensor(config, "post_layernorm.weight"), {width}) ||
        !has_shape(ctx, pi0_vit_tensor(config, "post_layernorm.bias"), {width})) {
        return false;
    }
    const int last = pi0.vision.layers - 1;
    for (int layer : {0, last}) {
        const std::string base = pi0_vit_layer_prefix(config, layer);
        if (!has_shape(ctx, base + "layer_norm1.weight", {width}) ||
            !has_shape(ctx, base + "layer_norm1.bias", {width}) ||
            !has_shape(ctx, base + "self_attn.q_proj.weight", {width, width}) ||
            !has_shape(ctx, base + "self_attn.q_proj.bias", {width}) ||
            !has_shape(ctx, base + "self_attn.out_proj.weight", {width, width}) ||
            !has_shape(ctx, base + "self_attn.out_proj.bias", {width}) ||
            !has_shape(ctx, base + "layer_norm2.weight", {width}) ||
            !has_shape(ctx, base + "layer_norm2.bias", {width})) {
            return false;
        }
    }
    return true;
}

bool has_valid_pi0_action_expert_tensors(const Pi0ModelConfig & config, ggml_context * ctx) {
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
        ctx,
        pi0_action_decoder_tensor(config, "norm.scale"),
        pi0_action_decoder_tensor(config, "layers."),
        pi0.action.expert_layers,
        width,
        q_out,
        kv_out,
        mlp);
}

bool has_valid_pi0_language_tensors(const Pi0ModelConfig & config, ggml_context * ctx) {
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
    ggml_tensor * lm_head = find_pi0_tensor(ctx, pi0_lm_head(config));
    if (lm_head == nullptr ||
        ggml_n_dims(lm_head) != 2 ||
        lm_head->ne[0] != width ||
        lm_head->ne[1] <= 0) {
        return false;
    }
    return has_transformer_stack(
        ctx,
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

void merge_pi0_role_config(Pi0ModelConfig & base, const Pi0ModelConfig & current, const char * role) {
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

bool validate_pi0_component_dtypes(const Pi0ModelConfig & config) {
    const Pi0Config & pi0 = pi0_config(config);
    auto check = [](const char * role, const Pi0ComponentConfig & component) {
        if (is_component_dtype(component.runtime.data_type)) {
            return true;
        }
        return pi0_load_error(
            std::string("unsupported pi0 component dtype for ") + role + ": " + component.runtime.data_type);
    };

    return check("vit", pi0.vision.component) &&
        check("mmproj", pi0.mmproj.component) &&
        check("llm", pi0.llm.component) &&
        check("state", pi0.state.component) &&
        check("action_decoder", pi0.action.component);
}

bool read_component_metadata(
    const std::string & path,
    const char * role,
    Pi0ModelConfig & component_config) {
    ggml_context * meta = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta,
    };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        if (meta != nullptr) {
            ggml_free(meta);
        }
        return pi0_load_error(std::string("failed to load pi0 component metadata: ") + path);
    }

    bool ok = false;
    try {
        ok = parse_pi0_gguf_metadata(gguf, component_config);
    } catch (const std::exception & error) {
        pi0_load_error(error.what());
        ok = false;
    }
    if (meta != nullptr) {
        ggml_free(meta);
    }
    gguf_free(gguf);
    if (!ok) {
        return false;
    }
    if (component_config.common.model_type != "pi0") {
        return pi0_load_error("unsupported pi0 component model_type: " + component_config.common.model_type);
    }
    if (component_config.component_role != role) {
        const std::string actual = component_config.component_role.empty() ? "<missing>" : component_config.component_role;
        return pi0_load_error(std::string("pi0 component role mismatch: expected ") + role + ", got " + actual);
    }
    return true;
}

bool merge_component_config(
    const std::string & path,
    const char * role,
    Pi0ModelConfig & config,
    bool & have_config) {
    if (!require_path(path, role)) {
        return false;
    }
    Pi0ModelConfig component_config;
    if (!read_component_metadata(path, role, component_config)) {
        return false;
    }
    if (!have_config) {
        config = component_config;
        have_config = true;
    } else {
        const Pi0CommonConfig & lhs = config.common;
        const Pi0CommonConfig & rhs = component_config.common;
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
            return pi0_load_error(std::string("pi0 component config mismatch: ") + role);
        }
    }
    merge_pi0_role_config(config, component_config, role);
    return true;
}

bool load_component_tensors(
    const std::string & path,
    const char * role,
    ggml_backend_buffer_type_t model_buft,
    gguf_load_result & out_component) {
    Pi0ModelConfig component_config;
    Pi0ComponentLoader loader(component_config);
    if (!loader.load(path.c_str(), model_buft, out_component, 0)) {
        return pi0_load_error(loader.error());
    }
    if (component_config.common.model_type != "pi0") {
        free_pi0_loaded_component(out_component);
        return pi0_load_error("unsupported pi0 component model_type: " + component_config.common.model_type);
    }
    if (component_config.component_role != role) {
        const std::string actual = component_config.component_role.empty() ? "<missing>" : component_config.component_role;
        free_pi0_loaded_component(out_component);
        return pi0_load_error(std::string("pi0 component role mismatch: expected ") + role + ", got " + actual);
    }
    return true;
}

bool init_pi0_component_backends(
    const Pi0ModelConfig & config,
    const Pi0BackendConfig & backend,
    Pi0Components & components,
    int verbosity) {
    const Pi0Config & pi0 = pi0_config(config);
    try {
        pi0_init_component_runtime(components.vit.runtime, backend, pi0.vision.component, "vit", verbosity);
        pi0_init_component_runtime(components.mmproj.runtime, backend, pi0.mmproj.component, "mmproj", verbosity);
        pi0_init_component_runtime(components.llm.runtime, backend, pi0.llm.component, "llm", verbosity);
        pi0_init_component_runtime(components.state.runtime, backend, pi0.state.component, "state", verbosity);
        pi0_init_component_runtime(components.action_decoder.runtime, backend, pi0.action.component, "action_decoder", verbosity);
    } catch (const std::exception & error) {
        return pi0_load_error(error.what());
    }
    return true;
}

} // namespace

bool load_pi0_components(
    const Pi0ComponentPaths & paths,
    const Pi0BackendConfig & backend,
    Pi0ModelConfig & out_config,
    Pi0Components & out_components,
    int verbosity) {
    Pi0ModelConfig config;
    bool have_config = false;
    if (!merge_component_config(paths.vit, "vit", config, have_config)) {
        return false;
    }
    if (!merge_component_config(paths.mmproj, "mmproj", config, have_config)) {
        return false;
    }
    if (!merge_component_config(paths.llm, "llm", config, have_config)) {
        return false;
    }
    if (!merge_component_config(paths.tokenizer, "tokenizer", config, have_config)) {
        return false;
    }
    if (!merge_component_config(paths.state, "state", config, have_config)) {
        return false;
    }
    if (!merge_component_config(paths.action_decoder, "action_decoder", config, have_config)) {
        return false;
    }

    config.component_role.clear();
    if (!validate_pi0_component_dtypes(config)) {
        return false;
    }

    Pi0Components components;
    if (!init_pi0_component_backends(config, backend, components, verbosity)) {
        return false;
    }
    if (!load_component_tensors(paths.vit, "vit", components.vit.runtime.buft_policy.model_buft, components.vit.loaded)) {
        return false;
    }
    if (!load_component_tensors(paths.mmproj, "mmproj", components.mmproj.runtime.buft_policy.model_buft, components.mmproj.loaded)) {
        return false;
    }
    if (!load_component_tensors(paths.llm, "llm", components.llm.runtime.buft_policy.model_buft, components.llm.loaded)) {
        return false;
    }
    if (!load_component_tensors(paths.state, "state", components.state.runtime.buft_policy.model_buft, components.state.loaded)) {
        return false;
    }
    if (!load_component_tensors(
            paths.action_decoder,
            "action_decoder",
            components.action_decoder.runtime.buft_policy.model_buft,
            components.action_decoder.loaded)) {
        return false;
    }
    if (!validate_pi0_model_config(config, components)) {
        return false;
    }
    out_config = std::move(config);
    out_components = std::move(components);
    return true;
}

void free_pi0_loaded_component(gguf_load_result & component) {
    if (component.model_buffer != nullptr) {
        ggml_backend_buffer_free(component.model_buffer);
    }
    if (component.ctx_data != nullptr) {
        ggml_free(component.ctx_data);
    }
    if (component.gguf != nullptr) {
        gguf_free(component.gguf);
    }
    component = {};
}

static bool validate_pi0_model_config(Pi0ModelConfig & config, const Pi0Components & components) {
    Pi0Config & pi0 = ensure_pi0_config(config);
    if (!has_valid_pi0_merger_tensors(config, components.mmproj.loaded.ctx_data)) {
        return pi0_load_error("pi0 merger tensors must use ggml ne order [vision_width, language_width]");
    }
    if (config.common.state_dim > 0) {
        ggml_tensor * state_w = find_pi0_tensor(components.state.loaded.ctx_data, pi0_state_tensor(config, "weight"));
        ggml_tensor * state_b = find_pi0_tensor(components.state.loaded.ctx_data, pi0_state_tensor(config, "bias"));
        if (state_w == nullptr || state_b == nullptr ||
            ggml_n_dims(state_w) != 2 ||
            state_w->ne[0] != config.common.state_dim ||
            ggml_n_dims(state_b) != 1 ||
            state_b->ne[0] != state_w->ne[1] ||
            (pi0.action.width > 0 && state_w->ne[1] != pi0.action.width)) {
            return pi0_load_error("pi0 state component requires state_proj tensors with ggml ne order [state_dim, action_width]");
        }
    }
    if (!has_valid_pi0_action_projection_tensors(config, components.action_decoder.loaded.ctx_data)) {
        return pi0_load_error("pi0 model requires pi0 action decoder projection tensors");
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
        has_valid_pi0_vit_tensors(config, components.vit.loaded.ctx_data) &&
        has_valid_pi0_merger_tensors(config, components.mmproj.loaded.ctx_data) &&
        has_valid_pi0_language_tensors(config, components.llm.loaded.ctx_data) &&
        has_valid_pi0_action_expert_tensors(config, components.action_decoder.loaded.ctx_data);
    if (!required_weights_present) {
        return pi0_load_error("pi0 component model requires full vit, mmproj, llm, and action decoder weights");
    }
    return true;
}

} // namespace robotcpp::pi0
