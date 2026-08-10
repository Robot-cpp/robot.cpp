#include "models/starvla/pi_v3_policy.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "models/ggml_backend.h"
#include "models/gguf_loader.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::starvla {

namespace {

constexpr size_t kGraphSize = 32768;
constexpr int kKQMaskPad = 32;
constexpr int kReleasedLayerCount = 36;

struct PIV3BlockWeights {
    ggml_tensor * ada_norm_weight = nullptr;
    ggml_tensor * ada_norm_bias = nullptr;
    ggml_tensor * query_weight = nullptr;
    ggml_tensor * query_bias = nullptr;
    ggml_tensor * key_weight = nullptr;
    ggml_tensor * key_bias = nullptr;
    ggml_tensor * value_weight = nullptr;
    ggml_tensor * value_bias = nullptr;
    ggml_tensor * attention_output_weight = nullptr;
    ggml_tensor * attention_output_bias = nullptr;
    ggml_tensor * feed_forward_input_weight = nullptr;
    ggml_tensor * feed_forward_input_bias = nullptr;
    ggml_tensor * feed_forward_output_weight = nullptr;
    ggml_tensor * feed_forward_output_bias = nullptr;
};

struct PIV3ProjectorWeights {
    ggml_tensor * norm_weight = nullptr;
    ggml_tensor * norm_bias = nullptr;
    ggml_tensor * projection_weight = nullptr;
    ggml_tensor * projection_bias = nullptr;
};

struct PIV3Weights {
    ggml_tensor * timestep_input_weight = nullptr;
    ggml_tensor * timestep_input_bias = nullptr;
    ggml_tensor * timestep_output_weight = nullptr;
    ggml_tensor * timestep_output_bias = nullptr;
    std::vector<PIV3BlockWeights> blocks;
    std::vector<PIV3ProjectorWeights> projectors;
    ggml_tensor * action_input_weight = nullptr;
    ggml_tensor * action_input_bias = nullptr;
    ggml_tensor * action_time_mix_weight = nullptr;
    ggml_tensor * action_time_mix_bias = nullptr;
    ggml_tensor * action_output_weight = nullptr;
    ggml_tensor * action_output_bias = nullptr;
    ggml_tensor * velocity_input_weight = nullptr;
    ggml_tensor * velocity_input_bias = nullptr;
    ggml_tensor * velocity_output_weight = nullptr;
    ggml_tensor * velocity_output_bias = nullptr;
    ggml_tensor * future_tokens = nullptr;
    ggml_tensor * action_position = nullptr;
};

int require_key(gguf_context * gguf, const char * key, gguf_type type) {
    const int index = gguf_find_key(gguf, key);
    if (index < 0) {
        throw std::runtime_error(std::string("missing required StarVLA GGUF metadata: ") + key);
    }
    if (gguf_get_kv_type(gguf, index) != type) {
        throw std::runtime_error(std::string("invalid StarVLA GGUF metadata type: ") + key);
    }
    return index;
}

std::string require_string(gguf_context * gguf, const char * key) {
    return gguf_get_val_str(gguf, require_key(gguf, key, GGUF_TYPE_STRING));
}

int require_i32(gguf_context * gguf, const char * key) {
    return gguf_get_val_i32(gguf, require_key(gguf, key, GGUF_TYPE_INT32));
}

float require_f32(gguf_context * gguf, const char * key) {
    return gguf_get_val_f32(gguf, require_key(gguf, key, GGUF_TYPE_FLOAT32));
}

bool require_bool(gguf_context * gguf, const char * key) {
    return gguf_get_val_bool(gguf, require_key(gguf, key, GGUF_TYPE_BOOL));
}

int require_array(gguf_context * gguf, const char * key, gguf_type element_type) {
    const int index = require_key(gguf, key, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(gguf, index) != element_type) {
        throw std::runtime_error(std::string("invalid StarVLA GGUF array element type: ") + key);
    }
    return index;
}

std::vector<std::string> require_string_array(gguf_context * gguf, const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_STRING);
    const size_t count = gguf_get_arr_n(gguf, index);
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(gguf_get_arr_str(gguf, index, i));
    }
    return result;
}

std::vector<int32_t> require_i32_array(gguf_context * gguf, const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_INT32);
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(gguf, index));
    if (data == nullptr && count != 0) {
        throw std::runtime_error(std::string("missing StarVLA GGUF array data: ") + key);
    }
    return count == 0 ? std::vector<int32_t>() : std::vector<int32_t>(data, data + count);
}

std::vector<float> require_f32_array(gguf_context * gguf, const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_FLOAT32);
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto * data = static_cast<const float *>(gguf_get_arr_data(gguf, index));
    if (data == nullptr && count != 0) {
        throw std::runtime_error(std::string("missing StarVLA GGUF array data: ") + key);
    }
    return count == 0 ? std::vector<float>() : std::vector<float>(data, data + count);
}

std::vector<uint8_t> require_bool_array(gguf_context * gguf, const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_BOOL);
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto * data = static_cast<const uint8_t *>(gguf_get_arr_data(gguf, index));
    if (data == nullptr && count != 0) {
        throw std::runtime_error(std::string("missing StarVLA GGUF array data: ") + key);
    }
    std::vector<uint8_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = data[i] != 0 ? 1 : 0;
    }
    return result;
}

std::string profile_key(int profile_index, const char * suffix) {
    return "starvla.normalization.profile." + std::to_string(profile_index) + "." + suffix;
}

bool has_shape(const ggml_tensor * tensor, std::initializer_list<int64_t> expected) {
    if (tensor == nullptr || static_cast<size_t>(ggml_n_dims(tensor)) != expected.size()) {
        return false;
    }
    size_t dimension = 0;
    for (const int64_t value : expected) {
        if (tensor->ne[dimension++] != value) {
            return false;
        }
    }
    return true;
}

const char * mode_name(backend_mode mode) {
    switch (mode) {
    case backend_mode::cpu:
        return "cpu";
    case backend_mode::cuda:
        return "cuda";
    case backend_mode::metal:
        return "metal";
    }
    return "unknown";
}

std::vector<int32_t> integer_range(int first, int count) {
    std::vector<int32_t> result(static_cast<size_t>(count));
    for (int index = 0; index < count; ++index) {
        result[static_cast<size_t>(index)] = first + index;
    }
    return result;
}

class PIV3GGUFLoader final : public gguf_loader {
  public:
    PIV3GGUFLoader(PIV3PolicyConfig & config, PIV3Weights & weights) : config_(config), weights_(weights) {}

  protected:
    bool parse_metadata(gguf_context * gguf) override {
        if (require_string(gguf, "general.architecture") != "starvla-policy" ||
            require_i32(gguf, "starvla.schema_version") != 1 ||
            require_string(gguf, "starvla.framework") != "pi_v3") {
            throw std::runtime_error("GGUF is not a supported StarVLA PI-v3 policy");
        }

        config_.backbone_arch = require_string(gguf, "starvla.backbone.arch");
        if (config_.backbone_arch != "qwen3_vl") {
            throw std::runtime_error("StarVLA PI-v3 requires a Qwen3-VL backbone");
        }
        config_.bundle_uuid = require_string(gguf, "starvla.bundle.uuid");
        config_.text_filename = require_string(gguf, "starvla.component.text.filename");
        config_.mmproj_filename = require_string(gguf, "starvla.component.mmproj.filename");
        if (config_.bundle_uuid.empty() || config_.text_filename.empty() ||
            config_.mmproj_filename.empty()) {
            throw std::runtime_error("StarVLA PI-v3 bundle metadata is incomplete");
        }

        config_.qwen_hidden_dim = require_i32(gguf, "starvla.qwen.hidden_size");
        const int embedding_key = gguf_find_key(gguf, "starvla.qwen.input_embedding_size");
        config_.qwen_input_embedding_dim =
            embedding_key < 0 ? 4 * config_.qwen_hidden_dim
                              : require_i32(gguf, "starvla.qwen.input_embedding_size");
        config_.qwen_layer_count = require_i32(gguf, "starvla.qwen.layer_count");
        config_.qwen_vocab_size = require_i32(gguf, "starvla.qwen.vocab_size");
        config_.cot_template = require_string(gguf, "starvla.prompt.cot_template");

        config_.image_count = require_i32(gguf, "starvla.image.count");
        config_.image_names = require_string_array(gguf, "starvla.image.names");
        config_.image_processor_min_pixels =
            require_i32(gguf, "starvla.image.processor_min_pixels");
        config_.image_processor_max_pixels =
            require_i32(gguf, "starvla.image.processor_max_pixels");
        config_.image_patch_size = require_i32(gguf, "starvla.image.patch_size");
        config_.image_spatial_merge_size =
            require_i32(gguf, "starvla.image.spatial_merge_size");
        config_.image_min_token_count =
            require_i32(gguf, "starvla.image.min_token_count");
        config_.image_max_token_count =
            require_i32(gguf, "starvla.image.max_token_count");

        config_.dit_width = require_i32(gguf, "starvla.pi_v3.dit_width");
        config_.block_count = require_i32(gguf, "starvla.pi_v3.block_count");
        config_.projector_count = require_i32(gguf, "starvla.pi_v3.projector_count");
        config_.attention_head_count =
            require_i32(gguf, "starvla.pi_v3.attention_head_count");
        config_.attention_head_dim =
            require_i32(gguf, "starvla.pi_v3.attention_head_dim");
        config_.feed_forward_dim =
            require_i32(gguf, "starvla.pi_v3.feed_forward_dim");
        config_.mlp_hidden_dim =
            require_i32(gguf, "starvla.pi_v3.mlp_hidden_dimension");
        config_.future_token_count =
            require_i32(gguf, "starvla.pi_v3.future_token_count");
        config_.action_position_count =
            require_i32(gguf, "starvla.pi_v3.action_position_count");
        config_.no_state_sequence_length =
            require_i32(gguf, "starvla.pi_v3.no_state_sequence_length");
        config_.timestep_projection_dim =
            require_i32(gguf, "starvla.pi_v3.timestep_projection_dim");
        config_.num_timestep_buckets =
            require_i32(gguf, "starvla.pi_v3.num_timestep_buckets");
        config_.num_inference_timesteps =
            require_i32(gguf, "starvla.pi_v3.num_inference_timesteps");
        config_.ada_norm_epsilon = require_f32(gguf, "starvla.pi_v3.ada_norm_epsilon");
        config_.projector_norm_epsilon =
            require_f32(gguf, "starvla.pi_v3.projector_norm_epsilon");
        config_.euler_dt = require_f32(gguf, "starvla.pi_v3.euler_dt");
        config_.action_dim = require_i32(gguf, "starvla.action.dimension");
        config_.horizon = require_i32(gguf, "starvla.action.horizon");

        const bool valid =
            config_.qwen_hidden_dim > 0 && config_.qwen_input_embedding_dim > 0 &&
            config_.qwen_layer_count == kReleasedLayerCount && config_.qwen_vocab_size > 0 &&
            !config_.cot_template.empty() && config_.image_count > 0 &&
            config_.image_names.size() == static_cast<size_t>(config_.image_count) &&
            config_.image_processor_min_pixels > 0 &&
            config_.image_processor_max_pixels >= config_.image_processor_min_pixels &&
            config_.image_patch_size > 0 && config_.image_spatial_merge_size > 0 &&
            config_.image_min_token_count > 0 &&
            config_.image_max_token_count >= config_.image_min_token_count &&
            config_.dit_width > 0 && config_.block_count == kReleasedLayerCount &&
            config_.projector_count == config_.block_count &&
            config_.attention_head_count > 0 && config_.attention_head_dim > 0 &&
            config_.attention_head_count * config_.attention_head_dim == config_.dit_width &&
            config_.feed_forward_dim > 0 && config_.mlp_hidden_dim > 0 &&
            config_.action_dim > 0 && config_.horizon > 0 &&
            config_.future_token_count > 0 &&
            config_.action_position_count >= config_.horizon &&
            config_.no_state_sequence_length ==
                config_.future_token_count + config_.horizon &&
            config_.timestep_projection_dim >= 4 &&
            config_.timestep_projection_dim % 2 == 0 &&
            config_.num_timestep_buckets > 0 && config_.num_inference_timesteps == 4 &&
            config_.ada_norm_epsilon > 0.0f && config_.projector_norm_epsilon > 0.0f &&
            config_.euler_dt > 0.0f;
        if (!valid) {
            throw std::runtime_error("StarVLA PI-v3 metadata has incompatible dimensions");
        }

        config_.qwen_hidden_tuple_indices = integer_range(1, config_.qwen_layer_count);
        config_.timestep_ids.resize(static_cast<size_t>(config_.num_inference_timesteps));
        for (int step = 0; step < config_.num_inference_timesteps; ++step) {
            config_.timestep_ids[static_cast<size_t>(step)] =
                step * config_.num_timestep_buckets / config_.num_inference_timesteps;
        }

        NormalizationConfig & normalization = config_.normalization;
        normalization.clip_actions = require_bool(gguf, "starvla.normalization.clip_actions");
        normalization.binary_threshold =
            require_f32(gguf, "starvla.normalization.binary_threshold");
        normalization.binary_comparison =
            require_string(gguf, "starvla.normalization.binary_comparison");
        normalization.continuous_dimensions =
            require_i32_array(gguf, "starvla.action.continuous_dimensions");
        normalization.binary_dimensions =
            require_i32_array(gguf, "starvla.action.binary_dimensions");
        const int profile_count =
            require_i32(gguf, "starvla.normalization.profile_count");
        const std::vector<std::string> keys =
            require_string_array(gguf, "starvla.normalization.profile_keys");
        if (profile_count <= 0 || keys.size() != static_cast<size_t>(profile_count)) {
            throw std::runtime_error("StarVLA PI-v3 normalization profiles are inconsistent");
        }
        normalization.profiles.clear();
        normalization.profiles.reserve(static_cast<size_t>(profile_count));
        for (int index = 0; index < profile_count; ++index) {
            NormalizationProfile profile;
            profile.key = require_string(gguf, profile_key(index, "key").c_str());
            profile.action_q01 =
                require_f32_array(gguf, profile_key(index, "action_q01").c_str());
            profile.action_q99 =
                require_f32_array(gguf, profile_key(index, "action_q99").c_str());
            profile.action_mask =
                require_bool_array(gguf, profile_key(index, "action_mask").c_str());
            if (profile.key != keys[static_cast<size_t>(index)]) {
                throw std::runtime_error("StarVLA PI-v3 normalization profile order is inconsistent");
            }
            normalization.profiles.push_back(std::move(profile));
        }
        std::string normalization_error;
        if (!validate_normalization_config(normalization, config_.action_dim,
                                           normalization_error)) {
            throw std::runtime_error(normalization_error);
        }
        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        auto bind = [&](ggml_tensor *& destination, const std::string & name) {
            destination = require_tensor(ctx_data, name);
        };
        bind(weights_.timestep_input_weight, "starvla.policy.pi_v3.timestep.input.weight");
        bind(weights_.timestep_input_bias, "starvla.policy.pi_v3.timestep.input.bias");
        bind(weights_.timestep_output_weight, "starvla.policy.pi_v3.timestep.output.weight");
        bind(weights_.timestep_output_bias, "starvla.policy.pi_v3.timestep.output.bias");
        weights_.blocks.clear();
        weights_.blocks.reserve(static_cast<size_t>(config_.block_count));
        for (int block = 0; block < config_.block_count; ++block) {
            const std::string prefix = "starvla.policy.pi_v3.block." + std::to_string(block) + ".";
            PIV3BlockWeights current;
            bind(current.ada_norm_weight, prefix + "ada_norm.weight");
            bind(current.ada_norm_bias, prefix + "ada_norm.bias");
            bind(current.query_weight, prefix + "attention.query.weight");
            bind(current.query_bias, prefix + "attention.query.bias");
            bind(current.key_weight, prefix + "attention.key.weight");
            bind(current.key_bias, prefix + "attention.key.bias");
            bind(current.value_weight, prefix + "attention.value.weight");
            bind(current.value_bias, prefix + "attention.value.bias");
            bind(current.attention_output_weight, prefix + "attention.output.weight");
            bind(current.attention_output_bias, prefix + "attention.output.bias");
            bind(current.feed_forward_input_weight, prefix + "feed_forward.input.weight");
            bind(current.feed_forward_input_bias, prefix + "feed_forward.input.bias");
            bind(current.feed_forward_output_weight, prefix + "feed_forward.output.weight");
            bind(current.feed_forward_output_bias, prefix + "feed_forward.output.bias");
            weights_.blocks.push_back(current);
        }
        weights_.projectors.clear();
        weights_.projectors.reserve(static_cast<size_t>(config_.projector_count));
        for (int projector = 0; projector < config_.projector_count; ++projector) {
            const std::string prefix = "starvla.policy.pi_v3.projector." +
                                       std::to_string(projector) + ".";
            PIV3ProjectorWeights current;
            bind(current.norm_weight, prefix + "norm.weight");
            bind(current.norm_bias, prefix + "norm.bias");
            bind(current.projection_weight, prefix + "projection.weight");
            bind(current.projection_bias, prefix + "projection.bias");
            weights_.projectors.push_back(current);
        }
        bind(weights_.action_input_weight, "starvla.policy.pi_v3.action.input.weight");
        bind(weights_.action_input_bias, "starvla.policy.pi_v3.action.input.bias");
        bind(weights_.action_time_mix_weight, "starvla.policy.pi_v3.action.time_mix.weight");
        bind(weights_.action_time_mix_bias, "starvla.policy.pi_v3.action.time_mix.bias");
        bind(weights_.action_output_weight, "starvla.policy.pi_v3.action.output.weight");
        bind(weights_.action_output_bias, "starvla.policy.pi_v3.action.output.bias");
        bind(weights_.velocity_input_weight, "starvla.policy.pi_v3.velocity.input.weight");
        bind(weights_.velocity_input_bias, "starvla.policy.pi_v3.velocity.input.bias");
        bind(weights_.velocity_output_weight, "starvla.policy.pi_v3.velocity.output.weight");
        bind(weights_.velocity_output_bias, "starvla.policy.pi_v3.velocity.output.bias");
        bind(weights_.future_tokens, "starvla.policy.pi_v3.future_tokens.weight");
        bind(weights_.action_position, "starvla.policy.pi_v3.action_position.weight");

        const int width = config_.dit_width;
        if (!has_shape(weights_.timestep_input_weight, {config_.timestep_projection_dim, width}) ||
            !has_shape(weights_.timestep_input_bias, {width}) ||
            !has_shape(weights_.timestep_output_weight, {width, width}) ||
            !has_shape(weights_.timestep_output_bias, {width}) ||
            !has_shape(weights_.action_input_weight, {config_.action_dim, width}) ||
            !has_shape(weights_.action_input_bias, {width}) ||
            !has_shape(weights_.action_time_mix_weight, {2 * width, width}) ||
            !has_shape(weights_.action_time_mix_bias, {width}) ||
            !has_shape(weights_.action_output_weight, {width, width}) ||
            !has_shape(weights_.action_output_bias, {width}) ||
            !has_shape(weights_.velocity_input_weight, {width, config_.mlp_hidden_dim}) ||
            !has_shape(weights_.velocity_input_bias, {config_.mlp_hidden_dim}) ||
            !has_shape(weights_.velocity_output_weight, {config_.mlp_hidden_dim, config_.action_dim}) ||
            !has_shape(weights_.velocity_output_bias, {config_.action_dim}) ||
            !has_shape(weights_.future_tokens, {width, config_.future_token_count}) ||
            !has_shape(weights_.action_position, {width, config_.action_position_count})) {
            throw std::runtime_error("StarVLA PI_v3 non-block tensor has an incompatible ggml shape");
        }
        for (const PIV3BlockWeights & current : weights_.blocks) {
            if (!has_shape(current.ada_norm_weight, {width, 2 * width}) ||
                !has_shape(current.ada_norm_bias, {2 * width}) ||
                !has_shape(current.query_weight, {width, width}) ||
                !has_shape(current.query_bias, {width}) ||
                !has_shape(current.key_weight, {width, width}) ||
                !has_shape(current.key_bias, {width}) ||
                !has_shape(current.value_weight, {width, width}) ||
                !has_shape(current.value_bias, {width}) ||
                !has_shape(current.attention_output_weight, {width, width}) ||
                !has_shape(current.attention_output_bias, {width}) ||
                !has_shape(current.feed_forward_input_weight, {width, config_.feed_forward_dim}) ||
                !has_shape(current.feed_forward_input_bias, {config_.feed_forward_dim}) ||
                !has_shape(current.feed_forward_output_weight, {config_.feed_forward_dim, width}) ||
                !has_shape(current.feed_forward_output_bias, {width})) {
                throw std::runtime_error("StarVLA PI_v3 transformer block tensor has an incompatible ggml shape");
            }
        }
        for (const PIV3ProjectorWeights & current : weights_.projectors) {
            if (!has_shape(current.norm_weight, {config_.qwen_hidden_dim}) ||
                !has_shape(current.norm_bias, {config_.qwen_hidden_dim}) ||
                !has_shape(current.projection_weight, {config_.qwen_hidden_dim, width}) ||
                !has_shape(current.projection_bias, {width})) {
                throw std::runtime_error("StarVLA PI_v3 projector tensor has an incompatible ggml shape");
            }
        }
        return true;
    }

  private:
    PIV3PolicyConfig & config_;
    PIV3Weights & weights_;
};

std::vector<float> timestep_projection_table(const PIV3PolicyConfig & config) {
    const int dim = config.timestep_projection_dim;
    const int half = dim / 2;
    const float denominator = static_cast<float>(half - 1);
    std::vector<float> table(static_cast<size_t>(dim) * 4, 0.0f);
    for (int step = 0; step < 4; ++step) {
        const float timestep = static_cast<float>(config.timestep_ids[static_cast<size_t>(step)]);
        float * row = table.data() + static_cast<size_t>(step) * dim;
        for (int index = 0; index < half; ++index) {
            const float frequency = std::exp(-std::log(10000.0f) * static_cast<float>(index) / denominator);
            const float angle = timestep * frequency;
            row[index] = std::cos(angle);
            row[index + half] = std::sin(angle);
        }
    }
    return table;
}

std::vector<float> action_time_table(const PIV3PolicyConfig & config) {
    const int dim = config.dit_width;
    const int half = dim / 2;
    const float denominator = static_cast<float>(half);
    std::vector<float> table(static_cast<size_t>(dim) * 4, 0.0f);
    for (int step = 0; step < 4; ++step) {
        const float timestep = static_cast<float>(config.timestep_ids[static_cast<size_t>(step)]);
        float * row = table.data() + static_cast<size_t>(step) * dim;
        for (int index = 0; index < half; ++index) {
            const float frequency = std::exp(-std::log(10000.0f) * static_cast<float>(index) / denominator);
            const float angle = timestep * frequency;
            row[index] = std::sin(angle);
            row[index + half] = std::cos(angle);
        }
    }
    return table;
}

} // namespace

struct PIV3Policy::Impl {
    PIV3PolicyConfig config;
    PIV3Weights weights;
    gguf_load_result loaded;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t scheduler = nullptr;
    backend_buft_policy buft_policy;
    backend_mode mode = backend_mode::cpu;
    int n_threads = 0;
    int verbosity = 0;
    ggml_context * graph_context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden_input = nullptr;
    ggml_tensor * cross_mask_input = nullptr;
    ggml_tensor * noise_input = nullptr;
    ggml_tensor * timestep_projection_input = nullptr;
    ggml_tensor * action_time_input = nullptr;
    ggml_tensor * scalar_one_input = nullptr;
    ggml_tensor * output = nullptr;
    size_t conditioning_token_count = 0;
    std::vector<float> timestep_table;
    std::vector<float> action_table;
    ~Impl() {
        clear_graph();
        if (scheduler != nullptr) {
            ggml_backend_sched_synchronize(scheduler);
            ggml_backend_sched_free(scheduler);
            scheduler = nullptr;
        }
        if (loaded.model_buffer != nullptr) {
            ggml_backend_buffer_free(loaded.model_buffer);
            loaded.model_buffer = nullptr;
        }
        if (loaded.ctx_data != nullptr) {
            ggml_free(loaded.ctx_data);
            loaded.ctx_data = nullptr;
        }
        if (loaded.gguf != nullptr) {
            gguf_free(loaded.gguf);
            loaded.gguf = nullptr;
        }
        for (ggml_backend_t backend : backends) {
            if (backend != nullptr) {
                ggml_backend_free(backend);
            }
        }
        backends.clear();
        backend_cpu = nullptr;
    }

    void clear_graph() {
        if (scheduler != nullptr) {
            ggml_backend_sched_synchronize(scheduler);
            ggml_backend_sched_reset(scheduler);
        }
        if (graph_context != nullptr) {
            ggml_free(graph_context);
            graph_context = nullptr;
        }
        graph = nullptr;
        hidden_input = nullptr;
        cross_mask_input = nullptr;
        noise_input = nullptr;
        timestep_projection_input = nullptr;
        action_time_input = nullptr;
        scalar_one_input = nullptr;
        output = nullptr;
        conditioning_token_count = 0;
    }

    void build_graph(size_t token_count) {
        clear_graph();
        if (token_count == 0 || token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("invalid StarVLA PI_v3 conditioning token count");
        }

        ggml_init_params params{};
        params.mem_size = kGraphSize * ggml_tensor_overhead() +
                          ggml_graph_overhead_custom(kGraphSize, false);
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph_context = ggml_init(params);
        if (graph_context == nullptr) {
            throw std::runtime_error("failed to initialize StarVLA PI_v3 graph context");
        }

        const int width = config.dit_width;
        const int heads = config.attention_head_count;
        const int head_dim = config.attention_head_dim;
        const int sequence_length = config.no_state_sequence_length;
        const int mask_queries = GGML_PAD(sequence_length, kKQMaskPad);

        hidden_input = ggml_new_tensor_3d(graph_context, GGML_TYPE_F32, config.qwen_hidden_dim,
                                          static_cast<int64_t>(token_count), config.qwen_layer_count);
        cross_mask_input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32,
                                              static_cast<int64_t>(token_count), mask_queries);
        noise_input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32,
                                         config.action_dim, config.horizon);
        timestep_projection_input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32,
                                                       config.timestep_projection_dim, 4);
        action_time_input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, width, 4);
        scalar_one_input = ggml_new_tensor_1d(graph_context, GGML_TYPE_F32, 1);
        if (hidden_input == nullptr || cross_mask_input == nullptr || noise_input == nullptr ||
            timestep_projection_input == nullptr || action_time_input == nullptr ||
            scalar_one_input == nullptr) {
            throw std::runtime_error("failed to create StarVLA PI_v3 graph inputs");
        }
        ggml_set_name(hidden_input, "starvla_pi_v3_qwen_hidden_states");
        ggml_set_name(cross_mask_input, "starvla_pi_v3_qwen_attention_mask");
        ggml_set_name(noise_input, "starvla_pi_v3_initial_noise");
        ggml_set_name(timestep_projection_input, "starvla_pi_v3_timestep_projection_table");
        ggml_set_name(action_time_input, "starvla_pi_v3_action_time_table");
        ggml_set_name(scalar_one_input, "starvla_pi_v3_scalar_one");
        ggml_set_input(hidden_input);
        ggml_set_input(cross_mask_input);
        ggml_set_input(noise_input);
        ggml_set_input(timestep_projection_input);
        ggml_set_input(action_time_input);
        ggml_set_input(scalar_one_input);

        auto f32 = [&](ggml_tensor * tensor) {
            return tensor->type == GGML_TYPE_F32 ? tensor :
                                                   ggml_cast(graph_context, tensor, GGML_TYPE_F32);
        };
        auto bf16_roundtrip = [&](ggml_tensor * tensor) {
            return ggml_cast(graph_context,
                             ggml_cast(graph_context, tensor, GGML_TYPE_BF16),
                             GGML_TYPE_F32);
        };
        auto linear = [&](ggml_tensor * value, ggml_tensor * weight, ggml_tensor * bias) {
            ggml_tensor * projected = ggml_mul_mat(graph_context, weight, value);
            ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
            return ggml_add(graph_context, projected, f32(bias));
        };
        auto projector_linear = [&](ggml_tensor * value, ggml_tensor * weight,
                                    ggml_tensor * bias) {
            ggml_tensor * bf16_value = ggml_cast(graph_context, value, GGML_TYPE_BF16);
            ggml_tensor * bf16_weight = ggml_cast(graph_context, weight, GGML_TYPE_BF16);
            ggml_tensor * projected =
                ggml_mul_mat(graph_context, bf16_weight, bf16_value);
            ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
            projected = ggml_add(graph_context, projected, bf16_roundtrip(bias));
            return bf16_roundtrip(projected);
        };
        auto ada_norm = [&](ggml_tensor * value, ggml_tensor * temb,
                            const PIV3BlockWeights & block) {
            ggml_tensor * modulation = linear(ggml_silu(graph_context, temb),
                                              block.ada_norm_weight, block.ada_norm_bias);
            ggml_tensor * scale = ggml_view_1d(graph_context, modulation, width, 0);
            ggml_tensor * shift = ggml_view_1d(graph_context, modulation, width,
                                               static_cast<size_t>(width) * sizeof(float));
            ggml_tensor * normalized = ggml_norm(graph_context, value, config.ada_norm_epsilon);
            return ggml_add(graph_context,
                            ggml_mul(graph_context, normalized,
                                     ggml_add(graph_context, scale, scalar_one_input)),
                            shift);
        };
        auto attention = [&](ggml_tensor * query_source, ggml_tensor * key_value_source,
                             const PIV3BlockWeights & block) {
            const int64_t query_count = query_source->ne[1];
            const int64_t key_value_count = key_value_source->ne[1];
            ggml_tensor * query = linear(query_source, block.query_weight, block.query_bias);
            ggml_tensor * key = linear(key_value_source, block.key_weight, block.key_bias);
            ggml_tensor * value = linear(key_value_source, block.value_weight, block.value_bias);
            query = ggml_reshape_3d(graph_context, query, head_dim, heads, query_count);
            key = ggml_reshape_3d(graph_context, key, head_dim, heads, key_value_count);
            value = ggml_reshape_3d(graph_context, value, head_dim, heads, key_value_count);
            query = ggml_permute(graph_context, query, 0, 2, 1, 3);
            key = ggml_permute(graph_context, key, 0, 2, 1, 3);
            value = ggml_cont(graph_context, ggml_permute(graph_context, value, 1, 2, 0, 3));
            ggml_tensor * scores = ggml_mul_mat(graph_context, key, query);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores = ggml_soft_max_ext(graph_context, scores, cross_mask_input,
                                       1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
            ggml_tensor * attended = ggml_mul_mat(graph_context, value, scores);
            ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
            attended = ggml_permute(graph_context, attended, 0, 2, 1, 3);
            attended = ggml_cont_2d(graph_context, attended, width, query_count);
            return linear(attended, block.attention_output_weight, block.attention_output_bias);
        };

        std::vector<ggml_tensor *> projected_hidden_states;
        projected_hidden_states.reserve(static_cast<size_t>(config.projector_count));
        for (int layer = 0; layer < config.projector_count; ++layer) {
            const PIV3ProjectorWeights & projector = weights.projectors[static_cast<size_t>(layer)];
            ggml_tensor * layer_hidden = ggml_view_2d(
                graph_context, hidden_input, config.qwen_hidden_dim,
                static_cast<int64_t>(token_count), hidden_input->nb[1],
                static_cast<size_t>(layer) * hidden_input->nb[2]);
            layer_hidden = bf16_roundtrip(layer_hidden);
            layer_hidden = ggml_norm(graph_context, layer_hidden, config.projector_norm_epsilon);
            layer_hidden = ggml_mul(graph_context, layer_hidden, f32(projector.norm_weight));
            layer_hidden = ggml_add(graph_context, layer_hidden, f32(projector.norm_bias));
            ggml_tensor * projected = projector_linear(
                layer_hidden, projector.projection_weight,
                projector.projection_bias);
            projected_hidden_states.push_back(projected);
        }

        ggml_tensor * future = f32(weights.future_tokens);
        ggml_tensor * position_view = ggml_view_2d(
            graph_context, weights.action_position, width, config.horizon,
            weights.action_position->nb[1], 0);
        ggml_tensor * position = f32(position_view);
        // Qwen/projector inference and torch.randn run at BF16 in the released
        // script. The action head then enters CUDA autocast(float32).
        ggml_tensor * actions = bf16_roundtrip(noise_input);

        for (int step = 0; step < 4; ++step) {
            ggml_tensor * timestep_projection = ggml_view_1d(
                graph_context, timestep_projection_input, config.timestep_projection_dim,
                static_cast<size_t>(step) * config.timestep_projection_dim * sizeof(float));
            ggml_tensor * temb = linear(timestep_projection, weights.timestep_input_weight,
                                        weights.timestep_input_bias);
            temb = ggml_silu(graph_context, temb);
            temb = linear(temb, weights.timestep_output_weight, weights.timestep_output_bias);

            ggml_tensor * action_features = linear(actions, weights.action_input_weight,
                                                    weights.action_input_bias);
            ggml_tensor * action_time = ggml_view_1d(
                graph_context, action_time_input, width,
                static_cast<size_t>(step) * width * sizeof(float));
            action_time = ggml_repeat(graph_context, action_time, action_features);
            action_features = ggml_concat(graph_context, action_features, action_time, 0);
            action_features = linear(action_features, weights.action_time_mix_weight,
                                     weights.action_time_mix_bias);
            action_features = ggml_silu(graph_context, action_features);
            action_features = linear(action_features, weights.action_output_weight,
                                     weights.action_output_bias);
            action_features = ggml_add(graph_context, action_features, position);

            ggml_tensor * hidden = ggml_concat(graph_context, future, action_features, 1);
            for (int block_index = 0; block_index < config.block_count; ++block_index) {
                const PIV3BlockWeights & block = weights.blocks[static_cast<size_t>(block_index)];
                ggml_tensor * normalized = ada_norm(hidden, temb, block);
                ggml_tensor * attended = attention(
                    normalized, projected_hidden_states[static_cast<size_t>(block_index)], block);
                hidden = ggml_add(graph_context, hidden, attended);
                ggml_tensor * ff = ggml_norm(graph_context, hidden, config.ada_norm_epsilon);
                ff = linear(ff, block.feed_forward_input_weight, block.feed_forward_input_bias);
                ff = ggml_gelu(graph_context, ff);
                ff = linear(ff, block.feed_forward_output_weight, block.feed_forward_output_bias);
                hidden = ggml_add(graph_context, hidden, ff);
            }

            // The released legacy sampler calls DiT with return_pre_output=true.
            // norm_out/proj_out_1/proj_out_2 are therefore intentionally inactive.
            hidden = ggml_relu(graph_context,
                               linear(hidden, weights.velocity_input_weight,
                                      weights.velocity_input_bias));
            hidden = linear(hidden, weights.velocity_output_weight,
                            weights.velocity_output_bias);
            ggml_tensor * velocity = ggml_view_2d(
                graph_context, hidden, config.action_dim, config.horizon, hidden->nb[1],
                static_cast<size_t>(config.future_token_count) * hidden->nb[1]);
            actions = ggml_add(graph_context, actions,
                               ggml_scale(graph_context, velocity, config.euler_dt));
        }

        output = actions;
        ggml_set_name(output, "starvla_pi_v3_normalized_actions");
        ggml_set_output(output);
        graph = ggml_new_graph_custom(graph_context, kGraphSize, false);
        if (graph == nullptr) {
            throw std::runtime_error("failed to create StarVLA PI_v3 graph");
        }
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("failed to allocate StarVLA PI_v3 graph");
        }

        conditioning_token_count = token_count;
    }
};

PIV3Policy::PIV3Policy(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PIV3Policy::~PIV3Policy() = default;

std::unique_ptr<PIV3Policy> PIV3Policy::load(const std::string & path, int n_threads, int verbosity,
                                             std::string & error) {
    error.clear();
    if (path.empty()) {
        error = "StarVLA PI_v3 policy path is required";
        return nullptr;
    }

    std::unique_ptr<Impl> impl(new Impl());
    impl->n_threads = n_threads;
    impl->verbosity = verbosity;
    try {
        backend_scheduler_config scheduler_config;
        scheduler_config.max_nodes = static_cast<int>(kGraphSize);
        scheduler_config.parallel = false;
        scheduler_config.op_offload = true;
        backend_loader backend;
        if (!backend.load(impl->backend_cpu, impl->backends, impl->scheduler,
                          impl->buft_policy, true, scheduler_config, verbosity)) {
            error = "failed to initialize StarVLA PI_v3 backend: " + backend.error();
            return nullptr;
        }
        impl->mode = backend.mode();

        PIV3GGUFLoader loader(impl->config, impl->weights);
        if (!loader.load(path.c_str(), impl->buft_policy.model_buft, impl->loaded, verbosity)) {
            error = loader.error();
            return nullptr;
        }
        if (impl->loaded.ctx_data == nullptr || impl->loaded.model_buffer == nullptr) {
            error = "StarVLA PI_v3 policy GGUF has no tensors";
            return nullptr;
        }
        ggml_backend_buffer_set_usage(impl->loaded.model_buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        impl->timestep_table = timestep_projection_table(impl->config);
        impl->action_table = action_time_table(impl->config);
        if (verbosity >= 1) {
            std::fprintf(stderr,
                         "%s: backend=%s qwen=%d width=%d layers=%d horizon=%d action_dim=%d profiles=%zu\n",
                         __func__, mode_name(impl->mode), impl->config.qwen_hidden_dim,
                         impl->config.dit_width, impl->config.block_count, impl->config.horizon,
                         impl->config.action_dim, impl->config.normalization.profiles.size());
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
    return std::unique_ptr<PIV3Policy>(new PIV3Policy(std::move(impl)));
}

bool PIV3Policy::evaluate(const float * qwen_hidden_states, size_t hidden_element_count,
                          const uint8_t * qwen_attention_mask, size_t mask_element_count,
                          const float * initial_noise, size_t noise_element_count,
                          std::vector<float> & normalized_actions, std::string & error) {
    return evaluate_internal(qwen_hidden_states, hidden_element_count,
                             qwen_attention_mask, mask_element_count,
                             initial_noise, noise_element_count,
                             normalized_actions, error);
}

bool PIV3Policy::evaluate_internal(
    const float * qwen_hidden_states, size_t hidden_element_count,
    const uint8_t * qwen_attention_mask, size_t mask_element_count,
    const float * initial_noise, size_t noise_element_count,
    std::vector<float> & normalized_actions,
    std::string & error) {
    normalized_actions.clear();
    error.clear();
    if (impl_ == nullptr || impl_->scheduler == nullptr) {
        error = "StarVLA PI_v3 policy is not initialized";
        return false;
    }
    const size_t hidden_width = static_cast<size_t>(impl_->config.qwen_hidden_dim);
    const size_t layer_count = static_cast<size_t>(impl_->config.qwen_layer_count);
    if (qwen_hidden_states == nullptr || qwen_attention_mask == nullptr || initial_noise == nullptr ||
        mask_element_count == 0 ||
        mask_element_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        mask_element_count > std::numeric_limits<size_t>::max() / hidden_width ||
        mask_element_count * hidden_width > std::numeric_limits<size_t>::max() / layer_count ||
        hidden_element_count != mask_element_count * hidden_width * layer_count) {
        error = "StarVLA PI_v3 layerwise Qwen conditioning tensor or attention mask has an incompatible shape";
        return false;
    }
    const size_t expected_noise =
        static_cast<size_t>(impl_->config.horizon) * impl_->config.action_dim;
    if (noise_element_count != expected_noise) {
        error = "StarVLA PI_v3 initial-noise tensor has an incompatible shape";
        return false;
    }
    if (std::any_of(qwen_hidden_states, qwen_hidden_states + hidden_element_count,
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(initial_noise, initial_noise + noise_element_count,
                    [](float value) { return !std::isfinite(value); })) {
        error = "StarVLA PI_v3 conditioning and initial noise must be finite";
        return false;
    }
    bool has_valid_token = false;
    for (size_t token = 0; token < mask_element_count; ++token) {
        if (qwen_attention_mask[token] > 1) {
            error = "StarVLA PI_v3 attention mask values must be zero or one";
            return false;
        }
        has_valid_token = has_valid_token || qwen_attention_mask[token] != 0;
    }
    if (!has_valid_token) {
        error = "StarVLA PI_v3 attention mask must contain at least one valid token";
        return false;
    }

    try {
        if (impl_->graph == nullptr ||
            impl_->conditioning_token_count != mask_element_count) {
            impl_->build_graph(mask_element_count);
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }

    const int query_count = impl_->config.no_state_sequence_length;
    const int padded_queries = GGML_PAD(query_count, kKQMaskPad);
    std::vector<float> additive_mask(mask_element_count * static_cast<size_t>(padded_queries),
                                     -std::numeric_limits<float>::infinity());
    for (int query = 0; query < query_count; ++query) {
        float * row = additive_mask.data() + static_cast<size_t>(query) * mask_element_count;
        for (size_t token = 0; token < mask_element_count; ++token) {
            row[token] = qwen_attention_mask[token] != 0
                             ? 0.0f
                             : -std::numeric_limits<float>::infinity();
        }
    }

    ggml_backend_tensor_set(impl_->hidden_input, qwen_hidden_states, 0,
                            hidden_element_count * sizeof(float));
    ggml_backend_tensor_set(impl_->cross_mask_input, additive_mask.data(), 0,
                            additive_mask.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->noise_input, initial_noise, 0,
                            noise_element_count * sizeof(float));
    ggml_backend_tensor_set(impl_->timestep_projection_input, impl_->timestep_table.data(), 0,
                            impl_->timestep_table.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->action_time_input, impl_->action_table.data(), 0,
                            impl_->action_table.size() * sizeof(float));
    const float one = 1.0f;
    ggml_backend_tensor_set(impl_->scalar_one_input, &one, 0, sizeof(one));
    set_backend_threads(impl_->backends, impl_->n_threads);
    if (ggml_backend_sched_graph_compute(impl_->scheduler, impl_->graph) != GGML_STATUS_SUCCESS) {
        error = "StarVLA PI_v3 graph compute failed";
        return false;
    }

    normalized_actions.resize(expected_noise);
    ggml_backend_tensor_get(impl_->output, normalized_actions.data(), 0,
                            expected_noise * sizeof(float));
    if (std::any_of(normalized_actions.begin(), normalized_actions.end(),
                    [](float value) { return !std::isfinite(value); })) {
        normalized_actions.clear();
        error = "StarVLA PI_v3 graph produced non-finite actions";
        return false;
    }
    return true;
}

bool PIV3Policy::unnormalize(const std::vector<float> & normalized_actions,
                             const std::string & profile_key_value,
                             std::vector<float> & actions, std::string & error) const {
    if (impl_ == nullptr) {
        actions.clear();
        error = "StarVLA PI_v3 policy is not initialized";
        return false;
    }
    return denormalize_actions(impl_->config.normalization, profile_key_value,
                               normalized_actions, impl_->config.horizon,
                               impl_->config.action_dim, actions, error);
}

const PIV3PolicyConfig & PIV3Policy::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("StarVLA PI_v3 policy is not initialized");
    }
    return impl_->config;
}

const char * PIV3Policy::backend_name() const {
    return impl_ != nullptr ? mode_name(impl_->mode) : "unknown";
}

} // namespace robotcpp::starvla
