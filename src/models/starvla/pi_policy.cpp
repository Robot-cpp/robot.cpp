#include "models/starvla/pi_policy.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "models/ggml_backend.h"
#include "models/gguf_loader.h"
#include "models/starvla/policy_gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::starvla {

namespace {

constexpr size_t kGraphSize = 16384;
struct PIBlockWeights {
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

struct PIWeights {
    ggml_tensor * timestep_input_weight = nullptr;
    ggml_tensor * timestep_input_bias = nullptr;
    ggml_tensor * timestep_output_weight = nullptr;
    ggml_tensor * timestep_output_bias = nullptr;
    std::vector<PIBlockWeights> blocks;
    ggml_tensor * state_input_weight = nullptr;
    ggml_tensor * state_input_bias = nullptr;
    ggml_tensor * state_output_weight = nullptr;
    ggml_tensor * state_output_bias = nullptr;
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

using detail::has_shape;
using detail::require_f32;
using detail::require_i32;
using detail::require_i32_array;
using detail::require_string;
using detail::require_string_array;

std::vector<int32_t> expected_hidden_tuple_indices(int qwen_layer_count,
                                                   int block_count) {
    std::vector<int32_t> result;
    result.reserve(static_cast<size_t>(block_count));
    const int first = qwen_layer_count + 1 - block_count;
    for (int index = first; index <= qwen_layer_count; ++index) {
        result.push_back(index);
    }
    return result;
}

class PIGGUFLoader final : public gguf_loader {
  public:
    PIGGUFLoader(PIPolicyConfig & config, PIWeights & weights)
        : config_(config), weights_(weights) {}

  protected:
    bool parse_metadata(gguf_context * gguf) override {
        if (require_string(gguf, "general.architecture") != "starvla-policy" ||
            require_i32(gguf, "starvla.schema_version") != 1 ||
            require_string(gguf, "starvla.framework") != "pi") {
            throw std::runtime_error("GGUF is not a supported StarVLA PI policy");
        }
        config_.backbone_arch = require_string(gguf, "starvla.backbone.arch");
        config_.bundle_uuid = require_string(gguf, "starvla.bundle.uuid");
        config_.text_filename =
            require_string(gguf, "starvla.component.text.filename");
        config_.mmproj_filename =
            require_string(gguf, "starvla.component.mmproj.filename");
        if (config_.backbone_arch != "qwen2_5_vl" ||
            config_.bundle_uuid.empty() || config_.text_filename.empty() ||
            config_.mmproj_filename.empty()) {
            throw std::runtime_error("StarVLA PI bundle metadata is incomplete");
        }

        config_.qwen_hidden_dim =
            require_i32(gguf, "starvla.qwen.hidden_size");
        config_.qwen_input_embedding_dim =
            require_i32(gguf, "starvla.qwen.input_embedding_size");
        config_.qwen_layer_count =
            require_i32(gguf, "starvla.qwen.layer_count");
        config_.qwen_vocab_size = require_i32(gguf, "starvla.qwen.vocab_size");
        config_.cot_template =
            require_string(gguf, "starvla.prompt.cot_template");
        config_.qwen_hidden_tuple_indices =
            require_i32_array(gguf, "starvla.conditioning.hidden_tuple_indices");

        config_.image_count = require_i32(gguf, "starvla.image.count");
        config_.image_names =
            require_string_array(gguf, "starvla.image.names");
        config_.image_framework_inference_pre_resize_width =
            require_i32(gguf, "starvla.image.framework_inference_pre_resize_width");
        config_.image_framework_inference_pre_resize_height =
            require_i32(gguf, "starvla.image.framework_inference_pre_resize_height");
        config_.image_processor_min_pixels =
            require_i32(gguf, "starvla.image.processor_min_pixels");
        config_.image_processor_max_pixels =
            require_i32(gguf, "starvla.image.processor_max_pixels");
        config_.image_patch_size =
            require_i32(gguf, "starvla.image.patch_size");
        config_.image_spatial_merge_size =
            require_i32(gguf, "starvla.image.spatial_merge_size");
        config_.image_min_token_count =
            require_i32(gguf, "starvla.image.min_token_count");
        config_.image_max_token_count =
            require_i32(gguf, "starvla.image.max_token_count");

        config_.dit_width = require_i32(gguf, "starvla.pi.dit_width");
        config_.block_count = require_i32(gguf, "starvla.pi.block_count");
        config_.attention_head_count =
            require_i32(gguf, "starvla.pi.attention_head_count");
        config_.attention_head_dim =
            require_i32(gguf, "starvla.pi.attention_head_dim");
        config_.cross_attention_dim =
            require_i32(gguf, "starvla.pi.cross_attention_dim");
        config_.feed_forward_dim =
            require_i32(gguf, "starvla.pi.feed_forward_dim");
        config_.mlp_hidden_dim =
            require_i32(gguf, "starvla.pi.mlp_hidden_dimension");
        config_.state_dim = require_i32(gguf, "starvla.state.dimension");
        config_.action_dim = require_i32(gguf, "starvla.action.dimension");
        config_.horizon = require_i32(gguf, "starvla.action.horizon");
        config_.state_token_count =
            require_i32(gguf, "starvla.pi.state_token_count");
        config_.future_token_count =
            require_i32(gguf, "starvla.pi.future_token_count");
        config_.action_position_count =
            require_i32(gguf, "starvla.pi.action_position_count");
        config_.timestep_projection_dim =
            require_i32(gguf, "starvla.pi.timestep_projection_dim");
        config_.num_inference_timesteps =
            require_i32(gguf, "starvla.pi.num_inference_timesteps");
        config_.ada_norm_epsilon =
            require_f32(gguf, "starvla.pi.ada_norm_epsilon");
        config_.euler_dt = require_f32(gguf, "starvla.pi.euler_dt");
        config_.timestep_ids =
            require_i32_array(gguf, "starvla.pi.timestep_ids");

        const std::vector<int32_t> expected_indices =
            expected_hidden_tuple_indices(config_.qwen_layer_count,
                                          config_.block_count);
        const bool dimensions_valid =
            config_.qwen_hidden_dim > 0 &&
            config_.qwen_input_embedding_dim == config_.qwen_hidden_dim &&
            config_.qwen_layer_count >= config_.block_count &&
            config_.qwen_vocab_size > 0 && config_.dit_width > 0 &&
            config_.dit_width % 2 == 0 && config_.block_count > 0 &&
            config_.attention_head_count > 0 &&
            config_.attention_head_dim > 0 &&
            config_.attention_head_count * config_.attention_head_dim ==
                config_.dit_width &&
            config_.cross_attention_dim == config_.qwen_hidden_dim &&
            config_.feed_forward_dim > 0 &&
            config_.mlp_hidden_dim > 0 && config_.state_dim > 0 &&
            config_.action_dim > 0 && config_.horizon > 0 &&
            config_.state_token_count == 1 &&
            config_.future_token_count > 0 &&
            config_.action_position_count >= config_.horizon &&
            config_.timestep_projection_dim >= 4 &&
            config_.timestep_projection_dim % 2 == 0 &&
            config_.num_inference_timesteps > 0 &&
            config_.timestep_ids.size() ==
                static_cast<size_t>(config_.num_inference_timesteps) &&
            std::isfinite(config_.ada_norm_epsilon) &&
            config_.ada_norm_epsilon > 0.0f && std::isfinite(config_.euler_dt) &&
            config_.euler_dt > 0.0f &&
            config_.qwen_hidden_tuple_indices == expected_indices &&
            config_.image_count > 0 &&
            config_.image_names.size() == static_cast<size_t>(config_.image_count) &&
            config_.image_framework_inference_pre_resize_width > 0 &&
            config_.image_framework_inference_pre_resize_height > 0 &&
            config_.image_processor_min_pixels > 0 &&
            config_.image_processor_max_pixels >=
                config_.image_processor_min_pixels &&
            config_.image_patch_size > 0 &&
            config_.image_spatial_merge_size > 0 &&
            config_.image_min_token_count > 0 &&
            config_.image_max_token_count >=
                config_.image_min_token_count &&
            !config_.cot_template.empty();
        if (!dimensions_valid) {
            throw std::runtime_error(
                "StarVLA PI dimensions, hidden taps, or sampler schedule are incompatible");
        }

        NormalizationConfig & normalization = config_.normalization;
        normalization = detail::require_normalization(gguf, config_.action_dim);
        if (!normalization.clip_actions ||
            normalization.binary_comparison != "ge") {
            throw std::runtime_error(
                "StarVLA PI normalization must clip actions and use "
                "binary comparison 'ge'");
        }
        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        auto bind = [&](ggml_tensor *& destination, const std::string & name) {
            destination = require_tensor(ctx_data, name);
        };
        bind(weights_.timestep_input_weight,
             "starvla.policy.pi.timestep.input.weight");
        bind(weights_.timestep_input_bias,
             "starvla.policy.pi.timestep.input.bias");
        bind(weights_.timestep_output_weight,
             "starvla.policy.pi.timestep.output.weight");
        bind(weights_.timestep_output_bias,
             "starvla.policy.pi.timestep.output.bias");
        weights_.blocks.clear();
        weights_.blocks.reserve(static_cast<size_t>(config_.block_count));
        for (int block = 0; block < config_.block_count; ++block) {
            const std::string prefix =
                "starvla.policy.pi.block." + std::to_string(block) + ".";
            PIBlockWeights current;
            bind(current.ada_norm_weight, prefix + "ada_norm.weight");
            bind(current.ada_norm_bias, prefix + "ada_norm.bias");
            bind(current.query_weight, prefix + "attention.query.weight");
            bind(current.query_bias, prefix + "attention.query.bias");
            bind(current.key_weight, prefix + "attention.key.weight");
            bind(current.key_bias, prefix + "attention.key.bias");
            bind(current.value_weight, prefix + "attention.value.weight");
            bind(current.value_bias, prefix + "attention.value.bias");
            bind(current.attention_output_weight,
                 prefix + "attention.output.weight");
            bind(current.attention_output_bias,
                 prefix + "attention.output.bias");
            bind(current.feed_forward_input_weight,
                 prefix + "feed_forward.input.weight");
            bind(current.feed_forward_input_bias,
                 prefix + "feed_forward.input.bias");
            bind(current.feed_forward_output_weight,
                 prefix + "feed_forward.output.weight");
            bind(current.feed_forward_output_bias,
                 prefix + "feed_forward.output.bias");
            weights_.blocks.push_back(current);
        }
        bind(weights_.state_input_weight,
             "starvla.policy.pi.state.input.weight");
        bind(weights_.state_input_bias, "starvla.policy.pi.state.input.bias");
        bind(weights_.state_output_weight,
             "starvla.policy.pi.state.output.weight");
        bind(weights_.state_output_bias, "starvla.policy.pi.state.output.bias");
        bind(weights_.action_input_weight,
             "starvla.policy.pi.action.input.weight");
        bind(weights_.action_input_bias, "starvla.policy.pi.action.input.bias");
        bind(weights_.action_time_mix_weight,
             "starvla.policy.pi.action.time_mix.weight");
        bind(weights_.action_time_mix_bias,
             "starvla.policy.pi.action.time_mix.bias");
        bind(weights_.action_output_weight,
             "starvla.policy.pi.action.output.weight");
        bind(weights_.action_output_bias,
             "starvla.policy.pi.action.output.bias");
        bind(weights_.velocity_input_weight,
             "starvla.policy.pi.velocity.input.weight");
        bind(weights_.velocity_input_bias,
             "starvla.policy.pi.velocity.input.bias");
        bind(weights_.velocity_output_weight,
             "starvla.policy.pi.velocity.output.weight");
        bind(weights_.velocity_output_bias,
             "starvla.policy.pi.velocity.output.bias");
        bind(weights_.future_tokens, "starvla.policy.pi.future_tokens.weight");
        bind(weights_.action_position,
             "starvla.policy.pi.action_position.weight");

        const int64_t width = config_.dit_width;
        if (!has_shape(weights_.timestep_input_weight,
                       {config_.timestep_projection_dim, width}) ||
            !has_shape(weights_.timestep_input_bias, {width}) ||
            !has_shape(weights_.timestep_output_weight, {width, width}) ||
            !has_shape(weights_.timestep_output_bias, {width}) ||
            !has_shape(weights_.state_input_weight,
                       {config_.state_dim, config_.mlp_hidden_dim}) ||
            !has_shape(weights_.state_input_bias, {config_.mlp_hidden_dim}) ||
            !has_shape(weights_.state_output_weight,
                       {config_.mlp_hidden_dim, width}) ||
            !has_shape(weights_.state_output_bias, {width}) ||
            !has_shape(weights_.action_input_weight,
                       {config_.action_dim, width}) ||
            !has_shape(weights_.action_input_bias, {width}) ||
            !has_shape(weights_.action_time_mix_weight,
                       {2 * width, width}) ||
            !has_shape(weights_.action_time_mix_bias, {width}) ||
            !has_shape(weights_.action_output_weight, {width, width}) ||
            !has_shape(weights_.action_output_bias, {width}) ||
            !has_shape(weights_.velocity_input_weight,
                       {width, config_.mlp_hidden_dim}) ||
            !has_shape(weights_.velocity_input_bias,
                       {config_.mlp_hidden_dim}) ||
            !has_shape(weights_.velocity_output_weight,
                       {config_.mlp_hidden_dim, config_.action_dim}) ||
            !has_shape(weights_.velocity_output_bias, {config_.action_dim}) ||
            !has_shape(weights_.future_tokens,
                       {width, config_.future_token_count}) ||
            !has_shape(weights_.action_position,
                       {width, config_.action_position_count})) {
            throw std::runtime_error(
                "StarVLA PI non-transformer tensor has an incompatible ggml shape");
        }
        for (const PIBlockWeights & block : weights_.blocks) {
            if (!has_shape(block.ada_norm_weight, {width, 2 * width}) ||
                !has_shape(block.ada_norm_bias, {2 * width}) ||
                !has_shape(block.query_weight, {width, width}) ||
                !has_shape(block.query_bias, {width}) ||
                !has_shape(block.key_weight,
                           {config_.cross_attention_dim, width}) ||
                !has_shape(block.key_bias, {width}) ||
                !has_shape(block.value_weight,
                           {config_.cross_attention_dim, width}) ||
                !has_shape(block.value_bias, {width}) ||
                !has_shape(block.attention_output_weight, {width, width}) ||
                !has_shape(block.attention_output_bias, {width}) ||
                !has_shape(block.feed_forward_input_weight,
                           {width, config_.feed_forward_dim}) ||
                !has_shape(block.feed_forward_input_bias,
                           {config_.feed_forward_dim}) ||
                !has_shape(block.feed_forward_output_weight,
                           {config_.feed_forward_dim, width}) ||
                !has_shape(block.feed_forward_output_bias, {width})) {
                throw std::runtime_error(
                    "StarVLA PI transformer tensor has an incompatible ggml shape");
            }
        }
        return true;
    }

  private:
    PIPolicyConfig & config_;
    PIWeights & weights_;
};

std::vector<float> timestep_projection_table(const PIPolicyConfig & config) {
    std::vector<float> result(
        static_cast<size_t>(config.num_inference_timesteps) *
        config.timestep_projection_dim);
    const int half = config.timestep_projection_dim / 2;
    for (int step = 0; step < config.num_inference_timesteps; ++step) {
        const float timestep =
            static_cast<float>(config.timestep_ids[static_cast<size_t>(step)]);
        for (int i = 0; i < half; ++i) {
            const float exponent =
                -std::log(10000.0f) * i / static_cast<float>(half - 1);
            const float angle = timestep * std::exp(exponent);
            const size_t offset =
                static_cast<size_t>(step) * config.timestep_projection_dim;
            result[offset + static_cast<size_t>(i)] = std::cos(angle);
            result[offset + static_cast<size_t>(i + half)] = std::sin(angle);
        }
    }
    return result;
}

std::vector<float> action_time_table(const PIPolicyConfig & config) {
    std::vector<float> result(
        static_cast<size_t>(config.num_inference_timesteps) * config.dit_width);
    const int half = config.dit_width / 2;
    for (int step = 0; step < config.num_inference_timesteps; ++step) {
        const float timestep =
            static_cast<float>(config.timestep_ids[static_cast<size_t>(step)]);
        for (int i = 0; i < half; ++i) {
            const float exponent =
                -std::log(10000.0f) * i / static_cast<float>(half);
            const float angle = timestep * std::exp(exponent);
            const size_t offset =
                static_cast<size_t>(step) * config.dit_width;
            result[offset + static_cast<size_t>(i)] = std::sin(angle);
            result[offset + static_cast<size_t>(i + half)] = std::cos(angle);
        }
    }
    return result;
}

} // namespace

struct PIPolicy::Impl {
    PIPolicyConfig config;
    PIWeights weights;
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
    ggml_tensor * state_input = nullptr;
    ggml_tensor * noise_input = nullptr;
    ggml_tensor * timestep_projection_input = nullptr;
    ggml_tensor * action_time_input = nullptr;
    ggml_tensor * scalar_one_input = nullptr;
    ggml_tensor * output = nullptr;
    size_t conditioning_token_count = 0;
    bool graph_uses_state = false;
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
        state_input = nullptr;
        noise_input = nullptr;
        timestep_projection_input = nullptr;
        action_time_input = nullptr;
        scalar_one_input = nullptr;
        output = nullptr;
        conditioning_token_count = 0;
        graph_uses_state = false;
    }

    void build_graph(size_t token_count, bool include_state) {
        clear_graph();
        if (token_count == 0 ||
            token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "invalid StarVLA PI conditioning token count");
        }

        ggml_init_params params{};
        params.mem_size =
            kGraphSize * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(kGraphSize, false);
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph_context = ggml_init(params);
        if (graph_context == nullptr) {
            throw std::runtime_error(
                "failed to initialize StarVLA PI graph context");
        }

        const int width = config.dit_width;
        const int heads = config.attention_head_count;
        const int head_dim = config.attention_head_dim;
        hidden_input = ggml_new_tensor_3d(
            graph_context, GGML_TYPE_F32, config.qwen_hidden_dim,
            static_cast<int64_t>(token_count), config.block_count);
        if (include_state) {
            state_input = ggml_new_tensor_1d(graph_context, GGML_TYPE_F32,
                                             config.state_dim);
        }
        noise_input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32,
                                         config.action_dim, config.horizon);
        timestep_projection_input = ggml_new_tensor_2d(
            graph_context, GGML_TYPE_F32, config.timestep_projection_dim,
            config.num_inference_timesteps);
        action_time_input = ggml_new_tensor_2d(
            graph_context, GGML_TYPE_F32, width,
            config.num_inference_timesteps);
        scalar_one_input =
            ggml_new_tensor_1d(graph_context, GGML_TYPE_F32, 1);
        if (hidden_input == nullptr ||
            (include_state && state_input == nullptr) ||
            noise_input == nullptr || timestep_projection_input == nullptr ||
            action_time_input == nullptr || scalar_one_input == nullptr) {
            throw std::runtime_error(
                "failed to create StarVLA PI graph inputs");
        }
        ggml_set_name(hidden_input, "starvla_pi_qwen_hidden_states");
        if (state_input != nullptr) {
            ggml_set_name(state_input, "starvla_pi_state");
        }
        ggml_set_name(noise_input, "starvla_pi_initial_noise");
        ggml_set_name(timestep_projection_input,
                      "starvla_pi_timestep_projection_table");
        ggml_set_name(action_time_input, "starvla_pi_action_time_table");
        ggml_set_name(scalar_one_input, "starvla_pi_scalar_one");
        ggml_set_input(hidden_input);
        if (state_input != nullptr) {
            ggml_set_input(state_input);
        }
        ggml_set_input(noise_input);
        ggml_set_input(timestep_projection_input);
        ggml_set_input(action_time_input);
        ggml_set_input(scalar_one_input);

        auto f32 = [&](ggml_tensor * tensor) {
            return tensor->type == GGML_TYPE_F32
                       ? tensor
                       : ggml_cast(graph_context, tensor, GGML_TYPE_F32);
        };
        auto linear = [&](ggml_tensor * value, ggml_tensor * weight,
                          ggml_tensor * bias) {
            ggml_tensor * projected =
                ggml_mul_mat(graph_context, weight, value);
            ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
            return ggml_add(graph_context, projected, f32(bias));
        };
        auto ada_norm = [&](ggml_tensor * value, ggml_tensor * temb,
                            const PIBlockWeights & block) {
            ggml_tensor * modulation =
                linear(ggml_silu(graph_context, temb),
                       block.ada_norm_weight, block.ada_norm_bias);
            ggml_tensor * scale =
                ggml_view_1d(graph_context, modulation, width, 0);
            ggml_tensor * shift = ggml_view_1d(
                graph_context, modulation, width,
                static_cast<size_t>(width) * sizeof(float));
            ggml_tensor * normalized =
                ggml_norm(graph_context, value, config.ada_norm_epsilon);
            return ggml_add(
                graph_context,
                ggml_mul(graph_context, normalized,
                         ggml_add(graph_context, scale, scalar_one_input)),
                shift);
        };
        auto attention = [&](ggml_tensor * query_source,
                             ggml_tensor * key_value_source,
                             const PIBlockWeights & block) {
            const int64_t query_count = query_source->ne[1];
            const int64_t key_value_count = key_value_source->ne[1];
            ggml_tensor * query =
                linear(query_source, block.query_weight, block.query_bias);
            ggml_tensor * key =
                linear(key_value_source, block.key_weight, block.key_bias);
            ggml_tensor * value =
                linear(key_value_source, block.value_weight, block.value_bias);
            query = ggml_reshape_3d(graph_context, query, head_dim, heads,
                                    query_count);
            key = ggml_reshape_3d(graph_context, key, head_dim, heads,
                                  key_value_count);
            value = ggml_reshape_3d(graph_context, value, head_dim, heads,
                                    key_value_count);
            query = ggml_permute(graph_context, query, 0, 2, 1, 3);
            key = ggml_permute(graph_context, key, 0, 2, 1, 3);
            value = ggml_cont(
                graph_context,
                ggml_permute(graph_context, value, 1, 2, 0, 3));
            ggml_tensor * scores = ggml_mul_mat(graph_context, key, query);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores = ggml_soft_max_ext(
                graph_context, scores, nullptr,
                1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
            ggml_tensor * attended =
                ggml_mul_mat(graph_context, value, scores);
            ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
            attended =
                ggml_permute(graph_context, attended, 0, 2, 1, 3);
            attended =
                ggml_cont_2d(graph_context, attended, width, query_count);
            return linear(attended, block.attention_output_weight,
                          block.attention_output_bias);
        };

        ggml_tensor * state_features = nullptr;
        if (include_state) {
            state_features =
                ggml_relu(graph_context,
                          linear(state_input, weights.state_input_weight,
                                 weights.state_input_bias));
            state_features =
                linear(state_features, weights.state_output_weight,
                       weights.state_output_bias);
            state_features =
                ggml_reshape_2d(graph_context, state_features, width, 1);
        }
        ggml_tensor * future = f32(weights.future_tokens);
        ggml_tensor * position_view = ggml_view_2d(
            graph_context, weights.action_position, width, config.horizon,
            weights.action_position->nb[1], 0);
        ggml_tensor * position = f32(position_view);
        ggml_tensor * actions = noise_input;

        for (int step = 0; step < config.num_inference_timesteps; ++step) {
            ggml_tensor * timestep_projection = ggml_view_1d(
                graph_context, timestep_projection_input,
                config.timestep_projection_dim,
                static_cast<size_t>(step) *
                    config.timestep_projection_dim * sizeof(float));
            ggml_tensor * temb =
                linear(timestep_projection, weights.timestep_input_weight,
                       weights.timestep_input_bias);
            temb = ggml_silu(graph_context, temb);
            temb = linear(temb, weights.timestep_output_weight,
                          weights.timestep_output_bias);

            ggml_tensor * action_features =
                linear(actions, weights.action_input_weight,
                       weights.action_input_bias);
            ggml_tensor * action_time = ggml_view_1d(
                graph_context, action_time_input, width,
                static_cast<size_t>(step) * width * sizeof(float));
            action_time =
                ggml_repeat(graph_context, action_time, action_features);
            action_features =
                ggml_concat(graph_context, action_features, action_time, 0);
            action_features =
                linear(action_features, weights.action_time_mix_weight,
                       weights.action_time_mix_bias);
            action_features = ggml_silu(graph_context, action_features);
            action_features =
                linear(action_features, weights.action_output_weight,
                       weights.action_output_bias);
            action_features =
                ggml_add(graph_context, action_features, position);

            ggml_tensor * hidden = future;
            if (state_features != nullptr) {
                hidden =
                    ggml_concat(graph_context, state_features, hidden, 1);
            }
            hidden = ggml_concat(graph_context, hidden, action_features, 1);
            for (int block_index = 0; block_index < config.block_count;
                 ++block_index) {
                const PIBlockWeights & block =
                    weights.blocks[static_cast<size_t>(block_index)];
                ggml_tensor * layer_hidden = ggml_view_2d(
                    graph_context, hidden_input, config.qwen_hidden_dim,
                    static_cast<int64_t>(token_count), hidden_input->nb[1],
                    static_cast<size_t>(block_index) * hidden_input->nb[2]);
                ggml_tensor * normalized = ada_norm(hidden, temb, block);
                hidden = ggml_add(
                    graph_context, hidden,
                    attention(normalized, layer_hidden, block));
                ggml_tensor * ff =
                    ggml_norm(graph_context, hidden,
                              config.ada_norm_epsilon);
                ff = linear(ff, block.feed_forward_input_weight,
                            block.feed_forward_input_bias);
                ff = ggml_gelu(graph_context, ff);
                ff = linear(ff, block.feed_forward_output_weight,
                            block.feed_forward_output_bias);
                hidden = ggml_add(graph_context, hidden, ff);
            }

            hidden =
                ggml_relu(graph_context,
                          linear(hidden, weights.velocity_input_weight,
                                 weights.velocity_input_bias));
            hidden = linear(hidden, weights.velocity_output_weight,
                            weights.velocity_output_bias);
            ggml_tensor * velocity = ggml_view_2d(
                graph_context, hidden, config.action_dim, config.horizon,
                hidden->nb[1],
                static_cast<size_t>(
                    (include_state ? config.state_token_count : 0) +
                    config.future_token_count) *
                    hidden->nb[1]);
            actions = ggml_add(
                graph_context, actions,
                ggml_scale(graph_context, velocity, config.euler_dt));
        }

        output = actions;
        ggml_set_name(output, "starvla_pi_normalized_actions");
        ggml_set_output(output);
        graph = ggml_new_graph_custom(graph_context, kGraphSize, false);
        if (graph == nullptr) {
            throw std::runtime_error(
                "failed to create StarVLA PI graph");
        }
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error(
                "failed to allocate StarVLA PI graph");
        }
        conditioning_token_count = token_count;
        graph_uses_state = include_state;
    }
};

PIPolicy::PIPolicy(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PIPolicy::~PIPolicy() = default;

std::unique_ptr<PIPolicy> PIPolicy::load(const std::string & path, int n_threads,
                                         int verbosity, std::string & error) {
    error.clear();
    if (path.empty()) {
        error = "StarVLA PI policy path is required";
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
            error = "failed to initialize StarVLA PI backend: " + backend.error();
            return nullptr;
        }
        impl->mode = backend.mode();

        PIGGUFLoader loader(impl->config, impl->weights);
        if (!loader.load(path.c_str(), impl->buft_policy.model_buft,
                         impl->loaded, verbosity)) {
            error = loader.error();
            return nullptr;
        }
        if (impl->loaded.ctx_data == nullptr ||
            impl->loaded.model_buffer == nullptr) {
            error = "StarVLA PI policy GGUF has no tensors";
            return nullptr;
        }
        ggml_backend_buffer_set_usage(
            impl->loaded.model_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        impl->timestep_table = timestep_projection_table(impl->config);
        impl->action_table = action_time_table(impl->config);
        if (verbosity >= 1) {
            std::fprintf(
                stderr,
                "%s: backend=%s qwen=%d width=%d blocks=%d horizon=%d "
                "action_dim=%d profiles=%zu\n",
                __func__, backend_mode_name(impl->mode), impl->config.qwen_hidden_dim,
                impl->config.dit_width, impl->config.block_count,
                impl->config.horizon, impl->config.action_dim,
                impl->config.normalization.profiles.size());
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
    return std::unique_ptr<PIPolicy>(new PIPolicy(std::move(impl)));
}

bool PIPolicy::evaluate(const float * qwen_hidden_states,
                        size_t hidden_element_count, const float * state,
                        size_t state_element_count, const float * initial_noise,
                        size_t noise_element_count,
                        std::vector<float> & normalized_actions,
                        std::string & error) {
    normalized_actions.clear();
    error.clear();
    if (impl_ == nullptr || impl_->scheduler == nullptr) {
        error = "StarVLA PI policy is not initialized";
        return false;
    }
    const size_t layer_width =
        static_cast<size_t>(impl_->config.block_count) *
        impl_->config.qwen_hidden_dim;
    if (qwen_hidden_states == nullptr || layer_width == 0 ||
        hidden_element_count == 0 ||
        hidden_element_count % layer_width != 0) {
        error =
            "StarVLA PI layer-wise Qwen conditioning tensor has an incompatible shape";
        return false;
    }
    const size_t token_count = hidden_element_count / layer_width;
    if (token_count == 0 ||
        token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error =
            "StarVLA PI layer-wise Qwen conditioning tensor has an incompatible shape";
        return false;
    }
    const bool include_state = state_element_count != 0;
    if (include_state &&
        (state == nullptr ||
         state_element_count != static_cast<size_t>(impl_->config.state_dim))) {
        error = "StarVLA PI state tensor has an incompatible shape";
        return false;
    }
    const size_t expected_noise =
        static_cast<size_t>(impl_->config.horizon) * impl_->config.action_dim;
    if (initial_noise == nullptr || noise_element_count != expected_noise) {
        error = "StarVLA PI initial-noise tensor has an incompatible shape";
        return false;
    }
    if (std::any_of(qwen_hidden_states,
                    qwen_hidden_states + hidden_element_count,
                    [](float value) { return !std::isfinite(value); }) ||
        (include_state &&
         std::any_of(state, state + state_element_count,
                     [](float value) { return !std::isfinite(value); })) ||
        std::any_of(initial_noise, initial_noise + noise_element_count,
                    [](float value) { return !std::isfinite(value); })) {
        error =
            "StarVLA PI conditioning, state, and initial noise must be finite";
        return false;
    }

    try {
        if (impl_->graph == nullptr ||
            impl_->conditioning_token_count != token_count ||
            impl_->graph_uses_state != include_state) {
            impl_->build_graph(token_count, include_state);
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }

    ggml_backend_tensor_set(impl_->hidden_input, qwen_hidden_states, 0,
                            hidden_element_count * sizeof(float));
    if (include_state) {
        ggml_backend_tensor_set(impl_->state_input, state, 0,
                                state_element_count * sizeof(float));
    }
    ggml_backend_tensor_set(impl_->noise_input, initial_noise, 0,
                            noise_element_count * sizeof(float));
    ggml_backend_tensor_set(impl_->timestep_projection_input,
                            impl_->timestep_table.data(), 0,
                            impl_->timestep_table.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->action_time_input,
                            impl_->action_table.data(), 0,
                            impl_->action_table.size() * sizeof(float));
    const float one = 1.0f;
    ggml_backend_tensor_set(impl_->scalar_one_input, &one, 0, sizeof(one));
    set_backend_threads(impl_->backends, impl_->n_threads);
    if (ggml_backend_sched_graph_compute(impl_->scheduler, impl_->graph) !=
        GGML_STATUS_SUCCESS) {
        error = "StarVLA PI graph compute failed";
        return false;
    }

    normalized_actions.resize(expected_noise);
    ggml_backend_tensor_get(impl_->output, normalized_actions.data(), 0,
                            expected_noise * sizeof(float));
    if (std::any_of(normalized_actions.begin(), normalized_actions.end(),
                    [](float value) { return !std::isfinite(value); })) {
        normalized_actions.clear();
        error = "StarVLA PI graph produced non-finite actions";
        return false;
    }
    return true;
}

bool PIPolicy::unnormalize(const std::vector<float> & normalized_actions,
                           const std::string & profile_key_value,
                           std::vector<float> & actions,
                           std::string & error) const {
    if (impl_ == nullptr) {
        actions.clear();
        error = "StarVLA PI policy is not initialized";
        return false;
    }
    return denormalize_actions(impl_->config.normalization, profile_key_value,
                               normalized_actions, impl_->config.horizon,
                               impl_->config.action_dim, actions, error);
}

const PIPolicyConfig & PIPolicy::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("StarVLA PI policy is not initialized");
    }
    return impl_->config;
}

const char * PIPolicy::backend_name() const {
    return impl_ != nullptr ? backend_mode_name(impl_->mode) : "unknown";
}

} // namespace robotcpp::starvla
