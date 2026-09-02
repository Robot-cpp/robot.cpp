#include "models/starvla/groot_policy.h"

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
constexpr int kKQMaskPad    = 32;

struct GR00TBlockWeights {
    ggml_tensor * ada_norm_weight            = nullptr;
    ggml_tensor * ada_norm_bias              = nullptr;
    ggml_tensor * query_weight               = nullptr;
    ggml_tensor * query_bias                 = nullptr;
    ggml_tensor * key_weight                 = nullptr;
    ggml_tensor * key_bias                   = nullptr;
    ggml_tensor * value_weight               = nullptr;
    ggml_tensor * value_bias                 = nullptr;
    ggml_tensor * attention_output_weight    = nullptr;
    ggml_tensor * attention_output_bias      = nullptr;
    ggml_tensor * feed_forward_input_weight  = nullptr;
    ggml_tensor * feed_forward_input_bias    = nullptr;
    ggml_tensor * feed_forward_output_weight = nullptr;
    ggml_tensor * feed_forward_output_bias   = nullptr;
};

struct GR00TWeights {
    ggml_tensor * timestep_input_weight  = nullptr;
    ggml_tensor * timestep_input_bias    = nullptr;
    ggml_tensor * timestep_output_weight = nullptr;
    ggml_tensor * timestep_output_bias   = nullptr;
    std::vector<GR00TBlockWeights> blocks;
    ggml_tensor * output_modulation_weight = nullptr;
    ggml_tensor * output_modulation_bias   = nullptr;
    ggml_tensor * output_projection_weight = nullptr;
    ggml_tensor * output_projection_bias   = nullptr;
    ggml_tensor * action_input_weight      = nullptr;
    ggml_tensor * action_input_bias        = nullptr;
    ggml_tensor * action_time_mix_weight   = nullptr;
    ggml_tensor * action_time_mix_bias     = nullptr;
    ggml_tensor * action_output_weight     = nullptr;
    ggml_tensor * action_output_bias       = nullptr;
    ggml_tensor * velocity_input_weight    = nullptr;
    ggml_tensor * velocity_input_bias      = nullptr;
    ggml_tensor * velocity_output_weight   = nullptr;
    ggml_tensor * velocity_output_bias     = nullptr;
    ggml_tensor * future_tokens            = nullptr;
    ggml_tensor * action_position          = nullptr;
};

using detail::has_shape;
using detail::require_f32;
using detail::require_i32;
using detail::require_i32_array;
using detail::require_string;
using detail::require_string_array;

class GR00TGGUFLoader final : public gguf_loader {
  public:
    GR00TGGUFLoader(GR00TPolicyConfig & config, GR00TWeights & weights) : config_(config), weights_(weights) {}

  protected:
    bool parse_metadata(gguf_context * gguf) override {
        if (require_string(gguf, "general.architecture") != "starvla-policy") {
            throw std::runtime_error("StarVLA GR00T policy has incompatible general.architecture");
        }
        if (require_i32(gguf, "starvla.schema_version") != 1 || require_string(gguf, "starvla.framework") != "groot") {
            throw std::runtime_error("StarVLA policy GGUF is not a supported Qwen GR00T schema");
        }
        config_.backbone_arch = require_string(gguf, "starvla.backbone.arch");
        if (config_.backbone_arch != "qwen3_vl" && config_.backbone_arch != "qwen2_5_vl") {
            throw std::runtime_error("StarVLA GR00T policy has an unsupported Qwen backbone");
        }

        config_.bundle_uuid = require_string(gguf, "starvla.bundle.uuid");
        if (config_.bundle_uuid.empty()) {
            throw std::runtime_error("StarVLA GR00T bundle UUID is missing");
        }
        config_.text_filename   = require_string(gguf, "starvla.component.text.filename");
        config_.mmproj_filename = require_string(gguf, "starvla.component.mmproj.filename");
        if (config_.text_filename.empty() || config_.mmproj_filename.empty()) {
            throw std::runtime_error("StarVLA GR00T component filenames must be non-empty");
        }
        config_.qwen_hidden_dim          = require_i32(gguf, "starvla.qwen.hidden_size");
        config_.qwen_input_embedding_dim = require_i32(gguf, "starvla.qwen.input_embedding_size");
        config_.qwen_vocab_size          = require_i32(gguf, "starvla.qwen.vocab_size");
        config_.cot_template             = require_string(gguf, "starvla.prompt.cot_template");
        const bool qwen25                = config_.backbone_arch == "qwen2_5_vl";
        if (config_.cot_template.empty()) {
            throw std::runtime_error("StarVLA GR00T prompt template is missing");
        }

        config_.image_count                = require_i32(gguf, "starvla.image.count");
        config_.image_names                = require_string_array(gguf, "starvla.image.names");
        config_.image_processor_min_pixels = require_i32(gguf, "starvla.image.processor_min_pixels");
        config_.image_processor_max_pixels = require_i32(gguf, "starvla.image.processor_max_pixels");
        config_.image_patch_size           = require_i32(gguf, "starvla.image.patch_size");
        config_.image_spatial_merge_size   = require_i32(gguf, "starvla.image.spatial_merge_size");
        config_.image_min_token_count      = require_i32(gguf, "starvla.image.min_token_count");
        config_.image_max_token_count      = require_i32(gguf, "starvla.image.max_token_count");
        config_.dit_width                  = require_i32(gguf, "starvla.groot.dit_width");
        config_.block_count                = require_i32(gguf, "starvla.groot.block_count");
        config_.attention_head_count       = require_i32(gguf, "starvla.groot.attention_head_count");
        config_.attention_head_dim         = require_i32(gguf, "starvla.groot.attention_head_dim");
        config_.cross_attention_dim        = require_i32(gguf, "starvla.groot.cross_attention_dim");
        config_.feed_forward_dim           = require_i32(gguf, "starvla.groot.feed_forward_dim");
        config_.output_dim                 = require_i32(gguf, "starvla.groot.output_dimension");
        config_.mlp_hidden_dim             = require_i32(gguf, "starvla.groot.mlp_hidden_dimension");
        config_.future_token_count         = require_i32(gguf, "starvla.groot.future_token_count");
        config_.action_position_count      = require_i32(gguf, "starvla.groot.action_position_count");
        config_.no_state_sequence_length   = require_i32(gguf, "starvla.groot.no_state_sequence_length");
        config_.timestep_projection_dim    = require_i32(gguf, "starvla.groot.timestep_projection_dim");
        config_.ada_norm_epsilon           = require_f32(gguf, "starvla.groot.ada_norm_epsilon");
        config_.output_norm_epsilon        = require_f32(gguf, "starvla.groot.output_norm_epsilon");
        config_.euler_dt                   = require_f32(gguf, "starvla.groot.euler_dt");
        config_.timestep_ids               = require_i32_array(gguf, "starvla.groot.timestep_ids");
        config_.action_dim                 = require_i32(gguf, "starvla.action.dimension");
        config_.horizon                    = require_i32(gguf, "starvla.action.horizon");

        const int64_t expected_input_embedding_dim =
            qwen25 ? static_cast<int64_t>(config_.qwen_hidden_dim) : 4LL * config_.qwen_hidden_dim;
        const bool dimensions_valid =
            config_.qwen_hidden_dim > 0 && config_.qwen_vocab_size > 0 && config_.dit_width > 0 &&
            config_.qwen_input_embedding_dim == expected_input_embedding_dim && config_.dit_width % 2 == 0 &&
            config_.block_count > 0 && config_.block_count % 2 == 0 && config_.attention_head_count > 0 &&
            config_.attention_head_dim > 0 &&
            config_.attention_head_count * config_.attention_head_dim == config_.dit_width &&
            config_.cross_attention_dim == config_.qwen_hidden_dim && config_.feed_forward_dim > 0 &&
            config_.output_dim > 0 && config_.mlp_hidden_dim > 0 && config_.action_dim > 0 && config_.horizon > 0 &&
            config_.future_token_count > 0 && config_.action_position_count >= config_.horizon &&
            config_.no_state_sequence_length == config_.future_token_count + config_.horizon &&
            config_.timestep_projection_dim >= 4 && config_.timestep_projection_dim % 2 == 0 &&
            std::isfinite(config_.ada_norm_epsilon) && config_.ada_norm_epsilon > 0.0f &&
            std::isfinite(config_.output_norm_epsilon) && config_.output_norm_epsilon > 0.0f &&
            std::isfinite(config_.euler_dt) && config_.euler_dt > 0.0f && config_.timestep_ids.size() == 4 &&
            config_.image_count > 0 && config_.image_names.size() == static_cast<size_t>(config_.image_count) &&
            config_.image_processor_min_pixels > 0 &&
            config_.image_processor_max_pixels >= config_.image_processor_min_pixels && config_.image_patch_size > 0 &&
            config_.image_spatial_merge_size > 0 && config_.image_min_token_count > 0 &&
            config_.image_max_token_count >= config_.image_min_token_count;
        if (!dimensions_valid) {
            throw std::runtime_error("StarVLA GR00T policy metadata has incompatible dimensions");
        }
        config_.normalization = detail::require_normalization(gguf, config_.action_dim);
        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        auto bind = [&](ggml_tensor *& destination, const std::string & name) {
            destination = require_tensor(ctx_data, name);
        };
        bind(weights_.timestep_input_weight, "starvla.policy.groot.timestep.input.weight");
        bind(weights_.timestep_input_bias, "starvla.policy.groot.timestep.input.bias");
        bind(weights_.timestep_output_weight, "starvla.policy.groot.timestep.output.weight");
        bind(weights_.timestep_output_bias, "starvla.policy.groot.timestep.output.bias");
        weights_.blocks.clear();
        weights_.blocks.reserve(static_cast<size_t>(config_.block_count));
        for (int block = 0; block < config_.block_count; ++block) {
            const std::string prefix = "starvla.policy.groot.block." + std::to_string(block) + ".";
            GR00TBlockWeights current;
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
        bind(weights_.output_modulation_weight, "starvla.policy.groot.output.modulation.weight");
        bind(weights_.output_modulation_bias, "starvla.policy.groot.output.modulation.bias");
        bind(weights_.output_projection_weight, "starvla.policy.groot.output.projection.weight");
        bind(weights_.output_projection_bias, "starvla.policy.groot.output.projection.bias");
        bind(weights_.action_input_weight, "starvla.policy.groot.action.input.weight");
        bind(weights_.action_input_bias, "starvla.policy.groot.action.input.bias");
        bind(weights_.action_time_mix_weight, "starvla.policy.groot.action.time_mix.weight");
        bind(weights_.action_time_mix_bias, "starvla.policy.groot.action.time_mix.bias");
        bind(weights_.action_output_weight, "starvla.policy.groot.action.output.weight");
        bind(weights_.action_output_bias, "starvla.policy.groot.action.output.bias");
        bind(weights_.velocity_input_weight, "starvla.policy.groot.velocity.input.weight");
        bind(weights_.velocity_input_bias, "starvla.policy.groot.velocity.input.bias");
        bind(weights_.velocity_output_weight, "starvla.policy.groot.velocity.output.weight");
        bind(weights_.velocity_output_bias, "starvla.policy.groot.velocity.output.bias");
        bind(weights_.future_tokens, "starvla.policy.groot.future_tokens.weight");
        bind(weights_.action_position, "starvla.policy.groot.action_position.weight");

        const int width = config_.dit_width;
        if (!has_shape(weights_.timestep_input_weight, {config_.timestep_projection_dim, width}) ||
            !has_shape(weights_.timestep_input_bias, {width}) ||
            !has_shape(weights_.timestep_output_weight, {width, width}) ||
            !has_shape(weights_.timestep_output_bias, {width}) ||
            !has_shape(weights_.output_modulation_weight, {width, 2 * width}) ||
            !has_shape(weights_.output_modulation_bias, {2 * width}) ||
            !has_shape(weights_.output_projection_weight, {width, config_.output_dim}) ||
            !has_shape(weights_.output_projection_bias, {config_.output_dim}) ||
            !has_shape(weights_.action_input_weight, {config_.action_dim, width}) ||
            !has_shape(weights_.action_input_bias, {width}) ||
            !has_shape(weights_.action_time_mix_weight, {2 * width, width}) ||
            !has_shape(weights_.action_time_mix_bias, {width}) ||
            !has_shape(weights_.action_output_weight, {width, width}) ||
            !has_shape(weights_.action_output_bias, {width}) ||
            !has_shape(weights_.velocity_input_weight, {config_.output_dim, config_.mlp_hidden_dim}) ||
            !has_shape(weights_.velocity_input_bias, {config_.mlp_hidden_dim}) ||
            !has_shape(weights_.velocity_output_weight, {config_.mlp_hidden_dim, config_.action_dim}) ||
            !has_shape(weights_.velocity_output_bias, {config_.action_dim}) ||
            !has_shape(weights_.future_tokens, {width, config_.future_token_count}) ||
            !has_shape(weights_.action_position, {width, config_.action_position_count})) {
            throw std::runtime_error("StarVLA GR00T non-block tensor has an incompatible ggml shape");
        }
        for (int block = 0; block < config_.block_count; ++block) {
            const GR00TBlockWeights & current = weights_.blocks[static_cast<size_t>(block)];
            const int kv_input_dim            = block % 2 == 0 ? config_.cross_attention_dim : width;
            if (!has_shape(current.ada_norm_weight, {width, 2 * width}) ||
                !has_shape(current.ada_norm_bias, {2 * width}) || !has_shape(current.query_weight, {width, width}) ||
                !has_shape(current.query_bias, {width}) || !has_shape(current.key_weight, {kv_input_dim, width}) ||
                !has_shape(current.key_bias, {width}) || !has_shape(current.value_weight, {kv_input_dim, width}) ||
                !has_shape(current.value_bias, {width}) ||
                !has_shape(current.attention_output_weight, {width, width}) ||
                !has_shape(current.attention_output_bias, {width}) ||
                !has_shape(current.feed_forward_input_weight, {width, config_.feed_forward_dim}) ||
                !has_shape(current.feed_forward_input_bias, {config_.feed_forward_dim}) ||
                !has_shape(current.feed_forward_output_weight, {config_.feed_forward_dim, width}) ||
                !has_shape(current.feed_forward_output_bias, {width})) {
                throw std::runtime_error("StarVLA GR00T transformer block tensor has an incompatible ggml shape");
            }
        }
        return true;
    }

  private:
    GR00TPolicyConfig & config_;
    GR00TWeights & weights_;
};

std::vector<float> timestep_projection_table(const GR00TPolicyConfig & config) {
    const int dim           = config.timestep_projection_dim;
    const int half          = dim / 2;
    const float denominator = static_cast<float>(half - 1);
    std::vector<float> table(static_cast<size_t>(dim) * 4, 0.0f);
    for (int step = 0; step < 4; ++step) {
        const float timestep = static_cast<float>(config.timestep_ids[static_cast<size_t>(step)]);
        float * row          = table.data() + static_cast<size_t>(step) * dim;
        for (int index = 0; index < half; ++index) {
            const float frequency = std::exp(-std::log(10000.0f) * static_cast<float>(index) / denominator);
            const float angle     = timestep * frequency;
            row[index]            = std::cos(angle);
            row[index + half]     = std::sin(angle);
        }
    }
    return table;
}

std::vector<float> action_time_table(const GR00TPolicyConfig & config) {
    const int dim           = config.dit_width;
    const int half          = dim / 2;
    const float denominator = static_cast<float>(half);
    std::vector<float> table(static_cast<size_t>(dim) * 4, 0.0f);
    for (int step = 0; step < 4; ++step) {
        const float timestep = static_cast<float>(config.timestep_ids[static_cast<size_t>(step)]);
        float * row          = table.data() + static_cast<size_t>(step) * dim;
        for (int index = 0; index < half; ++index) {
            const float frequency = std::exp(-std::log(10000.0f) * static_cast<float>(index) / denominator);
            const float angle     = timestep * frequency;
            row[index]            = std::sin(angle);
            row[index + half]     = std::cos(angle);
        }
    }
    return table;
}

} // namespace

struct GR00TPolicy::Impl {
    GR00TPolicyConfig config;
    GR00TWeights weights;
    gguf_load_result loaded;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t scheduler = nullptr;
    backend_buft_policy buft_policy;
    backend_mode mode = backend_mode::cpu;
    int n_threads     = 0;
    int verbosity     = 0;

    ggml_context * graph_context            = nullptr;
    ggml_cgraph * graph                     = nullptr;
    ggml_tensor * hidden_input              = nullptr;
    ggml_tensor * cross_mask_input          = nullptr;
    ggml_tensor * noise_input               = nullptr;
    ggml_tensor * timestep_projection_input = nullptr;
    ggml_tensor * action_time_input         = nullptr;
    ggml_tensor * scalar_one_input          = nullptr;
    ggml_tensor * output                    = nullptr;
    size_t conditioning_token_count         = 0;
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
        graph                     = nullptr;
        hidden_input              = nullptr;
        cross_mask_input          = nullptr;
        noise_input               = nullptr;
        timestep_projection_input = nullptr;
        action_time_input         = nullptr;
        scalar_one_input          = nullptr;
        output                    = nullptr;
        conditioning_token_count  = 0;
    }

    void build_graph(size_t token_count) {
        clear_graph();
        if (token_count == 0 || token_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("invalid StarVLA GR00T conditioning token count");
        }

        ggml_init_params params{};
        params.mem_size   = kGraphSize * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphSize, false);
        params.mem_buffer = nullptr;
        params.no_alloc   = true;
        graph_context     = ggml_init(params);
        if (graph_context == nullptr) {
            throw std::runtime_error("failed to initialize StarVLA GR00T graph context");
        }

        const int width           = config.dit_width;
        const int heads           = config.attention_head_count;
        const int head_dim        = config.attention_head_dim;
        const int sequence_length = config.no_state_sequence_length;
        const int mask_queries    = GGML_PAD(sequence_length, kKQMaskPad);

        hidden_input =
            ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, config.qwen_hidden_dim, static_cast<int64_t>(token_count));
        cross_mask_input =
            ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, static_cast<int64_t>(token_count), mask_queries);
        noise_input               = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, config.action_dim, config.horizon);
        timestep_projection_input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, config.timestep_projection_dim, 4);
        action_time_input         = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, width, 4);
        scalar_one_input          = ggml_new_tensor_1d(graph_context, GGML_TYPE_F32, 1);
        if (hidden_input == nullptr || cross_mask_input == nullptr || noise_input == nullptr ||
            timestep_projection_input == nullptr || action_time_input == nullptr || scalar_one_input == nullptr) {
            throw std::runtime_error("failed to create StarVLA GR00T graph inputs");
        }
        ggml_set_name(hidden_input, "starvla_groot_qwen_hidden_states");
        ggml_set_name(cross_mask_input, "starvla_groot_qwen_attention_mask");
        ggml_set_name(noise_input, "starvla_groot_initial_noise");
        ggml_set_name(timestep_projection_input, "starvla_groot_timestep_projection_table");
        ggml_set_name(action_time_input, "starvla_groot_action_time_table");
        ggml_set_name(scalar_one_input, "starvla_groot_scalar_one");
        ggml_set_input(hidden_input);
        ggml_set_input(cross_mask_input);
        ggml_set_input(noise_input);
        ggml_set_input(timestep_projection_input);
        ggml_set_input(action_time_input);
        ggml_set_input(scalar_one_input);

        auto f32 = [&](ggml_tensor * tensor) {
            return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(graph_context, tensor, GGML_TYPE_F32);
        };
        auto linear = [&](ggml_tensor * value, ggml_tensor * weight, ggml_tensor * bias) {
            ggml_tensor * projected = ggml_mul_mat(graph_context, weight, value);
            ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
            return ggml_add(graph_context, projected, f32(bias));
        };
        auto ada_norm = [&](ggml_tensor * value, ggml_tensor * temb, const GR00TBlockWeights & block) {
            ggml_tensor * modulation =
                linear(ggml_silu(graph_context, temb), block.ada_norm_weight, block.ada_norm_bias);
            ggml_tensor * scale = ggml_view_1d(graph_context, modulation, width, 0);
            ggml_tensor * shift =
                ggml_view_1d(graph_context, modulation, width, static_cast<size_t>(width) * sizeof(float));
            ggml_tensor * normalized     = ggml_norm(graph_context, value, config.ada_norm_epsilon);
            ggml_tensor * one_plus_scale = ggml_add(graph_context, scale, scalar_one_input);
            return ggml_add(graph_context, ggml_mul(graph_context, normalized, one_plus_scale), shift);
        };
        auto attention = [&](ggml_tensor * query_source, ggml_tensor * key_value_source, ggml_tensor * mask,
                             const GR00TBlockWeights & block) {
            const int64_t query_count     = query_source->ne[1];
            const int64_t key_value_count = key_value_source->ne[1];
            ggml_tensor * query           = linear(query_source, block.query_weight, block.query_bias);
            ggml_tensor * key             = linear(key_value_source, block.key_weight, block.key_bias);
            ggml_tensor * value           = linear(key_value_source, block.value_weight, block.value_bias);
            query                         = ggml_reshape_3d(graph_context, query, head_dim, heads, query_count);
            key                           = ggml_reshape_3d(graph_context, key, head_dim, heads, key_value_count);
            value                         = ggml_reshape_3d(graph_context, value, head_dim, heads, key_value_count);
            query                         = ggml_permute(graph_context, query, 0, 2, 1, 3);
            key                           = ggml_permute(graph_context, key, 0, 2, 1, 3);
            value                         = ggml_cont(graph_context, ggml_permute(graph_context, value, 1, 2, 0, 3));
            ggml_tensor * scores          = ggml_mul_mat(graph_context, key, query);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores =
                ggml_soft_max_ext(graph_context, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
            ggml_tensor * attended = ggml_mul_mat(graph_context, value, scores);
            ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
            attended = ggml_permute(graph_context, attended, 0, 2, 1, 3);
            attended = ggml_cont_2d(graph_context, attended, width, query_count);
            return linear(attended, block.attention_output_weight, block.attention_output_bias);
        };
        ggml_tensor * future        = f32(weights.future_tokens);
        ggml_tensor * position_view = ggml_view_2d(graph_context, weights.action_position, width, config.horizon,
                                                   weights.action_position->nb[1], 0);
        ggml_tensor * position      = f32(position_view);
        ggml_tensor * actions       = noise_input;

        for (int step = 0; step < 4; ++step) {
            ggml_tensor * timestep_projection =
                ggml_view_1d(graph_context, timestep_projection_input, config.timestep_projection_dim,
                             static_cast<size_t>(step) * config.timestep_projection_dim * sizeof(float));
            ggml_tensor * temb =
                linear(timestep_projection, weights.timestep_input_weight, weights.timestep_input_bias);
            temb = ggml_silu(graph_context, temb);
            temb = linear(temb, weights.timestep_output_weight, weights.timestep_output_bias);

            ggml_tensor * action_features = linear(actions, weights.action_input_weight, weights.action_input_bias);
            ggml_tensor * action_time     = ggml_view_1d(graph_context, action_time_input, width,
                                                         static_cast<size_t>(step) * width * sizeof(float));
            action_time                   = ggml_repeat(graph_context, action_time, action_features);
            action_features               = ggml_concat(graph_context, action_features, action_time, 0);
            action_features = linear(action_features, weights.action_time_mix_weight, weights.action_time_mix_bias);
            action_features = ggml_silu(graph_context, action_features);
            action_features = linear(action_features, weights.action_output_weight, weights.action_output_bias);
            action_features = ggml_add(graph_context, action_features, position);
            ggml_tensor * hidden = ggml_concat(graph_context, future, action_features, 1);
            for (int block_index = 0; block_index < config.block_count; ++block_index) {
                const GR00TBlockWeights & block = weights.blocks[static_cast<size_t>(block_index)];
                ggml_tensor * normalized        = ada_norm(hidden, temb, block);
                ggml_tensor * attended          = block_index % 2 == 0
                                                      ? attention(normalized, hidden_input, cross_mask_input, block)
                                                      : attention(normalized, normalized, nullptr, block);
                hidden                          = ggml_add(graph_context, hidden, attended);
                ggml_tensor * ff                = ggml_norm(graph_context, hidden, config.ada_norm_epsilon);
                ff     = linear(ff, block.feed_forward_input_weight, block.feed_forward_input_bias);
                ff     = ggml_gelu(graph_context, ff);
                ff     = linear(ff, block.feed_forward_output_weight, block.feed_forward_output_bias);
                hidden = ggml_add(graph_context, hidden, ff);
            }

            ggml_tensor * output_modulation = linear(ggml_silu(graph_context, temb), weights.output_modulation_weight,
                                                     weights.output_modulation_bias);
            // DiT output uses shift then scale, unlike AdaLayerNorm's scale then shift.
            ggml_tensor * shift = ggml_view_1d(graph_context, output_modulation, width, 0);
            ggml_tensor * scale =
                ggml_view_1d(graph_context, output_modulation, width, static_cast<size_t>(width) * sizeof(float));
            hidden = ggml_norm(graph_context, hidden, config.output_norm_epsilon);
            hidden = ggml_mul(graph_context, hidden, ggml_add(graph_context, scale, scalar_one_input));
            hidden = ggml_add(graph_context, hidden, shift);
            hidden = linear(hidden, weights.output_projection_weight, weights.output_projection_bias);
            hidden =
                ggml_relu(graph_context, linear(hidden, weights.velocity_input_weight, weights.velocity_input_bias));
            hidden = linear(hidden, weights.velocity_output_weight, weights.velocity_output_bias);
            ggml_tensor * velocity =
                ggml_view_2d(graph_context, hidden, config.action_dim, config.horizon, hidden->nb[1],
                             static_cast<size_t>(config.future_token_count) * hidden->nb[1]);
            actions = ggml_add(graph_context, actions, ggml_scale(graph_context, velocity, config.euler_dt));
        }

        output = actions;
        ggml_set_name(output, "starvla_groot_normalized_actions");
        ggml_set_output(output);
        graph = ggml_new_graph_custom(graph_context, kGraphSize, false);
        if (graph == nullptr) {
            throw std::runtime_error("failed to create StarVLA GR00T graph");
        }
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("failed to allocate StarVLA GR00T graph");
        }

        conditioning_token_count = token_count;
    }
};

GR00TPolicy::GR00TPolicy(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GR00TPolicy::~GR00TPolicy() = default;

std::unique_ptr<GR00TPolicy> GR00TPolicy::load(const std::string & path, int n_threads, int verbosity,
                                               std::string & error) {
    error.clear();
    if (path.empty()) {
        error = "StarVLA GR00T policy path is required";
        return nullptr;
    }

    std::unique_ptr<Impl> impl(new Impl());
    impl->n_threads = n_threads;
    impl->verbosity = verbosity;
    try {
        backend_scheduler_config scheduler_config;
        scheduler_config.max_nodes  = static_cast<int>(kGraphSize);
        scheduler_config.parallel   = false;
        scheduler_config.op_offload = true;
        backend_loader backend;
        if (!backend.load(impl->backend_cpu, impl->backends, impl->scheduler, impl->buft_policy, true, scheduler_config,
                          verbosity)) {
            error = "failed to initialize StarVLA GR00T backend: " + backend.error();
            return nullptr;
        }
        impl->mode = backend.mode();

        GR00TGGUFLoader loader(impl->config, impl->weights);
        if (!loader.load(path.c_str(), impl->buft_policy.model_buft, impl->loaded, verbosity)) {
            error = loader.error();
            return nullptr;
        }
        if (impl->loaded.ctx_data == nullptr || impl->loaded.model_buffer == nullptr) {
            error = "StarVLA GR00T policy GGUF has no tensors";
            return nullptr;
        }
        ggml_backend_buffer_set_usage(impl->loaded.model_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        impl->timestep_table = timestep_projection_table(impl->config);
        impl->action_table   = action_time_table(impl->config);
        if (verbosity >= 1) {
            std::fprintf(stderr, "%s: backend=%s qwen=%d width=%d blocks=%d horizon=%d action_dim=%d profiles=%zu\n",
                         __func__, backend_mode_name(impl->mode), impl->config.qwen_hidden_dim, impl->config.dit_width,
                         impl->config.block_count, impl->config.horizon, impl->config.action_dim,
                         impl->config.normalization.profiles.size());
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
    return std::unique_ptr<GR00TPolicy>(new GR00TPolicy(std::move(impl)));
}

bool GR00TPolicy::evaluate(const float * qwen_hidden_states, size_t hidden_element_count,
                           const uint8_t * qwen_attention_mask, size_t mask_element_count, const float * initial_noise,
                           size_t noise_element_count, std::vector<float> & normalized_actions, std::string & error) {
    normalized_actions.clear();
    error.clear();
    if (impl_ == nullptr || impl_->scheduler == nullptr) {
        error = "StarVLA GR00T policy is not initialized";
        return false;
    }
    if (qwen_hidden_states == nullptr || qwen_attention_mask == nullptr || initial_noise == nullptr ||
        mask_element_count == 0 || mask_element_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        mask_element_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(impl_->config.qwen_hidden_dim) ||
        hidden_element_count != mask_element_count * static_cast<size_t>(impl_->config.qwen_hidden_dim)) {
        error = "StarVLA GR00T Qwen conditioning tensor or attention mask has an incompatible shape";
        return false;
    }
    const size_t expected_noise = static_cast<size_t>(impl_->config.horizon) * impl_->config.action_dim;
    if (noise_element_count != expected_noise) {
        error = "StarVLA GR00T initial-noise tensor has an incompatible shape";
        return false;
    }
    if (std::any_of(qwen_hidden_states, qwen_hidden_states + hidden_element_count,
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(initial_noise, initial_noise + noise_element_count,
                    [](float value) { return !std::isfinite(value); })) {
        error = "StarVLA GR00T conditioning and initial noise must be finite";
        return false;
    }
    bool has_valid_token = false;
    for (size_t token = 0; token < mask_element_count; ++token) {
        if (qwen_attention_mask[token] > 1) {
            error = "StarVLA GR00T attention mask values must be zero or one";
            return false;
        }
        has_valid_token = has_valid_token || qwen_attention_mask[token] != 0;
    }
    if (!has_valid_token) {
        error = "StarVLA GR00T attention mask must contain at least one valid token";
        return false;
    }

    try {
        if (impl_->graph == nullptr || impl_->conditioning_token_count != mask_element_count) {
            impl_->build_graph(mask_element_count);
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }

    const int query_count    = impl_->config.no_state_sequence_length;
    const int padded_queries = GGML_PAD(query_count, kKQMaskPad);
    std::vector<float> additive_mask(mask_element_count * static_cast<size_t>(padded_queries),
                                     -std::numeric_limits<float>::infinity());
    for (int query = 0; query < query_count; ++query) {
        float * row = additive_mask.data() + static_cast<size_t>(query) * mask_element_count;
        for (size_t token = 0; token < mask_element_count; ++token) {
            row[token] = qwen_attention_mask[token] != 0 ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    ggml_backend_tensor_set(impl_->hidden_input, qwen_hidden_states, 0, hidden_element_count * sizeof(float));
    ggml_backend_tensor_set(impl_->cross_mask_input, additive_mask.data(), 0, additive_mask.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->noise_input, initial_noise, 0, noise_element_count * sizeof(float));
    ggml_backend_tensor_set(impl_->timestep_projection_input, impl_->timestep_table.data(), 0,
                            impl_->timestep_table.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->action_time_input, impl_->action_table.data(), 0,
                            impl_->action_table.size() * sizeof(float));
    const float one = 1.0f;
    ggml_backend_tensor_set(impl_->scalar_one_input, &one, 0, sizeof(one));
    set_backend_threads(impl_->backends, impl_->n_threads);
    if (ggml_backend_sched_graph_compute(impl_->scheduler, impl_->graph) != GGML_STATUS_SUCCESS) {
        error = "StarVLA GR00T graph compute failed";
        return false;
    }

    normalized_actions.resize(expected_noise);
    ggml_backend_tensor_get(impl_->output, normalized_actions.data(), 0, expected_noise * sizeof(float));
    if (std::any_of(normalized_actions.begin(), normalized_actions.end(),
                    [](float value) { return !std::isfinite(value); })) {
        normalized_actions.clear();
        error = "StarVLA GR00T graph produced non-finite actions";
        return false;
    }
    return true;
}

bool GR00TPolicy::unnormalize(const std::vector<float> & normalized_actions, const std::string & profile_key_value,
                              std::vector<float> & actions, std::string & error) const {
    if (impl_ == nullptr) {
        actions.clear();
        error = "StarVLA GR00T policy is not initialized";
        return false;
    }
    return denormalize_actions(impl_->config.normalization, profile_key_value, normalized_actions,
                               impl_->config.horizon, impl_->config.action_dim, actions, error);
}

const GR00TPolicyConfig & GR00TPolicy::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("StarVLA GR00T policy is not initialized");
    }
    return impl_->config;
}

const char * GR00TPolicy::backend_name() const {
    return impl_ != nullptr ? backend_mode_name(impl_->mode) : "unknown";
}

} // namespace robotcpp::starvla
