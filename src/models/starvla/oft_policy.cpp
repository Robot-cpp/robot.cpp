#include "models/starvla/oft_policy.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "models/ggml_backend.h"
#include "models/gguf_loader.h"
#include "models/starvla/policy_gguf.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::starvla {

namespace {

struct OFTBlockWeights {
    ggml_tensor * norm_weight = nullptr;
    ggml_tensor * norm_bias = nullptr;
    ggml_tensor * linear_weight = nullptr;
    ggml_tensor * linear_bias = nullptr;
};

struct OFTWeights {
    ggml_tensor * input_norm_weight = nullptr;
    ggml_tensor * input_norm_bias = nullptr;
    ggml_tensor * input_proj_weight = nullptr;
    ggml_tensor * input_proj_bias = nullptr;
    std::vector<OFTBlockWeights> blocks;
    ggml_tensor * output_norm_weight = nullptr;
    ggml_tensor * output_norm_bias = nullptr;
    ggml_tensor * output_proj_weight = nullptr;
    ggml_tensor * output_proj_bias = nullptr;
};

using detail::has_shape;
using detail::require_bool;
using detail::require_f32;
using detail::require_i32;
using detail::require_string;
using detail::require_string_array;

class OFTGGUFLoader final : public gguf_loader {
  public:
    OFTGGUFLoader(OFTPolicyConfig & config, OFTWeights & weights) : config_(config), weights_(weights) {}

  protected:
    bool parse_metadata(gguf_context * gguf) override {
        const std::string architecture = require_string(gguf, "general.architecture");
        if (architecture != "starvla-policy") {
            throw std::runtime_error("StarVLA policy GGUF has incompatible general.architecture: " + architecture);
        }
        if (require_i32(gguf, "starvla.schema_version") != 1 ||
            require_string(gguf, "starvla.framework") != "oft") {
            throw std::runtime_error("StarVLA policy GGUF is not a supported Qwen OFT schema");
        }
        config_.backbone_arch = require_string(gguf, "starvla.backbone.arch");
        if (config_.backbone_arch != "qwen3_vl" &&
            config_.backbone_arch != "qwen2_5_vl") {
            throw std::runtime_error(
                "StarVLA OFT policy has an unsupported Qwen backbone");
        }

        config_.bundle_uuid = require_string(gguf, "starvla.bundle.uuid");
        if (config_.bundle_uuid.empty()) {
            throw std::runtime_error("StarVLA policy bundle UUID is missing");
        }
        config_.text_filename = require_string(gguf, "starvla.component.text.filename");
        config_.mmproj_filename = require_string(gguf, "starvla.component.mmproj.filename");
        if (config_.text_filename.empty() || config_.mmproj_filename.empty()) {
            throw std::runtime_error("StarVLA policy component filenames must be non-empty");
        }
        config_.input_dim = require_i32(gguf, "starvla.qwen.hidden_size");
        config_.input_embedding_dim = require_i32(gguf, "starvla.qwen.input_embedding_size");
        config_.vocab_size = require_i32(gguf, "starvla.qwen.vocab_size");
        config_.hidden_dim = require_i32(gguf, "starvla.oft.hidden_size");
        config_.block_count = require_i32(gguf, "starvla.oft.block_count");
        config_.action_dim = require_i32(gguf, "starvla.action.dimension");
        config_.horizon = require_i32(gguf, "starvla.action.horizon");
        config_.layer_norm_epsilon = require_f32(gguf, "starvla.oft.layer_norm_epsilon");
        if (config_.input_dim <= 0 || config_.input_embedding_dim <= 0 ||
            config_.vocab_size <= 0 || config_.hidden_dim <= 0 ||
            config_.block_count <= 0 ||
            config_.action_dim <= 0 || config_.horizon <= 0 || !std::isfinite(config_.layer_norm_epsilon) ||
            config_.layer_norm_epsilon <= 0.0f) {
            throw std::runtime_error("StarVLA OFT policy metadata has incompatible dimensions");
        }

        config_.prompt.horizon = config_.horizon;
        config_.prompt.action_token = require_string(gguf, "starvla.prompt.action_token");
        config_.prompt.action_suffix = require_string(gguf, "starvla.prompt.action_suffix");
        config_.prompt.cot_enabled = require_bool(gguf, "starvla.prompt.cot_enabled");
        config_.prompt.cot_template = require_string(gguf, "starvla.prompt.cot_template");
        config_.prompt.state_bins = require_i32(gguf, "starvla.prompt.state_bins");
        config_.prompt.state_bin_min = require_f32(gguf, "starvla.prompt.state_bin_min");
        config_.prompt.state_bin_max = require_f32(gguf, "starvla.prompt.state_bin_max");
        config_.prompt.state_clip = require_bool(gguf, "starvla.prompt.state_clip");
        config_.action_token_id = require_i32(gguf, "starvla.prompt.action_token_id");
        std::string prompt_error;
        if (!validate_oft_prompt_config(config_.prompt, prompt_error) || config_.action_token_id < 0) {
            throw std::runtime_error(prompt_error.empty() ?
                                         "StarVLA OFT token/template metadata is incompatible" :
                                         prompt_error);
        }
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
        if (config_.image_count <= 0 ||
            config_.image_names.size() != static_cast<size_t>(config_.image_count) ||
            config_.image_processor_min_pixels <= 0 ||
            config_.image_processor_max_pixels < config_.image_processor_min_pixels ||
            config_.image_patch_size <= 0 ||
            config_.image_spatial_merge_size <= 0 ||
            config_.image_min_token_count <= 0 ||
            config_.image_max_token_count < config_.image_min_token_count) {
            throw std::runtime_error("StarVLA OFT image metadata is incompatible");
        }

        config_.normalization = detail::require_normalization(gguf, config_.action_dim);
        return true;
    }

    bool bind_tensors(ggml_context * ctx_data) override {
        weights_.input_norm_weight = require_tensor(ctx_data, "starvla.policy.oft.input_norm.weight");
        weights_.input_norm_bias = require_tensor(ctx_data, "starvla.policy.oft.input_norm.bias");
        weights_.input_proj_weight = require_tensor(ctx_data, "starvla.policy.oft.input_proj.weight");
        weights_.input_proj_bias = require_tensor(ctx_data, "starvla.policy.oft.input_proj.bias");
        weights_.blocks.clear();
        weights_.blocks.reserve(static_cast<size_t>(config_.block_count));
        for (int block = 0; block < config_.block_count; ++block) {
            const std::string prefix = "starvla.policy.oft.block." + std::to_string(block) + ".";
            OFTBlockWeights current;
            current.norm_weight = require_tensor(ctx_data, prefix + "norm.weight");
            current.norm_bias = require_tensor(ctx_data, prefix + "norm.bias");
            current.linear_weight = require_tensor(ctx_data, prefix + "linear.weight");
            current.linear_bias = require_tensor(ctx_data, prefix + "linear.bias");
            weights_.blocks.push_back(current);
        }
        weights_.output_norm_weight = require_tensor(ctx_data, "starvla.policy.oft.output_norm.weight");
        weights_.output_norm_bias = require_tensor(ctx_data, "starvla.policy.oft.output_norm.bias");
        weights_.output_proj_weight = require_tensor(ctx_data, "starvla.policy.oft.output_proj.weight");
        weights_.output_proj_bias = require_tensor(ctx_data, "starvla.policy.oft.output_proj.bias");

        if (!has_shape(weights_.input_norm_weight, {config_.input_dim}) ||
            !has_shape(weights_.input_norm_bias, {config_.input_dim}) ||
            !has_shape(weights_.input_proj_weight, {config_.input_dim, config_.hidden_dim}) ||
            !has_shape(weights_.input_proj_bias, {config_.hidden_dim}) ||
            !has_shape(weights_.output_norm_weight, {config_.hidden_dim}) ||
            !has_shape(weights_.output_norm_bias, {config_.hidden_dim}) ||
            !has_shape(weights_.output_proj_weight, {config_.hidden_dim, config_.action_dim}) ||
            !has_shape(weights_.output_proj_bias, {config_.action_dim})) {
            throw std::runtime_error("StarVLA OFT projection tensor has an incompatible ggml shape");
        }
        for (const OFTBlockWeights & block : weights_.blocks) {
            if (!has_shape(block.norm_weight, {config_.hidden_dim}) ||
                !has_shape(block.norm_bias, {config_.hidden_dim}) ||
                !has_shape(block.linear_weight, {config_.hidden_dim, config_.hidden_dim}) ||
                !has_shape(block.linear_bias, {config_.hidden_dim})) {
                throw std::runtime_error("StarVLA OFT residual block tensor has an incompatible ggml shape");
            }
        }
        return true;
    }

  private:
    OFTPolicyConfig & config_;
    OFTWeights & weights_;
};

} // namespace

struct OFTPolicy::Impl {
    OFTPolicyConfig config;
    OFTWeights weights;
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
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    ~Impl() {
        if (scheduler != nullptr) {
            ggml_backend_sched_synchronize(scheduler);
            ggml_backend_sched_free(scheduler);
            scheduler = nullptr;
        }
        if (graph_context != nullptr) {
            ggml_free(graph_context);
            graph_context = nullptr;
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

    void build_graph() {
        const size_t graph_size = GGML_DEFAULT_GRAPH_SIZE;
        ggml_init_params params{};
        params.mem_size = graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph_context = ggml_init(params);
        if (graph_context == nullptr) {
            throw std::runtime_error("failed to initialize StarVLA OFT graph context");
        }

        auto f32_vector = [&](ggml_tensor * tensor) {
            return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(graph_context, tensor, GGML_TYPE_F32);
        };
        auto layer_norm = [&](ggml_tensor * value, ggml_tensor * weight, ggml_tensor * bias) {
            ggml_tensor * normalized = ggml_norm(graph_context, value, config.layer_norm_epsilon);
            normalized = ggml_mul(graph_context, normalized, f32_vector(weight));
            return ggml_add(graph_context, normalized, f32_vector(bias));
        };
        auto linear = [&](ggml_tensor * value, ggml_tensor * weight, ggml_tensor * bias) {
            ggml_tensor * projected = ggml_mul_mat(graph_context, weight, value);
            ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
            return ggml_add(graph_context, projected, f32_vector(bias));
        };

        input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, config.input_dim, config.horizon);
        ggml_set_name(input, "starvla_oft_action_queries");
        ggml_set_input(input);

        ggml_tensor * current = layer_norm(input, weights.input_norm_weight, weights.input_norm_bias);
        current = ggml_relu(graph_context, linear(current, weights.input_proj_weight, weights.input_proj_bias));
        for (const OFTBlockWeights & block : weights.blocks) {
            ggml_tensor * residual = current;
            current = layer_norm(current, block.norm_weight, block.norm_bias);
            current = ggml_relu(graph_context, linear(current, block.linear_weight, block.linear_bias));
            current = ggml_add(graph_context, current, residual);
        }
        current = layer_norm(current, weights.output_norm_weight, weights.output_norm_bias);
        output = linear(current, weights.output_proj_weight, weights.output_proj_bias);
        ggml_set_name(output, "starvla_oft_normalized_actions");
        ggml_set_output(output);

        graph = ggml_new_graph_custom(graph_context, graph_size, false);
        if (graph == nullptr) {
            throw std::runtime_error("failed to create StarVLA OFT graph");
        }
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_reset(scheduler);
        if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
            throw std::runtime_error("failed to allocate StarVLA OFT graph");
        }
    }
};

OFTPolicy::OFTPolicy(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

OFTPolicy::~OFTPolicy() = default;

std::unique_ptr<OFTPolicy> OFTPolicy::load(const std::string & path, int n_threads, int verbosity,
                                           std::string & error) {
    error.clear();
    if (path.empty()) {
        error = "StarVLA OFT policy path is required";
        return nullptr;
    }

    std::unique_ptr<Impl> impl(new Impl());
    impl->n_threads = n_threads;
    impl->verbosity = verbosity;
    try {
        backend_scheduler_config scheduler_config;
        scheduler_config.max_nodes = GGML_DEFAULT_GRAPH_SIZE;
        scheduler_config.parallel = false;
        scheduler_config.op_offload = true;
        backend_loader backend;
        if (!backend.load(impl->backend_cpu, impl->backends, impl->scheduler, impl->buft_policy, true,
                          scheduler_config, verbosity)) {
            error = "failed to initialize StarVLA OFT backend: " + backend.error();
            return nullptr;
        }
        impl->mode = backend.mode();

        OFTGGUFLoader loader(impl->config, impl->weights);
        if (!loader.load(path.c_str(), impl->buft_policy.model_buft, impl->loaded, verbosity)) {
            error = loader.error();
            return nullptr;
        }
        if (impl->loaded.ctx_data == nullptr || impl->loaded.model_buffer == nullptr) {
            error = "StarVLA OFT policy GGUF has no tensors";
            return nullptr;
        }
        ggml_backend_buffer_set_usage(impl->loaded.model_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        impl->build_graph();
        if (verbosity >= 1) {
            std::fprintf(stderr,
                         "%s: backend=%s input=%d hidden=%d blocks=%d horizon=%d action_dim=%d profiles=%zu\n",
                         __func__, backend_mode_name(impl->mode), impl->config.input_dim, impl->config.hidden_dim,
                         impl->config.block_count, impl->config.horizon, impl->config.action_dim,
                         impl->config.normalization.profiles.size());
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
    return std::unique_ptr<OFTPolicy>(new OFTPolicy(std::move(impl)));
}

bool OFTPolicy::evaluate(const float * action_queries, size_t element_count,
                         std::vector<float> & normalized_actions, std::string & error) {
    normalized_actions.clear();
    error.clear();
    if (impl_ == nullptr) {
        error = "StarVLA OFT policy is not initialized";
        return false;
    }
    const size_t expected = static_cast<size_t>(impl_->config.horizon) *
                            static_cast<size_t>(impl_->config.input_dim);
    if (action_queries == nullptr || element_count != expected) {
        error = "StarVLA OFT action-query tensor has an incompatible shape";
        return false;
    }

    if (impl_->scheduler == nullptr || impl_->graph == nullptr || impl_->input == nullptr ||
        impl_->output == nullptr) {
        error = "StarVLA OFT policy graph is not initialized";
        return false;
    }

    ggml_backend_tensor_set(impl_->input, action_queries, 0, element_count * sizeof(float));
    set_backend_threads(impl_->backends, impl_->n_threads);
    if (ggml_backend_sched_graph_compute(impl_->scheduler, impl_->graph) != GGML_STATUS_SUCCESS) {
        error = "StarVLA OFT graph compute failed";
        return false;
    }

    const size_t output_count = static_cast<size_t>(impl_->config.horizon) *
                                static_cast<size_t>(impl_->config.action_dim);
    normalized_actions.resize(output_count);
    ggml_backend_tensor_get(impl_->output, normalized_actions.data(), 0, output_count * sizeof(float));
    return true;
}

bool OFTPolicy::unnormalize(const std::vector<float> & normalized_actions, const std::string & profile_key,
                            std::vector<float> & actions, std::string & error) const {
    if (impl_ == nullptr) {
        actions.clear();
        error = "StarVLA OFT policy is not initialized";
        return false;
    }
    return denormalize_actions(impl_->config.normalization, profile_key, normalized_actions, impl_->config.horizon,
                               impl_->config.action_dim, actions, error);
}

const OFTPolicyConfig & OFTPolicy::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("StarVLA OFT policy is not initialized");
    }
    return impl_->config;
}

const char * OFTPolicy::backend_name() const {
    return impl_ != nullptr ? backend_mode_name(impl_->mode) : "unknown";
}

} // namespace robotcpp::starvla
