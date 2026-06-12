#pragma once

#include "models/ggml_runtime.h"
#include "models/pi0/load.h"
#include "models/pi0/vlm.h"
#include "models/pi0/weights.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vlacpp {

struct Pi0PrefixKvRuntime {
    size_t token_count = 0;
    uint64_t generation = 0;
    std::vector<ggml_tensor *> k_layers;
    std::vector<ggml_tensor *> v_layers;

    void reset() {
        token_count = 0;
        ++generation;
        k_layers.clear();
        v_layers.clear();
    }
};

struct Pi0ComponentRuntime {
    TensorMap tensors;
    std::unique_ptr<GgmlRunner> runner;
    ggml_type default_2d_weight_type = GGML_TYPE_F32;
};

struct Pi0Components {
    Pi0ComponentRuntime vit;
    Pi0ComponentRuntime mmproj;
    Pi0ComponentRuntime llm;
    Pi0ComponentRuntime state;
    Pi0ComponentRuntime action_decoder;
};

struct Pi0Context {
    ModelConfig config;
    Pi0Components components;
    Pi0Weights weights;
    std::unordered_map<const Tensor *, const GgmlRunner *> tensor_runners;
    Pi0Tokenizer tokenizer;
    mutable Pi0PrefixKvRuntime prefix_kv;

    Pi0Context(
        ModelConfig model_config,
        const BackendConfig & backend,
        const std::string & tokenizer_path,
        Pi0ComponentTensors model_tensors)
        : config(std::move(model_config)),
          components(make_components(config, backend, std::move(model_tensors))),
          weights(build_pi0_weights(
              config,
              components.vit.tensors,
              components.mmproj.tensors,
              components.llm.tensors,
              components.state.tensors,
              components.action_decoder.tensors)),
          tokenizer(tokenizer_path) {
        register_tensor_runners();
    }

    const GgmlRunner & runner_for_tensor(const Tensor & tensor) const {
        auto it = tensor_runners.find(&tensor);
        if (it == tensor_runners.end() || it->second == nullptr) {
            throw std::invalid_argument("missing runner for pi0 tensor: " + tensor_name(tensor));
        }
        return *it->second;
    }

    std::string tensor_name(const Tensor & tensor) const {
        const TensorMap * maps[] = {
            &components.vit.tensors,
            &components.mmproj.tensors,
            &components.llm.tensors,
            &components.state.tensors,
            &components.action_decoder.tensors,
        };
        for (const TensorMap * tensors : maps) {
            for (const auto & item : *tensors) {
                if (&item.second == &tensor) {
                    return item.first;
                }
            }
        }
        return "<unknown>";
    }

private:
    static BackendConfig component_backend(const BackendConfig & base, const VLAComponentConfig & component) {
        BackendConfig out = base;
        if (component.runtime.backend == "cpu") {
            out.backend = VLACPP_BACKEND_CPU;
        } else if (component.runtime.backend == "cuda") {
            out.backend = VLACPP_BACKEND_CUDA;
        } else if (component.runtime.backend != "inherit") {
            throw std::invalid_argument("unsupported pi0 component backend: " + component.runtime.backend);
        }
        if (component.runtime.n_threads > 0) {
            out.n_threads = component.runtime.n_threads;
        }
        return out;
    }

    static ggml_type component_weight_type(const VLAComponentConfig & config) {
        if (config.runtime.data_type == "preserve") {
            return GGML_TYPE_COUNT;
        }
        if (config.runtime.data_type == "fp32") {
            return GGML_TYPE_F32;
        }
        if (config.runtime.data_type == "f16") {
            return GGML_TYPE_F16;
        }
        if (config.runtime.data_type == "bf16") {
            return GGML_TYPE_BF16;
        }
        throw std::invalid_argument("unsupported pi0 component dtype: " + config.runtime.data_type);
    }

    static Pi0ComponentRuntime make_component(
        const VLAComponentConfig & config,
        const BackendConfig & backend) {
        Pi0ComponentRuntime component;
        component.runner = std::make_unique<GgmlRunner>(component_backend(backend, config));
        component.default_2d_weight_type = component_weight_type(config);
        return component;
    }

    static Pi0Components make_components(
        const ModelConfig & model_config,
        const BackendConfig & backend,
        Pi0ComponentTensors model_tensors) {
        const Pi0Config & pi0 = pi0_config(model_config);
        Pi0Components out;
        out.vit = make_component(pi0.vision.component, backend);
        out.mmproj = make_component(pi0.mmproj.component, backend);
        out.llm = make_component(pi0.llm.component, backend);
        out.state = make_component(pi0.state.component, backend);
        out.action_decoder = make_component(pi0.action.component, backend);
        out.vit.tensors = std::move(model_tensors.vit);
        out.mmproj.tensors = std::move(model_tensors.mmproj);
        out.llm.tensors = std::move(model_tensors.llm);
        out.state.tensors = std::move(model_tensors.state);
        out.action_decoder.tensors = std::move(model_tensors.action_decoder);
        return out;
    }

    void register_tensor_runners() {
        tensor_runners.reserve(
            components.vit.tensors.size() +
            components.mmproj.tensors.size() +
            components.llm.tensors.size() +
            components.state.tensors.size() +
            components.action_decoder.tensors.size());
        register_tensor_runners(components.vit);
        register_tensor_runners(components.mmproj);
        register_tensor_runners(components.llm);
        register_tensor_runners(components.state);
        register_tensor_runners(components.action_decoder);
    }

    void register_tensor_runners(const Pi0ComponentRuntime & component) {
        for (const auto & item : component.tensors) {
            tensor_runners[&item.second] = component.runner.get();
        }
    }
};

inline const Tensor & pi0_tensor(const Tensor * tensor, const char * name) {
    if (tensor == nullptr) {
        throw std::invalid_argument(std::string("missing pi0 tensor: ") + name);
    }
    return *tensor;
}

inline std::string pi0_tensor_name(const Pi0Context & ctx, const Tensor & tensor) {
    return ctx.tensor_name(tensor);
}

inline ggml_tensor * pi0_weight(const Pi0Context & ctx, const Tensor & tensor, size_t rank, ggml_type required_type = GGML_TYPE_COUNT) {
    if (tensor.shape.size() != rank) {
        throw std::invalid_argument("pi0 weight has incompatible rank: " + pi0_tensor_name(ctx, tensor));
    }
    ggml_tensor * stored = ctx.runner_for_tensor(tensor).stored_weight(tensor);
    if (stored == nullptr) {
        throw std::invalid_argument("missing materialized pi0 weight: " + pi0_tensor_name(ctx, tensor));
    }
    if (required_type != GGML_TYPE_COUNT && stored->type != required_type) {
        throw std::invalid_argument("pi0 materialized weight has incompatible type: " + pi0_tensor_name(ctx, tensor));
    }
    return stored;
}

inline void pi0_linear_batch(
    const Pi0Context & ctx,
    const Tensor & weight,
    const Tensor & bias,
    const std::vector<float> & input,
    int batch,
    std::vector<float> & output,
    const char * label,
    bool silu = false) {
    if (weight.shape.size() != 2 || bias.shape.size() != 1) {
        throw std::invalid_argument("pi0 linear expects rank-2 weight and rank-1 bias");
    }
    const int64_t in = weight.shape[0];
    const int64_t out = weight.shape[1];
    if (in <= 0 || out <= 0 || batch <= 0 ||
        bias.shape[0] != out ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(in)) {
        throw std::invalid_argument("pi0 linear input has incompatible shape");
    }

    const size_t tensor_bytes =
        (static_cast<size_t>(in) * static_cast<size_t>(out) + static_cast<size_t>(out) + input.size()) *
        sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(out) * sizeof(float);
    static const char graph_tag = 0;
    const GgmlRunner & runner = ctx.runner_for_tensor(weight);
    GgmlContext gctx(runner.init_params(ggml_graph_context_size(tensor_bytes), &graph_tag));

    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, in, batch);
    std::vector<GgmlInput> inputs;
    runner.set_input(inputs, x, input.data(), input.size() * sizeof(float));

    ggml_tensor * y = ggml_add(gctx, ggml_mul_mat(gctx, pi0_weight(ctx, weight, 2), x), pi0_weight(ctx, bias, 1));
    if (silu) {
        y = ggml_silu(gctx, y);
    }
    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, y);
    runner.compute(gctx, graph, inputs, label);

    output.resize(static_cast<size_t>(batch) * static_cast<size_t>(out));
    runner.get_output(y, output.data(), output.size() * sizeof(float));
}

} // namespace vlacpp
