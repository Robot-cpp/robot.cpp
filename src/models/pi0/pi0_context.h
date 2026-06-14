#pragma once

#include "models/pi0/component_runtime.h"
#include "models/pi0/load.h"
#include "models/pi0/vlm.h"
#include "models/pi0/weights.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace robotcpp::pi0 {

struct Pi0PrefixKvRuntime {
    size_t token_count = 0;
    std::vector<ggml_tensor *> k_layers;
    std::vector<ggml_tensor *> v_layers;
    ggml_tensor * vision_prefix_embeddings = nullptr;
    ggml_tensor * prompt_embeddings = nullptr;

    void reset() {
        token_count = 0;
    }
};

struct Pi0Context {
    Pi0ModelConfig config;
    Pi0Components components;
    Pi0Weights weights;
    Pi0Tokenizer tokenizer;
    mutable Pi0PrefixKvRuntime prefix_kv;

    Pi0Context(
        Pi0ModelConfig model_config,
        const std::string & tokenizer_path,
        Pi0Components model_components)
        : config(std::move(model_config)),
          components(std::move(model_components)),
          weights(build_pi0_weights(
              config,
              components.vit.loaded.ctx_data,
              components.mmproj.loaded.ctx_data,
              components.llm.loaded.ctx_data,
              components.state.loaded.ctx_data,
              components.action_decoder.loaded.ctx_data)),
          tokenizer(tokenizer_path) {}

    std::string tensor_name(const ggml_tensor * tensor) const {
        if (tensor != nullptr) {
            return ggml_get_name(tensor);
        }
        return "<unknown>";
    }

};

inline ggml_tensor * pi0_tensor(ggml_tensor * tensor, const char * name) {
    if (tensor == nullptr) {
        throw std::invalid_argument(std::string("missing pi0 tensor: ") + name);
    }
    return tensor;
}

inline std::string pi0_tensor_name(const Pi0Context & ctx, const ggml_tensor * tensor) {
    return ctx.tensor_name(tensor);
}

inline ggml_tensor * pi0_weight(const Pi0Context & ctx, ggml_tensor * tensor, size_t rank, ggml_type required_type = GGML_TYPE_COUNT) {
    if (tensor == nullptr || static_cast<size_t>(ggml_n_dims(tensor)) != rank) {
        throw std::invalid_argument("pi0 weight has incompatible rank: " + pi0_tensor_name(ctx, tensor));
    }
    if (required_type != GGML_TYPE_COUNT && tensor->type != required_type) {
        throw std::invalid_argument("pi0 weight has incompatible type: " + pi0_tensor_name(ctx, tensor));
    }
    return tensor;
}

inline ggml_tensor * pi0_f32_weight(
    ggml_context * gctx,
    const Pi0Context & ctx,
    ggml_tensor * tensor,
    size_t rank) {
    ggml_tensor * weight = pi0_weight(ctx, tensor, rank);
    return weight->type == GGML_TYPE_F32 ? weight : ggml_cast(gctx, weight, GGML_TYPE_F32);
}

inline void pi0_linear_batch(
    const Pi0Context & ctx,
    const Pi0ComponentRuntime & runtime,
    ggml_tensor * weight,
    ggml_tensor * bias,
    const std::vector<float> & input,
    int batch,
    std::vector<float> & output,
    const char * label,
    bool silu = false) {
    if (weight == nullptr || bias == nullptr || ggml_n_dims(weight) != 2 || ggml_n_dims(bias) != 1) {
        throw std::invalid_argument("pi0 linear expects rank-2 weight and rank-1 bias");
    }
    const int64_t in = weight->ne[0];
    const int64_t out = weight->ne[1];
    if (in <= 0 || out <= 0 || batch <= 0 ||
        bias->ne[0] != out ||
        input.size() != static_cast<size_t>(batch) * static_cast<size_t>(in)) {
        throw std::invalid_argument("pi0 linear input has incompatible shape");
    }

    const size_t tensor_bytes =
        (static_cast<size_t>(in) * static_cast<size_t>(out) + static_cast<size_t>(out) + input.size()) *
        sizeof(float) +
        static_cast<size_t>(batch) * static_cast<size_t>(out) * sizeof(float);
    static const char graph_tag = 0;
    (void) graph_tag;
    Pi0GraphContext gctx(pi0_graph_init_params(pi0_graph_context_size(tensor_bytes)));

    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, in, batch);
    ggml_set_input(x);

    ggml_tensor * y = ggml_add(
        gctx,
        ggml_mul_mat(gctx, pi0_weight(ctx, weight, 2), x),
        pi0_f32_weight(gctx, ctx, bias, 1));
    if (silu) {
        y = ggml_silu(gctx, y);
    }
    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_sched_reset(runtime.sched);
    if (!ggml_backend_sched_alloc_graph(runtime.sched, graph)) {
        throw std::runtime_error("pi0 linear graph allocation failed");
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    set_backend_threads(runtime.backends, runtime.n_threads);
    if (ggml_backend_sched_graph_compute(runtime.sched, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(label);
    }

    output.resize(static_cast<size_t>(batch) * static_cast<size_t>(out));
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
}

} // namespace robotcpp::pi0
