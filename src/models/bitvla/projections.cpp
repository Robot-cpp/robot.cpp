#include "component.h"
#include "ops.h"

#include <algorithm>

namespace robotcpp::bitvla {
void build_proprio(component & model) {
    auto * ctx  = model.begin_graph();
    model.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.integer("vla.proprio_dim"), 1);
    ggml_set_input(model.input);
    auto * cur = linear(ctx, model.tensor("vla.proprio.0.weight"), model.input, model.tensor("vla.proprio.0.bias"));
    cur        = ggml_gelu_erf(ctx, cur);
    cur        = linear(ctx, model.tensor("vla.proprio.2.weight"), cur, model.tensor("vla.proprio.2.bias"));
    model.finish_graph(cur);
}
std::vector<float> encode_proprio(component & model, const std::vector<float> & state) {
    const std::string key = "vla.norm_stats." + model.text("bitvla.dataset_name") + ".proprio.";
    const auto low = model.array(key + "q01"), high = model.array(key + "q99"), mask = model.array(key + "mask");
    std::vector<float> normalized(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        const double value =
            mask[i] ? std::clamp(2.0 * (state[i] - double(low[i])) / (double(high[i]) - low[i] + 1e-8) - 1.0, -1.0, 1.0)
                    : state[i];
        normalized[i] = static_cast<float>(value);
    }
    ggml_backend_tensor_set(model.input, normalized.data(), 0, normalized.size() * sizeof(float));
    model.compute();
    auto result = model.read(model.output);
    return result;
}
void build_action(component & model) {
    auto * ctx          = model.begin_graph();
    const int dimension = model.integer("vla.action_dim");
    const int chunks    = model.integer("vla.action_chunk_size");
    const int width     = model.integer("vla.hidden_size");
    const float eps     = model.scalar("vla.action_head.layer_norm_epsilon");
    model.input         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width * dimension, chunks);
    ggml_set_input(model.input);
    auto weight = [&](const std::string & name) { return model.tensor("vla.action_head." + name + ".weight"); };
    auto bias   = [&](const std::string & name) { return model.tensor("vla.action_head." + name + ".bias"); };
    auto * cur  = layer_norm(ctx, model.input, weight("ln1"), bias("ln1"), eps);
    cur         = ggml_relu(ctx, linear(ctx, weight("fc1"), cur, bias("fc1")));
    for (int block = 0; block < 2; ++block) {
        const std::string prefix = "resnet." + std::to_string(block) + ".";
        auto * residual          = cur;
        cur                      = layer_norm(ctx, cur, weight(prefix + "0"), bias(prefix + "0"), eps);
        cur                      = ggml_relu(ctx, linear(ctx, weight(prefix + "1"), cur, bias(prefix + "1")));
        cur                      = ggml_add(ctx, residual, cur);
    }
    cur = layer_norm(ctx, cur, weight("ln2"), bias("ln2"), eps);
    cur = linear(ctx, weight("fc2"), cur, bias("fc2"));
    model.finish_graph(cur);
}
static std::vector<float> read_action_result(component & model) {
    auto actions          = model.read(model.output);
    const std::string key = "vla.norm_stats." + model.text("bitvla.dataset_name") + ".action.";
    const auto low = model.array(key + "q01"), high = model.array(key + "q99"), mask = model.array(key + "mask");
    for (size_t i = 0; i < actions.size(); ++i) {
        const size_t d = i % low.size();
        if (mask[d])
            actions[i] = 0.5 * (actions[i] + 1.0) * (double(high[d]) - low[d] + 1e-8) + low[d];
    }
    return actions;
}

std::vector<float> decode_action(component & model, const std::vector<float> & hidden) {
    ggml_backend_tensor_set(model.input, hidden.data(), 0, hidden.size() * sizeof(float));
    model.compute();
    return read_action_result(model);
}

void copy_language_to_action_host_input(ggml_tensor * destination, const ggml_tensor * source) {
    GGML_ASSERT(destination->type == GGML_TYPE_F32 && source->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(destination) && ggml_is_contiguous(source));
    GGML_ASSERT(ggml_nelements(destination) == ggml_nelements(source));
    ggml_tensor src = *source, dst = *destination;
    // Hidden rows and action chunks describe exactly the same contiguous F32 bytes.
    for (int i = 0; i < 4; ++i) {
        src.ne[i] = dst.ne[i];
        src.nb[i] = dst.nb[i];
    }
    const ggml_tensor * owner = source;
    while (!owner->buffer && owner->view_src)
        owner = owner->view_src;
    GGML_ASSERT(owner->buffer);
    src.buffer = owner->buffer;
    owner      = destination;
    while (!owner->buffer && owner->view_src)
        owner = owner->view_src;
    GGML_ASSERT(owner->buffer);
    dst.buffer   = owner->buffer;
    src.view_src = dst.view_src = nullptr;
    src.view_offs = dst.view_offs = 0;
    // The original action input remains host memory with its original graph placement.
    GGML_ASSERT(ggml_backend_buffer_is_host(dst.buffer));
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    if (ggml_backend_buffer_is_host(src.buffer)) {
        GGML_ASSERT(source->ne[0] == 2560 && source->ne[1] == 56 && source->ne[2] == 1 && source->ne[3] == 1);
        GGML_ASSERT(destination->ne[0] == 17920 && destination->ne[1] == 8 && destination->ne[2] == 1 &&
                    destination->ne[3] == 1);
        // The pinned host-source dispatch calls tensor_set directly; no temporary vector.
        ggml_backend_tensor_copy(&src, &dst);
        return;
    }
#endif
    GGML_ASSERT(!ggml_backend_buffer_is_host(src.buffer));
    // Pinned GGML dispatch performs the original D2H directly into this host input.
    ggml_backend_tensor_copy(&src, &dst);
}

std::vector<float> decode_action_direct_host_input(component & model, const ggml_tensor * hidden) {
    copy_language_to_action_host_input(model.input, hidden);
    model.compute();
    return read_action_result(model);
}

void encode_proprio_to_host(component & model, const std::vector<float> & state, std::vector<float> & result,
                            size_t offset) {
    const size_t count = ggml_nelements(model.output);
    GGML_ASSERT(&result != &state && model.output->type == GGML_TYPE_F32 && ggml_is_contiguous(model.output));
    GGML_ASSERT(result.size() >= count && offset <= result.size() - count);
    const std::string key = "vla.norm_stats." + model.text("bitvla.dataset_name") + ".proprio.";
    const auto low = model.array(key + "q01"), high = model.array(key + "q99"), mask = model.array(key + "mask");
    std::vector<float> normalized(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        const double value =
            mask[i] ? std::clamp(2.0 * (state[i] - double(low[i])) / (double(high[i]) - low[i] + 1e-8) - 1.0, -1.0, 1.0)
                    : state[i];
        normalized[i] = static_cast<float>(value);
    }
    ggml_backend_tensor_set(model.input, normalized.data(), 0, normalized.size() * sizeof(float));
    model.compute();
    ggml_backend_tensor_get(model.output, result.data() + offset, 0, count * sizeof(float));
}
} // namespace robotcpp::bitvla
