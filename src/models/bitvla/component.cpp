#include "component.h"
#include "ops.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cstring>

namespace robotcpp::bitvla {
component::graph_state::~graph_state() {
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (ctx) ggml_free(ctx);
}

void component::synchronize_backends() {
    for (auto backend : backends) ggml_backend_synchronize(backend);
}

component::~component() {
    synchronize_backends();
    language_graphs.clear();
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (ctx) ggml_free(ctx);
    if (weights.model_buffer) ggml_backend_buffer_free(weights.model_buffer);
    if (weights.ctx_data) ggml_free(weights.ctx_data);
    if (weights.gguf) gguf_free(weights.gguf);
    for (auto backend : backends) ggml_backend_free(backend);
}

void component::load(const std::string & path, int threads, int verbosity) {
    backend_loader backend;
    backend_scheduler_config config;
    config.max_nodes = 8192;
    if (!backend.load(cpu, backends, scheduler, buft, true, config, verbosity))
        throw std::runtime_error(backend.error());
    mode_ = backend.mode();
    optimization_policy_ = make_optimization_policy(mode_);
    scheduler_verbosity = verbosity;
    set_backend_threads(backends, threads);
    // CUDA leaves only small input-normalization fragments on the CPU.
    // Avoid an OpenMP team for these fragments while the GPU waits.
    if (is_cuda()) ggml_backend_cpu_set_n_threads(cpu, 1);
    if (!gguf_loader::load(path.c_str(), buft.model_buft, weights, verbosity))
        throw std::runtime_error(error());
}

bool component::is_cpu() const { return mode_ == backend_mode::cpu; }
bool component::is_cuda() const { return mode_ == backend_mode::cuda; }
ggml_tensor * component::tensor(const std::string & name) { return require_tensor(weights.ctx_data, name); }
int component::integer(const char * name) { return require_u32(weights.gguf, name); }
float component::scalar(const char * name) {
    const int key = gguf_find_key(weights.gguf, name);
    if (key < 0) throw std::runtime_error(std::string("missing metadata: ") + name);
    return gguf_get_val_f32(weights.gguf, key);
}
std::string component::text(const char * name) {
    const int key = gguf_find_key(weights.gguf, name);
    if (key < 0) throw std::runtime_error(std::string("missing metadata: ") + name);
    return gguf_get_val_str(weights.gguf, key);
}
std::vector<float> component::array(const std::string & name) {
    const int key = gguf_find_key(weights.gguf, name.c_str());
    if (key < 0) throw std::runtime_error("missing metadata: " + name);
    const size_t count = gguf_get_arr_n(weights.gguf, key);
    const void * data = gguf_get_arr_data(weights.gguf, key);
    std::vector<float> result(count);
    if (gguf_get_arr_type(weights.gguf, key) == GGUF_TYPE_BOOL) {
        const auto * values = static_cast<const bool *>(data);
        for (size_t i = 0; i < count; ++i) result[i] = values[i] ? 1.0f : 0.0f;
    } else {
        const auto * values = static_cast<const float *>(data);
        result.assign(values, values + count);
    }
    return result;
}

ggml_context * component::begin_graph() {
    ggml_backend_sched_reset(scheduler);
    if (ctx) ggml_free(ctx);
    constexpr size_t nodes = 8192;
    graph_memory.resize(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false));
    ctx = ggml_init({graph_memory.size(), graph_memory.data(), true});
    graph = ggml_new_graph_custom(ctx, nodes, false);
    return ctx;
}
void component::set_input(ggml_tensor * value) {
    ggml_set_input(value);
    ggml_backend_sched_set_tensor_backend(scheduler, value, backends.front());
}
void component::finish_graph(ggml_tensor * result) {
    output = result;
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) throw std::runtime_error("BitVLA graph allocation failed");
}
void component::swap_graph_state(graph_state & state) {
    std::swap(ctx, state.ctx);
    std::swap(input, state.input);
    std::swap(output, state.output);
    std::swap(graph_size, state.graph_size);
    std::swap(graph_action_start, state.action_start);
    std::swap(scheduler, state.scheduler);
    std::swap(graph, state.graph);
    graph_memory.swap(state.graph_memory);
}

void component::prepare_language_graph(int tokens, int action_start) {
    if (!is_cuda()) {
        if (graph_size != tokens) build_language(*this, tokens, action_start);
        return;
    }
    if (graph_size == tokens && graph_action_start == action_start) return;

    // Prediction is serialized. Quiesce shared backend streams before swapping or freeing states.
    synchronize_backends();
    const auto found = std::find_if(language_graphs.begin(), language_graphs.end(),
        [&](const auto & state) { return state->graph_size == tokens && state->action_start == action_start; });
    if (found != language_graphs.end()) {
        swap_graph_state(**found);
        // The previous active state is now the most recently used inactive state.
        language_graphs.splice(language_graphs.begin(), language_graphs, found);
        return;
    }

    // Seven inactive states plus the active state is the hard limit, including construction.
    if (language_graphs.size() == 7) language_graphs.pop_back();
    auto next = std::make_unique<graph_state>();
    backend_loader backend;
    backend_scheduler_config config;
    config.max_nodes = 8192;
    if (!backend.init_scheduler(backends, next->scheduler, buft, config, scheduler_verbosity))
        throw std::runtime_error(backend.error());
    language_graphs.push_front(std::move(next));
    auto & previous = *language_graphs.front();
    swap_graph_state(previous);
    try {
        build_language(*this, tokens, action_start);
        graph_action_start = action_start;
    } catch (...) {
        synchronize_backends();
        swap_graph_state(previous);
        language_graphs.pop_front();
        throw;
    }
    // The original load-time scheduler has no graph on the first successful construction.
    if (previous.graph_size == 0) language_graphs.pop_front();
}

void component::compute() {
    if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("BitVLA graph execution failed");
}
std::vector<float> component::read(ggml_tensor * value) {
    std::vector<float> result(ggml_nelements(value));
    ggml_backend_tensor_get(value, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// Original CPU F32 normalization, computed once at graph build.
static std::vector<float> vision_rgb_lookup(component & model) {
    const auto mean = model.array("clip.vision.image_mean");
    const auto stdv = model.array("clip.vision.image_std");
    std::vector<float> lookup(768);
    for (int c = 0; c < 3; ++c) for (int level = 0; level < 256; ++level) {
        const float rescaled = static_cast<float>(level / 255.0);
        const float value = (rescaled - mean[c]) / stdv[c];
        lookup[c*256+level] = value;
    }
    return lookup;
}
void component::create_vision_input(int size) {
    auto & model = *this;
    auto * ctx = this->ctx;
    ggml_tensor * lookup = nullptr;
    if (model.policy().vision_rgb_lookup_in_graph) {
        auto * rgb = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, int64_t(size)*size*3);
        lookup = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);
        model.set_input(rgb);
        model.set_input(lookup);
        // The table is immutable across frames: prevent allocator parent reuse.
        ggml_set_output(lookup);
        model.input = vision_rgb_u8_f32(ctx, rgb, lookup, size);
    } else {
        model.input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, size, size, 3, 1);
        ggml_set_input(model.input);
    }
    vision_lookup_ = lookup;
}

void component::finish_vision_graph(ggml_tensor * result) {
    auto & model = *this;
    auto * lookup = vision_lookup_;
    finish_graph(result);
    if (lookup) {
        const auto table = vision_rgb_lookup(model);
        ggml_backend_tensor_set(lookup, table.data(), 0, table.size()*sizeof(float));
    }
}

void component::compute_vision(const uint8_t * rgb, int stride) {
    auto & model = *this;
    const int size = model.integer("clip.vision.image_size");
    if (model.policy().vision_rgb_lookup_in_graph) {
        auto * bytes = model.input->src[0];
        GGML_ASSERT(model.input->op == GGML_OP_CUSTOM && bytes && bytes->type == GGML_TYPE_I8);
        const size_t row_bytes = size_t(size)*3;
        GGML_ASSERT(rgb && stride >= int(row_bytes));
        if (stride == int(row_bytes)) {
            ggml_backend_tensor_set(bytes, rgb, 0, row_bytes*size);
        } else {
            std::vector<uint8_t> packed(row_bytes*size);
            for (int y=0; y<size; ++y) std::memcpy(packed.data()+size_t(y)*row_bytes, rgb+size_t(y)*stride, row_bytes);
            ggml_backend_tensor_set(bytes, packed.data(), 0, packed.size());
        }
        model.compute();
        return;
    }
    const auto mean = model.array("clip.vision.image_mean");
    const auto stdv = model.array("clip.vision.image_std");
    // RGB has only 256 levels. Compute the same normalization once per level.
    float lookup[3][256];
    for (int c = 0; c < 3; ++c) {
        for (int level = 0; level < 256; ++level) {
            const float rescaled = static_cast<float>(level / 255.0);
            const float value = (rescaled - mean[c]) / stdv[c];
            lookup[c][level] = value;
        }
    }
    std::vector<float> pixels(size * size * 3);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                pixels[(c * size + y) * size + x] = lookup[c][rgb[y * stride + x * 3 + c]];
            }
        }
    }
    ggml_backend_tensor_set(model.input, pixels.data(), 0, pixels.size() * sizeof(float));
    model.compute();
}

void component::register_language_input() {
    if (is_cuda()) {
        set_input(input);
        ggml_set_output(input);
    } else ggml_set_input(input);
}

void component::register_rope_inputs(ggml_tensor * cosine, ggml_tensor * sine) {
    set_input(cosine);
    set_input(sine);
    ggml_set_output(cosine);
    ggml_set_output(sine);
}

void component::finish_language_graph(ggml_tensor * result, ggml_tensor * cosine,
                                      ggml_tensor * sine, int tokens) {
    finish_graph(result);
    initialize_rope_cache(cosine, sine);
    graph_size = tokens;
}

static ggml_backend_buffer_t input_buffer(const ggml_tensor * tensor) {
    while (!tensor->buffer && tensor->view_src) tensor = tensor->view_src;
    GGML_ASSERT(tensor->buffer);
    return tensor->buffer;
}

void copy_vision_to_language_input(ggml_tensor * destination, const ggml_tensor * source,
                                   const std::vector<size_t> & image_positions, size_t first_image_row) {
    GGML_ASSERT(destination->type == GGML_TYPE_F32 && source->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(destination) && ggml_is_contiguous(source));
    GGML_ASSERT(destination->ne[0] == source->ne[0] && destination->ne[2] == 1 && destination->ne[3] == 1);
    GGML_ASSERT(source->ne[2] == 1 && source->ne[3] == 1);
    const size_t rows = source->ne[1], width = source->ne[0];
    GGML_ASSERT(first_image_row + rows <= image_positions.size());
    for (size_t row = 0; row < rows;) {
        size_t count = 1;
        while (row + count < rows && image_positions[first_image_row + row + count] == image_positions[first_image_row + row] + count) ++count;
        const size_t target = image_positions[first_image_row + row];
        GGML_ASSERT(target + count <= size_t(destination->ne[1]));
        ggml_tensor src = *source, dst = *destination;
        src.ne[1] = dst.ne[1] = count;
        src.ne[2] = src.ne[3] = dst.ne[2] = dst.ne[3] = 1;
        src.nb[0] = dst.nb[0] = sizeof(float);
        src.nb[1] = dst.nb[1] = width * sizeof(float);
        src.nb[2] = src.nb[3] = dst.nb[2] = dst.nb[3] = count * width * sizeof(float);
        src.data = static_cast<char *>(source->data) + row * source->nb[1];
        dst.data = static_cast<char *>(destination->data) + target * destination->nb[1];
        src.buffer = input_buffer(source); dst.buffer = input_buffer(destination);
        src.view_src = dst.view_src = nullptr; src.view_offs = dst.view_offs = 0;
        // Pinned CUDA buffer cpy_tensor synchronizes its D2D copy before return.
        // Therefore the next image may safely reuse the vision output allocation.
        ggml_backend_tensor_copy(&src, &dst);
        row += count;
    }
}

void set_language_nonimage_input(ggml_tensor * destination, const float * embeddings, size_t count,
                                  const std::vector<size_t> & image_positions) {
    GGML_ASSERT(destination->type == GGML_TYPE_F32 && ggml_is_contiguous(destination));
    GGML_ASSERT(destination->ne[2] == 1 && destination->ne[3] == 1 && count == size_t(ggml_nelements(destination)));
    const size_t width = destination->ne[0], rows = destination->ne[1];
    size_t cursor = 0;
    for (size_t row : image_positions) {
        GGML_ASSERT(row >= cursor && row < rows);
        if (row > cursor) ggml_backend_tensor_set(destination, embeddings + cursor * width,
                                                 cursor * width * sizeof(float), (row - cursor) * width * sizeof(float));
        cursor = row + 1;
    }
    if (cursor < rows) ggml_backend_tensor_set(destination, embeddings + cursor * width,
                                              cursor * width * sizeof(float), (rows - cursor) * width * sizeof(float));
}

// Image rows are freshly copied by the vision path for every predict.
// Static rows are initialized anew after every prompt/graph switch. No result cache.
void set_language_predict_input(ggml_tensor * destination, const float * embeddings, size_t count,
                               const std::vector<size_t> & image_positions, size_t state_row, bool initialize_static) {
    GGML_ASSERT(destination->type == GGML_TYPE_F32 && ggml_is_contiguous(destination));
    GGML_ASSERT(count == size_t(ggml_nelements(destination)) && state_row < size_t(destination->ne[1]));
    if (initialize_static) {
        for (size_t row : image_positions) GGML_ASSERT(row != state_row);
        set_language_nonimage_input(destination, embeddings, count, image_positions);
    } else {
        const size_t width = destination->ne[0];
        ggml_backend_tensor_set(destination, embeddings + state_row * width,
                                state_row * width * sizeof(float), width * sizeof(float));
    }
}

void compute_language_with_host_state(component & model, const float * embeddings, size_t count,
                                        const std::vector<size_t> & image_positions, size_t state_row,
                                        const float * state_features, size_t state_count, bool initialize_static) {
    auto * destination = model.input;
    GGML_ASSERT(destination->type == GGML_TYPE_F32 && ggml_is_contiguous(destination));
    GGML_ASSERT(destination->ne[0] == 2560 && destination->ne[2] == 1 && destination->ne[3] == 1);
    GGML_ASSERT(count == size_t(ggml_nelements(destination)) && state_row < size_t(destination->ne[1]));
    GGML_ASSERT(state_features && state_count == 2560);
    for (size_t row : image_positions) GGML_ASSERT(row != state_row);
    if (initialize_static) set_language_nonimage_input(destination, embeddings, count, image_positions);
    // The original CPU-readable proprio result remains fresh for this predict.
    // Replace its placeholder before any graph computation; the prompt is read-only.
    ggml_backend_tensor_set(destination, state_features, state_row * 2560 * sizeof(float), 2560 * sizeof(float));
    model.compute();
}

void compute_language_with_static_input(component & model, const float * embeddings, size_t count,
                                        const std::vector<size_t> & image_positions, size_t state_row, bool initialize_static) {
    set_language_predict_input(model.input, embeddings, count, image_positions, state_row, initialize_static);
    model.compute();
}

void compute_language_with_device_images(component & model, const float * embeddings,
                                         size_t count, const std::vector<size_t> & image_positions) {
    set_language_nonimage_input(model.input, embeddings, count, image_positions);
    model.compute();
}

std::vector<float> run_language_with_device_images(component & model, const float * embeddings,
                                                    size_t count, const std::vector<size_t> & image_positions) {
    compute_language_with_device_images(model, embeddings, count, image_positions);
    return model.read(model.output);
}

std::vector<float> run_language(component & model, const float * embeddings, std::size_t count) {
    GGML_ASSERT(count == static_cast<std::size_t>(ggml_nelements(model.input)));
    ggml_backend_tensor_set(model.input, embeddings, 0, count * sizeof(float));
    model.compute();
    return model.read(model.output);
}
std::vector<float> run_language(component & model, const std::vector<float> & embeddings) {
    return run_language(model,embeddings.data(),embeddings.size());
}

bool uses_split_host_state_input(const component & vision, const component & language, const component & action) {
    return vision.is_cuda() && language.is_cuda() && action.is_cuda();
}

void encode_vision_for_language(component & vision, component & language, const uint8_t * rgb, int stride,
        float * embeddings, const std::vector<size_t> & image_positions, size_t & image_index) {
                if (vision.is_cuda()) {
                    bitvla::encode_vision_device(vision, rgb, stride);
                    bitvla::copy_vision_to_language_input(language.input, vision.output, image_positions, image_index);
                    image_index += 256;
                    return;
                }
                const auto features = bitvla::encode_vision(vision, rgb, stride);
                for (size_t row = 0; row < 256;) {
                    size_t count = 1;
                    while (row + count < 256 &&
                           image_positions[image_index + count] == image_positions[image_index] + count) ++count;
                    std::copy_n(features.data() + row * 2560, count * 2560,
                                embeddings + image_positions[image_index] * 2560);
                    image_index += count;
                    row += count;
                }
}

LanguageOutput component::compute_for_action(component & vision, component & action, bool split_host_state_input,
        const std::vector<float> & prompt_embeddings, const float * embeddings,
        const std::vector<size_t> & image_positions, size_t proprio_position,
        const std::vector<float> & state_features_host) {
    auto & language = *this;
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // CPU103: keep the original full input upload; avoid a hidden-result vector.
    if(vision.is_cpu() && language.is_cpu() && action.is_cpu() &&
       !split_host_state_input && embeddings && input && output && action.input &&
       input->type==GGML_TYPE_F32 && input->ne[0]==2560 && input->ne[2]==1 && input->ne[3]==1 &&
       ggml_is_contiguous(input) && size_t(ggml_nelements(input))==prompt_embeddings.size() &&
       output->type==GGML_TYPE_F32 && output->ne[0]==2560 && output->ne[1]==56 && output->ne[2]==1 && output->ne[3]==1 &&
       action.input->type==GGML_TYPE_F32 && action.input->ne[0]==17920 && action.input->ne[1]==8 && action.input->ne[2]==1 && action.input->ne[3]==1 &&
       ggml_is_contiguous(output) && ggml_is_contiguous(action.input) && (output->flags&GGML_TENSOR_FLAG_OUTPUT) &&
       output->data && action.input->data && output->data!=action.input->data &&
       ggml_backend_buffer_is_host(input_buffer(output)) && ggml_backend_buffer_is_host(input_buffer(action.input))) {
        ggml_backend_tensor_set(input,embeddings,0,prompt_embeddings.size()*sizeof(float));
        compute();
        // finish_graph already retains output; decode consumes it before another predict.
        return {std::vector<float>(),output};
    }
#endif
            const bool direct_host_action = vision.is_cuda() && language.is_cuda() && action.is_cuda();
            std::vector<float> hidden;
            if (direct_host_action) {
                if (split_host_state_input) {
                    bitvla::compute_language_with_host_state(language, prompt_embeddings.data(), prompt_embeddings.size(), image_positions,
                        proprio_position, state_features_host.data(), state_features_host.size(), !static_input_ready_);
                } else {
                    bitvla::compute_language_with_static_input(language, embeddings, prompt_embeddings.size(), image_positions, proprio_position, !static_input_ready_);
                }
                static_input_ready_ = true;
            } else {
                hidden = vision.is_cuda()
                    ? bitvla::run_language_with_device_images(language, embeddings, prompt_embeddings.size(), image_positions)
                    : bitvla::run_language(language, embeddings, prompt_embeddings.size());
            }

    return {std::move(hidden), direct_host_action ? output : nullptr};
}

std::vector<float> decode_language_output(component & action, const LanguageOutput & hidden) {
    return hidden.tensor ? decode_action_direct_host_input(action, hidden.tensor) : decode_action(action, hidden.values);
}

} // namespace robotcpp::bitvla
