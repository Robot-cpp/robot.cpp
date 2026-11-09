#pragma once

#include "models/ggml_backend.h"
#include "models/gguf_loader.h"
#include "optimization_policy.h"

#include <list>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::bitvla {

struct LanguageOutput {
    std::vector<float> values;
    const ggml_tensor * tensor = nullptr;
};

// One weight file and reusable graph per model component.
class component : private gguf_loader {
  public:
    component() = default;
    ~component();
    component(const component &)             = delete;
    component & operator=(const component &) = delete;

    void load(const std::string & path, int threads, int verbosity);
    bool is_cpu() const;
    bool is_cuda() const;
    const OptimizationPolicy & policy() const { return optimization_policy_; }
    ggml_tensor * tensor(const std::string & name);
    int integer(const char * name);
    float scalar(const char * name);
    std::string text(const char * name);
    std::vector<float> array(const std::string & name);
    ggml_context * begin_graph();
    void set_input(ggml_tensor * value);
    void finish_graph(ggml_tensor * result);
    void prepare_language_graph(int tokens, int action_start);
    void compute();
    void create_vision_input(int size);
    void finish_vision_graph(ggml_tensor * result);
    void compute_vision(const uint8_t * rgb, int stride);
    void register_language_input();
    void register_rope_inputs(ggml_tensor * cosine, ggml_tensor * sine);
    void finish_language_graph(ggml_tensor * result, ggml_tensor * cosine, ggml_tensor * sine, int tokens);
    void invalidate_static_input() { static_input_ready_ = false; }
    LanguageOutput compute_for_action(component & vision, component & action, bool split_host_state_input,
                                      const std::vector<float> & prompt_embeddings, const float * embeddings,
                                      const std::vector<size_t> & image_positions, size_t proprio_position,
                                      const std::vector<float> & state_features_host);
    std::vector<float> read(ggml_tensor * tensor);

    ggml_context * ctx   = nullptr;
    ggml_tensor * input  = nullptr;
    ggml_tensor * output = nullptr;
    int graph_size       = 0;

  private:
    // Inactive language states retain their own scheduler and activation allocations.
    struct graph_state {
        graph_state() = default;
        ~graph_state();
        graph_state(const graph_state &)             = delete;
        graph_state & operator=(const graph_state &) = delete;
        ggml_context * ctx                           = nullptr;
        ggml_tensor * input                          = nullptr;
        ggml_tensor * output                         = nullptr;
        int graph_size                               = 0;
        int action_start                             = -1;
        ggml_backend_sched_t scheduler               = nullptr;
        ggml_cgraph * graph                          = nullptr;
        std::vector<uint8_t> graph_memory;
    };
    void swap_graph_state(graph_state & state);
    void synchronize_backends();
    bool parse_metadata(gguf_context *) override { return true; }
    bool bind_tensors(ggml_context *) override { return true; }
    gguf_load_result weights;
    ggml_backend_t cpu = nullptr;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t scheduler = nullptr;
    backend_buft_policy buft;
    ggml_cgraph * graph = nullptr;
    std::vector<uint8_t> graph_memory;
    std::list<std::unique_ptr<graph_state>> language_graphs;
    int graph_action_start  = -1;
    int scheduler_verbosity = 0;
    backend_mode mode_      = backend_mode::cpu;
    OptimizationPolicy optimization_policy_;
    ggml_tensor * vision_lookup_ = nullptr;
    bool static_input_ready_     = false;
};

void build_vision(component & model);
void encode_vision_device(component & model, const uint8_t * rgb, int stride);
std::vector<float> encode_vision(component & model, const uint8_t * rgb, int stride);
void build_proprio(component & model);
std::vector<float> encode_proprio(component & model, const std::vector<float> & state);
void build_action(component & model);
std::vector<float> decode_action(component & model, const std::vector<float> & hidden);
void build_language(component & model, int tokens, int action_start);
std::vector<float> run_language(component & model, const std::vector<float> & embeddings);
std::vector<float> run_language(component & model, const float * embeddings, std::size_t count);
void copy_vision_to_language_input(ggml_tensor * destination, const ggml_tensor * source,
                                   const std::vector<size_t> & image_positions, size_t first_image_row);
void set_language_nonimage_input(ggml_tensor * destination, const float * embeddings, size_t count,
                                 const std::vector<size_t> & image_positions);
std::vector<float> run_language_with_device_images(component & model, const float * embeddings, size_t count,
                                                   const std::vector<size_t> & image_positions);

} // namespace robotcpp::bitvla

namespace robotcpp::bitvla {
void compute_language_with_device_images(component & model, const float * embeddings, size_t count,
                                         const std::vector<size_t> & image_positions);
void copy_language_to_action_host_input(ggml_tensor * destination, const ggml_tensor * source);
std::vector<float> decode_action_direct_host_input(component & model, const ggml_tensor * hidden);
} // namespace robotcpp::bitvla

namespace robotcpp::bitvla {
void set_language_predict_input(ggml_tensor * destination, const float * embeddings, size_t count,
                                const std::vector<size_t> & image_positions, size_t state_row, bool initialize_static);
void compute_language_with_static_input(component & model, const float * embeddings, size_t count,
                                        const std::vector<size_t> & image_positions, size_t state_row,
                                        bool initialize_static);
} // namespace robotcpp::bitvla

namespace robotcpp::bitvla {
void compute_language_with_host_state(component & model, const float * embeddings, size_t count,
                                      const std::vector<size_t> & image_positions, size_t state_row,
                                      const float * state_features, size_t state_count, bool initialize_static);
}

namespace robotcpp::bitvla {
void encode_proprio_to_host(component & model, const std::vector<float> & state, std::vector<float> & result,
                            size_t offset = 0);
}

namespace robotcpp::bitvla {
bool uses_split_host_state_input(const component & vision, const component & language, const component & action);
void encode_vision_for_language(component & vision, component & language, const uint8_t * rgb, int stride,
                                float * embeddings, const std::vector<size_t> & image_positions, size_t & image_index);
std::vector<float> decode_language_output(component & action, const LanguageOutput & hidden);
} // namespace robotcpp::bitvla
