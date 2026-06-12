#pragma once

#include "core/types.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vlacpp {

struct GgmlRunnerResources;

struct GgmlInput {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t size = 0;
};

struct GgmlBackendCapabilities {
    bool device_weight_store = false;
    ggml_type default_2d_weight_type = GGML_TYPE_F32;
    bool stable_graph_workspace = false;
};

struct GgmlBuftPolicy {
    ggml_backend_buffer_type_t model_buft = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    ggml_backend_buffer_type_t backend_buft = nullptr;
};

void require_ggml_weight_2d(const Tensor & tensor, int64_t ne0, int64_t ne1, const char * name);
void require_ggml_vector_1d(const Tensor & tensor, int64_t ne0, const char * name);
size_t ggml_graph_context_size(size_t tensor_bytes);

class GgmlContext {
public:
    explicit GgmlContext(ggml_init_params params);
    ~GgmlContext();

    GgmlContext(const GgmlContext &) = delete;
    GgmlContext & operator=(const GgmlContext &) = delete;

    ggml_context * get() const;
    operator ggml_context *() const;

private:
    ggml_context * ctx_ = nullptr;
};

class GgmlRunner {
public:
    explicit GgmlRunner(const BackendConfig & backend);
    ~GgmlRunner();

    GgmlRunner(const GgmlRunner &) = delete;
    GgmlRunner & operator=(const GgmlRunner &) = delete;

    bool uses_backend() const;
    const GgmlBackendCapabilities & capabilities() const;
    const GgmlBuftPolicy & buft_policy() const;
    ggml_init_params init_params(size_t mem_size, const void * stable_id = nullptr, uint64_t stable_variant = 0) const;
    void materialize_weights(const std::vector<Tensor *> & tensors) const;
    void materialize_weights(const std::vector<Tensor *> & tensors, ggml_type default_2d_weight_type) const;
    ggml_tensor * stored_weight(const Tensor & tensor) const;
    ggml_tensor * materialize_device_f32_3d_from_backend(
        ggml_context * ctx,
        const void * key,
        const ggml_tensor * source,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) const;

    void set_input(
        std::vector<GgmlInput> & inputs,
        ggml_tensor * tensor,
        const void * data,
        size_t size) const;

    void compute(
        ggml_context * ctx,
        ggml_cgraph * graph,
        const std::vector<GgmlInput> & inputs,
        const char * error_message) const;

    void get_output(const ggml_tensor * tensor, void * data, size_t size) const;

private:
    BackendConfig backend_config_;
    GgmlBackendCapabilities capabilities_;
    GgmlBuftPolicy buft_policy_;
    std::unique_ptr<GgmlRunnerResources> resources_;
    ggml_backend_t gpu_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    mutable std::string profile_label_;
};

} // namespace vlacpp
