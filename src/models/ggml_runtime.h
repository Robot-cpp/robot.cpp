#pragma once

#include "core/types.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vlacpp {

struct GgmlInput {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t size = 0;
};

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
    ggml_init_params init_params(size_t mem_size, const void * stable_id = nullptr, uint64_t stable_variant = 0) const;
    ggml_tensor * new_weight_1d(ggml_context * ctx, const Tensor & tensor) const;
    ggml_tensor * new_weight_1d_plus_one(ggml_context * ctx, const Tensor & tensor) const;
    ggml_tensor * new_weight_2d(ggml_context * ctx, const Tensor & tensor) const;
    ggml_tensor * new_cached_f32_3d(
        ggml_context * ctx,
        const void * key,
        uint64_t generation,
        const float * data,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) const;
    ggml_tensor * new_cached_f32_3d_from_backend(
        ggml_context * ctx,
        const void * key,
        uint64_t generation,
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
    const BackendConfig & backend_config_;
    ggml_backend_t gpu_backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    mutable const void * stable_id_ = nullptr;
    mutable uint64_t stable_variant_ = 0;
    mutable std::string profile_label_;
};

} // namespace vlacpp
