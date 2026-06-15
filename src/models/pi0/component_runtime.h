#pragma once

#include "models/ggml_backend.h"
#include "models/pi0/types.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace robotcpp::pi0 {

struct Pi0PersistentAllocation {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    Pi0PersistentAllocation() = default;
    Pi0PersistentAllocation(ggml_context * context, ggml_backend_buffer_t backend_buffer);
    ~Pi0PersistentAllocation();

    Pi0PersistentAllocation(const Pi0PersistentAllocation &) = delete;
    Pi0PersistentAllocation & operator=(const Pi0PersistentAllocation &) = delete;

    Pi0PersistentAllocation(Pi0PersistentAllocation && other) noexcept;
    Pi0PersistentAllocation & operator=(Pi0PersistentAllocation && other) noexcept;

    void reset();
};

struct Pi0ComponentRuntime {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<ggml_backend_t> backends;
    backend_buft_policy buft_policy;
    int n_threads = 0;
    mutable std::vector<Pi0PersistentAllocation> persistent_allocations;

    Pi0ComponentRuntime() = default;
    ~Pi0ComponentRuntime();

    Pi0ComponentRuntime(const Pi0ComponentRuntime &) = delete;
    Pi0ComponentRuntime & operator=(const Pi0ComponentRuntime &) = delete;

    Pi0ComponentRuntime(Pi0ComponentRuntime && other) noexcept;
    Pi0ComponentRuntime & operator=(Pi0ComponentRuntime && other) noexcept;

    void reset();
};

class Pi0GraphContext {
public:
    explicit Pi0GraphContext(ggml_init_params params);
    ~Pi0GraphContext();

    Pi0GraphContext(const Pi0GraphContext &) = delete;
    Pi0GraphContext & operator=(const Pi0GraphContext &) = delete;

    operator ggml_context *() const;

private:
    ggml_context * ctx_ = nullptr;
};

size_t pi0_graph_context_size(size_t tensor_bytes);

void pi0_init_component_runtime(
    Pi0ComponentRuntime & runtime,
    const Pi0BackendConfig & base,
    const Pi0ComponentConfig & component,
    const char * label,
    int verbosity);

ggml_init_params pi0_graph_init_params(size_t mem_size);

ggml_tensor * pi0_persist_backend_f32(
    const Pi0ComponentRuntime & runtime,
    ggml_tensor ** slot,
    const ggml_tensor * source,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2);

ggml_tensor * pi0_persist_host_f32_2d(
    const Pi0ComponentRuntime & runtime,
    ggml_tensor ** slot,
    const float * source,
    int64_t ne0,
    int64_t ne1);

} // namespace robotcpp::pi0
