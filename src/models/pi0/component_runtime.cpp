#include "models/pi0/component_runtime.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace robotcpp::pi0 {

namespace {

Pi0BackendConfig component_backend_config(const Pi0BackendConfig & base, const Pi0ComponentConfig & component) {
    Pi0BackendConfig out = base;
    if (component.runtime.backend == "cpu") {
        out.use_accel = false;
    } else if (component.runtime.backend == "cuda" || component.runtime.backend == "accel") {
        out.use_accel = true;
    } else if (component.runtime.backend != "inherit") {
        throw std::invalid_argument("unsupported pi0 component backend: " + component.runtime.backend);
    }
    if (component.runtime.n_threads > 0) {
        out.n_threads = component.runtime.n_threads;
    }
    return out;
}

const char * pi0_requested_backend_name(bool use_accel) {
    return use_accel ? "accel" : "cpu";
}

const char * pi0_resolved_backend_name(backend_mode mode) {
    switch (mode) {
        case backend_mode::cpu:   return "cpu";
        case backend_mode::cuda:  return "cuda";
        case backend_mode::metal: return "metal";
    }
    return "unknown";
}

const char * pi0_buft_name(ggml_backend_buffer_type_t buft) {
    return buft ? ggml_backend_buft_name(buft) : "(null)";
}

bool persistent_tensor_matches(
    const ggml_tensor * tensor,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    return tensor != nullptr &&
        tensor->type == GGML_TYPE_F32 &&
        ggml_n_dims(tensor) == n_dims &&
        tensor->ne[0] == ne0 &&
        tensor->ne[1] == ne1 &&
        tensor->ne[2] == ne2;
}

ggml_tensor * persistent_f32(
    const Pi0ComponentRuntime & runtime,
    ggml_tensor ** slot,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    if (runtime.sched == nullptr) {
        throw std::runtime_error("pi0 component runtime scheduler is not initialized");
    }
    if (slot == nullptr) {
        throw std::invalid_argument("pi0 persistent tensor requires a slot");
    }
    if (persistent_tensor_matches(*slot, n_dims, ne0, ne1, ne2)) {
        return *slot;
    }

    const int64_t ne[] = {ne0, ne1, ne2, 1};
    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize pi0 persistent tensor context");
    }
    ggml_tensor * tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    if (tensor == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate pi0 persistent tensor metadata");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, runtime.buft_policy.runtime_buft);
    if (buffer == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate pi0 persistent runtime buffer");
    }

    Pi0PersistentAllocation allocation(ctx, buffer);
    runtime.persistent_allocations.push_back(std::move(allocation));
    *slot = tensor;
    return tensor;
}

} // namespace

Pi0PersistentAllocation::Pi0PersistentAllocation(ggml_context * context, ggml_backend_buffer_t backend_buffer)
    : ctx(context),
      buffer(backend_buffer) {
}

Pi0PersistentAllocation::~Pi0PersistentAllocation() {
    reset();
}

Pi0PersistentAllocation::Pi0PersistentAllocation(Pi0PersistentAllocation && other) noexcept
    : ctx(other.ctx),
      buffer(other.buffer) {
    other.ctx = nullptr;
    other.buffer = nullptr;
}

Pi0PersistentAllocation & Pi0PersistentAllocation::operator=(Pi0PersistentAllocation && other) noexcept {
    if (this != &other) {
        reset();
        ctx = other.ctx;
        buffer = other.buffer;
        other.ctx = nullptr;
        other.buffer = nullptr;
    }
    return *this;
}

void Pi0PersistentAllocation::reset() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

Pi0ComponentRuntime::~Pi0ComponentRuntime() {
    reset();
}

Pi0ComponentRuntime::Pi0ComponentRuntime(Pi0ComponentRuntime && other) noexcept
    : backend_cpu(other.backend_cpu),
      sched(other.sched),
      backends(std::move(other.backends)),
      buft_policy(other.buft_policy),
      n_threads(other.n_threads),
      persistent_allocations(std::move(other.persistent_allocations)) {
    other.backend_cpu = nullptr;
    other.sched = nullptr;
    other.backends.clear();
    other.persistent_allocations.clear();
    other.buft_policy = {};
    other.n_threads = 0;
}

Pi0ComponentRuntime & Pi0ComponentRuntime::operator=(Pi0ComponentRuntime && other) noexcept {
    if (this != &other) {
        reset();
        backend_cpu = other.backend_cpu;
        sched = other.sched;
        backends = std::move(other.backends);
        buft_policy = other.buft_policy;
        n_threads = other.n_threads;
        persistent_allocations = std::move(other.persistent_allocations);
        other.backend_cpu = nullptr;
        other.sched = nullptr;
        other.backends.clear();
        other.persistent_allocations.clear();
        other.buft_policy = {};
        other.n_threads = 0;
    }
    return *this;
}

void Pi0ComponentRuntime::reset() {
    persistent_allocations.clear();
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    for (ggml_backend_t backend : backends) {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
    backends.clear();
    backend_cpu = nullptr;
    buft_policy = {};
    n_threads = 0;
}

Pi0GraphContext::Pi0GraphContext(ggml_init_params params) {
    ctx_ = ggml_init(params);
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize pi0 ggml graph context");
    }
}

Pi0GraphContext::~Pi0GraphContext() {
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

Pi0GraphContext::operator ggml_context *() const {
    return ctx_;
}

size_t pi0_graph_context_size(size_t tensor_bytes) {
    return std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
}

void pi0_init_component_runtime(
    Pi0ComponentRuntime & runtime,
    const Pi0BackendConfig & base,
    const Pi0ComponentConfig & component,
    const char * label,
    int verbosity) {
    runtime.reset();
    const Pi0BackendConfig resolved = component_backend_config(base, component);
    const bool use_accel = resolved.use_accel;
    runtime.n_threads = resolved.n_threads;

    backend_scheduler_config scheduler_config;
    scheduler_config.max_nodes = GGML_DEFAULT_GRAPH_SIZE;
    scheduler_config.parallel = false;
    scheduler_config.op_offload = use_accel;

    backend_loader loader;
    if (!loader.load(
            runtime.backend_cpu,
            runtime.backends,
            runtime.sched,
            runtime.buft_policy,
            use_accel,
            scheduler_config,
            verbosity)) {
        throw std::runtime_error(
            std::string("failed to initialize pi0 ") +
            (label != nullptr ? label : "component") +
            " backend: " + loader.error());
    }

    if (verbosity >= 1) {
        std::fprintf(stderr,
            "%s: component=%s requested=%s metadata=%s resolved=%s model_buft=%s runtime_buft=%s backend_count=%zu\n",
            __func__,
            label != nullptr ? label : "component",
            pi0_requested_backend_name(use_accel),
            component.runtime.backend.c_str(),
            pi0_resolved_backend_name(loader.mode()),
            pi0_buft_name(runtime.buft_policy.model_buft),
            pi0_buft_name(runtime.buft_policy.runtime_buft),
            runtime.backends.size());
    }
}

ggml_init_params pi0_graph_init_params(size_t mem_size) {
    ggml_init_params params{};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    return params;
}

ggml_tensor * pi0_persist_backend_f32(
    const Pi0ComponentRuntime & runtime,
    ggml_tensor ** slot,
    const ggml_tensor * source,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    ggml_tensor * persistent = persistent_f32(runtime, slot, n_dims, ne0, ne1, ne2);
    ggml_backend_tensor_copy(source, persistent);
    return persistent;
}

ggml_tensor * pi0_persist_host_f32_2d(
    const Pi0ComponentRuntime & runtime,
    ggml_tensor ** slot,
    const float * source,
    int64_t ne0,
    int64_t ne1) {
    if (source == nullptr && ne0 * ne1 > 0) {
        throw std::invalid_argument("pi0 persistent tensor upload requires source data");
    }
    ggml_tensor * persistent = persistent_f32(runtime, slot, 2, ne0, ne1, 1);
    ggml_backend_tensor_set(persistent, source, 0, static_cast<size_t>(ne0) * static_cast<size_t>(ne1) * sizeof(float));
    return persistent;
}

} // namespace robotcpp::pi0
