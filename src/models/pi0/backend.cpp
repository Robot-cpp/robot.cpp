#include "models/pi0/backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace robotcpp::pi0 {

Pi0DeviceTensor::~Pi0DeviceTensor() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

namespace {

Pi0BackendConfig component_backend_config(const Pi0BackendConfig & base, const Pi0ComponentConfig & component) {
    Pi0BackendConfig out = base;
    if (component.runtime.backend == "cpu") {
        out.backend = PI0_BACKEND_CPU;
    } else if (component.runtime.backend == "cuda" || component.runtime.backend == "accel") {
        out.backend = PI0_BACKEND_ACCEL;
    } else if (component.runtime.backend != "inherit") {
        throw std::invalid_argument("unsupported pi0 component backend: " + component.runtime.backend);
    }
    if (component.runtime.n_threads > 0) {
        out.n_threads = component.runtime.n_threads;
    }
    return out;
}

bool use_accel_backend(const Pi0BackendConfig & config) {
    return config.backend == PI0_BACKEND_ACCEL;
}

const char * pi0_backend_mode_name(Pi0BackendMode mode) {
    switch (mode) {
        case PI0_BACKEND_CPU:   return "cpu";
        case PI0_BACKEND_ACCEL: return "accel";
    }
    return "unknown";
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

} // namespace

bool Pi0DeviceTensorKey::operator<(const Pi0DeviceTensorKey & other) const {
    if (tensor != other.tensor) {
        return tensor < other.tensor;
    }
    if (type != other.type) {
        return type < other.type;
    }
    if (n_dims != other.n_dims) {
        return n_dims < other.n_dims;
    }
    if (ne0 != other.ne0) {
        return ne0 < other.ne0;
    }
    if (ne1 != other.ne1) {
        return ne1 < other.ne1;
    }
    if (ne2 != other.ne2) {
        return ne2 < other.ne2;
    }
    return ne3 < other.ne3;
}

Pi0ComponentBackend::~Pi0ComponentBackend() {
    reset();
}

Pi0ComponentBackend::Pi0ComponentBackend(Pi0ComponentBackend && other) noexcept
    : backend_cpu(other.backend_cpu),
      sched(other.sched),
      backends(std::move(other.backends)),
      buft_policy(other.buft_policy),
      n_threads(other.n_threads),
      device_tensors(std::move(other.device_tensors)) {
    other.backend_cpu = nullptr;
    other.sched = nullptr;
    other.backends.clear();
    other.buft_policy = {};
    other.n_threads = 0;
}

Pi0ComponentBackend & Pi0ComponentBackend::operator=(Pi0ComponentBackend && other) noexcept {
    if (this != &other) {
        reset();
        backend_cpu = other.backend_cpu;
        sched = other.sched;
        backends = std::move(other.backends);
        buft_policy = other.buft_policy;
        n_threads = other.n_threads;
        device_tensors = std::move(other.device_tensors);
        other.backend_cpu = nullptr;
        other.sched = nullptr;
        other.backends.clear();
        other.buft_policy = {};
        other.n_threads = 0;
    }
    return *this;
}

void Pi0ComponentBackend::reset() {
    device_tensors.clear();
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

void pi0_init_backend(
    Pi0ComponentBackend & runtime,
    const Pi0BackendConfig & base,
    const Pi0ComponentConfig & component,
    const char * label,
    int verbosity) {
    runtime.reset();
    const Pi0BackendConfig resolved = component_backend_config(base, component);
    runtime.n_threads = resolved.n_threads;

    backend_scheduler_config scheduler_config;
    scheduler_config.max_nodes = GGML_DEFAULT_GRAPH_SIZE;
    scheduler_config.parallel = false;
    scheduler_config.op_offload = use_accel_backend(resolved);

    backend_loader loader;
    if (!loader.load(
            runtime.backend_cpu,
            runtime.backends,
            runtime.sched,
            runtime.buft_policy,
            use_accel_backend(resolved),
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
            pi0_backend_mode_name(resolved.backend),
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

ggml_tensor * pi0_materialize_device_f32_3d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    const ggml_tensor * source,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    if (runtime.sched == nullptr) {
        throw std::runtime_error("pi0 backend scheduler is not initialized");
    }
    Pi0DeviceTensorKey tensor_key{key, GGML_TYPE_F32, 3, ne0, ne1, ne2, 1};
    auto it = runtime.device_tensors.find(tensor_key);
    if (it == runtime.device_tensors.end()) {
        auto cached = std::make_unique<Pi0DeviceTensor>();
        ggml_init_params params{};
        params.mem_size = 2 * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        cached->ctx = ggml_init(params);
        if (cached->ctx == nullptr) {
            throw std::runtime_error("failed to initialize pi0 cached tensor context");
        }
        cached->tensor = ggml_new_tensor_3d(cached->ctx, GGML_TYPE_F32, ne0, ne1, ne2);
        if (cached->tensor == nullptr) {
            throw std::runtime_error("failed to allocate pi0 cached tensor metadata");
        }
        cached->buffer = ggml_backend_alloc_ctx_tensors_from_buft(cached->ctx, runtime.buft_policy.runtime_buft);
        if (cached->buffer == nullptr) {
            throw std::runtime_error("failed to allocate pi0 cached backend buffer");
        }
        it = runtime.device_tensors.emplace(tensor_key, std::move(cached)).first;
    }
    ggml_backend_tensor_copy(source, it->second->tensor);
    return it->second->tensor;
}

namespace {

ggml_tensor * cached_device_f32_2d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    int64_t ne0,
    int64_t ne1) {
    if (runtime.sched == nullptr) {
        throw std::runtime_error("pi0 backend scheduler is not initialized");
    }
    Pi0DeviceTensorKey tensor_key{key, GGML_TYPE_F32, 2, ne0, ne1, 1, 1};
    auto it = runtime.device_tensors.find(tensor_key);
    if (it == runtime.device_tensors.end()) {
        auto cached = std::make_unique<Pi0DeviceTensor>();
        ggml_init_params params{};
        params.mem_size = 2 * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        cached->ctx = ggml_init(params);
        if (cached->ctx == nullptr) {
            throw std::runtime_error("failed to initialize pi0 cached tensor context");
        }
        cached->tensor = ggml_new_tensor_2d(cached->ctx, GGML_TYPE_F32, ne0, ne1);
        if (cached->tensor == nullptr) {
            throw std::runtime_error("failed to allocate pi0 cached tensor metadata");
        }
        cached->buffer = ggml_backend_alloc_ctx_tensors_from_buft(cached->ctx, runtime.buft_policy.runtime_buft);
        if (cached->buffer == nullptr) {
            throw std::runtime_error("failed to allocate pi0 cached backend buffer");
        }
        it = runtime.device_tensors.emplace(tensor_key, std::move(cached)).first;
    }
    return it->second->tensor;
}

} // namespace

ggml_tensor * pi0_materialize_device_f32_2d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    const ggml_tensor * source,
    int64_t ne0,
    int64_t ne1) {
    ggml_tensor * cached = cached_device_f32_2d(runtime, key, ne0, ne1);
    ggml_backend_tensor_copy(source, cached);
    return cached;
}

ggml_tensor * pi0_upload_device_f32_2d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    const float * source,
    int64_t ne0,
    int64_t ne1) {
    if (source == nullptr && ne0 * ne1 > 0) {
        throw std::invalid_argument("pi0 device upload requires source data");
    }
    ggml_tensor * cached = cached_device_f32_2d(runtime, key, ne0, ne1);
    ggml_backend_tensor_set(cached, source, 0, static_cast<size_t>(ne0) * static_cast<size_t>(ne1) * sizeof(float));
    return cached;
}

} // namespace robotcpp::pi0
