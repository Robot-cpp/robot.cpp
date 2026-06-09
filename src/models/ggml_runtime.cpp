#include "models/ggml_runtime.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace vlacpp {

void require_ggml_weight_2d(const Tensor & tensor, int64_t ne0, int64_t ne1, const char * name) {
    if (tensor.shape.size() != 2 ||
        tensor.shape[0] != ne0 ||
        tensor.shape[1] != ne1) {
        throw std::invalid_argument(std::string(name) + " has incompatible ggml shape");
    }
}

void require_ggml_vector_1d(const Tensor & tensor, int64_t ne0, const char * name) {
    if (tensor.shape.size() != 1 ||
        tensor.shape[0] != ne0) {
        throw std::invalid_argument(std::string(name) + " has incompatible ggml shape");
    }
}

size_t ggml_graph_context_size(size_t tensor_bytes) {
    return std::max<size_t>(64 * 1024 * 1024, tensor_bytes * 4 + 1024 * 1024);
}

struct BackendState {
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ~BackendState() {
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
        }
        if (cpu != nullptr) {
            ggml_backend_free(cpu);
        }
        if (gpu != nullptr) {
            ggml_backend_free(gpu);
        }
    }
};

ggml_backend_buffer_type_t default_buft(ggml_backend_t backend) {
    return backend != nullptr ? ggml_backend_get_default_buffer_type(backend) : nullptr;
}

ggml_backend_buffer_type_t host_buft_for_backend(ggml_backend_t backend) {
    if (backend != nullptr) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft != nullptr) {
            return buft;
        }
    }
    return ggml_backend_cpu_buffer_type();
}

GgmlBuftPolicy make_buft_policy(ggml_backend_t model_backend, ggml_backend_t cpu_backend) {
    GgmlBuftPolicy policy;
    policy.backend_buft = default_buft(model_backend);
    policy.model_buft = policy.backend_buft ? policy.backend_buft : default_buft(cpu_backend);
    policy.host_buft = host_buft_for_backend(model_backend);
    if (policy.host_buft == nullptr) {
        policy.host_buft = default_buft(cpu_backend);
    }
    return policy;
}

struct AlignedBuffer {
    void * data = nullptr;
    size_t size = 0;

    ~AlignedBuffer() {
        std::free(data);
    }

    void ensure(size_t requested_size) {
        const size_t padded_size = GGML_PAD(requested_size, GGML_MEM_ALIGN);
        if (data != nullptr && size >= padded_size) {
            return;
        }
        std::free(data);
        data = nullptr;
        size = 0;
        void * buffer = nullptr;
        if (posix_memalign(&buffer, GGML_MEM_ALIGN, padded_size) != 0) {
            throw std::bad_alloc();
        }
        data = buffer;
        size = padded_size;
    }
};

struct GraphWorkspaceKey {
    const void * backend = nullptr;
    const void * id = nullptr;
    uint64_t variant = 0;
    size_t size = 0;

    bool operator<(const GraphWorkspaceKey & other) const {
        if (backend != other.backend) {
            return backend < other.backend;
        }
        if (id != other.id) {
            return id < other.id;
        }
        if (variant != other.variant) {
            return variant < other.variant;
        }
        return size < other.size;
    }
};

struct DeviceTensor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;

    ~DeviceTensor() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct DeviceTensorKey {
    const void * backend = nullptr;
    const void * tensor = nullptr;
    uint64_t generation = 0;
    ggml_type type = GGML_TYPE_F32;
    int n_dims = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;
    int64_t ne3 = 0;

    bool operator<(const DeviceTensorKey & other) const {
        if (backend != other.backend) {
            return backend < other.backend;
        }
        if (tensor != other.tensor) {
            return tensor < other.tensor;
        }
        if (generation != other.generation) {
            return generation < other.generation;
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
};

thread_local std::unique_ptr<BackendState> cuda_state;
thread_local std::unique_ptr<BackendState> cpu_state;

struct ProfileStats {
    int64_t calls = 0;
    double alloc_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double download_ms = 0.0;
    size_t upload_bytes = 0;
    size_t download_bytes = 0;
};

struct GgmlRunnerResources {
    std::map<DeviceTensorKey, std::unique_ptr<DeviceTensor>> device_tensors;
    std::map<GraphWorkspaceKey, AlignedBuffer> graph_workspaces;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    std::map<const Tensor *, ggml_tensor *> weights;

    ~GgmlRunnerResources() {
        if (weight_buffer != nullptr) {
            ggml_backend_buffer_free(weight_buffer);
        }
        if (weight_ctx != nullptr) {
            ggml_free(weight_ctx);
        }
    }
};

std::mutex profile_mutex;
std::map<std::string, ProfileStats> profile_stats;
bool profile_registered = false;

bool profile_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("VLACPP_PROFILE");
        return value != nullptr && value[0] != '\0' && std::string(value) != "0";
    }();
    return enabled;
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int effective_cpu_threads(int n_threads) {
    if (n_threads > 0) {
        return n_threads;
    }
    const unsigned int detected = std::thread::hardware_concurrency();
    if (detected == 0) {
        return 4;
    }
    return std::max(1, std::min(32, static_cast<int>(detected)));
}

void dump_profile() {
    std::lock_guard<std::mutex> lock(profile_mutex);
    if (profile_stats.empty()) {
        return;
    }
    std::cerr << "vlacpp ggml profile\n";
    std::cerr << std::left << std::setw(48) << "label" << std::right << std::setw(8) << "calls"
              << std::setw(12) << "alloc_ms" << std::setw(12) << "upload_ms"
              << std::setw(12) << "compute_ms" << std::setw(12) << "download_ms"
              << std::setw(14) << "upload_mb" << std::setw(14) << "download_mb" << '\n';
    for (const auto & entry : profile_stats) {
        const ProfileStats & stats = entry.second;
        std::cerr << std::left << std::setw(48) << entry.first << std::right << std::setw(8) << stats.calls
                  << std::setw(12) << std::fixed << std::setprecision(3) << stats.alloc_ms
                  << std::setw(12) << stats.upload_ms
                  << std::setw(12) << stats.compute_ms
                  << std::setw(12) << stats.download_ms
                  << std::setw(14) << static_cast<double>(stats.upload_bytes) / (1024.0 * 1024.0)
                  << std::setw(14) << static_cast<double>(stats.download_bytes) / (1024.0 * 1024.0) << '\n';
    }
}

void register_profile_dump_once() {
    if (!profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(profile_mutex);
    if (!profile_registered) {
        std::atexit(dump_profile);
        profile_registered = true;
    }
}

void record_profile(const char * label, const ProfileStats & delta) {
    if (!profile_enabled()) {
        return;
    }
    register_profile_dump_once();
    std::lock_guard<std::mutex> lock(profile_mutex);
    ProfileStats & stats = profile_stats[label != nullptr ? label : "<unknown>"];
    stats.calls += delta.calls;
    stats.alloc_ms += delta.alloc_ms;
    stats.upload_ms += delta.upload_ms;
    stats.compute_ms += delta.compute_ms;
    stats.download_ms += delta.download_ms;
    stats.upload_bytes += delta.upload_bytes;
    stats.download_bytes += delta.download_bytes;
}

ggml_tensor * materialize_empty_f32_tensor(
    GgmlRunnerResources & resources,
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const void * key_ptr,
    uint64_t generation,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    DeviceTensorKey key{backend, key_ptr, generation, GGML_TYPE_F32, n_dims, ne0, ne1, ne2, 1};
    auto it = resources.device_tensors.find(key);
    if (it != resources.device_tensors.end()) {
        return it->second->tensor;
    }

    auto cached = std::make_unique<DeviceTensor>();
    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    cached->ctx = ggml_init(params);
    if (cached->ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml cached tensor context");
    }
    if (n_dims == 3) {
        cached->tensor = ggml_new_tensor_3d(cached->ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    } else if (n_dims == 2) {
        cached->tensor = ggml_new_tensor_2d(cached->ctx, GGML_TYPE_F32, ne0, ne1);
    } else {
        cached->tensor = ggml_new_tensor_1d(cached->ctx, GGML_TYPE_F32, ne0);
    }
    if (cached->tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml cached tensor metadata");
    }
    cached->buffer = ggml_backend_alloc_ctx_tensors_from_buft(cached->ctx, buft ? buft : default_buft(backend));
    if (cached->buffer == nullptr) {
        throw std::runtime_error("failed to allocate ggml cached backend buffer");
    }
    ggml_backend_buffer_set_usage(cached->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tensor * result = cached->tensor;
    resources.device_tensors.emplace(key, std::move(cached));
    return result;
}

ggml_tensor * new_weight_tensor(ggml_context * ctx, const Tensor & source, ggml_type type) {
    if (source.shape.size() == 1) {
        return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, source.shape[0]);
    }
    if (source.shape.size() == 2) {
        return ggml_new_tensor_2d(ctx, type, source.shape[0], source.shape[1]);
    }
    if (source.shape.size() == 3) {
        return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, source.shape[0], source.shape[1], source.shape[2]);
    }
    if (source.shape.size() == 4) {
        return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
    }
    return nullptr;
}

ggml_type tensor_storage_type(const Tensor & source) {
    if (source.data_type == "fp32") {
        return GGML_TYPE_F32;
    }
    if (source.data_type == "f16") {
        return GGML_TYPE_F16;
    }
    if (source.data_type == "bf16") {
        return GGML_TYPE_BF16;
    }
    throw std::runtime_error("unsupported GGUF tensor storage dtype: " + source.data_type);
}

void upload_weight_tensor(
    ggml_tensor * dst,
    const Tensor & source,
    std::vector<ggml_fp16_t> & f16,
    std::vector<ggml_bf16_t> & bf16) {
    const size_t nbytes = ggml_nbytes(dst);
    if (dst->type == GGML_TYPE_F16) {
        f16.resize(source.data.size());
        ggml_fp32_to_fp16_row(source.data.data(), f16.data(), static_cast<int64_t>(source.data.size()));
        ggml_backend_tensor_set(dst, f16.data(), 0, nbytes);
        return;
    }
    if (dst->type == GGML_TYPE_BF16) {
        bf16.resize(source.data.size());
        ggml_fp32_to_bf16_row(source.data.data(), bf16.data(), static_cast<int64_t>(source.data.size()));
        ggml_backend_tensor_set(dst, bf16.data(), 0, nbytes);
        return;
    }
    ggml_backend_tensor_set(dst, source.data.data(), 0, nbytes);
}

void set_cpu_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
    if (set_threads != nullptr) {
        set_threads(backend, effective_cpu_threads(n_threads));
    }
}

BackendState & get_cuda_state(int n_threads) {
    if (!cuda_state) {
        cuda_state = std::make_unique<BackendState>();
        cuda_state->gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (cuda_state->gpu == nullptr) {
            cuda_state.reset();
            throw std::runtime_error("CUDA backend requested but no ggml GPU backend is available");
        }
        cuda_state->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cuda_state->cpu == nullptr) {
            cuda_state.reset();
            throw std::runtime_error("failed to initialize ggml CPU host backend");
        }

        ggml_backend_t backends[] = {cuda_state->gpu, cuda_state->cpu};
        ggml_backend_buffer_type_t bufts[] = {
            default_buft(cuda_state->gpu),
            host_buft_for_backend(cuda_state->gpu),
        };
        cuda_state->sched = ggml_backend_sched_new(backends, bufts, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        if (cuda_state->sched == nullptr) {
            cuda_state.reset();
            throw std::runtime_error("failed to initialize ggml backend scheduler");
        }
    }
    set_cpu_threads(cuda_state->cpu, n_threads);
    return *cuda_state;
}

BackendState & get_cpu_state(int n_threads) {
    if (!cpu_state) {
        cpu_state = std::make_unique<BackendState>();
        cpu_state->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu_state->cpu == nullptr) {
            cpu_state.reset();
            throw std::runtime_error("failed to initialize ggml CPU backend");
        }
        ggml_backend_t backends[] = {cpu_state->cpu};
        ggml_backend_buffer_type_t bufts[] = {default_buft(cpu_state->cpu)};
        cpu_state->sched = ggml_backend_sched_new(backends, bufts, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (cpu_state->sched == nullptr) {
            cpu_state.reset();
            throw std::runtime_error("failed to initialize ggml CPU backend scheduler");
        }
    }
    set_cpu_threads(cpu_state->cpu, n_threads);
    return *cpu_state;
}

GgmlContext::GgmlContext(ggml_init_params params) {
    ctx_ = ggml_init(params);
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize ggml context");
    }
}

GgmlContext::~GgmlContext() {
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

ggml_context * GgmlContext::get() const {
    return ctx_;
}

GgmlContext::operator ggml_context *() const {
    return ctx_;
}

GgmlRunner::GgmlRunner(const BackendConfig & backend)
    : backend_config_(backend),
      resources_(std::make_unique<GgmlRunnerResources>()) {
    if (backend_config_.backend == VLACPP_BACKEND_CUDA) {
        BackendState & state = get_cuda_state(backend_config_.n_threads);
        gpu_backend_ = state.gpu;
        cpu_backend_ = state.cpu;
        sched_ = state.sched;
        capabilities_.default_2d_weight_type = GGML_TYPE_BF16;
    } else {
        BackendState & state = get_cpu_state(backend_config_.n_threads);
        gpu_backend_ = state.cpu;
        cpu_backend_ = state.cpu;
        sched_ = state.sched;
    }
    capabilities_.device_weight_store = uses_backend();
    capabilities_.stable_graph_workspace = uses_backend();
    buft_policy_ = make_buft_policy(gpu_backend_, cpu_backend_);
}

GgmlRunner::~GgmlRunner() = default;

bool GgmlRunner::uses_backend() const {
    return sched_ != nullptr;
}

const GgmlBackendCapabilities & GgmlRunner::capabilities() const {
    return capabilities_;
}

const GgmlBuftPolicy & GgmlRunner::buft_policy() const {
    return buft_policy_;
}

ggml_init_params GgmlRunner::init_params(size_t mem_size, const void * stable_id, uint64_t stable_variant) const {
    ggml_init_params params{};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = uses_backend();
    if (capabilities_.stable_graph_workspace && stable_id != nullptr) {
        GraphWorkspaceKey key{gpu_backend_, stable_id, stable_variant, GGML_PAD(mem_size, GGML_MEM_ALIGN)};
        AlignedBuffer & buffer = resources_->graph_workspaces[key];
        buffer.ensure(mem_size);
        params.mem_size = buffer.size;
        params.mem_buffer = buffer.data;
    }
    return params;
}

void GgmlRunner::materialize_weights(const std::vector<Tensor *> & tensors) const {
    materialize_weights(tensors, capabilities_.default_2d_weight_type);
}

void GgmlRunner::materialize_weights(const std::vector<Tensor *> & tensors, ggml_type default_2d_weight_type) const {
    if (!capabilities_.device_weight_store || tensors.empty()) {
        return;
    }
    if (resources_->weight_buffer != nullptr) {
        return;
    }

    ggml_init_params params{};
    params.mem_size = (tensors.size() + 1) * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    resources_->weight_ctx = ggml_init(params);
    if (resources_->weight_ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml weight store context");
    }

    for (Tensor * source : tensors) {
        if (source == nullptr || source->shape.empty() || source->shape.size() > 4) {
            continue;
        }
        ggml_type type = GGML_TYPE_F32;
        if (source->shape.size() == 2) {
            type = default_2d_weight_type == GGML_TYPE_COUNT ? tensor_storage_type(*source) : default_2d_weight_type;
        }
        ggml_tensor * tensor = new_weight_tensor(resources_->weight_ctx, *source, type);
        if (tensor == nullptr) {
            throw std::runtime_error("failed to allocate ggml weight store tensor metadata");
        }
        resources_->weights[source] = tensor;
    }

    resources_->weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(resources_->weight_ctx, buft_policy_.model_buft);
    if (resources_->weight_buffer == nullptr) {
        throw std::runtime_error("failed to allocate ggml weight store backend buffer");
    }
    ggml_backend_buffer_set_usage(resources_->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<ggml_fp16_t> f16;
    std::vector<ggml_bf16_t> bf16;
    for (const auto & item : resources_->weights) {
        upload_weight_tensor(item.second, *item.first, f16, bf16);
    }
    for (Tensor * source : tensors) {
        if (resources_->weights.find(source) == resources_->weights.end()) {
            continue;
        }
        source->data.clear();
        source->data.shrink_to_fit();
    }
}

ggml_tensor * GgmlRunner::stored_weight(const Tensor & tensor) const {
    if (!capabilities_.device_weight_store) {
        return nullptr;
    }
    auto it = resources_->weights.find(&tensor);
    if (it == resources_->weights.end()) {
        return nullptr;
    }
    return it->second;
}

ggml_tensor * GgmlRunner::materialize_device_f32_3d_from_backend(
    ggml_context * ctx,
    const void * key,
    const ggml_tensor * source,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) const {
    if (!uses_backend()) {
        (void) ctx;
        (void) key;
        (void) source;
        (void) ne0;
        (void) ne1;
        (void) ne2;
        throw std::invalid_argument("backend cached tensor copy requires a backend");
    }
    ggml_tensor * cached = materialize_empty_f32_tensor(
        *resources_, gpu_backend_, buft_policy_.backend_buft, key, 0, 3, ne0, ne1, ne2);
    ggml_backend_tensor_copy(source, cached);
    return cached;
}

void GgmlRunner::set_input(
    std::vector<GgmlInput> & inputs,
    ggml_tensor * tensor,
    const void * data,
    size_t size) const {
    if (uses_backend()) {
        ggml_set_input(tensor);
        inputs.push_back({tensor, data, size});
        return;
    }
    std::memcpy(tensor->data, data, size);
}

void GgmlRunner::compute(
    ggml_context * ctx,
    ggml_cgraph * graph,
    const std::vector<GgmlInput> & inputs,
    const char * error_message) const {
    ggml_status status = GGML_STATUS_SUCCESS;
    const bool do_profile = profile_enabled();
    if (!do_profile) {
        if (!uses_backend()) {
            status = ggml_graph_compute_with_ctx(ctx, graph, effective_cpu_threads(backend_config_.n_threads));
        } else {
            ggml_backend_sched_reset(sched_);
            if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
                throw std::runtime_error("ggml backend graph allocation failed");
            }
            for (const GgmlInput & input : inputs) {
                ggml_backend_tensor_set(input.tensor, input.data, 0, input.size);
            }
            status = ggml_backend_sched_graph_compute(sched_, graph);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(error_message);
        }
        return;
    }

    ProfileStats profile{};
    profile.calls = 1;
    if (!uses_backend()) {
        const auto compute_start = std::chrono::steady_clock::now();
        status = ggml_graph_compute_with_ctx(ctx, graph, effective_cpu_threads(backend_config_.n_threads));
        profile.compute_ms = elapsed_ms(compute_start, std::chrono::steady_clock::now());
    } else {
        auto start = std::chrono::steady_clock::now();
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
            throw std::runtime_error("ggml backend graph allocation failed");
        }
        profile.alloc_ms = elapsed_ms(start, std::chrono::steady_clock::now());
        start = std::chrono::steady_clock::now();
        for (const GgmlInput & input : inputs) {
            ggml_backend_tensor_set(input.tensor, input.data, 0, input.size);
            profile.upload_bytes += input.size;
        }
        profile.upload_ms = elapsed_ms(start, std::chrono::steady_clock::now());
        start = std::chrono::steady_clock::now();
        status = ggml_backend_sched_graph_compute(sched_, graph);
        profile.compute_ms = elapsed_ms(start, std::chrono::steady_clock::now());
    }
    profile_label_ = error_message != nullptr ? error_message : "<unknown>";
    record_profile(profile_label_.c_str(), profile);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(error_message);
    }
}

void GgmlRunner::get_output(const ggml_tensor * tensor, void * data, size_t size) const {
    if (uses_backend()) {
        const auto download_start = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(tensor, data, 0, size);
        if (profile_enabled()) {
            ProfileStats profile{};
            profile.download_ms = elapsed_ms(download_start, std::chrono::steady_clock::now());
            profile.download_bytes = size;
            record_profile(profile_label_.empty() ? "<output>" : profile_label_.c_str(), profile);
        }
        return;
    }
    std::memcpy(data, tensor->data, size);
}

} // namespace vlacpp