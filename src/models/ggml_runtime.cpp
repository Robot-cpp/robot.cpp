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
#include <vector>

namespace vlacpp {
namespace {

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

struct StableBufferKey {
    const void * backend = nullptr;
    const void * id = nullptr;
    uint64_t variant = 0;
    size_t size = 0;

    bool operator<(const StableBufferKey & other) const {
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

struct CachedWeight {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;

    ~CachedWeight() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct WeightKey {
    const void * backend = nullptr;
    const void * tensor = nullptr;
    uint64_t generation = 0;
    ggml_type type = GGML_TYPE_F32;
    int n_dims = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;

    bool operator<(const WeightKey & other) const {
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
        return ne2 < other.ne2;
    }
};

thread_local std::unique_ptr<BackendState> cuda_state;
thread_local std::unique_ptr<BackendState> cpu_state;
thread_local std::map<WeightKey, std::unique_ptr<CachedWeight>> cuda_weight_cache;
thread_local std::map<StableBufferKey, AlignedBuffer> cuda_stable_compute_buffers;

struct ProfileStats {
    int64_t calls = 0;
    double alloc_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double download_ms = 0.0;
    size_t upload_bytes = 0;
    size_t download_bytes = 0;
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

ggml_tensor * cache_f32_tensor(
    ggml_backend_t backend,
    const void * key_ptr,
    uint64_t generation,
    const float * data,
    size_t element_count,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    WeightKey key{backend, key_ptr, generation, GGML_TYPE_F32, n_dims, ne0, ne1, ne2};
    auto it = cuda_weight_cache.find(key);
    if (it != cuda_weight_cache.end()) {
        return it->second->tensor;
    }

    auto cached = std::make_unique<CachedWeight>();
    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    cached->ctx = ggml_init(params);
    if (cached->ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml weight context");
    }
    if (n_dims == 1) {
        cached->tensor = ggml_new_tensor_1d(cached->ctx, GGML_TYPE_F32, ne0);
    } else if (n_dims == 2) {
        cached->tensor = ggml_new_tensor_2d(cached->ctx, GGML_TYPE_F32, ne0, ne1);
    } else {
        cached->tensor = ggml_new_tensor_3d(cached->ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    }
    if (cached->tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml weight tensor metadata");
    }
    cached->buffer = ggml_backend_alloc_ctx_tensors(cached->ctx, backend);
    if (cached->buffer == nullptr) {
        throw std::runtime_error("failed to allocate ggml backend weight buffer");
    }
    ggml_backend_buffer_set_usage(cached->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(cached->tensor, data, 0, element_count * sizeof(float));
    ggml_tensor * result = cached->tensor;
    cuda_weight_cache.emplace(key, std::move(cached));
    return result;
}

ggml_tensor * cache_f16_tensor(
    ggml_backend_t backend,
    const void * key_ptr,
    const float * data,
    size_t element_count,
    int64_t ne0,
    int64_t ne1) {
    WeightKey key{backend, key_ptr, 0, GGML_TYPE_F16, 2, ne0, ne1, 1};
    auto it = cuda_weight_cache.find(key);
    if (it != cuda_weight_cache.end()) {
        return it->second->tensor;
    }

    auto cached = std::make_unique<CachedWeight>();
    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    cached->ctx = ggml_init(params);
    if (cached->ctx == nullptr) {
        throw std::runtime_error("failed to initialize ggml weight context");
    }
    cached->tensor = ggml_new_tensor_2d(cached->ctx, GGML_TYPE_F16, ne0, ne1);
    if (cached->tensor == nullptr) {
        throw std::runtime_error("failed to allocate ggml weight tensor metadata");
    }
    cached->buffer = ggml_backend_alloc_ctx_tensors(cached->ctx, backend);
    if (cached->buffer == nullptr) {
        throw std::runtime_error("failed to allocate ggml backend weight buffer");
    }
    std::vector<ggml_fp16_t> f16(element_count);
    ggml_fp32_to_fp16_row(data, f16.data(), static_cast<int64_t>(element_count));
    ggml_backend_buffer_set_usage(cached->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(cached->tensor, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
    ggml_tensor * result = cached->tensor;
    cuda_weight_cache.emplace(key, std::move(cached));
    return result;
}

ggml_tensor * cache_empty_f32_tensor(
    ggml_backend_t backend,
    const void * key_ptr,
    uint64_t generation,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) {
    WeightKey key{backend, key_ptr, generation, GGML_TYPE_F32, n_dims, ne0, ne1, ne2};
    auto it = cuda_weight_cache.find(key);
    if (it != cuda_weight_cache.end()) {
        return it->second->tensor;
    }

    auto cached = std::make_unique<CachedWeight>();
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
    cached->buffer = ggml_backend_alloc_ctx_tensors(cached->ctx, backend);
    if (cached->buffer == nullptr) {
        throw std::runtime_error("failed to allocate ggml cached backend buffer");
    }
    ggml_backend_buffer_set_usage(cached->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tensor * result = cached->tensor;
    cuda_weight_cache.emplace(key, std::move(cached));
    return result;
}

ggml_tensor * cache_weight(
    ggml_backend_t backend,
    const Tensor & source,
    int n_dims,
    int64_t ne0,
    int64_t ne1,
    bool use_f16_2d) {
    if (n_dims == 2 && use_f16_2d) {
        return cache_f16_tensor(backend, &source, source.data.data(), source.data.size(), ne0, ne1);
    }
    return cache_f32_tensor(backend, &source, 0, source.data.data(), source.data.size(), n_dims, ne0, ne1, 1);
}

void set_cpu_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads"));
    if (set_threads != nullptr) {
        set_threads(backend, std::max(1, n_threads));
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
            throw std::runtime_error("failed to initialize ggml CPU fallback backend");
        }

        ggml_backend_t backends[] = {cuda_state->gpu, cuda_state->cpu};
        cuda_state->sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
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
        cpu_state->sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        if (cpu_state->sched == nullptr) {
            cpu_state.reset();
            throw std::runtime_error("failed to initialize ggml CPU backend scheduler");
        }
    }
    set_cpu_threads(cpu_state->cpu, n_threads);
    return *cpu_state;
}

} // namespace

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
    : backend_config_(backend) {
    if (backend_config_.backend == VLACPP_BACKEND_CUDA) {
        BackendState & state = get_cuda_state(backend_config_.n_threads);
        gpu_backend_ = state.gpu;
        cpu_backend_ = state.cpu;
        sched_ = state.sched;
    } else {
        BackendState & state = get_cpu_state(backend_config_.n_threads);
        gpu_backend_ = state.cpu;
        cpu_backend_ = state.cpu;
        sched_ = state.sched;
    }
}

GgmlRunner::~GgmlRunner() = default;

bool GgmlRunner::uses_backend() const {
    return sched_ != nullptr;
}

ggml_init_params GgmlRunner::init_params(size_t mem_size, const void * stable_id, uint64_t stable_variant) const {
    ggml_init_params params{};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = uses_backend();
    stable_id_ = stable_id;
    stable_variant_ = stable_variant;
    if (uses_backend() && stable_id != nullptr) {
        StableBufferKey key{gpu_backend_, stable_id, stable_variant, GGML_PAD(mem_size, GGML_MEM_ALIGN)};
        AlignedBuffer & buffer = cuda_stable_compute_buffers[key];
        buffer.ensure(mem_size);
        params.mem_size = buffer.size;
        params.mem_buffer = buffer.data;
    }
    return params;
}

ggml_tensor * GgmlRunner::new_weight_1d(ggml_context * ctx, const Tensor & tensor) const {
    if (uses_backend()) {
        return cache_weight(gpu_backend_, tensor, 1, tensor.shape[0], 1, false);
    }
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor.shape[0]);
    std::memcpy(ggml_get_data_f32(weight), tensor.data.data(), tensor.data.size() * sizeof(float));
    return weight;
}

ggml_tensor * GgmlRunner::new_weight_1d_plus_one(ggml_context * ctx, const Tensor & tensor) const {
    std::vector<float> scaled(tensor.data.size());
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        scaled[i] = 1.0f + tensor.data[i];
    }
    if (uses_backend()) {
        return cache_f32_tensor(gpu_backend_, &tensor, 1, scaled.data(), scaled.size(), 1, tensor.shape[0], 1, 1);
    }
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor.shape[0]);
    std::memcpy(ggml_get_data_f32(weight), scaled.data(), scaled.size() * sizeof(float));
    return weight;
}

ggml_tensor * GgmlRunner::new_weight_2d(ggml_context * ctx, const Tensor & tensor) const {
    if (uses_backend()) {
        return cache_weight(
            gpu_backend_,
            tensor,
            2,
            tensor.shape[0],
            tensor.shape[1],
            backend_config_.backend == VLACPP_BACKEND_CUDA);
    }
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tensor.shape[0], tensor.shape[1]);
    std::memcpy(ggml_get_data_f32(weight), tensor.data.data(), tensor.data.size() * sizeof(float));
    return weight;
}

ggml_tensor * GgmlRunner::new_cached_f32_3d(
    ggml_context * ctx,
    const void * key,
    uint64_t generation,
    const float * data,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) const {
    const size_t element_count = static_cast<size_t>(ne0) * static_cast<size_t>(ne1) * static_cast<size_t>(ne2);
    if (uses_backend()) {
        if (data == nullptr) {
            return cache_empty_f32_tensor(gpu_backend_, key, generation, 3, ne0, ne1, ne2);
        }
        return cache_f32_tensor(gpu_backend_, key, generation, data, element_count, 3, ne0, ne1, ne2);
    }
    if (data == nullptr) {
        throw std::invalid_argument("CPU cached tensor requires host data");
    }
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    std::memcpy(ggml_get_data_f32(tensor), data, element_count * sizeof(float));
    return tensor;
}

ggml_tensor * GgmlRunner::new_cached_f32_3d_from_backend(
    ggml_context * ctx,
    const void * key,
    uint64_t generation,
    const ggml_tensor * source,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2) const {
    if (!uses_backend()) {
        (void) ctx;
        (void) key;
        (void) generation;
        (void) source;
        (void) ne0;
        (void) ne1;
        (void) ne2;
        throw std::invalid_argument("backend cached tensor copy requires a backend");
    }
    ggml_tensor * cached = cache_empty_f32_tensor(gpu_backend_, key, generation, 3, ne0, ne1, ne2);
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
    ProfileStats profile{};
    profile.calls = 1;
    if (!uses_backend()) {
        const auto compute_start = std::chrono::steady_clock::now();
        status = ggml_graph_compute_with_ctx(ctx, graph, std::max(1, backend_config_.n_threads));
        profile.compute_ms = elapsed_ms(compute_start, std::chrono::steady_clock::now());
    } else {
        const auto alloc_start = std::chrono::steady_clock::now();
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
            throw std::runtime_error("ggml backend graph allocation failed");
        }
        profile.alloc_ms = elapsed_ms(alloc_start, std::chrono::steady_clock::now());
        const auto upload_start = std::chrono::steady_clock::now();
        for (const GgmlInput & input : inputs) {
            ggml_backend_tensor_set(input.tensor, input.data, 0, input.size);
            profile.upload_bytes += input.size;
        }
        profile.upload_ms = elapsed_ms(upload_start, std::chrono::steady_clock::now());
        const auto compute_start = std::chrono::steady_clock::now();
        status = ggml_backend_sched_graph_compute(sched_, graph);
        profile.compute_ms = elapsed_ms(compute_start, std::chrono::steady_clock::now());
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
