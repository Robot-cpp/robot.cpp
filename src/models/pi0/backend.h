#pragma once

#include "models/ggml_backend.h"
#include "models/pi0/types.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp::pi0 {

struct Pi0DeviceTensor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;

    ~Pi0DeviceTensor();
};

struct Pi0DeviceTensorKey {
    const void * tensor = nullptr;
    ggml_type type = GGML_TYPE_F32;
    int n_dims = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;
    int64_t ne3 = 0;

    bool operator<(const Pi0DeviceTensorKey & other) const;
};

struct Pi0ComponentBackend {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<ggml_backend_t> backends;
    backend_buft_policy buft_policy;
    int n_threads = 0;
    mutable std::map<Pi0DeviceTensorKey, std::unique_ptr<Pi0DeviceTensor>> device_tensors;

    Pi0ComponentBackend() = default;
    ~Pi0ComponentBackend();

    Pi0ComponentBackend(const Pi0ComponentBackend &) = delete;
    Pi0ComponentBackend & operator=(const Pi0ComponentBackend &) = delete;

    Pi0ComponentBackend(Pi0ComponentBackend && other) noexcept;
    Pi0ComponentBackend & operator=(Pi0ComponentBackend && other) noexcept;

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

void pi0_init_backend(
    Pi0ComponentBackend & runtime,
    const Pi0BackendConfig & base,
    const Pi0ComponentConfig & component,
    const char * label,
    int verbosity);

ggml_init_params pi0_graph_init_params(size_t mem_size);

ggml_tensor * pi0_materialize_device_f32_3d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    const ggml_tensor * source,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2);

ggml_tensor * pi0_materialize_device_f32_2d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    const ggml_tensor * source,
    int64_t ne0,
    int64_t ne1);

ggml_tensor * pi0_upload_device_f32_2d(
    const Pi0ComponentBackend & runtime,
    const void * key,
    const float * source,
    int64_t ne0,
    int64_t ne1);

} // namespace robotcpp::pi0
