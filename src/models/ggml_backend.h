#pragma once

#include "ggml-backend.h"

#include <string>
#include <vector>

enum class backend_mode {
    cpu,
    cuda,
    metal,
};

struct backend_buft_policy {
    ggml_backend_buffer_type_t model_buft = nullptr;
    ggml_backend_buffer_type_t runtime_buft = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
};

struct backend_scheduler_config {
    int max_nodes = 4096;
    bool parallel = false;
    bool op_offload = true;
};

class backend_loader {
public:
    backend_loader() = default;

    bool load(
        ggml_backend_t & cpu_backend,
        std::vector<ggml_backend_t> & backends,
        ggml_backend_sched_t & scheduler,
        backend_buft_policy & buft_policy,
        bool use_accel,
        const backend_scheduler_config & scheduler_config,
        int verbosity);

    backend_mode mode() const;
    const std::string & error() const;

private:
    backend_mode desired_mode() const;

    bool init_backends(
        ggml_backend_t & cpu_backend,
        std::vector<ggml_backend_t> & backends,
        bool use_accel,
        int verbosity);

    bool init_buft_policy(
        ggml_backend_t cpu_backend,
        const std::vector<ggml_backend_t> & backends,
        backend_buft_policy & buft_policy,
        int verbosity);

    bool init_scheduler(
        std::vector<ggml_backend_t> & backends,
        ggml_backend_sched_t & scheduler,
        const backend_buft_policy & buft_policy,
        const backend_scheduler_config & scheduler_config,
        int verbosity);

    bool fail(const std::string & message);

    backend_mode mode_ = backend_mode::cpu;
    std::string error_;
};

void set_backend_threads(
    const std::vector<ggml_backend_t> & backends,
    int n_threads);

bool robotcpp_backend_use_accel_from_env(bool default_value);
