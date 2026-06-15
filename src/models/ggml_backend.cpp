#include "models/ggml_backend.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

static const char * backend_mode_name(backend_mode mode) {
    switch (mode) {
        case backend_mode::cuda:  return "cuda";
        case backend_mode::metal: return "metal";
        case backend_mode::cpu:   return "cpu";
    }
    return "unknown";
}

static const char * backend_buft_name(ggml_backend_buffer_type_t buft) {
    return buft ? ggml_backend_buft_name(buft) : "(null)";
}

static std::string backend_env_lower(const char * value) {
    std::string out(value ? value : "");
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return out;
}

static ggml_backend_buffer_type_t backend_host_buft(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
}

static bool mode_for_device(
    ggml_backend_dev_t dev,
    backend_mode & mode,
    std::string & error) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";
    if (reg_name && std::strcmp(reg_name, "CUDA") == 0) {
        mode = backend_mode::cuda;
        return true;
    }
    else if (reg_name && (std::strcmp(reg_name, "Metal") == 0 || std::strcmp(reg_name, "MTL") == 0)) {
        mode = backend_mode::metal;
        return true;
    }
    else if (reg_name && std::strcmp(reg_name, "CPU") == 0) {
        mode = backend_mode::cpu;
        return true;
    }
    else if (reg_name && std::strcmp(reg_name, "BLAS") == 0) {
        mode = backend_mode::cpu;
        return true;
    }

    error = std::string("unknown GGML backend registry: ") +
        (reg_name && reg_name[0] ? reg_name : "(null)");
    return false;
}

backend_mode backend_loader::desired_mode() const {
#if defined(GGML_USE_CUDA)
    return backend_mode::cuda;
#elif defined(GGML_USE_METAL)
    return backend_mode::metal;
#else
    return backend_mode::cpu;
#endif
}

bool backend_loader::load(
    ggml_backend_t & cpu_backend,
    std::vector<ggml_backend_t> & backends,
    ggml_backend_sched_t & scheduler,
    backend_buft_policy & buft_policy,
    bool use_accel,
    const backend_scheduler_config & scheduler_config,
    int verbosity) {
    cpu_backend = nullptr;
    scheduler = nullptr;
    backends.clear();
    buft_policy = backend_buft_policy();
    mode_ = backend_mode::cpu;

    if (!init_backends(cpu_backend, backends, use_accel, verbosity) ||
        !init_buft_policy(cpu_backend, backends, buft_policy, verbosity) ||
        !init_scheduler(backends, scheduler, buft_policy, scheduler_config, verbosity)) {
        return false;
    }

    return true;
}

bool backend_loader::init_backends(
    ggml_backend_t & cpu_backend,
    std::vector<ggml_backend_t> & backends,
    bool use_accel,
    int verbosity) {

    mode_ = backend_mode::cpu;
    const backend_mode wanted = desired_mode();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        backend_mode dev_mode = backend_mode::cpu;
        std::string mode_error;
        if (!mode_for_device(dev, dev_mode, mode_error)) {
            return fail(mode_error);
        }

        if (dev_mode != wanted) {
            continue;
        }
        
        if (type == GGML_BACKEND_DEVICE_TYPE_ACCEL && !use_accel) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend) {
            backends.push_back(backend);
            mode_ = wanted;
            break;
        }
    }

    cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        for (ggml_backend_t backend : backends) {
            if (backend) {
                ggml_backend_free(backend);
            }
        }
        backends.clear();
        return fail("failed to initialize CPU backend");
    }
    backends.push_back(cpu_backend);

    if (verbosity >= 1) {
        std::fprintf(stderr,
            "%s: desired_mode=%s resolved_mode=%s backend_count=%zu\n",
            __func__,
            backend_mode_name(wanted),
            backend_mode_name(mode_),
            backends.size());
    }

    return true;
}


bool backend_loader::init_buft_policy(
    ggml_backend_t cpu_backend,
    const std::vector<ggml_backend_t> & backends,
    backend_buft_policy & buft_policy,
    int verbosity) {
    if (!cpu_backend) {
        return fail("CPU backend is required for buffer policy");
    }

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(cpu_backend);
    if (!cpu_buft) {
        return fail("failed to resolve CPU backend buffer type");
    }
    if (backends.empty()) {
        return fail("buffer policy requires at least one backend");
    }

    ggml_backend_buffer_type_t host_buft = backend_host_buft(backends.front());
    buft_policy.host_buft = host_buft ? host_buft : cpu_buft;
    if(mode_ == backend_mode::cpu) {
        buft_policy.model_buft = cpu_buft;
        buft_policy.runtime_buft = cpu_buft;
    }
    else if(mode_ == backend_mode::cuda || mode_ == backend_mode::metal) {
        ggml_backend_buffer_type_t device_buft = ggml_backend_get_default_buffer_type(backends.front());
        if (!device_buft) {
            return fail("failed to resolve primary backend buffer type");
        }
        buft_policy.model_buft = device_buft;
        buft_policy.runtime_buft = device_buft;
    }
    else
    {
        return fail("unsupported backend mode for buffer policy");
    }

    if (verbosity >= 1) {
        std::fprintf(stderr,
            "%s: model_buft=%s runtime_buft=%s host_buft=%s\n",
            __func__,
            backend_buft_name(buft_policy.model_buft),
            backend_buft_name(buft_policy.runtime_buft),
            backend_buft_name(buft_policy.host_buft));
    }

    return true;
}

bool backend_loader::init_scheduler(
    std::vector<ggml_backend_t> & backends,
    ggml_backend_sched_t & scheduler,
    const backend_buft_policy & buft_policy,
    const backend_scheduler_config & scheduler_config,
    int verbosity) {
    if (backends.empty()) {
        return fail("scheduler requires at least one backend");
    }

    std::vector<ggml_backend_buffer_type_t> sched_bufts;
    sched_bufts.reserve(backends.size());
    for (ggml_backend_t backend : backends) {
        if (ggml_backend_is_cpu(backend) && buft_policy.host_buft) {
            sched_bufts.push_back(buft_policy.host_buft);
        } else {
            sched_bufts.push_back(ggml_backend_get_default_buffer_type(backend));
        }
    }

    scheduler = ggml_backend_sched_new(
        backends.data(),
        sched_bufts.data(),
        backends.size(),
        scheduler_config.max_nodes,
        scheduler_config.parallel,
        scheduler_config.op_offload);
    if (!scheduler) {
        return fail("failed to initialize backend scheduler");
    }

    if (verbosity >= 1) {
        std::fprintf(stderr,
            "%s: enabled scheduler with %zu backend(s)\n",
            __func__,
            backends.size());
    }

    return true;
}

backend_mode backend_loader::mode() const {
    return mode_;
}

const std::string & backend_loader::error() const {
    return error_;
}

bool backend_loader::fail(const std::string & message) {
    error_ = message;
    std::fprintf(stderr, "%s: %s\n", __func__, error_.c_str());
    return false;
}

void set_backend_threads(
    const std::vector<ggml_backend_t> & backends,
    int n_threads) {
    if (n_threads <= 0) {
        return;
    }

    for (ggml_backend_t backend : backends) {
        if (!backend) {
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg) {
            continue;
        }

        auto set_n_threads_fn = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads_fn) {
            set_n_threads_fn(backend, n_threads);
        }
    }
}

bool robotcpp_backend_use_accel_from_env(bool default_value) {
    const char * value = std::getenv("ROBOTCPP_BACKEND");
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    const std::string backend = backend_env_lower(value);
    if (backend == "cpu") {
        return false;
    }
    if (backend == "auto" ||
        backend == "cuda" ||
        backend == "metal" ||
        backend == "blas") {
        return true;
    }

    std::fprintf(stderr,
        "%s: unsupported ROBOTCPP_BACKEND='%s'; using default backend selection\n",
        __func__,
        value);
    return default_value;
}
