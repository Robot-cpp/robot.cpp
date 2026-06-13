#include "models/gguf_loader.h"

#include <cstring>
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>

bool gguf_loader::load(
    const char * path,
    ggml_backend_buffer_type_t model_buft,
    gguf_load_result & out,
    int verbosity) {
    if (!path || !model_buft) {
        return fail("invalid GGUF loader arguments");
    }

    out = gguf_load_result();

    ggml_context * meta = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta,
    };

    gguf_context * gguf = gguf_init_from_file(path, params);
    if (!gguf) {
        return fail(std::string("failed to load GGUF: ") + path);
    }

    ggml_context * ctx_data = nullptr;
    ggml_backend_buffer_t model_buffer = nullptr;

    auto cleanup = [&]() {
        if (model_buffer) {
            ggml_backend_buffer_free(model_buffer);
            model_buffer = nullptr;
        }
        if (ctx_data) {
            ggml_free(ctx_data);
            ctx_data = nullptr;
        }
        if (meta) {
            ggml_free(meta);
            meta = nullptr;
        }
        if (gguf) {
            gguf_free(gguf);
            gguf = nullptr;
        }
    };

    try {
        const int n_tensors = gguf_get_n_tensors(gguf);
        if (verbosity >= 1) {
            std::fprintf(stderr,
                "%s: loaded GGUF: %d tensors, %lld kv pairs\n",
                __func__, n_tensors, (long long) gguf_get_n_kv(gguf));
        }

        if (!parse_metadata(gguf)) {
            cleanup();
            return false;
        }

        size_t model_size = 0;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * tensor = ggml_get_tensor(meta, name);
            if (!tensor) {
                cleanup();
                return fail(std::string("missing tensor metadata: ") + name);
            }
            model_size += ggml_nbytes(tensor);
        }
        if (verbosity >= 1) {
            std::fprintf(stderr, "%s: model size = %.2f MB\n",
                __func__, model_size / (1024.0 * 1024.0));
        }

        const size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 1);
        ggml_init_params ggml_params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_data = ggml_init(ggml_params);
        if (!ctx_data) {
            cleanup();
            return fail("failed to create GGUF tensor data context");
        }

        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * meta_tensor = ggml_get_tensor(meta, name);
            ggml_tensor * cur = ggml_dup_tensor(ctx_data, meta_tensor);
            ggml_set_name(cur, name);
        }

        model_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, model_buft);
        if (!model_buffer) {
            cleanup();
            return fail("failed to allocate GGUF model backend buffer");
        }

        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            cleanup();
            return fail(std::string("failed to open GGUF for tensor loading: ") + path);
        }

        std::vector<uint8_t> read_buf;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
            const size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                cleanup();
                return fail(std::string("failed to seek tensor: ") + name);
            }

            const size_t nbytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(model_buffer)) {
                fin.read(reinterpret_cast<char *>(cur->data), nbytes);
            } else {
                read_buf.resize(nbytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), nbytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, nbytes);
            }
            if (!fin) {
                cleanup();
                return fail(std::string("failed to read tensor: ") + name);
            }
        }
        fin.close();

        if (meta) {
            ggml_free(meta);
            meta = nullptr;
        }

        if (!bind_tensors(ctx_data)) {
            cleanup();
            return false;
        }
    } catch (const std::exception & e) {
        cleanup();
        return fail(e.what());
    }

    out.gguf = gguf;
    out.ctx_data = ctx_data;
    out.model_buffer = model_buffer;
    gguf = nullptr;
    ctx_data = nullptr;
    model_buffer = nullptr;
    cleanup();
    return true;
}

uint32_t gguf_loader::require_u32(gguf_context * gguf, const char * key) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0) {
        throw std::runtime_error(std::string("missing required GGUF metadata key: ") + key);
    }
    return gguf_get_val_u32(gguf, idx);
}

uint32_t gguf_loader::u32_or(gguf_context * gguf, const char * key, uint32_t fallback) {
    const int idx = gguf_find_key(gguf, key);
    return idx >= 0 ? gguf_get_val_u32(gguf, idx) : fallback;
}

float gguf_loader::f32_or(gguf_context * gguf, const char * key, float fallback) {
    const int idx = gguf_find_key(gguf, key);
    return idx >= 0 ? gguf_get_val_f32(gguf, idx) : fallback;
}

void gguf_loader::f32_arr3_or(gguf_context * gguf, const char * key, float out[3], const float fallback[3]) {
    const int n = arr_n(gguf, key);
    const float * data = static_cast<const float *>(arr_data(gguf, key));
    if (n < 3 || !data) {
        out[0] = fallback[0];
        out[1] = fallback[1];
        out[2] = fallback[2];
        return;
    }

    out[0] = data[0];
    out[1] = data[1];
    out[2] = data[2];
}

int gguf_loader::arr_n(gguf_context * gguf, const char * key) const {
    const int idx = gguf_find_key(gguf, key);
    return idx >= 0 ? gguf_get_arr_n(gguf, idx) : -1;
}

const void * gguf_loader::arr_data(gguf_context * gguf, const char * key) const {
    const int idx = gguf_find_key(gguf, key);
    return idx >= 0 ? gguf_get_arr_data(gguf, idx) : nullptr;
}

ggml_tensor * gguf_loader::require_tensor(ggml_context * ctx_data, const std::string & name) {
    ggml_tensor * tensor = ggml_get_tensor(ctx_data, name.c_str());
    if (!tensor) {
        throw std::runtime_error("tensor not found: " + name);
    }
    return tensor;
}

std::vector<float> gguf_loader::read_f32_tensor(ggml_context * ctx_data, const std::string & name) {
    ggml_tensor * tensor = require_tensor(ctx_data, name);
    const int64_t n = ggml_nelements(tensor);
    std::vector<float> out(n);
    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(tensor, raw.data(), 0, nbytes);

    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), raw.data(), static_cast<size_t>(n) * sizeof(float));
        return out;
    }
    if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
        return out;
    }
    if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(raw.data()), out.data(), n);
        return out;
    }

    throw std::runtime_error("unsupported tensor type for f32 read: " + name);
}

const std::string & gguf_loader::error() const {
    return error_;
}

void gguf_loader::set_error(const std::string & error) {
    error_ = error;
    std::fprintf(stderr, "%s: %s\n", __func__, error_.c_str());
}

bool gguf_loader::fail(const std::string & error) {
    set_error(error);
    return false;
}
