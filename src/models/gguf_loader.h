#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_load_result {
    gguf_context * gguf = nullptr;
    ggml_context * ctx_data = nullptr;
    ggml_backend_buffer_t model_buffer = nullptr;
};

class gguf_loader {
public:
    virtual ~gguf_loader() = default;

    bool load(
        const char * path,
        ggml_backend_buffer_type_t model_buft,
        gguf_load_result & out,
        int verbosity);

    const std::string & error() const;

protected:
    virtual bool parse_metadata(gguf_context * gguf) = 0;
    virtual bool bind_tensors(ggml_context * ctx_data) = 0;
    virtual bool validate_model() = 0;

    uint32_t require_u32(gguf_context * gguf, const char * key);
    uint32_t u32_or(gguf_context * gguf, const char * key, uint32_t fallback);
    float f32_or(gguf_context * gguf, const char * key, float fallback);

    int arr_n(gguf_context * gguf, const char * key) const;
    const void * arr_data(gguf_context * gguf, const char * key) const;

    ggml_tensor * require_tensor(ggml_context * ctx_data, const std::string & name);
    std::vector<float> read_f32_tensor(ggml_context * ctx_data, const std::string & name);

    void set_error(const std::string & error);

private:
    bool fail(const std::string & error);

    std::string error_;
};
