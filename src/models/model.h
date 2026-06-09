#pragma once

#include "core/types.h"
#include "vlacpp.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robotcpp {

enum class model_type {
    smolvla,
};

struct model_image {
    std::string name;
    const uint8_t * data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int stride_bytes = 0;
};

struct observation {
    std::vector<model_image> images;
    std::vector<float> state;
    std::string task;
};

struct model_metric {
    std::string name;
    double value = 0.0;
};

struct model_result {
    std::vector<float> actions;
    int chunk_size = 0;
    int action_dim = 0;
    std::vector<model_metric> metrics;
};

struct model_runtime_options {
    int threads = 0;
    int verbosity = 1;
};

struct model_options {
    model_type type = model_type::smolvla;
    model_runtime_options runtime;
    std::any config;
};

class Model {
public:
    virtual ~Model() = default;
    virtual const char * type() const = 0;
    virtual bool predict(
        const observation & obs,
        model_result & out,
        std::string & error) = 0;
    virtual void reset() {}
};

bool make_model(
    const model_options & options,
    std::unique_ptr<Model> & out,
    std::string & error);

} // namespace robotcpp

namespace vlacpp {

vlacpp_status load_model_from_path(
    const std::string & path,
    const BackendConfig & backend,
    std::unique_ptr<RuntimeModel> & out);

} // namespace vlacpp
