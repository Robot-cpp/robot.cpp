#pragma once

#include "core/types.h"
#include "vlacpp.h"

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace vlacpp {

constexpr uint32_t kGgufTypeUint32 = 4;
constexpr uint32_t kGgufTypeInt32 = 5;
constexpr uint32_t kGgufTypeFloat32 = 6;
constexpr uint32_t kGgufTypeBool = 7;
constexpr uint32_t kGgufTypeString = 8;
constexpr uint32_t kGgufTypeArray = 9;
constexpr uint32_t kGgufTypeUint64 = 10;
constexpr uint32_t kGgufTypeInt64 = 11;
constexpr uint32_t kGgufTypeFloat64 = 12;

class GgufMetadataReader {
public:
    explicit GgufMetadataReader(std::istream & input) : input_(input) {}

    template <typename T>
    bool read_scalar(T & out) {
        input_.read(reinterpret_cast<char *>(&out), sizeof(T));
        return static_cast<bool>(input_);
    }

    bool read_string(std::string & out);
    bool read_raw(char * data, std::streamsize size);

private:
    std::istream & input_;
};

using GgufTensorLoadFilter = bool (*)(const ModelConfig & config, const std::string & name);
using GgufMetadataHandler = bool (*)(
    GgufMetadataReader & reader,
    const std::string & key,
    uint32_t type,
    ModelConfig & config);

bool skip_gguf_metadata_value(GgufMetadataReader & reader, uint32_t type);
bool read_gguf_float_array(GgufMetadataReader & reader, std::vector<float> & out);
bool read_gguf_string_array(GgufMetadataReader & reader, std::vector<std::string> & out);
bool read_gguf_component_metadata(
    GgufMetadataReader & reader,
    const std::string & key,
    uint32_t type,
    const char * prefix,
    VLAComponentConfig & component);

vlacpp_status load_gguf_model_file(
    const std::string & path,
    ModelConfig & out_config,
    TensorMap & out_tensors,
    GgufTensorLoadFilter filter = nullptr,
    GgufMetadataHandler metadata_handler = nullptr);

bool read_gguf_tensor_f32(
    const std::string & path,
    const std::string & tensor_name,
    std::vector<float> & out);

} // namespace vlacpp
