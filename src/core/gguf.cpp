#include "core/gguf.h"

#include "core/error.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace vlacpp {
namespace {

constexpr uint32_t kGgufVersion = 3;
constexpr uint64_t kTensorAlignment = 32;

} // namespace

bool GgufMetadataReader::read_string(std::string & out) {
    uint64_t size = 0;
    if (!read_scalar(size) || size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    out.resize(static_cast<size_t>(size));
    if (size == 0) {
        return true;
    }
    input_.read(&out[0], static_cast<std::streamsize>(size));
    return static_cast<bool>(input_);
}

bool GgufMetadataReader::read_raw(char * data, std::streamsize size) {
    input_.read(data, size);
    return static_cast<bool>(input_);
}

bool read_gguf_float_array(GgufMetadataReader & reader, std::vector<float> & out) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!reader.read_scalar(elem_type) || !reader.read_scalar(count) || elem_type != kGgufTypeFloat32) {
        return false;
    }
    if (count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    out.resize(static_cast<size_t>(count));
    if (count == 0) {
        return true;
    }
    return reader.read_raw(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
}

bool read_gguf_string_array(GgufMetadataReader & reader, std::vector<std::string> & out) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!reader.read_scalar(elem_type) || !reader.read_scalar(count) || elem_type != kGgufTypeString) {
        return false;
    }
    if (count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    out.clear();
    out.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        std::string value;
        if (!reader.read_string(value)) {
            return false;
        }
        out.push_back(std::move(value));
    }
    return true;
}

namespace {

bool read_component_backend_name(GgufMetadataReader & reader, std::string & out) {
    std::string value;
    if (!reader.read_string(value)) {
        return false;
    }
    if (value == "inherit" || value == "cpu" || value == "cuda") {
        out = value;
        return true;
    }
    return false;
}

} // namespace

bool skip_gguf_metadata_value(GgufMetadataReader & reader, uint32_t type) {
    switch (type) {
        case 0:
        case 1: {
            uint8_t ignored = 0;
            return reader.read_scalar(ignored);
        }
        case 2:
        case 3: {
            uint16_t ignored = 0;
            return reader.read_scalar(ignored);
        }
        case kGgufTypeUint32:
        case kGgufTypeInt32:
        case kGgufTypeFloat32: {
            uint32_t ignored = 0;
            return reader.read_scalar(ignored);
        }
        case kGgufTypeBool: {
            bool ignored = false;
            return reader.read_scalar(ignored);
        }
        case kGgufTypeUint64:
        case kGgufTypeInt64:
        case kGgufTypeFloat64: {
            uint64_t ignored = 0;
            return reader.read_scalar(ignored);
        }
        case kGgufTypeString: {
            std::string ignored;
            return reader.read_string(ignored);
        }
        case kGgufTypeArray: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!reader.read_scalar(elem_type) || !reader.read_scalar(count)) {
                return false;
            }
            for (uint64_t i = 0; i < count; ++i) {
                if (!skip_gguf_metadata_value(reader, elem_type)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool read_gguf_component_metadata(
    GgufMetadataReader & reader,
    const std::string & key,
    uint32_t type,
    const char * prefix,
    VLAComponentConfig & component) {
    const std::string key_prefix(prefix);
    if (key.rfind(key_prefix, 0) != 0) {
        return false;
    }
    const std::string suffix = key.substr(key_prefix.size());
    if (suffix == "architecture" && type == kGgufTypeString) {
        return reader.read_string(component.architecture);
    }
    if (suffix == "prefix" && type == kGgufTypeString) {
        return reader.read_string(component.tensor_prefix);
    }
    if (suffix == "backend" && type == kGgufTypeString) {
        return read_component_backend_name(reader, component.runtime.backend);
    }
    if (suffix == "dtype" && type == kGgufTypeString) {
        return reader.read_string(component.runtime.data_type);
    }
    if (suffix == "n_threads" && type == kGgufTypeInt32) {
        return reader.read_scalar(component.runtime.n_threads);
    }
    return skip_gguf_metadata_value(reader, type);
}

namespace {

bool read_model_type(GgufMetadataReader & reader, ModelConfig & config) {
    std::string model_type;
    if (!reader.read_string(model_type)) {
        return false;
    }
    config.common.model_type = model_type;
    return true;
}

bool set_metadata_value(
    GgufMetadataReader & reader,
    const std::string & key,
    uint32_t type,
    ModelConfig & config,
    GgufMetadataHandler metadata_handler) {
    if (key == "vlacpp.model_type") {
        if (type != kGgufTypeString) {
            return false;
        }
        return read_model_type(reader, config);
    }
    if (key == "vlacpp.component.role") {
        if (type != kGgufTypeString) {
            return false;
        }
        return reader.read_string(config.component_role);
    }
    if (key == "vlacpp.image_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.common.image_width);
    }
    if (key == "vlacpp.image_height" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.common.image_height);
    }
    if (key == "vlacpp.state_dim" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.common.state_dim);
    }
    if (key == "vlacpp.action_dim" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.common.action_dim);
    }
    if (key == "vlacpp.action_horizon" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.common.action_horizon);
    }
    if (key == "vlacpp.max_token_len" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.common.max_token_len);
    }
    if (key == "vlacpp.image_keys" && type == kGgufTypeArray) {
        return read_gguf_string_array(reader, config.common.image_keys);
    }
    if (key == "vlacpp.state_mean" && type == kGgufTypeArray) {
        return read_gguf_float_array(reader, config.common.state_mean);
    }
    if (key == "vlacpp.state_std" && type == kGgufTypeArray) {
        return read_gguf_float_array(reader, config.common.state_std);
    }
    if (key == "vlacpp.action_mean" && type == kGgufTypeArray) {
        return read_gguf_float_array(reader, config.common.action_mean);
    }
    if (key == "vlacpp.action_std" && type == kGgufTypeArray) {
        return read_gguf_float_array(reader, config.common.action_std);
    }
    if (metadata_handler != nullptr) {
        return metadata_handler(reader, key, type, config);
    }
    return skip_gguf_metadata_value(reader, type);
}

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
};

bool is_supported_tensor_type(uint32_t type) {
    return type == static_cast<uint32_t>(GGML_TYPE_F32) ||
        type == static_cast<uint32_t>(GGML_TYPE_F16) ||
        type == static_cast<uint32_t>(GGML_TYPE_BF16);
}

const char * tensor_type_name(uint32_t type) {
    if (type == static_cast<uint32_t>(GGML_TYPE_F32)) {
        return "fp32";
    }
    if (type == static_cast<uint32_t>(GGML_TYPE_F16)) {
        return "f16";
    }
    if (type == static_cast<uint32_t>(GGML_TYPE_BF16)) {
        return "bf16";
    }
    return "unsupported";
}

uint64_t align_to(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

bool tensor_element_count(const TensorInfo & info, uint64_t & out) {
    out = 1;
    for (int64_t dim : info.shape) {
        if (dim <= 0 ||
            out > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            return false;
        }
        out *= static_cast<uint64_t>(dim);
    }
    return true;
}

bool read_tensor_data(std::ifstream & file, const TensorInfo & info, Tensor & tensor) {
    uint64_t element_count = 0;
    if (!tensor_element_count(info, element_count) ||
        element_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    tensor.shape = info.shape;
    tensor.data_type = tensor_type_name(info.type);
    tensor.data.resize(static_cast<size_t>(element_count));
    if (element_count == 0) {
        return true;
    }
    if (info.type == static_cast<uint32_t>(GGML_TYPE_F32)) {
        file.read(
            reinterpret_cast<char *>(tensor.data.data()),
            static_cast<std::streamsize>(element_count * sizeof(float)));
        return static_cast<bool>(file);
    }
    if (info.type == static_cast<uint32_t>(GGML_TYPE_F16)) {
        std::vector<ggml_fp16_t> raw(static_cast<size_t>(element_count));
        file.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(element_count * sizeof(ggml_fp16_t)));
        if (!file) {
            return false;
        }
        ggml_fp16_to_fp32_row(raw.data(), tensor.data.data(), static_cast<int64_t>(element_count));
        return true;
    }
    if (info.type == static_cast<uint32_t>(GGML_TYPE_BF16)) {
        std::vector<ggml_bf16_t> raw(static_cast<size_t>(element_count));
        file.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(element_count * sizeof(ggml_bf16_t)));
        if (!file) {
            return false;
        }
        ggml_bf16_to_fp32_row(raw.data(), tensor.data.data(), static_cast<int64_t>(element_count));
        return true;
    }
    return false;
}

vlacpp_status validate_config(const ModelConfig & config) {
    if (config.common.action_dim <= 0 || config.common.action_horizon <= 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "model config requires positive action_dim and action_horizon");
    }
    if (config.common.state_dim < 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "state_dim must be non-negative");
    }
    if ((!config.common.state_mean.empty() || !config.common.state_std.empty()) &&
        (config.common.state_mean.size() != static_cast<size_t>(config.common.state_dim) ||
            config.common.state_std.size() != static_cast<size_t>(config.common.state_dim))) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "state_mean and state_std must both match state_dim");
    }
    if ((!config.common.action_mean.empty() || !config.common.action_std.empty()) &&
        (config.common.action_mean.size() != static_cast<size_t>(config.common.action_dim) ||
            config.common.action_std.size() != static_cast<size_t>(config.common.action_dim))) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "action_mean and action_std must both match action_dim");
    }
    return VLACPP_STATUS_OK;
}

} // namespace

vlacpp_status load_gguf_model_file(
    const std::string & path,
    ModelConfig & out_config,
    TensorMap & out_tensors,
    GgufTensorLoadFilter filter,
    GgufMetadataHandler metadata_handler) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(VLACPP_STATUS_IO_ERROR, "failed to open model file: " + path);
    }
    GgufMetadataReader reader(file);

    char magic[4] = {};
    file.read(magic, sizeof(magic));
    if (!file || std::memcmp(magic, "GGUF", 4) != 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "model file is not GGUF");
    }
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    if (!reader.read_scalar(version) || !reader.read_scalar(tensor_count) || !reader.read_scalar(metadata_count)) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "truncated GGUF header");
    }
    if (version != kGgufVersion) {
        return fail(VLACPP_STATUS_UNSUPPORTED, "unsupported GGUF version");
    }

    ModelConfig config;
    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!reader.read_string(key) ||
            !reader.read_scalar(type) ||
            !set_metadata_value(reader, key, type, config, metadata_handler)) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "invalid GGUF metadata");
        }
    }

    std::vector<TensorInfo> infos;
    infos.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        uint32_t n_dims = 0;
        if (!reader.read_string(info.name) || !reader.read_scalar(n_dims) || n_dims > 4) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "invalid GGUF tensor info");
        }
        info.shape.resize(n_dims);
        for (uint32_t dim = 0; dim < n_dims; ++dim) {
            uint64_t size = 0;
            if (!reader.read_scalar(size) || size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                return fail(VLACPP_STATUS_PARSE_ERROR, "invalid GGUF tensor dimension");
            }
            info.shape[dim] = static_cast<int64_t>(size);
        }
        if (!reader.read_scalar(info.type) || !reader.read_scalar(info.offset) || !is_supported_tensor_type(info.type)) {
            return fail(VLACPP_STATUS_UNSUPPORTED, "only F32, F16, and BF16 GGUF tensors are supported");
        }
        infos.push_back(std::move(info));
    }

    const uint64_t data_start = align_to(static_cast<uint64_t>(file.tellg()), kTensorAlignment);
    out_tensors.clear();
    for (const TensorInfo & info : infos) {
        if (filter != nullptr && !filter(config, info.name)) {
            continue;
        }
        Tensor tensor;
        file.seekg(static_cast<std::streamoff>(data_start + info.offset), std::ios::beg);
        if (!read_tensor_data(file, info, tensor)) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "truncated GGUF tensor data");
        }
        out_tensors[info.name] = std::move(tensor);
    }

    vlacpp_status status = validate_config(config);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    out_config = std::move(config);
    return VLACPP_STATUS_OK;
}

bool read_gguf_tensor_f32(
    const std::string & path,
    const std::string & tensor_name,
    std::vector<float> & out) {
    out.clear();
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    GgufMetadataReader reader(file);

    char magic[4] = {};
    file.read(magic, sizeof(magic));
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    if (!file ||
        std::memcmp(magic, "GGUF", 4) != 0 ||
        !reader.read_scalar(version) ||
        !reader.read_scalar(tensor_count) ||
        !reader.read_scalar(metadata_count) ||
        version != kGgufVersion) {
        return false;
    }

    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!reader.read_string(key) || !reader.read_scalar(type) || !skip_gguf_metadata_value(reader, type)) {
            return false;
        }
    }

    std::vector<TensorInfo> infos;
    infos.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        uint32_t n_dims = 0;
        if (!reader.read_string(info.name) || !reader.read_scalar(n_dims) || n_dims > 4) {
            return false;
        }
        info.shape.resize(n_dims);
        for (uint32_t dim = 0; dim < n_dims; ++dim) {
            uint64_t size = 0;
            if (!reader.read_scalar(size) || size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                return false;
            }
            info.shape[dim] = static_cast<int64_t>(size);
        }
        if (!reader.read_scalar(info.type) || !reader.read_scalar(info.offset)) {
            return false;
        }
        infos.push_back(std::move(info));
    }

    const uint64_t data_start = align_to(static_cast<uint64_t>(file.tellg()), kTensorAlignment);
    for (const TensorInfo & info : infos) {
        if (info.name != tensor_name || !is_supported_tensor_type(info.type)) {
            continue;
        }
        Tensor tensor;
        file.seekg(static_cast<std::streamoff>(data_start + info.offset), std::ios::beg);
        if (!read_tensor_data(file, info, tensor)) {
            return false;
        }
        out = std::move(tensor.data);
        return true;
    }
    return false;
}

} // namespace vlacpp
