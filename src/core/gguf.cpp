#include "core/gguf.h"

#include "core/error.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace vlacpp {
namespace {

constexpr uint32_t kGgufVersion = 3;
constexpr uint32_t kGgufTypeUint32 = 4;
constexpr uint32_t kGgufTypeInt32 = 5;
constexpr uint32_t kGgufTypeFloat32 = 6;
constexpr uint32_t kGgufTypeBool = 7;
constexpr uint32_t kGgufTypeString = 8;
constexpr uint32_t kGgufTypeArray = 9;
constexpr uint32_t kGgufTypeUint64 = 10;
constexpr uint32_t kGgufTypeInt64 = 11;
constexpr uint32_t kGgufTypeFloat64 = 12;
constexpr uint32_t kGgmlTypeF32 = 0;
constexpr uint64_t kTensorAlignment = 32;

class Reader {
public:
    explicit Reader(std::ifstream & file) : file_(file) {}

    template <typename T>
    bool read_scalar(T & out) {
        file_.read(reinterpret_cast<char *>(&out), sizeof(T));
        return static_cast<bool>(file_);
    }

    bool read_string(std::string & out) {
        uint64_t size = 0;
        if (!read_scalar(size) || size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        out.resize(static_cast<size_t>(size));
        if (size == 0) {
            return true;
        }
        file_.read(&out[0], static_cast<std::streamsize>(size));
        return static_cast<bool>(file_);
    }

    std::ifstream & file() {
        return file_;
    }

private:
    std::ifstream & file_;
};

bool read_float_array(Reader & reader, std::vector<float> & out) {
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
    reader.file().read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(reader.file());
}

bool read_string_array(Reader & reader, std::vector<std::string> & out) {
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

bool skip_value(Reader & reader, uint32_t type) {
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
                if (!skip_value(reader, elem_type)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool set_metadata_value(Reader & reader, const std::string & key, uint32_t type, ModelConfig & config) {
    if (key == "general.architecture" || key == "vlacpp.model_type") {
        if (type != kGgufTypeString) {
            return false;
        }
        return reader.read_string(config.model_type);
    }
    if (key == "vlacpp.image_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.image_width);
    }
    if (key == "vlacpp.image_height" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.image_height);
    }
    if (key == "vlacpp.state_dim" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.state_dim);
    }
    if (key == "vlacpp.action_dim" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.action_dim);
    }
    if (key == "vlacpp.action_horizon" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.action_horizon);
    }
    if (key == "vlacpp.max_token_len" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.max_token_len);
    }
    if (key == "vlacpp.openpi.action_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_action_width);
    }
    if (key == "vlacpp.openpi.vision_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_vision_width);
    }
    if (key == "vlacpp.openpi.vision_patch_height" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_vision_patch_height);
    }
    if (key == "vlacpp.openpi.vision_patch_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_vision_patch_width);
    }
    if (key == "vlacpp.openpi.vision_layers" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_vision_layers);
    }
    if (key == "vlacpp.openpi.language_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_language_width);
    }
    if (key == "vlacpp.openpi.language_q_out" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_language_q_out);
    }
    if (key == "vlacpp.openpi.language_kv_out" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_language_kv_out);
    }
    if (key == "vlacpp.openpi.language_mlp_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_language_mlp_width);
    }
    if (key == "vlacpp.openpi.language_layers" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_language_layers);
    }
    if (key == "vlacpp.openpi.action_expert_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_action_expert_width);
    }
    if (key == "vlacpp.openpi.action_expert_q_out" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_action_expert_q_out);
    }
    if (key == "vlacpp.openpi.action_expert_kv_out" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_action_expert_kv_out);
    }
    if (key == "vlacpp.openpi.action_expert_mlp_width" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_action_expert_mlp_width);
    }
    if (key == "vlacpp.openpi.action_expert_layers" && type == kGgufTypeInt32) {
        return reader.read_scalar(config.openpi_action_expert_layers);
    }
    if (key == "vlacpp.image_keys" && type == kGgufTypeArray) {
        return read_string_array(reader, config.image_keys);
    }
    if (key == "vlacpp.state_mean" && type == kGgufTypeArray) {
        return read_float_array(reader, config.state_mean);
    }
    if (key == "vlacpp.state_std" && type == kGgufTypeArray) {
        return read_float_array(reader, config.state_std);
    }
    if (key == "vlacpp.action_mean" && type == kGgufTypeArray) {
        return read_float_array(reader, config.action_mean);
    }
    if (key == "vlacpp.action_std" && type == kGgufTypeArray) {
        return read_float_array(reader, config.action_std);
    }
    return skip_value(reader, type);
}

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
};

uint64_t align_to(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

vlacpp_status validate_config(const ModelConfig & config) {
    if (config.action_dim <= 0 || config.action_horizon <= 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "model config requires positive action_dim and action_horizon");
    }
    if (config.state_dim < 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "state_dim must be non-negative");
    }
    if ((!config.state_mean.empty() || !config.state_std.empty()) &&
        (config.state_mean.size() != static_cast<size_t>(config.state_dim) ||
            config.state_std.size() != static_cast<size_t>(config.state_dim))) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "state_mean and state_std must both match state_dim");
    }
    if ((!config.action_mean.empty() || !config.action_std.empty()) &&
        (config.action_mean.size() != static_cast<size_t>(config.action_dim) ||
            config.action_std.size() != static_cast<size_t>(config.action_dim))) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "action_mean and action_std must both match action_dim");
    }
    return VLACPP_STATUS_OK;
}

} // namespace

vlacpp_status load_gguf_model_file(
    const std::string & path,
    ModelConfig & out_config,
    TensorMap & out_tensors) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(VLACPP_STATUS_IO_ERROR, "failed to open model file: " + path);
    }
    Reader reader(file);

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
        if (!reader.read_string(key) || !reader.read_scalar(type) || !set_metadata_value(reader, key, type, config)) {
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
        if (!reader.read_scalar(info.type) || !reader.read_scalar(info.offset) || info.type != kGgmlTypeF32) {
            return fail(VLACPP_STATUS_UNSUPPORTED, "only F32 GGUF tensors are supported");
        }
        infos.push_back(std::move(info));
    }

    const uint64_t data_start = align_to(static_cast<uint64_t>(file.tellg()), kTensorAlignment);
    out_tensors.clear();
    for (const TensorInfo & info : infos) {
        uint64_t element_count = 1;
        for (int64_t dim : info.shape) {
            element_count *= static_cast<uint64_t>(dim);
        }
        Tensor tensor;
        tensor.shape = info.shape;
        tensor.data.resize(static_cast<size_t>(element_count));
        file.seekg(static_cast<std::streamoff>(data_start + info.offset), std::ios::beg);
        file.read(
            reinterpret_cast<char *>(tensor.data.data()),
            static_cast<std::streamsize>(element_count * sizeof(float)));
        if (!file) {
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
    Reader reader(file);

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
        if (!reader.read_string(key) || !reader.read_scalar(type) || !skip_value(reader, type)) {
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
        if (info.name != tensor_name || info.type != kGgmlTypeF32) {
            continue;
        }
        uint64_t element_count = 1;
        for (int64_t dim : info.shape) {
            if (dim <= 0 ||
                element_count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
                return false;
            }
            element_count *= static_cast<uint64_t>(dim);
        }
        if (element_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            return false;
        }
        out.resize(static_cast<size_t>(element_count));
        file.seekg(static_cast<std::streamoff>(data_start + info.offset), std::ios::beg);
        file.read(
            reinterpret_cast<char *>(out.data()),
            static_cast<std::streamsize>(element_count * sizeof(float)));
        return static_cast<bool>(file);
    }
    return false;
}

} // namespace vlacpp
