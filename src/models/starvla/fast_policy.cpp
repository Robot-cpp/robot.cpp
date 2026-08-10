#include "models/starvla/fast_policy.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::starvla {
namespace {

constexpr const char * kArchitecture = "starvla-policy";
constexpr const char * kActionMapTensor =
    "starvla.policy.fast.action_token_map";
constexpr const char * kOffsetsTensor =
    "starvla.policy.fast.codec.token_offsets";
constexpr const char * kTokenBytesTensor =
    "starvla.policy.fast.codec.token_bytes";

int require_key(gguf_context * gguf, const char * key, gguf_type type) {
    const int index = gguf_find_key(gguf, key);
    if (index < 0) {
        throw std::runtime_error(std::string("missing required FAST GGUF metadata: ") +
                                 key);
    }
    if (gguf_get_kv_type(gguf, index) != type) {
        throw std::runtime_error(std::string("invalid FAST GGUF metadata type: ") +
                                 key);
    }
    return index;
}

std::string require_string(gguf_context * gguf, const char * key) {
    return gguf_get_val_str(gguf, require_key(gguf, key, GGUF_TYPE_STRING));
}

int32_t require_i32(gguf_context * gguf, const char * key) {
    return gguf_get_val_i32(gguf, require_key(gguf, key, GGUF_TYPE_INT32));
}

float require_f32(gguf_context * gguf, const char * key) {
    return gguf_get_val_f32(gguf, require_key(gguf, key, GGUF_TYPE_FLOAT32));
}

bool require_bool(gguf_context * gguf, const char * key) {
    return gguf_get_val_bool(gguf, require_key(gguf, key, GGUF_TYPE_BOOL));
}

int require_array(gguf_context * gguf, const char * key, gguf_type type) {
    const int index = require_key(gguf, key, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(gguf, index) != type) {
        throw std::runtime_error(
            std::string("invalid FAST GGUF array element type: ") + key);
    }
    return index;
}

std::vector<int32_t> require_i32_array(gguf_context * gguf, const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_INT32);
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto * data =
        static_cast<const int32_t *>(gguf_get_arr_data(gguf, index));
    if (count != 0 && data == nullptr) {
        throw std::runtime_error(std::string("missing FAST GGUF array data: ") +
                                 key);
    }
    return std::vector<int32_t>(data, data + count);
}

std::vector<float> require_f32_array(gguf_context * gguf, const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_FLOAT32);
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto * data =
        static_cast<const float *>(gguf_get_arr_data(gguf, index));
    if (count != 0 && data == nullptr) {
        throw std::runtime_error(std::string("missing FAST GGUF array data: ") +
                                 key);
    }
    return std::vector<float>(data, data + count);
}

std::vector<uint8_t> require_bool_array(gguf_context * gguf,
                                        const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_BOOL);
    const size_t count = gguf_get_arr_n(gguf, index);
    const auto * data =
        static_cast<const uint8_t *>(gguf_get_arr_data(gguf, index));
    if (count != 0 && data == nullptr) {
        throw std::runtime_error(std::string("missing FAST GGUF array data: ") +
                                 key);
    }
    std::vector<uint8_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = data[i] != 0 ? uint8_t{1} : uint8_t{0};
    }
    return values;
}

std::vector<std::string> require_string_array(gguf_context * gguf,
                                              const char * key) {
    const int index = require_array(gguf, key, GGUF_TYPE_STRING);
    const size_t count = gguf_get_arr_n(gguf, index);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        values.emplace_back(gguf_get_arr_str(gguf, index, i));
    }
    return values;
}

std::string profile_key(int index, const char * suffix) {
    return "starvla.normalization.profile." + std::to_string(index) + "." +
           suffix;
}

struct FastRuntimeMetadata {
    FastCodecConfig codec;
    int token_bytes_count = 0;
};

FastRuntimeMetadata parse_metadata(gguf_context * gguf,
                                   FastPolicyConfig & config) {
    if (require_string(gguf, "general.architecture") != kArchitecture ||
        require_i32(gguf, "starvla.schema_version") != 1 ||
        require_string(gguf, "starvla.framework") != "fast") {
        throw std::runtime_error("GGUF is not a supported StarVLA FAST policy");
    }
    config.backbone_arch = require_string(gguf, "starvla.backbone.arch");
    config.bundle_uuid = require_string(gguf, "starvla.bundle.uuid");
    config.text_filename = require_string(gguf, "starvla.component.text.filename");
    config.mmproj_filename = require_string(gguf, "starvla.component.mmproj.filename");
    if (config.backbone_arch != "qwen2_5_vl" || config.bundle_uuid.empty() ||
        config.text_filename.empty() || config.mmproj_filename.empty()) {
        throw std::runtime_error("StarVLA FAST bundle metadata is incomplete");
    }

    config.qwen_hidden_dim = require_i32(gguf, "starvla.qwen.hidden_size");
    config.qwen_input_embedding_dim =
        require_i32(gguf, "starvla.qwen.input_embedding_size");
    config.qwen_vocab_size = require_i32(gguf, "starvla.qwen.vocab_size");
    config.qwen_layer_count = require_i32(gguf, "starvla.qwen.layer_count");
    config.cot_template = require_string(gguf, "starvla.prompt.cot_template");

    config.action_dim = require_i32(gguf, "starvla.action.dimension");
    config.horizon = require_i32(gguf, "starvla.action.horizon");
    config.image_count = require_i32(gguf, "starvla.image.count");
    config.image_names = require_string_array(gguf, "starvla.image.names");
    config.image_processor_min_pixels =
        require_i32(gguf, "starvla.image.processor_min_pixels");
    config.image_processor_max_pixels =
        require_i32(gguf, "starvla.image.processor_max_pixels");
    config.image_patch_size = require_i32(gguf, "starvla.image.patch_size");
    config.image_spatial_merge_size =
        require_i32(gguf, "starvla.image.spatial_merge_size");
    config.image_min_token_count = require_i32(gguf, "starvla.image.min_token_count");
    config.image_max_token_count = require_i32(gguf, "starvla.image.max_token_count");

    const int max_length = require_i32(gguf, "starvla.fast.generation.max_length");
    config.generation_eos_token_ids =
        require_i32_array(gguf, "starvla.fast.generation.eos_token_ids");
    config.generation_top_k = require_i32(gguf, "starvla.fast.generation.top_k");
    config.generation_repetition_penalty =
        require_f32(gguf, "starvla.fast.generation.repetition_penalty");

    FastRuntimeMetadata runtime;
    runtime.codec.scale = require_f32(gguf, "starvla.fast.codec.scale");
    runtime.codec.min_token = require_i32(gguf, "starvla.fast.codec.min_token");
    runtime.codec.vocab_size =
        static_cast<size_t>(require_i32(gguf, "starvla.fast.codec.vocab_size"));
    runtime.codec.time_horizon =
        static_cast<size_t>(require_i32(gguf, "starvla.fast.codec.time_horizon"));
    runtime.codec.action_dim =
        static_cast<size_t>(require_i32(gguf, "starvla.fast.codec.action_dimension"));
    const int action_token_count =
        require_i32(gguf, "starvla.fast.action_token.count");
    const int offsets_count =
        require_i32(gguf, "starvla.fast.codec.token_offsets_count");
    runtime.token_bytes_count =
        require_i32(gguf, "starvla.fast.codec.token_bytes_count");

    const bool valid =
        config.qwen_hidden_dim > 0 && config.qwen_input_embedding_dim > 0 &&
        config.qwen_vocab_size > 0 && config.qwen_layer_count > 0 &&
        !config.cot_template.empty() && config.action_dim > 0 && config.horizon > 0 &&
        config.image_count > 0 &&
        config.image_names.size() == static_cast<size_t>(config.image_count) &&
        config.image_processor_min_pixels > 0 &&
        config.image_processor_max_pixels >= config.image_processor_min_pixels &&
        config.image_patch_size > 0 && config.image_spatial_merge_size > 0 &&
        config.image_min_token_count > 0 &&
        config.image_max_token_count >= config.image_min_token_count &&
        max_length > 0 && !config.generation_eos_token_ids.empty() &&
        config.generation_top_k > 0 &&
        std::isfinite(config.generation_repetition_penalty) &&
        config.generation_repetition_penalty > 0.0f &&
        runtime.codec.vocab_size > 0 &&
        action_token_count == static_cast<int>(runtime.codec.vocab_size) &&
        offsets_count == action_token_count + 1 && runtime.token_bytes_count > 0 &&
        runtime.codec.time_horizon == static_cast<size_t>(config.horizon) &&
        runtime.codec.action_dim == static_cast<size_t>(config.action_dim);
    if (!valid) {
        throw std::runtime_error("StarVLA FAST metadata has incompatible dimensions");
    }
    config.generation_max_length = static_cast<size_t>(max_length);

    NormalizationConfig & normalization = config.normalization;
    normalization.clip_actions = require_bool(gguf, "starvla.normalization.clip_actions");
    normalization.binary_threshold =
        require_f32(gguf, "starvla.normalization.binary_threshold");
    normalization.binary_comparison =
        require_string(gguf, "starvla.normalization.binary_comparison");
    normalization.continuous_dimensions =
        require_i32_array(gguf, "starvla.action.continuous_dimensions");
    normalization.binary_dimensions =
        require_i32_array(gguf, "starvla.action.binary_dimensions");
    const int profile_count = require_i32(gguf, "starvla.normalization.profile_count");
    const std::vector<std::string> profile_keys =
        require_string_array(gguf, "starvla.normalization.profile_keys");
    if (profile_count <= 0 || profile_keys.size() != static_cast<size_t>(profile_count)) {
        throw std::runtime_error("StarVLA FAST normalization profiles are inconsistent");
    }
    normalization.profiles.clear();
    normalization.profiles.reserve(static_cast<size_t>(profile_count));
    for (int index = 0; index < profile_count; ++index) {
        NormalizationProfile profile;
        profile.key = require_string(gguf, profile_key(index, "key").c_str());
        profile.action_q01 =
            require_f32_array(gguf, profile_key(index, "action_q01").c_str());
        profile.action_q99 =
            require_f32_array(gguf, profile_key(index, "action_q99").c_str());
        profile.action_mask =
            require_bool_array(gguf, profile_key(index, "action_mask").c_str());
        if (profile.key != profile_keys[static_cast<size_t>(index)]) {
            throw std::runtime_error("StarVLA FAST normalization profile order is inconsistent");
        }
        normalization.profiles.push_back(std::move(profile));
    }
    std::string normalization_error;
    if (!validate_normalization_config(normalization, config.action_dim,
                                       normalization_error)) {
        throw std::runtime_error(normalization_error);
    }
    return runtime;
}

struct RawTensor {
    ggml_tensor * metadata = nullptr;
    int index = -1;
    std::vector<uint8_t> bytes;
};

RawTensor read_tensor(const std::string & path, gguf_context * gguf,
                      ggml_context * metadata_context, const char * name,
                      ggml_type expected_type, int64_t expected_elements) {
    RawTensor result;
    result.metadata = ggml_get_tensor(metadata_context, name);
    result.index = gguf_find_tensor(gguf, name);
    if (result.metadata == nullptr || result.index < 0 ||
        result.metadata->type != expected_type ||
        ggml_n_dims(result.metadata) != 1 ||
        result.metadata->ne[0] != expected_elements ||
        ggml_nelements(result.metadata) != expected_elements) {
        throw std::runtime_error(std::string("FAST runtime tensor shape/type mismatch: ") +
                                 name);
    }
    result.bytes.resize(ggml_nbytes(result.metadata));
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open FAST policy GGUF tensor data");
    }
    const size_t offset =
        gguf_get_data_offset(gguf) +
        gguf_get_tensor_offset(gguf, result.index);
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!stream ||
        offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error(
            std::string("failed to seek FAST runtime tensor: ") + name);
    }
    stream.read(reinterpret_cast<char *>(result.bytes.data()),
                static_cast<std::streamsize>(result.bytes.size()));
    if (!stream) {
        throw std::runtime_error(
            std::string("failed to read FAST runtime tensor: ") + name);
    }
    return result;
}

} // namespace

struct FastPolicy::Impl {
    FastPolicyConfig config;
    std::unique_ptr<FastCodec> codec;
};

FastPolicy::FastPolicy(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

FastPolicy::~FastPolicy() = default;

std::unique_ptr<FastPolicy> FastPolicy::load(const std::string & path,
                                             int verbosity,
                                             std::string & error) {
    error.clear();
    if (path.empty()) {
        error = "StarVLA FAST policy path is required";
        return nullptr;
    }

    ggml_context * metadata_context = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &metadata_context;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr || metadata_context == nullptr) {
        if (metadata_context != nullptr) {
            ggml_free(metadata_context);
        }
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        error = "failed to read StarVLA FAST policy GGUF";
        return nullptr;
    }
    auto cleanup = [&]() {
        ggml_free(metadata_context);
        metadata_context = nullptr;
        gguf_free(gguf);
        gguf = nullptr;
    };

    std::unique_ptr<Impl> impl(new Impl());
    try {
        const FastRuntimeMetadata runtime = parse_metadata(gguf, impl->config);

        RawTensor action_map =
            read_tensor(path, gguf, metadata_context, kActionMapTensor,
                        GGML_TYPE_I32, static_cast<int64_t>(runtime.codec.vocab_size));
        RawTensor offsets =
            read_tensor(path, gguf, metadata_context, kOffsetsTensor,
                        GGML_TYPE_I32,
                        static_cast<int64_t>(runtime.codec.vocab_size + 1));
        RawTensor token_bytes =
            read_tensor(path, gguf, metadata_context, kTokenBytesTensor,
                        GGML_TYPE_I8, runtime.token_bytes_count);

        const uint32_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
            throw std::runtime_error(
                "FAST runtime currently requires a little-endian host");
        }
        std::vector<int32_t> action_ids(runtime.codec.vocab_size);
        std::vector<int32_t> token_offsets(runtime.codec.vocab_size + 1);
        std::memcpy(action_ids.data(), action_map.bytes.data(),
                    action_map.bytes.size());
        std::memcpy(token_offsets.data(), offsets.bytes.data(),
                    offsets.bytes.size());
        impl->codec = FastCodec::create_compiled(
            runtime.codec, std::move(token_offsets),
            std::move(token_bytes.bytes), std::move(action_ids), error);
        if (impl->codec == nullptr) {
            throw std::runtime_error("failed to construct embedded FAST codec: " +
                                     error);
        }
        if (verbosity >= 1) {
            std::fprintf(stderr,
                         "%s: bundle=%s runtime_tensors=3 codec_vocab=%zu "
                         "generation_max_length=%zu profiles=%zu\n",
                         __func__, impl->config.bundle_uuid.c_str(),
                         runtime.codec.vocab_size, impl->config.generation_max_length,
                         impl->config.normalization.profiles.size());
        }
        cleanup();
    } catch (const std::exception & exception) {
        cleanup();
        error = exception.what();
        return nullptr;
    }
    return std::unique_ptr<FastPolicy>(new FastPolicy(std::move(impl)));
}

bool FastPolicy::decode_generated(
    const std::vector<int32_t> & full_sequence,
    std::vector<float> & normalized_actions,
    std::string & error) const {
    normalized_actions.clear();
    error.clear();
    if (impl_ == nullptr || impl_->codec == nullptr) {
        error = "StarVLA FAST policy is not initialized";
        return false;
    }
    FastDecodeResult decoded;
    if (!impl_->codec->decode_generated_tokens({full_sequence}, decoded, error)) {
        return false;
    }
    if (decoded.batch_size != 1 ||
        decoded.time_horizon != static_cast<size_t>(impl_->config.horizon) ||
        decoded.action_dim != static_cast<size_t>(impl_->config.action_dim) ||
        decoded.actions.size() !=
            static_cast<size_t>(impl_->config.horizon * impl_->config.action_dim)) {
        error = "embedded FAST codec returned an incompatible action tensor";
        return false;
    }
    normalized_actions.reserve(decoded.actions.size());
    for (double value : decoded.actions) {
        const float converted = static_cast<float>(value);
        if (!std::isfinite(converted)) {
            normalized_actions.clear();
            error = "embedded FAST codec returned a non-finite action";
            return false;
        }
        normalized_actions.push_back(converted);
    }
    return true;
}

bool FastPolicy::unnormalize(
    const std::vector<float> & normalized_actions,
    const std::string & profile_key, std::vector<float> & actions,
    std::string & error) const {
    if (impl_ == nullptr) {
        actions.clear();
        error = "StarVLA FAST policy is not initialized";
        return false;
    }
    return denormalize_actions(impl_->config.normalization, profile_key,
                               normalized_actions, impl_->config.horizon,
                               impl_->config.action_dim, actions, error);
}

const FastPolicyConfig & FastPolicy::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("StarVLA FAST policy is not initialized");
    }
    return impl_->config;
}

const char * FastPolicy::backend_name() const {
    return "cpu";
}

} // namespace robotcpp::starvla
