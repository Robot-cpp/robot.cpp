#pragma once

#include "ggml.h"
#include "gguf.h"
#include "models/starvla/normalization.h"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::starvla::detail {

inline int require_key(gguf_context * gguf, const char * key, gguf_type type) {
    const int index = gguf_find_key(gguf, key);
    if (index < 0) {
        throw std::runtime_error(std::string("missing required StarVLA GGUF metadata: ") + key);
    }
    if (gguf_get_kv_type(gguf, index) != type) {
        throw std::runtime_error(std::string("invalid StarVLA GGUF metadata type: ") + key);
    }
    return index;
}

inline std::string require_string(gguf_context * gguf, const char * key) {
    return gguf_get_val_str(gguf, require_key(gguf, key, GGUF_TYPE_STRING));
}

inline int32_t require_i32(gguf_context * gguf, const char * key) {
    return gguf_get_val_i32(gguf, require_key(gguf, key, GGUF_TYPE_INT32));
}

inline float require_f32(gguf_context * gguf, const char * key) {
    return gguf_get_val_f32(gguf, require_key(gguf, key, GGUF_TYPE_FLOAT32));
}

inline bool require_bool(gguf_context * gguf, const char * key) {
    return gguf_get_val_bool(gguf, require_key(gguf, key, GGUF_TYPE_BOOL));
}

inline int require_array(gguf_context * gguf, const char * key, gguf_type element_type) {
    const int index = require_key(gguf, key, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(gguf, index) != element_type) {
        throw std::runtime_error(std::string("invalid StarVLA GGUF array element type: ") + key);
    }
    return index;
}

inline std::vector<std::string> require_string_array(gguf_context * gguf, const char * key) {
    const int index    = require_array(gguf, key, GGUF_TYPE_STRING);
    const size_t count = gguf_get_arr_n(gguf, index);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        values.emplace_back(gguf_get_arr_str(gguf, index, i));
    }
    return values;
}

template <typename T>
inline std::vector<T> require_numeric_array(gguf_context * gguf, const char * key, gguf_type type) {
    const int index    = require_array(gguf, key, type);
    const size_t count = gguf_get_arr_n(gguf, index);
    if (count == 0) {
        return {};
    }
    const auto * data = static_cast<const T *>(gguf_get_arr_data(gguf, index));
    if (data == nullptr) {
        throw std::runtime_error(std::string("missing StarVLA GGUF array data: ") + key);
    }
    return std::vector<T>(data, data + count);
}

inline std::vector<int32_t> require_i32_array(gguf_context * gguf, const char * key) {
    return require_numeric_array<int32_t>(gguf, key, GGUF_TYPE_INT32);
}

inline std::vector<float> require_f32_array(gguf_context * gguf, const char * key) {
    return require_numeric_array<float>(gguf, key, GGUF_TYPE_FLOAT32);
}

inline std::vector<uint8_t> require_bool_array(gguf_context * gguf, const char * key) {
    const auto raw = require_numeric_array<uint8_t>(gguf, key, GGUF_TYPE_BOOL);
    std::vector<uint8_t> values(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        values[i] = raw[i] != 0 ? 1 : 0;
    }
    return values;
}

inline std::string profile_key(int index, const char * suffix) {
    return "starvla.normalization.profile." + std::to_string(index) + "." + suffix;
}

inline NormalizationConfig require_normalization(gguf_context * gguf, int action_dim) {
    NormalizationConfig config;
    config.clip_actions          = require_bool(gguf, "starvla.normalization.clip_actions");
    config.binary_threshold      = require_f32(gguf, "starvla.normalization.binary_threshold");
    config.binary_comparison     = require_string(gguf, "starvla.normalization.binary_comparison");
    config.continuous_dimensions = require_i32_array(gguf, "starvla.action.continuous_dimensions");
    config.binary_dimensions     = require_i32_array(gguf, "starvla.action.binary_dimensions");

    const int profile_count = require_i32(gguf, "starvla.normalization.profile_count");
    const auto keys         = require_string_array(gguf, "starvla.normalization.profile_keys");
    if (profile_count <= 0 || keys.size() != static_cast<size_t>(profile_count)) {
        throw std::runtime_error("StarVLA normalization profile count is inconsistent");
    }
    config.default_profile_key = keys.front();
    config.profiles.reserve(static_cast<size_t>(profile_count));
    for (int index = 0; index < profile_count; ++index) {
        NormalizationProfile profile;
        profile.key         = require_string(gguf, profile_key(index, "key").c_str());
        profile.action_q01  = require_f32_array(gguf, profile_key(index, "action_q01").c_str());
        profile.action_q99  = require_f32_array(gguf, profile_key(index, "action_q99").c_str());
        profile.action_mask = require_bool_array(gguf, profile_key(index, "action_mask").c_str());
        if (profile.key != keys[static_cast<size_t>(index)]) {
            throw std::runtime_error("StarVLA normalization profile order is inconsistent");
        }
        config.profiles.push_back(std::move(profile));
    }

    std::string error;
    if (!validate_normalization_config(config, action_dim, error)) {
        throw std::runtime_error(error);
    }
    return config;
}

inline bool has_shape(const ggml_tensor * tensor, std::initializer_list<int64_t> expected) {
    if (tensor == nullptr || static_cast<size_t>(ggml_n_dims(tensor)) != expected.size()) {
        return false;
    }
    size_t dimension = 0;
    for (const int64_t value : expected) {
        if (tensor->ne[dimension++] != value) {
            return false;
        }
    }
    return true;
}

} // namespace robotcpp::starvla::detail
