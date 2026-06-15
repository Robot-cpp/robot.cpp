#include "models/pi0/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace robotcpp::pi0 {
namespace {

bool pi0_preprocess_error_at(const char * func, const std::string & message) {
    std::fprintf(stderr, "[Pi0] Error: %s: %s\n", func, message.c_str());
    return false;
}

#define pi0_preprocess_error(message) pi0_preprocess_error_at(__func__, (message))

struct ResizeCoord {
    int lo = 0;
    int hi = 0;
    float lo_weight = 1.0f;
    float hi_weight = 0.0f;
};

std::vector<ResizeCoord> build_resize_coords(int src_size, int dst_size) {
    std::vector<ResizeCoord> coords(static_cast<size_t>(dst_size));
    for (int i = 0; i < dst_size; ++i) {
        float src = (static_cast<float>(i) + 0.5f) * static_cast<float>(src_size) /
                static_cast<float>(dst_size) -
            0.5f;
        src = std::max(0.0f, std::min(src, static_cast<float>(src_size - 1)));
        const int lo = static_cast<int>(std::floor(src));
        const int hi = std::min(lo + 1, src_size - 1);
        const float hi_weight = src - static_cast<float>(lo);
        coords[static_cast<size_t>(i)] = {lo, hi, 1.0f - hi_weight, hi_weight};
    }
    return coords;
}

inline float normalize_u8(float value) {
    return value * (2.0f / 255.0f) - 1.0f;
}

bool is_empty_camera_key(const std::string & key) {
    return key.find(".empty_camera_") != std::string::npos;
}

int find_real_image_key(const std::vector<std::string> & keys, const std::string & name) {
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!is_empty_camera_key(keys[i]) && keys[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace

bool validate_and_preprocess_pi0(
    const Pi0ModelConfig & config,
    const Pi0RawObservation & raw,
    Pi0Observation & out) {
    out = {};

    if (config.common.state_dim > 0) {
        if (raw.state == nullptr) {
            return pi0_preprocess_error("observation state is required");
        }
        if (raw.state_count != static_cast<size_t>(config.common.state_dim)) {
            return pi0_preprocess_error("observation state dimension mismatch");
        }
        out.state.assign(raw.state, raw.state + raw.state_count);
        if (config.common.state_mean.size() == static_cast<size_t>(config.common.state_dim) ||
            config.common.state_std.size() == static_cast<size_t>(config.common.state_dim)) {
            if (config.common.state_mean.size() != static_cast<size_t>(config.common.state_dim) ||
                config.common.state_std.size() != static_cast<size_t>(config.common.state_dim)) {
                return pi0_preprocess_error("state_mean and state_std must both match state_dim");
            }
            for (int i = 0; i < config.common.state_dim; ++i) {
                const float std = config.common.state_std[static_cast<size_t>(i)];
                if (std == 0.0f) {
                    return pi0_preprocess_error("state_std contains zero");
                }
                out.state[static_cast<size_t>(i)] =
                    (out.state[static_cast<size_t>(i)] - config.common.state_mean[static_cast<size_t>(i)]) / std;
            }
        }
    }

    if (raw.prompt != nullptr) {
        out.prompt = raw.prompt;
    }
    if (raw.prompt_tokens != nullptr && raw.prompt_token_count > 0) {
        out.prompt_tokens.assign(raw.prompt_tokens, raw.prompt_tokens + raw.prompt_token_count);
        for (int32_t token : out.prompt_tokens) {
            if (token < 0) {
                return pi0_preprocess_error("prompt token id must be non-negative");
            }
        }
    }
    if (raw.noise != nullptr && raw.noise_count > 0) {
        const size_t expected_noise =
            static_cast<size_t>(config.common.action_horizon) * static_cast<size_t>(config.common.action_dim);
        if (raw.noise_count != expected_noise) {
            return pi0_preprocess_error("observation noise dimension mismatch");
        }
        out.noise.assign(raw.noise, raw.noise + raw.noise_count);
    }

    if (config.common.image_keys.empty()) {
        return pi0_preprocess_error("model config has no image_keys");
    }

    std::vector<Pi0ImageTensor> images_by_key(config.common.image_keys.size());
    std::vector<bool> seen_by_key(config.common.image_keys.size(), false);
    for (size_t i = 0; i < raw.image_count; ++i) {
        const Pi0RawImageView & image = raw.images[i];
        if (image.name == nullptr || image.data == nullptr) {
            return pi0_preprocess_error("image view has null name or data");
        }
        const int key_index = find_real_image_key(config.common.image_keys, image.name);
        if (key_index < 0) {
            return pi0_preprocess_error(std::string("observation has unknown image view: ") + image.name);
        }
        if (seen_by_key[static_cast<size_t>(key_index)]) {
            return pi0_preprocess_error(std::string("observation has duplicate image view: ") + image.name);
        }
        if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.channels > 4) {
            return pi0_preprocess_error("image view has invalid dimensions");
        }
        if (image.stride_bytes < image.width * image.channels) {
            return pi0_preprocess_error("image stride is too small");
        }

        Pi0ImageTensor tensor;
        tensor.name = image.name;
        tensor.width = config.common.image_width;
        tensor.height = config.common.image_height;
        tensor.channels = 3;
        tensor.data.resize(static_cast<size_t>(tensor.width) * tensor.height * tensor.channels);

        const float ratio = std::max(
            static_cast<float>(image.width) / static_cast<float>(tensor.width),
            static_cast<float>(image.height) / static_cast<float>(tensor.height));
        const int resized_width = std::max(1, static_cast<int>(static_cast<float>(image.width) / ratio));
        const int resized_height = std::max(1, static_cast<int>(static_cast<float>(image.height) / ratio));
        const int pad_x0 = (tensor.width - resized_width) / 2;
        const int pad_y0 = (tensor.height - resized_height) / 2;

        const std::vector<ResizeCoord> x_coords = build_resize_coords(image.width, resized_width);
        const std::vector<ResizeCoord> y_coords = build_resize_coords(image.height, resized_height);
        const int channel_0 = 0;
        const int channel_1 = std::min(1, image.channels - 1);
        const int channel_2 = std::min(2, image.channels - 1);
        std::fill(tensor.data.begin(), tensor.data.end(), -1.0f);
        for (int ry = 0; ry < resized_height; ++ry) {
            const ResizeCoord & yc = y_coords[static_cast<size_t>(ry)];
            const uint8_t * row0 = image.data + static_cast<size_t>(yc.lo) * image.stride_bytes;
            const uint8_t * row1 = image.data + static_cast<size_t>(yc.hi) * image.stride_bytes;
            const int dst_y = pad_y0 + ry;
            for (int rx = 0; rx < resized_width; ++rx) {
                const ResizeCoord & xc = x_coords[static_cast<size_t>(rx)];
                const int x0 = xc.lo * image.channels;
                const int x1 = xc.hi * image.channels;
                const int dst_x = pad_x0 + rx;
                float * dst = tensor.data.data() +
                    (static_cast<size_t>(dst_y) * tensor.width + static_cast<size_t>(dst_x)) * 3;

                const float r_top =
                    static_cast<float>(row0[x0 + channel_0]) * xc.lo_weight +
                    static_cast<float>(row0[x1 + channel_0]) * xc.hi_weight;
                const float r_bottom =
                    static_cast<float>(row1[x0 + channel_0]) * xc.lo_weight +
                    static_cast<float>(row1[x1 + channel_0]) * xc.hi_weight;
                const float g_top =
                    static_cast<float>(row0[x0 + channel_1]) * xc.lo_weight +
                    static_cast<float>(row0[x1 + channel_1]) * xc.hi_weight;
                const float g_bottom =
                    static_cast<float>(row1[x0 + channel_1]) * xc.lo_weight +
                    static_cast<float>(row1[x1 + channel_1]) * xc.hi_weight;
                const float b_top =
                    static_cast<float>(row0[x0 + channel_2]) * xc.lo_weight +
                    static_cast<float>(row0[x1 + channel_2]) * xc.hi_weight;
                const float b_bottom =
                    static_cast<float>(row1[x0 + channel_2]) * xc.lo_weight +
                    static_cast<float>(row1[x1 + channel_2]) * xc.hi_weight;
                dst[0] = normalize_u8(r_top * yc.lo_weight + r_bottom * yc.hi_weight);
                dst[1] = normalize_u8(g_top * yc.lo_weight + g_bottom * yc.hi_weight);
                dst[2] = normalize_u8(b_top * yc.lo_weight + b_bottom * yc.hi_weight);
            }
        }
        images_by_key[static_cast<size_t>(key_index)] = std::move(tensor);
        seen_by_key[static_cast<size_t>(key_index)] = true;
    }

    out.images.reserve(raw.image_count);
    for (size_t i = 0; i < config.common.image_keys.size(); ++i) {
        const std::string & key = config.common.image_keys[i];
        if (is_empty_camera_key(key)) {
            continue;
        }
        if (!seen_by_key[i]) {
            return pi0_preprocess_error("observation missing required image view: " + key);
        }
        out.images.push_back(std::move(images_by_key[i]));
    }

    return true;
}

} // namespace robotcpp::pi0
