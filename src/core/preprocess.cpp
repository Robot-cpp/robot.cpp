#include "core/preprocess.h"

#include "core/error.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

namespace vlacpp {
namespace {

float sample_u8_rgb(const vlacpp_image_view & image, int x, int y, int c) {
    const int channel = std::min(c, image.channels - 1);
    const uint8_t * row = image.data + static_cast<size_t>(y) * image.stride_bytes;
    return static_cast<float>(row[x * image.channels + channel]) / 255.0f * 2.0f - 1.0f;
}

float sample_bilinear_u8_rgb(const vlacpp_image_view & image, float x, float y, int c) {
    x = std::max(0.0f, std::min(x, static_cast<float>(image.width - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(image.height - 1)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);
    const float top =
        sample_u8_rgb(image, x0, y0, c) * (1.0f - wx) +
        sample_u8_rgb(image, x1, y0, c) * wx;
    const float bottom =
        sample_u8_rgb(image, x0, y1, c) * (1.0f - wx) +
        sample_u8_rgb(image, x1, y1, c) * wx;
    return top * (1.0f - wy) + bottom * wy;
}

} // namespace

vlacpp_status validate_and_preprocess(
    const ModelConfig & config,
    const vlacpp_observation & raw,
    ObservationData & out) {
    out = {};

    if (config.state_dim > 0) {
        if (raw.state == nullptr) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation state is required");
        }
        if (raw.state_count != static_cast<size_t>(config.state_dim)) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation state dimension mismatch");
        }
        out.state.assign(raw.state, raw.state + raw.state_count);
        if (config.state_mean.size() == static_cast<size_t>(config.state_dim) ||
            config.state_std.size() == static_cast<size_t>(config.state_dim)) {
            if (config.state_mean.size() != static_cast<size_t>(config.state_dim) ||
                config.state_std.size() != static_cast<size_t>(config.state_dim)) {
                return fail(VLACPP_STATUS_PARSE_ERROR, "state_mean and state_std must both match state_dim");
            }
            for (int i = 0; i < config.state_dim; ++i) {
                const float std = config.state_std[static_cast<size_t>(i)];
                if (std == 0.0f) {
                    return fail(VLACPP_STATUS_PARSE_ERROR, "state_std contains zero");
                }
                out.state[static_cast<size_t>(i)] =
                    (out.state[static_cast<size_t>(i)] - config.state_mean[static_cast<size_t>(i)]) / std;
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
                return fail(VLACPP_STATUS_INVALID_ARGUMENT, "prompt token id must be non-negative");
            }
        }
    }
    if (raw.noise != nullptr && raw.noise_count > 0) {
        const size_t expected_noise =
            static_cast<size_t>(config.action_horizon) * static_cast<size_t>(config.action_dim);
        if (raw.noise_count != expected_noise) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation noise dimension mismatch");
        }
        out.noise.assign(raw.noise, raw.noise + raw.noise_count);
    }

    std::set<std::string> required(config.image_keys.begin(), config.image_keys.end());
    for (size_t i = 0; i < raw.image_count; ++i) {
        const vlacpp_image_view & image = raw.images[i];
        if (image.name == nullptr || image.data == nullptr) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "image view has null name or data");
        }
        if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.channels > 4) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "image view has invalid dimensions");
        }
        if (image.stride_bytes < image.width * image.channels) {
            return fail(VLACPP_STATUS_INVALID_ARGUMENT, "image stride is too small");
        }

        required.erase(image.name);

        ImageTensor tensor;
        tensor.name = image.name;
        tensor.width = config.image_width;
        tensor.height = config.image_height;
        tensor.channels = 3;
        tensor.data.resize(static_cast<size_t>(tensor.width) * tensor.height * tensor.channels);

        const float ratio = std::max(
            static_cast<float>(image.width) / static_cast<float>(tensor.width),
            static_cast<float>(image.height) / static_cast<float>(tensor.height));
        const int resized_width = std::max(1, static_cast<int>(static_cast<float>(image.width) / ratio));
        const int resized_height = std::max(1, static_cast<int>(static_cast<float>(image.height) / ratio));
        const int pad_x0 = (tensor.width - resized_width) / 2;
        const int pad_y0 = (tensor.height - resized_height) / 2;

        std::fill(tensor.data.begin(), tensor.data.end(), -1.0f);
        for (int y = pad_y0; y < pad_y0 + resized_height; ++y) {
            const float resized_y = static_cast<float>(y - pad_y0);
            const float src_y = (resized_y + 0.5f) * static_cast<float>(image.height) /
                                    static_cast<float>(resized_height) -
                                0.5f;
            for (int x = pad_x0; x < pad_x0 + resized_width; ++x) {
                const float resized_x = static_cast<float>(x - pad_x0);
                const float src_x = (resized_x + 0.5f) * static_cast<float>(image.width) /
                                        static_cast<float>(resized_width) -
                                    0.5f;
                for (int c = 0; c < 3; ++c) {
                    tensor.data[(static_cast<size_t>(y) * tensor.width + x) * 3 + c] =
                        sample_bilinear_u8_rgb(image, src_x, src_y, c);
                }
            }
        }
        out.images.push_back(std::move(tensor));
    }

    if (!required.empty()) {
        return fail(VLACPP_STATUS_INVALID_ARGUMENT, "observation missing required image view: " + *required.begin());
    }

    return VLACPP_STATUS_OK;
}

} // namespace vlacpp
