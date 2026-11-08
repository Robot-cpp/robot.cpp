#include "models/starvla/oft_image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace robotcpp::starvla {

bool qwen3vl_smart_resize_dimensions(int source_width, int source_height, int factor, int min_pixels, int max_pixels,
                                     int & target_width, int & target_height, std::string & error) {
    target_width  = 0;
    target_height = 0;
    error.clear();
    if (source_width <= 0 || source_height <= 0 || factor <= 0 || min_pixels <= 0 || max_pixels < min_pixels) {
        error = "Qwen3-VL smart resize received an invalid dimension or pixel bound";
        return false;
    }
    const int minimum_side = std::min(source_width, source_height);
    const int maximum_side = std::max(source_width, source_height);
    if (static_cast<double>(maximum_side) / minimum_side > 200.0) {
        error = "Qwen3-VL smart resize requires an absolute aspect ratio of at most 200";
        return false;
    }

    // Python round() uses ties-to-even. Integer quotient/remainder arithmetic
    // makes the common first smart_resize step independent of the host FP mode.
    const auto round_div_ties_to_even = [](int value, int divisor) -> int64_t {
        const int64_t quotient  = value / divisor;
        const int64_t remainder = value % divisor;
        const int64_t doubled   = remainder * 2;
        if (doubled < divisor || (doubled == divisor && quotient % 2 == 0)) {
            return quotient;
        }
        return quotient + 1;
    };

    int64_t resized_height       = round_div_ties_to_even(source_height, factor) * factor;
    int64_t resized_width        = round_div_ties_to_even(source_width, factor) * factor;
    const int64_t source_pixels  = static_cast<int64_t>(source_height) * source_width;
    const int64_t rounded_pixels = resized_height * resized_width;
    if (rounded_pixels > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(source_pixels) / max_pixels);
        resized_height =
            std::max<int64_t>(factor, static_cast<int64_t>(std::floor(source_height / beta / factor)) * factor);
        resized_width =
            std::max<int64_t>(factor, static_cast<int64_t>(std::floor(source_width / beta / factor)) * factor);
    } else if (rounded_pixels < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / source_pixels);
        resized_height    = static_cast<int64_t>(std::ceil(source_height * beta / factor)) * factor;
        resized_width     = static_cast<int64_t>(std::ceil(source_width * beta / factor)) * factor;
    }

    if (resized_width <= 0 || resized_height <= 0 || resized_width > std::numeric_limits<int>::max() ||
        resized_height > std::numeric_limits<int>::max() ||
        resized_width > std::numeric_limits<int64_t>::max() / resized_height) {
        error = "Qwen3-VL smart resize produced an unsupported output dimension";
        return false;
    }
    target_width  = static_cast<int>(resized_width);
    target_height = static_cast<int>(resized_height);
    return true;
}

bool preprocess_qwen3vl_rgb(const uint8_t * source, int source_width, int source_height, int channels,
                            int source_stride, int patch_size, int spatial_merge_size, int min_pixels, int max_pixels,
                            std::vector<uint8_t> & target, int & target_width, int & target_height,
                            int & image_token_count, std::string & error) {
    target.clear();
    target_width      = 0;
    target_height     = 0;
    image_token_count = 0;
    if (channels != 3) {
        error = "Qwen3-VL input image must be RGB";
        return false;
    }
    if (patch_size <= 0 || spatial_merge_size <= 0 ||
        patch_size > std::numeric_limits<int>::max() / spatial_merge_size) {
        error = "Qwen3-VL patch or spatial merge size is invalid";
        return false;
    }
    const int factor = patch_size * spatial_merge_size;
    if (!qwen3vl_smart_resize_dimensions(source_width, source_height, factor, min_pixels, max_pixels, target_width,
                                         target_height, error)) {
        return false;
    }
    if (!resize_torchvision_bicubic_aa_rgb(source, source_width, source_height, source_stride, target_width,
                                           target_height, target, error)) {
        target_width  = 0;
        target_height = 0;
        return false;
    }
    const int64_t grid_width  = target_width / factor;
    const int64_t grid_height = target_height / factor;
    const int64_t tokens      = grid_width * grid_height;
    if (tokens <= 0 || tokens > std::numeric_limits<int>::max()) {
        target.clear();
        target_width  = 0;
        target_height = 0;
        error         = "Qwen3-VL smart resize produced an unsupported image token count";
        return false;
    }
    image_token_count = static_cast<int>(tokens);
    return true;
}

bool preprocess_oft_rgb(const uint8_t * source, int source_width, int source_height, int channels, int source_stride,
                        int training_width, int training_height, int processor_width, int processor_height,
                        std::vector<uint8_t> & target, std::string & error) {
    target.clear();
    if (channels != 3) {
        error = "StarVLA OFT input image must be RGB";
        return false;
    }
    std::vector<uint8_t> training_image;
    if (!resize_pillow_bicubic_rgb(source, source_width, source_height, source_stride, training_width, training_height,
                                   training_image, error)) {
        return false;
    }
    return resize_torchvision_bicubic_aa_rgb(training_image.data(), training_width, training_height, training_width * 3,
                                             processor_width, processor_height, target, error);
}

} // namespace robotcpp::starvla
