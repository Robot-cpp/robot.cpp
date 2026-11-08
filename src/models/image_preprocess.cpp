#include "models/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace robotcpp {
namespace {

struct RGBImage {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

struct FilterTable {
    int kernel_size = 0;
    int precision   = 0;
    std::vector<int> first;
    std::vector<int> count;
    std::vector<int32_t> weights;
};

double keys_cubic(double value) {
    constexpr double a = -0.5;
    value              = std::fabs(value);
    if (value < 1.0) {
        return ((a + 2.0) * value - (a + 3.0)) * value * value + 1.0;
    }
    if (value < 2.0) {
        return (((value - 5.0) * value + 8.0) * value - 4.0) * a;
    }
    return 0.0;
}

bool validate_resize(const uint8_t * source, int source_width, int source_height, int source_stride, int target_width,
                     int target_height, std::string & error) {
    error.clear();
    if (source == nullptr || source_width <= 0 || source_height <= 0 || target_width <= 0 || target_height <= 0 ||
        source_width > std::numeric_limits<int>::max() / 3 || target_width > std::numeric_limits<int>::max() / 3) {
        error = "Image resize received an invalid image or dimension";
        return false;
    }
    const int packed_stride = source_width * 3;
    if (source_stride != 0 && source_stride < packed_stride) {
        error = "Image stride is smaller than a packed RGB row";
        return false;
    }
    const uint64_t output_bytes = static_cast<uint64_t>(target_width) * static_cast<uint64_t>(target_height) * 3;
    if (output_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "Resized image is too large";
        return false;
    }
    return true;
}

RGBImage pack_source(const uint8_t * source, int width, int height, int stride) {
    RGBImage image;
    image.width                = width;
    image.height               = height;
    const size_t row_bytes     = static_cast<size_t>(width) * 3;
    const size_t actual_stride = stride > 0 ? static_cast<size_t>(stride) : row_bytes;
    image.pixels.resize(row_bytes * static_cast<size_t>(height));
    for (int row = 0; row < height; ++row) {
        std::copy(source + static_cast<size_t>(row) * actual_stride,
                  source + static_cast<size_t>(row) * actual_stride + row_bytes,
                  image.pixels.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(row) * row_bytes));
    }
    return image;
}

FilterTable make_filter_table(int input_size, int output_size, bool pillow_precision) {
    const double scale        = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double filter_scale = std::max(scale, 1.0);
    const double support      = 2.0 * filter_scale;

    FilterTable table;
    table.kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    table.first.resize(static_cast<size_t>(output_size));
    table.count.resize(static_cast<size_t>(output_size));
    std::vector<double> floating_weights(static_cast<size_t>(output_size) * static_cast<size_t>(table.kernel_size),
                                         0.0);
    double maximum_weight = 0.0;

    for (int output = 0; output < output_size; ++output) {
        const double center                      = (static_cast<double>(output) + 0.5) * scale;
        const int first                          = std::max(static_cast<int>(center - support + 0.5), 0);
        const int end                            = std::min(static_cast<int>(center + support + 0.5), input_size);
        const int count                          = std::max(0, std::min(end - first, table.kernel_size));
        table.first[static_cast<size_t>(output)] = first;
        table.count[static_cast<size_t>(output)] = count;

        double sum = 0.0;
        for (int index = 0; index < count; ++index) {
            const double distance = (static_cast<double>(index + first) - center + 0.5) / filter_scale;
            const double weight   = keys_cubic(distance);
            floating_weights[static_cast<size_t>(output) * table.kernel_size + index] = weight;
            sum += weight;
        }
        if (sum != 0.0) {
            for (int index = 0; index < count; ++index) {
                double & weight = floating_weights[static_cast<size_t>(output) * table.kernel_size + index];
                weight /= sum;
                maximum_weight = std::max(maximum_weight, weight);
            }
        }
    }

    if (pillow_precision) {
        table.precision = 22;
    } else {
        for (table.precision = 0; table.precision < 22; ++table.precision) {
            const int next =
                static_cast<int>(0.5 + maximum_weight * static_cast<double>(uint32_t{1} << (table.precision + 1)));
            if (next >= (1 << 15)) {
                break;
            }
        }
    }

    const double multiplier = static_cast<double>(uint32_t{1} << table.precision);
    table.weights.resize(floating_weights.size());
    for (size_t index = 0; index < floating_weights.size(); ++index) {
        const double scaled  = floating_weights[index] * multiplier;
        table.weights[index] = static_cast<int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
    }
    return table;
}

uint8_t fixed_point_pixel(int64_t accumulator, int precision) {
    const int64_t value = accumulator >> precision;
    return static_cast<uint8_t>(std::max<int64_t>(0, std::min<int64_t>(255, value)));
}

RGBImage resize_horizontal(const RGBImage & source, int target_width, const FilterTable & table) {
    RGBImage target;
    target.width  = target_width;
    target.height = source.height;
    target.pixels.resize(static_cast<size_t>(target.width) * target.height * 3);
    const int64_t rounding = int64_t{1} << (table.precision - 1);
    for (int row = 0; row < source.height; ++row) {
        for (int column = 0; column < target.width; ++column) {
            const int first = table.first[static_cast<size_t>(column)];
            const int count = table.count[static_cast<size_t>(column)];
            for (int channel = 0; channel < 3; ++channel) {
                int64_t accumulator = rounding;
                for (int index = 0; index < count; ++index) {
                    const size_t source_index = (static_cast<size_t>(row) * source.width + first + index) * 3 + channel;
                    const int32_t weight      = table.weights[static_cast<size_t>(column) * table.kernel_size + index];
                    accumulator += static_cast<int64_t>(source.pixels[source_index]) * weight;
                }
                const size_t target_index   = (static_cast<size_t>(row) * target.width + column) * 3 + channel;
                target.pixels[target_index] = fixed_point_pixel(accumulator, table.precision);
            }
        }
    }
    return target;
}

RGBImage resize_vertical(const RGBImage & source, int target_height, const FilterTable & table) {
    RGBImage target;
    target.width  = source.width;
    target.height = target_height;
    target.pixels.resize(static_cast<size_t>(target.width) * target.height * 3);
    const int64_t rounding = int64_t{1} << (table.precision - 1);
    for (int row = 0; row < target.height; ++row) {
        const int first = table.first[static_cast<size_t>(row)];
        const int count = table.count[static_cast<size_t>(row)];
        for (int column = 0; column < target.width; ++column) {
            for (int channel = 0; channel < 3; ++channel) {
                int64_t accumulator = rounding;
                for (int index = 0; index < count; ++index) {
                    const size_t source_index =
                        (static_cast<size_t>(first + index) * source.width + column) * 3 + channel;
                    const int32_t weight = table.weights[static_cast<size_t>(row) * table.kernel_size + index];
                    accumulator += static_cast<int64_t>(source.pixels[source_index]) * weight;
                }
                const size_t target_index   = (static_cast<size_t>(row) * target.width + column) * 3 + channel;
                target.pixels[target_index] = fixed_point_pixel(accumulator, table.precision);
            }
        }
    }
    return target;
}

bool resize_rgb(const uint8_t * source, int source_width, int source_height, int source_stride, int target_width,
                int target_height, bool pillow_precision, std::vector<uint8_t> & target, std::string & error) {
    target.clear();
    if (!validate_resize(source, source_width, source_height, source_stride, target_width, target_height, error)) {
        return false;
    }

    RGBImage current = pack_source(source, source_width, source_height, source_stride);
    if (source_width != target_width) {
        current =
            resize_horizontal(current, target_width, make_filter_table(source_width, target_width, pillow_precision));
    }
    if (source_height != target_height) {
        current =
            resize_vertical(current, target_height, make_filter_table(source_height, target_height, pillow_precision));
    }
    target = std::move(current.pixels);
    return true;
}

} // namespace

bool resize_pillow_bicubic_rgb(const uint8_t * source, int source_width, int source_height, int source_stride,
                               int target_width, int target_height, std::vector<uint8_t> & target,
                               std::string & error) {
    return resize_rgb(source, source_width, source_height, source_stride, target_width, target_height, true, target,
                      error);
}

bool resize_torchvision_bicubic_aa_rgb(const uint8_t * source, int source_width, int source_height, int source_stride,
                                       int target_width, int target_height, std::vector<uint8_t> & target,
                                       std::string & error) {
    return resize_rgb(source, source_width, source_height, source_stride, target_width, target_height, false, target,
                      error);
}

} // namespace robotcpp
