#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace robotcpp::starvla {

bool resize_pillow_bicubic_rgb(const uint8_t * source, int source_width, int source_height,
                               int source_stride, int target_width, int target_height,
                               std::vector<uint8_t> & target, std::string & error);

bool resize_torchvision_bicubic_aa_rgb(const uint8_t * source, int source_width,
                                       int source_height, int source_stride, int target_width,
                                       int target_height, std::vector<uint8_t> & target,
                                       std::string & error);

bool qwen3vl_smart_resize_dimensions(int source_width, int source_height, int factor,
                                     int min_pixels, int max_pixels, int & target_width,
                                     int & target_height, std::string & error);

bool preprocess_qwen3vl_rgb(const uint8_t * source, int source_width, int source_height,
                            int channels, int source_stride, int patch_size,
                            int spatial_merge_size, int min_pixels, int max_pixels,
                            std::vector<uint8_t> & target, int & target_width,
                            int & target_height, int & image_token_count,
                            std::string & error);

bool preprocess_oft_rgb(const uint8_t * source, int source_width, int source_height,
                        int channels, int source_stride, int training_width,
                        int training_height, int processor_width, int processor_height,
                        std::vector<uint8_t> & target, std::string & error);

} // namespace robotcpp::starvla
