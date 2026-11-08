#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace robotcpp {

bool resize_pillow_bicubic_rgb(const uint8_t * source, int source_width, int source_height, int source_stride,
                               int target_width, int target_height, std::vector<uint8_t> & target, std::string & error);

bool resize_torchvision_bicubic_aa_rgb(const uint8_t * source, int source_width, int source_height, int source_stride,
                                       int target_width, int target_height, std::vector<uint8_t> & target,
                                       std::string & error);

} // namespace robotcpp
