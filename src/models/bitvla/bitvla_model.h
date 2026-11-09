#pragma once

#include "models/model.h"
#include <memory>
#include <vector>
#include <cstddef>

namespace robotcpp {
namespace bitvla {
std::unique_ptr<float[]> prompt_input_without_images(const std::vector<float> & prompt,
                                                     const std::vector<size_t> & image_positions);
}
bool make_bitvla_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error);
} // namespace robotcpp
