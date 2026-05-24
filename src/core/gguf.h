#pragma once

#include "core/types.h"
#include "vlacpp.h"

#include <string>
#include <vector>

namespace vlacpp {

vlacpp_status load_gguf_model_file(
    const std::string & path,
    ModelConfig & out_config,
    TensorMap & out_tensors);

bool read_gguf_tensor_f32(
    const std::string & path,
    const std::string & tensor_name,
    std::vector<float> & out);

} // namespace vlacpp
