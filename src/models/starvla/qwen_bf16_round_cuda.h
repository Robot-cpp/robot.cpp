#pragma once

#include <cstddef>
#include <string>

struct ggml_tensor;

namespace robotcpp::starvla {

enum class QwenBF16RoundStatus { unavailable, success, error };

QwenBF16RoundStatus qwen_bf16_round_cuda(ggml_tensor * tensor, size_t count,
                                         std::string & error);

} // namespace robotcpp::starvla
