#pragma once

#include <cstddef>
#include <string>

struct ggml_tensor;

namespace robotcpp::starvla {

enum class QwenBF16RoundStatus { unavailable, success, error };

struct QwenBF16CaptureCuda {
    QwenBF16CaptureCuda() = default;
    ~QwenBF16CaptureCuda();
    QwenBF16CaptureCuda(const QwenBF16CaptureCuda &) = delete;
    QwenBF16CaptureCuda & operator=(const QwenBF16CaptureCuda &) = delete;

    void * data = nullptr;
    size_t capacity = 0;
    int device = -1;
};

QwenBF16RoundStatus qwen_bf16_round_cuda(ggml_tensor * tensor, size_t count,
                                         std::string & error);
QwenBF16RoundStatus qwen_bf16_capture_cuda(
    ggml_tensor * tensor, size_t count, size_t offset, size_t total_count,
    QwenBF16CaptureCuda & capture, std::string & error);
bool qwen_bf16_capture_download_cuda(QwenBF16CaptureCuda & capture,
                                     float * values, size_t count,
                                     std::string & error);

} // namespace robotcpp::starvla
