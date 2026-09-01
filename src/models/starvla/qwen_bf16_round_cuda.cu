#include "models/starvla/qwen_bf16_round_cuda.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace robotcpp::starvla {
namespace {

__global__ void round_bf16(float * values, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        uint32_t bits = __float_as_uint(values[index]);
        uint32_t bf16 = (bits & 0x7fffffffU) > 0x7f800000U
                            ? (bits >> 16) | 64U
                            : (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
        values[index] = __uint_as_float(bf16 << 16);
    }
}

const char * cuda_error(cudaError_t status) {
    return cudaGetErrorString(status);
}

} // namespace

QwenBF16RoundStatus qwen_bf16_round_cuda(ggml_tensor * tensor, size_t count,
                                         std::string & error) {
    error.clear();
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr ||
        tensor->type != GGML_TYPE_F32 || count == 0) {
        error = "Qwen3-VL CUDA BF16 roundtrip received an invalid tensor";
        return QwenBF16RoundStatus::error;
    }

    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_buffer_get_type(tensor->buffer);
    ggml_backend_dev_t device =
        buffer_type == nullptr ? nullptr : ggml_backend_buft_get_device(buffer_type);
    if (device == nullptr ||
        ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return QwenBF16RoundStatus::unavailable;
    }

    cudaPointerAttributes attributes{};
    cudaError_t status = cudaPointerGetAttributes(&attributes, tensor->data);
    if (status == cudaErrorInvalidValue) {
        cudaGetLastError();
        return QwenBF16RoundStatus::unavailable;
    }
    if (status != cudaSuccess) {
        error = std::string("failed to inspect Qwen3-VL CUDA tensor: ") + cuda_error(status);
        return QwenBF16RoundStatus::error;
    }
    if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
        return QwenBF16RoundStatus::unavailable;
    }

    int previous_device = 0;
    status = cudaGetDevice(&previous_device);
    if (status != cudaSuccess) {
        error = std::string("failed to read the active CUDA device: ") + cuda_error(status);
        return QwenBF16RoundStatus::error;
    }
    status = cudaSetDevice(attributes.device);
    if (status != cudaSuccess) {
        error = std::string("failed to select the Qwen3-VL CUDA device: ") + cuda_error(status);
        return QwenBF16RoundStatus::error;
    }

    constexpr int block_size = 256;
    const size_t block_count = (count + block_size - 1) / block_size;
    round_bf16<<<block_count, block_size, 0, cudaStreamPerThread>>>(
        static_cast<float *>(tensor->data), count);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(cudaStreamPerThread);
    }
    const cudaError_t restore_status = cudaSetDevice(previous_device);
    if (status != cudaSuccess) {
        error = std::string("failed to round Qwen3-VL residuals on CUDA: ") + cuda_error(status);
        return QwenBF16RoundStatus::error;
    }
    if (restore_status != cudaSuccess) {
        error = std::string("failed to restore the active CUDA device: ") +
                cuda_error(restore_status);
        return QwenBF16RoundStatus::error;
    }
    return QwenBF16RoundStatus::success;
}

} // namespace robotcpp::starvla
