#include "models/starvla/qwen_bf16_round_cuda.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace robotcpp::starvla {
namespace {

__device__ uint32_t bf16_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x7fffffffU) > 0x7f800000U
               ? (bits >> 16) | 64U
               : (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
}

__global__ void round_bf16(float * values, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = __uint_as_float(bf16_bits(values[index]) << 16);
    }
}

__global__ void capture_bf16(const float * source, float * destination,
                             size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = __uint_as_float(bf16_bits(source[index]) << 16);
    }
}

const char * cuda_error(cudaError_t status) {
    return cudaGetErrorString(status);
}

cudaError_t select_device(int device, int & previous_device) {
    const cudaError_t status = cudaGetDevice(&previous_device);
    return status == cudaSuccess ? cudaSetDevice(device) : status;
}

QwenBF16RoundStatus tensor_device(ggml_tensor * tensor, size_t count,
                                  int & device_id, std::string & error) {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr ||
        tensor->type != GGML_TYPE_F32 || count == 0) {
        error = "Qwen-VL CUDA BF16 operation received an invalid tensor";
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
    const cudaError_t status = cudaPointerGetAttributes(&attributes, tensor->data);
    if (status == cudaErrorInvalidValue) {
        cudaGetLastError();
        return QwenBF16RoundStatus::unavailable;
    }
    if (status != cudaSuccess) {
        error = std::string("failed to inspect Qwen-VL CUDA tensor: ") +
                cuda_error(status);
        return QwenBF16RoundStatus::error;
    }
    if (attributes.type != cudaMemoryTypeDevice &&
        attributes.type != cudaMemoryTypeManaged) {
        return QwenBF16RoundStatus::unavailable;
    }
    device_id = attributes.device;
    return QwenBF16RoundStatus::success;
}

} // namespace

QwenBF16CaptureCuda::~QwenBF16CaptureCuda() {
    if (data == nullptr) {
        return;
    }
    int previous_device = 0;
    if (select_device(device, previous_device) == cudaSuccess) {
        cudaFree(data);
        cudaSetDevice(previous_device);
    }
}

QwenBF16RoundStatus qwen_bf16_round_cuda(ggml_tensor * tensor, size_t count,
                                         std::string & error) {
    error.clear();
    int device = -1;
    const QwenBF16RoundStatus available =
        tensor_device(tensor, count, device, error);
    if (available != QwenBF16RoundStatus::success) {
        return available;
    }

    int previous_device = 0;
    cudaError_t status = select_device(device, previous_device);
    if (status != cudaSuccess) {
        error = std::string("failed to select the Qwen-VL CUDA device: ") +
                cuda_error(status);
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
        error = std::string("failed to round Qwen-VL residuals on CUDA: ") +
                cuda_error(status);
        return QwenBF16RoundStatus::error;
    }
    if (restore_status != cudaSuccess) {
        error = std::string("failed to restore the active CUDA device: ") +
                cuda_error(restore_status);
        return QwenBF16RoundStatus::error;
    }
    return QwenBF16RoundStatus::success;
}

QwenBF16RoundStatus qwen_bf16_capture_cuda(
    ggml_tensor * tensor, size_t count, size_t offset, size_t total_count,
    QwenBF16CaptureCuda & capture, std::string & error) {
    error.clear();
    if (offset > total_count || count > total_count - offset ||
        total_count > SIZE_MAX / sizeof(float)) {
        error = "Qwen-VL CUDA capture range is invalid";
        return QwenBF16RoundStatus::error;
    }
    int device = -1;
    const QwenBF16RoundStatus available =
        tensor_device(tensor, count, device, error);
    if (available != QwenBF16RoundStatus::success) {
        return available;
    }

    int previous_device = 0;
    cudaError_t status = select_device(device, previous_device);
    if (status != cudaSuccess) {
        error = std::string("failed to select the Qwen-VL CUDA device: ") +
                cuda_error(status);
        return QwenBF16RoundStatus::error;
    }
    if (capture.data != nullptr &&
        (capture.device != device || capture.capacity < total_count)) {
        status = cudaFree(capture.data);
        if (status == cudaSuccess) {
            capture.data = nullptr;
            capture.capacity = 0;
            capture.device = -1;
        }
    }
    if (status == cudaSuccess && capture.data == nullptr) {
        status = cudaMalloc(&capture.data, total_count * sizeof(float));
        if (status == cudaSuccess) {
            capture.capacity = total_count;
            capture.device = device;
        }
    }
    constexpr int block_size = 256;
    const size_t block_count = (count + block_size - 1) / block_size;
    if (status == cudaSuccess) {
        capture_bf16<<<block_count, block_size, 0, cudaStreamPerThread>>>(
            static_cast<const float *>(tensor->data),
            static_cast<float *>(capture.data) + offset, count);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(cudaStreamPerThread);
    }
    const cudaError_t restore_status = cudaSetDevice(previous_device);
    if (status != cudaSuccess || restore_status != cudaSuccess) {
        error = std::string("failed to capture Qwen-VL hidden states on CUDA: ") +
                cuda_error(status != cudaSuccess ? status : restore_status);
        return QwenBF16RoundStatus::error;
    }
    return QwenBF16RoundStatus::success;
}

bool qwen_bf16_capture_download_cuda(QwenBF16CaptureCuda & capture,
                                     float * values, size_t count,
                                     std::string & error) {
    error.clear();
    if (capture.data == nullptr || values == nullptr || count == 0 ||
        count > capture.capacity || count > SIZE_MAX / sizeof(float)) {
        error = "Qwen-VL CUDA capture download is invalid";
        return false;
    }
    int previous_device = 0;
    cudaError_t status = select_device(capture.device, previous_device);
    if (status != cudaSuccess) {
        error = std::string("failed to select the Qwen-VL CUDA device: ") +
                cuda_error(status);
        return false;
    }
    status = cudaMemcpy(values, capture.data, count * sizeof(float),
                        cudaMemcpyDeviceToHost);
    const cudaError_t restore_status = cudaSetDevice(previous_device);
    if (status != cudaSuccess || restore_status != cudaSuccess) {
        error = std::string("failed to download Qwen-VL CUDA hidden states: ") +
                cuda_error(status != cudaSuccess ? status : restore_status);
        return false;
    }
    return true;
}

} // namespace robotcpp::starvla
