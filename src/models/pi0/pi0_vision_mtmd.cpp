#include "models/pi0/pi0_vision_mtmd.h"

#include "core/gguf.h"

#include "clip.h"

#include <algorithm>
#include <fstream>

namespace vlacpp {
namespace {

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return static_cast<bool>(file);
}

std::string strip_gguf_suffix(const std::string & path) {
    constexpr const char * suffix = ".gguf";
    if (path.size() >= 5 && path.compare(path.size() - 5, 5, suffix) == 0) {
        return path.substr(0, path.size() - 5);
    }
    return path;
}

std::vector<std::string> sidecar_candidates(const std::string & model_path) {
    if (model_path.empty()) {
        return {};
    }
    const std::string stem = strip_gguf_suffix(model_path);
    return {
        stem + ".vision-mtmd.gguf",
        stem + "-vision-mtmd.gguf",
        model_path + ".vision-mtmd.gguf",
    };
}

bool add_projection_bias(
    const std::vector<float> & bias,
    int width,
    std::vector<float> & embeddings) {
    if (bias.empty()) {
        return true;
    }
    if (width <= 0 || bias.size() != static_cast<size_t>(width) ||
        embeddings.size() % static_cast<size_t>(width) != 0) {
        return false;
    }
    const size_t tokens = embeddings.size() / static_cast<size_t>(width);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t i = 0; i < bias.size(); ++i) {
            embeddings[token * bias.size() + i] += bias[i];
        }
    }
    return true;
}

} // namespace

Pi0VisionMtmd::Pi0VisionMtmd(const ModelConfig & config, const BackendConfig & backend) {
    for (const std::string & candidate : sidecar_candidates(config.source_path)) {
        if (file_exists(candidate)) {
            path_ = candidate;
            break;
        }
    }
    if (path_.empty()) {
        return;
    }
    openpi_projector_ = read_gguf_tensor_f32(path_, "mm.input_projection.bias", openpi_projection_bias_);
    openpi_output_width_ = static_cast<int>(openpi_projection_bias_.size());

    clip_context_params params{};
    params.use_gpu = backend.backend == VLACPP_BACKEND_CUDA;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.image_min_tokens = -1;
    params.image_max_tokens = -1;
    params.warmup = false;
    clip_init_result result = clip_init(path_.c_str(), params);
    ctx_ = result.ctx_v;
    if (result.ctx_a != nullptr) {
        clip_free(result.ctx_a);
    }
    if (ctx_ != nullptr && !clip_has_vision_encoder(ctx_)) {
        clip_free(ctx_);
        ctx_ = nullptr;
    }
}

Pi0VisionMtmd::~Pi0VisionMtmd() {
    if (ctx_ != nullptr) {
        clip_free(ctx_);
    }
}

bool Pi0VisionMtmd::available() const {
    return ctx_ != nullptr;
}

int Pi0VisionMtmd::output_width() const {
    if (ctx_ == nullptr) {
        return 0;
    }
    return openpi_projector_ && openpi_output_width_ > 0 ? openpi_output_width_ : clip_n_mmproj_embd(ctx_);
}

const std::string & Pi0VisionMtmd::path() const {
    return path_;
}

bool Pi0VisionMtmd::encode(
    const std::vector<ImageTensor> & images,
    int n_threads,
    std::vector<float> & out_embeddings,
    int & out_token_count) const {
    out_embeddings.clear();
    out_token_count = 0;
    if (ctx_ == nullptr) {
        return false;
    }

    const int width = output_width();
    if (width <= 0) {
        return false;
    }

    for (const ImageTensor & image : images) {
        if (image.width <= 0 || image.height <= 0 || image.channels != 3) {
            return false;
        }
        const size_t expected =
            static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * 3;
        if (image.data.size() != expected) {
            return false;
        }

        std::vector<float> encoded;
        const size_t nbytes = clip_embd_nbytes_by_img(ctx_, image.width, image.height);
        if (nbytes == 0 || nbytes % sizeof(float) != 0) {
            return false;
        }
        encoded.resize(nbytes / sizeof(float));
        if (!clip_encode_float_image(
                ctx_,
                std::max(1, n_threads),
                const_cast<float *>(image.data.data()),
                image.height,
                image.width,
                encoded.data())) {
            return false;
        }
        if (openpi_projector_ && !add_projection_bias(openpi_projection_bias_, width, encoded)) {
            return false;
        }
        if (encoded.empty() || encoded.size() % static_cast<size_t>(width) != 0) {
            return false;
        }
        out_embeddings.insert(out_embeddings.end(), encoded.begin(), encoded.end());
        out_token_count += static_cast<int>(encoded.size() / static_cast<size_t>(width));
    }

    return true;
}

} // namespace vlacpp
