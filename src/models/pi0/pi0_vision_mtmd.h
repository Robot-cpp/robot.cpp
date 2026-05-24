#pragma once

#include "models/model.h"

#include <string>
#include <vector>

struct clip_ctx;

namespace vlacpp {

class Pi0VisionMtmd {
public:
    Pi0VisionMtmd(const ModelConfig & config, const BackendConfig & backend);
    ~Pi0VisionMtmd();

    Pi0VisionMtmd(const Pi0VisionMtmd &) = delete;
    Pi0VisionMtmd & operator=(const Pi0VisionMtmd &) = delete;

    bool available() const;
    int output_width() const;
    const std::string & path() const;

    bool encode(
        const std::vector<ImageTensor> & images,
        int n_threads,
        std::vector<float> & out_embeddings,
        int & out_token_count) const;

private:
    clip_ctx * ctx_ = nullptr;
    std::string path_;
    bool openpi_projector_ = false;
    int openpi_output_width_ = 0;
    std::vector<float> openpi_projection_bias_;
};

} // namespace vlacpp
