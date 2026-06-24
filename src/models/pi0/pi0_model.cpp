#include "models/pi0/pi0_model.h"

#include "models/pi0/pi0_engine.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace robotcpp {
namespace {

void add_metric(model_result & out, const char * name, double value) {
    model_metric metric;
    metric.name = name;
    metric.value = value;
    out.metrics.push_back(metric);
}

bool validate_options(const model_args & args, std::string & error) {
    if (args.vit_path.empty()) {
        error = "Pi0 vit_path is required";
        return false;
    }
    if (args.mmproj_path.empty()) {
        error = "Pi0 mmproj_path is required";
        return false;
    }
    if (args.llm_path.empty()) {
        error = "Pi0 llm_path is required";
        return false;
    }
    if (args.tokenizer_path.empty()) {
        error = "Pi0 tokenizer_path is required";
        return false;
    }
    if (args.state_path.empty()) {
        error = "Pi0 state_path is required";
        return false;
    }
    if (args.action_decoder_path.empty()) {
        error = "Pi0 action_decoder_path is required";
        return false;
    }
    return true;
}

} // namespace

Pi0Model::Pi0Model(const model_args & args) : args_(args) {
    pi0_params params = pi0_default_params();
    params.vit_path = args_.vit_path.c_str();
    params.mmproj_path = args_.mmproj_path.c_str();
    params.llm_path = args_.llm_path.c_str();
    params.tokenizer_path = args_.tokenizer_path.c_str();
    params.state_path = args_.state_path.c_str();
    params.action_decoder_path = args_.action_decoder_path.c_str();
    params.n_threads = args_.threads;
    params.noise_seed = args_.noise_seed;
    params.verbosity = args_.verbosity;

    ctx_ = pi0_init(params);
}

Pi0Model::~Pi0Model() {
    if (ctx_ != nullptr) {
        pi0_free(ctx_);
    }
}

const char * Pi0Model::type() const {
    return "pi0";
}

bool Pi0Model::predict(const observation & obs, model_result & out, std::string & error) {
    out = model_result{};
    if (ctx_ == nullptr) {
        error = "Pi0 model is not initialized";
        return false;
    }
    if (obs.images.empty()) {
        error = "Pi0 requires at least one image";
        return false;
    }
    std::vector<pi0_image_view> views;
    views.reserve(obs.images.size());
    for (const model_image & image : obs.images) {
        if (image.name.empty()) {
            error = "Pi0 image name is required";
            return false;
        }
        if (image.data == nullptr) {
            error = "Pi0 image data is null";
            return false;
        }
        pi0_image_view view{};
        view.name = image.name.c_str();
        view.data = image.data;
        view.width = image.width;
        view.height = image.height;
        view.channels = image.channels;
        view.stride_bytes = image.stride_bytes;
        views.push_back(view);
    }

    const pi0_result result =
        pi0_predict_raw_rgb(ctx_, views.data(), views.size(), obs.state.empty() ? nullptr : obs.state.data(),
                            obs.state.size(), obs.task.c_str());
    if (result.actions == nullptr) {
        error = "pi0_predict_raw_rgb failed";
        return false;
    }

    out.chunk_size = result.chunk_size;
    out.action_dim = result.action_dim;
    const size_t count = static_cast<size_t>(result.chunk_size) * static_cast<size_t>(result.action_dim);
    out.actions.assign(result.actions, result.actions + count);

    const pi0_stage_timings timings = pi0_get_last_stage_timings(ctx_);
    add_metric(out, "preprocess_ms", timings.preprocess_ms);
    add_metric(out, "prefix_ms", timings.prefix_ms);
    add_metric(out, "state_ms", timings.state_ms);
    add_metric(out, "denoise_ms", timings.denoise_ms);
    add_metric(out, "output_ms", timings.output_ms);
    add_metric(out, "model_total_ms", timings.total_ms);
    return true;
}

void Pi0Model::reset() {
    if (ctx_ != nullptr) {
        pi0_reset(ctx_);
    }
}

bool Pi0Model::is_ready() const {
    return ctx_ != nullptr;
}

bool make_pi0_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error) {
    out.reset();
    if (!validate_options(args, error)) {
        return false;
    }
    std::unique_ptr<Pi0Model> model(new (std::nothrow) Pi0Model(args));
    if (!model) {
        error = "failed to allocate Pi0 model";
        return false;
    }
    if (!model->is_ready()) {
        error = "failed to initialize Pi0 model";
        return false;
    }
    out = std::move(model);
    return true;
}

} // namespace robotcpp
