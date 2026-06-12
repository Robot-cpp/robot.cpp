#include "models/pi0/pi0_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>

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

std::string last_error_or_status(vlacpp_status status) {
    const char * error = vlacpp_last_error();
    if (error != nullptr && error[0] != '\0') {
        return error;
    }
    return "vlacpp status " + std::to_string(static_cast<int>(status));
}

} // namespace

Pi0Model::Pi0Model(const model_args & args)
    : args_(args) {
    vlacpp_model_artifact artifact_items[] = {
        {"vit", args_.vit_path.c_str()},
        {"mmproj", args_.mmproj_path.c_str()},
        {"llm", args_.llm_path.c_str()},
        {"tokenizer", args_.tokenizer_path.c_str()},
        {"state", args_.state_path.c_str()},
        {"action_decoder", args_.action_decoder_path.c_str()},
    };
    vlacpp_model_artifacts artifacts{};
    artifacts.items = artifact_items;
    artifacts.count = sizeof(artifact_items) / sizeof(artifact_items[0]);

    vlacpp_model_params model_params = vlacpp_default_model_params();
    model_params.n_threads = args_.threads;
    vlacpp_status status = vlacpp_load_model(&artifacts, &model_params, &model_);
    if (status != VLACPP_STATUS_OK) {
        return;
    }

    vlacpp_context_params context_params = vlacpp_default_context_params();
    context_params.seed = args_.noise_seed >= 0 ? static_cast<uint32_t>(args_.noise_seed) : 0U;
    status = vlacpp_create_context(model_, &context_params, &context_);
    if (status != VLACPP_STATUS_OK) {
        return;
    }
    vlacpp_get_model_info(model_, &info_);
}

Pi0Model::~Pi0Model() {
    if (context_ != nullptr) {
        vlacpp_free_context(context_);
    }
    if (model_ != nullptr) {
        vlacpp_free_model(model_);
    }
}

const char * Pi0Model::type() const {
    return "pi0";
}

bool Pi0Model::predict(const observation & obs, model_result & out, std::string & error) {
    out = model_result{};
    if (model_ == nullptr || context_ == nullptr) {
        error = "Pi0 model is not initialized";
        return false;
    }
    if (obs.images.empty()) {
        error = "Pi0 requires at least one image";
        return false;
    }
    const model_image & image = obs.images[0];
    if (image.data == nullptr) {
        error = "Pi0 image data is null";
        return false;
    }

    vlacpp_image_view view{};
    view.name = "base_0_rgb";
    view.data = image.data;
    view.width = image.width;
    view.height = image.height;
    view.channels = image.channels;
    view.stride_bytes = image.stride_bytes;

    vlacpp_observation raw{};
    raw.images = &view;
    raw.image_count = 1;
    raw.state = obs.state.empty() ? nullptr : obs.state.data();
    raw.state_count = obs.state.size();
    raw.prompt = obs.task.c_str();

    vlacpp_action_chunk actions{};
    vlacpp_status status = vlacpp_infer_actions(context_, &raw, &actions);
    if (status != VLACPP_STATUS_OK) {
        error = last_error_or_status(status);
        return false;
    }

    out.chunk_size = actions.horizon;
    out.action_dim = actions.action_dim;
    const size_t count = static_cast<size_t>(actions.horizon) * static_cast<size_t>(actions.action_dim);
    out.actions.assign(actions.data, actions.data + count);
    vlacpp_free_action_chunk(&actions);

    vlacpp_infer_timings timings{};
    if (vlacpp_context_last_timings(context_, &timings) == VLACPP_STATUS_OK) {
        add_metric(out, "preprocess_ms", timings.preprocess_ms);
        add_metric(out, "prefix_ms", timings.prefix_ms);
        add_metric(out, "state_ms", timings.state_ms);
        add_metric(out, "denoise_ms", timings.denoise_ms);
        add_metric(out, "output_ms", timings.output_ms);
        add_metric(out, "model_total_ms", timings.total_ms);
    }
    return true;
}

void Pi0Model::reset() {
    if (context_ != nullptr) {
        vlacpp_reset_cache(context_);
    }
}

bool Pi0Model::is_ready() const {
    return model_ != nullptr && context_ != nullptr;
}

bool make_pi0_model(
    const model_args & args,
    std::unique_ptr<Model> & out,
    std::string & error) {
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
        error = last_error_or_status(VLACPP_STATUS_RUNTIME_ERROR);
        return false;
    }
    out = std::move(model);
    return true;
}

} // namespace robotcpp
