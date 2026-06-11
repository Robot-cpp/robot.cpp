#include "models/smolvla/smolvla_model.h"

#include "models/smolvla/smolvla_engine.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>

namespace robotcpp {
namespace {

void add_metric(model_result & out, const char * name, double value) {
    model_metric metric;
    metric.name = name;
    metric.value = value;
    out.metrics.push_back(metric);
}

bool validate_options(const model_args & options, std::string & error) {
    if (options.llm_path.empty()) {
        error = "SmolVLA llm_path is required";
        return false;
    }
    if (options.mmproj_path.empty()) {
        error = "SmolVLA mmproj_path is required";
        return false;
    }
    if (options.state_proj_path.empty()) {
        error = "SmolVLA state_proj_path is required";
        return false;
    }
    if (options.action_expert_path.empty()) {
        error = "SmolVLA action_expert_path is required";
        return false;
    }
    return true;
}

} // namespace

SmolVLAModel::SmolVLAModel(const model_args & args)
    : args_(args) {
    smolvla_params params = smolvla_default_params();
    params.llm_path = args_.llm_path.c_str();
    params.mmproj_path = args_.mmproj_path.c_str();
    params.state_proj_path = args_.state_proj_path.c_str();
    params.action_expert_path = args_.action_expert_path.c_str();
    params.n_threads = args_.threads;
    params.n_batch = args_.n_batch;
    params.n_ctx = args_.n_ctx;
    params.noise_mode = args_.noise_mode;
    params.noise_seed = args_.noise_seed;
    params.verbosity = args_.verbosity;

    ctx_ = smolvla_init(params);
}

SmolVLAModel::~SmolVLAModel() {
    if (ctx_ != nullptr) {
        smolvla_free(ctx_);
    }
}

const char * SmolVLAModel::type() const {
    return "smolvla";
}

bool SmolVLAModel::predict(const observation & obs, model_result & out, std::string & error) {
    out = model_result{};
    if (ctx_ == nullptr) {
        error = "SmolVLA model is not initialized";
        return false;
    }
    if (obs.images.size() != 1) {
        error = "SmolVLA currently requires exactly one image; received " + std::to_string(obs.images.size()) + " images";
        return false;
    }
    const model_image & image = obs.images[0];

    const smolvla_result result = smolvla_predict_raw_rgb(
        ctx_,
        image.data,
        image.width,
        image.height,
        image.channels,
        image.stride_bytes,
        obs.state.empty() ? nullptr : obs.state.data(),
        static_cast<int>(obs.state.size()),
        obs.task.empty() ? "grab the block." : obs.task.c_str());
    if (result.actions == nullptr) {
        error = "smolvla_predict_raw_rgb failed";
        return false;
    }

    out.chunk_size = result.chunk_size;
    out.action_dim = result.action_dim;
    out.actions.assign(
        result.actions,
        result.actions + static_cast<size_t>(result.chunk_size) * static_cast<size_t>(result.action_dim));

    const smolvla_stage_timings timings = smolvla_get_last_stage_timings(ctx_);
    add_metric(out, "vision_ms", timings.vision_ms);
    add_metric(out, "state_proj_ms", timings.state_proj_ms);
    add_metric(out, "llm_ms", timings.llm_ms);
    add_metric(out, "kv_extract_ms", timings.kv_extract_ms);
    add_metric(out, "phase2_ms", timings.phase2_ms);
    add_metric(out, "model_total_ms", timings.total_ms);
    return true;
}

void SmolVLAModel::reset() {}

bool SmolVLAModel::is_ready() const {
    return ctx_ != nullptr;
}

bool make_smolvla_model(
    const model_args & args,
    std::unique_ptr<Model> & out,
    std::string & error) {
    out.reset();
    if (!validate_options(args, error)) {
        return false;
    }

    std::unique_ptr<SmolVLAModel> model(new (std::nothrow) SmolVLAModel(args));
    if (!model) {
        error = "failed to allocate SmolVLA model";
        return false;
    }
    if (!model->is_ready()) {
        error = "failed to initialize SmolVLA model";
        return false;
    }
    out = std::move(model);
    return true;
}

} // namespace robotcpp
