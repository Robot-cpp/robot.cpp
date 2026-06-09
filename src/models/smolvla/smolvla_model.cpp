#include "models/smolvla/smolvla_model.h"

#include "models/smolvla/smolvla_engine.h"

#include <cstddef>
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

bool validate_options(const smolvla_model_options & options, std::string & error) {
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

SmolVLAModel::SmolVLAModel(const smolvla_model_options & options, const common_options & common)
    : options_(options), common_(common) {
    smolvla_params params = smolvla_default_params();
    params.llm_path = options_.llm_path.c_str();
    params.mmproj_path = options_.mmproj_path.c_str();
    params.state_proj_path = options_.state_proj_path.c_str();
    params.action_expert_path = options_.action_expert_path.c_str();
    params.task = options_.task.c_str();
    params.n_threads = common_.threads;
    params.n_batch = options_.n_batch;
    params.n_ctx = options_.n_ctx;
    params.action_dim = options_.action_dim;
    params.chunk_size = options_.chunk_size;
    params.num_steps = options_.num_steps;
    params.noise_mode = options_.noise_mode;
    params.noise_seed = options_.noise_seed;
    params.verbosity = common_.verbosity;

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
        error = "SmolVLA currently requires exactly one image";
        return false;
    }
    if (!obs.task.empty() && obs.task != options_.task) {
        error = "SmolVLA dynamic per-request task is not supported yet";
        return false;
    }

    const model_image & image = obs.images[0];
    if (image.data == nullptr || image.width <= 0 || image.height <= 0 || image.channels != 3) {
        error = "SmolVLA requires one valid RGB uint8 image";
        return false;
    }

    const smolvla_result result = smolvla_predict_raw_rgb(
        ctx_,
        image.data,
        image.width,
        image.height,
        image.channels,
        image.stride_bytes,
        obs.state.empty() ? nullptr : obs.state.data(),
        static_cast<int>(obs.state.size()));
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
    const smolvla_model_options & options,
    const common_options & common,
    std::unique_ptr<Model> & out,
    std::string & error) {
    out.reset();
    if (!validate_options(options, error)) {
        return false;
    }

    std::unique_ptr<SmolVLAModel> model(new (std::nothrow) SmolVLAModel(options, common));
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
