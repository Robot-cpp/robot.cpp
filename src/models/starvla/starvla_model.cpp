#include "models/starvla/starvla_model.h"

#include "models/starvla/starvla_engine.h"

#include <memory>
#include <new>
#include <utility>

namespace robotcpp {
namespace {

void add_metric(model_result & out, const char * name, double value) {
    model_metric metric;
    metric.name = name;
    metric.value = value;
    out.metrics.push_back(std::move(metric));
}

} // namespace

StarVLAModel::StarVLAModel(std::unique_ptr<starvla::StarVLAEngine> engine)
    : engine_(std::move(engine)) {}

StarVLAModel::~StarVLAModel() = default;

const char * StarVLAModel::type() const {
    return "starvla";
}

bool StarVLAModel::predict(const observation & obs, model_result & out, std::string & error) {
    out = model_result{};
    error.clear();
    if (engine_ == nullptr) {
        error = "StarVLA model is not initialized";
        return false;
    }

    starvla::StarVLAEngineResult result;
    const bool predicted = obs.initial_noise.empty()
                               ? engine_->predict(obs, result, error)
                               : engine_->predict_with_noise(obs, obs.initial_noise, result, error);
    if (!predicted) {
        return false;
    }
    out.actions = std::move(result.actions);
    out.chunk_size = result.chunk_size;
    out.action_dim = result.action_dim;
    add_metric(out, "image_preprocess_ms", result.timings.image_preprocess_ms);
    add_metric(out, "prompt_ms", result.timings.prompt_ms);
    add_metric(out, "qwen3vl_ms", result.timings.qwen3vl_ms);
    add_metric(out, "policy_ms", result.timings.policy_ms);
    add_metric(out, "unnormalize_ms", result.timings.unnormalize_ms);
    add_metric(out, "model_total_ms", result.timings.total_ms);
    return true;
}

void StarVLAModel::reset() {
    if (engine_ != nullptr) {
        engine_->reset();
    }
}

bool make_starvla_model(const model_args & args, std::unique_ptr<Model> & out,
                        std::string & error) {
    out.reset();
    error.clear();
    if (!is_starvla_model_type(args.type)) {
        error = std::string("model type '") + model_type_name(args.type) +
                "' is not a StarVLA model type";
        return false;
    }
    if (args.noise_mode != 0) {
        error = "StarVLA does not support SmolVLA --noise-mode debug-sin; use Gaussian noise and --noise-seed";
        return false;
    }
    starvla::StarVLAEngineConfig config;
    config.policy_path = args.policy_path;
    config.text_path_override = args.llm_path;
    config.mmproj_path_override = args.mmproj_path;
    config.unnorm_key = args.unnorm_key;
    config.n_threads = args.threads;
    config.n_ctx = args.n_ctx;
    config.n_batch = args.n_batch;
    config.noise_seed = args.noise_seed;
    config.verbosity = args.verbosity;
    std::unique_ptr<starvla::StarVLAEngine> engine =
        starvla::StarVLAEngine::load(config, error);
    if (engine == nullptr) {
        return false;
    }

    std::unique_ptr<StarVLAModel> model(
        new (std::nothrow) StarVLAModel(std::move(engine)));
    if (model == nullptr) {
        error = "failed to allocate StarVLA model";
        return false;
    }
    out = std::move(model);
    return true;
}

} // namespace robotcpp
