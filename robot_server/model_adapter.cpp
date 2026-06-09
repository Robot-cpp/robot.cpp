#include "model_adapter.h"

#include "models/model.h"

#include <utility>

namespace proto = smolvla::protocol;

namespace robot_server {

model_adapter::model_adapter(std::unique_ptr<robotcpp::Model> model) : model_(std::move(model)) {}

model_adapter::~model_adapter() = default;

bool model_adapter::predict(const proto::predict_request & req, proto::predict_response & resp, std::string & error) {
    if (!model_) {
        error = "model adapter is not initialized";
        return false;
    }

    robotcpp::observation obs;
    robotcpp::model_image image;
    image.name = "image";
    image.data = req.image.empty() ? nullptr : req.image.data();
    image.width = static_cast<int>(req.width);
    image.height = static_cast<int>(req.height);
    image.channels = static_cast<int>(req.channels);
    image.stride_bytes = static_cast<int>(req.stride_bytes);
    obs.images.push_back(image);
    obs.state = req.state;
    obs.task = task_;

    robotcpp::model_result result;
    if (!model_->predict(obs, result, error)) {
        return false;
    }

    resp.chunk_size = static_cast<uint32_t>(result.chunk_size);
    resp.action_dim = static_cast<uint32_t>(result.action_dim);
    resp.actions = std::move(result.actions);

    resp.timing.vision_ms = metric_value(result, "vision_ms");
    resp.timing.state_proj_ms = metric_value(result, "state_proj_ms");
    resp.timing.llm_ms = metric_value(result, "llm_ms");
    resp.timing.kv_extract_ms = metric_value(result, "kv_extract_ms");
    resp.timing.phase2_ms = metric_value(result, "phase2_ms");
    resp.timing.model_total_ms = metric_value(result, "model_total_ms");
    return true;
}

void model_adapter::reset() {
    if (model_) {
        model_->reset();
    }
}

const char * model_adapter::name() const {
    return model_ ? model_->type() : "invalid";
}

void model_adapter::set_task(std::string task) {
    task_ = std::move(task);
}

double model_adapter::metric_value(const robotcpp::model_result & result, const char * name) {
    for (const robotcpp::model_metric & metric : result.metrics) {
        if (metric.name == name) {
            return metric.value;
        }
    }
    return 0.0;
}

} // namespace robot_server
