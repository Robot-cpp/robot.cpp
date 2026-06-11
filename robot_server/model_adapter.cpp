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
    obs.images.reserve(req.images.size());
    for (const proto::image_payload & src : req.images) {
        robotcpp::model_image image;
        image.name = src.name.empty() ? "image" : src.name;
        image.data = src.data.empty() ? nullptr : src.data.data();
        image.width = static_cast<int>(src.width);
        image.height = static_cast<int>(src.height);
        image.channels = static_cast<int>(src.channels);
        image.stride_bytes = static_cast<int>(src.stride_bytes);
        image.data_size = src.data.size();
        obs.images.push_back(image);
    }
    obs.state = req.state;
    obs.task = task_;

    robotcpp::model_result result;
    if (!model_->predict(obs, result, error)) {
        return false;
    }

    resp.chunk_size = static_cast<uint32_t>(result.chunk_size);
    resp.action_dim = static_cast<uint32_t>(result.action_dim);
    resp.actions = std::move(result.actions);
    resp.metrics.reserve(result.metrics.size());
    for (const robotcpp::model_metric & src : result.metrics) {
        proto::metric dst;
        dst.name = src.name;
        dst.value = src.value;
        resp.metrics.push_back(std::move(dst));
    }
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

} // namespace robot_server
