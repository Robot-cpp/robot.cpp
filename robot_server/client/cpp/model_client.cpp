#include "model_client.h"

#include "socket.h"

#include <cstring>
#include <utility>

namespace proto = robot_server::protocol;
namespace sockets = robot_server::sockets;

namespace robot_server {
namespace client {
namespace {

bool send_message(
    sockets::socket_handle fd,
    uint16_t op,
    uint32_t request_id,
    const std::vector<uint8_t> & payload,
    std::string & error) {
    proto::header h;
    h.op = op;
    h.request_id = request_id;
    h.payload_len = payload.size();

    std::vector<uint8_t> header;
    proto::encode_header(h, header);
    if (!sockets::send_all(fd, header.data(), header.size(), error)) {
        return false;
    }
    if (!payload.empty() && !sockets::send_all(fd, payload.data(), payload.size(), error)) {
        return false;
    }
    return true;
}

bool recv_message(
    sockets::socket_handle fd,
    proto::header & h,
    std::vector<uint8_t> & payload,
    std::string & error) {
    std::vector<uint8_t> header(proto::k_header_size);
    if (!sockets::recv_exact(fd, header.data(), header.size(), error)) {
        return false;
    }
    if (!proto::decode_header(header.data(), header.size(), h, error)) {
        return false;
    }
    if (!proto::validate_header(h, proto::k_default_max_payload, error)) {
        return false;
    }

    payload.assign((size_t) h.payload_len, 0);
    if (!payload.empty() && !sockets::recv_exact(fd, payload.data(), payload.size(), error)) {
        return false;
    }
    return true;
}

bool make_predict_request(const ModelObservation & obs, proto::predict_request & req, std::string & error) {
    if (obs.images.empty()) {
        error = "observation.images must contain at least one image";
        return false;
    }

    req.task = obs.prompt;
    req.state = obs.state;
    req.images.clear();
    req.images.reserve(obs.images.size());

    for (size_t i = 0; i < obs.images.size(); ++i) {
        const ModelImage & src = obs.images[i];
        if (!src.rgb_hwc_u8 || src.width == 0 || src.height == 0) {
            error = "observation.images[" + std::to_string(i) + "] must be non-empty RAW_RGB_U8";
            return false;
        }

        const uint32_t stride = src.stride_bytes == 0 ? src.width * 3 : src.stride_bytes;
        if (stride < src.width * 3) {
            error = "observation.images[" + std::to_string(i) + "].stride_bytes is smaller than width * 3";
            return false;
        }

        const size_t image_size = (size_t) stride * (size_t) src.height;
        proto::image_payload image;
        image.name = src.name.empty() ? "image" + std::to_string(i) : src.name;
        image.image_format = proto::image_raw_rgb_u8;
        image.width = src.width;
        image.height = src.height;
        image.channels = 3;
        image.stride_bytes = stride;
        image.data.resize(image_size);
        std::memcpy(image.data.data(), src.rgb_hwc_u8, image_size);
        req.images.push_back(std::move(image));
    }
    return true;
}

} // namespace

const float * ModelResponse::action_row(uint32_t index) const {
    if (action_dim == 0 || index >= chunk_size) {
        return nullptr;
    }
    const size_t offset = (size_t) index * (size_t) action_dim;
    if (offset + action_dim > actions_flat.size()) {
        return nullptr;
    }
    return actions_flat.data() + offset;
}

double ModelResponse::metric_value(const std::string & name, double fallback) const {
    for (const proto::metric & metric : metrics) {
        if (metric.name == name) {
            return metric.value;
        }
    }
    return fallback;
}

ModelClient::ModelClient(std::string host, uint16_t port) : host_(std::move(host)), port_(port) {}

bool ModelClient::health(std::string & text, std::string & error) {
    std::vector<uint8_t> payload;
    if (!call(proto::op_health, {}, payload, error)) {
        return false;
    }
    return proto::decode_text_payload(payload, text, error);
}

bool ModelClient::reset(std::string & text, std::string & error) {
    std::vector<uint8_t> payload;
    if (!call(proto::op_reset, {}, payload, error)) {
        return false;
    }
    return proto::decode_text_payload(payload, text, error);
}

bool ModelClient::shutdown(std::string & text, std::string & error) {
    std::vector<uint8_t> payload;
    if (!call(proto::op_shutdown, {}, payload, error)) {
        return false;
    }
    return proto::decode_text_payload(payload, text, error);
}

bool ModelClient::predict(const ModelObservation & obs, ModelResponse & response, std::string & error) {
    proto::predict_request req;
    if (!make_predict_request(obs, req, error)) {
        return false;
    }

    std::vector<uint8_t> request_payload;
    if (!proto::encode_predict_request(req, request_payload, error)) {
        return false;
    }

    std::vector<uint8_t> response_payload;
    if (!call(proto::op_predict, request_payload, response_payload, error)) {
        return false;
    }

    proto::predict_response resp;
    if (!proto::decode_predict_response(response_payload, resp, error)) {
        return false;
    }

    response.chunk_size = resp.chunk_size;
    response.action_dim = resp.action_dim;
    response.actions_flat = std::move(resp.actions);
    response.metrics = std::move(resp.metrics);
    return true;
}

bool ModelClient::call(
    uint16_t op,
    const std::vector<uint8_t> & request_payload,
    std::vector<uint8_t> & response_payload,
    std::string & error) {
    if (!sockets::startup(error)) {
        return false;
    }

    sockets::socket_handle fd = sockets::tcp_connect(host_.c_str(), port_, error);
    if (fd == sockets::invalid_socket) {
        sockets::cleanup();
        return false;
    }

    const uint32_t request_id = next_request_id_++;
    bool ok = send_message(fd, op, request_id, request_payload, error);
    proto::header response_header;
    if (ok) {
        ok = recv_message(fd, response_header, response_payload, error);
    }
    sockets::close(fd);
    sockets::cleanup();
    if (!ok) {
        return false;
    }

    if (response_header.request_id != request_id) {
        error = "response request_id mismatch";
        return false;
    }
    if (response_header.status != proto::status_ok) {
        std::string text;
        std::string decode_error;
        proto::decode_text_payload(response_payload, text, decode_error);
        error = text.empty() ? "server returned error" : text;
        return false;
    }
    return true;
}

} // namespace client
} // namespace robot_server
