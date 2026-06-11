#ifndef ROBOT_SERVER_CLIENT_CPP_MODEL_CLIENT_H
#define ROBOT_SERVER_CLIENT_CPP_MODEL_CLIENT_H

#include "protocol.h"

#include <cstdint>
#include <string>
#include <vector>

namespace robot_server {
namespace client {

struct ModelImage {
    const uint8_t * rgb_hwc_u8 = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride_bytes = 0;
    std::string name = "image";
};

struct ModelObservation {
    std::vector<ModelImage> images;
    std::vector<float> state;
    std::string prompt = "grab the block.";
};

struct ModelResponse {
    uint32_t chunk_size = 0;
    uint32_t action_dim = 0;
    std::vector<float> actions_flat;
    std::vector<robot_server::protocol::metric> metrics;

    const float * action_row(uint32_t index) const;
    double metric_value(const std::string & name, double fallback = 0.0) const;
};

class ModelClient {
public:
    ModelClient(std::string host = "127.0.0.1", uint16_t port = 5555);

    bool health(std::string & text, std::string & error);
    bool reset(std::string & text, std::string & error);
    bool shutdown(std::string & text, std::string & error);
    bool predict(const ModelObservation & obs, ModelResponse & response, std::string & error);

private:
    bool call(uint16_t op, const std::vector<uint8_t> & request_payload, std::vector<uint8_t> & response_payload, std::string & error);

    std::string host_;
    uint16_t port_ = 5555;
    uint32_t next_request_id_ = 1;
};

} // namespace client
} // namespace robot_server

#endif // ROBOT_SERVER_CLIENT_CPP_MODEL_CLIENT_H
