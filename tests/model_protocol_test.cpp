#include "protocol.h"

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

int main() {
    namespace protocol = robot_server::protocol;
    protocol::predict_request request;
    protocol::image_payload image;
    image.name = "image_0";
    image.width = 2;
    image.height = 1;
    image.channels = 3;
    image.stride_bytes = 6;
    image.data = {0, 1, 2, 3, 4, 5};
    request.images.push_back(image);
    request.state = {1.0f, -2.0f};
    request.initial_noise = {0.25f, -0.5f, 1.5f};
    request.task = "test task";

    std::vector<uint8_t> payload;
    std::string error;
    if (!protocol::encode_predict_request(request, payload, error)) {
        std::fprintf(stderr, "encode failed: %s\n", error.c_str());
        return 1;
    }
    protocol::predict_request decoded;
    if (!protocol::decode_predict_request(payload, decoded, error) ||
        decoded.images.size() != 1 || decoded.images[0].data != image.data ||
        decoded.state != request.state || decoded.initial_noise != request.initial_noise ||
        decoded.task != request.task) {
        std::fprintf(stderr, "roundtrip failed: %s\n", error.c_str());
        return 1;
    }
    payload.push_back(0);
    if (protocol::decode_predict_request(payload, decoded, error) ||
        error != "trailing bytes in predict request") {
        std::fprintf(stderr, "trailing-byte validation failed\n");
        return 1;
    }
    request.initial_noise = {std::numeric_limits<float>::infinity()};
    if (protocol::encode_predict_request(request, payload, error) ||
        error != "initial noise contains a non-finite value") {
        std::fprintf(stderr, "non-finite noise validation failed\n");
        return 1;
    }
    return 0;
}
