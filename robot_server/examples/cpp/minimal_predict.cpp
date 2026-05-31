#include "client/cpp/smolvla_client.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace client = robot_server::client;

namespace {

std::vector<uint8_t> make_random_rgb_image(uint32_t width, uint32_t height) {
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<uint8_t> image((size_t) width * (size_t) height * 3);
    for (uint8_t & pixel : image) {
        pixel = (uint8_t) dist(rng);
    }
    return image;
}

std::vector<float> make_random_state(uint32_t dim) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> state(dim);
    for (float & value : state) {
        value = dist(rng);
    }
    return state;
}

} // namespace

int main(int argc, char ** argv) {
    std::string host = "127.0.0.1";
    uint16_t port = 5555;
    if (argc > 1) {
        host = argv[1];
    }
    if (argc > 2) {
        port = (uint16_t) std::stoi(argv[2]);
    }

    const uint32_t width = 224;
    const uint32_t height = 224;
    std::vector<uint8_t> image = make_random_rgb_image(width, height);
    std::vector<float> state = make_random_state(6);

    client::SmolVLAObservation obs;
    obs.rgb_hwc_u8 = image.data();
    obs.width = width;
    obs.height = height;
    obs.stride_bytes = width * 3;
    obs.state = state.data();
    obs.state_dim = (uint32_t) state.size();
    obs.prompt = "grab the block.";

    client::SmolVLAClient smolvla(host, port);
    client::SmolVLAResponse response;
    std::string error;
    if (!smolvla.predict(obs, response, error)) {
        std::cerr << "predict failed: " << error << "\n";
        return 1;
    }

    const float * first_action = response.action_row(0);
    std::cout << "chunk_size: " << response.chunk_size << "\n";
    std::cout << "action_dim: " << response.action_dim << "\n";
    std::cout << "first_action:";
    for (uint32_t i = 0; first_action && i < response.action_dim; ++i) {
        std::cout << " " << std::fixed << std::setprecision(6) << first_action[i];
    }
    std::cout << "\n";
    std::cout << "model_total_ms: " << std::fixed << std::setprecision(2)
              << response.timing.model_total_ms << "\n";
    return 0;
}
