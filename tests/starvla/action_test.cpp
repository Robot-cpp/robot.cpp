#include "models/starvla/starvla_engine.h"

#include "stb_image.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct image_data {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
};

bool read_image(const std::string & path, image_data & image) {
    int channels = 0;
    unsigned char * pixels =
        stbi_load(path.c_str(), &image.width, &image.height, &channels, 3);
    if (pixels == nullptr) {
        return false;
    }
    image.pixels.assign(
        pixels, pixels + static_cast<size_t>(image.width) * image.height * 3);
    stbi_image_free(pixels);
    return true;
}

bool read_noise(const std::string & path, std::vector<float> & values) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }
    const std::streamsize bytes = stream.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        return false;
    }
    values.resize(static_cast<size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    return static_cast<bool>(
        stream.read(reinterpret_cast<char *>(values.data()), bytes));
}

bool parse_state(const std::string & csv, std::vector<float> & state) {
    std::stringstream stream(csv);
    std::string value;
    while (std::getline(stream, value, ',')) {
        try {
            state.push_back(std::stof(value));
        } catch (...) {
            return false;
        }
    }
    return true;
}

void print_actions(const char * name, const std::vector<float> & values,
                   int chunk_size, int action_dim) {
    std::cout << '"' << name << "\":[";
    for (int step = 0; step < chunk_size; ++step) {
        if (step != 0) {
            std::cout << ',';
        }
        std::cout << '[';
        for (int dim = 0; dim < action_dim; ++dim) {
            if (dim != 0) {
                std::cout << ',';
            }
            std::cout << values[static_cast<size_t>(step) * action_dim + dim];
        }
        std::cout << ']';
    }
    std::cout << ']';
}

void usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s --policy FILE [--llm FILE --mmproj FILE] [--load-only] "
        "[--image FILE --task TEXT --unnorm-key KEY --initial-noise FILE]\n",
        program);
}

} // namespace

int main(int argc, char ** argv) {
    robotcpp::starvla::StarVLAEngineConfig config;
    std::string image_path;
    std::string image_name = "image_0";
    std::string task = "grab the block.";
    std::string noise_path;
    std::string state_csv;
    bool load_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--policy" && i + 1 < argc) config.policy_path = argv[++i];
        else if (arg == "--llm" && i + 1 < argc) config.text_path_override = argv[++i];
        else if (arg == "--mmproj" && i + 1 < argc) config.mmproj_path_override = argv[++i];
        else if (arg == "--unnorm-key" && i + 1 < argc) config.unnorm_key = argv[++i];
        else if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--image-name" && i + 1 < argc) image_name = argv[++i];
        else if (arg == "--task" && i + 1 < argc) task = argv[++i];
        else if (arg == "--state" && i + 1 < argc) state_csv = argv[++i];
        else if (arg == "--initial-noise" && i + 1 < argc) noise_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) config.n_threads = std::stoi(argv[++i]);
        else if (arg == "--n-ctx" && i + 1 < argc) config.n_ctx = std::stoi(argv[++i]);
        else if (arg == "--n-batch" && i + 1 < argc) config.n_batch = std::stoi(argv[++i]);
        else if (arg == "--load-only") load_only = true;
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (config.policy_path.empty() || (!load_only && image_path.empty())) {
        usage(argv[0]);
        return 2;
    }

    std::string error;
    std::unique_ptr<robotcpp::starvla::StarVLAEngine> engine =
        robotcpp::starvla::StarVLAEngine::load(config, error);
    if (!engine) {
        std::fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }
    if (load_only) {
        std::cout << "{\"ready\":true}\n";
        return 0;
    }

    image_data image;
    robotcpp::observation observation;
    if (!read_image(image_path, image) || !parse_state(state_csv, observation.state)) {
        std::fprintf(stderr, "failed to read input\n");
        return 1;
    }
    observation.images.push_back({image_name, image.pixels.data(), image.width,
                                  image.height, 3, image.width * 3});
    observation.task = task;

    robotcpp::starvla::StarVLAEngineResult result;
    bool predicted = false;
    if (noise_path.empty()) {
        predicted = engine->predict(observation, result, error);
    } else {
        std::vector<float> noise;
        if (!read_noise(noise_path, noise)) {
            std::fprintf(stderr, "failed to read initial noise\n");
            return 1;
        }
        predicted = engine->predict_with_noise(observation, noise, result, error);
    }
    if (!predicted) {
        std::fprintf(stderr, "predict failed: %s\n", error.c_str());
        return 1;
    }

    std::cout << std::setprecision(9) << "{\"chunk_size\":" << result.chunk_size
              << ",\"action_dim\":" << result.action_dim << ',';
    print_actions("normalized_actions", result.normalized_actions,
                  result.chunk_size, result.action_dim);
    std::cout << ',';
    print_actions("actions", result.actions, result.chunk_size, result.action_dim);
    std::cout << "}\n";
    return 0;
}
