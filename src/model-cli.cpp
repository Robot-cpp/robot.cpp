// model-cli.cpp — common robotcpp::Model CLI frontend

#include "models/model.h"
#include "models/smolvla/smolvla_engine.h"
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct loaded_image {
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;
    int channels = 3;
    int stride_bytes = 0;
};

bool parse_model_type(const std::string & value, robotcpp::model_type & out) {
    if (value == "smolvla") {
        out = robotcpp::model_type::smolvla;
        return true;
    }
    return false;
}

bool parse_noise_mode(const std::string & value, int & out_mode) {
    if (value == "gaussian") {
        out_mode = SMOLVLA_NOISE_MODE_GAUSSIAN;
        return true;
    }
    if (value == "debug-sin" || value == "debug_sin" || value == "sin") {
        out_mode = SMOLVLA_NOISE_MODE_DEBUG_SIN;
        return true;
    }
    return false;
}

void print_usage(const char * prog) {
    std::fprintf(stderr, "\nModel CLI - robotcpp::Model frontend\n\n");
    std::fprintf(stderr, "Usage:\n");
    std::fprintf(stderr, "  %s --model-type smolvla [options]\n\n", prog);
    std::fprintf(stderr, "Common options:\n");
    std::fprintf(stderr, "  --model-type <type>      Model type (default: smolvla)\n");
    std::fprintf(stderr, "  --image <path>           Input image (JPEG/PNG)\n");
    std::fprintf(stderr, "  --state <csv>            Proprio/state values (comma-separated)\n");
    std::fprintf(stderr, "  --task <str>             Task instruction (default: \"grab the block.\")\n");
    std::fprintf(stderr, "  --threads <n>            Number of threads (default: auto)\n");
    std::fprintf(stderr, "  -v, --verbose            Increase verbosity\n");
    std::fprintf(stderr, "  -h, --help               Show this help\n\n");
    std::fprintf(stderr, "SmolVLA options:\n");
    std::fprintf(stderr, "  --llm <path>             LLM GGUF path\n");
    std::fprintf(stderr, "  --mmproj <path>          Vision GGUF path\n");
    std::fprintf(stderr, "  --state-proj <path>      State projector GGUF path\n");
    std::fprintf(stderr, "  --action-expert <path>   Action expert GGUF path\n");
    std::fprintf(stderr, "  --n-batch <n>            LLM batch size (default: 512)\n");
    std::fprintf(stderr, "  --n-ctx <n>              LLM context size (default: 2048)\n");
    std::fprintf(stderr, "  --noise-mode <mode>      gaussian|debug-sin (default: gaussian)\n");
    std::fprintf(stderr, "  --noise-seed <n>         Noise RNG seed, <0 means auto (default: -1)\n");
}

bool parse_state(const char * csv, std::vector<float> & out) {
    out.clear();
    if (!csv || std::strlen(csv) == 0) {
        return true;
    }

    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            out.push_back(std::stof(item));
        } catch (...) {
            std::fprintf(stderr, "Error: invalid state value '%s'\n", item.c_str());
            return false;
        }
    }
    return true;
}

bool load_rgb_image(const std::string & path, loaded_image & out) {
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char * pixels = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!pixels) {
        std::fprintf(stderr, "Error: failed to load image '%s'\n", path.c_str());
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 3;
    out.stride_bytes = w * out.channels;
    out.data.assign(pixels, pixels + (size_t) out.stride_bytes * (size_t) out.height);
    stbi_image_free(pixels);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    robotcpp::model_args args;
    std::string image_path;
    std::string state_csv;
    std::string task = "grab the block.";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbosity++;
        } else if (arg == "--model-type" && i + 1 < argc) {
            if (!parse_model_type(argv[++i], args.type)) {
                std::fprintf(stderr, "Error: unsupported model type '%s'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--llm" && i + 1 < argc) {
            args.llm_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (arg == "--state-proj" && i + 1 < argc) {
            args.state_proj_path = argv[++i];
        } else if (arg == "--action-expert" && i + 1 < argc) {
            args.action_expert_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--state" && i + 1 < argc) {
            state_csv = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            task = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::atoi(argv[++i]);
        } else if (arg == "--n-batch" && i + 1 < argc) {
            args.n_batch = std::atoi(argv[++i]);
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            args.n_ctx = std::atoi(argv[++i]);
        } else if (arg == "--noise-mode" && i + 1 < argc) {
            if (!parse_noise_mode(argv[++i], args.noise_mode)) {
                std::fprintf(stderr, "Error: invalid noise mode '%s'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--noise-seed" && i + 1 < argc) {
            args.noise_seed = std::atoll(argv[++i]);
        } else {
            std::fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (image_path.empty()) {
        std::fprintf(stderr, "Error: --image is required\n");
        print_usage(argv[0]);
        return 1;
    }

    std::vector<float> state_vec;
    if (!parse_state(state_csv.c_str(), state_vec)) {
        return 1;
    }

    loaded_image image;
    if (!load_rgb_image(image_path, image)) {
        return 1;
    }

    const auto init_start = std::chrono::high_resolution_clock::now();
    std::string error;
    std::unique_ptr<robotcpp::Model> model;
    if (!robotcpp::make_model(args, model, error)) {
        std::fprintf(stderr, "Error: failed to initialize model: %s\n", error.c_str());
        return 1;
    }
    const auto init_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_time = init_end - init_start;
    std::fprintf(stderr, "[model-cli] Model initialization: %.2f seconds\n", init_time.count());

    robotcpp::observation obs;
    robotcpp::model_image model_image;
    model_image.name = "image";
    model_image.data = image.data.data();
    model_image.width = image.width;
    model_image.height = image.height;
    model_image.channels = image.channels;
    model_image.stride_bytes = image.stride_bytes;
    obs.images.push_back(model_image);
    obs.state = state_vec;
    obs.task = task;

    const auto pred_start = std::chrono::high_resolution_clock::now();
    robotcpp::model_result result;
    if (!model->predict(obs, result, error)) {
        std::fprintf(stderr, "Error: prediction failed: %s\n", error.c_str());
        return 1;
    }
    const auto pred_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pred_time = pred_end - pred_start;
    std::fprintf(stderr, "[model-cli] Prediction: %.2f seconds\n", pred_time.count());

    if (result.actions.empty()) {
        std::fprintf(stderr, "Error: no action prediction result\n");
        return 1;
    }

    std::cout << "\n=== Predicted Actions ("
              << result.chunk_size << " steps x "
              << result.action_dim << " dims) ===\n";

    const int n_print = std::min(5, result.chunk_size);
    for (int t = 0; t < n_print; t++) {
        std::cout << "Step " << std::setw(2) << t << ": [";
        for (int d = 0; d < result.action_dim; d++) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(8) << result.actions[(size_t) t * (size_t) result.action_dim + (size_t) d];
            if (d < result.action_dim - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }

    if (result.chunk_size > 2 * n_print) {
        std::cout << "  ...\n";
    }

    if (result.chunk_size > n_print) {
        const int start_idx = std::max(n_print, result.chunk_size - n_print);
        for (int t = start_idx; t < result.chunk_size; t++) {
            std::cout << "Step " << std::setw(2) << t << ": [";
            for (int d = 0; d < result.action_dim; d++) {
                std::cout << std::fixed << std::setprecision(4)
                          << std::setw(8) << result.actions[(size_t) t * (size_t) result.action_dim + (size_t) d];
                if (d < result.action_dim - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]\n";
        }
    }

    return 0;
}
