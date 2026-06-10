// smolvla-cli.cpp — CLI frontend for SmolVLA engine
//
// Usage:
//   smolvla-cli --llm <llm.gguf> --mmproj <vision.gguf> \
//     --state-proj <state.gguf> --action-expert <expert.gguf> \
//     --image <image.jpg> --state <csv_values> --task "grab the block."

#include "smolvla_engine.h"

#include "models/model.h"
#include "models/smolvla/smolvla_model.h"
#include "stb_image.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>

static bool parse_noise_mode(const std::string & value, int & out_mode) {
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

static void print_usage(const char * prog) {
    fprintf(stderr, "\n SmolVLA CLI - Vision-Language-Action Model\n\n");
    fprintf(stderr, " Usage:\n");
    fprintf(stderr, "   %s [options]\n\n", prog);
    fprintf(stderr, " Required:\n");
    fprintf(stderr, "   --llm <path>           LLM GGUF path (smolvla-llm-f16.gguf)\n");
    fprintf(stderr, "   --mmproj <path>        Vision GGUF path (mmproj-smolvla-f16.gguf)\n");
    fprintf(stderr, "   --image <path>         Input image (JPEG/PNG)\n\n");
    fprintf(stderr, " Optional:\n");
    fprintf(stderr, "   --state-proj <path>    State projector GGUF path\n");
    fprintf(stderr, "   --action-expert <path> Action expert GGUF path\n");
    fprintf(stderr, "   --state <csv>          Proprio/state values (comma-separated)\n");
    fprintf(stderr, "   --task <str>           Task instruction (default: \"grab the block.\")\n");
    fprintf(stderr, "   --threads <n>          Number of threads (default: auto)\n");
    fprintf(stderr, "   --action-dim <n>       Action dimension (default: 6)\n");
    fprintf(stderr, "   --chunk-size <n>       Action chunk size (default: 50)\n");
    fprintf(stderr, "   --num-steps <n>        Denoising steps (default: 10)\n");
    fprintf(stderr, "   --noise-mode <mode>    gaussian|debug-sin (default: gaussian)\n");
    fprintf(stderr, "   --noise-seed <n>       Noise RNG seed, <0 means auto (default: -1)\n");
    fprintf(stderr, "   -v, --verbose          Increase verbosity\n");
    fprintf(stderr, "   -h, --help             Show this help\n\n");
    fprintf(stderr, " Example:\n");
    fprintf(stderr, "   %s --llm smolvla-llm-f16.gguf --mmproj mmproj-smolvla-f16.gguf \\\n", prog);
    fprintf(stderr, "     --state-proj state-proj-smolvla-f16.gguf \\\n");
    fprintf(stderr, "     --action-expert action-expert-smolvla-f16.gguf \\\n");
    fprintf(stderr, "     --image test.jpg --state \"0.5,-0.1,0.7,0.4,-0.8,0.95\" \\\n");
    fprintf(stderr, "     --task \"grab the block.\"\n\n");
}

static bool parse_state(const char * csv, std::vector<float> & out) {
    out.clear();
    if (!csv || strlen(csv) == 0) return true;

    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            float val = std::stof(item);
            out.push_back(val);
        } catch (...) {
            fprintf(stderr, "Error: Invalid state value '%s'\n", item.c_str());
            return false;
        }
    }
    return true;
}

struct loaded_image {
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;
    int channels = 3;
    int stride_bytes = 0;
};

static bool load_rgb_image(const std::string & path, loaded_image & out) {
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char * pixels = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!pixels) {
        fprintf(stderr, "Error: failed to load image '%s'\n", path.c_str());
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

int main(int argc, char ** argv) {
    // Default parameters
    std::string llm_path;
    std::string mmproj_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string image_path;
    std::string state_csv;
    std::string task = "grab the block.";
    int n_threads   = 0;
    int action_dim  = 6;
    int chunk_size  = 50;
    int num_steps   = 10;
    int noise_mode  = SMOLVLA_NOISE_MODE_GAUSSIAN;
    long long noise_seed = -1;
    int verbosity   = 1;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-v" || arg == "--verbose") {
            verbosity++;
        }
        else if (arg == "--llm" && i + 1 < argc) {
            llm_path = argv[++i];
        }
        else if (arg == "--mmproj" && i + 1 < argc) {
            mmproj_path = argv[++i];
        }
        else if (arg == "--state-proj" && i + 1 < argc) {
            state_proj_path = argv[++i];
        }
        else if (arg == "--action-expert" && i + 1 < argc) {
            action_expert_path = argv[++i];
        }
        else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        }
        else if (arg == "--state" && i + 1 < argc) {
            state_csv = argv[++i];
        }
        else if (arg == "--task" && i + 1 < argc) {
            task = argv[++i];
        }
        else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        }
        else if (arg == "--action-dim" && i + 1 < argc) {
            action_dim = std::atoi(argv[++i]);
        }
        else if (arg == "--chunk-size" && i + 1 < argc) {
            chunk_size = std::atoi(argv[++i]);
        }
        else if (arg == "--num-steps" && i + 1 < argc) {
            num_steps = std::atoi(argv[++i]);
        }
        else if (arg == "--noise-mode" && i + 1 < argc) {
            if (!parse_noise_mode(argv[++i], noise_mode)) {
                fprintf(stderr, "Error: Invalid noise mode '%s'\n", argv[i]);
                return 1;
            }
        }
        else if (arg == "--noise-seed" && i + 1 < argc) {
            noise_seed = std::atoll(argv[++i]);
        }
        else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (image_path.empty()) {
        fprintf(stderr, "Error: --image is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Parse state
    std::vector<float> state_vec;
    if (!parse_state(state_csv.c_str(), state_vec)) {
        return 1;
    }

    loaded_image image;
    if (!load_rgb_image(image_path, image)) {
        return 1;
    }

    robotcpp::smolvla_model_options smolvla_options;
    smolvla_options.llm_path = llm_path;
    smolvla_options.mmproj_path = mmproj_path;
    smolvla_options.state_proj_path = state_proj_path;
    smolvla_options.action_expert_path = action_expert_path;
    smolvla_options.task = task;
    smolvla_options.action_dim = action_dim;
    smolvla_options.chunk_size = chunk_size;
    smolvla_options.num_steps = num_steps;
    smolvla_options.noise_mode = noise_mode;
    smolvla_options.noise_seed = (int64_t) noise_seed;

    robotcpp::model_options model_options;
    model_options.type = robotcpp::model_type::smolvla;
    model_options.common.threads = n_threads;
    model_options.common.verbosity = verbosity;
    model_options.config = smolvla_options;

    // Initialize model through the common robotcpp dispatch path.
    auto start = std::chrono::high_resolution_clock::now();

    std::string error;
    std::unique_ptr<robotcpp::Model> model;
    if (!robotcpp::make_model(model_options, model, error)) {
        fprintf(stderr, "Failed to initialize SmolVLA model: %s\n", error.c_str());
        return 1;
    }

    auto init_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_time = init_end - start;
    fprintf(stderr, "[SmolVLA] Model initialization: %.2f seconds\n", init_time.count());

    // Run prediction
    robotcpp::observation obs;
    robotcpp::model_image model_image;
    model_image.name = "image";
    model_image.data = image.data.data();
    model_image.width = image.width;
    model_image.height = image.height;
    model_image.channels = image.channels;
    model_image.stride_bytes = image.stride_bytes;
    model_image.data_size = image.data.size();
    obs.images.push_back(model_image);
    obs.state = state_vec;
    obs.task = task;

    auto pred_start = std::chrono::high_resolution_clock::now();

    robotcpp::model_result result;
    if (!model->predict(obs, result, error)) {
        fprintf(stderr, "Error: %s\n", error.c_str());
        return 1;
    }

    auto pred_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pred_time = pred_end - pred_start;
    fprintf(stderr, "[SmolVLA] Prediction: %.2f seconds\n", pred_time.count());

    // Print results
    if (!result.actions.empty()) {
        std::cout << "\n=== Predicted Actions ("
                  << result.chunk_size << " steps x "
                  << result.action_dim << " dims) ===\n";

        // Print first few and last few chunks
        int n_print = std::min(5, result.chunk_size);
        for (int t = 0; t < n_print; t++) {
            std::cout << "Step " << std::setw(2) << t << ": [";
            for (int d = 0; d < result.action_dim; d++) {
                std::cout << std::fixed << std::setprecision(4)
                          << std::setw(8) << result.actions[(size_t) t * (size_t) result.action_dim + (size_t) d];
                if (d < result.action_dim - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }

        if (result.chunk_size > 2 * n_print) {
            std::cout << "  ...\n";
        }

        if (result.chunk_size > n_print) {
            int start_idx = std::max(n_print, result.chunk_size - n_print);
            for (int t = start_idx; t < result.chunk_size; t++) {
                std::cout << "Step " << std::setw(2) << t << ": [";
                for (int d = 0; d < result.action_dim; d++) {
                    std::cout << std::fixed << std::setprecision(4)
                              << std::setw(8) << result.actions[(size_t) t * (size_t) result.action_dim + (size_t) d];
                    if (d < result.action_dim - 1) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
    } else {
        fprintf(stderr, "Error: No action prediction result\n");
    }

    return 0;
}
