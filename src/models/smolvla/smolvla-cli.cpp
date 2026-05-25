// smolvla-cli.cpp — CLI frontend for SmolVLA engine
//
// Usage:
//   smolvla-cli --vlm <vlm.gguf> --mmproj <vision.gguf> \
//     --state-proj <state.gguf> --action-expert <expert.gguf> \
//     --image <image.jpg> --state <csv_values> --task "grab the block."

#include "smolvla_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    fprintf(stderr, "   --vlm <path>           VLM GGUF path (smolvla-vlm-f16.gguf)\n");
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
    fprintf(stderr, "   %s --vlm smolvla-vlm-f16.gguf --mmproj mmproj-smolvla-f16.gguf \\\n", prog);
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

int main(int argc, char ** argv) {
    // Default parameters
    std::string vlm_path;
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
        else if (arg == "--vlm" && i + 1 < argc) {
            vlm_path = argv[++i];
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

    // Build engine params
    struct smolvla_params params = smolvla_default_params();
    params.vlm_path           = vlm_path.empty() ? nullptr : vlm_path.c_str();
    params.mmproj_path        = mmproj_path.empty() ? nullptr : mmproj_path.c_str();
    params.state_proj_path    = state_proj_path.empty() ? nullptr : state_proj_path.c_str();
    params.action_expert_path = action_expert_path.empty() ? nullptr : action_expert_path.c_str();
    params.task               = task.c_str();
    params.n_threads          = n_threads;
    params.action_dim         = action_dim;
    params.chunk_size         = chunk_size;
    params.num_steps          = num_steps;
    params.noise_mode         = noise_mode;
    params.noise_seed         = (int64_t) noise_seed;
    params.verbosity          = verbosity;

    // Initialize engine
    auto start = std::chrono::high_resolution_clock::now();

    struct smolvla_context * ctx = smolvla_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize SmolVLA engine\n");
        return 1;
    }

    auto init_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_time = init_end - start;
    fprintf(stderr, "[SmolVLA] Engine initialization: %.2f seconds\n", init_time.count());

    // Run prediction
    auto pred_start = std::chrono::high_resolution_clock::now();

    struct smolvla_result result = smolvla_predict(
        ctx,
        image_path.c_str(),
        state_vec.empty() ? nullptr : state_vec.data(),
        (int)state_vec.size());

    auto pred_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pred_time = pred_end - pred_start;
    fprintf(stderr, "[SmolVLA] Prediction: %.2f seconds\n", pred_time.count());

    // Print results
    if (result.actions) {
        std::cout << "\n=== Predicted Actions ("
                  << result.chunk_size << " steps x "
                  << result.action_dim << " dims) ===\n";

        // Print first few and last few chunks
        int n_print = std::min(5, result.chunk_size);
        for (int t = 0; t < n_print; t++) {
            std::cout << "Step " << std::setw(2) << t << ": [";
            for (int d = 0; d < result.action_dim; d++) {
                std::cout << std::fixed << std::setprecision(4)
                          << std::setw(8) << result.actions[t * result.action_dim + d];
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
                              << std::setw(8) << result.actions[t * result.action_dim + d];
                    if (d < result.action_dim - 1) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
    } else {
        fprintf(stderr, "Error: No action prediction result\n");
    }

    // Cleanup
    smolvla_free(ctx);

    return 0;
}
