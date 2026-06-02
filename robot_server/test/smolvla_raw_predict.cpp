#include "smolvla_engine.h"

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct args_t {
    std::string vlm_path;
    std::string mmproj_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string raw_rgb_path;
    std::string dump_dir;
    std::string state_csv;
    std::string task = "grab the block.";
    int width = 0;
    int height = 0;
    int channels = 3;
    int stride_bytes = 0;
    int threads = 0;
    int action_dim = 6;
    int chunk_size = 50;
    int num_steps = 10;
    int noise_mode = SMOLVLA_NOISE_MODE_GAUSSIAN;
    int64_t noise_seed = -1;
    int verbosity = 0;
};

static void quiet_llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

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

static bool parse_state(const std::string & csv, std::vector<float> & out) {
    out.clear();
    if (csv.empty()) {
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

static bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::fprintf(stderr, "Error: failed to open '%s'\n", path.c_str());
        return false;
    }
    fin.seekg(0, std::ios::end);
    const std::streampos size = fin.tellg();
    if (size < 0) {
        std::fprintf(stderr, "Error: failed to stat '%s'\n", path.c_str());
        return false;
    }
    fin.seekg(0, std::ios::beg);
    out.assign((size_t) size, 0);
    if (!out.empty()) {
        fin.read((char *) out.data(), size);
    }
    return (bool) fin || fin.eof();
}

static bool write_actions_dump(const std::string & dir, const smolvla_result & result) {
    if (dir.empty()) {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "Error: failed to create dump dir '%s'\n", dir.c_str());
        return false;
    }

    {
        std::ofstream meta(dir + "/meta.txt", std::ios::binary);
        if (!meta) {
            std::fprintf(stderr, "Error: failed to write meta.txt\n");
            return false;
        }
        meta << "chunk_size " << result.chunk_size << "\n";
        meta << "action_dim " << result.action_dim << "\n";
    }
    {
        std::ofstream fout(dir + "/final_actions.bin", std::ios::binary);
        if (!fout) {
            std::fprintf(stderr, "Error: failed to write final_actions.bin\n");
            return false;
        }
        fout.write((const char *) result.actions, (size_t) result.chunk_size * (size_t) result.action_dim * sizeof(float));
    }
    return true;
}

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s --vlm <path> --mmproj <path> --raw-rgb <path> --width <w> --height <h> [options]\n"
        "  --state-proj <path>      State projector GGUF path\n"
        "  --action-expert <path>   Action expert GGUF path\n"
        "  --state <csv>            State values\n"
        "  --task <str>             Task instruction\n"
        "  --channels <n>           Must be 3 (default: 3)\n"
        "  --stride-bytes <n>       Row stride, <=0 means width*3\n"
        "  --threads <n>            CPU threads\n"
        "  --action-dim <n>         Action dimension\n"
        "  --chunk-size <n>         Action chunk size\n"
        "  --num-steps <n>          Denoising steps\n"
        "  --noise-mode <mode>      gaussian|debug-sin\n"
        "  --noise-seed <n>         RNG seed\n"
        "  --dump-dir <path>        Write meta.txt and final_actions.bin\n"
        "  --verbosity <n>          Engine verbosity\n",
        prog);
}

static bool parse_args(int argc, char ** argv, args_t & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--vlm" && i + 1 < argc) {
            args.vlm_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (arg == "--state-proj" && i + 1 < argc) {
            args.state_proj_path = argv[++i];
        } else if (arg == "--action-expert" && i + 1 < argc) {
            args.action_expert_path = argv[++i];
        } else if (arg == "--raw-rgb" && i + 1 < argc) {
            args.raw_rgb_path = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            args.width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            args.height = std::atoi(argv[++i]);
        } else if (arg == "--channels" && i + 1 < argc) {
            args.channels = std::atoi(argv[++i]);
        } else if (arg == "--stride-bytes" && i + 1 < argc) {
            args.stride_bytes = std::atoi(argv[++i]);
        } else if (arg == "--state" && i + 1 < argc) {
            args.state_csv = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            args.task = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::atoi(argv[++i]);
        } else if (arg == "--action-dim" && i + 1 < argc) {
            args.action_dim = std::atoi(argv[++i]);
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            args.chunk_size = std::atoi(argv[++i]);
        } else if (arg == "--num-steps" && i + 1 < argc) {
            args.num_steps = std::atoi(argv[++i]);
        } else if (arg == "--noise-mode" && i + 1 < argc) {
            if (!parse_noise_mode(argv[++i], args.noise_mode)) {
                std::fprintf(stderr, "Error: invalid noise mode '%s'\n", argv[i]);
                return false;
            }
        } else if (arg == "--noise-seed" && i + 1 < argc) {
            args.noise_seed = (int64_t) std::atoll(argv[++i]);
        } else if (arg == "--dump-dir" && i + 1 < argc) {
            args.dump_dir = argv[++i];
        } else if (arg == "--verbosity" && i + 1 < argc) {
            args.verbosity = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }
    if (args.vlm_path.empty() || args.mmproj_path.empty() || args.raw_rgb_path.empty() ||
        args.width <= 0 || args.height <= 0) {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    llama_log_set(quiet_llama_log_callback, nullptr);

    args_t args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint8_t> raw;
    if (!read_file_bytes(args.raw_rgb_path, raw)) {
        return 1;
    }

    const int stride = args.stride_bytes <= 0 ? args.width * args.channels : args.stride_bytes;
    const size_t min_len = (size_t) stride * (size_t) args.height;
    if (raw.size() < min_len) {
        std::fprintf(stderr, "Error: raw RGB file too small: got %zu need at least %zu\n", raw.size(), min_len);
        return 1;
    }

    std::vector<float> state;
    if (!parse_state(args.state_csv, state)) {
        return 1;
    }

    smolvla_params params = smolvla_default_params();
    params.vlm_path = args.vlm_path.c_str();
    params.mmproj_path = args.mmproj_path.c_str();
    params.state_proj_path = args.state_proj_path.empty() ? nullptr : args.state_proj_path.c_str();
    params.action_expert_path = args.action_expert_path.empty() ? nullptr : args.action_expert_path.c_str();
    params.task = args.task.c_str();
    params.n_threads = args.threads;
    params.action_dim = args.action_dim;
    params.chunk_size = args.chunk_size;
    params.num_steps = args.num_steps;
    params.noise_mode = args.noise_mode;
    params.noise_seed = args.noise_seed;
    params.verbosity = args.verbosity;

    smolvla_context * ctx = smolvla_init(params);
    if (!ctx) {
        std::fprintf(stderr, "Error: failed to initialize SmolVLA engine\n");
        return 1;
    }

    smolvla_result result = smolvla_predict_raw_rgb(
        ctx,
        raw.data(),
        args.width,
        args.height,
        args.channels,
        args.stride_bytes,
        state.empty() ? nullptr : state.data(),
        (int) state.size());
    if (!result.actions) {
        std::fprintf(stderr, "Error: raw predict failed\n");
        smolvla_free(ctx);
        return 1;
    }

    std::cout << "chunk_size=" << result.chunk_size
              << " action_dim=" << result.action_dim
              << " first_action=" << std::fixed << std::setprecision(6) << result.actions[0] << "\n";

    const bool dump_ok = write_actions_dump(args.dump_dir, result);
    smolvla_free(ctx);
    return dump_ok ? 0 : 1;
}
