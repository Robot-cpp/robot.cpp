// benchmark-smolvla-cpp.cpp — warm-loop benchmark for the SmolVLA C++ engine
//
// This benchmark loads the engine once, optionally preloads image bytes once,
// and then measures repeated predict()/predict_bytes() calls in-process.

#include "smolvla_engine.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

static void quiet_llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

struct bench_args {
    std::string llm_path;
    std::string mmproj_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string image_path;
    std::string state_csv;
    std::string task = "grab the block.";
    std::string mode = "bytes";
    std::string result_tsv;

    int threads = 0;
    int warmup = 5;
    int loops = 100;
    int verbosity = 0;

    long long noise_seed = -1;
    int noise_mode = SMOLVLA_NOISE_MODE_GAUSSIAN;
};

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

static bool parse_state_csv(const std::string & csv, std::vector<float> & out) {
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

static bool read_file_bytes(const std::string & path, std::vector<unsigned char> & out) {
    out.clear();
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::fprintf(stderr, "Error: failed to open image file '%s'\n", path.c_str());
        return false;
    }

    fin.seekg(0, std::ios::end);
    const std::streampos size = fin.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "Error: image file '%s' is empty\n", path.c_str());
        return false;
    }
    fin.seekg(0, std::ios::beg);

    out.resize(static_cast<size_t>(size));
    fin.read(reinterpret_cast<char *>(out.data()), size);
    if (!fin) {
        std::fprintf(stderr, "Error: failed to read image file '%s'\n", path.c_str());
        return false;
    }
    return true;
}

static double percentile_ms(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double pos = (q / 100.0) * (values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s --llm <path> --mmproj <path> --image <path> [options]\n"
        "\n"
        "Required:\n"
        "  --llm <path>            LLM GGUF path\n"
        "  --mmproj <path>         Vision GGUF path\n"
        "  --image <path>          Input image path\n"
        "\n"
        "Optional:\n"
        "  --state-proj <path>     State projector GGUF path\n"
        "  --action-expert <path>  Action expert GGUF path\n"
        "  --state <csv>           State values (comma-separated)\n"
        "  --task <str>            Task instruction (default: \"grab the block.\")\n"
        "  --mode <path|bytes>     Benchmark predict() or predict_bytes() (default: bytes)\n"
        "  --threads <n>           CPU threads (default: auto)\n"
        "  --noise-mode <mode>     gaussian|debug-sin (default: gaussian)\n"
        "  --noise-seed <n>        RNG seed, <0 means auto (default: -1)\n"
        "  --warmup <n>            Warmup iterations (default: 5)\n"
        "  --loops <n>             Measured iterations (default: 100)\n"
        "  --result-tsv <path>     Optional TSV output file\n"
        "  -v, --verbose           Increase engine verbosity\n"
        "  -h, --help              Show this help\n",
        prog);
}

static bool parse_args(int argc, char ** argv, bench_args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbosity++;
        } else if (arg == "--llm" && i + 1 < argc) {
            args.llm_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (arg == "--state-proj" && i + 1 < argc) {
            args.state_proj_path = argv[++i];
        } else if (arg == "--action-expert" && i + 1 < argc) {
            args.action_expert_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (arg == "--state" && i + 1 < argc) {
            args.state_csv = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            args.task = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::atoi(argv[++i]);
        } else if (arg == "--noise-mode" && i + 1 < argc) {
            if (!parse_noise_mode(argv[++i], args.noise_mode)) {
                std::fprintf(stderr, "Error: invalid noise mode '%s'\n", argv[i]);
                return false;
            }
        } else if (arg == "--noise-seed" && i + 1 < argc) {
            args.noise_seed = std::atoll(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = std::atoi(argv[++i]);
        } else if (arg == "--loops" && i + 1 < argc) {
            args.loops = std::atoi(argv[++i]);
        } else if (arg == "--result-tsv" && i + 1 < argc) {
            args.result_tsv = argv[++i];
        } else {
            std::fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (args.llm_path.empty() || args.mmproj_path.empty() || args.image_path.empty()) {
        std::fprintf(stderr, "Error: --llm, --mmproj, and --image are required\n");
        return false;
    }
    if (args.mode != "path" && args.mode != "bytes") {
        std::fprintf(stderr, "Error: --mode must be 'path' or 'bytes'\n");
        return false;
    }
    if (args.warmup < 0 || args.loops <= 0) {
        std::fprintf(stderr, "Error: --warmup must be >= 0 and --loops must be > 0\n");
        return false;
    }
    return true;
}

struct stage_accum {
    double vision_ms = 0.0;
    double state_proj_ms = 0.0;
    double llm_ms = 0.0;
    double kv_extract_ms = 0.0;
    double phase2_ms = 0.0;
    double total_ms = 0.0;
};

static void append_result_tsv(
    const std::string & path,
    const bench_args & args,
    double init_ms,
    const std::vector<double> & samples_ms,
    const stage_accum & stage,
    float first_action) {
    if (path.empty()) {
        return;
    }

    const bool exists = std::ifstream(path).good();
    std::ofstream fout(path, std::ios::app);
    if (!fout) {
        std::fprintf(stderr, "Warning: failed to open result TSV '%s'\n", path.c_str());
        return;
    }

    if (!exists) {
        fout << "mode\tthreads\twarmup\tloops\tavg_ms\tp50_ms\tp95_ms\tmin_ms\tmax_ms\tinit_ms"
             << "\tvision_ms\tstate_proj_ms\tllm_ms\tkv_extract_ms\tphase2_ms\tstage_total_ms\tfirst_action0\n";
    }

    const auto mm = std::minmax_element(samples_ms.begin(), samples_ms.end());
    const double avg = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / samples_ms.size();

    fout << args.mode << '\t'
         << args.threads << '\t'
         << args.warmup << '\t'
         << args.loops << '\t'
         << std::fixed << std::setprecision(2)
         << avg << '\t'
         << percentile_ms(samples_ms, 50.0) << '\t'
         << percentile_ms(samples_ms, 95.0) << '\t'
         << *mm.first << '\t'
         << *mm.second << '\t'
         << init_ms << '\t'
         << (stage.vision_ms / samples_ms.size()) << '\t'
         << (stage.state_proj_ms / samples_ms.size()) << '\t'
         << (stage.llm_ms / samples_ms.size()) << '\t'
         << (stage.kv_extract_ms / samples_ms.size()) << '\t'
         << (stage.phase2_ms / samples_ms.size()) << '\t'
         << (stage.total_ms / samples_ms.size()) << '\t'
         << first_action << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    if (!env_flag_enabled("SMOLVLA_LLAMA_LOG")) {
        llama_log_set(quiet_llama_log_callback, nullptr);
    }

    bench_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<float> state_vec;
    if (!parse_state_csv(args.state_csv, state_vec)) {
        return 1;
    }

    std::vector<unsigned char> image_bytes;
    if (args.mode == "bytes" && !read_file_bytes(args.image_path, image_bytes)) {
        return 1;
    }

    smolvla_params params = smolvla_default_params();
    params.llm_path           = args.llm_path.c_str();
    params.mmproj_path        = args.mmproj_path.c_str();
    params.state_proj_path    = args.state_proj_path.empty() ? nullptr : args.state_proj_path.c_str();
    params.action_expert_path = args.action_expert_path.empty() ? nullptr : args.action_expert_path.c_str();
    params.task               = args.task.c_str();
    params.n_threads          = args.threads;
    params.noise_mode         = args.noise_mode;
    params.noise_seed         = static_cast<int64_t>(args.noise_seed);
    params.verbosity          = args.verbosity;

    const auto t_init0 = std::chrono::high_resolution_clock::now();
    smolvla_context * ctx = smolvla_init(params);
    const auto t_init1 = std::chrono::high_resolution_clock::now();
    if (!ctx) {
        std::fprintf(stderr, "Error: failed to initialize SmolVLA engine\n");
        return 1;
    }
    const double init_ms = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();

    auto run_predict = [&](smolvla_result & result) {
        if (args.mode == "bytes") {
            result = smolvla_predict_bytes(
                ctx,
                image_bytes.data(),
                static_cast<int>(image_bytes.size()),
                state_vec.empty() ? nullptr : state_vec.data(),
                static_cast<int>(state_vec.size()));
        } else {
            result = smolvla_predict(
                ctx,
                args.image_path.c_str(),
                state_vec.empty() ? nullptr : state_vec.data(),
                static_cast<int>(state_vec.size()));
        }
    };

    smolvla_result result = {};
    for (int i = 0; i < args.warmup; ++i) {
        run_predict(result);
        if (!result.actions) {
            std::fprintf(stderr, "Error: warmup predict failed at iteration %d\n", i);
            smolvla_free(ctx);
            return 1;
        }
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(args.loops));
    stage_accum stage_sum;
    float first_action = 0.0f;

    for (int i = 0; i < args.loops; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_predict(result);
        const auto t1 = std::chrono::high_resolution_clock::now();

        if (!result.actions) {
            std::fprintf(stderr, "Error: measured predict failed at iteration %d\n", i);
            smolvla_free(ctx);
            return 1;
        }

        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms.push_back(dt_ms);

        const smolvla_stage_timings stage = smolvla_get_last_stage_timings(ctx);
        stage_sum.vision_ms += stage.vision_ms;
        stage_sum.state_proj_ms += stage.state_proj_ms;
        stage_sum.llm_ms += stage.llm_ms;
        stage_sum.kv_extract_ms += stage.kv_extract_ms;
        stage_sum.phase2_ms += stage.phase2_ms;
        stage_sum.total_ms += stage.total_ms;

        if (i == 0 && result.chunk_size > 0 && result.action_dim > 0) {
            first_action = result.actions[0];
        }
    }

    smolvla_free(ctx);

    const auto mm = std::minmax_element(samples_ms.begin(), samples_ms.end());
    const double avg_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / samples_ms.size();
    const double p50_ms = percentile_ms(samples_ms, 50.0);
    const double p95_ms = percentile_ms(samples_ms, 95.0);
    const double loops_f = static_cast<double>(samples_ms.size());

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "mode=" << args.mode
              << " threads=" << args.threads
              << " warmup=" << args.warmup
              << " loops=" << args.loops << '\n';
    std::cout << "init_ms=" << init_ms
              << " avg_ms=" << avg_ms
              << " p50_ms=" << p50_ms
              << " p95_ms=" << p95_ms
              << " min_ms=" << *mm.first
              << " max_ms=" << *mm.second << '\n';
    std::cout << "stage_avg_ms"
              << " vision=" << (stage_sum.vision_ms / loops_f)
              << " state_proj=" << (stage_sum.state_proj_ms / loops_f)
              << " llm=" << (stage_sum.llm_ms / loops_f)
              << " kv_extract=" << (stage_sum.kv_extract_ms / loops_f)
              << " phase2=" << (stage_sum.phase2_ms / loops_f)
              << " total=" << (stage_sum.total_ms / loops_f) << '\n';
    std::cout << "first_action0=" << std::setprecision(6) << first_action << '\n';

    append_result_tsv(args.result_tsv, args, init_ms, samples_ms, stage_sum, first_action);
    return 0;
}
