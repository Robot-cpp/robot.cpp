#include "smolvla_engine.h"
#include "session.h"
#include "socket.h"
#include "vla_policy.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

namespace sockets = robot_server::sockets;

namespace {

struct server_args {
    std::string llm_path;
    std::string mmproj_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string task = "grab the block.";
    std::string host = "127.0.0.1";
    int port = 5555;
    int threads = 0;
    int n_batch = 512;
    int n_ctx = 2048;
    int action_dim = 6;
    int chunk_size = 50;
    int num_steps = 10;
    int noise_mode = SMOLVLA_NOISE_MODE_GAUSSIAN;
    int64_t noise_seed = -1;
    int verbosity = 1;
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

static void quiet_llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s --llm <path> --mmproj <path> [options]\n"
        "\n"
        "Model options:\n"
        "  --llm <path>             LLM GGUF path\n"
        "  --mmproj <path>          Vision GGUF path\n"
        "  --state-proj <path>      State projector GGUF path\n"
        "  --action-expert <path>   Action expert GGUF path\n"
        "  --task <str>             Task instruction\n"
        "\n"
        "Runtime options:\n"
        "  --host <ip>              Listen host (default: 127.0.0.1)\n"
        "  --port <n>               Listen port (default: 5555)\n"
        "  --threads <n>            CPU threads (default: auto)\n"
        "  --n-batch <n>            LLM batch size (default: 512)\n"
        "  --n-ctx <n>              LLM context size (default: 2048)\n"
        "  --action-dim <n>         Action dimension (default: 6)\n"
        "  --chunk-size <n>         Action chunk size (default: 50)\n"
        "  --noise-mode <mode>      gaussian|debug-sin (default: gaussian)\n"
        "  --noise-seed <n>         RNG seed, <0 means auto (default: -1)\n"
        "  --verbosity <n>          Log verbosity (default: 1)\n"
        "  -h, --help               Show this help\n",
        prog);
}

// TODO: may need to be cleaned up and optimized
static bool parse_args(int argc, char ** argv, server_args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--llm" && i + 1 < argc) {
            args.llm_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (arg == "--state-proj" && i + 1 < argc) {
            args.state_proj_path = argv[++i];
        } else if (arg == "--action-expert" && i + 1 < argc) {
            args.action_expert_path = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            args.task = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            args.port = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::atoi(argv[++i]);
        } else if (arg == "--n-batch" && i + 1 < argc) {
            args.n_batch = std::atoi(argv[++i]);
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            args.n_ctx = std::atoi(argv[++i]);
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
        } else if (arg == "--verbosity" && i + 1 < argc) {
            args.verbosity = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }
    if (args.port <= 0 || args.port > 65535) {
        std::fprintf(stderr, "Error: --port must be in 1..65535\n");
        return false;
    }
    if (args.host != "127.0.0.1") {
        std::fprintf(stderr, "Error: smolvla-server only listens on 127.0.0.1 in this phase\n");
        return false;
    }
    if (args.llm_path.empty() || args.mmproj_path.empty()) {
        std::fprintf(stderr, "Error: --llm and --mmproj are required\n");
        return false;
    }
    return true;
}

static robot_server::smolvla_policy_options make_smolvla_options(const server_args & args) {
    robot_server::smolvla_policy_options options;
    options.llm_path = args.llm_path;
    options.mmproj_path = args.mmproj_path;
    options.state_proj_path = args.state_proj_path;
    options.action_expert_path = args.action_expert_path;
    options.task = args.task;
    options.threads = args.threads;
    options.n_batch = args.n_batch;
    options.n_ctx = args.n_ctx;
    options.action_dim = args.action_dim;
    options.chunk_size = args.chunk_size;
    options.num_steps = args.num_steps;
    options.noise_mode = args.noise_mode;
    options.noise_seed = args.noise_seed;
    options.verbosity = args.verbosity;
    return options;
}

} // namespace

int main(int argc, char ** argv) {
    llama_log_set(quiet_llama_log_callback, nullptr);

    server_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    std::string error;
    if (!sockets::startup(error)) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        return 1;
    }

    std::unique_ptr<robot_server::vla_policy> policy = robot_server::make_smolvla_policy(make_smolvla_options(args), error);
    if (!policy) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        sockets::cleanup();
        return 1;
    }

    sockets::socket_handle server = sockets::tcp_listen(args.host.c_str(), (uint16_t) args.port, 16, error);
    if (server == sockets::invalid_socket) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        sockets::cleanup();
        return 1;
    }

    std::fprintf(stderr, "[SmolVLA server] listening on %s:%d policy=%s\n",
                 args.host.c_str(), args.port, policy->name());

    bool shutdown_requested = false;
    std::mutex predict_mutex;
    while (!shutdown_requested) {
        std::string peer;
        sockets::socket_handle client = sockets::tcp_accept(server, peer, error);
        if (client == sockets::invalid_socket) {
            std::fprintf(stderr, "[SmolVLA server] accept failed: %s\n", error.c_str());
            continue;
        }
        if (args.verbosity >= 1) {
            std::fprintf(stderr, "[SmolVLA server] client connected: %s\n", peer.c_str());
        }
        robot_server::handle_client(client, *policy, predict_mutex, shutdown_requested);
        sockets::close(client);
    }

    sockets::close(server);
    sockets::cleanup();
    return 0;
}
