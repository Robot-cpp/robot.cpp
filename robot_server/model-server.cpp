#include "smolvla_engine.h"
#include "session.h"
#include "socket.h"
#include "model_adapter.h"

#include "llama.h"
#include "models/argument_parse.h"
#include "models/model.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

namespace sockets = robot_server::sockets;

namespace {

struct server_args {
    robotcpp::model_type model_type = robotcpp::model_type::smolvla;
    std::string llm_path;
    std::string mmproj_path;
    std::string vit_path;
    std::string tokenizer_path;
    std::string state_path;
    std::string action_decoder_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string policy_path;
    std::string task   = "grab the block.";
    std::string host   = "127.0.0.1";
    int port           = 5555;
    int threads        = 0;
    int n_batch        = 512;
    int n_ctx          = 2048;
    int noise_mode     = SMOLVLA_NOISE_MODE_GAUSSIAN;
    int64_t noise_seed = -1;
    int verbosity      = 0;
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
    (void)user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

static void print_usage(const char * prog) {
    std::fprintf(stderr,
                 "Usage: %s --model-type smolvla --llm <path> --mmproj <path> --state-proj <path> --action-expert "
                 "<path> [options]\n"
                 "       %s --model-type pi0 --vit <path> --mmproj <path> --llm <path> --tokenizer <path> --state-gguf "
                 "<path> --action-decoder <path> [options]\n"
                 "       %s --model-type starvla --llm <path> --mmproj <path> --policy <path> [options]\n"
                 "\n"
                 "Common options:\n"
                 "  --model-type <type>      smolvla|pi0|starvla\n"
                 "                           (default: smolvla)\n"
                 "\n"
                 "SmolVLA options:\n"
                 "  --llm <path>             LLM GGUF path\n"
                 "  --mmproj <path>          Vision GGUF path\n"
                 "  --state-proj <path>      State projector GGUF path\n"
                 "  --action-expert <path>   Action expert GGUF path\n"
                 "  --task <str>             Accepted for compatibility; predict request task is used\n"
                 "\n"
                 "Pi0 options:\n"
                 "  --vit <path>             ViT GGUF path\n"
                 "  --mmproj <path>          Merger GGUF path\n"
                 "  --llm <path>             LLM GGUF path\n"
                 "  --tokenizer <path>       Tokenizer GGUF path\n"
                 "  --state-gguf <path>      State projector GGUF path\n"
                 "  --action-decoder <path>  Action decoder GGUF path\n"
                 "\n"
                 "StarVLA options:\n"
                 "  --policy <path>          StarVLA policy GGUF path (required)\n"
                 "  --llm <path>             Qwen text GGUF path (required)\n"
                 "  --mmproj <path>          Qwen vision GGUF path (required)\n"
                 "\n"
                 "Runtime options:\n"
                 "  --host <ip>              Listen host (default: 127.0.0.1)\n"
                 "  --port <n>               Listen port (default: 5555)\n"
                 "  --threads <n>            CPU threads (default: auto)\n"
                 "  --n-batch <n>            LLM batch size (default: 512)\n"
                 "  --n-ctx <n>              LLM context size (default: 2048)\n"
                 "  --noise-mode <mode>      gaussian|debug-sin (default: gaussian)\n"
                 "  --noise-seed <n>         RNG seed, <0 means auto (default: -1)\n"
                 "  --verbosity <n>          Log verbosity (default: 0)\n"
                 "  -h, --help               Show this help\n",
                 prog, prog, prog);
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
        } else if (arg == "--model-type" && i + 1 < argc) {
            if (!robotcpp::parse_model_type(argv[++i], args.model_type)) {
                std::fprintf(stderr, "Error: unsupported model type '%s'\n", argv[i]);
                return false;
            }
        } else if (arg == "--policy" && i + 1 < argc) {
            args.policy_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (arg == "--vit" && i + 1 < argc) {
            args.vit_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (arg == "--state-gguf" && i + 1 < argc) {
            args.state_path = argv[++i];
        } else if (arg == "--action-decoder" && i + 1 < argc) {
            args.action_decoder_path = argv[++i];
        } else if (arg == "--state-proj" && i + 1 < argc) {
            args.state_proj_path = argv[++i];
        } else if (arg == "--action-expert" && i + 1 < argc) {
            args.action_expert_path = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            args.task = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            const char * value = argv[++i];
            if (!robotcpp::parse_integer_argument(value, args.port)) {
                std::fprintf(stderr, "Error: invalid --port value '%s'\n", value);
                return false;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            const char * value = argv[++i];
            if (!robotcpp::parse_integer_argument(value, args.threads)) {
                std::fprintf(stderr, "Error: invalid --threads value '%s'\n", value);
                return false;
            }
        } else if (arg == "--n-batch" && i + 1 < argc) {
            const char * value = argv[++i];
            if (!robotcpp::parse_integer_argument(value, args.n_batch)) {
                std::fprintf(stderr, "Error: invalid --n-batch value '%s'\n", value);
                return false;
            }
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            const char * value = argv[++i];
            if (!robotcpp::parse_integer_argument(value, args.n_ctx)) {
                std::fprintf(stderr, "Error: invalid --n-ctx value '%s'\n", value);
                return false;
            }
        } else if (arg == "--noise-mode" && i + 1 < argc) {
            if (!parse_noise_mode(argv[++i], args.noise_mode)) {
                std::fprintf(stderr, "Error: invalid noise mode '%s'\n", argv[i]);
                return false;
            }
        } else if (arg == "--noise-seed" && i + 1 < argc) {
            const char * value = argv[++i];
            if (!robotcpp::parse_integer_argument(value, args.noise_seed)) {
                std::fprintf(stderr, "Error: invalid --noise-seed value '%s'\n", value);
                return false;
            }
        } else if (arg == "--verbosity" && i + 1 < argc) {
            const char * value = argv[++i];
            if (!robotcpp::parse_integer_argument(value, args.verbosity)) {
                std::fprintf(stderr, "Error: invalid --verbosity value '%s'\n", value);
                return false;
            }
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
        std::fprintf(stderr, "Error: model-server only listens on 127.0.0.1 in this phase\n");
        return false;
    }
    if (args.threads < 0 || args.n_batch <= 0 || args.n_ctx <= 0 || args.verbosity < 0) {
        std::fprintf(stderr,
                     "Error: --threads/--verbosity must be non-negative and --n-batch/--n-ctx must be positive\n");
        return false;
    }
    if (robotcpp::is_starvla_model_type(args.model_type) &&
        args.noise_mode != SMOLVLA_NOISE_MODE_GAUSSIAN) {
        std::fprintf(stderr,
                     "Error: StarVLA does not support --noise-mode debug-sin; use Gaussian noise\n");
        return false;
    }
    if (args.model_type == robotcpp::model_type::smolvla) {
        if (args.llm_path.empty() || args.mmproj_path.empty() || args.state_proj_path.empty() ||
            args.action_expert_path.empty()) {
            std::fprintf(stderr, "Error: smolvla requires --llm --mmproj --state-proj --action-expert\n");
            return false;
        }
    } else if (args.model_type == robotcpp::model_type::pi0) {
        if (args.vit_path.empty() || args.mmproj_path.empty() || args.llm_path.empty() ||
            args.tokenizer_path.empty() || args.state_path.empty() || args.action_decoder_path.empty()) {
            std::fprintf(stderr,
                         "Error: pi0 requires --vit --mmproj --llm --tokenizer --state-gguf --action-decoder\n");
            return false;
        }
    } else if (robotcpp::is_starvla_model_type(args.model_type)) {
        if (args.llm_path.empty() || args.mmproj_path.empty() || args.policy_path.empty()) {
            std::fprintf(stderr, "Error: %s requires --llm --mmproj --policy\n",
                         robotcpp::model_type_name(args.model_type));
            return false;
        }
    } else {
        std::fprintf(stderr, "Error: unsupported model type '%s'\n", robotcpp::model_type_name(args.model_type));
        return false;
    }
    return true;
}

static robotcpp::model_args make_model_args(const server_args & args) {
    robotcpp::model_args model_args;
    model_args.type                = args.model_type;
    model_args.threads             = args.threads;
    model_args.verbosity           = args.verbosity;
    model_args.llm_path            = args.llm_path;
    model_args.mmproj_path         = args.mmproj_path;
    model_args.vit_path            = args.vit_path;
    model_args.tokenizer_path      = args.tokenizer_path;
    model_args.state_path          = args.state_path;
    model_args.action_decoder_path = args.action_decoder_path;
    model_args.state_proj_path     = args.state_proj_path;
    model_args.action_expert_path  = args.action_expert_path;
    model_args.policy_path         = args.policy_path;
    model_args.n_batch             = args.n_batch;
    model_args.n_ctx               = args.n_ctx;
    model_args.noise_mode          = args.noise_mode;
    model_args.noise_seed          = args.noise_seed;
    return model_args;
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

    std::unique_ptr<robotcpp::Model> model;
    if (!robotcpp::make_model(make_model_args(args), model, error)) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        sockets::cleanup();
        return 1;
    }
    robot_server::model_adapter adapter(std::move(model));

    sockets::socket_handle server = sockets::tcp_listen(args.host.c_str(), (uint16_t)args.port, 16, error);
    if (server == sockets::invalid_socket) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        sockets::cleanup();
        return 1;
    }

    std::fprintf(stderr, "[model-server] listening on %s:%d model=%s\n", args.host.c_str(), args.port, adapter.name());

    bool shutdown_requested = false;
    std::mutex predict_mutex;
    while (!shutdown_requested) {
        std::string peer;
        sockets::socket_handle client = sockets::tcp_accept(server, peer, error);
        if (client == sockets::invalid_socket) {
            std::fprintf(stderr, "[model-server] accept failed: %s\n", error.c_str());
            continue;
        }
        if (args.verbosity >= 1) {
            std::fprintf(stderr, "[model-server] client connected: %s\n", peer.c_str());
        }
        robot_server::handle_client(client, adapter, predict_mutex, shutdown_requested);
        sockets::close(client);
    }

    sockets::close(server);
    sockets::cleanup();
    return 0;
}
