#include "vlacpp.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " --vit vit.gguf --mmproj mmproj.gguf --llm llm.gguf --tokenizer tokenizer.gguf"
              << " --state-gguf state.gguf --action-decoder action_decoder.gguf"
              << " [--prompt text] [--state v0,v1,...] [--steps n] [--seed n]"
              << " [--backend cpu|cuda] [--vit-dtype fp32|f16|bf16] [--mmproj-dtype fp32|f16|bf16]"
              << " [--llm-dtype fp32|f16|bf16] [--state-dtype fp32|f16|bf16]"
              << " [--action-decoder-dtype fp32|f16|bf16] [--info]\n";
}

std::vector<float> parse_state(const std::string & text) {
    std::vector<float> result;
    size_t begin = 0;
    while (begin < text.size()) {
        size_t end = text.find(',', begin);
        std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        result.push_back(std::strtof(item.c_str(), nullptr));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    std::string vit_path;
    std::string mmproj_path;
    std::string llm_path;
    std::string tokenizer_path;
    std::string state_path;
    std::string action_decoder_path;
    std::string prompt;
    std::string state_text;
    std::string vit_dtype;
    std::string mmproj_dtype;
    std::string llm_dtype;
    std::string state_dtype;
    std::string action_decoder_dtype;
    int steps = 10;
    uint32_t seed = 1;
    vlacpp_backend backend = VLACPP_BACKEND_CPU;
    bool info = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vit") == 0 && i + 1 < argc) {
            vit_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (std::strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            llm_path = argv[++i];
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--state-gguf") == 0 && i + 1 < argc) {
            state_path = argv[++i];
        } else if (std::strcmp(argv[i], "--action-decoder") == 0 && i + 1 < argc) {
            action_decoder_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--state") == 0 && i + 1 < argc) {
            state_text = argv[++i];
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (std::strcmp(value, "cpu") == 0) {
                backend = VLACPP_BACKEND_CPU;
            } else if (std::strcmp(value, "cuda") == 0) {
                backend = VLACPP_BACKEND_CUDA;
            } else {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--vit-dtype") == 0 && i + 1 < argc) {
            vit_dtype = argv[++i];
        } else if (std::strcmp(argv[i], "--mmproj-dtype") == 0 && i + 1 < argc) {
            mmproj_dtype = argv[++i];
        } else if (std::strcmp(argv[i], "--llm-dtype") == 0 && i + 1 < argc) {
            llm_dtype = argv[++i];
        } else if (std::strcmp(argv[i], "--state-dtype") == 0 && i + 1 < argc) {
            state_dtype = argv[++i];
        } else if (std::strcmp(argv[i], "--action-decoder-dtype") == 0 && i + 1 < argc) {
            action_decoder_dtype = argv[++i];
        } else if (std::strcmp(argv[i], "--info") == 0) {
            info = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (vit_path.empty() ||
        mmproj_path.empty() ||
        llm_path.empty() ||
        tokenizer_path.empty() ||
        state_path.empty() ||
        action_decoder_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    vlacpp_model_params model_params = vlacpp_default_model_params();
    model_params.backend = backend;
    std::vector<vlacpp_component_dtype_override> dtype_overrides;
    if (!vit_dtype.empty()) {
        dtype_overrides.push_back({"vit", vit_dtype.c_str()});
    }
    if (!mmproj_dtype.empty()) {
        dtype_overrides.push_back({"mmproj", mmproj_dtype.c_str()});
    }
    if (!llm_dtype.empty()) {
        dtype_overrides.push_back({"llm", llm_dtype.c_str()});
    }
    if (!state_dtype.empty()) {
        dtype_overrides.push_back({"state", state_dtype.c_str()});
    }
    if (!action_decoder_dtype.empty()) {
        dtype_overrides.push_back({"action_decoder", action_decoder_dtype.c_str()});
    }
    model_params.dtype_overrides = dtype_overrides.empty() ? nullptr : dtype_overrides.data();
    model_params.dtype_override_count = dtype_overrides.size();
    vlacpp_model_artifact artifact_items[] = {
        {"vit", vit_path.c_str()},
        {"mmproj", mmproj_path.c_str()},
        {"llm", llm_path.c_str()},
        {"tokenizer", tokenizer_path.c_str()},
        {"state", state_path.c_str()},
        {"action_decoder", action_decoder_path.c_str()},
    };
    vlacpp_model_artifacts artifacts{};
    artifacts.items = artifact_items;
    artifacts.count = sizeof(artifact_items) / sizeof(artifact_items[0]);
    vlacpp_model * model = nullptr;
    vlacpp_status status = vlacpp_load_model(&artifacts, &model_params, &model);
    if (status != VLACPP_STATUS_OK) {
        std::cerr << "load failed: " << vlacpp_last_error() << "\n";
        return 1;
    }

    if (info) {
        vlacpp_model_info model_info{};
        status = vlacpp_get_model_info(model, &model_info);
        if (status != VLACPP_STATUS_OK) {
            std::cerr << "info failed: " << vlacpp_last_error() << "\n";
            vlacpp_free_model(model);
            return 1;
        }
        std::cout << "{\n  \"model_info\": {"
                  << "\n    \"model_type\": \"" << (model_info.model_type ? model_info.model_type : "")
                  << "\",\n    \"image_width\": " << model_info.image_width
                  << ",\n    \"image_height\": " << model_info.image_height
                  << ",\n    \"state_dim\": " << model_info.state_dim
                  << ",\n    \"action_dim\": " << model_info.action_dim
                  << ",\n    \"action_horizon\": " << model_info.action_horizon
                  << ",\n    \"max_token_len\": " << model_info.max_token_len
                  << "\n  }\n}\n";
        vlacpp_free_model(model);
        return 0;
    }
    vlacpp_context_params context_params = vlacpp_default_context_params();
    context_params.flow_steps = steps;
    context_params.seed = seed;
    vlacpp_context * context = nullptr;
    status = vlacpp_create_context(model, &context_params, &context);
    if (status != VLACPP_STATUS_OK) {
        std::cerr << "context failed: " << vlacpp_last_error() << "\n";
        vlacpp_free_model(model);
        return 1;
    }

    std::vector<uint8_t> image(static_cast<size_t>(224) * 224 * 3, 127);
    const char * image_names[] = {
        "base_0_rgb",
        "observation.images.image",
        "observation.images.image2",
        "observation.images.empty_camera_0",
    };
    std::vector<vlacpp_image_view> views(sizeof(image_names) / sizeof(image_names[0]));
    for (size_t i = 0; i < views.size(); ++i) {
        views[i].name = image_names[i];
        views[i].data = image.data();
        views[i].width = 224;
        views[i].height = 224;
        views[i].channels = 3;
        views[i].stride_bytes = 224 * 3;
    }

    std::vector<float> state = parse_state(state_text);
    vlacpp_observation obs{};
    obs.images = views.data();
    obs.image_count = views.size();
    obs.state = state.empty() ? nullptr : state.data();
    obs.state_count = state.size();
    obs.prompt = prompt.c_str();

    vlacpp_action_chunk actions{};
    status = vlacpp_infer_actions(context, &obs, &actions);
    if (status != VLACPP_STATUS_OK) {
        std::cerr << "infer failed: " << vlacpp_last_error() << "\n";
        vlacpp_free_context(context);
        vlacpp_free_model(model);
        return 1;
    }

    std::cout << "{\n  \"horizon\": " << actions.horizon
              << ",\n  \"action_dim\": " << actions.action_dim
              << ",\n  \"actions\": [";
    const int n = actions.horizon * actions.action_dim;
    for (int i = 0; i < n; ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << actions.data[i];
    }
    std::cout << "]\n}\n";

    vlacpp_free_action_chunk(&actions);
    vlacpp_free_context(context);
    vlacpp_free_model(model);
    return 0;
}
