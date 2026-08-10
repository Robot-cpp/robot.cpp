#include "models/model.h"

#include <cstdio>
#include <memory>
#include <string>

int main() {
    robotcpp::model_args args;
    args.type = robotcpp::model_type::smolvla;
    std::unique_ptr<robotcpp::Model> model;
    std::string error;

    const auto rejects = [&](const char * expected) {
        error.clear();
        if (robotcpp::make_model(args, model, error) || model || error != expected) {
            std::fprintf(stderr, "expected %s, got %s\n", expected, error.c_str());
            return false;
        }
        return true;
    };

    if (!rejects("SmolVLA llm_path is required")) return 1;
    args.llm_path = "llm.gguf";
    if (!rejects("SmolVLA mmproj_path is required")) return 1;
    args.mmproj_path = "mmproj.gguf";
    if (!rejects("SmolVLA state_proj_path is required")) return 1;
    args.state_proj_path = "state.gguf";
    if (!rejects("SmolVLA action_expert_path is required")) return 1;
    return 0;
}
