#include "models/model.h"

#include <cstdio>
#include <memory>
#include <string>

int main() {
    robotcpp::model_args args;
    args.type = robotcpp::model_type::pi0;
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

    if (!rejects("Pi0 vit_path is required")) return 1;
    args.vit_path = "vit.gguf";
    if (!rejects("Pi0 mmproj_path is required")) return 1;
    args.mmproj_path = "mmproj.gguf";
    if (!rejects("Pi0 llm_path is required")) return 1;
    args.llm_path = "llm.gguf";
    if (!rejects("Pi0 tokenizer_path is required")) return 1;
    args.tokenizer_path = "tokenizer.model";
    if (!rejects("Pi0 state_path is required")) return 1;
    args.state_path = "state.gguf";
    if (!rejects("Pi0 action_decoder_path is required")) return 1;
    return 0;
}
