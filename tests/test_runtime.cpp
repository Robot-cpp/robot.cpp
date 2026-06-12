#include "vlacpp.h"

#include <cstdlib>
#include <iostream>

namespace {

void require_not_status(vlacpp_status status, const char * what) {
    if (status == VLACPP_STATUS_OK) {
        std::cerr << what << ": expected failure\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    vlacpp_model_params model_params = vlacpp_default_model_params();
    vlacpp_model * model = nullptr;

    require_not_status(
        vlacpp_load_model(nullptr, &model_params, &model),
        "load with null artifacts");

    vlacpp_model_artifact dtype_artifact_items[] = {
        {"vit", "missing.vit.gguf"},
    };
    vlacpp_model_artifacts dtype_artifacts{};
    dtype_artifacts.items = dtype_artifact_items;
    dtype_artifacts.count = sizeof(dtype_artifact_items) / sizeof(dtype_artifact_items[0]);
    vlacpp_model_params invalid_dtype_params = vlacpp_default_model_params();
    invalid_dtype_params.dtype_override_count = 1;
    require_not_status(
        vlacpp_load_model(&dtype_artifacts, &invalid_dtype_params, &model),
        "load with invalid dtype overrides");

    vlacpp_model_artifact artifact_items[] = {
        {"vit", "missing.vit.gguf"},
        {"mmproj", "missing.mmproj.gguf"},
        {"llm", "missing.llm.gguf"},
        {"tokenizer", "missing.tokenizer.gguf"},
        {"state", "missing.state.gguf"},
    };
    vlacpp_model_artifacts artifacts{};
    artifacts.items = artifact_items;
    artifacts.count = sizeof(artifact_items) / sizeof(artifact_items[0]);
    require_not_status(
        vlacpp_load_model(&artifacts, &model_params, &model),
        "load with missing action decoder path");

    if (model != nullptr) {
        std::cerr << "invalid model should not be returned\n";
        return 1;
    }
    return 0;
}
