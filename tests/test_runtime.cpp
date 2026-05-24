#include "vlacpp.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace {

void require_status(vlacpp_status status, const char * what) {
    if (status != VLACPP_STATUS_OK) {
        std::cerr << what << ": " << vlacpp_last_error() << "\n";
        std::exit(1);
    }
}

void require_not_status(vlacpp_status status, const char * what) {
    if (status == VLACPP_STATUS_OK) {
        std::cerr << what << ": expected failure\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    vlacpp_model_params model_params = vlacpp_default_model_params();
    const char * pi0_without_tensors_path = "vlacpp-test-pi0-without-tensors.json";
    {
        std::ofstream file(pi0_without_tensors_path);
        file << "{"
             << "\"model_type\":\"pi0\","
             << "\"image_width\":16,"
             << "\"image_height\":16,"
             << "\"state_dim\":3,"
             << "\"action_dim\":2,"
             << "\"action_horizon\":4,"
             << "\"image_keys\":[\"base_0_rgb\"]"
             << "}";
    }
    vlacpp_model * invalid_model = nullptr;
    require_not_status(
        vlacpp_load_model(pi0_without_tensors_path, &model_params, &invalid_model),
        "load pi0 without tensors");
    if (invalid_model != nullptr) {
        std::cerr << "invalid model should not be returned\n";
        return 1;
    }
    std::remove(pi0_without_tensors_path);
    return 0;
}
