#include "models/starvla/qwen3vl_bridge.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void require_source(robotcpp::starvla::QwenVLArchitecture architecture,
                    int layer_count, int deepstack_count, int tuple_index,
                    robotcpp::starvla::QwenVLHiddenStateSourceKind expected_kind,
                    int expected_layer, const char * message) {
    robotcpp::starvla::QwenVLHiddenStateSource source;
    std::string error;
    require(robotcpp::starvla::qwen_vl_hidden_state_source(
                architecture, layer_count, deepstack_count, tuple_index,
                source, error),
            message);
    require(error.empty(), "successful hidden-state mapping must clear the error");
    require(source.kind == expected_kind, "hidden-state source kind must match");
    require(source.layer == expected_layer, "hidden-state source layer must match");
}

} // namespace

int main() {
    using namespace robotcpp::starvla;

    QwenVLArchitecture architecture = QwenVLArchitecture::unknown;
    std::string error;
    require(qwen_vl_resolve_architecture(
                "qwen2vl", "qwen2.5vl_merger", architecture, error),
            "Qwen2.5-VL text and projector metadata must resolve");
    require(architecture == QwenVLArchitecture::qwen2_5_vl,
            "Qwen2.5-VL metadata must select the Qwen2.5 profile");
    require(error.empty(), "successful architecture resolution must clear the error");
    require(std::string(qwen_vl_architecture_name(architecture)) == "qwen2.5-vl",
            "Qwen2.5-VL architecture name must be stable");

    require(qwen_vl_resolve_architecture(
                "qwen3vl", "qwen3vl_merger", architecture, error),
            "Qwen3-VL text and projector metadata must resolve");
    require(architecture == QwenVLArchitecture::qwen3_vl,
            "Qwen3-VL metadata must select the Qwen3 profile");
    require(std::string(qwen_vl_architecture_name(architecture)) == "qwen3-vl",
            "Qwen3-VL architecture name must be stable");

    require(qwen_vl_is_final_norm_tensor_name("result_norm"),
            "the native llama.cpp final norm name must be recognized");
    require(qwen_vl_is_final_norm_tensor_name("result_embd_pooled"),
            "the pooling-none alias of the final norm must be recognized");
    require(!qwen_vl_is_final_norm_tensor_name("result_output"),
            "the language-model output must not be treated as final norm");
    require(!qwen_vl_is_final_norm_tensor_name("result_embd_pooled-0"),
            "indexed lookalikes must not be treated as final norm");
    require(!qwen_vl_is_final_norm_tensor_name(nullptr),
            "a missing graph tensor name must not be treated as final norm");

    require(!qwen_vl_resolve_architecture(
                "qwen2vl", "qwen3vl_merger", architecture, error),
            "mismatched Qwen-VL text and projector metadata must fail");
    require(architecture == QwenVLArchitecture::unknown,
            "failed architecture resolution must leave an unknown profile");
    require(!error.empty(), "mismatched metadata must report an error");
    require(!qwen_vl_resolve_architecture(
                "qwen2", "qwen2.5vl_merger", architecture, error),
            "unsupported Qwen text architectures must fail");

    require_source(QwenVLArchitecture::qwen2_5_vl, 36, 0, 1,
                   QwenVLHiddenStateSourceKind::decoder_output, 0,
                   "Qwen2.5 tuple index one must map to decoder layer zero");
    require_source(QwenVLArchitecture::qwen2_5_vl, 36, 0, 21,
                   QwenVLHiddenStateSourceKind::decoder_output, 20,
                   "Qwen2.5 tuple index 21 must map to decoder layer 20");
    require_source(QwenVLArchitecture::qwen2_5_vl, 36, 0, 35,
                   QwenVLHiddenStateSourceKind::decoder_output, 34,
                   "Qwen2.5 tuple index 35 must map to decoder layer 34");
    require_source(QwenVLArchitecture::qwen2_5_vl, 36, 0, 36,
                   QwenVLHiddenStateSourceKind::final_norm, -1,
                   "Qwen2.5 final tuple item must map to result_norm");

    QwenVLHiddenStateSource source;
    require(!qwen_vl_hidden_state_source(
                QwenVLArchitecture::qwen2_5_vl, 36, 0, 0, source, error),
            "embedding tuple index zero must not be exposed");
    require(!qwen_vl_hidden_state_source(
                QwenVLArchitecture::qwen2_5_vl, 36, 0, 37, source, error),
            "tuple indices past the final decoder layer must fail");
    require(!qwen_vl_hidden_state_source(
                QwenVLArchitecture::qwen2_5_vl, 36, 1, 21, source, error),
            "Qwen2.5 profiles with DeepStack must fail");

    require_source(QwenVLArchitecture::qwen3_vl, 36, 3, 1,
                   QwenVLHiddenStateSourceKind::deepstack_output, 0,
                   "Qwen3 tuple index one must map to DeepStack layer zero");
    require_source(QwenVLArchitecture::qwen3_vl, 36, 3, 3,
                   QwenVLHiddenStateSourceKind::deepstack_output, 2,
                   "Qwen3 tuple index three must map to DeepStack layer two");
    require_source(QwenVLArchitecture::qwen3_vl, 36, 3, 4,
                   QwenVLHiddenStateSourceKind::decoder_output, 3,
                   "Qwen3 tuple index four must map to decoder layer three");
    require_source(QwenVLArchitecture::qwen3_vl, 36, 3, 36,
                   QwenVLHiddenStateSourceKind::decoder_output, 35,
                   "Qwen3 final tuple item must retain the raw decoder output");
    require(!qwen_vl_hidden_state_source(
                QwenVLArchitecture::qwen3_vl, 36, 0, 1, source, error),
            "Qwen3 profiles without DeepStack must fail");
    require(!qwen_vl_hidden_state_source(
                QwenVLArchitecture::unknown, 36, 0, 1, source, error),
            "unknown Qwen-VL profiles must fail");

    std::cout << "starvla Qwen-VL profile tests passed\n";
    return 0;
}
