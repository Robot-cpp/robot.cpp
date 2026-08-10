#include "models/model.h"
#include "models/starvla/starvla_engine.h"

#include <array>
#include <cstdio>
#include <memory>
#include <string>

namespace {

using robotcpp::starvla::StarVLAVariant;

struct VariantCase {
    const char * framework;
    const char * backbone;
    const char * name;
    StarVLAVariant variant;
};

constexpr std::array<VariantCase, 7> kVariants = {{
    // Qwen3-VL
    {"oft", "qwen3_vl", "qwen3_oft", StarVLAVariant::qwen3_oft},
    {"groot", "qwen3_vl", "qwen3_groot", StarVLAVariant::qwen3_groot},
    {"pi_v3", "qwen3_vl", "qwen3_pi_v3", StarVLAVariant::qwen3_pi_v3},

    // Qwen2.5-VL
    {"oft", "qwen2_5_vl", "qwen25_oft", StarVLAVariant::qwen25_oft},
    {"groot", "qwen2_5_vl", "qwen25_groot", StarVLAVariant::qwen25_groot},
    {"pi", "qwen2_5_vl", "qwen25_pi", StarVLAVariant::qwen25_pi},
    {"fast", "qwen2_5_vl", "qwen25_fast", StarVLAVariant::qwen25_fast},
}};

} // namespace

int main() {
    for (const VariantCase & test : kVariants) {
        StarVLAVariant variant = StarVLAVariant::qwen3_oft;
        if (!robotcpp::starvla::starvla_variant_from_metadata(
                test.framework, test.backbone, variant) ||
            variant != test.variant ||
            std::string(robotcpp::starvla::starvla_variant_name(variant)) != test.name ||
            std::string(robotcpp::starvla::starvla_variant_framework(variant)) !=
                test.framework) {
            std::fprintf(stderr, "variant check failed: %s\n", test.name);
            return 1;
        }
    }

    robotcpp::model_args args;
    args.type = robotcpp::model_type::starvla;
    std::unique_ptr<robotcpp::Model> model;
    std::string error;
    if (robotcpp::make_model(args, model, error) || model ||
        error.find("policy path is required") == std::string::npos) {
        std::fprintf(stderr, "unexpected factory result: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
