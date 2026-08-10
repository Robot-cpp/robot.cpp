#include "models/argument_parse.h"
#include "models/model.h"
#ifdef ROBOT_CPP_BUILD_STARVLA
#include "models/starvla/starvla_engine.h"
#endif

#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace {

bool check(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "model_type_test.cpp:%d: check failed: %s\n", line,
                     expression);
    }
    return condition;
}

} // namespace

#define CHECK(expression) \
    do { \
        if (!check((expression), #expression, __LINE__)) return 1; \
    } while (false)

int main() {
    using robotcpp::model_type;

    int parsed_int = 17;
    CHECK(robotcpp::parse_integer_argument("-42", parsed_int));
    CHECK(parsed_int == -42);
    CHECK(!robotcpp::parse_integer_argument("", parsed_int));
    CHECK(!robotcpp::parse_integer_argument("12x", parsed_int));

    int64_t parsed_seed = 0;
    CHECK(robotcpp::parse_integer_argument(
        std::to_string(std::numeric_limits<int64_t>::max()).c_str(), parsed_seed));

    constexpr std::array<model_type, 3> types = {
        model_type::smolvla, model_type::pi0, model_type::starvla,
    };
    for (model_type type : types) {
        model_type parsed = model_type::smolvla;
        CHECK(robotcpp::parse_model_type(robotcpp::model_type_name(type), parsed));
        CHECK(parsed == type);
    }
    model_type parsed = model_type::smolvla;
    CHECK(!robotcpp::parse_model_type("starvla_qwen_oft", parsed));
    CHECK(!robotcpp::parse_model_type("unknown", parsed));
    CHECK(robotcpp::is_starvla_model_type(model_type::starvla));
    CHECK(!robotcpp::is_starvla_model_type(model_type::pi0));

#ifdef ROBOT_CPP_BUILD_STARVLA
    using robotcpp::starvla::StarVLAVariant;
    constexpr std::array<StarVLAVariant, 7> variants = {
        StarVLAVariant::qwen3_oft,   StarVLAVariant::qwen3_groot,
        StarVLAVariant::qwen3_pi_v3, StarVLAVariant::qwen25_oft,
        StarVLAVariant::qwen25_groot, StarVLAVariant::qwen25_pi,
        StarVLAVariant::qwen25_fast,
    };
    for (StarVLAVariant variant : variants) {
        CHECK(std::string(robotcpp::starvla::starvla_variant_name(variant)) !=
              "unknown");
        CHECK(std::string(robotcpp::starvla::starvla_variant_framework(variant)) !=
              "unknown");
    }
    const std::array<std::pair<const char *, const char *>, 7> metadata = {{
        {"oft", "qwen3_vl"}, {"groot", "qwen3_vl"},
        {"pi_v3", "qwen3_vl"}, {"oft", "qwen2_5_vl"},
        {"groot", "qwen2_5_vl"}, {"pi", "qwen2_5_vl"},
        {"fast", "qwen2_5_vl"},
    }};
    for (size_t index = 0; index < variants.size(); ++index) {
        StarVLAVariant actual = StarVLAVariant::qwen3_oft;
        CHECK(robotcpp::starvla::starvla_variant_from_metadata(
            metadata[index].first, metadata[index].second, actual));
        CHECK(actual == variants[index]);
    }
    StarVLAVariant unsupported = StarVLAVariant::qwen3_oft;
    CHECK(!robotcpp::starvla::starvla_variant_from_metadata(
        "fast", "qwen3_vl", unsupported));
#endif

    robotcpp::model_args args;
    args.type = model_type::starvla;
    std::unique_ptr<robotcpp::Model> model;
    std::string error;
    CHECK(!robotcpp::make_model(args, model, error));
    CHECK(model == nullptr);
#ifdef ROBOT_CPP_BUILD_STARVLA
    CHECK(error.find("policy path") != std::string::npos);

    args.noise_mode = 1;
    error.clear();
    CHECK(!robotcpp::make_model(args, model, error));
    CHECK(error.find("noise-mode debug-sin") != std::string::npos);
#else
    CHECK(error.find("ROBOT_CPP_BUILD_STARVLA=ON") != std::string::npos);
#endif
    return 0;
}

#undef CHECK
