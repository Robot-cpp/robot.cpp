#include "models/starvla/fast_codec.h"
#include "models/starvla/fast_policy.h"
#include "models/starvla/qwen3vl_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using robotcpp::starvla::FastCodec;
using robotcpp::starvla::FastCodecConfig;
using robotcpp::starvla::FastDecodeResult;
using robotcpp::starvla::FastPolicy;

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void test_generation_selector() {
    std::string error;
    int32_t token = -1;
    std::vector<float> logits = {10.0f, 9.0f, 1.0f};
    require(robotcpp::starvla::qwen_vl_select_repetition_penalized_top1(
                logits.data(), logits.size(), {0}, 2.0f, token, error),
            "repetition-penalized selector must succeed: " + error);
    require(token == 1,
            "a repeated positive logit must be divided before top_k=1");

    logits = {-1.0f, -1.5f, -4.0f};
    require(robotcpp::starvla::qwen_vl_select_repetition_penalized_top1(
                logits.data(), logits.size(), {0}, 2.0f, token, error),
            "negative-logit selector must succeed");
    require(token == 1,
            "a repeated negative logit must be multiplied before top_k=1");

    logits = {3.0f, 3.0f};
    require(robotcpp::starvla::qwen_vl_select_repetition_penalized_top1(
                logits.data(), logits.size(), {}, 1.05f, token, error) &&
                token == 0,
            "top_k=1 tie handling must match torch.argmax first-index semantics");
    require(!robotcpp::starvla::qwen_vl_select_repetition_penalized_top1(
                logits.data(), logits.size(), {2}, 1.05f, token, error),
            "out-of-vocabulary history must be rejected");
    logits[0] = std::numeric_limits<float>::quiet_NaN();
    require(!robotcpp::starvla::qwen_vl_select_repetition_penalized_top1(
                logits.data(), logits.size(), {}, 1.05f, token, error),
            "NaN generation logits must fail closed");
}

void test_compiled_codec() {
    FastCodecConfig config;
    config.scale = 1.0;
    config.min_token = 0;
    config.vocab_size = 4;
    config.time_horizon = 2;
    config.action_dim = 2;
    std::string error;
    auto codec = FastCodec::create_compiled(
        config, {0, 1, 2, 3, 4}, {10, 20, 30, 40},
        {100, 101, 102, 103}, error);
    require(codec != nullptr,
            "compiled FAST codec must construct without sidecars: " + error);

    FastDecodeResult decoded;
    require(codec->decode_generated_tokens(
                {{999, 100, 101, 888, 102, 103}}, decoded, error),
            "compiled FAST codec must filter and decode a full Qwen sequence");
    const double root_half = std::sqrt(0.5);
    const std::vector<double> expected = {
        root_half * 40.0, root_half * 60.0,
        root_half * -20.0, root_half * -20.0,
    };
    require(decoded.actions.size() == expected.size(),
            "compiled FAST codec must return the configured 2x2 action shape");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::fabs(decoded.actions[i] - expected[i]) < 1.0e-12,
                "compiled FAST codec IDCT differs from the analytical result");
    }

    require(FastCodec::create_compiled(
                config, {0, 1, 1, 3, 4}, {10, 20, 30, 40},
                {100, 101, 102, 103}, error) == nullptr,
            "compiled FAST codec must reject non-increasing offsets");
}

void test_policy(const std::string & path) {
    std::string error;
    std::unique_ptr<FastPolicy> policy = FastPolicy::load(path, 0, error);
    require(policy != nullptr, "official FAST policy GGUF must load: " + error);
    const auto & config = policy->config();
    require(config.bundle_uuid == "b2651406-918b-524b-9df6-66861d744f29" &&
                config.qwen_hidden_dim == 2048 &&
                config.qwen_vocab_size == 153713 &&
                config.generation_max_length == 2048 &&
                config.generation_eos_token_ids ==
                    std::vector<int32_t>({151645, 151643}),
            "official FAST policy metadata must expose the pinned runtime");

    std::vector<float> normalized;
    require(!policy->decode_generated(
                {100, 151665, 200}, normalized, error),
            "incomplete FAST output must fail");
    require(normalized.empty(), "failed FAST decode must not return actions");
}

} // namespace

int main(int argc, char ** argv) {
    test_generation_selector();
    test_compiled_codec();

    if (argc == 3 && std::string(argv[1]) == "--policy") {
        test_policy(argv[2]);
    } else if (argc == 3 && std::string(argv[1]) == "--expect-reject") {
        std::string error;
        require(FastPolicy::load(argv[2], 0, error) == nullptr && !error.empty(),
                "tampered FAST policy GGUF must fail closed");
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0]
                  << " [--policy|--expect-reject <policy.gguf>]\n";
        return 2;
    }
    return 0;
}
