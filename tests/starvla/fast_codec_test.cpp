#include "models/starvla/fast_codec.h"

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

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

uint32_t byte_level_codepoint(uint8_t target) {
    auto is_direct = [](int value) {
        return (value >= 0x21 && value <= 0x7e) || (value >= 0xa1 && value <= 0xac) ||
               (value >= 0xae && value <= 0xff);
    };
    if (is_direct(target)) {
        return target;
    }
    uint32_t extra = 0;
    for (int value = 0; value < target; ++value) {
        if (!is_direct(value)) {
            ++extra;
        }
    }
    return 256U + extra;
}

std::string utf8(uint32_t codepoint) {
    std::string output;
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
    return output;
}

std::unique_ptr<FastCodec> make_synthetic_codec(size_t time_horizon = 2,
                                                size_t action_dim = 2) {
    const std::vector<uint8_t> raw_bytes = {10, 20, 30, 40, 0xe2, 0x82, 0x28};
    std::vector<std::string> vocab;
    for (uint8_t byte : raw_bytes) {
        vocab.push_back(utf8(byte_level_codepoint(byte)));
    }
    FastCodecConfig config;
    config.scale = 1.0;
    config.min_token = 0;
    config.vocab_size = vocab.size();
    config.time_horizon = time_horizon;
    config.action_dim = action_dim;
    std::string error;
    auto codec = FastCodec::create(config, vocab, {100, 42, 999, 7, 501, 502, 503}, error);
    require(codec != nullptr, "synthetic FAST codec must construct: " + error);
    return codec;
}

void run_unit_tests() {
    auto codec = make_synthetic_codec();
    std::string error;

    std::vector<int32_t> vlm_ids;
    require(codec->map_fast_to_vlm({3, 0, 2, 1}, vlm_ids, error),
            "FAST-to-VLM mapping must succeed");
    require(vlm_ids == std::vector<int32_t>({7, 100, 999, 42}),
            "FAST-to-VLM mapping must use the explicit non-contiguous table");

    std::vector<int32_t> fast_ids;
    require(codec->map_vlm_to_fast(vlm_ids, fast_ids, error),
            "VLM-to-FAST mapping must succeed");
    require(fast_ids == std::vector<int32_t>({3, 0, 2, 1}),
            "VLM-to-FAST mapping must invert the explicit table");
    require(!codec->map_vlm_to_fast({101}, fast_ids, error),
            "an unmapped Qwen token must be rejected");

    require(codec->extract_fast_tokens({-1, 42, 1234, 100, 7, 42}, fast_ids, error),
            "full Qwen sequence action extraction must succeed");
    require(fast_ids == std::vector<int32_t>({1, 0, 3, 1}),
            "action extraction must filter with the inverse map and preserve order");

    std::vector<uint32_t> codepoints;
    require(codec->byte_level_decode({0, 1, 2, 3}, codepoints, error),
            "ByteLevel decode must succeed");
    require(codepoints == std::vector<uint32_t>({10, 20, 30, 40}),
            "ByteLevel decode must invert the GPT-2 byte alphabet");
    require(codec->byte_level_decode({4, 5, 6}, codepoints, error),
            "lossy UTF-8 ByteLevel decode must succeed");
    require(codepoints == std::vector<uint32_t>({0xfffdU, 0x28U}),
            "ByteLevel decode must match Rust UTF-8 replacement semantics");

    FastDecodeResult decoded;
    require(codec->decode_fast_tokens({{0, 1, 2, 3}}, decoded, error),
            "synthetic inverse DCT must succeed");
    require(decoded.actions.size() == 4,
            "valid synthetic tokens must produce one 2x2 action chunk");
    const double root_half = std::sqrt(0.5);
    const std::vector<double> expected = {
        root_half * (10.0 + 30.0), root_half * (20.0 + 40.0),
        root_half * (10.0 - 30.0), root_half * (20.0 - 40.0),
    };
    for (size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(decoded.actions[index] - expected[index]) < 1e-12,
                "orthonormal inverse DCT must match the analytical result");
    }

    require(!codec->decode_fast_tokens({{}, {0, 1, 2}, {9999}}, decoded, error),
            "malformed FAST coefficients must fail");
    require(decoded.actions.empty(), "failed FAST decode must not return zero actions");
    require(!codec->decode_fast_tokens({}, decoded, error), "an empty batch must fail explicitly");

    require(!codec->decode_fast_tokens(
                {{0, 1, 2, 3}, {9999}, {0, 1, 2, 3}}, decoded, error),
            "a malformed FAST batch member must fail the batch");

    FastDecodeResult decoded_generated;
    require(codec->decode_generated_tokens(
                {{-1, 100, 123456, 42, 999, 555555, 7}}, decoded_generated, error),
            "complete generated_ids must filter then decode");
    bool generated_matches_expected = decoded_generated.actions.size() == expected.size();
    for (size_t index = 0; generated_matches_expected && index < expected.size(); ++index) {
        generated_matches_expected =
            std::abs(decoded_generated.actions[index] - expected[index]) < 1e-12;
    }
    require(generated_matches_expected,
            "complete generated_ids must match the pure FAST-token action decode");
    require(!codec->decode_generated_tokens({{1, 2, 3}}, decoded_generated, error),
            "a generated sequence without action tokens must fail");
    require(!codec->decode_vlm_action_tokens({{123456}}, decoded_generated, error),
            "strict low-level VLM action-token decode must reject ordinary Qwen tokens");

    std::vector<int32_t> maximum_generated_sequence(2048, 123456);
    require(!codec->decode_generated_tokens(
                {maximum_generated_sequence}, decoded_generated, error),
            "max_length text without action tokens must fail decode");
    std::vector<int32_t> oversized_sequence(2049, 100);
    require(!codec->extract_fast_tokens(oversized_sequence, fast_ids, error),
            "generated token sequences beyond official max_length must fail before allocation");

    FastCodecConfig invalid_config = codec->config();
    invalid_config.scale = 0.0;
    auto invalid = FastCodec::create(invalid_config,
                                     {"a", "b", "c", "d", "e", "f", "g"},
                                     {0, 1, 2, 3, 4, 5, 6}, error);
    require(invalid == nullptr, "zero FAST scale must be rejected");

    invalid_config = codec->config();
    invalid = FastCodec::create(invalid_config,
                                {"a", "b", "c", "d", "e", "f", "g"},
                                {0, 1, 2, 3, 4, 5, 5}, error);
    require(invalid == nullptr, "duplicate VLM action-token IDs must be rejected");

    invalid_config = codec->config();
    invalid_config.time_horizon = 1025;
    invalid = FastCodec::create(invalid_config,
                                {"a", "b", "c", "d", "e", "f", "g"},
                                {0, 1, 2, 3, 4, 5, 6}, error);
    require(invalid == nullptr, "oversized FAST horizons must fail before graph work");

    std::vector<std::vector<int32_t>> oversized_batch(1025);
    require(!codec->decode_fast_tokens(oversized_batch, decoded, error),
            "oversized FAST batches must fail before output allocation");

    auto work_limited_codec = make_synthetic_codec(257, 1);
    std::vector<std::vector<int32_t>> excessive_idct_batch(1024);
    require(!work_limited_codec->decode_fast_tokens(excessive_idct_batch, decoded, error) &&
                error.find("work limit") != std::string::npos,
            "inverse-DCT work accounting must include the full batch dimension");
}

} // namespace

int main() {
    run_unit_tests();
    std::cout << "starvla FAST codec unit tests passed\n";
    return 0;
}
