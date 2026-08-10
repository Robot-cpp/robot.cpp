#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace robotcpp::starvla {

struct FastCodecConfig {
    double scale = 0.0;
    int32_t min_token = 0;
    size_t vocab_size = 0;
    size_t time_horizon = 0;
    size_t action_dim = 0;
};

struct FastDecodeResult {
    size_t batch_size = 0;
    size_t time_horizon = 0;
    size_t action_dim = 0;
    std::vector<double> actions;
};

class FastCodec {
public:
    static std::unique_ptr<FastCodec> create(
        FastCodecConfig config, std::vector<std::string> vocab_by_id,
        std::vector<int32_t> fast_to_vlm_id, std::string & error);

    // Constructs directly from the converter-compiled ByteLevel pieces stored
    // in policy GGUF. offsets has vocab_size + 1 entries and indexes the flat
    // byte buffer; no external tokenizer JSON is consulted.
    static std::unique_ptr<FastCodec> create_compiled(
        FastCodecConfig config, std::vector<int32_t> token_offsets,
        std::vector<uint8_t> token_bytes,
        std::vector<int32_t> fast_to_vlm_id, std::string & error);

    const FastCodecConfig & config() const;
    const std::vector<int32_t> & fast_to_vlm_ids() const;

    bool map_fast_to_vlm(const std::vector<int32_t> & fast_ids,
                         std::vector<int32_t> & vlm_ids, std::string & error) const;
    bool map_vlm_to_fast(const std::vector<int32_t> & vlm_ids,
                         std::vector<int32_t> & fast_ids, std::string & error) const;

    // Extracts every mapped action token from a generated Qwen sequence in order.
    // EOS stopping remains the generator's responsibility; ordinary EOS/pad/text
    // IDs in the returned sequence are ignored and do not terminate this scan.
    bool extract_fast_tokens(const std::vector<int32_t> & generated_ids,
                             std::vector<int32_t> & fast_ids, std::string & error) const;

    // Exposed for focused parity diagnostics. This is the Hugging Face ByteLevel
    // decoder output before min_token adjustment and inverse DCT.
    bool byte_level_decode(const std::vector<int32_t> & fast_ids,
                           std::vector<uint32_t> & codepoints, std::string & error) const;

    bool decode_fast_tokens(const std::vector<std::vector<int32_t>> & batch_fast_ids,
                            FastDecodeResult & result, std::string & error) const;

    // Strict low-level API: every input ID must be an action token. Use
    // decode_generated_tokens for complete Qwen sequences containing text.
    bool decode_vlm_action_tokens(const std::vector<std::vector<int32_t>> & batch_vlm_ids,
                                  FastDecodeResult & result, std::string & error) const;

    // Production entry point for complete Qwen generated_ids. Ordinary text and
    // control tokens are filtered through the explicit inverse action-token map.
    bool decode_generated_tokens(const std::vector<std::vector<int32_t>> & batch_generated_ids,
                                 FastDecodeResult & result, std::string & error) const;

private:
    FastCodec(FastCodecConfig config, std::vector<std::vector<uint8_t>> token_bytes,
              std::vector<int32_t> fast_to_vlm_id);

    FastCodecConfig config_;
    std::vector<std::vector<uint8_t>> token_bytes_;
    std::vector<int32_t> fast_to_vlm_id_;
    std::vector<std::pair<int32_t, int32_t>> vlm_to_fast_id_;
};

} // namespace robotcpp::starvla
