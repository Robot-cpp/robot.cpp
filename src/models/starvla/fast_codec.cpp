#include "models/starvla/fast_codec.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace robotcpp::starvla {
namespace {

using Json = nlohmann::json;

constexpr size_t kMaximumJsonBytes = 16U * 1024U * 1024U;
constexpr size_t kOfficialVocabSize = 2048U;
constexpr size_t kMaximumVocabSize = 65536U;
constexpr size_t kMaximumTimeHorizon = 1024U;
constexpr size_t kMaximumActionDim = 1024U;
constexpr size_t kMaximumBatchSize = 1024U;
constexpr size_t kMaximumTokenSequence = 4096U;
constexpr size_t kMaximumGeneratedSequence = 2048U;
constexpr size_t kMaximumDecodedBytes = 1024U * 1024U;
constexpr size_t kMaximumOutputScalars = 16U * 1024U * 1024U;
constexpr uint64_t kMaximumIdctMultiplyAdds = 64ULL * 1024ULL * 1024ULL;
constexpr const char * kActionTokenPrefix = "<robot_action_";
constexpr double kPi = 3.141592653589793238462643383279502884;

bool read_json(const std::filesystem::path & path, Json & output, std::string & error) {
    std::error_code filesystem_error;
    const uintmax_t size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) {
        error = "cannot stat StarVLA FAST JSON asset '" + path.string() + "': " +
                filesystem_error.message();
        return false;
    }
    if (size > kMaximumJsonBytes) {
        error = "StarVLA FAST JSON asset exceeds the 16 MiB limit: " + path.string();
        return false;
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot open StarVLA FAST JSON asset: " + path.string();
        return false;
    }
    std::string contents;
    contents.reserve(static_cast<size_t>(size));
    std::array<char, 64U * 1024U> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count <= 0) {
            continue;
        }
        const size_t chunk_size = static_cast<size_t>(count);
        if (contents.size() > kMaximumJsonBytes - chunk_size) {
            error = "StarVLA FAST JSON asset exceeds the 16 MiB limit while reading: " +
                    path.string();
            return false;
        }
        contents.append(buffer.data(), chunk_size);
    }
    if (!stream.eof() || stream.bad()) {
        error = "cannot read StarVLA FAST JSON asset: " + path.string();
        return false;
    }
    output = Json::parse(contents, nullptr, false);
    if (output.is_discarded()) {
        error = "cannot parse StarVLA FAST JSON asset: " + path.string();
        return false;
    }
    return true;
}

bool decode_utf8_strict(const std::string & input, std::vector<uint32_t> & output,
                        std::string & error) {
    output.clear();
    for (size_t i = 0; i < input.size();) {
        const uint8_t first = static_cast<uint8_t>(input[i]);
        uint32_t value = 0;
        size_t length = 0;
        if (first <= 0x7fU) {
            value = first;
            length = 1;
        } else if (first >= 0xc2U && first <= 0xdfU) {
            value = first & 0x1fU;
            length = 2;
        } else if (first >= 0xe0U && first <= 0xefU) {
            value = first & 0x0fU;
            length = 3;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            value = first & 0x07U;
            length = 4;
        } else {
            error = "StarVLA FAST tokenizer vocabulary contains invalid UTF-8";
            return false;
        }
        if (i + length > input.size()) {
            error = "StarVLA FAST tokenizer vocabulary contains truncated UTF-8";
            return false;
        }
        for (size_t j = 1; j < length; ++j) {
            const uint8_t continuation = static_cast<uint8_t>(input[i + j]);
            if ((continuation & 0xc0U) != 0x80U) {
                error = "StarVLA FAST tokenizer vocabulary contains invalid UTF-8 continuation";
                return false;
            }
            value = (value << 6U) | (continuation & 0x3fU);
        }
        const bool overlong = (length == 2 && value < 0x80U) ||
                              (length == 3 && value < 0x800U) ||
                              (length == 4 && value < 0x10000U);
        if (overlong || value > 0x10ffffU || (value >= 0xd800U && value <= 0xdfffU)) {
            error = "StarVLA FAST tokenizer vocabulary contains a non-scalar UTF-8 value";
            return false;
        }
        output.push_back(value);
        i += length;
    }
    return true;
}

std::unordered_map<uint32_t, uint8_t> byte_level_inverse_alphabet() {
    std::unordered_map<uint32_t, uint8_t> result;
    std::unordered_set<int> direct;
    for (int value = 0x21; value <= 0x7e; ++value) {
        direct.insert(value);
        result.emplace(static_cast<uint32_t>(value), static_cast<uint8_t>(value));
    }
    for (int value = 0xa1; value <= 0xac; ++value) {
        direct.insert(value);
        result.emplace(static_cast<uint32_t>(value), static_cast<uint8_t>(value));
    }
    for (int value = 0xae; value <= 0xff; ++value) {
        direct.insert(value);
        result.emplace(static_cast<uint32_t>(value), static_cast<uint8_t>(value));
    }
    uint32_t extra = 0;
    for (int value = 0; value <= 0xff; ++value) {
        if (direct.count(value) == 0) {
            result.emplace(256U + extra, static_cast<uint8_t>(value));
            ++extra;
        }
    }
    return result;
}

bool compile_token_bytes(const std::vector<std::string> & vocab_by_id,
                         std::vector<std::vector<uint8_t>> & token_bytes,
                         std::string & error) {
    const auto inverse_alphabet = byte_level_inverse_alphabet();
    token_bytes.clear();
    token_bytes.reserve(vocab_by_id.size());
    for (size_t token_id = 0; token_id < vocab_by_id.size(); ++token_id) {
        if (vocab_by_id[token_id].empty()) {
            error = "StarVLA FAST tokenizer has an empty vocabulary piece at ID " +
                    std::to_string(token_id);
            return false;
        }
        std::vector<uint32_t> piece_codepoints;
        if (!decode_utf8_strict(vocab_by_id[token_id], piece_codepoints, error)) {
            error += " at token ID " + std::to_string(token_id);
            return false;
        }
        std::vector<uint8_t> bytes;
        bytes.reserve(piece_codepoints.size());
        for (uint32_t codepoint : piece_codepoints) {
            const auto found = inverse_alphabet.find(codepoint);
            if (found == inverse_alphabet.end()) {
                error = "StarVLA FAST tokenizer piece contains a code point outside the ByteLevel "
                        "alphabet at ID " +
                        std::to_string(token_id);
                return false;
            }
            bytes.push_back(found->second);
        }
        token_bytes.push_back(std::move(bytes));
    }
    return true;
}

void decode_utf8_lossy(const std::vector<uint8_t> & input, std::vector<uint32_t> & output) {
    output.clear();
    for (size_t i = 0; i < input.size();) {
        const uint8_t first = input[i];
        if (first <= 0x7fU) {
            output.push_back(first);
            ++i;
            continue;
        }

        size_t length = 0;
        uint32_t value = 0;
        if (first >= 0xc2U && first <= 0xdfU) {
            length = 2;
            value = first & 0x1fU;
        } else if (first >= 0xe0U && first <= 0xefU) {
            length = 3;
            value = first & 0x0fU;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            length = 4;
            value = first & 0x07U;
        } else {
            output.push_back(0xfffdU);
            ++i;
            continue;
        }

        if (i + 1 >= input.size()) {
            output.push_back(0xfffdU);
            break;
        }
        const uint8_t second = input[i + 1];
        const bool second_is_continuation = (second & 0xc0U) == 0x80U;
        const bool second_in_scalar_range =
            !(first == 0xe0U && second < 0xa0U) &&
            !(first == 0xedU && second > 0x9fU) &&
            !(first == 0xf0U && second < 0x90U) &&
            !(first == 0xf4U && second > 0x8fU);
        if (!second_is_continuation || !second_in_scalar_range) {
            output.push_back(0xfffdU);
            ++i;
            continue;
        }
        value = (value << 6U) | (second & 0x3fU);

        bool invalid = false;
        size_t consumed_prefix = 2;
        for (size_t j = 2; j < length; ++j) {
            if (i + j >= input.size()) {
                output.push_back(0xfffdU);
                i = input.size();
                invalid = true;
                break;
            }
            const uint8_t continuation = input[i + j];
            if ((continuation & 0xc0U) != 0x80U) {
                output.push_back(0xfffdU);
                i += consumed_prefix;
                invalid = true;
                break;
            }
            value = (value << 6U) | (continuation & 0x3fU);
            ++consumed_prefix;
        }
        if (invalid) {
            continue;
        }
        output.push_back(value);
        i += length;
    }
}

bool parse_positive_size(const Json & value, const char * name, size_t & output,
                         std::string & error) {
    if (!value.is_number_integer()) {
        error = std::string("StarVLA FAST ") + name + " must be an integer";
        return false;
    }
    try {
        const int64_t parsed = value.get<int64_t>();
        if (parsed <= 0 || static_cast<uint64_t>(parsed) >
                               static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            error = std::string("StarVLA FAST ") + name + " is out of range";
            return false;
        }
        output = static_cast<size_t>(parsed);
        return true;
    } catch (const std::exception &) {
        error = std::string("StarVLA FAST ") + name + " is out of range";
        return false;
    }
}

bool parse_processor_config(const Json & json, size_t time_horizon_override,
                            size_t action_dim_override, FastCodecConfig & config,
                            std::string & error) {
    if (!json.is_object() || !json.contains("processor_class") ||
        json["processor_class"] != "UniversalActionProcessor" || !json.contains("scale") ||
        !json["scale"].is_number() || !json.contains("vocab_size") ||
        !json.contains("min_token") || !json["min_token"].is_number_integer()) {
        error = "StarVLA FAST processor_config.json has an incompatible schema";
        return false;
    }
    config.scale = json["scale"].get<double>();
    if (!std::isfinite(config.scale) || config.scale == 0.0) {
        error = "StarVLA FAST processor scale must be finite and non-zero";
        return false;
    }
    if (!parse_positive_size(json["vocab_size"], "vocab_size", config.vocab_size, error)) {
        return false;
    }
    if (config.vocab_size != kOfficialVocabSize) {
        error = "StarVLA FAST pinned vocabulary must contain exactly 2048 tokens";
        return false;
    }
    try {
        if (json["min_token"].is_number_unsigned()) {
            const uint64_t min_token = json["min_token"].get<uint64_t>();
            if (min_token > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                error = "StarVLA FAST min_token is out of int32 range";
                return false;
            }
            config.min_token = static_cast<int32_t>(min_token);
        } else {
            const int64_t min_token = json["min_token"].get<int64_t>();
            if (min_token < std::numeric_limits<int32_t>::min() ||
                min_token > std::numeric_limits<int32_t>::max()) {
                error = "StarVLA FAST min_token is out of int32 range";
                return false;
            }
            config.min_token = static_cast<int32_t>(min_token);
        }
    } catch (const std::exception &) {
        error = "StarVLA FAST min_token is out of int32 range";
        return false;
    }

    auto choose_dimension = [&](const char * name, size_t override_value, size_t & target) {
        if (override_value != 0) {
            target = override_value;
            return true;
        }
        if (!json.contains(name) || json[name].is_null()) {
            error = std::string("StarVLA FAST ") + name +
                    " is absent; pass the policy dimension explicitly";
            return false;
        }
        return parse_positive_size(json[name], name, target, error);
    };
    return choose_dimension("time_horizon", time_horizon_override, config.time_horizon) &&
           choose_dimension("action_dim", action_dim_override, config.action_dim);
}

bool parse_tokenizer_vocab(const Json & json, size_t expected_vocab_size,
                           std::vector<std::string> & vocab_by_id, std::string & error) {
    if (!json.is_object() || !json.contains("version") || json["version"] != "1.0" ||
        !json.contains("added_tokens") || !json["added_tokens"].is_array() ||
        !json["added_tokens"].empty() || !json.contains("decoder") ||
        !json["decoder"].is_object() || !json["decoder"].contains("type") ||
        json["decoder"]["type"] != "ByteLevel" || !json.contains("model") ||
        !json["model"].is_object() || !json["model"].contains("type") ||
        json["model"]["type"] != "BPE" || !json["model"].contains("vocab") ||
        !json["model"]["vocab"].is_object()) {
        error = "StarVLA FAST tokenizer.json is not the required ByteLevel BPE schema";
        return false;
    }
    const Json & decoder = json["decoder"];
    if (decoder.size() != 4 || !decoder.contains("add_prefix_space") ||
        decoder["add_prefix_space"] != true || !decoder.contains("trim_offsets") ||
        decoder["trim_offsets"] != true || !decoder.contains("use_regex") ||
        decoder["use_regex"] != true) {
        error = "StarVLA FAST tokenizer.json has an incompatible ByteLevel decoder contract";
        return false;
    }
    const Json & vocab = json["model"]["vocab"];
    if (vocab.size() != expected_vocab_size) {
        error = "StarVLA FAST tokenizer vocabulary size does not match processor_config.json";
        return false;
    }
    vocab_by_id.assign(expected_vocab_size, std::string());
    std::vector<bool> seen(expected_vocab_size, false);
    for (auto iterator = vocab.begin(); iterator != vocab.end(); ++iterator) {
        if (!iterator.value().is_number_integer()) {
            error = "StarVLA FAST tokenizer vocabulary ID is not an integer";
            return false;
        }
        int64_t token_id = -1;
        try {
            token_id = iterator.value().get<int64_t>();
        } catch (const std::exception &) {
            error = "StarVLA FAST tokenizer vocabulary ID is out of range";
            return false;
        }
        if (token_id < 0 || static_cast<uint64_t>(token_id) >= expected_vocab_size ||
            seen[static_cast<size_t>(token_id)]) {
            error = "StarVLA FAST tokenizer vocabulary IDs are not a bijection";
            return false;
        }
        seen[static_cast<size_t>(token_id)] = true;
        vocab_by_id[static_cast<size_t>(token_id)] = iterator.key();
    }
    return true;
}

bool parse_action_index(const std::string & value, size_t & index) {
    const std::string prefix(kActionTokenPrefix);
    if (value.size() <= prefix.size() + 1 || value.compare(0, prefix.size(), prefix) != 0 ||
        value.back() != '>') {
        return false;
    }
    const std::string digits = value.substr(prefix.size(), value.size() - prefix.size() - 1);
    if (digits.empty() || (digits.size() > 1 && digits.front() == '0')) {
        return false;
    }
    size_t parsed = 0;
    for (char character : digits) {
        if (character < '0' || character > '9') {
            return false;
        }
        const size_t digit = static_cast<size_t>(character - '0');
        if (parsed > (std::numeric_limits<size_t>::max() - digit) / 10U) {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    index = parsed;
    return true;
}

bool parse_action_map(const Json & json, size_t vocab_size,
                      std::vector<int32_t> & fast_to_vlm, std::string & error) {
    if (!json.is_object() || json.size() != vocab_size) {
        error = "StarVLA FAST action-token map must contain exactly one entry per FAST token";
        return false;
    }
    fast_to_vlm.assign(vocab_size, -1);
    std::vector<bool> seen(vocab_size, false);
    std::unordered_set<int32_t> vlm_ids;
    for (auto iterator = json.begin(); iterator != json.end(); ++iterator) {
        size_t fast_id = 0;
        if (!parse_action_index(iterator.key(), fast_id) || fast_id >= vocab_size || seen[fast_id]) {
            error = "StarVLA FAST action-token map has a malformed or duplicate token name";
            return false;
        }
        if (!iterator.value().is_number_integer()) {
            error = "StarVLA FAST action-token map contains a non-integer VLM ID";
            return false;
        }
        int64_t vlm_id = -1;
        try {
            vlm_id = iterator.value().get<int64_t>();
        } catch (const std::exception &) {
            error = "StarVLA FAST action-token VLM ID is out of range";
            return false;
        }
        if (vlm_id < 0 || vlm_id > std::numeric_limits<int32_t>::max() ||
            !vlm_ids.insert(static_cast<int32_t>(vlm_id)).second) {
            error = "StarVLA FAST action-token VLM IDs must be unique non-negative int32 values";
            return false;
        }
        seen[fast_id] = true;
        fast_to_vlm[fast_id] = static_cast<int32_t>(vlm_id);
    }
    return true;
}

bool checked_action_count(const FastCodecConfig & config, size_t batch_size,
                          size_t & per_sample, size_t & total, std::string & error) {
    if (config.vocab_size > kMaximumVocabSize || config.time_horizon > kMaximumTimeHorizon ||
        config.action_dim > kMaximumActionDim) {
        error = "StarVLA FAST codec dimensions exceed the runtime safety limits";
        return false;
    }
    if (batch_size == 0 || batch_size > kMaximumBatchSize) {
        error = "StarVLA FAST batch size exceeds the runtime safety limit";
        return false;
    }
    if (config.time_horizon > std::numeric_limits<size_t>::max() / config.action_dim) {
        error = "StarVLA FAST action shape overflows size_t";
        return false;
    }
    per_sample = config.time_horizon * config.action_dim;
    if (batch_size > std::numeric_limits<size_t>::max() / per_sample) {
        error = "StarVLA FAST batch shape overflows size_t";
        return false;
    }
    total = batch_size * per_sample;
    if (total > kMaximumOutputScalars) {
        error = "StarVLA FAST output tensor exceeds the runtime scalar limit";
        return false;
    }
    const uint64_t horizon = static_cast<uint64_t>(config.time_horizon);
    const uint64_t action_dim = static_cast<uint64_t>(config.action_dim);
    const uint64_t batch = static_cast<uint64_t>(batch_size);
    if (horizon > kMaximumIdctMultiplyAdds / horizon) {
        error = "StarVLA FAST inverse DCT exceeds the runtime work limit";
        return false;
    }
    uint64_t multiply_adds = horizon * horizon;
    if (action_dim > kMaximumIdctMultiplyAdds / multiply_adds) {
        error = "StarVLA FAST inverse DCT exceeds the runtime work limit";
        return false;
    }
    multiply_adds *= action_dim;
    if (batch > kMaximumIdctMultiplyAdds / multiply_adds) {
        error = "StarVLA FAST inverse DCT exceeds the runtime work limit";
        return false;
    }
    return true;
}

} // namespace

FastCodec::FastCodec(FastCodecConfig config, std::vector<std::vector<uint8_t>> token_bytes,
                     std::vector<int32_t> fast_to_vlm_id)
    : config_(config), token_bytes_(std::move(token_bytes)),
      fast_to_vlm_id_(std::move(fast_to_vlm_id)) {
    vlm_to_fast_id_.reserve(fast_to_vlm_id_.size());
    for (size_t fast_id = 0; fast_id < fast_to_vlm_id_.size(); ++fast_id) {
        vlm_to_fast_id_.emplace_back(fast_to_vlm_id_[fast_id], static_cast<int32_t>(fast_id));
    }
    std::sort(vlm_to_fast_id_.begin(), vlm_to_fast_id_.end());
}

std::unique_ptr<FastCodec> FastCodec::create(FastCodecConfig config,
                                             std::vector<std::string> vocab_by_id,
                                             std::vector<int32_t> fast_to_vlm_id,
                                             std::string & error) {
    error.clear();
    if (!std::isfinite(config.scale) || config.scale == 0.0 || config.vocab_size == 0 ||
        config.time_horizon == 0 || config.action_dim == 0) {
        error = "StarVLA FAST codec dimensions and scale must be non-zero and finite";
        return nullptr;
    }
    if (config.vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        error = "StarVLA FAST vocabulary exceeds the int32 token-ID range";
        return nullptr;
    }
    if (vocab_by_id.size() != config.vocab_size ||
        fast_to_vlm_id.size() != config.vocab_size) {
        error = "StarVLA FAST codec vocabulary or action-token map has the wrong size";
        return nullptr;
    }
    size_t per_sample = 0;
    size_t total = 0;
    if (!checked_action_count(config, 1, per_sample, total, error)) {
        return nullptr;
    }
    std::unordered_set<int32_t> unique_vlm_ids;
    for (int32_t vlm_id : fast_to_vlm_id) {
        if (vlm_id < 0 || !unique_vlm_ids.insert(vlm_id).second) {
            error = "StarVLA FAST action-token VLM IDs must be unique and non-negative";
            return nullptr;
        }
    }
    std::vector<std::vector<uint8_t>> token_bytes;
    if (!compile_token_bytes(vocab_by_id, token_bytes, error)) {
        return nullptr;
    }
    return std::unique_ptr<FastCodec>(
        new FastCodec(config, std::move(token_bytes), std::move(fast_to_vlm_id)));
}

std::unique_ptr<FastCodec> FastCodec::create_compiled(
    FastCodecConfig config, std::vector<int32_t> token_offsets,
    std::vector<uint8_t> token_bytes, std::vector<int32_t> fast_to_vlm_id,
    std::string & error) {
    error.clear();
    if (!std::isfinite(config.scale) || config.scale == 0.0 ||
        config.vocab_size == 0 || config.time_horizon == 0 ||
        config.action_dim == 0 ||
        config.vocab_size >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        error = "StarVLA FAST compiled codec dimensions and scale are invalid";
        return nullptr;
    }
    if (config.vocab_size == std::numeric_limits<size_t>::max() ||
        token_offsets.size() != config.vocab_size + 1U ||
        fast_to_vlm_id.size() != config.vocab_size ||
        token_offsets.empty() || token_offsets.front() != 0 ||
        token_offsets.back() < 0 ||
        static_cast<size_t>(token_offsets.back()) != token_bytes.size()) {
        error = "StarVLA FAST compiled codec tensor shapes are incompatible";
        return nullptr;
    }
    size_t per_sample = 0;
    size_t total = 0;
    if (!checked_action_count(config, 1, per_sample, total, error)) {
        return nullptr;
    }

    std::unordered_set<int32_t> unique_vlm_ids;
    for (int32_t vlm_id : fast_to_vlm_id) {
        if (vlm_id < 0 || !unique_vlm_ids.insert(vlm_id).second) {
            error =
                "StarVLA FAST compiled action-token IDs must be unique and non-negative";
            return nullptr;
        }
    }

    std::vector<std::vector<uint8_t>> pieces;
    pieces.reserve(config.vocab_size);
    for (size_t index = 0; index < config.vocab_size; ++index) {
        const int32_t begin = token_offsets[index];
        const int32_t end = token_offsets[index + 1U];
        if (begin < 0 || end <= begin ||
            static_cast<size_t>(end) > token_bytes.size()) {
            error = "StarVLA FAST compiled codec offsets are not strictly increasing";
            return nullptr;
        }
        pieces.emplace_back(token_bytes.begin() + begin, token_bytes.begin() + end);
    }
    return std::unique_ptr<FastCodec>(
        new FastCodec(config, std::move(pieces), std::move(fast_to_vlm_id)));
}

std::unique_ptr<FastCodec> FastCodec::load_hf_assets(
    const std::filesystem::path & tokenizer_json,
    const std::filesystem::path & processor_config_json,
    const std::filesystem::path & action_token_map_json,
    size_t time_horizon, size_t action_dim, std::string & error) {
    error.clear();
    Json processor;
    Json tokenizer;
    Json action_map;
    if (!read_json(processor_config_json, processor, error) ||
        !read_json(tokenizer_json, tokenizer, error) ||
        !read_json(action_token_map_json, action_map, error)) {
        return nullptr;
    }

    FastCodecConfig config;
    std::vector<std::string> vocab_by_id;
    std::vector<int32_t> fast_to_vlm;
    size_t per_sample = 0;
    size_t total = 0;
    if (!parse_processor_config(processor, time_horizon, action_dim, config, error) ||
        !checked_action_count(config, 1, per_sample, total, error) ||
        !parse_tokenizer_vocab(tokenizer, config.vocab_size, vocab_by_id, error) ||
        !parse_action_map(action_map, config.vocab_size, fast_to_vlm, error)) {
        return nullptr;
    }
    return create(config, std::move(vocab_by_id), std::move(fast_to_vlm), error);
}

const FastCodecConfig & FastCodec::config() const {
    return config_;
}

const std::vector<int32_t> & FastCodec::fast_to_vlm_ids() const {
    return fast_to_vlm_id_;
}

bool FastCodec::map_fast_to_vlm(const std::vector<int32_t> & fast_ids,
                                std::vector<int32_t> & vlm_ids, std::string & error) const {
    vlm_ids.clear();
    error.clear();
    if (fast_ids.size() > kMaximumTokenSequence) {
        error = "StarVLA FAST token sequence exceeds the runtime length limit";
        return false;
    }
    vlm_ids.reserve(fast_ids.size());
    for (int32_t fast_id : fast_ids) {
        if (fast_id < 0 || static_cast<size_t>(fast_id) >= fast_to_vlm_id_.size()) {
            error = "StarVLA FAST token ID is outside the codec vocabulary";
            vlm_ids.clear();
            return false;
        }
        vlm_ids.push_back(fast_to_vlm_id_[static_cast<size_t>(fast_id)]);
    }
    return true;
}

bool FastCodec::map_vlm_to_fast(const std::vector<int32_t> & vlm_ids,
                                std::vector<int32_t> & fast_ids, std::string & error) const {
    fast_ids.clear();
    error.clear();
    if (vlm_ids.size() > kMaximumTokenSequence) {
        error = "StarVLA FAST action-token sequence exceeds the runtime length limit";
        return false;
    }
    fast_ids.reserve(vlm_ids.size());
    for (int32_t vlm_id : vlm_ids) {
        const auto found = std::lower_bound(
            vlm_to_fast_id_.begin(), vlm_to_fast_id_.end(), vlm_id,
            [](const std::pair<int32_t, int32_t> & entry, int32_t value) {
                return entry.first < value;
            });
        if (found == vlm_to_fast_id_.end() || found->first != vlm_id) {
            error = "Qwen token ID is not present in the StarVLA FAST action-token map";
            fast_ids.clear();
            return false;
        }
        fast_ids.push_back(found->second);
    }
    return true;
}

bool FastCodec::extract_fast_tokens(const std::vector<int32_t> & generated_ids,
                                    std::vector<int32_t> & fast_ids,
                                    std::string & error) const {
    fast_ids.clear();
    error.clear();
    if (generated_ids.size() > kMaximumGeneratedSequence) {
        error = "Qwen generated sequence exceeds the StarVLA FAST runtime length limit";
        return false;
    }
    for (int32_t vlm_id : generated_ids) {
        const auto found = std::lower_bound(
            vlm_to_fast_id_.begin(), vlm_to_fast_id_.end(), vlm_id,
            [](const std::pair<int32_t, int32_t> & entry, int32_t value) {
                return entry.first < value;
            });
        if (found != vlm_to_fast_id_.end() && found->first == vlm_id) {
            fast_ids.push_back(found->second);
        }
    }
    return true;
}

bool FastCodec::byte_level_decode(const std::vector<int32_t> & fast_ids,
                                  std::vector<uint32_t> & codepoints,
                                  std::string & error) const {
    codepoints.clear();
    error.clear();
    if (fast_ids.size() > kMaximumTokenSequence) {
        error = "StarVLA FAST token sequence exceeds the runtime length limit";
        return false;
    }
    size_t byte_count = 0;
    for (int32_t fast_id : fast_ids) {
        if (fast_id < 0 || static_cast<size_t>(fast_id) >= token_bytes_.size()) {
            error = "StarVLA FAST token ID is outside the ByteLevel BPE vocabulary";
            return false;
        }
        const size_t piece_size = token_bytes_[static_cast<size_t>(fast_id)].size();
        if (byte_count > std::numeric_limits<size_t>::max() - piece_size) {
            error = "StarVLA FAST ByteLevel output size overflows size_t";
            return false;
        }
        byte_count += piece_size;
        if (byte_count > kMaximumDecodedBytes) {
            error = "StarVLA FAST ByteLevel decode exceeds the runtime byte limit";
            return false;
        }
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(byte_count);
    for (int32_t fast_id : fast_ids) {
        const auto & piece = token_bytes_[static_cast<size_t>(fast_id)];
        bytes.insert(bytes.end(), piece.begin(), piece.end());
    }
    decode_utf8_lossy(bytes, codepoints);
    return true;
}

bool FastCodec::decode_fast_tokens(const std::vector<std::vector<int32_t>> & batch_fast_ids,
                                   FastDecodeResult & result, std::string & error) const {
    result = {};
    error.clear();
    if (batch_fast_ids.empty()) {
        error = "StarVLA FAST decode batch must contain at least one sequence";
        return false;
    }
    if (batch_fast_ids.size() > kMaximumBatchSize) {
        error = "StarVLA FAST decode batch exceeds the runtime size limit";
        return false;
    }
    for (const auto & fast_ids : batch_fast_ids) {
        if (fast_ids.size() > kMaximumTokenSequence) {
            error = "StarVLA FAST token sequence exceeds the runtime length limit";
            return false;
        }
    }
    size_t per_sample = 0;
    size_t total = 0;
    if (!checked_action_count(config_, batch_fast_ids.size(), per_sample, total, error)) {
        return false;
    }

    result.batch_size = batch_fast_ids.size();
    result.time_horizon = config_.time_horizon;
    result.action_dim = config_.action_dim;
    result.actions.assign(total, 0.0);

    const double dc_scale = 1.0 / std::sqrt(static_cast<double>(config_.time_horizon));
    const double ac_scale = std::sqrt(2.0 / static_cast<double>(config_.time_horizon));
    for (size_t batch = 0; batch < batch_fast_ids.size(); ++batch) {
        std::vector<uint32_t> codepoints;
        std::string sequence_error;
        if (!byte_level_decode(batch_fast_ids[batch], codepoints, sequence_error) ||
            codepoints.size() != per_sample) {
            error = "StarVLA FAST sequence " + std::to_string(batch) + ": " +
                    (sequence_error.empty() ? "decoded DCT coefficient shape mismatch"
                                            : sequence_error);
            result = {};
            return false;
        }

        for (size_t action = 0; action < config_.action_dim; ++action) {
            const double dc =
                (static_cast<double>(codepoints[action]) + config_.min_token) / config_.scale;
            for (size_t time = 0; time < config_.time_horizon; ++time) {
                double value = dc_scale * dc;
                for (size_t frequency = 1; frequency < config_.time_horizon; ++frequency) {
                    const size_t coefficient_index = frequency * config_.action_dim + action;
                    const double coefficient =
                        (static_cast<double>(codepoints[coefficient_index]) + config_.min_token) /
                        config_.scale;
                    const double angle = kPi * static_cast<double>(frequency) *
                                         static_cast<double>(2U * time + 1U) /
                                         (2.0 * static_cast<double>(config_.time_horizon));
                    value += ac_scale * coefficient * std::cos(angle);
                }
                result.actions[batch * per_sample + time * config_.action_dim + action] = value;
            }
        }
    }
    return true;
}

bool FastCodec::decode_vlm_action_tokens(
    const std::vector<std::vector<int32_t>> & batch_vlm_ids, FastDecodeResult & result,
    std::string & error) const {
    if (batch_vlm_ids.empty() || batch_vlm_ids.size() > kMaximumBatchSize) {
        result = {};
        error = "StarVLA FAST action-token batch is empty or exceeds the runtime size limit";
        return false;
    }
    std::vector<std::vector<int32_t>> batch_fast_ids;
    batch_fast_ids.reserve(batch_vlm_ids.size());
    for (const auto & vlm_ids : batch_vlm_ids) {
        std::vector<int32_t> fast_ids;
        if (!map_vlm_to_fast(vlm_ids, fast_ids, error)) {
            result = {};
            return false;
        }
        batch_fast_ids.push_back(std::move(fast_ids));
    }
    return decode_fast_tokens(batch_fast_ids, result, error);
}

bool FastCodec::decode_generated_tokens(
    const std::vector<std::vector<int32_t>> & batch_generated_ids, FastDecodeResult & result,
    std::string & error) const {
    if (batch_generated_ids.empty() || batch_generated_ids.size() > kMaximumBatchSize) {
        result = {};
        error = "StarVLA FAST generated-token batch is empty or exceeds the runtime size limit";
        return false;
    }
    std::vector<std::vector<int32_t>> batch_fast_ids;
    batch_fast_ids.reserve(batch_generated_ids.size());
    for (const auto & generated_ids : batch_generated_ids) {
        std::vector<int32_t> fast_ids;
        if (!extract_fast_tokens(generated_ids, fast_ids, error)) {
            result = {};
            return false;
        }
        batch_fast_ids.push_back(std::move(fast_ids));
    }
    return decode_fast_tokens(batch_fast_ids, result, error);
}

} // namespace robotcpp::starvla
