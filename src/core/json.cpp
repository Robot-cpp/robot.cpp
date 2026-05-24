#include "core/json.h"

#include "core/error.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace vlacpp {
namespace {

#define VLACPP_STATUS_TRY(expr)       \
    do {                              \
        vlacpp_status s__ = (expr);   \
        if (s__ != VLACPP_STATUS_OK)  \
            return s__;               \
    } while (0)

class Parser {
public:
    explicit Parser(const std::string & input) : input_(input) {}

    vlacpp_status parse(JsonObject & out) {
        skip_ws();
        if (!consume('{')) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "expected JSON object");
        }
        skip_ws();
        if (consume('}')) {
            return VLACPP_STATUS_OK;
        }

        while (true) {
            std::string key;
            VLACPP_STATUS_TRY(parse_string(key));
            skip_ws();
            if (!consume(':')) {
                return fail(VLACPP_STATUS_PARSE_ERROR, "expected ':' after key");
            }
            JsonObject::Value value{JsonObject::Kind::String, {}, {}, {}};
            VLACPP_STATUS_TRY(parse_value(value));
            out.values_[key] = value;

            skip_ws();
            if (consume('}')) {
                return VLACPP_STATUS_OK;
            }
            if (!consume(',')) {
                return fail(VLACPP_STATUS_PARSE_ERROR, "expected ',' or '}'");
            }
            skip_ws();
        }
    }

private:
    void skip_ws() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    bool consume(char c) {
        skip_ws();
        if (pos_ < input_.size() && input_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    vlacpp_status parse_string(std::string & out) {
        skip_ws();
        if (pos_ >= input_.size() || input_[pos_] != '"') {
            return fail(VLACPP_STATUS_PARSE_ERROR, "expected string");
        }
        ++pos_;
        out.clear();
        while (pos_ < input_.size()) {
            char c = input_[pos_++];
            if (c == '"') {
                return VLACPP_STATUS_OK;
            }
            if (c == '\\') {
                if (pos_ >= input_.size()) {
                    return fail(VLACPP_STATUS_PARSE_ERROR, "unterminated escape");
                }
                char esc = input_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default:
                        return fail(VLACPP_STATUS_PARSE_ERROR, "unsupported JSON escape");
                }
            } else {
                out.push_back(c);
            }
        }
        return fail(VLACPP_STATUS_PARSE_ERROR, "unterminated string");
    }

    vlacpp_status parse_number(float & out) {
        skip_ws();
        const char * begin = input_.c_str() + pos_;
        char * end = nullptr;
        errno = 0;
        float value = std::strtof(begin, &end);
        if (end == begin || errno == ERANGE) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "invalid number");
        }
        pos_ += static_cast<size_t>(end - begin);
        out = value;
        return VLACPP_STATUS_OK;
    }

    vlacpp_status parse_value(JsonObject::Value & out) {
        skip_ws();
        if (pos_ >= input_.size()) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "unexpected end of input");
        }
        if (input_[pos_] == '"') {
            out.kind = JsonObject::Kind::String;
            return parse_string(out.text);
        }
        if (input_[pos_] == '[') {
            return parse_array(out);
        }
        float number = 0.0f;
        VLACPP_STATUS_TRY(parse_number(number));
        out.kind = JsonObject::Kind::Number;
        out.numbers = {number};
        return VLACPP_STATUS_OK;
    }

    vlacpp_status parse_array(JsonObject::Value & out) {
        if (!consume('[')) {
            return fail(VLACPP_STATUS_PARSE_ERROR, "expected array");
        }
        skip_ws();
        if (consume(']')) {
            out.kind = JsonObject::Kind::NumberArray;
            return VLACPP_STATUS_OK;
        }

        skip_ws();
        if (pos_ < input_.size() && input_[pos_] == '"') {
            out.kind = JsonObject::Kind::StringArray;
            while (true) {
                std::string item;
                VLACPP_STATUS_TRY(parse_string(item));
                out.strings.push_back(item);
                skip_ws();
                if (consume(']')) {
                    return VLACPP_STATUS_OK;
                }
                if (!consume(',')) {
                    return fail(VLACPP_STATUS_PARSE_ERROR, "expected ',' or ']'");
                }
            }
        }

        out.kind = JsonObject::Kind::NumberArray;
        while (true) {
            float item = 0.0f;
            VLACPP_STATUS_TRY(parse_number(item));
            out.numbers.push_back(item);
            skip_ws();
            if (consume(']')) {
                return VLACPP_STATUS_OK;
            }
            if (!consume(',')) {
                return fail(VLACPP_STATUS_PARSE_ERROR, "expected ',' or ']'");
            }
        }
    }

    const std::string & input_;
    size_t pos_ = 0;
};

vlacpp_status validate_config(const ModelConfig & config) {
    if (config.action_dim <= 0 || config.action_horizon <= 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "model config requires positive action_dim and action_horizon");
    }
    if (config.state_dim < 0) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "state_dim must be non-negative");
    }
    if ((!config.state_mean.empty() || !config.state_std.empty()) &&
        (config.state_mean.size() != static_cast<size_t>(config.state_dim) ||
            config.state_std.size() != static_cast<size_t>(config.state_dim))) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "state_mean and state_std must both match state_dim");
    }
    if ((!config.action_mean.empty() || !config.action_std.empty()) &&
        (config.action_mean.size() != static_cast<size_t>(config.action_dim) ||
            config.action_std.size() != static_cast<size_t>(config.action_dim))) {
        return fail(VLACPP_STATUS_PARSE_ERROR, "action_mean and action_std must both match action_dim");
    }
    return VLACPP_STATUS_OK;
}

} // namespace

bool JsonObject::has(const std::string & key) const {
    return values_.find(key) != values_.end();
}

std::string JsonObject::get_string(const std::string & key, const std::string & fallback) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.kind != Kind::String) {
        return fallback;
    }
    return it->second.text;
}

int JsonObject::get_int(const std::string & key, int fallback) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.numbers.empty()) {
        return fallback;
    }
    return static_cast<int>(it->second.numbers.front());
}

std::vector<std::string> JsonObject::get_string_array(const std::string & key) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.kind != Kind::StringArray) {
        return {};
    }
    return it->second.strings;
}

std::vector<float> JsonObject::get_float_array(const std::string & key) const {
    auto it = values_.find(key);
    if (it == values_.end() ||
        (it->second.kind != Kind::NumberArray && it->second.kind != Kind::Number)) {
        return {};
    }
    return it->second.numbers;
}

vlacpp_status parse_json_object(const std::string & text, JsonObject & out) {
    return Parser(text).parse(out);
}

vlacpp_status load_config_file(const std::string & path, ModelConfig & out) {
    std::ifstream file(path);
    if (!file) {
        return fail(VLACPP_STATUS_IO_ERROR, "failed to open model file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();

    JsonObject obj;
    vlacpp_status status = parse_json_object(buffer.str(), obj);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }

    out.model_type = obj.get_string("model_type", out.model_type);
    out.image_width = obj.get_int("image_width", out.image_width);
    out.image_height = obj.get_int("image_height", out.image_height);
    out.state_dim = obj.get_int("state_dim", out.state_dim);
    out.action_dim = obj.get_int("action_dim", out.action_dim);
    out.action_horizon = obj.get_int("action_horizon", out.action_horizon);
    out.max_token_len = obj.get_int("max_token_len", out.max_token_len);
    out.image_keys = obj.get_string_array("image_keys");
    out.state_mean = obj.get_float_array("state_mean");
    out.state_std = obj.get_float_array("state_std");
    out.action_mean = obj.get_float_array("action_mean");
    out.action_std = obj.get_float_array("action_std");
    out.openpi_action_width = obj.get_int("openpi_action_width", out.openpi_action_width);
    out.openpi_vision_width = obj.get_int("openpi_vision_width", out.openpi_vision_width);
    out.openpi_vision_patch_height = obj.get_int("openpi_vision_patch_height", out.openpi_vision_patch_height);
    out.openpi_vision_patch_width = obj.get_int("openpi_vision_patch_width", out.openpi_vision_patch_width);
    out.openpi_vision_layers = obj.get_int("openpi_vision_layers", out.openpi_vision_layers);
    out.openpi_language_width = obj.get_int("openpi_language_width", out.openpi_language_width);
    out.openpi_language_q_out = obj.get_int("openpi_language_q_out", out.openpi_language_q_out);
    out.openpi_language_kv_out = obj.get_int("openpi_language_kv_out", out.openpi_language_kv_out);
    out.openpi_language_mlp_width = obj.get_int("openpi_language_mlp_width", out.openpi_language_mlp_width);
    out.openpi_language_layers = obj.get_int("openpi_language_layers", out.openpi_language_layers);
    out.openpi_action_expert_width = obj.get_int("openpi_action_expert_width", out.openpi_action_expert_width);
    out.openpi_action_expert_q_out = obj.get_int("openpi_action_expert_q_out", out.openpi_action_expert_q_out);
    out.openpi_action_expert_kv_out = obj.get_int("openpi_action_expert_kv_out", out.openpi_action_expert_kv_out);
    out.openpi_action_expert_mlp_width = obj.get_int("openpi_action_expert_mlp_width", out.openpi_action_expert_mlp_width);
    out.openpi_action_expert_layers = obj.get_int("openpi_action_expert_layers", out.openpi_action_expert_layers);

    return validate_config(out);
}

} // namespace vlacpp
