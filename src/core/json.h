#pragma once

#include "core/types.h"
#include "vlacpp.h"

#include <map>
#include <string>
#include <vector>

namespace vlacpp {

class JsonObject {
public:
    bool has(const std::string & key) const;
    std::string get_string(const std::string & key, const std::string & fallback) const;
    int get_int(const std::string & key, int fallback) const;
    std::vector<std::string> get_string_array(const std::string & key) const;
    std::vector<float> get_float_array(const std::string & key) const;

private:
    friend vlacpp_status parse_json_object(const std::string &, JsonObject &);

public:
    // Internal parser representation. This intentionally supports only the
    // small JSON subset needed for model metadata and smoke-test configs.

    enum class Kind {
        String,
        Number,
        StringArray,
        NumberArray,
    };

    struct Value {
        Kind kind;
        std::string text;
        std::vector<std::string> strings;
        std::vector<float> numbers;
    };

    std::map<std::string, Value> values_;
};

vlacpp_status parse_json_object(const std::string & text, JsonObject & out);
vlacpp_status load_config_file(const std::string & path, ModelConfig & out);

} // namespace vlacpp
