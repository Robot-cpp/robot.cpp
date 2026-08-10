#include "models/model.h"

#include <array>

namespace robotcpp {
namespace {

struct model_type_entry {
    model_type type;
    const char * name;
};

constexpr std::array<model_type_entry, 3> MODEL_TYPES = {{
    {model_type::smolvla, "smolvla"},
    {model_type::pi0, "pi0"},
    {model_type::starvla, "starvla"},
}};

const model_type_entry * find_entry(model_type type) {
    for (const model_type_entry & entry : MODEL_TYPES) {
        if (entry.type == type) {
            return &entry;
        }
    }
    return nullptr;
}

} // namespace

const char * model_type_name(model_type type) {
    const model_type_entry * entry = find_entry(type);
    return entry ? entry->name : "unknown";
}

bool parse_model_type(const std::string & value, model_type & out) {
    for (const model_type_entry & entry : MODEL_TYPES) {
        if (value == entry.name) {
            out = entry.type;
            return true;
        }
    }
    return false;
}

bool is_starvla_model_type(model_type type) {
    return type == model_type::starvla;
}

} // namespace robotcpp
