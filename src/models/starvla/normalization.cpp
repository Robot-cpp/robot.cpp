#include "models/starvla/normalization.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace robotcpp::starvla {

namespace {

std::string profile_keys(const NormalizationConfig & config) {
    std::ostringstream out;
    for (size_t i = 0; i < config.profiles.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << config.profiles[i].key;
    }
    return out.str();
}

} // namespace

bool validate_normalization_config(const NormalizationConfig & config, int action_dim, std::string & error) {
    error.clear();
    if (action_dim <= 0) {
        error = "StarVLA normalization requires a positive action dimension";
        return false;
    }
    if (!std::isfinite(config.binary_threshold) || config.binary_threshold != 0.5f) {
        error = "StarVLA normalization binary threshold must use the canonical value 0.5";
        return false;
    }
    if (config.binary_comparison != "gt" && config.binary_comparison != "ge") {
        error = "StarVLA normalization binary comparison must be 'gt' or 'ge'";
        return false;
    }
    if (config.profiles.empty()) {
        error = "StarVLA policy has no normalization profiles";
        return false;
    }

    std::vector<uint8_t> dimension_kind(static_cast<size_t>(action_dim), 0);
    for (int32_t dim : config.continuous_dimensions) {
        if (dim < 0 || dim >= action_dim || dimension_kind[static_cast<size_t>(dim)] != 0) {
            error = "StarVLA continuous action dimensions are invalid or duplicated";
            return false;
        }
        dimension_kind[static_cast<size_t>(dim)] = 1;
    }
    for (int32_t dim : config.binary_dimensions) {
        if (dim < 0 || dim >= action_dim || dimension_kind[static_cast<size_t>(dim)] != 0) {
            error = "StarVLA binary action dimensions are invalid or duplicated";
            return false;
        }
        dimension_kind[static_cast<size_t>(dim)] = 2;
    }
    if (std::find(dimension_kind.begin(), dimension_kind.end(), uint8_t{0}) != dimension_kind.end()) {
        error = "StarVLA continuous and binary action dimensions must cover every action column";
        return false;
    }

    std::vector<std::string> seen_keys;
    seen_keys.reserve(config.profiles.size());
    for (const NormalizationProfile & profile : config.profiles) {
        if (profile.key.empty() || std::find(seen_keys.begin(), seen_keys.end(), profile.key) != seen_keys.end()) {
            error = "StarVLA normalization profile keys must be non-empty and unique";
            return false;
        }
        seen_keys.push_back(profile.key);
        if (profile.action_q01.size() != static_cast<size_t>(action_dim) ||
            profile.action_q99.size() != static_cast<size_t>(action_dim) ||
            profile.action_mask.size() != static_cast<size_t>(action_dim)) {
            error = "StarVLA normalization profile shape does not match action dimension: " + profile.key;
            return false;
        }
        for (int dim = 0; dim < action_dim; ++dim) {
            const size_t index = static_cast<size_t>(dim);
            if (!std::isfinite(profile.action_q01[index]) || !std::isfinite(profile.action_q99[index])) {
                error = "StarVLA normalization quantiles must be finite: " + profile.key;
                return false;
            }
            if (dimension_kind[index] == 1 && profile.action_q99[index] < profile.action_q01[index]) {
                error = "StarVLA normalization q99 must not be below q01: " + profile.key;
                return false;
            }
            if (dimension_kind[index] == 1 && profile.action_mask[index] == 0) {
                error = "StarVLA continuous action dimension is disabled by the normalization mask: " + profile.key;
                return false;
            }
            if (dimension_kind[index] == 2 && profile.action_mask[index] != 0) {
                error = "StarVLA binary action dimension must not use q01/q99 scaling: " + profile.key;
                return false;
            }
        }
    }
    return true;
}

const NormalizationProfile * resolve_normalization_profile(const NormalizationConfig & config,
                                                           const std::string & profile_key, std::string & error) {
    error.clear();
    if (profile_key.empty()) {
        if (config.profiles.size() == 1) {
            return &config.profiles.front();
        }
        error = "StarVLA policy has multiple normalization profiles; select one of: " + profile_keys(config);
        return nullptr;
    }
    for (const NormalizationProfile & profile : config.profiles) {
        if (profile.key == profile_key) {
            return &profile;
        }
    }
    error = "unknown StarVLA normalization profile '" + profile_key + "'; expected one of: " + profile_keys(config);
    return nullptr;
}

bool denormalize_actions(const NormalizationConfig & config, const std::string & profile_key,
                         const std::vector<float> & normalized, int horizon, int action_dim,
                         std::vector<float> & actions, std::string & error) {
    actions.clear();
    error.clear();
    if (!validate_normalization_config(config, action_dim, error)) {
        return false;
    }
    if (horizon <= 0 || normalized.size() != static_cast<size_t>(horizon) * static_cast<size_t>(action_dim)) {
        error = "StarVLA normalized action tensor has an incompatible shape";
        return false;
    }
    const NormalizationProfile * profile = resolve_normalization_profile(config, profile_key, error);
    if (profile == nullptr) {
        return false;
    }

    std::vector<uint8_t> is_binary(static_cast<size_t>(action_dim), 0);
    for (int32_t dim : config.binary_dimensions) {
        is_binary[static_cast<size_t>(dim)] = 1;
    }

    actions.resize(normalized.size());
    for (int step = 0; step < horizon; ++step) {
        for (int dim = 0; dim < action_dim; ++dim) {
            const size_t index = static_cast<size_t>(step) * static_cast<size_t>(action_dim) +
                                 static_cast<size_t>(dim);
            const float input_value = normalized[index];
            if (!std::isfinite(input_value)) {
                actions.clear();
                error = "StarVLA normalized actions must be finite";
                return false;
            }
            const float value =
                config.clip_actions ? std::clamp(input_value, -1.0f, 1.0f)
                                    : input_value;
            if (is_binary[static_cast<size_t>(dim)] != 0) {
                const bool active =
                    config.binary_comparison == "ge"
                        ? value >= config.binary_threshold
                        : value > config.binary_threshold;
                actions[index] = active ? 1.0f : 0.0f;
            } else {
                const float low = profile->action_q01[static_cast<size_t>(dim)];
                const float high = profile->action_q99[static_cast<size_t>(dim)];
                actions[index] = (value + 1.0f) * 0.5f * (high - low) + low;
            }
        }
    }
    return true;
}

} // namespace robotcpp::starvla
