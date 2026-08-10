#include "models/starvla/starvla_engine.h"

#include "ggml.h"
#include "gguf.h"
#include "models/starvla/fast_policy.h"
#include "models/starvla/groot_policy.h"
#include "models/starvla/groot_prompt.h"
#include "models/starvla/normalization.h"
#include "models/starvla/oft_image_preprocess.h"
#include "models/starvla/oft_policy.h"
#include "models/starvla/oft_prompt.h"
#include "models/starvla/pi_policy.h"
#include "models/starvla/pi_v3_policy.h"
#include "models/starvla/qwen3vl_bridge.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace robotcpp::starvla {

const char * starvla_variant_name(StarVLAVariant variant) noexcept {
    switch (variant) {
    case StarVLAVariant::qwen3_oft: return "qwen3_oft";
    case StarVLAVariant::qwen3_groot: return "qwen3_groot";
    case StarVLAVariant::qwen3_pi_v3: return "qwen3_pi_v3";
    case StarVLAVariant::qwen25_oft: return "qwen25_oft";
    case StarVLAVariant::qwen25_groot: return "qwen25_groot";
    case StarVLAVariant::qwen25_pi: return "qwen25_pi";
    case StarVLAVariant::qwen25_fast: return "qwen25_fast";
    }
    return "unknown";
}

const char * starvla_variant_framework(StarVLAVariant variant) noexcept {
    switch (variant) {
    case StarVLAVariant::qwen3_oft:
    case StarVLAVariant::qwen25_oft: return "oft";
    case StarVLAVariant::qwen3_groot:
    case StarVLAVariant::qwen25_groot: return "groot";
    case StarVLAVariant::qwen3_pi_v3: return "pi_v3";
    case StarVLAVariant::qwen25_pi: return "pi";
    case StarVLAVariant::qwen25_fast: return "fast";
    }
    return "unknown";
}

bool starvla_variant_from_metadata(const std::string & framework,
                                   const std::string & backbone,
                                   StarVLAVariant & variant) noexcept {
    if (backbone == "qwen3_vl") {
        if (framework == "oft") variant = StarVLAVariant::qwen3_oft;
        else if (framework == "groot") variant = StarVLAVariant::qwen3_groot;
        else if (framework == "pi_v3") variant = StarVLAVariant::qwen3_pi_v3;
        else return false;
        return true;
    }
    if (backbone == "qwen2_5_vl") {
        if (framework == "oft") variant = StarVLAVariant::qwen25_oft;
        else if (framework == "groot") variant = StarVLAVariant::qwen25_groot;
        else if (framework == "pi") variant = StarVLAVariant::qwen25_pi;
        else if (framework == "fast") variant = StarVLAVariant::qwen25_fast;
        else return false;
        return true;
    }
    return false;
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDefaultThreadCount = 4;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool read_policy_variant(const std::filesystem::path & path,
                         StarVLAVariant & variant, std::string & error) {
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context * gguf = gguf_init_from_file(path.string().c_str(), params);
    if (gguf == nullptr) {
        error = "failed to read StarVLA policy GGUF metadata";
        return false;
    }
    const auto read_string = [&](const char * key, std::string & value) {
        const int index = gguf_find_key(gguf, key);
        if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_STRING) {
            error = std::string("missing StarVLA policy metadata: ") + key;
            return false;
        }
        value = gguf_get_val_str(gguf, index);
        return true;
    };

    std::string framework;
    std::string backbone;
    const bool valid = read_string("starvla.framework", framework) &&
                       read_string("starvla.backbone.arch", backbone);
    gguf_free(gguf);
    if (!valid) {
        return false;
    }

    if (starvla_variant_from_metadata(framework, backbone, variant)) {
        return true;
    }
    error = "unsupported StarVLA variant: " + backbone + "/" + framework;
    return false;
}

bool is_plain_basename(const std::string & value) {
    if (value.empty() || value.find('\0') != std::string::npos) {
        return false;
    }
    const std::filesystem::path path(value);
    return !path.has_root_path() && !path.has_parent_path() && path.filename() == path &&
           value != "." && value != "..";
}

bool require_regular_file(const std::filesystem::path & path, const char * label,
                          std::string & error) {
    std::error_code status_error;
    const bool regular = std::filesystem::is_regular_file(path, status_error);
    if (!regular) {
        error = std::string("StarVLA ") + label + " is not a regular file: " + path.string();
        if (status_error) {
            error += " (" + status_error.message() + ")";
        }
        return false;
    }
    return true;
}

bool resolve_component_path(const std::filesystem::path & policy_path,
                            const std::string & metadata_filename,
                            const std::string & override_path, const char * label,
                            std::filesystem::path & resolved, std::string & error) {
    if (!is_plain_basename(metadata_filename)) {
        error = std::string("StarVLA policy ") + label +
                " filename must be a plain basename: " + metadata_filename;
        return false;
    }

    resolved = override_path.empty() ? policy_path.parent_path() / metadata_filename
                                     : std::filesystem::path(override_path);
    if (!override_path.empty() && override_path.find('\0') != std::string::npos) {
        error = std::string("StarVLA ") + label + " override contains an embedded NUL";
        return false;
    }
    if (resolved.filename().string() != metadata_filename) {
        error = std::string("StarVLA ") + label + " basename must match policy metadata '" +
                metadata_filename + "': " + resolved.string();
        return false;
    }
    return require_regular_file(resolved, label, error);
}

bool same_file(const std::filesystem::path & lhs, const std::filesystem::path & rhs) {
    std::error_code equivalent_error;
    return std::filesystem::equivalent(lhs, rhs, equivalent_error) && !equivalent_error;
}

bool validate_observation(const observation & obs, int image_count,
                          const std::vector<std::string> & image_names,
                          bool state_supported, const char * framework, std::string & error) {
    const std::string label = std::string("StarVLA ") + framework;
    if (obs.images.size() != static_cast<size_t>(image_count)) {
        error = label + " requires exactly " + std::to_string(image_count) +
                " image(s) in policy order";
        return false;
    }
    if (obs.task.empty()) {
        error = label + " task must not be empty";
        return false;
    }
    if (obs.task.find('\0') != std::string::npos) {
        error = label + " task contains an embedded NUL";
        return false;
    }
    if (!state_supported && !obs.state.empty()) {
        error = label + " released checkpoint does not support state input";
        return false;
    }
    for (float value : obs.state) {
        if (!std::isfinite(value)) {
            error = label + " state must contain only finite values";
            return false;
        }
    }

    for (size_t i = 0; i < obs.images.size(); ++i) {
        const model_image & image = obs.images[i];
        if (image.name != image_names[i]) {
            error = label + " image " + std::to_string(i) + " must be named '" +
                    image_names[i] + "'";
            return false;
        }
        if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
            error = label + " image '" + image.name + "' has invalid data or dimensions";
            return false;
        }
        if (image.channels != 3) {
            error = label + " image '" + image.name + "' must use interleaved RGB channels";
            return false;
        }
        if (image.width > std::numeric_limits<int>::max() / image.channels) {
            error = label + " image '" + image.name + "' row size overflows";
            return false;
        }
        const int packed_stride = image.width * image.channels;
        if (image.stride_bytes < 0 ||
            (image.stride_bytes != 0 && image.stride_bytes < packed_stride)) {
            error = label + " image '" + image.name +
                    "' stride is smaller than a packed RGB row";
            return false;
        }
    }
    return true;
}

template <typename PolicyConfig>
bool prepare_qwen_images(const observation & obs, const PolicyConfig & config,
                         const char * framework,
                         std::vector<std::vector<uint8_t>> & processed_images,
                         std::vector<Qwen3VLImageView> & qwen_images,
                         std::string & error) {
    processed_images.clear();
    qwen_images.clear();
    processed_images.resize(obs.images.size());
    qwen_images.reserve(obs.images.size());
    for (size_t i = 0; i < obs.images.size(); ++i) {
        const model_image & image = obs.images[i];
        int target_width = 0;
        int target_height = 0;
        int image_token_count = 0;
        std::string preprocess_error;
        if (!preprocess_qwen3vl_rgb(
                image.data, image.width, image.height, image.channels, image.stride_bytes,
                config.image_patch_size, config.image_spatial_merge_size,
                config.image_processor_min_pixels, config.image_processor_max_pixels,
                processed_images[i], target_width, target_height, image_token_count,
                preprocess_error)) {
            error = std::string("failed to preprocess StarVLA ") + framework + " image '" +
                    image.name + "': " + preprocess_error;
            return false;
        }
        const uint64_t expected_bytes = static_cast<uint64_t>(target_width) *
                                        static_cast<uint64_t>(target_height) * 3;
        if (expected_bytes != processed_images[i].size() ||
            image_token_count < config.image_min_token_count ||
            image_token_count > config.image_max_token_count) {
            error = std::string("StarVLA ") + framework +
                    " image preprocessor returned an incompatible dynamic grid";
            return false;
        }
        Qwen3VLImageView view;
        view.data = processed_images[i].data();
        view.width = target_width;
        view.height = target_height;
        view.channels = 3;
        view.stride_bytes = target_width * 3;
        qwen_images.push_back(view);
    }
    return true;
}

bool prepare_pi_qwen_images(
    const observation & obs, const PIPolicyConfig & config,
    std::vector<std::vector<uint8_t>> & pre_resized_images,
    std::vector<std::vector<uint8_t>> & processed_images,
    std::vector<Qwen3VLImageView> & qwen_images, std::string & error) {
    pre_resized_images.clear();
    processed_images.clear();
    qwen_images.clear();
    pre_resized_images.resize(obs.images.size());
    processed_images.resize(obs.images.size());
    qwen_images.reserve(obs.images.size());
    for (size_t i = 0; i < obs.images.size(); ++i) {
        const model_image & image = obs.images[i];
        std::string preprocess_error;
        if (!resize_torchvision_bicubic_aa_rgb(
                image.data, image.width, image.height, image.stride_bytes,
                config.image_framework_inference_pre_resize_width,
                config.image_framework_inference_pre_resize_height,
                pre_resized_images[i], preprocess_error)) {
            error = "failed to pre-resize StarVLA PI image '" + image.name +
                    "': " + preprocess_error;
            return false;
        }

        int target_width = 0;
        int target_height = 0;
        int image_token_count = 0;
        if (!preprocess_qwen3vl_rgb(
                pre_resized_images[i].data(),
                config.image_framework_inference_pre_resize_width,
                config.image_framework_inference_pre_resize_height, 3,
                config.image_framework_inference_pre_resize_width * 3,
                config.image_patch_size, config.image_spatial_merge_size,
                config.image_processor_min_pixels,
                config.image_processor_max_pixels, processed_images[i],
                target_width, target_height, image_token_count,
                preprocess_error)) {
            error = "failed to preprocess StarVLA PI image '" + image.name +
                    "': " + preprocess_error;
            return false;
        }
        const uint64_t expected_bytes =
            static_cast<uint64_t>(target_width) *
            static_cast<uint64_t>(target_height) * 3;
        if (expected_bytes != processed_images[i].size() ||
            image_token_count < config.image_min_token_count ||
            image_token_count > config.image_max_token_count) {
            error =
                "StarVLA PI image preprocessor returned an incompatible dynamic grid";
            return false;
        }
        Qwen3VLImageView view;
        view.data = processed_images[i].data();
        view.width = target_width;
        view.height = target_height;
        view.channels = 3;
        view.stride_bytes = target_width * 3;
        qwen_images.push_back(view);
    }
    return true;
}

} // namespace

struct StarVLAEngine::Impl {
    StarVLAEngineConfig options;
    StarVLAVariant variant = StarVLAVariant::qwen3_oft;
    std::filesystem::path policy_path;
    std::filesystem::path text_path;
    std::filesystem::path mmproj_path;
    std::string default_unnorm_key;
    const NormalizationConfig * normalization = nullptr;
    std::mt19937_64 noise_rng;
    // Destroy the policy scheduler/backends before Qwen releases llama's global backend state.
    std::unique_ptr<Qwen3VLBridge> qwen;
    std::unique_ptr<OFTPolicy> oft_policy;
    std::unique_ptr<GR00TPolicy> groot_policy;
    std::unique_ptr<PIPolicy> pi_policy;
    std::unique_ptr<PIV3Policy> pi_v3_policy;
    std::unique_ptr<FastPolicy> fast_policy;
};

StarVLAEngine::StarVLAEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

StarVLAEngine::~StarVLAEngine() = default;

std::unique_ptr<StarVLAEngine> StarVLAEngine::load(const StarVLAEngineConfig & config,
                                                  std::string & error) {
    error.clear();
    if (config.policy_path.empty() || config.policy_path.find('\0') != std::string::npos) {
        error = "StarVLA policy path is required and must not contain an embedded NUL";
        return nullptr;
    }
    if (config.n_ctx <= 0 || config.n_batch <= 0 || config.n_threads < 0) {
        error = "StarVLA n_ctx/n_batch must be positive and n_threads must be non-negative";
        return nullptr;
    }

    std::unique_ptr<Impl> impl(new Impl());
    impl->options = config;
    const int effective_threads = config.n_threads > 0 ? config.n_threads : kDefaultThreadCount;
    impl->policy_path = std::filesystem::path(config.policy_path);
    if (!require_regular_file(impl->policy_path, "policy GGUF", error)) {
        return nullptr;
    }
    if (!read_policy_variant(impl->policy_path, impl->variant, error)) {
        return nullptr;
    }
    const bool is_oft = impl->variant == StarVLAVariant::qwen3_oft ||
                        impl->variant == StarVLAVariant::qwen25_oft;
    const bool is_groot = impl->variant == StarVLAVariant::qwen3_groot ||
                          impl->variant == StarVLAVariant::qwen25_groot;
    const bool is_pi = impl->variant == StarVLAVariant::qwen25_pi;
    const bool is_pi_v3 = impl->variant == StarVLAVariant::qwen3_pi_v3;
    const bool is_fast = impl->variant == StarVLAVariant::qwen25_fast;
    const char * framework = starvla_variant_framework(impl->variant);

    std::string bundle_uuid;
    std::string text_filename;
    std::string mmproj_filename;
    std::string qwen_backbone_arch;
    int qwen_hidden_size = 0;
    int qwen_input_embedding_size = 0;
    int qwen_vocab_size = 0;
    int image_count = 0;
    int image_min_tokens = 0;
    int image_max_tokens = 0;
    int image_spatial_merge_size = 0;
    const NormalizationConfig * normalization = nullptr;
    if (is_oft) {
        impl->oft_policy = OFTPolicy::load(impl->policy_path.string(), effective_threads,
                                           config.verbosity, error);
        if (impl->oft_policy == nullptr) {
            error = "failed to load StarVLA OFT policy: " + error;
            return nullptr;
        }
        const OFTPolicyConfig & policy = impl->oft_policy->config();
        bundle_uuid = policy.bundle_uuid;
        text_filename = policy.text_filename;
        mmproj_filename = policy.mmproj_filename;
        qwen_backbone_arch = policy.backbone_arch;
        qwen_hidden_size = policy.input_dim;
        qwen_input_embedding_size = policy.input_embedding_dim;
        qwen_vocab_size = policy.vocab_size;
        image_count = policy.image_count;
        image_min_tokens = policy.image_min_token_count;
        image_max_tokens = policy.image_max_token_count;
        image_spatial_merge_size = policy.image_spatial_merge_size;
        normalization = &policy.normalization;
    } else if (is_groot) {
        impl->groot_policy = GR00TPolicy::load(impl->policy_path.string(), effective_threads,
                                               config.verbosity, error);
        if (impl->groot_policy == nullptr) {
            error = "failed to load StarVLA GR00T policy: " + error;
            return nullptr;
        }
        const GR00TPolicyConfig & policy = impl->groot_policy->config();
        bundle_uuid = policy.bundle_uuid;
        text_filename = policy.text_filename;
        mmproj_filename = policy.mmproj_filename;
        qwen_backbone_arch = policy.backbone_arch;
        qwen_hidden_size = policy.qwen_hidden_dim;
        qwen_input_embedding_size = policy.qwen_input_embedding_dim;
        qwen_vocab_size = policy.qwen_vocab_size;
        image_count = policy.image_count;
        image_min_tokens = policy.image_min_token_count;
        image_max_tokens = policy.image_max_token_count;
        image_spatial_merge_size = policy.image_spatial_merge_size;
        normalization = &policy.normalization;
    } else if (is_pi) {
        impl->pi_policy = PIPolicy::load(impl->policy_path.string(),
                                         effective_threads, config.verbosity,
                                         error);
        if (impl->pi_policy == nullptr) {
            error = "failed to load StarVLA PI policy: " + error;
            return nullptr;
        }
        const PIPolicyConfig & policy = impl->pi_policy->config();
        bundle_uuid = policy.bundle_uuid;
        text_filename = policy.text_filename;
        mmproj_filename = policy.mmproj_filename;
        qwen_backbone_arch = policy.backbone_arch;
        qwen_hidden_size = policy.qwen_hidden_dim;
        qwen_input_embedding_size = policy.qwen_input_embedding_dim;
        qwen_vocab_size = policy.qwen_vocab_size;
        image_count = policy.image_count;
        image_min_tokens = policy.image_min_token_count;
        image_max_tokens = policy.image_max_token_count;
        image_spatial_merge_size = policy.image_spatial_merge_size;
        normalization = &policy.normalization;
    } else if (is_pi_v3) {
        impl->pi_v3_policy = PIV3Policy::load(impl->policy_path.string(), effective_threads,
                                              config.verbosity, error);
        if (impl->pi_v3_policy == nullptr) {
            error = "failed to load StarVLA PI_v3 policy: " + error;
            return nullptr;
        }
        const PIV3PolicyConfig & policy = impl->pi_v3_policy->config();
        bundle_uuid = policy.bundle_uuid;
        text_filename = policy.text_filename;
        mmproj_filename = policy.mmproj_filename;
        qwen_backbone_arch = policy.backbone_arch;
        qwen_hidden_size = policy.qwen_hidden_dim;
        qwen_input_embedding_size = policy.qwen_input_embedding_dim;
        qwen_vocab_size = policy.qwen_vocab_size;
        image_count = policy.image_count;
        image_min_tokens = policy.image_min_token_count;
        image_max_tokens = policy.image_max_token_count;
        image_spatial_merge_size = policy.image_spatial_merge_size;
        normalization = &policy.normalization;
    } else {
        impl->fast_policy =
            FastPolicy::load(impl->policy_path.string(), config.verbosity, error);
        if (impl->fast_policy == nullptr) {
            error = "failed to load StarVLA FAST policy: " + error;
            return nullptr;
        }
        const FastPolicyConfig & policy = impl->fast_policy->config();
        bundle_uuid = policy.bundle_uuid;
        text_filename = policy.text_filename;
        mmproj_filename = policy.mmproj_filename;
        qwen_backbone_arch = policy.backbone_arch;
        qwen_hidden_size = policy.qwen_hidden_dim;
        qwen_input_embedding_size = policy.qwen_input_embedding_dim;
        qwen_vocab_size = policy.qwen_vocab_size;
        image_count = policy.image_count;
        image_min_tokens = policy.image_min_token_count;
        image_max_tokens = policy.image_max_token_count;
        image_spatial_merge_size = policy.image_spatial_merge_size;
        normalization = &policy.normalization;
        if (config.n_ctx <
            static_cast<int>(policy.generation_max_length)) {
            error =
                "StarVLA FAST --n-ctx must be at least the official max_length=2048";
            return nullptr;
        }
    }
    if (!resolve_component_path(impl->policy_path, text_filename,
                                config.text_path_override, "text GGUF", impl->text_path,
                                error) ||
        !resolve_component_path(impl->policy_path, mmproj_filename,
                                config.mmproj_path_override, "mmproj GGUF", impl->mmproj_path,
                                error)) {
        return nullptr;
    }
    if (same_file(impl->policy_path, impl->text_path) ||
        same_file(impl->policy_path, impl->mmproj_path) ||
        same_file(impl->text_path, impl->mmproj_path)) {
        error = "StarVLA policy, text, and mmproj GGUF paths must identify three distinct files";
        return nullptr;
    }

    impl->normalization = normalization;
    if (!config.unnorm_key.empty() || normalization->profiles.size() == 1) {
        std::string profile_error;
        const NormalizationProfile * profile = resolve_normalization_profile(
            *normalization, config.unnorm_key, profile_error);
        if (profile == nullptr) {
            error = std::string("failed to select StarVLA ") + framework +
                    " normalization profile: " + profile_error;
            return nullptr;
        }
        impl->default_unnorm_key = profile->key;
    }

    Qwen3VLBridgeConfig qwen_config;
    qwen_config.text_path = impl->text_path.string();
    qwen_config.mmproj_path = impl->mmproj_path.string();
    qwen_config.bundle_uuid = bundle_uuid;
    qwen_config.hidden_size = qwen_hidden_size;
    qwen_config.input_embedding_size = qwen_input_embedding_size;
    qwen_config.vocab_size = qwen_vocab_size;
    if (is_oft) {
        qwen_config.action_token = impl->oft_policy->config().prompt.action_token;
        qwen_config.action_token_id = impl->oft_policy->config().action_token_id;
    } else {
        qwen_config.action_token.clear();
        qwen_config.action_token_id = -1;
    }
    qwen_config.expected_image_count = image_count;
    qwen_config.image_min_tokens = image_min_tokens;
    qwen_config.image_max_tokens = image_max_tokens;
    qwen_config.image_spatial_merge_size = image_spatial_merge_size;
    qwen_config.n_ctx = config.n_ctx;
    qwen_config.n_batch = config.n_batch;
    qwen_config.n_threads = effective_threads;
    qwen_config.verbosity = config.verbosity;
    qwen_config.flash_text_attention = is_oft || is_pi;
    qwen_config.bf16_residual_layer_boundaries = is_groot;
    qwen_config.disable_text_backend_native_graphs = true;
    qwen_config.disable_vision_backend_native_graphs = true;
    if (qwen_config.input_embedding_size <= 0) {
        error = "StarVLA Qwen-VL input embedding size is invalid";
        return nullptr;
    }
    impl->qwen = Qwen3VLBridge::load(qwen_config, error);
    if (impl->qwen == nullptr) {
        error = "failed to load StarVLA Qwen-VL components: " + error;
        return nullptr;
    }
    const QwenVLArchitecture expected_architecture =
        qwen_backbone_arch == "qwen2_5_vl"
            ? QwenVLArchitecture::qwen2_5_vl
            : QwenVLArchitecture::qwen3_vl;
    if (impl->qwen->architecture() != expected_architecture) {
        error = "StarVLA policy backbone metadata does not match the Qwen-VL components";
        return nullptr;
    }

    if (!is_oft && !is_fast) {
        if (config.noise_seed >= 0) {
            impl->noise_rng.seed(static_cast<uint64_t>(config.noise_seed));
        } else {
            std::random_device device;
            std::seed_seq seed{
                device(), device(), device(), device(),
                static_cast<unsigned int>(Clock::now().time_since_epoch().count())};
            impl->noise_rng.seed(seed);
        }
    }

    if (config.verbosity >= 1) {
        const char * variant_name = starvla_variant_name(impl->variant);
        const char * policy_backend =
            is_oft ? impl->oft_policy->backend_name()
                   : (is_groot ? impl->groot_policy->backend_name()
                               : (is_pi ? impl->pi_policy->backend_name()
                                        : (is_pi_v3
                                               ? impl->pi_v3_policy->backend_name()
                                               : impl->fast_policy->backend_name())));
        std::fprintf(stderr,
                     "%s: variant=%s policy=%s text=%s mmproj=%s profile=%s "
                     "qwen_backend=%s policy_backend=%s\n",
                     __func__, variant_name, impl->policy_path.string().c_str(),
                     impl->text_path.string().c_str(),
                     impl->mmproj_path.string().c_str(),
                     impl->default_unnorm_key.empty() ? "<request-required>"
                                                      : impl->default_unnorm_key.c_str(),
                     impl->qwen->backend_name(), policy_backend);
    }
    return std::unique_ptr<StarVLAEngine>(new StarVLAEngine(std::move(impl)));
}

bool StarVLAEngine::predict(const observation & obs, StarVLAEngineResult & result,
                            std::string & error) {
    return predict_impl(obs, nullptr, result, error);
}

bool StarVLAEngine::predict_with_noise(const observation & obs,
                                       const std::vector<float> & initial_noise,
                                       StarVLAEngineResult & result,
                                       std::string & error) {
    return predict_impl(obs, &initial_noise, result, error);
}

bool StarVLAEngine::predict_impl(const observation & obs,
                                 const std::vector<float> * initial_noise,
                                 StarVLAEngineResult & result,
                                 std::string & error) {
    result = StarVLAEngineResult{};
    error.clear();
    const Clock::time_point total_start = Clock::now();
    const auto fail = [&]() {
        result.actions.clear();
        result.normalized_actions.clear();
        result.action_queries.clear();
        result.generated_token_ids.clear();
        result.action_token_ids.clear();
        result.fast_token_ids.clear();
        result.instruction.clear();
        result.unnorm_key_used.clear();
        result.chunk_size = 0;
        result.action_dim = 0;
        result.timings.total_ms = elapsed_ms(total_start, Clock::now());
        return false;
    };

    if (impl_ == nullptr || impl_->qwen == nullptr ||
        (!impl_->oft_policy && !impl_->groot_policy && !impl_->pi_policy &&
         !impl_->pi_v3_policy && !impl_->fast_policy)) {
        error = "StarVLA engine is not initialized";
        return fail();
    }

    if (impl_->normalization == nullptr) {
        error = "StarVLA normalization metadata is not initialized";
        return fail();
    }
    std::string profile_error;
    const NormalizationProfile * profile = resolve_normalization_profile(
        *impl_->normalization, impl_->default_unnorm_key, profile_error);
    if (profile == nullptr) {
        error = "failed to select StarVLA normalization profile: " + profile_error;
        return fail();
    }
    result.unnorm_key_used = profile->key;

    const auto make_noise = [&](size_t count, std::vector<float> & noise) {
        if (initial_noise != nullptr) {
            if (initial_noise->size() != count ||
                !std::all_of(initial_noise->begin(), initial_noise->end(),
                             [](float value) { return std::isfinite(value); })) {
                error = "initial noise has an incompatible shape or non-finite value";
                return false;
            }
            noise = *initial_noise;
            return true;
        }
        noise.resize(count);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float & value : noise) {
            value = ggml_bf16_to_fp32(
                ggml_fp32_to_bf16(normal(impl_->noise_rng)));
        }
        return true;
    };

    if (impl_->variant == StarVLAVariant::qwen25_fast) {
        if (initial_noise != nullptr) {
            error = "StarVLA FAST does not use diffusion noise";
            return fail();
        }
        const FastPolicyConfig & config = impl_->fast_policy->config();
        if (!validate_observation(obs, config.image_count, config.image_names,
                                  false, "FAST", error)) {
            return fail();
        }

        Clock::time_point stage_start = Clock::now();
        std::vector<std::vector<uint8_t>> processed_images;
        std::vector<Qwen3VLImageView> qwen_images;
        if (!prepare_qwen_images(obs, config, "FAST", processed_images,
                                 qwen_images, error)) {
            return fail();
        }
        result.timings.image_preprocess_ms =
            elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!build_fast_instruction(config.cot_template, obs.task,
                                    result.instruction, error)) {
            error = "failed to build the StarVLA FAST prompt: " + error;
            return fail();
        }
        result.timings.prompt_ms = elapsed_ms(stage_start, Clock::now());

        QwenVLGenerationConfig generation;
        generation.max_length = config.generation_max_length;
        generation.eos_token_ids = config.generation_eos_token_ids;
        generation.top_k = config.generation_top_k;
        generation.repetition_penalty =
            config.generation_repetition_penalty;
        QwenVLGenerationResult generated;
        stage_start = Clock::now();
        if (!impl_->qwen->generate_autoregressive(
                qwen_images, result.instruction, generation, generated,
                error)) {
            error = "StarVLA FAST Qwen2.5-VL generation failed: " + error;
            return fail();
        }
        result.timings.qwen3vl_ms = elapsed_ms(stage_start, Clock::now());
        if (generated.prompt_token_count == 0 ||
            generated.full_sequence.size() <
                generated.prompt_token_count ||
            generated.full_sequence.size() >
                config.generation_max_length) {
            error =
                "StarVLA FAST Qwen2.5-VL returned an incompatible generated sequence";
            return fail();
        }
        result.generated_token_ids = std::move(generated.full_sequence);

        stage_start = Clock::now();
        if (!impl_->fast_policy->decode_generated(
                result.generated_token_ids, result.action_token_ids,
                result.fast_token_ids, result.normalized_actions, error)) {
            error = "StarVLA FAST codec decode failed: " + error;
            return fail();
        }
        result.timings.policy_ms = elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!impl_->fast_policy->unnormalize(
                result.normalized_actions, result.unnorm_key_used,
                result.actions, error)) {
            error = "StarVLA FAST action unnormalization failed: " + error;
            return fail();
        }
        result.timings.unnormalize_ms =
            elapsed_ms(stage_start, Clock::now());
        const size_t expected_actions =
            static_cast<size_t>(config.horizon) * config.action_dim;
        if (result.actions.size() != expected_actions ||
            result.normalized_actions.size() != expected_actions ||
            !std::all_of(
                result.actions.begin(), result.actions.end(),
                [](float action) { return std::isfinite(action); })) {
            error =
                "StarVLA FAST returned an incompatible or non-finite action tensor";
            return fail();
        }
        result.chunk_size = config.horizon;
        result.action_dim = config.action_dim;
        result.timings.total_ms = elapsed_ms(total_start, Clock::now());
        return true;
    }

    if (impl_->variant == StarVLAVariant::qwen25_pi) {
        const PIPolicyConfig & config = impl_->pi_policy->config();
        if (!validate_observation(obs, config.image_count, config.image_names,
                                  true, "PI", error)) {
            return fail();
        }
        if (!obs.state.empty() &&
            obs.state.size() != static_cast<size_t>(config.state_dim)) {
            error = "StarVLA PI accepts either no state or exactly " +
                    std::to_string(config.state_dim) + " state values";
            return fail();
        }

        Clock::time_point stage_start = Clock::now();
        std::vector<std::vector<uint8_t>> pre_resized_images;
        std::vector<std::vector<uint8_t>> processed_images;
        std::vector<Qwen3VLImageView> qwen_images;
        if (!prepare_pi_qwen_images(obs, config, pre_resized_images,
                                    processed_images, qwen_images, error)) {
            return fail();
        }
        result.timings.image_preprocess_ms =
            elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!build_pi_v3_instruction(config.cot_template, obs.task,
                                     result.instruction, error)) {
            error = "failed to build the StarVLA PI prompt: " + error;
            return fail();
        }
        result.timings.prompt_ms = elapsed_ms(stage_start, Clock::now());

        std::vector<float> hidden_states;
        std::vector<uint8_t> attention_mask;
        stage_start = Clock::now();
        if (!impl_->qwen->extract_layer_hidden_states(
                qwen_images, result.instruction,
                config.qwen_hidden_tuple_indices, hidden_states,
                attention_mask, error)) {
            error = "StarVLA PI Qwen2.5-VL inference failed: " + error;
            return fail();
        }
        result.timings.qwen3vl_ms = elapsed_ms(stage_start, Clock::now());
        const size_t expected_hidden =
            static_cast<size_t>(config.block_count) * attention_mask.size() *
            static_cast<size_t>(config.qwen_hidden_dim);
        if (attention_mask.empty() || hidden_states.size() != expected_hidden) {
            error =
                "StarVLA PI Qwen2.5-VL returned an incompatible layerwise conditioning shape";
            return fail();
        }

        std::vector<float> noise;
        if (!make_noise(static_cast<size_t>(config.horizon) * config.action_dim,
                        noise)) {
            return fail();
        }

        stage_start = Clock::now();
        if (!impl_->pi_policy->evaluate(
                hidden_states.data(), hidden_states.size(),
                obs.state.empty() ? nullptr : obs.state.data(),
                obs.state.size(), noise.data(), noise.size(),
                result.normalized_actions, error)) {
            error = "StarVLA PI policy inference failed: " + error;
            return fail();
        }
        result.timings.policy_ms = elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!impl_->pi_policy->unnormalize(
                result.normalized_actions, result.unnorm_key_used,
                result.actions, error)) {
            error = "StarVLA PI action unnormalization failed: " + error;
            return fail();
        }
        result.timings.unnormalize_ms =
            elapsed_ms(stage_start, Clock::now());
        const size_t expected_actions =
            static_cast<size_t>(config.horizon) * config.action_dim;
        if (result.actions.size() != expected_actions ||
            !std::all_of(result.actions.begin(), result.actions.end(),
                         [](float action) { return std::isfinite(action); })) {
            error =
                "StarVLA PI returned an incompatible or non-finite action tensor";
            return fail();
        }
        result.chunk_size = config.horizon;
        result.action_dim = config.action_dim;
        result.timings.total_ms = elapsed_ms(total_start, Clock::now());
        return true;
    }

    if (impl_->variant == StarVLAVariant::qwen3_pi_v3) {
        const PIV3PolicyConfig & config = impl_->pi_v3_policy->config();
        if (!validate_observation(obs, config.image_count, config.image_names, false, "PI_v3",
                                  error)) {
            return fail();
        }

        Clock::time_point stage_start = Clock::now();
        std::vector<std::vector<uint8_t>> processed_images;
        std::vector<Qwen3VLImageView> qwen_images;
        if (!prepare_qwen_images(obs, config, "PI_v3", processed_images, qwen_images, error)) {
            return fail();
        }
        result.timings.image_preprocess_ms = elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!build_pi_v3_instruction(config.cot_template, obs.task, result.instruction, error)) {
            error = "failed to build the StarVLA PI_v3 prompt: " + error;
            return fail();
        }
        result.timings.prompt_ms = elapsed_ms(stage_start, Clock::now());

        std::vector<float> hidden_states;
        std::vector<uint8_t> attention_mask;
        stage_start = Clock::now();
        if (!impl_->qwen->extract_layer_hidden_states(
                qwen_images, result.instruction, config.qwen_hidden_tuple_indices,
                hidden_states, attention_mask, error)) {
            error = "StarVLA PI_v3 Qwen3-VL inference failed: " + error;
            return fail();
        }
        result.timings.qwen3vl_ms = elapsed_ms(stage_start, Clock::now());
        const size_t expected_hidden = static_cast<size_t>(config.qwen_layer_count) *
                                       attention_mask.size() * config.qwen_hidden_dim;
        if (attention_mask.empty() || hidden_states.size() != expected_hidden) {
            error = "StarVLA PI_v3 Qwen3-VL returned an incompatible layerwise conditioning shape";
            return fail();
        }

        std::vector<float> noise;
        if (!make_noise(static_cast<size_t>(config.horizon) * config.action_dim,
                        noise)) {
            return fail();
        }

        stage_start = Clock::now();
        if (!impl_->pi_v3_policy->evaluate(
                hidden_states.data(), hidden_states.size(), attention_mask.data(),
                attention_mask.size(), noise.data(), noise.size(), result.normalized_actions,
                error)) {
            error = "StarVLA PI_v3 policy inference failed: " + error;
            return fail();
        }
        result.timings.policy_ms = elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!impl_->pi_v3_policy->unnormalize(result.normalized_actions,
                                              result.unnorm_key_used, result.actions,
                                              error)) {
            error = "StarVLA PI_v3 action unnormalization failed: " + error;
            return fail();
        }
        result.timings.unnormalize_ms = elapsed_ms(stage_start, Clock::now());
        const size_t expected_actions =
            static_cast<size_t>(config.horizon) * config.action_dim;
        if (result.actions.size() != expected_actions ||
            !std::all_of(result.actions.begin(), result.actions.end(),
                         [](float action) { return std::isfinite(action); })) {
            error = "StarVLA PI_v3 returned an incompatible or non-finite action tensor";
            return fail();
        }
        result.chunk_size = config.horizon;
        result.action_dim = config.action_dim;
        result.timings.total_ms = elapsed_ms(total_start, Clock::now());
        return true;
    }

    if (impl_->variant == StarVLAVariant::qwen3_groot ||
        impl_->variant == StarVLAVariant::qwen25_groot) {
        const GR00TPolicyConfig & config = impl_->groot_policy->config();
        if (!validate_observation(obs, config.image_count, config.image_names, false, "GR00T",
                                  error)) {
            return fail();
        }

        Clock::time_point stage_start = Clock::now();
        std::vector<std::vector<uint8_t>> processed_images;
        std::vector<Qwen3VLImageView> qwen_images;
        if (!prepare_qwen_images(obs, config, "GR00T", processed_images, qwen_images, error)) {
            return fail();
        }
        result.timings.image_preprocess_ms = elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!build_groot_instruction(config.cot_template, obs.task, result.instruction, error)) {
            error = "failed to build the StarVLA GR00T prompt: " + error;
            return fail();
        }
        result.timings.prompt_ms = elapsed_ms(stage_start, Clock::now());

        std::vector<float> hidden_states;
        std::vector<uint8_t> attention_mask;
        stage_start = Clock::now();
        if (!impl_->qwen->extract_full_hidden_states(qwen_images, result.instruction,
                                                     hidden_states, attention_mask, error)) {
            error = "StarVLA GR00T Qwen3-VL inference failed: " + error;
            return fail();
        }
        result.timings.qwen3vl_ms = elapsed_ms(stage_start, Clock::now());
        if (hidden_states.empty() ||
            hidden_states.size() % static_cast<size_t>(config.qwen_hidden_dim) != 0 ||
            hidden_states.size() / static_cast<size_t>(config.qwen_hidden_dim) !=
                attention_mask.size()) {
            error = "StarVLA GR00T Qwen3-VL returned an incompatible conditioning shape";
            return fail();
        }

        std::vector<float> noise;
        if (!make_noise(static_cast<size_t>(config.horizon) * config.action_dim,
                        noise)) {
            return fail();
        }

        stage_start = Clock::now();
        if (!impl_->groot_policy->evaluate(
                hidden_states.data(), hidden_states.size(), attention_mask.data(),
                attention_mask.size(), noise.data(), noise.size(), result.normalized_actions,
                error)) {
            error = "StarVLA GR00T policy inference failed: " + error;
            return fail();
        }
        result.timings.policy_ms = elapsed_ms(stage_start, Clock::now());

        stage_start = Clock::now();
        if (!impl_->groot_policy->unnormalize(result.normalized_actions,
                                              result.unnorm_key_used, result.actions,
                                              error)) {
            error = "StarVLA GR00T action unnormalization failed: " + error;
            return fail();
        }
        result.timings.unnormalize_ms = elapsed_ms(stage_start, Clock::now());
        const size_t expected_actions =
            static_cast<size_t>(config.horizon) * config.action_dim;
        if (result.actions.size() != expected_actions ||
            !std::all_of(result.actions.begin(), result.actions.end(),
                         [](float action) { return std::isfinite(action); })) {
            error = "StarVLA GR00T returned an incompatible or non-finite action tensor";
            return fail();
        }
        result.chunk_size = config.horizon;
        result.action_dim = config.action_dim;
        result.timings.total_ms = elapsed_ms(total_start, Clock::now());
        return true;
    }

    if (initial_noise != nullptr) {
        error = "StarVLA OFT does not use diffusion noise";
        return fail();
    }
    const OFTPolicyConfig & config = impl_->oft_policy->config();
    if (!validate_observation(obs, config.image_count, config.image_names, true, "OFT", error)) {
        return fail();
    }

    Clock::time_point stage_start = Clock::now();
    std::vector<std::vector<uint8_t>> processed_images;
    std::vector<Qwen3VLImageView> qwen_images;
    if (!prepare_qwen_images(obs, config, "OFT", processed_images, qwen_images, error)) {
        return fail();
    }
    result.timings.image_preprocess_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    if (!build_oft_instruction(config.prompt, obs.task, obs.state, result.instruction, error)) {
        error = "failed to build the StarVLA OFT prompt: " + error;
        return fail();
    }
    result.timings.prompt_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    if (!impl_->qwen->extract_token_embeddings(
            qwen_images, result.instruction, config.action_token_id,
            static_cast<size_t>(config.horizon), result.action_queries, error)) {
        error = "StarVLA OFT Qwen3-VL inference failed: " + error;
        return fail();
    }
    result.timings.qwen3vl_ms = elapsed_ms(stage_start, Clock::now());
    const size_t expected_queries = static_cast<size_t>(config.horizon) * config.input_dim;
    if (result.action_queries.size() != expected_queries) {
        error = "StarVLA OFT Qwen3-VL returned an incompatible action-query shape";
        return fail();
    }

    stage_start = Clock::now();
    if (!impl_->oft_policy->evaluate(result.action_queries.data(), result.action_queries.size(),
                                     result.normalized_actions, error)) {
        error = "StarVLA OFT policy inference failed: " + error;
        return fail();
    }
    result.timings.policy_ms = elapsed_ms(stage_start, Clock::now());

    stage_start = Clock::now();
    if (!impl_->oft_policy->unnormalize(result.normalized_actions, result.unnorm_key_used,
                                        result.actions, error)) {
        error = "StarVLA OFT action unnormalization failed: " + error;
        return fail();
    }
    result.timings.unnormalize_ms = elapsed_ms(stage_start, Clock::now());
    const size_t expected_actions = static_cast<size_t>(config.horizon) * config.action_dim;
    if (result.actions.size() != expected_actions) {
        error = "StarVLA OFT returned an incompatible action tensor shape";
        return fail();
    }
    for (float action : result.actions) {
        if (!std::isfinite(action)) {
            error = "StarVLA OFT returned a non-finite unnormalized action";
            return fail();
        }
    }

    result.chunk_size = config.horizon;
    result.action_dim = config.action_dim;
    result.timings.total_ms = elapsed_ms(total_start, Clock::now());
    return true;
}

void StarVLAEngine::reset() {
    if (impl_ != nullptr && impl_->qwen != nullptr) {
        impl_->qwen->reset();
    }
}


} // namespace robotcpp::starvla
