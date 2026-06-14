#include "models/pi0/pi0_engine.h"

#include "models/pi0/action.h"
#include "models/pi0/load.h"
#include "models/pi0/pi0_context.h"
#include "models/pi0/preprocess.h"
#include "models/pi0/vlm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

pi0_result empty_result() {
    pi0_result result{};
    return result;
}

pi0_stage_timings empty_stage_timings() {
    pi0_stage_timings timings{};
    return timings;
}

uint32_t seed_from_params(const pi0_params & params) {
    if (params.noise_seed >= 0) {
        return static_cast<uint32_t>(params.noise_seed);
    }
    std::random_device device;
    return device();
}

void pi0_log_error(const std::string & message) {
    std::fprintf(stderr, "[Pi0] Error: %s\n", message.c_str());
}

bool pi0_env_flag_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0 &&
        std::strcmp(value, "off") != 0 &&
        std::strcmp(value, "OFF") != 0 &&
        std::strcmp(value, "no") != 0 &&
        std::strcmp(value, "NO") != 0;
}

} // namespace

struct pi0_context {
    robotcpp::pi0::Pi0Context model;
    robotcpp::pi0::Pi0Sampler sampler;
    robotcpp::pi0::Pi0RuntimeConfig runtime;
    robotcpp::pi0::Pi0KvCache cache;
    std::vector<float> action_buffer;
    pi0_stage_timings last_timings = {};

    pi0_context(
        robotcpp::pi0::Pi0ModelConfig config,
        std::string tokenizer_path,
        robotcpp::pi0::Pi0Components components,
        uint32_t seed)
        : model(std::move(config), tokenizer_path, std::move(components)),
          sampler(model) {
        runtime.seed = seed;
        runtime.flow_steps = 10;
        runtime.rng.seed(seed);
    }
};

pi0_params pi0_default_params(void) {
    pi0_params params;
    std::memset(&params, 0, sizeof(params));
    params.n_threads = 0;
    params.noise_seed = -1;
    params.verbosity = 1;
    return params;
}

pi0_context * pi0_init(pi0_params params) {
    robotcpp::pi0::Pi0ComponentPaths paths;
    paths.vit = params.vit_path ? params.vit_path : "";
    paths.mmproj = params.mmproj_path ? params.mmproj_path : "";
    paths.llm = params.llm_path ? params.llm_path : "";
    paths.tokenizer = params.tokenizer_path ? params.tokenizer_path : "";
    paths.state = params.state_path ? params.state_path : "";
    paths.action_decoder = params.action_decoder_path ? params.action_decoder_path : "";

    robotcpp::pi0::Pi0BackendConfig backend;
    backend.backend = pi0_env_flag_enabled("PI0_USE_ACCEL_BACKEND", true) ?
        robotcpp::pi0::PI0_BACKEND_ACCEL :
        robotcpp::pi0::PI0_BACKEND_CPU;
    backend.n_threads = params.n_threads;

    robotcpp::pi0::Pi0ModelConfig config;
    robotcpp::pi0::Pi0Components components;
    if (!robotcpp::pi0::load_pi0_components(paths, backend, config, components, params.verbosity)) {
        return nullptr;
    }

    try {
        std::unique_ptr<pi0_context> ctx(new (std::nothrow) pi0_context(
            std::move(config),
            paths.tokenizer,
            std::move(components),
            seed_from_params(params)));
        if (!ctx) {
            pi0_log_error("failed to allocate pi0 context");
            return nullptr;
        }
        return ctx.release();
    } catch (const std::exception & error) {
        pi0_log_error(error.what());
        return nullptr;
    }
}

pi0_result pi0_predict_raw_rgb(
    pi0_context * ctx,
    const pi0_image_view * images,
    size_t image_count,
    const float * state,
    size_t state_dim,
    const char * task) {
    if (ctx == nullptr || (image_count > 0 && images == nullptr)) {
        pi0_log_error("pi0 context and image views are required");
        return empty_result();
    }

    ctx->last_timings = empty_stage_timings();
    ctx->runtime.last_timings = {};

    std::vector<robotcpp::pi0::Pi0RawImageView> raw_images;
    raw_images.reserve(image_count);
    for (size_t i = 0; i < image_count; ++i) {
        robotcpp::pi0::Pi0RawImageView image;
        image.name = images[i].name;
        image.data = images[i].data;
        image.width = images[i].width;
        image.height = images[i].height;
        image.channels = images[i].channels;
        image.stride_bytes = images[i].stride_bytes;
        raw_images.push_back(image);
    }

    robotcpp::pi0::Pi0RawObservation raw;
    raw.images = raw_images.empty() ? nullptr : raw_images.data();
    raw.image_count = raw_images.size();
    raw.state = state;
    raw.state_count = state_dim;
    raw.prompt = task;

    const auto total_start = Clock::now();
    robotcpp::pi0::Pi0Observation observation;
    const bool ok = robotcpp::pi0::validate_and_preprocess_pi0(ctx->model.config, raw, observation);
    const auto preprocess_done = Clock::now();
    ctx->runtime.last_timings.preprocess_ms = elapsed_ms(total_start, preprocess_done);
    if (!ok) {
        ctx->last_timings.preprocess_ms = ctx->runtime.last_timings.preprocess_ms;
        ctx->last_timings.total_ms = elapsed_ms(total_start, Clock::now());
        return empty_result();
    }

    try {
        const auto prefix_start = Clock::now();
        robotcpp::pi0::pi0_prefill_prefix(ctx->model, ctx->cache, observation);
        const auto prefix_done = Clock::now();
        ctx->runtime.last_timings.prefix_ms = elapsed_ms(prefix_start, prefix_done);

        if (!robotcpp::pi0::pi0_has_action_head(ctx->model)) {
            pi0_log_error("pi0 inference requires mapped OpenPI action decoder tensors");
            return empty_result();
        }

        std::vector<float> state_context;
        const auto state_start = Clock::now();
        robotcpp::pi0::pi0_state_context(ctx->model, observation.state, state_context);
        const auto state_done = Clock::now();
        ctx->runtime.last_timings.state_ms = elapsed_ms(state_start, state_done);

        const auto denoise_start = Clock::now();
        ctx->sampler.sample_actions(ctx->runtime, state_context, ctx->cache, observation.noise, ctx->action_buffer);
        const auto denoise_done = Clock::now();
        ctx->runtime.last_timings.denoise_ms = elapsed_ms(denoise_start, denoise_done);
    } catch (const std::exception & error) {
        pi0_log_error(error.what());
        return empty_result();
    }

    const auto output_done = Clock::now();
    const double output_ms = elapsed_ms(preprocess_done, output_done) -
        ctx->runtime.last_timings.prefix_ms -
        ctx->runtime.last_timings.state_ms -
        ctx->runtime.last_timings.denoise_ms;
    ctx->runtime.last_timings.output_ms = std::max(0.0, output_ms);
    ctx->runtime.last_timings.total_ms = elapsed_ms(total_start, output_done);
    ctx->last_timings.preprocess_ms = ctx->runtime.last_timings.preprocess_ms;
    ctx->last_timings.prefix_ms = ctx->runtime.last_timings.prefix_ms;
    ctx->last_timings.state_ms = ctx->runtime.last_timings.state_ms;
    ctx->last_timings.denoise_ms = ctx->runtime.last_timings.denoise_ms;
    ctx->last_timings.output_ms = ctx->runtime.last_timings.output_ms;
    ctx->last_timings.total_ms = ctx->runtime.last_timings.total_ms;

    pi0_result result;
    result.actions = ctx->action_buffer.empty() ? nullptr : ctx->action_buffer.data();
    result.chunk_size = ctx->model.config.common.action_horizon;
    result.action_dim = ctx->model.config.common.action_dim;
    return result;
}

pi0_stage_timings pi0_get_last_stage_timings(const pi0_context * ctx) {
    return ctx == nullptr ? empty_stage_timings() : ctx->last_timings;
}

void pi0_reset(pi0_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    ctx->cache.reset();
    ctx->model.prefix_kv.reset();
}

void pi0_free(pi0_context * ctx) {
    delete ctx;
}
