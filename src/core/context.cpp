#include "vlacpp.h"

#include "core/error.h"
#include "core/preprocess.h"
#include "core/types.h"
#include "models/model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

vlacpp_status make_backend_config(const vlacpp_model_params * params, vlacpp::BackendConfig & backend) {
    vlacpp_model_params effective = params ? *params : vlacpp_default_model_params();
    backend.backend = effective.backend;
    backend.n_threads = effective.n_threads;
    backend.component_dtype_overrides.clear();
    if (effective.dtype_override_count > 0 && effective.dtype_overrides == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "dtype override items are required when count is non-zero");
    }
    for (size_t i = 0; i < effective.dtype_override_count; ++i) {
        const vlacpp_component_dtype_override & item = effective.dtype_overrides[i];
        if (item.role == nullptr || item.dtype == nullptr) {
            return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "dtype override role and dtype are required");
        }
        if (!backend.component_dtype_overrides.emplace(item.role, item.dtype).second) {
            return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, std::string("duplicate dtype override role: ") + item.role);
        }
    }
    return VLACPP_STATUS_OK;
}

} // namespace

struct vlacpp_model {
    std::unique_ptr<vlacpp::RuntimeModel> impl;
};

struct vlacpp_context {
    vlacpp_model * model = nullptr;
    vlacpp::RuntimeConfig runtime;
    vlacpp::KvCache cache;
};

vlacpp_model_params vlacpp_default_model_params(void) {
    vlacpp_model_params params;
    params.backend = VLACPP_BACKEND_CPU;
    params.n_threads = 0;
    params.dtype_overrides = nullptr;
    params.dtype_override_count = 0;
    return params;
}

vlacpp_context_params vlacpp_default_context_params(void) {
    vlacpp_context_params params;
    params.seed = 0;
    params.flow_steps = 10;
    return params;
}

vlacpp_status vlacpp_load_model(
    const vlacpp_model_artifacts * artifacts,
    const vlacpp_model_params * params,
    vlacpp_model ** out_model) {
    if (artifacts == nullptr || out_model == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "artifacts and out_model are required");
    }
    *out_model = nullptr;
    if (artifacts->items == nullptr || artifacts->count == 0) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "model artifacts are required");
    }

    vlacpp::BackendConfig backend;
    vlacpp_status status = make_backend_config(params, backend);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }
    vlacpp::ModelArtifacts model_artifacts;
    model_artifacts.reserve(artifacts->count);
    for (size_t i = 0; i < artifacts->count; ++i) {
        const vlacpp_model_artifact & item = artifacts->items[i];
        if (item.role == nullptr || item.path == nullptr) {
            return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "model artifact role and path are required");
        }
        model_artifacts.push_back({item.role, item.path});
    }

    std::unique_ptr<vlacpp::RuntimeModel> impl;
    try {
        status = vlacpp::load_model_from_artifacts(model_artifacts, backend, impl);
        if (status != VLACPP_STATUS_OK) {
            return status;
        }
    } catch (const std::exception & error) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, error.what());
    }

    std::unique_ptr<vlacpp_model> model(new (std::nothrow) vlacpp_model);
    if (!model) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, "failed to allocate model");
    }
    model->impl = std::move(impl);
    *out_model = model.release();
    return VLACPP_STATUS_OK;
}

void vlacpp_free_model(vlacpp_model * model) {
    delete model;
}

vlacpp_status vlacpp_get_model_info(
    vlacpp_model * model,
    vlacpp_model_info * out_info) {
    if (model == nullptr || model->impl == nullptr || out_info == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "model and out_info are required");
    }
    const vlacpp::ModelConfig & config = model->impl->config();
    out_info->model_type = config.common.model_type.c_str();
    out_info->image_width = config.common.image_width;
    out_info->image_height = config.common.image_height;
    out_info->state_dim = config.common.state_dim;
    out_info->action_dim = config.common.action_dim;
    out_info->action_horizon = config.common.action_horizon;
    out_info->max_token_len = config.common.max_token_len;
    return VLACPP_STATUS_OK;
}

vlacpp_status vlacpp_create_context(
    vlacpp_model * model,
    const vlacpp_context_params * params,
    vlacpp_context ** out_context) {
    if (model == nullptr || out_context == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "model and out_context are required");
    }
    *out_context = nullptr;

    vlacpp_context_params effective = params ? *params : vlacpp_default_context_params();
    std::unique_ptr<vlacpp_context> context(new (std::nothrow) vlacpp_context);
    if (!context) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, "failed to allocate context");
    }
    context->model = model;
    context->runtime.seed = effective.seed;
    context->runtime.flow_steps = effective.flow_steps > 0 ? effective.flow_steps : 10;
    context->runtime.rng.seed(effective.seed);

    *out_context = context.release();
    return VLACPP_STATUS_OK;
}

void vlacpp_free_context(vlacpp_context * context) {
    delete context;
}

vlacpp_status vlacpp_reset_cache(vlacpp_context * context) {
    if (context == nullptr || context->model == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "context is required");
    }
    try {
        return context->model->impl->reset_cache(context->cache);
    } catch (const std::exception & error) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, error.what());
    }
}

vlacpp_status vlacpp_infer_actions(
    vlacpp_context * context,
    const vlacpp_observation * observation,
    vlacpp_action_chunk * out_actions) {
    if (context == nullptr || context->model == nullptr || observation == nullptr || out_actions == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "context, observation, and out_actions are required");
    }
    out_actions->data = nullptr;
    out_actions->horizon = 0;
    out_actions->action_dim = 0;
    context->runtime.last_timings = {};

    const auto total_start = Clock::now();
    vlacpp::ObservationData processed;
    const vlacpp::ModelConfig & config = context->model->impl->config();
    vlacpp_status status = vlacpp::validate_and_preprocess(config, *observation, processed);
    const auto preprocess_done = Clock::now();
    context->runtime.last_timings.preprocess_ms = elapsed_ms(total_start, preprocess_done);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }

    std::vector<float> actions;
    try {
        status = context->model->impl->infer(context->cache, context->runtime, processed, actions);
        if (status != VLACPP_STATUS_OK) {
            return status;
        }
    } catch (const std::exception & error) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, error.what());
    }

    const size_t bytes = actions.size() * sizeof(float);
    float * data = static_cast<float *>(std::malloc(bytes));
    if (data == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, "failed to allocate action chunk");
    }
    std::copy(actions.begin(), actions.end(), data);
    const auto output_done = Clock::now();
    const double output_ms = elapsed_ms(preprocess_done, output_done) -
        context->runtime.last_timings.prefix_ms -
        context->runtime.last_timings.state_ms -
        context->runtime.last_timings.denoise_ms;
    context->runtime.last_timings.output_ms = std::max(0.0, output_ms);
    context->runtime.last_timings.total_ms = elapsed_ms(total_start, output_done);
    out_actions->data = data;
    out_actions->horizon = config.common.action_horizon;
    out_actions->action_dim = config.common.action_dim;
    return VLACPP_STATUS_OK;
}

vlacpp_status vlacpp_context_last_timings(
    vlacpp_context * context,
    vlacpp_infer_timings * out_timings) {
    if (context == nullptr || out_timings == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "context and out_timings are required");
    }
    *out_timings = context->runtime.last_timings;
    return VLACPP_STATUS_OK;
}

void vlacpp_free_action_chunk(vlacpp_action_chunk * actions) {
    if (actions == nullptr) {
        return;
    }
    std::free(actions->data);
    actions->data = nullptr;
    actions->horizon = 0;
    actions->action_dim = 0;
}
