#include "vlacpp.h"

#include "core/error.h"
#include "core/preprocess.h"
#include "core/types.h"
#include "models/model.h"

#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct vlacpp_model {
    std::unique_ptr<vlacpp::RuntimeModel> impl;
    vlacpp::BackendConfig backend;
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
    return params;
}

vlacpp_context_params vlacpp_default_context_params(void) {
    vlacpp_context_params params;
    params.seed = 0;
    params.flow_steps = 10;
    return params;
}

vlacpp_status vlacpp_load_model(
    const char * path,
    const vlacpp_model_params * params,
    vlacpp_model ** out_model) {
    if (path == nullptr || out_model == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "path and out_model are required");
    }
    *out_model = nullptr;

    vlacpp_model_params effective = params ? *params : vlacpp_default_model_params();
    vlacpp::BackendConfig backend;
    backend.backend = effective.backend;
    backend.n_threads = effective.n_threads;

    std::unique_ptr<vlacpp::RuntimeModel> impl;
    vlacpp_status status = vlacpp::load_model_from_path(path, backend, impl);
    if (status != VLACPP_STATUS_OK) {
        return status;
    }

    std::unique_ptr<vlacpp_model> model(new (std::nothrow) vlacpp_model);
    if (!model) {
        return vlacpp::fail(VLACPP_STATUS_RUNTIME_ERROR, "failed to allocate model");
    }
    model->backend = backend;
    model->impl = std::move(impl);
    *out_model = model.release();
    return VLACPP_STATUS_OK;
}

void vlacpp_free_model(vlacpp_model * model) {
    delete model;
}

const char * vlacpp_model_capability(vlacpp_model * model) {
    if (model == nullptr || model->impl == nullptr) {
        return "invalid";
    }
    return model->impl->capability();
}

vlacpp_status vlacpp_model_openpi_graph_info(
    vlacpp_model * model,
    vlacpp_openpi_graph_info * out_info) {
    if (model == nullptr || model->impl == nullptr || out_info == nullptr) {
        return vlacpp::fail(VLACPP_STATUS_INVALID_ARGUMENT, "model and out_info are required");
    }
    const vlacpp::ModelConfig & config = model->impl->config();
    out_info->action_width = config.openpi_action_width;
    out_info->vision_width = config.openpi_vision_width;
    out_info->vision_patch_height = config.openpi_vision_patch_height;
    out_info->vision_patch_width = config.openpi_vision_patch_width;
    out_info->vision_layers = config.openpi_vision_layers;
    out_info->language_width = config.openpi_language_width;
    out_info->language_q_out = config.openpi_language_q_out;
    out_info->language_kv_out = config.openpi_language_kv_out;
    out_info->language_mlp_width = config.openpi_language_mlp_width;
    out_info->language_layers = config.openpi_language_layers;
    out_info->action_expert_width = config.openpi_action_expert_width;
    out_info->action_expert_q_out = config.openpi_action_expert_q_out;
    out_info->action_expert_kv_out = config.openpi_action_expert_kv_out;
    out_info->action_expert_mlp_width = config.openpi_action_expert_mlp_width;
    out_info->action_expert_layers = config.openpi_action_expert_layers;
    out_info->full_weights_present = config.openpi_full_weights_present ? 1 : 0;
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
    return context->model->impl->reset_cache(context->cache);
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

    vlacpp::ObservationData processed;
    const vlacpp::ModelConfig & config = context->model->impl->config();
    vlacpp_status status = vlacpp::validate_and_preprocess(config, *observation, processed);
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
    out_actions->data = data;
    out_actions->horizon = config.action_horizon;
    out_actions->action_dim = config.action_dim;
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
