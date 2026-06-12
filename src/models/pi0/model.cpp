#include "models/pi0/load.h"

#include "core/error.h"
#include "models/pi0/action.h"
#include "models/pi0/pi0_context.h"
#include "models/pi0/vlm.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vlacpp {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void materialize_pi0_component_weights(Pi0ComponentRuntime & component) {
    std::vector<Tensor *> weights;
    weights.reserve(component.tensors.size());
    for (auto & item : component.tensors) {
        weights.push_back(&item.second);
    }
    component.runner->materialize_weights(weights, component.default_2d_weight_type);
}

void materialize_pi0_device_weights(Pi0Context & ctx) {
    materialize_pi0_component_weights(ctx.components.vit);
    materialize_pi0_component_weights(ctx.components.mmproj);
    materialize_pi0_component_weights(ctx.components.llm);
    materialize_pi0_component_weights(ctx.components.state);
    materialize_pi0_component_weights(ctx.components.action_decoder);
}

class Pi0Model final : public RuntimeModel {
public:
    Pi0Model(ModelConfig config, BackendConfig backend, std::string tokenizer_path, Pi0ComponentTensors tensors)
        : ctx_(std::move(config), backend, tokenizer_path, std::move(tensors)),
          sampler_(ctx_) {
        materialize_pi0_device_weights(ctx_);
    }

    const ModelConfig & config() const override {
        return ctx_.config;
    }

    vlacpp_status reset_cache(KvCache & cache) override {
        cache.reset();
        ctx_.prefix_kv.reset();
        return VLACPP_STATUS_OK;
    }

    vlacpp_status infer(
        KvCache & cache,
        RuntimeConfig & runtime,
        const ObservationData & observation,
        std::vector<float> & out_actions) override {
        const auto prefix_start = Clock::now();
        pi0_prefill_prefix(ctx_, cache, observation);
        const auto prefix_done = Clock::now();
        runtime.last_timings.prefix_ms = elapsed_ms(prefix_start, prefix_done);
        if (!pi0_has_action_head(ctx_)) {
            return fail(VLACPP_STATUS_UNSUPPORTED, "pi0 inference requires mapped OpenPI action decoder tensors");
        }
        std::vector<float> state_context;
        const auto state_start = Clock::now();
        pi0_state_context(ctx_, observation.state, state_context);
        const auto state_done = Clock::now();
        runtime.last_timings.state_ms = elapsed_ms(state_start, state_done);
        const auto denoise_start = Clock::now();
        sampler_.sample_actions(runtime, state_context, cache, observation.noise, out_actions);
        const auto denoise_done = Clock::now();
        runtime.last_timings.denoise_ms = elapsed_ms(denoise_start, denoise_done);
        return VLACPP_STATUS_OK;
    }

private:
    Pi0Context ctx_;
    Pi0Sampler sampler_;
};

} // namespace

std::unique_ptr<RuntimeModel> make_pi0_model(
    ModelConfig config,
    BackendConfig backend,
    std::string tokenizer_path,
    Pi0ComponentTensors tensors) {
    return std::unique_ptr<RuntimeModel>(
        new Pi0Model(std::move(config), backend, std::move(tokenizer_path), std::move(tensors)));
}

} // namespace vlacpp
