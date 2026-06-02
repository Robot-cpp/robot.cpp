#include "vla_policy.h"

#include "smolvla_engine.h"

namespace proto = smolvla::protocol;

namespace robot_server {
namespace {

class smolvla_policy : public vla_policy {
public:
    explicit smolvla_policy(const smolvla_policy_options & options) {
        smolvla_params params = smolvla_default_params();
        params.vlm_path = options.vlm_path.c_str();
        params.mmproj_path = options.mmproj_path.c_str();
        params.state_proj_path = options.state_proj_path.empty() ? nullptr : options.state_proj_path.c_str();
        params.action_expert_path = options.action_expert_path.empty() ? nullptr : options.action_expert_path.c_str();
        params.task = options.task.c_str();
        params.n_threads = options.threads;
        params.n_batch = options.n_batch;
        params.n_ctx = options.n_ctx;
        params.action_dim = options.action_dim;
        params.chunk_size = options.chunk_size;
        params.num_steps = options.num_steps;
        params.noise_mode = options.noise_mode;
        params.noise_seed = options.noise_seed;
        params.verbosity = options.verbosity;

        ctx_ = smolvla_init(params);
    }

    ~smolvla_policy() override {
        if (ctx_) {
            smolvla_free(ctx_);
        }
    }

    bool is_ready() const { return ctx_ != nullptr; }

    bool predict(const proto::predict_request & req, proto::predict_response & resp, std::string & error) override {
        if (!ctx_) {
            error = "SmolVLA policy is not initialized";
            return false;
        }

        smolvla_result result = smolvla_predict_raw_rgb(
            ctx_,
            req.image.data(),
            (int) req.width,
            (int) req.height,
            (int) req.channels,
            (int) req.stride_bytes,
            req.state.empty() ? nullptr : req.state.data(),
            (int) req.state.size());
        if (!result.actions) {
            error = "smolvla_predict_raw_rgb failed";
            return false;
        }

        resp.chunk_size = (uint32_t) result.chunk_size;
        resp.action_dim = (uint32_t) result.action_dim;
        resp.actions.assign(result.actions, result.actions + (size_t) result.chunk_size * (size_t) result.action_dim);

        const smolvla_stage_timings t = smolvla_get_last_stage_timings(ctx_);
        resp.timing.vision_ms = t.vision_ms;
        resp.timing.state_proj_ms = t.state_proj_ms;
        resp.timing.vlm_ms = t.vlm_ms;
        resp.timing.kv_extract_ms = t.kv_extract_ms;
        resp.timing.phase2_ms = t.phase2_ms;
        resp.timing.model_total_ms = t.total_ms;
        return true;
    }

    const char * name() const override { return "smolvla"; }

private:
    smolvla_context * ctx_ = nullptr;
};

} // namespace

//TODO: support more models!
std::unique_ptr<vla_policy> make_smolvla_policy(const smolvla_policy_options & options, std::string & error) {
    std::unique_ptr<smolvla_policy> policy(new smolvla_policy(options));
    if (!policy->is_ready()) {
        error = "failed to initialize SmolVLA policy";
        return nullptr;
    }
    return policy;
}

} // namespace robot_server
