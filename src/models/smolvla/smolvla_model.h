#pragma once

#include "models/model.h"

#include <cstdint>
#include <memory>
#include <string>

struct smolvla_context;

namespace robotcpp {

struct smolvla_model_options {
    std::string llm_path;
    std::string mmproj_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string task = "grab the block.";
    int n_batch = 512;
    int n_ctx = 2048;
    int noise_mode = 0;
    int64_t noise_seed = -1;
};

class SmolVLAModel final : public Model {
public:
    SmolVLAModel(const smolvla_model_options & options, const common_options & common);
    ~SmolVLAModel() override;

    SmolVLAModel(const SmolVLAModel &) = delete;
    SmolVLAModel & operator=(const SmolVLAModel &) = delete;

    const char * type() const override;
    bool predict(
        const observation & obs,
        model_result & out,
        std::string & error) override;
    void reset() override;

    bool is_ready() const;

private:
    smolvla_model_options options_;
    common_options common_;
    smolvla_context * ctx_ = nullptr;
};

bool make_smolvla_model(
    const smolvla_model_options & options,
    const common_options & common,
    std::unique_ptr<Model> & out,
    std::string & error);

} // namespace robotcpp
