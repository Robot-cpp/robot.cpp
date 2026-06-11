#pragma once

#include "models/model.h"

#include <cstdint>
#include <memory>
#include <string>

struct smolvla_context;

namespace robotcpp {

class SmolVLAModel final : public Model {
public:
    SmolVLAModel(const model_args & args);
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
    model_args args_;
    smolvla_context * ctx_ = nullptr;
};

bool make_smolvla_model(
    const model_args & args,
    std::unique_ptr<Model> & out,
    std::string & error);

} // namespace robotcpp
