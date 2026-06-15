#pragma once

#include "models/model.h"

#include <memory>
#include <string>

struct pi0_context;

namespace robotcpp {

class Pi0Model final : public Model {
public:
    explicit Pi0Model(const model_args & args);
    ~Pi0Model() override;

    Pi0Model(const Pi0Model &) = delete;
    Pi0Model & operator=(const Pi0Model &) = delete;

    const char * type() const override;
    bool predict(const observation & obs, model_result & out, std::string & error) override;
    void reset() override;

    bool is_ready() const;

private:
    model_args args_;
    pi0_context * ctx_ = nullptr;
};

bool make_pi0_model(
    const model_args & args,
    std::unique_ptr<Model> & out,
    std::string & error);

} // namespace robotcpp
