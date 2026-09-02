#pragma once

#include "models/model.h"

#include <memory>
#include <string>

namespace robotcpp::starvla {
class StarVLAEngine;
}

namespace robotcpp {

class StarVLAModel final : public Model {
  public:
    ~StarVLAModel() override;

    StarVLAModel(const StarVLAModel &)             = delete;
    StarVLAModel & operator=(const StarVLAModel &) = delete;

    const char * type() const override;
    bool predict(const observation & obs, model_result & out, std::string & error) override;
    void reset() override;

  private:
    explicit StarVLAModel(std::unique_ptr<starvla::StarVLAEngine> engine);

    friend bool make_starvla_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error);

    std::unique_ptr<starvla::StarVLAEngine> engine_;
};

bool make_starvla_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error);

} // namespace robotcpp
