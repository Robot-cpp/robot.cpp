#include "models/model.h"

#include "models/smolvla/smolvla_model.h"

#include <any>

namespace robotcpp {

bool make_model(
    const model_options & options,
    std::unique_ptr<Model> & out,
    std::string & error) {
    out.reset();
    if (options.type == model_type::smolvla) {
        const smolvla_model_options * smolvla_options = std::any_cast<smolvla_model_options>(&options.config);
        if (smolvla_options == nullptr) {
            error = "model_options.config must contain robotcpp::smolvla_model_options";
            return false;
        }
        return make_smolvla_model(*smolvla_options, options.common, out, error);
    }

    error = "unsupported model type";
    return false;
}

} // namespace robotcpp
