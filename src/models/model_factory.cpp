#include "models/model.h"

#include "models/smolvla/smolvla_model.h"

#include <any>

namespace robotcpp {

bool make_model(
    const model_args & args,
    std::unique_ptr<Model> & out,
    std::string & error) {
    out.reset();
    if (args.type == model_type::smolvla) {
        return make_smolvla_model(args, out, error);
    }

    error = "unsupported model type";
    return false;
}

} // namespace robotcpp
