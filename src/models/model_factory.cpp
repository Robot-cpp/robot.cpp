#include "models/model.h"

#include "models/pi0/pi0_model.h"
#include "models/smolvla/smolvla_model.h"

namespace robotcpp {

bool make_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error) {
    out.reset();
    if (args.type == model_type::smolvla) {
        return make_smolvla_model(args, out, error);
    }
    if (args.type == model_type::pi0) {
        return make_pi0_model(args, out, error);
    }

    error = "unsupported model type";
    return false;
}

} // namespace robotcpp
