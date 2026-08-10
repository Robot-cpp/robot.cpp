#include "models/model.h"

#include "models/pi0/pi0_model.h"
#include "models/smolvla/smolvla_model.h"
#ifdef ROBOT_CPP_BUILD_STARVLA
#include "models/starvla/starvla_model.h"
#endif

namespace robotcpp {

bool make_model(const model_args & args, std::unique_ptr<Model> & out, std::string & error) {
    out.reset();
    if (args.type == model_type::smolvla) {
        return make_smolvla_model(args, out, error);
    }
    if (args.type == model_type::pi0) {
        return make_pi0_model(args, out, error);
    }
    if (args.type == model_type::starvla) {
#ifdef ROBOT_CPP_BUILD_STARVLA
        return make_starvla_model(args, out, error);
#else
        error = "StarVLA support was not built; configure with -DROBOT_CPP_BUILD_STARVLA=ON";
        return false;
#endif
    }

    error = std::string("unsupported model type: ") + model_type_name(args.type);
    return false;
}

} // namespace robotcpp
