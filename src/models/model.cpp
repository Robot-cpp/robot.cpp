#include "models/model.h"

#include "models/pi0/load.h"

namespace vlacpp {

vlacpp_status load_model_from_artifacts(
    const ModelArtifacts & artifacts,
    const BackendConfig & backend,
    std::unique_ptr<RuntimeModel> & out) {
    return load_pi0_model_from_artifacts(artifacts, backend, out);
}

} // namespace vlacpp
