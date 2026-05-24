#pragma once

#include "core/types.h"
#include "vlacpp.h"

#include <memory>
#include <string>

namespace vlacpp {

vlacpp_status load_model_from_path(
    const std::string & path,
    const BackendConfig & backend,
    std::unique_ptr<Model> & out);

} // namespace vlacpp
