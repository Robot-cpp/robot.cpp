#pragma once

#include "core/types.h"
#include "vlacpp.h"

namespace vlacpp {

vlacpp_status validate_and_preprocess(
    const ModelConfig & config,
    const vlacpp_observation & raw,
    ObservationData & out);

} // namespace vlacpp
