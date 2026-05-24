#pragma once

#include "vlacpp.h"

#include <string>

namespace vlacpp {

void set_error(std::string message);
vlacpp_status fail(vlacpp_status status, std::string message);

} // namespace vlacpp
