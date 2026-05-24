#include "core/error.h"

#include <utility>

namespace {
thread_local std::string g_last_error;
}

namespace vlacpp {

void set_error(std::string message) {
    g_last_error = std::move(message);
}

vlacpp_status fail(vlacpp_status status, std::string message) {
    set_error(std::move(message));
    return status;
}

} // namespace vlacpp

const char * vlacpp_last_error(void) {
    return g_last_error.c_str();
}
