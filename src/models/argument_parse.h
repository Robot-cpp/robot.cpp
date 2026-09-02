#pragma once

#include <charconv>
#include <cstring>
#include <system_error>
#include <type_traits>

namespace robotcpp {

template <typename Integer> bool parse_integer_argument(const char * value, Integer & output) {
    static_assert(std::is_integral<Integer>::value && !std::is_same<Integer, bool>::value,
                  "Integer must be a non-bool integral type");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    Integer parsed                      = 0;
    const char * end                    = value + std::strlen(value);
    const std::from_chars_result result = std::from_chars(value, end, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    output = parsed;
    return true;
}

} // namespace robotcpp
