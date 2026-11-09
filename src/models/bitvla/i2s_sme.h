#pragma once
#include <cstdint>
extern "C" uint64_t cpu_sme_words();
extern "C" void cpu_sme_i8(const int8_t *, const int8_t *, int32_t *, int, int, int, int, int, int);
