#include "core/gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
void write_scalar(std::ofstream & file, T value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

void write_string(std::ofstream & file, const std::string & value) {
    write_scalar<uint64_t>(file, static_cast<uint64_t>(value.size()));
    file.write(value.data(), static_cast<std::streamsize>(value.size()));
}

uint64_t align_to(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

bool write_tensor_fixture(const char * path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write("GGUF", 4);
    write_scalar<uint32_t>(file, 3);
    write_scalar<uint64_t>(file, 1);
    write_scalar<uint64_t>(file, 0);

    write_string(file, "sample");
    write_scalar<uint32_t>(file, 1);
    write_scalar<uint64_t>(file, 3);
    write_scalar<uint32_t>(file, 0);
    write_scalar<uint64_t>(file, 0);

    const uint64_t data_start = align_to(static_cast<uint64_t>(file.tellp()), 32);
    while (static_cast<uint64_t>(file.tellp()) < data_start) {
        write_scalar<uint8_t>(file, 0);
    }
    const float values[3] = {1.0f, -2.0f, 3.5f};
    file.write(reinterpret_cast<const char *>(values), sizeof(values));
    return static_cast<bool>(file);
}

} // namespace

int main() {
    const char * path = "vlacpp-test-tensor.gguf";
    if (!write_tensor_fixture(path)) {
        std::cerr << "failed to write GGUF tensor fixture\n";
        return 1;
    }

    std::vector<float> values;
    if (!vlacpp::read_gguf_tensor_f32(path, "sample", values)) {
        std::cerr << "failed to read GGUF tensor fixture\n";
        std::remove(path);
        return 1;
    }
    if (values.size() != 3 || values[0] != 1.0f || values[1] != -2.0f || values[2] != 3.5f) {
        std::cerr << "unexpected GGUF tensor values\n";
        std::remove(path);
        return 1;
    }
    if (vlacpp::read_gguf_tensor_f32(path, "missing", values)) {
        std::cerr << "missing GGUF tensor should not be found\n";
        std::remove(path);
        return 1;
    }

    std::remove(path);
    return 0;
}
