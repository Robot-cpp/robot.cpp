#include "core/gguf.h"
#include "models/pi0/load.h"

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

void write_metadata_i32(std::ofstream & file, const std::string & key, int32_t value) {
    write_string(file, key);
    write_scalar<uint32_t>(file, 5);
    write_scalar<int32_t>(file, value);
}

uint64_t align_to(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void write_tensor_info(std::ofstream & file, const std::string & name, uint32_t type, uint64_t offset) {
    write_string(file, name);
    write_scalar<uint32_t>(file, 1);
    write_scalar<uint64_t>(file, 3);
    write_scalar<uint32_t>(file, type);
    write_scalar<uint64_t>(file, offset);
}

bool write_tensor_fixture(const char * path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write("GGUF", 4);
    write_scalar<uint32_t>(file, 3);
    write_scalar<uint64_t>(file, 2);
    write_scalar<uint64_t>(file, 2);

    write_metadata_i32(file, "vlacpp.action_dim", 3);
    write_metadata_i32(file, "vlacpp.action_horizon", 1);

    write_tensor_info(file, "sample", 1, 0);
    write_tensor_info(file, "skip", 30, 32);

    const uint64_t data_start = align_to(static_cast<uint64_t>(file.tellp()), 32);
    while (static_cast<uint64_t>(file.tellp()) < data_start) {
        write_scalar<uint8_t>(file, 0);
    }
    const uint16_t values[3] = {0x3c00, 0xc000, 0x4300};
    file.write(reinterpret_cast<const char *>(values), sizeof(values));
    while (static_cast<uint64_t>(file.tellp()) < data_start + 32) {
        write_scalar<uint8_t>(file, 0);
    }
    const uint16_t skipped[3] = {0x4080, 0x40a0, 0x40c0};
    file.write(reinterpret_cast<const char *>(skipped), sizeof(skipped));
    return static_cast<bool>(file);
}

bool only_sample(const vlacpp::ModelConfig &, const std::string & name) {
    return name == "sample";
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

    vlacpp::ModelConfig config;
    vlacpp::TensorMap tensors;
    if (vlacpp::load_gguf_model_file(path, config, tensors) != VLACPP_STATUS_OK || tensors.size() != 2) {
        std::cerr << "default GGUF load should read all tensors\n";
        std::remove(path);
        return 1;
    }
    if (tensors["sample"].data_type != "f16" || tensors["skip"].data_type != "bf16") {
        std::cerr << "GGUF loader should preserve tensor storage dtype names\n";
        std::remove(path);
        return 1;
    }
    if (vlacpp::load_gguf_model_file(path, config, tensors, only_sample) != VLACPP_STATUS_OK ||
        tensors.size() != 1 ||
        tensors.find("sample") == tensors.end()) {
        std::cerr << "filtered GGUF load should read only matching tensors\n";
        std::remove(path);
        return 1;
    }
    vlacpp::ensure_pi0_config(config);
    if (!vlacpp::should_load_pi0_tensor(config, "pi0.action_decoder.action_in_proj.weight") ||
        !vlacpp::should_load_pi0_tensor(config, "pi0.merger.weight") ||
        !vlacpp::should_load_pi0_tensor(config, "pi0.llm.layers.0.self_attn.q_proj.weight") ||
        !vlacpp::should_load_pi0_tensor(config, "pi0.vit.encoder.layers.0.self_attn.q_proj.weight")) {
        std::cerr << "pi0 GGUF load filter should include action, merger, ViT, and LLM tensors\n";
        std::remove(path);
        return 1;
    }
    if (!vlacpp::should_load_pi0_tensor(config, "pi0.llm.layers.0.input_layernorm.scale")) {
        std::cerr << "pi0 GGUF load filter should include latest split GGUF RMSNorm scales\n";
        std::remove(path);
        return 1;
    }

    std::remove(path);
    return 0;
}
