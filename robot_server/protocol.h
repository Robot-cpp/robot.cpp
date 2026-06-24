#ifndef ROBOT_SERVER_PROTOCOL_H
#define ROBOT_SERVER_PROTOCOL_H

#include <cstdint>
#include <string>
#include <vector>

namespace robot_server {
namespace protocol {

static constexpr uint32_t k_magic               = 0x414c5653u; // "SVLA" in little-endian bytes.
static constexpr uint16_t k_version             = 3;
static constexpr uint16_t k_header_size         = 32;
static constexpr uint64_t k_default_max_payload = 256ull * 1024ull * 1024ull;

enum op : uint16_t {
    op_health   = 1,
    op_reset    = 2,
    op_predict  = 3,
    op_shutdown = 4,
};

enum status : uint32_t {
    status_ok              = 0,
    status_bad_request     = 1,
    status_bad_magic       = 2,
    status_bad_version     = 3,
    status_payload_too_big = 4,
    status_internal_error  = 5,
};

enum image_format : uint32_t {
    image_raw_rgb_u8 = 1,
};

struct image_payload {
    uint32_t image_format = image_raw_rgb_u8;
    uint32_t width        = 0;
    uint32_t height       = 0;
    uint32_t channels     = 0;
    uint32_t stride_bytes = 0;
    std::string name;
    std::vector<uint8_t> data;
};

struct header {
    uint32_t magic       = k_magic;
    uint16_t version     = k_version;
    uint16_t header_size = k_header_size;
    uint16_t op          = 0;
    uint16_t flags       = 0;
    uint32_t request_id  = 0;
    uint32_t status      = status_ok;
    uint64_t payload_len = 0;
    uint32_t reserved    = 0;
};

struct metric {
    std::string name;
    double value = 0.0;
};

struct predict_request {
    std::vector<image_payload> images;
    std::vector<float> state;
    std::string task;
};

struct predict_response {
    uint32_t chunk_size = 0;
    uint32_t action_dim = 0;
    std::vector<float> actions;
    std::vector<metric> metrics;
};

void encode_header(const header & h, std::vector<uint8_t> & out);
bool decode_header(const uint8_t * data, size_t len, header & h, std::string & error);
bool validate_header(const header & h, uint64_t max_payload, std::string & error);

bool encode_text_payload(const std::string & text, std::vector<uint8_t> & out);
bool decode_text_payload(const std::vector<uint8_t> & payload, std::string & text, std::string & error);

bool encode_predict_request(const predict_request & req, std::vector<uint8_t> & out, std::string & error);
bool decode_predict_request(const std::vector<uint8_t> & payload, predict_request & req, std::string & error);

bool encode_predict_response(const predict_response & resp, std::vector<uint8_t> & out, std::string & error);
bool decode_predict_response(const std::vector<uint8_t> & payload, predict_response & resp, std::string & error);

} // namespace protocol
} // namespace robot_server

#endif // ROBOT_SERVER_PROTOCOL_H
