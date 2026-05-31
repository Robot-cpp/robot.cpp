#include "smolvla_protocol.h"

#include <cstring>
#include <limits>

namespace smolvla {
namespace protocol {
namespace {

static void put_u16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t) (v & 0xffu));
    out.push_back((uint8_t) ((v >> 8) & 0xffu));
}

static void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back((uint8_t) ((v >> (8 * i)) & 0xffu));
    }
}

static void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back((uint8_t) ((v >> (8 * i)) & 0xffu));
    }
}

static void put_f32(std::vector<uint8_t> & out, float v) {
    uint32_t u = 0;
    static_assert(sizeof(u) == sizeof(v), "float32 expected");
    std::memcpy(&u, &v, sizeof(v));
    put_u32(out, u);
}

static void put_f64(std::vector<uint8_t> & out, double v) {
    uint64_t u = 0;
    static_assert(sizeof(u) == sizeof(v), "float64 expected");
    std::memcpy(&u, &v, sizeof(v));
    put_u64(out, u);
}

class reader {
public:
    reader(const uint8_t * data, size_t len) : data_(data), len_(len) {}

    bool u16(uint16_t & v) {
        if (!need(2)) return false;
        v = (uint16_t) data_[pos_] | ((uint16_t) data_[pos_ + 1] << 8);
        pos_ += 2;
        return true;
    }

    bool u32(uint32_t & v) {
        if (!need(4)) return false;
        v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= ((uint32_t) data_[pos_ + i]) << (8 * i);
        }
        pos_ += 4;
        return true;
    }

    bool u64(uint64_t & v) {
        if (!need(8)) return false;
        v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= ((uint64_t) data_[pos_ + i]) << (8 * i);
        }
        pos_ += 8;
        return true;
    }

    bool f32(float & v) {
        uint32_t u = 0;
        if (!u32(u)) return false;
        std::memcpy(&v, &u, sizeof(v));
        return true;
    }

    bool f64(double & v) {
        uint64_t u = 0;
        if (!u64(u)) return false;
        std::memcpy(&v, &u, sizeof(v));
        return true;
    }

    bool bytes(std::vector<uint8_t> & out, size_t n) {
        if (!need(n)) return false;
        out.assign(data_ + pos_, data_ + pos_ + n);
        pos_ += n;
        return true;
    }

    bool string(std::string & out, size_t n) {
        if (!need(n)) return false;
        out.assign((const char *) data_ + pos_, n);
        pos_ += n;
        return true;
    }

    size_t remaining() const {
        return len_ - pos_;
    }

private:
    bool need(size_t n) const {
        return n <= len_ - pos_;
    }

    const uint8_t * data_;
    size_t len_;
    size_t pos_ = 0;
};

static bool checked_u32_count(size_t n, const char * label, std::string & error) {
    if (n > std::numeric_limits<uint32_t>::max()) {
        error = std::string(label) + " too large";
        return false;
    }
    return true;
}

} // namespace

void encode_header(const header & h, std::vector<uint8_t> & out) {
    out.clear();
    out.reserve(k_header_size);
    put_u32(out, h.magic);
    put_u16(out, h.version);
    put_u16(out, h.header_size);
    put_u16(out, h.op);
    put_u16(out, h.flags);
    put_u32(out, h.request_id);
    put_u32(out, h.status);
    put_u64(out, h.payload_len);
    put_u32(out, h.reserved);
}

bool decode_header(const uint8_t * data, size_t len, header & h, std::string & error) {
    if (!data || len < k_header_size) {
        error = "short header";
        return false;
    }
    reader r(data, len);
    if (!r.u32(h.magic) || !r.u16(h.version) || !r.u16(h.header_size) ||
        !r.u16(h.op) || !r.u16(h.flags) || !r.u32(h.request_id) ||
        !r.u32(h.status) || !r.u64(h.payload_len) || !r.u32(h.reserved)) {
        error = "short header";
        return false;
    }
    return true;
}

bool validate_header(const header & h, uint64_t max_payload, std::string & error) {
    if (h.magic != k_magic) {
        error = "bad magic";
        return false;
    }
    if (h.version != k_version) {
        error = "bad version";
        return false;
    }
    if (h.header_size != k_header_size) {
        error = "bad header size";
        return false;
    }
    if (h.payload_len > max_payload) {
        error = "payload too large";
        return false;
    }
    return true;
}

bool encode_text_payload(const std::string & text, std::vector<uint8_t> & out) {
    out.assign(text.begin(), text.end());
    return true;
}

bool decode_text_payload(const std::vector<uint8_t> & payload, std::string & text, std::string & error) {
    (void) error;
    text.assign(payload.begin(), payload.end());
    return true;
}

//TODO: predict-related is actually design for one-picture smolvla for now
//      if we want to support pi0 or others, this may need to be refactored
bool encode_predict_request(const predict_request & req, std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    if (req.image_format != image_raw_rgb_u8) {
        error = "only RAW_RGB_U8 is supported";
        return false;
    }
    if (req.width == 0 || req.height == 0 || req.channels != 3 || req.image.empty()) {
        error = "invalid raw RGB image";
        return false;
    }
    if (req.state.size() > std::numeric_limits<uint32_t>::max() ||
        req.task.size() > std::numeric_limits<uint32_t>::max()) {
        error = "state or task too large";
        return false;
    }

    put_u32(out, req.image_format);
    put_u32(out, req.width);
    put_u32(out, req.height);
    put_u32(out, req.channels);
    put_u32(out, req.stride_bytes);
    put_u32(out, (uint32_t) req.state.size());
    put_u32(out, (uint32_t) req.task.size());
    put_u64(out, (uint64_t) req.image.size());

    for (float v : req.state) {
        put_f32(out, v);
    }
    out.insert(out.end(), req.task.begin(), req.task.end());
    out.insert(out.end(), req.image.begin(), req.image.end());
    return true;
}

bool decode_predict_request(const std::vector<uint8_t> & payload, predict_request & req, std::string & error) {
    reader r(payload.data(), payload.size());
    uint32_t state_dim = 0;
    uint32_t task_len = 0;
    uint64_t image_len = 0;

    if (!r.u32(req.image_format) || !r.u32(req.width) || !r.u32(req.height) ||
        !r.u32(req.channels) || !r.u32(req.stride_bytes) || !r.u32(state_dim) ||
        !r.u32(task_len) || !r.u64(image_len)) {
        error = "short predict request";
        return false;
    }
    if (req.image_format != image_raw_rgb_u8) {
        error = "unsupported image format";
        return false;
    }
    if (req.width == 0 || req.height == 0 || req.channels != 3) {
        error = "invalid raw RGB dimensions";
        return false;
    }
    if (image_len > r.remaining()) {
        error = "image length exceeds payload";
        return false;
    }

    req.state.assign(state_dim, 0.0f);
    for (uint32_t i = 0; i < state_dim; ++i) {
        if (!r.f32(req.state[i])) {
            error = "short state array";
            return false;
        }
    }
    if (!r.string(req.task, task_len)) {
        error = "short task string";
        return false;
    }
    if (!r.bytes(req.image, (size_t) image_len)) {
        error = "short image bytes";
        return false;
    }
    if (r.remaining() != 0) {
        error = "trailing bytes in predict request";
        return false;
    }
    return true;
}

bool encode_predict_response(const predict_response & resp, std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    const size_t expected = (size_t) resp.chunk_size * (size_t) resp.action_dim;
    if (expected != resp.actions.size()) {
        error = "action shape does not match action count";
        return false;
    }
    if (!checked_u32_count(resp.actions.size(), "actions", error)) {
        return false;
    }

    put_u32(out, resp.chunk_size);
    put_u32(out, resp.action_dim);
    put_u32(out, (uint32_t) resp.actions.size());
    put_f64(out, resp.timing.vision_ms);
    put_f64(out, resp.timing.state_proj_ms);
    put_f64(out, resp.timing.vlm_ms);
    put_f64(out, resp.timing.kv_extract_ms);
    put_f64(out, resp.timing.phase2_ms);
    put_f64(out, resp.timing.model_total_ms);
    put_f64(out, resp.timing.server_recv_ms);
    put_f64(out, resp.timing.server_queue_ms);
    put_f64(out, resp.timing.server_predict_ms);
    for (float v : resp.actions) {
        put_f32(out, v);
    }
    return true;
}

bool decode_predict_response(const std::vector<uint8_t> & payload, predict_response & resp, std::string & error) {
    reader r(payload.data(), payload.size());
    uint32_t action_count = 0;
    if (!r.u32(resp.chunk_size) || !r.u32(resp.action_dim) || !r.u32(action_count) ||
        !r.f64(resp.timing.vision_ms) || !r.f64(resp.timing.state_proj_ms) ||
        !r.f64(resp.timing.vlm_ms) || !r.f64(resp.timing.kv_extract_ms) ||
        !r.f64(resp.timing.phase2_ms) || !r.f64(resp.timing.model_total_ms) ||
        !r.f64(resp.timing.server_recv_ms) || !r.f64(resp.timing.server_queue_ms) ||
        !r.f64(resp.timing.server_predict_ms)) {
        error = "short predict response";
        return false;
    }
    if ((size_t) resp.chunk_size * (size_t) resp.action_dim != (size_t) action_count) {
        error = "action shape does not match action count";
        return false;
    }
    resp.actions.assign(action_count, 0.0f);
    for (uint32_t i = 0; i < action_count; ++i) {
        if (!r.f32(resp.actions[i])) {
            error = "short action array";
            return false;
        }
    }
    if (r.remaining() != 0) {
        error = "trailing bytes in predict response";
        return false;
    }
    return true;
}

} // namespace protocol
} // namespace smolvla
