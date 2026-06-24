#include "protocol.h"

#include <cstring>
#include <limits>

namespace robot_server {
namespace protocol {
namespace {

static void put_u16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t)(v & 0xffu));
    out.push_back((uint8_t)((v >> 8) & 0xffu));
}

static void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back((uint8_t)((v >> (8 * i)) & 0xffu));
    }
}

static void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back((uint8_t)((v >> (8 * i)) & 0xffu));
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
        if (!need(2))
            return false;
        v = (uint16_t)data_[pos_] | ((uint16_t)data_[pos_ + 1] << 8);
        pos_ += 2;
        return true;
    }

    bool u32(uint32_t & v) {
        if (!need(4))
            return false;
        v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= ((uint32_t)data_[pos_ + i]) << (8 * i);
        }
        pos_ += 4;
        return true;
    }

    bool u64(uint64_t & v) {
        if (!need(8))
            return false;
        v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= ((uint64_t)data_[pos_ + i]) << (8 * i);
        }
        pos_ += 8;
        return true;
    }

    bool f32(float & v) {
        uint32_t u = 0;
        if (!u32(u))
            return false;
        std::memcpy(&v, &u, sizeof(v));
        return true;
    }

    bool f64(double & v) {
        uint64_t u = 0;
        if (!u64(u))
            return false;
        std::memcpy(&v, &u, sizeof(v));
        return true;
    }

    bool bytes(std::vector<uint8_t> & out, size_t n) {
        if (!need(n))
            return false;
        out.assign(data_ + pos_, data_ + pos_ + n);
        pos_ += n;
        return true;
    }

    bool string(std::string & out, size_t n) {
        if (!need(n))
            return false;
        out.assign((const char *)data_ + pos_, n);
        pos_ += n;
        return true;
    }

    size_t remaining() const { return len_ - pos_; }

  private:
    bool need(size_t n) const { return n <= len_ - pos_; }

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

static bool validate_image_payload(const image_payload & image, const char * label, std::string & error) {
    if (image.image_format != image_raw_rgb_u8) {
        error = std::string(label) + " unsupported image format";
        return false;
    }
    if (image.width == 0 || image.height == 0 || image.channels != 3 || image.data.empty()) {
        error = std::string(label) + " invalid raw RGB image";
        return false;
    }
    const uint32_t stride = image.stride_bytes == 0 ? image.width * image.channels : image.stride_bytes;
    if (stride < image.width * image.channels) {
        error = std::string(label) + " stride_bytes is smaller than width * channels";
        return false;
    }
    const uint64_t min_size = (uint64_t)stride * (uint64_t)image.height;
    if (image.data.size() < min_size) {
        error = std::string(label) + " data is smaller than stride_bytes * height";
        return false;
    }
    if (image.name.size() > std::numeric_limits<uint32_t>::max()) {
        error = std::string(label) + " name too large";
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
    if (!r.u32(h.magic) || !r.u16(h.version) || !r.u16(h.header_size) || !r.u16(h.op) || !r.u16(h.flags) ||
        !r.u32(h.request_id) || !r.u32(h.status) || !r.u64(h.payload_len) || !r.u32(h.reserved)) {
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
    (void)error;
    text.assign(payload.begin(), payload.end());
    return true;
}

bool encode_predict_request(const predict_request & req, std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    if (req.images.empty()) {
        error = "predict request requires at least one image";
        return false;
    }
    if (!checked_u32_count(req.images.size(), "images", error) ||
        !checked_u32_count(req.state.size(), "state", error) || !checked_u32_count(req.task.size(), "task", error)) {
        return false;
    }
    for (size_t i = 0; i < req.images.size(); ++i) {
        const std::string label = "image[" + std::to_string(i) + "]";
        if (!validate_image_payload(req.images[i], label.c_str(), error)) {
            return false;
        }
    }

    put_u32(out, (uint32_t)req.images.size());
    put_u32(out, (uint32_t)req.state.size());
    put_u32(out, (uint32_t)req.task.size());
    for (const image_payload & image : req.images) {
        put_u32(out, image.image_format);
        put_u32(out, (uint32_t)image.name.size());
        put_u32(out, image.width);
        put_u32(out, image.height);
        put_u32(out, image.channels);
        put_u32(out, image.stride_bytes);
        put_u64(out, (uint64_t)image.data.size());
    }

    for (float v : req.state) {
        put_f32(out, v);
    }
    out.insert(out.end(), req.task.begin(), req.task.end());
    for (const image_payload & image : req.images) {
        out.insert(out.end(), image.name.begin(), image.name.end());
        out.insert(out.end(), image.data.begin(), image.data.end());
    }
    return true;
}

bool decode_predict_request(const std::vector<uint8_t> & payload, predict_request & req, std::string & error) {
    req = predict_request{};
    reader r(payload.data(), payload.size());
    uint32_t image_count = 0;
    uint32_t state_dim = 0;
    uint32_t task_len = 0;

    if (!r.u32(image_count) || !r.u32(state_dim) || !r.u32(task_len)) {
        error = "short predict request";
        return false;
    }
    if (image_count == 0) {
        error = "predict request requires at least one image";
        return false;
    }

    req.images.assign(image_count, image_payload{});
    std::vector<uint32_t> name_lens(image_count, 0);
    std::vector<uint64_t> image_lens(image_count, 0);
    for (uint32_t i = 0; i < image_count; ++i) {
        image_payload & image = req.images[i];
        if (!r.u32(image.image_format) || !r.u32(name_lens[i]) || !r.u32(image.width) || !r.u32(image.height) ||
            !r.u32(image.channels) || !r.u32(image.stride_bytes) || !r.u64(image_lens[i])) {
            error = "short image metadata";
            return false;
        }
        if (image.image_format != image_raw_rgb_u8) {
            error = "unsupported image format";
            return false;
        }
        if (image.width == 0 || image.height == 0 || image.channels != 3) {
            error = "invalid raw RGB dimensions";
            return false;
        }
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
    for (uint32_t i = 0; i < image_count; ++i) {
        image_payload & image = req.images[i];
        if (!r.string(image.name, name_lens[i])) {
            error = "short image name";
            return false;
        }
        if (image_lens[i] > r.remaining()) {
            error = "image length exceeds payload";
            return false;
        }
        if (!r.bytes(image.data, (size_t)image_lens[i])) {
            error = "short image bytes";
            return false;
        }
        const std::string label = "image[" + std::to_string(i) + "]";
        if (!validate_image_payload(image, label.c_str(), error)) {
            return false;
        }
    }
    if (r.remaining() != 0) {
        error = "trailing bytes in predict request";
        return false;
    }
    return true;
}

bool encode_predict_response(const predict_response & resp, std::vector<uint8_t> & out, std::string & error) {
    out.clear();
    const size_t expected = (size_t)resp.chunk_size * (size_t)resp.action_dim;
    if (expected != resp.actions.size()) {
        error = "action shape does not match action count";
        return false;
    }
    if (!checked_u32_count(resp.actions.size(), "actions", error) ||
        !checked_u32_count(resp.metrics.size(), "metrics", error)) {
        return false;
    }
    for (size_t i = 0; i < resp.metrics.size(); ++i) {
        const metric & m = resp.metrics[i];
        if (m.name.empty()) {
            error = "metric name is empty";
            return false;
        }
        if (!checked_u32_count(m.name.size(), "metric name", error)) {
            return false;
        }
    }

    put_u32(out, resp.chunk_size);
    put_u32(out, resp.action_dim);
    put_u32(out, (uint32_t)resp.actions.size());
    put_u32(out, (uint32_t)resp.metrics.size());
    for (const metric & m : resp.metrics) {
        put_u32(out, (uint32_t)m.name.size());
        put_f64(out, m.value);
        out.insert(out.end(), m.name.begin(), m.name.end());
    }
    for (float v : resp.actions) {
        put_f32(out, v);
    }
    return true;
}

bool decode_predict_response(const std::vector<uint8_t> & payload, predict_response & resp, std::string & error) {
    resp = predict_response{};
    reader r(payload.data(), payload.size());
    uint32_t action_count = 0;
    uint32_t metric_count = 0;
    if (!r.u32(resp.chunk_size) || !r.u32(resp.action_dim) || !r.u32(action_count) || !r.u32(metric_count)) {
        error = "short predict response";
        return false;
    }
    if ((size_t)resp.chunk_size * (size_t)resp.action_dim != (size_t)action_count) {
        error = "action shape does not match action count";
        return false;
    }
    resp.metrics.assign(metric_count, metric{});
    for (uint32_t i = 0; i < metric_count; ++i) {
        uint32_t name_len = 0;
        if (!r.u32(name_len) || !r.f64(resp.metrics[i].value)) {
            error = "short metric";
            return false;
        }
        if (!r.string(resp.metrics[i].name, name_len)) {
            error = "short metric name";
            return false;
        }
        if (resp.metrics[i].name.empty()) {
            error = "metric name is empty";
            return false;
        }
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
} // namespace robot_server
