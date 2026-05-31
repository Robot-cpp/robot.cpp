#include "session.h"

#include "smolvla_protocol.h"

// Session owns the request/response conversation for one connected client.
//
// Response behavior:
// - health:   text payload, "ok policy=<name>"
// - reset:    text payload, "ok"
// - shutdown: text payload, "ok", then asks the outer server loop to exit
// - predict:  binary predict_response payload with action chunk and timings
//
// Errors are returned as text payloads with a non-OK protocol status. The policy
// engine itself is shared by the server, so predict/reset calls are serialized by
// predict_mutex; socket read/write and protocol encode/decode stay in this layer.

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace proto = smolvla::protocol;
using clock_type = std::chrono::steady_clock;

namespace robot_server {
namespace {

static double elapsed_ms(const clock_type::time_point & a, const clock_type::time_point & b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static bool send_message(
    sockets::socket_handle fd,
    uint16_t op,
    uint32_t request_id,
    uint32_t status,
    const std::vector<uint8_t> & payload,
    std::string & error) {
    proto::header h;
    h.op = op;
    h.request_id = request_id;
    h.status = status;
    h.payload_len = payload.size();

    std::vector<uint8_t> hdr;
    proto::encode_header(h, hdr);
    return sockets::send_all(fd, hdr.data(), hdr.size(), error) &&
           (payload.empty() || sockets::send_all(fd, payload.data(), payload.size(), error));
}

static bool send_text(
    sockets::socket_handle fd,
    uint16_t op,
    uint32_t request_id,
    uint32_t status,
    const std::string & text) {
    std::vector<uint8_t> payload;
    proto::encode_text_payload(text, payload);
    std::string error;
    if (!send_message(fd, op, request_id, status, payload, error)) {
        std::fprintf(stderr, "[SmolVLA server] failed to send response: %s\n", error.c_str());
        return false;
    }
    return true;
}

static bool read_message(
    sockets::socket_handle fd,
    proto::header & h,
    std::vector<uint8_t> & payload,
    double & recv_ms,
    std::string & error) {
    const auto t0 = clock_type::now();
    std::vector<uint8_t> hdr(proto::k_header_size);
    if (!sockets::recv_exact(fd, hdr.data(), hdr.size(), error)) {
        return false;
    }
    if (!proto::decode_header(hdr.data(), hdr.size(), h, error)) {
        return false;
    }
    if (!proto::validate_header(h, proto::k_default_max_payload, error)) {
        return false;
    }
    payload.assign((size_t) h.payload_len, 0);
    if (!payload.empty() && !sockets::recv_exact(fd, payload.data(), payload.size(), error)) {
        return false;
    }
    recv_ms = elapsed_ms(t0, clock_type::now());
    return true;
}

} // namespace

bool handle_client(
    sockets::socket_handle fd,
    vla_policy & policy,
    std::mutex & predict_mutex,
    bool & shutdown_requested) {
    while (!shutdown_requested) {
        proto::header h;
        std::vector<uint8_t> payload;
        double recv_ms = 0.0;
        std::string error;
        if (!read_message(fd, h, payload, recv_ms, error)) {
            if (error != "peer closed connection") {
                std::fprintf(stderr, "[SmolVLA server] read failed: %s\n", error.c_str());
            }
            return false;
        }

        if (h.op == proto::op_health) {
            send_text(fd, h.op, h.request_id, proto::status_ok, std::string("ok policy=") + policy.name());
        } else if (h.op == proto::op_reset) {
            std::lock_guard<std::mutex> lock(predict_mutex);
            policy.reset();
            send_text(fd, h.op, h.request_id, proto::status_ok, "ok");
        } else if (h.op == proto::op_shutdown) {
            send_text(fd, h.op, h.request_id, proto::status_ok, "ok");
            shutdown_requested = true;
            return true;
        } else if (h.op == proto::op_predict) {
            proto::predict_request req;
            if (!proto::decode_predict_request(payload, req, error)) {
                send_text(fd, h.op, h.request_id, proto::status_bad_request, error);
                continue;
            }

            proto::predict_response resp;
            resp.timing.server_recv_ms = recv_ms;

            const auto t_queue0 = clock_type::now();
            std::unique_lock<std::mutex> lock(predict_mutex);
            resp.timing.server_queue_ms = elapsed_ms(t_queue0, clock_type::now());

            const auto t_predict0 = clock_type::now();
            const bool ok = policy.predict(req, resp, error);
            resp.timing.server_predict_ms = elapsed_ms(t_predict0, clock_type::now());
            lock.unlock();

            if (!ok) {
                send_text(fd, h.op, h.request_id, proto::status_internal_error, error);
                continue;
            }

            std::vector<uint8_t> out;
            if (!proto::encode_predict_response(resp, out, error)) {
                send_text(fd, h.op, h.request_id, proto::status_internal_error, error);
                continue;
            }
            if (!send_message(fd, h.op, h.request_id, proto::status_ok, out, error)) {
                std::fprintf(stderr, "[SmolVLA server] send failed: %s\n", error.c_str());
                return false;
            }
        } else {
            send_text(fd, h.op, h.request_id, proto::status_bad_request, "unknown op");
        }
    }
    return true;
}

} // namespace robot_server
