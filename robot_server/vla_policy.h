#ifndef ROBOT_SERVER_VLA_POLICY_H
#define ROBOT_SERVER_VLA_POLICY_H

#include "smolvla_protocol.h"

#include <cstdint>
#include <memory>
#include <string>

namespace robot_server {

class vla_policy {
public:
    virtual ~vla_policy() = default;
    virtual bool predict(
        const smolvla::protocol::predict_request & req,
        smolvla::protocol::predict_response & resp,
        std::string & error) = 0;
    virtual void reset() {}
    virtual const char * name() const = 0;
};

struct smolvla_policy_options {
    std::string vlm_path;
    std::string mmproj_path;
    std::string state_proj_path;
    std::string action_expert_path;
    std::string task = "grab the block.";
    int threads = 0;
    int n_batch = 512;
    int n_ctx = 2048;
    int action_dim = 6;
    int chunk_size = 50;
    int num_steps = 10;
    int noise_mode = 0;
    int64_t noise_seed = -1;
    int verbosity = 1;
};

std::unique_ptr<vla_policy> make_smolvla_policy(const smolvla_policy_options & options, std::string & error);

} // namespace robot_server

#endif // ROBOT_SERVER_VLA_POLICY_H
