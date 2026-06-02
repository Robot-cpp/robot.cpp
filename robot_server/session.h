#ifndef ROBOT_SERVER_SESSION_H
#define ROBOT_SERVER_SESSION_H

#include "socket.h"
#include "vla_policy.h"

#include <mutex>

namespace robot_server {

bool handle_client(
    sockets::socket_handle fd,
    vla_policy & policy,
    std::mutex & predict_mutex,
    bool & shutdown_requested);

} // namespace robot_server

#endif // ROBOT_SERVER_SESSION_H
