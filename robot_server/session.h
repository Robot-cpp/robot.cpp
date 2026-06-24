#ifndef ROBOT_SERVER_SESSION_H
#define ROBOT_SERVER_SESSION_H

#include "model_adapter.h"
#include "socket.h"

#include <mutex>

namespace robot_server {

bool handle_client(sockets::socket_handle fd, model_adapter & policy, std::mutex & predict_mutex,
                   bool & shutdown_requested);

} // namespace robot_server

#endif // ROBOT_SERVER_SESSION_H
