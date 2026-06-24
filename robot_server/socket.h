#ifndef ROBOT_SERVER_SOCKET_H
#define ROBOT_SERVER_SOCKET_H

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#endif

namespace robot_server {
namespace sockets {

#ifdef _WIN32
using socket_handle                           = SOCKET;
static constexpr socket_handle invalid_socket = INVALID_SOCKET;
#else
using socket_handle                           = int;
static constexpr socket_handle invalid_socket = -1;
#endif

bool startup(std::string & error);
void cleanup();

socket_handle tcp_listen(const char * host, uint16_t port, int backlog, std::string & error);
socket_handle tcp_accept(socket_handle server, std::string & peer, std::string & error);
socket_handle tcp_connect(const char * host, uint16_t port, std::string & error);

bool send_all(socket_handle fd, const void * data, size_t len, std::string & error);
bool recv_exact(socket_handle fd, void * data, size_t len, std::string & error);
void close(socket_handle fd);

} // namespace sockets
} // namespace robot_server

#endif // ROBOT_SERVER_SOCKET_H
