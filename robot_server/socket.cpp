#include "socket.h"

#include <cerrno>
#include <cstring>

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

static std::string socket_error_string(const char * prefix) {
#ifdef _WIN32
    return std::string(prefix) + ": WSA error " + std::to_string(WSAGetLastError());
#else
    return std::string(prefix) + ": " + std::strerror(errno);
#endif
}

} // namespace

namespace robot_server {
namespace sockets {

bool startup(std::string & error) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        error = socket_error_string("WSAStartup failed");
        return false;
    }
#else
    (void)error;
#endif
    return true;
}

void cleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

socket_handle tcp_listen(const char * host, uint16_t port, int backlog, std::string & error) {
    socket_handle fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == invalid_socket) {
        error = socket_error_string("socket failed");
        return invalid_socket;
    }

    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        error = std::string("invalid IPv4 host: ") + host;
        close(fd);
        return invalid_socket;
    }

    if (::bind(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        error = socket_error_string("bind failed");
        close(fd);
        return invalid_socket;
    }
    if (::listen(fd, backlog) != 0) {
        error = socket_error_string("listen failed");
        close(fd);
        return invalid_socket;
    }
    return fd;
}

socket_handle tcp_accept(socket_handle server, std::string & peer, std::string & error) {
    sockaddr_in addr;
    socklen_t len    = sizeof(addr);
    socket_handle fd = ::accept(server, (sockaddr *)&addr, &len);
    if (fd == invalid_socket) {
        error = socket_error_string("accept failed");
        return invalid_socket;
    }

    char ip[INET_ADDRSTRLEN] = {};
    if (inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip))) {
        peer = std::string(ip) + ":" + std::to_string(ntohs(addr.sin_port));
    } else {
        peer = "<unknown>";
    }
    return fd;
}

socket_handle tcp_connect(const char * host, uint16_t port, std::string & error) {
    socket_handle fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == invalid_socket) {
        error = socket_error_string("socket failed");
        return invalid_socket;
    }

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        error = std::string("invalid IPv4 host: ") + host;
        close(fd);
        return invalid_socket;
    }
    if (::connect(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        error = socket_error_string("connect failed");
        close(fd);
        return invalid_socket;
    }
    return fd;
}

bool send_all(socket_handle fd, const void * data, size_t len, std::string & error) {
    const uint8_t * p = (const uint8_t *)data;
    size_t sent       = 0;
    while (sent < len) {
#ifdef _WIN32
        int n = ::send(fd, (const char *)p + sent, (int)(len - sent), 0);
#else
        ssize_t n = ::send(fd, p + sent, len - sent, 0);
#endif
        if (n <= 0) {
            error = socket_error_string("send failed");
            return false;
        }
        sent += (size_t)n;
    }
    return true;
}

bool recv_exact(socket_handle fd, void * data, size_t len, std::string & error) {
    uint8_t * p     = (uint8_t *)data;
    size_t received = 0;
    while (received < len) {
#ifdef _WIN32
        int n = ::recv(fd, (char *)p + received, (int)(len - received), 0);
#else
        ssize_t n = ::recv(fd, p + received, len - received, 0);
#endif
        if (n == 0) {
            error = "peer closed connection";
            return false;
        }
        if (n < 0) {
            error = socket_error_string("recv failed");
            return false;
        }
        received += (size_t)n;
    }
    return true;
}

void close(socket_handle fd) {
    if (fd == invalid_socket) {
        return;
    }
#ifdef _WIN32
    closesocket(fd);
#else
    ::close(fd);
#endif
}

} // namespace sockets
} // namespace robot_server
