#pragma once

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string_view>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace rinha {

#define RINHA_LIKELY(x) __builtin_expect(!!(x), 1)
#define RINHA_UNLIKELY(x) __builtin_expect(!!(x), 0)

struct Response {
    const char* data;
    std::size_t len;
};

template <std::size_t N>
consteval Response make_response(const char (&s)[N]) {
    return {s, N - 1};
}

inline constexpr Response kFraudResponses[] = {
    make_response("HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}"),
    make_response("HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}"),
    make_response("HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}"),
    make_response("HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}"),
    make_response("HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}"),
    make_response("HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"),
};

inline constexpr auto kReadyResponse = make_response(
    "HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}");

inline constexpr auto k404Response = make_response(
    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");

static constexpr std::size_t kBufSize = 2048;

struct Connection {
    int fd = -1;
    uint32_t have = 0;
    bool in_use = false;
    char buf[kBufSize];
};

template <typename Derived>
class Server {
public:
    enum class Mode : uint8_t { kTcp, kUnix };

    explicit Server(uint16_t port) : mode_(Mode::kTcp), port_(port) {}

    explicit Server(const char* path) : mode_(Mode::kUnix), port_(0) {
        auto len = std::strlen(path);
        if (RINHA_UNLIKELY(len >= sizeof(socket_path_))) {
            std::fprintf(stderr, "socket path too long: %s\n", path);
            std::_Exit(1);
        }
        std::memcpy(socket_path_, path, len + 1);
    }

    [[noreturn]] void run() {
        ::signal(SIGPIPE, SIG_IGN);
        listen_fd_ = create_listener();
        ep_fd_     = ::epoll_create1(EPOLL_CLOEXEC);
        if (RINHA_UNLIKELY(ep_fd_ < 0)) fatal("epoll_create1");

        epoll_event ev{};
        ev.events  = EPOLLIN | EPOLLET;
        ev.data.fd = listen_fd_;
        ::epoll_ctl(ep_fd_, EPOLL_CTL_ADD, listen_fd_, &ev);

        for (;;) {
            int n = ::epoll_wait(ep_fd_, events_.data(), kMaxEvents, -1);
            if (RINHA_UNLIKELY(n < 0)) {
                if (errno == EINTR) continue;
                fatal("epoll_wait");
            }
            for (int i = 0; i < n; ++i) {
                const int      fd      = events_[static_cast<unsigned>(i)].data.fd;
                const uint32_t revents = events_[static_cast<unsigned>(i)].events;

                if (fd == listen_fd_) { accept_all(); continue; }

                auto uidx = static_cast<unsigned>(fd);
                if (RINHA_UNLIKELY(uidx >= conns_len_ || !conns_[uidx].in_use))
                    continue;

                bool alive = !(revents & (EPOLLERR | EPOLLHUP));
                if (alive && (revents & EPOLLIN))    alive = drain(conns_[uidx]);
                if (alive && (revents & EPOLLRDHUP)) alive = false;
                if (!alive) close_conn(fd);
            }
        }
    }

private:
    static constexpr int      kMaxEvents = 512;
    static constexpr unsigned kMaxConns  = 4096;

    Derived& self() { return static_cast<Derived&>(*this); }

    [[noreturn, gnu::cold, gnu::noinline]]
    static void fatal(const char* msg) { std::perror(msg); std::_Exit(1); }

    int create_listener() const {
        return mode_ == Mode::kUnix ? create_listener_unix()
                                    : create_listener_tcp();
    }

    int create_listener_tcp() const {
        int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (RINHA_UNLIKELY(fd < 0)) fatal("socket");

        int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof one);

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(port_);

        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) fatal("bind");
        if (::listen(fd, 4096) != 0) fatal("listen");
        return fd;
    }

    int create_listener_unix() const {
        ::unlink(socket_path_);

        int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (RINHA_UNLIKELY(fd < 0)) fatal("socket(unix)");

        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, socket_path_, sizeof(addr.sun_path) - 1);

        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) fatal("bind(unix)");
        ::chmod(socket_path_, 0777);
        if (::listen(fd, 4096) != 0) fatal("listen(unix)");
        return fd;
    }

    static void tune_tcp(int fd) {
        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef TCP_QUICKACK
        ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one);
#endif
    }

    void accept_all() {
        for (;;) {
            int cfd = ::accept4(listen_fd_, nullptr, nullptr,
                                SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (mode_ == Mode::kTcp) tune_tcp(cfd);

            auto uidx = static_cast<unsigned>(cfd);
            if (RINHA_UNLIKELY(uidx >= kMaxConns)) { ::close(cfd); continue; }
            if (uidx >= conns_len_) conns_len_ = uidx + 1;

            Connection& c = conns_[uidx];
            c.fd     = cfd;
            c.have   = 0;
            c.in_use = true;

            epoll_event ev{};
            ev.events  = EPOLLIN | EPOLLRDHUP | EPOLLET;
            ev.data.fd = cfd;
            if (::epoll_ctl(ep_fd_, EPOLL_CTL_ADD, cfd, &ev) != 0) {
                c.in_use = false;
                ::close(cfd);
            }
        }
    }

    void close_conn(int fd) {
        ::epoll_ctl(ep_fd_, EPOLL_CTL_DEL, fd, nullptr);
        ::close(fd);
        auto uidx = static_cast<unsigned>(fd);
        if (uidx < conns_len_) {
            conns_[uidx].in_use = false;
            conns_[uidx].fd     = -1;
        }
    }

    bool drain(Connection& conn) {
        for (;;) {
            if (RINHA_UNLIKELY(conn.have == kBufSize)) return false;
            ssize_t r = ::read(conn.fd, conn.buf + conn.have,
                               kBufSize - conn.have);
            if (RINHA_LIKELY(r > 0)) {
                conn.have += static_cast<uint32_t>(r);
                if (!process(conn)) return false;
                continue;
            }
            if (r == 0) return false;
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
            return false;
        }
    }

    [[gnu::always_inline]]
    static const char* find_header_end(const char* data, std::size_t len) {
        return static_cast<const char*>(::memmem(data, len, "\r\n\r\n", 4));
    }

    [[gnu::always_inline]]
    Response dispatch(std::string_view path, std::string_view body) {
        if (RINHA_LIKELY(path.size() >= 2)) {
            if (RINHA_LIKELY(path[1] == 'f')) return self().on_fraud_score(body);
            if (path[1] == 'r')               return kReadyResponse;
        }
        return k404Response;
    }

    [[gnu::always_inline]]
    static int parse_content_length(const char* hdr, std::size_t len) {
        const char* p = static_cast<const char*>(::memmem(hdr, len, "ength:", 6));
        if (RINHA_UNLIKELY(!p)) return 0;
        p += 6;
        const char* end = hdr + len;
        while (p < end && *p == ' ') ++p;
        int n = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            n = n * 10 + (*p - '0');
            ++p;
        }
        return n;
    }

    [[gnu::always_inline]]
    static std::string_view extract_path(const char* req, std::size_t len) {
        const char* sp1 = static_cast<const char*>(std::memchr(req, ' ', len));
        if (RINHA_UNLIKELY(!sp1)) return {};
        ++sp1;
        std::size_t rem = len - static_cast<std::size_t>(sp1 - req);
        const char* sp2 = static_cast<const char*>(std::memchr(sp1, ' ', rem));
        if (RINHA_UNLIKELY(!sp2)) return {};
        return {sp1, static_cast<std::size_t>(sp2 - sp1)};
    }

    bool process(Connection& conn) {
        std::size_t consumed = 0;
        const char* base  = conn.buf;
        std::size_t avail = conn.have;

        while (consumed < avail) {
            const char* cur = base + consumed;
            std::size_t rem = avail - consumed;

            const char* hdr_end = find_header_end(cur, rem);
            if (!hdr_end) break;

            std::size_t hdr_len = static_cast<std::size_t>(hdr_end - cur) + 4;
            int         clen    = parse_content_length(cur, hdr_len);
            std::size_t total   = hdr_len + static_cast<std::size_t>(clen);
            if (rem < total) break;

            auto path = extract_path(cur, hdr_len);
            auto body = std::string_view(hdr_end + 4, static_cast<std::size_t>(clen));

            if (!send_all(conn.fd, dispatch(path, body))) return false;
            consumed += total;
        }

        if (consumed > 0) {
            auto remaining = static_cast<uint32_t>(avail - consumed);
            if (remaining > 0)
                std::memmove(conn.buf, conn.buf + consumed, remaining);
            conn.have = remaining;
        }
        return conn.have < kBufSize;
    }

    [[gnu::always_inline]]
    static bool send_all(int fd, Response resp) {
        std::size_t off = 0;
        while (off < resp.len) {
            ssize_t n = ::send(fd, resp.data + off, resp.len - off, MSG_NOSIGNAL);
            if (RINHA_LIKELY(n > 0)) { off += static_cast<std::size_t>(n); continue; }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd pfd{fd, POLLOUT, 0};
                int r;
                do { r = ::poll(&pfd, 1, 2); } while (r < 0 && errno == EINTR);
                if (r > 0 && (pfd.revents & POLLOUT)) continue;
            }
            return false;
        }
        return true;
    }

    Mode     mode_;
    char     socket_path_[108]{};
    uint16_t port_;
    int      listen_fd_ = -1;
    int      ep_fd_     = -1;
    unsigned conns_len_ = 0;

    struct ConnStorage { Connection c[kMaxConns]{}; };
    ConnStorage* conns_storage_ = new ConnStorage();
    Connection*  conns_         = conns_storage_->c;

    std::array<epoll_event, kMaxEvents> events_{};
};

} // namespace rinha
