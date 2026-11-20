#include "tokspur/http.hpp"
#include "tokspur/serve.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

// Original generated wire/socket fixtures only. No model, authentication secret,
// external server or engine is launched. All waits have short bounded deadlines.
namespace {
using namespace tokspur::http;
size_t checks = 0;
void need(bool condition, const char* message) {
    ++checks; if (!condition) throw std::runtime_error(message);
}
Deadline soon() { return Clock::now() + std::chrono::seconds(1); }
void rejects(const std::function<void()>& operation, int status, bool disconnected = false) {
    ++checks;
    try { operation(); }
    catch (const Error& error) {
        need(error.status() == status, "HTTP rejection status differs");
        if (disconnected) need(error.disconnected(), "HTTP disconnect was not identified");
        return;
    }
    throw std::runtime_error("Invalid HTTP operation accepted");
}
struct Pair {
    int server = -1, peer = -1;
    Pair() {
        std::array<int, 2> fds{};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds.data()) < 0) throw std::runtime_error("Cannot create HTTP test socketpair");
        server = fds[0]; peer = fds[1];
#ifdef SO_NOSIGPIPE
        const int yes = 1;
        if (setsockopt(peer, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) < 0) {
            ::close(server); ::close(peer); server = peer = -1;
            throw std::runtime_error("Cannot protect test socket SIGPIPE");
        }
#endif
    }
    ~Pair() { if (server >= 0) ::close(server); if (peer >= 0) ::close(peer); }
    int take() { return std::exchange(server, -1); }
    void close_peer() { if (peer >= 0) { ::close(peer); peer = -1; } }
};
void put(int fd, std::string_view data) {
    while (!data.empty()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const auto count = send(fd, data.data(), data.size(), flags);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Test HTTP peer write failed");
        data.remove_prefix(size_t(count));
    }
}
std::string get_all(int fd) {
    std::string result;
    const auto deadline = soon();
    std::array<char, 4096> bytes{};
    for (;;) {
        need(Clock::now() < deadline, "Test peer read did not reach bounded EOF");
        pollfd item{fd, POLLIN, 0};
        const auto ready = poll(&item, 1, 20);
        if (ready < 0 && errno == EINTR) continue;
        need(ready >= 0, "Test peer poll failed");
        if (!ready) continue;
        const auto count = recv(fd, bytes.data(), bytes.size(), MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        need(count >= 0, "Test peer response read failed");
        if (!count) return result;
        need(result.size() + size_t(count) <= 1024 * 1024, "Test HTTP response exceeded fixture bound");
        result.append(bytes.data(), size_t(count));
    }
}
std::string post(std::string_view body) {
    return "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:18081\r\nContent-Type: application/json\r\nAuthorization: Bearer fixture-only\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
}
std::string_view get_request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";

void fragmented_bytes() {
    const std::string utf8 = "{\"text\":\"\xf0\x9f\xa7\xa0\xe9\x9b\xaa\"}";
    const auto wire = post(utf8);
    for (size_t split = 0; split <= wire.size(); ++split) {
        Pair pair; Connection connection(pair.take());
        std::exception_ptr sender_error;
        std::jthread sender([&] {
            try { put(pair.peer, std::string_view(wire).substr(0, split)); std::this_thread::yield(); put(pair.peer, std::string_view(wire).substr(split)); }
            catch (...) { sender_error = std::current_exception(); }
        });
        const auto request = connection.read_request(soon()); sender.join();
        if (sender_error) std::rethrow_exception(sender_error);
        need(request.method == "POST" && request.path == "/v1/chat/completions" && request.body == utf8,
            "Fragmented HTTP request changed raw UTF-8 body bytes");
        need(request.header("Authorization") == "Bearer fixture-only" && request.header("CONTENT-TYPE") == "application/json" &&
            request.header("missing").empty(), "HTTP field normalization/lookup differs");
        need(!connection.disconnected(), "No-readable-data incorrectly treated as disconnect");
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, 400);
    }
    Pair pair; Connection connection(pair.take());
    const std::string binary = std::string("a\0b", 3) + "\xf0\x9f\xa7\xa0";
    put(pair.peer, post(binary));
    need(connection.read_request(soon()).body == binary, "Transport changed raw body NUL/UTF-8 before caller JSON validation");
}

void polled_requests() {
    const std::string body = "{\"text\":\"\xf0\x9f\xa7\xa0\"}";
    const auto wire = post(body);
    Pair pair; Connection connection(pair.take());
    need(!connection.poll_request(soon()).has_value(), "Polling a silent HTTP peer returned a request");
    for (size_t index = 0; index < wire.size(); ++index) {
        put(pair.peer, std::string_view(&wire[index], 1));
        const auto polled = connection.poll_request(soon());
        if (index + 1 < wire.size()) {
            need(!polled.has_value(), "Partial HTTP request was returned");
            continue;
        }
        need(polled.has_value(), "Whole HTTP request was not returned");
        need(polled->method == "POST" && polled->path == "/v1/chat/completions" &&
            polled->header("authorization") == "Bearer fixture-only" && polled->body == body,
            "Polled HTTP request differs from the blocking read");
    }
    rejects([&] { static_cast<void>(connection.poll_request(soon())); }, 400);
    rejects([&] { static_cast<void>(connection.read_request(soon())); }, 400);
    for (const auto& [bad_wire, status] : std::vector<std::pair<std::string_view, int>>{
        {"GET / HTTP/1.1\nHost: a\r\n\r\n", 400},
        {"GET / HTTP/1.0\r\nHost: a\r\n\r\n", 505},
        {"POST / HTTP/1.1\r\nHost: a\r\n\r\n", 411},
        {"GET /health HTTP/1.1\r\nHost: a\r\n\r\nGET", 400},
    }) {
        Pair bad; Connection bad_conn(bad.take());
        put(bad.peer, bad_wire);
        rejects([&] { static_cast<void>(bad_conn.poll_request(soon())); }, status);
        rejects([&] { static_cast<void>(bad_conn.poll_request(soon())); }, 400);
    }
    {
        Pair stalled; Connection stalled_conn(stalled.take());
        put(stalled.peer, "GET / HTTP/1.1\r\n");
        rejects([&] { static_cast<void>(stalled_conn.poll_request(Clock::now())); }, 408);
    }
    {
        Pair closed; Connection closed_conn(closed.take());
        put(closed.peer, "GET / HTTP/1.1\r\n");
        need(!closed_conn.poll_request(soon()).has_value(), "Partial HTTP request was returned");
        closed.close_peer();
        rejects([&] { static_cast<void>(closed_conn.poll_request(soon())); }, 400, true);
    }
    {
        Pair blocked; Connection blocked_conn(blocked.take());
        rejects([&] { static_cast<void>(blocked_conn.read_request(Clock::now())); }, 408);
        rejects([&] { static_cast<void>(blocked_conn.poll_request(soon())); }, 400);
    }
}

void malformed_requests() {
    const std::vector<std::pair<std::string, int>> cases{
        {"GET / HTTP/1.1\nHost: a\n\n", 400},
        {"GET / HTTP/1.1\rHost: a\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost:\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nhOsT: a\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost : a\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\n X-Test: value\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a b\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nX: a\r\nX: b\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\n\r\n", 411},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0,0\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: +1\r\n\r\nx", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: -1\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 00\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 184467440737095516160\r\n\r\n", 413},
        {"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: identity\r\n\r\n", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nExpect: 100-continue\r\n\r\n", 417},
        {"GET / HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nConnection: content-length\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nProxy-Connection: keep-alive\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nTrailer: X-Foo\r\n\r\n", 400},
        {"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nx", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n\r\nx", 400},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nxy", 400},
        {std::string(get_request) + std::string(get_request), 400},
        {"GET  / HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {"GET / HTTP/1.0\r\nHost: a\r\n\r\n", 505},
        {"PUT / HTTP/1.1\r\nHost: a\r\n\r\n", 405},
        {"GET http://localhost/ HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {"GET /#fragment HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {"GET /\\other HTTP/1.1\r\nHost: a\r\n\r\n", 400},
        {std::string("GET / HTTP/1.1\r\nHost: a\r\nX: a") + std::string("\0", 1) + "b\r\n\r\n", 400}
    };
    for (const auto& [wire, status] : cases) {
        Pair pair; Connection connection(pair.take()); put(pair.peer, wire);
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, status);
    }
    for (const auto wire : {std::string_view("GET / HTTP/1.1\r\nHost: a\r\n"),
        std::string_view("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\n\r\nx")}) {
        Pair pair; Connection connection(pair.take()); put(pair.peer, wire); shutdown(pair.peer, SHUT_WR);
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, 400, true);
    }
    Pair pair; Connection connection(pair.take()); pair.close_peer();
    rejects([&] { static_cast<void>(connection.read_request(soon())); }, 0, true);
}

void limits_and_deadlines() {
    for (int variant = 0; variant < 5; ++variant) {
        Limits limits; std::string wire; int status = 431;
        if (variant == 0) { limits.header_bytes = 64; limits.header_line_bytes = 64; wire = "GET / HTTP/1.1\r\nHost: a\r\nX: " + std::string(70, 'x'); }
        if (variant == 1) { limits.header_line_bytes = 16; wire = "GET / HTTP/1.1\r\nHost: a\r\nX: " + std::string(17, 'x') + "\r\n\r\n"; }
        if (variant == 2) { limits.header_count = 1; wire = "GET / HTTP/1.1\r\nHost: a\r\nX: x\r\n\r\n"; }
        if (variant == 3) { limits.target_bytes = 2; wire = "GET /long HTTP/1.1\r\nHost: a\r\n\r\n"; status = 400; }
        if (variant == 4) { limits.body_bytes = 2; wire = post("123"); status = 413; }
        Pair pair; Connection connection(pair.take(), limits); put(pair.peer, wire);
        rejects([&] { static_cast<void>(connection.read_request(soon())); }, status);
    }
    {
        Pair pair; Connection connection(pair.take());
        rejects([&] { static_cast<void>(connection.read_request(Clock::now())); }, 408);
    }
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, "GET / HTTP/1.1\r\n");
        rejects([&] { static_cast<void>(connection.read_request(Clock::now() + std::chrono::milliseconds(2))); }, 408);
    }
    {
        Pair pair; Limits invalid; invalid.event_bytes = 0; const auto fd = pair.take();
        rejects([&] { Connection adopted(fd, invalid); }, 500);
        need(fcntl(fd, F_GETFD) < 0 && errno == EBADF, "Failed HTTP constructor leaked adopted fd");
    }
    {
        Pair pair;
        const int small = 1024;
        need(setsockopt(pair.server, SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)) == 0, "Cannot bound backpressure fixture socket");
        Connection connection(pair.take());
        connection.begin_sse(soon());
        rejects([&] { connection.send_sse_data(std::string(512 * 1024, 'x'), Clock::now() + std::chrono::milliseconds(3)); }, 408);
    }
}

std::vector<std::string> decode_chunks(std::string_view body) {
    std::vector<std::string> chunks;
    for (;;) {
        const auto end = body.find("\r\n"); need(end != std::string_view::npos && end > 0, "Missing HTTP chunk size");
        size_t count = 0;
        const auto parsed = std::from_chars(body.data(), body.data() + end, count, 16);
        need(parsed.ec == std::errc{} && parsed.ptr == body.data() + end, "Invalid HTTP hex chunk size");
        body.remove_prefix(end + 2);
        if (!count) { need(body == "\r\n", "Invalid terminal HTTP zero chunk/trailer"); return chunks; }
        need(count <= body.size() && body.size() - count >= 2 && body.substr(count, 2) == "\r\n", "HTTP chunk byte length/framing differs");
        chunks.emplace_back(body.substr(0, count)); body.remove_prefix(count + 2);
        need(chunks.size() <= 8, "Bounded test chunk count exceeded");
    }
}
void response_wire() {
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, get_request); static_cast<void>(connection.read_request(soon()));
        const std::string body = "{\"text\":\"\xe9\x9b\xaa\"}";
        connection.send_json(200, body, soon());
        rejects([&] { connection.send_json(200, "{}", soon()); }, 500);
        rejects([&] { connection.begin_sse(soon()); }, 500);
        connection.close(); const auto raw = get_all(pair.peer); const auto end = raw.find("\r\n\r\n");
        need(raw.starts_with("HTTP/1.1 200 OK\r\n") && end != std::string::npos && raw.substr(end + 4) == body, "JSON HTTP status/body changed");
        need(raw.find("Content-Length: " + std::to_string(body.size()) + "\r\n") != std::string::npos &&
            raw.find("Connection: close\r\n") != std::string::npos && raw.find("Access-Control-") == std::string::npos &&
            raw.find("Transfer-Encoding:") == std::string::npos, "JSON HTTP response framing or private policy differs");
    }
    {
        // A response is not bounded by the request body limit: a reply may be larger than any request the server admits.
        Pair pair; Limits limits; limits.body_bytes = 16;
        Connection connection(pair.take(), limits); put(pair.peer, get_request); static_cast<void>(connection.read_request(soon()));
        const std::string body(4096, 'x');
        connection.send_json(200, body, soon()); connection.close();
        const auto raw = get_all(pair.peer);
        need(raw.find("\r\nContent-Length: 4096\r\n") != std::string::npos && raw.ends_with(body),
            "JSON response was cut to the request body limit");
        // The response bound is its own, and refusing by it leaves the connection free to send the error.
        Pair second; Connection err_conn(second.take(), limits);
        const std::string oversized(json_response_bytes + 1, 'x');
        rejects([&] { err_conn.send_json(200, oversized, soon()); }, 500);
        err_conn.send_json(502, "{}", soon()); err_conn.close();
        need(get_all(second.peer).starts_with("HTTP/1.1 502 Bad Gateway\r\n"),
            "Oversized JSON response was partly written");
    }
    {
        // A refusal goes out whole without the request having been read, and is the connection's one response.
        Pair pair; Connection connection(pair.take());
        put(pair.peer, "GET /health HTTP/1.1\r\nHost: loc");
        constexpr std::string_view body = "{\"error\":\"busy\"}";
        connection.refuse_json(503, body);
        rejects([&] { connection.refuse_json(503, body); }, 500);
        rejects([&] { connection.send_json(200, "{}", soon()); }, 500);
        rejects([&] { static_cast<void>(connection.poll_request(soon())); }, 400);
        connection.close();
        const auto raw = get_all(pair.peer);
        const auto end = raw.find("\r\n\r\n");
        need(raw.starts_with("HTTP/1.1 503 Service Unavailable\r\n") &&
            raw.find("\r\nRetry-After: 5\r\n") != std::string::npos &&
            raw.find("\r\nConnection: close\r\n") != std::string::npos &&
            end != std::string::npos && raw.substr(end + 4) == body,
            "Unwaited JSON refusal status/framing/body differs");
        Pair gone; Connection gone_conn(gone.take()); gone.close_peer();
        rejects([&] { gone_conn.refuse_json(503, body); }, 0, true);
        Pair bad_status; Connection bad_conn(bad_status.take());
        rejects([&] { bad_conn.refuse_json(599, body); }, 500);
        bad_conn.send_json(200, "{}", soon());
    }
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, get_request); static_cast<void>(connection.read_request(soon()));
        connection.begin_sse(soon());
        const std::string data = "\xf0\x9f\xa7\xa0\nsecond\n";
        connection.send_sse_data(data, soon()); connection.send_sse_data("", soon()); connection.send_sse_data("[DONE]", soon());
        rejects([&] { connection.send_sse_data("bad\rline", soon()); }, 500);
        rejects([&] { connection.send_sse_data(std::string("a\0b", 3), soon()); }, 500);
        connection.finish_sse(soon());
        rejects([&] { connection.finish_sse(soon()); }, 500);
        rejects([&] { connection.send_sse_data("x", soon()); }, 500);
        connection.close(); const auto raw = get_all(pair.peer); const auto end = raw.find("\r\n\r\n");
        need(end != std::string::npos && raw.find("Transfer-Encoding: chunked\r\n") < end &&
            raw.find("Content-Length:") == std::string::npos && raw.find("Content-Type: text/event-stream; charset=utf-8\r\n") < end,
            "SSE response must use valid chunked framing and UTF-8 media type");
        const auto chunks = decode_chunks(std::string_view(raw).substr(end + 4));
        need(chunks == std::vector<std::string>{"data: \xf0\x9f\xa7\xa0\ndata: second\ndata: \n\n", "data: \n\n", "data: [DONE]\n\n"},
            "SSE UTF-8/multiline/empty/terminal data framing differs");
    }
    {
        Pair pair; Limits limits; limits.event_bytes = 8; Connection connection(pair.take(), limits);
        connection.begin_sse(soon()); connection.send_sse_data("", soon());
        rejects([&] { connection.send_sse_data("x", soon()); }, 500);
        rejects([&] { connection.send_sse_data("\n", soon()); }, 500);
        connection.finish_sse(soon());
    }
    for (const auto status : {401, 405}) {
        Pair pair; Connection connection(pair.take()); connection.send_json(status, "{}", soon()); connection.close();
        const auto response = get_all(pair.peer);
        need(response.find(status == 401 ? "WWW-Authenticate: Bearer realm=\"tokspur\"\r\n" : "Allow: GET, POST\r\n") != std::string::npos,
            "Required authentication/method response metadata missing");
    }
    {
        Pair pair; Connection connection(pair.take()); connection.send_json(502, "{}", soon()); connection.close();
        need(get_all(pair.peer).starts_with("HTTP/1.1 502 Bad Gateway\r\n"), "Owned worker failure status could not be written");
    }
}

void ownership_disconnect_listener() {
    {
        Pair pair; Connection first(pair.take()); const auto fd = first.native_handle();
        need((fcntl(fd, F_GETFL) & O_NONBLOCK) && (fcntl(fd, F_GETFD) & FD_CLOEXEC), "HTTP socket flags are not nonblocking+CLOEXEC");
        Connection moved(std::move(first)); need(first.native_handle() == -1 && moved.native_handle() == fd, "HTTP move constructor duplicated ownership");
        Pair replacement; Connection second(replacement.take()); const auto old = second.native_handle(); second = std::move(moved);
        need(moved.native_handle() == -1 && second.native_handle() == fd && fcntl(old, F_GETFD) < 0, "HTTP move assignment leaked previous fd");
        second.close(); need(fcntl(fd, F_GETFD) < 0 && second.disconnected(), "HTTP close failed to drain fd ownership");
    }
    {
        Pair pair; Connection connection(pair.take()); put(pair.peer, get_request);
        need(!connection.disconnected(), "Unread initial HTTP request misclassified as disconnect");
        static_cast<void>(connection.read_request(soon())); put(pair.peer, "x");
        need(connection.disconnected(), "Delayed HTTP pipeline byte did not cancel connection");
        char value = 0;
        need(recv(connection.native_handle(), &value, 1, MSG_PEEK | MSG_DONTWAIT) == 1 && value == 'x', "HTTP disconnect check consumed unexpected input");
        rejects([&] { connection.send_json(200, "{}", soon()); }, 0, true);
    }
    {
        Pair pair; Connection connection(pair.take()); pair.close_peer(); need(connection.disconnected(), "Peer close not detected");
        rejects([&] { connection.send_json(200, "{}", soon()); }, 0, true); // Must not deliver SIGPIPE.
    }
    Listener listener(0); need(listener.port() != 0, "Loopback test listener failed to adopt ephemeral port");
    sockaddr_in address{}; socklen_t bytes = sizeof(address);
    need(getsockname(listener.native_handle(), reinterpret_cast<sockaddr*>(&address), &bytes) == 0 &&
        address.sin_family == AF_INET && address.sin_addr.s_addr == htonl(INADDR_LOOPBACK), "Listener escaped IPv4 loopback");
    rejects([&] { Listener occupied(listener.port()); }, 0);
    rejects([&] { static_cast<void>(listener.accept(Clock::now())); }, 408);
    const auto client = socket(AF_INET, SOCK_STREAM, 0); need(client >= 0, "Cannot create loopback test client");
    struct Client { int fd; ~Client() { if (fd >= 0) ::close(fd); } } owned_client{client};
    need(connect(client, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect to owned loopback listener");
    auto connection = listener.accept(soon()); put(client, get_request);
    need(connection.read_request(soon()).path == "/health", "Accepted loopback request differs");
    connection.send_json(200, "{}", soon()); connection.close(); need(get_all(client).ends_with("\r\n\r\n{}"), "Loopback HTTP exchange failed");
    {
        // A burst of abortive RST peers before/during accept never ends the listener; it still serves the next valid client.
        const auto reset_peer = [&] {
            const int rst = socket(AF_INET, SOCK_STREAM, 0);
            need(rst >= 0, "Cannot create loopback RST client");
            const linger abortive{1, 0};
            need(setsockopt(rst, SOL_SOCKET, SO_LINGER, &abortive, sizeof(abortive)) == 0, "Cannot arm abortive close");
            need(connect(rst, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect loopback RST client");
            ::close(rst);
        };
        for (int i = 0; i < 6; ++i) reset_peer();
        for (;;) {
            try {
                auto stale = listener.accept(Clock::now() + std::chrono::milliseconds(15));
                rejects([&] { static_cast<void>(stale.read_request(soon())); }, 0, true);
            } catch (const Error& error) {
                need(error.status() == 408 && !error.disconnected(), "Abortive peer burst failed listener accept");
                break;
            }
        }
        reset_peer();
        const int next_fd = socket(AF_INET, SOCK_STREAM, 0);
        need(next_fd >= 0, "Cannot create post-RST test client");
        Client next_client{next_fd};
        need(connect(next_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect post-RST test client");
        put(next_fd, get_request);
        bool served = false;
        for (int attempt = 0; attempt < 4 && !served; ++attempt) {
            auto accepted = listener.accept(soon());
            try {
                need(accepted.read_request(soon()).path == "/health", "Post-RST request path differs");
                accepted.send_json(200, "{}", soon());
                served = true;
            } catch (const Error& error) {
                need(error.disconnected(), "Unexpected non-disconnect error on post-RST accept");
            }
        }
        need(served && get_all(next_fd).ends_with("\r\n\r\n{}"), "Listener did not serve valid client after RST burst");
    }
    {
        // Descriptor exhaustion (EMFILE) during accept waits to the deadline and returns 408 instead of ending the listener.
        const int pending_fd = socket(AF_INET, SOCK_STREAM, 0);
        need(pending_fd >= 0, "Cannot create EMFILE test client");
        Client pending_client{pending_fd};
        need(connect(pending_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect EMFILE test client");
        rlimit saved{};
        need(getrlimit(RLIMIT_NOFILE, &saved) == 0, "Cannot inspect RLIMIT_NOFILE");
        {
            struct FillFds {
                std::vector<int> fds;
                ~FillFds() { for (const int fd : fds) ::close(fd); }
            } filler;
            int highest = std::max(listener.native_handle(), pending_fd);
            for (int i = 0; i < 64; ++i) {
                int pair_fds[2]{-1, -1};
                need(::pipe(pair_fds) == 0, "Cannot allocate filler pipe for EMFILE check");
                filler.fds.push_back(pair_fds[0]);
                filler.fds.push_back(pair_fds[1]);
                const int top = std::max(pair_fds[0], pair_fds[1]);
                if (top > highest) { highest = top; break; }
            }
            struct RestoreLimit {
                rlimit original;
                ~RestoreLimit() { static_cast<void>(setrlimit(RLIMIT_NOFILE, &original)); }
            } restore{saved};
            const rlimit tight{rlim_t(highest + 1), saved.rlim_max};
            need(setrlimit(RLIMIT_NOFILE, &tight) == 0, "Cannot lower RLIMIT_NOFILE for EMFILE check");
            rejects([&] { static_cast<void>(listener.accept(Clock::now() + std::chrono::milliseconds(15))); }, 408);
        }
        ::close(std::exchange(pending_client.fd, -1));
        for (;;) {
            try { static_cast<void>(listener.accept(Clock::now() + std::chrono::milliseconds(5))); }
            catch (const Error& error) { need(error.status() == 408, "Unexpected accept status draining EMFILE peer"); break; }
        }
        const int recovered_fd = socket(AF_INET, SOCK_STREAM, 0);
        need(recovered_fd >= 0, "Cannot create post-EMFILE test client");
        Client recovered_client{recovered_fd};
        need(connect(recovered_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect post-EMFILE test client");
        auto recovered = listener.accept(soon());
        put(recovered_fd, get_request);
        need(recovered.read_request(soon()).path == "/health", "Post-EMFILE accepted request differs");
        recovered.send_json(200, "{}", soon());
        recovered.close();
        need(get_all(recovered_fd).ends_with("\r\n\r\n{}"), "Post-EMFILE HTTP exchange failed");
    }
    Listener moved(std::move(listener)); need(listener.native_handle() == -1 && moved.port() != 0, "Listener move lost socket/port ownership");
    const auto port = moved.port(); moved.close(); rejects([&] { static_cast<void>(moved.accept(soon())); }, 0);
    rejects([&] { static_cast<void>(moved.poll_input({}, 0)); }, 0);
    Listener restarted(port); need(restarted.port() == port, "Closed listener could not restart while its connection was in TIME_WAIT");
}

void busy_and_multiplexed_peers() {
    const auto assert_busy = [&](std::string_view response) {
        need(response.starts_with("HTTP/1.1 503 Service Unavailable\r\n") &&
            response.find("\r\nRetry-After: 5\r\n") != std::string_view::npos &&
            response.ends_with(R"({"error":{"message":"server is busy with other connections; retry shortly","type":"server_busy","code":"server_busy"}})"),
            "503 server_busy response framing/body differs");
    };
    {
        // A full reading table evicts its oldest unfinished peer with 503 server_busy.
        std::vector<std::pair<Connection, Clock::time_point>> reading;
        std::vector<Pair> peers(tokspur::reading_peers);
        for (size_t i = 0; i < tokspur::reading_peers; ++i) {
            tokspur::admit_reading(reading, Connection(peers[i].take()), Clock::now());
        }
        const int second_handle = reading[1].first.native_handle();
        Pair newest_pair;
        Connection newest(newest_pair.take());
        const int newest_handle = newest.native_handle();
        tokspur::admit_reading(reading, std::move(newest), Clock::now());
        need(reading.size() == tokspur::reading_peers &&
            reading.front().first.native_handle() == second_handle &&
            reading.back().first.native_handle() == newest_handle,
            "Reading table eviction order differs");
        assert_busy(get_all(peers[0].peer));
        reading.clear();
        for (size_t i = 1; i < peers.size(); ++i) {
            need(get_all(peers[i].peer).empty(), "Non-evicted reading peer received an unexpected response");
        }
    }
    {
        // A full hand-off queue answers 503 server_busy without touching queued peers, and reports stopped workers.
        tokspur::ArrivalQueue queue(1, 1);
        Pair queued_pair, refused_pair, late_pair;
        need(queue.hand_off(tokspur::Arrival{Connection(queued_pair.take()), Clock::now(), Request{}}), "Initial hand_off failed");
        need(queue.hand_off(tokspur::Arrival{Connection(refused_pair.take()), Clock::now(), Request{}}), "Full hand_off reported stopped workers");
        assert_busy(get_all(refused_pair.peer));
        auto taken = queue.recv();
        need(taken.has_value(), "Queued arrival was lost");
        taken.reset();
        need(get_all(queued_pair.peer).empty(), "Queued peer received an unexpected busy response");
        queue.close();
        need(!queue.hand_off(tokspur::Arrival{Connection(late_pair.take()), Clock::now(), Request{}}), "Closed ArrivalQueue accepted hand_off");
    }
    {
        // Multiplexed pre-auth reading: 24 silent/stalled peers and disconnected peers never block valid /health requests.
        Listener listener(0, 8);
        sockaddr_in address{}; socklen_t bytes = sizeof(address);
        need(getsockname(listener.native_handle(), reinterpret_cast<sockaddr*>(&address), &bytes) == 0, "Cannot inspect listener address");
        struct CloseFds { std::vector<int> fds; ~CloseFds() { for (const int fd : fds) if (fd >= 0) ::close(fd); } } quiet;
        std::vector<std::pair<Connection, Clock::time_point>> reading;
        for (int index = 0; index < 24; ++index) {
            const int fd = socket(AF_INET, SOCK_STREAM, 0);
            need(fd >= 0 && connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect pre-auth peer");
            if (index % 3 == 1) put(fd, "GET /health HTTP/1.1\r\nHost: loc");
            if (index % 3 == 2) ::close(fd);
            else quiet.fds.push_back(fd);
            tokspur::admit_reading(reading, listener.accept(soon()), Clock::now());
        }
        const int valid_fd = socket(AF_INET, SOCK_STREAM, 0);
        need(valid_fd >= 0 && connect(valid_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "Cannot connect valid client behind silent peers");
        quiet.fds.push_back(valid_fd);
        put(valid_fd, get_request);
        bool answered_valid = false;
        const auto deadline = soon();
        while (!answered_valid) {
            need(Clock::now() < deadline, "Valid request behind silent pre-auth peers timed out");
            std::vector<int> handles;
            handles.reserve(reading.size());
            for (const auto& [conn, arrival] : reading) { static_cast<void>(arrival); handles.push_back(conn.native_handle()); }
            const auto ready = listener.poll_input(handles, 20);
            need(ready.size() == handles.size() + 1, "Listener::poll_input result size differs");
            std::vector<std::pair<Connection, Clock::time_point>> unfinished;
            for (size_t i = 0; i < reading.size(); ++i) {
                auto [conn, arrival] = std::move(reading[i]);
                if (!ready[i + 1]) { unfinished.emplace_back(std::move(conn), arrival); continue; }
                try {
                    auto polled = conn.poll_request(arrival + tokspur::http_read_seconds);
                    if (!polled) { unfinished.emplace_back(std::move(conn), arrival); continue; }
                    need(polled->path == "/health", "Multiplexed request path differs");
                    conn.send_json(200, "{\"status\":\"ready\"}", soon());
                    conn.close();
                    answered_valid = true;
                } catch (const Error& error) {
                    need(error.disconnected(), "Unexpected error polling pre-auth peer");
                }
            }
            reading = std::move(unfinished);
            if (ready[0]) {
                try { tokspur::admit_reading(reading, listener.accept(Clock::now() + std::chrono::milliseconds(5)), Clock::now()); }
                catch (const Error& error) { need(error.status() == 408, "Unexpected accept failure while multiplexing"); }
            }
        }
        need(get_all(valid_fd).ends_with("{\"status\":\"ready\"}"), "Multiplexed /health response differs");
        need(reading.size() == 16, "Closed pre-auth peers were not pruned while stalled peers stayed tracked");
    }
}
} // namespace
int main() {
    try {
        fragmented_bytes(); polled_requests(); malformed_requests(); limits_and_deadlines(); response_wire(); ownership_disconnect_listener(); busy_and_multiplexed_peers();
        std::cout << "Owned CPU HTTP/SSE checks passed: " << checks << "; model-free socket/wire fixtures, not native inference or release qualification\n";
    } catch (const std::exception& error) { std::cerr << "Owned HTTP/SSE check failed: " << error.what() << '\n'; return 1; }
}
