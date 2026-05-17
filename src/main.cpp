#include "http.hpp"
#include "index.hpp"
#include "vectorize.hpp"

#include <cstdlib>

static rinha::KMKNNIndex g_index;
static thread_local rinha::Vectorizer g_vec;

struct RinhaServer : rinha::Server<RinhaServer> {
    using Server::Server;

    [[gnu::hot]]
    rinha::Response on_fraud_score(std::string_view body) {
        rinha::Vec16 query{};
        uint8_t key{};
        if (RINHA_UNLIKELY(!g_vec.vectorize(body.data(), body.size(), query, key)))
            return rinha::k404Response;
        return rinha::kFraudResponses[g_index.query(query, key)];
    }
};

int main() {
    const char* index_path = std::getenv("RINHA_INDEX");
    if (!index_path) index_path = "/resources/index.bin";

    if (!g_index.load(index_path)) {
        std::fprintf(stderr, "fatal: index load failed: %s\n", index_path);
        return 1;
    }

    if (const char* path = std::getenv("SOCKET_PATH")) {
        std::fprintf(stderr, "listening unix:%s\n", path);
        RinhaServer{path}.run();
    }

    uint16_t port = 8080;
    if (const char* env = std::getenv("PORT"))
        port = static_cast<uint16_t>(std::atoi(env));

    std::fprintf(stderr, "listening :%u\n", port);
    RinhaServer{port}.run();
}
