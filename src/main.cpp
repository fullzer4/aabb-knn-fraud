#include "http.hpp"
#include "ivf.hpp"
#include "vectorize.hpp"

static rinha::KNN        g_index;
static thread_local rinha::Vectorizer g_vec;

struct RinhaServer : rinha::Server<RinhaServer> {
    using Server::Server;

    [[gnu::hot]] rinha::Response on_fraud_score(std::string_view body) {
        alignas(32) float query[rinha::kDimsPad];

        if (RINHA_UNLIKELY(!g_vec.vectorize(body.data(), body.size(), query)))
            return rinha::k404Response;

        int fraud_count = g_index.query(query);
        return rinha::kFraudResponses[fraud_count];
    }
};

int main() {
    const char* refs_path = std::getenv("REFS_PATH");
    if (!refs_path) refs_path = "/resources/references.json.gz";

    int nclusters = 1024;
    int nprobe    = 20;
    if (const char* env = std::getenv("NCLUSTERS"))
        nclusters = std::atoi(env);
    if (const char* env = std::getenv("NPROBE"))
        nprobe = std::atoi(env);

    std::fprintf(stderr, "rinha 2026 | loading %s\n", refs_path);
    auto dataset = rinha::Dataset::load(refs_path);
    g_index.build(dataset, nclusters, nprobe);
    std::fprintf(stderr, "rinha 2026 | index ready\n");

    if (const char* path = std::getenv("SOCKET_PATH")) {
        std::fprintf(stderr, "rinha 2026 | listening on unix:%s\n", path);
        RinhaServer{path}.run();
    }

    uint16_t port = 8080;
    if (const char* env = std::getenv("PORT"))
        port = static_cast<uint16_t>(std::atoi(env));

    std::fprintf(stderr, "rinha 2026 | listening on :%u\n", port);
    RinhaServer{port}.run();
}
