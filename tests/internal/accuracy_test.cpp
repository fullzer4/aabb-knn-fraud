#include "index.hpp"
#include "vec.hpp"
#include "vectorize.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <simdjson.h>

using namespace rinha;

class AccuracyTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto env_or = [](const char* var, const char* def) {
            const char* v = std::getenv(var);
            return (v && v[0]) ? v : def;
        };
        const char* index_path = env_or("INDEX_PATH", "src/index.bin");
        test_path_ = env_or("TESTDATA_PATH", "tests/internal/testdata.json");
        ASSERT_TRUE(index_.load(index_path)) << "index load failed: " << index_path;
    }

    static simdjson::dom::array load_entries() {
        static std::string json;
        static simdjson::dom::parser dom;
        static bool loaded = false;

        if (!loaded) {
            std::ifstream file(test_path_);
            json.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            loaded = true;
        }
        return dom.parse(json)["entries"].get_array().value_unsafe();
    }

    static const char* test_path_;
    static KMKNNIndex  index_;
};

const char* AccuracyTest::test_path_;
KMKNNIndex  AccuracyTest::index_;

TEST_F(AccuracyTest, ZeroFailureRate) {
    auto entries = load_entries();
    Vectorizer vectorizer;

    uint32_t total = 0, failures = 0;

    for (auto entry : entries) {
        bool expected_approved = entry["expected_approved"].get_bool().value_unsafe();

        simdjson::dom::element req_elem;
        if (entry["request"].get(req_elem)) { ++failures; ++total; continue; }
        std::string req_str = simdjson::minify(req_elem);

        Vec16 query{};
        uint8_t key{};
        if (!vectorizer.vectorize(req_str.data(), req_str.size(), query, key)) {
            ++failures;
            ++total;
            continue;
        }

        int fraud = index_.query(query, key);
        bool approved = fraud < 3;

        if (approved != expected_approved) ++failures;
        ++total;
    }

    double failure_rate = static_cast<double>(failures) / total;
    std::fprintf(stderr, "\n  %u/%u correct (failure_rate=%.4f%%)\n\n",
                 total - failures, total, failure_rate * 100.0);

    EXPECT_EQ(failures, 0u);
}

TEST_F(AccuracyTest, AllPayloadsVectorize) {
    auto entries = load_entries();
    Vectorizer vectorizer;

    uint32_t tested = 0, failed = 0;

    for (auto entry : entries) {
        simdjson::dom::element req_elem;
        if (entry["request"].get(req_elem)) { ++failed; ++tested; continue; }
        std::string req_str = simdjson::minify(req_elem);

        Vec16 query{};
        uint8_t key{};
        if (!vectorizer.vectorize(req_str.data(), req_str.size(), query, key))
            ++failed;
        ++tested;
    }

    EXPECT_EQ(failed, 0u) << failed << "/" << tested << " failed";
}

TEST_F(AccuracyTest, FraudCountInRange) {
    auto entries = load_entries();
    Vectorizer vectorizer;

    uint32_t tested = 0;
    for (auto entry : entries) {
        simdjson::dom::element req_elem;
        if (entry["request"].get(req_elem)) continue;
        std::string req_str = simdjson::minify(req_elem);

        Vec16 query{};
        uint8_t key{};
        if (!vectorizer.vectorize(req_str.data(), req_str.size(), query, key)) continue;

        int fraud = index_.query(query, key);
        ASSERT_GE(fraud, 0) << "entry " << tested;
        ASSERT_LE(fraud, kK) << "entry " << tested;

        if (++tested >= 1000) break;
    }
}
