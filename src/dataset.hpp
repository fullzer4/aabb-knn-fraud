#pragma once

#include "vec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <zlib.h>

namespace rinha {

struct Dataset {
    static constexpr uint32_t kMaxRefs = 3'100'000;

    uint32_t count = 0;
    double*  vectors = nullptr;
    uint8_t* labels  = nullptr;

    ~Dataset() { std::free(vectors); std::free(labels); }
    Dataset() = default;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    Dataset(Dataset&& o) noexcept
        : count(o.count),
          vectors(std::exchange(o.vectors, nullptr)),
          labels(std::exchange(o.labels, nullptr)) {}

    Dataset& operator=(Dataset&& o) noexcept {
        if (this != &o) {
            std::free(vectors);
            std::free(labels);
            count = o.count;
            vectors = std::exchange(o.vectors, nullptr);
            labels = std::exchange(o.labels, nullptr);
        }
        return *this;
    }

    bool is_fraud(uint32_t i) const noexcept {
        return (labels[i >> 3] >> (i & 7)) & 1;
    }

    void set_fraud(uint32_t i) noexcept {
        labels[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    }

    static Dataset load(const char* path) {
        Dataset ds;
        gzFile gz = gzopen(path, "rb");
        if (!gz) { std::fprintf(stderr, "open failed: %s\n", path); std::_Exit(1); }
        gzbuffer(gz, 1 << 17);

        std::size_t vb = std::size_t(kMaxRefs) * kDimsPad * sizeof(double);
        ds.vectors = static_cast<double*>(std::aligned_alloc(32, (vb + 31) & ~31u));
        std::memset(ds.vectors, 0, (vb + 31) & ~31u);
        ds.labels = static_cast<uint8_t*>(std::calloc((kMaxRefs + 7) / 8, 1));

        constexpr int kBuf = 1 << 16;
        char buf[kBuf];
        int blen = 0, bpos = 0;
        double tmp[kDimsPad]{};
        uint32_t idx = 0;

        auto refill = [&]() -> bool {
            if (bpos > 0 && bpos < blen) {
                std::memmove(buf, buf + bpos, std::size_t(blen - bpos));
                blen -= bpos;
                bpos = 0;
            } else if (bpos >= blen) {
                blen = bpos = 0;
            }
            int n = gzread(gz, buf + blen, kBuf - blen);
            if (n > 0) blen += n;
            return blen > bpos;
        };

        auto skip_ws = [&] {
            while ((bpos < blen || refill()) &&
                   (buf[bpos] == ' ' || buf[bpos] == '\n' || buf[bpos] == '\r' || buf[bpos] == '\t'))
                ++bpos;
        };

        auto peek = [&]() -> int {
            return (bpos < blen || refill()) ? static_cast<uint8_t>(buf[bpos]) : -1;
        };

        auto adv = [&] { ++bpos; };

        auto skip_to = [&](char t) {
            while ((bpos < blen || refill()) && buf[bpos] != t) ++bpos;
        };

        auto parse_double = [&]() -> double {
            char num[32];
            int ni = 0;
            while (ni < 31 && (bpos < blen || refill())) {
                char c = buf[bpos];
                if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') {
                    num[ni++] = c;
                    ++bpos;
                } else {
                    break;
                }
            }
            num[ni] = '\0';
            return std::strtod(num, nullptr);
        };

        skip_ws();
        if (peek() == '[') adv();

        while (idx < kMaxRefs) {
            skip_ws();
            int c = peek();
            if (c == ']' || c == -1) break;
            if (c == ',') { adv(); skip_ws(); c = peek(); }
            if (c == ']' || c == -1) break;
            if (c != '{') { adv(); continue; }
            adv();

            int dims = 0;
            bool fraud = false, got_v = false, got_l = false;

            for (int f = 0; f < 2; ++f) {
                skip_ws();
                int fc = peek();
                if (fc == ',') { adv(); skip_ws(); fc = peek(); }
                if (fc != '"') break;
                adv();
                char fn = (bpos < blen || refill()) ? buf[bpos] : '\0';
                skip_to('"'); adv(); skip_ws();
                if (peek() == ':') adv();
                skip_ws();

                if (fn == 'v') {
                    got_v = true;
                    if (peek() == '[') adv();
                    for (;;) {
                        skip_ws();
                        int vc = peek();
                        if (vc == ']') { adv(); break; }
                        if (vc == ',') { adv(); skip_ws(); }
                        if (dims < kDims) tmp[dims++] = parse_double();
                        else parse_double();
                    }
                } else if (fn == 'l') {
                    got_l = true;
                    if (peek() == '"') adv();
                    fraud = (peek() == 'f');
                    skip_to('"'); adv();
                }
            }
            skip_to('}'); adv();

            if (got_v) {
                for (int i = dims; i < kDimsPad; ++i) tmp[i] = 0.0;
                std::memcpy(ds.vectors + std::size_t(idx) * kDimsPad, tmp, sizeof(tmp));
                if (got_l && fraud) ds.set_fraud(idx);
                ++idx;
                if ((idx & 0xFFFFF) == 0) std::fprintf(stderr, "  %uM...\n", idx >> 20);
            }
        }

        gzclose(gz);
        ds.count = idx;
        std::fprintf(stderr, "loaded %u refs\n", ds.count);
        return ds;
    }
};

} // namespace rinha
