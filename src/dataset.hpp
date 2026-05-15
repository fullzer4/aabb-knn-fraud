#pragma once

#include "vec.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <zlib.h>

namespace rinha {

struct Dataset {
    static constexpr uint32_t kMaxRefs = 3'100'000;

    uint32_t       count = 0;
    uint16_t*      vectors = nullptr;  // count × kDimsPad float16s, 32-byte aligned
    uint8_t*       labels  = nullptr;  // packed bits: 1=fraud, 0=legit

    ~Dataset() {
        std::free(vectors);
        std::free(labels);
    }

    Dataset() = default;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&& o) noexcept
        : count(o.count), vectors(o.vectors), labels(o.labels) {
        o.vectors = nullptr;
        o.labels = nullptr;
    }

    bool is_fraud(uint32_t idx) const {
        return (labels[idx >> 3] >> (idx & 7)) & 1;
    }

    void set_fraud(uint32_t idx) {
        labels[idx >> 3] |= static_cast<uint8_t>(1u << (idx & 7));
    }

    // ── Streaming parser ──────────────────────────────────────
    // Parses gzipped JSON array of {"vector": [...], "label": "..."}
    // without loading the full decompressed text into memory.
    // Uses a small read buffer and a simple state machine.

    static Dataset load(const char* path) {
        Dataset ds;

        gzFile gz = gzopen(path, "rb");
        if (!gz) {
            std::fprintf(stderr, "failed to open: %s\n", path);
            std::_Exit(1);
        }
        gzbuffer(gz, 1 << 17); // 128KB zlib buffer

        // Pre-allocate output arrays
        std::size_t vec_bytes = static_cast<std::size_t>(kMaxRefs) * kDimsPad * sizeof(uint16_t);
        ds.vectors = static_cast<uint16_t*>(std::aligned_alloc(32, (vec_bytes + 31) & ~31u));
        std::memset(ds.vectors, 0, (vec_bytes + 31) & ~31u);

        std::size_t label_bytes = (kMaxRefs + 7) / 8;
        ds.labels = static_cast<uint8_t*>(std::calloc(label_bytes, 1));

        // Read buffer for streaming
        constexpr int kBufSize = 1 << 16; // 64KB
        char buf[kBufSize];
        int buf_len = 0;
        int buf_pos = 0;

        alignas(32) float tmp[kDimsPad]{};
        uint32_t idx = 0;

        // Lambda to refill buffer, keeping unprocessed data
        auto refill = [&]() -> bool {
            if (buf_pos > 0 && buf_pos < buf_len) {
                std::memmove(buf, buf + buf_pos, static_cast<std::size_t>(buf_len - buf_pos));
                buf_len -= buf_pos;
                buf_pos = 0;
            } else if (buf_pos >= buf_len) {
                buf_len = 0;
                buf_pos = 0;
            }
            int n = gzread(gz, buf + buf_len, kBufSize - buf_len);
            if (n > 0) buf_len += n;
            return buf_len > buf_pos;
        };

        // Skip whitespace
        auto skip_ws = [&]() {
            while (buf_pos < buf_len || refill()) {
                char c = buf[buf_pos];
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
                    ++buf_pos;
                else
                    return;
            }
        };

        // Peek current char (with refill)
        auto peek = [&]() -> int {
            if (buf_pos >= buf_len && !refill()) return -1;
            return static_cast<unsigned char>(buf[buf_pos]);
        };

        // Advance one char
        auto advance = [&]() { ++buf_pos; };

        // Skip until we find a specific char
        auto skip_to = [&](char target) {
            while (buf_pos < buf_len || refill()) {
                if (buf[buf_pos] == target) return;
                ++buf_pos;
            }
        };

        // Parse a float value (fast path)
        auto parse_float = [&]() -> float {
            // Collect chars into a small local buffer for strtof
            char num[32];
            int ni = 0;
            while (ni < 31) {
                if (buf_pos >= buf_len && !refill()) break;
                char c = buf[buf_pos];
                if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') {
                    num[ni++] = c;
                    ++buf_pos;
                } else {
                    break;
                }
            }
            num[ni] = '\0';
            return std::strtof(num, nullptr);
        };

        // Parse: expects the outer '[' then stream of objects
        skip_ws();
        if (peek() == '[') advance(); // skip opening '['

        while (idx < kMaxRefs) {
            skip_ws();
            int c = peek();
            if (c == ']' || c == -1) break;
            if (c == ',') { advance(); skip_ws(); c = peek(); }
            if (c == ']' || c == -1) break;
            if (c != '{') { advance(); continue; }
            advance(); // skip '{'

            // Parse one object: {"vector": [...], "label": "..."}
            // Fields can be in any order
            int dims_read = 0;
            bool is_fraud_label = false;
            bool got_vector = false, got_label = false;

            for (int field = 0; field < 2; ++field) {
                skip_ws();
                int fc = peek();
                if (fc == ',') { advance(); skip_ws(); fc = peek(); }
                if (fc != '"') break;
                advance(); // skip opening '"'

                // Read field name
                char fname = '\0';
                if (buf_pos < buf_len || refill())
                    fname = buf[buf_pos]; // 'v' for vector, 'l' for label

                // Skip to end of key string
                skip_to('"');
                advance(); // skip closing '"'
                skip_ws();
                if (peek() == ':') advance(); // skip ':'
                skip_ws();

                if (fname == 'v') {
                    // Parse vector array
                    got_vector = true;
                    if (peek() == '[') advance(); // skip '['
                    dims_read = 0;
                    for (;;) {
                        skip_ws();
                        int vc = peek();
                        if (vc == ']') { advance(); break; }
                        if (vc == ',') { advance(); skip_ws(); }
                        if (dims_read < kDims)
                            tmp[dims_read++] = parse_float();
                        else
                            parse_float(); // discard extra dims
                    }
                } else if (fname == 'l') {
                    // Parse label string
                    got_label = true;
                    if (peek() == '"') advance(); // skip opening '"'
                    is_fraud_label = (peek() == 'f'); // "fraud" starts with 'f'
                    skip_to('"');
                    advance(); // skip closing '"'
                }
            }

            // Skip to end of object
            skip_to('}');
            advance(); // skip '}'

            if (got_vector) {
                // Pad remaining dims
                for (int i = dims_read; i < kDimsPad; ++i) tmp[i] = 0.0f;

                vec_f32_to_f16(tmp, ds.vectors + static_cast<std::size_t>(idx) * kDimsPad);

                if (got_label && is_fraud_label)
                    ds.set_fraud(idx);

                ++idx;

                if ((idx & 0xFFFFF) == 0)
                    std::fprintf(stderr, "  loaded %uM references...\n", idx >> 20);
            }
        }

        gzclose(gz);

        ds.count = idx;
        std::fprintf(stderr, "loaded %u references\n", ds.count);
        return ds;
    }
};

} // namespace rinha
