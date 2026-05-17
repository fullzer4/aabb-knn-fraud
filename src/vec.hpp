#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <immintrin.h>

namespace rinha {

inline constexpr int   kDims      = 14;
inline constexpr int   kDimsPad   = 16;
inline constexpr int   kK         = 5;
inline constexpr float kQuantScale = 10000.0f;

struct alignas(32) Vec16 : std::array<int16_t, kDimsPad> {
    using std::array<int16_t, kDimsPad>::array;
};

inline int16_t quantize_f32(float v) noexcept {
    float scaled = std::roundf(v * kQuantScale);
    return static_cast<int16_t>(std::clamp(scaled, -32768.0f, 32767.0f));
}

inline int16_t quantize_f64(double v) noexcept {
    double scaled = std::round(v * static_cast<double>(kQuantScale));
    return static_cast<int16_t>(std::clamp(scaled, -32768.0, 32767.0));
}

inline void quantize_vec(std::span<const float, kDimsPad> src, Vec16& dst) noexcept {
    for (std::size_t i = 0; i < kDimsPad; ++i)
        dst[i] = quantize_f32(src[i]);
}

[[gnu::always_inline, gnu::hot]]
inline int32_t distance(const Vec16& a, const Vec16& b) noexcept {
    __m256i va   = _mm256_load_si256(reinterpret_cast<const __m256i*>(a.data()));
    __m256i vb   = _mm256_load_si256(reinterpret_cast<const __m256i*>(b.data()));
    __m256i diff = _mm256_sub_epi16(va, vb);
    __m256i prod = _mm256_madd_epi16(diff, diff);

    __m128i hi  = _mm256_extracti128_si256(prod, 1);
    __m128i lo  = _mm256_castsi256_si128(prod);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

// AABB lower bound: LB = Σ max(0, bmin[d]-q[d], q[d]-bmax[d])²
// Guarantees LB ≤ d(q, v) for any v in the box zero false pruning.
[[gnu::always_inline, gnu::hot]]
inline int32_t aabb_lower_bound(const Vec16& q, const Vec16& bmin, const Vec16& bmax) noexcept {
    __m256i vq   = _mm256_load_si256(reinterpret_cast<const __m256i*>(q.data()));
    __m256i vmin = _mm256_load_si256(reinterpret_cast<const __m256i*>(bmin.data()));
    __m256i vmax = _mm256_load_si256(reinterpret_cast<const __m256i*>(bmax.data()));
    __m256i zero = _mm256_setzero_si256();

    __m256i below = _mm256_max_epi16(_mm256_sub_epi16(vmin, vq), zero);
    __m256i above = _mm256_max_epi16(_mm256_sub_epi16(vq, vmax), zero);
    __m256i gap   = _mm256_max_epi16(below, above);
    __m256i prod  = _mm256_madd_epi16(gap, gap);

    __m128i hi  = _mm256_extracti128_si256(prod, 1);
    __m128i lo  = _mm256_castsi256_si128(prod);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

struct TopK {
    std::array<int32_t, kK>  dists;
    std::array<uint32_t, kK> idxs;

    void reset() noexcept {
        dists.fill(std::numeric_limits<int32_t>::max());
        idxs.fill(0);
    }

    [[gnu::always_inline, gnu::hot]]
    void push(int32_t d, uint32_t idx) noexcept {
        if (d >= dists.back()) return;
        int pos = kK - 1;
        while (pos > 0 && d < dists[pos - 1]) {
            dists[pos] = dists[pos - 1];
            idxs[pos]  = idxs[pos - 1];
            --pos;
        }
        dists[pos] = d;
        idxs[pos]  = idx;
    }

    [[nodiscard]] int32_t worst() const noexcept { return dists.back(); }
};

constexpr uint8_t compute_partition_key(std::span<const float, kDimsPad> f) noexcept {
    uint8_t key = 0;
    if (f[5]  >= 0.0f)  key |= 0x01;
    if (f[9]  >  0.5f)  key |= 0x02;
    if (f[10] >  0.5f)  key |= 0x04;
    if (f[11] >  0.5f)  key |= 0x08;
    int mcc_bin = std::clamp(static_cast<int>(f[12] * 4.0f), 0, 3);
    key |= static_cast<uint8_t>(mcc_bin << 4);
    if (f[0]  >  0.5f)  key |= 0x40;
    if (f[8]  >  0.25f) key |= 0x80;
    return key;
}

constexpr float clamp01(float x) noexcept {
    return std::clamp(x, 0.0f, 1.0f);
}

} // namespace rinha
