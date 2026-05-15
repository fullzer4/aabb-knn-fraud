#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace rinha {

static constexpr int kDims    = 14;
static constexpr int kDimsPad = 16; // padded to 16 for AVX2 alignment (2 × 8)
static constexpr int kK       = 5;

using Vec16  = uint16_t[kDimsPad]; // float16 storage
using Vec32  = float[kDimsPad];    // float32 working

inline float f16_to_f32(uint16_t h) {
    __m128i v = _mm_set1_epi16(static_cast<short>(h));
    __m128  f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

inline uint16_t f32_to_f16(float f) {
    __m128  v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

inline void vec_f32_to_f16(const float* src, uint16_t* dst) {
    __m256 lo = _mm256_load_ps(src);
    __m256 hi = _mm256_load_ps(src + 8);
    __m128i lo16 = _mm256_cvtps_ph(lo, _MM_FROUND_TO_NEAREST_INT);
    __m128i hi16 = _mm256_cvtps_ph(hi, _MM_FROUND_TO_NEAREST_INT);
    _mm_store_si128(reinterpret_cast<__m128i*>(dst),     lo16);
    _mm_store_si128(reinterpret_cast<__m128i*>(dst + 8), hi16);
}

[[gnu::always_inline, gnu::hot]]
inline float distance_f16(const uint16_t* __restrict__ ref, const float* __restrict__ query) {
    __m256 r0 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(ref)));
    __m256 r1 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(ref + 8)));

    __m256 q0 = _mm256_load_ps(query);
    __m256 q1 = _mm256_load_ps(query + 8);

    __m256 d0 = _mm256_sub_ps(r0, q0);
    __m256 d1 = _mm256_sub_ps(r1, q1);

    __m256 acc = _mm256_fmadd_ps(d0, d0, _mm256_mul_ps(d1, d1));

    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);          // 4 floats
    sum = _mm_hadd_ps(sum, sum);              // 2 floats
    sum = _mm_hadd_ps(sum, sum);              // 1 float
    return _mm_cvtss_f32(sum);
}

[[gnu::always_inline, gnu::hot]]
inline void distance_f16_x4(const uint16_t* __restrict__ vecs,
                            const float* __restrict__ query,
                            float* __restrict__ out) {
    __m256 q0 = _mm256_load_ps(query);
    __m256 q1 = _mm256_load_ps(query + 8);

    __m256 r0_0 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs)));
    __m256 r0_1 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + 8)));
    __m256 r1_0 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + kDimsPad)));
    __m256 r1_1 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + kDimsPad + 8)));
    __m256 r2_0 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + 2*kDimsPad)));
    __m256 r2_1 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + 2*kDimsPad + 8)));
    __m256 r3_0 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + 3*kDimsPad)));
    __m256 r3_1 = _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(vecs + 3*kDimsPad + 8)));

    __m256 d0, d1, acc;

    d0 = _mm256_sub_ps(r0_0, q0); d1 = _mm256_sub_ps(r0_1, q1);
    acc = _mm256_fmadd_ps(d0, d0, _mm256_mul_ps(d1, d1));
    __m128 h = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), h);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    out[0] = _mm_cvtss_f32(s);

    d0 = _mm256_sub_ps(r1_0, q0); d1 = _mm256_sub_ps(r1_1, q1);
    acc = _mm256_fmadd_ps(d0, d0, _mm256_mul_ps(d1, d1));
    h = _mm256_extractf128_ps(acc, 1);
    s = _mm_add_ps(_mm256_castps256_ps128(acc), h);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    out[1] = _mm_cvtss_f32(s);

    d0 = _mm256_sub_ps(r2_0, q0); d1 = _mm256_sub_ps(r2_1, q1);
    acc = _mm256_fmadd_ps(d0, d0, _mm256_mul_ps(d1, d1));
    h = _mm256_extractf128_ps(acc, 1);
    s = _mm_add_ps(_mm256_castps256_ps128(acc), h);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    out[2] = _mm_cvtss_f32(s);

    d0 = _mm256_sub_ps(r3_0, q0); d1 = _mm256_sub_ps(r3_1, q1);
    acc = _mm256_fmadd_ps(d0, d0, _mm256_mul_ps(d1, d1));
    h = _mm256_extractf128_ps(acc, 1);
    s = _mm_add_ps(_mm256_castps256_ps128(acc), h);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    out[3] = _mm_cvtss_f32(s);
}

[[gnu::always_inline]]
inline float distance_f32(const float* ref, const float* query) {
    __m256 r0 = _mm256_load_ps(ref);
    __m256 r1 = _mm256_load_ps(ref + 8);

    __m256 q0 = _mm256_load_ps(query);
    __m256 q1 = _mm256_load_ps(query + 8);

    __m256 d0 = _mm256_sub_ps(r0, q0);
    __m256 d1 = _mm256_sub_ps(r1, q1);

    __m256 acc = _mm256_fmadd_ps(d0, d0, _mm256_mul_ps(d1, d1));

    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

struct TopK {
    float    dists[kK];
    uint32_t idxs[kK];

    void reset() {
        for (int i = 0; i < kK; ++i) {
            dists[i] = __FLT_MAX__;
            idxs[i]  = 0;
        }
    }

    [[gnu::always_inline, gnu::hot]]
    void push(float d, uint32_t idx) {
        if (d >= dists[kK - 1]) return;  // fast reject
        int pos = kK - 1;
        while (pos > 0 && d < dists[pos - 1]) {
            dists[pos] = dists[pos - 1];
            idxs[pos]  = idxs[pos - 1];
            --pos;
        }
        dists[pos] = d;
        idxs[pos]  = idx;
    }

    [[gnu::always_inline]]
    float threshold() const { return dists[kK - 1]; }
};

inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

} // namespace rinha
