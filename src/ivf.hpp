#pragma once

#include "dataset.hpp"
#include "vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace rinha {

struct KNN {
    static constexpr int kMaxIter    = 25;
    static constexpr int kSampleSize = 80000;

    int nclusters = 1024;
    int nprobe    = 32;

    // Centroids: nclusters × kDimsPad, float32, 32-byte aligned
    float*    centroids       = nullptr;
    uint32_t* cluster_offsets = nullptr; // nclusters + 1

    uint16_t* vectors = nullptr;
    uint8_t*  labels  = nullptr;
    uint32_t  total   = 0;

    ~KNN() {
        std::free(centroids);
        std::free(cluster_offsets);
        std::free(vectors);
        std::free(labels);
    }

    void build(Dataset& ds, int nclusters_, int nprobe_) {
        total     = ds.count;
        nclusters = nclusters_;
        nprobe    = nprobe_;

        std::fprintf(stderr, "building index: %d clusters, nprobe=%d, %u vectors\n",
                     nclusters, nprobe, total);

        std::size_t cent_bytes = static_cast<std::size_t>(nclusters) * kDimsPad * sizeof(float);
        centroids = static_cast<float*>(std::aligned_alloc(32, (cent_bytes + 31) & ~31u));
        std::memset(centroids, 0, (cent_bytes + 31) & ~31u);

        cluster_offsets = static_cast<uint32_t*>(std::calloc(nclusters + 1, sizeof(uint32_t)));

        alignas(32) float tmp[kDimsPad];

        // ── K-means++ initialization ─────────────────────────
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist(0, total - 1);

        {
            const uint16_t* v = ds.vectors + static_cast<std::size_t>(dist(rng)) * kDimsPad;
            float* cent = centroids;
            for (int d = 0; d < kDimsPad; ++d)
                cent[d] = f16_to_f32(v[d]);
        }

        uint32_t init_sample_n = std::min(static_cast<uint32_t>(20000), total);
        auto* init_sample = static_cast<uint32_t*>(std::malloc(init_sample_n * sizeof(uint32_t)));
        for (uint32_t i = 0; i < init_sample_n; ++i)
            init_sample[i] = dist(rng);

        auto* min_dists = static_cast<float*>(std::malloc(init_sample_n * sizeof(float)));
        for (uint32_t i = 0; i < init_sample_n; ++i)
            min_dists[i] = __FLT_MAX__;

        for (int c = 1; c < nclusters; ++c) {
            float* prev_cent = centroids + static_cast<std::size_t>(c - 1) * kDimsPad;
            double sum_d = 0.0;
            for (uint32_t i = 0; i < init_sample_n; ++i) {
                const uint16_t* v = ds.vectors + static_cast<std::size_t>(init_sample[i]) * kDimsPad;
                for (int d = 0; d < kDimsPad; ++d)
                    tmp[d] = f16_to_f32(v[d]);
                float dd = distance_f32(prev_cent, tmp);
                if (dd < min_dists[i]) min_dists[i] = dd;
                sum_d += static_cast<double>(min_dists[i]);
            }

            std::uniform_real_distribution<double> udist(0.0, sum_d);
            double target = udist(rng);
            double cumul = 0.0;
            uint32_t chosen = init_sample[0];
            for (uint32_t i = 0; i < init_sample_n; ++i) {
                cumul += static_cast<double>(min_dists[i]);
                if (cumul >= target) { chosen = init_sample[i]; break; }
            }

            const uint16_t* v = ds.vectors + static_cast<std::size_t>(chosen) * kDimsPad;
            float* cent = centroids + static_cast<std::size_t>(c) * kDimsPad;
            for (int d = 0; d < kDimsPad; ++d)
                cent[d] = f16_to_f32(v[d]);
        }
        std::free(init_sample);
        std::free(min_dists);

        // ── K-means iterations on subsample ──────────────────
        uint32_t sample_n = std::min(static_cast<uint32_t>(kSampleSize), total);
        auto* sample_idx = static_cast<uint32_t*>(std::malloc(sample_n * sizeof(uint32_t)));
        for (uint32_t i = 0; i < sample_n; ++i)
            sample_idx[i] = dist(rng);

        std::size_t sums_bytes = static_cast<std::size_t>(nclusters) * kDimsPad * sizeof(float);
        auto* sums = static_cast<float*>(std::calloc(1, sums_bytes));
        auto* counts = static_cast<uint32_t*>(std::calloc(nclusters, sizeof(uint32_t)));

        for (int iter = 0; iter < kMaxIter; ++iter) {
            std::memset(sums, 0, sums_bytes);
            std::memset(counts, 0, nclusters * sizeof(uint32_t));

            for (uint32_t si = 0; si < sample_n; ++si) {
                uint32_t idx = sample_idx[si];
                const uint16_t* v = ds.vectors + static_cast<std::size_t>(idx) * kDimsPad;
                for (int d = 0; d < kDimsPad; ++d)
                    tmp[d] = f16_to_f32(v[d]);

                float best_d = __FLT_MAX__;
                int   best_c = 0;
                for (int c = 0; c < nclusters; ++c) {
                    float d = distance_f32(centroids + static_cast<std::size_t>(c) * kDimsPad, tmp);
                    if (d < best_d) { best_d = d; best_c = c; }
                }

                counts[best_c]++;
                float* s = sums + static_cast<std::size_t>(best_c) * kDimsPad;
                for (int d = 0; d < kDimsPad; ++d)
                    s[d] += tmp[d];
            }

            bool changed = false;
            for (int c = 0; c < nclusters; ++c) {
                if (counts[c] == 0) continue;
                float inv = 1.0f / static_cast<float>(counts[c]);
                float* cent = centroids + static_cast<std::size_t>(c) * kDimsPad;
                for (int d = 0; d < kDimsPad; ++d) {
                    float nv = sums[static_cast<std::size_t>(c) * kDimsPad + d] * inv;
                    if (std::abs(nv - cent[d]) > 1e-6f) changed = true;
                    cent[d] = nv;
                }
            }
            if (!changed) {
                std::fprintf(stderr, "k-means converged at iter %d\n", iter);
                break;
            }
        }
        std::free(sample_idx);
        std::free(sums);
        std::free(counts);

        // ── Assign ALL vectors to nearest centroid ────────────
        auto* assignments = static_cast<uint32_t*>(std::malloc(total * sizeof(uint32_t)));

        for (uint32_t i = 0; i < total; ++i) {
            const uint16_t* v = ds.vectors + static_cast<std::size_t>(i) * kDimsPad;
            for (int d = 0; d < kDimsPad; ++d)
                tmp[d] = f16_to_f32(v[d]);

            float best_d = __FLT_MAX__;
            int   best_c = 0;
            for (int c = 0; c < nclusters; ++c) {
                float d = distance_f32(centroids + static_cast<std::size_t>(c) * kDimsPad, tmp);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assignments[i] = static_cast<uint32_t>(best_c);
        }

        // ── Compute cluster offsets ───────────────────────────
        auto* cluster_counts = static_cast<uint32_t*>(std::calloc(nclusters, sizeof(uint32_t)));
        for (uint32_t i = 0; i < total; ++i)
            cluster_counts[assignments[i]]++;

        cluster_offsets[0] = 0;
        for (int c = 0; c < nclusters; ++c)
            cluster_offsets[c + 1] = cluster_offsets[c] + cluster_counts[c];

        // ── In-place reorder via cycle-following ──────────────
        auto* new_pos = static_cast<uint32_t*>(std::malloc(total * sizeof(uint32_t)));
        uint32_t* cursor = cluster_counts; // reuse
        std::memcpy(cursor, cluster_offsets, nclusters * sizeof(uint32_t));
        for (uint32_t i = 0; i < total; ++i)
            new_pos[i] = cursor[assignments[i]]++;

        std::free(assignments);

        auto* labels_byte = static_cast<uint8_t*>(std::malloc(total));
        for (uint32_t i = 0; i < total; ++i)
            labels_byte[i] = ds.is_fraud(i) ? 1 : 0;

        auto* visited = static_cast<uint8_t*>(std::calloc((total + 7) / 8, 1));
        alignas(32) uint16_t vec_tmp[kDimsPad];
        constexpr std::size_t vec_stride = kDimsPad * sizeof(uint16_t);

        for (uint32_t i = 0; i < total; ++i) {
            if ((visited[i >> 3] >> (i & 7)) & 1) continue;
            if (new_pos[i] == i) {
                visited[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
                continue;
            }
            std::memcpy(vec_tmp, ds.vectors + static_cast<std::size_t>(i) * kDimsPad, vec_stride);
            uint8_t lbl_tmp = labels_byte[i];
            uint32_t j = i;
            for (;;) {
                visited[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
                uint32_t k = new_pos[j];
                if (k == i) {
                    std::memcpy(ds.vectors + static_cast<std::size_t>(j) * kDimsPad, vec_tmp, vec_stride);
                    labels_byte[j] = lbl_tmp;
                    break;
                }
                std::memcpy(
                    ds.vectors + static_cast<std::size_t>(j) * kDimsPad,
                    ds.vectors + static_cast<std::size_t>(k) * kDimsPad,
                    vec_stride
                );
                labels_byte[j] = labels_byte[k];
                j = k;
            }
        }

        std::free(visited);
        std::free(new_pos);
        std::free(cluster_counts);

        std::size_t label_bytes = (total + 7) / 8;
        std::memset(ds.labels, 0, label_bytes);
        for (uint32_t i = 0; i < total; ++i)
            if (labels_byte[i])
                ds.labels[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
        std::free(labels_byte);

        vectors = ds.vectors;
        labels  = ds.labels;
        ds.vectors = nullptr;
        ds.labels  = nullptr;

        uint32_t min_sz = total, max_sz = 0;
        for (int c = 0; c < nclusters; ++c) {
            uint32_t sz = cluster_offsets[c + 1] - cluster_offsets[c];
            if (sz < min_sz) min_sz = sz;
            if (sz > max_sz) max_sz = sz;
        }
        std::fprintf(stderr, "index built: min=%u, max=%u, avg=%u\n",
                     min_sz, max_sz, total / static_cast<uint32_t>(nclusters));
    }

    [[gnu::hot]]
    int query(const float* __restrict__ q) const {
        // ── Phase 1: Find nprobe nearest centroids ───────────
        struct CentDist { float d; int c; };
        CentDist best[64]; // max nprobe
        int np = nprobe < 64 ? nprobe : 64;
        for (int i = 0; i < np; ++i)
            best[i] = {__FLT_MAX__, 0};

        for (int c = 0; c < nclusters; ++c) {
            const float* cent = centroids + static_cast<std::size_t>(c) * kDimsPad;
            if (c + 1 < nclusters)
                __builtin_prefetch(centroids + static_cast<std::size_t>(c + 1) * kDimsPad, 0, 1);

            float d = distance_f32(cent, q);
            if (d < best[np - 1].d) {
                int pos = np - 1;
                while (pos > 0 && d < best[pos - 1].d) {
                    best[pos] = best[pos - 1];
                    --pos;
                }
                best[pos] = {d, c};
            }
        }

        // ── Phase 2: Scan selected clusters with prefetch + unrolling ─
        TopK topk;
        topk.reset();

        for (int p = 0; p < np; ++p) {
            int      c     = best[p].c;
            uint32_t start = cluster_offsets[c];
            uint32_t end   = cluster_offsets[c + 1];
            uint32_t count = end - start;

            const uint16_t* vecs = vectors + static_cast<std::size_t>(start) * kDimsPad;

            __builtin_prefetch(vecs, 0, 0);
            __builtin_prefetch(vecs + 32, 0, 0);

            uint32_t i = start;
            uint32_t end4 = start + (count & ~3u);

            for (; i < end4; i += 4) {
                __builtin_prefetch(vecs + 4 * kDimsPad, 0, 0);
                __builtin_prefetch(vecs + 6 * kDimsPad, 0, 0);

                float dists4[4];
                distance_f16_x4(vecs, q, dists4);

                topk.push(dists4[0], i);
                topk.push(dists4[1], i + 1);
                topk.push(dists4[2], i + 2);
                topk.push(dists4[3], i + 3);

                vecs += 4 * kDimsPad;
            }

            for (; i < end; ++i) {
                float d = distance_f16(vecs, q);
                topk.push(d, i);
                vecs += kDimsPad;
            }
        }

        // ── Phase 3: Count fraud among K nearest ─────────────
        int fraud_count = 0;
        for (int k = 0; k < kK; ++k) {
            uint32_t idx = topk.idxs[k];
            if ((labels[idx >> 3] >> (idx & 7)) & 1)
                ++fraud_count;
        }
        return fraud_count;
    }
};

} // namespace rinha
