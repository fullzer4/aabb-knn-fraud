#include "dataset.hpp"
#include "index.hpp"
#include "vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <unistd.h>
#include <vector>

using namespace rinha;

struct VecData {
    Vec16   vec{};
    uint8_t partition_key{};
    bool    is_fraud{};
};

struct BuiltCluster {
    alignas(32) std::array<float, kDimsPad> centroid_f32{};
    Vec16 centroid_i16{};
    Vec16 bbox_min{};
    Vec16 bbox_max{};
    std::vector<uint32_t> members;
};

static void build_clusters(std::span<const uint32_t> indices,
                           const VecData* vecs,
                           int k,
                           std::vector<BuiltCluster>& out) {
    const auto n = static_cast<uint32_t>(indices.size());
    if (n == 0) return;
    k = std::clamp(k, 1, static_cast<int>(n));
    out.resize(static_cast<std::size_t>(k));

    std::mt19937 rng(42u + n);
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);

    // K-means initialization
    {
        const auto& v = vecs[indices[pick(rng)]].vec;
        for (int d = 0; d < kDimsPad; ++d)
            out[0].centroid_f32[d] = static_cast<float>(v[d]);
    }

    std::vector<float> min_d(n, std::numeric_limits<float>::max());
    Vec16 tmp{};

    for (std::size_t c = 1; c < static_cast<std::size_t>(k); ++c) {
        auto& prev = out[c - 1].centroid_f32;
        for (int d = 0; d < kDimsPad; ++d)
            tmp[d] = static_cast<int16_t>(prev[d]);

        double sum = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            float dd = static_cast<float>(distance(tmp, vecs[indices[i]].vec));
            min_d[i] = std::min(min_d[i], dd);
            sum += min_d[i];
        }

        std::uniform_real_distribution<double> ud(0.0, sum);
        double target = ud(rng), cumul = 0.0;
        uint32_t chosen = 0;
        for (uint32_t i = 0; i < n; ++i) {
            cumul += min_d[i];
            if (cumul >= target) { chosen = i; break; }
        }

        const auto& v = vecs[indices[chosen]].vec;
        for (int d = 0; d < kDimsPad; ++d)
            out[c].centroid_f32[d] = static_cast<float>(v[d]);
    }

    // Lloyd iterations
    std::vector<uint32_t> assign(n);
    std::vector<double> sums(static_cast<std::size_t>(k) * kDimsPad);
    std::vector<uint32_t> counts(static_cast<std::size_t>(k));
    std::vector<Vec16> cent_i16(static_cast<std::size_t>(k));

    for (int iter = 0; iter < 15; ++iter) {
        for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c)
            for (int d = 0; d < kDimsPad; ++d)
                cent_i16[c][d] = static_cast<int16_t>(out[c].centroid_f32[d]);

        std::ranges::fill(sums, 0.0);
        std::ranges::fill(counts, 0u);

        for (uint32_t i = 0; i < n; ++i) {
            const auto& v = vecs[indices[i]].vec;
            int32_t best_d = std::numeric_limits<int32_t>::max();
            std::size_t best_c = 0;
            for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
                int32_t dd = distance(cent_i16[c], v);
                if (dd < best_d) { best_d = dd; best_c = c; }
            }
            assign[i] = static_cast<uint32_t>(best_c);
            counts[best_c]++;
            double* s = sums.data() + best_c * kDimsPad;
            for (int d = 0; d < kDimsPad; ++d)
                s[d] += static_cast<double>(v[d]);
        }

        bool changed = false;
        for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
            if (counts[c] == 0) continue;
            double inv = 1.0 / counts[c];
            auto& cent = out[c].centroid_f32;
            double* s = sums.data() + c * kDimsPad;
            for (int d = 0; d < kDimsPad; ++d) {
                float nv = static_cast<float>(s[d] * inv);
                if (std::abs(nv - cent[d]) > 0.5f) changed = true;
                cent[d] = nv;
            }
        }
        if (!changed) break;
    }

    // Final assignment + AABB bounding boxes
    for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
        auto& cl = out[c];
        cl.members.clear();
        for (int d = 0; d < kDimsPad; ++d)
            cl.centroid_i16[d] = static_cast<int16_t>(cl.centroid_f32[d]);
        cl.bbox_min.fill(std::numeric_limits<int16_t>::max());
        cl.bbox_max.fill(std::numeric_limits<int16_t>::min());
    }

    for (uint32_t i = 0; i < n; ++i) {
        auto& cl = out[assign[i]];
        cl.members.push_back(i);
        const auto& v = vecs[indices[i]].vec;
        for (int d = 0; d < kDimsPad; ++d) {
            cl.bbox_min[d] = std::min(cl.bbox_min[d], v[d]);
            cl.bbox_max[d] = std::max(cl.bbox_max[d], v[d]);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <references.json.gz> <output.bin>\n", argv[0]);
        return 1;
    }

    std::fprintf(stderr, "loading %s...\n", argv[1]);
    auto dataset = Dataset::load(argv[1]);
    const uint32_t total = dataset.count;
    std::fprintf(stderr, "loaded %u vectors\n", total);

    auto vecs = std::make_unique<VecData[]>(total);
    {
        alignas(32) std::array<float, kDimsPad> ftmp{};
        for (uint32_t i = 0; i < total; ++i) {
            const double* src = dataset.vectors + std::size_t(i) * kDimsPad;
            for (int d = 0; d < kDimsPad; ++d)
                vecs[i].vec[d] = quantize_f64(src[d]);
            for (int d = 0; d < kDimsPad; ++d)
                ftmp[d] = static_cast<float>(src[d]);
            vecs[i].partition_key = compute_partition_key(ftmp);
            vecs[i].is_fraud = dataset.is_fraud(i);
        }
    }

    // Group by partition key
    struct PartGroup { uint8_t key; std::vector<uint32_t> indices; };
    std::vector<PartGroup> parts;
    std::array<uint32_t, 256> key_map;
    key_map.fill(UINT32_MAX);

    for (uint32_t i = 0; i < total; ++i) {
        uint8_t k = vecs[i].partition_key;
        if (key_map[k] == UINT32_MAX) {
            key_map[k] = static_cast<uint32_t>(parts.size());
            parts.push_back({k, {}});
        }
        parts[key_map[k]].indices.push_back(i);
    }

    std::ranges::sort(parts, [](const PartGroup& a, const PartGroup& b) {
        return a.indices.size() > b.indices.size();
    });
    std::fprintf(stderr, "%zu partitions\n", parts.size());

    // Build clusters per partition
    struct PartBuild { uint8_t key; std::vector<BuiltCluster> clusters; };
    std::vector<PartBuild> built(parts.size());

    for (std::size_t pi = 0; pi < parts.size(); ++pi) {
        auto& pg = parts[pi];
        built[pi].key = pg.key;
        int nc = std::clamp(static_cast<int>(std::sqrt(pg.indices.size())), 1, 256);
        build_clusters(pg.indices, vecs.get(), nc, built[pi].clusters);
        if ((pi & 0xF) == 0)
            std::fprintf(stderr, "  %zu/%zu key=%u n=%zu k=%d\n",
                         pi, parts.size(), pg.key, pg.indices.size(), nc);
    }

    // Compute binary layout
    uint32_t num_parts = static_cast<uint32_t>(built.size());
    uint32_t num_clusters = 0;
    for (auto& b : built) num_clusters += static_cast<uint32_t>(b.clusters.size());

    uint32_t labels_bytes = (total + 7) / 8;
    uint32_t pt_off    = sizeof(IndexHeader);
    uint32_t cent_off  = (pt_off + num_parts * sizeof(PartitionEntry) + 31) & ~31u;
    uint32_t bmin_off  = cent_off + num_clusters * sizeof(Vec16);
    uint32_t bmax_off  = bmin_off + num_clusters * sizeof(Vec16);
    uint32_t cm_off    = bmax_off + num_clusters * sizeof(Vec16);
    uint32_t vec_off   = (cm_off + num_clusters * sizeof(ClusterMeta) + 31) & ~31u;
    uint32_t lbl_off   = vec_off + total * sizeof(Vec16);
    uint32_t pbmin_off = (lbl_off + labels_bytes + 31) & ~31u;
    uint32_t pbmax_off = pbmin_off + num_parts * sizeof(Vec16);
    uint32_t file_sz   = (pbmax_off + num_parts * sizeof(Vec16) + 31) & ~31u;

    std::fprintf(stderr, "index v3: %u bytes (%.1f MB)\n", file_sz, file_sz / 1e6);

    auto* buf = static_cast<uint8_t*>(std::aligned_alloc(32, (file_sz + 31) & ~31u));
    std::memset(buf, 0, file_sz);

    auto* hdr = reinterpret_cast<IndexHeader*>(buf);
    *hdr = IndexHeader{
        .magic = kMagic, .version = kVersion,
        .total_vectors = total, .num_partitions = num_parts,
        .scale = kQuantScale, .dims = kDimsPad, .k = kK,
        .num_clusters_total = num_clusters,
        .partition_table_offset = pt_off,
        .centroids_offset = cent_off,
        .bbox_min_offset = bmin_off,
        .bbox_max_offset = bmax_off,
        .cluster_meta_offset = cm_off,
        .vectors_offset = vec_off,
        .labels_offset = lbl_off,
        .part_bbox_min_offset = pbmin_off,
        .part_bbox_max_offset = pbmax_off,
        .padding = {}
    };

    auto* pt     = reinterpret_cast<PartitionEntry*>(buf + pt_off);
    auto* cent   = reinterpret_cast<Vec16*>(buf + cent_off);
    auto* bmin   = reinterpret_cast<Vec16*>(buf + bmin_off);
    auto* bmax   = reinterpret_cast<Vec16*>(buf + bmax_off);
    auto* cm     = reinterpret_cast<ClusterMeta*>(buf + cm_off);
    auto* ov     = reinterpret_cast<Vec16*>(buf + vec_off);
    auto* labels = buf + lbl_off;
    auto* pbmin  = reinterpret_cast<Vec16*>(buf + pbmin_off);
    auto* pbmax  = reinterpret_cast<Vec16*>(buf + pbmax_off);

    uint32_t ci = 0, vi = 0;
    for (uint32_t pi = 0; pi < num_parts; ++pi) {
        auto& b = built[pi];
        auto& pg = parts[pi];

        pt[pi] = PartitionEntry{
            .partition_key = b.key, .padding1 = 0,
            .num_clusters = static_cast<uint16_t>(b.clusters.size()),
            .first_cluster_idx = ci,
            .first_vector_idx = vi,
            .num_vectors = 0
        };

        Vec16 part_bmin{}, part_bmax{};
        part_bmin.fill(std::numeric_limits<int16_t>::max());
        part_bmax.fill(std::numeric_limits<int16_t>::min());

        uint32_t pvc = 0, lvo = 0;
        for (auto& cl : b.clusters) {
            cent[ci] = cl.centroid_i16;
            bmin[ci] = cl.bbox_min;
            bmax[ci] = cl.bbox_max;

            for (int d = 0; d < kDimsPad; ++d) {
                part_bmin[d] = std::min(part_bmin[d], cl.bbox_min[d]);
                part_bmax[d] = std::max(part_bmax[d], cl.bbox_max[d]);
            }

            cm[ci] = ClusterMeta{.vec_offset = lvo, .vec_count = static_cast<uint32_t>(cl.members.size())};

            for (uint32_t mi = 0; mi < cm[ci].vec_count; ++mi) {
                uint32_t gi = pg.indices[cl.members[mi]];
                ov[vi] = vecs[gi].vec;
                if (vecs[gi].is_fraud)
                    labels[vi >> 3] |= static_cast<uint8_t>(1u << (vi & 7));
                ++vi;
            }
            lvo += cm[ci].vec_count;
            pvc += cm[ci].vec_count;
            ++ci;
        }
        pt[pi].num_vectors = pvc;
        pbmin[pi] = part_bmin;
        pbmax[pi] = part_bmax;
    }

    std::fprintf(stderr, "wrote %u vecs, %u clusters\n", vi, ci);

    int fd = ::open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { std::perror("open"); std::free(buf); return 1; }
    for (std::size_t w = 0; w < file_sz; ) {
        auto n = ::write(fd, buf + w, file_sz - w);
        if (n <= 0) { std::perror("write"); ::close(fd); std::free(buf); return 1; }
        w += static_cast<std::size_t>(n);
    }
    ::close(fd);
    std::free(buf);

    std::fprintf(stderr, "done: %s\n", argv[2]);
    return 0;
}
