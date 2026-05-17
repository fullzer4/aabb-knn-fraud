#pragma once

#include "vec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace rinha {

inline constexpr uint32_t kMagic   = 0x4B4D4B33;
inline constexpr uint32_t kVersion = 3;

struct IndexHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t total_vectors;
    uint32_t num_partitions;
    float    scale;
    uint32_t dims;
    uint32_t k;
    uint32_t num_clusters_total;
    uint32_t partition_table_offset;
    uint32_t centroids_offset;
    uint32_t bbox_min_offset;
    uint32_t bbox_max_offset;
    uint32_t cluster_meta_offset;
    uint32_t vectors_offset;
    uint32_t labels_offset;
    uint32_t part_bbox_min_offset;
    uint32_t part_bbox_max_offset;
    uint32_t padding[1];
};
static_assert(sizeof(IndexHeader) == 72);

struct PartitionEntry {
    uint8_t  partition_key;
    uint8_t  padding1;
    uint16_t num_clusters;
    uint32_t first_cluster_idx;
    uint32_t first_vector_idx;
    uint32_t num_vectors;
};
static_assert(sizeof(PartitionEntry) == 16);

struct ClusterMeta {
    uint32_t vec_offset;
    uint32_t vec_count;
};
static_assert(sizeof(ClusterMeta) == 8);

class KMKNNIndex {
public:
    KMKNNIndex() { part_lookup_.fill(0xFFFF); }

    ~KMKNNIndex() {
        if (mapped_) ::munmap(mapped_, mapped_size_);
    }

    KMKNNIndex(const KMKNNIndex&) = delete;
    KMKNNIndex& operator=(const KMKNNIndex&) = delete;

    bool load(const char* path) noexcept {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;

        struct stat st{};
        if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
        mapped_size_ = static_cast<std::size_t>(st.st_size);

        mapped_ = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        ::close(fd);
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; return false; }
        ::madvise(mapped_, mapped_size_, MADV_WILLNEED);
        ::madvise(mapped_, mapped_size_, MADV_HUGEPAGE);

        const auto* base = static_cast<const uint8_t*>(mapped_);
        header_ = reinterpret_cast<const IndexHeader*>(base);

        if (header_->magic != kMagic || header_->version != kVersion) return false;

        partitions_    = reinterpret_cast<const PartitionEntry*>(base + header_->partition_table_offset);
        centroids_     = reinterpret_cast<const Vec16*>(base + header_->centroids_offset);
        bbox_min_      = reinterpret_cast<const Vec16*>(base + header_->bbox_min_offset);
        bbox_max_      = reinterpret_cast<const Vec16*>(base + header_->bbox_max_offset);
        clusters_      = reinterpret_cast<const ClusterMeta*>(base + header_->cluster_meta_offset);
        vectors_       = reinterpret_cast<const Vec16*>(base + header_->vectors_offset);
        labels_        = base + header_->labels_offset;
        part_bbox_min_ = reinterpret_cast<const Vec16*>(base + header_->part_bbox_min_offset);
        part_bbox_max_ = reinterpret_cast<const Vec16*>(base + header_->part_bbox_max_offset);

        for (uint32_t i = 0; i < header_->num_partitions; ++i)
            part_lookup_[partitions_[i].partition_key] = static_cast<uint16_t>(i);

        std::fprintf(stderr, "index: %u vecs, %u partitions, %u clusters\n",
                     header_->total_vectors, header_->num_partitions, header_->num_clusters_total);
        return true;
    }

    [[gnu::hot]]
    int query(const Vec16& q, uint8_t partition_key) const noexcept {
        auto topk = search(q, partition_key);
        int fraud = 0;
        for (uint32_t idx : topk.idxs)
            fraud += get_label(idx) ? 1 : 0;
        return fraud;
    }

    [[gnu::hot]]
    TopK search(const Vec16& q, uint8_t partition_key) const noexcept {
        TopK topk;
        topk.reset();

        uint16_t primary = part_lookup_[partition_key];
        if (primary != 0xFFFF)
            search_partition(partitions_[primary], q, topk);

        for (uint32_t i = 0; i < header_->num_partitions; ++i) {
            if (i == primary) continue;
            if (aabb_lower_bound(q, part_bbox_min_[i], part_bbox_max_[i]) > topk.worst())
                continue;
            search_partition(partitions_[i], q, topk);
        }

        return topk;
    }

    TopK brute_force(const Vec16& q) const noexcept {
        TopK topk;
        topk.reset();
        for (uint32_t i = 0; i < header_->total_vectors; ++i)
            topk.push(distance(q, vectors_[i]), i);
        return topk;
    }

    bool get_label(uint32_t idx) const noexcept {
        return (labels_[idx >> 3] >> (idx & 7)) & 1;
    }

    uint32_t total_vectors() const noexcept { return header_->total_vectors; }

private:
    [[gnu::hot]]
    void search_partition(const PartitionEntry& part, const Vec16& q, TopK& topk) const noexcept {
        const uint32_t nc = part.num_clusters;
        const uint32_t first_ci = part.first_cluster_idx;

        struct CentDist { int32_t d; uint32_t ci; };
        std::array<CentDist, 512> cd;

        for (uint32_t i = 0; i < nc; ++i)
            cd[i] = {distance(q, centroids_[first_ci + i]), first_ci + i};

        std::sort(cd.begin(), cd.begin() + nc,
                  [](const CentDist& a, const CentDist& b) { return a.d < b.d; });

        for (uint32_t i = 0; i < nc; ++i) {
            uint32_t ci = cd[i].ci;
            if (aabb_lower_bound(q, bbox_min_[ci], bbox_max_[ci]) > topk.worst())
                continue;

            const auto& cm = clusters_[ci];
            const uint32_t base = part.first_vector_idx + cm.vec_offset;
            const Vec16* vecs = vectors_ + base;

            for (uint32_t j = 0; j < cm.vec_count; ++j) {
                if (j + 8 < cm.vec_count) __builtin_prefetch(&vecs[j + 8], 0, 1);
                topk.push(distance(q, vecs[j]), base + j);
            }
        }
    }

    void*       mapped_ = nullptr;
    std::size_t mapped_size_ = 0;

    const IndexHeader*    header_ = nullptr;
    const PartitionEntry* partitions_ = nullptr;
    const Vec16*          centroids_ = nullptr;
    const Vec16*          bbox_min_ = nullptr;
    const Vec16*          bbox_max_ = nullptr;
    const ClusterMeta*    clusters_ = nullptr;
    const Vec16*          vectors_ = nullptr;
    const uint8_t*        labels_ = nullptr;
    const Vec16*          part_bbox_min_ = nullptr;
    const Vec16*          part_bbox_max_ = nullptr;

    std::array<uint16_t, 256> part_lookup_{};
};

} // namespace rinha
