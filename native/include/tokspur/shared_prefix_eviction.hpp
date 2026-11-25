#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tokspur {

// The shared-prefix anchor victim choice, kept pure so a fixture test can pin it.
// `lru` is the only policy and is the default route's rule: the least recently used anchor
// that is not a strict prefix of the prefix being captured; when every anchor is such a
// prefix, the least recently used anchor overall. Ties keep the lowest index.
enum class SharedPrefixEvictionPolicy : uint8_t { lru };

// Why a capture had to evict: the byte cap, the anchor count cap, or both at once.
enum class SharedPrefixEvictionCause : uint8_t { bytes, count, both };

inline SharedPrefixEvictionCause shared_prefix_eviction_cause(bool over_bytes, bool over_count) {
    if (over_bytes && over_count) return SharedPrefixEvictionCause::both;
    return over_bytes ? SharedPrefixEvictionCause::bytes : SharedPrefixEvictionCause::count;
}

struct SharedPrefixEvictionCounters {
    uint64_t bytes = 0, count = 0, both = 0;
    void record(SharedPrefixEvictionCause cause) {
        switch (cause) {
        case SharedPrefixEvictionCause::bytes: ++bytes; break;
        case SharedPrefixEvictionCause::count: ++count; break;
        case SharedPrefixEvictionCause::both: ++both; break;
        }
    }
};

// `anchors` must not be empty. `tokens_of` returns a contiguous range of uint32_t token IDs
// and `tick_of` the anchor's last access tick.
template<class Anchors, class TokensOf, class TickOf>
size_t shared_prefix_victim(SharedPrefixEvictionPolicy, const Anchors& anchors, std::span<const uint32_t> prefix,
    TokensOf&& tokens_of, TickOf&& tick_of) {
    const size_t n = anchors.size();
    size_t victim = n;
    for (size_t idx = 0; idx < n; ++idx) {
        const auto& tokens = tokens_of(anchors[idx]);
        const bool is_own_prefix = tokens.size() < prefix.size() && std::equal(tokens.begin(), tokens.end(), prefix.begin());
        if (!is_own_prefix && (victim == n || tick_of(anchors[idx]) < tick_of(anchors[victim]))) victim = idx;
    }
    if (victim == n) {
        victim = 0;
        for (size_t idx = 1; idx < n; ++idx) {
            if (tick_of(anchors[idx]) < tick_of(anchors[victim])) victim = idx;
        }
    }
    return victim;
}

// ~sqrt(N)-spaced milestone schedule up to 131072 tokens:
// - <= chunk_align (2048): every boundary_tokens (512, 1024, 1536, 2048)
// - > chunk_align: chunk_align-aligned cumulative-square schedule floor(512 * k^2 / chunk_align) * chunk_align
//   (4096, 8192, 12288, 18432, 24576, 32768, 40960, 51200, 61440, 73728, 86016, 100352, 114688, 131072).
[[nodiscard]] inline bool is_sqrt_checkpoint_milestone(uint32_t position, uint32_t boundary_tokens = 512,
    uint32_t chunk_align = 2048) noexcept {
    if (!boundary_tokens || !chunk_align || position < boundary_tokens) return false;
    if (position <= chunk_align) return (position % boundary_tokens) == 0;
    if ((position % chunk_align) != 0) return false;
    for (uint64_t k = 2; k <= 64; ++k) {
        const uint64_t raw = uint64_t(boundary_tokens) * k * k;
        const uint64_t aligned = (raw / chunk_align) * chunk_align;
        if (aligned == position) return true;
        if (aligned > position) return false;
    }
    return false;
}

// Returns true when chunk [base, end_position) overlaps the trailing `window` rows (M - window, M]
// of any ~sqrt(N) milestone M > chunk_align with M <= max_position.
[[nodiscard]] inline uint32_t sqrt_milestone_window_target(uint32_t base, uint32_t end_position,
    uint32_t max_position, uint32_t window = 2048, uint32_t boundary_tokens = 512,
    uint32_t chunk_align = 2048) noexcept {
    if (!boundary_tokens || !chunk_align || !window || end_position <= base) return 0;
    for (uint64_t k = 3; k <= 64; ++k) {
        const uint64_t raw = uint64_t(boundary_tokens) * k * k;
        const uint64_t aligned = (raw / chunk_align) * chunk_align;
        if (aligned <= chunk_align) continue;
        if (aligned > max_position) break;
        const uint64_t win_begin = aligned > window ? aligned - window : 0;
        if (end_position > win_begin && base < aligned) return uint32_t(aligned);
    }
    return 0;
}

// Selects an interior victim index from sorted retained checkpoint positions 0 < p_0 < ... < p_{m-1}
// (same lineage) that minimizes the resulting gap (p_{j+1} - p_{j-1}), preserving protected system anchors
// (<= protected_anchor_max) and the newest frontier p_{m-1} whenever possible.
[[nodiscard]] inline size_t sqrt_thinning_victim(std::span<const uint32_t> sorted_positions,
    std::span<const uint64_t> access_epochs = {}, uint32_t protected_anchor_max = 1024) noexcept {
    const size_t m = sorted_positions.size();
    if (m <= 1) return 0;
    if (m == 2) {
        return (sorted_positions[0] <= protected_anchor_max && sorted_positions[1] > protected_anchor_max) ? 1 : 0;
    }
    size_t best = m;
    uint64_t best_gap = UINT64_MAX;
    uint64_t best_epoch = UINT64_MAX;
    for (size_t j = 1; j + 1 < m; ++j) {
        if (sorted_positions[j] <= protected_anchor_max) continue;
        const uint64_t gap = uint64_t(sorted_positions[j + 1]) - uint64_t(sorted_positions[j - 1]);
        const uint64_t epoch = (j < access_epochs.size()) ? access_epochs[j] : 0;
        if (best == m || gap < best_gap || (gap == best_gap && epoch < best_epoch)) {
            best = j;
            best_gap = gap;
            best_epoch = epoch;
        }
    }
    if (best != m) return best;
    // Fallback if all interior positions were <= protected_anchor_max: evict an earlier position before newest frontier.
    for (size_t j = 0; j + 1 < m; ++j) {
        if (sorted_positions[j] > protected_anchor_max) return j;
    }
    return 0;
}

// Non-milestone SSD spills (end-of-prefill coalescing and park_live_lane fallback)
// are suppressed once a >= chunk_align sqrt(N) milestone has been spilled for the
// session, preventing ~2 GiB leaf-specific checkpoints (e.g. 73,683 / 73,731) from
// out-prioritizing reusable sqrt(N) branch milestones (e.g. 61,440) under tight caps.
[[nodiscard]] inline bool should_spill_ssd_tail(
    uint32_t position, uint32_t last_spilled_prefix,
    uint32_t boundary_tokens = 512, uint32_t chunk_align = 2048) noexcept {
    if (position <= last_spilled_prefix) return false;
    if (position <= chunk_align || last_spilled_prefix < chunk_align) return true;
    return is_sqrt_checkpoint_milestone(position, boundary_tokens, chunk_align);
}

} // namespace tokspur
