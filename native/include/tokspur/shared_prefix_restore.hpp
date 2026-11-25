#pragma once

#include "tokspur/gguf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tokspur {

// Maximum exact common prefix length (capped at max_prefix_tokens) shared with any recent session, or 0 when below boundary_tokens.
template<class RecentPrefixes>
[[nodiscard]] inline size_t shared_prefix_best_lcp(const RecentPrefixes& recent_prefixes, std::span<const uint32_t> prefix,
    size_t boundary_tokens, size_t max_prefix_tokens, std::span<const uint32_t> continuation = {}) noexcept {
    const size_t split = std::min(prefix.size(), max_prefix_tokens);
    const size_t limit = split + std::min(continuation.size(), max_prefix_tokens - split);
    if (!boundary_tokens || limit < boundary_tokens) return 0;
    size_t best = 0;
    for (const auto& prev : recent_prefixes) {
        const size_t bound = std::min(limit, prev.size());
        if (bound <= best || bound < boundary_tokens) continue;
        const size_t first = std::min(split, bound);
        size_t lcp = 0;
        while (lcp < first && prev[lcp] == prefix[lcp]) ++lcp;
        if (lcp == split) {
            while (lcp < bound && prev[lcp] == continuation[lcp - split]) ++lcp;
        }
        best = std::max(best, lcp);
        if (best == limit) break;
    }
    return best >= boundary_tokens ? best : 0;
}

// Clamps a prefill slice of length `chunk_count` starting at `abs_pos` so wide NAX prefill slices (up to 2,048 rows) do not step over a shared-prefix boundary eligible for capture.
[[nodiscard]] inline size_t clamp_chunk_to_shared_prefix_boundary(size_t chunk_count, size_t abs_pos, size_t lcp_boundary,
    size_t boundary_tokens) noexcept {
    if (!chunk_count || !boundary_tokens || lcp_boundary <= abs_pos) return chunk_count;
    const size_t target = (lcp_boundary / boundary_tokens) * boundary_tokens;
    if (target <= abs_pos || abs_pos + chunk_count <= target) return chunk_count;
    return target - abs_pos;
}

// A failed rollback is fatal: the target must not continue from an uncertain target/drafter pair.
class SharedPrefixRollbackError final : public std::runtime_error {
public:
    SharedPrefixRollbackError() : std::runtime_error("shared prefix rollback failed; target state is invalid") {}
};

// Snapshot the actual checkpoint interfaces before either write. A false result permits cold prefill only after both prior states have been restored.
template<class Stack, class Mtp = std::nullptr_t>
bool restore_shared_prefix_checkpoint(Stack& stack, uint32_t target_prefix, std::span<const std::byte> stack_payload,
    Mtp* mtp = nullptr, uint32_t target_mtp_first = 1, uint32_t target_mtp_rows = 0, std::span<const std::byte> mtp_payload = {}) {
    if (stack.ssd_checkpoint_payload_bytes(target_prefix) != stack_payload.size()) return false;
    if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
        if ((mtp ? mtp->ssd_cache_bytes(target_mtp_rows) : 0) != mtp_payload.size()) return false;
    } else if (!mtp_payload.empty()) return false;

    const uint32_t previous_prefix = stack.position();
    std::vector<std::byte> previous_stack;
    if (previous_prefix) {
        previous_stack.resize(size_t(stack.ssd_checkpoint_payload_bytes(previous_prefix)));
        if (!stack.read_ssd_checkpoint_span(0, previous_stack)) return false;
    }
    [[maybe_unused]] uint32_t previous_mtp_first = 1, previous_mtp_rows = 0;
    [[maybe_unused]] std::vector<std::byte> previous_mtp;
    if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
        if (mtp) {
            previous_mtp_first = mtp->first_absolute_position();
            previous_mtp_rows = mtp->cache_rows();
            if (previous_mtp_rows) {
                previous_mtp.resize(size_t(mtp->ssd_cache_bytes(previous_mtp_rows)));
                if (!mtp->read_ssd_cache_span(0, previous_mtp)) return false;
            }
        }
    }
    const auto rollback = [&]() {
        bool restored = true;
        try {
            if (!previous_prefix) stack.reset();
            else restored = stack.write_ssd_checkpoint_span(previous_prefix, 0, previous_stack);
        } catch (...) { restored = false; }
        if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
            if (mtp) {
                try {
                    if (!previous_mtp_rows) mtp->reset(previous_mtp_first);
                    else if (!mtp->write_ssd_cache_span(previous_mtp_first, previous_mtp_rows, 0, previous_mtp)) restored = false;
                } catch (...) { restored = false; }
            }
        }
        if (!restored) throw SharedPrefixRollbackError();
    };
    bool restored = false;
    try {
        restored = stack.write_ssd_checkpoint_span(target_prefix, 0, stack_payload);
        if constexpr (!std::is_same_v<Mtp, std::nullptr_t>) {
            if (restored && mtp) restored = mtp->write_ssd_cache_span(target_mtp_first, target_mtp_rows, 0, mtp_payload);
        }
    } catch (...) {
        const auto failure = std::current_exception();
        rollback();
        std::rethrow_exception(failure);
    }
    if (!restored) rollback();
    return restored;
}

} // namespace tokspur
