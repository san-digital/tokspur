#include "tokspur/shared_prefix_eviction.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
size_t checks = 0;
void require(bool b, const char* m) { ++checks; if (!b) throw std::runtime_error(m); }

struct Anchor { std::vector<uint32_t> tokens; uint64_t last_access_tick = 0; };

// Oracle: the victim loop as it stood inline in native/src/target.cpp at 33cfd20 (lines 6370-6387),
// copied verbatim apart from the container name.
size_t inline_loop_victim(const std::vector<Anchor>& anchors, const std::vector<uint32_t>& prefix_tokens) {
    size_t victim = anchors.size();
    for (size_t idx = 0; idx < anchors.size(); ++idx) {
        const auto& item = anchors[idx];
        const bool is_own_prefix = item.tokens.size() < prefix_tokens.size() &&
            std::equal(item.tokens.begin(), item.tokens.end(), prefix_tokens.begin());
        if (!is_own_prefix && (victim == anchors.size() ||
            item.last_access_tick < anchors[victim].last_access_tick)) {
            victim = idx;
        }
    }
    if (victim == anchors.size()) {
        victim = 0;
        for (size_t idx = 1; idx < anchors.size(); ++idx) {
            if (anchors[idx].last_access_tick < anchors[victim].last_access_tick) {
                victim = idx;
            }
        }
    }
    return victim;
}

size_t header_victim(const std::vector<Anchor>& anchors, const std::vector<uint32_t>& prefix) {
    return tokspur::shared_prefix_victim(tokspur::SharedPrefixEvictionPolicy::lru, anchors, prefix,
        [](const Anchor& a) -> const std::vector<uint32_t>& { return a.tokens; },
        [](const Anchor& a) { return a.last_access_tick; });
}
}

int main() {
    try {
        using namespace tokspur;
        // Hand cases: own prefixes are skipped, all-own falls back to oldest, ties keep the lowest index.
        const std::vector<uint32_t> prefix{1, 2, 3, 4, 5, 6};
        require(header_victim({{{1, 2}, 1}, {{9}, 5}, {{1, 2, 3}, 0}}, prefix) == 1, "own prefixes are kept");
        require(header_victim({{{1, 2}, 4}, {{1, 2, 3}, 2}}, prefix) == 1, "all own prefixes: oldest overall");
        require(header_victim({{{7}, 3}, {{8}, 3}}, prefix) == 0, "ties keep the lowest index");
        require(header_victim({{{1, 2, 3, 4, 5, 6}, 0}, {{9}, 1}}, prefix) == 0, "an equal prefix is not an own prefix");
        require(header_victim({{{1, 2, 3, 4, 5, 6, 7}, 0}}, prefix) == 0, "a longer anchor is not an own prefix");

        // Randomised equivalence with the inline loop, including ties and shared stems.
        std::mt19937_64 rng(20261111);
        for (int round = 0; round < 200000; ++round) {
            const size_t count = 1 + rng() % 8;
            std::vector<uint32_t> capture(1 + rng() % 12);
            for (auto& t : capture) t = uint32_t(rng() % 3);
            std::vector<Anchor> anchors(count);
            for (auto& anchor : anchors) {
                const size_t length = 1 + rng() % 12;
                if (rng() % 2) {
                    anchor.tokens.assign(capture.begin(), capture.begin() + std::ptrdiff_t(std::min(length, capture.size())));
                } else {
                    anchor.tokens.resize(length);
                    for (auto& t : anchor.tokens) t = uint32_t(rng() % 3);
                }
                anchor.last_access_tick = rng() % 6;
            }
            require(header_victim(anchors, capture) == inline_loop_victim(anchors, capture), "lru reproduces the inline loop");
        }

        require(shared_prefix_eviction_cause(true, false) == SharedPrefixEvictionCause::bytes, "byte cause");
        require(shared_prefix_eviction_cause(false, true) == SharedPrefixEvictionCause::count, "count cause");
        require(shared_prefix_eviction_cause(true, true) == SharedPrefixEvictionCause::both, "both causes");
        SharedPrefixEvictionCounters counters;
        counters.record(SharedPrefixEvictionCause::bytes);
        counters.record(SharedPrefixEvictionCause::count);
        counters.record(SharedPrefixEvictionCause::count);
        counters.record(SharedPrefixEvictionCause::both);
        require(counters.bytes == 1 && counters.count == 2 && counters.both == 1, "eviction counters by cause");

        // F18: ~sqrt(N) milestone schedule & gap-minimizing thinning up to 131072 tokens.
        require(is_sqrt_checkpoint_milestone(512) && is_sqrt_checkpoint_milestone(1024) && is_sqrt_checkpoint_milestone(2048),
            "512-aligned milestones <= 2048");
        require(!is_sqrt_checkpoint_milestone(3072) && is_sqrt_checkpoint_milestone(4096) && is_sqrt_checkpoint_milestone(8192) &&
            is_sqrt_checkpoint_milestone(32768) && is_sqrt_checkpoint_milestone(131072),
            "2048-aligned sqrt milestones");
        require(sqrt_milestone_window_target(2048, 4096, 131072, 2048u) == 4096, "sqrt window target 4096");
        require(sqrt_milestone_window_target(4096, 6144, 131072, 2048u) == 0, "no target outside trailing milestone window");
        require(sqrt_milestone_window_target(0, 512, 131072, 2048u) == 0, "no sqrt target inside <= 2048 dense range (512)");
        require(sqrt_milestone_window_target(1536, 2048, 131072, 2048u) == 0, "no sqrt target inside <= 2048 dense range (2048)");
        require(sqrt_milestone_window_target(2048, 3072, 131072, 2048u) == 4096, "interior chunk overlapping 4096 window");
        require(sqrt_milestone_window_target(2048, 2560, 131072, 2048u) == 4096, "512-token interior chunk overlapping 4096 window");
        require(sqrt_milestone_window_target(6144, 7168, 131072, 2048u) == 8192, "interior chunk overlapping 8192 window");
        require(sqrt_milestone_window_target(71680, 73728, 131072, 2048u) == 73728, "second-half milestone 73728");
        require(sqrt_milestone_window_target(83968, 86016, 131072, 2048u) == 86016, "second-half milestone 86016");
        require(sqrt_milestone_window_target(98304, 100352, 131072, 2048u) == 100352, "second-half milestone 100352");
        require(sqrt_milestone_window_target(112640, 114688, 131072, 2048u) == 114688, "second-half milestone 114688");
        require(sqrt_milestone_window_target(129024, 131072, 131072, 2048u) == 131072, "second-half milestone 131072");
        require(sqrt_milestone_window_target(73728, 75776, 131072, 2048u) == 0, "non-milestone chunk past 73728");
        require(sqrt_milestone_window_target(71680, 73728, 65536, 2048u) == 0, "64K max_position caps >64K milestones");
        uint32_t sweep_hits = 0, sweep_skips = 0;
        for (uint32_t base = 0; base < 131072; base += 2048) {
            if (sqrt_milestone_window_target(base, base + 2048, 131072, 2048u) != 0) ++sweep_hits;
            else ++sweep_skips;
        }
        require(sweep_hits == 14 && sweep_skips == 50, "128K sweep milestone hit/skip count mismatch");
        const std::array<uint32_t, 5> lineage{1024, 2048, 4096, 8192, 16384};
        const std::array<uint64_t, 5> epochs{1, 2, 3, 4, 5};
        // Removing 2048 yields gap (4096 - 1024 = 3072), smallest among interior positions (>1024).
        require(sqrt_thinning_victim(lineage, epochs, 1024) == 1,
            "sqrt thinning must keep <=1024 anchor and newest frontier while minimizing gap");

        // F18_04: should_spill_ssd_tail gates non-milestone tail/fallback spills once >= 2048 milestone exists.
        require(should_spill_ssd_tail(1500, 1024), "allows <= 2048 tails");
        require(should_spill_ssd_tail(512, 0), "allows <= 2048 initial tail");
        require(should_spill_ssd_tail(3000, 1024), "allows first > 2048 spill when no >= 2048 milestone spilled");
        require(!should_spill_ssd_tail(73683, 61440), "rejects non-milestone 73683 leaf once 61440 spilled");
        require(!should_spill_ssd_tail(73731, 73728), "rejects non-milestone 73731 leaf once 73728 spilled");
        require(should_spill_ssd_tail(73728, 61440), "allows exact sqrt milestone 73728");
        require(!should_spill_ssd_tail(61440, 61440), "rejects duplicate milestone 61440");

        std::cout << "Shared-prefix eviction checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Shared-prefix eviction check failed: " << error.what() << '\n';
        return 1;
    }
}
