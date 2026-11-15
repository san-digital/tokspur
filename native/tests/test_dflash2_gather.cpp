// Model-free checks for the gathered DFlash2 draft vocabulary: draft rows are an explicit ascending list of token IDs, not the first N.
#include "tokspur/dflash2_drafter.hpp"
#include "tokspur/expert_lanes.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/grouped_mm.hpp"
#include "tokspur/memory.hpp"
#include "tokspur/sampled_path.hpp"
#include "tokspur/speculative_controls.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

using namespace tokspur;

namespace {

size_t checks = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) throw FormatError(why);
}

template <class F>
void rejects(F&& action, const char* why) {
    ++checks;
    try {
        action();
    } catch (const FormatError&) {
        return;
    }
    throw FormatError(why);
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

float bf16_to_f32(uint16_t raw) {
    const uint32_t bits = uint32_t(raw) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

constexpr uint16_t scale_bits = 0x2400u; // FP16 2^-6 = 0.015625, the scale of every block.

constexpr uint32_t hidden = 256u, full_vocab = 4096u, slice_vocab = 1024u, rank = 256u, positions = 3u;
constexpr uint32_t row_bytes = hidden / 32u * 34u;
constexpr size_t selector_bytes = 7u * 16u * 8u + 7u * 256u * 4u;
// Stand-ins for the real layout: the "added tokens" sit at the top of the vocabulary, far outside the first-N slice.
constexpr uint32_t control_first = 4044u, control_count = 33u;
constexpr uint32_t tok_end_turn = 4046u, tok_tool_call = 4058u, tok_think_close = 4069u;
constexpr uint32_t tok_dropped = 1000u;   // Inside the first-N slice, absent from the gathered set.
constexpr uint32_t tok_outside = 2000u;   // In neither draft vocabulary.
constexpr uint32_t anchor = 4045u;

void set_row(std::vector<std::byte>& gguf, uint32_t token, const std::array<int8_t, hidden>& quants) {
    for (uint32_t block = 0; block < hidden / 32u; ++block) {
        std::byte* at = gguf.data() + size_t(token) * row_bytes + size_t(block) * 34u;
        std::memcpy(at, &scale_bits, 2);
        std::memcpy(at + 2, quants.data() + size_t(block) * 32u, 32);
    }
}

struct Fixture {
    std::vector<std::byte> gguf;
    std::vector<float> final_h, selector_h;
    std::vector<uint16_t> predecessor, successor;
    std::vector<uint32_t> identity_ids, control_ids;
};

Fixture make_fixture() {
    Fixture f;
    f.gguf.resize(size_t(full_vocab) * row_bytes);
    std::mt19937 rng(0x6a7e12u);
    std::uniform_int_distribution<int> quant(-80, 80);
    std::array<int8_t, hidden> row{};
    for (uint32_t token = 0; token < full_vocab; ++token) {
        for (auto& q : row) q = int8_t(quant(rng));
        set_row(f.gguf, token, row);
    }
    f.final_h.resize(size_t(positions) * hidden);
    for (uint32_t c = 0; c < hidden; ++c) {
        f.final_h[c] = float(int((c * 7u + 3u) % 23u) - 11) * 0.125f;       // Position 0: mixed signs.
        f.final_h[hidden + c] = 0.125f;                                       // Position 1: a logit is the row's quant sum / 512.
        f.final_h[2u * hidden + c] = float(int((c * 5u + 1u) % 19u) - 9) * 0.125f;
    }
    // Position 0: the end-of-turn token wins by hundreds of logits.
    for (uint32_t c = 0; c < hidden; ++c) row[c] = int8_t(f.final_h[c] > 0 ? 127 : f.final_h[c] < 0 ? -127 : 0);
    set_row(f.gguf, tok_end_turn, row);
    // Position 1: graded logits 15.0, 10.5, 10.0, 9.5, 9.0, 8.5 so that two control tokens share a sampled law with ordinary tokens. Random rows stay below 8.
    const std::array<std::pair<uint32_t, int8_t>, 6> graded{{{tok_dropped, 30}, {tok_outside, 21}, {tok_think_close, 20}, {5u, 19}, {tok_tool_call, 18}, {700u, 17}}};
    for (const auto& [token, value] : graded) {
        row.fill(value);
        set_row(f.gguf, token, row);
    }
    f.selector_h.resize(size_t(positions) * rank);
    for (size_t i = 0; i < f.selector_h.size(); ++i) f.selector_h[i] = float(int((i * 3 + 1) % 19) - 9) * 0.0625f;
    f.predecessor.resize(size_t(full_vocab) * rank);
    f.successor.resize(size_t(full_vocab) * rank);
    for (size_t i = 0; i < f.predecessor.size(); ++i) {
        f.predecessor[i] = f32_to_bf16(float(int((i * 5 + 7) % 17) - 8) * 0.0625f);
        f.successor[i] = f32_to_bf16(float(int((i * 11 + 3) % 21) - 10) * 0.0625f);
    }
    f.identity_ids.resize(slice_vocab);
    std::iota(f.identity_ids.begin(), f.identity_ids.end(), 0u);
    // "First N minus the least useful rows, plus the added tokens", at the same row count as the slice.
    for (uint32_t token = 0; token < slice_vocab - control_count; ++token) f.control_ids.push_back(token);
    for (uint32_t i = 0; i < control_count; ++i) f.control_ids.push_back(control_first + i);
    return f;
}

struct Run {
    std::vector<float> logits;
    std::vector<std::byte> selector;
    uint64_t dispatches = 0;
};

struct Device {
    MetalGraph& g;
    GpuView h, selector_h, predecessor, successor, logits, top_ids, top_vals;
    // Projects final hidden rows through `weight` and runs the selector; row_token_ids is the gathered table or empty.
    Run draft(GpuView weight, uint32_t vocabulary, GpuView row_token_ids) const {
        const auto selector_out = g.zeros(selector_bytes, "gather.selector_out");
        const auto out = g.slice(logits, 0, uint64_t(positions) * vocabulary * 4u);
        g.begin();
        g.set_dflash2_scope(true);
        try {
            g.dflash2_dense_q8(weight, GpuRowsF32{h, positions, hidden, hidden * 4u}, GpuRowsF32{out, positions, vocabulary, uint64_t(vocabulary) * 4u}, false);
            g.dflash2_select(out, top_ids, top_vals, selector_out, selector_h, predecessor, successor, positions, vocabulary, anchor, full_vocab, row_token_ids);
        } catch (...) {
            g.set_dflash2_scope(false);
            g.abort();
            g.release(selector_out);
            throw;
        }
        g.set_dflash2_scope(false);
        Run run;
        run.dispatches = g.submit().dispatches;
        run.logits.resize(size_t(positions) * vocabulary);
        g.read(out, std::as_writable_bytes(std::span(run.logits)));
        run.selector = g.read(selector_out);
        g.release(selector_out);
        return run;
    }
    // Selector alone over caller-supplied logits (the full-vocabulary oracle).
    std::vector<std::byte> select(std::span<const float> host_logits, uint32_t vocabulary) const {
        const auto selector_out = g.zeros(selector_bytes, "gather.oracle_out");
        const auto input = g.upload(std::as_bytes(host_logits), "gather.oracle_logits");
        g.begin();
        g.set_dflash2_scope(true);
        g.dflash2_select(input, top_ids, top_vals, selector_out, selector_h, predecessor, successor, positions, vocabulary, anchor, full_vocab);
        g.set_dflash2_scope(false);
        static_cast<void>(g.submit());
        auto bytes = g.read(selector_out);
        g.release(input);
        g.release(selector_out);
        return bytes;
    }
};

const uint32_t* candidates_of(const std::vector<std::byte>& selector) { return reinterpret_cast<const uint32_t*>(selector.data()); }
const float* unary_of(const std::vector<std::byte>& selector) { return reinterpret_cast<const float*>(selector.data() + 7u * 16u * 4u); }
const float* tables_of(const std::vector<std::byte>& selector) { return reinterpret_cast<const float*>(selector.data() + 7u * 16u * 8u); }

bool same_bits(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// Independent host oracle: the 16 best tokens of a set under full-vocabulary logits (higher logit first, lower token ID on a tie), and their codebook edge scores looked up by token ID.
void check_against_host_oracle(const Fixture& f, const std::vector<float>& full_logits, std::span<const uint32_t> token_ids, const std::vector<std::byte>& selector) {
    const auto* cands = candidates_of(selector);
    const auto* unary = unary_of(selector);
    const auto* tables = tables_of(selector);
    for (uint32_t pos = 0; pos < positions; ++pos) {
        std::vector<uint32_t> order(token_ids.begin(), token_ids.end());
        const float* row = full_logits.data() + size_t(pos) * full_vocab;
        std::partial_sort(order.begin(), order.begin() + 16, order.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
        for (uint32_t j = 0; j < 16u; ++j) {
            check(cands[pos * 16u + j] == order[j], "gathered candidate is not the token ID of the best gathered row");
            check(std::memcmp(&unary[pos * 16u + j], &row[order[j]], 4) == 0, "gathered candidate score differs from that token's full-vocabulary logit");
        }
        const uint32_t predecessors = pos ? 16u : 1u;
        for (uint32_t p = 0; p < predecessors; ++p) {
            const uint32_t previous = pos ? cands[(pos - 1u) * 16u + p] : anchor;
            for (uint32_t j = 0; j < 16u; ++j) {
                const uint32_t token = cands[pos * 16u + j];
                double expected = 0.0;
                for (uint32_t d = 0; d < rank; ++d)
                    expected += double(bf16_to_f32(f.predecessor[size_t(previous) * rank + d])) * double(f.selector_h[size_t(pos) * rank + d]) *
                        double(bf16_to_f32(f.successor[size_t(token) * rank + d]));
                check(std::abs(double(tables[(pos * 16u + p) * 16u + j]) - expected) <= 1e-3 * std::max(1.0, std::abs(expected)),
                      "edge score did not use the codebook rows of the predecessor and candidate token IDs");
            }
        }
    }
}

void test_layout(const char* metallib, const Fixture& f, PrefillWeightLayout layout, bool m8) {
    MetalGraph g(metallib, 512ull << 20);
    g.configure_verify_nax(true);
    if (m8) g.configure_verify_nax_m8(true);
    g.prepare_verify_nax({});
    g.prepare_dflash2();
    const auto initial_bytes = g.memory().live_bytes;
    const bool cached = q8_tile64_cache_enabled;

    auto payload = f.gguf;
    if (layout == PrefillWeightLayout::q8_lanes_v1) lanes::pack_rows(lanes::row_layout(8u, hidden), f.gguf, payload);
    const auto full_w = g.upload(payload, "gather.full_out", true);
    if (layout == PrefillWeightLayout::q8_lanes_v1) g.register_lane_matrix(full_w, {8u, hidden, full_vocab, 1u});
    else g.register_q8_matrix(full_w, hidden, full_vocab);
    const Device device{g,
        g.upload(std::as_bytes(std::span(f.final_h)), "gather.h"), g.upload(std::as_bytes(std::span(f.selector_h)), "gather.selector_h"),
        g.upload(std::as_bytes(std::span(f.predecessor)), "gather.predecessor", true), g.upload(std::as_bytes(std::span(f.successor)), "gather.successor", true),
        g.zeros(uint64_t(positions) * full_vocab * 4u, "gather.logits"), g.zeros(7u * 8u * 16u * 4u, "gather.top_ids"), g.zeros(7u * 8u * 16u * 4u, "gather.top_vals")};

    // Current behaviour: full vocabulary, then the first-N slice. These batches also close lane registration, as serving has by the time a drafter is configured.
    const auto full = device.draft(full_w, full_vocab, {});
    const auto prefix_w = g.suballocate_draft_output_slice(full_w, slice_vocab, hidden);
    const auto prefix = device.draft(prefix_w, slice_vocab, {});
    check(prefix.dispatches == 3, "first-N slice dispatch count changed");
    for (uint32_t i = 0; i < positions * 16u; ++i) check(candidates_of(prefix.selector)[i] < slice_vocab, "first-N slice proposed a token outside the slice");
    if (cached) check(g.q8_tile64_cache_stats().owning_allocations == 1 && g.q8_tile64_cache_stats().aliases == 1, "first-N slice cache state differs");

    // Refusals leave the first-N slice in place.
    const std::array<uint32_t, 3> unsorted{4u, 3u, 9u}, repeated{4u, 4u, 9u}, outside{4u, 9u, full_vocab};
    rejects([&] { static_cast<void>(g.gather_draft_output_slice(full_w, unsorted, hidden)); }, "gather accepted descending token IDs");
    rejects([&] { static_cast<void>(g.gather_draft_output_slice(full_w, repeated, hidden)); }, "gather accepted a repeated token ID");
    rejects([&] { static_cast<void>(g.gather_draft_output_slice(full_w, outside, hidden)); }, "gather accepted a token ID outside the parent");
    rejects([&] { static_cast<void>(g.gather_draft_output_slice(full_w, std::span<const uint32_t>{}, hidden)); }, "gather accepted an empty set");
    rejects([&] { static_cast<void>(g.gather_draft_output_slice(full_w, f.identity_ids, hidden * 2u)); }, "gather accepted a wrong row width");

    // Identity set: rows 0..N-1 gathered into a copy must reproduce the first-N slice bit for bit, with and without the table.
    const auto before_gather = g.memory().live_bytes;
    const auto identity_w = g.gather_draft_output_slice(full_w, f.identity_ids, hidden);
    check(identity_w.bytes == uint64_t(slice_vocab) * row_bytes && identity_w.slot != full_w.slot, "gathered slice is not its own allocation of N rows");
    check(g.prefill_weight_layout(identity_w) == layout, "gathered slice lost the parent's layout");
    check(std::ranges::equal(g.read(identity_w), std::span(payload).first(size_t(slice_vocab) * row_bytes)), "identity gather differs from the first N parent rows");
    if (cached) check(g.q8_tile64_cache_stats().owning_allocations == 2 && g.q8_tile64_cache_stats().aliases == 0, "gathered slice must own its cache planes and drop the alias");
    const auto identity_table = g.upload(std::as_bytes(std::span(f.identity_ids)), "gather.identity_table", true);
    const auto identity = device.draft(identity_w, slice_vocab, identity_table);
    check(identity.dispatches == 4, "gathered slice must add exactly one remap dispatch");
    check(same_bits(identity.logits, prefix.logits), "identity gather logits differ from the first-N slice");
    check(identity.selector == prefix.selector, "identity gather selector result differs from the first-N slice");
    check(device.draft(identity_w, slice_vocab, {}).selector == prefix.selector, "identity gather without a table differs from the first-N slice");
    rejects([&] { static_cast<void>(g.gather_draft_output_slice(full_w, f.control_ids, hidden)); }, "a live gathered slice was silently replaced");
    rejects([&] { static_cast<void>(g.suballocate_draft_output_slice(full_w, slice_vocab, hidden)); }, "a live gathered slice was silently replaced by a prefix slice");
    rejects([&] { static_cast<void>(device.draft(identity_w, slice_vocab, g.slice(identity_table, 0, 64))); }, "selector accepted a table shorter than the draft vocabulary");
    if (layout == PrefillWeightLayout::q8_lanes_v1)
        rejects([&] { static_cast<void>(g.prefill_weight_layout(g.slice(identity_w, 0, uint64_t(64) * row_bytes))); }, "a subview of the packed gathered slice lost its layout silently");
    g.release(identity_table);
    g.release(identity_w);
    rejects([&] { static_cast<void>(g.slice(identity_w, 0, 4)); }, "released gathered slice remained usable");
    check(g.memory().live_bytes == before_gather, "released gathered slice leaked storage or cache planes");
    if (cached) check(g.q8_tile64_cache_stats().owning_allocations == 1, "released gathered slice left cache planes behind");

    // Control set: the first N minus its last rows, plus control tokens from the top of the vocabulary.
    const auto control_w = g.gather_draft_output_slice(full_w, f.control_ids, hidden);
    const auto control_table = g.upload(std::as_bytes(std::span(f.control_ids)), "gather.control_table", true);
    const auto control = device.draft(control_w, slice_vocab, control_table);
    for (uint32_t pos = 0; pos < positions; ++pos)
        for (uint32_t row = 0; row < slice_vocab; ++row)
            check(std::memcmp(&control.logits[size_t(pos) * slice_vocab + row], &full.logits[size_t(pos) * full_vocab + f.control_ids[row]], 4) == 0,
                  "gathered row logit differs from that token's full-vocabulary logit");
    check_against_host_oracle(f, full.logits, f.control_ids, control.selector);
    check_against_host_oracle(f, full.logits, f.identity_ids, prefix.selector);
    // GPU oracle: the unchanged full-vocabulary selector over logits with every token outside the set pushed out of reach must give the same bytes.
    auto masked = full.logits;
    std::vector<uint8_t> member(full_vocab, 0);
    for (const auto token : f.control_ids) member[token] = 1;
    for (uint32_t pos = 0; pos < positions; ++pos)
        for (uint32_t token = 0; token < full_vocab; ++token)
            if (!member[token]) masked[size_t(pos) * full_vocab + token] = -1.0e30f;
    check(device.select(masked, full_vocab) == control.selector, "gathered selector differs from full-vocabulary selection restricted to the set");
    // Without the table the same rows would be reported under their slice row numbers, which is the first-N assumption this replaces.
    const auto untranslated = device.draft(control_w, slice_vocab, {});
    check(untranslated.selector != control.selector && *std::max_element(candidates_of(untranslated.selector), candidates_of(untranslated.selector) + positions * 16u) < slice_vocab,
          "row-to-token table made no difference for a set that is not a prefix");

    // A control token can be proposed, under its own ID, and the first-N slice cannot do that.
    const auto* cands = candidates_of(control.selector);
    check(cands[0] == tok_end_turn, "gathered slice did not rank the end-of-turn token first");
    check(std::find(cands + 16, cands + 32, tok_think_close) != cands + 32 && std::find(cands + 16, cands + 32, tok_tool_call) != cands + 32,
          "gathered slice did not shortlist the thinking-close and tool-call tokens");
    check(std::find(cands + 16, cands + 32, tok_dropped) == cands + 32 && std::find(cands + 16, cands + 32, tok_outside) == cands + 32,
          "gathered slice shortlisted a token that is not in the set");
    check(std::find(candidates_of(prefix.selector) + 16, candidates_of(prefix.selector) + 32, tok_dropped) != candidates_of(prefix.selector) + 32,
          "fixture lost the row that only the first-N slice can propose");

    // Host half, production code: greedy proposes the end-of-turn token as a point mass over the full vocabulary.
    constexpr uint32_t base_position = 99u;
    const SampledPathIdentity identity_of_path{1ull, 1ull, 1ull, base_position + 1u, full_vocab, anchor};
    const SamplingRng start(7);
    const auto greedy = dflash2_selector_proposals(control.selector, positions, full_vocab, base_position, anchor, SamplingPolicy{}, start);
    check(greedy.proposals[0].selected == tok_end_turn && greedy.proposals[0].predecessor == anchor && greedy.proposals[0].position == base_position + 1u,
          "greedy proposal is not the end-of-turn token at the first drafted position");
    check(greedy.proposals[0].actual_q.vocabulary() == full_vocab && greedy.proposals[0].actual_q.units(tok_end_turn) == sampled_path_grid_size &&
          greedy.random_after == start, "greedy control-token law is not a full-vocabulary point mass");
    const auto greedy_prefix = dflash2_selector_proposals(prefix.selector, positions, full_vocab, base_position, anchor, SamplingPolicy{}, start);
    check(greedy_prefix.proposals[0].selected < slice_vocab && greedy_prefix.proposals[1].selected == tok_dropped, "first-N slice greedy proposals changed");

    // Target rows over the full vocabulary. Row r is the target law after r accepted proposals.
    std::vector<float> target(size_t(positions + 1u) * full_vocab, -30.0f);
    const auto at = [&](uint32_t row, uint32_t token) -> float& { return target[size_t(row) * full_vocab + token]; };
    at(0, tok_end_turn) = 2.0f; at(0, 7u) = 1.6f; at(0, tok_outside) = 1.2f;
    at(1, tok_think_close) = 2.0f; at(1, 5u) = 0.8f; at(1, 3000u) = 2.2f;
    at(2, tok_tool_call) = 2.0f; at(2, 1500u) = 2.0f;
    at(3, 9u) = 2.0f;
    std::array<SampledTargetRow, positions + 1u> targets{};
    for (uint32_t r = 0; r <= positions; ++r) targets[r] = SampledTargetRow{base_position + 1u + r, std::span<const float>(target).subspan(size_t(r) * full_vocab, full_vocab)};
    // Greedy target that agrees: the drafted control token is accepted. The first-N slice has to be corrected to it instead.
    {
        std::vector<float> agree(target.size(), -30.0f);
        agree[tok_end_turn] = 9.0f; agree[full_vocab + tok_think_close] = 9.0f; agree[2u * full_vocab + 3u] = 9.0f; agree[3u * full_vocab + 3u] = 9.0f;
        std::array<SampledTargetRow, positions + 1u> rows{};
        for (uint32_t r = 0; r <= positions; ++r) rows[r] = SampledTargetRow{base_position + 1u + r, std::span<const float>(agree).subspan(size_t(r) * full_vocab, full_vocab)};
        const auto accepted = decide_sampled_path(identity_of_path, std::span(greedy.proposals).first(positions), identity_of_path, rows, SamplingPolicy{}, 16u, {}, greedy.random_after, start.words_consumed());
        check(accepted.accepted >= 1u && accepted.tokens[0].id == tok_end_turn && accepted.tokens[0].accepted_proposal, "drafted control token was not accepted by an agreeing target");
        const auto corrected = decide_sampled_path(identity_of_path, std::span(greedy_prefix.proposals).first(positions), identity_of_path, rows, SamplingPolicy{}, 16u, {}, greedy_prefix.random_after, start.words_consumed());
        check(corrected.accepted == 0u && corrected.count == 1u && corrected.tokens[0].id == tok_end_turn && !corrected.tokens[0].accepted_proposal,
              "first-N slice did not need a correction for the control token");
    }

    // Sampled path. Each law must be the law its proposal was drawn from, keyed by token IDs of the set, and the committed tokens must follow the target law whatever the draft law is.
    const SamplingPolicy sampled{0.8, 20u, 1.0, 0.0};
    const auto p0 = sampling_distribution(targets[0].logits, sampled), p1 = sampling_distribution(targets[1].logits, sampled);
    std::map<uint32_t, uint32_t> first_counts, second_counts;
    uint32_t first_accepted = 0, control_drawn_at_1 = 0, tool_drawn_at_1 = 0, outcomes = 0;
    constexpr uint32_t trials = 40000u;
    for (uint32_t seed = 0; seed < trials; ++seed) {
        const SamplingRng rng(seed);
        const auto drawn = dflash2_selector_proposals(control.selector, positions, full_vocab, base_position, anchor, sampled, rng);
        if (seed < 256u) {
            auto replay = rng;
            uint32_t previous = anchor;
            for (uint32_t pos = 0; pos < positions; ++pos) {
                const auto& proposal = drawn.proposals[pos];
                check(proposal.actual_q.vocabulary() == full_vocab && proposal.predecessor == previous, "sampled proposal law or predecessor is not over full-vocabulary token IDs");
                uint64_t units = 0;
                for (uint32_t s = 0; s < proposal.actual_q.size(); ++s) {
                    check(member[proposal.actual_q.identifiers()[s]] == 1, "sampled proposal law has mass on a token outside the gathered set");
                    units += proposal.actual_q.mass_units()[s];
                }
                check(units == sampled_path_grid_size, "sampled proposal law does not sum to the whole grid");
                const uint64_t ticket = proposal.actual_q.deterministic() ? 0ull : uint64_t(std::ldexp(replay.uniform(), 53));
                check(proposal.actual_q.sample_at(ticket) == proposal.selected, "proposal was not drawn from the law that verification will use");
                previous = proposal.selected;
            }
            check(replay == drawn.random_after, "proposal entropy accounting differs");
            const auto& law1 = drawn.proposals[1].actual_q;
            check(law1.units(tok_think_close) > 0 && law1.units(tok_think_close) < sampled_path_grid_size && law1.units(tok_tool_call) > 0 && law1.units(5u) > 0,
                  "position 1 law lost the control tokens or stopped being a mixed law");
        }
        control_drawn_at_1 += drawn.proposals[1].selected == tok_think_close;
        tool_drawn_at_1 += drawn.proposals[1].selected == tok_tool_call;
        const auto decision = decide_sampled_path(identity_of_path, std::span(drawn.proposals).first(positions), identity_of_path, targets, sampled, 16u, {}, drawn.random_after, rng.words_consumed());
        check(decision.count >= 1u, "sampled path committed nothing");
        ++first_counts[decision.tokens[0].id];
        outcomes |= 1u << std::min(decision.accepted, 3u);
        if (decision.accepted >= 1u) {
            check(decision.tokens[0].id == tok_end_turn && decision.count >= 2u, "accepted first proposal is not the drafted control token");
            ++first_accepted;
            ++second_counts[decision.tokens[1].id];
        }
    }
    check(control_drawn_at_1 > 0 && tool_drawn_at_1 > 0, "control tokens were not drawn from the sampled law");
    check((outcomes & 7u) == 7u, "sampled path did not exercise rejection, one acceptance and two acceptances");
    for (const auto token : {tok_end_turn, 7u, tok_outside})
        check(std::abs(double(first_counts[token]) / trials - p0.probability(token)) < 0.015, "first committed token does not follow the target law");
    for (const auto token : {tok_think_close, 5u, 3000u})
        check(std::abs(double(second_counts[token]) / first_accepted - p1.probability(token)) < 0.015, "second committed token does not follow the target law");
    check(first_counts[tok_outside] > 0 && second_counts[3000u] > 0, "tokens outside the gathered set were not recoverable");

    g.release(control_table);
    g.release(control_w);
    // The first-N slice can be chosen again once the gathered slice is released, with the same result as before.
    const auto again_w = g.suballocate_draft_output_slice(full_w, slice_vocab, hidden);
    const auto again = device.draft(again_w, slice_vocab, {});
    check(same_bits(again.logits, prefix.logits) && again.selector == prefix.selector, "first-N slice differs after a gathered slice was used and released");
    for (const auto view : {device.top_vals, device.top_ids, device.logits, device.successor, device.predecessor, device.selector_h, device.h, full_w}) g.release(view);
    check(g.memory().live_bytes == initial_bytes, "gathered draft vocabulary fixture leaked GPU storage");
}

void test_parser_and_first_set() {
    // The first production set: the 33 added tokens replace 33 single-byte tokens that valid UTF-8 text cannot contain or that are C0 controls.
    const auto ids = parse_dflash2_vocab_gather("0-123,126-176,197-201,213-32767,248044-248076", 248320u);
    check(ids.size() == 32768u && ids.front() == 0u && ids.back() == 248076u && std::ranges::is_sorted(ids), "first gathered set is not 32768 ascending IDs");
    check(std::ranges::binary_search(ids, 248046u) && std::ranges::binary_search(ids, 248058u) && std::ranges::binary_search(ids, 248059u) &&
          std::ranges::binary_search(ids, 248068u) && std::ranges::binary_search(ids, 248069u), "first gathered set lost a control token");
    check(!std::ranges::binary_search(ids, 124u) && !std::ranges::binary_search(ids, 187u) && !std::ranges::binary_search(ids, 212u) &&
          std::ranges::binary_search(ids, 197u) && std::ranges::binary_search(ids, 213u), "first gathered set drop list differs");
}

void bench(const char* metallib) {
    // Production geometry, informational only: 2560 -> 32768 projection plus selector, first-N slice against the first gathered set.
    constexpr uint32_t bh = 2560u, bv = 248320u, bs = 32768u, brow = bh / 32u * 34u;
    MetalGraph g(metallib, 3072ull << 20);
    g.configure_verify_nax(true);
    g.configure_verify_nax_m8(true);
    g.prepare_verify_nax({});
    g.prepare_dflash2();
    GpuView w{};
    {
        std::vector<std::byte> packed(size_t(bv) * brow);
        for (uint32_t r = 0; r < bv; ++r) {
            std::byte* row = packed.data() + size_t(r) * brow;
            std::memset(row, int((r * 2654435761u) >> 25) - 64, brow);
            for (uint32_t b = 0; b < bh / 32u; ++b) std::memcpy(row + size_t(b) * 2u, &scale_bits, 2);
        }
        w = g.upload(packed, "gather.bench_out", true);
    }
    g.register_lane_matrix(w, {8u, bh, bv, 1u});
    const std::vector<float> h7(7u * bh, 0.0625f), sh7(7u * rank, 0.0625f);
    const auto h = g.upload(std::as_bytes(std::span(h7)), "gather.bench_h");
    const auto sh = g.upload(std::as_bytes(std::span(sh7)), "gather.bench_sh");
    const auto pred = g.zeros(uint64_t(bv) * rank * 2u, "gather.bench_pred");
    const auto succ = g.zeros(uint64_t(bv) * rank * 2u, "gather.bench_succ");
    const auto logits = g.zeros(7ull * bs * 4u, "gather.bench_logits");
    const auto top_ids = g.zeros(7u * 8u * 16u * 4u, "gather.bench_top_ids");
    const auto top_vals = g.zeros(7u * 8u * 16u * 4u, "gather.bench_top_vals");
    const auto out = g.zeros(selector_bytes, "gather.bench_out_sel");
    const auto time = [&](GpuView weight, GpuView table, uint32_t k) {
        double total = 0.0;
        constexpr uint32_t warm = 4u, iterations = 24u;
        for (uint32_t i = 0; i < warm + iterations; ++i) {
            g.begin();
            g.set_dflash2_scope(true);
            g.dflash2_dense_q8(weight, GpuRowsF32{g.slice(h, 0, uint64_t(k) * bh * 4u), k, bh, bh * 4u}, GpuRowsF32{g.slice(logits, 0, uint64_t(k) * bs * 4u), k, bs, uint64_t(bs) * 4u}, false);
            g.dflash2_select(g.slice(logits, 0, uint64_t(k) * bs * 4u), top_ids, top_vals, out, g.slice(sh, 0, uint64_t(k) * rank * 4u), pred, succ, k, bs, 200000u, bv, table);
            g.set_dflash2_scope(false);
            const auto batch = g.submit();
            if (i >= warm) total += batch.gpu_ms;
        }
        return total / iterations;
    };
    const auto prefix_w = g.suballocate_draft_output_slice(w, bs, bh);
    const double prefix3 = time(prefix_w, {}, 3u), prefix7 = time(prefix_w, {}, 7u);
    const auto ids = parse_dflash2_vocab_gather("0-123,126-176,197-201,213-32767,248044-248076", bv);
    const auto before = g.memory().live_bytes;
    const auto gathered_w = g.gather_draft_output_slice(w, ids, bh);
    const auto table = g.upload(std::as_bytes(std::span(ids)), "gather.bench_table", true);
    const auto extra = g.memory().live_bytes - before;
    check(std::ranges::equal(g.read(g.slice(gathered_w, uint64_t(bs - 33u) * brow, 33ull * brow)), g.read(g.slice(w, uint64_t(248044u) * brow, 33ull * brow))),
          "production-geometry gather did not place the added tokens in the last 33 rows");
    const double gathered3 = time(gathered_w, table, 3u), gathered7 = time(gathered_w, table, 7u);
    g.release(table);
    g.release(gathered_w);
    const auto prefix_again = g.suballocate_draft_output_slice(w, bs, bh);
    const double prefix3b = time(prefix_again, {}, 3u), prefix7b = time(prefix_again, {}, 7u);
    std::cout << "dflash2_gather bench (model-free, 2560->32768 projection + selector, gpu_ms per draft): first_n_k3=" << prefix3 << " gathered_k3=" << gathered3
              << " first_n_k3_repeat=" << prefix3b << " first_n_k7=" << prefix7 << " gathered_k7=" << gathered7 << " first_n_k7_repeat=" << prefix7b
              << " gathered_extra_bytes=" << extra << " q8_tile64_cache=" << q8_tile64_cache_enabled << '\n';
    for (const auto view : {out, top_vals, top_ids, logits, succ, pred, sh, h, w}) g.release(view);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool correctness_only = argc == 3 && std::strcmp(argv[2], "--correctness-only") == 0;
        check(argc == 2 || correctness_only, "usage: test-dflash2_gather METALLIB [--correctness-only]");
        test_parser_and_first_set();
        const auto fixture = make_fixture();
        for (const auto layout : {PrefillWeightLayout::gguf, PrefillWeightLayout::q8_lanes_v1})
            for (const bool m8 : {false, true}) test_layout(argv[1], fixture, layout, m8);
        std::cout << "PASS DFlash2 gathered draft vocabulary: identity set bit-identical to the first-N slice, control tokens proposed under their own IDs, "
                     "selector equal to full-vocabulary selection restricted to the set, sampled path follows the target law; checks=" << checks
                  << " q8_tile64_cache=" << q8_tile64_cache_enabled << '\n';
        if (!correctness_only) bench(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
