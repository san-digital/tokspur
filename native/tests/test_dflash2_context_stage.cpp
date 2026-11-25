#include "tokspur/dflash2_drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using namespace tokspur;

namespace {
void check(bool ok, const char* why) { if (!ok) throw FormatError(why); }
template<class F> void rejects(F&& action) {
    try { action(); } catch (const FormatError&) { return; }
    throw FormatError("invalid pending context admitted");
}
uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return uint16_t(bits >> 16);
}
float bf16_to_f32(uint16_t raw) {
    const uint32_t bits = uint32_t(raw) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
void check_dflash2_bf16_kernel(const char* metallib) {
    MetalGraph graph(metallib, 8ull << 20);
    graph.prepare_dflash2();
    const auto run_dflash2 = [&](GpuView w, GpuRowsF32 in, GpuRowsF32 out, bool swiglu) {
        struct ScopeGuard {
            MetalGraph& g;
            ~ScopeGuard() { try { g.set_dflash2_scope(false); } catch (...) {} }
        };
        graph.begin();
        ScopeGuard guard{graph};
        graph.set_dflash2_scope(true);
        graph.dflash2_dense_bf16(w, in, out, swiglu);
        static_cast<void>(graph.submit());
    };
    constexpr uint32_t columns = 64, rows = 37;
    std::vector<uint16_t> w_bf16(size_t(rows) * columns);
    for (size_t i = 0; i < w_bf16.size(); ++i) {
        w_bf16[i] = f32_to_bf16(float(int(i % 29) - 14) * 0.0625f);
    }
    const auto w_gpu = graph.upload(std::as_bytes(std::span(w_bf16)), "dflash2 bf16 test weights");
    for (uint32_t tokens : {1u, 8u, 17u}) {
        std::vector<float> x_host(size_t(tokens) * columns), gate_host(size_t(tokens) * rows), out_host(size_t(tokens) * rows);
        for (size_t i = 0; i < x_host.size(); ++i) x_host[i] = float(int((i * 7 + 3) % 31) - 15) * 0.125f;
        for (size_t i = 0; i < gate_host.size(); ++i) gate_host[i] = float(int((i * 5 + 1) % 23) - 11) * 0.25f;
        const auto x_gpu = graph.upload(std::as_bytes(std::span(x_host)), "dflash2 bf16 test input");
        const auto out_gpu = graph.zeros(out_host.size() * sizeof(float), "dflash2 bf16 test output");
        const auto x_rows = GpuRowsF32{x_gpu, tokens, columns, uint64_t(columns) * sizeof(float)};
        const auto out_rows = GpuRowsF32{out_gpu, tokens, rows, uint64_t(rows) * sizeof(float)};
        run_dflash2(w_gpu, x_rows, out_rows, false);
        graph.read(out_gpu, std::as_writable_bytes(std::span(out_host)));
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t r = 0; r < rows; ++r) {
                float ref = 0.0f;
                for (uint32_t c = 0; c < columns; ++c) {
                    ref += bf16_to_f32(w_bf16[size_t(r) * columns + c]) * x_host[size_t(t) * columns + c];
                }
                check(std::fabs(out_host[size_t(t) * rows + r] - ref) <= 1e-5f, "dflash2_dense_bf16 linear mismatch");
            }
        }
        graph.write(out_gpu, std::as_bytes(std::span(gate_host)));
        run_dflash2(w_gpu, x_rows, out_rows, true);
        graph.read(out_gpu, std::as_writable_bytes(std::span(out_host)));
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t r = 0; r < rows; ++r) {
                float up = 0.0f;
                for (uint32_t c = 0; c < columns; ++c) {
                    up += bf16_to_f32(w_bf16[size_t(r) * columns + c]) * x_host[size_t(t) * columns + c];
                }
                const float g = gate_host[size_t(t) * rows + r];
                const float sig = g >= 0.0f ? 1.0f / (1.0f + std::exp(-g)) : std::exp(g) / (1.0f + std::exp(g));
                const float ref = (g * sig) * up;
                check(std::fabs(out_host[size_t(t) * rows + r] - ref) <= 1e-4f, "dflash2_dense_bf16 swiglu mismatch");
            }
        }
        graph.release(out_gpu);
        graph.release(x_gpu);
    }
    // Non-finite input, BF16 weight and SwiGLU gate propagate without masking and are caught at the selector/end-of-batch boundary.
    {
        const auto check_out_nonfinite = [&](GpuView out_v, uint32_t expected_rows, uint32_t expected_cols) {
            std::vector<float> got(size_t(expected_rows) * expected_cols);
            graph.read(out_v, std::as_writable_bytes(std::span(got)));
            check(std::any_of(got.begin(), got.end(), [](float v) { return !std::isfinite(v); }),
                  "dflash2_dense_bf16 must propagate non-finite values without zero substitution");
        };
        std::vector<float> bad_x(columns, 1.0f);
        bad_x[7] = std::numeric_limits<float>::quiet_NaN();
        const auto x_gpu = graph.upload(std::as_bytes(std::span(bad_x)), "dflash2 bad input", false);
        const auto out_gpu = graph.zeros(rows * sizeof(float), "dflash2 bad output");
        const auto x_rows = GpuRowsF32{x_gpu, 1u, columns, uint64_t(columns) * sizeof(float)};
        const auto out_rows = GpuRowsF32{out_gpu, 1u, rows, uint64_t(rows) * sizeof(float)};
        run_dflash2(w_gpu, x_rows, out_rows, false);
        check_out_nonfinite(out_gpu, 1u, rows);

        std::fill(bad_x.begin(), bad_x.end(), 1.0f);
        graph.write(x_gpu, std::as_bytes(std::span(bad_x)));
        std::vector<float> bad_gate(rows, 0.5f);
        bad_gate[3] = std::numeric_limits<float>::infinity();
        graph.write(out_gpu, std::as_bytes(std::span(bad_gate)));
        run_dflash2(w_gpu, x_rows, out_rows, true);
        check_out_nonfinite(out_gpu, 1u, rows);

        std::vector<uint16_t> bad_w = w_bf16;
        bad_w[19] = 0x7f80u;
        const auto bad_w_gpu = graph.upload(std::as_bytes(std::span(bad_w)), "dflash2 bad weight");
        std::fill(bad_gate.begin(), bad_gate.end(), 0.0f);
        graph.write(out_gpu, std::as_bytes(std::span(bad_gate)));
        run_dflash2(bad_w_gpu, x_rows, out_rows, false);
        check_out_nonfinite(out_gpu, 1u, rows);

        // Verify selector top16 sharded gate refuses non-finite logits (e.g., from propagated out_gpu),
        // and verify host dflash2_selector_proposals refuses non-finite unary/edges.
        constexpr uint32_t sel_vocab = 64u;
        std::vector<float> bad_logits(sel_vocab, 0.25f);
        bad_logits[11] = std::numeric_limits<float>::quiet_NaN();
        const auto sel_logits = graph.upload(std::as_bytes(std::span(bad_logits)), "dflash2 sel logits");
        const auto part_ids = graph.zeros(1u * 8u * 16u * 4u, "dflash2 sel part_ids");
        const auto part_vals = graph.zeros(1u * 8u * 16u * 4u, "dflash2 sel part_vals");
        const auto sel_out = graph.zeros(7ull * 16ull * 8ull + 7ull * 256ull * 4ull, "dflash2 sel_out");
        const auto sel_hid = graph.zeros(1u * 256u * 4u, "dflash2 sel_hid");
        const auto pred_cb = graph.zeros(uint64_t(sel_vocab) * 256ull * 2ull, "dflash2 pred_cb");
        const auto succ_cb = graph.zeros(uint64_t(sel_vocab) * 256ull * 2ull, "dflash2 succ_cb");
        rejects([&] {
            graph.begin();
            graph.dflash2_select(sel_logits, part_ids, part_vals, sel_out, sel_hid, pred_cb, succ_cb, 1u, sel_vocab, 0u);
            static_cast<void>(graph.submit());
        });

        std::vector<std::byte> bad_sel_bytes(7ull * 16ull * 8ull + 7ull * 256ull * 4ull, std::byte{0});
        for (uint32_t i = 0; i < 16u; ++i) {
            std::memcpy(bad_sel_bytes.data() + i * 4ull, &i, 4);
        }
        const float nan_val = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(bad_sel_bytes.data() + 7ull * 16ull * 4ull + 5ull * 4ull, &nan_val, 4);
        SamplingPolicy greedy{};
        SamplingRng rng(1ull);
        rejects([&] {
            static_cast<void>(dflash2_selector_proposals(bad_sel_bytes, 1u, sel_vocab, 0u, 0u, greedy, rng));
        });

        graph.release(succ_cb);
        graph.release(pred_cb);
        graph.release(sel_hid);
        graph.release(sel_out);
        graph.release(part_vals);
        graph.release(part_ids);
        graph.release(sel_logits);
        graph.release(bad_w_gpu);
        graph.release(out_gpu);
        graph.release(x_gpu);
    }
    graph.release(w_gpu);
}

void check_dflash2_nan_inf_propagation(const char* metallib) {
    MetalGraph graph(metallib, 32ull << 20);
    graph.prepare_dflash2();

    // 1. rsqrt(+Inf) in dflash2_conv_finish_rms_f32 must yield NaN, not 0.0f.
    {
        constexpr uint32_t rows = 8u, hidden = 2560u, group_size = 16u;
        const std::vector<float> ones_in(rows * hidden, 1.0f);
        const auto input = graph.upload(std::as_bytes(std::span(ones_in)), "dflash2 conv_rms in");
        const auto dynamic = graph.zeros(uint64_t(rows) * 4u * (hidden / group_size) * 4u, "dflash2 conv_rms dyn");
        const auto base = graph.zeros(4ull * hidden * 4u, "dflash2 conv_rms base");
        std::vector<float> huge_res(size_t(rows) * hidden, 1.0f);
        huge_res[0] = 1.0e20f; // sum(v^2) overflows FP32 to +Inf; rsqrt(+Inf) would be 0.0f without guard
        const auto residual = graph.upload(std::as_bytes(std::span(huge_res)), "dflash2 conv_rms res");
        const std::vector<float> ones_g(hidden, 1.0f);
        const auto gains = graph.upload(std::as_bytes(std::span(ones_g)), "dflash2 conv_rms gains");
        const auto res_out = graph.zeros(uint64_t(rows) * hidden * 4u, "dflash2 conv_rms res_out");
        const auto norm_out = graph.zeros(uint64_t(rows) * hidden * 4u, "dflash2 conv_rms norm_out");

        graph.begin();
        graph.dflash2_conv_finish_rms_f32(input, dynamic, base, residual, gains, res_out, norm_out, hidden, 1e-6f, rows, group_size);
        static_cast<void>(graph.submit());

        std::vector<float> got(size_t(rows) * hidden);
        graph.read(norm_out, std::as_writable_bytes(std::span(got)));
        check(std::isnan(got[0]), "dflash2_conv_finish_rms_f32 rsqrt(+Inf) must produce NaN, not 0.0f");

        graph.release(norm_out);
        graph.release(res_out);
        graph.release(gains);
        graph.release(residual);
        graph.release(base);
        graph.release(dynamic);
        graph.release(input);
    }

    // 2. NaN in attention scores must propagate via max_nan/simd_max_nan/inv_total instead of collapsing to 0.0f.
    {
        constexpr uint32_t cache_len = 8u, window = 2048u;
        std::vector<uint16_t> q_bf16(32ull * 8ull * 128ull, f32_to_bf16(0.5f));
        q_bf16[0] = 0x7fc0u; // NaN in BF16 for row 0 head 0
        const std::vector<uint16_t> rk_h(8ull * window * 128ull, f32_to_bf16(0.25f));
        const std::vector<uint16_t> rv_h(8ull * window * 128ull, f32_to_bf16(0.5f));
        const std::vector<uint16_t> qk_h(8ull * 8ull * 128ull, f32_to_bf16(0.25f));
        const std::vector<uint16_t> qv_h(8ull * 128ull * 8ull, f32_to_bf16(0.5f));
        const auto queries = graph.upload(std::as_bytes(std::span(q_bf16)), "dflash2 attn q");
        const auto ring_k = graph.upload(std::as_bytes(std::span(rk_h)), "dflash2 attn k");
        const auto ring_v = graph.upload(std::as_bytes(std::span(rv_h)), "dflash2 attn v");
        const auto qk = graph.upload(std::as_bytes(std::span(qk_h)), "dflash2 attn qk");
        const auto qv = graph.upload(std::as_bytes(std::span(qv_h)), "dflash2 attn qv");
        const auto partials = graph.zeros(8ull * 16ull * (32ull * 128ull + 64ull) * 4ull, "dflash2 attn part");
        const auto out = graph.zeros(8ull * 4096ull * 4ull, "dflash2 attn out");

        graph.begin();
        graph.dflash2_attention(queries, ring_k, ring_v, qk, qv, partials, out, cache_len, window);
        static_cast<void>(graph.submit());

        std::vector<float> got(8ull * 4096ull);
        graph.read(out, std::as_writable_bytes(std::span(got)));
        check(std::isnan(got[0]), "dflash2_attention must propagate NaN instead of collapsing to 0.0f");

        graph.release(out);
        graph.release(partials);
        graph.release(qv);
        graph.release(qk);
        graph.release(ring_v);
        graph.release(ring_k);
        graph.release(queries);
    }
}

void check_dflash2_verify_budget_save_restore() {
    DFlash2VerifyBudget budget;
    const std::array<float, DFlash2VerifyBudget::kPositions> probe_conf{
        0.92f, 0.81f, 0.72f, 0.58f, 0.44f, 0.31f, 0.19f};
    const DFlash2VerifyBudget initial;
    const auto initial_costs = initial.costs();
    const uint32_t initial_rows = initial.choose_verify_rows(probe_conf, 8u);

    for (uint32_t i = 0; i < DFlash2VerifyBudget::kRefitCycles; ++i) {
        const std::array<float, DFlash2VerifyBudget::kPositions> conf{
            0.90f - 0.01f * float(i % 4u), 0.78f, 0.65f, 0.52f, 0.40f, 0.30f, 0.20f};
        budget.observe(conf, 6u, (i % 5u), 8.5 + 0.25 * double(i % 7u), true);
    }
    check(budget.cycles() == DFlash2VerifyBudget::kRefitCycles, "verify budget cycles mismatch");
    check(budget.label_count() > 0, "verify budget expected recorded labels");

    const DFlash2VerifyBudget saved = budget;
    const auto saved_labels = saved.labels();
    const auto saved_costs = saved.costs();
    const uint32_t saved_rows = saved.choose_verify_rows(probe_conf, 8u);

    for (uint32_t i = 0; i < 65u; ++i) {
        const std::array<float, DFlash2VerifyBudget::kPositions> conf{
            0.25f, 0.20f, 0.18f, 0.15f, 0.12f, 0.10f, 0.08f};
        budget.observe(conf, 8u, 0u, 35.0 + double(i % 3u), true);
    }
    check(budget.cycles() != saved.cycles(), "mutated verify budget must advance cycles");
    check(budget.label_count() != saved.label_count(), "mutated verify budget must add labels");

    budget = saved;
    check(budget.cycles() == saved.cycles(), "restored verify budget cycles mismatch");
    check(budget.label_count() == saved.label_count(), "restored verify budget label_count mismatch");
    const auto restored_labels = budget.labels();
    check(restored_labels.size() == saved_labels.size(), "restored verify budget labels size mismatch");
    for (size_t i = 0; i < restored_labels.size(); ++i) {
        check(restored_labels[i].position == saved_labels[i].position &&
              restored_labels[i].feature == saved_labels[i].feature &&
              restored_labels[i].accepted == saved_labels[i].accepted,
              "restored verify budget labels element mismatch");
    }
    check(budget.acceptance().slope == saved.acceptance().slope &&
          budget.acceptance().intercepts == saved.acceptance().intercepts,
          "restored verify budget acceptance mismatch");
    check(budget.costs() == saved_costs, "restored verify budget costs mismatch");
    check(budget.choose_verify_rows(probe_conf, 8u) == saved_rows,
          "restored verify budget choose_verify_rows mismatch");

    budget.reset();
    check(budget.cycles() == 0, "reset verify budget cycles must be 0");
    check(budget.label_count() == 0 && budget.labels().empty(), "reset verify budget labels must be empty");
    check(budget.acceptance().slope == initial.acceptance().slope &&
          budget.acceptance().intercepts == initial.acceptance().intercepts,
          "reset verify budget acceptance must match initial");
    check(budget.costs() == initial_costs, "reset verify budget costs must match initial");
    check(budget.choose_verify_rows(probe_conf, 8u) == initial_rows,
          "reset verify budget choose_verify_rows must match initial");

    // Verify move construction and move assignment clear next_label_ and cycles_ on moved-from objects.
    DFlash2VerifyBudget move_src = saved;
    for (uint32_t i = 0; i < DFlash2VerifyBudget::kLabels + 10u; ++i) {
        move_src.observe(probe_conf, 4u, 2u, 10.0, true);
    }
    const auto pre_move_labels = move_src.labels();
    const auto pre_move_costs = move_src.costs();
    const uint64_t pre_move_cycles = move_src.cycles();
    const uint32_t pre_move_rows = move_src.choose_verify_rows(probe_conf, 8u);

    DFlash2VerifyBudget move_dst(std::move(move_src));
    check(move_dst.cycles() == pre_move_cycles && move_dst.label_count() == pre_move_labels.size(),
          "move-constructed verify budget lost state");
    check(move_dst.costs() == pre_move_costs && move_dst.choose_verify_rows(probe_conf, 8u) == pre_move_rows,
          "move-constructed verify budget costs/rows mismatch");
    check(move_src.label_count() == 0 && move_src.labels().empty() && move_src.cycles() == 0,
          "moved-from verify budget must reset next_label_ and cycles_");
    move_src.observe(probe_conf, 4u, 1u, 12.0, true);
    check(move_src.cycles() == 1 && move_src.label_count() == 2u,
          "moved-from verify budget must remain safely reusable");

    DFlash2VerifyBudget move_assign_dst;
    move_assign_dst = std::move(move_dst);
    check(move_assign_dst.cycles() == pre_move_cycles && move_assign_dst.label_count() == pre_move_labels.size(),
          "move-assigned verify budget lost state");
    check(move_dst.label_count() == 0 && move_dst.labels().empty() && move_dst.cycles() == 0,
          "move-assigned source verify budget must reset next_label_ and cycles_");
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: test-dflash2_context_stage METALLIB");
        check_dflash2_verify_budget_save_restore();
        check_dflash2_bf16_kernel(argv[1]);
        check_dflash2_nan_inf_propagation(argv[1]);
        MetalGraph graph(argv[1], 1ull << 20);
        const auto initial_bytes = graph.memory().live_bytes;
        constexpr uint32_t width = 32;
        const auto source = graph.zeros(8 * width * sizeof(float), "generated target tap");
        const auto owned = graph.zeros(source.bytes, "generated owned pending context");
        DFlash2ContextStage stage;
        stage.prepare(owned, width);
        rejects([&] { stage.prepare(owned, width); });
        const auto allocations = graph.memory().allocations;
        std::array<float, 8 * width> original{}, replacement{}, actual{};
        for (size_t i = 0; i < original.size(); ++i) { original[i] = float(i + 1); replacement[i] = -float(i + 1); }
        graph.write(source, std::as_bytes(std::span(original)));
        stage.stage(graph, source, 3, 100);
        // An ordinary target step reuses its tap before pending draft priming runs.
        graph.write(source, std::as_bytes(std::span(replacement)));
        graph.read(stage.storage(), std::as_writable_bytes(std::span(actual)));
        check(std::equal(original.begin(), original.begin() + 3 * width, actual.begin()), "target tap overwrite corrupted pending context");
        rejects([&] { stage.stage(graph, source, 1, 103); });
        stage.truncate(103);
        check(stage.rows() == 3 && stage.start() == 100, "same-frontier truncate lost pending context");
        stage.truncate(102);
        check(stage.rows() == 2 && stage.start() == 100, "partial truncate lost surviving pending prefix");
        stage.truncate(100);
        check(!stage.rows(), "truncate retained discarded rows");
        rejects([&] { stage.stage(graph, source, 9, 100); });
        rejects([&] { stage.stage(graph, source, 1, UINT32_MAX); });
        replacement[0] = std::numeric_limits<float>::quiet_NaN();
        graph.write(source, std::as_bytes(std::span(replacement)));
        rejects([&] { stage.stage(graph, source, 1, 100); });
        check(!stage.rows(), "nonfinite refusal published pending context");
        graph.write(source, std::as_bytes(std::span(original)));
        stage.stage(graph, source, 8, 0);
        check(stage.rows() == 8 && !stage.start(), "full pending block failed");
        // F17 lazy flush coherence & idempotency contract: staging pending rows preserves exact tap payload across
        // source reuse until flush/clear consumes them, and a second clear/truncate at frontier is a no-op.
        std::array<float, 8 * width> flushed{};
        graph.read(stage.storage(), std::as_writable_bytes(std::span(flushed)));
        check(std::equal(original.begin(), original.end(), flushed.begin()), "pending context staging altered payload before lazy flush");
        stage.clear();
        check(!stage.rows() && graph.memory().allocations == allocations, "pending reset changed allocation or retained rows");
        stage.clear();
        stage.truncate(0);
        check(!stage.rows() && graph.memory().allocations == allocations, "idempotent lazy flush/clear altered allocations");
        graph.release(owned); graph.release(source);
        check(graph.memory().live_bytes == initial_bytes, "pending fixture leaked GPU storage");
        std::cout << "DFlash2 pending context: owned copy, tap overwrite, truncate, reset, idempotent flush-clear and refusal passed; generated buffers only\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
