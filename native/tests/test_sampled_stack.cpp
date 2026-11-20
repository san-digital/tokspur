// Generated small owned assets only; no original model/weights are opened.
#define main inherited_stack_entry_not_run
#include "test_stack.cpp"
#undef main
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
#define main inherited_sampled_entry_not_run
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
#include <CommonCrypto/CommonDigest.h>
#endif

#ifndef TOKSPUR_PRIVATE_VERIFY_EIGHT
#error "sampled stack test requires the private V8 build"
#endif

namespace {
uint32_t nonfinite_output_refusals = 0;
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
uint32_t cost_width3_cases = 0, production_three_refusals = 0, cost_three_refusals = 0;
#endif
bool same_bytes(std::span<const float> a, std::span<const float> b) {
    const auto x = std::as_bytes(a), y = std::as_bytes(b);
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin());
}
struct SampledFrame {
    std::vector<float> logits, hc;
    std::vector<std::byte> state;
};
struct Select {
    TextStack* target;
    std::span<const SampledFrame> expected;
    uint32_t commit, calls = 0;
    bool throwing = false;
    static uint32_t choose(void* raw, const StackSampledVerificationRows& rows) {
        auto& s = *static_cast<Select*>(raw); ++s.calls;
        require(rows.count == s.expected.size() && rows.residual_width == s.expected[0].hc.size() &&
            rows.final_hc_rows.size() == uint64_t(rows.count) * rows.residual_width, "selector HC geometry");
        for (uint32_t i = 0; i < rows.count; ++i) {
            require(same_bytes(rows.logits[i], s.expected[i].logits), "selector target row differs from ordinary");
            require(same_bytes(rows.final_hc_rows.subspan(uint64_t(i) * rows.residual_width, rows.residual_width),
                s.expected[i].hc), "selector full-stream HC row differs from ordinary");
        }
        for (size_t i = rows.count; i < rows.logits.size(); ++i)
            require(rows.logits[i].empty(), "unused target row is not empty");
        reject([&] { static_cast<void>(s.target->position()); });
        reject([&] { s.target->reset(); });
        reject([&] { static_cast<void>(s.target->last_final_hc_rows()); });
        if (s.throwing) throw FormatError("selector deliberately refused");
        return s.commit;
    }
};

void sampled_stack(const char* library, bool reuse, [[maybe_unused]] bool concurrent, bool host_views = false) {
    MetalGraph graph(library, 128ull << 20);
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    graph.configure_q8_three_rows(true);
#endif
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
    graph.configure_concurrent_dispatch(concurrent);
#endif
    graph.configure_small_gemv(16); graph.configure_block_gemv(3);
#ifdef TOKSPUR_PRIVATE_HC_EXACT
    graph.configure_hc_exact(HcExactMode::combined);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_STAGE_V8
    graph.configure_qsa_stage_v8(true);
#endif
#ifdef TOKSPUR_PRIVATE_QSA_CHANNEL_CHRONOLOGICAL
    graph.configure_qsa_channel_chronological(true);
#endif
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    constexpr uint32_t sampled_hidden = 32;
#else
    constexpr uint32_t sampled_hidden = 8;
#endif
    const StackShape shape{{4, 32, 3}, {32, 1, 2, 128, 32, 4},
        {32, 4, 2, 16, 2, 4, 4, 4, 8, 64}, {32, sampled_hidden, 4, 2}, {4, 32, 4, 4, 3}, 13, 1};
    const StackPolicy policy{{0.5f, 1e-6f, L2Mode::add_squared_epsilon, 1e-6f},
        {QueryPacking::head_query_gate, RotaryPairs::split_half, 1e-6f}, RouterMode::softmax, true,
        {HashOrder::newest_first, NgramBoundary::literal, true}};
    const NgramSpec spec{3, 2, 1, 13, 11, 12, {3, 5, 7}, {0, 7, 18, 31}, {7, 11, 13, 17}};
    Assets assets{graph, std::mt19937(0x73616d70), {}, {}, {}};
    auto f = fixture(assets, shape);
    std::deque<std::vector<std::byte>> encoded;
    const auto q8 = [&](uint32_t columns, uint32_t rows, bool immutable = true) {
        encoded.emplace_back(uint64_t(columns / 32) * rows * 34); auto& bytes = encoded.back();
        for (uint32_t row = 0; row < rows; ++row) for (uint32_t block = 0; block < columns / 32; ++block) {
            const size_t at = (size_t(row) * (columns / 32) + block) * 34;
            const uint16_t scale = 0x1403; std::memcpy(bytes.data() + at, &scale, 2);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const int8_t value = int8_t(int((row * 17 + block * 11 + lane * 7) % 31) - 15);
                std::memcpy(bytes.data() + at + 2 + lane, &value, 1);
            }
        }
        const auto view = graph.upload(bytes, "generated sampled-stack Q8", immutable); assets.views.push_back(view);
        return GpuMatrix{view, 8};
    };
    const auto bf16 = [&](uint32_t columns, uint32_t rows) {
        const auto values = assets.values(uint64_t(columns) * rows); encoded.emplace_back(values.size() * 2);
        auto& bytes = encoded.back();
        for (size_t i = 0; i < values.size(); ++i) {
            const auto value = finite_bf16(values[i]); std::memcpy(bytes.data() + i * 2, &value, 2);
        }
        const auto view = graph.upload(bytes, "generated sampled-stack BF16"); assets.views.push_back(view);
        return GpuMatrix{view, 30};
    };
    for (auto& layer : f.weights.layers) {
        if (auto* w = std::get_if<RecurrentWeights>(&layer)) {
            w->attention.qkv = q8(32, uint32_t(gdn_qkv_count(shape.recurrent)));
            w->attention.gate = q8(32, 64); w->attention.output = q8(64, 32);
        } else {
            auto& a = std::get<AttentionLayerWeights>(layer).attention;
            a.query_gate = q8(32, 128); a.key = q8(32, 32); a.value = q8(32, 32); a.output = q8(64, 32);
            a.index_query = bf16(32, 8); a.index_key = bf16(32, 4);
        }
    }
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    for(auto& layer:f.weights.layers)std::visit([&](auto& w){w.ffn.shared_gate=q8(32,sampled_hidden);
        w.ffn.shared_up=q8(32,sampled_hidden);w.ffn.shared_down=q8(sampled_hidden,32);},layer);
#endif
    // This generated output matrix is writable only for GPU nonfinite failure injection below.
    f.weights.output = q8(32, shape.vocabulary, false);
    const auto weights_only = graph.memory();
    {
        TextStack target(graph, f.weights, shape, policy, spec, StackExecution::direct, 8, reuse);
        TextStack serial(graph, f.weights, shape, policy, spec);
        require(!target.sampled_logit_views(), "sampled shared views enabled by default");
        target.configure_sampled_logit_views(host_views);
        require(target.sampled_logit_views() == host_views, "sampled shared view selector differs");
        require(!target.final_hc_tap_memory().prepared, "tap prepared by default");
        reject([&] { static_cast<void>(target.last_final_hc_rows()); });
        reject([&] { target.prepare_final_hc_tap(1); });
        reject([&] { target.prepare_sampled_verification2(32ull << 20); });
#ifdef TOKSPUR_PRIVATE_KNOWN_QSA_ROWS
        // Separate known-prefill resources remain prepared and enabled.
        // The new sampled preparation alone consents to their composition.
        target.prepare_known_qsa_rows(true);
#endif
        target.prepare_final_hc_tap(1ull << 20); serial.prepare_final_hc_tap(1ull << 20);
        const auto tap = target.final_hc_tap_memory();
        require(tap.prepared && tap.capacity == 8 && tap.residual_width == 128 && tap.host_charged_bytes <= (1ull << 20),
            "bounded full-four-stream host tap");
        reject([&] { target.prepare_final_hc_tap(1ull << 20); });
        reject([&] { target.prepare_checkpoints({1, 16ull << 20}); });
        target.prepare_sampled_verification2(32ull << 20
#ifdef TOKSPUR_PRIVATE_VERIFY_QSA_ROWS
            , {reuse, reuse ? QsaVerifyRowsMode::core_and_q8 : QsaVerifyRowsMode::off}
#endif
            );
        reject([&] { target.prepare_sampled_verification2(32ull << 20); });
        reject([&] { target.configure_sampled_logit_views(!host_views); });
        const auto resident = graph.memory();
        const auto ledger = [&] { const auto now = graph.memory(); require(now.live_bytes == resident.live_bytes &&
            now.allocations == resident.allocations, "sampled request changed prepared GPU ledger"); };
        const auto row = [&](uint32_t token) { std::vector<float> out(shape.residual.channels);
            for (size_t i = 0; i < out.size(); ++i) out[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16; return out; };
        const auto memory = [&](std::span<const uint64_t> ids) { std::vector<float> out(shape.memory.embedding);
            for (size_t i = 0; i < out.size(); ++i) out[i] = float(int(ids[i / spec.row_width] % 17) - 8) / 32; return out; };
        const auto advance = [&](TextStack& stack, uint32_t token) { const auto lookup = stack.lookup(token); ++batches;
            const auto out = stack.step(token, row(token), memory(lookup.rows));
            require(stack.last_final_hc_count() == 1 && stack.last_final_hc_residual().size() == 128, "ordinary HC tap geometry");
            auto state = stack.accepted_snapshot();
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
            std::array<std::byte, 4096> scratch{};
            const auto actual = stack.accepted_fingerprint(scratch);
            std::array<unsigned char, 32> expected{};
            CC_SHA256(state.data(), CC_LONG(state.size()), expected.data());
            require(actual.bytes == state.size() && actual.sha256 == expected, "streaming target fingerprint differs from snapshot");
#endif
            return SampledFrame{out.logits, {stack.last_final_hc_residual().begin(), stack.last_final_hc_residual().end()}, std::move(state)}; };
        const auto seed = [&](TextStack& stack, uint32_t n) { stack.reset(); SampledFrame out;
            require(!stack.last_final_hc_count() && stack.last_final_hc_rows().empty(), "reset retained HC history");
            reject([&] { static_cast<void>(stack.last_final_hc_residual()); });
            for (uint32_t i = 0; i < n; ++i) out = advance(stack, i % shape.vocabulary); return out; };
        struct Inputs { std::vector<uint32_t> tokens; std::vector<float> rows, memory; };
        const auto inputs = [&](std::span<const uint32_t> ids) {
            Inputs in{{ids.begin(), ids.end()}, {}, {}};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            const auto lookups = target.sampled_verification8_lookups(ids);
#elif defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
            const auto lookups = target.sampled_verification_cost_lookups(ids);
#else
            const auto lookups = target.sampled_verification_lookups(ids);
#endif
            for (size_t i = 0; i < ids.size(); ++i) {
                const auto r = row(ids[i]), m = memory(lookups[i].rows);
                in.rows.insert(in.rows.end(), r.begin(), r.end()); in.memory.insert(in.memory.end(), m.begin(), m.end());
            }
            return in;
        };
        for (uint32_t prefix : {1, 3, 4, 7}) {
            const auto base = seed(serial, prefix);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
            constexpr uint32_t limit = 8;
#elif defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST)
            constexpr uint32_t limit = 3;
#else
            constexpr uint32_t limit = 2;
#endif
            std::array<uint32_t, limit> ids{}; std::array<SampledFrame, limit> frames;
            auto logits = base.logits;
            for (uint32_t i = 0; i < limit; ++i) {
                ids[i] = uint32_t((std::max_element(logits.begin(), logits.end()) - logits.begin() + 1) % shape.vocabulary);
                frames[i] = advance(serial, ids[i]); logits = frames[i].logits;
            }
            for (uint32_t n = 1; n <= limit; ++n) for (uint32_t k = 1; k <= n; ++k) {
                const auto target_base = seed(target, prefix); require(target_base.state == base.state, "sampled base differs");
                const auto in = inputs(std::span(ids).first(n)); Select selection{&target, std::span(frames).first(n), k};
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
                const auto out =
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                    n > 3 ? target.verify_sampled8(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection}) :
#endif
                    n == 3 ?
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                    target.verify_sampled_depth2(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection}) :
#else
                    target.verify_sampled_cost3(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection}) :
#endif
                    target.verify_sampled2(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection});
                cost_width3_cases += n == 3;
#else
                const auto out = target.verify_sampled2(in.tokens, in.rows, in.memory, prefix, {Select::choose, &selection});
#endif
                batches += 1 + bool(out.recovery.dispatches);
                require(selection.calls == 1 && out.position == prefix && out.committed == k && out.evaluated.count == n &&
                    target.position() == prefix + k && same_bytes(out.logits, frames[k-1].logits) &&
                    target.accepted_snapshot() == frames[k-1].state, "arbitrary sampled prefix/state differs from ordinary");
                require(target.last_final_hc_count() == k && same_bytes(target.last_final_hc_residual(), frames[k-1].hc) &&
                    same_bytes(out.evaluated.final_hc_rows.last(128), frames[n-1].hc), "committed/evaluated HC confusion");
                static_cast<void>(seed(serial, prefix));
                for (uint32_t i = 0; i < k; ++i) static_cast<void>(advance(serial, ids[i]));
                const auto a = advance(target, 6), b = advance(serial, 6);
                require(a.state == b.state && same_bytes(a.logits,b.logits) && same_bytes(a.hc,b.hc), "rejected tail contaminated append");
                ledger();
            }
        }
        // Known-input prefill must retain every row, including the first row a
        // head needs when priming from a prior target HC residual.
        static_cast<void>(seed(target, 1)); static_cast<void>(seed(serial, 1));
        const std::array<uint32_t, 8> known{2, 3, 4, 5, 6, 7, 8, 9};
        const auto lookups = target.lookups(known); std::vector<float> kr, km, expected_hc; SampledFrame last;
        for (size_t i = 0; i < known.size(); ++i) {
            const auto r = row(known[i]), m = memory(lookups[i].rows);
            kr.insert(kr.end(),r.begin(),r.end()); km.insert(km.end(),m.begin(),m.end());
            last = advance(serial,known[i]); expected_hc.insert(expected_hc.end(),last.hc.begin(),last.hc.end());
        }
        const auto known_out = target.step_known(known,kr,km); ++batches;
        require(target.last_final_hc_count()==8 && same_bytes(target.last_final_hc_rows(),expected_hc) &&
            same_bytes(known_out.logits,last.logits) && target.accepted_snapshot()==last.state,"known prefill lost/reordered HC rows");
        for (uint32_t mode = 0; mode < 7; ++mode) {
            const auto base = seed(target,3); static_cast<void>(seed(serial,3));
            const std::array<uint32_t,2> ids{5,8}; std::array<SampledFrame,2> frames{advance(serial,5),advance(serial,8)};
            const auto in=inputs(ids); Select selection{&target,frames,mode==0?0u:mode==1?3u:1u}; selection.throwing=mode==2;
            struct Cancel { uint32_t at,polls=0; static bool poll(void* p) { auto& c=*static_cast<Cancel*>(p); return ++c.polls==c.at; } };
            Cancel cancel{mode>=3?mode-2:99};
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens,in.rows,in.memory,3,{Select::choose,&selection},{Cancel::poll,&cancel})); });
            require(target.position()==3 && target.accepted_snapshot()==base.state &&
                target.last_final_hc_count()==1 && same_bytes(target.last_final_hc_residual(),base.hc),"refused sampled transaction published state/tap");
            ledger();
        }
        {
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            const std::array<uint32_t, 2> ids{5, 8};
            std::array<SampledFrame, 2> frames{advance(serial, 5), advance(serial, 8)};
            const auto in = inputs(ids); Select selection{&target, frames, 1};
            const auto after_selection = [](void* value) { return static_cast<Select*>(value)->calls != 0; };
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection}, {after_selection, &selection})); });
            require(selection.calls == 1 && target.position() == 3 && target.accepted_snapshot() == base.state && same_bytes(target.last_final_hc_residual(), base.hc), "post-selection cancellation published state or HC");
            selection.calls = 0;
            const auto retry = target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection});
            require(retry.committed == 1 && same_bytes(retry.logits, frames[0].logits) && target.accepted_snapshot() == frames[0].state, "cancelled shared view corrupted retry");
            ledger();
        }
        for (const uint16_t nonfinite_scale : {uint16_t(0x7e00), uint16_t(0x7c00)}) {
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            const std::array<uint32_t, 2> ids{5, 8};
            std::array<SampledFrame, 2> frames{advance(serial, 5), advance(serial, 8)};
            const auto in = inputs(ids); Select selection{&target, frames, 1};
            const auto saved_output = graph.read(f.weights.output.view);
            auto poisoned_output = saved_output;
            std::memcpy(poisoned_output.data(), &nonfinite_scale, sizeof(nonfinite_scale));
            graph.write(f.weights.output.view, poisoned_output);
            // Inputs remain finite. The evaluated output projection must reject its NaN/Inf before the selector or state publication.
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection})); });
            require(selection.calls == 0 && target.position() == 3 && target.accepted_snapshot() == base.state && same_bytes(target.last_final_hc_residual(), base.hc), "nonfinite evaluated output published state, HC or selector input");
            ++nonfinite_output_refusals;
            graph.write(f.weights.output.view, saved_output);
            const auto retry = target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {Select::choose, &selection});
            require(selection.calls == 1 && retry.committed == 1 && same_bytes(retry.logits, frames[0].logits) && target.accepted_snapshot() == frames[0].state && same_bytes(target.last_final_hc_residual(), frames[0].hc), "nonfinite output refusal corrupted valid retry");
            ledger();
        }
        {
            // Host-tap mode: failed step following committed step invalidates aliased GPU view while keeping host HC row.
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            require(!target.gpu_final_hc_tap() && target.has_last_final_hc_gpu(), "committed host-tap step missing GPU view");
            std::vector<float> gpu_hc(128);
            graph.read(target.last_final_hc_gpu_residual(), std::as_writable_bytes(std::span(gpu_hc)));
            require(same_bytes(gpu_hc, base.hc), "committed step GPU HC residual differs from host tap");
            std::vector<std::byte> expected_ssd(size_t(target.ssd_checkpoint_payload_bytes()));
            require(target.read_ssd_checkpoint_span(0, expected_ssd), "baseline SSD checkpoint span read failed");
            const auto lookup5 = target.lookup(5);
            const auto r5 = row(5), m5 = memory(lookup5.rows);
            reject([&] { static_cast<void>(target.step(shape.vocabulary, r5, m5)); });
            reject([&] { static_cast<void>(target.step(5, std::span(r5).first(r5.size() - 1), m5)); });
            require(target.has_last_final_hc_gpu(), "pre-staging step refusal invalidated committed GPU HC view");
            const auto saved_output = graph.read(f.weights.output.view);
            auto poisoned_output = saved_output;
            const uint16_t nan_scale = 0x7e00;
            std::memcpy(poisoned_output.data(), &nan_scale, sizeof(nan_scale));
            graph.write(f.weights.output.view, poisoned_output);
            reject([&] { static_cast<void>(target.step(5, r5, m5)); });
            graph.write(f.weights.output.view, saved_output);
            require(target.position() == 3 && target.accepted_snapshot() == base.state &&
                !target.has_last_final_hc_gpu() && target.last_final_hc_count() == 1 &&
                same_bytes(target.last_final_hc_residual(), base.hc),
                "failed step left aliased GPU HC view valid or corrupted host HC tap");
            reject([&] { static_cast<void>(target.last_final_hc_gpu_rows()); });
            reject([&] { static_cast<void>(target.last_final_hc_gpu_residual()); });
            std::vector<std::byte> actual_ssd(expected_ssd.size());
            require(target.read_ssd_checkpoint_span(0, actual_ssd) && actual_ssd == expected_ssd,
                "SSD checkpoint span failed to read intact host HC row after failed step");

            // Host-tap mode: failed verify_sampled2 following committed verify_sampled2 invalidates aliased GPU view.
            static_cast<void>(seed(target, 2)); static_cast<void>(seed(serial, 2));
            const std::array<uint32_t, 2> ids_a{5, 8}, ids_b{3, 7};
            std::array<SampledFrame, 2> frames_a{advance(serial, 5), advance(serial, 8)};
            const auto in_a = inputs(ids_a); Select sel_a{&target, frames_a, 2};
            const auto committed_a = target.verify_sampled2(in_a.tokens, in_a.rows, in_a.memory, 2, {Select::choose, &sel_a});
            ++batches;
            require(committed_a.committed == 2 && target.has_last_final_hc_gpu() &&
                same_bytes(target.last_final_hc_residual(), frames_a[1].hc), "committed verify_sampled2 missing GPU/host HC");
            std::vector<std::byte> verify_ssd(size_t(target.ssd_checkpoint_payload_bytes()));
            require(target.read_ssd_checkpoint_span(0, verify_ssd), "verify_sampled2 SSD checkpoint read failed");
            std::array<SampledFrame, 2> frames_b{advance(serial, 3), advance(serial, 7)};
            const auto in_b = inputs(ids_b); Select sel_throw{&target, frames_b, 1}; sel_throw.throwing = true;
            reject([&] { static_cast<void>(target.verify_sampled2(in_b.tokens, in_b.rows, in_b.memory, 4, {Select::choose, &sel_throw})); });
            require(!target.has_last_final_hc_gpu() && target.last_final_hc_count() == 2 &&
                same_bytes(target.last_final_hc_residual(), frames_a[1].hc),
                "failed verify_sampled2 left aliased GPU HC view valid or corrupted host HC tap");
            reject([&] { static_cast<void>(target.last_final_hc_gpu_rows()); });
            reject([&] { static_cast<void>(target.last_final_hc_gpu_residual()); });
            std::vector<std::byte> verify_ssd_after(verify_ssd.size());
            require(target.read_ssd_checkpoint_span(0, verify_ssd_after) && verify_ssd_after == verify_ssd,
                "SSD checkpoint span changed after failed host-tap verify_sampled2");
            ledger();
        }
        {
            target.reset();
            target.configure_gpu_final_hc_tap(true);
            require(target.gpu_final_hc_tap(), "GPU final HC tap toggle failed");
            const auto base = seed(target, 3); static_cast<void>(seed(serial, 3));
            (void)base;
            const std::array<uint32_t, 2> ids{5, 8};
            std::array<SampledFrame, 2> frames{advance(serial, 5), advance(serial, 8)};
            const auto in = inputs(ids);
            struct GpuTapSelect {
                uint32_t commit = 2;
                uint32_t calls = 0;
                bool throwing = false;
                bool saw_gpu_only = false;
                static uint32_t choose(void* ctx, const StackSampledVerificationRows& rows) {
                    auto& self = *static_cast<GpuTapSelect*>(ctx);
                    ++self.calls;
                    self.saw_gpu_only = rows.count == 2 && rows.final_hc_rows.empty() &&
                        rows.final_hc_gpu_rows.owner != 0 && rows.final_hc_gpu_rows.bytes == 2 * 128 * sizeof(float);
                    if (self.throwing) throw FormatError("GPU tap selector deliberately refused");
                    return self.commit;
                }
            };
            GpuTapSelect gpu_sel{};
            const auto res = target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {GpuTapSelect::choose, &gpu_sel});
            ++batches;
            require(gpu_sel.saw_gpu_only && res.evaluated.final_hc_rows.empty() && res.evaluated.final_hc_gpu_rows.owner != 0 &&
                target.has_last_final_hc_gpu(), "GPU final HC tap did not suppress eager host copy");
            // Keep final_hc_host_valid == false here and verify pre-staging refusals preserve GPU backing.
            std::vector<float> committed_gpu_hc(128);
            graph.read(target.last_final_hc_gpu_residual(), std::as_writable_bytes(std::span(committed_gpu_hc)));
            require(same_bytes(committed_gpu_hc, frames[1].hc), "GPU final HC tap committed residual differs");

            const std::array<uint32_t, 2> next_ids{3, 7};
            std::array<SampledFrame, 2> next_frames{advance(serial, 3), advance(serial, 7)};
            const auto next_in = inputs(next_ids);
            const std::array<uint32_t, 2> bad_ids{shape.vocabulary, 7};
            reject([&] { static_cast<void>(target.verify_sampled2(bad_ids, next_in.rows, next_in.memory, 5, {GpuTapSelect::choose, &gpu_sel})); });
            reject([&] { static_cast<void>(target.verify_sampled2(next_in.tokens, std::span(next_in.rows).first(next_in.rows.size() - 1), next_in.memory, 5, {GpuTapSelect::choose, &gpu_sel})); });
            reject([&] { static_cast<void>(target.verify_sampled2(next_in.tokens, next_in.rows, next_in.memory, 4, {GpuTapSelect::choose, &gpu_sel})); });
            struct CancelOnce { bool done = false; static bool poll(void* p) { auto& c = *static_cast<CancelOnce*>(p); if (!c.done) { c.done = true; return true; } return false; } };
            CancelOnce pre_cancel{};
            reject([&] { static_cast<void>(target.verify_sampled2(next_in.tokens, next_in.rows, next_in.memory, 5, {GpuTapSelect::choose, &gpu_sel}, {CancelOnce::poll, &pre_cancel})); });
            // verify_greedy8 k=0 mismatch before staging must also preserve committed GPU HC view.
            const std::array<uint32_t, 2> g8_mismatch_ids{uint32_t((std::max_element(frames[1].logits.begin(), frames[1].logits.end()) - frames[1].logits.begin() + 1) % shape.vocabulary), 0u};
            const auto g8_mm_in = inputs(g8_mismatch_ids);
            const auto g8_mm = target.verify_greedy8(g8_mm_in.tokens, g8_mm_in.rows, g8_mm_in.memory, 5, frames[1].logits, 2);
            require(g8_mm.accepted == 0 && target.has_last_final_hc_gpu() && target.last_final_hc_count() == 2,
                "pre-staging refusal or early verify_greedy8 exit invalidated GPU final HC tap");
            std::fill(committed_gpu_hc.begin(), committed_gpu_hc.end(), 0.0f);
            graph.read(target.last_final_hc_gpu_residual(), std::as_writable_bytes(std::span(committed_gpu_hc)));
            require(same_bytes(committed_gpu_hc, frames[1].hc), "pre-staging refusal corrupted GPU final HC tap bytes");

            // 1a) Post-staging failure in GPU-tap mode via selector throw (finite candidate rows overwrote v.streams/v.second).
            GpuTapSelect throw_sel{2, 0, true, false};
            reject([&] { static_cast<void>(target.verify_sampled2(next_in.tokens, next_in.rows, next_in.memory, 5, {GpuTapSelect::choose, &throw_sel})); });
            require(throw_sel.calls == 1 && throw_sel.saw_gpu_only && target.position() == 5 &&
                !target.has_last_final_hc_gpu() && target.last_final_hc_count() == 0 && target.last_final_hc_rows().empty(),
                "selector throw after staging in GPU-tap mode retained aliased final HC tap");
            reject([&] { static_cast<void>(target.last_final_hc_residual()); });
            reject([&] { static_cast<void>(target.last_final_hc_gpu_rows()); });
            reject([&] { static_cast<void>(target.last_final_hc_gpu_residual()); });
            std::vector<std::byte> ssd_buf(size_t(target.ssd_checkpoint_payload_bytes()));
            require(!target.read_ssd_checkpoint_span(0, ssd_buf),
                "SSD checkpoint span succeeded with invalidated GPU-only final HC tap after selector throw");

            // Valid retry restores committed GPU tap, then post-selection cancellation invalidates it again cleanly.
            GpuTapSelect commit1_sel{1, 0, false, false};
            const auto retry1 = target.verify_sampled2(next_in.tokens, next_in.rows, next_in.memory, 5, {GpuTapSelect::choose, &commit1_sel});
            batches += 1 + bool(retry1.recovery.dispatches);
            require(retry1.committed == 1 && target.position() == 6 && target.has_last_final_hc_gpu() && target.last_final_hc_count() == 1,
                "GPU-tap retry after selector throw did not restore GPU HC tap");
            const std::array<uint32_t, 2> after_ids{7, 2};
            const auto after_in = inputs(after_ids);
            GpuTapSelect post_cancel_sel{1, 0, false, false};
            const auto cancel_after_choose = [](void* value) { return static_cast<GpuTapSelect*>(value)->calls != 0; };
            reject([&] { static_cast<void>(target.verify_sampled2(after_in.tokens, after_in.rows, after_in.memory, 6,
                {GpuTapSelect::choose, &post_cancel_sel}, {cancel_after_choose, &post_cancel_sel})); });
            require(post_cancel_sel.calls == 1 && target.position() == 6 && !target.has_last_final_hc_gpu() &&
                target.last_final_hc_count() == 0 && target.last_final_hc_rows().empty(),
                "post-selection cancellation in GPU-tap mode retained aliased final HC tap");
            reject([&] { static_cast<void>(target.last_final_hc_residual()); });
            reject([&] { static_cast<void>(target.last_final_hc_gpu_residual()); });
            ssd_buf.resize(size_t(target.ssd_checkpoint_payload_bytes()));
            require(!target.read_ssd_checkpoint_span(0, ssd_buf),
                "SSD checkpoint span succeeded after post-selection cancellation in GPU-tap mode");

            // 1b) Post-staging failure in GPU-tap mode via poisoned nonfinite output weights (NaN and Inf) + valid retry.
            const auto saved_output = graph.read(f.weights.output.view);
            for (const uint16_t nonfinite_scale : {uint16_t(0x7e00), uint16_t(0x7c00)}) {
                static_cast<void>(seed(target, 3));
                GpuTapSelect setup_sel{2, 0, false, false};
                static_cast<void>(target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {GpuTapSelect::choose, &setup_sel}));
                ++batches;
                require(target.has_last_final_hc_gpu() && target.last_final_hc_count() == 2, "GPU-tap setup missing GPU HC");
                auto poisoned_output = saved_output;
                std::memcpy(poisoned_output.data(), &nonfinite_scale, sizeof(nonfinite_scale));
                graph.write(f.weights.output.view, poisoned_output);
                GpuTapSelect poisoned_sel{2, 0, false, false};
                reject([&] { static_cast<void>(target.verify_sampled2(next_in.tokens, next_in.rows, next_in.memory, 5, {GpuTapSelect::choose, &poisoned_sel})); });
                graph.write(f.weights.output.view, saved_output);
                require(poisoned_sel.calls == 0 && target.position() == 5 && !target.has_last_final_hc_gpu() &&
                    target.last_final_hc_count() == 0 && target.last_final_hc_rows().empty(),
                    "nonfinite verify_sampled2 in GPU-tap mode retained aliased final HC tap");
                reject([&] { static_cast<void>(target.last_final_hc_residual()); });
                reject([&] { static_cast<void>(target.last_final_hc_gpu_rows()); });
                reject([&] { static_cast<void>(target.last_final_hc_gpu_residual()); });
                ssd_buf.resize(size_t(target.ssd_checkpoint_payload_bytes()));
                require(!target.read_ssd_checkpoint_span(0, ssd_buf),
                    "SSD checkpoint span succeeded after nonfinite verify_sampled2 invalidated GPU tap");

                GpuTapSelect retry_sel{2, 0, false, false};
                const auto ok_retry = target.verify_sampled2(next_in.tokens, next_in.rows, next_in.memory, 5, {GpuTapSelect::choose, &retry_sel});
                ++batches;
                require(ok_retry.committed == 2 && target.position() == 7 && target.has_last_final_hc_gpu() &&
                    target.last_final_hc_count() == 2 && same_bytes(target.last_final_hc_residual(), next_frames[1].hc) &&
                    same_bytes(target.last_final_hc_rows().first(next_frames[0].hc.size()), next_frames[0].hc),
                    "GPU final HC tap retry or lazy host materialisation differs after nonfinite output refusal");
                ssd_buf.resize(size_t(target.ssd_checkpoint_payload_bytes()));
                require(target.read_ssd_checkpoint_span(0, ssd_buf), "SSD checkpoint span failed after valid GPU-tap retry");
            }

            // 1c) Also verify_greedy8 post-staging failure invalidates GPU-tap final HC backing from verify_sampled2.
            {
                static_cast<void>(seed(target, 3));
                GpuTapSelect setup_sel{2, 0, false, false};
                static_cast<void>(target.verify_sampled2(in.tokens, in.rows, in.memory, 3, {GpuTapSelect::choose, &setup_sel}));
                ++batches;
                require(target.has_last_final_hc_gpu() && target.last_final_hc_count() == 2, "verify_greedy8 setup missing GPU HC");
                const uint32_t greedy_first = uint32_t(std::max_element(frames[1].logits.begin(), frames[1].logits.end()) - frames[1].logits.begin());
                const auto g8_in = inputs(std::array<uint32_t, 2>{greedy_first, 2u});
                auto poisoned_output = saved_output;
                const uint16_t nan_scale = 0x7e00;
                std::memcpy(poisoned_output.data(), &nan_scale, sizeof(nan_scale));
                graph.write(f.weights.output.view, poisoned_output);
                reject([&] { static_cast<void>(target.verify_greedy8(g8_in.tokens, g8_in.rows, g8_in.memory, 5, frames[1].logits, 2)); });
                graph.write(f.weights.output.view, saved_output);
                require(!target.has_last_final_hc_gpu() && target.last_final_hc_count() == 0 && target.last_final_hc_rows().empty(),
                    "failed verify_greedy8 retained aliased GPU final HC tap");
                ssd_buf.resize(size_t(target.ssd_checkpoint_payload_bytes()));
                require(!target.read_ssd_checkpoint_span(0, ssd_buf),
                    "SSD checkpoint span succeeded after failed verify_greedy8 invalidated GPU tap");
            }

            // 1d) step_known in GPU-tap mode followed by failed step_known (nonfinite output weights) + valid retry.
            {
                target.reset(); serial.reset();
                const std::array<uint32_t, 3> known_a{2, 3, 4};
                const auto lk_a = target.lookups(known_a);
                std::vector<float> kr_a, km_a;
                for (size_t i = 0; i < known_a.size(); ++i) {
                    const auto r = row(known_a[i]), m = memory(lk_a[i].rows);
                    kr_a.insert(kr_a.end(), r.begin(), r.end()); km_a.insert(km_a.end(), m.begin(), m.end());
                    static_cast<void>(advance(serial, known_a[i]));
                }
                static_cast<void>(target.step_known(known_a, kr_a, km_a));
                ++batches;
                require(target.position() == 3 && target.has_last_final_hc_gpu() && target.last_final_hc_count() == 3,
                    "GPU-tap step_known did not publish GPU HC view");

                const std::array<uint32_t, 2> known_b{5, 6};
                const auto lk_b = target.lookups(known_b);
                std::vector<float> kr_b, km_b; SampledFrame serial_last;
                for (size_t i = 0; i < known_b.size(); ++i) {
                    const auto r = row(known_b[i]), m = memory(lk_b[i].rows);
                    kr_b.insert(kr_b.end(), r.begin(), r.end()); km_b.insert(km_b.end(), m.begin(), m.end());
                    serial_last = advance(serial, known_b[i]);
                }
                const std::array<uint32_t, 2> bad_known{shape.vocabulary, 6};
                reject([&] { static_cast<void>(target.step_known(bad_known, kr_b, km_b)); });
                require(target.has_last_final_hc_gpu() && target.last_final_hc_count() == 3,
                    "pre-staging step_known refusal invalidated GPU HC view");

                auto poisoned_output = saved_output;
                const uint16_t nan_scale = 0x7e00;
                std::memcpy(poisoned_output.data(), &nan_scale, sizeof(nan_scale));
                graph.write(f.weights.output.view, poisoned_output);
                reject([&] { static_cast<void>(target.step_known(known_b, kr_b, km_b)); });
                graph.write(f.weights.output.view, saved_output);
                require(target.position() == 3 && !target.has_last_final_hc_gpu() &&
                    target.last_final_hc_count() == 0 && target.last_final_hc_rows().empty(),
                    "failed GPU-tap step_known retained aliased final HC tap");
                reject([&] { static_cast<void>(target.last_final_hc_residual()); });
                reject([&] { static_cast<void>(target.last_final_hc_gpu_residual()); });
                ssd_buf.resize(size_t(target.ssd_checkpoint_payload_bytes()));
                require(!target.read_ssd_checkpoint_span(0, ssd_buf),
                    "SSD checkpoint span succeeded after failed GPU-tap step_known");

                static_cast<void>(target.step_known(known_b, kr_b, km_b));
                ++batches;
                require(target.position() == 5 && target.has_last_final_hc_gpu() && target.last_final_hc_count() == 2 &&
                    same_bytes(target.last_final_hc_residual(), serial_last.hc),
                    "valid step_known retry after failed GPU-tap step_known diverged");
            }

            target.reset();
            target.configure_gpu_final_hc_tap(false);
            ledger();
        }
        reject([&] { static_cast<void>(target.sampled_verification_lookups({})); });
        reject([&] { static_cast<void>(target.sampled_verification_lookups(std::array<uint32_t,3>{1,2,3})); });
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        for (uint32_t width : {1u,2u}) {
            const std::array<uint32_t,2> ids{1,2};const auto in=inputs(std::span(ids).first(width));Select selector{&target,{},1};
            reject([&]{static_cast<void>(target.verify_sampled_depth2(in.tokens,in.rows,in.memory,target.position(),{Select::choose,&selector}));});
            require(!selector.calls,"wrong-width depth-two entry reached selector");
        }
#endif
        reject([&] { static_cast<void>(target.sampled_verification_cost_lookups({})); });
        reject([&] { static_cast<void>(target.sampled_verification_cost_lookups(std::array<uint32_t,4>{1,2,3,4})); });
        { const auto in = inputs(std::array<uint32_t,3>{1,2,3}); Select selection{&target, {}, 1};
            reject([&] { static_cast<void>(target.verify_sampled2(in.tokens,in.rows,in.memory,target.position(),{Select::choose,&selection})); });
            require(selection.calls == 0, "production width-three refusal reached selector"); ++production_three_refusals; }
        for (uint32_t mode : {0,1}) {
            const auto base = seed(target,3); static_cast<void>(seed(serial,3));
            const std::array<uint32_t,3> ids{5,8,4};
            std::array<SampledFrame,3> frames{advance(serial,5),advance(serial,8),advance(serial,4)};
            const auto in = inputs(ids); Select selection{&target,frames,4}; selection.throwing = mode == 1;
            reject([&] { static_cast<void>(
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
                target.verify_sampled_depth2(in.tokens,in.rows,in.memory,3,{Select::choose,&selection})
#else
                target.verify_sampled_cost3(in.tokens,in.rows,in.memory,3,{Select::choose,&selection})
#endif
                ); });
            ++cost_three_refusals;
            require(target.position()==3 && target.accepted_snapshot()==base.state &&
                same_bytes(target.last_final_hc_residual(),base.hc),"refused COST width-three transaction published state"); ledger();
        }
#endif
        ledger();
    }
    require(graph.memory().live_bytes==weights_only.live_bytes,"sampled stack leaked prepared GPU resources");
    {
        const auto row = [&](uint32_t token) { std::vector<float> out(shape.residual.channels);
            for (size_t i = 0; i < out.size(); ++i) out[i] = float(int((token * 7 + i * 3) % 23) - 11) / 16; return out; };
        const auto memory = [&](std::span<const uint64_t> ids) { std::vector<float> out(shape.memory.embedding);
            for (size_t i = 0; i < out.size(); ++i) out[i] = float(int(ids[i / spec.row_width] % 17) - 8) / 32; return out; };
        {
            TextStack greedy2_target(graph, f.weights, shape, policy, spec, StackExecution::direct, 8, reuse);
            TextStack serial(graph, f.weights, shape, policy, spec);
            greedy2_target.prepare_final_hc_tap(1ull << 20);
            serial.prepare_final_hc_tap(1ull << 20);
            greedy2_target.prepare_transient_verification(32ull << 20);
            const auto g2_resident = graph.memory();
            const auto advance = [&](TextStack& stack, uint32_t token) {
                const auto lookup = stack.lookup(token); ++batches;
                const auto out = stack.step(token, row(token), memory(lookup.rows));
                return SampledFrame{out.logits, {stack.last_final_hc_residual().begin(), stack.last_final_hc_residual().end()}, stack.accepted_snapshot()};
            };
            SampledFrame g2_base;
            for (uint32_t i = 0; i < 2; ++i) { g2_base = advance(greedy2_target, i); static_cast<void>(advance(serial, i)); }
            const uint32_t g2_t0 = uint32_t(std::max_element(g2_base.logits.begin(), g2_base.logits.end()) - g2_base.logits.begin());
            const auto g2_f0 = advance(serial, g2_t0);
            const uint32_t g2_t1 = uint32_t(std::max_element(g2_f0.logits.begin(), g2_f0.logits.end()) - g2_f0.logits.begin());
            const auto g2_f1 = advance(serial, g2_t1);
            const std::array<uint32_t, 2> g2_ids{g2_t0, g2_t1};
            const auto g2_lookups = greedy2_target.verification_lookups(g2_ids);
            std::vector<float> g2_rows, g2_mem;
            for (size_t i = 0; i < 2; ++i) {
                const auto r = row(g2_ids[i]), m = memory(g2_lookups[i].rows);
                g2_rows.insert(g2_rows.end(), r.begin(), r.end()); g2_mem.insert(g2_mem.end(), m.begin(), m.end());
            }
            const auto g2_committed = greedy2_target.verify_greedy2(g2_ids, g2_rows, g2_mem, 2, g2_base.logits, 2);
            ++batches;
            require(g2_committed.accepted == 2 && greedy2_target.has_last_final_hc_gpu() &&
                same_bytes(greedy2_target.last_final_hc_residual(), g2_f1.hc), "verify_greedy2 HC commit failed");
            const std::array<uint32_t, 2> mismatch_ids{uint32_t((std::max_element(g2_committed.logits.begin(), g2_committed.logits.end()) - g2_committed.logits.begin() + 1) % shape.vocabulary), 0u};
            const auto mismatch_lookups = greedy2_target.verification_lookups(mismatch_ids);
            std::vector<float> mm_rows, mm_mem;
            for (size_t i = 0; i < 2; ++i) {
                const auto r = row(mismatch_ids[i]), m = memory(mismatch_lookups[i].rows);
                mm_rows.insert(mm_rows.end(), r.begin(), r.end()); mm_mem.insert(mm_mem.end(), m.begin(), m.end());
            }
            const auto g2_mismatch = greedy2_target.verify_greedy2(mismatch_ids, mm_rows, mm_mem, 4, g2_committed.logits, 2);
            require(g2_mismatch.accepted == 0 && greedy2_target.has_last_final_hc_gpu(),
                "verify_greedy2 k=0 early exit invalidated committed GPU HC view");
            const uint32_t next_t0 = uint32_t(std::max_element(g2_committed.logits.begin(), g2_committed.logits.end()) - g2_committed.logits.begin());
            const std::array<uint32_t, 2> next_ids{next_t0, 0u};
            const auto next_lookups = greedy2_target.verification_lookups(next_ids);
            std::vector<float> next_rows, next_mem;
            for (size_t i = 0; i < 2; ++i) {
                const auto r = row(next_ids[i]), m = memory(next_lookups[i].rows);
                next_rows.insert(next_rows.end(), r.begin(), r.end()); next_mem.insert(next_mem.end(), m.begin(), m.end());
            }
            const auto saved_output = graph.read(f.weights.output.view);
            auto poisoned_output = saved_output;
            const uint16_t nan_scale = 0x7e00;
            std::memcpy(poisoned_output.data(), &nan_scale, sizeof(nan_scale));
            graph.write(f.weights.output.view, poisoned_output);
            reject([&] { static_cast<void>(greedy2_target.verify_greedy2(next_ids, next_rows, next_mem, 4, g2_committed.logits, 2)); });
            graph.write(f.weights.output.view, saved_output);
            require(!greedy2_target.has_last_final_hc_gpu() && same_bytes(greedy2_target.last_final_hc_residual(), g2_f1.hc),
                "failed verify_greedy2 left aliased GPU HC view valid or corrupted host HC tap");
            const auto now = graph.memory();
            require(now.live_bytes == g2_resident.live_bytes && now.allocations == g2_resident.allocations,
                "verify_greedy2 HC test changed prepared GPU ledger");
        }
        require(graph.memory().live_bytes == weights_only.live_bytes, "verify_greedy2 HC test leaked prepared GPU resources");
    }
#ifdef TOKSPUR_TEST_Q8_THREE_ROWS
    const auto n3_routes=graph.q8_three_rows_routes();require(n3_routes[0]>0,"sampled state test never dispatched genuine N3");
#endif
}
} // namespace

int main(int argc,char** argv) {
    try {
        bool concurrent = false;
#ifdef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
        if (argc == 3) { require(std::string_view(argv[2]) == "--concurrent", "unknown sampled stack scheduling argument"); concurrent = true; }
        else
#endif
        if(argc!=2) throw std::runtime_error("usage: test-sampled-stack <metallib> [--concurrent]");
        for (const bool host_views : {false, true}) {
            sampled_stack(argv[1],false,concurrent,host_views);
            sampled_stack(argv[1],true,concurrent,host_views);
        }
        require(nonfinite_output_refusals == 8, "copy/shared-view NaN and infinity coverage differs");
#if defined(TOKSPUR_PRIVATE_SAMPLED_VERIFICATION_COST) || defined(TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2)
        require(cost_width3_cases == 48 && production_three_refusals == 4 && cost_three_refusals == 8,
            "sampled COST width-three control coverage differs");
        std::cout << "sampled copy/shared-view width3 cases=48 production-three-refusals=4 cost-three-refusals=8\n";
#endif
        std::cout<<"sampled stack generated checks="<<checks<<" batches="<<batches<<" nonfinite-output-refusals="<<nonfinite_output_refusals<<"; no model/serving qualification\n";
    } catch(const std::exception& e) { std::cerr<<"sampled stack: "<<e.what()<<'\n'; return 1; }
    return 0;
}
