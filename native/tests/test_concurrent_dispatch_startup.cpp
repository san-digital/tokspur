#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"
#include <iostream>
#include <stdexcept>
#include <array>
#include <vector>

#ifndef TOKSPUR_PRIVATE_CONCURRENT_DISPATCH
#error This test requires the private concurrent-dispatch build.
#endif

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void refuses(F action) {
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("private concurrent operation unexpectedly admitted");
}

void physical_command_counts(const char* library, bool concurrent) {
    tokspur::MetalGraph graph(library, 64ull << 20);
    graph.configure_concurrent_dispatch(concurrent);
    const std::array<float, 4> values{0.25f, 0.5f, 1.0f, 2.0f};
    const auto input = graph.upload(std::as_bytes(std::span(values)), "split counter input");
    const auto mid = graph.zeros(sizeof(values), "split counter intermediate");
    const auto output = graph.zeros(sizeof(values), "split counter output");
    require(graph.lifetime_command_buffers() == 0 && graph.lifetime_dispatches() == 0, "startup counted unsubmitted work");
    std::vector<std::byte> split_output;
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        graph.begin();
        require(!graph.commit_partial() && graph.lifetime_command_buffers() == 2 * repeat, "empty commit_partial submitted a command");
        graph.activate(input, mid, tokspur::Activation::silu);
        require(graph.commit_partial(), "commit_partial refused non-empty direct batch");
        require(graph.lifetime_command_buffers() == 2 * repeat + 1, "commit_partial omitted its physical command buffer");
        graph.activate(mid, output, tokspur::Activation::silu);
        const auto batch = graph.submit();
        require(batch.dispatches == 2 && graph.lifetime_dispatches() == 2 * (repeat + 1), "split batch double-counted or omitted dispatches");
        require(graph.lifetime_command_buffers() == 2 * (repeat + 1), "split logical batch must count two physical command buffers");
        split_output = graph.read(output);
    }
    graph.begin();
    graph.activate(input, mid, tokspur::Activation::silu);
    graph.activate(mid, output, tokspur::Activation::silu);
    static_cast<void>(graph.submit());
    require(graph.lifetime_command_buffers() == 5 && graph.lifetime_dispatches() == 6, "ordinary batch did not add exactly one physical command");
    require(graph.read(output) == split_output, "partial command split changed generated results");
    graph.begin(); graph.abort();
    require(graph.lifetime_command_buffers() == 5 && graph.lifetime_dispatches() == 6, "empty abort changed completed physical counters");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: test-concurrent-startup <metallib>");
        tokspur::MetalGraph profiled(argv[1], 64 * 1024 * 1024);
        profiled.configure_concurrent_dispatch(false);
        const auto mode = profiled.configure_profiling(8);
        require(mode != tokspur::ProfileMode::disabled, "startup profile-lock regression needs a supported profiler");
        refuses([&] { profiled.configure_concurrent_dispatch(true); });
        require(!profiled.concurrent_dispatch(), "refused selection changed mode");
        const auto input = profiled.zeros(64 * sizeof(float), "startup input");
        const auto output = profiled.zeros(64 * sizeof(float), "startup output");
        profiled.begin(); profiled.activate(input, output, tokspur::Activation::silu);
        const auto batch = profiled.submit(); require(batch.dispatches == 1, "startup test dispatch was not exercised");
        static_cast<void>(profiled.configure_profiling(0));
        // This used to admit ON after an encoder-boundary profiled batch.
        refuses([&] { profiled.configure_concurrent_dispatch(true); });
        require(!profiled.concurrent_dispatch(), "profiled first batch left policy mutable");

        tokspur::MetalGraph concurrent(argv[1], 64 * 1024 * 1024);
        concurrent.configure_concurrent_dispatch(true);
        refuses([&] { static_cast<void>(concurrent.configure_profiling(8)); });
        refuses([&] { concurrent.begin_capture(8); });
        refuses([&] { concurrent.begin_indirect_compatible(); });
        concurrent.begin(); concurrent.abort();
        refuses([&] { concurrent.configure_concurrent_dispatch(false); });
        require(concurrent.concurrent_dispatch(), "aborted first batch left policy mutable");
        const auto execute = [&](bool enabled) {
            tokspur::MetalGraph graph(argv[1], 64 * 1024 * 1024);
            graph.configure_concurrent_dispatch(enabled);
            const std::array<float, 4> values{-2, -0.5f, 0.5f, 2};
            const auto input = graph.upload(std::as_bytes(std::span(values)), "dependency input");
            const auto a = graph.zeros(sizeof(values), "dependency a");
            const auto b = graph.zeros(sizeof(values), "dependency b");
            const auto c = graph.zeros(sizeof(values), "dependency c");
            graph.begin();
            graph.activate(input, a, tokspur::Activation::silu);
            graph.activate(input, b, tokspur::Activation::silu);
            graph.multiply(a, b, c);
            graph.activate(c, a, tokspur::Activation::silu);
            const auto result = graph.submit();
            require(result.dispatches == 4 && graph.concurrent_barriers() == (enabled ? 2 : 4), "RAW/WAR treatment not exercised");
            return graph.read(a);
        };
        require(execute(false) == execute(true), "concurrent dependency output differs from serial");
        physical_command_counts(argv[1], false);
        physical_command_counts(argv[1], true);
        std::cout << "concurrent startup: 6 refusals passed; profiler mode=" << static_cast<unsigned>(mode) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "concurrent startup refused: " << error.what() << '\n'; return 1;
    }
}
