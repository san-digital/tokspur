"""CPU-only checks of the entropy stream a restore publishes in a native-MTP build.

The MTP build admits stochastic branch requests, so this compiles the actual
branch coordinator with that configuration. No Metal or model evidence.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The native-MTP coordinator hashes with CommonCrypto, which only Apple ships.
# Elsewhere, and under TOKSPUR_TEST_PORTABLE_SHA256, the unit supplies the same
# five names itself so the coordinator compiles unchanged.
SHA256 = r"""
#if defined(__APPLE__) && !defined(TOKSPUR_TEST_PORTABLE_SHA256)
#include <CommonCrypto/CommonCrypto.h>
#else
#include <cstddef>
#include <cstdint>
using CC_LONG = uint32_t;
struct CC_SHA256_CTX { uint32_t state[8]; uint64_t bytes; unsigned char block[64]; };
inline uint32_t tokspur_sha256_rotr(uint32_t value, int by) { return value >> by | value << (32 - by); }
inline void tokspur_sha256_block(CC_SHA256_CTX* c) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = uint32_t(c->block[4 * i]) << 24 | uint32_t(c->block[4 * i + 1]) << 16 |
            uint32_t(c->block[4 * i + 2]) << 8 | uint32_t(c->block[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = tokspur_sha256_rotr(w[i - 15], 7) ^ tokspur_sha256_rotr(w[i - 15], 18) ^ w[i - 15] >> 3;
        const uint32_t s1 = tokspur_sha256_rotr(w[i - 2], 17) ^ tokspur_sha256_rotr(w[i - 2], 19) ^ w[i - 2] >> 10;
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t v[8];
    for (int i = 0; i < 8; ++i) v[i] = c->state[i];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = tokspur_sha256_rotr(v[4], 6) ^ tokspur_sha256_rotr(v[4], 11) ^ tokspur_sha256_rotr(v[4], 25);
        const uint32_t t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
        const uint32_t s0 = tokspur_sha256_rotr(v[0], 2) ^ tokspur_sha256_rotr(v[0], 13) ^ tokspur_sha256_rotr(v[0], 22);
        const uint32_t t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
        v[7] = v[6]; v[6] = v[5]; v[5] = v[4]; v[4] = v[3] + t1;
        v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t1 + t2;
    }
    for (int i = 0; i < 8; ++i) c->state[i] += v[i];
}
inline int CC_SHA256_Init(CC_SHA256_CTX* c) {
    static constexpr uint32_t initial[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    for (int i = 0; i < 8; ++i) c->state[i] = initial[i];
    c->bytes = 0;
    return 1;
}
inline int CC_SHA256_Update(CC_SHA256_CTX* c, const void* data, CC_LONG length) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (CC_LONG i = 0; i < length; ++i) {
        c->block[c->bytes++ % 64] = bytes[i];
        if (c->bytes % 64 == 0) tokspur_sha256_block(c);
    }
    return 1;
}
inline int CC_SHA256_Final(unsigned char* digest, CC_SHA256_CTX* c) {
    const uint64_t bits = c->bytes * 8;
    const unsigned char one = 0x80, zero = 0;
    CC_SHA256_Update(c, &one, 1);
    while (c->bytes % 64 != 56) CC_SHA256_Update(c, &zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; ++i) length[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    CC_SHA256_Update(c, length, 8);
    for (int i = 0; i < 32; ++i) digest[i] = static_cast<unsigned char>(c->state[i / 4] >> (24 - 8 * (i % 4)));
    return 1;
}
#endif
"""

CHECKS = r"""
struct FakeResult { std::vector<float> logits; };
struct FakeHandle { uint32_t prefix = 0; };
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void refuses(F&& action) {
    bool refused = false;
    try { action(); } catch (const FormatError&) { refused = true; }
    check(refused, "invalid restore admitted");
}
constexpr SamplingPolicy stochastic{0.75, 0, 1, 0};
struct Fixture {
    BranchSession<FakeHandle, FakeResult> branch{2, 4};
    CommittedResult<FakeResult> retained;
    TargetSampler sampler;
    std::vector<uint32_t> history;
    uint32_t prefix = 0;
    explicit Fixture(TargetSampling options) : sampler(options) {}
    void append(std::initializer_list<uint32_t> tokens) {
        const auto before = prefix;
        prefix += uint32_t(tokens.size());
        retained.commit(FakeResult{{1, 2, 3, 4}}, prefix);
        for (uint32_t p = before + 1; p <= prefix; ++p) check(sampler.committed_input(p), "input sampler publication failed");
        check(branch.committed(prefix, std::span(tokens.begin(), tokens.size())), "token history publication failed");
        history.insert(history.end(), tokens.begin(), tokens.end());
    }
    uint32_t generate() {
        return sampled_step(sampler, retained.get(prefix).logits, prefix, [&](uint32_t id) { append({id}); });
    }
    void capture(bool with_stream) {
        const auto retain = [&] { return FakeHandle{prefix}; };
        if (with_stream) branch.capture(prefix, retained, &sampler, nullptr, 1, 0, std::nullopt, 0, nullptr, retain);
        else branch.capture(prefix, retained, retain);
    }
    void restore(size_t index, TargetSampling options) {
        prefix = branch.restore(index, retained, sampler, options, [&](FakeHandle handle) { return handle.prefix; });
        history.resize(prefix);
    }
};
void branch_restore_streams() {
    // The record is taken after generation, so its stream has spent entropy.
    Fixture f({stochastic, 3});
    f.append({7, 8});
    static_cast<void>(f.generate()); static_cast<void>(f.generate());
    check(f.prefix == 4 && f.sampler.random().words_consumed() == 2, "stochastic generation consumed unexpected entropy");
    f.capture(true);
    const auto saved = f.sampler.random();
    f.append({9});
    static_cast<void>(f.generate());
    check(f.sampler.random() != saved, "later generation left the stream unchanged");

    // Selection has no view of the request's sampling, so the next request
    // reaches restore with this record whatever its seed.
    auto prompt = f.history; prompt.resize(4); prompt.push_back(1);
    check(f.branch.select(prompt, 1) == std::optional<size_t>(0), "record was not selected for the shared prefix");

    // Same stochastic seed: the request continues the record's stream.
    f.restore(0, {stochastic, 3});
    check(f.prefix == 4 && f.sampler.matches(4) && f.sampler.random() == saved, "same-seed restore lost the saved stream");

    // A different seed starts its own stream at the restored prefix, as a cold
    // prefill of the same tokens would. It used to be refused on every retry.
    f.append({9});
    f.restore(0, {stochastic, 4});
    check(f.prefix == 4 && f.sampler.matches(4) && f.sampler.options().seed == 4 &&
        f.sampler.options().policy.temperature == 0.75 && f.sampler.random() == SamplingRng(4),
        "different-seed stochastic restore did not start a fresh stream");
    static_cast<void>(f.generate());
    check(f.prefix == 5 && f.sampler.random().words_consumed() == 1, "restored stochastic session cannot generate");

    // Greedy restores are unchanged: fresh options, no entropy.
    f.restore(0, {});
    check(f.prefix == 4 && f.sampler.matches(4) && f.sampler.options().policy.temperature == 0 &&
        f.sampler.random() == SamplingRng(0), "greedy restore changed");
    f.restore(0, {SamplingPolicy{}, 11});
    check(f.sampler.matches(4) && f.sampler.random() == SamplingRng(11), "seeded greedy restore changed");

    // A record captured without a stream cannot continue one, whatever the seed.
    Fixture bare({stochastic, 3});
    bare.append({7, 8});
    static_cast<void>(bare.generate());
    bare.capture(false);
    bare.append({9});
    bare.restore(0, {stochastic, 3});
    check(bare.prefix == 3 && bare.sampler.matches(3) && bare.sampler.random() == SamplingRng(3),
        "streamless record did not start a fresh stream");

    // Refusals before native mutation still leave the session untouched.
    bare.append({9});
    const auto before = bare.sampler.random();
    refuses([&] { bare.branch.restore(5, bare.retained, bare.sampler, {stochastic, 4}, [&](FakeHandle handle) { return handle.prefix; }); });
    check(bare.prefix == 4 && bare.sampler.matches(4) && bare.sampler.random() == before, "refused restore changed the stream");
}
void cached_prefix_streams() {
    // A session that has generated: its stream has spent entropy.
    Fixture f({stochastic, 3});
    f.append({7, 8});
    static_cast<void>(f.generate()); static_cast<void>(f.generate());
    const auto live = f.sampler.random();
    check(live.words_consumed() == 2, "stochastic generation consumed unexpected entropy");
    SamplingRng same_seed(3);
    for (int i = 0; i < 5; ++i) static_cast<void>(same_seed.uniform());
    SamplingRng other_seed(4);
    static_cast<void>(other_seed.uniform());
    const auto options = f.sampler.options();

    // Append packet: a cache hit publishes the new prefix and nothing else.
    // The stream is neither rewound to word 0 nor swapped for the saved one,
    // so later draws are the ones a cache miss would have made.
    for (const SamplingRng* saved : std::initializer_list<const SamplingRng*>{&same_seed, &other_seed, &live}) {
        const auto hit = TargetSampler::restore_cached_prefix(options, 9, *saved, &live);
        check(hit.matches(9) && hit.random() == live && hit.options().seed == 3 &&
            hit.options().policy.temperature == 0.75, "append cache hit replaced the live entropy stream");
    }

    // Reset packet: the request's own policy and seed decide the stream.
    check(TargetSampler::restore_cached_prefix({stochastic, 3}, 9, same_seed, nullptr).random() == same_seed,
        "same-seed stochastic reset did not continue the saved stream");
    check(TargetSampler::restore_cached_prefix({stochastic, 3}, 9, other_seed, nullptr).random() == SamplingRng(3),
        "different-seed reset did not start a fresh stream");
    const auto greedy = TargetSampler::restore_cached_prefix({SamplingPolicy{}, 3}, 9, same_seed, nullptr);
    check(greedy.matches(9) && greedy.random() == SamplingRng(3) && greedy.options().policy.temperature == 0,
        "greedy reset did not start a fresh stream");

    for (const SamplingRng* inherited : std::initializer_list<const SamplingRng*>{nullptr, &live}) {
        refuses([&] { static_cast<void>(TargetSampler::restore_cached_prefix(options, 0, same_seed, inherited)); });
        refuses([&] { static_cast<void>(TargetSampler::restore_cached_prefix(options, 65537, same_seed, inherited)); });
        check(TargetSampler::restore_cached_prefix(options, 70000, same_seed, inherited, 131072).matches(70000),
            "128K cached prefix refused");
    }
}
std::string sha256_hex(std::string_view text, size_t piece) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    for (size_t at = 0; at < text.size(); at += piece)
        CC_SHA256_Update(&ctx, text.data() + at, CC_LONG(std::min(piece, text.size() - at)));
    std::array<unsigned char, 32> digest{};
    CC_SHA256_Final(digest.data(), &ctx);
    std::string hex;
    for (const auto byte : digest) { hex += "0123456789abcdef"[byte >> 4]; hex += "0123456789abcdef"[byte & 15]; }
    return hex;
}
void sha256_matches_the_standard() {
    // FIPS 180-4 examples, fed whole, a byte at a time and across block edges.
    const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const std::string million(1000000, 'a');
    for (const size_t piece : {size_t(1), size_t(7), size_t(64), size_t(1) << 20}) {
        check(sha256_hex("", piece) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 of the empty message");
        check(sha256_hex("abc", piece) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 of abc");
        check(sha256_hex(two_blocks, piece) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "SHA-256 of the two-block message");
        check(sha256_hex(million, piece) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA-256 of a million a");
    }
}
void run() { sha256_matches_the_standard(); branch_restore_streams(); cached_prefix_streams(); }
"""


def build(tmp_path, checks, sha256):
    spec = importlib.util.spec_from_file_location(
        "stochastic_restore_extract", ROOT / "scripts/check_owned_target_protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    unit = (
        '#include "tokspur/json.hpp"\n#include "tokspur/gguf.hpp"\n#include "tokspur/sampling.hpp"\n'
        '#include "tokspur/prompt_lookup.hpp"\n#include "tokspur/mtp_anchor_state.hpp"\n'
        '#include "tokspur/dflash2_drafter.hpp"\n'
        + SHA256
        + "#include <algorithm>\n#include <array>\n#include <charconv>\n#include <cmath>\n"
        "#include <initializer_list>\n#include <iostream>\n#include <limits>\n#include <optional>\n"
        "#include <set>\n#include <span>\n#include <string>\n#include <string_view>\n"
        "#include <type_traits>\n#include <utility>\n#include <vector>\n"
        "namespace tokspur { namespace {\n"
        + protocol.extract((ROOT / "native/src/target.cpp").read_text())
        + checks
        + "\n} }\nint main(int argc, char**) { try { tokspur::run(); "
        + "if (argc > 1) static_cast<void>(tokspur::request(tokspur::Json(), 0, 0, 0, false, false, false)); return 0; } "
        + "catch (const std::exception& error) { std::cerr << error.what() << '\\n'; return 1; } }\n"
    )
    path, executable = (
        tmp_path / "stochastic restore.cpp",
        tmp_path / "stochastic restore",
    )
    path.write_text(unit)
    compiled = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-DTOKSPUR_PRIVATE_NATIVE_MTP=1",
            *(["-DTOKSPUR_TEST_PORTABLE_SHA256=1"] if sha256 == "portable" else []),
            "-I" + str(ROOT / "native/include"),
            str(path),
            *[
                str(ROOT / ("native/src/" + name + ".cpp"))
                for name in ("json", "unicode", "sampling")
            ],
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stderr
    return executable


# Linux has only the portable hash; macOS compiles both so the portable one
# stays checked where the worker is built.
@pytest.mark.parametrize(
    "sha256", ("system", "portable") if sys.platform == "darwin" else ("portable",)
)
def test_stochastic_restores_publish_the_expected_entropy_stream(tmp_path, sha256):
    """Compile the extracted native-MTP coordinator with fake native commits."""
    executable = build(tmp_path, CHECKS, sha256)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
