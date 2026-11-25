#pragma once

#include "tokspur/gguf.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur {

enum class DFlash2Precision : uint32_t { q8_0 = 0, bf16 = 1 };

inline constexpr uint64_t maximum_dflash2_tap_budget_bytes = 256ull << 20;

inline uint64_t dflash2_tap_budget(uint32_t known_capacity, uint32_t verify_rows = 8) {
    if (verify_rows != 8 && verify_rows != 16) throw FormatError("unsupported DFlash2 verify rows");
    switch (known_capacity) {
    case 0: case 8: return (verify_rows == 16 ? 16ull : 8ull) << 20;
    case 512: return 48ull << 20;
    case 1024: return 64ull << 20;
    case 2048: return 160ull << 20;
    default: throw FormatError("unsupported DFlash2 tap capacity");
    }
}

struct DFlash2TapPlan {
    uint32_t known_capacity;
    uint64_t known_bytes, verify_bytes;
};

// Plan both live banks before any Metal allocation. The graph still checks the actual allocated-size charge against this budget.
inline DFlash2TapPlan dflash2_tap_plan(uint32_t known_capacity, uint32_t slots, uint32_t width, uint64_t budget, uint32_t verify_rows = 8) {
    static_cast<void>(dflash2_tap_budget(known_capacity));
    if (!slots || slots > 16 || !width || width % 4 || !budget || budget > maximum_dflash2_tap_budget_bytes ||
        (verify_rows != 8 && verify_rows != 16))
        throw FormatError("invalid DFlash2 tap geometry or budget");
    const uint32_t capacity = known_capacity ? known_capacity : 8;
    // The admitted capacity and slot bounds keep these products within uint64_t even at the largest uint32_t width.
    const uint64_t row_bytes = uint64_t(slots) * width * sizeof(float);
    const DFlash2TapPlan plan{capacity, uint64_t(capacity) * row_bytes, uint64_t(verify_rows) * row_bytes};
    if (plan.known_bytes + plan.verify_bytes > budget) throw FormatError("DFlash tap exceeds its explicit GPU budget");
    return plan;
}

inline DFlash2Precision parse_dflash2_precision(std::string_view value) {
    if (value == "q8_0") return DFlash2Precision::q8_0;
    if (value == "bf16") return DFlash2Precision::bf16;
    throw FormatError("native DFlash2 precision must be q8_0 or bf16");
}

inline uint32_t parse_native_speculative_depth(std::string_view value) {
    if (value == "1") return 1;
    if (value == "2") return 2;
    if (value == "3") return 3;
    if (value == "7") return 7;
    throw FormatError("native speculative depth must be 1, 2, 3 or 7");
}

inline constexpr uint32_t dflash2_default_max_drafts = 7;
inline constexpr uint32_t dflash2_maximum_drafts = 15;
inline constexpr uint32_t dflash2_width4_max_drafts = 3;
inline constexpr uint32_t dflash2_default_vocab_slice = 32768;

inline uint32_t parse_dflash2_max_drafts(std::string_view value) {
    if (value.size() == 1 && value[0] >= '1' && value[0] <= '9') return uint32_t(value[0] - '0');
    if (value.size() == 2 && value[0] == '1' && value[1] >= '0' && value[1] <= '5') return 10u + uint32_t(value[1] - '0');
    throw FormatError("native DFlash2 max_drafts must be an integer in 1..15");
}

inline uint32_t parse_dflash2_vocab_slice(std::string_view value) {
    if (value == "0" || value == "full") return 0;
    if (value.empty() || value[0] == '0') {
        throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
    }
    uint64_t parsed = 0;
    for (char ch : value) {
        if (ch < '0' || ch > '9') {
            throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
        }
        parsed = parsed * 10ull + uint64_t(ch - '0');
        if (parsed > 1048576ull) {
            throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
        }
    }
    if (parsed < 256ull || (parsed % 32ull) != 0ull) {
        throw FormatError("native DFlash2 vocab_slice must be 0, full, or a positive multiple of 32 in [256, 1048576]");
    }
    return uint32_t(parsed);
}

// Gathered draft vocabulary: comma-separated token IDs and inclusive ranges in strictly ascending order, for example "0-32734,248044-248076". The result is the ID of each draft row, a multiple of 32 in [256, vocabulary).
inline std::vector<uint32_t> parse_dflash2_vocab_gather(std::string_view value, uint32_t vocabulary) {
    const char* refusal = "native DFlash2 vocab_gather must be ascending token IDs or ranges such as 0-32734,248044-248076, a multiple of 32 in [256, vocabulary)";
    if (value.empty() || value.size() > 65536 || !vocabulary) throw FormatError(refusal);
    std::vector<uint32_t> ids;
    size_t at = 0;
    const auto number = [&]() {
        const size_t begin = at;
        uint64_t parsed = 0;
        while (at < value.size() && value[at] >= '0' && value[at] <= '9') {
            parsed = parsed * 10ull + uint64_t(value[at++] - '0');
            if (parsed >= vocabulary) throw FormatError(refusal);
        }
        if (at == begin || (value[begin] == '0' && at - begin > 1)) throw FormatError(refusal);
        return uint32_t(parsed);
    };
    for (;;) {
        const uint32_t first = number();
        uint32_t last = first;
        if (at < value.size() && value[at] == '-') {
            ++at;
            last = number();
        }
        if (last < first || (!ids.empty() && first <= ids.back())) throw FormatError(refusal);
        for (uint32_t id = first;; ++id) {
            ids.push_back(id);
            if (id == last) break;
        }
        if (at == value.size()) break;
        if (value[at++] != ',') throw FormatError(refusal);
    }
    if (ids.size() < 256 || ids.size() >= vocabulary || ids.size() % 32 != 0) throw FormatError(refusal);
    return ids;
}

inline constexpr uint32_t dflash2_qwen_ordinary_vocab = 248044;
inline constexpr uint32_t dflash2_qwen_control_end = 248076;

inline bool is_dflash2_vocab_gather_preset(std::string_view value) noexcept {
    return value == "control32k" || value == "cjk48k" || value == "cjk64k" || value == "cjk80k";
}

inline void admit_dflash2_vocab_gather(std::string_view value, uint32_t vocabulary) {
    if (is_dflash2_vocab_gather_preset(value)) {
        if (vocabulary <= dflash2_qwen_control_end) {
            throw FormatError("native DFlash2 named vocab_gather preset requires a Qwen3.8 vocabulary (> 248076)");
        }
        return;
    }
    static_cast<void>(parse_dflash2_vocab_gather(value, vocabulary));
}

namespace detail {

inline const std::array<int16_t, 324>& dflash2_reverse_byte_alphabet() {
    static const auto table = [] {
        std::array<int16_t, 324> rev{};
        rev.fill(-1);
        uint16_t extension = 256;
        for (uint16_t byte = 0; byte < 256; ++byte) {
            const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            const uint16_t cp = visible ? byte : extension++;
            rev[cp] = int16_t(byte);
        }
        return rev;
    }();
    return table;
}

inline bool decode_gpt2_byte_spelling(std::string_view spelling, std::string& raw_out) {
    raw_out.clear();
    const auto& rev = dflash2_reverse_byte_alphabet();
    for (size_t i = 0; i < spelling.size();) {
        uint32_t cp = static_cast<unsigned char>(spelling[i++]);
        if (cp >= 128) {
            if (cp < 0xc2 || cp > 0xc5 || i == spelling.size()) return false;
            const auto cont = static_cast<unsigned char>(spelling[i++]);
            if ((cont & 0xc0) != 0x80) return false;
            cp = ((cp & 31u) << 6u) | (cont & 63u);
        }
        if (cp >= rev.size() || rev[cp] < 0) return false;
        raw_out.push_back(static_cast<char>(rev[cp]));
    }
    return true;
}

inline bool inspect_utf8_cjk(std::string_view bytes, uint32_t& codepoint_count, bool& has_kana_or_punct, bool& has_han) {
    codepoint_count = 0;
    has_kana_or_punct = false;
    has_han = false;
    for (size_t i = 0; i < bytes.size();) {
        const auto b0 = static_cast<unsigned char>(bytes[i]);
        uint32_t cp = 0;
        size_t width = 0;
        if (b0 < 0x80) {
            cp = b0;
            width = 1;
        } else if (b0 >= 0xc2 && b0 <= 0xdf) {
            if (i + 2 > bytes.size()) return false;
            const auto b1 = static_cast<unsigned char>(bytes[i + 1]);
            if ((b1 & 0xc0) != 0x80) return false;
            cp = ((b0 & 0x1fu) << 6u) | (b1 & 0x3fu);
            width = 2;
        } else if (b0 >= 0xe0 && b0 <= 0xef) {
            if (i + 3 > bytes.size()) return false;
            const auto b1 = static_cast<unsigned char>(bytes[i + 1]);
            const auto b2 = static_cast<unsigned char>(bytes[i + 2]);
            if ((b1 & 0xc0) != 0x80 || (b2 & 0xc0) != 0x80) return false;
            if ((b0 == 0xe0 && b1 < 0xa0) || (b0 == 0xed && b1 > 0x9f)) return false;
            cp = ((b0 & 0x0fu) << 12u) | ((b1 & 0x3fu) << 6u) | (b2 & 0x3fu);
            width = 3;
        } else if (b0 >= 0xf0 && b0 <= 0xf4) {
            if (i + 4 > bytes.size()) return false;
            const auto b1 = static_cast<unsigned char>(bytes[i + 1]);
            const auto b2 = static_cast<unsigned char>(bytes[i + 2]);
            const auto b3 = static_cast<unsigned char>(bytes[i + 3]);
            if ((b1 & 0xc0) != 0x80 || (b2 & 0xc0) != 0x80 || (b3 & 0xc0) != 0x80) return false;
            if ((b0 == 0xf0 && b1 < 0x90) || (b0 == 0xf4 && b1 > 0x8f)) return false;
            cp = ((b0 & 0x07u) << 18u) | ((b1 & 0x3fu) << 12u) | ((b2 & 0x3fu) << 6u) | (b3 & 0x3fu);
            width = 4;
        } else {
            return false;
        }
        ++codepoint_count;
        if ((cp >= 0x3000 && cp <= 0x30ff) || (cp >= 0x31f0 && cp <= 0x31ff) || (cp >= 0xff00 && cp <= 0xffef)) {
            has_kana_or_punct = true;
        }
        if ((cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff)) {
            has_han = true;
        }
        i += width;
    }
    return true;
}

template <typename TokenString>
inline std::vector<uint32_t> resolve_dflash2_vocab_gather_impl(
    std::string_view value, uint32_t vocabulary, std::span<const TokenString> gguf_tokens) {
    admit_dflash2_vocab_gather(value, vocabulary);
    if (value == "control32k") {
        return parse_dflash2_vocab_gather("0-32734,248044-248076", vocabulary);
    }
    if (!is_dflash2_vocab_gather_preset(value)) {
        return parse_dflash2_vocab_gather(value, vocabulary);
    }
    uint32_t total_rows = 49152;
    uint32_t base_prefix = 24576;
    if (value == "cjk64k") {
        total_rows = 65536;
        base_prefix = 28672;
    } else if (value == "cjk80k") {
        total_rows = 81920;
        base_prefix = 32768;
    }
    if (gguf_tokens.size() <= dflash2_qwen_control_end || vocabulary <= total_rows) {
        throw FormatError("DFlash2 CJK vocab_gather preset requires GGUF tokenizer.ggml.tokens covering Qwen3.8");
    }
    std::vector<uint8_t> selected(vocabulary, 0);
    uint32_t chosen = 0;
    const auto add_id = [&](uint32_t id) {
        if (!selected[id]) {
            selected[id] = 1;
            ++chosen;
        }
    };
    for (uint32_t id = 0; id < base_prefix; ++id) add_id(id);
    for (uint32_t id = dflash2_qwen_ordinary_vocab; id <= dflash2_qwen_control_end; ++id) add_id(id);

    std::vector<uint32_t> partial_utf8, kana_punct, cjk_kana, single_han, jp_ext_han, multi_han_2, multi_han_rest;
    std::string raw;
    raw.reserve(64);
    for (uint32_t id = 0; id < dflash2_qwen_ordinary_vocab; ++id) {
        if (!decode_gpt2_byte_spelling(gguf_tokens[id], raw) || raw.empty()) continue;
        uint32_t cp_count = 0;
        bool has_kp = false;
        bool has_han = false;
        if (!inspect_utf8_cjk(raw, cp_count, has_kp, has_han)) {
            partial_utf8.push_back(id);
            continue;
        }
        if (has_kp && !has_han) kana_punct.push_back(id);
        else if (has_kp && has_han) cjk_kana.push_back(id);
        else if (has_han && cp_count == 1) single_han.push_back(id);
        else if (has_han && id >= 148000) jp_ext_han.push_back(id);
        else if (has_han && cp_count == 2) multi_han_2.push_back(id);
        else if (has_han) multi_han_rest.push_back(id);
    }
    for (const auto* bucket : {&partial_utf8, &kana_punct, &cjk_kana, &single_han, &jp_ext_han, &multi_han_2, &multi_han_rest}) {
        for (uint32_t id : *bucket) {
            if (chosen >= total_rows) break;
            add_id(id);
        }
        if (chosen >= total_rows) break;
    }
    for (uint32_t id = base_prefix; chosen < total_rows && id < dflash2_qwen_ordinary_vocab; ++id) {
        add_id(id);
    }
    if (chosen != total_rows) {
        throw FormatError("DFlash2 CJK vocab_gather preset failed to populate exact multiple-of-32 row count");
    }
    std::vector<uint32_t> out;
    out.reserve(total_rows);
    for (uint32_t id = 0; id < vocabulary; ++id) {
        if (selected[id]) out.push_back(id);
    }
    return out;
}

} // namespace detail

inline std::vector<uint32_t> resolve_dflash2_vocab_gather(
    std::string_view value, uint32_t vocabulary, std::span<const std::string_view> gguf_tokens = {}) {
    return detail::resolve_dflash2_vocab_gather_impl<std::string_view>(value, vocabulary, gguf_tokens);
}

template <typename Range>
requires (!std::is_convertible_v<Range, std::span<const std::string_view>>)
inline std::vector<uint32_t> resolve_dflash2_vocab_gather(
    std::string_view value, uint32_t vocabulary, const Range& gguf_tokens) {
    return detail::resolve_dflash2_vocab_gather_impl<std::string>(value, vocabulary, std::span<const std::string>(gguf_tokens));
}

} // namespace tokspur
