#include "tokspur/serve.hpp"
#include "tokspur/owned_worker.hpp"
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#include "tokspur/mtp_wire.hpp"
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
#include "tokspur/raw_decode_wire.hpp"
#endif
#endif
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
#include "tokspur/serve_verify8.hpp"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <chrono>
#include <map>
#include <limits>
#include <set>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tokspur {
namespace {
void require(bool ok, const char* why) { if (!ok) throw FormatError(why); }
const Json& field(const Json& value, std::string_view key) {
    const auto* p = value.find(key); require(p, "missing required API field"); return *p;
}
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
Json project_weight_lanes(Json value, const Json& identity) {
    if (identity.is_null()) return value; // Preserve unselected response bytes and shape.
    auto out = value.object();
    Json::Object details;
    if (const auto* present = value.find("tokspur_private")) details = present->object();
    for (const auto name : owned_weight_lane_fields) details.emplace_back(std::string(name), field(identity, name));
    if (identity.find(owned_lane_auth_fields[0])) for (const auto name : owned_lane_auth_fields)
        details.emplace_back(std::string(name), field(identity, name));
    for (auto& [name, item] : out) if (name == "tokspur_private") { item = Json(std::move(details)); return Json(std::move(out)); }
    out.emplace_back("tokspur_private", Json(std::move(details))); return Json(std::move(out));
}
#endif
void keys(const Json& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& [key, item] : value.object()) {
        static_cast<void>(item);
        require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(), "unsupported API/schema field");
    }
}
double number(const Json& value, double low, double high) {
    const auto text = value.number_lexeme(); double result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(result) && result >= low && result <= high,
        "API number outside supported range"); return result;
}
Json real(double value) {
    std::array<char, 64> text{}; const auto r = std::to_chars(text.data(), text.data() + text.size(), value);
    require(r.ec == std::errc{}, "cannot encode API number"); return parse_json({text.data(), size_t(r.ptr - text.data())});
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
struct PrivateMtpRequestTrace {
    struct ClockSum {
        double sum = 0;
        uint32_t expected = 0, observed = 0;
        void add(const Json& source, std::string_view name) {
            require(expected < 131072, "private native trace sample bound"); ++expected;
            if (const auto* value = source.find(name)) {
                const auto sample = number(*value, 0, std::numeric_limits<double>::max());
                require(sample <= std::numeric_limits<double>::max() - sum, "private native trace clock overflow");
                sum += sample; ++observed;
            }
        }
        Json report() const {
            return Json(Json::Object{{"total_ms", expected && observed == expected ? real(sum) : Json()},
                {"observed_samples", Json::integer(observed)}, {"expected_samples", Json::integer(expected)}});
        }
    };
    inline static constexpr std::array<std::string_view, 8> round_clocks{"draft_wall_ms", "draft_gpu_ms", "verification_wall_ms", "verification_gpu_ms",
        "recovery_wall_ms", "recovery_gpu_ms", "materialization_wall_ms", "round_wall_ms"};
    inline static constexpr std::array<std::string_view, 4> counter_names{"native_mtp_rounds", "native_mtp_proposed", "native_mtp_accepted", "native_mtp_recovery_rounds"};
    ClockSum input_request, input_prefill, output_request, output_generation_wall, output_generation_gpu;
    std::array<ClockSum, round_clocks.size()> components{};
    std::array<uint64_t, counter_names.size()> counters{};
    uint64_t round_committed_tokens = 0, ordinary_tokens = 0, ordinary_tail_tokens = 0;
    void admit(const Json& previous, const Json& done, const Json& job, std::span<const Json> events) {
        const bool known_input = !field(job, "tokens").array().empty();
        if (known_input) {
            input_request.add(done, "request_wall_ms"); input_prefill.add(done, "prefill_wall_ms"); return;
        }
        output_request.add(done, "request_wall_ms"); output_generation_wall.add(done, "generation_batch_wall_ms"); output_generation_gpu.add(done, "generation_batch_gpu_ms");
        // A cold input reset may catch up unadmitted lifetime work. Count output deltas only against this packet's current admitted predecessor.
        for (size_t i = 0; i < counters.size(); ++i) {
            const auto before = field(previous, counter_names[i]).unsigned_integer(), after = field(done, counter_names[i]).unsigned_integer();
            require(after >= before && after - before <= 131072 && counters[i] <= 131072 - (after - before), "private native trace counter bound");
            counters[i] += after - before;
        }
        uint64_t round_tokens = 0, packet_tokens = 0, packet_ordinary = 0, packet_tails = 0;
        const auto maximum = field(job, "generate").unsigned_integer();
        for (const auto& event : events) {
            if (field(event, "event").string() == "mtp_round") {
                round_tokens += field(event, "count").unsigned_integer();
                for (size_t i = 0; i < components.size(); ++i) components[i].add(event, round_clocks[i]);
            } else {
                require(packet_tokens < maximum, "private native trace packet token bound");
                if (!event.find("round_committed_prefix")) { ++packet_ordinary; packet_tails += maximum - packet_tokens <= mtp_depth(done); }
                ++packet_tokens;
            }
        }
        const auto outputs = field(done, "generated_ids").array().size();
        require(packet_tokens == outputs && round_tokens <= outputs && packet_ordinary == outputs - round_tokens &&
            round_committed_tokens <= 131072 - round_tokens && ordinary_tokens <= 131072 - packet_ordinary &&
            ordinary_tail_tokens <= 131072 - packet_tails, "private native trace ordinary-token bound");
        round_committed_tokens += round_tokens; ordinary_tokens += packet_ordinary; ordinary_tail_tokens += packet_tails;
    }
    Json report() const {
        const bool complete_round_wall = counters[0] && components[7].expected == counters[0] && components[7].observed == counters[0];
        Json::Object rounds{{"rounds", Json::integer(counters[0])}, {"proposed_tokens", Json::integer(counters[1])},
            {"accepted_tokens", Json::integer(counters[2])}, {"round_committed_tokens", Json::integer(round_committed_tokens)},
            {"recovery_rounds", Json::integer(counters[3])}, {"ordinary_tokens", Json::integer(ordinary_tokens)},
            {"ordinary_tail_tokens", Json::integer(ordinary_tail_tokens)}, {"ordinary_non_tail_tokens", Json::integer(ordinary_tokens - ordinary_tail_tokens)},
            {"committed_tokens_per_round", counters[0] ? real(double(round_committed_tokens) / double(counters[0])) : Json()},
            {"wall_ms_per_round", complete_round_wall ? real(components[7].sum / double(counters[0])) : Json()},
            {"round_tokens_per_sec", (complete_round_wall && components[7].sum > 0.0) ? real((double(round_committed_tokens) * 1000.0) / components[7].sum) : Json()}};
        for (size_t i = 0; i < components.size(); ++i) rounds.emplace_back(std::string(round_clocks[i]), components[i].report());
        rounds.emplace_back("materialization_gpu_ms", Json()); rounds.emplace_back("round_gpu_ms", Json());
        return Json(Json::Object{{"packet_wall_scope", Json("admitted native worker packets, including each first output packet")},
            {"generation_batch_scope", Json("worker-reported generation batches, including MTP priming and ordinary tails")},
            {"round_component_scope", Json("admitted MTP round events, incomplete as a whole-generation GPU clock")},
            {"ordinary_tail_scope", Json("ordinary output tokens with at most native depth remaining in their output packet")},
            {"isolated_decode_measured", Json(false)}, {"timings_qualified", Json(false)},
            {"input_packets", Json(Json::Object{{"count", Json::integer(input_request.expected)}, {"request_wall_ms", input_request.report()}, {"prefill_wall_ms", input_prefill.report()}})},
            {"output_packets", Json(Json::Object{{"count", Json::integer(output_request.expected)}, {"request_wall_ms", output_request.report()},
                {"generation_batch_wall_ms", output_generation_wall.report()}, {"generation_batch_gpu_ms", output_generation_gpu.report()}})}, {"mtp", Json(std::move(rounds))}});
    }
};
#endif
struct Decimal { bool negative; std::string digits; int64_t exponent; };
Decimal decimal(const Json& value) {
    const auto text = value.number_lexeme(); require(text.size() <= 128, "tool numeric lexeme bound");
    size_t i = text.front() == '-' ? 1 : 0; const bool negative = i != 0;
    std::string digits; int64_t fraction = 0; bool after_point = false;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
        if (text[i] == '.') after_point = true;
        else { digits += text[i]; if (after_point) ++fraction; }
    }
    int64_t exponent = 0;
    if (i < text.size()) {
        ++i; bool minus = false;
        if (text[i] == '-' || text[i] == '+') { minus = text[i] == '-'; ++i; }
        const auto r = std::from_chars(text.data() + i, text.data() + text.size(), exponent);
        require(r.ec == std::errc{} && r.ptr == text.data() + text.size() && exponent <= 1000000, "tool numeric exponent bound");
        if (minus) exponent = -exponent;
    }
    exponent -= fraction;
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) return {false, "0", 0};
    digits.erase(0, first);
    while (digits.back() == '0') { digits.pop_back(); ++exponent; }
    return {negative, std::move(digits), exponent};
}
int compare(const Decimal& left, const Decimal& right) {
    if (left.digits == "0" && right.digits == "0") return 0;
    if (left.negative != right.negative) return left.negative ? -1 : 1;
    int order = 0;
    if (left.digits == "0") order = -1;
    else if (right.digits == "0") order = 1;
    else {
        const int64_t l = int64_t(left.digits.size()) + left.exponent, r = int64_t(right.digits.size()) + right.exponent;
        if (l != r) order = l < r ? -1 : 1;
        else for (size_t i = 0; i < std::max(left.digits.size(), right.digits.size()); ++i) {
            const char a = i < left.digits.size() ? left.digits[i] : '0', b = i < right.digits.size() ? right.digits[i] : '0';
            if (a != b) { order = a < b ? -1 : 1; break; }
        }
    }
    return left.negative ? -order : order;
}
// The multipleOf divisor as its significant digits and decimal exponent. Eighteen digits keep every remainder step below 10^19, inside uint64_t; the shortest lexeme that round-trips a double has at most seventeen.
std::pair<uint64_t, int64_t> divisor(const Json& value) {
    const auto step = decimal(value);
    require(!step.negative && step.digits != "0", "multipleOf must be positive");
    require(step.digits.size() <= 18, "multipleOf digit bound");
    uint64_t digits = 0; for (const char c : step.digits) digits = digits * 10 + uint64_t(c - '0');
    return {digits, step.exponent};
}
// Whether value divided by the divisor is an integer, in exact decimal arithmetic.
bool multiple(const Decimal& value, std::pair<uint64_t, int64_t> step) {
    const auto [digits, exponent] = step;
    if (value.digits == "0") return true;
    // The value's digits do not end in zero, so they are never divisible by the power of ten that a negative shift leaves in the denominator.
    const int64_t shift = value.exponent - exponent;
    if (shift < 0) return false;
    uint64_t rest = 0; for (const char c : value.digits) rest = (rest * 10 + uint64_t(c - '0')) % digits;
    // The divisor is below 2^60, so it holds at most 59 factors of two and fewer of five: beyond 60, further powers of ten cannot change whether the remainder is zero.
    for (int64_t i = 0; i < std::min<int64_t>(shift, 60); ++i) rest = (rest * 10) % digits;
    return rest == 0;
}
// Appends a form in which two values are the same bytes exactly when JSON Schema calls them equal: numbers by exact decimal value, object members in name order, everything else as dump writes it.
void canonical(const Json& value, std::string& out) {
    if (value.kind() == Json::Kind::number) {
        const auto n = decimal(value);
        if (n.negative) out += '-';
        out += n.digits; out += 'e'; out += std::to_string(n.exponent);
    } else if (value.kind() == Json::Kind::array) {
        out += '[';
        for (const auto& item : value.array()) { canonical(item, out); out += ','; }
        out += ']';
    } else if (value.kind() == Json::Kind::object) {
        std::vector<const Json::Object::value_type*> members;
        for (const auto& member : value.object()) members.push_back(&member);
        std::stable_sort(members.begin(), members.end(), [](const auto* left, const auto* right) { return left->first < right->first; });
        out += '{';
        for (const auto* member : members) { out += Json(member->first).dump(); out += ':'; canonical(member->second, out); out += ','; }
        out += '}';
    } else out += value.dump();
}
// The JSON Schema pattern keyword for tool parameters. A pattern is an ECMA-262 regular expression read the way JSON Schema recommends, in Unicode mode with no flags. It works on code points, ^ and $ match only at the two ends of the string, . matches anything but the four line terminators, \d, \w and \b are ASCII, \s is the ECMA-262 white space set, and the pattern matches when it matches anywhere in the string. One reading goes beyond Unicode mode: a backslash before any character of the Basic Multilingual Plane that is not an ASCII letter or digit means that character. Unicode mode allows that only before syntax characters, but zod-to-json-schema escapes every literal this way.
// Refused: backreferences, named groups, Unicode property escapes, escapes that name a lone surrogate, and everything Unicode mode calls a syntax error.
// Without captures or backreferences a pattern describes a regular language, and whether a lookaround assertion holds depends on the position alone. So each lookaround body is scanned once over the whole string to tabulate where it holds, and the pattern itself is then scanned with those tables. A scan follows every thread at once and visits each step at most once per position: the time is the string length times the compiled size, whatever the pattern. No backtracking order is involved, so a lazy quantifier matches the same strings as a greedy one.
namespace pattern {
using Ranges = std::vector<std::pair<uint32_t, uint32_t>>;
// The most steps a compiled pattern may hold, over its own sequence and every lookaround body. A scan visits at most this many steps per code point.
constexpr uint64_t steps_bound = 4096;
// The deepest group nesting, which bounds the recursion of the parser and the compiler.
constexpr size_t depth_bound = 32;
// The most lookaround assertions. Each one costs a table of one flag per position of the matched string.
constexpr size_t looks_bound = 16;
// The most bytes of pattern text, which bounds what the parser builds before the step count is known.
constexpr size_t bytes_bound = 16384;
constexpr const char* unsupported = "unsupported pattern";
constexpr const char* bound = "pattern exceeds the tool pattern bound";
// A parsed pattern. A node is empty exactly when it compiles to no step.
struct Node {
    enum class Kind { empty, set, start, end, word, look, concat, either, repeat };
    Kind kind = Kind::empty;
    size_t index = 0; // The set, or the lookaround body.
    bool wanted = false; // Whether a word boundary, or a match of the lookaround body, is required or forbidden.
    uint64_t low = 0, high = 0; // The counts of a repeat.
    bool open = false; // A repeat with no upper count.
    std::vector<Node> items; // The members of a sequence or a choice, or the one body of a repeat.
};
struct Step {
    enum class Kind { set, split, jump, start, end, word, look, match };
    Kind kind = Kind::match;
    size_t first = 0, second = 0;
    bool wanted = false;
};
// One step sequence. A lookahead body is compiled in reverse and scanned from the end of the string, which finds the positions where it starts.
struct Piece { std::vector<Step> steps; bool backward = false; };
// A compiled pattern: its character sets, then one piece per lookaround body in the order the bodies close, then the pattern's own piece.
struct Pattern {
    std::vector<Ranges> sets;
    std::vector<Piece> pieces;
    bool matches(const std::vector<uint32_t>& text) const;
    std::vector<bool> scan(const Piece& piece, const std::vector<uint32_t>& text, const std::vector<std::vector<bool>>& tables, bool once) const;
};
// The character as an ASCII byte, or zero for anything else.
char ascii(uint32_t c) { return c < 128 ? char(c) : '\0'; }
bool digit(char c) { return c >= '0' && c <= '9'; }
bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool word(uint32_t c) { return digit(ascii(c)) || letter(ascii(c)) || c == '_'; }
// Sorts and merges the ranges, then takes the complement when negate is set.
Ranges normal(Ranges ranges, bool negate) {
    std::sort(ranges.begin(), ranges.end());
    Ranges merged;
    for (const auto& [low, high] : ranges) {
        if (!merged.empty() && low <= merged.back().second + 1) merged.back().second = std::max(merged.back().second, high);
        else merged.emplace_back(low, high);
    }
    if (!negate) return merged;
    Ranges rest; uint32_t next = 0;
    for (const auto& [low, high] : merged) {
        if (low > next) rest.emplace_back(next, low - 1);
        next = high + 1;
    }
    if (next <= 0x10ffff) rest.emplace_back(next, 0x10ffff);
    return rest;
}
// The set a \d, \w or \s escape names, complemented for the capital letter.
std::optional<Ranges> shorthand(uint32_t c) {
    Ranges ranges;
    switch (ascii(c)) {
        case 'd': case 'D': ranges = {{0x30, 0x39}}; break;
        case 'w': case 'W': ranges = {{0x30, 0x39}, {0x41, 0x5a}, {0x5f, 0x5f}, {0x61, 0x7a}}; break;
        // ECMA-262 WhiteSpace and LineTerminator: the ASCII controls, the byte order mark and the Unicode space separators.
        case 's': case 'S': ranges = {{0x09, 0x0d}, {0x20, 0x20}, {0xa0, 0xa0}, {0x1680, 0x1680}, {0x2000, 0x200a}, {0x2028, 0x2029}, {0x202f, 0x202f}, {0x205f, 0x205f}, {0x3000, 0x3000}, {0xfeff, 0xfeff}}; break;
        default: return std::nullopt;
    }
    return normal(std::move(ranges), ascii(c) < 'a');
}
struct Parser {
    std::vector<uint32_t> source;
    size_t at = 0;
    std::vector<Ranges> sets;
    // Lookaround bodies in the order they close, so a body never refers to a later one. true marks a lookahead.
    std::vector<std::pair<Node, bool>> looks;
    // The next character as an ASCII byte, or zero at the end and for anything else.
    char peek() const { return at < source.size() ? ascii(source[at]) : '\0'; }
    bool eat(char c) {
        const bool found = at < source.size() && source[at] == uint32_t(c);
        if (found) ++at;
        return found;
    }
    uint32_t take() { require(at < source.size(), unsupported); return source[at++]; }
    Node set(Ranges ranges) {
        sets.push_back(std::move(ranges));
        Node node; node.kind = Node::Kind::set; node.index = sets.size() - 1; return node;
    }
    Node disjunction(size_t depth) {
        Node choices; choices.kind = Node::Kind::either;
        choices.items.push_back(sequence(depth));
        while (eat('|')) choices.items.push_back(sequence(depth));
        if (choices.items.size() == 1) return std::move(choices.items[0]);
        return choices;
    }
    Node sequence(size_t depth) {
        Node items; items.kind = Node::Kind::concat;
        while (at < source.size() && peek() != '|' && peek() != ')') {
            auto item = term(depth);
            if (item.kind != Node::Kind::empty) items.items.push_back(std::move(item));
        }
        if (items.items.empty()) return Node{};
        if (items.items.size() == 1) return std::move(items.items[0]);
        return items;
    }
    Node term(size_t depth) {
        const uint32_t c = take();
        Node atom;
        switch (ascii(c)) {
            case '^': atom.kind = Node::Kind::start; return atom;
            case '$': atom.kind = Node::Kind::end; return atom;
            case '(': {
                require(depth < depth_bound, bound);
                const auto look = eat('?') ? group() : std::nullopt;
                auto body = disjunction(depth + 1);
                require(eat(')'), unsupported);
                if (!look) return quantified(std::move(body));
                // An assertion takes no quantifier in Unicode mode.
                require(looks.size() < looks_bound, bound);
                looks.emplace_back(std::move(body), look->first);
                atom.kind = Node::Kind::look; atom.index = looks.size() - 1; atom.wanted = look->second; return atom;
            }
            case '[': atom = bracket(); break;
            case '.': atom = set(normal({{0x0a, 0x0a}, {0x0d, 0x0d}, {0x2028, 0x2029}}, true)); break;
            case '\\': {
                const uint32_t escaped = take();
                if (ascii(escaped) == 'b' || ascii(escaped) == 'B') { atom.kind = Node::Kind::word; atom.wanted = ascii(escaped) == 'b'; return atom; }
                if (auto ranges = shorthand(escaped)) atom = set(std::move(*ranges));
                else { const uint32_t point = escape(escaped, false); atom = set({{point, point}}); }
                break;
            }
            // A quantifier with nothing to repeat, or a bracket or brace that Unicode mode does not take literally.
            case '*': case '+': case '?': case '{': case '}': case ']': require(false, unsupported); break;
            default: atom = set({{c, c}}); break;
        }
        return quantified(std::move(atom));
    }
    // What follows (?: nothing to record for a plain group, otherwise whether the assertion looks ahead and whether its body must match.
    std::optional<std::pair<bool, bool>> group() {
        switch (ascii(take())) {
            case ':': return std::nullopt;
            case '=': return std::pair{true, true};
            case '!': return std::pair{true, false};
            case '<':
                switch (ascii(take())) {
                    case '=': return std::pair{false, true};
                    case '!': return std::pair{false, false};
                    default: break;
                }
                break;
            default: break;
        }
        require(false, unsupported);
        return std::nullopt;
    }
    Node quantified(Node atom) {
        uint64_t low = 0, high = 0; bool open = false;
        switch (peek()) {
            case '*': open = true; break;
            case '+': low = 1; open = true; break;
            case '?': high = 1; break;
            case '{':
                ++at;
                low = high = count();
                if (eat(',')) {
                    if (digit(peek())) high = count();
                    else open = true;
                }
                require(peek() == '}' && (open || low <= high), unsupported);
                break;
            default: return atom;
        }
        ++at;
        // A lazy quantifier changes which match is found first, never whether one exists.
        eat('?');
        if (atom.kind == Node::Kind::empty || (!open && high == 0)) return Node{};
        Node repeat; repeat.kind = Node::Kind::repeat; repeat.low = low; repeat.high = high; repeat.open = open;
        repeat.items.push_back(std::move(atom));
        return repeat;
    }
    uint64_t count() {
        const size_t start = at; uint64_t value = 0;
        while (digit(peek())) { value = std::min(value * 10 + uint64_t(peek() - '0'), steps_bound + 1); ++at; }
        require(at > start, unsupported);
        require(value <= steps_bound, bound);
        return value;
    }
    Node bracket() {
        const bool negate = eat('^');
        Ranges ranges;
        while (true) {
            const uint32_t c = take();
            if (c == ']') break;
            const auto low = member(c, ranges);
            // A dash is literal at either end of the class and makes a range anywhere else.
            if (peek() == '-' && at + 1 < source.size() && source[at + 1] != ']') {
                ++at;
                const auto high = member(take(), ranges);
                require(low && high && *low <= *high, unsupported);
                ranges.emplace_back(*low, *high);
            } else if (low) ranges.emplace_back(*low, *low);
        }
        return set(normal(std::move(ranges), negate));
    }
    // One member of a class: its code point, or nothing after adding the ranges of a shorthand set, which cannot end a range.
    std::optional<uint32_t> member(uint32_t c, Ranges& ranges) {
        if (c != '\\') return c;
        const uint32_t escaped = take();
        if (const auto named = shorthand(escaped)) { ranges.insert(ranges.end(), named->begin(), named->end()); return std::nullopt; }
        return escape(escaped, true);
    }
    // The code point an escape names, given the character after the backslash.
    uint32_t escape(uint32_t c, bool bracketed) {
        switch (ascii(c)) {
            case 't': return 0x09;
            case 'n': return 0x0a;
            case 'v': return 0x0b;
            case 'f': return 0x0c;
            case 'r': return 0x0d;
            case '0': require(!digit(peek()), unsupported); return 0;
            case 'c': { const uint32_t control = take(); require(letter(ascii(control)), unsupported); return control % 32; }
            case 'x': return hex(2);
            case 'u': {
                uint32_t point = 0;
                if (eat('{')) {
                    size_t digits = 0;
                    while (!eat('}')) { point = point * 16 + hex(1); require(point <= 0x10ffff, unsupported); ++digits; }
                    require(digits > 0, unsupported);
                } else {
                    point = hex(4);
                    if (point >= 0xd800 && point < 0xdc00) {
                        // Two escapes that form a surrogate pair name one code point.
                        require(eat('\\') && eat('u'), unsupported);
                        const uint32_t trail = hex(4);
                        require(trail >= 0xdc00 && trail < 0xe000, unsupported);
                        point = 0x10000 + ((point - 0xd800) << 10) + (trail - 0xdc00);
                    }
                }
                // A surrogate on its own never occurs in a string of scalars.
                require(point < 0xd800 || point >= 0xe000, unsupported);
                return point;
            }
            default: break;
        }
        if (bracketed && ascii(c) == 'b') return 0x08;
        // An ASCII letter or digit reaching here is a backreference, a property escape or no escape at all. Any other character of the Basic Multilingual Plane stands for itself. Beyond that plane engines that take such an escape disagree on what it means.
        require(!digit(ascii(c)) && !letter(ascii(c)) && c < 0x10000, unsupported);
        return c;
    }
    uint32_t hex(size_t count) {
        uint32_t value = 0;
        for (size_t i = 0; i < count; ++i) {
            const char c = ascii(take());
            require(digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'), unsupported);
            value = value * 16 + uint32_t(digit(c) ? c - '0' : (c | 0x20) - 'a' + 10);
        }
        return value;
    }
};
// The steps a node compiles to, clamped just above the bound so that no sum overflows.
uint64_t size(const Node& node) {
    uint64_t total = 0;
    switch (node.kind) {
        case Node::Kind::empty: break;
        case Node::Kind::set: case Node::Kind::start: case Node::Kind::end: case Node::Kind::word: case Node::Kind::look: total = 1; break;
        case Node::Kind::either: total = 2 * (uint64_t(node.items.size()) - 1); [[fallthrough]];
        case Node::Kind::concat: for (const auto& item : node.items) total = std::min(total + size(item), steps_bound + 1); break;
        case Node::Kind::repeat: {
            const uint64_t body = size(node.items[0]);
            if (!node.open) total = node.low * body + (node.high - node.low) * (body + 1);
            else if (node.low == 0) total = body + 2;
            else total = node.low * body + 1;
            break;
        }
    }
    return std::min(total, steps_bound + 1);
}
// Appends the node's steps. A backward piece consumes the string from its end, so a sequence is laid out last item first.
void emit(const Node& node, bool backward, std::vector<Step>& steps) {
    const auto push = [&](Step::Kind kind, size_t first = 0, size_t second = 0, bool wanted = false) { steps.push_back({kind, first, second, wanted}); };
    switch (node.kind) {
        case Node::Kind::empty: break;
        case Node::Kind::set: push(Step::Kind::set, node.index); break;
        case Node::Kind::start: push(Step::Kind::start); break;
        case Node::Kind::end: push(Step::Kind::end); break;
        case Node::Kind::word: push(Step::Kind::word, 0, 0, node.wanted); break;
        case Node::Kind::look: push(Step::Kind::look, node.index, 0, node.wanted); break;
        case Node::Kind::concat:
            if (backward) for (size_t i = node.items.size(); i-- > 0;) emit(node.items[i], backward, steps);
            else for (const auto& item : node.items) emit(item, backward, steps);
            break;
        case Node::Kind::either: {
            std::vector<size_t> exits;
            for (size_t i = 0; i < node.items.size(); ++i) {
                if (i + 1 == node.items.size()) { emit(node.items[i], backward, steps); break; }
                const size_t split = steps.size();
                push(Step::Kind::split, split + 1);
                emit(node.items[i], backward, steps);
                exits.push_back(steps.size());
                push(Step::Kind::jump);
                steps[split].second = steps.size();
            }
            for (const size_t exit : exits) steps[exit].first = steps.size();
            break;
        }
        case Node::Kind::repeat: {
            const auto& body = node.items[0];
            size_t last = steps.size();
            for (uint64_t i = 0; i < node.low; ++i) { last = steps.size(); emit(body, backward, steps); }
            if (!node.open) {
                std::vector<size_t> splits;
                for (uint64_t i = node.low; i < node.high; ++i) {
                    splits.push_back(steps.size());
                    push(Step::Kind::split, steps.size() + 1);
                    emit(body, backward, steps);
                }
                for (const size_t split : splits) steps[split].second = steps.size();
            } else if (node.low == 0) {
                const size_t split = steps.size();
                push(Step::Kind::split, split + 1);
                emit(body, backward, steps);
                push(Step::Kind::jump, split);
                steps[split].second = steps.size();
            } else push(Step::Kind::split, last, steps.size() + 1);
            break;
        }
    }
}
// Parses and compiles a pattern, refusing one outside the supported grammar or over a bound.
Pattern compile(std::string_view source) {
    require(source.size() <= bytes_bound, bound);
    Parser parser;
    parser.source = unicode_scalars(source);
    auto root = parser.disjunction(0);
    // Only an unmatched closing parenthesis stops the parser early.
    require(parser.at == parser.source.size(), unsupported);
    // The pattern itself is the last piece and scans forward, as a lookbehind body does.
    parser.looks.emplace_back(std::move(root), false);
    uint64_t total = 0;
    for (const auto& [body, backward] : parser.looks) total = std::min(total + size(body) + 1, steps_bound + 1);
    require(total <= steps_bound, bound);
    Pattern compiled;
    for (const auto& [body, backward] : parser.looks) {
        Piece piece; piece.backward = backward;
        emit(body, backward, piece.steps);
        piece.steps.push_back({});
        compiled.pieces.push_back(std::move(piece));
    }
    compiled.sets = std::move(parser.sets);
    return compiled;
}
// Whether the pattern matches anywhere in the string, given as code points.
bool Pattern::matches(const std::vector<uint32_t>& text) const {
    std::vector<std::vector<bool>> tables;
    // A lookaround table needs every position. The pattern's own piece comes last and needs one match.
    for (size_t index = 0; index < pieces.size(); ++index) tables.push_back(scan(pieces[index], text, tables, index + 1 == pieces.size()));
    const auto& hits = tables.back();
    return std::find(hits.begin(), hits.end(), true) != hits.end();
}
// The positions at which a match of the piece ends, for a backward piece the positions at which one starts. A thread starts at every position. With once set the scan stops at its first match.
std::vector<bool> Pattern::scan(const Piece& piece, const std::vector<uint32_t>& text, const std::vector<std::vector<bool>>& tables, bool once) const {
    std::vector<bool> hits(text.size() + 1);
    // The scan pass in which each step was last entered, so a step is entered once per position however many threads reach it.
    std::vector<size_t> entered(piece.steps.size()), stack, waiting;
    for (size_t pass = 1; pass <= text.size() + 1; ++pass) {
        const size_t at = piece.backward ? text.size() + 1 - pass : pass - 1;
        stack.push_back(0);
        waiting.clear();
        while (!stack.empty()) {
            const size_t step = stack.back(); stack.pop_back();
            if (entered[step] == pass) continue;
            entered[step] = pass;
            const auto& now = piece.steps[step];
            bool proceed = false;
            switch (now.kind) {
                case Step::Kind::set: waiting.push_back(step); break;
                case Step::Kind::split: stack.push_back(now.first); stack.push_back(now.second); break;
                case Step::Kind::jump: stack.push_back(now.first); break;
                case Step::Kind::start: proceed = at == 0; break;
                case Step::Kind::end: proceed = at == text.size(); break;
                case Step::Kind::word: {
                    const bool before = at > 0 && word(text[at - 1]), after = at < text.size() && word(text[at]);
                    proceed = (before != after) == now.wanted;
                    break;
                }
                case Step::Kind::look: proceed = tables[now.first][at] == now.wanted; break;
                case Step::Kind::match: hits[at] = true; break;
            }
            if (proceed) stack.push_back(step + 1);
        }
        if (pass > text.size() || (once && hits[at])) break;
        const uint32_t c = piece.backward ? text[at - 1] : text[at];
        // The stack is empty here, so it carries the threads that consumed this character into the next position.
        for (const size_t step : waiting) {
            const auto& ranges = sets[piece.steps[step].first];
            const auto range = std::partition_point(ranges.begin(), ranges.end(), [c](const auto& item) { return item.second < c; });
            if (range != ranges.end() && range->first <= c) stack.push_back(step + 1);
        }
    }
    return hits;
}
} // namespace pattern
void safe_name(std::string_view name, bool function) {
    require(!name.empty() && name.size() <= (function ? 64 : 1024), "tool name bound");
    for (const unsigned char c : name) {
        require(c >= 32 && c < 127 && c != '<' && c != '>', "unsupported tool name");
        if (function) require(std::isalnum(c) || c == '_' || c == '-' || c == '.', "unsupported function name");
    }
}
std::string_view schema_type(const Json& schema) {
    if (schema.find("anyOf")) return "any";
    const auto value = field(schema, "type").string();
    require(value == "string" || value == "boolean" || value == "number" || value == "integer" || value == "object" || value == "array" || value == "null",
        "unsupported tool schema type"); return value;
}
void schema_admit(const Json& schema, size_t depth, size_t& nodes) {
    require(depth <= 16 && ++nodes <= 4096, "tool schema complexity bound");
    keys(schema, {"type", "anyOf", "title", "default", "examples", "format", "$schema", "description", "properties", "required", "additionalProperties", "items", "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern", "enum", "const", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum", "multipleOf"});
    const auto type = schema_type(schema);
    if (const auto* dialect = schema.find("$schema")) require(depth == 0 && dialect->string() == "https://json-schema.org/draft/2020-12/schema", "unsupported tool schema dialect");
    if (const auto* p = schema.find("description")) require(p->string().size() <= 16384, "schema description bound");
    for (const auto name : {"title", "format"}) if (const auto* value = schema.find(name)) require(value->string().size() <= 16384, "schema annotation bound");
    if (const auto* value = schema.find("examples")) require(value->array().size() <= 128, "schema examples bound");
    if (type == "any") {
        keys(schema, {"anyOf", "$schema", "title", "description", "default", "examples", "format"});
        const auto& alternatives = field(schema, "anyOf").array();
        require(!alternatives.empty() && alternatives.size() <= 8, "schema union bound");
        for (const auto& branch : alternatives) schema_admit(branch, depth + 1, nodes);
        return;
    }
    if (type == "object") {
        const auto* properties = schema.find("properties");
        if (properties) {
            require(properties->object().size() <= 32, "too many tool parameters");
            for (const auto& [name, child] : properties->object()) { safe_name(name, false); schema_admit(child, depth + 1, nodes); }
        }
        if (const auto* additional = schema.find("additionalProperties")) require(!additional->boolean(), "additional tool properties are unsupported");
        if (const auto* required = schema.find("required")) {
            std::set<std::string_view> seen;
            for (const auto& item : required->array()) {
                const auto name = item.string(); require(properties && properties->find(name) && seen.insert(name).second, "invalid required tool parameter");
            }
        }
    } else require(!schema.find("properties") && !schema.find("required") && !schema.find("additionalProperties"), "object schema keyword on other type");
    if (type == "array") {
        schema_admit(field(schema, "items"), depth + 1, nodes);
        const auto* low = schema.find("minItems");
        const auto* high = schema.find("maxItems");
        if (low) static_cast<void>(low->unsigned_integer());
        if (high) static_cast<void>(high->unsigned_integer());
        // Validation refuses any array above 4096 items, so a larger minimum could never be met. A larger maximum stays admitted: the same cap still applies at validation.
        if (low) require(low->unsigned_integer() <= 4096, "minItems exceeds the tool array bound");
        if (low && high) require(low->unsigned_integer() <= high->unsigned_integer(), "inverted schema bounds");
        if (const auto* unique = schema.find("uniqueItems")) static_cast<void>(unique->boolean());
    } else require(!schema.find("items") && !schema.find("minItems") && !schema.find("maxItems") && !schema.find("uniqueItems"), "array schema keyword on other type");
    if (type == "string") {
        const auto* low = schema.find("minLength");
        const auto* high = schema.find("maxLength");
        if (low) static_cast<void>(low->unsigned_integer());
        if (high) static_cast<void>(high->unsigned_integer());
        // The response parser caps a generated tool call at 64 KiB (ResponseLimits::call_bytes), and a code point takes at least one byte, so no call this server emits could meet a larger minimum. A larger maximum stays admitted: it only fails to constrain.
        if (low) require(low->unsigned_integer() <= 65536, "minLength exceeds the tool call byte limit");
        if (low && high) require(low->unsigned_integer() <= high->unsigned_integer(), "inverted schema bounds");
        if (const auto* p = schema.find("pattern")) static_cast<void>(pattern::compile(p->string()));
    } else require(!schema.find("minLength") && !schema.find("maxLength") && !schema.find("pattern"), "string schema keyword on other type");
    if (const auto* values = schema.find("enum")) {
        require(type != "object" && type != "array" && !values->array().empty() && values->array().size() <= 128, "unsupported enum");
        // Admission checks enum members against the same type, without enum.
        Json::Object base; for (const auto& [key, value] : schema.object()) if (key != "enum") base.emplace_back(key, value);
        for (const auto& value : values->array()) validate_tool_value(Json(base), value);
    }
    if (const auto* value = schema.find("const")) {
        require(type != "object" && type != "array", "unsupported const");
        // Admission checks the constant against the same type, without const.
        Json::Object base; for (const auto& [key, item] : schema.object()) if (key != "const") base.emplace_back(key, item);
        validate_tool_value(Json(base), *value);
    }
    if (schema.find("minimum") || schema.find("maximum") || schema.find("exclusiveMinimum") || schema.find("exclusiveMaximum") || schema.find("multipleOf")) {
        require(type == "integer" || type == "number", "numeric bound on other schema type");
        if (const auto* low = schema.find("minimum")) { static_cast<void>(number(*low, -1e300, 1e300)); static_cast<void>(decimal(*low)); }
        if (const auto* high = schema.find("maximum")) { static_cast<void>(number(*high, -1e300, 1e300)); static_cast<void>(decimal(*high)); }
        if (const auto* low = schema.find("exclusiveMinimum")) {
            static_cast<void>(number(*low, -1e300, 1e300)); static_cast<void>(decimal(*low));
            if (const auto* high = schema.find("maximum")) require(compare(decimal(*low), decimal(*high)) < 0, "inverted schema bounds");
        }
        if (const auto* high = schema.find("exclusiveMaximum")) {
            static_cast<void>(number(*high, -1e300, 1e300)); static_cast<void>(decimal(*high));
            if (const auto* low = schema.find("minimum")) require(compare(decimal(*low), decimal(*high)) < 0, "inverted schema bounds");
            if (const auto* low = schema.find("exclusiveMinimum")) require(compare(decimal(*low), decimal(*high)) < 0, "inverted schema bounds");
        }
        if (schema.find("minimum") && schema.find("maximum")) require(compare(decimal(*schema.find("minimum")), decimal(*schema.find("maximum"))) <= 0, "inverted schema bounds");
        if (const auto* step = schema.find("multipleOf")) { static_cast<void>(number(*step, -1e300, 1e300)); static_cast<void>(divisor(*step)); }
    }
}
ParameterType parameter_type(std::string_view type) {
    if (type == "string") return ParameterType::raw_string;
    if (type == "boolean") return ParameterType::boolean;
    if (type == "number" || type == "integer") return ParameterType::number;
    if (type == "object") return ParameterType::object;
    if (type == "array") return ParameterType::array;
    if (type == "any") return ParameterType::json_value;
    return ParameterType::null_value;
}
void validate_executor_string(std::string_view raw) {
    require(raw.find('\0') == std::string_view::npos, "NUL byte inside tool parameter");
}
void validate_executor_node(const Json& value, size_t depth, size_t& nodes) {
    require(depth <= 32 && ++nodes <= 16384, "executor tool argument complexity bound");
    if (value.kind() == Json::Kind::string) {
        validate_executor_string(value.string());
    } else if (value.kind() == Json::Kind::number) {
        static_cast<void>(number(value, -1e300, 1e300));
    } else if (value.kind() == Json::Kind::array) {
        require(value.array().size() <= 4096, "tool array bound");
        for (const auto& item : value.array()) validate_executor_node(item, depth + 1, nodes);
    } else if (value.kind() == Json::Kind::object) {
        require(value.object().size() <= 256, "too many tool parameters");
        std::set<std::string_view> seen;
        for (const auto& [key, item] : value.object()) {
            if (depth == 0) safe_name(key, false);
            require(seen.insert(key).second, "duplicate executor tool property");
            validate_executor_string(key);
            validate_executor_node(item, depth + 1, nodes);
        }
    }
}
void thinking_history(const Json& message, std::string_view role, size_t max_request_bytes) {
    const auto* blocks = message.find("thinking_blocks");
    if (!blocks) return;
    require(role == "assistant", "thinking blocks require assistant history");
    require(blocks->array().size() <= 4096, "thinking block count bound");
    std::string text;
    for (const auto& block : blocks->array()) {
        keys(block, {"type", "thinking", "signature"});
        require(field(block, "type").string() == "thinking", "unsupported thinking block type");
        if (const auto* signature = block.find("signature")) require(signature->string().empty(), "signed thinking blocks are unsupported");
        const auto part = field(block, "thinking").string();
        require(part.size() <= max_request_bytes - text.size(), "thinking history bound");
        text += part;
    }
    const auto* reasoning = message.find("reasoning_content");
    require(text == (reasoning && !reasoning->is_null() ? reasoning->string() : std::string_view{}), "thinking history differs from reasoning content");
}
void history(const Json& messages, const std::vector<std::string>& tools, const std::vector<Json>& schemas, ToolEnumValidation tool_enums, ToolArgValidation tool_args, size_t max_request_bytes) {
    require(!messages.array().empty() && messages.array().size() <= 4096, "API history bound");
    std::map<std::string, std::string, std::less<>> pending;
    std::set<std::string, std::less<>> used;
    for (const auto& message : messages.array()) {
        keys(message, {"role", "content", "reasoning_content", "thinking_blocks", "tool_calls", "tool_call_id", "name"});
        const auto role = field(message, "role").string();
        require(role == "system" || role == "developer" || role == "user" || role == "assistant" || role == "tool", "unsupported API message role");
        thinking_history(message, role, max_request_bytes);
        if (const auto* content = message.find("content")) {
            if (content->kind() == Json::Kind::array) {
                require(content->array().size() <= 4096, "text content part bound");
                for (const auto& part : content->array()) { keys(part, {"type", "text"}); require(field(part, "type").string() == "text", "only text content parts supported"); static_cast<void>(field(part, "text").string()); }
            } else require(content->is_null() || content->kind() == Json::Kind::string, "unsupported message content");
        }
        if (role == "tool") {
            require(!message.find("tool_calls") && !message.find("reasoning_content"), "invalid tool result fields");
            require(!field(message, "content").is_null(), "tool result requires text content");
            const auto id = field(message, "tool_call_id").string(); const auto found = pending.find(id);
            require(found != pending.end(), "unsolicited or duplicate tool result");
            if (const auto* name = message.find("name")) require(name->string() == found->second, "tool result name/ID disagreement");
            pending.erase(found);
        } else {
            require(pending.empty(), "missing tool results before next message");
            require(!message.find("tool_call_id") && !message.find("name"), "tool identity field outside tool result");
            if (role != "assistant") require(!field(message, "content").is_null(), "message requires text content");
            if (const auto* reasoning = message.find("reasoning_content")) require(role == "assistant" && (reasoning->is_null() || reasoning->kind() == Json::Kind::string), "invalid reasoning history");
            if (const auto* calls = message.find("tool_calls")) {
                require(role == "assistant" && !calls->array().empty() && calls->array().size() <= 128, "invalid assistant tool history");
                for (const auto& call : calls->array()) {
                    keys(call, {"id", "type", "function", "index"});
                    if (const auto* index = call.find("index")) require(index->unsigned_integer() < 128, "historical tool index bound");
                    require(field(call, "type").string() == "function", "unsupported historical call type");
                    const auto id = std::string(field(call, "id").string()); require(!id.empty() && id.size() <= 128 && used.insert(id).second, "invalid/duplicate historical tool ID");
                    const auto& function = field(call, "function"); keys(function, {"name", "arguments"});
                    const auto name = field(function, "name").string();
                    const auto found = std::find(tools.begin(), tools.end(), name); require(found != tools.end(), "historical function absent from admitted tools");
                    const auto& arguments = field(function, "arguments"); require(arguments.kind() == Json::Kind::object || arguments.kind() == Json::Kind::string, "invalid historical tool arguments");
                    const Json parsed = arguments.kind() == Json::Kind::string ? parse_json(arguments.string(), {6 * 65536, 65536, 33}) : arguments;
                    if (tool_args == ToolArgValidation::executor) validate_executor_tool_arguments(parsed);
                    else validate_tool_value(schemas.at(size_t(found - tools.begin())), parsed, tool_enums);
                    pending.emplace(id, name);
                }
            }
        }
    }
    require(pending.empty(), "request ends with unanswered tool calls");
}
Json parser_manifest(const std::vector<ToolSpec>& tools) {
    Json::Array result;
    for (const auto& tool : tools) {
        Json::Array parameters;
        for (const auto& p : tool.parameters) {
            static constexpr const char* types[]{"string", "boolean", "number", "object", "array", "null", "any"};
            parameters.emplace_back(Json::Object{{"name", Json(p.name)}, {"type", Json(types[uint32_t(p.type)])}, {"required", Json(p.required)}});
        }
        result.emplace_back(Json::Object{{"name", Json(tool.name)}, {"parameters", Json(std::move(parameters))}});
    }
    return Json(std::move(result));
}
Json chunk(std::string_view id, std::string_view alias, uint64_t created, Json delta, Json finish = {}) {
    return Json(Json::Object{{"id", Json(std::string(id))}, {"object", Json("chat.completion.chunk")}, {"created", Json::integer(created)},
        {"model", Json(std::string(alias))}, {"choices", Json(Json::Array{Json(Json::Object{{"index", Json::integer(0)}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)}})})}});
}
void worker_execution(const Json& value, std::string_view selected) {
    require(selected == "direct" || selected == "recurrent-replay", "unsupported serving startup execution mode");
    require(field(value, "small_gemv_column_splits").unsigned_integer() == 16 &&
        field(value, "block_gemv_formats").unsigned_integer() == 3 &&
        field(value, "stack_execution").string() == selected, "worker execution policy changed");
    if (selected == "recurrent-replay") {
        require(field(value, "replay_prepared").boolean() &&
            field(value, "recurrent_replay_captures").unsigned_integer() == 24 &&
            field(value, "mapped_weight_region_limit_bytes").unsigned_integer() == uint64_t(4) * 1024 * 1024 * 1024,
            "owned recurrent replay preparation/capture/region identity mismatch");
    }
}
void broker_storage(const Json& value, const Json& ready) {
    const auto slots = field(ready, "checkpoint_slots").unsigned_integer();
    const auto budget = field(ready, "checkpoint_combined_budget_bytes").unsigned_integer();
    require(slots && slots <= 4 && budget && budget <= (512ull << 20), "invalid serving broker storage preparation");
    require(field(value, "checkpoint_slots").unsigned_integer() == slots &&
        field(value, "checkpoint_combined_budget_bytes").unsigned_integer() == budget, "serving broker configuration changed");
    uint64_t total = 0;
    for (const auto name : {"checkpoint_host_bytes", "checkpoint_store_host_bytes", "checkpoint_gpu_bytes"}) {
        const auto bytes = field(value, name).unsigned_integer();
        require(bytes && bytes <= budget - total && bytes == field(ready, name).unsigned_integer(), "serving broker storage charge changed");
        total += bytes;
    }
}
} // namespace

void validate_worker_ready(const Json& ready, std::string_view stack_execution, uint32_t checkpoint_slots, uint64_t checkpoint_budget) {
    require(field(ready, "event").string() == "ready" &&
        field(ready, "maximum_generated_tokens").unsigned_integer() == 4096,
        "owned worker readiness/execution identity mismatch");
    worker_execution(ready, stack_execution);
    require(checkpoint_slots <= 4 && checkpoint_budget <= (512ull << 20) && bool(checkpoint_slots) == bool(checkpoint_budget),
        "invalid serving broker preparation");
    if (checkpoint_slots) {
        require(stack_execution == "direct" && field(ready, "checkpoint_broker").boolean() &&
            field(ready, "broker_epoch").unsigned_integer() && field(ready, "checkpoint_slots").unsigned_integer() == checkpoint_slots &&
            field(ready, "checkpoint_combined_budget_bytes").unsigned_integer() == checkpoint_budget, "owned broker readiness identity mismatch");
        broker_storage(ready, ready);
    } else if (const auto* enabled = ready.find("checkpoint_broker")) require(!enabled->boolean(), "unexpected serving broker capability");
}

ToolEnumValidation parse_tool_enum_validation(std::string_view value) {
    if (value == "strict") return ToolEnumValidation::strict;
    if (value == "executor") return ToolEnumValidation::executor;
    throw FormatError("tool-enum-validation must be strict or executor");
}
std::string_view tool_enum_validation_name(ToolEnumValidation value) {
    switch (value) {
        case ToolEnumValidation::strict: return "strict";
        case ToolEnumValidation::executor: return "executor";
    }
    throw FormatError("invalid tool enum validation policy");
}
std::string_view tool_enum_validation_policy(ToolEnumValidation value) {
    return tool_enum_validation_name(value) == "executor" ? "original-agent-executor-enums-v1" : "strict-server-enums-v1";
}
ToolArgValidation parse_tool_arg_validation(std::string_view value) {
    if (value == "strict") return ToolArgValidation::strict;
    if (value == "executor") return ToolArgValidation::executor;
    throw FormatError("tool-arg-validation must be strict or executor");
}
std::string_view tool_arg_validation_name(ToolArgValidation value) {
    switch (value) {
        case ToolArgValidation::strict: return "strict";
        case ToolArgValidation::executor: return "executor";
    }
    throw FormatError("invalid tool arg validation policy");
}
std::string_view tool_arg_validation_policy(ToolArgValidation value) {
    return tool_arg_validation_name(value) == "executor" ? "executor-deferred-args-v1" : "strict-server-args-v1";
}
void validate_executor_tool_arguments(const Json& value) {
    require(value.kind() == Json::Kind::object, "executor tool arguments must be a JSON object");
    size_t nodes = 0;
    validate_executor_node(value, 0, nodes);
}
void validate_tool_value(const Json& schema, const Json& value, ToolEnumValidation tool_enums) {
    static_cast<void>(tool_enum_validation_name(tool_enums));
    const auto type = schema_type(schema);
    if (type == "any") {
        for (const auto& branch : field(schema, "anyOf").array()) {
            try { validate_tool_value(branch, value, tool_enums); return; }
            catch (const FormatError&) {}
        }
        require(false, "tool value matches no schema alternative");
    }
    if (type == "string") {
        require(value.kind() == Json::Kind::string, "tool value is not string");
        if (tool_enums == ToolEnumValidation::executor) validate_executor_string(value.string());
        const auto* minimum = schema.find("minLength");
        const auto* maximum = schema.find("maxLength");
        const auto* expression = schema.find("pattern");
        if (minimum || maximum || expression) {
            // JSON Schema counts a string's length, and matches a pattern, in Unicode code points, not bytes.
            const auto scalars = expression ? unicode_scalars(value.string()) : std::vector<uint32_t>{};
            const auto length = expression ? scalars.size() : unicode_scalar_count(value.string());
            if (minimum) require(length >= minimum->unsigned_integer(), "tool string below minLength");
            if (maximum) require(length <= maximum->unsigned_integer(), "tool string above maxLength");
            if (expression) require(pattern::compile(expression->string()).matches(scalars), "tool string does not match pattern");
        }
    }
    else if (type == "boolean") require(value.kind() == Json::Kind::boolean, "tool value is not boolean");
    else if (type == "null") require(value.is_null(), "tool value is not null");
    else if (type == "number" || type == "integer") {
        static_cast<void>(number(value, -1e300, 1e300)); const auto n = decimal(value);
        const auto safe = decimal(parse_json("9007199254740991")); auto magnitude = n; magnitude.negative = false;
        require(type != "integer" || ((n.digits == "0" || n.exponent >= 0) && compare(magnitude, safe) <= 0 && value.number_lexeme().find_first_of(".eE") == std::string_view::npos), "tool integer must use exact safe-integer syntax");
        if (const auto* minimum = schema.find("minimum")) require(compare(n, decimal(*minimum)) >= 0, "tool value below minimum");
        if (const auto* maximum = schema.find("maximum")) require(compare(n, decimal(*maximum)) <= 0, "tool value above maximum");
        if (const auto* minimum = schema.find("exclusiveMinimum")) require(compare(n, decimal(*minimum)) > 0, "tool value at or below exclusive minimum");
        if (const auto* maximum = schema.find("exclusiveMaximum")) require(compare(n, decimal(*maximum)) < 0, "tool value at or above exclusive maximum");
        if (const auto* step = schema.find("multipleOf")) require(multiple(n, divisor(*step)), "tool value is not a multiple of multipleOf");
    } else if (type == "array") {
        const auto& items = value.array();
        require(items.size() <= 4096, "tool array bound");
        if (const auto* minimum = schema.find("minItems")) require(uint64_t(items.size()) >= minimum->unsigned_integer(), "tool array below minItems");
        if (const auto* maximum = schema.find("maxItems")) require(uint64_t(items.size()) <= maximum->unsigned_integer(), "tool array above maxItems");
        for (const auto& item : items) validate_tool_value(field(schema, "items"), item, tool_enums);
        // After the items are validated, so every number below has an admitted lexeme and nesting is bounded by the schema.
        if (const auto* unique = schema.find("uniqueItems"); unique && unique->boolean()) {
            std::set<std::string> seen;
            for (const auto& item : items) { std::string key; canonical(item, key); require(seen.insert(std::move(key)).second, "tool array repeats an item"); }
        }
    } else {
        const auto* properties = schema.find("properties");
        for (const auto& [name, item] : value.object()) {
            const auto* child = properties ? properties->find(name) : nullptr; require(child, "unknown tool property"); validate_tool_value(*child, item, tool_enums);
        }
        if (const auto* required = schema.find("required")) for (const auto& name : required->array()) require(value.find(name.string()), "missing required tool property");
    }
    if (const auto* values = schema.find("enum"); values && tool_enums == ToolEnumValidation::strict) {
        bool found = false;
        for (const auto& candidate : values->array()) {
            if (candidate.kind() == Json::Kind::number && value.kind() == Json::Kind::number) found |= compare(decimal(candidate), decimal(value)) == 0;
            else found |= candidate.dump() == value.dump();
        }
        require(found, "tool value outside enum");
    }
    // A constant is an enum of one value, so it follows the enum policy.
    if (const auto* expected = schema.find("const"); expected && tool_enums == ToolEnumValidation::strict) {
        const bool same = expected->kind() == Json::Kind::number && value.kind() == Json::Kind::number
            ? compare(decimal(*expected), decimal(value)) == 0 : expected->dump() == value.dump();
        require(same, "tool value differs from const");
    }
}

namespace {
constexpr size_t dump_growth = 3;
size_t short_escape_savings(const Json& value) {
    const auto text = [](std::string_view bytes) {
        size_t count = 0;
        for (const unsigned char c : bytes) {
            if (c == 0x08 || c == 0x09 || c == 0x0a || c == 0x0c || c == 0x0d) ++count;
        }
        return 4 * count;
    };
    switch (value.kind()) {
        case Json::Kind::string: return text(value.string());
        case Json::Kind::array: {
            size_t sum = 0;
            for (const auto& item : value.array()) sum += short_escape_savings(item);
            return sum;
        }
        case Json::Kind::object: {
            size_t sum = 0;
            for (const auto& [name, item] : value.object()) sum += text(name) + short_escape_savings(item);
            return sum;
        }
        case Json::Kind::null_value:
        case Json::Kind::boolean:
        case Json::Kind::number:
            return 0;
    }
    return 0;
}
} // namespace

AdmittedChat admit_chat(const Json& request, std::string_view alias, ToolEnumValidation tool_enums, ToolArgValidation tool_args, uint32_t context_length) {
    static_cast<void>(tool_enum_validation_name(tool_enums));
    static_cast<void>(tool_arg_validation_name(tool_args));
    require(context_length == 65536 || context_length == 131072, "unsupported serving context length");
    const size_t max_request_bytes = context_length > 65536 ? 4 * 1024 * 1024 : 1024 * 1024;
    const auto dumped = request.dump(dump_growth * max_request_bytes);
    require(dumped.size() - short_escape_savings(request) <= max_request_bytes, "chat request exceeds byte limit");
    keys(request, {"model", "messages", "n", "parallel_tool_calls", "stream", "stream_options", "keep_alive", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "tools", "tool_choice", "enable_thinking", "preserve_thinking", "reasoning_effort", "thinking_budget", "chat_template_kwargs", "presence_penalty", "repetition_penalty"});
    require(field(request, "model").string() == alias, "unknown model alias");
    AdmittedChat out;
    if (const auto* n = request.find("n")) require(n->unsigned_integer() == 1, "only n=1 is supported");
    if (const auto* parallel = request.find("parallel_tool_calls")) out.single_tool_call = !parallel->boolean();
    if (const auto* p = request.find("stream")) out.stream = p->boolean();
    if (const auto* options = request.find("stream_options")) {
        require(out.stream, "stream_options requires streaming"); keys(*options, {"include_usage"});
        if (const auto* include = options->find("include_usage")) out.include_usage = include->boolean();
    }
    // Numeric lifetime hints do not change the owned worker's supervised lifetime.
    if (const auto* keep = request.find("keep_alive")) {
        if (keep->kind() == Json::Kind::string) {
            const auto value = keep->string(); auto numeric = value;
            for (const auto suffix : {"ms", "s", "m", "h"}) if (numeric.ends_with(suffix)) { numeric.remove_suffix(std::string_view(suffix).size()); break; }
            require(!value.empty() && value.size() <= 64 && (value == "-1" || (!numeric.empty() && std::all_of(numeric.begin(), numeric.end(), [](unsigned char c) { return std::isdigit(c) || c == '.'; }) && std::count(numeric.begin(), numeric.end(), '.') <= 1 && std::any_of(numeric.begin(), numeric.end(), [](unsigned char c) { return std::isdigit(c); }))), "invalid keep_alive duration hint");
        } else static_cast<void>(number(*keep, -1, 1e12));
    }
    const auto* template_options = request.find("chat_template_kwargs");
    if (template_options) keys(*template_options, {"enable_thinking", "preserve_thinking", "reasoning_effort", "thinking_budget"});
    Json::Object canonical_options;
    for (const auto name : {"enable_thinking", "preserve_thinking", "reasoning_effort"}) {
        const auto* outer = request.find(name);
        const auto* inner = template_options ? template_options->find(name) : nullptr;
        if (std::string_view(name) == "reasoning_effort") {
            if (outer) static_cast<void>(outer->string());
            if (inner) static_cast<void>(inner->string());
        } else {
            if (outer) static_cast<void>(outer->boolean());
            if (inner) static_cast<void>(inner->boolean());
        }
        require(!outer || !inner || outer->dump() == inner->dump(), "conflicting chat template options");
        if (const auto* selected = outer ? outer : inner) canonical_options.emplace_back(name, *selected);
    }
    std::optional<uint32_t> raw_thinking_budget;
    {
        const auto* outer = request.find("thinking_budget");
        const auto* inner = template_options ? template_options->find("thinking_budget") : nullptr;
        if (outer) {
            const auto n = outer->unsigned_integer();
            require(n <= 32768, "thinking budget must be 0..32768");
            raw_thinking_budget = uint32_t(n);
        }
        if (inner) {
            const auto n = inner->unsigned_integer();
            require(n <= 32768, "thinking budget must be 0..32768");
            require(!outer || *raw_thinking_budget == uint32_t(n), "conflicting chat template options");
            raw_thinking_budget = uint32_t(n);
        }
    }
    const Json options(std::move(canonical_options));
    std::optional<bool> explicit_thinking;
    if (const auto* p = options.find("enable_thinking")) {
        explicit_thinking = p->boolean();
        out.thinking = *explicit_thinking;
    }
    if (raw_thinking_budget) {
        if (*raw_thinking_budget == 0) {
            require(explicit_thinking != std::optional<bool>(true), "conflicting thinking options");
            out.thinking = false;
        } else if (explicit_thinking != std::optional<bool>(false)) {
            // A positive budget only caps thinking. With thinking explicitly off there is nothing to cap, so the budget is ignored and not refused: clients keep a fixed budget and toggle thinking.
            out.thinking_budget = *raw_thinking_budget;
        }
    }
    if (const auto* p = options.find("reasoning_effort")) {
        const auto effort = p->string();
        require(effort == "xhigh" || effort == "high" || effort == "medium" || effort == "low" ||
            (!out.thinking && effort == "none"), "unsupported reasoning effort");
    }
    require(!(request.find("max_tokens") && request.find("max_completion_tokens")), "provide only one output limit");
    const auto* limit = request.find("max_tokens"); if (!limit) limit = request.find("max_completion_tokens");
    // The HTTP request budget is separate from the scalar, V8 or explicitly negotiated MTP packet capacity.
    if (limit) { const auto n = limit->unsigned_integer(); require(n && n <= 32768, "output limit must be 1..32768"); out.maximum_tokens = uint32_t(n); }
    for (const auto name : {"presence_penalty", "repetition_penalty"}) if (const auto* p = request.find(name)) {
        // Compare decimal values exactly: a non-neutral value must not become
        // neutral merely because parsing it as double would round it to 0/1.
        const auto neutral = parse_json(std::string_view(name) == "presence_penalty" ? "0" : "1");
        require(compare(decimal(*p), decimal(neutral)) == 0, "non-neutral sampling penalty is unsupported");
    }
    const double temperature = request.find("temperature") ? number(*request.find("temperature"), 0, 100) : 0;
    const double top_p = request.find("top_p") ? number(*request.find("top_p"), 0, 1) : 1; require(top_p > 0, "top_p must be positive");
    const double min_p = request.find("min_p") ? number(*request.find("min_p"), 0, 1) : 0;
    const auto top_k = request.find("top_k") ? request.find("top_k")->unsigned_integer() : 0; require(top_k <= 248320, "top_k bound");
    out.sampling = Json(Json::Object{{"temperature", real(temperature)}, {"top_k", Json::integer(top_k)}, {"top_p", real(top_p)}, {"min_p", real(min_p)}});
    if (const auto* seed = request.find("seed")) out.seed = seed->unsigned_integer();
    Json::Array declarations; std::set<std::string, std::less<>> names;
    if (const auto* tools = request.find("tools")) {
        require(tools->array().size() <= 64, "API tool limit");
        for (const auto& declaration : tools->array()) {
            keys(declaration, {"type", "function"}); require(field(declaration, "type").string() == "function", "unsupported tool type");
            const auto& function = field(declaration, "function"); keys(function, {"name", "description", "parameters", "strict"});
            const auto name = std::string(field(function, "name").string()); safe_name(name, true); require(names.insert(name).second, "duplicate tool name");
            if (const auto* description = function.find("description")) require(description->string().size() <= 16384, "tool description bound");
            if (const auto* strict = function.find("strict")) static_cast<void>(strict->boolean());
            Json schema = function.find("parameters") ? *function.find("parameters") : parse_json("{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}");
            size_t nodes = 0; schema_admit(schema, 0, nodes); require(schema_type(schema) == "object", "function parameters must be object");
            ToolSpec manifest{name, {}}; std::set<std::string_view> required;
            if (const auto* p = schema.find("required")) for (const auto& value : p->array()) required.insert(value.string());
            if (const auto* p = schema.find("properties")) for (const auto& [key, value] : p->object()) manifest.parameters.push_back({key, parameter_type(schema_type(value)), required.contains(key)});
            out.manifest.push_back(std::move(manifest)); out.schemas.push_back(std::move(schema)); out.names.push_back(name); declarations.push_back(declaration);
        }
    }
    out.tool_choice = "auto";
    if (const auto* choice = request.find("tool_choice")) {
        if (choice->kind() == Json::Kind::string) { out.tool_choice = choice->string(); require(out.tool_choice == "auto" || out.tool_choice == "none" || out.tool_choice == "required", "unsupported tool_choice"); }
        else { out.named_tool_choice = true; keys(*choice, {"type", "function"}); require(field(*choice, "type").string() == "function", "unsupported named tool_choice"); const auto& function = field(*choice, "function"); keys(function, {"name"}); out.tool_choice = field(function, "name").string(); require(names.contains(out.tool_choice), "unknown named tool_choice"); }
    }
    require(out.tool_choice == "auto" || out.tool_choice == "none" || !declarations.empty(), "tool choice requires tools");
    const auto& messages = field(request, "messages"); history(messages, out.names, out.schemas, tool_enums, tool_args, max_request_bytes);
    Json::Array rendered_messages = messages.array();
    if (out.named_tool_choice || out.tool_choice == "required") {
        const std::string instruction = !out.named_tool_choice ? "For this response, call at least one of the declared tools. Do not answer without a tool call." : "For this response, call only the declared function named " + out.tool_choice + ". Do not answer without that tool call.";
        rendered_messages.insert(rendered_messages.begin(), Json(Json::Object{{"role", Json("system")}, {"content", Json(instruction)}}));
    }
    if (!out.named_tool_choice && out.tool_choice == "none") { declarations.clear(); out.manifest.clear(); out.schemas.clear(); out.names.clear(); }
    else if (out.named_tool_choice) {
        const auto index = size_t(std::find(out.names.begin(), out.names.end(), out.tool_choice) - out.names.begin());
        declarations = {declarations.at(index)}; out.manifest = {out.manifest.at(index)}; out.schemas = {out.schemas.at(index)}; out.names = {out.names.at(index)};
    }
    Json::Object render{{"messages", Json(std::move(rendered_messages))}, {"tools", Json(std::move(declarations))}, {"enable_thinking", Json(out.thinking)}};
    for (const auto name : {"preserve_thinking", "reasoning_effort"}) if (const auto* p = options.find(name)) render.emplace_back(name, *p);
    out.render = Json(std::move(render)); return out;
}

ServeSession::ServeSession(ServeCodec codec, std::string alias, std::string stack_execution, Json broker_ready
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    , bool verify8, Json verify_ready
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    , bool branch8
#endif
    , ToolEnumValidation tool_enums
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    , bool sampled_prefill
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , Json mtp_ready
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    , Json weight_lanes_ready
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , std::optional<uint32_t> mtp_first_output_packet
#endif
    , ToolArgValidation tool_args
    , uint32_t context_length
    , bool lane_parking
    )
    : codec_(std::move(codec)), alias_(std::move(alias)), stack_execution_(std::move(stack_execution)), tool_enums_(tool_enums), tool_args_(tool_args), context_length_(context_length)
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    , sampled_prefill_(sampled_prefill)
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , mtp_(!mtp_ready.is_null()), mtp_ready_(std::move(mtp_ready)), mtp_previous_(mtp_ready_), mtp_first_output_packet_(mtp_first_output_packet)
#endif
    , broker_ready_(std::move(broker_ready))
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    , verify8_(verify8), verify_ready_(std::move(verify_ready))
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    , branch8_(branch8)
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    , lane_parking_(lane_parking && mtp_)
#else
    , lane_parking_(false)
#endif
    {
#ifndef TOKSPUR_PRIVATE_NATIVE_MTP
    static_cast<void>(lane_parking);
#endif
    static_cast<void>(tool_enum_validation_name(tool_enums_));
    static_cast<void>(tool_arg_validation_name(tool_args_));
    require(bool(codec_.render) && bool(codec_.encode) && bool(codec_.decode) && !alias_.empty(), "invalid serving codec");
    for (const auto id : codec_.nonordinary_ids) require(id < codec_.admission.vocabulary, "nonordinary token outside vocabulary");
    std::sort(codec_.nonordinary_ids.begin(), codec_.nonordinary_ids.end());
    codec_.nonordinary_ids.erase(std::unique(codec_.nonordinary_ids.begin(), codec_.nonordinary_ids.end()), codec_.nonordinary_ids.end());
    require(context_length_ == 65536 || context_length_ == 131072, "unsupported serving context length");
    require(stack_execution_ == "direct" || stack_execution_ == "recurrent-replay", "unsupported serving startup execution mode");
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
    if (!weight_lanes_ready.is_null()) {
        const auto workers = owned_lane_auth_workers_requested(weight_lanes_ready);
        validate_owned_weight_lanes_ready(weight_lanes_ready, true, workers);
        Json::Object identity;
        for (const auto name : owned_weight_lane_fields) identity.emplace_back(std::string(name), field(weight_lanes_ready, name));
        if (workers) for (const auto name : owned_lane_auth_fields)
            identity.emplace_back(std::string(name), field(weight_lanes_ready, name));
        weight_lanes_identity_ = Json(std::move(identity));
    }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (mtp_) {
        validate_owned_weight_lanes_ready(mtp_ready_, !weight_lanes_ready.is_null(), owned_lane_auth_workers_requested(mtp_ready_));
        if (!weight_lanes_ready.is_null()) validate_owned_weight_lanes_done(mtp_ready_, weight_lanes_ready);
    }
#endif
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
    require(!sampled_prefill_ || (stack_execution_ == "direct" && !verify8_ && !branch8_ && broker_ready_.is_null()),
        "sampled input8 refuses greedy verification, replay and checkpoint brokers");
    if (sampled_prefill_) committed_.reserve(context_length_); // Before all-N publication.
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    require(!mtp_ || (stack_execution_ == "direct" && !verify8_ && !branch8_ && !sampled_prefill_ && broker_ready_.is_null()),
        "sampled MTP refuses another transport, replay and checkpoint brokers");
    if (mtp_) { validate_mtp_ready(mtp_ready_, mtp_draft_attention(mtp_ready_), mtp_request_carry(mtp_ready_), mtp_depth(mtp_ready_)); committed_.reserve(context_length_); }
    if (mtp_first_output_packet_) {
        require(mtp_, "MTP first-output-packet requires native MTP readiness");
        const auto cap = mtp_output_packet(mtp_ready_);
        require(*mtp_first_output_packet_ >= 1 && *mtp_first_output_packet_ <= cap, "MTP first-output-packet outside negotiated output packet bound");
    }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    if (verify8_) {
        require(stack_execution_ == "direct", "private HTTP V8 cannot use replay");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (branch8_) {
            require(!broker_ready_.is_null() && broker_ready_.dump() == verify_ready_.dump(), "branch8 requires the same explicit broker/verifier readiness");
            validate_branch8_ready(verify_ready_); verify_previous_ = verify_ready_;
        } else
#endif
        { require(broker_ready_.is_null(), "private HTTP V8 cannot use checkpoints without explicit branch8 consent"); validate_verify8_ready(verify_ready_); }
        committed_.reserve(context_length_);
    } else require(verify_ready_.is_null(), "unexpected private V8 readiness");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    require(!branch8_ || verify8_, "branch8 requires explicit private V8 scheduling");
#endif
    if (!broker_ready_.is_null()) {
        const auto slots = field(broker_ready_, "checkpoint_slots").unsigned_integer();
        require(slots && slots <= 4, "serving broker slot bound");
        validate_worker_ready(broker_ready_, stack_execution_, uint32_t(slots), field(broker_ready_, "checkpoint_combined_budget_bytes").unsigned_integer());
        broker_epoch_ = field(broker_ready_, "broker_epoch").unsigned_integer();
        committed_.reserve(context_length_); // Before any restore publication.
    }
}
void ServeSession::invalidate() noexcept { saved_reasoning_.clear(); parked_reasoning_.clear(); committed_.clear(); sampling_.clear(); reusable_ = false; seed_ = 0; broker_resync_ = true;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    mtp_resync_ = true;
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    verify_resync_ = true;
#endif
}

std::optional<Json> ServeSession::fill_saved_reasoning(
    const Json& render, const std::vector<std::pair<size_t, std::string>>& saved) {
    if (saved.empty()) return std::nullopt;
    auto fields = render.object();
    auto messages_it = std::find_if(fields.begin(), fields.end(),
        [](const auto& item) { return item.first == "messages"; });
    if (messages_it == fields.end()) return std::nullopt;
    auto messages = messages_it->second.array();
    size_t ordinal = 0;
    bool filled = false;
    for (auto& message : messages) {
        if (field(message, "role").string() != "assistant") continue;
        const auto* present = message.find("reasoning_content");
        const auto found = std::find_if(saved.begin(), saved.end(),
            [&](const auto& entry) { return entry.first == ordinal; });
        if ((!present || present->is_null()) && found != saved.end()) {
            auto entry = message.object();
            std::erase_if(entry, [](const auto& item) { return item.first == "reasoning_content"; });
            entry.emplace_back("reasoning_content", Json(found->second));
            message = Json(std::move(entry));
            filled = true;
        }
        ++ordinal;
    }
    if (!filled) return std::nullopt;
    messages_it->second = Json(std::move(messages));
    return Json(std::move(fields));
}

void ServeSession::park_active_reasoning() {
    if (!lane_parking_ || !reusable_ || committed_.empty()) return;
    auto committed = committed_;
    auto turns = std::move(saved_reasoning_);
    saved_reasoning_.clear();
    const auto sampling = sampling_;
    const uint64_t seed = seed_;
    std::erase_if(parked_reasoning_, [&](const ParkedReasoning& old) {
        return old.sampling == sampling && old.seed == seed &&
            old.committed.size() <= committed.size() &&
            std::equal(old.committed.begin(), old.committed.end(), committed.begin());
    });
    if (parked_reasoning_.size() >= max_parked_reasoning_lanes) {
        parked_reasoning_.erase(parked_reasoning_.begin());
    }
    parked_reasoning_.push_back(ParkedReasoning{
        std::move(committed), std::move(turns), sampling, seed});
}

std::optional<std::vector<uint32_t>> ServeSession::respelled_on_committed(
    std::span<const uint32_t> committed, std::span<const uint32_t> canonical) const {
    const auto frame = [&](uint32_t id) {
        return id == codec_.admission.im_start || codec_.admission.is_stop(id) ||
            std::ranges::find(codec_.control_ids, id) != codec_.control_ids.end() ||
            std::binary_search(codec_.nonordinary_ids.begin(), codec_.nonordinary_ids.end(), id);
    };
    if (committed.empty() || !codec_.admission.is_stop(committed.back())) return std::nullopt;
    size_t shared = 0;
    while (shared < committed.size() && shared < canonical.size() && committed[shared] == canonical[shared]) ++shared;
    size_t live = shared, fresh = shared;
    while (live < committed.size()) {
        const auto live_end = size_t(std::find_if(committed.begin() + ptrdiff_t(live), committed.end(), frame) - committed.begin());
        const auto fresh_end = size_t(std::find_if(canonical.begin() + ptrdiff_t(fresh), canonical.end(), frame) - canonical.begin());
        if (live_end == committed.size() || fresh_end == canonical.size() || committed[live_end] != canonical[fresh_end]) {
            return std::nullopt;
        }
        auto live_changed = live_end, fresh_changed = fresh_end;
        while (live < live_changed && fresh < fresh_changed && committed[live] == canonical[fresh]) { ++live; ++fresh; }
        while (live < live_changed && fresh < fresh_changed && committed[live_changed - 1] == canonical[fresh_changed - 1]) {
            --live_changed; --fresh_changed;
        }
        if ((live < live_changed || fresh < fresh_changed) &&
            codec_.decode(committed.subspan(live, live_changed - live)) !=
                codec_.decode(canonical.subspan(fresh, fresh_changed - fresh))) {
            return std::nullopt;
        }
        live = live_end + 1; fresh = fresh_end + 1;
    }
    std::vector<uint32_t> respelled(committed.begin(), committed.end());
    respelled.insert(respelled.end(), canonical.begin() + ptrdiff_t(fresh), canonical.end());
    return respelled;
}

Json ServeSession::complete(const Json& input, std::string_view response_id, const ServeTarget& target, const ServeSink& sink) {
    const auto request_start = std::chrono::steady_clock::now();
    const auto request = admit_chat(input, alias_, tool_enums_, tool_args_, context_length_);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    uint64_t gdn_old_order_admitted_calls = 0;
    uint64_t hc_down_f32_admitted_calls = 0;
    std::array<uint64_t, 3> hc_rows_admitted{};
    std::array<uint64_t, 4> router_window_admitted{};
    uint64_t hc_down_window_admitted = 0;
    std::array<uint64_t, 2> qsa_sparse_admitted{};
    std::array<uint64_t, 5> gdn_ported_admitted{};
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
    RawDecodeRouteCounts raw_decode_admitted{};
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
    Json::Object q8_three_admitted_routes{{"raw",Json::integer(0)},{"packed",Json::integer(0)},{"hc_raw",Json::integer(0)},{"wide_qkv_raw",Json::integer(0)},{"wide_output_raw",Json::integer(0)}};
#endif
#endif
    [[maybe_unused]] uint64_t input_batches = 0, restored_input_tokens = 0, output_batches = 0, generation_rounds = 0, verified_rounds = 0, accepted_tokens = 0;
    [[maybe_unused]] size_t input_width = 0, output_width = 0;
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
    require(!verify8_ || number(field(request.sampling, "temperature"), 0, 100) == 0,
        "private HTTP V8 is greedy-only; nonzero temperature is explicitly refused");
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
    uint64_t checkpoint_captures = 0, checkpoint_restores = 0, checkpoint_cold_resyncs = 0;
#endif
    require(bool(target) && bool(sink.cancelled) && bool(sink.new_seed), "missing serving callbacks");
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    if (const auto* measured = mtp_ready_.is_null() ? nullptr : mtp_ready_.find("native_measured_controller"); measured && measured->boolean()) {
        const auto temperature = number(field(request.sampling, "temperature"), 0, 100);
        std::string_view mode = "greedy";
        if (temperature != 0) {
            require(temperature == 1 && number(field(request.sampling, "min_p"), 0, 1) == 0, "sampling policy is not calibrated by measured profile");
            const auto top_k = field(request.sampling, "top_k").unsigned_integer(); const auto top_p = number(field(request.sampling, "top_p"), 0, 1);
            if (top_k == 20 && top_p == 0.95) mode = "coding";
            else { require(top_p == 1 && (top_k == 0 || top_k == 20), "sampling policy is not calibrated by measured profile"); mode = top_k == 20 ? "topk" : "full"; }
        }
        const auto& modes = field(mtp_ready_, "native_controller_sampling_modes").array();
        require(!modes.empty() && modes.size() <= 4 && std::any_of(modes.begin(), modes.end(), [mode](const auto& value) { return value.string() == mode; }), "sampling mode is not calibrated by measured profile");
    }
#endif
    const auto policy = request.sampling.dump();
    const bool compatible_sampling = sampling_ == policy && (!request.seed || *request.seed == seed_);
    const auto rendered = codec_.render(request.render);
    const auto base_prompt = codec_.encode(rendered);
    auto prompt = base_prompt;
    std::optional<std::vector<std::pair<size_t, std::string>>> restored_reasoning;
    std::optional<size_t> restored_lane_idx;
    std::vector<std::pair<const std::vector<std::pair<size_t, std::string>>*, std::optional<std::vector<uint32_t>>>> cached_by_turns;
    const auto matches_committed = [&](const auto& value) { return value.size() >= committed_.size() && std::equal(committed_.begin(), committed_.end(), value.begin()); };
    // Sampling can commit a token split that re-encoding the same bytes does not reproduce. Reuse requires matching added tokens, reserved padding slots and admission markers, with equal bytes between them. Only the canonical tail after the committed terminal is appended.
    if (reusable_ && !matches_committed(prompt)) {
        std::optional<std::vector<uint32_t>> restored;
        if (!saved_reasoning_.empty()) {
            if (auto filled = fill_saved_reasoning(request.render, saved_reasoning_)) {
                auto candidate = codec_.encode(codec_.render(*filled));
                cached_by_turns.emplace_back(&saved_reasoning_, candidate);
                if (candidate.size() < context_length_ && matches_committed(candidate)) prompt = std::move(candidate);
                else restored = std::move(candidate);
            } else {
                cached_by_turns.emplace_back(&saved_reasoning_, std::nullopt);
            }
        }
        if (compatible_sampling && !matches_committed(prompt)) {
            if (auto respelled = respelled_on_committed(committed_, restored ? *restored : prompt);
                respelled && respelled->size() < context_length_) {
                prompt = std::move(*respelled);
            }
        }
    }
    if (lane_parking_ && (!reusable_ || !compatible_sampling || !matches_committed(prompt)) && !parked_reasoning_.empty()) {
        std::optional<size_t> best_idx;
        std::vector<uint32_t> best_prompt;
        bool ambiguous_seed = false;
        for (size_t idx = 0; idx < parked_reasoning_.size(); ++idx) {
            const auto& lane = parked_reasoning_[idx];
            if (best_idx && lane.committed.size() < parked_reasoning_[*best_idx].committed.size()) continue;
            if (lane.sampling != policy || (request.seed && *request.seed != lane.seed)) continue;
            const std::vector<uint32_t>* candidate_ptr = &base_prompt;
            if (!lane.turns.empty()) {
                const auto cached_it = std::find_if(cached_by_turns.begin(), cached_by_turns.end(),
                    [&](const auto& entry) { return *entry.first == lane.turns; });
                if (cached_it != cached_by_turns.end()) {
                    if (cached_it->second) candidate_ptr = &*cached_it->second;
                } else if (auto filled = fill_saved_reasoning(request.render, lane.turns)) {
                    auto encoded = codec_.encode(codec_.render(*filled));
                    cached_by_turns.emplace_back(&lane.turns, std::move(encoded));
                    candidate_ptr = &*cached_by_turns.back().second;
                } else {
                    cached_by_turns.emplace_back(&lane.turns, std::nullopt);
                }
            }
            const auto& candidate = *candidate_ptr;
            std::optional<std::vector<uint32_t>> matched;
            if (candidate.size() > lane.committed.size() && candidate.size() < context_length_ &&
                std::equal(lane.committed.begin(), lane.committed.end(), candidate.begin())) {
                matched = candidate;
            } else if (auto respelled = respelled_on_committed(lane.committed, candidate);
                respelled && respelled->size() > lane.committed.size() && respelled->size() < context_length_) {
                matched = std::move(*respelled);
            } else {
                continue;
            }
            if (!best_idx || lane.committed.size() > parked_reasoning_[*best_idx].committed.size()) {
                best_idx = idx;
                best_prompt = std::move(*matched);
                ambiguous_seed = false;
            } else if (lane.seed != parked_reasoning_[*best_idx].seed) {
                ambiguous_seed = true;
            }
        }
        if (best_idx && !ambiguous_seed) {
            prompt = std::move(best_prompt);
            restored_reasoning = parked_reasoning_[*best_idx].turns;
            restored_lane_idx = *best_idx;
        }
    }
    require(!prompt.empty() && prompt.size() < context_length_,
        context_length_ == 131072 ? "request leaves no generation space in the 128K context" : "request leaves no generation space in the 64K context");
    const auto maximum_tokens = std::min(request.maximum_tokens, context_length_ - uint32_t(prompt.size()));
    for (const auto id : prompt) require(id < codec_.admission.vocabulary, "rendered token outside vocabulary");
    const auto created_signed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    require(created_signed >= 0, "invalid API clock"); const auto created = uint64_t(created_signed);
    const bool append = reusable_ && compatible_sampling &&
        prompt.size() >= committed_.size() && std::equal(committed_.begin(), committed_.end(), prompt.begin());
    size_t cached_prefix = append ? committed_.size() : 0;
    size_t common = 0;
    if (reusable_) while (common < committed_.size() && common < prompt.size() && committed_[common] == prompt[common]) ++common;
    uint64_t seed = append ? seed_ : request.seed ? *request.seed : sink.new_seed();
    require(!response_id.empty() && response_id.size() <= 96, "response ID bound");
    ResponseParser parser(request.manifest, request.thinking, {}, tool_args_ == ToolArgValidation::executor, true);
    if (!codec_.control_ids.empty()) parser.enable_control_tokens();
    const auto feed_id = [&](uint32_t id) {
        const auto bytes = codec_.decode(std::span(&id, 1));
        return std::ranges::find(codec_.control_ids, id) != codec_.control_ids.end() ? parser.feed_control(bytes) : parser.feed(bytes);
    };
    std::string streamed_content, streamed_thinking;
    std::vector<uint32_t> generated; generated.reserve(maximum_tokens);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
    std::optional<PrivateMtpRequestTrace> native_trace;
    if (sink.prefix_trace && mtp_) native_trace.emplace();
#endif
    const auto cancelled = [&] { require(!sink.cancelled(), "request cancelled or deadline expired"); };
    const auto emit = [&](Json delta, Json finish = Json()) {
        if (request.stream) {
            require(bool(sink.chunk), "missing streaming sink"); cancelled();
            auto value = chunk(response_id, alias_, created, std::move(delta), finish);
            if (request.include_usage) { auto fields = value.object(); fields.emplace_back("usage", Json()); value = Json(std::move(fields)); }
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
            if (!finish.is_null()) value = project_weight_lanes(std::move(value), weight_lanes_identity_);
#endif
            sink.chunk(value);
        }
    };
    size_t record_index = 0, record_offset = 0;
    const auto flush = [&] {
        const auto records = parser.records();
        if (records.empty()) {
            require(record_index == 0 && record_offset == 0, "nonmonotonic response parser");
            return;
        }
        const auto thinking_size = streamed_thinking.size(), content_size = streamed_content.size();
        require(record_index < records.size(), "nonmonotonic response parser");
        for (const auto& record : records.subspan(record_index)) {
            require(record_offset <= record.text.size(), "nonmonotonic response parser");
            if (record.kind == ResponseKind::thinking) streamed_thinking.append(record.text, record_offset);
            else if (record.kind == ResponseKind::text) streamed_content.append(record.text, record_offset);
            record_offset = 0;
        }
        // Only the last record can grow between successful feeds.
        record_index = records.size() - 1;
        record_offset = records.back().text.size();
        if (request.stream && streamed_thinking.size() > thinking_size) emit(Json(Json::Object{{"reasoning_content", Json(streamed_thinking.substr(thinking_size))}}));
        if (request.stream && streamed_content.size() > content_size) emit(Json(Json::Object{{"content", Json(streamed_content.substr(content_size))}}));
    };
    bool state_touched = false;
    try {
        cancelled();
        if (sink.prefix_trace) {
            Json::Array ids; ids.reserve(prompt.size()); for (const auto id : prompt) ids.push_back(Json::integer(id));
            Json::Array prior; prior.reserve(committed_.size()); for (const auto id : committed_) prior.push_back(Json::integer(id));
            sink.prefix_trace(Json(Json::Object{{"event", Json("prompt_prefix")}, {"schema", Json::integer(1)}, {"response_id", Json(std::string(response_id))},
                {"prompt_ids", Json(std::move(ids))}, {"prior_committed_ids", Json(std::move(prior))}, {"prior_committed_tokens", Json::integer(committed_.size())}, {"common_prefix", Json::integer(common)},
                {"reusable", Json(reusable_)}, {"append", Json(append)}, {"sampling_matches", Json(sampling_ == policy)}, {"seed_matches", Json(!request.seed || *request.seed == seed_)},
                {"prompt_tokens", Json::integer(prompt.size())}, {"cached_tokens", Json::integer(cached_prefix)}, {"new_tokens", Json::integer(prompt.size() - cached_prefix)}}));
        }
        emit(Json(Json::Object{{"role", Json("assistant")}}));
        state_touched = true;
        size_t position = append ? committed_.size() : 0;
        const auto broker_reply = [&](const Json& value, bool fresh) {
            if (!broker_epoch_) return;
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (branch8_) require(field(value, "broker_protocol").string() == "branch8-v1" &&
                field(value, "broker_batch_width").unsigned_integer() == 8, "serving branch8 wire identity changed");
#endif
            require(field(value, "broker_epoch").unsigned_integer() == broker_epoch_, "serving worker epoch changed");
            const auto command = field(value, "command_id").unsigned_integer(), lease = field(value, "lease_id").unsigned_integer();
            require(command > broker_command_ && (broker_resync_ || command == broker_command_ + 1) &&
                (fresh ? lease > broker_lease_ : lease && lease == broker_lease_), "stale serving broker command/lease");
            broker_storage(value, broker_ready_);
            require(field(value, "checkpoint_retained").unsigned_integer() <= field(broker_ready_, "checkpoint_slots").unsigned_integer(),
                "serving broker retained count exceeds capacity");
            broker_command_ = command; broker_lease_ = lease; broker_resync_ = false;
        };
        const auto sampling_matches = [&](const Json& value, size_t prefix, uint64_t expected_seed) {
            const auto& sampled = field(value, "sampling");
            require(field(sampled, "valid").boolean() && field(sampled, "entropy_prefix").unsigned_integer() == prefix &&
                field(sampled, "seed").unsigned_integer() == expected_seed, "worker sampling prefix/seed mismatch");
            for (const auto& [name, value] : request.sampling.object())
                require(number(field(sampled, name), 0, 1e9) == number(value, 0, 1e9), "worker sampling policy changed");
        };
        const auto control = [&](Json job, std::string_view operation, bool fresh) {
            cancelled();
            const auto done = target(job, [](const Json&) { throw FormatError("zero-step broker control emitted a token"); });
            cancelled();
            require(field(done, "event").string() == "control_done" && field(done, "op").string() == operation &&
                field(done, "model_steps").unsigned_integer() == 0 && field(done, "generated_ids").array().empty(),
                "invalid zero-step broker completion");
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (branch8_) {
                validate_branch8_control(verify_ready_, verify_previous_, job, done, verify_resync_);
                if (operation == "branch_capture") ++checkpoint_captures;
                else { checkpoint_restores += field(done, "outcome").string() == "restored"; checkpoint_cold_resyncs += verify_resync_; }
                verify_previous_ = done; verify_resync_ = false;
            }
#endif
            broker_reply(done, fresh);
            const auto prefix = field(done, "committed_prefix").unsigned_integer();
            require(prefix <= prompt.size(), "broker restored beyond future prompt"); sampling_matches(done, prefix, seed);
            const bool logits = field(done, "logits_available").boolean(); const auto& stop = field(done, "stop_id");
            require(logits == bool(prefix) && (stop.is_null() || (logits && codec_.admission.is_stop(stop.unsigned_integer()))) &&
                field(done, "continuation_allowed").boolean() == (logits && stop.is_null()), "invalid broker terminal metadata");
            return done;
        };
        const auto prefill_start = std::chrono::steady_clock::now();
        bool began = false;
        if (broker_epoch_ && !append && number(field(request.sampling, "temperature"), 0, 100) == 0) {
            Json::Array future; future.reserve(prompt.size()); for (auto id : prompt) future.push_back(Json::integer(id));
            const bool reuse = reusable_;
            const auto done = control(Json(Json::Object{{"op", Json("branch_begin")}, {"prompt_ids", Json(std::move(future))},
                {"generation_requested", Json(true)}, {"allow_restore", Json(reuse)}, {"sampling", request.sampling}, {"seed", Json::integer(seed)}}), "branch_begin", true);
            const auto prefix = field(done, "committed_prefix").unsigned_integer(), lcp = field(done, "common_prefix").unsigned_integer();
            const auto outcome = field(done, "outcome").string();
            require(lcp == (reuse ? common : 0) && prefix <= lcp &&
                ((outcome == "cold_miss" && !prefix) || (outcome == "restored" && prefix && reuse)), "broker exact-LCP transaction mismatch");
            require(prefix < prompt.size() || field(done, "stop_id").is_null(), "terminal restored prompt cannot generate without new input");
            // Reserved at preparation; shrinking cannot allocate after restore.
            committed_.resize(size_t(prefix)); position = size_t(prefix); cached_prefix = position; began = true;
        } else if (!append) {
            if (restored_lane_idx) {
                parked_reasoning_.erase(parked_reasoning_.begin() + ptrdiff_t(*restored_lane_idx));
                restored_lane_idx.reset();
            }
            park_active_reasoning();
            committed_.clear();
        }
        if (!append) reusable_ = false;
        const auto input_start = position;
        const auto run = [&](Json job, std::optional<uint32_t> input_token, bool decode) {
            cancelled(); std::vector<uint32_t> events;
            const auto done = target(job, [&](const Json& event) {
                require(decode && field(event, "event").string() == "token", "unexpected worker token event");
                const auto id = field(event, "id").unsigned_integer(); require(id < codec_.admission.vocabulary && events.empty(), "invalid worker token count/ID");
                require(field(event, "committed_prefix").unsigned_integer() == position + 1, "worker token prefix mismatch"); events.push_back(uint32_t(id));
                if (broker_epoch_) require(field(event, "broker_epoch").unsigned_integer() == broker_epoch_ &&
                    field(event, "command_id").unsigned_integer() == broker_command_ + 1 &&
                    field(event, "lease_id").unsigned_integer() == broker_lease_, "stale serving broker token event");
            });
            require(field(done, "event").string() == "done" && field(done, "committed_prefix").unsigned_integer() == position + 1, "worker completion prefix mismatch");
            worker_execution(done, stack_execution_);
            broker_reply(done, field(job, "reset").boolean()); sampling_matches(done, position + 1, seed);
            const auto& ids = field(done, "generated_ids").array(); require(ids.size() == (decode ? 1U : 0U) && events.size() == ids.size(), "worker generated count mismatch");
            if (decode) require(ids[0].unsigned_integer() == events[0], "worker token/done disagreement");
            ++position; committed_.push_back(input_token ? *input_token : events.at(0));
            return done;
        };
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        uint64_t controller_committed = 0, controller_emitted = 0, controller_pending = 0;
        Json controller_feedback;
        const auto run_mtp = [&](Json job) {
            cancelled(); MtpWireReply reply(mtp_ready_, mtp_previous_, job, mtp_resync_);
            const auto done = target(job, [&](const Json& event) { reply.event(event); });
            reply.finish(done); cancelled();
            if (const auto* generation = done.find("native_mtp_generation")) {
                if (const auto* committed = generation->find("committed")) controller_committed = committed->unsigned_integer();
                if (const auto* emitted = generation->find("emitted")) controller_emitted = emitted->unsigned_integer();
                if (const auto* pending = generation->find("overflow")) controller_pending = pending->unsigned_integer();
                if (const auto* feedback = generation->find("stage_a_controller")) controller_feedback = *feedback;
            }
            const uint64_t effective_seed = reply.lane_seed().value_or(seed);
            worker_execution(done, stack_execution_); sampling_matches(done, field(done, "committed_prefix").unsigned_integer(), effective_seed);
            const auto& inputs = field(job, "tokens").array();
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
            if (mtp_ready_.find("native_raw_decode_mode")) {
                const auto routes = raw_decode_routes_value(field(done, "native_raw_decode_request_routes"));
                for (size_t i = 0; i < routes.size(); ++i) {
                    require(routes[i] <= std::numeric_limits<uint64_t>::max() - raw_decode_admitted[i], "admitted raw decode route overflow");
                    raw_decode_admitted[i] += routes[i];
                }
            }
#endif
            if (mtp_hc_rows(mtp_ready_)) for (size_t i = 0; i < 3; ++i) {
                const auto calls = field(done, mtp_hc_rows_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - hc_rows_admitted[i], "admitted HC rows overflow"); hc_rows_admitted[i] += calls;
            }
            if (mtp_router_window(mtp_ready_)) for (size_t i = 0; i < 4; ++i) {
                const auto calls = field(done, mtp_router_window_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - router_window_admitted[i], "admitted router window overflow"); router_window_admitted[i] += calls;
            }
            if (mtp_hc_down_window(mtp_ready_)) {
                const auto calls = field(done, mtp_hc_down_window_request_name).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - hc_down_window_admitted, "admitted HC down window overflow"); hc_down_window_admitted += calls;
            }
            if (mtp_qsa_sparse(mtp_ready_)) for (size_t i = 0; i < 2; ++i) {
                const auto calls = field(done, mtp_qsa_sparse_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - qsa_sparse_admitted[i], "admitted sparse QSA overflow"); qsa_sparse_admitted[i] += calls;
            }
            if (mtp_gdn_ported(mtp_ready_)) for (size_t i = 0; i < 5; ++i) {
                const auto calls = field(done, mtp_gdn_ported_request_names[i]).unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - gdn_ported_admitted[i], "admitted ported GDN overflow"); gdn_ported_admitted[i] += calls;
            }
            if (mtp_hc_down_f32(mtp_ready_)) {
                const auto calls = field(done, "native_prefill_hc_down_f32_request_encoded_calls").unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - hc_down_f32_admitted_calls, "admitted HC down encoded call overflow"); hc_down_f32_admitted_calls += calls;
            }
            if (mtp_gdn_old_order(mtp_ready_)) {
                const auto calls = field(done, "native_gdn_old_order_request_encoded_calls").unsigned_integer();
                require(calls <= std::numeric_limits<uint64_t>::max() - gdn_old_order_admitted_calls, "admitted old-order GDN call count overflow");
                gdn_old_order_admitted_calls += calls;
            }
            if (native_trace) native_trace->admit(mtp_previous_, done, job, reply.events());
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
            if (const auto* selected=mtp_ready_.find("q8_three_rows_mode");selected&&selected->string()=="on") {
                const auto& routes=field(done,"q8_three_rows_request_routes");
                for(auto& [name,total]:q8_three_admitted_routes){const auto n=field(routes,name).unsigned_integer(),before=total.unsigned_integer();require(n<=std::numeric_limits<uint64_t>::max()-before,"admitted N3 route overflow");total=Json::integer(before+n);}
            }
#endif
            if (!inputs.empty()) {
                restored_input_tokens += reply.restored_input_tokens();
                ++input_batches; input_width = std::max(input_width, inputs.size());
                for (const auto& id : inputs) committed_.push_back(uint32_t(id.unsigned_integer()));
            } else {
                const auto& outputs = field(done, "generated_ids").array();
                ++output_batches; output_width = std::max(output_width, outputs.size());
                generation_rounds += field(done, "native_mtp_rounds").unsigned_integer() - field(mtp_previous_, "native_mtp_rounds").unsigned_integer();
                accepted_tokens += field(done, "native_mtp_accepted").unsigned_integer() - field(mtp_previous_, "native_mtp_accepted").unsigned_integer();
                for (const auto& id : outputs) committed_.push_back(uint32_t(id.unsigned_integer()));
            }
            seed = effective_seed;
            position = size_t(field(done, "committed_prefix").unsigned_integer()); mtp_previous_ = done; mtp_resync_ = false;
            return done;
        };
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        const auto run_many = [&](Json job) {
            cancelled();
            Verify8Reply reply(verify_ready_, verify_previous_, job, uint32_t(position), request.sampling, seed, verify_resync_
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                , branch8_
#endif
                );
            const auto done = target(job, [&](const Json& event) {
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (branch8_) require(field(event, "broker_epoch").unsigned_integer() == broker_epoch_ &&
                    field(event, "command_id").unsigned_integer() == broker_command_ + 1 && field(event, "lease_id").unsigned_integer() == broker_lease_ &&
                    field(event, "broker_protocol").string() == "branch8-v1" && field(event, "broker_batch_width").unsigned_integer() == 8,
                    "stale serving branch8 batched event");
#endif
                reply.event(event);
            });
            reply.finish(done); cancelled(); // No response parsing or history publication before full admission.
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            if (branch8_) broker_reply(done, field(job, "reset").boolean());
#endif
            const auto count = field(job, "tokens").array().size();
            if (count) {
                ++input_batches; input_width = std::max(input_width, count);
                for (const auto& id : field(job, "tokens").array()) committed_.push_back(uint32_t(id.unsigned_integer()));
            } else {
                ++output_batches; output_width = std::max(output_width, field(done, "generated_ids").array().size());
                generation_rounds += field(done, "lookup_rounds").unsigned_integer();
                verified_rounds += field(done, "lookup_verified_rounds").unsigned_integer();
                accepted_tokens += field(done, "lookup_accepted_tokens").unsigned_integer();
                for (const auto& id : field(done, "generated_ids").array()) committed_.push_back(uint32_t(id.unsigned_integer()));
            }
            position = size_t(field(done, "committed_prefix").unsigned_integer()); verify_previous_ = done; verify_resync_ = false;
            return done;
        };
        if (verify8_) {
            while (position < prompt.size()) {
                size_t n = std::min<size_t>(8, prompt.size() - position);
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (branch8_) n = std::min(n, 512 - position % 512); // Never step over a useful ancestor boundary.
#endif
                Json::Array ids; ids.reserve(n);
                for (size_t t = 0; t < n; ++t) ids.push_back(Json::integer(prompt[position + t]));
                const bool fresh = !append && !began && !position;
                Json::Object job{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)}, {"reset", Json(fresh)}};
                if (fresh) { job.emplace_back("sampling", request.sampling); job.emplace_back("seed", Json::integer(seed)); job.emplace_back("prompt_lookup", Json(true)); }
                static_cast<void>(run_many(Json(std::move(job))));
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                if (branch8_ && (position % 512 == 0 || position == prompt.size())) {
                    const auto captured = control(Json(Json::Object{{"op", Json("branch_capture")}}), "branch_capture", false);
                    require(field(captured, "outcome").string() == "captured" && field(captured, "committed_prefix").unsigned_integer() == position &&
                        field(captured, "common_prefix").unsigned_integer() == position, "branch8 scheduled capture mismatch");
                }
#endif
            }
        } else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        if (sampled_prefill_
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            || mtp_
#endif
            ) {
            while (position < prompt.size()) {
                cancelled();
                size_t input_packet = 8;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (mtp_) input_packet = mtp_input_packet(mtp_ready_);
#endif
                const auto n = std::min(input_packet, prompt.size() - position);
                Json::Array ids; ids.reserve(n);
                for (size_t t = 0; t < n; ++t) ids.push_back(Json::integer(prompt[position + t]));
                const bool fresh = !append && !began && position == 0;
                Json::Object job{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)},
                    {"reset", Json(fresh)}};
                if (fresh) {
                    job.emplace_back("sampling", request.sampling); job.emplace_back("seed", Json::integer(seed));
                }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                if (mtp_) {
                    if (fresh && lane_parking_ && !request.seed) job.emplace_back("lane_inherit_seed", Json(true));
                    const bool lane_prompt = fresh && lane_parking_ && n < prompt.size();
                    if (lane_prompt) {
                        Json::Array all; all.reserve(prompt.size());
                        for (const auto id : prompt) all.push_back(Json::integer(id));
                        job.emplace_back("lane_prompt", Json(std::move(all)));
                    }
                    if (mtp_final_input_logits(mtp_ready_)) job.emplace_back("prefill_final", Json(position + n == prompt.size()));
                    const size_t packet_end = position + n;
                    static_cast<void>(run_mtp(Json(std::move(job))));
                    if (lane_prompt && position > packet_end) {
                        require(position < prompt.size() && committed_.size() == packet_end,
                            "parked-lane restore exceeds the prompt");
                        committed_.insert(committed_.end(), prompt.begin() + ptrdiff_t(packet_end), prompt.begin() + ptrdiff_t(position));
                        restored_input_tokens += uint64_t(position - packet_end);
                    }
                    continue;
                }
#endif
                const auto done = target(Json(std::move(job)), [](const Json&) { throw FormatError("known input emitted a generated token"); });
                cancelled(); // Never publish serving history after cancelled/ambiguous device work.
                require(field(done, "event").string() == "done" &&
                    field(done, "sampled_prefill_protocol").string() == "sampled-prefill-input8-v1" &&
                    field(done, "committed_prefix").unsigned_integer() == position + n &&
                    field(done, "generated_ids").array().empty() && field(done, "status").string() == "prefilled" &&
                    field(done, "finish_reason").is_null() && field(done, "stop_id").is_null(), "invalid sampled input8 completion");
                worker_execution(done, stack_execution_); sampling_matches(done, position + n, seed);
                for (size_t t = 0; t < n; ++t) committed_.push_back(prompt[position + t]);
                position += n;
            }
        } else
#endif
        for (size_t i = position; i < prompt.size(); ++i) {
            Json::Object job{{"tokens", Json(Json::Array{Json::integer(prompt[i])})}, {"generate", Json::integer(0)}, {"reset", Json(!append && !began && i == 0)}};
            if (!append && !began && i == 0) { job.emplace_back("sampling", request.sampling); job.emplace_back("seed", Json::integer(seed)); }
            const auto done = run(Json(std::move(job)), prompt[i], false);
            require(field(done, "status").string() == "prefilled" && field(done, "finish_reason").is_null() && field(done, "stop_id").is_null(), "invalid prefill termination");
        }
        if (broker_epoch_ && position > input_start
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            && !branch8_
#endif
            ) {
            const auto done = control(Json(Json::Object{{"op", Json("branch_capture")}}), "branch_capture", false);
            require(field(done, "outcome").string() == "captured" && field(done, "committed_prefix").unsigned_integer() == position &&
                field(done, "common_prefix").unsigned_integer() == position, "broker input-boundary capture mismatch");
        }
        const size_t prefill_committed_end = position;
        require(restored_input_tokens <= prompt.size() - cached_prefix, "restored input accounting exceeds uncached prompt");
        cached_prefix += size_t(restored_input_tokens);
        const auto generation_start = std::chrono::steady_clock::now();
        bool stopped = false;
        std::optional<size_t> forced_thinking_close_index;
        const auto encode_forced_tokens = [&](std::string_view text) {
            auto ids = codec_.encode(RenderedChat{std::string(text), Json(Json::Array{})});
            require(!ids.empty(), "forced thinking close encoded no tokens");
            for (const auto id : ids) {
                require(id < codec_.admission.vocabulary && id != codec_.admission.im_start && !codec_.admission.is_stop(id),
                    "forced thinking close token outside vocabulary or terminal");
            }
            return ids;
        };
        const auto run_generation_phase = [&](size_t phase_limit) {
            if (stopped || generated.size() >= phase_limit) return;
            const size_t phase_budget = phase_limit - generated.size();
            static_cast<void>(phase_budget);
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (mtp_) {
                controller_committed = controller_emitted = controller_pending = 0; controller_feedback = Json();
                const bool scoped_carry = mtp_request_carry(mtp_ready_);
                if (scoped_carry) { require(mtp_generation_id_ < std::numeric_limits<uint64_t>::max(), "MTP generation ID exhausted"); ++mtp_generation_id_; }
                const size_t output_cap = mtp_output_packet(mtp_ready_);
                const auto build_mtp_generation = [&](bool begin, size_t rem) {
                    Json::Object generation{{"id", Json::integer(mtp_generation_id_)},
                        {"begin", Json(begin)}, {"remaining", Json::integer(rem)}};
                    if (field(mtp_ready_, "native_mtp_carry_scope").string() == "http-generation-v2") {
                        generation.emplace_back("committed", Json::integer(controller_committed));
                        generation.emplace_back("emitted", Json::integer(controller_emitted));
                        generation.emplace_back("pending", Json::integer(controller_pending));
                        generation.emplace_back("remaining_compute", Json::integer(phase_budget - std::min<uint64_t>(phase_budget, controller_committed)));
                        generation.emplace_back("total_budget", Json::integer(phase_budget));
                        if (!controller_feedback.is_null()) generation.emplace_back("stage_a_controller", controller_feedback);
                    }
                    return Json(std::move(generation));
                };
                size_t phase_emitted = 0;
                while (phase_emitted < phase_budget) {
                    const size_t packet_cap = (generated.empty() && mtp_first_output_packet_)
                        ? std::min<size_t>(*mtp_first_output_packet_, output_cap)
                        : output_cap;
                    const auto n = std::min<size_t>(packet_cap, phase_budget - phase_emitted);
                    Json::Object job{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(n)}, {"reset", Json(false)},
                        {"stop_ids", Json(Json::Array{Json::integer(codec_.admission.im_end), Json::integer(codec_.admission.end_of_text)})}};
                    if (scoped_carry) job.emplace_back("mtp_generation", build_mtp_generation(phase_emitted == 0, phase_budget - phase_emitted));
                    const auto done = run_mtp(Json(std::move(job)));
                    for (const auto& token : field(done, "generated_ids").array()) {
                        const auto id = uint32_t(token.unsigned_integer());
                        require(id != codec_.admission.im_start && !stopped, "MTP unexpected generated role/terminal");
                        stopped = codec_.admission.is_stop(id); generated.push_back(id); ++phase_emitted;
                        require(stopped || feed_id(generated.back()) != ResponseStatus::error,
                            "response bytes/frame validation failed");
                        if (request.stream) flush();
                    }
                    if (stopped) break;
                }
            } else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
            if (verify8_) {
                while (generated.size() < phase_limit) {
                    const auto n = std::min<size_t>(8, phase_limit - generated.size());
                    const auto done = run_many(Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(n)}, {"reset", Json(false)},
                        {"stop_ids", Json(Json::Array{Json::integer(codec_.admission.im_end), Json::integer(codec_.admission.end_of_text)})}}));
                    for (const auto& token : field(done, "generated_ids").array()) {
                        const auto id = uint32_t(token.unsigned_integer());
                        require(id != codec_.admission.im_start && !stopped, "private V8 unexpected generated role/terminal");
                        stopped = codec_.admission.is_stop(id); generated.push_back(id);
                        require(stopped || feed_id(generated.back()) != ResponseStatus::error, "response bytes/frame validation failed");
                        if (request.stream) flush();
                    }
                    if (stopped) break;
                }
            } else
#endif
            while (generated.size() < phase_limit) {
                const auto done = run(Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(1)}, {"reset", Json(false)},
                    {"stop_ids", Json(Json::Array{Json::integer(codec_.admission.im_end), Json::integer(codec_.admission.end_of_text)})}}), {}, true);
                const auto id = committed_.back();
                require(id != codec_.admission.im_start, "unexpected generated role/terminal token");
                stopped = codec_.admission.is_stop(id);
                require(field(done, "finish_reason").string() == (stopped ? "stop" : "length") && field(done, "status").string() == (stopped ? "stopped" : "length"), "worker termination mismatch");
                if (stopped) require(field(done, "stop_id").unsigned_integer() == id && !field(done, "continuation_allowed").boolean(), "invalid terminal admission");
                else require(field(done, "stop_id").is_null() && field(done, "continuation_allowed").boolean(), "invalid continuation admission");
                generated.push_back(id); require(stopped || feed_id(generated.back()) != ResponseStatus::error, "response bytes/frame validation failed");
                if (request.stream) flush();
                if (stopped) break;
            }
        };
        const auto inject_forced_thinking_close = [&] {
            const auto nl_ids = parser.thinking_ends_with_newline() ? std::vector<uint32_t>{} : encode_forced_tokens("\n");
            const auto close_ids = encode_forced_tokens("</think>");
            const auto nl2_ids = encode_forced_tokens("\n\n");
            std::vector<uint32_t> inject;
            inject.reserve(nl_ids.size() + close_ids.size() + nl2_ids.size());
            inject.insert(inject.end(), nl_ids.begin(), nl_ids.end());
            const size_t close_offset = inject.size();
            inject.insert(inject.end(), close_ids.begin(), close_ids.end());
            inject.insert(inject.end(), nl2_ids.begin(), nl2_ids.end());
            const size_t rem = size_t(maximum_tokens) - generated.size();
            if (inject.size() > rem) inject.resize(rem);
            if (inject.empty()) return;
            if (inject.size() > close_offset) forced_thinking_close_index = generated.size() + close_offset;
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
            if (verify8_) {
                for (size_t at = 0; at < inject.size();) {
                    const size_t n = std::min<size_t>(8, inject.size() - at);
                    Json::Array ids; ids.reserve(n);
                    for (size_t t = 0; t < n; ++t) ids.push_back(Json::integer(inject[at + t]));
                    static_cast<void>(run_many(Json(Json::Object{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)}, {"reset", Json(false)}})));
                    at += n;
                }
            } else
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            if (sampled_prefill_
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                || mtp_
#endif
                ) {
                for (size_t at = 0; at < inject.size();) {
                    cancelled();
                    size_t input_packet = 8;
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    if (mtp_) input_packet = mtp_input_packet(mtp_ready_);
#endif
                    const auto n = std::min<size_t>(input_packet, inject.size() - at);
                    Json::Array ids; ids.reserve(n);
                    for (size_t t = 0; t < n; ++t) ids.push_back(Json::integer(inject[at + t]));
                    Json::Object job{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)}, {"reset", Json(false)}};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    if (mtp_) {
                        if (mtp_final_input_logits(mtp_ready_)) job.emplace_back("prefill_final", Json(at + n == inject.size()));
                        static_cast<void>(run_mtp(Json(std::move(job))));
                        at += n;
                        continue;
                    }
#endif
                    const auto done = target(Json(std::move(job)), [](const Json&) { throw FormatError("known input emitted a generated token"); });
                    cancelled();
                    require(field(done, "event").string() == "done" &&
                        field(done, "sampled_prefill_protocol").string() == "sampled-prefill-input8-v1" &&
                        field(done, "committed_prefix").unsigned_integer() == position + n &&
                        field(done, "generated_ids").array().empty() && field(done, "status").string() == "prefilled" &&
                        field(done, "finish_reason").is_null() && field(done, "stop_id").is_null(), "invalid sampled input8 completion");
                    worker_execution(done, stack_execution_); sampling_matches(done, position + n, seed);
                    for (size_t t = 0; t < n; ++t) committed_.push_back(inject[at + t]);
                    position += n;
                    at += n;
                }
            } else
#endif
            for (const auto token : inject) {
                Json::Object job{{"tokens", Json(Json::Array{Json::integer(token)})}, {"generate", Json::integer(0)}, {"reset", Json(false)}};
                const auto done = run(Json(std::move(job)), token, false);
                require(field(done, "status").string() == "prefilled" && field(done, "finish_reason").is_null() && field(done, "stop_id").is_null(), "invalid prefill termination");
            }
            for (size_t i = 0; i < inject.size(); ++i) {
                if (i == close_offset) parser.prepare_forced_thinking_close();
                generated.push_back(inject[i]);
                require(feed_id(generated.back()) != ResponseStatus::error, "response bytes/frame validation failed");
                if (request.stream) flush();
            }
        };
        if (request.thinking && request.thinking_budget && *request.thinking_budget < maximum_tokens) {
            run_generation_phase(*request.thinking_budget);
            while (!stopped && parser.in_thinking() && parser.has_incomplete_utf8() && generated.size() < maximum_tokens) {
                run_generation_phase(generated.size() + 1);
            }
            if (!stopped && parser.in_thinking() && generated.size() < maximum_tokens) {
                inject_forced_thinking_close();
            }
            if (!stopped && generated.size() < maximum_tokens) {
                run_generation_phase(maximum_tokens);
            }
        } else {
            run_generation_phase(maximum_tokens);
        }
        if (!request.stream) flush();
        Json::Array ids; for (const auto id : generated) ids.push_back(Json::integer(id));
        Json::Object final_req{{"ids", Json(std::move(ids))}, {"tools", parser_manifest(request.manifest)}, {"initial_thinking", Json(request.thinking)},
            {"finish_reason", Json(stopped ? "stop" : "length")}, {"stop_id", stopped ? Json::integer(generated.back()) : Json()}};
        if (tool_args_ == ToolArgValidation::executor) final_req.emplace_back("executor_parameters", Json(true));
        if (forced_thinking_close_index) final_req.emplace_back("forced_thinking_close_index", Json::integer(*forced_thinking_close_index));
        const auto final = parse_native_response_with_controls(Json(std::move(final_req)), codec_.admission, codec_.decode, response_maximum_accumulated_ids, codec_.control_ids);
        require(!request.single_tool_call || field(final, "tool_calls").array().size() <= 1, "parallel tool calls are disabled for this request");
        Json::Array calls;
        for (const auto& call : field(final, "tool_calls").array()) {
            const auto& function = field(call, "function"); const auto name = field(function, "name").string();
            const auto found = std::find(request.names.begin(), request.names.end(), name); require(found != request.names.end(), "unadmitted final tool");
            if (tool_args_ == ToolArgValidation::executor) validate_executor_tool_arguments(field(function, "arguments"));
            else validate_tool_value(request.schemas.at(size_t(found - request.names.begin())), field(function, "arguments"), tool_enums_);
            calls.emplace_back(Json::Object{{"id", Json(std::string(response_id) + "_call_" + std::to_string(calls.size()))}, {"type", Json("function")},
                {"function", Json(Json::Object{{"name", Json(std::string(name))}, {"arguments", Json(field(function, "arguments").dump())}})}});
        }
        require((!request.named_tool_choice && (request.tool_choice == "auto" || request.tool_choice == "none")) || !stopped || !calls.empty(), "required tool was not produced");
        cancelled();
        const auto content = std::string(field(final, "content").string()), thinking = std::string(field(final, "reasoning_content").string());
        require(content.starts_with(streamed_content) && thinking.starts_with(streamed_thinking), "final response disagrees with streamed prefix");
        if (thinking.size() > streamed_thinking.size()) emit(Json(Json::Object{{"reasoning_content", Json(thinking.substr(streamed_thinking.size()))}}));
        if (content.size() > streamed_content.size()) emit(Json(Json::Object{{"content", Json(content.substr(streamed_content.size()))}}));
        for (size_t i = 0; i < calls.size(); ++i) {
            auto fields = calls[i].object(); fields.emplace_back("index", Json::integer(i));
            emit(Json(Json::Object{{"tool_calls", Json(Json::Array{Json(std::move(fields))})}}));
        }
        const std::string finish = !calls.empty() ? "tool_calls" : stopped ? "stop" : "length";
        emit(Json(Json::Object{}), Json(finish));
        sampling_ = policy; seed_ = seed; reusable_ = true;
        if (!append) saved_reasoning_ = restored_reasoning ? std::move(*restored_reasoning) : std::vector<std::pair<size_t, std::string>>{};
        if (!thinking.empty()) {
            size_t ordinal = 0;
            for (const auto& message : field(request.render, "messages").array()) ordinal += field(message, "role").string() == "assistant";
            saved_reasoning_.emplace_back(ordinal, thinking);
        }
        const auto generation_end = std::chrono::steady_clock::now();
        const auto admission_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(prefill_start - request_start).count();
        const auto prefill_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(generation_start - prefill_start).count();
        const auto generation_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(generation_end - generation_start).count();
        const auto request_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(generation_end - request_start).count();
        const double server_admission_wall_ms = static_cast<double>(admission_ns) / 1e6;
        const double server_prefill_wall_ms = static_cast<double>(prefill_ns) / 1e6;
        const double server_generation_wall_ms = static_cast<double>(generation_ns) / 1e6;
        const double server_request_wall_ms = static_cast<double>(request_ns) / 1e6;
        const double server_generation_throughput = (!generated.empty() && generation_ns > 0)
            ? (static_cast<double>(generated.size()) * 1e9) / static_cast<double>(generation_ns) : 0.0;
        std::string_view prefill_generation_overlap = "none";
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_ && prefill_committed_end > input_start) prefill_generation_overlap = "known_input_mtp_priming";
#else
        static_cast<void>(prefill_committed_end);
#endif
        Json::Object message{{"role", Json("assistant")}, {"content", Json(content)}};
        if (!thinking.empty()) message.emplace_back("reasoning_content", Json(thinking));
        if (!calls.empty()) message.emplace_back("tool_calls", Json(std::move(calls)));
        Json::Object result{{"id", Json(std::string(response_id))}, {"object", Json("chat.completion")}, {"created", Json::integer(created)}, {"model", Json(alias_)},
            {"choices", Json(Json::Array{Json(Json::Object{{"index", Json::integer(0)}, {"message", Json(std::move(message))}, {"finish_reason", Json(finish)}})})},
            {"usage", Json(Json::Object{{"prompt_tokens", Json::integer(prompt.size())}, {"completion_tokens", Json::integer(generated.size())}, {"total_tokens", Json::integer(prompt.size() + generated.size())},
                {"prompt_tokens_details", Json(Json::Object{{"cached_tokens", Json::integer(cached_prefix)}})}})}};
        if (sink.prefix_trace) {
            Json::Object trace{{"event", Json("completion")}, {"schema", Json::integer(1)}, {"response_id", Json(std::string(response_id))},
                {"prompt_tokens", Json::integer(prompt.size())}, {"cached_tokens", Json::integer(cached_prefix)}, {"new_tokens", Json::integer(prompt.size() - cached_prefix)},
                {"completion_tokens", Json::integer(generated.size())}, {"total_tokens", Json::integer(prompt.size() + generated.size())},
                {"finish_reason", Json(finish)}, {"stopped_on_eos", Json(stopped)},
                {"server_admission_wall_ms", real(server_admission_wall_ms)},
                {"server_prefill_wall_ms", real(server_prefill_wall_ms)},
                {"server_generation_wall_ms", real(server_generation_wall_ms)},
                {"server_request_wall_ms", real(server_request_wall_ms)},
                {"server_generation_throughput", real(server_generation_throughput)},
                {"prefill_generation_overlap", Json(std::string(prefill_generation_overlap))}};
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
            if (native_trace) trace.emplace_back("native_worker", native_trace->report());
#ifdef TOKSPUR_PRIVATE_HTTP_PREFILL512
            if (mtp_ && mtp_ready_.find("native_prefill_dense_kernel")) {
                trace.emplace_back("native_prefill_dense", field(mtp_ready_, "native_prefill_dense"));
                trace.emplace_back("native_prefill_dense_kernel", field(mtp_ready_, "native_prefill_dense_kernel"));
            }
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
            if (mtp_ && mtp_ready_.find("native_raw_decode_mode")) {
                Json::Object route;
                for (const auto name : raw_decode_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                route.emplace_back("admitted_routes", raw_decode_routes_json(raw_decode_admitted));
                route.emplace_back("scope", Json("ported-kernel successful completed physical dispatches from admitted native packets in this HTTP request, only where eligible raw routed calls execute; cancelled unadmitted packets excluded"));
                trace.emplace_back("raw_decode", Json(std::move(route)));
            }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
            if (mtp_) if (const auto* selected=mtp_ready_.find("q8_three_rows_mode");selected&&selected->string()=="on") trace.emplace_back("q8_three_rows",Json(Json::Object{
                {"schema",field(mtp_ready_,"q8_three_rows_schema")},{"mode",*selected},{"padding_rows",field(mtp_ready_,"q8_three_rows_padding_rows")},{"hc_route",field(mtp_ready_,"q8_three_rows_hc_route")},
                {"admitted_routes",Json(q8_three_admitted_routes)},{"scope",Json("completed routes from admitted native packets in this HTTP request only")}}));
#endif
            if (mtp_ && mtp_hc_rows(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_hc_rows_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : hc_rows_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_encoded_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("encoded inject/read/write calls from admitted packets only; not completed GPU work"));
                trace.emplace_back("prefill_hc_rows", Json(std::move(route)));
            }
            if (mtp_ && mtp_router_window(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_router_window_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : router_window_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_encoded_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("encoded router/route/shared/mixture calls from admitted packets only; not completed GPU work"));
                trace.emplace_back("prefill_router_window", Json(std::move(route)));
            }
            if (mtp_ && mtp_hc_down_window(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_hc_down_window_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                route.emplace_back("admitted_encoded_calls", Json::integer(hc_down_window_admitted));
                route.emplace_back("scope", Json("encoded HC down window calls from admitted packets only; not completed GPU work"));
                trace.emplace_back("prefill_hc_down_window", Json(std::move(route)));
            }
            if (mtp_ && mtp_qsa_sparse(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_qsa_sparse_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : qsa_sparse_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_completed_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("completed decode/verify sparse QSA calls from admitted packets only; cancelled unadmitted packets excluded"));
                trace.emplace_back("qsa_sparse", Json(std::move(route)));
            }
            if (mtp_ && mtp_gdn_ported(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_gdn_ported_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : gdn_ported_admitted) calls.push_back(Json::integer(n));
                route.emplace_back("admitted_completed_calls", Json(std::move(calls)));
                route.emplace_back("scope", Json("completed prefill/decode/verify/replay/norm_gate ported GDN calls from admitted packets only; cancelled unadmitted packets excluded"));
                trace.emplace_back("gdn_ported", Json(std::move(route)));
            }
            if (mtp_ && mtp_hc_down_f32(mtp_ready_)) {
                Json::Object route; for (const auto name : mtp_hc_down_f32_identity_names) route.emplace_back(std::string(name), field(mtp_ready_, name));
                route.emplace_back("admitted_encoded_calls", Json::integer(hc_down_f32_admitted_calls));
                route.emplace_back("scope", Json("encoded calls from admitted native packets only; cancelled/rejected packets excluded; not completed GPU work"));
                trace.emplace_back("prefill_hc_down_f32", Json(std::move(route)));
            }
            if (mtp_ && mtp_gdn_old_order(mtp_ready_)) trace.emplace_back("gdn_old_order", Json(Json::Object{
                {"version", field(mtp_ready_, "native_gdn_old_order_version")}, {"kernel", field(mtp_ready_, "native_gdn_old_order_kernel")},
                {"admitted_encoded_calls", Json::integer(gdn_old_order_admitted_calls)}, {"scope", Json("admitted native packets in this HTTP request only")}}));
#endif
#endif
            sink.prefix_trace(Json(std::move(trace)));
        }
        if (request.stream && request.include_usage) {
            cancelled();
            sink.chunk(Json(Json::Object{{"id", Json(std::string(response_id))}, {"object", Json("chat.completion.chunk")},
                {"created", Json::integer(created)}, {"model", Json(alias_)}, {"choices", Json(Json::Array{})},
                {"usage", field(Json(result), "usage")}}));
        }
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        if (verify8_) result.emplace_back("tokspur_private", Json(Json::Object{{"native_verify8", Json(true)}, {"verification_max_tokens", Json::integer(8)},
            {"known_input_batches", Json::integer(input_batches)}, {"maximum_known_input_width", Json::integer(input_width)},
            {"generation_batches", Json::integer(output_batches)}, {"maximum_generation_width", Json::integer(output_width)},
            {"lookup_rounds", Json::integer(generation_rounds)}, {"lookup_verified_rounds", Json::integer(verified_rounds)}, {"lookup_accepted_tokens", Json::integer(accepted_tokens)},
            {"hc_exact_final_logit_sha256", field(verify_previous_, "hc_exact_final_logit_sha256")},
            {"performance_qualified", Json(false)}}));
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        if (mtp_) result.emplace_back("tokspur_private", Json(Json::Object{{"native_mtp", Json(true)},
            {"native_mtp_protocol", field(mtp_ready_, "native_mtp_protocol")}, {"known_input_batches", Json::integer(input_batches)},
            {"maximum_known_input_width", Json::integer(input_width)}, {"generation_batches", Json::integer(output_batches)},
            {"maximum_generation_width", Json::integer(output_width)}, {"mtp_rounds", Json::integer(generation_rounds)},
            {"mtp_accepted_tokens", Json::integer(accepted_tokens)},
            {"performance_qualified", Json(false)}}));
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
#ifdef TOKSPUR_PRIVATE_RAW_DECODE
        if (mtp_ && mtp_ready_.find("native_raw_decode_mode")) {
            auto details = field(Json(result), "tokspur_private").object();
            for (const auto name : raw_decode_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
            details.emplace_back("native_raw_decode_admitted_routes", raw_decode_routes_json(raw_decode_admitted));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
#ifdef TOKSPUR_PRIVATE_Q8_THREE_ROWS
        if (mtp_) if (const auto* selected=mtp_ready_.find("q8_three_rows_mode");selected&&selected->string()=="on") {
            auto details=field(Json(result),"tokspur_private").object();
            for(const auto name:{"q8_three_rows_schema","q8_three_rows_mode","q8_three_rows_padding_rows","q8_three_rows_hc_route"})details.emplace_back(name,field(mtp_ready_,name));
            details.emplace_back("q8_three_rows_admitted_routes",Json(q8_three_admitted_routes));
            for(auto& [name,value]:result)if(name=="tokspur_private")value=Json(std::move(details));
        }
#endif
        if (mtp_ && mtp_input_packet(mtp_ready_) >= 512) {
            auto details = field(Json(result), "tokspur_private").object();
            for (const auto name : {"native_mtp_input_packet_schema", "native_mtp_input_packet_max", "native_mtp_priming_batch_capacity", "native_mtp_priming_batch_bytes", "native_prefill_dense"})
                details.emplace_back(name, field(mtp_ready_, name));
            for (const auto name : {"native_prefill_grouped", "native_prefill_grouped_kernel", "native_prefill_dense_kernel"})
                if (const auto* value = mtp_ready_.find(name)) details.emplace_back(name, *value);
            if (mtp_hc_rows(mtp_ready_)) {
                for (const auto name : mtp_hc_rows_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : hc_rows_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_prefill_hc_rows_admitted_encoded_calls", Json(std::move(calls)));
                details.emplace_back("native_prefill_hc_rows_admitted_scope", Json("encoded inject/read/write calls from admitted packets only; not completed GPU work"));
            }
            if (mtp_router_window(mtp_ready_)) {
                for (const auto name : mtp_router_window_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : router_window_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_prefill_router_window_admitted_encoded_calls", Json(std::move(calls)));
                details.emplace_back("native_prefill_router_window_admitted_scope", Json("encoded router/route/shared/mixture calls from admitted packets only; not completed GPU work"));
            }
            if (mtp_hc_down_window(mtp_ready_)) {
                for (const auto name : mtp_hc_down_window_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                details.emplace_back("native_prefill_hc_down_window_admitted_encoded_calls", Json::integer(hc_down_window_admitted));
                details.emplace_back("native_prefill_hc_down_window_admitted_scope", Json("encoded HC down window calls from admitted packets only; not completed GPU work"));
            }
            if (mtp_qsa_sparse(mtp_ready_)) {
                for (const auto name : mtp_qsa_sparse_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : qsa_sparse_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_qsa_sparse_admitted_completed_calls", Json(std::move(calls)));
                details.emplace_back("native_qsa_sparse_admitted_scope", Json("completed decode/verify sparse QSA calls from admitted packets only; cancelled unadmitted packets excluded"));
            }
            if (mtp_gdn_ported(mtp_ready_)) {
                for (const auto name : mtp_gdn_ported_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                Json::Array calls; for (auto n : gdn_ported_admitted) calls.push_back(Json::integer(n));
                details.emplace_back("native_gdn_ported_admitted_completed_calls", Json(std::move(calls)));
                details.emplace_back("native_gdn_ported_admitted_scope", Json("completed prefill/decode/verify/replay/norm_gate ported GDN calls from admitted packets only; cancelled unadmitted packets excluded"));
            }
            if (mtp_hc_down_f32(mtp_ready_)) {
                for (const auto name : mtp_hc_down_f32_identity_names) details.emplace_back(std::string(name), field(mtp_ready_, name));
                details.emplace_back("native_prefill_hc_down_f32_admitted_encoded_calls", Json::integer(hc_down_f32_admitted_calls));
                details.emplace_back("native_prefill_hc_down_f32_admitted_scope", Json("encoded calls from admitted native packets only; cancelled/rejected packets excluded; not completed GPU work"));
            }
            if (mtp_gdn_old_order(mtp_ready_)) {
                for (const auto name : {"native_gdn_old_order_version", "native_gdn_old_order_kernel", "native_gdn_old_order_layers", "native_gdn_old_order_scratch_bytes"}) details.emplace_back(name, field(mtp_ready_, name));
                details.emplace_back("native_gdn_old_order_admitted_encoded_calls", Json::integer(gdn_old_order_admitted_calls));
            }
            if (mtp_final_input_logits(mtp_ready_)) {
                details.emplace_back("native_prefill_output_policy", field(mtp_ready_, "native_prefill_output_policy"));
                details.emplace_back("logits_available", field(mtp_previous_, "logits_available"));
            }
            details.emplace_back("native_mtp_priming_batch_submits", field(mtp_previous_, "native_mtp_priming_batch_submits"));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
        if (mtp_ && (mtp_depth(mtp_ready_) == 2 || mtp_depth(mtp_ready_) == 3 || mtp_depth(mtp_ready_) == 7)) {
            auto details = field(Json(result), "tokspur_private").object();
            details.emplace_back("native_mtp_depth", Json::integer(mtp_depth(mtp_ready_)));
            details.emplace_back("native_mtp_tail_policy", field(mtp_ready_, "native_mtp_tail_policy"));
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
            if (mtp_ready_.find("native_mtp_output_packet")) {
                details.emplace_back("native_mtp_output_packet", field(mtp_ready_, "native_mtp_output_packet"));
                details.emplace_back("native_mtp_output_packet_schema", field(mtp_ready_, "native_mtp_output_packet_schema"));
            }
#endif
            if (mtp_first_output_packet_) details.emplace_back("native_mtp_first_output_packet", Json::integer(*mtp_first_output_packet_));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
        if (mtp_ && mtp_request_carry(mtp_ready_)) {
            auto details = field(Json(result), "tokspur_private").object();
            details.emplace_back("native_mtp_carry_scope", field(mtp_ready_, "native_mtp_carry_scope"));
            details.emplace_back("native_mtp_generation", field(mtp_previous_, "native_mtp_generation"));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        if (branch8_) {
            auto details = field(Json(result), "tokspur_private").object();
            details.emplace_back("branch8_reuse", Json(true)); details.emplace_back("broker_protocol", Json("branch8-v1"));
            details.emplace_back("checkpoint_captures", Json::integer(checkpoint_captures)); details.emplace_back("checkpoint_restores", Json::integer(checkpoint_restores));
            details.emplace_back("checkpoint_cold_resyncs", Json::integer(checkpoint_cold_resyncs));
            details.emplace_back("checkpoint_slots", field(verify_ready_, "checkpoint_slots"));
            for (auto& [name, value] : result) if (name == "tokspur_private") value = Json(std::move(details));
        }
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        return project_weight_lanes(Json(std::move(result)), weight_lanes_identity_);
#else
        return Json(std::move(result));
#endif
    } catch (...) { if (state_touched) invalidate(); throw; }
}

void refuse_busy_connection(http::Connection connection) noexcept {
    try {
        const auto body = Json(Json::Object{{"error", Json(Json::Object{
            {"message", Json("server is busy with other connections; retry shortly")},
            {"type", Json("server_busy")},
            {"code", Json("server_busy")}})}}).dump(8192);
        connection.refuse_json(503, body);
    } catch (...) {
        // The peer is gone or not reading; there is nobody to tell.
    }
}

void admit_reading(std::vector<std::pair<http::Connection, http::Clock::time_point>>& reading,
                   http::Connection connection, http::Clock::time_point now) {
    if (reading.size() >= reading_peers) {
        auto oldest = std::move(reading.front().first);
        reading.erase(reading.begin());
        refuse_busy_connection(std::move(oldest));
    }
    reading.emplace_back(std::move(connection), now);
}

ArrivalQueue::ArrivalQueue(size_t capacity, size_t workers)
    : capacity_(capacity), active_workers_(workers) {}

bool ArrivalQueue::hand_off(Arrival arrival) {
    std::unique_lock lock(mutex_);
    if (closed_ || active_workers_ == 0 || unexpected_exit_ || failure_) return false;
    if (queue_.size() >= capacity_) {
        auto refused = std::move(arrival.connection);
        lock.unlock();
        refuse_busy_connection(std::move(refused));
        return true;
    }
    queue_.push_back(std::move(arrival));
    lock.unlock();
    condition_.notify_one();
    return true;
}

std::optional<Arrival> ArrivalQueue::recv() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (closed_) return std::nullopt;
    Arrival next = std::move(queue_.front());
    queue_.pop_front();
    return next;
}

void ArrivalQueue::worker_stopped(bool orderly, std::exception_ptr failure) noexcept {
    std::lock_guard lock(mutex_);
    if (failure && !failure_) failure_ = failure;
    if (!closed_ && !orderly) unexpected_exit_ = true;
    if (active_workers_ > 0) --active_workers_;
}

void ArrivalQueue::check_workers() const {
    std::lock_guard lock(mutex_);
    if (failure_) std::rethrow_exception(failure_);
    require(!unexpected_exit_ && active_workers_ > 0, "HTTP readers stopped");
}

void ArrivalQueue::close() noexcept {
    {
        std::lock_guard lock(mutex_);
        closed_ = true;
        queue_.clear();
    }
    condition_.notify_all();
}

namespace {
constexpr const char* prefix_trace_full_cstr =
    "private prefix trace is full (16 MiB bound); no request is served until the server is restarted with a new trace file";
constexpr const char* prefix_trace_unavailable_cstr = "private prefix trace is unavailable";
constexpr const char* prefix_trace_refused_cstr = "private prefix trace requires a new owned mode 0600 file";
constexpr const char* prefix_trace_exists_cstr =
    "private prefix trace requires a new file: move or remove prefix-trace.jsonl from the state directory, or start without --private-prefix-trace";
} // namespace

PrefixTraceLog::~PrefixTraceLog() {
    if (fd_ >= 0) ::close(fd_);
}

void PrefixTraceLog::enable(int directory_fd) {
    const int opened = ::openat(directory_fd, "prefix-trace.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (opened < 0) throw FormatError(errno == EEXIST ? prefix_trace_exists_cstr : prefix_trace_refused_cstr);
    struct stat info{};
    if (::fstat(opened, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != ::geteuid() ||
        info.st_nlink != 1 || (info.st_mode & 0777) != 0600) {
        ::close(opened);
        throw FormatError(prefix_trace_refused_cstr);
    }
    if (::fsync(directory_fd) != 0) {
        ::close(opened);
        throw FormatError("cannot persist private prefix trace entry");
    }
    if (fd_ >= 0) ::close(fd_);
    fd_ = opened;
    bytes_ = 0;
    refusal_ = nullptr;
}

void PrefixTraceLog::write(const Json& record) {
    if (refusal_) throw PrefixTraceError(refusal_);
    if (fd_ < 0) {
        refusal_ = prefix_trace_unavailable_cstr;
        throw PrefixTraceError(prefix_trace_unavailable_cstr);
    }
    const auto bytes = record.dump(4 * 1024 * 1024) + '\n';
    if (bytes.size() > prefix_trace_limit_bytes - bytes_) {
        refusal_ = prefix_trace_full_cstr;
        throw PrefixTraceError(prefix_trace_full_cstr);
    }
    size_t offset = 0;
    while (offset < bytes.size()) {
        const auto written = ::write(fd_, bytes.data() + offset, bytes.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            refusal_ = prefix_trace_unavailable_cstr;
            throw PrefixTraceError("cannot persist private prefix trace");
        }
        offset += size_t(written);
    }
    if (::fsync(fd_) != 0) {
        refusal_ = prefix_trace_unavailable_cstr;
        throw PrefixTraceError("cannot persist private prefix trace");
    }
    bytes_ += bytes.size();
}

std::string_view job_failure_kind(const std::exception& error, bool target_started) noexcept {
    if (dynamic_cast<const WorkerHostMemoryPressure*>(&error)) return "host_memory_pressure";
    if (dynamic_cast<const PrefixTraceError*>(&error) != nullptr) return "server_error";
    return target_started ? "generation_failed_no_retry" : "invalid_request_error";
}

int job_failure_status(const std::exception& error, bool expired, bool target_started) noexcept {
    if (expired) return 408;
    if (dynamic_cast<const WorkerHostMemoryPressure*>(&error)) return 503;
    if (dynamic_cast<const PrefixTraceError*>(&error) != nullptr) return 500;
    return target_started ? 502 : 400;
}
} // namespace tokspur
