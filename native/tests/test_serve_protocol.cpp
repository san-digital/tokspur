#include "tokspur/owned_worker.hpp"
#include "tokspur/serve.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <source_location>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace tokspur;
namespace {
size_t checks = 0;
void check(bool value, std::source_location at = std::source_location::current()) { ++checks; if (!value) throw std::runtime_error("serving protocol check failed at line " + std::to_string(at.line())); }
double parse_number(const Json& value) {
    check(value.kind() == Json::Kind::number);
    const auto text = value.number_lexeme(); double result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    check(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(result));
    return result;
}
template<class F> void refusal(F&& function, std::source_location at = std::source_location::current()) {
    bool refused = false; try { function(); } catch (const FormatError&) { refused = true; } check(refused, at);
}
Json with(Json input, std::string_view key, Json value) {
    auto fields = input.object();
    for (auto& [name, item] : fields) if (name == key) { item = std::move(value); return Json(std::move(fields)); }
    fields.emplace_back(key, std::move(value)); return Json(std::move(fields));
}
Json without(const Json& input, std::string_view key) {
    Json::Object result; for (const auto& item : input.object()) if (item.first != key) result.push_back(item);
    return Json(std::move(result));
}
Json replay_metadata(Json input) {
    input = with(std::move(input), "replay_prepared", Json(true));
    input = with(std::move(input), "recurrent_replay_captures", Json::integer(24));
    return with(std::move(input), "mapped_weight_region_limit_bytes", Json::integer(uint64_t(4) * 1024 * 1024 * 1024));
}
void worker_ready_admission() {
    const auto ready = parse_json(R"({"event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct","maximum_generated_tokens":4096})");
    validate_worker_ready(ready); ++checks;
    validate_worker_ready(with(ready, "unrelated_diagnostic", Json(true))); ++checks;
    for (const auto& [key, expected] : ready.object()) {
        Json::Object missing;
        for (const auto& item : ready.object()) if (item.first != key) missing.push_back(item);
        refusal([&] { validate_worker_ready(Json(missing)); });
        for (const auto& invalid : {Json(), Json(true), Json(false), Json(Json::Array{}), Json(Json::Object{})})
            refusal([&] { validate_worker_ready(with(ready, key, invalid)); });
        if (expected.kind() == Json::Kind::string) {
            for (const auto& invalid : {Json(""), Json("wrong"), Json::integer(1)})
                refusal([&] { validate_worker_ready(with(ready, key, invalid)); });
        } else {
            const auto n = expected.unsigned_integer();
            for (const auto& invalid : {Json::integer(0), Json::integer(n - 1), Json::integer(n + 1), Json(std::to_string(n)),
                parse_json(std::to_string(n) + ".0"), parse_json(std::to_string(n) + "e0")})
                refusal([&] { validate_worker_ready(with(ready, key, invalid)); });
        }
    }
    for (const auto* mode : {"indirect-compatible", "recurrent-replay", "capture"})
        refusal([&] { validate_worker_ready(with(ready, "stack_execution", Json(mode))); });
    for (const auto& invalid : {Json(), Json(true), Json("ready"), Json(Json::Array{}), Json(Json::Object{})})
        refusal([&] { validate_worker_ready(invalid); });
    // Refusals are pure: a later exact ready record remains admissible.
    validate_worker_ready(ready); ++checks;
    const auto replay = replay_metadata(with(ready, "stack_execution", Json("recurrent-replay")));
    validate_worker_ready(replay, "recurrent-replay"); ++checks;
    refusal([&] { validate_worker_ready(replay); });
    refusal([&] { validate_worker_ready(ready, "recurrent-replay"); });
    for (const auto* mode : {"", "capture", "indirect-compatible", "unknown"}) {
        refusal([&] { validate_worker_ready(replay, mode); });
        refusal([&] { validate_worker_ready(ready, mode); });
    }
    for (const auto* key : {"replay_prepared", "recurrent_replay_captures", "mapped_weight_region_limit_bytes"}) {
        refusal([&] { validate_worker_ready(without(replay, key), "recurrent-replay"); });
        for (const auto& invalid : {Json(), Json("true"), Json(Json::Array{}), Json(Json::Object{}), Json(false)})
            refusal([&] { validate_worker_ready(with(replay, key, invalid), "recurrent-replay"); });
        if (std::string_view(key) == "replay_prepared") {
            refusal([&] { validate_worker_ready(with(replay, key, Json::integer(1)), "recurrent-replay"); });
        } else {
            const auto n = replay.find(key)->unsigned_integer();
            for (const auto& invalid : {Json(true), Json::integer(0), Json::integer(n - 1), Json::integer(n + 1),
                Json(std::to_string(n)), parse_json(std::to_string(n) + ".0"), parse_json(std::to_string(n) + "e0")})
                refusal([&] { validate_worker_ready(with(replay, key, invalid), "recurrent-replay"); });
        }
    }
    validate_worker_ready(replay, "recurrent-replay"); ++checks;
}
std::vector<uint32_t> encode(std::string_view text) {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < text.size();) {
        if (text.substr(i).starts_with("<|im_end|>")) { result.push_back(1000); i += 10; }
        else if (text.substr(i).starts_with("<|endoftext|>")) { result.push_back(1001); i += 13; }
        else if (text.substr(i).starts_with("<|im_start|>")) { result.push_back(1002); i += 12; }
        else result.push_back(static_cast<unsigned char>(text[i++]) + 1U);
    }
    return result;
}
std::string decode(std::span<const uint32_t> ids) {
    std::string result;
    for (const auto id : ids) {
        if (id == 1000) result += "<|im_end|>";
        else if (id == 1001) result += "<|endoftext|>";
        else if (id == 1002) result += "<|im_start|>";
        else { check(id > 0 && id <= 256); result += char(id - 1); }
    }
    return result;
}
ServeCodec codec() {
    return {[](const Json& request) {
        auto text = std::string(request.find("messages")->array().back().find("content")->string());
        for (size_t at = 0; (at = text.find("[fixture_end]", at)) != std::string::npos;) text.replace(at, 13, "<|im_end|>");
        return RenderedChat{std::move(text), Json(Json::Array{}), 0, 0};
    }, [](const RenderedChat& rendered) { return encode(rendered.prompt); }, decode, {4096, 1000, 1001, 1002}};
}
Json request(std::string text = "A", bool stream = false, bool thinking = false) {
    for (size_t at = 0; (at = text.find("<|im_end|>", at)) != std::string::npos;) text.replace(at, 10, "[fixture_end]");
    return Json(Json::Object{{"model", Json("tokspur")}, {"messages", Json(Json::Array{Json(Json::Object{{"role", Json("user")}, {"content", Json(std::move(text))}})})},
        {"max_tokens", Json::integer(256)}, {"stream", Json(stream)}, {"enable_thinking", Json(thinking)}});
}
Json tools() {
    return parse_json(R"([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}])");
}
struct Fake {
    uint32_t position = 0;
    uint64_t seed = 0;
    Json sampling;
    std::vector<uint32_t> output;
    size_t next = 0, resets = 0, prefills = 0, generations = 0, maximum_generation = 0;
    std::string corrupt, stack_execution = "direct";
    void reply(std::string_view text) { output = encode(text); next = 0; }
    Json operator()(const Json& job, const std::function<void(const Json&)>& event) {
        if (job.find("reset")->boolean()) { position = 0; ++resets; sampling = *job.find("sampling"); seed = job.find("seed")->unsigned_integer(); }
        const auto budget = job.find("generate")->unsigned_integer();
        check(budget <= 4096); maximum_generation = std::max(maximum_generation, size_t(budget));
        const bool generation = budget != 0;
        check(!generation || budget == 1); // The ordinary serving scheduler remains packet-bounded.
        check(job.find("tokens")->array().size() == (generation ? 0U : 1U));
        const bool stopped = generation && (output.at(next) == 1000 || output.at(next) == 1001);
        Json::Array ids;
        ++position;
        if (generation) {
            ++generations; const auto id = output.at(next++); ids.push_back(Json::integer(id));
            event(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)}, {"committed_prefix", Json::integer(position)}}));
        } else ++prefills;
        auto policy = sampling.object(); policy.emplace_back("valid", Json(true)); policy.emplace_back("entropy_prefix", Json::integer(position)); policy.emplace_back("seed", Json::integer(seed));
        Json done(Json::Object{{"event", Json("done")}, {"committed_prefix", Json::integer(position)}, {"generated_ids", Json(std::move(ids))},
            {"status", Json(generation ? stopped ? "stopped" : "length" : "prefilled")}, {"finish_reason", generation ? Json(stopped ? "stop" : "length") : Json()},
            {"stop_id", stopped ? Json::integer(output[next - 1]) : Json()}, {"continuation_allowed", Json(!stopped)}, {"sampling", Json(std::move(policy))},
            {"small_gemv_column_splits", Json::integer(16)}, {"block_gemv_formats", Json::integer(3)}, {"stack_execution", Json(stack_execution)}});
        if (stack_execution == "recurrent-replay") done = replay_metadata(std::move(done));
        if (generation && corrupt == "prefix") done = with(std::move(done), "committed_prefix", Json::integer(position + 1));
        if (generation && corrupt == "policy") done = with(std::move(done), "block_gemv_formats", Json::integer(1));
        if (generation && corrupt == "execution") done = with(std::move(done), "stack_execution", Json(stack_execution == "direct" ? "recurrent-replay" : "direct"));
        if (generation && corrupt == "replay-prepared") done = with(std::move(done), "replay_prepared", Json(false));
        if (generation && corrupt == "replay-captures") done = with(std::move(done), "recurrent_replay_captures", Json::integer(23));
        if (generation && corrupt == "replay-region") done = with(std::move(done), "mapped_weight_region_limit_bytes", Json::integer(0));
        if (generation && corrupt == "replay-missing") done = without(done, "replay_prepared");
        if (stopped && corrupt == "terminal") done = with(std::move(done), "stop_id", Json::integer(1001));
        if (generation && corrupt == "throw") throw FormatError("fixture final failure after token");
        return done;
    }
};
Json broker_metadata(Json value) {
    value = with(std::move(value), "checkpoint_slots", Json::integer(2));
    value = with(std::move(value), "checkpoint_combined_budget_bytes", Json::integer(1 << 20));
    value = with(std::move(value), "checkpoint_host_bytes", Json::integer(512));
    value = with(std::move(value), "checkpoint_store_host_bytes", Json::integer(256));
    return with(std::move(value), "checkpoint_gpu_bytes", Json::integer(1024));
}
Json broker_ready() {
    return broker_metadata(parse_json(R"({"event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct","maximum_generated_tokens":4096,"checkpoint_broker":true,"broker_epoch":77})"));
}
struct BrokerFake : Fake {
    struct Saved { size_t prefix = 0, order = 0; bool valid = false; };
    std::array<Saved, 2> saved{};
    std::vector<uint32_t> history;
    uint64_t command = 0, lease = 0;
    size_t order = 0, begins = 0, captures = 0, hits = 0;
    bool last_restore = false, fail_suffix = false;
    std::string control_corrupt;
    Json stamp(Json value) const {
        value = broker_metadata(std::move(value));
        value = with(std::move(value), "checkpoint_retained", Json::integer(size_t(saved[0].valid) + size_t(saved[1].valid)));
        value = with(std::move(value), "broker_epoch", Json::integer(control_corrupt == "epoch" ? 78 : 77));
        value = with(std::move(value), "command_id", Json::integer(command));
        if (control_corrupt == "storage") value = with(std::move(value), "checkpoint_gpu_bytes", Json::integer(1025));
        return with(std::move(value), "lease_id", Json::integer(control_corrupt == "lease" ? lease - 1 : lease));
    }
    Json operator()(const Json& job, const std::function<void(const Json&)>& event) {
        ++command;
        if (const auto* op = job.find("op")) {
            check(!job.find("tokens") && !job.find("generate"));
            const bool begin = op->string() == "branch_begin"; std::string outcome = "captured"; size_t common = position;
            if (begin) {
                ++begins; ++lease; last_restore = job.find("allow_restore")->boolean();
                const auto& future = job.find("prompt_ids")->array(); common = 0;
                if (last_restore) while (common < future.size() && common < history.size() && history[common] == future[common].unsigned_integer()) ++common;
                std::optional<size_t> selected;
                for (size_t i = 0; last_restore && i < saved.size(); ++i) if (saved[i].valid && saved[i].prefix <= common &&
                    (!selected || saved[i].prefix > saved[*selected].prefix ||
                     (saved[i].prefix == saved[*selected].prefix && saved[i].order > saved[*selected].order))) selected = i;
                if (selected) {
                    position = uint32_t(saved[*selected].prefix); ++hits; outcome = "restored";
                    for (auto& record : saved) if (record.valid && record.prefix > position) record.valid = false;
                } else { position = 0; ++resets; for (auto& record : saved) record.valid = false; outcome = "cold_miss"; }
                history.resize(position); sampling = *job.find("sampling"); seed = job.find("seed")->unsigned_integer();
            } else {
                check(op->string() == "branch_capture" && position); ++captures; size_t at = 0;
                for (size_t i = 0; i < saved.size(); ++i) {
                    if (!saved[i].valid) { at = i; break; }
                    if (saved[i].order < saved[at].order) at = i;
                }
                saved[at] = {position, ++order, true};
            }
            auto policy = sampling.object(); policy.emplace_back("valid", Json(true)); policy.emplace_back("entropy_prefix", Json::integer(position)); policy.emplace_back("seed", Json::integer(seed));
            if (control_corrupt == "token") event(stamp(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(1)}, {"committed_prefix", Json::integer(position + 1)}})));
            return stamp(Json(Json::Object{{"event", Json("control_done")}, {"op", *op}, {"outcome", Json(outcome)}, {"model_steps", Json::integer(0)},
                {"generated_ids", Json(Json::Array{})}, {"committed_prefix", Json::integer(position)},
                {"common_prefix", Json::integer(common + (control_corrupt == "lcp"))}, {"sampling", Json(std::move(policy))},
                {"logits_available", Json(bool(position))}, {"stop_id", Json()}, {"continuation_allowed", Json(bool(position))}}));
        }
        if (job.find("reset")->boolean()) { ++lease; history.clear(); for (auto& record : saved) record.valid = false; }
        const bool generated = job.find("generate")->unsigned_integer() != 0;
        auto done = Fake::operator()(job, [&](const Json& token) { history.push_back(uint32_t(token.find("id")->unsigned_integer())); event(stamp(token)); });
        if (!generated) history.push_back(uint32_t(job.find("tokens")->array()[0].unsigned_integer()));
        if (fail_suffix && !generated) throw FormatError("fake suffix failure after committed model input");
        return stamp(std::move(done));
    }
};
const Json& message(const Json& result) { return *result.find("choices")->array()[0].find("message"); }
const Json& finish(const Json& result) { return *result.find("choices")->array()[0].find("finish_reason"); }
void original_agent_contract() {
    // The frozen production agent's initial Rust request shape/settings, with
    // synthetic messages only: no task transcript, persona or fixture sources.
    const auto wire = parse_json(R"({"model":"tokspur","messages":[{"role":"system","content":"contract fixture"},{"role":"user","content":"Read the project."}],"tools":[
      {"type":"function","function":{"name":"list_files","description":"List files in this isolated project.","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}},
      {"type":"function","function":{"name":"read_file","description":"Read a project file by relative path.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}},
      {"type":"function","function":{"name":"edit_file","description":"Replace exactly one occurrence in an allowed production source file. Tests/config must not change.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"],"additionalProperties":false}}},
      {"type":"function","function":{"name":"run_command","description":"Run the project's permitted verification command.","strict":true,"parameters":{"type":"object","properties":{"command":{"type":"string","enum":["cargo test --offline"]}},"required":["command"],"additionalProperties":false}}}
    ],"stream":false,"max_tokens":32768,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"repetition_penalty":1,"seed":44,"reasoning_effort":"xhigh","chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}})");
    const auto admitted = admit_chat(wire, "tokspur");
    check(admitted.maximum_tokens == 32768 && admitted.thinking && admitted.seed == 44);
    check(admitted.render.find("enable_thinking")->boolean() && admitted.render.find("preserve_thinking")->boolean());
    check(admitted.render.find("reasoning_effort")->string() == "xhigh");
    check(admitted.render.find("messages")->dump() == wire.find("messages")->dump());
    check(admitted.render.find("tools")->dump() == wire.find("tools")->dump() && admitted.manifest.size() == 4);
    check(admitted.sampling.dump() == R"({"temperature":1,"top_k":20,"top_p":0.95,"min_p":0})");
    auto legacy = without(wire, "chat_template_kwargs");
    legacy = with(with(legacy, "enable_thinking", Json(true)), "preserve_thinking", Json(true));
    check(admit_chat(legacy, "tokspur").render.dump() == admitted.render.dump());
    const auto defaults = admit_chat(without(without(request(), "max_tokens"), "enable_thinking"), "tokspur");
    check(defaults.maximum_tokens == 8192 && defaults.thinking && !defaults.render.find("preserve_thinking") && !defaults.render.find("reasoning_effort"));
    check(admit_chat(with(without(wire, "max_tokens"), "max_completion_tokens", Json::integer(32768)), "tokspur").maximum_tokens == 32768);
    check(admit_chat(with(wire, "max_tokens", Json::integer(4097)), "tokspur").maximum_tokens == 4097);
    for (const auto name : {"presence_penalty", "repetition_penalty"}) {
        const bool presence = std::string_view(name) == "presence_penalty";
        for (const auto value : {presence ? "-0.0" : "1.0", presence ? "0e10" : "1e0"})
            check(admit_chat(with(wire, name, parse_json(value)), "tokspur").sampling.dump() == admitted.sampling.dump());
        for (const auto value : {"0.5", presence ? "0.00000000000000000001" : "1.00000000000000000001", "null", "true", "\"0\""})
            refusal([&] { admit_chat(with(wire, name, parse_json(value)), "tokspur"); });
    }
    for (const auto value : {"0", "32769", "32768.0", "\"32768\"", "-1"})
        refusal([&] { admit_chat(with(wire, "max_tokens", parse_json(value)), "tokspur"); });
    for (const auto& value : {Json(), Json(true), Json("options"), Json(Json::Array{})})
        refusal([&] { admit_chat(with(wire, "chat_template_kwargs", value), "tokspur"); });
    for (const auto name : {"enable_thinking", "preserve_thinking"}) {
        refusal([&] { admit_chat(with(wire, name, Json(false)), "tokspur"); });
        auto bad = with(*wire.find("chat_template_kwargs"), name, Json("true"));
        refusal([&] { admit_chat(with(wire, "chat_template_kwargs", bad), "tokspur"); });
    }
    refusal([&] { admit_chat(with(wire, "reasoning_effort", Json("low")), "tokspur"); });
    refusal([&] { admit_chat(with(wire, "chat_template_kwargs", with(*wire.find("chat_template_kwargs"), "thinking_mode", Json("enabled"))), "tokspur"); });
    refusal([&] { admit_chat(with(without(wire, "reasoning_effort"), "chat_template_kwargs", with(*wire.find("chat_template_kwargs"), "reasoning_effort", Json("unknown"))), "tokspur"); });
    refusal([&] { admit_chat(with(wire, "frequency_penalty", Json::integer(0)), "tokspur"); });
    auto disabled = with(request(), "reasoning_effort", Json("none"));
    check(!admit_chat(disabled, "tokspur").thinking);
    refusal([&] { admit_chat(with(disabled, "enable_thinking", Json(true)), "tokspur"); });

    ServeSession session(codec(), "tokspur"); Fake fake;
    fake.reply("Read it.</think>\n\n<tool_call><function=read_file><parameter=path>\nsrc/lib.rs\n</parameter></function></tool_call><|im_end|>");
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(999); }};
    const auto result = session.complete(wire, "frozen-wire", [&](const Json& job, const auto& event) { return fake(job, event); }, sink);
    check(finish(result).string() == "tool_calls" && fake.seed == 44 && fake.maximum_generation == 1);
    check(message(result).find("reasoning_content")->string() == "Read it.");
    const auto& call = message(result).find("tool_calls")->array()[0];
    check(call.find("function")->find("name")->string() == "read_file" && call.find("function")->find("arguments")->string() == "{\"path\":\"src/lib.rs\"}");
}
Json refused_prefix_trace(ServeSession& session, const Json& input) {
    Json observed;
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }, [&](const Json& value) {
        observed = value;
        throw FormatError("controlled trace refusal");
    }};
    refusal([&] { session.complete(input, "prefix-test", [](const Json&, const auto&) -> Json {
        throw std::logic_error("prefix trace refusal submitted worker work");
    }, sink); });
    check(observed.find("event") && observed.find("event")->string() == "prompt_prefix");
    return observed;
}
void respelled_prefix_reuse() {
    // Token 300 spells "Vi" in one token. Re-encoding the echoed text spells it as two.
    auto spelled = codec();
    spelled.decode = [](std::span<const uint32_t> ids) {
        std::string result;
        for (const auto& id : ids) result += id == 300 ? std::string("Vi") : decode(std::span<const uint32_t>(&id, 1));
        return result;
    };
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
    const auto sampled = [] { auto ids = encode("sible<|im_end|>"); ids.insert(ids.begin(), 300); return ids; };
    {
        ServeSession session(spelled); Fake fake;
        const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };
        fake.output = sampled();
        const auto first = session.complete(request(), "spelled", target, sink);
        check(message(first).find("content")->string() == "Visible" && session.committed_tokens() == 8 && fake.resets == 1);
        fake.reply("Next<|im_end|>");
        const auto same = session.complete(request("AVisible<|im_end|>B"), "same-text", target, sink);
        check(finish(same).string() == "stop" && fake.resets == 1 && fake.prefills == 2);
        check(same.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == 8);
    }
    {
        ServeSession session(spelled); Fake fake;
        const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };
        fake.output = sampled();
        static_cast<void>(session.complete(request(), "spelled", target, sink));
        fake.reply("Next<|im_end|>");
        const auto edited = session.complete(request("AVisibly<|im_end|>B"), "edited-text", target, sink);
        check(finish(edited).string() == "stop" && fake.resets == 2);
    }
    for (const bool change_seed : {true, false}) {
        ServeSession session(spelled); Fake fake;
        const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };
        fake.output = sampled();
        const auto initial = with(with(request(), "seed", Json::integer(42)), "temperature", parse_json("0.7"));
        static_cast<void>(session.complete(initial, "spelled", target, sink));
        auto next = with(with(request("AVisible<|im_end|>B"), "seed", Json::integer(change_seed ? 99 : 42)), "temperature", parse_json(change_seed ? "0.7" : "0.5"));
        const auto trace = refused_prefix_trace(session, next);
        check(!trace.find("append")->boolean());
        const auto expected = encode("AVisible<|im_end|>B");
        const auto& actual = trace.find("prompt_ids")->array();
        check(actual.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) check(actual[i].unsigned_integer() == expected[i]);
    }
}
void respelled_prefix_preserves_nonordinary_tokens() {
    for (const auto& [marker, spelling] : std::vector<std::pair<uint32_t, std::string>>{
        {301, "<think>"}, {302, "</think>"}, {303, "<tool_call>"}, {304, "</tool_call>"},
        {305, "<tool_response>"}, {306, "</tool_response>"}, {307, "<added>"}, {308, "[PAD308]"}}) {
        for (const bool original_special : {false, true}) for (const bool echoed_special : {false, true}) {
            bool first_encode = true;
            auto spelled = codec();
            spelled.nonordinary_ids = {marker, marker};
            spelled.encode = [&](const RenderedChat&) {
                const bool first = first_encode;
                first_encode = false;
                auto ids = first ? std::vector<uint32_t>{300} : encode("Vi");
                if (first ? original_special : echoed_special) ids.push_back(marker);
                else { const auto bytes = encode(spelling); ids.insert(ids.end(), bytes.begin(), bytes.end()); }
                ids.push_back(uint32_t('x') + 1);
                if (!first) { const auto tail = encode("ok<|im_end|>B"); ids.insert(ids.end(), tail.begin(), tail.end()); }
                return ids;
            };
            spelled.decode = [&](std::span<const uint32_t> ids) {
                std::string result;
                for (const auto& id : ids) {
                    if (id == 300) result += "Vi";
                    else if (id == marker) result += spelling;
                    else result += decode(std::span<const uint32_t>(&id, 1));
                }
                return result;
            };
            ServeSession session(spelled); Fake fake; fake.reply("ok<|im_end|>");
            const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
            static_cast<void>(session.complete(request(), "nonordinary", [&](const Json& job, const auto& event) { return fake(job, event); }, sink));
            const auto trace = refused_prefix_trace(session, request("B"));
            check(trace.find("append")->boolean() == (original_special == echoed_special));
        }
    }
    auto invalid = codec(); invalid.nonordinary_ids = {invalid.admission.vocabulary};
    refusal([&] { ServeSession rejected(invalid); });
}
void respelled_prefix_decodes_only_changed_tokens() {
    bool first_encode = true;
    size_t decoded = 0;
    const std::vector<uint32_t> equal(20000, uint32_t('x') + 1);
    auto spelled = codec();
    spelled.encode = [&](const RenderedChat&) {
        const bool first = first_encode;
        first_encode = false;
        auto ids = first ? std::vector<uint32_t>{300} : encode("Vi");
        ids.push_back(1000);
        ids.insert(ids.end(), equal.begin(), equal.end()); ids.push_back(1000);
        ids.insert(ids.end(), equal.begin(), equal.end());
        if (first) ids.push_back(300);
        else { ids.push_back(uint32_t('V') + 1); ids.push_back(uint32_t('i') + 1); }
        ids.insert(ids.end(), equal.begin(), equal.end());
        if (!first) { const auto tail = encode("ok<|im_end|>B"); ids.insert(ids.end(), tail.begin(), tail.end()); }
        return ids;
    };
    spelled.decode = [&](std::span<const uint32_t> ids) {
        decoded += ids.size();
        std::string result;
        for (const auto& id : ids) result += id == 300 ? std::string("Vi") : decode(std::span<const uint32_t>(&id, 1));
        return result;
    };
    ServeSession session(spelled); Fake fake; fake.reply("ok<|im_end|>");
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
    static_cast<void>(session.complete(request(), "long-prefix", [&](const Json& job, const auto& event) { return fake(job, event); }, sink));
    decoded = 0;
    const auto trace = refused_prefix_trace(session, request("B"));
    check(trace.find("append")->boolean());
    check(decoded == 6);
}
void extended_output_budget() {
    ServeSession session(codec()); Fake fake;
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
    const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };
    const std::string text(32768, 'x'); fake.reply(text);
    const auto result = session.complete(with(request(), "max_tokens", Json::integer(32768)), "long", target, sink);
    check(finish(result).string() == "length" && message(result).find("content")->string() == text);
    check(result.find("usage")->find("completion_tokens")->unsigned_integer() == 32768);
    check(fake.generations == 32768 && fake.maximum_generation == 1 && fake.resets == 1 && session.committed_tokens() == 32769);
    fake.reply("Next<|im_end|>");
    const auto appended = session.complete(with(request("A" + text + "B"), "max_tokens", Json::integer(8)), "append", target, sink);
    check(finish(appended).string() == "stop" && fake.resets == 1 && fake.prefills == 2);
    check(appended.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == 32769);
    // A requested upper bound is clamped to the remaining native context.
    // The original agent can keep asking for32K turns above a32K prompt.
    ServeSession over_half(codec()); Fake over_half_fake; over_half_fake.reply("z<|im_end|>");
    const auto over_half_result = over_half.complete(with(request(std::string(32769,'A')), "max_tokens", Json::integer(32768)), "over-half-context",
        [&](const Json& job, const auto& event) { return over_half_fake(job,event); }, sink);
    check(finish(over_half_result).string()=="stop" && over_half_fake.prefills==32769 && over_half_fake.generations==2);
    ServeSession boundary(codec()); Fake boundary_fake; boundary_fake.reply("<|im_end|>");
    const auto exact = boundary.complete(with(request(std::string(32768, 'A')), "max_tokens", Json::integer(32768)), "exact-context",
        [&](const Json& job, const auto& event) { return boundary_fake(job, event); }, sink);
    check(finish(exact).string() == "stop" && boundary_fake.prefills == 32768 && boundary_fake.generations == 1);
    for (uint32_t remaining : {1u,2u}) {
        ServeSession full(codec()); Fake full_fake; full_fake.reply("xyz");
        const auto limited=full.complete(with(request(std::string(65536-remaining,'A')), "max_tokens", Json::integer(32768)), "context-tail",
            [&](const Json& job,const auto& event) { return full_fake(job,event); },sink);
        check(finish(limited).string()=="length" && full_fake.generations==remaining && full.committed_tokens()==65536);
        check(limited.find("usage")->find("completion_tokens")->unsigned_integer()==remaining && limited.find("usage")->find("total_tokens")->unsigned_integer()==65536);
        check(message(limited).find("content")->string()==std::string_view("xyz").substr(0,remaining));
    }
    const auto before=fake.prefills+fake.generations;
    refusal([&] { session.complete(with(request(std::string(65536,'A')), "max_tokens", Json::integer(32768)), "no-context",target,sink); });
    check(before==fake.prefills+fake.generations);
}
void parked_lane_http_fast_math_compatibility() {
    ServeSession session(codec(), "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , false
#endif
        , ToolEnumValidation::strict
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        , false
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , Json()
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        , Json()
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , std::nullopt
#endif
        , ToolArgValidation::strict, 65536, true);
    Fake fake;
    fake.reply("ok<|im_end|>");
    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
    const auto res = session.complete(request("A"), "fast-math-compat", [&](const Json& job, const auto& cb) { return fake(job, cb); }, sink);
    check(finish(res).string() == "stop");
}
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
Json parked_mtp_ready() {
    return parse_json(R"({
      "event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct",
      "maximum_generated_tokens":4096,"known_token_capacity":8,"known_residual_reuse":true,
      "prompt_lookup":false,"verification_max_tokens":2,"verification_head_many":false,"verification8_branch_reuse":false,
      "sampling":{"valid":true,"seed":0,"entropy_prefix":0,"entropy_words":0,"filter_order":"fixture","rng":"fixture","ownership":"fixture","temperature":1,"top_k":20,"top_p":0.95,"min_p":0},
      "hc_q8_splits":0,"hc_q8_numerical_tier":"ordinary","hc_q8_scratch_logical_bytes":0,"hc_q8_scratch_charged_bytes":0,
      "replay_prepared":false,"recurrent_replay_captures":0,"mapped_weight_region_limit_bytes":0,
      "capture_resource_bytes":0,"capture_binding_aliases":0,"capture_binding_alias_span_bytes":0,
      "capture_binding_alias_resource_bytes":0,"live_captures":0,"profile_counter_payload_bytes":0,
      "checkpoint_slots":0,"checkpoint_combined_budget_bytes":0,"checkpoint_host_bytes":0,
      "checkpoint_store_host_bytes":0,"checkpoint_gpu_bytes":0,"checkpoint_broker":false,"broker_epoch":0,
      "native_limit_bytes":103079215104,"weight_copy":false,"gpu_profile_mode":"disabled","timings_qualified":false,
      "allow_prompt_lookup_experiments":false,"allow_request_kernel_experiments":false,
      "allow_request_stack_experiments":false,"allow_request_prefill_experiments":false,
      "lookup_match_tokens":0,"lookup_history_tokens":0,"lookup_proposal_tokens":0,
      "lookup_verification_budget_bytes":0,"lookup_caller_host_bytes":0,
      "verification_host_bytes":1000000,"verification_gpu_bytes":4000000,
      "lookup_verification_charged_bytes":0,"verification_tape_payload_bytes":1024,
      "tracked_metal_buffer_bytes":84214816768,"peak_metal_buffer_bytes":84214816768,
      "gpu_buffer_limit_bytes":100394860544,"imported_weight_backing_bytes":81848123392,"gpu_allocation_count":613,
      "hc_exact_mode":"combined","hc_exact_rows_per_group":1,"hc_exact_unroll":16,
      "hc_exact_scratch_logical_bytes":0,"hc_exact_scratch_charged_bytes":0,
      "hc_exact_routes":{"scalar":0,"many1":0,"many2":0,"many4":0,"many8":0},
      "wide_q8_exact_enabled":false,"wide_q8_exact_rows_per_group":4,"wide_q8_exact_unroll":16,
      "wide_q8_exact_scratch_logical_bytes":0,"wide_q8_exact_scratch_charged_bytes":0,
      "wide_q8_exact_routes":{"qkv_scalar":0,"qkv_many1":0,"output_scalar":0,"output_many1":0},
      "wide_q8_many_enabled":false,"wide_q8_many_rows_per_group":4,"wide_q8_many_unroll":4,
      "wide_q8_many_scratch_logical_bytes":0,"wide_q8_many_scratch_charged_bytes":0,
      "wide_q8_many_routes":{"qkv_many2":0,"qkv_many4":0,"qkv_many8":0,"output_many2":0,"output_many4":0,"output_many8":0},
      "qsa_stage_v8_prepared":true,"qsa_stage_v8_enabled":true,"qsa_stage_v8_scratch_logical_bytes":196896,
      "qsa_stage_v8_scratch_charged_bytes":212992,"qsa_stage_v8_calls":0,"qsa_stage_v8_physical_dispatches":0,"qsa_stage_v8_extra_dispatches":0,
      "ple_gate_transform":"signed-root","ple_gate_transform_schema":1,
      "qsa_channel_chronological_mode":"on","qsa_channel_chronological_schema":1,
      "qsa_channel_factor_logical_bytes":393792,"qsa_channel_factor_charged_bytes":409600,
      "qsa_channel_calls":0,"qsa_channel_extra_dispatches":0,
      "known_qsa_rows_schema":1,"known_qsa_rows_mode":"q8","known_qsa_rows_logical_bytes":8000000,
      "known_qsa_rows_charged_bytes":8003584,"known_qsa_rows_allocations":46,"known_qsa_rows_physical_tiles":0,
      "known_qsa_rows_logical_rows":0,"known_qsa_rows_http_admitted":false,
      "native_mtp_protocol":"native-mtp-depth1-lookahead-v2","native_mtp_depth":1,"native_mtp_attention":"dense-compression0",
      "native_mtp_first_cache_position":1,"native_mtp_lookahead_anchor":true,
      "native_mtp_visible_frontier":"all emitted tokens physically consumed; sampled terminal carried invisibly within request",
      "native_mtp_workspace_bytes":1000000,
      "native_mtp_head_calls":0,"native_mtp_prime_calls":0,"native_mtp_rounds":0,"native_mtp_proposed":0,
      "native_mtp_accepted":0,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0,
      "native_mtp_head_gpu_ms":0,"native_mtp_head_wall_ms":0,"native_mtp_prime_gpu_ms":0,"native_mtp_prime_wall_ms":0
    })");
}
void parked_lane_prefix_respelling_and_hoisting() {
    size_t render_calls = 0, encode_calls = 0;
    auto parked_codec = codec();
    parked_codec.nonordinary_ids = {301};
    parked_codec.render = [&](const Json& req) {
        ++render_calls;
        std::string prompt;
        for (const auto& msg : req.find("messages")->array()) {
            const auto role = msg.find("role")->string();
            const auto content = msg.find("content")->string();
            if (role == "assistant") {
                if (const auto* r = msg.find("reasoning_content"); r && !r->is_null()) {
                    prompt += std::string(r->string()) + "</think>";
                }
                prompt += std::string(content) + "<|im_end|>";
            } else {
                prompt += std::string(content);
            }
        }
        return RenderedChat{std::move(prompt), Json(Json::Array{}), 0, 0};
    };
    parked_codec.encode = [&](const RenderedChat& r) {
        ++encode_calls;
        std::vector<uint32_t> out;
        const std::string& s = r.prompt;
        for (size_t i = 0; i < s.size();) {
            if (s.substr(i).starts_with("<|im_end|>")) { out.push_back(1000); i += 10; }
            else if (s.substr(i).starts_with("</think>")) { out.push_back(301); i += 8; }
            else out.push_back(static_cast<unsigned char>(s[i++]) + 1U);
        }
        return out;
    };
    parked_codec.decode = [&](std::span<const uint32_t> ids) {
        std::string out;
        for (const auto id : ids) {
            if (id == 300) out += "Vi";
            else if (id == 301) out += "</think>";
            else out += decode(std::span<const uint32_t>(&id, 1));
        }
        return out;
    };

    const auto make_parked_session = [&] {
        return ServeSession(parked_codec, "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
            , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
            , false
#endif
            , ToolEnumValidation::strict
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
            , false
#endif
            , parked_mtp_ready()
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
            , Json()
#endif
            , std::nullopt, ToolArgValidation::strict, 65536, true);
    };

    const auto mtp_ready = parked_mtp_ready();
    std::vector<uint32_t> out_tokens;
    uint32_t mtp_pos = 0;
    uint64_t mtp_primes = 0, mtp_tiles = 0, mtp_rows = 0, mtp_stage = 0;
    uint64_t mtp_seed = 42;
    Json mtp_sampling;
    const auto mtp_target = [&](const Json& job, const std::function<void(const Json&)>& event) -> Json {
        const bool reset = job.find("reset")->boolean();
        if (reset) { mtp_pos = 0; mtp_seed = job.find("seed")->unsigned_integer(); mtp_sampling = *job.find("sampling"); }
        const auto n = uint32_t(job.find("tokens")->array().size());
        const auto maximum = uint32_t(job.find("generate")->unsigned_integer());
        Json::Array gen_ids;
        uint32_t ordinary = 0;
        bool stopped = false;
        if (n > 0) {
            mtp_pos += n;
            mtp_primes += n - (reset ? 1u : 0u);
            mtp_tiles += 12;
            mtp_rows += 12 * n;
        } else {
            while (gen_ids.size() < maximum && !out_tokens.empty() && !stopped) {
                const auto id = out_tokens.front(); out_tokens.erase(out_tokens.begin());
                ++mtp_pos; ++ordinary; ++mtp_primes;
                mtp_stage += 12;
                gen_ids.push_back(Json::integer(id));
                event(Json(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)}, {"source_logit", parse_json("0.5")},
                    {"committed_prefix", Json::integer(mtp_pos)}, {"entropy_words", Json::integer(0)}, {"gpu_profile", Json()}}));
                stopped = (id == 1000);
            }
        }
        auto done = with(mtp_ready, "event", Json("done"));
        for (const auto name : {"lookup_verification_budget_bytes", "lookup_caller_host_bytes", "verification_host_bytes",
            "verification_gpu_bytes", "lookup_verification_charged_bytes", "verification_tape_payload_bytes",
            "prompt_lookup", "checkpoint_broker", "broker_epoch", "gpu_profile_mode",
            "allow_prompt_lookup_experiments", "allow_request_kernel_experiments",
            "allow_request_stack_experiments", "allow_request_prefill_experiments"}) {
            done = without(std::move(done), name);
        }
        done = with(std::move(done), "committed_prefix", Json::integer(mtp_pos));
        done = with(std::move(done), "generated_ids", Json(std::move(gen_ids)));
        done = with(std::move(done), "known_token_prefill", Json(true));
        done = with(std::move(done), "status", Json(n ? "prefilled" : stopped ? "stopped" : "length"));
        done = with(std::move(done), "finish_reason", n ? Json() : Json(stopped ? "stop" : "length"));
        done = with(std::move(done), "stop_id", stopped ? Json::integer(1000) : Json());
        done = with(std::move(done), "continuation_allowed", Json(!stopped));
        auto samp = mtp_sampling.object();
        for (const auto& item : Json::Object{{"valid", Json(true)}, {"seed", Json::integer(mtp_seed)},
            {"entropy_prefix", Json::integer(mtp_pos)}, {"entropy_words", Json::integer(0)},
            {"filter_order", Json("fixture")}, {"rng", Json("fixture")}, {"ownership", Json("fixture")}}) {
            samp.push_back(item);
        }
        done = with(std::move(done), "sampling", Json(std::move(samp)));
        done = with(std::move(done), "native_mtp_prime_calls", Json::integer(mtp_primes));
        done = with(std::move(done), "known_qsa_rows_physical_tiles", Json::integer(mtp_tiles));
        done = with(std::move(done), "known_qsa_rows_logical_rows", Json::integer(mtp_rows));
        done = with(std::move(done), "qsa_stage_v8_calls", Json::integer(mtp_stage));
        done = with(std::move(done), "qsa_stage_v8_physical_dispatches", Json::integer(2 * mtp_stage));
        done = with(std::move(done), "qsa_stage_v8_extra_dispatches", Json::integer(mtp_stage));
        done = with(std::move(done), "qsa_channel_calls", Json::integer(mtp_stage));
        done = with(std::move(done), "qsa_channel_extra_dispatches", Json::integer(mtp_stage));
        done = with(std::move(done), "qsa_stage_v8_request_calls", Json::integer(12 * ordinary));
        done = with(std::move(done), "qsa_channel_request_calls", Json::integer(12 * ordinary));
        done = with(std::move(done), "hc_exact_request_routes", parse_json(R"({"scalar":0,"many1":0,"many2":0,"many4":0,"many8":0})"));
        return done;
    };

    const ServeSink sink{{}, [] { return false; }, [] { return uint64_t(42); }};
    const auto trace_contains = [](const Json& trace, uint32_t id) {
        const auto& arr = trace.find("prompt_ids")->array();
        return std::any_of(arr.begin(), arr.end(), [id](const Json& v) { return v.unsigned_integer() == id; });
    };

    // 1. enable_thinking=false parks lane A (seed=11) & lane A2 (seed=22) with non-canonical token 300 ("Vi").
    //    Switch to B (seed=99), verify ambiguous_seed miss when seed omitted, explicit seed=11 hit, and temp mismatch miss.
    {
        auto session = make_parked_session();
        for (uint64_t s : {11ull, 22ull}) {
            out_tokens = {300};
            for (char c : std::string_view("sible")) out_tokens.push_back(static_cast<unsigned char>(c) + 1U);
            out_tokens.push_back(1000);
            const auto r = with(with(request("A", false, false), "seed", Json::integer(s)), "temperature", parse_json("0.7"));
            static_cast<void>(session.complete(r, "park-a", mtp_target, sink));
        }
        // Park second lane by switching to B (seed=99).
        out_tokens = {uint32_t('B') + 1U, 1000};
        static_cast<void>(session.complete(with(with(request("Z", false, false), "seed", Json::integer(99)), "temperature", parse_json("0.7")), "park-b", mtp_target, sink));

        const auto msgs_a = parse_json(R"([{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}])");
        const auto ret_no_seed = with(with(request(), "temperature", parse_json("0.7")), "messages", msgs_a);
        const auto trace_ambig = refused_prefix_trace(session, ret_no_seed);
        check(!trace_contains(trace_ambig, 300)); // ambiguous_seed across seeds 11 vs 22!

        const auto ret_seed_11 = with(ret_no_seed, "seed", Json::integer(11));
        render_calls = encode_calls = 0;
        const auto trace_11 = refused_prefix_trace(session, ret_seed_11);
        check(trace_contains(trace_11, 300));
        check(render_calls == 1 && encode_calls == 1); // empty turns hoisted to base_prompt!

        const auto ret_bad_temp = with(with(ret_no_seed, "seed", Json::integer(11)), "temperature", parse_json("0.5"));
        const auto trace_bad_temp = refused_prefix_trace(session, ret_bad_temp);
        check(!trace_contains(trace_bad_temp, 300));
    }

    // 2. Echoed reasoning_content falls back to base_prompt and respells; tampered reasoning misses;
    //    omitted reasoning with identical turns deduplicates render/encode.
    {
        mtp_primes = mtp_tiles = mtp_rows = mtp_stage = 0;
        auto session = make_parked_session();
        out_tokens.clear();
        for (char c : std::string_view("why</think>")) out_tokens.push_back(static_cast<unsigned char>(c) + 1U);
        // Replace "</think>" bytes with token 301 and "Vi" with 300.
        out_tokens = {uint32_t('w') + 1U, uint32_t('h') + 1U, uint32_t('y') + 1U, 301, 300};
        for (char c : std::string_view("sible")) out_tokens.push_back(static_cast<unsigned char>(c) + 1U);
        out_tokens.push_back(1000);
        static_cast<void>(session.complete(with(with(request("A", false, true), "seed", Json::integer(42)), "temperature", parse_json("0.7")), "think-a", mtp_target, sink));

        // Park think-a by switching to B.
        out_tokens = {uint32_t('B') + 1U, 1000};
        static_cast<void>(session.complete(with(with(request("Z", false, false), "seed", Json::integer(99)), "temperature", parse_json("0.7")), "think-b", mtp_target, sink));

        const auto echoed = with(with(with(request(), "seed", Json::integer(42)), "temperature", parse_json("0.7")), "messages",
            parse_json(R"([{"role":"user","content":"A"},{"role":"assistant","reasoning_content":"why","content":"Visible"},{"role":"user","content":"B"}])"));
        render_calls = encode_calls = 0;
        const auto trace_echoed = refused_prefix_trace(session, echoed);
        check(trace_contains(trace_echoed, 300) && render_calls == 1 && encode_calls == 1);

        const auto tampered = with(echoed, "messages",
            parse_json(R"([{"role":"user","content":"A"},{"role":"assistant","reasoning_content":"wrong","content":"Visible"},{"role":"user","content":"B"}])"));
        check(!trace_contains(refused_prefix_trace(session, tampered), 300));

        const auto omitted = with(echoed, "messages",
            parse_json(R"([{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}])"));
        render_calls = encode_calls = 0;
        const auto trace_omitted = refused_prefix_trace(session, omitted);
        check(trace_contains(trace_omitted, 300) && render_calls == 2 && encode_calls == 2);
    }

    // 3. Capacity > 16 evicts oldest parked lane cleanly.
    {
        mtp_primes = mtp_tiles = mtp_rows = mtp_stage = 0;
        auto session = make_parked_session();
        for (uint64_t s = 1; s <= 18; ++s) {
            out_tokens = {300};
            for (char c : std::string_view("sible")) out_tokens.push_back(static_cast<unsigned char>(c) + 1U);
            out_tokens.push_back(1000);
            const auto r = with(with(request("L" + std::to_string(s), false, false), "seed", Json::integer(s)), "temperature", parse_json("0.7"));
            static_cast<void>(session.complete(r, "cap-" + std::to_string(s), mtp_target, sink));
        }
        // Lane 1 (L1, seed 1) was evicted (17 parked + 1 active = 18 total -> first parked evicted); Lane 17 (L17, seed 17) remains parked.
        const auto probe_1 = with(with(with(request(), "seed", Json::integer(1)), "temperature", parse_json("0.7")), "messages",
            parse_json(R"([{"role":"user","content":"L1"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}])"));
        check(!trace_contains(refused_prefix_trace(session, probe_1), 300));
        const auto probe_17 = with(with(with(request(), "seed", Json::integer(17)), "temperature", parse_json("0.7")), "messages",
            parse_json(R"([{"role":"user","content":"L17"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}])"));
        check(trace_contains(refused_prefix_trace(session, probe_17), 300));
    }
}
#endif
void startup_execution_policy() {
    for (const auto* invalid : {"", "capture", "indirect-compatible", "unknown"})
        refusal([&] { ServeSession rejected(codec(), "tokspur", invalid); });
    for (const auto* mode : {"direct", "recurrent-replay"}) {
        ServeSession session(codec(), "tokspur", mode); Fake fake; fake.stack_execution = mode;
        std::vector<Json> chunks;
        const ServeSink sink{[&](const Json& part) { chunks.push_back(part); }, [] { return false; }, [] { return uint64_t(42); }};
        const auto call = [&](const Json& job, const auto& event) { return fake(job, event); };
        fake.reply("valid<|im_end|>");
        check(finish(session.complete(request("A", true), "startup-policy", call, sink)).string() == "stop");
        const auto submitted = fake.prefills + fake.generations;
        for (const auto* body_key : {"stack_execution", "stack-execution"})
            refusal([&] { session.complete(with(request(), body_key, Json(mode)), "body-switch", call, sink); });
        check(submitted == fake.prefills + fake.generations);
        const auto tool_request = with(request("B", true), "tools", tools());
        for (const auto* corrupt : {"execution", "replay-prepared", "replay-captures", "replay-region", "replay-missing"}) {
            if (std::string_view(mode) == "direct" && std::string_view(corrupt) != "execution") continue;
            chunks.clear(); fake.corrupt = corrupt;
            fake.reply("<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>");
            refusal([&] { session.complete(tool_request, "execution-mismatch", call, sink); });
            check(session.committed_tokens() == 0);
            for (const auto& part : chunks) check(!part.find("choices")->array()[0].find("delta")->find("tool_calls"));
            fake.corrupt.clear(); fake.reply("recovered<|im_end|>");
            check(finish(session.complete(request(), "after-policy-refusal", call, sink)).string() == "stop");
        }
        session.invalidate(); fake.stack_execution = std::string_view(mode) == "direct" ? "recurrent-replay" : "direct";
        fake.reply("never accepted<|im_end|>"); const auto generations = fake.generations;
        refusal([&] { session.complete(request(), "startup-mode-mismatch", call, sink); });
        check(session.committed_tokens() == 0 && fake.generations == generations);
    }
}
void broker_sessions() {
    const auto ready = broker_ready(); validate_worker_ready(ready, "direct", 2, 1 << 20); ++checks;
    refusal([&] { validate_worker_ready(ready); });
    for (const auto* key : {"checkpoint_broker", "broker_epoch", "checkpoint_host_bytes", "checkpoint_gpu_bytes"})
        refusal([&] { validate_worker_ready(without(ready, key), "direct", 2, 1 << 20); });
    ServeSession session(codec(), "tokspur", "direct", ready); BrokerFake fake; bool cancel = false;
    const ServeSink sink{{}, [&] { return cancel; }, [] { return uint64_t(42); }};
    const auto call = [&](const Json& job, const auto& event) { return fake(job, event); };
    fake.reply("X<|im_end|>"); session.complete(request("AB"), "broker-first", call, sink);
    check(fake.begins == 1 && fake.resets == 1 && fake.captures == 1 && fake.prefills == 2);
    fake.reply("Y<|im_end|>"); session.complete(request("ABX<|im_end|>C"), "broker-append", call, sink);
    check(fake.begins == 1 && fake.resets == 1 && fake.captures == 2);
    const auto before = fake.prefills; fake.reply("Z<|im_end|>");
    const auto branch = session.complete(request("ABD"), "broker-branch", call, sink);
    check(fake.hits == 1 && fake.prefills == before + 1 &&
        branch.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == 2);
    const auto captures = fake.captures; fake.reply("Z<|im_end|>"); session.complete(request("ABD"), "broker-exact", call, sink);
    check(fake.captures == captures && fake.hits == 2);
    fake.fail_suffix = true; fake.reply("unused<|im_end|>");
    refusal([&] { session.complete(request("ABE"), "broker-suffix-failed", call, sink); });
    check(!session.committed_tokens()); fake.fail_suffix = false;
    fake.reply("safe<|im_end|>"); session.complete(request("AB"), "broker-after-suffix", call, sink); check(!fake.last_restore);
    cancel = true; const auto controls = fake.begins;
    refusal([&] { session.complete(request("Q"), "broker-before-cancel", call, sink); }); check(fake.begins == controls); cancel = false;
    const auto after_control = [&](const Json& job, const auto& event) {
        auto done = fake(job, event); if (job.find("op") && job.find("op")->string() == "branch_begin") cancel = true; return done;
    };
    refusal([&] { session.complete(request("Q"), "broker-after-control-cancel", after_control, sink); });
    check(!session.committed_tokens()); cancel = false;
    fake.reply("safe<|im_end|>"); session.complete(request("Q"), "broker-after-cancel", call, sink); check(!fake.last_restore);
    const auto greedy_begins = fake.begins, ordinary_resets = fake.resets; fake.reply("safe<|im_end|>");
    auto stochastic = with(request("S"), "temperature", parse_json("0.5")); stochastic = with(std::move(stochastic), "seed", Json::integer(7));
    session.complete(stochastic, "broker-stochastic-reset", call, sink);
    check(fake.begins == greedy_begins && fake.resets == ordinary_resets + 1);
    for (const auto* corrupt : {"epoch", "lease", "token", "lcp", "storage"}) {
        ServeSession invalid(codec(), "tokspur", "direct", ready); BrokerFake broken; broken.control_corrupt = corrupt;
        const auto bad = [&](const Json& job, const auto& event) { return broken(job, event); };
        refusal([&] { invalid.complete(request("A"), "broker-invalid-control", bad, sink); }); check(!invalid.committed_tokens());
    }
}
ServeSession enum_session(ToolEnumValidation policy) {
    return ServeSession(codec(), "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , false
#endif
        , policy);
}
void executor_enum_contract() {
    constexpr auto deferred = ToolEnumValidation::executor;
    check(parse_tool_enum_validation("strict") == ToolEnumValidation::strict && parse_tool_enum_validation("executor") == deferred);
    check(tool_enum_validation_policy(deferred) == "original-agent-executor-enums-v1");
    refusal([&] { parse_tool_enum_validation("permissive"); });
    refusal([&] { enum_session(static_cast<ToolEnumValidation>(99)); });
    const auto declarations = parse_json(R"([{"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string","enum":["cargo test --offline"]}},"required":["command"],"additionalProperties":false}}}])");
    auto job = with(request(), "tools", declarations);
    const std::string literal = "cargo test --offline 2>&1 | head -60";
    const std::string frame = "<tool_call><function=run_command><parameter=command>\n" + literal + "\n</parameter></function></tool_call><|im_end|>";
    ServeSink sink{[](const Json&){}, []{return false;}, []{return uint64_t(44);}};
    auto strict = enum_session(ToolEnumValidation::strict); Fake rejected; rejected.reply(frame);
    refusal([&] { strict.complete(job, "strict-enum", [&](const Json& command, const auto& event){return rejected(command,event);}, sink); });
    auto executor = enum_session(deferred); Fake accepted; accepted.reply(frame);
    const auto result = executor.complete(job, "executor-enum", [&](const Json& command, const auto& event){return accepted(command,event);}, sink);
    const auto proposed = message(result).find("tool_calls")->array()[0];
    check(parse_json(proposed.find("function")->find("arguments")->string()).find("command")->string() == literal);
    check(literal != "cargo test --offline"); // The unchanged executor must reject; this fixture executes nothing.
    auto prior = Json::Array{message(result), Json(Json::Object{{"role",Json("tool")}, {"tool_call_id",*proposed.find("id")},
        {"content",Json(R"({"error":"command not allowed"})")}}), Json(Json::Object{{"role",Json("user")},{"content",Json("continue")}})};
    auto next = with(job, "messages", Json(prior));
    check(admit_chat(next, "tokspur", deferred).manifest.size() == 1);
    refusal([&] { admit_chat(next, "tokspur"); });
    accepted.reply("done<|im_end|>");
    check(finish(executor.complete(next, "after-executor-error", [&](const Json& command, const auto& event){return accepted(command,event);}, sink)).string() == "stop");
    for (const auto& [value, allowed] : {std::pair{R"({"command":1})", false}, std::pair{R"({"command":"bad","extra":true})", false}, std::pair{R"({})", false}, std::pair{R"({"command":"<|im_start|>"})", true}, std::pair{R"({"command":"<|endoftext|>"})", true}}) {
        auto function = with(*proposed.find("function"), "arguments", Json(value));
        auto bad = prior; bad[0] = with(bad[0], "tool_calls", Json(Json::Array{with(proposed,"function",function)}));
        if (allowed) { static_cast<void>(admit_chat(with(next,"messages",Json(bad)),"tokspur",deferred)); ++checks; }
        else refusal([&] { admit_chat(with(next,"messages",Json(bad)),"tokspur",deferred); });
    }
    for (const auto key : {"tool_enum_validation", "tool-enum-validation"})
        refusal([&] { admit_chat(with(job,key,Json("executor")),"tokspur",deferred); });
    const auto nested = parse_json(R"({"type":"object","properties":{"items":{"type":"array","items":{"type":"integer","enum":[1],"maximum":3}}},"required":["items"],"additionalProperties":false})");
    validate_tool_value(nested,parse_json(R"({"items":[2,3]})"),deferred);++checks;
    refusal([&] { validate_tool_value(nested,parse_json(R"({"items":[2]})")); });
    for (const auto value : {R"({"items":[4]})",R"({"items":["2"]})",R"({"items":[2],"extra":true})",R"({})"})
        refusal([&] { validate_tool_value(nested,parse_json(value),deferred); });
    const auto malformed = parse_json(R"([{"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string","enum":[1]}}}}}])");
    refusal([&] { admit_chat(with(job,"tools",malformed),"tokspur",deferred); });
    for (const auto raw : {"<tool_call><function=unknown></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=extra>\nbad\n</parameter></function></tool_call><|im_end|>",
        "<tool_call><function=run_command></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=command>\n<|im_start|>\n</parameter></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=command>\n<|endoftext|>\n</parameter></function></tool_call><|im_end|>"}) {
        accepted.reply(raw);
        refusal([&] { executor.complete(job,"still-refused",[&](const Json& command,const auto& event){return accepted(command,event);},sink); });
    }
    accepted.reply("done<|im_end|>");
    refusal([&] { executor.complete(with(job,"tool_choice",Json("required")),"required",[&](const Json& command,const auto& event){return accepted(command,event);},sink); });
    accepted.reply(frame);
    const auto partial = executor.complete(with(job,"max_tokens",Json::integer(10)),"partial",[&](const Json& command,const auto& event){return accepted(command,event);},sink);
    check(finish(partial).string() == "length" && !message(partial).find("tool_calls"));
}

ServeSession arg_session(ToolArgValidation policy) {
    return ServeSession(codec(), "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
        , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
        , false
#endif
        , ToolEnumValidation::strict
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
        , false
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , Json()
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
        , Json()
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        , std::nullopt
#endif
        , policy);
}
void executor_arg_contract() {
    constexpr auto deferred = ToolArgValidation::executor;
    check(parse_tool_arg_validation("strict") == ToolArgValidation::strict && parse_tool_arg_validation("executor") == deferred);
    check(tool_arg_validation_policy(ToolArgValidation::strict) == "strict-server-args-v1");
    check(tool_arg_validation_policy(deferred) == "executor-deferred-args-v1");
    refusal([&] { parse_tool_arg_validation("permissive"); });
    refusal([&] { arg_session(static_cast<ToolArgValidation>(99)); });
    const auto declarations = parse_json(R"([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"],"additionalProperties":false}}}])");
    auto job = with(request(), "tools", declarations);
    const std::string extra_frame = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter><parameter=line>\n10\n</parameter><parameter=limit>\nten\n</parameter></function></tool_call><|im_end|>";
    ServeSink sink{[](const Json&){}, []{return false;}, []{return uint64_t(44);}};
    auto strict = arg_session(ToolArgValidation::strict); Fake rejected; rejected.reply(extra_frame);
    refusal([&] { strict.complete(job, "strict-arg", [&](const Json& command, const auto& event){return rejected(command,event);}, sink); });
    auto executor = arg_session(deferred); Fake accepted; accepted.reply(extra_frame);
    const auto result = executor.complete(job, "executor-arg", [&](const Json& command, const auto& event){return accepted(command,event);}, sink);
    const auto proposed = message(result).find("tool_calls")->array()[0];
    const auto args_json = parse_json(proposed.find("function")->find("arguments")->string());
    check(args_json.find("path")->string() == "a.rs" && args_json.find("line")->unsigned_integer() == 10 && args_json.find("limit")->string() == "ten");
    // Missing required parameter also passes to the executor as an empty object.
    accepted.reply("<tool_call><function=read_file></function></tool_call><|im_end|>");
    const auto missing_res = executor.complete(job, "executor-missing-req", [&](const Json& command, const auto& event){return accepted(command,event);}, sink);
    check(parse_json(message(missing_res).find("tool_calls")->array()[0].find("function")->find("arguments")->string()).object().empty());
    // Turn 2 history round-trip succeeds under executor arg policy and refuses under strict.
    auto prior = Json::Array{message(result), Json(Json::Object{{"role",Json("tool")}, {"tool_call_id",*proposed.find("id")},
        {"content",Json(R"({"error":"unexpected argument line"})")}}), Json(Json::Object{{"role",Json("user")},{"content",Json("retry without line")}})};
    auto next = with(job, "messages", Json(prior));
    check(admit_chat(next, "tokspur", ToolEnumValidation::strict, deferred).manifest.size() == 1);
    refusal([&] { admit_chat(next, "tokspur"); });
    accepted.reply("done<|im_end|>");
    check(finish(executor.complete(next, "after-executor-arg-error", [&](const Json& command, const auto& event){return accepted(command,event);}, sink)).string() == "stop");
    // Raw structural delimiters and unknown tools remain refused. JSON-escaped string values are inert data.
    for (const auto& [raw, allowed] : {std::pair{"<tool_call><function=unknown><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>", false},
        std::pair{"<tool_call><function=read_file><parameter=path>\n<|im_start|>\n</parameter></function></tool_call><|im_end|>", false},
        std::pair{"<tool_call><function=read_file><parameter=extra>\n{\"nested\":\"\\u003ctool_call\\u003e\"}\n</parameter></function></tool_call><|im_end|>", true},
        std::pair{"<tool_call><function=read_file><parameter=extra>\n{\"nested\":\"\\u003c|endoftext|\\u003e\"}\n</parameter></function></tool_call><|im_end|>", true}}) {
        accepted.reply(raw);
        const auto run = [&] { return executor.complete(job,"escaped-value",[&](const Json& command,const auto& event){return accepted(command,event);},sink); };
        if (allowed) check(message(run()).find("tool_calls")->array().size() == 1);
        else refusal([&] { static_cast<void>(run()); });
    }
    for (const auto& [value, allowed] : {std::pair{R"("not-an-object")", false}, std::pair{R"([1,2])", false}, std::pair{R"(123)", false}, std::pair{R"({"path":"<|im_start|>"})", true}, std::pair{R"({"path":"<|endoftext|>"})", true}}) {
        auto function = with(*proposed.find("function"), "arguments", Json(value));
        auto bad = prior; bad[0] = with(bad[0], "tool_calls", Json(Json::Array{with(proposed,"function",function)}));
        if (allowed) { static_cast<void>(admit_chat(with(next,"messages",Json(bad)),"tokspur",ToolEnumValidation::strict,deferred)); ++checks; }
        else refusal([&] { admit_chat(with(next,"messages",Json(bad)),"tokspur",ToolEnumValidation::strict,deferred); });
    }
}
// The refusal text of one call, or empty when it is admitted.
template<class F> std::string refusal_text(F&& function) { try { function(); } catch (const FormatError& error) { return error.what(); } return {}; }
// Checks one table row, naming the row when it differs.
void expect_text(const std::string& actual, std::string_view expected, std::string_view row, std::source_location at = std::source_location::current()) {
    if (actual != expected) std::cerr << "schema keyword row " << row << ": got '" << actual << "', expected '" << expected << "'\n";
    check(actual == expected, at);
}
// The same rows as generator_schema_keywords in rust/crates/tokspur-server/tests/serve_protocol.rs, with the same refusal text.
void generator_schema_keywords() {
    // Tool declarations: each schema is one property of the parameters object. Empty text means admitted.
    for (const auto& [schema, expected] : std::initializer_list<std::pair<const char*, const char*>>{
        {R"({"type":"number","exclusiveMaximum":1})", ""},
        {R"({"type":"integer","minimum":0,"exclusiveMaximum":10})", ""},
        {R"({"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1})", ""},
        {R"({"type":"number","maximum":5,"exclusiveMaximum":5})", ""},
        {R"({"type":"number","exclusiveMaximum":-1e300})", ""},
        {R"({"type":"number","minimum":1,"exclusiveMaximum":1.0000000000000000001})", ""},
        {R"({"type":"number","exclusiveMaximum":true})", "expected JSON number"},
        {R"({"type":"number","exclusiveMaximum":"1"})", "expected JSON number"},
        {R"({"type":"number","exclusiveMaximum":1e301})", "API number outside supported range"},
        {R"({"type":"string","exclusiveMaximum":0})", "numeric bound on other schema type"},
        {R"({"type":"number","minimum":1,"exclusiveMaximum":1})", "inverted schema bounds"},
        {R"({"type":"number","minimum":1.0000000000000000001,"exclusiveMaximum":1})", "inverted schema bounds"},
        {R"({"type":"number","exclusiveMinimum":1,"exclusiveMaximum":1})", "inverted schema bounds"},
        {R"({"type":"number","exclusiveMinimum":2,"exclusiveMaximum":1})", "inverted schema bounds"},
        {R"({"type":"number","exclusiveMaximum":0,"minimum":"x"})", "expected JSON number"},
        {R"({"anyOf":[{"type":"number"}],"exclusiveMaximum":1})", "unsupported API/schema field"},
        {R"({"type":"string","minLength":1,"maxLength":3})", ""},
        {R"({"type":"string","minLength":0,"maxLength":0})", ""},
        {R"({"type":"string","minLength":65536})", ""},
        {R"({"type":"string","maxLength":65537})", ""},
        {R"({"type":"string","minLength":65536,"maxLength":18446744073709551615})", ""},
        {R"({"type":"string","minLength":65537})", "minLength exceeds the tool call byte limit"},
        {R"({"type":"string","minLength":65537,"maxLength":2})", "minLength exceeds the tool call byte limit"},
        {R"({"type":"string","minLength":3,"maxLength":2})", "inverted schema bounds"},
        {R"({"type":"string","minLength":-1})", "JSON unsigned integer out of range or nonintegral"},
        {R"({"type":"string","minLength":1.0})", "JSON unsigned integer out of range or nonintegral"},
        {R"({"type":"string","minLength":18446744073709551616})", "JSON unsigned integer out of range or nonintegral"},
        {R"({"type":"string","maxLength":"2"})", "expected JSON unsigned integer"},
        {R"({"type":"string","maxLength":true,"minLength":1.5})", "JSON unsigned integer out of range or nonintegral"},
        {R"({"type":"integer","minLength":1})", "string schema keyword on other type"},
        {R"({"type":"array","items":{"type":"string"},"maxLength":2})", "string schema keyword on other type"},
        {R"({"anyOf":[{"type":"string"}],"minLength":1})", "unsupported API/schema field"},
        {R"({"type":"string","enum":["abc"],"minLength":3})", ""},
        {R"({"type":"string","enum":["ab"],"minLength":3})", "tool string below minLength"},
        {R"({"type":"array","items":{"type":"string"},"uniqueItems":true})", ""},
        {R"({"type":"array","items":{"type":"string"},"uniqueItems":false})", ""},
        {R"({"type":"array","items":{"type":"string"},"uniqueItems":1})", "expected JSON boolean"},
        {R"({"type":"array","items":{"type":"string"},"uniqueItems":"true"})", "expected JSON boolean"},
        {R"({"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2,"uniqueItems":1})", "inverted schema bounds"},
        {R"({"type":"string","uniqueItems":true})", "array schema keyword on other type"},
        {R"({"type":"object","properties":{},"uniqueItems":false})", "array schema keyword on other type"},
        {R"({"anyOf":[{"type":"array","items":{"type":"string"}}],"uniqueItems":true})", "unsupported API/schema field"},
        {R"({"type":"string","const":"a"})", ""},
        {R"({"type":"integer","const":3})", ""},
        {R"({"type":"number","const":1.50})", ""},
        {R"({"type":"boolean","const":true})", ""},
        {R"({"type":"null","const":null})", ""},
        {R"({"type":"string","enum":["a"],"const":"a"})", ""},
        {R"({"type":"string","const":1})", "tool value is not string"},
        {R"({"type":"boolean","const":"true"})", "tool value is not boolean"},
        {R"({"type":"integer","const":1.0})", "tool integer must use exact safe-integer syntax"},
        {R"({"type":"integer","const":5,"maximum":3})", "tool value above maximum"},
        {R"({"type":"string","const":"abc","maxLength":2})", "tool string above maxLength"},
        {R"({"type":"number","const":1e301})", "API number outside supported range"},
        {R"({"type":"string","enum":["a","b"],"const":"a"})", "tool value differs from const"},
        {R"({"type":"string","enum":["a"],"const":"b"})", "tool value differs from const"},
        {R"({"type":"object","properties":{},"const":{}})", "unsupported const"},
        {R"({"type":"array","items":{"type":"string"},"const":[]})", "unsupported const"},
        {R"({"anyOf":[{"type":"string"}],"const":"a"})", "unsupported API/schema field"},
        {R"({"type":"integer","multipleOf":5})", ""},
        {R"({"type":"number","multipleOf":0.01})", ""},
        {R"({"type":"number","multipleOf":1e-7})", ""},
        {R"({"type":"number","multipleOf":0.30000000000000004})", ""},
        {R"({"type":"number","multipleOf":123456789012345678})", ""},
        {R"({"type":"number","multipleOf":1234567890123456780})", ""},
        {R"({"type":"number","multipleOf":1234567890123456789})", "multipleOf digit bound"},
        {R"({"type":"number","multipleOf":0})", "multipleOf must be positive"},
        {R"({"type":"number","multipleOf":0.0})", "multipleOf must be positive"},
        {R"({"type":"number","multipleOf":-2})", "multipleOf must be positive"},
        {R"({"type":"number","multipleOf":"2"})", "expected JSON number"},
        {R"({"type":"number","multipleOf":1e301})", "API number outside supported range"},
        {R"({"type":"number","minimum":2,"maximum":1,"multipleOf":0})", "inverted schema bounds"},
        {R"({"type":"string","multipleOf":2})", "numeric bound on other schema type"},
        {R"({"anyOf":[{"type":"integer"}],"multipleOf":2})", "unsupported API/schema field"},
        {R"({"type":"integer","enum":[4,6],"multipleOf":2})", ""},
        {R"({"type":"integer","enum":[4,5],"multipleOf":2})", "tool value is not a multiple of multipleOf"},
        {R"({"type":"integer","enum":[4],"multipleOf":0})", "multipleOf must be positive"},
    }) {
        auto declaration = tools().array()[0];
        auto function = with(*declaration.find("function"), "parameters", parse_json(std::string(R"({"type":"object","properties":{"p":)") + schema + "}}"));
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(with(request(), "tools", Json(Json::Array{with(declaration, "function", function)})), "tokspur")); }), expected, schema);
    }
    // Strict validation of one value against one schema.
    for (const auto& [schema, rows] : std::initializer_list<std::pair<const char*, std::vector<std::pair<const char*, const char*>>>>{
        {R"({"type":"number","minimum":0,"exclusiveMaximum":1})", {
            {R"(0)", ""}, {R"(-0.0)", ""}, {R"(0.5)", ""}, {R"(9.9e-1)", ""}, {R"(0.99999999999999999999)", ""}, {R"(1)", "tool value at or above exclusive maximum"},
            {R"(1.0)", "tool value at or above exclusive maximum"}, {R"(1e0)", "tool value at or above exclusive maximum"}, {R"(1.00000000000000000001)", "tool value at or above exclusive maximum"},
            {R"(2)", "tool value at or above exclusive maximum"}, {R"(-0.1)", "tool value below minimum"},
        }},
        {R"({"type":"integer","exclusiveMinimum":1,"exclusiveMaximum":4})", {
            {R"(2)", ""}, {R"(3)", ""}, {R"(1)", "tool value at or below exclusive minimum"}, {R"(4)", "tool value at or above exclusive maximum"},
        }},
        {R"({"type":"number","maximum":1,"exclusiveMaximum":2})", {
            {R"(1)", ""}, {R"(1.5)", "tool value above maximum"}, {R"(2)", "tool value above maximum"},
        }},
        {R"({"type":"number","exclusiveMaximum":-1.5})", {
            {R"(-1.6)", ""}, {R"(-1.5)", "tool value at or above exclusive maximum"}, {R"(-1.50)", "tool value at or above exclusive maximum"}, {R"(-1.4)", "tool value at or above exclusive maximum"},
        }},
        {R"({"type":"string","minLength":2,"maxLength":3})", {
            {R"("ab")", ""}, {R"("abc")", ""}, {R"("")", "tool string below minLength"}, {R"("a")", "tool string below minLength"}, {R"("abcd")", "tool string above maxLength"},
            {R"("é")", "tool string below minLength"}, {R"("éé")", ""}, {R"("日本語")", ""}, {R"("日本語x")", "tool string above maxLength"}, {R"("😀")", "tool string below minLength"}, {R"("😀😀")", ""}, {R"("😀😀😀")", ""},
            {R"("😀😀😀😀")", "tool string above maxLength"}, {R"("\ud83d\ude00")", "tool string below minLength"}, {R"("\u0061\u0062")", ""},
            {R"("\ud83d\ude00\ud83d\ude00\ud83d\ude00\ud83d\ude00")", "tool string above maxLength"}, {R"("e\u0301")", ""}, {R"(5)", "tool value is not string"},
        }},
        {R"({"type":"string","minLength":0,"maxLength":0})", {
            {R"("")", ""}, {R"("a")", "tool string above maxLength"},
        }},
        {R"({"type":"array","items":{"type":"number"},"uniqueItems":true})", {
            {R"([])", ""}, {R"([1])", ""}, {R"([1,2,3])", ""}, {R"([1,1.00000000000000000001])", ""}, {R"([1,1])", "tool array repeats an item"}, {R"([1,2,1])", "tool array repeats an item"},
            {R"([1,1.0])", "tool array repeats an item"}, {R"([1,1e0])", "tool array repeats an item"}, {R"([10,1e1])", "tool array repeats an item"}, {R"([0,-0.0])", "tool array repeats an item"},
            {R"([1.5,15e-1])", "tool array repeats an item"}, {R"([1,1,"x"])", "expected JSON number"},
        }},
        {R"({"type":"array","items":{"type":"number"},"uniqueItems":false})", {
            {R"([1,1])", ""},
        }},
        {R"({"type":"array","items":{"type":"integer"},"minItems":2,"uniqueItems":true})", {
            {R"([1,2])", ""}, {R"([1])", "tool array below minItems"}, {R"([1,1])", "tool array repeats an item"},
        }},
        {R"({"type":"array","items":{"type":"string"},"uniqueItems":true})", {
            {R"(["a","b"])", ""}, {R"(["a","A"])", ""}, {R"(["é","e\u0301"])", ""}, {R"(["a","a"])", "tool array repeats an item"}, {R"(["\u0061","a"])", "tool array repeats an item"},
        }},
        {R"({"type":"array","items":{"anyOf":[{"type":"integer"},{"type":"string"},{"type":"boolean"},{"type":"null"}]},"uniqueItems":true})", {
            {R"([1,"1",true,null])", ""}, {R"([1,"1e0"])", ""}, {R"([0,false])", ""}, {R"(["true",true])", ""}, {R"(["null",null])", ""}, {R"([null,null])", "tool array repeats an item"},
            {R"([true,true])", "tool array repeats an item"}, {R"([1,"1",1])", "tool array repeats an item"},
        }},
        {R"({"type":"array","items":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"string"}}},"uniqueItems":true})", {
            {R"([{"a":1,"b":"x"},{"a":1,"b":"y"}])", ""}, {R"([{"a":1},{"b":"1"}])", ""}, {R"([{"a":1},{}])", ""}, {R"([{"a":1,"b":"x"},{"b":"x","a":1.0}])", "tool array repeats an item"},
            {R"([{},{}])", "tool array repeats an item"},
        }},
        {R"({"type":"array","items":{"type":"array","items":{"type":"number"}},"uniqueItems":true})", {
            {R"([[1,2],[2,1]])", ""}, {R"([[1],[1,1]])", ""}, {R"([[],[0]])", ""}, {R"([[1,2],[1.0,2e0]])", "tool array repeats an item"}, {R"([[],[]])", "tool array repeats an item"},
        }},
        {R"({"type":"string","const":"a"})", {
            {R"("a")", ""}, {R"("\u0061")", ""}, {R"("b")", "tool value differs from const"}, {R"("A")", "tool value differs from const"}, {R"(1)", "tool value is not string"},
        }},
        {R"({"type":"number","const":1.50})", {
            {R"(1.5)", ""}, {R"(15e-1)", ""}, {R"(1.50000000000000000001)", "tool value differs from const"}, {R"(1)", "tool value differs from const"},
        }},
        {R"({"type":"integer","const":0})", {
            {R"(0)", ""}, {R"(-0)", ""}, {R"(1)", "tool value differs from const"},
        }},
        {R"({"type":"boolean","const":true})", {
            {R"(true)", ""}, {R"(false)", "tool value differs from const"},
        }},
        {R"({"type":"null","const":null})", {
            {R"(null)", ""}, {R"(0)", "tool value is not null"},
        }},
        {R"({"type":"integer","multipleOf":5})", {
            {R"(0)", ""}, {R"(5)", ""}, {R"(-10)", ""}, {R"(7)", "tool value is not a multiple of multipleOf"}, {R"(-3)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":0.01})", {
            {R"(1.23)", ""}, {R"(0.1)", ""}, {R"(1e2)", ""}, {R"(-4.5)", ""}, {R"(0.0)", ""}, {R"(1.234)", "tool value is not a multiple of multipleOf"}, {R"(1e-3)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":0.1})", {
            {R"(0.3)", ""}, {R"(1e300)", ""}, {R"(0.30000000000000004)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":1.5})", {
            {R"(3)", ""}, {R"(4.5)", ""}, {R"(3e1)", ""}, {R"(1.5e299)", ""}, {R"(1)", "tool value is not a multiple of multipleOf"}, {R"(1e1)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":2.5})", {
            {R"(5)", ""}, {R"(7.5)", ""}, {R"(1e1)", ""}, {R"(1)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":8})", {
            {R"(1e3)", ""}, {R"(1e300)", ""}, {R"(1e2)", "tool value is not a multiple of multipleOf"}, {R"(4)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":3})", {
            {R"(3e299)", ""}, {R"(9007199254740993)", ""}, {R"(9007199254740992)", "tool value is not a multiple of multipleOf"}, {R"(1e300)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":1e-7})", {
            {R"(0.0000001)", ""}, {R"(1)", ""}, {R"(0.00000005)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":123456789012345678})", {
            {R"(123456789012345678)", ""}, {R"(246913578024691356)", ""}, {R"(1234567890123456780)", ""}, {R"(123456789012345679)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":576460752303423488})", {
            {R"(1e59)", ""}, {R"(1e60)", ""}, {R"(1e58)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":298023223876953125})", {
            {R"(1e25)", ""}, {R"(1e24)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"number","multipleOf":999999999999999999})", {
            {R"(999999999999999999)", ""}, {R"(122999999999999999877)", ""}, {R"(999999999999999999e40)", ""}, {R"(999999999999999998999999999999999998)", "tool value is not a multiple of multipleOf"},
        }},
        {R"({"type":"integer","minimum":0,"exclusiveMaximum":100,"multipleOf":25})", {
            {R"(75)", ""}, {R"(0)", ""}, {R"(100)", "tool value at or above exclusive maximum"}, {R"(30)", "tool value is not a multiple of multipleOf"}, {R"(50.0)", "tool integer must use exact safe-integer syntax"},
            {R"(-25)", "tool value below minimum"},
        }},
    }) {
        const auto parsed = parse_json(schema);
        for (const auto& [value, expected] : rows) expect_text(refusal_text([&] { validate_tool_value(parsed, parse_json(value)); }), expected, std::string(schema) + " " + value);
    }
    // A value built in memory can hold bytes that are not UTF-8. Its length cannot be counted, so it is refused as the JSON codec refuses it.
    expect_text(refusal_text([&] { validate_tool_value(parse_json(R"({"type":"string","maxLength":4})"), Json(std::string("a\xff"))); }), "invalid UTF-8 input", "bytes that are not UTF-8");
    // With enum validation deferred to the executor only const is deferred with it; every bound is still enforced.
    for (const auto& [schema, value, expected] : std::initializer_list<std::array<const char*, 3>>{
        {R"({"type":"string","const":"a"})", R"("b")", ""},
        {R"({"type":"string","const":"a"})", R"(1)", "tool value is not string"},
        {R"({"type":"number","const":1,"exclusiveMaximum":2})", R"(1.5)", ""},
        {R"({"type":"number","const":1,"exclusiveMaximum":2})", R"(2)", "tool value at or above exclusive maximum"},
        {R"({"type":"integer","multipleOf":5})", R"(7)", "tool value is not a multiple of multipleOf"},
        {R"({"type":"string","maxLength":1})", R"("ab")", "tool string above maxLength"},
        {R"({"type":"array","items":{"type":"integer"},"uniqueItems":true})", R"([1,1])", "tool array repeats an item"},
    }) expect_text(refusal_text([&] { validate_tool_value(parse_json(schema), parse_json(value), ToolEnumValidation::executor); }), expected, std::string(schema) + " " + value);
    // A recorded call: strict argument validation refuses each violation with its own text, and executor validation defers all of them.
    const auto declared = parse_json(R"([{"type":"function","function":{"name":"save","parameters":{"type":"object","properties":{"name":{"type":"string","minLength":2,"maxLength":4},"tags":{"type":"array","items":{"type":"string"},"uniqueItems":true},"ratio":{"type":"number","exclusiveMaximum":1},"step":{"type":"integer","multipleOf":5},"kind":{"type":"string","const":"file"}},"additionalProperties":false}}}])");
    for (const auto& [arguments, strict, deferred_enum] : std::initializer_list<std::array<const char*, 3>>{
        {R"({"name":"ab","tags":["a","b"],"ratio":0.5,"step":10,"kind":"file"})", "", ""},
        {R"({"name":"a"})", "tool string below minLength", "tool string below minLength"},
        {R"({"name":"abcde"})", "tool string above maxLength", "tool string above maxLength"},
        {R"({"tags":["a","a"]})", "tool array repeats an item", "tool array repeats an item"},
        {R"({"ratio":1})", "tool value at or above exclusive maximum", "tool value at or above exclusive maximum"},
        {R"({"step":7})", "tool value is not a multiple of multipleOf", "tool value is not a multiple of multipleOf"},
        {R"({"kind":"dir"})", "tool value differs from const", ""},
    }) {
        const auto history = parse_json(std::string(R"([{"role":"user","content":"save it"},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"save","arguments":)") + arguments + R"(}}]},{"role":"tool","tool_call_id":"call_1","content":"saved"},{"role":"user","content":"again"}])");
        const auto job = with(with(request(), "tools", declared), "messages", history);
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(job, "tokspur")); }), strict, arguments);
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(job, "tokspur", ToolEnumValidation::executor)); }), deferred_enum, arguments);
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(job, "tokspur", ToolEnumValidation::strict, ToolArgValidation::executor)); }), "", arguments);
    }
    // A generated call takes the same validator: the strict session refuses with its text and the executor session passes the call on.
    const auto job = with(request(), "tools", declared);
    ServeSink sink{[](const Json&){}, []{return false;}, []{return uint64_t(44);}};
    for (const auto& [parameter, value, expected] : std::initializer_list<std::array<const char*, 3>>{
        {"name", R"(abc)", ""},
        {"name", R"(a)", "tool string below minLength"},
        {"name", R"(abcde)", "tool string above maxLength"},
        {"tags", R"(["a","a"])", "tool array repeats an item"},
        {"ratio", R"(1)", "tool value at or above exclusive maximum"},
        {"step", R"(7)", "tool value is not a multiple of multipleOf"},
        {"kind", R"(dir)", "tool value differs from const"},
    }) {
        const auto frame = std::string("<tool_call><function=save><parameter=") + parameter + ">\n" + value + "\n</parameter></function></tool_call><|im_end|>";
        auto strict = arg_session(ToolArgValidation::strict); Fake rejected; rejected.reply(frame);
        expect_text(refusal_text([&] { static_cast<void>(strict.complete(job, "strict-keyword", [&](const Json& command, const auto& event){return rejected(command,event);}, sink)); }), expected, frame);
        auto executor = arg_session(ToolArgValidation::executor); Fake accepted; accepted.reply(frame);
        check(message(executor.complete(job, "executor-keyword", [&](const Json& command, const auto& event){return accepted(command,event);}, sink)).find("tool_calls")->array().size() == 1);
    }
}
// The same rows as tool_schema_patterns in rust/crates/tokspur-server/tests/serve_protocol.rs, with the same refusal text. Both are written from one table.
void tool_schema_patterns() {
    // Tool declarations: each schema is one property of the parameters object. Empty text means admitted.
    for (const auto& [schema, expected] : std::initializer_list<std::pair<const char*, const char*>>{
        {R"~({"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"})~", ""},
        {R"~({"type":"string","pattern":"^(?=.{1,253}\\.?$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[-0-9a-zA-Z]{0,61}[0-9a-zA-Z])?)*\\.?$"})~", ""},
        {R"~({"type":"string","pattern":"^P(?:(\\d+W)|(?!.*W)(?=\\d|T\\d)(\\d+Y)?(\\d+M)?(\\d+D)?(T(?=\\d)(\\d+H)?(\\d+M)?(\\d+([.,]\\d+)?S)?)?)$"})~", ""},
        {R"~({"type":"string","pattern":"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)$"})~", ""},
        {R"~({"type":"string","pattern":"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$"})~", ""},
        {R"~({"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))$"})~", ""},
        {R"~({"type":"string","pattern":"^(?:(?:\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|(?:02)-(?:0[1-9]|1\\d|2[0-8])))T(?:(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?(?:Z|([+-](?:[01]\\d|2[0-3]):[0-5]\\d)))$"})~", ""},
        {R"~({"type":"string","pattern":"^$|^(?:[0-9a-zA-Z+/]{4})*(?:(?:[0-9a-zA-Z+/]{2}==)|(?:[0-9a-zA-Z+/]{3}=))?$"})~", ""},
        {R"~({"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"})~", ""},
        {R"~({"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*(?:\\d{0,3}|(?=[\\d.]{1,6}0*$)\\d{0,3}\\.\\d{0,2}0*$)"})~", ""},
        {R"~({"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$"})~", ""},
        {R"~({"type":"string","pattern":"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"})~", ""},
        {R"~({"type":"string","pattern":"^[a-zA-Z0-9_-]{21}$"})~", ""},
        {R"~({"type":"string","pattern":"^[cC][^\\s-]{8,}$"})~", ""},
        {R"~({"type":"string","pattern":"^[0-9a-z]+$"})~", ""},
        {R"~({"type":"string","pattern":"^[0-9A-HJKMNP-TV-Za-hjkmnp-tv-z]{26}$"})~", ""},
        {R"~({"type":"string","pattern":"^[0-9a-vA-V]{20}$"})~", ""},
        {R"~({"type":"string","pattern":"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\/([0-9]|[1-2][0-9]|3[0-2])$"})~", ""},
        {R"~({"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|::|([0-9a-fA-F]{1,4})?::([0-9a-fA-F]{1,4}:?){0,6})\\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$"})~", ""},
        {R"~({"type":"string","pattern":"^[A-Za-z0-9_-]*$"})~", ""},
        {R"~({"type":"string","pattern":"^\\+(?:[0-9]){6,14}[0-9]$"})~", ""},
        {R"~({"type":"string","pattern":"^(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?$"})~", ""},
        {R"~({"type":"string","pattern":"^https:\\/\\/a b_c-d\\..*"})~", ""},
        {R"~({"type":"string","pattern":".*\\.json$"})~", ""},
        {R"~({"type":"string","pattern":"a\\+b \\(c\\)"})~", ""},
        {R"~({"type":"string","pattern":"^id--?\\d+(?:\\.\\d+)?-(a|b)$"})~", ""},
        {R"~({"type":"string","pattern":"^[^A-Z]*$"})~", ""},
        {R"~({"type":"string","pattern":"^[0-9a-fA-F]{8}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{12}$"})~", ""},
        {R"~({"type":"string","pattern":"^(([a-f0-9]{1,4}:){7}|::([a-f0-9]{1,4}:){0,6}|([a-f0-9]{1,4}:){1}:([a-f0-9]{1,4}:){0,5}|([a-f0-9]{1,4}:){2}:([a-f0-9]{1,4}:){0,4}|([a-f0-9]{1,4}:){3}:([a-f0-9]{1,4}:){0,3}|([a-f0-9]{1,4}:){4}:([a-f0-9]{1,4}:){0,2}|([a-f0-9]{1,4}:){5}:([a-f0-9]{1,4}:){0,1})([a-f0-9]{1,4}|(((25[0-5])|(2[0-4][0-9])|(1[0-9]{2})|([0-9]{1,2}))\\.){3}((25[0-5])|(2[0-4][0-9])|(1[0-9]{2})|([0-9]{1,2})))$"})~", ""},
        {R"~({"type":"string","pattern":"^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$"})~", ""},
        {R"~({"type":"string","pattern":"^[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]*$"})~", ""},
        {R"~({"type":"string","pattern":"^https\\:\\/\\/a\\ b\\_c\\-d\\."})~", ""},
        {R"~({"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*(?:\\d{0,8}|(?=[\\d.]{1,9}0*$)\\d*\\.\\d*0*$)"})~", ""},
        {R"~({"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d{0,3}0*$"})~", ""},
        {R"~({"type":"string","pattern":"^\\s*(\\d*\\.?\\d+)\\s*(\\w+)?"})~", ""},
        {R"~({"type":"string","pattern":"^\\d{3}-\\d{2}-\\d{4}$"})~", ""},
        {R"~({"type":"string","pattern":"^[a-z_][a-z0-9_]*$"})~", ""},
        {R"~({"type":"string","pattern":"^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"^(?P<area>\\d{3})-\\d{4}$"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(a)\\1"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\1"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\9"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?<n>a)"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?<n>a)\\k<n>"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\k<n>"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\p{L}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\P{Lu}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[\\p{L}]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?i)a"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?i:a)"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?#c)a"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\uD83D"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\uDE00"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\uD83Dx"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\uD83D\\u0041"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\u{D800}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\\ud83d\ude00"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[\\\ud83d\ude00]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\a"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\e"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\A"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\Z"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\z"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\G"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\Q"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[\\B]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[\\1]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[\\c1]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\c1"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\c"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\00"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\01"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\x1"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\xg0"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\u12"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\u{}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\u{110000}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\u{1F600"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"*"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"+a"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"?"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a**"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a+*"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a*??"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{1"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{,2}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{1,2"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{2,1}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{x}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"{1}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"^*"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"$+"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"\\b?"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?=a)*"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?<!a){2}"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"("})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":")"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(a"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a)"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?a)"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"(?<"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"["})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[a"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[a-"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[b-a]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[\\d-z]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[a-\\w]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"[a--]"})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"a{4096}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(?:a{64}){64}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"a{0,2048}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(?:(?:(?:a{20}){20}){20}){20}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(){4097}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"a{5000,4000}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"a{99999999999999999999999999}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"a{4096,}"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(((((((((((((((((((((((((((((((((a)))))))))))))))))))))))))))))))))"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=a"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})"})~", "pattern exceeds the tool pattern bound"},
        {R"~({"type":"string","pattern":"a{4095}"})~", ""},
        {R"~({"type":"string","pattern":"(?:a{63}){65}"})~", ""},
        {R"~({"type":"string","pattern":"a{0,2047}"})~", ""},
        {R"~({"type":"string","pattern":"(?:(?:(?:a*)*)*)*"})~", ""},
        {R"~({"type":"string","pattern":"(){4096}"})~", ""},
        {R"~({"type":"string","pattern":"((((((((((((((((((((((((((((((((a))))))))))))))))))))))))))))))))"})~", ""},
        {R"~({"type":"string","pattern":"(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)"})~", ""},
        {R"~({"type":"string","pattern":7})~", "expected JSON string"},
        {R"~({"type":"string","pattern":null})~", "expected JSON string"},
        {R"~({"type":"integer","pattern":"^1"})~", "string schema keyword on other type"},
        {R"~({"type":"array","items":{"type":"string"},"pattern":"a"})~", "string schema keyword on other type"},
        {R"~({"type":"array","items":{"type":"string","pattern":"^a"}})~", ""},
        {R"~({"anyOf":[{"type":"string"}],"pattern":"a"})~", "unsupported API/schema field"},
        {R"~({"anyOf":[{"type":"number"},{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"}]})~", ""},
        {R"~({"anyOf":[{"type":"number"},{"type":"string","pattern":"(a)\\1"}]})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"^a","enum":["ab","ac"]})~", ""},
        {R"~({"type":"string","pattern":"^a","enum":["ab","b"]})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"^a","const":"ab"})~", ""},
        {R"~({"type":"string","pattern":"^a","const":"b"})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"(","minLength":3,"maxLength":2})~", "inverted schema bounds"},
        {R"~({"type":"string","pattern":"(","minLength":65537})~", "minLength exceeds the tool call byte limit"},
        {R"~({"type":"string","pattern":"(","enum":[1]})~", "unsupported pattern"},
        {R"~({"type":"string","pattern":"^b","enum":["abc"],"minLength":4})~", "tool string below minLength"},
        {R"~({"type":"string","pattern":"^b","enum":["abc"],"minLength":3})~", "tool string does not match pattern"},
    }) {
        auto declaration = tools().array()[0];
        auto function = with(*declaration.find("function"), "parameters", parse_json(std::string(R"({"type":"object","properties":{"p":)") + schema + "}}"));
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(with(request(), "tools", Json(Json::Array{with(declaration, "function", function)})), "tokspur")); }), expected, schema);
    }
    // The pattern text is bounded too, whatever it compiles to: a class of 16382 members is one step.
    for (const auto& [members, expected] : std::initializer_list<std::pair<size_t, const char*>>{{16382, ""}, {16383, "pattern exceeds the tool pattern bound"}}) {
        const Json schema(Json::Object{{"type", Json(std::string("string"))}, {"pattern", Json("[" + std::string(members, 'a') + "]")}});
        expect_text(refusal_text([&] { validate_tool_value(schema, Json(std::string("a"))); }), expected, "pattern text length");
    }
    // Strict validation of one string against one pattern: the strings it matches, then the strings it does not.
    struct Matching { const char* schema; std::vector<const char*> yes, no; };
    for (const auto& [schema, yes, no] : std::initializer_list<Matching>{
        {R"~({"type":"string","pattern":"b"})~", {R"~("b")~", R"~("abc")~"}, {R"~("")~", R"~("a")~", R"~("B")~"}},
        {R"~({"type":"string","pattern":"^b"})~", {R"~("b")~", R"~("bc")~"}, {R"~("ab")~", R"~("a\nb")~"}},
        {R"~({"type":"string","pattern":"b$"})~", {R"~("b")~", R"~("ab")~"}, {R"~("ba")~", R"~("b\n")~"}},
        {R"~({"type":"string","pattern":"^$"})~", {R"~("")~"}, {R"~("a")~", R"~("\n")~"}},
        {R"~({"type":"string","pattern":""})~", {R"~("")~", R"~("a")~"}, {}},
        {R"~({"type":"string","pattern":"^a|b$"})~", {R"~("ax")~", R"~("xb")~"}, {R"~("xa")~", R"~("bx")~"}},
        {R"~({"type":"string","pattern":"a^b"})~", {}, {R"~("ab")~", R"~("a\nb")~"}},
        {R"~({"type":"string","pattern":"a$b"})~", {}, {R"~("ab")~", R"~("a\nb")~"}},
        {R"~({"type":"string","pattern":"^[a-c]$"})~", {R"~("a")~", R"~("c")~"}, {R"~("d")~", R"~("-")~", R"~("ab")~"}},
        {R"~({"type":"string","pattern":"^[^a-c]$"})~", {R"~("d")~", R"~("\n")~", R"~("\ud83d\ude00")~"}, {R"~("a")~", R"~("")~"}},
        {R"~({"type":"string","pattern":"^[-a]$"})~", {R"~("-")~", R"~("a")~"}, {R"~("b")~"}},
        {R"~({"type":"string","pattern":"^[a-]$"})~", {R"~("-")~", R"~("a")~"}, {R"~("b")~"}},
        {R"~({"type":"string","pattern":"^[a\\-c]$"})~", {R"~("a")~", R"~("-")~", R"~("c")~"}, {R"~("b")~"}},
        {R"~({"type":"string","pattern":"^[+--]$"})~", {R"~("+")~", R"~(",")~", R"~("-")~"}, {R"~(".")~"}},
        {R"~({"type":"string","pattern":"^[a-c-e]$"})~", {R"~("b")~", R"~("-")~", R"~("e")~"}, {R"~("d")~"}},
        {R"~({"type":"string","pattern":"^[A-Za-z0-9-_]$"})~", {R"~("Q")~", R"~("5")~", R"~("-")~", R"~("_")~"}, {R"~("+")~"}},
        {R"~({"type":"string","pattern":"^[\\d-]$"})~", {R"~("7")~", R"~("-")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"^[\\w.]$"})~", {R"~("a")~", R"~("_")~", R"~(".")~"}, {R"~("-")~"}},
        {R"~({"type":"string","pattern":"^[^\\s-]$"})~", {R"~("a")~"}, {R"~(" ")~", R"~("-")~", R"~("\t")~"}},
        {R"~({"type":"string","pattern":"^[\\b]$"})~", {R"~("\b")~"}, {R"~("b")~"}},
        {R"~({"type":"string","pattern":"^[.*+?(){}|^$\\[\\]\\\\]$"})~", {R"~(".")~", R"~("{")~", R"~("^")~", R"~("[")~", R"~("]")~", R"~("\\")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"^[]$"})~", {}, {R"~("")~", R"~("a")~"}},
        {R"~({"type":"string","pattern":"^[^]$"})~", {R"~("a")~", R"~("\n")~", R"~("\udbff\udfff")~"}, {R"~("")~", R"~("ab")~"}},
        {R"~({"type":"string","pattern":"^\\d$"})~", {R"~("0")~", R"~("9")~"}, {R"~("\u0663")~", R"~("a")~", R"~("\uff11")~"}},
        {R"~({"type":"string","pattern":"^\\D$"})~", {R"~("\u0663")~", R"~("a")~", R"~("\n")~"}, {R"~("0")~"}},
        {R"~({"type":"string","pattern":"^\\w$"})~", {R"~("a")~", R"~("Z")~", R"~("0")~", R"~("_")~"}, {R"~("\u00e9")~", R"~("-")~", R"~("\u017f")~", R"~("\u212a")~"}},
        {R"~({"type":"string","pattern":"^\\W$"})~", {R"~("\u00e9")~", R"~("-")~", R"~(" ")~"}, {R"~("a")~", R"~("_")~"}},
        {R"~({"type":"string","pattern":"^\\s$"})~", {R"~("\t")~", R"~("\n")~", R"~("\u000b")~", R"~("\f")~", R"~("\r")~", R"~(" ")~", R"~("\u00a0")~", R"~("\u1680")~", R"~("\u2000")~", R"~("\u200a")~", R"~("\u2028")~", R"~("\u2029")~", R"~("\u202f")~", R"~("\u205f")~", R"~("\u3000")~", R"~("\ufeff")~"}, {R"~("\u0085")~", R"~("\u180e")~", R"~("\u200b")~", R"~("\u2060")~", R"~("a")~", R"~("\u001c")~"}},
        {R"~({"type":"string","pattern":"^\\S$"})~", {R"~("\u0085")~", R"~("\u180e")~", R"~("\u200b")~", R"~("a")~"}, {R"~(" ")~", R"~("\u00a0")~", R"~("\u2028")~", R"~("\ufeff")~"}},
        {R"~({"type":"string","pattern":"^.$"})~", {R"~("a")~", R"~("\u0085")~", R"~("\t")~", R"~("\u000b")~", R"~("\ud83d\ude00")~"}, {R"~("\n")~", R"~("\r")~", R"~("\u2028")~", R"~("\u2029")~", R"~("")~"}},
        {R"~({"type":"string","pattern":"^.{2}$"})~", {R"~("e\u0301")~", R"~("\ud83d\ude00a")~"}, {R"~("\ud83d\ude00")~", R"~("\u00e9")~"}},
        {R"~({"type":"string","pattern":"^\ud83d\ude00+$"})~", {R"~("\ud83d\ude00")~", R"~("\ud83d\ude00\ud83d\ude00")~"}, {R"~("")~", R"~("\ud83d\ude01")~"}},
        {R"~({"type":"string","pattern":"^[\ud83d\ude00-\ud83d\ude4f]$"})~", {R"~("\ud83d\ude00")~", R"~("\ud83d\ude4f")~"}, {R"~("\ud83d\ude50")~", R"~("a")~"}},
        {R"~({"type":"string","pattern":"^\\u{1F600}\\uD83D\\uDE00$"})~", {R"~("\ud83d\ude00\ud83d\ude00")~"}, {R"~("\ud83d\ude00")~"}},
        {R"~({"type":"string","pattern":"^\\u00e9\\xe9\\u{0000e9}$"})~", {R"~("\u00e9\u00e9\u00e9")~"}, {R"~("eee")~"}},
        {R"~({"type":"string","pattern":"^\\t\\n\\v\\f\\r\\0\\cJ\\cj$"})~", {R"~("\t\n\u000b\f\r\u0000\n\n")~"}, {R"~("\t\n\u000b\f\r0\n\n")~"}},
        {R"~({"type":"string","pattern":"^a*$"})~", {R"~("")~", R"~("aaa")~"}, {R"~("b")~", R"~("ab")~"}},
        {R"~({"type":"string","pattern":"^a+?$"})~", {R"~("a")~", R"~("aaa")~"}, {R"~("")~"}},
        {R"~({"type":"string","pattern":"^a??b$"})~", {R"~("b")~", R"~("ab")~"}, {R"~("aab")~"}},
        {R"~({"type":"string","pattern":"^a{2}$"})~", {R"~("aa")~"}, {R"~("a")~", R"~("aaa")~"}},
        {R"~({"type":"string","pattern":"^a{2,}$"})~", {R"~("aa")~", R"~("aaaaa")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"^a{1,3}?$"})~", {R"~("a")~", R"~("aaa")~"}, {R"~("")~", R"~("aaaa")~"}},
        {R"~({"type":"string","pattern":"^a{0}b$"})~", {R"~("b")~"}, {R"~("ab")~"}},
        {R"~({"type":"string","pattern":"^a{0,0}$"})~", {R"~("")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"^(ab|c)+$"})~", {R"~("ab")~", R"~("c")~", R"~("abcab")~"}, {R"~("a")~", R"~("ac")~", R"~("")~"}},
        {R"~({"type":"string","pattern":"^(?:a|)b$"})~", {R"~("ab")~", R"~("b")~"}, {R"~("aab")~"}},
        {R"~({"type":"string","pattern":"^(|a|)$"})~", {R"~("")~", R"~("a")~"}, {R"~("aa")~"}},
        {R"~({"type":"string","pattern":"^()*a(?:)+$"})~", {R"~("a")~"}, {R"~("")~", R"~("aa")~"}},
        {R"~({"type":"string","pattern":"^(a*)*$"})~", {R"~("")~", R"~("aaaa")~"}, {R"~("b")~"}},
        {R"~({"type":"string","pattern":"^(a*)+b$"})~", {R"~("b")~", R"~("aab")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"^(a|ab)(c|bcd)(d*)$"})~", {R"~("abcd")~", R"~("ac")~", R"~("abc")~", R"~("acdd")~"}, {R"~("ab")~", R"~("abd")~"}},
        {R"~({"type":"string","pattern":"^((a)|(b))*?c$"})~", {R"~("c")~", R"~("abbac")~"}, {R"~("abd")~"}},
        {R"~({"type":"string","pattern":"^(?:a{2}){2,3}$"})~", {R"~("aaaa")~", R"~("aaaaaa")~"}, {R"~("aa")~", R"~("aaaaa")~", R"~("aaaaaaaa")~"}},
        {R"~({"type":"string","pattern":"\\bcat\\b"})~", {R"~("cat")~", R"~("a cat.")~", R"~("cat-flap")~"}, {R"~("cats")~", R"~("concat")~", R"~("\u00e9cat_")~"}},
        {R"~({"type":"string","pattern":"\\Bcat"})~", {R"~("concat")~", R"~("_cat")~"}, {R"~("cat")~", R"~(" cat")~", R"~("\u00e9cat")~"}},
        {R"~({"type":"string","pattern":"^\\b"})~", {R"~("a")~"}, {R"~("")~", R"~("-a")~", R"~("\u00e9")~"}},
        {R"~({"type":"string","pattern":"\\b$"})~", {R"~("a")~", R"~("-a")~"}, {R"~("")~", R"~("a-")~"}},
        {R"~({"type":"string","pattern":"^\\B$"})~", {R"~("")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"^(?=.*\\d)(?=.*[a-z]).{4,}$"})~", {R"~("ab12")~", R"~("1a..")~"}, {R"~("abcd")~", R"~("1234")~", R"~("a1")~"}},
        {R"~({"type":"string","pattern":"^(?!ab)\\w+$"})~", {R"~("ba")~", R"~("a")~", R"~("aab")~"}, {R"~("ab")~", R"~("abc")~"}},
        {R"~({"type":"string","pattern":"a(?=b)"})~", {R"~("ab")~", R"~("xab")~"}, {R"~("a")~", R"~("ac")~", R"~("ba")~"}},
        {R"~({"type":"string","pattern":"a(?!b)"})~", {R"~("a")~", R"~("ac")~", R"~("aba")~"}, {R"~("ab")~", R"~("xab")~"}},
        {R"~({"type":"string","pattern":"(?<=a)b"})~", {R"~("ab")~", R"~("xab")~"}, {R"~("b")~", R"~("cb")~"}},
        {R"~({"type":"string","pattern":"(?<!a)b"})~", {R"~("b")~", R"~("cb")~", R"~("abb")~"}, {R"~("ab")~", R"~("xab")~"}},
        {R"~({"type":"string","pattern":"^(?=a$)a$"})~", {R"~("a")~"}, {R"~("aa")~"}},
        {R"~({"type":"string","pattern":"^a(?<=^a)$"})~", {R"~("a")~"}, {R"~("aa")~"}},
        {R"~({"type":"string","pattern":"^(?:(?=a)\\w|-)+$"})~", {R"~("a-a")~", R"~("-")~", R"~("aa")~"}, {R"~("ab")~", R"~("b")~"}},
        {R"~({"type":"string","pattern":"^(?=(?!ab)a)\\w+$"})~", {R"~("ac")~", R"~("a")~"}, {R"~("ab")~", R"~("b")~"}},
        {R"~({"type":"string","pattern":"^(?=a(?<=^a)b)ab$"})~", {R"~("ab")~"}, {R"~("abb")~"}},
        {R"~({"type":"string","pattern":"(?<=(?<!x)a)b"})~", {R"~("ab")~"}, {R"~("xab")~", R"~("b")~"}},
        {R"~({"type":"string","pattern":"^(?!)"})~", {}, {R"~("")~", R"~("a")~"}},
        {R"~({"type":"string","pattern":"^(?=)$"})~", {R"~("")~"}, {R"~("a")~"}},
        {R"~({"type":"string","pattern":"(?=\\b)a"})~", {R"~("a")~", R"~(" a")~"}, {R"~("ba")~"}},
        {R"~({"type":"string","pattern":"^(?=(a|b)*c)\\w+$"})~", {R"~("abc")~", R"~("c")~", R"~("cab")~"}, {R"~("abd")~"}},
        {R"~({"type":"string","pattern":"(?<=\\d{2})-(?=\\d{2})"})~", {R"~("12-34")~", R"~("x99-00y")~"}, {R"~("1-23")~", R"~("12-3")~", R"~("ab-cd")~"}},
        {R"~({"type":"string","pattern":"^https\\:\\/\\/a\\ b\\_c\\-d\\.$"})~", {R"~("https://a b_c-d.")~"}, {R"~("https://a b_c-dx")~"}},
        {R"~({"type":"string","pattern":"^\\@\\#\\%\\&\\,\\;\\<\\=\\>\\~\\`\\!\\'\\\"$"})~", {R"~("@#%&,;<=>~`!'\"")~"}, {R"~("@")~"}},
        {R"~({"type":"string","pattern":"^na\\\u00efve\\\u2028$"})~", {R"~("na\u00efve\u2028")~"}, {R"~("naive")~"}},
        {R"~({"type":"string","pattern":"^[\\ -\\~]+$"})~", {R"~("a b~")~"}, {R"~("\t")~", R"~("\u00e9")~"}},
        {R"~({"type":"string","pattern":"^\\^\\$\\\\\\.\\*\\+\\?\\(\\)\\[\\]\\{\\}\\|$"})~", {R"~("^$\\.*+?()[]{}|")~"}, {R"~("^")~"}},
        {R"~({"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"})~", {R"~("a@b.co")~", R"~("first.last+tag@sub.example.org")~", R"~("o'neil@x-y.io")~"}, {R"~(".a@b.co")~", R"~("a..b@c.de")~", R"~("a.@b.co")~", R"~("a@b")~", R"~("a@-b.co")~", R"~("a b@c.de")~", R"~("a@b.c")~"}},
        {R"~({"type":"string","pattern":"^(?=.{1,253}\\.?$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[-0-9a-zA-Z]{0,61}[0-9a-zA-Z])?)*\\.?$"})~", {R"~("example.com")~", R"~("a")~", R"~("x-1.y.")~", R"~("localhost")~"}, {R"~("")~", R"~("-a.com")~", R"~("a-.com")~", R"~("a..b")~", R"~("a_b.com")~", R"~("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")~"}},
        {R"~({"type":"string","pattern":"^P(?:(\\d+W)|(?!.*W)(?=\\d|T\\d)(\\d+Y)?(\\d+M)?(\\d+D)?(T(?=\\d)(\\d+H)?(\\d+M)?(\\d+([.,]\\d+)?S)?)?)$"})~", {R"~("P1W")~", R"~("P1Y2M3D")~", R"~("PT1H")~", R"~("P1DT2H3M4.5S")~", R"~("PT0,5S")~"}, {R"~("P")~", R"~("PT")~", R"~("P1WT1H")~", R"~("P1S")~", R"~("1Y")~", R"~("P1Y2W")~"}},
        {R"~({"type":"string","pattern":"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)$"})~", {R"~("123e4567-e89b-12d3-a456-426614174000")~", R"~("00000000-0000-0000-0000-000000000000")~"}, {R"~("123e4567-e89b-02d3-a456-426614174000")~", R"~("123e4567e89b12d3a456426614174000")~", R"~("")~"}},
        {R"~({"type":"string","pattern":"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$"})~", {R"~("0.0.0.0")~", R"~("192.168.1.255")~"}, {R"~("256.1.1.1")~", R"~("1.2.3")~", R"~("01.2.3.4")~", R"~("1.2.3.4 ")~"}},
        {R"~({"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))$"})~", {R"~("::")~", R"~("::1")~", R"~("2001:db8::8a2e:370:7334")~", R"~("1:2:3:4:5:6:7:8")~"}, {R"~("1:2:3:4:5:6:7")~", R"~(":::")~", R"~("g::1")~", R"~("1::2::3")~"}},
        {R"~({"type":"string","pattern":"^(?:(?:\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|(?:02)-(?:0[1-9]|1\\d|2[0-8])))T(?:(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?(?:Z|([+-](?:[01]\\d|2[0-3]):[0-5]\\d)))$"})~", {R"~("2024-02-29T23:59:59Z")~", R"~("2025-12-31T00:00:00.123+05:30")~", R"~("2000-01-01T12:30Z")~"}, {R"~("2023-02-29T00:00:00Z")~", R"~("2024-13-01T00:00:00Z")~", R"~("2024-01-01 00:00:00Z")~", R"~("2024-01-01T24:00:00Z")~", R"~("2024-01-01T00:00:00")~"}},
        {R"~({"type":"string","pattern":"^$|^(?:[0-9a-zA-Z+/]{4})*(?:(?:[0-9a-zA-Z+/]{2}==)|(?:[0-9a-zA-Z+/]{3}=))?$"})~", {R"~("")~", R"~("QUJD")~", R"~("QUI=")~", R"~("QQ==")~"}, {R"~("QUJ")~", R"~("QQ=")~", R"~("QU JD")~", R"~("QUJD==")~"}},
        {R"~({"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"})~", {R"~("1")~", R"~("-1.5")~", R"~("+.5")~", R"~("007")~", R"~("1.")~"}, {R"~("")~", R"~("-")~", R"~("+.")~", R"~(".")~", R"~("1.2.3")~", R"~("1e5")~", R"~("--1")~"}},
        {R"~({"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*(?:\\d{0,3}|(?=[\\d.]{1,6}0*$)\\d{0,3}\\.\\d{0,2}0*$)"})~", {R"~("123")~", R"~("123.45")~", R"~("-0.5")~", R"~("1.50")~"}, {R"~("-")~", R"~(".")~", R"~("+")~"}},
        {R"~({"type":"string","pattern":"^[cC][^\\s-]{8,}$"})~", {R"~("cjld2cjxh0000qzrmn831i7rn")~", R"~("C12345678")~"}, {R"~("c1234567")~", R"~("c1234 5678")~", R"~("djld2cjxh0000")~"}},
        {R"~({"type":"string","pattern":"^id--?\\d+(?:\\.\\d+)?-(a|b)$"})~", {R"~("id-1-a")~", R"~("id--2.5-b")~"}, {R"~("id-1-c")~", R"~("id-1.-a")~"}},
        {R"~({"type":"string","pattern":"^https\\:\\/\\/a\\ b\\_c\\-d\\."})~", {R"~("https://a b_c-d.x")~"}, {R"~("http://a b_c-d.")~"}},
    }) {
        const auto parsed = parse_json(schema);
        for (const auto& [values, expected] : {std::pair{yes, ""}, std::pair{no, "tool string does not match pattern"}})
            for (const auto* value : values) expect_text(refusal_text([&] { validate_tool_value(parsed, parse_json(value)); }), expected, std::string(schema) + " " + value);
    }
    // Matching follows every thread at once, so nested quantifiers that would take a backtracking matcher longer than this suite runs are answered at once.
    const Json lengthy(std::string(20000, 'a'));
    for (const auto& [schema, expected] : std::initializer_list<std::pair<const char*, const char*>>{
        {R"~({"type":"string","pattern":"^(a*)*b$"})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"^(a|a)*b$"})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"^(a+)+b$"})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"(.*a){12}b"})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"^(?=(a*)*$)(a|aa)+$"})~", ""},
        {R"~({"type":"string","pattern":"^(?!(a+)+$)"})~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"(?<=(a*)*)a$"})~", ""},
        {R"~({"type":"string","pattern":"^(?:a{64}){63}a{0,31}"})~", ""},
    }) expect_text(refusal_text([&] { validate_tool_value(parse_json(schema), lengthy); }), expected, schema);
    // A value built in memory can hold bytes that are not UTF-8. It cannot be read as code points, so it is refused as the JSON codec refuses it.
    expect_text(refusal_text([&] { validate_tool_value(parse_json(R"({"type":"string","pattern":"a"})"), Json(std::string("a\xff"))); }), "invalid UTF-8 input", "bytes that are not UTF-8");
    // With enum validation deferred to the executor a pattern is still enforced.
    for (const auto& [schema, value, expected] : std::initializer_list<std::array<const char*, 3>>{
        {R"~({"type":"string","pattern":"^a","enum":["ab"]})~", R"~("ac")~", ""},
        {R"~({"type":"string","pattern":"^a","enum":["ab"]})~", R"~("b")~", "tool string does not match pattern"},
        {R"~({"type":"string","pattern":"^a","const":"ab"})~", R"~("ac")~", ""},
        {R"~({"type":"string","pattern":"^a","const":"ab"})~", R"~("b")~", "tool string does not match pattern"},
    }) expect_text(refusal_text([&] { validate_tool_value(parse_json(schema), parse_json(value), ToolEnumValidation::executor); }), expected, std::string(schema) + " " + value);
    // A recorded call: strict argument validation refuses a string that does not match, and executor validation defers it.
    const auto declared = parse_json(R"~([{"type":"function","function":{"name":"tag","parameters":{"type":"object","properties":{"slug":{"type":"string","pattern":"^[a-z]+(?:-[a-z]+)*$","maxLength":12},"email":{"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"},"amount":{"anyOf":[{"type":"number"},{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"}]}},"additionalProperties":false}}}])~");
    for (const auto& [arguments, strict] : std::initializer_list<std::pair<const char*, const char*>>{
        {R"~({"slug":"ab-cd","email":"a@b.co","amount":"1.5"})~", ""},
        {R"~({"slug":"ab--cd"})~", "tool string does not match pattern"},
        {R"~({"slug":"Ab"})~", "tool string does not match pattern"},
        {R"~({"slug":"ab-cd-ef-gh-ij"})~", "tool string above maxLength"},
        {R"~({"email":"a..b@c.de"})~", "tool string does not match pattern"},
        {R"~({"amount":2})~", ""},
        {R"~({"amount":"-."})~", "tool value matches no schema alternative"},
    }) {
        const auto history = parse_json(std::string(R"([{"role":"user","content":"tag it"},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"tag","arguments":)") + arguments + R"(}}]},{"role":"tool","tool_call_id":"call_1","content":"tagged"},{"role":"user","content":"again"}])");
        const auto job = with(with(request(), "tools", declared), "messages", history);
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(job, "tokspur")); }), strict, arguments);
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(job, "tokspur", ToolEnumValidation::executor)); }), strict, arguments);
        expect_text(refusal_text([&] { static_cast<void>(admit_chat(job, "tokspur", ToolEnumValidation::strict, ToolArgValidation::executor)); }), "", arguments);
    }
    // A generated call takes the same validator: the strict session refuses with its text and the executor session passes the call on.
    const auto job = with(request(), "tools", declared);
    ServeSink sink{[](const Json&){}, []{return false;}, []{return uint64_t(44);}};
    for (const auto& [parameter, value, expected] : std::initializer_list<std::array<const char*, 3>>{
        {"slug", "ab-cd", ""},
        {"slug", "ab--cd", "tool string does not match pattern"},
        {"slug", "ab-cd-ef-gh-ij", "tool string above maxLength"},
        {"email", "a@b.co", ""},
        {"email", ".a@b.co", "tool string does not match pattern"},
    }) {
        const auto frame = std::string("<tool_call><function=tag><parameter=") + parameter + ">\n" + value + "\n</parameter></function></tool_call><|im_end|>";
        auto strict = arg_session(ToolArgValidation::strict); Fake rejected; rejected.reply(frame);
        expect_text(refusal_text([&] { static_cast<void>(strict.complete(job, "strict-pattern", [&](const Json& command, const auto& event){return rejected(command,event);}, sink)); }), expected, frame);
        auto executor = arg_session(ToolArgValidation::executor); Fake accepted; accepted.reply(frame);
        check(message(executor.complete(job, "executor-pattern", [&](const Json& command, const auto& event){return accepted(command,event);}, sink)).find("tool_calls")->array().size() == 1);
    }
}
void server_generation_clock_contract() {
    ServeSession session(codec(), "tokspur");
    Fake fake;
    fake.reply("Generated token sequence for testing.<|im_end|>");
    std::vector<Json> traces;
    const ServeSink sink{[](const Json& chunk) { check(chunk.dump().find("server_generation_wall_ms") == std::string::npos); },
        [] { return false; }, [] { return uint64_t(44); }, [&](const Json& row) { traces.push_back(row); }};
    const auto target = [&](const Json& job, const auto& event) {
        return fake(job, event);
    };

    const auto result = session.complete(request("Test input prompt", false), "clock-contract-01", target, sink);
    check(finish(result).string() == "stop");
    check(result.find("tokspur_private") == nullptr);
    check(traces.size() == 2 && traces[0].find("event")->string() == "prompt_prefix" && traces[1].find("event")->string() == "completion");

    const auto& trace = traces[1];
    check(trace.find("response_id")->string() == "clock-contract-01");
    check(trace.find("prefill_generation_overlap")->string() == "none");
    check(trace.find("stopped_on_eos")->boolean());

    const double admission_ms = parse_number(*trace.find("server_admission_wall_ms"));
    const double prefill_ms = parse_number(*trace.find("server_prefill_wall_ms"));
    const double wall_ms = parse_number(*trace.find("server_generation_wall_ms"));
    const double request_ms = parse_number(*trace.find("server_request_wall_ms"));
    const double tp = parse_number(*trace.find("server_generation_throughput"));
    check(admission_ms >= 0.0 && prefill_ms >= 0.0 && wall_ms > 0.0 && request_ms >= prefill_ms + wall_ms - 1e-4);
    check(std::abs((admission_ms + prefill_ms + wall_ms) - request_ms) < 1e-3);
    check(tp > 0.0);

    const auto completion_tokens = result.find("usage")->find("completion_tokens")->unsigned_integer();
    check(completion_tokens > 0 && completion_tokens == fake.generations);
    check(trace.find("completion_tokens")->unsigned_integer() == completion_tokens);

    const double expected_tp = (static_cast<double>(completion_tokens) * 1000.0) / wall_ms;
    check(std::abs(tp - expected_tp) / expected_tp < 1e-4);
}

void server_generation_terminal_and_determinism_contract() {
    Json::Array outputs;
    for (int repeat = 0; repeat < 2; ++repeat) {
        ServeSession session(codec(), "tokspur");
        Fake fake;
        fake.reply("<|im_end|>");
        std::vector<Json> traces;
        const ServeSink sink{[](const Json&) {}, [] { return false; }, [] { return uint64_t(44); },
            [&](const Json& row) { traces.push_back(row); }};
        const auto target = [&](const Json& job, const auto& event) { return fake(job, event); };

        const auto result = session.complete(request("A", false), "terminal-contract-01", target, sink);
        check(finish(result).string() == "stop");
        check(result.find("tokspur_private") == nullptr);
        check(traces.size() == 2 && traces[1].find("event")->string() == "completion");

        const auto completion_tokens = result.find("usage")->find("completion_tokens")->unsigned_integer();
        check(completion_tokens == 1 && traces[1].find("completion_tokens")->unsigned_integer() == 1);
        const double wall_ms = parse_number(*traces[1].find("server_generation_wall_ms"));
        const double tp = parse_number(*traces[1].find("server_generation_throughput"));
        check(wall_ms > 0.0 && tp > 0.0);
        const double expected_tp = (static_cast<double>(completion_tokens) * 1000.0) / wall_ms;
        check(std::abs(tp - expected_tp) / expected_tp < 1e-4);

        Json::Object canonical;
        for (const auto& [name, value] : result.object()) if (name != "created") canonical.emplace_back(name, value);
        outputs.emplace_back(std::move(canonical));
    }
    check(outputs[0].dump() == outputs[1].dump());
}
} // namespace

void literal_marker_serving() {
    const std::string body = "```xml\n<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call>\n```\nDone.";
    for (const auto terminal : {"<|im_end|>", "<|endoftext|>"}) {
        ServeSession serving(codec(), "tokspur"); Fake fake; fake.reply(body + terminal);
        std::vector<Json> chunks;
        ServeSink sink{[&](const Json& value) { chunks.push_back(value); }, [] { return false; }, [] { return uint64_t(44); }};
        const auto result = serving.complete(with(request("literal", true, false), "tools", tools()), "literal-stop", [&](const Json& job, const auto& event) { return fake(job, event); }, sink);
        check(message(result).find("content")->string() == body);
        check(!message(result).find("tool_calls")); check(finish(result).string() == "stop");
        std::string streamed;
        for (const auto& chunk : chunks) {
            const auto* delta = chunk.find("choices")->array()[0].find("delta");
            if (const auto* content = delta->find("content")) streamed += content->string();
        }
        check(streamed == body);
    }
}

void translated_history_metadata() {
    for (const auto* blocks : {
        R"([{"type":"thinking","thinking":"fixture &amp; <tool_call>","signature":""}])",
        R"([{"type":"thinking","thinking":"fixture &amp; "},{"type":"thinking","thinking":"<tool_call>"}])"}) {
        const auto job = parse_json(std::string(R"({"model":"m","messages":[{"role":"assistant","content":"done","reasoning_content":"fixture &amp; <tool_call>","thinking_blocks":)") + blocks + R"(},{"role":"user","content":"next"}]})");
        const auto admitted = admit_chat(job, "m");
        check(admitted.render.find("messages")->dump() == job.find("messages")->dump());
    }
    for (const auto* message : {
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"hidden"}]})",
        R"({"role":"assistant","content":"done","reasoning_content":"visible","thinking_blocks":[{"type":"thinking","thinking":"different"}]})",
        R"({"role":"user","content":"done","thinking_blocks":[]})",
        R"({"role":"assistant","content":"done","thinking_blocks":null})",
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"redacted_thinking","data":"opaque"}]})",
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"","signature":"unverified"}]})",
        R"({"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"","extra":true}]})"}) {
        refusal([&] { admit_chat(parse_json(std::string(R"({"model":"m","messages":[)") + message + "]}"), "m"); });
    }
    static_cast<void>(admit_chat(parse_json(R"({"model":"m","messages":[{"role":"assistant","content":"done","thinking_blocks":[]}]})"), "m")); ++checks;
    for (const auto* index : {"0", "1", "127", "128", "-1", "0.0", "\"0\""}) {
        const auto job = parse_json(std::string(R"({"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"call-a","type":"function","index":)") + index + R"(,"function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-a","content":"done"}],"tools":[{"type":"function","function":{"name":"f"}}]})");
        if (std::string_view(index) == "0" || std::string_view(index) == "1" || std::string_view(index) == "127") { static_cast<void>(admit_chat(job, "m")); ++checks; }
        else refusal([&] { admit_chat(job, "m"); });
    }
    for (const auto* index : {"0", "1", "2", "128"}) {
        const auto job = parse_json(std::string(R"({"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"a","type":"function","index":0,"function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","index":)") + index + R"(,"function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}],"tools":[{"type":"function","function":{"name":"f"}}]})");
        if (std::string_view(index) != "128") { static_cast<void>(admit_chat(job, "m")); ++checks; }
        else refusal([&] { admit_chat(job, "m"); });
    }
}
void added_token_framing() {
    constexpr uint32_t open = 2500, close = 2501;
    for (const bool stream : {false, true}) {
        for (const bool marked : {false, true}) {
            auto bindings = codec();
            bindings.control_ids = {open, close, 2502, 2503};
            bindings.decode = [](std::span<const uint32_t> ids) {
                std::string text;
                for (const auto& id : ids) {
                    if (id >= 2500 && id < 2504) text += response_control_markers[id - 2500];
                    else text += decode(std::span(&id, 1));
                }
                return text;
            };
            ServeSession session(std::move(bindings));
            Fake fake;
            if (marked) fake.output = {open, 'A' + 1, close, 'B' + 1, 1000};
            else fake.reply("<think>A</think>B<|im_end|>");
            std::string content, thinking;
            ServeSink sink{[&](const Json& chunk) {
                const auto* delta = chunk.find("choices")->array()[0].find("delta");
                if (const auto* part = delta->find("content")) content += part->string();
                if (const auto* part = delta->find("reasoning_content")) thinking += part->string();
            }, [] { return false; }, [] { return uint64_t(42); }};
            const auto result = session.complete(request("fixture", stream), "frames", [&](const Json& job, const auto& event) { return fake(job, event); }, sink);
            const std::string expected = marked ? "B" : "<think>A</think>B";
            check(message(result).find("content")->string() == expected);
            const auto* reasoning = message(result).find("reasoning_content");
            check(marked ? reasoning && reasoning->string() == "A" : !reasoning);
            check(content == (stream ? expected : ""));
            check(thinking == (stream && marked ? "A" : ""));
        }
    }
}

void accumulated_frame_refusal_order() {
    const auto input = parse_json(R"({"ids":[66,67],"tools":[],"initial_thinking":false,"finish_reason":"length","stop_id":null})");
    const std::array<uint32_t, 1> controls{2500};
    for (const bool decoder_fails : {true, false}) {
        size_t singles = 0;
        const auto decoder = [&](std::span<const uint32_t> ids) -> std::string {
            if (ids.size() == 2) return "AB";
            ++singles;
            if (ids[0] == 67 && decoder_fails) throw FormatError("fixture decoder failure");
            return "X";
        };
        try {
            static_cast<void>(parse_native_response_with_controls(input, {4096, 1000, 1001, 1002}, decoder, 32, controls));
            check(false);
        } catch (const FormatError& error) {
            check(std::string_view(error.what()) == (decoder_fails ? "fixture decoder failure" : "response decoder/terminal identity mismatch"));
        }
        check(singles == 2);
    }
}

void final_prefix_check() {
    for (const bool stream : {false, true}) {
        auto bindings = codec();
        bindings.decode = [](std::span<const uint32_t> ids) {
            auto text = decode(ids);
            if (ids.size() > 1 && text == "AB<|im_end|>") text.front() = 'X';
            return text;
        };
        ServeSession session(std::move(bindings));
        Fake fake;
        fake.reply("AB<|im_end|>");
        std::string delivered;
        ServeSink sink{[&](const Json& chunk) {
            const auto* delta = chunk.find("choices")->array()[0].find("delta")->find("content");
            if (delta) delivered += delta->string();
        }, [] { return false; }, [] { return uint64_t(42); }};
        try {
            session.complete(request("fixture", stream), "prefix", [&](const Json& job, const auto& event) { return fake(job, event); }, sink);
            check(false);
        } catch (const FormatError& error) {
            check(std::string_view(error.what()) == "final response disagrees with streamed prefix");
        }
        check(delivered == (stream ? "AB" : ""));
    }
}

void parallel_tool_streams() {
    for (const bool valid : {true, false}) {
        ServeSession session(codec()); Fake fake; Json::Array chunks;
        ServeSink sink{[&](const Json& value) { chunks.push_back(value); }, [] { return false; }, [] { return uint64_t(42); }};
        const auto frame = [](std::string_view name, std::string_view path) { return "<tool_call><function=" + std::string(name) + "><parameter=path>\n" + std::string(path) + "\n</parameter></function></tool_call>"; };
        fake.reply(frame("read_file", "a.rs") + "\n" + frame(valid ? "read_file" : "unknown", "b.rs") + "<|im_end|>");
        const auto job = with(with(request("fixture", true), "tools", tools()), "max_tokens", Json::integer(512));
        const auto call = [&](const Json& command, const auto& event) { return fake(command, event); };
        if (valid) {
            const auto result = session.complete(job, "parallel", call, sink);
            check(message(result).find("tool_calls")->array().size() == 2);
            size_t index = 0;
            for (const auto& chunk : chunks) {
                const auto* calls = chunk.find("choices")->array()[0].find("delta")->find("tool_calls");
                if (!calls) continue;
                check(calls->array().size() == 1 && calls->array()[0].find("index")->unsigned_integer() == index);
                check(parse_json(calls->array()[0].find("function")->find("arguments")->string()).find("path")->string() == (index ? "b.rs" : "a.rs"));
                ++index;
            }
            check(index == 2);
        } else {
            refusal([&] { session.complete(job, "parallel-invalid", call, sink); });
            for (const auto& chunk : chunks) check(!chunk.find("choices")->array()[0].find("delta")->find("tool_calls"));
        }
    }
}
int main() {
    try {
        added_token_framing();
        accumulated_frame_refusal_order();
        final_prefix_check();
        parallel_tool_streams();
        translated_history_metadata();
        literal_marker_serving();
        worker_ready_admission();
        original_agent_contract();
        extended_output_budget();
        respelled_prefix_reuse();
        respelled_prefix_preserves_nonordinary_tokens();
        respelled_prefix_decodes_only_changed_tokens();
        parked_lane_http_fast_math_compatibility();
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
        parked_lane_prefix_respelling_and_hoisting();
#endif
        startup_execution_policy();
        broker_sessions();
        executor_enum_contract();
        executor_arg_contract();
        generator_schema_keywords();
        tool_schema_patterns();
        server_generation_clock_contract();
        server_generation_terminal_and_determinism_contract();
        for (const auto text : {
            R"({"model":"wrong","messages":[]})", R"({"model":"tokspur","messages":[],"n":2})",
            R"({"model":"tokspur","messages":[],"max_tokens":0})", R"({"model":"tokspur","messages":[],"max_tokens":32769})",
            R"({"model":"tokspur","messages":[],"temperature":-1})", R"({"model":"tokspur","messages":[],"top_p":0})",
            R"({"model":"tokspur","messages":[],"max_tokens":1,"max_completion_tokens":1})",
            R"({"model":"tokspur","messages":[{"role":"tool","content":"x","tool_call_id":"fake"}]})",
            R"({"model":"tokspur","messages":[{"role":"user","content":[{"type":"image_url"}]}]})"
        }) refusal([&] { static_cast<void>(admit_chat(parse_json(text), "tokspur")); });
        auto admitted = admit_chat(with(request(), "tools", tools()), "tokspur"); check(admitted.manifest.size() == 1 && admitted.names[0] == "read_file");
        auto text_parts = with(request(), "messages", parse_json(R"([{"role":"user","content":[{"type":"text","text":"hello"}]}])"));
        check(admit_chat(text_parts, "tokspur").maximum_tokens == 256);
        const auto usage_option = parse_json(R"({"include_usage":true})");
        check(admit_chat(with(with(request("A",true), "stream_options", usage_option), "keep_alive", parse_json("-1.0")), "tokspur").include_usage);
        refusal([&] { admit_chat(with(request(), "stream_options", usage_option), "tokspur"); });
        check(admit_chat(with(request(), "keep_alive", Json("5m")), "tokspur").maximum_tokens == 256);
        for (const auto& invalid : {Json(true), Json(), Json("tomorrow"), parse_json("-2")})
            refusal([&] { admit_chat(with(request(), "keep_alive", invalid), "tokspur"); });
        for (const auto& invalid : {Json(true), Json(), parse_json(R"({"include_usage":1})"), parse_json(R"({"other":true})")})
            refusal([&] { admit_chat(with(request("A",true), "stream_options", invalid), "tokspur"); });
        {
            ServeSession usage_session(codec()); Fake worker; worker.reply("Hello<|im_end|>"); std::vector<Json> events;
            const auto target = [&](const Json& job, const auto& event) { return worker(job,event); };
            const ServeSink output{[&](const Json& value) { events.push_back(value); }, [] { return false; }, [] { return uint64_t(44); }};
            const auto response = usage_session.complete(with(with(request("A",true), "stream_options", usage_option), "keep_alive", parse_json("-1")), "usage", target, output);
            check(events.size() > 2 && events.back().find("choices")->array().empty());
            check(events.back().find("usage")->dump() == response.find("usage")->dump());
            size_t finishes = 0;
            for (size_t n = 0; n + 1 < events.size(); ++n) {
                check(events[n].find("usage") && events[n].find("usage")->is_null());
                if (!events[n].find("choices")->array()[0].find("finish_reason")->is_null()) ++finishes;
            }
            check(finishes == 1);
            const auto old = usage_session.committed_tokens(); events.clear(); worker.reply("Next<|im_end|>");
            const auto next = usage_session.complete(with(with(request("AHello<|im_end|>B",true), "stream_options", parse_json(R"({"include_usage":false})")), "keep_alive", Json::integer(0)), "no-usage", target, output);
            check(worker.resets == 1 && next.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == old);
            for (const auto& event : events) check(!event.find("usage") && event.find("choices")->array().size() == 1);
        }
        for (const auto schema : {
            R"({"type":"object","oneOf":[]})", R"({"type":"object","additionalProperties":true})",
            R"({"type":"array"})", R"({"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2})",
            R"({"type":"array","items":{"type":"string"},"minItems":-1})", R"({"type":"string","minItems":1})",
            R"({"type":"array","items":{"type":"string"},"minItems":4097})", R"({"type":"array","items":{"type":"string"},"minItems":1.0})",
            R"({"type":"array","items":{"type":"string"},"maxItems":"2"})", R"({"anyOf":[{"type":"array","items":{"type":"string"}}],"minItems":1})",
            R"({"type":["string","null"]})", R"({"type":"object","required":["missing"]})"
        }) {
            auto declaration = tools().array()[0]; auto function = with(*declaration.find("function"), "parameters", parse_json(schema));
            refusal([&] { static_cast<void>(admit_chat(with(request(), "tools", Json(Json::Array{with(declaration, "function", function)})), "tokspur")); });
        }
        const auto schema = parse_json(R"({"type":"object","properties":{"n":{"type":"integer","minimum":1,"maximum":3},"values":{"type":"array","items":{"type":"string","enum":["x","y"]},"minItems":1,"maxItems":2}},"required":["n"]})");
        validate_tool_value(schema, parse_json(R"({"n":2,"values":["x","y"]})")); ++checks;
        for (const auto value : {R"({"n":1.5})", R"({"n":1.0})", R"({"n":4})", R"({"n":9007199254740993})", R"({"n":2,"values":[]})", R"({"n":2,"values":["x","y","x"]})", R"({"n":2,"values":["z"]})", R"({"n":2,"extra":true})", R"({})"}) refusal([&] { validate_tool_value(schema, parse_json(value)); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"integer"})"), parse_json("1.00000000000000001")); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"number","enum":[1]})"), parse_json("1.00000000000000001")); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"number","enum":[9007199254740992]})"), parse_json("9007199254740993")); });
        refusal([&] { validate_tool_value(parse_json(R"({"type":"number","maximum":1})"), parse_json("1.00000000000000001")); });
        validate_tool_value(parse_json(R"({"type":"number","enum":[1.0]})"), parse_json("1e0")); ++checks;
        validate_tool_value(parse_json(R"({"type":"number","minimum":-1,"maximum":0})"), parse_json("-0.1")); ++checks;
        const auto opencode_schema = parse_json(R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"timeout":{"type":"number","exclusiveMinimum":0,"maximum":60}},"required":["timeout"]})");
        auto opencode_declaration = tools().array()[0];
        auto opencode_function = with(*opencode_declaration.find("function"), "parameters", opencode_schema);
        check(admit_chat(with(request(), "tools", Json(Json::Array{with(opencode_declaration, "function", opencode_function)})), "tokspur").manifest.size() == 1);
        for (const auto value : {R"({"timeout":0.0001})", R"({"timeout":1})", R"({"timeout":60})"}) { validate_tool_value(opencode_schema, parse_json(value)); ++checks; }
        for (const auto value : {R"({"timeout":0})", R"({"timeout":-0.0})", R"({"timeout":-1})", R"({"timeout":60.01})"}) refusal([&] { validate_tool_value(opencode_schema, parse_json(value)); });
        std::vector<Json> chunks; bool cancel = false; size_t seeds = 0;
        ServeSink sink{[&](const Json& chunk) { chunks.push_back(chunk); }, [&] { return cancel; }, [&] { ++seeds; return uint64_t(42); }};
        ServeSession session(codec()); Fake fake; fake.reply("Hello 🌍<|im_end|>");
        auto call = [&](const Json& job, const auto& event) { return fake(job, event); };
        auto result = session.complete(request("A", true), "first", call, sink);
        check(message(result).find("content")->string() == "Hello 🌍" && finish(result).string() == "stop");
        std::string streamed;
        for (const auto& event : chunks) {
            const auto reparsed = parse_json(event.dump()); // Every SSE JSON payload has complete UTF-8.
            if (const auto* part = reparsed.find("choices")->array()[0].find("delta")->find("content")) streamed += part->string();
        }
        check(streamed == "Hello 🌍");
        check(fake.resets == 1 && seeds == 1 && !chunks.empty());
        const size_t old = session.committed_tokens(); fake.reply("Next<|im_end|>");
        result = session.complete(request("AHello 🌍<|im_end|>B"), "second", call, sink);
        check(fake.resets == 1 && seeds == 1 && result.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == old);
        fake.reply("Changed<|im_end|>");
        result = session.complete(with(request("AHello 🌍<|im_end|>BNext<|im_end|>C"), "temperature", parse_json("0.5")), "third", call, sink);
        check(fake.resets == 2 && seeds == 2);
        const auto before = fake.prefills + fake.generations;
        refusal([&] { session.complete(with(request(), "n", Json::integer(2)), "invalid", call, sink); }); check(before == fake.prefills + fake.generations);
        refusal([&] { session.complete(request(std::string(65536, 'A')), "overflow", call, sink); }); check(before == fake.prefills + fake.generations);
        {
            const auto make_session_ctx = [&](uint32_t ctx) {
                return ServeSession(codec(), "tokspur", "direct", Json()
#ifdef TOKSPUR_PRIVATE_HTTP_VERIFY8
                    , false, Json()
#endif
#ifdef TOKSPUR_PRIVATE_HTTP_BRANCH8
                    , false
#endif
                    , ToolEnumValidation::strict
#ifdef TOKSPUR_PRIVATE_HTTP_SAMPLED_PREFILL
                    , false
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    , Json()
#endif
#ifdef TOKSPUR_PRIVATE_LANE_LAYOUT
                    , Json()
#endif
#ifdef TOKSPUR_PRIVATE_NATIVE_MTP
                    , std::nullopt
#endif
                    , ToolArgValidation::strict, ctx);
            };
            const auto large_req = request(std::string(1100000, 'A'));
            refusal([&] { admit_chat(large_req, "tokspur"); });
            check(admit_chat(large_req, "tokspur", ToolEnumValidation::strict, ToolArgValidation::strict, 131072).maximum_tokens == 256);
            refusal([&] { admit_chat(request(std::string(4200000, 'A')), "tokspur", ToolEnumValidation::strict, ToolArgValidation::strict, 131072); });
            refusal([&] { admit_chat(request(), "tokspur", ToolEnumValidation::strict, ToolArgValidation::strict, 100000); });
            refusal([&] { static_cast<void>(make_session_ctx(100000)); });
            auto session128k = make_session_ctx(131072);
            Fake fake128k;
            auto call128k = [&](const Json& job, const auto& event) { return fake128k(job, event); };
            ServeSink sink128k{[](const Json&) {}, [] { return false; }, [] { return uint64_t(77); }};
            refusal([&] { session128k.complete(request(std::string(131072, 'A')), "overflow128k", call128k, sink128k); });
            check(fake128k.prefills + fake128k.generations == 0);
            fake128k.reply("ok<|im_end|>");
            const auto r128k = session128k.complete(request(std::string(65536, 'A')), "admitted128k", call128k, sink128k);
            check(finish(r128k).string() == "stop" && session128k.committed_tokens() == 65539);
            fake128k.reply("xyz");
            const auto r128k_clamp = session128k.complete(with(request(std::string(65536, 'A') + "ok<|im_end|>" + std::string(65531, 'B')), "max_tokens", Json::integer(100)), "clamped128k", call128k, sink128k);
            check(finish(r128k_clamp).string() == "length" && message(r128k_clamp).find("content")->string() == "xy" && session128k.committed_tokens() == 131072);
        }
        chunks.clear(); fake.reply("why</think>\n\nVisible<|im_end|>");
        result = session.complete(request("T", true, true), "thinking", call, sink);
        check(message(result).find("reasoning_content")->string() == "why" && message(result).find("content")->string() == "\n\nVisible");
        auto tool_request = with(request("U", true), "tools", tools());
        const std::string tool_response = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>";
        chunks.clear(); fake.reply(tool_response); result = session.complete(tool_request, "tool", call, sink);
        check(finish(result).string() == "tool_calls" && message(result).find("tool_calls")->array()[0].find("id")->string() == "tool_call_0");
        size_t tool_chunks = 0; for (const auto& chunk : chunks) if (chunk.find("choices")->array()[0].find("delta")->find("tool_calls")) ++tool_chunks; check(tool_chunks == 1);
        auto history_request = with(request("R"), "tools", tools());
        history_request = with(std::move(history_request), "messages", parse_json(R"([{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},{"role":"tool","tool_call_id":"a","content":"file"},{"role":"user","content":"R"}])"));
        check(admit_chat(history_request, "tokspur").manifest.size() == 1);
        auto bad_history = history_request.find("messages")->array(); bad_history.push_back(parse_json(R"({"role":"tool","tool_call_id":"a","content":"duplicate"})"));
        refusal([&] { admit_chat(with(history_request, "messages", Json(bad_history)), "tokspur"); });
        bad_history = history_request.find("messages")->array(); bad_history[1] = with(bad_history[1], "name", Json("other"));
        refusal([&] { admit_chat(with(history_request, "messages", Json(bad_history)), "tokspur"); });
        for (const auto corrupt : {"prefix", "policy", "terminal", "throw"}) {
            chunks.clear(); fake.corrupt = corrupt; fake.reply(tool_response);
            refusal([&] { session.complete(tool_request, "bad", call, sink); }); check(session.committed_tokens() == 0);
            for (const auto& chunk : chunks) check(!chunk.find("choices")->array()[0].find("delta")->find("tool_calls"));
        }
        fake.corrupt.clear(); fake.reply("safe<|im_end|>"); result = session.complete(request(), "recovery", call, sink); check(finish(result).string() == "stop");
        chunks.clear(); fake.reply(tool_response); auto limited = with(tool_request, "max_tokens", Json::integer(10));
        result = session.complete(limited, "truncated", call, sink); check(finish(result).string() == "length" && !message(result).find("tool_calls"));
        chunks.clear(); fake.reply("not a call<|im_end|>");
        refusal([&] { session.complete(with(tool_request, "tool_choice", Json("required")), "required", call, sink); });
        for (const auto* name : {"auto", "none", "required"}) {
            auto declarations = tools().array();
            declarations.push_back(Json(Json::Object{{"type", Json("function")}, {"function", Json(Json::Object{{"name", Json(name)}})}}));
            auto named = with(request("named"), "tools", Json(std::move(declarations)));
            named = with(named, "tool_choice", Json(Json::Object{{"type", Json("function")}, {"function", Json(Json::Object{{"name", Json(name)}})}}));
            const auto admitted = admit_chat(named, "tokspur");
            check(admitted.names.size() == 1 && admitted.names[0] == name && admitted.render.find("tools")->array().size() == 1);
            check(admitted.render.find("messages")->array()[0].find("content")->string().find(std::string("named ") + name) != std::string_view::npos);
            fake.reply("not a call<|im_end|>");
            refusal([&] { session.complete(named, "named-missing", call, sink); });
            fake.reply(std::string("<tool_call><function=") + name + "></function></tool_call><|im_end|>");
            const auto selected = session.complete(named, "named-ok", call, sink);
            check(finish(selected).string() == "tool_calls" && message(selected).find("tool_calls")->array()[0].find("function")->find("name")->string() == name);
        }
        cancel = true; const auto previous = fake.prefills + fake.generations; const auto previous_tokens = session.committed_tokens();
        refusal([&] { session.complete(request(), "cancelled", call, sink); }); check(fake.prefills + fake.generations == previous && session.committed_tokens() == previous_tokens); cancel = false;
        fake.reply("more<|im_end|>");
        auto cancel_after_commit = [&](const Json& job, const auto& event) { auto done = fake(job, event); if (job.find("generate")->unsigned_integer()) cancel = true; return done; };
        refusal([&] { session.complete(request("C", true), "midcancel", cancel_after_commit, sink); }); check(session.committed_tokens() == 0); cancel = false;
        fake.reply("<tool_call><function=unknown></function></tool_call><|im_end|>");
        refusal([&] { session.complete(tool_request, "unknown", call, sink); });
        fake.reply("<tool_call><function=read_file><parameter=path>\ntrue\n</parameter></function></tool_call><|im_end|>");
        result = session.complete(tool_request, "rawstring", call, sink); check(message(result).find("tool_calls")->array()[0].find("function")->find("arguments")->string() == "{\"path\":\"true\"}");
        {
            const auto base = parse_json(R"({"model":"tokspur","messages":[{"role":"user","content":"Q"}]})");
            check(!admit_chat(base, "tokspur").thinking_budget.has_value());
            const auto top = admit_chat(with(base, "thinking_budget", Json::integer(128)), "tokspur");
            check(top.thinking && top.thinking_budget == std::optional<uint32_t>(128) && !top.render.find("thinking_budget"));
            const auto kw = admit_chat(with(base, "chat_template_kwargs", parse_json(R"({"thinking_budget":256,"reasoning_effort":"medium"})")), "tokspur");
            check(kw.thinking && kw.thinking_budget == std::optional<uint32_t>(256) && kw.render.find("reasoning_effort")->string() == "medium");
            const auto both = admit_chat(with(with(base, "thinking_budget", Json::integer(512)), "chat_template_kwargs", parse_json(R"({"thinking_budget":512})")), "tokspur");
            check(both.thinking && both.thinking_budget == std::optional<uint32_t>(512));
            const auto max_b = admit_chat(with(base, "thinking_budget", Json::integer(32768)), "tokspur");
            check(max_b.thinking && max_b.thinking_budget == std::optional<uint32_t>(32768));
            const auto zero = admit_chat(with(base, "thinking_budget", Json::integer(0)), "tokspur");
            check(!zero.thinking && !zero.thinking_budget.has_value() && !zero.render.find("enable_thinking")->boolean());
            const auto zero_none = admit_chat(with(with(base, "thinking_budget", Json::integer(0)), "reasoning_effort", Json("none")), "tokspur");
            check(!zero_none.thinking && !zero_none.thinking_budget.has_value());
            const auto zero_false = admit_chat(with(with(base, "thinking_budget", Json::integer(0)), "enable_thinking", Json(false)), "tokspur");
            check(!zero_false.thinking && !zero_false.thinking_budget.has_value());
            for (const auto& invalid : {Json(), Json(true), Json("128"), parse_json("-1"), parse_json("1.5"), Json::integer(32769)})
                refusal([&] { static_cast<void>(admit_chat(with(base, "thinking_budget", invalid), "tokspur")); });
            refusal([&] { static_cast<void>(admit_chat(with(with(base, "thinking_budget", Json::integer(0)), "enable_thinking", Json(true)), "tokspur")); });
            // A positive budget with thinking explicitly off is admitted and ignored, wherever either option is given; it still has to be a valid budget.
            for (const auto& off_request : {
                with(with(base, "thinking_budget", Json::integer(64)), "enable_thinking", Json(false)),
                with(base, "chat_template_kwargs", parse_json(R"({"thinking_budget":64,"enable_thinking":false})")),
                with(with(base, "thinking_budget", Json::integer(64)), "chat_template_kwargs", parse_json(R"({"enable_thinking":false})")),
                with(with(with(base, "thinking_budget", Json::integer(64)), "enable_thinking", Json(false)), "reasoning_effort", Json("none"))}) {
                const auto off = admit_chat(off_request, "tokspur");
                check(!off.thinking && !off.thinking_budget.has_value() && !off.render.find("enable_thinking")->boolean());
            }
            refusal([&] { static_cast<void>(admit_chat(with(with(base, "thinking_budget", Json::integer(32769)), "enable_thinking", Json(false)), "tokspur")); });
            refusal([&] { static_cast<void>(admit_chat(with(with(base, "thinking_budget", Json::integer(64)), "chat_template_kwargs", parse_json(R"({"thinking_budget":128})")), "tokspur")); });
            refusal([&] { static_cast<void>(admit_chat(with(with(base, "thinking_budget", Json::integer(64)), "reasoning_effort", Json("none")), "tokspur")); });

            chunks.clear(); fake.reply("abcdefVisible<|im_end|>");
            auto tb_res = session.complete(with(request("T", true, true), "thinking_budget", Json::integer(6)), "tb-cutoff", call, sink);
            check(finish(tb_res).string() == "stop" && message(tb_res).find("reasoning_content")->string() == "abcdef\n" &&
                message(tb_res).find("content")->string() == "\n\nVisible");
            std::string s_think, s_text;
            for (const auto& ev : chunks) {
                const auto& delta = *ev.find("choices")->array()[0].find("delta");
                if (const auto* t = delta.find("reasoning_content")) s_think += t->string();
                if (const auto* c = delta.find("content")) s_text += c->string();
            }
            check(s_think == "abcdef\n" && s_text == "\n\nVisible");

            chunks.clear(); fake.reply("abc\nDone<|im_end|>");
            tb_res = session.complete(with(request("T2", false, true), "thinking_budget", Json::integer(4)), "tb-nl", call, sink);
            check(message(tb_res).find("reasoning_content")->string() == "abc\n" && message(tb_res).find("content")->string() == "\n\nDone");

            const std::string fence_think = "```py\nx=1";
            chunks.clear(); fake.reply(fence_think + tool_response);
            tb_res = session.complete(with(with(request("F", true, true), "tools", tools()), "thinking_budget", Json::integer(fence_think.size())), "tb-fence-tool", call, sink);
            check(finish(tb_res).string() == "tool_calls" && message(tb_res).find("reasoning_content")->string() == "```py\nx=1\n" &&
                message(tb_res).find("tool_calls")->array().size() == 1);

            chunks.clear(); fake.reply("考Yes<|im_end|>");
            tb_res = session.complete(with(request("U8", false, true), "thinking_budget", Json::integer(1)), "tb-utf8", call, sink);
            check(message(tb_res).find("reasoning_content")->string() == "考\n" && message(tb_res).find("content")->string() == "\n\nYes");

            chunks.clear(); fake.reply("hi</think>\n\nNatural<|im_end|>");
            tb_res = session.complete(with(request("N", false, true), "thinking_budget", Json::integer(50)), "tb-natural", call, sink);
            check(message(tb_res).find("reasoning_content")->string() == "hi" && message(tb_res).find("content")->string() == "\n\nNatural");

            // The budget is reached after the model closed thinking on its own: on the close's last token (10) or inside the answer (13). Nothing is injected and the answer continues in phase 2.
            for (const uint64_t budget : {10u, 13u}) {
                chunks.clear(); fake.reply("hi</think>\n\nNatural<|im_end|>");
                tb_res = session.complete(with(request("NB", false, true), "thinking_budget", Json::integer(budget)), "tb-natural-boundary", call, sink);
                check(finish(tb_res).string() == "stop" && message(tb_res).find("reasoning_content")->string() == "hi" &&
                    message(tb_res).find("content")->string() == "\n\nNatural" && tb_res.find("usage")->find("completion_tokens")->unsigned_integer() == 20);
            }

            chunks.clear(); fake.reply("abcdefghij");
            tb_res = session.complete(with(with(request("L", false, true), "thinking_budget", Json::integer(3)), "max_tokens", Json::integer(5)), "tb-tight", call, sink);
            check(finish(tb_res).string() == "length" && tb_res.find("usage")->find("completion_tokens")->unsigned_integer() == 5);
        }
        {
            const auto escaped_body = [](std::string_view control, size_t escapes) {
                std::string body = R"({"model":"m","messages":[{"role":"user","content":")";
                body.reserve(body.size() + 2 * escapes + 4);
                for (size_t i = 0; i < escapes; ++i) body.append(control);
                body.append(R"("}]})");
                return body;
            };
            for (const auto& [context_length, limit, values] : std::vector<std::tuple<uint32_t, size_t, size_t>>{
                {65536u, 1024u * 1024u, 150000u},
                {131072u, 4u * 1024u * 1024u, 300000u},
            }) {
                for (const std::string_view control : {"\\n", "\\t"}) {
                    const size_t frame = escaped_body(control, 0).size();
                    const auto body = escaped_body(control, (limit - frame) / 2);
                    check(limit - body.size() < 2);
                    const auto parsed = parse_json(body, {limit, values, 32});
                    check(parsed.dump().size() > 2 * limit);
                    const auto admitted = admit_chat(parsed, "m", ToolEnumValidation::strict, ToolArgValidation::strict, context_length);
                    check(admitted.maximum_tokens == 8192);
                }
            }
            const auto built = [](std::string content) {
                return Json(Json::Object{
                    {"model", Json("m")},
                    {"messages", Json(Json::Array{Json(Json::Object{
                        {"role", Json("user")},
                        {"content", Json(std::move(content))},
                    })})},
                });
            };
            const size_t limit = 1024 * 1024;
            const auto fits_as_sent = built(std::string(limit / 2 - 64, '\n'));
            const auto too_long = built(std::string(limit, 'a'));
            const auto long_escapes = built(std::string(limit / 6, '\x01'));
            const auto beyond_any_dump = built(std::string(3 * limit, 'a'));
            const auto invalid = built("\xc0\x80");
            check(fits_as_sent.dump().size() > 2 * limit);
            check(admit_chat(fits_as_sent, "m").maximum_tokens == 8192);
            const auto expect_reason = [&](const Json& candidate, std::string_view expected) {
                ++checks;
                try {
                    static_cast<void>(admit_chat(candidate, "m"));
                } catch (const FormatError& error) {
                    check(error.what() == expected);
                    return;
                }
                throw std::runtime_error("oversized or invalid built request was admitted");
            };
            expect_reason(too_long, "chat request exceeds byte limit");
            expect_reason(long_escapes, "chat request exceeds byte limit");
            expect_reason(beyond_any_dump, "JSON output byte limit exceeded");
            expect_reason(invalid, "invalid UTF-8 input");
        }
        {
            struct TempDir {
                std::filesystem::path path;
                int fd = -1;
                explicit TempDir(std::string_view label) {
                    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    path = std::filesystem::temp_directory_path() /
                        ("tokspur-" + std::string(label) + "-" + std::to_string(::getpid()) + "-" + std::to_string(nanos));
                    check(::mkdir(path.c_str(), 0700) == 0);
                    fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
                    check(fd >= 0);
                }
                ~TempDir() {
                    if (fd >= 0) ::close(fd);
                    std::error_code ec;
                    std::filesystem::remove_all(path, ec);
                }
            };
            const auto read_file_bytes = [](const std::filesystem::path& file) {
                std::ifstream in(file, std::ios::binary);
                return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            };
            {
                TempDir dir("prefix-trace-full-cpp");
                PrefixTraceLog trace;
                trace.enable(dir.fd);
                const Json record(Json::Object{{"event", Json("prompt_prefix")}});
                trace.write(record);
                const uint64_t line = trace.bytes_written();
                trace.set_bytes_written_for_test(prefix_trace_limit_bytes - line + 1);
                for (int attempt = 0; attempt < 2; ++attempt) {
                    if (attempt == 1) trace.set_bytes_written_for_test(line);
                    bool caught = false;
                    try {
                        trace.write(record);
                    } catch (const PrefixTraceError& error) {
                        caught = true;
                        check(error.what() == prefix_trace_full_message);
                        for (const bool started : {false, true}) {
                            check(job_failure_kind(error, started) == "server_error");
                            check(job_failure_status(error, false, started) == 500);
                        }
                        check(job_failure_status(error, true, false) == 408);
                    }
                    check(caught);
                }
                check(read_file_bytes(dir.path / "prefix-trace.jsonl") == "{\"event\":\"prompt_prefix\"}\n");

                const PrefixTraceError persist("cannot persist private prefix trace");
                check(job_failure_kind(persist, false) == "server_error");
                check(job_failure_status(persist, false, false) == 500);
                const FormatError refused("missing required API field");
                const WorkerHostMemoryPressure pressure;
                for (bool started : {false, true}) {
                    check(job_failure_kind(pressure, started) == "host_memory_pressure");
                    check(job_failure_status(pressure, false, started) == 503);
                    check(job_failure_status(pressure, true, started) == 408);
                }
                check(job_failure_kind(refused, false) == "invalid_request_error");
                check(job_failure_status(refused, false, false) == 400);
                check(job_failure_kind(refused, true) == "generation_failed_no_retry");
                check(job_failure_status(refused, false, true) == 502);
                check(job_failure_status(refused, true, true) == 408);
            }
            {
                TempDir dir("prefix-trace-exists-cpp");
                const Json record(Json::Object{{"event", Json("prompt_prefix")}});
                {
                    PrefixTraceLog earlier;
                    earlier.enable(dir.fd);
                    earlier.write(record);
                }
                PrefixTraceLog restarted;
                bool startup_refused = false;
                try {
                    restarted.enable(dir.fd);
                } catch (const FormatError& error) {
                    startup_refused = true;
                    check(std::string_view(error.what()) ==
                        "private prefix trace requires a new file: move or remove prefix-trace.jsonl from the state directory, or start without --private-prefix-trace");
                }
                check(startup_refused);
                check(read_file_bytes(dir.path / "prefix-trace.jsonl") == "{\"event\":\"prompt_prefix\"}\n");
                bool write_refused = false;
                try {
                    restarted.write(record);
                } catch (const PrefixTraceError& error) {
                    write_refused = true;
                    check(error.what() == prefix_trace_unavailable_message);
                    check(job_failure_status(error, false, false) == 500);
                }
                check(write_refused);
            }
        }
        std::cout << "PASS owned serving protocol checks=" << checks << " (model-free; no tools executed)\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
