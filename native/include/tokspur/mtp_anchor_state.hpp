#pragma once
#include "tokspur/measured_controller.hpp"

#include "tokspur/gguf.hpp"
#include "tokspur/sampled_path.hpp"
#include "tokspur/json.hpp"

#include <algorithm>
#include <array>
#include <span>

#include <cmath>
#include <limits>
#include <optional>

namespace tokspur {

struct AnchorEntropyMetrics {
    double entropy_nats = 0.0;
    double normalized_entropy = 0.0;
    double top1_prob = 1.0;
    double top2_prob_margin = 1.0;
    double top2_logit_margin = 0.0;
    double top4_logit_margin = 0.0;
    uint32_t topk_count = 0;
    bool valid = false;
    bool operator==(const AnchorEntropyMetrics&) const = default;
};

[[nodiscard]] inline AnchorEntropyMetrics compute_anchor_entropy_metrics(
    const GpuLogitTopKRow& row, SamplingPolicy policy = {}) {
    if (row.vocabulary == 0 || row.vocabulary > maximum_sampling_vocabulary)
        throw FormatError("anchor entropy sampling vocabulary limit");
    if ((row.first_nonfinite_col != 0xffffffffu && row.first_nonfinite_col >= row.vocabulary) ||
        (row.first_subnormal_col != 0xffffffffu && row.first_subnormal_col >= row.vocabulary)) {
        throw FormatError("corrupted GPU top-k bad-logit index");
    }
    if (row.first_nonfinite_col != 0xffffffffu || row.first_subnormal_col != 0xffffffffu) {
        if (row.first_nonfinite_col <= row.first_subnormal_col)
            throw FormatError("nonfinite sampling logit");
        throw FormatError("subnormal sampling logit");
    }
    if (!std::isfinite(policy.temperature) || policy.temperature < 0.0)
        throw FormatError("invalid sampling temperature for anchor entropy");
    const uint32_t expected = std::min<uint32_t>(gpu_logit_topk_capacity, row.vocabulary);
    if (row.count != expected)
        throw FormatError("incomplete GPU top-k summary");
    const auto canon = [](float val) noexcept -> float { return val == 0.0f ? 0.0f : val; };
    for (uint32_t k = 0; k < expected; ++k) {
        const uint32_t id = row.ids[k];
        const float val = row.logits[k];
        if (id >= row.vocabulary || !std::isfinite(val) || std::fpclassify(val) == FP_SUBNORMAL)
            throw FormatError("invalid GPU top-k candidate entry");
        if (k > 0) {
            const float prev = canon(row.logits[k - 1]);
            const float cur = canon(val);
            const uint32_t prev_id = row.ids[k - 1];
            if (!(prev > cur || (prev == cur && prev_id < id)))
                throw FormatError("unsorted or duplicate GPU top-k candidate entry");
        }
    }

    const double t_eff = (policy.temperature > 0.0) ? std::max(policy.temperature, 1.0e-3) : 1.0;
    const double z0 = double(canon(row.logits[0]));
    const double d12 = (expected >= 2u)
        ? (z0 - double(canon(row.logits[1])))
        : std::numeric_limits<double>::infinity();
    const uint32_t k4 = std::min<uint32_t>(expected, 4u);
    const double d14 = (expected >= 2u)
        ? (z0 - double(canon(row.logits[k4 - 1u])))
        : std::numeric_limits<double>::infinity();

    double z_sum = 1.0;
    double weighted_diff = 0.0;
    double w1 = 0.0;
    for (uint32_t i = 1u; i < expected; ++i) {
        const double d_i = (z0 - double(canon(row.logits[i]))) / t_eff;
        const double w_i = std::exp(-d_i);
        if (i == 1u) w1 = w_i;
        z_sum += w_i;
        if (w_i > 0.0) weighted_diff += w_i * d_i;
    }
    const double p1 = 1.0 / z_sum;
    const double p2_margin = (expected >= 2u) ? ((1.0 - w1) / z_sum) : 1.0;
    const double h_nats = std::max(0.0, std::log(z_sum) + (weighted_diff / z_sum));
    const double h_norm = (expected >= 2u) ? std::min(1.0, h_nats / std::log(double(expected))) : 0.0;
    return AnchorEntropyMetrics{
        .entropy_nats = h_nats,
        .normalized_entropy = h_norm,
        .top1_prob = p1,
        .top2_prob_margin = p2_margin,
        .top2_logit_margin = d12,
        .top4_logit_margin = d14,
        .topk_count = expected,
        .valid = true,
    };
}

[[nodiscard]] inline AnchorEntropyMetrics compute_anchor_entropy_metrics(
    std::span<const float> logits, SamplingPolicy policy = {}) {
    return compute_anchor_entropy_metrics(cpu_logit_topk_row(logits), policy);
}

struct AnchorWidthGate {
    uint32_t max_mtp_depth = 3;
    uint32_t max_dflash2_verify_rows = 8;
    bool allow_lookup = true;
    uint32_t max_lookup_width = 7;
};

[[nodiscard]] inline AnchorWidthGate decide_anchor_width_gate(
    const AnchorEntropyMetrics& m, uint32_t configured_mtp_depth, bool lookup_enabled,
    uint32_t max_dflash2_rows = 8u) noexcept {
    const uint32_t full_lookup_w = (configured_mtp_depth == 7u) ? 7u : std::clamp(configured_mtp_depth, 1u, 3u);
    const uint32_t full_mtp_d = std::clamp(configured_mtp_depth, 1u, 3u);
    const uint32_t full_dflash_rows = std::clamp(max_dflash2_rows, 2u, 16u);
    AnchorWidthGate gate{full_mtp_d, full_dflash_rows, lookup_enabled, full_lookup_w};
    if (!m.valid) return gate;

    const bool high_certainty = m.entropy_nats <= 0.85 && (m.top2_logit_margin >= 1.50 || m.top1_prob >= 0.70);
    if (high_certainty) {
        return gate;
    }

    const bool high_entropy = m.entropy_nats > 1.85 || (m.top2_logit_margin < 0.50 && m.entropy_nats > 1.25);
    if (high_entropy) {
        gate.allow_lookup = false;
        gate.max_lookup_width = 0u;
        gate.max_mtp_depth = 1u;
        gate.max_dflash2_verify_rows = (m.entropy_nats > 2.40 && m.top2_logit_margin < 0.40) ? 2u : 3u;
        return gate;
    }

    gate.allow_lookup = lookup_enabled && (m.top2_logit_margin >= 0.50 && m.entropy_nats <= 1.50);
    gate.max_lookup_width = gate.allow_lookup ? ((configured_mtp_depth == 7u) ? 4u : std::min(full_lookup_w, 2u)) : 0u;
    gate.max_mtp_depth = (m.entropy_nats > 1.25 || m.top2_logit_margin < 1.0) ? std::min(full_mtp_d, 2u) : full_mtp_d;
    gate.max_dflash2_verify_rows = 4u;
    return gate;
}

// Request-local controller state only. A carried terminal is sampled already,
// but remains invisible and is never counted in the consumed target prefix.
struct MtpAnchorPublication {
    uint32_t prefix = 0, emitted = 0;
    bool stopped = false;
    std::optional<SampledPathToken> pending;
    std::optional<AnchorEntropyMetrics> anchor_metrics{};
};

inline MtpAnchorPublication prepare_mtp_anchor_publication(uint32_t base, const SampledPathDecision& decision, uint32_t context_limit = 65536) {
    if ((context_limit != 65536 && context_limit != 131072) || base >= context_limit || decision.identity.prefix != base + 1 || decision.accepted > 1 || !decision.count ||
        decision.count > 2)
        throw FormatError("invalid MTP anchor publication decision");
    const auto& terminal = decision.tokens[decision.count - 1];
    if (decision.tokens[0].accepted_proposal != bool(decision.accepted) ||
        terminal.id >= decision.identity.vocabulary)
        throw FormatError("MTP anchor proposal/terminal identity differs from acceptance");
    const bool accepted_stop = decision.accepted && decision.stopped && terminal.accepted_proposal;
    if (decision.count != (decision.accepted && !accepted_stop ? 2u : 1u))
        throw FormatError("MTP anchor decision count differs from its terminal kind");
    // A bonus may itself be EOS; unlike an accepted draft EOS it is not yet
    // consumed, so its terminal status belongs to its later materialisation.
    if (decision.accepted && decision.count == 1 && !accepted_stop)
        throw FormatError("MTP anchor lost accepted proposal/terminal identity");
    if (!std::isfinite(terminal.source_logit) || (!accepted_stop && terminal.accepted_proposal))
        throw FormatError("MTP anchor terminal is not a correction or bonus");
    const uint32_t count = 1 + decision.accepted;
    if (count > context_limit - base) throw FormatError("MTP anchor consumed prefix exceeds context");
    return {base + count, count, accepted_stop,
        accepted_stop ? std::optional<SampledPathToken>{} : std::optional<SampledPathToken>{terminal}};
}

#ifdef TOKSPUR_PRIVATE_NATIVE_MTP_DEPTH2
inline MtpAnchorPublication prepare_mtp_anchor_publication2(uint32_t base, const SampledPathDecision& decision, uint32_t context_limit = 65536) {
    if ((context_limit != 65536 && context_limit != 131072) || base >= context_limit || decision.identity.prefix != base + 1 ||
        decision.active_proposals < 1 || decision.active_proposals > 3 ||
        decision.accepted > decision.active_proposals || !decision.count ||
        decision.count > 1 + decision.active_proposals)
        throw FormatError("invalid depth-two MTP anchor decision");
    const bool accepted_stop = decision.terminal == SampledPathTerminal::accepted_stop;
    if (decision.count != decision.accepted + (accepted_stop ? 0u : 1u) ||
        (accepted_stop && (!decision.stopped || !decision.accepted)))
        throw FormatError("depth-two MTP terminal count/kind differs");
    for (uint32_t i = 0; i < decision.count; ++i) {
        const auto& token = decision.tokens[i];
        if (token.accepted_proposal != (i < decision.accepted) || token.id >= decision.identity.vocabulary ||
            !std::isfinite(token.source_logit)) throw FormatError("depth-two MTP token identity differs");
    }
    const auto count = 1 + decision.accepted;
    if (count > context_limit - base) throw FormatError("depth-two MTP consumed prefix exceeds context");
    return {base + count, count, accepted_stop, accepted_stop ? std::optional<SampledPathToken>{} :
        std::optional<SampledPathToken>{decision.tokens[decision.count - 1]}};
}
#endif

#ifdef TOKSPUR_PRIVATE_VERIFY_EIGHT
inline MtpAnchorPublication prepare_mtp_anchor_publication8(uint32_t base, const SampledPathDecision& decision, uint32_t context_limit = 65536) {
    if ((context_limit != 65536 && context_limit != 131072) || base >= context_limit || decision.identity.prefix != base + 1 ||
        decision.active_proposals < 1 || decision.active_proposals > 7 ||
        decision.accepted > decision.active_proposals || !decision.count ||
        decision.count > 1 + decision.active_proposals)
        throw FormatError("invalid bounded-eight anchor decision");
    const bool accepted_stop = decision.terminal == SampledPathTerminal::accepted_stop;
    if (decision.count != decision.accepted + (accepted_stop ? 0u : 1u) ||
        (accepted_stop && (!decision.stopped || !decision.accepted)))
        throw FormatError("bounded-eight terminal count/kind differs");
    for (uint32_t i = 0; i < decision.count; ++i) {
        const auto& token = decision.tokens[i];
        if (token.accepted_proposal != (i < decision.accepted) || token.id >= decision.identity.vocabulary ||
            !std::isfinite(token.source_logit)) throw FormatError("bounded-eight token identity differs");
    }
    const auto count = 1 + decision.accepted;
    if (count > context_limit - base) throw FormatError("bounded-eight consumed prefix exceeds context");
    return {base + count, count, accepted_stop, accepted_stop ? std::optional<SampledPathToken>{} :
        std::optional<SampledPathToken>{decision.tokens[decision.count - 1]}};
}

inline MtpAnchorPublication prepare_mtp_anchor_publication16(uint32_t base, const SampledPathDecision& decision, uint32_t context_limit = 65536) {
    if ((context_limit != 65536 && context_limit != 131072) || base >= context_limit || decision.identity.prefix != base + 1 ||
        decision.active_proposals < 1 || decision.active_proposals > 15 ||
        decision.accepted > decision.active_proposals || !decision.count ||
        decision.count > 1 + decision.active_proposals)
        throw FormatError("invalid bounded-sixteen anchor decision");
    const bool accepted_stop = decision.terminal == SampledPathTerminal::accepted_stop;
    if (decision.count != decision.accepted + (accepted_stop ? 0u : 1u) ||
        (accepted_stop && (!decision.stopped || !decision.accepted)))
        throw FormatError("bounded-sixteen terminal count/kind differs");
    for (uint32_t i = 0; i < decision.count; ++i) {
        const auto& token = decision.tokens[i];
        if (token.accepted_proposal != (i < decision.accepted) || token.id >= decision.identity.vocabulary ||
            !std::isfinite(token.source_logit)) throw FormatError("bounded-sixteen token identity differs");
    }
    const auto count = 1 + decision.accepted;
    if (count > context_limit - base) throw FormatError("bounded-sixteen consumed prefix exceeds context");
    return {base + count, count, accepted_stop, accepted_stop ? std::optional<SampledPathToken>{} :
        std::optional<SampledPathToken>{decision.tokens[decision.count - 1]}};
}
#endif

class MtpAnchorCarry {
public:
    [[nodiscard]] const SampledPathToken* peek(uint32_t prefix) const {
        if (pending_ && prefix != prefix_) throw FormatError("stale MTP carried-anchor prefix");
        return pending_ ? &*pending_ : nullptr;
    }
    [[nodiscard]] const AnchorEntropyMetrics* peek_metrics(uint32_t prefix) const {
        if (pending_ && prefix != prefix_) throw FormatError("stale MTP carried-anchor prefix");
        return (pending_ && metrics_) ? &*metrics_ : nullptr;
    }
    void publish(const MtpAnchorPublication& publication) noexcept {
        prefix_ = publication.prefix; pending_ = publication.pending;
        metrics_ = publication.pending ? publication.anchor_metrics : std::nullopt;
    }
    // An output-budget boundary may discard an invisible terminal. Its RNG
    // words remain consumed by the caller's already-published decision RNG.
    [[nodiscard]] bool clear() noexcept { const bool had = bool(pending_); pending_.reset(); metrics_.reset(); return had; }
private:
    uint32_t prefix_ = 0;
    std::optional<SampledPathToken> pending_;
    std::optional<AnchorEntropyMetrics> metrics_;
};

// Explicit HTTP-generation lifetime, separate from the legacy packet-local
// algorithm. Invisible terminals retain their original RNG draw and score.
// Frozen bytes: the SHA-256 of the cost table as first recorded, with k3 extrapolated to 90498. Checkpoint names and the wire default carry them, so they do not follow profiles/cost-table.json, which records the compiled costs below.
inline constexpr std::array<uint8_t, 32> default_controller_policy_digest = {
    0xa9, 0x72, 0x87, 0x4d, 0x5a, 0xdb, 0x82, 0xb0,
    0x2e, 0xea, 0x0d, 0x7d, 0xa5, 0xd5, 0x24, 0x2a,
    0x86, 0xe4, 0xaa, 0xa3, 0xff, 0x16, 0x6d, 0xc3,
    0xab, 0x58, 0x19, 0xdb, 0x7f, 0x2c, 0xbf, 0x5a
};

// Immutable compatibility bytes in lookup and model-state identities. They do not describe new controller policies.
inline constexpr auto legacy_checkpoint_identity_tag = default_controller_policy_digest;

enum class StageAMode : uint8_t { Speculative = 0, OrdinaryK0 = 1 };

struct StageAControllerState {
    double ewma_alpha1 = 0.45;
    uint8_t current_mode = 0;
    uint8_t consecutive_ordinary_steps = 0;
    std::array<uint8_t, 32> controller_policy_digest = default_controller_policy_digest;
    std::optional<MeasuredObservation> measured;
    bool operator==(const StageAControllerState&) const = default;
};

// Version-2 checkpoints cannot prove the sampling/estimator provenance of observations.
inline StageAControllerState restore_controller_policy(StageAControllerState saved,
    const std::array<uint8_t, 32>& active, bool measured) {
    if (measured || saved.controller_policy_digest != active) {
        saved = StageAControllerState{};
        saved.controller_policy_digest = active;
    }
    return saved;
}

struct StageACostTable {
    static constexpr double alpha_0 = 0.45;
    static constexpr double beta = 0.125;
    static constexpr double eta_exit = 0.00;
    static constexpr double eta_enter = 0.04;
    static constexpr uint32_t probe_interval = 16;
    static constexpr std::array<uint8_t, 32> policy_digest = default_controller_policy_digest;

    static constexpr std::array<double, 16> f_cal_lut = {
        0.182, 0.224, 0.278, 0.335, 0.389, 0.441, 0.495, 0.548,
        0.601, 0.655, 0.708, 0.762, 0.811, 0.854, 0.889, 0.915
    };

    static constexpr uint32_t t_round_k1_us = 56664;
    static constexpr uint32_t t_round_k2_us = 73581;
    static constexpr uint32_t t_round_k3_us = 81000;
    static constexpr uint32_t verify_nax_t_round_k1_us = 24500;
    static constexpr uint32_t verify_nax_t_round_k2_us = 25800;
    static constexpr uint32_t verify_nax_t_round_k3_us = 27100;

    static double lookup_alpha2_hat(double alpha1) noexcept {
        if (alpha1 <= 0.0) return f_cal_lut[0];
        if (alpha1 >= 1.0) return f_cal_lut[15];
        double pos = alpha1 * 15.0;
        size_t idx = static_cast<size_t>(pos);
        if (idx >= 15) return f_cal_lut[15];
        double frac = pos - double(idx);
        return (1.0 - frac) * f_cal_lut[idx] + frac * f_cal_lut[idx + 1];
    }

    static uint32_t lookup_t_ord_us(uint32_t context_length, bool verify_nax = false) noexcept {
        if (verify_nax) {
            if (context_length <= 512) return 19800;
            if (context_length <= 1024) return 20100;
            if (context_length <= 2048) return 20500;
            return 21000;
        }
        if (context_length <= 512) return 34430;
        if (context_length <= 1024) return 36324;
        if (context_length <= 2048) return 44652;
        return 44673;
    }

    static double expected_tokens(double alpha1, uint32_t k) noexcept {
        if (k == 0) return 1.0;
        if (k == 1) return 1.0 + alpha1;
        double alpha2 = lookup_alpha2_hat(alpha1);
        if (k == 2) return 1.0 + alpha1 + alpha1 * alpha2;
        double alpha3 = lookup_alpha2_hat(alpha2);
        return 1.0 + alpha1 + alpha1 * alpha2 + alpha1 * alpha2 * alpha3;
    }

    static double expected_speculative_throughput(double alpha1, uint32_t k, bool verify_nax = false) noexcept {
        const uint32_t t1 = verify_nax ? verify_nax_t_round_k1_us : t_round_k1_us;
        const uint32_t t2 = verify_nax ? verify_nax_t_round_k2_us : t_round_k2_us;
        const uint32_t t3 = verify_nax ? verify_nax_t_round_k3_us : t_round_k3_us;
        if (k == 1) return (1.0 + alpha1) / double(t1);
        if (k == 2) {
            double alpha2 = lookup_alpha2_hat(alpha1);
            return (1.0 + alpha1 + alpha1 * alpha2) / double(t2);
        }
        if (k == 3) {
            return expected_tokens(alpha1, 3) / double(t3);
        }
        return 0.0;
    }

    static double max_speculative_throughput(double alpha1, bool verify_nax = false) noexcept {
        const double best12 = std::max(expected_speculative_throughput(alpha1, 1, verify_nax),
                                       expected_speculative_throughput(alpha1, 2, verify_nax));
        return verify_nax ? std::max(best12, expected_speculative_throughput(alpha1, 3, true)) : best12;
    }
};

inline void stage_a_step_ordinary(StageAControllerState& state, uint32_t context_length, bool verify_nax = false) {
    if (state.current_mode != uint8_t(StageAMode::OrdinaryK0)) return;
    const double t_ord = double(StageACostTable::lookup_t_ord_us(context_length, verify_nax));
    const double r_ord = 1.0 / t_ord;
    const double spec_tp = StageACostTable::max_speculative_throughput(state.ewma_alpha1, verify_nax);
    if (spec_tp > (1.0 + StageACostTable::eta_enter) * r_ord) {
        state.current_mode = uint8_t(StageAMode::Speculative);
        state.consecutive_ordinary_steps = 0;
        return;
    }
    if (uint32_t(state.consecutive_ordinary_steps) + 1 < StageACostTable::probe_interval) {
        state.consecutive_ordinary_steps += 1;
    }
}

inline void stage_a_commit_speculative(StageAControllerState& state, double y1, uint32_t context_length, bool verify_nax = false) {
    state.ewma_alpha1 = (1.0 - StageACostTable::beta) * state.ewma_alpha1 + StageACostTable::beta * y1;
    state.consecutive_ordinary_steps = 0;
    const double t_ord = double(StageACostTable::lookup_t_ord_us(context_length, verify_nax));
    const double r_ord = 1.0 / t_ord;
    const double spec_tp = StageACostTable::max_speculative_throughput(state.ewma_alpha1, verify_nax);
    if (spec_tp < (1.0 + StageACostTable::eta_exit) * r_ord) {
        state.current_mode = uint8_t(StageAMode::OrdinaryK0);
        state.consecutive_ordinary_steps = 0;
    }
}

inline void stage_a_commit_probe(StageAControllerState& state, double y1, uint32_t context_length, bool verify_nax = false) {
    state.ewma_alpha1 = (1.0 - StageACostTable::beta) * state.ewma_alpha1 + StageACostTable::beta * y1;
    state.consecutive_ordinary_steps = 0;
    const double t_ord = double(StageACostTable::lookup_t_ord_us(context_length, verify_nax));
    const double r_ord = 1.0 / t_ord;
    const double spec_tp = StageACostTable::max_speculative_throughput(state.ewma_alpha1, verify_nax);
    if (spec_tp > (1.0 + StageACostTable::eta_enter) * r_ord) {
        state.current_mode = uint8_t(StageAMode::Speculative);
    }
}

struct MtpGenerationCommand {
    uint64_t id = 0;
    uint32_t remaining = 0;
    bool begin = false, abort = false;
    uint32_t committed = 0;
    uint32_t emitted = 0;
    uint32_t pending = 0;
    uint32_t remaining_compute = 0;
    uint32_t total_budget = 0;
    std::optional<StageAControllerState> stage_a_controller{};
    bool v2 = false;
};
inline MtpGenerationCommand mtp_generation_command(const Json& value) {
    MtpGenerationCommand out;
    if (value.kind() != Json::Kind::object) throw FormatError("MTP generation command must be an object");
    const auto* id = value.find("id");
    if (!id || !(out.id = id->unsigned_integer())) throw FormatError("MTP generation requires a nonzero ID");
    const auto* abort = value.find("abort");
    if (abort) {
        if (!abort->boolean() || value.object().size() != 2) throw FormatError("invalid MTP generation abort");
        out.abort = true;
    } else {
        const auto* begin = value.find("begin"); const auto* remaining = value.find("remaining");
        if (!begin || !remaining) throw FormatError("invalid MTP generation packet");
        out.begin = begin->boolean(); const auto n = remaining->unsigned_integer();
        if (!n || n > 32768) throw FormatError("MTP generation output budget outside 1..32768");
        out.remaining = uint32_t(n);
        out.total_budget = out.remaining;
        out.remaining_compute = out.remaining;
        if (value.object().size() != 3) {
            const auto* committed = value.find("committed");
            const auto* emitted = value.find("emitted");
            const auto* pending = value.find("pending");
            if (!committed || !emitted || !pending) throw FormatError("invalid MTP generation packet");
            for (const auto& [k, v] : value.object()) {
                if (k != "id" && k != "begin" && k != "remaining" && k != "committed" &&
                    k != "emitted" && k != "pending" && k != "remaining_compute" &&
                    k != "total_budget" && k != "stage_a_controller")
                    throw FormatError("unknown MTP generation command field");
            }
            out.v2 = true;
            const auto parse_u32_bounded = [](const Json& v, uint64_t max_val, const char* msg) {
                const auto raw = v.unsigned_integer();
                if (raw > max_val) throw FormatError(msg);
                return uint32_t(raw);
            };
            out.committed = parse_u32_bounded(*committed, 131072, "MTP generation committed bound exceeded");
            out.emitted = parse_u32_bounded(*emitted, 131072, "MTP generation emitted bound exceeded");
            out.pending = parse_u32_bounded(*pending, 131072, "MTP generation pending bound exceeded");
            if (const auto* rc = value.find("remaining_compute")) out.remaining_compute = parse_u32_bounded(*rc, 131072, "MTP generation remaining_compute bound exceeded");
            if (const auto* tb = value.find("total_budget")) out.total_budget = parse_u32_bounded(*tb, 131072, "MTP generation total_budget bound exceeded");
            if (const auto* sc = value.find("stage_a_controller")) {
                if (sc->kind() != Json::Kind::object) throw FormatError("stage_a_controller must be an object");
                StageAControllerState ctrl;
                if (const auto* ea = sc->find("ewma_alpha1")) ctrl.ewma_alpha1 = std::stod(std::string(ea->number_lexeme()));
                if (const auto* cm = sc->find("current_mode")) {
                    const auto mode = cm->unsigned_integer();
                    if (mode > 255) throw FormatError("controller current_mode outside byte range");
                    ctrl.current_mode = uint8_t(mode);
                }
                if (const auto* co = sc->find("consecutive_ordinary_steps")) {
                    const auto steps = co->unsigned_integer();
                    if (steps > 255) throw FormatError("controller consecutive_ordinary_steps outside byte range");
                    ctrl.consecutive_ordinary_steps = uint8_t(steps);
                }
                if (const auto* cpd = sc->find("controller_policy_digest")) {
                    if (cpd->kind() != Json::Kind::array || cpd->array().size() != 32)
                        throw FormatError("controller_policy_digest must be a 32-element array");
                    for (size_t idx = 0; idx < 32; ++idx) {
                        const auto byte = cpd->array()[idx].unsigned_integer();
                        if (byte > 255) throw FormatError("controller policy digest byte outside range");
                        ctrl.controller_policy_digest[idx] = uint8_t(byte);
                    }
                }
                if (const auto* m = sc->find("measured")) ctrl.measured = MeasuredObservation::parse(*m);
                out.stage_a_controller = ctrl;
            }
        }
    }
    return out;
}

class MtpGenerationCarry {
public:
    struct OverflowToken {
        uint32_t id = 0;
        float score = 0.0f;
        uint32_t token_prefix = 0;
        uint32_t committed_prefix = 0;
    };

    void configure_controller_policy(const std::array<uint8_t, 32>& digest) {
        if (active_) throw FormatError("cannot change an active generation controller policy");
        expected_policy_ = digest;
        controller_state_ = restore_controller_policy(controller_state_, digest, digest != default_controller_policy_digest);
    }
    [[nodiscard]] const std::array<uint8_t, 32>& controller_policy() const noexcept { return expected_policy_; }

    // The caller supplies its startup-negotiated packet limit. Compiled support alone never widens the default-eight transaction.
    void admit(const MtpGenerationCommand& command, uint32_t maximum, uint32_t prefix, std::span<const uint32_t> stops, uint32_t packet_limit = 8) {
        const bool supported_limit = packet_limit == 8
#ifdef TOKSPUR_PRIVATE_MTP_OUTPUT64
            || packet_limit == 64
#endif
            ;
        if (!supported_limit || !command.id || command.abort || !maximum || maximum > packet_limit || command.remaining > 32768 || maximum > command.remaining || stops.size() > stops_.size())
            throw FormatError("invalid MTP generation output transaction");
        if (expected_policy_ != default_controller_policy_digest && !command.v2)
            throw FormatError("measured controllers refuse legacy generation packets");
        if (!command.begin && controller_state_.measured && !command.stage_a_controller)
            throw FormatError("measured controller continuation requires live feedback");
        if (command.stage_a_controller && command.stage_a_controller->controller_policy_digest != expected_policy_)
            throw FormatError("controller policy digest mismatch");
        if (command.begin) {
            if (command.stage_a_controller && command.stage_a_controller->measured) throw FormatError("measured generation must start with fresh observations");
            if (active_ || command.id <= id_) throw FormatError("overlapping or stale MTP generation begin");
            if (carry_.peek(prefix)) throw FormatError("inactive MTP generation retained a terminal");
            id_ = command.id; remaining_ = command.remaining; prefix_ = prefix; base_prefix_ = prefix; active_ = true;
            stop_count_ = stops.size(); std::copy(stops.begin(), stops.end(), stops_.begin());
            v2_ = command.v2;
            total_budget_ = command.total_budget ? command.total_budget : command.remaining;
            committed_ = 0; emitted_ = 0; pending_overflow_ = 0; compute_remaining_ = total_budget_;
            continuation_allowed_ = true; stop_queued_ = false; stop_drained_ = false;
            if (expected_policy_ != default_controller_policy_digest) {
                controller_state_ = StageAControllerState{}; controller_state_.controller_policy_digest = expected_policy_;
            } else if (command.stage_a_controller) controller_state_ = *command.stage_a_controller;
            else { controller_state_ = StageAControllerState{}; controller_state_.controller_policy_digest = expected_policy_; }
        } else if (!active_ || command.id != id_ || command.v2 != v2_ || command.remaining != remaining_ || prefix != prefix_ ||
                   stops.size() != stop_count_ || !std::equal(stops.begin(), stops.end(), stops_.begin()))
            throw FormatError("MTP generation continuation identity/budget/frontier/stops differ");
        else if (command.v2) {
            v2_ = true;
            if (command.committed != committed_ || command.emitted != emitted_ ||
                command.pending != pending_overflow_ || command.remaining_compute != compute_remaining_)
                throw FormatError("MTP generation v2 frontier mismatch on continuation");
            if (command.stage_a_controller) {
                if (command.stage_a_controller->controller_policy_digest != expected_policy_)
                    throw FormatError("controller policy digest mismatch");
                if (controller_state_.measured && *command.stage_a_controller != controller_state_)
                    throw FormatError("measured controller feedback differs from the active generation");
                controller_state_ = *command.stage_a_controller;
            }
        }
        maximum_ = maximum;
    }

    [[nodiscard]] bool is_v2() const noexcept { return v2_; }
    void set_v2(bool v) noexcept { v2_ = v; }
    [[nodiscard]] uint32_t pending_overflow() const noexcept { return pending_overflow_; }
    [[nodiscard]] OverflowToken pop_overflow() {
        if (!pending_overflow_) throw FormatError("no pending overflow to pop");
        OverflowToken token = overflow_tokens_[0];
        for (uint32_t i = 0; i + 1 < pending_overflow_; ++i) {
            overflow_tokens_[i] = overflow_tokens_[i + 1];
        }
        --pending_overflow_;
        ++emitted_;
        return token;
    }

    void commit_round(uint32_t count, std::span<const uint32_t> ids, std::span<const float> scores,
                      uint32_t round_base, uint32_t round_committed,
                      uint32_t emitted_now, bool stopped, uint32_t max_overflow = 2) {
        committed_ += count;
        if (compute_remaining_ >= count) compute_remaining_ -= count;
        else compute_remaining_ = 0;
        emitted_ += emitted_now;
        const uint32_t overflow = count > emitted_now ? count - emitted_now : 0;
        if (overflow > max_overflow || overflow > overflow_tokens_.size())
            throw FormatError("overflow exceeds speculative bound");
        pending_overflow_ = overflow;
        for (uint32_t i = 0; i < overflow; ++i) {
            overflow_tokens_[i] = OverflowToken{
                .id = ids[emitted_now + i],
                .score = scores[emitted_now + i],
                .token_prefix = round_base + emitted_now + i + 1,
                .committed_prefix = round_committed,
            };
        }
        if (stopped) stop_queued_ = true;
    }

    [[nodiscard]] bool finish(uint32_t emitted, uint32_t prefix, bool stopped) {
        if (!active_ || !emitted || emitted > maximum_ || emitted > remaining_ ||
            (!v2_ && prefix != prefix_ + emitted) ||
            (v2_ && prefix != base_prefix_ + committed_) ||
            (!stopped && emitted != maximum_)) throw FormatError("MTP generation publication differs");
        static_cast<void>(carry_.peek(prefix));
        prefix_ = prefix; remaining_ -= emitted;
        return stopped || !remaining_ ? clear() : false;
    }
    // Only the diagnostic worker calls this after exactly one recorded action.
    // Keep ordinary serving's full-packet publication check unchanged.
    [[nodiscard]] bool finish_diagnostic_action(uint32_t emitted, uint32_t prefix, bool stopped) {
        if (!v2_ || !active_ || prefix_ != base_prefix_ || pending_overflow_ ||
            emitted != committed_ || emitted != emitted_ || !emitted || emitted > maximum_)
            throw FormatError("diagnostic action publication frontier differs");
        const auto requested = maximum_;
        maximum_ = emitted;
        try {
            const bool discarded = finish(emitted, prefix, stopped);
            if (active_) maximum_ = requested;
            return discarded;
        } catch (...) {
            maximum_ = requested;
            throw;
        }
    }
    [[nodiscard]] bool abort(const MtpGenerationCommand& command) {
        if (!command.abort || !active_ || command.id != id_) throw FormatError("missing or stale MTP generation abort");
        if (committed_ > emitted_) {
            continuation_allowed_ = false;
        }
        return clear();
    }
    [[nodiscard]] bool clear() noexcept {
        active_ = false; remaining_ = maximum_ = committed_ = emitted_ = pending_overflow_ = compute_remaining_ = base_prefix_ = 0;
        stop_queued_ = stop_drained_ = false;
        return carry_.clear();
    }
    [[nodiscard]] MtpAnchorCarry& carry() noexcept { return carry_; }
    [[nodiscard]] uint64_t id() const noexcept { return id_; }
    [[nodiscard]] uint32_t remaining() const noexcept { return remaining_; }
    [[nodiscard]] uint32_t compute_remaining() const noexcept { return compute_remaining_; }
    [[nodiscard]] uint32_t committed() const noexcept { return committed_; }
    [[nodiscard]] uint32_t emitted() const noexcept { return emitted_; }
    [[nodiscard]] uint32_t total_budget() const noexcept { return total_budget_; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] bool continuation_allowed() const noexcept { return continuation_allowed_; }
    [[nodiscard]] bool stop_queued() const noexcept { return stop_queued_; }
    [[nodiscard]] bool stop_drained() const noexcept { return stop_drained_; }
    void mark_stop_drained() noexcept { stop_drained_ = true; }
    [[nodiscard]] const StageAControllerState& controller_state() const noexcept { return controller_state_; }
    [[nodiscard]] StageAControllerState& controller_state() noexcept { return controller_state_; }

    [[nodiscard]] bool pending(uint32_t prefix) const { return bool(carry_.peek(prefix)); }
    [[nodiscard]] Json metadata(uint32_t prefix) const {
        Json::Object obj{
            {"id", Json::integer(id_)},
            {"active", Json(active_)},
            {"remaining", Json::integer(remaining_)},
            {"pending", Json(pending(prefix))},
        };
        if (v2_) {
            obj.emplace_back("committed", Json::integer(committed_));
            obj.emplace_back("emitted", Json::integer(emitted_));
            obj.emplace_back("overflow", Json::integer(pending_overflow_));
            obj.emplace_back("remaining_compute", Json::integer(compute_remaining_));
            obj.emplace_back("total_budget", Json::integer(total_budget_));
            obj.emplace_back("continuation_allowed", Json(continuation_allowed_));
            obj.emplace_back("stop_queued", Json(stop_queued_));
            obj.emplace_back("stop_drained", Json(stop_drained_));
            Json::Object ctrl{
                {"ewma_alpha1", Json::number(controller_state_.ewma_alpha1)},
                {"current_mode", Json::integer(controller_state_.current_mode)},
                {"consecutive_ordinary_steps", Json::integer(controller_state_.consecutive_ordinary_steps)},
            };
            Json::Array digest;
            for (auto b : controller_state_.controller_policy_digest) digest.push_back(Json::integer(b));
            ctrl.emplace_back("controller_policy_digest", Json(std::move(digest)));
            if (controller_state_.measured) ctrl.emplace_back("measured", controller_state_.measured->json());
            obj.emplace_back("stage_a_controller", Json(std::move(ctrl)));
        }
        return Json(std::move(obj));
    }
private:
    MtpAnchorCarry carry_;
    uint64_t id_ = 0;
    uint32_t remaining_ = 0, maximum_ = 0, prefix_ = 0, base_prefix_ = 0;
    uint32_t committed_ = 0, emitted_ = 0, pending_overflow_ = 0, compute_remaining_ = 0, total_budget_ = 0;
    std::array<OverflowToken, 15> overflow_tokens_{};
    StageAControllerState controller_state_{};
    std::array<uint8_t, 32> expected_policy_ = default_controller_policy_digest;
    std::array<uint32_t, 16> stops_{};
    size_t stop_count_ = 0;
    bool active_ = false;
    bool v2_ = false;
    bool continuation_allowed_ = true;
    bool stop_queued_ = false, stop_drained_ = false;
};

} // namespace tokspur
