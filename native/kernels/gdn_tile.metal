#include <metal_stdlib>
using namespace metal;

bool gdn_tile_count(uint capacity, uint steps) {
    return (capacity == 2 || capacity == 4 || capacity == 8 || capacity == 16 || capacity == 32
#ifdef TOKSPUR_PRIVATE_PREFILL_CHUNK
        || capacity == 512
#endif
        ) && steps && steps <= capacity;
}

// Actual GDN convolution only: kernel four, dilation one. A channel owns its
// three chronological histories and four coefficients. No state publication.
kernel void graph_gdn_convolution_tile(device const float* history [[buffer(0)]],
                                       device const float* projected [[buffer(1)]],
                                       device const float* weights [[buffer(2)]],
                                       device float* next_history [[buffer(3)]],
                                       device float* output [[buffer(4)]],
                                       constant uint4& s [[buffer(5)]],
                                       device atomic_uint* error [[buffer(7)]],
                                       uint channel [[thread_position_in_grid]]) {
    if (!s.x || s.x > 65536 || !gdn_tile_count(s.y, s.z)) {
        if (!channel) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
    }
    if (channel >= s.x) return;
    float h0 = history[channel], h1 = history[s.x + channel], h2 = history[2 * s.x + channel];
    const float w0 = weights[4 * channel], w1 = weights[4 * channel + 1], w2 = weights[4 * channel + 2], w3 = weights[4 * channel + 3];
    if (!isfinite(h0) || !isfinite(h1) || !isfinite(h2) || !isfinite(w0) || !isfinite(w1) || !isfinite(w2) || !isfinite(w3)) {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return;
    }
    for (uint step = 0; step < s.z; ++step) {
        const float x = projected[ulong(step) * s.x + channel];
        if (!isfinite(x)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        // Same current-product-first sequence as graph_convolution. Do not
        // rearrange tap order or fuse the following SiLU/normalization stages.
        float result = x * w3;
        result += h0 * w0; result += h1 * w1; result += h2 * w2;
        if (!isfinite(result)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        output[ulong(step) * s.x + channel] = result;
        h0 = h1; h1 = h2; h2 = x;
    }
    next_history[channel] = h0; next_history[s.x + channel] = h1; next_history[2 * s.x + channel] = h2;
}

#if defined(TOKSPUR_PRIVATE_GDN_COOPERATIVE4) && TOKSPUR_PRIVATE_GDN_COOPERATIVE4
// Four lanes own 32 keys apiece for
// one value column. Broadcast operands, then retain the ascending scalar
// accumulation and original expressions in every participating lane.
kernel void graph_gdn_delta_tile(device const float* accepted [[buffer(0)]],
                                device const float* query [[buffer(1)]],
                                device const float* key [[buffer(2)]],
                                device const float* value [[buffer(3)]],
                                device const float* decay [[buffer(4)]],
                                device const float* beta [[buffer(5)]],
                                device float* candidate [[buffer(6)]],
                                device atomic_uint* error [[buffer(7)]],
                                device float* output [[buffer(8)]],
                                constant uint4& s [[buffer(9)]],
                                constant uint& steps [[buffer(10)]],
                                uint i [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simd_width [[threads_per_simdgroup]]) {
    if (!s.x || s.x > 48 || !s.y || s.y > 128 || s.w != 128 ||
        !gdn_tile_count(s.z, steps) || simd_width != 32) {
        if (!i) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
    }
    const uint columns_per_head = (s.y + 7) / 8;
    const uint simd_index = i / 32;
    const uint head = simd_index / columns_per_head;
    if (head >= s.x) return; // Uniform for the complete SIMD group.
    const uint column_lane = lane % 8, segment = lane / 8;
    const uint column = (simd_index % columns_per_head) * 8 + column_lane;
    const bool active = column < s.y;
    const uint base = head * 128 * s.y + column;
    const uint value_index = head * s.y + column;
    float state[32]; bool invalid = false;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32; ++j) {
        state[j] = active ? accepted[base + (segment * 32 + j) * s.y] : 0.0f;
        invalid = invalid || !isfinite(state[j]);
    }
    if (simd_any(invalid)) {
        if (invalid) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        return;
    }
    for (uint step = 0; step < steps; ++step) {
        const uint gate = step * s.x + head, qbase = gate * 128;
        const float a = decay[gate], b = beta[gate];
        if (!isfinite(a) || !isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1) {
            atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
        }
        float predicted = 0;
        for (uint part = 0; part < 4; ++part) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 32; ++j) {
                const float prior = simd_shuffle(state[j], column_lane + part * 8);
                predicted += (a * prior) * key[qbase + part * 32 + j];
            }
        }
        const float delta = (active ? value[ulong(step) * s.x * s.y + value_index] : 0.0f) - predicted;
        invalid = false;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) {
            const uint k = segment * 32 + j;
            const float updated = a * state[j] + b * key[qbase + k] * delta;
            state[j] = updated; invalid = invalid || !isfinite(updated);
        }
        if (simd_any(invalid)) {
            if (invalid) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            return;
        }
        float result = 0;
        for (uint part = 0; part < 4; ++part) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 32; ++j) {
                const float updated = simd_shuffle(state[j], column_lane + part * 8);
                result += updated * query[qbase + part * 32 + j];
            }
        }
        if (simd_any(!isfinite(result))) {
            if (!isfinite(result)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            return;
        }
        if (active && segment == 0) output[ulong(step) * s.x * s.y + value_index] = result;
    }
    if (active) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) candidate[base + (segment * 32 + j) * s.y] = state[j];
    }
}
#else
// Fixed key width 128; one thread per (head,value column), with coalesced value
// lanes as graph_delta. Partition the 128-key state across 68 unrolled scalar
// registers and 60 conflict-free threadgroup SRAM slots to avoid stack spills
// while preserving the exact chronological 0..127 accumulation and update order.
template <uint KReg, uint KSram, uint MaxThr>
static inline void gdn_delta_tile_impl(device const float* accepted,
                                       device const float* query,
                                       device const float* key,
                                       device const float* value,
                                       device const float* decay,
                                       device const float* beta,
                                       device float* candidate,
                                       device atomic_uint* error,
                                       device float* output,
                                       constant uint4& s,
                                       uint steps,
                                       uint i,
                                       ushort tid,
                                       threadgroup float (*state_hi)[MaxThr]) {
    // s={heads,value_width,capacity,128}; steps may be a strict prefix for replay.
    if (!s.x || s.x > 48 || !s.y || s.y > 128 || s.w != 128 || !gdn_tile_count(s.z, steps) || tid >= MaxThr) {
        if (!i) atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
    }
    if (i >= s.x * s.y) return;
    const uint head = i / s.y, column = i % s.y, base = head * 128 * s.y + column;
    float state_lo[KReg];
    bool invalid = false;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < KReg; ++k) {
        const float v = accepted[base + k * s.y];
        state_lo[k] = v;
        invalid = invalid || !isfinite(v);
    }
    #pragma clang loop unroll_count(4)
    for (uint k = 0; k < KSram; ++k) {
        const float v = accepted[base + (KReg + k) * s.y];
        state_hi[k][tid] = v;
        invalid = invalid || !isfinite(v);
    }
    if (invalid) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
    for (uint step = 0; step < steps; ++step) {
        const uint gate = step * s.x + head, qbase = gate * 128;
        const float a = decay[gate], b = beta[gate];
        if (!isfinite(a) || !isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1) {
            atomic_fetch_or_explicit(error, 2u, memory_order_relaxed); return;
        }
        device const packed_float4* k4_ptr = reinterpret_cast<device const packed_float4*>(key + qbase);
        device const packed_float4* q4_ptr = reinterpret_cast<device const packed_float4*>(query + qbase);
        float predicted = 0;
        #pragma clang loop unroll(full)
        for (uint k4 = 0; k4 < KReg / 4; ++k4) {
            const float4 kv = float4(k4_ptr[k4]);
            predicted += (a * state_lo[k4 * 4 + 0]) * kv.x;
            predicted += (a * state_lo[k4 * 4 + 1]) * kv.y;
            predicted += (a * state_lo[k4 * 4 + 2]) * kv.z;
            predicted += (a * state_lo[k4 * 4 + 3]) * kv.w;
        }
        #pragma clang loop unroll_count(2)
        for (uint k4 = 0; k4 < KSram / 4; ++k4) {
            const float4 kv = float4(k4_ptr[KReg / 4 + k4]);
            const uint kh = k4 * 4;
            predicted += (a * state_hi[kh + 0][tid]) * kv.x;
            predicted += (a * state_hi[kh + 1][tid]) * kv.y;
            predicted += (a * state_hi[kh + 2][tid]) * kv.z;
            predicted += (a * state_hi[kh + 3][tid]) * kv.w;
        }
        const float delta = value[ulong(step) * s.x * s.y + i] - predicted;
        float result = 0;
        #pragma clang loop unroll(full)
        for (uint k4 = 0; k4 < KReg / 4; ++k4) {
            const float4 kv = float4(k4_ptr[k4]);
            const float4 qv = float4(q4_ptr[k4]);
            const float u0 = a * state_lo[k4 * 4 + 0] + b * kv.x * delta;
            const float u1 = a * state_lo[k4 * 4 + 1] + b * kv.y * delta;
            const float u2 = a * state_lo[k4 * 4 + 2] + b * kv.z * delta;
            const float u3 = a * state_lo[k4 * 4 + 3] + b * kv.w * delta;
            state_lo[k4 * 4 + 0] = u0; result += u0 * qv.x;
            state_lo[k4 * 4 + 1] = u1; result += u1 * qv.y;
            state_lo[k4 * 4 + 2] = u2; result += u2 * qv.z;
            state_lo[k4 * 4 + 3] = u3; result += u3 * qv.w;
        }
        #pragma clang loop unroll_count(2)
        for (uint k4 = 0; k4 < KSram / 4; ++k4) {
            const float4 kv = float4(k4_ptr[KReg / 4 + k4]);
            const float4 qv = float4(q4_ptr[KReg / 4 + k4]);
            const uint kh = k4 * 4;
            const float u0 = a * state_hi[kh + 0][tid] + b * kv.x * delta;
            const float u1 = a * state_hi[kh + 1][tid] + b * kv.y * delta;
            const float u2 = a * state_hi[kh + 2][tid] + b * kv.z * delta;
            const float u3 = a * state_hi[kh + 3][tid] + b * kv.w * delta;
            state_hi[kh + 0][tid] = u0; result += u0 * qv.x;
            state_hi[kh + 1][tid] = u1; result += u1 * qv.y;
            state_hi[kh + 2][tid] = u2; result += u2 * qv.z;
            state_hi[kh + 3][tid] = u3; result += u3 * qv.w;
        }
        if (!isfinite(result)) { atomic_fetch_or_explicit(error, 1u, memory_order_relaxed); return; }
        output[ulong(step) * s.x * s.y + i] = result;
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k < KReg; ++k) candidate[base + k * s.y] = state_lo[k];
    #pragma clang loop unroll_count(4)
    for (uint k = 0; k < KSram; ++k) candidate[base + (KReg + k) * s.y] = state_hi[k][tid];
}

kernel void graph_gdn_delta_tile(device const float* accepted [[buffer(0)]],
                                device const float* query [[buffer(1)]],
                                device const float* key [[buffer(2)]],
                                device const float* value [[buffer(3)]],
                                device const float* decay [[buffer(4)]],
                                device const float* beta [[buffer(5)]],
                                device float* candidate [[buffer(6)]],
                                device atomic_uint* error [[buffer(7)]],
                                device float* output [[buffer(8)]],
                                constant uint4& s [[buffer(9)]],
                                constant uint& steps [[buffer(10)]],
                                uint i [[thread_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float state_hi[88][32];
    gdn_delta_tile_impl<40, 88, 32>(accepted, query, key, value, decay, beta, candidate, error, output, s, steps, i, tid, state_hi);
}

kernel void graph_gdn_delta_tile_w128(device const float* accepted [[buffer(0)]],
                                      device const float* query [[buffer(1)]],
                                      device const float* key [[buffer(2)]],
                                      device const float* value [[buffer(3)]],
                                      device const float* decay [[buffer(4)]],
                                      device const float* beta [[buffer(5)]],
                                      device float* candidate [[buffer(6)]],
                                      device atomic_uint* error [[buffer(7)]],
                                      device float* output [[buffer(8)]],
                                      constant uint4& s [[buffer(9)]],
                                      constant uint& steps [[buffer(10)]],
                                      uint i [[thread_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float state_hi[60][128];
    gdn_delta_tile_impl<68, 60, 128>(accepted, query, key, value, decay, beta, candidate, error, output, s, steps, i, tid, state_hi);
}
#endif
