// Exact windowed GGUF / q8_lanes_v1 Q8_0 Hyper-Connections down projection for 10240-to-320 prefill rows.
// Preserves the single-SIMD-group 0..319 block accumulation order and simd_sum reduction of graph_hc_q8_exact_r1_u16 under -fno-fast-math.
#include <metal_stdlib>
using namespace metal;

struct HcDownWindowArgs {
    uint columns, rows, tokens, reserved;
    ulong input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(HcDownWindowArgs) == 32, "HC down window argument ABI");

template <ushort W_TOK, ushort R_PER_SG, ushort B_TILE, bool SwizzleL2>
inline void hc_down_q8_window_core(
    device const uchar* weights,
    device const float* input,
    device float* output,
    constant HcDownWindowArgs& p,
    device atomic_uint* error,
    uint2 raw_group,
    ushort subgroup,
    ushort lane,
    ushort tid,
    threadgroup float4* shared_x4
) {
    if (p.columns != 10240 || p.rows != 320 || !p.tokens || p.tokens > 2048 || p.reserved > 1 ||
        p.input_stride_bytes < 40960 || p.output_stride_bytes < 1280 || p.input_stride_bytes % 4 || p.output_stride_bytes % 4) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    constexpr ushort SG_COUNT = 8;
    constexpr ushort THREADS = SG_COUNT * 32;
    constexpr ushort R_TG = SG_COUNT * R_PER_SG;
    constexpr ushort COLS = B_TILE * 32;
    constexpr ushort COLS4 = COLS / 4;
    static_assert(uint(W_TOK) * uint(COLS) * sizeof(float) <= 8192u, "HC down window 8 KiB SRAM budget");
    static_assert(320u % R_TG == 0u, "R_TG must tile 320 rows evenly");
    static_assert(320u % B_TILE == 0u, "B_TILE must tile 320 blocks evenly");

    uint gx_idx = raw_group.x;
    uint gy_idx = raw_group.y;
    if (SwizzleL2) {
        constexpr uint GX = 320u / R_TG;
        const uint gy = (p.tokens + uint(W_TOK) - 1u) / uint(W_TOK);
        if (gx_idx < GX && gy_idx < gy) {
            const uint linear = gy_idx * GX + gx_idx;
            gx_idx = linear / gy;
            gy_idx = linear % gy;
        }
    }

    const uint tg_row0 = gx_idx * R_TG;
    const uint token_base = gy_idx * W_TOK;
    if (tg_row0 >= p.rows || token_base >= p.tokens) return;
    const uint first_row = tg_row0 + subgroup * R_PER_SG;

    const uint in_stride = uint(p.input_stride_bytes / sizeof(float));
    const uint out_stride = uint(p.output_stride_bytes / sizeof(float));
    const uint count = min(uint(W_TOK), p.tokens - token_base);
    device const float* row_in = input + ulong(token_base) * in_stride;
    threadgroup float* shared_x = reinterpret_cast<threadgroup float*>(shared_x4);

    float sums[R_PER_SG][W_TOK];
    #pragma unroll
    for (ushort r = 0; r < R_PER_SG; ++r) {
        #pragma unroll
        for (ushort m = 0; m < W_TOK; ++m) sums[r][m] = 0.0f;
    }

    if (p.reserved == 0u) {
        for (uint base = 0; base < 320u; base += B_TILE) {
            half sc_pre[R_PER_SG][B_TILE];
            char q_pre[R_PER_SG][B_TILE];
            #pragma unroll
            for (ushort r = 0; r < R_PER_SG; ++r) {
                device const uchar* rbase = weights + (ulong(first_row + r) * 320ul + base) * 34ul;
                #pragma unroll
                for (ushort s = 0; s < B_TILE; ++s) {
                    device const uchar* bp = rbase + ulong(s) * 34ul;
                    sc_pre[r][s] = *reinterpret_cast<device const half*>(bp);
                    q_pre[r][s] = *reinterpret_cast<device const char*>(bp + 2ul + lane);
                }
            }

            const uint col_base = base * 32u;
            if (count == W_TOK) {
                #pragma unroll
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4u));
                }
            } else {
                #pragma unroll
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = (m < count)
                        ? float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4u))
                        : float4(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (count == W_TOK) {
                #pragma unroll
                for (ushort s = 0; s < B_TILE; ++s) {
                    const ushort c = s * 32u + lane;
                    float w[R_PER_SG];
                    #pragma unroll
                    for (ushort r = 0; r < R_PER_SG; ++r) {
                        w[r] = float(sc_pre[r][s]) * float(q_pre[r][s]);
                    }
                    #pragma unroll
                    for (ushort m = 0; m < W_TOK; ++m) {
                        const float xv = shared_x[m * COLS + c];
                        #pragma unroll
                        for (ushort r = 0; r < R_PER_SG; ++r) sums[r][m] += w[r] * xv;
                    }
                }
            } else {
                #pragma unroll
                for (ushort s = 0; s < B_TILE; ++s) {
                    const ushort c = s * 32u + lane;
                    float w[R_PER_SG];
                    #pragma unroll
                    for (ushort r = 0; r < R_PER_SG; ++r) {
                        w[r] = float(sc_pre[r][s]) * float(q_pre[r][s]);
                    }
                    for (ushort m = 0; m < count; ++m) {
                        const float xv = shared_x[m * COLS + c];
                        #pragma unroll
                        for (ushort r = 0; r < R_PER_SG; ++r) sums[r][m] += w[r] * xv;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        constexpr ushort Q_VEC4S = B_TILE / 4;
        for (uint base = 0; base < 320u; base += B_TILE) {
            half4 sc_pre[R_PER_SG][Q_VEC4S];
            char4 q_pre[R_PER_SG][Q_VEC4S];
            const uint run = base / 16u;
            const ulong q_off = 640ul + ulong(run) * 512ul + ulong(lane) * 16ul + ulong(base % 16u);
            const ulong sc_off = ulong(base) * 2ul;
            #pragma unroll
            for (ushort r = 0; r < R_PER_SG; ++r) {
                device const uchar* row_ptr = weights + ulong(first_row + r) * 10880ul;
                #pragma unroll
                for (ushort v = 0; v < Q_VEC4S; ++v) {
                    sc_pre[r][v] = half4(*reinterpret_cast<device const packed_half4*>(row_ptr + sc_off + ulong(v) * 8ul));
                    q_pre[r][v] = char4(*reinterpret_cast<device const packed_char4*>(row_ptr + q_off + ulong(v) * 4ul));
                }
            }

            const uint col_base = base * 32u;
            if (count == W_TOK) {
                #pragma unroll
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4u));
                }
            } else {
                #pragma unroll
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = (m < count)
                        ? float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4u))
                        : float4(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (count == W_TOK) {
                #pragma unroll
                for (ushort v = 0; v < Q_VEC4S; ++v) {
                    #pragma unroll
                    for (ushort sub = 0; sub < 4; ++sub) {
                        const ushort s = v * 4u + sub;
                        const ushort c = s * 32u + lane;
                        float w[R_PER_SG];
                        #pragma unroll
                        for (ushort r = 0; r < R_PER_SG; ++r) {
                            w[r] = float(sc_pre[r][v][sub]) * float(q_pre[r][v][sub]);
                        }
                        #pragma unroll
                        for (ushort m = 0; m < W_TOK; ++m) {
                            const float xv = shared_x[m * COLS + c];
                            #pragma unroll
                            for (ushort r = 0; r < R_PER_SG; ++r) sums[r][m] += w[r] * xv;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (ushort v = 0; v < Q_VEC4S; ++v) {
                    #pragma unroll
                    for (ushort sub = 0; sub < 4; ++sub) {
                        const ushort s = v * 4u + sub;
                        const ushort c = s * 32u + lane;
                        float w[R_PER_SG];
                        #pragma unroll
                        for (ushort r = 0; r < R_PER_SG; ++r) {
                            w[r] = float(sc_pre[r][v][sub]) * float(q_pre[r][v][sub]);
                        }
                        for (ushort m = 0; m < count; ++m) {
                            const float xv = shared_x[m * COLS + c];
                            #pragma unroll
                            for (ushort r = 0; r < R_PER_SG; ++r) sums[r][m] += w[r] * xv;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma unroll
    for (ushort r = 0; r < R_PER_SG; ++r) {
        const uint row = first_row + r;
        for (uint m = 0; m < count; ++m) {
            const float total = simd_sum(sums[r][m]);
            if (!lane) {
                if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                output[ulong(token_base + m) * out_stride + row] = total;
            }
        }
    }
}

kernel void graph_hc_down_q8_window(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant HcDownWindowArgs& p [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 shared_x4[512];
    hc_down_q8_window_core<16, 2, 4, false>(weights, input, output, p, error, group, subgroup, lane, tid, shared_x4);
}

kernel void prefill_hc_down_q8_w4_r16(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant HcDownWindowArgs& p [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 shared_x4[512];
    hc_down_q8_window_core<4, 2, 16, false>(weights, input, output, p, error, group, subgroup, lane, tid, shared_x4);
}

kernel void prefill_hc_down_q8_w8_r32(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant HcDownWindowArgs& p [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 shared_x4[256];
    hc_down_q8_window_core<8, 4, 4, true>(weights, input, output, p, error, group, subgroup, lane, tid, shared_x4);
}
