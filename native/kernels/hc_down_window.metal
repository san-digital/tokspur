// Exact windowed GGUF Q8_0 Hyper-Connections down projection for 10240-to-320 prefill rows.
// Stages 16 token rows of 128 FP32 columns in 8 KiB threadgroup memory per step while preserving the single-SIMD-group 0..319 block accumulation order and simd_sum reduction of graph_hc_q8_exact_r1_u16 under -fno-fast-math.
#include <metal_stdlib>
using namespace metal;

struct HcDownWindowArgs {
    uint columns, rows, tokens, reserved;
    ulong input_stride_bytes, output_stride_bytes;
};
static_assert(sizeof(HcDownWindowArgs) == 32, "HC down window argument ABI");

inline float hc_down_weight(device const uchar* weights, uint row, uint b, ushort lane, uint reserved) {
    if (reserved == 1) {
        device const uchar* row_ptr = weights + ulong(row) * 10880;
        const float scale = float(*reinterpret_cast<device const half*>(row_ptr + ulong(b) * 2));
        const uint run = b / 16;
        const uint quant_offset = 640 + run * 512 + uint(lane) * 16 + (b % 16);
        const float quant = float(*reinterpret_cast<device const char*>(row_ptr + quant_offset));
        return scale * quant;
    }
    device const uchar* block = weights + (ulong(row) * 320 + b) * 34;
    return float(*reinterpret_cast<device const half*>(block)) * float(*reinterpret_cast<device const char*>(block + 2 + lane));
}

kernel void graph_hc_down_q8_window(device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant HcDownWindowArgs& p [[buffer(3)]], device atomic_uint* error [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], ushort subgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (p.columns != 10240 || p.rows != 320 || !p.tokens || p.tokens > 2048 || p.reserved > 1 ||
        p.input_stride_bytes < 40960 || p.output_stride_bytes < 1280 || p.input_stride_bytes % 4 || p.output_stride_bytes % 4) {
        if (!tid) atomic_fetch_or_explicit(error, 4u, memory_order_relaxed);
        return;
    }
    constexpr ushort W_TOK = 16, SG_COUNT = 8, R_PER_SG = 2, B_TILE = 4;
    constexpr ushort THREADS = SG_COUNT * 32, R_TG = SG_COUNT * R_PER_SG, COLS = B_TILE * 32, COLS4 = COLS / 4;
    const uint first_row = group.x * R_TG + subgroup * R_PER_SG;
    const uint token_base = group.y * W_TOK;
    if (token_base >= p.tokens) return;
    const uint in_stride = uint(p.input_stride_bytes / sizeof(float));
    const uint out_stride = uint(p.output_stride_bytes / sizeof(float));
    const uint count = min(uint(W_TOK), p.tokens - token_base);
    device const float* row_in = input + ulong(token_base) * in_stride;

    threadgroup float4 shared_x4[(W_TOK * COLS) / 4];
    threadgroup float* shared_x = reinterpret_cast<threadgroup float*>(shared_x4);

    float sums[R_PER_SG][W_TOK];
    #pragma unroll
    for (ushort r = 0; r < R_PER_SG; ++r)
        #pragma unroll
        for (ushort m = 0; m < W_TOK; ++m) sums[r][m] = 0.0f;

    const bool full_rows = (first_row + R_PER_SG <= p.rows);

    if (p.reserved == 0u) {
        for (uint base = 0; base < 320; base += B_TILE) {
            half sc_pre[R_PER_SG][B_TILE];
            char q_pre[R_PER_SG][B_TILE];
            if (full_rows) {
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
            }

            const uint col_base = base * 32;
            if (count == W_TOK) {
                #pragma unroll
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4));
                }
            } else {
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = (m < count)
                        ? float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4))
                        : float4(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (count == W_TOK && full_rows) {
                #pragma unroll
                for (ushort s = 0; s < B_TILE; ++s) {
                    const ushort c = s * 32 + lane;
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
            } else if (first_row < p.rows) {
                for (ushort s = 0; s < B_TILE; ++s) {
                    const ushort c = s * 32 + lane;
                    float w[R_PER_SG];
                    #pragma unroll
                    for (ushort r = 0; r < R_PER_SG; ++r) {
                        const uint row = first_row + r;
                        w[r] = row < p.rows ? hc_down_weight(weights, row, base + s, lane, 0u) : 0.0f;
                    }
                    for (uint m = 0; m < count; ++m) {
                        const float xv = shared_x[m * COLS + c];
                        #pragma unroll
                        for (ushort r = 0; r < R_PER_SG; ++r) sums[r][m] += w[r] * xv;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint base = 0; base < 320; base += B_TILE) {
            half4 sc_pre[R_PER_SG];
            char4 q_pre[R_PER_SG];
            if (full_rows) {
                const uint run = base / 16u;
                const ulong q_off = 640ul + ulong(run) * 512ul + ulong(lane) * 16ul + ulong(base % 16u);
                const ulong sc_off = ulong(base) * 2ul;
                #pragma unroll
                for (ushort r = 0; r < R_PER_SG; ++r) {
                    device const uchar* row_ptr = weights + ulong(first_row + r) * 10880ul;
                    sc_pre[r] = half4(*reinterpret_cast<device const packed_half4*>(row_ptr + sc_off));
                    q_pre[r] = char4(*reinterpret_cast<device const packed_char4*>(row_ptr + q_off));
                }
            }

            const uint col_base = base * 32;
            if (count == W_TOK) {
                #pragma unroll
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4));
                }
            } else {
                for (uint idx4 = tid; idx4 < uint(W_TOK) * COLS4; idx4 += THREADS) {
                    const ushort m = idx4 / COLS4, c4 = idx4 % COLS4;
                    shared_x4[idx4] = (m < count)
                        ? float4(*reinterpret_cast<device const packed_float4*>(row_in + ulong(m) * in_stride + col_base + uint(c4) * 4))
                        : float4(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (count == W_TOK && full_rows) {
                #pragma unroll
                for (ushort s = 0; s < B_TILE; ++s) {
                    const ushort c = s * 32 + lane;
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
            } else if (first_row < p.rows) {
                for (ushort s = 0; s < B_TILE; ++s) {
                    const ushort c = s * 32 + lane;
                    float w[R_PER_SG];
                    #pragma unroll
                    for (ushort r = 0; r < R_PER_SG; ++r) {
                        const uint row = first_row + r;
                        w[r] = row < p.rows ? hc_down_weight(weights, row, base + s, lane, 1u) : 0.0f;
                    }
                    for (uint m = 0; m < count; ++m) {
                        const float xv = shared_x[m * COLS + c];
                        #pragma unroll
                        for (ushort r = 0; r < R_PER_SG; ++r) sums[r][m] += w[r] * xv;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma unroll
    for (ushort r = 0; r < R_PER_SG; ++r) {
        const uint row = first_row + r;
        if (row < p.rows) {
            for (uint m = 0; m < count; ++m) {
                const float total = simd_sum(sums[r][m]);
                if (!lane) {
                    if (!isfinite(total)) atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                    output[ulong(token_base + m) * out_stride + row] = total;
                }
            }
        }
    }
}
