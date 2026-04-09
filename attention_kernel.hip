#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace {

using fp16_t = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

constexpr int kWaveSize = 64;
constexpr int kSeqTile = 16;
constexpr int kKTile = 16;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxHeadTiles = kMaxHeadDim / kKTile;  // 8
constexpr int kPrefetchFrags = 4;                     // dual-buffer width
constexpr float kLog2e = 1.4426950408889634f;

// --- helpers ---

__device__ __forceinline__ fp16x4_t zero_f16x4() {
    return fp16x4_t{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ fp32x4_t zero_f32x4() {
    return fp32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ fp16x4_t fp32x4_to_fp16x4(fp32x4_t x) {
    return fp16x4_t{
        static_cast<fp16_t>(x[0]),
        static_cast<fp16_t>(x[1]),
        static_cast<fp16_t>(x[2]),
        static_cast<fp16_t>(x[3]),
    };
}

__device__ __forceinline__ float wave_reduce_max_16(float x) {
    x = fmaxf(x, __shfl_xor(x, 8, 16));
    x = fmaxf(x, __shfl_xor(x, 4, 16));
    x = fmaxf(x, __shfl_xor(x, 2, 16));
    x = fmaxf(x, __shfl_xor(x, 1, 16));
    return x;
}

__device__ __forceinline__ float wave_reduce_sum_16(float x) {
    x += __shfl_xor(x, 8, 16);
    x += __shfl_xor(x, 4, 16);
    x += __shfl_xor(x, 2, 16);
    x += __shfl_xor(x, 1, 16);
    return x;
}

__device__ __forceinline__ fp32x4_t mfma16x16x16(fp16x4_t a, fp16x4_t b, fp32x4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

// Load 4 contiguous fp16 from row-major [rows, stride] layout.
__device__ __forceinline__ fp16x4_t load_contiguous4(
    const fp16_t* base, int row, int col, int stride
) {
    return *reinterpret_cast<const fp16x4_t*>(base + row * stride + col);
}

// Load V fragment: 4 elements from different rows (transposed access for PV matmul).
__device__ __forceinline__ fp16x4_t load_v_fragment(
    const fp16_t* base, int key_base, int dim, int row_group, int stride
) {
    fp16x4_t frag;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        frag[i] = base[(key_base + 4 * row_group + i) * stride + dim];
    return frag;
}

// --- kernel ---

__global__ __launch_bounds__(kWaveSize) void attention_forward(
    const fp16_t* __restrict__ Q,
    const fp16_t* __restrict__ K,
    const fp16_t* __restrict__ V,
    float* __restrict__ Out,
    int seq_len,
    int head_dim,
    float scale_log2e
) {
    // MFMA 16x16x16 lane mapping: lane_col picks column, lane_group picks
    // which 4-element fragment row this lane owns.
    const int lane = threadIdx.x;
    const int lane_col = lane & 15;
    const int lane_group = lane >> 4;
    const int bh = blockIdx.y;
    const int q_tile = blockIdx.x;
    const int head_tiles = head_dim / kKTile;
    const int q_tile_base = q_tile * kSeqTile;
    const int q_load_row = q_tile_base + lane_col;
    const int out_row_base = q_tile_base + 4 * lane_group;

    const fp16_t* q_slice = Q + static_cast<long long>(bh) * seq_len * head_dim;
    const fp16_t* k_slice = K + static_cast<long long>(bh) * seq_len * head_dim;
    const fp16_t* v_slice = V + static_cast<long long>(bh) * seq_len * head_dim;
    float* out_slice = Out + static_cast<long long>(bh) * seq_len * head_dim;

    // Q loaded once: fp16x4[8] = 16 VGPRs.
    fp16x4_t q_regs[kMaxHeadTiles];

    // O accumulator in fp32 for precision: fp32x4[8] = 32 VGPRs.
    fp32x4_t o_regs[kMaxHeadTiles];

    // Online softmax state: 8 VGPRs.
    float row_max[4];
    float row_sum[4];

    #pragma unroll
    for (int t = 0; t < kMaxHeadTiles; ++t) {
        q_regs[t] = zero_f16x4();
        o_regs[t] = zero_f32x4();
        if (t < head_tiles) {
            q_regs[t] = load_contiguous4(
                q_slice, q_load_row, t * kKTile + 4 * lane_group, head_dim
            );
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }

    __shared__ fp16_t prob_tile[kSeqTile][kSeqTile];

    const int m_tiles = seq_len / kSeqTile;
    for (int m = 0; m < m_tiles; ++m) {
        const int key_base = m * kSeqTile;
        const int key_col = key_base + lane_col;

        // --- QK matmul: scores[16x16] = Q_tile[16xD] * K_tile[16xD]^T ---
        // Dual-buffer prefetch: k_curr[4] + k_next[4] = 16 VGPRs.
        fp32x4_t scores = zero_f32x4();
        fp16x4_t k_curr[kPrefetchFrags];
        fp16x4_t k_next[kPrefetchFrags];

        // Load first chunk into k_curr, second into k_next.
        #pragma unroll
        for (int i = 0; i < kPrefetchFrags; ++i) {
            k_curr[i] = load_contiguous4(
                k_slice, key_col, i * kKTile + 4 * lane_group, head_dim
            );
            k_next[i] = load_contiguous4(
                k_slice, key_col, (kPrefetchFrags + i) * kKTile + 4 * lane_group, head_dim
            );
        }

        // First chunk: compute on k_curr, move k_next -> k_curr.
        #pragma unroll
        for (int i = 0; i < kPrefetchFrags; ++i) {
            scores = mfma16x16x16(q_regs[i], k_curr[i], scores);
            k_curr[i] = k_next[i];
        }

        // Second chunk: compute remaining with what's now in k_curr.
        #pragma unroll
        for (int i = 0; i < kPrefetchFrags; ++i) {
            scores = mfma16x16x16(q_regs[kPrefetchFrags + i], k_curr[i], scores);
        }

        // --- Online softmax ---
        float alpha[4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float score = scores[i] * scale_log2e;
            const float next_max = fmaxf(row_max[i], wave_reduce_max_16(score));
            const float prev_scale = row_sum[i] > 0.0f ? exp2f(row_max[i] - next_max) : 0.0f;
            const float p = exp2f(score - next_max);
            const float tile_sum = wave_reduce_sum_16(p);

            alpha[i] = prev_scale;
            row_max[i] = next_max;
            row_sum[i] = row_sum[i] * prev_scale + tile_sum;
            prob_tile[4 * lane_group + i][lane_col] = static_cast<fp16_t>(p);
        }

        __syncthreads();

        // Reload P in MFMA-A layout (transposed from how it was stored).
        const fp16x4_t p_reg =
            *reinterpret_cast<const fp16x4_t*>(&prob_tile[lane_col][4 * lane_group]);

        // Rescale previous O by alpha.
        #pragma unroll
        for (int t = 0; t < kMaxHeadTiles; ++t) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                o_regs[t][i] *= alpha[i];
        }

        // --- PV matmul: O += P[16x16] * V[16xD] ---
        // Dual-buffer prefetch: v_curr[4] + v_next[4] = 16 VGPRs.
        fp16x4_t v_curr[kPrefetchFrags];
        fp16x4_t v_next[kPrefetchFrags];

        #pragma unroll
        for (int i = 0; i < kPrefetchFrags; ++i) {
            v_curr[i] = load_v_fragment(
                v_slice, key_base, i * kKTile + lane_col, lane_group, head_dim
            );
            v_next[i] = load_v_fragment(
                v_slice, key_base, (kPrefetchFrags + i) * kKTile + lane_col, lane_group, head_dim
            );
        }

        // First chunk.
        #pragma unroll
        for (int i = 0; i < kPrefetchFrags; ++i) {
            o_regs[i] = mfma16x16x16(p_reg, v_curr[i], o_regs[i]);
            v_curr[i] = v_next[i];
        }

        // Second chunk.
        #pragma unroll
        for (int i = 0; i < kPrefetchFrags; ++i) {
            o_regs[kPrefetchFrags + i] = mfma16x16x16(p_reg, v_curr[i], o_regs[kPrefetchFrags + i]);
        }

        __syncthreads();
    }

    // --- Write output: O / row_sum ---
    #pragma unroll
    for (int t = 0; t < kMaxHeadTiles; ++t) {
        if (t < head_tiles) {
            const int dim = t * kKTile + lane_col;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                out_slice[(out_row_base + i) * head_dim + dim] =
                    o_regs[t][i] / row_sum[i];
            }
        }
    }
}

}  // namespace

extern "C" {

hipError_t launch_attention_forward(
    const fp16_t* Q,
    const fp16_t* K,
    const fp16_t* V,
    float* Out,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    hipStream_t stream
) {
    if (seq_len <= 0 || (seq_len % kSeqTile) != 0)
        return hipErrorInvalidValue;
    if (head_dim <= 0 || head_dim > kMaxHeadDim || (head_dim % kKTile) != 0)
        return hipErrorInvalidValue;

    const float scale_log2e = kLog2e / sqrtf(static_cast<float>(head_dim));
    const int bh = batch * heads;
    const int q_tiles = seq_len / kSeqTile;

    dim3 grid(q_tiles, bh);
    dim3 block(kWaveSize);

    hipLaunchKernelGGL(
        attention_forward,
        grid, block, 0, stream,
        Q, K, V, Out, seq_len, head_dim, scale_log2e
    );

    return hipGetLastError();
}

}  // extern "C"
