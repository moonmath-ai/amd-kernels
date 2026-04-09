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
constexpr int kMaxHeadTiles = kMaxHeadDim / kKTile;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ fp16x4_t zero_f16x4() {
    return fp16x4_t{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ fp32x4_t zero_f32x4() {
    return fp32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
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

__device__ __forceinline__ fp16x4_t load_contiguous4(
    const fp16_t* base,
    int row,
    int col,
    int head_dim,
    bool row_valid
) {
    if (!row_valid || col + 3 >= head_dim) {
        fp16x4_t frag = zero_f16x4();
        if (!row_valid || col >= head_dim) {
            return frag;
        }
        const fp16_t* src = base + row * head_dim + col;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (col + i < head_dim) {
                frag[i] = src[i];
            }
        }
        return frag;
    }

    const fp16_t* src = base + row * head_dim + col;
    return *reinterpret_cast<const fp16x4_t*>(src);
}

__device__ __forceinline__ fp16x4_t load_v_fragment(
    const fp16_t* base,
    int key_base,
    int dim,
    int row_group,
    int seq_len,
    int head_dim
) {
    fp16x4_t frag = zero_f16x4();
    if (dim >= head_dim) {
        return frag;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int key = key_base + 4 * row_group + i;
        if (key < seq_len) {
            frag[i] = base[key * head_dim + dim];
        }
    }
    return frag;
}

__global__ __launch_bounds__(kWaveSize) void attention_forward(
    const fp16_t* __restrict__ Q,
    const fp16_t* __restrict__ K,
    const fp16_t* __restrict__ V,
    float* __restrict__ Out,
    int seq_len,
    int head_dim,
    float scale_log2e
) {
    // For 16x16x16 MFMA, lane_col selects the 16-way row/column coordinate and
    // lane_group selects the 4-element fragment within that coordinate.
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

    fp16x4_t q_regs[kMaxHeadTiles];
    fp32x4_t o_regs[kMaxHeadTiles];
    float row_max[4];
    float row_sum[4];

    #pragma unroll
    for (int t = 0; t < kMaxHeadTiles; ++t) {
        q_regs[t] = zero_f16x4();
        o_regs[t] = zero_f32x4();
        if (t < head_tiles) {
            q_regs[t] = load_contiguous4(
                q_slice,
                q_load_row,
                t * kKTile + 4 * lane_group,
                head_dim,
                q_load_row < seq_len
            );
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }

    __shared__ fp16_t prob_tile[kSeqTile][kSeqTile];

    const int m_tiles = (seq_len + kSeqTile - 1) / kSeqTile;
    for (int m = 0; m < m_tiles; ++m) {
        const int key_col = m * kSeqTile + lane_col;

        fp32x4_t scores = zero_f32x4();
        #pragma unroll
        for (int t = 0; t < kMaxHeadTiles; ++t) {
            if (t < head_tiles) {
                fp16x4_t k_reg = load_contiguous4(
                    k_slice,
                    key_col,
                    t * kKTile + 4 * lane_group,
                    head_dim,
                    key_col < seq_len
                );
                scores = mfma16x16x16(q_regs[t], k_reg, scores);
            }
        }

        float alpha[4];
        float prob[4];
        fp16x4_t prob_half = zero_f16x4();

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = out_row_base + i;
            const bool row_valid = row < seq_len;
            const bool score_valid = row_valid && key_col < seq_len;
            const float score = score_valid ? scores[i] * scale_log2e : -INFINITY;
            const float tile_max = wave_reduce_max_16(score);
            const float next_max = row_valid ? fmaxf(row_max[i], tile_max) : 0.0f;
            const float prev_scale = row_sum[i] > 0.0f ? exp2f(row_max[i] - next_max) : 0.0f;
            const float p = score_valid ? exp2f(score - next_max) : 0.0f;
            const float tile_sum = wave_reduce_sum_16(p);

            alpha[i] = prev_scale;
            prob[i] = p;
            prob_half[i] = static_cast<fp16_t>(p);
            row_max[i] = next_max;
            row_sum[i] = row_valid ? row_sum[i] * prev_scale + tile_sum : 0.0f;
            prob_tile[4 * lane_group + i][lane_col] = prob_half[i];
        }

        __syncthreads();

        // Softmax probabilities come out in MFMA accumulator layout, so stage
        // them once in LDS and reload them in MFMA-A layout for the PV step.
        fp16x4_t p_reg =
            *reinterpret_cast<const fp16x4_t*>(&prob_tile[lane_col][4 * lane_group]);

        #pragma unroll
        for (int t = 0; t < kMaxHeadTiles; ++t) {
            if (t < head_tiles) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o_regs[t][i] *= alpha[i];
                }

                fp16x4_t v_reg = load_v_fragment(
                    v_slice,
                    m * kSeqTile,
                    t * kKTile + lane_col,
                    lane_group,
                    seq_len,
                    head_dim
                );
                o_regs[t] = mfma16x16x16(p_reg, v_reg, o_regs[t]);
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < kMaxHeadTiles; ++t) {
        if (t >= head_tiles) {
            continue;
        }

        const int dim = t * kKTile + lane_col;
        if (dim >= head_dim) {
            continue;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = out_row_base + i;
            if (row < seq_len) {
                const float denom = row_sum[i] > 0.0f ? row_sum[i] : 1.0f;
                out_slice[row * head_dim + dim] = o_regs[t][i] / denom;
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
    if (seq_len <= 0 || head_dim <= 0 || head_dim > kMaxHeadDim || (head_dim % kKTile) != 0) {
        return hipErrorInvalidValue;
    }

    const float scale_log2e = kLog2e / sqrtf(static_cast<float>(head_dim));
    const int bh = batch * heads;
    const int q_tiles = (seq_len + kSeqTile - 1) / kSeqTile;

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
