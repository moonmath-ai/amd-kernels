#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>

#define TILE_SIZE 16

// Tiled scaled dot-product attention kernel.
// Each block handles one (batch*heads, query_tile_n) where query_tile_n has TILE_SIZE rows.
// Q, K, V: [batch, heads, seq_len, head_dim]
// Out:     [batch, heads, seq_len, head_dim]
__global__ void attention_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int seq_len,
    int head_dim,
    float scale
) {
    int bh = blockIdx.y;
    int kv_local = threadIdx.x;
    int q_local = threadIdx.y;
    int q_tile_n = blockIdx.x;
    int q_row = q_tile_n * TILE_SIZE + q_local;
    int offset = bh * seq_len * head_dim;
    bool q_valid = (q_row < seq_len);

    // log2(e) lets us use exp2 for softmax while preserving exp semantics.
    float scale_log2e = scale * 1.4426950408889634f;

    __shared__ float q_sub[TILE_SIZE][TILE_SIZE];
    __shared__ float k_sub[TILE_SIZE][TILE_SIZE];
    __shared__ float v_sub[TILE_SIZE][TILE_SIZE];
    __shared__ float score_sub[TILE_SIZE][TILE_SIZE];
    __shared__ float prob_sub[TILE_SIZE][TILE_SIZE];
    __shared__ float row_m[TILE_SIZE];
    __shared__ float row_l[TILE_SIZE];
    __shared__ float row_beta[TILE_SIZE];
    __shared__ float row_m_new[TILE_SIZE];

    if (q_valid) {
        for (int d = kv_local; d < head_dim; d += TILE_SIZE) {
            int out_idx = offset + q_row * head_dim + d;
            Out[out_idx] = 0.0f;
        }
    }
    if (kv_local == 0) {
        row_m[q_local] = -INFINITY;
        row_l[q_local] = 0.0f;
    }
    __syncthreads();

    int m_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < m_tiles; ++m) {
        float score = 0.0f;
        for (int k_tile = 0; k_tile < head_dim; k_tile += TILE_SIZE) {
            int d = k_tile + kv_local;
            int q_idx = offset + q_row * head_dim + d;
            int kv_load = m * TILE_SIZE + q_local;
            int k_idx = offset + kv_load * head_dim + d;

            if (q_valid && d < head_dim) {
                q_sub[q_local][kv_local] = Q[q_idx];
            } else {
                q_sub[q_local][kv_local] = 0.0f;
            }
            if (kv_load < seq_len && d < head_dim) {
                k_sub[q_local][kv_local] = K[k_idx];
            } else {
                k_sub[q_local][kv_local] = 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int d_local = 0; d_local < TILE_SIZE; ++d_local) {
                score += q_sub[q_local][d_local] * k_sub[kv_local][d_local];
            }
            __syncthreads();
        }

        int kv = m * TILE_SIZE + kv_local;
        float scaled = q_valid && kv < seq_len ? (score * scale_log2e) : -INFINITY;
        score_sub[q_local][kv_local] = scaled;
        __syncthreads();

        if (kv_local == 0) {
            if (!q_valid) {
                row_m_new[q_local] = 0.0f;
                row_beta[q_local] = 0.0f;
                row_m[q_local] = 0.0f;
                row_l[q_local] = 0.0f;
            } else {
                float tile_max = -INFINITY;
                for (int j = 0; j < TILE_SIZE; ++j) {
                    tile_max = fmaxf(tile_max, score_sub[q_local][j]);
                }
                float prev_m = row_m[q_local];
                float new_m = fmaxf(prev_m, tile_max);
                float beta = 0.0f;
                if (isfinite(prev_m) && isfinite(new_m)) {
                    beta = exp2f(prev_m - new_m);
                }
                row_m_new[q_local] = new_m;
                row_beta[q_local] = beta;
            }
        }
        __syncthreads();

        float p = 0.0f;
        if (q_valid && kv < seq_len) {
            p = exp2f(score_sub[q_local][kv_local] - row_m_new[q_local]);
        }
        prob_sub[q_local][kv_local] = p;
        __syncthreads();

        if (kv_local == 0 && q_valid) {
            float tile_sum = 0.0f;
            for (int j = 0; j < TILE_SIZE; ++j) {
                tile_sum += prob_sub[q_local][j];
            }
            row_l[q_local] = row_l[q_local] * row_beta[q_local] + tile_sum;
            row_m[q_local] = row_m_new[q_local];
        }
        __syncthreads();

        for (int d_tile = 0; d_tile < head_dim; d_tile += TILE_SIZE) {
            int d = d_tile + kv_local;
            int kv_load = m * TILE_SIZE + q_local;
            int v_idx = offset + kv_load * head_dim + d;

            if (kv_load < seq_len && d < head_dim) {
                v_sub[q_local][kv_local] = V[v_idx];
            } else {
                v_sub[q_local][kv_local] = 0.0f;
            }
            __syncthreads();

            if (q_valid && d < head_dim) {
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; ++j) {
                    acc += prob_sub[q_local][j] * v_sub[j][kv_local];
                }
                int out_idx = offset + q_row * head_dim + d;
                Out[out_idx] = Out[out_idx] * row_beta[q_local] + acc;
            }
            __syncthreads();
        }
    }

    if (q_valid) {
        float inv_l = 1.0f / fmaxf(row_l[q_local], 1e-20f);
        for (int d = kv_local; d < head_dim; d += TILE_SIZE) {
            int out_idx = offset + q_row * head_dim + d;
            Out[out_idx] *= inv_l;
        }
    }
}

// C interface for Python ctypes
extern "C" {

hipError_t launch_attention_forward(
    const float* Q, const float* K, const float* V, float* Out,
    int batch, int heads, int seq_len, int head_dim,
    hipStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int bh = batch * heads;
    int q_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    dim3 grid(q_tiles, bh);
    dim3 block(TILE_SIZE, TILE_SIZE);

    hipLaunchKernelGGL(
        attention_forward,
        grid, block, 0, stream,
        Q, K, V, Out, seq_len, head_dim, scale
    );

    return hipGetLastError();
}

}  // extern "C"
