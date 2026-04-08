#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>

#define TILE_SIZE 16

// Naive scaled dot-product attention kernel
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
    // Each block handles one (batch, head, query_row)
    int bh = blockIdx.y;  // batch * heads index
    int q_row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores = smem;  // [seq_len]

    int offset = bh * seq_len * head_dim;
    const float* q = Q + offset + q_row * head_dim;
    const float* k = K + offset;
    const float* v = V + offset;
    float* out = Out + offset + q_row * head_dim;

    // Step 1: Compute QK^T scores for this query row
    for (int kv = tid; kv < seq_len; kv += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q[d] * k[kv * head_dim + d];
        }
        scores[kv] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax over scores
    // Find max for numerical stability
    float local_max = -INFINITY;
    for (int kv = tid; kv < seq_len; kv += blockDim.x) {
        local_max = fmaxf(local_max, scores[kv]);
    }
    // Reduce max across threads
    __shared__ float s_max;
    if (tid == 0) s_max = -INFINITY;
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(local_max));
    __syncthreads();
    float max_val = s_max;

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int kv = tid; kv < seq_len; kv += blockDim.x) {
        scores[kv] = expf(scores[kv] - max_val);
        local_sum += scores[kv];
    }
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / s_sum;
    for (int kv = tid; kv < seq_len; kv += blockDim.x) {
        scores[kv] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum over V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv = 0; kv < seq_len; kv++) {
            acc += scores[kv] * v[kv * head_dim + d];
        }
        out[d] = acc;
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
    size_t smem_bytes = seq_len * sizeof(float);
    int threads = min(256, seq_len);

    dim3 grid(seq_len, bh);
    dim3 block(threads);

    hipLaunchKernelGGL(
        attention_forward,
        grid, block, smem_bytes, stream,
        Q, K, V, Out, seq_len, head_dim, scale
    );

    return hipGetLastError();
}

}  // extern "C"
