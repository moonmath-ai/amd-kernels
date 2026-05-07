// Isolated LDS port queueing test for gfx942 (CDNA3 / MI300X).
//
// Demonstrates: even with ZERO bank conflicts, back-to-back LDS reads show a
// "first few fast → tail at high latency" ladder in ATT.  The cause is
// *frontend back-pressure* (lgkm queue saturation), not bank conflicts.
//
// Layout: lane stride = 1 dword within each stripe → bank = lane%32 across
// 32 lanes per banking sub-cycle → 32 distinct → ZERO bank conflicts
// (verifiable via SQ_LDS_BANK_CONFLICT counter).
//
// Variant 0: ds_read2_b32  — 8 B/lane, 4 cy port floor.  Compiler emits
//   `ds_read2_b32 vDST, vADDR offset0:0 offset1:64` when 2 reads are at
//   constant offsets.
// Variant 1: ds_read2st64_b64 (via uint4 from far-apart positions) —
//   16 B/lane, 8 cy port floor.

#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int kWaveSize     = 64;
constexpr int kWavesPerCTA  = 4;     // occ=1; 8 waves on the CU compete for 1 LDS unit
constexpr int kBlockSize    = kWaveSize * kWavesPerCTA;
constexpr int kReads        = 32;    // back-to-back LDS reads per loop iter

// Each read uses 2 stripes (offset0 + offset1).  Stripe size = 64 dwords (256 B).
// Reads are spaced 128 dwords apart so they don't overlap.
constexpr int kStripeSize   = 64;    // 64 dwords (lane-natural stride 1)
constexpr int kReadGap      = 128;   // dwords between consecutive reads' bases
constexpr int kLdsTotal     = kReads * kReadGap;  // 16 * 128 = 2048 dw = 8 KB (shared across waves)

extern "C" __global__ void __launch_bounds__(kBlockSize, 1)
lds_queue_test_b32(const uint32_t* __restrict__ in,
                    uint32_t* __restrict__ out,
                    int n_iters)
{
    __shared__ uint32_t lds[kLdsTotal];

    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);

    // Prefill: each thread fills several dwords.
    for (int i = tid; i < kLdsTotal; i += kBlockSize) {
        lds[i] = in[blockIdx.x * kBlockSize + tid] ^ (i * 0x9E3779B9u);
    }
    __syncthreads();

    uint32_t acc[kReads] = {0};

    #pragma unroll
    for (int r = 0; r < kReads; ++r) {
        const uint32_t a = lds[r * kReadGap + 0           + lane];
        const uint32_t b = lds[r * kReadGap + kStripeSize + lane];
        acc[r] ^= a ^ b;
    }

    for (int it = 0; it < n_iters; ++it) {
        // 32 back-to-back ds_read2_b32 — each lane reads 2 dwords at fixed
        // offset0 + offset1 (compiler emits ds_read2_b32 with offset1:64).
        // Lane stride = 1 dw per stripe → bank = lane%32 → ZERO conflicts.
        #pragma unroll
        for (int r = 0; r < kReads; ++r) {
            const uint32_t a = lds[r * kReadGap + 0           + lane];
            const uint32_t b = lds[r * kReadGap + kStripeSize + lane];
            acc[r] ^= a ^ b;
        }
    }

    #pragma unroll
    for (int r = 0; r < kReads; ++r) {
        const uint32_t a = lds[r * kReadGap + 0           + lane];
        const uint32_t b = lds[r * kReadGap + kStripeSize + lane];
        acc[r] ^= a ^ b;
    }

    uint32_t reduced = 0;
    #pragma unroll
    for (int r = 0; r < kReads; ++r) reduced ^= acc[r];
        out[blockIdx.x * kBlockSize + tid] = reduced;
}

extern "C" __global__ void __launch_bounds__(kBlockSize, 1)
lds_queue_test_b64(const uint32_t* __restrict__ in,
                    uint32_t* __restrict__ out,
                    int n_iters)
{
    // 16 back-to-back uint4 reads (16 B/lane → ds_read_b128 or ds_read2_b64).
    // Lane stride = 4 dw within each stripe (uint4 = 4 contiguous dw per lane).
    // Bank = (L*4)%32 = 4*(L%8) → 8 distinct banks → 4-WAY CONFLICT.
    // Included as comparison: shows actual bank conflicts (counter > 0) plus
    // queueing.  8 cy floor + 4-way → 16 cy minimum per read.

    __shared__ uint32_t lds[kLdsTotal];

    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);

    for (int i = tid; i < kLdsTotal; i += kBlockSize) {
        lds[i] = in[blockIdx.x * kBlockSize + tid] ^ (i * 0x9E3779B9u);
    }
    __syncthreads();

    uint4 acc[kReads];
    #pragma unroll
    for (int r = 0; r < kReads; ++r) {
        acc[r] = make_uint4(0, 0, 0, 0);
    }

    #pragma unroll
    for (int r = 0; r < kReads; ++r) {
        const uint4 v = *(const uint4*)&lds[r * kReadGap + lane * 4];
        acc[r].x ^= v.x; acc[r].y ^= v.y; acc[r].z ^= v.z; acc[r].w ^= v.w;
    }

    for (int it = 0; it < n_iters; ++it) {
        #pragma unroll
        for (int r = 0; r < kReads; ++r) {
            const uint4 v = *(const uint4*)&lds[r * kReadGap + lane * 4];
            acc[r].x ^= v.x; acc[r].y ^= v.y; acc[r].z ^= v.z; acc[r].w ^= v.w;
        }
    }

    #pragma unroll
    for (int r = 0; r < kReads; ++r) {
        const uint4 v = *(const uint4*)&lds[r * kReadGap + lane * 4];
        acc[r].x ^= v.x; acc[r].y ^= v.y; acc[r].z ^= v.z; acc[r].w ^= v.w;
    }

    for (int r = 0; r < kReads; ++r) {
        out[blockIdx.x * kBlockSize + tid] ^= acc[r].x ^ acc[r].y ^ acc[r].z ^ acc[r].w;
    }
}

// ─── Host launcher (callable from Python) ───────────────────────────────
extern "C" int launch_lds_queue_test(const uint32_t* d_in,
                                      uint32_t* d_out,
                                      int n_blocks,
                                      int n_iters,
                                      int variant,
                                      void* stream)
{
    dim3 grid(304);
    dim3 block(kBlockSize);
    hipStream_t s = static_cast<hipStream_t>(stream);
    if (variant == 0) {
        hipLaunchKernelGGL(lds_queue_test_b32, grid, block, 0, s,
                           d_in, d_out, n_iters);
    } else {
        hipLaunchKernelGGL(lds_queue_test_b64, grid, block, 0, s,
                           d_in, d_out, n_iters);
    }
    return 0;
}
