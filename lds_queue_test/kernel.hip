// Microbenchmark mirroring v5pp `pv_phase` LDS V access pattern (gfx942).
//
// Reproduces the EXACT lane indexing of attention_kernel_v5pp.hip pv_phase:
//
//     col   = lane & 15;          // 0..15
//     quad  = lane >> 4;          // 0..3
//     k_swiz = col << 1;
//     for (dt = 0..7)
//       for (sn = 0..3)
//         dw = (dt*16 + col) * kVDwPerRow + ((sn*8 + 2*quad) ^ k_swiz);
//         bf16x4_t vf = *(const bf16x4_t*)&lds_v[dw];
//
// LDS layout = 16 KB lds_v (kVStorageDw = 4096 dw), matching v5pp.  No mfma —
// pure read pattern so ATT shows ONLY the LDS port ladder.
//
// Per-iteration structure deliberately avoids read-modify-write deps on the
// accumulator: each LDS read goes to its own kReads-element temp array, then
// reduced at end-of-iter into a single scalar.  This lets all kReads ds_reads
// issue back-to-back into independent VGPR slots → exposes the full LDS-queue
// back-pressure ladder instead of being serialized by acc-chain.
//
// Launch: 4 waves/CTA, occ=2 (= 8 waves/CU), grid=304 (= MI300X CU count).

#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int kWaveSize     = 64;
constexpr int kWavesPerCTA  = 4;
constexpr int kBlockSize    = kWaveSize * kWavesPerCTA;   // 256

// Match v5pp constants exactly:
constexpr int kHeadDim    = 128;
constexpr int kKTile      = 16;
constexpr int kNBlock     = 64;
constexpr int kDTiles     = kHeadDim / kKTile;        // 8
constexpr int kNSub       = 4;
constexpr int kVDwPerRow  = kNBlock / 2;              // 32
constexpr int kVStorageDw = kHeadDim * kVDwPerRow;    // 4096 dw = 16 KB
constexpr int kReads      = kDTiles * kNSub;          // 32 reads per pv_phase

extern "C" __global__ void __launch_bounds__(kBlockSize, 2)
pv_phase_microbench(const uint32_t* __restrict__ in,
                    uint32_t*       __restrict__ out,
                    int n_iters)
{
    __shared__ uint32_t lds_v[kVStorageDw];   // 16 KB, identical to v5pp

    const int tid    = threadIdx.x;
    const int lane   = tid & (kWaveSize - 1);
    const int col    = lane & 15;
    const int quad   = lane >> 4;
    const int k_swiz = col << 1;

    // Prefill (cooperative).
    for (int i = tid; i < kVStorageDw; i += kBlockSize) {
        lds_v[i] = in[blockIdx.x * kBlockSize + tid] ^ (i * 0x9E3779B9u);
    }
    __syncthreads();

    uint32_t reduced = 0;

    for (int it = 0; it < n_iters; ++it) {
        // 32 independent reads into their own VGPR slots — no acc dep, so the
        // compiler can issue them back-to-back and the LDS port queue fills.
        uint32_t t0[kReads];
        uint32_t t1[kReads];

        #pragma unroll
        for (int r = 0; r < kReads; ++r) {
            const int dt = r / kNSub;
            const int sn = r % kNSub;
            const int dw = (dt * 16 + col) * kVDwPerRow
                         + ((sn * 8 + 2 * quad) ^ k_swiz);
            t0[r] = lds_v[dw + 0];
            t1[r] = lds_v[dw + 1];
        }

        // Reduce after all reads complete (single s_waitcnt drain).
        #pragma unroll
        for (int r = 0; r < kReads; ++r) {
            reduced ^= t0[r] ^ t1[r];
        }
    }

    out[blockIdx.x * kBlockSize + tid] = reduced;
}

// ─── Host launcher ──────────────────────────────────────────────────────
extern "C" int launch_lds_queue_test(const uint32_t* d_in,
                                      uint32_t* d_out,
                                      int n_blocks,
                                      int n_iters,
                                      int variant,
                                      void* stream)
{
    (void)variant;
    dim3 grid(n_blocks);
    dim3 block(kBlockSize);
    hipStream_t s = static_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(pv_phase_microbench, grid, block, 0, s,
                       d_in, d_out, n_iters);
    return 0;
}
