// Attention forward bf16, gfx942 (MI300X).
//   4-wave block; each wave processes 2 × 16-row Q tiles (lo/hi).
//   K double-buffered in LDS; V written transposed in LDS.
//   16x16x16 MFMA. Online softmax. Output bf16 (RTZ pack).

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace {

using bf16     = __hip_bfloat16;
using bf16x4_t = short    __attribute__((ext_vector_type(4)));
using bf16x8_t = short    __attribute__((ext_vector_type(8)));
using fp32x4_t = float    __attribute__((ext_vector_type(4)));
using u32x2_t  = uint32_t __attribute__((ext_vector_type(2)));
using u32x4_t  = uint32_t __attribute__((ext_vector_type(4)));

constexpr int kWaveSize  = 64;
constexpr int kNumWaves  = 4;
constexpr int kBlockSize = kWaveSize * kNumWaves;
constexpr int kNBlock    = 64;
constexpr int kNSub      = 4;     // 4 MFMA sub-tiles along the N dim per K-block (kNBlock / kKTile)
constexpr int kKTile     = 16;    // 16-row MFMA tile
constexpr int kHeadDim   = 128;
constexpr int kDTiles    = kHeadDim / kKTile;
constexpr int kHalfDim   = kHeadDim / 2;
constexpr int kLdsStride = 68;
constexpr float kLog2e   = 1.4426950408889634f;

__device__ __forceinline__ fp32x4_t zero4() { return fp32x4_t{0,0,0,0}; }
__device__ __forceinline__ float    exp2c(float x) { return __builtin_amdgcn_exp2f(fmaxf(x, -126.f)); }
__device__ __forceinline__ float    rcpf(float x)  { return __builtin_amdgcn_rcpf(x); }
__device__ __forceinline__ float    sxor(float v, int m) { return __shfl_xor(v, m, 64); }
__device__ __forceinline__ fp32x4_t mfma(bf16x4_t a, bf16x4_t b, fp32x4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Buffer resource config bits used for raw-buffer operations on gfx9+.
// This becomes the "config" dword in the 128-bit buffer resource descriptor.
// The code uses RAW buffer ops (buffer_load/store), so we build a descriptor
// once and then access memory via byte offsets (`off`).
constexpr uint32_t kBufCfg = 0x00020000u;

// Build an AMDGPU buffer resource descriptor pointing at `p`.
// - Base pointer: `p`
// - Stride: 0 (not used for raw buffer ops here)
// - Num records / bound: 0xFFFFFFFF (effectively "large"; we rely on kernel
//   logic for safety rather than descriptor bounds)
// - Config: `kBufCfg` (see above)
__device__ __forceinline__ __amdgpu_buffer_rsrc_t mkr(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, 0xFFFFFFFFu, kBufCfg);
}

// Load 8 bytes (64-bit) from buffer resource `r` at byte offset `off`,
// reinterpret as 4 packed bf16 values (bf16x4_t).
__device__ __forceinline__ bf16x4_t bld8(__amdgpu_buffer_rsrc_t r, int off) {
    u32x2_t raw = __builtin_amdgcn_raw_buffer_load_b64(r, off, 0, 0);
    bf16x4_t o; __builtin_memcpy(&o, &raw, sizeof(o)); return o;
}

// Load 16 bytes (128-bit) from buffer resource `r` at byte offset `off`,
// reinterpret as 8 packed bf16 values (bf16x8_t).
__device__ __forceinline__ bf16x8_t bld16(__amdgpu_buffer_rsrc_t r, int off) {
    u32x4_t raw = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0);
    bf16x8_t o; __builtin_memcpy(&o, &raw, sizeof(o)); return o;
}

// Store 8 bytes (64-bit) to buffer resource `r` at byte offset `off` from
// 4 packed bf16 values (bf16x4_t).
__device__ __forceinline__ void bst8(__amdgpu_buffer_rsrc_t r, int off, bf16x4_t v) {
    u32x2_t b; __builtin_memcpy(&b, &v, 8);
    __builtin_amdgcn_raw_buffer_store_b64(b, r, off, 0, 0);
}

// s_waitcnt SIMM16 encoding on gfx942:
//   bits[3:0]   = vmcnt[3:0]   (0..15; full 6-bit encoding unused here)
//   bits[6:4]   = exp_cnt      (0..7)
//   bits[11:8]  = lgkmcnt[3:0] (0..15)
// "Any / no wait" = max value for that counter. So:
//   0x007F = lgkm=0, vmcnt=any, exp=any  → wait all LDS
//   0x0F7N = lgkm=any, vmcnt=N, exp=any  → wait vmem to N
__device__ __forceinline__ void wait_lgkm0() { __builtin_amdgcn_s_waitcnt(0x007F); }

__device__ __forceinline__ void lds_barrier() {
    wait_lgkm0();
    __builtin_amdgcn_s_barrier();
}
template <int N>
__device__ __forceinline__ void wait_vmcnt() {
    static_assert(N >= 0 && N <= 15);
    __builtin_amdgcn_s_waitcnt(0x0F70 | N);
}

// fp32 → bf16 RTZ pack. v_perm_b32 with selector 0x07060302 picks the two
// high bytes of each input dword (bf16 = high 16 bits of fp32), packing
// two bf16 into one dword. Matches AITER's RTZ output bit-for-bit.
__device__ __forceinline__ bf16x4_t pack4_bf16(float a, float b, float c, float d) {
    const unsigned ua = __builtin_bit_cast(unsigned, a);
    const unsigned ub = __builtin_bit_cast(unsigned, b);
    const unsigned uc = __builtin_bit_cast(unsigned, c);
    const unsigned ud = __builtin_bit_cast(unsigned, d);
    const unsigned w0 = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
    const unsigned w1 = __builtin_amdgcn_perm(ud, uc, 0x07060302u);
    bf16x4_t o;
    __builtin_memcpy(&o, &w0, 4);
    __builtin_memcpy((char*)&o + 4, &w1, 4);
    return o;
}

// Wave-64 XOR-butterfly reduction step via ds_bpermute. The intrinsic issues
// an async LDS-network read; we manually wait on lgkmcnt(0) before consuming
// the result. This mirrors what `__shfl_xor` lowers to, but gives us direct
// control over where the wait lands — so later work can be scheduled between
// the issue and the wait.
__device__ __forceinline__ int sxor_lane_addr(int mask) {
    const int lane = __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
    return (lane ^ mask) << 2;
}
__device__ __forceinline__ float sxor_max(float v, int mask) {
    const int peer = __builtin_amdgcn_ds_bpermute(sxor_lane_addr(mask), __builtin_bit_cast(int, v));
    wait_lgkm0();
    return fmaxf(v, __builtin_bit_cast(float, peer));
}
__device__ __forceinline__ float sxor_sum(float v, int mask) {
    const int peer = __builtin_amdgcn_ds_bpermute(sxor_lane_addr(mask), __builtin_bit_cast(int, v));
    wait_lgkm0();
    return v + __builtin_bit_cast(float, peer);
}

__device__ __forceinline__ void softmax_rescale(
    fp32x4_t sacc[], fp32x4_t oreg[], bf16x4_t preg[],
    float& row_m, float& row_l, float scale_log2e
) {
    float bmax = -FLT_MAX;
    #pragma unroll
    for (int sn = 0; sn < kNSub; ++sn)
        bmax = fmaxf(bmax, fmaxf(fmaxf(sacc[sn][0], sacc[sn][1]),
                                  fmaxf(sacc[sn][2], sacc[sn][3])));
    bmax = sxor_max(bmax, 16);
    bmax = sxor_max(bmax, 32);

    float new_m   = fmaxf(row_m, bmax * scale_log2e);
    float rescale = (row_l > 0.f) ? exp2c(row_m - new_m) : 0.f;
    float neg_m   = -new_m;

    #pragma unroll
    for (int t = 0; t < kDTiles; ++t) {
        oreg[t][0] *= rescale; oreg[t][1] *= rescale;
        oreg[t][2] *= rescale; oreg[t][3] *= rescale;
    }

    float bsum = 0.f;
    #pragma unroll
    for (int sn = 0; sn < kNSub; ++sn) {
        float p0 = exp2c(fmaf(sacc[sn][0], scale_log2e, neg_m));
        float p1 = exp2c(fmaf(sacc[sn][1], scale_log2e, neg_m));
        float p2 = exp2c(fmaf(sacc[sn][2], scale_log2e, neg_m));
        float p3 = exp2c(fmaf(sacc[sn][3], scale_log2e, neg_m));
        bsum += (p0 + p1) + (p2 + p3);
        preg[sn] = pack4_bf16(p0, p1, p2, p3);
    }
    bsum = sxor_sum(bsum, 16);
    bsum = sxor_sum(bsum, 32);
    row_m = new_m;
    row_l = row_l * rescale + bsum;
}

// ═══════════════════════════════════════════════════════════════════════
//  Attention forward — 4-wave, 16x16x16 MFMA, buffer_load_b128 pipeline
//  Each wave handles 2 × 16-row Q tiles (interleaved lo/hi).
// ═══════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(kBlockSize, 2)
void attention_forward(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16*       __restrict__ Out,
    int seq_len, int q_tiles, float scale_log2e
) {
    const int tid  = threadIdx.x;
    const int wid  = tid >> 6; // wave index
    const int lane = tid & 63;
    const int col  = lane & 15;        // 0..15, the 16-lane MFMA column
    const int quad = lane >> 4;        // 0..3, which 16-lane quad within the wave

    // Each CTA covers kNumWaves*2 = 8 consecutive 16-row Q tiles.
    // blockIdx.x selects the group of 8; wid*2 picks this wave's lo/hi pair
    // within that group (wave 0→tiles 0,1 ; wave 1→2,3 ; wave 2→4,5 ; wave 3→6,7).
    const int qtile_base = blockIdx.x * kNumWaves * 2 + wid * 2;
    // Bounds-check both tiles; the tail CTA may not have a full hi tile.
    const bool valid[2]  = {qtile_base < q_tiles, qtile_base + 1 < q_tiles};
    // Convert tile index → absolute first row index (tile × 16 rows/tile).
    // Fall back to row 0 for invalid tiles; those lanes are masked out later.
    const int  qbase[2]  = {valid[0] ? qtile_base * 16       : 0,
                            valid[1] ? (qtile_base + 1) * 16 : 0};
    const int bh         = blockIdx.y;

    const long long bhoff = (long long)bh * seq_len * kHeadDim;
    const auto qr  = mkr(Q + bhoff);
    const auto kr  = mkr(K + bhoff);
    const auto vr  = mkr(V + bhoff);
    const auto or_ = mkr(Out + bhoff);

    __shared__ bf16 lds_k[2][kNBlock][kLdsStride];
    __shared__ bf16 lds_v[4][kKTile][kLdsStride];

    // Q → registers (held for entire kernel). q[0]=lo tile, q[1]=hi tile.
    bf16x4_t q[2][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d)
            q[qg][d] = valid[qg]
                // Byte offset into Q for this lane's 4 bf16 elements (= 8 bytes):
                //   row          = qbase[qg] + col
                //                  col (lane & 15) maps the 16-lane column-group
                //                  onto the 16 rows of this MFMA tile.
                //   elem_in_row  = d * kKTile + 4 * quad
                //                  d selects the 16-wide d-tile within head_dim (0..kDTiles-1),
                //                  quad (lane >> 4, 0..3) selects which 4-element group within
                //                  that tile (at offsets 0, 4, 8, 12) — matching the MFMA
                //                  A-matrix lane layout where each lane owns 4 consecutive
                //                  K-dimension elements for one row.
                //   * kHeadDim   = row stride (128 bf16 elements per row)
                //   * 2          = bytes per bf16
                ? bld8(qr, ((qbase[qg] + col) * kHeadDim + d * kKTile + 4 * quad) * 2)
                : bf16x4_t{0,0,0,0};

    fp32x4_t o[2][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d) o[qg][d] = zero4();

    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0.f, 0.f};

    // K/V thread mapping (krow==vrow; both = tid>>2)
    const int kvrow = tid >> 2;           // 0..63
    const int kcol  = (tid & 3) * 16;     // 0,16,32,48
    const int vdim  = (tid & 3) * 8;      // 0,8,16,24

    // V d-tile/offset for transpose stores (precomputed)
    const int vdt0 = vdim >> 4,         vdo0 = vdim & 15;
    const int vdt1 = (32+vdim) >> 4,    vdo1 = (32+vdim) & 15;
    const int vdt2 = ((64+vdim)>>4)-4,  vdo2 = (64+vdim) & 15;
    const int vdt3 = ((96+vdim)>>4)-4,  vdo3 = (96+vdim) & 15;

    // K pipeline (buffer_load_b128): 2 loads per half-dim
    auto issue_k = [&](int nb, int h, bf16x8_t& a, bf16x8_t& b) {
        int base = (nb * kNBlock + kvrow) * kHeadDim * 2;
        int dim  = (h * kHalfDim + kcol) * 2;
        a = bld16(kr, base + dim);
        b = bld16(kr, base + dim + 16);
    };
    auto store_k = [&](int buf, bf16x8_t a, bf16x8_t b) {
        *reinterpret_cast<bf16x8_t*>(&lds_k[buf][kvrow][kcol])     = a;
        *reinterpret_cast<bf16x8_t*>(&lds_k[buf][kvrow][kcol + 8]) = b;
    };

    // V transpose store: write 8 consecutive head_dim bytes from one thread as
    // 8 scattered ds_write_b16 to column kvrow of the dv-th LDS slab. The
    // source is a short-vector (bf16x8_t); bit-cast each lane to bf16 (bf16 is
    // a class type, so vector-element reference is illegal).
    auto vstore8 = [&](int dt, int d0, bf16x8_t src) {
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            lds_v[dt][d0 + i][kvrow] = __builtin_bit_cast(bf16, (short)src[i]);
    };

    const int nblocks = seq_len / kNBlock;

    // Prologue: K[0] → LDS
    {
        bf16x8_t k0a, k0b, k1a, k1b;
        issue_k(0, 0, k0a, k0b);
        issue_k(0, 1, k1a, k1b);
        wait_vmcnt<0>();
        store_k(0, k0a, k0b);
        store_k(1, k1a, k1b);
    }
    lds_barrier();

    // Main K-block loop. Pipelining DAG per iteration:
    //
    //   [PREFETCH]  issue K[nb+1] (4× b128), V[nb] as V_a + V_b (4× b128)
    //   [QK]        S = Q · K[nb]^T                   — 64 MFMAs, reads lds_k
    //   [SOFTMAX]   online-rescale O, produce P       — wave-reductions + exp2
    //   [STORE V_a] wait_vmcnt<2>, transpose V_a → lds_v ; stash K[nb+1] → lds_k
    //   [PV half 0] O[0..3] += P · V_a                — 32 MFMAs, reads lds_v
    //   [STORE V_b] wait_vmcnt<0>, transpose V_b → lds_v
    //   [PV half 1] O[4..7] += P · V_b                — 32 MFMAs, reads lds_v
    //
    // The V global loads are hoisted to the top so their vmcnt drains during the
    // QK phase; V_b is held until after PV half 0 so lds_v can be reused.
    for (int nb = 0; nb < nblocks; ++nb) {
        const bool has_next = (nb + 1 < nblocks);

        // ── PREFETCH: K[nb+1] and all of V[nb] ──────────────────────────────
        bf16x8_t kn0a{}, kn0b{}, kn1a{}, kn1b{};
        if (has_next) {
            issue_k(nb + 1, 0, kn0a, kn0b);
            issue_k(nb + 1, 1, kn1a, kn1b);
        }
        const int vbase = (nb * kNBlock + kvrow) * kHeadDim * 2;
        bf16x8_t va0 = bld16(vr, vbase + (     vdim) * 2);
        bf16x8_t va1 = bld16(vr, vbase + (32 + vdim) * 2);
        bf16x8_t vb0 = bld16(vr, vbase + (64 + vdim) * 2);
        bf16x8_t vb1 = bld16(vr, vbase + (96 + vdim) * 2);

        // ── QK: S = Q · K[nb]^T ─────────────────────────────────────────────
        // sched_group_barriers cluster (4 ds_reads, 8 MFMAs) ×8 — load-bearing
        // on gfx942, see feedback_ablation_results.md (+18 TFLOP/s).
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, 8, 0);
        }

        fp32x4_t s[2][kNSub];
        #pragma unroll
        for (int qg = 0; qg < 2; ++qg)
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn) s[qg][sn] = zero4();

        // K is split across lds_k[0]=low-half, lds_k[1]=high-half: d>>2 picks
        // the slab, d&3 picks the d-tile within that slab.
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d) {
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn) {
                bf16x4_t kf = *reinterpret_cast<const bf16x4_t*>(
                    &lds_k[d >> 2][sn * kKTile + col][(d & 3) * kKTile + 4 * quad]);
                #pragma unroll
                for (int qg = 0; qg < 2; ++qg)
                    s[qg][sn] = mfma(kf, q[qg][d], s[qg][sn]);
            }
        }

        // ── SOFTMAX: online rescale O, produce P ────────────────────────────
        bf16x4_t p[2][kNSub];
        #pragma unroll
        for (int qg = 0; qg < 2; ++qg)
            softmax_rescale(s[qg], o[qg], p[qg], m[qg], l[qg], scale_log2e);

        // ── STORE V_a → lds_v ; stash prefetched K[nb+1] ────────────────────
        // Drain the 6 oldest vmem ops (K[nb+1]×4 + V_a×2), keep V_b×2 in flight.
        wait_vmcnt<2>();
        vstore8(vdt0, vdo0, va0);
        vstore8(vdt1, vdo1, va1);
        lds_barrier();
        if (has_next) { store_k(0, kn0a, kn0b); store_k(1, kn1a, kn1b); }

        // ── PV: O += P · V, in two halves ───────────────────────────────────
        // lds_v is reused across both halves, so between them we drain phase-1
        // ds_reads (lgkmcnt) AND the pending V_b global loads (vmcnt) before
        // overwriting. The two halves otherwise execute the same MFMA pattern.
        #pragma unroll
        for (int vh = 0; vh < 2; ++vh) {
            if (vh == 1) {
                lds_barrier();
                wait_vmcnt<0>();
                vstore8(vdt2, vdo2, vb0);
                vstore8(vdt3, vdo3, vb1);
                lds_barrier();
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 8, 0);
            }
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn) {
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) {
                    bf16x4_t vf = *reinterpret_cast<const bf16x4_t*>(
                        &lds_v[dv][col][sn * kKTile + 4 * quad]);
                    #pragma unroll
                    for (int qg = 0; qg < 2; ++qg)
                        o[qg][vh * 4 + dv] = mfma(vf, p[qg][sn], o[qg][vh * 4 + dv]);
                }
            }
        }
    }

    // Epilogue: bf16 output (RTZ pack) via b64 stores
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg) {
        if (!valid[qg]) continue;
        const float inv = rcpf(l[qg]);
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            bf16x4_t out = pack4_bf16(o[qg][t][0]*inv, o[qg][t][1]*inv,
                                       o[qg][t][2]*inv, o[qg][t][3]*inv);
            bst8(or_, ((qbase[qg] + col) * kHeadDim + t * kKTile + 4 * quad) * 2, out);
        }
    }
}

} // namespace

extern "C" int launch_attention_forward(
    const void* q, const void* k, const void* v, void* out,
    int batch, int heads, int seq_len, int head_dim, void* stream
) {
    if (batch <= 0 || heads <= 0) return 1;
    if (seq_len <= 0 || (seq_len % kNBlock) != 0) return 2;
    if (head_dim != kHeadDim) return 3;

    const int bh = batch * heads;
    const int qt = seq_len / 16;
    const dim3 grid((qt + kNumWaves * 2 - 1) / (kNumWaves * 2), bh);
    const float sl2e = kLog2e / sqrtf((float)kHeadDim);

    hipLaunchKernelGGL(attention_forward, grid, dim3(kBlockSize), 0, (hipStream_t)stream,
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out,
        seq_len, qt, sl2e);
    return (int)hipGetLastError();
}
