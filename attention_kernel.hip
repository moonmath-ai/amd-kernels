// Forward attention (bf16, gfx942 / MI300X), FA2-style online softmax.
//
// Block: 4 waves × 64 threads, occupancy 2 (8 waves/CU). 16×16×16 MFMA tiles,
// head_dim = 128. Each wave handles 2 × 16-row Q tiles (lo/hi), so a CTA
// covers 8 consecutive tiles along the Q axis. Output and softmax-P pack to
// bf16 with rounding selected at compile time: -DBF16_ROUND=RTNE (default,
// lower error) or RTZ (cheaper, no bias add).
//
// LDS layout:
//   lds_k[2][64][68]  double-buffered K block, slab 0 = low-half-dim,
//                      slab 1 = high-half-dim.
//   lds_v[4][16][68]  single-buffered V slabs (4 × 16-row tiles). Each
//                      iteration first holds V_a (head_dim rows 0..63) for
//                      PV vh=0; V_b (rows 64..127) then overwrites it in
//                      two staged vstores for PV vh=1.
//
// Per K-block iteration:
//   1. Prefetch K[nb+1] (4× b128) and V[nb] = V_a + V_b (4× b128).
//   2. QK: S = K[nb] · Qᵀ                              — 64 MFMAs, reads lds_k.
//   3. wait_vmcnt<2>; vstore V_a → lds_v               — hidden under softmax.
//   4. Softmax: rescale O, produce P                   — cross-lane via LDS.
//   5. Barrier; stash K[nb+1] → lds_k; wait_vmcnt<0>.
//   6. PV vh=0 (dv=0..3): O[0..3] += V_aᵀ · Pᵀ; V_b vstores interleaved:
//        stage A: read V_a slabs 0,1; barrier; vstore V_b vdt2 → slabs 0,1.
//        stage B: read V_a slabs 2,3; barrier; vstore V_b vdt3 → slabs 2,3.
//   7. PV vh=1 (dv=0..3 → o[4..7]): O[4..7] += V_bᵀ · Pᵀ:
//        stage A: read V_b slabs 0,1 (vdt2). Barrier drains vdt3 writes.
//        stage B: read V_b slabs 2,3 (vdt3).
//
// 4 lds_barriers per iter. K vstore retires under PV vh=0 stage A; V_b
// vstores ride under PV vh=0's MFMA shadow.

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
constexpr int kNSub      = 4;     // 4 MFMA sub-tiles along the KV-row dim per K-block (kNBlock / kKTile)
constexpr int kKTile     = 16;    // 16-row MFMA tile
constexpr int kHeadDim   = 128;
constexpr int kDTiles    = kHeadDim / kKTile;
constexpr int kHalfDim   = kHeadDim / 2;
constexpr int kLdsStride = 68;
constexpr float kLog2e   = 1.4426950408889634f;
constexpr int kBarrierLgkm = 0;

// fp32 → bf16 rounding mode, compile-time selectable. RTNE adds a bias before
// the high-16 truncation (1 ULP → 0.5 ULP max error); RTZ truncates directly.
enum class BF16Round { RTNE, RTZ };
#ifndef BF16_ROUND
#define BF16_ROUND RTNE
#endif
inline constexpr BF16Round kBf16Round = BF16Round::BF16_ROUND;

__device__ __forceinline__ fp32x4_t zero4() { return fp32x4_t{0,0,0,0}; }
__device__ __forceinline__ float    exp2c(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float    rcpf(float x)  { return __builtin_amdgcn_rcpf(x); }
__device__ __forceinline__ float    sxor(float v, int m) { return __shfl_xor(v, m, 64); }
__device__ __forceinline__ fp32x4_t mfma(bf16x4_t a, bf16x4_t b, fp32x4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Raw-buffer descriptor: stride=0, bound=0xFFFFFFFF (kernel logic guards bounds),
// config dword = 0x00020000.
constexpr uint32_t kBufCfg = 0x00020000u;
__device__ __forceinline__ __amdgpu_buffer_rsrc_t mkr(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, 0xFFFFFFFFu, kBufCfg);
}

__device__ __forceinline__ bf16x4_t bld8(__amdgpu_buffer_rsrc_t r, int off) {
    u32x2_t raw = __builtin_amdgcn_raw_buffer_load_b64(r, off, 0, 0);
    bf16x4_t o; __builtin_memcpy(&o, &raw, sizeof(o)); return o;
}
__device__ __forceinline__ bf16x8_t bld16(__amdgpu_buffer_rsrc_t r, int off) {
    u32x4_t raw = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0);
    bf16x8_t o; __builtin_memcpy(&o, &raw, sizeof(o)); return o;
}
__device__ __forceinline__ void bst8(__amdgpu_buffer_rsrc_t r, int off, bf16x4_t v) {
    u32x2_t b; __builtin_memcpy(&b, &v, 8);
    __builtin_amdgcn_raw_buffer_store_b64(b, r, off, 0, 0);
}

// s_waitcnt SIMM16 encoding on gfx942: lgkmcnt = bits[11:8], exp = bits[6:4],
// vmcnt[3:0] = bits[3:0]. Max value per field means "no wait".
//   0x007F = lgkm=0, vmcnt=any, exp=any   → drain all LDS
//   0x0F70 | N = lgkm=any, vmcnt=N        → leave N vmem ops in flight
__device__ __forceinline__ void wait_lgkm0() { __builtin_amdgcn_s_waitcnt(0x007F); }
template <int N>
__device__ __forceinline__ void wait_lgkm() {
    static_assert(N >= 0 && N <= 15);
    __builtin_amdgcn_s_waitcnt(0x007F | (N << 8));
}

__device__ __forceinline__ void lds_barrier() {
    wait_lgkm<kBarrierLgkm>();
    __builtin_amdgcn_s_barrier();
}
template <int N>
__device__ __forceinline__ void wait_vmcnt() {
    static_assert(N >= 0 && N <= 15);
    __builtin_amdgcn_s_waitcnt(0x0F70 | N);
}

// fp32 → bf16 with kBf16Round. RTNE: u += 0x7FFF + (bit16 of u) before taking
// the high 16. RTZ: take the high 16 of u directly. v_perm_b32 with mask
// 0x07060302 packs two rounded fp32s into a 32-bit dword of two bf16s.
__device__ __forceinline__ unsigned bf16_round(float f) {
    const unsigned u = __builtin_bit_cast(unsigned, f);
    if constexpr (kBf16Round == BF16Round::RTNE) {
        const unsigned bias = 0x7FFFu + ((u >> 16) & 1u);
        return u + bias;
    } else {
        return u;
    }
}
__device__ __forceinline__ bf16x4_t pack4_bf16(float a, float b, float c, float d) {
    const unsigned ua = bf16_round(a);
    const unsigned ub = bf16_round(b);
    const unsigned uc = bf16_round(c);
    const unsigned ud = bf16_round(d);
    const unsigned w0 = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
    const unsigned w1 = __builtin_amdgcn_perm(ud, uc, 0x07060302u);
    bf16x4_t o;
    __builtin_memcpy(&o, &w0, 4);
    __builtin_memcpy((char*)&o + 4, &w1, 4);
    return o;
}

// Wave-64 XOR-butterfly reduction over the LDS pipe (gfx942 has no permlane).
// xor-16 uses ds_swizzle_b32 with imm 0x401F (and=0x1F, xor=0x10) — the
// selector is baked into the instruction, no address VGPR. xor-32 falls back
// to ds_bpermute since ds_swizzle is restricted to 32-lane groups.
__device__ __forceinline__ int sxor_lane() {
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}
__device__ __forceinline__ float sxor_max(float v, int mask) {
    const int v_i = __builtin_bit_cast(int, v);
    int peer;
    if (mask == 16) {
        peer = __builtin_amdgcn_ds_swizzle(v_i, 0x401F);
    } else {
        peer = __builtin_amdgcn_ds_bpermute((sxor_lane() ^ mask) << 2, v_i);
    }
    wait_lgkm0();
    return fmaxf(v, __builtin_bit_cast(float, peer));
}
__device__ __forceinline__ float sxor_sum(float v, int mask) {
    const int v_i = __builtin_bit_cast(int, v);
    int peer;
    if (mask == 16) {
        peer = __builtin_amdgcn_ds_swizzle(v_i, 0x401F);
    } else {
        peer = __builtin_amdgcn_ds_bpermute((sxor_lane() ^ mask) << 2, v_i);
    }
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
    float rescale = exp2c(row_m - new_m);
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

    // CTA covers 8 consecutive 16-row Q tiles; this wave owns tiles (wid*2, wid*2+1).
    // The tail CTA may not have a full hi tile — masked out by `valid`.
    const int qtile_base = blockIdx.x * kNumWaves * 2 + wid * 2;
    const bool valid[2]  = {qtile_base < q_tiles, qtile_base + 1 < q_tiles};
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

    // Q → registers, held for the whole kernel. MFMA B-operand layout:
    // lane (col, quad) owns row (qbase + col), 4 consecutive d-elements at
    // (d * kKTile + 4 * quad). q[0]=lo tile, q[1]=hi tile.
    bf16x4_t q[2][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d)
            q[qg][d] = valid[qg]
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

    // V transpose store: 8 consecutive head_dim values from one thread go to
    // 8 scattered ds_write_b16 down column `kvrow` of slab `dt`. Bit-cast each
    // vector element to bf16 (bf16 is a class type, no element reference).
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

        // ── QK: S = K[nb] · Qᵀ  (A=kf, B=q) ──────────────────────────────────
        // sched_group_barriers cluster (4 ds_reads, 8 MFMAs) ×8.
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

        // ── Vstore V_a → lds_v ──────────────────────────────────────────────
        // Drain the 6 oldest vmem ops (4× K[nb+1] + 2× V_a), keep V_b×2 in flight.
        // The ds_writes retire under the upcoming softmax VALU/LDS work.
        wait_vmcnt<2>();
        vstore8(vdt0, vdo0, va0);
        vstore8(vdt1, vdo1, va1);

        // ── SOFTMAX: online rescale O, produce P ────────────────────────────
        bf16x4_t p[2][kNSub];
        #pragma unroll
        for (int qg = 0; qg < 2; ++qg)
            softmax_rescale(s[qg], o[qg], p[qg], m[qg], l[qg], scale_log2e);

        // ── Publish V_a; stash prefetched K[nb+1] ───────────────────────────
        // K vstore retires under PV vh=0 stage A. V_b vmem must be drained here
        // because its first ds_write (vdt2) happens between stages A and B below.
        lds_barrier();
        if (has_next) { store_k(0, kn0a, kn0b); store_k(1, kn1a, kn1b); }
        wait_vmcnt<0>();

        asm volatile("s_setprio 1");

        // ── PV vh=0 STAGE A: dv=0,1 → consumes V_a slabs 0,1 ────────────────
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // 2 ds_reads
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);  // 4 mfmas
        }
        #pragma unroll
        for (int sn = 0; sn < kNSub; ++sn) {
            #pragma unroll
            for (int dv = 0; dv < 2; ++dv) {
                bf16x4_t vf = *reinterpret_cast<const bf16x4_t*>(
                    &lds_v[dv][col][sn * kKTile + 4 * quad]);
                #pragma unroll
                for (int qg = 0; qg < 2; ++qg)
                    o[qg][dv] = mfma(vf, p[qg][sn], o[qg][dv]);
            }
        }

        // This wave finished slabs 0,1, but other waves may still be in stage A.
        // Barrier before vdt2 overwrites them with V_b.
        lds_barrier();
        vstore8(vdt2, vdo2, vb0);

        // ── PV vh=0 STAGE B: dv=2,3 → consumes V_a slabs 2,3 ────────────────
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            __builtin_amdgcn_sched_group_barrier(0x040, 4, 0);  // 4 ds_writes (V_b)
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // 2 ds_reads
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);  // 4 mfmas
        }
        #pragma unroll
        for (int sn = 0; sn < kNSub; ++sn) {
            #pragma unroll
            for (int dv = 2; dv < 4; ++dv) {
                bf16x4_t vf = *reinterpret_cast<const bf16x4_t*>(
                    &lds_v[dv][col][sn * kKTile + 4 * quad]);
                #pragma unroll
                for (int qg = 0; qg < 2; ++qg)
                    o[qg][dv] = mfma(vf, p[qg][sn], o[qg][dv]);
            }
        }

        // Wait for other waves to finish reading slabs 2,3 before vdt3 overwrites them.
        lds_barrier();
        vstore8(vdt3, vdo3, vb1);

        // ── PV vh=1 STAGE A: dv=0,1 reads slabs 0,1 (V_b vdt2, already drained).
        //    Independent of vdt3 writes still in flight.
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            __builtin_amdgcn_sched_group_barrier(0x040, 4, 0);  // 4 ds_writes (vdt3)
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // 2 ds_reads
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);  // 4 mfmas
        }
        #pragma unroll
        for (int sn = 0; sn < kNSub; ++sn) {
            #pragma unroll
            for (int dv = 0; dv < 2; ++dv) {
                bf16x4_t vf = *reinterpret_cast<const bf16x4_t*>(
                    &lds_v[dv][col][sn * kKTile + 4 * quad]);
                #pragma unroll
                for (int qg = 0; qg < 2; ++qg)
                    o[qg][4 + dv] = mfma(vf, p[qg][sn], o[qg][4 + dv]);
            }
        }

        // Drain vdt3 writes before stage B reads slabs 2,3.
        lds_barrier();

        // ── PV vh=1 STAGE B: dv=2,3 reads slabs 2,3 (V_b vdt3) ──────────────
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);
        }
        #pragma unroll
        for (int sn = 0; sn < kNSub; ++sn) {
            #pragma unroll
            for (int dv = 2; dv < 4; ++dv) {
                bf16x4_t vf = *reinterpret_cast<const bf16x4_t*>(
                    &lds_v[dv][col][sn * kKTile + 4 * quad]);
                #pragma unroll
                for (int qg = 0; qg < 2; ++qg)
                    o[qg][4 + dv] = mfma(vf, p[qg][sn], o[qg][4 + dv]);
            }
        }
        asm volatile("s_setprio 0");
    }

    // Epilogue: normalize O by 1/l, pack to bf16 (RTNE/RTZ), b64 store.
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
