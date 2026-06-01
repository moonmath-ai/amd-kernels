// Forward attention for AMD CDNA3 (MI300X / gfx942), bf16, FA2-style online softmax.
// 32x32x8 bf16 MFMA. 8-wave warp-spec CTA, two groups of 4 waves; per K-tile each
// iter runs as two phases:
//
//                    Group A                      Group B
//   ───────  ────────────────────────  ────────────────────────────
//   Phase 1  pv[N-1] + qk[N]           K[N+1] HBM→LDS + softmax[N-1]
//   Phase 2  softmax[N] + V_t prefetch  pv[N-1] + qk[N]
//
// One group keeps the SIMD's MFMA engine active while the other does VALU+memory.
// L1V variant: V is pre-transposed offline (v_transpose_kernel) into V_t laid
// out for direct bld16 consumption in pv_phase — no LDS staging for V.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace {

using bf16_t   = __hip_bfloat16;
using bf16x4_t = short   __attribute__((ext_vector_type(4)));  // 4 bf16 = 8 bytes
// RTNE/RTZ/RTNA bf16 rounding selected at compile time via -DBF16_ROUND.
#ifndef BF16_ROUND
#define BF16_ROUND RTZ
#endif
enum class BF16Round { RTNE, RTZ, RTNA };
constexpr BF16Round kBf16Round = BF16Round::BF16_ROUND;
using fp32x16_t = float  __attribute__((ext_vector_type(16)));
using u32x2_t  = uint32_t __attribute__((ext_vector_type(2)));
using u32x4_t  = uint32_t __attribute__((ext_vector_type(4)));

constexpr int   kWaveSize       = 64;
constexpr int   kNumWaves       = 8;
constexpr int   kLoadWaves      = 4;
constexpr int   kBlockSize      = kWaveSize * kNumWaves;
constexpr int   kNBlock         = 64;
constexpr int   kKvTile         = 32;          // 32x32x8 mfma kv-tile
constexpr int   kNKvTiles       = kNBlock / kKvTile;     // 2
constexpr int   kKInner         = 8;           // mfma K=8
constexpr int   kHeadDim        = 128;
constexpr int   kDChunks        = kHeadDim / kKInner;    // 16
constexpr int   kQRows          = 32;          // per wave: 1 mfma covers 32 q rows
constexpr int   kHdTiles        = kHeadDim / kKvTile;    // 4 hd-tiles per pv loop
constexpr int   kKLdsStrideDw   = 64;          // 1 full K row per LDS row, swizzled
constexpr int   kKRowsPerLW     = kNBlock / kLoadWaves;  // 16
constexpr float kLog2e          = 1.4426950408889634f;
constexpr uint32_t kBufCfg      = 0x00020000u;

__device__ __forceinline__ fp32x16_t zero16() {
    fp32x16_t z;
    #pragma unroll
    for (int i = 0; i < 16; ++i) z[i] = 0.f;
    return z;
}

__device__ __forceinline__ float exp2c(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float rcpf (float x) { return __builtin_amdgcn_rcpf (x); }

__device__ __forceinline__ fp32x16_t mfma(bf16x4_t a, bf16x4_t b, fp32x16_t c) {
    return __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, 0, 0);
}

// Inline-asm mfma: forces in/out same VGPR for accumulator (no compiler rename
// → no v_mov copies) and keeps issue in source order via asm-volatile.
__device__ __forceinline__ void asm_mfma(bf16x4_t a, bf16x4_t b, fp32x16_t& c) {
    asm volatile("v_mfma_f32_32x32x8_bf16 %0, %1, %2, %0"
                 : "+v"(c) : "v"(a), "v"(b));
}

// Literal-0 C operand skips the v_mov_b64 init the compiler emits to zero a
// fresh accumulator. "=&v" prevents aliasing dst onto the A VGPRs.
__device__ __forceinline__ fp32x16_t asm_mfma_zero(bf16x4_t a, bf16x4_t b) {
    fp32x16_t r;
    asm volatile("v_mfma_f32_32x32x8_bf16 %0, %1, %2, 0"
                 : "=&v"(r) : "v"(a), "v"(b));
    return r;
}

// Scalar v_add/v_mul: avoids compiler-natural v_pk_*_f32 sharing writeback
// ports with MFMA on the same wave-pair.
__device__ __forceinline__ float vadd_f32(float a, float b) {
    float r; asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ float vmul_f32(float a, float b) {
    float r; asm volatile("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

// bf16 packing. MaybeNaN=true guards against 0×inf NaN with a __builtin_isunordered
// branch (compiler can interleave the v_cmp_u with mfma, unlike inline asm).
template <bool MaybeNaN>
__device__ __forceinline__ unsigned bf16_round_one(float f) {
    if constexpr (kBf16Round == BF16Round::RTNE) {
        const unsigned u      = __builtin_bit_cast(unsigned, f);
        const unsigned biased = u + 0x7FFFu + ((u >> 16) & 1u);
        if constexpr (MaybeNaN) {
            unsigned out = biased;
            asm volatile(
                "v_cmp_u_f32_e64 vcc, %1, %1\n\t"
                "v_cndmask_b32_e32 %0, %0, %2, vcc"
                : "+v"(out)
                : "v"(f), "v"(0x7FFF0000u)
                : "vcc"
            );
            return out;
        }
        else                    return biased;
    } else if constexpr (kBf16Round == BF16Round::RTNA) {
        const unsigned u      = __builtin_bit_cast(unsigned, f);
        const unsigned biased = u + 0x8000u;
        if constexpr (MaybeNaN) {
            unsigned out = biased;
            asm volatile(
                "v_cmp_u_f32_e64 vcc, %1, %1\n\t"
                "v_cndmask_b32_e32 %0, %0, %2, vcc"
                : "+v"(out)
                : "v"(f), "v"(0x7FFF0000u)
                : "vcc"
            );
            return out;
        }
        else                    return biased;
    } else {
        return __builtin_bit_cast(unsigned, f);  // RTZ — top 16 bits
    }
}
template <bool MaybeNaN = false>
__device__ __forceinline__ bf16x4_t pack4_bf16(float a, float b, float c, float d) {
    const unsigned ua = bf16_round_one<MaybeNaN>(a), ub = bf16_round_one<MaybeNaN>(b);
    const unsigned uc = bf16_round_one<MaybeNaN>(c), ud = bf16_round_one<MaybeNaN>(d);
    const unsigned w0 = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
    const unsigned w1 = __builtin_amdgcn_perm(ud, uc, 0x07060302u);
    bf16x4_t o;
    __builtin_memcpy(&o,            &w0, 4);
    __builtin_memcpy((char*)&o + 4, &w1, 4);
    return o;
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t mkr(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, 0xFFFFFFFFu, kBufCfg);
}
__device__ __forceinline__ bf16x4_t bld8(__amdgpu_buffer_rsrc_t r, int off) {
    u32x2_t raw = __builtin_amdgcn_raw_buffer_load_b64(r, off, 0, 0);
    bf16x4_t out; __builtin_memcpy(&out, &raw, sizeof(out)); return out;
}
__device__ __forceinline__ u32x4_t bld16(__amdgpu_buffer_rsrc_t r, int off) {
    return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0);
}
__device__ __forceinline__ void bst8(__amdgpu_buffer_rsrc_t r, int off, bf16x4_t v) {
    u32x2_t bits; __builtin_memcpy(&bits, &v, 8);
    __builtin_amdgcn_raw_buffer_store_b64(bits, r, off, 0, 0);
}

// ds_read2_b64 offset1:16 = +128 B. "=&v" prevents dst↔base aliasing.
__device__ __forceinline__ u32x4_t ds_read2_b64_off_0_16(uint32_t base) {
    u32x4_t r;
    asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:16" : "=&v"(r) : "v"(base));
    return r;
}

// Pinned lgkmcnt wait — anchors the consumer so the compiler can't reorder above.
#define WAIT_LGKM_CASE(N) \
    if constexpr (kN == N) asm volatile("s_waitcnt lgkmcnt(" #N ")" : "+v"(r) :: "memory");
template <int kN, typename T>
__device__ __forceinline__ void wait_lgkm(T& r) {
    static_assert(kN >= 0 && kN <= 7, "kN out of range");
    WAIT_LGKM_CASE(0) else WAIT_LGKM_CASE(1) else WAIT_LGKM_CASE(2) else
    WAIT_LGKM_CASE(3) else WAIT_LGKM_CASE(4) else WAIT_LGKM_CASE(5) else
    WAIT_LGKM_CASE(6) else WAIT_LGKM_CASE(7);
}
#undef WAIT_LGKM_CASE

__device__ __forceinline__ void wait_vmcnt0() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_s_waitcnt(0x007F);  // lgkmcnt(0)
    __builtin_amdgcn_s_barrier();
}
__device__ __forceinline__ int wave_lane_id() {
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

// HBM→LDS direct dword load. Inline-asm `s_mov_b32 m0` (not s_add chain) so the
// compiler sees m0 as a fresh write per call — Clang's cselect pattern around
// the builtin form can substitute m0=-1 when lds_dest is small, silently
// dropping writes.
__device__ __forceinline__ void buffer_load_b32_to_lds(
    __amdgpu_buffer_rsrc_t rsrc, uint32_t lds_dest,
    uint32_t voff, uint32_t soff
) {
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "buffer_load_dword %1, %2, %3 offen lds"
        :
        : "s"(lds_dest), "v"(voff), "s"(rsrc), "s"(soff)
        : "memory"
    );
}

// Head-first chiplet swizzle: chunk_size = blocks_per_head co-locates all
// q-blocks of one (B,H) on a single XCD's L2 → K+V stay resident.
constexpr int kNumXcds = 8;
__device__ __forceinline__ int chiplet_swizzle(int wgid, int num_wgs, int chunk_size) {
    const int block = kNumXcds * chunk_size;
    const int limit = (num_wgs / block) * block;
    if (wgid >= limit) return wgid;
    const int xcd          = wgid % kNumXcds;
    const int local_pid    = wgid / kNumXcds;
    const int chunk_idx    = local_pid / chunk_size;
    const int pos_in_chunk = local_pid % chunk_size;
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

// Cross-lane reduce L ↔ L^32: lanes (col,0)/(col,1) share q=col but cover
// disjoint kv halves. Explicit lgkm(0) wait avoids a race when ds_bpermute is
// the only LGKM op in flight (compiler may omit the auto-inserted wait).
__device__ __forceinline__ float xor32_max(float v) {
    const int lane = wave_lane_id();
    const int p = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, v));
    __builtin_amdgcn_s_waitcnt(0x007F);
    return fmaxf(v, __builtin_bit_cast(float, p));
}
__device__ __forceinline__ float xor32_sum(float v) {
    const int lane = wave_lane_id();
    const int p = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, v));
    __builtin_amdgcn_s_waitcnt(0x007F);
    return v + __builtin_bit_cast(float, p);
}

__global__ __launch_bounds__(kBlockSize, 1)
void attention_forward(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ K,
    const bf16_t* __restrict__ V,    // points to V_t (pre-transposed)
    bf16_t* __restrict__ Out,
    int seq_len, float scale_log2e
) {
    const int tid  = threadIdx.x;
    const int wid  = tid >> 6;
    const int lane = tid & 63;
    const int col      = lane & 31;
    const int k_outer  = lane >> 5;
    const bool group_a = (wid < kLoadWaves);
    const int wid_in_g = group_a ? wid : (wid - kLoadWaves);

    const int wgid_in   = blockIdx.y * gridDim.x + blockIdx.x;
    const int total_wgs = gridDim.x * gridDim.y;
    const int wgid      = chiplet_swizzle(wgid_in, total_wgs, gridDim.x);
    const int sw_bx     = wgid % gridDim.x;
    const int sw_by     = wgid / gridDim.x;

    const int q_tiles = seq_len / kQRows;
    const bool valid  = ((sw_bx * kNumWaves + wid) < q_tiles);
    const int q_my    = valid ? (sw_bx * kNumWaves * kQRows + wid * kQRows) : 0;

    const long long bhoff = (long long)sw_by * seq_len * kHeadDim;
    const auto qr  = mkr(Q + bhoff);
    const auto kr  = mkr(K + bhoff);
    const auto vr  = mkr(V + bhoff);
    const auto or_ = mkr(Out + bhoff);

    __shared__ uint32_t lds_k[2][kNBlock * kKLdsStrideDw];   // 32 KB (2-buf)
    // V LDS staging removed — V_t is consumed directly from L1 via bld16.

    // Prologue: K[0] → LDS (group B only). Group A has no work until iter 1.
    if (!group_a) {
        const uint32_t lds_base = __builtin_amdgcn_readfirstlane(
            (uint32_t)(uintptr_t)&lds_k[0][0]
            + (uint32_t)(wid_in_g * kKRowsPerLW * kKLdsStrideDw * 4));
        const uint32_t soff = 0;
        #pragma unroll
        for (int c = 0; c < kKRowsPerLW; ++c) {
            const int row = wid_in_g * kKRowsPerLW + c;
            const int swizzle = (row & 0xF) << 1;
            const uint32_t voff = (uint32_t)(row * kHeadDim * 2 + (lane ^ swizzle) * 4);
            buffer_load_b32_to_lds(kr, lds_base + c * (kKLdsStrideDw * 4), voff, soff);
        }
    }

    // Persistent VGPR Q (no reload across K-blocks).
    bf16x4_t qreg[kDChunks];
    #pragma unroll
    for (int dc = 0; dc < kDChunks; ++dc) {
        const int q_byte = (q_my + col) * kHeadDim * 2 + (dc * kKInner + k_outer * 4) * 2;
        qreg[dc] = valid ? bld8(qr, q_byte) : bf16x4_t{0,0,0,0};
    }

    fp32x16_t oreg[kHdTiles];
    #pragma unroll
    for (int h = 0; h < kHdTiles; ++h) oreg[h] = zero16();
    float row_m = -FLT_MAX, row_l = 0.f;

    const uint32_t lane_byte_off = (uint32_t)(lane * 16);

    // K direct-LDS loader (group B). Swizzle `(row & 0xF) << 1` shifts LDS bank
    // collisions to 16-apart lanes so the LDS arbiter pipelines them.
    auto direct_lds_k = [&](int nb, int kbuf) {
        const uint32_t lds_base = __builtin_amdgcn_readfirstlane(
            (uint32_t)(uintptr_t)&lds_k[kbuf][0]
            + (uint32_t)(wid_in_g * kKRowsPerLW * kKLdsStrideDw * 4));
        const uint32_t soff = __builtin_amdgcn_readfirstlane(
            (uint32_t)((uint64_t)nb * kNBlock * kHeadDim * 2));
        #pragma unroll
        for (int c = 0; c < kKRowsPerLW; ++c) {
            const int row = wid_in_g * kKRowsPerLW + c;
            const int swizzle = (row & 0xF) << 1;
            const uint32_t voff = (uint32_t)(row * kHeadDim * 2 + (lane ^ swizzle) * 4);
            buffer_load_b32_to_lds(kr, lds_base + c * (kKLdsStrideDw * 4), voff, soff);
        }
    };

    // ─── QK depth-3 ping-pong ──────────────────────────────────────────────
    // S = K × Qᵀ via mfma(A=K, B=Q, C=sacc). 2 kv-tiles × 16 d-chunks per
    // K-block. K reads use ds_read2_b64 (dc, dc+8) with depth-3 prefetch:
    // s0/s1/s2 are pre-issued by `prefetch_qk` BEFORE pv_phase so their LDS
    // reads overlap with pv's bld16 tail — single lgkm(0) drain at qk end.
    const int qk_sw_dw = (col & 0xF) << 1;

    auto qk_addr_at = [&](uint32_t k_byte_base, int i) -> uint32_t {
        const int t = i / (kDChunks / 2), dc = i % (kDChunks / 2);
        const int dw = (t * kKvTile + col) * kKLdsStrideDw + ((dc * 4 + k_outer * 2) ^ qk_sw_dw);
        return k_byte_base + (uint32_t)(dw * 4);
    };

    // Caller-issued QK LDS prefetch: fires 3 ds_read2_b64 BEFORE pv_phase so
    // their lgkmcnt counter advances independently of pv's vmcnt-tracked
    // bld16 — qk_phase sees s0/s1/s2 already in flight or arrived. The
    // sched_barrier(0) bracket LOCKS placement (compiler cannot sink the
    // reads into pv_phase / pv tail).
    auto prefetch_qk = [&](uint32_t k_byte_base, u32x4_t& s0, u32x4_t& s1, u32x4_t& s2) {
        __builtin_amdgcn_sched_barrier(0);
        s0 = ds_read2_b64_off_0_16(qk_addr_at(k_byte_base, 0));
        s1 = ds_read2_b64_off_0_16(qk_addr_at(k_byte_base, 1));
        s2 = ds_read2_b64_off_0_16(qk_addr_at(k_byte_base, 2));
        __builtin_amdgcn_sched_barrier(0);
    };

    // Pass s0/s1/s2 by value so qk_phase sees them as locals (compiler may
    // freely reorder the lgkm waits relative to mfma issue).
    auto qk_phase = [&](uint32_t k_byte_base, u32x4_t s0, u32x4_t s1, u32x4_t s2,
                        fp32x16_t sacc[kNKvTiles]) {
        constexpr int kTotalReads = kNKvTiles * (kDChunks / 2);  // 16
        constexpr int kDepth      = 3;
        asm volatile("s_setprio 3" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int i = 0; i < kTotalReads; ++i) {
            const int  t    = i / (kDChunks / 2);
            const int  dc   = i % (kDChunks / 2);
            const int  slot = i % kDepth;
            u32x4_t& cur = (slot == 0) ? s0 : (slot == 1) ? s1 : s2;
            if (i == kTotalReads - 1) wait_lgkm<0>(cur);
            else                      wait_lgkm<kDepth - 1>(cur);
            bf16x4_t kf_lo, kf_hi;
            u32x2_t lo = {cur[0], cur[1]}, hi = {cur[2], cur[3]};
            __builtin_memcpy(&kf_lo, &lo, 8);
            __builtin_memcpy(&kf_hi, &hi, 8);
            if (dc == 0) sacc[t] = asm_mfma_zero(kf_lo, qreg[dc]);
            else         asm_mfma(kf_lo, qreg[dc], sacc[t]);
            // Last mfma of the chain uses BUILTIN form so LLVM tracks the
            // sacc[t] write — softmax's bmax fmaxf reads after writeback
            // retires. inline-asm form would hide the dep → stale read.
            sacc[t] = mfma(kf_hi, qreg[dc + 8], sacc[t]);
            if (i + kDepth < kTotalReads)
                cur = ds_read2_b64_off_0_16(qk_addr_at(k_byte_base, i + kDepth));
        }
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 0" ::: "memory");
    };

    // ─── PV phase (L1V bld16, depth-5 prefetch) ────────────────────────────
    // O += V × Pᵀ. V_t bytes at (it*16384 + pair*1024 + lane*16) = bf16x8 =
    // [bf16x4 for mfma_a=2*pair | bf16x4 for mfma_b=2*pair+1]. 16 pairs per
    // iter; each pair runs 2 mfmas sharing preg[t][g]. Caller pre-issues the
    // first 2 bld16 (cross-phase) so their VMCNT-tracked latency overlaps
    // with the prior phase's mfma drain — pv_phase issues bld16 #3..4 itself
    // to reach depth=5 before the first consumer.
    auto pair_off_of = [&](int it, int p) -> uint32_t {
        return (uint32_t)((uint64_t)it * (kNBlock * kHeadDim * 2))
             + (uint32_t)(p * 1024) + lane_byte_off;
    };

    constexpr int kPvPrePrefetch = 2;
    auto prefetch_pv = [&](int it, u32x4_t r_pre[/*kPvPrePrefetch*/]) {
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int p = 0; p < kPvPrePrefetch; ++p) {
            r_pre[p] = bld16(vr, (int)pair_off_of(it, p));
        }
        __builtin_amdgcn_sched_barrier(0);
    };

    auto pv_phase = [&](int it, u32x4_t r_pre[/*kPvPrePrefetch*/],
                        const bf16x4_t preg[kNKvTiles][4]) {
        constexpr int kPvPairs = 16;
        constexpr int kPvDepth = 5;
        asm volatile("s_setprio 3" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        u32x4_t r[kPvDepth];
        #pragma unroll
        for (int p = 0; p < kPvPrePrefetch; ++p) r[p] = r_pre[p];
        #pragma unroll
        for (int p_init = kPvPrePrefetch; p_init < kPvDepth; ++p_init) {
            r[p_init] = bld16(vr, (int)pair_off_of(it, p_init));
        }
        #pragma unroll
        for (int p = 0; p < kPvPairs; ++p) {
            const int j  = p >> 1;
            const int pq = p & 1;
            const int h  = pq * 2;
            const int t  = j >> 2;
            const int g  = j & 3;
            const int slot = p % kPvDepth;
            u32x4_t& cur = r[slot];
            bf16x4_t vbf_a, vbf_b;
            u32x2_t lo = {cur[0], cur[1]}, hi = {cur[2], cur[3]};
            __builtin_memcpy(&vbf_a, &lo, 8);
            __builtin_memcpy(&vbf_b, &hi, 8);
            asm_mfma(vbf_a, preg[t][g], oreg[h]);
            asm_mfma(vbf_b, preg[t][g], oreg[h + 1]);
            if (p + kPvDepth < kPvPairs) {
                cur = bld16(vr, (int)pair_off_of(it, p + kPvDepth));
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 0" ::: "memory");
    };

    // L1 prefetcher for V_t[it_next], fired from Group A's softmax-only P2.
    // vmcnt is per-wave so this is independent of any consumer's pending
    // bld16. Wave-split: each of 4 waves takes 4 unique pairs → 128 unique
    // cache line requests hit all L1 sets in parallel. Inline-asm clobber-only
    // (no output operand) → no per-prefetch vmcnt drain emitted.
    auto prefetch_v_t_l1 = [&](int it_next) {
        if (it_next < 0) return;
        const uint32_t pf_base = (uint32_t)((uint64_t)it_next * (kNBlock * kHeadDim * 2));
        const int p_start = wid_in_g * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int p = p_start + i;
            const uint32_t pair_off = pf_base + (uint32_t)(p * 1024) + lane_byte_off;
            asm volatile(
                "buffer_load_dwordx4 v[140:143], %0, %1, 0 offen"
                :
                : "v"(pair_off), "s"(vr)
                : "v140", "v141", "v142", "v143", "memory"
            );
        }
    };

    // ─── Softmax: bmax fold + exp2 + rescale + pack ────────────────────────
    auto softmax = [&](fp32x16_t sacc[kNKvTiles], bf16x4_t preg[kNKvTiles][4]) {
        // bmax fold happens IN softmax (P2), where the other group is in
        // pv+qk (MFMA-heavy) — the v_max3 chain hides under their MFMA.
        float bmax_local = -FLT_MAX;
        #pragma unroll
        for (int t = 0; t < kNKvTiles; ++t)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                bmax_local = fmaxf(bmax_local,
                    fmaxf(fmaxf(sacc[t][q*4    ], sacc[t][q*4 + 1]),
                          fmaxf(sacc[t][q*4 + 2], sacc[t][q*4 + 3])));
        constexpr float kRescaleThr = 8.0f;
        const float new_m_local = bmax_local * scale_log2e;
        const bool  first_block = row_m == -FLT_MAX;
        const bool  all_ok = __all((int)((new_m_local - row_m) < kRescaleThr)) != 0
                            && !first_block;
        float neg_m;
        float rescale = 1.f;
        if (__builtin_expect(all_ok, 1)) {
            neg_m = -row_m;
        } else {
            const float new_m = fmaxf(row_m, xor32_max(bmax_local) * scale_log2e);
            rescale = first_block ? 0.f : exp2c(row_m - new_m);
            if (!first_block) {
                // Force in-place v_mul_f32_e32: per-subscript `oreg[h][i] *=`
                // compiles to v_pk_mul_f32 (partial-subreg write → mfma C-read
                // hazard). Inline `+v(oreg[h][i])` keeps it scalar.
                #pragma unroll
                for (int h = 0; h < kHdTiles; ++h)
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        asm volatile("v_mul_f32_e32 %0, %1, %0"
                                     : "+v"(oreg[h][i]) : "v"(rescale));
            }
            row_m = new_m;
            neg_m = -new_m;
        }
        float bs = 0.f;
        #pragma unroll
        for (int t = 0; t < kNKvTiles; ++t) {
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float p0 = exp2c(fmaf(sacc[t][g*4    ], scale_log2e, neg_m));
                const float p1 = exp2c(fmaf(sacc[t][g*4 + 1], scale_log2e, neg_m));
                const float p2 = exp2c(fmaf(sacc[t][g*4 + 2], scale_log2e, neg_m));
                const float p3 = exp2c(fmaf(sacc[t][g*4 + 3], scale_log2e, neg_m));
                // NaN-safe pack: a masked row gives all-zero p (safe); a zero
                // row_l is handled in the epilogue. MaybeNaN scrubs any drift.
                preg[t][g] = pack4_bf16</*MaybeNaN=*/true>(p0, p1, p2, p3);
                bs = vadd_f32(bs, (p0 + p1) + (p2 + p3));
            }
        }
        row_l = vadd_f32(vmul_f32(row_l, rescale), bs);
    };

    // ─── Main loop ─────────────────────────────────────────────────────────
    //   Group A  P1: pv(V_t[N-1]) + qk(K[N])      P2: softmax + V_t prefetch
    //   Group B  P1: K[N+1] HBM→LDS + softmax     P2: pv(V_t[N-1]) + qk(K[N])
    // V_t prefetch lives in Group A's softmax-only P2 — bursts in pv loop
    // regress perf by stalling consumer issue on vmcnt pressure.
    wait_vmcnt0();
    lds_barrier();
    fp32x16_t sacc[kNKvTiles];
    bf16x4_t  preg[kNKvTiles][4];
    const int nblocks = seq_len / kNBlock;
    bool first = true;
    for (int nb = 0; nb < nblocks; ++nb) {
        const int k_cur = nb & 1;
        const int k_nxt = (nb + 1) & 1;
        const bool has_next = (nb + 1 < nblocks);
        const uint32_t k_byte_base = (uint32_t)(uintptr_t)&lds_k[k_cur][0];

        // Slots declared inside the loop so the compiler treats them as
        // single-iter lifetime (no cross-iter register pressure).
        u32x4_t s0, s1, s2;
        u32x4_t r_pre[kPvPrePrefetch];

        // Phase 1
        if (group_a) {
            // Cross-phase: fire qk LDS reads (lgkmcnt) AND pv's first 2 bld16
            // (vmcnt) BEFORE pv_phase. Independent counters → no contention;
            // qk reads cover pv's mfma duration, pv pre-issues fill the depth-5
            // ring without an initial cold-issue burst.
            prefetch_qk(k_byte_base, s0, s1, s2);
            if (!first) {
                prefetch_pv(nb - 1, r_pre);
                pv_phase(nb - 1, r_pre, preg);
            }
            qk_phase(k_byte_base, s0, s1, s2, sacc);
        } else {
            if (!first) softmax(sacc, preg);
            if (has_next) direct_lds_k(nb + 1, k_nxt);
        }

        // Phase 2
        if (group_a) {
            softmax(sacc, preg);
            if (has_next) prefetch_v_t_l1(nb);
        } else {
            prefetch_qk(k_byte_base, s0, s1, s2);
            if (!first) {
                prefetch_pv(nb - 1, r_pre);
                pv_phase(nb - 1, r_pre, preg);
            }
            qk_phase(k_byte_base, s0, s1, s2, sacc);
        }

        // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes
        // (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so the next iter's
        // qk_phase doesn't read stale K. lds_barrier covers the LGKMCNT side.
        wait_vmcnt0();
        lds_barrier();
        first = false;
        (void)k_nxt;
    }

    // Epilogue: drain. Group B softmaxes its final qk; both groups pv on the
    // last V tile.
    if (!group_a) softmax(sacc, preg);
    {
        const int it = nblocks - 1;
        u32x4_t r_pre[kPvPrePrefetch];
        prefetch_pv(it, r_pre);
        pv_phase(it, r_pre, preg);
    }

    // ─── Output ────────────────────────────────────────────────────────────
    // row_l is summed across lane^32 once here (math-equiv since rescale
    // is global to the row). invl=0 when row_l==0 prevents 0×inf=NaN.
    row_l = xor32_sum(row_l);
    if (valid) {
        const float invl  = (row_l == 0.0f) ? 0.0f : rcpf(row_l);
        const int   q_row = q_my + col;
        // Hoist invl out of pack args so v_pk_mul_f32 is safe here (no PV
        // mfmas follow — no C-read hazard). MaybeNaN pack4 scrubs any
        // residual NaN/Inf in oreg from pathological inputs.
        #pragma unroll
        for (int h = 0; h < kHdTiles; ++h) {
            oreg[h] = oreg[h] * invl;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int hd_base = h * kKvTile + k_outer * 4 + g * 8;
                bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(
                    oreg[h][g*4    ], oreg[h][g*4 + 1],
                    oreg[h][g*4 + 2], oreg[h][g*4 + 3]);
                bst8(or_, (q_row * kHeadDim + hd_base) * 2, out);
            }
        }
    }
}

// ─── V transpose ───────────────────────────────────────────────────────────
// Rearranges V[seq_len, 128] (bf16) into V_t for pv_phase. 1 WG per 64-kv
// iter, 256 threads, 16 KB LDS stage for coalesced HBM reads.
//
// V_t bytes at (iter*16384 + pair*1024 + lane*16) = bf16x8 =
//   [V[iter*64 + (2p>>2)*8 + (l>>5)*4 + i, (2p&3)*32 + (l&31)] for i=0..3,
//    V[iter*64 + ((2p+1)>>2)*8 + (l>>5)*4 + i, ((2p+1)&3)*32 + (l&31)] for i=0..3]
constexpr int kXposeThreadsPerWG = 256;
__global__ __launch_bounds__(kXposeThreadsPerWG, 4)
void v_transpose_kernel(
    const bf16_t* __restrict__ V,
    bf16_t*       __restrict__ V_t,
    int           seq_len_per_head
) {
    const int it  = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ uint32_t sV_dw[kNBlock * (kHeadDim / 2)];  // 16 KB
    const auto vr_x = mkr(V);

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int op = k * kXposeThreadsPerWG + tid;
        const int row    = op >> 4;          // 0..63
        const int col_8  = op & 15;          // 0..15
        const long long gmem_off_bf16 = (long long)(it * kNBlock + row) * kHeadDim + col_8 * 8;
        u32x4_t v4 = __builtin_amdgcn_raw_buffer_load_b128(
            vr_x, (uint32_t)(gmem_off_bf16 * 2), 0, 0);
        const int sV_dw_idx = row * (kHeadDim / 2) + col_8 * 4;
        sV_dw[sV_dw_idx + 0] = v4[0];
        sV_dw[sV_dw_idx + 1] = v4[1];
        sV_dw[sV_dw_idx + 2] = v4[2];
        sV_dw[sV_dw_idx + 3] = v4[3];
    }

    __syncthreads();

    bf16_t* sV = reinterpret_cast<bf16_t*>(sV_dw);
    const auto otr_x = mkr(V_t);

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int op = k * kXposeThreadsPerWG + tid;
        const int p  = op >> 6;
        const int l  = op & 63;

        const int mfma_a = 2 * p;
        const int mfma_b = 2 * p + 1;
        const int hd_tile_a  = mfma_a & 3;
        const int hd_tile_b  = mfma_b & 3;
        const int kv_chunk_a = mfma_a >> 2;
        const int kv_chunk_b = mfma_b >> 2;

        const int k_outer = l >> 5;
        const int col     = l & 31;
        const int hd_a    = hd_tile_a * 32 + col;
        const int hd_b    = hd_tile_b * 32 + col;
        const int kv_a0   = kv_chunk_a * 8 + k_outer * 4;
        const int kv_b0   = kv_chunk_b * 8 + k_outer * 4;

        bf16_t packed[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed[i]     = sV[(kv_a0 + i) * kHeadDim + hd_a];
            packed[i + 4] = sV[(kv_b0 + i) * kHeadDim + hd_b];
        }
        u32x4_t packed_v;
        __builtin_memcpy(&packed_v, packed, sizeof(packed));

        const long long out_off_bf16 =
            (long long)it * 16 * 64 * 8 + (long long)p * 64 * 8 + (long long)l * 8;
        __builtin_amdgcn_raw_buffer_store_b128(
            packed_v, otr_x, (uint32_t)(out_off_bf16 * 2), 0, 0);
    }
    (void)seq_len_per_head;
}

}  // namespace

extern "C" {
// IMPORTANT: V here must point to pre-transposed V_t. Call launch_v_transpose
// first to produce V_t from raw V. V_t and V have identical total byte size.
hipError_t launch_attention_forward(
    const void* Q, const void* K, const void* V, void* Out,
    int batch, int heads, int seq_len, int head_dim, hipStream_t stream
) {
    if (batch <= 0 || heads <= 0)               return hipErrorInvalidValue;
    if (seq_len <= 0 || (seq_len % kNBlock))    return hipErrorInvalidValue;
    if (head_dim != kHeadDim)                   return hipErrorInvalidValue;

    const int   qt   = seq_len / kQRows;
    const dim3  grid((qt + kNumWaves - 1) / kNumWaves, batch * heads);
    const float sl2e = kLog2e / sqrtf((float)kHeadDim);

    hipLaunchKernelGGL(
        attention_forward, grid, dim3(kBlockSize), 0, stream,
        (const bf16_t*)Q, (const bf16_t*)K, (const bf16_t*)V,
        (bf16_t*)Out, seq_len, sl2e);
    return hipGetLastError();
}

// V transpose: V[seq_len, 128] (bf16) → V_t in pv_phase's consumed layout.
// Caller allocates V_t with same byte size as V.
//   seq_len_total   = batch * heads * seq_len
//   seq_len_per_head must be a multiple of 64
hipError_t launch_v_transpose(
    const void* V, void* V_t,
    int seq_len_total, int seq_len_per_head, hipStream_t stream
) {
    if (seq_len_per_head <= 0 || (seq_len_per_head % kNBlock)) return hipErrorInvalidValue;
    if (seq_len_total    <= 0 || (seq_len_total    % kNBlock)) return hipErrorInvalidValue;
    const int iters_total = seq_len_total / kNBlock;
    hipLaunchKernelGGL(
        v_transpose_kernel,
        dim3(iters_total, 1, 1), dim3(kXposeThreadsPerWG, 1, 1), 0, stream,
        (const bf16_t*)V, (bf16_t*)V_t, seq_len_per_head);
    return hipGetLastError();
}
}
