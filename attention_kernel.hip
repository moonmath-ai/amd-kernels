// Forward attention v18 (bf16, gfx942 / MI300X), FA2-style online softmax.
//
//   • Symmetric pipeline: both groups run sm[n] → pv[n] → qk[n+1]. Group B
//     is offset by one phase from group A so the SIMD's mfma engine always
//     has exactly one group active. Single sacc/preg storage shared by both.
//
//   • Bootstrap: A's first iter skips pv (no preg yet); B trails by one
//     phase. Epilogue finishes pv[N-1] for both groups.
//
//   • qk_phase uses inline-asm batched-2 ds_read2_b64 with "+v"-pinned slots
//     (depth-2 ping-pong, 4 K reads in flight). Pin is load-bearing — without
//     it the compiler reorders mfma above the wait → stale LDS data → NaN.
//   • pv_phase uses compiler-natural reads (lets the scheduler pick its own
//     prefetch depth and reuse VGPRs that the manual-pin version would lock).
//
//   • 32 KB K LDS + 32 KB V LDS, both double-buffered.
//   • Head-first chiplet swizzle co-locates one (B,H)'s q-blocks on a single
//     XCD so its 4 MB K+V stays resident in that XCD's L2.

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

constexpr int   kWaveSize  = 64;
constexpr int   kNumWaves  = 8;
constexpr int   kBlockSize = kWaveSize * kNumWaves;
constexpr int   kLoadWaves = 4;
constexpr int   kNBlock    = 64;
constexpr int   kNSub      = 4;
constexpr int   kKTile     = 16;
constexpr int   kHeadDim   = 128;
constexpr int   kDTiles    = kHeadDim / kKTile;   // 8
constexpr int   kKDwPerRow = kHeadDim / 2;        // 64
constexpr int   kVDwPerRow = kNBlock  / 2;        // 32
constexpr float kLog2e     = 1.4426950408889634f;
constexpr float kRescaleThreshold = 8.0f;

enum class BF16Round { RTNE, RTZ };
#ifndef BF16_ROUND
#define BF16_ROUND RTNE
#endif
inline constexpr BF16Round kBf16Round = BF16Round::BF16_ROUND;


// ─── primitives ────────────────────────────────────────────────────────────

__device__ __forceinline__ fp32x4_t zero4() { return fp32x4_t{0,0,0,0}; }
__device__ __forceinline__ float    exp2c(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float    rcpf (float x) { return __builtin_amdgcn_rcpf (x); }

__device__ __forceinline__ fp32x4_t mfma(bf16x4_t a, bf16x4_t b, fp32x4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ fp32x4_t scale4(fp32x4_t v, float s) {
    return fp32x4_t{v[0]*s, v[1]*s, v[2]*s, v[3]*s};
}

// Compiler-natural emits v_pk_mul_f32, this helper forces 4 unpacked v_mul_f32_e32 instructions
__device__ __forceinline__ fp32x4_t scale4_unpacked(fp32x4_t v, float s) {
    float v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
    asm volatile("s_nop 7" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3));
    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(v0) : "v"(s));
    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(v1) : "v"(s));
    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(v2) : "v"(s));
    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(v3) : "v"(s));
    return fp32x4_t{v0, v1, v2, v3};
}

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

__device__ __forceinline__ void wait_vmcnt0()   { asm volatile("s_waitcnt vmcnt(0)"  ::: "memory"); }
__device__ __forceinline__ void wait_lgkmcnt0() { __builtin_amdgcn_s_waitcnt(0x007F); }
__device__ __forceinline__ void lds_barrier()   { wait_lgkmcnt0(); __builtin_amdgcn_s_barrier(); }
__device__ __forceinline__ void setprio_high()   { asm volatile("s_setprio 1" ::: "memory"); }
__device__ __forceinline__ void setprio_normal() { asm volatile("s_setprio 0" ::: "memory"); }


// HBM→LDS direct dword load. Bypasses clang's `s_cselect_b32 m0, s, -1`
// guard that substitutes m0=0xFFFFFFFF when the LDS destination is 0,
// silently dropping the writes off-end. The guard is emitted around every
// `__builtin_amdgcn_*_buffer_load_lds` variant; raw asm avoids it.
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


// Head-first chiplet swizzle: chunk of `chunk_size` consecutive workgroups
// land on the same XCD. With chunk_size = blocks_per_head, all q-blocks of
// one (B,H) co-locate on one XCD's L2.
constexpr int kNumXcds = 8;
__device__ __forceinline__ int chiplet_swizzle(int wgid, int num_wgs, int chunk_size) {
    const int block = kNumXcds * chunk_size;
    const int limit = (num_wgs / block) * block;
    if (wgid >= limit) return wgid;                       // partial trailing block
    const int xcd          = wgid % kNumXcds;
    const int local_pid    = wgid / kNumXcds;
    const int chunk_idx    = local_pid / chunk_size;
    const int pos_in_chunk = local_pid % chunk_size;
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}


// ─── bf16 packing with selectable rounding ─────────────────────────────────

// MaybeNaN=true → NaN preservation: NaN propagates as
// 0x7FFF0000, finite values get tie-to-even bias.
template <bool MaybeNaN>
__device__ __forceinline__ unsigned bf16_round(float f) {
    if constexpr (kBf16Round == BF16Round::RTNE) {
        if constexpr (MaybeNaN) {
            uint32_t u_bits = __builtin_bit_cast(uint32_t, f);
            uint32_t tmp;
            u32x2_t  nan_mask;   // SGPR pair on gfx9 wave64
            asm volatile(
                "v_cmp_u_f32 %0, %2, %2\n"
                "v_bfe_u32 %1, %2, 16, 1\n"
                "v_add3_u32 %1, %2, %1, %3\n"
                "v_cndmask_b32 %2, %1, %4, %0"
                : "=s"(nan_mask), "=v"(tmp), "+v"(u_bits)
                : "v"(0x7FFFu), "v"(0x7FFF0000u)
            );
            return u_bits;
        } else {
            const unsigned u = __builtin_bit_cast(unsigned, f);
            return u + 0x7FFFu + ((u >> 16) & 1u);
        }
    } else {
        return __builtin_bit_cast(unsigned, f);
    }
}
template <bool MaybeNaN = false>
__device__ __forceinline__ bf16x4_t pack4_bf16(float a, float b, float c, float d) {
    const unsigned ua = bf16_round<MaybeNaN>(a), ub = bf16_round<MaybeNaN>(b);
    const unsigned uc = bf16_round<MaybeNaN>(c), ud = bf16_round<MaybeNaN>(d);
    const unsigned w0 = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
    const unsigned w1 = __builtin_amdgcn_perm(ud, uc, 0x07060302u);
    bf16x4_t o;
    __builtin_memcpy(&o,             &w0, 4);
    __builtin_memcpy((char*)&o + 4,  &w1, 4);
    return o;
}


// ─── cross-lane reductions across the wave ─────────────────────────────────

__device__ __forceinline__ int wave_lane_id() {
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

// Cross-lane reductions over the 4-lane subset {i, i^16, i^32, i^48}.
// In our mfma 16x16x16 output layout each lane outputs at q_row = col, so
// lanes with the same col but varying quad process the same q-row across
// all 64 kv positions (4 quads × 16 kv per quad). Reducing over those 4
// lanes is exactly the per-q-row softmax reduction.
__device__ __forceinline__ float xor_max(float v) {
    const int lane = wave_lane_id();
    const int p16  = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
    const float v1 = fmaxf(v, __builtin_bit_cast(float, p16));
    const int p32  = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, v1));
    wait_lgkmcnt0();
    return fmaxf(v1, __builtin_bit_cast(float, p32));
}
__device__ __forceinline__ float xor_sum(float v) {
    const int lane = wave_lane_id();
    const int p16  = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
    float r1; asm volatile("v_add_f32_e32 %0, %1, %2"
                           : "=v"(r1) : "v"(v), "v"(__builtin_bit_cast(float, p16)));
    const int p32  = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, r1));
    wait_lgkmcnt0();
    float r2; asm volatile("v_add_f32_e32 %0, %1, %2"
                           : "=v"(r2) : "v"(r1), "v"(__builtin_bit_cast(float, p32)));
    return r2;
}

__device__ __forceinline__ bool wave_all(bool p) { return __all((int)p) != 0; }


// ─── inline-asm LDS reads (offsets must be compile-time literals) ──────────
//
// qk_phase pairs (d, d+4) bf16x4 K rows (128-byte stride). ds_read2_b64
// has 8-byte offset units, so offset1:16 = 128 bytes.
__device__ __forceinline__ u32x4_t ds_read2_b64_d_pair(uint32_t base) {
    u32x4_t r;
    asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:16" : "=v"(r) : "v"(base));
    return r;
}

// Two-slot lgkmcnt wait. "+v" pin on both slots forces the wait to happen
// before any consumer (mfma) reads from a/b. Without the pin the compiler
// reorders the mfma chain above the wait → mfma reads stale LDS data.
// Pin lifetime is local to the inlined lambda; softmax sees these slots dead.
template <int N>
__device__ __forceinline__ void wait_lgkm_pin2(u32x4_t& a, u32x4_t& b) {
    static_assert(N >= 0 && N <= 4);
    if      constexpr (N == 0) asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a), "+v"(b) :: "memory");
    else if constexpr (N == 1) asm volatile("s_waitcnt lgkmcnt(1)" : "+v"(a), "+v"(b) :: "memory");
    else if constexpr (N == 2) asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(a), "+v"(b) :: "memory");
    else if constexpr (N == 3) asm volatile("s_waitcnt lgkmcnt(3)" : "+v"(a), "+v"(b) :: "memory");
    else if constexpr (N == 4) asm volatile("s_waitcnt lgkmcnt(4)" : "+v"(a), "+v"(b) :: "memory");
}


// ─── online softmax with skip-rescale gate ─────────────────────────────────

// Force scalar v_add_f32 / v_mul_f32 so the compiler can't pack pairs of
// qg's softmax math into v_pk_*_f32. Packed ops produce a long linear
// chain on a single VGPR pair whose wide writebacks contend with the
// concurrent pv_phase mfma writebacks on the same SIMD's wave-pair.
__device__ __forceinline__ float vadd_f32(float a, float b) {
    float r;
    asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ float vmul_f32(float a, float b) {
    float r;
    asm volatile("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ float vfma_f32(float a, float b, float c) {
    float r;
    asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}
__device__ __forceinline__ fp32x4_t scale4_scalar(fp32x4_t v, float s) {
    return fp32x4_t{vmul_f32(v[0], s), vmul_f32(v[1], s),
                    vmul_f32(v[2], s), vmul_f32(v[3], s)};
}

__device__ __forceinline__ void softmax_update(
    fp32x4_t sacc[], fp32x4_t oreg[], bf16x4_t preg[],
    float& row_m, float& row_l, float scale_log2e, float bmax_local
) {
    const float new_m_local = bmax_local * scale_log2e;
    const bool  first_block = row_m == -FLT_MAX;
    const bool  lane_ok = (new_m_local - row_m) < kRescaleThreshold;
    const bool  all_ok  = wave_all(lane_ok) && !first_block;

    float neg_m;
    float rescale = 1.f;
    if (__builtin_expect(all_ok, 1)) {
        neg_m = -row_m;
    } else {
        const float bmax  = xor_max(bmax_local);
        const float old_m = row_m;
        const float new_m = fmaxf(old_m, bmax * scale_log2e);
        if (new_m != old_m) {
            rescale = first_block ? 0.f : exp2c(old_m - new_m);
            if (!first_block) {
                #pragma unroll
                for (int t = 0; t < kDTiles; ++t)
                    oreg[t] = scale4_unpacked(oreg[t], rescale);
            }
        }
        row_m = new_m;
        neg_m = -new_m;
    }

    float bs[kNSub];
    #pragma unroll
    for (int sn = 0; sn < kNSub; ++sn) {
        const float p0 = exp2c(fmaf(sacc[sn][0], scale_log2e, neg_m));
        const float p1 = exp2c(fmaf(sacc[sn][1], scale_log2e, neg_m));
        const float p2 = exp2c(fmaf(sacc[sn][2], scale_log2e, neg_m));
        const float p3 = exp2c(fmaf(sacc[sn][3], scale_log2e, neg_m));
        bs[sn]   = (p0 + p1) + (p2 + p3);
        // NaN-preserving RTNE on softmax pack.
        preg[sn] = pack4_bf16</*MaybeNaN=*/true>(p0, p1, p2, p3);
    }
    float bsum = vadd_f32(vadd_f32(bs[0], bs[1]), vadd_f32(bs[2], bs[3]));
    bsum = xor_sum(bsum);
    row_l = vfma_f32(row_l, rescale, bsum);
}


// ─── kernel ────────────────────────────────────────────────────────────────

__global__ __launch_bounds__(kBlockSize, 1)
void attention_forward_v18(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16*       __restrict__ Out,
    int seq_len, int q_tiles, float scale_log2e
) {
    const int  tid     = threadIdx.x;
    const int  wid     = tid >> 6;
    const int  lane    = tid & 63;
    const int  col     = lane & 15;
    const int  quad    = lane >> 4;
    const bool group_a = wid < kLoadWaves;

    // Precomputed lane swizzle bases.
    // qk and pv swizzles factor as
    //   dw = row_stride + col*ROW_DW + (lane_perm ^ (k*8))
    // with lane_perm = (2*quad) ^ (col<<1) per-lane-constant and
    // k ∈ {0,1,2,3} covering all distinct values of the iter-varying XOR
    // (k=dp for qk; k = sn ^ (dt&1) for pv). Precomputing the 4 lane
    // bases as VGPR constants lets the compiler collapse the per-call
    // 3-4-op XOR chain to a single v_lshl_or_b32
    const int lane_perm = (2 * quad) ^ (col << 1);
    const int dw_base_pv[4] = {
        col * 32 + (lane_perm ^ 0),
        col * 32 + (lane_perm ^ 8),
        col * 32 + (lane_perm ^ 16),
        col * 32 + (lane_perm ^ 24),
    };
    const int dw_base_qk[4] = {
        col * 64 + (lane_perm ^ 0),
        col * 64 + (lane_perm ^ 8),
        col * 64 + (lane_perm ^ 16),
        col * 64 + (lane_perm ^ 24),
    };

    // Head-first XCD swizzle: chunk_size = gridDim.x = blocks_per_head puts
    // every q-block of one (B,H) on the same XCD for L2 reuse of K and V.
    const int wgid_in   = blockIdx.y * gridDim.x + blockIdx.x;
    const int total_wgs = gridDim.x * gridDim.y;
    const int wgid      = chiplet_swizzle(wgid_in, total_wgs, gridDim.x);
    const int sw_bx     = wgid % gridDim.x;
    const int sw_by     = wgid / gridDim.x;

    // Each wave handles 2 q-tiles (16 Q rows each).
    const int qtile_base = sw_bx * (kNumWaves * 2) + wid * 2;
    const bool valid[2]  = {qtile_base     < q_tiles,
                            qtile_base + 1 < q_tiles};
    const int qbase[2]   = {valid[0] ?  qtile_base      * 16 : 0,
                            valid[1] ? (qtile_base + 1) * 16 : 0};

    const long long bhoff = (long long)sw_by * seq_len * kHeadDim;
    const auto qr  = mkr(Q   + bhoff);
    const auto kr  = mkr(K   + bhoff);
    const auto vr  = mkr(V   + bhoff);
    const auto or_ = mkr(Out + bhoff);

    __shared__ uint32_t lds_k[2][kNBlock  * kKDwPerRow];   // 32 KB (2-buf)
    __shared__ uint32_t lds_v[2][kHeadDim * kVDwPerRow];   // 32 KB (2-buf)

    // ─── Cached lane-swizzle byte offsets for QK reads ─────────────────────
    //
    //   qk byte addr = qk_b_dw[dp] | (sn << 12) | (kbuf << 14)
    //
    // bit layout: [kbuf:14][sn:12-13][col:8-11][lane_perm*4:2-6] — disjoint.
    const uint32_t qk_b_dw[4] = {
        (uint32_t)((col << 8) | (((unsigned)lane_perm << 2) ^   0u)),
        (uint32_t)((col << 8) | (((unsigned)lane_perm << 2) ^  32u)),
        (uint32_t)((col << 8) | (((unsigned)lane_perm << 2) ^  64u)),
        (uint32_t)((col << 8) | (((unsigned)lane_perm << 2) ^  96u)),
    };

    // ─── Q load (persistent VGPR for whole kernel) ─────────────────────────
    bf16x4_t q[2][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d) {
            q[qg][d] = valid[qg]
                ? bld8(qr, ((qbase[qg] + col) * kHeadDim + d * kKTile + 4 * quad) * 2)
                : bf16x4_t{0,0,0,0};
        }

    // ─── output accumulator + running m, l ─────────────────────────────────
    fp32x4_t o[2][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d) o[qg][d] = zero4();
    float m_max[2] = {-FLT_MAX, -FLT_MAX};
    float l_sum[2] = { 0.f,      0.f    };

    // Persistent state shared by both group branches. Lifetimes span iters.
    fp32x4_t sacc[2][kNSub];
    bf16x4_t preg[2][kNSub];
    float    bmax[2] = {-FLT_MAX, -FLT_MAX};

    #pragma unroll
    for (int qg = 0; qg < 2; ++qg) {
        #pragma unroll
        for (int sn = 0; sn < kNSub; ++sn) {
            sacc[qg][sn] = zero4();
            preg[qg][sn] = bf16x4_t{0,0,0,0};
        }
    }

    // ─── HBM K direct→LDS loader (group B handles K loads) ─────────────────
    constexpr int kKRowsPerWaveGroup = kNBlock / kLoadWaves;
    const int wid_in_group = wid - (group_a ? 0 : kLoadWaves);
    auto direct_lds_k = [&](int nb, int buf) {
        const uint32_t lds_base = __builtin_amdgcn_readfirstlane(
            (uint32_t)(uintptr_t)&lds_k[buf][0]
            + (uint32_t)(wid_in_group * kKRowsPerWaveGroup * kKDwPerRow * 4));
        const uint32_t soff = __builtin_amdgcn_readfirstlane(
            (uint32_t)((uint64_t)nb * kNBlock * kHeadDim * 2));
        #pragma unroll
        for (int call = 0; call < kKRowsPerWaveGroup; ++call) {
            const int row     = wid_in_group * kKRowsPerWaveGroup + call;
            const int swizzle = (row & 0xF) << 1;
            const int voff    = row * kHeadDim * 2 + (lane ^ swizzle) * 4;
            buffer_load_b32_to_lds(kr, lds_base + call * 256, voff, soff);
        }
    };


    // ─── HBM V loader (group A) + DPP transpose + LDS store ────────────────
    const int ltid      = tid & ((kLoadWaves * kWaveSize) - 1);
    const int kvrow     = ltid >> 2;
    const int vdim      = (ltid & 3) * 8;
    const int npair_idx = kvrow >> 1;
    const int is_odd_k  = kvrow  & 1;

    auto issue_v = [&](int nb, bf16x8_t& c0, bf16x8_t& c1, bf16x8_t& c2, bf16x8_t& c3) {
        const int vbase = (nb * kNBlock + kvrow) * kHeadDim * 2;
        c0 = bld16(vr, vbase + ( 0 + vdim) * 2);
        c1 = bld16(vr, vbase + (32 + vdim) * 2);
        c2 = bld16(vr, vbase + (64 + vdim) * 2);
        c3 = bld16(vr, vbase + (96 + vdim) * 2);
    };

    auto transpose_chunk = [&](bf16x8_t own) {
        constexpr uint32_t kPermLoLo = 0x05040100u;
        constexpr uint32_t kPermHiHi = 0x07060302u;
        u32x4_t own_dw;  __builtin_memcpy(&own_dw, &own, 16);
        u32x4_t out;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t shl4, shr4;
            asm volatile("v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1"
                         : "=v"(shl4) : "v"(own_dw[i]));
            asm volatile("v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1"
                         : "=v"(shr4) : "v"(own_dw[i]));
            const uint32_t pe = __builtin_amdgcn_perm(shl4, own_dw[i], kPermLoLo);
            const uint32_t po = __builtin_amdgcn_perm(own_dw[i], shr4, kPermHiHi);
            out[i] = is_odd_k ? po : pe;
        }
        return out;
    };

    auto store_v_pair = [&](int buf, int chunk_base, u32x4_t v0, u32x4_t v1) {
        const int base = chunk_base * 32 * kVDwPerRow;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d        = vdim + 2 * i + is_odd_k;
            const int swizzled = npair_idx ^ ((d & 0xF) << 1) ^ (((d >> 4) & 1) << 3);
            const int addr     = base + d * kVDwPerRow + swizzled;
            lds_v[buf][addr]                   = v0[i];
            lds_v[buf][addr + 32 * kVDwPerRow] = v1[i];
        }
    };

    auto vstore_v = [&](int buf, bf16x8_t c0, bf16x8_t c1, bf16x8_t c2, bf16x8_t c3) {
        if constexpr (kBf16Round == BF16Round::RTNE) {
            u32x4_t t0 = transpose_chunk(c0);
            u32x4_t t1 = transpose_chunk(c1);
            u32x4_t t2 = transpose_chunk(c2);
            u32x4_t t3 = transpose_chunk(c3);
            store_v_pair(buf, 0, t0, t1);
            store_v_pair(buf, 2, t2, t3);
        } else {
            store_v_pair(buf, 0, transpose_chunk(c0), transpose_chunk(c1));
            store_v_pair(buf, 2, transpose_chunk(c2), transpose_chunk(c3));
        }
    };

    // ─── qk_phase: outer-dp / inner-sn, 4 K reads in flight (batched-2) ────
    //
    // outer-dp/inner-sn keeps all 8 accumulators alive at every dp step
    // (8 chains × 4cy issue = 32cy covers the 16cy mfma latency). Each dp
    // step consumes 2 and re-issues 2 under lgkmcnt(2) to keep 4 reads
    // outstanding. bmax_local fold + xor_max swizzle pre-issued at dp=3
    // tail so the cross-lane LDS round-trip hides under the lgkm drain
    // and softmax_update's exp chain doesn't wait for it on the critical
    // path.
    auto qk_phase = [&](int kbuf, fp32x4_t sacc_dst[2][kNSub], float bmax_dst[2]) {
        // byte_addr = qk_b_dw[dp] | (sn << 12) | (kbuf << 14)  — all bit-disjoint.
        auto k_addr = [&](int sn, int dp) -> uint32_t {
            return qk_b_dw[dp] | ((uint32_t)sn << 12) | ((uint32_t)kbuf << 14);
        };

        // Pre-issue dp=0 reads. Caller has placed this call right after the
        // preceding pv_phase, so the LDS round-trip overlaps the trailing
        // pv mfmas and qk_phase's first wait sees zero stall.
        u32x4_t s0 = ds_read2_b64_d_pair(k_addr(0, 0));
        u32x4_t s1 = ds_read2_b64_d_pair(k_addr(1, 0));
        u32x4_t s2 = ds_read2_b64_d_pair(k_addr(2, 0));
        u32x4_t s3 = ds_read2_b64_d_pair(k_addr(3, 0));

        #pragma unroll
        for (int qg = 0; qg < 2; ++qg)
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn) sacc_dst[qg][sn] = zero4();

        auto mfma_consume = [&](u32x4_t& slot, int dp, int sn) {
            bf16x4_t k_lo, k_hi;
            __builtin_memcpy(&k_lo, &slot,            8);
            __builtin_memcpy(&k_hi, (char*)&slot + 8, 8);
            #pragma unroll
            for (int qg = 0; qg < 2; ++qg) {
                sacc_dst[qg][sn] = mfma(k_lo, q[qg][dp    ], sacc_dst[qg][sn]);
                sacc_dst[qg][sn] = mfma(k_hi, q[qg][dp + 4], sacc_dst[qg][sn]);
            }
        };

        // 4 reads in flight; consume 2 + re-issue 2 per half-iter.
        #pragma unroll
        for (int dp = 0; dp < 3; ++dp) {
            wait_lgkm_pin2<2>(s0, s1);
            mfma_consume(s0, dp, 0);
            mfma_consume(s1, dp, 1);
            s0 = ds_read2_b64_d_pair(k_addr(0, dp + 1));
            s1 = ds_read2_b64_d_pair(k_addr(1, dp + 1));

            wait_lgkm_pin2<2>(s2, s3);
            mfma_consume(s2, dp, 2);
            mfma_consume(s3, dp, 3);
            s2 = ds_read2_b64_d_pair(k_addr(2, dp + 1));
            s3 = ds_read2_b64_d_pair(k_addr(3, dp + 1));
        }

        // Per-lane bmax fold, computed during dp=3 tail so the fmaxf chain
        // hides under the trailing mfma latency.
        float bm0 = -FLT_MAX, bm1 = -FLT_MAX;
        auto fold_one_sn = [&](int sn) {
            const float a0 = fmaxf(sacc_dst[0][sn][0], sacc_dst[0][sn][1]);
            const float b0 = fmaxf(sacc_dst[0][sn][2], sacc_dst[0][sn][3]);
            bm0 = fmaxf(bm0, fmaxf(a0, b0));
            const float a1 = fmaxf(sacc_dst[1][sn][0], sacc_dst[1][sn][1]);
            const float b1 = fmaxf(sacc_dst[1][sn][2], sacc_dst[1][sn][3]);
            bm1 = fmaxf(bm1, fmaxf(a1, b1));
        };

        // dp=3: consume the last 4 in-flight reads with no re-issue.
        wait_lgkm_pin2<2>(s0, s1);
        mfma_consume(s0, 3, 0); fold_one_sn(0);
        mfma_consume(s1, 3, 1); fold_one_sn(1);
        wait_lgkm_pin2<0>(s2, s3);
        mfma_consume(s2, 3, 2); fold_one_sn(2);
        mfma_consume(s3, 3, 3); fold_one_sn(3);
        bmax_dst[0] = bm0;
        bmax_dst[1] = bm1;
    };

    // ─── pv_phase: compiler-natural reads ──────────────────────────────────
    auto pv_phase = [&](int vbuf, bf16x4_t prob[2][kNSub]) {
        #pragma unroll
        for (int dt = 0; dt < kDTiles; ++dt)
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn) {
                const int k  = sn ^ (dt & 1);
                const int dw = dt * 16 * kVDwPerRow + dw_base_pv[k];
                bf16x4_t vf = *reinterpret_cast<const bf16x4_t*>(&lds_v[vbuf][dw]);
                #pragma unroll
                for (int qg = 0; qg < 2; ++qg)
                    o[qg][dt] = mfma(vf, prob[qg][sn], o[qg][dt]);
            }
    };

    auto softmax_phase = [&](fp32x4_t sacc_in[2][kNSub], bf16x4_t prob[2][kNSub],
                             float bmax_local[2]) {
        #pragma unroll
        for (int qg = 0; qg < 2; ++qg)
            softmax_update(sacc_in[qg], o[qg], prob[qg],
                           m_max[qg], l_sum[qg], scale_log2e, bmax_local[qg]);
    };

    // V VGPR-stage (group A). Lifecycle: issue_v(N) writes nv0..3 in iter
    // N-1's phase-2 tail; vstore_v consumes them in iter N's phase 2.
    bf16x8_t nv0{}, nv1{}, nv2{}, nv3{};

    const int nblocks = seq_len / kNBlock;

    // ─── Prologue: B loads K[0]→LDS, A pre-issues V[0]→VGPR ────────────────
    // V loads are pipeline-shifted: issue_v(N+1) fires at iter N's phase-2
    // tail (after vstore consumes V[N]), giving the next iter's V load a
    // long enough time-in-flight to hide HBM latency.
    if (!group_a) direct_lds_k(0, 0);
    if ( group_a) issue_v(0, nv0, nv1, nv2, nv3);
    wait_vmcnt0();
    lds_barrier();

    // ─── Main loop ─────────────────────────────────────────────────────────
    //              Group A           Group B
    //   ────────   ────────────      ────────────
    //   Phase 1    pv + qk           sm  + K[N+1] HBM→LDS
    //   Phase 2    sm + V HBM/LDS    pv  + qk
    for (int N = 0; N < nblocks; ++N) {
        const int  p        = N & 1;
        const int  K_cur    = p,     K_nxt = 1 - p;
        const int  V_cur    = 1 - p, V_nxt = p;
        const bool first    = (N == 0);
        const bool has_next = (N + 1) < nblocks;

        if (group_a) {
            if (!first) pv_phase(V_cur, preg);
            qk_phase(K_cur, sacc, bmax);
        } else {
            if (has_next) direct_lds_k(N + 1, K_nxt);
            if (!first)  softmax_phase(sacc, preg, bmax);
        }

        __builtin_amdgcn_s_barrier();

        if (group_a) {
            softmax_phase(sacc, preg, bmax);
            wait_vmcnt0();
            vstore_v(V_nxt, nv0, nv1, nv2, nv3);
            if (has_next) issue_v(N + 1, nv0, nv1, nv2, nv3);
        } else {
            if (!first) pv_phase(V_cur, preg);
            qk_phase(K_cur, sacc, bmax);
        }
        lds_barrier();   // end-of-iter sync (cross-iter buffer-swap visibility)
    }

    // ─── Epilogue: drain last K tile ───────────────────────────────────────
    if (nblocks > 0) {
        const int last_v = (nblocks - 1) & 1;
        if (!group_a) softmax_phase(sacc, preg, bmax);
        pv_phase(last_v, preg);
    }

    // ─── Output: divide by row sum, pack to bf16, store ────────────────────
    #pragma unroll
    for (int qg = 0; qg < 2; ++qg) {
        if (!valid[qg]) continue;
        const float inv = rcpf(l_sum[qg]);
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            const bf16x4_t out = pack4_bf16</*MaybeNaN=*/false>(
                o[qg][t][0] * inv, o[qg][t][1] * inv,
                o[qg][t][2] * inv, o[qg][t][3] * inv);
            bst8(or_, ((qbase[qg] + col) * kHeadDim + t * kKTile + 4 * quad) * 2, out);
        }
    }
}

} // namespace


extern "C" int launch_attention_forward(
    const void* q, const void* k, const void* v, void* out,
    int batch, int heads, int seq_len, int head_dim, void* stream
) {
    if (batch <= 0 || heads <= 0)             return 1;
    if (seq_len <= 0 || (seq_len % kNBlock))  return 2;
    if (head_dim != kHeadDim)                 return 3;

    const int   bh   = batch * heads;
    const int   qt   = seq_len / 16;
    const dim3  grid((qt + kNumWaves * 2 - 1) / (kNumWaves * 2), bh);
    const float sl2e = kLog2e / sqrtf((float)kHeadDim);

    hipLaunchKernelGGL(attention_forward_v18,
        grid, dim3(kBlockSize), 0, (hipStream_t)stream,
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out,
        seq_len, qt, sl2e);
    return (int)hipGetLastError();
}
