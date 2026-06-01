// Forward attention for AMD CDNA3 (MI300X / gfx942), bf16, FA2-style online softmax.
// 16x16x16 bf16 MFMA. 8-wave warp-spec CTA, two groups of 4 waves; per K-tile each
// iter runs as two phases:
//
//                    Group A                      Group B
//   ───────  ────────────────────────  ────────────────────────────
//   Phase 1  pv[N-1] + qk[N]           K[N+1] HBM→LDS + softmax[N-1]
//   Phase 2  softmax[N] + V_t prefetch  pv[N-1] + qk[N]
//
// One group keeps the SIMD's MFMA engine active while the other does VALU+memory.
// L1V variant: V is pre-transposed offline (v_transpose_kernel_16x16) into V_t laid
// out for direct bld16 consumption in pv_phase — no LDS staging for V.
//
// Structure mirrors the 32x32x8 variant (see L1V-32x32x8 branch); only the
// indexing and MFMA-fragment math differ for the 16x16x16 instruction:
//   • 16x16x16 lane split: col = lane&15, quad = lane>>4 (4 quads, fp32x4 C-frag).
//   • Each wave handles kQGroups=2 q-tiles of 16 rows (vs one 32-row tile).
//   • qk reads K in d-tile pairs {dp, dp+4}; pv consumes V_t super-chunks.
// Indexing (layouts, fragments, V_t packing) is explained in docs/indexing.ipynb.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include "opus.hpp"

namespace {

using bf16_t   = __hip_bfloat16;
using bf16x4_t = short   __attribute__((ext_vector_type(4)));  // 4 bf16 = 8 bytes
// RTNE/RTZ/RTNA bf16 rounding selected at compile time via -DBF16_ROUND.
#ifndef BF16_ROUND
#define BF16_ROUND RTZ
#endif
enum class BF16Round { RTNE, RTZ, RTNA };
constexpr BF16Round kBf16Round = BF16Round::BF16_ROUND;
using fp32x4_t = float    __attribute__((ext_vector_type(4)));  // 16x16x16 C-fragment
using u32x2_t  = uint32_t __attribute__((ext_vector_type(2)));
using u32x4_t  = uint32_t __attribute__((ext_vector_type(4)));

constexpr int   kWaveSize       = 64;
constexpr int   kNumWaves       = 8;
constexpr int   kLoadWaves      = 4;
constexpr int   kBlockSize      = kWaveSize * kNumWaves;
constexpr int   kNBlock         = 64;
constexpr int   kKTile          = 16;          // 16x16x16 mfma tile (M=N)
constexpr int   kNSub           = kNBlock / kKTile;     // 4 kv-subtiles / block
constexpr int   kKInner         = 16;          // mfma K (contraction)
constexpr int   kHeadDim        = 128;
constexpr int   kDTiles         = kHeadDim / kKInner;   // 8 head-dim tiles
constexpr int   kQTileRows      = 16;          // rows per q-tile (one mfma)
constexpr int   kQGroups        = 2;           // q-tiles per wave → 32 rows/wave
constexpr int   kKRowDw         = kHeadDim / 2;         // 64 = real K row width (dw)
constexpr int   kKLdsStrideDw   = kKRowDw + 2;         // 66 = padded LDS row stride
constexpr int   kKRowsPerLW     = kNBlock / kLoadWaves; // 16
constexpr float kLog2e          = 1.4426950408889634f;
constexpr uint32_t kBufCfg      = 0x00020000u;

__device__ __forceinline__ fp32x4_t zero4() { return fp32x4_t{0,0,0,0}; }

__device__ __forceinline__ float exp2c(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float rcpf (float x) { return __builtin_amdgcn_rcpf (x); }

__device__ __forceinline__ fp32x4_t mfma(bf16x4_t a, bf16x4_t b, fp32x4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Inline-asm mfma: forces in/out same VGPR for accumulator (no compiler rename
// → no v_mov copies) and keeps issue in source order via asm-volatile.
__device__ __forceinline__ void asm_mfma(bf16x4_t a, bf16x4_t b, fp32x4_t& c) {
    asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
                 : "+v"(c) : "v"(a), "v"(b));
}

// Literal-0 C operand skips the v_mov init the compiler emits to zero a fresh
// accumulator. "=&v" prevents aliasing dst onto the A VGPRs.
__device__ __forceinline__ fp32x4_t asm_mfma_zero(bf16x4_t a, bf16x4_t b) {
    fp32x4_t r;
    asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, 0"
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

// bf16 packing. MaybeNaN=true guards against 0×inf NaN with a v_cmp_u/v_cndmask
// (compiler can interleave the v_cmp_u with mfma, unlike pure inline asm).
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

// Immediate-offset ds_read2: O0/O1 are compile-time (units of 8 B) baked into the
// instruction. Lets one per-kv-subtile base register serve all d-tile pairs — only
// the padded (un-swizzled) K-LDS layout makes the address affine in dp, so dp folds
// into the immediate instead of a per-read VALU add. %c emits the bare constant.
template <int O0, int O1>
__device__ __forceinline__ u32x4_t ds_read2_b64_imm(uint32_t base) {
    u32x4_t r;
    asm volatile("ds_read2_b64 %0, %1 offset0:%c2 offset1:%c3"
                 : "=&v"(r) : "v"(base), "i"(O0), "i"(O1));
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

// QK inner body, compile-time unrolled (recursion → I is constant → dp folds into
// the ds_read2 immediate). Depth-D ring over s0/s1/s2: consume read I, reissue I+D
// into its slot. Each read's two halves are d-tiles {dp, dp+4}; both q-groups
// multiply against the shared K operand. Last read drains lgkm(0), else holds D-1.
// Iteration order: sn = I % kNSub (kv-subtile → base/sacc), dp = I / kNSub (d-pair).
template <int I, int D, int N>
__device__ __forceinline__ void qk_step(
    u32x4_t& s0, u32x4_t& s1, u32x4_t& s2,
    const bf16x4_t q[][kDTiles], fp32x4_t sacc[][kNSub],
    const uint32_t* base_sn)
{
    if constexpr (I < N) {
        constexpr int sn   = I % kNSub;     // kv-subtile → base_sn[sn], sacc[qg][sn]
        constexpr int dp   = I / kNSub;     // d-tile pair → immediate, q[qg][dp]/[dp+4]
        constexpr int slot = I % D;         // depth-D ring slot (s0/s1/s2)
        u32x4_t& cur = (slot == 0) ? s0 : (slot == 1) ? s1 : s2;
        if constexpr (I == N - 1) wait_lgkm<0>(cur);
        else                      wait_lgkm<D - 1>(cur);
        // one ds_read2 = 4 dwords = two bf16x4 K-operands: d-tile dp and dp+4.
        const bf16x4_t k_lo = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[0], cur[1]});
        const bf16x4_t k_hi = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[2], cur[3]});
        #pragma unroll
        for (int qg = 0; qg < kQGroups; ++qg) {
            if constexpr (dp == 0) sacc[qg][sn] = asm_mfma_zero(k_lo, q[qg][0]);
            else                   asm_mfma(k_lo, q[qg][dp], sacc[qg][sn]);
            // Last mfma BUILTIN form so LLVM tracks the sacc write for softmax.
            sacc[qg][sn] = mfma(k_hi, q[qg][dp + 4], sacc[qg][sn]);
        }
        if constexpr (I + D < N) {
            constexpr int J = I + D, jsn = J % kNSub, jdp = J / kNSub;
            cur = ds_read2_b64_imm<jdp * 4, jdp * 4 + 16>(base_sn[jsn]);
        }
        qk_step<I + 1, D, N>(s0, s1, s2, q, sacc, base_sn);
    }
}

__device__ __forceinline__ void wait_vmcnt0() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void wait_lgkmcnt0() { __builtin_amdgcn_s_waitcnt(0x007F); }
__device__ __forceinline__ void lds_barrier() {
    wait_lgkmcnt0();  // lgkmcnt(0)
    __builtin_amdgcn_s_barrier();
}
__device__ __forceinline__ int wave_lane_id() {
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

// Issue-priority hints: in the 8-wave warp-spec, the MFMA-heavy group raises
// priority so it wins issue arbitration over the memory/VALU group, then lowers
// it back. sched_fence = compiler scheduling barrier (pins instruction order so
// hand-placed prefetches don't sink past it). All fold to one inline op each.
__device__ __forceinline__ void mfma_prio_hi() { asm volatile("s_setprio 3" ::: "memory"); }
__device__ __forceinline__ void mfma_prio_lo() { asm volatile("s_setprio 0" ::: "memory"); }
__device__ __forceinline__ void sched_fence()  { __builtin_amdgcn_sched_barrier(0); }

// opus layouts: offset(coords) = Σ coordᵢ·strideᵢ, in bf16 ELEMENTS (×2 → bytes).
// Static strides (opus::number<>) fold to shifts. See indexing notebook + opus README.
template <int... S>
constexpr auto strides() { return opus::make_tuple(opus::number<S>{}...); }
constexpr auto kShape2 = opus::make_tuple(opus::number<1>{}, opus::number<1>{});
constexpr auto kShape3 = opus::make_tuple(opus::number<1>{}, opus::number<1>{}, opus::number<1>{});

// Q / Out: row-major (seq, head_dim).
constexpr auto kQOLayout = opus::make_layout(kShape2, strides<kHeadDim, 1>());

// V_t (pre-transposed): (kv-iter, super-chunk, lane); 16 chunks/iter, bf16x8/lane.
constexpr int  kVtIterElems = kNBlock * kHeadDim;   // 8192 elems / kv-iter
constexpr int  kVtScElems   = kVtIterElems / 16;    // 512  elems / super-chunk
constexpr int  kVtLaneElems = 8;                    // bf16x8 / lane
constexpr auto kVtLayout = opus::make_layout(
    kShape3, strides<kVtIterElems, kVtScElems, kVtLaneElems>());

// K addressed in DWORDS (2 bf16), row-major (kv_row, dword):
//   HBM source: 64 dw/row.  LDS dest: padded to 66 dw/row, which breaks the
//   32-way qk-read bank conflict (stride 64 ≡ 0 mod 32). Linear, no swizzle.
constexpr auto kKHbmDword = opus::make_layout(kShape2, strides<kKRowDw, 1>());
constexpr auto kKLdsDword = opus::make_layout(kShape2, strides<kKLdsStrideDw, 1>());

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

// Cross-lane reduce over {l, l^16, l^32, l^48}: in 16x16x16 C-layout each lane
// outputs at q_row = col, so the 4 quad-lanes sharing col span all 64 kvs of the
// q-row. ds_swizzle 0x401F folds l^16, ds_bpermute folds l^32 (covers l^48 too).
// Explicit lgkm(0) wait avoids a race when the bpermute is the only LGKM in flight.
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
    const float v1 = v + __builtin_bit_cast(float, p16);
    const int p32  = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, v1));
    wait_lgkmcnt0();
    return v1 + __builtin_bit_cast(float, p32);
}

__global__ __launch_bounds__(kBlockSize, 1)
void attention_forward(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ K,
    const bf16_t* __restrict__ V,    // points to V_t (pre-transposed)
    bf16_t* __restrict__ Out,
    int seq_len, float scale_log2e
) {
    // MFMA 16x16x16 lane split: col = q-col / kv-row in tile, quad = which 4 of 64.
    // Tile dims walked by the loops: sn (kv-subtile), dp (d-tile pair), dt (hd-tile),
    // qg (q-group). 8 waves → 2 groups of 4 for warp-spec. See indexing notebook.
    const int tid  = threadIdx.x;
    const int wid  = tid >> 6;
    const int lane = tid & 63;
    const int col      = lane & 15;
    const int quad     = lane >> 4;
    const bool group_a = (wid < kLoadWaves);
    const int wid_in_g = group_a ? wid : (wid - kLoadWaves);

    const int wgid_in   = blockIdx.y * gridDim.x + blockIdx.x;
    const int total_wgs = gridDim.x * gridDim.y;
    const int wgid      = chiplet_swizzle(wgid_in, total_wgs, gridDim.x);
    const int sw_bx     = wgid % gridDim.x;
    const int sw_by     = wgid / gridDim.x;

    // Each wave owns kQGroups q-tiles (16 rows each). WG = 8 waves × 2 × 16 = 256 rows.
    const int q_tiles    = seq_len / kQTileRows;
    const int qtile_base = sw_bx * (kNumWaves * kQGroups) + wid * kQGroups;
    bool valid[kQGroups];
    int  qbase[kQGroups];
    #pragma unroll
    for (int qg = 0; qg < kQGroups; ++qg) {
        valid[qg] = (qtile_base + qg) < q_tiles;
        qbase[qg] = valid[qg] ? (qtile_base + qg) * kQTileRows : 0;
    }

    const long long bhoff = (long long)sw_by * seq_len * kHeadDim;
    const auto qr  = mkr(Q + bhoff);
    const auto kr  = mkr(K + bhoff);
    const auto vr  = mkr(V + bhoff);
    const auto or_ = mkr(Out + bhoff);

    __shared__ uint32_t lds_k[2][kNBlock * kKLdsStrideDw];   // 33 KB (2-buf)
    // V LDS staging removed — V_t is consumed directly from L1 via bld16.

    // K[nb] HBM→LDS (group B). Wave w owns rows [w·16, +16); for each, its 64
    // lanes stream one full K row — lane L copies dword L. Linear padded LDS.
    constexpr uint32_t kKRowBytes   = (uint32_t)(kKLdsStrideDw * 4);
    constexpr uint32_t kKBlockBytes = (uint32_t)(kNBlock * kHeadDim * 2);
    const int wave_row0 = wid_in_g * kKRowsPerLW;   // first K row this wave owns
    auto direct_lds_k = [&](int nb, int kbuf) {
        const uint32_t lds_row0  = __builtin_amdgcn_readfirstlane(
            (uint32_t)(uintptr_t)&lds_k[kbuf][kKLdsDword(wave_row0, 0)]);
        const uint32_t hbm_block = __builtin_amdgcn_readfirstlane(
            (uint32_t)((uint64_t)nb * kKBlockBytes));
        #pragma unroll
        for (int c = 0; c < kKRowsPerLW; ++c) {
            const int kv_row = wave_row0 + c;
            const uint32_t hbm_byte = (uint32_t)(kKHbmDword(kv_row, lane) * 4);
            buffer_load_b32_to_lds(kr, lds_row0 + c * kKRowBytes, hbm_byte, hbm_block);
        }
    };

    // Prologue: K[0] → LDS (group B only). Group A has no work until iter 1.
    if (!group_a) direct_lds_k(0, 0);

    // Persistent VGPR Q (no reload across K-blocks). Lane (qbase+col) owns one
    // q-row; each bld8 grabs 4 head-dims of d-tile d (quad picks which 4 of 16).
    bf16x4_t q[kQGroups][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < kQGroups; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d) {
            const int q_elem = kQOLayout(qbase[qg] + col, d * kKTile + quad * 4);
            q[qg][d] = valid[qg] ? bld8(qr, q_elem * 2) : bf16x4_t{0,0,0,0};
        }

    fp32x4_t o[kQGroups][kDTiles];
    #pragma unroll
    for (int qg = 0; qg < kQGroups; ++qg)
        #pragma unroll
        for (int d = 0; d < kDTiles; ++d) o[qg][d] = zero4();
    float m_max[kQGroups], l_sum[kQGroups];
    #pragma unroll
    for (int qg = 0; qg < kQGroups; ++qg) { m_max[qg] = -FLT_MAX; l_sum[qg] = 0.f; }

    // ─── QK: S = K·Qᵀ via mfma(A=K, B=Q). 4 kv-subtiles × 4 d-tile pairs. ───────
    // Linear K-LDS ⇒ read address is affine in dp, so one base reg per kv-subtile +
    // dp on the ds_read2 immediate (qk_step). base = k_byte_base + this lane's
    // (kv_row=col, dword=2*quad)·4; per-subtile delta steps kv_row by kKTile rows.
    auto qk_base = [&](uint32_t k_byte_base) -> uint32_t {
        return k_byte_base + (uint32_t)(kKLdsDword(col, quad * 2) * 4);  // (kv_row=col, dword)
    };
    constexpr uint32_t kQkBaseSnDelta = (uint32_t)(kKLdsDword(kKTile, 0) * 4);  // +one kv-subtile

    // Prefetch first 3 ds_read2 BEFORE pv_phase: their lgkmcnt advances
    // independently of pv's vmcnt bld16. sched_barrier(0) locks placement.
    // Reads 0,1,2 = (sn=0,1,2; dp=0) → shared immediate, stepped base.
    auto prefetch_qk = [&](uint32_t k_byte_base, u32x4_t& s0, u32x4_t& s1, u32x4_t& s2) {
        sched_fence();
        const uint32_t b0 = qk_base(k_byte_base);
        s0 = ds_read2_b64_imm<0, 16>(b0);
        s1 = ds_read2_b64_imm<0, 16>(b0 + kQkBaseSnDelta);
        s2 = ds_read2_b64_imm<0, 16>(b0 + 2 * kQkBaseSnDelta);
        sched_fence();
    };

    // Pass s0/s1/s2 by value so qk_phase sees them as locals (compiler may
    // freely reorder the lgkm waits relative to mfma issue). The depth-3 ring
    // and immediate-offset reissue live in the recursive qk_step.
    auto qk_phase = [&](uint32_t k_byte_base, u32x4_t s0, u32x4_t s1, u32x4_t s2,
                        fp32x4_t sacc[kQGroups][kNSub]) {
        constexpr int kTotalReads = kNSub * (kDTiles / 2);  // 16
        constexpr int kDepth      = 3;
        const uint32_t b0 = qk_base(k_byte_base);
        uint32_t base_sn[kNSub];
        #pragma unroll
        for (int sn = 0; sn < kNSub; ++sn) base_sn[sn] = b0 + (uint32_t)sn * kQkBaseSnDelta;
        mfma_prio_hi();
        sched_fence();
        qk_step<0, kDepth, kTotalReads>(s0, s1, s2, q, sacc, base_sn);
        sched_fence();
        mfma_prio_lo();
    };

    // ─── PV: O += V·Pᵀ via mfma(A=V_t, B=P). 16 super-chunks/iter, each lane
    // bld16s its pre-packed bf16x8 from V_t (kVtLayout). Depth-4 prefetch ring.
    // Super-chunk sc → hd-tile dt=sc/2, kv-subtile pair {sn0,sn1}; both q-groups
    // share the V operand. ──────────────────────────────────────────────────────
    auto sc_off_of = [&](int it, int sc) -> uint32_t {
        return (uint32_t)(kVtLayout(it, sc, lane) * 2);   // bf16 elems → bytes
    };

    auto pv_phase = [&](int it, const bf16x4_t preg[kQGroups][kNSub]) {
        constexpr int kPvChunks = kDTiles * (kNSub / 2);  // 16
        constexpr int kPvDepth  = 4;
        mfma_prio_hi();
        sched_fence();
        u32x4_t r[kPvDepth];
        #pragma unroll
        for (int p_init = 0; p_init < kPvDepth; ++p_init) {
            r[p_init] = bld16(vr, (int)sc_off_of(it, p_init));
        }
        #pragma unroll
        for (int sc = 0; sc < kPvChunks; ++sc) {
            const int dt      = sc >> 1;        // hd-tile 0..7
            const int sn_pair = sc & 1;         // 0 → {sn0,sn1}={0,1}; 1 → {2,3}
            const int sn0     = sn_pair * 2;
            const int sn1     = sn0 + 1;
            const int slot    = sc % kPvDepth;
            u32x4_t& cur = r[slot];
            // bld16 = 4 dwords = two bf16x4 V-operands (kv-subtiles sn0, sn1).
            const bf16x4_t v_lo = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[0], cur[1]});
            const bf16x4_t v_hi = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[2], cur[3]});
            asm_mfma(v_lo, preg[0][sn0], o[0][dt]);
            asm_mfma(v_lo, preg[1][sn0], o[1][dt]);
            asm_mfma(v_hi, preg[0][sn1], o[0][dt]);
            asm_mfma(v_hi, preg[1][sn1], o[1][dt]);
            if (sc + kPvDepth < kPvChunks) {
                cur = bld16(vr, (int)sc_off_of(it, sc + kPvDepth));
            }
        }
        sched_fence();
        mfma_prio_lo();
    };

    // L1 prefetcher for V_t[it_next], fired from Group A's softmax-only P2.
    // vmcnt is per-wave so this is independent of any consumer's pending bld16.
    // Wave-split: each of 4 waves takes 4 unique super-chunks → 16 unique cache
    // line requests hit all L1 sets in parallel. The dest v[140:143] is a discard
    // sink: this is the LAST op before the end-of-iter wait_vmcnt0, so the async
    // write lands and is drained before any reuse (same invariant as the 32x32).
    auto prefetch_v_t_l1 = [&](int it_next) {
        if (it_next < 0) return;
        const int sc_start = wid_in_g * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int sc = sc_start + i;
            const uint32_t sc_off = (uint32_t)(kVtLayout(it_next, sc, lane) * 2);
            asm volatile(
                "buffer_load_dwordx4 v[140:143], %0, %1, 0 offen"
                :
                : "v"(sc_off), "s"(vr)
                : "v140", "v141", "v142", "v143", "memory"
            );
        }
    };

    // ─── Softmax: bmax fold + exp2 + rescale + pack, per q-group ────────────────
    auto softmax = [&](fp32x4_t sacc[kQGroups][kNSub], bf16x4_t preg[kQGroups][kNSub]) {
        #pragma unroll
        for (int qg = 0; qg < kQGroups; ++qg) {
            // bmax fold happens IN softmax (P2), where the other group is in
            // pv+qk (MFMA-heavy) — the v_max3 chain hides under their MFMA.
            float bmax_local = -FLT_MAX;
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn)
                bmax_local = fmaxf(bmax_local,
                    fmaxf(fmaxf(sacc[qg][sn][0], sacc[qg][sn][1]),
                          fmaxf(sacc[qg][sn][2], sacc[qg][sn][3])));
            constexpr float kRescaleThr = 8.0f;
            const float new_m_local = bmax_local * scale_log2e;
            const bool  first_block = m_max[qg] == -FLT_MAX;
            const bool  all_ok = __all((int)((new_m_local - m_max[qg]) < kRescaleThr)) != 0
                                && !first_block;
            float neg_m;
            float rescale = 1.f;
            if (__builtin_expect(all_ok, 1)) {
                neg_m = -m_max[qg];
            } else {
                const float new_m = fmaxf(m_max[qg], xor_max(bmax_local) * scale_log2e);
                rescale = first_block ? 0.f : exp2c(m_max[qg] - new_m);
                if (!first_block) {
                    // Force in-place v_mul_f32_e32: per-subscript `o[qg][t][i] *=`
                    // compiles to v_pk_mul_f32 (partial-subreg write → mfma C-read
                    // hazard). Inline `+v(o[qg][t][i])` keeps it scalar.
                    #pragma unroll
                    for (int t = 0; t < kDTiles; ++t)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                            asm volatile("v_mul_f32_e32 %0, %1, %0"
                                         : "+v"(o[qg][t][i]) : "v"(rescale));
                }
                m_max[qg] = new_m;
                neg_m = -new_m;
            }
            float bs = 0.f;
            #pragma unroll
            for (int sn = 0; sn < kNSub; ++sn) {
                const float p0 = exp2c(fmaf(sacc[qg][sn][0], scale_log2e, neg_m));
                const float p1 = exp2c(fmaf(sacc[qg][sn][1], scale_log2e, neg_m));
                const float p2 = exp2c(fmaf(sacc[qg][sn][2], scale_log2e, neg_m));
                const float p3 = exp2c(fmaf(sacc[qg][sn][3], scale_log2e, neg_m));
                // NaN-safe pack: a masked row gives all-zero p (safe); a zero
                // l_sum is handled in the epilogue. MaybeNaN scrubs any drift.
                preg[qg][sn] = pack4_bf16</*MaybeNaN=*/true>(p0, p1, p2, p3);
                bs = vadd_f32(bs, (p0 + p1) + (p2 + p3));
            }
            // Cross-lane sum of bs is deferred to the output epilogue (math-equiv
            // since rescale is a global per-row factor).
            l_sum[qg] = vadd_f32(vmul_f32(l_sum[qg], rescale), bs);
        }
    };

    // ─── Main loop ─────────────────────────────────────────────────────────
    //   Group A  P1: pv(V_t[N-1]) + qk(K[N])      P2: softmax + V_t prefetch
    //   Group B  P1: K[N+1] HBM→LDS + softmax     P2: pv(V_t[N-1]) + qk(K[N])
    // V_t prefetch lives in Group A's softmax-only P2 — bursts in the pv loop
    // regress perf by stalling consumer issue on vmcnt pressure.
    wait_vmcnt0();
    lds_barrier();
    fp32x4_t sacc[kQGroups][kNSub];
    bf16x4_t preg[kQGroups][kNSub];
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

        // Phase 1
        if (group_a) {
            // qk LDS reads (lgkmcnt) fired BEFORE pv_phase so they hide under
            // pv's mfmas; consumed in qk_phase. sched_barrier locks placement.
            prefetch_qk(k_byte_base, s0, s1, s2);
            if (!first) pv_phase(nb - 1, preg);
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
            if (!first) pv_phase(nb - 1, preg);
            qk_phase(k_byte_base, s0, s1, s2, sacc);
        }

        // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes
        // (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so the next iter's
        // qk_phase doesn't read stale K. lds_barrier covers the LGKMCNT side.
        wait_vmcnt0();
        lds_barrier();
        first = false;
    }

    // Epilogue: drain. Group B softmaxes its final qk; both groups pv on the
    // last V tile.
    if (!group_a) softmax(sacc, preg);
    {
        const int it = nblocks - 1;
        pv_phase(it, preg);
    }

    // ─── Output ────────────────────────────────────────────────────────────
    // l_sum is summed across the 4 quad-lanes once here (math-equiv since
    // rescale is global to the row). invl=0 when l_sum==0 prevents 0×inf=NaN.
    #pragma unroll
    for (int qg = 0; qg < kQGroups; ++qg) l_sum[qg] = xor_sum(l_sum[qg]);
    #pragma unroll
    for (int qg = 0; qg < kQGroups; ++qg) {
        if (!valid[qg]) continue;
        const float invl = (l_sum[qg] == 0.0f) ? 0.0f : rcpf(l_sum[qg]);
        // No PV mfma follows so a packed v_pk_mul_f32 is safe here. MaybeNaN
        // pack4 scrubs any residual NaN/Inf in o from pathological inputs.
        #pragma unroll
        for (int t = 0; t < kDTiles; ++t) {
            o[qg][t] = o[qg][t] * invl;
            const bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(
                o[qg][t][0], o[qg][t][1], o[qg][t][2], o[qg][t][3]);
            bst8(or_, kQOLayout(qbase[qg] + col, t * kKTile + quad * 4) * 2, out);
        }
    }
}

// ─── V transpose for 16x16x16 mfma ─────────────────────────────────────────────
// Pre-packs V[seq,128] into V_t so pv_phase reads each lane's MFMA A-fragment with
// one contiguous bld16. Phase 1: V block HBM→LDS (coalesced). Phase 2: re-read in
// fragment order → V_t (kVtLayout, shared with pv_phase). Super-chunk sc = dt*2 +
// sn_pair; lane l's 16 bytes are two A-operands (kv-subtiles sn0 then sn1):
//   bytes [l*16 .. l*16+7]   = V[k=sn0*16 + (l/16)*4 + i, hd=dt*16 + l%16]
//   bytes [l*16+8 .. l*16+15] = V[k=sn1*16 + (l/16)*4 + i, hd=dt*16 + l%16]
// V_t layout and the fragment map are drawn in the indexing notebook.
constexpr int kXposeThreadsPerWG = 256;
constexpr auto kVStageWr = opus::make_layout(kShape2, strides<kHeadDim / 2, 4>()); // (row, octet)→dword
constexpr auto kVtScLane = opus::make_layout(kShape2, strides<kVtScElems, kVtLaneElems>()); // (super-chunk, lane)
__global__ __launch_bounds__(kXposeThreadsPerWG, 4)
void v_transpose_kernel_16x16(
    const bf16_t* __restrict__ V,
    bf16_t*       __restrict__ V_t,
    int           seq_len_per_head
) {
    const int it  = blockIdx.x;                 // which 64-row kv-block (over all B·H·S)
    const int tid = threadIdx.x;
    const long long block_base = (long long)it * kVtIterElems;  // bf16 elems before this block

    __shared__ uint32_t sV_dw[kNBlock * (kHeadDim / 2)];   // 64 rows × 64 dw = 16 KB
    const auto vr_x = mkr(V);

    // ── Phase 1: V[block] HBM → LDS, coalesced. 1024 thread-ops cover 64 rows ×
    //    16 head-dim octets; each op copies one bf16x8 (= 4 dwords). ───────────
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int op    = k * kXposeThreadsPerWG + tid;
        const int row   = op >> 4;             // 0..63 kv-row in this block
        const int octet = op & 15;             // 0..15 head-dim octet (×8 bf16)
        u32x4_t v8 = __builtin_amdgcn_raw_buffer_load_b128(
            vr_x, (uint32_t)((block_base + kQOLayout(row, octet * 8)) * 2), 0, 0);
        const int dst = kVStageWr(row, octet);            // dword offset in sV_dw
        sV_dw[dst + 0] = v8[0]; sV_dw[dst + 1] = v8[1];
        sV_dw[dst + 2] = v8[2]; sV_dw[dst + 3] = v8[3];
    }

    __syncthreads();

    // staged tile is bf16, row-major (kv_row, head_dim) — same shape as Q (kQOLayout).
    const bf16_t* sV = reinterpret_cast<const bf16_t*>(sV_dw);
    const auto otr_x = mkr(V_t);

    // ── Phase 2: gather each (super-chunk, lane)'s two A-fragments from LDS → V_t. ──
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int flat_idx = tid * 4 + k;        // 0..1023
        const int sc       = flat_idx >> 6;      // super-chunk 0..15
        const int lane_in  = flat_idx & 63;      // lane 0..63
        const int dt       = sc >> 1;            // hd-tile 0..7
        const int sn_pair  = sc &  1;            // 0..1
        const int sn0      = sn_pair * 2;
        const int sn1      = sn0 + 1;
        const int d        = dt * 16 + (lane_in & 15);
        const int k0_base  = (lane_in >> 4) * 4;

        bf16_t packed[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed[i]     = sV[kQOLayout(sn0 * 16 + k0_base + i, d)];
            packed[4 + i] = sV[kQOLayout(sn1 * 16 + k0_base + i, d)];
        }
        u32x4_t packed_v;
        __builtin_memcpy(&packed_v, packed, sizeof(packed));
        __builtin_amdgcn_raw_buffer_store_b128(
            packed_v, otr_x, (uint32_t)((block_base + kVtScLane(sc, lane_in)) * 2), 0, 0);
    }
    (void)seq_len_per_head;
}

}  // namespace

extern "C" {
// IMPORTANT: V here must point to pre-transposed V_t. Call launch_v_transpose
// first to produce V_t from raw V. V_t and V have identical total byte size.
int launch_attention_forward(
    const void* q, const void* k, const void* v, void* out,
    int batch, int heads, int seq_len, int head_dim, void* stream
) {
    if (batch <= 0 || heads <= 0)             return 1;
    if (seq_len <= 0 || (seq_len % kNBlock))  return 2;
    if (head_dim != kHeadDim)                 return 3;

    const int   bh   = batch * heads;
    const int   qt   = seq_len / kQTileRows;
    const dim3  grid((qt + kNumWaves * kQGroups - 1) / (kNumWaves * kQGroups), bh);
    const float sl2e = kLog2e / sqrtf((float)kHeadDim);

    hipLaunchKernelGGL(
        attention_forward, grid, dim3(kBlockSize), 0, (hipStream_t)stream,
        (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
        (bf16_t*)out, seq_len, sl2e);
    return (int)hipGetLastError();
}

// V transpose: V[seq_len, 128] (bf16) → V_t in pv_phase's consumed layout.
// Caller allocates V_t with same byte size as V. n_rows = batch*heads*seq_len.
int launch_v_transpose(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, void* stream
) {
    if (n_rows <= 0 || (n_rows % kNBlock)) return 2;
    const int iters = n_rows / kNBlock;
    hipLaunchKernelGGL(
        v_transpose_kernel_16x16,
        dim3(iters), dim3(kXposeThreadsPerWG), 0, (hipStream_t)stream,
        (const bf16_t*)v, (bf16_t*)v_t, seq_len_per_head);
    return (int)hipGetLastError();
}
}
