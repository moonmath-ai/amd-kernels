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
// out for direct load_u32x4 consumption in pv_phase — no LDS staging for V.
//
// Key facts for the 16x16x16 MFMA layout:
//   • Lane split: col = lane&15, quad = lane>>4 (4 quads, fp32x4 C-frag).
//   • Each wave handles QTilesPerWave=3 q-tiles of 16 rows.
//   • qk reads K in d-tile pairs {d_pair, d_pair+4}; pv consumes V_t super-chunks.

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
constexpr BF16Round RoundMode = BF16Round::BF16_ROUND;
using fp32x4_t = float    __attribute__((ext_vector_type(4)));  // 16x16x16 C-fragment
using u32x2_t  = uint32_t __attribute__((ext_vector_type(2)));
using u32x4_t  = uint32_t __attribute__((ext_vector_type(4)));

constexpr int   WaveSize         = 64;
constexpr int   NumWaves         = 8;
constexpr int   LoadWaves        = 4;
constexpr int   BlockSize        = WaveSize * NumWaves;
constexpr int   KvBlockRows      = 64;                          // kv rows processed per main-loop iter
constexpr int   MfmaTile         = 16;                          // 16x16x16 mfma tile (M=N)
constexpr int   KvSubtiles       = KvBlockRows / MfmaTile;    // 4 kv-subtiles / block
constexpr int   MfmaK            = 16;                          // mfma K (contraction)
constexpr int   HeadDim          = 128;
constexpr int   HeadDimTiles     = HeadDim / MfmaK;           // 8 head-dim tiles
constexpr int   QTileRows        = 16;                          // rows per q-tile (one mfma)
constexpr int   QTilesPerWave    = 3;                           // → 48 q-rows / wave (3 q-tiles)
constexpr int   QTilesPerWG      = NumWaves * QTilesPerWave;  // 24 q-tiles = 384 rows / workgroup
constexpr int   KRowDwords       = HeadDim / 2;                // 64 = real K row width (dwords)
constexpr int   KLdsStrideDwords = KRowDwords;               // 64 = LDS row stride (XOR-swizzled, no pad)
constexpr int   KRowsPerLoadWave = KvBlockRows / LoadWaves;  // 16
constexpr float Log2e            = 1.4426950408889634f;
constexpr uint32_t BufferConfig  = 0x00020000u;  // SRD word3: DATA_FORMAT=32 (raw 32-bit elems)

// Online-softmax rescale gate: rescale o_acc only when a block raises the running max by
// MORE than RescaleThreshold (log2). DEFAULT 0 = exact running max → dominant softmax prob
// is exactly 1.0 (lossless in bf16) → output rmse vs an fp32 reference ~6e-5; a positive threshold lets the max
// lag (dominant prob exp2(lag)≠1.0, bf16-packs with ~2^-8 error → rmse ~1.2e-3 at 8). It's
// perf-neutral on this warp-spec pipeline (rescale hides in P2 under the other group's MFMAs),
// so 0 costs nothing. Override only where softmax is on the critical path: -DRESCALE_THRESHOLD=8.0f
#ifndef RESCALE_THRESHOLD
#define RESCALE_THRESHOLD 0.0f
#endif
constexpr float RescaleThreshold = RESCALE_THRESHOLD;

__device__ __forceinline__ fp32x4_t zero_fp32x4() { return fp32x4_t{0,0,0,0}; }

__device__ __forceinline__ float fast_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float fast_rcp (float x) { return __builtin_amdgcn_rcpf (x); }

// Inline-asm mfma: forces in/out same VGPR for accumulator (no compiler rename
// → no v_mov copies) and keeps issue in source order via asm-volatile.
__device__ __forceinline__ void asm_mfma(bf16x4_t a, bf16x4_t b, fp32x4_t& c) {
    asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
                 : "+v"(c) : "v"(a), "v"(b));
}

// Literal-0 C operand skips the v_mov init the compiler emits to zero a fresh
// accumulator. "=&v" prevents aliasing the dst onto the A VGPRs.
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
__device__ __forceinline__ float vfma_f32(float a, float b, float c) {  // a*b + c
    float r; asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}
__device__ __forceinline__ bool wave_all(bool p) { return __all((int)p) != 0; }

// bf16 packing. MaybeNaN=true guards 0×inf NaN. The NaN guard is left to the compiler
// (v_cmp_u + fused v_add3 / builtin select) rather than pinned inline asm — at 3 q-groups
// the compiler-scheduled form holds ~50 VGPR fewer in the softmax pack (pinning f + an out
// copy per round, ×4 per pack, spills the 256-VGPR budget).
template <bool MaybeNaN>
__device__ __forceinline__ unsigned bf16_round_one(float f) {
    if constexpr (RoundMode == BF16Round::RTNE) {
        if constexpr (MaybeNaN) {
            uint32_t u_bits = __builtin_bit_cast(uint32_t, f);
            uint32_t tmp;
            u32x2_t  nan_mask;
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
    } else if constexpr (RoundMode == BF16Round::RTNA) {
        unsigned biased = __builtin_bit_cast(unsigned, f) + 0x8000u;   // ties-away rounding bias
        if constexpr (MaybeNaN) {
            // NaN → canonical 0x7FFF0000. The check MUST be inline asm:
            // -ffast-math (-ffinite-math-only) folds __builtin_isunordered to false, silently
            // disabling a C-level guard — RTNA would then leak the biased NaN bits (0xFFC0)
            // instead of 0x7FFF. (RTNE's guard survives only because it too is inline asm.)
            u32x2_t nan_mask;
            asm volatile(
                "v_cmp_u_f32 %0, %2, %2\n"
                "v_cndmask_b32 %1, %1, %3, %0"
                : "=s"(nan_mask), "+v"(biased)
                : "v"(f), "v"(0x7FFF0000u)
            );
        }
        return biased;
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

// Wrap a global pointer in a buffer resource descriptor (SRD = the 128-bit "V#" that
// raw_buffer_load/store address through). Args: (base, stride=0 → not structured,
// num_records=0xFFFFFFFF → effectively unbounded, config=BufferConfig). const_cast
// because the builtin takes a non-const base; a load resource is never written through.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer(const void* p, uint32_t num_records = 0xFFFFFFFFu) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, num_records, BufferConfig);
}
// Raw buffer load/store, named by the value moved (bf16x4 = 8 B, u32x4 = 16 B).
__device__ __forceinline__ bf16x4_t load_bf16x4(__amdgpu_buffer_rsrc_t buffer, int byte_off) {
    u32x2_t raw = __builtin_amdgcn_raw_buffer_load_b64(buffer, byte_off, 0, 0);
    bf16x4_t out; __builtin_memcpy(&out, &raw, sizeof(out)); return out;
}
__device__ __forceinline__ u32x4_t load_u32x4(__amdgpu_buffer_rsrc_t buffer, int byte_off) {
    return __builtin_amdgcn_raw_buffer_load_b128(buffer, byte_off, 0, 0);
}
__device__ __forceinline__ void store_bf16x4(__amdgpu_buffer_rsrc_t buffer, int byte_off, bf16x4_t v) {
    u32x2_t bits; __builtin_memcpy(&bits, &v, 8);
    __builtin_amdgcn_raw_buffer_store_b64(bits, buffer, byte_off, 0, 0);
}
// Split a u32x4 (one b128 K/V load = a d-tile pair) into its two bf16x4 MFMA operands:
// lo2 = the low d-tile (dwords 0,1), hi2 = the high d-tile (dwords 2,3).
__device__ __forceinline__ bf16x4_t lo2(u32x4_t v) { return __builtin_bit_cast(bf16x4_t, u32x2_t{v[0], v[1]}); }
__device__ __forceinline__ bf16x4_t hi2(u32x4_t v) { return __builtin_bit_cast(bf16x4_t, u32x2_t{v[2], v[3]}); }

// Low 32 bits of an LDS pointer = its byte offset within the LDS segment. Hides
// the pointer→integer cast at call sites.
template <typename T>
__device__ __forceinline__ uint32_t lds_addr(const T* p) { return (uint32_t)(uintptr_t)p; }

// Broadcast a wave-uniform value into a scalar register (SGPR) via readfirstlane.
// Every lane computes the same value here, so this lets the compiler use scalar
// addressing instead of carrying it in a per-lane VGPR.
__device__ __forceinline__ uint32_t to_scalar(uint32_t v) {
    return __builtin_amdgcn_readfirstlane(v);
}

// Immediate-offset ds_read2_b64: Offset0/Offset1 (units of 8 B) are baked into the
// instruction; <0,16> fetches a d-tile pair in one op — offset0=0 is d-tile d_pair,
// offset1=16 (+128 B) is the hi d-tile d_pair+4. The d_pair iteration rides the swizzled
// base (k_addr's per-read xor+shift), not the immediate. %c emits the bare constant.
template <int Offset0, int Offset1>
__device__ __forceinline__ u32x4_t ds_read2_b64_imm(uint32_t lds_base) {
    u32x4_t result;
    asm volatile("ds_read2_b64 %0, %1 offset0:%c2 offset1:%c3"
                 : "=&v"(result) : "v"(lds_base), "i"(Offset0), "i"(Offset1));
    return result;
}

// Pinned lgkmcnt wait — anchors the consumer so the compiler can't reorder above.
// Count = how many outstanding LDS reads may remain in flight past this point.
#define WAIT_LGKM_CASE(N) \
    if constexpr (Count == N) asm volatile("s_waitcnt lgkmcnt(" #N ")" : "+v"(anchor) :: "memory");
template <int Count, typename T>
__device__ __forceinline__ void wait_lgkm(T& anchor) {
    static_assert(Count >= 0 && Count <= 7, "Count out of range");
    WAIT_LGKM_CASE(0) else WAIT_LGKM_CASE(1) else WAIT_LGKM_CASE(2) else
    WAIT_LGKM_CASE(3) else WAIT_LGKM_CASE(4) else WAIT_LGKM_CASE(5) else
    WAIT_LGKM_CASE(6) else WAIT_LGKM_CASE(7);
}
#undef WAIT_LGKM_CASE

__device__ __forceinline__ void wait_vmcnt0() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void wait_lgkmcnt0() { __builtin_amdgcn_s_waitcnt(0x007F); }
__device__ __forceinline__ void lds_barrier() {
    wait_lgkmcnt0();  // lgkmcnt(0)
    __builtin_amdgcn_s_barrier();
}
// Issue-priority hints: in the 8-wave warp-spec, the MFMA-heavy group raises
// priority so it wins issue arbitration over the memory/VALU group, then lowers
// it back. sched_fence = compiler scheduling barrier (pins instruction order so
// hand-placed prefetches don't sink past it). All fold to one inline op each.
__device__ __forceinline__ void mfma_prio_hi() { asm volatile("s_setprio 3" ::: "memory"); }
__device__ __forceinline__ void mfma_prio_lo() { asm volatile("s_setprio 0" ::: "memory"); }
__device__ __forceinline__ void sched_fence()  { __builtin_amdgcn_sched_barrier(0); }

// opus layouts: offset(coords) = Σ coordᵢ·strideᵢ, in bf16 ELEMENTS (×2 → bytes).
// Static strides (opus::number<>) fold to shifts at compile time.
template <int... S>
constexpr auto strides() { return opus::make_tuple(opus::number<S>{}...); }
constexpr auto Shape2 = opus::make_tuple(opus::number<1>{}, opus::number<1>{});
constexpr auto Shape3 = opus::make_tuple(opus::number<1>{}, opus::number<1>{}, opus::number<1>{});

// Q / Out: row-major (seq, head_dim).
constexpr auto QOLayout = opus::make_layout(Shape2, strides<HeadDim, 1>());

// V_t (pre-transposed): (kv-iter, super-chunk, lane). bf16x8 / lane.
constexpr int  VtChunks     = 16;                         // super-chunks / kv-block
constexpr int  VtBlockElems = KvBlockRows * HeadDim;    // 8192 elems / kv-block
constexpr int  VtChunkElems = VtBlockElems / VtChunks;  // 512  elems / super-chunk
constexpr int  VtLaneElems  = 8;                          // bf16x8 / lane
constexpr auto VtLayout = opus::make_layout(
    Shape3, strides<VtBlockElems, VtChunkElems, VtLaneElems>());

// K addressed in DWORDS (2 bf16), row-major (kv_row, dword):
//   HBM source: 64 dw/row.  LDS dest: same stride 64, but XOR-swizzled (see direct_lds_k
//   and qk_swz_base) so the 32-way qk-read bank conflict (stride 64 ≡ 0 mod 32) is broken
//   without padding (padded stride 66 is slower here — non-pow2 address mults).
constexpr auto KLdsDword = opus::make_layout(Shape2, strides<KLdsStrideDwords, 1>());

// HBM (seq, head-dim/dword) → offset layout for Q/K/Out, parameterized by the input layout.
// BHSD: rows per-head contiguous → static seq-row stride Stride (byte- and ISA-identical).
// BSHD: heads interleaved → seq-row stride heads·Stride (dynamic). Unused branch folds away.
template <bool kBSHD, int Stride>
__device__ __forceinline__ auto seq_layout(int heads) {
    if constexpr (kBSHD) return opus::make_layout(Shape2, opus::make_tuple(heads * Stride, opus::number<1>{}));
    else                 return opus::make_layout(Shape2, opus::make_tuple(opus::number<Stride>{}, opus::number<1>{}));
}

// qg2 LDS park: the ENTIRE q-group 2 (all HeadDimTiles d-tiles) staged in LDS, addressed in
// DWORDS by (wave, d-tile, lane); each lane holds a bf16x4 (2 dwords) → lane stride 2.
constexpr int  Q2HalfTiles = HeadDimTiles / 2;                         // lo→hi d-tile offset (d_pair vs d_pair+4)
constexpr int  Q2LdsDwords = NumWaves * HeadDimTiles * WaveSize * 2; // 8192 dwords = whole qg2 (32 KB)
constexpr auto Q2Lds = opus::make_layout(Shape3, strides<HeadDimTiles * WaveSize * 2, WaveSize * 2, 2>());

// ─── Grid + MFMA thread→fragment index maps ─────────────────────────────────
// Same rule as the memory layouts above (offset = Σ coordᵢ·strideᵢ) but these
// yield LOGICAL indices — which q-tile / which kv-row / which fragment element —
// not byte offsets, so callers don't multiply by 2.

// (block_x, wave, q-group) → absolute q-tile index. A workgroup owns
// QTilesPerWG tiles, a wave owns QTilesPerWave, a q-group is one tile.
constexpr auto QTileGrid = opus::make_layout(Shape3, strides<QTilesPerWG, QTilesPerWave, 1>());

// (load-wave, local row) → kv-row that load-wave owns. LoadWaves waves ×
// KRowsPerLoadWave rows tile the KvBlockRows-row K block.
constexpr auto KLoadRow  = opus::make_layout(Shape2, strides<KRowsPerLoadWave, 1>());

// (load-wave, i) → V_t super-chunk to L1-prefetch (VtChunks chunks / LoadWaves).
constexpr auto VtPrefetch = opus::make_layout(Shape2, strides<VtChunks / LoadWaves, 1>());

// MFMA 16x16x16 A/B fragment: lane = (col = lane&15, quad = lane>>4); the 16-wide
// contraction is tiled by 4 quad-lanes, each holding QuadSpan contiguous elems.
constexpr int  QuadSpan  = MfmaK / 4;   // 4 bf16 contraction elems / quad-lane
// (d-tile, quad) → head-dim element, for Q / Out (bf16 elements).
constexpr auto HdFrag    = opus::make_layout(Shape2, strides<MfmaK, QuadSpan>());

// HBM→LDS direct dword load. Inline-asm `s_mov_b32 m0` (not s_add chain) so the
// compiler sees m0 as a fresh write per call — Clang's cselect pattern around
// the builtin form can substitute m0=-1 when lds_dest is small, silently
// dropping writes.
__device__ __forceinline__ void buffer_load_b32_to_lds(
    __amdgpu_buffer_rsrc_t buffer, uint32_t lds_dest,
    uint32_t voff, uint32_t soff
) {
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "buffer_load_dword %1, %2, %3 offen lds"
        :
        : "s"(lds_dest), "v"(voff), "s"(buffer), "s"(soff)
        : "memory"
    );
}

// Head-first chiplet swizzle: chunk_size = blocks_per_head co-locates all
// q-blocks of one (B,H) on a single XCD's L2 → K+V stay resident.
constexpr int NumXcds = 8;
__device__ __forceinline__ int chiplet_swizzle(int wg_id, int num_wgs, int chunk_size) {
    const int block = NumXcds * chunk_size;
    const int limit = (num_wgs / block) * block;
    if (wg_id >= limit) return wg_id;
    const int xcd          = wg_id % NumXcds;
    const int local_id    = wg_id / NumXcds;
    const int chunk_idx    = local_id / chunk_size;
    const int pos_in_chunk = local_id % chunk_size;
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

// Cross-lane reduce over {l, l^16, l^32, l^48}: in 16x16x16 C-layout each lane
// outputs at q_row = col, so the 4 quad-lanes sharing col span all 64 kvs of the
// q-row. ds_swizzle 0x401F folds l^16, ds_bpermute folds l^32 (covers l^48 too).
// Explicit lgkm(0) wait avoids a race when the bpermute is the only LGKM in flight.
// Cross-32-half bpermute. The source-lane index ((lane^32)<<2) is computed INSIDE the asm
// from mbcnt each call — recomputable/pure, so the allocator never pins a long-lived index
// VGPR that would SPILL at the softmax-top peak (the kernel runs at the 256-VGPR ceiling).
// The lgkmcnt(0) wait is BAKED IN right after the bpermute so the result read can't run
// before it (a correctness guard tied to THIS bpermute).
__device__ __forceinline__ int bperm_xor32(int data) {
    int idx, r;
    asm volatile(
        "v_mbcnt_lo_u32_b32 %0, -1, 0\n"
        "v_mbcnt_hi_u32_b32 %0, -1, %0\n"
        "v_lshlrev_b32_e32 %0, 2, %0\n"
        "v_xor_b32_e32 %0, 0x80, %0\n"
        "ds_bpermute_b32 %1, %0, %2\n"
        "s_waitcnt lgkmcnt(0)"
        : "=&v"(idx), "=&v"(r) : "v"(data) : "memory");
    return r;
}
__device__ __forceinline__ float cross_lane_max(float v) {
    const int peer16  = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
    const float partial = fmaxf(v, __builtin_bit_cast(float, peer16));
    const int peer32  = bperm_xor32(__builtin_bit_cast(int, partial));
    return fmaxf(partial, __builtin_bit_cast(float, peer32));
}
__device__ __forceinline__ float cross_lane_sum(float v) {
    const int peer16  = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
    const float partial = v + __builtin_bit_cast(float, peer16);
    const int peer32  = bperm_xor32(__builtin_bit_cast(int, partial));
    return partial + __builtin_bit_cast(float, peer32);
}

// Online-softmax update for ONE q-group. Extracting this per-q-group keeps each q-group's
// softmax temporaries scoped to the call, which holds the 3-q-group kernel under the 256-VGPR
// cap — inlining all q-groups' softmax into the kernel scope spills ~110 VGPR. Folds the block
// into the running (max, sum, o_acc): skip-rescale gate (RescaleThreshold), then exp2 +
// NaN-safe pack of probs, then row_sum. Cross-lane sum is deferred to the epilogue (rescale is
// a per-row global factor → math-equivalent).
__device__ __forceinline__ void softmax_update(
    fp32x4_t scores[], fp32x4_t oreg[], bf16x4_t probs[],
    float& row_m, float& row_l, float scale_log2e, float bmax_local
) {
    const float new_m_local = bmax_local * scale_log2e;
    const bool  first_block = row_m == -FLT_MAX;
    const bool  lane_ok = (new_m_local - row_m) < RescaleThreshold;
    const bool  all_ok  = wave_all(lane_ok) && !first_block;

    float neg_m;
    float rescale = 1.f;
    if (__builtin_expect(all_ok, 1)) {
        neg_m = -row_m;
    } else {
        const float bmax  = cross_lane_max(bmax_local);
        const float old_m = row_m;
        const float new_m = fmaxf(old_m, bmax * scale_log2e);
        if (new_m != old_m) {
            rescale = first_block ? 0.f : fast_exp2(old_m - new_m);
            if (!first_block) {
                // o_acc writeback fence: the "+v"(oreg[t]) pins + "memory" clobber order this
                // rescale after pv's asm MFMA writes (LLVM cannot see those defs); the same-wave
                // MFMA->VALU hardware interlock covers writeback latency.
                asm volatile(""
                    : "+v"(oreg[0]),"+v"(oreg[1]),"+v"(oreg[2]),"+v"(oreg[3]),
                      "+v"(oreg[4]),"+v"(oreg[5]),"+v"(oreg[6]),"+v"(oreg[7]) :: "memory");
                #pragma unroll
                for (int t = 0; t < HeadDimTiles; ++t) {
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][0]) : "v"(rescale));
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][1]) : "v"(rescale));
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][2]) : "v"(rescale));
                    asm volatile("v_mul_f32_e32 %0, %1, %0" : "+v"(oreg[t][3]) : "v"(rescale));
                }
            }
        }
        row_m = new_m;
        neg_m = -new_m;
    }

    float bs[KvSubtiles];
    #pragma unroll
    for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
        const float p0 = fast_exp2(fmaf(scores[kv_sub][0], scale_log2e, neg_m));
        const float p1 = fast_exp2(fmaf(scores[kv_sub][1], scale_log2e, neg_m));
        const float p2 = fast_exp2(fmaf(scores[kv_sub][2], scale_log2e, neg_m));
        const float p3 = fast_exp2(fmaf(scores[kv_sub][3], scale_log2e, neg_m));
        bs[kv_sub] = (p0 + p1) + (p2 + p3);
        // No NaN guard here (hot path): exp2 yields [0,1] or NaN — never Inf — and a NaN packs
        // to *some* bf16 NaN that the PV MFMA turns into an fp32 NaN regardless of bit pattern.
        // The output pack (epilogue) then canonicalizes that to 0x7FFF. Guarding
        // here too would just cost a cmp+cndmask per element with no effect on the final result.
        probs[kv_sub] = pack4_bf16</*MaybeNaN=*/false>(p0, p1, p2, p3);
    }
    const float bsum = vadd_f32(vadd_f32(bs[0], bs[1]), vadd_f32(bs[2], bs[3]));
    row_l = vfma_f32(row_l, rescale, bsum);
}

template <bool kBSHD>
__global__ __launch_bounds__(BlockSize, 1)
void attention_forward(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ K,
    const bf16_t* __restrict__ V,    // points to V_t (pre-transposed)
    bf16_t* __restrict__ Out,
    int seq_len, int seq_len_kv, float scale_log2e,  // seq_len = Q rows; seq_len_kv = K/V rows (cross-attn)
    int heads                                        // #heads — only needed for BSHD addressing (kBSHD)
) {
    // MFMA 16x16x16 lane split: col = q-col / kv-row in tile, quad = which 4 of 64.
    // Tile dims walked by the loops: kv_sub (kv-subtile), d_pair (d-tile pair), d_tile (hd-tile),
    // q_group (q-group). 8 waves → 2 groups of 4 for warp-spec.
    const int  tid           = threadIdx.x;
    const int  wave_id       = tid >> 6;
    const int  lane          = tid & 63;
    const int  col           = lane & 15;   // q-col / kv-row within a 16-wide tile
    const int  quad          = lane >> 4;   // which group of 4 lanes (0..3)
    const bool group_a       = (wave_id < LoadWaves);
    const int  wave_in_group = group_a ? wave_id : (wave_id - LoadWaves);

    // K-LDS XOR swizzle (stride-64, bank-conflict-free) — matches direct_lds_k's HBM-source
    // swizzle. Read-offset fields are bit-disjoint so the per-(d_pair,kv_sub,k_buf) address is
    // an OR (≡ add): col=kv_row [8-11] | kv_sub [12-13] | k_buf [14] | (quad 8-byte unit ^
    // col_lo) [3-4] | (d_pair ^ col_hi) [5-6]; the d-tile-high (+4) rides ds_read2 offset1=16.
    const uint32_t col_lo = (uint32_t)(col & 3);
    const uint32_t col_hi = (uint32_t)(col >> 2);
    const uint32_t qk_swz_base = (uint32_t)(((unsigned)col << 8) | (((unsigned)quad ^ col_lo) << 3));

    // Grid: x = #q-blocks (384-row blocks = ceil(seq_len/384)), y = batch·heads.
    // One workgroup (CTA) per (q-block, batch·head).
    const int num_q_blocks = gridDim.x;
    const int num_bh       = gridDim.y;
    const int wg_id_raw = blockIdx.y * num_q_blocks + blockIdx.x;
    const int num_wgs   = num_q_blocks * num_bh;
    // Head-first chiplet swizzle co-locates a (B,H)'s q-blocks on one XCD's L2
    // (~+1.4% @ moderate S) so K+V stay resident.
    const int wg_id     = chiplet_swizzle(wg_id_raw, num_wgs, num_q_blocks);
    const int q_block   = wg_id % num_q_blocks;   // which q-block
    const int bh_index  = wg_id / num_q_blocks;   // which (batch·head)

    // Q-tile ownership: each wave owns QTilesPerWave (=3) 16-row q-tiles. For each,
    // q_tile_start_row is the absolute seq-row where that tile begins — a per-tile
    // origin used for addressing; lane `col` later reads/writes row (start + col).
    // Parallel arrays valid/q_tile_start_row/q_frag/o_acc/scores are all indexed by
    // q_group; valid[] masks a tile that runs past the sequence end.
    const int num_q_tiles = (seq_len + QTileRows - 1) / QTileRows;  // ceil
    bool valid[QTilesPerWave];
    int  q_tile_start_row[QTilesPerWave];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
        const int q_tile = QTileGrid(q_block, wave_id, q_group);
        valid[q_group] = q_tile < num_q_tiles;
        q_tile_start_row[q_group] = valid[q_group] ? q_tile * QTileRows : 0;
    }

    // ceil block count; last_kv_rows ∈ [1,64] = valid rows in the last (partial) K/V block.
    const int num_kv_blocks = (seq_len_kv + KvBlockRows - 1) / KvBlockRows;
    const int last_kv_rows  = seq_len_kv - (num_kv_blocks - 1) * KvBlockRows;

    // Cross-attention offsets. Q/K/Out keep the TRUE seq lens with their SRD bounded to the valid
    // byte extent: a partial last q-tile / K block (rows past the true seq len) then reads 0 /
    // drops its store IN HARDWARE — zero-VGPR OOB handling, no caller padding, no per-row gates.
    // V_t is per-head padded to num_kv_blocks·64 rows by v_transpose (tail zeroed) → in-bounds.
    // Aligned → bounds = exact extent (no-op) → byte-identical (BHSD).
    // Only the HBM addressing of Q/K/Out depends on the layout (masking + the LDS/V_t hot loop
    // don't). Shadow the (seq,hd)→offset opus layouts with the layout's seq-row stride; head base below.
    const auto QOLayout  = seq_layout<kBSHD, HeadDim>(heads);      // Q/Out (seq, head-dim elems)
    const auto KHbmDword = seq_layout<kBSHD, KRowDwords>(heads);   // K     (seq, dwords)
    const int  k_row_dwords = kBSHD ? heads * KRowDwords : KRowDwords;
    long long bh_offset_q, bh_offset_kv;
    if constexpr (kBSHD) {
        const long long b = bh_index / heads, h = bh_index % heads;
        bh_offset_q  = b * (long long)seq_len    * heads * HeadDim + h * HeadDim;
        bh_offset_kv = b * (long long)seq_len_kv * heads * HeadDim + h * HeadDim;
    } else {
        bh_offset_q  = (long long)bh_index * seq_len    * HeadDim;
        bh_offset_kv = (long long)bh_index * seq_len_kv * HeadDim;
    }
    const long long bh_offset_v  = (long long)bh_index * num_kv_blocks * KvBlockRows * HeadDim;
    // BHSD: per-head contiguous (seq·HeadDim·2). BSHD: rows interleaved → ×heads addressable span.
    const uint32_t  q_bytes  = (uint32_t)((long long)seq_len    * (kBSHD ? heads : 1) * HeadDim * 2);
    const uint32_t  kv_bytes = (uint32_t)((long long)seq_len_kv * (kBSHD ? heads : 1) * HeadDim * 2);
    const auto q_buffer  = make_buffer(Q + bh_offset_q,  q_bytes);
    const auto k_buffer  = make_buffer(K + bh_offset_kv, kv_bytes);
    const auto v_buffer  = make_buffer(V + bh_offset_v);
    const auto out_buffer = make_buffer(Out + bh_offset_q, q_bytes);

    constexpr int KLdsBufDwords = KvBlockRows * KLdsStrideDwords;   // 4096 dwords/buffer (64 rows × 64)
    __shared__ uint32_t lds_k[2][KLdsBufDwords];   // 32 KB (2-buf, stride 64)
    // V LDS staging removed — V_t is consumed directly from L1 via load_u32x4.
    // Single-group Q-park: q-groups 0,1 stay fully in VGPR (clean, no LDS reads); the ENTIRE
    // q-group 2 (all 8 d-tiles) is parked in one 32 KB LDS array (Q2Lds, d-tile t at slot t). qk
    // reads the lo MFMA operand from d-tile d_pair, the hi operand from d-tile d_pair+Q2HalfTiles.
    // 32 KB K + 32 KB qg2 = exactly 64 KB LDS. qg2 reads (lgkmcnt) overlap pv's V loads (vmcnt).
    __shared__ uint32_t lds_q2[Q2LdsDwords];       // 32 KB (entire qg2, d-tiles 0..7)
    auto load_q2 = [&](int d_tile) -> bf16x4_t {    // d-tile d_tile of qg2 for this lane
        return *reinterpret_cast<const bf16x4_t*>(&lds_q2[Q2Lds(wave_id, d_tile, lane)]);
    };

    // K[kv_block] HBM→LDS into double-buffer half `k_buf` (group B). Wave w owns rows
    // [w·16, +16); for each, its 64 lanes stream one full K row. The XOR swizzle is realized
    // on the HBM SOURCE: lane L fetches the dword of swizzled 8-byte unit (L>>1)^c keeping
    // the in-unit dword (L&1); c = kv_row & 0xF. LDS dest is hardware-contiguous (stride 64),
    // so LDS unit p ends up holding HBM unit (p^c) — matching qk_phase's swizzled read.
    constexpr uint32_t KRowBytes   = (uint32_t)(KLdsStrideDwords * 4);
    // K block stride (bytes) = KvBlockRows seq-rows × the seq-row byte stride. BHSD folds to the
    // contiguous block size KvBlockRows·HeadDim·2 = 16384; BSHD is ×heads (rows interleaved).
    const uint32_t KBlockBytes = (uint32_t)(KvBlockRows * k_row_dwords * 4);
    const int wave_row0 = KLoadRow(wave_in_group, 0);   // first kv-row this load-wave owns
    const uint32_t k_row_bytes = (uint32_t)(k_row_dwords * 4);   // per-seq-row HBM byte stride
    auto direct_lds_k = [&](int kv_block, int k_buf) {
        const uint32_t lds_row0  = to_scalar(lds_addr(&lds_k[k_buf][KLdsDword(wave_row0, 0)]));
        const uint32_t hbm_block = to_scalar((uint32_t)((uint64_t)kv_block * KBlockBytes));
        if constexpr (kBSHD) {
            // BSHD seq-row stride is a runtime heads·KRowDwords MULTIPLY → if the full per-lane byte
            // offset (kv_row·stride + swz_lane)·4 is the offen VGPR, LLVM can't rematerialize it (the
            // multiply) so it pins all 16 per-c values live across the main loop and spills 4 (reloaded
            // EVERY iter). Split the address into a SCALAR soffset and a cheap per-lane offen VGPR,
            // mirroring how BHSD's static stride already compiles (offen = base XOR const):
            //   soffset = hbm_block + (wave_row0+c)·stride·4  — wave-uniform; carried as a scalar that
            //             accumulates +stride·4 per c (s_add chain, like BHSD's m0 +0x100), never a VGPR.
            //   offen   = swz_lane·4 = (lane·4) XOR (c·8) — one `lane·4` base, const XOR per c → 0 spill.
            const uint32_t lane_x4   = (uint32_t)(lane << 2);                              // per-lane base
            const uint32_t base_soff = to_scalar(hbm_block + (uint32_t)wave_row0 * k_row_bytes);
            #pragma unroll
            for (int c = 0; c < KRowsPerLoadWave; ++c) {
                const uint32_t soff = to_scalar(base_soff + (uint32_t)c * k_row_bytes);
                const uint32_t off  = lane_x4 ^ (uint32_t)(c << 3);   // swz_lane·4 = (lane^(c<<1))·4
                buffer_load_b32_to_lds(k_buffer, lds_row0 + c * KRowBytes, off, soff);
            }
        } else {
            // BHSD: static stride → kv_row·stride·4 folds to a shift LLVM rematerializes for free
            // (0 persistent regs, 0 spill). Keep the original single-VGPR-offset form byte-identical.
            #pragma unroll
            for (int c = 0; c < KRowsPerLoadWave; ++c) {
                const int kv_row = KLoadRow(wave_in_group, c);
                const uint32_t swz_lane = (uint32_t)((((lane >> 1) ^ c) << 1) | (lane & 1));
                const uint32_t hbm_byte = (uint32_t)((KHbmDword(kv_row, 0) + swz_lane) * 4);
                buffer_load_b32_to_lds(k_buffer, lds_row0 + c * KRowBytes, hbm_byte, hbm_block);
            }
        }
    };


    // Prologue: first processed K-block → LDS (group B only); group A idle until iter 1.
    const int  first_blk = 0;
    const bool any       = (num_kv_blocks > 0);
    if (!group_a && any) direct_lds_k(first_blk, 0);

    // Persistent VGPR Q (no reload across K-blocks). Lane (q_tile_start_row+col) owns one
    // q-row; each load_bf16x4 grabs 4 head-dims of d-tile d (quad picks which 4 of 16).
    // Single-group park: q-groups 0,1 fully in VGPR; q-group 2 entirely in lds_q2 (d-tile t @ slot t).
    bf16x4_t q_frag[2][HeadDimTiles];
    auto load_q = [&](int q_group, int d) -> bf16x4_t {
        return valid[q_group]
            ? load_bf16x4(q_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(d, quad)) * 2)
            : bf16x4_t{0,0,0,0};
    };
    // NOTE: keep qg0 and qg1 as SEPARATE prologue load loops — collapsing them into one
    // `for (g)` loop re-introduces a VGPR spill (allocator coloring is sensitive to the
    // load structure here). The qk-phase qg0/qg1 collapse is safe; this prologue one is not.
    #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d) q_frag[0][d] = load_q(0, d);
    #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d) q_frag[1][d] = load_q(1, d);
    // q_group=2: all 8 d-tiles → lds_q2 (d-tile d at slot d). HBM→VGPR temp→LDS.
    #pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d) {
        bf16x4_t v = load_q(2, d);
        u32x2_t raw; __builtin_memcpy(&raw, &v, 8);
        const uint32_t off = Q2Lds(wave_id, d, lane);
        lds_q2[off + 0] = raw[0]; lds_q2[off + 1] = raw[1];
    }
    __syncthreads();   // lds_q2 visible to all reads below

    fp32x4_t o_acc[QTilesPerWave][HeadDimTiles];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
        #pragma unroll
        for (int d = 0; d < HeadDimTiles; ++d) o_acc[q_group][d] = zero_fp32x4();
    float row_max[QTilesPerWave], row_sum[QTilesPerWave];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) { row_max[q_group] = -FLT_MAX; row_sum[q_group] = 0.f; }

    // Per-wave state, loop-carried across K-blocks: scores written by qk_phase before softmax
    // reads it, probs by softmax_update before pv_phase reads it (true on every path, including
    // masked q-tiles whose qk operands are zero). The values themselves are never read before
    // written, so this zero-init is logically dead — but it is LOAD-BEARING for register
    // allocation: it gives the allocator a single clean def point for these loop-carried arrays.
    // Dropping it respills 64 VGPR (verified). Keep it.
    fp32x4_t scores[QTilesPerWave][KvSubtiles];
    bf16x4_t probs[QTilesPerWave][KvSubtiles];
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
        #pragma unroll
        for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
            scores[q_group][kv_sub] = zero_fp32x4();
            probs[q_group][kv_sub] = bf16x4_t{0,0,0,0};
        }


    auto qk_phase = [&](int kbuf, fp32x4_t scores[QTilesPerWave][KvSubtiles]) {
        // Swizzled LDS read address (OR of disjoint bit fields, see qk_swz_base): d_pair^col_hi
        // is a cheap per-read xor+shift; kv_sub and k_buf complete the address.
        const uint32_t lane_off = (uint32_t)kbuf << 14;
        auto k_addr = [&](int kv_sub, int d_pair) -> uint32_t {
            return qk_swz_base | (((uint32_t)d_pair ^ col_hi) << 5) | ((uint32_t)kv_sub << 12) | lane_off;
        };

        constexpr int TotalReads = KvSubtiles * 4;
        constexpr int Depth      = 3;   // K-tile prefetch ring depth
        constexpr int Q2Depth    = 2;   // qg2-from-LDS ping-pong depth
        mfma_prio_hi();
        sched_fence();

        // K pre-issues (depth-3 ring): each ds_read2 fetches a d-tile PAIR (lo @offset0, hi @offset1).
        u32x4_t s0 = ds_read2_b64_imm<0, 16>(k_addr(0, 0));
        u32x4_t s1 = ds_read2_b64_imm<0, 16>(k_addr(1, 0));
        u32x4_t s2 = ds_read2_b64_imm<0, 16>(k_addr(2, 0));

        // qg2 is the only LDS-resident q-group: one depth-2 ping-pong per head-dim half h
        // (h=0 → lo d-tiles 0..3, h=1 → hi d-tiles 4..7). q2[h][slot] = d-tile slot+h·Q2HalfTiles.
        bf16x4_t q2[2][Q2Depth];
        #pragma unroll
        for (int p = 0; p < Q2Depth; ++p)
            #pragma unroll
            for (int h = 0; h < 2; ++h) q2[h][p] = load_q2(p + h * Q2HalfTiles);

        #pragma unroll
        for (int i = 0; i < TotalReads; ++i) {
            const int kv_sub = i % KvSubtiles;
            const int d_pair = i / KvSubtiles;
            const int slot   = i % Depth;
            u32x4_t& cur = (slot == 0) ? s0 : (slot == 1) ? s1 : s2;
            if (i == TotalReads - 1) wait_lgkm<0>(cur);
            else                      wait_lgkm<Depth - 1>(cur);
            // Contract the d-tile pair {lo=d_pair, hi=d_pair+Q2HalfTiles} into every q-group's
            // scores: K from this ring slot, Q from q_frag (qg0,1 in VGPR) or q2 (qg2 in LDS).
            const bf16x4_t k_pair[2] = { lo2(cur), hi2(cur) };
            const int q2_slot = d_pair % Q2Depth;
            #pragma unroll
            for (int g = 0; g < QTilesPerWave; ++g)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const bf16x4_t qf = (g < 2) ? q_frag[g][d_pair + h * Q2HalfTiles] : q2[h][q2_slot];
                    if (d_pair == 0 && h == 0) scores[g][kv_sub] = asm_mfma_zero(k_pair[0], qf);
                    else                       asm_mfma(k_pair[h], qf, scores[g][kv_sub]);
                }
            // Prefetch the next K pair into this ring slot (depth-3 ahead).
            if (i + Depth < TotalReads) {
                const int next_i = i + Depth;
                cur = ds_read2_b64_imm<0, 16>(k_addr(next_i % KvSubtiles, next_i / KvSubtiles));
            }
            // Prefetch the next qg2 d-tile pair (d_pair+Q2Depth) into its ping-pong slot.
            if (kv_sub == KvSubtiles - 1 && d_pair + Q2Depth < Q2HalfTiles) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) q2[h][q2_slot] = load_q2(d_pair + Q2Depth + h * Q2HalfTiles);
            }
        }
        mfma_prio_lo();
    };

    // ─── PV: O += V·Pᵀ via mfma(A=V_t, B=P). 16 super-chunks/iter, each lane
    // load_u32x4s its pre-packed bf16x8 from V_t (VtLayout). Depth-4 prefetch ring.
    // chunk → hd-tile d_tile=chunk/2, kv-subtile pair {kv_sub0,kv_sub1}; all q-groups
    // share the V operand. ──────────────────────────────────────────────────────
    auto chunk_byte_off = [&](int kv_iter, int chunk) -> uint32_t {
        return (uint32_t)(VtLayout(kv_iter, chunk, lane) * 2);   // bf16 elems → bytes
    };

    auto pv_phase = [&](int kv_iter, const bf16x4_t probs[QTilesPerWave][KvSubtiles]) {
        constexpr int PvDepth  = 4;   // V_t prefetch ring depth (VtChunks=16 super-chunks / kv-iter)
        mfma_prio_hi();
        sched_fence();
        u32x4_t v_ring[PvDepth];
        #pragma unroll
        for (int p = 0; p < PvDepth; ++p) v_ring[p] = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, p));
        #pragma unroll
        for (int chunk = 0; chunk < VtChunks; ++chunk) {
            const int d_tile  = chunk >> 1;          // hd-tile 0..7
            const int kv_sub0 = (chunk & 1) * 2;     // this chunk's kv-subtile pair base ({0,1} or {2,3})
            u32x4_t&  cur     = v_ring[chunk % PvDepth];
            // one b128 V load = a kv-subtile pair; v_pair[h] is the operand for kv-subtile kv_sub0+h.
            const bf16x4_t v_pair[2] = { lo2(cur), hi2(cur) };
            // Issue the next ring V-load BEFORE this chunk's MFMAs (latency lead).
            if (chunk + PvDepth < VtChunks)
                cur = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, chunk + PvDepth));
            // v_pair[h] feeds all q-groups back-to-back (each dies in one mfma) before the next
            // half — shorter V-reg lifetime than q-group-outer ordering.
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
                    asm_mfma(v_pair[h], probs[q_group][kv_sub0 + h], o_acc[q_group][d_tile]);
        }
        sched_fence();
        mfma_prio_lo();
    };

    // L1 prefetcher for V_t[kv_iter_next], fired from Group A's softmax-only P2.
    // vmcnt is per-wave so this is independent of any consumer's pending load_u32x4.
    // Wave-split: each of 4 waves takes 4 unique super-chunks → 16 unique cache
    // line requests hit all L1 sets in parallel. The dest v[140:143] is a discard
    // sink: this is the LAST op before the end-of-iter wait_vmcnt0, so the async
    // write lands and is drained before any reuse.
    auto prefetch_v_t_l1 = [&](int kv_iter_next) {
        if (kv_iter_next < 0) return;
        #pragma unroll
        for (int i = 0; i < VtChunks / LoadWaves; ++i) {
            const int chunk = VtPrefetch(wave_in_group, i);
            const uint32_t chunk_off = (uint32_t)(VtLayout(kv_iter_next, chunk, lane) * 2);
            asm volatile(
                "buffer_load_dwordx4 v[140:143], %0, %1, 0 offen"
                :
                : "v"(chunk_off), "s"(v_buffer)
                : "v140", "v141", "v142", "v143", "memory"
            );
        }
    };

    auto softmax = [&](fp32x4_t scores[QTilesPerWave][KvSubtiles], bf16x4_t probs[QTilesPerWave][KvSubtiles],
                             int step) {
        // Per-q-group block max — local: written below before every read, not loop-carried.
        float bmax_local[QTilesPerWave];
        // Partial last K/V block (seq_len_kv%64≠0): K-OOB rows read 0 (SRD-bounded) → score 0,
        // which would pollute the denominator. Force kv-rows ≥ last_kv_rows to -inf (exp2→0).
        // Bound is q_group-independent (computed once, applied to all 3 — compiler CSEs). `step`
        // = block index; partial block is last. No-op for aligned (last_kv_rows==64).
        if (last_kv_rows < KvBlockRows && step == num_kv_blocks - 1) {
            const int row0 = quad * 4;
            #pragma unroll
            for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub)
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (kv_sub * MfmaTile + row0 + i >= last_kv_rows) {
                        scores[0][kv_sub][i] = -FLT_MAX; scores[1][kv_sub][i] = -FLT_MAX; scores[2][kv_sub][i] = -FLT_MAX; }
        }
        // Per-lane bmax fold runs HERE (not in qk_phase). Softmax runs in P2
        // when the other group is in pv+qk (MFMA-heavy) — the v_max3 VALU
        // chain hides under their MFMA writeback.
        #pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
            float bm = -FLT_MAX;
            #pragma unroll
            for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
                bm = fmaxf(bm,
                    fmaxf(fmaxf(scores[q_group][kv_sub][0], scores[q_group][kv_sub][1]),
                          fmaxf(scores[q_group][kv_sub][2], scores[q_group][kv_sub][3])));
            }
            bmax_local[q_group] = bm;
        }
        // ── Fold + online update, per q-group (free function → scoped temporaries). ──
        #pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
            softmax_update(scores[q_group], o_acc[q_group], probs[q_group],
                           row_max[q_group], row_sum[q_group], scale_log2e, bmax_local[q_group]);
    };

    // ─── Main loop ─────────────────────────────────────────────────────────
    //   Group A  P1: pv(V_t[N-1]) + qk(K[N])      P2: softmax + V_t prefetch
    //   Group B  P1: K[N+1] HBM→LDS + softmax     P2: pv(V_t[N-1]) + qk(K[N])
    // V_t prefetch lives in Group A's softmax-only P2 — bursts in the pv loop
    // regress perf by stalling consumer issue on vmcnt pressure.
    wait_vmcnt0();
    lds_barrier();
    // ─── Pipeline over K-blocks (forward 0..num_kv_blocks-1) ────────────────────
    // `step` = processing-step counter (drives K double-buffer parity); `kv_block`/
    // `prev_block` are the blocks being qk'd / pv'd; `next_block` is what group B streams
    // into LDS. Two-phase warp-spec: each iter A and B swap roles across the mid-barrier.
    int  prev_block = -1, step = 0;
    bool have = (num_kv_blocks > 0);

    while (have) {
        const int  kv_block  = step;
        const int  next_block = step + 1;
        const bool has_next  = (step + 1 < num_kv_blocks);
        const int  k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
        const bool first = (step == 0);
        if (group_a) {  // P1
            if (!first) pv_phase(prev_block, probs);
            qk_phase(k_buf_cur, scores);
        } else {
            if (!first) softmax(scores, probs, step - 1);
            if (has_next) direct_lds_k(next_block, k_buf_next);
        }
        __builtin_amdgcn_s_barrier();
        if (group_a) {  // P2
            softmax(scores, probs, step);
            if (has_next) prefetch_v_t_l1(kv_block);
        } else {
            if (!first) pv_phase(prev_block, probs);
            qk_phase(k_buf_cur, scores);
        }
        // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is
        // VMCNT-tracked, not LGKMCNT) so the next iter's qk_phase can't read stale K;
        // lds_barrier covers the LGKMCNT side.
        wait_vmcnt0();
        lds_barrier();
        prev_block = kv_block; ++step;
        have = (step < num_kv_blocks);
    }
    // Epilogue: last block's softmax + pv.
    if (!group_a && step > 0) softmax(scores, probs, step - 1);
    if (step > 0) pv_phase(prev_block, probs);

    // ─── Output ────────────────────────────────────────────────────────────
    // row_sum is summed across the 4 quad-lanes once here (math-equiv since
    // rescale is global to the row). inv_sum=0 when row_sum==0 prevents 0×inf=NaN.
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) row_sum[q_group] = cross_lane_sum(row_sum[q_group]);
    #pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
        if (!valid[q_group]) continue;
        const float inv_sum = (row_sum[q_group] == 0.0f) ? 0.0f : fast_rcp(row_sum[q_group]);
        // No PV mfma follows so a packed v_pk_mul_f32 is safe here. MaybeNaN
        // pack4 scrubs any residual NaN/Inf in o_acc from pathological inputs.
        #pragma unroll
        for (int t = 0; t < HeadDimTiles; ++t) {
            o_acc[q_group][t] = o_acc[q_group][t] * inv_sum;
            const bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(
                o_acc[q_group][t][0], o_acc[q_group][t][1], o_acc[q_group][t][2], o_acc[q_group][t][3]);
            store_bf16x4(out_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(t, quad)) * 2, out);
        }
    }
}

// ─── V transpose for 16x16x16 mfma ─────────────────────────────────────────────
// Pre-packs V[seq,128] into V_t so pv_phase reads each lane's MFMA A-fragment with
// one contiguous load_u32x4. Phase 1: V block HBM→LDS (coalesced). Phase 2: re-read in
// fragment order → V_t (VtLayout, shared with pv_phase). super-chunk = d_tile*2 +
// kv_sub_pair; lane l's 16 bytes are two A-operands (kv-subtiles kv_sub0 then kv_sub1):
//   bytes [l*16 .. l*16+7]   = V[k=kv_sub0*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]
//   bytes [l*16+8 .. l*16+15] = V[k=kv_sub1*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]

constexpr int XposeThreadsPerWG = 256;
constexpr auto VStageWr = opus::make_layout(Shape2, strides<HeadDim / 2, 4>()); // (row, octet)→dword
constexpr auto VtChunkLane = opus::make_layout(Shape2, strides<VtChunkElems, VtLaneElems>()); // (super-chunk, lane)
template <bool kBSHD>
__global__ __launch_bounds__(XposeThreadsPerWG, 4)
void v_transpose_kernel_16x16(
    const bf16_t* __restrict__ V,
    bf16_t*       __restrict__ V_t,
    int           seq_len_per_head,
    int           heads               // #heads — only for BSHD V addressing (kBSHD)
) {
    const int num_kv_blocks = (seq_len_per_head + KvBlockRows - 1) / KvBlockRows;
    const int head     = blockIdx.x / num_kv_blocks;
    const int kv_iter  = blockIdx.x % num_kv_blocks;
    const int tid = threadIdx.x;
    // head = bh index over B·H; for BSHD split into b=head/heads, h=head%heads.
    long long src_block_base;
    if constexpr (kBSHD) {
        const long long b = head / heads, h = head % heads;
        src_block_base = b * (long long)seq_len_per_head * heads * HeadDim + h * HeadDim
                       + (long long)kv_iter * KvBlockRows * heads * HeadDim;   // unpadded BSHD source base
    } else {
        src_block_base = (long long)head * seq_len_per_head * HeadDim
                       + (long long)kv_iter * VtBlockElems;                     // unpadded BHSD source base
    }
    const long long dst_block_base = (long long)head * num_kv_blocks * VtBlockElems
                                   + (long long)kv_iter * VtBlockElems;

    __shared__ uint32_t stage_dw[KvBlockRows * (HeadDim / 2)];   // 64 rows × 64 dw = 16 KB
    const auto v_buffer = make_buffer(V);

    // ── Phase 1: V[block] HBM → LDS, coalesced; rows ≥ seq_len_per_head are zero (padded tail).
    // Shadow QOLayout with the source-V seq-row stride (scoped to phase 1; phase 2 reads the
    // contiguous LDS stage via the global static QOLayout).
    {
    const auto QOLayout = seq_layout<kBSHD, HeadDim>(heads);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int elem_idx = k * XposeThreadsPerWG + tid;
        const int row      = elem_idx >> 4;    // 0..63 kv-row in this block
        const int octet    = elem_idx & 15;    // 0..15 head-dim octet (×8 bf16)
        const bool row_valid = (kv_iter * KvBlockRows + row) < seq_len_per_head;
        u32x4_t v_oct = row_valid
            ? __builtin_amdgcn_raw_buffer_load_b128(
                  v_buffer, (uint32_t)((src_block_base + QOLayout(row, octet * 8)) * 2), 0, 0)
            : u32x4_t{0, 0, 0, 0};
        const int dst_dw = VStageWr(row, octet);            // dword offset in stage_dw
        stage_dw[dst_dw + 0] = v_oct[0]; stage_dw[dst_dw + 1] = v_oct[1];
        stage_dw[dst_dw + 2] = v_oct[2]; stage_dw[dst_dw + 3] = v_oct[3];
    }
    }

    __syncthreads();

    // staged tile is bf16, row-major (kv_row, head_dim) — same shape as Q (QOLayout).
    const bf16_t* stage = reinterpret_cast<const bf16_t*>(stage_dw);
    const auto vt_buffer = make_buffer(V_t);

    // ── Phase 2: gather each (super-chunk, lane)'s two A-fragments from LDS → V_t. ──
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int frag_idx    = tid * 4 + k;       // 0..1023
        const int chunk          = frag_idx >> 6;     // super-chunk 0..15
        const int frag_lane   = frag_idx & 63;     // lane 0..63
        const int d_tile      = chunk >> 1;           // hd-tile 0..7
        const int kv_sub_pair = chunk & 1;            // 0..1
        const int kv_sub0     = kv_sub_pair * 2;
        const int kv_sub1     = kv_sub0 + 1;
        const int hd          = d_tile * 16 + (frag_lane & 15);  // head-dim element 0..127
        const int kv_row0     = (frag_lane >> 4) * 4;            // first of this lane's 4 kv-rows

        bf16_t packed[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed[i]     = stage[QOLayout(kv_sub0 * 16 + kv_row0 + i, hd)];
            packed[4 + i] = stage[QOLayout(kv_sub1 * 16 + kv_row0 + i, hd)];
        }
        u32x4_t packed_v;
        __builtin_memcpy(&packed_v, packed, sizeof(packed));
        __builtin_amdgcn_raw_buffer_store_b128(
            packed_v, vt_buffer, (uint32_t)((dst_block_base + VtChunkLane(chunk, frag_lane)) * 2), 0, 0);
    }
}

}  // namespace

extern "C" {
// IMPORTANT: V here must point to pre-transposed V_t. Call launch_v_transpose
// first to produce V_t from raw V. V_t and V have identical total byte size.
int launch_attention_forward(
    const void* q, const void* k, const void* v, void* out,
    int batch, int heads, int seq_len, int seq_len_kv, int head_dim, int layout, void* stream
) {
    if (batch <= 0 || heads <= 0)             return 1;
    if (seq_len <= 0)                         return 2;   // any Q len (partial last q-tile SRD-padded)
    if (seq_len_kv <= 0)                      return 5;   // any K/V len (partial last block masked)
    if (head_dim != HeadDim)                 return 3;

    const int   bh   = batch * heads;
    const int   qt   = (seq_len + QTileRows - 1) / QTileRows;
    const dim3  grid((qt + QTilesPerWG - 1) / QTilesPerWG, bh);
    const float sl2e = Log2e / sqrtf((float)HeadDim);

    // layout: 0 = BHSD (q,k,v,out are [B,H,S,D]); 1 = BSHD ([B,S,H,D], read natively, no transpose).
    // chevron launch (hipLaunchKernelGGL can't parse the template's comma — single arg here, but
    // keep chevron for symmetry with the BSHD dispatch).
    if (layout)
        attention_forward<true><<<grid, dim3(BlockSize), 0, (hipStream_t)stream>>>(
            (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
            (bf16_t*)out, seq_len, seq_len_kv, sl2e, heads);
    else
        attention_forward<false><<<grid, dim3(BlockSize), 0, (hipStream_t)stream>>>(
            (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
            (bf16_t*)out, seq_len, seq_len_kv, sl2e, heads);
    return (int)hipGetLastError();
}

// V transpose: V[seq_len, 128] (bf16) → V_t in pv_phase's consumed layout.
// Caller allocates V_t with same byte size as V. n_rows = batch*heads*seq_len.
int launch_v_transpose(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream
) {
    if (n_rows <= 0 || seq_len_per_head <= 0 || (n_rows % seq_len_per_head)) return 2;
    const int bh            = n_rows / seq_len_per_head;
    const int num_kv_blocks = (seq_len_per_head + KvBlockRows - 1) / KvBlockRows;
    const int iters         = bh * num_kv_blocks;
    // heads/layout select the source V layout: 0 = BHSD ([B,H,S,D]); 1 = BSHD ([B,S,H,D], strided).
    if (layout)
        hipLaunchKernelGGL(
            v_transpose_kernel_16x16<true>,
            dim3(iters), dim3(XposeThreadsPerWG), 0, (hipStream_t)stream,
            (const bf16_t*)v, (bf16_t*)v_t, seq_len_per_head, heads);
    else
        hipLaunchKernelGGL(
            v_transpose_kernel_16x16<false>,
            dim3(iters), dim3(XposeThreadsPerWG), 0, (hipStream_t)stream,
            (const bf16_t*)v, (bf16_t*)v_t, seq_len_per_head, heads);
    return (int)hipGetLastError();
}
}
