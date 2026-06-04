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
// Structure mirrors the 32x32x8 variant (see L1V-32x32x8 branch); only the
// indexing and MFMA-fragment math differ for the 16x16x16 instruction:
//   • 16x16x16 lane split: col = lane&15, quad = lane>>4 (4 quads, fp32x4 C-frag).
//   • Each wave handles kQTilesPerWave=2 q-tiles of 16 rows (vs one 32-row tile).
//   • qk reads K in d-tile pairs {d_pair, d_pair+4}; pv consumes V_t super-chunks.
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

constexpr int   kWaveSize         = 64;
constexpr int   kNumWaves         = 8;
constexpr int   kLoadWaves        = 4;
constexpr int   kBlockSize        = kWaveSize * kNumWaves;
constexpr int   kKvBlockRows      = 64;                          // kv rows processed per main-loop iter
constexpr int   kMfmaTile         = 16;                          // 16x16x16 mfma tile (M=N)
constexpr int   kKvSubtiles       = kKvBlockRows / kMfmaTile;    // 4 kv-subtiles / block
constexpr int   kMfmaK            = 16;                          // mfma K (contraction)
constexpr int   kHeadDim          = 128;
constexpr int   kHeadDimTiles     = kHeadDim / kMfmaK;           // 8 head-dim tiles
constexpr int   kQTileRows        = 16;                          // rows per q-tile (one mfma)
constexpr int   kQTilesPerWave    = 2;                           // → 32 q-rows / wave
constexpr int   kQTilesPerWG      = kNumWaves * kQTilesPerWave;  // 16 q-tiles = 256 rows / workgroup
constexpr int   kKRowDwords       = kHeadDim / 2;                // 64 = real K row width (dwords)
constexpr int   kKLdsStrideDwords = kKRowDwords + 2;            // 66 = padded LDS row stride
constexpr int   kKRowsPerLoadWave = kKvBlockRows / kLoadWaves;  // 16
constexpr int   kVerdictRing      = 2;  // lite skip-vote LDS ring = live window {step, step-1}; pow2
constexpr float kLog2e            = 1.4426950408889634f;
constexpr uint32_t kBufferConfig  = 0x00020000u;  // SRD word3: DATA_FORMAT=32 (raw 32-bit elems)

__device__ __forceinline__ fp32x4_t zero_fp32x4() { return fp32x4_t{0,0,0,0}; }

__device__ __forceinline__ float fast_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float fast_rcp (float x) { return __builtin_amdgcn_rcpf (x); }

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

// Wrap a global pointer in a buffer resource descriptor (SRD = the 128-bit "V#" that
// raw_buffer_load/store address through). Args: (base, stride=0 → not structured,
// num_records=0xFFFFFFFF → effectively unbounded, config=kBufferConfig). const_cast
// because the builtin takes a non-const base; a load resource is never written through.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, 0xFFFFFFFFu, kBufferConfig);
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

// Low 32 bits of an LDS pointer = its byte offset within the LDS segment. Hides
// the pointer→integer cast at call sites.
template <typename T>
__device__ __forceinline__ uint32_t lds_addr(const T* p) { return (uint32_t)(uintptr_t)p; }

// Broadcast a wave-uniform value into a scalar register (SGPR) via readfirstlane.
// Every lane computes the same value here, so this lets the compiler use scalar
// addressing instead of carrying it in a per-lane VGPR. The int overload keeps
// signed list values (skip-list entries can be -1) in an SGPR without a u32 round-trip.
__device__ __forceinline__ uint32_t to_scalar(uint32_t v) {
    return __builtin_amdgcn_readfirstlane(v);
}
__device__ __forceinline__ int to_scalar(int v) {
    return __builtin_amdgcn_readfirstlane(v);
}

// Immediate-offset ds_read2: kOffset0/kOffset1 are compile-time (units of 8 B) baked
// into the instruction. Lets one per-kv-subtile base register serve all d-tile pairs —
// only the padded (un-swizzled) K-LDS layout makes the address affine in d_pair, so
// d_pair folds into the immediate instead of a per-read VALU add. %c emits the bare constant.
template <int kOffset0, int kOffset1>
__device__ __forceinline__ u32x4_t ds_read2_b64_imm(uint32_t lds_base) {
    u32x4_t result;
    asm volatile("ds_read2_b64 %0, %1 offset0:%c2 offset1:%c3"
                 : "=&v"(result) : "v"(lds_base), "i"(kOffset0), "i"(kOffset1));
    return result;
}

// Pinned lgkmcnt wait — anchors the consumer so the compiler can't reorder above.
// kCount = how many outstanding LDS reads may remain in flight past this point.
#define WAIT_LGKM_CASE(N) \
    if constexpr (kCount == N) asm volatile("s_waitcnt lgkmcnt(" #N ")" : "+v"(anchor) :: "memory");
template <int kCount, typename T>
__device__ __forceinline__ void wait_lgkm(T& anchor) {
    static_assert(kCount >= 0 && kCount <= 7, "kCount out of range");
    WAIT_LGKM_CASE(0) else WAIT_LGKM_CASE(1) else WAIT_LGKM_CASE(2) else
    WAIT_LGKM_CASE(3) else WAIT_LGKM_CASE(4) else WAIT_LGKM_CASE(5) else
    WAIT_LGKM_CASE(6) else WAIT_LGKM_CASE(7);
}
#undef WAIT_LGKM_CASE

// QK inner body, compile-time unrolled (recursion → kReadIdx is constant → d_pair folds
// into the ds_read2 immediate). Depth-kDepth ring over k_ring0/1/2: consume read kReadIdx,
// reissue read kReadIdx+kDepth into the same slot. Each read's two halves are d-tiles
// {d_pair, d_pair+4}; both q-groups multiply against the shared K operand. The last read
// drains lgkm(0), every earlier one leaves kDepth-1 still in flight.
// Iteration order: kv_sub = kReadIdx % kKvSubtiles, d_pair = kReadIdx / kKvSubtiles.
template <int kReadIdx, int kDepth, int kNumReads>
__device__ __forceinline__ void qk_step(
    u32x4_t& k_ring0, u32x4_t& k_ring1, u32x4_t& k_ring2,
    const bf16x4_t q_frag[][kHeadDimTiles], fp32x4_t scores[][kKvSubtiles],
    const uint32_t* kv_sub_base)
{
    if constexpr (kReadIdx < kNumReads) {
        constexpr int kv_sub = kReadIdx % kKvSubtiles;  // kv-subtile  → kv_sub_base[], scores[q_group][kv_sub]
        constexpr int d_pair = kReadIdx / kKvSubtiles;  // d-tile pair → ds_read2 immediate, q_frag[q_group][d_pair]
        constexpr int slot   = kReadIdx % kDepth;       // ring slot (k_ring0/1/2)
        u32x4_t& cur = (slot == 0) ? k_ring0 : (slot == 1) ? k_ring1 : k_ring2;
        if constexpr (kReadIdx == kNumReads - 1) wait_lgkm<0>(cur);
        else                                     wait_lgkm<kDepth - 1>(cur);
        // one ds_read2 = 4 dwords = two bf16x4 K-operands: d-tile d_pair and d_pair+4.
        const bf16x4_t k_lo = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[0], cur[1]});
        const bf16x4_t k_hi = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[2], cur[3]});
        #pragma unroll
        for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) {
            if constexpr (d_pair == 0) scores[q_group][kv_sub] = asm_mfma_zero(k_lo, q_frag[q_group][0]);
            else                       asm_mfma(k_lo, q_frag[q_group][d_pair], scores[q_group][kv_sub]);
            // Last mfma BUILTIN form so LLVM tracks the scores write for softmax.
            scores[q_group][kv_sub] = mfma(k_hi, q_frag[q_group][d_pair + 4], scores[q_group][kv_sub]);
        }
        if constexpr (kReadIdx + kDepth < kNumReads) {
            constexpr int kNext = kReadIdx + kDepth;
            constexpr int next_kv_sub = kNext % kKvSubtiles, next_d_pair = kNext / kKvSubtiles;
            cur = ds_read2_b64_imm<next_d_pair * 4, next_d_pair * 4 + 16>(kv_sub_base[next_kv_sub]);
        }
        qk_step<kReadIdx + 1, kDepth, kNumReads>(k_ring0, k_ring1, k_ring2, q_frag, scores, kv_sub_base);
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

// V_t (pre-transposed): (kv-iter, super-chunk, lane). bf16x8 / lane.
constexpr int  kVtChunks     = 16;                         // super-chunks / kv-block
constexpr int  kVtBlockElems = kKvBlockRows * kHeadDim;    // 8192 elems / kv-block
constexpr int  kVtChunkElems = kVtBlockElems / kVtChunks;  // 512  elems / super-chunk
constexpr int  kVtLaneElems  = 8;                          // bf16x8 / lane
constexpr auto kVtLayout = opus::make_layout(
    kShape3, strides<kVtBlockElems, kVtChunkElems, kVtLaneElems>());

// K addressed in DWORDS (2 bf16), row-major (kv_row, dword):
//   HBM source: 64 dw/row.  LDS dest: padded to 66 dw/row, which breaks the
//   32-way qk-read bank conflict (stride 64 ≡ 0 mod 32). Linear, no swizzle.
constexpr auto kKHbmDword = opus::make_layout(kShape2, strides<kKRowDwords, 1>());
constexpr auto kKLdsDword = opus::make_layout(kShape2, strides<kKLdsStrideDwords, 1>());

// ─── Grid + MFMA thread→fragment index maps ─────────────────────────────────
// Same rule as the memory layouts above (offset = Σ coordᵢ·strideᵢ) but these
// yield LOGICAL indices — which q-tile / which kv-row / which fragment element —
// not byte offsets, so callers don't multiply by 2.

// (block_x, wave, q-group) → absolute q-tile index. A workgroup owns
// kQTilesPerWG tiles, a wave owns kQTilesPerWave, a q-group is one tile.
constexpr auto kQTileGrid = opus::make_layout(kShape3, strides<kQTilesPerWG, kQTilesPerWave, 1>());

// (load-wave, local row) → kv-row that load-wave owns. kLoadWaves waves ×
// kKRowsPerLoadWave rows tile the kKvBlockRows-row K block.
constexpr auto kKLoadRow  = opus::make_layout(kShape2, strides<kKRowsPerLoadWave, 1>());

// (load-wave, i) → V_t super-chunk to L1-prefetch (kVtChunks chunks / kLoadWaves).
constexpr auto kVtPrefetch = opus::make_layout(kShape2, strides<kVtChunks / kLoadWaves, 1>());

// MFMA 16x16x16 A/B fragment: lane = (col = lane&15, quad = lane>>4); the 16-wide
// contraction is tiled by 4 quad-lanes, each holding kQuadSpan contiguous elems.
constexpr int  kQuadSpan  = kMfmaK / 4;   // 4 bf16 contraction elems / quad-lane
// (d-tile, quad) → head-dim element, for Q / Out (bf16 elements).
constexpr auto kHdFrag    = opus::make_layout(kShape2, strides<kMfmaK, kQuadSpan>());
// (kv-row, quad) → K-LDS dword (K is dword-addressed: kQuadSpan/2 dwords / quad).
constexpr auto kKLaneDw   = opus::make_layout(kShape2, strides<kKLdsStrideDwords, kQuadSpan / 2>());

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
constexpr int kNumXcds = 8;
__device__ __forceinline__ int chiplet_swizzle(int wg_id, int num_wgs, int chunk_size) {
    const int block = kNumXcds * chunk_size;
    const int limit = (num_wgs / block) * block;
    if (wg_id >= limit) return wg_id;
    const int xcd          = wg_id % kNumXcds;
    const int local_id    = wg_id / kNumXcds;
    const int chunk_idx    = local_id / chunk_size;
    const int pos_in_chunk = local_id % chunk_size;
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

// Cross-lane reduce over {l, l^16, l^32, l^48}: in 16x16x16 C-layout each lane
// outputs at q_row = col, so the 4 quad-lanes sharing col span all 64 kvs of the
// q-row. ds_swizzle 0x401F folds l^16, ds_bpermute folds l^32 (covers l^48 too).
// Explicit lgkm(0) wait avoids a race when the bpermute is the only LGKM in flight.
__device__ __forceinline__ float cross_lane_max(float v) {
    const int lane = wave_lane_id();
    const int peer16  = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
    const float partial = fmaxf(v, __builtin_bit_cast(float, peer16));
    const int peer32  = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, partial));
    wait_lgkmcnt0();
    return fmaxf(partial, __builtin_bit_cast(float, peer32));
}
__device__ __forceinline__ float cross_lane_sum(float v) {
    const int lane = wave_lane_id();
    const int peer16  = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
    const float partial = v + __builtin_bit_cast(float, peer16);
    const int peer32  = __builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __builtin_bit_cast(int, partial));
    wait_lgkmcnt0();
    return partial + __builtin_bit_cast(float, peer32);
}

template <bool kLite>
__global__ __launch_bounds__(kBlockSize, 1)
void attention_forward(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ K,
    const bf16_t* __restrict__ V,    // points to V_t (pre-transposed)
    bf16_t* __restrict__ Out,
    int seq_len, float scale_log2e,
    const int16_t* __restrict__ read_list,   // LiteAttention skip list (this phase's slice)
    int16_t* __restrict__ write_list,        // skip list written for the next timestep
    float thr, int phase                     // skip threshold (log2) + reverse-list phase
) {
    // MFMA 16x16x16 lane split: col = q-col / kv-row in tile, quad = which 4 of 64.
    // Tile dims walked by the loops: kv_sub (kv-subtile), d_pair (d-tile pair), d_tile (hd-tile),
    // q_group (q-group). 8 waves → 2 groups of 4 for warp-spec. See indexing notebook.
    const int  tid           = threadIdx.x;
    const int  wave_id       = tid >> 6;
    const int  lane          = tid & 63;
    const int  col           = lane & 15;   // q-col / kv-row within a 16-wide tile
    const int  quad          = lane >> 4;   // which group of 4 lanes (0..3)
    const bool group_a       = (wave_id < kLoadWaves);
    const int  wave_in_group = group_a ? wave_id : (wave_id - kLoadWaves);

    // Grid: x = #q-blocks (256-row blocks = ceil(seq_len/256)), y = batch·heads.
    // One workgroup (CTA) per (q-block, batch·head).
    const int num_q_blocks = gridDim.x;
    const int num_bh       = gridDim.y;
    const int wg_id_raw = blockIdx.y * num_q_blocks + blockIdx.x;
    const int num_wgs   = num_q_blocks * num_bh;
    // Dense: head-first chiplet swizzle co-locates a (B,H)'s q-blocks on one XCD's
    // L2 (~+1.4% @ moderate S). Lite: that pinning is HARMFUL — uneven per-head
    // sparsity saturates the dense heads' XCDs while sparse heads' XCDs idle
    // (per-head 50% skip measured 0.83x, i.e. SLOWER). Identity mapping spreads each
    // head's q-tiles across all 8 XCDs → per-head 50% skip load-balances to ~1.9x.
    // kLite is a compile-time bool, so this is a zero-cost static branch.
    const int wg_id     = kLite ? wg_id_raw : chiplet_swizzle(wg_id_raw, num_wgs, num_q_blocks);
    const int q_block   = wg_id % num_q_blocks;   // which 256-row q-block
    const int bh_index  = wg_id / num_q_blocks;   // which (batch·head)

    // Q-tile ownership: each wave owns kQTilesPerWave (=2) 16-row q-tiles. For each,
    // q_tile_start_row is the absolute seq-row where that tile begins — a per-tile
    // origin used for addressing; lane `col` later reads/writes row (start + col).
    // Parallel arrays valid/q_tile_start_row/q_frag/o_acc/scores are all indexed by
    // q_group; valid[] masks a tile that runs past the sequence end.
    const int num_q_tiles = seq_len / kQTileRows;
    bool valid[kQTilesPerWave];
    int  q_tile_start_row[kQTilesPerWave];
    #pragma unroll
    for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) {
        const int q_tile = kQTileGrid(q_block, wave_id, q_group);
        valid[q_group] = q_tile < num_q_tiles;
        q_tile_start_row[q_group] = valid[q_group] ? q_tile * kQTileRows : 0;
    }

    const long long bh_offset = (long long)bh_index * seq_len * kHeadDim;
    const auto q_buffer  = make_buffer(Q + bh_offset);
    const auto k_buffer  = make_buffer(K + bh_offset);
    const auto v_buffer  = make_buffer(V + bh_offset);
    const auto out_buffer = make_buffer(Out + bh_offset);

    __shared__ uint32_t lds_k[2][kKvBlockRows * kKLdsStrideDwords];   // 33 KB (2-buf)
    // V LDS staging removed — V_t is consumed directly from L1 via load_u32x4.

    // K[kv_block] HBM→LDS into double-buffer half `k_buf` (group B). Wave w owns rows
    // [w·16, +16); for each, its 64 lanes stream one full K row — lane L copies dword L.
    // Linear padded LDS.
    constexpr uint32_t kKRowBytes   = (uint32_t)(kKLdsStrideDwords * 4);
    constexpr uint32_t kKBlockBytes = (uint32_t)(kKvBlockRows * kHeadDim * 2);
    const int wave_row0 = kKLoadRow(wave_in_group, 0);   // first kv-row this load-wave owns
    auto direct_lds_k = [&](int kv_block, int k_buf) {
        const uint32_t lds_row0  = to_scalar(lds_addr(&lds_k[k_buf][kKLdsDword(wave_row0, 0)]));
        const uint32_t hbm_block = to_scalar((uint32_t)((uint64_t)kv_block * kKBlockBytes));
        #pragma unroll
        for (int c = 0; c < kKRowsPerLoadWave; ++c) {
            const int kv_row = kKLoadRow(wave_in_group, c);
            const uint32_t hbm_byte = (uint32_t)(kKHbmDword(kv_row, lane) * 4);
            buffer_load_b32_to_lds(k_buffer, lds_row0 + c * kKRowBytes, hbm_byte, hbm_block);
        }
    };

    // ─── LiteAttention skip state (dynamic LDS; 0 bytes when !kLite) ─────────────
    // The main loop walks the read list's RLE ranges directly (no block-list
    // expansion). sverdict[wave][step % kVerdictRing] holds each wave's per-block skip
    // vote; the loop's incremental writer ANDs the 8 waves to emit the next-step write
    // list. Only the live window (step and step-1) is ever read, so a fixed ring suffices
    // — the LDS cost is constant in seq_len (vs the old [kNumWaves*num_kv_blocks]).
    const int num_kv_blocks = seq_len / kKvBlockRows;   // total K-blocks (= ktiles)
    const int lstep         = phase ? 1 : -1;           // reverse-list step (skip_list.h)
    extern __shared__ char g_dyn[];
    uint8_t* const sverdict = (uint8_t*)g_dyn;          // [kNumWaves*kVerdictRing] ring, kLite only

    // Skip-list is row-major [bh, q-block, num_k_blocks] in HBM — one RLE row per CTA.
    // lbase = that row's origin (a Layout, like every other index; strides are runtime/
    // seq-len-dependent so it's built per-launch, not constexpr). Within the row the RLE
    // is walked by a running cursor (lbase + idx). to_scalar() keeps cursor reads in SGPRs,
    // not VGPRs across the MFMA loop.
    long long lbase = 0;
    int       list_len = 0;
    auto rl = [&](int idx) -> int { return to_scalar((int)read_list[lbase + idx]); };

    // Prologue: first processed K-block → LDS (group B only); group A idle until iter 1.
    int  first_blk = 0;                       // dense: block 0
    bool any       = (num_kv_blocks > 0);
    if constexpr (kLite) {
        const long long num_k_blocks = num_kv_blocks + 2;
        const auto kSkipRow = opus::make_layout(kShape2, opus::make_tuple(num_q_blocks * num_k_blocks, num_k_blocks));
        lbase     = kSkipRow(bh_index, q_block);          // origin of this CTA's skip-list row
        list_len  = to_scalar((int)read_list[lbase]);     // RLE length
        any       = (list_len > 0);
        first_blk = any ? (rl(list_len) + lstep) : 0;      // first range's start
    }
    if (!group_a && any) direct_lds_k(first_blk, 0);

    // Persistent VGPR Q (no reload across K-blocks). Lane (q_tile_start_row+col) owns one
    // q-row; each load_bf16x4 grabs 4 head-dims of d-tile d (quad picks which 4 of 16).
    bf16x4_t q_frag[kQTilesPerWave][kHeadDimTiles];
    #pragma unroll
    for (int q_group = 0; q_group < kQTilesPerWave; ++q_group)
        #pragma unroll
        for (int d = 0; d < kHeadDimTiles; ++d) {
            const int q_elem = kQOLayout(q_tile_start_row[q_group] + col, kHdFrag(d, quad));
            q_frag[q_group][d] = valid[q_group] ? load_bf16x4(q_buffer, q_elem * 2) : bf16x4_t{0,0,0,0};
        }

    fp32x4_t o_acc[kQTilesPerWave][kHeadDimTiles];
    #pragma unroll
    for (int q_group = 0; q_group < kQTilesPerWave; ++q_group)
        #pragma unroll
        for (int d = 0; d < kHeadDimTiles; ++d) o_acc[q_group][d] = zero_fp32x4();
    float row_max[kQTilesPerWave], row_sum[kQTilesPerWave];
    #pragma unroll
    for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) { row_max[q_group] = -FLT_MAX; row_sum[q_group] = 0.f; }

    // ─── QK: S = K·Qᵀ via mfma(A=K, B=Q). 4 kv-subtiles × 4 d-tile pairs. ───────
    // Linear K-LDS ⇒ read address is affine in d_pair, so one base reg per kv-subtile +
    // d_pair on the ds_read2 immediate (qk_step). base = k_byte_base + this lane's
    // (kv_row=col, dword=2*quad)·4; per-subtile delta steps kv_row by kMfmaTile rows.
    auto qk_base = [&](uint32_t k_byte_base) -> uint32_t {
        return k_byte_base + (uint32_t)(kKLaneDw(col, quad) * 4);  // (kv-row=col, quad→dword)
    };
    constexpr uint32_t kQkSubtileDelta = (uint32_t)(kKLdsDword(kMfmaTile, 0) * 4);  // +one kv-subtile

    // Prefetch first 3 ds_read2 BEFORE pv_phase: their lgkmcnt advances
    // independently of pv's vmcnt load_u32x4. sched_barrier(0) locks placement.
    // Reads 0,1,2 = (kv_sub=0,1,2; d_pair=0) → shared immediate, stepped base.
    auto prefetch_qk = [&](uint32_t k_byte_base, u32x4_t& k_ring0, u32x4_t& k_ring1, u32x4_t& k_ring2) {
        sched_fence();
        const uint32_t lane_base = qk_base(k_byte_base);
        k_ring0 = ds_read2_b64_imm<0, 16>(lane_base);
        k_ring1 = ds_read2_b64_imm<0, 16>(lane_base + kQkSubtileDelta);
        k_ring2 = ds_read2_b64_imm<0, 16>(lane_base + 2 * kQkSubtileDelta);
        sched_fence();
    };

    // Pass k_ring0/k_ring1/k_ring2 by value so qk_phase sees them as locals (compiler may
    // freely reorder the lgkm waits relative to mfma issue). The depth-3 ring
    // and immediate-offset reissue live in the recursive qk_step.
    auto qk_phase = [&](uint32_t k_byte_base, u32x4_t k_ring0, u32x4_t k_ring1, u32x4_t k_ring2,
                        fp32x4_t scores[kQTilesPerWave][kKvSubtiles]) {
        constexpr int kTotalReads = kKvSubtiles * (kHeadDimTiles / 2);  // 16
        constexpr int kDepth      = 3;
        const uint32_t lane_base = qk_base(k_byte_base);
        uint32_t kv_sub_base[kKvSubtiles];
        #pragma unroll
        for (int kv_sub = 0; kv_sub < kKvSubtiles; ++kv_sub) kv_sub_base[kv_sub] = lane_base + (uint32_t)kv_sub * kQkSubtileDelta;
        mfma_prio_hi();
        sched_fence();
        qk_step<0, kDepth, kTotalReads>(k_ring0, k_ring1, k_ring2, q_frag, scores, kv_sub_base);
        sched_fence();
        mfma_prio_lo();
    };

    // ─── PV: O += V·Pᵀ via mfma(A=V_t, B=P). 16 super-chunks/iter, each lane
    // load_u32x4s its pre-packed bf16x8 from V_t (kVtLayout). Depth-4 prefetch ring.
    // chunk → hd-tile d_tile=chunk/2, kv-subtile pair {kv_sub0,kv_sub1}; both q-groups
    // share the V operand. ──────────────────────────────────────────────────────
    auto chunk_byte_off = [&](int kv_iter, int chunk) -> uint32_t {
        return (uint32_t)(kVtLayout(kv_iter, chunk, lane) * 2);   // bf16 elems → bytes
    };

    // verdict_step = the processing step of the block being pv'd (its verdict-ring slot);
    // ignored when !kLite. Soft block-skip: a wave whose verdict for this block is "skip"
    // does no pv at all (its probs were already zeroed in softmax, so the result is
    // unchanged — this just drops the 0-adding MFMAs + V_t loads). The verdict is in
    // LDS from this wave's own softmax (group A: prev iter P2; group B: this iter P1),
    // visible across the same barriers record() relies on.
    auto pv_phase = [&](int kv_iter, const bf16x4_t probs[kQTilesPerWave][kKvSubtiles], int verdict_step) {
        if constexpr (kLite) {
            if (sverdict[wave_id * kVerdictRing + (verdict_step & (kVerdictRing - 1))]) return;
        }
        constexpr int kPvChunks = kVtChunks;  // 16 V_t super-chunks / kv-iter
        constexpr int kPvDepth  = 4;
        mfma_prio_hi();
        sched_fence();
        u32x4_t v_ring[kPvDepth];
        #pragma unroll
        for (int p_init = 0; p_init < kPvDepth; ++p_init) {
            v_ring[p_init] = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, p_init));
        }
        #pragma unroll
        for (int chunk = 0; chunk < kPvChunks; ++chunk) {
            const int d_tile      = chunk >> 1;        // hd-tile 0..7
            const int kv_sub_pair = chunk & 1;         // 0 → {kv_sub0,kv_sub1}={0,1}; 1 → {2,3}
            const int kv_sub0     = kv_sub_pair * 2;
            const int kv_sub1     = kv_sub0 + 1;
            const int slot        = chunk % kPvDepth;   // depth-4 ring slot
            u32x4_t&  cur         = v_ring[slot];
            // load_u32x4 = 4 dwords = two bf16x4 V-operands (kv-subtiles kv_sub0, kv_sub1).
            const bf16x4_t v_lo = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[0], cur[1]});
            const bf16x4_t v_hi = __builtin_bit_cast(bf16x4_t, u32x2_t{cur[2], cur[3]});
            asm_mfma(v_lo, probs[0][kv_sub0], o_acc[0][d_tile]);
            asm_mfma(v_lo, probs[1][kv_sub0], o_acc[1][d_tile]);
            asm_mfma(v_hi, probs[0][kv_sub1], o_acc[0][d_tile]);
            asm_mfma(v_hi, probs[1][kv_sub1], o_acc[1][d_tile]);
            if (chunk + kPvDepth < kPvChunks) {
                cur = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, chunk + kPvDepth));
            }
        }
        sched_fence();
        mfma_prio_lo();
    };

    // L1 prefetcher for V_t[kv_iter_next], fired from Group A's softmax-only P2.
    // vmcnt is per-wave so this is independent of any consumer's pending load_u32x4.
    // Wave-split: each of 4 waves takes 4 unique super-chunks → 16 unique cache
    // line requests hit all L1 sets in parallel. The dest v[140:143] is a discard
    // sink: this is the LAST op before the end-of-iter wait_vmcnt0, so the async
    // write lands and is drained before any reuse (same invariant as the 32x32).
    auto prefetch_v_t_l1 = [&](int kv_iter_next) {
        if (kv_iter_next < 0) return;
        #pragma unroll
        for (int i = 0; i < kVtChunks / kLoadWaves; ++i) {
            const int chunk = kVtPrefetch(wave_in_group, i);
            const uint32_t chunk_off = (uint32_t)(kVtLayout(kv_iter_next, chunk, lane) * 2);
            asm volatile(
                "buffer_load_dwordx4 v[140:143], %0, %1, 0 offen"
                :
                : "v"(chunk_off), "s"(v_buffer)
                : "v140", "v141", "v142", "v143", "memory"
            );
        }
    };

    // ─── Softmax: block-max fold + exp2 + rescale + pack, per q-group ───────────
    auto softmax = [&](fp32x4_t scores[kQTilesPerWave][kKvSubtiles], bf16x4_t probs[kQTilesPerWave][kKvSubtiles], int step) {
        // ── LiteAttention verdict pre-pass (kLite) — decide this wave's skip from the
        // per-q-group block maxes BEFORE the fold, so a skipping wave can drop the WHOLE
        // contribution (probs→0, no row_sum/o_acc update) = "block absent". (Dropping
        // only pv while the fold ran would bias o/row_sum — verdict is a MAX test, not a
        // contribution test.) The verdict ALWAYS feeds the next-step write list and ALSO
        // drives this step's per-wave fold+pv skip. Keep iff some q-row's
        // block max is within `thr` (log2) of its running max — FA3 cond2/!__any_sync;
        // the per-lane partial max + __any reduces across the 4 quad-lanes of each col.
        if constexpr (kLite) {
            bool any_keep = false;
            #pragma unroll
            for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) {
                float bm = -FLT_MAX;
                #pragma unroll
                for (int kv_sub = 0; kv_sub < kKvSubtiles; ++kv_sub)
                    bm = fmaxf(bm,
                        fmaxf(fmaxf(scores[q_group][kv_sub][0], scores[q_group][kv_sub][1]),
                              fmaxf(scores[q_group][kv_sub][2], scores[q_group][kv_sub][3])));
                any_keep = any_keep || (valid[q_group] && ((bm * scale_log2e - row_max[q_group]) > thr));
            }
            bool skip_wave = (__any(any_keep) == 0);
            if (step == 0) skip_wave = false;   // FA3 Is_first: never skip the first processed block
            if (lane == 0) sverdict[wave_id * kVerdictRing + (step & (kVerdictRing - 1))] = skip_wave ? (uint8_t)1 : (uint8_t)0;
            if (skip_wave) {
                // Block absent for this wave: zero probs so the (skipped) pv adds
                // nothing; row_max/row_sum/o_acc untouched — the skip predicate
                // (bm·scale − row_max ≤ thr) guarantees no new running max.
                #pragma unroll
                for (int q_group = 0; q_group < kQTilesPerWave; ++q_group)
                    #pragma unroll
                    for (int kv_sub = 0; kv_sub < kKvSubtiles; ++kv_sub)
                        probs[q_group][kv_sub] = bf16x4_t{0, 0, 0, 0};
                return;
            }
        }
        // ── Fold (dense + lite-keep) — byte-identical to the champion. ──
        #pragma unroll
        for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) {
            // block-max fold happens IN softmax (P2), where the other group is in
            // pv+qk (MFMA-heavy) — the v_max3 chain hides under their MFMA.
            float block_max = -FLT_MAX;
            #pragma unroll
            for (int kv_sub = 0; kv_sub < kKvSubtiles; ++kv_sub)
                block_max = fmaxf(block_max,
                    fmaxf(fmaxf(scores[q_group][kv_sub][0], scores[q_group][kv_sub][1]),
                          fmaxf(scores[q_group][kv_sub][2], scores[q_group][kv_sub][3])));
            constexpr float kRescaleThr = 8.0f;
            const float new_max_cand = block_max * scale_log2e;
            const bool  first_block = row_max[q_group] == -FLT_MAX;
            const bool  all_ok = __all((int)((new_max_cand - row_max[q_group]) < kRescaleThr)) != 0
                                && !first_block;
            float neg_max;
            float rescale = 1.f;
            if (__builtin_expect(all_ok, 1)) {
                neg_max = -row_max[q_group];
            } else {
                const float new_max = fmaxf(row_max[q_group], cross_lane_max(block_max) * scale_log2e);
                rescale = first_block ? 0.f : fast_exp2(row_max[q_group] - new_max);
                if (!first_block) {
                    // Force in-place v_mul_f32_e32: per-subscript `o_acc[q_group][t][i] *=`
                    // compiles to v_pk_mul_f32 (partial-subreg write → mfma C-read
                    // hazard). Inline `+v(o_acc[q_group][t][i])` keeps it scalar.
                    #pragma unroll
                    for (int t = 0; t < kHeadDimTiles; ++t)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                            asm volatile("v_mul_f32_e32 %0, %1, %0"
                                         : "+v"(o_acc[q_group][t][i]) : "v"(rescale));
                }
                row_max[q_group] = new_max;
                neg_max = -new_max;
            }
            float block_sum = 0.f;
            #pragma unroll
            for (int kv_sub = 0; kv_sub < kKvSubtiles; ++kv_sub) {
                const float p0 = fast_exp2(fmaf(scores[q_group][kv_sub][0], scale_log2e, neg_max));
                const float p1 = fast_exp2(fmaf(scores[q_group][kv_sub][1], scale_log2e, neg_max));
                const float p2 = fast_exp2(fmaf(scores[q_group][kv_sub][2], scale_log2e, neg_max));
                const float p3 = fast_exp2(fmaf(scores[q_group][kv_sub][3], scale_log2e, neg_max));
                // NaN-safe pack: a masked row gives all-zero p (safe); a zero
                // row_sum is handled in the epilogue. MaybeNaN scrubs any drift.
                probs[q_group][kv_sub] = pack4_bf16</*MaybeNaN=*/true>(p0, p1, p2, p3);
                block_sum = vadd_f32(block_sum, (p0 + p1) + (p2 + p3));
            }
            // Cross-lane sum of block_sum is deferred to the output epilogue (math-equiv
            // since rescale is a global per-row factor).
            row_sum[q_group] = vadd_f32(vmul_f32(row_sum[q_group], rescale), block_sum);
        }
    };

    // ─── Main loop ─────────────────────────────────────────────────────────
    //   Group A  P1: pv(V_t[N-1]) + qk(K[N])      P2: softmax + V_t prefetch
    //   Group B  P1: K[N+1] HBM→LDS + softmax     P2: pv(V_t[N-1]) + qk(K[N])
    // V_t prefetch lives in Group A's softmax-only P2 — bursts in the pv loop
    // regress perf by stalling consumer issue on vmcnt pressure.
    wait_vmcnt0();
    lds_barrier();
    fp32x4_t scores[kQTilesPerWave][kKvSubtiles];
    bf16x4_t probs[kQTilesPerWave][kKvSubtiles];
    // ─── Unified pipeline over processed K-blocks ──────────────────────────────
    // Dense and lite share ONE warp-spec body; only how each step's blocks are produced
    // differs. `step` = processing-step counter 0,1,2,… (drives K double-buffer parity +
    // verdict-ring slot); `kv_block`/`prev_block` are the blocks being qk'd / pv'd;
    // `next_block` is what group B streams into LDS. Dense: kv_block==prev_block+1==step,
    // walking 0..num_kv_blocks-1 — every lite-only construct below is `if constexpr`'d
    // out, leaving the champion's loop. Lite: a cursor walks the read-list's RLE ranges
    // (kv_block/prev_block decoupled from step) and tid 0 records the next-step write list
    // under group B's MFMA. has_next = "another block follows" (dense: step+1<N; lite:
    // range continues OR another range remains).

    // Lite cursor + incremental write-list state (unused → elided when !kLite).
    auto load_range = [&](int& range_block, int& range_end, int& rle_idx) {  // open next RLE range
        range_block = rl(rle_idx) + lstep;
        range_end   = rl(rle_idx - 1) + lstep;
        rle_idx    -= 2;
    };
    int  rle_idx = list_len, range_block = 0, range_end = 0;
    int  write_idx   = 1;       // write-list output cursor
    bool skip_state  = true;    // running skip/compute state (for transition detection)
    bool prev_ended_range = false;   // did prev_block end its range?
    // record `block` at verdict-ring slot `verdict_step`: AND its 8 wave-votes, emit a
    // skip↔compute transition when the state flips, and a range-end marker if is_range_end.
    auto record = [&](int block, int verdict_step, bool is_range_end) {
        int sk = 1;
        #pragma unroll
        for (int w = 0; w < kNumWaves; ++w) sk &= (int)sverdict[w * kVerdictRing + (verdict_step & (kVerdictRing - 1))];
        const bool skip = (sk != 0);
        if (skip != skip_state) { write_list[lbase + write_idx++] = (int16_t)block; skip_state = skip; }
        if (is_range_end) { skip_state = true; if (!skip) write_list[lbase + write_idx++] = (int16_t)(block + lstep); }
    };

    int  prev_block = -1, step = 0;
    bool have = kLite ? (list_len > 0) : (num_kv_blocks > 0);
    if constexpr (kLite) { if (have) load_range(range_block, range_end, rle_idx); }

    while (have) {
        // Per-step block descriptor: dense counts straight up, lite reads the cursor.
        int  kv_block, next_block;  bool has_next, range_continues = false;
        if constexpr (kLite) {
            kv_block             = range_block;
            const int next_in_range = kv_block + lstep;
            range_continues      = (lstep > 0) ? (next_in_range < range_end) : (next_in_range > range_end);
            has_next             = range_continues || (rle_idx >= 1);
            next_block           = range_continues ? next_in_range : (has_next ? (rl(rle_idx) + lstep) : -1);
        } else {
            kv_block = step;  next_block = step + 1;  has_next = (step + 1 < num_kv_blocks);
        }
        const int  k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
        const bool first = (step == 0);
        const uint32_t k_byte_base = lds_addr(&lds_k[k_buf_cur][0]);
        // Declared in-loop = single-iter lifetime (no cross-iter register pressure).
        u32x4_t k_ring0, k_ring1, k_ring2;
        if (group_a) {  // P1
            // qk LDS reads (lgkmcnt) fired BEFORE pv_phase so they hide under pv's
            // mfmas; consumed in qk_phase. prefetch_qk's sched_barrier locks placement.
            prefetch_qk(k_byte_base, k_ring0, k_ring1, k_ring2);
            if (!first) pv_phase(prev_block, probs, step - 1);
            qk_phase(k_byte_base, k_ring0, k_ring1, k_ring2, scores);
        } else {
            if (!first) softmax(scores, probs, step - 1);
            if (has_next) direct_lds_k(next_block, k_buf_next);
        }
        __builtin_amdgcn_s_barrier();
        if (group_a) {  // P2
            softmax(scores, probs, step);
            if constexpr (kLite) {   // record prev block's completed vote under group B's MFMA
                if (tid == 0 && !first) record(prev_block, step - 1, prev_ended_range);
            }
            if (has_next) prefetch_v_t_l1(kv_block);
        } else {
            prefetch_qk(k_byte_base, k_ring0, k_ring1, k_ring2);
            if (!first) pv_phase(prev_block, probs, step - 1);
            qk_phase(k_byte_base, k_ring0, k_ring1, k_ring2, scores);
        }
        // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is
        // VMCNT-tracked, not LGKMCNT) so the next iter's qk_phase can't read stale K;
        // lds_barrier covers the LGKMCNT side.
        wait_vmcnt0();
        lds_barrier();
        prev_block = kv_block; ++step;
        if constexpr (kLite) {
            prev_ended_range = !range_continues;
            if (range_continues)   range_block = kv_block + lstep;
            else if (rle_idx >= 1) load_range(range_block, range_end, rle_idx);
            else                   have = false;
        } else {
            have = (step < num_kv_blocks);
        }
    }
    // Epilogue: last block's softmax + pv, then (lite) flush its always-range-end vote.
    if (!group_a && step > 0) softmax(scores, probs, step - 1);
    if (step > 0) pv_phase(prev_block, probs, step - 1);
    if constexpr (kLite) {
        lds_barrier();   // group B's last-block vote → visible to tid 0
        if (tid == 0 && step > 0) {
            record(prev_block, step - 1, /*is_range_end=*/true);
            write_list[lbase] = (int16_t)(write_idx - 1);   // finalize count
        }
    }

    // ─── Output ────────────────────────────────────────────────────────────
    // row_sum is summed across the 4 quad-lanes once here (math-equiv since
    // rescale is global to the row). inv_sum=0 when row_sum==0 prevents 0×inf=NaN.
    #pragma unroll
    for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) row_sum[q_group] = cross_lane_sum(row_sum[q_group]);
    #pragma unroll
    for (int q_group = 0; q_group < kQTilesPerWave; ++q_group) {
        if (!valid[q_group]) continue;
        const float inv_sum = (row_sum[q_group] == 0.0f) ? 0.0f : fast_rcp(row_sum[q_group]);
        // No PV mfma follows so a packed v_pk_mul_f32 is safe here. MaybeNaN
        // pack4 scrubs any residual NaN/Inf in o_acc from pathological inputs.
        #pragma unroll
        for (int t = 0; t < kHeadDimTiles; ++t) {
            o_acc[q_group][t] = o_acc[q_group][t] * inv_sum;
            const bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(
                o_acc[q_group][t][0], o_acc[q_group][t][1], o_acc[q_group][t][2], o_acc[q_group][t][3]);
            store_bf16x4(out_buffer, kQOLayout(q_tile_start_row[q_group] + col, kHdFrag(t, quad)) * 2, out);
        }
    }
}

// ─── V transpose for 16x16x16 mfma ─────────────────────────────────────────────
// Pre-packs V[seq,128] into V_t so pv_phase reads each lane's MFMA A-fragment with
// one contiguous load_u32x4. Phase 1: V block HBM→LDS (coalesced). Phase 2: re-read in
// fragment order → V_t (kVtLayout, shared with pv_phase). super-chunk = d_tile*2 +
// kv_sub_pair; lane l's 16 bytes are two A-operands (kv-subtiles kv_sub0 then kv_sub1):
//   bytes [l*16 .. l*16+7]   = V[k=kv_sub0*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]
//   bytes [l*16+8 .. l*16+15] = V[k=kv_sub1*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]
// V_t layout and the fragment map are drawn in the indexing notebook.
constexpr int kXposeThreadsPerWG = 256;
constexpr auto kVStageWr = opus::make_layout(kShape2, strides<kHeadDim / 2, 4>()); // (row, octet)→dword
constexpr auto kVtChunkLane = opus::make_layout(kShape2, strides<kVtChunkElems, kVtLaneElems>()); // (super-chunk, lane)
__global__ __launch_bounds__(kXposeThreadsPerWG, 4)
void v_transpose_kernel_16x16(
    const bf16_t* __restrict__ V,
    bf16_t*       __restrict__ V_t,
    int           seq_len_per_head
) {
    const int kv_iter  = blockIdx.x;                 // which 64-row kv-block (over all B·H·S)
    const int tid = threadIdx.x;
    const long long block_base = (long long)kv_iter * kVtBlockElems;  // bf16 elems before this block

    __shared__ uint32_t stage_dw[kKvBlockRows * (kHeadDim / 2)];   // 64 rows × 64 dw = 16 KB
    const auto v_buffer = make_buffer(V);

    // ── Phase 1: V[block] HBM → LDS, coalesced. 1024 element-copies cover 64 rows ×
    //    16 head-dim octets; each copies one bf16x8 (= 4 dwords). ───────────
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int elem_idx = k * kXposeThreadsPerWG + tid;
        const int row      = elem_idx >> 4;    // 0..63 kv-row in this block
        const int octet    = elem_idx & 15;    // 0..15 head-dim octet (×8 bf16)
        u32x4_t v_oct = __builtin_amdgcn_raw_buffer_load_b128(
            v_buffer, (uint32_t)((block_base + kQOLayout(row, octet * 8)) * 2), 0, 0);
        const int dst_dw = kVStageWr(row, octet);            // dword offset in stage_dw
        stage_dw[dst_dw + 0] = v_oct[0]; stage_dw[dst_dw + 1] = v_oct[1];
        stage_dw[dst_dw + 2] = v_oct[2]; stage_dw[dst_dw + 3] = v_oct[3];
    }

    __syncthreads();

    // staged tile is bf16, row-major (kv_row, head_dim) — same shape as Q (kQOLayout).
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
            packed[i]     = stage[kQOLayout(kv_sub0 * 16 + kv_row0 + i, hd)];
            packed[4 + i] = stage[kQOLayout(kv_sub1 * 16 + kv_row0 + i, hd)];
        }
        u32x4_t packed_v;
        __builtin_memcpy(&packed_v, packed, sizeof(packed));
        __builtin_amdgcn_raw_buffer_store_b128(
            packed_v, vt_buffer, (uint32_t)((block_base + kVtChunkLane(chunk, frag_lane)) * 2), 0, 0);
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
    if (seq_len <= 0 || (seq_len % kKvBlockRows))  return 2;
    if (head_dim != kHeadDim)                 return 3;

    const int   bh   = batch * heads;
    const int   qt   = seq_len / kQTileRows;
    const dim3  grid((qt + kQTilesPerWG - 1) / kQTilesPerWG, bh);
    const float sl2e = kLog2e / sqrtf((float)kHeadDim);

    hipLaunchKernelGGL(
        attention_forward<false>, grid, dim3(kBlockSize), 0, (hipStream_t)stream,
        (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
        (bf16_t*)out, seq_len, sl2e,
        (const int16_t*)nullptr, (int16_t*)nullptr, 0.f, 0);
    return (int)hipGetLastError();
}

// LiteAttention forward: read_list / write_list are int16 [B,H,qtiles,nblocks+2]
// slices (read = skip_list[phase], write = skip_list[1-phase]); kBlockM=256,
// kBlockN=64. threshold is in log2 units (~ -10). phase selects reverse-list step.
// V must already be pre-transposed into V_t (call launch_v_transpose first).
int launch_attention_forward_lite(
    const void* q, const void* k, const void* v, void* out,
    int batch, int heads, int seq_len, int head_dim,
    const void* read_list, void* write_list, float threshold, int phase, void* stream
) {
    if (batch <= 0 || heads <= 0)                  return 1;
    if (seq_len <= 0 || (seq_len % kKvBlockRows))  return 2;
    if (head_dim != kHeadDim)                      return 3;
    if (!read_list || !write_list)                 return 4;

    const int   bh   = batch * heads;
    const int   qt   = seq_len / kQTileRows;
    const dim3  grid((qt + kQTilesPerWG - 1) / kQTilesPerWG, bh);
    const float sl2e = kLog2e / sqrtf((float)kHeadDim);
    // dynamic LDS: sverdict[kNumWaves*kVerdictRing] ring — constant (live window 2),
    // independent of seq_len, so the lite path is not LDS-capped by sequence length.
    const size_t dyn = (size_t)kNumWaves * kVerdictRing;

    hipLaunchKernelGGL(
        attention_forward<true>, grid, dim3(kBlockSize), dyn, (hipStream_t)stream,
        (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
        (bf16_t*)out, seq_len, sl2e,
        (const int16_t*)read_list, (int16_t*)write_list, threshold, phase);
    return (int)hipGetLastError();
}

// V transpose: V[seq_len, 128] (bf16) → V_t in pv_phase's consumed layout.
// Caller allocates V_t with same byte size as V. n_rows = batch*heads*seq_len.
int launch_v_transpose(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, void* stream
) {
    if (n_rows <= 0 || (n_rows % kKvBlockRows)) return 2;
    const int iters = n_rows / kKvBlockRows;
    hipLaunchKernelGGL(
        v_transpose_kernel_16x16,
        dim3(iters), dim3(kXposeThreadsPerWG), 0, (hipStream_t)stream,
        (const bf16_t*)v, (bf16_t*)v_t, seq_len_per_head);
    return (int)hipGetLastError();
}
}
