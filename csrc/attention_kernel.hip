// Forward attention for AMD CDNA3 (MI300X / gfx942), bf16, FA2-style online softmax.
// 16x16x16 bf16 MFMA. 8-wave warp-spec CTA, two groups of 4 waves; per K-tile each
// iter runs as two phases:
//
//                    Group A                      Group B
//   ───────  ────────────────────────  ────────────────────────────
//   Phase 1  pv[N-1] + qk[N]           K[N+1] HBM→LDS + softmax[N-1]
//   Phase 2  softmax[N] + V_t prefetch  pv[N-1] + qk[N]
//
// One group keeps the MFMA engine active while the other does VALU+memory. L1V variant: V is
// pre-transposed offline (v_transpose_kernel_16x16) into V_t for direct load_u32x4 in pv_phase, no LDS staging.
//
// MFMA layout: lane col=lane&15, quad=lane>>4 (4 quads/fp32x4 C-frag); wave owns QTilesPerWave=3
// q-tiles of 16 rows; qk reads K in d-tile pairs {d_pair,d_pair+4}; pv consumes V_t super-chunks.
//
// One TU, top to bottom. Round modes built by attention_{rtna,rtne,rtz}.hip shims (#define launcher, #include this).

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "opus.hpp"

namespace {

// ===== BUILDING BLOCKS: types, constants, intrinsics, layouts, tail scratch

using bf16_t = __hip_bfloat16;
using bf16x4_t = short __attribute__((ext_vector_type(4)));  // 4 bf16 = 8 bytes
// bf16 rounding selected at compile time via -DBF16_ROUND.
#ifndef BF16_ROUND
#define BF16_ROUND RTZ
#endif
enum class BF16Round { RTNE, RTZ, RTNA };
constexpr BF16Round RoundMode = BF16Round::BF16_ROUND;
using fp32x4_t = float __attribute__((ext_vector_type(4)));  // 16x16x16 C-fragment
using u32x2_t = uint32_t __attribute__((ext_vector_type(2)));
using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));

constexpr int WaveSize = 64;
constexpr int NumWaves = 8;
constexpr int LoadWaves = 4;
constexpr int BlockSize = WaveSize * NumWaves;
constexpr int KvBlockRows = 64;                     // kv rows processed per main-loop iter
constexpr int MfmaTile = 16;                        // 16x16x16 mfma tile (M=N)
constexpr int KvSubtiles = KvBlockRows / MfmaTile;  // 4 kv-subtiles / block
constexpr int MfmaK = 16;                           // mfma K (contraction)
constexpr int HeadDim = 128;
constexpr int HeadDimTiles = HeadDim / MfmaK;              // 8 head-dim tiles
constexpr int QTileRows = 16;                              // rows per q-tile (one mfma)
constexpr int QTilesPerWave = 3;                           // → 48 q-rows / wave (3 q-tiles)
constexpr int QTilesPerWG = NumWaves * QTilesPerWave;      // 24 q-tiles = 384 rows / workgroup
constexpr int KRowDwords = HeadDim / 2;                    // 64 = real K row width (dwords)
constexpr int KLdsStrideDwords = KRowDwords;               // 64 = LDS row stride (XOR-swizzled, no pad)
constexpr int KRowsPerLoadWave = KvBlockRows / LoadWaves;  // 16
constexpr float Log2e = 1.4426950408889634f;
constexpr uint32_t BufferConfig = 0x00020000u;  // SRD word3: DATA_FORMAT=32 (raw 32-bit elems)

// Rescale gate: rescale o_acc only when block raises running max by >RescaleThreshold (log2). DEFAULT 0 =
// exact max → rmse ~6e-5, perf-neutral (rescale hides in P2). Override -DRESCALE_THRESHOLD=8.0f if softmax critical.
#ifndef RESCALE_THRESHOLD
#define RESCALE_THRESHOLD 0.0f
#endif
constexpr float RescaleThreshold = RESCALE_THRESHOLD;

// ─── MFMA + scalar math ──────────────────────────────────────────────────────
__device__ __forceinline__ fp32x4_t zero_fp32x4() {
  return fp32x4_t{0, 0, 0, 0};
}

__device__ __forceinline__ float fast_exp2(float x) {
  return __builtin_amdgcn_exp2f(x);
}
__device__ __forceinline__ float fast_rcp(float x) {
  return __builtin_amdgcn_rcpf(x);
}

// Inline-asm mfma: forces in/out same VGPR (no v_mov copies) and keeps issue in source order.
__device__ __forceinline__ void asm_mfma(bf16x4_t a, bf16x4_t b, fp32x4_t& c) {
  asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
}

// Literal-0 C operand skips the v_mov zero-init; "=&v" prevents aliasing dst onto A VGPRs.
__device__ __forceinline__ fp32x4_t asm_mfma_zero(bf16x4_t a, bf16x4_t b) {
  fp32x4_t r;
  asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, 0" : "=&v"(r) : "v"(a), "v"(b));
  return r;
}

// Scalar v_add/v_mul: avoids v_pk_*_f32 sharing writeback ports with MFMA on the same wave-pair.
__device__ __forceinline__ float vadd_f32(float a, float b) {
  float r;
  asm volatile("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
__device__ __forceinline__ float vfma_f32(float a, float b, float c) {  // a*b + c
  float r;
  asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}
__device__ __forceinline__ bool wave_all(bool p) {
  return __all((int)p) != 0;
}

// ─── bf16 rounding + pack ────────────────────────────────────────────────────
// bf16 packing. MaybeNaN=true guards 0×inf NaN; guard left to compiler (pinned asm spills ~50 VGPR past 256 budget).
template <bool MaybeNaN>
__device__ __forceinline__ unsigned bf16_round_one(float f) {
  if constexpr (RoundMode == BF16Round::RTNE) {
    if constexpr (MaybeNaN) {
      uint32_t u_bits = __builtin_bit_cast(uint32_t, f);
      uint32_t tmp;
      u32x2_t nan_mask;
      asm volatile(
          "v_cmp_u_f32 %0, %2, %2\n"
          "v_bfe_u32 %1, %2, 16, 1\n"
          "v_add3_u32 %1, %2, %1, %3\n"
          "v_cndmask_b32 %2, %1, %4, %0"
          : "=s"(nan_mask), "=v"(tmp), "+v"(u_bits)
          : "v"(0x7FFFu), "v"(0x7FFF0000u));
      return u_bits;
    } else {
      const unsigned u = __builtin_bit_cast(unsigned, f);
      return u + 0x7FFFu + ((u >> 16) & 1u);
    }
  } else if constexpr (RoundMode == BF16Round::RTNA) {
    unsigned biased = __builtin_bit_cast(unsigned, f) + 0x8000u;  // ties-away rounding bias
    if constexpr (MaybeNaN) {
      // NaN → canonical 0x7FFF0000. MUST be inline asm: -ffast-math folds __builtin_isunordered to false,
      // disabling a C guard → RTNA leaks biased NaN 0xFFC0 instead of 0x7FFF.
      u32x2_t nan_mask;
      asm volatile(
          "v_cmp_u_f32 %0, %2, %2\n"
          "v_cndmask_b32 %1, %1, %3, %0"
          : "=s"(nan_mask), "+v"(biased)
          : "v"(f), "v"(0x7FFF0000u));
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
  // perm 0x07060302 keeps bytes [3,2] (upper 16b = bf16) of each source, packs 2 bf16/dword: w0=[bf16(b)|bf16(a)].
  const unsigned w0 = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
  const unsigned w1 = __builtin_amdgcn_perm(ud, uc, 0x07060302u);
  bf16x4_t o;
  __builtin_memcpy(&o, &w0, 4);
  __builtin_memcpy((char*)&o + 4, &w1, 4);
  return o;
}

// ─── Buffer / LDS load-store ─────────────────────────────────────────────────
// Wrap a global ptr in a 128-bit SRD ("V#"): stride=0 (unstructured), num_records=0xFFFFFFFF (unbounded),
// config=BufferConfig. const_cast: builtin takes non-const base, but a load resource is never written.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer(const void* p, uint32_t num_records = 0xFFFFFFFFu) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, num_records, BufferConfig);
}
// Raw buffer load/store, named by the value moved (bf16x4 = 8 B, u32x4 = 16 B).
__device__ __forceinline__ bf16x4_t load_bf16x4(__amdgpu_buffer_rsrc_t buffer, int byte_off) {
  u32x2_t raw = __builtin_amdgcn_raw_buffer_load_b64(buffer, byte_off, 0, 0);
  bf16x4_t out;
  __builtin_memcpy(&out, &raw, sizeof(out));
  return out;
}
__device__ __forceinline__ u32x4_t load_u32x4(__amdgpu_buffer_rsrc_t buffer, int byte_off) {
  return __builtin_amdgcn_raw_buffer_load_b128(buffer, byte_off, 0, 0);
}
__device__ __forceinline__ void store_bf16x4(__amdgpu_buffer_rsrc_t buffer, int byte_off, bf16x4_t v) {
  u32x2_t bits;
  __builtin_memcpy(&bits, &v, 8);
  __builtin_amdgcn_raw_buffer_store_b64(bits, buffer, byte_off, 0, 0);
}
// Split a u32x4 (one b128 K/V load = d-tile pair) into 2 bf16x4 MFMA operands: lo2=dwords 0,1; hi2=dwords 2,3.
__device__ __forceinline__ bf16x4_t lo2(u32x4_t v) {
  return __builtin_bit_cast(bf16x4_t, u32x2_t{v[0], v[1]});
}
__device__ __forceinline__ bf16x4_t hi2(u32x4_t v) {
  return __builtin_bit_cast(bf16x4_t, u32x2_t{v[2], v[3]});
}

// Low 32 bits of an LDS pointer = its byte offset within the LDS segment.
template <typename T>
__device__ __forceinline__ uint32_t lds_addr(const T* p) {
  return (uint32_t)(uintptr_t)p;
}

// Broadcast a wave-uniform value into an SGPR via readfirstlane → scalar addressing, not a per-lane VGPR.
__device__ __forceinline__ uint32_t to_scalar(uint32_t v) {
  return __builtin_amdgcn_readfirstlane(v);
}

// Immediate-offset ds_read2_b64: Offset0/Offset1 (8-B units) baked in; <0,16> fetches a d-tile pair in one op
// (offset1=16=128B=HeadDimTiles/2 ahead). d_pair rides the swizzled base, not the immediate; %c emits the bare const.
template <int Offset0, int Offset1>
__device__ __forceinline__ u32x4_t ds_read2_b64_imm(uint32_t lds_base) {
  u32x4_t result;
  asm volatile("ds_read2_b64 %0, %1 offset0:%c2 offset1:%c3"
               : "=&v"(result)
               : "v"(lds_base), "i"(Offset0), "i"(Offset1));
  return result;
}

// ─── Wait-counts, barriers, scheduling ───────────────────────────────────────
// Pinned lgkmcnt wait — anchors the consumer so the compiler can't reorder above it. Count = LDS reads left in flight.
#define WAIT_LGKM_CASE(N) \
  if constexpr (Count == N) asm volatile("s_waitcnt lgkmcnt(" #N ")" : "+v"(anchor)::"memory");
template <int Count, typename T>
__device__ __forceinline__ void wait_lgkm(T& anchor) {
  static_assert(Count >= 0 && Count <= 7, "Count out of range");
  WAIT_LGKM_CASE(0)
  else WAIT_LGKM_CASE(1) else WAIT_LGKM_CASE(2) else WAIT_LGKM_CASE(3) else WAIT_LGKM_CASE(4) else WAIT_LGKM_CASE(
      5) else WAIT_LGKM_CASE(6) else WAIT_LGKM_CASE(7);
}
#undef WAIT_LGKM_CASE

__device__ __forceinline__ void wait_vmcnt0() {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}
__device__ __forceinline__ void wait_lgkmcnt0() {
  __builtin_amdgcn_s_waitcnt(0x007F);
}
__device__ __forceinline__ void lds_barrier() {
  wait_lgkmcnt0();  // lgkmcnt(0)
  __builtin_amdgcn_s_barrier();
}
// Issue-priority: the MFMA-heavy group raises prio to win issue arbitration over the memory/VALU group, then lowers.
// sched_fence = compiler scheduling barrier (pins order so hand-placed prefetches don't sink past it).
__device__ __forceinline__ void mfma_prio_hi() {
  asm volatile("s_setprio 3" ::: "memory");
}
__device__ __forceinline__ void mfma_prio_lo() {
  asm volatile("s_setprio 0" ::: "memory");
}
__device__ __forceinline__ void sched_fence() {
  __builtin_amdgcn_sched_barrier(0);
}

// ─── HBM→LDS direct load ─────────────────────────────────────────────────────
// HBM→LDS direct dword load. Inline-asm `s_mov_b32 m0`: Clang's cselect around the builtin can sub m0=-1 when
// lds_dest is small, silently dropping writes.
__device__ __forceinline__ void
buffer_load_b32_to_lds(__amdgpu_buffer_rsrc_t buffer, uint32_t lds_dest, uint32_t voff, uint32_t soff) {
  asm volatile(
      "s_mov_b32 m0, %0\n\t"
      "buffer_load_dword %1, %2, %3 offen lds"
      :
      : "s"(lds_dest), "v"(voff), "s"(buffer), "s"(soff)
      : "memory");
}

// ─── Cross-lane reductions ───────────────────────────────────────────────────
// Cross-lane reduce over {l,l^16,l^32,l^48}: 4 quad-lanes sharing col span all 64 kvs. ds_swizzle 0x401F folds
// l^16; ds_bpermute folds l^32 (covers l^48). bperm_xor32: source-lane ((lane^32)<<2) computed INSIDE the asm from
// mbcnt each call (pure/remat) so no long-lived index VGPR SPILLS at the softmax peak (256-VGPR); lgkmcnt(0) baked
// in after bpermute. In asm: <<2 = lane→byte (4B/lane); XOR 0x80 = the "^32" peer (32 lanes·4B).
__device__ __forceinline__ int bperm_xor32(int data) {
  int idx, r;
  asm volatile(
      "v_mbcnt_lo_u32_b32 %0, -1, 0\n"
      "v_mbcnt_hi_u32_b32 %0, -1, %0\n"
      "v_lshlrev_b32_e32 %0, 2, %0\n"
      "v_xor_b32_e32 %0, 0x80, %0\n"
      "ds_bpermute_b32 %1, %0, %2\n"
      "s_waitcnt lgkmcnt(0)"
      : "=&v"(idx), "=&v"(r)
      : "v"(data)
      : "memory");
  return r;
}
__device__ __forceinline__ float cross_lane_max(float v) {
  const int peer16 = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
  const float partial = fmaxf(v, __builtin_bit_cast(float, peer16));
  const int peer32 = bperm_xor32(__builtin_bit_cast(int, partial));
  return fmaxf(partial, __builtin_bit_cast(float, peer32));
}
__device__ __forceinline__ float cross_lane_sum(float v) {
  const int peer16 = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F);
  const float partial = v + __builtin_bit_cast(float, peer16);
  const int peer32 = bperm_xor32(__builtin_bit_cast(int, partial));
  return partial + __builtin_bit_cast(float, peer32);
}

// ─── opus layouts + index maps ───────────────────────────────────────────────
template <int... S>
constexpr auto strides() {
  return opus::make_tuple(opus::number<S>{}...);
}
constexpr auto Shape2 = opus::make_tuple(opus::number<1>{}, opus::number<1>{});
constexpr auto Shape3 = opus::make_tuple(opus::number<1>{}, opus::number<1>{}, opus::number<1>{});

// Q / Out: row-major (seq, head_dim).
constexpr auto QOLayout = opus::make_layout(Shape2, strides<HeadDim, 1>());

// V_t (pre-transposed): (kv-iter, super-chunk, lane). bf16x8 / lane.
constexpr int VtChunks = 16;                           // super-chunks / kv-block
constexpr int VtBlockElems = KvBlockRows * HeadDim;    // 8192 elems / kv-block
constexpr int VtChunkElems = VtBlockElems / VtChunks;  // 512  elems / super-chunk
constexpr int VtLaneElems = 8;                         // bf16x8 / lane
constexpr auto VtLayout = opus::make_layout(Shape3, strides<VtBlockElems, VtChunkElems, VtLaneElems>());

// K in DWORDS (2 bf16), row-major (kv_row,dword), 64 dw/row. LDS dest: same stride 64 but XOR-swizzled
// (direct_lds_k/qk_swz_base) to break the 32-way qk-read bank conflict (stride 64≡0 mod 32) without padding (pad is
// slower).
constexpr auto KLdsDword = opus::make_layout(Shape2, strides<KLdsStrideDwords, 1>());

// HBM (seq, head-dim/dword) → Q/K/Out offset layout by input layout. BHSD: per-head contiguous → static stride
// Stride (byte-identical). BSHD: heads interleaved → dynamic stride heads·Stride. Unused branch folds away.
template <bool kBSHD, int Stride>
__device__ __forceinline__ auto seq_layout(int heads) {
  if constexpr (kBSHD)
    return opus::make_layout(Shape2, opus::make_tuple(heads * Stride, opus::number<1>{}));
  else
    return opus::make_layout(Shape2, opus::make_tuple(opus::number<Stride>{}, opus::number<1>{}));
}

// qg2 LDS park: entire q-group 2 (all HeadDimTiles d-tiles) staged in LDS, addressed in DWORDS by
// (wave, d-tile, lane); each lane holds a bf16x4 (2 dwords) → lane stride 2.
constexpr int Q2HalfTiles = HeadDimTiles / 2;  // 4 = #d-tiles per half; also the lo→hi stride (d_pair → d_pair+4)
constexpr int Q2LdsDwords = NumWaves * HeadDimTiles * WaveSize * 2;  // 8192 dwords = whole qg2 (32 KB)
constexpr auto Q2Lds = opus::make_layout(Shape3, strides<HeadDimTiles * WaveSize * 2, WaveSize * 2, 2>());
// LITE skip-vote ring: live window {step,step-1} → depth 2. Reuses the 4 KB lds_q2 slot freed by evicting qg2's last
// d-tile to VGPR.
constexpr int VerdictRing = 2;

// ─── Grid + MFMA thread→fragment index maps ─────────────────────────────────
// Same rule as the memory layouts (offset = Σ coordᵢ·strideᵢ) but yield LOGICAL indices (q-tile/kv-row/frag elem), not
// bytes.

// (block_x, wave, q-group) → absolute q-tile index (WG owns QTilesPerWG, wave QTilesPerWave, q-group 1).
constexpr auto QTileGrid = opus::make_layout(Shape3, strides<QTilesPerWG, QTilesPerWave, 1>());

// (load-wave, local row) → kv-row owned: LoadWaves × KRowsPerLoadWave tile the K block.
constexpr auto KLoadRow = opus::make_layout(Shape2, strides<KRowsPerLoadWave, 1>());

// (load-wave, i) → V_t super-chunk to L1-prefetch (VtChunks chunks / LoadWaves).
constexpr auto VtPrefetch = opus::make_layout(Shape2, strides<VtChunks / LoadWaves, 1>());

// MFMA 16x16x16 A/B fragment: lane=(col=lane&15, quad=lane>>4); 16-wide contraction tiled by 4 quad-lanes, QuadSpan
// elems each.
constexpr int QuadSpan = MfmaK / 4;  // 4 bf16 contraction elems / quad-lane
// (d-tile, quad) → head-dim element, for Q / Out (bf16 elements).
constexpr auto HdFrag = opus::make_layout(Shape2, strides<MfmaK, QuadSpan>());

// Head-first chiplet swizzle: chunk_size=blocks_per_head co-locates one (B,H)'s q-blocks on one XCD's L2 → K+V stay
// resident.
constexpr int NumXcds = 8;
__device__ __forceinline__ int chiplet_swizzle(int wg_id, int num_wgs, int chunk_size) {
  const int block = NumXcds * chunk_size;
  const int limit = (num_wgs / block) * block;
  if (wg_id >= limit) return wg_id;
  const int xcd = wg_id % NumXcds;
  const int local_id = wg_id / NumXcds;
  const int chunk_idx = local_id / chunk_size;
  const int pos_in_chunk = local_id % chunk_size;
  return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

// ─── Dense tail KV-split scratch (dense only) ────────────────────────────────
// 384-row q-blocks run in rounds of NumCUs; a fractional last round strands CUs. Recover it by splitting each
// stranded q-block's kv work G ways (split_ways): part p walks every G-th kv block from offset p, writing an fp32
// (o,m,l) partial; merge_parts recombines a q-block's G partials via online-softmax into bf16.
constexpr int NumCUs = 304;           // MI300X compute units (= one occupancy-1 CTA each)
constexpr int MaxParts = 2 * NumCUs;  // upper bound on total parts (the planner caps the split at ≤2 CU rounds)
constexpr int PartRows = QTilesPerWG * QTileRows;  // 384 q-rows owned by one part (== one q-block)
constexpr int PartOFloats = PartRows * HeadDim;    // fp32 `o` values per part (384·128)
constexpr int PartMlFloats = PartRows * 2;         // fp32 (m,l) pair per row, per part (384·2)
// One flat float buffer: `o` region then `(m,l)` region, each addressed by a compile-time-strided opus layout.
constexpr auto PartO = opus::make_layout(Shape3, strides<PartOFloats, HeadDim, 1>());  // (part, row, hd) → o
constexpr auto PartMl = opus::make_layout(Shape2, strides<PartMlFloats, 2>());         // (part, row) → m; +1 → l
struct TailScratch {
  float* o;   // [MaxParts][PartRows][HeadDim] fp32 output partials, indexed by PartO(part, row, hd)
  float* ml;  // [MaxParts][PartRows][2]       fp32 (row max, row sum), indexed by PartMl(part, row){+1}
  __device__ __host__ explicit TailScratch(float* ws) : o(ws), ml(ws + (long long)MaxParts * PartOFloats) {}
};

// Host-uploaded schedule (static single-stream buffer) read by split+merge kernels: [0,parts)=per-part
// (q_block,bh,_,_); [MaxParts,+tail)=per-q-block (q_block,bh,part_base,n_parts=G).
__device__ int4 tail_sched[2 * MaxParts];
// ===== FORWARD KERNEL: online softmax + attention_forward (dense / lite / split)

// Online-softmax update for ONE q-group (per-q-group scoping keeps kernel <256-VGPR; inlining all spills ~110). Folds
// block into running (max,sum,o_acc): skip-rescale gate, exp2+NaN-safe pack, row_sum; cross-lane sum to epilogue.
__device__ __forceinline__ void softmax_update(
    fp32x4_t scores[],
    fp32x4_t oreg[],
    bf16x4_t probs[],
    float& row_m,
    float& row_l,
    float scale_log2e,
    float bmax_local) {
  const float new_m_local = bmax_local * scale_log2e;
  const bool first_block = row_m == -FLT_MAX;
  const bool lane_ok = (new_m_local - row_m) < RescaleThreshold;
  const bool all_ok = wave_all(lane_ok) && !first_block;

  float neg_max;
  float rescale = 1.f;
  if (__builtin_expect(all_ok, 1)) {
    neg_max = -row_m;
  } else {
    const float bmax = cross_lane_max(bmax_local);
    const float old_m = row_m;
    const float new_m = fmaxf(old_m, bmax * scale_log2e);
    if (new_m != old_m) {
      rescale = first_block ? 0.f : fast_exp2(old_m - new_m);
      if (!first_block) {
        // o_acc writeback fence: "+v"(oreg)+"memory" orders rescale after pv's asm MFMA writes (LLVM can't see those
        // defs); same-wave MFMA→VALU HW interlock covers writeback.
        asm volatile(""
                     : "+v"(oreg[0]),
                       "+v"(oreg[1]),
                       "+v"(oreg[2]),
                       "+v"(oreg[3]),
                       "+v"(oreg[4]),
                       "+v"(oreg[5]),
                       "+v"(oreg[6]),
                       "+v"(oreg[7])::"memory");
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
    neg_max = -new_m;
  }

  float sub_sum[KvSubtiles];
#pragma unroll
  for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
    const float p0 = fast_exp2(fmaf(scores[kv_sub][0], scale_log2e, neg_max));
    const float p1 = fast_exp2(fmaf(scores[kv_sub][1], scale_log2e, neg_max));
    const float p2 = fast_exp2(fmaf(scores[kv_sub][2], scale_log2e, neg_max));
    const float p3 = fast_exp2(fmaf(scores[kv_sub][3], scale_log2e, neg_max));
    sub_sum[kv_sub] = (p0 + p1) + (p2 + p3);
    // No NaN guard (hot path): exp2 → [0,1] or NaN (never Inf); NaN packs to bf16 NaN → PV MFMA → fp32 NaN → epilogue
    // pack canonicalizes to 0x7FFF. Guarding = cmp+cndmask/elem for no effect.
    probs[kv_sub] = pack4_bf16</*MaybeNaN=*/false>(p0, p1, p2, p3);
  }
  const float block_sum = vadd_f32(vadd_f32(sub_sum[0], sub_sum[1]), vadd_f32(sub_sum[2], sub_sum[3]));
  row_l = vfma_f32(row_l, rescale, block_sum);
}

template <bool kBSHD, bool isLite = false, bool Split = false>
__global__ __launch_bounds__(BlockSize, 1) void attention_forward(
    const bf16_t* __restrict__ Q,
    const bf16_t* __restrict__ K,
    const bf16_t* __restrict__ V,  // points to V_t (pre-transposed)
    bf16_t* __restrict__ Out,
    int seq_len,
    int seq_len_kv,
    float scale_log2e,  // seq_len = Q rows; seq_len_kv = K/V rows (cross-attn)
    int heads,          // #heads — only needed for BSHD addressing (kBSHD)
    // ── LiteAttention skip (isLite only; dense passes the defaults) ──
    const int16_t* __restrict__ read_list = nullptr,     // this phase's RLE keep-list (one row per CTA)
    int16_t* __restrict__ write_list = nullptr,          // RLE keep-list emitted for the next timestep
    const int16_t* __restrict__ must_do_list = nullptr,  // global force-compute ranges [len,s0,e0,…] (nullable)
    float thr = 0.f,
    int phase = 0,  // skip threshold (log2) + reverse-list phase
    // ── Tail KV-split ──
    float* __restrict__ part_buf = nullptr,  // Split: the TailScratch workspace
    int grid_nqb = 0,                        // real #q-blocks when the grid is 1-D (prefix/split);
                                             //   0 ⇒ 2-D full grid (uses gridDim.x)
    int split_ways = 0,                      // Split: G = strided parts per q-block (= the kv stride)
    // ── LiteAttention LPT work-balance reorder (lite full grid only) ──
    const int* __restrict__ lpt_order = nullptr  // lpt_order[grid slot] = work-group to run here (longest-first)
) {
  static_assert(!(isLite && Split), "lite has no tail KV-split; Split is dense-only");
  // ROADMAP (one CTA = one 384-row q-block of one (batch,head)):
  //   1. Thread/lane setup + per-(B,H) HBM base offsets for Q/K/V_t/Out.
  //   2. Prologue — Q→VGPR/LDS, first K-block→LDS, zero accumulators.
  //   3. Main loop over K-blocks — 8-wave warp-spec: 2 groups swap MFMA-vs-memory across a mid-iter barrier (qk/pv
  //   overlap softmax+K/V loads).
  //   4. Epilogue — finish last block; Split: raw fp32 partials to tail scratch, else normalize by denom + store bf16
  //   to Out.
  // Dense/lite(skip)/split share this body; `if constexpr` selects per template.
  //
  // MFMA 16x16x16 lane split: col = q-col/kv-row in tile, quad = which 4 of 64. 8 waves → 2 groups of 4 for warp-spec.
  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane = tid & 63;
  const int col = lane & 15;   // q-col / kv-row within a 16-wide tile
  const int quad = lane >> 4;  // which group of 4 lanes (0..3)
  const bool group_a = (wave_id < LoadWaves);
  const int wave_in_group = group_a ? wave_id : (wave_id - LoadWaves);

  // K-LDS XOR swizzle (stride-64, bank-conflict-free), matches direct_lds_k's HBM-source swizzle. Read-offset fields
  // are bit-disjoint (OR≡add): col[8-11] | kv_sub[12-13] | k_buf[14] | (quad^col_lo)[3-4] | (d_pair^col_hi)[5-6];
  // d-tile-high rides offset1=16.
  const uint32_t col_lo = (uint32_t)(col & 3);
  const uint32_t col_hi = (uint32_t)(col >> 2);
  // worked example col=5,quad=2: col_lo=1 → (5<<8) | ((2^1)<<3) = 0x500 | 0x18 = 0x518.
  const uint32_t qk_swz_base = (uint32_t)(((unsigned)col << 8) | (((unsigned)quad ^ col_lo) << 3));

  // Grid: x = #q-blocks (384-row = ceil(seq_len/384)), y = batch·heads. One CTA per (q-block, bh).
  const int num_q_blocks = gridDim.x;
  const int num_bh = gridDim.y;
  const int wg_id_raw = blockIdx.y * num_q_blocks + blockIdx.x;
  const int num_wgs = num_q_blocks * num_bh;
  // Block selection. Split: (q_block,bh) from tail_sched. 2-D full grid (grid_nqb==0): lite uses identity (chiplet
  // swizzle HARMFUL under uneven sparsity: 0.83x); dense uses head-first chiplet swizzle (co-locate on one XCD's L2,
  // ~+1.4%).
  int q_block, bh_index;
  if constexpr (Split) {
    // One CTA per part: its (q_block,bh) come from the uploaded per-part schedule entry.
    const int4 sched_entry = tail_sched[blockIdx.x];
    q_block = sched_entry.x;
    bh_index = sched_entry.y;
  } else if (grid_nqb) {
    // 1-D grid. lite's lpt_order runs heaviest q-blocks first (null=identity); pure grid-slot→WG reorder →
    // bit-identical. Radix is grid_nqb (TRUE #q-blocks), not gridDim.x (prefix/split's x-dim is #parts).
    const int wg = lpt_order ? lpt_order[wg_id_raw] : wg_id_raw;
    q_block = wg % grid_nqb;
    bh_index = wg / grid_nqb;
  } else {
    const int wg_id = isLite ? wg_id_raw : chiplet_swizzle(wg_id_raw, num_wgs, num_q_blocks);
    q_block = wg_id % num_q_blocks;
    bh_index = wg_id / num_q_blocks;
  }

  // Q-tile ownership: each wave owns QTilesPerWave(=3) 16-row q-tiles. q_tile_start_row = tile's origin seq-row, lane
  // `col` reads/writes row (start+col). Parallel arrays indexed by q_group; valid[] masks past-seq-end.
  const int num_q_tiles = (seq_len + QTileRows - 1) / QTileRows;  // ceil
  bool valid[QTilesPerWave];
  int q_tile_start_row[QTilesPerWave];
#pragma unroll
  for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
    const int q_tile = QTileGrid(q_block, wave_id, q_group);
    valid[q_group] = q_tile < num_q_tiles;
    q_tile_start_row[q_group] = valid[q_group] ? q_tile * QTileRows : 0;
  }

  // ceil block count; last_kv_rows ∈ [1,64] = valid rows in the last (partial) K/V block.
  const int num_kv_blocks = (seq_len_kv + KvBlockRows - 1) / KvBlockRows;
  const int last_kv_rows = seq_len_kv - (num_kv_blocks - 1) * KvBlockRows;
  // KV stride/offset over [0,num_kv_blocks). Dense Split part walks every split_ways-th block from its offset; everyone
  // else (incl all lite) walks every block (stride 1, off 0).
  const int kv_stride = Split ? split_ways : 1;
  const int kv_offset = Split ? (blockIdx.x % split_ways) : 0;

  // Cross-attention offsets. Q/K/Out keep TRUE seq lens, SRD bounded to valid byte extent: partial last q-tile/K-block
  // reads 0 / drops store IN HARDWARE (zero-VGPR OOB). V_t padded to num_kv_blocks·64 rows (tail zeroed). Only Q/K/Out
  // addressing is layout-dependent.
  const auto QOLayout = seq_layout<kBSHD, HeadDim>(heads);      // Q/Out (seq, head-dim elems)
  const auto KHbmDword = seq_layout<kBSHD, KRowDwords>(heads);  // K     (seq, dwords)
  const int k_row_dwords = kBSHD ? heads * KRowDwords : KRowDwords;
  long long bh_offset_q, bh_offset_kv;
  if constexpr (kBSHD) {
    const long long b = bh_index / heads, h = bh_index % heads;
    bh_offset_q = b * (long long)seq_len * heads * HeadDim + h * HeadDim;
    bh_offset_kv = b * (long long)seq_len_kv * heads * HeadDim + h * HeadDim;
  } else {
    bh_offset_q = (long long)bh_index * seq_len * HeadDim;
    bh_offset_kv = (long long)bh_index * seq_len_kv * HeadDim;
  }
  const long long bh_offset_v = (long long)bh_index * num_kv_blocks * KvBlockRows * HeadDim;
  // BHSD: per-head contiguous (seq·HeadDim·2). BSHD: rows interleaved → ×heads span.
  const uint32_t q_bytes = (uint32_t)((long long)seq_len * (kBSHD ? heads : 1) * HeadDim * 2);
  const uint32_t kv_bytes = (uint32_t)((long long)seq_len_kv * (kBSHD ? heads : 1) * HeadDim * 2);
  // V_t per-head-contiguous (padded to num_kv_blocks·64 rows). Bound SRD to that span so a stray pv offset CLAMPS to 0,
  // not faults.
  const uint32_t v_bytes = (uint32_t)((long long)num_kv_blocks * KvBlockRows * HeadDim * 2);
  const auto q_buffer = make_buffer(Q + bh_offset_q, q_bytes);
  const auto k_buffer = make_buffer(K + bh_offset_kv, kv_bytes);
  const auto v_buffer = make_buffer(V + bh_offset_v, v_bytes);
  const auto out_buffer = make_buffer(Out + bh_offset_q, q_bytes);

  constexpr int KLdsBufDwords = KvBlockRows * KLdsStrideDwords;  // 4096 dwords/buffer (64 rows × 64)
  __shared__ uint32_t lds_k[2][KLdsBufDwords];                   // 32 KB (2-buf, stride 64)
  // V_t consumed directly from L1 (no LDS staging). Q-park: qg0,1 in VGPR; qg2 (all 8 d-tiles) in one 32 KB LDS array
  // (Q2Lds, d-tile t @ slot t). 32 KB K + 32 KB qg2 = exactly 64 KB LDS; qg2 reads (lgkmcnt) overlap pv's V loads
  // (vmcnt).
  __shared__ uint32_t lds_q2[Q2LdsDwords];      // 32 KB (entire qg2, d-tiles 0..7; lite: 0..6)
  auto load_q2 = [&](int d_tile) -> bf16x4_t {  // d-tile d_tile of qg2 for this lane (d_tile < last for lite)
    return *reinterpret_cast<const bf16x4_t*>(&lds_q2[Q2Lds(wave_id, d_tile, lane)]);
  };
  // LITE skip-vote ring, reusing the evicted last-d-tile lds_q2 slot. Per-wave region → no cross-wave contention; dword
  // (0/1) avoids sub-dword RMW.
  auto sverdict = [&](int w, int r) -> uint32_t& { return lds_q2[Q2Lds(w, HeadDimTiles - 1, 0) + r]; };

  // K[kv_block] HBM→LDS into double-buffer half `k_buf` (group B). Wave w owns rows [w·16,+16). XOR swizzle on the HBM
  // SOURCE: lane L fetches dword of swizzled 8B unit (L>>1)^c, in-unit dword L&1, c=kv_row&0xF. LDS contiguous → unit p
  // holds HBM (p^c), matches qk read.
  constexpr uint32_t KRowBytes = (uint32_t)(KLdsStrideDwords * 4);
  // K block stride (bytes) = KvBlockRows × seq-row stride. BHSD folds to 16384; BSHD is ×heads (rows interleaved).
  const uint32_t KBlockBytes = (uint32_t)(KvBlockRows * k_row_dwords * 4);
  const int wave_row0 = KLoadRow(wave_in_group, 0);           // first kv-row this load-wave owns
  const uint32_t k_row_bytes = (uint32_t)(k_row_dwords * 4);  // per-seq-row HBM byte stride
  auto direct_lds_k = [&](int kv_block, int k_buf) {
    const uint32_t lds_row0 = to_scalar(lds_addr(&lds_k[k_buf][KLdsDword(wave_row0, 0)]));
    const uint32_t hbm_block = to_scalar((uint32_t)((uint64_t)kv_block * KBlockBytes));
    if constexpr (kBSHD) {
      // BSHD seq-row stride is a runtime heads·KRowDwords MULTIPLY → per-lane offen VGPR can't be remat → LLVM pins all
      // 16 per-c values and spills 4 EVERY iter. Fix: scalar soffset = hbm_block + (wave_row0+c)·stride·4
      // (wave-uniform) + cheap per-lane offen = swz_lane·4 = (lane·4) XOR (c·8) → 0 spill.
      const uint32_t lane_x4 = (uint32_t)(lane << 2);  // per-lane base
      const uint32_t base_soff = to_scalar(hbm_block + (uint32_t)wave_row0 * k_row_bytes);
#pragma unroll
      for (int c = 0; c < KRowsPerLoadWave; ++c) {
        const uint32_t soff = to_scalar(base_soff + (uint32_t)c * k_row_bytes);
        // == (lane^(c<<1))<<2: XOR distributes through <<2, and (c<<1)<<2 = c<<3, so this is swz_lane·4.
        const uint32_t off = lane_x4 ^ (uint32_t)(c << 3);
        buffer_load_b32_to_lds(k_buffer, lds_row0 + c * KRowBytes, off, soff);
      }
    } else {
// BHSD: static stride → kv_row·stride·4 folds to a shift LLVM remats for free (0 spill). Keep single-VGPR-offset form
// byte-identical.
#pragma unroll
      for (int c = 0; c < KRowsPerLoadWave; ++c) {
        const int kv_row = KLoadRow(wave_in_group, c);
        // 8B-unit = (lane>>1)^c, in-unit dword = lane&1. e.g. lane=5,c=2: unit=(2^2)=0, dword=1.
        const uint32_t swz_lane = (uint32_t)((((lane >> 1) ^ c) << 1) | (lane & 1));
        const uint32_t hbm_byte = (uint32_t)((KHbmDword(kv_row, 0) + swz_lane) * 4);
        buffer_load_b32_to_lds(k_buffer, lds_row0 + c * KRowBytes, hbm_byte, hbm_block);
      }
    }
  };

  // ── LiteAttention skip machinery (lite only; compiled away for dense) ─────────
  // [&]-lambdas over locals: a read-list cursor (which K-blocks this step computes) + a write-list emitter (next-step
  // RLE + MustDoList force-compute). ── Read-list cursor: which K-blocks this step computes ──────────────────────
  // Skip-list row-major [bh,q_block,num_k_blocks]; lbase = CTA's row origin. read_list[lbase]=RLE length then ranges.
  // lstep = reverse-list dir (phase 1 asc, 0 desc). rl() reads via to_scalar → cursor stays in SGPRs, off the MFMA VGPR
  // budget.
  const int lstep = phase ? 1 : -1;
  long long lbase = 0;
  int list_len = 0;
  auto rl = [&](int idx) -> int { return to_scalar((int)read_list[lbase + idx]); };

  // Cursor state: range_block=current kept block; range_end=exclusive walk-end; rle_idx=next marker pair;
  // range_cont=last step stayed in range (drives record()'s range-end marker). Honors kv_stride/kv_offset, but lite is
  // never Split → stride 1.
  int rle_idx = 0, range_block = 0, range_end = 0;
  bool range_cont = false;
  auto open_range = [&]() {  // load the next range at rle_idx (always non-empty)
    range_block = rl(rle_idx) + lstep;
    range_end = rl(rle_idx - 1) + lstep;
    rle_idx -= 2;
  };
  auto next_kept = [&](int cur) -> int {  // next kept block after `cur` (one step), -1 if done
    const int next_blk = cur + lstep;
    if ((lstep > 0) ? (next_blk < range_end) : (next_blk > range_end)) {
      range_block = next_blk;
      range_cont = true;
      return next_blk;
    }
    range_cont = false;
    if (rle_idx >= 1) {
      open_range();
      return range_block;
    }
    return -1;
  };
  auto advance = [&](int cur, int n) -> int {  // skip `n` kept blocks (stride), -1 if exhausted
    for (int s = 0; s < n && cur >= 0; ++s)
      cur = next_kept(cur);
    return cur;
  };

  // ── Write-list emitter: next-step skip list (tid 0 only) ─────────────────────

  int write_idx = 1;       // output cursor (slot 0 holds the final length)
  bool skip_state = true;  // running skip/compute state, for transition detection
  bool prev_ended_range = false;

  // MustDoList: global force-compute RLE [len,s0,e0,…] of [start,end) K-blocks pinned "compute" (vote can't drop them,
  // e.g. sinks). Walked monotonically in lstep order; nullptr/len-0 ⇒ dormant (byte-identical to plain skip).
  const bool have_md = isLite && (must_do_list != nullptr) && ((int)must_do_list[0] > 0);
  const int md_nr = have_md ? ((int)must_do_list[0] >> 1) : 0;       // # of [s,e) ranges
  int md_ri = (lstep > 0) ? 0 : (md_nr - 1);                         // range cursor (walks in lstep dir)
  auto md_lo = [&](int r) { return (int)must_do_list[1 + 2 * r]; };  // range r start (inclusive)
  auto md_hi = [&](int r) { return (int)must_do_list[2 + 2 * r]; };  // range r end   (exclusive)
  auto md_in_cur = [&](int b) { return md_ri >= 0 && md_ri < md_nr && b >= md_lo(md_ri) && b < md_hi(md_ri); };
  // Is `block` forced-compute? Advance the monotonic cursor past ranges we've walked off, then test.
  auto md_force = [&](int block) -> bool {
    if (!have_md) return false;
    if (lstep > 0)
      while (md_ri < md_nr && block >= md_hi(md_ri))
        ++md_ri;  // ascending: drop ranges behind us
    else
      while (md_ri >= 0 && block < md_lo(md_ri))
        --md_ri;  // descending
    return md_in_cur(block);
  };
  // block's completed vote: AND the 8 wave votes (skip iff all voted skip); MustDoList pins → compute.
  auto vote = [&](int block, int verdict_step) -> bool {
    int sk = 1;
#pragma unroll
    for (int w = 0; w < NumWaves; ++w)
      sk &= (int)sverdict(w, verdict_step & (VerdictRing - 1));
    bool skip = (sk != 0);
    if (md_force(block)) skip = false;  // MustDoList: pin as compute
    return skip;
  };
  // RLE-encode the vote inline: a marker on each skip<->compute flip, plus a range-end marker when a kept range closes.
  auto record = [&](int block, int verdict_step, bool is_range_end) {
    bool skip = vote(block, verdict_step);
    if (skip != skip_state) {
      write_list[lbase + write_idx++] = (int16_t)block;
      skip_state = skip;
    }
    if (is_range_end) {
      skip_state = true;
      if (!skip) write_list[lbase + write_idx++] = (int16_t)(block + lstep);
    }
  };

  // Prologue: first processed K-block → LDS (group B only; A idle until iter 1). Dense starts at kv_offset; lite
  // advances the cursor kv_offset KEPT blocks in.
  int first_blk = kv_offset;
  bool has_any_block = (kv_offset < num_kv_blocks);
  if constexpr (isLite) {
    const int num_k_blocks = num_kv_blocks + 2;               // RLE row width: [len, markers…]
    const int nqb_list = grid_nqb ? grid_nqb : num_q_blocks;  // 1-D grids: gridDim.x ≠ #q-blocks
    lbase = ((long long)bh_index * nqb_list + q_block) * num_k_blocks;
    list_len = to_scalar((int)read_list[lbase]);
    rle_idx = list_len;
    if (list_len > 0) {
      open_range();
      first_blk = advance(range_block, kv_offset);
    }
    has_any_block = (first_blk >= 0);
  }
  if (!group_a && has_any_block) direct_lds_k(first_blk, /*k_buf=*/0);  // step starts at 0 → slot 0

  // Persistent VGPR Q (no reload across K-blocks). Lane (q_tile_start_row+col) owns one q-row; each load_bf16x4 grabs
  // 4 head-dims of d-tile d (quad picks which 4 of 16). qg0,1 in VGPR; qg2 in lds_q2 (d-tile t @ slot t).
  bf16x4_t q_frag[2][HeadDimTiles];
  auto load_q = [&](int q_group, int d) -> bf16x4_t {
    return valid[q_group] ? load_bf16x4(q_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(d, quad)) * 2)
                          : bf16x4_t{0, 0, 0, 0};
  };
// NOTE: keep qg0/qg1 as SEPARATE prologue load loops — collapsing into one `for (g)` loop re-introduces a VGPR spill
// (allocator coloring is load-structure-sensitive). The qk-phase qg0/qg1 collapse is safe; this prologue one is not.
#pragma unroll
  for (int d = 0; d < HeadDimTiles; ++d)
    q_frag[0][d] = load_q(0, d);
#pragma unroll
  for (int d = 0; d < HeadDimTiles; ++d)
    q_frag[1][d] = load_q(1, d);
  // qg2: all 8 d-tiles → lds_q2 (d-tile d @ slot d), HBM→VGPR→LDS. LITE: LAST d-tile stays in VGPR (q2_last), freeing
  // its 4 KB slot for the skip-vote ring; dense parks all 8.
  bf16x4_t q2_last{0, 0, 0, 0};
#pragma unroll
  for (int d = 0; d < HeadDimTiles; ++d) {
    bf16x4_t v = load_q(2, d);
    if constexpr (isLite) {
      if (d == HeadDimTiles - 1) {
        q2_last = v;
        continue;
      }
    }
    u32x2_t raw;
    __builtin_memcpy(&raw, &v, 8);
    const uint32_t off = Q2Lds(wave_id, d, lane);
    lds_q2[off + 0] = raw[0];
    lds_q2[off + 1] = raw[1];
  }
  __syncthreads();  // lds_q2 visible to all reads below

  fp32x4_t o_acc[QTilesPerWave][HeadDimTiles];
#pragma unroll
  for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
#pragma unroll
    for (int d = 0; d < HeadDimTiles; ++d)
      o_acc[q_group][d] = zero_fp32x4();
  float row_max[QTilesPerWave], row_sum[QTilesPerWave];
#pragma unroll
  for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
    row_max[q_group] = -FLT_MAX;
    row_sum[q_group] = 0.f;
  }

  // Loop-carried per-wave state: scores written by qk_phase before softmax reads, probs by softmax_update before pv.
  // Zero-init is logically dead but LOAD-BEARING: one clean def point for the allocator; dropping it respills 64 VGPR.
  fp32x4_t scores[QTilesPerWave][KvSubtiles];
  bf16x4_t probs[QTilesPerWave][KvSubtiles];
#pragma unroll
  for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
#pragma unroll
    for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
      scores[q_group][kv_sub] = zero_fp32x4();
      probs[q_group][kv_sub] = bf16x4_t{0, 0, 0, 0};
    }

  auto qk_phase = [&](int k_buf, fp32x4_t scores[QTilesPerWave][KvSubtiles]) {
    // Swizzled LDS read addr = OR of disjoint bit-fields (OR≡add): qk_swz_base[11:3] | (d_pair^col_hi)<<5 (bank
    // swizzle) | kv_sub<<12 | k_buf<<14 (which of 2 K LDS halves). d_pair^col_hi is the only per-read compute.
    const uint32_t lane_off = (uint32_t)k_buf << 14;
    auto k_addr = [&](int kv_sub, int d_pair) -> uint32_t {
      return qk_swz_base | (((uint32_t)d_pair ^ col_hi) << 5) | ((uint32_t)kv_sub << 12) | lane_off;
    };

    // Loop i in [0,TotalReads) splits 3 ways: kv_sub=i%KvSubtiles (scores tile), d_pair=i/KvSubtiles (head-dim
    // col-pair, K+Q frag), slot=i%Depth (ring reg s0/s1/s2). e.g. i=5 → kv_sub=1,d_pair=1,slot=2. Depth(3) coprime with
    // KvSubtiles(4) so the ring rotates independently.
    constexpr int TotalReads = KvSubtiles * 4;  // 16 ds_read2 = 4 kv-subtiles × 4 d-pairs
    constexpr int Depth = 3;                    // K-tile prefetch ring depth
    constexpr int Q2Depth = 2;                  // qg2-from-LDS ping-pong depth
    mfma_prio_hi();
    sched_fence();

    // K pre-issues (depth-3 ring): each ds_read2 fetches a d-tile PAIR (lo @offset0, hi @offset1).
    u32x4_t s0 = ds_read2_b64_imm<0, 16>(k_addr(0, 0));
    u32x4_t s1 = ds_read2_b64_imm<0, 16>(k_addr(1, 0));
    u32x4_t s2 = ds_read2_b64_imm<0, 16>(k_addr(2, 0));

    // qg2 (only LDS-resident q-group): depth-2 ping-pong per head-dim half h (h=0→d-tiles 0..3, h=1→4..7). q2[h][slot]
    // = d-tile slot+h·Q2HalfTiles.
    bf16x4_t q2[2][Q2Depth];
#pragma unroll
    for (int p = 0; p < Q2Depth; ++p)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        q2[h][p] = load_q2(p + h * Q2HalfTiles);

    // Q fragment for one qk MFMA (g, half h, col-pair d_pair): qg0/qg1 in VGPR (q_frag); qg2 streams LDS ping-pong
    // (q2[h][slot]) — except lite-only qg2 last d-tile, evicted to VGPR (q2_last) so its LDS slot holds the skip-vote
    // ring.
    auto q_fragment = [&](int g, int h, int d_pair, int q2_slot) -> bf16x4_t {
      if (g < 2) return q_frag[g][d_pair + h * Q2HalfTiles];
      if constexpr (isLite)
        if (h == 1 && d_pair == Q2HalfTiles - 1) return q2_last;
      return q2[h][q2_slot];
    };

#pragma unroll
    for (int i = 0; i < TotalReads; ++i) {
      const int kv_sub = i % KvSubtiles;
      const int d_pair = i / KvSubtiles;
      const int slot = i % Depth;
      u32x4_t& cur = (slot == 0) ? s0 : (slot == 1) ? s1 : s2;
      if (i == TotalReads - 1)
        wait_lgkm<0>(cur);
      else
        wait_lgkm<Depth - 1>(cur);
      // Contract d-tile pair {lo=d_pair, hi=d_pair+Q2HalfTiles} into every q-group's scores: K from this ring slot, Q
      // from q_frag (qg0,1) or q2 (qg2).
      const bf16x4_t k_pair[2] = {lo2(cur), hi2(cur)};
      const int q2_slot =
          d_pair % Q2Depth;  // qg2 ping-pong slot keyed on d_pair (all 4 kv_sub of one d_pair reuse the same qg2 read)
#pragma unroll
      for (int g = 0; g < QTilesPerWave; ++g)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const bf16x4_t qf = q_fragment(g, h, d_pair, q2_slot);
          // First MFMA into this scores tile inits from zero (no separate memset); later d_pair/h accumulate in place.
          if (d_pair == 0 && h == 0)
            scores[g][kv_sub] = asm_mfma_zero(k_pair[0], qf);
          else
            asm_mfma(k_pair[h], qf, scores[g][kv_sub]);
        }
      // Prefetch the next K pair into this ring slot (depth-3 ahead).
      if (i + Depth < TotalReads) {
        const int next_i = i + Depth;
        cur = ds_read2_b64_imm<0, 16>(k_addr(next_i % KvSubtiles, next_i / KvSubtiles));
      }
      // Prefetch the next qg2 d-tile pair (d_pair+Q2Depth) into its ping-pong slot.
      if (kv_sub == KvSubtiles - 1 && d_pair + Q2Depth < Q2HalfTiles) {
        const int pf_d_pair = d_pair + Q2Depth;  // d_pair being prefetched
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          // LITE: the last d-tile (h=1, top d_pair) lives in q2_last (VGPR), not LDS.
          if constexpr (isLite) {
            if (h == 1 && pf_d_pair == Q2HalfTiles - 1) continue;
          }
          q2[h][q2_slot] = load_q2(pf_d_pair + h * Q2HalfTiles);
        }
      }
    }
    mfma_prio_lo();
  };

  // ─── PV: O += V·Pᵀ via mfma(A=V_t, B=P). 16 super-chunks/iter, each lane load_u32x4s its
  // pre-packed bf16x8 from V_t (VtLayout). Depth-4 prefetch ring. chunk → hd-tile chunk/2 +
  // kv-subtile pair; all q-groups share the V operand. ────────────────────────
  auto chunk_byte_off = [&](int kv_iter, int chunk) -> uint32_t {
    return (uint32_t)(VtLayout(kv_iter, chunk, lane) * 2);  // bf16 elems → bytes
  };

  // verdict_step = processing step of the block being pv'd (verdict-ring slot; ignored !isLite). Part B soft
  // block-skip: a "skip" wave does NO pv (its probs were zeroed in softmax → result unchanged), dropping 0-adding MFMAs
  // + V_t loads. Vote is in LDS from this wave's softmax (A: prev iter P2; B: this iter P1), visible across record()'s
  // barriers.
  auto pv_phase = [&](int kv_iter, const bf16x4_t probs[QTilesPerWave][KvSubtiles], int verdict_step) {
    if constexpr (isLite) {
      if (sverdict(wave_id, verdict_step & (VerdictRing - 1))) return;
    }
    constexpr int PvDepth = 4;  // V_t prefetch ring depth (VtChunks=16 super-chunks / kv-iter)
    mfma_prio_hi();
    sched_fence();
    u32x4_t v_ring[PvDepth];
#pragma unroll
    for (int p = 0; p < PvDepth; ++p)
      v_ring[p] = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, p));
#pragma unroll
    for (int chunk = 0; chunk < VtChunks; ++chunk) {
      // chunk in [0,16): d_tile=chunk/2 (hd-tile 0..7); kv_sub0=(chunk&1)*2 (kv-subtile pair base, even→{0,1}
      // odd→{2,3}). e.g. chunk=5 → d_tile=2, kv_sub0=2.
      const int d_tile = chunk >> 1;
      const int kv_sub0 = (chunk & 1) * 2;
      u32x4_t& cur = v_ring[chunk % PvDepth];
      // one b128 V load = a kv-subtile pair; v_pair[h] is the operand for kv-subtile kv_sub0+h.
      const bf16x4_t v_pair[2] = {lo2(cur), hi2(cur)};
      // Issue the next ring V-load BEFORE this chunk's MFMAs (latency lead).
      if (chunk + PvDepth < VtChunks) cur = load_u32x4(v_buffer, (int)chunk_byte_off(kv_iter, chunk + PvDepth));
// v_pair[h] feeds all q-groups back-to-back before the next half — shorter V-reg lifetime than q-group-outer ordering.
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
          asm_mfma(v_pair[h], probs[q_group][kv_sub0 + h], o_acc[q_group][d_tile]);
    }
    sched_fence();
    mfma_prio_lo();
  };

  // L1 prefetcher for V_t[kv_iter_next], fired from Group A's softmax-only P2 (vmcnt per-wave → independent of
  // consumers' load_u32x4). Wave-split 4×4 = 16 unique cache-line requests hit all L1 sets. Dest v[140:143] = discard
  // sink, LAST op before end-of-iter wait_vmcnt0.
  auto prefetch_v_t_l1 = [&](int kv_iter_next) {
    if (kv_iter_next < 0) return;
#pragma unroll
    for (int i = 0; i < VtChunks / LoadWaves; ++i) {
      const int chunk = VtPrefetch(wave_in_group, i);
      const uint32_t chunk_off = (uint32_t)(VtLayout(kv_iter_next, chunk, lane) * 2);
      // v[140:143] = discard sink (L1-fill side effect only; data never read). A fixed GPR keeps it out of the
      // allocator's live set.
      asm volatile("buffer_load_dwordx4 v[140:143], %0, %1, 0 offen"
                   :
                   : "v"(chunk_off), "s"(v_buffer)
                   : "v140", "v141", "v142", "v143", "memory");
    }
  };

  auto softmax = [&](fp32x4_t scores[QTilesPerWave][KvSubtiles],
                     bf16x4_t probs[QTilesPerWave][KvSubtiles],
                     int step,
                     int kv_block) {
    // Per-q-group block max — local, not loop-carried.
    float bmax_local[QTilesPerWave];
    // Partial last K/V block (seq_len_kv%64≠0): K-OOB rows read 0 → score 0, polluting denom. Force kv-rows ≥
    // last_kv_rows to -inf (exp2→0). No-op if aligned.
    if (last_kv_rows < KvBlockRows && kv_block == num_kv_blocks - 1) {
      const int row0 = quad * 4;
#pragma unroll
      for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (kv_sub * MfmaTile + row0 + i >= last_kv_rows) {
            scores[0][kv_sub][i] = -FLT_MAX;
            scores[1][kv_sub][i] = -FLT_MAX;
            scores[2][kv_sub][i] = -FLT_MAX;
          }
    }
// Per-lane bmax fold runs HERE (not qk_phase). Softmax runs in P2 while the other group is in pv+qk → the v_max3 VALU
// chain hides under their MFMA writeback.
#pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
      float bm = -FLT_MAX;
#pragma unroll
      for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub) {
        bm = fmaxf(
            bm,
            fmaxf(
                fmaxf(scores[q_group][kv_sub][0], scores[q_group][kv_sub][1]),
                fmaxf(scores[q_group][kv_sub][2], scores[q_group][kv_sub][3])));
      }
      bmax_local[q_group] = bm;
    }
    // LITE skip vote: keep iff some valid q-row's block-max is within thr (log2) of its running max. A skipping wave
    // zeros its probs; record() ANDs the 8 votes.
    if constexpr (isLite) {
      bool any_keep = false;
#pragma unroll
      for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
        const float block_max_log2 = bmax_local[q_group] * scale_log2e;     // raw block-max → log2 units
        const bool within_thr = (block_max_log2 - row_max[q_group]) > thr;  // thr<0: gap above |thr| ⇒ keep
        any_keep = any_keep || (valid[q_group] && within_thr);
      }
      bool skip_wave = !__any(any_keep);  // skip only if NO lane in the wave voted keep
      if (step == 0) skip_wave = false;   // never skip the first processed block
      if (lane == 0) sverdict(wave_id, step & (VerdictRing - 1)) = skip_wave ? 1u : 0u;
      if (skip_wave) {
// Zero probs so the skipped pv adds nothing; the keep predicate guarantees this block never raises the running max →
// state stays correct.
#pragma unroll
        for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
#pragma unroll
          for (int kv_sub = 0; kv_sub < KvSubtiles; ++kv_sub)
            probs[q_group][kv_sub] = bf16x4_t{0, 0, 0, 0};
        return;
      }
    }
// ── Fold + online update, per q-group (free function → scoped temporaries). ──
#pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
      softmax_update(
          scores[q_group],
          o_acc[q_group],
          probs[q_group],
          row_max[q_group],
          row_sum[q_group],
          scale_log2e,
          bmax_local[q_group]);
  };

  // ─── Main loop ─────────────────────────────────────────────────────────
  //   Group A  P1: pv(V_t[N-1]) + qk(K[N])      P2: softmax + V_t prefetch
  //   Group B  P1: K[N+1] HBM→LDS + softmax     P2: pv(V_t[N-1]) + qk(K[N])
  wait_vmcnt0();
  lds_barrier();
  // ─── Pipeline over K-blocks (forward 0..num_kv_blocks-1) ────────────────────
  // `step` = step counter (K double-buffer parity); kv_block/prev_block qk'd/pv'd; next_block streamed by B. Dense+lite
  // share ONE body: dense kv_block follows the arithmetic stride; lite's cursor walks kept blocks (stride 1) + tid 0
  // records the write list (`if constexpr`'d out for dense).
  int prev_block = -1, step = 0;  // step counts PROCESSED blocks (drives K double-buffer parity)
  int kv_block = first_blk;       // current processed block
  bool has_work = has_any_block;

  while (has_work) {
    // Next processed block: lite cursor (every kv_stride-th KEPT block) or dense stride. block_closed_range = "kv_block
    // ended its RLE range" → record's flush.
    int next_block;
    bool has_next;
    if constexpr (isLite) {
      next_block = advance(kv_block, kv_stride);
      has_next = (next_block >= 0);
    } else {
      next_block = kv_block + kv_stride;
      has_next = (next_block < num_kv_blocks);
    }
    const bool block_closed_range = isLite ? !range_cont : false;
    const int k_buf_cur = step & 1, k_buf_next = (step + 1) & 1;
    const bool first = (step == 0);
    if (group_a) {  // P1
      if (!first) pv_phase(prev_block, probs, step - 1);
      qk_phase(k_buf_cur, scores);
    } else {
      if (!first) softmax(scores, probs, step - 1, prev_block);
      if (has_next) direct_lds_k(next_block, k_buf_next);
    }
    __builtin_amdgcn_s_barrier();
    if (group_a) {  // P2
      softmax(scores, probs, step, kv_block);
      // tid 0 folds prev_block's skip verdict into the write-list RLE. Safe: vote stored to sverdict last iter
      // (VerdictRing depth 2 keeps it live).
      if constexpr (isLite) {
        if (tid == 0 && !first) record(prev_block, step - 1, prev_ended_range);
      }
      if (has_next) prefetch_v_t_l1(kv_block);
    } else {
      if (!first) pv_phase(prev_block, probs, step - 1);
      qk_phase(k_buf_cur, scores);
    }
    // End-of-iter barrier: vmcnt(0) drains direct_lds_k writes (buffer_load…lds is VMCNT-tracked, not LGKMCNT) so next
    // qk can't read stale K; lds_barrier covers LGKMCNT.
    wait_vmcnt0();
    lds_barrier();
    prev_block = kv_block;
    ++step;
    prev_ended_range = block_closed_range;
    kv_block = next_block;
    has_work = has_next;
  }
  // Epilogue. The loop defers each block's pv (and B's softmax) by one iter, so prev_block is still pending: B owes a
  // softmax (A did its at P2), both owe the final pv. Lite then flushes the last block's range-end vote + RLE length.
  if (!group_a && step > 0) softmax(scores, probs, step - 1, prev_block);
  if (step > 0) pv_phase(prev_block, probs, step - 1);
  if constexpr (isLite) {
    lds_barrier();  // group B's last-block vote → visible to tid 0
    if (tid == 0 && step > 0) {
      record(prev_block, step - 1, /*is_range_end=*/true);  // finalize inline RLE
      write_list[lbase] = (int16_t)(write_idx - 1);         // RLE length
    }
  }

// ─── Output ────────────────────────────────────────────────────────────
// row_sum summed across the 4 quad-lanes once here (math-equiv, rescale is row-global). inv_sum=0 when row_sum==0
// prevents 0×inf=NaN.
#pragma unroll
  for (int q_group = 0; q_group < QTilesPerWave; ++q_group)
    row_sum[q_group] = cross_lane_sum(row_sum[q_group]);
  if constexpr (Split) {
    // A split part exports raw fp32 partials (o, plus row max `m` and sum `l`), UNSCALED — o/l division + bf16 pack are
    // merge_parts' job. One CTA == one part.
    TailScratch parts(part_buf);
    const int part = blockIdx.x;
#pragma unroll
    for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
      // Absolute row of this (wave, q_group, lane) within the part's 384 rows.
      const int row = (wave_id * QTilesPerWave + q_group) * QTileRows + col;
#pragma unroll
      for (int t = 0; t < HeadDimTiles; ++t)  // store each head-dim fragment (4 fp32) of the o accumulator
        *reinterpret_cast<fp32x4_t*>(&parts.o[PartO(part, row, HdFrag(t, quad))]) = o_acc[q_group][t];
      if (quad == 0) {  // m and l are per-row scalars — only the first quad-lane of the row writes them
        parts.ml[PartMl(part, row)] = row_max[q_group];      // m = running softmax max
        parts.ml[PartMl(part, row) + 1] = row_sum[q_group];  // l = running softmax denominator
      }
    }
    return;
  }
#pragma unroll
  for (int q_group = 0; q_group < QTilesPerWave; ++q_group) {
    if (!valid[q_group]) continue;
    const float inv_sum = (row_sum[q_group] == 0.0f) ? 0.0f : fast_rcp(row_sum[q_group]);
// No PV mfma follows → packed v_pk_mul_f32 is safe. MaybeNaN pack4 scrubs residual NaN/Inf in o_acc from pathological
// inputs.
#pragma unroll
    for (int t = 0; t < HeadDimTiles; ++t) {
      o_acc[q_group][t] = o_acc[q_group][t] * inv_sum;
      const bf16x4_t out = pack4_bf16</*MaybeNaN=*/true>(
          o_acc[q_group][t][0], o_acc[q_group][t][1], o_acc[q_group][t][2], o_acc[q_group][t][3]);
      store_bf16x4(out_buffer, QOLayout(q_tile_start_row[q_group] + col, HdFrag(t, quad)) * 2, out);
    }
  }
}
// ===== V TRANSPOSE: V → V_t pre-transpose for the PV mfma

// ─── V transpose for 16x16x16 mfma ─────────────────────────────────────────────
// Pre-packs V[seq,128] into V_t for pv_phase's one contiguous load_u32x4/lane. Phase 1: V block HBM→LDS coalesced;
// phase 2: re-read in fragment order → V_t (VtLayout). super-chunk = d_tile*2 + kv_sub_pair = 64 A-frags; lane l's 16 B
// = 2 A-operands. l=frag_lane 0..63, i=0..3 = bf16 in operand:
//   bytes [l*16 .. l*16+7]   = V[k=kv_sub0*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]
//   bytes [l*16+8 .. l*16+15] = V[k=kv_sub1*16 + (l/16)*4 + i, hd=d_tile*16 + l%16]

constexpr int XposeThreadsPerWG = 256;
constexpr auto VStageWr = opus::make_layout(Shape2, strides<HeadDim / 2, 4>());                // (row, octet)→dword
constexpr auto VtChunkLane = opus::make_layout(Shape2, strides<VtChunkElems, VtLaneElems>());  // (super-chunk, lane)
template <bool kBSHD>
__global__ __launch_bounds__(XposeThreadsPerWG, 4) void v_transpose_kernel_16x16(
    const bf16_t* __restrict__ V,
    bf16_t* __restrict__ V_t,
    int seq_len_per_head,
    int heads  // #heads — only for BSHD V addressing (kBSHD)
) {
  const int num_kv_blocks = (seq_len_per_head + KvBlockRows - 1) / KvBlockRows;
  const int head = blockIdx.x / num_kv_blocks;
  const int kv_iter = blockIdx.x % num_kv_blocks;
  const int tid = threadIdx.x;
  // head = bh index over B·H; BSHD splits it into b=head/heads, h=head%heads.
  long long src_block_base;
  if constexpr (kBSHD) {
    const long long b = head / heads, h = head % heads;
    src_block_base = b * (long long)seq_len_per_head * heads * HeadDim + h * HeadDim +
                     (long long)kv_iter * KvBlockRows * heads * HeadDim;  // unpadded BSHD source base
  } else {
    src_block_base =
        (long long)head * seq_len_per_head * HeadDim + (long long)kv_iter * VtBlockElems;  // unpadded BHSD source base
  }
  const long long dst_block_base = (long long)head * num_kv_blocks * VtBlockElems + (long long)kv_iter * VtBlockElems;

  __shared__ uint32_t stage_dw[KvBlockRows * (HeadDim / 2)];  // 64 rows × 64 dw = 16 KB
  const auto v_buffer = make_buffer(V);

  // ── Phase 1: V[block] HBM → LDS, coalesced; rows ≥ seq_len_per_head zero (padded tail). Shadow QOLayout with the
  // source-V seq-row stride (phase-1 only; phase 2 reads the contiguous LDS stage via global QOLayout).
  {
    const auto QOLayout = seq_layout<kBSHD, HeadDim>(heads);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int elem_idx = k * XposeThreadsPerWG + tid;
      const int row = elem_idx >> 4;    // 0..63 kv-row in this block
      const int octet = elem_idx & 15;  // 0..15: which b128 load-unit (8 bf16 = 16 B) along the 128 head-dim
      const bool row_valid = (kv_iter * KvBlockRows + row) < seq_len_per_head;
      u32x4_t v_oct = row_valid ? __builtin_amdgcn_raw_buffer_load_b128(
                                      v_buffer, (uint32_t)((src_block_base + QOLayout(row, octet * 8)) * 2), 0, 0)
                                : u32x4_t{0, 0, 0, 0};
      const int dst_dw = VStageWr(row, octet);  // dword offset in stage_dw
      stage_dw[dst_dw + 0] = v_oct[0];
      stage_dw[dst_dw + 1] = v_oct[1];
      stage_dw[dst_dw + 2] = v_oct[2];
      stage_dw[dst_dw + 3] = v_oct[3];
    }
  }

  __syncthreads();

  // staged tile is bf16, row-major (kv_row, head_dim) — same shape as Q (QOLayout).
  const bf16_t* stage = reinterpret_cast<const bf16_t*>(stage_dw);
  const auto vt_buffer = make_buffer(V_t);

// ── Phase 2: gather each (super-chunk, lane)'s two A-fragments from LDS → V_t. ──
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int frag_idx = tid * 4 + k;     // 0..1023 = VtChunks(16) super-chunks × 64 lanes (256 threads × 4)
    const int chunk = frag_idx >> 6;      // super-chunk 0..15
    const int frag_lane = frag_idx & 63;  // lane 0..63
    const int d_tile = chunk >> 1;        // hd-tile 0..7
    const int kv_sub_pair = chunk & 1;    // 0..1
    const int kv_sub0 = kv_sub_pair * 2;
    const int kv_sub1 = kv_sub0 + 1;
    const int hd = d_tile * 16 + (frag_lane & 15);  // head-dim element 0..127
    const int kv_row0 = (frag_lane >> 4) * 4;       // hi 2 bits of lane pick a row-quad: owns kv-rows kv_row0..+3

    bf16_t packed[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      packed[i] = stage[QOLayout(kv_sub0 * 16 + kv_row0 + i, hd)];
      packed[4 + i] = stage[QOLayout(kv_sub1 * 16 + kv_row0 + i, hd)];
    }
    u32x4_t packed_v;
    __builtin_memcpy(&packed_v, packed, sizeof(packed));
    __builtin_amdgcn_raw_buffer_store_b128(
        packed_v, vt_buffer, (uint32_t)((dst_block_base + VtChunkLane(chunk, frag_lane)) * 2), 0, 0);
  }
}

// ===== DISPATCH: tail-part merge kernel + planners + launch glue

// ─── Tail-part merge kernel ──────────────────────────────────────────────────
constexpr int MergeThreads = 256;                                // threads per merge CTA
constexpr int MergeWGs = PartRows * HeadDim / 4 / MergeThreads;  // 48 CTAs per q-block's 384×128 output

// Per-(B,H) base elem offset + SRD byte span of [seq,(heads,)HeadDim] output for head-row `bh`. BHSD: heads contiguous;
// BSHD: interleaved → ×heads span, start at h·HeadDim.
template <bool kBSHD>
__device__ __forceinline__ long long head_base_offset(int bh, int seq, int heads) {
  if constexpr (kBSHD)
    return (long long)(bh / heads) * seq * heads * HeadDim + (long long)(bh % heads) * HeadDim;
  else
    return (long long)bh * seq * HeadDim;
}
template <bool kBSHD>
__device__ __forceinline__ uint32_t head_span_bytes(int seq, int heads) {
  return (uint32_t)((long long)seq * (kBSHD ? heads : 1) * HeadDim * sizeof(bf16_t));
}

// Recombine a tail q-block's `n_parts`(=G) strided KV partials. Each thread owns ONE output cell = 4 head-dims of one
// row. Online-softmax merge: run_max = maxₚ mₚ, weight wₚ = exp2(mₚ−run_max), out = (Σₚ wₚ·oₚ)/(Σₚ wₚ·lₚ), bf16-packed.
template <bool kBSHD>
__global__ __launch_bounds__(MergeThreads, 4) void merge_parts(
    bf16_t* __restrict__ Out, int seq_len, int heads, const float* __restrict__ part_buf) {
  // tail_sched (see plan_tail_split): [0,MaxParts) per-part records, [MaxParts,+tail) per-q-block meta = {q_block=.x,
  // bh=.y, part_base=.z, n_parts=.w}.
  const int4 meta = tail_sched[MaxParts + blockIdx.x / MergeWGs];
  const int part_base = meta.z, n_parts = meta.w;
  // This thread's output cell: (row, hd), hd 4-aligned.
  const int cell = ((blockIdx.x % MergeWGs) * MergeThreads + threadIdx.x) * 4;
  const int row = cell / HeadDim, hd = cell % HeadDim;
  const TailScratch parts(const_cast<float*>(part_buf));

  // Pass 1: run_max = max of the parts' softmax maxima (each m is a lower bound on the true max).
  float run_max = -FLT_MAX;
  for (int p = 0; p < n_parts; ++p)
    run_max = fmaxf(run_max, parts.ml[PartMl(part_base + p, row)]);
  // Pass 2: weighted numerators (out0..out3) + shared denom l_sum.
  float out0 = 0, out1 = 0, out2 = 0, out3 = 0, l_sum = 0;
  for (int p = 0; p < n_parts; ++p) {
    const float* ml = &parts.ml[PartMl(part_base + p, row)];   // ml[0] = m_p, ml[1] = l_p
    const float* o = &parts.o[PartO(part_base + p, row, hd)];  // o_p[hd .. hd+3]
    const float weight = fast_exp2(ml[0] - run_max);
    out0 = fmaf(weight, o[0], out0);
    out1 = fmaf(weight, o[1], out1);
    out2 = fmaf(weight, o[2], out2);
    l_sum = fmaf(weight, ml[1], l_sum);  // denom interleaved between out2/out3 deliberately:
    out3 = fmaf(weight, o[3], out3);     // splits the 4-wide FMA dependency chain for the scheduler.
  }
  const float inv_l = (l_sum == 0.0f) ? 0.0f : fast_rcp(l_sum);

  // Scatter normalized bf16x4 to Out at (q-block row, hd).
  const auto OutLayout = seq_layout<kBSHD, HeadDim>(heads);
  const auto out_buffer =
      make_buffer(Out + head_base_offset<kBSHD>(meta.y, seq_len, heads), head_span_bytes<kBSHD>(seq_len, heads));
  store_bf16x4(
      out_buffer,
      OutLayout(meta.x * PartRows + row, hd) * 2,
      pack4_bf16</*MaybeNaN=*/true>(out0 * inv_l, out1 * inv_l, out2 * inv_l, out3 * inv_l));
}

// ─── Planners ────────────────────────────────────────────────────────────────
// ─── Tail KV-split planner (dense) ───────────────────────────────────────────
// Recover the stranded fractional CU-round by KV-splitting it, uploading a schedule if it pays off. "No split"
// (tail==0, split_ways==1: whole grid runs as ordinary prefix) when not worth the merge; parts<0 signals a
// schedule-upload failure.
struct TailPlan {
  int prefix_wgs;  // CTAs run as the ordinary full-stream prefix (wgs − tail; 0 ⇒ none)
  int tail;        // stranded q-blocks recovered by the split (0 ⇒ no split at all)
  int split_ways;  // G: number of strided parts per stranded q-block
  int parts;       // total split parts = tail·G (<0 ⇒ upload error)
};
constexpr int MinKvBlocksToSplit = 48;  // below this, the merge can't amortize — don't split
constexpr int MinBlocksPerPart = 4;     // floor on kv blocks/part — finer splits let fixed cost dominate
constexpr int MergeCostBlocks = 3;      // merge kernel's modelled cost, in kv-block-time units

// ─── LiteAttention LPT (longest-processing-time) work-balance order ───────────
// Lite is work-bound: each q-block keeps a different # of K-blocks; the occ-1 grid's dynamic dispatch balances the
// bulk, leaving only ramp-down loss when a heavy q-block dispatches late. LPT runs heaviest q-blocks first → makespan ≈
// total_work/NumCUs. We only choose the ORDER (counting sort by kept-count, desc); HW dispatch IS the assignment.
// lpt_order[slot] = WG the CTA at grid slot `slot` runs. Each WG owns its own rows → bit-identical, no split/merge.
constexpr int kWorkBuckets = 256;  // kept-count buckets for the counting sort (coarse order is fine)
__global__ __launch_bounds__(256, 1) void plan_lite_order(
    const int16_t* __restrict__ read_list,
    int* __restrict__ lpt_order,
    int num_q_blocks,
    int num_kv_blocks,
    int num_bh,
    int phase) {
  const int num_wgs = num_q_blocks * num_bh;
  const int lstep = phase ? 1 : -1;
  __shared__ int bucket_count[kWorkBuckets];  // # work-groups whose work falls in each bucket
  __shared__ int bucket_slot[kWorkBuckets];   // running next free output slot per bucket
  for (int b = threadIdx.x; b < kWorkBuckets; b += blockDim.x)
    bucket_count[b] = 0;
  __syncthreads();

  // Count kept K-blocks from the WG's skip-list RLE row [len, hi0,lo0, …] (len int16s of (hi,lo) pairs), bucketize by
  // count. +lstep rebases endpoints into phase; kept/pair=|hi-lo|. e.g. (lstep=+1) row=[4,9,5,20,18] → 4+2 = kept 6.
  auto work_bucket = [&](int wg) -> int {
    const int q_block = wg % num_q_blocks, bh = wg / num_q_blocks;
    const long long row = ((long long)bh * num_q_blocks + q_block) * (num_kv_blocks + 2);
    const int len = (int)read_list[row];
    int kept = 0;
    for (int i = len; i >= 2; i -= 2) {
      const int lo = (int)read_list[row + i] + lstep;
      const int hi = (int)read_list[row + i - 1] + lstep;
      kept += (lstep > 0) ? (hi - lo) : (lo - hi);
    }
    const int bucket = kept * kWorkBuckets / (num_kv_blocks + 1);
    return bucket < kWorkBuckets ? bucket : kWorkBuckets - 1;  // clamp the top bucket
  };

  for (int wg = threadIdx.x; wg < num_wgs; wg += blockDim.x)
    atomicAdd(&bucket_count[work_bucket(wg)], 1);
  __syncthreads();
  // Exclusive scan HIGH→LOW so the heaviest buckets get the earliest grid slots (longest-first).
  if (threadIdx.x == 0) {
    int slot = 0;
    for (int b = kWorkBuckets - 1; b >= 0; --b) {
      bucket_slot[b] = slot;
      slot += bucket_count[b];
    }
  }
  __syncthreads();
  for (int wg = threadIdx.x; wg < num_wgs; wg += blockDim.x)
    lpt_order[atomicAdd(&bucket_slot[work_bucket(wg)], 1)] = wg;
}

TailPlan plan_tail_split(int nqb, int num_bh, int nkb, bool enable, hipStream_t stream) {
  const int wgs = nqb * num_bh;
  const TailPlan no_split{wgs, 0, 1, 0};
  const int tail = enable ? (wgs % NumCUs) : 0;  // CTAs left over after full CU-rounds = the stranded round
  // Single ordinary launch (no split) when: grid tiles evenly, last round ≥95% full (too little idle), or kv too short
  // for the merge to pay off.
  const bool grid_tiles_evenly = !tail;
  const bool last_round_nearly_full = tail > NumCUs * 19 / 20;
  const bool kv_too_short = nkb < MinKvBlocksToSplit;
  if (grid_tiles_evenly || last_round_nearly_full || kv_too_short) return no_split;

  // Choose G minimizing span = ceil(tail·G/NumCUs) rounds × ceil(nkb/G) blocks/part + merge cost (G=1 = whole nkb).
  // e.g. tail=40,nkb=192,NumCUs=304: G=1 → 192+3=195; G=4 → 160 parts (1 round)×48 = 48+3=51 (wins).
  int G = 1;
  long long best_span = (long long)nkb + MergeCostBlocks;
  for (int g = 2; g <= MaxParts / tail && nkb / g >= MinBlocksPerPart; ++g) {
    const long long cu_rounds = (tail * g + NumCUs - 1) / NumCUs;  // ceil(tail·g / NumCUs)
    const long long blocks_per_part = (nkb + g - 1) / g;           // ceil(nkb / g)
    const long long span = cu_rounds * blocks_per_part + MergeCostBlocks;
    if (span < best_span) {
      best_span = span;
      G = g;
    }
  }
  if (G == 1) return no_split;  // no split factor beats running the full stranded round

  // Per stranded q-block: G strided parts + one merge-meta entry. NOTE: static `sched` reused across calls —
  // correctness relies on the single-stream contract (stream-ordered H2D before the next plan overwrites it;
  // multi-stream is UB).
  static int4 sched[2 * MaxParts];
  int parts = 0;
  for (int t = 0; t < tail; ++t) {
    const int wg = (wgs - tail) + t;  // stranded CTA index, in the prefix's bh-major order
    const int q_block = wg % nqb, bh = wg / nqb;
    sched[MaxParts + t] = int4{q_block, bh, parts, G};  // per-q-block: (q_block, bh, part_base, n_parts)
    for (int i = 0; i < G; ++i)
      sched[parts + i] = int4{q_block, bh, 0, 0};  // per-part:    (q_block, bh, _, _)
    parts += G;
  }
  if (hipMemcpyToSymbolAsync(
          HIP_SYMBOL(tail_sched), sched, sizeof(int4) * (MaxParts + tail), 0, hipMemcpyHostToDevice, stream) !=
      hipSuccess)
    return TailPlan{wgs, 0, 1, -1};
  return TailPlan{wgs - tail, tail, G, parts};
}

// ─── Launch glue ─────────────────────────────────────────────────────────────
// Single launch site for the forward kernel: resolves runtime `bshd` to the kBSHD template + forwards defaulted args.
template <bool isLite, bool Split>
void fwd_launch(
    bool bshd,
    dim3 grid,
    hipStream_t stream,
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int seq_len,
    int seq_len_kv,
    float scale_log2e,
    int heads,
    const void* read_list = nullptr,
    void* write_list = nullptr,
    const void* must_do_list = nullptr,
    float thr = 0.f,
    int phase = 0,
    float* part_buf = nullptr,
    int grid_nqb = 0,
    int split_ways = 0,
    const int* lpt_order = nullptr) {
#define FWD_ARGS                                                                                                     \
  (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v, (bf16_t*)out, seq_len, seq_len_kv, scale_log2e, heads,       \
      (const int16_t*)read_list, (int16_t*)write_list, (const int16_t*)must_do_list, thr, phase, part_buf, grid_nqb, \
      split_ways, lpt_order
  if (bshd)
    attention_forward<true, isLite, Split><<<grid, dim3(BlockSize), 0, stream>>>(FWD_ARGS);
  else
    attention_forward<false, isLite, Split><<<grid, dim3(BlockSize), 0, stream>>>(FWD_ARGS);
#undef FWD_ARGS
}

// Dense tail KV-split: prefix (if non-empty) → split parts → merge, all stream-ordered.
void launch_tail_split(
    bool bshd,
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int seq_len,
    int seq_len_kv,
    float scale_log2e,
    int heads,
    int nqb,
    void* workspace,
    const TailPlan& tp,
    hipStream_t stream) {
  // Prefix: full-stream waves as one 1-D bh-major launch (grid_nqb=nqb strips the tail); empty when the whole grid is
  // the stranded round.
  if (tp.prefix_wgs)
    fwd_launch</*isLite=*/false, /*Split=*/false>(
        bshd,
        dim3(tp.prefix_wgs),
        stream,
        q,
        k,
        v,
        out,
        seq_len,
        seq_len_kv,
        scale_log2e,
        heads,
        nullptr,
        nullptr,
        nullptr,
        0.f,
        0,
        /*part_buf=*/nullptr,
        /*grid_nqb=*/nqb);
  // Split: one CTA per part, each walking every G-th kv block of its q-block into an fp32 partial.
  fwd_launch</*isLite=*/false, /*Split=*/true>(
      bshd,
      dim3(tp.parts),
      stream,
      q,
      k,
      v,
      out,
      seq_len,
      seq_len_kv,
      scale_log2e,
      heads,
      nullptr,
      nullptr,
      nullptr,
      0.f,
      0,
      /*part_buf=*/(float*)workspace,
      /*grid_nqb=*/nqb,
      /*split_ways=*/tp.split_ways);
  // Merge: recombine each stranded q-block's parts into the final bf16 output.
  if (bshd)
    merge_parts<true><<<dim3(tp.tail * MergeWGs), dim3(MergeThreads), 0, stream>>>(
        (bf16_t*)out, seq_len, heads, (const float*)workspace);
  else
    merge_parts<false><<<dim3(tp.tail * MergeWGs), dim3(MergeThreads), 0, stream>>>(
        (bf16_t*)out, seq_len, heads, (const float*)workspace);
}

// Lite full-grid launch with LPT reordering: grid slot i processes lpt_order[i]. Pure grid-slot→WG reorder →
// bit-identical, tighter makespan. `order_buf` = num_wgs ints (the permutation).
void launch_lite_ordered(
    bool bshd,
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int seq_len,
    int seq_len_kv,
    float scale_log2e,
    int heads,
    int nqb,
    int bh,
    int nkb,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    void* order_buf,
    hipStream_t stream) {
  int* lpt_order = (int*)order_buf;
  plan_lite_order<<<1, dim3(256), 0, stream>>>((const int16_t*)read_list, lpt_order, nqb, nkb, bh, phase);
  fwd_launch</*isLite=*/true, /*Split=*/false>(
      bshd,
      dim3(nqb * bh),
      stream,
      q,
      k,
      v,
      out,
      seq_len,
      seq_len_kv,
      scale_log2e,
      heads,
      read_list,
      write_list,
      must_do_list,
      threshold,
      phase,
      /*part_buf=*/nullptr,
      /*grid_nqb=*/nqb,
      /*split_ways=*/0,
      lpt_order);
}

}  // namespace

extern "C" {
// IMPORTANT: V here must point to pre-transposed V_t (call launch_v_transpose first; V_t and V have identical byte
// size). Tail KV-split workspace (dense only): fp32 o + (m,l) for MaxParts parts. workspace==nullptr ⇒ no split.
size_t tail_workspace_bytes(void) {
  return (size_t)MaxParts * (PartOFloats + PartMlFloats) * sizeof(float);
}

// Lite LPT order buffer: one int per work-group (q-block × batch·head). workspace==nullptr ⇒ plain full-grid lite.
size_t lite_workspace_bytes(int batch, int heads, int seq_len) {
  const int qt = (seq_len + QTileRows - 1) / QTileRows;
  const int nqb = (qt + QTilesPerWG - 1) / QTilesPerWG;
  return (size_t)batch * heads * nqb * sizeof(int);
}

int launch_attention_forward(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* workspace,
    void* stream) {
  if (batch <= 0 || heads <= 0) return 1;
  if (seq_len <= 0) return 2;     // any Q len (partial last q-tile SRD-padded)
  if (seq_len_kv <= 0) return 5;  // any K/V len (partial last block masked)
  if (head_dim != HeadDim) return 3;

  const int bh = batch * heads;
  const int qt = (seq_len + QTileRows - 1) / QTileRows;
  const int nqb = (qt + QTilesPerWG - 1) / QTilesPerWG, nkb = (seq_len_kv + 63) / 64;
  const dim3 grid(nqb, bh);
  const float scale_log2e = Log2e / sqrtf((float)HeadDim);
  // layout: 0 = BHSD ([B,H,S,D]); 1 = BSHD ([B,S,H,D], read natively, no transpose).

  // Tail KV-split: recover the stranded fractional round (when a workspace is given). Prefix = full waves as one
  // ordinary launch; split parts + merge run the stranded round.
  const TailPlan tp = plan_tail_split(nqb, bh, nkb, workspace != nullptr, (hipStream_t)stream);
  if (tp.parts < 0) return 6;
  if (tp.tail) {
    launch_tail_split(
        layout, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads, nqb, workspace, tp, (hipStream_t)stream);
    return (int)hipGetLastError();
  }
  fwd_launch</*isLite=*/false, /*Split=*/false>(
      layout, grid, (hipStream_t)stream, q, k, v, out, seq_len, seq_len_kv, scale_log2e, heads);
  return (int)hipGetLastError();
}

// LiteAttention forward (Part A cross-timestep block-skip + Part B per-wave soft-skip). read/write_list = int16 RLE
// keep-lists [B,H,nqb,nkb+2] (read=skip_list[phase], write=[1-phase]); must_do_list = optional force-compute RLE
// (nullable); threshold log2 (~-10), phase = reverse dir, V pre-transposed. Verdict ring reuses evicted qg2 LDS slot →
// total LDS == dense's. Full-grid launch, each CTA writes its RLE inline; workspace (LPT buffer) runs q-blocks
// heaviest-first (bit-identical), else plain identity-order.
int launch_attention_forward_lite(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* workspace,
    void* stream) {
  if (batch <= 0 || heads <= 0) return 1;
  if (seq_len <= 0) return 2;     // any Q len (partial last q-tile masked by valid[])
  if (seq_len_kv <= 0) return 5;  // any K/V len (partial last block masked in softmax)
  if (head_dim != HeadDim) return 3;
  if (!read_list || !write_list) return 4;  // must_do_list is optional

  const int bh = batch * heads;
  const int qt = (seq_len + QTileRows - 1) / QTileRows;
  const int nqb = (qt + QTilesPerWG - 1) / QTilesPerWG, nkb = (seq_len_kv + 63) / 64;
  const dim3 grid(nqb, bh);
  const float scale_log2e = Log2e / sqrtf((float)HeadDim);

  // LPT work-balance reorder (default when a workspace is given and kv is long enough to amortize the ~few-µs sort):
  // run heaviest q-blocks first so HW dynamic dispatch finishes near the work-bound floor. Pure reorder →
  // bit-identical.
  if (workspace && nkb >= MinKvBlocksToSplit) {
    launch_lite_ordered(
        layout,
        q,
        k,
        v,
        out,
        seq_len,
        seq_len_kv,
        scale_log2e,
        heads,
        nqb,
        bh,
        nkb,
        read_list,
        write_list,
        must_do_list,
        threshold,
        phase,
        workspace,
        (hipStream_t)stream);
    return (int)hipGetLastError();
  }

  // Lite runs the full grid as one launch (no tail KV-split — stranded round is sparse); each CTA walks its kept blocks
  // and RLE-encodes the write list inline.
  fwd_launch</*isLite=*/true, /*Split=*/false>(
      layout,
      grid,
      (hipStream_t)stream,
      q,
      k,
      v,
      out,
      seq_len,
      seq_len_kv,
      scale_log2e,
      heads,
      read_list,
      write_list,
      must_do_list,
      threshold,
      phase);
  return (int)hipGetLastError();
}

// V transpose: V[seq_len,128] (bf16) → V_t in pv_phase's layout. Caller allocates V_t same byte size as V; n_rows =
// batch*heads*seq_len.
int launch_v_transpose(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream) {
  if (n_rows <= 0 || seq_len_per_head <= 0 || (n_rows % seq_len_per_head)) return 2;
  const int bh = n_rows / seq_len_per_head;
  const int num_kv_blocks = (seq_len_per_head + KvBlockRows - 1) / KvBlockRows;
  const int iters = bh * num_kv_blocks;
  // heads/layout select the source V layout: 0 = BHSD ([B,H,S,D]); 1 = BSHD ([B,S,H,D], strided).
  if (layout)
    hipLaunchKernelGGL(
        v_transpose_kernel_16x16<true>,
        dim3(iters),
        dim3(XposeThreadsPerWG),
        0,
        (hipStream_t)stream,
        (const bf16_t*)v,
        (bf16_t*)v_t,
        seq_len_per_head,
        heads);
  else
    hipLaunchKernelGGL(
        v_transpose_kernel_16x16<false>,
        dim3(iters),
        dim3(XposeThreadsPerWG),
        0,
        (hipStream_t)stream,
        (const bf16_t*)v,
        (bf16_t*)v_t,
        seq_len_per_head,
        heads);
  return (int)hipGetLastError();
}
}
