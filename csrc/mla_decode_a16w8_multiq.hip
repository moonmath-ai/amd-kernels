// CDNA3 (gfx942 / MI300X) absorbed-decode MLA over a multi-query draft window.
//
//     score[t,h,n] = (q_lat[t,h,:] . c_kv[n,0:512] + q_pe[t,h,:] . c_kv[n,512:576]) * scale
//     o_lat[t,h,:] = sum_n softmax_n(score)[t,h,n] * c_kv[n,0:512]
//
// W_UK / W_UV are absorbed outside the kernel, so one c_kv row serves as both K and V: a fused fp8
// e4m3-fnuz 576-wide row ([0:512] latent, [512:576] rope) at one per-tensor kv_scale. Q stays bf16,
// so every MFMA is v_mfma_f32_16x16x16bf16_1k and the fp8 KV is unpacked once per token by the LDS
// fill, not per MFMA operand. Query row (t,h) is head h of draft position t and attends to
// KV [0, S-q_len+t]. Tokens are split across CTAs FlashDecoding style and the decode kernel ALWAYS
// writes partials, which the merge kernel below reduces.
//
// SUPPORTED DOMAIN, enforced host-side by check_domain(). Outside it the launchers return an error
// rather than mis-plan, because the code those cases would need is not present:
//   H      1..16    a request is one MFMA N-tile of heads; there is no head-tile axis in the grid
//   q_len  4..8     compile-time width QL is 4 or 8 and a runtime q_len < QL masks the rest
//   B      1..32    and with it B*QGroups <= NumCU/2, so the grid is never quantisation-bound
//   S      >= 2048  the tuned range; shorter sequences stay correct
//   paged  primary (`Paged` picks a [num_slots,1,576] pool); a [B,S,576] slab is the second path
//
// q_len 1 is a different CTA shape and is served by mla_decode_a16w8.hip.
//
// CTA SHAPE. The PV accumulator sets the register budget: q_pos * 16 * 512 dims * 4 B per CTA, the
// 16 being the MFMA N dimension, not the head count. DraftPerCta = 4 fits at 8 waves and one CTA per
// CU only because o_acc is split by OUTPUT DIMENSION: 64 dims of every N-tile per wave.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

namespace {
// ── Types ──
using bf16   = __bf16;
using bf16x4 = __attribute__((ext_vector_type(4))) bf16;
using bf16x8 = __attribute__((ext_vector_type(8))) bf16;
using f32x2  = __attribute__((ext_vector_type(2))) float;
using f32x4  = __attribute__((ext_vector_type(4))) float;
using u32x2  = __attribute__((ext_vector_type(2))) uint32_t;
using fp8_t  = __hip_fp8_e4m3_fnuz;
// A compile-time loop index, which `#pragma unroll` does not provide: the paged slot pick needs it.
template <int V> struct IC { static constexpr int value = V; };
template <int I, int N, class F> __device__ __forceinline__ void static_for(F&& f) {
  if constexpr (I < N) { f(IC<I>{}); static_for<I + 1, N>(f); }
}

// ── Problem constants ──
constexpr int WaveSize = 64, MfmaTile = 16, QuadSpan = 4;   // wave; MFMA M==N; C-fragment quad
constexpr int LatDim = 512, RopeDim = 64;                   // kv_lora_rank, qk_rope_head_dim
constexpr int FusedDim = LatDim + RopeDim;                  // 576 fp8 bytes of KV per token
constexpr int KQ = 16, QLane = KQ / QuadSpan;    // MFMA contraction length, and one lane's share
constexpr int LatIters = LatDim / KQ, RopeIters = RopeDim / KQ;   // 32 latent + 4 rope K-steps
constexpr int QkSteps  = LatIters + RopeIters;   // 36 K-steps of one full score contraction
constexpr int PvChunks = LatDim / MfmaTile;      // 32 PV output-dim tiles
constexpr float NegInf = -3.0e38f;
constexpr int MaxQL = 8, MaxHeads = 16, MaxBatch = 32;   // the SUPPORTED DOMAIN, as constants
constexpr int DraftPerCta = 4;                   // draft positions resident per CTA
constexpr int NumWaves = 8, CtaThreads = NumWaves * WaveSize;    // one 8-wave CTA per CU
constexpr int NumCU = 304;                       // MI300X multiProcessorCount
constexpr int qgroups_of(int ql) { return ql / DraftPerCta; }    // CTAs per (request, kv-part)
// Probabilities in (0,1] are lifted by ProbLift before the bf16 convert and divided out in the
// epilogue. Exact: it is a power of two.
constexpr int   Log2ProbLift = 7;
constexpr float ProbLiftLog2 = (float)Log2ProbLift, ProbLift = (float)(1 << Log2ProbLift);

// ── Data movement and packing primitives ──
__device__ __forceinline__ f32x4 mfma_bf16(bf16x4 a, bf16x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
// gfx942 has no v_cvt_bf16_f32: f32 -> bf16 is done on the bit pattern and no mode handles NaN.
enum Round { Trunc, Rtne, Rna };
template <Round R> __device__ __forceinline__ unsigned narrow_bf16(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return R == Trunc ? u : R == Rna ? u + 0x8000u : u + 0x7FFFu + ((u >> 16) & 1u);
}
template <Round R> __device__ __forceinline__ bf16x4 pack4(float a, float b, float c, float d) {
  const unsigned w0 = __builtin_amdgcn_perm(narrow_bf16<R>(b), narrow_bf16<R>(a), 0x07060302u);
  const unsigned w1 = __builtin_amdgcn_perm(narrow_bf16<R>(d), narrow_bf16<R>(c), 0x07060302u);
  bf16x4 o; __builtin_memcpy(&o, &w0, 4); __builtin_memcpy((char*)&o + 4, &w1, 4); return o;
}
// One dword = 4 fp8 dims of one token -> the two raw dwords a ds_write wants. Must route THROUGH a
// bf16x4: writing the cvt_pk/perm pair by hand over `f32x2` subscripts miscompiles.
__device__ __forceinline__ u32x2 cvt4_bf16(uint32_t w) {
  const f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false);
  const f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
  const bf16x4 v = pack4<Trunc>(lo[0], lo[1], hi[0], hi[1]);
  u32x2 o; __builtin_memcpy(&o, &v, 8); return o;
}
__device__ __forceinline__ bf16x4 as_bf16x4(uint32_t lo, uint32_t hi) {
  const u32x2 w = {lo, hi};  bf16x4 o; __builtin_memcpy(&o, &w, 8); return o;
}
__device__ __forceinline__ bf16x4 as_bf16x4(u32x2 w) { return as_bf16x4(w[0], w[1]); }
__device__ __forceinline__ void transpose_4tok(const uint32_t* in, uint32_t* out) {
  out[0] = __builtin_amdgcn_perm(in[1], in[0], 0x05040100u);   // dim 2a,   tokens 0,1
  out[1] = __builtin_amdgcn_perm(in[3], in[2], 0x05040100u);   // dim 2a,   tokens 2,3
  out[2] = __builtin_amdgcn_perm(in[1], in[0], 0x07060302u);   // dim 2a+1, tokens 0,1
  out[3] = __builtin_amdgcn_perm(in[3], in[2], 0x07060302u);   // dim 2a+1, tokens 2,3
}
__device__ __forceinline__ bf16x8 pack8(float a, float b, float c, float d,
                                        float e, float f, float g, float h) {
  const bf16x4 lo = pack4<Rtne>(a, b, c, d), hi = pack4<Rtne>(e, f, g, h);
  bf16x8 o;  __builtin_memcpy(&o, &lo, 8);  __builtin_memcpy((char*)&o + 8, &hi, 8);  return o;
}
__device__ __forceinline__ float mul_f32(float a, float b) {
  float r; asm volatile("v_mul_f32_e32 %0,%1,%2" : "=v"(r) : "v"(a), "v"(b)); return r;
}

// ── Synchronisation and cross-lane primitives ──
// Raw s_barrier as asm, not the IntrNoMem intrinsic, so LDS accesses are not hoisted across it.
__device__ __forceinline__ void barrier() { asm volatile("s_barrier" ::: "memory"); }
// s_barrier drains no counter. A ds_write that must be VISIBLE to a peer wave after it needs this.
__device__ __forceinline__ void lds_drain() { __builtin_amdgcn_s_waitcnt(0xC07F); }
// Per-lane select against a lane mask. The mask MUST come from a ballot: given a constant such as
// `~0ULL << L`, LLVM mis-materialises the SGPR pair and mis-selects lanes 32..63.
__device__ __forceinline__ int sel_by_mask(int a, int b, uint64_t m) {
  int r;  asm("v_cndmask_b32_e64 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "s"(m));  return r;
}
#define LANE_GE(L) ((uint64_t)__builtin_amdgcn_ballot_w64((int)(threadIdx.x & (WaveSize - 1)) >= (L)))
// Cross-lane exchange over the MFMA-C quad axis. XOR by 32 as +128 folds into the bpermute offset:
// bpermute wraps mod 64, so (l+32)%64 == l^32 and no register holds a materialised l^32.
__device__ inline float lane_xor16(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F));
}
__device__ inline float lane_swap32(float v, int l) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((l << 2) + 128, __builtin_bit_cast(int, v)));
}
__device__ inline float lane_xor(float v, int m) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((__lane_id() ^ m) << 2, __builtin_bit_cast(int, v)));
}
__device__ inline float wave_max(float v) { const int l = __lane_id(); v = fmaxf(v, lane_xor16(v)); return fmaxf(v, lane_swap32(v, l)); }
__device__ inline float wave_sum(float v) { const int l = __lane_id(); v += lane_xor16(v); return v + lane_swap32(v, l); }
__device__ __forceinline__ int lane_xor_i(int v, int m) {
  return __builtin_amdgcn_ds_bpermute((__lane_id() ^ m) << 2, v);
}
__device__ __forceinline__ long long ll_join(unsigned lo, unsigned hi) { return (long long)(((unsigned long long)hi << 32) | lo); }
__device__ __forceinline__ long long lane_xor_ll(long long v, int m) {
  return ll_join(lane_xor_i((int)(unsigned)v, m), lane_xor_i((int)(unsigned)((unsigned long long)v >> 32), m));
}
__device__ __forceinline__ long long readfirstlane_ll(long long v) {
  return ll_join(__builtin_amdgcn_readfirstlane((int)(unsigned)v),
                 __builtin_amdgcn_readfirstlane((int)(unsigned)((unsigned long long)v >> 32)));
}
__device__ __forceinline__ int wave_sum_i(int v) {
  for (int m = 1; m < WaveSize; m <<= 1) v += lane_xor_i(v, m);
  return __builtin_amdgcn_readfirstlane(v);
}

// ── LDS layout ──
// KV tile, token-major rows of 576 bf16 plus 8 B of pad: KvRowBytes/4 = 290 == 2 (mod 32 banks), so
// a slot's 16 rows land on 16 distinct bank PAIRS and a QK read covers all 32 banks once. The slot
// stride is 0 (mod 32 dwords), so both slots bank alike and no token swizzle is needed.
constexpr int TileTokens  = 16, TileSlots = 2;            // 16 KV tokens; read tile h, fill h+1
constexpr int KvRowBytes  = FusedDim * (int)sizeof(bf16) + 8;         // 1160
constexpr int KvRowElems  = KvRowBytes / (int)sizeof(bf16);           // 580
constexpr int KvSlotElems = TileTokens * KvRowElems, KvLdsElems = TileSlots * KvSlotElems;
constexpr int PvTokensPerLane = TileTokens / QuadSpan;    // tokens a lane reads per PV group
static_assert(KvRowBytes % 8 == 0, "ds_read_b64 / ds_write2_b64 need 8-B aligned rows");
static_assert(PvTokensPerLane == QuadSpan, "PV needs 4 tokens per lane for the register transpose");
// K-split rendezvous planes: partial QK scores exchanged between the waves sharing an N-tile. File
// scope because the CTA body takes them as a parameter -- a __shared__ inside it would be allocated
// once per instantiation and the packed shape inlines the body twice. One plane per wave per parity
// plus (packed only) a zero plane, so 2-way and 3-way K-splits share one collect sequence.
constexpr int RvStride = MfmaTile * QuadSpan * QuadSpan + 8;
constexpr int rv_planes(int tiles) { return tiles == 3 ? NumWaves + 1 : 1; }
// Cross-wave probability planes and the per-(row, tile) lazy-rescale factors. The rescale factor is
// PER QUERY ROW, laid out [parity][row][tile] so a consumer reads all of ITS row's in one b128.
constexpr int MaxNTiles         = 4;                          // NTiles is 3 (packed) or 4
constexpr int ProbPlaneElems    = 2 * MaxNTiles * WaveSize;   // parity x tile x lane, bf16x4
constexpr int RescalePlaneElems = 2 * MfmaTile;               // parity x row, one f32x4 of tiles

// ── Head packing ──
// The MFMA N dimension is 16 INDEPENDENT query rows, which need not share a draft position: at H in
// [9,12] the unpacked map wastes 16-H of 16, while 4 positions x 12 heads = 3 full N-tiles. Packed
// rows are POSITION-MAJOR (r = t*H + h), keeping q and o linear in r.
__host__ __device__ inline bool use_head_packing(int H) { return H >= 9 && H <= 12; }

// ── Partial workspace layout ──
// A bf16 [cta][slot][512] payload slab then slot-major (run_max, run_sum) fp32 header planes; `slot`
// is (q position in the CTA) * H + head. The merge grid is sized by the LIVE slots, but the slab
// STRIDE stays SlabSlots -- shrinking it would move every partials-store address.
constexpr int SlabSlots       = MaxHeads * DraftPerCta;   // slab STRIDE: 4 positions x 16 head slots
constexpr int PackedSlabSlots = 3 * MfmaTile;             // LIVE slots when the rows are packed
static_assert(PackedSlabSlots == 12 * DraftPerCta, "packed live slots must be the H=12 row block");
__host__ __device__ __forceinline__ int live_slots(int packH) {
  return packH ? PackedSlabSlots : SlabSlots;
}
__device__ __host__ __forceinline__ size_t hdr_off(size_t nctas, int slots, int slot, int k) {
  return ((((size_t)nctas * slots * LatDim / 2) + 3) & ~(size_t)3) + ((size_t)slot * 2 + k) * nctas;
}

// ── KV split plan ──
// A uniform `parts` balances only at uniform S, so the same B*parts flat units are instead handed
// out in proportion to each request's tiles: one each, the rest by floor(). The floors sum to at
// most U-B, so the plan can never overrun the grid -- no second pass, no atomic, no host round trip,
// which matters because seq_lens is on the DEVICE and the launch is graph-captured.
//   INVARIANT: decode and merge rederive this plan INDEPENDENTLY, so every reduction here is exact
// integer arithmetic whose value is order-independent.
constexpr int MaxParts          = 384;
constexpr int MaxTilesPerRequest = 1 << 20;        // clamps one request's tiles (S <= 16.7M)
__device__ __host__ __forceinline__ int kv_tiles(int S) {
  const int t = (S + TileTokens - 1) / TileTokens;
  return t < 1 ? 1 : (t > MaxTilesPerRequest ? MaxTilesPerRequest : t);
}
// One sweep of seq_lens; the `same` fast path keeps the merge, which reruns this, off B divisions.
struct SplitPlan {
  long long tot; int t0; bool same;
  __device__ __forceinline__ int units(const int* __restrict__ sl, int i, int B, int parts) const {
    if (same) return t0 < parts ? t0 : parts;
    const long long den = tot - B;                 // sum over requests of (tiles-1); >= 0
    const int U = B * parts, t = kv_tiles(sl[i]);
    long long n = 1;
    if (den > 0 && U > B) n += ((long long)(t - 1) * (U - B)) / den;
    if (n > t) n = t;                              // never split a request past its own tiles
    return (int)(n > MaxParts ? MaxParts : n);     // s_w[MaxParts] in the merge bounds this
  }
};
constexpr int SerialScanMaxBatch = 8;
// Lane-parallel sweep of seq_lens: ceil(B/64) INDEPENDENT loads plus a 6-step butterfly, with
// readfirstlane handing the result back in SGPRs so `b`, `part` and `nparts` stay scalar. The
// butterfly is six DEPENDENT round trips whatever B is, hence the crossover below.
__device__ __forceinline__ SplitPlan scan_split_plan(const int* __restrict__ sl, int B) {
  const int t0 = kv_tiles(sl[0]);                  // uniform address: still one s_load
  if (B <= SerialScanMaxBatch) {
    SplitPlan p{0, t0, true};
    for (int i = 0; i < B; ++i) { const int t = kv_tiles(sl[i]); p.tot += t; p.same &= (t == p.t0); }
    return p;
  }
  long long tot = 0;
  int same = 1;
  for (int i = (int)__lane_id(); i < B; i += WaveSize) {
    const int t = kv_tiles(sl[i]);
    tot += t;
    same &= (t == t0);
  }
  for (int m = 1; m < WaveSize; m <<= 1) {         // both butterflies in ONE loop: 6 round trips
    tot  += lane_xor_ll(tot, m);
    same &= lane_xor_i(same, m);
  }
  return SplitPlan{readfirstlane_ll(tot), t0, __builtin_amdgcn_readfirstlane(same) != 0};
}
__device__ __forceinline__ int locate_unit(const SplitPlan& pl, const int* __restrict__ sl, int B,
                                           int parts, int unit, int& off, int& n) {
  const int l = (int)__lane_id();
  off = 0;
  n = 0;
  for (int base = 0; base < B; base += WaveSize) {
    const int idx = base + l;
    const int u   = idx < B ? pl.units(sl, idx, B, parts) : 0;
    int pre = u;                                   // inclusive prefix within the wave
    for (int m = 1; m < WaveSize; m <<= 1) {
      const int v = __builtin_amdgcn_ds_bpermute(((l - m) & (WaveSize - 1)) << 2, pre);
      pre += l >= m ? v : 0;                       // lanes below m read a wrapped lane; discarded
    }
    const unsigned long long hit = __ballot(pre > unit - off);
    if (hit) {
      const int L = (int)__builtin_ctzll(hit);
      n    = __builtin_amdgcn_readlane(u, L);
      off += __builtin_amdgcn_readlane(pre - u, L);
      return base + L;
    }
    off += __builtin_amdgcn_readlane(pre, WaveSize - 1);
  }
  return B;                                        // surplus unit: no request owns it
}

// ── The decode CTA ──
// PIPELINE. All eight waves fill AND compute; there are no dedicated producer waves. The tile is 16
// KV tokens and the loop is software pipelined one tile deep, so a tile's softmax runs in the shadow
// of the next tile's LDS reads. ONE barrier per tile, separating (reads of tile h, plane set h) from
// (the fill of tile h+2, plane set h), so two slots and two plane sets cover it.
//   PV COLLAPSE. The PV LDS read for a dim slice does not depend on the N-tile, so one fetch feeds
// every tile's MFMA and each wave owns a 64-dim output slice of EVERY N-tile. What crosses waves
// instead is the probability matrix, published by one designated wave per tile -- and it must cross
// WITHOUT a second barrier, so probs(h-1) published at step h are consumed at step h+1.
constexpr float RescaleSlack = 4.0f;               // see softmax_tile
// WAVE MAP (packed, NTiles == 3). Three N-tiles do not divide eight waves, and the uneven share is
// aimed at the barrier skew: the slot-0 waves (wave/4 == 0) get LDS service first, so they take the
// HEAVY role and slot 1 takes the LIGHT one -- half the K-steps, half the PV chunks, and the fill.
//
//   wave    0    1    2    3  |  4    5    6    7      role   KSteps
//   tile    0    0    1    2  |  1    1    2    2      heavy   18   (waves 0-3, slot 0)
//   kstep   0   18    0    0  | 18   27   18   27      light    9   (waves 4-7, slot 1)
//
// At NTiles == 4 waves 0-3 own one N-tile each and its whole 36-step contraction, 4-7 fill and PV.
template <int QL, bool Paged, bool NonTemporal, int NTiles, int KSteps>
__device__ __forceinline__ void decode_body(
    const bf16* __restrict__ q_lat, const bf16* __restrict__ q_pe, const fp8_t* __restrict__ c_kv,
    float* __restrict__ partials, int B, int H, int S_scalar, int parts, int q_len,
    float scale_log2e, float kv_scale, const int* __restrict__ seq_lens,
    const int* __restrict__ rows, const int* __restrict__ kv_indices,
    const int* __restrict__ kv_indptr, bf16* __restrict__ sKv,
    float (*__restrict__ sQkPeer)[RvStride], bf16x4* __restrict__ sProb,
    f32x4* __restrict__ sRescale) {
  // PV ownership: o_acc[t*PvSlots + a][e] is dim pv_base + PvSlots*(4*quad + e) + a of N-tile t.
  constexpr int PvSlots = PvChunks / NumWaves;      // 4 output-dim slots per wave per tile
  constexpr int PvAccs = PvSlots * NTiles, PvDwords = PvSlots / 2;   // accumulators; dwords/token
  constexpr int QGroups = QL / DraftPerCta;         // CTAs per (request, kv-part)
  constexpr bool Packed = (NTiles == 3);
  constexpr int NumPeers = Packed ? 2 : 0;          // unpacked: no K-split, no rendezvous
  constexpr int RvSetStride = rv_planes(NTiles);    // stride between the two parity plane sets
  static_assert(DraftPerCta * QGroups == QL, "the CTA rows must split the waves evenly");
  static_assert(!Packed || KSteps == 18 || 18 + KSteps + KSteps == QkSteps, "packed K-split must cover 36");
  static_assert(Packed || KSteps == 0 || KSteps == QkSteps, "unpacked: a QK wave owns all 36 K-steps or none");

  // ── Grid mapping ──
  // The QGroups q groups of a request share one KV stream, and MI300X round-robins workgroups over
  // 8 XCDs, so place the sharers XCDs apart or their duplicate reads miss every L2.
  constexpr int XCDs = 8;
  const int cta = blockIdx.x;
  int qgrp = 0, unit = cta;
  if constexpr (QGroups > 1) {
    constexpr int blk = XCDs * QGroups;
    const int nfull = (int)(gridDim.x / blk) * blk;
    if (cta < nfull) {
      const int r = cta / XCDs;
      qgrp = r % QGroups;
      unit = (r / QGroups) * XCDs + (cta % XCDs);
    } else {
      const int d = cta - nfull;
      qgrp = d % QGroups;
      unit = nfull / QGroups + d / QGroups;
    }
  }
  int part, b, nparts = parts;
  if constexpr (Paged) {
    const SplitPlan pl = scan_split_plan(seq_lens, B);
    int off = 0, i = 0, n = pl.units(seq_lens, 0, B, parts);
    if (pl.same) { i = unit / n; off = i * n; }
    else i = locate_unit(pl, seq_lens, B, parts, unit, off, n);
    if (i >= B) return;            // surplus unit: the quota plan asked for fewer CTAs than the grid
    b = i; part = unit - off; nparts = n;
  } else {
    part = unit % parts; b = unit / parts;
  }
  const int slab_cta = unit * QGroups + qgrp;      // LOGICAL (unit, qgrp): what the merge walks
  const int S        = Paged ? seq_lens[b] : S_scalar;
  // `rows[b]` is the q/out batch row for KV-request b. It reaches a STORE, so it is clamped.
  const int q_row = (Paged && rows) ? min(max(rows[b], 0), B - 1) : b;
  const int slot0 = Paged ? kv_indptr[b] : 0;

  // ── Wave role ──
  const int lane = threadIdx.x & (WaveSize - 1), wave = threadIdx.x / WaveSize;
  const int mrow = lane & (MfmaTile - 1), quad = lane / MfmaTile;
  int tile, kstep_lo, qk_peer0, qk_peer1 = 0;
  bool publishes;                                  // this wave writes the tile's (run_max, run_sum)
  if constexpr (!Packed) {
    tile = (KSteps > 0) ? wave : 0; kstep_lo = 0; qk_peer0 = 0;
    publishes = (KSteps > 0);
  } else if constexpr (KSteps == 18) {              // heavy: waves 0-3
    tile     = (wave <= 1) ? 0 : (wave - 1);
    kstep_lo = (wave == 1) ? 18 : 0;
    qk_peer0 = (wave <= 1) ? (1 - wave) : (2 * wave);
    qk_peer1 = (wave <= 1) ? NumWaves : (2 * wave + 1);   // tile 0 is 2-way: pad with the zero plane
    publishes = (wave != 1);
  } else {                                          // light: waves 4-7
    tile = (wave < 6) ? 1 : 2; kstep_lo = (wave & 1) ? 27 : 18;
    qk_peer0 = (wave < 6) ? 2 : 3; qk_peer1 = wave ^ 1;
    publishes = false;                              // the light role never publishes
  }
  const int qk_plane = wave, qk_slot = (quad * MfmaTile + mrow) * QuadSpan;

  // ── This CTA's KV part ──
  const int part_tokens = ((S + nparts * TileTokens - 1) / (nparts * TileTokens)) * TileTokens;
  const int kv_lo  = part * part_tokens;
  const int kv_hi  = min(S, kv_lo + part_tokens);
  const int ntiles = (kv_hi - kv_lo + TileTokens - 1) / TileTokens;

  // ── Query rows ──
  // Unpacked: the tile IS the draft position and the row is head mrow. Packed: the row is
  // prow = tile*16 + mrow in the CTA's 4*H rows, and position*H + head == prow, so no divide.
  const int  qpos_g = qgrp * DraftPerCta + tile;    // (unpacked) draft position within the request
  const int  nact   = min(DraftPerCta, max(0, q_len - qgrp * DraftPerCta));  // positions this CTA owns
  const int  prow   = tile * MfmaTile + mrow;       // (packed) row within the CTA's block
  const bool active = Packed ? (prow < H * nact) : (mrow < H && qpos_g < q_len);
  // Causal inside the draft window: position p sees KV [0, S - (q_len-1-p)). Per lane once packed.
  const int tok_lim = Packed
      ? (active ? (S - (q_len - 1 - (qgrp * DraftPerCta + prow / (H > 0 ? H : 1)))) : S)
      : ((qpos_g < q_len) ? (S - (q_len - 1 - qpos_g)) : 0);
  bf16x4 q_frag[KSteps ? KSteps : 1] = {};
  {
    const size_t qh = Packed ? ((size_t)(q_row * q_len + qgrp * DraftPerCta) * H + (active ? prow : 0))
                             : (size_t)((q_row * q_len + (active ? qpos_g : 0)) * H + (active ? mrow : 0));
    const bf16* ql = q_lat + qh * LatDim;
    const bf16* qp = q_pe  + qh * RopeDim;
    const bf16x4 zero = {};
    #pragma unroll
    for (int k = 0; k < KSteps; ++k) {
      const int kg = kstep_lo + k;
      const bf16* src = (kg < LatIters) ? (ql + (size_t)kg * KQ) : (qp + (size_t)(kg - LatIters) * KQ);
      const bf16x4 f = *(const bf16x4*)(src + quad * QLane);
      q_frag[k] = active ? f : zero;
    }
  }
  const float qk_descale = kv_scale * scale_log2e;  // no q_scale: q was never quantized
  const int   mask_end = min(kv_hi, tok_lim);      // last KV token this lane's row may see
  // Loop-invariant softmax bias: the lift, or a hard floor if this row's whole part is masked out.
  const float exp_bias = (kv_lo >= mask_end) ? NegInf : ProbLiftLog2;
  float run_max = NegInf, run_sum = 0.f;
  f32x4 o_acc[PvAccs] = {};
  const int pv_base      = wave * (PvSlots * MfmaTile);          // this wave's 64 output dims
  const int pv_col_bytes = (pv_base + PvSlots * mrow) * (int)sizeof(bf16);

  // ── Tile fill ──
  // 8 B in / 16 B out keeps the ds_write lane stride at 4 dwords, so 8 lanes cover all 32 banks. The
  // fill belongs to the CTA, not a query tile, so all of it goes to the light-QK waves.
  constexpr bool DoFill = Packed ? (KSteps != 18) : (KSteps == 0);
  constexpr int FillTokensPerWave = DoFill ? (TileTokens / 4) : 0;   // 4 filling waves either way
  constexpr int FillBytesPerWave = FillTokensPerWave * FusedDim;
  constexpr int FillWideOps = FillBytesPerWave / (WaveSize * 8);     // whole 64-lane x 8-B loads
  constexpr int FillOpsArray = FillWideOps > 0 ? FillWideOps : 1;    // array sizing only
  constexpr int FillTailOffset = FillWideOps * WaveSize * 8;
  static_assert(FusedDim % 8 == 0, "an 8-B chunk must never straddle a token row");
  static_assert(!DoFill || FillTailOffset + WaveSize * 4 == FillBytesPerWave,
                "the fill tail must be exactly one 4-B op per lane");
  struct FillRegs { u32x2 w[FillOpsArray]; uint32_t tail; };
  const int fillw = wave & 3, tokbase = fillw * FillTokensPerWave;   // waves 4-7 fill on both paths
  // Paged slot ids are prefetched a phase before use, as scalar loads (lgkmcnt, never vmcnt).
  const int fwv = __builtin_amdgcn_readfirstlane(fillw);
  auto load_slots = [&](int iter_, int* dst) {
    if constexpr (Paged && DoFill) {
      const int it   = iter_ < ntiles ? iter_ : (ntiles > 0 ? ntiles - 1 : 0);
      const int kv0  = kv_lo + it * TileTokens;
      const int tc   = min(TileTokens, kv_hi - kv0);
      const int last = tc > 0 ? tc - 1 : 0;
      for (int tw = 0; tw < FillTokensPerWave; ++tw) {
        const int t    = fwv * FillTokensPerWave + tw;
        const int gi   = kv0 + (t < tc ? t : last);
        const int gc   = gi < S ? gi : (S > 0 ? S - 1 : 0);
        dst[tw] = S > 0 ? kv_indices[slot0 + gc] : 0;   // S==0: slot0 may be one past the end
      }
    }
  };
  const fp8_t* kv_base = Paged ? c_kv : (c_kv + (size_t)b * S * FusedDim);
  // Per-lane fill maps, from static_for because a runtime index spills: wgoff/wgoffT the offset in
  // the lane's row (paged), wb8/wb4 the flat offset (contiguous), wldso/wldsoT the LDS offset.
  uint32_t wgoff[FillOpsArray], wldso[FillOpsArray], wgoffT = 0, wldsoT = 0;
  const uint32_t wb8 = (uint32_t)(tokbase * FusedDim + lane * 8);
  const uint32_t wb4 = (uint32_t)(tokbase * FusedDim + FillTailOffset + lane * 4);
  static_for<0, FillWideOps>([&](auto ic) {
    constexpr int j = decltype(ic)::value;
    const int off = j * WaveSize * 8 + (int)lane * 8, tl = off / FusedDim;
    wgoff[j] = (uint32_t)(off - tl * FusedDim);
    wldso[j] = (uint32_t)((tokbase + tl) * KvRowBytes + (off - tl * FusedDim) * 2);
  });
  if constexpr (DoFill) {
    const int off = FillTailOffset + (int)lane * 4, tl = off / FusedDim;
    wgoffT = (uint32_t)(off - tl * FusedDim);
    wldsoT = (uint32_t)((tokbase + tl) * KvRowBytes + (off - tl * FusedDim) * 2);
  }
  // Issue this wave's loads for one tile. Paged: a 512-B op spans at most 2 of the 576-B token rows
  // and which rows is a compile-time function of the op index, so the second is picked with one
  // lane-mask select. Contiguous: clamp to the part's last token. The loads stay COMPILER-VISIBLE:
  // LLVM does not model inline-asm vmcnt and would hoist the convert above a hand-written wait.
  auto fill_load = [&](int kv0, int tc, const int* slots, FillRegs& dst) {
    const fp8_t* rb = Paged ? kv_base : kv_base + (size_t)kv0 * FusedDim;
    static_for<0, FillWideOps>([&](auto ic) {
      constexpr int j = decltype(ic)::value, B0 = j * WaveSize * 8;
      const fp8_t* a;
      if constexpr (Paged) {
        constexpr int t0 = B0 / FusedDim, tN = (B0 + (WaveSize - 1) * 8) / FusedDim;
        static_assert(tN <= t0 + 1 && tN < FillTokensPerWave, "wide fill op spans more rows than the slot pick");
        int src = slots[t0];
        if constexpr (tN > t0)
          src = sel_by_mask(src, slots[t0 + 1], LANE_GE(((t0 + 1) * FusedDim - B0 + 7) / 8));
        a = rb + (size_t)src * FusedDim + wgoff[j];
      } else {
        const int over = tokbase + (B0 + (int)lane * 8) / FusedDim - (tc - 1);
        a = rb + (wb8 + B0 - (uint32_t)((over > 0 ? over : 0) * FusedDim));
      }
      dst.w[j] = NonTemporal ? __builtin_nontemporal_load((const u32x2*)a) : *(const u32x2*)a;
    });
    if constexpr (DoFill) {
      constexpr int t0 = FillTailOffset / FusedDim;
      static_assert(t0 == FillTokensPerWave - 1 && (FillBytesPerWave - 1) / FusedDim == t0,
                    "the fill tail must sit inside one token row");
      const int over = tokbase + t0 - (tc - 1);
      const fp8_t* a = Paged ? rb + (size_t)slots[t0] * FusedDim + wgoffT
                             : rb + (wb4 - (uint32_t)((over > 0 ? over : 0) * FusedDim));
      dst.tail = NonTemporal ? __builtin_nontemporal_load((const uint32_t*)a) : *(const uint32_t*)a;
    }
  };
  auto fill_store = [&](const FillRegs& src, int s) {
    char* tm = (char*)sKv + s * (KvSlotElems * (int)sizeof(bf16));
    static_for<0, FillWideOps>([&](auto ic) {
      constexpr int j = decltype(ic)::value;
      for (int i = 0; i < 2; ++i) *(u32x2*)&tm[wldso[j] + i * 8] = cvt4_bf16(src.w[j][i]);
    });
    if constexpr (DoFill) *(u32x2*)&tm[wldsoT] = cvt4_bf16(src.tail);
  };

  // ── QK ──
  // This wave's KSteps K-steps over the 16-token tile -> partial f32x4 scores. Token row == mrow, so
  // the C-fragment is score[token = quad*4+e][row = mrow], the layout PV wants. Loads and MFMAs are
  // separate lambdas so the previous tile's softmax and PV can sit between them.
  constexpr int QkChunk = KSteps > 18 ? 18 : (KSteps ? KSteps : 1);   // 36 live A-operands would spill
  auto qk_load = [&](const bf16* rb, u32x2* a, int k0) {
    #pragma unroll
    for (int k = 0; k < QkChunk; ++k) a[k] = *(const u32x2*)(rb + (k0 + k) * KQ);
  };
  auto qk_mfma = [&](const u32x2* a, f32x4& acc, int k0) {
    #pragma unroll
    for (int k = 0; k < QkChunk; ++k) acc = mfma_bf16(as_bf16x4(a[k]), q_frag[k0 + k], acc);
  };

  // ── PV ── pull dim-group `g` of tile slot `base`; two u32x2 per token, KvRowBytes is 8-B aligned
  auto pv_fetch = [&](uint32_t* D, const char* base) {
    for (int j = 0; j < PvTokensPerLane; ++j) {
      const char* r = base + (quad * PvTokensPerLane + j) * KvRowBytes + pv_col_bytes;
      const u32x2 v0 = *(const u32x2*)r;            // 4 dims, not 8: ONE b64 per token
      D[0 * PvTokensPerLane + j] = v0[0];
      D[1 * PvTokensPerLane + j] = v0[1];
    }
  };
  auto pv_mfma = [&](const uint32_t* D, const bf16x4* prob) {
    for (int hh = 0; hh < PvDwords; ++hh) {
      uint32_t d[QuadSpan];
      transpose_4tok(&D[hh * PvTokensPerLane], d);  // ONE transpose feeds every N-tile
      for (int t = 0; t < NTiles; ++t)
        for (int p = 0; p < 2; ++p) {
          f32x4& acc = o_acc[t * PvSlots + 2 * hh + p];
          acc = mfma_bf16(as_bf16x4(d[2 * p], d[2 * p + 1]), prob[t], acc);
        }
    }
  };
  auto rescale_tile = [&](auto ti, float rescale) {
    constexpr int t = decltype(ti)::value;
    for (int c = t * PvSlots; c < (t + 1) * PvSlots; ++c) {
      o_acc[c][0] = mul_f32(o_acc[c][0], rescale); o_acc[c][1] = mul_f32(o_acc[c][1], rescale);
      o_acc[c][2] = mul_f32(o_acc[c][2], rescale); o_acc[c][3] = mul_f32(o_acc[c][3], rescale);
    }
  };

  // ── Softmax ── online softmax for one 16-token tile -> (run_max, run_sum) + the PV B-operand
  auto softmax_tile = [&](f32x4& scores, int kv0, bf16x4& prob_out, float& rescale_out) {
    float tile_max = NegInf;
    // The causal mask is IF-CONVERTED, never branched: the selects issue inside the MFMA run free.
    const int n_live_lane = mask_end - kv0;
    for (int e = 0; e < QuadSpan; ++e)
      scores[e] = (quad * PvTokensPerLane + e >= n_live_lane) ? NegInf : scores[e];
    for (int e = 0; e < QuadSpan; ++e) tile_max = fmaxf(tile_max, scores[e]);
    tile_max *= qk_descale;
    // Rescale slack: run_max (and o_acc) moves only when a score beats it by RescaleSlack; without
    // it the gate fires on nearly every tile. Costs log2(slack) of o_acc's 24 bits.
    const bool moved = (__ballot(tile_max > run_max + RescaleSlack) != 0ull);
    float new_max = run_max;
    if (__builtin_expect(moved, 0)) new_max = fmaxf(run_max, wave_max(tile_max));
    // A tile past the causal limit is all NegInf, so the exponent would cancel to the lift -- and
    // kv0 only walks forward, so exp_bias decides that once, in the prologue.
    const float exponent_bias = exp_bias - new_max;
    float tile_sum = 0.f;
    float prob[QuadSpan];
    for (int e = 0; e < QuadSpan; ++e) {
      prob[e] = __builtin_amdgcn_exp2f(__builtin_fmaf(scores[e], qk_descale, exponent_bias));
      tile_sum += prob[e];
    }
    prob_out = Packed ? pack4<Rna>(prob[0], prob[1], prob[2], prob[3])
                      : pack4<Rtne>(prob[0], prob[1], prob[2], prob[3]);
    tile_sum *= (1.0f / ProbLift);   // unlift the denominator back to true units
    if (!moved) {
      run_sum = run_sum + tile_sum;
      rescale_out = 1.0f;            // consumers skip the multiply on a wave-uniform compare
    } else {
      const bool first = (run_max == NegInf);
      const float rescale = first ? 0.f : __builtin_amdgcn_exp2f(run_max - new_max);
      run_sum = run_sum * rescale + tile_sum;
      rescale_out = rescale;         // applied by every wave to THIS tile's slice, one step later
      run_max = new_max;
    }
  };

  // ── Main loop ──
  // Step h reads tile h (slot h&1) for BOTH the QK and the V operands but spends those reads'
  // latency on tile h-1's softmax and tile h-2's PV. One barrier per step.
  constexpr int SlotBytes = KvSlotElems * (int)sizeof(bf16);
  uint32_t v_frag[2][PvDwords * PvTokensPerLane] = {};   // V of tile h, consumed at h+2
  f32x4    qk_own = {};                                  // this wave's own K-slice partials
  FillRegs fill0, fill1;                                 // two tiles of HBM loads in flight
  int slot_ids[FillTokensPerWave ? FillTokensPerWave : 1] = {};
  auto fill_issue = [&](int it, FillRegs& R) {
    const int kv0 = kv_lo + it * TileTokens;
    if (it < ntiles) fill_load(kv0, min(TileTokens, kv_hi - kv0), slot_ids, R);
  };
  // No hand-written vmcnt: LLVM waits for exactly THIS tile's and leaves the next tile's in flight.
  auto fill_commit = [&](int it, int s, const FillRegs& R) { if (it < ntiles) fill_store(R, s); };
  // One pipeline step. P is h&1 and fixes the LDS slot, the rendezvous plane set and which V buffer
  // is filled versus consumed, all at compile time. Steady is false only for the peeled step 0.
  auto step = [&](int h, auto pc, auto fc, FillRegs& fill) {
    constexpr int  P = decltype(pc)::value;
    constexpr bool Steady = decltype(fc)::value != 0;
    const int rd = P, wr = 1 - P;                  // tile h reads slot P; tile h+1 fills the other
    load_slots(h + 3, slot_ids);
    f32x4 scores;
    // Peers' partials for tile h-1 FIRST: LDS returns in order, so the softmax starts early.
    if (KSteps > 0 && (Steady || h > 0)) {
      scores = qk_own;
      if constexpr (NumPeers >= 1) scores = scores + *(const f32x4*)&sQkPeer[(1 - P) * RvSetStride + qk_peer0][qk_slot];
      if constexpr (NumPeers == 2) scores = scores + *(const f32x4*)&sQkPeer[(1 - P) * RvSetStride + qk_peer1][qk_slot];
      // Pin the peer read's CONSUMER: it only pays if the softmax's s_waitcnt covers THAT and not
      // the QK reads below. Emits no instruction.
      asm("" : "+v"(scores));
    }
    u32x2 kfrag[QkChunk];
    const bf16* qk_row = &sKv[rd * KvSlotElems + mrow * KvRowElems + kstep_lo * KQ + quad * QLane];
    // No `h < ntiles` guard: at h == ntiles this reads a stale but finite slot with no consumer.
    if constexpr (KSteps > 0) qk_load(qk_row, kfrag, 0);
    if (KSteps > 0 && (Steady || h > 0)) {
      bf16x4 prob = {};  float rescale = 1.0f;
      softmax_tile(scores, kv_lo + (h - 1) * TileTokens, prob, rescale);
      if (publishes) {
        sProb[((1 - P) * MaxNTiles + tile) * WaveSize + lane] = prob;   // 1 ds_write_b64
        if (quad == 0) ((float*)sRescale)[(((1 - P) * MfmaTile + mrow) * MaxNTiles) + tile] = rescale;
      }
    }
    // Tile h-2's PV operands and lazy rescale come AFTER the softmax, and must stay there. They touch
    // plane P and o_acc, the softmax plane 1-P, so either order is correct -- but this one keeps the
    // rescale guards out of the softmax-to-MFMA span, on both the paged and the contiguous path.
    bf16x4 pv_prob[NTiles];
    for (int t = 0; t < NTiles; ++t) pv_prob[t] = sProb[(P * MaxNTiles + t) * WaveSize + lane];
    {
      const f32x4 rv = sRescale[P * MfmaTile + mrow];              // this ROW's factors, one b128
      static_for<0, NTiles>([&](auto ti) {
        const float r = rv[decltype(ti)::value];
        if (__builtin_expect(r != 1.0f, 0)) rescale_tile(ti, r);
      });
    }
    // PV of tile h-2, published at step h-1 and visible across the barrier that ended it. Tile h-2
    // has parity P and tile h-1 has 1-P, so they never collide: NO second barrier may be added.
    // Unguarded: the prologue leaves both PV planes NEUTRAL, so steps 0 and 1 add an exact 0.0 * V.
    pv_mfma(v_frag[P], pv_prob);
    pv_fetch(v_frag[P], (const char*)sKv + rd * SlotBytes);   // V of tile h, consumed two steps on
    if constexpr (KSteps > 0) {
      f32x4 acc = {};                                        // covers the fetch just issued
      static_for<0, KSteps / QkChunk>([&](auto cc) {
        constexpr int c = decltype(cc)::value;
        if constexpr (c > 0) qk_load(qk_row, kfrag, c * QkChunk);
        qk_mfma(kfrag, acc, c * QkChunk);
      });
      if constexpr (NumPeers > 0) *(f32x4*)&sQkPeer[P * RvSetStride + qk_plane][qk_slot] = acc;
      qk_own = acc;
    }
    fill_commit(h + 1, wr, fill);
    fill_issue(h + 3, fill);
    lds_drain();
    barrier();
  };

  // ── Prologue: tile 0 converted into slot 0; tiles 1 and 2 already in flight ──
  load_slots(0, slot_ids);
  fill_issue(0, fill0);
  load_slots(1, slot_ids);
  fill_commit(0, 0, fill0);
  fill_issue(1, fill0);
  load_slots(2, slot_ids);
  fill_issue(2, fill1);
  // The zero plane the 2-way K-split collects against: written once, never touched again.
  if constexpr (Packed) if (wave == 0) {
    const f32x4 z = {};
    *(f32x4*)&sQkPeer[NumWaves][qk_slot] = z;  *(f32x4*)&sQkPeer[RvSetStride + NumWaves][qk_slot] = z;
  }
  // Neutral PV planes (probability +0.0, rescale 1.0), BOTH parities: 512 threads cover them all.
  const int tid0 = (int)threadIdx.x;
  if (tid0 < 2 * MaxNTiles * WaveSize) { const bf16x4 z = {}; sProb[tid0] = z; }
  if (tid0 < 2 * MfmaTile * MaxNTiles) ((float*)sRescale)[tid0] = 1.0f;
  lds_drain();
  barrier();                                           // tile 0 readable
  // Step 0 is peeled -- no previous tile to soft-max -- so the loop body is the pure steady state.
  step(0, IC<0>{}, IC<0>{}, fill0);
  int h = 1;
  #pragma unroll 1
  for (; h + 1 <= ntiles; h += 2) {
    step(h,     IC<1>{}, IC<1>{}, fill1);
    step(h + 1, IC<0>{}, IC<1>{}, fill0);
  }
  if (h <= ntiles) step(h, IC<1>{}, IC<1>{}, fill1);

  // ── Drain: one tile's PV is still outstanding ──
  // Published by step ntiles and fetched by step ntiles-1, both visible across that step's barrier.
  auto pv_drain = [&](auto qc) {
    constexpr int Q = decltype(qc)::value;
    bf16x4 pv_prob[NTiles];
    for (int t = 0; t < NTiles; ++t) pv_prob[t] = sProb[(Q * MaxNTiles + t) * WaveSize + lane];
    const f32x4 rv = sRescale[Q * MfmaTile + mrow];
    static_for<0, NTiles>([&](auto ti) {
      const float r = rv[decltype(ti)::value];
      if (__builtin_expect(r != 1.0f, 0)) rescale_tile(ti, r);
    });
    pv_mfma(v_frag[Q], pv_prob);
  };
  if (ntiles > 0) { if (ntiles & 1) pv_drain(IC<0>{}); else pv_drain(IC<1>{}); }

  // ── Epilogue ──
  if constexpr (KSteps > 0) run_sum = wave_sum(run_sum);
  // Opaque pin: without it LLVM re-derives `mrow` at each epilogue use, 1 VGPR for 3 extra v_mov.
  int mrow_o = mrow;
  asm("" : "+v"(mrow_o));
  const int prow_o = tile * MfmaTile + mrow_o;
  auto store_slice = [&](bf16* dst, int t, float sc) {
    for (int e = 0; e < QuadSpan; e += 2) {
      const int dim = pv_base + PvSlots * (QuadSpan * quad + e);
      *(bf16x8*)&dst[dim] = pack8(o_acc[t * PvSlots + 0][e] * sc, o_acc[t * PvSlots + 1][e] * sc,
                                  o_acc[t * PvSlots + 2][e] * sc, o_acc[t * PvSlots + 3][e] * sc,
                                  o_acc[t * PvSlots + 0][e + 1] * sc, o_acc[t * PvSlots + 1][e + 1] * sc,
                                  o_acc[t * PvSlots + 2][e + 1] * sc, o_acc[t * PvSlots + 3][e + 1] * sc);
    }
  };
  for (int t = 0; t < NTiles; ++t) {
    const int  prow_t = t * MfmaTile + mrow_o;
    const bool store_t = Packed ? (prow_t < H * nact)
                                : (mrow_o < H && qgrp * DraftPerCta + t < q_len);
    if (!store_t) continue;
    const int slot = Packed ? prow_t : (t * MaxHeads + mrow_o);
    store_slice((bf16*)partials + ((size_t)slab_cta * SlabSlots + slot) * LatDim, t,
                kv_scale * (1.0f / ProbLift));
  }
  if (publishes && quad == 0
      && (Packed ? (prow_o < H * nact) : (mrow_o < H && qpos_g < q_len))) {
    const int slot = Packed ? prow_o : (tile * MaxHeads + mrow_o);
    const size_t nctas = (size_t)gridDim.x;
    partials[hdr_off(nctas, SlabSlots, slot, 0) + slab_cta] = run_max;
    partials[hdr_off(nctas, SlabSlots, slot, 1) + slab_cta] = run_sum;
  }
}
// Dispatcher: owns the LDS and picks each wave's role. Both roles run the same s_barrier count.
template <int QL, bool Paged, bool NonTemporal, int NTiles>
__global__ void __launch_bounds__(CtaThreads, 1) mla_decode_a16w8_multiq_kernel(
    const bf16* __restrict__ q_lat,   // [B, QL, H, 512] bf16; q_pe is [B, QL, H, 64]
    const bf16* __restrict__ q_pe,
    const fp8_t* __restrict__ c_kv,   // [B, S, 576] fp8 slab | fused [num_slots,1,576] pool (Paged)
    float* __restrict__ partials,     // fp32 split workspace; the merge turns it into o_lat
    int B, int H, int S_scalar, int parts, int q_len, float scale_log2e, float kv_scale,
    const int* __restrict__ seq_lens,              // Paged, all four: [B] lengths; [B] q/out rows
    const int* __restrict__ rows,                  // (null -> b); [sum seq] slots; [B+1] offsets
    const int* __restrict__ kv_indices, const int* __restrict__ kv_indptr) {
  constexpr int RvPlanes = rv_planes(NTiles) * 2;  // double-buffered by tile parity
  static_assert(KvLdsElems * (int)sizeof(bf16) + RvPlanes * RvStride * (int)sizeof(float)
      + ProbPlaneElems * (int)sizeof(bf16x4) + RescalePlaneElems * (int)sizeof(f32x4) <= 65536, "LDS");
  __shared__ bf16   sKv[KvLdsElems];
  __shared__ float  sQkPeer[RvPlanes][RvStride];
  __shared__ bf16x4 sProb[ProbPlaneElems];             // cross-wave probability planes
  __shared__ f32x4  sRescale[RescalePlaneElems];       // per-(row, tile) lazy-rescale factors
  #define DECODE_BODY(KSTEPS)                                                                     \
    decode_body<QL, Paged, NonTemporal, NTiles, KSTEPS>(                                          \
        q_lat, q_pe, c_kv, partials, B, H, S_scalar, parts, q_len, scale_log2e, kv_scale,          \
        seq_lens, rows, kv_indices, kv_indptr, sKv, sQkPeer, sProb, sRescale)
  if constexpr (NTiles == 3) {
    if (threadIdx.x < 4 * WaveSize) { DECODE_BODY(18); }     // heavy: slot-0 waves
    else { asm volatile("s_setprio 1" ::: "memory");         // light waves win SIMD arbitration
           DECODE_BODY(9); }                                 // light: slot-1 waves
  } else {
    if (threadIdx.x < 4 * WaveSize) { DECODE_BODY(QkSteps); }  // one wave owns an N-tile's 36 steps
    else { asm volatile("s_setprio 1" ::: "memory");
           DECODE_BODY(0); }                                  // fill + PV only
  }
  #undef DECODE_BODY
}

// ── Merge kernel ──
// Online-softmax reduce of the partials: a CTA owns one (request, q position, head) and a
// DimsPerCta = 512/DimSplit dim slice, with a fixed sum association.
template <int QL, int DimSplit>
__global__ void __launch_bounds__(256) mla_decode_a16w8_merge(
    const float* __restrict__ partials, bf16* __restrict__ o_lat, int H, int parts, int q_len,
    int nctas, int packH, const int* __restrict__ rows, int B,
    const int* __restrict__ seq_lens) {
  constexpr int Threads = 256, NW = Threads / WaveSize, QGroups = qgroups_of(QL);
  constexpr int DimsPerCta = LatDim / DimSplit, DimsPerLane = 8;
  constexpr int LanesPerRow = DimsPerCta / DimsPerLane, RowsPerWave = WaveSize / LanesPerRow;
  constexpr int RowsPerCta = NW * RowsPerWave, HdrPerThread = (MaxParts + Threads - 1) / Threads;
  static_assert(LanesPerRow >= 1 && LanesPerRow * DimsPerLane == DimsPerCta && RowsPerWave * LanesPerRow == WaveSize);
  const int live = live_slots(packH);             // slots this grid enumerates
  const int bh = blockIdx.x / DimSplit, dblk = blockIdx.x - bh * DimSplit;
  const int slot = bh % live, bq = bh / live;     // packed slots are 0..PackedSlabSlots-1
  const int qgrp = QGroups > 1 ? bq % QGroups : 0, b = QGroups > 1 ? bq / QGroups : bq;
  // Paged `rows`: ONLY the destination is remapped; the workspace is addressed by LAUNCH-ORDER b.
  const int b_out = rows ? min(max(rows[b], 0), B - 1) : b;
  const int hdiv = packH ? packH : MaxHeads;
  const int head = slot % hdiv, qpos_g = qgrp * DraftPerCta + slot / hdiv;
  const bool slot_live = !packH || slot < packH * DraftPerCta;
  const int tid = threadIdx.x, lane = tid & (WaveSize - 1), wv = tid / WaveSize;
  // The decode writes partials at the LOGICAL (unit, qgrp), so these parts stride by QGroups.
  int ubase = b * parts, nparts = parts;
  if (seq_lens) {
    const SplitPlan pl = scan_split_plan(seq_lens, B);
    nparts = pl.units(seq_lens, b, B, parts);
    ubase = b * nparts;
  // Ragged: the unit base is a prefix sum over the requests below, lane-parallel (units() divides).
    if (!pl.same) {
      int acc = 0;
      for (int i = lane; i < b; i += WaveSize) acc += pl.units(seq_lens, i, B, parts);
      ubase = wave_sum_i(acc);
    }
  }
  const size_t cta0 = (size_t)ubase * QGroups + qgrp;
  const size_t cstr = QGroups;
  __shared__ float s_w[MaxParts];               // per-part softmax weight exp2(max_p - max)
  __shared__ float s_rm[NW], s_rs[NW];          // cross-wave folds of run_max / run_sum
  __shared__ float s_o[(NW - 1) * DimsPerCta];  // cross-wave fold of the accumulated O slice

  // ── Pass 1: reduce the (run_max, run_sum) header planes into a global max and per-part weights ──
  const float* __restrict__ hmax = partials + hdr_off((size_t)nctas, SlabSlots, slot, 0);
  const float* __restrict__ hsum = partials + hdr_off((size_t)nctas, SlabSlots, slot, 1);
  float lm[HdrPerThread], ls[HdrPerThread], lmax = NegInf;
  for (int k = 0; k < HdrPerThread; ++k) {      // each thread loads HdrPerThread header entries
    const int pp = tid + k * Threads;
    lm[k] = (pp < nparts) ? hmax[cta0 + (size_t)pp * cstr] : NegInf;
    ls[k] = (pp < nparts) ? hsum[cta0 + (size_t)pp * cstr] : 0.f;
    lmax  = fmaxf(lmax, lm[k]);
  }
  for (int m = 1; m < WaveSize; m <<= 1) lmax = fmaxf(lmax, lane_xor(lmax, m));
  if (lane == 0) s_rm[wv] = lmax;
  __syncthreads();
  float mmax = s_rm[0];
  for (int w = 1; w < NW; ++w) mmax = fmaxf(mmax, s_rm[w]);
  const bool dead = (mmax == NegInf);           // every part empty (over-split / S==0) -> zeros
  float lsum = 0.f;
  for (int k = 0; k < HdrPerThread; ++k) {
    const int pp = tid + k * Threads;
    const float w = dead ? 0.f : __builtin_amdgcn_exp2f(lm[k] - mmax);
    if (pp < nparts) s_w[pp] = w;
    lsum += w * ls[k];
  }
  for (int m = 1; m < WaveSize; m <<= 1) lsum += lane_xor(lsum, m);
  if (lane == 0) s_rs[wv] = lsum;
  __syncthreads();
  float msum = s_rs[0];
  for (int w = 1; w < NW; ++w) msum += s_rs[w];
  const float inv = msum > 0.f ? 1.f / msum : 0.f;

  // ── Pass 2: sum_p w_p * O_p over this CTA's output-dim slice; waves stride parts by RowsPerCta
  const int sub = lane / LanesPerRow, dl = lane - sub * LanesPerRow;
  const size_t rowStride = (size_t)cstr * SlabSlots * LatDim;
  const bf16* __restrict__ src = (const bf16*)partials + (cta0 * SlabSlots + slot) * LatDim
                                 + dblk * DimsPerCta + dl * DimsPerLane;
  float acc[DimsPerLane] = {};
  #pragma unroll 4
  for (int pp = wv * RowsPerWave + sub; pp < nparts; pp += RowsPerCta) {
    const bf16x8 v = *(const bf16x8*)(src + (size_t)pp * rowStride);
    const float w = s_w[pp];
    for (int d = 0; d < DimsPerLane; ++d) acc[d] = __builtin_fmaf(w, (float)v[d], acc[d]);
  }
  for (int m = LanesPerRow; m < WaveSize; m <<= 1)
    for (int d = 0; d < DimsPerLane; ++d) acc[d] += lane_xor(acc[d], m);
  if (wv != 0 && lane < LanesPerRow) {
    float* d = &s_o[(wv - 1) * DimsPerCta + lane * DimsPerLane];
    for (int i = 0; i < DimsPerLane; ++i) d[i] = acc[i];
  }
  __syncthreads();
  if (wv == 0 && lane < LanesPerRow && qpos_g < q_len && head < H && slot_live) {
    for (int w = 0; w < NW - 1; ++w) {
      const float* s = &s_o[w * DimsPerCta + lane * DimsPerLane];
      for (int i = 0; i < DimsPerLane; ++i) acc[i] += s[i];
    }
    const bf16x4 p0 = pack4<Rtne>(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
    const bf16x4 p1 = pack4<Rtne>(acc[4] * inv, acc[5] * inv, acc[6] * inv, acc[7] * inv);
    bf16x8 out; __builtin_memcpy(&out, &p0, 8); __builtin_memcpy((char*)&out + 8, &p1, 8);
    *(bf16x8*)(o_lat + (size_t)((b_out * q_len + qpos_g) * H + head) * LatDim
                       + dblk * DimsPerCta + lane * DimsPerLane) = out;
  }
}
// ── Host-side split planning ──
constexpr size_t FusedDimHost = 576;             // one fused KV row: 512 latent + 64 rope
constexpr size_t MallBytes    = 256ull << 20;    // MI300X MALL capacity
constexpr size_t MaxCTAs      = 2048 + 2 * NumCU;
constexpr int SplitGrid = 256, SplitLoop = 20;
constexpr int PlannerTileTokens = 64;
// Non-temporal gate: bypass L2/MALL only above the MALL, since below it the pool is reused across
// decode steps. With QGroups > 1 the CTAs re-read tiles at a one-tile distance, so they stay cached.
__host__ inline bool want_nt(size_t kv_bytes, int qgroups) { return qgroups == 1 && kv_bytes > MallBytes; }
// KV-splits (`parts`) per request. Occupancy is 1, so fill the machine once
// (parts = NumCU/(B*QGroups)) and floor by the workspace CTA cap. Then SHAPE it: a part's fixed
// prologue and epilogue are paid in PARALLEL, so cutting `parts` trades grid width for critical
// path -- hence the 256-CTA rung and the round trip through `per` that drops empty parts.
__host__ inline int plan_parts(int B, int S, int qgroups) {
  const int bb = (B > 0 ? B : 1) * qgroups, tiles = kv_tiles(S);
  int pmax = (S + PlannerTileTokens - 1) / PlannerTileTokens;   // pmax = min(rungs, MaxParts, cap)
  const int cap = (B > 0) ? (int)(MaxCTAs / ((size_t)B * qgroups)) : 1;
  if (pmax > MaxParts) pmax = MaxParts;
  if (pmax > cap)      pmax = cap;
  int p = NumCU / bb;
  if (p > pmax) p = pmax;
  if (p < 1)    p = 1;
  if (p >= tiles) return tiles;                                 // already one tile per part
  if ((long long)tiles >= (long long)SplitLoop * p) return p;   // loop dominates: leave it alone
  const int lo = SplitGrid / bb;
  if (lo >= 1 && lo < p) p = lo;
  const int per = (tiles + p - 1) / p;
  return (tiles + per - 1) / per;
}
__host__ inline int ql_pad(int q_len) { return q_len <= 4 ? 4 : 8; }
// `parts` arrives unvalidated: the sizer and the launcher clamp it through ONE function.
__host__ inline int capped_parts(int parts, int B, int ng) {
  const int bb   = B  > 0 ? B  : 1;
  const int cap  = (int)(MaxCTAs / ((size_t)bb * (ng > 0 ? ng : 1)));
  const int maxp = cap >= MaxParts ? MaxParts : (cap > 1 ? cap : 1);
  const int p    = parts < 1 ? 1 : parts;
  return p > maxp ? maxp : p;
}

// ── Launchers ──
// One resolved launch: the host has already picked the padded width, the KV split and the NT gate.
struct Launch {
  const void* q_lat; const void* q_pe; const void* c_kv; void* o_lat; void* workspace;
  int B, H, S, parts, q_len, ngroups;  float scale_log2e, kv_scale;
  const int *seq_lens, *rows, *kv_indices, *kv_indptr;  bool paged, nt;  hipStream_t stream;
};
template <int QLP, bool Paged, bool NonTemporal, int NTiles>
__host__ inline void launch_tiles(const Launch& a) {
  mla_decode_a16w8_multiq_kernel<QLP, Paged, NonTemporal, NTiles>
      <<<dim3(a.B * a.parts * a.ngroups), dim3(CtaThreads), 0, a.stream>>>(
          (const bf16*)a.q_lat, (const bf16*)a.q_pe, (const fp8_t*)a.c_kv,
          (float*)a.workspace, a.B, a.H, a.S, a.parts, a.q_len, a.scale_log2e, a.kv_scale,
          a.seq_lens, a.rows, a.kv_indices, a.kv_indptr);
}
// use_head_packing, not a local flag, decides NTiles here AND feeds the merge's packH.
template <int QLP, bool Paged, bool NonTemporal>
__host__ inline void launch_pack(const Launch& a) {
  use_head_packing(a.H) ? launch_tiles<QLP, Paged, NonTemporal, 3>(a)
                        : launch_tiles<QLP, Paged, NonTemporal, 4>(a);
}
// Decode, then merge. DimSplit rises until the merge grid is ~2 CTAs per CU, capped at 16.
template <int QL>
__host__ inline void launch_ql(const Launch& a, int packH) {
  a.paged ? (a.nt ? launch_pack<QL, true,  true>(a) : launch_pack<QL, true,  false>(a))
          : (a.nt ? launch_pack<QL, false, true>(a) : launch_pack<QL, false, false>(a));
  const int units = a.B * qgroups_of(QL) * live_slots(packH);
  const int nctas = a.B * a.parts * qgroups_of(QL);
  int ds = 1;
  while (ds < 16 && units * ds * 2 < 4 * NumCU) ds <<= 1;
  #define MERGE_CASE(D)                                                                  \
    case D: mla_decode_a16w8_merge<QL, D><<<dim3(units*(D)), dim3(256), 0, a.stream>>>(   \
                (const float*)a.workspace, (bf16*)a.o_lat, a.H, a.parts, a.q_len,         \
                nctas, packH, a.rows, a.B, a.seq_lens); break;
  switch (ds) {
    MERGE_CASE(2) MERGE_CASE(4) MERGE_CASE(8) MERGE_CASE(16)
    default: MERGE_CASE(1)
  }
  #undef MERGE_CASE
}
// The merge is unconditional: the decode kernel only ever writes partials.
__host__ inline int launch(const Launch& a, int qlp) {
  const int packH = use_head_packing(a.H) ? a.H : 0;
  qlp <= 4 ? launch_ql<4>(a, packH) : launch_ql<8>(a, packH);
  return (int)hipGetLastError();
}
// Host-side domain gate: it rejects exactly the inputs this kernel carries no code for -- a
// head-tile grid axis, a grid-quantisation planner, the q_len < 4 CTA shape, and a single-part
// epilogue that would write o_lat directly.
static_assert(MaxBatch * qgroups_of(MaxQL) <= NumCU / 2, "the domain's B must fit the planner");
__host__ inline int check_domain(int B, int H, int q_len, int lat_dim, int rope_dim,
                                 int ngroups, const void* workspace) {
  if (lat_dim != LatDim || rope_dim != RopeDim)            return -2;
  if (H < 1 || H > MaxHeads || q_len < 4 || q_len > MaxQL) return -3;
  if (B < 1 || B * ngroups > NumCU / 2)                    return -4;
  return workspace ? 0 : -5;
}
}  // namespace

// ── extern "C" ABI ── (the _multiq suffix: the q_len=1 kernel shares this extension)
// KV-splits for a planned launch. H is ABI but no longer shapes the grid: at H <= 16, one tile.
extern "C" __host__ int mla_decode_a16w8_multiq_plan_parts_q(int B, int max_seq_len, int q_len, int H) {
  (void)H;
  return plan_parts(B, max_seq_len, qgroups_of(ql_pad(q_len)));
}
// Workspace: nctas = B*parts*QGroups slabs of SlabSlots slots, plus the two header planes.
extern "C" __host__ size_t mla_decode_a16w8_multiq_workspace_for(int B, int H, int parts, int q_len) {
  (void)H;
  const int ng = qgroups_of(ql_pad(q_len));
  const size_t nctas = (size_t)(B > 0 ? B : 1) * capped_parts(parts, B, ng) * ng;
  return hdr_off(nctas, SlabSlots, SlabSlots - 1, 2) * sizeof(float);
}
// Contiguous launcher: c_kv is a fused fp8 [B,S,576] slab. q_lat/q_pe/o_lat are [B,q_len,H,*].
extern "C" __host__ int launch_mla_decode_a16w8_multiq(
    const void* q_lat, const void* q_pe, const void* c_kv, void* o_lat,
    int B, int H, int S, int q_len, int lat_dim, int rope_dim, float scale, float kv_scale,
    void* workspace, void* stream_) {
  const int qlp = ql_pad(q_len);
  const int ng  = qgroups_of(qlp);
  if (const int rc = check_domain(B, H, q_len, lat_dim, rope_dim, ng, workspace)) return rc;
  Launch a{};
  a.q_lat = q_lat; a.q_pe = q_pe; a.c_kv = c_kv; a.o_lat = o_lat; a.workspace = workspace;
  a.B = B; a.H = H; a.S = S; a.q_len = q_len; a.ngroups = ng; a.parts = plan_parts(B, S, ng);
  a.nt = want_nt((size_t)B * (size_t)S * FusedDimHost, ng); a.kv_scale = kv_scale;
  a.scale_log2e = 1.4426950408889634f * scale; a.stream = (hipStream_t)stream_;
  return launch(a, qlp);
}
// Paged launcher: graph-safe and device-driven, so `parts` is fixed at capture.
extern "C" __host__ int launch_mla_decode_a16w8_multiq_paged_dev(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat,
    const void* seq_lens, const void* rows, const void* kv_indices, const void* kv_indptr,
    int B, int H, int parts, int q_len, int lat_dim, int rope_dim, float scale, float kv_scale,
    void* workspace, void* stream_, size_t kv_bytes) {
  const int qlp = ql_pad(q_len);
  const int ng  = qgroups_of(qlp);
  if (const int rc = check_domain(B, H, q_len, lat_dim, rope_dim, ng, workspace)) return rc;
  Launch a{};
  a.q_lat = q_lat; a.q_pe = q_pe; a.c_kv = kv_pool; a.o_lat = o_lat; a.workspace = workspace;
  a.B = B; a.H = H; a.S = 0; a.q_len = q_len; a.ngroups = ng;   // S comes from device seq_lens
  // `parts` is the caller's, fixed at capture: clamp it to the grid the workspace was sized for.
  a.parts = capped_parts(parts, B, ng); a.nt = want_nt(kv_bytes, ng); a.kv_scale = kv_scale;
  a.scale_log2e = 1.4426950408889634f * scale; a.stream = (hipStream_t)stream_; a.paged = true;
  a.seq_lens = (const int*)seq_lens; a.rows = (const int*)rows;
  a.kv_indices = (const int*)kv_indices; a.kv_indptr = (const int*)kv_indptr;
  return launch(a, qlp);
}
