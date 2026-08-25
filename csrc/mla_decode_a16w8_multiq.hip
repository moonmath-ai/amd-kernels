// ============================================================================
// MLA a16w8 multi-query decode for CDNA3 (gfx942 / MI300X), absorbed form:
//
//   score[t,h,n] = (q_lat[t,h,:] . c_kv[n,0:512] + q_pe[t,h,:] . c_kv[n,512:576]) * scale
//   o_lat[t,h,:] = sum_n softmax_n(score)[t,h,n] * c_kv[n,0:512]
//
// One fused fp8 c_kv row is BOTH K and V (W_UK/W_UV absorbed outside) and Q stays bf16, so every MFMA is
// v_mfma_f32_16x16x16bf16_1k. Decode writes FlashDecoding partials; the merge kernel below reduces them.
//
// One CTA of 8 waves per CU, occupancy 1. Query rows pack into NTiles tiles of 16 rows, KV into tiles of 16
// tokens, and the CTA walks the KV tiles one at a time. Each wave has up to three jobs per tile:
//
//   always        PV-accumulate 64 of the 512 output dims, for EVERY query tile
//   most waves    contract a slice of one query tile's 36 QK K-steps (see the wave map below)
//   waves 6, 7    stage the next KV tile out of HBM: fp8 -> bf16 -> the LDS slot the CTA is not reading
//
// There is exactly ONE barrier per KV tile, and it separates "read tile h" from "fill tile h+1".
//
// Three arms exist, chosen by how many query tiles the shape packs into: NTiles 3 (q_len 4), 4 (H >= 13) and
// 6 (q_len 8, H <= 12). NTiles is a template parameter, as is each wave's role, so every arm compiles to
// straight-line code with no map lookups in it.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

namespace {
using bf16 = __bf16;
using bf16x4 = __attribute__((ext_vector_type(4))) bf16;
using bf16x8 = __attribute__((ext_vector_type(8))) bf16;
using f32x2 = __attribute__((ext_vector_type(2))) float;
using f32x4 = __attribute__((ext_vector_type(4))) float;
using u32x2 = __attribute__((ext_vector_type(2))) uint32_t;
using fp8_t = __hip_fp8_e4m3_fnuz;

template <class T, class U> __device__ __forceinline__ T bitcast(const U& u) {
  static_assert(sizeof(T) == sizeof(U), "bitcast width");
  T t;
  __builtin_memcpy(&t, &u, sizeof(T));
  return t;
}

// ── Problem constants ─────────────────────────────────────────────────────────────────────────
constexpr int WaveSize = 64, MfmaTile = 16, QuadSpan = 4;              // wave; MFMA M==N; C-fragment quad
constexpr int LatDim = 512, RopeDim = 64, FusedDim = LatDim + RopeDim; // kv_lora_rank, rope: 576 fp8 B/token
constexpr int KQ = 16, QLane = KQ / QuadSpan;                          // MFMA contraction length, a lane's share
constexpr int LatIters = LatDim / KQ, RopeIters = RopeDim / KQ;        // 32 latent + 4 rope K-steps
constexpr int QkSteps = LatIters + RopeIters;                          // 36 K-steps per score contraction
constexpr int PvChunks = LatDim / MfmaTile;                            // 32 PV output-dim tiles
constexpr int NumWaves = 8, CtaThreads = NumWaves * WaveSize, NumCU = 304;
constexpr int MaxQL = 8, MaxHeads = 16, MaxBatch = 32, MaxDraftPerCta = 8;   // the SUPPORTED domain
constexpr int TileTokens = 16, FillWaves = 2, FillTok = TileTokens / FillWaves;
constexpr float NegInf = -3.0e38f;

// Probabilities go to bf16 for the PV MFMA. Lifting by 2^7 keeps the small end out of bf16 denormals;
// the epilogue and run_sum divide it back out. RescaleSlack lets the running max drift 4 log2 units
// before a rescale is paid, so a lifted probability still peaks far inside bf16's range.
constexpr int Log2ProbLift = 7;
constexpr float ProbLiftLog2 = (float)Log2ProbLift, ProbLift = (float)(1 << Log2ProbLift);
constexpr float RescaleSlack = 4.0f;

// ── Packing primitives ────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ f32x4 mfma_bf16(bf16x4 a, bf16x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ bf16x4 as_bf16x4(uint32_t lo, uint32_t hi) { return bitcast<bf16x4>(u32x2{lo, hi}); }

enum Round { Trunc, Rtne, Rna };
template <Round R> __device__ __forceinline__ unsigned narrow_bf16(float f) {
  const unsigned u = bitcast<unsigned>(f);
  return R == Trunc ? u : R == Rna ? u + 0x8000u : u + 0x7FFFu + ((u >> 16) & 1u);
}

template <Round R> __device__ __forceinline__ bf16x4 pack4(float a, float b, float c, float d) {
  return as_bf16x4(__builtin_amdgcn_perm(narrow_bf16<R>(b), narrow_bf16<R>(a), 0x07060302u),
                   __builtin_amdgcn_perm(narrow_bf16<R>(d), narrow_bf16<R>(c), 0x07060302u));
}

__device__ __forceinline__ bf16x8 pack8(float a, float b, float c, float d, float e, float f, float g, float h) {
  const bf16x4 v[2] = {pack4<Rtne>(a, b, c, d), pack4<Rtne>(e, f, g, h)};
  return bitcast<bf16x8>(v);
}

// One dword = 4 fp8 dims -> the two dwords a ds_write wants. MUST go THROUGH a bf16x4: a hand-written
// cvt_pk/perm pair over f32x2 subscripts is a live miscompile.
__device__ __forceinline__ u32x2 cvt4_bf16(uint32_t w) {
  const f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false), hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
  return bitcast<u32x2>(pack4<Trunc>(lo[0], lo[1], hi[0], hi[1]));
}

__device__ __forceinline__ void transpose_4tok(const uint32_t* in, uint32_t* out) {
  out[0] = __builtin_amdgcn_perm(in[1], in[0], 0x05040100u);      // dim 2a,   tokens 0,1
  out[1] = __builtin_amdgcn_perm(in[3], in[2], 0x05040100u);      // dim 2a,   tokens 2,3
  out[2] = __builtin_amdgcn_perm(in[1], in[0], 0x07060302u);      // dim 2a+1, tokens 0,1
  out[3] = __builtin_amdgcn_perm(in[3], in[2], 0x07060302u); }    // dim 2a+1, tokens 2,3
// KEEP AS ASM: plain `*=` becomes v_pk_mul_f32, which shares the FP32 MAC with the MFMA pipe.
__device__ __forceinline__ float mul_f32(float a, float b) {
  float r;
  asm volatile("v_mul_f32_e32 %0,%1,%2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

// ── Synchronisation ── KEEP AS ASM: __syncthreads() adds a fence this kernel never needs, and that fence costs
// 1.6%. s_barrier drains no counter, so a ds_write a peer must SEE needs lds_drain first.
// 0xC07F is vmcnt(63) expcnt(7) lgkmcnt(0): wait on LDS only, and leave the KV loads in flight.
__device__ __forceinline__ void tile_barrier() {
  __builtin_amdgcn_s_waitcnt(0xC07F);
  asm volatile("s_barrier" ::: "memory");
}
template <class T> __device__ __forceinline__ T lane_xor(T v, int m) { return __shfl_xor(v, m, WaveSize); }

// The full 6-step XOR butterfly: every lane ends holding `op` over all 64. NOT the merge's output-dim fold,
// which starts at LanesPerRow and is deliberately PARTIAL.
template <class T, class Op> __device__ __forceinline__ T wave_all_reduce(T v, Op op) {
  for (int m = 1; m < WaveSize; m <<= 1) v = op(v, lane_xor(v, m));
  return v;
}

// A C-fragment row repeats across the lane quads, so a fold over ROWS crosses only the top two lane bits. NOT
// __shfl_xor, which materialises both addresses and so holds 4 more VGPRs.
// 0x401F selects ds_swizzle's 32-lane XOR mode with mask 16.
__device__ inline float xor16(float v) { return bitcast<float>(__builtin_amdgcn_ds_swizzle(bitcast<int>(v), 0x401F)); }
// ds_bpermute addresses in BYTES, hence lane*4; +128 bytes is +32 lanes, and it wraps inside the 64-lane wave.
__device__ inline float xor32(float v) { return bitcast<float>(__builtin_amdgcn_ds_bpermute((__lane_id() << 2) + 128, bitcast<int>(v)));
}
__device__ inline float fold_max(float v) { v = fmaxf(v, xor16(v)); return fmaxf(v, xor32(v)); }
__device__ inline float fold_sum(float v) { v += xor16(v); return v + xor32(v); }

// One value per thread down to one per CTA, which every thread then reads back: a butterfly inside each wave,
// then one slot per wave through LDS. Each call needs its OWN slot array, or one call's read races the next
// call's write.
template <int Waves, class Op>
__device__ __forceinline__ float cta_reduce(float v, float* slot, int wave, int lane, Op op) {
  v = wave_all_reduce(v, op);
  if (lane == 0) slot[wave] = v;
  __syncthreads();
  float r = slot[0];
  for (int w = 1; w < Waves; ++w) r = op(r, slot[w]);
  return r;
}


// ── LDS layout ── 576 bf16 + 8 B pad, so KvRowBytes/4 == 2 (mod 32 banks): 16 rows, 16 distinct bank pairs.
constexpr int KvRowBytes = FusedDim * (int)sizeof(bf16) + 8, KvRowElems = KvRowBytes / (int)sizeof(bf16);
constexpr int KvSlotElems = TileTokens * KvRowElems, KvLdsElems = 2 * KvSlotElems;   // read tile h, fill h+1
constexpr int PvTokensPerLane = TileTokens / QuadSpan;
static_assert(KvRowBytes % 8 == 0, "ds_read_b64 / ds_write2_b64 need 8-B aligned rows");
static_assert(PvTokensPerLane == QuadSpan, "PV needs 4 tokens per lane for the register transpose");

// What a tile's finisher publishes for the other waves to read: probabilities as [parity][tile][lane], and
// lazy-rescale factors as [parity][row][tile].
constexpr int prob_plane_elems(int nt) { return 2 * nt * WaveSize; }
// The rescale row stride is a BANK constant, not a size. 16 rows each read one float, so the stride decides
// which of the 32 LDS banks they land on. A stride of 4 already gives 16 distinct banks and is the natural
// choice up to 4 tiles; 5 and 6 tiles need at least 6, and strides 6 through 9 all collide. 10 is the
// smallest conflict-free stride above 6, and is worth 3% over the natural 6.
constexpr int rescale_row_floats(int nt) { return nt <= 4 ? 4 : 10; }
constexpr int rescale_plane_floats(int nt) { return 2 * MfmaTile * rescale_row_floats(nt); }

// ── Wave map ── which query tile a wave contracts, over which of the 36 K-steps, and who finishes it.
//
// NTiles 4 and 6 give every query tile its own wave: 36 K-steps each, and nothing crosses waves. NTiles 3
// (q_len 4) would leave FIVE of the eight waves with no QK to do, so there each tile's 36 K-steps split
// across waves and the partial scores meet in an LDS plane. Tile 0 splits two ways; tiles 1 and 2 split
// three ways, because two of their waves also stage KV and so take a short slice:
//
//     wave       0       1       2       3       4       5       6       7
//     tile       0       0       1       2       1       1       2       2
//     K-steps  0..17   18..35  0..17   0..17   18..26  27..35  18..26  27..35
//     finishes   yes     no      yes     yes     no      no      no      no
//     also fills                                                 yes     yes
//
// The tile column looks arbitrary and is not. A wave runs on SIMD (wave % 4), and wall time is linear in the
// BUSIEST SIMD, so the map has to balance SIMDs, not waves. Read the table four columns apart: every SIMD
// gets one 18-step wave and one 9-step wave, 27 each. The obvious map -- tiles 0/1/2 taking waves
// (0,1)/(2,3,4)/(5,6,7) -- balances the waves just as well and still loses, because it piles 36 K-steps on
// SIMD 1 and only 18 on SIMD 3.
struct QkRole {
  int tile;                 // the query tile this wave contracts into
  int k_lo, k_n;            // its slice of the 36 K-steps; k_n == 0 means the wave contracts nothing
  int peer0, peer1;         // the waves holding the REST of this tile; -1 for "no such wave"
  bool finishes;            // does this wave sum the partials, run the softmax, and publish the tile?
};

__host__ __device__ constexpr QkRole qk_role(int ntiles, int w) {
  if (ntiles != 3) {                                 // one wave per query tile, all 36 K-steps, no peers
    const bool owns = w < ntiles;
    // `tile` is out of range on a wave that owns nothing, which is harmless: k_n == 0 means it is never read.
    return {.tile = w, .k_lo = 0, .k_n = owns ? QkSteps : 0, .peer0 = -1, .peer1 = -1, .finishes = owns};
  }
  switch (w) {              // the table above, one row per wave
    case 0:  return {0,  0, 18,  1, -1, true};
    case 1:  return {0, 18, 18, -1, -1, false};
    case 2:  return {1,  0, 18,  4,  5, true};
    case 3:  return {2,  0, 18,  6,  7, true};
    case 4:  return {1, 18,  9, -1, -1, false};
    case 5:  return {1, 27,  9, -1, -1, false};
    case 6:  return {2, 18,  9, -1, -1, false};
    default: return {2, 27,  9, -1, -1, false};
  }
}

// The rendezvous itself: one f32x4 per lane per wave, double-buffered by tile parity exactly like the
// probability planes. Only the five waves that do NOT finish a tile ever write one, but indexing by wave
// keeps every address a compile-time constant, and LDS is not the binding resource here.
constexpr int PeerPlaneFloats = WaveSize * QuadSpan;                  // one wave's plane, indexed by lane
constexpr int peer_plane_floats(int nt) { return nt == 3 ? 2 * NumWaves * PeerPlaneFloats : 0; }

// ── Row packing ── DPC positions x H heads pack POSITION-MAJOR into ceil(DPC*H/16) MFMA tiles of 16 rows.
// 8 draft positions x 12 heads is 96 rows, which is exactly 6 tiles of 16 -- and 6 is the ceiling, because
// waves 6 and 7 stage KV and so cannot also finish a query tile. 13 heads would need 7 tiles and does not fit.
constexpr int MaxPackedHeads = 12;
__host__ __device__ inline bool fits_single_pass(int H) { return H <= MaxPackedHeads; }

// One CTA per draft position, or all 8 positions in one CTA when the heads pack? This is deliberately a
// pure function of the SHAPE. An earlier version also gated on the B*S token total, which bought 6.4% at
// B*S ~ 64K and nothing above ~256K -- but the planner sees B*max_seq_len while a paged launch sees the
// ACTUAL total, and a ragged batch can straddle the threshold, so it needed a whole "recover the split from
// `parts`, never re-decide it" mechanism to stop the planner and the two launchers disagreeing. Dropping
// the gate deletes all of that, costs that one shape 6.4%, and is free or better everywhere else.
__host__ inline int draft_per_cta(int ql, int H) {
  return (ql >= MaxDraftPerCta && fits_single_pass(H)) ? MaxDraftPerCta : 4;
}
__host__ inline int qgroups_of(int ql, int H) { return ql / draft_per_cta(ql, H); }

__device__ __host__ __forceinline__ constexpr int live_tiles_of(int nact, int H) {
  return (nact * H + MfmaTile - 1) / MfmaTile;
}

// ── Partial workspace ── a bf16 [cta][slot][512] slab, then (run_max, run_sum) fp32 planes. Stride: widest H.
__device__ __host__ __forceinline__ size_t o_slab_floats(size_t nctas, int slots) {
  return (((size_t)nctas * slots * LatDim / 2)      // the O slab is bf16, but this whole workspace counts floats
          + 3) & ~(size_t)3;                        // round to 16 B so the fp32 planes after it stay f32x4-aligned
}
// Plane k of packed row `slot`: k == 0 is run_max, k == 1 is run_sum.
__device__ __host__ __forceinline__ size_t hdr_off(size_t nctas, int slots, int slot, int k) {
  return o_slab_floats(nctas, slots) + ((size_t)slot * 2 + k) * nctas;
}
// The whole workspace: the slab plus both planes of all `slots` rows.
__host__ inline size_t partials_floats(size_t nctas, int slots) {
  return o_slab_floats(nctas, slots) + (size_t)slots * 2 * nctas;
}

// ── KV split plan ── units go out in proportion to WORK. Decode and merge rederive it INDEPENDENTLY, so every
// reduction below is exact integer arithmetic.
// MaxParts is an ARRAY BOUND, not a tuning knob: the merge holds one weight per part in LDS, in a
// weight[MaxParts] array. MaxTilesPerRequest clamps one request to S <= 16.7M tokens.
constexpr int MaxParts = 384, MaxTilesPerRequest = 1 << 20;
// The merge runs 256 threads, NOT one per part: a 384-thread CTA is 6 waves, and at B=32 that costs the merge
// grid 8 us of occupancy. 256 threads means a thread carries more than one part, so both merge passes stride.
constexpr int MergeThreads = 256, MaxDimSplit = 16;
__device__ __host__ __forceinline__ int kv_tiles(int S) {
  const int t = (S + TileTokens - 1) / TileTokens;
  return t < 1 ? 1 : min(t, MaxTilesPerRequest);
}

// A request's QUERY weight, used to share out KV parts. Live query tiles ALONE is the obvious weight and is
// badly wrong -- it measured 0.44x -- because most of a KV tile's cost is the fill, which is the same whether
// a CTA owns one query tile or six. FillTileWeight is that fixed fill cost expressed in tile units, so a
// 1-tile and a 6-tile request come out about 1.3x apart rather than 6x.
constexpr int FillTileWeight = 14;

struct Split {
  const int* __restrict__ sl;    // seq_lens[B]: each request's KV length
  const int* __restrict__ ql;    // q_lens[B], or null: each request's LIVE draft window
  int q_len;                     // the PADDED draft width every request is launched at
  int H;
  int dpc;                       // draft positions per CTA; see draft_per_cta()
  long long tot;            // sum over requests of (tiles - 1) * weight: the proportional-split denominator
  int t0;                   // tiles in request 0 -- the shared tile count when `same`
  bool same;                // do ALL requests have the same tile count AND the same query weight?

  __device__ __forceinline__ int weight(int i) const {
    int q = ql ? ql[i] : q_len;                                   // clamped, and floored at 0 too:
    q = q < 0 ? 0 : (q > q_len ? q_len : q);                      // a negative weight breaks the floor proof
    return live_tiles_of(q < dpc ? q : dpc, H) + FillTileWeight;
  }

  __device__ __forceinline__ int units(int i, int B, int parts) const {
    if (same) return t0 < parts ? t0 : parts;      // a rectangular batch shares out evenly; see make_split
    // Every request is guaranteed one unit, so B of the U units are spoken for and U - B are left to share
    // out. Request i's share is proportional to (its tiles - 1) * its weight, and `tot` is exactly that
    // quantity summed over the batch. Integer division only ever rounds DOWN, so the units actually handed
    // out can never exceed the grid that was sized for them.
    const int U = B * parts, t = kv_tiles(sl[i]);
    long long n = 1;
    if (tot > 0 && U > B) n += ((long long)(t - 1) * weight(i) * (U - B)) / tot;
    if (n > t) n = t;                                             // never split a request past its own tiles
    return (int)(n > MaxParts ? MaxParts : n); }                  // weight[MaxParts] in the merge bounds this

  // One lane per request, a wave-wide prefix sum of their unit counts, then the first lane whose
  // running total passes `unit`. B MAY EXCEED ONE WAVE, so sweep the batch a wave at a time and carry
  // each finished sweep's total into `off`.
  __device__ __forceinline__ int owner(int B, int parts, int unit, int& off, int& n) const {
    if (same) {                                    // every request the same size: the sweep is a divide
      n = t0 < parts ? t0 : parts;
      const int i = unit / n;
      off = i * n;
      return i < B ? i : B;
    }
    const int l = (int)__lane_id();
    off = 0;
    n = 0;

    for (int base = 0; base < B; base += WaveSize) {
      const int i = base + l, mine = i < B ? units(i, B, parts) : 0;

      int upto = mine;                                            // units of requests base..i inclusive
      for (int m = 1; m < WaveSize; m <<= 1) {
        const int up = __shfl_up(upto, m, WaveSize);              // UNCONDITIONAL: a shuffle under a
        upto += (l >= m) ? up : 0;                                // divergent ternary reads an INACTIVE lane
      }

      const unsigned long long past = __ballot(upto > unit - off);
      if (past) {
        const int L = (int)__builtin_ctzll(past);
        n = __builtin_amdgcn_readlane(mine, L);
        off += __builtin_amdgcn_readlane(upto - mine, L);         // units before the owner
        return base + L;
      }
      off += __builtin_amdgcn_readlane(upto, WaveSize - 1);       // no owner here: carry the sweep's total
    }
    return B;                                                     // surplus unit: no request owns it
  }
};

// One lane-parallel sweep of seq_lens. BOTH kernels call this and must agree exactly, so the fast path has
// to be a property of the data, never of the caller.
//   The vote is `__ballot(same == 0) == 0`, not `__ballot(same)`: an inactive lane contributes 0 to a ballot,
// which under "does anyone disagree?" reads as "does not disagree" -- the safe answer. The other spelling
// would let a partial exec mask silently vote the batch uniform.
__device__ __forceinline__ Split make_split(const int* __restrict__ sl, const int* __restrict__ ql,
                                            int q_len, int H, int dpc, int B) {
  Split s{sl, ql, q_len, H, dpc, 0, kv_tiles(sl[0]), false};
  const int w0 = s.weight(0);
  long long tot = 0;
  int agrees = 1;
  for (int i = (int)__lane_id(); i < B; i += WaveSize) {
    const int t = kv_tiles(sl[i]), w = s.weight(i);
    tot += (long long)(t - 1) * w;
    agrees &= (t == s.t0) & (w == w0);
  }
  if (__ballot(agrees == 0) == 0ull) {               // nobody disagreed: `tot` is dead on this path
    s.same = true;
    return s;
  }
  tot = wave_all_reduce(tot, [](long long x, long long y) { return x + y; });
  const unsigned lo = __builtin_amdgcn_readfirstlane((int)(unsigned)tot);   // no 64-bit readfirstlane
  const unsigned hi = __builtin_amdgcn_readfirstlane((int)(unsigned)((unsigned long long)tot >> 32));
  s.tot = (long long)(((unsigned long long)hi << 32) | lo);
  return s;
}

// ── Where this CTA sits ── resolved BEFORE the wave bodies so they can be SPECIALISED on it (13%).
struct CtaMap {
  int b;            // which request in the batch
  int qgrp;         // which group of draft positions, when the window needs more than one CTA
  int part;         // which KV part of that request, and how many parts it was split into
  int nparts;
  int slab_cta;     // this CTA's row in the partials slab: units are dense, each carrying QGroups rows
  int ql_b;         // this request's live draft window, <= the padded q_len
  int nact;         // how many of this CTA's DPC positions are live
  int live_tiles;   // ... and therefore how many of the NTiles query tiles carry real rows
};
template <int QL, int DPC, bool Paged>
__device__ __forceinline__ bool map_cta(int B, int H, int parts, int q_len, const int* __restrict__ seq_lens,
                                        const int* __restrict__ q_lens, CtaMap& m) {
  constexpr int QGroups = QL / DPC, XCDs = 8;         // CTAs per (request, kv-part); 1 == single pass

  // MI300X gives consecutive blockIdx to consecutive XCDs, and the groups of one unit read the same KV.
  const int cta = blockIdx.x;
  int qgrp = 0, unit = cta;
  if constexpr (QGroups > 1) {
    constexpr int Block = XCDs * QGroups;              // one full round-robin over the XCDs per group
    const int whole = (int)(gridDim.x / Block) * Block;
    if (cta < whole) {
      const int xcd = cta % XCDs, round = cta / XCDs;
      qgrp = round % QGroups;
      unit = (round / QGroups) * XCDs + xcd;
    } else {                                           // the ragged tail cannot be swizzled: in order
      const int t = cta - whole;
      qgrp = t % QGroups;
      unit = whole / QGroups + t / QGroups;
    }
  }

  int part, b, nparts = parts;
  if constexpr (Paged) {
    int off = 0, n = 0;
    b = make_split(seq_lens, q_lens, q_len, H, DPC, B).owner(B, parts, unit, off, n);
    if (b >= B) return false;                          // surplus unit: the plan wanted fewer CTAs than the grid
    part = unit - off;
    nparts = n;
  } else {
    part = unit % parts;
    b = unit / parts;
  }

  const int ql_b = (Paged && q_lens) ? min(q_lens[b], q_len) : q_len;   // masks only: strides keep PADDED q_len
  const int nact = min(DPC, max(0, ql_b - qgrp * DPC));                 // draft positions this CTA owns
  const int live = __builtin_amdgcn_readfirstlane(live_tiles_of(nact, H));   // scalar: CTA-uniform
  m = CtaMap{b, qgrp, part, nparts, unit * QGroups + qgrp, ql_b, nact, live};

  return live > 0;
}

struct KArgs {
  const bf16* __restrict__ q_lat;        // [B, q_len, H, 512] bf16
  const bf16* __restrict__ q_pe;         // [B, q_len, H, 64]  bf16
  const fp8_t* __restrict__ c_kv;        // [B, S, 576] slab, or a [num_slots, 576] pool when paged
  float* __restrict__ partials;          // the split workspace; the merge turns it into o_lat

  const int* __restrict__ seq_lens;      // paged, all four: [B] KV lengths
  const int* __restrict__ rows;          // [B] batch row per request, or null for the identity
  const int* __restrict__ kv_indices;    // kv_indices[kv_indptr[b] + t] is token t of request b
  const int* __restrict__ kv_indptr;     // [B + 1] offsets into kv_indices

  int B, H, S_scalar, q_len;
  float scale_log2e;                     // the softmax scale, already folded with log2(e) for exp2
  float kv_scale;                        // the fp8 KV dequantisation scale
};
// The five LDS regions, gathered so a wave can be handed all of them at once.
struct Smem {
  bf16* kv;         // [2][16 tokens][576] bf16: the tile being read, and the tile being filled
  bf16x4* prob;     // [parity][query tile][lane]: probabilities a tile's finisher publishes
  float* resc;      // [parity][row][query tile]: this row's lazy-rescale factor
  float* peer;      // [parity][wave][lane]: partial score columns, when a query tile splits across waves
  u32x2* moved;     // [parity]: one byte per query tile, nonzero if that tile rescaled
};

__device__ __forceinline__ f32x4 peer_read(const Smem& sh, int parity, int w, int slot) {
  return *(const f32x4*)&sh.peer[(parity * NumWaves + w) * PeerPlaneFloats + slot];
}
__device__ __forceinline__ void peer_write(const Smem& sh, int parity, int w, int slot, f32x4 v) {
  *(f32x4*)&sh.peer[(parity * NumWaves + w) * PeerPlaneFloats + slot] = v;
}

// ── The KV ring ── a filler wave's 8 tokens of tile h+1: HBM fp8 -> registers -> bf16 -> LDS. The lane map is
// ROW-MAJOR -- lane l owns token row l/8 -- so a lane op NEVER straddles a row.
constexpr int FillLanesPerRow = WaveSize / FillTok, FillOps = FusedDim / (FillLanesPerRow * 8);
constexpr int SrcStride = FillLanesPerRow * 8;       // one pass over the row, in fp8 source bytes
constexpr int DstStride = 2 * SrcStride;             // the same dims once widened to bf16
static_assert(FillLanesPerRow * 8 * FillOps == FusedDim, "the fill must tile a token row exactly");
template <bool Paged, bool NonTemporal> struct Ring {
  const fp8_t* pool;                                 // the fp8 KV: one slab per request, or a shared pool
  const int* pool_index;                             // paged only: logical token -> pool row
  char* lds;                                         // the two-slot KV staging buffer

  int kv_lo, kv_hi;                                  // the token range this CTA's KV part covers
  int ntiles;                                        // ... and how many 16-token tiles that is
  int pool_base;                                     // paged only: where this request's index run starts
  int wave_tok0;                                     // the first token of the half-tile THIS wave stages

  u32x2 regs[2][FillOps];                            // two tiles of raw fp8, still in flight from HBM
  int row;                                           // this lane's token row inside the wave's half-tile
  int pool_row;                                      // that token's row in the pool, resolved a phase ahead
  uint32_t hbm_byte;                                 // this lane's 8-byte slice of the row: source offset
  uint32_t lds_byte;                                 // ... and its destination, which is twice as wide in bf16

  // Point this wave at the KV part it streams, and at the half of each tile it is responsible for.
  __device__ __forceinline__ void bind(const fp8_t* kv, const int* kv_index, bf16* kv_lds,
                                       int lo, int hi, int n, int base, int tok0, int lane) {
    pool = kv;
    pool_index = kv_index;
    lds = (char*)kv_lds;
    kv_lo = lo;
    kv_hi = hi;
    ntiles = n;
    pool_base = base;
    wave_tok0 = tok0;
    row = lane / FillLanesPerRow;
    hbm_byte = (uint32_t)(lane % FillLanesPerRow) * 8;
    lds_byte = (uint32_t)((tok0 + row) * KvRowBytes + hbm_byte * 2);
  }

  // Resolve this lane's token to a pool row, one phase ahead of the load that uses it.
  //   The `it >= ntiles` guard is the same one load() and store() use, and it discharges every bound:
  // it < ntiles gives tok0 = kv_lo + it*TileTokens < kv_hi, hence n_live >= 1; and the clamped index is at
  // most tok0 + n_live - 1 < kv_hi <= S. Safe only because every resolve_row(it) is paired with a load(it)
  // for the SAME it, so a skipped row is never read.
  __device__ __forceinline__ void resolve_row(int it) {
    if (it >= ntiles) return;
    const int tok0 = kv_lo + it * TileTokens, n_live = min(TileTokens, kv_hi - tok0);
    const int mine = wave_tok0 + row;                          // this lane's token within the tile
    const int tok = tok0 + (mine < n_live ? mine : n_live - 1);   // a tail tile clamps onto its last token
    pool_row = Paged ? pool_index[pool_base + tok] : tok;
  }

  // The loads stay COMPILER-VISIBLE: LLVM does not model inline-asm vmcnt and would hoist the convert past a wait.
  __device__ __forceinline__ void load(int it, int buf) {
    if (it >= ntiles) return;
    const fp8_t* src = pool + (size_t)pool_row * FusedDim + hbm_byte;
    #pragma unroll
    for (int j = 0; j < FillOps; ++j) {
      const fp8_t* q = src + j * SrcStride;
      regs[buf][j] = NonTemporal ? __builtin_nontemporal_load((const u32x2*)q) : *(const u32x2*)q;
    }
  }

  // Registers -> LDS, widening fp8 to bf16 on the way. Each 8-byte load becomes 16 bytes, so the
  // destination stride is twice the source stride, and the two source dwords land 8 bytes apart.
  __device__ __forceinline__ void store(int it, int buf, int slot) {
    if (it >= ntiles) return;
    char* dst = lds + slot * (KvSlotElems * (int)sizeof(bf16)) + lds_byte;
    #pragma unroll
    for (int j = 0; j < FillOps; ++j)
      for (int i = 0; i < 2; ++i)
        *(u32x2*)(dst + j * DstStride + i * 8) = cvt4_bf16(regs[buf][j][i]);
  }
};

// ── The QK contraction ── this wave's Steps of the 36 K-steps of 16, chunked so the next chunk's LDS reads
// hide under this chunk's MFMAs. A wave that contracts nothing instantiates Steps == 1 and is dead-coded away.
template <int Steps, int Chunk> struct Qk {
  static_assert(Steps % Chunk == 0, "a chunk that does not divide this wave's K-steps silently drops steps");
  bf16x4 q[Steps];                                   // this lane's slice of the query row, held for the walk
  u32x2 k[Chunk];                                    // one chunk of the KV tile, in flight from LDS
  f32x4 scores = {};
  const bf16* row = nullptr;                         // this lane's row of LDS slot 0; slot P sits KvSlotElems on

  // The first 32 K-steps walk the latent half of the row, the last 4 the rope half, so a slice that starts
  // at k_lo just keeps counting: step kg lives in q_lat when kg < 32 and in q_pe otherwise.
  __device__ __forceinline__ void load_q(const bf16* ql, const bf16* qp, int k_lo, int quad, bool on) {
    const bf16x4 zero = {};                          // a masked row must contract to nothing
    #pragma unroll
    for (int i = 0; i < Steps; ++i) {
      const int kg = k_lo + i;
      const bf16* src = (kg < LatIters) ? (ql + (size_t)kg * KQ) : (qp + (size_t)(kg - LatIters) * KQ);
      const bf16x4 f = *(const bf16x4*)(src + quad * QLane);
      q[i] = on ? f : zero;
    }
  }

  __device__ __forceinline__ void prefetch(int parity, int k0 = 0) {
    const bf16* p = row + parity * KvSlotElems;
    #pragma unroll
    for (int i = 0; i < Chunk; ++i) k[i] = *(const u32x2*)(p + (k0 + i) * KQ);
  }

  __device__ __forceinline__ void contract(int parity) {
    f32x4 acc = {};
    #pragma unroll
    for (int c = 0; c < Steps / Chunk; ++c) {
      if (c > 0) prefetch(parity, c * Chunk);
      #pragma unroll
      for (int i = 0; i < Chunk; ++i) acc = mfma_bf16(bitcast<bf16x4>(k[i]), q[c * Chunk + i], acc);
    }
    scores = acc;
  }
};

// ── Online softmax ── a KV tile's scores fold into this row's running (max, sum). The rescale is LAZY.
struct Online {
  // What a tile's finisher hands to the other waves for this KV tile.
  struct Result {
    bf16x4 prob;      // this lane's four probabilities, lifted and packed to bf16
    float rescale;    // the factor every accumulator of this row must be scaled by first
    bool moved;       // whether the running max moved at all -- wave-uniform, see the ballot below
  };
  float run_max = NegInf, run_sum = 0.f;             // this row's running softmax state
  float scale;                                       // kv_scale * log2(e): folds into the exp2 argument
  float lift;                                        // log2 of the probability lift, or NegInf on a dead row
  int end;                                           // the last KV token this lane's row may see
  int quad;

  __device__ __forceinline__ Result tile(f32x4 s, int kv0) {
    const int live_tokens = end - kv0;               // a tail tile, or a causal limit, can cut this short
    float tile_max = NegInf;
    for (int e = 0; e < QuadSpan; ++e) {
      const int my_token = quad * PvTokensPerLane + e;   // this lane's e-th token inside the 16-token tile
      s[e] = (my_token >= live_tokens) ? NegInf : s[e];
    }
    for (int e = 0; e < QuadSpan; ++e) tile_max = fmaxf(tile_max, s[e]);
    tile_max *= scale;

    // WAVE-UNIFORM by construction: if any lane in this row would overflow its slack, the whole wave
    // rescales. Out::any_moved depends on that -- a lane whose own factor came out 1.0 must still take
    // the branch, or it would apply a stale accumulator against a rescaled probability.
    const bool moved = (__ballot(tile_max > run_max + RescaleSlack) != 0ull);
    const float new_max = __builtin_expect(moved, 0) ? fmaxf(run_max, fold_max(tile_max)) : run_max;
    const float eb = lift - new_max;
    float tile_sum = 0.f, prob[QuadSpan];
    for (int e = 0; e < QuadSpan; ++e) {
      prob[e] = __builtin_amdgcn_exp2f(__builtin_fmaf(s[e], scale, eb));
      tile_sum += prob[e];
    }
    const bf16x4 packed = pack4<Rna>(prob[0], prob[1], prob[2], prob[3]);
    tile_sum *= (1.0f / ProbLift);

    if (!moved) {                                    // the common path: no rescale, nothing to undo
      run_sum += tile_sum;
      return {packed, 1.0f, false};
    }
    // First tile of this row: run_max is still -inf, and -inf minus new_max is NaN when the row is fully
    // masked and new_max is -inf too. The accumulator is zero at this point, so 0.f is right either way.
    const float rescale = (run_max == NegInf) ? 0.f : __builtin_amdgcn_exp2f(run_max - new_max);
    run_sum = run_sum * rescale + tile_sum;
    run_max = new_max;
    return {packed, rescale, true};
  }
};

// ── PV accumulator ── this wave's 64 output dims of EVERY query tile, so one V fetch feeds all their MFMAs.
template <int NTiles> struct Out {
  static constexpr int DimTiles = PvChunks / NumWaves, VPairs = DimTiles / 2;
  static constexpr int RsRowF = rescale_row_floats(NTiles);
  static_assert(DimTiles == QuadSpan, "the epilogue packs one tile's dim tiles into one bf16x8");
  f32x4 acc[DimTiles * NTiles] = {};                 // this wave's 64 dims of every query tile
  uint32_t v[2][VPairs][PvTokensPerLane] = {};       // the V fragment: [LDS parity][dim pair][token]
  const bf16x4* prob;
  const float* resc;
  const u32x2* moved;
  // Wave w owns output dims [64w, 64w + 64). Inside that block the four accumulators are INTERLEAVED, not
  // four contiguous 16-dim runs: accumulator c of MFMA row M holds dim 64w + 4*M + c. That is why store()
  // packs acc[0..3] side by side, and why col_bytes scales the row index by DimTiles.
  int dim0;                                          // the first of the 64 output dims this wave owns
  int col_bytes;                                     // byte offset of this lane's V column inside a KV row
  int row, quad, lane;

  __device__ __forceinline__ void bind(const Smem& sh, int wave, int mrow, int q, int l) {
    prob = sh.prob;
    resc = sh.resc;
    moved = sh.moved;
    dim0 = wave * (DimTiles * MfmaTile);
    col_bytes = (dim0 + DimTiles * mrow) * (int)sizeof(bf16);
    row = mrow;
    quad = q;
    lane = l;
  }

  // Did ANY query tile move its running max on the step that published plane P? One b64 at a WAVE-uniform
  // address, one byte per tile. Without it the rescale below is NTiles separate per-lane compares and
  // branches, every step, on a path that almost never runs. The byte comes from the wave-uniform decision
  // in Online::tile, never from a lane's own factor -- a lane whose factor happens to be 1.0 on a tile that
  // DID move must still take the branch.
  __device__ __forceinline__ bool any_moved(int parity) const {
    const u32x2 m = moved[parity];
    return (m[0] | m[1]) != 0u;
  }

  __device__ __forceinline__ void fetch(int parity, const char* slot) {
    for (int j = 0; j < PvTokensPerLane; ++j) {
      const u32x2 d = *(const u32x2*)(slot + (quad * PvTokensPerLane + j) * KvRowBytes + col_bytes);
      v[parity][0][j] = d[0];                             // one b64 is exactly VPairs == 2 dim pairs
      v[parity][1][j] = d[1];
    }
  }

  // This row's factors for plane P, read as f32x2 pairs. Read HERE, well ahead of the apply in consume(),
  // so the LDS latency is spent on the PV MFMAs in between rather than exposed.
  __device__ __forceinline__ void read_rescale(int parity, float* out_rescale) const {
    const float* row_base = resc + (parity * MfmaTile + row) * RsRowF;
    #pragma unroll
    for (int i = 0; i < (NTiles + 1) / 2; ++i) {
      const f32x2 pair = *(const f32x2*)(row_base + 2 * i);
      out_rescale[2 * i] = pair[0];
      if (2 * i + 1 < NTiles) out_rescale[2 * i + 1] = pair[1];
    }
  }

  __device__ __forceinline__ void consume(int parity, const float* rescale, bool any, int live) {
    bf16x4 pr[NTiles];
    for (int t = 0; t < NTiles; ++t) pr[t] = prob[(parity * NTiles + t) * WaveSize + lane];
    if (__builtin_expect(any, 0))                    // rescale[t] is exactly 1.0f on a tile that stood still
      #pragma unroll
      for (int t = 0; t < NTiles; ++t)
        if (__builtin_expect(rescale[t] != 1.0f, 0))
          #pragma unroll
          for (int c = 0; c < DimTiles; ++c)
            for (int e = 0; e < QuadSpan; ++e)
              acc[t * DimTiles + c][e] = mul_f32(acc[t * DimTiles + c][e], rescale[t]);

    for (int hh = 0; hh < VPairs; ++hh) {
      uint32_t d[QuadSpan];
      transpose_4tok(v[parity][hh], d);
      for (int t = 0; t < NTiles; ++t)
        if (t < live)
          for (int p = 0; p < 2; ++p) {
            f32x4& a = acc[t * DimTiles + 2 * hh + p];
            a = mfma_bf16(as_bf16x4(d[2 * p], d[2 * p + 1]), pr[t], a);
      }
    }
  }

  __device__ __forceinline__ void store(bf16* dst, int t, float sc) const {
    const f32x4* p = &acc[t * DimTiles];
    for (int e = 0; e < QuadSpan; e += 2) *(bf16x8*)&dst[dim0 + DimTiles * (QuadSpan * quad + e)] =
          pack8(p[0][e] * sc, p[1][e] * sc, p[2][e] * sc, p[3][e] * sc,
                p[0][e + 1] * sc, p[1][e + 1] * sc, p[2][e + 1] * sc, p[3][e + 1] * sc);
  }
};

// ── One wave's whole job ── two LDS slots and a five-stage software pipeline over the KV tiles; the
// pipeline itself is laid out in the comment above `step` below.
template <int DPC, bool Paged, bool NonTemporal, int NTiles, int W, bool Ragged>
__device__ __forceinline__ void wave_body(const KArgs& a, const Smem& sh, const CtaMap& m) {
  constexpr QkRole Role = qk_role(NTiles, W);      // see the wave-map table above
  constexpr int Tile = Role.tile, KLo = Role.k_lo, KN = Role.k_n;
  constexpr bool Contracts = (KN > 0), Finishes = Role.finishes, Fills = (W >= NumWaves - FillWaves);
  constexpr int Slab = MaxHeads * DPC, SlotBytes = KvSlotElems * (int)sizeof(bf16);
  constexpr int RsRowF = Out<NTiles>::RsRowF;
  // 6 query tiles mean 6 PV accumulators, so a chunk of 18 would spill there; a 9-step slice can only chunk 9.
  constexpr int Chunk = (KN == 0) ? 1 : ((NTiles == 6) ? 12 : (KN >= 18 ? 18 : KN));
  const int lane = threadIdx.x & (WaveSize - 1), wave = threadIdx.x / WaveSize;
  const int mrow = lane & (MfmaTile - 1), quad = lane / MfmaTile;
  const int H = a.H, ql_b = m.ql_b;

  const int live = Ragged ? m.live_tiles : NTiles;
  // Does this wave's query tile carry any real rows? readfirstlane, or Tile is not provably wave-uniform to
  // LLVM and the skip becomes an exec mask instead of a scalar branch. Peers share a tile with their
  // finisher, so this single flag turns all of a tile's waves off together.
  const bool tile_live = Contracts && (!Ragged || __builtin_amdgcn_readfirstlane(Tile) < live);

  const int S = Paged ? a.seq_lens[m.b] : a.S_scalar, slot0 = Paged ? a.kv_indptr[m.b] : 0;
  // rows[b] is the real batch row this request's Q lives on. It indexes a LOAD here, and the merge writes
  // O back to the same row; clamped rather than trusted, so a bad index cannot fault.
  const int q_row = (Paged && a.rows) ? min(max(a.rows[m.b], 0), a.B - 1) : m.b;
  const int part_tokens = ((S + m.nparts * TileTokens - 1) / (m.nparts * TileTokens)) * TileTokens;
  const int kv_lo = m.part * part_tokens, kv_hi = min(S, kv_lo + part_tokens);
  // A part that lands entirely past the end of S has ntiles == 0. It still runs one step and still writes
  // its slab -- an all-zero O with a (NegInf, 0) header -- which the merge then weights to nothing.
  const int ntiles = (kv_hi - kv_lo + TileTokens - 1) / TileTokens;

  const int prow = Tile * MfmaTile + mrow;           // this lane's packed row = position*H + head
  const int live_rows = H * m.nact;
  const bool active = Contracts && prow < live_rows;
  // Causal inside the draft window: position p sees KV [0, S - (ql_b-1-p)). Per lane, as a tile spans positions.
  const int qpos = m.qgrp * DPC + prow / H;          // the draft position this lane's row belongs to
  const int tok_lim = active ? (S - (ql_b - 1 - qpos)) : S;

  constexpr int FillHalf = Fills ? W - (NumWaves - FillWaves) : 0;   // which half of the tile this wave stages
  Ring<Paged, NonTemporal> ring;
  if constexpr (Fills) {
    // Contiguous KV is one slab per request; paged KV is one shared pool indexed through kv_indices.
    const fp8_t* kv = Paged ? a.c_kv : a.c_kv + (size_t)m.b * S * FusedDim;
    ring.bind(kv, a.kv_indices, sh.kv, kv_lo, kv_hi, ntiles, slot0, FillHalf * FillTok, lane);
  }

  Qk<Contracts ? KN : 1, Chunk> qk;
  const int peer_off = lane * QuadSpan;              // this lane's f32x4 inside a rendezvous plane
  if constexpr (Contracts) {
    qk.row = &sh.kv[mrow * KvRowElems + KLo * KQ + quad * QLane];
    const size_t qh = (size_t)(q_row * a.q_len + m.qgrp * DPC) * H + (active ? prow : 0);
    qk.load_q(a.q_lat + qh * LatDim, a.q_pe + qh * RopeDim, KLo, quad, active);
  }

  const int kv_end = min(kv_hi, tok_lim);            // the last KV token this lane's row may see
  const bool row_dead = (kv_lo >= kv_end);           // this row's causal window misses the part entirely
  // `lift` MUST be NegInf on a dead row: every score is masked to NegInf, so `lift - new_max` would come
  // out at +1e38 and exp2 would return inf instead of the 0 the row needs.
  Online soft;
  soft.scale = a.kv_scale * a.scale_log2e;
  soft.lift = row_dead ? NegInf : ProbLiftLog2;
  soft.end = kv_end;
  soft.quad = quad;

  Out<NTiles> out;
  out.bind(sh, wave, mrow, quad, lane);

  // ── One step of the pipeline ── `kv_tile` is the tile being READ this step, and `parity` is its LDS
  // slot, kv_tile & 1;
  // tile kv_tile+1 fills the other slot meanwhile, which is why there are two slots and one barrier.
  //
  // A step deliberately has FIVE tiles in flight at once, at five different stages. That is what pays for
  // the latency: nothing here ever waits on a load it issued itself.
  //
  //   kv_tile-2   its probabilities were published last step -- PV-accumulate it now    out.consume
  //   kv_tile-1   its scores are sitting in registers -- soft-max and publish them      sm.tile
  //   kv_tile     read K and V out of the LDS slot, and contract its scores  qk.contract / out.fetch
  //   kv_tile+1   already in the filler waves' registers -- drain it into the other slot  ring.store
  //   kv_tile+3   issue its HBM loads, three steps before anything waits on them        ring.load
  //
  // So a tile is read at its own step, soft-maxed one step later and PV-accumulated one after that: the
  // loop runs one step PAST the last tile, and even then the final PV falls out after the loop.
  auto step = [&](int kv_tile, int parity) {
    float rescale[NTiles];                           // one lazy-rescale factor per query tile
    out.read_rescale(parity, rescale);
    const bool any_rescaled = out.any_moved(parity);      // issued WITH the factors, never as a gate on them
    if constexpr (Fills) ring.resolve_row(kv_tile + 3);

    // The peers' halves of tile kv_tile-1 go FIRST: LDS returns in order, so the softmax's wait is already
    // satisfied by the time the prefetch below lands. `finishing` is wave-uniform, so this is a scalar branch.
    const bool finishing = Finishes && tile_live && kv_tile > 0;
    if (finishing) {
      if constexpr (Role.peer0 >= 0) qk.scores = qk.scores + peer_read(sh, 1 - parity, Role.peer0, peer_off);
      if constexpr (Role.peer1 >= 0) qk.scores = qk.scores + peer_read(sh, 1 - parity, Role.peer1, peer_off);
      // Pin the CONSUMER of those reads: without it the softmax's s_waitcnt can float above them.
      if constexpr (Role.peer0 >= 0) asm("" : "+v"(qk.scores));
    }
    if (tile_live) qk.prefetch(parity);
    if (finishing) {
      const Online::Result r = soft.tile(qk.scores, kv_lo + (kv_tile - 1) * TileTokens);
      sh.prob[((1 - parity) * NTiles + Tile) * WaveSize + lane] = r.prob;
      if (quad == 0) sh.resc[((1 - parity) * MfmaTile + mrow) * RsRowF + Tile] = r.rescale;
      // UNCONDITIONAL, so a byte set by one tile can never survive into a step where that tile stood still.
      if (lane == 0) ((unsigned char*)&sh.moved[1 - parity])[Tile] = (unsigned char)r.moved;
    }

    out.consume(parity, rescale, any_rescaled, live);
    out.fetch(parity, (const char*)sh.kv + parity * SlotBytes);
    if (tile_live) qk.contract(parity);
    // A peer's whole output is its partial score column; its finisher adds it at the next step.
    if constexpr (Contracts && !Finishes) if (tile_live) peer_write(sh, parity, W, peer_off, qk.scores);
    // The register buffer IS the parity: tile kv_tile+1 was loaded into buffer parity three steps back, and once it
    // has drained into LDS that buffer is free for tile kv_tile+3, which has the same parity. Swapping these two
    // lines silently corrupts the KV stream.
    if constexpr (Fills) {
      ring.store(kv_tile + 1, parity, 1 - parity);
      ring.load(kv_tile + 3, parity);
    }
    tile_barrier();
  };

  if constexpr (Fills) {
    ring.resolve_row(0);
    ring.load(0, 0);
    ring.store(0, 0, 0);                             // tile 0 drains buffer 0 into LDS slot 0
    ring.resolve_row(1);
    ring.load(1, 0);                                 // tile 1 refills buffer 0; step 0 drains it into slot 1
    ring.resolve_row(2);
    ring.load(2, 1);                                 // tile 2 fills buffer 1; step 1 drains it into slot 0
  }
  { const int tid = (int)threadIdx.x;               // a tile that never publishes must read prob 0, rescale 1
    for (int i = tid; i < prob_plane_elems(NTiles); i += CtaThreads) sh.prob[i] = bf16x4{};
    if (tid < rescale_plane_floats(NTiles)) sh.resc[tid] = 1.0f;
    if (tid < 2) sh.moved[tid] = u32x2{};            // no tile has moved yet
  }
  tile_barrier();

  // Hand-unrolled by two so the parity stays a COMPILE-TIME constant. It has to: the V fragment and the
  // fill registers are register arrays, and a runtime index into either sends them to scratch memory. A
  // `#pragma unroll 2` would leave the remainder loop's parity unknown and do exactly that.
  step(0, 0);
  int h = 1;
  #pragma unroll 1
  for (; h < ntiles; h += 2) { step(h, 1);  step(h + 1, 0); }
  if (h <= ntiles) step(h, 1);                       // the tail step is always odd: the pairs start at h == 1
  // The last tile's V was fetched at step ntiles-1 and its probabilities published at step ntiles, so its
  // PV is still outstanding, in plane (ntiles + 1) & 1. Same reason as above: the parity has to reach the
  // lambda as a literal, so the branch is written out rather than computed.
  auto final_pv = [&](int parity) {
    float rescale[NTiles];
    out.read_rescale(parity, rescale);
    out.consume(parity, rescale, out.any_moved(parity), live);
  };
  if (ntiles > 0) {
    if (ntiles & 1) final_pv(0);
    else            final_pv(1);
  }

  // run_max is already row-uniform -- fold_max inside Online::tile did that. run_sum is still per-quad,
  // so fold it across the row before quad 0 publishes the pair.
  if constexpr (Finishes) soft.run_sum = fold_sum(soft.run_sum);
  for (int t = 0; t < NTiles; ++t) {
    const int slot = t * MfmaTile + mrow;
    if (slot >= live_rows) continue;
    // kv_scale here dequantises V; the copy of it in soft.scale dequantises K. 1/ProbLift undoes the 2^7
    // probability lift -- the same divide Online::tile already applied to run_sum, so O and the
    // denominator the merge divides by stay consistent.
    out.store((bf16*)a.partials + ((size_t)m.slab_cta * Slab + slot) * LatDim, t,
              a.kv_scale * (1.0f / ProbLift));
  }
  if constexpr (Finishes) if (quad == 0 && active) {
      const int slot = prow, nctas = (int)gridDim.x;
      a.partials[hdr_off(nctas, Slab, slot, 0) + m.slab_cta] = soft.run_max;
      a.partials[hdr_off(nctas, Slab, slot, 1) + m.slab_cta] = soft.run_sum;
  }
}

// A wave's whole role -- its query tile, its K-steps, whether it fills -- is a TEMPLATE parameter, so the
// wave map folds away entirely and each wave runs straight-line code with no map lookups in it. The price is
// that the wave index has to be turned back into a template argument, which only a peeled chain can do.
template <int DPC, bool Paged, bool NonTemporal, int NTiles, bool Ragged, int W = 0>
__device__ __forceinline__ void dispatch_wave(int wave, const KArgs& a, const Smem& sh, const CtaMap& m) {
  if constexpr (W < NumWaves) {
    if (wave == W) wave_body<DPC, Paged, NonTemporal, NTiles, W, Ragged>(a, sh, m);
    else dispatch_wave<DPC, Paged, NonTemporal, NTiles, Ragged, W + 1>(wave, a, sh, m);
  }
}

// ── The decode kernel ── LDS, the per-wave dispatch, and the CTA-level ragged arm ──────────────
template <int QL, int DPC, bool Paged, bool NonTemporal, int NTiles>
__global__ void __launch_bounds__(CtaThreads, 1) mla_decode_a16w8_multiq_kernel(KArgs a, int parts, const int* __restrict__ q_lens) {
  static_assert(NTiles <= NumWaves, "one wave owns at most one query tile");
  static_assert(KvLdsElems * (int)sizeof(bf16) + prob_plane_elems(NTiles) * (int)sizeof(bf16x4)
                + rescale_plane_floats(NTiles) * 4 + peer_plane_floats(NTiles) * 4 <= 65536, "LDS");

  __shared__ bf16   sKv[KvLdsElems];
  __shared__ bf16x4 sProb[prob_plane_elems(NTiles)];
  __shared__ __align__(16) float sRescale[rescale_plane_floats(NTiles)];
  // Empty unless NTiles == 3 splits tiles across waves; a zero-length array is not a type, hence the 1.
  __shared__ __align__(16) float sPeer[peer_plane_floats(NTiles) ? peer_plane_floats(NTiles) : 1];
  __shared__ u32x2 sMoved[2];                        // per parity, one byte per query tile: did it rescale?
  static_assert(NTiles <= 8, "sMoved holds one byte per query tile");
  const Smem sh{sKv, sProb, sRescale, sPeer, sMoved};

  CtaMap m;
  if (!map_cta<QL, DPC, Paged>(a.B, a.H, parts, a.q_len, a.seq_lens, q_lens, m)) return;

  // The fillers must run ahead of their consumers; moving or dropping this costs 1.4%.
  if (threadIdx.x / WaveSize >= NumWaves - FillWaves) asm volatile("s_setprio 1" ::: "memory");

  // ONE missing tile still takes the dense arm: its rows are all masked, so it contracts zeros and the
  // epilogue's row guard drops its garbage PV. Only two or more missing tiles earn the Ragged arm.
  const bool dense = (m.live_tiles + 1 >= NTiles);
  const int wave = threadIdx.x / WaveSize;
  if (dense) dispatch_wave<DPC, Paged, NonTemporal, NTiles, false>(wave, a, sh, m);
  else       dispatch_wave<DPC, Paged, NonTemporal, NTiles, true >(wave, a, sh, m);
}

// ── Merge kernel ── every decode CTA left one (run_max, run_sum, O) per query row; rebase onto a common max
// and sum. One CTA per (query row, output-dim slice). 256 threads against up to 384 parts, so a thread
// carries more than one part and both passes stride.
template <int QL, int DPC, int DimSplit>
__global__ void __launch_bounds__(MergeThreads) mla_decode_a16w8_merge(
    const float* __restrict__ partials, bf16* __restrict__ o_lat, int H, int parts, int q_len, int nctas,
    const int* __restrict__ rows, int B, const int* __restrict__ seq_lens, const int* __restrict__ q_lens) {
  constexpr int Threads = MergeThreads, Waves = Threads / WaveSize, QGroups = QL / DPC;
  constexpr int PartsPerThread = (MaxParts + Threads - 1) / Threads;   // parts outnumber threads, so stride
  constexpr int SlabSlots = MaxHeads * DPC, DimsPerCta = LatDim / DimSplit, DimsPerLane = 8;
  constexpr int LanesPerRow = DimsPerCta / DimsPerLane, PartsPerWave = WaveSize / LanesPerRow;
  constexpr int PartsPerCta = Waves * PartsPerWave;
  static_assert(LanesPerRow * DimsPerLane == DimsPerCta && PartsPerWave * LanesPerRow == WaveSize);
  static_assert(Threads % WaveSize == 0, "the merge folds through one LDS slot per wave");

  const int tid = threadIdx.x, lane = tid % WaveSize, wave = tid / WaveSize;

  const int rows_per_req = H * DPC;                              // packed rows: position*H + head
  const int row_id = blockIdx.x / DimSplit, dim_blk = blockIdx.x % DimSplit;
  const int slot = row_id % rows_per_req, req = row_id / rows_per_req;
  const int qgrp = QGroups > 1 ? req % QGroups : 0, b = QGroups > 1 ? req / QGroups : req;
  const int head = slot % H, qpos = qgrp * DPC + slot / H;

  const int ql_b = q_lens ? min(q_lens[b], q_len) : q_len;
  const bool row_live = qpos < ql_b;
  const int b_out = rows ? min(max(rows[b], 0), B - 1) : b;      // the same remap decode used to read Q
  // A padded row's partials were never written by decode. We read and reduce them anyway -- cheaper than
  // branching around it -- and simply never store the result: `row_live` gates the only write to o_lat.

  // Decode indexed the partials by (unit, q group) off this same plan, so re-run it here.
  int unit0 = b * parts, nparts = parts;
  if (seq_lens) {
    const Split plan = make_split(seq_lens, q_lens, q_len, H, DPC, B);
    nparts = plan.units(b, B, parts);
    int acc = 0;
    for (int i = lane; i < b; i += WaveSize) acc += plan.units(i, B, parts);
    acc = wave_all_reduce(acc, [](int x, int y) { return x + y; });
    unit0 = __builtin_amdgcn_readfirstlane(acc);
  }

  const size_t cta0 = (size_t)unit0 * QGroups + qgrp;
  __shared__ float weight[MaxParts];                             // pass 1 fills this, pass 2 reads it
  __shared__ float s_max[Waves], s_sum[Waves], s_out[(Waves - 1) * DimsPerCta];

  const float* __restrict__ part_max_p = partials + hdr_off((size_t)nctas, SlabSlots, slot, 0);
  const float* __restrict__ part_sum_p = partials + hdr_off((size_t)nctas, SlabSlots, slot, 1);
  // ── Pass 1 ── every part's (run_max, run_sum) down to one row max, then one weight per part.
  float part_max[PartsPerThread], part_sum[PartsPerThread], my_max = NegInf;
  for (int k = 0; k < PartsPerThread; ++k) {
    const int p = tid + k * Threads;                             // thread `tid` carries parts tid, tid+256, ...
    part_max[k] = (p < nparts) ? part_max_p[cta0 + (size_t)p * QGroups] : NegInf;
    part_sum[k] = (p < nparts) ? part_sum_p[cta0 + (size_t)p * QGroups] : 0.f;
    my_max = fmaxf(my_max, part_max[k]);
  }
  const float row_max = cta_reduce<Waves>(my_max, s_max, wave, lane, [](float x, float y) { return fmaxf(x, y); });
  const bool empty = (row_max == NegInf);                        // no part had any live KV at all

  float my_sum = 0.f;
  for (int k = 0; k < PartsPerThread; ++k) {
    const int p = tid + k * Threads;
    const float w = empty ? 0.f : __builtin_amdgcn_exp2f(part_max[k] - row_max);
    if (p < nparts) weight[p] = w;
    my_sum += w * part_sum[k];
  }
  // This reduce's __syncthreads is also what publishes `weight` to pass 2 below.
  const float denom = cta_reduce<Waves>(my_sum, s_sum, wave, lane, [](float x, float y) { return x + y; });
  const float inv_denom = denom > 0.f ? 1.f / denom : 0.f;

  // ── Pass 2 ── sum_p weight[p] * O_p over this CTA's slice of the 512 output dims.
  const int part_grp = lane / LanesPerRow, dim_lane = lane % LanesPerRow;
  const size_t part_stride = (size_t)QGroups * SlabSlots * LatDim;
  const bf16* __restrict__ src =
      (const bf16*)partials + (cta0 * SlabSlots + slot) * LatDim + dim_blk * DimsPerCta + dim_lane * DimsPerLane;
  float acc[DimsPerLane] = {};
  #pragma unroll 4
  for (int p = wave * PartsPerWave + part_grp; p < nparts; p += PartsPerCta) {
    const bf16x8 o = *(const bf16x8*)(src + (size_t)p * part_stride);
    const float w = weight[p];
    for (int d = 0; d < DimsPerLane; ++d) acc[d] = __builtin_fmaf(w, (float)o[d], acc[d]);
  }

  for (int m = LanesPerRow; m < WaveSize; m <<= 1)               // fold the lane groups together
    for (int d = 0; d < DimsPerLane; ++d) acc[d] += lane_xor(acc[d], m);
  if (wave != 0 && lane < LanesPerRow)                           // then the waves, through LDS
    for (int d = 0; d < DimsPerLane; ++d) s_out[(wave - 1) * DimsPerCta + dim_lane * DimsPerLane + d] = acc[d];
  __syncthreads();

  if (wave == 0 && lane < LanesPerRow && row_live) {
    for (int w = 0; w < Waves - 1; ++w)
      for (int d = 0; d < DimsPerLane; ++d) acc[d] += s_out[w * DimsPerCta + dim_lane * DimsPerLane + d];
    for (int d = 0; d < DimsPerLane; ++d) acc[d] *= inv_denom;
    bf16* dst = o_lat + (size_t)((b_out * q_len + qpos) * H + head) * LatDim + dim_blk * DimsPerCta + dim_lane * DimsPerLane;
    *(bf16x8*)dst = pack8(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
  }
}

// ── Host-side planning ────────────────────────────────────────────────────────────────────────
// MallBytes: the MI300X infinity cache. Above it the KV pool cannot be reused between steps, so bypassing
// the cache is a win; below it, it is a loss. MaxCTAs bounds the partials workspace -- workspace_for() sizes
// the slab from it -- and 2048 plus two per CU is simply the widest grid we are willing to pay for.
constexpr size_t MallBytes = 256ull << 20, MaxCTAs = 2048 + 2 * NumCU;
// SplitGrid: the grid width past which extra parts stop paying for their own prologue and epilogue.
// SplitLoop: if a part still walks 20x its own prologue, the walk dominates and splitting further is noise.
// PlannerTileTokens: the planner will not cut a request finer than 64 tokens per part, even though the
// kernel itself tiles at 16 -- four KV tiles is the smallest part worth a CTA.
constexpr int SplitGrid = 256, SplitLoop = 20, PlannerTileTokens = 64;
// Bypass L2/MALL only ABOVE the MALL; below it the pool is reused across steps. Worth 1.76x at the hero shape.
__host__ inline bool want_nt(size_t kv_bytes, int ngroups) { return ngroups == 1 && kv_bytes > MallBytes; }
__host__ inline int ql_pad(int q_len) { return q_len <= 4 ? 4 : 8; }

// KV-splits per request. Every part pays the prologue and epilogue in PARALLEL: grid width vs critical path.
__host__ inline int plan_parts(int B, int S, int ngroups) {
  const int lanes = (B > 0 ? B : 1) * ngroups, tiles = kv_tiles(S);
  const int by_len = (S + PlannerTileTokens - 1) / PlannerTileTokens;   // splits the length allows
  const int by_ws = (B > 0) ? (int)(MaxCTAs / ((size_t)B * ngroups)) : 1;   // ... and the workspace allows
  int p = max(1, min(NumCU / lanes, min(by_len, min(MaxParts, by_ws))));   // one CTA per CU is the machine
  if (p >= tiles) return tiles;                                  // already one tile per part
  if ((long long)tiles >= (long long)SplitLoop * p) return p;    // the walk dominates: leave it alone

  const int narrow = SplitGrid / lanes;              // else trade grid width for a shorter critical path
  if (narrow >= 1 && narrow < p) p = narrow;
  // MUST stay monotone in S. workspace_for() sizes the slab once, for max_seq_len, but the contiguous
  // launcher re-plans from the ACTUAL S. An earlier version rounded to whole tiles per part here, and that
  // DIPS every time `tiles` crosses a multiple of p -- so a SHORTER sequence could plan MORE parts than the
  // workspace holds. Sizing at S=16409 (206 parts) and then launching S=16312 (255) overran it by 6.2 MB.
  // As written the result is min(p(S), tiles(S)): a min of two non-decreasing functions, so it cannot dip.
  return p;
}

__host__ inline int capped_parts(int parts, int B, int ng) {
  const int cap = (int)(MaxCTAs / ((size_t)(B > 0 ? B : 1) * (ng > 0 ? ng : 1)));
  const int maxp = cap >= MaxParts ? MaxParts : (cap > 1 ? cap : 1), p = parts < 1 ? 1 : parts;
  return p > maxp ? maxp : p;
}
static_assert(MaxBatch * (MaxQL / 4) <= NumCU / 2, "the domain's B must fit the planner");
__host__ inline int check_domain(int B, int H, int q_len, int lat_dim, int rope_dim, int ng, const void* ws) {
  if (lat_dim != LatDim || rope_dim != RopeDim) return -2;
  if (H < 1 || H > MaxHeads || q_len < 4 || q_len > MaxQL) return -3;
  return (B < 1 || B * ng > NumCU / 2) ? -4 : (ws ? 0 : -5);
}

// ── Launchers ── one resolved launch: padded width, KV split and the NT gate are already picked.
struct Launch {
  KArgs k;
  void* o_lat;
  const int* q_lens;
  int parts;                             // KV parts per request
  int ngroups;                           // CTAs per request's draft window: 1 == one pass over KV
  bool paged;
  bool nt;                               // bypass the cache on the KV loads; see want_nt()
  hipStream_t stream;
};

template <int QLP, int DPC, bool Paged, bool NT, int NTiles>
__host__ inline void launch_one(const Launch& a) {
  mla_decode_a16w8_multiq_kernel<QLP, DPC, Paged, NT, NTiles>
      <<<dim3(a.k.B * a.parts * a.ngroups), dim3(CtaThreads), 0, a.stream>>>(a.k, a.parts, a.q_lens);
}
template <int QLP, int DPC, bool Paged, bool NT>
__host__ inline void launch_pack(const Launch& a) {
  // NTiles is live_tiles_of(DPC, the WIDEST H the arm can see): 8x12 -> 6, 4x12 -> 3, 4x16 -> 4.
  if constexpr (DPC == MaxDraftPerCta) launch_one<QLP, DPC, Paged, NT, live_tiles_of(DPC, MaxPackedHeads)>(a);
  else fits_single_pass(a.k.H) ? launch_one<QLP, DPC, Paged, NT, live_tiles_of(DPC, MaxPackedHeads)>(a)
                               : launch_one<QLP, DPC, Paged, NT, live_tiles_of(DPC, MaxHeads)>(a);
}

// One instantiation per DimSplit rung, peeled the same way the wave dispatch is: 1, 2, 4, 8, 16.
template <int QL, int DPC, int D = 1>
__host__ inline void launch_merge(const Launch& a, int ds, int units, int nctas) {
  if constexpr (D <= MaxDimSplit) {
    if (ds != D && D < MaxDimSplit) { launch_merge<QL, DPC, D * 2>(a, ds, units, nctas); return; }
    mla_decode_a16w8_merge<QL, DPC, D><<<dim3(units * D), dim3(MergeThreads), 0, a.stream>>>(
        a.k.partials, (bf16*)a.o_lat, a.k.H, a.parts, a.k.q_len, nctas, a.k.rows, a.k.B,
        a.k.seq_lens, a.q_lens);
  }
}

template <int QL, int DPC>
__host__ inline void launch_ql(const Launch& a) {
  a.paged ? (a.nt ? launch_pack<QL, DPC, true, true>(a) : launch_pack<QL, DPC, true, false>(a))
          : (a.nt ? launch_pack<QL, DPC, false, true>(a) : launch_pack<QL, DPC, false, false>(a));
  // nctas MUST equal the decode grid the launch above used -- the decode kernel reads it back as gridDim.x
  // -- because it is the stride of the run_max/run_sum header planes that the merge is about to read.
  const int units = a.k.B * (QL / DPC) * a.k.H * DPC, nctas = a.k.B * a.parts * (QL / DPC);
  // DimSplit: split the 512 output dims across more CTAs until the merge grid is ~2 CTAs per CU. It is a
  // template parameter (the lane map depends on it), so the runtime choice picks one of five instantiations.
  int ds = 1;
  while (ds < MaxDimSplit && units * ds * 2 < 4 * NumCU) ds <<= 1;
  launch_merge<QL, DPC>(a, ds, units, nctas);
}

__host__ inline int launch(const Launch& a, int qlp) {
  if (qlp <= 4) launch_ql<4, 4>(a);
  else if (a.ngroups == 1) launch_ql<8, MaxDraftPerCta>(a);
  else launch_ql<8, 4>(a);
  return (int)hipGetLastError();
}

__host__ inline Launch make_launch(const void* ql, const void* qp, const void* kv, void* o, void* ws,
                                   int B, int H, int q_len, float scale, float kv_scale, void* stream) {
  Launch a{};
  a.k = KArgs{(const bf16*)ql, (const bf16*)qp, (const fp8_t*)kv, (float*)ws, nullptr, nullptr, nullptr, nullptr,
              B, H, 0, q_len, 1.4426950408889634f * scale, kv_scale};
  a.o_lat = o;  a.stream = (hipStream_t)stream;
  return a;
}
}  // namespace

// ── extern "C" ABI ── (the _multiq suffix separates these from the mla_decode_a16w8 ops, which
//   live in the same extension)
extern "C" __host__ int mla_decode_a16w8_multiq_plan_parts_q(int B, int max_seq_len, int q_len, int H) {
  return plan_parts(B, max_seq_len, qgroups_of(ql_pad(q_len), H));
}

extern "C" __host__ size_t mla_decode_a16w8_multiq_workspace_for(int B, int H, int parts, int q_len) {
  const int qlp = ql_pad(q_len);
  size_t need = 0;
  (void)H;
  for (int dpc = 4; dpc <= MaxDraftPerCta && dpc <= qlp; dpc <<= 1) {          // whichever split is chosen
    const int ng = qlp / dpc, slab = MaxHeads * dpc;
    const size_t nctas = (size_t)(B > 0 ? B : 1) * capped_parts(parts, B, ng) * ng;
    need = max(need, partials_floats(nctas, slab) * sizeof(float));
  }
  return need;
}

extern "C" __host__ int launch_mla_decode_a16w8_multiq(
    const void* q_lat, const void* q_pe, const void* c_kv, void* o_lat, int B, int H, int S, int q_len,
    int lat_dim, int rope_dim, float scale, float kv_scale, void* workspace, void* stream_) {

  const int qlp = ql_pad(q_len);
  const int ng = qgroups_of(qlp, H);
  if (const int rc = check_domain(B, H, q_len, lat_dim, rope_dim, ng, workspace)) return rc;
  Launch a = make_launch(q_lat, q_pe, c_kv, o_lat, workspace, B, H, q_len, scale, kv_scale, stream_);
  a.k.S_scalar = S;
  a.ngroups = ng;
  a.parts = plan_parts(B, S, ng);
  a.nt = want_nt((size_t)(B > 0 ? B : 1) * (size_t)(S > 0 ? S : 0) * FusedDim, ng);
  return launch(a, qlp);
}

// Paged launcher: graph-safe and device-driven, so `parts` is fixed at capture and the split must follow it.
//   `q_lens` ([B] int32 DEVICE, null = rectangular) is the RAGGED QUERY WINDOW: only positions [0, q_lens[b]) of
// the padded q_len are live. q_lens[b] <= q_len is the caller's contract; the kernel clamps, never faults.
extern "C" __host__ int launch_mla_decode_a16w8_multiq_paged_dev_ql(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat, const void* seq_lens,
    const void* rows, const void* kv_indices, const void* kv_indptr, int B, int H, int parts, int q_len,
    int lat_dim, int rope_dim, float scale, float kv_scale, void* workspace, void* stream_, size_t kv_bytes, const void* q_lens) {
  const int qlp = ql_pad(q_len);
  const int ng = qgroups_of(qlp, H);
  if (const int rc = check_domain(B, H, q_len, lat_dim, rope_dim, ng, workspace)) return rc;
  Launch a = make_launch(q_lat, q_pe, kv_pool, o_lat, workspace, B, H, q_len, scale, kv_scale, stream_);
  a.k.seq_lens = (const int*)seq_lens;  a.k.rows = (const int*)rows;
  a.k.kv_indices = (const int*)kv_indices;  a.k.kv_indptr = (const int*)kv_indptr;
  a.q_lens = (const int*)q_lens;  a.ngroups = ng;  a.paged = true;
  a.parts = capped_parts(parts, B, ng);              // clamp to the grid the workspace was sized for
  a.nt = want_nt(kv_bytes, ng);
  return launch(a, qlp);
}
extern "C" __host__ int launch_mla_decode_a16w8_multiq_paged_dev(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat, const void* seq_lens,
    const void* rows, const void* kv_indices, const void* kv_indptr, int B, int H, int parts, int q_len,
    int lat_dim, int rope_dim, float scale, float kv_scale, void* workspace, void* stream_, size_t kv_bytes) {
  return launch_mla_decode_a16w8_multiq_paged_dev_ql(q_lat, q_pe, kv_pool, o_lat, seq_lens, rows, kv_indices,
      kv_indptr, B, H, parts, q_len, lat_dim, rope_dim, scale, kv_scale, workspace, stream_, kv_bytes, nullptr);
}