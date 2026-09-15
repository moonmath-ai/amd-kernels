// MLA a16w8 decode for CDNA3 (gfx942 / MI300X), with optional decode context parallelism (DCP).
//
//   score[t,h,n] = (q_lat[t,h,:] . c_kv[n,0:512] + q_pe[t,h,:] . c_kv[n,512:576]) * scale
//   o_lat[t,h,:] = sum_n softmax_n(score)[t,h,n] * c_kv[n,0:512]
//
// Dense batch (one q_len per request), paged pool with page_size 1, launch planned from host
// shapes only (cuda-graph safe). Position t attends [0, len - (q_len-1-t)): end-aligned causal.
//
// DCP shards the pool by position: p lives on rank p % cp_world at row p // cp_world. A rank runs
// all H heads over its 1/N positions and returns a partial plus its base-2 LSE; the merge op
// combines ranks after the all-to-all. The walk is rank-local, the causal limit is global.
//
// CTA: 8 waves, NT contract query tiles of 16 rows against 16-token KV tiles; the rest stage KV
// HBM -> LDS. Every wave accumulates PV for 64 of the 512 output dims. Per-part (max, sum, O)
// partials are rebased onto one max per row by the merge kernel (split-K online softmax, base 2).
//
// Domain: H 1..128, q_len >= 1, B * slices <= 304 CTAs.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

namespace {
using bf16 = __bf16;
using bf16x4 = __attribute__((ext_vector_type(4))) bf16;
using bf16x8 = __attribute__((ext_vector_type(8))) bf16;
using f32x2 = __attribute__((ext_vector_type(2))) float;
using f32x4 = __attribute__((ext_vector_type(4))) float;
using u32x2 = __attribute__((ext_vector_type(2))) uint32_t;
template <int N> using u32vec = __attribute__((ext_vector_type(N))) uint32_t;
using fp8_t = __hip_fp8_e4m3_fnuz;

template <class T, class U> __device__ __forceinline__ T bitcast(const U& u) {
  static_assert(sizeof(T) == sizeof(U), "bitcast width");
  return __builtin_bit_cast(T, u);
}

// ── Layout constants ────────────────────────────────────────────────────────────────────────────
constexpr int WaveSize = 64, MfmaTile = 16, QuadSpan = 4;
constexpr int LatDim = 512, RopeDim = 64, FusedDim = LatDim + RopeDim;  // 576 fp8 bytes per token
constexpr int MfmaK = 16, KQuadElems = MfmaK / QuadSpan;  // each lane holds 4 bf16 along K
constexpr int LatSteps = LatDim / MfmaK, QkSteps = FusedDim / MfmaK;    // 36 QK contraction steps

constexpr int NumWaves = 8, CtaThreads = NumWaves * WaveSize, NumCU = 304;
constexpr int TileTokens = 16;                     // KV tokens staged per pipeline step
constexpr int TokensPerLane = TileTokens / QuadSpan;
constexpr int DimTiles = LatDim / MfmaTile / NumWaves;   // output-dim tiles a wave owns: 4 of 32
constexpr int VPairs = DimTiles / 2;
constexpr int FillOps = 9;                         // a 576-byte token row, in 9 loads per lane
constexpr int QkChunk = 6;                         // K-steps per chunk; chunk c+1 reads under c
constexpr int LookAhead = 2;                       // KV tiles a fill wave keeps between load and store
constexpr int MaxNTiles = 6;                       // a 7th query tile would spill the PV accumulator
constexpr int MaxHeads = 128;

// -3e38 marks "no tokens": past any real base-2 LSE, and finite so arithmetic stays NaN-free
// under -ffast-math (a true -inf would fold to NaN). Dead rows then give exp2 == 0, sum == 0.
constexpr float LseSentinel = 3.0e38f, NegInf = -LseSentinel;
constexpr float Log2E = 1.4426950408889634f;
constexpr float ProbLiftLog2 = 7.0f, ProbLift = 128.0f;
constexpr float RescaleSlack = 4.0f;               // how far a row max may drift before a rescale

static_assert(TokensPerLane == QuadSpan, "PV needs 4 tokens per lane for the register transpose");

// LDS pad: 4 bf16 per KV row puts KvRowElems/2 at 2 (mod 32 banks), so the tile's 16 rows land on
// distinct bank pairs. RescaleStride avoids the colliding 6..9.
constexpr int KvRowElems = FusedDim + 4, KvRowBytes = KvRowElems * (int)sizeof(bf16);
constexpr int RescaleStride = 10;
static_assert(KvRowBytes % 8 == 0, "ds_read_b64 / ds_write2_b64 need 8-B aligned rows");

__host__ __device__ constexpr __forceinline__ int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float lse_base2(float max, float sum) { return max + __logf(sum) * Log2E; }

// NaN, +inf and the sentinel all mean "no tokens".
__device__ __forceinline__ float finite_lse(float lse) {
  return fabsf(lse) < LseSentinel ? lse : NegInf;
}

// ── MFMA fragment layout ────────────────────────────────────────────────────────────────────────
// D[16x16] = A[16x16] * B[16x16] + C over K = 16. With mfma_row = lane % 16 and quad = lane / 16,
// each lane holds four elements per operand:
//
//   A: a[mfma_row][4*quad + e]    B: b[4*quad + e][mfma_row]    D: d[4*quad + e][mfma_row]
//
// So a lane's A elements walk the contracted axis and its D elements walk M. For QK, M is the
// tile's 16 tokens, N the 16 query rows, K 16 of the 576 dims; for PV, M is 16 output dims, N the
// query rows, K the tile's 16 tokens. Every LDS read lands data already in this shape.
__device__ __forceinline__ f32x4 mfma_bf16(bf16x4 a, bf16x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ bf16x4 as_bf16x4(uint32_t lo, uint32_t hi) {
  return bitcast<bf16x4>(u32x2{lo, hi});
}

enum Round { Trunc, Rtne, Rna };
template <Round R> __device__ __forceinline__ unsigned narrow_bf16(float f) {
  const unsigned u = bitcast<unsigned>(f);
  return R == Trunc ? u : R == Rna ? u + 0x8000u : u + 0x7FFFu + ((u >> 16) & 1u);
}

template <Round R> __device__ __forceinline__ bf16x4 pack4(float a, float b, float c, float d) {
  return as_bf16x4(__builtin_amdgcn_perm(narrow_bf16<R>(b), narrow_bf16<R>(a), 0x07060302u),
                   __builtin_amdgcn_perm(narrow_bf16<R>(d), narrow_bf16<R>(c), 0x07060302u));
}

__device__ __forceinline__ bf16x8 pack8(float a, float b, float c, float d,
                                        float e, float f, float g, float h) {
  const bf16x4 halves[2] = {pack4<Rtne>(a, b, c, d), pack4<Rtne>(e, f, g, h)};
  return bitcast<bf16x8>(halves);
}

// Go through bf16x4: a hand-written cvt_pk/perm pair over f32x2 subscripts is a live miscompile.
__device__ __forceinline__ u32x2 widen4_fp8(uint32_t four_fp8) {
  const f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)four_fp8, false);
  const f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)four_fp8, true);
  return bitcast<u32x2>(pack4<Trunc>(lo[0], lo[1], hi[0], hi[1]));
}

// Not `*=`: that selects v_pk_mul_f32, which shares the FP32 MAC with the matrix pipe.
__device__ __forceinline__ float mul_f32(float a, float b) {
  float r;
  asm volatile("v_mul_f32_e32 %0,%1,%2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

// 0xC07F is lgkmcnt(0); __syncthreads() would add a memory fence this kernel never needs.
__device__ __forceinline__ void tile_barrier() {
  __builtin_amdgcn_s_waitcnt(0xC07F);
  asm volatile("s_barrier" ::: "memory");
}
template <class T> __device__ __forceinline__ T lane_xor(T v, int mask) {
  return __shfl_xor(v, mask, WaveSize);
}

template <class T, class Op> __device__ __forceinline__ T wave_all_reduce(T value, Op op) {
  for (int mask = 1; mask < WaveSize; mask <<= 1) value = op(value, lane_xor(value, mask));
  return value;
}

// A row fold crosses only the top two lane bits; ds_bpermute addresses in bytes. Not __shfl_xor.
__device__ inline float xor16(float v) {
  return bitcast<float>(__builtin_amdgcn_ds_swizzle(bitcast<int>(v), 0x401F));
}
__device__ inline float xor32(float v) {
  return bitcast<float>(__builtin_amdgcn_ds_bpermute((__lane_id() << 2) + 128, bitcast<int>(v)));
}
__device__ inline float fold_max(float v) { v = fmaxf(v, xor16(v)); return fmaxf(v, xor32(v)); }
__device__ inline float fold_sum(float v) { v += xor16(v); return v + xor32(v); }

template <int Waves, class Op>
__device__ __forceinline__ float cta_reduce(float value, float* slot, int wave, int lane, Op op) {
  value = wave_all_reduce(value, op);
  if (lane == 0) slot[wave] = value;
  __syncthreads();
  float acc = slot[0];
  for (int w = 1; w < Waves; ++w) acc = op(acc, slot[w]);
  return acc;
}

__device__ __forceinline__ int wave_min(int value) {
  for (int mask = 1; mask < WaveSize; mask <<= 1) value = min(value, lane_xor(value, mask));
  return value;
}

// The fill waves split the tile's tokens; a lane's share of a row must be a dword..dwordx4.
constexpr int fill_waves(int nt)  { return NumWaves - nt >= 4 ? 4 : 2; }
constexpr int fill_tokens(int nt) { return TileTokens / fill_waves(nt); }
constexpr int fill_lanes(int nt)  { return WaveSize / fill_tokens(nt); }
constexpr int fill_bytes(int nt)  { return FusedDim / (fill_lanes(nt) * FillOps); }

// Partial workspace: a bf16 (cta, row, dim) slab, then two fp32 header planes of each CTA's
// (max, sum) per row, indexed (row, cta) for constant-stride reads. Round-up keeps them
// f32x4-aligned.
__host__ __device__ __forceinline__ size_t slab_floats(size_t n_ctas, int rows) {
  return ((n_ctas * (size_t)rows * LatDim / 2) + 3) & ~(size_t)3;
}
__device__ __forceinline__ bf16* partial_row(float* base, int rows, size_t cta, int row) {
  return (bf16*)base + (cta * rows + row) * LatDim;
}
__device__ __forceinline__ float* partial_max(float* base, size_t n_ctas, int rows, int row) {
  return base + slab_floats(n_ctas, rows) + (size_t)2 * row * n_ctas;
}

__host__ __device__ constexpr __forceinline__ int slices_of(int q_len, int H, int rows) {
  return cdiv(q_len * H, rows);
}

// Launch arguments, filled once on the host.
struct KArgs {
  const bf16* __restrict__ q_lat;        // [B*q_len, H, 512] bf16, request b at rows b*q_len ..
  const bf16* __restrict__ q_pe;         // [B*q_len, H, 64]
  const fp8_t* __restrict__ c_kv;        // rank-local paged pool: [num_slots, 576] fp8
  float* __restrict__ partials;

  const int* __restrict__ seq_lens;      // [B] rank-local KV lengths
  const int* __restrict__ kv_indices;    // [sum(seq_lens)] flat per-token pool slots (page_size 1)
  const int* __restrict__ kv_indptr;     // [B+1] per-request slot offsets
  const int* __restrict__ glen;          // [B] global KV lengths for the causal limit (null: world 1)

  int B, q_len, H;
  int rows;                              // rows per CTA: NTiles * 16
  int slices, nparts;                    // uniform across requests, both chosen on the host
  int cp_rank, cp_world;
  float scale_log2e;
  float kv_scale;
  bool non_temporal;                     // KV fill bypasses cache when the pool exceeds the MALL
};

// Draft position qpos may attend global positions [0, glen - (q_len-1-qpos)). This rank holds p
// at row p / cp_world only when p % cp_world == cp_rank, which divides the limit down.
__device__ __forceinline__ int local_causal_end(const KArgs& args, int global_len, int qpos) {
  const int global_end = global_len - (args.q_len - 1 - qpos);
  return global_end <= args.cp_rank ? 0 : cdiv(global_end - args.cp_rank, args.cp_world);
}

// Four tokens' bf16 pairs -> one dim across four tokens: the V operand of the PV. Passed by
// value, since a pointer into the fragment stops SROA promoting it.
__device__ __forceinline__ bf16x4 transpose_even(uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
  return as_bf16x4(__builtin_amdgcn_perm(t1, t0, 0x05040100u),      // dim 2a, tokens 0,1 | 2,3
                   __builtin_amdgcn_perm(t3, t2, 0x05040100u));
}
__device__ __forceinline__ bf16x4 transpose_odd(uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
  return as_bf16x4(__builtin_amdgcn_perm(t1, t0, 0x07060302u),      // dim 2a+1
                   __builtin_amdgcn_perm(t3, t2, 0x07060302u));
}

// ── Per-wave walk over the KV stream ─────────────────────────────────────────────────────────────
// Both roles run this body; IsFill picks which. The phases are lambdas over the register state,
// called in one fixed order from `step` near the bottom.
template <int NT, bool IsFill, bool AllTilesLive>
__device__ __forceinline__ void wave_body(
    int wave, int req, int slice, int part, int live_rows, const KArgs& args,
    bf16 (&s_kv)[2][TileTokens][KvRowElems],            // ring: the slot being read, the filling one
    bf16x4 (&s_prob)[2][NT][WaveSize],                  // softmax -> PV, published a step ahead
    float (&s_rescale)[2][MfmaTile][RescaleStride]) {   // per (row, tile) rescale; 1.0f == no move

  constexpr int Rows = NT * MfmaTile;
  constexpr int FillWaves = fill_waves(NT), FillTok = fill_tokens(NT);
  constexpr int FillLanes = fill_lanes(NT), FillBytes = fill_bytes(NT), FillWords = FillBytes / 4;
  constexpr int SrcStride = FillLanes * FillBytes, DstStride = 2 * SrcStride;
  constexpr int QRegs = IsFill ? 1 : QkSteps, KRegs = IsFill ? 1 : QkChunk;
  using FillVec = u32vec<FillWords>;

  static_assert(FillWaves * FillTok == TileTokens, "the fill waves must split the tile exactly");
  static_assert(FillLanes * FillBytes * FillOps == FusedDim, "the fill must tile a row exactly");
  static_assert(FillBytes == 4 || FillBytes == 8 || FillBytes == 16, "lane op must be dword..dwordx4");

  const int lane = (int)threadIdx.x & (WaveSize - 1);
  const int mfma_row = lane & (MfmaTile - 1);
  const int quad = lane / MfmaTile;

  const int query_tile = IsFill ? 0 : wave;
  const int fill_share = IsFill ? wave - NT : 0;
  const bool stages_kv = IsFill && fill_share < FillWaves;

  // Live tiles are a PREFIX of NTiles, so AllTilesLive folds the test away entirely.
  const int live_tiles = cdiv(live_rows, MfmaTile);
  const bool tile_live = !IsFill &&
                         (AllTilesLive || __builtin_amdgcn_readfirstlane(query_tile) < live_tiles);

  // This CTA's slice of the KV stream: whole tiles per part, so no part starts mid-tile.
  const int kv_len = args.seq_lens[req], pool_base = args.kv_indptr[req];
  const int per_part = cdiv(kv_len, args.nparts * TileTokens) * TileTokens;
  const int win_lo = part * per_part, win_hi = min(kv_len, win_lo + per_part);
  const int kv_tiles = win_hi > win_lo ? cdiv(win_hi - win_lo, TileTokens) : 0;

  // This wave's row and the Q it contracts: a masked row reads Q row 0 and contracts to nothing.
  const int packed_row = query_tile * MfmaTile + mfma_row;
  const bool active = !IsFill && packed_row < live_rows;
  const int global_row = slice * Rows + packed_row;
  const int qpos = global_row / args.H;

  bf16x4 q[QRegs];
  if constexpr (!IsFill) {
    const int head = global_row - qpos * args.H;
    const size_t r = (size_t)(req * args.q_len + (active ? qpos : 0)) * args.H + (active ? head : 0);
    const bf16x4 zero = {};
    #pragma unroll
    for (int i = 0; i < QkSteps; ++i) {
      const bf16* src = (i < LatSteps) ? args.q_lat + r * LatDim + (size_t)i * MfmaK
                                       : args.q_pe + r * RopeDim + (size_t)(i - LatSteps) * MfmaK;
      q[i] = active ? *(const bf16x4*)(src + quad * KQuadElems) : zero;
    }
  }

  // Where this row's softmax stops, and the running (max, sum) it folds into.
  const int global_len = args.glen ? args.glen[req] : kv_len;
  const int tok_lim = active ? min(kv_len, local_causal_end(args, global_len, qpos)) : kv_len;
  const int kv_end = min(win_hi, tok_lim);
  const float score_scale = args.kv_scale * args.scale_log2e;   // folds into the exp2 argument
  // The lift must be NegInf on a dead row, or exp2 returns inf where 0 is needed.
  const float lift = (win_lo >= kv_end) ? NegInf : ProbLiftLog2;
  // Earliest causal end in the wave: a tile ending at or before it is fully live for every lane
  // and needs no masking (every tile but the row's last).
  const int kv_end_min = wave_min(kv_end);
  float run_max = NegInf, run_sum = 0.f;

  // Register state: arrays must stay promoted, so they live here and the phases that touch them
  // are inlined lambdas, never callees taking a pointer into one.
  f32x4 pv[NT][DimTiles] = {};                       // this wave's 64 output dims for every tile
  uint32_t v[2][VPairs][TokensPerLane] = {};         // V fragment: (LDS parity, dim pair, token)
  bf16x4 k[2][KRegs];                                // double-buffered: chunk c reads into k[c & 1]
  f32x4 scores = {};
  float rescale[NT];                                 // this step's per-tile rescale for this lane's row
  bool any_moved;
  FillVec fill_reg[2][FillOps];                      // batches of raw fp8 in flight from HBM
  // Pool slots resolve two steps ahead of their load and are double-buffered, so both gathers are
  // in flight: the lookup is a random HBM read and needs more cover than one step gives.
  int pool_row[2] = {0, 0};

  const int dim0 = wave * (DimTiles * MfmaTile);     // first of this wave's 64 output dims
  const int v_elem = dim0 + DimTiles * mfma_row;     // where they start inside one KV row
  const int fill_tok0 = fill_share * FillTok;        // first token of this wave's share
  const int fill_row = lane / FillLanes;             // which token of it this lane carries
  const int fill_byte = (lane % FillLanes) * FillBytes;

  // ── Pipeline phases ──────────────────────────────────────────────────────────────────────────
  // any_moved is local: a tile that stood still has rescale == 1.0f exactly.
  auto read_rescales = [&](int parity) {
    any_moved = false;
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
      rescale[t] = s_rescale[parity][mfma_row][t];
      any_moved |= rescale[t] != 1.0f;
    }
  };

  auto read_k = [&](int chunk, int parity) {         // one chunk of the 36 K-steps, into k[chunk & 1]
    if constexpr (!IsFill) {
      #pragma unroll
      for (int i = 0; i < QkChunk; ++i) {
        const int step = chunk * QkChunk + i;
        k[chunk & 1][i] = *(const bf16x4*)&s_kv[parity][mfma_row][quad * KQuadElems + step * MfmaK];
      }
    }
  };

  // Chunk c+1's LDS reads issue under chunk c's matrix work, so no group ever waits on its own
  // reads. One 36-deep accumulator chain: it already issues at the matrix pipe's rate.
  auto contract_qk = [&](int parity) {
    if constexpr (!IsFill) {
      constexpr int Chunks = QkSteps / QkChunk;
      f32x4 acc = {};
      #pragma unroll
      for (int c = 0; c < Chunks; ++c) {
        if (c + 1 < Chunks) read_k(c + 1, parity);
        #pragma unroll
        for (int i = 0; i < QkChunk; ++i) acc = mfma_bf16(k[c & 1][i], q[c * QkChunk + i], acc);
      }
      scores = acc;
    }
  };

  // Online softmax: tile kv_tile-1's scores fold into the row's running (max, sum). The rescale is
  // lazy -- the max only moves on an overshoot past RescaleSlack, so most steps hand the PV a 1.0.
  auto run_softmax = [&](int kv_tile, int parity) {
    const int tok0 = win_lo + (kv_tile - 1) * TileTokens;
    const int live = kv_end - tok0, my_tok0 = quad * TokensPerLane;
    const bool whole = tok0 + TileTokens <= kv_end_min;      // wave-uniform

    float tile_score[QuadSpan], tile_max = NegInf;
    if (__builtin_expect(whole, 1)) {
      #pragma unroll
      for (int i = 0; i < QuadSpan; ++i) {
        tile_score[i] = scores[i];
        tile_max = fmaxf(tile_max, tile_score[i]);
      }
    } else {
      #pragma unroll
      for (int i = 0; i < QuadSpan; ++i) {
        tile_score[i] = (my_tok0 + i >= live) ? NegInf : scores[i];   // past the causal end
        tile_max = fmaxf(tile_max, tile_score[i]);
      }
    }
    tile_max *= score_scale;

    // Wave-uniform: one lane overflowing its slack rescales the whole wave, which ballot needs.
    const bool moved = (__ballot(tile_max > run_max + RescaleSlack) != 0ull);
    const float new_max = __builtin_expect(moved, 0) ? fmaxf(run_max, fold_max(tile_max)) : run_max;
    const float bias = lift - new_max;
    float tile_sum = 0.f, p[QuadSpan];
    #pragma unroll
    for (int i = 0; i < QuadSpan; ++i) {
      p[i] = __builtin_amdgcn_exp2f(__builtin_fmaf(tile_score[i], score_scale, bias));
      tile_sum += p[i];
    }
    tile_sum *= (1.0f / ProbLift);

    float row_rescale = 1.0f;
    if (__builtin_expect(moved, 0)) {
      // First tile of a row: pv is zero and a dead row's NegInf arithmetic is still finite, so 0.
      row_rescale = (run_max == NegInf) ? 0.f : __builtin_amdgcn_exp2f(run_max - new_max);
      run_sum = run_sum * row_rescale + tile_sum;
      run_max = new_max;
    } else {
      run_sum += tile_sum;
    }

    const int next = 1 - parity;                     // what the next step's PV consumes
    s_prob[next][query_tile][lane] = pack4<Rna>(p[0], p[1], p[2], p[3]);
    if (quad == 0) s_rescale[next][mfma_row][query_tile] = row_rescale;
  };

  auto fetch_v = [&](int parity) {
    #pragma unroll
    for (int j = 0; j < TokensPerLane; ++j) {
      const bf16* row = &s_kv[parity][quad * TokensPerLane + j][v_elem];
      #pragma unroll                                    // a KV row is 8-B aligned, never 16: read
      for (int pair = 0; pair + 1 < VPairs; pair += 2) {// in dword pairs, ds_read2_b64 folds them
        const u32x2 two = *(const u32x2*)(row + 2 * pair);
        v[parity][pair][j] = two[0];
        v[parity][pair + 1][j] = two[1];
      }
      if constexpr (VPairs % 2) v[parity][VPairs - 1][j] = *(const uint32_t*)(row + 2 * (VPairs - 1));
    }
  };

  // A wave owns 64 of the 512 output dims for every query tile (not all 512 of one), so one V
  // fragment feeds NT matrix ops. `plane` is the probabilities' LDS plane, `buf` the V fragment;
  // only the drain passes different ones.
  auto accumulate_pv = [&](int plane, int buf) {
    bf16x4 prob[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t) prob[t] = s_prob[plane][t][lane];

    if (__builtin_expect(any_moved, 0))
      #pragma unroll
      for (int t = 0; t < NT; ++t)
        if (__builtin_expect(rescale[t] != 1.0f, 0))
          #pragma unroll
          for (int c = 0; c < DimTiles; ++c)
            #pragma unroll
            for (int e = 0; e < QuadSpan; ++e) pv[t][c][e] = mul_f32(pv[t][c][e], rescale[t]);

    #pragma unroll
    for (int pair = 0; pair < VPairs; ++pair) {
      const uint32_t t0 = v[buf][pair][0], t1 = v[buf][pair][1];
      const uint32_t t2 = v[buf][pair][2], t3 = v[buf][pair][3];
      const bf16x4 even = transpose_even(t0, t1, t2, t3), odd = transpose_odd(t0, t1, t2, t3);
      #pragma unroll
      for (int t = 0; t < NT; ++t) {
        pv[t][2 * pair]     = mfma_bf16(even, prob[t], pv[t][2 * pair]);
        pv[t][2 * pair + 1] = mfma_bf16(odd,  prob[t], pv[t][2 * pair + 1]);
      }
    }
  };

  // KV fill: HBM fp8 -> registers -> bf16 -> LDS, one batch per step, LookAhead tiles ahead.
  auto resolve_row = [&](int it, int buf) {          // two phases ahead of its paired load
    if (it >= kv_tiles) return;
    const int tok0 = win_lo + it * TileTokens, live = min(TileTokens, win_hi - tok0);
    const int mine = fill_tok0 + fill_row;           // a tail tile clamps onto its last token
    pool_row[buf] = args.kv_indices[pool_base + tok0 + (mine < live ? mine : live - 1)];
  };

  // The loads stay compiler-visible: LLVM does not model inline-asm vmcnt and would hoist a convert.
  auto fill_load = [&](int it, int buf, int row) {
    if (it >= kv_tiles) return;
    const fp8_t* src = args.c_kv + (size_t)pool_row[row] * FusedDim + fill_byte;
    if (args.non_temporal) {                         // one scalar branch, uniform for the launch
      #pragma unroll
      for (int j = 0; j < FillOps; ++j)
        fill_reg[buf][j] = __builtin_nontemporal_load((const FillVec*)(src + j * SrcStride));
    } else {
      #pragma unroll
      for (int j = 0; j < FillOps; ++j) fill_reg[buf][j] = *(const FillVec*)(src + j * SrcStride);
    }
  };

  auto fill_store = [&](int it, int buf, int slot) {
    if (it >= kv_tiles) return;
    char* dst = (char*)&s_kv[slot][fill_tok0 + fill_row][0] + 2 * fill_byte;
    #pragma unroll
    for (int j = 0; j < FillOps; ++j)
      #pragma unroll
      for (int i = 0; i < FillWords; ++i)
        *(u32x2*)(dst + j * DstStride + i * 8) = widen4_fp8(fill_reg[buf][j][i]);
  };

  // One step, five KV tiles in flight: kv_tile-2 in PV, kv_tile-1 in softmax, kv_tile contracting,
  // kv_tile+1 going to LDS, kv_tile+3 loading.
  auto step = [&](int kv_tile, int parity) {
    read_rescales(parity);
    if (stages_kv) resolve_row(kv_tile + 2 + LookAhead, parity);

    if (tile_live) read_k(0, parity);                // latency hides under the softmax below
    if (tile_live && kv_tile > 0) run_softmax(kv_tile, parity);

    accumulate_pv(parity, parity);
    fetch_v(parity);

    // Store above the contraction: a ds_write after the last matrix op stalls every wave on the
    // barrier's lgkmcnt(0). Load below it, because the store must free its buffer first.
    if (stages_kv) fill_store(kv_tile + 1, parity, 1 - parity);
    if (tile_live) contract_qk(parity);
    if (stages_kv) fill_load(kv_tile + 1 + LookAhead, parity, 1 - parity);

    // Interleaves the matrix group with the LDS reads instead of issuing it back to back.
    __builtin_amdgcn_iglp_opt(0);
    tile_barrier();
  };

  // Prologue: tile 0 straight to LDS frees buffer 1; tiles 1 and 2 stay in flight.
  if (stages_kv) {
    resolve_row(0, 0); fill_load(0, 1, 0); fill_store(0, 1, 0);
    resolve_row(1, 0); fill_load(1, 0, 0);
    resolve_row(2, 0); fill_load(2, 1, 0);
    resolve_row(3, 1);                               // step 0 loads tile 3 from the row this resolves
  }
  const int tid = (int)threadIdx.x;                  // a silent tile still reads a 0 prob and a 1.0
  for (int i = tid; i < 2 * NT * WaveSize; i += CtaThreads) (&s_prob[0][0][0])[i] = bf16x4{};
  if (tid < 2 * MfmaTile * RescaleStride) (&s_rescale[0][0][0])[tid] = 1.0f;
  tile_barrier();

  // One step per KV tile plus one to drain; unrolled by two so parity is a compile-time constant.
  step(0, 0);
  int it = 1;
  #pragma unroll 1
  for (; it < kv_tiles; it += 2) { step(it, 1);  step(it + 1, 0); }
  if (it <= kv_tiles) step(it, 1);

  // The last PV accumulate the walk owes: the PV lags the softmax by a step. The V fragment index
  // must be a compile-time constant, so normalize it to buffer 0 before the accumulate.
  if (kv_tiles > 0) {
    const int parity = (kv_tiles & 1) ? 0 : 1;
    read_rescales(parity);
    if (parity) {
      #pragma unroll
      for (int pair = 0; pair < VPairs; ++pair)
        #pragma unroll
        for (int j = 0; j < TokensPerLane; ++j) v[0][pair][j] = v[1][pair][j];
    }
    accumulate_pv(parity, 0);
  }

  // Epilogue: kv_scale dequantises V (score_scale did K) and 1/ProbLift undoes the lift the
  // softmax took off run_sum. Element e of accumulator c is dim0 + DimTiles*(4*quad + e) + c --
  // one contiguous run, so the stores stay bf16x8 for any DimTiles.
  if constexpr (!IsFill) run_sum = fold_sum(run_sum);
  const size_t cta_id = blockIdx.x, n_ctas = gridDim.x;
  const float out_scale = args.kv_scale * (1.0f / ProbLift);

  #pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int row = t * MfmaTile + mfma_row;
    if (row >= live_rows) continue;                  // a row this request does not have is never stored
    bf16* out = partial_row(args.partials, Rows, cta_id, row) + dim0 + DimTiles * QuadSpan * quad;
    #pragma unroll
    for (int i = 0; i < QuadSpan * DimTiles; i += 8) {
      float x[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) x[j] = pv[t][(i + j) % DimTiles][(i + j) / DimTiles] * out_scale;
      *(bf16x8*)(out + i) = pack8(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    }
  }

  if (quad == 0 && active) {
    float* hdr = partial_max(args.partials, n_ctas, Rows, packed_row);
    hdr[cta_id] = run_max;
    hdr[n_ctas + cta_id] = run_sum;
  }
}

// ── Decode kernel ── LDS planes, the CTA's place in the grid, then per-wave role dispatch. ──────
template <int NT, bool AllTilesLive>
__global__ void __launch_bounds__(CtaThreads, 1) mla_decode_a16w8_kernel(KArgs args) {
  static_assert(NT >= 1 && NT < NumWaves, "one wave must contract and one must be spare");

  __shared__ bf16 s_kv[2][TileTokens][KvRowElems];
  __shared__ bf16x4 s_prob[2][NT][WaveSize];
  __shared__ __align__(16) float s_rescale[2][MfmaTile][RescaleStride];
  static_assert(sizeof(s_kv) + sizeof(s_prob) + sizeof(s_rescale) <= 65536, "LDS");

  // Dense (slice, request, part) grid: no device plan, no scan, no table. The `slices` CTAs sharing
  // a (request, part) read the same KV, and MI300X sends workgroup i to XCD i % 8, so `slice` is the
  // slowest index and lands them on one XCD's L2.
  const int cta = (int)blockIdx.x, ctas_per_slice = args.B * args.nparts;
  const int slice = cta / ctas_per_slice, within_slice = cta - slice * ctas_per_slice;
  const int req = within_slice / args.nparts, part = within_slice - req * args.nparts;
  const int rows_left = args.q_len * args.H - slice * (NT * MfmaTile);
  const int live_rows = min(rows_left, NT * MfmaTile);
  if (live_rows <= 0) return;

  const int wave = (int)threadIdx.x / WaveSize;
  // With only two fill waves, a priority bump keeps them a step ahead of the contract waves.
  if (wave >= NT && NumWaves - NT < 4) asm volatile("s_setprio 1" ::: "memory");

  if (wave < NT)
    wave_body<NT, false, AllTilesLive>(wave, req, slice, part, live_rows, args,
                                       s_kv, s_prob, s_rescale);
  else
    wave_body<NT, true, AllTilesLive>(wave, req, slice, part, live_rows, args,
                                      s_kv, s_prob, s_rescale);
}

// ── Local merge ── one CTA per (token, head): rebase every KV part's (max, sum, O) onto one max. ──
constexpr int MergeThreads = 256, MaxDimSplit = 4;
constexpr int MaxParts = NumCU;                    // one part per CU at most; sizes the merge's weight[]

template <int DimSplit>
__global__ void __launch_bounds__(MergeThreads) mla_decode_a16w8_merge(
    float* __restrict__ partials, bf16* __restrict__ o_lat, float* __restrict__ lse,
    int H, int q_len, int rows_per_cta, int slices, int nparts, int n_ctas) {
  constexpr int Threads = MergeThreads, Waves = Threads / WaveSize;
  constexpr int PartsPerThread = cdiv(MaxParts, Threads);
  constexpr int DimsPerCta = LatDim / DimSplit, DimsPerLane = 8;
  constexpr int LanesPerRow = DimsPerCta / DimsPerLane, PartsPerWave = WaveSize / LanesPerRow;
  constexpr int PartsPerCta = Waves * PartsPerWave;
  static_assert(LanesPerRow * DimsPerLane == DimsPerCta && PartsPerWave * LanesPerRow == WaveSize);
  static_assert(Threads % WaveSize == 0, "the merge folds through one LDS slot per wave");

  const int tid = threadIdx.x, lane = tid % WaveSize, wave = tid / WaveSize;
  const int row_id = blockIdx.x / DimSplit, dim_block = blockIdx.x % DimSplit;

  const int rows_per_req = q_len * H;
  const int req = row_id / rows_per_req, row = row_id - req * rows_per_req;
  const int slice = row / rows_per_cta, slot = row - slice * rows_per_cta;
  const int ctas_per_slice = n_ctas / slices;   // B * nparts; the grid's order makes parts adjacent
  const size_t cta0 = (size_t)slice * ctas_per_slice + (size_t)req * nparts;
  const size_t out_row = row_id;                       // o_lat is [B*q_len, H, 512], packed the same
  __shared__ float weight[MaxParts];
  __shared__ float s_max[Waves], s_sum[Waves], s_fold[(Waves - 1) * DimsPerCta];

  const float* __restrict__ part_max = partial_max(partials, (size_t)n_ctas, rows_per_cta, slot);
  const float* __restrict__ part_sum = part_max + n_ctas;
  // Pass 1: fold every part's (max, sum) to one row max, then one weight per part.
  float my_max[PartsPerThread], my_sum[PartsPerThread], row_hi = NegInf;
  for (int k = 0; k < PartsPerThread; ++k) {
    const int part = tid + k * Threads;
    my_max[k] = (part < nparts) ? part_max[cta0 + part] : NegInf;
    my_sum[k] = (part < nparts) ? part_sum[cta0 + part] : 0.f;
    row_hi = fmaxf(row_hi, my_max[k]);
  }
  const float row_max =
      cta_reduce<Waves>(row_hi, s_max, wave, lane, [](float x, float y) { return fmaxf(x, y); });
  const bool row_empty = (row_max == NegInf);

  float sum_acc = 0.f;
  for (int k = 0; k < PartsPerThread; ++k) {
    const int part = tid + k * Threads;
    const float w = row_empty ? 0.f : __builtin_amdgcn_exp2f(my_max[k] - row_max);
    if (part < nparts) weight[part] = w;
    sum_acc += w * my_sum[k];
  }
  // Its __syncthreads also publishes `weight` to pass 2.
  const float denom = cta_reduce<Waves>(sum_acc, s_sum, wave, lane, [](float x, float y) { return x + y; });
  const float inv_denom = denom > 0.f ? 1.f / denom : 0.f;

  if (dim_block == 0 && tid == 0) lse[out_row] = row_empty ? neg_inf() : lse_base2(row_max, denom);

  // Pass 2: sum_p weight[p] * O_p over this CTA's slice of the 512 output dims.
  const int part_grp = lane / LanesPerRow, dim_lane = lane % LanesPerRow;
  const size_t part_stride = (size_t)rows_per_cta * LatDim;    // parts are adjacent CTAs now
  const int dim0 = dim_block * DimsPerCta + dim_lane * DimsPerLane;
  const bf16* __restrict__ src = partial_row(partials, rows_per_cta, cta0, slot) + dim0;
  float acc[DimsPerLane] = {};
  #pragma unroll 4
  for (int part = wave * PartsPerWave + part_grp; part < nparts; part += PartsPerCta) {
    const bf16x8 o_part = *(const bf16x8*)(src + (size_t)part * part_stride);
    const float w = weight[part];
    for (int i = 0; i < DimsPerLane; ++i) acc[i] = __builtin_fmaf(w, (float)o_part[i], acc[i]);
  }

  for (int mask = LanesPerRow; mask < WaveSize; mask <<= 1)
    for (int i = 0; i < DimsPerLane; ++i) acc[i] += lane_xor(acc[i], mask);
  if (wave != 0 && lane < LanesPerRow)
    for (int i = 0; i < DimsPerLane; ++i)
      s_fold[(wave - 1) * DimsPerCta + dim_lane * DimsPerLane + i] = acc[i];
  __syncthreads();

  if (wave == 0 && lane < LanesPerRow) {
    for (int w = 0; w < Waves - 1; ++w)
      for (int i = 0; i < DimsPerLane; ++i) acc[i] += s_fold[w * DimsPerCta + dim_lane * DimsPerLane + i];
    for (int i = 0; i < DimsPerLane; ++i) acc[i] *= inv_denom;
    *(bf16x8*)(o_lat + out_row * LatDim + dim0) =
        pack8(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
  }
}

// ── Cross-rank merge ── one wave per row, base 2 like the decode. RankPad is compile-time: a
// runtime rank count would spill rank_lse[] and weight[] to scratch.
template <int RankPad>
__global__ void __launch_bounds__(WaveSize) mla_dcp_lse_merge_ranks(
    const bf16* __restrict__ parts_in, const float* __restrict__ lse_in,
    bf16* __restrict__ out, float* __restrict__ lse_out, int ranks, int T, int HL) {
  static_assert(RankPad <= 16, "rank pad bound");
  constexpr int DimsPerLane = LatDim / WaveSize;
  const int row = blockIdx.x;                        // row = token * HL + head
  const int n_rows = T * HL;
  if (row >= n_rows) return;
  const int lane = (int)__lane_id();

  float rank_lse[RankPad], max_lse = NegInf;
  #pragma unroll
  for (int rank = 0; rank < RankPad; ++rank) {
    rank_lse[rank] = (rank < ranks) ? finite_lse(lse_in[(size_t)rank * n_rows + row]) : NegInf;
    max_lse = fmaxf(max_lse, rank_lse[rank]);
  }
  bf16* dst = out + (size_t)row * LatDim + lane * DimsPerLane;
  if (max_lse == NegInf) {                           // no rank saw a token for this row
    *(bf16x8*)dst = bf16x8{};
    if (lane == 0 && lse_out) lse_out[row] = neg_inf();
    return;
  }

  float weight[RankPad], weight_sum = 0.f;
  #pragma unroll
  for (int rank = 0; rank < RankPad; ++rank) {
    weight[rank] = __builtin_amdgcn_exp2f(rank_lse[rank] - max_lse);
    weight_sum += weight[rank];
  }
  const float inv = weight_sum > 0.f ? 1.0f / weight_sum : 0.f;

  const bf16* src = parts_in + (size_t)row * LatDim + lane * DimsPerLane;
  float acc[DimsPerLane] = {};
  #pragma unroll
  for (int rank = 0; rank < RankPad; ++rank) {
    if (rank >= ranks) break;
    const bf16x8 o_part = *(const bf16x8*)(src + (size_t)rank * n_rows * LatDim);
    #pragma unroll
    for (int i = 0; i < DimsPerLane; ++i) acc[i] = __builtin_fmaf(weight[rank], (float)o_part[i], acc[i]);
  }
  #pragma unroll
  for (int i = 0; i < DimsPerLane; ++i) acc[i] *= inv;
  *(bf16x8*)dst = pack8(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
  if (lane == 0 && lse_out) lse_out[row] = lse_base2(max_lse, weight_sum);
}

// ── Host-side planning ─────────────────────────────────────────────────────────────────────────

// KV parts per (request, slice): whatever is left of the machine once each has taken a CU.
__host__ inline int parts_for(int B, int slices) {
  const int groups = B * (slices > 0 ? slices : 1);
  const int per_group = NumCU / (groups > 0 ? groups : 1);
  return per_group < 1 ? 1 : per_group;
}

constexpr int SimdOf[NumWaves] = {0, 2, 1, 3, 0, 2, 1, 3};   // verified on hardware via HW_REG_HW_ID

__host__ inline int busiest_simd(int nt) {
  int per_simd[4] = {0, 0, 0, 0};
  for (int w = 0; w < NumWaves; ++w)
    per_simd[SimdOf[w]] += DimTiles * nt + (w < nt ? QkSteps : 0);
  int worst = per_simd[0];
  for (int i = 1; i < 4; ++i) worst = per_simd[i] > worst ? per_simd[i] : worst;
  return worst;
}

// Picking the tile: three costs trade off -- the walk over kv_len/(16*parts) tiles, each costing
// its busiest SIMD's matrix work; the slices `rows` cuts a request into, sharing the machine with
// the KV parts; and the CTA itself, loading Q and writing partials once. The third decides short
// shards. CtaOverhead is the one fitted constant.
constexpr float StepOverhead = 1058.0f;    // measured cycles a KV tile costs beyond its matrix work
constexpr float CtaOverhead = 6000.0f;     // fitted cycles a CTA spends per query tile before/after
constexpr int AssumedKvLen = 16384;        // stands in when the caller passes no length

// Returns 0 only when no tile fits, which is the one reason to reject a shape.
__host__ inline int pick_tile(int q_len, int H, int B, int kv_len) {
  const float kv = (float)(kv_len > 0 ? kv_len : AssumedKvLen);
  int best = 0;
  float best_cost = 0.0f;

  for (int nt = 1; nt <= MaxNTiles; ++nt) {
    const int slices = slices_of(q_len, H, nt * MfmaTile);
    if ((long long)B * slices > NumCU) continue;

    const float tiles = kv / (float)(TileTokens * parts_for(B, slices));
    const float cost = tiles * (16.0f * busiest_simd(nt) + StepOverhead) + CtaOverhead * nt;
    if (!best || cost < best_cost) {
      best_cost = cost;
      best = nt;
    }
  }
  return best;
}

constexpr size_t MallBytes = 256ull << 20;

// Shared by several slices means re-read, so this only helps at one slice past the MALL.
__host__ inline bool want_non_temporal(size_t kv_bytes, int slices) {
  return kv_bytes > MallBytes && slices == 1;
}

__host__ inline int check_domain(int B, int q_len, int H, int lat_dim, int rope_dim, int kv_len,
                                 const void* ws) {
  if (lat_dim != LatDim || rope_dim != RopeDim) return -2;
  if (H < 1 || H > MaxHeads || q_len < 1) return -3;
  if (B < 1 || pick_tile(q_len, H, B, kv_len) == 0) return -4;
  return ws ? 0 : -5;
}

template <int D = 1>   // the recursion is the dispatch table
__host__ inline void launch_merge(int dim_split, int units, int n_ctas, const KArgs& a, void* o_lat,
                                  float* lse, hipStream_t stream) {
  if constexpr (D <= MaxDimSplit) {
    if (dim_split != D && D < MaxDimSplit) {
      launch_merge<D * 2>(dim_split, units, n_ctas, a, o_lat, lse, stream); return;
    }
    mla_decode_a16w8_merge<D><<<dim3(units * D), dim3(MergeThreads), 0, stream>>>(
        a.partials, (bf16*)o_lat, lse, a.H, a.q_len, a.rows, a.slices, a.nparts, n_ctas);
  }
}

template <int NT = 1>   // AllTilesLive lets a rowless wave skip its matrix work; both arms correct
__host__ inline void launch_decode(int ntiles, int grid, bool partial_tail, const KArgs& a,
                                   hipStream_t stream) {
  if constexpr (NT <= MaxNTiles) {
    if (ntiles != NT) {
      launch_decode<NT + 1>(ntiles, grid, partial_tail, a, stream);
      return;
    }
    if (partial_tail)
      mla_decode_a16w8_kernel<NT, false><<<dim3(grid), dim3(CtaThreads), 0, stream>>>(a);
    else
      mla_decode_a16w8_kernel<NT, true><<<dim3(grid), dim3(CtaThreads), 0, stream>>>(a);
  }
}
}  // namespace

// ── extern "C" ABI ────────────────────────────────────────────────────────────────────────────
// Sized for the worst tile the planner could pick, so it never depends on the KV length hint.
extern "C" __host__ size_t mla_decode_a16w8_workspace_for(int B, int q_len, int H) {
  const int b = B > 0 ? B : 1, q = q_len > 0 ? q_len : 1, h = H > 0 ? H : 1;
  size_t worst = 0;
  for (int nt = 1; nt <= MaxNTiles; ++nt) {
    const int rows = nt * MfmaTile, slices = slices_of(q, h, rows);
    if ((long long)b * slices > NumCU) continue;           // never picked, never needs room
    const size_t n_ctas = (size_t)b * slices * parts_for(b, slices);
    const size_t need = (slab_floats(n_ctas, rows) + (size_t)rows * 2 * n_ctas) * sizeof(float);
    worst = need > worst ? need : worst;
  }
  return worst;
}

// Graph-safe: the grid comes from host values only. `kv_len` is kv_indices.numel()/B, steers just
// `pick_tile`, and 0 means unknown.
extern "C" __host__ int launch_mla_decode_a16w8(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat, void* lse,
    const void* seq_lens, const void* kv_indices, const void* kv_indptr, const void* glen,
    int cp_rank, int cp_world, int B, int q_len, int H, int lat_dim, int rope_dim,
    float scale, float kv_scale, void* workspace, void* stream_, size_t kv_bytes, int kv_len) {
  if (const int rc = check_domain(B, q_len, H, lat_dim, rope_dim, kv_len, workspace)) return rc;
  const auto stream = (hipStream_t)stream_;

  const int ntiles = pick_tile(q_len, H, B, kv_len);

  KArgs a{};
  a.q_lat = (const bf16*)q_lat;  a.q_pe = (const bf16*)q_pe;  a.c_kv = (const fp8_t*)kv_pool;
  a.partials = (float*)workspace;
  a.seq_lens = (const int*)seq_lens;  a.kv_indices = (const int*)kv_indices;
  a.kv_indptr = (const int*)kv_indptr;  a.glen = (const int*)glen;
  a.B = B;  a.q_len = q_len;  a.H = H;
  a.rows = ntiles * MfmaTile;
  a.slices = slices_of(q_len, H, a.rows);
  a.nparts = parts_for(B, a.slices);
  a.cp_rank = cp_rank;  a.cp_world = cp_world < 1 ? 1 : cp_world;
  a.scale_log2e = Log2E * scale;  a.kv_scale = kv_scale;
  a.non_temporal = want_non_temporal(kv_bytes, a.slices);

  const int grid = B * a.slices * a.nparts;         // exact: every CTA has work
  // The one case where a CTA owns rows that do not exist: a single slice `rows` does not divide.
  const bool partial_tail = (a.slices == 1) && (q_len * H % a.rows != 0);
  launch_decode<>(ntiles, grid, partial_tail, a, stream);

  const int units = B * q_len * H;                  // one merge CTA per (token, head)
  int dim_split = 1;
  while (dim_split < MaxDimSplit && units * dim_split * 2 < 4 * NumCU) dim_split <<= 1;
  launch_merge<>(dim_split, units, grid, a, o_lat, (float*)lse, stream);
  return (int)hipGetLastError();
}

template <int RP>
__host__ inline int launch_ranks_r(const void* parts_in, const void* lse_in, void* out, void* lse_out,
                                   int R, int T, int HL, hipStream_t stream) {
  mla_dcp_lse_merge_ranks<RP><<<dim3(T * HL), dim3(WaveSize), 0, stream>>>(
      (const bf16*)parts_in, (const float*)lse_in, (bf16*)out, (float*)lse_out, R, T, HL);
  return (int)hipGetLastError();
}

extern "C" __host__ int launch_mla_dcp_lse_merge_ranks(
    const void* parts_in, const void* lse_in, void* out, void* lse_out,
    int R, int T, int HL, void* stream_) {
  if (R < 1 || R > 16 || T < 1 || HL < 1) return -1;
  auto stream = (hipStream_t)stream_;
  if (R <= 2)  return launch_ranks_r<2>(parts_in, lse_in, out, lse_out, R, T, HL, stream);
  if (R <= 4)  return launch_ranks_r<4>(parts_in, lse_in, out, lse_out, R, T, HL, stream);
  if (R <= 8)  return launch_ranks_r<8>(parts_in, lse_in, out, lse_out, R, T, HL, stream);
  return launch_ranks_r<16>(parts_in, lse_in, out, lse_out, R, T, HL, stream);
}
