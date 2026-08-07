// =============================================================================
// A16W8: CDNA3 (gfx942 / MI300X) absorbed-decode MLA. bf16 activations, fp8 KV. <=16 query heads.
//   kv_lora_rank=512, qk_rope_head_dim=64, q_len == 1.
//     score[h,t] = (q_lat[h,:].c_kv[t,0:512] + q_pe[h,:].c_kv[t,512:576]) * scale
//     o_lat[h,:] = sum_t softmax_t(score)[h,t] * c_kv[t,0:512]
// W_UK/W_UV are absorbed outside the kernel, so one c_kv row is both K and V: a fused fp8 e4m3-fnuz
//   576-dim row ([0:512] latent, [512:576] rope) at one per-tensor kv_scale. `Paged` selects contiguous
//   [B,S,576] vs a fused [num_slots,1,576] pool.
//
// Q stays bf16 (no quantization, no per-head scale). KV is fp8, unpacked to bf16 for both QK and PV.
//   Every MFMA is v_mfma_f32_16x16x16bf16_1k (K=16). Probabilities are carried in bf16, so softmax uses
//   exact v_exp_f32 (not a bit-trick approximation).
//
// TileTok=64, CONS=NSub=4 (one consumer per MFMA subtile, no K-split), PROD=4. A consumer owns all 36
//   bf16 K-steps (32 latent + 4 rope) and the score rendezvous exchanges finished scores. ChunksC=8 output
//   dims per consumer, so PV needs no cross-wave reduce.
//
// ONE LDS slot (a 64-token tile is 37376 B; two don't fit). All consumer LDS reads of tile N retire inside
//   segment 1; segment 2 is free for the producer to overwrite the slot with tile N+1.
//     seg 1: QK(N) with PV(N-1) in its DS-read shadow -> pv_fetch(N) -> publish subtile score -> bar
//     seg 2: collect the 4 planes -> softmax(N) -> bar
//   Shared Q: one consumer loads and publishes q_frag through LDS.
//
// Tokens are split across CTAs FlashDecoding-style; parts>1 exports partials for the merge kernel.
// =============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstdlib>

namespace {
using bf16   = __bf16;
using bf16x4 = __attribute__((ext_vector_type(4))) bf16;
using bf16x8 = __attribute__((ext_vector_type(8))) bf16;
using f32x2  = __attribute__((ext_vector_type(2))) float;
using f32x4  = __attribute__((ext_vector_type(4))) float;
using u32x2  = __attribute__((ext_vector_type(2))) uint32_t;
using u32x4  = __attribute__((ext_vector_type(4))) uint32_t;
using fp8_t  = __hip_fp8_e4m3_fnuz;
// static_for gives a compile-time op index (unlike #pragma unroll); the fill's paged slot pick needs it.
template <int V> struct IC { static constexpr int value = V; };
template <int I, int N, class F> __device__ __forceinline__ void static_for(F&& f) {
  if constexpr (I < N) { f(IC<I>{}); static_for<I + 1, N>(f); }
}
// Probability lift: probabilities in (0,1] are lifted by PLift before the bf16 cvt, divided back in the
//   epilogue (exact, power of two). Rescale slack: run_max may lag the true max by RSLACK before rescaling.
constexpr int   LOG2PLIFT = 7;
constexpr float kLog2PLift = (float)(LOG2PLIFT);
constexpr float PLift      = (float)(1 << LOG2PLIFT);   // 128
constexpr float RSLACK = 0.8f;
__device__ __forceinline__ f32x4 mfma_bf16(bf16x4 a, bf16x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
// Pack 4 f32 as bf16x4 by truncation (exact for fp8-derived values; see header).
__device__ __forceinline__ bf16x4 pack4_trunc(float a, float b, float c, float d) {
  const uint32_t au = __builtin_bit_cast(uint32_t, a), bu = __builtin_bit_cast(uint32_t, b);
  const uint32_t cu = __builtin_bit_cast(uint32_t, c), du = __builtin_bit_cast(uint32_t, d);
  const uint32_t w0 = __builtin_amdgcn_perm(bu, au, 0x07060302u);
  const uint32_t w1 = __builtin_amdgcn_perm(du, cu, 0x07060302u);
  bf16x4 o; __builtin_memcpy(&o, &w0, 4); __builtin_memcpy((char*)&o + 4, &w1, 4); return o;
}
// One LDS dword = 4 fp8 dims of one token -> one bf16 MFMA A-fragment.
__device__ __forceinline__ bf16x4 fp8x4_to_bf16x4(uint32_t w) {
  const f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false);
  const f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
  return pack4_trunc(lo[0], lo[1], hi[0], hi[1]);
}
// 4x4 byte transpose: in[j]=dword of token j (4 dims), out[p]=one dim across 4 tokens.
__device__ __forceinline__ void t4x4(const uint32_t* in, uint32_t* out) {
  const uint32_t t0 = __builtin_amdgcn_perm(in[1], in[0], 0x05010400u);
  const uint32_t t1 = __builtin_amdgcn_perm(in[1], in[0], 0x07030602u);
  const uint32_t t2 = __builtin_amdgcn_perm(in[3], in[2], 0x05010400u);
  const uint32_t t3 = __builtin_amdgcn_perm(in[3], in[2], 0x07030602u);
  out[0] = __builtin_amdgcn_perm(t2, t0, 0x05040100u);
  out[1] = __builtin_amdgcn_perm(t2, t0, 0x07060302u);
  out[2] = __builtin_amdgcn_perm(t3, t1, 0x05040100u);
  out[3] = __builtin_amdgcn_perm(t3, t1, 0x07060302u);
}
__device__ __forceinline__ void sched_fence() { __builtin_amdgcn_sched_barrier(0); }
// Raw s_barrier (asm, not the IntrNoMem intrinsic) so LDS accesses are not hoisted across it.
__device__ __forceinline__ void bar() { asm volatile("s_barrier" ::: "memory"); }
__device__ __forceinline__ void wait_lgkm0_only() { __builtin_amdgcn_s_waitcnt(0xC07F); }
// KV loads are inline asm so drain_to's hand-counted vmcnt is the only wait. NT bypasses L2/MALL (gated
//   host-side: only when the KV working set exceeds the 256 MB MALL).
template <bool NT>
__device__ __forceinline__ u32x4 gload_x4(const void* p) {
  u32x4 r;
  if constexpr (NT) asm volatile("global_load_dwordx4 %0, %1, off nt" : "=&v"(r) : "v"(p));
  else              asm volatile("global_load_dwordx4 %0, %1, off"    : "=&v"(r) : "v"(p));
  return r;
}
template <bool NT>
__device__ __forceinline__ u32x4 gload_x4_s(const void* sbase, uint32_t voff) {
  u32x4 r;
  if constexpr (NT) asm volatile("global_load_dwordx4 %0, %1, %2 nt" : "=&v"(r) : "v"(voff), "s"(sbase));
  else              asm volatile("global_load_dwordx4 %0, %1, %2"    : "=&v"(r) : "v"(voff), "s"(sbase));
  return r;
}
// Per-lane value select against an explicit lane mask. The mask MUST come from a ballot (not a constant
//   `~0ULL << L`): LLVM mis-materialises the SGPR pair and mis-selects lanes 32..63.
__device__ __forceinline__ int vsel_m(int a, int b, uint64_t m) {
  int r;
  asm("v_cndmask_b32_e64 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "s"(m));
  return r;
}
#define LANE_GE(L) ((uint64_t)__builtin_amdgcn_ballot_w64((int)(threadIdx.x & (WaveSize - 1)) >= (L)))
constexpr int   MfmaTile = 16;
constexpr int   QuadSpan = 4;
constexpr int   WaveSize = 64;
constexpr int   CONS     = 4;                   // consumer waves (== NSub: one per subtile, no K-split)
constexpr int   PROD     = 4;                   // producer waves; the fill splits a tile 4 ways
constexpr int   Waves    = CONS + PROD;         // 8
constexpr int   Threads  = Waves * WaveSize;    // 512
constexpr int   LatDim   = 512;
constexpr int   RopeDim  = 64;
constexpr int   KQ       = 16;                  // bf16 MFMA contraction length (QK and PV)
constexpr int   QLane    = 4;                   // bf16 K-elems per lane per K-step
constexpr int   Fp8Lane  = 8;                   // fp8 elems per lane per PV LDS read
constexpr int   LatIters = LatDim  / KQ;        // 32 latent K-steps
constexpr int   RopeIters= RopeDim / KQ;        // 4  rope   K-steps
constexpr int   PvChunks = LatDim / MfmaTile;   // 32 PV out-dim tiles
constexpr float NegInf   = -3.0e38f;
constexpr int TileTok = 64;                      // KV tokens per tile
constexpr int NSub    = TileTok / MfmaTile;      // 4 MFMA M-subtiles
constexpr int NPV     = TileTok / (Fp8Lane * QuadSpan);  // 2 PV read groups of 32 tokens
// CONS == NSub: one consumer per subtile, all 36 K-steps, finished-score rendezvous. Each consumer
//   softmaxes the whole tile and owns ChunksC disjoint PV output-dim chunks (no cross-wave PV reduce).
constexpr int   ChunksC     = PvChunks / CONS;    // 8 (128 dims)
constexpr int   QkSteps     = LatIters + RopeIters;       // 36
static_assert(CONS == NSub && TileTok % (Fp8Lane * QuadSpan) == 0,
              "CONS must be one consumer per subtile, and the tile a whole number of PV read groups");
// PV layout. PvWide dims per lane per LDS read; NPV read groups × 2 lo/hi K-steps = 4 MFMAs per out-dim tile.
constexpr int PvWide = 2 * QuadSpan;                      // 8 fp8 dims per lane per LDS read
constexpr int PvHalf = PvWide / QuadSpan;                 // dwords per token (2)
constexpr int NG = ChunksC / PvWide;                      // dim-groups (1)
constexpr int PvC = NG * PvHalf;                          // t4x4 blocks per read group (2)
constexpr int PvDw = NPV * PvC * Fp8Lane;                 // dwords = whole PV operand set
static_assert(PvC * QuadSpan == ChunksC, "o_acc must be exactly the (t4x4 block, dim) product");
__device__ __forceinline__ float smul(float a, float b) {
  float r; asm volatile("v_mul_f32_e32 %0,%1,%2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
// f32 to bf16 RTNE by hand (gfx942 has no v_cvt_bf16_f32). NaN is not handled.
__device__ __forceinline__ unsigned bf16_rtne_nonan(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return u + 0x7FFFu + ((u >> 16) & 1u);
}
__device__ __forceinline__ bf16x4 pack4(float a, float b, float c, float d) {
  const unsigned w0 = __builtin_amdgcn_perm(bf16_rtne_nonan(b), bf16_rtne_nonan(a), 0x07060302u);
  const unsigned w1 = __builtin_amdgcn_perm(bf16_rtne_nonan(d), bf16_rtne_nonan(c), 0x07060302u);
  bf16x4 o; __builtin_memcpy(&o, &w0, 4); __builtin_memcpy((char*)&o + 4, &w1, 4); return o;
}
__device__ __forceinline__ bf16x8 pack8(float a, float b, float c, float d,
                                        float e, float f, float g, float h) {
  const bf16x4 lo = pack4(a, b, c, d), hi = pack4(e, f, g, h);
  bf16x8 o; __builtin_memcpy(&o, &lo, 8); __builtin_memcpy((char*)&o + 8, &hi, 8); return o;
}
// Cross-lane reduce over the MFMA-C quad axis {mrow,+16,+32,+48} (gfx942 has no v_permlane; DPP can't cross 16).
__device__ inline float lane_bperm(float v, int lane) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(lane << 2, __builtin_bit_cast(int, v)));
}
__device__ inline float lane_xor16(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401F));
}
__device__ inline float wave_max(float v) { const int l = __lane_id(); v = fmaxf(v, lane_xor16(v)); return fmaxf(v, lane_bperm(v, l ^ 32)); }
__device__ inline float wave_sum(float v) { const int l = __lane_id(); v += lane_xor16(v); return v + lane_bperm(v, l ^ 32); }
__device__ __forceinline__ float lane_xor(float v, int m) { return lane_bperm(v, __lane_id() ^ m); }
constexpr int    NumCU    = 304;               // MI300X multiProcessorCount
constexpr int    MaxParts = 384;
constexpr int    StepTok  = TileTok;
constexpr int    PartHead = 4;                 // fp32 header slots (ABI-stable workspace sizing only)
constexpr size_t MaxCTAs  = 2048 + 2 * NumCU;
constexpr int    MaxHeads = 16;
constexpr int    MaxQL    = 8;   // widest draft window
// Partial workspace: bf16 [cta][head][512] payload slab + head-major (run_max,run_sum) header planes.
constexpr size_t POElems  = MaxCTAs * (size_t)MaxHeads * LatDim;
constexpr size_t HdrBase  = POElems / 2;
__device__ __host__ __forceinline__ size_t hdr_off(int h, int k) { return HdrBase + ((size_t)h * 2 + k) * MaxCTAs; }

extern "C" __host__ size_t mla_decode_a16w8_workspace_bytes() {
  static_assert(HdrBase + 2 * (size_t)MaxHeads * MaxCTAs <= MaxCTAs * (size_t)MaxHeads * (PartHead + LatDim));
  return MaxCTAs * (size_t)MaxHeads * (PartHead + LatDim) * sizeof(float);   // fp32-sized (ABI-stable)
}

// Kernel. Paged=false: contiguous [B,S,576], S from host. Paged=true: fused [num_slots,1,576] pool,
//   S from device seq_lens, `rows` picks the q/out row, `parts` fixed at capture.
template <bool Paged = false, bool NT = false>
__global__ void __launch_bounds__(Threads, 1) mla_decode_a16w8_kernel(
    const bf16*  __restrict__ q_lat,  // [B, 16, 512] bf16
    const bf16*  __restrict__ q_pe,   // [B, 16, 64]  bf16
    const fp8_t* __restrict__ c_kv,   // [B, S, 576] fp8 | fused [num_slots,1,576] pool (Paged)
    const bf16*  __restrict__ k_pe,   // UNUSED: rope lives in c_kv. ABI only.
    bf16* __restrict__ o_lat,         // [B, 16, 512] (parts==1)
    float* __restrict__ partials,     // fp32 split workspace (parts>1)
    int B, int H, int S_scalar, int parts, float scale_log2e, float kv_scale, float scale_p,
    int q_len,                                     // draft window; 1 = plain decode
    const int* __restrict__ seq_lens = nullptr,    // [B] device (Paged)
    const int* __restrict__ rows = nullptr,        // [B] q/out row (Paged; null -> b)
    const int* __restrict__ kv_indices = nullptr,  // [sum seq] slots (Paged)
    const int* __restrict__ kv_indptr = nullptr) { // [B+1] slot offsets (Paged)
  constexpr int FusedDim = LatDim + RopeDim;       // 576
  // Draft window at QLC=1: one draft position per CTA, so the PV accumulator and
  //   q_frag budgets are exactly the q_len==1 ones and the register allocation is
  //   untouched. The multiq kernel keeps up to DraftPerCta=4 positions resident, which it
  //   can afford only because its fp8 q_frag is 36 VGPR instead of this kernel's 72.
  //   The cost here is that the QGroups=q_len CTAs sharing a (batch, kv-part) each
  //   re-read the KV stream; they are laid out FASTEST on the grid axis so those
  //   duplicate reads are one tile apart and land in MALL/L2 rather than HBM.
  const int cta  = blockIdx.x;
  const int qpos = q_len > 1 ? cta % q_len : 0;
  const int rest = q_len > 1 ? cta / q_len : cta;
  const int part = rest % parts;
  const int b    = rest / parts;
  const int S      = Paged ? seq_lens[b] : S_scalar;
  // Causal inside the draft window: position qpos attends KV [0, S - (q_len-1-qpos)).
  //   seq_lens INCLUDES the q_len draft tokens just written (verified against a
  //   per-position q_len=1 reference), so excluding the window here would make every
  //   query attend q_len tokens too few against stale KV.
  const int tok_lim = S - (q_len - 1 - qpos);
  const int q_row  = (Paged && rows) ? rows[b] : b;
  const int slot0  = Paged ? kv_indptr[b] : 0;
  const bool single_part = (parts == 1);
  const int lane = threadIdx.x & (WaveSize - 1);
  const int wave = threadIdx.x / WaveSize;
  const bool is_cons = (wave < CONS);
  const int cwave = wave;
  static_assert(TileTok % PROD == 0, "the fill must split the tile evenly across the producer waves");
  const int pwave = wave - CONS;
  const int mrow = lane & (MfmaTile - 1);
  const int quad = lane / MfmaTile;
  const int chunk_lo = cwave * ChunksC;
  const int msub = cwave;   // CONS == NSub, so msub == cwave
  // PV-order token relabeling: group u wants P[token = u*32 + quad*8 + 0..7]. Subtile n=2u+v feeds group u
  //   at slots v*4+0..3, so a lane's 8 group-u probs are sarr[2u][0..3] then sarr[2u+1][0..3].
  auto tok_row = [&](int n, int m) {
    return (n >> 1) * (Fp8Lane * QuadSpan) + (m >> 2) * Fp8Lane + (n & 1) * QuadSpan + (m & 3);
  };
  const int part_tok = ((S + parts * StepTok - 1) / (parts * StepTok)) * StepTok;
  const int kv_lo = part * part_tok;
  const int kv_hi = min(S, kv_lo + part_tok);
  const int ntiles = (kv_hi - kv_lo + TileTok - 1) / TileTok;
  // Token-major LDS row: 512 fp8 latent + 64 fp8 rope + 8 pad. The pad (RowBytes=584 = 18 mod 32 dw) is the
  //   bank swizzle so a quad's 16 rows hit 16 distinct bank pairs.
  constexpr int LatBytes   = LatDim;
  constexpr int RopeBytes  = RopeDim;
  constexpr int RowBytes   = LatBytes + RopeBytes + 8;   // 584
  constexpr int TileElems  = TileTok * RowBytes;
  // Score rendezvous: CONS planes, one per consumer, +8 pad for bank-shift.
  constexpr int RvSlots  = MfmaTile * QuadSpan;
  constexpr int RvStride = RvSlots * QuadSpan + 8;
  // ONE LDS slot (not a ring): all tile-N reads retire in seg 1, seg 2 overwrites with tile N+1.
  // Shared Q: one consumer loads it; all read back after the prologue barrier.
  constexpr int QStage = QkSteps * WaveSize * (int)sizeof(bf16x4);
  static_assert(TileElems + CONS * RvStride * (int)sizeof(float) + QStage <= 65536, "LDS over 64 KB");
  __shared__ fp8_t  sKv[TileElems];
  __shared__ float  sRv[CONS][RvStride];
  __shared__ bf16x4 sQ[QkSteps][WaveSize];
  // Q prologue: one consumer loads Q and publishes to sQ. Loads are unconditional (an `active` guard
  //   spills inside the saveexec block); the guard moves onto the stored fragment.
  bf16x4 q_frag[QkSteps] = {};
  const bool q_owner = is_cons && cwave == 0;
  if (q_owner) {
    const bool active = mrow < H;
    const size_t qh = (size_t)((q_row * q_len + qpos) * H + (active ? mrow : 0));
    const bf16* ql = q_lat + qh * LatDim;
    const bf16* qp = q_pe  + qh * RopeDim;
    auto qstep = [&](int step) -> bf16x4 {
      const bf16* src = (step < LatIters) ? (ql + (size_t)step * KQ)
                                          : (qp + (size_t)(step - LatIters) * KQ);
      return *(const bf16x4*)(src + quad * QLane);
    };
    const bf16x4 zero = {};
    #pragma unroll
    for (int k = 0; k < QkSteps; ++k) {
      bf16x4 f = qstep(k);
      sQ[k][lane] = active ? f : zero;
    }
    wait_lgkm0_only();
  }
  const float qk_dsc = kv_scale * scale_log2e;   // no q_scale: q was never quantized
  float run_max = NegInf, run_sum = 0.f;
  f32x4 o_acc[ChunksC] = {};
  // ── Producer fill infrastructure ──
  constexpr int ToksPerFill = TileTok / PROD;             // 16 rows/producer = 9216 B, tail-free (9 dwordx4)
  constexpr int ShareB  = ToksPerFill * FusedDim;
  constexpr int WX4     = ShareB / (WaveSize * 16);       // 9
  static_assert(ShareB == WX4 * WaveSize * 16, "wide fill: tail-free");
  static_assert(FusedDim % 16 == 0, "a 16-B chunk must never straddle a token row");
  using FillV = u32x4;
  constexpr int FillRegs   = WX4;
  constexpr int FillEvents  = FillRegs;
  constexpr int FDEPTH    = 2;                            // tiles of loads in flight
  // Paged slot-id prefetch: scalar loads (lgkmcnt, never vmcnt) issued in the drain window, branch-free.
  const int fwv = __builtin_amdgcn_readfirstlane(pwave);
  auto load_slots = [&](int iter_, int* dst) {
    if constexpr (Paged) {
      const int it   = iter_ < ntiles ? iter_ : (ntiles > 0 ? ntiles - 1 : 0);
      const int kv0  = kv_lo + it * TileTok;
      const int tc   = min(TileTok, kv_hi - kv0);
      const int last = tc > 0 ? tc - 1 : 0;
      #pragma unroll
      for (int tw = 0; tw < ToksPerFill; ++tw) {
        const int t    = fwv * ToksPerFill + tw;
        const int ltok = t < tc ? t : last;
        const int gi   = kv0 + ltok;
        const int gc   = gi < S ? gi : (S > 0 ? S - 1 : 0);
        dst[tw] = kv_indices[slot0 + gc];
      }
    }
  };
  const fp8_t* kv_base = Paged ? c_kv : (c_kv + (size_t)b * S * FusedDim);
  // Per-lane fill maps. Must be indexed from static_for (a runtime index spills to scratch).
  uint32_t wgoff[FillRegs], wldso[FillRegs];
  const uint32_t wb16 = (uint32_t)(pwave * ShareB + lane * 16);
  static_for<0, FillRegs>([&](auto ic) {
    constexpr int j  = decltype(ic)::value;
    constexpr int B0 = j * WaveSize * 16;
    const int off = B0 + lane * 16;
    const int tl  = off / FusedDim;
    wgoff[j] = (uint32_t)(off - tl * FusedDim);
    wldso[j] = (uint32_t)((pwave * ToksPerFill + tl) * RowBytes + (off - tl * FusedDim));
  });
  // CORRECTNESS: exactly one definition site per dst[j] (inline asm loads; two paths make LLVM insert a
  //   v_mov that reads the register before the load lands). The pad clamp folds into the VGPR offset.
  auto fill_load = [&](int kv0, int tc, const int* slots, FillV (&dst)[FillRegs]) {
    const fp8_t* rb = Paged ? kv_base : kv_base + (size_t)kv0 * FusedDim;
    static_for<0, FillRegs>([&](auto ic) {
      constexpr int j  = decltype(ic)::value;
      constexpr int B0 = j * WaveSize * 16;
      if constexpr (Paged) {
        // Pick the slot id for each lane's chunk: a 1024-B op spans at most 3 of the 576-B rows, so the
        //   slot is selected per-row via lane-mask cndmask (which rows is a compile-time function of j).
        constexpr int t0 = B0 / FusedDim;
        constexpr int tN = (B0 + (WaveSize - 1) * 16) / FusedDim;
        static_assert(tN <= t0 + 2 && tN < ToksPerFill, "wide fill: op spans more rows than the slot pick");
        int src = slots[t0];
        if constexpr (tN >= t0 + 1) {
          constexpr int L1 = ((t0 + 1) * FusedDim - B0 + 15) / 16;
          src = vsel_m(src, slots[t0 + 1], LANE_GE(L1));
        }
        if constexpr (tN >= t0 + 2) {
          constexpr int L2 = ((t0 + 2) * FusedDim - B0 + 15) / 16;
          src = vsel_m(src, slots[t0 + 2], LANE_GE(L2));
        }
        dst[j] = gload_x4<NT>(rb + (size_t)src * FusedDim + wgoff[j]);
      } else {
        const int t    = pwave * ToksPerFill + (B0 + (int)lane * 16) / FusedDim;
        const int over = t - (tc - 1);
        const uint32_t voff = wb16 + B0 - (uint32_t)((over > 0 ? over : 0) * FusedDim);
        dst[j] = gload_x4_s<NT>(rb, voff);
      }
    });
  };
  // Commit landed registers to the LDS tile slot (two 8-B stores per 16-B chunk; ds_write_b128 is illegal
  //   on the 584-B stride, but the pair folds into one ds_write2_b64).
  auto fill_store = [&](const FillV (&src)[FillRegs]) {
    fp8_t* tm = sKv;
    static_for<0, FillRegs>([&](auto ic) {
      constexpr int j = decltype(ic)::value;
      *(u32x2*)&tm[wldso[j]]     = u32x2{src[j][0], src[j][1]};
      *(u32x2*)&tm[wldso[j] + 8] = u32x2{src[j][2], src[j][3]};
    });
  };
  // ── QK: contract all 36 K-steps over this wave's subtile -> one complete f32x4 score ──
  // `filler` is register-only work (PV(N-1) then pv_fetch(N)) run inside the DS-read shadow.
  auto qk_subtile_score = [&](auto&& filler) {
    sched_fence();
    const fp8_t* rb = &sKv[tok_row(msub, mrow) * RowBytes + quad * QLane];
    uint32_t a[QkSteps];
    #pragma unroll
    for (int k = 0; k < QkSteps; ++k) a[k] = *(const uint32_t*)(rb + k * KQ);
    filler();
    f32x4 s = {};
    #pragma unroll
    for (int k = 0; k < QkSteps; ++k) s = mfma_bf16(fp8x4_to_bf16x4(a[k]), q_frag[k], s);
    // Four sched groups so the filler's MFMAs cover the QK read latency and vice versa.
    __builtin_amdgcn_sched_group_barrier(0x100, (QkSteps + 1) / 2, 0);   // QK DS reads
    __builtin_amdgcn_sched_group_barrier(0x008, NPV * 2 * ChunksC, 0);    // PV(N-1) MFMA
    __builtin_amdgcn_sched_group_barrier(0x100, PvDw / (2 * PvHalf), 0);  // pv_fetch DS reads
    __builtin_amdgcn_sched_group_barrier(0x008, QkSteps, 0);              // QK MFMA
    sched_fence();
    return s;
  };
  // Score rendezvous: publish to own plane, bar, collect all NSub planes (incl own — cheaper than a switch).
  auto score_publish = [&](f32x4 s) {
    *(f32x4*)&sRv[cwave][(quad * MfmaTile + mrow) * QuadSpan] = s;
  };
  auto score_collect = [&](f32x4* sarr) {
    const int slot = (quad * MfmaTile + mrow) * QuadSpan;
    static_for<0, NSub>([&](auto ic) {
      constexpr int n = decltype(ic)::value;
      sarr[n] = *(const f32x4*)&sRv[n][slot];
    });
  };
  // PV fetch: pull the whole tile's PV input into registers (segment 1, alongside QK). No clamp needed
  //   (spans 0..63 by construction; pad rows carry prob 0).
  auto pv_fetch = [&](uint32_t* D) {
    const fp8_t* tm = sKv;
    #pragma unroll
    for (int u = 0; u < NPV; ++u)
      #pragma unroll
      for (int g = 0; g < NG; ++g) {
        const int col = chunk_lo * MfmaTile + g * (PvWide * MfmaTile) + PvWide * mrow;
        #pragma unroll
        for (int j = 0; j < Fp8Lane; ++j) {
          const u32x2 v = *(const u32x2*)&tm[(u * (Fp8Lane * QuadSpan) + quad * Fp8Lane + j) * RowBytes + col];
          #pragma unroll
          for (int h = 0; h < PvHalf; ++h) D[((u * NG + g) * PvHalf + h) * Fp8Lane + j] = v[h];
        }
      }
  };
  // PV MFMA: transpose D, accumulate into o_acc. Register-only, run as qk_subtile_score's filler.
  auto pv_mfma = [&](const uint32_t* D, const bf16x8* probs) {
    #pragma unroll
    for (int u = 0; u < NPV; ++u) {
      const bf16x4 ph[2] = {{probs[u][0], probs[u][1], probs[u][2], probs[u][3]},
                            {probs[u][4], probs[u][5], probs[u][6], probs[u][7]}};
      #pragma unroll
      for (int c = 0; c < PvC; ++c) {
        uint32_t d[2][QuadSpan];
        const uint32_t* Du = &D[(u * PvC + c) * Fp8Lane];
        t4x4(Du, d[0]); t4x4(Du + QuadSpan, d[1]);
        #pragma unroll
        for (int p = 0; p < QuadSpan; ++p) {
          f32x4& acc = o_acc[c * QuadSpan + p];
          acc = mfma_bf16(fp8x4_to_bf16x4(d[0][p]), ph[0], acc);
          acc = mfma_bf16(fp8x4_to_bf16x4(d[1][p]), ph[1], acc);
        }
      }
    }
  };
  // Rescale o_acc (scalar v_mul: packed fp32 and MFMA share the FP32 MAC datapath).
  auto rescale_oacc = [&](float rescale) {
    #pragma unroll
    for (int c = 0; c < ChunksC; ++c) {
      o_acc[c][0] = smul(o_acc[c][0], rescale); o_acc[c][1] = smul(o_acc[c][1], rescale);
      o_acc[c][2] = smul(o_acc[c][2], rescale); o_acc[c][3] = smul(o_acc[c][3], rescale);
    }
  };
  // Online softmax for one tile: fold scores into (run_max, run_sum, o_acc), emit NPV bf16x8 prob operands.
  auto softmax_tile = [&](f32x4* sarr, int kv0, bf16x8* probs) {
    // Tail mask: zero out scores past the valid token count (only the last tile of a part is partial).
    const int tc = min(kv_hi, tok_lim) - kv0;
    const bool full = (tc >= TileTok);
    float bmax = NegInf;
    if (__builtin_expect(!full, 0)) {
      asm volatile("" ::: "memory");   // keep the branch (if-convert costs ~30 VALU ops/tile)
      #pragma unroll
      for (int n = 0; n < NSub; ++n)
        #pragma unroll
        for (int e = 0; e < QuadSpan; ++e)
          if ((n >> 1) * (Fp8Lane * QuadSpan) + quad * Fp8Lane + (n & 1) * QuadSpan + e >= tc) sarr[n][e] = NegInf;
    }
    // Reduce this lane's 16 scores to a per-lane max, then scale into log2 units.
    #pragma unroll
    for (int n = 0; n < NSub; ++n)
      #pragma unroll
      for (int e = 0; e < QuadSpan; ++e) bmax = fmaxf(bmax, sarr[n][e]);
    bmax *= qk_dsc;
    // Rescale slack: only update run_max (and rescale o_acc) if a score exceeds it by more than RSLACK.
    //   wave_max is needed only on rescaling tiles; the ballot makes the guard wave-uniform.
    const bool first = (run_max == NegInf);
    const bool moved = (__ballot(bmax > run_max + (RSLACK)) != 0ull);
    float nmax = run_max;
    if (__builtin_expect(moved, 0)) nmax = fmaxf(run_max, wave_max(bmax));
    // `dead`: this part has seen no live token yet -> emit zero probs. readfirstlane makes it scalar.
    const bool dead_p = first && (nmax <= -1.0e30f);
    const bool dead = __builtin_amdgcn_readfirstlane(dead_p ? 1 : 0) != 0;
    // Compute probabilities: p = exp2(score*dsc - nmax + log2(PLift)). PLift lifts (0,1] into a range
    //   that uses more of bf16's precision; the epilogue divides it back out (exact, power of two).
    const float negm = -nmax + kLog2PLift;
    float bsum = 0.f;
    float pe[NSub * QuadSpan];
    if (__builtin_expect(dead, 0)) {
      #pragma unroll
      for (int i = 0; i < NSub * QuadSpan; ++i) pe[i] = 0.f;
    } else {
      #pragma unroll
      for (int n = 0; n < NSub; ++n)
        #pragma unroll
        for (int e = 0; e < QuadSpan; ++e) {
          const float p = __builtin_amdgcn_exp2f(__builtin_fmaf(sarr[n][e], qk_dsc, negm));
          bsum += p; pe[n * QuadSpan + e] = p;
        }
    }
    bsum *= (1.0f / PLift);   // unlift the denominator back to true units
    // Pack this lane's 16 probs into NPV bf16x8 B-operands (one per PV read group).
    #pragma unroll
    for (int u = 0; u < NPV; ++u)
      probs[u] = pack8(pe[u * Fp8Lane], pe[u * Fp8Lane + 1], pe[u * Fp8Lane + 2], pe[u * Fp8Lane + 3],
                       pe[u * Fp8Lane + 4], pe[u * Fp8Lane + 5], pe[u * Fp8Lane + 6], pe[u * Fp8Lane + 7]);
    // Lazy rescale: if run_max didn't move, the factor is exp2(0)==1, so skip. Otherwise rescale the
    //   running o_acc and run_sum by exp2(old_max - new_max) to keep them in a consistent scale.
    if (!moved) {
      run_sum = run_sum + bsum;
    } else {
      const float rescale = first ? 0.f : __builtin_amdgcn_exp2f(run_max - nmax);
      run_sum = run_sum * rescale + bsum;
      rescale_oacc(rescale);
      run_max = nmax;
    }
  };
  // ── Warp-specialized pipeline: 2 barriers/tile, no other handshake ──
  //   producer: refill FDEPTH ahead, bar#1, drain+ds_write tile N+1, bar#2
  //   consumer: QK(N)+PV(N-1)+pv_fetch(N), publish, bar#1, collect, softmax(N), bar#2
  if (!is_cons) {
    // Producer: two VGPR register buffers (R0, R1) rotate across tiles. FDEPTH=2 tiles of loads are in
    //   flight at once; the LDS slot is overwritten only in segment 2 (after consumers finish reading).
    FillV R0[FillRegs], R1[FillRegs];
    int b0[ToksPerFill] = {}, b1[ToksPerFill] = {}, b2[ToksPerFill] = {};  // rotating paged slot-id banks
    auto tile_tc = [&](int it) { return min(TileTok, kv_hi - (kv_lo + it * TileTok)); };
    auto load_tile = [&](int it, const int* slots, FillV (&dst)[FillRegs]) {
      if (it < ntiles) fill_load(kv_lo + it * TileTok, tile_tc(it), slots, dst);
    };
    // Wait for tile `it` to land, leaving deeper tiles in flight. Overcounting on the tail is silent corruption.
    auto drain_to = [&](int it) {
      const int rem = ntiles - it;
      if (rem >= FDEPTH) asm volatile("s_waitcnt vmcnt(%0)" :: "i"((FDEPTH - 1) * FillEvents) : "memory");
      else               asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    };
    // Refill `cur` FDEPTH tiles ahead, and rotate the paged slot-id banks (no-op when contiguous).
    auto refill = [&](int it, FillV (&cur)[FillRegs]) {
      load_tile(it + FDEPTH, b2, cur);
      load_slots(it + FDEPTH + 1, b0);
      #pragma unroll
      for (int tw = 0; tw < ToksPerFill; ++tw) {
        const int t = b0[tw]; b0[tw] = b1[tw]; b1[tw] = b2[tw]; b2[tw] = t;
      }
    };
    load_slots(0, b0);
    load_tile(0, b0, R0); load_slots(1, b1);
    load_tile(1, b1, R1); load_slots(2, b2);
    auto store_pub = [&](int it, FillV (&cur)[FillRegs]) {
      drain_to(it);       // wait for tile `it` to land in registers
      fill_store(cur);    // ds_write into the LDS slot
      wait_lgkm0_only();  // ds_writes retired -> next barrier publishes them
    };
    // Prologue: publish tile 0, then enter the 2-barrier loop.
    if (ntiles > 0) store_pub(0, R0);
    bar();                                               // PROLOGUE: tile 0 readable
    // Main loop: rb = buffer stored last iteration (holds tile t), sb = buffer holding tile t+1.
    //   seg 1: refill rb (FDEPTH ahead), bar#1 (consumers finished reading tile t)
    //   seg 2: drain+store tile t+1 into the slot, bar#2 (tile t+1 readable)
    auto bstep2 = [&](int t, FillV (&rb)[FillRegs], FillV (&sb)[FillRegs]) {
      refill(t, rb);                                     // segment 1
      bar();                                             // #1
      if (t + 1 < ntiles) store_pub(t + 1, sb);
      bar();                                             // #2
    };
    #pragma unroll 1
    for (int t = 0; t < ntiles; t += 2) {
      bstep2(t, R0, R1);
      if (t + 1 < ntiles) bstep2(t + 1, R1, R0);
    }
    return;
  }
  // ── Consumer ──
  bf16x8 prev_probs[NPV] = {}; bool have_prev = false;
  uint32_t Dp[PvDw] = {};
  bar();                                               // PROLOGUE: tile 0 readable AND Q published
  // Read Q back from LDS (published by consumer 0). All consumers need the full 36-step q_frag.
  static_for<0, QkSteps>([&](auto ic) {
    constexpr int k = decltype(ic)::value;
    q_frag[k] = sQ[k][lane];
  });
  #pragma unroll 1
  for (int iter = 0; iter < ntiles; ++iter) {
    // Filler: PV(N-1) then pv_fetch(N) — PV(N-1) must precede pv_fetch (overwrites Dp). Both unconditional
    //   (iteration 0 accumulates zeros).
    const f32x4 sp = qk_subtile_score([&] { pv_mfma(Dp, prev_probs); pv_fetch(Dp); });
    score_publish(sp);
    wait_lgkm0_only();
    bar();                                             // #1: scores published AND tile-N reads retired
    f32x4 s[NSub]; score_collect(s);
    softmax_tile(s, kv_lo + iter * TileTok, prev_probs);
    have_prev = true;
    wait_lgkm0_only();
    bar();                                             // #2: planes read, tile N+1 readable
  }
  if (have_prev) { sched_fence(); pv_mfma(Dp, prev_probs); sched_fence(); }

  // Epilogue: run_sum cross-lane fold (all 64 lanes), then store o_acc with the dequant scale folded in.
  run_sum = wave_sum(run_sum);
  const int head = mrow;
  if (head < H) {
    // o_acc[g*PvWide+i][e] holds dim = chunk_lo*16 + g*128 + PvWide*(4*quad+e) + i.
    auto store_o = [&](bf16* dst, float sc) {
      #pragma unroll
      for (int g = 0; g < NG; ++g)
        #pragma unroll
        for (int e = 0; e < QuadSpan; ++e) {
          const int dim = chunk_lo * MfmaTile + g * (PvWide * MfmaTile) + PvWide * (QuadSpan * quad + e);
          *(bf16x8*)&dst[dim] = pack8(o_acc[g * PvWide + 0][e] * sc, o_acc[g * PvWide + 1][e] * sc,
                                      o_acc[g * PvWide + 2][e] * sc, o_acc[g * PvWide + 3][e] * sc,
                                      o_acc[g * PvWide + 4][e] * sc, o_acc[g * PvWide + 5][e] * sc,
                                      o_acc[g * PvWide + 6][e] * sc, o_acc[g * PvWide + 7][e] * sc);
        }
    };
    // single_part: normalize and write directly to o_lat. Both dequants (kv_scale for V, /PLift for the
    //   lift) fold into the normalization factor inv.
    if (single_part) {
      const float inv = (run_sum > 0.f) ? kv_scale / (run_sum * PLift) : 0.f;
      store_o(o_lat + (size_t)((b * q_len + qpos) * H + head) * LatDim, inv);
    } else {
      // parts>1: export un-normalized but fully dequantized O-partials (run_sum is in true units, so the
      //   payload must be too). The merge kernel reweights by w_p = exp2(max_p - max) and normalizes.
      const float pdq = kv_scale * (1.0f / PLift);
      if (cwave == 0 && quad == 0) {
        partials[hdr_off(head, 0) + cta] = run_max;   // header: (run_max, run_sum) per CTA
        partials[hdr_off(head, 1) + cta] = run_sum;
      }
      store_o((bf16*)partials + (size_t)(cta * MaxHeads + head) * LatDim, pdq);
    }
  }
}

// Merge kernel (parts>1): online-softmax reduce of per-CTA partials. A CTA owns one (batch,head) and a
//   DPC=512/DimSplit slice of output dims. Sum association is fixed, hence deterministic.
template <int DimSplit>
__global__ void __launch_bounds__(256) mla_decode_a16w8_merge(
    const float* __restrict__ partials, bf16* __restrict__ o_lat, int B, int H, int parts,
    int q_len) {
  constexpr int NT  = 256, NW = NT / WaveSize;
  constexpr int DPC = LatDim / DimSplit;
  constexpr int DPL = 8;
  constexpr int LPR = DPC / DPL;
  constexpr int SUB = WaveSize / LPR;
  constexpr int GRP = NW * SUB;
  constexpr int HPT = (MaxParts + NT - 1) / NT;
  constexpr int U   = 4;
  static_assert(LPR >= 1 && LPR * DPL == DPC && SUB * LPR == WaveSize);

  const int bh = blockIdx.x / DimSplit, dblk = blockIdx.x - bh * DimSplit;
  const int bq = bh / H, h = bh - bq * H;
  const int qpos = q_len > 1 ? bq % q_len : 0;
  const int b    = q_len > 1 ? bq / q_len : bq;
  const int tid = threadIdx.x, lane = tid & (WaveSize - 1), wv = tid / WaveSize;
  // The kernel laid the q_len window out FASTEST on the CTA axis, so this
  //   (b, qpos)'s parts are q_len CTAs apart, not adjacent.
  const size_t cta0 = ((size_t)b * parts) * q_len + qpos;
  const size_t cstr = (size_t)q_len;
  __shared__ float s_w[MaxParts];           // per-part softmax weight exp2(max_p - max)
  __shared__ float s_rm[NW], s_rs[NW];      // cross-wave folds of run_max / run_sum
  __shared__ float s_o[(NW - 1) * DPC];     // cross-wave fold of the accumulated O slice

  // ── Pass 1: reduce the (run_max, run_sum) header planes into a global max and per-part weights ──
  const float* __restrict__ hmax = partials + hdr_off(h, 0) + cta0;
  const float* __restrict__ hsum = partials + hdr_off(h, 1) + cta0;
  float lm[HPT], ls[HPT], lmax = NegInf;
  #pragma unroll
  for (int k = 0; k < HPT; ++k) {           // each thread loads HPT header entries
    const int pp = tid + k * NT;
    lm[k] = (pp < parts) ? hmax[(size_t)pp * cstr] : NegInf;
    ls[k] = (pp < parts) ? hsum[(size_t)pp * cstr] : 0.f;
    lmax  = fmaxf(lmax, lm[k]);
  }
  // Reduce max across the wave, then across waves via LDS.
  #pragma unroll
  for (int m = 1; m < WaveSize; m <<= 1) lmax = fmaxf(lmax, lane_xor(lmax, m));
  if (lane == 0) s_rm[wv] = lmax;
  __syncthreads();
  float mmax = s_rm[0];
  #pragma unroll
  for (int w = 1; w < NW; ++w) mmax = fmaxf(mmax, s_rm[w]);
  const bool dead = (mmax == NegInf);        // every part empty (over-split / S==0) -> emit zeros
  // Compute per-part weight w_p = exp2(max_p - mmax), and the weighted sum of run_sum.
  float lsum = 0.f;
  #pragma unroll
  for (int k = 0; k < HPT; ++k) {
    const int pp = tid + k * NT;
    const float w = dead ? 0.f : __builtin_amdgcn_exp2f(lm[k] - mmax);
    if (pp < parts) s_w[pp] = w;
    lsum += w * ls[k];
  }
  // Reduce the weighted sum across the wave, then across waves.
  #pragma unroll
  for (int m = 1; m < WaveSize; m <<= 1) lsum += lane_xor(lsum, m);
  if (lane == 0) s_rs[wv] = lsum;
  __syncthreads();
  float msum = s_rs[0];
  #pragma unroll
  for (int w = 1; w < NW; ++w) msum += s_rs[w];
  const float inv = msum > 0.f ? 1.f / msum : 0.f;

  // ── Pass 2: accumulate sum_p w_p * O_p over this CTA's output-dim slice ──
  // Each thread owns DPL=8 dims of one partial row; waves stride over parts by GRP.
  const int sub = lane / LPR, dl = lane - sub * LPR;
  const size_t rowStride = (size_t)MaxHeads * LatDim * cstr;
  const bf16* __restrict__ src = (const bf16*)partials + (cta0 * MaxHeads + h) * LatDim
                                 + dblk * DPC + dl * DPL;
  float acc[DPL] = {};
  int pp = wv * SUB + sub;
  // Unrolled inner loop: U partials in flight per group iteration.
  for (; pp + (U - 1) * GRP < parts; pp += U * GRP) {
    bf16x8 v[U]; float w[U];
    #pragma unroll
    for (int u = 0; u < U; ++u) {
      v[u] = *(const bf16x8*)(src + (size_t)(pp + u * GRP) * rowStride);
      w[u] = s_w[pp + u * GRP];
    }
    #pragma unroll
    for (int u = 0; u < U; ++u)
      #pragma unroll
      for (int d = 0; d < DPL; ++d) acc[d] = __builtin_fmaf(w[u], (float)v[u][d], acc[d]);
  }
  // Remainder loop for the tail that doesn't fill U.
  for (; pp < parts; pp += GRP) {
    const bf16x8 v = *(const bf16x8*)(src + (size_t)pp * rowStride);
    const float w = s_w[pp];
    #pragma unroll
    for (int d = 0; d < DPL; ++d) acc[d] = __builtin_fmaf(w, (float)v[d], acc[d]);
  }
  // Fold the SUB row-subgroups inside the wave (no-op at DimSplit==1), then the NW waves through LDS.
  #pragma unroll
  for (int m = LPR; m < WaveSize; m <<= 1)
    #pragma unroll
    for (int d = 0; d < DPL; ++d) acc[d] += lane_xor(acc[d], m);
  if (wv != 0 && lane < LPR) {
    float* d = &s_o[(wv - 1) * DPC + lane * DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) d[i] = acc[i];
  }
  __syncthreads();
  // Wave 0 reduces the other waves' contributions and writes the normalized output.
  if (wv == 0 && lane < LPR) {
    #pragma unroll
    for (int w = 0; w < NW - 1; ++w) {
      const float* s = &s_o[w * DPC + lane * DPL];
      #pragma unroll
      for (int i = 0; i < DPL; ++i) acc[i] += s[i];
    }
    const bf16x4 p0 = pack4(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
    const bf16x4 p1 = pack4(acc[4] * inv, acc[5] * inv, acc[6] * inv, acc[7] * inv);
    bf16x8 out; __builtin_memcpy(&out, &p0, 8); __builtin_memcpy((char*)&out + 8, &p1, 8);
    *(bf16x8*)(o_lat + (size_t)((b * q_len + qpos) * H + h) * LatDim + dblk * DPC + lane * DPL) = out;
  }
}
// Dim-split pick: raise DimSplit until the merge grid is ~2 CTAs per CU. Capped at 16 (DPC=32 = one cache line).
__host__ inline int merge_dsplit(int B, int H) {
  int bh = B * H; if (bh < 1) bh = 1;
  int ds = 1;
  while (ds < 16 && bh * ds * 2 < 4 * NumCU) ds <<= 1;
  return ds;
}
__host__ inline void launch_merge(const float* partials, bf16* o_lat, int B, int H, int parts,
                                  hipStream_t stream, int q_len = 1) {
  const int ds = merge_dsplit(B * q_len, H);
  #define A16W8_MERGE_CASE(D)                                                                       \
    case D: mla_decode_a16w8_merge<D><<<dim3(B * q_len * H * (D)), dim3(256), 0, stream>>>(    \
                partials, o_lat, B, H, parts, q_len); break;
  switch (ds) {
    A16W8_MERGE_CASE(2) A16W8_MERGE_CASE(4) A16W8_MERGE_CASE(8) A16W8_MERGE_CASE(16)
    default: A16W8_MERGE_CASE(1)
  }
  #undef A16W8_MERGE_CASE
}
constexpr size_t FusedDimHost = 576;                 // one fused KV row: 512 latent + 64 rope
constexpr size_t MallBytes    = 256ull << 20;        // MI300X MALL capacity
// NT gate: bypass L2/MALL only when the KV working set exceeds the MALL. Below that, the pool is reused
//   across decode steps and bypassing discards real hits.
__host__ inline bool want_nt(size_t kv_bytes) { return kv_bytes > MallBytes; }
// Split planner: KV-splits (`parts`) per request. Occupancy is 1, so fill the machine once (parts = NumCU/B),
//   floored by MinTilesPerPart (amortise the prologues) and the workspace CTA cap.
__host__ inline int plan_parts(int B, int S, int qgroups = 1) {
  int tiles = (S + StepTok - 1) / StepTok;
  if (tiles < 1) tiles = 1;
  if (qgroups < 1) qgroups = 1;
  // At QLC=1 the window costs qgroups=q_len CTAs per (batch, kv-part), so the
  //   KV split has that many fewer parts to spend before hitting the grid cap.
  const int bb = (B > 0 ? B : 1) * qgroups;
  int p = NumCU / bb;
  if (p < 1) p = 1;
  constexpr int MinTilesPerPart = 1;
  if (p > tiles / MinTilesPerPart) p = tiles / MinTilesPerPart;
  if (p > tiles)     p = tiles;
  if (p > MaxParts)  p = MaxParts;
  const int cap = (B > 0) ? (int)(MaxCTAs / ((size_t)B * (size_t)qgroups)) : 1;
  if (p > cap) p = cap;
  if (p < 1)   p = 1;
  return p;
}

}  // namespace

// Contiguous launcher: c_kv is a fused fp8 [B,S,576] slab. `k_pe`/`scale_p` are unused (ABI only).
extern "C" __host__ int launch_mla_decode_a16w8(
    const void* q_lat, const void* q_pe, const void* c_kv, const void* k_pe, void* o_lat,
    int B, int H, int S, int lat_dim, int rope_dim, float scale, float kv_scale, float scale_p,
    void* workspace, void* stream_) {
  if (lat_dim != 512 || rope_dim != 64) return -2;
  if (H > 16) return -3;
  const auto stream = (hipStream_t)stream_;
  const float scale_log2e = 1.4426950408889634f * scale;
  int parts = workspace ? plan_parts(B, S) : 1;
  if (parts > 1 && (size_t)B * parts > MaxCTAs) parts = (int)(MaxCTAs / (size_t)B);
  if (parts > MaxParts) parts = MaxParts;
  if (parts < 1) parts = 1;
  const bool nt = want_nt((size_t)B * (size_t)S * FusedDimHost);
  if (nt)
    mla_decode_a16w8_kernel</*Paged=*/false, /*NT=*/true><<<dim3(B * parts), dim3(Threads), 0, stream>>>(
        (const bf16*)q_lat, (const bf16*)q_pe, (const fp8_t*)c_kv, (const bf16*)k_pe,
        (bf16*)o_lat, (float*)workspace, B, H, S, parts, scale_log2e, kv_scale, scale_p,
        /*q_len=*/1);
  else
    mla_decode_a16w8_kernel</*Paged=*/false, /*NT=*/false><<<dim3(B * parts), dim3(Threads), 0, stream>>>(
        (const bf16*)q_lat, (const bf16*)q_pe, (const fp8_t*)c_kv, (const bf16*)k_pe,
        (bf16*)o_lat, (float*)workspace, B, H, S, parts, scale_log2e, kv_scale, scale_p,
        /*q_len=*/1);
  if (parts > 1) launch_merge((const float*)workspace, (bf16*)o_lat, B, H, parts, stream);
  return (int)hipGetLastError();
}

// Paged launcher: graph-safe, device-driven. kv_pool is a fused fp8 [num_slots,1,576] pool; seq_lens is on
//   device, so `parts` is fixed at capture and the NT gate takes kv_bytes as a parameter.
extern "C" __host__ int launch_mla_decode_a16w8_paged_dev(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat,
    const void* seq_lens, const void* rows, const void* kv_indices, const void* kv_indptr,
    int B, int H, int parts, int lat_dim, int rope_dim, float scale, float kv_scale, float scale_p,
    void* workspace, void* stream_, size_t kv_bytes, int q_len) {
  if (q_len < 1 || q_len > MaxQL) return -4;
  if (lat_dim != 512 || rope_dim != 64) return -2;
  if (H > 16) return -3;
  const auto stream = (hipStream_t)stream_;
  const float scale_log2e = 1.4426950408889634f * scale;
  int p = parts < 1 ? 1 : parts;
  {
    const int grid_cap = (int)(MaxCTAs / ((size_t)B * (size_t)q_len));
    const int max_p = grid_cap < MaxParts ? grid_cap : MaxParts;
    if (p > max_p) p = max_p;
    if (p < 1) p = 1;
  }
  if (p > 1 && !workspace) p = 1;
  const bool nt = want_nt(kv_bytes);
  if (nt)
    mla_decode_a16w8_kernel</*Paged=*/true, /*NT=*/true><<<dim3(B * p * q_len), dim3(Threads), 0, stream>>>(
        (const bf16*)q_lat, (const bf16*)q_pe, (const fp8_t*)kv_pool, /*k_pe=*/nullptr,
        (bf16*)o_lat, (float*)workspace, B, H, /*S_scalar=*/0, p, scale_log2e, kv_scale, scale_p,
        q_len, (const int*)seq_lens, (const int*)rows, (const int*)kv_indices, (const int*)kv_indptr);
  else
    mla_decode_a16w8_kernel</*Paged=*/true, /*NT=*/false><<<dim3(B * p * q_len), dim3(Threads), 0, stream>>>(
        (const bf16*)q_lat, (const bf16*)q_pe, (const fp8_t*)kv_pool, /*k_pe=*/nullptr,
        (bf16*)o_lat, (float*)workspace, B, H, /*S_scalar=*/0, p, scale_log2e, kv_scale, scale_p,
        q_len, (const int*)seq_lens, (const int*)rows, (const int*)kv_indices, (const int*)kv_indptr);
  if (p > 1) launch_merge((const float*)workspace, (bf16*)o_lat, B, H, p, stream, q_len);
  return (int)hipGetLastError();
}

extern "C" __host__ int mla_decode_a16w8_plan_parts(int B, int H, int max_seq_len, int lat_dim) {
  (void)H; (void)lat_dim;
  int parts = plan_parts(B, max_seq_len);
  if (parts < 1) parts = 1;
  return parts;
}

extern "C" __host__ int mla_decode_a16w8_max_ctas() { return (int)MaxCTAs; }

// q_len-aware fixed KV-split count for a graph captured at (B, max_seq_len, q_len).
//   Prefer this over plan_parts_capped when q_len > 1: at QLC=1 every draft position
//   is its own CTA, so the window multiplies the CTA count and the KV split has to
//   give parts back to stay inside the grid cap. H is accepted for ABI symmetry with
//   the entry point and ignored (H <= 16 is a single query tile).
extern "C" __host__ int mla_decode_a16w8_plan_parts_q(int B, int max_seq_len, int q_len, int H) {
  (void)H;
  if (q_len < 1) q_len = 1;
  if (q_len > MaxQL) return -4;
  int parts = plan_parts(B, max_seq_len, q_len);
  if (parts < 1) parts = 1;
  return parts;
}

extern "C" __host__ int mla_decode_a16w8_plan_parts_capped(int B, int H, int max_seq_len, int lat_dim) {
  (void)H; (void)lat_dim;
  int parts = plan_parts(B, max_seq_len);
  const int max_parts = (int)(MaxCTAs / (size_t)B);
  if (parts > max_parts) parts = max_parts;
  if (parts < 1) parts = 1;
  return parts;
}
