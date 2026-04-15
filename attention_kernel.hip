// Flash-attention forward — CDNA3 gfx942, single-wave, 16x16x16 MFMA, bf16.
//
// MFMA output mapping:
//   Thread t, element e stores D[(t/16)*4+e, t%16]
//     blk  = t/16  (0-3)   → selects which 4 output ROWS
//     lane = t%16  (0-15)  → output COLUMN
//
// QK trick: call mfma16(K, Q, acc) so D = K × Q^T = QK^T.
//   Then D[blk*4+e, lane] = QK^T[blk*4+e, lane] = QK[lane, blk*4+e].
//   Each thread holds 4 COLUMNS of the same QK ROW (lane).
//   Softmax reduces across blk groups (XOR 16,32)
//   No probability transpose needed.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

using bf16_t     = __hip_bfloat16;
using bf16x4_t   = short  __attribute__((ext_vector_type(4)));
using buffer_rsrc_t = int __attribute__((ext_vector_type(4)));
using acc4_t     = float  __attribute__((ext_vector_type(4)));

static constexpr int HD       = 128;
static constexpr int QBLOCK   = 64;
static constexpr int KVTILE   = 16;
static constexpr int WAVE     = 64;
static constexpr int QGROUPS  = QBLOCK / 16;   // 4
static constexpr int HDGROUPS = HD / 16;        // 8
static constexpr int KSTEPS   = HD / 16;        // 8

static constexpr int LDS_ROW   = 130;
static constexpr int LDS_ROW_B = LDS_ROW * 2;  // 260 bytes

__device__ __forceinline__ buffer_rsrc_t make_rsrc(const void* p) {
  auto r = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<void*>(p), short(0), 0xFFFFFFFFu, 0x20000u);
  buffer_rsrc_t o;
  __builtin_memcpy(&o, &r, sizeof(o));
  return o;
}

__device__ __forceinline__ void bcast_rsrc(buffer_rsrc_t& r) {
  r[0] = __builtin_amdgcn_readfirstlane(r[0]);
  r[1] = __builtin_amdgcn_readfirstlane(r[1]);
  r[2] = __builtin_amdgcn_readfirstlane(r[2]);
  r[3] = __builtin_amdgcn_readfirstlane(r[3]);
}

extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(
    buffer_rsrc_t rsrc,
    __attribute__((address_space(3))) unsigned* lds,
    unsigned size, unsigned voff, unsigned soff,
    unsigned offset, unsigned aux)
  __asm("llvm.amdgcn.raw.buffer.load.lds");

template <int vm, int lgkm>
__device__ __forceinline__ void wait_cnt() {
  constexpr unsigned enc =
      ((vm & 0x0F))              |
      (((vm & 0x30) >> 4) << 14) |
      (7u << 4)                  |
      ((lgkm & 0x3F) << 8);
  __builtin_amdgcn_s_waitcnt(enc);
}
#define WAIT_VM(n)   wait_cnt<n, 63>()

__device__ __forceinline__ acc4_t
mfma16(bf16x4_t a, bf16x4_t b, acc4_t c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ unsigned pk_bf16(unsigned lo, unsigned hi) {
  unsigned o;
  asm volatile("v_perm_b32 %0, %1, %2, %3"
               : "=v"(o) : "v"(lo), "v"(hi), "s"(0x07060302u));
  return o;
}

__device__ __forceinline__ bf16x4_t
pack4(float a, float b, float c, float d) {
  unsigned w0 = pk_bf16(__builtin_bit_cast(unsigned, b),
                        __builtin_bit_cast(unsigned, a));
  unsigned w1 = pk_bf16(__builtin_bit_cast(unsigned, d),
                        __builtin_bit_cast(unsigned, c));
  bf16x4_t o;
  __builtin_memcpy(&o, &w0, 4);
  __builtin_memcpy((char*)&o + 4, &w1, 4);
  return o;
}

// Reduce across blk groups (XOR 16, 32) — 4 threads per QK row.
__device__ __forceinline__ float bperm_xor(int xv, float v) {
  int tid = __builtin_amdgcn_mbcnt_hi(-1u, __builtin_amdgcn_mbcnt_lo(-1u, 0));
  int b   = __builtin_bit_cast(int, v);
  int r   = __builtin_amdgcn_ds_bpermute(((tid ^ xv) << 2), b);
  return __builtin_bit_cast(float, r);
}

__device__ __forceinline__ float row_max(float v) {
  v = fmaxf(v, bperm_xor(16, v));
  v = fmaxf(v, bperm_xor(32, v));
  return v;
}
__device__ __forceinline__ float row_sum(float v) {
  v += bperm_xor(16, v);
  v += bperm_xor(32, v);
  return v;
}

__device__ __forceinline__ void
tile_to_lds(buffer_rsrc_t rsrc, bf16_t* lds, int tile) {
  const unsigned base = unsigned(tile) * (KVTILE * HD * 2u);
  #pragma unroll
  for (int r = 0; r < KVTILE; ++r) {
    unsigned voff = base + unsigned(r * 256 + threadIdx.x * 4);
    unsigned loff = unsigned(r * LDS_ROW_B);
    auto* lp = (__attribute__((address_space(3))) unsigned*)(
        (__attribute__((address_space(3))) char*)lds + loff);
    llvm_amdgcn_raw_buffer_load_lds(rsrc, lp, 4, voff, 0, 0, 0);
  }
}

// =====================================================================

extern "C" __global__ __launch_bounds__(64, 1)
void attn_fwd_16x16(const bf16_t* __restrict__ q,
                    const bf16_t* __restrict__ k,
                    const bf16_t* __restrict__ v,
                    float* __restrict__ out,
                    int seq_len,
                    float scale) {

  const int tid  = threadIdx.x;
  const int lane = tid & 15;    // MFMA N-index = QK row = output column
  const int blk  = tid >> 4;    // MFMA M-group; M-index = blk*4+e = KV col / output row-group

  const int hstride = seq_len * HD;
  const bf16_t* qh = q + blockIdx.y * hstride;
  const bf16_t* kh = k + blockIdx.y * hstride;
  const bf16_t* vh = v + blockIdx.y * hstride;
  float*        oh = out + blockIdx.y * hstride;
  const int qbase  = blockIdx.x * QBLOCK;

  buffer_rsrc_t kr = make_rsrc(kh);  bcast_rsrc(kr);
  buffer_rsrc_t vr = make_rsrc(vh);  bcast_rsrc(vr);

  __shared__ __align__(16) bf16_t lds_k[2][KVTILE * LDS_ROW];
  __shared__ __align__(16) bf16_t lds_v[KVTILE * LDS_ROW];

  // Q: B-operand of QK MFMA.  B[lane, blk*4+0..3] = Q[lane_row, k_cols].
  bf16x4_t Q[QGROUPS][KSTEPS];
  #pragma unroll
  for (int g = 0; g < QGROUPS; ++g) {
    int qrow = qbase + g * 16 + lane;
    #pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      int kcol = s * 16 + blk * 4;
      long long w = *reinterpret_cast<const long long*>(
          &qh[qrow * HD + kcol]);
      __builtin_memcpy(&Q[g][s], &w, sizeof(bf16x4_t));
    }
  }

  // O[g][h][e] accumulates PV[(blk*4+e), h*16+lane]
  acc4_t O[QGROUPS][HDGROUPS] = {};

  // mx/sm: softmax state for Q-row `lane` (used for prob computation).
  // mx_o[g][e]: running max for O-element's row (blk*4+e), gathered from lane=blk*4+e.
  float mx[QGROUPS], sm[QGROUPS];
  float mx_o[QGROUPS][4];
  #pragma unroll
  for (int g = 0; g < QGROUPS; ++g) {
    mx[g] = -3e38f; sm[g] = 0.f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) mx_o[g][e] = -3e38f;
  }

  const int ntiles = seq_len / KVTILE;
  tile_to_lds(kr, lds_k[0], 0);

  for (int t = 0; t < ntiles; ++t) {
    const int cb = t & 1, nb = 1 - cb;

    WAIT_VM(0);
    // BARRIER_HERE for multi-wave

    bf16x4_t K[KSTEPS];
    #pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      long long w = *reinterpret_cast<const long long*>(
          &lds_k[cb][lane * LDS_ROW + s * 16 + blk * 4]);
      __builtin_memcpy(&K[s], &w, sizeof(bf16x4_t));
    }

    // Issue V loads FIRST so they are oldest in vmcnt queue.
    // This lets vmcnt(16) drain V while K prefetch stays in flight.
    tile_to_lds(vr, lds_v, t);

    if (t + 1 < ntiles)
      tile_to_lds(kr, lds_k[nb], t + 1);

    acc4_t qk[QGROUPS];
    #pragma unroll
    for (int g = 0; g < QGROUPS; ++g) {
      qk[g] = {};
      #pragma unroll
      for (int s = 0; s < KSTEPS; ++s)
        qk[g] = mfma16(K[s], Q[g][s], qk[g]);
    }

    // Online softmax. qk[g][0..3] = QK[lane, blk*4+0..3].
    bf16x4_t P[QGROUPS];
    #pragma unroll
    for (int g = 0; g < QGROUPS; ++g) {
      float lm = fmaxf(fmaxf(qk[g][0], qk[g][1]),
                        fmaxf(qk[g][2], qk[g][3]));
      float tm = row_max(lm);
      float nm = fmaxf(mx[g], tm);

      // Gather nm for each O-row (blk*4+e) and rescale O accumulators.
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        int b = __builtin_bit_cast(int, nm);
        float nm_e = __builtin_bit_cast(float,
            __builtin_amdgcn_ds_bpermute((blk * 4 + e) * 4, b));
        float c = __builtin_amdgcn_exp2f((mx_o[g][e] - nm_e) * scale);
        #pragma unroll
        for (int h = 0; h < HDGROUPS; ++h)
          O[g][h][e] *= c;
        mx_o[g][e] = nm_e;
      }

      float corr = __builtin_amdgcn_exp2f((mx[g] - nm) * scale);
      float nms = -nm * scale;
      float p[4], ps = 0.f;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        p[i] = __builtin_amdgcn_exp2f(
            __builtin_fmaf(qk[g][i], scale, nms));
        ps += p[i];
      }
      sm[g] = sm[g] * corr + row_sum(ps);
      mx[g] = nm;
      P[g]  = pack4(p[0], p[1], p[2], p[3]);
    }

    // V loads (oldest 16) are done; K prefetch (newest 16) stays in flight.
    if (t + 1 < ntiles)
      WAIT_VM(16);
    else
      WAIT_VM(0);
    // BARRIER_HERE for multi-wave

    #pragma unroll
    for (int h = 0; h < HDGROUPS; ++h) {
      short vv[4];
      #pragma unroll
      for (int e = 0; e < 4; ++e)
        vv[e] = *reinterpret_cast<const short*>(
            &lds_v[(blk * 4 + e) * LDS_ROW + h * 16 + lane]);
      bf16x4_t Vr;
      __builtin_memcpy(&Vr, vv, sizeof(bf16x4_t));

      #pragma unroll
      for (int g = 0; g < QGROUPS; ++g)
        O[g][h] = mfma16(P[g], Vr, O[g][h]);
    }
  }

  // Epilogue: gather sm for each O-row and normalize.
  #pragma unroll
  for (int g = 0; g < QGROUPS; ++g) {
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      int b = __builtin_bit_cast(int, sm[g]);
      float sm_e = __builtin_bit_cast(float,
          __builtin_amdgcn_ds_bpermute((blk * 4 + e) * 4, b));
      float inv = __builtin_amdgcn_rcpf(sm_e);
      int qrow = qbase + g * 16 + blk * 4 + e;
      #pragma unroll
      for (int h = 0; h < HDGROUPS; ++h)
        oh[qrow * HD + h * 16 + lane] = O[g][h][e] * inv;
    }
  }
}

extern "C" int launch_attention_forward(const void* q, const void* k,
                                        const void* v, void* out,
                                        int batch, int heads,
                                        int seq_len, int head_dim,
                                        void* stream) {
  if (head_dim != HD)             return 1;
  if ((seq_len % KVTILE) != 0)   return 2;
  if ((seq_len % QBLOCK) != 0)   return 3;

  dim3 grid(seq_len / QBLOCK, batch * heads, 1);
  dim3 block(WAVE, 1, 1);
  float scale = (1.f / sqrtf(float(head_dim))) * 1.44269504088896f;

  hipLaunchKernelGGL(attn_fwd_16x16,
                     grid, block, 0, (hipStream_t)stream,
                     (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
                     (float*)out, seq_len, scale);
  return 0;
}
