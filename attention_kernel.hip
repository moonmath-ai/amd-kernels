// =====================================================================================
//  CDNA3 / MI300 (gfx942) — K LDS double-buffering skeleton.
//
//  Minimal kernel used to verify the prefetch pipeline in isolation.
//  Each iteration body:
//      1. issue prefetch for NEXT 2 K-tiles into LDS (direct global→LDS)
//      2. s_waitcnt vmcnt(16)  — drain OLD prefetch only; NEW stays in flight
//      3. s_barrier + ds_read  — consume OLD tiles into VGPRs
//      4. 32 × mfma            — NEW VMEM lands during these MFMAs
//
//  Softmax / V / PV are intentionally absent
// =====================================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

using bf16_t = __hip_bfloat16;
using bf16x4 = short __attribute__((ext_vector_type(4)));
using f32x16 = float __attribute__((ext_vector_type(16)));
using i32x4  = int   __attribute__((ext_vector_type(4)));

static constexpr int HD           = 128;
static constexpr int BM           = 128;     // Q rows / block
static constexpr int BN           = 32;      // K rows / tile
static constexpr int TH           = 256;     // 4 waves × 64
static constexpr int K_ROW_STRIDE = HD + 8;  // pad for LDS banks
static constexpr int K_LDS_BF16   = BN * K_ROW_STRIDE;

// -------------------------------------------------------------------------------------
//  Intrinsics
// -------------------------------------------------------------------------------------
__device__ __forceinline__ f32x16 mfma(bf16x4 a, bf16x4 b, f32x16 c) {
  return __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ i32x4 make_buffer_rsrc(const void* p) {
  auto r = __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short)0,
                                             0xFFFFFFFFu, 0x00020000u);
  i32x4 out; __builtin_memcpy(&out, &r, sizeof(out)); return out;
}

__device__ __forceinline__ i32x4 to_sgpr(i32x4 v) {
  v[0] = __builtin_amdgcn_readfirstlane(v[0]);
  v[1] = __builtin_amdgcn_readfirstlane(v[1]);
  v[2] = __builtin_amdgcn_readfirstlane(v[2]);
  v[3] = __builtin_amdgcn_readfirstlane(v[3]);
  return v;
}

extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(i32x4 rsrc,
                                __attribute__((address_space(3))) unsigned* lds,
                                unsigned size, unsigned voff, unsigned soff,
                                unsigned off, unsigned aux)
  __asm("llvm.amdgcn.raw.buffer.load.lds");

// s_waitcnt with independent VMCNT / LGKMCNT; 63 means "don't care".
template <int vm, int lgkm>
__device__ __forceinline__ void wait_cnt() {
  constexpr unsigned enc = (vm & 0x0F) | (((vm & 0x30) >> 4) << 14)
                         | (7u << 4) | ((lgkm & 0x3F) << 8);
  __builtin_amdgcn_s_waitcnt(enc);
}
template <int N> __device__ __forceinline__ void wait_vm()   { wait_cnt<N, 63>(); }
template <int N> __device__ __forceinline__ void wait_lgkm() { wait_cnt<63, N>(); }

// =====================================================================================
//  Kernel
// =====================================================================================
extern "C" __global__ __launch_bounds__(TH, 1)
void attn_fwd_32x32(const bf16_t* __restrict__ Q,
                    const bf16_t* __restrict__ K,
                    const bf16_t* __restrict__ /*V*/,
                    float*        __restrict__ O,
                    int /*B*/, int /*H*/, int S, int /*D*/,
                    float /*scale*/)
{
  const int tid       = threadIdx.x;
  const int lane      = tid & 63;
  const int wave      = tid >> 6;
  const int lane_row  = lane & 31;
  const int lane_half = lane >> 5;

  const int bh = blockIdx.y;
  const int qb = blockIdx.x;

  const int head_stride = S * HD;
  const bf16_t* Qbh = Q + bh * head_stride;
  const bf16_t* Kbh = K + bh * head_stride;
  float*        Obh = O + bh * head_stride;

  __shared__ __align__(16) bf16_t Klds[2][K_LDS_BF16];

  const i32x4 rsrc_K = to_sgpr(make_buffer_rsrc(Kbh));

  // Prefetch one K-tile into LDS buffer `buf` via direct global→LDS.
  //   4 waves × 8 rows = BN=32 rows per tile. Each lane contributes 1 dword
  //   (2 bf16) per kv_row, so the row is covered by 64 lanes × 4 B = 256 B = HD*2.
  const unsigned lane_hd = (unsigned)lane * 2u;
  const int      wave_s  = __builtin_amdgcn_readfirstlane(wave);
  auto launch_K = [&, lane_hd, wave_s](int kt, int buf) {
    const unsigned kt_base = (unsigned)(kt * BN * HD * 2);
    auto* lds_u = (__attribute__((address_space(3))) unsigned*)&Klds[buf][0];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      unsigned kv_row = (unsigned)(wave_s * 8 + c);
      unsigned voff   = kt_base + kv_row * (unsigned)(HD * 2) + lane_hd * 2u;
      llvm_amdgcn_raw_buffer_load_lds(rsrc_K,
          lds_u + kv_row * (unsigned)(K_ROW_STRIDE / 2),
          4u, voff, 0u, 0u, 0u);
    }
  };

  // 8 × ds_read_b128 per lane → 16 bf16x4 covering one K tile.
  auto load_K_from_lds = [&](int buf, bf16x4 Kreg[16]) {
    using i4 = int __attribute__((ext_vector_type(4)));
    auto* lds_b  = (__attribute__((address_space(3))) char*)&Klds[buf][0];
    auto* lds_i4 = (__attribute__((address_space(3))) i4*)(
        lds_b + (unsigned)(lane_row * K_ROW_STRIDE * 2 + lane_half * 128));
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      i4 v = lds_i4[p];
      __builtin_memcpy(&Kreg[2*p    ], (const char*)&v,     8);
      __builtin_memcpy(&Kreg[2*p + 1], (const char*)&v + 8, 8);
    }
  };

  // Q stays in VGPRs — it's the B operand of the swapped MFMA (see comment below).
  bf16x4 Qreg[16];
  {
    const int qrow = qb * BM + wave * 32 + lane_row;
    #pragma unroll
    for (int mk = 0; mk < 16; ++mk) {
      int hd = mk * 8 + lane_half * 4;
      long long v = *reinterpret_cast<const long long*>(&Qbh[qrow * HD + hd]);
      __builtin_memcpy(&Qreg[mk], &v, 8);
    }
  }

  f32x16 Sacc0 = {}, Sacc1 = {};
  const int num_tiles = S / BN;

  // --- Prime the pipe: 16 VMEM ops in flight after this (8 per launch_K). ---
  launch_K(0, 0);
  launch_K(1, 1);

  // --- Main loop — double buffered K prefetch ---
  //
  //  VMCNT evolution per iter (per wave):
  //      enter           : 16 in flight  (prev iter's prefetch)
  //      launch 16 new   : 32 in flight
  //      wait_vm<16>     : drain 16 old, keep 16 new traveling
  //      ds_read + MFMAs : new VMEMs land during the MFMA window
  //
  //  Aliasing safety: ds_read latency (~20 cyc) ≪ buffer_load_lds latency
  //  (~400 cyc), so the new VMEM write never lands on buf[k] before the
  //  ds_read of the old data has drained. Within one buffer, VMEM writes
  //  are FIFO so old-then-new is ordered correctly.
  for (int kt = 2; kt < num_tiles; kt += 2) {
    launch_K(kt,     0);
    launch_K(kt + 1, 1);
    wait_vm<16>();
    __builtin_amdgcn_s_barrier();

    bf16x4 Kreg0[16], Kreg1[16];
    load_K_from_lds(0, Kreg0);
    load_K_from_lds(1, Kreg1);
    wait_lgkm<0>();

    __builtin_amdgcn_iglp_opt(1);
    #pragma unroll
    for (int mk = 0; mk < 16; ++mk) {
      Sacc0 = mfma(Kreg0[mk], Qreg[mk], Sacc0);
      Sacc1 = mfma(Kreg1[mk], Qreg[mk], Sacc1);
    }
  }

  // --- Drain last prefetched pair ---
  {
    wait_vm<0>();
    __builtin_amdgcn_s_barrier();
    bf16x4 Kreg0[16], Kreg1[16];
    load_K_from_lds(0, Kreg0);
    load_K_from_lds(1, Kreg1);
    wait_lgkm<0>();
    #pragma unroll
    for (int mk = 0; mk < 16; ++mk) {
      Sacc0 = mfma(Kreg0[mk], Qreg[mk], Sacc0);
      Sacc1 = mfma(Kreg1[mk], Qreg[mk], Sacc1);
    }
  }

  // Keep MFMAs live for the compiler.
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc += Sacc0[i] + Sacc1[i];
  Obh[qb * BM * HD + tid] = acc;
}

// =====================================================================================
//  Host entry
// =====================================================================================
extern "C" int launch_attention_forward(const void* q, const void* k, const void* v,
                                        void* out, int batch, int heads,
                                        int seq_len, int head_dim, void* stream)
{
  if (head_dim != HD)          return 1;
  if (seq_len % BM)            return 2;
  if ((seq_len / BN) % 2)      return 3;   // main loop processes tile pairs

  dim3 grid(seq_len / BM, batch * heads, 1);
  dim3 block(TH, 1, 1);
  hipLaunchKernelGGL(attn_fwd_32x32, grid, block, 0, (hipStream_t)stream,
                     (const bf16_t*)q, (const bf16_t*)k, (const bf16_t*)v,
                     (float*)out, batch, heads, seq_len, head_dim, 0.f);
  return 0;
}
