// DeepSeek-V4 FP4 indexer logits for CDNA3 (gfx942 / MI300X), which has no FP4 MFMA.
//
//   logits[r, j] = weight_scale * sum_h w[r, h] * relu(q[r, h, :] . k[page_table[r, j / 64], j % 64, :])
//
// for j < lens[r]; logits past a row's length are left unwritten. Q and K are E2M1 with one
// E8M0 scale per 32 dims, in aiter's preshuffle layouts (see the address comments below).
// E2M1 times a power of two is exact in bf16, so each value is rebuilt as bf16 bits with its
// scale added into the exponent -- integer ops only -- and contracted on the bf16 MFMA.
//
// A workgroup takes ROWS consecutive rows, one wave per row, and a run of PAGES key pages.
// A wave holds its row's H heads x 128 dims as MFMA A fragments for the whole run. Rows of one
// prefill request read the same pages: when every row of the workgroup maps page j to the same
// physical page, the waves convert it once, together, into LDS in B-fragment order; a page
// where they differ (a tile straddling two requests) is converted by each wave for itself.
//
// The dot product is invariant to the order of the 128 dims, so a lane's 4 dims of K-step
// ks = 2c + h are chosen as bytes [16c + 4*quad + 2h, +2) of chunk c: one 2-byte load per
// fragment, the same mapping for Q and K.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {
using bf16 = __bf16;
using bf16x4 = __attribute__((ext_vector_type(4))) bf16;
using u16x4 = __attribute__((ext_vector_type(4))) uint16_t;
using f32x4 = __attribute__((ext_vector_type(4))) float;

constexpr int WaveSize = 64;
constexpr int P = 64;                      // tokens per key page
constexpr int D = 128;                     // head dim
constexpr int Chunks = D / 32;             // E8M0 scale groups
constexpr int KSteps = D / 16;             // 16x16x16 MFMA K-steps
constexpr int NTiles = P / 16;             // 16-key column tiles per page
constexpr int Frags = NTiles * KSteps;     // B fragments per page (per lane)
constexpr int PageBytes = Chunks * P * 16; // [chunk][token][16 B]
constexpr int PageScales = Chunks * P;     // [chunk][token, nt-interleaved]

__device__ __forceinline__ f32x4 mfma_bf16(bf16x4 a, bf16x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// One E2M1 nibble times 2^(s - 127) as bf16 bits: |v| = 2^(e-1) * (1 + m/2) has exponent field
// e - 1 + s and mantissa m << 6; 0.5 (e = 0, m = 1) has field s - 1; +-0 is 0.
__device__ __forceinline__ uint16_t e2m1_bf16(uint32_t nib, int s) {
  const int mag = nib & 7, e = mag >> 1, m = mag & 1;
  const int field = e ? e - 1 + s : s - 1;
  const uint32_t bits = ((uint32_t)field << 7) | (e ? (uint32_t)m << 6 : 0u) | ((nib & 8u) << 12);
  return (mag == 0 || field <= 0) ? 0 : (uint16_t)bits;
}

// Two bytes = four nibbles in dim order (low nibble first), all under scale s.
__device__ __forceinline__ bf16x4 widen(uint32_t two_bytes, int s) {
  const u16x4 v = {e2m1_bf16(two_bytes, s), e2m1_bf16(two_bytes >> 4, s),
                   e2m1_bf16(two_bytes >> 8, s), e2m1_bf16(two_bytes >> 12, s)};
  return __builtin_bit_cast(bf16x4, v);
}

// B fragment `f` (column tile nt, K-step ks) of `page` for this lane: key nt*16 + col.
// k_payload: [pages][chunk][token][16 B]; k_scale: [pages][chunk][(t % 16) * 4 + t / 16].
__device__ __forceinline__ bf16x4 k_fragment(const uint8_t* kp, const uint8_t* ks, long page,
                                             int f, int col, int quad) {
  const int nt = f / KSteps, step = f % KSteps, c = step >> 1, h = step & 1;
  const int key = nt * 16 + col;
  const uint32_t bytes = *(const uint16_t*)(kp + page * PageBytes + c * (P * 16) + key * 16 + quad * 4 + h * 2);
  const int s = ks[page * PageScales + c * P + col * 4 + nt];
  return widen(bytes, s);
}

template <int ROWS, int PAGES, int MT>
__global__ __launch_bounds__(ROWS * WaveSize) void fp4_logits_kernel(
    const uint8_t* __restrict__ q,       // [T][H][64 B]
    const uint8_t* __restrict__ qs,      // [T][chunk][16][qs_pad]: head m*16 + i at (i, m)
    const uint8_t* __restrict__ kp, const uint8_t* __restrict__ kscale,
    const bf16* __restrict__ w,          // [T][H]
    const int* __restrict__ pt,          // [T][pt_stride]
    const int* __restrict__ lens,        // [T]
    float* __restrict__ out,             // [T][out_stride]
    int T, int pt_stride, int out_stride, int qs_pad, float weight_scale) {
  constexpr int H = MT * 16;
  __shared__ bf16x4 frag[Frags][WaveSize];                // one page (16 KB; two halve occupancy)
  __shared__ int s_page[ROWS][PAGES];                      // each row's physical page per slot

  const int lane = threadIdx.x % WaveSize, wave = threadIdx.x / WaveSize;
  const int col = lane & 15, quad = lane >> 4;
  const int r0 = blockIdx.x * ROWS, row = r0 + wave;
  const bool live = row < T;
  const int my_row = live ? row : r0;
  const int len = live ? lens[row] : 0;

  int tile_len = (lane < ROWS && r0 + lane < T) ? lens[r0 + lane] : 0;
  for (int o = 1; o < WaveSize; o <<= 1) tile_len = max(tile_len, __shfl_xor(tile_len, o));
  const int p_begin = blockIdx.y * PAGES;
  if (p_begin * P >= tile_len) return;                     // workgroup-uniform
  const int p_end = min(p_begin + PAGES, (tile_len + P - 1) / P);

  // A fragments: head mt*16 + col, the lane's 4 dims of every K-step.
  bf16x4 a[MT][KSteps];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int step = 0; step < KSteps; ++step) {
      const int c = step >> 1, h = step & 1, head = mt * 16 + col;
      const uint32_t bytes = *(const uint16_t*)(q + ((long)my_row * H + head) * (D / 2) + c * 16 + quad * 4 + h * 2);
      const int s = qs[(long)my_row * (Chunks * 16 * qs_pad) + c * (16 * qs_pad) + col * qs_pad + mt];
      a[mt][step] = widen(bytes, s);
    }
  // Head weights for this lane's accumulator rows: head mt*16 + 4*quad + e.
  float wt[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) wt[mt][e] = (float)w[(long)my_row * H + mt * 16 + 4 * quad + e];

  // Every row's pages for the whole run up front, so only the fragment buffer needs barriers.
  if (lane < PAGES)
    s_page[wave][lane] = (live && p_begin + lane < p_end) ? pt[(long)my_row * pt_stride + p_begin + lane] : -1;
  __syncthreads();

  for (int p = p_begin; p < p_end; ++p) {
    const int slot = p - p_begin;
    const int my_page = s_page[wave][slot];
    int page0 = -1;
    bool uniform = true;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      const int v = s_page[i][slot];
      if (v < 0) continue;
      if (page0 < 0) page0 = v;
      else uniform &= (v == page0);
    }
    const bool active = live && p * P < len;               // wave-uniform

    f32x4 acc[MT][NTiles];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NTiles; ++nt) acc[mt][nt] = f32x4{0.f, 0.f, 0.f, 0.f};

    if (uniform) {                                         // workgroup-uniform
      for (int f = wave; f < Frags; f += ROWS)
        frag[f][lane] = k_fragment(kp, kscale, page0, f, col, quad);
      __syncthreads();
      if (active)
#pragma unroll
        for (int f = 0; f < Frags; ++f) {
          const bf16x4 b = frag[f][lane];
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            acc[mt][f / KSteps] = mfma_bf16(a[mt][f % KSteps], b, acc[mt][f / KSteps]);
        }
      __syncthreads();                                     // before the next page overwrites frag
    } else if (active) {                                   // straddling tile: no LDS
#pragma unroll
      for (int f = 0; f < Frags; ++f) {
        const bf16x4 b = k_fragment(kp, kscale, my_page, f, col, quad);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          acc[mt][f / KSteps] = mfma_bf16(a[mt][f % KSteps], b, acc[mt][f / KSteps]);
      }
    }

    if (active) {
      // Accumulator element e of tile (mt, nt) is head mt*16 + 4*quad + e, key nt*16 + col.
#pragma unroll
      for (int nt = 0; nt < NTiles; ++nt) {
        float v = 0.f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) v += fmaxf(acc[mt][nt][e], 0.f) * wt[mt][e];
        v += __shfl_xor(v, 16);
        v += __shfl_xor(v, 32);
        const int key = p * P + nt * 16 + col;
        if (quad == nt && key < len) out[(long)row * out_stride + key] = v * weight_scale;
      }
    }
  }
}

template <int ROWS, int PAGES, int MT>
void launch(const void* q, const void* qs, const void* kp, const void* ks, const void* w,
            const void* pt, const void* lens, void* out, int T, int pt_width, int pt_stride,
            int out_stride, int qs_pad, float weight_scale, hipStream_t stream) {
  const dim3 grid((T + ROWS - 1) / ROWS, (pt_width + PAGES - 1) / PAGES);
  fp4_logits_kernel<ROWS, PAGES, MT><<<grid, dim3(ROWS * WaveSize), 0, stream>>>(
      (const uint8_t*)q, (const uint8_t*)qs, (const uint8_t*)kp, (const uint8_t*)ks,
      (const bf16*)w, (const int*)pt, (const int*)lens, (float*)out, T, pt_stride, out_stride,
      qs_pad, weight_scale);
}
}  // namespace

extern "C" int launch_dsv4_fp4_indexer_logits(
    const void* q, const void* qs, const void* kp, const void* ks, const void* w, const void* pt,
    const void* lens, void* out, int T, int H, int pt_width, int pt_stride, int out_stride,
    int qs_pad, float weight_scale, int rows, void* stream_) {
  const auto stream = (hipStream_t)stream_;
  if (T <= 0 || pt_width <= 0) return 0;
#define DSV4_IDX_LAUNCH(R, MT_)                                                                  \
  launch<R, 16, MT_>(q, qs, kp, ks, w, pt, lens, out, T, pt_width, pt_stride, out_stride, qs_pad, \
                     weight_scale, stream)
  if (H == 32) {
    if (rows == 16) DSV4_IDX_LAUNCH(16, 2);
    else if (rows == 8) DSV4_IDX_LAUNCH(8, 2);
    else if (rows == 4) DSV4_IDX_LAUNCH(4, 2);
    else return -2;
  } else if (H == 64) {
    if (rows == 16) DSV4_IDX_LAUNCH(16, 4);
    else if (rows == 8) DSV4_IDX_LAUNCH(8, 4);
    else if (rows == 4) DSV4_IDX_LAUNCH(4, 4);
    else return -2;
  } else {
    return -1;
  }
#undef DSV4_IDX_LAUNCH
  return (int)hipGetLastError();
}
