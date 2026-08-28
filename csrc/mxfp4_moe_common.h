// ===========================================================================
//  mxfp4_moe_common.h -- shared helpers for the MXFP4 MoE kernels (gate/up + down).
//
//  Four things here are relied on throughout both kernels, and none of them is
//  obvious from the call sites:
//
//    * v_perm_b32 byte-shuffle table lookups, which is how MXFP4 is decoded
//    * the buffer resource (SRD) as the only addressing mode, for fault safety
//      and register economy
//    * hand-managed vmcnt: every vector load is spelled in asm and every wait
//      is a literal placed by hand
//    * the MXFP4 unpack itself: E2M1 magnitudes with the E8M0 exponent folded
//      into a per-lane table, four values per v_perm, in natural k order
// ===========================================================================
#pragma once
#include <hip/hip_runtime.h>
#include <type_traits>
#include <utility>

typedef __bf16 bf16;
typedef bf16   bf16x4 __attribute__((ext_vector_type(4)));
typedef float  f32x4  __attribute__((ext_vector_type(4)));
typedef unsigned short u16;
typedef unsigned int   u32;
typedef u16    u16x2 __attribute__((ext_vector_type(2)));
typedef u32    u32x2 __attribute__((ext_vector_type(2)));
typedef u32    u32x4 __attribute__((ext_vector_type(4)));

#define DEVINL __device__ __forceinline__

//  Compile-time int, for passing tile constants to templated launch bodies.
template <int N> using ic = std::integral_constant<int, N>;


// ---------------------------------------------------------------------------
//  v_perm_b32, the byte shuffle both kernels are built on.
//
//  Concatenate {src1, src0} into one 8-byte table: bytes 0..3 from src1, bytes
//  4..7 from src0. Then result byte i = table[(sel >> 8i) & 7]. One
//  instruction gathers four bytes from an eight-byte table.
//
//  Selector byte 0x0C is special and emits a literal 0x00; scaled_table() uses
//  that to produce an exact zero for free.
// ---------------------------------------------------------------------------
DEVINL u32 vperm(u32 src0, u32 src1, u32 sel) {
    return __builtin_amdgcn_perm(src0, src1, sel);
}


// ===========================================================================
//  Hand-managed memory pipeline
//
//  gfx942 counts outstanding vector loads in one in-order per-wave counter, so
//  `s_waitcnt vmcnt(N)` waits until at most N remain and waiting on an old load
//  drains every newer one. The compiler places waits from register dependencies
//  and cannot prove the count at a loop back-edge, so it falls back to vmcnt(0)
//  and throws the pipeline away.
//
//  So every steady-state load is spelled in asm and every wait placed by hand.
//  All or nothing: one compiler-inserted vmcnt(0) anywhere in the loop redrains
//  it, which is why nothing bypasses the helpers below.
//
//  Loads go through a buffer resource rather than a per-lane pointer:
//    * lanes past the last column issue freely -- num_records clamps in
//      hardware and returns 0, and covers one expert, so a stray offset cannot
//      reach a neighbour
//    * 4 SGPRs and one loop-invariant VGPR, against 2 VGPRs and a 64-bit add
//      per issue
// ===========================================================================
constexpr u32 BufferConfig = 0x00020000u;   // SRD word3: DATA_FORMAT=32 (raw)

//  A 128-bit buffer resource over [p, p + num_records). Any offset at or past
//  num_records reads as 0 in hardware instead of faulting.
DEVINL __amdgpu_buffer_rsrc_t make_buffer(const void* p, u32 num_records) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), /*stride=*/0,
                                             num_records, BufferConfig);
}

//  Saturating cast. A region bigger than 4 GB cannot be range-checked by an
//  SRD, so fall back to unbounded rather than silently clamping real data.
DEVINL u32 records(long bytes) {
    return (bytes < 0 || bytes > 0xFFFFFFFFll) ? 0xFFFFFFFFu : (u32)bytes;
}


//  buffer_load ... offen
//
//      address  = base + soff + voff    (soff wave-uniform, voff per-lane)
//      in range = (voff + soff + payload) <= num_records
//
//  The SGPR offset counts toward the range check. Both kernels depend on that:
//  each carries a cursor that deliberately runs off the end of an expert's
//  slab in soffset and relies on reading zero rather than the next expert's
//  weights.
DEVINL u32x4 load_b128(__amdgpu_buffer_rsrc_t rsrc, u32 voff, u32 soff) {
    u32x4 r;
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen"
                 : "=v"(r) : "v"(voff), "s"(rsrc), "s"(soff) : "memory");
    return r;
}

DEVINL u32x2 load_b64(__amdgpu_buffer_rsrc_t rsrc, u32 voff, u32 soff) {
    u32x2 r;
    asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen"
                 : "=v"(r) : "v"(voff), "s"(rsrc), "s"(soff) : "memory");
    return r;
}

DEVINL u32 load_b16(__amdgpu_buffer_rsrc_t rsrc, u32 voff, u32 soff) {
    u32 r;
    asm volatile("buffer_load_ushort %0, %1, %2, %3 offen"
                 : "=v"(r) : "v"(voff), "s"(rsrc), "s"(soff) : "memory");
    return r;
}

DEVINL u32 load_b8(__amdgpu_buffer_rsrc_t rsrc, u32 voff, u32 soff) {
    u32 r;
    asm volatile("buffer_load_ubyte %0, %1, %2, %3 offen"
                 : "=v"(r) : "v"(voff), "s"(rsrc), "s"(soff) : "memory");
    return r;
}


//  A dead row's store offset is the C buffer's own num_records, the smallest
//  offset the hardware always rejects (for any wave-uniform soff too, since
//  the check is against num_records - soff). The row then costs one dropped
//  store and no branch, and every lane still issues unconditionally, which is
//  what lets a downstream wait rung count stores as a compile-time constant.
//
//  The sentinel has to be derived, not fixed: a constant stops being out of
//  range once C reaches 2 GiB (EP8 at 16384 tokens). The remaining limit is
//  C + one row under 4 GiB, where a 32-bit byte offset stops working at all.

//  The load-side equivalent: a voff the range check always rejects, so a
//  padded row reads zeros instead of costing a branch.
constexpr u32 OFF_DROP = 0xFFFFFFFFu;

DEVINL void store_b16(__amdgpu_buffer_rsrc_t rsrc, u32 v, u32 voff) {
    asm volatile("buffer_store_short %0, %1, %2, 0 offen"
                 :: "v"(v), "v"(voff), "s"(rsrc) : "memory");
}

//  Writes bits [31:16] of the data register, which is exactly where the
//  rounded bf16 from bits16_hi() already sits.
DEVINL void store_b16hi(__amdgpu_buffer_rsrc_t rsrc, u32 v, u32 voff, u32 soff) {
    asm volatile("buffer_store_short_d16_hi %0, %1, %2, %3 offen"
                 :: "v"(v), "v"(voff), "s"(rsrc), "s"(soff) : "memory");
}

DEVINL void store_b32(__amdgpu_buffer_rsrc_t rsrc, u32 v, u32 voff, u32 soff) {
    asm volatile("buffer_store_dword %0, %1, %2, %3 offen"
                 :: "v"(v), "v"(voff), "s"(rsrc), "s"(soff) : "memory");
}


//  s_waitcnt takes its count as a literal, and `%c` substitutes the template
//  argument directly into the mnemonic, so one line covers the whole 0..63
//  range that vmcnt's 6-bit field can express.
template <int Cnt>
DEVINL void wait_vmcnt() {
    static_assert(Cnt >= 0 && Cnt <= 63, "vmcnt is a 6-bit field");
    asm volatile("s_waitcnt vmcnt(%c0)" :: "i"(Cnt) : "memory");
}

//  Broadcast a wave-uniform value into an SGPR. The "s" soffset constraint
//  needs one, and without this the compiler will sometimes keep the cursor in
//  a VGPR and pay a readfirstlane per issue anyway.
DEVINL u32 to_scalar(u32 v) {
    return __builtin_amdgcn_readfirstlane(v);
}

//  __syncthreads() is a full fence and lowers to vmcnt(0) + lgkmcnt(0) +
//  s_barrier. Only the LDS half is wanted: the barrier orders this workgroup's
//  LDS traffic, while the weight stream is per-wave and must keep flying
//  straight through it.
DEVINL void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

DEVINL bf16x4 as_bf16x4(u32 lo, u32 hi) {
    u32x2 t = {lo, hi};
    return __builtin_bit_cast(bf16x4, t);
}


// ---------------------------------------------------------------------------
//  MI300X hands workgroups to its eight XCDs round-robin by block id, so
//  neighbouring blocks -- the ones sharing weights -- land on eight L2s and
//  each pulls the same bytes. Re-linearising gives every die a contiguous run,
//  making one expert's weights one L2's working set. A bijection for any grid.
//
//  `live` is the blocks that carry work, not the grid: the host sizes tables
//  for the worst case and over-counts by up to num_experts*(BM-1)/BM. Remapping
//  the whole grid would hand that dead tail to the last dies as a contiguous
//  run and idle them. Returns -1 past the live range.
//
//  Unsigned: this sits on the caller's pid_m divide chain, and the signed
//  forms' fix-ups would be four divides of pure latency on it.
// ---------------------------------------------------------------------------
DEVINL int xcd_linear(int live) {
    constexpr u32 XCD = 8;
    const u32 lin = blockIdx.x + gridDim.x * blockIdx.y;
    if (lin >= (u32)live) return -1;
    const u32 per = (u32)live / XCD, rem = (u32)live % XCD;
    const u32 die = lin % XCD, seq = lin / XCD;
    return (int)(die * per + (die < rem ? die : rem) + seq);
}


// ---------------------------------------------------------------------------
//  f32 -> bf16 bits, round to nearest even, by hand.
//
//  gfx942 has no f32-to-bf16 convert, so `(bf16)f` lowers to five VALU whose
//  v_cmp/v_cndmask/v_or quieten a NaN. -ffast-math cannot remove that guard --
//  it lives inside a builtin -- so every stored value pays for a case
//  -ffast-math already promised cannot happen.
//
//  Bit-identical to the builtin on every finite input, verified over all 2^32
//  patterns. NaN is the only divergence.
// ---------------------------------------------------------------------------
DEVINL u32 bits16(float f) {
    const u32 x = __builtin_bit_cast(u32, f);
    return (x + 0x7FFFu + ((x >> 16) & 1u)) >> 16;
}

//  The same rounding left in place in bits [31:16]. Pair with store_b16hi, or
//  with a v_perm that assembles two of them into one dword; either way the
//  final shift disappears.
DEVINL u32 bits16_hi(float f) {
    const u32 x = __builtin_bit_cast(u32, f);
    return x + 0x7FFFu + ((x >> 16) & 1u);
}


// ===========================================================================
//  E2M1 -> bf16, with the E8M0 scale already applied
//
//    magnitude q     0      1      2      3      4      5      6      7
//    value           0.0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
//    bf16 bits     0x0000 0x3F00 0x3F80 0x3FC0 0x4000 0x4040 0x4080 0x40C0
//
//  Multiplying a bf16 by 2^d adds d to its exponent field, which begins at bit
//  7, so it is just adding (d << 7) to the raw bits. The whole scaled table is
//  therefore four packed 16-bit adds over the four bf16 pairs.
//
//  The result is split into a hi-byte table and a lo-byte table, eight bytes
//  each, because eight bytes is exactly one v_perm_b32 source pair.
// ===========================================================================
struct ScaledTable {
    u32 hi_lo, hi_hi;   // hi bytes for q = 0..3 and q = 4..7
    u32 lo_lo, lo_hi;   // lo bytes, same split
};

DEVINL ScaledTable scaled_table(u32 e8m0) {
    //  Packed bf16 pairs [q_hi, q_lo], each 16-bit lane pre-biased by -0x3F80.
    //  The exponent delta is ((e8m0 - 127) << 7) = (e8m0 << 7) - 0x3F80, and
    //  the add below is modular in u16, so folding the bias into the constants
    //  makes the scale word a bare shift and removes the subtract.
    //
    //  The slots are shifted one place, so q = 0 comes from no constant at all.
    //  It has to be exactly 0.0, which would otherwise cost a mask to re-zero
    //  after the biased add; selector byte 0x0C makes v_perm emit a literal
    //  0x00 for free.
    //
    //  1.0 is paired with 0.5 rather than 1.5, and that is not cosmetic. 1.0's
    //  biased constant is 0x0000, and a pair of fully OR-able constants makes
    //  the compiler expand the packed add into a broadcast plus an OR, giving
    //  back half the saving. Pairing it with 0.5 carries, and keeps the packed
    //  add. Slot q = 0 is a don't-care, so it absorbs the reshuffle for free.
    const u32 c0 = 0x0000FF80u;   // [ 1.0 , 0.5 ]
    const u32 c1 = 0xFF000040u;   // [ dc  , 1.5 ]
    const u32 c2 = 0x00C00080u;   // [ 3.0 , 2.0 ]
    const u32 c3 = 0x01400100u;   // [ 6.0 , 4.0 ]

    const u32   sh = e8m0 << 7;                // e8m0 is always a byte
    const u16x2 dd = {(u16)sh, (u16)sh};

    const u32 s0 = __builtin_bit_cast(u32, __builtin_bit_cast(u16x2, c0) + dd);
    const u32 s1 = __builtin_bit_cast(u32, __builtin_bit_cast(u16x2, c1) + dd);
    const u32 s2 = __builtin_bit_cast(u32, __builtin_bit_cast(u16x2, c2) + dd);
    const u32 s3 = __builtin_bit_cast(u32, __builtin_bit_cast(u16x2, c3) + dd);

    ScaledTable T;                             // 0x0C -> a literal 0x00 == q0
    T.hi_lo = vperm(s1, s0, 0x0503010Cu);
    T.hi_hi = vperm(s3, s2, 0x07050301u);
    T.lo_lo = vperm(s1, s0, 0x0402000Cu);
    T.lo_hi = vperm(s3, s2, 0x06040200u);
    return T;
}


// ---------------------------------------------------------------------------
//  Unpack one u32 (8 MXFP4 weights) into 4 u32 of packed bf16.
//
//  Eight v_perm plus six masks and shifts, so 14 instructions per 8 values.
//  That is a floor rather than a current number: a 16-byte value table cannot
//  fit a v_perm's 8-byte source window, so a one-stage lookup is impossible,
//  and given two stages, 16 intermediate plus 16 output bytes at 4 bytes per
//  op is exactly 8 perms.
//
//  On repacked weights, byte j of word i holds k = 8i+j in its low nibble and
//  k = 8i+4+j in its high nibble. So "all low nibbles, then all high nibbles"
//  is natural k order, which is why A needs no matching shuffle.
// ---------------------------------------------------------------------------
DEVINL void unpack8(u32 w, const ScaledTable& T, u32* out) {
    const u32 NIB = 0x0F0F0F0Fu, MAG = 0x07070707u, SGN = 0x08080808u;

    const u32 half[2] = {w & NIB, (w >> 4) & NIB};   // one nibble per byte lane

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const u32 x   = half[h];
        const u32 mag = x & MAG;            // 0..7, already a v_perm selector
        const u32 sgn = (x & SGN) << 4;     // E2M1 bit 3 -> bf16 sign bit 15
        const u32 hb  = vperm(T.hi_hi, T.hi_lo, mag) | sgn;
        const u32 lb  = vperm(T.lo_hi, T.lo_lo, mag);

        //  zip the (lo, hi) byte pairs back into packed bf16
        out[2 * h + 0] = vperm(hb, lb, 0x05010400u);   // values 0,1
        out[2 * h + 1] = vperm(hb, lb, 0x07030602u);   // values 2,3
    }
}


// ===========================================================================
//  Epilogues
// ===========================================================================
enum Epilogue { EPI_NONE = 0, EPI_SITU = 1 };

DEVINL float sigmoid(float x) { return 1.0f / (1.0f + exp2f(-x * 1.44269504089f)); }

//  Kimi-K3 SituGLU:
//      gate_out = B * tanh(gate/B) * sigmoid(gate)
//      up_out   = L * tanh(up/L)
//  using tanh(x) = 2*sigmoid(2x) - 1, matching the reference implementation.
DEVINL float situ_glu(float g, float u, float B, float L) {
    const float tg = 2.0f * sigmoid(2.0f * g / B) - 1.0f;
    const float tu = 2.0f * sigmoid(2.0f * u / L) - 1.0f;
    return (B * tg * sigmoid(g)) * (L * tu);
}
