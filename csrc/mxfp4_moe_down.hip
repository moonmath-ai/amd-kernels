// ===========================================================================
//  MXFP4 MoE down projection, decode and prefill.
//
//      C[m][n] = sum_k A[m][k] * dequant(B[e][k][n])     e = expert of block m
//
//  K is short, so a workgroup's whole A tile fits in LDS: stage once, sweep n
//  against it, one barrier. Sweeping all of N would leave far fewer workgroups
//  than CUs, so a workgroup takes a chunk of n; moe_down_n_steps sets it.
//
//      grid = ceil((N/BN) / n_steps) x num_m_blocks
//
//  Per lane, col = lane & 15 and k_quarter = lane >> 4:
//      B   one 16-byte granule, 32 k of column n_base + col
//      Bs  the E8M0 byte covering those 32 k
//      A   rows 16*rt + col, from LDS
//
//  SRD addressing, hand-managed vmcnt and the v_perm unpack: see
//  mxfp4_moe_common.h.
// ===========================================================================
#include "mxfp4_moe_common.h"

namespace down {

//  k per step: one 32-k granule per lane, four k_quarter values.
constexpr int BK = 128;

//  A 64-row tile needs 48 KB of LDS, so only one workgroup fits per CU.
//  Sixteen waves in that group restore the occupancy two 32-row groups would
//  have had. Shorter tiles use eight.
template <int BM> struct Waves { static constexpr int NW = (BM >= 64) ? 16 : 8; };

//  NT=1 is built for the 16-row tile only; everywhere else it is 2.
constexpr int nt_for(int block_m, int nt) { return (nt == 1 && block_m == 16) ? 1 : 2; }

//  Column extent of one workgroup, host side. Mirrors Cfg<BM,NT>::BN.
constexpr int bn_for(int block_m, int nt) {
    return ((block_m >= 64) ? 16 : 8) * nt_for(block_m, nt) * 16;
}

//  Column tiles per wave. Tile t owns column col*NT + t, so at NT=2 a lane's
//  two results are adjacent columns and store as one dword -- twice the
//  contiguous run into C, which is the largest stream at prefill. The second
//  tile costs 21 VGPR, halving workgroups per CU, so decode takes NT=1.
template <int BM, int NT> struct Cfg {
    static constexpr int NW = Waves<BM>::NW, NTHR = NW * 64, BN = NW * NT * 16;
};


//  These waits name the registers they protect as read-write operands -- that
//  is load-bearing, and the why lives with wait_vmcnt in mxfp4_moe_common.h.
//  Dropping the operands leaves BM=16 computing wrong results.
template <int N>
DEVINL void wait_b(u32x4 (&b)[1], u32 (&s)[1]) {
    asm volatile("s_waitcnt vmcnt(%c2)" : "+v"(b[0]), "+v"(s[0]) : "n"(N) : "memory");
}

//  Two overloads is all NT reaches: one granule and one scale byte at NT=1,
//  two granules and one packed scale word at NT=2. See issue().
template <int N>
DEVINL void wait_b(u32x4 (&b)[2], u32 (&s)[1]) {
    asm volatile("s_waitcnt vmcnt(%c3)"
                 : "+v"(b[0]), "+v"(b[1]), "+v"(s[0]) : "n"(N) : "memory");
}


//  LDS offset of granule g in row r. The row stride is a multiple of 32 dwords,
//  so flat the 16 lanes of an MFMA row tile hit one bank pair; the XOR spreads
//  them over all 16 and is a bijection, so staging writes use the same map.
//  GRANS_PER_ROW is a template argument, not K/4 from the arg list: a runtime
//  stride costs the ds_read2st64_b64 pairing.
template <int GRANS_PER_ROW>
DEVINL int a_addr(int r, int g) {
    return r * (GRANS_PER_ROW * 8) + ((((g >> 1) ^ (r & 15)) << 4) | ((g & 1) << 3));
}


// ---------------------------------------------------------------------------
//  K_STEPS = K/BK is a template argument so the prefetch ring index is compile time.
//  A runtime index into a register array lowers to scratch.
// ---------------------------------------------------------------------------
template <int BM, int K_STEPS, int NT>
__global__ __launch_bounds__(Waves<BM>::NW * 64) void kernel(
    const bf16* __restrict__ A, const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ Bs, bf16* __restrict__ C,
    const bf16* __restrict__ topk_weights, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    int N, int K, int num_valid_tokens, int top_k, long stride_am,
    long stride_be, long stride_bkg, long stride_bn,
    long stride_se, long stride_skg, long stride_sn,
    long stride_cm, long stride_cn,
    int mul_routed_weight, int n_steps)
{
    constexpr int ROW_TILES   = BM / 16;
    constexpr int NW   = Cfg<BM, NT>::NW;
    constexpr int NTHR = Cfg<BM, NT>::NTHR;
    constexpr int BN   = Cfg<BM, NT>::BN;

    //  Vector-memory ops, counted in the order they enter the queue. vmcnt
    //  retires in issue order, so every wait below is spelled as "how many may
    //  still be outstanding" rather than as a raw number.
    constexpr int SLOT_OPS  = NT + 1;                       // NT granules + 1 scale
    constexpr int RING_OPS  = K_STEPS * SLOT_OPS;                // the whole prefetch ring

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int col  = lane & 15;
    const int k_quarter   = lane >> 4;

    //  Rows are sorted by expert, so consecutive m-blocks share one; xcd_linear
    //  groups them onto an XCD for L2 locality. Unsigned because the signed
    //  divide's fix-up lands on the *num_tokens_post_padded chain.
    const int live_m = (int)(((u32)*num_tokens_post_padded + BM - 1) / BM);
    const int linear_id    = xcd_linear(live_m * (int)gridDim.x);
    if (linear_id < 0) return;

    const u32 grid_x     = gridDim.x;
    const int pid_m  = (int)((u32)linear_id / grid_x);
    const int pid_n  = (int)((u32)linear_id - (u32)pid_m * grid_x);
    const int expert = expert_ids[pid_m];

    //  In range whether or not the compiler keeps it above the early-out:
    //  xcd_linear only returns linear_id < live_m*gridDim.x, so pid_m < live_m.
    const int stage_row  = tid / (NTHR / BM);
    u32       stage_slot = (u32)sorted_token_ids[pid_m * BM + stage_row];

    if (expert < 0) return;

    constexpr int GRANS_PER_ROW = K_STEPS * BK / 4;  // 8-byte A granules per row
    const int n_beg = (pid_n * n_steps) * BN;
    if (n_beg >= N) return;

    extern __shared__ char lds[];
    bf16* a_lds = reinterpret_cast<bf16*>(lds);

    //  The output row offset and routing weight are per ROW, so all sixteen
    //  lanes of a column group hold the same pair. Held in registers that is
    //  ROW_TILES*4 of each in every lane; held in LDS it is BM of each for the whole
    //  workgroup, and the registers that frees are what a second column tile
    //  costs. Stored as byte offsets, so the epilogue stores straight through
    //  a buffer SRD.
    u32*   row_off = reinterpret_cast<u32*>(lds + (size_t)BM * K * 2);
    float* row_w   = reinterpret_cast<float*>(row_off + BM);

    // -- B addressing -------------------------------------------------------
    //  Each SRD covers this expert's slice, so a cursor past the end reads as
    //  zero rather than reaching into the next expert.
    const auto b_rsrc = make_buffer(B  + (long)expert * stride_be,
                                    records((long)K / 32 * stride_bkg));
    const auto s_rsrc = make_buffer(Bs + (long)expert * stride_se,
                                    records((long)K / 32 * stride_skg));
    const u32  b_koff = (u32)((long)k_quarter * stride_bkg);
    const u32  s_koff = (u32)((long)k_quarter * stride_skg);

    //  One slot per k-step, so slot == ks and the n-step wrap is compile time.
    //  One scale word per slot, not NT: a lane's tiles are adjacent columns, so
    //  their E8M0 bytes are adjacent in Bs and one 16-bit load takes both.
    //  Requires stride_sn == 1, and at NT=2 an even N.
    u32x4 b_ring[K_STEPS][NT];
    u32   scale_ring[K_STEPS][1];   // one packed word at NT=2, else one each

    //  A tile's column advances by BN per n-step, so its byte offset advances by
    //  a constant. A buffer load adds its SGPR offset for free and the sweep is
    //  wave-uniform, so the lane's column sits in a VGPR that never changes and
    //  the advance in a scalar accumulator: one SALU add per n-step.
    u32 b_voff[NT], s_voff;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const long col_global = (long)(n_beg + wave * NT * 16 + col * NT + t);
        b_voff[t] = (u32)(col_global * stride_bn) + b_koff;
        if (t == 0) s_voff = (u32)(col_global * stride_sn) + s_koff;
    }
    const u32 b_nstep = (u32)((long)BN * stride_bn);
    const u32 s_nstep = (u32)((long)BN * stride_sn);
    u32 b_sweep = 0, s_sweep = 0;                     // wave-uniform sweep position

    //  `step` is how many n-steps past the current cursor the slot is being
    //  filled for: 1 in the steady state, 0 when the tail re-reads.
    auto issue = [&](int step, int ks) {
        const u32 b_soff = (u32)((long)ks * (BK / 32) * stride_bkg)
                         + b_sweep + step * b_nstep;
        const u32 s_soff = (u32)((long)ks * (BK / 32) * stride_skg)
                         + s_sweep + step * s_nstep;
#pragma unroll
        for (int t = 0; t < NT; ++t)
            b_ring[ks][t] = load_b128(b_rsrc, b_voff[t], b_soff);
        if constexpr (NT == 2) scale_ring[ks][0] = load_b16(s_rsrc, s_voff, s_soff);
        else                   scale_ring[ks][0] = load_b8 (s_rsrc, s_voff, s_soff);
    };

    //  A dead row's store offset: num_records, the smallest offset the hardware
    //  always rejects -- one dropped store and no branch. Why the sentinel is
    //  the SRD's own bound and not a fixed 2^31: mxfp4_moe_common.h.
    const u32 c_records = records((long)num_valid_tokens * stride_cm * 2);

    // -- stage the whole A tile, once, and prime the ring inside it ---------
    //  One row per THREADS_PER_ROW threads, a power of two so the row index is a
    //  shift. A thread's granules are strided by it, so every address is known
    //  before any is waited on and the loads issue back to back.
    //
    //  The ring primes behind the LAST batch: vmcnt retires in order, so
    //  priming earlier would make the wait for A drain the weights too.
    {
        const long a_bytes = (long)((num_valid_tokens - 1) / top_k + 1) * stride_am * 2;
        const auto a_rsrc  = make_buffer(A, records(a_bytes));

        constexpr int THREADS_PER_ROW   = NTHR / BM;            // threads on one row
        constexpr int GRANS_PER_THREAD   = GRANS_PER_ROW / THREADS_PER_ROW;          // 8-byte granules per thread
        constexpr int GRANS_PER_BATCH = (GRANS_PER_THREAD % 4 == 0) ? 4 : ((GRANS_PER_THREAD % 3 == 0) ? 3 : 2);
        constexpr int BATCHES  = GRANS_PER_THREAD / GRANS_PER_BATCH;

        //  The mapping only works if tid/THREADS_PER_ROW and tid%THREADS_PER_ROW are a shift and a
        //  mask, and if every thread stages a whole number of granules of one
        //  row. Both are properties of the instantiated set, not of the code,
        //  so they are checked rather than assumed.
        static_assert(NTHR % BM == 0 && (THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0,
                      "THREADS_PER_ROW = NTHR/BM must be a power of two");
        static_assert(GRANS_PER_ROW % THREADS_PER_ROW == 0 && GRANS_PER_THREAD % GRANS_PER_BATCH == 0,
                      "a thread stages whole granules, in whole batches");

        const int  row  = stage_row, granule_base = tid % THREADS_PER_ROW;
        const int  slot = (int)stage_slot;
        const bool row_live   = slot < num_valid_tokens;
        const u32  row_base =
            (u32)((long)(row_live ? slot / top_k : 0) * stride_am * 2);

        //  Published by the one thread per row that already holds the slot,
        //  which is what makes a separate metadata pass unnecessary.
        if (granule_base == 0) {
            row_off[row] = row_live ? (u32)((long)slot * stride_cm * 2) : c_records;
            row_w[row]   =
                (row_live && mul_routed_weight) ? (float)topk_weights[slot] : 1.f;
        }

        auto batch = [&](int first_gran, bool prime_ring) {
            u32x2 a_gran[GRANS_PER_BATCH];
#pragma unroll
            for (int p = 0; p < GRANS_PER_BATCH; ++p) {
                const int g = granule_base + (first_gran + p) * THREADS_PER_ROW;
                a_gran[p] = load_b64(
                    a_rsrc, row_live ? row_base + (u32)(g * 8) : OFF_DROP, 0);
            }
            if (prime_ring) {
#pragma unroll
                for (int ks = 0; ks < K_STEPS; ++ks) issue(0, ks);
                wait_vmcnt<RING_OPS>();
            } else {
                wait_vmcnt<0>();
            }
#pragma unroll
            for (int p = 0; p < GRANS_PER_BATCH; ++p) {
                *reinterpret_cast<u32x2*>(
                    &a_lds[a_addr<GRANS_PER_ROW>(
                        row, granule_base + (first_gran + p) * THREADS_PER_ROW) / 2]) = a_gran[p];
            }
        };
#pragma unroll
        for (int b = 0; b < BATCHES - 1; ++b) batch(b * GRANS_PER_BATCH, /*prime_ring=*/false);
        batch((BATCHES - 1) * GRANS_PER_BATCH, /*prime_ring=*/true);
    }
    lds_barrier();

    //  The MFMA operand order is load-bearing. Feeding the weights as srcA
    //  would transpose the result so a lane holds four consecutive columns and
    //  could store eight bytes at once - but then sixteen lanes write sixteen
    //  different rows, 7 KB apart. As it stands sixteen lanes write sixteen
    //  consecutive columns of one row, which is a single 32-byte transaction.

    const int n_end = (n_beg + n_steps * BN < N) ? n_beg + n_steps * BN : N;

    const auto c_rsrc = make_buffer(C, c_records);

    //  Resolve the C address once: the per-lane half is loop-invariant and goes
    //  in a VGPR, the per-step half is wave-uniform and rides the store's SGPR
    //  offset. By hand, because every store is an asm volatile with a memory
    //  clobber and the compiler will not hoist the LDS reads past it.
    const u32 lane_c = (u32)((long)(col * NT) * stride_cn * 2);
    u32       csoff  = to_scalar((u32)((long)(n_beg + wave * NT * 16) * stride_cn * 2));
    const u32 cstep  = (u32)((long)BN * stride_cn * 2);

    //  n_out < N is equivalent to nb < N - (wave*NT*16 + col*NT), and that
    //  constant is per-lane and loop-invariant, so the guard becomes one
    //  compare against the scalar loop counter.
    const int nlim = N - (wave * NT * 16 + col * NT);

    u32 coff[ROW_TILES][4];
#pragma unroll
    for (int r = 0; r < ROW_TILES; ++r) {
        //  A lane's four rows are consecutive (row = r*16 + 4*k_quarter + j), so the
        //  table arrives as one 16-byte LDS read instead of four.
        const u32x4 row_offs =
            *reinterpret_cast<const u32x4*>(&row_off[r * 16 + 4 * k_quarter]);
#pragma unroll
        for (int j = 0; j < 4; ++j) coff[r][j] = row_offs[j] + lane_c;
    }

    for (int nb = n_beg; nb < n_end; nb += BN) {
        f32x4 acc[NT][ROW_TILES];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int r = 0; r < ROW_TILES; ++r) acc[t][r] = (f32x4){0.f, 0.f, 0.f, 0.f};

        const int next = nb + BN;

#pragma unroll
        for (int ks = 0; ks < K_STEPS; ++ks) {
            //  This slot has landed once the only things still outstanding are
            //  the other K_STEPS-1 slots.
            wait_b<RING_OPS - SLOT_OPS>(b_ring[ks], scale_ring[ks]);

            //  No masking on the scale: the load zero-extends (b8 at NT=1, b16
            //  at NT=2), so the low byte is already clean and the high byte is
            //  clean after the shift. LLVM cannot see that through the asm and
            //  keeps a v_and inside every scaled_table otherwise.
            ScaledTable tbl[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t)
                tbl[t] = scaled_table(NT == 1 ? scale_ring[ks][0]
                                            : (t == 0 ? (scale_ring[ks][0] & 0xFFu)
                                                      : (scale_ring[ks][0] >> 8)));

#pragma unroll
            for (int w = 0; w < 4; ++w) {
                u32 b_frag[NT][4];
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    unpack8(b_ring[ks][t][w], tbl[t], b_frag[t]);
                const int g = ks * (BK / 4) + k_quarter * 8 + w * 2;
#pragma unroll
                for (int r = 0; r < ROW_TILES; ++r) {
                    const u32x4 a_frag = *reinterpret_cast<const u32x4*>(
                        &a_lds[a_addr<GRANS_PER_ROW>(r * 16 + col, g) / 2]);
#pragma unroll
                    for (int u = 0; u < 2; ++u)
#pragma unroll
                        for (int t = 0; t < NT; ++t)
                            acc[t][r] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                                as_bf16x4(a_frag[2 * u], a_frag[2 * u + 1]),
                                as_bf16x4(b_frag[t][2 * u], b_frag[t][2 * u + 1]),
                                acc[t][r], 0, 0, 0);
                }
            }

            //  Refill only after the MFMAs have read the slot. The last n-step
            //  has nothing left to prefetch but must still refill: the rung is
            //  a constant, so a draining queue would stop covering this slot.
            //  Re-read the current n-step and discard it.
            issue((next < n_end) ? 1 : 0, ks);
        }

        //  Lane l holds C[4*(l>>4) + j][l & 15] of each tile. At NT=2 those are
        //  two adjacent columns, stored as one dword: sixteen lanes cover 64
        //  contiguous bytes instead of 32. B's fragment goes strided in
        //  exchange, which the caches absorb.
#pragma unroll
        for (int t = 0; t < NT; t += (NT == 2 ? 2 : 1)) {
            //  n_out only runs past N on a workgroup's last n-step, so the
            //  stores this drops are never counted by a later rung.
            if (nb >= nlim - t) continue;
#pragma unroll
            for (int r = 0; r < ROW_TILES; ++r) {
                const f32x4 row_scale =
                    *reinterpret_cast<const f32x4*>(&row_w[r * 16 + 4 * k_quarter]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    //  bits16_hi leaves the rounded bf16 in bits [31:16], which
                    //  is where both consumers want it: the d16_hi store writes
                    //  that half directly, and the paired store assembles two of
                    //  them with one v_perm. Either way the shift disappears.
                    const u32 lo = bits16_hi(acc[t][r][j] * row_scale[j]);
                    if constexpr (NT == 2) {
                        const u32 hi = bits16_hi(acc[t + 1][r][j] * row_scale[j]);
                        store_b32(c_rsrc, vperm(hi, lo, 0x07060302u), coff[r][j], csoff);
                    } else {
                        store_b16hi(c_rsrc, lo, coff[r][j], csoff);
                    }
                }
            }
        }

        b_sweep  += b_nstep;
        s_sweep  += s_nstep;
        csoff += cstep;
    }
    wait_vmcnt<0>();
}

}  // namespace down


// ---------------------------------------------------------------------------
//  Tile height, or 0 when this kernel cannot serve the shape.
//
//  Dequant is 130/BM VALU per MFMA against a budget of four, so the tile is as
//  tall as the rows allow before padding eats the gain.
//
//  A workgroup stages its WHOLE A tile in LDS (BM*K*2 bytes) -- that is what
//  leaves one barrier, and what makes this short-K. Instantiated for
//  K/BK in {3,4,6}, i.e. K in {384,512,768}: the intermediate size at
//  TP8/TP6/TP4. BM=64 needs K=384; 64*768*2 is 96 KB against 64 KB of LDS.
//
//  Expert parallelism leaves the experts unsharded at K=3072, which fits no
//  tile here. Route that to the gate/up kernel with EPI_NONE, which stages A
//  one slab at a time. See moe_gateup_supports_k.
// ---------------------------------------------------------------------------
extern "C" int moe_down_block_m(int num_valid_tokens, int num_active_experts, int K) {
    const int ks = K / down::BK;
    if (K % down::BK != 0 || (ks != 3 && ks != 4 && ks != 6)) return 0;

    const float per_expert = (float)num_valid_tokens
                           / (float)(num_active_experts > 0 ? num_active_experts : 1);
    if (per_expert <= 12.f)  return 16;

    //  32 vs 64 alternates with the grid inside 7%; fix the endpoints only.
    if (per_expert <= 200.f || ks != 3) return 32;
    return 64;
}


//  Column tiles per wave. Two pairs a lane's results into one dword store,
//  which prefill wants; one costs 24 VGPR less -- four workgroups per CU
//  against two -- which wins while C is small.
extern "C" int moe_down_nt(int num_valid_tokens, int num_active_experts) {
    const float per_expert = (float)num_valid_tokens
                           / (float)(num_active_experts > 0 ? num_active_experts : 1);
    return (per_expert <= 4.f) ? 1 : 2;
}


// ---------------------------------------------------------------------------
//  Take the widest n-chunk that still leaves about `want` workgroups:
//
//      grid = ceil(total/steps) * num_m_blocks >= want
//
//  A wider chunk stages A fewer times, so widen until the grid rather than the
//  traffic is the binding constraint. The optimum is a plateau, not a point,
//  so the target only has to land inside it.
// ---------------------------------------------------------------------------
extern "C" int moe_down_n_steps(int num_m_blocks, int N, int num_cus, int block_m,
                                int nt) {
    const int bn    = down::bn_for(block_m, nt);
    const int total = (N + bn - 1) / bn;                     // n-steps in all of N

    //  Three generations of resident workgroups: the 16-row tile is four deep,
    //  the 32-row three, so one constant covers both. Below one generation the
    //  split never triggers; past five, staging A twice (BM*K*2 per workgroup)
    //  outgrows the occupancy it buys.
    const int want = 12 * (num_cus > 0 ? num_cus : 304);

    long want_steps = (num_m_blocks > 0)
                    ? (long)total * num_m_blocks / want : total;
    int steps = (want_steps < 1) ? 1
              : (want_steps > total ? total : (int)want_steps);

    //  At decode the m-blocks are about the live expert count, so the 16-row
    //  tile's grid is thin however it is cut and a wider chunk buys a cheaper
    //  A staging rather than costing occupancy.
    if (block_m == 16 && steps < 4 && total >= 4 && num_m_blocks <= 2 * num_cus
        && (long)(total / 4) * num_m_blocks >= 2L * num_cus) steps = 4;

    //  At most a two-way split above the 16-row tile: a taller tile's A slab is
    //  24-48 KB and is staged once per chunk.
    if (block_m > 16 && total >= 2 && steps < total / 2) steps = total / 2;

    steps = total / ((total + steps - 1) / steps);            // even split, no tail CTA
    return steps < 1 ? 1 : (steps > total ? total : steps);
}


extern "C" void moe_down_launch(
    int num_m_blocks, int block_m, int n_steps, int nt,
    const void* A, const void* B, const void* Bs, void* C, const void* topk_weights,
    const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_padded,
    int N, int K, int num_valid_tokens, int top_k, long stride_am,
    long stride_be, long stride_bkg, long stride_bn,
    long stride_se, long stride_skg, long stride_sn,
    long stride_cm, long stride_cn,
    int mul_routed_weight, void* stream)
{
    using namespace down;

    const int ntv   = nt_for(block_m, nt);
    const int bn    = bn_for(block_m, nt);
    const int total = (N + bn - 1) / bn;
    if (n_steps < 1 || n_steps > total) n_steps = total;

    const dim3 grid((total + n_steps - 1) / n_steps, num_m_blocks);
    auto st = static_cast<hipStream_t>(stream);

    auto go = [&](auto bm, auto ks, auto ntt) {
        constexpr int M = decltype(bm)::value, S = decltype(ks)::value;
        constexpr int NTv = decltype(ntt)::value;
        constexpr int THR = Cfg<M, NTv>::NTHR;
        hipLaunchKernelGGL((kernel<M, S, NTv>), grid, dim3(THR),
            (size_t)M * K * 2 + (size_t)M * 8, st,
            (const bf16*)A, (const unsigned char*)B, (const unsigned char*)Bs, (bf16*)C,
            (const bf16*)topk_weights, (const int*)sorted_token_ids, (const int*)expert_ids,
            (const int*)num_tokens_post_padded, N, K, num_valid_tokens, top_k, stride_am,
            stride_be, stride_bkg, stride_bn, stride_se, stride_skg, stride_sn,
            stride_cm, stride_cn, mul_routed_weight, n_steps);
    };

    //  BM=64 runs sixteen waves, which caps it at 128 VGPR per lane -- only
    //  the shortest K sweep fits.
    auto pick_ks = [&](auto bm, auto ntt) {
        constexpr int M = decltype(bm)::value;
        switch (K / BK) {
            case 3: go(bm, ic<3>{}, ntt); break;
            case 4: if constexpr (M < 64) go(bm, ic<4>{}, ntt); break;
            case 6: if constexpr (M < 64) go(bm, ic<6>{}, ntt); break;
            default: break;
        }
    };

    //  An unhandled (block_m, K) gets no launch and leaves C untouched, rather
    //  than a tile that does not match how the caller padded its metadata.
    //  moe_down_block_m returns 0 for exactly those shapes, so callers route.
    switch (block_m) {
        case 16:
            if (ntv == 1) pick_ks(ic<16>{}, ic<1>{});
            else          pick_ks(ic<16>{}, ic<2>{});
            break;
        case 32: pick_ks(ic<32>{}, ic<2>{}); break;
        case 64: pick_ks(ic<64>{}, ic<2>{}); break;
        default: break;
    }
}
