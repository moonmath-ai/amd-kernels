// ===========================================================================
//  MXFP4 MoE gate/up projection. Also serves the down projection at any K the
//  short-K down kernel cannot take -- see moe_gateup_supports_k.
//
//      C[m][n] = sum_k A[m][k] * dequant(B[e][k][n])     e = expert of block m
//
//  B is MXFP4: an E2M1 nibble per weight, an E8M0 scale byte per 32 k.
//
//  EPI_SITU computes gate and up in paired wave halves that meet through LDS
//  to apply the GLU; EPI_NONE writes C straight out.
//
//  Per lane, col = lane & 15 and k_quarter = lane >> 4:
//      B   one 16-byte granule, 32 k of column n_base + col
//      Bs  the E8M0 byte covering those 32 k
//        A   rows (16*rt + col) for rt = 0 .. BM/16-1, read from LDS
//
//  `col` names the output column and the MFMA row at once. The row carries a
//  separate tile index rt, so one dequantized B fragment feeds BM/16
//  accumulators.
// ===========================================================================
#include "mxfp4_moe_common.h"

namespace gateup {

//  A lane's B granule is 32 k of one column and k_quarter takes four values, so the
//  64 lanes of a wave span 4 * 32 = 128 k. That is one slab.
constexpr int BK   = 128;
constexpr int NW   = 8;            // waves per workgroup
constexpr int NTHR = NW * 64;

//  BM is the workgroup's row extent. It sets how many MFMAs one dequantized B
//  fragment feeds (BM/16) and how often an expert's weight slab is re-read
//  (once per BM rows), against a padding cost of up to BM-1 rows per expert.
template <int BM> struct Tile {
    static constexpr int ROW_TILES        = BM / 16;        // MFMA row tiles per wave
    static constexpr int A_SLAB_BYTES     = BM * BK * 2;    // bytes per A slab
    static constexpr int GRANS_PER_ROW    = BK / 4;         // 8-byte granules per row
    static constexpr int GRANS_PER_THREAD = BM * GRANS_PER_ROW / NTHR;

    static_assert(BM % 16 == 0 && BM * GRANS_PER_ROW % NTHR == 0, "whole tiles and passes");
};

//  NT is the number of column tiles a wave owns. A slab costs one barrier, one
//  A staging pass and ROW_TILES*BK/16 LDS reads however many columns a wave holds, so
//  more column tiles divide all three across more MFMAs. The dequant ratio is
//  unchanged: NT column tiles need NT B granules and drive NT times the MFMAs.


//  These waits name the registers they protect as read-write operands -- that
//  is load-bearing, and the why lives with wait_vmcnt in mxfp4_moe_common.h.
//  Dropping the operands leaves BM=16 computing wrong results.
template <int N>
DEVINL void wait_b(u32x4 (&b)[1], u32 (&s)[1]) {
    asm volatile("s_waitcnt vmcnt(%c2)" : "+v"(b[0]), "+v"(s[0]) : "n"(N) : "memory");
}

template <int N>
DEVINL void wait_b(u32x4 (&b)[2], u32 (&s)[2]) {
    asm volatile("s_waitcnt vmcnt(%c4)"
                 : "+v"(b[0]), "+v"(b[1]), "+v"(s[0]), "+v"(s[1]) : "n"(N) : "memory");
}

template <int N>
DEVINL void wait_b(u32x4 (&b)[3], u32 (&s)[3]) {
    asm volatile("s_waitcnt vmcnt(%c6)"
                 : "+v"(b[0]), "+v"(b[1]), "+v"(b[2]),
                   "+v"(s[0]), "+v"(s[1]), "+v"(s[2]) : "n"(N) : "memory");
}

//  GRANS_PER_THREAD is BM/16, so the A overloads cover BM = 16, 32 and 48.
template <int N>
DEVINL void wait_a(u32x2 (&a)[1]) {
    asm volatile("s_waitcnt vmcnt(%c1)" : "+v"(a[0]) : "n"(N) : "memory");
}

template <int N>
DEVINL void wait_a(u32x2 (&a)[2]) {
    asm volatile("s_waitcnt vmcnt(%c2)" : "+v"(a[0]), "+v"(a[1]) : "n"(N) : "memory");
}

template <int N>
DEVINL void wait_a(u32x2 (&a)[3]) {
    asm volatile("s_waitcnt vmcnt(%c3)"
                 : "+v"(a[0]), "+v"(a[1]), "+v"(a[2]) : "n"(N) : "memory");
}


//  LDS byte offset of granule g in row r, XOR-swizzled on the row's low bits.
//
//  Laid out flat the row stride is 64 dwords, so the 16 lanes of an MFMA row
//  tile all land on one bank pair. XOR-ing the granule index with the row
//  spreads them over all 16 pairs. The map is a bijection, so the staging
//  write uses it unchanged.
template <int BM>
DEVINL int a_addr(int slab, int r, int g) {
    return slab * Tile<BM>::A_SLAB_BYTES + r * (BK * 2) + ((g ^ (r & 15)) << 3);
}


// ---------------------------------------------------------------------------
//  One workgroup computes C[pid_m*BM .. +BM][pid_n*BN .. +BN].
//
//  Waves split N, not M: every wave holds all BM rows, so its dequantized B
//  fragment feeds ROW_TILES MFMAs. A goes through LDS, B never does. Under
//  EPI_SITU wave w pairs with w + NW/2, gate half and up half.
// ---------------------------------------------------------------------------
template <int EPI, int NT, int BM>
__global__ __launch_bounds__(NTHR, 2) void kernel(
    const bf16* __restrict__ A, const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ Bs, bf16* __restrict__ C,
    const bf16* __restrict__ topk_weights, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    int N, int K, int num_valid_tokens, int top_k, long stride_am,
    long stride_be, long stride_bkg, long stride_bn,
    long stride_se, long stride_skg, long stride_sn,
    long stride_cm, long stride_cn,
    int mul_routed_weight, float situ_beta, float situ_linear_beta)
{
    //  The 16-row tile issues its A granule a body early; see slab_body.
    constexpr bool A_EARLY = (BM <= 16);

    constexpr int ROW_TILES    = Tile<BM>::ROW_TILES;
    constexpr int GRANS_PER_THREAD = Tile<BM>::GRANS_PER_THREAD;
    constexpr int WAVES_PER_PROJ   = (EPI == EPI_SITU) ? NW / 2 : NW;   // waves per projection
    constexpr int BN    = WAVES_PER_PROJ * NT * 16;

    //  Vector-memory ops one slab puts in the queue, in issue order: GRANS_PER_THREAD A
    //  granules, then NT pairs of (b128 weights, scale byte).
    constexpr int B_OPS    = NT * 2;
    constexpr int SLAB_OPS = GRANS_PER_THREAD + B_OPS;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int col  = lane & 15;
    const int k_quarter   = lane >> 4;

    //  Blocks in an m-block share an expert's weights; xcd_linear groups them
    //  onto one XCD for L2 locality, worth more than the serialised round trip
    //  it costs. Unsigned because the signed divide's fix-up lands on that same
    //  chain, the longest serial run in the kernel.
    const int live_m = (int)(((u32)*num_tokens_post_padded + BM - 1) / BM);
    const int linear_id    = xcd_linear(live_m * (int)gridDim.x);
    if (linear_id < 0) return;

    const u32 grid_x    = gridDim.x;
    const int pid_m = (int)((u32)linear_id / grid_x);
    const int pid_n = (int)((u32)linear_id - (u32)pid_m * grid_x);
    const int expert = expert_ids[pid_m];

    //  Read before the early-out: these depend only on pid_m, so they fly
    //  alongside the expert_ids load. In range because xcd_linear only returns
    //  linear_id < live_m*gridDim.x.
    u32 stage_slot[GRANS_PER_THREAD];
#pragma unroll
    for (int p = 0; p < GRANS_PER_THREAD; ++p)
        stage_slot[p] =
            (u32)sorted_token_ids[pid_m * BM + (tid + p * NTHR) / Tile<BM>::GRANS_PER_ROW];

    if (expert < 0) return;

    const int proj  = (EPI == EPI_SITU) ? wave / WAVES_PER_PROJ : 0;
    const int col_tile = (EPI == EPI_SITU) ? wave % WAVES_PER_PROJ : wave;
    const int n_base = pid_n * BN + col_tile * NT * 16;

    // -- LDS ---------------------------------------------------------------
    extern __shared__ char lds[];
    bf16* a_lds = reinterpret_cast<bf16*>(lds);

    //  The gate exchange buffer aliases the A slabs, which are dead by the time
    //  it is written. At NT=2 it needs both of them, since BM*BN floats is
    //  exactly 2 * A_SLAB_BYTES.
    float* gate = reinterpret_cast<float*>(lds);

    //  Per-row store metadata, above both. Resolved once into LDS, a dead row
    //  becomes an out-of-range store offset that the hardware drops, so the
    //  epilogue needs no branch. Resolved in the epilogue instead it would be a
    //  global load per stored element, each fenced by a full vmcnt(0).
    u32*   row_off = reinterpret_cast<u32*>(lds + 2 * Tile<BM>::A_SLAB_BYTES);
    float* row_w   = reinterpret_cast<float*>(row_off + BM);   // EPI_NONE only

    //  Every ScaledTable there is, 4 KB of LDS. scaled_table() is a pure
    //  function of one E8M0 byte, so all 256 results are built once in the
    //  prologue and a slab costs one ds_read_b128 instead of rebuilding it.
    //  Lanes hold different columns and so different scale bytes, which makes
    //  the read bank-scattered, and it still beats rebuilding.
    u32* scale_tbl_lds = reinterpret_cast<u32*>(lds + 2 * Tile<BM>::A_SLAB_BYTES + BM * 8);

    //  A dead row's store offset: num_records, the smallest offset the hardware
    //  always rejects -- one dropped store and no branch. Why the sentinel is
    //  the SRD's own bound and not a fixed 2^31: mxfp4_moe_common.h.
    const u32 c_records = records((long)num_valid_tokens * stride_cm * 2);

    // -- A staging assignment ----------------------------------------------
    //  Thread t owns granules t, t+NTHR, ... of the slab. Row and granule come
    //  from a plain divide because GRANS_PER_ROW is a power of two.
    int stage_row[GRANS_PER_THREAD], stage_gran[GRANS_PER_THREAD];
    u32 stage_voff[GRANS_PER_THREAD];
    bool stage_live[GRANS_PER_THREAD];
#pragma unroll
    for (int p = 0; p < GRANS_PER_THREAD; ++p) {
        const int flat = tid + p * NTHR;
        stage_row[p]  = flat / Tile<BM>::GRANS_PER_ROW;
        stage_gran[p] = flat % Tile<BM>::GRANS_PER_ROW;
        const int slot = (int)stage_slot[p];
        stage_live[p] = slot < num_valid_tokens;
        const long r = (long)(slot / top_k);
        stage_voff[p] = (u32)(r * stride_am * 2 + stage_gran[p] * 8);

        //  row_off comes from the values this pass already loaded. Granule 0 of
        //  a row has exactly one owner, and that thread publishes it.
        if (stage_gran[p] == 0) {
            row_off[stage_row[p]] = stage_live[p]
                                ? (u32)((long)slot * stride_cm * 2) : c_records;

            //  row_w is unused on the SiTU path and would cost a third
            //  dependent load, so compile it out there.
            if constexpr (EPI != EPI_SITU)
                row_w[stage_row[p]] = (stage_live[p] && mul_routed_weight)
                                  ? (float)topk_weights[slot] : 1.f;
        }
    }

    //  Published by the same barrier that publishes the primed A slab.
    if (tid < 256) {
        const ScaledTable tbl = scaled_table((u32)tid);
        *reinterpret_cast<u32x4*>(scale_tbl_lds + tid * 4) =
            (u32x4){tbl.hi_lo, tbl.hi_hi, tbl.lo_lo, tbl.lo_hi};
    }

    const long a_bytes = (long)((num_valid_tokens - 1) / top_k + 1) * stride_am * 2;
    const auto a_rsrc  = make_buffer(A, records(a_bytes));
    const auto c_rsrc  = make_buffer(C, c_records);

    // -- B addressing -------------------------------------------------------
    //  One granule per lane per slab per column tile: column (n_base + col),
    //  k group k_quarter. The k cursor is wave-uniform and rides the scalar offset.
    const long n_col = (long)(n_base + col) + (long)proj * N;

    //  Each SRD covers exactly this expert's slice, so every offset is small
    //  and a k cursor past the end reads as zero instead of reaching into the
    //  next expert. The pipeline relies on that below.
    const auto b_rsrc = make_buffer(B  + (long)expert * stride_be,
                                    records((long)K / 32 * stride_bkg));
    const auto s_rsrc = make_buffer(Bs + (long)expert * stride_se,
                                    records((long)K / 32 * stride_skg));
    const u32  b_voff = (u32)(n_col * stride_bn + (long)k_quarter * stride_bkg);
    const u32  s_voff = (u32)(n_col * stride_sn + (long)k_quarter * stride_skg);

    const int nslabs = K / BK;

    // -- accumulators -------------------------------------------------------
    f32x4 acc[NT][ROW_TILES];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int r = 0; r < ROW_TILES; ++r) acc[t][r] = (f32x4){0.f, 0.f, 0.f, 0.f};

    // -- pipeline -----------------------------------------------------------
    //  Slab i is consumed out of slot i&1 while slab i+1 is issued into the
    //  other. The ring keeps the fragment out of a copy, which would be free to
    //  schedule above the wait gating it. vmcnt is a FIFO count, so both rungs
    //  are literals: the other slot's A is ready once only this slot's B ops
    //  remain, and this slot's B once only the next slot's slab remains.
    u32x2 a_ring[2][GRANS_PER_THREAD];
    u32x4 b_ring[2][NT];
    u32   scale_ring[2][NT];

    //  Cursors advance by a constant per slab, so an address costs an add, not
    //  a 64-bit multiply on the load's own chain. Nothing clamps them at the
    //  end of K: the last two bodies prefetch past it to keep the rungs
    //  constant, and each SRD covers only this expert, so it reads zero.
    u32 a_cursor = 0, b_cursor = 0, s_cursor = 0;
    constexpr int ASTEP = BK * 2;
    const u32 b_kstep = (u32)((BK / 32) * stride_bkg);
    const u32 s_kstep = (u32)((BK / 32) * stride_skg);

    auto issue_a = [&](int slot) {
#pragma unroll
        for (int p = 0; p < GRANS_PER_THREAD; ++p)
            a_ring[slot][p] = load_b64(
                a_rsrc, stage_live[p] ? stage_voff[p] : OFF_DROP, a_cursor);
        a_cursor += ASTEP;
    };

    auto issue_b = [&](int slot) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            b_ring[slot][t] = load_b128(
                b_rsrc, b_voff + (u32)(t * 16 * stride_bn), b_cursor);
            scale_ring[slot][t] = load_b8(
                s_rsrc, s_voff + (u32)(t * 16 * stride_sn), s_cursor);
        }
        b_cursor += b_kstep; s_cursor += s_kstep;
    };

    //  One slab. `slot` is a template argument so the ring index never becomes
    //  a runtime index into a register array.
    //
    //  The barrier sits at the END, after the MFMAs, and the LDS write it
    //  publishes belongs to the NEXT slab. That keeps the barrier off the path
    //  between a write and the read that depends on it, and puts this slab's
    //  matrix work between A landing and A being needed.
    auto slab_body = [&](auto slot_tag) {
        constexpr int slot = decltype(slot_tag)::value;
        //  The other half of the ping-pong. This body consumes `slot` and
        //  publishes `next_slot`'s A for the slab after it.
        constexpr int next_slot = slot ^ 1;
        wait_b<SLAB_OPS>(b_ring[slot], scale_ring[slot]);

        //  a_ring[slot] died in the previous body, so refilling here gives the
        //  load two slab bodies to land in. 16-row tile only: a wider tile
        //  already covers the latency in one body, and issuing BM/16 granules
        //  early would push the B refill back.
        if constexpr (A_EARLY) issue_a(slot);

        //  buffer_load_ubyte zero-extends, so the scale byte is already a table
        //  index and needs no mask.
        ScaledTable tbl[NT];
#pragma unroll
        for (int b = 0; b < NT; ++b) {
            const u32x4 words = *reinterpret_cast<const u32x4*>(
                scale_tbl_lds + scale_ring[slot][b] * 4);
            tbl[b] = ScaledTable{words[0], words[1], words[2], words[3]};
        }

        //  The lane's granule is 32 k of one column: four unpack8 calls, each
        //  feeding ROW_TILES MFMAs against ROW_TILES A row tiles, for each column tile.
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            u32 b_frag[NT][4];
#pragma unroll
            for (int t = 0; t < NT; ++t) unpack8(b_ring[slot][t][w], tbl[t], b_frag[t]);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int g = k_quarter * 8 + w * 2 + u;      // 8-byte granule in the row
#pragma unroll
                for (int r = 0; r < ROW_TILES; ++r) {
                    const u32x2 a_frag = *reinterpret_cast<const u32x2*>(
                        &a_lds[a_addr<BM>(slot, r * 16 + col, g) / 2]);
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        acc[t][r] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                            as_bf16x4(a_frag[0], a_frag[1]),
                            as_bf16x4(b_frag[t][2 * u], b_frag[t][2 * u + 1]),
                            acc[t][r], 0, 0, 0);
                }
            }
        }

        //  With A_EARLY the granule for slab i+1 went out at the top of the
        //  previous body, so the queue behind it holds that body's B group and
        //  this body's A. Without it, only this slot's B group.
        wait_a<A_EARLY ? B_OPS + GRANS_PER_THREAD : B_OPS>(a_ring[next_slot]);
#pragma unroll
        for (int p = 0; p < GRANS_PER_THREAD; ++p)
            *reinterpret_cast<u32x2*>(&a_lds[
                a_addr<BM>(next_slot, stage_row[p], stage_gran[p]) / 2]) =
                a_ring[next_slot][p];
        if constexpr (!A_EARLY) issue_a(slot);

        //  Refill B after its MFMAs have read it, for the slab two ahead. Same
        //  two registers, and the load now has a whole slab of matrix work to
        //  land behind instead of half of one.
        issue_b(slot);
        lds_barrier();
    };

    //  Prime. The issue order here must match the body's, A then B for each
    //  slab, because the rungs above name positions in the vmcnt FIFO rather
    //  than particular loads.
    issue_a(0);
    issue_b(0);
    issue_a(1);
    issue_b(1);
    wait_a<B_OPS + SLAB_OPS>(a_ring[0]);
#pragma unroll
    for (int p = 0; p < GRANS_PER_THREAD; ++p)
        *reinterpret_cast<u32x2*>(&a_lds[
            a_addr<BM>(0, stage_row[p], stage_gran[p]) / 2]) = a_ring[0][p];
    lds_barrier();

    //  nslabs is even -- moe_gateup_supports_k rejects an odd count, because
    //  the rungs only balance over a whole pair. So there is no tail body: a
    //  lone slab_body would run with the rungs of a full pair and read the
    //  wrong registers.
    for (int i = 0; i + 2 <= nslabs; i += 2) {
        slab_body(ic<0>{});
        slab_body(ic<1>{});
    }

    // -- epilogue -----------------------------------------------------------
    //  MFMA result layout: lane l holds C[4*(l>>4) + j][l & 15] of each tile.
    //
    //  Drain the surplus prefetch first. The ring's last two issues are the
    //  out-of-range re-reads that keep the rungs constant, and they still have
    //  registers in flight.
    wait_vmcnt<0>();

    if (EPI == EPI_SITU) {
        __syncthreads();
        if (proj == 0) {
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int r = 0; r < ROW_TILES; ++r)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        gate[(r * 16 + 4 * k_quarter + j) * BN
                             + col_tile * NT * 16 + t * 16 + col] =
                            acc[t][r][j];
        }
        __syncthreads();
        if (proj == 1) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int n_out = n_base + t * 16 + col;
                if (n_out >= N) continue;
                const u32 cbyte = (u32)((long)n_out * stride_cn * 2);
#pragma unroll
                for (int r = 0; r < ROW_TILES; ++r)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int row = r * 16 + 4 * k_quarter + j;
                        const float gate_v =
                            gate[row * BN + col_tile * NT * 16 + t * 16 + col];
                        store_b16(c_rsrc, bits16(situ_glu(gate_v, acc[t][r][j],
                                                 situ_beta, situ_linear_beta)),
                                  row_off[row] + cbyte);
                    }
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int n_out = n_base + t * 16 + col;
            if (n_out >= N) continue;
            const u32 cbyte = (u32)((long)n_out * stride_cn * 2);
#pragma unroll
            for (int r = 0; r < ROW_TILES; ++r)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int row = r * 16 + 4 * k_quarter + j;
                    store_b16(c_rsrc, bits16(acc[t][r][j] * row_w[row]),
                              row_off[row] + cbyte);
                }
        }
    }
}

}  // namespace gateup


// ---------------------------------------------------------------------------
//  The vmcnt rungs name positions in the load FIFO, and only balance over a
//  whole slab pair, so the slab count must be even. A lone tail body would run
//  with a full pair's rungs and read the wrong registers.
// ---------------------------------------------------------------------------
extern "C" int moe_gateup_supports_k(int K) {
    return (K % (2 * gateup::BK) == 0) ? 1 : 0;
}


// ---------------------------------------------------------------------------
//  Tile height, from how many rows an expert owns. A small tile wastes MFMAs
//  on padding; a large one re-reads the weight slab less often and feeds each
//  dequantized fragment to more MFMAs.
// ---------------------------------------------------------------------------
extern "C" int moe_gateup_block_m(int num_valid_tokens, int num_active_experts) {
    const float per_expert = (float)num_valid_tokens / (float)(num_active_experts > 0
                                                              ? num_active_experts : 1);
    if (per_expert <= 12.f) return 16;
    if (per_expert < 29.f)  return 32;
    return 48;
}


// ---------------------------------------------------------------------------
//  Launcher. `num_m_blocks` counts BM-row blocks, so the host must align its
//  metadata with the same block size.
// ---------------------------------------------------------------------------
extern "C" void moe_gateup_launch(
    int epilogue, int num_m_blocks, int block_m,
    const void* A, const void* B, const void* Bs, void* C, const void* topk_weights,
    const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_padded,
    int N, int K, int num_valid_tokens, int top_k, long stride_am,
    long stride_be, long stride_bkg, long stride_bn,
    long stride_se, long stride_skg, long stride_sn,
    long stride_cm, long stride_cn,
    int mul_routed_weight, float situ_beta, float situ_linear_beta, void* stream)
{
    using namespace gateup;

    //  A wave's A fragment feeds one MFMA per column tile, so two column tiles
    //  halve the LDS reads per MFMA. With one, the wide tiles run the LDS pipe
    //  87% busy at 84% of its byte ceiling and nothing else is close; with two
    //  it drops to 50% and the SIMD issue port becomes the limit. The 16-row
    //  tile keeps one: its A slab is a third the size and its grid is already
    //  the finest.
    //
    //  The cost is a twice-wider column block, so take it only when that block
    //  covers no more columns than the narrow one already did. Where it covers
    //  more, the dead columns run the same MFMAs as live ones and the ~1.3x
    //  goes with them.
    constexpr int NT_WIDE = 2, NT_NARROW = 1, NT_WIDEST = 3;

    const bool situ = (epilogue == EPI_SITU);
    const int  waves_per_proj = situ ? NW / 2 : NW;
    auto columns_covered = [&](int nt) {
        const int bn = waves_per_proj * nt * 16;
        return (N + bn - 1) / bn * bn;
    };
    //  A third tile divides them again. EPI_NONE only: SituGLU exchanges gate
    //  and up through the A slab, where BM*BN floats fit at two and not three.
    const int nt_max = situ ? NT_WIDE : NT_WIDEST;
    int nt_pick = NT_NARROW;
    if (block_m > 16)
        for (int c = nt_max; c >= NT_WIDE; --c)
            if (columns_covered(c) == columns_covered(NT_NARROW)) { nt_pick = c; break; }

    const int  BNs = waves_per_proj * nt_pick * 16;
    const dim3 grid((N + BNs - 1) / BNs, num_m_blocks);
    auto st = static_cast<hipStream_t>(stream);

    auto go = [&](auto epi, auto bm, auto nt) {
        constexpr int E = decltype(epi)::value, M = decltype(bm)::value,
                      T = decltype(nt)::value;
        if constexpr ((M > 16 || T == NT_NARROW) && (E == EPI_NONE || T <= NT_WIDE)) {
            hipLaunchKernelGGL((kernel<E, T, M>),
                grid, dim3(NTHR), 2 * Tile<M>::A_SLAB_BYTES + M * 8 + 4096, st,
                (const bf16*)A, (const unsigned char*)B, (const unsigned char*)Bs, (bf16*)C,
                (const bf16*)topk_weights, (const int*)sorted_token_ids, (const int*)expert_ids,
                (const int*)num_tokens_post_padded, N, K, num_valid_tokens, top_k, stride_am,
                stride_be, stride_bkg, stride_bn, stride_se, stride_skg, stride_sn,
                stride_cm, stride_cn, mul_routed_weight, situ_beta, situ_linear_beta);
        }
    };

    //  16, 32 and 48 only. A block size outside this set gets no launch, rather
    //  than a tile that does not match how the caller padded its metadata.
    auto pick = [&](auto epi) {
        auto tile = [&](auto bm) {
            if      (nt_pick == NT_WIDEST) go(epi, bm, ic<NT_WIDEST>{});
            else if (nt_pick == NT_WIDE)   go(epi, bm, ic<NT_WIDE>{});
            else                           go(epi, bm, ic<NT_NARROW>{});
        };
        switch (block_m) {
            case 16: tile(ic<16>{}); break;
            case 32: tile(ic<32>{}); break;
            case 48: tile(ic<48>{}); break;
            default: break;
        }
    };

    if (situ) pick(std::integral_constant<int, EPI_SITU>{});
    else      pick(std::integral_constant<int, EPI_NONE>{});
}
