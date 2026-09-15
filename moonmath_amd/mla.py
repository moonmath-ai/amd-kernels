"""MLA (DeepSeek-V3) absorbed-decode ops for CDNA3 (MI300X/gfx942).

A16W8: bf16 Q against fp8 KV. Q stays bf16 (no quantization error); KV is fp8
e4m3-fnuz, unpacked to bf16 for both QK and PV.

``mla_decode_a16w8`` is one op for plain decode, speculative-decode verify and
decode context parallelism: a dense batch of any q_len, H 1..128, over a
device-driven paged pool (page_size 1). It plans its query tile and KV split per
launch from tensor shapes alone, so it is cuda-graph capturable with no host
synchronization and nothing to plan at capture time.

Draft position t of a q_len window attends KV [0, seq_len - (q_len-1-t)) --
end-aligned causal, so the last position sees the whole sequence and q_len 1 is
ordinary decode.

Under decode context parallelism (cp_world > 1) the pool is position-sharded:
global position p lives on rank p % cp_world at pool row p // cp_world, and Q is
head-replicated, so one rank runs all heads over its 1/N of the positions. The
causal limit then comes from the global lengths (``glen``), the op returns a
rank-local partial plus its base-2 LSE, and ``mla_dcp_lse_merge_ranks`` combines
the ranks' (partial, lse) pairs after the DCP all-to-all.
"""
import torch
import moonmath_amd._C as _C

__all__ = [
    "mla_decode_a16w8",
    "mla_dcp_lse_merge_ranks",
]


def mla_decode_a16w8(
    q_lat: torch.Tensor,
    q_pe: torch.Tensor,
    kv_pool: torch.Tensor,
    o_lat: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    scale: float,
    kv_scale: float,
    lse: torch.Tensor | None = None,
    glen: torch.Tensor | None = None,
    cp_rank: int = 0,
    cp_world: int = 1,
) -> None:
    """A16W8 absorbed-decode MLA over a dense batch, device-driven paged (graph capturable).

    q_lat/q_pe: [B*q_len, H, 512] / [B*q_len, H, 64] bf16 — request b owns rows [b*q_len, (b+1)*q_len).
                o_lat matches q_lat. B comes from kv_indptr, and q_len from the row count.
    kv_pool:    [num_slots, 1, 576] fp8 e4m3-fnuz — fused rows ([..., :512] latent, [512:] rope) at
                per-tensor kv_scale.
    seq_lens:   [B] int32 device — KV counts, including the q_len draft tokens just written.
    kv_indices: [sum(seq_lens)] int32 device — flat per-token pool slots (MLA page_size 1).
    kv_indptr:  [B+1] int32 device — per-request slot offsets. Its length fixes B.
    scale:      softmax scale (1/sqrt(qk_head_dim) x YaRN mscale).
    kv_scale:   per-tensor fp8 KV descale.
    lse:        [B*q_len, H] fp32 or None — base-2 log-sum-exp of each row's window, written in
                place; -inf (and a zero o_lat row) where that window is empty.

    Decode context parallelism, off by default:

    glen:       [B] int32 device or None — global KV counts including the draft window. With it,
                seq_lens are this rank's rank-local counts (~ glen / cp_world) and draft position t
                may attend global positions [0, glen[b] - (q_len-1-t)). Required when cp_world > 1.
    cp_rank:    this rank's index, 0 <= cp_rank < cp_world.
    cp_world:   the DCP world size; 1 is plain decode.
    """
    _C.mla_decode_a16w8(
        q_lat, q_pe, kv_pool, o_lat, seq_lens, kv_indices, kv_indptr,
        float(scale), float(kv_scale), lse, glen, int(cp_rank), int(cp_world),
    )


def mla_dcp_lse_merge_ranks(
    parts_in: torch.Tensor,
    lse_in: torch.Tensor,
    out: torch.Tensor,
    lse_out: torch.Tensor | None = None,
) -> None:
    """Cross-rank log-sum-exp merge, run after the DCP all-to-all.

    Combines the R ranks' (normalized partial, lse) pairs for the heads this rank owns:
    w_r = exp2(lse_r - max_r lse_r), out = sum_r w_r * parts_in[r] / sum_r w_r, with an all-empty
    (every lse -inf) row left at zero.

    parts_in: [R, T, HL, 512] bf16 — the ranks' normalized partials, stacked.
    lse_in:   [R, T, HL] fp32 — their base-2 log-sum-exps, as the decode op emits them.
    out:      [T, HL, 512] bf16 — merged output (written in-place).
    lse_out:  [T, HL] fp32 or None — the merged base-2 log-sum-exp.
    """
    _C.mla_dcp_lse_merge_ranks(parts_in, lse_in, out, lse_out)
