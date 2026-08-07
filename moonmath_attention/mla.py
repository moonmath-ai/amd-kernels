"""MLA (DeepSeek-V3) absorbed-decode ops for CDNA3 (MI300X/gfx942).

A16W8: bf16 Q against fp8 KV. Q stays bf16 (no quantization error); KV is fp8
e4m3-fnuz, unpacked to bf16 for both QK and PV.

Two CTA shapes are exposed, picked by op rather than by argument:

- ``mla_decode_a16w8*`` — one query row per head, H <= 16. TileTok=64, 8-wave
  warp-specialized (4 consumer / 4 producer waves). q_len 1..8, one draft
  position per CTA (QLC=1), so the register allocation is the q_len==1 one.
- ``mla_decode_a16w8_multiq*`` — a q_len 4..8 draft window resident per CTA
  (speculative-decode verify), H <= 16, B <= 32. TileTok=16, all 8 waves compute
  and the fp8->bf16 unpack happens once per token on the LDS fill. Draft position
  t attends to KV [0, seq_len - q_len + t] inclusive (end-aligned causal), so the
  last draft position sees the whole sequence.

Both take the same fused fp8 e4m3-fnuz 576-wide KV rows ([...,:512] latent,
[512:576] rope) at one per-tensor kv_scale, either as a contiguous [B, S, 576]
slab or as a [num_slots, 1, 576] paged pool, and both run on the caller's current
stream with no host synchronization.
"""
import torch
import moonmath_attention._C as _C

__all__ = [
    "mla_decode_a16w8",
    "mla_decode_a16w8_paged_dev",
    "mla_decode_a16w8_plan_parts",
    "mla_decode_a16w8_plan_parts_capped",
    "mla_decode_a16w8_plan_parts_q",
    "mla_decode_a16w8_multiq",
    "mla_decode_a16w8_multiq_paged_dev",
    "mla_decode_a16w8_multiq_plan_parts_q",
]

_LAT = 512
_ROPE = 64
_QK = _LAT + _ROPE  # 576


def mla_decode_a16w8(
    q_lat: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    o_lat: torch.Tensor,
    scale: float,
    kv_scale: float,
    scale_p: float = 1.0,
) -> None:
    """A16W8 absorbed-decode MLA (q_len=1, H<=16).

    Args:
        q_lat: [B, H, 512] bf16 — latent query.
        q_pe:  [B, H, 64]  bf16 — RoPE query.
        kv:    [B, S, 576] fp8 e4m3-fnuz — fused KV ([...,:512]=latent, [512:576]=rope).
        o_lat: [B, H, 512] bf16 — output (written in-place).
        scale:     softmax scale (1/sqrt(qk_head_dim) x YaRN mscale).
        kv_scale:  per-tensor fp8 KV descale.
        scale_p:   unused (ABI symmetry). Defaults to 1.0.
    """
    _C.mla_decode_a16w8(q_lat, q_pe, kv, o_lat, float(scale), float(kv_scale), float(scale_p))


def mla_decode_a16w8_paged_dev(
    q_lat: torch.Tensor,
    q_pe: torch.Tensor,
    kv_pool: torch.Tensor,
    o_lat: torch.Tensor,
    seq_lens: torch.Tensor,
    rows: torch.Tensor | None,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    parts: int,
    scale: float,
    kv_scale: float,
    scale_p: float = 1.0,
) -> None:
    """A16W8 absorbed-decode MLA, device-driven paged (cuda-graph capturable).

    q_lat/q_pe: bf16 [B,H,*] (q_len=1) or [B,q_len,H,*] (draft window, q_len<=8).
                Position t attends KV [0, S - q_len + t] (end-aligned causal), so
                `seq_lens` INCLUDES the q_len draft tokens just written.
    kv_pool:    [num_slots, 1, 576] fp8 e4m3-fnuz — fused pool at per-tensor kv_scale.
    seq_lens:   [B] int32 device — per-request KV lengths.
    rows:       [B] int32 device or None — maps req_pool_index -> q/out batch row (None -> identity).
    kv_indices: [sum(seq_lens)] int32 device — flat per-token slots.
    kv_indptr:  [B+1] int32 device — per-request slot offsets.
    parts:      fixed at graph capture (mla_decode_a16w8_plan_parts_capped, or
                _plan_parts_q when q_len > 1 -- the window costs q_len CTAs per
                (batch, kv-part), so the KV split has to give parts back).
    """
    _C.mla_decode_a16w8_paged_dev(
        q_lat, q_pe, kv_pool, o_lat, seq_lens, rows, kv_indices, kv_indptr,
        int(parts), float(scale), float(kv_scale), float(scale_p),
    )


def mla_decode_a16w8_plan_parts(B: int, H: int, max_seq_len: int, lat_dim: int = 512) -> int:
    """Fixed KV-split count for a graph captured at (B, H, max_seq_len). Call at capture, reuse on replay."""
    return int(_C.mla_decode_a16w8_plan_parts(int(B), int(H), int(max_seq_len), int(lat_dim)))


def mla_decode_a16w8_plan_parts_capped(B: int, H: int, max_seq_len: int, lat_dim: int = 512) -> int:
    """As plan_parts, additionally clamped so B*parts <= MaxCTAs."""
    return int(_C.mla_decode_a16w8_plan_parts_capped(int(B), int(H), int(max_seq_len), int(lat_dim)))


def mla_decode_a16w8_plan_parts_q(B: int, max_seq_len: int, q_len: int = 1, H: int = 16) -> int:
    """q_len-aware fixed KV-split count for a graph captured at (B, max_seq_len, q_len).

    Prefer this over plan_parts_capped when q_len > 1. This kernel runs the draft
    window at one position per CTA (QLC=1), so a window of q_len multiplies the CTA
    count by q_len and the KV split must give parts back to stay inside the grid cap.
    H is accepted for ABI stability and ignored (H<=16 is a single query tile).
    """
    return int(_C.mla_decode_a16w8_plan_parts_q(int(B), int(max_seq_len), int(q_len), int(H)))


# ---- multi-query: a q_len 4..8 draft window resident per CTA ----------------


def mla_decode_a16w8_multiq(
    q_lat: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    o_lat: torch.Tensor,
    scale: float,
    kv_scale: float,
    scale_p: float = 1.0,
) -> None:
    """A16W8 multi-query absorbed-decode MLA (q_len 4..8, H<=16, B<=32).

    Args:
        q_lat: [B, q_len, H, 512] bf16 — latent query.
        q_pe:  [B, q_len, H, 64]  bf16 — RoPE query.
        kv:    [B, S, 576] fp8 e4m3-fnuz — fused KV ([...,:512]=latent, [512:576]=rope).
        o_lat: [B, q_len, H, 512] bf16 — output (written in-place).
        scale:     softmax scale (1/sqrt(qk_head_dim) x YaRN mscale).
        kv_scale:  per-tensor fp8 KV descale.
        scale_p:   unused (ABI symmetry with mla_decode_a16w8). Defaults to 1.0.

    Draft position t attends to kv[:, : S - q_len + t + 1]. q_len 1 is a different
    CTA shape and is served by mla_decode_a16w8.
    """
    _C.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, float(scale), float(kv_scale), float(scale_p))


def mla_decode_a16w8_multiq_paged_dev(
    q_lat: torch.Tensor,
    q_pe: torch.Tensor,
    kv_pool: torch.Tensor,
    o_lat: torch.Tensor,
    seq_lens: torch.Tensor,
    rows: torch.Tensor | None,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    parts: int,
    scale: float,
    kv_scale: float,
    scale_p: float = 1.0,
) -> None:
    """A16W8 multi-query decode, device-driven paged (cuda-graph capturable). q_len 4..8.

    q_lat/q_pe: [B, q_len, H, 512] / [B, q_len, H, 64] bf16; o_lat matches q_lat.
    kv_pool:    [num_slots, 1, 576] fp8 e4m3-fnuz — fused pool at per-tensor kv_scale.
    seq_lens:   [B] int32 device — per-request KV lengths; draft position t sees the first
                seq_lens[b] - q_len + t + 1 of them.
    rows:       [B] int32 device or None — maps req_pool_index -> q/out batch row (None -> identity).
    kv_indices: [sum(seq_lens)] int32 device — flat per-token slots (MLA page_size=1).
    kv_indptr:  [B+1] int32 device — per-request slot offsets.
    parts:      fixed at graph capture (use mla_decode_a16w8_multiq_plan_parts_q).
    """
    _C.mla_decode_a16w8_multiq_paged_dev(
        q_lat, q_pe, kv_pool, o_lat, seq_lens, rows, kv_indices, kv_indptr,
        int(parts), float(scale), float(kv_scale), float(scale_p),
    )


def mla_decode_a16w8_multiq_plan_parts_q(B: int, max_seq_len: int, q_len: int = 4, H: int = 16) -> int:
    """Fixed KV-split count for a multi-query graph captured at (B, max_seq_len, q_len, H).

    Accounts for the q_len/4 CTAs that share one KV stream, so the launched grid stays inside
    the workspace cap. H is accepted for ABI symmetry and does not shape the grid (H<=16 is one
    query tile). Call at capture, reuse on replay.
    """
    return int(_C.mla_decode_a16w8_multiq_plan_parts_q(int(B), int(max_seq_len), int(q_len), int(H)))
