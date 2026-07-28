"""MLA (DeepSeek-V3) absorbed-decode ops for CDNA3 (MI300X/gfx942).

A16W8: bf16 Q against fp8 KV. Q stays bf16 (no quantization error); KV is fp8
e4m3-fnuz, unpacked to bf16 for both QK and PV. TileTok=64, 8-wave warp-spec.
"""
import torch
import moonmath_attention._C as _C

__all__ = [
    "mla_decode_a16w8",
    "mla_decode_a16w8_paged_dev",
    "mla_decode_a16w8_plan_parts",
    "mla_decode_a16w8_plan_parts_capped",
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

    kv_pool:    [num_slots, 1, 576] fp8 e4m3-fnuz — fused pool at per-tensor kv_scale.
    seq_lens:   [B] int32 device — per-request KV lengths.
    rows:       [B] int32 device or None — maps req_pool_index -> q/out batch row (None -> identity).
    kv_indices: [sum(seq_lens)] int32 device — flat per-token slots.
    kv_indptr:  [B+1] int32 device — per-request slot offsets.
    parts:      fixed at graph capture (use mla_decode_a16w8_plan_parts_capped).
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
