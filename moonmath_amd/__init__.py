"""Hand-tuned kernels for AMD CDNA3 (MI300X / gfx942) and CDNA4 (MI355X / gfx950).

Every op dispatches on the live GPU's architecture; see moonmath_amd._arch.

>>> import torch
>>> import moonmath_amd as ma
>>> q = torch.randn(1, 1024, 4, 128, dtype=torch.bfloat16)   # shape [B, S, H, D]
>>> k = torch.randn(1, 1024, 4, 128, dtype=torch.bfloat16)
>>> v = torch.randn(1, 1024, 4, 128, dtype=torch.bfloat16)
>>> out = ma.forward(q, k, v, layout="bshd")   # bf16 result, same shape
"""

from ._kernel import VarlenPlan, forward, forward_lite, forward_paged, forward_varlen
from .lite import LiteAttention
from .mla import (
    dsv4_sparse_decode,
    mla_dcp_lse_merge_ranks,
    mla_decode_a16w8,
)
from .moe import (
    EPI_NONE,
    EPI_SITU,
    mxfp4_moe_down,
    mxfp4_moe_down_block_m,
    mxfp4_moe_down_n_steps,
    mxfp4_moe_down_nt,
    mxfp4_moe_down_plan,
    mxfp4_moe_gateup,
    mxfp4_moe_gateup_block_m,
    mxfp4_moe_gateup_supports_k,
    repack_mxfp4,
    repack_mxfp4_scales,
)

__all__ = [
    "forward", "forward_lite", "forward_varlen", "forward_paged", "VarlenPlan", "LiteAttention",
    "mla_decode_a16w8", "mla_dcp_lse_merge_ranks", "dsv4_sparse_decode",
    "repack_mxfp4", "repack_mxfp4_scales",
    "mxfp4_moe_gateup", "mxfp4_moe_down",
    "mxfp4_moe_gateup_block_m", "mxfp4_moe_gateup_supports_k",
    "mxfp4_moe_down_block_m",
    "mxfp4_moe_down_nt", "mxfp4_moe_down_n_steps", "mxfp4_moe_down_plan",
    "EPI_NONE", "EPI_SITU",
]
