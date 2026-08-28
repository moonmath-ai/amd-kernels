"""Hand-tuned kernels for AMD CDNA3 (MI300X / gfx942).

>>> import torch
>>> import moonmath_amd as ma
>>> q = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
>>> k = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
>>> v = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
>>> out = ma.forward(q, k, v)         # torch.bfloat16, same shape
"""

from ._kernel import forward, forward_lite
from .lite import LiteAttention
from .mla import (
    mla_decode_a16w8,
    mla_decode_a16w8_multiq,
    mla_decode_a16w8_multiq_paged_dev,
    mla_decode_a16w8_multiq_plan_parts_q,
    mla_decode_a16w8_paged_dev,
    mla_decode_a16w8_plan_parts,
    mla_decode_a16w8_plan_parts_capped,
    mla_decode_a16w8_plan_parts_q,
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
    "forward", "forward_lite", "LiteAttention",
    "mla_decode_a16w8", "mla_decode_a16w8_paged_dev",
    "mla_decode_a16w8_plan_parts", "mla_decode_a16w8_plan_parts_capped",
    "mla_decode_a16w8_plan_parts_q",
    "mla_decode_a16w8_multiq", "mla_decode_a16w8_multiq_paged_dev",
    "mla_decode_a16w8_multiq_plan_parts_q",
    "repack_mxfp4", "repack_mxfp4_scales",
    "mxfp4_moe_gateup", "mxfp4_moe_down",
    "mxfp4_moe_gateup_block_m", "mxfp4_moe_gateup_supports_k",
    "mxfp4_moe_down_block_m",
    "mxfp4_moe_down_nt", "mxfp4_moe_down_n_steps", "mxfp4_moe_down_plan",
    "EPI_NONE", "EPI_SITU",
]
