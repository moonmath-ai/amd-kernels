"""Hand-tuned bf16 forward attention kernel for AMD CDNA3 (MI300X / gfx942).

>>> import torch
>>> import moonmath_attention as ma
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

__all__ = [
    "forward", "forward_lite", "LiteAttention",
    "mla_decode_a16w8", "mla_decode_a16w8_paged_dev",
    "mla_decode_a16w8_plan_parts", "mla_decode_a16w8_plan_parts_capped",
    "mla_decode_a16w8_plan_parts_q",
    "mla_decode_a16w8_multiq", "mla_decode_a16w8_multiq_paged_dev",
    "mla_decode_a16w8_multiq_plan_parts_q",
]
