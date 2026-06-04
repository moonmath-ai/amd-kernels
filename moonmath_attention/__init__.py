"""Hand-tuned bf16 forward attention kernel for AMD CDNA3 (MI300X / gfx942).

    >>> import torch
    >>> import moonmath_attention as ma
    >>> q = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
    >>> k = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
    >>> v = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
    >>> out = ma.forward(q, k, v)         # torch.bfloat16, same shape
"""
from ._kernel import forward, forward_lite
from .lite import LiteAttention, MoonLiteAttention  # MoonLiteAttention: deprecated alias

__all__ = ["forward", "forward_lite", "LiteAttention", "MoonLiteAttention"]
__version__ = "0.1.0"
