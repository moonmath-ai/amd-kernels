"""moonmath_attention vs SDPA (sanity).

Shapes match runner.py (B=2, H=24, S=16384, D=128) and README cross-attn (KV=512).
"""

from collections import namedtuple

import pytest
import torch
import torch.nn.functional as F

import moonmath_attention as ma

LAYOUTS = ("bshd", "bhsd")

# Shape suites: (B, H, S, D) with S_kv the cross-attention KV length.
# Kernel constraints: D == 128, S % 256 == 0, S_kv % 64 == 0.
Shape = namedtuple("Shape", "B H S D S_kv")
SHAPES = {
    "small": Shape(1, 4, 1024, 128, 256),
    "medium": Shape(2, 8, 4096, 128, 512),
    "runner": Shape(2, 24, 16384, 128, 512),
}


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    return torch.device("cuda")


def _randn(B, S, H, D, device):
    return torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)


def _rand_self_bshd(shape, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = _randn(shape.B, shape.S, shape.H, shape.D, device)
    k = _randn(shape.B, shape.S, shape.H, shape.D, device)
    v = _randn(shape.B, shape.S, shape.H, shape.D, device)
    return q, k, v


def _rand_cross_bshd(shape, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = _randn(shape.B, shape.S, shape.H, shape.D, device)
    k = _randn(shape.B, shape.S_kv, shape.H, shape.D, device)
    v = _randn(shape.B, shape.S_kv, shape.H, shape.D, device)
    return q, k, v


def _as_layout(q, k, v, layout):
    """Base tensors are BSHD; convert them to the requested `layout`."""
    if layout == "bshd":
        return q, k, v
    return tuple(t.transpose(1, 2).contiguous() for t in (q, k, v))


def _sdpa(q, k, v, layout):
    """SDPA expects (B, H, S, D); convert from `layout` and back."""
    if layout == "bshd":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.transpose(1, 2).contiguous() if layout == "bshd" else out


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_self_attention_sdpa_sanity(device, shape, layout):
    q, k, v = _as_layout(*_rand_self_bshd(shape, device), layout)
    out = ma.forward(q, k, v, round_mode="rtna", layout=layout)
    ref = _sdpa(q, k, v, layout)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.05)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_cross_attention_sdpa_sanity(device, shape, layout):
    q, k, v = _as_layout(*_rand_cross_bshd(shape, device), layout)
    out = ma.forward(q, k, v, round_mode="rtna", layout=layout)
    ref = _sdpa(q, k, v, layout)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.05)
