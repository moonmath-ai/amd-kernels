# moonmath-attention

Hand-tuned bf16 forward attention kernel for AMD CDNA3 (MI300X / gfx942).

## Install

Requires ROCm with `hipcc` on PATH and a gfx942 device.

```sh
pip install -e .
```

That builds two `.so` variants (RTNE and RTZ packing) into the package.

## Use

```python
import torch
import moonmath_attention as ma

q = torch.randn(2, 24, 8192, 128, dtype=torch.bfloat16)
k = torch.randn(2, 24, 8192, 128, dtype=torch.bfloat16)
v = torch.randn(2, 24, 8192, 128, dtype=torch.bfloat16)

out = ma.forward(q, k, v)                       # round_mode="rtne" by default
out_rtz = ma.forward(q, k, v, round_mode="rtz") # round_mode="rtz"
```

The kernel runs on the AMD GPU. CPU tensors are copied to the GPU and back
under the hood; if you have a ROCm-built torch and place tensors on a
`cuda`/`hip` device, `data_ptr()` is used in place (no copy).

## Constraints

- bf16 inputs / bf16 outputs.
- `head_dim == 128`, `seq_len % 64 == 0`.
- No causal mask, no GQA, no varlen.
- gfx942 / MI300X only (CDNA3).

## Layout / build internals

- `attention_kernel.hip` — the kernel.
- `moonmath_attention/` — Python package (ctypes wrapper around the `.so`).
- `Makefile` — direct kernel build (`make` produces RTNE + RTZ root-level `.so`).
- `runner.py` — standalone benchmark harness comparing against AITER.
- `attention_kernel_aiter_v3.cpp` — AITER reference for comparison.

## Bench

```sh
python runner.py --warmup-iters 8 --benchmark-iters 30
```

See `examples/basic.py` for a small correctness check using a fp32 reference.
