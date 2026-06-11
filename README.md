# moonmath-attention

Hand-tuned bf16 forward attention kernel for AMD CDNA3 (MI300X / gfx942).

8-wave warp-specialized CTA: each wave owns 3 q-tiles (48 q-rows), two parked in
registers, the third staged through LDS; K streams HBM→LDS by direct DMA and V is
consumed pre-transposed straight from L1. Inputs are taken natively in either
`[B, S, H, D]` (BSHD) or `[B, H, S, D]` (BHSD) layout — no transposes anywhere.

## Install

Requires ROCm with `hipcc` on PATH and a gfx942 device.

```sh
pip install -e .
```

That builds three `.so` variants (RTNA, RTNE and RTZ bf16 rounding) into the package.

## Use

```python
import torch
import moonmath_attention as ma

# diffusion-style BSHD tensors, no transpose needed
q = torch.randn(2, 8192, 24, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(2, 8192, 24, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(2, 8192, 24, 128, dtype=torch.bfloat16, device="cuda")

out      = ma.forward(q, k, v, layout="bshd")                    # RTNE rounding by default
out_rtna = ma.forward(q, k, v, layout="bshd", round_mode="rtna")
out_rtz  = ma.forward(q, k, v, layout="bshd", round_mode="rtz")

# classic BHSD works the same way (default layout)
qh = q.transpose(1, 2).contiguous()
out_h = ma.forward(qh, qh, qh)

# cross-attention: any KV length, no padding
ctx = torch.randn(2, 512, 24, 128, dtype=torch.bfloat16, device="cuda")
out_x = ma.forward(q, ctx, ctx, layout="bshd")
```

The kernel runs on the AMD GPU and is launched on the caller's current stream
(no device synchronization, so it overlaps cleanly inside larger pipelines).
CPU tensors are copied to the GPU and back under the hood.

## Constraints

- bf16 inputs / bf16 outputs.
- `head_dim == 128`.
- Any `seq_len ≥ 1` for Q and K/V independently (cross-attention supported);
  out-of-range rows are handled by hardware buffer bounds, not padding.
- No causal mask, no GQA, no varlen batching.
- gfx942 / MI300X only (CDNA3).

## Numerics

All three bf16 rounding modes match AITER's per-mode rounding rule. NaN/Inf
handling is bit- and position-identical with AITER for every rounding mode
(canonical `0x7FFF` NaN output), and every finite output element is within
1 bf16 ULP of AITER's. Outputs are deterministic run-to-run.

## Layout / build internals

- `csrc/attention_kernel.hip` — the kernel (attention + V pre-transpose).
- `moonmath_attention/` — Python package (ctypes wrapper around the `.so`).
- `Makefile` — direct kernel build (`make` produces root-level `.so` variants).
- `runner.py` — single-shape benchmark vs AITER and (optionally) Modular MAX.
- `bench_table.py` — multi-shape sweep with median-over-passes timing.
- `third_party/aiter/` — AITER as a git submodule, called through its Python API.

## Bench

`runner.py` compares `ma.forward` against
[AITER](https://github.com/ROCm/aiter)'s `flash_attn_func` (V3 ASM forward) on
identical BSHD inputs across all three rounding modes. If the
[Modular MAX](https://www.modular.com/max) package is installed it also benches
`max.nn.kernels.flash_attention_gpu`; MAX is loaded and timed only after the
HIP/AITER timings complete so its runtime cannot perturb them.

### Results — MI300X, bf16, head\_dim = 128

Median of 5 independent timing passes (30 iters each) per shape. Speedups are
`other_ms / ours_ms`, so >1× means we win. Modular MAX has no rounding-mode
selector and rounds RTNE internally (verified empirically).

| Shape (B, H, S, D) | Round | Ours (ms) | AITER v3 (ms) | Speedup vs AITER | Modular MAX (ms) | Speedup vs MAX |
|---|---|---|---|---|---|---|
| (2, 24, 8192, 128) | RTNE | **3.345** | 3.796 | 1.13× | 4.237 | 1.27× |
| (2, 24, 8192, 128) | RTNA | **3.274** | 3.604 | 1.10× | 4.237 | 1.29× |
| (2, 24, 8192, 128) | RTZ | **3.226** | 3.303 | 1.02× | 4.237 | 1.31× |
| (2, 24, 16384, 128) | RTNE | **11.670** | 14.669 | 1.26× | 17.923 | 1.54× |
| (2, 24, 16384, 128) | RTNA | **11.525** | 13.785 | 1.20× | 17.923 | 1.56× |
| (2, 24, 16384, 128) | RTZ | **11.406** | 12.591 | 1.10× | 17.923 | 1.57× |
| (1, 32, 16384, 128) | RTNE | **8.505** | 8.995 | 1.06× | 11.030 | 1.30× |
| (1, 32, 16384, 128) | RTNA | **8.431** | 8.574 | 1.02× | 11.030 | 1.31× |
| (1, 32, 16384, 128) | RTZ | 8.338 | **7.919** | 0.95× | 11.030 | 1.32× |
| (4, 16, 16384, 128) | RTNE | **17.335** | 18.245 | 1.05× | 22.061 | 1.27× |
| (4, 16, 16384, 128) | RTNA | **17.068** | 17.496 | 1.03× | 22.061 | 1.29× |
| (4, 16, 16384, 128) | RTZ | 16.958 | **16.138** | 0.95× | 22.061 | 1.30× |
| (1, 64, 16384, 128) | RTNE | **17.270** | 18.262 | 1.06× | 22.763 | 1.32× |
| (1, 64, 16384, 128) | RTNA | **17.041** | 17.511 | 1.03× | 22.763 | 1.34× |
| (1, 64, 16384, 128) | RTZ | 16.891 | **16.128** | 0.95× | 22.763 | 1.35× |
| (2, 24, 32768, 128) | RTNE | **46.444** | 54.747 | 1.18× | 69.947 | 1.51× |
| (2, 24, 32768, 128) | RTNA | **45.809** | 52.400 | 1.14× | 69.947 | 1.53× |
| (2, 24, 32768, 128) | RTZ | **45.354** | 48.468 | 1.07× | 69.947 | 1.54× |
| (2, 16, 65536, 128) | RTNE | **117.228** | 136.591 | 1.17× | 171.273 | 1.46× |
| (2, 16, 65536, 128) | RTNA | **115.663** | 130.469 | 1.13× | 171.273 | 1.48× |
| (2, 16, 65536, 128) | RTZ | **114.837** | 121.431 | 1.06× | 171.273 | 1.49× |
| (2, 8, 86016, 128) | RTNE | **100.713** | 118.902 | 1.18× | 141.319 | 1.40× |
| (2, 8, 86016, 128) | RTNA | **100.181** | 114.447 | 1.14× | 141.319 | 1.41× |
| (2, 8, 86016, 128) | RTZ | **99.530** | 106.613 | 1.07× | 141.319 | 1.42× |
| (1, 16, 131072, 128) | RTNE | **231.065** | 269.271 | 1.17× | 339.322 | 1.47× |
| (1, 16, 131072, 128) | RTNA | **228.830** | 258.065 | 1.13× | 339.322 | 1.48× |
| (1, 16, 131072, 128) | RTZ | **227.051** | 240.015 | 1.06× | 339.322 | 1.49× |


Geomean speedup across shapes:
- **RTNE** — ours **1.14×** vs AITER, **1.39×** vs MAX
- **RTNA** — ours **1.10×** vs AITER, **1.41×** vs MAX
- **RTZ** — ours **1.02×** vs AITER, **1.42×** vs MAX

We beat AITER on RTNE and RTNA on every shape (up to 1.26× at 16K), and on RTZ on
6 of 9 shapes (1.02–1.10×); the three 16K shapes with B·H ≥ 32 are within 5% on
RTZ. The lead grows with context — 32K through 128K hold 1.06–1.18× across all
modes. Against Modular MAX we are 1.27–1.57× faster everywhere.

Reproduce with:

```sh
python bench_table.py --benchmark-iters 30 --warmup-iters 8 --passes 5
```

### Running the bench from scratch

```sh
# 1. clone with the AITER submodule
#    (no recursion needed — we don't use AITER's own 3rdparty/composable_kernel)
git clone https://github.com/moonmath-ai/cdna3-attention.git
cd cdna3-attention
git submodule update --init third_party/aiter

# 2. python env. AITER JIT-compiles a Python-ABI-bound .so, so pin 3.11.
conda create -n cdna3 python=3.11 ninja -y
conda activate cdna3

# 3. ROCm-built torch + AITER's runtime deps (skipping flydsl/matplotlib/pytest)
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install pandas pybind11 einops pyyaml psutil flydsl==0.1.3

# 4. install our package (compiles RTNA + RTNE + RTZ kernels via hipcc)
pip install -e .

# 5. (optional) Modular MAX for the third bench column
pip install max

# 6. run. First call JIT-builds two AITER modules (~50s, then cached
#    under third_party/aiter/aiter/jit/build/).
python runner.py --warmup-iters 8 --benchmark-iters 30
```

`ninja` must be on `$PATH` for AITER's JIT, not just installed — the
conda recipe above takes care of it.

If `max` isn't installed (or you pass `--no-max`), runner skips the MAX row
and prints a one-line "skipped" notice. MAX is initialized only after the
HIP and AITER timing loops have finished, so its runtime cannot perturb them.

See `examples/basic.py` for a small correctness check using a fp32 reference.
