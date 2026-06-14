# moonmath-attention

Hand-tuned bf16 forward attention kernel for AMD CDNA3 (MI300X / gfx942).

8-wave warp-specialized CTA: each wave owns 3 q-tiles (48 q-rows), two parked in
registers, the third staged through LDS; K streams HBM→LDS by direct DMA and V is
consumed pre-transposed straight from L1. Inputs are taken natively in either
`[B, S, H, D]` (BSHD) or `[B, H, S, D]` (BHSD) layout — no transposes anywhere.

A FlashDecoding-style **dense tail KV-split** recovers the stranded fractional
CU-round: when the grid doesn't tile evenly across the 304 CUs, the last partial
round's q-blocks are split along KV across the idle CUs and merged in fp32. It
turns on automatically only when a cost model says it pays (otherwise a single
launch), and is the main reason RTZ now beats AITER on every benchmarked shape.

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
- `benchmark/runner.py` — single-shape benchmark vs AITER and (optionally) Modular MAX.
- `benchmark/bench_table.py` — multi-shape sweep with median-over-passes timing.

## Bench

`runner.py` compares `ma.forward` against
[AITER](https://github.com/ROCm/aiter)'s `flash_attn_func` (V3 ASM forward) on
identical BSHD inputs across all three rounding modes. If the
[Modular MAX](https://www.modular.com/max) package is installed it also benches
`max.nn.kernels.flash_attention_gpu`; MAX is loaded and timed only after the
HIP/AITER timings complete so its runtime cannot perturb them.

### Results — MI300X, bf16, head\_dim = 128

Median of 5 independent timing passes (30 iters each) per shape, **with the dense
tail KV-split enabled**. Speedups are `other_ms / ours_ms`, so >1× means we win.
Ours and AITER are a fresh idle-GPU run; Modular MAX figures are carried from the
prior measurement on the same GPU (MAX is kernel-independent — the tail only
affects our column — and its runtime perturbs co-located timings). MAX has no
rounding-mode selector and rounds RTNE internally (verified empirically).

| Shape (B, H, S, D) | Round | Ours (ms) | AITER v3 (ms) | Speedup vs AITER | Modular MAX (ms) | Speedup vs MAX |
|---|---|---|---|---|---|---|
| (2, 24, 8192, 128) | RTNE | **3.083** | 3.792 | 1.23× | 4.237 | 1.37× |
| (2, 24, 8192, 128) | RTNA | **3.022** | 3.605 | 1.19× | 4.237 | 1.40× |
| (2, 24, 8192, 128) | RTZ | **2.983** | 3.303 | 1.11× | 4.237 | 1.42× |
| (2, 24, 16384, 128) | RTNE | **11.670** | 14.691 | 1.26× | 17.923 | 1.54× |
| (2, 24, 16384, 128) | RTNA | **11.479** | 13.801 | 1.20× | 17.923 | 1.56× |
| (2, 24, 16384, 128) | RTZ | **11.385** | 12.629 | 1.11× | 17.923 | 1.57× |
| (1, 32, 16384, 128) | RTNE | **8.013** | 9.031 | 1.13× | 11.030 | 1.38× |
| (1, 32, 16384, 128) | RTNA | **7.828** | 8.656 | 1.11× | 11.030 | 1.41× |
| (1, 32, 16384, 128) | RTZ | **7.731** | 7.989 | 1.03× | 11.030 | 1.43× |
| (4, 16, 16384, 128) | RTNE | **15.591** | 18.337 | 1.18× | 22.061 | 1.41× |
| (4, 16, 16384, 128) | RTNA | **15.331** | 17.567 | 1.15× | 22.061 | 1.44× |
| (4, 16, 16384, 128) | RTZ | **15.055** | 16.183 | 1.07× | 22.061 | 1.47× |
| (1, 64, 16384, 128) | RTNE | **15.528** | 18.333 | 1.18× | 22.763 | 1.47× |
| (1, 64, 16384, 128) | RTNA | **15.239** | 17.535 | 1.15× | 22.763 | 1.49× |
| (1, 64, 16384, 128) | RTZ | **15.040** | 16.161 | 1.07× | 22.763 | 1.51× |
| (2, 24, 32768, 128) | RTNE | **46.002** | 54.794 | 1.19× | 69.947 | 1.52× |
| (2, 24, 32768, 128) | RTNA | **44.440** | 52.363 | 1.18× | 69.947 | 1.57× |
| (2, 24, 32768, 128) | RTZ | **44.075** | 48.549 | 1.10× | 69.947 | 1.59× |
| (2, 16, 65536, 128) | RTNE | **117.612** | 136.301 | 1.16× | 171.273 | 1.46× |
| (2, 16, 65536, 128) | RTNA | **115.550** | 130.278 | 1.13× | 171.273 | 1.48× |
| (2, 16, 65536, 128) | RTZ | **114.665** | 121.668 | 1.06× | 171.273 | 1.49× |
| (2, 8, 86016, 128) | RTNE | **101.071** | 118.939 | 1.18× | 141.319 | 1.40× |
| (2, 8, 86016, 128) | RTNA | **100.165** | 114.515 | 1.14× | 141.319 | 1.41× |
| (2, 8, 86016, 128) | RTZ | **99.397** | 106.513 | 1.07× | 141.319 | 1.42× |
| (1, 16, 131072, 128) | RTNE | **232.517** | 269.278 | 1.16× | 339.322 | 1.46× |
| (1, 16, 131072, 128) | RTNA | **228.475** | 258.092 | 1.13× | 339.322 | 1.49× |
| (1, 16, 131072, 128) | RTZ | **226.152** | 239.587 | 1.06× | 339.322 | 1.50× |


Geomean speedup across shapes:
- **RTNE** — ours **1.18×** vs AITER, **1.44×** vs MAX
- **RTNA** — ours **1.15×** vs AITER, **1.47×** vs MAX
- **RTZ** — ours **1.08×** vs AITER, **1.49×** vs MAX

We now beat AITER on **every shape and every rounding mode**. RTNE/RTNA lead by
1.11–1.26×; RTZ — historically the tightest race, since RTZ is AITER's own fastest
variant — wins 1.03–1.11×. The dense tail KV-split is what erased the prior RTZ
losses at the three 16K B·H ≥ 32 shapes (e.g. (4, 16, 16384) RTZ went 0.95× → 1.07×).
The lead holds with context — 32K through 128K stay 1.06–1.19× across all modes.
Against Modular MAX we are 1.37–1.59× faster everywhere.

Reproduce with:

```sh
# --no-max gives the cleanest ours/AITER numbers (MAX's runtime perturbs co-located timings);
# drop it to also measure Modular MAX.
python bench_table.py --benchmark-iters 30 --warmup-iters 8 --passes 5 --no-max
```

### Running the bench from scratch

```sh
git clone https://github.com/moonmath-ai/cdna3-attention.git
cd cdna3-attention

# python env (ninja required for AITER JIT and our kernel build)
conda create -n cdna3 python=3.11 ninja -y
conda activate cdna3

# install package + bench deps (torch, amd-aiter, numpy; optional max)
pip install -e '.[bench]'

# --- or with uv ---
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[bench]'

# run. First AITER call JIT-builds fmha modules (~50s, then cached under ~/.aiter/).
python benchmark/runner.py --warmup-iters 8 --benchmark-iters 30
```

`ninja` must be on `$PATH` for AITER's JIT, not just installed — the
conda recipe above takes care of it.

If `max` isn't installed (or you pass `--no-max`), runner skips the MAX row
and prints a one-line "skipped" notice. MAX is initialized only after the
HIP and AITER timing loops have finished, so its runtime cannot perturb them.

See `examples/basic.py` for a small correctness check using a fp32 reference.
