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
- `runner.py` — standalone benchmark harness comparing against AITER and (optionally) Modular MAX.
- `third_party/aiter/` — AITER as a git submodule, called through its Python API.

## Bench

`runner.py` compares the package's `ma.forward` against
[AITER](https://github.com/ROCm/aiter)'s `flash_attn_func` (V3 ASM forward),
RTNE and RTZ, on the same inputs. If the [Modular MAX](https://www.modular.com/max)
package is installed, it also benches `max.nn.kernels.flash_attention_gpu`.
For a multi-shape comparison, `bench_table.py` runs every shape against all
three backends with median-over-passes timing.

### Results — MI300X, bf16, head\_dim = 128

Median of 5 independent timing passes (30 iters each) per shape. Speedups are
`other_ms / ours_ms`, so >1× means we win. Modular MAX has no rounding-mode
selector and rounds RTNE internally (verified empirically).

| Shape (B, H, S, D) | Round | Ours (ms) | AITER v3 (ms) | Speedup vs AITER | Modular MAX (ms) | Speedup vs MAX |
|---|---|---|---|---|---|---|
| (2, 16, 2048, 128) | RTNE | **0.165** | 0.169 | 1.02× | 0.202 | 1.22× |
| (2, 16, 2048, 128) | RTZ | 0.155 | **0.145** | 0.94× | 0.202 | 1.30× |
| (2, 24, 4096, 128) | RTNE | **0.891** | 0.916 | 1.03× | 1.125 | 1.26× |
| (2, 24, 4096, 128) | RTZ | 0.833 | **0.806** | 0.97× | 1.125 | 1.35× |
| (2, 24, 8192, 128) | RTNE | **3.644** | 3.765 | 1.03× | 4.230 | 1.16× |
| (2, 24, 8192, 128) | RTZ | 3.368 | **3.267** | 0.97× | 4.230 | 1.26× |
| (2, 24, 16384, 128) | RTNE | **13.961** | 14.517 | 1.04× | 17.912 | 1.28× |
| (2, 24, 16384, 128) | RTZ | 12.991 | **12.464** | 0.96× | 17.912 | 1.38× |
| (2, 24, 32768, 128) | RTNE | **51.416** | 54.731 | 1.06× | 69.984 | 1.36× |
| (2, 24, 32768, 128) | RTZ | 48.202 | **48.200** | 1.00× | 69.984 | 1.45× |
| (1, 32, 8192, 128) | RTNE | **2.326** | 2.441 | 1.05× | 2.669 | 1.15× |
| (1, 32, 8192, 128) | RTZ | 2.160 | **2.136** | 0.99× | 2.669 | 1.24× |
| (1, 32, 16384, 128) | RTNE | **8.327** | 8.868 | 1.06× | 11.021 | 1.32× |
| (1, 32, 16384, 128) | RTZ | **7.754** | 7.801 | 1.01× | 11.021 | 1.42× |
| (4, 16, 4096, 128) | RTNE | **1.187** | 1.226 | 1.03× | 1.352 | 1.14× |
| (4, 16, 4096, 128) | RTZ | 1.113 | **1.086** | 0.98× | 1.352 | 1.21× |
| (4, 16, 16384, 128) | RTNE | **17.371** | 18.028 | 1.04× | 22.050 | 1.27× |
| (4, 16, 16384, 128) | RTZ | 16.038 | **15.953** | 0.99× | 22.050 | 1.37× |
| (1, 64, 16384, 128) | RTNE | **17.367** | 18.045 | 1.04× | 22.702 | 1.31× |
| (1, 64, 16384, 128) | RTZ | 16.102 | **15.971** | 0.99× | 22.702 | 1.41× |
| (2, 16, 65536, 128) | RTNE | **129.182** | 134.472 | 1.04× | 169.843 | 1.31× |
| (2, 16, 65536, 128) | RTZ | 120.561 | **119.409** | 0.99× | 169.843 | 1.41× |
| (2, 8, 86016, 128) | RTNE | **112.191** | 117.301 | 1.05× | 139.962 | 1.25× |
| (2, 8, 86016, 128) | RTZ | **104.692** | 105.022 | 1.00× | 139.962 | 1.34× |
| (1, 16, 131072, 128) | RTNE | **256.755** | 265.830 | 1.04× | 335.897 | 1.31× |
| (1, 16, 131072, 128) | RTZ | 238.855 | **236.990** | 0.99× | 335.897 | 1.41× |
| (1, 8, 262144, 128) | RTNE | **512.729** | 534.747 | 1.04× | 663.681 | 1.29× |
| (1, 8, 262144, 128) | RTZ | **476.776** | 478.115 | 1.00× | 663.681 | 1.39× |

Geomean speedup across shapes:
- **RTNE** — ours **1.04×** vs AITER, **1.26×** vs MAX
- **RTZ** — ours **0.98×** vs AITER, **1.35×** vs MAX

We beat AITER on RTNE on every shape (1.02–1.06×) and are within
noise of it on RTZ (0.94–1.01×). The win holds at long context: from 2K up through
256K positions we stay ahead of AITER on RTNE and match it on RTZ, and we are
consistently 14–45% faster than Modular MAX across the whole sweep.

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
conda create -n cdna3 python=3.10 ninja -y
conda activate cdna3

# 3. ROCm-built torch + AITER's runtime deps (skipping flydsl/matplotlib/pytest)
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install pandas pybind11 einops pyyaml psutil flydsl==0.1.3

# 4. install our package (compiles RTNE + RTZ kernels via hipcc)
pip install -e .

# 5. (optional) Modular MAX for the third bench row
pip install max

# 6. run. First call JIT-builds two AITER modules (~50s, then cached
#    under third_party/aiter/aiter/jit/build/).
python runner.py --warmup-iters 8 --benchmark-iters 30
```

`ninja` must be on `$PATH` for AITER's JIT, not just installed — the
conda recipe above takes care of it.

If `max` isn't installed (or you pass `--no-max`), runner skips the MAX row
and prints a one-line "skipped" notice. The MAX path round-trips through
the host once at setup to seed its `Buffer`s; the timed loop is device-only.

See `examples/basic.py` for a small correctness check using a fp32 reference.
