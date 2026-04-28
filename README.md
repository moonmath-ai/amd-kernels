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
| (2, 16, 2048, 128) | RTNE | 0.194 | **0.169** | 0.87× | 0.201 | 1.04× |
| (2, 16, 2048, 128) | RTZ | 0.188 | **0.145** | 0.77× | 0.201 | 1.07× |
| (2, 24, 4096, 128) | RTNE | 0.991 | **0.915** | 0.92× | 1.122 | 1.13× |
| (2, 24, 4096, 128) | RTZ | 0.961 | **0.804** | 0.84× | 1.122 | 1.17× |
| (2, 24, 8192, 128) | RTNE | **3.668** | 3.747 | 1.02× | 4.216 | 1.15× |
| (2, 24, 8192, 128) | RTZ | 3.577 | **3.265** | 0.91× | 4.216 | 1.18× |
| (2, 24, 16384, 128) | RTNE | 14.901 | **14.499** | 0.97× | 17.848 | 1.20× |
| (2, 24, 16384, 128) | RTZ | 14.548 | **12.429** | 0.85× | 17.848 | 1.23× |
| (2, 24, 32768, 128) | RTNE | 56.043 | **54.747** | 0.98× | 70.233 | 1.25× |
| (2, 24, 32768, 128) | RTZ | 54.869 | **48.284** | 0.88× | 70.233 | 1.28× |
| (1, 32, 8192, 128)  | RTNE | **2.360** | 2.434 | 1.03× | 2.667 | 1.13× |
| (1, 32, 8192, 128)  | RTZ | 2.318 | **2.131** | 0.92× | 2.667 | 1.15× |
| (1, 32, 16384, 128) | RTNE | 9.609 | **8.889** | 0.93× | 11.024 | 1.15× |
| (1, 32, 16384, 128) | RTZ | 9.340 | **7.822** | 0.84× | 11.024 | 1.18× |
| (4, 16, 4096, 128)  | RTNE | **1.207** | 1.226 | 1.02× | 1.352 | 1.12× |
| (4, 16, 4096, 128)  | RTZ | 1.187 | **1.086** | 0.91× | 1.352 | 1.14× |
| (4, 16, 16384, 128) | RTNE | 18.421 | **18.041** | 0.98× | 22.046 | 1.20× |
| (4, 16, 16384, 128) | RTZ | 17.904 | **15.967** | 0.89× | 22.046 | 1.23× |
| (1, 64, 16384, 128) | RTNE | 18.464 | **18.040** | 0.98× | 22.766 | 1.23× |
| (1, 64, 16384, 128) | RTZ | 17.951 | **15.975** | 0.89× | 22.766 | 1.27× |

Geomean speedup across shapes:
- **RTNE** — ours **0.97×** vs AITER, **1.16×** vs MAX
- **RTZ** — ours **0.87×** vs AITER, **1.19×** vs MAX

We trade blows with AITER on RTNE (parity within a few % each way, depending on
shape) and lose ~10% on RTZ — AITER's RTZ path uses 32×32×8 MFMA + 8 waves
at occ=1 + 64 KB LDS, structurally different from our 16×16×16 + 4 waves at
occ=2 + 32 KB LDS. We are consistently ~15-25% faster than Modular MAX.

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
