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

From scratch:

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
