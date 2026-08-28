# moonmath-attention

Hand-tuned kernels for AMD CDNA3 (MI300X / gfx942): a bf16 MHA forward kernel, a
pair of MLA (DeepSeek-V3) absorbed-decode kernels, and the grouped GEMMs of an
MXFP4 mixture-of-experts layer.

**MHA forward** — 8-wave warp-specialized CTA: each wave owns 3 q-tiles (48 q-rows),
two parked in registers, the third staged through LDS; K streams HBM→LDS by direct
DMA and V is consumed pre-transposed straight from L1. Inputs are taken natively in
either `[B, S, H, D]` (BSHD) or `[B, H, S, D]` (BHSD) layout — no transposes anywhere.

A FlashDecoding-style **dense tail KV-split** recovers the stranded fractional
CU-round: when the grid doesn't tile evenly across the 304 CUs, the last partial
round's q-blocks are split along KV across the idle CUs and merged in fp32. It
turns on automatically only when a cost model says it pays (otherwise a single
launch), and is the main reason RTZ now beats AITER on every benchmarked shape.

**MLA absorbed decode** — bf16 Q against fp8 e4m3-fnuz KV. One fused 576-wide KV row
is BOTH K and V (`W_UK`/`W_UV` absorbed outside), so the KV stream is read once at fp8
width and every MFMA is `v_mfma_f32_16x16x16bf16_1k`. Two CTA shapes, picked by op
rather than by argument:

- `mla_decode_a16w8*` — one draft position per CTA, `TileTok=64`, 8 waves split
  4 consumer / 4 producer. Serves q_len 1..8.
- `mla_decode_a16w8_multiq*` — a q_len 4..8 draft window resident per CTA
  (speculative-decode verify), `TileTok=16`, all 8 waves compute and the fp8→bf16
  unpack happens once per token on the LDS fill rather than per MFMA operand. At
  q_len 8 with H ≤ 12 the heads pack into six MFMA N-tiles and one CTA takes the
  whole window in a single pass over KV.

Both serve a contiguous KV slab and a device-driven paged pool (`page_size = 1`)
whose KV-split is fixed at capture, so the paged path is cuda-graph capturable with
no host synchronization.

**MXFP4 MoE GEMMs** — `mxfp4_moe_gateup` and `mxfp4_moe_down`, the grouped GEMMs of
an MXFP4 mixture-of-experts layer: E2M1 weight nibbles with one E8M0 scale per 32 k,
bf16 activations. [Details below](#mxfp4-moe-gemms).

## Install

Requires ROCm with `hipcc` on PATH and a gfx942 device.

```sh
pip install -e .
```

That compiles the MHA kernel in all three bf16 rounding modes (RTNA, RTNE, RTZ),
both MLA decode kernels and the MXFP4 MoE GEMMs into the package's `_C` extension.

## Use

### MHA forward

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

### MLA decode

`q_lat` / `q_pe` are the absorbed latent and RoPE queries; `kv` holds ONE fused fp8
row per token (`[..., :512]` latent, `[..., 512:576]` rope) at a single per-tensor
`kv_scale`. The output is written into `o_lat` in place.

```python
import torch
import moonmath_attention as ma

B, H, S, LAT, ROPE = 8, 16, 8192, 512, 64
scale, kv_scale = (LAT + ROPE) ** -0.5, 1.0 / 32.0
kv = (torch.randn(B, S, LAT + ROPE, device="cuda") / kv_scale).to(torch.float8_e4m3fnuz)

# q_len = 1 — plain decode.
q_lat = torch.randn(B, H, LAT, dtype=torch.bfloat16, device="cuda")
q_pe = torch.randn(B, H, ROPE, dtype=torch.bfloat16, device="cuda")
o_lat = torch.empty_like(q_lat)
ma.mla_decode_a16w8(q_lat, q_pe, kv, o_lat, scale, kv_scale)

# q_len = 4 — a speculative-decode draft window, resident per CTA. Draft position t
# attends KV [0, S - q_len + t] inclusive, so the last position sees the whole sequence.
q_len = 4
q_lat = torch.randn(B, q_len, H, LAT, dtype=torch.bfloat16, device="cuda")
q_pe = torch.randn(B, q_len, H, ROPE, dtype=torch.bfloat16, device="cuda")
o_lat = torch.empty_like(q_lat)
ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, scale, kv_scale)
```

The paged ops take a `[num_slots, 1, 576]` pool plus device `seq_lens` / `kv_indices`
/ `kv_indptr`, and a `parts` KV-split count fixed once at graph capture:

```python
parts = ma.mla_decode_a16w8_multiq_plan_parts_q(B, max_seq_len, q_len, H)
ma.mla_decode_a16w8_multiq_paged_dev(
    q_lat, q_pe, pool, o_lat, seq_lens, None, kv_indices, kv_indptr,
    parts, scale, kv_scale,
)
```

Passing `q_lens=` (a `[B]` int32 device tensor) gives each request a shorter live
window inside the padded, still rectangular tensors — a ragged speculative batch.
Rows past each request's window are left untouched in `o_lat`.

## Constraints

### MHA forward

- bf16 inputs / bf16 outputs.
- `head_dim == 128`.
- Any `seq_len ≥ 1` for Q and K/V independently (cross-attention supported);
  out-of-range rows are handled by hardware buffer bounds, not padding.
- No causal mask, no GQA, no varlen batching.
- gfx942 / MI300X only (CDNA3).

### MLA decode

- bf16 Q (`q_lat` `[.., H, 512]`, `q_pe` `[.., H, 64]`) against fp8 e4m3-fnuz KV;
  bf16 output. `kv_lora_rank = 512`, `qk_rope_head_dim = 64`.
- Fused 576-wide KV rows at ONE per-tensor `kv_scale` — latent and rope share it.
- `H ≤ 16`.
- `mla_decode_a16w8`: q_len 1..8. The contiguous entry point is q_len 1; the draft
  window is paged-only.
- `mla_decode_a16w8_multiq`: q_len 4..8, capped at `B * groups ≤ 152`, where `groups`
  is the CTAs one draft window costs — 1 at q_len 4, or at q_len 8 with H ≤ 12 where
  the whole window fits one CTA, and 2 otherwise. `B ≤ 32` is the tuned range.
  Below q_len 4, use `mla_decode_a16w8`.
- Paged pools are `page_size = 1`, so `kv_indices` is a flat per-token slot list and
  any permutation or subset of it is legal.
- gfx942 / MI300X only (CDNA3).

## Numerics

All three bf16 rounding modes match AITER's per-mode rounding rule. NaN/Inf
handling is bit- and position-identical with AITER for every rounding mode
(canonical `0x7FFF` NaN output), and every finite output element is within
1 bf16 ULP of AITER's. Outputs are deterministic run-to-run.

The MLA decode kernels are deterministic too: the KV-split is fixed at capture and
the fp32 partial merge sums in a fixed association. Against an fp32 reference over
the same dequantized KV (B=2, S=8192, H=16, q_len 4) they land at **2.6e-3**
relative error, against AITER's own a16w8 asm kernel at 6.3e-3 on those same inputs.
`benchmark/bench_mla.py` prints both before it times anything.

## MXFP4 MoE GEMMs

`mxfp4_moe_gateup` and `mxfp4_moe_down` are the grouped GEMMs of an MXFP4
mixture-of-experts layer. Weights are E2M1 nibbles with one E8M0 scale per 32 k.
Activations stay bf16 and are not quantized.

Gate/up walks K in slabs and takes an optional fused SituGLU epilogue. Down stages
its whole A tile in LDS and sweeps column chunks against it, which costs one barrier
for the GEMM but needs a short K. Down serves K = 384/512/768, the tensor-parallel
shards of the expert width; at any other K `mxfp4_moe_down_block_m` returns 0 and
the down projection runs on the gate/up kernel with `EPI_NONE`.

```python
w13 = ma.repack_mxfp4(w13_raw)          # once, at weight load
w13s = ma.repack_mxfp4_scales(w13s_raw)

bm = ma.mxfp4_moe_gateup_block_m(rows, num_active_experts)
sorted_ids, expert_ids, ntpp, nblk = moe_align_block_size(topk_ids, bm)
ma.mxfp4_moe_gateup(hidden, w13, w13s, out, None, sorted_ids, expert_ids,
                    ntpp, nblk, bm, rows, top_k, epilogue=ma.EPI_SITU)
```

### Results — Kimi-K3 at TP8, MI300X

896 routed experts of width 3072, top-16, K = 3584 (K3's
`routed_expert_hidden_size`, not the model's 7168 `hidden_size`). I = 384 is the TP8
shard. Four shapes: decode at T = 8 and T = 32, which land on 123 and 394 of the 896
experts, and chunked-prefill chunks of 8192 and 16384 tokens, which land on all 896.
Each token draws 16 distinct experts, uniform.

The baseline is a tuned AITER. There is no `gfx942-MOE-MX_FP4.json`, so the bench
times twelve tile candidates at every shape and reports the fastest, aligning the
metadata separately per tile so each pays its own padding. Median of 5 passes,
`fused_moe_mxfp4` from a stock aiter.

| Tokens | Rows | Projection | Ours (ms) | AITER best (ms) | Best tile | Speedup | Ours TFLOP/s |
|---|---|---|---|---|---|---|---|
| 8 | 128 | gate/up | **0.050** | 0.066 | 16x64_k2_e4 | 1.32× | 14 |
| 8 | 128 | down | **0.028** | 0.052 | 16x64_w2_s1 | 1.89× | 13 |
| 32 | 512 | gate/up | **0.172** | 0.240 | 16x64_k2_e4 | 1.39× | 16 |
| 32 | 512 | down | **0.088** | 0.108 | 16x64_w2_s1 | 1.22× | 16 |
| 8192 | 131072 | gate/up | **1.628** | 2.346 | 64x256_s1_k2_e2 | 1.44× | 443 |
| 8192 | 131072 | down | **1.055** | 1.463 | 64x256_s1_k2_e2 | 1.39× | 342 |
| 16384 | 262144 | gate/up | **2.989** | 4.022 | 64x256_s1_k2_e2 | 1.35× | 483 |
| 16384 | 262144 | down | **1.796** | 2.551 | 64x256_s1_k2_e2 | 1.42× | 402 |

Geomean 1.37× gate/up, 1.46× down. At decode, 512 rows over 394 experts is 1.3 rows
per expert, so most of each tile is padding and only the 16-row tile is competitive.

Reproduce with:

```sh
python benchmark/bench_moe.py --markdown            # the table above
python benchmark/bench_moe.py --tokens 8 512 2048   # any other token counts
python benchmark/bench_moe.py --layout ep8          # expert-parallel geometry
python benchmark/bench_moe.py --show-tiles          # every AITER tile, not just the winner
```

The bench needs AITER on the path for the Triton baseline
(`PYTHONPATH=/path/to/aiter`), and `python -m pytest tests/test_moe.py -q` checks
both kernels against a dense reference that never sees the repacked weights.

## Layout / build internals

- `csrc/attention_kernel.hip` — the MHA kernel (attention + V pre-transpose).
- `csrc/mla_decode_a16w8.hip` — MLA absorbed decode, one draft position per CTA.
- `csrc/mla_decode_a16w8_multiq.hip` — MLA absorbed decode, q_len 4..8 window.
- `csrc/mxfp4_moe_gateup.hip`, `csrc/mxfp4_moe_down.hip` — the MoE GEMMs.
- `csrc/*_api.cpp` — the torch bindings for each.
- `moonmath_attention/` — Python package (ctypes wrapper around the `.so`).
- `Makefile` — direct kernel build (`make` produces root-level `.so` variants).
- `benchmark/runner.py` — single-shape benchmark vs AITER and (optionally) Modular MAX.
- `benchmark/bench_table.py` — multi-shape sweep with median-over-passes timing.
- `benchmark/bench_mla.py` — MLA decode vs AITER's a16w8 ASM kernel, CUDA-graph timed.
- `benchmark/bench_moe.py` — MXFP4 MoE GEMMs vs AITER's Triton MXFP4 kernel.
- `tests/test_mla_decode.py` — the MLA ops against an fp32 reference built from the
  same dequantized KV: both CTA shapes, contiguous and paged, the end-aligned causal
  window, the `rows` remap, the ragged `q_lens` window and the domain rejections.
- `tests/test_moe.py` — the MoE GEMMs against a dense reference built from the
  dequantized stock weights, plus the repack, tile-shape and domain contracts.

## Bench

`runner.py` compares `ma.forward` against
[AITER](https://github.com/ROCm/aiter)'s `flash_attn_func` (V3 ASM forward) on
identical BSHD inputs across all three rounding modes. If the
[Modular MAX](https://www.modular.com/max) package is installed it also benches
`max.nn.kernels.flash_attention_gpu`; MAX is loaded and timed only after the
HIP/AITER timings complete so its runtime cannot perturb them.

### Results — MHA forward, MI300X, bf16, head\_dim = 128

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

### Results — MLA decode, MI300X, bf16 Q / fp8 KV, H = 16, q\_len = 4

`mla_decode_a16w8_multiq` against AITER's `a16w8` MLA decode ASM kernel — the only
AITER cell with our dtypes. Both sides run in one process on one shared paged fp8 KV
pool, same Q, same softmax scale, same end-aligned causal mask, same `page_size = 1`
shuffled slot permutation (a kernel that assumed contiguous slots would fail the
validation step). Timed as CUDA-graph replays — AITER's python op wrappers cost
~78 µs/call, which would otherwise swamp the kernel below S ≈ 64K — with
`num_kv_splits` swept per shape and only validated configs kept, and each candidate
timed once per round in alternating order, median over rounds. Speedups are
`aiter_µs / ours_µs`, so >1× means we win.

| Shape (B, S) | KV (MB) | Ours (µs) | AITER a16w8 (µs) | Speedup | Ours (TB/s) | AITER (TB/s) |
|---|---|---|---|---|---|---|
| (1, 150000) | 86 | **87.6** | 134.9 | 1.54× | 0.99 | 0.64 |
| (2, 150000) | 173 | **124.2** | 164.1 | 1.32× | 1.39 | 1.05 |
| (8, 8192) | 38 | **53.0** | 59.0 | 1.11× | 0.71 | 0.64 |
| (8, 32768) | 151 | **113.7** | 131.9 | 1.16× | 1.33 | 1.14 |
| (8, 65536) | 302 | **208.5** | 246.4 | 1.18× | 1.45 | 1.23 |
| (8, 150000) | 691 | **444.0** | 524.9 | 1.18× | 1.56 | 1.32 |
| (16, 150000) | 1382 | **868.4** | 1027.9 | 1.18× | 1.59 | 1.34 |
| (32, 8192) | 151 | **117.4** | 137.6 | 1.17× | 1.29 | 1.10 |

AITER's `a16w8` kernel asserts `nhead == 16`, and rejects q_len > 4 on fp8 KV, so
H = 12 (a DSV3 TP8 shard) and q_len 8 have no like-for-like AITER cell at all. That is
why the script pins H and q_len rather than sweeping them.

Reproduce with:

```sh
# Needs AITER importable with its gfx942 MLA kernels built (hsa/gfx942/mla/*.co).
# Point AITER_PATH at a checkout if it isn't already on sys.path.
python benchmark/bench_mla.py

python benchmark/bench_mla.py --shapes 8:150000,16:150000   # pick shapes
python benchmark/bench_mla.py -v                            # show the num_kv_splits sweep
```

The script validates both kernels against a chunked fp32 streaming-softmax reference
over the same dequantized KV before timing anything, and refuses to rank a config that
produced NaN.

See `examples/basic.py` for a small correctness check using a fp32 reference.
