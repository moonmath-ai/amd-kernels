#!/usr/bin/env python3
"""Benchmark CDNA3 HIP attention (RTNE + RTZ) vs AITER's flash_attn_func.

Requires ROCm-built torch, the AITER submodule under third_party/aiter, and
`moonmath_attention` installed (`pip install -e .` builds the kernel .so).
TFLOP/s = 4*B*H*S^2*D / time (QK + PV matmuls only).
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party" / "aiter"))

from aiter import flash_attn_func  # noqa: E402

import moonmath_attention as ma  # noqa: E402

# AITER's how_v3_bf16_cvt: 0=RTNE, 1=RTNA, 2=RTZ.
AITER_RTNE = 0
AITER_RTZ  = 2


def make_inputs(B, H, S, D, device, pattern):
    """Generate (Q, K, V) on `device` in (B, H, S, D) layout, bf16."""
    torch.manual_seed(42)
    if pattern == "random":
        return tuple(torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device) for _ in range(3))
    if pattern == "ones":
        return tuple(torch.ones(B, H, S, D, dtype=torch.bfloat16, device=device) for _ in range(3))
    if pattern == "linear":
        s_idx = (torch.arange(S, dtype=torch.float32, device=device) / S).view(1, 1, S, 1)
        d_idx = (torch.arange(D, dtype=torch.float32, device=device) / D).view(1, 1, 1, D)
        q = (s_idx + 0.1 * d_idx).expand(B, H, S, D).contiguous().bfloat16()
        k = s_idx.expand(B, H, S, D).contiguous().bfloat16()
        v = (s_idx + d_idx).expand(B, H, S, D).contiguous().bfloat16()
        return q, k, v
    if pattern == "diag":
        eye = torch.eye(D, dtype=torch.float32, device=device)
        qk = eye.repeat(S // D + 1, 1)[:S]
        v_pat = (torch.arange(S, dtype=torch.float32, device=device) / S).unsqueeze(1).expand(S, D)
        q = qk.expand(B, H, S, D).contiguous().bfloat16()
        k = q.clone()
        v = v_pat.expand(B, H, S, D).contiguous().bfloat16()
        return q, k, v
    raise ValueError(f"Unknown input pattern: {pattern}")


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iters


def run(B=2, H=24, S=8192, D=128, warmup=8, iters=30, pattern="random"):
    if not torch.cuda.is_available():
        sys.exit("Need ROCm-built torch with a HIP device.")
    device = torch.device("cuda")

    q, k, v = make_inputs(B, H, S, D, device, pattern)

    # AITER's flash_attn_func expects (B, S, H, D); transpose once outside the timed loop.
    q_a = q.transpose(1, 2).contiguous()
    k_a = k.transpose(1, 2).contiguous()
    v_a = v.transpose(1, 2).contiguous()

    flops = 4.0 * B * H * S * S * D

    fns = {
        "HIP   RTNE": lambda: ma.forward(q, k, v, round_mode="rtne"),
        "HIP   RTZ":  lambda: ma.forward(q, k, v, round_mode="rtz"),
        "AITER RTNE": lambda: flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTNE),
        "AITER RTZ":  lambda: flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTZ),
    }
    timings = {name: time_fn(fn, warmup, iters) for name, fn in fns.items()}

    # One-shot correctness check: HIP vs AITER at the same rounding.
    def diff_stats(a, b):
        d = (a.float() - b.float()).abs()
        return d.max().item(), d.pow(2).mean().sqrt().item()

    out_hip_rtne = fns["HIP   RTNE"]()
    out_hip_rtz  = fns["HIP   RTZ"]()
    out_aiter_rtne = fns["AITER RTNE"]().transpose(1, 2).contiguous()  # (B,S,H,D) → (B,H,S,D)
    out_aiter_rtz  = fns["AITER RTZ"]().transpose(1, 2).contiguous()
    rtne_max, rtne_rmse = diff_stats(out_hip_rtne, out_aiter_rtne)
    rtz_max,  rtz_rmse  = diff_stats(out_hip_rtz,  out_aiter_rtz)

    print(f"inputs={pattern}  shape=({B},{H},{S},{D})  flops={flops:.3e} (4*B*H*S^2*D)")
    print(f"              ms      TFLOP/s    HIP/AITER (same round)")
    for name, ms in timings.items():
        tflops = flops / (ms * 1e-3) / 1e12
        if name == "HIP   RTNE":
            ratio = f"{ms / timings['AITER RTNE']:.2f}x"
        elif name == "HIP   RTZ":
            ratio = f"{ms / timings['AITER RTZ']:.2f}x"
        else:
            ratio = ""
        print(f"{name:11s} {ms:.4f}   {tflops:6.1f}    {ratio}")
    print(f"HIP vs AITER (max_abs / rmse):  RTNE {rtne_max:.2e} / {rtne_rmse:.2e}   RTZ {rtz_max:.2e} / {rtz_rmse:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark HIP attention vs AITER")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--benchmark-iters", type=int, default=20)
    ap.add_argument("--warmup-iters", type=int, default=0)
    ap.add_argument("--inputs", choices=["random", "ones", "linear", "diag"], default="random")
    args = ap.parse_args()
    run(args.batch, args.heads, args.seq_len, args.head_dim,
        args.warmup_iters, args.benchmark_iters, args.inputs)
