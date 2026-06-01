#!/usr/bin/env python3
"""Benchmark CDNA3 HIP attention (RTNA + RTNE + RTZ) vs AITER's flash_attn_func, with
optional Modular MAX `flash_attention_gpu` if `max` is installed.

Requires ROCm-built torch, the AITER submodule under third_party/aiter, and
`moonmath_attention` installed (`pip install -e .` builds the kernel .so).
TFLOP/s = 4*B*H*S^2*D / time (QK + PV matmuls only).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party" / "aiter"))

from aiter import flash_attn_func  # noqa: E402

import moonmath_attention as ma  # noqa: E402

# AITER's how_v3_bf16_cvt: 0=RTNE, 1=RTNA (default), 2=RTZ.
AITER_RTNE = 0
AITER_RTNA = 1
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


# --- Modular MAX: flash_attention_gpu (max.nn.kernels) ---------------------------

_MAX_FA_CACHE: dict = {}
_MAX_FA_SESSION = None


def _load_max_modules():
    """Return namespace dict for MAX `flash_attention_gpu` (or raise ImportError)."""
    from max.dtype import DType
    from max.driver import Accelerator, Buffer
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType
    from max.nn.attention.mask_config import MHAMaskVariant
    from max.nn.kernels import flash_attention_gpu
    return {
        "DType": DType, "Buffer": Buffer, "Accelerator": Accelerator,
        "InferenceSession": InferenceSession, "DeviceRef": DeviceRef,
        "Graph": Graph, "TensorType": TensorType,
        "MHAMaskVariant": MHAMaskVariant,
        "flash_attention_gpu": flash_attention_gpu,
    }


def _max_session(mod):
    global _MAX_FA_SESSION
    if _MAX_FA_SESSION is None:
        _MAX_FA_SESSION = mod["InferenceSession"](devices=[mod["Accelerator"](0)])
    return _MAX_FA_SESSION


def _max_model(B, S, H, D, mod, session):
    """Compile once per shape. NULL_MASK matches non-causal HIP/AITER."""
    key = (B, S, H, D)
    if key in _MAX_FA_CACHE:
        return _MAX_FA_CACHE[key]
    DType, DeviceRef = mod["DType"], mod["DeviceRef"]
    Graph, TensorType = mod["Graph"], mod["TensorType"]
    MHAMaskVariant = mod["MHAMaskVariant"]
    flash_attention_gpu = mod["flash_attention_gpu"]
    inp = TensorType(DType.bfloat16, (B, S, H, D), DeviceRef.GPU(0))
    scale = 1.0 / (D ** 0.5)

    def forward(q, k, v):
        return flash_attention_gpu(q, k, v, MHAMaskVariant.NULL_MASK, scale)

    graph = Graph("flash_attention_mha", forward=forward, input_types=[inp, inp, inp])
    model = session.load(graph)
    _MAX_FA_CACHE[key] = model
    return model


def _torch_bshd_to_max_buf(t_bshd, mod, acc):
    """torch.bfloat16 (B,S,H,D) GPU tensor → MAX bf16 device Buffer (B,S,H,D)."""
    DType, Buffer = mod["DType"], mod["Buffer"]
    u16 = t_bshd.cpu().contiguous().view(torch.uint16).numpy()
    return Buffer.from_numpy(u16).view(DType.bfloat16, t_bshd.shape).to(acc)


def _max_buf_to_fp32_bhsd(buf, mod):
    """MAX bf16 Buffer (B,S,H,D) → fp32 torch tensor (B,H,S,D)."""
    DType = mod["DType"]
    u16 = buf.view(DType.uint16, buf.shape).to_numpy()
    f32 = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return torch.from_numpy(np.ascontiguousarray(np.transpose(f32, (0, 2, 1, 3))))


def run(B=2, H=24, S=16384, D=128, warmup=8, iters=30, pattern="random", include_max=True):
    if not torch.cuda.is_available():
        sys.exit("Need ROCm-built torch with a HIP device.")
    device = torch.device("cuda")

    q, k, v = make_inputs(B, H, S, D, device, pattern)

    # AITER + MAX expect (B, S, H, D); transpose once outside the timed loop.
    q_a = q.transpose(1, 2).contiguous()
    k_a = k.transpose(1, 2).contiguous()
    v_a = v.transpose(1, 2).contiguous()

    flops = 4.0 * B * H * S * S * D

    # HIP: preallocate output per variant — allocation outside the timed loop;
    # the correctness section reuses each buffer's last value from the loop.
    # ma.forward runs the L1V pipeline: it pre-transposes V into V_t (an extra
    # launch_v_transpose kernel) and then the attention kernel — so the timed
    # HIP figure INCLUDES the V-transpose precompute. AITER's high-level wrapper
    # allocates + transposes V internally too, so the timing is apples-to-apples.
    out_hip_rtna = torch.empty_like(q)
    out_hip_rtne = torch.empty_like(q)
    out_hip_rtz  = torch.empty_like(q)

    aiter_last = {"rtna": None, "rtne": None, "rtz": None}
    fns = {
        "HIP   RTNA": lambda: ma.forward(q, k, v, out=out_hip_rtna, round_mode="rtna"),
        "HIP   RTNE": lambda: ma.forward(q, k, v, out=out_hip_rtne, round_mode="rtne"),
        "HIP   RTZ":  lambda: ma.forward(q, k, v, out=out_hip_rtz,  round_mode="rtz"),
        "AITER RTNA": lambda: aiter_last.__setitem__("rtna", flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTNA)),
        "AITER RTNE": lambda: aiter_last.__setitem__("rtne", flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTNE)),
        "AITER RTZ":  lambda: aiter_last.__setitem__("rtz",  flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTZ)),
    }

    max_mod = max_skip_reason = None
    bq = bk = bv = None
    if include_max:
        try:
            max_mod = _load_max_modules()
            session = _max_session(max_mod)
            model = _max_model(B, S, H, D, max_mod, session)
            acc = max_mod["Accelerator"](0)
            bq = _torch_bshd_to_max_buf(q_a, max_mod, acc)
            bk = _torch_bshd_to_max_buf(k_a, max_mod, acc)
            bv = _torch_bshd_to_max_buf(v_a, max_mod, acc)
            fns["MAX"] = lambda: model(bq, bk, bv)[0]
        except (ImportError, OSError, RuntimeError) as exc:
            max_skip_reason = f"{type(exc).__name__}: {exc}"
            max_mod = None

    timings = {name: time_fn(fn, warmup, iters) for name, fn in fns.items()}

    def diff_stats(a, b):
        a, b = a.float().cpu(), b.float().cpu()
        d = (a - b).abs()
        return d.max().item(), d.pow(2).mean().sqrt().item()

    # Buffers already hold each kernel's last output from the timing loop.
    # Re-layout AITER outputs from (B, S, H, D) to (B, H, S, D) for comparison.
    aiter_rtna_bhsd = aiter_last["rtna"].transpose(1, 2).contiguous()
    aiter_rtne_bhsd = aiter_last["rtne"].transpose(1, 2).contiguous()
    aiter_rtz_bhsd  = aiter_last["rtz"].transpose(1, 2).contiguous()
    rtna_max, rtna_rmse = diff_stats(out_hip_rtna, aiter_rtna_bhsd)
    rtne_max, rtne_rmse = diff_stats(out_hip_rtne, aiter_rtne_bhsd)
    rtz_max,  rtz_rmse  = diff_stats(out_hip_rtz,  aiter_rtz_bhsd)

    out_max_f32 = None
    if max_mod is not None:
        out_max_f32 = _max_buf_to_fp32_bhsd(fns["MAX"](), max_mod)
        max_vs_hip   = diff_stats(out_max_f32, out_hip_rtne)
        max_vs_aiter = diff_stats(out_max_f32, aiter_rtne_bhsd)

    print(f"inputs={pattern}  shape=({B},{H},{S},{D})  flops={flops:.3e} (4*B*H*S^2*D)")
    print(f"              ms      TFLOP/s    ratios")
    for name, ms in timings.items():
        tflops = flops / (ms * 1e-3) / 1e12
        if name == "HIP   RTNA":
            ratios = f"HIP/AITER {ms / timings['AITER RTNA']:.2f}x"
        elif name == "HIP   RTNE":
            ratios = f"HIP/AITER {ms / timings['AITER RTNE']:.2f}x"
        elif name == "HIP   RTZ":
            ratios = f"HIP/AITER {ms / timings['AITER RTZ']:.2f}x"
        elif name == "MAX":
            ratios = (f"HIP/MAX {timings['HIP   RTNE'] / ms:.2f}x   "
                      f"AITER/MAX {timings['AITER RTNE'] / ms:.2f}x")
        else:
            ratios = ""
        print(f"{name:11s} {ms:.4f}   {tflops:6.1f}    {ratios}")

    print(f"HIP   vs AITER (max_abs / rmse):  RTNA {rtna_max:.2e} / {rtna_rmse:.2e}   RTNE {rtne_max:.2e} / {rtne_rmse:.2e}   RTZ {rtz_max:.2e} / {rtz_rmse:.2e}")
    if out_max_f32 is not None:
        print(f"MAX   vs HIP   RTNE (max_abs / rmse):  {max_vs_hip[0]:.2e} / {max_vs_hip[1]:.2e}")
        print(f"MAX   vs AITER RTNE (max_abs / rmse):  {max_vs_aiter[0]:.2e} / {max_vs_aiter[1]:.2e}")
    elif max_skip_reason is not None:
        print(f"MAX:  skipped ({max_skip_reason})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark HIP attention vs AITER (and optional MAX)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--benchmark-iters", type=int, default=20)
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--inputs", choices=["random", "ones", "linear", "diag"], default="random")
    ap.add_argument("--no-max", action="store_true",
                    help="Do not run Modular MAX flash_attention_gpu (if installed)")
    args = ap.parse_args()
    run(args.batch, args.heads, args.seq_len, args.head_dim,
        args.warmup_iters, args.benchmark_iters, args.inputs, include_max=not args.no_max)
