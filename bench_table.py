#!/usr/bin/env python3
"""Multi-shape comparison: HIP attention vs AITER v3 vs Modular MAX.

Produces a markdown table suitable for publication. Each shape is benchmarked
in TFLOP/s for both RTNE and RTZ rounding (Mojo MAX has no rounding selector;
it's listed once per shape).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party" / "aiter"))

from aiter import flash_attn_func  # noqa: E402
import moonmath_attention as ma     # noqa: E402

AITER_RTNE = 0
AITER_RTZ  = 2

# Shapes: (label, B, H, S, D). All D=128 since AITER v3 is hd128-only.
SHAPES = [
    ("small",     2, 16,  2048, 128),
    ("medium",    2, 24,  4096, 128),
    ("std",       2, 24,  8192, 128),
    ("long",      2, 24, 16384, 128),
    ("xlong",     2, 24, 32768, 128),
    ("wide",      1, 32,  8192, 128),
    ("wide-long", 1, 32, 16384, 128),
    ("batch",     4, 16,  4096, 128),
    ("batch-long",4, 16, 16384, 128),
    ("70B",       1, 64, 16384, 128),
]


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def time_fn_passes(fn, warmup, iters, passes):
    """Run `passes` independent timing runs and return the *median* ms-per-call.
    Median is robust against thermal / power-state outliers."""
    samples = [time_fn(fn, warmup, iters) for _ in range(passes)]
    samples.sort()
    return samples[len(samples) // 2]


def make_qkv(B, H, S, D, device):
    torch.manual_seed(42)
    return tuple(torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device) for _ in range(3))


# ---- Modular MAX flash_attention_gpu ------------------------------------------
_MAX_MOD = None
_MAX_SESSION = None
_MAX_CACHE = {}


def _load_max():
    global _MAX_MOD
    if _MAX_MOD is not None:
        return _MAX_MOD
    from max.dtype import DType
    from max.driver import Accelerator, Buffer
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType
    from max.nn.attention.mask_config import MHAMaskVariant
    from max.nn.kernels import flash_attention_gpu
    _MAX_MOD = dict(
        DType=DType, Buffer=Buffer, Accelerator=Accelerator,
        InferenceSession=InferenceSession, DeviceRef=DeviceRef,
        Graph=Graph, TensorType=TensorType,
        MHAMaskVariant=MHAMaskVariant, flash_attention_gpu=flash_attention_gpu,
    )
    return _MAX_MOD


def _max_session(mod):
    global _MAX_SESSION
    if _MAX_SESSION is None:
        _MAX_SESSION = mod["InferenceSession"](devices=[mod["Accelerator"](0)])
    return _MAX_SESSION


def _max_model(B, S, H, D):
    key = (B, S, H, D)
    if key in _MAX_CACHE:
        return _MAX_CACHE[key]
    mod = _load_max()
    sess = _max_session(mod)
    DType, DeviceRef = mod["DType"], mod["DeviceRef"]
    Graph, TensorType = mod["Graph"], mod["TensorType"]
    MHAMaskVariant = mod["MHAMaskVariant"]
    flash_attention_gpu = mod["flash_attention_gpu"]
    inp = TensorType(DType.bfloat16, (B, S, H, D), DeviceRef.GPU(0))
    scale = 1.0 / (D ** 0.5)

    def forward(q, k, v):
        return flash_attention_gpu(q, k, v, MHAMaskVariant.NULL_MASK, scale)

    g = Graph("flash_attention_mha", forward=forward, input_types=[inp, inp, inp])
    model = sess.load(g)
    _MAX_CACHE[key] = model
    return model


def _torch_to_max_buf(t_bshd, mod, acc):
    DType, Buffer = mod["DType"], mod["Buffer"]
    u16 = t_bshd.cpu().contiguous().view(torch.uint16).numpy()
    return Buffer.from_numpy(u16).view(DType.bfloat16, t_bshd.shape).to(acc)


def diff_stats(out_bhsd, ref_bhsd):
    """Return (max_abs, rmse) of `out` vs `ref` in fp32 (both already in
    BHSD layout)."""
    a = out_bhsd.float().cpu()
    b = ref_bhsd.float().cpu()
    d = (a - b).abs()
    return d.max().item(), d.pow(2).mean().sqrt().item()


warmup_passes = 3  # number of timing-pass repeats; CLI overrides

def bench_shape(B, H, S, D, warmup, iters):
    if not torch.cuda.is_available():
        sys.exit("Need ROCm-built torch")
    device = torch.device("cuda")
    q, k, v = make_qkv(B, H, S, D, device)
    q_a = q.transpose(1, 2).contiguous()
    k_a = k.transpose(1, 2).contiguous()
    v_a = v.transpose(1, 2).contiguous()

    out_rtne = torch.empty_like(q)
    out_rtz  = torch.empty_like(q)
    sink = {}

    fns = {
        "hip_rtne":   lambda: ma.forward(q, k, v, out=out_rtne, round_mode="rtne"),
        "hip_rtz":    lambda: ma.forward(q, k, v, out=out_rtz,  round_mode="rtz"),
        "aiter_rtne": lambda: sink.__setitem__("a_rtne",
                          flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTNE)),
        "aiter_rtz":  lambda: sink.__setitem__("a_rtz",
                          flash_attn_func(q_a, k_a, v_a, causal=False, how_v3_bf16_cvt=AITER_RTZ)),
    }

    has_max = False
    try:
        mod = _load_max()
        _ = _max_session(mod)
        model = _max_model(B, S, H, D)
        acc = mod["Accelerator"](0)
        bq = _torch_to_max_buf(q_a, mod, acc)
        bk = _torch_to_max_buf(k_a, mod, acc)
        bv = _torch_to_max_buf(v_a, mod, acc)
        fns["max"] = lambda: model(bq, bk, bv)[0]
        has_max = True
    except Exception as exc:
        print(f"# MAX skipped for ({B},{H},{S},{D}): {type(exc).__name__}: {exc}", file=sys.stderr)

    passes = warmup_passes  # populated by caller via globals
    timings = {name: time_fn_passes(fn, warmup, iters, passes) for name, fn in fns.items()}
    flops = 4.0 * B * H * S * S * D
    tf = {name: flops / (ms * 1e-3) / 1e12 for name, ms in timings.items()}

    # Accuracy vs AITER reference (AITER's matched-rounding output is the
    # comparand). Same-rounding pairs: ours-RTNE vs AITER-RTNE,
    # ours-RTZ vs AITER-RTZ. MAX has no rounding selector → compared to
    # AITER-RTNE only.
    aiter_rtne_bhsd = sink["a_rtne"].transpose(1, 2).contiguous()
    aiter_rtz_bhsd  = sink["a_rtz" ].transpose(1, 2).contiguous()
    err = {}
    err["hip_rtne"] = diff_stats(out_rtne, aiter_rtne_bhsd)
    err["hip_rtz"]  = diff_stats(out_rtz,  aiter_rtz_bhsd)
    if has_max:
        DType = mod["DType"]
        out_buf = fns["max"]()
        u16 = out_buf.view(DType.uint16, out_buf.shape).to_numpy()
        f32 = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        max_t = torch.from_numpy(np.ascontiguousarray(np.transpose(f32, (0, 2, 1, 3))))
        err["max"] = diff_stats(max_t, aiter_rtne_bhsd)
    return timings, tf, err


def fmt_md(rows):
    """Format as a clean markdown table in ms (lower is better). Per-row
    winner in bold; speedup multipliers (= other_ms / ours_ms, so >1× means
    ours is faster) shown alongside."""
    cols = ["Shape (B, H, S, D)", "Round", "Ours (ms)", "AITER v3 (ms)",
            "Speedup vs AITER", "Modular MAX (ms)", "Speedup vs MAX"]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for shape, rnd, ours_v, aiter_v, max_v in rows:
        vals = {"Ours": ours_v, "AITER v3": aiter_v, "Modular MAX": max_v}
        nums = {k: float(v) for k, v in vals.items() if v != "—"}
        # Lower ms is better, so winner is *min*.
        best = min(nums, key=nums.get) if nums else None

        def cell(label, raw):
            if raw == "—":
                return "—"
            return f"**{raw}**" if label == best else raw

        ours_f = float(ours_v)
        aiter_f = float(aiter_v)
        max_f = float(max_v) if max_v != "—" else None
        # Speedup = other_ms / ours_ms (>1 means ours is faster).
        ratio_a = f"{aiter_f / ours_f:.2f}×"
        ratio_m = f"{max_f / ours_f:.2f}×" if max_f else "—"

        out.append(f"| {shape} | {rnd} | {cell('Ours', ours_v)} | "
                   f"{cell('AITER v3', aiter_v)} | {ratio_a} | "
                   f"{cell('Modular MAX', max_v)} | {ratio_m} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-iters", type=int, default=8)
    ap.add_argument("--benchmark-iters", type=int, default=30)
    ap.add_argument("--passes", type=int, default=5,
                    help="Number of independent timing passes to take median over.")
    args = ap.parse_args()
    global warmup_passes
    warmup_passes = args.passes

    rows = []
    acc_rows = []
    raw = []
    for label, B, H, S, D in SHAPES:
        try:
            timings, tf, err = bench_shape(B, H, S, D, args.warmup_iters, args.benchmark_iters)
        except Exception as e:
            print(f"# Shape ({B},{H},{S},{D}) failed: {e}", file=sys.stderr)
            continue
        raw.append((B, H, S, D, timings))
        shape_str = f"({B}, {H}, {S}, {D})"
        max_str = f"{timings['max']:.3f}" if "max" in timings else "—"
        rows.append([shape_str, "RTNE", f"{timings['hip_rtne']:.3f}",
                     f"{timings['aiter_rtne']:.3f}", max_str])
        rows.append([shape_str, "RTZ",  f"{timings['hip_rtz']:.3f}",
                     f"{timings['aiter_rtz']:.3f}", max_str])

        def fmt_err(name):
            if name not in err:
                return "—"
            mx, rm = err[name]
            return f"{mx:.2e} / {rm:.2e}"

        acc_rows.append([shape_str, "RTNE", fmt_err("hip_rtne"), fmt_err("max")])
        acc_rows.append([shape_str, "RTZ",  fmt_err("hip_rtz"),  "—"])

    print(f"**Forward attention runtime (ms per call, lower is better; "
          f"median of {warmup_passes} passes × {args.benchmark_iters} iters)**")
    print()
    print("Hardware: AMD MI300X (gfx942, 304 CUs). bf16 inputs / outputs, head_dim = 128.")
    print("Modular MAX `flash_attention_gpu` is rounding-mode-free and uses RTNE internally")
    print("(verified empirically); listed once per shape.")
    print()
    print(fmt_md(rows))
    print()

    # Accuracy table — AITER as the reference.
    print("**Numerical accuracy vs AITER v3 reference (max\\_abs / rmse, lower is better)**")
    print()
    print("Reference: matched-rounding AITER v3 output. Modular MAX is rounding-mode-free")
    print("and is compared against AITER RTNE only.")
    print()
    cols = ["Shape (B, H, S, D)", "Round", "Ours vs AITER", "Modular MAX vs AITER RTNE"]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in acc_rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    print("\n".join(out))
    print()

    # Headline summary — geomean speedup vs each backend over all shapes.
    if raw:
        def gmean_speedup(other_key, ours_key):
            xs = []
            for _, _, _, _, t in raw:
                if other_key in t and ours_key in t:
                    xs.append(t[other_key] / t[ours_key])
            return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")

        sp_aiter_rtne = gmean_speedup("aiter_rtne", "hip_rtne")
        sp_aiter_rtz  = gmean_speedup("aiter_rtz",  "hip_rtz")
        sp_max_rtne   = gmean_speedup("max",        "hip_rtne")
        sp_max_rtz    = gmean_speedup("max",        "hip_rtz")
        print("_Geomean speedup across shapes (=other_ms / ours_ms, >1× = ours faster)_")
        print(f"_RTNE — ours: **{sp_aiter_rtne:.2f}×** vs AITER, **{sp_max_rtne:.2f}×** vs MAX._")
        print(f"_RTZ  — ours: **{sp_aiter_rtz:.2f}×** vs AITER, **{sp_max_rtz:.2f}×** vs MAX._")


if __name__ == "__main__":
    main()
