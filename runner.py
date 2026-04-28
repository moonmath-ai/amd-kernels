#!/usr/bin/env python3
"""Benchmark CDNA3 HIP attention (RTNE + RTZ) vs AITER. TFLOP/s = 4*B*H*S^2*D / time (QK+PV matmuls only)."""

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
HIP_LIB_RTNE_PATH = ROOT / "libattention_rtne.so"
HIP_LIB_RTZ_PATH  = ROOT / "libattention_rtz.so"
AITER_LIB_PATH    = ROOT / "libattention_aiter.so"

LAUNCH_ARGS = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]


def build_target(target: str):
    print(f"Building {target}...")
    r = subprocess.run(["make", "-C", str(ROOT), target], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    print(f"Build OK: {target}")


def ensure_lib(path: Path, target: str):
    if not path.exists():
        build_target(target)


def load_hip_runtime():
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("ERROR: Could not load libamdhip64.so — is ROCm installed?")
        sys.exit(1)
    for name, restype, argtypes in [
        ("hipMalloc", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        ("hipMemcpy", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
        ("hipFree", ctypes.c_int, [ctypes.c_void_p]),
        ("hipDeviceSynchronize", ctypes.c_int, []),
        ("hipEventCreate", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
        ("hipEventRecord", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        ("hipEventSynchronize", ctypes.c_int, [ctypes.c_void_p]),
        ("hipEventElapsedTime", ctypes.c_int, [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]),
        ("hipEventDestroy", ctypes.c_int, [ctypes.c_void_p]),
    ]:
        getattr(hip, name).restype = restype
        getattr(hip, name).argtypes = argtypes
    return hip


def load_kernel(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.launch_attention_forward.restype = ctypes.c_int
    lib.launch_attention_forward.argtypes = LAUNCH_ARGS
    return lib


def set_aiter_variant(variant: str):
    """Switch the AITER wrapper to load a different HSACO variant on next launch."""
    lib = ctypes.CDLL(str(AITER_LIB_PATH))
    lib.set_aiter_variant.argtypes = [ctypes.c_char_p]
    lib.set_aiter_variant.restype = ctypes.c_int
    rc = lib.set_aiter_variant(variant.encode())
    if rc != 0:
        raise RuntimeError(f"AITER set_aiter_variant({variant!r}) failed (rc={rc})")


def launch(lib, d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim) -> int:
    return lib.launch_attention_forward(d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim, None)


def matmul_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> float:
    return 4.0 * batch * heads * seq_len * seq_len * head_dim


def float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).copy()
    u += np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (u >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint16)
    return (x.astype(np.uint32) << np.uint32(16)).view(np.float32)


def make_fp32_pattern(shape, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (Q, K, V) fp32 tensors for a named structured pattern."""
    _, _, S, D = shape
    rng = np.random.default_rng(42)
    if pattern == "random":
        qf = rng.standard_normal(shape, dtype=np.float32)
        kf = rng.standard_normal(shape, dtype=np.float32)
        vf = rng.standard_normal(shape, dtype=np.float32)
    elif pattern == "ones":
        # Q=K=V=1: softmax is uniform, output should equal V mean = 1.
        qf = np.ones(shape, dtype=np.float32)
        kf = np.ones(shape, dtype=np.float32)
        vf = np.ones(shape, dtype=np.float32)
    elif pattern == "linear":
        # Position-dependent gradient. Q,K vary along seq_len so attention is
        # non-uniform; V varies along seq+head so output is position-correlated.
        s_idx = np.arange(S, dtype=np.float32)[None, None, :, None] / S
        d_idx = np.arange(D, dtype=np.float32)[None, None, None, :] / D
        qf = np.broadcast_to(s_idx + 0.1 * d_idx, shape).astype(np.float32, copy=True)
        kf = np.broadcast_to(s_idx, shape).astype(np.float32, copy=True)
        vf = np.broadcast_to(s_idx + d_idx, shape).astype(np.float32, copy=True)
    elif pattern == "diag":
        # Q[i] = K[i] = e_{i % D}: each Q row matches one K dim strongly.
        # V[i,d] = (i / S). Tests that softmax peaks correctly along the diag.
        eye = np.eye(D, dtype=np.float32)
        qk = np.tile(eye, (S // D + 1, 1))[:S]  # (S, D)
        v_pat = (np.arange(S, dtype=np.float32) / S)[:, None] * np.ones((1, D), dtype=np.float32)
        qf = np.broadcast_to(qk[None, None], shape).astype(np.float32, copy=True)
        kf = qf.copy()
        vf = np.broadcast_to(v_pat[None, None], shape).astype(np.float32, copy=True)
    else:
        raise ValueError(f"Unknown input pattern: {pattern}")
    return qf, kf, vf


def make_inputs(batch: int, heads: int, seq_len: int, head_dim: int, dtype: str, pattern: str = "random"):
    """Return (q_bits, k_bits, v_bits, q_fp32, k_fp32, v_fp32). The fp32 views
    are bf16-quantized so the reference sees the same precision as the kernel."""
    shape = (batch, heads, seq_len, head_dim)
    qf, kf, vf = make_fp32_pattern(shape, pattern)
    if dtype == "fp16":
        return (qf.astype(np.float16), kf.astype(np.float16), vf.astype(np.float16),
                qf.astype(np.float32), kf.astype(np.float32), vf.astype(np.float32))
    if dtype == "bf16":
        qb, kb, vb = float32_to_bf16_bits(qf), float32_to_bf16_bits(kf), float32_to_bf16_bits(vf)
        return qb, kb, vb, bf16_bits_to_float32(qb), bf16_bits_to_float32(kb), bf16_bits_to_float32(vb)
    raise ValueError(f"Unsupported dtype: {dtype}")


def reference_attention_fp32(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    """Forward attention in fp32, per (b, h) slice to bound peak memory.
    Inputs and output are fp32; softmax is numerically stable (subtract row max)."""
    B, H, S, D = q.shape
    out = np.empty((B, H, S, D), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            s = (q[b, h] @ k[b, h].T) * scale
            s -= s.max(axis=1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(axis=1, keepdims=True)
            out[b, h] = s @ v[b, h]
    return out


def hip_alloc(hip, host_arr):
    ptr = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(ptr), host_arr.nbytes) == 0
    assert hip.hipMemcpy(ptr, host_arr.ctypes.data, host_arr.nbytes, 1) == 0
    return ptr


def copy_out(hip, d_out, shape, dtype: str):
    if dtype == "bf16":
        out = np.zeros(shape, dtype=np.uint16)
        assert hip.hipMemcpy(out.ctypes.data, d_out, out.nbytes, 2) == 0
        return bf16_bits_to_float32(out)
    if dtype == "fp32":
        out = np.zeros(shape, dtype=np.float32)
        assert hip.hipMemcpy(out.ctypes.data, d_out, out.nbytes, 2) == 0
        return out
    raise ValueError(f"Unsupported output dtype: {dtype}")


def benchmark_path(hip, lib_path: Path, q, k, v, batch, heads, seq_len, head_dim, warmup_iters, benchmark_iters, out_dtype: str):
    lib = load_kernel(lib_path)
    shape = q.shape
    d_q, d_k = hip_alloc(hip, q), hip_alloc(hip, k)
    d_v = hip_alloc(hip, v)
    d_out = ctypes.c_void_p()
    out_bpe = 2 if out_dtype == "bf16" else 4
    assert hip.hipMalloc(ctypes.byref(d_out), int(np.prod(shape)) * out_bpe) == 0

    for _ in range(max(0, warmup_iters)):
        assert launch(lib, d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim) == 0
    assert hip.hipDeviceSynchronize() == 0

    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipEventCreate(ctypes.byref(start)) == 0 and hip.hipEventCreate(ctypes.byref(stop)) == 0
    n = max(1, benchmark_iters)
    assert hip.hipEventRecord(start, None) == 0
    for _ in range(n):
        assert launch(lib, d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim) == 0
    assert hip.hipEventRecord(stop, None) == 0 and hip.hipEventSynchronize(stop) == 0
    ms = ctypes.c_float()
    assert hip.hipEventElapsedTime(ctypes.byref(ms), start, stop) == 0
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(stop)
    avg_ms = ms.value / n

    out = copy_out(hip, d_out, shape, out_dtype)
    for p in (d_q, d_k, d_v, d_out):
        hip.hipFree(p)

    flops = matmul_flops(batch, heads, seq_len, head_dim)
    return {"avg_ms": avg_ms, "tflops": flops / (avg_ms * 1e-3) / 1e12, "flops": flops, "out": out}


def compare_outputs(ref, test):
    diff = test - ref
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(ref), 1e-5)
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_rel": float((abs_diff / denom).max()),
        "mean_rel": float((abs_diff / denom).mean()),
    }


def run(
    batch=2,
    heads=24,
    seq_len=8192,
    head_dim=128,
    benchmark_iters=20,
    warmup_iters=0,
    hip_dtype="bf16",
    inputs="random",
    skip_ref=False,
    report=True,
):
    if hip_dtype != "bf16":
        raise ValueError("Current HIP and AITER attention kernels expect bf16 inputs and produce bf16 outputs.")

    ensure_lib(HIP_LIB_RTNE_PATH, "rtne")
    ensure_lib(HIP_LIB_RTZ_PATH,  "rtz")
    ensure_lib(AITER_LIB_PATH,    "aiter")

    hip = load_hip_runtime()
    q, k, v, qf, kf, vf = make_inputs(batch, heads, seq_len, head_dim, hip_dtype, inputs)

    bargs = (q, k, v, batch, heads, seq_len, head_dim, warmup_iters, benchmark_iters, "bf16")
    hip_rtne  = benchmark_path(hip, HIP_LIB_RTNE_PATH, *bargs)
    hip_rtz   = benchmark_path(hip, HIP_LIB_RTZ_PATH,  *bargs)
    set_aiter_variant("rtne")
    aiter_rtne = benchmark_path(hip, AITER_LIB_PATH,    *bargs)
    set_aiter_variant("rtz")
    aiter_rtz  = benchmark_path(hip, AITER_LIB_PATH,    *bargs)

    ref = None
    corr = {"hip_rtne": None, "hip_rtz": None, "aiter_rtne": None, "aiter_rtz": None}
    if not skip_ref:
        scale = 1.0 / float(np.sqrt(head_dim))
        ref = reference_attention_fp32(qf, kf, vf, scale)
        corr["hip_rtne"]   = compare_outputs(ref, hip_rtne["out"])
        corr["hip_rtz"]    = compare_outputs(ref, hip_rtz["out"])
        corr["aiter_rtne"] = compare_outputs(ref, aiter_rtne["out"])
        corr["aiter_rtz"]  = compare_outputs(ref, aiter_rtz["out"])

    if report:
        print(f"inputs={inputs}  shape={hip_rtne['out'].shape}  flops={hip_rtne['flops']:.3e} (4*B*H*S^2*D)")
        print(f"              ms      TFLOP/s    vs fp32 ref (max_abs / rmse)    same-round HIP/AITER")
        def _row(name, r, ref_corr, ratio):
            ref_str = f"{ref_corr['max_abs']:.2e} / {ref_corr['rmse']:.2e}" if ref_corr else "         -        "
            ratio_str = f"{ratio:.2f}x" if ratio is not None else ""
            print(f"{name:11s} {r['avg_ms']:.4f}   {r['tflops']:6.1f}    {ref_str:28s}    {ratio_str}")
        _row("HIP   RTNE", hip_rtne,   corr["hip_rtne"],   hip_rtne["avg_ms"] / aiter_rtne["avg_ms"])
        _row("HIP   RTZ",  hip_rtz,    corr["hip_rtz"],    hip_rtz["avg_ms"]  / aiter_rtz["avg_ms"])
        _row("AITER RTNE", aiter_rtne, corr["aiter_rtne"], None)
        _row("AITER RTZ",  aiter_rtz,  corr["aiter_rtz"],  None)
        if skip_ref:
            print("(fp32 reference skipped — drop --skip-ref to enable)")

    return {"hip_rtne": hip_rtne, "hip_rtz": hip_rtz,
            "aiter_rtne": aiter_rtne, "aiter_rtz": aiter_rtz,
            "correctness": corr, "ref": ref}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark HIP attention vs AITER")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--benchmark-iters", type=int, default=20)
    ap.add_argument("--warmup-iters", type=int, default=0)
    ap.add_argument("--hip-dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--inputs", choices=["random", "ones", "linear", "diag"], default="random",
                    help="Input pattern for Q/K/V (default: random standard-normal)")
    ap.add_argument("--skip-ref", action="store_true",
                    help="Skip the fp32 reference correctness check (faster)")
    args = ap.parse_args()
    run(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        benchmark_iters=args.benchmark_iters,
        warmup_iters=args.warmup_iters,
        hip_dtype=args.hip_dtype,
        inputs=args.inputs,
        skip_ref=args.skip_ref,
    )
