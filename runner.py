#!/usr/bin/env python3
"""Benchmark CDNA3 HIP attention vs AITER. TFLOP/s = 4*B*H*S^2*D / time (QK+PV matmuls only)."""

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
HIP_LIB_PATH = ROOT / "libattention.so"
AITER_LIB_PATH = ROOT / "libattention_aiter.so"

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


def launch(lib, d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim) -> int:
    return lib.launch_attention_forward(d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim, None)


def matmul_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> float:
    return 4.0 * batch * heads * seq_len * seq_len * head_dim


def float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).copy()
    u += np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (u >> np.uint32(16)).astype(np.uint16)


def make_inputs(batch: int, heads: int, seq_len: int, head_dim: int, dtype: str):
    rng = np.random.default_rng(42)
    shape = (batch, heads, seq_len, head_dim)
    qf = rng.standard_normal(shape, dtype=np.float32)
    kf = rng.standard_normal(shape, dtype=np.float32)
    vf = rng.standard_normal(shape, dtype=np.float32)
    if dtype == "fp16":
        return qf.astype(np.float16), kf.astype(np.float16), vf.astype(np.float16)
    if dtype == "bf16":
        return float32_to_bf16_bits(qf), float32_to_bf16_bits(kf), float32_to_bf16_bits(vf)
    raise ValueError(f"Unsupported dtype: {dtype}")


def hip_alloc(hip, host_arr):
    ptr = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(ptr), host_arr.nbytes) == 0
    assert hip.hipMemcpy(ptr, host_arr.ctypes.data, host_arr.nbytes, 1) == 0
    return ptr


def copy_out(hip, d_out, shape):
    out = np.zeros(shape, dtype=np.float32)
    assert hip.hipMemcpy(out.ctypes.data, d_out, out.nbytes, 2) == 0
    return out


def benchmark_path(hip, lib_path: Path, q, k, v, batch, heads, seq_len, head_dim, warmup_iters, benchmark_iters):
    lib = load_kernel(lib_path)
    shape = q.shape
    d_q, d_k = hip_alloc(hip, q), hip_alloc(hip, k)
    d_v = hip_alloc(hip, v)
    d_out = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(d_out), int(np.prod(shape)) * 4) == 0

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

    out = copy_out(hip, d_out, shape)
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
    report=True,
):
    ensure_lib(HIP_LIB_PATH, "all")
    if hip_dtype == "bf16":
        ensure_lib(AITER_LIB_PATH, "aiter")

    hip = load_hip_runtime()
    q, k, v = make_inputs(batch, heads, seq_len, head_dim, hip_dtype)

    hip_r = benchmark_path(hip, HIP_LIB_PATH, q, k, v, batch, heads, seq_len, head_dim, warmup_iters, benchmark_iters)
    aiter_r = None
    if hip_dtype == "bf16":
        aiter_r = benchmark_path(hip, AITER_LIB_PATH, q, k, v, batch, heads, seq_len, head_dim, warmup_iters, benchmark_iters)

    ok_cmp = head_dim == 128 and seq_len >= 256
    correctness = compare_outputs(aiter_r["out"], hip_r["out"]) if aiter_r and ok_cmp else None

    if report:
        print(f"shape={hip_r['out'].shape}  dtype={hip_dtype}  flops={hip_r['flops']:.3e} (4*B*H*S^2*D)")
        print(f"HIP   {hip_r['avg_ms']:.4f} ms  {hip_r['tflops']:.1f} TFLOP/s")
        if aiter_r:
            print(f"AITER {aiter_r['avg_ms']:.4f} ms  {aiter_r['tflops']:.1f} TFLOP/s  (FMHA+bf16→fp32; non-causal, non-group)")
            print(f"ratio HIP/AITER time: {hip_r['avg_ms'] / aiter_r['avg_ms']:.2f}x")
        if correctness:
            c = correctness
            print(f"diff max_abs={c['max_abs']:.3e} rmse={c['rmse']:.3e}")
        elif aiter_r and not ok_cmp:
            print("diff: skipped (use head_dim=128, seq_len>=256)")

    return {"hip": hip_r, "aiter": aiter_r, "correctness": correctness}


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
    args = ap.parse_args()
    run(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        benchmark_iters=args.benchmark_iters,
        warmup_iters=args.warmup_iters,
        hip_dtype=args.hip_dtype,
    )
