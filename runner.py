#!/usr/bin/env python3
"""Runner for the CDNA3 attention HIP kernel."""

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

LIB_PATH = Path(__file__).parent / "libattention.so"


def build():
    print("Building kernel...")
    result = subprocess.run(["make", "-C", str(Path(__file__).parent)], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    print("Build OK")


def run(batch=2, heads=24, seq_len=8192, head_dim=128, benchmark_iters=20, report=True):
    if not LIB_PATH.exists():
        build()

    lib = ctypes.CDLL(str(LIB_PATH))
    lib.launch_attention_forward.restype = ctypes.c_int
    lib.launch_attention_forward.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p,  # K
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # Out
        ctypes.c_int,     # batch
        ctypes.c_int,     # heads
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # head_dim
        ctypes.c_void_p,  # stream (NULL = default)
    ]

    rng = np.random.default_rng(42)
    shape = (batch, heads, seq_len, head_dim)
    Q = rng.standard_normal(shape).astype(np.float16)
    K = rng.standard_normal(shape).astype(np.float16)
    V = rng.standard_normal(shape).astype(np.float16)
    Out = np.zeros(shape, dtype=np.float32)

    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        print("ERROR: Could not load libamdhip64.so — is ROCm installed?")
        sys.exit(1)

    hip.hipMalloc.restype = ctypes.c_int
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMemcpy.restype = ctypes.c_int
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipFree.restype = ctypes.c_int
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipDeviceSynchronize.restype = ctypes.c_int
    hip.hipEventCreate.restype = ctypes.c_int
    hip.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    hip.hipEventRecord.restype = ctypes.c_int
    hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    hip.hipEventSynchronize.restype = ctypes.c_int
    hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipEventElapsedTime.restype = ctypes.c_int
    hip.hipEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    hip.hipEventDestroy.restype = ctypes.c_int
    hip.hipEventDestroy.argtypes = [ctypes.c_void_p]

    HIP_MEMCPY_H2D = 1
    HIP_MEMCPY_D2H = 2
    out_nbytes = Out.nbytes

    def hip_alloc(host_arr):
        ptr = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(ptr), host_arr.nbytes) == 0
        assert hip.hipMemcpy(ptr, host_arr.ctypes.data, host_arr.nbytes, HIP_MEMCPY_H2D) == 0
        return ptr

    d_Q = hip_alloc(Q)
    d_K = hip_alloc(K)
    d_V = hip_alloc(V)
    d_Out = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(d_Out), out_nbytes) == 0

    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    assert hip.hipEventCreate(ctypes.byref(start_event)) == 0
    assert hip.hipEventCreate(ctypes.byref(stop_event)) == 0

    timed_iters = max(1, benchmark_iters)
    assert hip.hipEventRecord(start_event, None) == 0
    for _ in range(timed_iters):
        err = lib.launch_attention_forward(
            d_Q, d_K, d_V, d_Out,
            batch, heads, seq_len, head_dim,
            None,
        )
        assert err == 0, f"Kernel launch failed with error {err}"
    assert hip.hipEventRecord(stop_event, None) == 0
    assert hip.hipEventSynchronize(stop_event) == 0

    elapsed_ms = ctypes.c_float()
    assert hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, stop_event) == 0
    avg_ms = elapsed_ms.value / timed_iters
    attention_flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    tflops = attention_flops / (avg_ms * 1e-3) / 1e12

    assert hip.hipMemcpy(Out.ctypes.data, d_Out, out_nbytes, HIP_MEMCPY_D2H) == 0
    if report:
        print(f"Done. Output shape: {Out.shape}")
        print(f"Average kernel time: {avg_ms:.6f} ms over {timed_iters} iteration(s)")
        print(f"Attention throughput: {tflops:.3f} TFLOP/s")

    hip.hipEventDestroy(start_event)
    hip.hipEventDestroy(stop_event)

    for ptr in [d_Q, d_K, d_V, d_Out]:
        hip.hipFree(ptr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--benchmark-iters", type=int, default=20)
    ap.add_argument("--warmup-iters", type=int, default=0)
    args = ap.parse_args()
    for _ in range(args.warmup_iters):
        run(args.batch, args.heads, args.seq_len, args.head_dim, benchmark_iters=1, report=False)
    run(
        args.batch,
        args.heads,
        args.seq_len,
        args.head_dim,
        benchmark_iters=args.benchmark_iters,
    )
