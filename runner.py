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


def run(batch=2, heads=24, seq_len=8192, head_dim=128):
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
    Q = rng.standard_normal(shape).astype(np.float32)
    K = rng.standard_normal(shape).astype(np.float32)
    V = rng.standard_normal(shape).astype(np.float32)
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

    HIP_MEMCPY_H2D = 1
    HIP_MEMCPY_D2H = 2
    nbytes = Q.nbytes

    def hip_alloc(host_arr):
        ptr = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(ptr), nbytes) == 0
        assert hip.hipMemcpy(ptr, host_arr.ctypes.data, nbytes, HIP_MEMCPY_H2D) == 0
        return ptr

    d_Q = hip_alloc(Q)
    d_K = hip_alloc(K)
    d_V = hip_alloc(V)
    d_Out = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(d_Out), nbytes) == 0

    err = lib.launch_attention_forward(
        d_Q, d_K, d_V, d_Out,
        batch, heads, seq_len, head_dim,
        None,
    )
    assert err == 0, f"Kernel launch failed with error {err}"
    hip.hipDeviceSynchronize()

    assert hip.hipMemcpy(Out.ctypes.data, d_Out, nbytes, HIP_MEMCPY_D2H) == 0
    print(f"Done. Output shape: {Out.shape}")

    for ptr in [d_Q, d_K, d_V, d_Out]:
        hip.hipFree(ptr)


if __name__ == "__main__":
    run()
