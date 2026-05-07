#!/usr/bin/env python3
"""Runner for lds_queue_test.

Usage:
    python runner.py [--variant 0|1] [--n-blocks N] [--n-iters N] [--warmup-iters N]

variant 0: ds_read_b128 (16 B/lane, 8 cy port floor)
variant 1: ds_read_b64  (8 B/lane,  4 cy port floor)
"""
import argparse
import ctypes
from pathlib import Path

import torch

ROOT = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--variant",      type=int, default=0, choices=[0, 1],
                help="0=b32 (4 cy floor, ZERO conflicts), 1=b128 (8 cy floor, ZERO conflicts)")
ap.add_argument("--n-blocks",     type=int, default=304, help="grid CTAs (1 per CU on MI300X)")
ap.add_argument("--n-iters",      type=int, default=8096, help="loop iters per CTA (work amount). Larger = longer trace, better ATT samples.")
ap.add_argument("--warmup-iters", type=int, default=2)
ap.add_argument("--bench-iters",  type=int, default=10)
args = ap.parse_args()

so = ROOT / "liblds_queue_test.so"
lib = ctypes.CDLL(str(so))
lib.launch_lds_queue_test.restype = ctypes.c_int
lib.launch_lds_queue_test.argtypes = [
    ctypes.c_void_p,  # d_in
    ctypes.c_void_p,  # d_out
    ctypes.c_int,     # n_blocks
    ctypes.c_int,     # n_iters
    ctypes.c_int,     # variant
    ctypes.c_void_p,  # stream
]

device = torch.device("cuda")
torch.manual_seed(42)
BLOCK = 64 * 8  # kBlockSize
in_buf  = torch.randint(0, 2**31 - 1, (args.n_blocks * BLOCK,), dtype=torch.int32, device=device)
out_buf = torch.empty_like(in_buf)
stream = torch.cuda.current_stream(device).cuda_stream


def launch():
    lib.launch_lds_queue_test(
        in_buf.data_ptr(), out_buf.data_ptr(),
        args.n_blocks, args.n_iters, args.variant,
        ctypes.c_void_p(stream))


# Warmup
for _ in range(args.warmup_iters):
    launch()
torch.cuda.synchronize()

# Bench
start = torch.cuda.Event(enable_timing=True)
stop  = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(args.bench_iters):
    launch()
stop.record()
stop.synchronize()
ms = start.elapsed_time(stop) / args.bench_iters

# Profile dispatch (always 1 final dispatch, what ATT captures last)
launch()
torch.cuda.synchronize()

variant_name = ["b32 (8B/lane, 4 cy floor)", "b128 (16B/lane, 8 cy floor)"][args.variant]
print(f"variant={args.variant} ({variant_name})  blocks={args.n_blocks}  iters={args.n_iters}  ms/dispatch={ms:.3f}")
