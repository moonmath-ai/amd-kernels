"""ctypes wrapper that runs the .hip kernel from torch."""
import ctypes
from pathlib import Path

import torch

_PKG = Path(__file__).parent
_LAUNCH_ARGS = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
# launch_attention_forward_lite(q,k,v,out, B,H,S,D, read_list,write_list, thr, phase, stream)
_LITE_LAUNCH_ARGS = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
]
# launch_v_transpose(V, V_t, seq_len_total, seq_len_per_head, stream)
_VTRANSPOSE_ARGS = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
_HTOD, _DTOH = 1, 2
_hip = None
_libs: dict[str, ctypes.CDLL] = {}


def _hip_runtime():
    global _hip
    if _hip is not None:
        return _hip
    h = ctypes.CDLL("libamdhip64.so")
    for name, restype, argtypes in (
        ("hipMalloc", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        ("hipMemcpy", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
        ("hipFree", ctypes.c_int, [ctypes.c_void_p]),
        ("hipDeviceSynchronize", ctypes.c_int, []),
    ):
        fn = getattr(h, name); fn.restype, fn.argtypes = restype, argtypes
    _hip = h
    return h


def _kernel(round_mode):
    if round_mode in _libs:
        return _libs[round_mode]
    so = _PKG / f"libattention_{round_mode}.so"
    if not so.exists():
        raise RuntimeError(f"{so.name} missing — reinstall with `pip install -e .`")
    lib = ctypes.CDLL(str(so))
    lib.launch_attention_forward.restype = ctypes.c_int
    lib.launch_attention_forward.argtypes = _LAUNCH_ARGS
    lib.launch_v_transpose.restype = ctypes.c_int
    lib.launch_v_transpose.argtypes = _VTRANSPOSE_ARGS
    _libs[round_mode] = lib
    return lib


def _lite_kernel(round_mode):
    """Load the LiteAttention-capable .so (dense `launch_attention_forward` +
    `launch_attention_forward_lite`). Built from attention_kernel.hip (templated
    attention_forward<kLite>); its dense path is bit-exact to the champion
    across RTNA/RTNE/RTZ."""
    key = f"lite_{round_mode}"
    if key in _libs:
        return _libs[key]
    so = _PKG / f"libattention_lite_{round_mode}.so"
    if not so.exists():
        raise RuntimeError(f"{so.name} missing — reinstall with `pip install -e .`")
    lib = ctypes.CDLL(str(so))
    lib.launch_attention_forward.restype = ctypes.c_int
    lib.launch_attention_forward.argtypes = _LAUNCH_ARGS
    lib.launch_attention_forward_lite.restype = ctypes.c_int
    lib.launch_attention_forward_lite.argtypes = _LITE_LAUNCH_ARGS
    lib.launch_v_transpose.restype = ctypes.c_int
    lib.launch_v_transpose.argtypes = _VTRANSPOSE_ARGS
    _libs[key] = lib
    return lib


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
            out: torch.Tensor | None = None,
            round_mode: str = "rtna") -> torch.Tensor:
    """Fused forward attention: O = softmax(QKᵀ / √D) V on MI300X.

    Args:
        q, k, v: torch.bfloat16 tensors of shape (batch, heads, seq_len, head_dim).
                 Contiguous. head_dim=128, seq_len % 64 == 0.
                 CPU tensors are copied to the AMD GPU under the hood; ROCm-built
                 torch tensors on a CUDA/HIP device are used in place.
        out: optional preallocated output tensor. Must match q's shape, dtype,
             device, and be contiguous. If None, a new tensor is allocated.
        round_mode: "rtna" (default; AITER's default, ties-away-from-zero),
            "rtne" (~0.5 ULP error), or "rtz" (cheaper; ~1 ULP, truncation).

    Returns:
        torch.bfloat16 tensor of the same shape and device as the inputs.
    """
    if round_mode not in ("rtna", "rtne", "rtz"):
        raise ValueError(f"round_mode must be 'rtna', 'rtne', or 'rtz' (got {round_mode!r})")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor; got {type(t).__name__}")
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16; got dtype={t.dtype}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D (batch, heads, seq_len, head_dim); got shape={tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v shapes must match; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q/k/v must be on the same device")
    B, H, S, D = q.shape
    if D != 128:
        raise ValueError(f"head_dim must be 128 (got {D})")
    if S % 64 != 0:
        raise ValueError(f"seq_len must be a multiple of 64 (got {S})")

    hip = _hip_runtime()
    lib = _kernel(round_mode)
    if out is None:
        out = torch.empty_like(q)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"out must be torch.Tensor; got {type(out).__name__}")
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out must be torch.bfloat16; got dtype={out.dtype}")
        if tuple(out.shape) != tuple(q.shape):
            raise ValueError(f"out shape must match q; got {tuple(out.shape)} vs {tuple(q.shape)}")
        if out.device != q.device:
            raise ValueError(f"out must be on the same device as q; got {out.device} vs {q.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    if q.device.type == "cpu":
        # bf16 → uint16 view + .numpy() share storage with the torch tensors,
        # so the kernel reads/writes their CPU memory directly.
        q_u, k_u, v_u, o_u = (t.view(torch.uint16).numpy() for t in (q, k, v, out))
        nbytes = q_u.nbytes
        d_q, d_k, d_v, d_out, d_vt = (ctypes.c_void_p() for _ in range(5))
        try:
            for ptr, src in ((d_q, q_u), (d_k, k_u), (d_v, v_u)):
                if hip.hipMalloc(ctypes.byref(ptr), nbytes) != 0:
                    raise RuntimeError("hipMalloc failed")
                if hip.hipMemcpy(ptr, src.ctypes.data, nbytes, _HTOD) != 0:
                    raise RuntimeError("hipMemcpy H→D failed")
            if hip.hipMalloc(ctypes.byref(d_out), nbytes) != 0:
                raise RuntimeError("hipMalloc(out) failed")
            # L1V kernel: pre-transpose V into V_t before the attention launch.
            if hip.hipMalloc(ctypes.byref(d_vt), nbytes) != 0:
                raise RuntimeError("hipMalloc(v_t) failed")
            rc = lib.launch_v_transpose(d_v, d_vt, B * H * S, S, None)
            if rc != 0:
                raise RuntimeError(f"launch_v_transpose returned {rc}")
            rc = lib.launch_attention_forward(d_q, d_k, d_vt, d_out, B, H, S, D, None)
            if rc != 0:
                raise RuntimeError(f"launch_attention_forward returned {rc}")
            if hip.hipDeviceSynchronize() != 0:
                raise RuntimeError("hipDeviceSynchronize failed")
            if hip.hipMemcpy(o_u.ctypes.data, d_out, nbytes, _DTOH) != 0:
                raise RuntimeError("hipMemcpy D→H failed")
        finally:
            for ptr in (d_q, d_k, d_v, d_out, d_vt):
                if ptr.value:
                    hip.hipFree(ptr)
    elif q.device.type in ("cuda", "hip"):
        # L1V kernel: V must be pre-transposed into V_t (pv_phase consumes V_t
        # directly from L1). launch_v_transpose runs first, then the attention
        # kernel — both are issued here so a caller timing forward() includes
        # the V-transpose cost.
        v_t = torch.empty_like(v)
        rc = lib.launch_v_transpose(
            ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(v_t.data_ptr()),
            B * H * S, S, None,
        )
        if rc != 0:
            raise RuntimeError(f"launch_v_transpose returned {rc}")
        rc = lib.launch_attention_forward(
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(v_t.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            B, H, S, D, None,
        )
        if rc != 0:
            raise RuntimeError(f"launch_attention_forward returned {rc}")
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("hipDeviceSynchronize failed")
    else:
        raise NotImplementedError(f"Unsupported torch device {q.device!r}; expected 'cpu' or 'cuda'/'hip'")

    return out


def forward_lite(q, k, v, read_list, write_list, *, threshold, phase,
                 out=None, round_mode="rtna"):
    """LiteAttention forward: skip K-blocks per the cross-timestep skip list.

    `read_list` / `write_list` are int16 CUDA tensors shaped
    [B, H, qtiles, nblocks+2] (the per-phase slices of a [2, ...] double buffer):
    read_list = skip_list[phase], write_list = skip_list[1-phase]. The kernel
    processes only the K-blocks named in read_list and writes, for the next
    timestep, the blocks whose max score stayed within `threshold` (log2) of the
    running max for ALL 256 q-rows of the CTA. tile = (kBlockM=256, kBlockN=64).

    GPU tensors only (the diffusion use case). Output is an approximation of full
    attention (skipped blocks contribute 0) — exact only with a compute-all list.
    """
    if round_mode not in ("rtna", "rtne", "rtz"):
        raise ValueError(f"round_mode must be 'rtna', 'rtne', or 'rtz' (got {round_mode!r})")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous 4-D bfloat16")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q/k/v shapes must match")
    if q.device.type not in ("cuda", "hip"):
        raise NotImplementedError("forward_lite requires a CUDA/HIP tensor")
    for nm, t in (("read_list", read_list), ("write_list", write_list)):
        if t.dtype != torch.int16 or not t.is_contiguous() or t.device != q.device:
            raise ValueError(f"{nm} must be a contiguous int16 tensor on q's device")
    B, H, S, D = q.shape
    if D != 128:
        raise ValueError(f"head_dim must be 128 (got {D})")
    if S % 64 != 0:
        raise ValueError(f"seq_len must be a multiple of 64 (got {S})")

    hip = _hip_runtime()
    lib = _lite_kernel(round_mode)
    if out is None:
        out = torch.empty_like(q)

    v_t = torch.empty_like(v)
    rc = lib.launch_v_transpose(
        ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(v_t.data_ptr()),
        B * H * S, S, None)
    if rc != 0:
        raise RuntimeError(f"launch_v_transpose returned {rc}")
    rc = lib.launch_attention_forward_lite(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v_t.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        B, H, S, D,
        ctypes.c_void_p(read_list.data_ptr()), ctypes.c_void_p(write_list.data_ptr()),
        ctypes.c_float(float(threshold)), ctypes.c_int(int(phase)), None)
    if rc != 0:
        raise RuntimeError(f"launch_attention_forward_lite returned {rc}")
    if hip.hipDeviceSynchronize() != 0:
        raise RuntimeError("hipDeviceSynchronize failed")
    return out
