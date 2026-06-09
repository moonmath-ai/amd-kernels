"""ctypes wrapper that runs the .hip kernel from torch."""
import ctypes
from pathlib import Path

import torch

_PKG = Path(__file__).parent
# launch_attention_forward(q,k,v,out, B,H,Sq,Skv,D, layout, stream)  [Skv = K/V len; layout 0=BHSD 1=BSHD]
_LAUNCH_ARGS = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
# launch_attention_forward_lite(q,k,v,out, B,H,Sq,Skv,D, read_list,write_list,must_do_list, thr, phase, layout, stream)
_LITE_LAUNCH_ARGS = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
# launch_v_transpose(V, V_t, seq_len_total, seq_len_per_head, heads, layout, stream)
_VTRANSPOSE_ARGS = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
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
            round_mode: str = "rtna",
            layout: str = "bhsd") -> torch.Tensor:
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
    if layout not in ("bhsd", "bshd"):
        raise ValueError(f"layout must be 'bhsd' or 'bshd' (got {layout!r})")
    bshd = (layout == "bshd")
    layout_int = 1 if bshd else 0
    # Axis positions for (seq, heads) differ by layout. BHSD: (B,H,S,D); BSHD: (B,S,H,D).
    sax, hax = (1, 2) if bshd else (2, 1)
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor; got {type(t).__name__}")
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16; got dtype={t.dtype}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D ({layout.upper()}); got shape={tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    # Cross-attention: k/v may have a different seq_len than q. They must agree with
    # each other and share (batch, heads, head_dim) with q.
    if k.shape != v.shape:
        raise ValueError(f"k/v shapes must match; got {tuple(k.shape)}, {tuple(v.shape)}")
    if (q.shape[0], q.shape[hax], q.shape[3]) != (k.shape[0], k.shape[hax], k.shape[3]):
        raise ValueError(f"q/k must share (batch, heads, head_dim); got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q/k/v must be on the same device")
    B, H, Sq, D = q.shape[0], q.shape[hax], q.shape[sax], q.shape[3]
    Skv = k.shape[sax]
    if D != 128:
        raise ValueError(f"head_dim must be 128 (got {D})")
    # Sq and Skv may be any positive values: the kernel masks the partial last q-tile (store)
    # and the partial last K/V block (score); V_t is per-head padded to a 64-row boundary.
    if Sq <= 0 or Skv <= 0:
        raise ValueError(f"seq lens must be positive (got Sq={Sq}, Skv={Skv})")
    Skv_pad = ((Skv + 63) // 64) * 64    # per-head V_t padding (block-aligned)

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
        vt_nbytes = B * H * Skv_pad * D * 2     # per-head padded V_t (bf16)
        d_q, d_k, d_v, d_out, d_vt = (ctypes.c_void_p() for _ in range(5))
        try:
            for ptr, src in ((d_q, q_u), (d_k, k_u), (d_v, v_u)):   # q is Sq rows; k/v are Skv rows
                if hip.hipMalloc(ctypes.byref(ptr), src.nbytes) != 0:
                    raise RuntimeError("hipMalloc failed")
                if hip.hipMemcpy(ptr, src.ctypes.data, src.nbytes, _HTOD) != 0:
                    raise RuntimeError("hipMemcpy H→D failed")
            if hip.hipMalloc(ctypes.byref(d_out), o_u.nbytes) != 0:
                raise RuntimeError("hipMalloc(out) failed")
            # L1V kernel: pre-transpose V into V_t before the attention launch.
            # V_t is per-head padded to ceil(Skv/64)*64 rows, so it can be larger than V.
            if hip.hipMalloc(ctypes.byref(d_vt), vt_nbytes) != 0:
                raise RuntimeError("hipMalloc(v_t) failed")
            rc = lib.launch_v_transpose(d_v, d_vt, B * H * Skv, Skv, H, layout_int, None)
            if rc != 0:
                raise RuntimeError(f"launch_v_transpose returned {rc}")
            rc = lib.launch_attention_forward(d_q, d_k, d_vt, d_out, B, H, Sq, Skv, D, layout_int, None)
            if rc != 0:
                raise RuntimeError(f"launch_attention_forward returned {rc}")
            if hip.hipDeviceSynchronize() != 0:
                raise RuntimeError("hipDeviceSynchronize failed")
            if hip.hipMemcpy(o_u.ctypes.data, d_out, o_u.nbytes, _DTOH) != 0:
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
        v_t = torch.empty((B, H, Skv_pad, D), dtype=v.dtype, device=v.device)  # per-head padded V_t (internal layout)
        rc = lib.launch_v_transpose(
            ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(v_t.data_ptr()),
            B * H * Skv, Skv, H, layout_int, None,
        )
        if rc != 0:
            raise RuntimeError(f"launch_v_transpose returned {rc}")
        rc = lib.launch_attention_forward(
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(v_t.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            B, H, Sq, Skv, D, layout_int, None,
        )
        if rc != 0:
            raise RuntimeError(f"launch_attention_forward returned {rc}")
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("hipDeviceSynchronize failed")
    else:
        raise NotImplementedError(f"Unsupported torch device {q.device!r}; expected 'cpu' or 'cuda'/'hip'")

    return out


def forward_lite(q, k, v, read_list, write_list, *, threshold, phase,
                 must_do_list=None, out=None, round_mode="rtna"):
    """LiteAttention forward: skip K-blocks per the cross-timestep skip list.

    `read_list` / `write_list` are int16 CUDA tensors shaped
    [B, H, qtiles, nblocks+2] (the per-phase slices of a [2, ...] double buffer):
    read_list = skip_list[phase], write_list = skip_list[1-phase]. The kernel
    processes only the K-blocks named in read_list and writes, for the next
    timestep, the blocks whose max score stayed within `threshold` (log2) of the
    running max for ALL 256 q-rows of the CTA. tile = (kBlockM=256, kBlockN=64).

    `must_do_list` (optional): a single GLOBAL int16 1-D tensor in the format
    [len, tile_start0, tile_end0, tile_start1, tile_end1, ...] (end-exclusive
    K-block tile indices). Blocks in any [start, end) range are FORCED to be
    recorded as "compute" in write_list regardless of the skip vote — pinning
    them in every timestep's read list (e.g. attention sinks / always-attend
    blocks). None ⇒ no forcing (behaviour identical to before this arg existed).

    GPU tensors only (the diffusion use case). Output is an approximation of full
    attention (skipped blocks contribute 0) — exact only with a compute-all list.
    """
    if round_mode not in ("rtna", "rtne", "rtz"):
        raise ValueError(f"round_mode must be 'rtna', 'rtne', or 'rtz' (got {round_mode!r})")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous 4-D bfloat16")
    if k.shape != v.shape:
        raise ValueError(f"k/v shapes must match; got {tuple(k.shape)}, {tuple(v.shape)}")
    if (q.shape[0], q.shape[1], q.shape[3]) != (k.shape[0], k.shape[1], k.shape[3]):
        raise ValueError(f"q/k must share (batch, heads, head_dim); got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.device.type not in ("cuda", "hip"):
        raise NotImplementedError("forward_lite requires a CUDA/HIP tensor")
    for nm, t in (("read_list", read_list), ("write_list", write_list)):
        if t.dtype != torch.int16 or not t.is_contiguous() or t.device != q.device:
            raise ValueError(f"{nm} must be a contiguous int16 tensor on q's device")
    if must_do_list is not None:
        if must_do_list.dtype != torch.int16 or not must_do_list.is_contiguous() or must_do_list.device != q.device:
            raise ValueError("must_do_list must be a contiguous int16 tensor on q's device")
    B, H, Sq, D = q.shape
    Skv = k.shape[2]
    if D != 128:
        raise ValueError(f"head_dim must be 128 (got {D})")
    # Sq and Skv may be any positive values: the kernel masks the partial last q-tile (store)
    # and the partial last K/V block (score); V_t is per-head padded to a 64-row boundary.
    if Sq <= 0 or Skv <= 0:
        raise ValueError(f"seq lens must be positive (got Sq={Sq}, Skv={Skv})")
    Skv_pad = ((Skv + 63) // 64) * 64    # per-head V_t padding (block-aligned)

    hip = _hip_runtime()
    lib = _lite_kernel(round_mode)
    if out is None:
        out = torch.empty_like(q)

    v_t = torch.empty((B, H, Skv_pad, D), dtype=v.dtype, device=v.device)  # per-head padded V_t
    rc = lib.launch_v_transpose(
        ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(v_t.data_ptr()),
        B * H * Skv, Skv, H, 0, None)   # heads=H, layout=0 (BHSD)
    if rc != 0:
        raise RuntimeError(f"launch_v_transpose returned {rc}")
    rc = lib.launch_attention_forward_lite(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v_t.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        B, H, Sq, Skv, D,
        ctypes.c_void_p(read_list.data_ptr()), ctypes.c_void_p(write_list.data_ptr()),
        ctypes.c_void_p(must_do_list.data_ptr() if must_do_list is not None else None),
        ctypes.c_float(float(threshold)), ctypes.c_int(int(phase)), ctypes.c_int(0), None)  # layout=0 (BHSD)
    if rc != 0:
        raise RuntimeError(f"launch_attention_forward_lite returned {rc}")
    if hip.hipDeviceSynchronize() != 0:
        raise RuntimeError("hipDeviceSynchronize failed")
    return out
