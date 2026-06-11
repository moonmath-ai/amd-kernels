"""Wrapper that calls the compiled _C extension, with CPU fallback support."""

import torch

try:
    import moonmath_attention._C as _C
except ImportError as e:
    raise ImportError(
        "moonmath_attention: failed to import _C extension. "
        "Build with: pip install -e . --no-build-isolation (requires ROCm PyTorch + hipcc)"
    ) from e


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    round_mode: str = "rtna",
    layout: str = "bhsd",
) -> torch.Tensor:
    """Fused forward attention: O = softmax(QKᵀ / √D) V on MI300X.

    Args:
        q, k, v: torch.bfloat16 tensors of shape (batch, heads, seq_len, head_dim)
                 or (batch, seq_len, heads, head_dim) depending on layout.
                 Contiguous. head_dim=128.
                 CPU tensors are moved to GPU under the hood; GPU tensors used in place.
        out: optional preallocated output tensor. Must match q's shape, dtype,
             device, and be contiguous. If None, a new tensor is allocated.
        round_mode: "rtna" (default; ties-away-from-zero), "rtne" (~0.5 ULP error),
            or "rtz" (cheaper; ~1 ULP, truncation).
        layout: "bhsd" (default; batch, heads, seq, head_dim) or "bshd"
            (batch, seq, heads, head_dim).

    Returns:
        torch.bfloat16 tensor of the same shape and device as the inputs.
    """
    # Type and basic validation (before device check)
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor; got {type(t).__name__}")
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be torch.bfloat16; got dtype={t.dtype}")
        if t.dim() != 4:
            raise ValueError(
                f"{name} must be 4-D ({layout.upper()}); got shape={tuple(t.shape)}"
            )

    # CPU fallback: move to GPU, call _C, move result back
    if q.device.type == "cpu":
        # Ensure k, v are also on CPU
        if k.device.type != "cpu" or v.device.type != "cpu":
            raise ValueError(
                "q is on CPU but k or v is not; all must be on the same device"
            )

        # Move to first available CUDA device
        device = torch.device("cuda:0")
        q_gpu = q.to(device)
        k_gpu = k.to(device)
        v_gpu = v.to(device)
        out_gpu = out.to(device) if out is not None else None

        result_gpu = _C.forward(q_gpu, k_gpu, v_gpu, out_gpu, round_mode, layout)
        return result_gpu.cpu()

    # GPU path: call _C directly
    elif q.device.type in ("cuda", "hip"):
        # Validate k, v on same device
        if k.device != q.device or v.device != q.device:
            raise ValueError(
                f"q/k/v must be on the same device; got q={q.device}, k={k.device}, v={v.device}"
            )

        return _C.forward(q, k, v, out, round_mode, layout)

    else:
        raise NotImplementedError(
            f"Unsupported device {q.device!r}; expected 'cpu' or 'cuda'/'hip'"
        )


def forward_lite(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    read_list: torch.Tensor,
    write_list: torch.Tensor,
    *,
    threshold: float,
    phase: int,
    must_do_list: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    round_mode: str = "rtna",
    layout: str = "bhsd",
) -> torch.Tensor:
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
    blocks). None => no forcing.

    GPU tensors only (the diffusion use case). Output is an approximation of full
    attention (skipped blocks contribute 0) — exact only with a compute-all list.

    Note: The lite kernel symbol does not exist in the current attention_kernel.hip.
    This will raise a NotImplementedError until the kernel is added.
    """
    # Currently raises an error from _C.forward_lite
    return _C.forward_lite(
        q,
        k,
        v,
        read_list,
        write_list,
        threshold,
        phase,
        must_do_list,
        out,
        round_mode,
        layout,
    )
