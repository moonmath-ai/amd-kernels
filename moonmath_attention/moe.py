"""MXFP4 MoE GEMMs for CDNA3 (MI300X / gfx942).

Two kernels, the gate/up projection and the down projection, over a token list
that has already been sorted by expert and padded to a multiple of the tile
height (the usual ``moe_align_block_size``).  Each computes

    C[m][n] = sum_k A[m][k] * dequant(B[e][k][n])

with ``e`` the expert owning block ``m``.  Weights are MXFP4: one E2M1 nibble
per value and one E8M0 byte per 32 consecutive k.

ROW INDEXING, WHICH DIFFERS BETWEEN THE TWO OPS.  Both read ``A[slot // top_k]``
and write ``C[slot]``, where ``slot`` is the entry in ``sorted_token_ids``.  So:

* **gate/up** takes ``A`` as per-TOKEN hidden states, ``[T, K]``, and fans each
  token out over its ``top_k`` experts.  Pass the real ``top_k``.
* **down** consumes the gate/up output, which already has one row per
  (token, expert) pair, ``[T*top_k, K]``.  Pass ``top_k=1``.

Passing the model's ``top_k`` to down reads the wrong ``A`` rows and returns
plausible-looking numbers, so it is worth checking first when down disagrees
with a reference.

THE WEIGHT REPACK IS REQUIRED.  ``B`` and ``Bs`` must be passed through
:func:`repack_mxfp4` and :func:`repack_mxfp4_scales` once at weight load.  The
kernels read the repacked layout directly; handing them stock weights does not
raise, it silently computes wrong numbers.  Both transforms are permutations
apart from one canonicalization -- MXFP4's negative zero becomes ``+0`` -- so
they cost nothing at run time and change no value the kernels can observe.

Tile heights, column tiling and the down kernel's n-chunk width are chosen by
the planners below.  Call them once per shape, not per layer::

    import moonmath_attention as ma

    w13 = ma.repack_mxfp4(w13_raw)            # once, at load
    w13s = ma.repack_mxfp4_scales(w13s_raw)

    bm = ma.mxfp4_moe_gateup_block_m(rows, n_active)
    sorted_ids, expert_ids, ntpp, n_blocks = align(topk_ids, bm)   # caller's
    ma.mxfp4_moe_gateup(hidden, w13, w13s, out, None, sorted_ids, expert_ids,
                        ntpp, n_blocks, bm, rows, top_k, epilogue=ma.EPI_SITU,
                        situ_beta=4.0, situ_linear_beta=25.0)

Both ops run on the current stream and never sync with the host.
"""
import torch

import moonmath_attention._C as _C

__all__ = [
    "repack_mxfp4",
    "repack_mxfp4_scales",
    "mxfp4_moe_gateup",
    "mxfp4_moe_down",
    "mxfp4_moe_gateup_block_m",
    "mxfp4_moe_gateup_supports_k",
    "mxfp4_moe_down_block_m",
    "mxfp4_moe_down_nt",
    "mxfp4_moe_down_n_steps",
    "mxfp4_moe_down_plan",
    "EPI_NONE",
    "EPI_SITU",
]

#  Bytes of one lane's weight granule: 16 bytes = 32 k of one column.
_B_GRANULE = 16

EPI_NONE = 0
EPI_SITU = 1


# ---------------------------------------------------------------------------
#  Weight preparation.  Offline, once at load, and both are required.
# ---------------------------------------------------------------------------
def _canon_neg_zero(nib: torch.Tensor) -> torch.Tensor:
    """MXFP4 negative zero (nibble 0x8) -> +0.  See :func:`repack_mxfp4`."""
    return torch.where(nib == 0x8, torch.zeros_like(nib), nib)


def repack_mxfp4(B: torch.Tensor) -> torch.Tensor:
    """``[E, N, K/2] uint8 -> [E, K/32, N, 16] uint8``.

    Two permutations, neither of which changes a value:

    1. **Nibble relabel.**  In the standard layout byte ``b`` holds ``k=2b`` in
       its low nibble and ``k=2b+1`` in its high nibble.  Repacked, within each
       group of 8 consecutive k (4 bytes) byte ``j`` holds ``k=8i+j`` low and
       ``k=8i+4+j`` high.  That is exactly the permutation the kernel's
       ``v_perm_b32`` unpack applies, so undoing it here lets the kernel skip
       the LDS transpose and the per-iteration barrier a classical MXFP4 GEMM
       pays.

    2. **n-minor transpose.**  Swap the (n, k-granule) axes at 16-byte
       granularity, so a 16-lane group's fetch is 16 consecutive granules --
       256 contiguous bytes -- rather than 16 scattered ones.  The granule
       moves whole, so every unpack result and every MFMA operand is
       bit-identical; only which byte of memory a lane addresses changes.

    3. **Negative zero canonicalized.**  Nibble 0x8 is MXFP4's -0 and comes out
       as 0x0.  The only thing that changes is the sign of a zero, which no
       accumulation can observe, and it keeps the unpack free to treat the
       magnitude nibble as a plain table index.
    """
    if B.dtype != torch.uint8:
        raise TypeError(f"repack_mxfp4: B must be uint8, got {B.dtype}")
    if B.dim() != 3:
        raise ValueError(f"repack_mxfp4: B must be [E, N, K/2], got {tuple(B.shape)}")
    E, N, Kh = B.shape
    if Kh % _B_GRANULE:
        raise ValueError(f"repack_mxfp4: K/2 must be a multiple of {_B_GRANULE}, got {Kh}")

    lo = _canon_neg_zero(B & 0x0F).reshape(E, N, Kh // 4, 4)       # k = 8i + 2j
    hi = _canon_neg_zero(B >> 4).reshape(E, N, Kh // 4, 4)         # k = 8i + 2j + 1
    v = torch.stack([lo, hi], dim=-1).reshape(E, N, Kh // 4, 8)    # k = 8i + m
    R = (v[..., 0:4] | (v[..., 4:8] << 4)).reshape(E, N, Kh)

    #  (n, granule) -> (granule, n); .contiguous() is what actually reorders.
    return (R.reshape(E, N, Kh // _B_GRANULE, _B_GRANULE)
             .permute(0, 2, 1, 3).contiguous())


def repack_mxfp4_scales(Bs: torch.Tensor) -> torch.Tensor:
    """``[E, N, K/32] uint8 -> [E, K/32, N] uint8``, matching :func:`repack_mxfp4`.

    One E8M0 byte covers 32 k of one column.  Laid out n-minor, the sixteen
    lanes of a column group read sixteen contiguous bytes, so a scale fetch is
    one coalesced load.
    """
    if Bs.dtype != torch.uint8:
        raise TypeError(f"repack_mxfp4_scales: Bs must be uint8, got {Bs.dtype}")
    if Bs.dim() != 3:
        raise ValueError(f"repack_mxfp4_scales: Bs must be [E, N, K/32], got {tuple(Bs.shape)}")
    return Bs.permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
#  Planners.  Pure host arithmetic -- no device work, safe under graph capture.
# ---------------------------------------------------------------------------
def mxfp4_moe_gateup_block_m(num_valid_tokens: int, num_active_experts: int) -> int:
    """Tile height for gate/up: 16, 32 or 48, from rows per expert."""
    return _C.mxfp4_moe_gateup_block_m(num_valid_tokens, num_active_experts)


def mxfp4_moe_gateup_supports_k(K: int) -> bool:
    """Whether the gate/up kernel can serve this K.

    Its slab pipeline only balances its vmcnt rungs over a whole pair, so the
    slab count must be even: K a multiple of 256.
    """
    return bool(_C.mxfp4_moe_gateup_supports_k(K))


def mxfp4_moe_down_block_m(num_valid_tokens: int, num_active_experts: int, K: int) -> int:
    """Tile height for down: 16, 32 or 64, from rows per expert.

    Returns 0 when this kernel cannot serve the shape. It stages a whole A tile
    in LDS, so it only has tiles for the TP-sharded intermediate sizes
    (K = 384/512/768). Under expert parallelism the experts are unsharded and
    K is the full width; route those to gate/up with EPI_NONE.
    """
    return _C.mxfp4_moe_down_block_m(num_valid_tokens, num_active_experts, K)


def mxfp4_moe_down_nt(num_valid_tokens: int, num_active_experts: int) -> int:
    """Column tiles per wave for down: 1 or 2."""
    return _C.mxfp4_moe_down_nt(num_valid_tokens, num_active_experts)


def mxfp4_moe_down_n_steps(num_m_blocks: int, N: int, num_cus: int, block_m: int,
                           nt: int) -> int:
    """How many BN-wide column chunks one down workgroup sweeps."""
    return _C.mxfp4_moe_down_n_steps(num_m_blocks, N, num_cus, block_m, nt)


def mxfp4_moe_down_plan(num_valid_tokens: int, num_active_experts: int, N: int, K: int,
                        num_m_blocks: int, num_cus: int = 0):
    """``(block_m, nt, n_steps)`` for down in one call.

    ``num_m_blocks`` must be the count the caller aligned at ``block_m``, so
    plan in two steps if the alignment depends on it: take ``block_m`` from
    :func:`mxfp4_moe_down_block_m` first, align, then call this.  ``num_cus``
    defaults to the device's multiprocessor count.
    """
    if num_cus <= 0:
        num_cus = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    block_m = mxfp4_moe_down_block_m(num_valid_tokens, num_active_experts, K)
    nt = mxfp4_moe_down_nt(num_valid_tokens, num_active_experts)
    n_steps = mxfp4_moe_down_n_steps(num_m_blocks, N, num_cus, block_m, nt)
    return block_m, nt, n_steps


# ---------------------------------------------------------------------------
#  Ops
# ---------------------------------------------------------------------------
def mxfp4_moe_gateup(A: torch.Tensor, B: torch.Tensor, Bs: torch.Tensor, C: torch.Tensor,
                     topk_weights, sorted_token_ids: torch.Tensor,
                     expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor,
                     num_m_blocks: int, block_m: int, num_valid_tokens: int, top_k: int,
                     epilogue: int = EPI_NONE, mul_routed_weight: bool = False,
                     situ_beta: float = 1.0, situ_linear_beta: float = 1.0) -> None:
    """Gate/up projection, writing into ``C`` in place.

    ``A`` is ``[rows, K]`` bf16, ``C`` is ``[rows, N]`` bf16.  With
    ``epilogue=EPI_SITU`` the kernel computes gate and up in paired wave halves
    and applies the Kimi-K3 SituGLU, so ``B`` carries ``2*N`` columns while
    ``C`` stays ``N`` wide; with ``EPI_NONE`` it writes the dot product out
    directly and ``B`` carries ``N``.

    ``num_tokens_post_padded`` is read on the device by every workgroup, so it
    must be the live int32 device tensor the alignment produced, not a host
    scalar.
    """
    _C.mxfp4_moe_gateup(A, B, Bs, C, topk_weights, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, num_m_blocks, block_m, num_valid_tokens,
                        top_k, epilogue, mul_routed_weight, situ_beta, situ_linear_beta)


def mxfp4_moe_down(A: torch.Tensor, B: torch.Tensor, Bs: torch.Tensor, C: torch.Tensor,
                   topk_weights, sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                   num_tokens_post_padded: torch.Tensor, num_m_blocks: int, block_m: int,
                   n_steps: int, nt: int, num_valid_tokens: int, top_k: int,
                   mul_routed_weight: bool = False) -> None:
    """Down projection, writing into ``C`` in place.

    ``A`` is ``[rows, K]`` bf16 with ``K`` in {384, 512, 768}, ``C`` is
    ``[rows, N]`` bf16.  One workgroup stages its whole A tile in LDS and
    sweeps ``n_steps`` column chunks against it, which is why K must be short.

    At ``nt=2`` the kernel pairs each lane's two results into one dword store,
    which requires an even ``N`` and unit column strides on ``C`` and ``Bs``.
    """
    _C.mxfp4_moe_down(A, B, Bs, C, topk_weights, sorted_token_ids, expert_ids,
                      num_tokens_post_padded, num_m_blocks, block_m, n_steps, nt,
                      num_valid_tokens, top_k, mul_routed_weight)
