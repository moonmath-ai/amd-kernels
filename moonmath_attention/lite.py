"""LiteAttention — cross-timestep skip for the CDNA3 kernel.

Mirrors moonmath-ai/LiteAttention's SM90 logic (byte-exact reverse/phase skip
lists) on top of the hand-tuned MI300X attention kernel. Across diffusion
denoising steps the attention pattern is near-stable, so each step:

  • READS the skip list written last step → only loads/computes the named K-blocks
    (everything else is skipped end-to-end: no HBM K load, no QK, no PV, no softmax);
  • WRITES a fresh skip list for the next step — a K-block is marked skippable iff,
    for ALL 256 q-rows of the CTA, its max score stayed within `threshold` (log2)
    of the running softmax max (the AND-of-8-waves generalization of FA3's
    AND-of-4-warps).

The tile is (kBlockM=256, kBlockN=64) — the kernel's native CTA geometry. The
first step (compute-all seed) is exact full attention; later steps approximate
(skipped blocks contribute ~0, which is what the threshold guarantees).

Usage:
    >>> import torch, moonmath_attention as ma
    >>> attn = ma.LiteAttention(threshold=-6.0)
    >>> for t in range(num_denoise_steps):
    ...     out = attn(q, k, v)          # same shape/dtype as q; skips grow over steps
    >>> attn.reset_skip_state()          # e.g. on a new prompt / shape change
"""
import math

import torch

from . import _kernel

# Kernel tile geometry (CTA = 8 waves x kQGroups=2 x 16 rows = 256 q-rows; kNBlock=64).
KBLOCK_M = 256
KBLOCK_N = 64


def _ceil_div(a, b):
    return (a + b - 1) // b


class LiteAttention:
    """Stateful LiteAttention wrapper. Holds a double-buffered int16 skip list and
    flips the read/write role each call, exactly like LiteAttention's `_phase`.

    Args:
        threshold: log2-space skip threshold (negative; ~ -6). A block is
            skippable when its max log-score is > |threshold| below the running max.
            More negative => fewer skips (more conservative).
        enable_skipping: if False, every call is exact dense attention.
        round_mode: "rtna" (default), "rtne", or "rtz".
        reverse_skip_list: kept for LiteAttention API parity; always True here
            (the kernel implements the byte-exact reverse/phase encoding).
    """

    def __init__(self, threshold: float = -6.0, *, enable_skipping: bool = True,
                 round_mode: str = "rtna", reverse_skip_list: bool = True):
        if threshold >= 0:
            raise ValueError(f"threshold must be negative (log2 units); got {threshold}")
        self.threshold = float(threshold)
        self.enable_skipping = bool(enable_skipping)
        self.round_mode = round_mode
        self.reverse_skip_list = True  # only reverse/phase is implemented (byte-exact)
        self._skip = None             # [2, B, H, qtiles, ktiles+2] int16
        self._phase = 0
        self._sig = None              # (B, H, S, device) reinit signature

    # tile sizes (LiteAttention API parity)
    @staticmethod
    def get_MN(head_dim=128, dtype=torch.bfloat16, v_colmajor=False):
        return KBLOCK_M, KBLOCK_N

    def reset_skip_state(self):
        """Drop the learned skip list; the next call starts from compute-all."""
        self._skip = None
        self._phase = 0
        self._sig = None

    def set_threshold(self, threshold: float):
        if threshold >= 0:
            raise ValueError("threshold must be negative (log2 units)")
        self.threshold = float(threshold)

    def enable_skip_optimization(self, enable: bool = True):
        self.enable_skipping = bool(enable)

    @property
    def read_list(self):
        if self._skip is None:
            return None
        return self._skip[self._phase]

    @property
    def write_list(self):
        if self._skip is None:
            return None
        return self._skip[1 - self._phase]

    def _ensure(self, q, k):
        B, H, Sq, D = q.shape
        Skv = k.shape[2]                       # cross-attn: K/V len may differ from Q
        sig = (B, H, Sq, Skv, q.device)
        if self._skip is not None and self._sig == sig:
            return
        qtiles = _ceil_div(Sq, KBLOCK_M)
        ktiles = _ceil_div(Skv, KBLOCK_N)      # skip-list is over K-blocks
        skip = torch.zeros(2, B, H, qtiles, ktiles + 2, dtype=torch.int16, device=q.device)
        # Compute-all reverse seed (skip_list.h init): [2, ktiles-1, -1].
        skip[0, :, :, :, 0] = 2
        skip[0, :, :, :, 1] = ktiles - 1
        skip[0, :, :, :, 2] = -1
        self._skip = skip
        self._phase = 0
        self._sig = sig

    def __call__(self, q, k, v, *, out=None, must_do_list=None):
        return self.forward(q, k, v, out=out, must_do_list=must_do_list)

    def forward(self, q, k, v, *, out=None, must_do_list=None):
        # `must_do_list` (optional int16 [len, s0,e0, ...]): K-block ranges pinned as
        # always-compute in the emitted write list (e.g. attention sinks). None ⇒ pure vote.
        if not self.enable_skipping:
            return _kernel.forward(q, k, v, out=out, round_mode=self.round_mode)

        self._ensure(q, k)
        if self._phase == 0:
            read, write = self._skip[0], self._skip[1]
            self._phase = 1
        else:
            read, write = self._skip[1], self._skip[0]
            self._phase = 0
        # phase passed to the kernel is taken AFTER the swap (== LiteAttention).
        kphase = 1 if self._phase == 1 else 0
        return _kernel.forward_lite(
            q, k, v, read.contiguous(), write.contiguous(),
            threshold=self.threshold, phase=kphase, must_do_list=must_do_list,
            out=out, round_mode=self.round_mode)


# Backward-compatible alias (renamed from MoonLiteAttention -> LiteAttention).
MoonLiteAttention = LiteAttention
