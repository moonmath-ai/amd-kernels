"""LiteAttention — cross-timestep skip for the CDNA3 kernel.

Mirrors moonmath-ai/LiteAttention's SM90 logic (byte-exact reverse/phase skip
lists) on top of the hand-tuned MI300X attention kernel. Across diffusion
denoising steps the attention pattern is near-stable, so each step:

  • READS the skip list written last step → only loads/computes the named K-blocks
    (everything else is skipped end-to-end: no HBM K load, no QK, no PV, no softmax);
  • WRITES a fresh skip list for the next step — a K-block is marked skippable iff,
    for ALL 384 q-rows of the CTA, its max score stayed within `threshold` (log2)
    of the running softmax max (the AND-of-8-waves generalization of FA3's
    AND-of-4-warps).

The tile is (kBlockM=384, kBlockN=64) — the kernel's native 3q CTA geometry
(8 waves × QTilesPerWave=3 × 16 rows = 384 q-rows; KvBlockRows=64). The first
step (compute-all seed) is exact full attention; later steps approximate
(skipped blocks contribute ~0, which is what the threshold guarantees).

Usage:
    >>> import torch, moonmath_attention as ma
    >>> attn = ma.LiteAttention(threshold=-6.0)
    >>> for t in range(num_denoise_steps):
    ...     out = attn(q, k, v)          # same shape/dtype as q; skips grow over steps
    >>> attn.reset_skip_state()          # e.g. on a new prompt / shape change

The low-level entry point is `ma.forward_lite(q, k, v, read_list, write_list,
threshold=…, phase=…, must_do_list=…)`; this class just owns the double-buffered
skip list and flips the read/write role each step.
"""

import torch

from . import _kernel

# Kernel tile geometry (3q CTA = 8 waves × QTilesPerWave=3 × 16 rows = 384 q-rows; KvBlockRows=64).
KBLOCK_M = 384
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
        layout: "bhsd" [B,H,S,D] or "bshd" [B,S,H,D].
    """

    def __init__(
        self,
        threshold: float = -6.0,
        *,
        enable_skipping: bool = True,
        round_mode: str = "rtna",
        layout: str = "bhsd",
    ):
        if threshold >= 0:
            raise ValueError(
                f"threshold must be negative (log2 units); got {threshold}"
            )
        if layout not in ("bhsd", "bshd"):
            raise ValueError(f"layout must be 'bhsd' or 'bshd' (got {layout!r})")
        self.threshold = float(threshold)
        self.enable_skipping = bool(enable_skipping)
        self.round_mode = round_mode
        self.layout = layout  # "bhsd" [B,H,S,D] or "bshd" [B,S,H,D]
        self._skip = (
            None  # [2, B, H, qtiles, ktiles+2] int16 (read/write double buffer)
        )
        self._phase = 0
        self._sig = None  # (B, H, Sq, Skv, device) reinit signature

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
        return None if self._skip is None else self._skip[self._phase]

    @property
    def write_list(self):
        return None if self._skip is None else self._skip[1 - self._phase]

    def _ensure(self, q, k):
        sax, hax = (1, 2) if self.layout == "bshd" else (2, 1)
        B, H, Sq, D = q.shape[0], q.shape[hax], q.shape[sax], q.shape[3]
        Skv = k.shape[sax]  # cross-attn: K/V len may differ from Q
        sig = (B, H, Sq, Skv, q.device)
        if self._skip is not None and self._sig == sig:
            return
        qtiles = _ceil_div(
            Sq, KBLOCK_M
        )  # == kernel's num_q_blocks (one RLE row per CTA)
        ktiles = _ceil_div(Skv, KBLOCK_N)  # skip-list is over K-blocks
        skip = torch.zeros(
            2, B, H, qtiles, ktiles + 2, dtype=torch.int16, device=q.device
        )
        # Compute-all reverse seed (skip_list.h init, phase 1 ascending): [2, ktiles-1, -1].
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
            return _kernel.forward(
                q, k, v, out=out, round_mode=self.round_mode, layout=self.layout
            )
        self._ensure(q, k)
        # Swap read/write role, then pass the post-swap phase to the kernel (== LiteAttention).
        # Each CTA processes its read list's kept blocks and writes the next-step list inline.
        read, write = self._skip[self._phase], self._skip[1 - self._phase]
        self._phase = 1 - self._phase
        ret = _kernel.forward_lite(
            q,
            k,
            v,
            read.contiguous(),
            write.contiguous(),
            threshold=self.threshold,
            phase=self._phase,
            must_do_list=must_do_list,
            out=out,
            round_mode=self.round_mode,
            layout=self.layout,
        )
        if __import__("os").environ.get("MOONMATH_DEBUG_SKIP"):
            n = getattr(self, "_dbg", 0)
            if n < 8:
                import torch as _t

                w = self._skip[
                    1 - self._phase
                ]  # the list just written (becomes next read)
                row = w[0, 0, 0].tolist()
                ls = 1 if self._phase else -1
                ln = int(row[0])
                i = ln
                kept = 0
                while i >= 2:
                    rb = row[i] + ls
                    re = row[i - 1] + ls
                    i -= 2
                    b = rb
                    while (b < re) if ls > 0 else (b > re):
                        kept += 1
                        b += ls
                ktiles = w.shape[-1] - 2
                print(
                    f"[skipdbg] call={n} thr={self.threshold} wrote kept={kept}/{ktiles}",
                    flush=True,
                )
                self._dbg = n + 1
        return ret
