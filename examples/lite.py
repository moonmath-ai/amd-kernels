"""LiteAttention usage: cross-timestep K-block skipping for diffusion-style loops.

Across diffusion denoising steps the attention pattern is near-stable, so
``LiteAttention`` learns which K-blocks are negligible and skips them end-to-end
on the next step (no HBM K load, no QK, no PV, no softmax for skipped blocks).

  * Step 0 starts from a compute-all seed  -> exact full attention.
  * Each step WRITES a skip list (a K-block is skippable iff its max score stayed
    within ``threshold`` log2-units of the running softmax max, for ALL q-rows of
    the CTA) and the NEXT step READS it.

The wrapper is stateful: call ``reset_skip_state()`` on a new prompt / shape.
Requires an AMD GPU (gfx942) — the lite path is GPU-only.

Run:  HIP_VISIBLE_DEVICES=0 python examples/lite.py
"""
import torch

import moonmath_attention as ma

B, H, S, D = 1, 8, 4096, 128
assert torch.cuda.is_available(), "LiteAttention requires an AMD GPU (gfx942)"
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)

# Peaked keys (a few dominant directions) make the attention sparse enough that
# skipping actually fires — typical of real diffusion attention maps.
q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
kdir = torch.randn(B, H, S, D, device=dev, dtype=torch.float32, generator=g)
kdir = kdir / kdir.norm(dim=-1, keepdim=True)
knorm = torch.exp(torch.randn(B, H, S, 1, device=dev, generator=g) * 1.6) * (D ** 0.5)
k = (kdir * knorm).to(torch.bfloat16)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)

# threshold (log2, negative): more negative -> fewer skips / more conservative.
attn = ma.LiteAttention(threshold=-8.0, round_mode="rtna")

ktiles = (S + ma.lite.KBLOCK_N - 1) // ma.lite.KBLOCK_N


def computed_blocks(read_list):
    """Count K-blocks the skip list keeps (computes) for q-tile 0, head 0."""
    row = read_list[0, 0, 0].tolist()
    n = row[0]
    if n <= 0:
        return 0
    vals, total = row[1:1 + n], 0
    for j in range(0, len(vals), 2):
        total += abs(vals[j] - vals[j + 1]) + 1
    return total


print(f"shape (B,H,S,D)=({B},{H},{S},{D})  ktiles={ktiles}  threshold={attn.threshold}\n")

# --- Step 0: compute-all seed is EXACT; compare to the fp32 reference. -----------
# (read_list is allocated lazily inside the first call; step 0 consumes the
#  compute-all seed = all ktiles blocks.)
out0 = attn(q, k, v)
qf, kf, vf = q.float(), k.float(), v.float()
ref = torch.softmax(qf @ kf.transpose(-1, -2) / (D ** 0.5), dim=-1) @ vf
err0 = (out0.float() - ref).abs().max().item()
print(f"step 0 (compute-all seed): computed {ktiles}/{ktiles} blocks  "
      f"max|out-ref|={err0:.3e}  <- exact")

# --- Steps 1..N: read last step's learned skip list, skip negligible blocks. -----
# Snapshot read_list BEFORE the call — that is the list this step will consume
# (the wrapper flips read/write phase during the call).
for step in range(1, 5):
    kept = computed_blocks(attn.read_list)   # list this step WILL consume
    out = attn(q, k, v)
    drift = (out.float() - ref).abs().max().item()
    print(f"step {step}: computed {kept:3d}/{ktiles} blocks ({100*kept/ktiles:4.0f}%)  "
          f"max|out-ref|={drift:.3e}")

print("\nLower (more negative) threshold -> fewer skips, smaller drift. "
      "Call attn.reset_skip_state() when the prompt/shape changes.")

# --- Dense (exact) attention via the same package, for reference. ----------------
out_dense = ma.forward(q, k, v)
print(f"\ndense ma.forward max|out-ref| = {(out_dense.float() - ref).abs().max().item():.3e}")
