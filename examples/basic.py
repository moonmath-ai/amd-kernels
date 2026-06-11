"""Run the kernel on a small shape and verify against a fp32 reference."""

import torch
import moonmath_attention as ma

B, H, S, D = 1, 4, 1024, 128
torch.manual_seed(42)

q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

out = ma.forward(q, k, v)

# fp32 reference
qf, kf, vf = q.float(), k.float(), v.float()
ref = torch.softmax(qf @ kf.transpose(-1, -2) / (D**0.5), dim=-1) @ vf

diff = (out.float() - ref).abs()
print(f"shape={tuple(out.shape)}  dtype={out.dtype}")
print(f"max_abs={diff.max().item():.3e}  rmse={(diff**2).mean().sqrt().item():.3e}")
