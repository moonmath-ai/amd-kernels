# Vendored AITER HSACO

Repo-local copy of the minimal AITER forward HSACO set needed by
[`attention_kernel_aiter_v3.cpp`](/cdna3-attention/attention_kernel_aiter_v3.cpp).

Search order at runtime:

1. `AITER_ASM_DIR`
2. `cdna3-attention/vendor/aiter_hsa`
3. legacy `/tmp/aiter_rocm_791921/aiter_meta/hsa`

Included here:
- `gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_{rtne,rtna,rtz}.co`
- `gfx942/fmha_v3_fwd/MI308/fwd_hd128_bf16_{rtne,rtna,rtz}.co`
