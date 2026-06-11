# AITER V3 ASM Dumps

Generated from the external AITER code objects referenced by
[`attention_kernel_aiter_v3.cpp`](/home/tarik/cdna3-attn/cdna3-attention/attention_kernel_aiter_v3.cpp).

Regenerate with:

```bash
make -C cdna3-attention aiter-asm
```

Defaults:
- Source HSACO directory: `/tmp/aiter_rocm_791921/aiter_meta/hsa/gfx942/fmha_v3_fwd/MI300`
- Variants: `rtne`, `rtna`, `rtz`
- Outputs:
  - `cdna3-attention/asm/aiter_v3_fwd_hd128_bf16_<variant>_gfx942.s`
  - `cdna3-attention/asm/aiter_v3_fwd_hd128_bf16_<variant>_gfx942.notes.txt`
