#!/bin/bash
# Builds aiter's hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co (hipcc for gfx950).
# usage: build.sh [out.co]
set -e
cd "$(dirname "$0")"
out=${1:-f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ps.co}
hipcc -std=c++20 -O3 --offload-arch=gfx950 -mllvm -amdgpu-spill-vgpr-to-agpr=0 -mllvm -pragma-unroll-threshold=1000000 -mllvm -max-bytes-for-alignment=28 \
  --genco --no-gpu-bundle-output f4gemm_256x256_ps.hip -o "$out"
