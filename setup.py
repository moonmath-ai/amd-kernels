"""Build one torch CUDAExtension per GPU-architecture folder in csrc/ (ROCm-only).

    csrc/gfx942/  CDNA3 (MI300X): MHA forward, MLA a16w8 decode, MXFP4 MoE  -> moonmath_amd._C_gfx942
    csrc/gfx950/  CDNA4 (MI355X): bf16 attention forward (D128, MLA 192/128) -> moonmath_amd._C_gfx950

Targets come from GPU_ARCHS, as in AITER: "native" (default; the GPUs rocminfo lists) or an explicit list
like "gfx942;gfx950". A module is built only when its arch is a target. CDNA3_ARCH, the old single-arch
knob, is honored when GPU_ARCHS is unset; with neither set and no GPU present, the build targets gfx942,
just as before.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch
from setuptools import setup
from setuptools.command.build import build as _build
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Fail fast when built against a non-ROCm torch
if torch.version.hip is None:
    raise RuntimeError(
        "moonmath_amd requires a ROCm build of PyTorch. "
        "torch.version.hip is None, indicating this is not a ROCm installation. "
        "Install ROCm PyTorch and ensure hipcc is available on PATH."
    )

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"
DIST = ROOT / "dist"

_HIP_FLAGS = ["-O3", "-std=c++17", "-ffast-math", "-fno-math-errno", "-mllvm", "-amdgpu-early-inline-all=true"]

# One module per arch folder: its compile units (other files there are headers) and device flags.
MODULES = {
    "gfx942": dict(
        sources=[
            "attention_api.cpp",
            "attention_rtna.hip",
            "attention_rtne.hip",
            "attention_rtz.hip",
            # ── MLA (DeepSeek-V3) — absorbed decode with bf16 Q / fp8 KV ──
            "mla_decode_a16w8_api.cpp",  # exposes mla_decode_a16w8 and mla_dcp_lse_merge_ranks
            "mla_decode_a16w8.hip",      # a16w8 absorbed decode on a DENSE paged batch, any q_len, H 1..128,
                                         #   optional DCP: (position, head) rows in 16..96-row query tiles
                                         #   versus a 16-token KV tile, with a first-class base-2 LSE
            # ── DeepSeek-V4 sparse attention decode (from the MLA kernel) ──
            "dsv4_sparse_decode_api.cpp",  # exposes dsv4_sparse_decode
            "dsv4_sparse_decode.hip",      # two paged E8M0-fp8 caches, CSR slot lists, attention sink
            # ── MXFP4 MoE GEMMs ──
            "mxfp4_moe_api.cpp",         # exposes mxfp4_moe_gateup / _down and their planners
            "mxfp4_moe_gateup.hip",      # gate/up projection, 16/32/48 tiles, EPI_NONE or SituGLU
            "mxfp4_moe_down.hip",        # down projection: an LDS-staged A tile swept across n chunks
        ],
        hip_flags=_HIP_FLAGS,
    ),
    "gfx950": dict(
        sources=[
            "attention_api.cpp",         # exposes forward / forward_varlen / forward_paged / VarlenPlan
            "attention.hip",             # head dimension 128
            "attention_mla.hip",         # MLA prefill with Q/K 192 and V 128
        ],
        # -fno-slp-vectorize: SLP would pack the softmax into v_pk_*_f32, which can't co-issue with MFMA.
        hip_flags=_HIP_FLAGS + ["-fno-slp-vectorize"],
    ),
}


def _native_archs():
    """gfx archs reported by rocminfo, matching AITER's "native"."""
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
    except OSError as e:
        raise RuntimeError("GPU_ARCHS=native needs rocminfo; set GPU_ARCHS, e.g. GPU_ARCHS=\"gfx942;gfx950\"") from e
    archs = sorted(set(re.findall(r"^\s*Name:\s+(gfx\w+)\s*$", out, re.MULTILINE)))
    if not archs:
        raise RuntimeError("rocminfo lists no GPU; set GPU_ARCHS, e.g. GPU_ARCHS=\"gfx942;gfx950\"")
    return archs


def build_targets():
    """{arch folder: offload targets}; a target may include features, e.g. gfx942:xnack-."""
    env = os.environ.get("GPU_ARCHS") or os.environ.get("CDNA3_ARCH")
    targets = [t.strip() for t in (env or "native").split(";") if t.strip()]
    if not targets:
        raise RuntimeError(f"GPU_ARCHS={env!r} names no arch; supported: {', '.join(MODULES)}")
    if targets == ["native"]:
        try:
            targets = _native_archs()
        except RuntimeError:
            if env:                         # GPU_ARCHS=native explicitly requested the local GPUs
                raise
            targets = ["gfx942"]            # no knob and no GPU (e.g. a build container): keep the old default
            print("moonmath_amd: no GPU found and GPU_ARCHS not set: building for gfx942")
    by_arch = {}
    for t in targets:
        arch = t.split(":")[0]
        if arch not in MODULES:
            raise RuntimeError(f"GPU_ARCHS: moonmath_amd has no kernels for {t}; supported: {', '.join(MODULES)}")
        by_arch.setdefault(arch, []).append(t)
    return by_arch


# Stage csrc/<arch>/ into dist/<arch>/ and build there. torch's ROCm hipify writes
# its generated `<stem>_hip.cpp` beside the source it compiles, so building from
# dist/ keeps those intermediates out of csrc/.
def stage(arch):
    for src in (CSRC / arch).rglob("*"):
        dst = DIST / src.relative_to(CSRC)
        if src.is_file() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return [str((DIST / arch / name).relative_to(ROOT)) for name in MODULES[arch]["sources"]]


def extension(arch, targets):
    return CUDAExtension(
        name=f"moonmath_amd._C_{arch}",
        sources=stage(arch),
        # ROCm-only flags (no is_rocm branch: BuildExtension sends "nvcc" to hipcc on ROCm)
        extra_compile_args={
            "cxx": ["-O3"],                  # torch supplies the -std its headers require
            "nvcc": MODULES[arch]["hip_flags"] + [f"--offload-arch={t}" for t in targets],
        },
    )


class DistBuild(_build):
    """Keep every build artifact under dist/.

    Setting ``build_base`` places ``build_ext`` (object files, ninja temp files,
    the assembled ``lib.*`` tree) and ``bdist`` (wheel staging) under dist/,
    rather than spawning a separate top-level build/ directory.
    """

    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(DIST)


setup(
    ext_modules=[extension(arch, targets) for arch, targets in build_targets().items()],
    cmdclass={"build": DistBuild, "build_ext": BuildExtension},
)
