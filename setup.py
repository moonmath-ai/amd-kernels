"""Build moonmath_attention._C extension via torch CUDAExtension (ROCm-only)."""

import os
import shutil
from pathlib import Path

import torch
from setuptools import setup
from setuptools.command.build import build as _build
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Verify ROCm build (fail fast if someone tries to build on non-ROCm torch)
if torch.version.hip is None:
    raise RuntimeError(
        "moonmath_attention requires a ROCm build of PyTorch. "
        "torch.version.hip is None, indicating this is not a ROCm installation. "
        "Install ROCm PyTorch and ensure hipcc is available on PATH."
    )

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"
DIST = ROOT / "dist"

# Honor CDNA3_ARCH env (default gfx942 for MI300X)
ARCH = os.environ.get("CDNA3_ARCH", "gfx942")


# Stage csrc/ sources into dist/ and build from there. torch's ROCm hipify writes
# its generated `<stem>_hip.cpp` next to the source it compiles, so building from
# dist/ keeps those intermediates out of csrc/.
DIST.mkdir(parents=True, exist_ok=True)

_compile_names = [
    "attention_api.cpp",
    "attention_rtna.hip",
    "attention_rtne.hip",
    "attention_rtz.hip",
    # ── MLA (DeepSeek-V3) — absorbed decode, bf16 Q / fp8 KV ──
    "mla_decode_a16w8_api.cpp",  # binds _C.mla_decode_a16w8 / _paged_dev / _plan_parts
    "mla_decode_a16w8.hip",      # a16w8 absorbed decode: bf16 Q + fp8 KV, 8 waves (CONS=4/PROD=4), TileTok=64
    "mla_decode_a16w8_multiq_api.cpp",  # binds _C.mla_decode_a16w8_multiq / _paged_dev / _plan_parts_q
    "mla_decode_a16w8_multiq.hip",      # a16w8 multi-query decode, q_len 4..8: 8 computing waves, TileTok=16
]
_include_names = ["attention_kernel.hip", "opus.hpp"]

sources = []
for name in _compile_names + _include_names:
    src = CSRC / name
    dst = DIST / name
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
    if name in _compile_names:
        sources.append(str(dst.relative_to(ROOT)))

# ROCm-only flags (no is_rocm branch, BuildExtension routes "nvcc" to hipcc on ROCm)
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        f"--offload-arch={ARCH}",
        "-ffast-math",
        "-fno-math-errno",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
    ],
}


class DistBuild(_build):
    """Route every build artifact under dist/.

    Setting ``build_base`` makes ``build_ext`` (object files, ninja temp files,
    the assembled ``lib.*`` tree) and ``bdist`` (wheel staging) all live under
    dist/ instead of spawning a separate top-level build/ directory.
    """

    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(DIST)


setup(
    ext_modules=[
        CUDAExtension(
            name="moonmath_attention._C",
            sources=sources,
            # include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build": DistBuild, "build_ext": BuildExtension},
)
