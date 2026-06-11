"""Build hook: invoke `hipcc` to compile the dense kernel into RTNA, RTNE and RTZ
.so variants and bundle them as package_data."""
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent.resolve()
KERNEL = ROOT / "attention_kernel.hip"
PKG_DIR = ROOT / "moonmath_attention"

ARCH = os.environ.get("CDNA3_ARCH", "gfx942")
HIPCC = os.environ.get("HIPCC", "hipcc")
EXTRA_FLAGS = [f for f in os.environ.get("EXTRA_LLVM_FLAGS", "").split() if f]

OPUS_INC = ROOT / "third_party" / "aiter" / "csrc" / "include" / "opus"

HIPCC_FLAGS = [
    "-shared", "-fPIC", "-O3",
    f"--offload-arch={ARCH}",
    "-ffast-math", "-fno-math-errno",
    "-mllvm", "-amdgpu-early-inline-all=true",
    f"-I{OPUS_INC}",
] + EXTRA_FLAGS


def _build_kernel(round_mode: str, out_dir: Path, src: Path = KERNEL, tag: str = "") -> Path:
    out = out_dir / f"libattention_{tag}{round_mode.lower()}.so"
    cmd = [HIPCC, *HIPCC_FLAGS, f"-DBF16_ROUND={round_mode}",
           "-o", str(out), str(src)]
    print("[hipcc]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    return out


class BuildPyWithKernel(build_py):
    """Build the kernel .so files into the source package dir before packaging.

    Building into PKG_DIR (rather than build/lib/cdna3_attention) means the
    package_data glob picks them up, and editable installs work without extra
    plumbing.
    """
    def run(self):
        if shutil.which(HIPCC) is None:
            raise RuntimeError(
                f"`{HIPCC}` not found on PATH. Install ROCm and ensure hipcc is "
                f"reachable, or set HIPCC=/path/to/hipcc."
            )
        PKG_DIR.mkdir(exist_ok=True)
        for round_mode in ("RTNA", "RTNE", "RTZ"):
            _build_kernel(round_mode, PKG_DIR)                                  # dense
        super().run()


setup(cmdclass={"build_py": BuildPyWithKernel})
