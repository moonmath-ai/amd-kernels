#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../asm"
SRC_DIR="${AITER_V3_HSACO_DIR:-/tmp/aiter_rocm_791921/aiter_meta/hsa/gfx942/fmha_v3_fwd/MI300}"
ARCH="${AITER_V3_ARCH:-gfx942}"
PREFIX="${AITER_V3_PREFIX:-fwd_hd128_bf16}"

find_llvm_tool() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return 0
  fi

  local candidate
  for candidate in /opt/rocm-*/llvm/bin/"${name}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "error: could not find ${name}" >&2
  exit 1
}

LLVM_OBJDUMP="${LLVM_OBJDUMP:-$(find_llvm_tool llvm-objdump)}"
LLVM_READELF="${LLVM_READELF:-$(find_llvm_tool llvm-readelf)}"

mkdir -p "${OUT_DIR}"

variants=("$@")
if [[ ${#variants[@]} -eq 0 ]]; then
  variants=(rtne rtna rtz)
fi

for variant in "${variants[@]}"; do
  hsaco="${SRC_DIR}/${PREFIX}_${variant}.co"
  if [[ ! -f "${hsaco}" ]]; then
    echo "warning: skipping missing ${hsaco}" >&2
    continue
  fi

  base="${OUT_DIR}/aiter_v3_${PREFIX}_${variant}_${ARCH}"
  echo "Disassembling ${hsaco}"
  "${LLVM_OBJDUMP}" --disassemble --mcpu="${ARCH}" "${hsaco}" > "${base}.s"
  "${LLVM_READELF}" --notes "${hsaco}" > "${base}.notes.txt"
done
