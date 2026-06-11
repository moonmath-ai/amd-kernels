#!/usr/bin/env bash
# ATT-only profile: run rocprofv3 Advanced Thread Trace on the CDNA3 attention
# HIP kernel, then generate the ISA analysis HTML. No PMC, no analyze, no zip.
#
# Usage (from repo root):
#   ./tools/rocprof_isa_only.sh
#   WORKLOAD_NAME=attn_isa1 ./tools/rocprof_isa_only.sh
#   ROCPROF_UI_ATT_LIBRARY_PATH=/path/to/decoder ./tools/rocprof_isa_only.sh
#   PYTHON=/path/to/python ./tools/rocprof_isa_only.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Prepend conda lib so rocprofiler-sdk env reset doesn't drop libstdc++.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Auto-detect librocprof-trace-decoder.so for ATT.
if [[ -z "${ROCPROF_UI_ATT_LIBRARY_PATH:-}" ]]; then
  _decoder_found=""
  for _d in /opt/rocm/lib /opt/rocm-*/lib; do
    if [[ -f "${_d}/librocprof-trace-decoder.so" ]]; then
      _decoder_found="${_d}"; break
    fi
  done
  if [[ -z "${_decoder_found}" ]]; then
    _att_default="$(cd "${ROOT}/.." && pwd)/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"
    [[ -f "${_att_default}/librocprof-trace-decoder.so" ]] && _decoder_found="${_att_default}"
  fi
  if [[ -n "${_decoder_found}" ]]; then
    ROCPROF_UI_ATT_LIBRARY_PATH="${_decoder_found}"
    echo "Auto-detected decoder: ${_decoder_found}/librocprof-trace-decoder.so"
  fi
fi

# Resolve Python interpreter.
PYTHON="${PYTHON:-python}"
if [[ "${PYTHON}" == "python" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
if ! PYTHON="$(command -v "${PYTHON}")"; then
  echo "error: PYTHON not found" >&2; exit 1
fi

if ! command -v rocprofv3 &>/dev/null; then
  echo "error: rocprofv3 not found on PATH" >&2; exit 1
fi

# Build the kernel shared library with debug info if not already built.
if ! compgen -G "${ROOT}/libattention"*.so >/dev/null; then
  echo "Building libattention*.so..."
  make -C "${ROOT}"
fi

# Workload paths.
ROC_OUT_DIR="${ROC_OUT_DIR:-${ROOT}/rocprof_out}"
WORKLOAD_NAME="${WORKLOAD_NAME:-$(date +%Y%m%d_%H%M%S)}"
WORKLOAD_PATH="${ROC_OUT_DIR}/${WORKLOAD_NAME}"
export WORKLOAD_PATH
mkdir -p "${WORKLOAD_PATH}"

# Locate ISA HTML generator — prefer local copy, fall back to sibling conv3amd repo.
FORMAT_ISA_PY="${SCRIPT_DIR}/rocprof_att_stats_to_isa_html.py"
if [[ ! -f "${FORMAT_ISA_PY}" ]]; then
  _sibling="$(cd "${ROOT}/.." && pwd)/conv3amd/rocprof_att_stats_to_isa_html.py"
  [[ -f "${_sibling}" ]] && FORMAT_ISA_PY="${_sibling}"
fi

# Generate ISA HTML from ATT stats CSVs.
generate_isa_html() {
  local _utt="${ROCPROF_UI_TRACE_DIR:-${WORKLOAD_PATH}/ui_thread_trace}"
  [[ -f "${FORMAT_ISA_PY}" ]] || { echo "warning: ${FORMAT_ISA_PY} not found; skipping ISA HTML" >&2; return 0; }
  [[ -d "${_utt}" ]] || return 0
  local -a _files
  shopt -s nullglob
  _files=( "${_utt}"/stats_ui_output_agent_*_dispatch_*.csv )
  shopt -u nullglob
  (( ${#_files[@]} )) || return 0
  echo "Generating ISA HTML (${#_files[@]} stats CSV(s))..."
  "${PYTHON}" "${FORMAT_ISA_PY}" "${_files[@]}" || echo "warning: ISA HTML generation failed" >&2
}

export ROC_PROFILE_PYTHON="${PYTHON}"
export ROC_PROFILE_SCRIPT="${ROOT}/runner.py"
if [[ -z "${ROC_PROFILE_LD_PREFIX:-}" && -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export ROC_PROFILE_LD_PREFIX="${CONDA_PREFIX}/lib"
fi

if [[ -n "${ROC_PROFILE_RUNNER_ARGS:-}" ]]; then
  read -r -a ROC_PROFILE_RUNNER_ARGV <<<"${ROC_PROFILE_RUNNER_ARGS}"
else
  ROC_PROFILE_RUNNER_ARGV=()
fi

# Write a temp app launcher (rocprofiler-sdk execs a new process and resets LD_LIBRARY_PATH,
# so we need a real file to re-prepend the conda lib path before running runner.py).
_APP_SH="${SCRIPT_DIR}/.rocprof_app_$$.sh"
trap 'rm -f "${_APP_SH}"' EXIT
cat > "${_APP_SH}" <<'APPSH'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${ROC_PROFILE_LD_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${ROC_PROFILE_LD_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
exec "${ROC_PROFILE_PYTHON}" "${ROC_PROFILE_SCRIPT}" "$@"
APPSH
chmod +x "${_APP_SH}"

# --- ATT (Advanced Thread Trace) ---
RCV_UI_PARENT="${ROCPROF_UI_TRACE_DIR:-${WORKLOAD_PATH}/ui_thread_trace}"
mkdir -p "${RCV_UI_PARENT}"
echo "=== rocprofv3 ATT -> ${RCV_UI_PARENT} ==="
# --att-serialize-all is on by default on CDNA3 for reliable capture; ROCPROF_UI_ATT_SERIALIZE_ALL=0 to disable.
_rv3=(rocprofv3 --advanced-thread-trace 1 --kernel-trace 1 --att-serialize-all 1 -d "${RCV_UI_PARENT}")
if [[ -n "${ROCPROF_UI_ATT_LIBRARY_PATH:-}" ]]; then
  read -r -a _att_lib_dirs <<<"${ROCPROF_UI_ATT_LIBRARY_PATH//:/ }"
  _rv3+=(--att-library-path "${_att_lib_dirs[@]}")
fi
[[ -n "${ROCPROF_UI_ATT_TARGET_CU:-}" ]]   && _rv3+=(--att-target-cu  "${ROCPROF_UI_ATT_TARGET_CU}")
[[ -n "${ROCPROF_UI_ATT_GPU_INDEX:-}" ]]   && _rv3+=(--att-gpu-index  "${ROCPROF_UI_ATT_GPU_INDEX}")
[[ "${ROCPROF_UI_ATT_SERIALIZE_ALL:-1}" == "0" ]] && _rv3=(${_rv3[@]/--att-serialize-all 1/})

if ! "${_rv3[@]}" -- bash "${_APP_SH}" "${ROC_PROFILE_RUNNER_ARGV[@]}"; then
  echo "warning: rocprofv3 ATT failed (missing librocprof-trace-decoder.so?)" >&2
  echo "  Set ROCPROF_UI_ATT_LIBRARY_PATH or install ROCprof Trace Decoder." >&2
  exit 1
fi

_ui_dirs="$(find "${RCV_UI_PARENT}" -type d -name 'ui_output_agent_*' 2>/dev/null | head -5 || true)"
if [[ -n "${_ui_dirs}" ]]; then
  echo "  RCV import dirs:"; while IFS= read -r _d; do [[ -n "${_d}" ]] && echo "    ${_d}"; done <<<"${_ui_dirs}"
else
  echo "warning: no ui_output_agent_* found under ${RCV_UI_PARENT}" >&2
fi

# --- ISA HTML ---
generate_isa_html

echo "Done: ${WORKLOAD_PATH}"
echo "  Open ${RCV_UI_PARENT}/stats_*_isa.html in a browser."
