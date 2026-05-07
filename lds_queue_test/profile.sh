#!/usr/bin/env bash
# Profile lds_queue_test with rocprofv3 ATT + PMC bank-conflict counters.
# Generates ISA HTML with HIP source matching.
#
# Usage:
#   ./profile.sh             # default: variant 0 (b128)
#   VARIANT=1 ./profile.sh   # variant 1 (b64)
#   N_ITERS=64 ./profile.sh  # fewer loop iters (smaller trace)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VARIANT="${VARIANT:-0}"
N_ITERS="${N_ITERS:-64}"        # small loop — just enough to get clean ATT samples
N_BLOCKS="${N_BLOCKS:-304}"     # exactly MI300X CU count (1 CTA per CU)
WARMUP="${WARMUP:-2}"
PYTHON="${PYTHON:-/home/tarik/miniconda3/envs/lite_attention/bin/python}"
DECODER_LIB="${DECODER_LIB:-/opt/rocm-7.0.0/lib}"

# ISA HTML formatter (reused from cdna3-attention/).
FORMAT_ISA_PY="${SCRIPT_DIR}/../rocprof_att_stats_to_isa_html.py"

# Build with -g for ATT source matching.
if [[ ! -f "${SCRIPT_DIR}/liblds_queue_test.so" || "${SCRIPT_DIR}/kernel.hip" -nt "${SCRIPT_DIR}/liblds_queue_test.so" ]]; then
  echo "=== build ==="
  make -C "${SCRIPT_DIR}"
fi

OUT="${SCRIPT_DIR}/profile_v${VARIANT}_$(date +%H%M%S)"
mkdir -p "${OUT}"

# Conda lib path so rocprofv3 can find libstdc++ from this env.
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# --- 1. PMC counters (bank conflicts, LDS port active) ---
echo "=== PMC counters → ${OUT}/pmc/ ==="
cat > "${OUT}/pmc.txt" <<EOF
pmc: SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_INSTS_VALU
EOF

mkdir -p "${OUT}/pmc"
rocprofv3 -i "${OUT}/pmc.txt" -d "${OUT}/pmc" --output-format csv -- \
  "${PYTHON}" "${SCRIPT_DIR}/runner.py" \
    --variant "${VARIANT}" --n-blocks "${N_BLOCKS}" --n-iters "${N_ITERS}" \
    --warmup-iters "${WARMUP}" --bench-iters 1 \
  2>&1 | tail -3

# --- 2. ATT (Advanced Thread Trace) for per-instruction latency ---
echo "=== ATT → ${OUT}/att/ ==="
mkdir -p "${OUT}/att"
rocprofv3 \
  --advanced-thread-trace 1 \
  --kernel-trace 1 \
  --att-serialize-all 1 \
  --att-library-path "${DECODER_LIB}" \
  -d "${OUT}/att" \
  -- "${PYTHON}" "${SCRIPT_DIR}/runner.py" \
    --variant "${VARIANT}" --n-blocks "${N_BLOCKS}" --n-iters "${N_ITERS}" \
    --warmup-iters "${WARMUP}" --bench-iters 1 \
  2>&1 | tail -5

# --- 3. ISA HTML (source-matched) ---
if [[ -f "${FORMAT_ISA_PY}" ]]; then
  shopt -s nullglob
  for csv in "${OUT}/att"/stats_ui_output_agent_*_dispatch_*.csv "${OUT}/att"/**/stats_ui_output_agent_*_dispatch_*.csv; do
    [[ -f "${csv}" ]] || continue
    echo "  ISA HTML: ${csv}"
    "${PYTHON}" "${FORMAT_ISA_PY}" "${csv}" || echo "  warning: HTML gen failed for ${csv}"
  done
  shopt -u nullglob
else
  echo "warning: ${FORMAT_ISA_PY} not found — skipping ISA HTML"
fi

# --- 4. Quick PMC summary ---
echo
echo "=== PMC summary ==="
"${PYTHON}" - "${OUT}/pmc" <<'PY'
import csv, glob, sys
root = sys.argv[1]
sums = {}
for path in glob.glob(f"{root}/**/*counter_collection.csv", recursive=True):
    with open(path) as f:
        for row in csv.DictReader(f):
            if "lds_queue_test" not in row.get("Kernel_Name",""): continue
            try: v = float(row["Counter_Value"])
            except (KeyError, ValueError): continue
            cn = row["Counter_Name"]
            sums[cn] = sums.get(cn, 0.0) + v
for k in sorted(sums):
    print(f"  {k:<32} {sums[k]:>16,.0f}")
PY

echo
echo "Profile written to: ${OUT}"
echo
echo "View ISA-source HTML:"
echo "  xdg-open ${OUT}/att/*_isa.html  (or open the .html file in any browser)"
echo
echo "Import ATT into RCV:"
echo "  ${OUT}/att/<gpu>/ui_output_agent_*_dispatch_*"
