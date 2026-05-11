#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/bin"
RUN_ID="${RUN_ID:-ncu_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${ROOT_DIR}/data/ncu_reports/${RUN_ID}/stall"
CACHE_PATH="${ROOT_DIR}/data/baselines/autotune_cache.json"
CATALOG_PATH="${ROOT_DIR}/configs/kernel_catalog.json"
DISPATCH_ARCH="${DISPATCH_ARCH:-sm120}"
DISPATCH_DTYPE="${DISPATCH_DTYPE:-any}"
DISPATCH_LAYOUT="${DISPATCH_LAYOUT:-any}"
DISPATCH_SHAPE_BUCKET="${DISPATCH_SHAPE_BUCKET:-any}"
mkdir -p "${OUT_DIR}"

STALL_METRICS="\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"

NCU="ncu --target-processes all --kernel-name-base demangled --print-summary per-kernel --metrics ${STALL_METRICS}"
ALLOW_NCU_FAIL="${ALLOW_NCU_FAIL:-0}"

mapfile -t TARGETS < <(
  python3 "${ROOT_DIR}/scripts/kernel_dispatch.py" \
    --catalog "${CATALOG_PATH}" \
    --tier profile_stall \
    --autotune-cache "${CACHE_PATH}" \
    --arch "${DISPATCH_ARCH}" \
    --dtype "${DISPATCH_DTYPE}" \
    --layout "${DISPATCH_LAYOUT}" \
    --shape-bucket "${DISPATCH_SHAPE_BUCKET}"
)

if [[ -n "${NCU_STALL_QUICK:-}" ]]; then
  TARGETS=(gemm_v3 rmsnorm_v3 softmax_v3)
fi

extract_top_stalls() {
  local file="$1"
  python3 - <<'PY' "${file}"
import re
import sys
path = sys.argv[1]
text = open(path, "r").read()
sections = text.split('Section: Command line profiler metrics')
if len(sections) < 2:
    print('  (no metric data)')
    raise SystemExit(0)
last = sections[-1]
metrics = {}
for line in last.splitlines():
    m = re.search(r'smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio\s+inst\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
    if m:
        name = m.group(1).replace('_', ' ').title()
        avg = float(m.group(4))
        if avg > 0:
            metrics[name] = avg
if not metrics:
    print('  (all zero or no stall data)')
    raise SystemExit(0)
sorted_m = sorted(metrics.items(), key=lambda x: -x[1])
total = sum(v for _, v in sorted_m)
for rank, (name, val) in enumerate(sorted_m[:5], 1):
    pct = val / total * 100
    print(f'  {rank}. {name}: {pct:.1f}%')
PY
}

run_target() {
  local t="$1"
  local out="${OUT_DIR}/${t}.txt"
  echo "=== ${t} ==="
  ${NCU} "${BIN}/${t}" > "${out}" 2>&1
  echo "--- ${t} Top 5 Warp Stall Reasons ---"
  extract_top_stalls "${out}"
  echo
}

for t in "${TARGETS[@]}"; do
  if [[ "${ALLOW_NCU_FAIL}" == "1" ]]; then
    run_target "${t}" || echo "[ncu][warn] stall profile failed target=${t}"
  else
    run_target "${t}"
  fi
done

echo "[ncu-stall] done run_id=${RUN_ID}"
echo "[ncu-stall] reports: ${OUT_DIR}"
