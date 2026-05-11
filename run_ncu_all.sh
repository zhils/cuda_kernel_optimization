#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/bin"
RUN_ID="${RUN_ID:-ncu_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${ROOT_DIR}/data/ncu_reports/${RUN_ID}"
TEXT_DIR="${OUT_DIR}/text"
CACHE_PATH="${ROOT_DIR}/data/baselines/autotune_cache.json"
CATALOG_PATH="${ROOT_DIR}/configs/kernel_catalog.json"
mkdir -p "${TEXT_DIR}"

NCU="ncu --set basic --target-processes all --kernel-name-base demangled --print-summary per-kernel"
ALLOW_NCU_FAIL="${ALLOW_NCU_FAIL:-0}"

mapfile -t TARGETS < <(
  python3 "${ROOT_DIR}/scripts/kernel_dispatch.py" \
    --catalog "${CATALOG_PATH}" \
    --tier profile_all \
    --autotune-cache "${CACHE_PATH}"
)

if [[ -n "${NCU_ALL_QUICK:-}" ]]; then
  TARGETS=(gemm_v3 rmsnorm_v3 softmax_v3)
fi

run_target() {
  local t="$1"
  local out="${TEXT_DIR}/${t}.txt"
  echo "=== ${t} ==="

  if [[ "${t}" == gemm_* ]]; then
    if ! ${NCU} "${BIN}/${t}" -c 1 -n 1 > "${out}" 2>&1; then
      return 1
    fi
  else
    if ! ${NCU} "${BIN}/${t}" > "${out}" 2>&1; then
      return 1
    fi
  fi
  grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)" "${out}" || true
}

for t in "${TARGETS[@]}"; do
  if [[ "${ALLOW_NCU_FAIL}" == "1" ]]; then
    run_target "${t}" || echo "[ncu][warn] failed target=${t}"
  else
    run_target "${t}"
  fi
done

echo "[ncu] done run_id=${RUN_ID}"
echo "[ncu] text reports: ${TEXT_DIR}"
