#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_ROOT="${ROOT_DIR}/data/results"
eval "$(python3 "${ROOT_DIR}/scripts/collect_env_manifest.py" --results-root "${RESULTS_ROOT}" --run-id "${RUN_ID:-}" --emit-shell)"
LOG_DIR="${RUN_DIR}/regression_logs"
SUMMARY_PATH="${RUN_DIR}/regression_summary.csv"
CACHE_PATH="${ROOT_DIR}/data/baselines/autotune_cache.json"
CATALOG_PATH="${ROOT_DIR}/configs/kernel_catalog.json"
mkdir -p "${LOG_DIR}"

mapfile -t TARGETS < <(
  python3 "${ROOT_DIR}/scripts/kernel_dispatch.py" \
    --catalog "${CATALOG_PATH}" \
    --tier smoke \
    --autotune-cache "${CACHE_PATH}"
)

GEMM_CASES_PATH="${ROOT_DIR}/data/gemm/test_cases.csv"
GEMM_CASES_BACKUP=""
if [[ "${RANDOM_GEMM_CASES:-0}" == "1" ]]; then
  GEMM_CASES_BACKUP="${ROOT_DIR}/data/gemm/test_cases.backup.${RUN_ID}.csv"
  cp "${GEMM_CASES_PATH}" "${GEMM_CASES_BACKUP}"
  python3 "${ROOT_DIR}/scripts/generate_random_gemm_cases.py" \
    --output "${GEMM_CASES_PATH}" \
    --count "${RANDOM_GEMM_CASES_COUNT:-12}" \
    --seed "${RANDOM_GEMM_CASES_SEED:-20260511}"
  echo "[regression] RANDOM_GEMM_CASES enabled, backup=${GEMM_CASES_BACKUP}"
fi

restore_gemm_cases() {
  if [[ -n "${GEMM_CASES_BACKUP}" && -f "${GEMM_CASES_BACKUP}" ]]; then
    mv "${GEMM_CASES_BACKUP}" "${GEMM_CASES_PATH}"
    echo "[regression] restored ${GEMM_CASES_PATH}"
  fi
}
trap restore_gemm_cases EXIT

echo "[regression] building ${#TARGETS[@]} targets..."
cmake --build "${BUILD_DIR}" --target "${TARGETS[@]}"

failed=0
for bin in "${TARGETS[@]}"; do
  log_file="${LOG_DIR}/${bin}.log"
  echo "[regression] running ${bin}"
  "${BUILD_DIR}/bin/${bin}" > "${log_file}" 2>&1 || failed=1

  if grep -q "FAIL" "${log_file}"; then
    echo "[regression] FAIL detected in ${bin}, see ${log_file}"
    failed=1
  fi
done

python3 "${ROOT_DIR}/scripts/summarize_regression_logs.py" \
  --logs-dir "${LOG_DIR}" \
  --results-dir "${RESULTS_ROOT}" \
  --output "${SUMMARY_PATH}" \
  --run-id "${RUN_ID}"

cp "${SUMMARY_PATH}" "${RESULTS_ROOT}/regression_summary.csv"
python3 "${ROOT_DIR}/scripts/generate_run_report.py" \
  --run-dir "${RUN_DIR}" \
  --regression-summary "${SUMMARY_PATH}" \
  --output "${RUN_DIR}/report.md"

echo "[regression] summary written to:"
echo "  ${SUMMARY_PATH}"
echo "  ${RESULTS_ROOT}/regression_summary.csv (latest copy)"

if [[ "${failed}" -ne 0 ]]; then
  echo "[regression] completed with failures."
  exit 1
fi

echo "[regression] all selected targets passed."
