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
DISPATCH_ARCH="${DISPATCH_ARCH:-sm120}"
DISPATCH_DTYPE="${DISPATCH_DTYPE:-any}"
DISPATCH_LAYOUT="${DISPATCH_LAYOUT:-any}"
DISPATCH_SHAPE_BUCKET="${DISPATCH_SHAPE_BUCKET:-any}"
mkdir -p "${LOG_DIR}"

mapfile -t TARGETS < <(
  python3 "${ROOT_DIR}/scripts/kernel_dispatch.py" \
    --catalog "${CATALOG_PATH}" \
    --tier smoke \
    --autotune-cache "${CACHE_PATH}" \
    --arch "${DISPATCH_ARCH}" \
    --dtype "${DISPATCH_DTYPE}" \
    --layout "${DISPATCH_LAYOUT}" \
    --shape-bucket "${DISPATCH_SHAPE_BUCKET}"
)

declare -a CASE_BACKUPS=()

backup_and_generate_cases() {
  local target_csv="$1"
  shift
  local backup_csv="${target_csv}.backup.${RUN_ID}.csv"
  cp "${target_csv}" "${backup_csv}"
  CASE_BACKUPS+=("${target_csv}|${backup_csv}")
  python3 "${ROOT_DIR}/scripts/generate_random_matrix_cases.py" --output "${target_csv}" "$@"
}

if [[ "${RANDOM_GEMM_CASES:-0}" == "1" ]]; then
  backup_and_generate_cases "${ROOT_DIR}/data/gemm/test_cases.csv" \
    --count "${RANDOM_GEMM_CASES_COUNT:-12}" \
    --seed "${RANDOM_GEMM_CASES_SEED:-20260511}" \
    --min-rows 96 --max-rows 1536 \
    --min-cols 96 --max-cols 1536 \
    --align 16 \
    --max-elements "${RANDOM_GEMM_CASES_MAX_ELEMENTS:-6291456}" \
    --square-only
  echo "[regression] RANDOM_GEMM_CASES enabled"
fi

if [[ "${RANDOM_SOFTMAX_CASES:-0}" == "1" ]]; then
  backup_and_generate_cases "${ROOT_DIR}/data/softmax/test_cases.csv" \
    --count "${RANDOM_SOFTMAX_CASES_COUNT:-12}" \
    --seed "${RANDOM_SOFTMAX_CASES_SEED:-20260512}" \
    --min-rows 64 --max-rows 4096 \
    --min-cols 64 --max-cols 2048 \
    --align 16 \
    --max-elements "${RANDOM_SOFTMAX_CASES_MAX_ELEMENTS:-8388608}"
  echo "[regression] RANDOM_SOFTMAX_CASES enabled"
fi

if [[ "${RANDOM_QPATH_CASES:-0}" == "1" ]]; then
  backup_and_generate_cases "${ROOT_DIR}/data/q_path_fusion/test_cases.csv" \
    --count "${RANDOM_QPATH_CASES_COUNT:-10}" \
    --seed "${RANDOM_QPATH_CASES_SEED:-20260513}" \
    --min-rows 64 --max-rows 1024 \
    --min-cols 64 --max-cols 1024 \
    --align 16 \
    --max-elements "${RANDOM_QPATH_CASES_MAX_ELEMENTS:-2097152}" \
    --square-only
  echo "[regression] RANDOM_QPATH_CASES enabled"
fi

restore_case_files() {
  for pair in "${CASE_BACKUPS[@]}"; do
    local target="${pair%%|*}"
    local backup="${pair##*|}"
    if [[ -f "${backup}" ]]; then
      mv "${backup}" "${target}"
      echo "[regression] restored ${target}"
    fi
  done
}
trap restore_case_files EXIT

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
