#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_ROOT="${ROOT_DIR}/data/results"
eval "$(python3 "${ROOT_DIR}/scripts/collect_env_manifest.py" --results-root "${RESULTS_ROOT}" --run-id "${RUN_ID:-}" --emit-shell)"
LOG_DIR="${RUN_DIR}/regression_logs"
SUMMARY_PATH="${RUN_DIR}/regression_summary.csv"
mkdir -p "${LOG_DIR}"

# 统一小规模正确性回归。每个二进制都会打印 PASS/FAIL。
TARGETS=(
  gemm_v1
  gemm_v2
  gemm_v3
  gemm_v4
  gemm_fp16
  gemm_int8
  softmax_v1
  softmax_v2
  softmax_v3
  rmsnorm_v1
  rmsnorm_v2
  rmsnorm_v3
  flash_attention_v1
  flash_attention_v3
  fused_conv1d_silu_v1
  fused_conv1d_silu_v3
  fused_gated_delta_rule_v1
  fused_gated_delta_rule_v2
  fused_l2_norm_qk_v1
  fused_l2_norm_qk_v2
  fused_output_norm_gate_v1
  fused_output_norm_gate_v2
  q_path_fusion_v1
  q_path_fusion_v2
)

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
