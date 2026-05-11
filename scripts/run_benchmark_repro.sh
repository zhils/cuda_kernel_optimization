#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_ROOT="${ROOT_DIR}/data/results"
eval "$(python3 "${ROOT_DIR}/scripts/collect_env_manifest.py" --results-root "${RESULTS_ROOT}" --run-id "${RUN_ID:-}" --emit-shell)"
LOG_DIR="${RUN_DIR}/repro_logs"
SUMMARY_PATH="${RUN_DIR}/summary_standardized.csv"
mkdir -p "${LOG_DIR}"

# 选择每个算子的一版主力实现 + 参考实现。
TARGETS=(
  gemm_v3
  gemm_v4
  gemm_fp16
  gemm_cublas_ref
  softmax_v3
  softmax_benchmark_all
  rmsnorm_v3
  rmsnorm_cub_ref
  flash_attention_v3
  flash_attention_v4
  fused_conv1d_silu_v3
  fused_gated_delta_rule_v2
  fused_l2_norm_qk_v2
  fused_output_norm_gate_v2
  q_path_fusion_v2
)

echo "[repro] environment:"
echo "  - build dir: ${BUILD_DIR}"
echo "  - logs: ${LOG_DIR}"
echo "  - run_id: ${RUN_ID}"
echo "  - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  - CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-unset}"

echo "[repro] building ${#TARGETS[@]} targets..."
cmake --build "${BUILD_DIR}" --target "${TARGETS[@]}"

for bin in "${TARGETS[@]}"; do
  log_file="${LOG_DIR}/${bin}.log"
  echo "[repro] running ${bin}"
  if [[ "${bin}" == "softmax_benchmark_all" ]]; then
    NCU_QUICK=1 "${BUILD_DIR}/bin/${bin}" > "${log_file}" 2>&1
  else
    "${BUILD_DIR}/bin/${bin}" > "${log_file}" 2>&1
  fi
done

python3 "${ROOT_DIR}/scripts/normalize_results.py" \
  --results-dir "${RESULTS_ROOT}" \
  --output "${SUMMARY_PATH}" \
  --run-id "${RUN_ID}" \
  --manifest "${MANIFEST_PATH}"

cp "${SUMMARY_PATH}" "${RESULTS_ROOT}/summary_standardized.csv"
python3 "${ROOT_DIR}/scripts/generate_run_report.py" \
  --run-dir "${RUN_DIR}" \
  --performance-summary "${SUMMARY_PATH}" \
  --regression-summary "${RUN_DIR}/regression_summary.csv" \
  --ncu-summary "${ROOT_DIR}/data/ncu_reports/${RUN_ID}/ncu_summary.csv" \
  --output "${RUN_DIR}/report.md"

echo "[repro] done. standardized summary:"
echo "  ${SUMMARY_PATH}"
echo "  ${RESULTS_ROOT}/summary_standardized.csv (latest copy)"
