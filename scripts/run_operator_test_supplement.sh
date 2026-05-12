#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/data/results"
AUDIT_PATH="${ROOT_DIR}/docs/operator_test_coverage_audit.md"
RUN_LOG_DIR="${ROOT_DIR}/data/results/supplement_logs"
mkdir -p "${RUN_LOG_DIR}"

echo "[supplement] step1/4: randomized correctness regression"
set +e
RANDOM_GEMM_CASES=1 \
RANDOM_SOFTMAX_CASES=1 \
RANDOM_QPATH_CASES=1 \
bash "${ROOT_DIR}/scripts/run_correctness_regression.sh" \
  > "${RUN_LOG_DIR}/supplement_correctness.log" 2>&1
REG_EXIT=$?
set -e
if [[ "${REG_EXIT}" -ne 0 ]]; then
  echo "[supplement][warn] randomized correctness regression failed (exit=${REG_EXIT}), continue to produce audit"
fi

echo "[supplement] step2/4: build representative performance targets"
cmake --build "${BUILD_DIR}" --target \
  gemm_v3 gemm_fp16 softmax_v3 rmsnorm_v3 flash_attention_v4 \
  fused_conv1d_silu_v3 fused_gated_delta_rule_v2 fused_l2_norm_qk_v2 \
  fused_output_norm_gate_v2 q_path_fusion_v2 \
  > "${RUN_LOG_DIR}/supplement_build.log" 2>&1

echo "[supplement] step3/4: run representative performance targets"
for bin in \
  gemm_v3 gemm_fp16 softmax_v3 rmsnorm_v3 flash_attention_v4 \
  fused_conv1d_silu_v3 fused_gated_delta_rule_v2 fused_l2_norm_qk_v2 \
  fused_output_norm_gate_v2 q_path_fusion_v2; do
  echo "[supplement] running ${bin}"
  "${BUILD_DIR}/bin/${bin}" > "${RUN_LOG_DIR}/${bin}.log" 2>&1
done

echo "[supplement] step4/4: generate coverage audit"
python3 "${ROOT_DIR}/scripts/audit_operator_tests.py" \
  --results-dir "${RESULTS_DIR}" \
  --output "${AUDIT_PATH}"

echo "[supplement] done"
echo "  - audit: ${AUDIT_PATH}"
echo "  - logs: ${RUN_LOG_DIR}"
