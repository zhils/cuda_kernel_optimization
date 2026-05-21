#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/data/results"
AUDIT_PATH="${ROOT_DIR}/docs/operator_test_coverage_audit.md"
RUN_LOG_DIR="${ROOT_DIR}/data/results/supplement_logs"
mkdir -p "${RUN_LOG_DIR}"

echo "[supplement] step1/3: correctness regression"
set +e
RANDOM_GEMM_CASES=1 \
bash "${ROOT_DIR}/scripts/run_correctness_regression.sh" \
  > "${RUN_LOG_DIR}/supplement_correctness.log" 2>&1
REG_EXIT=$?
set -e
if [[ "${REG_EXIT}" -ne 0 ]]; then
  echo "[supplement][warn] regression failed (exit=${REG_EXIT}), continue audit"
fi

echo "[supplement] step2/3: build + run"
cmake --build "${BUILD_DIR}" --target \
  gemm_v3 gemm_fp16 rmsnorm_v3 fused_conv1d_silu_v3 \
  > "${RUN_LOG_DIR}/supplement_build.log" 2>&1

for bin in gemm_v3 gemm_fp16 rmsnorm_v3 fused_conv1d_silu_v3; do
  echo "[supplement] running ${bin}"
  "${BUILD_DIR}/bin/${bin}" > "${RUN_LOG_DIR}/${bin}.log" 2>&1
done

echo "[supplement] step3/3: audit"
python3 "${ROOT_DIR}/scripts/audit_operator_tests.py" \
  --results-dir "${RESULTS_DIR}" \
  --output "${AUDIT_PATH}"

echo "[supplement] done → ${AUDIT_PATH}"
