#!/usr/bin/env bash
# RTX 5060 Ti 性能复测脚本 — 详见 RETEST_5060Ti.md
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DATA_ROOT="${CUDA_DATA_ROOT:-${BUILD_DIR}/data}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${DATA_ROOT}/results/retest_5060ti/${STAMP}"
MODE="${1:-perf}"

mkdir -p "${LOG_DIR}"

run_bin() {
  local name="$1"
  local bin="${BUILD_DIR}/bin/${name}"
  if [[ ! -x "${bin}" ]]; then
    echo "[retest][skip] missing ${name}"
    return 0
  fi
  echo "[retest] running ${name} ..."
  "${bin}" > "${LOG_DIR}/${name}.log" 2>&1 || true
}

case "${MODE}" in
  perf)
    echo "[retest] mode=perf → logs: ${LOG_DIR}"
    if [[ ! -d "${BUILD_DIR}/bin" ]]; then
      echo "[retest] build dir missing, run cmake first"
      exit 1
    fi
    for b in gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4 gemm_fp16 \
             gemm_cublas_ref gemm_cublas_fp16 \
             rmsnorm_v0 rmsnorm_v1 rmsnorm_v2 rmsnorm_v3 rmsnorm_cub_ref \
             fused_conv1d_silu_v0 fused_conv1d_silu_v1 fused_conv1d_silu_v2 fused_conv1d_silu_v3; do
      run_bin "${b}"
    done
    echo "[retest] done. Fill README from logs in ${LOG_DIR}"
    ;;
  ncu-basic)
    echo "[retest] mode=ncu-basic"
    RUN_ID="retest_${STAMP}" bash "${ROOT_DIR}/run_ncu_all.sh"
    ;;
  ncu-stall)
    echo "[retest] mode=ncu-stall"
    RUN_ID="retest_${STAMP}" bash "${ROOT_DIR}/run_ncu_stall.sh"
    ;;
  ncu-stall-quick)
    echo "[retest] mode=ncu-stall-quick (demo core)"
    RUN_ID="retest_${STAMP}" NCU_STALL_QUICK=1 bash "${ROOT_DIR}/run_ncu_stall.sh"
    ;;
  *)
    echo "usage: bash scripts/run_retest_5060ti.sh {perf|ncu-basic|ncu-stall|ncu-stall-quick}"
    exit 1
    ;;
esac
