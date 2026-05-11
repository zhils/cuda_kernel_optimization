#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/bin"
RUN_ID="${RUN_ID:-ncu_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${ROOT_DIR}/data/ncu_reports/${RUN_ID}/roofline"
mkdir -p "${OUT_DIR}"

NCU="ncu --set roofline --target-processes all --kernel-name-base demangled"
ALLOW_NCU_FAIL="${ALLOW_NCU_FAIL:-0}"

TARGETS=(
  gemm_v3
  gemm_fp16
  rmsnorm_v3
  softmax_v3
  fused_conv1d_silu_v3
  flash_attention_v3
  fused_gated_delta_rule_v2
  fused_l2_norm_qk_v2
  fused_output_norm_gate_v2
  q_path_fusion_v2
)

if [[ -n "${NCU_ROOF_QUICK:-}" ]]; then
  TARGETS=(gemm_v3 rmsnorm_v3 softmax_v3)
fi

run_target() {
  local t="$1"
  echo "=== ${t} ==="
  ${NCU} -o "${OUT_DIR}/roofline_${t}" "${BIN}/${t}" > "${OUT_DIR}/${t}.txt" 2>&1
}

for t in "${TARGETS[@]}"; do
  if [[ "${ALLOW_NCU_FAIL}" == "1" ]]; then
    run_target "${t}" || echo "[ncu][warn] roofline failed target=${t}"
  else
    run_target "${t}"
  fi
done

echo "[ncu-roofline] done run_id=${RUN_ID}"
echo "[ncu-roofline] reports: ${OUT_DIR}"
