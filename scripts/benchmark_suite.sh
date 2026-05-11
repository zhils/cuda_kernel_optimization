#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-}"

if [[ -z "${TIER}" ]]; then
  echo "usage: bash scripts/benchmark_suite.sh <smoke|full|profile>"
  exit 1
fi

case "${TIER}" in
  smoke)
    echo "[suite] tier=smoke"
    bash "${ROOT_DIR}/scripts/run_correctness_regression.sh"
    ;;
  full)
    echo "[suite] tier=full"
    eval "$(python3 "${ROOT_DIR}/scripts/collect_env_manifest.py" --results-root "${ROOT_DIR}/data/results" --emit-shell)"
    export RUN_ID
    bash "${ROOT_DIR}/scripts/run_correctness_regression.sh"
    bash "${ROOT_DIR}/scripts/run_benchmark_repro.sh"
    AUTO_TUNE_UPDATE="${AUTO_TUNE_UPDATE:-1}"
    AUTO_TUNE_CACHE="${AUTO_TUNE_CACHE:-${ROOT_DIR}/data/baselines/autotune_cache.json}"
    if [[ "${AUTO_TUNE_UPDATE}" == "1" ]]; then
      python3 "${ROOT_DIR}/scripts/update_autotune_cache.py" \
        --summary "${ROOT_DIR}/data/results/runs/${RUN_ID}/summary_standardized.csv" \
        --output "${AUTO_TUNE_CACHE}"
    else
      echo "[suite] AUTO_TUNE_UPDATE disabled"
    fi
    PERF_GATE="${PERF_GATE:-1}"
    PERF_GATE_BASELINE="${PERF_GATE_BASELINE:-${ROOT_DIR}/data/baselines/perf_golden.csv}"
    PERF_GATE_WARN_MS_REGRESS_PCT="${PERF_GATE_WARN_MS_REGRESS_PCT:-8}"
    PERF_GATE_FAIL_MS_REGRESS_PCT="${PERF_GATE_FAIL_MS_REGRESS_PCT:-15}"
    PERF_GATE_WARN_THROUGHPUT_DROP_PCT="${PERF_GATE_WARN_THROUGHPUT_DROP_PCT:-5}"
    PERF_GATE_FAIL_THROUGHPUT_DROP_PCT="${PERF_GATE_FAIL_THROUGHPUT_DROP_PCT:-10}"
    PERF_GATE_REPORT="${ROOT_DIR}/data/results/runs/${RUN_ID}/performance_regression_check.csv"
    PERF_GATE_MD="${ROOT_DIR}/data/results/runs/${RUN_ID}/performance_gate_summary.md"
    if [[ "${PERF_GATE}" == "1" ]]; then
      if [[ -f "${PERF_GATE_BASELINE}" ]]; then
        python3 "${ROOT_DIR}/scripts/check_performance_regression.py" \
          --summary "${ROOT_DIR}/data/results/runs/${RUN_ID}/summary_standardized.csv" \
          --baseline "${PERF_GATE_BASELINE}" \
          --output "${PERF_GATE_REPORT}" \
          --warn-ms-regress-pct "${PERF_GATE_WARN_MS_REGRESS_PCT}" \
          --fail-ms-regress-pct "${PERF_GATE_FAIL_MS_REGRESS_PCT}" \
          --warn-throughput-drop-pct "${PERF_GATE_WARN_THROUGHPUT_DROP_PCT}" \
          --fail-throughput-drop-pct "${PERF_GATE_FAIL_THROUGHPUT_DROP_PCT}" \
          --markdown-output "${PERF_GATE_MD}"
      else
        echo "[suite][warn] PERF_GATE=1 but baseline missing: ${PERF_GATE_BASELINE}"
        echo "[suite][warn] generate baseline via scripts/update_performance_baseline.py"
      fi
    else
      echo "[suite] PERF_GATE disabled"
    fi
    python3 "${ROOT_DIR}/scripts/generate_run_report.py" \
      --run-dir "${ROOT_DIR}/data/results/runs/${RUN_ID}" \
      --performance-summary "${ROOT_DIR}/data/results/runs/${RUN_ID}/summary_standardized.csv" \
      --performance-gate-summary "${PERF_GATE_REPORT}" \
      --regression-summary "${ROOT_DIR}/data/results/runs/${RUN_ID}/regression_summary.csv" \
      --ncu-summary "${ROOT_DIR}/data/ncu_reports/${RUN_ID}/ncu_summary.csv" \
      --output "${ROOT_DIR}/data/results/runs/${RUN_ID}/report.md"
    ;;
  profile)
    echo "[suite] tier=profile"
    eval "$(python3 "${ROOT_DIR}/scripts/collect_env_manifest.py" --results-root "${ROOT_DIR}/data/results" --emit-shell)"
    export RUN_ID
    bash "${ROOT_DIR}/run_ncu_all.sh"
    bash "${ROOT_DIR}/run_ncu_stall.sh"
    bash "${ROOT_DIR}/run_ncu_roofline.sh"
    python3 "${ROOT_DIR}/scripts/parse_ncu_reports.py" \
      --run-id "${RUN_ID}" \
      --ncu-root "${ROOT_DIR}/data/ncu_reports/${RUN_ID}" \
      --output "${ROOT_DIR}/data/ncu_reports/${RUN_ID}/ncu_summary.csv"
    python3 "${ROOT_DIR}/scripts/generate_run_report.py" \
      --run-dir "${ROOT_DIR}/data/results/runs/${RUN_ID}" \
      --ncu-summary "${ROOT_DIR}/data/ncu_reports/${RUN_ID}/ncu_summary.csv" \
      --output "${ROOT_DIR}/data/results/runs/${RUN_ID}/report.md"
    ;;
  *)
    echo "unknown tier: ${TIER}"
    echo "usage: bash scripts/benchmark_suite.sh <smoke|full|profile>"
    exit 1
    ;;
esac
