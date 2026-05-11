#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_rows(path: Path):
    if not path or not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def summarize_regression(rows):
    if not rows:
        return "无回归数据。"
    fail = sum(1 for r in rows if r.get("status") == "FAIL")
    skip = sum(1 for r in rows if r.get("status") == "SKIP")
    not_run = sum(1 for r in rows if r.get("status") == "NOT_RUN")
    passed = sum(1 for r in rows if r.get("status") == "PASS")
    return f"PASS={passed}, FAIL={fail}, SKIP={skip}, NOT_RUN={not_run}"


def summarize_perf(rows):
    if not rows:
        return "无性能汇总数据。"
    status_cnt = {}
    for r in rows:
        k = r.get("verify_status", "NOT_RUN")
        status_cnt[k] = status_cnt.get(k, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(status_cnt.items())]
    return ", ".join(parts)


def summarize_ncu(rows):
    if not rows:
        return "无 NCU 结构化汇总。"
    return f"targets={len(rows)}"


def summarize_perf_gate(rows):
    if not rows:
        return "无性能门禁数据。"
    status_cnt = {}
    for r in rows:
        k = r.get("status", "PASS")
        status_cnt[k] = status_cnt.get(k, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(status_cnt.items())]
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate run-level markdown report.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--regression-summary", default="")
    parser.add_argument("--performance-summary", default="")
    parser.add_argument("--performance-gate-summary", default="")
    parser.add_argument("--ncu-summary", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    reg_path = Path(args.regression_summary) if args.regression_summary else None
    perf_path = Path(args.performance_summary) if args.performance_summary else None
    perf_gate_path = Path(args.performance_gate_summary) if args.performance_gate_summary else None
    ncu_path = Path(args.ncu_summary) if args.ncu_summary else None
    out_path = Path(args.output)

    reg_rows = read_rows(reg_path) if reg_path else []
    perf_rows = read_rows(perf_path) if perf_path else []
    perf_gate_rows = read_rows(perf_gate_path) if perf_gate_path else []
    ncu_rows = read_rows(ncu_path) if ncu_path else []

    lines = [
        "# Run Report",
        "",
        f"- run_dir: `{run_dir}`",
        f"- regression_summary: `{reg_path}`" if reg_path else "- regression_summary: `N/A`",
        f"- performance_summary: `{perf_path}`" if perf_path else "- performance_summary: `N/A`",
        f"- performance_gate_summary: `{perf_gate_path}`"
        if perf_gate_path
        else "- performance_gate_summary: `N/A`",
        f"- ncu_summary: `{ncu_path}`" if ncu_path else "- ncu_summary: `N/A`",
        "",
        "## Regression",
        summarize_regression(reg_rows),
        "",
        "## Performance",
        summarize_perf(perf_rows),
        "",
        "## NCU",
        summarize_ncu(ncu_rows),
        "",
        "## Performance Gate",
        summarize_perf_gate(perf_gate_rows),
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[run-report] wrote report -> {out_path}")


if __name__ == "__main__":
    main()
