#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


RESULT_CSV_RE = re.compile(r"Results saved to\s+(.+\.csv)")
VALID_STATUSES = {"PASS", "FAIL", "SKIP", "NOT_RUN"}


def parse_float(value: str):
    try:
        return float(value)
    except Exception:
        return None


def normalize_status(row):
    raw = row.get("verify_status", "") or row.get("check", "") or row.get("status", "")
    status = raw.strip().upper()
    if status in VALID_STATUSES:
        return status
    if status in ("", "N/A", "NA"):
        return "NOT_RUN"
    return status


def infer_result_csv(log_path: Path, results_dir: Path):
    text = log_path.read_text(errors="ignore")
    match = RESULT_CSV_RE.search(text)
    if match:
        rel = match.group(1).strip()
        maybe = (log_path.parents[2] / rel).resolve()
        if maybe.exists():
            return maybe
        maybe = (results_dir / Path(rel).name).resolve()
        if maybe.exists():
            return maybe

    target = log_path.stem
    fallback = (results_dir / f"{target}_results.csv").resolve()
    return fallback if fallback.exists() else None


def summarize_result_csv(csv_path: Path):
    if not csv_path or not csv_path.exists():
        return {
            "status": "NOT_RUN",
            "max_abs_err_worst": "",
            "pass_rows": 0,
            "fail_rows": 0,
            "skip_rows": 0,
            "not_run_rows": 0,
        }

    max_abs_err = None
    pass_rows = fail_rows = skip_rows = not_run_rows = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = normalize_status(row)
            if status == "PASS":
                pass_rows += 1
            elif status == "FAIL":
                fail_rows += 1
            elif status == "SKIP":
                skip_rows += 1
            else:
                not_run_rows += 1

            if status in ("PASS", "FAIL"):
                vals = []
                for key, value in row.items():
                    if key in ("max_abs_err", "max_abs_diff") or key.startswith("max_abs_err_") or key.startswith("max_abs_diff_"):
                        fv = parse_float(value)
                        if fv is not None:
                            vals.append(fv)
                if vals:
                    local_max = max(vals)
                    if max_abs_err is None or local_max > max_abs_err:
                        max_abs_err = local_max

    if fail_rows > 0:
        summary_status = "FAIL"
    elif pass_rows > 0:
        summary_status = "PASS"
    elif skip_rows > 0:
        summary_status = "SKIP"
    else:
        summary_status = "NOT_RUN"

    return {
        "status": summary_status,
        "max_abs_err_worst": "" if max_abs_err is None else f"{max_abs_err:.8g}",
        "pass_rows": pass_rows,
        "fail_rows": fail_rows,
        "skip_rows": skip_rows,
        "not_run_rows": not_run_rows,
    }


def summarize_log(log_path: Path, results_dir: Path):
    target = log_path.stem
    result_csv = infer_result_csv(log_path, results_dir)
    stats = summarize_result_csv(result_csv) if result_csv else {
        "status": "NOT_RUN",
        "max_abs_err_worst": "",
        "pass_rows": 0,
        "fail_rows": 0,
        "skip_rows": 0,
        "not_run_rows": 0,
    }

    # 兜底：若进程日志存在 FAIL 文本，标记失败（防止 CSV 缺失时漏报）
    text = log_path.read_text(errors="ignore")
    if "FAIL" in text and stats["status"] != "FAIL":
        stats["status"] = "FAIL"

    return {
        "target": target,
        "status": stats["status"],
        "max_abs_err_worst": stats["max_abs_err_worst"],
        "pass_rows": stats["pass_rows"],
        "fail_rows": stats["fail_rows"],
        "skip_rows": stats["skip_rows"],
        "not_run_rows": stats["not_run_rows"],
        "log_path": str(log_path),
        "result_csv_path": "" if result_csv is None else str(result_csv),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize regression logs into one CSV table.")
    parser.add_argument("--logs-dir", required=True, help="Path to regression log directory")
    parser.add_argument("--results-dir", required=True, help="Path to data/results directory")
    parser.add_argument("--output", required=True, help="Output summary CSV path")
    parser.add_argument("--run-id", default="", help="Run id for traceability")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    rows = []
    for log_path in sorted(logs_dir.glob("*.log")):
        row = summarize_log(log_path, results_dir)
        if args.run_id:
            row["run_id"] = args.run_id
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "target",
        "status",
        "max_abs_err_worst",
        "pass_rows",
        "fail_rows",
        "skip_rows",
        "not_run_rows",
        "log_path",
        "result_csv_path",
    ] if args.run_id else [
        "target",
        "status",
        "max_abs_err_worst",
        "pass_rows",
        "fail_rows",
        "skip_rows",
        "not_run_rows",
        "log_path",
        "result_csv_path",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[regression-summary] wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
