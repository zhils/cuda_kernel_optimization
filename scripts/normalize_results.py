#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def pick_first(row, keys, default=""):
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return default


def derive_op_target(csv_path: Path):
    stem = csv_path.stem
    if stem.endswith("_results"):
        target = stem[:-8]
    else:
        target = stem

    if target.startswith("fused_"):
        op_family = "_".join(target.split("_")[:2])
    else:
        op_family = target.split("_")[0]
    return op_family, target


def classify_shape(row):
    if all(k in row for k in ("M", "N", "K")):
        return "MxNxK", f"{row['M']}x{row['N']}x{row['K']}"
    if all(k in row for k in ("rows", "cols")):
        return "rows_cols", f"{row['rows']}x{row['cols']}"
    if all(k in row for k in ("B", "H", "N", "D")):
        return "BHND", f"B{row['B']}_H{row['H']}_N{row['N']}_D{row['D']}"
    if all(k in row for k in ("B", "L", "D", "H", "k_size")):
        return "BLDHk", f"B{row['B']}_L{row['L']}_D{row['D']}_H{row['H']}_k{row['k_size']}"
    if all(k in row for k in ("B", "L", "D", "H")):
        return "BLDH", f"B{row['B']}_L{row['L']}_D{row['D']}_H{row['H']}"
    if all(k in row for k in ("B", "N_q", "H_q", "N_k", "H_k")):
        return "BNqHqNkHk", (
            f"B{row['B']}_Nq{row['N_q']}_Hq{row['H_q']}_"
            f"Nk{row['N_k']}_Hk{row['H_k']}"
        )
    if all(k in row for k in ("B", "L", "D_in", "H", "D_out")):
        return "BLDInHDOut", (
            f"B{row['B']}_L{row['L']}_Din{row['D_in']}_H{row['H']}_Dout{row['D_out']}"
        )
    return "unknown", ""


def classify_throughput(row):
    if "gflops" in row:
        return row.get("gflops", ""), "GFLOP/s"
    if "bandwidth_gb_s" in row:
        return row.get("bandwidth_gb_s", ""), "GB/s"
    return "", "none"


def normalize_status(row):
    status = pick_first(row, ["verify_status", "check", "status"], default="")
    status = status.strip().upper() if status else ""
    if status in ("PASS", "FAIL", "SKIP", "NOT_RUN"):
        return status
    if status in ("N/A", "NA", ""):
        return "NOT_RUN"
    return status


def parse_error_fields(row):
    # 主误差字段优先
    direct = pick_first(row, ["max_abs_err", "max_abs_diff"], default="")
    direct_val = parse_float(direct)

    detail = {}
    for key, val in row.items():
        if key.startswith("max_abs_diff_") or key.startswith("max_abs_err_"):
            fv = parse_float(val)
            if fv is not None and not math.isnan(fv):
                detail[key] = fv

    if direct_val is not None and not math.isnan(direct_val):
        return f"{direct_val:.8g}", json.dumps(detail, separators=(",", ":")) if detail else ""

    if detail:
        worst = max(detail.values())
        return f"{worst:.8g}", json.dumps(detail, separators=(",", ":"))

    return "", ""


def extract_skip_reason(row):
    status = normalize_status(row)
    if status in ("SKIP", "NOT_RUN"):
        return pick_first(row, ["skip_reason", "note"], default="not_verified")
    return ""


def load_manifest(manifest_path: Path):
    if not manifest_path:
        return {}
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r") as f:
        return json.load(f)


def normalize_file(csv_path: Path, run_id: str, manifest: dict):
    records = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_family, target = derive_op_target(csv_path)
            shape_kind, shape = classify_shape(row)
            throughput_value, throughput_unit = classify_throughput(row)
            max_abs_err, max_abs_err_detail = parse_error_fields(row)
            verify_status = normalize_status(row)

            records.append(
                {
                    "run_id": run_id,
                    "op_family": op_family,
                    "kernel_target": target,
                    "shape_kind": shape_kind,
                    "shape": shape,
                    "M": row.get("M", ""),
                    "N": row.get("N", ""),
                    "K": row.get("K", ""),
                    "rows": row.get("rows", ""),
                    "cols": row.get("cols", ""),
                    "B": row.get("B", ""),
                    "L": row.get("L", ""),
                    "D": row.get("D", ""),
                    "H": row.get("H", ""),
                    "N_q": row.get("N_q", ""),
                    "H_q": row.get("H_q", ""),
                    "N_k": row.get("N_k", ""),
                    "H_k": row.get("H_k", ""),
                    "D_in": row.get("D_in", ""),
                    "D_out": row.get("D_out", ""),
                    "k_size": row.get("k_size", ""),
                    "gpu_mean_ms": pick_first(row, ["gpu_ms", "ms"], default=""),
                    "cpu_ms": row.get("cpu_ms", ""),
                    "throughput_value": throughput_value,
                    "throughput_unit": throughput_unit,
                    "speedup": row.get("speedup", ""),
                    "verify_status": verify_status,
                    "verify_reference": pick_first(row, ["verify_reference"], default="cpu_float"),
                    "max_abs_err": max_abs_err,
                    "max_abs_err_detail": max_abs_err_detail,
                    "threshold_used": pick_first(row, ["threshold_used"], default=""),
                    "skip_reason": extract_skip_reason(row),
                    "source_csv": str(csv_path),
                    "git_sha": manifest.get("git_sha", ""),
                    "gpu_name": manifest.get("gpu_name", ""),
                    "driver_version": manifest.get("driver_version", ""),
                    "cuda_version": manifest.get("cuda_version", ""),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(description="Normalize result CSV files to schema v1.")
    parser.add_argument("--results-dir", required=True, help="Path to data/results")
    parser.add_argument("--output", required=True, help="Output standardized CSV")
    parser.add_argument("--run-id", default="", help="Run id for traceability")
    parser.add_argument("--manifest", default="", help="Path to manifest.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else None
    manifest = load_manifest(manifest_path) if manifest_path else {}
    run_id = args.run_id or manifest.get("run_id", "adhoc")

    all_rows = []
    for csv_path in sorted(results_dir.glob("*results.csv")):
        all_rows.extend(normalize_file(csv_path, run_id, manifest))

    fieldnames = [
        "run_id",
        "op_family",
        "kernel_target",
        "shape_kind",
        "shape",
        "M",
        "N",
        "K",
        "rows",
        "cols",
        "B",
        "L",
        "D",
        "H",
        "N_q",
        "H_q",
        "N_k",
        "H_k",
        "D_in",
        "D_out",
        "k_size",
        "gpu_mean_ms",
        "cpu_ms",
        "throughput_value",
        "throughput_unit",
        "speedup",
        "verify_status",
        "verify_reference",
        "max_abs_err",
        "max_abs_err_detail",
        "threshold_used",
        "skip_reason",
        "source_csv",
        "git_sha",
        "gpu_name",
        "driver_version",
        "cuda_version",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[normalize] wrote {len(all_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
