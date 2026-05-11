#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def read_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Build autotune cache from standardized benchmark summary.")
    parser.add_argument("--summary", required=True, help="summary_standardized.csv")
    parser.add_argument("--output", required=True, help="autotune cache json output")
    args = parser.parse_args()

    rows = read_rows(Path(args.summary))
    best = {}
    run_id = ""

    for row in rows:
        if row.get("verify_status") != "PASS":
            continue
        if row.get("throughput_unit") == "none":
            continue

        op_family = row.get("op_family", "")
        kernel = row.get("kernel_target", "")
        ms = parse_float(row.get("gpu_mean_ms", ""))
        if not op_family or not kernel or ms is None:
            continue

        run_id = row.get("run_id", run_id)
        prev = best.get(op_family)
        if prev is None or ms < prev["gpu_mean_ms"]:
            best[op_family] = {
                "kernel_target": kernel,
                "gpu_mean_ms": ms,
            }

    output = {
        "schema_version": 1,
        "source_run_id": run_id,
        "preferred_by_family": {k: v["kernel_target"] for k, v in sorted(best.items())},
        "evidence": {k: {"gpu_mean_ms": f"{v['gpu_mean_ms']:.8g}", "kernel_target": v["kernel_target"]} for k, v in sorted(best.items())},
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"[autotune-cache] wrote {len(output['preferred_by_family'])} families -> {dst}")


if __name__ == "__main__":
    main()
