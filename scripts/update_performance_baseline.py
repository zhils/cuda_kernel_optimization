#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Create or refresh performance golden baseline from standardized summary.")
    parser.add_argument("--summary", required=True, help="Path to summary_standardized.csv")
    parser.add_argument("--output", required=True, help="Path to golden baseline csv")
    args = parser.parse_args()

    src = Path(args.summary)
    dst = Path(args.output)
    rows = []

    with src.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只记录有明确验证通过的数据行，降低噪音。
            if row.get("verify_status", "") != "PASS":
                continue

            gpu_ms = parse_float(row.get("gpu_mean_ms", ""))
            tp = parse_float(row.get("throughput_value", ""))
            if gpu_ms is None:
                continue

            rows.append(
                {
                    "kernel_target": row.get("kernel_target", ""),
                    "shape_kind": row.get("shape_kind", ""),
                    "shape": row.get("shape", ""),
                    "throughput_unit": row.get("throughput_unit", ""),
                    "gpu_mean_ms_baseline": f"{gpu_ms:.8g}",
                    "throughput_baseline": "" if tp is None else f"{tp:.8g}",
                    "source_run_id": row.get("run_id", ""),
                }
            )

    rows.sort(key=lambda r: (r["kernel_target"], r["shape_kind"], r["shape"]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kernel_target",
                "shape_kind",
                "shape",
                "throughput_unit",
                "gpu_mean_ms_baseline",
                "throughput_baseline",
                "source_run_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[perf-baseline] wrote {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    main()
