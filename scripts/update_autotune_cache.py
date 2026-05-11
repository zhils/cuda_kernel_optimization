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


def parse_int(value):
    try:
        return int(value)
    except Exception:
        return None


def bucket_by_dims(dim_values):
    dims = [x for x in dim_values if x is not None and x > 0]
    if not dims:
        return "any"
    max_dim = max(dims)
    if max_dim <= 256:
        return "small"
    if max_dim <= 1024:
        return "medium"
    return "large"


def infer_shape_bucket(row):
    m = parse_int(row.get("M", ""))
    n = parse_int(row.get("N", ""))
    k = parse_int(row.get("K", ""))
    if m and n and k:
        return bucket_by_dims([m, n, k])

    rows = parse_int(row.get("rows", ""))
    cols = parse_int(row.get("cols", ""))
    if rows and cols:
        return bucket_by_dims([rows, cols])

    b = parse_int(row.get("B", ""))
    l = parse_int(row.get("L", ""))
    d = parse_int(row.get("D", ""))
    h = parse_int(row.get("H", ""))
    n_q = parse_int(row.get("N_q", ""))
    n_k = parse_int(row.get("N_k", ""))
    d_in = parse_int(row.get("D_in", ""))
    d_out = parse_int(row.get("D_out", ""))
    return bucket_by_dims([b, l, d, h, n_q, n_k, d_in, d_out])


def main():
    parser = argparse.ArgumentParser(description="Build autotune cache from standardized benchmark summary.")
    parser.add_argument("--summary", required=True, help="summary_standardized.csv")
    parser.add_argument("--output", required=True, help="autotune cache json output")
    args = parser.parse_args()

    rows = read_rows(Path(args.summary))
    best = {}
    best_by_bucket = {}
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
        bucket = infer_shape_bucket(row)

        prev = best.get(op_family)
        if prev is None or ms < prev["gpu_mean_ms"]:
            best[op_family] = {
                "kernel_target": kernel,
                "gpu_mean_ms": ms,
            }

        by_bucket = best_by_bucket.setdefault(op_family, {})
        prev_bucket = by_bucket.get(bucket)
        if prev_bucket is None or ms < prev_bucket["gpu_mean_ms"]:
            by_bucket[bucket] = {
                "kernel_target": kernel,
                "gpu_mean_ms": ms,
            }

    output = {
        "schema_version": 1,
        "source_run_id": run_id,
        "preferred_by_family": {k: v["kernel_target"] for k, v in sorted(best.items())},
        "preferred_by_family_by_bucket": {
            family: {bucket: item["kernel_target"] for bucket, item in sorted(bucket_map.items())}
            for family, bucket_map in sorted(best_by_bucket.items())
        },
        "evidence": {k: {"gpu_mean_ms": f"{v['gpu_mean_ms']:.8g}", "kernel_target": v["kernel_target"]} for k, v in sorted(best.items())},
        "evidence_by_bucket": {
            family: {
                bucket: {"gpu_mean_ms": f"{item['gpu_mean_ms']:.8g}", "kernel_target": item["kernel_target"]}
                for bucket, item in sorted(bucket_map.items())
            }
            for family, bucket_map in sorted(best_by_bucket.items())
        },
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"[autotune-cache] wrote {len(output['preferred_by_family'])} families -> {dst}")


if __name__ == "__main__":
    main()
