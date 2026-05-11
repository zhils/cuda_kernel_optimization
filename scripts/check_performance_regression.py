#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def make_key(row):
    return (
        row.get("kernel_target", ""),
        row.get("shape_kind", ""),
        row.get("shape", ""),
        row.get("throughput_unit", ""),
    )


def load_csv(path):
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def append_reason(reason, text):
    if not text:
        return reason
    if reason:
        return reason + ";" + text
    return text


def write_markdown_summary(path: Path, checks, matched, warn_count, fail_count, warn_ms, fail_ms, warn_tp, fail_tp):
    top_fails = [r for r in checks if r["status"] == "FAIL"][:10]
    top_warns = [r for r in checks if r["status"] == "WARN"][:10]

    lines = [
        "# Performance Gate Summary",
        "",
        f"- matched_rows: {matched}",
        f"- PASS: {sum(1 for r in checks if r['status'] == 'PASS')}",
        f"- WARN: {warn_count}",
        f"- FAIL: {fail_count}",
        "",
        "## Thresholds",
        f"- gpu_mean_ms warn>{warn_ms}%, fail>{fail_ms}%",
        f"- throughput_drop warn>{warn_tp}%, fail>{fail_tp}%",
        "",
    ]

    if top_fails:
        lines.extend(["## Top FAIL", ""])
        for row in top_fails:
            lines.append(
                f"- `{row['kernel_target']}` `{row['shape']}`: "
                f"ms_regress={row['gpu_ms_regress_pct']}%, tp_drop={row['throughput_drop_pct']}%, reason={row['reason']}"
            )
        lines.append("")

    if top_warns:
        lines.extend(["## Top WARN", ""])
        for row in top_warns:
            lines.append(
                f"- `{row['kernel_target']}` `{row['shape']}`: "
                f"ms_regress={row['gpu_ms_regress_pct']}%, tp_drop={row['throughput_drop_pct']}%, reason={row['reason']}"
            )
        lines.append("")

    if not top_fails and not top_warns:
        lines.append("## Result")
        lines.append("未发现性能回退。")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Check performance regression against golden baseline.")
    parser.add_argument("--summary", required=True, help="Current summary_standardized.csv")
    parser.add_argument("--baseline", required=True, help="Golden baseline csv")
    parser.add_argument("--output", required=True, help="Output regression check csv")
    parser.add_argument("--warn-ms-regress-pct", type=float, default=8.0)
    parser.add_argument("--fail-ms-regress-pct", type=float, default=15.0)
    parser.add_argument("--warn-throughput-drop-pct", type=float, default=5.0)
    parser.add_argument("--fail-throughput-drop-pct", type=float, default=10.0)
    parser.add_argument("--markdown-output", default="", help="Optional markdown summary output path")
    args = parser.parse_args()

    current_rows = [r for r in load_csv(args.summary) if r.get("verify_status", "") == "PASS"]
    baseline_rows = load_csv(args.baseline)
    baseline_map = {make_key(r): r for r in baseline_rows}

    checks = []
    fail_count = 0
    warn_count = 0
    matched = 0

    for row in current_rows:
        key = make_key(row)
        if key not in baseline_map:
            continue
        matched += 1
        base = baseline_map[key]

        cur_ms = parse_float(row.get("gpu_mean_ms", ""))
        base_ms = parse_float(base.get("gpu_mean_ms_baseline", ""))
        cur_tp = parse_float(row.get("throughput_value", ""))
        base_tp = parse_float(base.get("throughput_baseline", ""))

        ms_regress_pct = None
        tp_drop_pct = None
        status = "PASS"
        reason = ""

        if cur_ms is not None and base_ms is not None and base_ms > 0:
            ms_regress_pct = (cur_ms - base_ms) / base_ms * 100.0
            if ms_regress_pct > args.fail_ms_regress_pct:
                status = "FAIL"
                reason = append_reason(reason, f"gpu_ms_regress>{args.fail_ms_regress_pct}%")
            elif ms_regress_pct > args.warn_ms_regress_pct:
                if status != "FAIL":
                    status = "WARN"
                reason = append_reason(reason, f"gpu_ms_regress>{args.warn_ms_regress_pct}%")

        if cur_tp is not None and base_tp is not None and base_tp > 0:
            tp_drop_pct = (base_tp - cur_tp) / base_tp * 100.0
            if tp_drop_pct > args.fail_throughput_drop_pct:
                status = "FAIL"
                reason = append_reason(reason, f"throughput_drop>{args.fail_throughput_drop_pct}%")
            elif tp_drop_pct > args.warn_throughput_drop_pct:
                if status != "FAIL":
                    status = "WARN"
                reason = append_reason(reason, f"throughput_drop>{args.warn_throughput_drop_pct}%")

        if status == "FAIL":
            fail_count += 1
        elif status == "WARN":
            warn_count += 1

        checks.append(
            {
                "kernel_target": row.get("kernel_target", ""),
                "shape_kind": row.get("shape_kind", ""),
                "shape": row.get("shape", ""),
                "throughput_unit": row.get("throughput_unit", ""),
                "gpu_mean_ms_current": row.get("gpu_mean_ms", ""),
                "gpu_mean_ms_baseline": base.get("gpu_mean_ms_baseline", ""),
                "gpu_ms_regress_pct": "" if ms_regress_pct is None else f"{ms_regress_pct:.4f}",
                "throughput_current": row.get("throughput_value", ""),
                "throughput_baseline": base.get("throughput_baseline", ""),
                "throughput_drop_pct": "" if tp_drop_pct is None else f"{tp_drop_pct:.4f}",
                "status": status,
                "reason": reason,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kernel_target",
                "shape_kind",
                "shape",
                "throughput_unit",
                "gpu_mean_ms_current",
                "gpu_mean_ms_baseline",
                "gpu_ms_regress_pct",
                "throughput_current",
                "throughput_baseline",
                "throughput_drop_pct",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(checks)

    if args.markdown_output:
        write_markdown_summary(
            Path(args.markdown_output),
            checks,
            matched,
            warn_count,
            fail_count,
            args.warn_ms_regress_pct,
            args.fail_ms_regress_pct,
            args.warn_throughput_drop_pct,
            args.fail_throughput_drop_pct,
        )

    print(f"[perf-gate] matched={matched}, warn={warn_count}, fail={fail_count}, report={out}")
    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
