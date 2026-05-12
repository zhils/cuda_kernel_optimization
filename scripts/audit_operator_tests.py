#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


OP_RESULTS = {
    "gemm": "gemm_v3_results.csv",
    "softmax": "softmax_v3_results.csv",
    "rmsnorm": "rmsnorm_v3_results.csv",
    "flash_attention": "flash_attention_v4_results.csv",
    "fused_conv1d_silu": "fused_conv1d_silu_v3_results.csv",
    "fused_gated_delta_rule": "fused_gated_delta_rule_v2_results.csv",
    "fused_l2_norm_qk": "fused_l2_norm_qk_v2_results.csv",
    "fused_output_norm_gate": "fused_output_norm_gate_v2_results.csv",
    "q_path_fusion": "q_path_fusion_v2_results.csv",
}


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def pick_ms_key(fieldnames):
    for k in ["gpu_ms", "gpu_ms_v3", "ms"]:
        if k in fieldnames:
            return k
    return None


def analyze_rows(rows):
    if not rows:
        return {
            "rows": 0,
            "pass_rows": 0,
            "fail_rows": 0,
            "skip_rows": 0,
            "ms_valid_rows": 0,
            "neg_cpu_rows": 0,
            "correctness_sufficient": False,
            "performance_sufficient": False,
            "issues": ["no_rows"],
        }

    fieldnames = set(rows[0].keys())
    ms_key = pick_ms_key(fieldnames)
    pass_rows = sum(1 for r in rows if r.get("check", "").upper() == "PASS")
    fail_rows = sum(1 for r in rows if r.get("check", "").upper() == "FAIL")
    skip_rows = sum(1 for r in rows if r.get("check", "").upper() in ("SKIP", "NOT_RUN"))

    ms_valid_rows = 0
    for r in rows:
        if not ms_key:
            continue
        ms = parse_float(r.get(ms_key, ""))
        if ms is not None and ms > 0:
            ms_valid_rows += 1

    neg_cpu_rows = 0
    if "cpu_ms" in fieldnames:
        for r in rows:
            v = parse_float(r.get("cpu_ms", ""))
            if v is not None and v < 0:
                neg_cpu_rows += 1

    correctness_sufficient = pass_rows >= 1 and fail_rows == 0
    performance_sufficient = ms_valid_rows >= 1
    issues = []
    if pass_rows == 0:
        issues.append("no_pass_rows")
    if fail_rows > 0:
        issues.append("has_fail_rows")
    if ms_valid_rows == 0:
        issues.append("no_valid_gpu_ms")
    if neg_cpu_rows > 0:
        issues.append("has_negative_cpu_ms")
    if len(rows) < 3:
        issues.append("too_few_shapes(<3)")

    return {
        "rows": len(rows),
        "pass_rows": pass_rows,
        "fail_rows": fail_rows,
        "skip_rows": skip_rows,
        "ms_valid_rows": ms_valid_rows,
        "neg_cpu_rows": neg_cpu_rows,
        "correctness_sufficient": correctness_sufficient,
        "performance_sufficient": performance_sufficient,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit operator correctness/performance test sufficiency.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.output)

    lines = [
        "# Operator Test Coverage Audit",
        "",
        "判定规则：",
        "- 正确性充分：至少 1 行 PASS 且 0 行 FAIL",
        "- 性能充分：至少 1 行有效 gpu_ms（>0）",
        "- 覆盖建议：每算子至少 3 个规模点（小/中/大）",
        "",
        "| 算子 | 结果文件 | 行数 | PASS | FAIL | SKIP/NOT_RUN | 有效gpu_ms | 正确性充分 | 性能充分 | 主要问题 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]

    for op, filename in OP_RESULTS.items():
        rows = read_csv(results_dir / filename)
        a = analyze_rows(rows)
        issues = ",".join(a["issues"]) if a["issues"] else "none"
        lines.append(
            f"| `{op}` | `{filename}` | {a['rows']} | {a['pass_rows']} | {a['fail_rows']} | "
            f"{a['skip_rows']} | {a['ms_valid_rows']} | "
            f"{'yes' if a['correctness_sufficient'] else 'no'} | "
            f"{'yes' if a['performance_sufficient'] else 'no'} | {issues} |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[audit] wrote -> {out}")


if __name__ == "__main__":
    main()
