#!/usr/bin/env python3
"""
Generate best-implementation summary from unified main-scenario CSV.

Input:
  data/results/main_scenario_unified.csv

Outputs:
  data/results/best_impl_summary.csv
  data/results/best_impl_summary.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass
class Row:
    op_family: str
    impl: str
    shape: str
    gpu_ms: float
    check_status: str
    source_csv: str
    retest_tag: str
    updated_utc: str


def load_rows(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Row] = []
        for r in reader:
            rows.append(
                Row(
                    op_family=r["op_family"],
                    impl=r["impl"],
                    shape=r["shape"],
                    gpu_ms=float(r["gpu_ms"]),
                    check_status=r["check_status"],
                    source_csv=r["source_csv"],
                    retest_tag=r.get("retest_tag", ""),
                    updated_utc=r.get("updated_utc", ""),
                )
            )
    return rows


def select_best(rows: List[Row]) -> Dict[str, Row]:
    by_family: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        by_family[row.op_family].append(row)

    best: Dict[str, Row] = {}
    for family, fam_rows in by_family.items():
        # Prefer PASS entries; fallback to minimal gpu_ms among all.
        pass_rows = [r for r in fam_rows if r.check_status == "PASS"]
        pool = pass_rows if pass_rows else fam_rows
        best[family] = min(pool, key=lambda x: x.gpu_ms)
    return best


def write_csv(path: Path, best_rows: List[Row]) -> None:
    fieldnames = [
        "op_family",
        "best_impl",
        "best_gpu_ms",
        "shape",
        "check_status",
        "source_csv",
        "retest_tag",
        "updated_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in best_rows:
            writer.writerow(
                {
                    "op_family": r.op_family,
                    "best_impl": r.impl,
                    "best_gpu_ms": f"{r.gpu_ms:.6f}",
                    "shape": r.shape,
                    "check_status": r.check_status,
                    "source_csv": r.source_csv,
                    "retest_tag": r.retest_tag,
                    "updated_utc": r.updated_utc,
                }
            )


def write_markdown(path: Path, best_rows: List[Row], source_csv: Path) -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines = []
    lines.append("# Best Implementation Summary")
    lines.append("")
    lines.append(f"- Source: `{source_csv.as_posix()}`")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append("")
    lines.append("| op_family | best_impl | best_gpu_ms | shape | check_status |")
    lines.append("|---|---|---:|---|---|")
    for r in best_rows:
        lines.append(f"| `{r.op_family}` | `{r.impl}` | {r.gpu_ms:.6f} | `{r.shape}` | `{r.check_status}` |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update best implementation summary from unified CSV.")
    parser.add_argument(
        "--input",
        default="data/results/main_scenario_unified.csv",
        help="Input unified main scenario CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/results/best_impl_summary.csv",
        help="Output best implementation CSV.",
    )
    parser.add_argument(
        "--output-md",
        default="data/results/best_impl_summary.md",
        help="Output best implementation markdown summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    rows = load_rows(input_path)
    best = select_best(rows)
    best_rows = [best[k] for k in sorted(best.keys())]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, best_rows)
    write_markdown(output_md, best_rows, input_path)

    print(f"[best-summary] wrote csv -> {output_csv}")
    print(f"[best-summary] wrote md  -> {output_md}")


if __name__ == "__main__":
    main()

