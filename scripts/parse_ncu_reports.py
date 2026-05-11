#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


VALUE_RE = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")
TOP_STALL_RE = re.compile(r"\s*\d+\.\s+([^:]+):\s+([0-9.]+)%")


def parse_first_number(line: str):
    m = VALUE_RE.search(line)
    if not m:
        return ""
    return m.group(1)


def parse_basic_text(path: Path):
    metrics = {
        "duration": "",
        "compute_sm_pct": "",
        "dram_pct": "",
        "memory_pct": "",
        "achieved_occupancy_pct": "",
    }
    for line in path.read_text(errors="ignore").splitlines():
        low = line.lower()
        if "duration" in low and metrics["duration"] == "":
            metrics["duration"] = parse_first_number(line)
        elif "compute" in low and "throughput" in low and metrics["compute_sm_pct"] == "":
            metrics["compute_sm_pct"] = parse_first_number(line)
        elif "dram" in low and metrics["dram_pct"] == "":
            metrics["dram_pct"] = parse_first_number(line)
        elif "memory" in low and metrics["memory_pct"] == "":
            metrics["memory_pct"] = parse_first_number(line)
        elif "occupancy" in low and metrics["achieved_occupancy_pct"] == "":
            metrics["achieved_occupancy_pct"] = parse_first_number(line)
    return metrics


def parse_stall_text(path: Path):
    top = []
    for line in path.read_text(errors="ignore").splitlines():
        m = TOP_STALL_RE.match(line)
        if m:
            top.append(f"{m.group(1).strip()}:{m.group(2)}%")
    return ";".join(top[:3])


def main():
    parser = argparse.ArgumentParser(description="Parse NCU text outputs into a structured summary.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ncu-root", required=True, help="Path to data/ncu_reports/<run_id>")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ncu_root = Path(args.ncu_root)
    text_dir = ncu_root / "text"
    stall_dir = ncu_root / "stall"

    rows = []
    if text_dir.exists():
        for txt in sorted(text_dir.glob("*.txt")):
            target = txt.stem
            metrics = parse_basic_text(txt)
            stall_file = stall_dir / f"{target}.txt"
            top_stalls = parse_stall_text(stall_file) if stall_file.exists() else ""
            rows.append(
                {
                    "run_id": args.run_id,
                    "target": target,
                    "duration": metrics["duration"],
                    "compute_sm_pct": metrics["compute_sm_pct"],
                    "dram_pct": metrics["dram_pct"],
                    "memory_pct": metrics["memory_pct"],
                    "achieved_occupancy_pct": metrics["achieved_occupancy_pct"],
                    "top_stalls": top_stalls,
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "target",
                "duration",
                "compute_sm_pct",
                "dram_pct",
                "memory_pct",
                "achieved_occupancy_pct",
                "top_stalls",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ncu-parse] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
