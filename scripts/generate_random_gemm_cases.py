#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def align_up(x, a):
    return ((x + a - 1) // a) * a


def main():
    parser = argparse.ArgumentParser(description="Generate randomized GEMM square cases for robustness checks.")
    parser.add_argument("--output", required=True, help="Output csv path")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--min-dim", type=int, default=96)
    parser.add_argument("--max-dim", type=int, default=1536)
    parser.add_argument("--align", type=int, default=16)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dims = set()
    while len(dims) < args.count:
        raw = rng.randint(args.min_dim, args.max_dim)
        dims.add(align_up(raw, args.align))

    rows = []
    for i, d in enumerate(sorted(dims)):
        rows.append({"id": i, "group": "R", "rows": d, "cols": d})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group", "rows", "cols"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[random-gemm-cases] wrote {len(rows)} cases -> {out}")


if __name__ == "__main__":
    main()
