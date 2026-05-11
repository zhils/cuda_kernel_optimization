#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def align_up(x, a):
    return ((x + a - 1) // a) * a


def main():
    parser = argparse.ArgumentParser(description="Generate random matrix test cases.")
    parser.add_argument("--output", required=True, help="Output csv path")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--min-cols", type=int, default=64)
    parser.add_argument("--max-cols", type=int, default=4096)
    parser.add_argument("--align", type=int, default=16)
    parser.add_argument("--max-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--square-only", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    seen = set()
    attempts = 0
    max_attempts = args.count * 100
    while len(rows) < args.count and attempts < max_attempts:
        attempts += 1
        r = align_up(rng.randint(args.min_rows, args.max_rows), args.align)
        c = r if args.square_only else align_up(rng.randint(args.min_cols, args.max_cols), args.align)
        if r * c > args.max_elements:
            continue
        key = (r, c)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": len(rows), "group": "R", "rows": r, "cols": c})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group", "rows", "cols"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[random-matrix-cases] wrote {len(rows)} cases -> {out}")


if __name__ == "__main__":
    main()
