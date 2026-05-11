#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def apply_autotune(base_targets, families, autotune_cache):
    if not autotune_cache:
        return base_targets

    tuned = []
    for name in base_targets:
        replaced = False
        for family, members in families.items():
            if name in members:
                preferred = autotune_cache.get("preferred_by_family", {}).get(family)
                if preferred and preferred in members:
                    tuned.append(preferred)
                    replaced = True
                break
        if not replaced:
            tuned.append(name)
    return unique_keep_order(tuned)


def main():
    parser = argparse.ArgumentParser(description="Dispatch benchmark/profile targets from unified catalog.")
    parser.add_argument("--catalog", required=True, help="kernel_catalog.json path")
    parser.add_argument(
        "--tier",
        required=True,
        choices=["smoke", "full", "profile_all", "profile_stall", "profile_roofline"],
    )
    parser.add_argument("--autotune-cache", default="", help="Optional autotune cache json")
    parser.add_argument("--print-format", choices=["lines", "space"], default="lines")
    args = parser.parse_args()

    catalog = load_json(Path(args.catalog))
    targets = list(catalog.get("targets", {}).get(args.tier, []))
    families = catalog.get("families", {})

    autotune_cache = {}
    if args.autotune_cache:
        path = Path(args.autotune_cache)
        if path.exists():
            autotune_cache = load_json(path)

    targets = apply_autotune(targets, families, autotune_cache)

    if args.print_format == "space":
        print(" ".join(targets))
    else:
        for t in targets:
            print(t)


if __name__ == "__main__":
    main()
