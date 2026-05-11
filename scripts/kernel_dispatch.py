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


def match_value(cond, value):
    if cond is None:
        return True
    if isinstance(cond, list):
        return value in cond or "any" in cond
    return cond == "any" or cond == value


def choose_rules(catalog, tier, arch, dtype, layout, shape_bucket):
    out = {}
    for rule in catalog.get("routing_rules", []):
        m = rule.get("match", {})
        if not match_value(m.get("tier"), tier):
            continue
        if not match_value(m.get("arch"), arch):
            continue
        if not match_value(m.get("dtype"), dtype):
            continue
        if not match_value(m.get("layout"), layout):
            continue
        if not match_value(m.get("shape_bucket"), shape_bucket):
            continue
        for family, preferred in rule.get("preferred_by_family", {}).items():
            out[family] = preferred
    return out


def resolve_preferred(autotune_cache, family, shape_bucket):
    by_bucket = autotune_cache.get("preferred_by_family_by_bucket", {})
    bucket_map = by_bucket.get(family, {})
    if shape_bucket in bucket_map:
        return bucket_map[shape_bucket]
    if "any" in bucket_map:
        return bucket_map["any"]
    return autotune_cache.get("preferred_by_family", {}).get(family)


def apply_routing(base_targets, families, autotune_cache, overrides, shape_bucket):
    if not autotune_cache and not overrides:
        return base_targets

    tuned = []
    for name in base_targets:
        replaced = False
        for family, members in families.items():
            if name in members:
                preferred = overrides.get(family, "")
                if not preferred:
                    preferred = resolve_preferred(autotune_cache, family, shape_bucket)
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
    parser.add_argument("--arch", default="any", help="Routing dimension: arch (e.g., sm120)")
    parser.add_argument("--dtype", default="any", help="Routing dimension: dtype (e.g., fp32)")
    parser.add_argument("--layout", default="any", help="Routing dimension: layout")
    parser.add_argument("--shape-bucket", default="any", help="Routing dimension: shape bucket")
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

    overrides = choose_rules(catalog, args.tier, args.arch, args.dtype, args.layout, args.shape_bucket)
    targets = apply_routing(targets, families, autotune_cache, overrides, args.shape_bucket)

    if args.print_format == "space":
        print(" ".join(targets))
    else:
        for t in targets:
            print(t)


if __name__ == "__main__":
    main()
