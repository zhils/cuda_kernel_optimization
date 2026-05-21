#!/usr/bin/env python3
"""Audit test case catalog coverage vs API validate expectations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "configs" / "test_cases"

FAMILIES = {
    "gemm": "gemm.json",
    "rmsnorm": "rmsnorm.json",
    "fused_conv1d_silu": "fused_conv1d_silu.json",
}

EXPECT_VALUES = {"pass", "skip", "invalid_argument", "unsupported", "fail"}


def load_catalog(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def audit_family(name: str, cat: dict) -> list[str]:
    issues: list[str] = []
    if cat.get("family") != name:
        issues.append(f"{name}: family field mismatch ({cat.get('family')})")
    cases = cat.get("cases", [])
    if not cases:
        issues.append(f"{name}: empty cases list")
    ids = set()
    tags: set[str] = set()
    for c in cases:
        cid = c.get("id")
        if not cid:
            issues.append(f"{name}: case missing id")
            continue
        if cid in ids:
            issues.append(f"{name}: duplicate id {cid}")
        ids.add(cid)
        exp = c.get("expect")
        if exp not in EXPECT_VALUES:
            issues.append(f"{name}/{cid}: invalid expect={exp}")
        for t in c.get("tags", []):
            tags.add(t)
        if "params" not in c:
            issues.append(f"{name}/{cid}: missing params")
    if "smoke" not in tags:
        issues.append(f"{name}: no case tagged smoke")
    if "boundary" not in tags:
        issues.append(f"{name}: no case tagged boundary")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON summary")
    args = parser.parse_args()

    summary = {"families": {}, "issues": []}
    for fam, fname in FAMILIES.items():
        path = CATALOG_DIR / fname
        if not path.exists():
            summary["issues"].append(f"missing catalog: {path}")
            continue
        cat = load_catalog(path)
        issues = audit_family(fam, cat)
        summary["families"][fam] = {
            "path": str(path),
            "num_cases": len(cat.get("cases", [])),
            "schema_version": cat.get("schema_version"),
        }
        summary["issues"].extend(issues)

    if args.json:
        import json as _json

        print(_json.dumps(summary, indent=2))
    else:
        print("=== Test catalog audit ===")
        for fam, info in summary["families"].items():
            print(f"{fam}: {info['num_cases']} cases ({info['path']})")
        if summary["issues"]:
            print("\nIssues:")
            for i in summary["issues"]:
                print(f"  - {i}")
        else:
            print("\nAll catalogs OK.")

    return 1 if summary["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
