#!/usr/bin/env python3
"""
GEMM All-Variants Comparison Runner

Runs all GEMM variants and produces formatted comparison report.

Usage:
    python3 scripts/compare_gemm_all.py              # full comparison
    python3 scripts/compare_gemm_all.py --smoke      # quick smoke test

Requires:
    - build/bin/gemm_all_compare (compiled via cmake)
    - build/bin/gemm_bf16 (optional, standalone)
"""

import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    smoke = "--smoke" in sys.argv
    binary = os.path.join(PROJECT_ROOT, "build", "bin", "gemm_all_compare")

    if not os.path.exists(binary):
        print(f"[ERROR] {binary} not found. Build first:")
        print(f"    cd {PROJECT_ROOT}")
        print(f"    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release")
        print(f"    cmake --build build -j4 --target gemm_all_compare")
        sys.exit(1)

    cmd = [binary]
    if smoke:
        cmd.append("--smoke")

    print("=" * 70)
    print("  GEMM All-Variants Comparison")
    print("=" * 70)
    print(f"  Binary: {binary}")
    print(f"  Mode: {'smoke' if smoke else 'full'}")
    print()

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

if __name__ == "__main__":
    main()
