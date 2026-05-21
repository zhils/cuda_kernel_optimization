#!/usr/bin/env python3
"""Run ncu profiling on core binaries."""

import subprocess
import re
import json
import os
import sys
import time
from collections import OrderedDict

BIN_DIR = "/home/zh0813/cuda_kernel_optimization/build/bin"
OUTPUT_DIR = "/home/zh0813/cuda_kernel_optimization/data/profiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BINARIES = OrderedDict([
    ("gemm_v0",       ["gemm",        "V0: Naive baseline"]),
    ("gemm_v1",       ["gemm",        "V1: SMEM 16x16 tiling"]),
    ("gemm_v2",       ["gemm",        "V2: Register 8x8/thread"]),
    ("gemm_v3",       ["gemm",        "V3: cp.async + TileK=32"]),
    ("gemm_v4",       ["gemm",        "V4: TF32 WMMA"]),
    ("gemm_fp16",     ["gemm",        "FP16 WMMA Tensor Core"]),
    ("gemm_cublas_ref", ["gemm",      "cuBLAS FP32 reference"]),
    ("gemm_cublas_fp16", ["gemm",     "cuBLAS FP16 reference"]),
    ("rmsnorm_v0",    ["rmsnorm",     "V0: 1 thread/row"]),
    ("rmsnorm_v1",    ["rmsnorm",     "V1: SMEM + float4"]),
    ("rmsnorm_v2",    ["rmsnorm",     "V2: + warp shuffle"]),
    ("rmsnorm_v3",    ["rmsnorm",     "V3: + weight in SMEM"]),
    ("fused_conv1d_silu_v0", ["fused_conv1d_silu", "V0: 5 separate kernels"]),
    ("fused_conv1d_silu_v1", ["fused_conv1d_silu", "V1: fully fused"]),
    ("fused_conv1d_silu_v3", ["fused_conv1d_silu", "V3: CUTLASS GEMM projection"]),
])
def run_ncu(bin_path, timeout=120):
    """Run ncu basic set profiling and capture output."""
    try:
        result = subprocess.run(
            ["ncu", "--set", "basic", "-c", "1", "--print-summary", "per-kernel", bin_path],
            capture_output=True, text=True, timeout=timeout,
            cwd="/home/zh0813/cuda_kernel_optimization/build"
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def extract_metrics(text, bin_name):
    """Extract key metrics from ncu text output."""
    # Find the main kernel name
    kernel_match = re.search(r'kernel\s*:\s*(\S+)', text, re.IGNORECASE)
    kernel_name = kernel_match.group(1) if kernel_match else "N/A"

    metrics = {"kernel": kernel_name}

    # Speed Of Light section
    sol_patterns = [
        ("mem_throughput_pct", r"Memory Throughput\s+%\s+([\d.]+)"),
        ("dram_throughput_pct", r"DRAM Throughput\s+%\s+([\d.]+)"),
        ("l1_throughput_pct", r"L1/TEX Cache Throughput\s+%\s+([\d.]+)"),
        ("l2_throughput_pct", r"L2 Cache Throughput\s+%\s+([\d.]+)"),
        ("compute_throughput_pct", r"Compute \(SM\) Throughput\s+%\s+([\d.]+)"),
        ("duration_us", r"Duration\s+us\s+([\d.]+)"),
    ]
    for key, pattern in sol_patterns:
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))

    # Launch Statistics
    launch_patterns = [
        ("registers_per_thread", r"Registers Per Thread\s+register/thread\s+([\d.]+)"),
        ("block_size", r"Block Size\s+([\d.]+)"),
        ("grid_size", r"Grid Size\s+([\d.]+)"),
        ("dynamic_smem", r"Dynamic Shared Memory Per Block\s+byte/block\s+([\d.]+)"),
        ("static_smem", r"Static Shared Memory Per Block\s+byte/block\s+([\d.]+)"),
        ("threads", r"Threads\s+thread\s+([\d.]+)"),
        ("waves_per_sm", r"Waves Per SM\s+([\d.]+)"),
    ]
    for key, pattern in launch_patterns:
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))

    # Occupancy
    occ_patterns = [
        ("theoretical_occupancy_pct", r"Theoretical Occupancy\s+%\s+([\d.]+)"),
        ("achieved_occupancy_pct", r"Achieved Occupancy\s+%\s+([\d.]+)"),
        ("achieved_active_warps", r"Achieved Active Warps Per SM\s+warp\s+([\d.]+)"),
        ("theoretical_active_warps", r"Theoretical Active Warps per SM\s+warp\s+([\d.]+)"),
    ]
    for key, pattern in occ_patterns:
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))

    # GFLOPS from app output
    gflops_match = re.search(r'(\d+)\s*\*\s*\1\s*\*\s*\1.*?\|\s*([\d.]+)\s*GFLOPS', text, re.DOTALL)
    if not gflops_match:
        gflops_match = re.search(r'GT/s\|?\s*([\d.]+)\s*GFLOPS', text)
    if gflops_match:
        metrics["gflops"] = float(gflops_match.group(1))

    return metrics

def format_metrics_table(metrics_list):
    """Format metrics as a markdown table."""
    header = "| Binary | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occ% | Reg/Thread | Block | Grid |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for m in metrics_list:
        name = m.get("name", "?")
        kernel = m.get("kernel", "?")[:40]
        dur = f"{m.get('duration_us', 0):.1f}"
        comp = f"{m.get('compute_throughput_pct', 0):.1f}"
        mem = f"{m.get('mem_throughput_pct', 0):.1f}"
        l1 = f"{m.get('l1_throughput_pct', 0):.1f}"
        l2 = f"{m.get('l2_throughput_pct', 0):.1f}"
        occ = f"{m.get('achieved_occupancy_pct', 0):.1f}"
        reg = f"{m.get('registers_per_thread', 0):.0f}"
        bs = f"{m.get('block_size', 0):.0f}"
        gs = f"{m.get('grid_size', 0):.0f}"
        rows.append(f"| {name} | {kernel} | {dur} | {comp} | {mem} | {l1} | {l2} | {occ} | {reg} | {bs} | {gs} |")
    return "\n".join([header, sep] + rows)

def main():
    all_results = {}

    for bin_name, (group, desc) in BINARIES.items():
        bin_path = os.path.join(BIN_DIR, bin_name)
        if not os.path.exists(bin_path):
            print(f"SKIP: {bin_name} (not found)")
            continue

        print(f"PROFILING: {bin_name} ({group}) - {desc}")
        sys.stdout.flush()

        text = run_ncu(bin_path)
        if text == "TIMEOUT":
            print(f"  TIMEOUT")
            continue
        if text.startswith("ERROR"):
            print(f"  {text}")
            continue

        metrics = extract_metrics(text, bin_name)
        metrics["name"] = bin_name
        metrics["group"] = group
        metrics["description"] = desc

        if group not in all_results:
            all_results[group] = []
        all_results[group].append(metrics)

        print(f"  Kernel: {metrics.get('kernel', '?')}")
        print(f"  Compute: {metrics.get('compute_throughput_pct', 0):.1f}% | Mem: {metrics.get('mem_throughput_pct', 0):.1f}% | Occ: {metrics.get('achieved_occupancy_pct', 0):.1f}%")
        time.sleep(1)  # Brief cooldown between runs

    # Save JSON
    with open(os.path.join(OUTPUT_DIR, "profile_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print markdown summary per group
    for group, metrics_list in all_results.items():
        print(f"\n\n=== {group.upper()} ===")
        print(format_metrics_table(metrics_list))

    # Save markdown summary
    with open(os.path.join(OUTPUT_DIR, "profile_summary.md"), "w") as f:
        f.write("# Nsight Compute Profiling Summary\n\n")
        f.write(f"GPU: NVIDIA RTX 5060 Ti (Blackwell, CC 12.0)\n")
        f.write(f"CUDA: 13.2 | Profiler: Nsight Compute 2026.1.1\n\n")
        for group, metrics_list in all_results.items():
            f.write(f"## {group.upper()}\n\n")
            f.write(format_metrics_table(metrics_list))
            f.write("\n\n")

    print(f"\n\nResults saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
