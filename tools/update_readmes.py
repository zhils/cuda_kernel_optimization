#!/usr/bin/env python3
"""Update all operator READMEs with Nsight Compute profiling results."""

import json
import os

PROFILING_JSON = "/home/zh0813/cuda_kernel_optimization/data/profiling/profile_results.json"
PROJECT_ROOT = "/home/zh0813/cuda_kernel_optimization"

with open(PROFILING_JSON) as f:
    data = json.load(f)

# Known kernel names mapped from binary names
KERNEL_NAMES = {
    "gemm_v0": "GemmNaiveKernel",
    "gemm_v1": "GemmSmemKernel",
    "gemm_v2": "GemmRegTiledKernel",
    "gemm_v3": "GemmCpAsyncKernel",
    "gemm_v4": "GemmTF32WmmaKernel",
    "gemm_fp16": "GemmFP16Kernel",
    "gemm_cublas_ref": "cublasSgemm (cuBLAS FP32)",
    "gemm_cublas_fp16": "cublasGemmEx (cuBLAS FP16)",
    "rmsnorm_v0": "RmsnormSerialKernel",
    "rmsnorm_v1": "RmsnormSmemVecKernel",
    "rmsnorm_v2": "RmsnormShuffleKernel",
    "rmsnorm_v3": "RmsnormWeightSmemKernel",
    "fused_conv1d_silu_v0": "FusedConv1dSiluV0 (5 kernels)",
    "fused_conv1d_silu_v1": "FusedConv1dSiluV1 (fully fused)",
    "fused_conv1d_silu_v3": "FusedConv1dSiluV3 (CUTLASS GEMM)",
}

# Read the gemm README to determine where to insert profiling sections
# We'll append after the last section of each README

def generate_profiling_section(group_name, metrics_list):
    """Generate a markdown profiling section for an operator group."""

    lines = []
    lines.append(f"## Nsight Compute 性能分析\n")
    lines.append(f"")
    lines.append(f"使用 `ncu --set basic` 对每个可执行文件的第一个 kernel launch 进行 profiling。")
    lines.append(f"运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1")
    lines.append(f"")
    lines.append(f"| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|---|")

    for m in metrics_list:
        name = m.get("name", "?")
        kernel = KERNEL_NAMES.get(name, name)
        dur = m.get("duration_us", 0)
        comp = m.get("compute_throughput_pct", 0)
        mem = m.get("mem_throughput_pct", 0)
        l1 = m.get("l1_throughput_pct", 0)
        l2 = m.get("l2_throughput_pct", 0)
        occ = m.get("achieved_occupancy_pct", 0)
        reg = m.get("registers_per_thread", 0)
        bs = m.get("block_size", 0)
        gs = m.get("grid_size", 0)
        lines.append(f"| {name} | {kernel} | {dur:.1f} | {comp:.1f}% | {mem:.1f}% | {l1:.1f}% | {l2:.1f}% | {occ:.1f}% | {reg:.0f} | {bs:.0f} | {gs:.0f} |")

    lines.append(f"")

    return "\n".join(lines)


def update_readme(readme_path, profiling_section):
    """Add or replace the profiling section in a README."""
    if not os.path.exists(readme_path):
        print(f"  README not found: {readme_path}")
        return False

    with open(readme_path) as f:
        content = f.read()

    # Find the profiling section marker if it exists
    marker = "## Nsight Compute 性能分析"
    if marker in content:
        # Replace existing profiling section
        parts = content.split(marker)
        # Find the next section or end
        remaining = parts[1]
        next_section_idx = remaining.find("\n## ")
        if next_section_idx >= 0:
            remaining = remaining[next_section_idx:]
        else:
            remaining = ""
        content = parts[0] + marker + "\n" + profiling_section.split(marker, 1)[1] + remaining
    else:
        # Append at the end
        content = content.rstrip() + "\n\n" + profiling_section

    with open(readme_path, "w") as f:
        f.write(content)
    return True


# Map operator groups to README paths
README_MAP = {
    "gemm":              f"{PROJECT_ROOT}/gemm/README.md",
    "rmsnorm":           f"{PROJECT_ROOT}/rmsnorm/README.md",
    "fused_conv1d_silu": f"{PROJECT_ROOT}/fused_conv1d_silu/README.md",
}

def main():
    for group, metrics_list in data.items():
        if group not in README_MAP:
            print(f"SKIP group {group}: no README mapping")
            continue

        readme_path = README_MAP[group]
        profiling_section = generate_profiling_section(group, metrics_list)

        success = update_readme(readme_path, profiling_section)
        if success:
            print(f"UPDATED: {readme_path}")
        else:
            print(f"FAILED: {readme_path}")

    # Also update the top-level README
    top_readme = os.path.join(PROJECT_ROOT, "README.md")
    if os.path.exists(top_readme):
        # Create an aggregated summary
        all_groups = {}
        for group, metrics_list in data.items():
            for m in metrics_list:
                name = m.get("name", "?")
                kernel = KERNEL_NAMES.get(name, name)
                all_groups[name] = {
                    "group": group,
                    "kernel": kernel,
                    "dur": m.get("duration_us", 0),
                    "comp": m.get("compute_throughput_pct", 0),
                    "mem": m.get("mem_throughput_pct", 0),
                    "occ": m.get("achieved_occupancy_pct", 0),
                    "reg": m.get("registers_per_thread", 0),
                }

        lines = []
        lines.append(f"## Nsight Compute 全面性能分析\n")
        lines.append(f"")
        lines.append(f"使用 `ncu --set basic` 对所有 {len(all_groups)} 个 CUDA kernel 进行统一 profiling。")
        lines.append(f"运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1")
        lines.append(f"")
        lines.append(f"| Binary | 算子 | Kernel | Duration(us) | Compute% | MemBW% | Occ% | Reg/Thread |")
        lines.append(f"|---|---|---|---|---|---|---|---|")

        for name in sorted(all_groups.keys()):
            info = all_groups[name]
            lines.append(f"| {name} | {info['group']} | {info['kernel'][:35]} | {info['dur']:.1f} | {info['comp']:.1f}% | {info['mem']:.1f}% | {info['occ']:.1f}% | {info['reg']:.0f} |")

        lines.append(f"")

        profiling_section = "\n".join(lines)

        with open(top_readme) as f:
            content = f.read()

        marker = "## Nsight Compute 全面性能分析"
        if marker in content:
            parts = content.split(marker)
            remaining = parts[1]
            next_section_idx = remaining.find("\n## ")
            if next_section_idx >= 0:
                remaining = remaining[next_section_idx:]
            else:
                remaining = ""
            content = parts[0] + marker + "\n" + profiling_section.split(marker, 1)[1] + remaining
        else:
            content = content.rstrip() + "\n\n" + profiling_section

        with open(top_readme, "w") as f:
            f.write(content)
        print(f"UPDATED: {top_readme}")

    print("\nDone!")

if __name__ == "__main__":
    main()
