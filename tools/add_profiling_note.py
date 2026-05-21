#!/usr/bin/env python3
"""Add methodology validation note to all README profiling sections."""

import os

PROJECT_ROOT = "/home/zh0813/cuda_kernel_optimization"

README_PATHS = [
    "/home/zh0813/cuda_kernel_optimization/gemm/README.md",
    "/home/zh0813/cuda_kernel_optimization/rmsnorm/README.md",
    "/home/zh0813/cuda_kernel_optimization/fused_conv1d_silu/README.md",
    "/home/zh0813/cuda_kernel_optimization/README.md",
]

NOTE = """**说明：** ncu `--set basic` 默认对程序的**第一个 kernel launch** 进行 profiling。对于 GEMM 等算子，这对应最小测试尺寸（128×128），GPU 远未饱和。因此表格中的 Compute% / MemBW% 表示的是**小尺寸下的资源利用率**，用于横向对比各版本的寄存器压力、occupancy 等结构性差异。大尺寸下的实际性能请参考各算子 README 中的完整 benchmark 表格。
"""

def main():
    for readme_path in README_PATHS:
        if not os.path.exists(readme_path):
            continue

        with open(readme_path) as f:
            content = f.read()

        marker = "## Nsight Compute 性能分析"
        note_marker = "**说明：**"
        if marker in content:
            if note_marker in content:
                # Already has note, skip
                print(f"SKIP (already noted): {readme_path}")
                continue

            # Insert note after the header line
            lines = content.split("\n")
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
                if line.startswith(marker):
                    # Insert note after the blank lines following the table
                    pass

            # Simpler approach: insert after the table (find the last row)
            table_end = content.rfind(marker)
            after_table = content[table_end:]
            # Find the blank line after the last table row
            lines = after_table.split("\n")
            last_data_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("|") and not line.startswith("|---"):
                    last_data_idx = i

            if last_data_idx > 0:
                # Insert note after the table
                insert_pos = content.find("\n", content.find(lines[last_data_idx]))
                insert_pos = content.find("\n", insert_pos + 1)  # Next blank line
                content = content[:insert_pos] + "\n" + NOTE + content[insert_pos:]

                with open(readme_path, "w") as f:
                    f.write(content)
                print(f"UPDATED: {readme_path}")
            else:
                print(f"SKIP (no table): {readme_path}")

    print("Done!")

if __name__ == "__main__":
    main()
