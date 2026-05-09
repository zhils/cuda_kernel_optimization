#!/bin/bash
# 收集 Warp Stall 原因 Top 5（使用明确的 stall 指标）
# 用法: bash run_ncu_stall.sh
# 输出: data/ncu_reports/stall/*.txt
#
# 如需快速验证（只跑关键目标）:
#   NCU_STALL_QUICK=1 bash run_ncu_stall.sh

STALL_METRICS="\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_predicated_off_threads_per_issue_active.ratio"

NCU="ncu --target-processes all --kernel-name-base demangled --print-summary per-kernel --metrics $STALL_METRICS"
BIN="./build/bin"
OUT="data/ncu_reports/stall"
mkdir -p "$OUT"

TARGETS=(
    gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4 gemm_fp16
    softmax_v0 softmax_v1 softmax_v2 softmax_v3
    rmsnorm_v0 rmsnorm_v1 rmsnorm_v2 rmsnorm_v3
    flash_attention_v0 flash_attention_v1 flash_attention_v2
    flash_attention_v3 flash_attention_v4
    fused_conv1d_silu_v0 fused_conv1d_silu_v1 fused_conv1d_silu_v2 fused_conv1d_silu_v3
    fused_gated_delta_rule_v0 fused_gated_delta_rule_v1 fused_gated_delta_rule_v2
    fused_l2_norm_qk_v0 fused_l2_norm_qk_v1 fused_l2_norm_qk_v2
    fused_output_norm_gate_v0 fused_output_norm_gate_v1 fused_output_norm_gate_v2
    q_path_fusion_v0 q_path_fusion_v1 q_path_fusion_v2
)

if [ -n "$NCU_STALL_QUICK" ]; then
    TARGETS=(gemm_v3 rmsnorm_v3 softmax_v3)
fi

for t in "${TARGETS[@]}"; do
    echo "=== $t ==="
    $NCU "$BIN/$t" > "$OUT/${t}.txt" 2>&1 || true

    echo "--- $t Top 5 Warp Stall Reasons ---"
    # 从 "Command line profiler metrics" 节提取各 stall 指标值
    # 取最后（最大规模）的 launch 配置的 Average
    python3 -c "
import re, sys

with open('$OUT/${t}.txt', 'r') as f:
    text = f.read()

# 找到所有 'Command line profiler metrics' 小节
sections = text.split('Section: Command line profiler metrics')
if len(sections) < 2:
    print('  (no metric data)')
    sys.exit(0)

# 取最后一节的指标
last = sections[-1]
metrics = {}
for line in last.split('\n'):
    m = re.search(r'smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio\s+inst\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
    if m:
        try:
            name = m.group(1).replace('_', ' ').title()
            avg = float(m.group(4))
            if avg > 0:
                metrics[name] = avg
        except ValueError:
            pass

if not metrics:
    print('  (all zero or no stall data)')
    sys.exit(0)

# 按值排序取 Top 5
sorted_m = sorted(metrics.items(), key=lambda x: -x[1])
total = sum(v for _, v in sorted_m)
for rank, (name, val) in enumerate(sorted_m[:5], 1):
    pct = val / total * 100
    print(f'  {rank}. {name}: {pct:.1f}%')
other_pct = sum(v/total*100 for _, v in sorted_m[5:]) if len(sorted_m) > 5 else 0
if other_pct > 0:
    print(f'  (other: {other_pct:.1f}%)')
"
    echo ""
done

echo "========== 全部完成 =========="
echo "结果保存至 data/ncu_reports/stall/"
