#!/bin/bash
# 收集各算子最优版的 Roofline 分析图
# 用法: bash run_ncu_roofline.sh
# 输出: data/ncu_reports/roofline/*.png, data/ncu_reports/roofline/*.txt

NCU="ncu --set roofline --target-processes all --kernel-name-base demangled"
BIN="./build/bin"
OUT="data/ncu_reports/roofline"
mkdir -p "$OUT"

# 只跑每个算子的最优版本
TARGETS=(
    gemm_v3       # GEMM CUDA Core 最优
    gemm_fp16     # GEMM FP16 Tensor Core
    rmsnorm_v3    # RMSNorm 最优
    softmax_v3    # Softmax 最优
    fused_conv1d_silu_v3  # Conv1D+SiLU 最优
    flash_attention_v1    # Flash Attention tiled 版
    fused_gated_delta_rule_v2  # Gated Delta Rule 最优
    fused_l2_norm_qk_v1   # L2 Norm Q/K 最优
    fused_output_norm_gate_v2 # Output Norm Gate 最优
    q_path_fusion_v2      # Q Path Fusion 最优
)

if [ -n "$NCU_ROOF_QUICK" ]; then
    TARGETS=(gemm_v3 rmsnorm_v3 softmax_v3)
fi

for t in "${TARGETS[@]}"; do
    echo "=== $t ==="
    $NCU -o "$OUT/roofline_${t}" "$BIN/$t" 2>&1 | tee "$OUT/${t}.txt"
    echo ""
done

echo "========== 全部完成 =========="
echo "Roofline PNG 保存至 $OUT/roofline_*.ncu-rep"
echo "可用 Nsight Compute GUI 打开 .ncu-rep 文件查看 Roofline 图表"
