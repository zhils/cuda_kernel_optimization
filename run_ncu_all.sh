#!/bin/bash
# 收集所有缺失的 ncu profiling 数据
# 用法: bash run_ncu_all.sh

set -e

NCU="ncu --set basic --target-processes all --kernel-name-base demangled --print-summary per-kernel"
BIN="./build/bin"
OUT="data/ncu_reports/text"

mkdir -p "$OUT"

# ===== GEMM 系列（缺失 Compute/DRAM/Memory/Occupancy） =====
echo "=== gemm_v0 ==="
$NCU $BIN/gemm_v0 -c 1 -n 1 2>&1 | tee "$OUT/gemm_v0.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== gemm_v1 ==="
$NCU $BIN/gemm_v1 -c 1 -n 1 2>&1 | tee "$OUT/gemm_v1.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== gemm_v2 ==="
$NCU $BIN/gemm_v2 -c 1 -n 1 2>&1 | tee "$OUT/gemm_v2.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== gemm_v3 ==="
$NCU $BIN/gemm_v3 -c 1 -n 1 2>&1 | tee "$OUT/gemm_v3.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== gemm_v4 ==="
$NCU $BIN/gemm_v4 -c 1 -n 1 2>&1 | tee "$OUT/gemm_v4.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== gemm_fp16 ==="
$NCU $BIN/gemm_fp16 -c 1 -n 1 2>&1 | tee "$OUT/gemm_fp16.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

# ===== Softmax 系列（缺少 Duration 和 Memory 列） =====
for v in v0 v1 v2 v3; do
    echo "=== softmax_${v} ==="
    $NCU $BIN/softmax_${v} 2>&1 | tee "$OUT/softmax_${v}.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"
done

# ===== Flash Attention v3/v4（全新，未在表中） =====
echo "=== flash_attention_v3 ==="
$NCU $BIN/flash_attention_v3 2>&1 | tee "$OUT/flash_attention_v3.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== flash_attention_v4 ==="
$NCU $BIN/flash_attention_v4 2>&1 | tee "$OUT/flash_attention_v4.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

# ===== Fused Gated Delta Rule v1/v2（全新） =====
echo "=== fused_gated_delta_rule_v1 ==="
$NCU $BIN/fused_gated_delta_rule_v1 2>&1 | tee "$OUT/fused_gated_delta_rule_v1.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== fused_gated_delta_rule_v2 ==="
$NCU $BIN/fused_gated_delta_rule_v2 2>&1 | tee "$OUT/fused_gated_delta_rule_v2.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

# ===== Fused L2 Norm Q/K v1/v2（全新） =====
echo "=== fused_l2_norm_qk_v1 ==="
$NCU $BIN/fused_l2_norm_qk_v1 2>&1 | tee "$OUT/fused_l2_norm_qk_v1.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== fused_l2_norm_qk_v2 ==="
$NCU $BIN/fused_l2_norm_qk_v2 2>&1 | tee "$OUT/fused_l2_norm_qk_v2.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

# ===== Fused Output Norm Gate v1/v2（全新） =====
echo "=== fused_output_norm_gate_v1 ==="
$NCU $BIN/fused_output_norm_gate_v1 2>&1 | tee "$OUT/fused_output_norm_gate_v1.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo "=== fused_output_norm_gate_v2 ==="
$NCU $BIN/fused_output_norm_gate_v2 2>&1 | tee "$OUT/fused_output_norm_gate_v2.txt" | grep -E "(Duration|Compute|DRAM|Memory|Occupancy|Registers)"

echo ""
echo "========== 全部完成 =========="
echo "结果保存至 data/ncu_reports/text/"
echo ""
echo "请将以下文件内容发给我："
ls -1 "$OUT"/*.txt
