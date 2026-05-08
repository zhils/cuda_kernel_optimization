# 后续路线图

## 1. 算子扩展
- GELU / LayerNorm / BatchNorm
- Flash Attention V2 + V3
- Conv 2D / Depthwise Conv

## 2. 精度与量化
- FP16 / BF16 全链路验证
- INT8 per-channel 量化 + KL 散度选阈值
- INT4 GPTQ / AWQ 量化方案

## 3. 框架集成
- PyTorch custom op extension（已完成 softmax / rmsnorm binding）
- TensorRT plugin 适配
- ONNX Runtime custom op

## 4. 自动化
- Kernel auto-tuning 框架
- Nsight Compute CI 流水线
- PTX/SASS 版本管理

## 5. 跨架构
- AMD ROCm / HIP 移植（warp → wavefront 适配）
- 华为 Ascend CANN 方案预研
