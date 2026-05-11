# 项目面试介绍稿（可直接口播）

本文沉淀两套内容：

- 面试官视角的项目介绍逻辑（结构化讲法）
- 5 分钟口播稿（可直接背诵）

---

## 一、项目介绍逻辑（结构化讲法）

### 1) 30 秒开场（电梯版）

这是一个面向高性能算子开发的 CUDA 工程项目。  
我做了两层工作：**算子优化层** 和 **工程可信层**。  
算子层覆盖 GEMM、Softmax、RMSNorm、Flash Attention 及多个融合算子，采用 v0→vN 的单变量迭代优化；工程层构建了 smoke/full/profile 分层、环境指纹、统一结果 schema、性能回归门禁和 Nsight Compute 结构化分析，保证结论可复现、可比较、可追责。

### 2) 项目逻辑（方法论）

项目遵循固定闭环，不做“凭感觉调参”：

1. 先判瓶颈：Roofline/算术强度判断 memory-bound vs compute-bound  
2. 做最小改动：每个版本只改一个关键变量  
3. 双重验证：正确性（CPU/reference）+ 性能（ms/吞吐/NCU）  
4. 证据固化：run_id、manifest、标准化结果、报告  
5. 回归治理：baseline + WARN/FAIL 门禁

一句话：**从“写快代码”升级为“写可证明快代码”**。

### 3) 项目结构思路

- Kernel 实现层：各算子 v0~vN
- 调度策略层：`kernel_catalog + dispatch`（arch/dtype/layout/shape_bucket）
- 工程基建层：benchmark suite、schema、autotune cache、NCU 解析、回归报告

### 4) 代表性技术点（建议只讲 3 个）

- GEMM：shared memory tiling -> register tiling -> 更高阶路径，并结合 stall 变化解释收益  
- Softmax：两遍到 Online 单遍，warp shuffle + 向量化加载  
- 融合算子：减少中间缓冲、减少 global memory 往返、降低 launch 开销

### 5) 高层总结（价值主张）

目标不是“最快 demo kernel”，而是“可持续演进的算子优化平台”：

- 新 kernel 可快速接入统一评测
- 数据口径统一、可复现
- 性能退化可自动发现
- 能回答“为什么快/为什么慢”

---

## 二、5 分钟口播稿（逐句版）

我这个项目是一个 CUDA 高性能算子优化工程，目标不是单点“跑快一个 kernel”，而是建立一套可复现、可验证、可持续优化的方法体系。  

项目分两层。第一层是算子优化层，我实现了 GEMM、Softmax、RMSNorm、Flash Attention 和多个融合算子，每个算子都按 v0 到 vN 的路线做单变量优化。第二层是工程治理层，我做了分层 benchmark（smoke/full/profile）、统一结果 schema、run_id 和环境指纹、Nsight Compute 结构化解析、以及性能回归门禁，这样每次改动都能知道“是否变快、是否变错、是否可追溯”。  

我在优化上坚持一个固定闭环：先做瓶颈判断，再做最小改动，然后做正确性和性能双验证，最后把结果固化进统一报告。比如在 memory-bound 的算子上，我会优先做访存模式、向量化、共享内存复用；在 compute-bound 场景里，会优先看 tile、指令管线和并行映射。  

以 GEMM 为例，我从朴素版本开始，逐步引入 shared memory tiling、register tiling、再到更高阶路径。过程中我不是只看 GFLOPS，而是同时看 NCU 的 occupancy、吞吐和 stall 分布，确认优化是“真实有效”而不是偶然波动。  

以 Softmax 为例，我从两遍算法改为 Online 单遍，并配合 warp shuffle 归约和向量化加载，核心目标是减少全局访存和同步次数。  

融合算子方向上，我重点做的是减少中间缓冲和 kernel launch 数量，提升端到端路径效率。  

工程侧我做了几件关键的事：第一，所有结果通过标准化脚本落到统一 schema，避免“每个算子一套口径”；第二，性能门禁用了 baseline + WARN/FAIL 双阈值，支持 CI 场景，出现退化能自动拦截；第三，我把目标选择升级到统一 dispatch，支持按 arch、dtype、layout、shape bucket 做策略路由，并叠加 autotune cache，减少人工维护目标列表；第四，正确性回归不再只跑固定 case，我补了随机鲁棒性入口，让 GEMM、Softmax、QPath 可以随机生成测试集并自动恢复原配置。  

这套项目的核心价值是：我能不仅写出性能好的 kernel，还能把优化过程变成团队可复用的工程能力。任何改动都有证据链、有回归防线、有定位抓手。  

如果进团队，我可以在做算子优化的同时，把 profiling 和回归治理体系一起搭起来，保证性能迭代是长期稳定增益，而不是一次性冲刺。
