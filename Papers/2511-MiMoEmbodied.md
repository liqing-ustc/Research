---
title: "MiMo-Embodied: X-Embodied Foundation Model Technical Report"
authors: [Xiaomi Embodied Intelligence Team]
institutes: [Xiaomi]
date_publish: 2025-11
venue: arXiv
tags: [VLM, cross-embodiment, spatial-reasoning]
paper: https://arxiv.org/abs/2511.16518
website: 
github: https://github.com/XiaomiMiMo/MiMo-Embodied
rating: 1
date_added: 2026-04-16
---
## 速查卡片

> [!summary] MiMo-Embodied: X-Embodied Foundation Model
> - **核心**: 首个跨 embodiment 的统一 VLM，同时覆盖 Embodied AI 和 Autonomous Driving，通过多阶段训练实现两域正迁移
> - **方法**: 基于 MiMo-VL 的四阶段渐进训练（Embodied SFT → AD SFT → CoT SFT → GRPO RL），配合跨域数据构建
> - **结果**: 17 个 Embodied AI benchmark 和 12 个 AD benchmark 上超越 open/closed-source 通用和专用 VLM
> - **Sources**: [paper](https://arxiv.org/abs/2511.16518) | [github](https://github.com/XiaomiMiMo/MiMo-Embodied)

---
## Summary

首个开源跨 embodiment VLM，统一 Embodied AI（affordance prediction、task planning、spatial understanding）和 Autonomous Driving（perception、prediction、planning）能力。

**Key Takeaways:**
1. **Cross-embodiment positive transfer**: 通过多阶段训练，Embodied AI 和 AD 两个领域产生显著正迁移，联合训练不仅不互损，反而互相增强
2. **Progressive multi-stage training**: 四阶段渐进策略（Embodied SFT → AD SFT → CoT SFT → GRPO RL）有效缓解跨域任务干扰，对比直接混合训练在两域均有显著提升
3. **Comprehensive benchmark coverage**: 在 29 个 benchmark 上全面评估，17 个 Embodied AI + 12 个 AD，建立了跨 embodiment 能力评估标准

**Teaser. MiMo-Embodied 能力概览**
![](https://arxiv.org/html/2511.16518v1/x2.png)

---
## Architecture

MiMo-Embodied 架构基于 MiMo-VL，包含三个核心组件：
1. **Vision Transformer (ViT)**：继承 MiMo-VL 的视觉编码器，支持高分辨率输入，处理单图、多图和视频
2. **MLP Projector**：将视觉 token 映射到 LLM 输入空间
3. **LLM backbone**：负责文本理解与推理

所有组件均从 MiMo-VL（7B-SFT-2508 checkpoint）预训练权重初始化。与 InternVL3 将图像离散为 patch 不同，MiMo-Embodied 使用 3D 卷积显著减少 LLM token 数（796 vs 4096），同时保留空间上下文。

**Figure 3. 模型架构**
![](https://arxiv.org/html/2511.16518v1/x3.png)

---
## Training Dataset

数据集分三大类：General、Embodied AI、Autonomous Driving。

**Figure 4. 训练数据概览**
![](https://arxiv.org/html/2511.16518v1/x4.png)

### General Dataset
继承 MiMo-VL 训练语料，覆盖 Visual Grounding、Document/Chart Comprehension、Video Understanding、Multimodal Reasoning 四大类。

### Embodied AI Dataset
按目标能力分为三类：
- **Affordance Prediction**: PixMo-Points（fine-grained localization）、RoboAfford（object + scene affordance）、RoboRefIt（cluttered scene referential grounding）
- **High-level Task Planning**: [[2503-CosmosReason1|Cosmos-Reason1]]（cross-embodiment 物理推理，含 DeepSeek-R1 生成的长链推理）、EgoPlan-IT（egocentric planning）、RoboVQA（long-horizon QA）
- **Spatial Understanding**: SQA3D + 自建 3D QA、VLM-3R（时空推理 + 导航）、RefSpatial（referring spatial tasks）、EmbSpatial（egocentric spatial）

### Autonomous Driving Dataset
按功能模块分为三类：
- **Environmental Perception**: General Scene Understanding（CODA-LM、DriveLM、OmniDrive 等）、Regional Object Understanding（DriveAction 等）、Regional Object Localization（DRAMA）
- **Status Prediction**: Intent Prediction（DriveLM、MME-RealWorld）
- **Driving Planning**: Action Decision（DriveLM、IDKB）、Driving Reasoning（CODA-LM、LingoQA、NuInstruct、BDD-X）

---
## Training Strategy

四阶段渐进训练，逐步引入专用领域数据：

**Stage 1: Embodied AI SFT** — 在 MiMo-VL 通用数据 + Embodied AI 数据上微调，建立 affordance 理解、task planning 和 spatial reasoning 基础能力

**Stage 2: Autonomous Driving SFT** — 在 Stage 1 基础上加入 AD 数据，专注多视图空间推理、时序一致性和安全关键感知

**Stage 3: Chain-of-Thought SFT** — 从训练数据中采样子集生成 CoT 推理链，增强多步推理的透明性和逻辑连贯性

**Stage 4: RL Fine-tuning** — 使用 GRPO 算法针对 corner case 优化。多任务混合训练中设计不同 reward signal：选择题用 exact match、空间 grounding 用 IoU/point-in-mask、格式合规用 template check

**Table 1. 各阶段训练配置**

| Stages | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| Dataset | General + Embodied AI | Previous + AD | Previous + CoT | RL Data |
| Batch Size | 512 | 512 | 512 | 32 |
| Learning Rate | 2e-6 | 2e-6 | 2e-6 | 1e-6 |
| Weight Decay | 0.05 | 0.05 | 0.05 | 0.0 |
| Max Seq Len | 32768 | 32768 | 32768 | 32768 |

---
## Evaluation

### Embodied AI Benchmarks

在三大能力维度（Affordance、Planning、Spatial）的 17 个 benchmark 上评估。

**Affordance Prediction**：MiMo-Embodied 在全部 5 个 benchmark 上取得 SOTA，尤其在 VABench-Point、Part-Afford 和 RoboAfford-Eval 上大幅领先其他 embodied VLM。

**Task Planning**：在 RoboVQA 上超越所有模型，EgoPlan2 上高度竞争，展示了因果推理和长 horizon 规划能力。

**Spatial Understanding**：在 CV-Bench 上 SOTA（88.82），在 RoboSpatial、RefSpatial-Bench、CRPE-relation 上领先，EmbSpatial、SAT、MetaVQA 上高度竞争。

### Autonomous Driving Benchmarks

在 12 个 benchmark 上评估 Perception、Prediction、Planning 三大能力。

MiMo-Embodied 在 NAVSIM 上的 trajectory planning 结果突出（PDMS 91.0，RL 后），使用仅 796 个 LLM token（vs InternVL3 的 4096、ReCogDrive 的 2304）。在 proprietary 数据集上相比 Qwen2.5-VL baseline 平均提升 7.7%，复杂场景（U-turn、lane change）提升更显著（+8-10%）。

### Ablation Study

**Table 7. Ablation 结果**

| Model | Embodied | AD | Multi-Stage | Affordance | Spatial | Plan | Embodied Avg. | AD |
|---|---|---|---|---|---|---|---|---|
| MiMo-VL (Baseline) | X | X | X | 38.7 | 55.3 | 46.2 | 46.76 | 32.2 |
| MiMo-VL w/ Embodied | ✓ | X | X | 58.9 | 61.0 | 51.0 | 56.9 | 57.6 |
| MiMo-VL w/ AD | X | ✓ | X | 26.3 | 56.3 | 47.0 | 43.2 | 57.5 |
| MiMo-VL w/ Embodied+AD | ✓ | ✓ | X | 59.6 | 62.0 | 53.8 | 58.4 | 55.2 |
| MiMo-Embodied (Ours) | ✓ | ✓ | ✓ | 65.6 | 66.0 | 55.6 | 62.4 | 63.3 |

**Insights**:
- 单独训练 AD 数据（w/ AD）在 embodied 任务上甚至低于 baseline，但 AD 成绩好——说明单域训练不足以跨域泛化
- 直接混合训练（w/ Embodied+AD）改善了 embodied 但 AD 略降（55.2 vs 57.5）——跨域 task interference
- 多阶段训练（Ours）在两域同时取得最优（Embodied 62.4 +4.0 vs 混合、AD 63.3 +8.1 vs 混合）——渐进策略有效缓解干扰
- 有趣发现：Embodied 训练意外提升 AD（57.6 vs baseline 32.2），暗示 embodied 的空间推理对 driving 理解有正迁移

---
## 论文点评

### Strengths

1. **清晰验证跨域正迁移假说**：Ablation 设计合理，多阶段 vs 单阶段 vs 单域对比完整，正迁移的证据充分
2. **全面的 benchmark 覆盖**：29 个 benchmark 提供了可信的能力全景，是 embodied VLM 领域最全面的评估之一
3. **实用的工程启示**：token 效率（796 vs 4096）、多阶段训练策略、多任务 reward 设计等具有工程参考价值
4. **Qualitative demo 有说服力**：导航和操作的实际部署展示了从 benchmark 到实际应用的闭环

### Weaknesses

1. **Method novelty 有限**：架构直接继承 MiMo-VL，训练策略是 SFT→CoT→RL 的 standard pipeline，核心贡献更多是数据工程和训练顺序
2. **跨域正迁移的机制解释不足**：只知道多阶段训练 work，但为什么 embodied 训练能提升 AD、正迁移的具体信号通路（spatial reasoning? scene understanding?）没有深入分析
3. **无端到端 action 能力**：MiMo-Embodied 本质是 VLM（输出文本/坐标），不直接输出 low-level action，需要下游 policy 配合
4. **数据构成不透明**：General 数据继承 MiMo-VL 但规模和比例未披露，各阶段数据量未明确，reproducibility 受限
5. **Autonomous driving 部分 benchmark 选择偏 QA-based**：大多数 AD benchmark 是 VQA 形式，与实际 driving 能力有 gap

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only（评估 pipeline 基于 lmms-eval）
- **模型权重**: MiMo-Embodied-7B（HuggingFace 发布）
- **训练细节**: 仅高层描述，超参已披露（Table 1），但数据规模和比例未公开
- **数据集**: 使用公开数据集组合 + MiMo-VL 私有语料，整体不可完全复现

#### Claim 可验证性
- ✅ 29 benchmark SOTA：评估代码已开源，可独立复现
- ✅ 多阶段训练优于直接混合：Ablation 设计合理，数据点充分
- ⚠️ "strong positive transfer"：ablation 支持跨域训练有益，但 transfer 的 mechanism 缺乏分析
- ⚠️ "first cross-embodied foundation model"：取决于 "cross-embodied" 的定义范围——如果只指 embodied AI + AD 的统一 VLM 确实是首个

---
## 关联工作
### 基于
- MiMo-VL: 架构和通用数据全部继承自 MiMo-VL（7B-SFT-2508），是直接的上游 base model

### 对比
- [[2507-RoboBrain2|RoboBrain 2.0]]: Embodied VLM baseline，7B 参数，在 affordance 和 spatial 上被超越
- [[2506-VeBrain|VeBrain]]: Embodied VLM，8B 参数，专注 spatial understanding
- RoboTron-Drive: AD 专用 VLM，8B 参数
- DriveLMM-o1: AD 专用 VLM，8B 参数，step-by-step reasoning
- GPT-4o / Claude Sonnet 4 / Gemini 2.5 Pro: Closed-source general VLM baselines
- InternVL3.5 / Qwen2.5-VL: Open-source general VLM baselines

### 方法相关
- GRPO: Stage 4 RL fine-tuning 使用的 policy optimization 算法（来自 DeepSeek-R1）
- [[2503-CosmosReason1|Cosmos-Reason1]]: 提供跨 embodiment 物理推理数据，含 DeepSeek-R1 生成的长链推理 trace

---
## Notes

