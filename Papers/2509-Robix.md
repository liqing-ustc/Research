---
title: "Robix: A Unified Model for Robot Interaction, Reasoning and Planning"
authors: [Huang Fang, Mengxi Zhang, Heng Dong, Wei Li, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, Hang Li]
institutes: [ByteDance Seed]
date_publish: 2025-09
venue: arXiv
tags: [embodied-reasoning, task-planning, VLM]
paper: https://arxiv.org/abs/2509.01106
website: https://robix-seed.github.io/robix/
github:
rating: 2
date_added: 2026-04-16
---
## 速查卡片

> [!summary] Robix: A Unified Model for Robot Interaction, Reasoning and Planning
> - **核心**: 将 robot reasoning、task planning、human-robot interaction 统一到单个 VLM 中，作为层级式机器人系统的高层认知模块
> - **方法**: 三阶段训练（continued pretraining → SFT with synthesized interaction data → RL with GRPO + thought-action consistency reward）
> - **结果**: 在 offline/online 评测中超越 GPT-4o 和 Gemini-2.5-Pro，OOD 泛化能力突出
> - **Sources**: [paper](https://arxiv.org/abs/2509.01106) | [website](https://robix-seed.github.io/robix/)

---
## Summary

Robix 是一个统一的 VLM，将机器人推理、任务规划和自然语言交互整合进单一端到端架构，充当层级式机器人系统的"大脑"，在 interactive task execution 上全面超越商业和开源 baseline。

**Key Takeaways:**
1. **统一架构替代模块化 pipeline**: Robix 不再拼接 planning + interaction + reasoning 的独立模块，而是用单个 VLM 在每个 iteration 同时输出 thought、action command 和 verbal response，实现 proactive dialogue、interruption handling、dynamic replanning 等交互能力
2. **三阶段训练策略是关键**: Continued pretraining（200B tokens，覆盖 3D spatial understanding / visual grounding / task-centric reasoning）→ SFT（合成 7 类 human-robot interaction 数据 + CoT reasoning traces）→ RL（GRPO + thought-action consistency reward），其中 CoT 和 RL 各自贡献显著的 OOD 提升
3. **RL 的核心设计是 thought-action consistency reward**: 用外部 LLM 评估模型生成的 thought 与 action 是否逻辑一致，负奖励惩罚不一致，配合 reward variance filtering 提升训练效率

**Teaser. Robix 交互式任务执行 demo**
![](https://www.youtube.com/watch?v=-uEDN31Ne_Y)

---
## Introduction

机器人系统需要在开放动态环境中执行复杂日常任务（如收拾餐桌），这要求同时具备：(1) 理解复杂指令 + commonsense reasoning，(2) long-horizon task planning，(3) 自然的 human-robot interaction（包括处理中断、主动澄清歧义）。

现有方法的两大局限：
- **模块化 pipeline**（将 LLM/VLM 仅用于 task decomposition）：灵活性差，忽略交互和 embodied reasoning
- **workflow-based 系统**：依赖手工设计，脆弱且不可扩展

Robix 的核心思路：用单个 VLM 统一 reasoning + planning + interaction，以 chain-of-thought reasoning 驱动，formulate interactive task execution 为 unified reasoning-action sequence。

**Figure 1. Robix 交互式任务执行 demo，展示 (1) 复杂指令理解 + commonsense reasoning；(2) 实时中断处理；(3) 任务状态监控与动态重规划；(4) 主动对话澄清歧义指令**
![](https://robix-public.bytedance.com/static/images/online_images/demo-cases.jpg)

---
## The Robix Model

Robix 在层级式机器人系统中担任高层认知模块（high-level cognitive layer），低层控制器（通常是 VLA 模型）执行 Robix 生成的原子命令。

**Figure 2. 层级式机器人系统架构**
![](https://robix-public.bytedance.com/static/images/online_images/model-architecture.png)

在每个 iteration，Robix 直接处理机器人摄像头的视觉观察和用户语音输入，选择性地输出：
- **Atomic action command**：发送给低层控制器执行
- **Verbal response**：回应用户

序列化决策过程建模为：

**Equation 1. Sequential decision-making**

$$
P\left(t_n, a_n, r_n \mid (o_1, u_1, t_1, a_1, r_1), \ldots, [(o_{n-i}, u_{n-i}, t_{n-i}, a_{n-i}, r_{n-i})]_{i=1}^{N}, o_n, u_n\right)
$$

**符号说明**：$t_n$ 为 thought（内部推理），$a_n$ 为 action command，$r_n$ 为 verbal response，$o_n$ 为 visual observation，$u_n$ 为 user instruction。
**含义**：每一步预测 thought + action + response，条件为当前观察、用户输入和完整交互历史。为了平衡 memory 和 inference 效率（32k context），只保留最近 $N$ 帧视觉观察作为显式输入。

---
## Training Recipe

基于 Qwen2.5-VL-7B 和 32B 进行 continual training，总计约 200B tokens，三阶段训练。

### Continued Pretraining

构建 200B tokens 的大规模预训练语料，覆盖 robot-relevant 和通用多模态能力。重点强化三个 embodied reasoning 维度：

**3D Spatial Understanding**: 30M+ instruction pairs（约 40B tokens），涵盖 5 类任务——multi-view correspondence、3D bounding box detection、relative depth sorting、absolute depth estimation、egomotion prediction。数据来源包括 Seed-1.5-VL 的 3D 训练语料和公开数据集（ScanNet、ScanNet++、3RScan、CA-1M、SUN RGB-D、ARKitScenes）。

**Visual Grounding**: 50M+ instruction-response pairs（约 70B tokens），覆盖 2D bounding box、point annotations、counting、visual prompt 四类任务。坐标统一归一化到 [0, 1000]。

**Task-centric Reasoning**: 5M+ examples（约 10B tokens），基于公开机器人和 egocentric 数据集（AgiBot、BridgeData V2、DROID、Egodex、Ego4D、RoboVQA、HoloAssist），针对 task status verification、action affordance、next action prediction 三类推理功能。用 Seed-1.5-VL-thinking 生成 step-by-step thought traces。

**General Multimodal Understanding**: 50M+ image-text pairs（80B+ tokens），涵盖 VQA、captioning、OCR。

**Instruction Tuning**: 1M high-quality examples，整合通用和 CoT instruction-following 数据。

训练分两阶段：Stage 1 在全量数据上 continue pretraining（含 5% text-only），cosine LR schedule（$1\times10^{-5} \to 1\times10^{-6}$），sequence length 32,768；Stage 2 在 curated instruction-following 数据上 tune，vision encoder frozen，LR 固定 $1\times10^{-6}$。

### Supervised Finetuning

核心挑战：缺乏大规模多轮 egocentric-vision 数据集来同时建模 human-robot interaction 和 task planning。

**解法：数据合成 pipeline**，包含两个模块：

**Interaction Synthesis** — 从两类数据源（teleoperated robot demonstrations + simulation & AIGC data）合成 7 类交互指令：
1. **Multi-Stage Instruction**: 包含 ≥10 个 atomic actions 的长轨迹
2. **Constrained Instruction**: 带约束条件的子任务指令
3. **Open-Ended Instruction**: 需要 commonsense 推理的开放式指令
4. **Anytime Interruption**: 随机注入用户中断并合成对应响应
5. **Invalid Instruction**: 4 类不可执行指令（不存在的物体、物理不可能、超出能力、危险指令）
6. **Ambiguous Instruction**: 需要主动澄清的模糊指令
7. **Chat Instruction**: 任务执行中穿插的对话

**Reasoning Synthesis** — 为每条交互数据生成 CoT reasoning traces，强调 4 个维度：scene understanding、task status reflection、long-term instruction following、next-step analysis。采用 ActRe + Thought Bootstrapping 方法，用 Seed-1.5-VL 生成简洁的 reasoning（≤200 tokens），并过滤幻觉和逻辑不一致的 traces。

### Reinforcement Learning

SFT 后的模型仍存在：(1) **irrational reasoning**（思维冲突、缺乏 commonsense）；(2) **thought-action inconsistency**（思考和行动脱节）。

采用 GRPO 进行 RL，两个核心策略：

**Co-training with General Visual Reasoning Data**: 混合 robot interaction data 和通用 visual reasoning data（task completion verification、action affordance、object localization 等），前者提升 OOD 泛化，后者缓解 irrational reasoning。

**Thought-Action Consistency Reward**: 除标准 format + accuracy reward 外，额外引入一致性 reward——用外部 LLM（Qwen-2.5-32B）评估每步生成的 thought 和 action 是否逻辑一致，不一致则给负奖励。

**Equation 2. RL data filtering（reward variance threshold）**

$$
\mathcal{D}_{\text{new}} = \left\{ (x_n, y_n^*) \in \mathcal{D} \;\middle|\; \text{Var}\left(\left\{R(y_n^{(i)}, y_n^*)\right\}_{i=1}^{M}\right) > \tau,\; y_n^{(i)} \sim \pi_{\text{SFT}}(\cdot \mid x_n) \right\}
$$

**含义**：过滤掉 reward variance 低的样本（$M=8$ 个候选答案的 reward variance $\leq \tau=0$），只保留对 policy 改进有信息量的训练数据。RL 训练使用 verl 框架。

---
## Experiments

### Fundamental Perception & Reasoning Evaluation

在 31 个公开 benchmark 上评测 Robix，覆盖 3D spatial understanding（8 个）、visual grounding（8 个）、embodied task-centric reasoning（6 个）、general multimodal understanding & reasoning（9 个）。

**Figure. Vision-language benchmark 性能对比**
![](https://robix-public.bytedance.com/static/images/online_images/tables.png)

关键结果：
- **3D Spatial Understanding**: Robix-7B/32B 在 7/8 个任务上超越 backbone（Qwen2.5-VL），平均准确率 73.4 / 75.8 vs. 66.9 / 70.7；超越 [[2503-CosmosReason1|Cosmos-Reason1]]-7B（64.0）和 RoboBrain-32B（72.2）
- **Visual Grounding**: Robix-7B/32B 在 LVIS-MG 上 F1 分别提升 39.6 和 25.0（绝对值），全面超越 backbone 和大部分商业模型
- **Task-centric Reasoning**: 在 Agibot-ER 上分别超越 backbone 12.8 和 7.2 个点，超越 [[2503-CosmosReason1|Cosmos-Reason1]]-7B 和 [[2507-RoboBrain2|RoboBrain-2.0]]-32B 达 23 和 8.3 个点
- **General Multimodal**: 保持 backbone 水平，部分 benchmark 有提升，但仍落后于大规模商业模型

### Offline Evaluation

设计三个评测集：(1) AGIBot Evaluation Set（16 个 OOD daily tasks），(2) Internal OOD Benchmark（16 个交互脚本），(3) Internal ID Benchmark（6 类任务）。评测采用 teacher-forcing，逐步预测 action 与 candidate action list 匹配。

**Figure. Offline evaluation results**
![](https://robix-public.bytedance.com/static/images/online_images/tables_offline.png)

**Table 3. Offline evaluation results**

| | AGIBot | Internal OOD | Multi. | Const. | Interrupt | Open. | Invalid (F1) | Replan (F1) |
|---|---|---|---|---|---|---|---|---|
| Gemini-2.5-Pro | 52.6 | **83.8** | 79.3 | 87.1 | 55.9 | 60 | 98.3 | 83.7 |
| GPT4-o | 45.9 | 77.0 | 76.1 | 84.4 | 44.8 | **66.7** | 79.2 | 73.7 |
| Qwen-2.5-VL-32B | 43.3 | 71.6 | 60.5 | 62.2 | 48.0 | 26.7 | 70.2 | 37.0 |
| RoboBrain-2.0-32B | 29.6 | 63.5 | 58.2 | 51.7 | 41.2 | 0.0 | 43.6 | 29.9 |
| Robix-7B-SFT-wo-R | 55.2 | 69.9 | 82.5 | 89.0 | 91.5 | 60.0 | **100** | 90.5 |
| Robix-7B-RL | 59.6 | 85.4 | 93.2 | 90.3 | 78.6 | 86.7 | 95.9 | 87.0 |
| Robix-32B-SFT | 83.5 | 89.3 | 93.0 | 89.7 | 80.0 | **100** | | 95.1 |
| Robix-32B-RL | **64.4** | **86.8** | **96.6** | **96.0** | **92.5** | **93.3** | **100** | **96.2** |

**Insights**:
- Robix-32B-RL 在所有评测集上排名第一，全面超越所有开源和商业 VLM
- **CoT reasoning 关键**：去掉 CoT 的 Robix-7B-SFT-wo-R 在 Internal OOD 上下降 7+ 个点，在 Open-Ended 上下降 26.7 个点
- **RL 关键**：Robix-7B-RL 和 32B-RL 在 Internal OOD 上分别比 SFT 版本提升 8.3 和 3.3 个点
- Gemini-2.5-Pro 是最强 baseline，在多数 baseline 方法中排名第一

### Online Evaluation

两组实验：(1) VLM + human UMI operator（排除低层控制器干扰），(2) Robix + GR-3 VLA on ByteMini robot。

**Figure 5. Online evaluation with UMI**
![](https://robix-public.bytedance.com/static/images/online_images/online_exp_2row_blue.png)

**UMI 设置结果**: Robix-32B 和 Gemini-2.5-Pro 各在 3/5 任务上排名第一，Robix-32B 平均 task progress 略高（92.6% vs. 91%），大幅超越 Qwen2.5-VL-32B（28%）。

**Figure 6. Online evaluation with GR-3 on ByteMini robot**
![](https://robix-public.bytedance.com/static/images/online_images/bar_chart_vla_blue.png)

**GR-3 设置结果**: Robix-32B 平均 task progress 92.5%，超越 Gemini-2.5-Pro 4.3 个百分点，超越 GPT-4o 28.1 个百分点。Baseline 方法（尤其 GPT-4o）会生成语义正确但 VLA 无法识别的 action（如 "put the biscuit box into the shopping basket" vs. VLA 只识别 "put the Oreo into the shopping basket"），VLM-VLA misalignment 是主要失败原因。

---
## 论文点评

### Strengths

1. **统一建模的实用价值高**: 将 reasoning + planning + interaction 统一到单个 VLM，避免了模块化系统的脆弱性和 workflow 的 hand-engineering，且 demo 展示的交互能力（中断处理、主动澄清、状态监控）在真实场景中非常实用
2. **数据合成 pipeline 设计精细**: 7 类交互指令 + 4 维 CoT reasoning traces 的合成方案覆盖了现实交互的主要模式，特别是 invalid instruction 和 ambiguous instruction 的合成增强了系统鲁棒性
3. **RL 设计有针对性**: Thought-action consistency reward 直接解决 SFT 模型的核心问题（推理和行动脱节），比纯 accuracy reward 更 fine-grained
4. **评测体系完善**: Offline（ID + OOD）+ Online（UMI + VLA）的评测设计，将 high-level planning 能力与 end-to-end 系统性能分开评估，方法论清晰

### Weaknesses

1. **数据和训练细节不够透明**: 合成数据的具体规模、质量分布、filtering 的通过率等关键信息缺失；200B tokens 的预训练数据中各类数据的比例配方也未详细披露
2. **低层控制器依赖强**: 整个系统的实际表现高度依赖低层 VLA（GR-3）的能力边界，且 VLM-VLA misalignment 问题（Robix 生成的 action 描述 VLA 无法执行）在论文中被归因于 baseline 但 Robix 自身也未完全解决
3. **长期记忆能力有限**: 论文自己承认 Robix 依赖 short-term context window（32k），对需要 long-term memory 的交互场景（跨 session 的偏好记忆等）无法支持
4. **OOD 泛化的边界不清晰**: AGIBot OOD 评测集 Robix-32B-RL 得分 64.4，虽是最高但绝对值不高，说明 OOD 泛化仍有很大空间，但论文对失败模式分析不足

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源
- **模型权重**: 未发布
- **训练细节**: 仅高层描述（三阶段训练的 LR / batch size / optimizer 参数有披露，但数据配比和合成 pipeline 细节不完整）
- **数据集**: 私有（合成数据 pipeline 和内部 teleoperation 数据均未开源；预训练用到了公开数据集但整合后的数据集未发布）

#### Claim 可验证性
- ✅ 在 31 个公开 benchmark 上的性能数字：可通过公开 benchmark 独立验证（如果模型开源）
- ✅ CoT reasoning 和 RL 各自的贡献：ablation（Table 3 中 SFT-wo-R vs. SFT vs. RL）提供了清晰的对比
- ⚠️ "outperforms GPT-4o and Gemini-2.5-Pro"：offline 评测基于内部设计的 benchmark（Internal OOD/ID），评测集未公开，难以独立复现；online 评测每个 task-model pair 仅重复 4 次，样本量有限
- ⚠️ 92.6% average task progress in online evaluation：task progress 由 human annotators 主观评估，评估标准和 inter-annotator agreement 未报告

---
## 关联工作

### 基于
- Qwen2.5-VL-7B/32B: 作为 backbone VLM 进行 continual training
- Seed-1.5-VL / Seed-1.5-VL-Think: 提供 3D spatial understanding 训练数据和 CoT thought trace 生成
- GR-3: ByteDance 内部 VLA 模型，作为低层控制器
- GRPO (DeepSeek-R1): RL 训练算法
- ActRe + Thought Bootstrapping ([[2501-UITARS|UI-TARS]]): CoT reasoning trace 合成方法

### 对比
- GPT-4o: 商业 VLM baseline，在 offline/online 评测中全面落后于 Robix-32B
- Gemini-2.5-Pro: 最强 baseline，offline 评测中在多数 baseline 方法中排名第一，online 评测与 Robix-32B 接近
- [[2507-RoboBrain2|RoboBrain-2.0]]: 开源 embodied reasoning 模型，在所有评测中落后
- [[2503-CosmosReason1|Cosmos-Reason1]]: NVIDIA 的 embodied reasoning 模型，在 task-centric reasoning 上落后明显

### 方法相关
- [[2502-HiRobot|Hi Robot]]: 层级式 VLM-VLA 系统，支持 open-ended instruction following，但依赖更复杂的框架
- RACER: VLM supervisor + physics simulation 用于 failure recovery
- [[2403-RTH|RT-H]]: 支持 language-based intervention 的层级架构

---
## Notes

