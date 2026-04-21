---
title: "Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning"
authors: [NVIDIA]
institutes: [NVIDIA]
date_publish: 2025-03
venue: arXiv
tags: [embodied-reasoning, VLM, spatial-reasoning]
paper: https://arxiv.org/abs/2503.15558
website: https://research.nvidia.com/labs/cosmos-lab/cosmos-reason1/
github: https://github.com/nvidia-cosmos/cosmos-reason1
rating: 2
date_added: 2026-04-16
---

## Summary

> [!summary] Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
> - **核心**: 面向 Physical AI 的推理 VLM，通过 physical common sense 和 embodied reasoning 两类能力定义 + 专用 SFT/RL 训练，让多模态 LLM 具备物理世界理解与具身决策能力
> - **方法**: 定义 physical common sense（Space/Time/Physics 三类 16 子类）和 embodied reasoning（4 能力 × 5 agent 类型）两套 ontology；基于 Qwen2.5-VL-7B 和 Nemotron-H-56B 做 Physical AI SFT + GRPO RL，用 MCQ 形式的 rule-based verifiable reward
> - **结果**: SFT 阶段 embodied reasoning 提升 10%+，RL 阶段在 intuitive physics（arrow of time, spatial puzzle, object permanence）上从近随机提升至 81.5%
> - **Sources**: [paper](https://arxiv.org/abs/2503.15558) | [website](https://research.nvidia.com/labs/cosmos-lab/cosmos-reason1/) | [github](https://github.com/nvidia-cosmos/cosmos-reason1)
> - **Rating**: 2 - Frontier（NVIDIA 推出的 Physical AI reasoning VLM，ontology + MCQ-RL 是有价值的范式，但 benchmark 自建自评、RL 规模有限、未成为领域 de facto 标准）

**Key Takeaways:**
1. **Ontology-driven 能力定义**: 提出 physical common sense（3 大类 16 子类）和 embodied reasoning（4 能力 × 5 agent 类型）两套 ontology，为 Physical AI 评估建立了结构化框架
2. **Rule-based verifiable reward for Physical AI RL**: 将 embodied reasoning 问题转化为 MCQ 格式，实现 rule-based 可验证奖励，绕过了 Physical AI 领域 reward design 的难题
3. **Intuitive physics 能力显著提升**: 在 arrow of time、spatial puzzle、object permanence 等任务上，现有 VLM 几乎随机猜测，而 Cosmos-Reason1 通过 SFT+RL 将平均准确率从 42% 提升至 81.5%

**Teaser. Cosmos-Reason1 overview**
![](https://research.nvidia.com/labs/cosmos-lab/cosmos-reason1/assets/teaser.svg)

![](https://www.youtube.com/watch?v=Eu25r-yisPc)

---
## Introduction

Physical AI 系统需要感知、理解并推理物理世界。当前 LLM 虽然在 coding 和 math 上推理能力强大，但在物理世界 grounding 方面存在关键局限——虽然通过海量文本可能获得了物理知识，但难以将知识与真实世界交互和动态联系起来。

Cosmos-Reason1 的核心思路：
- 定义 Physical AI 所需的基础能力（physical common sense + embodied reasoning）
- 提出两套 ontology 作为共享框架和评测标准
- 构建两个模型：Cosmos-Reason1-7B（基于 Qwen2.5-VL）和 Cosmos-Reason1-56B（基于 Nemotron-H）
- 两阶段训练：Physical AI SFT → Physical AI RL
- 探索用 MCQ 构建 rule-based verifiable reward 训练 Physical AI 推理

**Figure 1. Cosmos-Reason1 整体流程**
![](https://arxiv.org/html/2503.15558v3/x1.png)

---
## Physical AI Reasoning

Physical AI 推理包含两大能力：physical common sense reasoning 和 embodied reasoning。前者是 embodiment-agnostic 的环境理解，后者是面向具身决策的推理。两者都融合 System 1（快速直觉）和 System 2（慢速深思）的双系统思维。

### Common Sense Reasoning

提出三层 ontology：
- **Space**: 物体间关系、空间合理性、affordance、环境理解（4 子类）
- **Time**: 动作理解、事件顺序、因果关系、相机运动、规划（5 子类）
- **Fundamental Physics**: 物体属性、状态变化、object permanence、力学、电磁学、热力学、反物理（7 子类）

**Figure 2. Physical common sense ontology 饼图**
![](https://arxiv.org/html/2503.15558v3/x2.png)

### Embodied Reasoning

提出二维 ontology，覆盖 4 种能力 × 5 种 agent 类型：
- **能力维度**: Process Complex Sensory Inputs / Predict Action Effects / Respect Physical Constraints / Learn from Interactions
- **Agent 维度**: 人类、动物、机械臂、人形机器人、自动驾驶

本文聚焦前三种能力（Learn from Interactions 留作 future work），具体化为三个任务：
1. **Task-completion verification**: 判断任务是否完成
2. **Action affordance**: 评估某个 action 是否可行
3. **Next plausible action prediction**: 预测下一步最合理的 action

---
## Cosmos-Reason1

### Multimodal Architecture

采用 decoder-only 架构（类似 LLaVA / NVLM-D）：Vision Encoder → 2-layer MLP Projector（含 downsampling）→ LLM Backbone。

- **Cosmos-Reason1-7B**: 基于 Qwen2.5-VL，ViT-676M encoder，动态分辨率输入，dense Transformer backbone（28 层）
- **Cosmos-Reason1-56B**: InternViT-300M-V2.5 encoder + Nemotron-H hybrid Mamba-MLP-Transformer backbone（118 层），448×448 输入，最多 32 帧视频，PixelShuffle 2×2 下采样

**Figure 3. 多模态 LLM 架构**
![](https://arxiv.org/html/2503.15558v3/x3.png)

### Hybrid Mamba-MLP-Transformer Backbone

56B 模型使用 hybrid Mamba-MLP-Transformer 架构。Mamba 提供线性时间序列建模，少量 Transformer 层补充长上下文建模能力。

**Figure 4. Hybrid Mamba-MLP-Transformer backbone 架构**
![](https://arxiv.org/html/2503.15558v3/x4.png)

训练并行策略：7B 用 TP=4，56B 用 TP=8 + PP=2。

---
## Reinforcement Learning

### Algorithm

采用 GRPO 算法（无需训练单独的 critic model）。优势函数通过在同一 prompt 的 response group 内归一化 reward 计算：

**Equation 1. GRPO Advantage**

$$
A_{i}=\frac{R(o_{i})-\text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}
$$

**符号说明**: $R(o_i)$ 为 response $o_i$ 的 reward，$\mathcal{G}=\{o_1, \dots, o_G\}$ 为同一 prompt 生成的 response 组。

### Training Framework

提出全异步 RL 训练框架，包含三个核心部分：
1. **Dispatcher**: 调度训练数据、管理框架状态
2. **Actor Rollout**: 从 prompt 生成 response，计算 reward 和 advantage
3. **Policy Training**: 执行 GRPO，支持 5D 并行（DP, PP, CP, FSDP, TP）

关键优势：
- 异构部署策略（policy training 和 actor rollout 分离），相比 colocated 框架效率提升约 **160%**
- 节点故障时可快速重配继续训练，无需重启
- Dispatcher 冗余机制支持动态扩缩容

**Figure 5. RL 训练框架架构**
![](https://arxiv.org/html/2503.15558v3/x5.png)

---
## Data

### Physical AI Supervised Fine-Tuning

总计约 **4M** 标注的 video-text pairs。SFT 数据分三大类：

**Physical Common Sense VQA**: 通过 5 阶段 pipeline 构建——视频策展 → 详细 captioning → QA pair 生成 → DeepSeek-R1 提取 reasoning trace → 清洗重写。包括 99K free-form understanding + 59.4K reasoning SFT，以及 1.2M MCQ understanding + 605K MCQ reasoning。

**Embodied Reasoning**: 聚焦 task-completion verification、action affordance、next plausible action prediction 三个属性。数据来源包括：
- **BridgeData V2**: 129.2K clips，robot manipulation
- **RoboVQA**: 218.5K clips，robot + human demonstrations
- **AgiBot**: 19.4K clips，humanoid robot manipulation
- **HoloAssist**: 136.3K clips，egocentric human manipulation
- **AV**: 12.4K clips，autonomous driving（人工标注 caption）

**Figure 7. Embodied reasoning SFT 数据策展 pipeline（以 AgiBot 为例）**
![](https://arxiv.org/html/2503.15558v3/x6.png)

**Intuitive Physics**: 自监督构造的 SFT 数据
- **Spatial Puzzles**: 3K 视频 → 11K samples，将图片切为 2×2 patches 后打乱，模型推理正确空间位置
- **Arrow of Time (AoT)**: 30K 视频 + 反转版本，判断视频播放方向
- **Object Permanence**: 10K clips（Libero 机器人仿真），判断物体是否违反 object permanence 消失

### Physical AI Reinforcement Learning

RL 训练数据共 **30,304** 个 MCQ samples。Intuitive Physics 类数据天然适合 MCQ 格式且易于规模化；Common Sense 和 Embodied Reasoning 类数据通过人工转换为 MCQ，需要 human-in-the-loop 验证质量。

---
## Benchmark

共构建 **1,214** 个评测问题（604 common sense + 610 embodied reasoning），全部为 MCQ 格式。

**Physical Common Sense**: 从 5,737 个候选问题中人工挑选 604 个（来自 426 个视频），覆盖 Space（13.25%）、Time（49.33%）、Fundamental Physics（37.4%）。

**Figure 8. Common sense benchmark 问题类别分布**
![](https://arxiv.org/html/2503.15558v3/x7.png)

**Embodied Reasoning**: 6 个子 benchmark（BridgeData V2、RoboVQA、RoboFail、AgiBot、HoloAssist、AV），各 100 个问题。特别设计了 RoboFail 作为更困难的评测（需要高度观察力和 physical constraint 推理）。

---
## Experiments

### Physical AI Supervised Fine-Tuning

训练设置：7B 训练 12.5K iterations（lr 1e-5 → 1e-6 cosine），56B 训练 50K iterations（30K@1e-5 + 20K@1e-6），batch size 7B=256 / 56B=32。

**Table 7. Physical common sense benchmark 结果**

| Methods | Space | Time | Other Physics | Avg. |
| --- | --- | --- | --- | --- |
| Gemini 2.0 Flash | 53.8 | 50.0 | 46.9 | 50.2 |
| GPT-4o | 61.3 | 54.7 | 50.9 | 55.6 |
| OpenAI o1 | 63.8 | 58.1 | 58.0 | 59.9 |
| Qwen2.5-VL-7B | 48.8 | 56.4 | 37.2 | 47.4 |
| Nemotron-H-56B | 61.3 | 68.1 | 45.1 | 58.2 |
| Cosmos-Reason1-7B | 54.2 | 58.7 | 50.0 | 54.3 (+6.9) |
| Cosmos-Reason1-56B | 61.3 | 65.5 | 53.9 | 60.2 (+2.0) |

**Insights**: 56B 略胜 OpenAI o1（60.2 vs 59.9），SFT 对 7B 模型提升最显著（+6.9）。

**Table 8. Embodied reasoning benchmark 结果**

| Models | BridgeData V2 | RoboVQA | Agibot | HoloAssist | AV | RoboFail | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 2.0 Flash | 25.0 | 78.2 | 29.0 | 44.0 | 37.0 | 67.0 | 46.7 |
| GPT-4o | 42.0 | 71.8 | 32.0 | 65.0 | 46.0 | 63.0 | 53.3 |
| OpenAI o1 | 42.0 | 80.0 | 44.0 | 63.0 | 37.0 | 61.0 | 54.5 |
| Qwen2.5-VL-7B | 38.0 | 82.5 | 40.4 | 50.0 | 36.0 | 57.6 | 50.8 |
| Nemotron-H-56B | 37.0 | 77.2 | 37.0 | 65.0 | 41.0 | 64.0 | 53.5 |
| Cosmos-Reason1-7B | 58.8 | 83.8 | 49.4 | 63.0 | 55.6 | 60.0 | 61.8 (+11.0) |
| Cosmos-Reason1-56B | 65.0 | 80.0 | 47.6 | 57.8 | 65.8 | 66.2 | 63.7 (+10.2) |

**Insights**: SFT 带来 10%+ 的 embodied reasoning 提升。两个模型均大幅超过所有 baseline，表明 domain-specific SFT 的有效性。

### Physical AI Reinforcement Learning

RL 设置：batch size 128，每 prompt 采样 9 个 output，max length 6144 tokens，lr 4e-6，KL penalty 0.005，训练 500 iterations。

**Table 9. SFT + RL 对比**

| Models | Common Sense | BridgeData V2 | RoboVQA | Agibot | HoloAssist | AV | RoboFail | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos-Reason1-7B | 54.3 | 58.8 | 83.8 | 49.4 | 63.0 | 55.6 | 60.0 | 60.7 |
| + Physical AI RL | 56.2 | 73.5 | 86.8 | 54.2 | 60.0 | 67.0 | 62.0 | 65.7 (+5.0) |

**Table 10. Intuitive physics benchmark 结果**

| Models | Arrow of Time | Spatial Puzzle | Object Permanence | Avg. |
| --- | --- | --- | --- | --- |
| Random Guess | 50.0 | 25.0 | 50.0 | 41.7 |
| Gemini 2.0 Flash | 50.0 | 31.0 | 48.0 | 43.0 |
| GPT-4o | 50.0 | 77.0 | 48.0 | 58.3 |
| OpenAI o1 | 51.0 | 64.0 | 49.0 | 54.7 |
| Qwen2.5-VL-7B | 50.2 | 27.2 | 48.8 | 42.1 |
| Cosmos-Reason1-7B | 56.0 | 85.4 | 82.0 | 74.5 (+32.4) |
| + Physical AI RL | 64.5 | 94.0 | 86.0 | 81.5 (+7.0) |

**Insights**: 最striking的结果——现有 SOTA VLM（GPT-4o, o1, Gemini）在 arrow of time 和 object permanence 上几乎等于随机猜测。Cosmos-Reason1 通过专门的 intuitive physics 数据显著提升，RL 进一步强化。RoboFail 表现停滞，作者归因于训练数据覆盖不足（需要 highly observant perception 和 complex affordance reasoning）。

---
## 关联工作
### 基于
- Qwen2.5-VL: 7B 模型的 backbone VLM
- Nemotron-H: 56B 模型的 hybrid Mamba-MLP-Transformer backbone
- DeepSeek-R1: reasoning trace 蒸馏来源 + GRPO 算法来源
- InternViT-300M-V2.5: 56B 模型的 vision encoder

### 对比
- GPT-4o, OpenAI o1, Gemini 2.0 Flash: 通用 VLM baseline
- Qwen2.5-VL-7B: 7B backbone 的 before/after 对比
- Nemotron-H-56B: 56B backbone 的 before/after 对比

### 方法相关
- GRPO: RL 算法（group-relative advantage，无 critic model）
- LLaVA / NVLM-D: decoder-only 多模态架构
- BridgeData V2 / RoboVQA / AgiBot / HoloAssist: embodied reasoning 训练数据来源
- Libero: object permanence 仿真数据来源

---
## 论文点评

### Strengths

1. **Problem formulation 清晰且有价值**: Physical AI 能力的 ontology 定义（特别是 physical common sense 的 16 子类）为领域提供了结构化的评测和训练框架，避免了 ad-hoc benchmark 的问题
2. **Intuitive physics 发现有信息量**: 揭示现有 SOTA VLM 在 arrow of time、object permanence 上接近随机猜测，说明现有 benchmark 未能真正评估物理世界理解
3. **Rule-based verifiable reward 的 Physical AI 适配**: 将 DeepSeek-R1 式的 MCQ reward 机制迁移到 embodied reasoning 领域，提供了一条可行的 RL post-training 路径
4. **异步 RL 训练框架**: 160% 效率提升和容错设计是工程上的实质贡献
5. **开源**: 7B 模型权重 + 代码 + 训练 recipe 开放，降低复现门槛

### Weaknesses

1. **Benchmark 自建自评**: 训练数据和评测 benchmark 均由同一团队构建，存在 evaluation bias 风险——ontology 定义的 capability 和 benchmark 的 question 分布可能天然偏向自己的训练数据分布
2. **RL reward 的局限**: MCQ 格式的 rule-based reward 限制了 RL 训练的规模化和泛化——common sense 和 embodied reasoning 的 MCQ 数据量仅几千条，需要 human-in-the-loop，难以 scale
3. **56B 模型缺少 RL 结果**: RL 实验仅在 7B 上进行，56B 模型的 RL 收益未知
4. **Embodied reasoning 停留在语言层**: 模型输出 natural language 的决策，未直接生成 action token，与 VLA end-to-end 范式有 gap。"next action prediction" 的准确率（BridgeData V2 最高 73.5%）离实际部署还有距离
5. **RoboFail 表现暴露泛化瓶颈**: 作者坦承 RoboFail 上 SFT 和 RL 均无显著提升，说明模型在 OOD 的 hard cases 上泛化能力有限

### 可信评估

#### Artifact 可获取性
- **代码**: inference + post-training（含 SFT 和 RL recipe）
- **模型权重**: Cosmos-Reason1-7B（HuggingFace），56B 未公开
- **训练细节**: 超参 + 数据配比 + 训练步数完整披露
- **数据集**: SFT 数据基于公开数据集（BridgeData V2, RoboVQA, AgiBot, HoloAssist）+ 私有 AV 数据；RL 数据未直接开源但 recipe 公开

#### Claim 可验证性
- ✅ Physical AI SFT 提升 10%+: 有完整 benchmark 数字和 ablation
- ✅ 现有 VLM 在 intuitive physics 接近随机: 提供了 GPT-4o / o1 / Gemini 的评测数据
- ⚠️ "160% RL 训练效率提升": 仅声明，未给出具体 wall-clock 对比数据
- ⚠️ Cosmos-Reason1-56B "略优于 OpenAI o1": 差距极小（60.2 vs 59.9），且 benchmark 自建，统计显著性存疑

### Notes

### Rating

**分数**：2 - Frontier
**理由**：属于 Physical AI reasoning VLM 的代表工作之一——NVIDIA 品牌效应 + 开源 7B 权重 + ontology 框架让它成为该方向的重要参考，Strengths 指出的 intuitive physics 发现和 MCQ-RL 范式有方法论贡献。但未达到 Foundation 档：benchmark 自建自评、RL 数据仅 3 万条难以 scale（Weaknesses 2）、56B 没 RL 结果（Weaknesses 3）、输出停留语言层没有闭环到 action（Weaknesses 4），整体更像一个能力探索而非 de facto 标准。高于 Archived 档的理由是方法并未过气，社区仍在引用其 ontology 和 MCQ-reward 设计作为 Physical AI reasoning 的参考点。
