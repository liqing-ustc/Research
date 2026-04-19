---
title: "Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer"
authors: [Gemini Robotics Team]
institutes: [Google DeepMind]
date_publish: 2025-10
venue: arXiv
tags: [VLA, cross-embodiment, embodied-reasoning]
paper: https://arxiv.org/abs/2510.03342
website: https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/
github: https://github.com/google-deepmind/gemini-robotics-sdk
rating: 3
date_added: "2026-04-16"
---
## 速查卡片

> [!summary] Gemini Robotics 1.5
> - **核心**: 多 embodiment VLA + embodied reasoning VLM 组成 agentic robot 系统，引入 Motion Transfer 和 Thinking VLA
> - **方法**: Motion Transfer (MT) 训练范式实现跨 embodiment 技能迁移；VLA 生成多层级 thinking traces 后再出动作；GR-ER 1.5 作为 orchestrator 配合 GR 1.5 作为 action model
> - **结果**: 跨 3 种机器人 (ALOHA, Bi-arm Franka, Apollo) 零样本技能迁移；Thinking VLA 在多步任务上大幅提升；GR-ER 1.5 在 15 个 embodied reasoning benchmark 上 SOTA
> - **Sources**: [paper](https://arxiv.org/abs/2510.03342) | [website](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) | [github](https://github.com/google-deepmind/gemini-robotics-sdk)

---
## Summary

Gemini Robotics 1.5 通过 Motion Transfer 训练范式、Thinking VLA 和 SOTA embodied reasoning VLM 三大创新，构建了一个能跨 embodiment 泛化、能推理后行动、能处理长周期复杂任务的 agentic robot 系统。

**Key Takeaways:**
1. **Motion Transfer (MT)**: 新的模型架构和训练范式使 VLA 能从异构多 embodiment 数据中学习，实现跨机器人形态的零样本技能迁移（ALOHA/Franka/Apollo 之间互相迁移）
2. **Thinking VLA**: VLA 在输出动作前生成多层级自然语言 thinking traces（任务分解 → 下一步预测 → 运动描述），显著提升多步任务的成功率和可解释性
3. **Agentic 架构**: GR-ER 1.5 (VLM orchestrator) + GR 1.5 (VLA action model) 组成 agentic system，支持工具调用、长周期规划和错误恢复，在 8 个长周期任务上 progress score 接近 80%

**Teaser. GR 1.5 系统概览：GR-ER 1.5 作为 orchestrator 进行高层推理和规划，GR 1.5 作为 action model 将指令转化为跨 embodiment 的动作执行。**
![](https://lh3.googleusercontent.com/vSMY895gzWpK-nGK2Vn__6OsjX5DIlTkiBBqggd0f-oAw1oNgh5tFPiZPKtSKIONx4UCT7kiop72Nuj0RMrJ-bICxx21SZCG55LlqcqBA-n_znJw=w2880-rw-lo)

---
## Method Overview

### Model & Architecture

GR 1.5 model family 包含两个互补模型：

- **Gemini Robotics 1.5 (GR 1.5)**: VLA 模型，将中短周期语言指令翻译为机器人动作。支持 open-vocabulary 指令，能在 action 前进行推理，能原生控制多种 embodiment（ALOHA, Bi-arm Franka, Apollo humanoid）
- **Gemini Robotics-ER 1.5 (GR-ER 1.5)**: VLM，optimized for embodied reasoning——任务规划、空间推理、进度估计。支持 native tool calling（Search、代码执行、function calling）

**Agentic System Architecture** 由 orchestrator + action model 组成：
- **Orchestrator (GR-ER 1.5)**: 处理用户输入和环境反馈，将复杂任务分解为 VLA 可执行的简单步骤，执行 success detection 决定何时切换步骤，可调用数字工具获取外部信息
- **Action model (GR 1.5)**: 将 orchestrator 下发的自然语言指令转化为低层机器人动作，作为 orchestrator 的 specialized tool

### Embodied Thinking

Embodied Thinking 贯穿 VLM 和 VLA 两个模型：
- **GR-ER 1.5**: 利用 Gemini 的 thinking 能力进行高层规划，将复杂任务拆解为粗粒度计划，自适应更新计划，或调用外部工具
- **GR 1.5 (Thinking VLA)**: VLA 在指令和感知基础上生成自然语言 thinking traces，append 到 context window 后再输出动作。多层级思考包括：
  - 任务分解："将复杂指令拆为短周期子任务"
  - 下一步预测："pick up the rain jacket from the wardrobe"
  - 运动描述："move the gripper to the left"

### Motion Transfer (MT)

新的模型架构和训练范式，使模型能从不同机器人和数据源中学习，形成对运动和物理交互效果的统一理解。训练数据包含 ALOHA、Bi-arm Franka 和 Apollo humanoid 的多 embodiment 数据，以及公开可用的文本、图像和视频数据集。

---
## Gemini Robotics 1.5 is a general multi-embodiment Vision-Language-Action Model

全面评估基于 230 个任务的 benchmark，覆盖所有 embodiment，报告 progress score（0-1 的连续指标）。

### Generalization

GR 1.5 在四类泛化维度上全面超越 [[2503-GeminiRobotics|Gemini Robotics]] 和 Gemini Robotics On-Device (GRoD)：
- **Visual Generalization**: 对背景、光照、干扰物、纹理变化的鲁棒性
- **Instruction Generalization**: 对同义改写、拼写错误、多语言、不同详细程度的指令理解
- **Action Generalization**: 对新初始条件、新物体实例的运动适应
- **Task Generalization**: 在全新环境中执行全新任务——同时需要以上三种能力

### Motion Transfer Ablation

消融实验建立两个 baseline：单 embodiment 数据训练（无 MT）和多 embodiment 数据训练（无 MT）。结果表明：
- 多 embodiment 数据本身能提升性能
- MT 训练范式进一步放大了跨 embodiment 数据的正向迁移效果
- 对 Bi-arm Franka（中等数据量）效果最显著；对 humanoid（数据稀缺、embodiment gap 最大）MT 效果相对较弱

### Learning across different robot embodiments

GR 1.5 展现了零样本跨 embodiment 技能迁移：ALOHA 上只有 Franka 数据训练的任务也能完成，反之亦然。Humanoid 也能执行仅在其他 embodiment 数据中出现的技能，尽管其形态差异更大。

**Video. 跨 embodiment 技能迁移演示**
![](https://www.youtube.com/watch?v=9FV5ZYytkOQ)

### Thinking Helps Acting

Thinking 模式在多步任务上带来显著提升。性能增益来自两步分解：
1. 先将复杂多步语言指令转化为具体的短周期子任务（利用 VLM backbone 的语言-视觉能力）
2. 再将低层语言命令映射为动作（更简单的映射）

附加优势：
- **可解释性**: 可视化 thinking traces 来检查计划和预测下一步
- **隐式 success detection**: 模型自动感知子任务完成并切换目标
- **错误恢复**: 抓取失败时自动重新规划（如瓶子滑落后立即生成 "用左手捡起"）

---
## Gemini Robotics-ER 1.5 is a generalist embodied reasoning model

### Generality

GR-ER 1.5 是 generalist embodied reasoning model：在保持 frontier model 的广泛能力（MMMU, GPQA, Aider Polyglot）的同时，在 embodied reasoning 上达到 SOTA。在 generality vs embodied reasoning 的 Pareto frontier 上，GR-ER 1.5 (Thinking On) 扩展了边界。

### Frontier capabilities for Embodied Reasoning

**Complex Pointing**: 将 pointing 与推理结合。GR-ER 1.5 在 5 个学术 benchmark（Point-Bench, RefSpatial, RoboSpatial, Where2Place, PixMo Count）上 SOTA，尤其擅长需要物理、空间和语义约束推理的复杂 pointing 任务。

**Video. GR-ER 1.5 的多种 embodied reasoning 能力**

<video src="https://storage.googleapis.com/gdm-deepmind-com-prod-public/media/media/er_capabilities_sept24_NeAsztE.mp4#t=0.1" controls muted playsinline width="720"></video>

**Progress Understanding and Success Detection**: GR-ER 1.5 能预测任务完成百分比、多视角 success detection、视频帧排序。在 real-time 和 offline、multiview 和 singleview 四种 success detection 设置下均表现强劲。

**Real-World Robotic Use Cases**: 在 Trusted Tester 提供的真实场景 benchmark 上（inventory shelf inspection、in-the-wild 目标检测和 pointing），GR-ER 1.5 优于 GR-ER 和其他 SOTA 多模态模型。

### Thinking

Thinking 对 embodied reasoning 的效果：
- GR-ER 1.5 的性能随 thinking token budget 增长而提升
- 最优 thinking 量因任务而异：image/video QA 从更长的 thinking traces 中受益更多，pointing 需要较少
- GR-ER 1.5 能自动调节不同任务的 thinking token 数量
- 相比 Gemini 2.5 Flash，GR-ER 1.5 在 inference-time compute 上的 scaling 更好——说明强 thinking 能力不一定转化为有效的 embodied reasoning

---
## Gemini Robotics 1.5: A Physical Agent

将 GR-ER 1.5 + GR 1.5 组合为完整 agentic system，在 8 个长周期任务上评估（ALOHA 和 Bi-arm Franka），任务涵盖工具调用、记忆、规划和灵巧操作。

三个配置对比：
- **GR 1.5 (Thinking On)**: 单独的 Thinking VLA
- **Agentic (Gemini 2.5 Flash + GR 1.5)**: 通用 VLM 作 orchestrator
- **Agentic (GR-ER 1.5 + GR 1.5)**: 完整的 GR 1.5 Agent

结果：GR 1.5 Agent 在所有任务上一致且显著优于其他两个配置。Thinking VLA 单独最高约 44% progress score，而 GR 1.5 Agent 常接近 80%。对复杂任务，GR 1.5 Agent 比 Gemini 2.5 Flash orchestrator 的 progress score 接近翻倍。

**Video. GR 1.5 agentic 能力演示**
![](https://www.youtube.com/watch?v=AMRxbIO04kQ)

**Table 1. Long-horizon 评估的失败模式分析**

| Subtask failure modes | Agent (Gemini 2.5 Flash as orchestrator) | Agent (GR-ER 1.5 as orchestrator) |
| --- | --- | --- |
| Planning | 25.5% | 9% |
| Success detection | 6% | 4% |
| Action | 13% | 9% |
| Total failure rates | 44.5% | 22% |

**Insights**: GR-ER 1.5 作为 orchestrator 在所有失败类别上均优于 Gemini 2.5 Flash，最大的提升来自 task-planning（25.5% → 9%）。说明通用 VLM 即使配合先进 VLA 也无法可靠完成长周期任务——embodied reasoning 专门化的 orchestrator 是关键。

---
## Responsible Development and Safety

多层安全机制：
- **Safe Human-Robot Dialog**: 继承 Gemini Safety Policies，对抗测试显示对有害内容有强 compliance
- **Semantic Action Safety**: ASIMOV-2.0 benchmark 升级版——改进覆盖率、新增视频模态和物理约束相关的 embodied reasoning 任务。GR-ER 1.5 在 ASIMOV-2.0 上优于 GR-ER
- **Auto-Red-Teaming (ART)**: 三模型博弈框架（Attacker + Target + AutoRater），自动发现 ER 模型的幻觉和安全漏洞。验证了 (1) Thinking 增强鲁棒性 (2) AutoRater 可靠纠错 (3) ART 数据可缓解幻觉

---
## 论文点评

### Strengths

1. **Cross-embodiment 迁移的实证突破**: 首次在 3 种形态差异较大的机器人（双臂桌面、双臂固定、humanoid）之间展示可靠的零样本技能迁移，且有量化 benchmark 支撑
2. **Thinking VLA 设计巧妙**: 将复杂指令到动作的 end-to-end 映射分解为"语言思考 + 简单动作映射"两步，既利用了 VLM backbone 的语言能力，又降低了 action mapping 的难度
3. **完整的 agentic stack 评估**: 不仅单独评估各模块，还在 8 个长周期多步任务上做了端到端 ablation（VLA alone vs generic VLM orchestrator vs specialized ER orchestrator），证明了 embodied reasoning orchestrator 的不可替代性
4. **安全研究与模型开发同步**: ASIMOV-2.0 benchmark 和 Auto-Red-Teaming 框架是 safety 领域的实质性贡献

### Weaknesses

1. **Dexterity 未提升**: 论文坦承 GR 1.5 虽然泛化性大幅提升，但灵巧操作能力与前代持平——说明 MT 范式主要在"广度"而非"精度"上有收益
2. **MT 机制不透明**: Motion Transfer 被称为"新架构和训练范式"，但具体做了什么（统一 action space？latent alignment？auxiliary loss？）完全未披露，无法评估其 scalability 和局限
3. **GR 1.5 VLA 不可用**: VLA 模型仅对 select partners 开放，不公开可用，严重限制了 reproducibility 和社区验证
4. **Humanoid 数据稀缺导致 MT 效果受限**: 论文自身承认 MT 对 humanoid 效果较弱（embodiment gap 太大），但未深入分析原因或提出解决方向

### 可信评估

#### Artifact 可获取性
- **代码**: inference+train（Safari SDK 提供完整工具链：checkpoint access、serving、evaluation、data upload、finetuning）
- **模型权重**: GR-ER 1.5 通过 Gemini API 在 Google AI Studio 可用；GR 1.5 VLA 仅限 Trusted Tester；Gemini Robotics On Device 从 SDK v2.4.1 可用
- **训练细节**: 仅高层描述——提到使用 ALOHA/Franka/Apollo 多 embodiment 数据 + 公开文本/图像/视频数据，但未披露具体数据配比、训练步数、超参数
- **数据集**: 私有（多 embodiment robot data 未公开；公开数据的使用比例未说明）

#### Claim 可验证性
- ✅ GR-ER 1.5 在 15 个学术 benchmark 上 SOTA：使用公开 benchmark，且对比了 GPT-5、Gemini 2.5 等强 baseline
- ✅ Cross-embodiment 零样本迁移：有量化 benchmark + 定性视频演示
- ⚠️ GR 1.5 在 230 任务 benchmark 上超越前代：benchmark 为内部定义，非公开标准，无法独立复现
- ⚠️ "Thinking" 带来的性能提升：bar chart 显示明确提升，但具体 thinking trace 内容的质量和 failure cases 未系统分析
- ❌ "A milestone towards solving AGI in the physical world"：营销话术，无可操作定义

---
## 关联工作
### 基于
- [[2503-GeminiRobotics|Gemini Robotics]]: GR 1.5 的前代，建立了 VLA + embodied reasoning 的基础框架
- Gemini 2.5: GR 1.5 family 构建在最新一代 Gemini 之上

### 对比
- Gemini Robotics On-Device (GRoD): 前代 on-device VLA，作为 Franka 和 Apollo 上的 baseline
- Gemini 2.5 Flash: 作为通用 VLM orchestrator baseline，证明专门化 ER 模型的必要性
- GPT-5 / GPT-5-mini: 在 embodied reasoning benchmark 上的 frontier 对比

### 方法相关
- ASIMOV benchmark: 机器人语义安全评估，GR 1.5 同步发布 ASIMOV-2.0 升级版

---
## Notes

