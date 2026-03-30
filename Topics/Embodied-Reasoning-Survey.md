---
title: "Embodied Reasoning Survey"
tags: [survey, embodied-reasoning, VLA, VLM, spatial-reasoning, RL]
date_updated: "2026-03-30"
year_range: 2023-2026
papers_analyzed: 18
---
## Overview

**Embodied Reasoning** 指 AI agent 在物理世界（或其仿真）中，基于感知输入进行推理并输出可执行动作的能力。它是连接 foundation model 的通用智能与具身控制的核心桥梁——不仅要 "看懂" 场景，还要 "想明白" 该怎么操作、怎么导航。

这一领域在 2023-2026 年间经历了爆发式增长。2023 年 RT-2 首次证明 VLM 的 web knowledge 可迁移为 robot action，开启了 VLA (Vision-Language-Action) 范式。2024 年，ECoT 和 SpatialVLM 分别从 chain-of-thought 和 spatial data 两个维度奠定了 embodied reasoning 的方法论基础。2025 年是 **RL for Embodied Reasoning 元年**：Robot-R1、Embodied-R、Embodied-R1 等工作将 DeepSeek-R1 的 GRPO 算法引入具身推理，证明 RL 系统性地优于 SFT。同年，Gemini Robotics 提出了业界首个 Embodied Reasoning QA (ERQA) benchmark，FoMER 和 EmbodiedBench 进一步完善了评测体系。

进入 2026 年，领域呈现出 **reasoning-action 深度融合** 的趋势：DM0 提出 Embodied-Native 预训练范式，Thinker 构建了 4.8M 级 robotics-specific 数据集，VLASER 系统性地揭示了 in-domain reasoning data 的关键作用。在导航侧，GTA、SpatialNav、PROSPECT 则从 explicit world representation、scene graph、spatial-semantic fusion 等角度推进 embodied spatial reasoning。

总体趋势可概括为三个 shift：(1) **from implicit to explicit reasoning** — 从端到端黑盒到可解释的推理链；(2) **from SFT to RL** — 从监督模仿到强化学习驱动的自主推理；(3) **from general to in-domain** — 从通用 VLM 能力到 embodied-specific 的数据和训练策略。

## 技术路线

### 路线 1：Chain-of-Thought Embodied Reasoning

**核心思路**：在 action prediction 之前插入结构化推理链（plan → subtask → spatial grounding → action），让模型 "先想后做"。

**代表论文**：
- **[[Papers/2407-ECoT|ECoT]]** (2024)：开创性工作。在 OpenVLA 中插入 6 步 embodied CoT（task plan → subtask → movement description → gripper position → target bbox → reasoning summary），用 Gemini+SAM 自动生成训练数据。7B 模型超越 RT-2-X (55B)，空间关系任务 +45%。
- **[[Papers/2512-Lumo1|Lumo-1]]** (2025)：将 reasoning trace 结构化为 bbox → keypoint → trajectory，并加入 GRPO RL 精炼。在 Astribot S1 双臂平台验证，超越 pi-0 baseline。
- **[[Papers/2602-DM0|DM0]]** (2026)：提出 Spatial Scaffolding（subtask → bbox → trajectory → action）作为 coarse-to-fine 推理链，同时用 gradient decoupling 保护 VLM reasoning 不被 action training 侵蚀。

**优势**：可解释性强，支持人工干预和纠错（ECoT 人工纠正 +48%），推理结构可泛化到新任务。

**劣势**：固定推理步骤不够灵活，额外推理延迟（Lumo-1 full reasoning mode 延迟较高），依赖复杂的数据生成 pipeline。

### 路线 2：RL-based Embodied Reasoning (GRPO 范式)

**核心思路**：用 reinforcement learning（特别是 GRPO）训练 VLM 进行具身推理，让模型从 outcome feedback 中学习 reasoning，而非仅从人类标注中模仿。

**代表论文**：
- **[[Papers/2506-RobotR1|Robot-R1]]** (NeurIPS 2025)：将 next-state prediction 重构为 MCQ，降低 RL 探索复杂度。7B 模型超越 GPT-4o，SFT 0% vs RL 11.68%。
- **[[Papers/2504-EmbodiedR|Embodied-R]]** (2025)：解耦 perception (72B VLM) 和 reasoning (3B LM)，提出 logical consistency reward（用 reference model 验证推理链一致性，无需视觉输入）。3B 超越 OpenAI-o1 和 Gemini-2.5-Pro。仅需 5,000 训练样本。
- **[[Papers/2508-EmbodiedR1|Embodied-R1]]** (2025)：用 "pointing"（2D 坐标）作为 embodiment-agnostic 中间表示，两阶段 GRPO 训练。3B 模型超越 7B-13B baselines。
- **[[Papers/2512-ETPR1|ETP-R1]]** (2025)：首次将 GRPO 应用于 graph-based VLN-CE，在 R2R-CE 达到 65% SR。

**核心发现**：RL 系统性优于 SFT。Robot-R1 (SFT 0% vs RL 11.68%)、Embodied-R1 (65.50% vs 41.25%)、Embodied-R (OOD 泛化更强) 一致验证了这一结论。

**优势**：小模型可以击败大模型（3B > o1, 7B > GPT-4o），数据效率高（Embodied-R 仅需 5K 样本），泛化能力更强。

**劣势**：绝对成功率仍然较低（Robot-R1 仅 11.68%），多数仅在仿真验证，MCQ 离散化可能丢失精细空间信息。

### 路线 3：Data-Centric Embodied Reasoning

**核心思路**：通过大规模、高质量的 embodied-specific 数据驱动 reasoning 能力提升，而非单纯依赖模型架构或训练算法创新。

**代表论文**：
- **[[Papers/2401-SpatialVLM|SpatialVLM]]** (CVPR 2024)：自动化 3D spatial VQA 数据生成 pipeline，从 10M 真实图像生成 20 亿空间 VQA 样本。首个 internet-scale metric-space 空间推理数据集。
- **[[Papers/2601-Thinker|Thinker]]** (IROS 2025)：构建 4.8M 实例的 robotics-specific 数据集，覆盖 ego-view reasoning、visual grounding、spatial understanding、CoT planning。10B 模型超越 32B baselines。
- **[[Papers/2510-VLASER|VLASER]]** (2025)：系统性研究哪些 VLM embodied reasoning 能力可迁移到 VLA 控制。**关键发现：OOD reasoning data 几乎无法迁移到 VLA performance，in-domain reasoning data 才是关键驱动力**。

**优势**：可 scale，data pipeline 可复用，实证基础扎实。

**劣势**：in-domain 数据生成依赖目标环境仿真标注（新环境成本不清），SpatialVLM 定量精度仍有限（37.2% within 0.5x-2x）。

### 路线 4：Explicit Spatial Representation for Reasoning

**核心思路**：为 MLLM/VLM 提供显式的空间表征（scene graph、TSDF map、3D query），将空间推理从模型内部的隐式学习转化为基于结构化表征的显式推理。

**代表论文**：
- **[[Papers/2602-GTA|GTA]]** (2026)：构建 interactive metric world representation（TSDF + topological graph），用 counterfactual reasoning 和 ray-casting 确保动作物理合理性。SPL +16.4。
- **[[Papers/2601-SpatialNav|SpatialNav]]** (2026)：层级 Spatial Scene Graph (floor → room → object) + agent-centric spatial map。Zero-shot VLN 达到 64.0% SR，接近 supervised SOTA。
- **[[Papers/2603-PROSPECT|PROSPECT]]** (2026)：CUT3R (3D spatial) + SigLIP (2D semantic) 通过 cross-attention 融合，latent predictive representation 实现 "预判式推理"。长程任务 (100+ steps) SR +4.14%。
- **[[Papers/2507-MTU3D|MTU3D]]** (2025)：统一 3D visual grounding 和 active exploration 为同一决策框架，在 4 个导航 benchmark 达到 SOTA。

**优势**：空间推理精度高，可泛化到新环境（SpatialNav zero-shot 接近 supervised），physically grounded。

**劣势**：依赖外部感知模块（depth estimation、3D reconstruction），实时性受限，场景表征构建成本高。

## 发展时间线

| 时间 | 里程碑 | 论文 |
|:-----|:-------|:-----|
| 2023.07 | VLA 范式开创：VLM → robot action | [[Papers/2307-RT2\|RT-2]] |
| 2024.01 | Internet-scale spatial reasoning data | [[Papers/2401-SpatialVLM\|SpatialVLM]] |
| 2024.07 | Embodied CoT 范式提出 | [[Papers/2407-ECoT\|ECoT]] |
| 2025.02 | 综合 embodied agent benchmark | [[Papers/2502-EmbodiedBench\|EmbodiedBench]] |
| 2025.03 | Gemini Robotics 提出 ERQA benchmark | [[Papers/2503-GeminiRobotics\|Gemini Robotics]] |
| 2025.04 | Perception-reasoning 解耦 + logical consistency reward | [[Papers/2504-EmbodiedR\|Embodied-R]] |
| 2025.06 | GRPO for embodied reasoning (NeurIPS) | [[Papers/2506-RobotR1\|Robot-R1]] |
| 2025.07 | 统一 grounding-exploration 决策框架 | [[Papers/2507-MTU3D\|MTU3D]] |
| 2025.08 | Pointing as embodiment-agnostic reasoning | [[Papers/2508-EmbodiedR1\|Embodied-R1]] |
| 2025.09 | Step-by-step reasoning 质量评估 | [[Papers/2509-FoMER\|FoMER]] |
| 2025.10 | In-domain reasoning data 关键性验证 | [[Papers/2510-VLASER\|VLASER]] |
| 2025.12 | GRPO for VLN + 结构化推理 VLA | [[Papers/2512-ETPR1\|ETP-R1]], [[Papers/2512-Lumo1\|Lumo-1]] |
| 2026.01 | 10B embodied VLM + hierarchical scene graph VLN | [[Papers/2601-Thinker\|Thinker]], [[Papers/2601-SpatialNav\|SpatialNav]] |
| 2026.02 | Embodied-Native VLA + explicit world representation VLN | [[Papers/2602-DM0\|DM0]], [[Papers/2602-GTA\|GTA]] |
| 2026.03 | Streaming spatial-semantic VLN | [[Papers/2603-PROSPECT\|PROSPECT]] |

## Paper Comparison

| Paper | Venue | 技术路线 | 核心方法 | 关键结果 | 局限性 |
|:------|:-----|:---------|:---------|:---------|:-------|
| [[Papers/2307-RT2\|RT-2]] | arXiv 2023 | CoT reasoning | VLM → action tokens, emergent reasoning | 首次 web knowledge → robot control | 端到端黑盒，无显式推理 |
| [[Papers/2401-SpatialVLM\|SpatialVLM]] | CVPR 2024 | Data-centric | 2B spatial VQA from 10M images | 首个 internet-scale 空间推理数据集 | 定量精度有限 (37.2%) |
| [[Papers/2407-ECoT\|ECoT]] | arXiv 2024 | CoT reasoning | 6-step embodied CoT + auto data gen | 7B > RT-2-X (55B), 空间任务 +45% | 固定推理结构，额外延迟 |
| [[Papers/2502-EmbodiedBench\|EmbodiedBench]] | ICML 2025 | Benchmark | 4 环境, 1128 任务, 24 MLLMs | GPT-4o 仅 28.9%, manipulation 15.7% | 仿真 only, 天花板低 |
| [[Papers/2503-GeminiRobotics\|Gemini Robotics]] | arXiv 2025 | CoT + Benchmark | Gemini 2.0 VLA + ERQA benchmark | 首个 embodied reasoning 专用评测 | 闭源, 无法复现 |
| [[Papers/2504-EmbodiedR\|Embodied-R]] | arXiv 2025 | RL (GRPO) | 72B 感知 + 3B 推理, logical consistency reward | 3B > o1 (51.1% vs 37.2%), 仅需 5K 数据 | 72B 推理开销, 文本表征损失空间细节 |
| [[Papers/2506-RobotR1\|Robot-R1]] | NeurIPS 2025 | RL (GRPO) | Next-state prediction as MCQ + GRPO | 7B > GPT-4o, SFT 0% vs RL 11.68% | 仅 tabletop sim, 绝对成功率低 |
| [[Papers/2507-MTU3D\|MTU3D]] | arXiv 2025 | Spatial repr. | 统一 grounding-exploration 决策 | 4 个导航 benchmark SOTA | 仅导航任务 |
| [[Papers/2508-EmbodiedR1\|Embodied-R1]] | arXiv 2025 | RL (GRPO) | Pointing + 两阶段 GRPO | 3B > 7B-13B, 87.5% real success | 依赖 depth, 轨迹采样粗糙 |
| [[Papers/2509-FoMER\|FoMER]] | arXiv 2025 | Benchmark | 10-dim reasoning quality evaluation | 最佳 o4-mini 76.3%, 人类 84.5% | 小规模 (1112), 仅 single-step |
| [[Papers/2510-VLASER\|VLASER]] | arXiv 2025 | Data-centric | VLASER-6M data engine + domain analysis | OOD reasoning 不迁移, in-domain 是关键 | 仿真 only, 新环境成本不清 |
| [[Papers/2512-ETPR1\|ETP-R1]] | arXiv 2025 | RL (GRPO) | GRPO for graph-based VLN-CE | R2R-CE 65% SR | 推理提升有限 |
| [[Papers/2512-Lumo1\|Lumo-1]] | arXiv 2025 | CoT + RL | Structured trace + GRPO | 超越 pi-0, real dual-arm 验证 | 闭源, 高延迟, 128 H100 |
| [[Papers/2601-Thinker\|Thinker]] | IROS 2025 | Data-centric | 4.8M robotics dataset, 10B VLM | 7B/10B > RoboBrain2-32B | 仅 4 页短文, 无 real robot |
| [[Papers/2601-SpatialNav\|SpatialNav]] | arXiv 2026 | Spatial repr. | Hierarchical scene graph + zero-shot VLN | 64.0% SR, 接近 supervised SOTA | 依赖 pre-exploration |
| [[Papers/2602-DM0\|DM0]] | arXiv 2026 | CoT + Data-centric | Spatial Scaffolding + Embodied-Native 预训练 | Gradient decoupling 保护推理能力 | 2B params, 训练成本高 |
| [[Papers/2602-GTA\|GTA]] | arXiv 2026 | Spatial repr. | TSDF + topo graph + counterfactual reasoning | SPL +16.4 | 依赖 3D reconstruction |
| [[Papers/2603-PROSPECT\|PROSPECT]] | arXiv 2026 | Spatial repr. | CUT3R + SigLIP fusion, latent prediction | 长程任务 +4.14% SR | 部分组件不开源 |

## Datasets & Benchmarks

| Dataset/Benchmark | 来源 | 规模 | 评估维度 | 特点 |
|:------------------|:-----|:-----|:---------|:-----|
| **ERQA** (Gemini Robotics) | Real | 400 questions, 7 categories | Spatial, trajectory, action, state, multi-view, task reasoning | 首个 embodied reasoning 专用 benchmark，基于真实机器人场景 |
| **EmbodiedBench** | Sim | 1,128 tasks, 4 environments | High-level planning → low-level manipulation, 6 capability subsets | 最全面的 MLLM embodied agent 评测；ALFRED + Habitat 等仿真环境 |
| **FoMER** | Real + Sim | 1,112 samples, 10 task categories, 8 embodiments | 10-dim reasoning quality (不仅看结果对错) | 首次分离 perceptual grounding 与 action reasoning；覆盖真实视频和仿真 |
| **Robot-R1 Bench** | Sim | MCQ format, RLBench 基础 | Spatial understanding, state prediction, movement prediction | 为 RL-based reasoning 设计；基于 RLBench 仿真渲染 |
| **SpatialVLM Data** | Real | 2B VQA from 10M images | Metric-space 3D spatial reasoning | Internet-scale 真实图像自动标注，可做 RL dense reward |
| **VLASER-6M** | Sim | 6M samples, 4 reasoning types | Grounding, spatial, planning, in-domain QA | 仿真环境自动生成；验证不同 reasoning data 对 VLA 的迁移效果 |
| **Thinker Dataset** | Real + Sim | 4.8M instances | Ego-view reasoning, grounding, spatial, CoT planning | 混合 ego4d 真实视频 + 仿真数据；未开源 |
| **SIMPLEREnv** | Sim | WidowX/Google Robot 仿真 | Manipulation success rate | ECoT、Embodied-R1 等论文的标准评测平台；Google RT 系列数据仿真化 |

## Key Takeaways

1. **RL > SFT for embodied reasoning**：多篇论文一致验证 GRPO-based RL 系统性优于 SFT。这不是偶然——RL 让模型从 outcome 中学习推理策略，而非仅模仿人类标注。GRPO 已成为该领域的 de facto standard RL 算法。

2. **小模型 + 好训练 > 大模型 + 弱训练**：Embodied-R (3B > o1)、Robot-R1 (7B > GPT-4o)、ECoT (7B > RT-2-X 55B)、Thinker (10B > RoboBrain2-32B) 反复证明：对于 embodied reasoning，targeted training 比 model scale 更重要。

3. **In-domain data 是关键**：VLASER 的核心发现——OOD embodied reasoning data 几乎无法迁移到 VLA performance——对数据策略有深刻启示。Embodied reasoning 的 domain gap 比 NLP 大得多。

4. **Explicit spatial representation 大幅提升 navigation reasoning**：GTA (+16.4 SPL)、SpatialNav (zero-shot 64% SR ≈ supervised SOTA)、PROSPECT (+4.14% long-horizon SR) 一致证明：给 MLLM 提供结构化空间信息远优于让它从原始像素 "猜" 空间关系。

5. **Benchmark 覆盖仍有显著缺口**：当前 benchmark 偏重 tabletop manipulation 和简单 navigation。复杂多步推理、tool use、deformable object manipulation、human-robot interaction 等场景的评测基本缺失。FoMER 揭示的 "猜对答案但推理错误" 问题更警示我们：仅看 final accuracy 是不够的。

## Open Problems

1. **Real-world transfer gap**：10 篇新论文中仅 3 篇 (ECoT, Embodied-R1, Lumo-1) 有 real robot 实验。RL-based reasoning 在仿真中的优势能否迁移到真实世界仍是 open question。

2. **Reasoning 延迟 vs 实时控制**：Embodied CoT 引入额外推理步骤，与 real-time control 需求矛盾。DM0 的 Spatial Scaffolding 和 Embodied-R 的 key-frame extraction 尝试缓解，但 fast thinking vs slow thinking 的 trade-off 尚无系统性解决方案。

3. **Long-horizon multi-step reasoning**：FoMER 仅评估 single-step reasoning，EmbodiedBench 最佳模型仅 28.9%。真正的 embodied intelligence 需要跨数十步甚至数百步的 error-robust 推理链，当前方法远未达到。

4. **Reasoning quality 评估**：FoMER 发现 Cosmos-R1 "猜对率" 高但推理质量低。如何评估 reasoning 的过程质量（而非仅看结果）是一个被严重低估的问题。

5. **跨 embodiment 泛化**：不同机器人形态（单臂/双臂/移动/人形）需要不同的空间推理能力。Embodied-R1 的 pointing 和 DM0 的 cross-embodiment 是初步尝试，但系统性的跨形态推理迁移仍未解决。

6. **Reasoning 与 world model 的结合**：当前 embodied reasoning 大多基于 reactive perception（看到什么推理什么）。将 predictive world model（预测行动后果）与 reasoning 结合，实现 mental simulation，是通向更高级具身智能的关键方向。

7. **Safety-aware reasoning**：在人类共存环境中，reasoning 不仅要 "正确"，还要 "安全"。FoMER 揭示的 "reasoning 错误但结果碰巧正确" 问题在安全攸关场景中尤为危险。

## 调研日志
- **调研日期**: 2026-03-30
- **论文统计**: vault 已有 8 篇 + 新 digest 10 篇 = 共 18 篇分析
- **未能获取**: 无（所有候选论文均成功 digest）
- **搜索策略**: 10 条 WebSearch query，覆盖核心主题、RL 方法、CoT 方法、benchmark、应用场景
- **新增 digest**: [[Papers/2407-ECoT|ECoT]], [[Papers/2508-EmbodiedR1|Embodied-R1]], [[Papers/2512-Lumo1|Lumo-1]], [[Papers/2601-Thinker|Thinker]], [[Papers/2509-FoMER|FoMER]], [[Papers/2506-RobotR1|Robot-R1]], [[Papers/2504-EmbodiedR|Embodied-R]], [[Papers/2401-SpatialVLM|SpatialVLM]], [[Papers/2510-VLASER|VLASER]], [[Papers/2502-EmbodiedBench|EmbodiedBench]]
