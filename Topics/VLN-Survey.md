---
title: Vision-Language Navigation Survey
tags:
  - VLN
  - navigation
  - spatial-memory
date_updated: 2026-04-01
year_range: 2022-2026
papers_analyzed: 18
---
## Overview

Vision-and-Language Navigation (VLN) 是 embodied AI 的核心任务之一，要求 agent 根据自然语言指令在视觉环境中导航到目标位置。自 2018 年 R2R benchmark 提出以来，VLN 经历了从 discrete navigation graph 到 continuous environment、从 task-specific 架构到 foundation model backbone 的深刻演变。

近四年（2022-2026）是 VLN 研究的范式转折期与爆发期。核心趋势可以概括为五个方向的同步演进：

1. **From discrete to continuous**：从 nav-graph 上的 waypoint selection 走向连续环境中的真实导航。DUET → ETPNav → NaVid/NaVILA → StreamVLN/Efficient-VLN/ETP-R1 的演进清晰展示了这一轨迹。到 2025-2026 年，VLN-CE 已成为绝对主战场，**R2R-CE 的最高 SR 已从 ETPNav 时期的约 50% 攀升至 ETP-R1 的 65%**。

2. **From task-specific to foundation model**：从 task-specific transformer（DUET、ETPNav）到 VLM backbone（NaVid、StreamVLN、Efficient-VLN）。Foundation model 的引入带来了 zero-shot generalization、sim-to-real transfer 和 cross-dataset 迁移能力。

3. **Streaming VLA 成为主流框架**：StreamVLN、PROSPECT 和 Efficient-VLN 共同确立了 streaming Video-VLM 处理连续视觉流的范式——通过 KV cache 复用、spatial token pruning 或 progressive memory 压缩实现恒定延迟和有界内存。

4. **RL fine-tuning 作为新训练范式**：VLN-R1 和 ETP-R1 首次将 GRPO（DeepSeek-R1 路线）引入 VLN，在 SFT/DAgger 之上进一步通过 reinforcement signal 提升性能（ETP-R1: +2% SR），开辟了 VLN 训练的新方向。

5. **Explicit spatial representation 复兴**：SpatialNav、GTA、MapNav 和 VL-Nav 代表了一波 "给 VLM 提供结构化空间知识" 的新浪潮——通过 scene graph、semantic map 或 volumetric representation 弥补 VLM 空间推理的固有缺陷，在 zero-shot 设定下取得了令人瞩目的成果（SpatialNav R2R-CE 64.0% SR）。

此外，VLN 领域正在快速拓展新维度：VLN-PE 首次系统研究了不同 robot embodiment 的 physical gap（~34% 相对 SR 下降），AirNav 将 VLN 拓展到真实城市 UAV 场景，LH-VLN 将时间尺度从数十步拓展到 150+ 步。

## 技术路线

### 路线 1：Topological Map + Task-Specific Planning

代表论文：[[2202-DUET]]、[[2304-ETPNav]]、[[2512-ETPR1]]

VLN 的"经典范式"持续演进。核心思想是在导航过程中在线构建 topological map 作为结构化空间记忆，再用 cross-modal transformer 进行路径规划。

- **DUET**（CVPR 2022）提出 dual-scale graph transformer，在 topological map 上进行 global coarse-grained reasoning + local fine-grained grounding，在 R2R、REVERIE、SOON 上取得 SOTA。但仅限于 discrete nav-graph。
- **ETPNav**（TPAMI 2024）将 topological planning 扩展到 continuous environment，通过 waypoint prediction + hierarchical architecture，在 R2R-CE 和 RxR-CE 上大幅超越 prior SOTA。
- **ETP-R1**（arXiv 2025）在 ETPNav 基础上进行三重升级：（1）用 Gemini API 标注 1M 多样化指令替代传统 speaker model，（2）跨数据集（R2R+RxR）联合预训练，（3）首次引入 closed-loop online GRPO 进行 RL fine-tuning。R2R-CE test SR 64% / SPL 54%，**目前 VLN-CE 整体 SOTA**。

**优势**：topological map 提供结构化、可解释的空间表示；hierarchical design 自然解耦语义理解和运动控制；ETP-R1 证明经典框架仍可通过数据 scaling + RL 持续突破。
**局限**：依赖 waypoint prediction 的 low-level controller 是 heuristic；RL fine-tuning 提升有限（+2% SR）且对 hyperparameters 敏感。

### 路线 2：Streaming VLA（Video-VLM End-to-End）

代表论文：[[2305-NavGPT]]、[[2402-NaVid]]、[[2507-StreamVLN]]、[[2512-EfficientVLN]]、[[2603-PROSPECT]]

Foundation model 时代的核心范式。将预训练 Video-VLM 作为 navigation backbone，以 streaming 方式处理连续视觉流并直接输出 actions。

- **NavGPT**（AAAI 2024）首次将 GPT-4 用于 VLN 的 zero-shot reasoning，暴露了纯文本输入的瓶颈（R2R SR ~30%）。
- **NaVid**（RSS 2024）提出 video-based VLM 方案（LLaMA-VID），仅 4 tokens/frame 压缩历史，实现 map-free 导航。R2R-CE SPL 35.9%。
- **StreamVLN**（ICRA 2026）确立了 streaming VLN 范式：fast-streaming dialogue context（sliding-window KV cache，消除 99% prefilling 开销）+ slow-updating memory（voxel-based 3D spatial pruning，减少 20% tokens 且性能提升）。基于 LLaVA-Video 构建，R2R-CE SR 56.9%，部署于 Unitree Go2 实现真实机器人导航。
- **Efficient-VLN**（NAACL 2025）针对训练效率，提出 progressive memory（遗忘曲线式压缩）+ recursive memory（KV cache 传递）+ dynamic DAgger，仅 282 GPU hours 达到 R2R-CE 64.2% SR，**训练效率提升 78%**。
- **PROSPECT**（arXiv 2026）将 3D spatial intelligence 引入 streaming 框架：CUT3R（3D 空间编码器）+ SigLIP（2D 语义编码器）cross-attention fusion，加上 latent predictive representation（在 frozen teacher space 预测下一步特征，无推理开销）。R2R-CE SR 60.3%，long-horizon（≥100 步）任务 SR 提升 +4.14%。

**优势**：streaming 框架天然适配 real-time navigation；KV cache 复用和空间 pruning 实现恒定延迟；end-to-end 训练简化系统设计；多源数据 co-training 保持通用 VL 能力。
**局限**：streaming VLA 在导航精度上仍落后 graph-based 方法（PROSPECT 60.3% vs ETP-R1 65%）；推理延迟 0.25-1.5s/step 对快速移动场景可能不够；long-horizon 能力虽有提升但距离 LH-VLN 级别仍远。

### 路线 3：Navigation as VLA（语言化 Mid-Level Action）

代表论文：[[2412-NaVILA]]

将 VLN 重构为 navigation-focused VLA，是当前最前沿的架构统一方向。

- **NaVILA**（RSS 2025）将 VLM（VILA）微调为 navigation VLA，通过生成 mid-level 语言化动作指令（如 "move forward 75cm"）而非 low-level joint action，再由 RL locomotion policy 执行。R2R-CE 54% SR，真实 legged robot 88% SR。利用 YouTube 视频生成 20k 训练轨迹。

**优势**：语言化 mid-level action 优雅地解耦了语义理解与运动控制；robot-agnostic；sim-to-real 最佳。
**局限**：mid-level action 粒度需要 task-specific 调优；能否扩展到 manipulation 尚未验证。

### 路线 4：Explicit Spatial Representation + VLM Reasoning

代表论文：[[2502-VLNav]]、[[2601-SpatialNav]]、[[2502-MapNav]]、[[2602-GTA]]、[[2507-MTU3D]]

VLM 在空间推理方面存在固有缺陷——难以从 egocentric view 推断 global spatial relationships。这一路线通过构建 explicit spatial representation 并将其提供给 VLM 进行推理，是 2025-2026 年增长最快的方向。

- **VL-Nav**（arXiv 2025）提出 neuro-symbolic 架构：symbolic 3D scene graph + image memory 增强 VLM reasoning。NeSy Task Planner 做 coarse-to-fine 目标验证，NeSy Exploration System 融合 frontier-based 和 instance-based 目标。Indoor 83.4% SR，real-world 86.3% SR（含 483m 长距离导航）。
- **SpatialNav**（arXiv 2026）构建 Spatial Scene Graph (SSG) 用于 zero-shot VLN，设计 compass-style visual encoding（减少 62% tokens）和 remote object localization。**R2R-CE zero-shot 64.0% SR**——逼近 supervised SOTA。
- **MapNav**（arXiv 2025）用 Annotated Semantic Map (ASM) 替代 frame history，text annotation 将抽象语义转为 VLM 可理解的导航线索（+7.4% SR），memory 恒定 0.17MB（vs NaVid 在 300 步时 276MB）。
- **GTA**（arXiv 2026）将 spatial estimation 与 semantic planning 解耦：TSDF volumetric map + topological graph 提供 metric world representation，MLLM 通过 BEV visual prompting + counterfactual reasoning 做决策。Zero-shot R2R-CE 48.8% SR，跨 embodiment 部署（wheeled 40% + drone 42% SR）。
- **MTU3D**（ICCV 2025）提出 online query + dynamic spatial memory bank 统一 object grounding 与 frontier exploration，无需显式 3D 重建。

**优势**：为 VLM 补充了其最缺乏的空间推理能力；zero-shot 设定下表现惊人（SpatialNav 64.0%）；many methods 天然支持 cross-embodiment；spatial representation 可复用于 manipulation。
**局限**：scene graph / semantic map 的构建依赖额外模块（depth、SLAM、segmentation），增加系统复杂度；pre-exploration 假设（SpatialNav）在真实场景中受限；room segmentation 在开放空间需人工辅助。

### 路线 5：Long-Horizon Extension

代表论文：[[2412-LHVLN]]

将 VLN 从单阶段短程导航扩展到多阶段长程任务。

- **LH-VLN**（CVPR 2025）提出 LHPR-VLN benchmark（3,260 个多阶段任务，平均 150 步），所有现有方法 SR = 0%，揭示了 long-horizon 导航的根本性挑战。MGDM 模型通过 entropy-based memory forgetting + long-term retrieval 进行初步探索。

### 路线 6：Sim-to-Real Gap & 新领域拓展

代表论文：[[2507-VLNPE]]、[[2601-AirNav]]

- **VLN-PE**（ICCV 2025）首个支持 humanoid、quadruped、wheeled 三类机器人的 physically realistic VLN 平台。核心发现：sim→physical 迁移约 **34% 相对 SR 下降**；cross-embodiment co-training 一致最优；RGB-D 在光照变化下远比 RGB-only 鲁棒；**小模型 in-domain 训练优于大模型 zero-shot**。
- **AirNav**（arXiv 2026）首个基于真实城市航拍数据的大规模 UAV VLN benchmark（143K 样本），自然度得分 3.75 超过所有已有 UAV VLN 数据集。AirVLN-R1（Qwen2.5-VL-7B + SFT + RFT/GRPO）在 test-unseen 达到 51.75% SR，7B 模型击败 235B baseline。

### 交叉趋势：RL Fine-Tuning for VLN

代表论文：[[2506-VLNR1]]、[[2512-ETPR1]]、[[2601-AirNav]]

RL fine-tuning（特别是 GRPO）作为 SFT/DAgger 之后的第三阶段训练，正在成为 VLN 的通用训练范式。VLN-R1 首次将其引入 streaming VLN（R2R-CE 30.2% SR），ETP-R1 引入 graph-based VLN（+2% SR），AirNav 的 AirVLN-R1 引入 UAV VLN。Time-Decayed Reward（VLN-R1）和 task-specific reward shaping（ETP-R1）是关键设计。目前 RL 提升幅度有限但方向明确。

## Datasets & Benchmarks

| Dataset/Benchmark | 规模 | 评估指标 | SOTA | 特点 |
|:------------------|:-----|:---------|:-----|:-----|
| R2R (Room-to-Room) | 7,189 paths, 21,567 instructions | SR / SPL / nDTW | — | VLN 经典 benchmark，discrete nav-graph |
| R2R-CE | R2R → continuous env | SR / SPL | 65% SR (ETP-R1); 64.0% zero-shot (SpatialNav) | VLN-CE 主战场 |
| RxR / RxR-CE | 16,522 paths, 126K instructions | SR / SPL / nDTW | — | 多语言（en/hi/te），更详细指令 |
| REVERIE | 21,702 instructions | SR / SPL / RGS | — | Remote grounding：导航+找物体 |
| SOON | 4,000 instructions | SR / SPL | — | Scenario-oriented navigation |
| LHPR-VLN (LH-VLN) | 3,260 多阶段任务，avg 150 步 | SR | 0% (所有方法) | Long-horizon benchmark，揭示长程根本不足 |
| AirNav | 143K 真实 UAV 样本 | SR / SPL | 51.75% (AirVLN-R1) | 首个真实城市航拍 UAV VLN |
| VLN-PE | 3 embodiment types | SR (physical) | ~34% relative drop | 首个 physically realistic VLN 平台 |
| EgoPlan-Bench2 | 多领域 egocentric | Accuracy | — | Embodied planning 评测 |

## Key Takeaways

1. **R2R-CE 从"困难挑战"走向"基本可解"**：2022 年 R2R-CE SR ~50%（ETPNav），到 2026 年 ETP-R1 达到 65% SR、SpatialNav zero-shot 达到 64%。Graph-based 和 streaming VLA 两条主线都在快速收敛，VLN-CE 性能正在接近 discrete nav-graph 水平。

2. **两大技术范式竞争格局清晰**：graph-based topological planning（ETP-R1 65% SR）与 streaming VLA（Efficient-VLN 64.2% SR）代表了 VLN-CE 的两大范式。前者保留结构化空间推理，后者追求 end-to-end 简洁性。两者性能接近，但各自优缺点互补。

3. **Explicit spatial representation 强势回归**：SpatialNav 的 zero-shot 64.0% SR 是一个标志性结果——表明给 VLM 提供结构化空间信息可以在不做任何 task-specific 训练的情况下逼近 supervised SOTA。这一路线可能是 VLN 领域最值得关注的新趋势。（**建议加入 DomainMaps**）

4. **RL fine-tuning 成为通用第三阶段**：VLN-R1、ETP-R1、AirVLN-R1 不约而同地采用 GRPO 作为 SFT/DAgger 之后的 RL fine-tuning。虽然目前提升幅度有限（+2-5%），但训练范式的确立意义大于绝对数字。Time-Decayed Reward 和 task-specific reward shaping 是关键 design choice。

5. **Sim-to-real gap 被系统量化但远未解决**：VLN-PE 首次量化了 ~34% 的相对 SR 下降。Cross-embodiment co-training 和 depth fusion 是当前最有效的缓解策略，但 in-domain 小模型仍优于 zero-shot 大模型。VLN-VLA 统一的真正验证场需要从 simulation 转向 physical world。

6. **VLN 正在拓展到新维度**：AirNav 将 VLN 拓展到 UAV 场景（143K 真实数据），LH-VLN 拓展到 150+ 步多阶段任务，VLN-PE 拓展到多形态机器人。VLN 不再是"室内短程导航"的同义词。

## Open Problems

### 1. Long-Horizon Navigation（最根本挑战）
LH-VLN benchmark 上所有方法 SR=0%，暴露了当前 VLN 模型在多阶段、150+ 步任务上的根本不足。PROSPECT 的 latent predictive representation 在 ≥100 步上有 +4.14% 提升，Efficient-VLN 的 progressive memory 和 MEM 的 multi-scale memory 提供了初步方案。**核心瓶颈是 memory management：如何在有限 context window 下保留跨子任务的关键信息？** Streaming VLA 的 slow memory（StreamVLN）和 language long-term memory（MEM）是两个值得深入的方向。

### 2. Sim-to-Real Transfer
VLN-PE 量化了 ~34% 的 embodied gap，涵盖 viewpoint shift、motion error、光照变化、碰撞摔倒等因素。NaVILA 在 legged robot 88% SR、VL-Nav real 86.3% SR 是少数成功案例，但大多数方法仍困在 simulation。**关键洞察：cross-embodiment co-training 一致最优，RGB-D 远比 RGB-only 鲁棒，in-domain 小模型优于 zero-shot 大模型。** PROSPECT 和 StreamVLN 的 real robot 实验是正确方向。

### 3. Spatial Representation 的最优形式
VLN 领域已出现丰富的空间表示方案：topological map（DUET、ETPNav）、video tokens（NaVid）、voxel-based 3D pruning（StreamVLN）、scene graph（SpatialNav、VL-Nav）、annotated semantic map（MapNav）、TSDF volumetric（GTA）、online query（MTU3D）、latent prediction（PROSPECT）、language memory（NaVILA）。**哪种表示最适合同时服务 navigation 和 manipulation？** SpatialNav 和 GTA 证明了 explicit spatial representation 对 VLM reasoning 的巨大价值，但如何让这些表示 learnable 而非 hand-crafted 仍是开放问题。

### 4. VLN-VLA 架构统一
NaVILA 证明了 navigation 可以重构为 VLA。语言化 mid-level action 桥接了高层指令理解和低层运动控制。但 navigation 中构建的空间表示能否直接服务 manipulation？DM0 和 MEM 等 unified VLA 模型正在探索 navigation + manipulation 的联合框架，但真正的端到端验证尚未实现。详见 [[Topics/VLN-VLA-Unification]]。

### 5. Zero-Shot 是否会取代 Supervised？
SpatialNav 的 zero-shot R2R-CE 64.0% SR 已逼近 supervised SOTA（Efficient-VLN 64.2%）。GTA 也在 zero-shot 下取得 48.8% SR。如果 VLM 持续进化（GTA 的 scaling 实验显示 37.2% → 47.2% 随 MLLM 能力提升），**zero-shot VLN 是否会在 1-2 年内超越 supervised？** 这将根本改变 VLN 的研究范式——从"如何训练好导航模型"转向"如何提供好的空间表示"。

### 6. 训练效率与 Scaling
Efficient-VLN 将训练开销降至 282 GPU hours（vs StreamVLN 1500h），但 VLN 模型的 scaling law 尚不清楚。ETP-R1 的 Gemini 数据标注和 YouTube 数据（NaVILA）表明 web-scale 数据有价值。**小而精的 task-specific 模型（ETP-R1 graph-based）vs 大而通用的 VLM-based 模型——谁更有未来？** VLN-PE 的发现"in-domain 小模型优于 zero-shot 大模型"值得深思。

## 调研日志

### 第三轮更新（2026-04-01）
- **调研日期**: 2026-04-01
- **论文统计**: 无新增论文，模板对齐更新
- **新增 section**: Datasets & Benchmarks

### 第二轮更新（2026-03-30）
- **调研日期**: 2026-03-30（增量更新，基于 2026-03-27 版本）
- **论文统计**: vault 已有 8 篇 + 新 digest 10 篇 + 跳过 0 篇 + 失败 0 篇
- **未能获取**: 无
- **搜索统计**: 8 次 WebSearch，候选 15+ 篇，筛选 10 篇
