---
title: "VLN-VLA Unification: Foundation Models for Indoor Robot Navigation and Manipulation"
tags: [VLN, VLA, SLAM, embodied-AI, foundation-model, indoor-scene, navigation, manipulation]
status: draft
date_updated: "2026-03-24"
---

## Overview
本 Topic 从 foundation model 视角，梳理 VLN（Vision-and-Language Navigation）和 VLA（Vision-Language-Action）两个领域的交汇与趋同。核心问题：VLN 和 VLA 在架构上正在趋同（都使用 VLM backbone + action prediction），能否统一？SLAM-based 空间表示在其中扮演什么角色？

## 1. VLA 基础模型现状

Vision-Language-Action（VLA）模型是 embodied AI 领域近年来最重要的进展之一。其核心思路是将预训练 VLM 的视觉-语言理解能力迁移到 robot action 生成，形成"看懂场景 → 理解指令 → 输出动作"的端到端 pipeline。自 2023 年 [[Brohan2023-RT2|RT-2]] 开创 VLA 范式以来，该领域经历了三个关键演进：（1）**action representation 从 discrete token 到 continuous flow matching**——RT-2 和 [[Kim2024-OpenVLA|OpenVLA]] 将 action 离散化为 text token，控制频率受限于 autoregressive decoding（~3 Hz）；[[Black2024-Pi0|π₀]] 引入 flow matching + action expert 实现 50 Hz 连续控制；（2）**从单任务到 cross-embodiment generalist**——[[Ghosh2024-Octo|Octo]] 和 OpenVLA 在 Open X-Embodiment 数据集上训练，覆盖多种 robot 平台；π₀ 进一步扩展到 7 个平台 68 个任务；（3）**从短时操作到 long-horizon 自主系统**——[[Black2025-Pi05|π0.5]] 加入 hierarchical inference 实现 15 分钟级家务任务，[[Torne2026-MEM|MEM]] 引入多尺度记忆机制，[[Li2026-RoboClaw|RoboClaw]] 用 VLM agent loop 统一数据收集和执行。

### Key VLA Models

| Model | Year | VLM Backbone | Action Space | Training | Key Innovation |
|-------|------|-------------|-------------|----------|---------------|
| [[Brohan2023-RT2\|RT-2]] | 2023 | PaLM-E 12B / PaLI-X 55B | Discrete tokens (7-DoF, 256 bins) | Co-fine-tuning (web + robot) | VLM → action tokens 范式开创 |
| [[Ghosh2024-Octo\|Octo]] | 2024 | Transformer (27M/93M, 无 VLM) | Continuous (diffusion, action chunk) | 800k trajectories, OXE | 轻量开源 generalist + diffusion head |
| [[Kim2024-OpenVLA\|OpenVLA]] | 2024 | Llama 2 7B + DINOv2/SigLIP | Discrete tokens (7-DoF) | 970k demonstrations, OXE | 开源 VLA baseline，超越 RT-2-X |
| [[Black2024-Pi0\|π₀]] | 2024 | PaliGemma 3B + Action Expert 300M | Continuous (flow matching, 50 Hz) | 10K+ hrs, 7 platforms, 68 tasks | Flow matching + MoE-style action expert |
| [[Black2025-Pi05\|π0.5]] | 2025 | PaliGemma 3B (extended) | Hybrid (discrete pre-train → continuous post-train) | Co-training 5 类异构数据 | Hierarchical inference + open-world generalization |
| [[Torne2026-MEM\|MEM]] | 2026 | Gemma3-4B (π0.6 base) | Continuous (flow matching) | Robot demos + video + web | 多尺度记忆（视频短期 + 语言长期） |
| [[Li2026-RoboClaw\|RoboClaw]] | 2026 | Off-the-shelf VLM + π0.5 | Continuous (flow matching) | 自主采集 + 迭代学习 | EAP 自主数据收集 + VLM agent loop |

### 架构趋势 Takeaway

1. **Action representation 是核心分野**：discrete token（RT-2, OpenVLA）→ diffusion（Octo）→ flow matching（π₀ 系列）。连续 action 生成显著提升了控制频率和灵巧操作能力。
2. **VLM backbone 不是越大越好**：RT-2 用 55B 参数，但 π₀ 用 3B + 300M action expert 就实现了更强的操作能力。关键在于 action head 的设计和训练数据的多样性。
3. **Hierarchical 架构成为主流**：π0.5 和 MEM 都采用高层语义推理 + 低层 action 生成的分层设计，这与 VLN 领域的 high-level planning + low-level control 高度相似，暗示了 VLN-VLA 统一的可能性。
4. **开源生态推动快速迭代**：Octo 和 OpenVLA 的开源使社区能够快速复现和改进，Open X-Embodiment 数据集成为事实标准。

## 2. VLN 基础模型现状

Vision-and-Language Navigation（VLN）要求 agent 根据自然语言指令在未见环境中导航到目标位置。该领域经历了从 task-specific 架构到 foundation model 驱动的演进，与 VLA 领域的发展轨迹形成有趣的平行关系。

### 演进脉络：task-specific → topological planning → VLM/LLM-based

**Phase 1: Task-specific 架构（2019-2022）**。早期 VLN 模型使用 LSTM/Transformer encoder-decoder 架构，在 discrete nav-graph 上进行 action prediction。代表工作 [[Chen2022-DUET|VLN-DUET]] 提出 dual-scale graph transformer，通过在线构建 topological map 并结合 local fine-grained encoding 和 global coarse-grained reasoning，在 REVERIE、SOON、R2R 等 benchmarks 上取得 SOTA，奠定了 topological map 作为 VLN 核心 spatial representation 的范式。

**Phase 2: Continuous environments 与 hierarchical planning（2022-2024）**。VLN-CE（Vision-Language Navigation in Continuous Environments）将 VLN 从 discrete nav-graph 扩展到更接近真实场景的 continuous action space。[[An2024-ETPNav|ETPNav]] 提出 online topological mapping + hierarchical planning（transformer-based high-level planner + obstacle-avoiding low-level controller），在 R2R-CE 和 RxR-CE 上大幅超越 prior SOTA。这一阶段的关键架构创新——**hierarchical decomposition（high-level planning + low-level control）**——与 VLA 领域中 π0.5 的 hierarchical inference 高度平行。

**Phase 3: LLM/VLM backbone 引入（2023-present）**。[[Zhou2023-NavGPT|NavGPT]] 首次将 GPT-4 作为 zero-shot navigation reasoning engine，通过文本化视觉观测让 LLM 进行显式推理（sub-goal decomposition、landmark identification、progress tracking）。尽管 zero-shot 性能低于 trained models，但揭示了 LLM 在 navigation planning 中的潜力。其 follow-up NavGPT-2（ECCV 2024）通过 visual alignment 消除了与 VLN specialist 的性能差距，验证了 VLM backbone 在 VLN 中的可行性。[[Cheng2024-NaVILA|NaVILA]] 进一步将 VLM（VILA）微调为 navigation VLA，用语言化 mid-level action 作为高层规划和低层控制的桥梁，在 R2R-CE 上达到 54% SR 并实现了 legged robot 真实世界部署。NaVILA 本质上就是一个 navigation-focused VLA，是 VLN-VLA 架构趋同的最直接证据。

### Key VLN Models

| Model | Year | Backbone | Action Space | Environment | Key Innovation |
|-------|------|----------|-------------|-------------|---------------|
| [[Chen2022-DUET\|VLN-DUET]] | 2022 | Task-specific Transformer | Discrete（nav-graph nodes, 含远程跳转） | Discrete nav-graph (MP3D) | Dual-scale graph transformer + online topological map |
| [[An2024-ETPNav\|ETPNav]] | 2024 | Task-specific Transformer | Hybrid（high-level waypoint + low-level continuous） | Continuous (Habitat) | Online topological planning + obstacle-avoiding controller |
| [[Zhou2023-NavGPT\|NavGPT]] | 2023 | GPT-4 (frozen, zero-shot) | Discrete（nav-graph node selection） | Discrete nav-graph (MP3D) | LLM 作为 navigation reasoning engine，显式推理链 |
| NavGPT-2 | 2024 | Frozen LLM + visual alignment | Discrete | Discrete nav-graph | 消除 LLM agent 与 VLN specialist 的性能差距 |
| [[Cheng2024-NaVILA\|NaVILA]] | 2024 | VILA VLM (fine-tuned) | Mid-level language actions → RL locomotion | Continuous (Habitat + Isaac Sim + Real) | VLM → 语言化动作 → locomotion policy，真实 legged robot 部署 |

### VLN vs VLA：关键差异

| 维度 | VLN | VLA |
|------|-----|-----|
| **Action space** | Discrete waypoints / nav-graph nodes | Continuous joint torques / end-effector poses |
| **Primary environment** | Simulation（Habitat, MP3D, Gibson）| Real world + simulation |
| **Control frequency** | Low（~1-5 Hz, per-step decision） | High（10-50 Hz continuous control）|
| **Evaluation benchmarks** | R2R, REVERIE, SOON, R2R-CE, RxR-CE | 各种 real-world manipulation tasks |
| **Spatial representation** | Topological map / nav-graph | 通常无显式空间表示（end-to-end） |
| **Foundation model 使用方式** | VLM/LLM → high-level planning | VLM → end-to-end action generation |
| **核心挑战** | Sim-to-real gap, instruction grounding | Dexterous control, generalization |

### Sim-to-Real Gap 现状

VLN 领域面临显著的 sim-to-real gap：绝大多数工作在 Habitat/MP3D simulator 中评估，真实世界部署案例极少。NaVILA 是为数不多实现真实部署的工作（Unitree Go2 上 88% 成功率），其成功依赖两个关键设计：（1）用 YouTube 视频作为 real-world visual data source；（2）用语言化 mid-level action 解耦感知与控制，使 sim-to-real transfer 只需要在 low-level locomotion policy 层面进行。这一策略与 VLA 领域 [[Li2026-RoboClaw|RoboClaw]] 的自主数据收集 + VLM agent loop 形成有趣对比——两者都在寻找 scalable 的 real-world data 获取方案。

## 3. 语义 SLAM 与空间表示
<!-- Semantic SLAM and spatial representations for VLMs -->

## 4. 架构趋同分析
<!-- Cross-cutting comparison of VLN and VLA architectures -->

## 5. 现有 Nav+Manip 系统
<!-- Systems combining navigation and manipulation -->

## 6. Gap 分析与潜在方向
<!-- Research gaps, benchmarks, and future directions -->
