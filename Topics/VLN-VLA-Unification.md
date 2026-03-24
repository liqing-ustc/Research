---
title: "VLN-VLA Unification: Foundation Models for Indoor Robot Navigation and Manipulation"
tags: [VLN, VLA, SLAM, embodied-AI, foundation-model, indoor-scene, navigation, manipulation]
status: complete
date_updated: "2026-03-24"
---

## Overview

本 survey 从 foundation model 视角，系统梳理了 VLN（Vision-and-Language Navigation）和 VLA（Vision-Language-Action）两个领域的技术演进与架构趋同。通过分析 17 篇核心论文，我们发现：**VLN 和 VLA 在四个维度上正在趋同**——VLM backbone、language-conditioned action prediction、web-scale pre-training、hierarchical 架构——NaVILA 已经证明 VLN 可以重构为 navigation-focused VLA。然而，统一面临三大核心障碍：（1）action space mismatch（50 Hz continuous joint control vs. 1-5 Hz discrete waypoint selection）；（2）navigation 和 manipulation 缺乏 shared spatial representation；（3）simulation 生态无法同时满足 building-scale navigation 和 high-fidelity manipulation。Semantic SLAM（尤其 ConceptGraphs 式 3D scene graph）是弥合这一 gap 的关键基础设施，可以同时服务 navigation waypoints 和 manipulation targets。现有 Nav+Manip 系统（OK-Robot、SayCan、Mobile ALOHA）验证了联合任务的可行性，但都未实现 shared spatial representation。综合 gap 分析和 benchmark 评估（ALFRED、TEACh、HomeRobot OVMM），我们提出最具潜力的研究方向是 **Hierarchical VLA with Shared Semantic Scene Graph**——共享 VLM backbone + 分离 domain-specific action heads + ConceptGraphs 式 scene graph 作为统一空间记忆，在 HomeRobot OVMM 上验证。

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

### 为什么空间表示是 VLN-VLA 统一的关键？

Section 1 和 Section 2 揭示了一个核心矛盾：VLN 系统依赖显式空间表示（topological map、nav-graph）进行 high-level planning，而 VLA 系统通常采用 end-to-end 架构、缺乏显式空间表示。要统一两者，需要一种**既能支持 navigation planning 又能支持 manipulation grounding 的空间表示**——这正是 semantic SLAM 和 language-grounded spatial representations 的研究目标。

近年来，foundation models（CLIP、SAM、GPT-4）的突破催生了一类新的空间表示方法：它们在传统 SLAM 的 geometric map 基础上融合了 open-vocabulary 语义信息，使地图可以直接通过自然语言查询。这些方法按表示形式可分为三类：（1）**dense feature maps**（per-pixel/per-voxel 存储 VLM features）；（2）**3D scene graphs**（object-level nodes + semantic relations）；（3）**neural/Gaussian fields**（implicit 或 explicit 连续表示 + 可附加语义）。

### 方法总览

| Method | Year | Venue | 表示形式 | 语义 Grounding | 支持的下游任务 |
|--------|------|-------|----------|---------------|---------------|
| [[Huang2023-VLMaps\|VLMaps]] | 2023 | ICRA | Top-down 2D grid map（per-cell LSeg/CLIP features） | LSeg dense features + CLIP text encoder cosine similarity | Open-vocabulary navigation, spatial goal localization, multi-embodiment sharing |
| [[Gu2024-ConceptGraphs\|ConceptGraphs]] | 2024 | ICRA | 3D scene graph（nodes = objects, edges = semantic relations） | SAM segmentation + CLIP embeddings + GPT-4 captioning/reasoning | Text query, re-localization, navigation planning, manipulation planning |
| [[Keetha2024-SplaTAM\|SplaTAM]] | 2024 | CVPR | 3D Gaussian field（explicit volumetric） | 无内置语义（纯 geometric），但可扩展附加 CLIP features | Dense mapping, camera tracking, novel-view synthesis |
| CLIP-Fields | 2023 | RSS | Neural field（MLP mapping 3D coords → semantic embeddings） | CLIP + Detic + Sentence-BERT 弱监督 | Semantic navigation, object search |
| OpenScene | 2023 | CVPR | Per-point features on 3D point cloud | CLIP/OpenSeg features 蒸馏到 3D points | Open-vocabulary 3D scene understanding, 3D object retrieval |

### 三类表示的对比分析

**Dense feature maps（VLMaps 为代表）**
- 优势：spatial coverage 好，适合 navigation（可以直接在 map 上做 path planning）；构建简单；支持 open-vocabulary landmark query
- 局限：2D top-down 表示丢失高度信息；per-cell feature 缺乏 object-level abstraction；难以直接支持 manipulation（缺少 3D object geometry）
- **VLN 适配性：高**——可直接替代 topological map 用于 navigation planning
- **VLA 适配性：低**——缺少 manipulation 所需的 3D object 信息

**3D scene graphs（ConceptGraphs 为代表）**
- 优势：object-level abstraction 天然适合 LLM-based planning；graph 结构支持 relational reasoning；同时包含 geometric（point cloud）和 semantic（CLIP + caption）信息
- 局限：依赖高质量 instance segmentation；graph 构建计算开销大；对 cluttered scene 的 over-/under-segmentation 敏感
- **VLN 适配性：高**——graph nodes 可作为 navigation waypoints，类似 [[Chen2022-DUET|VLN-DUET]] 的 topological map 但语义更丰富
- **VLA 适配性：高**——object nodes 提供 manipulation targets + 空间关系，可直接传给 VLA 的 high-level planner

**Neural/Gaussian fields（SplaTAM、CLIP-Fields 为代表）**
- 优势：dense continuous representation，reconstruction 质量最高；SplaTAM 等 3DGS-based 方法高效且支持 incremental update；可通过附加 feature channels 扩展语义
- 局限：纯 geometric field 需额外步骤注入语义；implicit fields（NeRF-based）难以实时更新；缺少 object-level 结构
- **VLN 适配性：中**——需要从 field 中提取 navigable space，不如 grid map 直接
- **VLA 适配性：中高**——dense geometry 有利于 grasp planning，但缺少 semantic object segmentation

### 哪种表示能同时服务 VLN 和 VLA？

从上述分析可以看出，**3D scene graph（ConceptGraphs）是最有潜力同时服务 VLN 和 VLA 的表示形式**：
- 对 VLN：graph nodes 作为 navigation waypoints，graph edges 编码空间关系辅助 path planning，CLIP embeddings 支持 language-guided goal localization
- 对 VLA：object nodes 提供 manipulation targets 和 3D geometry，semantic relations 支持 task planning（如"把 A 放到 B 旁边"）
- 对 LLM/VLM planning：graph 可以文本化（序列化为节点和边的描述）后直接输入 LLM 进行推理

然而，scene graph 的局限在于缺少 dense spatial coverage——navigation 需要知道 free space 和 obstacles 的连续分布，而 scene graph 只包含 object-level 信息。因此，最理想的方案可能是**层次化组合**：

> **Dense geometric map（SplaTAM）** + **Semantic scene graph（ConceptGraphs）** + **Language interface（VLMaps-style query）**

这种层次化架构与 VLN-VLA 系统的需求天然契合：dense map 服务低层 obstacle avoidance 和 locomotion，scene graph 服务高层 task planning 和 manipulation grounding，language interface 统一两者的指令接口。

### SLAM 作为 "Spatial Memory"：长时任务的关键

Section 1 中讨论的 long-horizon VLA（[[Black2025-Pi05|π0.5]]、[[Torne2026-MEM|MEM]]）面临一个共同挑战：如何在长时间任务执行中维护 consistent 的空间理解。MEM 用 video memory 和 language memory 部分解决了这个问题，但缺乏 explicit spatial memory。

Semantic SLAM 天然提供了这种 "spatial memory"：
1. **Persistent map**：SLAM 维护一个随时间增量更新的环境地图，机器人可以随时回溯到之前探索过的区域
2. **Re-localization**：ConceptGraphs 展示了基于 scene graph 的 landmark-based re-localization，使机器人在长时间任务中不会"迷路"
3. **Incremental update**：SplaTAM 的 3DGS 表示支持高效增量更新，可以随着探索实时扩展地图
4. **Multi-session memory**：map 可以跨 session 保存和加载，使机器人在不同时间段积累环境知识

这种 spatial memory 对统一 VLN 和 VLA 至关重要：一个真正的 embodied agent 需要在导航到目标位置（VLN）后执行操作（VLA），再导航到下一个位置——整个过程需要一个 persistent、incrementally updated 的空间表示来维护 context。

### Section 3 Takeaway

1. **语义空间表示是 VLN-VLA 统一的 "missing piece"**：VLN 需要 spatial map 进行 path planning，VLA 需要 object-level semantics 进行 manipulation grounding，语义 SLAM 可以同时满足两者。
2. **3D scene graph 最适合作为统一 interface**：ConceptGraphs 式的 scene graph 同时支持 navigation waypoints 和 manipulation targets，且天然适配 LLM-based planning。
3. **层次化组合是实用方案**：dense geometry（SplaTAM）+ semantic graph（ConceptGraphs）+ language query（VLMaps）的层次化架构可以满足 VLN-VLA 系统的多层次需求。
4. **SLAM 提供 long-horizon 任务所需的 spatial memory**：persistent, incrementally updated map 是统一 navigation 和 manipulation 的基础设施。

## 4. 架构趋同分析

本节综合 Section 1-3 和 Section 5 的分析，从架构维度系统比较 VLA 和 VLN 模型，识别趋同点和分歧点，并探讨统一的可能路径。

### 4.1 VLA-VLN 模型全景对比

下表涵盖 Section 1（VLA）和 Section 2（VLN）中的所有主要模型，按统一维度进行比较：

| Model | 领域 | VLM Backbone | Action Space | 空间/记忆表示 | Training Data | 架构范式 | Task Horizon | Sim/Real |
|-------|------|-------------|-------------|-------------|--------------|---------|-------------|----------|
| [[Brohan2023-RT2\|RT-2]] | VLA | PaLM-E 12B / PaLI-X 55B | Discrete tokens（7-DoF, 256 bins, ~3 Hz） | 无显式空间表示 | RT-1 ~130k episodes + web VL data co-fine-tuning | End-to-end VLM→action tokens | 短（单步操作） | Real |
| [[Ghosh2024-Octo\|Octo]] | VLA | 无 VLM（Transformer 27M/93M） | Continuous（diffusion, action chunk 4步） | 无显式空间表示 | OXE 800k trajectories | End-to-end diffusion policy | 短 | Real |
| [[Kim2024-OpenVLA\|OpenVLA]] | VLA | Llama 2 7B + DINOv2/SigLIP | Discrete tokens（7-DoF） | 无显式空间表示 | OXE 970k demonstrations | End-to-end VLM→action tokens | 短 | Real |
| [[Black2024-Pi0\|π₀]] | VLA | PaliGemma 3B + Action Expert 300M | Continuous（flow matching, 50 Hz, chunk H=50） | 无显式空间表示 | 10K+ hrs, 7 platforms, 68 tasks | VLM + MoE-style action expert | 中（多阶段操作） | Real |
| [[Black2025-Pi05\|π0.5]] | VLA | PaliGemma 3B（extended） | Hybrid（discrete pre-train → continuous post-train, 50 Hz） | 无显式空间表示（implicit in VLM） | 5 类异构数据 co-training（MM/ME/CE/HL/WD） | Hierarchical（语义子任务 → flow matching action） | 长（10-15 min 家务） | Real |
| [[Torne2026-MEM\|MEM]] | VLA | Gemma3-4B（π0.6 base） | Continuous（flow matching） | 视频短期记忆 + 语言长期记忆（无 spatial map） | Robot demos + video + web | Hierarchical + 多尺度记忆 | 长（15 min） | Real |
| [[Li2026-RoboClaw\|RoboClaw]] | VLA | Off-the-shelf VLM + π0.5 | Continuous（flow matching） | VLM agent structured memory（非空间） | 自主采集（EAP）+ 迭代学习 | VLM agent loop + VLA primitives | 长（multi-step） | Real |
| [[Chen2022-DUET\|VLN-DUET]] | VLN | 无 VLM（task-specific Transformer） | Discrete（nav-graph node selection, 含远程跳转） | Online topological map | R2R/REVERIE/SOON supervised | Dual-scale graph transformer | 中（导航序列） | Sim（MP3D） |
| [[An2024-ETPNav\|ETPNav]] | VLN | 无 VLM（task-specific Transformer） | Hybrid（high-level waypoint + low-level continuous） | Online topological map + waypoint prediction | R2R-CE/RxR-CE supervised | Hierarchical（transformer planner + heuristic controller） | 中 | Sim（Habitat） |
| [[Zhou2023-NavGPT\|NavGPT]] | VLN | GPT-4（frozen, zero-shot） | Discrete（nav-graph node selection） | 无（文本化 history） | Zero-shot（无训练） | LLM reasoning engine | 中 | Sim（MP3D） |
| [[Cheng2024-NaVILA\|NaVILA]] | VLN/VLA | VILA VLM（fine-tuned） | Mid-level 语言化动作 → RL locomotion policy | 无显式 map（VLM implicit） | YouTube 视频 + Habitat sim + auxiliary VQA | Hierarchical（VLM → 语言动作 → RL policy） | 中 | Sim→Real |

### 4.2 共性分析（Commonalities）

尽管 VLA 和 VLN 起源于不同的研究社区（robotics manipulation vs. embodied navigation），两者在架构上呈现出显著的趋同趋势：

**1. VLM Backbone 正在成为通用基座**

从 RT-2（2023）首次将 VLM 用于 robot action 生成，到 NaVILA（2024）将 VLM 用于 navigation action 生成，**VLM 作为 embodied AI 的统一 backbone** 已成为两个领域的共识。对比早期：VLN 领域的 VLN-DUET 和 ETPNav 使用 task-specific transformer，VLA 领域的 Octo 使用轻量 transformer 而无 VLM 预训练——这些架构正在被 VLM-based 方案取代。VLM backbone 带来的核心优势是：（1）web-scale 预训练提供丰富的视觉-语言-常识知识；（2）instruction following 能力天然适配 language-conditioned 任务；（3）跨 domain 知识迁移（如 RT-2 的 emergent reasoning）。

**2. Instruction-conditioned action prediction 成为统一范式**

VLA 和 VLN 的核心 pipeline 可以抽象为同一个公式：$\pi(a | o, \ell)$——给定观测 $o$ 和语言指令 $\ell$，预测动作 $a$。两者的差异仅在于 action space 的具体形式和 observation 的模态。这一统一视角正是 NaVILA 能够将 VLN 重构为 navigation VLA 的根本原因。

**3. Pre-training on web data 的共同策略**

RT-2 的 co-fine-tuning（web VL + robot data）、π0.5 的 web data co-training、NaVILA 的 YouTube egocentric video 利用——两个领域都在探索如何利用海量互联网数据增强 embodied 模型的泛化能力。这反映了一个共同挑战：robot/navigation-specific 数据稀缺，而 web data 可以提供 visual grounding、commonsense reasoning 和 diverse scene understanding。

**4. Hierarchical 架构成为主流**

π0.5 的 hierarchical inference（语义子任务 → flow matching action）、MEM 的分层策略（高层子任务 + 记忆更新 → 低层动作生成）、ETPNav 的 hierarchical planning（transformer planner → obstacle-avoiding controller）、NaVILA 的 two-level hierarchy（VLM → 语言动作 → RL policy）——两个领域不约而同地采用了**高层语义规划 + 低层动作执行**的分层设计。这不是巧合，而是 long-horizon embodied tasks 的内在需求：高层需要抽象推理（"下一步应该去厨房拿杯子"），低层需要精细控制（具体的关节角度或 waypoint），单一层级无法同时满足两者。

### 4.3 差异分析（Differences）

**1. Action Space 粒度的根本差异**

这是 VLA-VLN 统一最核心的障碍。VLA 的 action space 是 continuous joint-level control（π₀: 18 维，含双臂 6-DoF + grippers + base + torso，50 Hz），而 VLN 的 action space 是 discrete waypoint selection（VLN-DUET: nav-graph node，~1-5 Hz）或 mid-level language commands（NaVILA: "move forward 75cm"）。两者的控制频率相差一个数量级（50 Hz vs. 1-5 Hz），反映了任务本质的差异：manipulation 需要精细力控制，navigation 需要全局路径规划。

**2. 环境表示的分野**

VLN 系统普遍依赖显式空间表示——VLN-DUET 的 topological map、ETPNav 的 online waypoint graph——作为 high-level planning 的基础。VLA 系统则通常不构建显式环境模型，而是通过 end-to-end learning 隐式编码空间信息。这一差异源于任务需求：navigation 必须知道"哪里可以去"（free space、obstacles、landmarks），而 table-top manipulation 的工作空间相对受限，不需要全局空间理解。

**3. Sim vs. Real 的训练范式差异**

VLN 研究绝大多数在 simulation 中进行（Habitat、Matterport3D、Gibson），仅 NaVILA 实现了真实部署。VLA 研究（尤其 π₀ 系列）主要在真实 robot 上训练和评估。这一差异有深层原因：（1）navigation 涉及整个建筑级别的场景，真实数据采集成本极高，而 3D 场景扫描（MP3D 提供 90 栋建筑）可以大规模生成导航数据；（2）manipulation 涉及物理接触和力控制，simulation 的 physics fidelity 不足以 transfer（sim-to-real gap 更大）。

**4. 评估体系的不可比性**

VLN 使用 R2R、REVERIE、SOON 等 benchmark 的 Success Rate / SPL 评估，VLA 使用 task-specific 成功率评估。两者的 evaluation protocol 完全不同，缺乏统一的 benchmark 来比较跨领域模型。

### 4.4 趋同点：统一的自然连接处（Where Unification is Natural）

**1. 共享 VLM Backbone**

最直接的统一点。一个预训练 VLM（如 PaliGemma、VILA）可以同时作为 navigation planner 和 manipulation controller 的 backbone。NaVILA 已经证明了 VLM 可以驱动 navigation，π₀ 证明了 VLM 可以驱动 manipulation——理论上，同一个 VLM 可以根据任务需求切换到不同的 action head。

**2. Language-conditioned hierarchical planning**

π0.5 的语义子任务预测和 NaVILA 的语言化 mid-level action 本质上是同一种思路：**用自然语言作为高层规划和低层执行之间的 interface**。这为统一提供了一个优雅的 abstraction layer——high-level planner 用语言描述下一步目标（"navigate to the kitchen sink" 或 "pick up the red mug"），low-level policy 根据具体 domain（navigation 或 manipulation）选择对应的 action generation module。

**3. Web-scale pre-training 的共享**

Navigation 和 manipulation 可以共享同一个 VLM 的 web pre-training，因为 visual scene understanding、object recognition、spatial reasoning 等能力是两者共有的。π0.5 的 co-training 实验已经证明，来自不同 embodiment 和任务的数据可以互相增强——将 navigation data 加入 co-training mixture 是自然的扩展。

### 4.5 分歧点：统一的核心挑战（Where Unification is Challenging）

**1. Action Space Mismatch**

Manipulation 需要 50 Hz 的 continuous joint-level control（π₀ 的 18 维 action space），navigation 需要 1-5 Hz 的 discrete/mid-level waypoint decisions。一个 unified action head 如何同时处理两种截然不同的 action space？可能的方案：（a）hierarchical decomposition，高层 planner 统一，低层 action head 分离（类似 NaVILA 的 VLM + RL locomotion policy）；（b）统一的 continuous action space，将 navigation 也建模为 continuous base velocity control（但会丧失 topological planning 的全局优势）；（c）multi-head design，类似 π₀ 的 action expert，为不同 action domain 设计专用 head。

**2. 控制频率的不兼容**

50 Hz 的 manipulation control 和 1-5 Hz 的 navigation decision 不仅是数值上的差异，更反映了不同的计算需求。Manipulation 需要在每个 control cycle（20ms）内完成 inference，而 navigation 可以容忍更长的 planning latency。一个统一系统可能需要**异步双 loop 设计**：高频 manipulation loop 持续运行，低频 navigation loop 在需要时触发。

**3. 环境表示的不一致**

VLA 系统的 end-to-end 架构不构建显式环境模型，而 VLN 系统依赖 topological map 或 semantic map 进行全局规划。统一系统需要同时支持：（a）全局空间理解（"厨房在卧室左边"）和（b）局部精细感知（"杯子在桌面边缘，需要从侧面抓取"）。Section 3 的分析表明，ConceptGraphs 式 3D scene graph 最有潜力同时满足两者，但目前还没有 VLA 系统真正集成这种 spatial representation。

**4. 训练数据的异构性**

Navigation 数据主要来自 simulation（Habitat trajectories），manipulation 数据主要来自真实 robot demonstrations。两者的 visual appearance、physics dynamics、action distribution 都有显著差异。π0.5 的异构 co-training 已经展示了混合不同来源数据的可行性，但 sim navigation data 和 real manipulation data 的混合是否有效仍是 open question。

### 4.6 SLAM 的角色：弥合 Navigation-Manipulation Gap 的空间基础设施

Section 3 的分析表明，semantic SLAM 可以作为 VLN-VLA 统一的 "missing piece"：

**Navigation 侧**：SLAM 提供的 persistent spatial map 可以替代 VLN 中 hand-crafted 的 topological map。VLMaps 的 language-queryable grid map 已经展示了 open-vocabulary navigation planning 的可能。ConceptGraphs 的 scene graph 提供了更丰富的 object-level abstraction，可以直接作为 navigation waypoints（类似 VLN-DUET 的 topological map nodes，但语义更丰富）。

**Manipulation 侧**：ConceptGraphs 的 3D object nodes 提供了 manipulation targets 和空间关系信息。SplaTAM 的 dense Gaussian field 提供了高质量的 3D geometry，有利于 grasp planning。这些 spatial representations 可以增强 VLA 系统目前缺失的环境理解能力。

**统一侧**：一个 incrementally updated 的 semantic spatial representation 可以同时服务 navigation planning（"导航到厨房水槽"→ 在 scene graph 中查找 "kitchen sink" node → path planning）和 manipulation grounding（"抓起桌上的杯子"→ 定位 "mug" node → 获取 3D pose → grasp planning）。更重要的是，这种 spatial memory 为 long-horizon Nav+Manip 任务提供了 persistent context——robot 可以记住之前探索过的区域和物体位置，在 navigation 和 manipulation 之间无缝切换。

**当前 gap**：尽管 Section 5 的 OK-Robot 使用了 VoxelMap（类似 VLMaps），但其 navigation 和 manipulation 模块并不共享这一空间表示。目前没有任何系统真正实现了 "shared semantic SLAM serving both navigation and manipulation"。这是一个重要的研究方向。

### 4.7 Simulation 与 Sim-to-Real：两个领域的不同策略

**VLN 的 simulation 生态**

VLN 深度依赖 simulation 环境：Habitat Simulator 配合 Matterport3D（90 栋建筑扫描）、Gibson（572 个场景）提供了大规模导航训练和评估的基础。这使得 VLN 研究可以低成本地大规模实验，但也导致了严重的 sim-to-real gap：大多数 VLN 模型从未在真实世界中测试。NaVILA 的突破在于通过两个策略缓解这一 gap：（1）引入 YouTube egocentric 视频作为 real-world visual data；（2）用 mid-level 语言动作解耦 high-level perception（可在 sim 中训练并 transfer）和 low-level locomotion（用真实 RL policy）。

**VLA 的 real-world 优先策略**

VLA 领域（尤其 π₀ 系列）选择了不同路线：直接在真实 robot 上收集数据和评估。这是因为 manipulation 涉及复杂的物理接触（摩擦、变形、柔软物体），当前 simulation 的 physics fidelity 不足以 reliable transfer。π₀ 使用 10K+ 小时真实数据，RoboClaw 通过 EAP 实现自主数据收集——两者都在寻找 scalable 的 real-world data 方案，而非依赖 simulation。

**统一系统的 simulation 需求**

一个统一的 Nav+Manip 系统需要在建筑级别的 navigation 和 object-level 的 manipulation 之间切换，这对 simulation 提出了极高要求。现有 sim 平台的局限：（1）Habitat/MP3D 适合 navigation 但缺乏精细物理交互（无法模拟抓取、推拉）；（2）Isaac Sim/MuJoCo 适合 manipulation 但缺乏大规模建筑级场景；（3）AI2-THOR 兼顾导航和交互但场景多样性有限。NaVILA 引入的 VLN-CE-Isaac benchmark（Isaac Sim 高保真环境）是一个有意义的尝试，将 VLN 评估扩展到更接近 real-world physics 的 simulation 环境。理想的统一 sim 平台需要同时具备：large-scale 建筑场景 + high-fidelity 物理交互 + diverse object assets。

### 4.8 Section 4 Takeaway

1. **VLA 和 VLN 在四个维度上趋同**：VLM backbone、language-conditioned action prediction、web-scale pre-training、hierarchical 架构。NaVILA 是两者趋同的最直接证据——它本质上就是一个 navigation-focused VLA。
2. **Action space mismatch 是统一的最大障碍**：50 Hz continuous joint control vs. 1-5 Hz discrete waypoint selection，控制频率相差一个数量级。Hierarchical decomposition（共享 VLM backbone + 分离 action heads）是最可行的统一路径。
3. **Semantic SLAM 是统一的空间基础设施**：ConceptGraphs 式 scene graph 可以同时服务 navigation waypoints 和 manipulation targets，但目前没有系统真正实现 shared spatial representation。
4. **Simulation 生态需要升级**：现有 sim 平台无法同时满足 building-scale navigation 和 high-fidelity manipulation 的需求，制约了 unified Nav+Manip 系统的开发和评估。
5. **统一架构的可能形态**：VLM backbone（shared）→ language-mediated hierarchical planner（shared）→ domain-specific action heads（navigation: waypoint selection / VLM mid-level commands; manipulation: flow matching continuous control）→ shared semantic spatial memory（ConceptGraphs + SplaTAM + VLMaps）。

## 5. 现有 Nav+Manip 系统

### 为什么 Nav+Manip 是 VLN-VLA 统一的试金石？

前面四个 Section 分别从 VLA（manipulation）、VLN（navigation）、语义 SLAM（spatial representation）角度梳理了各领域的进展。但一个真正有用的 embodied agent 必须**同时具备 navigation 和 manipulation 能力**——在家庭环境中，"把桌上的杯子放到厨房水槽里"这样的简单指令就需要：定位杯子（perception）→ 导航到桌子（navigation）→ 抓取杯子（manipulation）→ 导航到厨房（navigation）→ 放下杯子（manipulation）。现有的 Nav+Manip 系统是检验 VLN-VLA 能否统一的最直接试验场。

### 代表系统总览

| System | Year | 架构类型 | Navigation | Manipulation | Task Planning | 成功率 |
|--------|------|----------|------------|--------------|---------------|--------|
| [[Ahn2022-SayCan\|SayCan]] | 2022 | Modular（LLM planner + skill library） | Pretrained nav skill | Pretrained pick/place skills | LLM × affordance scoring | 74%（执行）/ 84%（规划） |
| HomeRobot OVMM | 2023 | Modular（heuristic/RL baselines） | Learned / heuristic nav | Learned / heuristic grasp | Heuristic pipeline | 20%（真实）/ 10.8%（竞赛最佳） |
| [[Liu2024-OKRobot\|OK-Robot]] | 2024 | Modular（VLM perception + nav + grasp） | A* on occupancy grid | AnyGrasp + LangSam | Linear state machine | 58.5%（真实家庭） |
| [[Fu2024-MobileALOHA\|Mobile ALOHA]] | 2024 | End-to-end（imitation learning） | Whole-body policy（含底盘） | Whole-body policy（含双臂） | 无显式 planning | ~90%（co-training 后，per-task） |
| TidyBot | 2023 | Modular（LLM preference + mobile manip） | Heuristic nav | Learned grasp | LLM 个性化偏好推理 | 85%（物品放置） |

### 两大架构范式

现有 Nav+Manip 系统可以清晰地分为两大架构范式：

#### 范式一：Modular Pipeline（SayCan、OK-Robot、TidyBot、HomeRobot）

**架构特征**：将 Nav+Manip 任务分解为独立模块——perception、navigation、manipulation、task planning——各模块独立开发和训练，通过 pipeline 或 state machine 串联。

**SayCan**（[[Ahn2022-SayCan|详细笔记]]）是这一范式的奠基性工作。其核心创新是 "Say × Can" scoring：LLM 评估每个 skill 的语义合理性（Say），value function 评估 skill 在当前状态下的可执行性（Can），两者相乘得到 grounded plan。Skill library 包含 551 个独立训练的 skills（navigation、pick、place 等），在 Google Everyday Robot 上实现 74% 的端到端成功率。**局限**：skill library 是 closed-set 的，无法处理 unseen 动作；navigation 和 manipulation skills 完全独立训练。

**OK-Robot**（[[Liu2024-OKRobot|详细笔记]]）代表了 modular 范式的最新进展。其创新在于用 open-knowledge models（CLIP、OWL-ViT、AnyGrasp、LangSam）替代 task-specific 训练，实现 **zero-shot** Nav+Manip。Pipeline 为：iPhone 扫描建 VoxelMap → CLIP query 定位目标 → A* 导航 → AnyGrasp 抓取。在 10 个真实家庭中达到 58.5% 成功率。**局限**：linear state machine 无错误恢复；nav 和 manip 无 shared representation；需要预先扫描建图。

**TidyBot** 聚焦个性化：用 LLM 从少量用户示例中推断物品放置偏好（"这类物品应该放到哪里"），然后驱动 mobile manipulator 执行。在 unseen 物品上达到 91.2% 的偏好预测准确率和 85% 的真实放置成功率。

**Modular Pipeline 的共同模式**：
1. **Perception 和 Planning 分离**：用 VLM/LLM 做 high-level 理解和规划，用 task-specific 模块执行
2. **Sequential handoff**：nav → manip 是硬切换，导航结束后才开始操作
3. **No shared spatial representation**：navigation 用 occupancy grid / nav skill，manipulation 用 point cloud / grasp model，两者不共享空间信息
4. **Error propagation without recovery**：一旦某个模块失败，整个任务失败

#### 范式二：End-to-End Learning（Mobile ALOHA）

**架构特征**：不显式区分 navigation 和 manipulation，用 unified policy 直接从 observation 映射到 whole-body action（包含底盘速度 + 关节角度）。

**Mobile ALOHA**（[[Fu2024-MobileALOHA|详细笔记]]）是这一范式的代表。通过低成本 whole-body teleoperation 系统收集 bimanual mobile manipulation 数据，用 ACT（Action Chunking with Transformers）做 supervised behavior cloning。核心创新是 **co-training**：仅需 50 条 mobile demonstrations，加上大量已有的 static ALOHA 数据联合训练，成功率提升高达 90%。成功完成煎虾、开柜门存锅、叫电梯等复杂任务。

**End-to-End 的优势**：
1. **Nav-Manip 自然融合**：policy 同时控制底盘和双臂，无 handoff 问题
2. **Implicit coordination**：model 学会在移动中调整姿态，如走近物体的同时伸出手臂
3. **Simple pipeline**：无需模块化设计和复杂 interface

**End-to-End 的局限**：
1. **Navigation 范围有限**：teleoperation 数据只覆盖短距离移动，无法 navigate 到远处房间
2. **No spatial understanding**：无地图、无 SLAM、无空间记忆
3. **No language conditioning**：每个任务需单独训练，不能通过语言指定新目标
4. **Per-task data collection**：不同于 OK-Robot 的 zero-shot，每个新任务需要约 50 条 demonstrations

### 共同瓶颈分析

无论 modular 还是 end-to-end，现有 Nav+Manip 系统面临以下共同瓶颈：

**1. Nav → Manip Handoff 问题**
Modular 系统中，navigation 结束后 robot 停在某个位置开始 manipulation——但这个位置可能不是 manipulation 的最优位置。OK-Robot 用 scoring function 尝试平衡"靠近目标"和"保持 gripper 空间"，但仍然是 heuristic 的。理想情况下，navigation 应该**根据 manipulation 的需求动态调整**终点位姿。

**2. 缺乏 Shared Spatial Representation**
OK-Robot 的 navigation 用 2D occupancy grid，manipulation 用 3D point cloud——两者是完全不同的空间表示。SayCan 更极端，navigation 和 manipulation 甚至没有共享的 scene understanding。这导致两个问题：（1）navigation 无法利用 manipulation 的场景理解（如"桌面太拥挤，需要从另一侧靠近"）；（2）manipulation 无法利用 navigation 的空间记忆（如"杯子在上次经过的桌子上"）。Section 3 讨论的 ConceptGraphs 式 scene graph 可以作为这种 shared representation。

**3. 独立训练导致的 Capability Gap**
SayCan 的 551 个 skills 各自独立训练，不存在"一边走一边抓"的 skill。Mobile ALOHA 的 end-to-end policy 虽然解决了这个问题，但缺乏 generalization。这反映了一个根本矛盾：modular 系统有 generalization 但缺 coordination，end-to-end 系统有 coordination 但缺 generalization。

**4. Open-Vocabulary vs Task-Specific 的 Tradeoff**
OK-Robot 和 SayCan 支持 open-vocabulary 指令但成功率较低（58-74%），Mobile ALOHA 在特定任务上成功率很高（~90%）但不支持语言指令。如何同时实现 open-vocabulary generalization 和 high task success rate 仍是 open problem。

### 理想的统一 Nav+Manip 系统应该是什么样的？

综合以上分析，结合 Section 1-3 的技术进展，一个理想的 Nav+Manip 统一系统应具备：

1. **Hierarchical VLA architecture**（来自 Section 1：[[Black2025-Pi05|π0.5]]、[[Torne2026-MEM|MEM]]）：
   - High-level：VLM 做 task decomposition 和 sub-goal generation（类似 SayCan 的 LLM planning，但用 VLM 替代纯文本 LLM 以获得 visual grounding）
   - Low-level：flow matching action generation 同时输出 navigation 和 manipulation actions（类似 Mobile ALOHA 的 whole-body control，但由 foundation model 驱动）

2. **Shared semantic spatial representation**（来自 Section 3：[[Gu2024-ConceptGraphs|ConceptGraphs]]）：
   - Navigation 和 manipulation 共享同一个 incrementally updated scene graph
   - Graph nodes 同时作为 navigation waypoints 和 manipulation targets
   - Language-queryable spatial memory 支持 long-horizon 任务

3. **Unified training with diverse data**（来自 Section 1：[[Black2024-Pi0|π₀]] 的 cross-embodiment training）：
   - Navigation data（VLN datasets）、manipulation data（robot demonstrations）、internet video 联合训练
   - 类似 π₀ 的异构数据 co-training 策略

4. **Closed-loop execution with error recovery**（现有系统的共同缺陷）：
   - Navigation 和 manipulation 不是 sequential handoff，而是 interleaved execution
   - Failure detection + re-planning：grasp 失败 → 重新调整位姿 → 再次尝试

这一方向与 VLN-VLA 统一的核心目标完全一致：**一个 foundation model 同时具备 navigate 和 manipulate 的能力，通过 shared spatial representation 在两种模式间无缝切换**。

### Section 5 Takeaway

1. **两大范式各有所长**：modular pipeline（SayCan、OK-Robot）擅长 open-vocabulary generalization 和 task planning，end-to-end（Mobile ALOHA）擅长 whole-body coordination 和 high success rate。
2. **Nav→Manip handoff 是核心瓶颈**：所有 modular 系统都在 nav 结束后硬切换到 manip，缺乏 nav-manip co-optimization。
3. **Shared spatial representation 缺失**：navigation 和 manipulation 使用不同的空间表示，无法互相利用对方的 scene understanding。Section 3 的 semantic SLAM（ConceptGraphs）可以填补这一空白。
4. **理想方案是 hierarchical VLA + shared scene graph**：结合 VLA 的 unified action generation、VLN 的 spatial planning、和 semantic SLAM 的 persistent spatial memory，实现 nav-manip-unified foundation model。

## 6. Gap 分析与潜在方向

综合 Section 1-5 的分析，本节识别当前 VLN-VLA 统一研究中的关键 gap，梳理相关 benchmarks，并提出潜在研究方向。

### 6.1 Research Gaps

1. **No end-to-end model handles both continuous navigation and dexterous manipulation。** 现有系统要么是 modular pipeline（OK-Robot、SayCan：nav 和 manip 完全独立），要么是 short-range end-to-end policy（Mobile ALOHA：只覆盖局部移动）。没有一个 foundation model 能在 building-scale navigation 和 dexterous manipulation 之间端到端切换。π0.5 实现了 long-horizon 家务任务，但其 navigation 能力有限（不涉及跨房间导航）；NaVILA 实现了 VLM-driven navigation，但不涉及 manipulation。

2. **SLAM representations 未针对 VLM consumption 优化。** Section 3 分析了三类语义空间表示（dense feature maps、3D scene graphs、neural/Gaussian fields），但它们都是独立构建的，未与 VLM backbone 端到端优化。ConceptGraphs 的 scene graph 需要离线构建，VLMaps 的 feature map 需要预先扫描——没有 "online, VLM-native spatial representation" 能实时更新并直接嵌入 VLM 的 reasoning loop。

3. **Navigation 和 manipulation 不共享 spatial representation。** Section 5 明确指出，即使是最先进的 OK-Robot，其 navigation（2D occupancy grid）和 manipulation（3D point cloud）使用完全不同的空间表示。没有系统实现了 "shared semantic SLAM serving both navigation and manipulation"。

4. **Action space mismatch 缺乏优雅的统一方案。** Section 4 识别了 50 Hz continuous joint control（VLA）vs. 1-5 Hz discrete waypoint selection（VLN）的根本差异。现有方案要么是 hierarchical decomposition（NaVILA 的 VLM → language action → RL policy），要么是 whole-body policy（Mobile ALOHA 的 ACT），但没有方案能同时实现 building-scale planning 和 fine-grained manipulation control。

5. **Sim-to-real gap 在 Nav+Manip 联合任务中更加严重。** VLN 主要在 simulation 中评估（Habitat/MP3D），VLA 主要在 real world 中训练。统一系统需要同时处理 building-scale navigation（sim 数据丰富）和 object-level manipulation（需要 real data），但没有 simulation 平台能同时提供 large-scale 建筑场景和 high-fidelity 物理交互。

6. **缺乏 closed-loop error recovery 机制。** Modular 系统（SayCan、OK-Robot）一旦某个模块失败，整个任务失败；end-to-end 系统（Mobile ALOHA）缺乏显式 failure detection。理想系统需要在 navigation 和 manipulation 之间实现 interleaved execution with re-planning。

7. **Long-horizon spatial memory 机制不成熟。** MEM 引入了视频短期记忆和语言长期记忆，但缺乏 explicit spatial memory。π0.5 依赖 VLM implicit memory，无法回溯到之前探索过的区域。SLAM 可以提供 persistent spatial memory，但如何将 SLAM map 高效注入 VLM 的 context window 仍是 open problem。

### 6.2 相关 Benchmarks

#### ALFRED（Action Learning From Realistic Environments and Directives）

- **平台**：AI2-THOR 2.0 simulator，120 个室内场景
- **数据规模**：25,743 条自然语言指令，8,055 个 expert demonstrations，平均每条 50 步，共 428,322 image-action pairs
- **任务类型**：家庭日常任务（用刀切苹果、在水槽清洗杯子、用微波炉加热等），同时涉及 navigation（MoveAhead, Rotate, Look）和 interaction（Pickup, Put, Open, Close, ToggleOn/Off, Slice）
- **评估能力**：language grounding + navigation + object interaction，是最早要求 nav+manip 联合的 benchmark 之一
- **SOTA 现状**：近期 EmbodiedBench 基于 ALFRED 构建了 EB-ALFRED 子集，GPT-4o 仅达 28.9%，ERA（Embodied Reasoning Agent）在此基础上提升约 8.4%，说明即使最强 VLM 也远未解决该任务
- **局限**：discrete action space，物理交互较为简化，与 real-world manipulation 差距较大

#### TEACh（Task-driven Embodied Agents that Chat）

- **平台**：基于 AI2-THOR，扩展自 ALFRED
- **数据规模**：3,000+ 条人-人交互对话
- **任务类型**：Commander（有 oracle 信息）通过自然语言对话指导 Follower 完成家务任务（从 "Make Coffee" 到 "Prepare Breakfast"），navigation + object manipulation + dialogue understanding
- **评估能力**：在 ALFRED 基础上增加了 interactive dialogue grounding——agent 需要理解多轮对话（含无关话语、指代消解、对话省略）并执行对应动作
- **SOTA 现状**：Episodic Transformer 在 test-unseen 上仅 5.02% SR（对比 ALFRED 的 8.57%），说明 dialogue-conditioned embodied task 难度远超单指令场景
- **局限**：与 ALFRED 共享 discrete action space 和简化物理的局限

#### HomeRobot OVMM（Open-Vocabulary Mobile Manipulation）

- **平台**：Habitat 2.0 simulator + real-world Hello Robot Stretch
- **任务类型**：open-vocabulary mobile manipulation——给定语言描述的目标物体和放置位置，agent 需要在多房间环境中导航、找到物体、抓取并放置到目标位置
- **评估能力**：真正的 Nav+Manip 联合评估，且要求 open-vocabulary 泛化（unseen objects/receptacles）
- **竞赛情况**：NeurIPS 2023 Challenge 有 61 支队伍参赛、368 次提交，仅 7 支队伍超过 baseline；CVPR 2024 继续举办
- **SOTA 现状**：simulation 最佳约 10.8%，real-world 约 20%——极低的成功率反映了 open-vocabulary Nav+Manip 的巨大挑战
- **意义**：**最接近 VLN-VLA 统一评估的现有 benchmark**，同时要求 building-scale navigation、open-vocabulary perception 和 object manipulation

#### Benchmark Gap 分析

| Benchmark | Navigation | Manipulation | Dialogue | Open-Vocab | Real-World | Continuous Actions |
|-----------|-----------|-------------|----------|-----------|-----------|-------------------|
| ALFRED | ✓ | ✓（简化） | ✗ | ✗ | ✗ | ✗ |
| TEACh | ✓ | ✓（简化） | ✓ | ✗ | ✗ | ✗ |
| HomeRobot OVMM | ✓ | ✓ | ✗ | ✓ | ✓ | ✓（部分） |
| R2R / R2R-CE | ✓ | ✗ | ✗ | ✗ | ✗ | ✓（CE） |
| OXE Benchmarks | ✗ | ✓ | ✗ | 部分 | ✓ | ✓ |

可以看到，**没有现有 benchmark 同时覆盖 building-scale continuous navigation + dexterous manipulation + open-vocabulary language grounding + real-world evaluation**。这本身就是一个重要 gap。

### 6.3 潜在方向

#### 方向一：Hierarchical VLA with Shared Semantic Scene Graph

**核心思路**：构建一个统一的 hierarchical VLA 架构，其中 VLM backbone 共享，action heads 按 domain 分离，semantic scene graph 作为 navigation 和 manipulation 的 shared spatial representation。

具体而言，系统包含三层：（1）**High-level VLM planner**：接受语言指令和 scene graph 的文本化描述，输出 sub-goal 序列（如 "navigate to kitchen → locate mug on counter → pick up mug → navigate to sink → place mug in sink"）；（2）**Mid-level scene graph manager**：实时维护 ConceptGraphs 式 3D scene graph，graph nodes 同时作为 navigation waypoints 和 manipulation targets，支持 language query 和 incremental update；（3）**Low-level domain-specific action heads**：navigation head 基于 scene graph 做 path planning + waypoint following，manipulation head 基于 flow matching 生成 continuous joint-level control。这一架构直接回应了 Section 4 识别的 "shared VLM backbone + separated action heads" 统一路径，以及 Section 3 提出的 "dense geometry + semantic graph + language interface" 层次化空间表示方案。关键技术挑战在于：scene graph 的 online 构建效率、graph-to-text serialization 对 VLM context window 的压力、以及 navigation 和 manipulation action heads 的 co-training 策略。

#### 方向二：VLM-Native Spatial Memory for Long-Horizon Embodied Tasks

**核心思路**：将 SLAM-based spatial representation 直接嵌入 VLM 的 token space，使 VLM 能够 "原生" 理解和操作空间信息，而非将 map 作为外部模块。

目前 VLM-based embodied systems（π0.5、NaVILA、NavGPT）要么不使用显式空间表示（依赖 VLM implicit memory），要么将空间信息文本化后作为 prompt 输入（NavGPT 的文本化视觉观测）。前者在 long-horizon 任务中丧失空间 context，后者效率低下（冗长的文本描述占据 context window）。潜在方案是将 scene graph 或 spatial features 编码为 special tokens，使 VLM 可以通过 attention 直接访问空间信息——类似 π₀ 的 action expert tokens，但用于 spatial representation。这需要探索：spatial token 的编码方式（graph embedding? voxel features? Gaussian splat features?）、与 VLM pre-trained weights 的兼容性、以及 incremental update 机制（如何在不重新编码整个 map 的情况下更新 spatial tokens）。MEM 的多尺度记忆机制（视频短期 + 语言长期）可以作为起点，扩展加入 spatial 维度。

#### 方向三：Unified Nav+Manip Benchmark with Continuous Actions and Real-World Transfer

**核心思路**：设计一个新的 benchmark，弥合现有 benchmarks 的 gap——同时要求 building-scale continuous navigation、dexterous manipulation、open-vocabulary language grounding，并提供 sim-to-real transfer protocol。

HomeRobot OVMM 是最接近的现有 benchmark，但其 manipulation 仍较为简化（pick-and-place），且 sim-to-real protocol 不够系统。新 benchmark 应包含：（1）**多房间连续导航**——不是 discrete nav-graph，而是 continuous locomotion in building-scale environments；（2）**多样化操作任务**——超越 pick-and-place，包含 articulated object interaction（开抽屉、旋转阀门）、tool use、bimanual tasks；（3）**open-vocabulary 指令**——支持自然语言描述 unseen objects 和 novel tasks；（4）**统一 sim 平台**——结合 Habitat 的 building-scale 场景和 Isaac Sim 的高保真物理（NaVILA 的 VLN-CE-Isaac 是有意义的起步）；（5）**标准化 sim-to-real protocol**——定义 transfer 评估方法，使不同方法可以公平比较。

### 6.4 Open Questions

1. **Hierarchical decomposition 是否是唯一路径？** 目前所有成功的 long-horizon embodied 系统（π0.5、MEM、NaVILA、ETPNav）都采用 hierarchical 架构。是否存在一种 truly unified action space 能同时表示 navigation 和 manipulation，使 single-level model 成为可能？
2. **Scene graph 的 scalability 问题**：ConceptGraphs 在小型场景中表现良好，但 building-scale 环境可能产生数千个 nodes——如何高效 serialize 并输入 VLM 的有限 context window？需要 graph summarization / attention-based selection 机制。
3. **Sim navigation data + real manipulation data 的 co-training 是否有效？** π0.5 证明了异构 real data 的 co-training 可行性，但 sim 和 real 数据的 domain gap 更大。是否需要 domain adaptation 层？还是足够 diverse 的 web data 可以作为 bridge？
4. **VLM 的 spatial reasoning 能力是否足够？** 现有 VLM（PaliGemma、VILA、GPT-4V）在 2D 图像理解上表现出色，但 3D spatial reasoning 能力仍然有限。是否需要专门的 spatial pre-training（如 3D scene understanding 数据），还是 scene graph 可以弥补这一不足？
5. **Multi-robot / multi-session 场景中的 shared spatial memory**：如果多个 robot 共享同一 scene graph，如何处理 concurrent updates、conflicting observations、和 map merging？这对 real-world deployment 至关重要但几乎未被探索。

### 6.5 Idea Seeds

> [!tip] **最具潜力方向：Hierarchical VLA with Shared Semantic Scene Graph**（方向一）
>
> **为什么最有潜力**：
> - 直接回应 Section 4 和 Section 5 共同识别的核心 gap（action space mismatch + 缺乏 shared spatial representation）
> - 每个组件都有成熟的技术基础：VLM backbone（π₀/NaVILA）、scene graph（ConceptGraphs）、flow matching action generation（π₀ 系列）、hierarchical planning（π0.5/ETPNav）
> - HomeRobot OVMM 提供了直接可用的评估平台
> - 渐进式研究路径：可以先在 simulation 中验证架构（Habitat + AI2-THOR），再 transfer 到 real robot（Hello Robot Stretch）
>
> **具体 Idea**：在 HomeRobot OVMM 平台上，实现一个 三层架构系统：
> 1. **VLM Planner**（基于 fine-tuned PaliGemma 或 VILA）：接收指令 + scene graph summary → 输出 sub-goal sequence
> 2. **Scene Graph Memory**（基于 ConceptGraphs 简化版）：online incremental 构建，nodes 包含 CLIP embedding + 3D position + navigability flag + graspability flag
> 3. **Dual Action Heads**：navigation head（waypoint selection on scene graph → local planner）+ manipulation head（flow matching à la π₀）
>
> 关键实验：对比 shared scene graph vs. separate representations（OK-Robot 式）的 Nav+Manip 成功率差异。
>
> **预期贡献**：首次在 unified benchmark 上验证 "shared semantic spatial representation improves Nav+Manip coordination" 的假设。

> [!note] **次要方向：VLM-Native Spatial Memory**（方向二）
>
> 更长期的研究方向，技术难度更高但潜在 impact 更大。可作为方向一的后续工作——先用 外部 scene graph 验证 shared representation 的价值，再探索将 spatial information 内化到 VLM token space。
