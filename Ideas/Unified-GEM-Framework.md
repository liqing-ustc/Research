---
title: "Unified GEM: Grounding-Exploration-Manipulation 三元决策框架"
tags: [mobile-manipulation, VLA, VLN]
status: raw
linked_project:
date_updated: "2026-03-31"
---
## Hypothesis

若将 MTU3D 的 unified grounding-exploration decision space 扩展为 grounding-exploration-manipulation 三元决策空间，则 agent 在 Nav+Manip 联合任务（如 HomeRobot OVMM）上的成功率将显著高于 sequential nav→manip pipeline（如 OK-Robot），因为三元决策允许 agent 根据场景状态动态选择最优行动模式（定位已知物体 / 探索未知区域 / 执行操作），而非在 navigation 和 manipulation 之间硬切换。

## Motivation

1. **解决的知识空白**：DomainMaps/VLA.md Open Question #1 （Nav+Manip 统一架构）和 DomainMaps/SpatialRep.md Open Question #1 （Shared spatial representation for Nav+Manip）。现有系统（OK-Robot、SayCan）采用 sequential handoff，navigation 和 manipulation 完全独立，信息不共享。
2. **成功的影响**：首次在 unified benchmark 上验证"统一决策空间优于 sequential pipeline"的假设。HomeRobot OVMM 当前 SOTA 仅约 10%，有巨大提升空间。
3. **时机成熟**：MTU3D（ICCV 2025）已在 navigation 中验证了 grounding-exploration 统一的有效性（GOAT-Bench +23% SR），Habitat 2.0 支持物理交互，AnyGrasp 等现成工具可提供 graspability prior。

## Related Work

- [[Papers/2507-MTU3D]] - 核心基础：unified grounding-exploration decision space + online query memory，GOAT-Bench +23% SR。GEM 在此基础上增加 manipulation 作为第三个决策维度。
- [[Papers/2401-OKRobot]] - 对标 baseline：modular sequential pipeline（VoxelMap → A* nav → AnyGrasp manip），58.5% pick-and-drop in real homes 但无 error recovery 和 shared representation。
- [[Papers/2309-ConceptGraphs]] - 3D scene graph 同时提供 navigation waypoints 和 manipulation targets 的理论基础。
- [[Papers/2204-SayCan]] - LLM × affordance scoring 的 modular pipeline，551 skills 独立训练，无 nav-manip coordination。

**Novelty**: 4/5 — closest works: [[Papers/2507-MTU3D]]（grounding-exploration 但无 manipulation）, [[Papers/2401-OKRobot]]（nav+manip 但 sequential pipeline）。无已有工作将 grounding-exploration-manipulation 统一到单一决策空间。

## Approach sketch

在 MTU3D 的 Spatial Reasoning Transformer 上增加第三类 query（manipulation query），与 object queries 和 frontier queries 共享 cross-attention：

1. **三类 Query 定义**：
   - **Object query**（来自 MTU3D）：已观测物体的 CLIP embedding + 3D bounding box + confidence，用于 grounding
   - **Frontier query**（来自 MTU3D）：unexplored 区域的 centroid + estimated information gain，用于 exploration
   - **Manipulation query**（新增）：当 object query 满足任务目标且在 gripper workspace 内时激活，score 基于 (a) 语义匹配度（CLIP embedding 与指令相似度），(b) 可达性（gripper workspace 约束），(c) graspability（基于 depth geometry 估计或 AnyGrasp prior）

2. **统一 Scoring**：三类 query 共享 Spatial Reasoning Transformer 的 attention，输出统一的 score distribution。Agent 每步选择 score 最高的 query 执行对应动作（navigate to object / explore frontier / manipulate target）。

3. **训练策略**：
   - Stage 1：在 MTU3D 原有数据上训练 grounding + exploration（复用已有 pipeline）
   - Stage 2：在 Habitat 2.0 + ReplicaCAD 中增加 pick-and-place 交互数据，联合训练三类 query
   - Stage 3：在 HomeRobot OVMM 上 fine-tune 和评估

4. **Manipulation Action Head**：manipulation query 激活时，使用独立的 action head 生成 grasp pose（可集成 AnyGrasp 或学习 grasp policy），与 navigation action head 分离。

## Expected outcome

- 在 HomeRobot OVMM 上 overall SR 从 baseline ~10% 提升至 20%+
- 关键 metric：nav-manip transition success rate 显著高于 OK-Robot 式 sequential handoff（预期 +15% 以上）
- Ablation：移除 manipulation query（退化为 MTU3D + heuristic manip）会导致 SR 显著下降，验证三元统一决策的价值

## Risk

1. **Graspability 估计精度**：基于 depth/geometry 的 graspability 可能不够准确，导致 manipulation 成功率低。缓解：用 AnyGrasp 提供 prior。
2. **三类 Query 负迁移**：联合训练可能导致 navigation 和 manipulation 的梯度冲突。缓解：分阶段训练，先冻结 nav weights 再加 manip。
3. **Sim-to-real gap**：Habitat 2.0 物理仿真对 manipulation 的精度有限。缓解：先在 sim 验证架构有效性，real transfer 作为 future work。
