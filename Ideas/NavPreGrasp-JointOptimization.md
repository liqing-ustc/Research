---
title: "NavPreGrasp: Navigation-Aware Manipulation Pre-Positioning"
tags: [mobile-manipulation, VLA, VLN, manipulation, navigation]
status: raw
linked_project:
date_updated: "2026-03-31"
---
## Hypothesis

若在 navigation 阶段就预测最优的 manipulation approach pose（机器人到达目标物体时的最终位姿），并将其作为 navigation 的 goal constraint，则 mobile manipulation 的 grasp success rate 将显著高于先导航到目标附近再调整位姿的 sequential 策略，因为 approach pose 同时影响 gripper reachability、视野质量和障碍物避让。

## Motivation

1. **解决的知识空白**：VLN-VLA-Unification survey Section 5 明确指出"Nav→Manip handoff 是核心瓶颈"——modular 系统中 robot 停在 navigation 终点后开始 manipulation，但该位置往往不是 manipulation 最优位置。OK-Robot 用 3 个 heuristic scoring function 缓解但未根本解决。
2. **成功的影响**：为所有 mobile manipulation 系统提供一个即插即用的模块——从 OK-Robot 到 π0.5 都可受益。Nav-manip transition 是失败率最高的环节之一。
3. **时机成熟**：AnyGrasp 等 grasp planning 工具可提供 graspability 先验；VLM 的 spatial reasoning 能力（SpatialVLM, CVPR 2024）可用于从图像预测 approach direction。

## Related Work

- [[Papers/2401-OKRobot]] - 使用 3 个 scoring function（reach target + gripper clearance + obstacle avoidance）选择 navigation 终点，但是 heuristic 的，不考虑具体 grasp pose。NavPreGrasp 将这一过程学习化，并联合优化 navigation goal 和 grasp approach。
- [[Papers/2504-Pi05]] - π0.5 的 hierarchical inference 在 high-level 规划时不考虑 low-level manipulation 的空间约束，可能导致 suboptimal 的 approach。
- [[Papers/2401-SpatialVLM]] - 暂无相关笔记，建议先 paper-digest SpatialVLM（CVPR 2024）。VLM spatial reasoning 可用于从 egocentric 视角预测 approach direction。
- [[Papers/2401-MobileALOHA]] - End-to-end whole-body policy 隐式学习了 approach pose，但无法泛化到新环境和新物体。

**Novelty**: 4/5 — closest works: [[Papers/2401-OKRobot]]（heuristic scoring, 非 learned joint optimization）, [[Papers/2401-MobileALOHA]]（implicit in end-to-end policy, 不可迁移）。无已有工作将 approach pose prediction 显式建模为 navigation goal constraint。

## Approach sketch

1. **Approach Pose Predictor (APP)**：给定 (a) 目标物体的 3D position + semantic info（来自 scene graph 或 VoxelMap），(b) 周围环境的 occupancy + obstacle info，(c) robot 的 kinematic constraints（arm reach, gripper workspace），预测最优 approach pose（base position + orientation）。APP 可以是一个轻量 MLP，也可以是 VLM 的一个 auxiliary prediction head。

2. **Navigation Goal Integration**：APP 输出的 approach pose 作为 navigation 的 goal（替代简单的"导航到物体附近"），navigation planner 规划到该 pose 的路径。如果 approach pose 不可达（被障碍物阻挡），APP 输出 top-K 候选 pose，planner 选择可达且最优的。

3. **训练数据生成**：在 Habitat 2.0 中，对每个 pick-and-place 任务：
   - 枚举 robot base 在目标物体周围的 N 个候选 pose（不同距离 × 不同角度）
   - 对每个 pose 执行 grasp 尝试，记录成功/失败
   - 成功率最高的 pose 作为 ground truth approach pose
   - 这种自动化数据收集可以大规模生成训练数据

4. **评估**：在 HomeRobot OVMM 上，对比 (a) NavPreGrasp（learned approach pose），(b) OK-Robot scoring（heuristic approach），(c) random approach（navigate to nearest free space）。

## Expected outcome

- Grasp success rate（given successful navigation）从 baseline ~70% 提升至 85%+
- Overall pick-and-place SR 提升 10-15%（因为 nav-manip transition 是主要失败点之一）
- 定性分析：NavPreGrasp 学会从桌子侧面而非正面接近（避免手臂碰到桌面边缘），从物体的 graspable 方向接近等

## Risk

1. **Approach pose 的 task-dependency**：不同 manipulation 任务（pick vs pour vs push）需要不同 approach pose，单一 predictor 可能无法覆盖。缓解：先限制 pick-and-place，再扩展。
2. **环境遮挡**：在导航阶段可能看不到目标物体周围的完整环境，导致 approach pose 预测不准。缓解：允许 APP 在接近过程中更新预测（iterative refinement）。
3. **Kinematic model dependency**：APP 需要知道 robot 的 arm reach 和 workspace，不同 robot 需要重新训练。缓解：将 kinematic constraints 作为 input 参数化。
