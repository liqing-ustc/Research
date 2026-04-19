---
title: "HomeRobot: Open-Vocabulary Mobile Manipulation"
authors: [Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, Zsolt Kira, Manolis Savva, Angel Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, Chris Paxton]
institutes: [Meta AI, Georgia Tech, CMU, Simon Fraser University]
date_publish: 2023-06
venue: NeurIPS 2023 Competition Track
tags: [mobile-manipulation, scene-understanding, navigation]
paper: https://arxiv.org/abs/2306.11565
website:
github: https://github.com/facebookresearch/home-robot
rating: 1
date_added: "2026-04-02"
---
## Summary
提出 Open-Vocabulary Mobile Manipulation (OVMM) benchmark 和配套的 HomeRobot 软件栈，定义了"在未见环境中抓取任意物体并放置到指定位置"的标准化评估协议，包括基于 Habitat 的仿真环境和 Hello Robot Stretch 真机平台。Heuristic 和 RL 两条 baseline 在真机上仅达到 15-20% 成功率，表明任务远未解决。

## Problem & Motivation
Mobile manipulation 是 embodied AI 的核心能力，但之前的工作普遍存在以下问题：(1) 使用离散动作空间简化问题；(2) 物体类别有限，无法测试 open-vocabulary 泛化；(3) 局限于单房间场景。更关键的是，**各实验室之间缺乏标准化的 benchmark**，导致方法之间无法公平比较。HomeRobot 试图通过统一的仿真 + 真机平台填补这一空白，使 OVMM 成为可复现、可比较的研究问题。

## Method
### 任务定义：OVMM
给定自然语言指令（指定目标物体和目标 receptacle），agent 需要在未见的多房间家庭环境中完成：FindObj → Pick → FindRec → Place 四个阶段。评估使用 partial success metric，逐阶段打分。

### 软件架构：Client-Server
- **Robot 端**：低层控制器 + Hector SLAM
- **桌面工作站**：GPU 驱动的高层感知与规划
- 通过 WiFi 连接，降低机器人端计算需求

### 感知模块
- **DETIC**：open-vocabulary object detector，基于 CLIP embedding 做物体检测与分割
- Depth sensor 提供 3D 信息
- 构建 2D semantic map 用于导航和放置决策

### Heuristic Baseline
- 使用 Fast Marching Method (FMM) 做 motion planning（假设 cylindrical agent）
- Top-down parallel gripper 的 voxelization-based grasping
- 基于 2D semantic map 的 heuristic placement
- 线性 state machine：FindObj → Gaze → Pick → FindRec → Place

### RL Baseline
- 使用 DDPPO（Distributed Proximal Policy Optimization）分别训练四个 skill policy：navigation、gaze、pick、place
- 输入：depth image + semantic segmentation（GT 或 DETIC）+ proprioceptive sensors
- 各 skill 独立训练，通过 state machine 串联

### 仿真环境
- 60 个 HSSD 场景（38 train / 12 val / 10 test），多房间家庭布局
- 2,535 个物体实例，129 个类别（聚合自 AI2-Thor、ABO、Google Scanned Objects、HSSD）
- 21 类 receptacle（桌子、椅子、柜子、沙发、水槽等）
- 动态物体放置 + 预计算 viewpoint + navmesh collision checking

### 真机平台
- Hello Robot Stretch（~$25K，40+ 大学已部署），6-DOF arm + parallel gripper
- 3 房间受控公寓，7 类 receptacle
- 20 episodes/baseline，8 类物体（5 seen + 3 unseen）

## Key Results
### 仿真（GT Segmentation）
| Method | FindObj | Pick | FindRec | Place |
|--------|---------|------|---------|-------|
| RL nav + RL skills | 66.6% | 61.1% | 50.9% | 14.8% |
| Heuristic nav + RL skills | 56.5% | 51.5% | 42.3% | 13.2% |

### 仿真（DETIC Segmentation）——性能断崖式下降
| Method | FindObj | Pick | FindRec | Place |
|--------|---------|------|---------|-------|
| RL nav + RL skills | 21.7% | 10.2% | 6.2% | 0.4% |
| Heuristic nav + RL skills | 29.4% | 13.2% | 5.8% | 0.5% |

### 真机
| Method | FindObj | Pick | FindRec | Overall |
|--------|---------|------|---------|---------|
| RL | 70% | 45% | 30% | **20%** |
| Heuristic | 70% | 35% | 30% | **15%** |

**核心发现**：
1. **感知是最大瓶颈**——从 GT segmentation 切换到 DETIC 后性能暴跌（Place 从 ~14% 降到 <1%）
2. RL 在 gaze alignment 和 placement precision 上优于 heuristic，但 heuristic 对感知噪声更鲁棒
3. 真机 20% 的成功率说明 OVMM 仍然是极具挑战性的开放问题

## Strengths & Weaknesses
### Strengths
- **标准化贡献大**：统一了仿真 + 真机的 OVMM 评估协议，降低了跨实验室比较的门槛
- **Partial success metric 设计合理**：逐阶段评估使 failure analysis 更透明
- **开源生态完整**：代码、仿真环境、真机软件栈全部开源，选用已广泛部署的 Stretch 平台
- **诚实的 baseline**：不 overclaim，明确指出 20% 成功率"promising but insufficient"

### Weaknesses
- **仿真中 grasping 用 magic snap 而非物理模拟**——这使得 sim 结果对 manipulation 能力的评估严重失真，sim-to-real 的 gap 分析因此也不完整
- **Baseline 方法本身缺乏创新**：Heuristic 是经典 FMM + rule-based grasping，RL 是标准 DDPPO——作为 benchmark paper 可以接受，但方法层面贡献有限
- **DETIC 依赖固定类别列表**，并非真正的 open-vocabulary；作者也承认 ConceptFusion 等方法可能更合适
- **真机实验规模极小**（20 episodes/baseline），统计置信度不足
- **不支持自然语言指令**，仅用结构化的 (object, receptacle) pair，与"open-vocabulary"的 framing 有一定落差
- **无 replanning / error recovery 机制**——线性 state machine 一旦某阶段失败即终止

## Mind Map
```mermaid
mindmap
  root((HomeRobot OVMM))
    Benchmark 设计
      仿真: 60 HSSD 场景, 129 类物体
      真机: Hello Robot Stretch
      Metric: Partial success (4 阶段)
    感知
      DETIC open-vocab detector
      GT vs DETIC 差距巨大
      感知是最大瓶颈
    Baseline
      Heuristic
        FMM motion planning
        Rule-based grasping
        对噪声更鲁棒
      RL (DDPPO)
        四个独立 skill policy
        Gaze/Place 更精确
        依赖 GT segmentation
    关键局限
      Magic snap grasping in sim
      无 error recovery
      20 episodes 统计量不足
      非真正 open-vocabulary
    后续工作
      OK-Robot (2024) 达 58.5%
      NeurIPS 2023 Competition
```

## Notes
- 这篇论文的核心价值是 **benchmark contribution** 而非方法创新。它定义了 OVMM 问题并提供了可复现的评估基础设施，但 baseline 本身相当朴素。
- 从 GT segmentation 到 DETIC 的性能暴跌是最有价值的 insight：**在 embodied AI 中，perception 的质量决定了整个 pipeline 的上限**。这与后续 OK-Robot 的经验一致——OK-Robot 通过更好的 semantic memory（VoxelMap + CLIP）显著提升了性能。
- 对比 [[2401-OKRobot]]：OK-Robot 在相似硬件（同为 Stretch）上达到 58.5% 成功率 vs HomeRobot 的 20%，关键差异在于 (1) 更好的 semantic memory (VoxelMap vs 2D semantic map)，(2) 更强的 grasping (AnyGrasp vs heuristic/RL)，(3) 预扫描环境。这说明在 modular pipeline 中，**每个模块的质量都直接影响整体性能**。
- Magic snap grasping 是仿真结果的最大 caveat——manipulation 是 OVMM 中最难的部分之一，仿真中跳过它使得 sim 结果的参考价值大打折扣。
- 作为 NeurIPS 2023 Competition Track 论文，其定位是激发社区参与而非提出 SOTA 方法，从这个角度评价应更宽容。Rating 3/5 反映"重要的问题定义 + 扎实的工程 + 有限的方法贡献"。
