---
title: "HomeRobot: Open-Vocabulary Mobile Manipulation"
authors: [Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, Zsolt Kira, Manolis Savva, Angel Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, Chris Paxton]
institutes: [Georgia Tech, FAIR Meta AI, Carnegie Mellon, Simon Fraser]
date_publish: 2023-06
venue: CoRL 2023
tags: [mobile-manipulation, navigation, manipulation, scene-understanding]
paper: https://arxiv.org/abs/2306.11565
website: https://ovmm.github.io/
github: https://github.com/facebookresearch/home-robot
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] HomeRobot: Open-Vocabulary Mobile Manipulation
> - **核心**: 提出 Open-Vocabulary Mobile Manipulation (OVMM) 任务和首个 sim+real 双轨基准——给定语言指令 "把 object 从 start_receptacle 移到 goal_receptacle"，机器人需在未知家庭环境中找物、抓取、找目标家具、放置
> - **方法**: HSSD 200 场景 + 2535 跨数据集 objects (AI2-Thor / ABO / GSO / HSSD) 构建 sim 数据；提供 Hello Robot Stretch 真实软硬件栈；OVMMAgent state machine 串联 FindObj → Gaze → Pick → FindRec → Place 五个 skill，每个 skill 提供 heuristic 和 RL (DDPPO) 两套 baseline；DETIC 提供 open-vocab 分割
> - **结果**: 真机 20% overall success（RL nav）/15%（heuristic）；sim 中用 GT segmentation 最高 48% partial / 14.8% overall，但换 DETIC 后骤降至 ~10-12% partial——感知是核心瓶颈
> - **Sources**: [paper](https://arxiv.org/abs/2306.11565) | [website](https://ovmm.github.io/) | [github](https://github.com/facebookresearch/home-robot)
> - **Rating**: 2 - Frontier（OVMM task + HomeRobot stack 被 NeurIPS 2023 / CVPR 2024 OVMM Challenge 复用为 de facto 基准，但 method 贡献薄，benchmark 地位稳固而方法未成 building block）

**Key Takeaways:**
1. **OVMM 把碎片化的 mobile manipulation 拉到一个整合 setting**：感知 + 语言 + 导航 + 操作必须串成端到端 pipeline，单独 SOTA 化某个组件意义不大。
2. **Sim2Real 双轨基准**：与多数纯 sim 工作不同，配套 real-world 协议 + 标准硬件 (Stretch) + 软件栈，强调跨实验室复现。
3. **感知是首要瓶颈**：从 GT segmentation 切到 DETIC 之后，所有 baseline 性能腰斩到 1/4——open-vocab perception 比 policy 设计更值得投入。
4. **Heuristic ≈ RL**：在 sim 中 RL 略胜，但都远未饱和；提示当前 mobile manipulation 的整体瓶颈不在 policy class 选择，而在感知与多 skill 衔接。

**Teaser. OVMM 任务可视化——在未知 3 房间公寓中执行 "move stuffed animal from chair to sofa" 等指令。**

<video src="https://ovmm.github.io/static/videos/teaser.mp4" controls muted playsinline width="720"></video>

---

## OVMM 任务定义

任务格式：**"Move (object) from the (start_receptacle) to the (goal_receptacle)"**。

- **object**：小型可操作物体（cup、stuffed toy、box 等）
- **start_receptacle / goal_receptacle**：大型家具表面（table、sofa、counter 等）
- 机器人在未知单层 home 环境中初始化，仅给出三个语义类别名；object 已知挂在 start_receptacle 上；任意一个 valid goal_receptacle 都可作为放置点

**Open-vocabulary**：测试时 object 来自训练**未见过**的 instance，部分来自训练**未见过**的 category；receptacle 类别全部见过但具体实例未见过。

**Scoring**：分四阶段——FindObj → Pick → FindRec → Place，全部成功才算 episode 成功；partial success = 完成的阶段数 / 4。

**Figure 1. OVMM 任务在 sim 和 real 中的对应实例。**
![](https://arxiv.org/html/2306.11565v2/x1.png)

## 与已有 benchmark 的对比

OVMM 同时具备 (a) 200 场景规模、(b) 7892 物体实例、(c) 连续动作空间、(d) sim2real 验证、(e) 完整 robotics stack——这是同时具备所有 5 项的唯一 benchmark。

**Table 1. OVMM 与 Room Rearrangement / Habitat ObjectNav / TDW-Transport / VirtualHome / ALFRED / Habitat 2.0 HAB / ProcTHOR / RoboTHOR / Behavior-1K / ManiSkill-2 的横向对比。**

| Benchmark | Scenes | Cats | Insts | Continuous | Sim2Real | Stack | Open License | Manip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Room Rearrangement | 120 | 118 | 118 | ✖ | ✖ | ✖ | ✔ | ✖ |
| Habitat ObjectNav | 216 | 6 | 7,599 | ✔ | ✖ | ✖ | ✔ | ✖ |
| ALFRED | 120 | 84 | 84 | ✖ | ✖ | ✖ | ✔ | ✓ |
| Habitat 2.0 HAB | 105 | 20 | 20 | ✔ | ✖ | ✖ | ✔ | ✔ |
| ProcTHOR | 10000 | 108 | 1633 | ✖ | ✖ | ✖ | ✔ | ✔ |
| RoboTHOR | 75 | 43 | 731 | ✖ | ✔ | ✖ | ✔ | ✖ |
| [[2403-Behavior1K\|Behavior-1K]] | 50 | 1265 | 5215 | ✔ | ✔ | ✖ | ✖ | ✓ |
| ManiSkill-2 | 1 | 2000 | 2000 | ✔ | ✓ | ✖ | ✓ | ✔ |
| **OVMM + HomeRobot** | **200** | **150** | **7892** | **✔** | **✔** | **✔** | **✔** | **✔** |

> ❓ "Continuous Actions" 列：HomeRobot 的 "continuous" 实际是 teleporting agent 通过 navmesh 检查移动到 waypoint，并非真正物理连续——见 §D.6。这种近似是否影响 sim2real 还需要更多消融。

## 数据与硬件

### 仿真数据集

基于 **Habitat Synthetic Scenes Dataset (HSSD)** 构建——200+ 人工创作的 3D 家居场景，含 18k 物体模型；筛选出 60 个支持 rearrangement 的场景，按 38/12/10 划分 train/val/test。

**物体来源**：跨数据集聚合 **AI2-Thor + Amazon-Berkeley Objects + Google Scanned Objects + HSSD**，共 2535 个 instance / 129 个 category。

**Table 2. 物体跨 (Seen/Unseen) Category × (Seen/Unseen) Instance 的划分。**

|  | SC,SI | SC,UI | UC,UI | Total |
| --- | --- | --- | --- | --- |
| Cats | 85 | 64 | 44 | 129 |
| Insts | 1363 | 748 | 424 | 2535 |

训练集只用 SC+SI；val/test 用 SC+UI 和 UC+UI——强制 open-vocabulary 评估。

### 真机平台

**Hello Robot Stretch + DexWrist**：$25K 价位、6-DoF 操作、人体尺寸、低风险。Sec H.1 解释选 Stretch 是因为已在 40+ 实验室部署，跨实验室复现有基础。

**Figure 4. HomeRobot 软件栈架构——Stretch 跑底层控制 + SLAM，工作站 GPU 跑感知 + 规划，路由器连接。**
![](https://arxiv.org/html/2306.11565v2/x3.png)

代码分三个 repo：`home_robot`（共享组件）/ `home_robot_sim`（Habitat 仿真）/ `home_robot_hw`（硬件 server + 工作站 client）。

### 真实测试环境

3 房间受控公寓，固定家具布局（sofa、kitchen table、counter with bar、TV stand），8 类目标物体——5 类 seen（Cup, Bowl, Stuffed Toy, Medicine Bottle, Toy Animal）+ 3 类 unseen（Rubik's cube, Toy Drill, Lemon），7 类 receptacle（Cabinet, Chair, Couch, Counter, Sink, Stool, Table）。每个 baseline 跑 20 个 episode。

## OVMMAgent: 五技能 state machine

**Figure 2. 真机和仿真中 OVMMAgent 任务执行示意。**
![](https://arxiv.org/html/2306.11565v2/x2.png)

OVMMAgent 是状态机，按固定顺序调用 5 个 skill：

1. **FindObj**：导航到 start_receptacle 并定位 object
2. **Gaze**：靠近物体并调整朝向以利于抓取
3. **Pick**：抓取（sim 中 magic snap，real 中真实抓取）
4. **FindRec**：导航到 goal_receptacle
5. **Place**：放置物体

**两套 baseline**：

- **Heuristic**：基于 Gervet 2022 的语义地图 + Frontier Exploration + Fast Marching Method 路径规划；Pick/Place 是体素化 + 简单几何规则；DETIC 提供 open-vocab 分割
- **RL**：用 DDPPO 训练 4 个 skill，输入 depth + 真值语义分割 + proprioception；测试时换成 DETIC 分割

### Heuristic Pick 策略

**Figure 15. 简单 grasp 策略在多种 lab 环境的测试。**
![](https://arxiv.org/html/2306.11565v2/x8.png)

实现细节（§E.3）：

1. 把 object 点云体素化为 0.5 cm voxels
2. 选 Z 最高的 top 10% voxels
3. 投影到 2D 网格，对每个 voxel 评估三区域（夹爪两指 + 指间空间）
4. 评分：指间区域占据 + 两侧自由
5. 平滑 + 阈值过滤

> 作者对比了 ContactGraspnet / 6-DoF GraspNet / GraspNet 1-Billion，发现这些预训练模型在 Stretch 的 sensor noise 下反而更不稳定——这是个有意思的负面结果，提示 SOTA grasp net 的 generalization 比 paper claim 弱。

### Heuristic Place 策略

**Figure 16. 真机中 heuristic place 把 stuffed animal 放到 sofa 上的实例。**
![](https://arxiv.org/html/2306.11565v2/x9.png)

DETIC 检测 receptacle → 投影到点云 → 50 个采样点选 "最大平台" → 旋转对齐 → 必要时前进 → 重新估计 → 设置 arm 位姿 → 释放。

### 导航 Planner

基于 Gervet 2022 的语义地图 + Frontier-based exploration + Fast Marching Method (FMM)，做了关键扩展：把"找小 object 难"的问题转化为"先找 start_receptacle"——

1. 如果 object 和 start_receptacle 在同一 cell 共现 → 规划到 object
2. 如果只看到 start_receptacle（排除 1m 内已访问区域）→ 规划到 start_receptacle
3. 否则 → 规划到最近 frontier

**语义地图**：$K \times M \times M$ 二值矩阵，$K = C + 4$，每 cell 5cm × 5cm 物理空间，C 个语义类别 + 障碍物 + 已探索 + 当前位置 + 历史位置。

## 实验结果

### Sim 结果

**Table 3. 不同 (perception, nav, gaze, place) 组合下的 partial 和 overall success rate (%)。**

| Perception | Nav | Gaze | Place | FindObj | Pick | FindRec | Overall | Avg Partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT | Heuristic | None | Heuristic | 54.1 | 48.5 | 31.5 | 5.1 | 34.8 |
| GT | Heuristic | RL | RL | 56.5 | 51.5 | 42.3 | 13.2 | 40.9 |
| GT | RL | None | Heuristic | 65.4 | 54.8 | 43.7 | 7.3 | 42.8 |
| **GT** | **RL** | **RL** | **RL** | **66.6** | **61.1** | **50.9** | **14.8** | **48.3** |
| DETIC | Heuristic | None | Heuristic | 28.7 | 15.2 | 5.3 | 0.4 | 12.4 |
| DETIC | Heuristic | RL | RL | 29.4 | 13.2 | 5.8 | 0.5 | 12.2 |
| DETIC | RL | None | Heuristic | 21.9 | 11.5 | 6.0 | 0.6 | 10.0 |
| DETIC | RL | RL | RL | 21.7 | 10.2 | 6.2 | 0.4 | 9.6 |

**关键发现**：

1. **GT → DETIC 性能崩塌**：最佳组合从 48.3 → 12.4 partial avg，overall success 从 14.8% → ~0.5%。Open-vocab 感知是头号瓶颈。
2. **RL > Heuristic（仅在 GT 下）**：RL 各 skill 略胜 heuristic；但用 DETIC 时 heuristic FindObj **反超** RL——作者归因 heuristic 通过 2D 语义地图能"积累 + 平滑"噪声预测，对小物体误检更鲁棒。这是个重要 insight：**model-based 方法在 noisy perception 下天然有 ensemble 效应**。
3. **Gaze 帮助 Pick**：除了 (Heuristic nav, DETIC) 组合，加 Gaze 都提升 pick 成功。

### Real 结果

**Table 4. 真机 OVMM 成功率 (%)，每 baseline 20 episodes。**

| Method | FindObj | Pick | FindRec | Overall |
| --- | --- | --- | --- | --- |
| Heuristic Only | 70 | 35 | 30 | 15 |
| **RL Only** | **70** | **45** | **30** | **20** |

RL 比 Heuristic 多成功 1 个 episode；差距主要来自 Pick——RL Gaze skill 让对齐更准；以及 RL place 更精准让物体落得更稳。

**真机示例视频**——拣起未见 stuffed animal 放到未见 sofa：

<video src="https://ovmm.github.io/static/videos/ovmm_real_world_success_1_edited.mp4" controls muted playsinline width="720"></video>

> ❓ 真机只跑 20 个 episode，统计噪声很大（1 个 episode = 5%），20% vs 15% 的差距能否算显著？需要更多 trial 才能下结论。

## Appendix 中的工程细节亮点

- **改进 Habitat 渲染**：重写 PBR 着色器、加 HBAO，仅 3% FPS 损失（340 → 330），但 visual quality 显著提升（§D.5）。
- **Magic snap 替换**：尝试用更严格的 grasp success（要求 arm 到达且不碰撞）——arm 能到达 79%，但不碰撞只有 47%。揭示 Habitat 中"无视碰撞"的 magic snap 大幅高估了真实 grasp 难度。
- **Discrete vs Continuous action**：作者实现了两套 nav action space，continuous 通过 navmesh 检查 + teleport 实现；这种 "fake continuous" 在工程上简化但牺牲了真实物理。

---

## 关联工作

### 基础设施 / 平台

- **AI Habitat / Habitat 2.0**：仿真器基础，OVMM 在其上构建
- **HSSD**：场景资产源
- **DDPPO**：RL baseline 的训练算法
- **Hello Robot Stretch**：标准化硬件平台

### 子任务方法借鉴

- **DETIC**：open-vocab 检测 / 分割，所有 baseline 的感知组件
- **Gervet et al. 2022 (Frontier-based ObjectNav)**：heuristic nav 的直接前作
- **Hector SLAM**：真机定位
- **Fast Marching Method**：路径规划

### 同时期 mobile manipulation

- [[2305-TidyBot|TidyBot]]：同期 LLM-based 个性化整理机器人，也用 heuristic grasp / 真机 Stretch 测试，但侧重 LLM-driven 偏好学习而非 open-vocab 物体
- [[2403-Behavior1K|BEHAVIOR-1K]]：另一个大规模 sim 基准，1265 类物体但无标准 robotics stack

### 对比 benchmark

- ALFRED / VirtualHome / TDW-Transport：早期室内任务基准，object 集合小、动作离散
- Habitat 2.0 HAB / ProcTHOR / RoboTHOR / ManiSkill-2：覆盖 nav 或 manip 单方面，无 OVMM 这种端到端 mobile manipulation 整合

---

## 论文点评

### Strengths

1. **问题定义有价值**：OVMM 真正把 mobile manipulation 的 4 个核心子问题（perception / language / navigation / manipulation）整合到同一 episode，避免了"每个 sub-skill 单独 SOTA 但端到端拉胯"的领域困境。
2. **基础设施贡献远超论文方法本身**：HSSD 60 场景 + 2535 物体跨 4 数据集聚合 + Stretch 端到端软件栈 + 仿真/真机对齐 API——这些 infra 后续被 NeurIPS 2023/CVPR 2024 challenge 复用，成为社区标准。
3. **诚实的 negative result**：明确报告 GT → DETIC 性能崩塌、heuristic 在 noisy perception 下反超 RL、SOTA grasp net 不如 simple voxel heuristic——这些 anti-conventional 发现本身就是 contribution。
4. **跨实验室复现性**：Stretch 已在 40+ 实验室部署，加上完整 stack，复现门槛远低于自定义机器人 benchmark（如 Behavior-1K、ManiSkill-2）。

### Weaknesses

1. **真机样本量过小**：20 episodes per baseline，统计噪声大。20% vs 15% 的差距能否归因于 method 而非随机性，没有给出 confidence interval。
2. **Continuous action 是 teleport hack**：navmesh check + 直接传送，并非真物理仿真——这种近似如何影响 sim2real 没有消融。
3. **任务格式偏简单**：固定模板 "move A from B to C"，未涉及多步指令、自然语言模糊性、长程依赖。Conclusion 也承认这是 future work。
4. **End-to-end baseline 缺失**：只对比模块化 heuristic vs 模块化 RL；没有 end-to-end VLA 或 LLM-based agent 对比，使得"模块化 vs 端到端"的核心设计选择无法判断。
5. **Pick 在 sim 中是 magic snap**：与真机 grasp 严重脱节，sim 中 Pick partial SR (50-60%) 几乎完全不能预测真机 Pick (35-45%)。

### 可信评估

#### Artifact 可获取性

- **代码**: inference + training 完整开源（github.com/facebookresearch/home-robot，MIT License）
- **模型权重**: RL baseline 训练脚本开源；具体 checkpoint 与论文对应关系在 README 未明确
- **训练细节**: 高层描述（DDPPO、4 个 skill 分别训练、depth + GT seg + proprio 输入），但具体超参 / 训练步数 / 数据配比需要查 config 文件
- **数据集**: HSSD 子集 + 物体合集均开源；真实公寓 layout 通过文档描述但无 3D 扫描共享

#### Claim 可验证性

- ✅ **Real 20% / 15% 成功率**：附带 challenge 复现协议，社区后续 challenge（NeurIPS 2023 / CVPR 2024）可独立验证
- ✅ **GT → DETIC 性能崩塌**：Table 3 数据完整，可复现
- ✅ **Heuristic place 不如 RL place（sim）**：Table 3 RL place +9.7 partial 相对 heuristic
- ⚠️ **"baselines achieve 20% success in real world"**：仅 20 episodes，置信区间可能 ±10%；overclaim 风险有限但应该报告 CI
- ⚠️ **"sim2real transfer demonstrated"**：sim 最高 14.8% overall，real 最高 20%——数字接近不代表 transfer 工作良好；可能两者都低到 noise floor
- ⚠️ **"heuristic grasp 与 SOTA grasp net comparable"**：未给定量对比表，仅文字描述

### Notes

- **核心洞见**：感知（DETIC 级别的 open-vocab segmentation）才是 mobile manipulation 当前真瓶颈。改 RL 算法 / 调 reward 边际收益远不如换更好的 perception backbone。这与后续 VLA（如 RT-2、π0）把 vision-language-action 端到端训练的方向一致。
- **方法论启发**：在 noisy perception 下 model-based + 显式语义地图反而比 RL 鲁棒——因为地图自然提供时序累积平滑。这对 "end-to-end vs modular" 的争论是个 nuanced 数据点。
- **基础设施投资 vs 方法新颖性**：本文 method contribution 不大（heuristic + DDPPO），但 infra contribution 让它成为 NeurIPS 2023 + CVPR 2024 challenge 的基准。这是个好的 "build the road, others run on it" 案例。
- **Follow-up**：NeurIPS 2023 OVMM Challenge 的总结论文（arxiv 2407.06939）值得读，看社区一年内 SOTA 推进到什么程度，验证哪些 component 的改进真正提升了 overall success。

### Rating

**分数**：2 - Frontier

**理由**：OVMM task formulation 和 HomeRobot stack 被 NeurIPS 2023 / CVPR 2024 Challenge 复用为 de facto 基准，具备较强的社区采纳信号，这是 frontier 档的核心证据。但本身 method 贡献较薄（heuristic + DDPPO，见 Weaknesses #4 缺 end-to-end baseline），且随后续 VLA / end-to-end 路线兴起，它作为 benchmark 的地位稳固但作为方法并未上升为 building block——因此留在 2 而非升到 3。
