---
title: Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation
authors: [Zhijie Yan, Shufei Li, Zuoxu Wang, Lixiu Wu, Han Wang, Jun Zhu, Lijiang Chen, Jihong Liu]
institutes: [Beihang University, City University of Hong Kong, Minzu University of China, Afanti Tech LLC]
date_publish: 2024-10-16
venue: RA-L 2025
tags: [mobile-manipulation, scene-understanding, semantic-map, task-planning]
paper: https://arxiv.org/abs/2410.11989
website: https://bjhyzj.github.io/dovsg-web/
github: https://github.com/BJHYZJ/DovSG
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] DovSG: Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation
> - **核心**: 用 open-vocab 3D scene graph 作为 mobile manipulation 的持久 memory，关键在 **local update**——交互后只更新被影响的 voxel/sub-graph，避免重建整个场景
> - **方法**: RGB-D 扫描 → DROID-SLAM 估姿 → RAM+GroundingDINO+SAM2+CLIP 做 open-vocab 物体检测与特征提取 → 构建带 "on/belong/inside" 关系的 3D scene graph → ACE+LightGlue+ICP 多阶段重定位 → 投影深度/颜色对比剔除过时 voxel → GPT-4o 任务分解 → AnyGrasp / 启发式抓取
> - **结果**: 4 个真实房间 ×3 种 modification ×80 trials；long-term 任务成功率 35% vs Ok-Robot 5%；memory 比 Ok-Robot 小 13×，更新比 ConceptGraphs 快 27×
> - **Sources**: [paper](https://arxiv.org/abs/2410.11989) | [website](https://bjhyzj.github.io/dovsg-web/) | [github](https://github.com/BJHYZJ/DovSG)
> - **Rating**: 2 - Frontier（dynamic mobile manipulation 下首个端到端可跑通的 local-update scene graph 方案，RA-L 2025，但方法组合性强、依赖 off-the-shelf components，非奠基）

**Key Takeaways:**
1. **Local sub-graph update 是 dynamic 场景下的关键工程点**：先用 RGB-D 投影筛出 affected voxels → 找出受影响 object → 顺藤摸瓜把它们的 parent/child 也拉进 affected set → 只对这部分重算关系。绕开了 ConceptGraphs / HOV-SG / Ok-Robot 必须全场景重建的代价。
2. **Relocalization pipeline 是 ACE → LightGlue → colored ICP 的三段精化**：ACE 给粗 pose，LightGlue 找最相似历史帧，colored ICP 做几何+光度精配准。这是 local update 能做对的前提——pose 不准则 voxel 删错。
3. **Scene graph 关系类型只取 "on/belong/inside" 三种**（继承自 RoboEXP），且利用对齐到 floor 的坐标系直接从几何推导，没用 GNN/relation predictor。简单但够用。
4. **Long-term task 5% → 35%**：绝对值仍然低，反映 mobile manipulation 整体仍未到产品级；但对比 Ok-Robot 在 Appearance / Positional Shift 下 0/80 的失败，dynamic adaptation 价值明确。

**Teaser. DovSG 系统总览。** 五大模块（perception / memory / task planning / navigation / manipulation），memory 同时维护 low-level semantic memory 与 high-level scene graph，二者随交互持续更新；当 keys 从 cabinet 被人移到 table，系统能检测变化并更新 graph 以正确执行 Task 2-2。

![](https://arxiv.org/html/2410.11989v6/x1.png)

<video src="https://bjhyzj.github.io/media/videos/dovsg_show_method.mp4" controls muted playsinline width="720"></video>

---

## Background

传统 3D scene representation 走两条路：用 foundation model 直接构 3D 表示（LERF、Distilled Feature Fields 等），或者把 2D VLM 投影到 3D 点云（OpenScene、ConceptGraphs、HOV-SG）。两类方法的共同缺陷：

- **Per-point dense feature 太重**——内存爆炸且不可分解
- **静态假设**——一旦人或机器人改动了场景，旧 memory 就过时

3D scene graph（ConceptGraphs / HOV-SG）通过把 object 抽象为 node、关系抽象为 edge，把 dense feature 压缩到 object 级别，但**仍假设静态**。RoboEXP 引入 action-conditioned 更新，但聚焦 tabletop 探索。DovSG 把这套思路搬到 mobile manipulation，重点解决 dynamic environment 下的 incremental local update。

---

## Method

### 系统结构

DovSG 由五个模块构成：perception、memory、task planning、navigation、manipulation。Memory 是核心，分两层：
- **Low-level semantic memory**：voxelized point cloud + per-object CLIP feature
- **High-level scene graph** $\mathcal{G}_t = \langle \mathbf{O}_t, \mathbf{E}_t \rangle$：node 是 3D object，edge 是 on/belong/inside 关系

### Stage 1：Home Scanning + 坐标对齐

用 Intel RealSense D455 录制 RGB-D 序列；DROID-SLAM 估计相机 pose（用真实深度替换其深度预测以保持真实尺度）。

为了让坐标系对下游 spatial reasoning 友好（z 轴朝上），用 Grounding DINO + SAM2 分割 floor mask → 投影到 3D → RANSAC 拟合平面 → 计算变换 $T^{\text{floor}}$ 把地面对齐到 $z=0$ 平面。每帧 pose 变换为 $\mathcal{P} = \mathcal{R}_x T^{\text{floor}} \mathcal{P}^{\text{droid}}$。

### Stage 2：Open-vocabulary 3D Object Mapping

每帧 $I_t$ 的 pipeline：

1. **Recognize-Anything (RAM)** 给出可能的 object class 列表 $\{c_{t,i}\}$
2. **Grounding DINO** 用这些 class 做 open-vocab detection，得到 bbox $\{b_{t,i}\}$
3. **SAM2** 把 bbox 精化为 mask $\{m_{t,i}\}$
4. 对每个 mask 的 cropped image 与 isolated mask image 分别用 **CLIP** 提特征，按 HOV-SG 的 weighted sum 融合成 $f^{\text{rgb}}_{t,i}$；text 特征 $f^{\text{text}}_{t,i}$ 来自 class name 的 CLIP embedding
5. 用 depth 投影到 3D，DBSCAN 去噪（$\varepsilon$ 由 k-NN 距离自适应），转到 map frame 得 $pcd_{t,i}$

**Object association**（贪心匹配到现有 object）综合三类相似度：

$$
s(i,j) = \omega_v \cdot s_{\text{vis}}(i,j) + \omega_g \cdot s_{\text{geo}}(i,j) + \omega_t \cdot s_{\text{text}}(i,j)
$$

其中 $s_{\text{geo}}$ 是 nearest-neighbor 命中率（$pcd_{t,i}$ 中点在 $pcd_{o_j}$ 距离阈值 $\delta_{\text{nn}}$ 内的比例），$s_{\text{vis}}/s_{\text{text}}$ 是 cosine similarity 归一到 [0,1]。匹配上则增量更新 feature（按 detection 计数加权平均）和点云（合并后下采样）；不匹配则建新 object。

### Stage 3：Scene Graph Generation

把 object 点云 voxelize（节省存储 + 利于后续局部更新）。利用已对齐的坐标系（z 朝上），直接从几何推导三类关系：

- **on**：堆叠/位置层级，如 apple on table
- **belong**：归属/附着，如 handle belong refrigerator
- **inside**：包含，仅限小尺度容器，如 keys inside drawer

**Figure 1. Initialization & Construction of 3D Scene Graphs.** DROID-SLAM 估姿 → open-vocab 分割并投影到 3D → 多视角融合得 3D objects → 几何推关系生成 edges。

![](https://arxiv.org/html/2410.11989v6/x2.png)

### Stage 4：Dynamic Scene Adaptation（核心创新）

整个流程分四步：

#### 4.1 Relocalization & Refinement

机器人执行任务后到新位置采集新 RGB-D 观测 $\{I_k\}$，需要在 prior map 中精确定位。三段式：

1. **ACE**（pre-trained 在 mapping 阶段的 $\langle I^{\text{rgb}}_i, I^{\text{pose}}_i \rangle$ 上）给出 coarse pose
2. **LightGlue** 在历史帧中找特征匹配最多的 $I^{\text{rgb}}_{\hat k}$，把 pose anchor 到最相似视角
3. **Multi-scale colored ICP** 用 RGB-D 做几何+光度精配准，得 $T^{\text{icp}}$，更新 $I^{\text{pose}}_k \leftarrow T^{\text{icp}} I^{\text{pose}}_k$

#### 4.2 Remove Obsolete Voxels

把 prior map 中所有 voxel 投影到当前相机坐标，对比新 RGB-D：

$$
\Delta z_i = \big| I^{\text{depth}}_k[u_j^i, v_j^i] - z_j^i \big|, \quad \Delta c_i = \big| I^{\text{rgb}}_k[u_j^i, v_j^i] - c_j^i \big|
$$

删除规则：

$$
\text{delete if} \begin{cases} \Delta z_i > \delta_z, \\ \Delta z_i > \delta_z' \text{ and } \Delta c_i > \delta_c \end{cases}
$$

即"深度差很大"或"深度差中等且颜色差也大"则删——双阈值避免噪声误删。

#### 4.3 Update Low-level Memory

对剩余的新观测 $I_k$ 重跑 Stage 2 pipeline，融合到 $\mathbf{O}_t$ 得 $\mathbf{O}_{t+1}$；新观测帧也加入图像序列，供下次 relocalization 用。

#### 4.4 Update High-level Memory（local sub-graph）

避免全图重算关系：

1. 比对 $\mathbf{O}_t$ vs $\mathbf{O}_{t+1}$ 找出 changed/deleted 的 $\mathbf{O}_{\text{affected}}$
2. 沿 edge 找 affected object 的 parent / sibling，扩入 $\mathbf{O}_{\text{affected}}$，删除所有相关 edge
3. 在 $\mathbf{O}_{t+1}$ 中找仍存在的 + 新增 object → $\mathbf{O}_{\text{need\_process}}$
4. 仅对 $\mathbf{O}_{\text{need\_process}}$ 重算关系，更新 $\mathbf{E}_{t+1}$

**Figure 2. Adaptation in interactions with manually modified scenes.** ACE-trained MLP 给 coarse pose → LightGlue + ICP 精化 → 新视角点云与存储 pose 对齐 → 局部更新 scene graph。

![](https://arxiv.org/html/2410.11989v6/x3.png)

### Stage 5：Task Planning + Navigation + Manipulation

- **Task planning**：GPT-4o 把自然语言长任务分解为 (action_name, object_name) 的子任务列表
- **Localization**：单 object 时取 CLIP cosine top-1；带空间关系（"A on B"）时取 top-k A × top-k B 中 Euclidean 距离最近的对
- **Mobile control**：A* 规划 + PID 控制
- **Pick**：以 target bbox 周围 cropped 点云喂给 **AnyGrasp**，按 translation/rotation cost 过滤选最高置信抓取；如果 AnyGrasp 失败，启用 heuristic strategy（基于 bbox 旋转 gripper 对齐主轴）
- **Place**：用 SAM2+GroundingDINO 分割 target 平面，取中位 $(x_m, y_m)$，drop height $z_{\max} = 0.1 + \max\{z \mid 0 \le x \le x_m, |y - y_m| < 0.1\}$

**Figure 3. Two grasp strategies.** 上：cropped 点云送 AnyGrasp + cost filtering；下：bbox-based heuristic 抓取，作为 AnyGrasp 失败时的兜底。

![](https://arxiv.org/html/2410.11989v6/x4.png)

---

## Experiments

### Setup

- 硬件：UFACTORY xARM6 + Agilex Ranger Mini 3 + RealSense D455
- 4 个真实房间，每个房间各 20 trials × 3 种修改类型 × 2 种 task 长度 = 80 trials per modification per method
- **三种 modification level**：
  - **Minor Adjustment**：轻微移动，原视野内仍可见
  - **Appearance**：原本隐藏的 object 出现（如开抽屉）
  - **Positional Shift**：大范围移动，原位置已不可见

### 主结果

**Table I. Scene Change Detection Accuracy (SCDA) & Scene Graph Accuracy (SGA).** Baseline 是 GPT-4o + CoT（仿照 RoboEXP 的方式：把新观测和最相似 memory 图对比）。

| Task             | Minor Adjustment GPT-4o / Ours | Appearance GPT-4o / Ours | Positional Shift GPT-4o / Ours |
| ---------------- | ------------------------------ | ------------------------ | ------------------------------ |
| SCDA             | 41.44% / **95.37%**            | 64.25% / **93.22%**      | 66.35% / **94.23%**            |
| SGA              | 54.60% / **88.75%**            | 52.25% / **84.86%**      | 46.18% / **83.72%**            |

GPT-4o 在 Minor Adjustment 上特别差（41%）——小位移不容易在 RGB 对比中识别；DovSG 因为有 voxel-level 精确重定位，能定位到具体哪个 voxel 改了。

**Table II. Long-term Tasks & Subtasks Success Rate.** 与 Ok-Robot（同样 mobile manipulation，但假设静态）对比。

| Task        | Method   | Minor       | Appearance  | Positional  | Total      |
| ----------- | -------- | ----------- | ----------- | ----------- | ---------- |
| Pick up     | Ok-Robot | 84/110      | 61/92       | 59/87       | 70.58%     |
| Pick up     | Ours     | 111/137     | 111/136     | 108/133     | **81.28%** |
| Place       | Ok-Robot | 53/64       | 40/51       | 42/51       | 81.32%     |
| Place       | Ours     | 80/93       | 83/95       | 74/85       | **86.81%** |
| Navigation  | Ok-Robot | 179/210     | 146/184     | 137/180     | 80.48%     |
| Navigation  | Ours     | 228/236     | 239/254     | 224/245     | **94.01%** |
| Long-term   | Ok-Robot | 12/80       | 0/80        | 0/80        | 5.00%      |
| Long-term   | Ours     | 33/80       | 28/80       | 23/80       | **35.00%** |

**关键观察**：
- Ok-Robot 在 Appearance / Positional Shift 下 long-term 成功率为 0——无法局部更新 memory，第二步必然找不到对的 object
- DovSG 长任务 35% 看绝对值仍低，但相对 Ok-Robot 是 7×；subtask 级别都是 80%+，说明 long-term 失败主要来自累积误差（任意一步失败整链失败）

**Table III. 内存与更新效率（1200 帧 RGB-D，40 m² 场景，1cm 分辨率）。**

| Method        | Memory (GB) | Update Time (min) |
| ------------- | ----------- | ----------------- |
| Ok-Robot      | 2           | 20                |
| ConceptGraphs | 0.15        | 27                |
| Ours          | 0.15        | **1**             |

DovSG 与 ConceptGraphs 同样用 scene graph 所以内存接近，但 local update 让更新时间从 27 min 降到 1 min（27×）；相对 Ok-Robot 内存少 13×、更新快 20×。

### Demos

**Video. Appearance task: 抓蓝色玩具放到绿色玩具，然后把青椒放到盘子。**

<video src="https://bjhyzj.github.io/media/videos/dovsg_demo_2.mp4" controls muted playsinline width="720"></video>

**Video. Positional Shift task: 把红椒和青椒分别放到绿色容器，然后把玉米也放进去。**

<video src="https://bjhyzj.github.io/media/videos/video_6.mp4" controls muted playsinline width="720"></video>

---

## 关联工作

### 基于
- [[2401-OKRobot|Ok-Robot]]：mobile manipulation 的核心 baseline，DovSG 直接把它的 pick-and-place 框架扩展到 dynamic 场景
- [[2309-ConceptGraphs|ConceptGraphs]]：3D scene graph 构建的核心思路（object association、CLIP 特征融合、graph 表示）来自此
- HOV-SG (Werby et al., RSS 2024)：CLIP 特征融合的 weighted sum 方法
- RoboEXP (Jiang et al., 2024)：on/belong/inside 三类关系定义及 action-conditioned scene graph 思想

### 对比
- [[2401-OKRobot|Ok-Robot]]：唯一的 mobile manipulation 长任务 baseline；在 dynamic 场景下 long-term 5% vs DovSG 35%
- [[2309-ConceptGraphs|ConceptGraphs]]：内存使用对比（同等水平）+ 更新速度对比（DovSG 快 27×）
- GPT-4o + CoT：作为场景变化检测和 scene graph 生成的 baseline（SCDA / SGA 上明显劣势）

### 方法相关
- DROID-SLAM：相机 pose 估计 backbone
- Grounding DINO + SAM2 + RAM + CLIP：open-vocab 感知 stack（成为 2024 年 robotics 标配组合）
- ACE (Brachmann 2023) + LightGlue：visual relocalization 基础
- AnyGrasp：抓取候选生成
- DynaMem (Liu et al., 2024)：同期 online dynamic spatio-semantic memory，未做对比但思想最接近

---

## 论文点评

### Strengths

1. **问题选取 well-scoped**：dynamic mobile manipulation 是 ConceptGraphs / HOV-SG / Ok-Robot 共同回避的死角，DovSG 提出一个端到端可跑通的方案
2. **Local update 是真 insight**：从 voxel 投影对比筛 affected → 沿 edge 扩 affected set → 仅对受影响 sub-graph 重算关系。简单但工程上 effective，27× 速度提升直接支持论点
3. **Relocalization pipeline 务实**：ACE → LightGlue → colored ICP 的三段精化，每段都有明确分工。这种"组合现有强 component 解决新问题"是 robotics 系统论文该有的姿态
4. **Evaluation 设定贴近现实**：3 种 modification × 4 房间 × 80 trials 的网格，真人 evaluator 构造 GT scene graph，比纯 sim 实验可信

### Weaknesses

1. **Long-term 35% 仍低**：subtask 都 80%+，但因为长任务是串联，任何一步失败整链失败。论文没有 error attribution——35% 中失败的 65% 主要卡在哪个 module（perception / planning / navigation / manipulation）？
2. **GPT-4o baseline 太弱**：把 RGB image 喂给 GPT-4o 让它判断场景变化和场景图，这是个明显劣势的 baseline——没有 3D 信息、没有 voxel 对齐能力。SCDA 上 41% vs 95% 的差距更多反映"应不应该用 LLM 做 voxel-level diff"这种本不该提的问题
3. **关系只取 on/belong/inside 三类**：对家居场景够用，但对更复杂的物理推理（如 "behind"、"between"、function affordance）不够。且 "belong" 的判定标准（点云贴近）容易把不相关物件误判
4. **依赖 floor 作为坐标对齐参考**：multi-floor 场景、有楼梯、地毯反光、户外，全部 break。论文 limitation 段只提了 "sparse texture" 一项
5. **没做 ablation**：local update vs 全局重建的速度对比是有的，但没拆 ACE / LightGlue / ICP 各自贡献，不知道哪步最 critical；也没拆 obsolete voxel removal 的双阈值是否必要
6. **缺与同期 dynamic memory 工作的比较**：DynaMem（参考文献 [16]）就是同期 online dynamic spatio-semantic memory，应该作为 baseline 而非只在 related work 提

### 可信评估

#### Artifact 可获取性
- **代码**：开源，inference-only（github.com/BJHYZJ/DovSG）
- **模型权重**：使用 off-the-shelf RAM / Grounding DINO / SAM2 / CLIP / DROID-SLAM / ACE / LightGlue / AnyGrasp，无需自训
- **训练细节**：ACE 是 scene-specific 在 mapping 阶段训，文中无具体 epochs / lr 等
- **数据集**：评估场景为作者搭建的 4 个真实房间，未公开

#### Claim 可验证性
- ✅ "long-term 35% vs Ok-Robot 5%"：有详细 trial 表（Table II），数字一致
- ✅ "memory 13×, update 27× 节省"：有 Table III 直接对比，方法论清晰
- ⚠️ "SCDA 95% / SGA 88%"：评估由人类 evaluator 做，缺 inter-annotator agreement，可能高估
- ⚠️ "DovSG outperforms in dynamic envs"：只对比 Ok-Robot 一个 baseline；同期 DynaMem 没比
- ⚠️ "local update 是关键"：结论合理，但缺独立 ablation——27× 速度的多少来自 local update vs voxel 化 vs 简化的关系类型？

### Notes

- DovSG 让我重新审视 "scene graph" 的价值。早期 ConceptGraphs / HOV-SG 把 scene graph 当成"压缩的 dense feature"，DovSG 才让它真正承担起 **memory 的本职**——可增量、可局部更新、可被 LLM 索引。这个 framing 比之前的"3D 表示"叙事更贴近 mobile manipulation 的工程需求。
- "On/belong/inside 三类关系够用"是个值得记录的 design decision：在长任务推理中，太多关系类型反而难维护一致性，而这三类直接覆盖 navigation 和 pick-place 的常见 spatial reasoning 需求。这种 minimal-but-sufficient 的设计取舍是好品味。
- 一个有点违和的地方：作者花了大篇幅讲 DROID-SLAM + ACE + LightGlue + ICP 的精确重定位，但 Table II 显示 navigation 成功率 94%——意味着 mobile platform 的定位不是瓶颈。真正的瓶颈在 manipulation（81% pick + 87% place），而 manipulation 又主要靠 AnyGrasp 这个外部 model。所以 long-term 35% 的天花板，本质上由 AnyGrasp 的可靠性决定，DovSG 自己优化的部分（perception + memory）反而不是限制因素。这是一个值得关注的归因——**系统论文的核心贡献 ≠ 系统性能瓶颈**。
- ❓ Local update 在频繁修改的环境下是否会导致 voxel-level drift 累积？论文实验只跑了"两个连续任务"，没测 5 个 / 10 个连续任务后 memory 是否仍准确。
- ❓ 如果一个 object 被遮挡（人挡住），depth diff 会触发误删除。论文的 $\delta_z, \delta_z', \delta_c$ 双阈值是否足以避免这种情况？没看到 robustness 实验。

### Rating

**分数**：2 - Frontier
**理由**：这是 dynamic mobile manipulation 场景下首个端到端把 3D scene graph 做成可增量 local update 的系统工作（RA-L 2025，代码开源），填补了 ConceptGraphs / HOV-SG / Ok-Robot 的静态假设死角，属于方向前沿的代表性参考；但方法本身是 off-the-shelf components（DROID-SLAM + RAM + GroundingDINO + SAM2 + CLIP + ACE + LightGlue + AnyGrasp）的工程整合，核心 insight（local sub-graph update）是重要但不颠覆性的贡献，同期还有 DynaMem 等类似思路，因此不到 Foundation；相较 Archived，它已经被 dynamic scene understanding 方向作为代表性 baseline 讨论，且提供的 13× memory / 27× update 比较是有实质工程价值的参考点。
