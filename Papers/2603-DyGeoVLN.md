---
title: "DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation"
authors: [Xiangchen Liu, Hanghan Zheng, Jeil Jeong, Minsung Yoon, Lin Zhao, Zhide Zhong, Haoang Li, Sung-Eui Yoon]
institutes: [KAIST, HKUST(GZ), JD Explore Academy]
date_publish: 2026-03-22
venue: arXiv
tags: [VLN, 3D-representation, spatial-reasoning]
paper: https://arxiv.org/abs/2603.21269
website:
github:
rating: 2
date_added: 2026-04-23
---

## Summary

> [!summary] DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation
> - **核心**: 把一个**自研的 dynamic-aware geometry foundation model (DGFM)** 作为 3D 分支塞进 MLLM-based VLN，用 cross-branch attention 融合 2D 语义 token 与 3D 几何 token，对付现有 VLN 在动态场景里 3D 几何崩坏的问题。
> - **方法**: DGFM = `π³`/`VGGT` 风格的 feed-forward 几何 backbone + `Depth Anything` 生成的 local point map 作为显式 3D 条件（zero-init conv 残差注入）+ 分层 latent 解码（camera / local / global）；训练用新构造的 DyHM3D 数据集（HM3D 上贴 ~50k 条 skeletal-driven 人体运动轨迹）。上层 VLN 用 [[2412-NaVILA|NaVILA]]-style Qwen2-VL 作为 2D 分支、cross-attention 把几何 token 融进去，再加一个 **pose-free occupancy-aware voxel token pruning** 压缩长历史。
> - **结果**: R2R-CE Val-Unseen **SR 60.8 / SPL 55.8 / NE 4.41 / OSR 70.1**，在单目 RGB 方法中 SOTA，并反超依赖 panoramic RGB-D + waypoint predictor 的 g3D-LF / ETPNav；HA-VLN dynamic benchmark SR 0.40（Val-Unseen），比 [[2507-StreamVLN|StreamVLN]] 的 0.33 高 +7 pt、CR 从 0.42→0.38。
> - **Sources**: [paper](https://arxiv.org/abs/2603.21269)
> - **Rating**: 2 - Frontier（动态 VLN 方向当下最强的 SR 数字 + 对 "dynamic 3D foundation model" 这一子方向做了一次 end-to-end 尝试，但 DGFM 和 VLN 两侧都缺 ablation 粒度、无代码）

**Key Takeaways:**
1. **VGGT/π³ 类 static 几何 FM 在动态场景会崩**：作者用红框 qualitative 给出证据（Fig. 2b）——人体部位 reconstruction 扭曲、定位不准。对 VLN 是致命的，因为动态障碍（行人）本身就是主要交互对象。
2. **"Dynamic geometry" 的实现其实是 depth-guided residual 注入**：DGFM 并没有重新设计 dynamic geometry 架构，而是把 Depth Anything 的单目深度→local point map→point-map Transformer→zero-init conv 作为**残差分支**加到 π³/VGGT backbone 上，再在 DyHM3D 上 finetune。这是工程上合理、但理论上保守的做法——"动态" 的增量主要来自**带人体运动的训练数据**，而不是架构级的时序建模。
3. **Pose-free token pruning 是实用亮点**：相比 StreamVLN 依赖 Habitat 提供的 GT pose / depth 做 voxel pruning，本文用 DGFM 自推 pose + point cloud 做 voxelization，真机部署只要单目 RGB——这个 pipeline 上的改动对 real-world deployment 的价值可能比 DGFM 本身更大。
4. **单目 RGB 反超 panoramic RGB-D 是本工作最硬的信号**：R2R-CE 上 60.8 SR > g3D-LF 61.0 SR 只差 0.2 但 SPL 55.8 > 52.0 明显更高，且只用 Monocular RGB；证明 "几何 FM 语义 token 融合" 比 "panoramic sensor + waypoint predictor" 这一条传统路径更 scalable。

---

### Method Overview

![](https://arxiv.org/html/2603.21269v1/x1.png)
**Figure 1.** DyGeoVLN 系统总览：vision encoder（Qwen2-VL）和 DGFM 并行产出 visual token / geometry token，cross-branch fusion 得到 spatial-semantic token，喂给 LLM 解出 action token；pruning 模块基于 DGFM 自推 pose / point cloud 做 voxel 压缩。

### Dynamic Geometry Foundation Model (DGFM)

DGFM 是本文的核心工作，基于已有的 static geometry foundation model（`π³` / `VGGT`）做三个增量：**depth-guided 显式 3D 注入、dynamic-aware fusion、分层 latent 解码**。

![](https://arxiv.org/html/2603.21269v1/x2.png)
**Figure 2.** (a) DGFM 架构：3D latent 分支与 2D ViT 分支在多层做 feature fusion，再沿时间轴对齐；(b) 重建 qualitative 对比——红框圈出 π³ 和 VGGT 对人体重建失败（位置偏、几何破碎），DGFM 保留完整人体几何。

**Depth-guided local point map**。对每个 RGB 帧 $I_t$，先用现成 Depth Anything 预测深度 $D_t$，然后借 intrinsic $K$ 反投影成 local point map $\mathbf{P}_t$。这是**显式 3D cue 的来源**，相当于给 geometry backbone 一个初始几何 anchor。

> ❓ Depth Anything 本身是单目深度估计，在动态场景里的 temporal 稳定性也有限，尤其是人体表面有强纹理变化时。DGFM 把这个 noisy 深度作为 backbone 的 residual 输入，效果能 generalize 到多大程度？文中没有对 depth 质量做消融。

**Point-map Transformer + 多层 3D embedding**。把 $\mathbf{P}_t$ patch 化后喂一个 self-attention stack，coarse-to-fine 输出 $\mathbf{G}_t^{(1)}, ..., \mathbf{G}_t^{(S)}$，作为显式几何 token。

**Dynamic-aware fusion（零初始化残差）**。2D 分支复用 π³/VGGT 的 frame-wise ViT backbone 产生多层 feature $\mathbf{E}_t^{(s)}$；把 3D 分支对齐到同 channel 后用 **zero-mean conv** 加到 2D feature 上：

$$
\tilde{\mathbf{E}}_{t}^{(s)} = g_{\text{zm}}\bigl(\mathbf{G}_{t}^{(s)}\bigr) + \mathbf{E}_{t}^{(s)}, \quad s=1,\ldots,S
$$

**含义**：$g_{\text{zm}}$ 的权重和 bias 初始化为 0，finetune 过程中逐步学出 3D 残差贡献——这保证 π³/VGGT 的 pretrained backbone 起点分布不被破坏，本质和 ControlNet / LoRA 的思想同构。

**分层 latent 解码**。fused feature 通过共享 decoder $\mathcal{D}$ 得到 $\mathbf{H}_t$，再 split 出 $\mathbf{H}_t^{\text{cam}}$（相机运动）、$\mathbf{H}_t^{\text{loc}}$（局部点几何），然后 aggregate 后解出 global latent $\mathbf{H}_t^{\text{glo}}$ —— 这个 global latent 才是喂给 VLN 上层的几何 token。

**DyHM3D 数据集**。在 HM3D 场景的 navigation mesh 上采样 waypoints，用 skeletal-driven 3D human model 生成自然行走动画，相机做线性插值跟拍，约 50k 条轨迹，每条带 RGB / depth / pose / intrinsic。

![](https://arxiv.org/html/2603.21269v1/x3.png)
**Figure 3.** DyHM3D 示例轨迹：第一人称视角、场景内有 skeletal human 在按采样路径移动。

> ❓ 50k trajectories 听起来大，但只有 skeletal human 一种动态实体，没有开门、家具移动、宠物等其他常见 VLN 动态。论文自己承认 "pipeline can be extended" 但没做，这限制了 "dynamic" 的语义范围——实际上是 "human-dynamic"。

### Cross-branch Feature Fusion VLN Architecture

2D 分支用 Qwen2-VL 的 vision encoder 把 $x_t$ 切成 patch token $\mathbf{V}_t$，再经多模态 projector 得 $\mathbf{X}_t$。3D 分支就是 DGFM 输出 $\tilde{\mathbf{H}}_t^{\text{glo}}$（已对齐到 visual token 维度）。两者用一次 **multi-head cross-attention** 融合：

$$
\mathbf{F}_{t} = \text{Cross-Atten}\bigl(Q=\mathbf{X}_{t}, K=\tilde{\mathbf{H}}_{t}^{\text{glo}}, V=\tilde{\mathbf{H}}_{t}^{\text{glo}}\bigr)
$$

**含义**：2D patch token 作为 query 拉取几何信息，最终每个 visual token 都携带全局空间上下文。这是"几何 condition 语义"的 asymmetric 设计，与 [[2507-StreamVLN|StreamVLN]] 的 slowfast 时间分支互补。

**Sliding-window KV cache**：滑窗保留最近 $N$ 帧做 active，超窗的历史靠下一节的 pruned memory token 表示。每步 append 新帧 KV、丢最老帧，保证 context 有界。

### Adaptive-resolution & Occupancy-aware Token Pruning

关键点：**pose-free & depth-free from sensor perspective**——pose 和 point cloud 都是 DGFM 自推出来的，不依赖 Habitat GT。四个子模块：

1. **Adaptive-resolution voxel grouping**：voxel 大小随 token 的深度 scale（近处细粒度、远处粗粒度），两级适应。
2. **Occupancy-aware selection**：同一 voxel 内多 token 竞争，支持 `latest` / `priority (recency+proximity)` / `top-K` 三种规则。
3. **Importance-aware completion**：保证每帧最小 keep ratio $\rho$，不足则按 feature magnitude + range + spatial distribution + temporal recency 加权挑重要的回补。
4. **Temporal smoothing**：多数投票平滑 binary mask，抑制孤立毛刺。

> ❓ 没看到 pruning 的定量分析——压缩多少 token、推理加速多少、token 预算对 SR 的曲线都没报。只在 ablation 里说 "带 pruning SR +3%"。pose-free 相比 StreamVLN 的 pose-dependent 是否有精度损失？没直接比较。

### Experiments

**Dynamic HA-VLN Benchmark（Val-Unseen）**：

| Method | Observation | NE↓ | TCR↓ | CR↓ | SR↑ |
|---|---|---|---|---|---|
| g3D-LF* | Panoramic RGB-D | 5.30 | 4.54 | 0.49 | 0.27 |
| NaVid | Monocular RGB | 7.49 | 6.17 | 0.49 | 0.34 |
| NaVILA | Monocular RGB | 6.39 | 4.42 | 0.45 | 0.32 |
| StreamVLN | Monocular RGB | 5.59 | 4.03 | 0.42 | 0.33 |
| **DyGeoVLN** | Monocular RGB | **5.12** | **3.69** | **0.38** | **0.40** |

比 StreamVLN +7 pt SR、CR 从 0.42→0.38。论文自述 "对比 VLM-based 方法平均 +10% SR / -8% CR"。

![](https://arxiv.org/html/2603.21269v1/x4.png)
**Figure 4.** 动态 HA-VLN 定性对比：StreamVLN 在有行人的场景丢失路径、撞人，DyGeoVLN 能绕行并到达目标。

**Static R2R-CE Benchmark（Val-Unseen）**：

| Method | Observation | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---|---|---|---|---|
| g3D-LF* | Panoramic RGB-D | 4.53 | 68.0 | 61.0 | 52.0 |
| [[2304-ETPNav\|ETPNav]]* | Panoramic RGB-D | 4.71 | 65.0 | 57.0 | 49.0 |
| [[2402-NaVid\|NaVid]] | Monocular RGB | 5.47 | 49.1 | 37.4 | 35.9 |
| [[2412-NaVILA\|NaVILA]] | Monocular RGB | 5.22 | 62.5 | 54.0 | 49.0 |
| [[2507-StreamVLN\|StreamVLN]] | Monocular RGB | 5.10 | 64.0 | 55.7 | 50.9 |
| NavFoM | Monocular RGB | 5.01 | 64.9 | 56.2 | 51.2 |
| **DyGeoVLN** | Monocular RGB | **4.41** | **70.1** | **60.8** | **55.8** |

**单目 RGB 反超 panoramic RGB-D** 是本工作最硬的信号：SPL 55.8 > g3D-LF 的 52.0。

![](https://arxiv.org/html/2603.21269v1/x5.png)
**Figure 5.** R2R-CE 定性对比：StreamVLN 中途停下，DyGeoVLN 轨迹一致性更好。

**真机实验（Unitree Go1 + D435i + Jetson Orin Nano）**。用 DyGeoVLN 作为 high-level 规划头输出 pixel goal，下接 diffusion action head 做连续轨迹，MPC 跟踪。20 episodes / 场景，SR 作为指标。

![](https://arxiv.org/html/2603.21269v1/x6.png)
**Figure 6.** 跨不同真实场景（corridor / lobby / indoor room / crowded）的 SR 对比，DyGeoVLN 在所有难度下都高于 NaVid / Uni-NaVid / NaVILA / StreamVLN。

![](https://arxiv.org/html/2603.21269v1/x7.png)
**Figure 7.** 真机动态场景示例：机器人按指令前行并避让行人。

**Ablation（HA-VLN Val-Unseen）**：

| Variant | NE↓ | TCR↓ | CR↓ | SR↑ |
|---|---|---|---|---|
| DyGeoVLN (Full) | 5.12 | 3.69 | 0.38 | 0.40 |
| w/o Visual Semantic | 6.82 | 4.96 | 0.51 | 0.30 |
| w/o Spatial Geometry | 5.54 | 4.03 | 0.43 | 0.34 |
| w/o Dynamic Spatial Injection | 5.37 | 4.10 | 0.43 | 0.36 |
| w/o Spatial Token Pruning | 5.33 | 3.94 | 0.39 | 0.37 |

结论：(1) 2D/3D 双分支都必要（去 2D 掉 10 pt SR 最严重）；(2) depth-guided dynamic injection 贡献 +4 pt SR；(3) pruning 贡献 +3 pt SR（说明 pruning 不仅减 token，也改善性能——去掉冗余帮模型聚焦关键区域）。

---
## 关联工作

### 基于
- **`π³` / `VGGT`**（Wang 2025）：feed-forward 几何 FM，本文的 DGFM 直接 inherit 其 frame-wise ViT encoder、camera decoder、local point decoder；本文只是额外加 3D embedding 分支、zero-init fusion、global-scale latent head。
- [[2412-NaVILA|NaVILA]]：Qwen2-VL-style VLM 作为 VLN 基座；本文 2D 分支直接用 Qwen2-VL vision encoder。
- **Depth Anything**（Yang 2024）：提供 local point map 的深度前端。
- **HM3D + Habitat**：DyHM3D 数据集构造所基于的底座。

### 对比
- [[2507-StreamVLN|StreamVLN]]：当前 SOTA VLM-based VLN，用 slowfast 时间分支 + voxel-based pruning 但依赖 Habitat GT pose/depth；本文**方法论核心差异**就是 pose-free pruning 和 explicit geometry 注入，两个 benchmark 都比它高。
- [[2402-NaVid|NaVid]] / Uni-NaVid / [[2412-NaVILA|NaVILA]]：同为 Monocular RGB VLM-based VLN，但没有显式几何分支。
- [[2304-ETPNav|ETPNav]] / g3D-LF：Panoramic RGB-D + waypoint predictor 路线；本文证明单目 + 几何 FM 可以反超。
- **JanusVLN / Efficient-VLN**（zeng2025 / zheng2025）：同样试图把 3D FM 塞进 VLN 的近期工作，但本文认为它们只是 plug-and-play off-the-shelf VGGT，在动态场景会崩——这是本文最直接的 diff point。

### 方法相关
- **Dynam3D**（wang2025dynam3d）：与本文名字最像的工作，走的是 panoramic RGB-D + 3D 记忆路线，定位与本文正交。
- **ControlNet-style zero-init residual**：DGFM 的 fusion 设计思想同构，不过论文没主动引。

---
## 论文点评

### Strengths

1. **明确的问题定位**：static geometry FM（VGGT/π³）在动态场景会崩这一 observation 精准——Fig. 2b 红框对比给出直观证据，对比 JanusVLN 只做 "plug-and-play 3D FM 进 VLN" 的做法，本文至少试图解决一个具体失败模式。
2. **Pose-free pruning 的工程价值**：相比 StreamVLN 依赖 Habitat GT pose/depth 的 voxel pruning，本文用 DGFM 自推 pose/point cloud 实现 real-world 单目部署，这在 deployability 上是一个真实进步。
3. **R2R-CE 单目 RGB 反超 Panoramic RGB-D**：SPL 55.8 > g3D-LF 52.0，说明 geometry FM 注入 > panoramic sensor 这条传统范式，有"表示 > 传感器"的意义。
4. **消融相对干净**：4 个组件各有 +3~+10 pt SR 的清晰贡献，不是 additive + 虚增。

### Weaknesses

1. **"Dynamic" 增量主要来自数据而非架构**：DGFM 的"动态感知"关键其实是 **DyHM3D 上 finetune + depth 作为 residual**，没有显式的时序动态建模（如 motion-aware attention、object permanence）。如果用同样的数据去 finetune 原始 VGGT，差距会缩到多少？论文没做这个关键 ablation。
2. **数据集 coverage 狭窄**：DyHM3D 只有 skeletal human，没有开关门、家具移动、宠物等 VLN 真实部署会遇到的动态类。"dynamic VLN" 的 claim 被 over-extended。
3. **缺代码 / 缺 pruning 定量分析**：没有开源仓库链接；pruning 只在 ablation 里有一行 SR，没有 token 压缩比、推理延迟、pose 自推误差 vs GT pose 的对比——这些恰好是 deployability claim 的硬依据。
4. **3D branch 训练成本不透明**：DGFM 在 DyHM3D 上 finetune 的 compute、VLN stage 的总 training step 都没披露。
5. **与 JanusVLN / Efficient-VLN 的定量对比缺失**：正文只在 related work 里提到"它们依赖 off-the-shelf 3D FM 在动态场景会崩"，但实验表没放它们的数字——这是最直接的对比点，缺这个对比让"我们的 dynamic 比他们的 static 3D FM 好"的 claim 变弱。

### 可信评估

#### Artifact 可获取性
- **代码**：未开源（arXiv v1 没给 code link，搜到的 GitHub 都是同名/相关但不相关工作）
- **模型权重**：未说明
- **训练细节**：仅高层描述（初始化策略 + DAgger + 几个训练数据集名字；超参、batch size、step 数未披露）
- **数据集**：DyHM3D 声明会提供（~50k trajectories）但未给链接；训练用的其他数据集（R2R-CE、RxR-CE、EnvDrop、HA-VLN、ScaleVLN）都公开

#### Claim 可验证性
- ✅ **R2R-CE 和 HA-VLN 数字表现**：有完整 table，benchmark 公开，数字可独立复现（前提是开源）
- ✅ **Sliding-window KV + pruning 的基本思路**：算法 1 给出伪代码，逻辑自洽
- ⚠️ **"Dynamic geometry FM > static FM"**：只有 qualitative Fig. 2b 和 Fig. 9 支持，没有在标准几何 benchmark（如 ScanNet dynamic subset、动态 3D reconstruction metric）上量化；DGFM 自身的几何质量没有定量报数
- ⚠️ **"Pose-free pruning 与 StreamVLN pose-based 相当或更好"**：没有直接的 pose 误差对比，只能靠端到端 SR 间接推断
- ⚠️ **真机 SR 曲线（Fig. 6）**：每模型 20 episodes 样本量偏小，缺置信区间

### Notes

- DGFM 的设计 pattern（**pretrained FM + zero-init residual 注入显式条件**）跨 domain 可迁移——类比 ControlNet 之于 SD、LoRA 之于 LLM。VLA 方向上"把显式物理 prior（碰撞检测、运动学）作为零初始化残差接入通用 VLA"可能是类似的 low-risk 工程 pattern。
- "Finetune VGGT / π³ on dynamic data" 的思路本身值得单独做成一篇工作——这篇其实把它作为副产物塞在 VLN 论文里了，DGFM 的独立评估缺位可惜。
- 可继承点：`pose-free occupancy-aware voxel pruning` 可以作为 long-context video LLM / streaming VLA 的通用 memory 压缩策略，不限于 VLN。

### Rating

**Metrics** (as of 2026-04-23): citation=0, influential=0 (0%), velocity=0.0/mo; HF upvotes=N/A（HF 无对应 paper 页）; github N/A（未开源）

**分数**：2 - Frontier
**理由**：在 VLN 方向上属于**当前时点最强的 Monocular RGB 数字**（R2R-CE SR 60.8 / SPL 55.8 反超 panoramic RGB-D；HA-VLN SR 0.40 明显超 StreamVLN 0.33），思路 clear 且对 "dynamic VLN" 这一被忽视的子方向做了正面回应——因此高于 Archived。但**不进 Foundation** 是因为：(1) 刚发布 1 个月 citation=0、无代码无 HF 页面，影响力信号未到 landmark 级；(2) DGFM 的"dynamic" 增量主要来自数据（DyHM3D human 轨迹）而非架构创新，相对 VGGT/π³ 的方法论增量较保守；(3) 动态范围仅限 skeletal human。若半年后 DGFM 开源且被后续 VLN/VLA 工作广泛作为 3D FM baseline，可升 3。
