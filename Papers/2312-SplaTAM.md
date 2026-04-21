---
title: "SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM"
authors: [Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, Jonathon Luiten]
institutes: [CMU, MIT]
date_publish: 2023-12-04
venue: CVPR 2024
tags: [SLAM, 3D-representation]
paper: https://arxiv.org/abs/2312.02126
website: https://spla-tam.github.io/
github: https://github.com/spla-tam/SplaTAM
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM
> - **核心**: 首次把 3D Gaussian Splatting 作为底层场景表示用于 RGB-D dense SLAM，单一 unposed monocular RGB-D 输入即可联合估计相机位姿与高保真稠密地图
> - **方法**: 简化 3DGS（isotropic + view-independent color），通过 silhouette-guided differentiable rendering 做 tracking / densification / map update 三步在线循环
> - **结果**: 在 Replica / TUM-RGBD / ScanNet++ 上 ATE RMSE 较 implicit baselines（NICE-SLAM, Point-SLAM）至多 2× 提升；rendering 速度可达 400 FPS
> - **Sources**: [paper](https://arxiv.org/abs/2312.02126) | [website](https://spla-tam.github.io/) | [github](https://github.com/spla-tam/SplaTAM)
> - **Rating**: 2 - Frontier（CVPR 2024 早期 3DGS-SLAM 代表作，被后续 3DGS-SLAM 工作普遍作为 baseline；但 SLAM 非我主线研究方向，按 field-centric rubric 定位 Frontier 而非 Foundation）

**Key Takeaways:**
1. **Explicit volumetric beats implicit for SLAM**: 3DGS 提供显式空间外延、可控容量、近线性梯度通路，比 NeRF-style implicit field 更适合 incremental SLAM——后者改一处影响全局，且 ray sampling 限制效率
2. **Silhouette mask is the trick**: 渲染 silhouette $S(\mathbf{p})$ 区分 "已建图 vs 新观测"，让 tracking loss 只在 well-mapped 像素上算，densification 只在新区域加 Gaussian——这一简洁机制是把 3DGS 装进在线 SLAM 的关键
3. **Direct gradient to camera pose**: 由于 Gaussian 有显式 3D 位置/颜色/半径，从 photometric loss 到 pose 参数是几乎线性（投影）的梯度通路，无需穿过 MLP，这是 tracking 收敛比 NeRF-SLAM 快的根本原因
4. **同时五篇并发工作出现**：作者 honesty 列出 GS-SLAM, Gaussian Splatting SLAM, Photo-SLAM, COLMAP-Free 3DGS, Gaussian-SLAM——说明 "3DGS for SLAM" 是 2023 年 12 月集体收敛的 obvious next step

**Teaser. SplaTAM enables sub-cm tracking and 400 FPS photo-realistic novel-view synthesis on RGB-D SLAM.**

<video src="https://spla-tam.github.io/data/scannetpp/scannetpp_s1_recon.mp4" controls muted playsinline width="720"></video>

<video src="https://spla-tam.github.io/data/scannetpp/scannetpp_s1_nvs_loop.mp4" controls muted playsinline width="720"></video>

---

## 1. Motivation: Why explicit volumetric for SLAM

Dense visual SLAM 的核心选择是 *map representation*——它决定了 tracking / mapping / 下游任务的全部设计空间。已有路线两类：

- **Handcrafted explicit**（points, surfels, TSDF）：production-ready，但只解释观测部分，无法做 novel-view synthesis；tracking 依赖丰富的几何特征 + 高帧率
- **Neural implicit**（iMAP, NICE-SLAM, Point-SLAM）：高保真全局地图 + dense photometric loss，但 (1) 计算昂贵 (2) 不可编辑 (3) 几何不显式 (4) 灾难性遗忘——网络改一处影响全局

> ❓ 作者把 implicit 的 "catastrophic forgetting" 单独列出，其实在 SLAM 设定下这与 "spatial frontier 不可控" 是同一问题的两面：implicit 网络无法局部更新，因为参数全局耦合。

3DGS [Kerbl et al. 2023] 提供了第三条路：**显式 + 可微渲染 + 快速光栅化**。作者论证它对 SLAM 的四个优势：

1. **Fast rendering & rich optimization**：rasterization 而非 ray marching，可承受 per-pixel dense photometric loss
2. **Maps with explicit spatial extent**：通过 silhouette 渲染立刻判断 "这个像素是不是已建图区域"
3. **Explicit map**：增加容量 = 加 Gaussian，可编辑
4. **Direct gradient flow**：Gaussian 参数（位置、颜色、半径）→ 渲染像素是近线性的投影，camera pose 也类似（"keeping camera still and moving the scene"）。无需穿过 MLP

**Figure 1. SplaTAM 在 texture-less 大幅相机运动场景下达到 sub-cm 定位，baselines tracking 失败。右图：train + novel view 在 876×584 下 400 FPS 渲染。**

![](https://ar5iv.labs.arxiv.org/html/2312.02126/assets/x1.png)

---

## 2. Method

### 2.1 Gaussian Map Representation

相比原版 3DGS，SplaTAM 做了两个简化：**view-independent color**（去掉 SH）+ **isotropic Gaussian**（单半径而非 covariance matrix）。每个 Gaussian 8 个参数：RGB color $\mathbf{c} \in \mathbb{R}^3$，center $\boldsymbol{\mu} \in \mathbb{R}^3$，radius $r$，opacity $o \in [0, 1]$。

**Equation 1. Gaussian influence function**

$$
f(\mathbf{x}) = o \exp\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{2r^2}\right)
$$

> ❓ 简化掉 anisotropy 是对 SLAM 场景的合理假设吗？原版 3DGS 用 anisotropic 是为了拟合细薄结构（如树叶），SLAM 场景大多是平面/凸面，isotropic 可能损失细节但换来 tracking 效率。Ablation 应该会说明。

### 2.2 Differentiable Splatting Rendering

Front-to-back sort 后 alpha-compositing。除了 RGB，作者额外渲染 **depth** 和 **silhouette**：

**Equation 2-4. Color / Depth / Silhouette rendering**

$$
C(\mathbf{p}) = \sum_{i=1}^n \mathbf{c}_i f_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - f_j(\mathbf{p}))
$$

$$
D(\mathbf{p}) = \sum_{i=1}^n d_i f_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - f_j(\mathbf{p}))
$$

$$
S(\mathbf{p}) = \sum_{i=1}^n f_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - f_j(\mathbf{p}))
$$

splatting 后 2D 参数 $\boldsymbol{\mu}^{2D} = K E_t \boldsymbol{\mu} / d$，$r^{2D} = fr/d$。silhouette 是 alpha 的累积，可解读为 "该像素被当前 map 覆盖的置信度"——这是后续 tracking mask 的来源。

### 2.3 SLAM Pipeline

**Figure 2. SplaTAM Overview。三步循环：tracking → densification → map update。**

![](https://ar5iv.labs.arxiv.org/html/2312.02126/assets/x2.png)

给定 frame $t+1$ 的 RGB-D：

**Step 1: Camera Tracking**
- Constant-velocity 初始化：$E_{t+1} = E_t + (E_t - E_{t-1})$，相机参数化为 quaternion + translation
- Gradient-based 更新 pose，**Gaussian 参数 frozen**，loss 仅在 silhouette $S(\mathbf{p}) > 0.99$ 的可信像素上：

$$
L_t = \sum_{\mathbf{p}} \big(S(\mathbf{p}) > 0.99\big) \Big( L_1\big(D(\mathbf{p})\big) + 0.5 \cdot L_1\big(C(\mathbf{p})\big) \Big)
$$

depth L1 + 0.5× color L1。silhouette gating 是核心——避免新观测区域（map 未覆盖）污染梯度。

**Step 2: Gaussian Densification**

定义 densification mask：

$$
M(\mathbf{p}) = \big(S(\mathbf{p}) < 0.5\big) + \big(D_{GT}(\mathbf{p}) < D(\mathbf{p})\big) \big(L_1(D(\mathbf{p})) > 50 \cdot \text{MDE}\big)
$$

两类像素需要新 Gaussian：(a) silhouette 不足（map 未覆盖）；(b) 真实深度在估计深度前面 + 深度误差超过 50× median depth error（说明前方有未建模的新几何挡住了已有几何）。

> ❓ "50× MDE" 的阈值在 rebuttal 中作者承认是经验调出来的（"empirically by visualizing the densification mask"）。这种 magic number 在 cross-dataset 上鲁棒性是个 open question。

新 Gaussian 初始化：center 在 unproject 到的 3D 点，opacity = 0.5，radius = $D_{GT}/f$（即投影回去恰好 1 像素半径）。

**Equation 5. Initialization radius**

$$
r = \frac{D_{GT}}{f}
$$

**Step 3: Map Update**
- Camera poses fixed，Gaussian 参数更新
- **Warm-start** 自上一轮 map（不是从头训）
- 选 keyframes 优化：当前帧 + 最近 keyframe + $k-2$ 个与当前帧 frustum overlap 最大的历史 keyframe（overlap = 当前帧深度反投影后落入历史 keyframe frustum 的点数）
- Loss 同 tracking 但不用 silhouette mask（要全像素优化），加上 SSIM RGB loss + cull 掉 opacity≈0 或体积过大的 Gaussian（沿用 [Kerbl et al. 2023]）

---

## 3. Experiments

> 抓取的论文文本在 Table 1 caption 后被截断，正文未拿到完整实验数字。以下信息综合 Abstract、Table 1 caption 与 project page。

**Table 1. Online Camera-Pose Estimation (ATE RMSE ↓ [cm])**：在 ScanNet++、Replica、TUM-RGBD 上一致优于 SOTA dense baselines；在 Orig-ScanNet 上 competitive。Baselines 数字取自 Point-SLAM。

**评估数据集**：
- **Replica** [Straub 2019]：合成 indoor，标准 RGB-D SLAM benchmark
- **TUM-RGBD** [Sturm 2012]：真实场景
- **ScanNet** [Dai 2017]：真实 indoor scans
- **ScanNet++** [Yeshwanth 2023]：高保真 3D indoor 数据集，作者引入用于 NVS 评估

**核心 claim**（来自 abstract）：camera pose / map construction / NVS 三个指标上至多 2× 提升 over SOTA；高分辨率下 400 FPS 渲染。

**Figure 3. ScanNet++ S2 重建可视化**：估计相机位姿（绿框 + 红轨迹）与 GT（蓝框 + 蓝轨迹）紧密贴合，重建为高保真稠密表面。

![](https://ar5iv.labs.arxiv.org/html/2312.02126/assets/figs/scene2_v3.png)

**Video. Replica Room 0 SplaTAM NVS 渲染**——SplaTAM 直接从其优化好的 Gaussian map 渲染 novel view RGB。注：作者明确披露同图比对中 NICE-SLAM / Point-SLAM 在 NVS 时使用了 GT novel view depth，而 SplaTAM 不依赖。

<video src="https://spla-tam.github.io/data/replica_nvs/splatam/room_0_rgb.mp4" controls muted playsinline width="720"></video>

**Video. iPhone 在线重建 collage**——RGB-D 来自 iPhone 摄像头 + ToF 传感器，展示真实手持设备可用性。

<video src="https://spla-tam.github.io/data/collage.mp4" controls muted playsinline width="720"></video>

---

## 4. Concurrent Work

作者在 project page 罕见地诚实列出 5 篇同期 3DGS-SLAM 工作，每篇思路不同：

- **GS-SLAM**：coarse-to-fine tracking based on sparse Gaussian selection
- **Gaussian Splatting SLAM**：monocular（不需深度），densification 用 depth 统计
- **Photo-SLAM**：ORB-SLAM3 tracking + 3DGS mapping 解耦
- **COLMAP-Free 3DGS**：mono depth estimation + 3DGS
- **Gaussian-SLAM**：DROID-SLAM tracking + 主动/非主动 3DGS sub-maps

> 这种 cluster 现象本身是个 signal——3DGS 出来 5 个月内多组独立收敛到 "用它做 SLAM"，说明显式可微表示对 SLAM 是 "obvious next step"。SplaTAM 的差异化在 silhouette-guided 的 unified pipeline，而非 hybrid coupling。

---

## 关联工作

### 基于
- **3D Gaussian Splatting** [Kerbl et al. SIGGRAPH 2023]: 底层表示与可微 rasterizer
- **Dynamic 3D Gaussians** [Luiten et al. 2023]: 同一作者前作，把 3DGS 扩到 dynamic scene 的 6-DOF tracking——SplaTAM 是其 SLAM 化

### 对比 (Implicit SLAM baselines)
- **iMAP** [Sucar et al. ICCV 2021]: 第一个用 neural implicit 做 SLAM 的工作
- **NICE-SLAM** [Zhu et al. CVPR 2022]: hierarchical multi-feature grids，扩展 iMAP scalability
- **Point-SLAM** [Sandström et al. ICCV 2023]: neural point cloud + volumetric rendering，最强 implicit baseline
- **ESLAM** [Johari et al. CVPR 2023]: hybrid SDF-based

### 对比 (Traditional dense SLAM)
- **KinectFusion** [Newcombe 2011]: TSDF 经典
- **BundleFusion** [Dai 2017]: globally consistent TSDF
- **ElasticFusion** [Whelan 2015]: surfel-based，differentiable rasterization 先驱
- **BAD SLAM** [Schops 2019]: bundle-adjusted RGB-D direct method

### Concurrent (3DGS-SLAM)
- GS-SLAM, Gaussian Splatting SLAM, Photo-SLAM, COLMAP-Free 3DGS, Gaussian-SLAM——见 §4

### 数据集
- **ScanNet++** [Yeshwanth ICCV 2023]: 高保真 indoor benchmark，作者用其做 NVS evaluation

---

## 论文点评

### Strengths

1. **方法极简而 unified**：tracking / densification / map update 共用一套 differentiable splatting + photometric loss，没有 ORB-SLAM / DROID 这样的外部 tracker。silhouette mask 一个机制同时解决 tracking gating 和 densification 触发——是 "simple, scalable" 的好例子
2. **Direct gradient 论证有 first-principles 味道**：作者明确指出 "Gaussian 参数到像素是近线性投影 + camera 等价于 inverse scene motion"，这是为什么 implicit-SLAM tracking 慢的根本原因。这个 framing 比单纯 benchmark 数字更有 explanatory power
3. **Honest disclosure**：concurrent work 全列、rebuttal 中承认 50×MDE 是经验值、明确 NICE-SLAM/Point-SLAM 用了 GT depth 做 NVS——这种 honesty 在 SOTA-claiming 论文里少见
4. **Open source + iPhone demo**：codebase + 真实手持设备验证，作为 building block 易于复用

### Weaknesses

1. **依赖准确深度**：是 RGB-D（不是 mono）SLAM——硬件门槛较高。Concurrent 的 Gaussian Splatting SLAM 直接做 mono，方法 ceiling 更高
2. **No loop closure / global BA**：纯前向增量优化，silhouette+keyframe overlap 是局部的。在大场景长轨迹下 drift 会累积，论文未讨论
3. **Magic numbers**：`S > 0.99` (tracking)，`S < 0.5` (densification)，`50× MDE`，`opacity 0.5 init`——这些阈值对 cross-dataset 鲁棒性的 ablation 不充分
4. **简化 3DGS 的 trade-off 不清楚**：去掉 anisotropy + view-dependent color 是否在反光/细薄结构场景下牺牲质量？只在 Replica/ScanNet++ 这类 benign indoor 场景验证
5. **Tracking 是 per-frame iterative optimization**：每帧需若干步梯度下降，"online" 但不是 real-time SLAM 意义下的 real-time。论文未给 tracking step 的耗时分解

### 可信评估

#### Artifact 可获取性

- **代码**: 完整开源（inference + training），github.com/spla-tam/SplaTAM
- **模型权重**: N/A —— SLAM 是 per-scene online optimization，无 pretrained weights
- **训练细节**: README 提供 Replica / ScanNet / ScanNet++ / TUM-RGBD / iPhone 的 config 文件；超参完整
- **数据集**: 全部公开 benchmark（Replica / TUM-RGBD / ScanNet / ScanNet++）

#### Claim 可验证性

- ✅ **400 FPS rendering @ 876×584**：可由开源代码 + 单 GPU 验证；splatting rasterization 速度有 [Kerbl 2023] 上游证据
- ✅ **优于 NICE-SLAM / Point-SLAM 的 ATE RMSE**：开源 + 标准 benchmark 可独立复现；作者注明 baselines 数字直接取自 Point-SLAM 论文
- ⚠️ **"up to 2× SOTA"**：是单 best case 而非平均；Orig-ScanNet 仅 "competitive"。一般化时应说 "consistently better on 3 of 4 benchmarks"
- ⚠️ **NVS 优势**：比较时 NICE-SLAM / Point-SLAM 用 GT novel-view depth render，SplaTAM 不用——作者明确披露但仍构成 apples-to-oranges
- ⚠️ **Sub-cm tracking on texture-less**：Figure 1 的 "baselines fail" 是定性 cherry-pick，未量化失败率

### Notes

- **对我研究的意义**：3DGS-SLAM 这一 line 与 spatial intelligence / embodied perception 间接相关——若 embodied agent 需要 online 构建 high-fidelity 可查询 3D map，SplaTAM 这类方法是自然候选。但纯 SLAM 不是我研究主线，本笔记定位为 indexed reference
- **可借鉴的 design pattern**：silhouette mask 作为 epistemic uncertainty proxy 是个简洁 idea——在其他在线学习/增量优化场景（如 streaming VLM continual learning）也许可借鉴 "用 model 自身的覆盖度判断该不该相信 loss"
- **Pivot 信号**：5 篇并发工作 = 这是 obvious idea。"早一步发出来" 的 SplaTAM 拿到了 CVPR + 高引，但同期任一篇都能做出类似贡献。从 idea-generation 角度：当一个方法变成 obvious 时，再追随就是 marginal contribution
- **Open question**：3DGS-SLAM 后续是否解决了 loop closure？需要查 2024-2025 的 follow-up（如 LoopSplat 类工作）

### Rating

**分数**：2 - Frontier
**理由**：SplaTAM 是 CVPR 2024 早期 3DGS-SLAM 代表作，开源完整、被 GS-SLAM/MonoGS/Photo-SLAM 等后续 3DGS-SLAM 普遍作为 baseline，方法上的 silhouette-guided unified pipeline 具备 first-principles 清晰度（见 Strengths #1-2）。之所以不给 3，是按 field-centric rubric——SLAM 不是我的主线方向（VLA / Agent / Embodied），且在 3DGS-SLAM 内部同期并发 5 篇收敛思路（见 §4），SplaTAM 是时间上稍先一步但思路非独家的 Frontier 而非奠基工作；之所以不给 1，是因为它仍是当前 3DGS-SLAM 方向的标准对比对象，作为 indexed reference 仍具参考价值。
