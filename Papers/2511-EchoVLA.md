---
title: "EchoVLA: Synergistic Declarative Memory for VLA-Driven Mobile Manipulation"
authors: [Min Lin, Xiwen Liang, Bingqian Lin, Liu Jingzhi, Zijian Jiao, Kehan Li, Yu Sun, Weijia Liufu, Yuhan Ma, Yuecheng Liu, Shen Zhao, Yuzheng Zhuang, Xiaodan Liang]
institutes: [Sun Yat-sen University (Shenzhen), Shanghai Jiao Tong University, Huawei Noah's Ark Lab]
date_publish: 2025-11-22
venue: arXiv preprint
tags: [mobile-manipulation, VLA, spatial-memory]
paper: https://arxiv.org/abs/2511.18112
website:
github:
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] EchoVLA: Synergistic Declarative Memory for VLA-Driven Mobile Manipulation
> - **核心**: 把人脑 declarative memory 的 PHC（场景）+ hippocampus（情景）双路结构搬到 VLA 里，给 mobile manipulation 加一个持久 voxel scene memory 和一个 FIFO episodic token buffer，分别用 coarse / fine cross-attention 检索后注入 per-part diffusion policy
> - **方法**: SigLIP（text + 3 view RGB，frozen）+ PointAttn（point cloud，trainable）+ MLP（proprio）→ token 序列 S_t；scene memory 是 voxelized 3D feature map，按 reconstruction-error 阈值 τ 触发更新；episodic memory 是窗口 L 的 FIFO；两路 cosine top-k 检索 + cross-attn 拼成条件 H_t，喂 base / arm 两个独立 diffusion denoiser
> - **结果**: RoboCasa 上 Manip+Nav avg SR 0.52 / mobile manip avg SR 0.31，分别比 π_0.5 baseline 高 +0.20 / +0.11；TidyBot++ 真机 6 任务 avg 0.44，π_0.5 0.33 / Diffusion Policy 0.32
> - **Sources**: [paper](https://arxiv.org/abs/2511.18112)
> - **Rating**: 2 - Frontier（当前 mobile-manip VLA 的 frontier 探索：explicit dual memory + per-part diffusion 的组合有新意且带诚实的 failure mode 分析，但未开源、memory 的因果贡献未干净隔离，尚未达到 foundation 级影响力）

**Key Takeaways:**
1. **Dual memory ≠ single buffer**: 把 "环境长期空间结构"（scene, voxel）和 "任务短期进度"（episodic, token FIFO）分开存、分开检索、分开融合，比 MemoryVLA 那种一锅端的隐式 perceptual cache 更解耦，主张是非 Markov 任务（"cabinet opened" vs "about to open"）需要 explicit time index
2. **Per-part diffusion**: 沿用 π_0.5 的 base / arm 解耦思路，但每路 denoiser 共享同一个 memory-augmented condition H_t，实验中比 π_0.5 在 mobile manip 上多 +0.11 SR
3. **MoMani benchmark**: 一个 MLLM-prompted 自动数据生成 pipeline（在线生成 + 硬质量门 + 离线词典序排序），产出 5K+ 仿真 nav-manip 联合轨迹，外加 1.2K TidyBot++ 真机 demo
4. **Failure mode 自爆**: 论文承认在 OR (open refrigerator) 上输给 baseline——动态遮挡（开门时几何剧变）让显式 voxel 3D memory 反而成累赘，依赖 implicit "muscle memory" 的方法此时更稳

**Teaser. EchoVLA 与既有 memory-aware mobile manip 方法（BSC-Nav、MemoryVLA）的设计对比。** EchoVLA 的关键差异是把 spatial 和 episodic 拆成两个独立 bank，分别走 coarse / fine cross-attn，最后融合后驱动 per-part diffusion。

![](https://arxiv.org/html/2511.18112v2/x1.png)

---

## 1. 问题与动机

### Mobile manipulation 的 non-Markov 困境

主流 OpenVLA / RT-2 系 VLA 都是 Markovian——`a_t = π(o_t, I)`，仅依赖当前观察。这在 table-top 短任务上 work，但 mobile manipulation 必须协调 navigation + manipulation，且：

- 视觉相似的两帧可能对应**完全不同的进度状态**（"cabinet opened" vs "about to open"）
- 跨房间长任务里，刚刚走过的房间布局对当前决策仍然相关，但已经不在视野内

需要某种 memory。论文的 framing 是从 declarative memory 的脑科学类比出发：parahippocampal cortex (PHC) 编码空间-语义结构，hippocampus 把上下文整合成时序索引的 episodic trace。

> ❓ 这种脑神经类比在 ML paper 里更多是 narrative 装裱，真正的 design choice 还是工程驱动的（voxel map + FIFO buffer）。读这类 paper 时建议直接看架构，不要被 PHC/hippocampus 这层叙事带偏对方法新颖性的判断。

### 与既有 memory-aware 方法的差异

- **BSC-Nav**：landmark memory + cognitive map → LLM-based reasoning 和 planning（高层）
- **MemoryVLA**：perceptual + cognitive memory → 直接条件 diffusion policy（低层），但 memory 是隐式 perceptual cache，没有 explicit 空间结构和 task-level experience
- **π_0.5**：base / arm per-part 分解，带 partial episodic memory，但缺 explicit scene-level memory
- **EchoVLA**：scene (voxel) + episodic (token FIFO) 两路独立存 + 两级 attention 融合，per-part diffusion 控制

---

## 2. 方法

### 2.1 架构总览

**Figure 2. EchoVLA 整体架构。** 多模态观测（多视角 RGB、点云、language、proprio）→ 统一 token 序列 → 经 episodic / scene memory 的 coarse + fine cross-attn 检索 → 拼合后作为 per-part diffusion policy 的 condition，分别生成 arm / base 动作。

![](https://arxiv.org/html/2511.18112v2/x2.png)

三个模块：multimodal state representation (3.2) → memory retrieval & interaction (3.3) → per-part diffusion (3.4)。

### 2.2 Problem Formulation

时刻 t，agent 观察多视角 RGB-D `O_t`、proprioceptive `s_t`、自然语言 instruction `I`，输出连续的 arm + base 动作：

$$
(a_{t}^{\text{arm}},\,a_{t}^{\text{base}})=\pi_{\theta}(\mathcal{I},\,\mathcal{O}_{1:t},\,\mathbf{s}_{1:t}).
$$

强调是 non-Markov：condition 是从 1 到 t 的整段历史而非单 frame。

### 2.3 Multimodal State Representation

每个模态独立 encoder：

- **Language**：SigLIP text tower（frozen）→ `L`
- **3 个固定相机的 RGB**：SigLIP vision tower（frozen），独立编码后 concat + 投影 → `V_t`
- **Depth → point cloud**：trainable PointAttn → `P_t`（提供 free-space、support surface、object boundary 等几何线索）
- **Proprio `s_t`**：MLP → `R_t`

拼成统一 token 序列：

$$
\mathbf{S}_{t}=[\mathbf{L},\ \mathbf{V}_{t},\ \mathbf{P}_{t},\ \mathbf{R}_{t}]
$$

`S_t` 既是 episodic memory 的 query，也是 diffusion policy 的 condition base。

> ❓ frozen SigLIP + trainable PointAttn 这种 "语义冻结、几何可学" 的组合很常见，但论文没说 RGB 与 PC token 之间是否有 cross-modal alignment loss——可能就是简单 concat，靠下游 cross-attn 自行学融合。

### 2.4 Scene Memory（PHC 类比）

**Figure 3. Voxelized 3D feature map 可视化。** 把多次 episode 的 depth 观测累积进 voxel 网格，再用 PointAttn 编码成 3D feature volume。

![](https://arxiv.org/html/2511.18112v2/x3.png)

形式：

$$
\mathbf{V}^{3D}_{t}\in\mathbb{R}^{X\times Y\times Z\times C}.
$$

**Discrepancy-driven 更新**：当前 voxel 特征 `V^3D_t` 与 memory 已有重建做对比，error 超阈值 `τ` 的区域才用新 PointAttn 特征覆盖，未变区域保留旧值。Inference 时也同样在线 refine，所以理论上能自适应环境变化（rearrangement）。

> ❓ "reconstruction" 怎么生成的没细写——猜是从 memory voxel 反向 decode 一组 PointAttn-comparable 特征做 cosine 比较。论文没给 decoder 结构，是 reproducibility 的洞。

### 2.5 Episodic Memory（hippocampus 类比）

短窗口 token buffer：

$$
\mathcal{M}^{\mathrm{epi}}=\{(\mathbf{S}_{t-k},\,t-k),\,\ldots,\,(\mathbf{S}_{t-1},\,t-1)\}
$$

固定大小 FIFO，保存原始 encoded token（不做 abstract summary），强调 "原始 token 才能保住 fine-grained 时序线索"。窗口 size `L` 由 capacity 决定，ablation 中 L=8 最优。

设计上和 scene memory 互补：scene 慢变 + 持久 + 空间结构；episodic 快变 + 短期 + 时序进度。

### 2.6 Memory Matching & Attention

两步式：cosine top-k 选 → cross-attn 交互。

**Scene memory（coarse）**：query 是当前 `V^3D_t`，从 `M^scene` 中选 top-k 相近的 stored map：

$$
\mathbf{Z}^{\mathrm{scene}}_{t}=\mathrm{CrossAttn}\bigl(\text{q}=\mathbf{V}^{3D}_{t},\;\text{k / v}=\mathcal{M}^{\mathrm{scene}}_{\mathrm{sel}}\bigr).
$$

**Episodic memory（fine）**：query 是 `S_t`，从 `M^epi` 中 match：

$$
\mathbf{Z}^{\mathrm{epi}}_{t}=\mathrm{CrossAttn}\bigl(\text{q}=\mathbf{S}_{t},\;\text{k / v}=\mathcal{M}^{\mathrm{epi}}_{\mathrm{sel}}\bigr).
$$

拼合：

$$
\mathbf{H}_{t}=\bigl[\mathbf{Z}^{\mathrm{scene}}_{t},\;\mathbf{Z}^{\mathrm{epi}}_{t}\bigr]
$$

`H_t` 即下游 diffusion 的 condition。

### 2.7 Per-part Diffusion Policy

base 和 arm 各一套 denoiser，共享 condition `H_t`：

$$
\epsilon_{\theta}^{(p)}=\text{Denoiser}_{p}(\mathbf{z}_{t},\mathbf{H}_{t},t),\quad p\in\{\text{base},\text{arm}\}
$$

训练目标是两路 denoising loss 之和：

$$
\mathcal{L}=\sum_{p\in\{\text{base},\text{arm}\}}\mathbb{E}_{t,\mathbf{z}_{t}}\Big[\|\epsilon-\epsilon_{\theta}^{(p)}(\mathbf{z}_{t},\mathbf{H}_{t},t)\|^{2}\Big].
$$

承袭 π0.5 的 per-part 解耦动机：base motion 和 arm manipulation 的动力学异质，强行共享 head 会互相干扰。

> ❓ Per-part 设计在 mobile manip 上越来越像标配了（π0.5、MoManipVLA、AC-DiT 都做这件事），但每家 condition 怎么 share / split 各有差异，是个值得整理的小 pattern。

---

## 3. MoMani Benchmark & Dataset

### 3.1 与既有 benchmark 的对比

**Table 1. MoMani vs 现有具身 benchmark。** Sub-tasks 是单 trajectory 内的子任务数；R/S 是 real / sim；Co-Gen 指 navigation + manipulation 联合自动生成；Mobile-M (Sim) 指 sim 中的 mobile manipulator；Real-Robot 指有实体机器人数据。

| Name | Sub-tasks | Real | Sim | Nav-gen | Manip-gen | Co-Gen | Mobile-M (Sim) | Real-Robot | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ManiSkill2 | 1-2 | - | ✓ | - | ✓ | - | ✓ | - | M |
| Social Nav | 1 | - | ✓ | ✓ | - | - | ✓ | - | N, M |
| HomeRobot | 3 | - | ✓ | ✓ | ✓ | - | ✓ | ✓ | N, M |
| ProcTHOR | 3 | - | ✓ | ✓ | ✓ | - | ✓ | - | N, M |
| Behavior-1K | >5 | - | ✓ | ✓ | ✓ | - | ✓ | - | N, M |
| Open X-Embodiment | 1-3 | ✓ | - | - | - | - | ✓ | ✓ | M |
| RoboCasa365 | 2-16 | - | ✓ | ✓ | ✓ | - | ✓ | - | N, M |
| InfinitedWorld | 3 | - | ✓ | ✓ | ✓ | ✓ | ✓ | - | N, M |
| **MoMani (Ours)** | 2-3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | N, M |

主张是只有 MoMani 同时勾上 Real + Sim + Co-Gen + Mobile-M (Sim) + Real-Robot。

> ❓ 表里 sub-tasks 一栏 MoMani 写 "2-3"，反而比 RoboCasa365 (2-16) 和 Behavior-1K (>5) 短，但被宣传为 "richer task coverage"——richness 应该用任务数和环境多样性而非 sub-task 长度衡量，此处 framing 略 sloppy。

### 3.2 仿真数据 pipeline

**Figure 5. Quality-Controlled Data Engine。** 两阶段——在线候选生成 + 离线挑选审计。

![](https://arxiv.org/html/2511.18112v2/x5.png)

**Stage I: Online Candidate Generation**
- MMLM prompter 处理任务规范，依次执行 Target-Aligned Sampling (L1)、Safety-Aware Navigation (L2)、Continuous Nav-Manip Stitching (L3)
- 引擎支持 base + arm 同时执行，用连续碰撞检测合成协调的 nav-manip 轨迹
- **Hard Quality Gates**：零碰撞、对齐精度（Δpos < 0.05 m、Δori < 5°）、100% task success → 进入候选池 `D_pool`

**Stage II: Offline Selection & Audit**
- Lexicographic Ranking (L4)：按 path length + planning cost 排序，取 top-K 作为专家级 `D_topK`
- Scene-Camera Audit 保证训练就绪
- 最终生成 5,000+ 多模态轨迹

### 3.3 Real-Robot 数据

基于 TidyBot++ 配置：Kinova Gen3 7-DoF 臂 + holonomic mobile base，前置 RGB-D + 顶视 stereo 相机，ROS 同步。Web teleop 30 Hz 录制，分割成 motion primitive，replay 验证成功的留下、失败丢弃。

### 3.4 Dataset Composition

**Figure 4. 仿真 + 真机数据分布。** Sim 包含 4 个 mobile manip 任务 (TOF/PnPS2C/PnPC2S/TOS) 和大量纯 navigation；Real-world 6 个 mobile manip 任务 (CM/OD/OR/PCIS/RK/EnP)。

![](https://arxiv.org/html/2511.18112v2/x4.png)

- **Sim (7,889 episodes)**：navigation-only 占 57.0%，PnPC2S/PnPS2C/TOS/TOF 各约 10.7-10.8%
- **Real-world (1,200 episodes)**：OR/CM/OD/PCIS 各 20.8%，EnP/RK 各 8.3%

---

## 4. 实验

### 4.1 Setup

仿真：在 RoboCasa simulator 上跑 multiple mobile manip 任务。真机：TidyBot++ 平台，7m × 7m arena，包含跨房间 EnP (Enter and Pick) 任务。8× A100 训练。

### 4.2 RoboCasa Simulator 主结果

**Table 2. RoboCasa 上 Manip / Nav 与 Mobile Manip 任务的 SR 对比。** Manip/Nav 评单一动作类型；Mobile Manip 强制 base + arm 协调。

| Category | Method | PnP C2S | PnP S2C | TOF | TOS | Nav only | Avg SR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Manip / Nav | BC-T | 0.04 | 0.46 | 0.33 | 0.45 | 0.10 | 0.28 |
|  | Diffusion Policy | 0.00 | 0.00 | 0.03 | 0.00 | 0.03 | 0.01 |
|  | DP3 | 0.03 | 0.25 | 0.43 | 0.26 | 0.13 | 0.22 |
|  | WB-VIMA | 0.00 | 0.12 | 0.11 | 0.19 | 0.31 | 0.15 |
|  | [[2504-Pi05\|π0.5]] | 0.18 | 0.20 | 0.44 | 0.52 | 0.24 | 0.32 |
|  | **EchoVLA** | **0.21** | **0.68** | **0.51** | **0.67** | **0.51** | **0.52** |
| Mobile Manip | BC-T | 0.00 | 0.11 | 0.04 | 0.04 | – | 0.05 |
|  | DP3 | 0.00 | 0.10 | 0.08 | 0.04 | – | 0.06 |
|  | WB-VIMA | 0.00 | 0.12 | 0.11 | 0.19 | – | 0.11 |
|  | [[2504-Pi05\|π0.5]] | 0.08 | 0.25 | 0.18 | 0.27 | – | 0.20 |
|  | **EchoVLA** | **0.17** | **0.34** | **0.29** | **0.43** | – | **0.31** |

观察：

1. Diffusion Policy 在这套 setup 下几乎全零——可能 vanilla DP 的 visuomotor 表征不足以处理 RoboCasa 的多视角 + 跨场景泛化
2. EchoVLA 在 PnP S2C 上跳到 0.68（baseline 最高 0.46）幅度异常大，暗示 spatial memory 对 "Sink → Counter" 这种长程参照变换帮助显著
3. 从 Manip/Nav 0.52 掉到 Mobile Manip 0.31，相对降幅 -40%，说明就算有 memory，"同时动 base + arm" 仍是真正的瓶颈

### 4.3 Ablation

**Table 3. 观测模态 + memory 模块 ablation。** M = mobile 变体的 SR；S = stationary 变体的 SR。

|  | RGB | PC | EM | SM | M | S |
| --- | --- | --- | --- | --- | --- | --- |
| (a) | ✗ | ✓ | ✓ | ✓ | 0.02 | 0.13 |
| (b) | ✓ | ✗ | ✓ | ✓ | 0.08 | 0.15 |
| (c) | ✓ | ✓ | ✓ | ✗ | 0.09 | 0.16 |
| (d) | ✓ | ✓ | ✗ | ✓ | 0.14 | 0.13 |
| Ours | ✓ | ✓ | ✓ | ✓ | 0.17 | 0.21 |

读法：

- **(a) 去 RGB**：mobile 0.02、stationary 0.13——RGB 损失对 mobile 远比 stationary 致命，因为 mobile 任务依赖语义识别远处目标
- **(b) 去 PC**：M 0.08、S 0.15——少了几何线索，grasp 精度全面下降
- **(c) 去 SM**：M 0.09 vs full 0.17（-0.08）——scene memory 在 mobile 场景下贡献最大
- **(d) 去 EM**：M 0.14 vs full 0.17（-0.03），但 S 0.13 vs full 0.21（-0.08）——episodic memory 有趣地在 stationary 场景反而更关键，可能是因为 stationary 任务更细粒度时序敏感（grasp 阶段衔接），mobile 场景的 SM 贡献掩盖了 EM
- 两路都去会进一步退化

> ❓ Ablation 没给 "EM + SM 都关、只剩 RGB+PC" 的 baseline，无法直接量化 memory 总贡献。从 (c)+(d) 推断 memory 总贡献约 +0.03 ~ +0.08，相对 baseline π_0.5 的 +0.11 增益，意味着 per-part diffusion + 数据集等其他变量也吃掉了不少 gain，单纯 memory 的因果效应没那么戏剧性。

**Table 4. EM 窗口 size L 与 SM 更新阈值 τ 的敏感性（PnPC2S Mobile）。**

| Window Size (L) | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| SR ↑ | 0.08 | 0.12 | 0.17 | 0.15 |

| Threshold (τ) | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| SR ↑ | 0.11 | 0.14 | 0.17 | 0.13 |

L 太小（2/4）"plan forgetting"；L 太大（16）latency 增加且收益递减。τ 太低导致 representation 不稳；τ 太高 memory 过时。

### 4.4 真机实验

**Figure 6. EchoVLA 真机 rollouts。** 完成 4 个 mobile manip 任务：关微波炉、把杯子从架子放到水槽、开抽屉、开冰箱。

![](https://arxiv.org/html/2511.18112v2/x6.png)

**Table 5. 真机 6 任务 SR。** 每任务 20 次随机起始位姿试验。

| Method | OD | CM | PCIS | OR | RK | EnP | Avg SR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [[2504-Pi05\|π0.5]] | 0.35 | 0.55 | 0.20 | 0.50 | 0.40 | 0.00 | 0.33 |
| Diffusion Policy | 0.40 | 0.60 | 0.50 | 0.30 | 0.10 | 0.03 | 0.32 |
| **EchoVLA** | **0.45** | **0.70** | 0.50 | 0.40 | **0.50** | **0.10** | **0.44** |

亮点：

- **CM 0.70 / RK 0.50** 上明显甩开 baseline；论文归因 "EM 作为时序锚点弥补 SM 中 voxel ghosting"
- **EnP 0.10**（vs baseline 0.00 / 0.03）：跨房间长程任务上虽然绝对值低但唯一非零
- **OR 0.40 < π_0.5 0.50**：作者诚实地分析 failure——开冰箱门时几何剧变让 explicit 3D scene memory 失效，而 baseline 靠 implicit "muscle memory" 反而更鲁棒。这是一个有意思的 explicit vs implicit memory trade-off

---

## 关联工作

### 基于
- π0.5：per-part (base/arm) 分解 + partial episodic memory 的直接前身，EchoVLA 在此基础上引入 explicit scene + episodic dual memory
- SigLIP：frozen text + vision encoder
- PointAttn：trainable 3D 表示

### 对比
- **MemoryVLA**（concurrent）：单 perceptual + cognitive memory 的隐式 cache，没有 explicit 空间结构和 task-level experience。EchoVLA 主张 explicit 拆分更适合 mobile manip
- **BSC-Nav**：landmark + cognitive map 给 LLM 做 high-level reasoning，而 EchoVLA 的 memory 直接条件 low-level diffusion policy
- OpenVLA / RT-2：纯 Markovian VLA，作为 "无 memory 基线" 的代表
- AC-DiT：另一种 base-arm coordination 的 adaptive diffusion 方案
- MoManipVLA：另一篇 mobile-manip-VLA 工作，可对比设计

### 方法相关
- **Diffusion Policy**：作为 single-head diffusion baseline 出现
- **DP3**：3D diffusion policy，被 EchoVLA 全面超越
- **WB-VIMA**：whole-body VIMA 类方法
- TidyBot / TidyBot++：硬件平台
- **RoboCasa**：仿真 benchmark
- Behavior-1K / ManiSkill2 / HomeRobot / ProcTHOR：MoMani 的 benchmark 对比对象

---

## 论文点评

### Strengths

1. **Memory 解耦的设计动机清晰**：把 "环境长期空间结构 vs 任务短期时序" 这两类性质截然不同的信息分开存、分开检索，有明确的 inductive bias 支撑（不是单纯堆参数）。Ablation 也部分验证了两路确有互补性
2. **诚实承认 failure mode**：OR 任务输给 baseline 时直接归因到 "explicit 3D memory 在动态遮挡下成累赘"，而非掩盖。这种 explicit vs implicit memory trade-off 的观察其实比 main result 更有 insight
3. **真机 EnP 任务**：跨房间长程 mobile manip，baseline 全军覆没，EchoVLA 0.10——绝对值不高但说明这条路至少能 non-trivial 地启动这类任务
4. **MoMani 的 nav-manip co-gen pipeline**：显式做 "base + arm 同时执行 + 连续碰撞检测" 来生成协调轨迹，比 stitching 离散 skill 的做法更自然

### Weaknesses

1. **没开源**：截至当前没有 code / project page / weight，整个 pipeline 的超参（discrepancy threshold τ、L、PointAttn 配置、diffusion 超参）只有 ablation 里的几个数字，复现成本极高
2. **Memory 总贡献可能被 oversell**：从 ablation (c)(d) 推断 EM+SM 联合贡献约 +0.03~+0.08，但 vs π_0.5 baseline 的 +0.11 gain 中至少一半可能来自 PointAttn / per-part diffusion / 训练数据差异。论文没有一个 "完全去 memory、其他全留" 的 baseline
3. **Scene memory update rule 描述不充分**：reconstruction 怎么生成、threshold 怎么校准、跨 episode 累积时如何处理坐标对齐（odometry drift？），都缺细节。论文末尾 limitations 直接承认 odometry drift 会导致 voxel ghosting，但没量化
4. **MoMani 的 sub-task 长度只有 2-3**：和 Behavior-1K (>5) / RoboCasa365 (2-16) 比短得多，论文却 frame 成 "richer task coverage"，是 spin
5. **脑科学类比的 narrative 重于 technical justification**：PHC + hippocampus 只是装饰——voxel map + FIFO buffer 的设计完全可以从工程动机推出，无需脑科学

### 可信评估

#### Artifact 可获取性

- **代码**：未开源（截至 v2，无 GitHub / project page 链接）
- **模型权重**：未发布
- **训练细节**：仅高层描述（8× A100、frozen SigLIP、trainable PointAttn、per-part diffusion，未给完整超参 / 训练步数 / 数据配比）
- **数据集**：MoMani 未说明是否会发布；real-robot 数据更不太可能开放

#### Claim 可验证性

- ✅ **EchoVLA SR 在论文 setup 下高于 π_0.5 baseline**：仿真 +0.20 / +0.11，真机 +0.11 avg；3 random seeds × 50 episodes 报告，比许多论文严谨
- ✅ **OR 任务 EchoVLA 输给 π_0.5**：作者主动汇报，反 cherry-pick
- ⚠️ **"synergistic memory" 的因果贡献**：ablation 设计未隔离 memory vs 其他变量，归因不严
- ⚠️ **MoMani "richer task coverage"**：sub-task 长度比 Behavior-1K / RoboCasa365 短，"richer" 是定性 claim
- ⚠️ **"neuro-inspired" framing**：脑科学类比对 design choice 的指导价值未验证，更像 narrative wrapping

### Notes

- **Memory pattern 整理**：mobile-manip 领域的 memory 设计目前可分三派——（i）implicit perceptual cache（MemoryVLA）；（ii）explicit landmark + cognitive map for LLM planning（BSC-Nav）；（iii）explicit dual memory + low-level policy condition（EchoVLA）。值得做一篇小调研梳理 memory granularity / update rule / retrieval mechanism / 接入位置 这几个维度
- **OR failure 是个真问题**：explicit 3D memory 在动态几何变化下劣于 implicit，提示我们 memory 系统应该有某种 "动态区域置信度衰减" 机制，或者用 hybrid（explicit static + implicit dynamic）。这可能是个 follow-up idea
- **per-part diffusion 在 mobile manip 已经是 default 了**：π0.5、MoManipVLA、AC-DiT、EchoVLA 都用这个思路。下一个问题应该是 "怎么在 part 之间显式建模 coupling"（base 速度直接影响 arm 末端）而非简单共享 condition
- **"5K+ trajectory" 不算大数据**：相比 Behavior-1K 或 OXE，MoMani 的规模其实有限，但 nav-manip co-gen 的 pipeline 本身可能比数据更有价值

### Rating

**分数**：2 - Frontier
**理由**：按笔记 Strengths 段的判断，EchoVLA 在 mobile-manip VLA 方向上提出了 explicit dual memory + per-part diffusion 的新组合，并带有反 cherry-pick 的 failure mode 分析，是当前 frontier 上值得对比的 baseline 候选；但还达不到 foundation 档——Weaknesses 明确指出方法未开源、memory 的因果贡献未被干净隔离，且作为 2025-11 的 arXiv preprint 尚无社区采纳或 baseline 化的证据；也不是 Archived，因为 memory-aware mobile-manip 仍是当前活跃方向，而非被后续工作取代的 incremental 变体。
