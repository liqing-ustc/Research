---
title: "SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation"
authors: [Ruisen Tu, Arth Shukla, Sohyun Yoo, Xuanlin Li, Junxi Li, Jianwen Xie, Hao Su, Zhuowen Tu]
institutes: [UC San Diego, Lambda Inc.]
date_publish: 2026-03-24
venue: arXiv
tags: [VLA, mobile-manipulation, imitation-learning]
paper: https://arxiv.org/abs/2603.22760
website:
github:
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation
> - **核心**: 在 mobile manipulation 场景下，把"直接 imitation 一个 13D action"换成"action + 5 类 spatial 辅助任务 co-training"，并配合多视角 RGB+Depth 输入，把 ManiSkill-HAB 平均成功率从 0.60 抬到 0.73。
> - **方法**: 1.3B 的 Prismatic VLM（DINOv2+SigLIP+Qwen2.5-0.5B）+ 5 个 auxiliary decoder（global pos / qpos / grasp / object pose / seg mask）+ 三阶段训练（先 freeze backbone 训 decoder，再联合微调，最后单独训 flow matching action expert）。
> - **结果**: ManiSkill-HAB 上 SG-VLA 0.73 vs. 直接 IL baseline 0.60；OpenVLA 7B 单视角 RGB 仅 0.04，加 multi-view+depth 后到 0.32。
> - **Sources**: [paper](https://arxiv.org/abs/2603.22760)
> - **Rating**: 2 - Frontier（mobile manipulation VLA 新兴方向下较完整的 systems-level ablation，progressive co-training recipe 有通用价值；但方法是 known recipe 组合、无 real robot、无 code、benchmark 自家主导，不构成 foundational 贡献）

**Key Takeaways:**
1. **Auxiliary co-training as dense supervision**: 在 13D 高维 action 上，单纯 BC 信号太稀疏；让共享 backbone 同时预测 robot global pos / qpos / grasp / object pose / seg mask，这五类 "interpretable intermediate signals" 提供了 dense 的 spatial 监督。+22% 平均成功率。
2. **Naive co-training 会掉点，progressive training 是必须的**: 随机初始化的 decoder 直接和 backbone 联合训，会拉坏 pretrained representation（0.60 → 0.51）。先 freeze gradient flow 训 decoder 适配 backbone 表示，再放开联合微调，才能拿到 0.73。这个 finding 比方法本身更通用。
3. **Multi-view + Depth 是 mobile manipulation 的硬需求**: 单视角 RGB 在 ManiSkill-HAB 上几乎完全 fail（OpenVLA 0.04）。加 hand camera + depth 后 8x 提升。但加 4 步 history 反而掉点（0.60 → 0.49），temporal context 在 reactive 任务里收益不显著。
4. **Flow matching action head 的收益是 task-dependent 的**: Pick 翻倍（0.13→0.27）、Place 提升（0.70→0.80），但 Open 系列任务掉点明显（0.87→0.76, 0.77→0.60）。作者解释为 "FM 擅长精细操作但不擅长含 base motion 的 mobility 控制"——mobility 用 discrete token 反而更稳。
5. **VLA 在 mobile manipulation 仍是 nascent area**: 作者明确承认 baseline 选择有限，所有数字都来自 ManiSkill-HAB sim，没有 real robot validation。

---

## 1. Problem Setup

VLA 在 tabletop manipulation 上已经做出了一系列 [[2406-OpenVLA|OpenVLA]] / [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]] 等成果，但 mobile manipulation——需要 base + arm + gripper 协调的 13 维 action 空间——上 standard VLA 直接 imitation 几乎完全 fail。作者的核心 hypothesis：

> 直接 imitation learning（从 visual-language representation 预测高维 action vector）提供的监督信号太稀疏，模型学不到 mobile manipulation 需要的 multi-faceted 理解。

解法是两条："enrich sensory input"（multi-view RGB + depth + history）和 "dense auxiliary supervision"（5 类 decoder co-training）。

## 2. Architecture

**Figure 2. SG-VLA Architecture Overview.** 多模态输入（head/hand 双摄 RGB + normalized depth + 文本指令）经过 Prismatic VLM backbone，latent 一方面送给 5 个 auxiliary decoder，一方面送给 action prediction head（discrete token 或 flow matching action expert 二选一）。

![](https://arxiv.org/html/2603.22760v1/x1.png)

**Backbone（1.3B 总参数）**：
- **Visual encoder**: dual-encoder = DINOv2（spatial 强）+ SigLIP（semantic 强），沿用 [[2406-OpenVLA|OpenVLA]] 的设计。
- **LLM**: Qwen2.5-0.5B —— 比 OpenVLA 的 7B 小一个数量级，但实验里反而效果更好（见后）。
- **Projector**: 把 vision feature 映到 language embedding 空间。

**Optional action expert（100M）**: Flow Matching action head，沿用 [[2410-Pi0|π0]] 的设计，从 backbone 的 frozen latent denoise 出连续 action。

**Action 空间（13D）**：
- ΔX：3D base pose（position + orientation）
- Δz：1D 躯干高度
- Δq：7D 机械臂关节角
- ΔG：2D 夹爪状态

## 3. Input Modalities

> ❓ 论文没有说每个 view 用了多少 token，也没说 depth 是和 RGB 共享 encoder 还是单独走。后者对 effective context length 影响很大。

3 类输入增强分别 ablate：

1. **Multi-view RGB**: head + hand camera。
2. **Depth**: head + hand 的 depth 都用 Eq. 1 归一化。

**Equation 1. Depth normalization.**

$$
p_{obs}=1-\tanh\left(\frac{\text{depth value}}{1000}\right)
$$

**符号说明**：depth value 单位是 mm；tanh 把范围压到 (0,1)，且对近处物体梯度大、远处饱和。
**含义**：把 unbounded depth map 转成与 RGB 同 scale 的输入，便于和 RGB 共用 patch embedding。

3. **Temporal history**: 过去 4 步观测。

## 4. Auxiliary Decoders

5 类 decoder 都从 backbone 的 shared latent 出发，预测不同 spatial signal：

**Table 1. Decoder specifications.**

| Decoder | Type | Key Components |
| ---- | ---- | ---- |
| Global Pose (2D x,y) | MLP | Proj (896→512), 3-layer MLP, Avg Pooling |
| Grasp Success (binary) | MLP | Proj (896→512), 3-layer MLP, Avg Pooling |
| Object Pose (3D pos + quat) | MLP | Proj (896→512), 3-layer MLP, Quat Norm |
| Joint Pose (12D qpos) | Transformer | 12 mask tokens, 2-layer Transformer, sin-cos PE |
| Mask (128×128) | CNN | 4-stage Transpose Conv, BatchNorm, GELU |

**为什么这样选架构**：低维 regression / 分类用 MLP；joint angle 之间有 kinematic 依赖，用 Transformer 让 mask token 互相 attend；segmentation 用 CNN upsample。

**Equations 2-4. Loss functions.**

$$
\mathcal{L}_{pos}=\text{MSE}(\hat{\mathbf{p}},\mathbf{p}),\quad \mathcal{L}_{grasp}=\text{CrossEntropy}(\hat{y},y)
$$

$$
\mathcal{L}_{obj}=\|\hat{\mathbf{t}}-\mathbf{t}\|_{2}^{2}+(1-|\hat{\mathbf{q}}\cdot\mathbf{q}|)
$$

$$
\mathcal{L}_{qpos}=\text{MSE}(\hat{\mathbf{J}},\mathbf{J}),\quad \mathcal{L}_{seg}=\text{CrossEntropy}(\hat{\mathbf{M}},\mathbf{M})
$$

**Total**: $\mathcal{L} = \mathcal{L}_{action} + \lambda_{pos}\mathcal{L}_{pos} + \lambda_{grasp}\mathcal{L}_{grasp} + \lambda_{qpos}\mathcal{L}_{qpos} + \lambda_{obj}\mathcal{L}_{obj} + \lambda_{seg}\mathcal{L}_{seg}$，其中 $\lambda_{grasp}=5.0$，其余为 1.0。

## 5. Multi-Stage Training

**Figure 3. Multi-stage training scheme.**

![](https://arxiv.org/html/2603.22760v1/x2.png)

**Preliminary finding**: 作者一开始尝试随机初始化 decoder + 直接联合训，发现 backbone 表示被搞坏了——untrained decoder 给出 noisy gradient 污染 pretrained representation。这是后续 progressive training 的动机。

- **Stage 1: Decoder Adaptation** (3 epochs)：阻断 auxiliary decoder 到 backbone 的梯度，只让 action token loss 更新 backbone；decoder 学着从 fixed latent 解出 spatial signal。
- **Stage 2: Joint Refinement** (7 epochs)：放开全部梯度，auxiliary loss 也回传到 backbone，引导 backbone 学到更 spatial-aware 的 representation。
- **Stage 3: Action Head Training**：完全 freeze backbone，只训 flow matching action head。作者发现 FM denoising loss 和别的 loss 一起训不收敛，所以隔离训。

> ❓ Stage 3 让 backbone 完全 frozen 训 FM head——这意味着 FM expert 看到的 latent 只是为 discrete token prediction 优化过的，没有 hint 它该 condition continuous action。这可能正是 Open task 加 FM head 反而掉点的原因。

## 6. Experiments

### 6.1 Setup

- **Benchmark**: ManiSkill-HAB（ICLR 2025），3 个 long-horizon 任务（TidyHouse / PrepareGroceries / SetTable）拆成 4 类 subtask（Pick / Place / Open / Close）。Sim only。
- 数据规模：44K episodes / 1.4M transitions（global pos / grasp / qpos 用 SetTable subset 8K/240K；seg / obj_pos 用全部 pick&place 40K/1.2M）。
- **训练**: 8× A100，Adam，lr=2e-5，global bs=512。带 seg decoder 的模型 5 epochs，否则 10 epochs。
- **评测**: 每个任务 30 episodes 的成功率均值。

### 6.2 Input Modality Ablation

**Table 2. Input modality ablation.** Base VLM = DINOv2+SigLIP + Qwen2.5-0.5B。

| Method | Pick | Place | Open-Fr | Open-Dr | Close-Fr | Close-Dr | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA (7B, single-view RGB) | 0.00 | 0.19 | 0.02 | 0.00 | 0.00 | 0.04 | 0.04 |
| OpenVLA + Multiview | 0.06 | 0.35 | 0.14 | 0.38 | 0.00 | 0.53 | 0.24 |
| OpenVLA + Multiview + Depth | 0.12 | 0.41 | 0.43 | 0.30 | 0.00 | 0.67 | 0.32 |
| Base VLM + Multiview | 0.06 | 0.53 | 0.60 | 0.30 | 0.63 | 0.93 | 0.52 |
| **Base VLM + Multiview + Depth** | **0.16** | **0.56** | **0.67** | **0.36** | **0.83** | **1.00** | **0.60** |
| + History (4 steps) | 0.00 | 0.47 | 0.57 | 0.40 | 0.47 | 1.00 | 0.49 |

观察：
- OpenVLA 单视角 RGB 本质上完全 fail（0.04），表示 standard VLA 在 mobile manipulation sim 上没有意义可言。
- Multiview 是最大的 single bump（0.04→0.24 on OpenVLA, 0.32→0.52 on Base VLM）。
- Qwen2.5-0.5B 比 OpenVLA 的 7B LLM backbone 在同样输入下要好很多（0.32→0.60）——0.5B vs 7B 反而胜出，这点很反直觉。作者归因到 "更高效"，但没深究。
- History 反而掉点（0.60→0.49）。作者推测 "tasks are sufficiently reactive"。

> ❓ "更小的 LLM 更好" 是在限定的训练量内的现象（10 epochs / 240K 数据），是否在更大数据量上也成立？还是说 OpenVLA 的 7B 在 240K 上 underfit 了？

### 6.3 Auxiliary Task Ablation

**Table 3. Auxiliary co-training ablation on SetTable.** SG-VLA = Base VLM + Multiview + Depth。

| Progressive | Method | Pick | Place | Open-Fr | Open-Dr | Close-Fr | Close-Dr | Avg |
|---|---|---|---|---|---|---|---|---|
| No | SG-VLA (no aux) | 0.16 | 0.56 | 0.67 | 0.36 | 0.83 | 1.00 | 0.60 |
| **No** | **+ all aux (naive)** | 0.03 | 0.50 | 0.60 | 0.23 | 0.67 | 1.00 | **0.51** |
| Yes | + is_grasped | 0.30 | 0.53 | 0.83 | 0.57 | 0.80 | 0.93 | 0.66 |
| Yes | + qpos | 0.23 | 0.67 | 0.87 | 0.70 | 0.90 | 0.90 | **0.71** |
| Yes | + global pos | 0.07 | 0.27 | 0.90 | 0.70 | 0.97 | 1.00 | 0.65 |
| **Yes** | **+ all (progressive)** | **0.13** | **0.70** | **0.87** | **0.77** | **0.90** | **1.00** | **0.73** |

**关键 finding**：
- Naive co-training 反而掉点（0.60→0.51）——验证了 stage-1 freeze 的必要性。
- qpos（关节角度）是单个 aux 中收益最大的（+0.11），尤其 Place 和 Open-Drawer。
- global pos 在 navigation-heavy 的 Open/Close 上很强（Open-Fr 0.90），但破坏了 Pick/Place（0.27）。这是个值得注意的 negative interaction。
- 全部组合（progressive）拿到 0.73，比最强单 aux（qpos 0.71）只好 +0.02——边际收益其实不大。

**Table 4. Seg + obj_pose 加进来后的 cross-task 收益**（在 pick&place 数据上）。

| Method | SetTable Pick | SetTable Place | PG Pick | PG Place | TH Pick | TH Place | Avg |
|---|---|---|---|---|---|---|---|
| SG-VLA | 0.16 | 0.56 | 0.10 | 0.33 | 0.07 | 0.40 | 0.27 |
| + seg + obj_pos | **0.26** | **0.78** | **0.13** | **0.60** | **0.33** | **0.73** | **0.47** |

### 6.4 Action Head

**Table 5. Discrete vs flow matching head（基于 SG-VLA + all aux）。**

| Method | Pick | Place | Open-Fr | Open-Dr | Close-Fr | Close-Dr | Avg |
|---|---|---|---|---|---|---|---|
| SG-VLA (discrete tokens) | 0.13 | 0.70 | **0.87** | **0.77** | **0.90** | **1.00** | **0.73** |
| SG-VLA + flow matching head | **0.27** | **0.80** | 0.76 | 0.60 | 0.76 | 0.97 | 0.69 |

- FM head 的 chunk size = 8，执行前 2 步；10 步 denoising。
- Pick / Place 提升明显（manipulation 部分受益于 continuous 控制）。
- Open / Close 大幅掉点（base motion 在 discrete token 下更稳）。
- Net effect 是负的（0.73→0.69），所以作者推 "task-adaptive action generation"——根据任务选择 discrete or continuous。

> ❓ 这个 "dichotomy" 的解释（FM 不擅长 mobility）值得怀疑。stage 3 完全 freeze backbone 训 FM head 的设计本身就限制了 FM head 能拿到的 base motion 信息。如果 stage 3 也允许 partial unfreeze，结果可能不一样。

**Figure 4. Sample execution trajectories.** 6 个 ManiSkill-HAB 任务的 rollout，大图是全局视角，小图是 head depth/RGB + hand depth/RGB。

![](https://arxiv.org/html/2603.22760v1/figures/stacked_strips.jpg)

---
## 关联工作

### 基于
- [[2406-OpenVLA|OpenVLA]]: 用了它的 dual visual encoder（DINOv2 + SigLIP）思路，并作为主要对比 baseline
- [[2410-Pi0|π0]]: Flow matching action expert 的设计直接沿用
- Prismatic VLM: backbone（Qwen2.5-0.5B 版本）
- ManiSkill-HAB: benchmark + 数据生成 pipeline（作者团队自己的工作）

### 对比
- 直接 IL on same architecture: 主要的 method 内 baseline，证明 aux 训练的增益
- [[2406-OpenVLA|OpenVLA]]：唯一的外部 VLA baseline

### 方法相关
- [[2504-Pi05|π0.5]]: 同类 mobile manipulation VLA，论文引用但未对比
- [[2401-MobileALOHA|Mobile ALOHA]]: 双臂 mobile manipulation，论文引用为 mobile manipulation 难度依据
- [[2412-NaVILA|NaVILA]]: legged robot VLA，引用为相关工作
- Hierarchical mobile manipulation systems（VLM 当 high-level planner + 专门 nav/manip policies）：作为对比的 paradigm

---
## 论文点评

### Strengths

1. **Auxiliary co-training 的 progressive training 这个 finding 是 generalizable 的**。"先 freeze backbone 训 decoder，再联合微调" 这个 recipe 在任何 multi-task add-on 场景都可能适用。0.60 → 0.51 vs. 0.60 → 0.73 的对比强度足够。
2. **Ablation 做得相对完整**：input modality / 单 aux task / aux task 组合 / action head 四组实验都有，能让读者大致 attribute 到每个 design choice 的贡献。
3. **Negative result 也写出来了**：history 掉点、FM head 在 Open 上掉点、global pos 破坏 pick&place——这些没有藏起来。

### Weaknesses

1. **完全没有 real robot 验证**。Sim-to-real 是 mobile manipulation 最关键的瓶颈，全部数字都来自 ManiSkill-HAB sim，整体可外推性受限。
2. **Baseline 太弱**。作者自己承认 "established baselines 有限"，但只对比 OpenVLA + 自己的 IL 变体，没有跟 [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]] / [[2401-MobileALOHA|Mobile ALOHA]] 等 mobile manipulation 直接相关的工作做对比。0.73 这个数字在 ManiSkill-HAB benchmark 自身的 leaderboard 里位置不清楚。
3. **"为什么 Qwen2.5-0.5B 比 OpenVLA 7B 好" 没有深入**。这是一个反直觉的现象——0.5B vs 7B 在同样输入下更强——值得展开，但论文只用一句 "more efficient" 带过。
4. **Auxiliary task 的设计是 sim-specific 的**。global pos、qpos、object pose、seg mask 这些 ground truth 在 sim 里免费，但在 real world 需要额外 instrumentation 或 pseudo-label。论文没讨论真实场景如何获得这些 supervision。
5. **方法本质是 known recipe 的组合**：multi-task co-training 不新（CV 一直在做），depth/multi-view 也不新（[[2502-OpenVLA-OFT|OpenVLA-OFT]] 等都在做），auxiliary head 也不新。SG-VLA 的核心贡献是 "在 mobile manipulation 这个具体场景下把它们调通了"，更接近 systems paper 而非 conceptual contribution。
6. **没有 code release**。复现需要从头搭。

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源（论文未提供 GitHub 链接，Web 搜索亦未找到）
- **模型权重**: 未发布
- **训练细节**: 仅高层描述（提供了 lr / batch size / epoch / λ 系数，但没有完整 hyperparameter 表，supplementary 提到了 decoder 详细结构表）
- **数据集**: 基于公开的 ManiSkill-HAB；auxiliary annotation 是作者团队自己生成，未提及发布

#### Claim 可验证性
- ✅ "SG-VLA 0.73 vs IL 0.60 on ManiSkill-HAB"：sim 实验数字明确，table 完整，原则上可复现（但缺 code 增加门槛）
- ✅ "Naive co-training 掉点 0.60→0.51"：这个 finding 的 evidence 直接来自 Table 3
- ✅ "Multi-view + depth 大幅提升"：Table 2 跨多个 backbone 一致
- ⚠️ "Qwen2.5-0.5B 优于 OpenVLA 7B"：只在 ManiSkill-HAB 上测，可能是 underfit 7B 的 artifact，没有用同样数据量微调 OpenVLA 7B 的对照
- ⚠️ "FM head 在 mobility 任务掉点是因为 base motion 不适合 continuous action"：作者给出的解释是事后合理化，没有控制变量实验（例如把 FM head 限制在 manipulation 子动作）
- ⚠️ "history 掉点是因为 task 足够 reactive"：另一个事后解释，可能是 history encoding 设计本身的问题
- 无明显 ❌：论文 tone 整体诚实，没有 marketing 修辞

### Notes

- 这篇论文的位置：在 mobile manipulation VLA 这个新兴方向上做了一篇相对完整的 systems-level study，验证了 "auxiliary spatial supervision + multi-modal input" 这个 recipe。但因为没有 real robot、baseline 弱、code 不开源、benchmark 自己人主导，对外部 lab 的影响力大概率有限。
- 我的复用价值：**progressive multi-task training 的 recipe**（先 freeze backbone 训 head，再联合微调）值得记下来——这在我自己设计 multi-task VLA 时几乎一定会用到。其他 finding（multi-view + depth 大于 single RGB、history 不一定有用）属于 "已知 prior 的额外 confirmation"。
- ❓ Open question：auxiliary signals 在 real-world 怎么获得？sim 里 free，real 里需要额外 instrumentation 或 self-supervised 方式获取（比如用 SAM 出 mask、用 odometry 出 global pos）。这是把这套方法搬到真实场景的最大障碍。

### Rating

**分数**：2 - Frontier
**理由**：方法上是 multi-task co-training + multi-view/depth + FM head 等 known recipe 在 mobile manipulation 场景的系统化组合，不构成 foundational 贡献（见 Weakness 5）；但 progressive training recipe（naive 0.51 vs progressive 0.73）是可推广的 finding，且 mobile manipulation VLA 目前仍是 nascent area、baselines 稀缺（Weakness 2），这篇属于该方向当前前沿参考，所以不是 1 - Archived。距 3 - Foundation 还差 real robot 验证、code release 和独立 lab 的采纳证据——arXiv 2026-03 新文，尚未观察到社区 baseline 化迹象。
