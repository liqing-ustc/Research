---
title: "IRASim: A Fine-Grained World Model for Robot Manipulation"
authors: [Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, Tao Kong]
institutes: [HKUST, ByteDance Seed]
date_publish: 2024-06-20
venue: ICCV 2025
tags: [world-model, manipulation, diffusion-policy]
paper: https://arxiv.org/abs/2406.14540
website: https://gen-irasim.github.io/
github: https://github.com/bytedance/IRASim
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] IRASim: A Fine-Grained World Model for Robot Manipulation
> - **核心**: 训练一个 trajectory-to-video 的 diffusion transformer 作为 fine-grained robot manipulation world model，要求每一帧画面与 trajectory 中对应 timestep 的 action 严格对齐。
> - **方法**: 在 DiT block 内引入 **frame-level action conditioning**——把每个 action 单独编码成 embedding，分别 modulate 对应 frame 的 spatial-attention LayerNorm 的 scale/shift 参数；temporal block 用 video-level shared embedding。VAE latent 空间扩散，autoregressive rollout 出长视频。
> - **结果**: RT-1/Bridge/Language-Table/RoboNet 上 latent L2 与 PSNR 全面优于 LVDM/VDM/iVideoGPT/MaskViT；与 LIBERO ground-truth simulator 在 policy success rate 上 Pearson 0.99；用作 ranking-based planner 把 Push-T 上 vanilla diffusion policy IoU 从 0.637 拉到 0.961。
> - **Sources**: [paper](https://arxiv.org/abs/2406.14540) | [website](https://gen-irasim.github.io/) | [github](https://github.com/bytedance/IRASim)
> - **Rating**: 2 - Frontier（ICCV 2025 被接收、bytedance 开源、被 WMPO 等后续工作作为 baseline/reference；frame-level action conditioning 是 trajectory-to-video 方向的代表设计，但尚未成为方向 de facto 基础）

**Key Takeaways:**
1. **Frame-level action conditioning**：把 trajectory 当成 frame-wise 信号（而非 video-level 的 single embedding）注入 DiT 的 AdaLN，是把通用 text-to-video DiT 改造成精确 action-conditioned world model 的最小且关键的改动。
2. **World model 的两个独立用途同时验证**：(a) policy evaluation——与 ground-truth simulator 评估结果 Pearson 0.99；(b) model-based planning——配合 reward / goal-image 做 trajectory ranking，IoU 0.637 → 0.961。
3. **Test-time scaling for manipulation**：sampled trajectory 数 K↑、post-trained rollouts 数 P↑ 都能稳定推动 policy performance，把 inference-time compute 投入显式拉到 manipulation 这个之前几乎没人讨论的领域。
4. **Action chunking 友好**：与现代 diffusion policy 输出 trajectory 而非 single action 的范式天然匹配——一次 forward 出整个 chunk 对应的视频。

**Teaser. IRASim 概览：以 1 张历史 observation + 一段 action trajectory 为条件，生成 fine-grained 模拟 robot-object 交互的视频。**

![](https://arxiv.org/html/2406.14540v2/x1.png)

---

## 1. Problem & Motivation

**Trajectory-to-video task** 的形式化：

$$
\mathbf{I}^{t+1:t+n+1} = f(\mathbf{I}^{t-h:t}, \mathbf{a}^{t:t+n})
$$

- $h$：历史帧数；$n$：trajectory 长度；$\mathbf{a}^i \in \mathbb{R}^d$，机械臂典型 $d=7$（3 平移 + 3 旋转 + 1 gripper）。
- 与 text-to-video 的关键区别：**text 是 video-level 的粗描述，trajectory 是 frame-level 的精确指令**——每个 action 严格对应一帧画面里的机器人位姿。沿用 text-to-video 的 single-embedding conditioning 会丢掉这种 per-frame correspondence。

> 作者的 framing 抓得很准：trajectory-to-video 不是 text-to-video 的子集，而是一类 conditioning 粒度根本不同的新问题。如果你把 trajectory pool 成一个 embedding 喂给 DiT，模型只能"知道这段视频大致要往哪走"，做不到"第 7 帧 gripper 应该闭合到 0.3"。

---

## 2. Method

**Figure 2. IRASim 网络架构。(a) 总体 DiT 结构；(b) Video-Ada（trajectory pool 成单 embedding）；(c) Frame-Ada（每个 action 独立 embedding，frame-wise modulate spatial block）。**

![](https://arxiv.org/html/2406.14540v2/x2.png)

### 2.1 Backbone：latent-space DiT + spatial-temporal attention

- **VAE**：直接复用 SDXL 预训练的 VAE encoder/decoder，全程冻结。每帧 $\mathbf{I}^t$ 被压到 latent $\mathbf{z}^t$，diffusion 在 latent 空间进行——降低长视频高分辨率的算力开销。
- **Backbone**：DiT blocks，但用 spatial-temporal factorized attention（spatial block 内做 frame-internal attention，temporal block 内做 cross-frame attention），把 token 数从 $O((NP)^2)$ 降到 $O(N \cdot P^2 + P \cdot N^2)$。
- **Historical frame conditioning**：把 $\mathbf{z}^{t-h:t}$ 作为 ground-truth 段拼到输入序列前，训练时只对 $\mathbf{z}^{t+1:t+n+1}$ 加噪、只在预测帧上算 diffusion loss——通过 attention 自动让预测帧"看到"历史帧，保证视觉一致性。

### 2.2 Trajectory conditioning：Video-Ada vs Frame-Ada

两种把 action trajectory 注入 DiT 的方式（论文核心 ablation）：

**Video-Ada（baseline 思路）**：trajectory 经一个 linear 编成单一 embedding，加到 diffusion timestep embedding 上得到 $\mathbf{c}_{ST}$，再回归 spatial / temporal block 共用的 $(\gamma, \alpha, \beta)$（AdaLN 参数）。

$$
\mathbf{x} = \mathbf{x} + (1+\alpha_1) \times \text{MHA}(\gamma_1 \times \text{LayerNorm}(\mathbf{x}) + \beta_1)
$$

**Frame-Ada（本文核心）**：每个 action $\mathbf{a}^i$ 单独经 linear 编 embedding，加 timestep embedding 得到 frame-specific $\mathbf{c}_S^i$。spatial block 对第 $i$ 帧用专属的 $(\gamma_1^i, \alpha_1^i, \beta_1^i)$；temporal block 仍用 video-level 共享 embedding $\mathbf{c}_T$。

$$
\mathbf{x}^i = \mathbf{x}^i + (1+\alpha_1^i) \times \text{MHA}(\gamma_1^i \times \text{LayerNorm}(\mathbf{x}^i + \beta_1^i))
$$

> 这个 design 很直观也很优雅：spatial block 处理"这一帧机器人长什么样"——必须 frame-specific；temporal block 处理"帧间如何过渡"——共享一个全局 trajectory embedding 足矣。等于把 conditioning 的粒度和 attention 的粒度对齐。

### 2.3 长视频：autoregressive rollout

短轨迹（≤ $n$ 步）一次 forward 出完；长轨迹则把上一段最后一帧作为下一段的 historical condition，autoregressive 生成。RT-1 平均 42.5 帧、Bridge 33.4、Language-Table 23.7。

---

## 3. Experiments

### 3.1 Trajectory-conditioned video prediction

**数据集**：RT-1（7-DoF, 256×320）、Bridge（7-DoF, 256×320）、Language-Table（2-DoF, 288×512）、RoboNet（5-DoF unified, 256×256）。短轨迹 1 帧历史 + 15 actions → 预测 15 帧。

**Baselines**：VDM（pixel-space U-Net diffusion）、LVDM（latent-space U-Net diffusion）、iVideoGPT（autoregressive token transformer）、MaskViT（iterative token refinement）。

**Table 1. 短轨迹 quantitative results（节选）。Latent L2 与 PSNR 是论文主指标——trajectory-to-video 是 reconstruction 任务，分布距离类指标 (FID/FVD) 反而和 human preference 不一致。**

| Dataset | Method | PSNR ↑ | SSIM ↑ | Latent L2 ↓ |
| --- | --- | --- | --- | --- |
| RT-1 | VDM | 13.762 | 0.554 | 0.4983 |
| RT-1 | LVDM | 25.041 | 0.815 | 0.2244 |
| RT-1 | Video-Ada | 25.446 | 0.823 | 0.2191 |
| RT-1 | **Frame-Ada (Ours)** | **26.048** | **0.833** | **0.2099** |
| Bridge | LVDM | 23.546 | 0.810 | 0.2155 |
| Bridge | **Frame-Ada (Ours)** | **25.275** | **0.833** | **0.1947** |
| Language Table | LVDM | 28.254 | 0.889 | 0.1704 |
| Language Table | **Frame-Ada (Ours)** | **28.818** | **0.888** | **0.1660** |

**Table 2. RoboNet 上对 token-based baselines 的对比。IRASim 用 SDXL VAE 直接训，不需要在 RoboNet 上 finetune VQGAN（iVideoGPT 与 MaskViT 都需要）。**

| Method | PSNR ↑ | SSIM ↑ |
| --- | --- | --- |
| MaskViT | 20.4 | 67.1 |
| iVideoGPT | 23.8 | 80.8 |
| **IRASim (Ours)** | **24.6** | **81.1** |

**Figure 3. 短/长轨迹定性结果**：predictions（橙框）与 ground truth（蓝框）在三个数据集上对比。

![](https://arxiv.org/html/2406.14540v2/x3.png)

**Scaling**：33M → 679M，三个 dataset 上 latent L2 与 PSNR 都随 model size、训练步数单调改善。

![](https://arxiv.org/html/2406.14540v2/x5.png)

> ❓ Frame-Ada 相对 Video-Ada 的 gap 在 PSNR / Latent L2 上其实不大（RT-1 PSNR 26.048 vs 25.446），主要差异在 long trajectory 与 human preference。是不是说 frame-level alignment 的好处主要体现在 long-horizon 一致性，而非 single-clip 重建？长轨迹时 LVDM 在 Language-Table PSNR 26.215，Frame-Ada 26.773 也并非碾压，这部分差距能否归因于 frame-level conditioning 而非其他设计差异，论文 ablation 不够干净。

**Video clips（来自 project page）**：

**Video 1. RT-1 短轨迹预测**

<video src="https://gen-irasim.github.io/assets/videos/short/rt1.mp4" controls muted playsinline width="720"></video>

**Video 2. Bridge 长轨迹预测**

<video src="https://gen-irasim.github.io/assets/videos/long/bridge.mp4" controls muted playsinline width="720"></video>

### 3.2 Policy evaluation：与 ground-truth simulator 对齐

LIBERO benchmark 上训 4 个不同 step 的 diffusion policy，分别在 Mujoco（GT）和 IRASim 中 rollout 50 次评估。IRASim 用 expert demos + post-trained rollouts（含成功 / 失败）训练，初始化自 OpenSora。

**Table 4. 四个 policy model 的 success rate。Pearson 相关系数 0.99——IRASim 作为 evaluator 与 GT simulator 高度一致。**

| Evaluator | Model 1 | Model 2 | Model 3 | Model 4 |
| --- | --- | --- | --- | --- |
| Ground-Truth Simulator | 0.18 | 0.50 | 0.80 | 1.00 |
| IRASim (Ours) | 0.28 | 0.48 | 0.74 | 0.96 |

**Figure 6. IRASim 能模拟成功与失败 rollout，包括 bowl 从 gripper 滑出这种 fine-grained failure mode。**

![](https://arxiv.org/html/2406.14540v2/x6.png)

> Pearson 0.99 看起来很漂亮，但只 4 个数据点的 correlation 信息量很有限——本质是 ranking 一致 + 大致单调。更有说服力的是论文承认必须用 post-trained rollouts（含失败案例）训练，否则 world model 学不会模拟失败，policy evaluation 会一边倒。这是一个很 honest 的细节。

### 3.3 Model-based planning：Push-T

Ranking-based planner：从 policy sample $K$ 条 trajectory → IRASim 各 rollout 一次 → ResNet50 预测每条 final-frame 的 IoU → 选最高的执行。Push-T 上 vanilla diffusion policy 的 IoU = 0.637。

**Table 5. Push-T benchmark, IRASim 作为 world model 配合 ranking planner。$K$ 是采样轨迹数，$P$ 是训练 IRASim 用的 post-trained rollouts 数。**

| Method | $P$ | $K=1$ | $K=5$ | $K=10$ | $K=50$ |
| --- | --- | --- | --- | --- | --- |
| GPC-RANK | N/A | 0.642 | - | - | 0.698 |
| GPC-RANK+OPT | N/A | 0.642 | 0.824 | 0.882 | - |
| IRASim | 0 | 0.637 | 0.679 | 0.572 | 0.418 |
| IRASim | 100 | 0.637 | 0.847 | 0.878 | 0.888 |
| IRASim | 200 | 0.637 | 0.866 | 0.916 | 0.912 |
| IRASim | 500 | 0.637 | 0.907 | 0.906 | 0.938 |
| IRASim | **1000** | 0.637 | 0.886 | 0.945 | **0.961** |

两个关键观察：
1. **$P=0$ 时 $K↑$ 反而退化**——只用 expert demos 训的 world model 不会模拟失败 trajectory，越采越多 ranking 越没意义。
2. **数据规模与 test-time compute 必须同时 scale**：$P=100$ 时 $K=50$ 收益已饱和，需要 $P=1000$ 才能让 $K=50$ 持续受益。

**Figure 7. Model-based planning 流程：sample → simulate in IRASim → reward predict → select best for execution。**

![](https://arxiv.org/html/2406.14540v2/x7.png)

### 3.4 Real-robot model-based planning

3 个真机任务（关抽屉、放橘子到绿盘、放橘子到红盘），$K=50$ 采样自一个简单球面采样 policy，goal-image similarity 作为 value function。

**Table 6. Real-robot 结果。MSE similarity 比 ResNet50 cosine similarity 显著更好，两者都远超 random baseline。**

| Method | Close Drawer | Place Mandarin on Green Plate | Place Mandarin on Red Plate |
| --- | --- | --- | --- |
| Random | 0.20 | 0.07 | 0.13 |
| IRASim (ResNet50) | 0.60 | 0.73 | 0.60 |
| IRASim (MSE) | 0.87 | 0.80 | 0.87 |

> ❓ MSE > ResNet50 这个反直觉结果几乎没解释。可能是 ResNet50 feature 没在该 domain 上 finetune；也可能是 goal-image 像素相似度恰好是这些短任务的好 proxy。但这意味着 value function 的选择对 model-based planning 的影响可能比 world model 本身的质量还大——这是个被论文淡化的 caveat。

### 3.5 Flexible action controllability

用键盘（Language-Table 2D）、VR controller（RT-1 3D）输入超出训练分布的 trajectory，IRASim 仍能合理 simulate 机器人 - 物体交互，包括对 physically implausible trajectory（指令机器人穿过桌面）的"物理 robust"——机械臂被 visually 卡在桌面上而非穿透。

> ❓ Physical-implausibility robustness 听起来像 emergent physics understanding，但更可能是数据分布里没有"桌面被穿透"的样本，模型 default 到"trajectory 卡住"——这是 OOD failure mode 而非 physical reasoning。这块需要更严谨的对照。

---

## 关联工作

### 基于
- **DiT (Peebles & Xie, 2023)**：backbone 架构与 AdaLN conditioning paradigm 直接来自 DiT
- **OpenSora**：policy evaluation 与 real-robot planning 实验中用作初始化权重
- **SDXL VAE**：latent 空间编解码，全程冻结
- **DDPM / PNDM**：diffusion 训练目标与 50-step sampling

### 对比 / Baselines
- **VDM (Video Diffusion Models)**：pixel-space U-Net diffusion；IRASim 通过 latent + DiT 全面超越
- **LVDM (Latent Video Diffusion Models)**：latent-space U-Net diffusion；最强 baseline，但缺乏 frame-level conditioning
- [[2402-Genie|Genie]]：generative interactive environments，token-based、focus on game / 2D；IRASim 是 robot-specific、diffusion-based、连续 action
- [[2408-GameNGen|GameNGen]]：diffusion-based real-time game engine（DOOM）；IRASim 同样是 action-conditioned diffusion world model 但 target real-robot 而非 game
- **iVideoGPT / MaskViT**：token-based autoregressive / iterative refinement 视频预测；在 RoboNet 上对比
- **GPC (Generative Predictive Control)**：autoregressive next-frame diffusion + planning；Push-T 上的最强对手，IRASim 用 trajectory-level joint generation 代替 next-frame autoregression

### 方法相关
- **Sora**：text-to-video DiT，作者的 conditioning paradigm 灵感来源；IRASim 把 video-level text condition 改造为 frame-level action condition
- **DreamerV3 / DayDreamer**：用 RSSM 学 latent world model 做 RL；IRASim 走的是 high-fidelity video generation 路线，trade off 是 inference 慢但 visual fidelity 高
- **UniSim**：language + action prompt text-to-video；与 IRASim 同样是 action-conditioned video world model，但 conditioning 粒度仍是 video-level
- [[2410-Pi0|π0]] / Diffusion Policy：action chunking 的代表方法，IRASim 与之天然兼容（一次预测整个 chunk 对应的视频）
- [[2411-WorldModelSurvey|World Model Survey]]：把 IRASim 归为 robot-manipulation 子方向的 fine-grained action-conditioned video model
- [[2501-RoboticWorldModel|Robotic World Model Survey]]：相关 survey

---

## 论文点评

### Strengths

1. **Frame-level conditioning 的设计非常精准**——把 trajectory-to-video 与 text-to-video 的 conditioning 粒度差异讲清楚，并用最小的架构改动（per-frame AdaLN scale/shift）解决，没有引入额外 module。这是好的 first-principles design：identify gap → minimal fix。
2. **三个下游任务一起验证 world model 的实用性**——video quality / policy evaluation / model-based planning，避免 "video looks good but not useful" 的 critique，证明 fine-grained alignment 不只是 visual fidelity 也对 downstream control 有意义。
3. **Test-time scaling for manipulation 的早期信号**——把 $K \times P$ scaling chart 摆出来，展示了 manipulation 中 inference-time compute 投入的回报，而当时 LLM 圈才刚开始讨论 test-time scaling。
4. **Honesty 加分项**：明确指出 policy evaluation 必须包含 post-trained rollouts（含失败）才能 work；明确指出 latent L2 / PSNR 比 FID / FVD 更适合 reconstruction-style task，并用 user study 论证；这些 negative observation 比 main result 信息量更大。

### Weaknesses

1. **Inference 速度问题被轻描淡写**：A100 上生成 16 帧需要 30 秒。作为 simulator 用于 RL 训练或 large-scale policy evaluation 时，这个速度根本不够——真正取代 Mujoco 还差 2-3 个数量级。论文承诺 future work distillation 但没给数据。
2. **Policy evaluation correlation 的 sample size 太小**：Pearson 0.99 来自 4 个 model 数据点，每个 model 50 次 rollout。换不同 task / 难度差异更大的 policy 是否仍 0.99 没验证。
3. **Real-robot planning 的 value function 设计可能 dominate 结果**：MSE > ResNet50 这个反直觉差异没有 ablation 解释，意味着 reported gain 有一部分可能来自 task-specific value function 而非 world model 自身。
4. **没有与 RL-based planner（CEM/MPPI）对比**——Ranking planner 是非常 simple 的算法，是否 fancy planner 还能进一步提升 / 是否 simple planner 已饱和，没有讨论。
5. **数据效率不明**：Push-T 上 $P=1000$ post-trained rollouts 是怎么 scale 到更复杂的任务？real-robot 实验用了多少数据训 IRASim 没在主文交代清楚（"similar to RT-1 size"）。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training，开源在 https://github.com/bytedance/IRASim
- **模型权重**: 已发布 RT-1 / Bridge / Language-Table 三个 dataset 的 checkpoint（每个 ~30G）
- **训练细节**: 较完整——hyperparameter table（layers=28, hidden=1152, AdamW 1e-4, batch 64, EMA 0.9999, 300k steps, gradient clip 0.1）；不同 model size 的配置（33M → 679M）；scaling 实验细节；compute resources 表
- **数据集**: 全开源——RT-1、Bridge、Language-Table、RoboNet 都是公开 dataset；作者额外打包了 IRASim Benchmark 数据放 ByteDance 公共 OSS 与 HuggingFace（fangqi/IRASim）

#### Claim 可验证性
- ✅ **Video quality 在 4 个 dataset 上超 baselines**：Latent L2 / PSNR / SSIM 全公开，code+ckpt+data 都开源，可独立复现
- ✅ **Frame-Ada > Video-Ada**：直接 architectural ablation，差距虽不大但 consistent
- ✅ **IRASim 与 LIBERO Mujoco evaluation 强相关**：4 个 policy model 的数据点都给了，方法可复现，但相关性的统计意义弱
- ✅ **Model-based planning 把 Push-T IoU 从 0.637 拉到 0.961**：完整 $K \times P$ ablation table；GPC 对比公平
- ⚠️ **"Test-time scaling for robot manipulation"**：只在 Push-T（2D 简化任务）+ 真机 3 个短任务上验证；推广到 long-horizon / multi-task 的能力是 extrapolation
- ⚠️ **"Robust to physically implausible trajectories"**：定性 demo 一例（机器人不穿透桌面），样本量过小且没有控制实验区分 "学到物理" vs "训练分布外 default 行为"
- ⚠️ **"Pearson correlation 0.99 with GT simulator"**：4 个数据点；correlation 的统计显著性与 generalization 都需要更多验证

### Notes

- **Idea seed**：frame-level conditioning 的 design pattern 不只适用于 trajectory-to-video——任何"per-step 信号要驱动 per-frame 输出"的任务（speech-driven face video、music-driven dance video）都可以借鉴这种 AdaLN frame-wise modulation。
- **For my agenda**：作为 VLA × world model 交叉的 reference 之一，IRASim 给出了一个清晰的 evidence——high-fidelity video world model 在 manipulation 上确实能提供 actionable signal（policy evaluation + planning），不只是 visual eye candy。但 inference 慢仍是 bottleneck，distillation 路线值得跟踪。
- **可能的延伸**：(a) 如果把 frame-level conditioning 换成 cross-attention 而非 AdaLN，能否处理 variable-length trajectory？(b) 把 reward / value model 也做成 trajectory-level（而非 final-frame ResNet50），能否进一步提升 model-based planning？(c) 在长 horizon autoregressive rollout 时，error accumulation 是否可以用 teacher forcing schedule 缓解？
- **Tracking**：作者的 follow-up（IRASim → ICCV 2025 版改名为 "Fine-Grained World Model for Robot Manipulation"）值得对比看变更。

### Rating

**分数**：2 - Frontier
**理由**：Frame-level AdaLN conditioning 是 trajectory-to-video world model 方向的代表性设计（Strengths #1），三个下游任务 joint-validation 足够扎实（Strengths #2–3），ICCV 2025 被接收且由 ByteDance 完整开源 code + weights + benchmark，已被 WMPO 等后续 world-model-based policy optimization 工作作为 reference/baseline。不到 3 - Foundation 因为：(a) inference 30s/16 帧的速度瓶颈决定它短期内无法像 Mujoco/DROID 那样成为 de facto simulator（Weaknesses #1），(b) Pearson 0.99 只有 4 个数据点、real-robot 任务 value-function 可能 dominate 结果（Weaknesses #2–3），方向地位尚未定型；高于 1 - Archived 因为它仍是当前 fine-grained action-conditioned video world model 的必引参考，方法未被取代。
