---
title: "Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution"
authors: [Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, Feng Qiu, Heng Qu, Yifei Su, Qiao Sun, Dong Wang, Donghao Wang, Yunhong Wang, Rujie Wu, Diyun Xiang, Yu Yang, Hangjun Ye, Yuan Zhang, Quanyun Zhou]
institutes: [Xiaomi]
date_publish: 2026-02
venue: arXiv preprint
tags: [VLA, manipulation, flow-matching]
paper: https://arxiv.org/abs/2602.12684
website: https://xiaomi-robotics-0.github.io
github: https://github.com/XiaomiRobotics/Xiaomi-Robotics-0
rating: 2
date_added: 2026-04-19
---
## Summary

> [!summary] Xiaomi-Robotics-0
> - **核心**: 4.7B MoT（Qwen3-VL-4B + 16-layer DiT）VLA，用 Λ-shape attention mask 阻断 action-prefix shortcut，在 RTX 4090 上 80 ms 延迟实现平滑异步实时执行
> - **方法**: pre-train 两步（Choice Policies + VL co-train 1:6 防遗忘 → 冻结 VLM 训 DiT 用 flow matching）；post-train RoPE offset + Λ-mask + L1-error 动态 loss reweight 解决 prefix shortcut
> - **结果**: LIBERO 98.7%，CALVIN ABCD→D 4.80 / ABC→D 4.75，SimplerEnv VM 85.5% / VA 74.7% / WidowX 79.2%；real-robot Towel Folding 1.2 vs [[2504-Pi05|π0.5]] 1.0 pcs/min；ERQA 40.8 略超基座 Qwen3-VL-4B 的 40.0
> - **Sources**: [paper](https://arxiv.org/abs/2602.12684) | [website](https://xiaomi-robotics-0.github.io) | [github](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)
> - **Rating**: 2 - Frontier（Λ-mask 针对 training RTC 的 prefix shortcut 给出 first-principles fix，三个 sim benchmark SOTA，但方法承袭 π0/π0.5 范式且只开源 inference，关键超参未放出，复现门槛高，属 Frontier baseline）

**Key Takeaways:**
1. **Λ-shape attention mask 治 action-prefix shortcut**：training-time RTC 把 committed prefix 拼到 noisy action 前，但 later-timestep 的 noisy action 会走捷径直接 copy prefix 而忽略视觉/语言条件，导致 reactive 能力下降。Λ-mask 让后段 token 无法 attend prefix，强制它们 attend VLM KV cache。
2. **VLM + DiT 的 MoT 架构 + KV cache 桥接**：Qwen3-VL-4B-Instruct 冻结后作 multimodal conditioner，16-layer DiT 从 scratch 训，只 condition 于 VLM 最后 16 层的 KV cache 来压缩推理延迟到 80 ms @ RTX 4090。
3. **VL co-train 是必需品而非 nice-to-have**：w/o VL data 的 ablation 在所有 10 个 VL benchmark 全部归零，VL co-train 后 ERQA 甚至略超基座（40.8 vs 40.0）——robot-trajectory-derived 的 VL 数据反而强化了 robot-centric 的 embodied perception。
4. **三个 sim benchmark SOTA + real-robot 最高吞吐**：LIBERO 98.7%、CALVIN 4.75/4.80、SimplerEnv 三个设置全胜；Towel Folding 1.2 pcs/min 比 [[2504-Pi05|π0.5]] 的 1.0 高 20%，Training RTC 变体会陷入 repetitive fling loop 失败——直接印证 prefix shortcut 假设。
5. **开源但 inference-only**：base + 5 个 fine-tune checkpoint 发布到 HuggingFace，推理代码走 transformers 生态，**未放出训练代码、VL 数据 curation pipeline 与 Λ-mask 窗口/Beta 分布/reweight 公式等关键超参**。

**Teaser.** 项目主页 hero 视频，展示 Lego Disassembly 与 Towel Folding 两个 in-house 任务的真实机器人执行效果。

<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0.mp4" controls muted playsinline width="720"></video>

---
## Introduction

VLA 建立在预训练 VLM 之上，把 observation + language instruction 映射到 action。主要痛点是：参数量大 → 推理延迟高 → consecutive chunks 之间不平滑会引入 OOD jerky 动作。本文的切入点是 training recipe + deployment strategy 两手兼修：

- **Pre-training**：cross-embodiment robot trajectory + vision-language (VL) data 联合训练，让 VLM 既获得 action generation 能力、又不忘掉原始的 vision-semantic 知识（catastrophic forgetting 是本文一贯强调的对手）。
- **Post-training (async)**：沿用 training RTC 的 prefix-conditioning 范式，但用 Λ-shape attention mask 阻止后段 noisy action 直接 copy prefix——强迫它们去 attend 视觉/语言，从而保留 reactive 能力。
- **Deployment**：严格对齐 consecutive chunks 的 timestep，保证接缝平滑。

**Figure 1. Overview.** Xiaomi-Robotics-0 在三个 sim benchmark 上 SOTA，在两个 bimanual real-robot 任务上达到最高吞吐，并在多个 VLM benchmark 上匹配基座 VLM。

![](https://arxiv.org/html/2602.12684v2/x1.png)

Headline 数字（正文声明）：LIBERO 98.7%，CALVIN ABCD→D 从 prior SOTA 4.67 → 4.80、ABC→D 从 4.54 → 4.75，SimplerEnv VM 85.5% / VA 74.7% / WidowX 79.2%；real-robot Lego Disassembly 和 Towel Folding 上 success rate 和 throughput 均高于 SOTA。发布 pre-trained + post-trained checkpoint 和 inference code。

---
## Xiaomi-Robotics-0

End-to-end VLA：输入观测图像 $\mathbf{o}_t$ + 语言指令 $l$ + proprioceptive state $\mathbf{s}_t$，输出 $T$-step action chunk $\mathbf{a}_{t:t+T}$ 控制 bimanual 机器人。

### Data

**Figure 2. Data.** Robot trajectory 和 vision-language 数据在 pre-training 阶段联合使用，覆盖 grounding / VQA / captioning / embodied reasoning & planning 四类 VL 任务。

![](https://arxiv.org/html/2602.12684v2/x2.png)

- **Robot trajectory** (~200M timesteps)：开源数据集（DROID、MolmoAct 等）+ 自采。自采集中于两个长任务——Lego Disassembly 338 小时 + Towel Folding 400 小时。
- **Vision-language 数据** (~80M samples)：两条主干——(1) 通用 VL 数据集（Conceptual 12M、Conceptual Captions、Cambrian-1、FineVision 等）；(2) 从 robot trajectory 衍生的 VL 数据，强化 egocentric / wrist camera 视角的 robot-centric 感知。
- **标注管道**：
  - Grounding：Grounded SAM + Grounding DINO 1.5 + LLMDet 的 cross-validated consensus 保证 pixel-level 精度；
  - VQA/captioning：用 SOTA VLM (Qwen3-VL) 重新 re-label；
  - EQA / high-level planning / point trajectory：用预训练 VLM 直接从 trajectory 生成。

### Model & Training

**Figure 3. Model & Training.** (a) Pre-training step 1：VLM 在 VL 数据（next-token-prediction）和 robot trajectory（Choice Policies 多 candidate + winner-takes-all）上联合训练。(b) Pre-training step 2：冻结 VLM，从 scratch 训 DiT 用 flow-matching 生成 action。(c) Post-training for async：clean action prefix 拼在 noisy action tokens 前。

![](https://arxiv.org/html/2602.12684v2/x3.png)

架构：**Mixture-of-Transformers (MoT)**——Qwen3-VL-4B-Instruct + 16-layer DiT，共 4.7B 参数。VLM 处理 $\mathbf{o}_t$ 和 $l$ 产生 KV cache；DiT 通过 flow-matching 生成 action chunk，conditioned on VLM KV cache 和 proprioceptive state。

#### Pre-training

**Step 1**：让 VLM 本身学会动作预测。采用 **Choice Policies** 处理 trajectory 的多模态——同时预测 $N$ 个 action chunk 候选 + 每个候选的 score：
- Action prediction 用 **winner-takes-all**：只对与 ground truth 的 $L_1$ 距离最小的候选做反向传播；
- Score prediction 的 target 是每个候选到 ground truth 的 $L_1$ 距离。

Token 序列为 $\mathbf{o}_t, l, \mathbf{s}_t, [A_1], \ldots, [A_T], [S]$。Proprioceptive state 用 MLP 编码；每个 $[A_i]$ 输出 $N$ 个 $i$-th timestep 预测；$[S]$ 输出 $N$ 个 score。

**防遗忘的关键**：VL 数据与 robot 数据按 **1:6** 比例 co-train（VL 走 next-token-prediction 目标）。Table 3 的 w/o VL data ablation 全零显示这一步是必需品。

**Step 2**：冻结 VLM，从 scratch 训 DiT 用 flow-matching：

$$
L(\theta) = \Big\|\mathbf{v}_{\theta}(\mathbf{o}_{t}, l, \mathbf{s}_{t}, \tilde{\mathbf{a}}_{t:t+T}^{\tau}, \tau) - \mathbf{u}(\tilde{\mathbf{a}}_{t:t+T}^{\tau}, \mathbf{a}_{t:t+T}, \tau)\Big\|^{2}_{2}
$$

**符号说明**：
- $\tau \in [0, 0.999]$ 是 flow-matching timestep，从 **Beta 分布**采样（偏 noisier timesteps，具体参数未披露）；
- $\tilde{\mathbf{a}}_{t:t+T}^{\tau} = \tau \mathbf{a}_{t:t+T} + (1-\tau)\boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是 noisy action；
- $\mathbf{v}_\theta$ 是 DiT 预测速度场，$\mathbf{u}$ 是 ground-truth 条件 flow field。

**架构细节**：adaLN 注入 timestep 条件；proprioceptive state 和 noisy action 走 MLP 编码；最前面插一个 learnable **attention sink token** 稳定 attention；DiT 内部 causal attention；DiT 只 condition 于 VLM 最后 16 层的 KV cache 压低延迟；Step 2 里 VLM 的输入不含 Step 1 引入的 `[A_i]`/`[S]` token。

> ❓ Beta 分布的具体 shape 参数（$\alpha, \beta$）论文没给，只说 "placing more weight on noisier timesteps"。

#### Post-training

针对特定机器人，只用该机器人的 trajectory 数据继续训练。分 synchronous 和 asynchronous 两种模式：

- **Sync**：简单 unfreeze 整个模型（VLM + DiT），继续 flow-matching 训练。
- **Async**：follow training RTC——把 $\Delta t_c$ 个 previously committed actions 作为 clean prefix 拼到 noisy action tokens 前。DiT 输入序列变为：

$$
[\text{SINK}], \mathbf{s}_t, \mathbf{a}_t, \ldots, \mathbf{a}_{t+\Delta t_c - 1}, \tilde{\mathbf{a}}^{\tau}_{t+\Delta t_c}, \ldots, \tilde{\mathbf{a}}^{\tau}_{t+T-1}
$$

**Problem**：这种 prefix conditioning 让 later-timestep 的 noisy action 直接 copy prefix（shortcut）——模型不再 attend 视觉/语言，reactive 能力下降。

**Figure 4. Λ-Shape Attention Mask for Post-Training.** Noisy action token 只能 attend VLM KV cache、sink、state、以及前 $w$ 个 timestep 的 action token。每个 token 标注的数字是 RoPE positional index——noisy action 的 RoPE 加了 10 的 offset，以便模型区分 noisy 与 clean prefix。

![](https://arxiv.org/html/2602.12684v2/x4.png)

**Fix**：两个小 trick：

1. **RoPE offset**：给 noisy action token 的 positional index 加 10，区分 noisy vs clean prefix。
2. **Λ-shape attention mask**：noisy action token 只能 attend 到紧邻 prefix 结尾的窗口 + VLM KV cache + sink + state；更后面的 noisy action 看不到 prefix，被迫 attend 视觉/语言。

**训练采样**：$\Delta t_c$ 从 $\{0, 1, \ldots, 6\}$ 均匀采样。当 $\Delta t_c > 0$ 时，根据 **online-predicted actions** 相对 ground truth 的 $L_1$ error 动态 reweight flow-matching loss——优先学偏差大的样本。

> ❓ Λ-mask 的窗口大小 $w$ 具体值、以及 reweight 公式都没在正文给出，需要看代码或 appendix。

### Deployment

**Figure 5. Asynchronous Execution.** 两个 consecutive chunk 如何在 robot rollout 中被拼接。

![](https://arxiv.org/html/2602.12684v2/x5.png)

- **Synchronous**：执行当前 chunk 前 $T_e$ 步 → 触发新推理 → **机器人 idle 等待**推理完成。
- **Asynchronous**：
  - 执行 $T_e$ 步后立即触发新推理；
  - 推理期间机器人继续执行当前 chunk 剩余动作；
  - 用第 $T_e$ 到 $T_e + \Delta t_c - 1$ 步的动作作为 prefix condition 新 chunk；
  - 新 chunk 从第 $\Delta t_{\mathrm{inf}}$ 步开始执行（$\Delta t_{\mathrm{inf}}$ 为推理延迟）；
  - 约束 $\Delta t_c \geq \Delta t_{\mathrm{inf}}$ 保证 prefix 覆盖整个推理窗口，衔接无缝。

**推理细节**：action chunk 从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样初始化，执行 5 步 flow-matching 积分（$\tau: 0 \to 1$）。**NVIDIA RTX 4090** 上 $t_{\mathrm{inf}} = 80$ ms。所有模态 resample 到 30Hz 统一 timeline，每个 tick 取时间最近的测量聚合成 model input。

---
## Experiments

### Simulation Benchmarks

三个 sim benchmark：**LIBERO**（4 个 suite，$T=10$）、**CALVIN**（ABCD→D in-distribution + ABC→D OOD，$T=10$）、**SimplerEnv**（Google Robot VM/VA + WidowX，$T=4$）。

**Table 1. Results on the LIBERO benchmark.**

| Method | Libero-Spatial | Libero-Object | Libero-Goal | Libero-Long | Average |
|---|---|---|---|---|---|
| [[2406-OpenVLA\|OpenVLA]] | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| [[2502-OpenVLA-OFT\|OpenVLA-OFT]] | 97.6% | 98.4% | 97.9% | 94.5% | 97.1% |
| [[2410-Pi0\|π0]] | 96.8% | 98.8% | 95.8% | 85.2% | 94.2% |
| π0-FAST | 96.4% | 96.8% | 88.6% | 60.2% | 85.5% |
| [[2504-Pi05\|π0.5]] | 98.8% | 98.2% | 98.0% | 92.4% | 96.9% |
| [[2503-GR00TN1\|GR00T-N1]] | 94.4% | 97.6% | 93.0% | 90.6% | 93.9% |
| UniVLA | 95.4% | 98.8% | 93.6% | 94.0% | 95.5% |
| Discrete Diffusion VLA | 97.2% | 98.6% | 97.4% | 92.0% | 96.3% |
| MemoryVLA | 98.4% | 98.4% | 96.4% | 93.4% | 96.7% |
| FLOWER | 97.5% | 99.1% | 96.1% | 94.9% | 96.9% |
| EO-1 | 99.7% | 99.8% | 99.2% | 94.8% | 98.2% |
| **Xiaomi-Robotics-0 (Ours)** | **98.8%** | **100.0%** | **98.8%** | **97.2%** | **98.7%** |

**Insights**：Libero-Long 上把次优 FLOWER 的 94.9% 拉到 97.2%（+2.3）——long-horizon 的 gap 明显大于其他 suite，是 ablation 中最 informative 的一列。LIBERO 本身已近饱和，98.7% vs EO-1 98.2% 的 0.5% 差距在噪声量级。

**Table 2. Results on the CALVIN benchmark.**

| Method | Setting | 1 | 2 | 3 | 4 | 5 | Avg. Len. ↑ |
|---|---|---|---|---|---|---|---|
| RoboFlamingo | ABCD→D | 96.4% | 89.6% | 82.4% | 74.0% | 66.0% | 4.09 |
| GR-1 | ABCD→D | 94.9% | 89.6% | 84.4% | 78.9% | 73.1% | 4.21 |
| MoDE | ABCD→D | 97.1% | 92.5% | 87.9% | 83.5% | 77.9% | 4.39 |
| [[2412-RoboVLMs\|RoboVLMs]] | ABCD→D | 96.7% | 93.0% | 89.9% | 86.5% | 82.6% | 4.49 |
| MDT | ABCD→D | 98.6% | 95.8% | 91.6% | 86.2% | 80.1% | 4.52 |
| UniVLA | ABCD→D | 98.5% | 96.1% | 93.1% | 89.9% | 85.1% | 4.63 |
| FLOWER | ABCD→D | 99.2% | 96.9% | 96.9% | 92.3% | 88.3% | 4.67 |
| **Xiaomi-Robotics-0 (Ours)** | ABCD→D | **99.7%** | **98.0%** | **96.7%** | **94.2%** | **91.8%** | **4.80** |
| RoboFlamingo | ABC→D | 82.4% | 61.9% | 46.6% | 33.1% | 23.5% | 2.48 |
| SuSIE | ABC→D | 87.0% | 69.0% | 49.0% | 38.0% | 26.0% | 2.69 |
| GR-1 | ABC→D | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06 |
| 3DDA | ABC→D | 93.8% | 80.3% | 66.2% | 53.3% | 41.2% | 3.35 |
| MoDE | ABC→D | 96.2% | 88.9% | 81.1% | 71.8% | 63.5% | 4.01 |
| GR-MG | ABC→D | 96.8% | 89.3% | 81.5% | 72.7% | 64.4% | 4.04 |
| [[2412-RoboVLMs\|RoboVLMs]] | ABC→D | 98.0% | 93.6% | 85.4% | 77.8% | 70.4% | 4.25 |
| Seer-Large | ABC→D | 96.3% | 91.6% | 86.1% | 80.3% | 74.0% | 4.28 |
| VPP | ABC→D | 95.7% | 91.2% | 86.3% | 81.0% | 75.0% | 4.29 |
| UniVLA | ABC→D | 98.9% | 94.8% | 89.0% | 82.8% | 75.1% | 4.41 |
| FLOWER | ABC→D | 99.4% | 95.8% | 90.7% | 84.9% | 77.8% | 4.53 |
| **Xiaomi-Robotics-0 (Ours)** | ABC→D | **100.0%** | **98.3%** | **96.0%** | **92.6%** | **88.1%** | **4.75** |

**Insights**：ABC→D 是 OOD 泛化 split（训练不含 D），Xiaomi-Robotics-0 在 Task-5 从 FLOWER 77.8% 拉到 88.1%——**10 个点的 gap 比 in-distribution ABCD→D split 更大**，说明 VL co-train 带来的表示质量对 OOD 迁移有实质增益。

**Table 3. SimplerEnv WidowX (Bridge-trained).**

| Method | Put Spoon on Towel | Put Carrot on Plate | Stack Blocks | Put Eggplant in Basket | Overall |
|---|---|---|---|---|---|
| [[2405-Octo\|Octo]]-Small | 47.2% | 9.7% | 4.2% | 56.9% | 29.5% |
| [[2406-OpenVLA\|OpenVLA]] | 0% | 0% | 0% | 4.1% | 1.0% |
| [[2410-Pi0\|π0]] | 83.8% | 52.5% | 52.5% | 87.9% | 69.2% |
| SpatialVLA | 16.7% | 25.0% | 29.2% | 100% | 42.7% |
| EO-1 | 63.6% | 54.5% | 81.8% | 90.9% | 72.7% |
| **Xiaomi-Robotics-0 (Ours)** | **95.8%** | **62.5%** | 75.0% | 83.3% | **79.2%** |

**Table 4. SimplerEnv Google Robot (Fractal-trained, Visual Matching).**

| Method | Pick Coke Can | Move Near | Open/Close Drawer | Drawer Apple | Overall |
|---|---|---|---|---|---|
| [[2406-OpenVLA\|OpenVLA]] | 16.3% | 46.2% | 35.6% | 0% | 24.5% |
| RT-1 | 85.7% | 44.2% | 73.0% | 6.5% | 52.4% |
| [[2307-RT2\|RT-2]]-X | 78.7% | 77.9% | 25.0% | 7.4% | 47.3% |
| [[2410-Pi0\|π0]] | 97.9% | 78.7% | 62.3% | 46.6% | 71.4% |
| EO-1 | 98.0% | 83.8% | 71.3% | 52.8% | 76.5% |
| **Xiaomi-Robotics-0 (Ours)** | **98.7%** | **88.8%** | **79.6%** | **75.0%** | **85.5%** |

**Insights**：Drawer Apple 任务（最难，涉及 compositional drawer + 物体操作）从 EO-1 52.8% 拉到 75.0%（+22）——和 CALVIN ABC→D 的 OOD gap 一致，都是在"更难的设置下差距更大"，暗示 VL co-train + 大 trajectory pool 对困难场景增益更明显。VA split 略（overall 74.7%），数字在 paper Appendix B Table 5 完整给出。

### Real-Robot Experiments

**Figure 6. Real-Robot Experiments.** (a) Lego Disassembly 评测设置（LA-5/10/20 + MA 34 bricks）。(b) Towel Folding 评测设置和使用的 6 种毛巾。(c) 两个任务上的定量结果（条形图，无数值表）。

![](https://arxiv.org/html/2602.12684v2/x6.png)

**平台**：bimanual robot，两个 6-DoF arm，三相机（2 wrist-mounted + 1 external global）。

**Tasks**：
- **Lego Disassembly**：拆 Lego → 按颜色分拣到 bin。要求 bimanual 协同 + 精确放置。两个 setting：**LA**（large-assembly，LA-5 / LA-10 / LA-20 三个尺寸 × 3 configs × 3 trials）和 **MA**（multi-assembly，共 34 块含单块和 2-3 块组合，3 trials）。Metric：正确分拣率 + throughput（#bricks/time）。
- **Towel Folding**：从托盘取毛巾 → 展平 → 对折两次 → 放到 staging area。毛巾 deformable，动力学复杂（wrinkles、遮挡）。用 6 条不同毛巾，每方法跑两次 30 分钟；单次折叠 > 2 分钟视为失败。Metric：throughput（#folded/time）。

**Baselines**：
- [[2504-Pi05|π0.5]]：按官方 OpenPi fine-tune 协议从 release base 启动，训练 setting 与本文一致；
- **Xiaomi-Robotics-0**：主方法（async + Λ-mask）；
- **Xiaomi-Robotics-0 (Sync)**：同步变体；
- **Xiaomi-Robotics-0 (Training RTC)**：纯 training RTC 变体（prefix conditioning 但无 Λ-mask），作为消融。

**Training**：pre-train 40k steps × batch 32,768；post-train Lego 40k steps、Towel 80k steps × batch 2,048；AdamW + DeepSpeed ZeRO-2；action chunk $T=30$ 对应 1 秒。

**Results**（定量在 Fig.6c 条形图）：

- **Lego Disassembly**：所有方法 avg success rate 接近。两个 sync 方法（[[2504-Pi05|π0.5]] 和 Xiaomi-Robotics-0 Sync）精度略高——异步方法 reactive 性稍差，会出现 gripper 和 brick 之间张力过大导致 brick 弹飞。**Throughput**：Xiaomi-Robotics-0 (Sync) > [[2504-Pi05|π0.5]]；**Xiaomi-Robotics-0 (async) 最高**，还超过 training RTC 变体。
- **Towel Folding**：[[2504-Pi05|π0.5]]、Xiaomi-Robotics-0 (Sync)、Xiaomi-Robotics-0 (Training RTC) 三者均为 1 pcs/min；**Xiaomi-Robotics-0 达到 1.2 pcs/min**（+20%）。Training RTC 变体有一个典型 failure mode——flinging 时不小心抓到两层毛巾会陷入 repetitive fling loop 无法自救，**直接印证 prefix shortcut 假设**（模型 copy prefix 而不 attend 当前观测）。

**Video 2. Lego Disassembly — complex task manipulation.** 拆解最多 20 块的复杂 Lego 组合，对应 LA-20 设置。
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-Lego-disassembly-complex-task-manipulation.mp4" controls muted playsinline width="720"></video>

**Video 3. Lego Disassembly — flexible motion switch.** 抓取失败后自适应切换抓取动作——展示 async + Λ-mask 保留的 reactive 能力。
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-Lego-disassembly-flexible-%20motion-switch.mp4" controls muted playsinline width="720"></video>

**Video 4. Towel Folding — fling motion.** 当毛巾角被遮挡时，单手甩动暴露隐藏角——这是 Training RTC 变体陷入 repetitive loop 的同一动作，但本方法能正常退出。
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-towel-folding-fling.mp4" controls muted playsinline width="720"></video>

**Video 5. Towel Folding — put back.** 抓出两条毛巾时先放回多余的再开始折叠，需要 attend 当前视觉而非按 prefix 惯性继续。
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-towel-folding-put-back.mp4" controls muted playsinline width="720"></video>

### Preservation of Vision-Language Capabilities

10 个 VL benchmark（含 ERQA embodied reasoning），对比 [[2410-Pi0|π0]]（无 VL co-train）、[[2504-Pi05|π0.5]]（有 VL 训练）、MolmoAct（有 VL 训练）、Xiaomi-Robotics-0 自己的 w/o VL data ablation、以及基座 Qwen3-VL-4B-Instruct。

**Table 5. Quantitative results on general VL + embodied reasoning benchmarks.**

| Model | ERQA | SEED | POPE | AI2D | MMBench | MME | MMMU | TextVQA | SciQA | ChartQA |
|---|---|---|---|---|---|---|---|---|---|---|
| [[2410-Pi0\|π0]] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 1.4 | 0.0 | 0.0 |
| [[2504-Pi05\|π0.5]] | 0.0 | 21.5 | 0.0 | 14.4 | 22.1 | 0.0 | 19.9 | 0.0 | 28.0 | 0.5 |
| MolmoAct | 33.5 | 72.7 | 86.6 | 72.0 | 80.1 | 69.5 | 38.0 | 67.3 | **91.1** | 57.1 |
| Xiaomi-Robotics-0 (w/o VL data) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Xiaomi-Robotics-0 (Ours)** | **40.8** | **78.6** | **88.5** | **78.7** | **84.4** | **81.8** | **46.2** | **72.0** | 79.4 | **59.2** |
| Qwen3-VL-4B-Instruct (base) | 40.0 | 78.8 | 89.7 | 81.6 | 88.7 | 87.1 | 51.7 | 78.0 | 92.7 | 76.8 |

**Insights**：
- **[[2410-Pi0|π0]] 几乎全归零，Xiaomi-Robotics-0 (w/o VL data) 全 0**——双重 ablation 证明：不 co-train VL 数据会**彻底**摧毁基座 VLM 的 VL 能力（catastrophic forgetting）。
- **[[2504-Pi05|π0.5]] 有零星残留（SEED 21.5、MMBench 22.1、MMMU 19.9 等）但远低于正常**——可能是其训练配方里某种程度保留了 VL 数据。
- **Ours 在 9/10 benchmark 上领先所有 VLA 基线，ERQA 40.8 略超基座 40.0**——作者 hypothesis 是 robot-trajectory-derived 的 VL 数据强化了 robot-centric embodied perception。
- **唯一输给 MolmoAct**：SciQA 79.4 vs 91.1。其他 benchmark 略落后基座但幅度都小（AI2D 78.7 vs 81.6；MMBench 84.4 vs 88.7），"preservation" 的 claim 成立。

> ❓ [[2504-Pi05|π0.5]] 的 VL 分数为什么是"非零但离谱低"的状态（如 SEED 21.5、MMMU 19.9）？值得和 [[2504-Pi05|π0.5]] 原论文对比其 VL 训练配比。

---
## 关联工作

### 基于

- [[2410-Pi0|π0]]：flow-matching VLA 的直接先驱，MoT 架构思路来源；本文用 Qwen3-VL-4B 替代 PaliGemma，并新增 Λ-mask。
- [[2504-Pi05|π0.5]]：training-time RTC 的出处，本文 async 训练范式的起点；同时是 real-robot 主 baseline。
- **Choice Policies**（Qi et al. 2512.25072）：pre-training Step 1 的 multi-candidate + winner-takes-all 范式。
- **DiT / adaLN**（Peebles & Xie）：Diffusion Transformer 架构基础。
- **Flow matching**（Lipman et al. / Liu et al.）：action generation 的目标函数来源。
- **Qwen3-VL-4B-Instruct**：VLM 基座。

### 对比

- **LIBERO**: [[2406-OpenVLA|OpenVLA]]、[[2502-OpenVLA-OFT|OpenVLA-OFT]]、[[2410-Pi0|π0]]、π0-FAST、[[2504-Pi05|π0.5]]、[[2503-GR00TN1|GR00T-N1]]、UniVLA、Discrete Diffusion VLA、MemoryVLA、FLOWER、EO-1。
- **CALVIN**: RoboFlamingo、GR-1、MoDE、[[2412-RoboVLMs|RoboVLMs]]、MDT、UniVLA、FLOWER、SuSIE、3DDA、GR-MG、Seer-Large、VPP。
- **SimplerEnv**: [[2405-Octo|Octo]]、[[2406-OpenVLA|OpenVLA]]、RT-1、[[2307-RT2|RT-2]]-X、Magma、[[2412-RoboVLMs|RoboVLMs]]、[[2410-Pi0|π0]]、π0-FAST、SpatialVLA、ThinkAct、EO-1、MolmoAct。
- **Real-robot**: [[2504-Pi05|π0.5]]（主 SOTA baseline）+ 自己的 Sync / Training RTC 变体。
- **VL preservation**: [[2410-Pi0|π0]]、[[2504-Pi05|π0.5]]、MolmoAct、Qwen3-VL-4B base。

### 方法相关

- **Training RTC / Real-Time Chunking**（Black et al. 2506.07339 / 2512.05964）：action chunk prefix conditioning + async execution 的基础 protocol。本文方法是对它的 fix。
- **Λ-shape attention mask**：原用于 multimodal / streaming generation（MInference、LM-Infinite、StreamingLLM），本文首次应用到 VLA action generation 中。
- **Action Chunking**（Zhao et al. ALOHA 的 action chunk 范式）。
- **Attention sink token**（Xiao et al. StreamingLLM）：DiT 训练稳定化。
- **Grounded SAM / Grounding DINO 1.5 / LLMDet**：VL 数据 grounding 标注的三方共识机制。
- **Knowledge Insulating VLA**（Driess et al. 2505.23705）：防 VLM 被破坏的另一条思路（detach 梯度）——本文选择的是 freeze + co-train 路线。

---
## 论文点评

### Strengths

1. **Λ-shape attention mask 是一个简洁的 first-principles fix**。精准定位到 training-time RTC 的失败机制（prefix shortcut）并用最小侵入的 attention mask 改动解决，比加 regularization 或堆数据的做法更可证伪、更可解释。Towel Folding 的 repetitive fling loop 失败案例给出了直观、非 SOTA-chasing 的 negative example。
2. **VL co-train ablation 做得漂亮**。Table 5 里 `w/o VL data` 全 0 这一行是最有信息量的数字，直接证伪"仅仅是更大 VLM base model 导致 VL 能力"的混淆解释，把 VL co-train 从 nice-to-have 变成 necessity。
3. **Real-time 工程约束形成 end-to-end 闭环**。80 ms @ RTX 4090 + 30Hz resample + $\Delta t_c \geq \Delta t_{\mathrm{inf}}$ 这组数字是能被 practitioner 直接复用的 deployment recipe，有工程说服力。
4. **OOD 场景的 gap 比 in-distribution 大**。CALVIN ABC→D vs ABCD→D、SimplerEnv Drawer Apple vs 其他 task，都呈现"越难 gap 越大"的 pattern，暗示方法对表示质量的改进不是 benchmark-overfitting。

### Weaknesses

1. **"Λ-mask 解决 shortcut"仅是强相关，未闭合因果**。缺少 mask 前后 noisy action token 对 visual token 的 attention 分布对比（entropy、attention mass）；目前只有 throughput 差距和一个 failure video。如果能给出 attention 可视化/定量分析会大大加强。
2. **Real-robot 结果只有 Fig.6c 条形图，没有数字表**。正文口述 "Towel 1.2 vs 1.0 pcs/min"，Lego 的 throughput 具体数字要读图。一个正式 table 会让 claim 更可验证。
3. **LIBERO 98.7% vs EO-1 98.2% 已近饱和噪声**，不应作为核心卖点；real-time 和 VL preservation 才是本文的 novelty。
4. **只开源 inference + fine-tuned checkpoint，训练代码 / VL 数据 curation pipeline / 关键超参缺失**。Beta 分布参数、Λ-mask 窗口 $w$、reweight 具体公式、Grounded SAM + DINO 1.5 + LLMDet 的共识机制阈值——这些都不是从 inference code 能反推的，**实际复现门槛相当高**。
5. **关键超参未做 ablation**：VL:robot 的 1:6 比例、Λ-mask 窗口 $w$——前者可能是拍脑袋，后者决定 reactive vs smooth 的 trade-off。一个 ratio sweep + 窗口 sweep 会让方法更 scalable。
6. **只在两个 in-house 任务验证 post-training**，对 mobile manipulation、接触丰富、非-bimanual embodiment 的泛化没给证据。

### 可信评估

#### Artifact 可获取性

- **代码**: **inference-only**（README 明确只包含 inference + evaluation scripts，无训练代码）
- **模型权重**: 6 个 checkpoint 全部发布在 HuggingFace `XiaomiRobotics` collection——base `Xiaomi-Robotics-0` (4.7B pre-trained)、`Xiaomi-Robotics-0-LIBERO`（4 suite 合并微调）、`Xiaomi-Robotics-0-Calvin-ABCD_D`、`Xiaomi-Robotics-0-Calvin-ABC_D`、`Xiaomi-Robotics-0-SimplerEnv-Google-Robot` (Fractal)、`Xiaomi-Robotics-0-SimplerEnv-WidowX` (Bridge)
- **训练细节**: 部分——pre-train 40k steps / batch 32,768，post-train Lego 40k / Towel 80k / batch 2,048，AdamW + DeepSpeed ZeRO-2，$T=30$ 对应 1 秒，VL:robot=1:6；**Beta 分布参数、Λ-mask 窗口 $w$、reweight 公式、学习率调度**等细节不全。完整复现困难。
- **数据集**: 部分公开。开源：DROID、MolmoAct 轨迹数据、Conceptual 12M/Conceptual Captions/Cambrian-1/FineVision 等通用 VL 数据集。私有：Lego 338h + Towel 400h in-house teleop；VL 数据 curation pipeline（Grounded SAM + DINO 1.5 + LLMDet 共识 + VLM re-label）只有高层描述。

#### Claim 可验证性

- ✅ **三个 sim benchmark SOTA**：数字来自 Table 1/2/4 和 Appendix B；可用发布的 fine-tuned checkpoint 直接复现（README 有每个 benchmark 的 eval 指引）。
- ✅ **VL 能力保留（ERQA 40.8 略超基座 40.0）**：数字清晰，w/o VL data 的全 0 ablation 强支持 catastrophic forgetting 假设；可用发布的 base checkpoint 在标准 VLM evaluator 上独立验证。
- ✅ **80 ms latency on RTX 4090**：消费级硬件 + 明确 inference step 数（5）；可用 inference code 直接测。
- ⚠️ **"Λ-mask 解决 prefix shortcut"**：证据是 Towel fling loop 失败模式 + throughput gap，但**无 attention 分布的定量/可视化证明 fix 的 mechanism**。因果链条没闭合，属于强相关 + 有说服力的 failure narrative，但 mechanism 层面未严格验证。
- ⚠️ **"pre-training recipe 泛化"**：只在两个 in-house 任务 post-train 验证；mobile manipulation、non-bimanual、contact-rich 之外的迁移能力未知。
- ⚠️ **Real-robot throughput（Towel 1.2 pcs/min）**：依赖 evaluation protocol 的实现细节（2 分钟超时、30 分钟 rollout），无独立复现时不易 cross-check。

### Notes

- **1:6 比例是拍脑袋还是 sweep？** 这是最想知道的 recipe 参数。如果没 sweep，那"只要 VL 数据比例非零就不会 forget"是更强的结论——一个 ratio sweep 能区分这两个假设。
- **Λ-mask 窗口 $w$ 没 ablation**。直觉上 $w$ 太小 → transition jerky，$w$ 太大 → 退化回 training-time RTC。这是一个小而有意思的 follow-up 实验，也是本文 fix 的潜在 trade-off 空间。
- **为什么 [[2410-Pi0|π0]] 全 0 而 [[2504-Pi05|π0.5]] 有零星残留？** Table 5 里 [[2504-Pi05|π0.5]] 的数字（SEED 21.5、AI2D 14.4、MMBench 22.1、MMMU 19.9、SciQA 28.0）处于一种"非零但离谱低"的状态，可能反映 [[2504-Pi05|π0.5]] 的训练配方已经部分包含 VL 数据——比对原论文可能揭示其 recipe。
- **命名暗示**："Xiaomi-Robotics-0" 是 version 0，对应 [[2410-Pi0|π0]] → [[2504-Pi05|π0.5]] → [[2604-Pi07|π0.7]] 的命名节奏。可以期待后续 iteration（Λ-mask 参数优化、VL 数据 scaling、mobile manipulation extension）。
- **与 [[2604-Pi07|π0.7]] 的对比**：π0.7 走的是"diverse prompts + heterogeneous data + compositional generalization"的路线，Xiaomi-Robotics-0 走的是"工程 recipe + real-time async + VL preservation"路线。两者关心的维度不同，前者是 generality，后者是 deployability；长期看二者需要合并。

### Rating

**分数**：2 - Frontier
**理由**：方法层面不是 VLA 的奠基工作——MoT、flow-matching、training RTC 都是继承 [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]] 的既有范式，Λ-mask 本身源自 streaming LLM 领域的迁移应用；但它在 real-time async execution + VL preservation 两条正交维度上给出了可复用的 deployment recipe，并在三个主流 sim benchmark（LIBERO/CALVIN/SimplerEnv）上当前 SOTA 且已发布 6 个 fine-tuned checkpoint，是 VLA 方向近期必须对比的 frontier baseline。不升 Foundation 是因为只开源 inference、关键超参（Λ-mask 窗口、Beta 分布、1:6 ratio、reweight 公式）未放，复现门槛高，尚未被后续工作作为 de facto baseline；不降 Archived 是因为 Λ-mask fix 是 training RTC shortcut 问题的 first-principles 解决方案，即使方法被替代，"prefix shortcut 存在且需要显式 fix" 这一 insight 有持久价值。
