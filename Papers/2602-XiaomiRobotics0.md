---
title: "Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution"
authors: [Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, Feng Qiu, Heng Qu, Yifei Su, Qiao Sun, Dong Wang, Donghao Wang, Yunhong Wang, Rujie Wu, Diyun Xiang, Yu Yang, Hangjun Ye, Yuan Zhang, Quanyun Zhou]
institutes: [Xiaomi]
date_publish: 2026-02
venue: arXiv preprint
tags: [VLA, manipulation, flow-matching]
arxiv: https://arxiv.org/abs/2602.12684
website: https://xiaomi-robotics-0.github.io
github: https://github.com/XiaomiRobotics/Xiaomi-Robotics-0
rating: 2
date_added: 2026-04-14
---
## 速查卡片

> [!summary] Xiaomi-Robotics-0
> - **核心**: 4.7B VLM+DiT VLA，用 Λ-shape attention mask 防止 async prefix shortcut，在消费级 4090 上 80ms 延迟平滑实时执行
> - **方法**: Qwen3-VL-4B 冻结 + 16-layer DiT flow-matching + VL data co-train (1:6) + Λ-mask + RoPE offset + dynamic loss reweight
> - **结果**: LIBERO 98.7% / CALVIN 4.80/4.75 / SimplerEnv 85.5-79.2%，real-robot Towel Folding 吞吐 1.2 vs [[2504-Pi05|π0.5]] 的 1.0 pcs/min，ERQA 40.8 略超基座 Qwen3-VL
> - **Sources**: [arxiv](https://arxiv.org/abs/2602.12684) | [website](https://xiaomi-robotics-0.github.io) | [github](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)

---
## Summary

一个 4.7B MoT（VLM + DiT）VLA 模型，通过 Λ-shape 注意力 mask + 动态 loss reweighting + timestep 对齐，在消费级 GPU 上实现平滑的异步实时控制。

**Key Takeaways:**
1. **Λ-shape attention mask 防止 action-prefix shortcut**：training-time RTC 的副作用是 later-timestep 动作直接 copy prefix 而不看视觉/语言；用 Λ-mask 让后段 token 无法 attend prefix，强制转向 visual/language 条件。
2. **VLM + DiT 的 MoT 架构 + KV cache 桥接**：VLM（Qwen3-VL-4B-Instruct）冻结后作为多模态 conditioner，16 层 DiT 从 scratch 训练，只 attend 最后 16 层的 KV cache，控制推理延迟到 80 ms。
3. **预训练阶段 VL 数据 co-train 是关键**：去掉 VL 数据后 VLM 能力在所有 benchmark 归零（catastrophic forgetting），而保留 VL co-train 后 ERQA 甚至略高于基座 Qwen3-VL-4B（40.8 vs 40.0），说明 robot-centric VL 数据反而增强了具身感知。
4. **三个 sim benchmark SOTA**：LIBERO 98.7% avg（14 对比），CALVIN ABC→D 4.75 / ABCD→D 4.80，SimplerEnv Google Robot VM 85.5% / VA 74.7% / WidowX 79.2%。
5. **开源但 inference-only**：放出 base + 5 个 fine-tuned checkpoint 上 HuggingFace，推理和评测代码通过 transformers ecosystem，**未开放训练代码**。

**Video 1.** Overview teaser — Xiaomi-Robotics-0 real-robot rollouts highlight reel
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0.mp4" controls muted playsinline width="720"></video>

---
## Introduction

VLA 模型建立在预训练 VLM 之上，把 observation + instruction 映射到 action。痛点：参数量大 → 推理延迟高 → consecutive chunks 不平滑会引入 OOD jerky 动作。

本文的切入点是 training recipe + deployment strategy 两手兼修：
- 预训练阶段用 cross-embodiment robot trajectory + VL data co-train，避免 catastrophic forgetting 基座 VLM 的 vision-language 能力。
- 后训练阶段为异步执行定制技术：把 previous inference 的 action 作为 prefix condition 生成新 chunk，但用 Λ-shape attention mask 防止 later-timestep 动作走捷径直接 copy prefix。
- 部署阶段严格对齐 consecutive chunks 的 timestep 以保证平滑过渡。

**Figure 1. Overview.** Xiaomi-Robotics-0 在三个 sim benchmark 上 SOTA，在两个真实双手操作任务上吞吐最高，并在多个 VLM benchmark 上匹配基座 VLM。

![](https://arxiv.org/html/2602.12684v2/x1.png)

Headline results（原文正文声明）：LIBERO 98.7%；CALVIN ABCD→D 从 4.67 → 4.80，ABC→D 从 4.54 → 4.75；SimplerEnv VM 85.5%、VA 74.7%、WidowX 79.2%；real-robot 在 Lego Disassembly 和 Towel Folding 上 success rate 和 throughput 都高于 state-of-the-art。发布了 pre-trained + post-trained checkpoints 和 inference code。

---
## Xiaomi-Robotics-0

End-to-end VLA：输入观测图像 + 语言指令 + 本体状态，输出 action chunk 控制双手 bimanual 机器人。

### Data

**Figure 2. Data.** Xiaomi-Robotics-0 在预训练阶段同时使用 robot trajectory 数据和 vision-language (VL) 数据。

![](https://arxiv.org/html/2602.12684v2/x2.png)

- **Robot trajectory**：开源数据集 DROID、MolmoAct 等 + 自采。自采集中在两个长任务：Lego Disassembly（338 小时）和 Towel Folding（400 小时）。总计约 200M timesteps。
- **Vision-language 数据**：~80M samples。两条主干：
  1. 通用 VL 数据集（captioning / VQA / grounding）；
  2. 从 robot trajectory 数据衍生出的 VL 数据，强化 robot-centric 视觉感知（egocentric、wrist camera 视角）。
- 覆盖的四类 VL 任务：visual grounding、VQA、captioning、embodied reasoning & planning。
- Grounding 标注管道：Grounded SAM + Grounding DINO 1.5 + LLMDet 交叉验证保证 pixel 级精度；VQA/captioning 用 SOTA VLM re-label；EQA / high-level planning / point trajectory 用预训练 VLM 从 trajectory 自动生成。

### Model & Training

**Figure 3. Model & Training.** (a) 预训练第一步：在 VL 数据（左，next-token prediction）和 robot trajectory 数据（右，Choice Policies 多 candidate + WTA）上联合训练 VLM。(b) 预训练第二步：冻结 VLM，从 scratch 训练 DiT 通过 flow-matching 生成 action。(c) 后训练 for 异步执行：把 clean action prefix 拼接在 noisy action token 前。

![](https://arxiv.org/html/2602.12684v2/x3.png)

架构：**Mixture-of-Transformers (MoT)**——Qwen3-VL-4B-Instruct 作为 VLM + 一个 16-layer DiT。VLM 处理 $\mathbf{o}_t$ 和语言 $l$，DiT 通过 flow-matching 生成 $T$-step action chunk $\mathbf{a}_{t:t+T}$，conditioned on VLM KV cache 和 proprioceptive state。总参数 4.7B。

#### Pre-training

两步走。

**Step 1**：让 VLM 本身学会动作预测。采用 Choice Policies 的范式处理多模态轨迹——同时预测 $N$ 个 action chunk 候选 + 每个候选的 score。Loss 设计：
- Action prediction 用 **winner-takes-all**，只对 $L_1$ 距离最小的候选做 BP；
- Score prediction 的 target 是每个候选到 ground truth 的 $L_1$ 距离。

Token 序列 `[𝐨_t, l, 𝐬_t, [A_1], ..., [A_T], [S]]`，proprioceptive state 由 MLP 编码，action/score 用额外 learnable token `[A_i]`、`[S]` 预测。VL 数据与 robot 数据按 1:6 比例 co-train（next-token-prediction 目标），这是防 catastrophic forgetting 的关键。

**Step 2**：冻结 VLM，从 scratch 训练 DiT 用 flow-matching 生成 action。Loss：

$$
L(\theta)=\big\|\mathbf{v}_{\theta}(\mathbf{o}_{t},l,\mathbf{s}_{t},\tilde{\mathbf{a}}_{t:t+T}^{\tau},\tau)-\mathbf{u}(\tilde{\mathbf{a}}_{t:t+T}^{\tau},\mathbf{a}_{t:t+T},\tau)\big\|^{2}_{2}
$$

**符号说明**：
- $\tau \in [0, 0.999]$ 是 flow-matching 的 timestep，从 Beta 分布采样（偏向 noisier timesteps）。
- $\tilde{\mathbf{a}}^{\tau}_{t:t+T} = \tau \mathbf{a}_{t:t+T} + (1-\tau)\boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
- $\mathbf{v}_\theta$ 是 DiT 预测的速度场，$\mathbf{u}$ 是 ground-truth 的条件 flow field。

**架构细节**：adaLN 注入 timestep 条件；proprioceptive state 和 noisy action 用 MLP 编码；前置一个 learnable **attention sink token** 稳定训练；DiT 内部 causal attention；DiT 只 condition 在 VLM 最后 16 层的 KV cache 上（减小延迟）；VLM 在 Step 2 只接收 $\mathbf{o}_t$ 和 $l$，不含 Step 1 引入的 action token。

#### Post-training

针对特定机器人，只用该机器人的 trajectory 数据继续训练。分 synchronous 和 asynchronous 两种模式。

- **Sync**：简单 unfreeze 整个模型（VLM + DiT），继续 flow-matching 训练。
- **Async**：遵循 training RTC 的做法，把 $\Delta t_c$ 个 previously committed actions 作为 clean prefix 拼接到 noisy action tokens 前。DiT 序列变成：`[SINK], 𝐬_t, 𝐚_t, ..., 𝐚_{t+Δt_c-1}, 𝐚̃^τ_{t+Δt_c}, ..., 𝐚̃^τ_{t+T-1}`。

**Problem**：这种 prefix conditioning 让 later-timestep 的 noisy action 直接 copy prefix（shortcut），不看 visual/language，reactive 性下降。

**Fix**：

**Figure 4. Λ-Shape Attention Mask for Post-Training.** Noisy action token 只能 attend VLM KV cache、sink token、state token、以及前 $w$ 个 timestep 的 action tokens。Token 上的数字是 RoPE positional index；noisy action token 的 RoPE 被加了 offset 10，用来和 clean prefix token 区分。

![](https://arxiv.org/html/2602.12684v2/x4.png)

两个 trick：
1. **RoPE offset**：给 noisy action token 的 positional index 加 10，让模型能区分它和 clean prefix。
2. **Λ-shape attention mask**：noisy action token 只能 attend 到紧邻 prefix 结尾的窗口 + VLM KV cache + sink + state；更后面的 noisy action 无法再看到 prefix，被迫去 attend 视觉/语言。

**训练采样**：$\Delta t_c$ 从 $\{0, 1, \ldots, 6\}$ 均匀采样。当 $\Delta t_c > 0$ 时，根据 **online-predicted actions** 相对 ground truth 的 $L_1$ error 动态 reweight flow-matching loss，重点学习偏差大的样本。

### Deployment

**Figure 5. Asynchronous Execution.** 可视化两个连续 chunk 如何在机器人 rollout 中被拼接。

![](https://arxiv.org/html/2602.12684v2/x5.png)

**Synchronous Execution**：每次执行 chunk 的前 $T_e$ 步 → idle → 用最新观测推新 chunk。

**Asynchronous Execution**：
- 执行 $T_e$ 步后立即触发下一轮推理；
- 推理期间机器人继续执行当前 chunk 的剩余动作；
- 用步 $T_e$ 到 $T_e + \Delta t_c - 1$ 的动作作为 prefix condition 新 chunk；
- 新 chunk 从步 $\Delta t_{\mathrm{inf}}$ 开始执行，其中 $\Delta t_{\mathrm{inf}}$ 是推理延迟；
- 要求 $\Delta t_c \geq \Delta t_{\mathrm{inf}}$ 保证 prefix 覆盖整个推理窗口，实现无缝衔接。

推理细节：action chunk 从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样后执行 5 步 flow-matching 积分（$\tau: 0 \to 1$）。**NVIDIA RTX 4090** 上延迟 $t_{\mathrm{inf}} = 80$ ms。所有模态在 30Hz timeline 上 resample 对齐，每个 tick 取时间最近的测量聚合成输入。

---
## Experiments

### Simulation Benchmarks

三个 sim benchmark：LIBERO、CALVIN、SimplerEnv。

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

**Insights**：在 Libero-Long（最考验 long-horizon）上把次优的 FLOWER 94.9% 提升到 97.2%，是本表最大 gap。

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

**Insights**：ABC→D 是 OOD 泛化 split（训练时没见过 D），本方法把 Task-5 成功率从 FLOWER 的 77.8% 拉到 88.1%——10 个点的 gap 比 ABCD→D 的 in-distribution split 更大，说明 VL co-train 带来的表示质量对 OOD 有实质增益。

**Benchmark action chunk 长度**：LIBERO $T=10$、CALVIN $T=10$、SimplerEnv $T=4$。SimplerEnv 详细数字在 Appendix B，笔记不展开。

### Real-Robot Experiments

**Figure 6. Real-Robot Experiments.** (a) Lego Disassembly 评测设置。(b) Towel Folding 评测设置和使用的 6 种毛巾。(c) 两个任务上的定量结果。

![](https://arxiv.org/html/2602.12684v2/x6.png)

**平台**：bimanual robot，两个 6-DoF arm，三个相机（2 个 wrist-mounted + 1 个 global external）。

**Tasks**：
- **Lego Disassembly**：拆解 Lego 结构为单独砖块，按颜色分拣到对应 bin。要求 bimanual 协同抓取 + 精确放置。两个 setting：LA（large-assembly，LA-5 / LA-10 / LA-20 三个尺寸）和 MA（multi-assembly，共 34 块含单块和 2-3 块组合）。Metric：正确分拣砖块数 / 总砖块数（success rate）和 throughput（单位时间分拣数）。
- **Towel Folding**：从托盘取毛巾 → 展平 → 对折两次 → 放到 staging area。毛巾 deformable，动力学复杂。用 6 条不同毛巾，每个方法跑两次 30 分钟 rollout；单次折叠超过 2 分钟算失败；metric 是 throughput（成功折叠数 / rollout 时间）。

**Implementation**：
- **Baselines**：[[2504-Pi05|π0.5]]（SOTA VLA）、Xiaomi-Robotics-0（主方法，异步）、Xiaomi-Robotics-0 (Sync)、Xiaomi-Robotics-0 (Training RTC)。[[2504-Pi05|π0.5]] 按官方 OpenPi 的 fine-tune 协议从 release 的 base 启动。
- **Training**：pre-train 40k steps, batch 32,768；post-train Lego 40k steps、Towel 80k steps, batch 2,048；AdamW + DeepSpeed ZeRO-2；**action chunk $T=30$ 对应 1 秒动作**。

**Results**（文中叙述，定量在 Fig.6c 不抽数字）：
- Lego Disassembly：所有方法 avg success rate 相当，sync 方法（[[2504-Pi05|π0.5]] 和 Xiaomi-Robotics-0 Sync）精度略高因为反应性更好（异步 inference 的动作生成稍晚，导致砖块 tension 爆开）。但 throughput 上 Xiaomi-Robotics-0 (Sync) 就超过 [[2504-Pi05|π0.5]]；Xiaomi-Robotics-0（异步）throughput 最高。
- Towel Folding：[[2504-Pi05|π0.5]]、Xiaomi-Robotics-0 (Sync)、Xiaomi-Robotics-0 (Training RTC) 都是 1 pcs/min；**Xiaomi-Robotics-0 达到 1.2 pcs/min**。Training RTC 变体有一个典型 failure：flinging 时不小心抓到两层毛巾会陷入重复 fling loop 无法自救——印证了 prefix shortcut 假设（模型只 copy prefix 而不 attend 当前观测）。

**Video 2. Lego Disassembly — complex task manipulation.**
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-Lego-disassembly-complex-task-manipulation.mp4" controls muted playsinline width="720"></video>

**Video 3. Lego Disassembly — flexible motion switch.**
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-Lego-disassembly-flexible-%20motion-switch.mp4" controls muted playsinline width="720"></video>

**Video 4. Towel Folding — fling motion.**
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-towel-folding-fling.mp4" controls muted playsinline width="720"></video>

**Video 5. Towel Folding — put back.**
<video src="https://robotics.xiaomi.com/robot-model/xiaomi-robotics-0-towel-folding-put-back.mp4" controls muted playsinline width="720"></video>

### Preservation of Vision-Language Capabilities

**Table 3. Quantitative results on general vision-language and embodied reasoning benchmarks.**

| Model | ERQA | SEED | POPE | AI2D | MMBench | MME | MMMU | TextVQA | SciQA | ChartQA |
|---|---|---|---|---|---|---|---|---|---|---|
| [[2410-Pi0\|π0]] | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 1.4 | 0.0 | 0.0 |
| [[2504-Pi05\|π0.5]] | 0.0 | 21.5 | 0.0 | 14.4 | 22.1 | 0.0 | 19.9 | 0.0 | 28.0 | 0.5 |
| MolmoAct | 33.5 | 72.7 | 86.6 | 72.0 | 80.1 | 69.5 | 38.0 | 67.3 | **91.1** | 57.1 |
| Xiaomi-Robotics-0 (w/o VL data) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Xiaomi-Robotics-0 (Ours)** | **40.8** | **78.6** | **88.5** | **78.7** | **84.4** | **81.8** | **46.2** | **72.0** | 79.4 | **59.2** |
| Qwen3-VL-4B-Instruct (base) | 40.0 | 78.8 | 89.7 | 81.6 | 88.7 | 87.1 | 51.7 | 78.0 | 92.7 | 76.8 |

**Insights**：
- **[[2410-Pi0|π0]] / [[2504-Pi05|π0.5]] 几乎全归零**：说明单纯在 robot trajectory 上训练会彻底摧毁基座 VLM 的 VL 能力。[[2504-Pi05|π0.5]] 在个别 benchmark 上有极小值可能是训练配方不同。
- **Xiaomi-Robotics-0 (w/o VL data) = 全 0**：本方法自己做 ablation，证明不 co-train VL 数据也会 catastrophic forgetting。
- **Ours 在 9/10 benchmark 上领先所有 VLA 基线，并在 ERQA 上略超基座**（40.8 vs 40.0）——作者假设是因为 robot-trajectory-derived VL 数据加强了 robot-centric 场景下的 embodied perception。其他 benchmark 略落后基座但幅度小（如 AI2D 78.7 vs 81.6）。
- **Vs MolmoAct**：唯一在 SciQA 上输给 MolmoAct（79.4 vs 91.1），其他 9 个 benchmark 全部领先。

---
## 论文点评

### Strengths

1. **Λ-shape attention mask 是一个简洁、first-principles 的 fix**：精准定位到 training-time RTC 的根本问题（prefix shortcut），并用最小侵入的 attention mask 改动解决。比单纯加 regularization 或扩大数据更可证伪、更可解释。
2. **VL co-train ablation 做得漂亮**：Table 3 里 "Xiaomi-Robotics-0 (w/o VL data) 全 0" 这一行是整篇 report 最有信息量的数字——直接证明 VL co-train 不是 nice-to-have 而是必需品，同时排除了"仅仅是更大 VLM base model 导致 VL 能力"的混淆解释。
3. **Real-time 工程约束的 end-to-end 闭环**：80 ms 延迟 @ RTX 4090 + 30Hz resample + $\Delta t_c \geq \Delta t_{\mathrm{inf}}$ 的 deployment 约束，这组数字是能被 industry practitioner 直接复用的 deployment recipe。
4. **硬证据支撑 Training RTC failure mode**：Towel Folding 的 repetitive fling loop 是一个直观、不可辩驳的 negative example，直接 visualize 了 shortcut 假设。这种 failure case 比 +0.3% SOTA 更有说服力。

### Weaknesses

1. **"Training RTC 有 shortcut" 和 "Λ-mask 能 fix" 是强关联但非严格因果**：虽然结论符合直觉，但缺少对 mask 后 attention 分布的定量分析（比如 attention 权重 entropy、visual token 的 attention mass 占比），只有 throughput 结果。如果能给出 mask 前后 noisy action token 对 visual token 的 attention 比例变化，会更有说服力。
2. **Real-robot 结果在 Figure 6c 中只有图，没有给出 throughput 的绝对数字表**：正文只提到 "Towel Folding 1.0 vs 1.2 pcs/min"，Lego 的具体数字要读图。一个正式 throughput/success table 会更可验证。
3. **LIBERO 98.7% vs EO-1 98.2% 的 gap 已经接近 benchmark 饱和噪声**，不能作为核心卖点；论文主要 novelty 应更清晰集中在 real-time + VL preservation 两条线。
4. **只开源 inference + fine-tuned checkpoint，不开源训练代码和 VL 数据 curation pipeline**。考虑到 VL co-train 是本方法的关键，这个 pipeline 的缺失会显著阻碍复现——例如 Grounded SAM + DINO 1.5 + LLMDet 三者的 cross-validation 机制具体如何实现、阈值设多少，都不是简单从 inference code 能反推的。
5. **Post-training 的 Beta 分布和 $\Delta t_c$ loss reweight 细节（具体 Beta 参数、reweight 公式）在正文没给全**，需要看 appendix 或者代码才能复现。
6. **Λ-mask 的具体形状参数 $w$ 没有 ablation**：窗口大小如何选、对 reactive-vs-smooth 的 trade-off 在哪里，文章只给了一个"用"而没给"为什么这么用"。

### 可信评估

#### Artifact 可获取性

- **代码**: inference-only（README 明确只包含 inference code 和 evaluation scripts，无 training 代码）
- **模型权重**: 6 个 checkpoint 全部发布在 HuggingFace 的 XiaomiRobotics collection —— base pre-trained `Xiaomi-Robotics-0` (4.7B)、`Xiaomi-Robotics-0-LIBERO`（4 个 LIBERO suite 微调）、`Xiaomi-Robotics-0-Calvin-ABCD_D`、`Xiaomi-Robotics-0-Calvin-ABC_D`、`Xiaomi-Robotics-0-SimplerEnv-Google-Robot`（Fractal 微调）、`Xiaomi-Robotics-0-SimplerEnv-WidowX`（Bridge 微调）
- **训练细节**: 高层描述 + 部分超参（pre-train 40k steps batch 32,768；post-train 40k/80k steps batch 2,048；AdamW + DeepSpeed ZeRO-2；$T=30$）。Beta 分布参数、Λ-mask 窗口 $w$、dynamic loss reweight 公式等细节不全；VL 数据比例 1:6 给出。完整复现困难。
- **数据集**: 部分公开。开源部分：DROID、MolmoAct、VL 通用数据集（引用）。私有部分：Lego Disassembly 338 小时 + Towel Folding 400 小时 in-house teleoperation 数据；VL 数据的 curation pipeline（Grounded SAM + DINO 1.5 + LLMDet 共识机制 + VLM re-label）只有高层描述。

#### Claim 可验证性

- ✅ **三个 sim benchmark SOTA（LIBERO 98.7%、CALVIN 4.80/4.75、SimplerEnv 85.5/74.7/79.2）**：数字来自论文 Table 1、Table 2、Appendix，可通过发布的 fine-tuned checkpoint 直接复现评测（README 里给了每个 benchmark 的 eval 指引）。
- ✅ **VL 能力保留（Table 3 ERQA 40.8 略超基座 40.0）**：数字和方法清晰；ablation 的 w/o VL data 全 0 强支持了 catastrophic forgetting 假设。可通过发布的 base checkpoint 在任意标准 VLM evaluator 上独立验证。
- ✅ **80 ms inference latency on RTX 4090**：消费级 GPU 的明确硬件 + 数字 + 推理步数（5 flow-matching steps）组合，可用发布的 inference code 直接测。
- ⚠️ **"Λ-shape mask 解决 prefix shortcut"**：论文给出的证据是 Towel Folding 的 repetitive fling loop 失败模式 + throughput gap，但没有 attention 可视化或 counter-example 证明 fix 的 mechanism 确实是通过改变 attention 分布实现的。相关性强，但因果链条没有闭合。
- ⚠️ **"pre-training recipe 泛化到新机器人 / 新任务"**：只在两个 in-house 任务（Lego、Towel）上验证 post-training。是否能泛化到移动操作、接触丰富的 manipulation、或 bimanual 之外的 embodiment 未知。
- ⚠️ **Real-robot throughput 数字（Towel 1.2 pcs/min）**：依赖 evaluation protocol 的实现细节（2 分钟超时判定、30 分钟 rollout 等），无独立复现时不易 cross-check。

---
## 关联工作

### 基于

- [[2410-Pi0|π0]]: Flow-matching VLA，MoT 架构的直接先驱。本文的 DiT + 冻结 VLM 架构思路来自 [[2410-Pi0|π0]]，但用 Qwen3-VL-4B 替代 PaliGemma，并加入 Λ-mask 创新。
- [[2504-Pi05|π0.5]]：training-time RTC（action prefix conditioning）的出处，本文的 async 训练范式直接基于它。
- Choice Policies：pre-training Step 1 用的 multi-candidate + winner-takes-all 范式来自这里。
- **DiT / adaLN**：Diffusion Transformer 架构（Peebles & Xie）。
- **Flow matching**：Lipman et al. / Liu et al. 的条件 flow matching 作为 action 生成的目标。

### 对比

- [[2406-OpenVLA|OpenVLA]] / [[2502-OpenVLA-OFT|OpenVLA-OFT]]: LIBERO 上的开源 VLA baseline。
- [[2503-GR00TN1|GR00T-N1]]: NVIDIA 的 foundation 模型，LIBERO 上对比。
- [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]] / π0-FAST：Physical Intelligence 的 VLA 系列，三个 sim benchmark 全部对比 + Table 3 里 VL 能力对比。
- FLOWER, EO-1, MemoryVLA, UniVLA, Discrete Diffusion VLA：LIBERO + CALVIN 次优 baseline。
- MolmoAct：Table 3 的 VL 能力对比，作为另一条 "VLA 模型保留 VLM 能力" 的路线。

### 方法相关

- **Training RTC / Real-Time Chunking**：action chunk prefix conditioning + async execution 的基础 protocol。
- **Λ-shape attention mask**：之前用于 multimodal / streaming generation 的 mask pattern（cited refs 20, 71, 16），本文首次应用到 VLA action generation。
- **Action Chunking**（Zhao et al.）：action chunk 预测范式的基础。
- **Qwen3-VL-4B-Instruct**：VLM 基座。
- **Grounded SAM / Grounding DINO 1.5 / LLMDet**：VL 数据 grounding 标注的三方共识机制。

---
## Notes

- 真正让我关心的一个问题：VL co-train 的 1:6 比例是拍脑袋还是 sweep 出来的？如果是拍的，那可能意味着"只要 VL 数据比例非零就不会 forget"是更强的结论——需要一个 ratio sweep 来确认。
- Λ-mask 的窗口大小 $w$ 没 ablation。直觉上 $w$ 太小会让 transition 不平滑（还是有 jerky），$w$ 太大会退化回 training-time RTC。这是一个小而有意思的 follow-up 实验。
- 注意 Table 3 里 [[2504-Pi05|π0.5]] 的 VL benchmark 数字非常奇怪（SEED 21.5 / AI2D 14.4 / MMBench 22.1 / MMMU 19.9 / SciQA 28.0）——这些不是 0 但也不是正常水平，像是某种 partial preservation。为什么 [[2410-Pi0|π0]] 完全归零而 [[2504-Pi05|π0.5]] 有零星残留？可能和 [[2504-Pi05|π0.5]] 的训练配方（更多 VL 数据 / post-training 策略）有关，值得比对 [[2504-Pi05|π0.5]] 原论文。
- Paper name 是 "Xiaomi-Robotics-0"，暗示后面会有 1、2，类似 [[2410-Pi0|π0]] → [[2504-Pi05|π0.5]]。本 report 是 v1.0 的内部代号。
