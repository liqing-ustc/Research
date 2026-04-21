---
title: "PROSPECT: Unified Streaming Vision-Language Navigation via Semantic–Spatial Fusion and Latent Predictive Representation"
authors: [Zehua Fan, Wenqi Lyu, Wenxuan Song, Linge Zhao, Yifei Yang, Xi Wang, Junjie He, Lida Huang, Haiyan Liu, Bingchuan Sun, Guangjun Bao, Xuanyao Mao, Liang Xu, Yan Wang, Feng Gao]
institutes: [Shanghai Jiao Tong University, Tsinghua University, University of Adelaide, Wuhan University, HKUST(GZ), Beijing Jiaotong University, Lenovo]
date_publish: 2026-03-04
venue: arXiv
tags: [VLN, VLA, world-model, spatial-reasoning]
paper: https://arxiv.org/abs/2603.03739
website:
github:
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] PROSPECT: Unified Streaming VLN via Semantic–Spatial Fusion and Latent Predictive Representation
> - **核心**: 把 streaming VLA 与 latent-space 未来表示预测统一在单一 streaming 模型里，做 mapless / odometry-free 的 VLN-CE。
> - **方法**: SigLIP（2D 语义） + CUT3R（streaming 3D, 绝对尺度）cross-attention 融合喂 LLM；训练阶段引入 stream query tokens，反向 query 流式上下文，分别预测下一步 2D/3D latent，分别用 cosine（SigLIP）和 MSE（CUT3R）对 frozen teacher 监督；推理时丢掉预测分支，零额外开销。
> - **结果**: VLN-CE R2R/RxR val-unseen 取得 first-tier（R2R SR 58.9 / SPL 54.0；RxR SR 54.6），long-horizon RxR 增益显著大于 R2R；ARX-Lift2 真机在不同光照下显著优于 NaVid / StreamVLN。
> - **Sources**: [paper](https://arxiv.org/abs/2603.03739)
> - **Rating**: 2 - Frontier（streaming VLN 上把 JEPA-style latent prediction 接入 VLA 的代表工作之一，方法干净、real-robot 有效，但 code/checkpoint 未放且核心 claim 缺直接 ablation，还达不到 Foundation 档）

**Key Takeaways:**
1. **Latent-space 预测优于像素/显式模态预测**: 借鉴 JEPA，预测 SigLIP 和 CUT3R 的 latent，而不是 RGB / depth / occupancy，避免 overfit 到 texture / illumination 等任务无关因素。
2. **CUT3R 比 VGGT 系更适合 streaming VLN**: VGGT 要 OOM、且只给相对尺度；CUT3R 天生 streaming + 绝对尺度，在长 episode 上同时更准更快（R2R SR 48.7 vs InfiniteVGGT 43.2，0.245s vs 0.284s/step）。
3. **Inference-free 的预测分支**: 通过专门设计的 streaming attention mask（causal + 跨 turn query 隔离 + 2D/3D query 互不可见），把 world-model 信号训练时注入 representation，推理时整支砍掉，零 latency 代价。
4. **长程任务收益最大**: 短任务（1–50 步）几乎追平 baseline，medium / long horizon 上 SR 各涨 ~4 个点，real-robot 在低照度场景同样保持显著优势——表征里的预测先验对 distribution shift 有帮助。

**Teaser. PROSPECT 的总览：streaming attention mask + SigLIP/CUT3R 双流融合 + 训练时 latent 预测分支 + 真机部署示意。**

![](https://arxiv.org/html/2603.03739v1/x1.png)

---

## 1 背景与动机

VLN 的现状：
- MLLM-based 端到端 VLN（[[2402-NaVid|NaVid]]、Uni-NaVid、[[2412-NaVILA|NaVILA]]、[[2507-StreamVLN|StreamVLN]]）已让 zero-shot VLA-style 单视角 RGB 流派接近 panoramic+depth+odometry 系。
- World model / 预测分支被认为有助于 robust navigation（[[2506-VJEPA2|V-JEPA2]]、Genie 系、[[2602-WorldVLALoop|WorldVLA]]、DreamVLA），但在 VLN 里要么用低维 state-space（NavMorph），要么在像素/显式 modality 上监督（NavForesee），后者容易 overfit 到 texture / lighting。
- 主流 2D 视觉 encoder（SigLIP）缺空间智能；3D foundation model（VGGT 系、CUT3R）开始用于 VLA。VGGT 在长 episode 易 OOM，且是 first-frame 相对尺度，长视角变化下不稳定。CUT3R 天生 streaming + 绝对尺度，更适合长程导航。

PROSPECT 的核心论点：**unified streaming model = streaming VLA + 在 SigLIP/CUT3R latent 空间里的预测分支 + 推理时整支砍掉**。

> ❓ "latent prediction 不会 overfit 到 texture" 这个论点合理但缺直接证据——文里没单独跑 pixel-target vs latent-target 的对照，只能间接通过总分和真机鲁棒性推断。一个直接的 ablation 会更有说服力。

---

## 2 方法

### 2.1 Problem Formulation: Streaming VLA for VLN

把 streaming VLN 当成 streaming VLA 问题。给定指令 $I$，每个时间步 $t$ agent 接收单视角 RGB $o_t \in \mathbb{R}^{3 \times H \times W}$，输出 atomic action 序列 $a_t = (a_t^{(1)}, \ldots, a_t^{(n_a)})$，$n_a = 4$，动作集 $\mathcal{A} = \{\uparrow, \leftarrow, \rightarrow, \texttt{STOP}\}$（前进 25 cm / 左右转 15°）。

短期 sliding window $\mathcal{W}_t$ 包含 $N{-}1$ 对历史 (obs, act)，长期记忆 $M$ 由均匀采样的关键帧汇总而来：

$$
\mathrm{Stream}_{0:t} := \{ \mathrm{KV}(\mathcal{W}_t),\ o_t,\ M \}
$$

policy 形如 $a_t = \text{VLA}(I,\ \mathrm{Stream}_{0:t})$。

### 2.2 Unified Model: Action + Latent Prediction

PROSPECT 把 VLA 和 next-step latent 预测合并成一个流：

$$
a_t,\ \mathbf{F}^{2\text{D}}_{t+1},\ \mathbf{F}^{3\text{D}}_{t+1} = \text{UM}(I,\ \mathrm{Stream}_{0:t})
$$

VLA 分支吃融合后的 2D-3D 特征做 autoregressive action；训练时附加按时间排序的 query tokens reverse-query 上下文，由轻量 decoder 预测下一步 2D/3D latent。

**Figure 2. PROSPECT 架构。** 指令和观察走同一管线：frozen SigLIP + CUT3R + cross-attention 融合，关键帧被压缩成 long-term memory M。LLM 自回归出 action；训练时 2D/3D query token reverse-query 流，轻量 decoder 在 frozen teacher 监督下预测 next-step latent。推理只跑 VLA。

![](https://arxiv.org/html/2603.03739v1/x2.png)

### 2.3 Perception: 2D-3D Fusion

- 2D：$\mathbf{F}^{2\text{D}}_t = \text{SigLIP}(o_t)$。
- 3D：CUT3R 先 ViT encoder 编 $\mathbf{F}^{3\text{D,pre}}_t$，再用 (state $\mathbf{s}_{t-1}$, learnable pose $\mathbf{p}_t$) 滚动出 streaming spatial feature 和新的 state：

$$
[\mathbf{p}'_t,\ \mathbf{F}^{3\text{D}}_t],\ \mathbf{s}_t = \text{Decoders}([\mathbf{p}_t,\ \mathbf{F}^{3\text{D,pre}}_t],\ \mathbf{s}_{t-1})
$$

- 融合：以 2D 为 query 跨注意 3D：

$$
\mathbf{F}^{\text{fuse}}_t = \text{softmax}\!\left(\frac{(\mathbf{F}^{2\text{D}}_t \mathbf{W}_Q)(\mathbf{F}^{3\text{D}}_t \mathbf{W}_K)^\top}{\sqrt{d_k}}\right)(\mathbf{F}^{3\text{D}}_t \mathbf{W}_V)
$$

每个 $\mathbf{F}^{\text{fuse}}_t$ 经 MLP 进 LLM embedding 空间。长期记忆 M 中每个关键帧也走相同管线，再压缩为单 token 喂 LLM。

### 2.4 Latent Prediction via Stream Query Tokens

每步 $t$ 在 LLM 输入末端追加 learnable token $\langle q^{2\text{D}}_t \rangle$ 和 $\langle q^{3\text{D}}_t \rangle$，让 LLM 把流上下文压成 $t+1$ 时刻的 compact embedding：

$$
\mathbf{e}^{2\text{D}}_{t+1} = \text{LLM}(I,\ \mathrm{Stream}_{0:t}\ |\ \langle q^{2\text{D}}_t \rangle),\quad
\mathbf{e}^{3\text{D}}_{t+1} = \text{LLM}(I,\ \mathrm{Stream}_{0:t}\ |\ \langle q^{3\text{D}}_t \rangle)
$$

两个 2 层 Transformer decoder 配 learnable masked tokens（$\langle m_t^{2\text{D}}\rangle$, $\langle m_t^{3\text{D}}\rangle$，重复到目标 token 长度）做 token-level latent reconstruction：

$$
\widehat{\mathbf{F}}^{2\text{D}}_{t+1} = \text{Decoder}_{2\text{D}}(\mathbf{e}^{2\text{D}}_{t+1}\ |\ \langle m_t^{2\text{D}} \rangle),\quad
\widehat{\mathbf{F}}^{3\text{D}}_{t+1} = \text{Decoder}_{3\text{D}}(\mathbf{e}^{3\text{D}}_{t+1}\ |\ \langle m_t^{3\text{D}} \rangle)
$$

**Loss**：target 来自 frozen SigLIP/CUT3R teacher，2D 用 cosine，3D 用 MSE：

$$
\mathcal{L}_{2\text{D}} = 1 - \cos\!\left(\widehat{\mathbf{F}}^{2\text{D}}_{t+1},\ \mathbf{F}^{2\text{D}}_{t+1}\right),\quad
\mathcal{L}_{3\text{D}} = \text{MSE}\!\left(\widehat{\mathbf{F}}^{3\text{D}}_{t+1},\ \mathbf{F}^{3\text{D}}_{t+1}\right)
$$

$$
\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{nav}} + \gamma\,(\alpha\,\mathcal{L}_{2\text{D}} + \beta\,\mathcal{L}_{3\text{D}})
$$

作者解释：SigLIP 在 $\ell_2$-normalized embedding 上做 sigmoid pairwise loss，cosine 与该几何对齐；MSE 加在 SigLIP 上会惩罚 norm 差异，训练不稳；CUT3R 上 MSE 反而稳定。

> 这个细节挺关键——"loss 选哪个" 经常被当作工程小事，但这里直接关系到 representation 的 normalize 几何与 teacher loss 的一致性。**Generalizable lesson**: 做 latent distillation / JEPA-style 监督时，先看 teacher 自己被训练时的几何（normalize 与否、用什么距离），目标 loss 最好同构。

### 2.5 Streaming Attention Mask

把短期上下文当 N 轮"对话"：每轮 $i$ 有 context $\text{ctxt}_i$（prompt + obs token）和 response $\text{act}_i$；首轮还含指令 + 长期记忆 M。训练时每轮末尾追加 $\langle q^{2\text{D}}_i\rangle$、$\langle q^{3\text{D}}_i\rangle$。三条约束：

1. **Causality**：query 只看自己当前轮 + 之前所有轮，不看未来。
2. **Cross-turn isolation**：不同轮的 query 互不可见，避免 query-to-query 信息渗漏 / 误差累积。
3. **Modality disentanglement**：同一轮内 2D / 3D query 互相 mask，避免 cross-task interference。

推理移除 query 分支后，剩下的 token 排序与 attention 结构与训练保持一致。Fig. 3 展示 mask 结构。

**Figure 3. Streaming attention mask。** 灰色：navigation context / action 走标准 causal；红色：每个 2D query 只能 attend 自己当前轮 ctxt/act 与之前所有轮，不可 attend 任何其他 Query2d / Query3d / 未来轮；蓝色：3D query 同理。

![](https://arxiv.org/html/2603.03739v1/x3.png)

> ❓ Cross-turn query 隔离的代价是 query 之间无法互相 condition——比如 2D query 看不到上一轮 2D query 抽出的预测信号。文章说这是为了避免 error accumulation；但如果 next-step prediction 真的有用，那 t 时刻的预测对 t+1 时刻的 query 应当也有信息量。这点能否通过 short-context 跨轮 attend 拿回来，可以做 ablation。

---

## 3 实验设置

- **Backbone**：StreamVLN 作 baseline，LLaVA-NeXT-Video-7B + Qwen1.5-7B；短期窗口 $N=8$，长期记忆采样 8 个 keyframe。
- **训练**：8×A800 两阶段。
  - **Stage 1 SFT**（一个 epoch，560 GPU-hr）：MP3D 上 R2R + RxR + R2R-EnvDrop（共 ~479K，比例 5/14/80%）。
  - **Stage 2 Augmented SFT**（一个 epoch，~1900 GPU-hr）：保留 Stage 1 R2R/RxR 防遗忘 + ~260K DAgger 样本（专家重标 off-policy 漂移的恢复动作） + ~314K ScaleVLN 样本（HM3D），混 LLaVA-Video-178K + ScanQA 做空间 / 几何 VQA。Stage 2 总 ~938K（71% VLN / 29% VQA）。
- **超参**：SigLIP lr 5e-6，其他可训模块 lr 2e-5，CUT3R 全 frozen；warmup 7.5% / 3%；loss 系数 $\gamma=0.01,\ \alpha=0.25,\ \beta=0.75$；196 masked token + 9 query token / modality。
- **评测**：VLN-CE in Habitat，R2R / RxR val-unseen 上的 SR / SPL / NE / OSR / nDTW。
- **真机**：ARX-Lift2，head-mounted RealSense 405 单视角 RGB。室内 dual-RTX-4090 / 室外 dual-A800 远程推理 Wi-Fi/LAN，~0.25–0.27 s/step（~4 Hz）；onboard 单 RTX 4070 + 降精度可用但成功率下降。

---

## 4 结果

### 4.1 VLN-CE 主表

**Table I. VLN-CE R2R / RxR val-unseen 主结果（节选）。** PROSPECT 单视角 RGB，无 depth / odometry / panorama，在两种数据 regime 下都取 first-tier。

| Method | Obs. | R2R SR ↑ | R2R SPL ↑ | RxR SR ↑ | RxR SPL ↑ | RxR nDTW ↑ |
|---|---|---|---|---|---|---|
| [[2402-NaVid\|NaVid]] [RSS24] | RGB | 37.4 | 35.9 | – | – | – |
| Uni-NaVid [RSS25] | RGB | 47.0 | 42.7 | 48.7 | 40.9 | – |
| [[2412-NaVILA\|NaVILA]] [RSS25] | RGB | 49.7 | 45.5 | – | – | – |
| [[2507-StreamVLN\|StreamVLN]]\* [arXiv25] | RGB | 50.8 | 45.7 | 48.6 | 42.5 | 60.2 |
| **PROSPECT (Ours)\*** | RGB | **52.0** | **46.2** | **52.7** | **42.8** | **60.6** |
| [[2412-NaVILA\|NaVILA]]† [RSS25] | RGB | 54.0 | 49.0 | 49.3 | 44.0 | 58.8 |
| [[2507-StreamVLN\|StreamVLN]]† [arXiv25] | RGB | 55.7 | 50.9 | 52.9 | 46.0 | 61.9 |
| **PROSPECT (Ours)†** | RGB | **58.9** | **54.0** | **54.6** | **46.2** | **62.1** |

\* MP3D + VideoQA only；† 加 ScaleVLN + MMC4。

值得注意：RxR（长指令~120 词、平均 15.32 m）增益明显大于 R2R（短指令 32 词、9.89 m）——SR 提升 +1.7 → +4.1（†），SPL 也同步抬升。作者据此 claim "对长程指令更有效"。

### 4.2 Module Ablation

**Table II. R2R val-unseen 的模块 ablation（一个 epoch SFT）。**

| Setting | NE ↓ | OSR ↑ | SR ↑ | SPL ↑ |
|---|---|---|---|---|
| Baseline (SigLIP only) | 6.05 | 53.8 | 45.5 | 41.6 |
| Ours (SigLIP + CUT3R) | 5.91 | 55.0 | 46.7 | 41.8 |
| Ours (+ WM-2D only) | 5.89 | 56.0 | 47.0 | 42.0 |
| Ours (+ WM-3D only) | 5.90 | 55.4 | 47.2 | 41.9 |
| Ours (+ WM-2D + WM-3D) | **5.82** | **57.6** | **48.7** | **42.9** |

- SigLIP+CUT3R 融合 alone：SR +1.2，OSR +1.2。
- 2D / 3D 预测各自再 +0.3–0.5 SR。
- 两个预测目标合起来 SR +2.0，OSR +2.6——complementary，但单项增益不算大，主要靠组合。

> ❓ 单项预测目标的增益几乎在噪声边缘（+0.3 SR），"complementary" 主要靠最后一行联合训练的 +2 SR 撑起来。建议独立 seed 多跑几遍确认稳定性。

### 4.3 Spatial Encoder: CUT3R vs VGGT-style

**Table III. R2R val-unseen 上的 spatial encoder 对比。**

| Encoder | Time (s) | SR ↑ | SPL ↑ | OSR ↑ | NE ↓ |
|---|---|---|---|---|---|
| VGGT | OOM | OOM | OOM | OOM | OOM |
| InfiniteVGGT | 0.284 | 43.2 | 38.0 | 54.4 | 6.61 |
| **Ours (CUT3R)** | **0.245** | **48.7** | **42.9** | **57.6** | **5.82** |

CUT3R 同时更快更准，作者归因于 absolute scale 优于 VGGT 系的 first-frame relative scale，特别是大视角变化时。

### 4.4 Task Horizon

**Table IV. 按执行步数分桶的 R2R val-unseen 表现。**

| Horizon | Model | #Ep | SR ↑ | SPL ↑ | OSR ↑ | NE ↓ |
|---|---|---|---|---|---|---|
| Short (1–50) | Baseline | 459 | 51.20 | 48.18 | 55.34 | 5.08 |
| | Ours | 486 | 51.23 | 48.84 | 54.53 | 4.86 |
| Medium (50–100) | Baseline | 1038 | 49.61 | 43.79 | 61.27 | 5.64 |
| | Ours | 1061 | **54.29** | **48.04** | **63.71** | 5.46 |
| Long (≥100) | Baseline | 342 | 20.18 | 10.61 | 34.21 | 9.11 |
| | Ours | 292 | **24.32** | **14.25** | **40.75** | 8.74 |
| Overall | Baseline | 1839 | 44.54 | 38.72 | 54.76 | 6.15 |
| | Ours | 1839 | **48.72** | **42.88** | **57.64** | **5.82** |

短任务追平、medium / long 上 SR +4 个点——印证"长程预测先验有用"。

> ❓ Long horizon 桶下 baseline #Ep=342、Ours #Ep=292，因为 long 桶按各模型自己执行步数划分的——Ours 更高效，更多 episode 落到 short/medium 桶里。这种分桶口径会让 "long 桶 +4 SR" 的解读多少有 selection bias 风险——更稳的做法是按 ground-truth path length 分桶。

### 4.5 Mask Ablation

**Table V. R2R val-unseen 上不同 mask 设计的对比。**

| Mask Design | NE ↓ | OSR ↑ | SR ↑ | SPL ↑ |
|---|---|---|---|---|
| Leaky | 6.81 | 51.3 | 40.2 | 35.7 |
| w/o Isolation | 6.98 | 51.1 | 39.9 | 35.3 |
| Ours | **5.82** | **57.6** | **48.7** | **42.9** |

去掉跨轮 query isolation 或允许 query 看未来 navigation token，SR 直接掉 ~9 个点。这意味着两件事：
- Mask 设计是这套 latent prediction loss 能 work 的关键，不是细节。
- 训练 / 测试 mismatch 的代价非常显著——leaky mask 下 query "看了未来"，训练阶段 representation 就被推去走 cheating shortcut，推理移除 query 后 representation 反而退化。

### 4.6 Real-Robot

**Figure 4. ARX-Lift2 在不同室内 / 室外光照下的第一人称视图。**

![](https://arxiv.org/html/2603.03739v1/x4.png)

**Table VI. 真机不同场景 / 光照下的成功率（completed/total）。**

| Scene | Lighting | NaVid | StreamVLN | Ours |
|---|---|---|---|---|
| Office (Indoor) | Bright | 7/30 | 12/30 | **20/30** |
| Warehouse (Indoor) | Bright | 6/30 | 12/30 | **18/30** |
| Corridor (Indoor) | Moderate | 11/30 | 16/30 | **22/30** |
| Afternoon (Outdoor) | Bright | 6/30 | 10/30 | **18/30** |
| Dusk (Outdoor) | Moderate | 4/30 | 6/30 | **11/30** |
| Night Street (Outdoor) | Low | 2/30 | 6/30 | **9/30** |

成功定义：500 步内到目标 0.3 m 内并 STOP，碰撞算失败。所有场景未在训练中见过。每场景 30 trials（3 horizon × 5 instructions × 2 repeats）。Lighting 越差所有方法都退化，但 PROSPECT 始终有相对优势——latent-space 监督避免 overfit appearance 的论点至少在真机上得到一些支持。

---

## 关联工作

### 基于
- [[2507-StreamVLN|StreamVLN]]：直接 baseline，PROSPECT 沿用其 short window + long-term memory 的 fast-slow context 框架。
- CUT3R (Wang et al. 2025)：streaming 3D foundation model，PROSPECT 把它当 frozen 3D encoder + frozen teacher。
- [[2506-VJEPA2|V-JEPA2]] / I-JEPA：latent-space prediction 思想的源头。

### 对比
- [[2402-NaVid|NaVid]]、Uni-NaVid、[[2412-NaVILA|NaVILA]]、[[2507-StreamVLN|StreamVLN]]：VLN-CE single-view RGB 同档对比。
- NavMorph (ICCV25)：VLN 里的 self-evolving world model，但用低维 state-space。
- NavForesee：concurrent，pixel/depth 监督的 unified VLN world model。
- JanusVLN：concurrent，VGGT-based dual-memory VLN encoder。
- VGGT / InfiniteVGGT：spatial encoder 对照组。

### 方法相关
- [[2602-WorldVLALoop|WorldVLA]]、DreamVLA、Mantis：unified VLA + world model 的相关 effort，但多在 manipulation / 短上下文，不在 streaming VLN。
- ScanQA / LLaVA-Video-178K：训练阶段 spatial / video QA 数据。
- DAgger (Ross et al. 2011) / ScaleVLN：Stage 2 数据扩充来源。

---

## 论文点评

### Strengths

1. **方法干净**：unified streaming model + 推理时砍掉预测分支，是把 world model / JEPA-style 信号塞进 VLA 的一个 minimally invasive 的方式。在线推理零开销 + 训练阶段表征 shaping，工程上对部署友好。
2. **CUT3R vs VGGT 的对比有信息量**：absolute-scale 对长 episode 的 streaming VLN 是结构性优势，不只是单点 +X SR。这是个 generalizable 的发现，值得记下来。
3. **Loss 选择的 grounding**：cosine for SigLIP / MSE for CUT3R 这个细节解释了为什么——结合 teacher 自身训练几何来选 distillation loss，是个 transferable 的 take。
4. **Mask ablation 很硬**：去掉关键 mask 性质 SR 直接 -9，说明这套架构的 inductive bias 不是 cosmetic。

### Weaknesses

1. **没做 latent vs pixel 直接对比**：核心 claim 是 "latent target 优于 pixel/depth target，能避免 overfit appearance"，但全文没有一个 controlled ablation 把 target 换成 pixel/depth 跑同模型。只能从总分和真机鲁棒性间接推断。
2. **真机统计量小**：每场景 30 trials，6 场景共 180。与 baselines 的对比方差很可能不小，但没有 confidence interval。
3. **Long horizon 分桶口径**：按各模型自己执行步数分桶，会让 "long 桶 SR 更好" 的解读包含 selection bias。更公平是按 GT path length 分桶。
4. **Cross-turn query 隔离的代价**：架构选择把 t 时刻预测对 t+1 时刻 query 完全 mask 掉，丢失了潜在的 next-step prediction → next-step planning 的迭代信号；缺一个 ablation。
5. **Code/checkpoint 都没放**："release soon" 是 VLN 圈的常见承诺；目前可复现性=0。

### 可信评估

#### Artifact 可获取性

- **代码**: 未开源（abstract 写 "We will release code for the community soon"，无 repo 链接）
- **模型权重**: 未发布
- **训练细节**: 仅高层描述 + 关键超参（lr、warmup、loss 系数 γ/α/β、mask token 数、query token 数、训练 GPU-hr），但具体 batch size、训练步数、数据加载顺序等未披露
- **数据集**: 全部公开来源（R2R / RxR / R2R-EnvDrop / ScaleVLN / LLaVA-Video-178K / ScanQA），DAgger 重标数据未说明是否会一并放出

#### Claim 可验证性

- ✅ **VLN-CE R2R / RxR 上 first-tier**：Table I 有完整数值，与 StreamVLN / NaVILA 的对比口径标注清楚（数据 regime *、† 区分）。
- ✅ **CUT3R 比 VGGT 系更适合 streaming long-episode**：Table III 直接对比时间和 SR，VGGT OOM 是结构性问题。
- ⚠️ **"Latent prediction 优于 pixel/depth prediction，避免 overfit appearance"**：核心论点之一，但全文无直接 controlled ablation；只能间接通过真机鲁棒性推断。
- ⚠️ **"对 long-horizon 任务收益更大"**：Table IV 数字支持，但 horizon 分桶按各模型自己步数划分，存在 selection bias。
- ⚠️ **真机优于 NaVid / StreamVLN**：单一硬件 + 单作者团队部署，缺少 inter-rater / multi-seed，180 trials 总量偏小。
- ⚠️ **"Inference-free 的预测分支"**：训练 cost 增加多少（额外 query token 数 × 2 layer decoder × N 轮）没有量化报告。

### Notes

- **Generalizable lesson**：做 JEPA-style 的 latent distillation 时，loss 的几何要跟 teacher 自身训练 loss 的几何同构（这里 SigLIP 是 normalized embedding + sigmoid pairwise，所以 cosine；CUT3R 是几何 regression，所以 MSE）。这点在 [[2506-VJEPA2|V-JEPA2]]、Spatial Forcing 等工作里都隐含但很少明写。
- **Architecture pattern**：训练用辅助 token + 推理移除 = 0 latency cost 的 representation shaping。这个 trick 在 multi-task / world-model loss 加进 VLA 时是个值得复用的模板，比如 spatial forecasting / affordance prediction 都可以这么塞。
- **Open question**：如果把 cross-turn query isolation 放松，让 t+1 query 看到 t 时刻的预测 embedding，会不会拿到 next-step planning 的 chained 收益？现在被 hard mask 掉了，未在表里覆盖。
- **可复用 finding**：Long-streaming VLN 上 absolute-scale 的 spatial encoder（CUT3R）显著优于 first-frame relative scale 的（VGGT 系）。如果以后做长上下文的 spatial reasoning / SLAM-like memory，应优先选 absolute-scale 的 backbone。
- **Concurrent works to track**：NavForesee、JanusVLN——同一时间段、相邻方向，是检验 PROSPECT 设计选择是否独到的天然对照组。

### Rating

**分数**：2 - Frontier
**理由**：PROSPECT 是 streaming VLN + latent-space world-model signal 方向的前沿代表工作：方法干净（unified streaming + 推理时砍掉预测分支），VLN-CE R2R/RxR val-unseen 取得 first-tier 成绩且真机鲁棒性优于 NaVid / StreamVLN（见 Strengths 1/2 与 Table I/VI）。但 distinguishing from Foundation：核心 "latent > pixel target" claim 缺直接 controlled ablation（Weaknesses 1），code/checkpoint 未开源（Weaknesses 5，Artifact 可获取性），且方向内有 NavForesee / JanusVLN 等 concurrent 工作，尚未被社区公认为奠基工作。Distinguishing from Archived：方法在 streaming VLN 上刷新 first-tier，mask ablation 显示设计 non-cosmetic，CUT3R vs VGGT 的比较提供了 generalizable 的 spatial encoder 选型 insight。
