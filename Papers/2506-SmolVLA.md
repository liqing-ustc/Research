---
title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"
authors: [Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, Remi Cadene]
institutes: [Hugging Face, Sorbonne University, valeo.ai, ENS Paris-Saclay]
date_publish: 2025-06-02
venue: arXiv
tags: [VLA, flow-matching, manipulation]
paper: https://arxiv.org/abs/2506.01844
website: https://huggingface.co/blog/smolvla
github: https://github.com/huggingface/lerobot
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics
> - **核心**: 一个 450M 参数的 VLA，仅在 ~23K 条 community-collected episodes 上预训练，性能可与 ~10× 更大的 VLA（如 π0-3.3B）持平甚至更好；同时提出 async inference stack 把 real-world 控制提速 ~30%。
> - **方法**: SmolVLM-2 backbone（冻结）+ flow-matching action expert；架构上 (i) 跳过 VLM 后半层，(ii) 限制每帧 64 个 visual tokens，(iii) 在 expert 内交错 cross-attention 与 causal self-attention 层；推理上把 chunk prediction 与 action execution 解耦到 RobotClient/PolicyServer。
> - **结果**: LIBERO 平均 87.3%（>π0-3.3B 的 86.0），Meta-World 57.3%，real-world SO100 三任务平均 78.3%（π0 为 61.7）；async vs sync 任务时长 9.7s vs 13.75s，固定时间内完成 cube 数 3.8 vs 1.8。
> - **Sources**: [paper](https://arxiv.org/abs/2506.01844) | [website](https://huggingface.co/blog/smolvla) | [github](https://github.com/huggingface/lerobot)
> - **Rating**: 2 - Frontier（全开源 recipe + async inference，sub-1B VLA 路线的必比 baseline，但方法是对 π0 的工程化而非奠基）

**Key Takeaways:**
1. **小模型 + 社区数据足以对标 SOTA**: 450M 参数 + <30K episodes（社区贡献）即可在 LIBERO/Meta-World 超过 π0-3.3B-Paligemma（无机器人预训练版本），且与机器人预训练的 π0 持平，挑战了"VLA 必须靠学术/工业大规模数据"的主流叙事。
2. **Async inference 是 model-agnostic 的免费午餐**: 把 chunk 预测与执行解耦后，吞吐 2×，成功率持平。这一 trick 可被任何 chunk-output 策略复用。
3. **架构 ablation 给出了可复用的 design rule**:（a）skip VLM 后半层比训小 VLM 更好；（b）interleaved CA+SA 比纯 CA 或纯 SA 都好（85.5 vs 79.0/74.5）；（c）action token 内部用 causal SA 比 bidirectional 好；（d）states 应送入 VLM（prefix）而非 expert（suffix）；（e）flow matching > L1 regression。
4. **完全开源**: 代码、权重（`lerobot/smolvla_base`）、数据列表、SO-100/101 硬件方案全开放，是目前 reproducibility 最好的 VLA stack 之一。

**Teaser. SmolVLA 总览：VLM backbone 输出特征 → flow-matching action expert 通过交错 CA/SA 生成动作 chunk。**

![](https://arxiv.org/html/2506.01844v1/x16.png)

---

## 1 Motivation

VLA 领域被几个特征绑住：(i) 模型动辄数 B 参数；(ii) 数据依赖学术/工业级 teleoperation；(iii) 主流开源模型（OpenVLA、RT-2-X）的"开"集中在权重，训练 recipe / 数据并不全开。作者主张推动一条更"经济"的路线：~10⁸ 量级参数、社区 contributed 数据、消费级 GPU 训练 / CPU 推理。

> ❓ "10× 更小却可比"的 framing 在 VLA 领域已被多个 sub-1B 模型用过（TinyVLA、π0 distillations 等）。SmolVLA 的真正差异在于 community data 这条供给侧路线，而不在尺寸本身。

## 2 Method

### 2.1 整体架构

两个组件：
- **VLM**: SmolVLM-2（vision encoder = SigLIP，language decoder = SmolLM2）。冻结。
- **Action expert** $\mathbf{v}_\theta$: 一个 transformer，用 conditional flow matching 训练，hidden size = $0.75 \times d_\text{VLM}$。

输入：多视角 RGB 图像 + 语言指令 + 机器人 sensorimotor state（用 linear projection 投到 token dim 作为单 token）。所有 tokens 拼接后过 VLM，VLM 第 $N$ 层（$N=L/2$）的特征喂给 action expert，输出长度 $n=50$ 的 action chunk。

### 2.2 三个效率设计

1. **Visual token reduction**: 不用 image tiling，只用 global image + pixel shuffle，每帧 64 个 visual tokens。
2. **Layer skipping**: action expert 用 VLM 第 $N=L/2$ 层（不是最后一层）的特征。Ablation 显示这比 training 一个 ~256M 的更小 VLM 更好（Table 8）。
3. **Interleaved CA + SA in expert**: 每个 expert block 是 *要么* CA（cross-attend VLM keys/values）*要么* causal SA（action tokens 互相 attend），而非两者并存。

### 2.3 Flow Matching Training Objective

**Equation 1. Flow matching loss**

$$
\mathcal{L}^{\tau}(\theta)=\mathbb{E}_{p(\mathbf{A}_{t}\mid \mathbf{o}_{t}),\,q(\mathbf{A}_{t}^{\tau}\mid \mathbf{A}_{t})}\!\left[\left\|\mathbf{v}_{\theta}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t})-\mathbf{u}(\mathbf{A}_{t}^{\tau}\mid \mathbf{A}_{t})\right\|^{2}\right]
$$

**符号说明**: $\mathbf{o}_t$ 是 VLM 第 $N$ 层特征；$\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1-\tau)\epsilon$, $\epsilon \sim \mathcal{N}(0,\mathbf{I})$；目标向量场 $\mathbf{u}(\mathbf{A}_t^\tau \mid \mathbf{A}_t) = \epsilon - \mathbf{A}_t$；$\tau \sim \text{Beta}$ (沿用 π0)。
**含义**: 训 expert 预测从 noisy action 到 clean action 的速度场，推理时 10 步 ODE 解算出 action chunk。

### 2.4 Community Pretraining Data

筛选 481 个 HF 上 community-contributed 数据集，得到：

| # datasets | # episodes | # frames |
| ---------- | ---------- | -------- |
| 481        | 22.9K      | 10.6M    |

**Table 1. Pretraining 数据规模。** 比 OpenVLA 的 ~1M trajectories 小一个数量级以上。

两个数据清洗动作：
- **Task annotation with VLM**: 原始 task 描述噪声很大（"task desc"、"Hold"、"Up" 之类）。用 Qwen2.5-VL-3B-Instruct 看 sampled frames + 原指令，生成简洁的 action-oriented 描述。
- **Camera viewpoint normalization**: 手工把每个 camera key 映到 `OBS_IMAGE_1/2/3`（top → wrist → side 优先序），多余视角丢弃。作者说这一步对训练稳定性"显著"必要。

> ❓ 手工标注 481 个数据集的 camera role 是规模化的瓶颈。论文承认未来要靠 VLM 自动化，但目前的 SmolVLA recipe 隐含了 ~人/天 量级的 curation 成本，这一项很容易被 reproducibility narrative 模糊掉。

### 2.5 Asynchronous Inference

把 chunk prediction 从 control loop 解耦：RobotClient 持续消费 action queue，当剩余动作占比 $|\mathbf{A}_t|/n < g$ 时，捕获新 observation 发给（可能是远程 GPU 上的）PolicyServer 触发新 chunk 预测，得到结果后与残余 queue 在 overlap 上 aggregate。同时用 joint-space 距离过滤近重复 observation 避免冗余 inference。

**Figure 2. Async inference 总体架构：RobotClient 与 PolicyServer 通过网络解耦。**

![](https://arxiv.org/html/2506.01844v1/x17.png)

队列阈值 $g$ 的定性分析（Algorithm 1 中）：
- $g=0$（sequential）: 队列耗尽才发请求 → 平均 $\mathbb{E}[\ell_S]$ 秒空闲。
- $g=0.7$（async）: 消耗约 $1-g=0.3$ 后触发预测，分摊算力同时保住队列。
- $g=1$（compute-intensive）: 每个 control tick 都触发 forward，最反应灵敏但成本最高。

避免空闲条件：$g \ge \frac{\mathbb{E}[\ell_S]/\Delta t}{n}$，其中 $\Delta t$ 是 control cycle（30 fps → 33ms）。

**Figure 3. 不同 $g$ 与是否启用 joint-space 相似度过滤下的 action queue 动态。** (B) 中红箭头标出了"队列空时强制处理近重复 observation"的 bypass 时刻。

![](https://arxiv.org/html/2506.01844v1/x18.png)

## 3 Experiments

### 3.1 Setup

- **Sim**: LIBERO（4 类 × 10 task = 40 task，1693 episodes）、Meta-World（50 task，2500 episodes）。
- **Real**: SO-100 上 Pick-Place / Stacking / Sorting 三任务（各 50 demos），SO-101 上 Pick-Place-Lego（注意 SO-101 完全未在预训练数据中）。
- **训练**: 200K steps, batch 256, AdamW, cosine LR 1e-4 → 2.5e-6, image 512², bf16 + torch.compile。pretraining 用 4 GPU，整个项目 ~30K GPU hours。Action expert ~100M / 总 450M。仅 train action expert，VLM 冻结。
- **Baselines**: [[2410-Pi0|π0]] (3.3B, Paligemma backbone, finetuned per task)，ACT (~80M, single-task only)。

**Figure 4. Real-world tasks 全景：SO100 三任务 + SO101 lego task 的 starting/terminal frame。**

![](https://arxiv.org/html/2506.01844v1/x19.png)

### 3.2 Main Results

**Table 2. Simulation (LIBERO + Meta-World) success rate (%).**

| Benchmark   | Policy            | VLA Pt | Avg     |
| ----------- | ----------------- | ------ | ------- |
| LIBERO      | Diffusion Policy  | No     | 72.4    |
| LIBERO      | Octo (0.09B)      | Yes    | 75.1    |
| LIBERO      | OpenVLA (7B)      | Yes    | 76.5    |
| LIBERO      | π0 (Paligemma-3B) | No     | 71.8    |
| LIBERO      | π0 (3.3B)         | Yes    | 86.0    |
| LIBERO      | SmolVLA (0.24B)   | No     | 82.75   |
| LIBERO      | **SmolVLA (0.45B)** | **No** | **87.3** |
| LIBERO      | SmolVLA (2.25B)   | No     | 88.75   |
| Meta-World  | Diffusion Policy  | No     | 10.5    |
| Meta-World  | TinyVLA           | No     | 31.6    |
| Meta-World  | π0 (3.5B-PG)      | No     | 50.5    |
| Meta-World  | π0 (3.5B)         | Yes    | 47.9    |
| Meta-World  | SmolVLA (0.45B)   | No     | 57.3    |
| Meta-World  | SmolVLA (2.25B)   | No     | 68.24   |

**核心**: SmolVLA-0.45B 在 LIBERO 上略胜 π0-3.3B（87.3 vs 86.0），且自身**未做任何机器人预训练**。论文同时给出 SmolVLA 比 π0 训练快 ~40%、显存少 6×。

> ❓ "SmolVLA is only initialized from the VLM"——但表中 SmolVLA 行实际上是用 community data 预训练后 finetune 在 LIBERO 上的（第 3.2 节多次提到 community pretraining 是核心贡献）。"无 VLA pretraining"的标签可能指**没用机器人专项数据预训练**，而非完全跳过 SmolVLA pretraining 阶段。这点表格 caption 写得有歧义。

**Table 3. Real-world SO100 三任务 multi-task SR (%).**

| Policy               | Pick-Place | Stacking | Sorting | Avg      |
| -------------------- | ---------- | -------- | ------- | -------- |
| ACT (single-task)    | 70         | 50       | 25      | 48.3     |
| π0 (3.5B, multi)     | 100        | 40       | 45      | 61.7     |
| **SmolVLA (0.45B)**  | 75         | **90**   | **70**  | **78.3** |

SmolVLA 平均胜 π0 ~17 个点，但 Pick-Place 单项被 π0 拉开 25 点——这暗示 SmolVLA 在更长 horizon / 更需要精细的 stacking & sorting 上反而更稳。值得追问：是 ACT/π0 的 Pick-Place 接近上限（成功率天花板效应）还是 SmolVLA 在简单任务上有 underfitting？

**Table 4. SO101 Pick-Place-Lego（embodiment 跨域）.**

| Policy             | In-Dist | OOD |
| ------------------ | ------- | --- |
| ACT (single)       | 70      | 40  |
| SmolVLA (single)   | 90      | 50  |

SmolVLA 在没见过 SO101 的情况下仍然 generalize，是 cross-embodiment 的好信号但样本量极小。

### 3.3 Pretraining + Multitask 消融

**Table 5. SmolVLA 自身的两个 axis（pretraining / multitask）.**

| Variant                             | Pick-Place | Stacking | Sorting | Avg      |
| ----------------------------------- | ---------- | -------- | ------- | -------- |
| Single-task, no pt                  | 55         | 45       | 20      | 40       |
| Multi-task, no pt                   | 80         | 40       | 35      | 51.7     |
| **Multi-task, with community pt**   | 75         | 90       | 70      | **78.3** |

Pretraining 给了 26.6 个点（最大那一刀），multi-task finetune 又额外加 ~12 点。这是论文里**最关键**的 evidence：community-collected data 的预训练价值是真实且可量化的。

### 3.4 Async vs Sync Inference

**Table 6. 同样的 SmolVLA-0.45B，sync vs async.**

| Mode | Pick-Place | Stacking | Sorting | Avg  | Total time (s) | Avg time (s) | Cubes/60s |
| ---- | ---------- | -------- | ------- | ---- | -------------- | ------------ | --------- |
| Sync | 75         | 90       | 70      | 78.3 | 137.5          | 13.75        | 1.8       |
| Async| 80         | 90       | 50      | 73.3 | 97.0           | 9.7          | 3.8       |

成功率 ~持平（async 在 Sorting 掉 20 点，作者归因为 hyperparams 是在 Pick-Place 上调的、被复用），但任务时长缩短 30%、固定时间内吞吐 2.1×。

> ❓ Sorting 的成功率从 70 → 50 是非平凡的回退。论文用"hyperparams 复用"轻描淡写，但这恰恰说明 async 的实用性强烈依赖 per-task tuning（特别是 $g$ 和相似度阈值 $\epsilon$），而非 turnkey 提速。

### 3.5 Architecture Ablations（LIBERO）

| Choice                | Variant                       | LIBERO Avg |
| --------------------- | ----------------------------- | ---------- |
| Attention             | CA only                       | 79.0       |
| Attention             | SA only                       | 74.5       |
| **Attention**         | **Interleaved CA+SA**         | **85.5**   |
| SA mask               | Bidirectional                 | 67.5       |
| SA mask               | Causal                        | 74.5       |
| VLM layers used (N)   | 8 / 16 / 24 / 32 / Skip%2 / 256M | 75 / 78.5 / 79.5 / 80.3 / 75.5 / 75.8 |
| Expert width          | ×1.00 / 0.75 / 0.50 / 0.25    | 82.3 / 77.5 / 80.3 / 73.8 |
| Objective             | Flow matching / L1 regression | 80.25 / 75.25 |
| State placement       | Prefix(VLM)+CA / Suffix(expert)+CA | 80.3 / 73.3 |
| Chunk size $n$        | 1 / 10 / 30 / 50 / 100        | 50.0 / 84.0 / 78.5 / 80.3 / 74.5 |
| Action steps before re-obs | 1 / 10 / 30 / 50         | 80.3 / 82.8 / 70.8 / 51.8 |

**关键 take-aways**:
- Interleaved CA+SA 比 single-flavor 高 ~6-11 点，**这是论文最干净的架构 insight**。
- 跳层后用 VLM 中段（N=16-32）比用全部层只略低；用 256M 小 VLM 反而更差 → "skip 一个大模型 > 训一个小模型"。
- Chunk size 10 在 LIBERO 上最佳，但实验主体用 $n=50$（real-world 长 horizon 友好）——LIBERO 上的 $n=50$ 并非最优，但被选用以求一致。
- 每个 chunk 执行 1-10 步就刷新 observation 显著好过执行 30+ 步——证实 async / 高频 re-observation 是 real-world 必要的。

## 4 Limitations（作者列出的 + 我自己的）

作者承认：
- 仅在单一机型 (SO100) 上预训练，cross-embodiment 仅靠 SO101 单任务做了 demonstration
- 数据 ~23K 条 << OpenVLA 的 1M
- VLM backbone 主要在 OCR / 文档数据上预训练，不一定适合机器人
- 短 horizon 任务，长 horizon 未触及
- 仅 imitation，未尝试 RL

我会补充：
- "community data" 的可扩展性论证不足。整个 pipeline 隐含了大量人工 curation（camera role 标注、quality filter）。
- Real-world benchmark 全是 SO100/101 上的桌面 cube manipulation，覆盖 task 多样性有限。
- 与 π0 的对比中，π0 是 finetune 而 SmolVLA 也是 finetune（pretraining 数据不同），但训练 step / data 量没有完全对齐，所以"40% 更快、6× 更省显存"的 claim 是端到端的而非 controlled。

---
## 关联工作

### 基于
- [[2410-Pi0|π0]]: 直接的架构祖先（VLM + flow-matching action expert，β-distribution τ sampling）。SmolVLA 把 backbone 缩小、加 layer skipping、把 action expert 内部改成交错 CA/SA。
- SmolVLM-2: 提供 backbone（SigLIP + SmolLM2），是 SmolVLA 能"小"的前提。
- LeRobot: 实现框架，也提供 baselines ([[2410-Pi0|π0]], ACT) 和 SO-100/101 硬件接口。

### 对比
- [[2406-OpenVLA|OpenVLA]] (7B): 主要 baseline 之一。SmolVLA 在 LIBERO 上 (87.3 vs 76.5) 用 1/15 的参数胜出。
- ACT (Aloha): single-task baseline，real-world 上被 SmolVLA 平均高 30 点。
- TinyVLA: 同样的"sub-1B VLA"路线但没用大规模机器人预训练，被作者用作"为什么需要 community pretraining"的反例。

### 方法相关
- [[2504-Pi05|π0.5]] / [[2604-Pi07|π0.7]] / [[2502-OpenVLA-OFT|OpenVLA-OFT]]: 同代 / 后续 VLA。
- DexVLA: 同样把 diffusion-based action decoder 嫁接到 VLM 上。
- [[2307-RT2|RT-2]] / RT-2-X: 把 VLM 微调到 robotics 数据的早期范式，被 SmolVLA 作为 large-VLA 路线的对照。
- Action chunking & async control: ACT / Diffusion Policy 等 chunk-output 策略都可以 plug-in SmolVLA 的 async stack。

---
## 论文点评

### Strengths

1. **Recipe 透明度**: 是少有的把架构、训练、数据、硬件全开源的 VLA 工作。`lerobot/smolvla_base` checkpoint + lerobot codebase + SO-100/101 硬件方案 = 真正"reproducible-by-an-individual-researcher"。
2. **Async inference 有独立价值**: 这是一个 model-agnostic 的工程贡献，可以被任何 chunk-output policy 复用。论文给了清晰的分析（队列阈值 $g$、相似度过滤）和真机数据。
3. **Ablation 密度高**: 8 个独立 axis 的 ablation（attention 模式、SA mask、VLM 层、expert 宽度、目标、state 位置、chunk size、re-observation 频率），都直接给到 LIBERO 数字。Interleaved CA+SA 与"skip layer > smaller model"是有可复用性的 design rule。
4. **Community data 可行性证明**: Table 5 的 51.7 → 78.3 是干净的 ablation，证明了 ~23K 条 community-collected episodes 这一量级足以解锁 real-world 多任务能力。这对 robotics 数据范式是真实的 update。

### Weaknesses

1. **"小但不输"的数字游戏带潜台词**: SmolVLA-0.45B 胜 π0-3.3B 的对比里，π0 是直接 finetune 在目标任务，而 SmolVLA 经过 community pretraining。两边 pretraining 数据 / step / strategy 都不同，"6× 省显存 / 40% 更快"的对比是端到端的，不能严格归因于架构。
2. **Real-world benchmark 集中在 SO100/101 桌面 cube tasks**: 4 个任务全是 pick / stack / sort cube/lego，task diversity 弱。OOD 测试也仅是 lego 的位置变化，远不是真正的 distribution shift。
3. **Async 的 sorting 回退被淡化**: 70 → 50 用"hyperparams 复用"解释。这说明 async 的实用性依赖 per-task tuning，是真正的 caveat 而非 minor detail。
4. **Cross-embodiment 论据薄弱**: 整个 pretraining 集只来自 SO100，SO101 的 single-task 测试样本量很小（10 demos × 5 starting positions）。"generalize to new embodiments" 仅是 anecdote。
5. **VLM 选择缺 ablation**: 只用 SmolVLM-2，没有对比换其他 same-scale VLM（如 PaliGemma-mini、MoonDream）的影响。架构其他组件都做了 ablation 唯独 backbone 没动，留下盲点。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training（lerobot 仓库内 SmolVLA policy 实现 + 训练脚本均开源）
- **模型权重**: `lerobot/smolvla_base`（450M, HuggingFace），加上 `fracapuano/smolvla_async` 等 async 演示。
- **训练细节**: 完整披露——200K steps、batch 256、AdamW betas (0.9, 0.95)、cosine LR 1e-4→2.5e-6、bf16+torch.compile、4 GPU、~30K GPU hours、$n=50$ chunk、10-step flow matching ODE。
- **数据集**: 开源（Pretraining: 481 个 HF community datasets，Appendix A.1 列表；Eval: 4 个 SO100/101 数据集 open-sourced on HF）。

#### Claim 可验证性
- ✅ **0.45B 模型在 LIBERO 平均 87.3% 胜 π0-3.3B 的 86.0%**: Table 2 直接数据，且 lerobot 提供 reproducible pipeline，可独立复现。
- ✅ **Async 任务时长平均 9.7s vs sync 13.75s（30% 更快）**: Figure 5 给出 std，硬件已开源（SO-100），可由第三方复现。
- ✅ **Community pretraining 把 multi-task SR 从 51.7 拉到 78.3**: Table 5 干净 ablation，是论文最强 evidence。
- ⚠️ **"6× less memory, 40% faster than π0 to train"**: 端到端比较，未控制 batch size / sequence length / hardware 一致性，是 implementation-level 的对比而非架构归因。
- ⚠️ **"Comparable to VLAs that are 10× larger"**: Marketing-friendly 说法。LIBERO/Meta-World 上确实成立，但 real-world Pick-Place 单项 SmolVLA (75) 远低于 π0 (100)，"全面持平"经不起细看。
- ⚠️ **Cross-embodiment generalization (SO101 OOD 50%)**: 单任务、小样本（5 OOD positions × 1-2 trials each），把"50%"当作"generalize" 太乐观。

### Notes

- **对我有什么用**: SmolVLA 的 async inference stack 是 model-agnostic 的，可以直接套到我们自己的 VLA / world-model 实验上做 real-time control benchmark。Layer skipping + interleaved CA+SA 这两个 design rule 也值得在自己的 expert design 里 try。
- **Open question 1**: Community data 里"质量"和"任务覆盖度"哪个是 78.3 这个数字的真正 driver？目前 Table 5 没法区分。需要 leave-one-dataset-out 或者按 quality bucket 切片的实验。
- **Open question 2**: VLM backbone 到底重要吗？如果换成同 scale 的 PaliGemma-mini 或 MoonDream，结果会差多少？这关系到"小模型 VLA"路线的 sensitivity 边界。
- **Open question 3**: Chunk size 在 LIBERO 上 $n=10$ 最优但论文用 $n=50$，因 real-world 偏好长 chunk。这其实暗示 sim 和 real 的最优 chunk size 是不一样的——这是个值得单独研究的 trade-off。
- **写作小问题**: Table 1 caption 说 "10M episodes" 但表格内是 22.9K episodes / 10.6M frames。明显笔误。

### Rating

**分数**：2 - Frontier
**理由**：按 field-centric rubric，SmolVLA 是当前"小型 VLA + 社区数据"路线最具代表性的开源工作，凭 Strengths 里的"完全 reproducible recipe + 密集 ablation + async inference"成为 sub-1B VLA 的必比 baseline；但 Weaknesses 指出其 real-world benchmark 局限在 SO100 桌面 cube、cross-embodiment 与 community-data scaling 论据都偏单薄，方法本身（flow-matching expert + skip layer + interleaved CA/SA）属于对 π0 的聪明工程化而非奠基性突破，故未到 3 - Foundation；相较 1 - Archived，它在发布不足一年内已被 HF 社区作为 de facto 起点且 lerobot codebase 正在被后续 VLA 工作复用，仍处前沿。
