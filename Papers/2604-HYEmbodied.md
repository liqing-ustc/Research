---
title: "HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents"
authors: [Tencent Robotics X, HY Vision Team]
institutes: [Tencent]
date_publish: 2026-04
venue: arXiv
tags: [VLM, spatial-reasoning, VLA]
paper: https://arxiv.org/abs/2604.07430
website: 
github: https://github.com/Tencent-Hunyuan/HY-Embodied
rating: 2
date_added: 2026-04-16
---
## Summary

> [!summary] HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents
> - **核心**: 面向 embodied agent 的 VLM 基座，用 MoT 架构 + 大规模 embodied 数据 + 迭代 RL/RFT 后训练打造 2B/32B 两档模型
> - **方法**: Mixture-of-Transformers (MoT) 架构实现模态自适应计算 + visual latent tokens + 100M+ embodied 数据预训练 + GRPO-based RL + on-policy distillation
> - **结果**: MoT-2B 在 22 个 embodied benchmark 中 16 个 SOTA；32B 超越 Gemini 3.0 Pro；VLA 下游 real-world 操作超越 [[2410-Pi0|π0]]/[[2504-Pi05|π0.5]]
> - **Sources**: [paper](https://arxiv.org/abs/2604.07430) | [github](https://github.com/Tencent-Hunyuan/HY-Embodied)
> - **Rating**: 2 - Frontier（embodied VLM 前沿 SOTA + 开源 2B，方法范式（MoT + 迭代 RL/RFT + OPD）是当下必须比较的 baseline，但 32B/数据闭源、尚未沉淀为 de facto 标准）

**Key Takeaways:**
1. **MoT 架构是小模型增强视觉能力的高效方案**: 通过复制 FFN/QKV 参数为视觉分支专用，4B total 参数但仅 2.2B activated，推理开销接近 Dense-2B 但视觉建模能力显著提升
2. **Embodied 数据的系统化构建**: 100M+ embodied/spatial 数据覆盖 perception → understanding → planning 三层，200B+ tokens 预训练建立物理世界理解
3. **迭代式 RL + RFT 自演化后训练**: RL 扩展能力边界 + RFT 巩固高质量推理 trace，交替执行逐步提升 deep thinking 能力；on-policy distillation 从 32B 迁移到 2B

**Teaser. HY-Embodied-0.5 在 spatial/embodied benchmark 和下游机器人控制任务上的表现概览**
![](https://arxiv.org/html/2604.07430v1/x1.png)

---
## Introduction

当前 VLM 要成为 physical agent 的 brain 需要在两个维度增强：(1) **Fine-grained visual perception** -- 精确的视觉感知是理解物理世界和执行动作的前提；(2) **Embodied prediction, interaction, and planning** -- 主流 VLM 主要在静态 web 数据上训练，缺乏面向 embodied 场景的动态预测、交互和规划能力。

HY-Embodied-0.5 家族包含两个变体：
- **MoT-2B**（2B activated / 4B total）: 面向边缘部署的高效模型
- **MoE-A32B**（32B activated / 407B total）: 面向复杂推理的强力模型

---
## Model Architecture

**Figure 2. HY-Embodied-0.5 MoT 架构总览**
![](https://arxiv.org/html/2604.07430v1/x2.png)

架构基于标准 VLM paradigm（ViT + LLM），针对视觉感知引入三个关键改进：

### HY-ViT 2.0: Efficient Native-Resolution Visual Encoder

- 400M 参数的原生分辨率 ViT，支持任意分辨率输入
- 从更大的内部 ViT 蒸馏而来，保证边缘设备上的效率和精度
- 额外训练一个大版本 ViT 生成 discrete visual representations（codebook size 2k，每 8x8 patch 压缩为一个 discrete code），用于监督模型视觉 token 输出

### Modality-Adaptive Computing with Mixture-of-Transformers

核心思想：为视觉和语言 token 引入独立的参数，避免大量视觉训练导致语言能力退化。

具体做法：
- 复制 LLM 的 FFN 和 QKV 参数，用预训练 LLM 权重初始化
- 视觉 token 用复制的参数计算，文本 token 用原参数计算
- 视觉 token 使用**双向 attention**（区别于文本的 causal attention）
- 引入 **visual next-code prediction** 任务：用 MLP 预测下一个 patch 的 discrete code，为视觉分支提供更强的监督信号

**Figure 3. MoT attention 计算方式**
![](https://arxiv.org/html/2604.07430v1/x3.png)

对小模型特别有效：总参数翻倍但推理效率几乎不变（decode 阶段主导推理时间，视觉分支仅在 prefill 阶段增加开销）。

### Visual Latent Tokens Connecting Vision and Language

在每个视觉元素（image/video frame）末尾追加一个可学习的 visual latent token，作为视觉 full attention 和语言 causal attention 之间的桥梁。预训练阶段用大 ViT 的 global feature 监督该 token 的输出。

---
## Pre-training

### Pre-training Data

**Figure 4. 预训练和中间训练阶段的数据分布**
![](https://arxiv.org/html/2604.07430v1/x4.png)

数据构建覆盖四大类：

**Visual Perception Data:**
- **Omni-Detection** (62M): 2D/3D 检测，坐标归一化到 (0, 1000)
- **Depth Estimation** (36M): 绝对和相对深度，跨数据源统一焦距归一化
- **Segmentation** (5M): 从 SA-1B 过滤高质量 mask，用 PaliGemma 方法编码为 QA 对
- **Pointing and Counting** (11M): 高难度点级感知和计数

**Embodied-Centric Data:** 按三层层次组织
- **Perception 层**: Grounding（点/框/referring expression）、Affordance（结合用户指令的 affordance 预测）
- **Semantic 层**: Understanding（多层级 VLM 能力综合）、Reasoning（长 horizon 复杂推理）
- **Planning 层**: Trajectory（从视频用 CoTracker3 提取运动轨迹，VLM judge 过滤）、Planning（VLM 标注任务 + 时序分割 + 动作预测）

**Spatial-Centric Data:** 五类（Correspondence / Geometry / Configuration / Measurement / Dynamics），来自 ScanNet、ScanNet++、ARKitScenes 等

**General Understanding Data:** 通用 VLM 数据（语义、STEM、文档解析、复杂推理、GUI 导航等）

### Training Recipe

**Figure 5. HY-Embodied-0.5 训练流水线**
![](https://arxiv.org/html/2604.07430v1/x5.png)

**Pre-training:**
- 389B tokens 通用 + 236B tokens embodied/perception（spatial+robotics 占 43%）
- LR: 5e-5（ViT: 5e-6），batch size 256，max context 32k tokens
- ViT + MoT + latent tokens 均可训练，ViT 梯度每 5 步更新一次

**Embodied-Spatial Mid-training:**
- ~30M 高质量 embodied/spatial 数据，general:embodied:spatial = 12:5:3
- MoT-2B 使用 long + short reasoning chain（\think / \no_think tokens）
- MoE-32B 仅用 short-chain 数据
- 冻结 ViT，仅更新 HY-Embodied 模块

### Training Strategy

三个 loss 联合优化：

**Equation 1. Vision loss（visual next-code prediction）**

$$
\mathcal{L}_{\text{vision}}=-\frac{1}{N_{v}}\sum_{i=1}^{N_{v}}\log p_{i}(z_{i})
$$

**符号说明**: $N_v$ 为视觉 token 数，$p_i$ 为第 $i$ 个 token 的预测概率分布，$z_i$ 为 teacher ViT 生成的目标 discrete code。

**Equation 2. Global loss（latent token 对齐）**

$$
\mathcal{L}_{\text{global}}=-\frac{f_{\text{latent}}^{\top}f_{\text{teacher}}}{\|f_{\text{latent}}\|\|f_{\text{teacher}}\|}
$$

**含义**: 负余弦相似度，将 latent token 的隐状态与 teacher ViT 的 global CLS feature 对齐。

总 loss: $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{llm}}+\mathcal{L}_{\text{vision}}+\mathcal{L}_{\text{global}}$。Mid-training 及之后仅保留 $\mathcal{L}_{\text{llm}}$。

---
## Post-training

### Supervised Fine-tuning

- 从 spatial/embodied/general 数据中采样高复杂度多步问题
- Human-model 协作构建 CoT 轨迹，LLM 多维度评估质量
- 约 100k cold-start CoT 实例用于 SFT
- 禁用 sequence packing，每个样本独立处理

### Reinforcement Learning

**数据构建**: 动态适应模型能力，维护大候选池，多次采样评估后只保留部分成功的样本（能力边界附近），每轮 50K 样本，跨 perception/prediction/interaction/planning 平衡。

**Reward 设计:**

**Figure 6. Embodied RL 的 reward 设计**
![](https://arxiv.org/html/2604.07430v1/x6.png)

四类 task-aware reward：
- **Grounding-Based**: IoU、Hungarian-matched IoU、点距离、Chamfer distance
- **Trajectory-Based**: DTW/Frechet distance + endpoint consistency
- **Regression-Based**: 平滑衰减的相对误差
- **Textual-Based**: exact match / LLM judge fallback

训练使用 GRPO，group-relative advantage normalization：

**Equation 3. GRPO advantage**

$$
A_{i}=\frac{r_{i}-\mu(\mathbf{r})}{\sigma(\mathbf{r})}
$$

G=16，asymmetric clipping [0.8, 1.35]，max prompt/response length 16384 tokens，batch size 128，LR 8e-7，每阶段 5 epochs。

### Evolving Deep Thinking with Iterative Training

RL 和 RFT 交替执行的迭代后训练：
- **RL**: 通过 reward-driven exploration 扩展能力边界
- **RFT (Rejection Sampling Fine-tuning)**: 多次采样 → 保留 partial success 样本 → teacher 模型评估推理质量 → 过滤约 1M → 300K 高质量 trace 进行 SFT
- RL 发现新能力，RFT 巩固为稳定行为，交替迭代逐步深化 deep thinking

### Large-to-Small On-Policy Distillation

On-policy distillation（OPD）将大模型的推理行为迁移到小模型：

**Equation 4. On-policy distillation loss**

$$
\mathcal{L}_{\mathrm{OPD}}=\mathbb{E}_{x,\,y\sim\pi_{s}(\cdot\mid x)}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\mathrm{KL}\!\left(\pi_{t}(\cdot\mid x,y_{<t})\,\|\,\pi_{s}(\cdot\mid x,y_{<t})\right)\right]
$$

**含义**: Student 先用自己的 policy 生成 response，然后 teacher 在 student 生成的 prefix 上做 teacher forcing，最小化 token-level KL divergence。关键优势是在 student 自身会犯错的 state 上学习，减少 train-inference mismatch。

---
## Evaluation

### Results of HY-Embodied-0.5 MoT-2B

22 个 benchmark 中 16 个最优，平均 58.0%，超过 Qwen3-VL-4B（+10.2%）和 [[2601-RoboBrain25|RoboBrain2.5]]-4B（+8.6%）。

**Table 1. MoT-2B Benchmark 结果（部分）**

| Benchmark | HY-Embodied 0.5 MoT-2B | Qwen3-VL 2B | Qwen3-VL 4B | [[2601-RoboBrain25\|RoboBrain 2.5]] 4B | MiMo-Embodied 7B |
|---|---|---|---|---|---|
| CV-Bench | **89.2** | 80.0 | 85.7 | 86.9 | 88.8 |
| DA-2K | **92.3** | 69.5 | 76.5 | 79.4 | 72.2 |
| 3DSRBench | **57.0** | 39.9 | 43.9 | 44.8 | 42.0 |
| MindCube | **66.3** | 28.4 | 31.0 | 26.9 | 36.2 |
| MMSI-Bench | **33.2** | 23.6 | 25.1 | 20.5 | 31.9 |
| ViewSpatial | **53.1** | 37.2 | 41.6 | 36.6 | 36.1 |
| VSIBench | **60.5** | 48.0 | 55.2 | 41.7 | 48.5 |
| EmbSpatial-Bench | **82.8** | 75.9 | 80.7 | 73.8 | 76.2 |
| RoboBench-MCQ | **49.2** | 36.9 | 45.8 | 44.4 | 43.6 |
| ShareRobot-Aff. | **26.8** | 19.8 | 25.5 | 25.5 | 9.0 |

**Insights**: 最显著优势出现在 spatial understanding benchmarks，表明 MoT 架构 + embodied 数据有效建立了 fine-grained spatial reasoning 能力。尽管只有 2B activated parameters，在多数 benchmark 上仍超过更大的模型。

**Figure 7. 通用 benchmark 上的表现**
![](https://arxiv.org/html/2604.07430v1/x7.png)

在通用 VLM benchmark 上与 Qwen3-VL-2B、InternVL 3.5-2B 等同规模模型性能相当，说明 embodied 优化没有牺牲通用能力。

### Results of HY-Embodied-0.5 MoE-A32B

**Table 2. MoE-A32B vs Frontier VLMs（部分）**

| Benchmark | HY-Embodied 0.5 MoE A32B | [[2602-KimiK25\|Kimi K2.5]] | Seed 2.0 | Qwen 3.5 A17B | Gemini 3.0 Pro |
|---|---|---|---|---|---|
| MindCube | **69.2** | 57.8 | 55.2 | 59.0 | 66.0 |
| ViewSpatial | **59.8** | 45.2 | 56.4 | 52.2 | 50.8 |
| VSIBench | **68.3** | 54.2 | 51.0 | 61.1 | 57.9 |
| SITE-Bench-Video | **72.5** | 71.5 | 68.9 | 72.3 | 69.8 |
| RoboSpatial-Home | **76.6** | 66.0 | 71.7 | 74.9 | 57.1 |
| ShareRobot-Traj. | **76.9** | 68.5 | 71.8 | 73.8 | 68.7 |

**Insights**: 整体得分 67.0%，超过 Gemini 3.0 Pro (63.6%) 3.4 个点。22 个 benchmark 中 7 个第一、6 个第二。

### Analysis

**Figure 11. MoT 效率分析**
![](https://arxiv.org/html/2604.07430v1/x11.png)

MoT 训练收敛更快、final loss 更低；推理时总时间接近 Dense-2B baseline，因为 decode 阶段主导推理时间而 MoT 额外开销仅在 prefill 阶段。

**Figure 12. Visual Latent Token 的 attention 可视化**
![](https://arxiv.org/html/2604.07430v1/x12.png)

Visual attention 精确定位显著物体和关键空间区域，language attention 聚焦对应的核心语义实体、状态和动作指令，验证 latent token 有效桥接了模态间隙。

---
## Robot Control Results

基于 MoT-2B 扩展 Action Expert（遵循 [[2410-Pi0|π0]]/[[2504-Pi05|π0.5]] 的设计），构建 VLA 模型。

训练流程：
1. 用 5K 小时 UMI 数据微调（32 GPU，batch 32，200K iterations），不接触具体 robot embodiment
2. 在 3 个真实任务上用 300-700 episodes SFT + 部署评估

**Figure 13. 机器人实验设置与结果**
![](https://arxiv.org/html/2604.07430v1/fig/robot_example.png)

| Task | [[2410-Pi0\|π0]] | [[2504-Pi05\|π0.5]] | HY-Embodied VLA |
|---|---|---|---|
| Precision Plug-in Packing | 80% | 85% | **85%** |
| Tableware Stacking | 60% | 85% | 80% |
| Mug Hanging | 45% | 50% | **75%** |

**Insights**: Mug Hanging 任务上大幅超越 [[2410-Pi0|π0]] (+30%) 和 [[2504-Pi05|π0.5]] (+25%)，说明 5K 小时 UMI 预训练 + MoT 架构建立了有效的 generalizable representations。

---
## 关联工作

### 基于
- Hunyuan-1.8B: 底座 LLM
- Mixture-of-Transformers (MoT): 模态自适应计算架构
- GRPO (DeepSeekMath): RL 训练的核心算法

### 对比
- Qwen3-VL 2B/4B: 通用 VLM baseline
- [[2601-RoboBrain25|RoboBrain 2.5]] 4B: embodied specialist VLM
- MiMo-Embodied 7B: embodied specialist VLM
- Gemini 3.0 Pro / Seed 2.0 / [[2602-KimiK25|Kimi K2.5]]: frontier VLM（32B 对比组）
- [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]]: VLA baseline（robot control 实验）

### 方法相关
- On-policy distillation: 从大模型到小模型的知识迁移
- Visual latent tokens / Vision registers: 增强视觉-语言连接
- CoTracker3: 从视频中提取运动轨迹
- PaliGemma: segmentation mask 的 tokenization 方法

---
## 论文点评

### Strengths

1. **MoT 架构设计精巧**: 通过参数复制而非增加新模块的方式，用最小推理开销换取视觉建模能力的显著提升。视觉分支的双向 attention + next-code prediction 是合理的 inductive bias
2. **Embodied 数据工程系统性强**: 100M+ 数据覆盖 perception→understanding→planning 三层，数据构建方法论清晰（CoTracker3 轨迹提取 + VLM judge 过滤 + 焦距归一化等）
3. **后训练 pipeline 完整且有理论动机**: RL（exploration）→ RFT（consolidation）→ OPD（transfer）的三阶段设计逻辑自洽，OPD 在 student policy 上做 teacher forcing 解决了 train-inference mismatch
4. **评估覆盖面广**: 22 个 benchmark + real-world robot control，且报告协议对自身更保守（只用 thinking mode，baseline 取两种 mode 中的较优）
5. **开源 2B 模型**: 对社区有实际价值

### Weaknesses

1. **32B 模型未开源**: MoE-A32B（407B total）是核心 teacher 模型，但仅开源 2B student，社区无法复现 distillation pipeline
2. **Robot control 实验局限**: 仅 3 个任务，每个 300-700 episodes，成功率基于未说明次数的评估。没有报告 SIMPLER 等标准化 benchmark
3. **数据未开源**: 100M+ embodied 数据（尤其是 in-house 部分）不可获取，大量数据工程细节无法复现
4. **RL 细节不足**: 迭代训练轮数、RL 和 RFT 各执行了几轮、每轮的提升幅度等关键 ablation 缺失
5. **Baseline 选择有争议**: 主要对比 Qwen3-VL 而非 Qwen3.5-VL（理由是后者输出重复），但这个理由不够充分；与 MiMo-Embodied 7B 对比时后者参数更多但总体仍弱于 2B 模型，是否公平值得讨论

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only（已开源推理代码，vLLM 和 fine-tuning 代码待发布）
- **模型权重**: HY-Embodied-0.5 MoT-2B（4B total / 2.2B activated，HuggingFace 发布，8GB）；MoE-A32B 未开源
- **训练细节**: 超参完整（LR、batch size、context length 等均已披露），数据配比已说明，但训练步数和迭代轮数部分缺失
- **数据集**: 部分公开（引用了 SA-1B、ScanNet、ScanNet++、ARKitScenes 等），大量 in-house 和 reasoning 数据未开源

#### Claim 可验证性
- ✅ MoT-2B 在 16/22 benchmark 上 SOTA：benchmark 均为公开数据集，可独立复现评估
- ✅ MoT 推理效率接近 Dense-2B：论文提供了具体的 prefill/decode 时间分解
- ⚠️ MoE-A32B 超越 Gemini 3.0 Pro：部分 baseline 结果为 "self-collected via API"，评估一致性无法完全保证
- ⚠️ 迭代 RL+RFT 的必要性：缺少 ablation 证明每轮迭代的边际收益
- ⚠️ Robot control 实验：仅 3 个任务、评估次数未说明、无标准化 benchmark

### Notes

### Rating

**分数**：2 - Frontier
**理由**：Field-centric 看，这是 embodied VLM 方向当下必须比较的前沿 baseline——2B 模型在 22 个 embodied/spatial benchmark 中拿下 16 个 SOTA，MoT + 迭代 RL/RFT + OPD 是方法范式的代表组合。但距离 Foundation 档（如 π0、ImageNet 级 de facto 标准）仍有距离：32B teacher 和 100M+ 数据闭源、robot control 仅 3 个任务缺乏标准化 benchmark 背书、发布时间尚短未经社区广泛采纳验证，未来可能升 3 也可能被快速迭代的同档工作取代。
