---
title: "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method"
authors: [Xinshuai Song, Weixing Chen, Yang Liu, Weikai Chen, Guanbin Li, Liang Lin]
institutes: [Sun Yat-sen University, Peng Cheng Laboratory, Tencent America]
date_publish: 2024-12-12
venue: CVPR 2025
tags: [VLN, navigation, task-planning]
paper: https://arxiv.org/abs/2412.09082
website: https://hcplab-sysu.github.io/LH-VLN/
github: https://github.com/HCPLab-SYSU/LH-VLN
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
> - **核心**: 提出 LH-VLN 任务（多 subtask 串联、平均 150 step 的长程 VLN），并配套数据生成平台 NavGen、benchmark LHPR-VLN（3260 任务）、新指标（ISR/CSR/CGT）和带 short-/long-term memory 的 MGDM 模型
> - **方法**: NavGen 用 GPT-4 双向（forward 出复杂任务、backward 把 trajectory 分段成 step-by-step 任务）合成数据；MGDM = Vicuna-7B + EVA-CLIP-ViT 视觉编码 + CoT feedback + entropy-pooling 短期记忆遗忘 + dataset-retrieval 长期记忆
> - **结果**: 2-3 subtask 任务下所有 baseline (含 NaviLLM、GPT-4+NaviLLM、GLM-4v) 的 SR/ISR/CSR/CGT 全为 0；MGDM 在 3-4 subtask 下 ISR=4.69、CSR=3.30、CGT=5.83、NE=1.23，相对最高但绝对值仍很低
> - **Sources**: [paper](https://arxiv.org/abs/2412.09082) | [website](https://hcplab-sysu.github.io/LH-VLN/) | [github](https://github.com/HCPLab-SYSU/LH-VLN)
> - **Rating**: 2 - Frontier（问题定义和指标有价值、CVPR 2025 新 benchmark，绝对性能极低 + 缺关键 LLM-based VLN baseline 使其尚未成为 de facto 标准）

**Key Takeaways:**
1. **Long-horizon 是 VLN 真正的瓶颈，不是另一个 +0.3% SOTA**：现有 baseline 在 2-3 subtask 长度上 SR 全 0，说明 single-stage VLN 训练出来的能力对多阶段顺序推理几乎没有迁移
2. **数据合成换 manual annotation**：GPT-4 双向生成 + RAM 视觉标注，把 VLN 数据生成从靠人写指令转向 scene-asset → task → trajectory → step-by-step instruction 的全自动 pipeline
3. **Sub-task-level 评估 (ISR/CSR/CGT) 比聚合 SR 更有信号**：在 long-horizon 任务里 overall SR 长期为 0，ISR 才能区分 "完全不会做" 和 "能做对几步"
4. **memory 不是单一长度的 buffer**：MGDM 用 entropy 最小化指导 average-pooling 决定哪段历史可被合并，而非简单 FIFO 丢弃；ablation 显示丢掉 long-term memory 性能掉到接近零

**Teaser. NavGen 数据生成 pipeline 与 LH-VLN 任务示意。**
![](https://arxiv.org/html/2412.09082v3/x1.png)

---

## 任务定义：LH-VLN 与现有 VLN 的差异

LH-VLN 任务格式："Find something somewhere, and take it to something somewhere, then ..."。每条 complex task 包含 2-4 个 sequential 单阶段导航 sub-task，平均 150 个 action step。Sub-task 完成判定：agent 接近目标物体 1 米测地距离内，且物体落在 60° 水平视野内。

与代表性 VLN benchmark 的对比：

**Table 1. VLN benchmark 对比**

| Benchmark | Avg. Instr. Length | Avg. Task Steps | Simulator | Task Type | Scenes | Task Num |
| --- | --- | --- | --- | --- | --- | --- |
| R2R | 29 | <8 | Matterport3D | Step-by-step Nav | 90 | 21567 |
| REVERIE | 18 | <8 | Matterport3D | Obj Loco-nav | 90 | 21702 |
| VLN-CE | 30 | 55.88 | Habitat | Step-by-step Nav | 90 | 4475 |
| FAO | 39 | 10 | Matterport3D | Obj Loco-nav | 90 | 3848 |
| Behavior-1k | 3.27 | - | OmniGibson | Complex Housework | 50 | 1000 |
| IVLN | - | - | M3D&Habitat | Iterative VLN | 72 | 789 |
| Goat-Bench | - | - | Habitat | Iterative VLN | 181 | 725360 |
| **LHPR-VLN (Ours)** | 18.17 | **150.95** | Habitat | Multi-stage VLN | 216 | 3260 |

**关键差异**：task step 数比传统 VLN 高 1-2 个数量级（150 vs <8 ~ 55），且要求 multi-stage 顺序完成；指令并不一定更长（18.17 接近 REVERIE 的 18），复杂性体现在结构 / 顺序而非单条指令的长度。

> ❓ Behavior-1k 已经支持 complex housework，IVLN/Goat-Bench 也支持 iterative VLN，"first LH-VLN benchmark" 的 claim 主要靠 task step 数量和 step-by-step 子任务结构；novelty 偏 incremental on benchmark side。

---

## NavGen：数据生成平台

NavGen 用一个 **bi-directional** pipeline：forward 生成 complex task → 在 simulator 跑出 trajectory → backward 切分 trajectory 为 step-level 子任务。

### Forward generation

- **Asset 池**：HM3D（216 个语义标注的 3D 室内场景）+ 机器人配置（Boston Dynamics Spot、Hello Robot Stretch，不同相机高度 / 任务空间 / 传感器）
- 用自定义 prompt 把场景细节 $S$ 与机器人配置 $R$ 喂给 GPT-4，输出 instruction list $D_{ins} = \mathcal{G}(S, R, \text{prompt}_1)$，包含 sub-task 和 multi-stage 指令
- $D_{ins}$ 导入 Habitat3 simulator，由 expert model（navmesh + greedy pathfinder）或预训练 nav model 生成 trajectory：

$$
D_{traj}=Sim(D_{ins},S,A,\mathbf{OR}(M,E))
$$

其中 $\mathbf{OR}$ 表示 expert model $E$ 或 nav model $M$ 之一。

### Backward generation

- 把单 sub-task 的 trajectory 用 trajectory splitting algorithm（dynamic sliding window 寻找最长连续 action 段）切成 "move forward / turn left / turn right / bypass forward" 的 action segment
- 每段用 RAM 模型给 high-confidence 视觉标注，与 action instruction 一起喂 GPT-4 生成 step-by-step VLN 任务

> ❓ Backward generation 的核心目的是把粗粒度任务"反向蒸馏"出 step-level 标注，但这条路径里 GPT-4 看到的只是 RAM 关键词 + action label，没有真实视觉 frame——生成的 step-by-step 指令的 grounding 质量值得怀疑。

---

## LHPR-VLN benchmark

3260 个 multi-stage VLN 任务，覆盖 216 个复杂场景，平均 150.95 步、18.17 token 指令长度。Robot 类型分布：Spot 50.5% / Stretch 49.5%；subtask 数量分布：2 个 39.0%、3 个 52.4%、4 个 8.6%。

观测设置：agent 每步从三个方向（+60°、0°、-60°）拿 RGB 观测；动作集 = {turn left, move forward, turn right, stop}。stop 表示 sub-task 完成，按 agent 终态相对目标的距离 / 视野判定成功。

**Figure 2. LHPR-VLN benchmark 统计概览**
![](https://arxiv.org/html/2412.09082v3/x2.png)

---

## 评估指标：ISR / CSR / CGT

传统 SR 在 long-horizon 任务上几乎全 0，无法区分 "完全不会做" 和 "做对若干步"。论文引入三个细粒度指标：

**Equation. Independent Success Rate**

$$
ISR=\sum_{j=0}^{M}\sum_{i=0}^{N}\frac{s_{j,i}}{M\cdot N}
$$

每个 sub-task 独立计算成功率，反映单步完成能力。

**Equation. Conditional Success Rate**

$$
CSR=\sum_{j=0}^{M}\sum_{i=0}^{N}\frac{s_{j,i}(1+(N-1)s_{i-1})}{M\cdot N^{2}}
$$

加入 sub-task 之间的依赖：第 $i$ 个 sub-task 的 credit 取决于前一个是否成功，编码任务序列的 inter-dependency。

**Equation. CSR weighted by Ground Truth (CGT)**

$$
CGT=\sum_{j=0}^{M}\sum_{i=0}^{N}\frac{P_{i}}{P}\cdot\frac{s_{j,i}(1+(N-1)s_{j,i-1})}{M\cdot N}
$$

按 ground-truth path 长度 $P_i / P$ 加权，惩罚在长 / 难子任务上的失败。

**符号**：$M$ 任务数，$N$ 单任务的 sub-task 数，$s_{j,i}$ 第 $j$ 个任务第 $i$ 个 sub-task 是否成功。

> ❓ CSR 公式里 inner term 用的是 $s_{i-1}$（无下标 $j$）而 CGT 用 $s_{j,i-1}$，看起来 CSR 的写法是 typo——按语义应当也是 $s_{j,i-1}$。

此外引入 Target Approach Rate (TAR) 基于 NE，作为 SR 太低时的额外信号。

---

## MGDM 模型

**Figure 3. MGDM 框架**
![](https://arxiv.org/html/2412.09082v3/x3.png)

MGDM = Base Model + CoT Feedback + Adaptive Memory Integration and Update (AMIU)。LLM 是 Vicuna-7B v0，视觉编码 ViT 来自 EVA-CLIP-02-Large 且训练时 frozen。

### Base model

每张方向图像 $I_i$ 经 ViT 得 $v_i$，多视角通过 Transformer encoder 融合得 $\{o_i\}$，再加方向 token 拼成 scene representation：

$$
S=[\mathcal{E}(\text{`left'}), o_{\text{left}}, ..., \mathcal{E}(\text{`right'}), o_{\text{right}}]
$$

历史观测 $H_n$ 类似处理但加 stepwise embedding 编码时序，最终拼接成 prompt 喂 LLM 选下一动作 $a_{n+1} = \mathcal{G}(\mathcal{E}(\text{prompt}_3), S, H_n)$。

### CoT Feedback

每个 sub-task 开头与导航中周期性触发：把 (current obs, hist obs, instruction, prompt) 喂 GPT-4 生成 chain of thought：$\text{CoT} = \text{GPT-4}(\text{Obs, Hist, Instruction, Prompt})$，用于 task decomposition 和 action planning。

> ❓ 周期触发 GPT-4 = 推理时持续依赖外部商业 API，这对 "agent" 的 deployment 是硬约束；论文未报告调用频率与每 episode token 成本。

### Adaptive Memory Integration and Update (AMIU)

**Short-term memory** $M_{st} = \{h_i\}_{i=0}^{n}$。当长度达到阈值 $N$ 触发 dynamic forgetting：每个 $h_i$ 关联一个 confidence $c_i$，构成向量 $C$。定义 pooling 算子 $\mathcal{P}$ 对窗口大小 2 的相邻元素 average pooling，把长度从 $n$ 降到 $n-1$：

$$
\mathcal{P}(C)_i=\{c_1,...,\text{AvgPool}(c_{i-1},c_i,c_{i+1}),...,c_n\}
$$

对每个 $i$ 都试一次 pooling 得 $\{C_i\}$，选 entropy 最小的 index：

$$
\text{arg}\min_{i}\left(-\sum_{j=1}^{n-1}s_{j}\log s_{j}\right),\;s_{j}=\frac{C_{i,j}}{\sum_{j=0}^{n-1}C_{i,j}}
$$

然后对 $M_{st}$ 在该 index 做相同 pooling，并追加新观测 $h_n^*$。**直觉**：合并最不会改变 confidence 分布形状（最低熵增量）的位置，等价于 "信息最冗余" 的相邻 step。

**Long-term memory** $M_{lt} = \text{Dataset}(T) = \{obs_j, act_j\}_{j=1}^m$ 从 LHPR-VLN 数据集按目标 $T$ 检索，用 cosine similarity 选 top-$k$ 与当前 observation $v$ 匹配的 obs-act pair：

$$
I_k=\text{argsort}_{t=0}^{k}\left(\left\{\frac{obs_j\cdot v}{\sqrt{\sum obs_{j,i}^2}\sqrt{\sum v_i^2}}\right\}_{j=1}^m\right)
$$

把 top-$k$ action 平均后乘到当前 decision vector：$a = a \cdot \text{avg}(\{act_t\}_{t=0}^k)$。

训练 loss 是模型决策 $a$ 与 expert 决策 $e$ 的交叉熵：$\arg\min_\Theta \mathcal{L}(a,e) = -\sum a_i \log e_i$。

> ❓ Long-term memory 直接从 training set retrieve 用于 inference-time 决策加权，本质是 retrieval-augmented inference + soft action voting。"long-term memory" 的命名容易让人误解为 episodic memory（同一 episode 内更早 step），实际是 cross-episode 数据集检索。

---

## 实验

### Setup

- Simulator: Habitat3，部分实验在 Isaac Sim
- Sensors: 三方向 RGB（front, +60°, -60°），可选 depth
- Actions: move forward (+0.25 m), turn left (+30°), turn right (-30°), stop
- Scene: HM3D + HSSD（用于测试 NavGen 数据生成）
- LLM: Vicuna-7B v0；ViT: EVA-CLIP-02-Large（frozen）
- 训练：Adam, lr=3e-5，alternate imitation learning 和 trajectory-based supervised learning

### Baselines

- **ETPNav**: graph-based waypoint predictor
- **GLM-4v prompt**: SOTA VLM with prompt engineering, zero-shot
- **NaviLLM**: SOTA discrete-environment nav model, adapted to continuous + 在 LHPR-VLN finetune
- **GPT-4 + NaviLLM**: GPT-4 拆任务，NaviLLM 顺序执行 sub-task

### 主结果

**Table 2. LH-VLN 任务上的性能（按 task length 拆分）**

| Method | Type | 2-3 SR | NE | ISR | CSR | CGT | 3-4 SR | NE | ISR | CSR | CGT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | - | 0 | 14.09 | 0 | 0 | 0 | 0 | 10.91 | 0 | 0 | 0 |
| GLM-4v prompt | Zero-shot | 0 | 15.63 | 0 | 0 | 0 | 0 | 10.97 | 0 | 0 | 0 |
| NaviLLM | Pretrain | 0 | 12.11 | 0 | 0 | 0 | 0 | 10.04 | 0 | 0 | 0 |
| NaviLLM | Finetuned | 0 | 12.24 | 0 | 0 | 0 | 0 | 9.79 | 3.54 | 2.53 | 5.24 |
| GPT-4 + NaviLLM | Pretrain | 0 | 12.23 | 0 | 0 | 0 | 0 | 10.00 | 4.37 | 2.91 | 5.23 |
| **MGDM (Ours)** | Finetuned | **0** | **3.54** | **0** | **0** | **0** | **0** | **1.23** | **4.69** | **3.30** | **5.83** |

**关键 observation**：
- **2-3 subtasks 列里所有方法 SR/ISR/CSR/CGT 全为 0**——MGDM 也没能完整完成任意一个 sub-task；唯一的 differentiator 是 NE（MGDM 3.54 vs baseline 12-15）
- 3-4 subtasks 反而比 2-3 表现好——作者解释为"长任务里前面阶段累积的 memory 反过来帮助后面阶段"，但更可能的解释是 3-4 任务的 navigation target 分布偏向场景内更显眼 / 更易达的位置（论文也提到"target distribution may also influence"）
- GPT-4+NaviLLM 在 ISR 上比 fine-tuned NaviLLM 提高 23%，但 CGT 反而更低，说明任务分解 + 单阶段执行能 boost 简单子任务但难胜任长 / 难子任务

**Table 3. Step-by-step LH-VLN 任务（单阶段细粒度指令）**

| Method | SR ↑ | OSR ↑ | SPL ↑ | NE ↓ |
| --- | --- | --- | --- | --- |
| Random | 0 | 0 | 0 | 8.59 |
| GLM-4v prompt | 0 | 11.1 | 0 | 6.50 |
| NaviLLM | 6.67 | 6.67 | 2.86 | 10.17 |
| **MGDM (Ours)** | 0 | 26.92 | 0 | 1.70 |

MGDM 在 step-by-step 任务上 OSR 高、NE 极低但 SR=0：说明能"路过"目标但**判断不出何时该 stop**。NaviLLM 反而 SR=6.67：在 step-by-step 任务上单阶段模型仍然有 stop 判定优势。

**Figure 4. 一个 partial success 长程导航的可视化**：第一段任务 agent 找浴室里的毛巾，找到了浴室和毛巾但没进入浴室、未足够接近，被判失败；下一阶段成功找到客厅里的箱子。

![](https://arxiv.org/html/2412.09082v3/x4.png)

### Ablation

**Table 4. MGDM ablation（3-4 subtasks 设置）**

| Method | NE ↓ | ISR ↑ | CSR ↑ | CGT ↑ |
| --- | --- | --- | --- | --- |
| MGDM w/o Adap Mem | 4.44 | 0 | 0 | 0 |
| MGDM w/o LT Mem | 11.13 | 2.20 | 1.27 | 2.08 |
| MGDM w/o CoT | 2.45 | 0 | 0 | 0 |
| MGDM (full) | 1.23 | 4.69 | 3.30 | 5.83 |

- 去 adaptive memory 或 CoT → ISR 直接归零，说明二者是 sub-task 完成的"前提"，而非锦上添花
- 去 long-term memory → 性能掉但非零，说明 LT mem 是 boost 而非必需；NE 反而显著恶化（11.13 vs 1.23）说明 LT retrieval 在防止 agent 走偏上贡献最大

---

## 关联工作

### 基于
- **Habitat3 / HM3D**: simulator 和 scene asset 来源
- **Vicuna-7B + EVA-CLIP**: base LLM 和视觉编码
- **NaviLLM**: backbone 灵感来源（README 致谢中提到"refer to some codes of NaviLLM"）
- **RAM (Recognize Anything)**: backward generation 中视觉标注

### 对比
- **[[2304-ETPNav|ETPNav]]**: graph-based waypoint VLN baseline，long-horizon 上 waypoint predictor 失效
- **NaviLLM**: SOTA discrete-env LLM-based VLN，被作为最强 single-stage baseline；GPT-4+NaviLLM 是显式 task decomposition 的对照
- **GLM-4v prompt**: zero-shot VLM baseline

### 方法相关
- **[[2402-NaVid|NaVid]]、[[2412-NaVILA|NaVILA]]**: 同期 LLM-based VLN 工作，论文未对比但属于直接相关方法
- **CoT prompting**: 用 GPT-4 在线 decompose task 是这一思路在 VLN 的应用
- **Retrieval-augmented agents**: long-term memory 本质是 cross-episode retrieval

### Benchmark 相关
- **R2R / REVERIE / VLN-CE / FAO**: short-horizon VLN baseline benchmark
- **Behavior-1k / IVLN / Goat-Bench**: 已有的 multi-stage / iterative VLN benchmark，作者的"first LH-VLN"主要靠 task step 数和细粒度子任务结构区分

---

## 论文点评

### Strengths

1. **问题定义有价值**：把 VLN 从 single-stage 推到 multi-stage / long-horizon 是一个真实的下一步——现实场景中机器人很少只完成一个 atomic task；当前 VLN benchmark 在 task step 数上确实是 bottleneck（55 → 150 是数量级提升）
2. **新指标解决了 SR=0 时的评估塌陷**：ISR/CSR/CGT 让长程任务下"完全不会做"和"能做对几步"被区分开，这是 long-horizon 任务必须的诊断粒度
3. **AMIU 的 entropy-pooling 思路**：基于 confidence 分布的熵选择最冗余位置 merge，比简单 FIFO 丢弃更符合"保留信息丰度"的直觉，是相对 novel 的 memory compression 设计
4. **Bi-directional data generation**：forward + backward 同时给 multi-stage 任务和 step-level 任务，复用一份 trajectory，scaling 友好

### Weaknesses

1. **绝对性能极低**：MGDM 在 2-3 subtasks 上 SR/ISR 全 0；3-4 subtasks 上 ISR 才 4.69%（即只能完成约 5% 的子任务），benchmark "可解决性" 仍未被证明——很难判断瓶颈是模型容量、数据规模还是任务定义本身
2. **关键 baseline 缺失**：没有对比 NaVILA、NaVid 这类同期 LLM-based VLN 模型（CVPR 投稿时间窗口可能解释一部分），也没有对比 hierarchical RL baseline；不易判断 MGDM 的 contribution 来自模型设计还是 finetuning on LHPR-VLN
3. **Long-term memory 是 cross-episode dataset retrieval**：本质上是 retrieval augmentation，agent 的 "long-term" 不是 episode 内累积的 episodic memory；命名 misleading 且让 evaluation 与 train set leakage 风险增加（test-time retrieve from train set）
4. **CoT 依赖 GPT-4 在线 API**：周期性调用 GPT-4 做 task decomposition，模型本质上是 GPT-4 + 小 LLM 执行的 hybrid，"7B 模型"的 claim 在 deployment 视角不真实
5. **3-4 subtask 反而比 2-3 更好的解释不充分**：作者归因于"memory 帮助后续 stage"，但同时承认 target distribution 也有影响——没有控制实验区分二者，这是评估方法论上的硬伤
6. **Step-by-step 任务上 SR=0 但 OSR=27**：说明 stop 判定是 critical bottleneck，但论文没专门设计 stop predictor 也没在 ablation 中讨论

### 可信评估

#### Artifact 可获取性

- **代码**: inference + training（GitHub repo 提供 train.py, sft.py, NavGen pipeline，支持 Llama/Qwen 多 scale 替换）
- **模型权重**: 未说明（README 未提供 checkpoint 链接，仅给出 ViT/CLIP/BERT 等 dependency 权重）
- **训练细节**: 高层描述（提到 Vicuna-7B、Adam lr=3e-5、imitation + supervised 交替；未给出 batch size、训练步数、数据配比）
- **数据集**: 开源（[Hugging Face](https://huggingface.co/datasets/Starry123/LHPR-VLN) + ModelScope）

#### Claim 可验证性

- ✅ 各 baseline 在 2-3 subtasks 上 SR=0：Table 2 数据明确，可独立复现 verify
- ✅ ISR/CSR/CGT 公式定义：可直接基于公开 benchmark 重新计算
- ⚠️ "MGDM achieves SOTA on LH-VLN"：在自家提出的 benchmark 上、缺关键 LLM-based VLN baseline（NaVILA/NaVid 等）的对比，SOTA claim 范围有限
- ⚠️ "3-4 subtasks 表现更好是因为 memory 帮助"：作者自己承认 target distribution 也有影响，无控制实验
- ⚠️ Long-term memory 来自 LHPR-VLN dataset，test-time retrieval 是否覆盖到 test split 无明确说明 → 潜在 train/test leakage
- ❌ "first dataset specifically designed for long-horizon VLN"：Behavior-1k、IVLN、Goat-Bench 都已有 multi-stage 元素，"first" 的 framing 偏 marketing

### Notes

- LH-VLN 把 VLN 评估推到 long-horizon 是对的方向，但这篇 paper 暴露了一个尴尬：当 SOTA 模型在 short benchmark 上做到 50%+ SR，搬到 long-horizon 立刻 0%，意味着现有 VLN paradigm 的 capability 几乎完全不能 transfer 到 multi-stage——这是 problem formulation 的胜利，不是方法的胜利
- 真正有意思的对照不是 MGDM vs ETPNav，而是 "single-stage 模型 + GPT-4 拆任务"（GPT-4+NaviLLM）vs "原生 multi-stage 模型"（MGDM）。Table 2 的结果倾向后者，但优势很小（CSR 3.30 vs 2.91），还要扣除 MGDM 自己依赖 GPT-4 做 CoT 的部分——单纯 "task decomposition 是否够用" 这个问题没被干净回答
- "Memory" 这个概念在论文里被用得太宽泛：short-term = entropy pooling buffer，long-term = retrieval from training set，CoT 又是另一种"context memory"。三者解决的问题不同，但都被打包进 "memory" 名号下，让 ablation 信号变模糊
- 对自己研究的启发：long-horizon embodied tasks 的评估指标设计（ISR/CSR/CGT 的 dependency-aware accumulation）值得借鉴，特别是 CGT 用 path length 加权惩罚短任务上的"投机成功"——这在 mobile-manipulation / 多技能 agent 的评估里同样有用
- ❓ 如果把 LHPR-VLN 的 task generation pipeline 换成更现代的 model（GPT-5 / Claude），forward task 的 grounding 质量和 instruction 多样性会有多大提升？这可能是 benchmark 本身的下一个迭代方向

### Rating

**分数**：2 - Frontier
**理由**：问题定义（multi-stage / 150-step VLN）和评估指标（ISR/CSR/CGT 的 dependency-aware accumulation，见 Strengths #1/#2）在 long-horizon VLN 方向具有实质贡献，CVPR 2025 接收 + 数据集已开源，属于当下活跃的前沿工作；但 Weaknesses 指出绝对性能极低（MGDM ISR 仅 4.69%）、缺 NaVid/NaVILA 关键 baseline 对比，benchmark 可解性未被证明，尚未形成 de facto 标准，距离 Foundation 档（如 R2R）还差社区广泛采纳的验证，因此定为 2 而非 3；相对 1 档则因为问题设定本身推动了 single-stage → multi-stage 的范式扩展，不是 incremental / niche 工作。
