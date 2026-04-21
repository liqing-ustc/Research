---
title: "UI-TARS: Pioneering Automated GUI Interaction with Native Agents"
authors: [Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, Guang Shi]
institutes: [ByteDance Seed, Tsinghua University]
date_publish: 2025-01-21
venue: arXiv
tags: [computer-use, gui-agent, VLM]
paper: https://arxiv.org/abs/2501.12326
website:
github: https://github.com/bytedance/UI-TARS
rating: 3
date_added: "2026-04-02"
---
## Summary

> [!summary] UI-TARS: Pioneering Automated GUI Interaction with Native Agents
> - **核心**: 端到端 native GUI agent，仅以 screenshot 为输入，把传统模块化 framework 中的 perception / grounding / planning / memory 收进单一 VLM 参数。
> - **方法**: Qwen-2-VL 7B/72B 上做 ~50B token 的三阶段训练（continual pretraining → annealing SFT → DPO），配合 unified action space、6 种 thought pattern 注入、online trace bootstrapping + reflection tuning。
> - **结果**: 10+ benchmark SOTA。OSWorld（15-step）22.7 vs Claude-CU 14.9；AndroidWorld 46.6 vs GPT-4o 34.5；ScreenSpot-Pro 38.1。
> - **Sources**: [paper](https://arxiv.org/abs/2501.12326) | [github](https://github.com/bytedance/UI-TARS)
> - **Rating**: 3 - Foundation（确立 native GUI agent 范式，成为后续 GUI agent 工作的 de facto baseline 和 framing 来源）

**Key Takeaways:**
1. **Native model > framework**：把 prompt-engineered framework（GPT-4o + 各种 grounding/memory 工具）替换成端到端训练的 VLM，能直接超过包了 Claude / GPT-4o 的 agent framework。规模化前提下，data-driven 系统胜过 design-driven 系统。
2. **System-2 reasoning 在 OOD 才显优势**：in-domain 单样本下 System-2 反而略输 System-1（thought 引入幻觉）；BoN=16/64 时反超；OOD（AndroidWorld）下 System-2 大幅领先。这给 "thinking helps" 加了边界条件。
3. **Reflection tuning + DPO 是数据飞轮的关键**：online bootstrap 自然产生大量 negative samples，SFT 只用正例浪费一半信号，DPO 把错误轨迹也利用上——OSWorld 上 DPO 相对 SFT 提升明显。
4. **Scale 显著影响 online > offline**：72B 在 online benchmark 上拉开 7B 更多，提示离线评测低估了 reasoning 收益，online evaluation 更能区分模型能力。

**Teaser. UI-TARS 帮用户订机票的 demo 案例，展示端到端 screenshot → thought → action 的闭环。**

![](https://arxiv.org/html/2501.12326v1/extracted/6146349/figures/case-1-v8.png)

---
## Problem & Motivation

现有 GUI agent 主要走 **agent framework** 路线：在商用 LLM/VLM（GPT-4o、Claude）外包一层 prompt engineering、grounding 工具、memory 模块。三大限制：

1. **Fragility & Maintenance**：workflow 的 prompt 和脚本对接口/任务变化敏感，每次新场景都要重写
2. **Disjoint Learning**：framework 几乎不更新底层模型参数，只靠离线 prompt engineering，新经验无法 compound
3. **Module Incompatibility**：多个模块靠 prompt 串联，任一环节出错都拖垮整条 pipeline，调试需要领域专家

同时 GUI 域自身有特殊难点：高信息密度、小元素（10×10 icon in 1920×1080）、需要精确坐标、多步轨迹数据稀缺。Native end-to-end 方案的核心瓶颈是**数据**：能整合 perception/reasoning/memory/action 的统一 workflow 数据历史上几乎没记录过。

论文提出 **GUI agent 的四阶段演化**：Rule-based → Agent Framework → **Native Agent Model** → Active & Lifelong Agent，并把自己定位在第三阶段，朝第四阶段铺路。

**Figure 2. GUI agent 演化路径——人类干预度递减、泛化能力递增。**
![](https://arxiv.org/html/2501.12326v1/extracted/6146349/figures/5-stage.png)

---
## Method

### 4.1 架构概览

UI-TARS 在每一步给定 instruction 和历史 `(o, t, a)` tuple 序列，输出 thought + action：

$$
P(t_n, a_n \mid \text{instruction}, t_1, a_1, \cdots, (o_{n-i}, t_{n-i}, a_{n-i})_{i=1}^N, o_n)
$$

为节约 32k context，只保留最后 $N=5$ 个 observation；完整的历史 thought 和 action 作为 short-term memory 保留。Thought $t_i$ 借鉴 ReAct，但比 ReAct 更结构化、更显式地按几种推理模板生成。

**Figure 4. UI-TARS 架构总览：core capabilities (perception / grounding / reasoning / memory) 与训练 pipeline。**
![](https://arxiv.org/html/2501.12326v1/extracted/6146349/figures/model_arc.png)

### 4.2 Perception 增强

五项 task 上训练增强 GUI 理解：

| 任务 | 作用 |
|---|---|
| **Element Description** | 描述 element type / visual / position / function 四要素 |
| **Dense Captioning** | 整体界面布局描述，含空间关系和层级 |
| **State Transition Captioning** | 两连续截图间的差异（含交互/非交互变化） |
| **Question Answering** | 多样化 GUI 理解 QA，鼓励 reasoning |
| **Set-of-Mark (SoM)** | 给元素叠加视觉 marker，强化 marker ↔ element 关联 |

数据来源：用 specialized parsing tool 自动 crawl 截图和 metadata（element type / depth / bbox / text），bottom-up 从单个元素到整体界面构造样本。

**Figure 5. Perception 与 grounding 的训练数据样例。**
![](https://arxiv.org/html/2501.12326v1/x1.png)

### 4.3 Unified Action Space + Grounding

跨平台统一 action 定义：

| Environment | Actions |
|---|---|
| Shared | Click(x,y), Drag(x1,y1,x2,y2), Scroll(x,y,direction), Type(content), Wait(), Finished(), CallUser() |
| Desktop | Hotkey(key), LeftDouble(x,y), RightSingle(x,y) |
| Mobile | LongPress(x,y), PressBack(), PressHome(), PressEnter() |

**Table 1. Unified action space across platforms.**

`CallUser()` 和 `Finished()` 是两个 terminal action，前者请求用户介入（如登录授权）。

**Action trace 数据**（Table 2）：
- Open source: Web 14.8M elements / 6.4k traces；Mobile 2.5M / 145k；Desktop 1.1M / 0
- 自标注：avg 14.9 步/trace（明显比开源数据 7.1/9.6 步更长）
- 开源数据来自 MM-Mind2Web、GUIAct、AITW、AITZ、AndroidControl、GUI-Odyssey、AMEX，统一对齐到上述 action space

**Grounding** 训练：每个 element 的 bbox 取中心点，输出 normalize 到屏幕分辨率的相对坐标。整合 SeeClick / GUIAct / MultiUI / Rico-SCA / WidgetCaption / MUG / Rico Icon / CLAY / UIBERT / OmniACT / AutoGUI / OS-ATLAS。

### 4.4 System-2 Reasoning 注入

#### Tutorial 预训练

从 MINT 和 OmniCorpus 两个 image-text interleaved 数据集中筛 **6M GUI tutorial**，三阶段过滤：(1) fastText 二分类粗筛 → (2) LLM 细筛去 false positive → (3) URL+LSH 去重 + LLM 改写。平均 510 text tokens + 3.3 images / tutorial。

#### Thought Augmentation

action trace 原本只有 `(o, a)`，需要插入 thought $t$。两阶段标注：

**(1) ActRe**：给定 ground-truth action，让 VLM 回填 thought：

$$
t_n = \operatorname{VLM}(\text{instruction}, (o_1, t_1, a_1), \ldots, o_n, a_n)
$$

**问题**：thought 可能只是表面 match action 而非真正的 causal reasoning。

**(2) Thought Bootstrapping**：early-stage 模型不看 ground-truth action，直接采样多个 `(t, a)` pair，挑选预测 action 等于 ground truth 的那对：

$$
\begin{aligned}
&(\hat{t}_{n_i}, \hat{a}_{n_i})_{i=1}^{\text{max-try}} = \operatorname{UI\text{-}TARS}_{\text{early}}(\text{instruction}, (o_1, t_1, a_1), \ldots, o_n) \\
&\operatorname{Select}(\hat{t}_{n_i}, \hat{a}_{n_i}), \text{ where } \hat{a}_{n_i} = a_n
\end{aligned}
$$

**六种 reasoning pattern**（在 ActRe prompt 中诱导 VLM 生成）：
- **Task Decomposition** — 分解复杂任务
- **Long-term Consistency** — 维持目标一致性，对抗多步漂移
- **Milestone Recognition** — 识别中间目标完成
- **Trial & Error** — 在不确定情况假设/试错
- **Reflection** — 失败后识别并纠正

> ❓ 6 种 pattern 是 prompt 层面诱导，并未做 ablation 验证每种 pattern 的边际贡献。"reasoning enrichment" 究竟是 6 个机制都需要还是其中一两个就够？

**Figure 6. 六种 thought pattern 的样例。**
![](https://arxiv.org/html/2501.12326v1/extracted/6146349/figures/thought_pattern.png)

### 4.5 Online Bootstrapping + Reflection Tuning

#### Online Trace Bootstrapping

**Figure 7. Online bootstrapping pipeline：跑数百台 VM、多级过滤、回灌训练。**
![](https://arxiv.org/html/2501.12326v1/extracted/6146349/figures/online_bootstrap.png)

迭代循环：

1. 当前模型 $M_n$ 在数百台 VM 上跑 instruction set $\mathcal{I}_n$，得 raw trace $\mathcal{T}_{\text{raw}, n}$
2. 三级过滤：rule-based reward → VLM scoring → human review（人工只看一部分，标记错误步并截断后续）
3. $M_{n+1} = \operatorname{FineTune}(M_n, \mathcal{T}_{\text{filtered}, n})$
4. 标注员扩充 $\mathcal{I}_{n+1}$，迭代

#### Reflection Tuning

针对 online 部署中常见的卡死循环（反复点不响应的按钮），主动暴露 model 自己产生的错误并标注修正。两类配对样本：

**(1) Error Correction**：在错误步 $\tau$，标注员标出正确 thought/action 替换：
- $\mathcal{T}_{-} = (\ldots, (o_\tau, t_\tau, a_\tau))$ ← 错误
- $\mathcal{T}_{+} = (\ldots, (o_\tau, t^*_\tau, a^*_\tau))$ ← 修正

**(2) Post-Reflection**（更有意思）：**保留**错误步 $a_\tau$，让标注员在 $\tau+1$ 步**承认错误并补救**。例如上一步误点了关闭按钮，下一步要先重开网页再点收藏。这教会模型从已发生的错误中恢复，而不是只学避免错误。

SFT 阶段只用 $\mathcal{T}_+$，且 loss 只算修正步（错误步不参与 loss）。

#### Agent DPO

SFT 不利用负样本是浪费。改用 DPO：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_\tau\left[\log\sigma\left(\beta\log\frac{\pi_\theta(a'_\tau | s_\tau)}{\pi_{\text{SFT}}(a'_\tau | s_\tau)} - \beta\log\frac{\pi_\theta(a_\tau | s_\tau)}{\pi_{\text{SFT}}(a_\tau | s_\tau)}\right)\right]
$$

把每个 error-correction pair 视作 preference data，强化 corrected action 同时压低 erroneous action 的 likelihood。

### 4.6 三阶段训练

| Phase | 数据 | 目的 |
|---|---|---|
| Continual Pre-training | § 4 全量数据（不含 reflection），constant LR | 打 GUI 基础（perception/grounding/action） |
| Annealing | 高质量子集 + reflection tuning data | 收敛到 GUI 任务（产出 UI-TARS-SFT） |
| DPO | online bootstrapping 的 preference pair | 区分 optimal/suboptimal（产出 UI-TARS-DPO） |

总训练量 ~50B tokens，base 是 Qwen-2-VL（2B/7B/72B）——同期 GUI-VLM 工作（如 [[2312-CogAgent|CogAgent]]、[[2410-OSAtlas|OS-Atlas]]、[[2411-ShowUI|ShowUI]]）的 base 选择各异。

---
## Results

### Perception (Table 3)

| Model | VisualWebBench | WebSRC | ScreenQA-short |
|---|---|---|---|
| Claude-3.5-Sonnet | 78.2 | 90.4 | 83.1 |
| GPT-4o | 78.5 | 87.7 | 82.3 |
| UI-TARS-7B | 79.7 | **93.6** | 87.7 |
| UI-TARS-72B | **82.8** | 89.3 | **88.6** |

### Grounding ([[2504-ScreenSpotPro|ScreenSpot-Pro]] / ScreenSpot / v2)

- ScreenSpot-Pro: UI-TARS-72B **38.1** vs UGround-V1-7B 31.1 vs [[2410-OSAtlas|OS-Atlas-7B]] 18.9
- ScreenSpot: UI-TARS-7B **89.5**
- ScreenSpot v2: UI-TARS-7B 91.6 / UI-TARS-72B 90.3 vs OS-Atlas-7B 87.1
- 7B → 72B 在 ScreenSpot v1/v2 几乎无提升，但 ScreenSpot-Pro 显著提升 → v1/v2 已饱和，无法体现 model scale 收益

### Offline Agent (Mind2Web / AndroidControl / GUI Odyssey)

- Multimodal Mind2Web：所有 agent model 显著超 framework-based（GPT-4o/4V planner）
- AndroidControl + GUI Odyssey：UI-TARS 相对 OS-Atlas-7B 绝对值提升 **+25**
- Claude-CU 在 web 强但在 mobile 明显挣扎 → 商用模型未把 GUI 能力迁移到 mobile

### Online Agent (Table 9)

| Benchmark | Setting | Best Baseline | UI-TARS |
|---|---|---|---|
| OSWorld (15 step) | screenshot-only | Claude 14.9 | **UI-TARS-72B-DPO 22.7** |
| OSWorld (50 step) | screenshot-only | Claude 22.0 | **UI-TARS-72B-DPO 24.6 (SOTA)** |
| AndroidWorld | — | GPT-4o + Aria-UI 44.8 | **UI-TARS-72B-SFT 46.6** |

**关键观察**：
- DPO 在 OSWorld 显著超 SFT — 负样本利用是对的
- 72B vs 7B 在 online 上的差距 > 在 offline 上的差距 → online evaluation 更能体现 reasoning 价值；scale 主要受益于 System-2

### System-1 vs System-2 (§5.5)

**Figure 8. System-1 vs System-2 在 in-domain (Mind2Web/AndroidControl/GUI Odyssey) 与 OOD (AndroidWorld) 上的对比，BoN ∈ {1, 16, 64}。**
![](https://arxiv.org/html/2501.12326v1/extracted/6146349/figures/Ablation_BoN.png)

- **In-domain, N=1**：System-2 略输 System-1（thought 引入幻觉路径）
- **In-domain, N=16/64**：System-2 反超（candidate diversity 弥补单样本劣势）
- **OOD (AndroidWorld), Bo1**：System-2 大幅领先 → reasoning depth 在没有训练数据覆盖的场景才真正起作用

> ❓ 这给 "thinking 一定有用" 加了重要边界：训练分布内、采样预算 1 时，加 thought 反而是负担。后续工作（DeepSeek-R1 类）大量证实 reasoning 在 OOD/hard 场景才显出来。

---
## 关联工作

### 基于
- **[[2312-CogAgent|CogAgent]]** / Qwen-2-VL：base VLM backbone，UI-TARS 在 Qwen-2-VL 上做 continual pretrain
- **ReAct**：thought-action 交替的早期范式，UI-TARS thought 机制的直接 inspiration
- **DPO (Rafailov et al.)**：preference learning 训练框架，用于 Agent DPO 阶段
- **MINT / OmniCorpus**：tutorial 数据的预训练源

### 同期 native agent model（直接对比）
- **[[2410-OSAtlas|OS-Atlas]]**：同样训练统一 GUI foundation model，UI-TARS 在 ScreenSpot-Pro 大幅超它（38.1 vs 18.9）
- **[[2411-ShowUI|ShowUI]]**：unified vision-language-action GUI agent，关注效率
- **Aguvis**：双阶段训练 GUI agent，UI-TARS 在 Mind2Web 超 Aguvis-72B
- **[[2401-SeeClick|SeeClick]]**：早期专注 GUI grounding 的 VLM，被并入 UI-TARS 训练数据
- **Claude Computer Use**：商用 framework 路线代表，UI-TARS 在 OSWorld 超它

### Framework 路线对比
- **[[2408-OmniParser|OmniParser]]**：grounding/parsing 工具，常配合 GPT-4o 使用
- **GPT-4o + Aria-UI** / **GPT-4V + SeeAct**：典型 framework 组合
- **Project Mariner (Gemini-2.0)**：Google 的商用 GUI agent

### 数据来源 / Benchmark
- 训练数据：MM-Mind2Web、GUIAct、AITW、AITZ、AndroidControl、GUI-Odyssey、AMEX、MultiUI、Rico-SCA、CLAY、UIBERT、OmniACT、AutoGUI
- Eval benchmark：VisualWebBench、WebSRC、ScreenQA-short、[[2504-ScreenSpotPro|ScreenSpot-Pro]] / ScreenSpot v1/v2、Mind2Web、AndroidControl、GUI Odyssey、[[2404-OSWorld|OSWorld]]、AndroidWorld

---
## 论文点评

### Strengths

1. **范式确立**：清晰划分 framework vs native model 两条路线，并用全面实验证明 native model 在规模化下完胜 framework，是 GUI agent 领域里程碑。后续工作（Aguvis、OS-Atlas、Operator、Claude Computer Use）的 framing 都受其影响。
2. **数据飞轮闭环完整**：online bootstrap → 多级过滤 → reflection 标注 → DPO，构成可持续的 self-improvement 管道。这套思路是 [[2410-OSAtlas|OS-Atlas]]、[[2411-ShowUI|ShowUI]] 等同期工作没做完整的部分。
3. **Post-Reflection 标注的 insight**：保留错误步、要求标注员"在错误已发生的前提下"标补救动作。这是 reflection tuning 比单纯 error correction 更有价值的一点——教模型 recover from error 而不是 avoid error。
4. **System-1 vs System-2 的边界条件**：明确指出 thought 在单样本 in-domain 下反而有害，BoN 或 OOD 才显优势。这种诚实的 negative finding 是好品味，避免了 "thinking always helps" 的过度推销。
5. **Scale ablation 翔实**：2B/7B/72B 三个尺寸 + offline/online 双轴比较，揭示 online benchmark 才能区分 model scale 的真实收益。

### Weaknesses

1. **Reasoning pattern 无 ablation**：六种 thought pattern 的边际贡献无单独验证，无法判断是不是真都需要。可能 1-2 个 pattern 就能拿到大部分收益。
2. **数据规模无法复现**：50B token、6M tutorial、数百台 VM、大量人工标注——只有大厂能做。论文未讨论 minimum viable data scale，对学术界参考价值有限。
3. **Reflection tuning 数据量未披露**：error correction 和 post-reflection 的样本量、标注成本、对最终性能的边际贡献都没说清。考虑到这是核心 claim，缺这块 ablation 很可惜。
4. **Safety 与对抗鲁棒性缺席**：computer-use 类 agent 直接接触用户文件和账号，prompt injection / 误操作 / phishing 风险明显，论文完全没讨论。
5. **System-1 在 OOD 反败的根因没追究**：只指出现象，未做 mechanistic 分析。是 thought distribution shift？还是 attention 模式变化？这块解释空间大。
6. **Mobile 数据明显短板**：自标注 trace 主要 PC，开源 mobile 数据 145k traces 但 desktop 0 — 这种不均衡可能解释为何 Claude-CU 在 mobile 弱（同样数据问题）。论文没把这个观察连起来。

### 可信评估

#### Artifact 可获取性

- **代码**：inference-only。GitHub 仓库主要包含 prompt template、deployment 工具（HuggingFace endpoint / VLLM）、coordinate visualization；不含 training pipeline、数据处理脚本、reflection tuning / DPO 训练代码。
- **模型权重**：UI-TARS-2B-SFT、UI-TARS-7B-SFT、UI-TARS-7B-DPO、UI-TARS-72B-SFT、UI-TARS-72B-DPO 已发布在 HuggingFace（以及后续的 UI-TARS-1.5）。
- **训练细节**：仅高层描述。三阶段（continual pretrain / annealing / DPO）和 50B 总 token 数公开；具体超参（lr schedule、batch size、DPO β、各阶段 step 数、数据配比）未披露。
- **数据集**：自建标注数据集（7.5M elements、avg 14.9 步 trace）**未开源**；6M GUI tutorial 数据集**未开源**；reflection tuning preference pair**未开源**。开源数据部分（MM-Mind2Web、GUIAct、AITW、AITZ、AndroidControl、GUI Odyssey、AMEX、SeeClick 等）按原始来源可获取。

#### Claim 可验证性

- ✅ **OSWorld / AndroidWorld / ScreenSpot-Pro 上 SOTA**：给出具体数字和 baseline 对比表，benchmark 公开可复现，已被多个独立工作复测确认。
- ✅ **System-2 在 OOD 优于 System-1**：Figure 8 直接对比，结论与 BoN 设置自洽，AndroidWorld 数据可复现。
- ✅ **DPO 在 OSWorld 提升 over SFT**：Table 9 明确显示 SFT vs DPO 行，差距明显。
- ⚠️ **"end-to-end model outperforms sophisticated frameworks"**：成立，但与 baseline 的训练 token 数、数据质量不可比。framework 没用 50B token 训过。比较的是 "用 50B GUI data 训 native" vs "用 commercial VLM 包 prompt"，不是同等输入下的方法对比。
- ⚠️ **"6M GUI tutorials 提升 reasoning"**：未做 ablation 隔离 tutorial 训练的贡献——无法区分是 tutorial 起作用还是 thought augmentation 起作用。
- ⚠️ **Reflection tuning 的贡献**：没单独 ablation reflection tuning 的提升幅度，只给出最终 DPO 模型分数。Reflection 数据质量（标注一致性、post-reflection 的 ground-truth recovery）也无报告。
- ❌ **"continuously learns from its mistakes ... with minimal human intervention"**：marketing 话术。论文自己说 human review 是过滤的关键步骤之一，且 instruction set 每轮 `HumanRefine`。这远不是 minimal human intervention。

### Notes

- 四阶段 GUI agent 演化框架（Rule-based → Framework → Native → Active Lifelong）成了 follow-up 工作引用 UI-TARS 时的标准 framing。
- Post-reflection 的标注思路（让模型看到自己犯过的错并学会绕回正轨）是后续 RL-based GUI agent（如 UI-TARS-1.5、各种 Agentic RL 工作）的雏形。
- "BoN 让 System-2 反超 System-1" 这个观察后来被 [[2404-OSWorld|OSWorld]] 等多个独立工作复现，是 inference-time scaling 在 agent 场景的早期证据。
- 与 ACU Survey 的六大 gap 对应：UI-TARS 在 generalization（pure-vision + unified action）、learning（iterative bootstrap）、planning（System-2）三个维度提供了具体方案，但 safety / reliability / evaluation 三个维度仍是 open。
- 7B 在 OSWorld DPO 设置下能拿到 18.7（甚至超 Claude 14.9），如果部署成本是主要考量，7B 是合理选择；但 online 上 72B 拉开 7B 较多（22.7 vs 18.7），online 任务建议直接上 72B。
- ❓ Reflection tuning 的核心数据是 model-generated trace + 人工纠错。这条路线在 RL fine-tuning 时代是否会被纯 RL（无人工纠错，靠 verifier reward）替代？UI-TARS-1.5 的方向值得追踪。

### Rating

**分数**：3 - Foundation
**理由**：Strengths 中"范式确立"与 Notes 中"四阶段演化框架成为后续工作标准 framing"共同说明 UI-TARS 在 GUI agent 方向扮演奠基角色——它不仅是 SOTA（这让它够 Frontier），更把 native agent model 作为一条清晰路线推上主流，后续 Aguvis、OS-Atlas、Operator、UI-TARS-1.5 等主要工作都以它为 baseline 或 framing 来源。相比 Frontier 档，它的 **数据飞轮闭环（online bootstrap + reflection tuning + DPO）** 和 **System-1 vs System-2 边界条件** 已被多个独立工作复现并作为 de facto 构件——这类影响力已超出"当期 SOTA"范畴。
