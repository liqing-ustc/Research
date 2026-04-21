---
title: "EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery"
authors: [Yougang Lyu, Xi Zhang, Xinhao Yi, Yuyue Zhao, Shuyu Guo, Wenxiang Hu, Jan Piotrowski, Jakub Kaliski, Jacopo Urbani, Zaiqiao Meng, Lun Zhou, Xiaohui Yan]
institutes: [Huawei Technologies, Vrije Universiteit Amsterdam]
date_publish: 2026-03-09
venue: arXiv 2603.08127
tags: [auto-research, LLM]
paper: https://arxiv.org/abs/2603.08127
website: https://evoscientist.ai/
github: https://github.com/EvoScientist/EvoScientist
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery
> - **核心**: 用三个 agent + 两块持久 memory 把 end-to-end 科学发现做成"会自我演化"的 pipeline——把每次跑出来的成功/失败方向和代码经验沉淀下来，让后续的 idea 生成和实验执行越跑越好
> - **方法**: Researcher Agent 做 idea tree search + Elo tournament，Engineer Agent 做四阶段实验 tree search，Evolution Manager Agent 用三种 evolution（Idea Direction / Idea Validation / Experiment Strategy）把 trace 蒸馏进 ideation memory $M_I$ 和 experimentation memory $M_E$；下次任务前 RAG 拼到 prompt 里
> - **结果**: 在 30-query 内部 benchmark 上对 7 个开源/商用 baseline（VirSci、AI-Researcher、InternAgent、AI Scientist-v2、Hypogenic、Novix、K-Dense）做 pairwise win，4 维度 Avg gap +29 ~ +93；code execution success rate 从 34.39 → 44.56（mean over 4 stages）；6 篇生成的论文全部被 ICAIS 2025 AI Scientist Track 接收，含 1 篇 Best Paper
> - **Sources**: [paper](https://arxiv.org/abs/2603.08127) | [website](https://evoscientist.ai/) | [github](https://github.com/EvoScientist/EvoScientist)
> - **Rating**: 2 - Frontier（multi-agent auto-research 当前前沿工作之一，external-venue eval 有说服力，但 "evolution" 本质是 prompt-RAG、baseline control 存疑，未到奠基级别）

**Key Takeaways:**
1. **从 static pipeline → 跨 task 演化**：现有 AI scientist 系统（AI Scientist-v2、AI-Researcher、InternAgent...）大多在单次 run 内做 tree search / debate / BO，但 agent 角色和决策策略 deploy 后基本不变，跨 task 的 trace 被丢弃。EvoScientist 把这些 trace 变成 first-class 资源
2. **两块 memory 解耦 idea / execution**：ideation memory $M_I$ 存 promising direction + failed direction（来自 Idea Direction Evolution 和 Idea Validation Evolution）；experimentation memory $M_E$ 存可复用的 data processing / model training strategy（来自 Experiment Strategy Evolution）
3. **Evaluation 锚定外部 venue**：6 篇论文投 ICAIS 2025 全部接收（接收率 31.71%），1 Best Paper + 1 AI Reviewer's Appraisal Award——比纯 internal benchmark 更可信，但 ICAIS 是 AI Scientist Track，本身样本量小、reviewer 池可能偏 favorable

**Teaser. EvoScientist 总览：RA + EA + EMA 三 agent 协同，EMA 从 trace 中蒸馏经验进 $M_I$ 和 $M_E$，RA/EA 在新任务时检索这两块 memory 来增强 prompt。**

![](https://arxiv.org/html/2603.08127v1/x1.png)

---

## Problem Setup 与动机

### 现有 AI Scientist 系统的盲点

作者把 end-to-end scientific discovery 文献分两线：
1. **早期 ideation**：VirSci、Co-Scientist、HypoGen、FutureGen、Spark、ResearchBench——多 agent 协作 propose / critique / refine
2. **End-to-end pipeline**：The AI Scientist、AI Scientist-v2、AI-Researcher、InternAgent、DeepScientist、OmniScientist——覆盖 ideation → 实验 → manuscript

**共同 limitation**：within-run 的探索机制（tree search, debate, BO）足够用，但 agent role 和 decision policy 是 pre-specified 的，**interaction outcome 不沉淀成跨 task 的 reusable experience**。后果：重复踩坑、漏掉 promising direction、在不可行 idea 上烧资源。

研究问题（原文 verbatim）：
> How can we formulate end-to-end scientific discovery as a learning problem in which multi-agent systems evolve their idea-generation and code-generation by learning from prior successes and failures?

### Problem Formulation

把 end-to-end scientific discovery 形式化成 goal-driven pipeline：从 user goal $G$ 到 proposal $P$ + 可执行 code $C$ + execution report $W$。
- **Stage 1 (Idea Generation)**：产生 idea $I$（method 描述 + experimental plan），扩成 proposal $P$（背景、related work、方法、实验计划、预期结果）
- **Stage 2 (Experiment Execution)**：在 $P$ 基础上搜索并跑 $C$，产出 logs/metrics 和 execution report $W$

> ❓ "学习问题" 的 formulation 比较 loose——这里没有定义 loss 或 reward signal，所谓 "learning" 实际上是 LLM-summarization-based 的经验蒸馏 + retrieval-augmented prompting，不是 gradient-based 学习。把它叫 "学习" 容易让 reader 误以为有 RL/SFT，但论文里只有 prompt-level 的记忆累积。

---

## Method

### 整体框架

三 agent：
- **Researcher Agent (RA)**：从 $M_I$ 检索方向知识 → 生成 $I$ → 扩成 $P$
- **Engineer Agent (EA)**：从 $M_E$ 检索执行经验 → 搜出 $C$ → 跑 → 产出 $W$
- **Evolution Manager Agent (EMA)**：在 task 结束后总结 trace，更新 $M_I$（promising / failed directions）和 $M_E$（reusable execution strategies）

### Researcher Agent: Idea Tree Search

**Memory retrieval**：

$$
K_I = \text{Retrieve}_I(M_I, G)
$$

embedding-based 余弦相似度，top-$k_I$（实现里 $k_I = 2$）。

**Idea Tree Search**：每个 node 存 (idea draft, review feedback)，每次 expansion 用 feedback 生成 refined child，输出：

$$
\{(I_1, \text{rev}_1), \ldots, (I_{N_I}, \text{rev}_{N_I})\} = \text{IdeaTreeSearch}(G, L, K_I)
$$

其中 $L$ 是检索到的 literature paper（用 Semantic Scholar API），$N_I = 21$。

**Tournament Selection**：用 Elo-based pairwise tournament 排序（理由：noisy judgment 下不需要 calibrated absolute score 也能稳定）：

$$
\{r_1, \ldots, r_{N_I}\} = \text{EloRank}(I_{1:N_I})
$$

留 top-3 给 direction summarization，把 top-1 扩成 proposal $P = \text{Extend}(\text{Top-1}(\cdot))$。

### Engineer Agent: Experiment Tree Search

**Memory retrieval**：$K_E = \text{Retrieve}_E(M_E, P)$，top-$k_E = 1$。

**四阶段 tree search**：$s \in \{1, 2, 3, 4\}$ 分别对应 initial implementation / hyperparameter tuning / proposed method / ablation。每阶段最大尝试数：$N_E^1 = 20, N_E^2 = 12, N_E^3 = 12, N_E^4 = 18$。

每阶段产出 (code, execution record) 序列，按 metric 选 best：

$$
C_{best}^s = \arg\max_{j} \text{Top-1}(E_j^s)
$$

最后把所有 stage 的 history 合并成 execution report $W$。

> ❓ 不同 stage 的 max attempts 拍得很死（20/12/12/18），论文没说怎么定的，也没做 sensitivity analysis。Stage 3（proposed method）只有 12 次尝试且最终 success rate 仍只有 21.57%，说明 budget 显然不够。

### Evolution Manager Agent: 三种 Self-Evolution

**1. Idea Direction Evolution (IDE)**：从 top-ranked ideas 总结 promising direction：

$$
F_I^{IDE} = \text{IDE}(G, \mathcal{I}_{\text{top}}), \quad M_I \leftarrow \text{Update}_I(M_I, F_I^{IDE})
$$

**2. Idea Validation Evolution (IVE)**：基于 execution report $W$ 判断 proposal 是否失败（rule-based：超 budget 找不到可执行 code 算失败；否则 LLM-based 比较 method vs baseline）：

$$
F_I^{IVE} = \text{IVE}(P, W), \quad M_I \leftarrow \text{Update}_I(M_I, F_I^{IVE})
$$

**3. Experiment Strategy Evolution (ESE)**：从 best code + 全 trajectory 中蒸馏 data processing strategy + model training strategy：

$$
F_E = \text{ESE}(P, \{H_E^s\}_{s=1}^4), \quad M_E \leftarrow \text{Update}_E(M_E, F_E)
$$

三个 evolution 都是 prompt LLM 来做 summarization——没有 fine-tuning。

> ❓ 失败判定的 LLM-based "method vs baseline" 比较是个潜在 bias source：用 LLM judge 来定义 ground-truth 失败信号，再把这个信号 distill 进 memory 来指导未来 idea，整个 loop 都靠 LLM judgment——容易自洽地 reinforce LLM 自己的偏好（比如喜欢 incremental 改进），偏离真实科研价值。

---

## Implementation Details

| 组件 | 选型 |
|---|---|
| Idea generation LLM | Gemini-2.5-Pro |
| Code generation LLM | Claude-4.5-Haiku |
| Manuscript writing LLM | Gemini-2.5-Pro |
| Embedding | mxbai-embed-large via Ollama |
| Literature search | Semantic Scholar API |
| Manuscript writer | 复用 AI Scientist-v2 模块（不是本工作贡献） |

---

## Experiments

### Datasets

自建 multi-level eval set（无公开 dataset 覆盖完整 pipeline）：
- **Idea Generation**：30 个 research query，AI 研究员命题，覆盖 MT、ASR、software engineering、healthcare agent、text-to-SQL、IE、RAG、multimodal arch、efficiency、data synthesis、safety/alignment 等
- **Code Generation**：上一步生成的 proposal 作为输入
- **End-to-end**：选 6 个 idea 写成完整论文投 ICAIS 2025

### Baselines

- **Open-source**: Virtual Scientist (VirSci), AI-Researcher, InternAgent, AI Scientist-v2
- **Commercial**: Hypogenic, Novix, K-Dense

### RQ1: Idea Generation Quality

Pairwise comparison（LLM judge = Gemini-3-flash + 3 PhD human annotator），4 维度：Novelty, Feasibility, Relevance, Clarity，position 双向 swap 去 bias。

**Table 1. LLM judge (Gemini-3-flash) 评估的 idea generation pairwise 结果。**

| Method | Avg. Gap |
|---|---|
| EvoScientist vs Virtual Scientist | +93.34 |
| EvoScientist vs AI-Researcher | +87.50 |
| EvoScientist vs InternAgent | +83.33 |
| EvoScientist vs AI Scientist-v2 | +29.17 |
| EvoScientist vs Hypogenic | +80.83 |
| EvoScientist vs Novix | +46.00 |
| EvoScientist vs K-Dense | +54.50 |

**Table 2. Human expert (3 PhD) 评估的 pairwise 结果（仅 strong baseline）。**

| Method | Avg. Gap |
|---|---|
| EvoScientist vs InternAgent | +84.17 |
| EvoScientist vs AI Scientist-v2 | +34.16 |
| EvoScientist vs Novix | +49.17 |
| EvoScientist vs K-Dense | +50.84 |

LLM judge 和 human judge agreement = 90.0% overall（Clarity 90.8%, Novelty 88.3%, Relevance 84.2%, Feasibility 83.3%）。

> ❓ Win rate 数字漂亮得有点过分——对 VirSci、AI-Researcher、Hypogenic、InternAgent 在 Clarity 上接近 96.67% win，几乎是 sweep。这种 magnitude 通常意味着 baseline 没有充分 tuning 或 baseline 的输出格式 / prompt template 处于劣势。论文没写 baseline 是否同 budget / 同 LLM 跑的，且 baseline 之间 gap 也没报，无法判断。

### RQ2: Code Generation Success Rate

**Figure 2.** 四阶段执行成功率，evolution 前后对比：

![](https://arxiv.org/html/2603.08127v1/x2.png)

均值 34.39 → 44.56（+10.17）。Stage 3 (proposed method) 最难，20.33 → 21.57（几乎没动）。

> 这是论文比较诚实的部分——明确说 stage 3 仍然很 challenging，"clear headroom"。

### RQ3: End-to-End ICAIS 2025

6 篇全部被 ICAIS 2025 AI Scientist Track 接收（接收率 31.71%，82 投 26 收）：

| Title | Result |
|---|---|
| Adaptive Evidential Meta-Learning with Hyper-Conditioned Priors for Calibrated ECG Personalisation | Best Paper Award |
| Hierarchical Change Signature Analysis ... Industrial Time Series | AI Reviewer's Appraisal Award |
| Robust Zero-Shot NER for Crises via Iterative Knowledge Distillation | Accepted |
| Adaptive Log Anomaly Detection ... Lifelong Learning | Accepted |
| (其余 2 篇) | Accepted |

Meta-review 的三个 pattern：
- **Strength**: methodological novelty（reviewer 反复表扬研究问题的 novelty/relevance/clarity）
- **Strength**: experimental validation 扎实（4/6 篇被赞 "comprehensive and sound experimental design"）
- **Weakness**: theoretical analysis 偏弱（"lack of deeper theoretical formalization"）——作者把这个 frame 成 "EvoScientist 负责 what，theoretical why 留给人类"

> ❓ ICAIS 2025 是新成立的 AI Scientist Track，82 submissions 是个相当小的 venue，且 reviewer 可能本身对 AI-generated 工作有 expectation gap（要求门槛偏低）。把它作为 "end-to-end 能力" 的核心证据，需要打折看。

### RQ4: Ablation

**Table 3. Ablation（all vs EvoScientist 用 Gemini-3-flash 评估）。**

| Variant | Avg. Gap |
|---|---|
| -IDE vs EvoScientist | -22.50 |
| -IVE vs EvoScientist | -20.00 |
| -all vs EvoScientist | -45.83 |

- 去掉 IDE: Novelty / Feasibility 双降
- 去掉 IVE: 主要伤 Feasibility（Lose 63.33%）—— validation evolution 的作用是过滤不可行方向
- 去掉 all: Novelty Lose 80%, Feasibility Lose 83.33%；Relevance / Clarity tie 比例高（46.67%）

结论：evolution 的核心增益在 originality + feasibility，而不是表层 relevance / clarity。

---

## 关联工作

### 基于
- **AI Scientist-v2** (Yamada et al.): 复用其 manuscript writing 模块；EvoScientist 主要在 ideation 和 execution evolution 上做差异化
- **Virtual Scientist (VirSci)**: 早期多 agent ideation 范式，被作为 baseline 和 method 比较
- **DeepAgents framework** (LangChain): 实际工程实现框架（github README 显示）

### 对比
- **AI Scientist / AI Scientist-v2**: tree search 仅在 within-run，无 cross-task memory
- **AI-Researcher**: 全 pipeline 但 static
- **InternAgent**: human-in-the-loop 增强 vs EvoScientist 的 cross-task evolution
- **DeepScientist**: 把 discovery 形式化成 sequential experimental optimization
- **Co-Scientist** (Google): generate-debate-refine for biomedical
- **Hypogenic / Novix / K-Dense**: 商用平台

### 方法相关
- **Self-evolving agents** literature: memory system、adaptive tool-use、reward-based / imitation-based / population-based learning
- **Elo-based tournament**: pairwise comparison ranking，noisy judgment 下稳定
- **Embedding retrieval (cosine)**: 标准 RAG 范式
- **mxbai-embed-large**: 嵌入模型

---

## 论文点评

### Strengths

1. **Problem framing 切中真痛点**：现有 AI scientist 系统确实存在 "每次从零开始、不积累经验" 的问题。把跨 task experience 蒸馏成 memory 是合理且重要的方向
2. **External venue evaluation**：投 ICAIS 2025 拿 award 比纯 internal benchmark 更有说服力，至少证明能出一些不是完全 trash 的论文
3. **Ablation 结构干净**：IDE / IVE / -all 三档 ablation 把两个 memory 组件的贡献分开测了
4. **Memory 组件解耦合理**：ideation 和 experimentation 分两块 memory 是 natural 的设计——idea-level 和 code-level 的经验粒度本来就不同

### Weaknesses

1. **"Evolution" 名字很大、实质是 prompt RAG**：所谓 self-evolution 本质是 LLM-summarize-then-retrieve，没有任何 weight update，把它叫 "evolution" / "learning problem" 有 marketing 嫌疑。和真正的 self-improving agent（涉及 RL / SFT loop）不在一个量级
2. **Baseline 强弱 evaluation 缺乏 control**：win rate 对一些 baseline 高达 96%+，需要质疑是否同 LLM、同 budget、同 prompt 工程下的对比。论文没写
3. **LLM judge bootstrap 风险**：失败判定 + ranking 都靠 LLM，evolution loop 也靠 LLM 总结，整个系统的 ground truth signal 都来自 LLM judgment，容易 reinforce LLM 自己的偏好
4. **泛化性靠假设**：六篇 ICAIS 论文是从生成结果中"select"的（"We select 6 research ideas"），不是 random 抽取——selection bias 显著。投稿成功率 6/6 在 cherry-pick 后并不能反映系统平均水平
5. **代码 reproducibility 模糊**：Github 仓库存在但论文里没明确说哪个 component 开源、prompt template 是否公开（v0.0.8 PyPI 包）。需要 hands-on 验证
6. **Memory 增长无管理**：ideation memory 跨 task 累积后会越来越大，retrieval 是 cosine top-$k$，没讨论 memory pruning / forgetting 机制——长期跑下去的 quality drift 是 open question

### 可信评估

#### Artifact 可获取性
- **代码**: 开源（github.com/EvoScientist/EvoScientist，PyPI 有 v0.0.8）。inference + evolution loop，training 不涉及（无 fine-tuning）
- **模型权重**: N/A（无训练）。embedding 用 mxbai-embed-large
- **训练细节**: N/A（prompt-only system）。但 prompt template 论文里有附录（Figures 4-12），实际开源仓库里是否完全 reproduce 待验证
- **数据集**: 30 query 的 idea generation eval set 论文 Appendix A 列出；ICAIS 2025 论文 link 公开（airaxiv.com）

#### Claim 可验证性
- ✅ **6/6 ICAIS 接收**：可从 ICAIS 官方和 airaxiv.com 链接核验
- ⚠️ **Pairwise win rate +29 ~ +93**：需要验证 baseline 是否在同 budget/LLM 下跑——论文未说明
- ⚠️ **Code execution success rate 34.39 → 44.56**：30 query 样本量偏小；evolution 前后是否同 task 同 random seed 跑的不明
- ⚠️ **"全部接收 + 1 Best Paper"**：6 篇是 selected 而非 random sampled，selection bias
- ❌ **"self-evolving / learning problem" 措辞**：实际是 prompt-level RAG memory，没有 gradient-based learning，标签夸大

### Notes

- **Connection to my interest**: 这篇属于 `auto-research` 方向，对 MindFlow 这种自身就是 "AI 辅助研究" 系统的项目有直接借鉴意义。两块 memory + 三种 evolution 的解耦框架可以参考——尤其是 ideation memory 区分 promising vs failed direction 这一点，比单纯 "记录所有 idea" 更有信息量
- **可借鉴的设计**: (a) idea-level 和 execution-level memory 分离；(b) 失败信号显式 distill 而非默认丢弃；(c) Elo tournament 替代 absolute scoring
- **可质疑的地方**: 把 prompt RAG 叫 "evolution" 是包装策略，但 underlying mechanism 是否真的 scale 到 100+ task / 10000+ memory entry 是 open question——cosine retrieval 的 top-$k$ 在 memory 大了之后 dilution 严重
- **Pivot 信号**: 如果要做类似工作，重点应该放在 (1) 真正 close-loop 的 RL / SFT-based evolution，(2) memory 的 lifecycle 管理（pruning, conflict resolution），而不是再叠 agent 数量
- **可能的 follow-up**: ESE 在 stage 3 (proposed method) 几乎没改善（20.33 → 21.57），说明 execution memory 对 "novel method 的实现" 帮助有限——这是 AI scientist 系统的核心难点，下一步该往这里突破而不是继续刷 idea quality

### Rating

**分数**：2 - Frontier
**理由**：在 multi-agent auto-research 方向属于当前前沿参考——problem framing（cross-task experience accumulation）抓住了 AI Scientist-v2 / AI-Researcher / InternAgent 等 static pipeline 的共同盲点，且 ICAIS 2025 6/6 接收 + Best Paper 是外部 venue 信号。但尚未到 Foundation：Weaknesses 已指出 "evolution" 本质是 prompt-level RAG、baseline control 不透明、ICAIS 样本小且有 selection bias，方法范式未被社区广泛采纳为 de facto 标准；相比 AI Scientist-v2（已作为 baseline 被多数后续工作引用）仍属 "重要参考 / 候选 baseline" 档而非奠基工作。不到 Archived 是因为它在 memory-driven auto-research 这一 subthread 里是目前最完整的工程化实现之一，做该方向必读必对比。
