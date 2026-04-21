---
title: "OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis"
authors: [Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, Ben Kao, Guohao Li, Junxian He, Yu Qiao, Zhiyong Wu]
institutes: [Shanghai AI Laboratory, The University of Hong Kong, Johns Hopkins University, Shanghai Jiao Tong University, University of Oxford, Hong Kong University of Science and Technology]
date_publish: 2024-12-27
venue: ACL 2025
tags: [gui-agent, computer-use, VLM]
paper: https://arxiv.org/abs/2412.19723
website: https://qiushisun.github.io/OS-Genesis-Home/
github: https://github.com/OS-Copilot/OS-Genesis
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
> - **核心**: 不再"先定任务再采集轨迹"，而是先在 GUI 里乱点（interaction-driven exploration），再 *reverse* 出 low-level 与 high-level instruction，最后用一个 Trajectory Reward Model (TRM) 做 graded sampling。
> - **方法**: (1) 在 Android emulator / Chrome 里 rule-based 遍历 UI 元素，收集 `<s_pre, a, s_post>` triplets；(2) GPT-4o 把每个 triplet 反推成 low-level 指令，再升格成 high-level 任务；(3) GPT-4o 重新执行 high-level 任务采集完整轨迹；(4) 1–5 分 TRM 给轨迹打分，按 reward 概率采样喂 SFT。
> - **结果**: AndroidWorld 上 Qwen2-VL-7B 从 task-driven 9.82% 升到 17.41%；WebArena overall 从 7.05% 升到 10.79%；trajectory diversity 高于 task-driven 与 self-instruct，平均与 human 数据的 SR retention > 80%。
> - **Sources**: [paper](https://arxiv.org/abs/2412.19723) | [website](https://qiushisun.github.io/OS-Genesis-Home/) | [github](https://github.com/OS-Copilot/OS-Genesis)
> - **Rating**: 2 - Frontier（reverse task synthesis 是 GUI agent 数据合成的代表性 frontier 方法，被 ACL 2025 收录，但尚未成为必引奠基工作）

**Key Takeaways:** 
1. **Reverse task synthesis 的核心 insight**：task-driven 合成数据 always 受限于人能枚举的 high-level 任务集合；从 actually executable 的 low-level transition 反推任务，可以 grounding 在环境里 actually 存在的 functionality 上，避免 "想象的任务在 UI 里 unreachable" 的失败模式。
2. **TRM > binary labeler**：用 1–5 graded reward + 概率采样代替 "丢弃所有 incomplete trajectory" 的 labeler，在 high-level 任务上的提升大于 labeler，说明 incomplete 轨迹依然带有可学的 exploration 信号。
3. **Synthetic vs human gap 收窄到 80% SR retention**——但这个 gap 衡量在 1K 量级、特定 backbone、AndroidControl 上，scale up 时是否依然成立没回答；性能在更大 scale 出现 saturation。
4. **Pipeline 的核心引擎仍是 GPT-4o**——exploration 时 input field 内容、reverse synthesis、轨迹执行、reward modeling 四个环节都依赖 GPT-4o；论文承认 open-source VLM 当前还接不上这个 pipeline。

**Teaser. Ideal trajectory 包含 high-level instruction + low-level instructions + states (screenshot + a11ytree) + actions 四要素，task-driven 方法从最顶层往下铺，OS-Genesis 反向从 action transition 往上反推任务。**

![](https://arxiv.org/html/2412.19723v3/x1.png)

---

## 1. 问题：GUI agent trajectory 怎么搞？

GUI agent 训练需要包含 high-level 指令、low-level 指令、actions、states 的完整轨迹。现有路径有两条：

- **Human collection**：人工标注完整轨迹+预定义 high-level 任务，贵且慢（AndroidControl 等就是这条）。
- **Model-based task-driven synthesis**：给 GPT-4o 起始 screenshot 和示例任务，让它先生成 high-level 指令再去环境里采集轨迹。问题：
  1. 依赖预定义的 high-level 任务列表 → 多样性受限
  2. 中间步骤错了或任务和 UI 错配 → 轨迹 incomplete / incoherent
  3. 想象的任务不一定在 UI 里 reachable → trajectories fail

> 作者的判断：task-driven 方向有结构性缺陷——人或模型 imagine 出的 high-level 任务 vs. 环境真正 afford 的功能，这两者天然 mismatch。

## 2. OS-Genesis pipeline

整体三步：interaction-driven exploration → reverse task synthesis → TRM-based sampling。

**Figure 2. Pipeline overview——human-free exploration 收集 `<s_pre, a, s_post>` triplets，反向合成 low-level 与 high-level 指令。**

![](https://arxiv.org/html/2412.19723v3/x2.png)

### 2.1 Interaction-Driven Functional Discovery

在 Android emulator 和 Chrome 里 rule-based 遍历 interactive UI 元素，action space ${a \in \{\texttt{CLICK}, \texttt{TYPE}, \texttt{SCROLL}\}}$。除了 input field 需要 GPT-4o 生成 contextually appropriate 内容外，整个 exploration 是 rule-based 的。

输出大量 triplet ${\langle s_{\text{pre}}, a, s_{\text{post}} \rangle}$（pre/post action screenshot + action）。

> ❓ Rule-based traversal 的具体策略论文里没细讲——是 BFS、随机、还是覆盖率驱动？这一步直接决定了 exploration coverage，是个 hidden hyperparameter。

### 2.2 Reverse Task Synthesis

**Step A：Triplet → low-level instruction**

GPT-4o 看 pre/post screenshot + 红框标出的交互元素 + action，反推 atomic 操作描述：

$$
f_{\text{low}}: \langle s_{\text{pre}}, a, s_{\text{post}} \rangle \xrightarrow{\mathcal{M}} \tau_{\text{low}}
$$

例：CLICK 后弹出下拉菜单 → "click the dropdown to display options"。

**Step B：Low-level → high-level instruction**

把 atomic 操作 contextualize 进 broader user intent：

$$
f_{\text{high}}: \tau_{\text{low}} \xrightarrow{\mathcal{M}} \tau_{\text{high}}
$$

例："click the dropdown to display options" → "configure application settings"。

**Step C：High-level instruction → 完整轨迹**

合成出来的 high-level 指令集合 ${\mathcal{T}}$ 重新交给 GPT-4o 在环境里执行，得到完整轨迹集合 ${\mathcal{G}}$。

### 2.3 Trajectory Reward Model (TRM)

**Figure 3. TRM 用 low-level instructions + 最后三帧 state 给整个 trajectory 打 1–5 分。**

![](https://arxiv.org/html/2412.19723v3/x3.png)

打分维度：
- **Completion**：是否完成 instructed task（完整性 + interaction handling 是否得当）
- **Coherence**：action sequence 是否 logical、有无 redundant/irrelevant step

采样规则：

$$
P(g_i) = R_i \,\Big/\, \sum_{k=1}^{N} R_k
$$

按概率采样喂训练，**不丢弃任何轨迹**——这与传统 labeler-based filtering（只保留 complete trajectory）的关键区别。

### 2.4 训练目标

两路 SFT loss：

- **Planning Training**：给 high-level $h_i$ 和历史 $c$，预测 low-level $\ell$ 和 action $a$

$$
\mathcal{L}_1 = -\sum_{t_i \in \mathcal{T}} \log\Big(p_\theta(\ell \mid s, h_i, c) \cdot p_\theta(a \mid s, h_i, c, \ell)\Big)
$$

- **Action Training**：给 low-level $\ell$，预测 action $a$

$$
\mathcal{L}_2 = -\sum_{t_i \in \mathcal{T}} \log p_\theta(a \mid s, c, \ell)
$$

输出是 ReAct-style 带 thought trace。Task-Driven baseline 与 OS-Genesis 都用 1K 轨迹（self-instruct 用 1.5K），平均 6.4 步。

## 3. 实验

### 3.1 Setting

- **Backbone**：InternVL2-4B/8B（无 GUI 数据预训练）+ Qwen2-VL-7B-Instruct（claim 自带 agentic 能力）
- **训练**：8×A100 80GB 全参 SFT
- **Benchmark**：
  - AndroidControl（offline, 833 apps，OS-Genesis 只 cover 20 个 → OOD 测试）
  - AndroidWorld（online, Pixel 6 emulator, 116 tasks / 20 apps，112 实际可用）
  - [[2307-WebArena|WebArena]]（241 task templates, online, EC2 hosting）

### 3.2 Mobile 主结果

**Table 1. AndroidWorld + AndroidControl。SR = success rate, Type = action type exact match。**

| Base Model | Strategy | AndroidWorld SR | AC-High SR | AC-High Type | AC-Low SR | AC-Low Type |
|---|---|---|---|---|---|---|
| GPT-4o | Zero-Shot (M3A) | 23.70 | 53.04 | 69.14 | 69.59 | 80.27 |
| InternVL2-4B | Zero-Shot | 0.00 | 16.62 | 39.96 | 33.69 | 60.65 |
| InternVL2-4B | Task-Driven | 4.02 | 27.37 | 47.08 | 66.48 | 90.37 |
| InternVL2-4B | + Self Instruct | 7.14 | 24.95 | 44.27 | 66.70 | 90.79 |
| InternVL2-4B | **OS-Genesis** | **15.18** | **33.39** | **56.20** | **73.38** | **91.32** |
| InternVL2-8B | Zero-Shot | 2.23 | 17.89 | 38.22 | 47.69 | 66.67 |
| InternVL2-8B | Task-Driven | 4.46 | 23.79 | 43.94 | 64.43 | 89.83 |
| InternVL2-8B | + Self Instruct | 5.36 | 23.43 | 44.43 | 64.69 | 89.85 |
| InternVL2-8B | **OS-Genesis** | **16.96** | **35.77** | **64.57** | **71.37** | **91.27** |
| Qwen2-VL-7B | Zero-Shot | 0.89 | 28.92 | 61.39 | 46.37 | 72.78 |
| Qwen2-VL-7B | Task-Driven | 6.25 | 38.84 | 58.08 | 71.33 | 88.71 |
| Qwen2-VL-7B | + Self Instruct | 9.82 | 39.36 | 58.28 | 71.51 | 89.73 |
| Qwen2-VL-7B | **OS-Genesis** | **17.41** | **44.54** | **66.15** | **74.17** | **90.72** |

观察：
- AndroidWorld 上 OS-Genesis ≈ 2× task-driven，缩小与 GPT-4o M3A (23.70) 的 gap
- self-instruct 用 1.5× 数据仍打不过 OS-Genesis → 不是 data quantity 问题
- AndroidControl OOD 上 OS-Genesis 在 high-level 提升尤其明显——验证了 exploration-first 任务比 imagined 任务更 logically coherent

### 3.3 Web 主结果

**Table 2. WebArena 各 site SR（仅列 OS-Genesis 配置和 GPT-4o zero-shot 对比，完整对照见 paper）。**

| Base | Strategy | Shop | CMS | Reddit | Gitlab | Maps | Overall |
|---|---|---|---|---|---|---|---|
| GPT-4o | Zero-Shot | 14.28 | 21.05 | 6.25 | 14.29 | 20.00 | 16.25 |
| InternVL2-4B | OS-Genesis | 10.71 | 7.02 | 3.13 | 7.94 | 7.50 | **7.88** |
| InternVL2-8B | OS-Genesis | 7.14 | 15.79 | 9.34 | 6.35 | 10.00 | **9.96** |
| Qwen2-VL-7B | OS-Genesis | 7.14 | 8.77 | 15.63 | 15.87 | 5.00 | **10.79** |

InternVL2 在 zero-shot 下 0% (输出格式都不对)，OS-Genesis 训练后能 functional。Qwen2-VL-7B 已经预训过 GUI 数据，OS-Genesis 仍能拉到 10.79 (vs. zero-shot 7.47, task-driven 7.05)。

> ❓ Reddit / Gitlab 上 Qwen2-VL-7B 显著高于 GPT-4o (15.63 vs 6.25, 15.87 vs 14.29)，但 Shopping / Maps 上明显低 (7.14 vs 14.28, 5.00 vs 20.00)。这个 site-specific 不一致性论文没细分析——可能与 reverse synthesis 在哪些 UI 类型上 coverage 更好相关。

## 4. Analysis

### 4.1 Diversity

**Figure 4. Sentence-BERT 余弦距离衡量 instruction 与 trajectory diversity。**

![](https://arxiv.org/html/2412.19723v3/x4.png)

OS-Genesis 在 instruction 与 trajectory 两个维度的 diversity 都最高。**有趣观察**：human data instruction 多样但 trajectory 不多样——人想得开但执行时偏好熟悉路径；OS-Genesis 在两个维度都高，因为 exploration-driven 不带人类的 motor habit。

### 4.2 TRM Ablation

**Figure 5. 三种策略：no RM / labeler / TRM。**

![](https://arxiv.org/html/2412.19723v3/x5.png)

- TRM > labeler > no RM，特别在 AndroidControl-High 和 AndroidWorld 这种 high-level 任务上
- Labeler（丢弃 incomplete）在 high-level 略涨但在 low-level 反而掉——验证了 "incomplete 轨迹的 step-level 信号有用"
- Low-level 任务上三种策略差异不大，因为 OS-Genesis 单步本身质量就高

### 4.3 Scaling

**Figure 6. AndroidWorld SR vs trajectory 数量。**

![](https://arxiv.org/html/2412.19723v3/x6.png)

数据量增加 → SR 上升，但在 main exp scale 之外开始 saturate。作者归因于 (1) VLM 容量，(2) GPT-4o materialize high-level 任务的能力上限。

> ❓ Saturation 也可能源于 reverse synthesis 自身的 coverage 上限——rule-based exploration 探不到的 functionality 就生不出对应任务。这点论文没区分。

### 4.4 vs Human data

- **High-level instructions 来源对比**：拿 AndroidControl 中的 500 个人写 high-level 任务给 GPT-4o 去执行采集轨迹，对比 OS-Genesis 自己合成的 instructions。结果：用人写指令训练的 agent **打不过** OS-Genesis 自合成指令训练的——作者解释为 (1) 人指令与 dynamic 环境 mismatch，(2) 模型解释人意图时引入 error。
- **完整轨迹对比**：1K human 轨迹 vs 1K OS-Genesis。OS-Genesis 平均 SR retention > 80%，high-level 上 gap 最小。

> ❓ "人写指令 < 合成指令" 这个 finding 是反直觉的——更可能的解释是：AndroidControl 中的人指令是为它自己的 task 设计的，到 GPT-4o 执行时存在 instruction-environment mismatch；并不能 generalize 成 "合成指令本质上比人指令好"。

## 5. Action Space & 实现细节

**Mobile actions** (10 种)：click / long_press / type / scroll / navigate_home / navigate_back / open_app / wait / terminate / keyboard_enter

**Web actions** (11 种)：click[id] / type[id][content] / hover[id] / press[key_comb] / scroll / new_tab / tab_focus / close_tab / goto[url] / go_back / go_forward

**Backbone 配置**：
- InternVL2-{4B,8B}: max_dynamic_patch=24（448×448 tile + thumbnail）
- Qwen2-VL-7B: image_resolution=1024
- a11ytree 过滤后只保留 visible 元素的 position/index

## 6. Limitations (作者自陈)

- **Proprietary dependency**: GPT-4o 是 exploration / synthesis / reward / 执行的核心引擎，open-source VLM 当下接不上
- **Modality**: 只用 text+visual 联合训练评估，partial modality 留待 future work
- **Trajectory 数量受限于 GPT-4o 的执行能力**——更强的 task-executing model 能解锁更大数据量

---

## 关联工作

### 基于
- **GPT-4o**：Pipeline 中 exploration / synthesis / reward / 任务执行四处依赖
- **InternVL2 / Qwen2-VL-7B**：训练 backbone
- **a11ytree**：状态文本表示，与 screenshot 配合作为 multimodal input

### 对比
- **Task-Driven baseline (NNetNav 等)**：从初始 screenshot + 示例任务用 GPT-4o 生 high-level 指令再执行——OS-Genesis 反向走
- **Self-Instruct**：从 task-driven 数据采样 ICL example 扩充指令——OS-Genesis 用 1× 数据打过其 1.5×
- **Labeler-based filtering**：传统只保留 complete trajectory 的二值过滤——TRM 用 graded reward 替代

### 方法相关
- **AndroidControl**：mobile control benchmark + human trajectories，作为 OOD 测试和 human data 对照
- **AndroidWorld**：online Android benchmark，主战场
- [[2307-WebArena|WebArena]]：online web benchmark，第二战场
- [[2410-OSAtlas|OS-Atlas]]：作者后续 / 关联工作，GUI foundation model 路线
- **MiniWob / Rico**：早期 GUI 数据集
- **M3A agent**：AndroidWorld 上的 GPT-4o baseline

---

## 论文点评

### Strengths

1. **Insight 干净**：把 task synthesis 反过来做这个 idea 很 clean——绑定到环境真正 afford 的功能上，结构性地避免了 task-driven 的 "imagined task unreachable" 失败模式。
2. **TRM > binary labeler 的 ablation 有教育意义**：证明 incomplete 轨迹的 step-level 信号有 retain value，这个发现对所有用 labeler filter 数据的 pipeline 都有借鉴。
3. **Diversity 分析切到了 human vs synthetic 的有意思 asymmetry**：人 instruction 多样而 trajectory 不多样的观察，暗示 "imitation learning from human" 在 trajectory diversity 上有 ceiling，合成数据未必劣于人数据。
4. **多 backbone × 多 benchmark 比较扎实**：3 个 backbone × 3 个 benchmark，task-driven / self-instruct / OS-Genesis 三种合成策略对照清晰。

### Weaknesses

1. **Pipeline 的核心引擎全是 GPT-4o**：exploration 中 input field 生成、low/high-level synthesis、轨迹执行、TRM 评分——四个关键节点都靠闭源大模型。论文 claim "without human supervision" 但等价于 "with GPT-4o supervision"，data quality ceiling 直接绑死在 GPT-4o 上。
2. **Rule-based exploration 策略不透明**：traversal 的具体策略（BFS / 随机 / 覆盖驱动？）、exploration 的终止条件、triplet 的去重，这些都是 hidden hyperparameter，影响 reproducibility 和 coverage 上限分析。
3. **"合成 high-level 指令打过人写指令" 的归因可疑**：作者归因于 instruction-environment alignment，但更可能是 AndroidControl 的人指令是为 AndroidControl task 设计、迁移到自由执行场景时 distribution shift。要验证 claim，需要 controlled experiment（同一环境下让人重新写 instruction）。
4. **Saturation 归因不充分**：归到 VLM 容量 + GPT-4o 能力，但更可能的瓶颈是 rule-based exploration 的 coverage 上限——这点没切开看。
5. **缺乏与 newer GUI agent 数据合成方法的对比**：比如 [[2410-OSAtlas|OS-Atlas]] 的 GUI grounding pretraining 路线、AGUVIS、Aria-UI 等同期工作没纳入比较，task-driven baseline 自己搭的版本可能弱于这些 SOTA。

### 可信评估

#### Artifact 可获取性

- **代码**: 开源 (https://github.com/OS-Copilot/OS-Genesis)，包含 collection scripts 与 evaluation 代码；training 部分指向 InternVL2 / Qwen2-VL 上游 repo，未提供独立训练 launcher
- **模型权重**: 已发布 9 个 checkpoint：OS-Genesis-{4B,7B,8B}-{AC,AW,WA}（AndroidControl / AndroidWorld / WebArena 三个版本 × 三个 backbone），全部在 HuggingFace OS-Copilot org 下
- **训练细节**: 仅高层描述（8×A100 80GB 全参 SFT, max_dynamic_patch=24, image_resolution=1024）；具体 lr / batch / epoch / data 配比未在正文披露，需查 GitHub config
- **数据集**: 开源——HuggingFace 上有 `OS-Genesis-mobile-data` 和 `OS-Genesis-web-data`（jsonl 格式 training data），Google Drive 上额外提供 raw `<s_pre, a, s_post>` triples + screenshots 供复现 reverse synthesis

#### Claim 可验证性

- ✅ **AndroidWorld Qwen2-VL-7B 17.41% vs task-driven 9.82%**：Table 1 直接报告，benchmark 公开，checkpoint 公开 → 可独立复现
- ✅ **WebArena Qwen2-VL-7B overall 10.79%**：Table 2 + 公开 checkpoint，可在 EC2 上复现
- ✅ **TRM > labeler > no RM**：Figure 5 ablation，sampling algorithm 1 描述完整，可复现
- ⚠️ **"OS-Genesis 合成指令优于人写指令"**：Figure 7 实验只用 500 条 AndroidControl 人指令，且让 GPT-4o 重新执行——可能是 instruction-environment distribution shift 而非合成 vs 人写本身的差异，归因不严
- ⚠️ **"Trajectory diversity 高于 human"**：Figure 4 用 Sentence-BERT 余弦距离作为 diversity proxy，但 low-level action 序列的 semantic embedding 距离是否真的反映 behavioral diversity 存疑（可能放大表层文字差异）
- ⚠️ **"Performance retention > 80% vs human"**：80% 衡量在 1K trajectory + 特定 backbone + AndroidControl 上，scale up 后是否依然成立未验证；且 "human gold standard" 本身在 AndroidControl 上有 ceiling
- ⚠️ **"Reverse task synthesis 自然桥接抽象指令与 dynamic GUI"**：作为方法 motivation 的 claim，论文未提供消融——比如对比 "用同样的 GPT-4o 直接生成 low+high-level 指令" 是否有同等效果

### Notes

- **核心 take**：interaction-driven exploration → reverse synthesis 是个有 structural argument 的 idea，不只是工程技巧。比起 task-driven 的 imagined task，反向合成强制 grounding 在 actually executable 的状态转移上，避免了一大类 silent failure。
- **对自己的 implication**：如果做 GUI/computer-use agent 的数据合成，应该把这个 reverse 的 mental model 加进 toolbox。但 pipeline 的 GPT-4o-heavy 性质决定了它本质上是 GPT-4o → smaller VLM 的 distillation pipeline——claim "without human supervision" 但 supervision 转移到了 frontier model 上。
- **与 RL 的连接**：TRM 的 graded score + 概率采样在结构上接近 weighted behavior cloning / Decision Transformer 的 return-conditioned sampling。一个自然的 next step 是把 TRM 直接变成 RL 信号（而不只是 SFT sampling weight），但作者没走这一步。
- **可批评的 framing**：论文反复强调 "without human supervision"，但 pipeline 实际是 "GPT-4o supervision + open-source training"，supervision cost 没消失，只是从 annotator 转移到 API 调用。这个 framing 在 GUI agent 圈是常见话术，但要诚实评估的话——OS-Genesis 的 contribution 是 *automation* 而非 *supervision-free*。
- **下一步可问的问题**：
  1. Reverse synthesis 能不能 bootstrap 自己？即用 OS-Genesis 训出的 7B agent 替代 GPT-4o 做新一轮 exploration / synthesis，迭代提升？
  2. TRM 能否替换为 verifier-based reward（在线验证轨迹是否真正完成 task）而非 GPT-4o 主观打分？
  3. Coverage saturation 是不是 rule-based traversal 的根本瓶颈？换成 curiosity-driven exploration 能不能解锁更多 functionality？

### Rating

**分数**：2 - Frontier
**理由**：Reverse task synthesis 是 GUI agent 数据合成的代表性 frontier 方法，被 ACL 2025 收录，HuggingFace 上开源了 9 个 checkpoint + 两个数据集，已在 GUI agent 社区作为 data synthesis 的 baseline 被引用对比。但方法 pipeline 深度依赖 GPT-4o 且 rule-based exploration 有结构性 coverage 上限（见 Weaknesses 1、2、4），未成为像 AndroidWorld / WebArena 那样的 de facto 标准，也还没在后续工作中沉淀为必引的奠基技术——所以是 Frontier 而非 Foundation；又明显高于 Archived，因为 idea 本身仍被当前 GUI agent 数据合成方向继续追随和扩展。
