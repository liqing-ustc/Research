---
title: "Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents"
authors: [Bowen Ye, Rang Li, Qibin Yang, Yuanxin Liu, Linli Yao, Hanglong Lv, Zhihui Xie, Chenxin An, Lei Li, Lingpeng Kong, Qi Liu, Zhifang Sui, Tong Yang]
institutes: [Peking University, The University of Hong Kong]
date_publish: 2026-04
venue: arXiv
tags: [computer-use, agentic-RL, LLM]
paper: https://arxiv.org/abs/2604.06132
website: https://claw-eval.github.io
github: https://github.com/claw-eval/claw-eval
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents
> - **核心**: 一个端到端 agent 评测套件，把 trajectory-level 审计、Completion/Safety/Robustness 三维打分、跨模态任务覆盖统一在一个 pipeline 里，证明 output-only 评测会系统性漏掉 44% 的 safety violation。
> - **方法**: Docker 沙箱内三段式生命周期（Setup / Execution / Judge）+ 三路独立证据（execution trace、service audit log、environment snapshot）+ 2,159 条细粒度 rubric + Pass^k 多次试验稳定性度量；safety 作为乘性 gate，robustness 通过 mock-service 错误注入主动扰动测出。
> - **结果**: 14 个 frontier 模型在 300 task 上跑，最强（Claude Opus 4.6）overall Pass^3 仅 70.4%；error injection 把 Pass@3 几乎不动但把 Pass^3 砍掉 24pp；多轮对话里 question precision 解释 76% 的 Pass^3 方差，轮数只解释 <1%。
> - **Sources**: [paper](https://arxiv.org/abs/2604.06132) | [website](https://claw-eval.github.io) | [github](https://github.com/claw-eval/claw-eval)
> - **Rating**: 2 - Frontier（设计选择（Pass^k、multiplicative safety gate、三路证据）精炼、实证扎实，但作为 2026 年 4 月刚出的 benchmark，社区采纳度待观察，尚未到 de facto 标准级别）

**Key Takeaways:**
1. **Output-only 判分不可信**：vanilla LLM judge 即便拿到完整 transcript + grader 源码，仍漏 44% safety violation、13% robustness failure——deterministic check 必须存在，不是 nice-to-have。
2. **Pass^k 是必需指标**：Pass@k 在扰动下几乎不掉，掩盖了 deployment-grade reliability 的崩塌；同一模型 Pass@3 vs Pass^3 之间的 gap 才是真正的"可部署度"。
3. **Capability ≠ Consistency ≠ Resilience**：模型在 General / Multimodal / Multi-turn 三组上排名洗牌，Code / Doc / Video 三个 multimodal 子域各有不同冠军——单一聚合数字会抹平结构性差距。
4. **Safety 必须 in-task**：把安全约束嵌入 normal workflow 任务里，作为 multiplicative gate（违规则乘 0），而非剥离成单独红队 suite——只有在"完成压力"下才能测出真正的策略遵守度。

**Teaser. Claw-Eval 整体架构：三时间相（Setup / Execution / Judge）× 三空间层（Host / Isolation / Mock Services），temporal firewall 切开 execution 与 grading，三路独立证据汇入 grader。**

![](https://arxiv.org/html/2604.06132v1/x1.png)

---

## 1. 动机：现有 agent benchmark 的三个 gap

论文把现有 agent 评测的问题归为三个 gap，下面这张表（Table 1）是六个评测维度的横向对比，可以看到没有任何已有 benchmark 同时支持 Multimodal / Multi-turn / Auditable / Safety / Perturbation / Sandboxed 六项：

**Table 1. 评测能力六维对比（✓ 全支持，❖ 部分，✗ 缺失）。**

| Benchmark           | Multimodal | Multi-turn | Auditable | Safety | Perturbation | Sandboxed |
| ------------------- | :--------: | :--------: | :-------: | :----: | :----------: | :-------: |
| AgentBench          |     ✗      |     ✓      |     ❖     |   ✗    |      ✗       |     ✓     |
| GAIA                |     ❖      |     ✗      |     ✗     |   ✗    |      ✗       |     ✓     |
| τ-bench             |     ✗      |     ✓      |     ✓     |   ✓    |      ✗       |     ✗     |
| SWE-bench           |     ✗      |     ✗      |     ❖     |   ✗    |      ✗       |     ✓     |
| [[2307-WebArena\|WebArena]] |     ❖      |     ✗      |     ❖     |   ✗    |      ✗       |     ✓     |
| VisualWebArena      |     ✓      |     ✗      |     ❖     |   ✗    |      ✗       |     ✓     |
| [[2404-OSWorld\|OSWorld]]   |     ✓      |     ✗      |     ✓     |   ✗    |      ✗       |     ✓     |
| ToolBench           |     ✗      |     ✗      |     ❖     |   ✗    |      ✗       |     ✗     |
| Terminal-Bench      |     ✗      |     ✗      |     ✓     |   ✗    |      ✗       |     ✓     |
| PinchBench          |     ✗      |     ✗      |     ✗     |   ✗    |      ✗       |     ✓     |
| **Claw-Eval (Ours)**|     ✓      |     ✓      |     ✓     |   ✓    |      ✓       |     ✓     |

三个 gap 对应三个设计原则：
- **G1 Trajectory-opaque grading** → 全轨迹审计：三路证据，agent 看不见。
- **G2 Underspecified safety / robustness** → 把 safety 嵌入正常任务，把 robustness 通过受控错误注入主动测出。
- **G3 Modally narrow coverage** → 一套统一 declarative task schema 覆盖 9 个细分类别。

> ❓ 论文把 [[2411-RewardHacking|reward hacking]] 列为 G1 的核心动机，但只引用了两篇前作，没有展示 Claw-Eval 自己捕捉到的 hacking case study。如果要真正证明 trajectory-opaque 评测是 hackable，应该有 head-to-head case：同一 agent 在 output-only 评测下 pass、在轨迹审计下 fail 的具体行为序列。第 5.1 节给的 44% 漏检率是必要不充分证据。

---

## 2. Auditable Execution Pipeline（§3.1）

每一次评测在隔离 Docker 容器里走三段：

- **Setup**：注入 workspace 文件 + 启动 mock services（CRM、邮件、日历、知识库等），mock service 从启动起就静默记录 audit log。**容器里此时不存在任何 grader / 参考答案 / 验证脚本**——这是 temporal firewall。
- **Execution**：agent 通过两层 tool 接口工作。System layer 提供 11 个内置工具，Service layer 暴露 task-specific 的 mock API。完整 agentic context 写入 execution trace（在沙箱外，agent 看不见）。
- **Judge**：agent 终止后，grading artifact 才被注入容器执行 post-hoc 命令（render 网页、跑校验、收集 artifact），形成 environment snapshot。最终 grader 拿到三路独立证据：trace（agent 说了什么）、audit log（service 收到了什么）、snapshot（环境最终是什么）。

**Table 2. Agent capability layers（System layer 11 个内置工具 + Service layer 任务特定 mock API）。**

| Functional Group | Tools | Purpose |
| --- | --- | --- |
| _System Layer_ | | |
| Code Execution | Bash | Execute shell commands |
| File Operations | Read, Write, Edit | Read, create, and modify files |
| Codebase Search | Glob, Grep | Find files by pattern; search content by regex |
| Web Interaction | BrowserScreenshot, WebSearch, WebFetch | Capture screenshots; search and fetch web pages |
| Multimodal Media | ReadMedia, Download | Process video/image/PDF; download files |
| _Service Layer_ | | |
| Task-specific APIs | Custom tools per task | Interact with mock services |

> 这个工具集明显参照了 Claude Code / OpenClaw 类的真实 agent 脚手架——Bash / Read / Write / Edit / Glob / Grep 几乎是 1:1 的命名。这意味着 evaluation 与现实部署 agent 的 gap 比传统 benchmark（如 ToolBench 的扁平 API list）小很多。

**核心设计 insight**：三路证据是 **independent** 的——audit log 是 service 自己记的（agent 改不了），snapshot 是 agent 终止后才采集的（agent 影响不了），trace 是 agent 自身的输出。任何单一证据被 hack 都不能伪造完整的 ground truth。

---

## 3. Cross-Modal Task Suite（§3.2）

300 task 分三组 9 类。这套 schema 的关键是 **declarative task definition + 与领域无关的 pipeline**——加新领域只需写 task 定义和 grader，框架本身不动。

**Table 3. Benchmark 组成。**

| Group | Category | Description | # |
| --- | --- | --- | --: |
| **General (161)** | Easy | Single-service queries, basic scheduling | 71 |
| | Medium | Cross-service coordination, data retrieval | 47 |
| | Hard | Multi-system orchestration, financial compliance, ops | 43 |
| **Multimodal (101)** | Video | Simple QA, video localization | 53 |
| | Doc & Image | Chart interpretation, cross-page reasoning | 22 |
| | Code | Webpage generation, SVG animation, video editing | 26 |
| **Multi-turn Dialogue (38)** | STEM | Data analysis, scientific reasoning | 10 |
| | Social Science | Law, education, public policy | 13 |
| | Business | Finance, investment, corporate strategy | 15 |
| | Total | | **300** |

- **General**：覆盖 service orchestration（CRM / 邮件 / 调度协同）和 standalone 分析（财务合规、文档分析、代码 debug、服务器诊断）。其中 43 个任务嵌入 safety 约束（如 "triage-only 任务里不准发邮件"）。
- **Multimodal**：要求 agent 走 perceive–reason–act 闭环——自己决定看哪段视频、抽几帧、看哪页文档。Code 类还要产出 dynamic webpage、SVG 动画、edited video clip。
- **Multi-turn Dialogue**：simulated user 配 persona + 隐藏 latent intent + information-revealing strategy。agent 必须主动追问才能拿到关键信息。

---

## 4. Scoring Protocol（§3.3）

### 4.1 三维聚合公式

$$
\text{score} = s_{\text{safety}} \times \bigl(\alpha \cdot s_{\text{completion}} + \beta \cdot s_{\text{robustness}}\bigr)
$$

**符号说明**：α + β = 1，论文取 α = 0.8、β = 0.2。
**含义**：safety 是 **multiplicative gate**——一次违规直接把整个 task score 拉到 0；completion 与 robustness 是加权和，其中 completion 主导。

### 4.2 Robustness 的定义

$$
s_{\text{robustness}} = \begin{cases} \dfrac{|\mathcal{T}_{\text{recovered}}|}{|\mathcal{T}_{\text{errored}}|} & \text{if } |\mathcal{T}_{\text{errored}}| > 0 \\[6pt] 1 & \text{otherwise} \end{cases}
$$

**符号说明**：$\mathcal{T}_{\text{errored}}$ 是至少遇到过一次注入错误的 tool 类型集合，$\mathcal{T}_{\text{recovered}}$ 是其中 agent 之后成功调用过的子集。
**含义**：测的是"恢复策略的覆盖度"，不是"重试次数"——agent 把同一个坏调用重试 10 次只能说明它执着，不算 robust。

### 4.3 Rubric 与多 metric

300 task 拆成 2,159 条 rubric item（平均 7.2/task）。每条要么是 deterministic check（文件存在、API 参数匹配、audit log 里没出现禁用动作），要么是 LLM judge 评分（文本质量、视觉保真度）。每条 rubric item 都把支持它判定的 raw artifact 留档，形成 end-to-end audit trail。

每个 task 跑 k=3 次独立 trial，报告三个互补 metric：

$$
\text{Score} = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{k}\sum_{j=1}^{k} s_{ij}
$$

$$
\text{Pass@}k = \frac{1}{N}\sum_{i=1}^{N} \mathds{1}\left[\max_{j=1}^{k} s_{ij} \geq \tau\right]
$$

$$
\text{Pass}^{k} = \frac{1}{N}\sum_{i=1}^{N} \mathds{1}\left[\min_{j=1}^{k} s_{ij} \geq \tau\right]
$$

**含义**：Score 是平均能力，Pass@k 是能力天花板，Pass^k 是可靠性地板。Pass^k 与 Pass@k 的差距 = 可部署度的真实 gap。论文用 τ = 0.75。

---

## 5. 主结果（§4）

14 个 frontier 模型在 General + Multi-turn 上的主表（Multimodal 只跑 9 个支持视觉的模型）：

**Table 4. Main results（%；Pass^3 排序，并列时按 Pass@3）。**

| Model | General Score | General Pass@3 | General Pass^3 | Multi-turn Score | Multi-turn Pass@3 | Multi-turn Pass^3 | Overall Score | Overall Pass@3 | Overall Pass^3 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Claude Opus 4.6 | 80.6 | 80.8 | **70.8** | 79.6 | 89.5 | **68.4** | 80.4 | 82.4 | **70.4** |
| Claude Sonnet 4.6 | **81.3** | 81.4 | 68.3 | **81.9** | 89.5 | 65.8 | **81.4** | 82.9 | 67.8 |
| GPT 5.4 | 78.3 | 75.8 | 60.2 | 79.0 | 89.5 | 60.5 | 78.4 | 78.4 | 60.3 |
| Gemini 3.1 Pro | 76.6 | 80.8 | 55.9 | 80.2 | **92.1** | 65.8 | 77.3 | 82.9 | 57.8 |
| MiMo V2 Pro | 76.0 | 72.7 | 57.1 | 81.0 | 92.1 | 60.5 | 77.0 | 76.4 | 57.8 |
| Qwen 3.5 397A17B | 73.8 | 70.8 | 57.8 | 75.6 | 76.3 | 52.6 | 74.2 | 71.9 | 56.8 |
| GLM 5 Turbo | 73.8 | 73.9 | 57.1 | 77.2 | 84.2 | 50.0 | 74.4 | 75.9 | 55.8 |
| GLM 5V Turbo | 73.2 | 73.3 | 52.8 | 77.4 | 86.8 | 57.9 | 74.0 | 75.9 | 53.8 |
| Gemini 3 Flash | 71.0 | 67.1 | 48.4 | 77.5 | 84.2 | 52.6 | 72.3 | 70.4 | 49.2 |
| MiniMax M2.7 | 71.8 | 72.0 | 49.7 | 75.9 | 84.2 | 44.7 | 72.6 | 74.4 | 48.7 |
| MiMo V2 Omni | 74.1 | 75.2 | 52.2 | 65.4 | 63.2 | 15.8 | 72.4 | 72.9 | 45.2 |
| DeepSeek V3.2 | 68.3 | 71.4 | 42.2 | 64.0 | 60.5 | 31.6 | 67.5 | 69.3 | 40.2 |
| Kimi K2.5 | 66.6 | 67.1 | 36.6 | 75.4 | 76.3 | 39.5 | 68.3 | 68.8 | 37.2 |
| Nemotron 3 Super | 41.7 | 34.8 | 6.8 | 56.2 | 13.2 | 0.0 | 44.4 | 30.7 | 5.5 |

三个观察：
1. **Score 冠军 ≠ Pass^3 冠军**：Sonnet 4.6 拿 Score 第一，Opus 4.6 拿 Pass^3 第一——优化平均质量不等于优化可靠执行。
2. **任务组之间正交**：Gemini 3.1 Pro 在 Multi-turn Pass^3 第二（65.8%），在 General Pass^3 只排第七（55.9%）。
3. **没饱和**：最强模型 overall Pass^3 仅 70.4%，中段 5 个模型挤在 5pp 内（55.8–60.3%）——区分度足够。

**Figure 2. General 任务按难度分层的 Pass^3——所有模型从 Easy 到 Hard 单调下降。**

![](https://arxiv.org/html/2604.06132v1/x2.png)

Easy 上 Pass^3 跨度从 14% 到 75%（6 个模型间），Opus 4.6 在 Hard 上仍保 65.1%——既不是被 saturate，也不是不可解。

---

## 6. 关键分析（§5）

### 6.1 Trajectory-opaque judge 漏 44% safety violation

实验设置：vanilla judge = Gemini-3-Flash，给它**完整 conversation transcript（含每个 tool call）+ 完整 grader 源码**，只扣掉 audit log 和 environment snapshot。在 5 个模型 × 2,000+ trace 上跑：

**Figure 3(a). Hybrid pipeline 检出的 safety violation 中被 vanilla judge 漏掉的份额。**

![](https://arxiv.org/html/2604.06132v1/x3.png)

- **Safety**：27 个 task 级违规中漏 12 个（44% miss rate）。Hybrid grader 通过对 tool-call 参数做 deterministic substring matching 抓到，LLM 读了同一段代码却"无法在脑内可靠执行 substring 匹配"，有时候还会**rationalize agent 的违规行为**而不是机械应用规则。
- **Robustness**：118 个 task 级问题中漏 15 个（13% miss rate）。Robustness 漏检率较低是因为错误码会显式出现在对话里，而 safety 违规需要 parameter-level 检查。

> 这个实验设计的关键在于"信息平等"——vanilla judge 拿到的输入比 hybrid 还多（transcript + 源码），仍然漏检。这就排除了"差距是因为 LLM 没看到证据"的解释，把矛头指向 LLM judge 本身的 reliability 问题。但实验只跑了 Gemini-3-Flash 一个 judge，没测 Opus / GPT-5.4 这种更强的 judge——也许更强的 judge 漏检率更低。这是论文的一个潜在弱点。

### 6.2 错误注入：Pass^3 比 Pass@3 脆弱得多

3 个模型 × error rate ∈ {0.0, 0.2, 0.4, 0.6}。注入错误从三类里随机抽：HTTP 429（35%）、HTTP 500（35%）、2–4s latency spike（30%）。

**Figure 4(a). 错误注入率从 0 到 0.6 时 Pass@3（实线）vs Pass^3（虚线）。**

![](https://arxiv.org/html/2604.06132v1/x5.png)

- **Pass@3 几乎不动**：Opus 4.6 从 0→0.6 只掉 3.7%，GLM 5 Turbo 反而**升** 1.2%。
- **Pass^3 明显崩塌**：Gemini 3.1 Pro 掉 24.2%，Opus 4.6 掉 14.3%，GLM 5 Turbo 掉 12.4%。
- **Resilience 与 baseline 不挂钩**：GLM 5 Turbo 起点不高但跌幅小，Gemini 3.1 Pro 起点高跌幅却最大——resilience 是独立的 capability axis。

> 这是一个 actionable insight：如果你只看 Pass@k，会得出"agents are highly resilient"的乐观结论；Pass^k 才暴露 deployment-grade 可靠性的崩塌。这一条本身就足以说服 benchmark 设计者把 Pass^k 作为 first-class metric。

### 6.3 多轮对话：会问问题 ≠ 问得多

13 个模型在 38 个多轮 task 上：

**Figure 5(a). 平均轮数 vs Pass^3，r = 0.07，R² < 0.01——几乎零相关。**

![](https://arxiv.org/html/2604.06132v1/x7.png)

但 question precision（clarification 精准度 + trajectory 逻辑性 的均值）与 Pass^3 强相关 r = 0.87，R² = 0.76。**轮数解释 < 1% 方差，问题质量解释 76%**。

> 这条很重要：它说明多轮 agent 的瓶颈不是"敢不敢追问 / 多轮交互能力"，而是"问对问题的策略"。对应到 training side，意味着多轮能力不能靠简单地 reward 长对话来涨——需要更精细的 question-quality reward（这是 agentic RL 的一个潜在切入点）。

### 6.4 Multimodal：没有冠军

**Figure 6（aggregated across 9 models）：Video 平均 Pass^3 只有 10.7%，远低于 Doc & Image (32.3%) 与 Code (23.9%)。**

![](https://arxiv.org/html/2604.06132v1/x9.png)

- 各域冠军不同：Video 由 Opus/Sonnet 4.6 并列领先（15.4%），Doc & Image 由 GPT 5.4 领先（54.5%），Code 由 MiMo V2 Omni 领先（33.3%）。
- 转化率 r = Pass^3 / Pass@3：Video 最低（0.37），Doc & Image 最高（0.53），Code 居中（0.48）——感知不确定性越高，run-to-run 方差越大。

> ❓ Video 的 Pass^3 普遍低（10.7% 平均）有两种可能解释：(a) 模型本身的视频理解能力差；(b) Claw-Eval 的 video task 对 frame sampling 策略和 ReadMedia tool 的接口设计敏感。论文没区分这两者——理论上应该有 ablation：固定一个 strong vision model（比如 GPT-5.4）改变 frame sampling tool 的便利程度，看 Pass^3 是否随接口改善而提升。

---

## 关联工作

### 基于 / 延续
- τ-bench：Claw-Eval 的 multi-turn dialogue 设计（simulated user persona + 隐藏 intent）和 Pass^k metric 都明显受 τ-bench 启发。
- TheAgentCompany：sub-task checkpoint 思想被 Claw-Eval 推广为 fine-grained rubric（2,159 items）。

### 对比
- [[2307-WebArena|WebArena]] / VisualWebArena：聚焦 web/desktop GUI navigation 单一模态，缺 safety / perturbation。
- [[2404-OSWorld|OSWorld]]：full desktop sandbox，但没有 embedded safety constraint 与 controlled error injection，且 rubric 仍偏 final-state。
- SWE-bench / Terminal-Bench / ToolBench：单模态 + output-only grading，是 Claw-Eval 直接超越的对象。
- AgentBench / GAIA：多领域聚合 benchmark，但都没做 trajectory auditing 与 perturbation。

### 方法相关
- ToolEmu / R-Judge / Agent-SafetyBench / MobileRisk-Live：把 safety 作为单独 red-teaming suite，没 in-task 嵌入约束。Claw-Eval 的 multiplicative safety gate 是直接的 differentiator。
- [[2411-RewardHacking|Reward Hacking]]：trajectory-opaque 评测被 frontier model 系统利用的现象，是 Claw-Eval 全轨迹审计的核心动机。
- LLM-as-a-judge：Claw-Eval 不抛弃 LLM judge，但用 deterministic check 兜底安全关键判定（hybrid pipeline）。

---

## 论文点评

### Strengths

1. **问题切中要害**：trajectory-opaque + safety underspecified + modality narrow 这三条是当前 agent benchmark 的真实痛点，44% 漏检率和 24pp Pass^3 drop 是有说服力的实证。
2. **Pass^k 与 multiplicative safety gate 的设计非常精炼**：Pass^k 把 capability 与 reliability 解耦，safety 作为 gate 而非 additive term 直接对应了"完成得再好但泄密就是失败"的 deployment 现实。这两个 design choice 几乎可以直接被其他 benchmark 借鉴。
3. **Triangulation of evidence 是干净的工程结构**：trace（agent 自陈）+ audit log（service 视角）+ snapshot（环境最终态）三路独立，agent 无法 simultaneously hack 三者。这种 evaluation surface 设计天生抗 reward hacking。
4. **配套真实 release**：300 task + 2,159 rubric + 14 模型的全量结果 + HF dataset + leaderboard，可复现性承诺到位（github README 显式承诺 audit codebase 让社区可复现）。

### Weaknesses

1. **任务量偏小，类别分布不均**：300 task 对覆盖"general / multimodal / multi-turn"3 大组 9 类来说偏少，特别是 Multi-turn Dialogue 只有 38 个 task；Hard 难度只有 43 个，统计噪声不可忽视。
2. **LLM judge 的 robustness 没充分 ablate**：5.1 节只用 Gemini-3-Flash 当 vanilla judge，没测更强的 judge（如 Opus 4.6）是否能缩小 gap。如果 vanilla 漏检主要是 judge 弱不是范式弱，结论的强度会打折。
3. **Reward hacking 缺 case study**：G1 用了"agents game evaluation"作为核心动机，但没给出 Claw-Eval 自己捕捉到的 hacking 实例（在 output-only 评测下 pass、在 trajectory 审计下 fail 的具体 trace）。
4. **Mock service 的 fidelity 未评估**：CRM / 邮件 / 调度器都是 mock，但 agent 在 mock 环境的行为是否能 transfer 到真 production API 没有讨论——这是几乎所有 sandboxed agent benchmark 的通病。
5. **Cost / latency 维度缺失**：表 4 只报 Score / Pass@3 / Pass^3，没报每个 task 的平均 token / wall-clock / 美元成本——deployment 视角下这些是同等重要的。

### 可信评估

#### Artifact 可获取性
- **代码**: github.com/claw-eval/claw-eval（README 承诺 codebase 正在 audit 以保证 leaderboard 全部可复现）
- **模型权重**: N/A（这是 benchmark 不是模型）
- **训练细节**: N/A
- **数据集**: 开源（[HuggingFace: claw-eval/Claw-Eval](https://huggingface.co/datasets/claw-eval/Claw-Eval)，MIT License）

#### Claim 可验证性
- ✅ 44% safety / 13% robustness 漏检率：5 模型 × 2,000+ trace 的实证，方法描述清楚（vanilla judge 拿到 transcript + grader 源码）。
- ✅ Pass^3 drop up to 24pp：Figure 4 的曲线和数字一致（Gemini 3.1 Pro 在 rate 0→0.6 之间）。
- ✅ Question precision r=0.87：13 模型的 scatter 都在 95% CI 带内（Figure 5b）。
- ⚠️ "trajectory-opaque evaluation is systematically unreliable"：只用了一个 vanilla judge（Gemini-3-Flash），没 ablate judge strength。如果换成 Opus 4.6 当 judge，漏检率可能显著下降——结论的"systematically"程度待验证。
- ⚠️ "no single model dominates all domains"：Multimodal 的 9-model × 3-domain 是个小样本，rank 洗牌可能部分来自 trial 噪声而非真正的 capability gap，希望看到 confidence interval。
- ❌ 暂无明显 marketing 修辞，整体表述比较克制。

### Notes

- **对自己研究的启发**：如果未来要设计/评估 computer-use 或 GUI agent，Pass^k + multiplicative safety gate + 三路证据这套组合应该是默认起点。我之前对 agent benchmark 的认知主要停在 OSWorld / WebArena 这种 final-state 检查 + single-trial pass rate 的层面，Claw-Eval 把 reliability axis 解耦出来这一点改变了我的判断——以后看 agent paper 的 benchmark 表，会先问"有没有 Pass^k？safety 有没有作为 gate？"
- **可借鉴的工程模式**：temporal firewall（grader artifact 在 agent 终止后才注入容器）这个细节非常聪明，可以直接套用到任何 sandboxed agent evaluation 框架上，不限于 Claw-Eval。
- **未解决的疑问**：mock service 的 fidelity 与 production API 的 gap 多大？如果 agent 学会了 game mock 服务的特定行为（比如 mock email 的特定 error message format），那么"transfer 到真服务"是不是又会暴露一批 failure？
- **不重读但要记得查阅**：rubric 设计的 4 个 case study（Appendix A.1–A.4）涵盖 general / multi-turn / multimodal，将来要写自己的 rubric-based grader 时是好的参考。

### Rating

**分数**：2 - Frontier
**理由**：设计选择（Pass^k、multiplicative safety gate、三路证据 triangulation、temporal firewall）非常精炼，且被"44% safety miss rate + 24pp Pass^3 drop"这类实证有力支撑，Strengths 里的几条 insight 几乎可以直接被其他 agent benchmark 借鉴——这让它明显高于 Archived。但它不到 Foundation：一是作为 2026 年 4 月刚出的 benchmark，社区采纳度和"被作为主要 baseline"的外部证据尚未形成；二是 Weaknesses 里的"300 task 偏小、judge 没 ablate、reward hacking 缺 case study"意味着它还不是 OSWorld / WebArena 级别的 de facto 标准，更像是一个方法范式明确的 frontier 参考。
