---
title: "A Comprehensive Survey of Agents for Computer Use: Foundations, Challenges, and Future Directions"
authors: [Pascal J. Sager, Benjamin Meyer, Peng Yan, Rebekka von Wartburg-Kottler, Layan Etaiwi, Aref Enayati, Gabriel Nobel, Ahmed Abdulkadir, Benjamin F. Grewe, Thilo Stadelmann]
institutes: [ZHAW, University of Zurich, ETH AI Center, Polytechnique Montreal, University of Fribourg, AlpineAI, European Centre for Living Technology]
date_publish: 2026-02-24
venue: JAIR vol. 85, 2026
tags: [computer-use, gui-agent, web-agent]
paper: https://arxiv.org/abs/2501.16150
website: https://sagerpascal.github.io/agents-for-computer-use/
github:
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] A Comprehensive Survey of Agents for Computer Use
> - **核心**: 把 87 篇 ACU agents + 33 个 datasets 按 domain / interaction / agent 三视角组织成统一 taxonomy，用以诊断领域里 6 个 research gap
> - **方法**: 半结构化文献检索 + snowballing；归纳式 coding 出三视角分类（domain × observation/action × agent design/learning）
> - **结果**: 识别趋势——specialized → foundation agent、text → image observation、BC 主导 environment learning；提出 6 条 recommendation——视觉观察 + low-level control、cost-efficient learning、planning、高复杂度 benchmark、统一以 task success rate 评估、对齐 deployment 假设
> - **Sources**: [paper](https://arxiv.org/abs/2501.16150) | [website](https://sagerpascal.github.io/agents-for-computer-use/)
> - **Rating**: 2 - Frontier（technology-agnostic 三视角 taxonomy 有结构性贡献，但 evidence 多 illustrative 且未覆盖 2025 年 GUI grounding + 商业系统前沿，不构成 foundation 级必读）

**Key Takeaways:**
1. **三视角 taxonomy**：domain（Web/Android/PC）、interaction（observation / action 空间）、agent（policy 类型 + 三阶段 learning：pre-training → environment learning → episodic improvement）。设计上 domain-agnostic，把 RL-era 和 foundation-agent-era 的工作放在同一坐标系里讨论
2. **观察空间正在从 text 转向 image**：2024 年图像观察首次成为最常见 observation type；作者论证文本观察（HTML、view hierarchy）依赖 *optional* 语义属性，泛化脆弱——MiniWoB++ 上文本 agent 表现极强但在 Mind2Web 上跌到 <10%
3. **Learning gap：foundation agent 缺 environment learning**：long-term memory 难抽象、RL/BC 资源密集。作者建议在 pre-training 与 environment learning 之间引入 self-supervised "ACU 对齐" 阶段，类比 RLHF 之于通用 LLM
4. **Planning 仍是瓶颈**：现有 informal planning（CoT、self-critique、sub-task decomposition）收益有限；formal planning + 学习到的 dynamics model 在小规模实验里展示 +50% success rate
5. **评估极不标准化**：MiniWoB++ 不同论文用不同 task subset 还直接比较；作者强烈推荐 task success rate 为主指标，并要求 stop action 作为评估必要条件

**Teaser. ACU 任务示例。** Figure 1(a) 展示一个 ACU 在手机上完成"约会议时间并发邮件"任务的交互过程。

![](https://arxiv.org/html/2501.16150v3/figures/ACU_example.png)

---

## 1. Problem Setup 与 Taxonomy 设计动机

ACU（Agents for Computer Use）= 给定自然语言 instruction $i$，通过模拟鼠标点击 / 触屏手势 / 键盘等低级动作在数字设备（PC / 手机）上完成任务的系统。形式化为 POMDP 风格的 (observation $o_t$, action $a_t$, history $h_t$, policy $\pi$)：

$$
a_{t}\sim\pi(\ \cdot\ |\ o_{t},\ i\ )
$$

引入两个工程化操作：observation simplification $o_t \to o_t^*$（如降采样 screenshot）与 action grounding $a_t^* \to a_t$（把 "click submit" 解析成 `click(x,y)`）。

> 作者强调他们的 taxonomy 跟传统 intelligent agent theory（Russell & Norvig）一脉相承，刻意做成 technology-agnostic——既能装下 RL-based specialized agent，也能装下 prompt-based foundation agent。这是相对其他 ACU survey（如 Wang et al. 等）最大差异，便于做跨范式的横向对比。

**Figure 2. ACU 出版量随时间的变化。** ChatGPT 出现是明显的转折点，把研究从 RL 主导转到 foundation model 主导。

![](https://arxiv.org/html/2501.16150v3/figures/data/timeline_new.png)

文献筛选流程：semi-structured 检索 → 多轮 snowballing → inductive coding 出 taxonomy。最终 87 ACUs + 33 datasets。约 1/3 是 preprint（reviewer 用 domain knowledge 控制质量）——**survey 的 reproducibility 由此打折扣**。

---

## 2. Domain Perspective

三个主流 domain：**Web、Android、Personal Computer**。共享的观察 / 动作抽象：

**Table 1. Observation types across domains。**

| Observation | Web | Android | PC |
| --- | --- | --- | --- |
| Image screen | Website screenshot | Phone screen | Foreground app |
| Textual screen | HTML, accessibility tree | View hierarchy, a11y tree | UI automation tree |
| Indirect | Network traffic | — | File contents |

**Table 2. Action types across domains。**

| Action | Web | Android | PC |
| --- | --- | --- | --- |
| Mouse/touch/kbd | click(x,y) | tap(x,y) | mouse + keyboard |
| Direct UI | HTML element id | Android element | UIA API |
| Task-tailored | "find on page" | "go back" | "switch app" / "send email" |
| Code | JS / Python / Selenium | adb | UIA API / Bash |

**Recommendation**：87 篇里只有 10 篇做 PC domain，作者明确呼吁更多关注 desktop——多窗口、跨应用工作流、共享文件系统才是 productivity automation 的真痛点。

---

## 3. Interaction Perspective

### 3.1 Observation Space

87 个 agent 里，35 个仅用 textual observation（受 LLM 时代影响），但 **2024 年 image 已超过 text 成为最常见 observation**（VLM 普及）。

四类 observation：image / textual / bi-modal / indirect。Textual 几乎都要简化（heuristic pruning、element filtering、representation embedding、text summarization），因为 raw HTML 太冗长。

> [!quote] 作者的核心 claim
> 文本观察依赖 *optional* 语义属性（如 `id`、`name`），在 sanitized benchmark（MiniWoB++）上效果好但跨真实站点崩溃；image-based 观察因为人类设计惯例反而更 consistent，generalization 更强。

证据链：MiniWoB++ 上去掉 textual modality 性能掉 75%；但在 Mind2Web 上 text agent 成功率 <10%，image agent（GPT-4V）能到 38%。

> ❓ 这个 evidence chain 略 cherry-picked——MiniWoB 的对照是 bi-modal 减 textual，Mind2Web 的对照跨了不同 backbone。"image 更 generalize" 的方向是对的，但量化幅度可能被 backbone 能力差异污染。

**Indirect observation**：少数 agent 不直接看屏幕，而是通过 routine 读 PowerPoint / Excel / 邮件列表 / REST 响应等系统状态。

### 3.2 Action Space

四类：mouse/touch+kbd、direct UI access、task-tailored、executable code。**Direct UI access 仍是最常见**——因为 action space 维度低且语义化，绕开了 spatial reasoning 的难题。

**Action grounding**：把 abstract "click submit" 落到具体 UI 元素，分 prediction-based（用 grounding model 预测元素）和 rule-based（文本匹配）。视觉 agent 主流是 **set-of-mark prompting**——给可点击元素叠加 bbox + ID，让模型选 ID 而非预测坐标。

> 作者断言 set-of-mark 是临时 workaround：等视觉基座能直接预测坐标，就不需要这层中介了。这个判断与 SeeClick / Aria-UI / AGUVIS 等 GUI grounding 模型的趋势一致。

**Figure 7. Sankey diagram：domain → observation → action 空间的连接。** 大量 vision agent 仍在用 direct UI action，作者认为这是技术约束下的妥协。

![](https://arxiv.org/html/2501.16150v3/figures/data/doa_sankey.png)

**Recommendation**：versatile ACU 应当用 mouse/touch/keyboard + image observation——modality 自然对齐，避免桥接 hack。

---

## 4. Agent Perspective

### 4.1 两种 agent 设计

**Table 3. Foundation vs Specialized agent。**

| Type | Architecture | Action | Memory | Learning |
| --- | --- | --- | --- | --- |
| Foundation agent | LLM / VLM | Generation | history-based | General + Episodic |
| Specialized agent | Custom NN | Prediction | state-based | Environment learning |

Specialized 在窄任务、弱 instruction conditioning 下高效；Foundation agent 在多步、强 conditioning 下靠 pretraining + CoT 主导。

### 4.2 Policy 类型

- **Memoryless** $\pi(\cdot|o_t,i)$：只看当前观察。简单任务才够。
- **History-based** $\pi(\cdot|o_t,i,h_t)$：foundation agent 主流，~60/87 篇用。$h_t$ 因 token 预算被简化为：actions only / 选择性保留若干 observation / embedded summary / text summary。
- **State-based** $\pi(\cdot|o_t,i,m_t)$：specialized agent 主流，用 RNN 等学习 state-update function $f_m$。
- **Mixed**：少数 hybrid。

> 作者提出一个洞察：state-based policy 的 $f_m$ 本质是 *learnable history simplification*，而 foundation agent 现在的 history simplification 都是手工或通用 summarizer——**应该把 history simplification 做成 learnable component**，与 world model 文献相通。这个观点比较 fresh，可能确实是个 underexplored 方向。

### 4.3 Learning Strategy（核心章节）

三阶段：**General pre-training → Environment learning → Episodic improvement**。

**Figure 8. 学习阶段的组合方式：** specialized agent 走 pre-training + RL/BC 通路；foundation agent 走 pre-training + (optional long-term memory) + ICL 通路。

![](https://arxiv.org/html/2501.16150v3/figures/learning_flow_colored.png)

**Environment learning** 三种途径：

- **RL**：依赖 controlled environment + 奖励信号；稀疏奖励常需 BC warmup（如某 web agent 用 240 万人类动作 BC 后再 RL，把成功率从 30% 拉到 95%）；reward shaping 和 curriculum learning 是缓解手段。瓶颈是 controlled env 的工程成本极高。
- **BC**：监督学习模仿人类轨迹，可在 uncontrolled env 上进行。fine-tune 全模型 / 部分组件 / LoRA 都有。
- **Long-term memory**（foundation agent 专属）：把成功 trajectory 存到外部库，下次按 instruction 检索作为 ICL 示例。两类存储——environment transitions $(o_t, a_t, o_{t+1})$ 和 task demonstrations $(i, \tau_i)$。少数工作把 memory 组织成图，对 action 做参数化抽象（`click(text=Bob)` → `click(text=[contact name])`）来增强泛化。

**Episodic improvement**：foundation agent 通过 ICL（instruction tuning + few-shot demonstrations）；specialized agent 大多没有这一步。视觉 prompt 工程包括 set-of-mark、UI 元素叠 bbox 等。

**Episodic improvement through Planning**：

- *Informal planning*：CoT、sub-task decomposition、self-critique。作者引文献指出 self-critique 收益有限。
- *Formal planning*：在 controlled env 里 search 5 步前瞻，task success +50%；后续工作训练 model 直接预测 action 对 observation 的影响，不需要外部 simulator。

> [!quote] 作者最强的 recommendation
> 在 pre-training 和 environment learning 之间引入一个 self-supervised "ACU alignment" 阶段——类比 RLHF 之于通用 LLM，给基座模型注入 computer-use inductive bias。**这是一个 actionable 的研究方向**，但具体 self-supervised objective 是什么作者没给。

> ❓ 这个 alignment 阶段的设计空间值得追问：是 next-action prediction on screenshots？是 UI grounding 任务？是 trajectory autoencoder？survey 没回答，留给后续工作。

---

## 5. Datasets & Evaluation

### 5.1 Datasets

两类：**controlled environments**（可重置、有 reward signal、支持 RL）vs **offline datasets**（人类录制 trajectory，安全但只覆盖单条解法）。

时间趋势：复杂度和真实度同步上升。Web 主线：MiniWoB → MiniWoB++（100 任务，sanitized HTML）→ WebShop（单一真实购物 app）→ Mind2Web（137 个真实站点 2350 demo）→ WebArena（4 个真实 web app 的 controlled env）→ VisualWebArena（+视觉任务）。Android 主线：PixelHelp → AndroidEnv → MoTIF → Android in the Wild（70 万 demo, 357 apps）。PC：AgentBench、OmniACT（9802 任务跨 57 应用）。

**Figure 11. Dataset 时间线。** 复杂度随时间稳步上升。

![](https://arxiv.org/html/2501.16150v3/figures/dataset_domain_timeline.png)

**主要批评**：现有 dataset 缺 *trajectory complexity*——任务里 actions 之间没有强 causal dependency，agent 只要分别填几个 form field 就行，不像真实工作需要"先查日历可用时段再发邀请"这种因果链。

### 5.2 Evaluation

| 层级 | 主要指标 |
| --- | --- |
| Task-level | task success rate（offline / online）；task progress；avg reward |
| Step-level | step success rate（macro / micro avg）；action F1；element accuracy |
| Other | efficiency（API calls）、safeguard rate |

**Online vs Offline gap**：同一 agent online success 36% vs offline 12%（[Zheng et al. on Mind2Web](https://arxiv.org/abs/2501.16150)）——offline metric 严重低估，因为 alternative valid trajectory 不被认。

**Recommendation**：以 task success rate 为主指标；在受控环境下报 task success；offline 时同时报 step success（reproducible proxy）+ online success（实际能力）；并强制 stop action 作为完成信号。

---

## 6. Conclusion 提炼的 6 个 gap → 6 条 recommendation

| # | Gap | Recommendation |
| --- | --- | --- |
| 1 | 输入模态结构不一致，泛化差 | 采用 image-based observation |
| 2 | Learning 资源密集 / 不易扩展 | 引入 cost-efficient learning（含 self-supervised alignment 阶段） |
| 3 | Planning 能力弱 | 长程推理架构 + neuro-symbolic |
| 4 | Benchmark 重 perception 轻 task complexity | 引入因果依赖更强的复杂任务 |
| 5 | 评估指标不统一 | 标准化 task success rate |
| 6 | 研究假设与部署条件脱节 | 关注 dynamic / non-stationary env、隐私、conditional autonomy |

---

## 关联工作

### 同类 Survey
- **OS Agents survey** (Hu et al., ACL 2025 Oral)：MLLM-based 跨设备 agent survey，scope 与本文重合度高；本文强调 technology-agnostic（含 RL 时代），OS Agents 偏 MLLM-only
- **GUI Agent Survey** (Wang et al.)：仅 mobile，本文 scope 更广

### Benchmark 与 Environment
- [[2307-WebArena|WebArena]] / [[2404-OSWorld|OSWorld]]：被多次引用作为"真实复杂度" benchmark 代表
- Mind2Web：作为 offline dataset 复杂度上限的代表
- AndroidWorld：作者承认未纳入 dynamic 评估场景

### GUI Grounding 模型（survey 未充分讨论的相邻线）
- [[2401-SeeClick|SeeClick]] 等：本文主要把 grounding 当 prompting 技巧（set-of-mark），未深入 grounding 模型这条 supply chain
- [[2501-UITARS|UI-TARS]]：同月发布的端到端 GUI agent 模型，作为 commercial-leaning 系统被排除

### Foundation Agent 代表
- Zheng et al. (GPT-4V web agent)：本文最频繁引用，作为 image-based / set-of-mark 范式的 reference

---

## 论文点评

### Strengths

1. **Taxonomy 设计 technology-agnostic**：三视角 + 三阶段 learning 框架确实把 RL 时代和 foundation model 时代的 ACU 工作放进了同一坐标系，对比性强。这是 survey 类工作里少见的"兼容旧范式"努力，比单纯堆 LLM agent 的 review 更有结构。
2. **Recommendation 部分有 actionable insight**：尤其 "在 pre-training 和 environment learning 之间引入 self-supervised alignment" 这一点，类比 RLHF 给出了清晰的研究方向。"history simplification 应当 learnable" 借了 world model 文献也是 fresh 的视角。
3. **图表组织清晰**：Sankey 图（Figure 7）展示 domain × observation × action 的连接非常直观地暴露 "vision agent 仍依赖 direct UI access" 这种妥协；timeline（Figure 11）说明 dataset 复杂度演化也很有用。
4. **明确划界 + 自我反省**：survey 范围（排除 game-playing、coding agent、software testing、纯 RPA）说得清楚；同时指出局限（1/3 preprint、subjective selection、scope 限于 text instruction）。
5. **JAIR final 版本 2026-02 收录**，相比 v1（2025-01）在文献覆盖上应有更新。

### Weaknesses

1. **Evidence 偏弱处不少**：support 关键 claim 的对比（如 "image > text" 的 MiniWoB → Mind2Web 性能掉幅）跨了不同 backbone 和 metric，attribution 不严，更像 illustrative 而非 evidential。
2. **缺 quantitative meta-analysis**：87 篇 agent 应该可以做 per-benchmark 的 success rate 对比表（即便不全），但 main text 几乎不报数字，所有定量信息都塞 appendix figure。读完很难知道"目前 SOTA 在 X benchmark 上是多少"。
3. **"6 个 gap" 之间高度耦合**：gap 1（modality）和 gap 5（evaluation）几乎是同一问题的两面（sanitized benchmark + text observation 互相支撑），但被并列陈述，读起来有 inflation 感。
4. **Self-supervised alignment 的 recommendation 没具体化**：比 RLHF 的类比好听，但具体 objective、data source、与 SFT/RL 的协同都没讲，actionable 程度被夸大。
5. **未涵盖最新 commercial system**：明确说排除 Claude Computer Use、Operator、UI-TARS 等闭源 / 半闭源系统的细节，但这些恰恰是把 ACU 推到实用边缘的工作。survey 自己也承认这是 limitation，但实际让 survey 在 2026-04 读起来"赶不上 frontier"。
6. **几乎不提 GUI grounding 专门模型**（SeeClick、OS-Atlas、Aria-UI、AGUVIS 等），把 grounding 仅作为 set-of-mark 的临时方案讨论，错过了 "grounding 模型本身已成为子领域" 这条线。

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源（survey 性质，无算法实现）；项目页 `sagerpascal.github.io/agents-for-computer-use` 提供 interactive 版本的统计图
- **模型权重**: 不适用
- **训练细节**: 不适用
- **数据集**: 不适用——survey 本身的 87 agent + 33 dataset 表格在 Appendix F 公开

#### Claim 可验证性
- ✅ **共识性的趋势观察**（"specialized → foundation"、"text → image 在 2024 年发生"、"BC 是最常用 environment learning"）：与 87 篇 paper 的 metadata 统计可核对（appendix table）
- ✅ **Online vs offline success rate 差距**（12% → 36%）：直接引自 Zheng et al.，可追溯
- ✅ **BC + RL 组合的 30% → 95% 提升**：引自原始 web agent 论文，可追溯
- ⚠️ **"image observation 比 text observation 泛化更好"**：主要靠 cross-paper / cross-benchmark 数字对比，未控制 backbone 与 metric，attribution 弱
- ⚠️ **"self-supervised alignment 阶段会有效"**：类比 RLHF 的论证是 plausible 但未经验证，属于 hypothesis 而非 finding
- ⚠️ **"现有 benchmark task complexity 不足"**：定义模糊——"causal dependency between actions" 没有量化指标，主观判断成分大
- 暂未发现明确 ❌ 营销话术

### Notes

- **对我的研究价值**：
  - 作为 ACU 领域的 entry-point 综述合格——尤其三阶段 learning 视角好用，可以放进自己的 mental model 当 axis 来定位新论文
  - "self-supervised alignment between pretraining and environment learning" 是个值得跟进的研究 direction——目前 GUI grounding 模型（SeeClick / OS-Atlas / Aria-UI）实际上已经在做某种 alignment，但没人把它系统化为 "ACU-RLHF 等价物"
  - Survey 没覆盖 AGUVIS / Aria-UI / Magma 等 GUI foundation model 这条线索，**也没覆盖 agentic RL 在 GUI 上的最新进展**（如 Claude Computer Use 后续 ZeroGUI、AgentRL 等），属于明显空白
- **Open question 列表**：
  1. ACU 的 self-supervised alignment objective 应该长什么样？是 next-action prediction、UI element 对比学习、还是 trajectory autoencoder？
  2. Learnable history simplification 有没有可能直接复用 world model 的 latent dynamics？
  3. Causal-dependency-rich benchmark 如何系统构造？是否可以用 LLM 自动生成因果图任务？
- **下一步若深入 ACU**：建议追 (a) 2025-2026 年 GUI grounding model 这条线（SeeClick → OS-Atlas → Aria-UI → AGUVIS → Magma），(b) Claude Computer Use 之后的 RL-based 改进（ZeroGUI、Agent Q 等），(c) 复杂 benchmark 演化（OSWorld、WindowsAgentArena、TheAgentCompany）

### Rating

**分数**：2 - Frontier
**理由**：三视角 technology-agnostic taxonomy + 三阶段 learning 框架（见 Strengths 1-2）把 RL 时代与 foundation agent 时代统一到一个坐标系，对 ACU 方向研究者有结构性参考价值，够到 Frontier 档。但 evidence 多 illustrative（Weakness 1）、缺定量 meta-analysis（Weakness 2）、且在 2026-04 读已显著落后于 GUI grounding model 线与商业系统线（Weakness 5-6），不是必读必引的 foundation 级综述——相邻领域存在 OS Agents (ACL 2025 Oral) 等竞争 survey，本文未成为 de facto reference。相比 Archived 档的一次性参考，本文 taxonomy 章节仍值得回查。
