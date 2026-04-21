---
title: "Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents"
authors: [Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, Xin Eric Wang]
institutes: [Simular Research]
date_publish: 2025-04-01
venue: COLM 2025
tags: [computer-use, gui-agent, task-planning]
paper: https://arxiv.org/abs/2504.00906
website: https://www.simular.ai/articles/agent-s2
github: https://github.com/simular-ai/Agent-S
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
> - **核心**: 用 generalist Manager/Worker（强 LLM）+ specialist grounding experts（视觉/文字/结构）的模块化分工，配合 proactive hierarchical replanning，把 CUA 的 grounding 与 long-horizon planning 两大瓶颈解耦
> - **方法**: Mixture of Grounding（visual=UI-TARS-72B-DPO, textual=Tesseract OCR, structural=UNO spreadsheet API）+ Manager 在每个 subgoal 完成后基于最新 observation 主动 replan
> - **结果**: OSWorld 50-step 34.5%（前 SOTA OpenAI CUA 32.6%）、WindowsAgentArena 29.8%（vs NAVI 19.5%）、AndroidWorld 54.3%（vs UI-TARS-72B-SFT 46.6%）
> - **Sources**: [paper](https://arxiv.org/abs/2504.00906) | [website](https://www.simular.ai/articles/agent-s2) | [github](https://github.com/simular-ai/Agent-S)
> - **Rating**: 2 - Frontier（CUA compositional 路线的代表 SOTA 与必比 baseline，但 router 依赖 prompt engineering、已被 Agent S2.5/S3 迭代，不具 foundation 级奠基性）

**Key Takeaways:** 
1. **Compositional > monolithic**: 即便每个组件单独次优，"generalist planner + specialist grounder" 的合成系统也能反超最强的 monolithic CUA（如 Claude 3.7 Computer-Use）。Claude-3.5-Sonnet 装进 Agent S2 后 50-step 33.7%，比直接用 Claude-3.7 CCU 的 26.0% 高 7.7 个点
2. **Grounding 不是单一问题**：Visual / Textual / Structural 是三类不同任务，OCR 与 spreadsheet API 这种"老派"专才比让通用 VLM 强行回归坐标更准。Office 类任务正是靠 textual+structural 两个 expert 翻盘的
3. **Proactive 比 reactive 更适合 long horizon**：每个 subgoal 完成后基于最新 obs 重新生成剩余 plan（而非只在失败后才改），使得 50-step 评测 +6.15%。从 15→50 step 的额外收益主要来自 Adaptive Navigation / Adaptive Interaction / Backward Correction 三类涌现行为
4. **Specialist size 可以很小**：UI-TARS-7B-DPO、UGround-V1-7B 这类 7B 级 grounder 在 Agent S2 框架内的 15-step 表现优于把 Claude-3.7 当 grounder 用——专精性 > 参数量

**Teaser. Agent S2 在 OSWorld 15/50-step 评测上的 SOTA 对比，凸显从 32.6% 推到 34.5% 的"框架而非更大模型"路径。**

![](https://arxiv.org/html/2504.00906v1/x1.png)

---

## Background：CUA 的三大瓶颈

论文把当前 computer-use agents 的局限归纳为三点（也是 Agent S2 想修的三个口子）：
1. **Grounding 不准**：从 UI 元素的语言描述到像素坐标的映射常常错位，尤其是文本 span 边缘和 spreadsheet 单元格
2. **Long-horizon 难**：背景应用、弹窗、用户上下文演化都会污染规划；MLLM 对 UI 噪声尤其敏感
3. **Monolithic 瓶颈**：单个 generalist 同时干 planning + execution + grounding，每项都次优

任务被形式化为 POMDP $\mathcal{M}=(\mathcal{S},\mathcal{O},\mathcal{A},\mathcal{T},\mathcal{R})$，输入是 screenshot+指令，动作空间为 click/type/hotkey/drag/...等（详见 Action Space 表）。

> ❓ 形式化为 POMDP 在 CUA 论文里几乎是 boilerplate，本文也没真用 POMDP 工具（值函数/信念更新），主要是符号占位。

---

## Method：Agent S2 的三段式分工

**Figure 2. Agent S2 框架。Manager $M$ 把任务拆成 subgoals；Worker $W$ 在每步生成 atomic action 并通过描述路由到对应 grounding expert（MoG）；每个 subgoal 完成后 Manager 基于新 observation 重生剩余 plan（PHP）。**

![](https://arxiv.org/html/2504.00906v1/extracted/6326309/assets/framework.png)

整体由三类模块组成：
- **Manager $M$**（generalist LLM）：高层 subgoal 分解
- **Worker $W$**（generalist LLM）：低层 atomic action 生成 + grounding 路由
- **Grounding Experts $\{G_i\}$**（specialist）：把语言描述转成像素坐标或结构化操作

外加沿用 Agent S 的 knowledge base：高层任务经验、低层 subgoal 经验、上下文 web 知识。

### Mixture of Grounding (MoG)

类比 MoE，但 gating 不是学出来的——Worker $W$ 自己显式决定路由到哪个 expert。三类 expert：

| Expert | 输入 | 输出 | 实现 |
|---|---|---|---|
| **Visual Grounding** | screenshot $o$ + 语言描述 $d$ | 坐标 $\langle x, y \rangle$ | UI-TARS-72B-DPO |
| **Textual Grounding** | screenshot + 起始短语 $p_1$ + 结束短语 $p_2$ | 起止坐标 $\langle x_{start}, y_{start}\rangle, \langle x_{end}, y_{end}\rangle$ | Tesseract OCR |
| **Structural Grounding** | $\langle$"cell": "value"$\rangle$ 字典 | 直接程序化更新 | UNO (spreadsheet API) |

设计理由很务实：
- Visual expert 是 default，让 Agent S2 摆脱 accessibility tree（屏幕截图唯一输入）
- Textual expert 解决 word-level 边缘对齐——VLM 回归坐标在 "选中第三段最后一句" 这种任务上系统性地差几像素
- Structural expert 直接绕过 grounding，对 spreadsheet 改单元格用 API call 而非 click，又快又准

> 这是对 "grounding 是单一问题" 的有效证伪——把它拆成三类截然不同的 sub-problem，分别用最匹配的工具是显然更对的方向。但选什么 expert 仍由 Worker prompt 决策，没有学习成分，覆盖度受 prompt engineering 限制。

### Proactive Hierarchical Planning (PHP)

**Figure 3. Reactive vs Proactive Planning 对比。Reactive 只在 subgoal 失败后才改 plan；Proactive 每完成一个 subgoal 就基于新 observation 重生剩余 plan。**

![](https://arxiv.org/html/2504.00906v1/extracted/6326309/assets/reactive_vs_proactive_planning.png)

形式化流程：

1. 高层时间步 $T$：Manager $M$ 看 $(I, o_0)$ 生成 plan $\{g_1', g_2', \dots, g_n'\}$
2. Worker 执行第一个 subgoal $g_1 = g_1'$；低层每步 $t$ 选 action $a_t$ 并路由 grounding
3. Worker 以 SUCCESS 或 FAILURE 结束 subgoal，控制权交回 Manager
4. Manager 看 $(I, \text{prior subgoals}, o_t)$ 重生 plan $\{g_2'', g_3'', \dots, g_n''\}$，下一个 Worker 目标 $g_2 = g_2''$
5. 循环直到 $I$ 完成

关键设计：把"先前的 subgoals"作为 context 传回，让 Manager 在 recontextualize 的同时不丢任务连贯性。这一点比 vanilla replanning 更重要——单纯让 LLM 看新 obs 重写 plan，常常会"忘了任务原本要干什么"。

> 这其实也是对 reactive replanning 的一种 reframing：reactive 在失败后改的成本太高（已经走错路要回退），proactive 在每个 checkpoint 都验一下方向。代价是每步多一次 Manager LLM call，token 成本几乎 ×2。

---

## Experiments

### 主结果：OSWorld

**Table 1. OSWorld 上 Success Rate (%)，Agent S2 在 15-step 与 50-step 评测上均 SOTA。除了 Agent S 用了 a11y tree+screenshot，其他 baseline 与 Agent S2 都仅用 screenshot。**

| Method | 15-step | 50-step |
|---|---|---|
| Aria-UI w/ GPT-4o | 15.2 | – |
| Aguvis-72B w/ GPT-4o | 17.0 | – |
| Agent S w/ GPT-4o | 20.6 | – |
| Agent S w/ Claude-3.5-Sonnet | 20.5 | – |
| UI-TARS-72B-SFT | 18.7 | 18.8 |
| UI-TARS-72B-DPO | 22.7 | 24.6 |
| OpenAI CUA | 19.7 | 32.6 |
| CCU w/ Claude-3.5-Sonnet (new) | 14.9 | 22.0 |
| CCU w/ Claude-3.7-Sonnet | 15.5 | 26.0 |
| **Agent S2 w/ Claude-3.5-Sonnet (new)** | **24.5** | **33.7** |
| **Agent S2 w/ Claude-3.7-Sonnet** | **27.0** | **34.5** |

最有 insight 的对比是：**Agent S2 w/ Claude-3.5 (旧模型) 把 CCU w/ Claude-3.7 (新模型) 在 15-step 上打了 58.1% relative**——证明框架收益大于模型迭代。

**Table 2. Agent S2 在 OSWorld 50-step 各类别上的 SR (%)。Office 类别 Claude-3.5 反而比 3.7 强（29.06 vs 25.64）——分析显示是因为 3.5 调用 textual/structural expert 的频率高近 2 倍。**

| Backbone | OS | Daily | Office | Professional | Workflow | Overall |
|---|---|---|---|---|---|---|
| GPT-4o | 50.00 | 30.70 | 18.97 | 51.02 | 14.93 | 26.62 |
| Claude-3.5-Sonnet (new) | 58.33 | 48.44 | 29.06 | 51.02 | 13.46 | 33.71 |
| Claude-3.7-Sonnet | 50.00 | 49.73 | 25.64 | 57.14 | 18.21 | 34.47 |

> Office 反转的解释——3.5 更愿意路由到专家——某种程度上暴露了 router 是 prompt-conditioned 的脆弱性：换模型换 router 行为，没法稳定预测。

**Figure 4. Agent S2 一个典型轨迹：先用 Visual Grounding 试图选段落 → 自我修正改用 Textual Grounding 完成 span 选择 → subgoal 完成后基于新状态 replan 启动下一个 subgoal。**

![](https://arxiv.org/html/2504.00906v1/x2.png)

### WindowsAgentArena & AndroidWorld 泛化

**Table 3. WindowsAgentArena 15-step SR (%)。注意 Agent S 与 NAVI 都用了 a11y tree+screenshot，Agent S2 仅用 screenshot 仍超出 NAVI 52.8% relative。**

| Method | Office | Web | Win System | Coding | Media | Win Utils | Overall |
|---|---|---|---|---|---|---|---|
| Agent S | 0.0 | 13.3 | 45.8 | 29.2 | 19.1 | 22.2 | 18.2 |
| NAVI | 0.0 | 27.3 | 33.3 | 27.3 | 30.3 | 8.3 | 19.5 |
| **Agent S2 (Ours)** | **7.0** | 16.4 | **54.2** | **62.5** | 28.6 | **33.3** | **29.8** |

**Table 4. AndroidWorld SR (%)。Agent S2 worker-only 设置（短 horizon）即可 SOTA。**

| Method | SR (%) |
|---|---|
| GPT-4o + UGround | 44.0 |
| GPT-4o + Aria-UI | 44.8 |
| UI-TARS-72B-SFT | 46.6 |
| **Agent S2 (Ours)** | **54.3** |

### Ablation

**Figure 5. MoG 与 PHP 的消融：MoG 把 OSWorld-65 子集 SR 从 27.69%→30.77% (15-step) 与 33.85%→38.46% (50-step)；PHP 在 15-step +4.62%、50-step +6.15%。两个组件在 long horizon 收益更大。**

![](https://arxiv.org/html/2504.00906v1/x3.png)

更细的 expert-wise 消融：去掉 textual expert，subtask SR 从 70.6%→65.2%；去掉 structural expert，从 73.7%→69.4%。

Visual grounding model 的对比也得到一个反直觉结论：**7B 的 specialist（UI-TARS-7B-DPO、UGround-V1-7B）作为 visual grounder 比让 Claude-3.7-Sonnet 兼任 grounder 还要好**——支撑了"compositional 框架不需要每个组件都最强"的核心论点。

### Test-time 行为分析

**Figure 7. 从 15-step 到 50-step 的成功率提升来自四种行为：Adaptive Navigation（换路径找元素）、Adaptive Interaction（换交互方式）、Backward Correction（修复早期 subgoal 的小错）、Task Complexity（任务本来就要 >15 步）。前三种都是 PHP+MoG 直接促成的涌现。**

![](https://arxiv.org/html/2504.00906v1/x5.png)

### Error Analysis

Failure 模式分布：planning > grounding > interaction > navigation > infeasible。这是有意义的偏移——前作把 grounding 列为头号 failure，Agent S2 把 grounding 压下来后 **planning 成为新的瓶颈**，下一代要修的应该是 Manager 的推理能力（或者 manager-worker 信息传递的保真度）。

### Action Space

**Table 8. Agent S2 的 13 类 atomic action。`highlight_text_span` 与 `set_cell_values` 直接对应 Textual / Structural Grounding expert。**

| Action | Args |
|---|---|
| click | element_description, num_clicks, button_type, hold_keys |
| type | element_description, text, overwrite, enter |
| scroll | element_description, clicks, shift |
| hotkey | keys |
| hold_and_press | hold_keys, press_keys |
| drag_and_drop | element_description_1, element_description_2, hold_keys |
| save_to_knowledge | text |
| switch_applications | app_name |
| highlight_text_span | starting_phrase, ending_phrase |
| set_cell_values | cell_values, app_name, sheet_name |
| wait | time |
| done / fail | None |

### Case Studies

**Figure 9. Textual Grounding 实例。任务："给最后一段文字加 strikethrough"。Agent S2 调用 textual expert 进行精准的 word-level span 选择。**

![](https://arxiv.org/html/2504.00906v1/x7.png)

**Figure 10. Structural Grounding 实例。任务："新增 'Profit' 列，按周从 'Sales' 减 'COGS' 计算"。Agent S2 选择 structural expert 用 cell-level API 完成更新。**

![](https://arxiv.org/html/2504.00906v1/x8.png)

**Figure 11. Proactive Planning 实例。任务："关闭 'Dim screen when inactive'"——该选项不存在 verbatim，Agent S2 通过 replan 找到等价 setting。**

![](https://arxiv.org/html/2504.00906v1/x9.png)

---

## 关联工作

### 基于
- [[2410-OSAtlas|OS-ATLAS]]: Agent S2 把这类训过的 GUI grounding model 当作 visual expert 之一在框架里替换测试
- Agent S (Agashe et al. 2024, ICLR 2025)：直接前作，Agent S2 沿用其 hierarchical Manager-Worker + knowledge base，主要新增 MoG 与 PHP

### 对比 (CUA baselines)
- [[2501-UITARS|UI-TARS-72B-DPO]]: 最强 monolithic native agent baseline；Agent S2 反而把它作为 visual grounding expert "招安"进自己的框架
- OpenAI CUA / Operator: 闭源 monolithic，Agent S2 在 50-step 上以 34.5 vs 32.6 反超
- Claude Computer Use (Anthropic): 闭源 monolithic，Agent S2 用更弱的 Claude 3.5 backbone 也能超它的 3.7 版本
- NAVI + [[2408-OmniParser|OmniParser]]: WindowsAgentArena 前 SOTA，被 Agent S2 反超 52.8% relative
- Aguvis、Aria-UI、UGround：作为 grounding-specialist baselines，在 AndroidWorld 上对比

### 方法相关
- Mixture-of-Experts (Jacobs et al. 1991)：MoG 的灵感来源，但 gating 改为 Worker 显式 prompt 决策而非可学 router
- [[2312-CogAgent|CogAgent]]: 早期视觉 GUI 模型，monolithic 路线
- [[2404-OSWorld|OSWorld]]: 主评测环境
- [[2409-WindowsAgentArena|WindowsAgentArena]]: 跨 OS 评测
- AgentStore (Jia et al. 2024)：另一种 "many specialist" 框架，组合的是 app-specific agent 而非 grounding expert
- Cradle (Tan et al. 2024)：multi-agent 框架，分配责任到多个模型
- OSCAR (Wang & Liu 2024)：state-aware reasoning + replanning，PHP 的近邻概念

### 后续
- Agent S2.5 (2025/08)：simpler/faster 版本，进一步在 OSWorld-Verified 上 SOTA
- [[2510-ScalingAgents|Agent S3]] (arXiv 2510.02250, 2025/10)：首次在 OSWorld 上超过人类（72.60%）

---

## 论文点评

### Strengths

1. **诊断准确**：把 CUA 三大病灶（grounding bottleneck / long-horizon brittleness / monolithic 极限）拆得很清楚，每个 component 一一对应一种病灶
2. **"模块化即 routing" 的执行很彻底**：MoG 不是把多个 grounder ensemble，而是让 Worker 显式选 expert——Textual/Structural 两个 expert 完全跳出 "VLM 回归坐标" 的 paradigm，这种解耦才有意义
3. **强 ablation**：MoG 与 PHP 各自消融，加上 visual grounder 替换实验，把"compositional > monolithic"的核心论点钉死
4. **Cross-OS 泛化做得彻底**：Linux (OSWorld) → Windows (WindowsAgentArena) → Android (AndroidWorld) 三个环境都 SOTA，证明框架不是 over-fit 到某一 env
5. **error mode shift 的观察很有价值**：grounding 被压制后 planning 反成瓶颈，这是给整个 CUA community 的诊断信号

### Weaknesses

1. **Router 是 prompt-engineering**：Worker 选哪个 expert 完全靠 LLM prompt 决策，没有学习信号；换 backbone 模型 router 行为就变（Office 任务上 Claude 3.5 vs 3.7 的反转就是证据），稳定性可疑
2. **Token 成本翻倍但未报**：PHP 让 Manager 在每个 subgoal 后都重新生成 plan，加上 MoG 的 expert 调用，整体 LLM call 与 token 消耗远超 monolithic baseline——但论文没有 cost-normalized 对比
3. **Specialist 对 application 的耦合**：Structural expert 用 UNO API 只对 LibreOffice 有效，Textual expert 依赖 Tesseract OCR——这套 expert 池的可扩展性、维护成本（每个新 app 是否都要新 expert？）没有讨论
4. **Knowledge base 模糊带过**：沿用 Agent S 的 knowledge base，但其对 Agent S2 性能贡献的量化 ablation 缺失。无法判断收益究竟来自新组件还是旧 KB
5. **OSWorld-65 子集 ablation**：消融实验全在 65 例子集上，子集是否能代表 369 全集的统计行为没有论证
6. **没和 monolithic SOTA 等参对比**：Agent S2 用 Claude-3.7 + UI-TARS-72B（72B 参数 specialist）的组合，对手 OpenAI CUA 是单模型——总参数量不对等

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only 开源 (https://github.com/simular-ai/Agent-S，pip 包名 `gui-agents`)，且已迭代到 Agent S3
- **模型权重**: 不发布新权重——visual grounder 复用 UI-TARS-72B-DPO（公开），textual 用 Tesseract（开源），structural 用 UNO（开源）
- **训练细节**: 不适用——Agent S2 本身无训练，全部组件 off-the-shelf
- **数据集**: 评测集（OSWorld、WindowsAgentArena、AndroidWorld）均公开，无新数据

#### Claim 可验证性
- ✅ **OSWorld 50-step 34.5% / 15-step 27.0% SOTA**：表格清晰、与 baseline 同 protocol，已被 README 中后续 Agent S2.5/S3 工作引用对比
- ✅ **MoG 与 PHP 各贡献 ~3-6% SR**：消融在 OSWorld-65 子集上做得规范，数值与图一致
- ✅ **Cross-OS 泛化**：三个 benchmark 均开源可复现
- ⚠️ **"compositional > best monolithic"**：在没控制 token/参数预算的前提下，结论需要打折——Agent S2 总算力（generalist + 72B grounder + Manager 多次 replan）比 monolithic 模型大得多
- ⚠️ **"Office 类 3.5 > 3.7 是因为更频繁调 expert"**：归因来自统计 expert 调用次数，但是相关而非因果——也可能是 3.5 在 spreadsheet 任务上别的 prompt-following 特性更好
- ⚠️ **OSWorld-65 子集消融的代表性**：未论证 65 例的类别分布与全集一致

### Notes

- 这篇是 CUA 路线 "compositional vs monolithic" 之争里一个干净的 evidence point。值得记住的不是 SOTA 数字，而是两点 design pattern：
  1. **Grounding 不是单一问题**——Visual / Textual / Structural 三类要分开处理，OCR 与 spreadsheet API 这种 90s 老技术在专用 niche 仍优于 VLM
  2. **Replan 应该是 default 而非 fallback**——只在失败后 replan 的成本太高（已经走偏要回退），每个 checkpoint 都验一下方向更划算
- 但 Agent S2 整个 router 是 prompt-conditioned 的，没有 RL/SFT 信号——这是天花板。下一步要么把 router 学出来（Agentic RL 路线），要么 Specialist 池继续扩大（更多 app-specific expert）。Agent S3 走了哪条路值得对照看
- "Specialist 不必很大"是一个 nice 的 secondary takeaway——7B grounder 在框架里能跑赢 Claude-3.7 当 grounder。这给小团队做 CUA 框架留了空间
- 我的 mental model 更新：**CUA 性能的下一道天花板是 planning 而非 grounding**。Manager 看不到 Worker 内部状态，只能看到最终 SUCCESS/FAILURE 信号，这种 hierarchical 信息瓶颈是 PHP 也没法完全弥补的
- ❓ 论文宣称 "compositional 即便每个组件次优也能超 best monolithic"——但没控制 token/参数预算。如果给 monolithic 同样预算（比如让它跑 50 step + extended thinking），差距还有这么大吗？

### Rating

**分数**：2 - Frontier
**理由**：CUA compositional 路线的代表 SOTA、COLM 2025 录用，在三个 OS benchmark 上反超当时所有 monolithic 系统，且 README 显示它已成为后续 Agent S2.5 / S3 等主要工作必对比的 baseline——符合 Frontier "必须比较的 Baseline、方法范式的代表工作"定义。但它不具 Foundation 级奠基性：MoG 的 router 仍是 prompt-engineering（Weaknesses #1），Agent S3 已在 OSWorld 上反超人类把它 10 个月内取代，且其 Manager-Worker 骨架本质继承自 Agent S 前作，自身创新点是 MoG + PHP 两个工程性组合而非奠基 paradigm，因此不到 3。
