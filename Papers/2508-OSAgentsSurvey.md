---
title: "OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use"
authors: [Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, Yuhuai Li, Shengze Xu, Shenzhi Wang, Xinchen Xu, Shuofei Qiao, Zhaokai Wang, Kun Kuang, Tieyong Zeng, Liang Wang, Jiwei Li, Yuchen Eleanor Jiang, Wangchunshu Zhou, Guoyin Wang, Keting Yin, Zhou Zhao, Hongxia Yang, Fan Wu, Shengyu Zhang, Fei Wu]
institutes: [Zhejiang University, Fudan University, OPPO AI Center, University of Chinese Academy of Sciences, Institute of Automation CAS, The Chinese University of Hong Kong, Tsinghua University, Shanghai Jiao Tong University, 01.AI, The Hong Kong Polytechnic University]
date_publish: 2025-08-06
venue: ACL 2025 (Oral, 9-page version) / arXiv (full)
tags: [computer-use, gui-agent, web-agent]
paper: https://arxiv.org/abs/2508.04482
website: https://os-agent-survey.github.io/
github: https://github.com/OS-Agent-Survey/OS-Agent-Survey
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use
> - **核心**: 系统综述基于 (M)LLM 的 OS Agents——能在桌面/移动/浏览器等操作系统提供的 GUI 环境中操控计算设备完成用户任务的 agent
> - **方法**: 三层框架——Fundamental（环境/观察空间/动作空间 + 理解/规划/grounding 三大能力）→ Construction（domain-specific foundation model + agent framework，后者由 perception/planning/memory/action 四组件构成）→ Evaluation（protocol + benchmark by platform/setting/task）
> - **结果**: 覆盖 ~30 个 foundation model（Table 1）、~30 个 agent framework（Table 2）、29 个 benchmark（Table 3），并梳理 safety/privacy 与 personalization/self-evolution 两大未来方向
> - **Sources**: [paper](https://arxiv.org/abs/2508.04482) | [website](https://os-agent-survey.github.io/) | [github](https://github.com/OS-Agent-Survey/OS-Agent-Survey)
> - **Rating**: 2 - Frontier（OS Agent 方向最完整的 component taxonomy + 三张 artifact 表 + 活跃 living repo；但截稿点偏早、2025 年 agentic RL 与新商业产品缺失，未达 Foundation 档。）

**Key Takeaways:**
1. **将 OS Agent 定义为三组件 + 三能力**：环境（desktop/mobile/web）、观察空间（screenshot / textual description like HTML、DOM、A11y tree）、动作空间（input ops、navigation ops、extended ops）；能力为 understanding、planning、grounding——这套定义大致是当前文献的共识。
2. **Foundation model 路线四分**：Existing LLMs（HTML 输入路线，e.g. WebAgent + Flan-U-PaLM/HTML-T5）、Existing MLLMs（直接复用 LLaVA/Qwen-VL/InternVL/CogVLM）、Concatenated MLLMs（自选 LLM + vision encoder）、Modified MLLMs（高分辨率改造，CogAgent/Ferret-UI/MobileFlow/UI-Hawk），训练上 PT/SFT/RL 三段式，目前 RL 用得仍少（Table 1 只有约 6/30 标了 RL）。
3. **Agent framework 四组件画地图**：Perception（textual description vs GUI screenshot，后者细分 visual/semantic/dual grounding）、Planning（global vs iterative，iterative 主流是 ReAct/Reflexion 衍生）、Memory（internal/external/specific × short/long-term，有 management/growth experience/retrieval 三种优化）、Action（input/navigation/extended）。
4. **Benchmark 按 platform × setting × task 三维切分**：Mobile/Desktop/Web × Static/Interactive(Simulated/Real-World) × GUI Grounding/Information Processing/Agentic/Code Generation。real-world interactive（OSWorld、AndroidWorld、WindowsAgentArena 等）正成为主流。
5. **未来挑战**：Safety/Privacy（WIPI、environmental injection、对抗 popup、jailbreak），Personalization/Self-Evolution（memory mechanism 长期累积用户数据）——defense 侧研究明显落后于 attack 侧。

**Teaser. Fundamentals of OS Agents（环境/观察/动作三组件 + 理解/规划/grounding 三能力的总览图）。**
![](https://arxiv.org/html/2508.04482v1/x2.png)

---

## 1 Fundamental of OS Agents

OS Agent 被定义为：在 OS 提供的环境与接口下使用计算设备、自主完成用户目标的 (M)LLM-based agent。综述把它拆成三个 **Key Component** 与三个 **Core Capability**。

### 1.1 Key Components

- **Environment**：desktop / mobile / web，三类平台的复杂度与可观测性差别很大（mobile GUI 元素简单但手势复杂；desktop OS/应用多样性高；web 易于 inspect 与 crowdsource human demonstration）。
- **Observation Space**：两类——
  - 视觉：screen images / annotated screenshot（含 SoM 标注）
  - 文本：screen description / HTML / DOM / accessibility tree
  - 多模态融合带来理解和动作 grounding 上的难度。
- **Action Space**：input operations（mouse、touch、keyboard）、navigation operations（scroll、back/forward、tab、URL）、extended operations（code execution、API call）。

### 1.2 Core Capabilities

- **Understanding**：解析高分辨率、密集小元素的 GUI 是核心瓶颈，HTML 长且稀疏，screenshot 文字小图标多。
- **Planning**：把复杂任务分解为子任务并按环境反馈动态调整，常见策略 ReAct / CoAT。
- **Grounding**：把指令翻译成可执行动作（具体到 element + 参数如 coordinates、input value），元素多导致搜索空间大。

> ❓ 综述把 "capability" 和 "component" 分开列其实有些重复——观察空间决定了 understanding 的输入形态，动作空间决定了 grounding 的输出形态。把它们拆成两个 section 读起来更像 taxonomy 罗列而非真正分层。

---

## 2 Construction of OS Agents

### 2.1 Foundation Models

按 architecture 与训练方式归纳为四类（Table 1 列了 30 篇代表工作）：

**Table 1. Recent foundation models for OS Agents（节选）。Arch.: Architecture; PT/SFT/RL: training stages.**

| Model | Arch. | PT | SFT | RL | Date |
| --- | --- | --- | --- | --- | --- |
| OS-Atlas | Exist. MLLMs | ✓ | ✓ | - | 10/2024 |
| AutoGLM | Exist. LLMs | ✓ | ✓ | ✓ | 10/2024 |
| ShowUI | Exist. MLLMs | ✓ | ✓ | - | 10/2024 |
| UGround | Exist. MLLMs | - | ✓ | - | 10/2024 |
| Ferret-UI 2 | Exist. MLLMs | - | ✓ | - | 10/2024 |
| MobileVLM | Exist. MLLMs | ✓ | ✓ | - | 09/2024 |
| UI-Hawk | Mod. MLLMs | ✓ | ✓ | - | 08/2024 |
| MobileFlow | Mod. MLLMs | ✓ | ✓ | - | 07/2024 |
| Ferret-UI | Exist. MLLMs | - | ✓ | - | 04/2024 |
| AutoWebGLM | Exist. LLMs | - | ✓ | ✓ | 04/2024 |
| ScreenAI | Exist. MLLMs | ✓ | ✓ | - | 02/2024 |
| SeeClick | Exist. MLLMs | ✓ | ✓ | - | 01/2024 |
| CogAgent | Mod. MLLMs | ✓ | ✓ | - | 12/2023 |
| WebAgent | Concat. LLMs | ✓ | ✓ | - | 07/2023 |

**Architecture 四分**：
1. **Existing LLMs**：直接用开源 LLM 处理 HTML / 文本描述。代表：WebAgent（Flan-U-PaLM + HTML-T5），AutoGLM，AutoWebGLM。
2. **Existing MLLMs**：复用 LLaVA / Qwen-VL / InternVL / CogVLM 等开源 MLLM 作为 backbone。
3. **Concatenated MLLMs**：自行选 LLM + vision encoder 拼接，强调让架构匹配 OS 任务（如 T5 encoder-decoder 适配 HTML 树结构）。
4. **Modified MLLMs**：针对 GUI 高分辨率改造 ——
   - CogAgent：加入 EVA-CLIP-L 高分辨率（1120×1120）vision encoder + cross-attention
   - Ferret-UI：any-resolution，sub-image 分块
   - MobileFlow：Qwen-VL backbone + LayoutLMv3 GUI encoder（embed image + OCR text + position）
   - UI-Hawk：shape-adaptive cropping

**Figure 2. Foundation model 部分的内容总结图（架构 + PT/SFT/RL 三段式训练）。**
![](https://arxiv.org/html/2508.04482v1/x3.png)

**Training**：
- **Pre-training**：多用 continual PT；数据上 publicly available（CommonCrawl HTML、Flickr30K）+ synthetic（SeeClick 从 HTML 抽取 visible text + position 构造 grounding/OCR 数据，OS-Atlas 用 A11y tree 跨平台 sample，MobileVLM 用 directed graph 收集 3M 真实交互样本）。任务上分 screen grounding / screen understanding / OCR 三类。
- **SFT**：planning trajectory + grounding instruction 是两条主线。planning 数据多通过遍历 app（fixed rule + LLM）、用 tutorial 文章映射成 action、或 directed graph 找最短路径生成；grounding 数据通过渲染 HTML/simulator 截图 + 自动生成 referring expression。
- **RL**：两个阶段——早期 RL 训 web/mobile 任务（WebShop、MiniWob++），LLM 当 feature extractor；近期 "LLMs as agents" 范式下，RL 用于把 LLM policy 与最终目标对齐（AutoGLM 的 self-evolving online curriculum RL、AGILE 框架的 PPO）。但 Table 1 中只有少数模型用 RL，相对 SFT 仍是非主流。

> ❓ 综述对 RL 的处理偏轻——2025 年以后 agentic RL（GRPO / verifier-free RL）在 GUI agent 上爆发，这部分应该会在该 survey 的下一版扩充很多。

### 2.2 Agent Frameworks

非微调路线：在已有 (M)LLM 上靠 prompt + memory + tool 搭框架。Table 2 列了 30 个代表 framework，按 perception / planning / memory / action 四组件刻画。

**Figure 3. Agent framework 部分的内容总结图（perception/planning/memory/action 四组件）。**
![](https://arxiv.org/html/2508.04482v1/x4.png)

**Table 2. Recent agent frameworks for OS Agents（节选）。**

| Agent | Perception | Planning | Memory | Action | Date |
| --- | --- | --- | --- | --- | --- |
| Agent S | GS, SG | GL | EA, AE, MA | IO, NO | 10/2024 |
| OSCAR | GS, DG | IT | AE | EO | 10/2024 |
| AgentOccam | TD | IT | MA | IO, NO | 10/2024 |
| Agent-E | TD | IT | AE, MA | IO, NO | 07/2024 |
| Cradle | GS | IT | EA, AE, MA | EO | 03/2024 |
| OS-Copilot | TD | GL | EA, AE | IO, EO | 02/2024 |
| Mobile-Agent | GS, SG | IT | AE | IO, NO | 01/2024 |
| WebVoyager | GS, VG | IT | MA | IO, NO | 01/2024 |
| SeeAct | GS, SG | - | AE | IO | 01/2024 |
| AppAgent | GS, DG | IT | AE | IO, NO | 12/2023 |
| RCI | - | IT | AE | IO, NO | 03/2023 |

缩写：TD=Textual Description, GS=GUI Screenshot, VG/SG/DG=Visual/Semantic/Dual Grounding, GL=Global, IT=Iterative, AE=Automated Exploration, EA=Experience-Augmented, MA=Management, IO/NO/EO=Input/Navigation/Extended Operation。

**Perception**：
- **Textual description**：HTML/DOM/A11y tree。问题是冗余信息多，因此 Agent-E 用 flexible DOM distillation，AgentOccam 仅在 take action 时展开 HTML，WebWise 用 `filterDOM` 按 tag/class 过滤。
- **GUI screenshot**：分三种 grounding——
  - Visual grounding：SoM prompting + OCR + element detector（ICONNet、Grounding DINO）
  - Semantic grounding：SeeAct 用 HTML 文档作为截图的语义参考
  - Dual grounding：AppAgent（labeled screenshot + XML）、OSCAR（A11y tree + 描述性 label）、PeriGuru、DUAL-VCR（Pix2Struct + MindAct 风格 HTML 对齐）

**Planning**：
- **Global**：CoT 一次产出全计划。OS-Copilot 把 plan 形式化为 DAG 支持并行执行；Agent S 提出 experience-augmented 层次规划（融合记忆与 online sources）；AIA 用 SOP 分解。
- **Iterative**：基于 ReAct/Reflexion，根据环境反馈动态调整。OSCAR 引入 task-driven replanning；SheetCopilot 用 state machine + 反馈/检索修正；RCI 让 LLM 找自己输出的 bug；CoAT 用 Screen Description → Action Thinking → Next Action Description → Action Result 的 OS-targeted 推理链。

**Memory**：
- **Sources**：
  - Internal Memory（短期：action history、screenshots、state data；长期：执行路径、学到的 skill code）
  - External Memory（外部 KB、API 文档）
  - Specific Memory（任务相关规则、子任务分解、用户 profile）
- **Optimization**：
  - Management：multimodal self-summarization（MM-Navigator、Cradle）、Context Clipping（WebVoyager 保留最近 3 步 observation + 完整 thought/action）、AgentOccam 的 planning tree（每个新 plan 独立 goal，剪掉旧 plan 的历史步骤）
  - Growth Experience：MobA dual reflection（事前评估 feasibility + 事后 review）、LASER 的 Memory Buffer 支持回溯
  - Experience Retrieval：AWM 提取 workflow 复用、PeriGuru 用 KNN 找相似 task case

**Action**：input（mouse 三类 + keyboard 两类）、navigation（basic + web-specific tab/URL）、extended（code execution、API integration）——标准化的 action API 设计依赖 platform。

---

## 3 Evaluation

### 3.1 Evaluation Protocol

- **Objective**：rule-based / hard-coded 在标准 benchmark 上算指标，覆盖 perception accuracy、generation quality、action effectiveness、operational efficiency。具体匹配方式：exact match、fuzzy match、semantic match。
- **Subjective**：human/LLM-as-judge 评估输出与人类期望的对齐（relevance、coherence、naturalness、harmlessness）。LLM-as-judge 提升了效率但可控性和可靠性受限。

**Metric 两个 scope**：
- **Step-level**：对每步的 action grounding 和 element matching 评分（operation acc/F1, element acc/F1, BLEU/ROUGE/BERTScore for QA），聚合得 step success rate。局限：长序列任务下不一定 robust，且任务可能存在多条 valid path。
- **Task-level**：
  - Task Completion：Overall SR、Accuracy、Reward function
  - Efficiency：Step Ratio（vs human optimal）、API Cost、Execution Time、Peak Memory Allocation

### 3.2 Benchmarks

按平台 × 环境设置 × 任务三维分类，Table 3 收录 29 个 benchmark。

**Table 3. Recent benchmarks for OS Agents（节选）。BS: Benchmark Setting (ST=Static, IT=Interactive); OET: Operation Environment Type (RW=Real-World, SM=Simulated); Task: GG/IF/AT/CG.**

| Benchmark | Platform | BS | OET | Task | Date |
| --- | --- | --- | --- | --- | --- |
| AndroidWorld | Mobile | IT | RW | AT | 05/2024 |
| AndroidControl | Mobile | ST | - | AT | 06/2024 |
| AITW | Mobile | ST | - | AT | 07/2023 |
| WindowsAgentArena | PC | IT | RW | AT | 09/2024 |
| OSWorld | PC | IT | RW | AT | 04/2024 |
| OmniACT | PC | ST | - | CG | 02/2024 |
| Mind2Web-Live | Web | IT | RW | IF, AT | 06/2024 |
| VisualWebArena | Web | IT | RW | GG, AT | 01/2024 |
| WebArena | Web | IT | RW | AT | 07/2023 |
| Mind2Web | Web | ST | - | IF, AT | 06/2023 |
| WebShop | Web | ST | - | AT | 07/2022 |
| MiniWoB | Web | ST | - | AT | 08/2017 |

- **Platform**：Mobile（Android/iOS，元素简单但手势复杂；Android API 大）、Desktop（OS/应用多样，scalability 难）、Web（HTML/CSS/JS 易 inspect，crowdsource demonstration 成本低）。
- **Setting**：Static（cached website snapshot，仅支持 one-step action，如 Mind2Web）vs Interactive（feedback loop，simulated 如 FormWoB / 真实如 OSWorld、WebArena、AndroidWorld、WindowsAgentArena）。
- **Task**：
  - GUI Grounding（PIXELHELP 等）
  - Information Processing（Retrieval + Summarizing，AndroidWorld、WebLINX 等）
  - Agentic Tasks（核心评估，需要 plan + execute 直至目标态，Mind2Web、MMInA 等）
  - Code Generation（OmniACT 等）

---

## 4 Challenges & Future

### 4.1 Safety & Privacy

**Attack** 路径已有相当多研究：
- **WIPI**（Web Indirect Prompt Injection）：把自然语言指令藏在网页里间接控制 web agent
- 对抗 caption：adversarial image → 错误 caption → agent 偏离用户目标
- **Environmental Injection**：在网页/环境中嵌入隐蔽的恶意指令窃取用户信息
- 对抗 pop-up window 攻击 web agent 决策
- 浏览器场景下 refusal-trained LLM 的拒绝能力不能 transfer，jailbreak 可绕过
- 移动 agent 的 4 类 realistic attack path × 8 种 attack method

**Defense** 严重落后：综述明确指出 "studies on defenses specific to OS Agents remain limited"，需要针对 injection、backdoor 等开发系统性防御。

**Benchmark**：ST-WebAgentBench（企业 web agent 安全/可信）、MobileSafetyBench（移动 agent，含 messaging/banking 等敏感任务）。

### 4.2 Personalization & Self-Evolution

定位为类似 J.A.R.V.I.S. 的长期目标：要持续根据用户偏好进化。OpenAI 的 memory feature 是早期实践。当前 OS Agent 研究因为缺乏真实用户数据，多停留在任务性能上而非个性化。Memory 机制是关键载体；模态扩展（image / voice）+ 高效检索是开放问题。

> ❓ 这两节是 survey 里最弱的部分——基本是 "存在某个未解问题、有几篇代表工作、未来值得做"。Personalization 那一节没有 method taxonomy，只有 motivation 段。

---

## 关联工作

### 同类 survey
- [[2411-GUIAgentSurvey]]（Microsoft, 2024-11）：同时期、同主题但更聚焦 GUI agent 视角，500+ refs，与本 survey 互补——前者更偏 cookbook，本 survey 更偏 taxonomy
- 综述里点名的 "concurrent works"：personalized agents、GUI Agents、generalist virtual agents

### Foundation models 代表（survey 引用）
- [[2410-OSAtlas|OS-Atlas]]：跨平台 GUI grounding foundation model
- [[2312-CogAgent|CogAgent]]：高分辨率 GUI MLLM 改造范式的代表
- [[2401-SeeClick|SeeClick]]：早期 GUI 领域 PT+SFT
- AutoGLM、Ferret-UI、ScreenAI、UGround、ShowUI、MobileVLM、UI-Hawk、MobileFlow

### Frameworks 代表
- WebVoyager、AppAgent、Mobile-Agent、SeeAct、Agent S、OSCAR、OS-Copilot、Cradle、AgentOccam、Agent-E
- 通用 reasoning 基座：ReAct、Reflexion、CoT、CoAT

### Benchmarks
- Mobile：AndroidWorld、AndroidControl、AITW、AndroidArena、B-MoCA
- Desktop：[[2404-OSWorld|OSWorld]]、[[2409-WindowsAgentArena|WindowsAgentArena]]、OfficeBench、ASSISTGUI、OmniACT
- Web：[[2307-WebArena|WebArena]]、VisualWebArena、Mind2Web、Mind2Web-Live、WebShop、MiniWoB、WebLINX、WorkArena、MMInA

### 商业产品提及
- Anthropic Claude Computer Use、Apple Intelligence、Zhipu AutoGLM、Google DeepMind Project Mariner

---

## 论文点评

### Strengths

1. **覆盖面广 + 表格扎实**：Table 1（30 个 foundation model 按 arch/PT/SFT/RL 列）、Table 2（30 个 framework 按 perception/planning/memory/action 列）、Table 3（29 个 benchmark 按 platform/setting/task 列）—— 三张表本身就是检索价值最高的 artifact，比正文更有用。
2. **定义清晰、taxonomy 干净**：环境/观察/动作 + 理解/规划/grounding 的拆分基本对齐了领域共识，subsection 命名一致；framework 的 perception 分 visual/semantic/dual grounding 这种细分粒度合适。
3. **配套 GitHub 仓库持续维护**：作为 living survey 的 dynamic resource，对追踪领域进展实用。
4. **ACL 2025 Oral 9 页 + 完整 arXiv 长版** 双载体兼顾会议传播和详细 reference。

### Weaknesses

1. **时间窗口已过期**：截稿点大约在 2024-10/11，2025 年大爆发的 agentic RL 路线（GUI 上的 GRPO / verifier-free RL / process reward）、computer-use 商业产品（Claude Computer Use 之后的 OpenAI Operator、Anthropic 升级、Project Mariner 落地）、新 benchmark（OSWorld-Human、AgentBench 系列）几乎都没覆盖。RL section 只有 6/30 个工作，与当前主流明显脱节。
2. **缺少量化对比**：Table 1/2/3 都是 categorical labels（用没用 PT/SFT/RL，是哪类 grounding），没有任何 benchmark 上的 SR 数字。读者读完仍不知道 "Agent S vs WebVoyager 在 OSWorld 上谁强多少"。这对一份 survey 是显著的信息损失。
3. **Capability 与 Component 划分有冗余**：observation space ↔ understanding、action space ↔ grounding 之间的对应关系没有说清，造成同样的内容（HTML / screenshot 处理）在 §2.1 和 §2.2 两处都谈。
4. **Safety/Defense + Personalization 两节单薄**：attack 列举详细但 defense 几乎 placeholder；personalization 没 taxonomy，只有 narrative。这两个被列为 "未来方向" 的 section 反而最该做扎实 mapping。
5. **缺少 failure mode / negative result 视角**：survey 通篇是 "X work proposes Y"，没有总结 "什么场景下哪类方法 break"。比如 dual grounding 何时优于 visual-only？iterative planning 何时反而 hurt？这些 cross-paper insight 缺失。
6. **作者列表非常长（29 人，10 个机构），但单位贡献度不清晰**——typical 大型 collaborative survey 的风险，章节可能不够 coherent（实际读起来章节风格切换确实较明显）。

### 可信评估

#### Artifact 可获取性
- **代码**: 不适用（survey paper），但 GitHub 仓库作为 paper list 持续维护
- **模型权重**: 不适用
- **训练细节**: 不适用
- **数据集**: 不适用；survey 引用的 benchmark 列表见 Table 3 与 GitHub 仓库

#### Claim 可验证性
- ✅ Foundation model / framework / benchmark 的分类与列表：可逐条对照原 paper 验证
- ✅ "Defense studies specific to OS Agents remain limited"：可通过 search 复核，符合 2024 年底的状态
- ⚠️ "Recent advancement … significantly advanced"：典型 survey narrative，定性 claim 难证伪
- ⚠️ Table 1/2 的 categorical 标签（如某模型用了 RL 与否）：依赖作者读论文准确性；抽查 OS-Atlas 标 PT+SFT 无 RL 与原文一致，但表格未提供 reference 链接到具体 section
- ❌ "This survey aims to consolidate the state of OS Agents research"：营销话术，不是技术 claim

### Notes

- **作为 entry point 价值高**：刚进入 OS Agent / GUI Agent 领域时，先读 §1.1（components）+ Table 1/2/3 + §4.1（safety taxonomy），能快速建起 mental map。但**不要把 Table 中的 categorical 标签当 ground truth**，需要回原文 verify。
- **与 [[2411-GUIAgentSurvey]] 配合读**：两篇时间点接近，前者更偏 historical narrative + cookbook，本 survey 更偏 component taxonomy。Table 之间有重叠但不完全一致，差异本身有信息量。
- **Pivot signal**：综述把 RL section 放在 §2.1.4，列了不到 10 个 RL 工作。考虑到 2025 上半年 GUI agent 的 RL 训练（含 verifier-based RL、process reward、online curriculum）爆发式增长，**这份 survey 的 SFT-centric 视角已经有明显时间偏差**。如果要追当前 SOTA，应该把它当 2024 年底的 snapshot 读。
- **可作为我自己 GUI agent 方向的 related work 检索表**：Table 2 的 framework 设计维度（perception / planning / memory / action 各自的实现选择）可以直接作为 design space 在自己的工作里复用。
- ❓ **想验证的疑问**：Table 1 中标 RL 的工作（AutoGLM、AutoWebGLM、WebAI、GLAINTEL、RUIG 等）实际 RL 形式差异很大（PPO vs behavior cloning + RL vs online curriculum），survey 没区分，需自己读原文。

### Rating

**分数**：2 - Frontier
**理由**：作为 field-centric 参考，本 survey 提供了 OS Agent 领域目前最完整的 component taxonomy（环境/观察/动作 + 理解/规划/grounding）与三张 artifact 表，是进入该方向的高价值检索入口，并配套 living GitHub 仓库，符合 "方向的研究前沿和重要参考" 的 Frontier 定位。但尚未成为 de facto 必读奠基（同主题有 [[2411-GUIAgentSurvey]] 等并列综述），且如 Weaknesses #1 指出截稿点在 2024-10/11、RL 与 2025 商业产品大面积缺失，时间窗口已偏移——不具备 Foundation 档需要的长期权威性，故定 2 而非 3；又因其仍是该方向被广泛引用的 taxonomy 参考、不是 incremental/niche 小综述，故高于 1。
