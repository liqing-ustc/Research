---
title: "Vision-and-Language Navigation Today and Tomorrow: A Survey in the Era of Foundation Models"
authors: [Yue Zhang, Ziqiao Ma, Jialu Li, Yanyuan Qiao, Zun Wang, Joyce Chai, Qi Wu, Mohit Bansal, Parisa Kordjamshidi]
institutes: [Michigan State University, University of Michigan, UNC Chapel Hill, University of Adelaide]
date_publish: 2024-07-09
venue: TMLR 2024
tags: [VLN, navigation, VLM]
paper: https://arxiv.org/abs/2407.07035
website:
github: https://github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models
rating: 2
date_added: 2026-04-23
---

## Summary

> [!summary] Vision-and-Language Navigation Today and Tomorrow: A Survey in the Era of Foundation Models
> - **核心**: 用 LAW（Language-Agent-World）框架把 VLN 研究切成 World Model / Human Model / VLN Agent 三个子问题，系统梳理 foundation model 时代的 VLN 方法与 benchmark。
> - **方法**: Top-down 综述——不按 benchmark 或模型时间线组织，而是按"foundation model 在 VLN 里扮演什么角色"来归类（history encoder / pre-trained repr / instruction synthesizer / planner / agent backbone）。
> - **结果**: 覆盖 24 个 VLN benchmark（indoor R2R/RxR/REVERIE 系到 outdoor TouchDown/CARLA/AerialVLN，按 world×human×agent×dataset 4 维分类），识别出 4 种 FM 角色，给出 5 条未来方向（2D→3D、instruction→dialogue、sim→real、embodied FM、hallucination）。
> - **Sources**: [paper](https://arxiv.org/abs/2407.07035) | [github](https://github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models)
> - **Rating**: 2 - Frontier （作为 LLM 时代 VLN 的第一本系统 survey，taxonomy 清晰，但一年多后 streaming VLA、VLN-R1、3D EWR 等新路线已出现，结构性覆盖度开始老化，不到 foundation 级）

**Key Takeaways:**
1. **LAW 三分法是全篇骨架**：World model（环境表征与记忆）+ Human model（指令解释与对话）+ VLN Agent（grounding / planning / 端到端 policy），是从 Hu & Shu 2023 借来的通用 agent 框架，用来把 VLN 的异质挑战标准化。
2. **Benchmark 四维分类**：world（domain × environment）× human（turn × format × granularity）× agent（embodiment × action space × extra task）× dataset collection（text × route）。这张表（Table 1）比 taxonomy 本身更有信息量，可以直接用来挑对标 benchmark。
3. **FM 有 4 种角色**：(a) 作为 visual/text encoder（CLIP、BERT、VLM pre-training）；(b) 作为 instruction synthesizer / oracle（Marky、GPT-3 分解指令）；(c) 作为 planner（LLM-Planner、SayCan、VL-Map 的 code-as-policy）；(d) 作为 agent 本体（NavGPT、MapGPT zero-shot；NavCoT 监督微调）。
4. **R2R + Matterport3D 是 de facto 起点**，但 VLN-CE（continuous）、REVERIE（object grounding）、CVDN / TEACh（dialogue）、ALFRED（manipulation）、TouchDown / AerialVLN（outdoor / aerial）分别暴露了该起点的不同盲区。
5. **Taxonomy 暴露 gap**：Human Model 章节远薄于 World / Agent——对话式 VLN、information seeking、oracle 机制都还是规则驱动而非 FM 驱动，是 survey 隐含的 under-explored 方向。

**Teaser. LAW 框架把 VLN 拆成 World Model / Human Model / VLN Agent**

![](https://arxiv.org/html/2407.07035v2/x1.png)

Figure 1 把 VLN agent 建模为：环境侧的 world model（理解 3D 场景 + action→state 的动力学）、人侧的 human model（解读语言指令、维护说话人 goal），两者作为输入共同喂给 agent 的 reasoning & planning 模块输出动作。这张图是理解整篇 survey 章节编排的 key。

---

## 1. Taxonomy & 分类框架

**Figure 2. VLN challenges & solutions within the LAW framework**

![](https://arxiv.org/html/2407.07035v2/x2.png)

Survey 把每个模块进一步拆成具体挑战：
- **World model** → history & memory（encoding / graph-based）+ generalization（pre-trained visual repr / environment augmentation）
- **Human model** → ambiguous instructions（perceptual context / information seeking）+ generalization of grounded instructions（pre-trained text repr / instruction synthesis）
- **VLN agent** → grounding & reasoning（explicit semantic / VLN pre-training）+ planning（graph-based / LLM-based）+ FM as agent（VLM / LLM）

**方法在 foundation model 里扮演的角色** 也被显式分 4 类（见 Figure 2 右下）：backbone encoder、data / oracle generator、planner、agent policy。这条正交轴是本 survey 对过往 VLN survey（Gu et al. 2022 / Park & Kim 2023 / Wu et al. 2024）的主要差异化——它们按 benchmark × method 组织，这篇按 FM 角色组织。

> ❓ LAW 框架把 Human model 抬到和 World model 平级，这在 VLN 里 OK（指令是核心输入），但迁移到 VLA / manipulation 就不自然——那里没有专门的 "human model"。这个框架的适用边界没有被讨论。

## 2. Benchmarks & Task Formulations

### 2.1 VLN 任务定义

Agent 接收语言指令序列、以 egocentric 视角感知 3D 环境，输出离散 view 序列或 low-level control（如 FORWARD 0.25m），距离目标 ≤3m 判定成功。可选扩展：多轮对话、manipulation（[[2204-SayCan|SayCan]]-style）、object detection。

### 2.2 Benchmark 总览

**Table 1（浓缩版）**：24 个 VLN benchmark 按 4 维分类。全表太宽，这里挑 highlight：

| Benchmark | Domain | Env | Turn | Lang Format | Action Space | Embodiment |
|---|---|---|---|---|---|---|
| **R2R** (Anderson 2018) | Indoor | Matterport3D | Single | Multi Instr | Graph | Robot |
| **R4R** (Jain 2019) | Indoor | Matterport3D | Single | Multi Instr | Graph | Robot |
| **RxR** (Ku 2020) | Indoor | Matterport3D | Single | Multi Instr (multilingual) | Graph | Robot |
| **REVERIE** (Qi 2020) | Indoor | Matterport3D | Single | Multi Instr (goal-oriented) | Graph | Robot + Detect |
| **SOON** (Zhu 2021) | Indoor | Matterport3D | Single | Multi Instr (goal) | Graph | Robot |
| **VLN-CE** (Krantz 2020) | Indoor | Habitat + MP3D | Single | Multi Instr | Discrete | Robot |
| **Robo-VLN** (Irshad 2021) | Indoor | Habitat + MP3D | Single | Multi Instr | Continuous | Robot |
| **CVDN** (Thomason 2020) | Indoor | MP3D | Multi | Restricted dialogue | Graph | Robot |
| **HANNA** (Nguyen 2019) | Indoor | MP3D | Multi | Multi Instr | Graph | Robot |
| **ALFRED** (Shridhar 2020) | Indoor | AI2-THOR | Single | Multi Instr | Discrete | Robot + Manipulation |
| **TEACh** (Padmakumar 2022) | Indoor | AI2-THOR | Multi | Freeform | Discrete | Robot + Manipulation |
| **DialFRED** (Gao 2022) | Indoor | AI2-THOR | Multi | Restricted | Discrete | Robot + Manipulation |
| **TouchDown** (Chen 2019) | Outdoor | Google Street View | Single | Multi Instr | Graph | — |
| **LCSD / CDNLI / SDN** | Outdoor | CARLA | Single/Multi | Multi Instr / Freeform | Discrete / Cont | Driving |
| **AerialVLN** (Liu 2023) | Outdoor | AirSim | Single | Multi Instr | Discrete | Aerial |
| **ANDH** (Fan 2023) | Outdoor | xView | Multi | Freeform | Discrete | Aerial |
| **RobotSlang** (Banerjee 2021) | Indoor | **Real** | Multi | Freeform | Discrete | Robot |

**四维分类提纲**（survey 原文 §2.3）：
- **World**：domain（indoor / outdoor）+ environment（MP3D / Habitat / AI2-THOR / Street View / CARLA / AirSim / Real）
- **Human**：turn（single / multi）+ format（freeform / restricted dialogue / multi-instruction）+ granularity（action-directed "A" / goal-directed "G"）
- **VLN Agent**：type（household robot / driving / aerial）+ action space（graph / discrete / continuous）+ extra task（manipulation / detection）
- **Dataset**：text collection（Human "H" / Templated "T"）+ route（Human "H" / Planner "P"）

### 2.3 Evaluation Metrics

1. **NE**（Navigation Error）：终点到 goal 的测地距离均值。
2. **SR**（Success Rate）：到目标 ≤3m 的比例。
3. **SPL**（Success weighted by Path Length）：SR 用轨迹长度归一，平衡成功率和路径效率。
4. **CLS**（Coverage weighted by Length Score, Jain 2019）：度量对参考路径的覆盖。
5. **nDTW**（Normalized Dynamic Time Warping, Ilharco 2019）：惩罚对 ground-truth 轨迹的偏离。
6. **sDTW**：nDTW × SR，同时考虑 fidelity 和 success。

## 3. World Model — 学习与表达视觉环境

### 3.1 History & Memory

**History Encoding 的演进**：
- **LSTM 隐状态 + attention + auxiliary task**（pre-FM 时代：Tan 2019、Wang 2019、Ma 2019、Zhu 2020）
- **Recurrent state token**：Hong 2021（RecBERT，单个 [CLS] token 作为 history state）、Lin 2022a（variable-length memory bank）。限制：step-by-step 更新，难任意步取用。
- **Sequential multi-modal Transformer**：Pashevich 2021（单视图/步）、**HAMT**（Chen 2021b, panorama encoder + history encoder 的分层结构，支持大规模 instruction-path pre-training）。后续简化：Kamath 2023 用 mean pooling、Qiao 2022 用前视图编码。
- **Text-as-history**：LLM agent 时代，视觉 → textual description，history 变成 image description 序列 + heading/elevation/distance（Zhou 2024b NavGPT-family）。HELPER（Sarch 2023）用 language-program pair 外部记忆 + retrieval-augmented prompting。

### 3.2 Graph-based History

用 topological / grid / semantic / metric map 作为额外 history 结构：
- **Topological**：[[2202-DUET|DUET]] (Chen 2022c)、Deng 2020、Wang 2023b、SOON (Zhu 2021a)
- **Grid map**：Wang 2023g、Liu 2023a
- **Semantic map**：Hong 2023a、Huang 2023a (VL-Map)、Georgakis 2022 (CM2)、Anderson 2019、Irshad 2022
- **Local metric map**：An 2023
- **LLM + map memory**：MapGPT（Chen 2024a，linguistic-formed map 存 topological graph）、MC-GPT（Zhan 2024b，topological map 存 viewpoint/object/spatial relation）

### 3.3 Generalization across Environments

**Pre-trained visual repr**：ResNet → CLIP encoder（Shen 2022 显示 CLIP 对 VLN 涨点）→ video-pretrained encoder（Wang 2022b，时序信息对导航关键）。

**Environment augmentation**：
- **Mixup-style**：EnvEdit（Li 2022b 改外观）、EnvMix（Liu 2021 混房间）、KED（Zhu 2023）、FDA（He 2024a 插值高频特征）
- **Synthesized future views**：Pathdreamer（Koh 2021）、SE3DS（Koh 2023）
- **从 fine-tune 到 pre-training**：LSTM 时代直接 fine-tune 在增强环境上；FM 时代已把增强环境吸收进 pre-training stage（Li & Bansal 2024、Kamath 2023、Chen 2022b、Wang 2023h、Lin 2023b、Guhur 2021a）。核心发现：in-domain pre-trained multi-modal Transformer > 从 Oscar / LXMERT 初始化。

## 4. Human Model — 解读与沟通

### 4.1 Ambiguous Instructions

**问题**：单轮指令里 landmark 可能不可见或多视图重复（VLN-Trans, Zhang & Kordjamshidi 2023）。

**Perceptual context + commonsense 路线**：
- **CLIP-based visual matching**：VLN-Trans（翻译为 sub-instruction）、LANA+（Wang 2023f，CLIP 匹配 landmark tag）、KERM（Li 2023a，retrieval-augmented navigation view knowledge）、NavHint（Zhang 2024b，详细视觉描述 hint dataset）
- **LLM commonsense**：Lin 2024b（开放世界 landmark 共现）、[[2204-SayCan|SayCan]]（把指令分解成 pre-defined actions + affordance weighting）

**Information Seeking 路线**（三个子问题）：
1. **何时问**（Chi 2020）；
2. **问什么**：下一个 action / object / direction（Roman 2020、Singh 2022）；
3. **谁来答**：真人 / 规则 / 神经 oracle（Nguyen & Daumé III 2019 HANNA）

FM 在这三步都可以插入：
- VLN-Copilot（Qiao 2024）：LLM 作 copilot 提供协助；
- Ren 2023：conformal prediction 判断 ask-or-not；
- Chen 2023c：in-context learning；
- Fan 2023b：GPT-3 分解 ground-truth 响应训练 oracle（SwinBert video-LM）；
- mPLUG-Owl 作为 zero-shot oracle（Fan 2023b）；
- Zhu 2021c：self-motivated agent，学 oracle confidence 实现推理时移除 oracle。

### 4.2 Generalization of Grounded Instructions

**Pre-trained text repr**：LSTM → BERT (PRESS, Li 2019b) → VL Transformer (VLN-BERT Majumdar 2020, PREVALENT Hao 2020) → Airbert (Guhur 2021b image-caption pre-train) → CLEAR (Li 2022a cross-lingual) → ProbES (Liang 2022, prompt tuning) → **NavGPT-2** (Zhou 2025, InstructBLIP + Flan-T5 / Vicuna)。

**Instruction synthesis**：
- **Speaker-Follower 时代**：Fried 2018、Tan 2019、Kurita 2020 —— 质量差（Zhao 2021 验证）
- **Marky**（Wang 2022a / Kamath 2023）：multilingual T5 + visual landmark alignment，在 R2R 路径上达到近人类质量
- **PASTS**（Wang 2023c）：progress-aware spatiotemporal speaker
- **SAS**（Gopinathan 2024）：语义 + 结构线索生成富含空间信息的指令
- **SRDF**（Wang 2024c）：iterative self-training 生成器
- **导航中合成**：LANA（Wang 2023e，边导航边描述）、VLN-Trans、Magassouba 2021

## 5. VLN Agent — 推理与规划

### 5.1 Grounding & Reasoning

**Explicit semantic grounding**（大多是 pre-FM 时代的）：motion/landmark 建模、syntactic 信息、spatial relation。FM 时代很少做显式 grounding，Lin 2023a 的 atomic-concept learning 是少数例外。

**Pre-training VLN foundation models**：
- Scene / object grounding pre-training（Lin 2021）
- LOViS（Zhang & Kordjamshidi 2022a）：orientation + visual 双任务
- HOP（Qiao 2022 / 2023a）：history-and-order aware
- Future view prediction（Li & Bansal 2023）：对长路径有效
- Masked path modeling（Dou 2023）
- Entity-aware pre-training（Cui 2023）

### 5.2 Planning

**Graph-based planner**：
- 全局图扩展局部动作空间：Wang 2021、[[2202-DUET|DUET]] (Chen 2022c)、Deng 2020、Zheng 2024b
- 高/低层分层：Gao 2023 (zone → node)、Liu 2023a (grid-level action)
- VLN-CE waypoint predictor：Krantz 2021、Hong 2022、Anderson 2021
- Map-based global planning：CM2 (Georgakis 2022)、An 2024 / 2023、Wang 2023g、Chang 2024、Wang 2022c
- Future waypoint prediction：Wang 2023a / 2024a（视频预测 / 神经辐射场）

**LLM-based planner**：
- **LLM-Planner**（Song 2023）：动态 sub-goal 拆解
- **Mic**（Qiao 2023b）：static + dynamic step-by-step 指令
- **A²Nav**（Chen 2023b）：GPT-3 解析 sub-task
- **ThinkBot**（Lu 2023）：CoT 补全缺失动作
- **VL-Map**（Huang 2023a）：Code-as-Policy style（Liang 2023）+ queryable map
- **SayNav**（Rajvanshi 2024）：3D scene graph → LLM → 高层规划

### 5.3 Foundation Models as VLN Agents

**VLMs as agents**：single-stream VLM 同时吃 text + vision + history token，self-attention 统一建模。代表：Hong 2021、Qi 2021、Moudgil 2021、Zhao 2022；zero-shot：CLIP-NAV（Dorbala 2022）。**VLN-CE 的 waypoint predictor** 是把 DE 方法搬到 CE 的通用桥梁（Krantz 2021、Hong 2022、Anderson 2021、An 2022、Zhang & Kordjamshidi 2024）。

**LLMs as agents**：
- **Zero-shot**：[[2305-NavGPT|NavGPT]]（Zhou 2024a，GPT-4 自主行动）、**MapGPT**（Chen 2024a，topological map → global exploration hint）、**DiscussNav**（Long 2024b，Instruction Analysis / Vision Perception / Completion Estimation / Decision Testing 多专家）、MC-GPT（Zhan 2024b）、InstructNav（Long 2024a，多源 value map）
- **Fine-tuned**：Zheng 2024a、Zhang 2024a、Pan 2024
- **CoT-based**：NavCoT（Lin 2024a，把 LLM 变成 world model + reasoning agent，模拟未来环境来决策）

## 6. Challenges & Future Directions

1. **Benchmarks**：
   - 需要 sim-to-real 统一平台（OVMM Yenamandra 2023）和面向真实人类需求的任务（BEHAVIOR-1K Li 2024a）
   - 动态环境：HAZARD、Habitat 3.0、HA-VLN
   - 从室内到室外：autonomous driving / aerial 正起势
2. **World Model: 2D → 3D**：目前主流是 2D repr，但 VLN 本质是 3D 任务。显式 3D 方向：semantic SLAM、volumetric、BEV grid、CLIP feature 投到 voxel（ConceptFusion）或 top-down map（VL-Map）、scene graph。3D foundation models（3D-LLM Hong 2023b、Huang 2024a、LRM、3D multimodal）是可能的下一步。
3. **Human Model: Instruction → Dialogue**：开放式对话 benchmark 已出现（TtW、RobotSlang、TEACh、SDN、ANDH），但方法仍靠规则模板，situated task-oriented dialogue 和 FM 还没真正结合。
4. **Agent Model: Adapting FM for VLN**：
   - Lack of embodied experience：embodiment foundation model 方向（EmbodiedGPT、[[2303-PaLME|PaLM-E]]、Octopus）
   - **Hallucination**：LLM 可能生成不存在的 object（"在沙发那里左转" 但房间里没沙发）
   - LLM in planning：PlanBench / CogEval 显示 LLM 在复杂规划上有限；VLN 动作空间受限，LLM 更适合做 coarse-grained 指令分解而非主决策者
5. **Deployment: Sim → Real**：perception gap（Wang 2024b 用 semantic map + 3D feature field 让 monocular robot 获得 panoramic perception）+ embodiment gap + data scarcity（robot teleoperation 是规模化数据收集方案，He 2024b）

## 7. 高频引用的 canonical VLN 工作

按 §3–5 中被多次引用的程度，survey 反复强调的 canonical 工作：
- **R2R / Anderson 2018**：VLN 的起点 benchmark，全篇几乎每节都引。
- **[[2202-DUET|DUET]] / Chen 2022c**：graph-based history + planning 的代表。
- **HAMT / Chen 2021b**：panorama + history encoder 的 hierarchical Transformer，多次作为 FM 时代 VLN 的里程碑。
- **PREVALENT / Hao 2020** + **VLN-BERT / Majumdar 2020**：in-domain VL pre-training 起点。
- **RecBERT / Hong 2021**：recurrent state token + single-stream VLM as agent。
- **[[2204-SayCan|SayCan]] / Ahn 2022**：LLM + affordance 的 instruction decomposition。
- **[[2305-NavGPT|NavGPT]] / Zhou 2024a** + **MapGPT / Chen 2024a**：LLM-as-agent zero-shot VLN。
- **Marky / Wang 2022a + Kamath 2023**：instruction synthesis 的 SOTA speaker。
- **VLN-CE / Krantz 2020** + **Robo-VLN / Irshad 2021**：从 DE 到 CE 到 continuous control 的三步。
- **Matterport3D / Chang 2018** + **Habitat / Savva 2019**：两大 indoor 3D simulator。

---

## 关联工作

### 同类 survey（被本篇显式超越的）
- **Gu et al. 2022 (ACL)**：pre-FM 时代 VLN survey，bottom-up（benchmark + method），本篇显式批评其没覆盖 FM。
- **Park & Kim 2023**：类似视角的 VLN survey。
- **Wu et al. 2024**：较新的 VLN 综述但仍以 benchmark 为主。
- **Zhu et al. 2021b / 2022、Zhang 2022a**：visual navigation / mobile robot navigation 综述，语言讨论少。

### 对齐的 framework / motivation
- **Hu & Shu 2023 (LAW framework)**：World model + Agent model 的双模型视角，本 survey 直接借用并扩展为 World / Human / Agent 三分。
- **Andreas 2022**：把 instruction 理解为"建立对说话人的 latent model"，被本篇作为 human model 的 motivation。
- **Ha & Schmidhuber 2018**：World Model 原始论文，作为 world model 定义来源。

### 已在 vault 的 canonical VLN / embodied FM 工作
- [[2202-DUET]]：graph-based VLN，survey §3.1 / §5.2 多次点名
- [[2305-NavGPT]]：LLM-as-agent zero-shot，§5.3 的代表
- [[2204-SayCan]]：LLM + affordance 的指令分解，§4.1 的代表
- [[2303-PaLME]]：embodied FM，§6 Agent Model 讨论的代表

### 与 [[DomainMaps/VLN]] 的关系
- 本 survey 写于 2024-07，截至 2026-04 的发展在 DomainMap 已更新：R2R-CE SOTA 从 survey 写作时的 ~55%（HAMT/DUET）被新一代（ETPNav 57%、NaVILA、PROSPECT 60.3%、Efficient-VLN 64.2%、GTA zero-shot 48.8%）超越；streaming VLA 路线和 VLN-R1 的 GRPO 训练在 survey 里完全没出现。所以本 survey 的"未来方向"部分（2D→3D、embodied FM）基本被验证，但它没预见到 **VLN → VLA unification** 这条路径。

---

## 论文点评

### Strengths

1. **Taxonomy 真正 principled**：LAW 三分法 + FM 4 角色的正交轴给异质的 VLN 工作一个统一坐标系。不是"按时间列 benchmark"，而是"这个工作在解哪个子问题、FM 扮演什么角色"——对 reader 想定位某类方法非常有用。
2. **Benchmark Table 1 是整篇最实用的 artifact**：24 个 benchmark × 4 维属性，是挑对比 benchmark 和理解任务变体的实用 cheat sheet。没有这张表，整个 VLN 子领域的异质性很难迅速把握。
3. **跨 dimension 的 coverage 均衡**：outdoor（TouchDown、CARLA、AerialVLN）、dialogue（CVDN、TEACh）、manipulation（ALFRED）、continuous（VLN-CE）、real-world（RobotSlang）都覆盖，不是只讲 R2R。
4. **作者组合强**：Joyce Chai、Mohit Bansal、Qi Wu、Parisa Kordjamshidi 全是 VLN / embodied 的一线组，引用选择可信。
5. **Challenges §6 清晰区分"benchmark"与"方法"两条 axis**：Benchmark 缺 dynamic environment / sim-to-real / outdoor；方法缺 3D world model / dialogue human model / embodied LLM / hallucination 缓解——未来方向可操作性强。

### Weaknesses

1. **Human model 章节偏薄**：§4 篇幅显著小于 §3 / §5，且大部分 FM 应用还停留在 CLIP 匹配 + GPT 分解指令，缺少对 situated dialogue、theory-of-mind、speaker intent modeling 的深入讨论。这恰好是框架里最有潜力 under-explored 的部分。
2. **没覆盖 2024 Q3 之后的 breakthrough**：streaming VLA（NaVILA、StreamVLN、PROSPECT）把 VLN 重构为连续视觉流 + 端到端控制的方向，本 survey 完全没预见；GRPO-based RL（VLN-R1）和 explicit world representation（GTA）也是后续发展。这是综述时点的自然限制，但意味着到 2026 年它已无法作为唯一入门材料。
3. **缺对 sim-to-real gap 的定量讨论**：§6 讨论了 deployment 挑战但没给出具体数字（sim 和 real 之间 SR / SPL 的 gap 是多少？哪些方法 close 了这个 gap？）。
4. **"VLN Agent 的 4 种 FM 角色" 分类有交叠**：backbone encoder / data generator / planner / agent policy 在实际论文里经常同时出现（e.g., [[2305-NavGPT|NavGPT]] 既是 planner 又是 agent policy），survey 没讨论这种交叠。
5. **评测方法学的讨论缺失**：SR / SPL / nDTW 在 dialogue / manipulation 混合任务上是否仍合适？如何评测 information seeking 的"问题质量"？这些 evaluation 层面的 open question 没被系统讨论。
6. **没讨论 VLN ↔ VLA 的结构同构**：VLN 的 hierarchical planner + waypoint predictor，和 VLA 的 high-level planner + low-level action decoder 在架构上高度对应，但 survey 完全没跨出 VLN 的边界。

### 可信评估

#### Artifact 可获取性
- **代码**：github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models 作为 reading list 仓库，持续更新（90d 内 3 次 commit，262⭐）；survey 本身不含代码。
- **模型权重**：N/A（survey）
- **训练细节**：N/A（survey）
- **数据集**：未在 survey 中发布新 dataset；所有引用 benchmark 的链接列在 GitHub。

#### Claim 可验证性
- ✅ "LAW 框架可以统一组织 VLN 工作"：可通过阅读全文和 Table 1 验证——24 个 benchmark 确实可映射到 world × human × agent 三维。
- ✅ "R2R / Matterport3D 是 VLN 的 de facto 起点"：通过引用分布可验证。
- ⚠️ "foundation model 时代的 VLN 有 4 种 FM 角色"：分类本身合理，但边界模糊（见 Weakness 4），不是严格的 partition。
- ⚠️ "in-domain pre-trained multi-modal Transformer > 通用 VLM 初始化（如 LXMERT / Oscar）"：是 survey 引用 HAMT / DUET 等结果得出，但没给出在 Matterport3D vs 非 Matterport3D 环境下的详细对比。
- ⚠️ "LLM 在 VLN 里更适合做 coarse-grained 指令分解而非主决策"：survey 的直觉正确但缺数据支撑，仅引 PlanBench / CogEval 做泛化论证。

### Notes
- Survey 的 **LAW 框架在 VLN 里很自然，但不直接迁移到 manipulation / VLA**——那里没有独立的 "human model"（指令通常更短、对话更少）。如果要用类似三分法组织 VLA survey，需要重新定义第二个模块——可能是 "Task Model" 或 "Skill Model"？可以作为 VLA-Survey 重构时的参考。
- Survey 没有给出 VLN-CE 下 SOTA 在 sim vs real 的 gap 表，但 DomainMap/VLN.md 里已补充：NaVILA 54% sim → 88% real（不同 setup），GTA 48.8% zero-shot sim。可以把这部分作为 VLN DomainMap "sim-to-real" section 的 seed。
- Survey 提到 **VLN-CE → Robo-VLN → waypoint predictor** 的三步演进，这正是现在 streaming VLA 取代 waypoint 路线的前置——survey 写作时 waypoint 还是 default，但 2025-26 streaming VLA 已尝试 skip 这一层。
- **连接到 [[Topics/VLN-VLA-Unification]]**：survey 把 VLN 作为孤立任务，没有预见 VLN-VLA 结构收敛；这正是该 topic 的切入点。

### Rating

**Metrics** (as of 2026-04-23): citation=79, influential=5 (6.3%), velocity=3.59/mo (22 months since publish); HF upvotes=0; github 262⭐ / forks=13 / 90d commits=3 / pushed 6d ago.

**分数**：2 - Frontier

**理由**：作为 LLM 时代 VLN 的第一本系统 survey，LAW taxonomy 和 Table 1 benchmark 分类是有结构性贡献的参考资料（不只是 reading list），在 VLN 子社区是相对核心的引用源（22 个月 79 引、3.6/mo velocity 对于 survey 是健康的，但 influential 比例只有 6.3% 偏低，反映被当 landmark reference 引而非技术继承）。但不到 Foundation 级：(a) influential citation 绝对数仅 5，说明没有成为新工作的结构性基础；(b) taxonomy 覆盖到 2024-07 为止，streaming VLA / VLN-R1 / 3D EWR 等 2025-26 关键路线未覆盖，结构性框架开始老化；(c) Human model 章节偏薄、与 VLA 的结构联动未讨论，作为 VLN 入门 reader 会有遗漏。高于 Archived 因它仍是理解 FM-era VLN benchmark 异质性和 pre-2024 方法脉络的最高效入口。
