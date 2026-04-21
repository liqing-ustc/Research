---
title: "ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents"
authors: [Fei Tang, Zhiqiong Lu, Boxuan Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen]
institutes: [Zhejiang University]
date_publish: 2026-04-13
venue: arXiv
tags: [gui-agent, agentic-RL, computer-use]
paper: https://arxiv.org/abs/2604.11784
website: https://zju-real.github.io/ClawGUI-Page/
github: https://github.com/ZJU-REAL/ClawGUI
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents
> - **核心**: 一个三模块开源 full-stack 框架（RL 训练 + 标准化评测 + 真机部署），主张 GUI agent 的瓶颈是 infrastructure 而非 modeling capacity。
> - **方法**: ClawGUI-RL（GiGPO + PRM 步级 dense reward，支持 Docker emulator 与真机）+ ClawGUI-Eval（Infer→Judge→Metric 三段式 pinned pipeline）+ ClawGUI-Agent（CLI/GUI 混合控制 + 12+ chat 平台 + 个性化记忆）。
> - **结果**: ClawGUI-2B 在 MobileWorld GUI-Only 上 17.1% SR，超越同尺寸 MAI-UI-2B（11.1%）与更大的 UI-Venus-72B（16.4%）；评测 95.8% 复现率（46/48 cells）。
> - **Sources**: [paper](https://arxiv.org/abs/2604.11784) | [website](https://zju-real.github.io/ClawGUI-Page/) | [github](https://github.com/ZJU-REAL/ClawGUI)
> - **Rating**: 2 - Frontier（infrastructure-first 的论点有 controlled comparison 支撑，eval pipeline 和 inference predictions 发布是稀缺 contribution；但 17.1 SR 未进 SOTA region，真机 RL 仅 capability 声明，长期价值取决于社区采纳）

**Key Takeaways:**
1. **Infrastructure-first 论点**：作者主张当前 GUI agent 的进步瓶颈是缺少端到端的 training/eval/deploy 基础设施，而不是某个孤立的 modeling 改进——这个论断被 ClawGUI-2B 与 MAI-UI-2B 共享 base weights、纯靠 pipeline 拉到 +6 absolute pts 所支持。
2. **Step-level credit assignment 的工程化**：把 GiGPO 的 anchor-state grouping 与一个 Qwen3.5-72B PRM 串起来，给长 horizon GUI 任务提供 dense per-step reward，比 episode-level GRPO 有 +2.6 abs / +17.9% rel 的提升。
3. **Eval reproducibility 是 infrastructure 问题**：同一套 pipeline、pinned 配置即可对 6 benchmarks × 11+ models 达到 95.8% 复现率；两个失败 case 都是 official config 未公开。
4. **Real-device RL training**：声称是首个开源、在物理 Android 设备上 validated 的 GUI agent RL infra；但论文实际报的训练实验仍主要在 64 个 Docker emulator 上。

**Teaser. ClawGUI 三模块总览图——左 RL 训练（虚拟+真机环境），中 Eval（Infer→Judge→Metric 三段），右 Agent（多 chat 平台 + 跨 OS 真机部署）。**

![](https://arxiv.org/html/2604.11784v1/x1.png)

---

## ClawGUI-RL: Scalable Online RL Training

ClawGUI-RL 把所有 device backend 抽象在统一接口后面，使 emulator 和真机能在同一训练 loop 里互换。

**Figure 2. ClawGUI-RL 架构：Environment Manager 负责跨 real device / Android emulator 的多任务并行 rollout，含健康检查、crash recovery 与 spare server rotation；任务评估同时用 system-level verification 与 MLLM-as-judge。**

![](https://arxiv.org/html/2604.11784v1/x2.png)

### Environment Manager

**Virtual Environment**：基于 MobileWorld 拉起几十个 Docker Android emulator，四阶段生命周期：
- **Task Reset**：每 episode 开始重置 device state、加载新任务。
- **Task Evaluation**：emulator 暴露 system-level root，可直接读 app state / DB；再叠 MLLM-as-judge 看终态截屏。
- **Spare Server Rotation**：维护 spare server 队列；container 不健康时自动 rotate，被影响的 task 不中断训练。
- **Teardown**：周期性重启容器防 state 累积。

**Real Device Training**：在物理 Android 或云手机上训练，差异主要两点：
- **Task Source**：不能 procedurally 生成，得人工 author 任务。
- **Task Evaluation**：没有 root，只能靠 MLLM-as-judge 判终态。

> ❓ "validated support for real physical devices" 是 ClawGUI-RL 的关键 selling point，但 §4 只报了 64 个 emulator 上的训练；真机训练只描述了 capability，没看到对应的训练曲线或 ablation。

### Reward Design

两层 reward：

**Equation. 总 reward**

$$R = R_{\text{outcome}} + R_{\text{step}}$$

- $R_{\text{outcome}}$：episode 末的 binary 0/1。
- $R_{\text{step}}$：PRM 接收 (前一截屏, 当前截屏, action 历史)，判断当前 action 是否对完成任务有意义贡献。实际用 Qwen3.5-72B 当 PRM。

### RL Trainer

基于 verl + verl-agent，开箱支持 Reinforce++/PPO/GSPO/GRPO/GiGPO。论文重点对比 GRPO vs GiGPO：

- **GRPO**：episode-level 优势——同任务的 group 内 normalize return；问题是 trajectory 内每步分到同样 advantage，4-step 完成与 8-step 完成的 rollout reward 一样。
- **GiGPO**：两层 hierarchical advantage——episode level 保留 macro 相对优势；step level 用 anchor-state grouping，把不同 rollout 中遇到相同中间 state 的 step 聚成 sub-group，在 sub-group 内用 discounted return normalization 估 micro advantage。不需要 value network，也不需要额外 rollout。

---

## ClawGUI-Eval: Reproducible GUI Evaluation

**Figure 3. ClawGUI-Eval 的 Infer → Judge → Metric 三段流水线，覆盖 6 benchmarks × 11+ models，整体达到 95.8% reproduction rate。**

![](https://arxiv.org/html/2604.11784v1/x3.png)

### Coverage

- **Benchmarks (6)**：[[2504-ScreenSpotPro|ScreenSpot-Pro]]、ScreenSpot-V2、UI-Vision、MMBench-GUI、OSWorld-G、AndroidControl。
- **Models (11+)**：Qwen3-VL、Qwen2.5-VL、[[2501-UITARS|UI-TARS]]、MAI-UI、GUI-G²、UI-Venus、GUI-Owl、StepGUI、Gemini、Seed 1.8 等。

### Pipeline

三个阶段解耦——单阶段可独立重跑：
- **Infer**：本地 transformers GPU 或 OpenAI-compatible API；多 GPU 自动多进程 + shard-level checkpointing。
- **Judge**：benchmark-specific——标准 grounding 用 point-in-box、OSWorld-G 用 polygon + refusal-aware、AndroidControl 用 multi-action judge。
- **Metric**：按 platform / element type / task category breakdown。

最大的卖点：**inference predictions 也一并发布**，下游可以仅 re-judge 而不重跑昂贵 inference。

### 复现率结果

**Table 3 节选. 各模型 official vs ClawGUI-Eval 复现的结果（Δ ≤ 2% 或 ≥ official 视为成功）**

| Model | SS-Pro Off. | SS-Pro Ours | SS-V2 Off. | SS-V2 Ours | OSW-G Off. | OSW-G Ours |
| --- | --- | --- | --- | --- | --- | --- |
| GUI-Owl 1.5-8B | 71.10 | 70.08 | 93.70 | 93.55 | 65.80 | 64.12 |
| Qwen3-VL-8B | 54.60 | 56.42 | – | 94.26 | – | 65.88 |
| UI-Venus 1.5-8B | 68.40 | 67.68 | 95.90 | 95.83 | 69.70 | 69.98 |
| MAI-UI-8B | 65.80 | 64.07 | 95.20 | 94.34 | 60.10 | 63.23 |
| Gemini 3.0 Pro (Zoom) | 72.70 | 75.08 | – | – | – | – |

整体 95.8%（46/48 cells）。两个失败 case（Qwen3-VL-2B 与 UI-TARS 1.5-7B 的 SS-Pro）都是 official 没公开 eval 配置——指向 prompt/resolution 的 silent drift 是不可复现的主因。Closed-source 模型用 Zoom（two-stage crop-then-ground，Gemini 25% / Seed 50% crop tile）恢复 official 性能。

---

## ClawGUI-Agent: Personal GUI Assistant

**Figure 4. ClawGUI-Agent 总览：用户经 12+ chat 平台下自然语言指令，server 端 message-driven agent loop 用 persistent memory + skill 控制 phone / browser / desktop 等真机或虚拟设备。**

![](https://arxiv.org/html/2604.11784v1/x4.png)

### Hybrid CLI-GUI Control

论点：CLI 精确高效但很多 app 没有 programmatic 接口、且对用户不可观测；GUI 通用但更慢。ClawGUI-Agent 走混合：能 CLI 就 CLI，否则 fallback 到 GUI。

### Personalized Memory

任务执行时自动抽结构化事实（联系人关系、常用 app、用户习惯偏好），存为 vector embedding；后续任务 top-k 检索注入 system context；duplicate 自动 merge 而非累积。

### Deployment Modes

- **Remote control**：通过 Feishu/DingTalk/Telegram/Discord/Slack/QQ 等 12+ 平台从异地设备发指令控制目标手机。
- **Local control**：在手机上某 chat 应用直接发指令，agent 接管本地。

### Eval as Skill

ClawGUI-Eval 被暴露为 agent 的 built-in skill：自然语言 "benchmark Qwen3-VL on ScreenSpot-Pro" 即触发完整评测流水线，无需写脚本。

---

## Experiments

**Setting**：ClawGUI-2B 基于 MAI-UI-2B，64 并行虚拟环境，8×A6000(48GB)，GiGPO，rollout group=8，T=0.7，lr=1e-6，3 epoch，bsz=8；PRM 用 Qwen3.5-72B；MobileWorld GUI-Only 117 task，max 50 step。

### Main Results

**Table 1. MobileWorld GUI-Only Success Rate (117 tasks)**

| Model | SR (GUI-Only) |
| --- | --- |
| Claude-4.5-Sonnet + UI-Ins-7B (agentic) | 47.8 |
| Gemini-3-Pro + UI-Ins-7B (agentic) | 55.6 |
| GPT-5 + UI-Ins-7B (agentic) | 54.0 |
| GUI-Owl-32B | 8.5 |
| UI-Venus-72B | 16.4 |
| Qwen3-VL-235B-A22B | 12.8 |
| Doubao-1.5-UI-TARS | 26.3 |
| MAI-UI-2B (baseline) | 11.1 |
| MAI-UI-8B | 19.7 |
| **ClawGUI-2B (ours)** | **17.1** |

观察：
1. ClawGUI-2B 与 MAI-UI-2B 共享 base weights，差距 +6 abs（+54% rel）来自 pipeline。
2. ClawGUI-2B 反超 UI-Venus-72B / Qwen3-VL-32B 等更大模型（在 end-to-end track），但仍远低于 agentic framework 的 47–55 区间——后者用 frontier 闭源 planner，被作者明确划为 separate regime。
3. Doubao-1.5-UI-TARS（26.3）仍领先 ClawGUI-2B 9.2 abs，作者没正面对比这一项。

### Ablation: GRPO vs GiGPO

**Table 2. Reward design ablation on MobileWorld GUI-Only**

| Method | Reward Type | SR (%) |
| --- | --- | --- |
| GRPO | Binary (episode-level) | 14.5 |
| GiGPO | Dense (episode- & step-level) | 17.1 |

+2.6 abs / +17.9% rel——验证 dense step-level credit assignment 的价值。

> ❓ 这个 ablation 没有把 PRM 和 GiGPO 的 anchor-state grouping 解耦——单独切掉 PRM 而保留 GiGPO 的 hierarchical advantage 会怎样？无法判断收益主要来自哪边。

---

## Demo Videos

**Video 1. ClawGUI-Agent 通过自然语言控制真机完成多步任务**

<video src="https://zju-real.github.io/assets/clawgui-agent.mp4" controls muted playsinline width="720"></video>

**Video 2. ClawGUI-RL 在并行 Android 环境中做 online RL 训练**

<video src="https://zju-real.github.io/assets/clawgui-rl.mp4" controls muted playsinline width="720"></video>

---

## Discussion 中值得关注的方向

- **Unified GUI-CLI harness**：作者把 ClawGUI 视作向 CLI/GUI/API 统一可互换 action space + 学习路由策略 的早期一步，对标 Claude Code、Hermes Agent、MiniMax M2.7 的 harness 设计。
- **Scaling RL beyond emulator**：两个方向——code-gen 重建的 mock app（绕开 auth）；on-device RL with privacy-preserving trajectory。声称这是 systems 而非 algorithm 问题。
- **On-device system agent**：与 Hermes Agent 的 Android 控制、Gemma 4 的 2B 移动模型一同被引为 trend。
- **GUI World Model**：作者把 ClawGUI-RL 的 dense step-level 轨迹日志称为训练 GUI world model 的天然 substrate，引了 code2world / VimoGen / Genie 3 这条线。

---

## 关联工作

### 基于
- **GiGPO**：核心 RL 算法，提供 anchor-state grouping 的 hierarchical advantage estimation。
- **MobileWorld**：虚拟环境与 117 GUI-Only 评测 split 的供给方。
- **MAI-UI-2B**：ClawGUI-2B 的 base weight 与同尺寸 baseline。
- **verl / verl-agent**：RL trainer 底层。

### 对比
- **MobileGUI-RL / [[2508-ComputerRL|ComputerRL]] / [[2509-UITARS2|UI-TARS-2]] / UI-Venus-1.5**：同期 online RL for GUI agent 工作，但训练 infra 均闭源。
- **OpenClaw / Hermes Agent / Claude Code**：CLI-based agent harness，作者主张 ClawGUI-Agent 的 hybrid CLI-GUI 是其自然演化。

### 方法相关
- **Process Reward Model (PRM)**：dense step-level supervision 的关键构件，由 Qwen3.5-72B 充当 judge。
- **MLLM-as-judge**：真机训练唯一可用的 reward signal。
- **[[2504-ScreenSpotPro|ScreenSpot-Pro]] / ScreenSpot-V2 / OSWorld-G / AndroidControl**：评测 benchmark suite。

---

## 论文点评

### Strengths

1. **问题诊断切中要害**：明确把 reproducibility crisis、closed-RL infra、broken deployment loop 作为三大 gap，比单纯做某个 SOTA model 的论文更具系统视角。
2. **Eval 这块的"释放 inference predictions"是真 contribution**：让下游 re-judge 而不必重跑昂贵推理，这是社区少有的做法，长期价值高于短期 SOTA。
3. **同 base、同尺寸的 controlled comparison（MAI-UI-2B → ClawGUI-2B）让 +6 abs 的 attribution 比较干净**，相对 MAI-UI-2B 的 11.1 → 17.1。
4. **GiGPO + PRM 的组合**作为 step-level credit assignment 的工程方案，提供了一个具体落地的 recipe。

### Weaknesses

1. **"首个开源真机 RL infra"的 claim 与论文实验 mismatch**：所有训练数据都跑在 emulator 上，真机部分只描述能力但没有训练实验或 ablation。
2. **GiGPO vs GRPO 的 ablation 没有解耦 PRM 贡献**：dense reward (PRM) 与 hierarchical advantage (GiGPO) 是两个变化叠在一起的。
3. **17.1 SR 在绝对值上离 agentic framework（47-55）和 Doubao-1.5-UI-TARS (26.3) 都还有显著差距**，论文回避了与 Doubao 的同尺寸正面对比（虽然 Doubao 不是 2B）。
4. **3 个组件耦合度未充分论证**：RL/Eval/Agent 单独都有价值，但"必须放在一起"的论证主要是叙事性的（"deployable skill"），缺少 quantitative 证据。
5. **Personalized memory 系统几乎没有任何 evaluation**——只描述设计，没有 ablation 看它实际在 task SR 或 user study 上带来多少提升。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training，Apache 2.0，三模块（clawgui-rl / clawgui-eval / clawgui-agent）均在主 repo。
- **模型权重**: ClawGUI-2B 已发布在 HuggingFace（SugarVapeur/OpenGUI-2B）与 ModelScope（SugarFree/OpenGUI-2B）。
- **训练细节**: 关键超参完整（GPU 数、并行环境数、lr、bsz、temperature、epoch、PRM 模型），但训练数据规模、PRM 提示词模板、GiGPO 具体超参（如 anchor 阈值、discount）未在正文披露，需查 repo。
- **数据集**: MobileWorld（开源）；真机训练任务集说明为 human-authored，但是否随 repo 一起发布未在论文中说明。

#### Claim 可验证性
- ✅ "ClawGUI-2B 17.1 SR vs MAI-UI-2B 11.1"：Table 1 实测，可由开源 weights + ClawGUI-Eval 复现。
- ✅ "95.8% reproduction rate across 6×11+"：Table 3 给出 cell-level 数据，inference predictions 已发布，可独立 re-judge。
- ⚠️ "First open-source GUI agent RL infrastructure with validated support for real physical devices"：infra 开源属实，但"validated on real device"的实证未在论文中给出训练曲线或可量化指标。
- ⚠️ "GiGPO 带来 2.6% 绝对提升"：未与 PRM-only 对照，归因不严。
- ⚠️ "12+ chat platforms + 3 OS 部署"：定性描述，缺少跨平台稳定性 / 端到端任务成功率的实测。

### Notes

- 这篇 paper 的真正长期价值不在于 ClawGUI-2B 的那 17.1，而在于 **inference predictions 的释放** + **eval pipeline pinned configuration**——如果社区真的开始用它做 cross-paper 比较，可以解决 GUI 领域臭名昭著的"3% 提升不知道来自方法还是 prompt"问题。
- "Infrastructure as the bottleneck"是个被低估的 framing——同 base + 同尺寸纯靠 pipeline 拉到 +54% rel 是相当 strong 的 evidence，值得借鉴到其他 agent 子领域的 narrative 中。
- 真机 RL training 这块更多是 capability 声明而非 evidence——下个版本如果能给出真机训练的具体 reward 曲线、unhealthy device 处理统计，会显著加分。
- ClawGUI-Eval 的"Eval as Skill"想法（自然语言触发完整 benchmark 流水线）是个有趣的 deployment pattern——把 evaluation pipeline 本身视为一种 agent skill 而非外部脚本，可能影响未来 agent harness 的设计。

### Rating

**分数**：2 - Frontier
**理由**：根据 Strengths 里 Eval pipeline 发布 inference predictions 与同 base controlled comparison +6 abs 这两点，这不是一次性 incremental 工作——infrastructure-first framing 对 GUI agent 领域是一个有价值的参考范式。但按照 Weaknesses，17.1 SR 仍被 Doubao-1.5-UI-TARS (26.3) 与 agentic framework (47–55) 显著领先，真机 RL 与 Agent 三组件协同也无 quantitative 证据，尚未达到必读/奠基工作的门槛，因此不到 3 - Foundation；长期价值取决于社区是否真正采纳其 eval pipeline 做 cross-paper 比较。
