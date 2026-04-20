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
rating: 1
date_added: 2026-04-20
---
## Summary

> [!summary] ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents
> - **核心**: 一个覆盖 GUI agent 全生命周期的开源 harness——在线 RL 训练（ClawGUI-RL）+ 标准化评测（ClawGUI-Eval）+ 真机部署（ClawGUI-Agent），三件套集成在一个仓库里
> - **方法**: RL 侧用 GiGPO + Process Reward Model 做 step-level credit assignment，环境层抽象统一 Docker 模拟器和真机；Eval 侧用 Infer→Judge→Metric 三段式 pipeline 并 pin 住每个模型的配置；Agent 侧 hybrid CLI-GUI 控制 + 12+ 聊天平台接入 + 向量化个性化 memory
> - **结果**: ClawGUI-2B 在 MobileWorld GUI-Only 上拿到 17.1% SR，同规模 MAI-UI-2B 是 11.1%（相对 +54%）；ClawGUI-Eval 对 6 个 benchmark × 11+ 模型达到 95.8% 复现率；GiGPO 相对 GRPO 带来 +2.6 绝对 / +17.9% 相对的 SR 提升
> - **Sources**: [paper](https://arxiv.org/abs/2604.11784) | [website](https://zju-real.github.io/ClawGUI-Page/) | [github](https://github.com/ZJU-REAL/ClawGUI)

**Key Takeaways:**
1. **Infrastructure-first 的系统论文**: 这不是一篇模型论文，而是一篇"工程缺位"论文——作者的核心 claim 是 GUI agent 社区的瓶颈不在模型，而在缺少 open-source full-stack 训练+评测+部署 harness
2. **GiGPO + PRM 是唯一的算法贡献**: 在 GRPO 基础上做 two-level（episode-level 宏观优势 + 基于 anchor-state clustering 的 step-level 微观优势）的 credit assignment，配合 process reward model 缓解 long-horizon GUI 任务的 reward 稀疏
3. **95.8% 复现率的诊断价值**: 在 ScreenSpot-Pro / V2 / UIVision / MMBench-GUI / OSWorld-G 上跑 11+ 个模型，48 个 cell 里 46 个落在 `|Δ|≤2%`，失败的 2 个（Qwen3-VL-2B、UI-TARS 1.5-7B）都是官方评测配置未公开——把"GUI 评测无法对齐"从业界玄学降级为可归因的 infra 问题
4. **规模对比有点作弊**: 论文用 ClawGUI-2B 打赢 UI-Venus-72B 和 Qwen3-VL-32B 这种叙事看起来很猛，但对比的是"同规模 SFT baseline + online RL"和"大模型纯 SFT"，并非在同一训练协议下的公平比较

**Teaser. 三模块框图——RL 训练 / Eval 评测 / Agent 部署串成一条 pipeline**

![](https://arxiv.org/html/2604.11784v1/x1.png)

---

## 1. Motivation：为什么 GUI Agent 社区需要 "harness"

作者把 GUI agent 研究的瓶颈诊断为**三个工程空洞**，而非模型能力不足：

1. **训练生态封闭**：UI-TARS-2、UI-Venus-1.5、MAI-UI 等在线 RL 工作都声称强结果，但没有一个开源训练基础设施；且所有工作仅在 emulator 里训，真机训练"在 open literature 里基本未被探索"
2. **评测严重漂移**：prompt 格式、坐标归一化、图像分辨率、采样温度任何一项变动都能让 [[2504-ScreenSpotPro|ScreenSpot-Pro]] 分数上下浮动几个点，而这些选择通常未披露——于是"+2% SOTA"无法区分是真进步还是配置差异
3. **部署死循环**：研究 agent 几乎不进用户手里；CLI harness（OpenClaw / Claude Code）精确但覆盖面窄（很多 app 根本没有 API），而 GUI harness 跨平台持久化部署开源的很少

这三个缺口共同构成 ClawGUI 的 motivation——不是做一个更强的 GUI 模型，而是做一套公共基础设施。**这是一个产品 / infra 论文的立论方式**，不是研究论文的立论方式。

> ❓ 一个 infra 论文的价值判断应该看"社区采纳度"而不是 leaderboard 数字。发布当下无法判断，需要看 6-12 个月后有多少论文真的在 ClawGUI-RL 上训新模型、在 ClawGUI-Eval 上报数。

---

## 2. ClawGUI-RL：在线 RL 训练基础设施

**Figure 2. ClawGUI-RL 架构——RL infrastructure + 统一环境后端（虚拟 / 真机）**

![](https://arxiv.org/html/2604.11784v1/x2.png)

### 2.1 Environment Manager

核心设计是**把虚拟环境和真机抽象在同一接口后面**，训练 loop 不需要知道跑的是 Docker emulator 还是物理手机。

**Virtual Environment**（基于 MobileWorld）：
- **Task Reset**: 每个 episode 开头重置设备状态加载新任务
- **Task Evaluation**: 虚拟环境有 root，通过直接检查 app state 和 database record 做系统级成功判定；再叠加一个 MLLM-as-judge 看最终截图与任务指令
- **Spare Server Rotation**: 长训练过程中 emulator container 容易变 unhealthy（卡死 / 崩溃），作者维护 spare server queue，检测到不健康就 rotate，受影响的 task 不中断训练——这是作者强调的一个工程 pain point
- **Teardown**: 周期性重启 container 防状态累积

**Real Device Training**：直接在物理 Android 或云手机上训，两个新挑战：
- **Task Source**: 真机任务不能程序生成，需要人工编写可执行可验证的任务
- **Task Evaluation**: 没有 root，只能靠 MLLM-as-judge 判断最终截图

> 真机训练的价值 claim 很重——但论文里没说 ClawGUI-2B 是不是在真机上训的，从 experiment setup 看（64 parallel virtual environments on 8 × A6000）**ClawGUI-2B 完全是在虚拟环境里训的**。真机训练只是框架"支持"，不是 flagship 模型实际使用的路径。

### 2.2 Reward Design：Binary Outcome + PRM Step Reward

**Binary Outcome Reward**：episode 结束打 1/0。long-horizon GUI 任务下 reward 极度稀疏，中间步骤几乎无 signal。

**Dense Step-Level Reward via PRM**：每一步之后，PRM 接收 (前一屏截图, 当前屏截图, 历史动作序列)，判断当前 action 是否对任务完成有贡献，输出一个 per-step 分数。总 reward：

$$
R = R_{\text{outcome}} + R_{\text{step}}
$$

实现上用 Qwen3.5-72B 作为 PRM 的 judge 模型。

> ❓ PRM 的质量决定整个 dense reward 的上限。论文完全没讨论 PRM 本身的准确率、false positive rate、或者 PRM 给 reward hacking 留的空间——一个靠 Qwen3.5-72B zero-shot 看截图判断"有无 meaningful contribution"的 judge，很难想象其判断稳定到足以作为优化目标。这是方法最薄的地方。

### 2.3 RL Trainer：GRPO vs GiGPO

底层基于 `verl` + `verl-agent`，支持 Reinforce++/PPO/GSPO/GRPO/GiGPO。论文重点对比 GRPO vs GiGPO。

**GRPO 的局限**：对同一 task 的一组 rollout 做 return 归一化，但给整条 trajectory 分配统一的 episode-level advantage。两条完成同一任务但步数分别为 4 和 8 的轨迹获得相同信号——**没法区分步骤级别的效率**。

**GiGPO** 的 two-level hierarchical advantage estimation：
- **Episode level**: 保留宏观跨 trajectory 的相对优势
- **Step level**: 引入 **anchor-state grouping**——不同 rollout 中遇到相同中间环境状态的 step 被 retroactively 聚成 sub-group，再在 sub-group 内用 discounted return normalization 算 micro advantage

不需要学习 value network，也不需要额外 rollout。这是论文最具体的算法点。

---

## 3. ClawGUI-Eval：可复现的 GUI 评测

**Figure 3. ClawGUI-Eval 的 Infer→Judge→Metric 三段式 pipeline，覆盖 6 个 benchmark × 11+ 模型**

![](https://arxiv.org/html/2604.11784v1/x3.png)

### 3.1 Coverage

- **Benchmarks**: [[2504-ScreenSpotPro|ScreenSpot-Pro]], ScreenSpot-V2, UI-Vision, MMBench-GUI, OSWorld-G, AndroidControl
- **Models**: Qwen3-VL, Qwen2.5-VL, UI-TARS, MAI-UI, GUI-G², UI-Venus, GUI-Owl, StepGUI, Gemini, Seed 1.8

### 3.2 Pipeline：Infer / Judge / Metric 三段解耦

- **Infer**: 支持本地 GPU（transformers）和 remote API（OpenAI-compatible endpoint）双后端，多 GPU 并行由 Python multiprocessing 处理（每进程绑定一个 GPU），shard-level checkpoint 支持断点续跑
- **Judge**: 每个 benchmark 有专门的 judge——标准 GUI grounding 用 point-in-box，OSWorld-G 用 polygon + refusal-aware，AndroidControl 用 multi-action judge
- **Metric**: per-sample label 聚合成按平台 / UI 元素类型 / 任务类别细分的精度

**解耦带来的实用价值**: 可以单独 re-run 任一 stage——例如用更新的 parser 重跑 Judge 而不必重新执行昂贵的 Infer。

### 3.3 核心实验：95.8% 复现率

**Reproduction rule**: 复现值达到或超过官方值，或绝对差 ≤ 2%，计为成功。

**结果**（Table 3 摘要，完整表格见原文）:
- 总体 46/48 cells 成功复现（95.8%）
- Open-source 模型 95.7%，closed-source 在 ScreenSpot-Pro 上 100%
- **两个失败 case**: Qwen3-VL-2B 在 SS-Pro 上（官方 48.50 → 复现 43.90）、UI-TARS 1.5-7B 在 SS-Pro 上（49.60 → 42.06），两者**官方评测配置均未公开**——作者把这个归因为"undisclosed prompt or resolution choices are the primary driver of irreproducibility"
- **Closed-source 模型**（Gemini 3 Pro、Seed 1.8）用 Zoom 两阶段 crop-then-ground 策略（Gemini 25% crop tiles，Seed 50% crop tiles）成功复现官方数字

> 这个 95.8% 的数字本身是有诊断价值的：它说明 GUI 评测的不可复现**不是**模型行为不稳定或算法复杂，**而是**配置未披露。反过来说，只要大家都走同一个 pin-down 的 eval harness，这个问题就能被 infra 消掉。

---

## 4. ClawGUI-Agent：真机部署与个性化

**Figure 4. ClawGUI-Agent——用户从 12+ 聊天平台发指令，server 端 message-driven agent loop 带持久化 memory 和 skill，控制真机 / 虚拟设备跨手机/浏览器/桌面**

![](https://arxiv.org/html/2604.11784v1/x4.png)

### 4.1 Hybrid CLI-GUI 控制

论文的核心 argument：**CLI 和 GUI 都不够**。
- CLI 精确高效但覆盖面窄（很多 app 无 API）、对用户不透明、无法观察干预
- GUI 覆盖面广可解释，但代价高（CLI 一步搞定的 GUI 要几步）

ClawGUI-Agent 的策略：接口支持 CLI 就走 CLI，否则 fallback 到 GUI。

### 4.2 Personalized Memory

- 执行任务时从交互抽取 structured facts（联系人、高频 app、用户习惯偏好）
- 存为 vector embedding
- 后续任务检索 top-k 最相似 memory 注入 system context
- 重复 memory 合并而非累积

### 4.3 部署模式

- **Remote control**: 通过 Feishu / DingTalk / Telegram / Discord / Slack / QQ 等 12+ chat platform 从一台设备发指令控制目标手机
- **Local control**: 直接从手机本地 chat app 下指令，agent 接管本机
- **ClawGUI-Eval as skill**: 自然语言指令（如 "benchmark Qwen3-VL on ScreenSpot-Pro"）即可触发完整评测流程，不用写脚本

---

## 5. Experiments

### 5.1 Setup

- **Base model**: MAI-UI-2B
- **Infra**: 64 parallel virtual environments on 8 × A6000 (48GB)
- **Algorithm**: GiGPO，rollout group size 8，temperature 0.7，lr 1e-6，3 epochs，train batch size 8
- **PRM judge**: Qwen3.5-72B
- **Eval**: MobileWorld GUI-Only split（117 tasks，纯视觉 GUI 控制），max 50 steps

### 5.2 Main Results：ClawGUI-2B vs 其他模型

**Table 1. MobileWorld GUI-Only (117 tasks) Success Rate**

| Model | SR (%) |
| --- | --- |
| *Agentic Framework* | |
| Claude-4.5-Sonnet + UI-Ins-7B | 47.8 |
| Gemini-3-Pro + UI-Ins-7B | **55.6** |
| GPT-5 + UI-Ins-7B | 54.0 |
| *End-to-End Model* | |
| GUI-Owl-7B | 7.7 |
| GUI-Owl-32B | 8.5 |
| UI-Venus-7B | 8.5 |
| UI-Venus-72B | 16.4 |
| Qwen3-VL-8B | 9.4 |
| Qwen3-VL-32B | 11.9 |
| Qwen3-VL-235B-A22B | 12.8 |
| Doubao-1.5-UI-TARS | 26.3 |
| MAI-UI-2B | 11.1 |
| MAI-UI-8B | 19.7 |
| *Ours* | |
| **ClawGUI-2B** | **17.1** |

作者强调三点：
1. **Infrastructure drives policy quality**: ClawGUI-2B 与 MAI-UI-2B 共享 base weight，17.1 vs 11.1 的 gap 完全来自 RL 训练
2. **Small well-trained > large untrained**: ClawGUI-2B 超过 Qwen3-VL-32B (11.9%) 和 UI-Venus-72B (16.4%)
3. **Agentic Framework 另算一档**: Gemini-3-Pro + UI-Ins-7B 达到 55.6%，但依赖闭源 planner，不与 end-to-end 小模型直接可比

> 对比 3 里 Doubao-1.5-UI-TARS 26.3% 这条**刻意没被作者讨论**——它也是 end-to-end model，规模未披露但大概率比 2B 大。不过 17.1% 胜过 UI-Venus-72B 这条比较本身是 honest 的。

### 5.3 Ablation：Reward Design

**Table 2. MobileWorld GUI-Only, reward design ablation**

| Method | Reward Type | SR (%) |
| --- | --- | --- |
| GRPO | Binary (episode-level) | 14.5 |
| GiGPO | Dense (episode- & step-level) | **17.1** |

+2.6 绝对 / +17.9% 相对。作者作为"dense step-level supervision 关键"的证据。

> 严格讲这个 ablation 把**两件事捆在一起**变了：GRPO→GiGPO 同时引入了 (a) GiGPO 的 hierarchical advantage、(b) PRM 的 process reward。如果不加 PRM 只用 GiGPO 呢？如果 GRPO + PRM 呢？论文没给。于是 +2.6 到底有多少来自算法 vs 来自额外 reward signal，不可分离。

---

## 论文点评

### Strengths

1. **诊断准确、目标务实**: 指出 GUI agent 领域三个 infra 空洞（闭源训练、评测漂移、部署死循环）并一次性填上，诊断本身比方法更有价值。95.8% 的复现率实验直接把"GUI eval 结果不可比"从玄学降级为 infra 可治的工程问题。
2. **环境管理的工程细节可信**: spare server rotation、periodic teardown、四阶段 episode 生命周期——这些东西在学术论文里写出来通常意味着作者真的长时间跑过，踩过坑。GiGPO + anchor-state grouping 对长程 GUI 任务的 credit assignment 是对的方向。
3. **Eval 的三段解耦很实用**: Infer / Judge / Metric 解耦后能独立 re-run 任一 stage（尤其是只 re-judge 不重跑 infer）是对后续研究者的真实便利。11+ 模型的预计算 prediction 全部开源，意味着别人可以在不消耗 GPU 的情况下复现数字或试新 judge。
4. **真把模型、代码和部署都发了**: ClawGUI-2B 有 HF/ModelScope checkpoint，代码 Apache 2.0，README 齐全。对一个声称 full-stack 的论文这是必要条件。

### Weaknesses

1. **PRM 的空洞**: 整个 dense reward 的有效性建立在 "Qwen3.5-72B 能可靠判断某个 GUI action 是否 meaningfully contributes to task completion" 这个假设上，但作者完全没报告 PRM 本身的判断准确率、false positive rate、或者针对 reward hacking 的压力测试。一个未验证的 judge 直接进 reward——这是一个黑盒依赖。
2. **核心 ablation 有耦合**: GRPO→GiGPO 的 +2.6 SR 同时混入了 hierarchical advantage estimation 和 PRM step reward 两个变量。没有 "GiGPO w/o PRM" 或 "GRPO w/ PRM" 的对照，无法归因。
3. **真机训练 claim vs 实际训练不一致**: 作者把 "validated support for real physical devices" 写进 abstract 和 contribution，但 ClawGUI-2B 的实际训练是 64 个虚拟 Android emulator，真机在全文没有被实际用于训练 flagship 模型，也没有 real-device 训练的 SR 数字——"支持"和"我在这上面训出过模型"是两件事。
4. **"Small well-trained > large untrained" 叙事偏移焦点**: 拿 ClawGUI-2B（RL post-trained）打 Qwen3-VL-32B / UI-Venus-72B（纯 SFT 或未 RL 的 base）本来就不是 apples-to-apples——真正公平的对比是：**用同一套 ClawGUI-RL 训 8B / 32B，scaling 趋势如何？** 否则这个对比只是在说 RL 比 SFT 强，和 scale 无关。Doubao-1.5-UI-TARS 26.3% 这条作者没主动讨论但在 table 里——大概率是规模+RL 都占了，这才是 ClawGUI-2B 的真正上限参照。
5. **"第一个开源 GUI RL infrastructure" 的 claim 比较激进**: 同时期 OpenClaw、MobileGUI-RL 等都在做类似的事（且被作者引用了），"first open-source" 的表述在 related work 的覆盖下看更像是 marketing 而非严格事实。
6. **Personalized memory 完全没 evaluation**: vector-based personalization 是一个在论文产品叙事里被强调的特性，但没有任何量化实验（个性化召回率？长期交互下 task 成功率提升？）——这是 feature description 而非 research contribution。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training + deployment 三套代码都开源（Apache 2.0）
- **模型权重**: ClawGUI-2B 已在 HuggingFace (`SugarVapeur/OpenGUI-2B`) 和 ModelScope 发布
- **训练细节**: 超参基本完整（lr、batch、temperature、rollout group、epoch、GPU 数），但 PRM 的 prompt 模板、task 数据配比、人工真机任务的具体内容未披露
- **数据集**: MobileWorld 已开源；真机任务"human-authored"但未说明是否公开

#### Claim 可验证性
- ✅ **ClawGUI-2B 17.1% SR on MobileWorld GUI-Only**: Table 1 明确，可通过开源 checkpoint 复现
- ✅ **95.8% reproduction rate across 6 benchmarks × 11+ models**: Table 3 给出每个 cell 的官方值与复现值，完全可审
- ✅ **GiGPO vs GRPO +2.6 SR**: Table 2，可复现但归因不纯（见 Weakness 2）
- ⚠️ **"Infrastructure drives policy quality"（ClawGUI-2B vs MAI-UI-2B 的 6.0 个点 gap 完全来自 RL infra）**: 仅单次对比，没 seed variance，单点差异可能与 RL 超参、数据遇到顺序等多重因素混淆
- ⚠️ **"Validated support for real physical devices"**: 框架声称支持，但 flagship 模型实际未用真机训；真机训练的 SR / 收敛行为没有数字
- ❌ **"First open-source GUI agent RL infrastructure"**: 同时期 MobileGUI-RL、ComputerRL 也都有类似方向——作者自己 related work 引了一堆，再用 "first" 定性需要更仔细的 scope 限定
- ❌ **Personalized memory 相关 claim**: 未 evaluation，纯 description

---

## 关联工作

### 基于
- **MAI-UI-2B**: ClawGUI-2B 的 base model，RL post-training 都是从这里起步
- **MobileWorld**: 提供 Docker-based parallel Android emulator 环境，ClawGUI-RL 的虚拟环境后端
- **GiGPO** (paper: gigpo)：核心 RL 算法，two-level hierarchical advantage + anchor-state grouping，ClawGUI 完全沿用
- **verl / verl-agent**: 底层 RL training framework
- **OpenClaw**: CLI harness 的前置工作，hybrid CLI-GUI 设计中的 CLI 一端

### 对比
- **MobileGUI-RL / ComputerRL / UI-Venus-1.5 / UI-TARS-2**: 同类 online RL for GUI 工作，作者的核心差异 claim 是"开源 infra + 真机支持"
- **UI-Venus-7B/72B, Qwen3-VL 系列, GUI-Owl, Doubao-1.5-UI-TARS**: Table 1 的 end-to-end baseline
- **Agentic framework (Claude-4.5 / Gemini-3-Pro / GPT-5 + UI-Ins-7B)**: 另一档规模，作者明确划为 complementary 而非竞争

### 方法相关
- **GRPO**: 作为 baseline 对比算法
- **Process Reward Model (PRM)**: 在 math / reasoning 领域已被反复验证，这里迁移到 GUI long-horizon 任务做 dense supervision
- **[[2504-ScreenSpotPro|ScreenSpot-Pro]] / V2, UI-Vision, MMBench-GUI, OSWorld-G, AndroidControl**: ClawGUI-Eval 覆盖的 6 个 benchmark

---

## Notes

- **这是一个 infra/product 论文伪装成 research 论文**: 真正的价值在于代码和评测套件是否被社区采纳，而不在 17.1% 这个数字本身。评分给 1（可参考）合适——作为 infra 工具在做 GUI RL 时值得参考，但方法上没有必须读的 insight（GiGPO 和 PRM 都是已有 idea 的组合）。
- **ClawGUI-Eval 可能是本文最大的外部价值**: 95.8% 复现率 + 11+ 模型的预计算 prediction 开源 + Infer/Judge/Metric 解耦 → 后续做 GUI grounding 的工作可以直接在这个 harness 里报数，免去自己搭 eval pipeline 和背 "你的数和 paper A 不 match" 的锅。
- **开源 GUI Agent 真机 RL 的未解问题**: 真机 task source 依赖人工编写 + MLLM-as-judge 做奖励判定——这两个都不可 scale。真正有意思的方向可能是 world model（论文 Discussion 里提到 `Code2World` / `VIMO` / `Genie-3`）作为 simulator，避开真机采样成本。
- **下一步值得做的对照实验**（如果自己在这套 harness 上跑）:
  1. GRPO + PRM vs GiGPO + PRM vs GiGPO w/o PRM，把 Table 2 的耦合拆掉
  2. PRM 质量 ablation：换不同大小的 PRM judge，观察下游 policy SR
  3. Scale curve：用 ClawGUI-RL 分别训 2B / 4B / 8B，对比"ClawGUI 训练 + 小模型"到底能把 scale 压到多小
