---
title: "RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks"
authors: [Ruiying Li, Yunlang Zhou, Yuyao Zhu, Kylin Chen, Jingyuan Wang, Sukai Wang, Kongtao Hu, Minhui Yu, Bowen Jiang, Zhan Su, Jiayao Ma, Xin He, Yongjian Shen, Yang Yang, Guanghui Ren, Maoqing Yao, Wenhao Wang, Yao Mu]
institutes: [AgiBot, ScaleLab]
date_publish: 2026-03-12
venue: arXiv
tags: [VLA, manipulation, task-planning]
paper: https://arxiv.org/abs/2603.11558
website: https://roboclaw-agibot.github.io/
github: https://github.com/RoboClaw-Robotics/RoboClaw
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks
> - **核心**: 用同一个 VLM agent 贯穿 data collection → policy training → long-horizon execution 三阶段，把 "data, policy, deployment 各自独立" 改造成单一 closed loop，关键机制是用 forward + inverse 两条策略组成的 self-resetting 数据采集对（Entangled Action Pairs）。
> - **方法**: VLM meta-controller 用 in-context learning 维护三段 structured memory（role/task/working），通过 MCP 工具调用 forward policy 和 inverse reset policy；底层 VLA 用 [[2504-Pi05|π0.5]] LoRA 微调；deployment 时同一 agent 监控子任务、retry / replan / 升级人工。
> - **结果**: AgiBot G01 双臂平台上 4 个真实场景；long-horizon 成功率比 baseline 提升 25%；同等数据量人力时间降低 53.7%（manual 需 2.16× 时间、8.04× 干预次数）；4 个 forward policy 的成功率随迭代从 4%–46% 单调上升至 46%–86%。
> - **Sources**: [paper](https://arxiv.org/abs/2603.11558) | [website](https://roboclaw-agibot.github.io/) | [github](https://github.com/RoboClaw-Robotics/RoboClaw)
> - **Rating**: 2 - Frontier （system-level lifecycle unification + EAP 是有工程价值的 frontier contribution，但核心 claim 缺消融、泛化性未证，未达 foundation 档）

**Key Takeaways:**
1. **EAP 是工程性的简化，不是新范式**: 把 reset 当成另一条 VLA policy 学，让 forward / inverse 在 agent 控制下交替——核心 insight 是 "reset 比 forward 简单，所以可以可靠学到，从而打破 manual reset 这个数据采集瓶颈"。Reset policy 成功率 72%–86%，forward 5 轮迭代后 46%–86%。
2. **Lifecycle unification 的真正卖点是 semantic consistency**: 同一个 VLM 在采集时打的 instruction $l_t$ 和 deployment 时打的 $l_t$ 来自同一 prompt 模板，缓解了 "操作员说的话" 和 "部署时上层规划说的话" 之间的 distribution gap。这是比 "省人力" 更有价值的 claim，但论文没单独消融。
3. **Long-horizon 25% gain 主要来自 monitor + recovery，不是更好的策略**: Baseline 1 是同样数据训的 [[2504-Pi05|π0.5]] 端到端跑长程任务；agent 加的是子任务边界判断 + 失败时 retry/switch policy + Call Human escalation。这是个 "把 hierarchical control 工程化" 的胜利，而非 policy capability 提升。
4. **Failure 分类（degrading vs non-degrading）有迁移价值**: 把执行失败按 "环境状态是否可逆" 分两类，前者直接 retry，后者训 dedicated recovery policy。这套分类框架对任何 long-horizon manipulation 系统都适用。

**Teaser. RoboClaw lifecycle 全景图——developer 配置 system / MCP tools / skills，RoboClaw 提供 file-based memory 与 embedding，数据通过人示范 + EAP 自重置在线 rollout 持续生成 VLA policy pool，部署时同一 agent 用 high-level plan 调度 policy 完成长程任务。**

![](https://arxiv.org/html/2603.11558v3/figure/teaser.jpg)

---

## 1. 问题与动机

VLA 系统在 short-horizon 任务上已表现不错，但 scale 到 long-horizon 受三重瓶颈：

1. **数据采集人力**: 操作员要 demonstrate、reset、监控失败、过滤轨迹——任务越长越贵
2. **Pipeline 分工导致语义割裂**: 采集者、训练者、部署者通常是不同人，对 "subtask 边界"、"成功标准" 理解不一致
3. **Train-execution distribution mismatch**: 采集时的状态分布 ≠ 部署时遇到的状态分布，long-horizon 下小误差级联放大

> ❓ 第 2 点 "信息 gap" 在论文里反复强调但没有量化证据。这其实是 RoboClaw 最有意思的 claim——同一个 agent 贯穿 lifecycle 应该能减少 semantic mismatch——但没单独消融 (e.g., 用 RoboClaw 采集 + 别的 controller 部署 vs. 全程 RoboClaw)，所以 unified-loop 的实际增益不可量化。

## 2. 系统架构

**Figure 2. RoboClaw 系统架构。VLM 作为 meta-controller，在 in-context learning 范式下运行。多模态观察与 structured memory（role identity / task-level memory / working memory）拼成 decision context；通过 chain-of-thought 推理生成 high-level decision，并经 MCP 接口调用工具。同一 agent core 同时 govern data collection 与 policy deployment，跨 lifecycle 保持一致的 control semantics。**

![](https://arxiv.org/html/2603.11558v3/figure/RoboClaw.png)

### 2.1 三层抽象

RoboClaw 的能力栈分三层（自上而下）：

- **Skills**: 可复用的 procedure，编排 tools 完成复杂工作流（如 "long-horizon-execution" skill 会先调 Env Summary 再调 Start Policy）
- **Tools**: 通过 MCP 暴露的系统接口（Start Policy / Terminate Policy / Change Policy / Env Summary / Fetch Robot Stats / Call Human）
- **Policies**: 底层 VLA 模型，输出 motor action

### 2.2 Structured Memory

每个 timestep $t$，agent 维护:

$$
m_{t}=(r_{t},\, g_{t},\, w_{t})
$$

- $r_t$ — **role identity**: 当前 operational mode + 可用工具集
- $g_t$ — **task-level memory**: 全局任务 + 拆分后子任务 + 各自执行状态（用于追踪长程进度）
- $w_t$ — **working memory**: 当前激活的 skill + tool invocation 历史

### 2.3 Closed-loop decision cycle

每步：observation + memory → CoT 推理（解析 scene → 判断当前 objective → 评估完成判据 → 决定下一步 action）→ MCP tool 调用 → 写回 memory。直到任务完成。

## 3. 数据采集：Entangled Action Pairs

**Figure 3. RoboClaw 自动数据采集流程。Agent 与用户交互启动一个数据采集任务（如 "把 primer 放进抽屉"），用 MCP 工具自主处理视觉观察、评估初始状态、制定计划，然后持续执行 forward-reverse loop（放进去 → 拿出来），同时实时监控异常，从而连续获取 manipulation 数据。**

![](https://arxiv.org/html/2603.11558v3/figure/eap_data.png)

### 3.1 核心思想

对每个 manipulation policy $k$ 学一对策略：

- **Forward policy** $\pi^{\rightarrow}_{\theta_k}$: 执行目标行为
- **Inverse reset policy** $\pi^{\leftarrow}_{\phi_k}$: 撤销 forward，把环境回归到可复用的 precondition region

两条策略交替执行，agent 判断 forward 完成后立即触发 reset，环境自动回到初始态——无需人工 reset。两条 trajectory 配对存入数据集 $\mathcal{D}$。

$$
\tau_k = (\tau_k^{\rightarrow},\, \tau_k^{\leftarrow})
$$

### 3.2 Policy 公式化

底层 VLA 用 [[2504-Pi05|π0.5]]（论文明确这一点），通过 conditional flow matching 学 action chunk 分布:

$$
A_{t} = \pi_{0.5}(o_{t},\, l_{t},\, q_{t}),\quad A_{t}=[a_{t},\ldots,a_{t+H-1}]
$$

其中 $l_t$ 不是人类直接给的指令，而是 RoboClaw agent 在 MCP tool 调用时**动态生成**的 structured instruction。

训练目标是学 velocity field $v_\theta$，把高斯噪声 transport 到真实 action 分布:

$$
\mathcal{L}^{\tau}(\theta) = \mathbb{E}\left[\left\|v_{\theta}(A_{t}^{\tau},o_{t},l_{t},q_{t}) - u(A_{t}^{\tau}\mid A_{t})\right\|^{2}\right]
$$

其中 $\tau \in [0,1]$，$A_t^\tau = (1-\tau)\epsilon + \tau A_t$ 是噪声与真实 action 的线性插值。

> ❓ 这个 flow matching loss 完全沿用 π0/π0.5，本文没有方法上的修改——那 EAP 的 self-resetting 是否依赖 flow matching policy 的特殊性质？换 [[2406-OpenVLA|OpenVLA]] 类型的 autoregressive policy 应该也成立。论文没讨论 policy class 的依赖性。

### 3.3 训练超参

**Table 1. π0.5 fine-tuning 超参。**

| General | Value | LoRA | Value |
|---|---|---|---|
| Precision | bfloat16 | Rank ($r$) | 16 |
| Batch size | 16 | Alpha ($\alpha$) | 16 |
| Training steps | 10k | Dropout | 0.1 |
| Warmup steps | 100 | Target modules | all-linear |
| Learning rate | $2.5 \times 10^{-5}$ | Inference steps | 3 |
| Gradient checkpointing | ✓ | | |

## 4. 部署时的 process supervision

部署阶段同样的 closed-loop 结构：

1. Agent 根据 $o_t, m_t$ 选下一个子任务 $z_t$
2. 通过 MCP 调用对应 forward policy
3. 周期性查询 Env Summary / Fetch Robot Stats 监控进度，写入 working memory
4. 子任务完成 → 更新 task memory，进入下一个；否则 retry 同一 policy 或 Change Policy 切到备选
5. 重复失败或检测到异常 → 重新规划尝试 recovery skill；若仍不成功或触发安全条件 → Call Human

**关键设计**: 部署时产生的轨迹会**回流**到 $\mathcal{D}$，作为新经验持续 refine forward policy。这构成 lifecycle learning loop。

## 5. 实验

平台: **AgiBot G01**，双臂 mobile manipulation 机器人，20 DoF（不含 end-effector），双臂各装 AGIBOT OmniPicker 自适应 gripper（单主动 DoF）。

四个真实场景: 卧室梳妆台、厨房储物架、书桌、便利店货架。

四个单技能子任务（用于 vanity table 长程任务的拆分）:
- **Body lotion placement** — 长距离 pick-and-place
- **Primer placement** — 放入抽屉 + 关抽屉（双步骤、有遮挡）
- **Lipstick insertion** — 紧密插入（精度敏感）
- **Tissue wipe** — 持续接触式擦拭（轨迹质量重于终态）

### 5.1 Q1: 数据采集效率

**Figure 4. Human effort 对比与长程任务成功率。(a) 同等数据量所需人力时间 (相对值, Ours = 1)；(b) Rollout 时人介入比例；(c) Vanity table 长程任务成功率随迭代变化——RoboClaw 显著优于 end-to-end VLA baseline 和 "四个独立子任务成功率乘积" 的预期上界。**

![](https://arxiv.org/html/2603.11558v3/x1.png)

- 同等数据量下 manual baseline 需 **2.16×** RoboClaw 的人力时间
- Rollout 期间 manual 需 **8.04×** RoboClaw 的人介入次数

### 5.2 Q2: 子任务策略成功率

**Reset policy 成功率（设计上比 forward 简单，以保证 self-reset 稳定）:**

**Table 2. Inverse reset policy 在四个任务上的成功率。**

| Task | Body Lotion | Primer | Lipstick | Tissue Wipe |
|---|---|---|---|---|
| Success Rate | 36/50 | 38/50 | 43/50 | 39/50 |

**Forward policy 随迭代提升（每次迭代加 50 个 EAP 轨迹）:**

**Table 3. Forward policy 成功率随 rollout 迭代变化。**

| Iteration | Body Lotion | Primer | Lipstick | Tissue Wipe |
|---|---|---|---|---|
| 1 | 21/50 | 23/50 | 2/50 | 11/50 |
| 2 | 25/50 | 31/50 | 4/50 | 13/50 |
| 3 | 32/50 | 31/50 | 11/50 | 14/50 |
| 4 | 37/50 | 34/50 | 16/50 | 21/50 |
| 5 | 43/50 | 40/50 | 23/50 | 26/50 |

注意 Lipstick insertion 从 4% 提升到 46%——最难的精密插入任务受益最大。但即便如此终值（46%–86%）距 deployment-ready 仍远。

> ❓ Iter 1 的 21/50 是从多少人示范开始训的？论文写 "fixed number of human demonstrations" 但没给具体数字，这让 "iteration 提升" 的绝对值无法解读——如果初始示范量很少，提升当然显著。

### 5.3 Q3: 长程任务

**Figure 5. Vanity table 整理任务的长程执行。同一个 VLM agent 在四个独立训练的 forward policy checkpoint（primer / lipstick / lotion / tissue）之间动态调度，并在需要时 re-plan。**

![](https://arxiv.org/html/2603.11558v3/figure/infer.jpg)

Baselines:
- **Baseline 1**: 同样数据训的 π0.5 端到端跑长程任务
- **Baseline 2**: 四个子任务成功率的乘积作为 expected long-horizon 成功率

Result: RoboClaw 显著高于两者。25% 提升的来源是 "monitor task progress + automatically invoke recovery"——不是策略本身的能力。

### 5.4 Q4: 从失败中学

把执行失败分两类:

- **Non-degrading failure**: 环境状态基本不变（如夹空），retry 同 policy 即可
- **Degrading failure**: 失败改变了环境（如瓶子倒了、滑出区域），需要 recovery action 才能继续

早期 degrading failure 触发 Call Human；累积经验后这些 recovery 行为被训成专门的 recovery policy 加入 policy library，后续可自动调用——这是 RoboClaw 的 behavioral repertoire 自我扩张机制。

---

## 关联工作

### 基于
- [[2504-Pi05|π0.5]]: RoboClaw 的 forward / inverse policy 都是用 π0.5 + LoRA 微调实现的，flow matching loss 与 action chunk 公式完全沿用
- In-context learning: VLM meta-controller 的决策机制基于 ICL，论文 cite Dong et al. 2024 ICL survey

### 对比
- 端到端 [[2504-Pi05|π0.5]]（Baseline 1）: 同数据训的端到端 VLA 跑长程任务，验证 "agent orchestration" 的增益
- Manual data collection (Baseline): 人示范 + 人 reset，验证 EAP 的人力节省

### 数据采集相关
- AnyTeleop / GELLO / Mobile ALOHA: 传统 teleoperation 系统，RoboClaw 在它们之上加了 self-reset
- RoboCopilot: human-in-the-loop residual correction，与 RoboClaw 的 Call Human escalation 思想接近但仍依赖人参与
- Genie Centurion: "rewind-and-refine" + Task Sentinel 失败检测；机制类似但仍是人主导
- VLAC: 与 Genie Centurion 类似的 critic-based 失败检测
- FieldGen: 半自动，人示范精细 + 自动生成 pre-manipulation——比 EAP 思路更复杂
- MimicGen / GenH2R-Sim / RoboCasa: simulation-based 数据合成；RoboClaw 是 real-world 自动采集的对照
- RoboTwin 2.0: MLLM + simulation-in-the-loop 验证；RoboClaw 是 real-world 版本
- HumanoidGen: LLM 生成空间约束 + STCR tree search
- CyberDemo: Auto Curriculum Learning 调整 augmentation 难度

### Hierarchical VLA / Long-horizon 相关
- [[2204-SayCan|SayCan]]: 早期 plan-and-act 的代表
- [[2502-HiRobot|Hi Robot]]: Hierarchical VLA，有 plan-verify 机制——RoboClaw 没和它直接比
- HAMSTER: Hierarchical action model
- Agentic Robot: brain-inspired framework for VLA——与 RoboClaw 思想最接近，但论文未作为 baseline
- Inner Monologue / LITEN: Replanning for failure recovery
- Code as Policies / VoxPoser / Language Models as Zero-Shot Planners: LLM-as-planner 方向

### 基础 VLA
- [[2410-Pi0|π0]]: π0.5 的前身，flow matching VLA
- [[2406-OpenVLA|OpenVLA]]: 开源 VLA baseline
- [[2307-RT2|RT-2]]: VLA 范式开创
- [[2303-PaLME|PaLM-E]]: Embodied multimodal LM

---

## 论文点评

### Strengths

1. **EAP 是简单且 generalizable 的工程方案**: 不依赖 simulation、不需要特殊硬件——任何能学 forward policy 的场景都能尝试学 inverse policy。Reset 比 forward 简单这个 prior 在 manipulation 中通常成立。
2. **失败分类（degrading vs non-degrading）有方法论价值**: 这套二分法比 "agent 检测到失败" 这种泛泛之谈具体得多，可直接迁移到任何 hierarchical control 系统。
3. **真实硬件、真实场景**: 不是 sim 或 toy 任务，AgiBot G01 + 四个家居/零售场景，且每个数据点用 50 trial 评估，比很多 VLA 论文的 10 trial 鲁棒。
4. **Lifecycle 闭环的工程设计完整**: MCP tools、structured memory（三段式）、Call Human escalation——是一个端到端可用的系统而非 paper-only demo。

### Weaknesses

1. **核心 claim "unified semantics" 没有消融**: "同一 agent 贯穿 lifecycle 减少 mismatch" 是论文最大卖点，但没有 "RoboClaw 采集 + 别的 controller 部署" 这种对照实验。25% gain 完全可能来自 deployment-time monitor + recovery，而非 collection/deployment 的语义一致性。
2. **EAP 依赖 "inverse policy 比 forward 简单" 的假设**: 这个假设在论文设定的 placement / insertion 任务上成立，但在 wiping / pouring 这类**没有清晰逆操作**的任务上未必。论文 4 个子任务都属于 "可逆 pick-place"，没测真正难逆的场景。
3. **"减少 53.7% 人力" 没有充分披露 baseline 设置**: Baseline manual reset 是用最高效流程吗？还是 naive？2.16× 和 8.04× 的具体协议（人示范的轨迹数、reset 的复杂度）没说清。
4. **Forward policy 终极成功率仍偏低**: 5 轮迭代后 Lipstick 才到 46%、Tissue 到 52%。如果 forward 只能到这个水平，长程任务靠 retry/replan 撑住成功率的代价是 wall-clock time 大幅增加——论文没报 long-horizon 任务的执行耗时。
5. **VLM controller 的具体模型没说**: 只写 "off-the-shelf VLM"，没披露用的是 GPT-4o / Claude / Gemini / 开源 VLM。这对 reproducibility 和 cost 估算都很关键。Conclusion 里 "cloud-based large models 的 latency" 暗示是云端 API。
6. **没有与同期 hierarchical VLA 对比**: [[2502-HiRobot|Hi Robot]]、Agentic Robot 在 related work 里提了，但没作为 baseline 比。这让 "agentic orchestration 的 25% 增益" 缺乏 method-level 对照。

### 可信评估

#### Artifact 可获取性

- **代码**: 已开源（GitHub `RoboClaw-Robotics/RoboClaw`），README 提供 TUI/GUI 启动命令；但完整 VLA 部署需要私聊获取 .whl 包
- **模型权重**: 未发布 forward / reset policy checkpoint
- **训练细节**: π0.5 微调超参完整披露（Table 1）；数据配比仅说 "human demo + EAP rollout + human intervention 三类" 但未给具体比例
- **数据集**: 未开源；EAP 收集的 trajectory 数据集 $\mathcal{D}$ 未公开
- **VLM 选择**: 完全未披露——是关键 reproducibility gap

#### Claim 可验证性

- ✅ **Forward policy 成功率随迭代提升**：Table 3 五轮数据完整、有明确 trial 数（50 each），可信。
- ✅ **EAP 减少 manual reset 次数**：机制上必然成立（reset 由 inverse policy 完成而非人工）。
- ⚠️ **"25% improvement on long-horizon"**：Baseline 是 π0.5 端到端，但 trial 数（averaged over 20）样本偏小；对比对象是单一 baseline 而非多个 hierarchical 方法。
- ⚠️ **"reduce human time investment by 53.7%"**：分母 "整个机器人 lifecycle" 的统计口径模糊（采集人时？部署人时？怎么聚合？）。
- ⚠️ **"unified contextual semantics 减少 mismatch"**：理论上合理，但无消融支持——25% gain 不能归因到这一点。
- ⚠️ **Inverse policy 总成功率 72%-86%**：论文说 "intentionally designed to be simpler" ——任务本身被设计得偏简单，所以成功率高，这并不能直接证明 reset policy 普遍可学。

### Notes

- 这篇论文最像 "把 [[2502-HiRobot|Hi Robot]] / Agentic Robot 那一类 hierarchical VLA 的部署工程化 + 加上自动 reset 数据闭环"，没有 architectural novelty，但**把数据采集 / 部署 / lifecycle learning 三件事用同一个 agent 串起来**这件事本身在 system level 是有价值的。
- **EAP 的真正贡献是 problem reformulation**: 把 "怎么 reset 环境" 从 "需要专门硬件 / 人工" 重构成 "再训一个 inverse policy"。这是个 simple, scalable 的 trick——能否泛化到不可逆任务（液体、不可逆变形）是它的边界。
- **Lifecycle data flywheel 的真伪**: 论文声称 deployment 轨迹回流训练能持续 improve policy，但只展示了 5 轮 EAP iteration 的曲线，没有 "deployment 数据混入后" 的独立 ablation。Flywheel 是否真转起来未被验证。
- **Pivot 触发**: 如果 inverse policy 学不动（如失败模式不可逆的任务），整个 EAP 框架就坍缩为标准 imitation learning + manual reset。这是它最大的 failure mode。
- **可借鉴 takeaway**: degrading vs non-degrading failure 二分法、把 reset 当 policy 学的思路、structured memory 三段式（role / task / working）——这三个工程模式可迁移到自己的 agent 系统设计。
- **后续观察点**: 这套系统在更难的 long-horizon 任务（如 cooking、组装）上是否成立？现有评估都是相对结构化的 "把物体放到指定位置" 类任务。

### Rating

**分数**：2 - Frontier
**理由**：RoboClaw 把 data collection / training / deployment 用同一 VLM agent 串成 closed loop + EAP self-reset，是 real-robot 上 frontier 级的 system contribution，且真机四场景 50-trial 评测比多数 VLA 论文扎实，值得作为 hierarchical VLA + lifecycle data 的参考 baseline。但不够 Foundation：核心 "unified semantics" claim 缺消融（Weaknesses #1）、EAP 依赖 "reset 比 forward 简单" 的假设在 wiping/pouring 等不可逆任务未验证（Weaknesses #2）、VLM controller 具体型号未披露影响 reproducibility（Weaknesses #5）；加之 2603 发表时间新、尚无后续工作以其为 de facto baseline 的信号，仍在 Frontier 档而非 Foundation。高于 Archived：EAP 的 problem reformulation 和 degrading/non-degrading 失败分类有方法论迁移价值（Notes），不是 incremental 或一次性参考。
