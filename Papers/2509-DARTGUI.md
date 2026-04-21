---
title: "Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation"
authors: [Pengxiang Li, Zechen Hu, Zirui Shang, Jingrong Wu, Yang Liu, Hui Liu, Zhi Gao, Chenrui Shi, Bofei Zhang, Zihao Zhang, Xiaochuan Shi, Zedong Yu, Yuwei Wu, Xinxiao Wu, Yunde Jia, Liuyu Xiang, Zhaofeng He, Qing Li]
institutes: [Beijing Institute of Technology, BIGAI, DataCanvas, Beijing University of Posts and Telecommunications, Shenzhen MSU-BIT University]
date_publish: 2025-09-28
venue: arXiv
tags: [agentic-RL, computer-use, gui-agent]
paper: https://arxiv.org/abs/2509.23866
website: https://computer-use-agents.github.io/dart-gui
github: https://github.com/computer-use-agents/dart-gui
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] DART-GUI: Decoupled Agentic RL Training for GUI Agents
> - **核心**: 把 multi-turn GUI agent 的 RL pipeline 拆成 4 个完全异步的模块（Env Cluster / Rollout Service / Data Manager / Trainer），叠加多粒度的 adaptive data curation，把 OSWorld 的 RL 训练效率与最终性能同时往前推
> - **方法**: 异步架构 + rollout-wise sampling + per-worker model sync；task / trajectory / step / token 四级 data curation（dynamic rollout N、experience pool、high-entropy step selection、truncated importance sampling）
> - **结果**: DART-GUI-7B 在 OSWorld-Verified 取得 42.13% SR（base UI-TARS-1.5-7B 27.52%，开源 SOTA +7.34%），rollout GPU 利用率 1.6×、训练吞吐 1.9×、env 利用率 5.5×
> - **Sources**: [paper](https://arxiv.org/abs/2509.23866) | [website](https://computer-use-agents.github.io/dart-gui) | [github](https://github.com/computer-use-agents/dart-gui)
> - **Rating**: 2 - Frontier（GUI agent RL 场景下少有的全流程开源 baseline，system + data curation 组合 OSWorld 提升显著，但 building block 多沿用既有工作、algorithmic novelty 有限且缺 ablation）

**Key Takeaways:**
1. **Decoupling 是 GUI RL 当前真正的 bottleneck**：传统 coupled pipeline 在 OSWorld 这种动辄几十步、单 episode 几十分钟的环境里，等待 batch 内最长 trajectory 完成会造成大量 GPU/env idle；rollout-wise scheduling + per-worker 滚动同步是 system 层面的合理解
2. **Sparse-reward 长 horizon 任务需要 data curation 而非更猛的 RL 算法**：DART 的性能增益里相当一部分来自 (a) experience pool 兜底（保证全失败任务也有正样本）、(b) dynamic N 防止过拟合简单任务、(c) 只在 top-80% entropy step 上算 loss——把 GRPO 的信号塞到真正关键的决策点
3. **Truncated importance sampling 修正 inference/training 引擎的分布漂移**：rollout engine（量化推理）和 trainer（FP16/BF16）天然 off-policy，加一个截断的 IS 权重就能稳定下来——这点在所有 decoupled RLVR 框架里都通用
4. **DART-GUI-7B 仅用 30 步上限就接近 Claude-4-Sonnet（100 步）**：sample efficiency 维度上的提示比绝对 SR 数字更值得关注

**Teaser. DART 框架总览——四个异步模块的关系图**

![](https://arxiv.org/html/2509.23866v1/x1.png)

---

## 1. 问题与动机

GUI agent（VLM-based, e.g. [[2501-UITARS|UI-TARS]], Aguvis, [[2410-OSAtlas|OS-Atlas]]）在 [[2404-OSWorld|OSWorld]] 这种真实桌面任务上需要长 horizon 多轮交互，应用 RL 时遇到两个结构性 bottleneck：

1. **Pipeline coupling**: 现有 RL 实现里 action prediction → env step → data buffer → trainer 是顺序阻塞的。GUI 任务单步要等浏览器 / OS 响应（秒级），整条 trajectory 几十分钟，coupled pipeline 让 GPU 大量空转。
2. **Task difficulty heterogeneity**: 同 batch 内任务难度差异大；简单任务容易过拟合，难任务大概率 0 reward → 没有 learning signal。已有方法（GUI-R1, InfiGUI-R1, ARPO, ZeroGUI）在 OSWorld 上 RL 提升只有 2-4%。

> 作者把 problem formulation 切成 "system efficiency" 和 "data curation" 两条互相正交的轴，是这篇 paper 的关键 framing 选择。这种切法本身合理——RL 的 wallclock 性能确实由 system + algorithm 共同决定——但同时也让贡献变成 "engineering 巧思 + 一组 well-known curation tricks 的组合"，缺乏单一的算法 insight。

## 2. DART 框架

### 2.1 Formulation

GUI 任务建模为 sequential decision-making：state $s_t$ 是 screenshot，history $h_t$ 包含过去 $m$ 步的 (state, thought, action)。policy $\pi_\theta$ 在 $(\tau, h_t, s_t)$ 上生成 thought $r_t$ + action $a_t$。

$$
r_t^*, a_t^* = \arg\max_{r_t, a_t} \pi_\theta(a_t \mid \tau, h_t, s_t)
$$

> 注意 history 里只保留最近 $m$ 步——typical "sliding window context"。Long-horizon 任务里这是 lossy 的，但对 OSWorld 还能 work，说明大多数任务的关键信息高度局部化。

### 2.2 四模块架构

**Figure 2. DART 框架架构图——Rollout Service 与多 env 并行，Data Manager 中转，Trainer 异步更新**

![](https://arxiv.org/html/2509.23866v1/x2.png)

四个模块：
- **Env Cluster**：上百个真实 desktop env（OSWorld Docker），从 Data Manager 接收任务序列
- **Rollout Service**：托管多份 policy model，给 env 提供 thought/action 推理；以 rollout 为最小调度单元
- **Data Manager**：缓存 trajectories + rewards；当任务的 N 条 trajectory 全部完成（或填补够正样本）→ 推给 Trainer
- **Trainer**：异步消费 trajectory，做 step-wise GRPO 更新，权重逐 worker 同步回 Rollout Service

模块间**非阻塞通信**：rollout 在更新权重时不暂停整个 service，env 在等待 trainer 时也不闲置。

### 2.3 Step-wise GRPO

对每个任务 $\tau$ 采 $N$ 条 trajectory，分解到 step 级：$\mathcal{D} = \{(h_{i,j}, s_{i,j}, r_{i,j}, a_{i,j}, R_i)\}$。reward 是 OSWorld 验证脚本给出的 $[0,1]$ 标量。

**Equation 1. Step-wise GRPO 目标**

$$
\begin{aligned}
\mathcal{J}(\theta) &= \mathbb{E}_{(h,s,a,R)\sim\mathcal{D}}\Big[\nabla_\theta \min\big(\rho A,\ \text{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\,A\big) \\
&\quad - \beta\, D_{\text{KL}}(\pi_\theta^{\text{Train}} \,\|\, \pi_\theta^{\text{Ref}})\Big]
\end{aligned}
$$

其中 $\rho = \pi_\theta^{\text{Train}}(a|h,s) / \pi_{\text{old}}^{\text{Train}}(a|h,s)$，advantage 在 step group 内 z-score 归一：

$$
A_{i,j} = \frac{R_i - \bar R}{\sigma_R}
$$

> trajectory-level reward 直接 broadcast 到所有 step——典型的 sparse-reward credit assignment 偷懒做法，但对 OSWorld 这种 outcome-based reward 几乎是唯一选择。后面的 high-entropy step selection 部分弥补了这点。

### 2.4 Rollout-wise Sampling

任务难度差异 + env 随机性 → trajectory 长度方差极大。三种调度对比：

**Figure 3. 三种 sampling timeline 对比（batch-wise / task-wise / rollout-wise）**

![](https://arxiv.org/html/2509.23866v1/x3.png)

- **(a) Batch-wise**：等整 batch 全部 trajectory 完成才开下一批 → 慢任务卡住快任务的资源
- **(b) Task-wise**：任务内 N 个 rollout 完成才开新任务 → 仍有 idle
- **(c) Rollout-wise（DART）**：单条 rollout 是最小调度单位，env 完成立即抓下一个请求

配合**动态 GPU pool**：所有 GPU 进入 shared Rollout Service，按当前 utilization 派单。这个写法实质就是 work-stealing scheduler 应用到 RL rollout 上。

### 2.5 Per-Worker Model Synchronization

**Figure 4. All-worker vs. per-worker 模型同步的 timeline 对比**

![](https://arxiv.org/html/2509.23866v1/x4.png)

传统 global sync：trainer 完成一轮 → 所有 rollout worker 暂停 → 全部更新权重 → 恢复服务。idle 集中爆发。

DART 的 per-worker：staggered，一次只更新一个 worker（或一小组 GPU），其余继续以旧权重 serve。代价是 rollout 阶段同时存在多个 model version，所以**必须**配 §3.4 的 truncated IS 来吃掉 policy mismatch。

> 这是借鉴游戏 AI（StarCraft II / Dota 2）的标准做法，AReal、ROLL 也都用过。DART 的差别在于把它专门做到 GUI 长 horizon 场景。

## 3. 多级 Adaptive Data Curation

### 3.1 Performance-Aware Task Rollout（task & trajectory level）

**Figure 5. 根据任务成功率动态调整 rollout 数 N**

![](https://arxiv.org/html/2509.23866v1/figure/dynamic_rollout.png)

- **Dynamic Rollout Frequency**: 任务 SR > 0.6 → N 从 8 降下来，腾资源给难任务；低 SR 任务保持 max N
- **Dynamic Trajectory Length**: 任务专属长度上限，由该任务历史成功 trajectory 的 max length 决定。简单点击任务 ~10 步停，复杂多应用任务延伸到 50 步

### 3.2 Experience Pool of Trajectories

难任务一组 N 条 rollout 全部失败时，advantage = 0，无 learning signal。DART 预先 sample 收集一批高质量 successful trajectory 进 Experience Pool；训练中检测到 "全失败" → 从池子里抽一条正样本塞进 batch。

> 实质就是 hindsight experience replay 的简化版——保证每个训练 task 至少 1 条正样本。简单但有效，特别是对 OSWorld 这种 SR 普遍 <40% 的环境。
>
> ❓ 论文没说 Experience Pool 里的 trajectory 是用什么 policy 采的、什么时候过期。如果一直用初始 policy 采的 trajectory 训到底，policy mismatch 会越来越严重——这就是后面 truncated IS 要解决的问题之一。

### 3.3 High-Entropy-Driven Step Optimization（step level）

借鉴 [Beyond 80/20: high-entropy minority tokens drive RL]——只在高 entropy token 上训练有效。DART 把它从 token 级抬到 step 级：

$$
H_t = \frac{1}{|r_t| + |a_t|} \sum_{i=1}^{|r_t|+|a_t|} H_{t,i}
$$

step 是 $r_t$（thought）+ $a_t$（action）拼接的整段 token 序列的平均 entropy。**只对 group 内 entropy ≥ 20 分位的 step 算 loss**（即保留 top 80%），低 entropy step 视为 "non-critical" 直接 mask 掉。

> 这个迁移合理——多轮 GUI 决策中很多步（比如等页面加载、确定的 click confirm）确实是 deterministic 的，对它们做 RL 没意义。但 20% 的阈值是经验数字，没看到 ablation 验证为什么不是 30% 或 10%。

### 3.4 Distribution Alignment for OOD Tokens（token level）

Rollout Service 用量化推理，Trainer 用 BF16；加上 Experience Pool 的 trajectory 来自旧 policy → rollout policy 与 train policy 之间天然分布漂移。引入 truncated importance sampling weight：

$$
w = \min\left(\frac{\pi_{\text{old}}^{\text{Train}}(a|h,s)}{\pi_{\text{old}}^{\text{Rollout}}(a|h,s)},\ C\right)
$$

最终训练目标（结合 high-entropy mask 和 truncated IS）：

**Equation 2. Final objective（high-entropy mask + truncated IS + GRPO）**

$$
\begin{aligned}
\mathcal{J}_{\text{HE}}(\theta) &= \mathbb{E}_{(h,s,a,R)\sim\mathcal{D}} \Big[\mathbb{I}[H_t \geq \tau_{\mathcal{D}}^{0.2}] \cdot \big(w \cdot \nabla_\theta \min(\rho A,\ \text{clip}(\rho)A) \\
&\quad - \beta\, D_{\text{KL}}\big)\Big]
\end{aligned}
$$

> Truncated IS 是 [Yao et al., off-policy RL framework] 中的标准做法，DART 直接搬过来。这部分没什么 algorithmic novelty，但工程上必要。

## 4. 实验

### 4.1 Setup

- **Base model**: [[2501-UITARS|UI-TARS-1.5-7B]]
- **Benchmark**: [[2404-OSWorld|OSWorld]]-Verified（执行式自动判分，reward ∈ [0,1]）
- **Training data**: 203 个任务（按 ARPO 的采样方法选），来自 OSWorld
- **Evaluation**: max 30 steps（对比 baseline 通常 100 steps）

### 4.2 Main Results（OSWorld）

**Table 1. OSWorld-Verified 主结果（节选自论文 Table 5.2）**

| Model | Max Steps | OSWorld SR |
|---|---|---|
| UI-TARS-1.5-7B (base) | 100 | 27.52% |
| OpenAI CUA o3 | 100 | 23.00% |
| Claude-4-Sonnet | 100 | 41.39% |
| **DART-GUI-7B** | **30** | **42.13%** |

- 相比 base：**+14.61% absolute**
- 相比此前开源 SOTA：**+7.34%**
- 关键的是只用 30 步达到 Claude-4-Sonnet（100 步）的水平
- 大幅提升的应用类别：OS（+31.25%）、LibreOffice Writer（+21.73%）、Thunderbird（+20.00%）——长 horizon、多步骤的复杂任务

### 4.3 System Efficiency

声称的相对 coupled baseline 提升：
- Rollout GPU utilization：**1.6×**
- Training throughput：**1.9×**
- Env utilization：**5.5×**

> 5.5× env util 是最大的数字——直接对应 rollout-wise scheduling 的收益（env 不再等 batch / task 同步点）。GPU 1.6× 相对温和，提示 rollout 推理本身的瓶颈不是 GPU 而是 env response time。

> ❓ 没有公开 ablation table（在 arxiv v1 HTML 截止于 §5.2 main results）。所以无法分清 14.61% gain 中哪部分来自 system（更多 sample / 更多 update）、哪部分来自 algorithm（high-entropy mask / experience pool / truncated IS）。这是一个比较关键的缺失。

---

## 关联工作

### 基于
- [[2501-UITARS|UI-TARS-1.5-7B]]：base policy model，DART-GUI-7B 的初始化
- [[2404-OSWorld|OSWorld]]：训练 + 评估 benchmark；reward 来自其执行验证脚本
- GRPO（DeepSeekMath / DeepSeek-R1）：RL 算法主干
- Beyond 80/20 (Wang et al. 2025)：high-entropy token RL 的 inspiration
- AReal、ROLL：异步 RL 框架先例

### 对比 / 同期
- ARPO (Lu et al. 2025)：GRPO + experience replay for GUI agent，DART 训练任务采样方法直接复用
- ZeroGUI (Yang et al. 2025)：自动 task/reward 生成，零人工成本
- [[2508-ComputerRL|ComputerRL]]：API-equipped GUI agent 的异步 RL 训练；架构思路最接近 DART
- GUI-R1、InfiGUI-R1：早期 offline RL，缺多轮 / online interaction
- [[2509-UITARS2|UI-TARS-2]]：闭源 SOTA，多轮 RL 的同期工作

### 方法相关
- Truncated IS (Yao et al. 2025, off-policy RL framework)：DART §3.4 token 级 distribution alignment 的来源
- DAPO (Yu et al. 2025)：scale 化的 GRPO 实现参考
- [[2401-SeeClick|SeeClick]]、[[2408-OmniParser|OmniParser]]、[[2411-ShowUI|ShowUI]]、[[2504-TongUI|TongUI]]、[[2508-OpenCUA|OpenCUA]]：GUI agent 范式的不同代表（visual / hybrid / structured）

---

## 论文点评

### Strengths

1. **Problem framing 清楚**：把 GUI agent RL 的低效拆成 system bottleneck（pipeline coupling）+ algorithm bottleneck（task heterogeneity / sparse reward），两条独立轴上各自给出针对性方案，整套系统 self-consistent
2. **System-side 工程贡献扎实**：rollout-wise scheduling + per-worker model sync 是 GUI agent RL 场景下值得借鉴的 baseline 设计；env 利用率 5.5× 的数字虽未独立复现，但与 long-horizon RL 的常识相符
3. **Data curation 是从问题驱动来的**：experience pool 解决 0-reward 问题，dynamic N 解决资源浪费，high-entropy step selection 解决 credit assignment 噪声——每个组件都有明确的 motivation
4. **Sample efficiency 真实**：30 步 vs Claude-4-Sonnet 100 步的对比比绝对 SR 数字更说明问题，意味着 DART-GUI 学到的不只是 "更多搜索" 而是 "更准的决策"
5. **完整开源（路线图）**：训练框架 + 数据 + checkpoint + Docker，对 community 友好

### Weaknesses

1. **缺 algorithm-level ablation**：arxiv v1 HTML 在 §5.2 main results 处截断。从公开内容看不到 dynamic N、experience pool、high-entropy mask、truncated IS 各自的贡献分解，最后 14.61% 的 gain 无法 attribute 到具体组件
2. **超参 magic number 多**：top 80% entropy threshold、SR 0.6 切换 N、experience pool 大小、truncated IS 的 C 值——都是经验值，没有 sensitivity 分析
3. **Experience Pool 的 staleness 问题没讨论**：随着 policy 更新，预收集的 trajectory 会越来越 OOD；truncated IS 能 mitigate 多少不清楚，长训练可能崩
4. **System novelty 边际**：异步 RL 框架（AReal、ROLL、ComputerRL）已经存在；DART 的差别在于专门为 GUI long-horizon 调优，但每个独立 building block（rollout-wise scheduling、per-worker sync、truncated IS）都来自已有工作
5. **Generalization 未验证**：只在 OSWorld 训 + 测，没在 AndroidWorld、WebArena、Mind2Web 上 cross-benchmark 验证。所谓 "decoupled framework" 的可迁移性还是 claim 而非 evidence
6. **Sliding-window history（$m$ 步）的局限**：很多 OSWorld 任务的关键 context 在早期几步（如 "open file X"），window 滑过去后 agent 没法 recall——但论文没分析这点

### 可信评估

#### Artifact 可获取性

- **代码**: inference + training（截至 2025-12-10 GitHub README 标注 training code、sampling code、SQL schema、Docker config 全部释出）
- **模型权重**: `dart-gui/dart-gui-7b` 已发布在 HuggingFace（README 链接）
- **训练细节**: 主体公式 + 主要超参（rollout N、entropy threshold、IS clip C 等）在正文披露；完整训练 schedule、batch 配比未在 arxiv v1 HTML 中露出（应在 Appendix A.4，但 HTML 渲染截止）
- **数据集**: 训练集是 OSWorld 的 203 个任务子集（按 ARPO 方法采样）；评估在 OSWorld-Verified 上跑。Experience Pool 内的 trajectory 是否单独发布未明确

#### Claim 可验证性

- ✅ **OSWorld 42.13% SR**：grounding——OSWorld 评估脚本是确定性 execution-based 验证，第三方可复现
- ✅ **+14.61% over base UI-TARS-1.5-7B**：base 模型公开，差值可独立验证
- ⚠️ **1.6× GPU / 1.9× training / 5.5× env utilization**：相对 "coupled baseline" 的对比，但论文没明确说 baseline 的具体实现细节（是 vanilla GRPO loop 还是 ARPO 这种已有的 RL 实现）。数字方向合理，但 specific multiplier 需谨慎
- ⚠️ **"30 步达到 100 步 Claude-4-Sonnet 水平"**：sample efficiency 的 framing 巧妙，但 Claude-4-Sonnet 也可能在 30 步设置下表现不同；apple-to-apple 对比缺失
- ⚠️ **"7.34% higher than open-source SOTA"**：依赖于 SOTA 选择和评估时间窗口；2025-09 的 SOTA 与 2025-11 不同
- ❌ 无明显 marketing 话术

### Notes

- DART 的最大价值在于它是少有的"GUI agent RL 全流程开源"工作，体系完整度比 algorithmic novelty 更值得关注。如果要做 GUI agent RL 训练，应该先把这个 codebase 跑起来作为 baseline
- 与 [[2509-UITARS2|UI-TARS-2]] 相比，DART 路线更轻量（7B vs 更大）、更 system-focused；UI-TARS-2 是闭源、更注重数据和 scale
- > ❓ Per-worker model sync 在多机分布式下的 staleness 上界是多少？论文没有分析。如果 worker 数量增加，最旧 worker 与最新 trainer 之间的版本差可能很大，truncated IS 的 C 值需要怎么 scale？
- > ❓ Experience Pool 在 OSWorld 任务上是 task-specific 还是 task-agnostic？如果是 task-specific，对于 unseen task 这个机制完全失效——这是否限制了方法的 generalization？
- > ❓ 30 步 max 训练 + 30 步 max 测试。如果换成 100 步测试，DART-GUI-7B 还能继续涨吗？还是会出现 over-commit / loop 行为？这关系到方法是真的学到了高效决策还是只是被训练 budget 约束了
- 联系自己研究：异步 RL + adaptive data curation 的组合在 VLA / embodied RL 训练里也适用——尤其是 sim2real env 同样有 long episode + sparse reward 特性。可考虑把 DART 的 rollout-wise scheduling + experience pool 思路迁移到机器人 manipulation RL

### Rating

**分数**：2 - Frontier
**理由**：DART-GUI 在 OSWorld-Verified 上 42.13% SR 与开源 +7.34% 的 gain 让它成为 GUI agent RL 当前的重要 reference（见 Strengths 1-4），且代码 / 权重 / 训练框架全开源，短期内会是 baseline 候选。但核心 building block（rollout-wise scheduling、per-worker sync、truncated IS、high-entropy mask）均沿用既有工作，主要贡献是 GUI long-horizon 场景下的组合调优（见 Weaknesses 1、4），且缺 component-wise ablation、未在 AndroidWorld / WebArena 上 cross-benchmark 验证——尚未成为方向必引的 Foundation，应归 Frontier 而非 Archived（仍是活跃 SOTA 对比对象）或 Foundation（缺 paradigm 级影响）。
