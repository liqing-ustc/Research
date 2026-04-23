---
title: "VLNVerse: A Benchmark for Vision-Language Navigation with Versatile, Embodied, Realistic Simulation and Evaluation"
authors: [Sihao Lin, Zerui Li, Xunyi Zhao, Gengze Zhou, Liuyi Wang, Rong Wei, Rui Tang, Juncheng Li, Hanqing Wang, Jiangmiao Pang, Anton van den Hengel, Jiajun Liu, Qi Wu]
institutes: [Adelaide University, AIML, Tongji University, ManyCore, Zhejiang University, Shanghai AI Lab, CSIRO Data61]
date_publish: 2025-12-22
venue: arXiv
tags: [VLN, navigation, embodied-reasoning]
paper: https://arxiv.org/abs/2512.19021
website: https://sihaoevery.github.io/vlnverse/
github: https://github.com/william13077/IAmGoodNavigator
rating: 2
date_added: 2026-04-23
---

## Summary

> [!summary] VLNVerse: A Benchmark for VLN with Versatile, Embodied, Realistic Simulation and Evaluation
> - **核心**: 在 Isaac Sim 之上构建 263 个手工制作的可交互 USD 3D 场景（首次真正意义上的"new scenes"），把 VLN 从静态 MP3D 图搜索扩展成全物理、连续控制、多任务统一的 embodied benchmark
> - **方法**: 三层解耦架构（Simulator / Environment / Task&Dataset）+ 五任务统一分类（fine-grained / coarse-grained / visual-reference / long-horizon / dialogue）+ MoE-based unified baseline GAMA（基于 SAME 的 state-adaptive routing）
> - **结果**: 现有 MLLM 代理在 Strict 物理设置下 SR 相比 Tel-Hop 下降 ~10pp（25.5→16.7），CR（Collision Rate）暴露了所有依赖 waypoint 预测的方法对物理 embodiment 的脆弱性；foundation model 在新场景下泛化明显退化（InternNav-N1 仅 17.5% coarse SR）
> - **Sources**: [paper](https://arxiv.org/abs/2512.19021) | [website](https://sihaoevery.github.io/vlnverse/) | [github](https://github.com/william13077/IAmGoodNavigator)
> - **Rating**: 2 - Frontier（显著推进 VLN 仿真维度——场景多样性 + 物理约束 + 任务统一是过去 5 年 VLN 停滞的真正瓶颈；但能否成为 de facto 标准取决于社区采纳）

**Key Takeaways:**
1. **Scene diversity 瓶颈终于被打破**：Table 1 揭示了过去几乎所有"新 VLN benchmark"贡献 0 个 new scenes——一直在 90 个 MP3D 场景上反复重标注；VLNVerse 一次给出 263 个全新手工可交互 USD 场景
2. **"Ghost agent" 幻觉被揭穿**：MLLM 代理在 Tel-Hop 下看似合理的 reasoning，到 Strict 物理设置就因为 collision 直接失败——表明当前 "waypoint + 大模型推理" 范式严重 under-specify 了 embodiment
3. **Online fine-tuning 的复权**：VLN-BERT 从零 shot SR 3.4% → 10 epochs 在线 fine-tune 到 25.7%，验证了 benchmark 支持 online training 的实用价值，也意味着纯 offline 训练的 VLN 模型对新环境的迁移性比过去认为的更差
4. **统一 task taxonomy 的务实选择**：五任务覆盖 granularity / modality / horizon / interaction 四个轴，但本质上都建在同一套 trajectory pool 上（coarse/visual-reference/dialogue 共享轨迹），实现上比 claim 的 "unified" 更 lightweight

---

## Background & Motivation

**作者诊断了 VLN 领域的四重耦合失败**：

1. **Simulator 瓶颈**：MP3D 把空间离散成 navigation graph，agent "teleport" 在 nodes 间跳跃；Habitat / Gibson 引入物理但用简化引擎，不支持 realistic robot kinematics；AI2-THOR 强交互但空间窄小，不适合长距离导航
2. **场景多样性停滞**：Table 1 的 "New Scenes #" 列是全文最有力的证据——R4R、RxR、REVERIE、SOON、VNLA、HANNA、CVDN、VLN-CE、Robo-VLN 全部贡献 0 个新场景，都在 90 个 MP3D 场景上反复重标注。VLN-PE 号称 101 scenes，但实际是 90 MP3D + 10 GRScene + 1 自定义扫描，前 90 个只是格式转成 USD，拓扑结构完全相同
3. **Task fragmentation**：R2R（fine-grained）、REVERIE（coarse）、HANNA（dialogue）、LH-VLN（long-horizon）各自为政，模型在一个 dataset 上开发后很难迁移到另一个
4. **数据规模与现代训练范式脱节**：静态、固定大小的数据集，绑死在 discrete environment + fixed modality（RGBD），无法服务 MLLM 式的大规模预训练

> ❓ 作者把"new scene"定义得很严（"distinct geometry and visual appearance, file format conversion doesn't count"）。这个定义本身是否公平？MP3D 的 90 个场景都是真实公寓扫描，而 VLNVerse 的 263 个是 hand-crafted USD——后者的视觉/布局分布是否充分反映真实世界？论文没有正面回答这一点。

---

## The VLNVerse Architecture

三层解耦设计：Simulator Layer（embodiment + control + perception APIs）/ Environment Layer（3D assets + scene graph）/ Task & Dataset Layer（task definition + data pipeline + eval）。

**Figure 1. VLNVerse 的三层架构总览**
![](https://arxiv.org/html/2512.19021v1/x1.png)

### Simulation Layer

核心贡献不是 engine 本身（用 Isaac Sim），而是**针对 VLN 的高层 API 抽象**：

- **Parameterizable Agent Embodiment**：agent 是有 physical footprint 的实体（可配 cylinder 高度/直径），不再是 floating camera
- **Physics-aware Controller**：统一的 controller 接口，支持连续控制和 camera articulation，底层走 physics 而非 grid teleport
- **Modular Perception Rig**：可 attach/detach RGB、Depth、LiDAR；所有参数（采样频率、分辨率、FoV）暴露给用户

这里的 contribution 是 engineering——把 Isaac Sim 的原生 robotics API 包装成研究者易用的 VLN workflow。相比 VLN-PE（同样基于 Isaac）的差异，论文强调 "first to harness Isaac Sim specifically for full-stack embodied navigation"，而 BEHAVIOR-1K / Arnold / GRUtopia 等 Isaac-based 工作聚焦 manipulation。

### Environment Layer: 263 Interactive USD Scenes

263 个**hand-crafted** USD 场景是本工作最硬的贡献：

- 每个 object 都是 physics-aware 资产：带 mass、friction、reflectivity
- 镜子真实反射、碰撞非二值（agent 会基于 mass/velocity 被 deflected）
- 每个场景生成 2D occupancy map（导航地面）+ spatial-semantic scene graph（object 带坐标+semantic label + "on/in/near" 关系）

scene graph 的存在是 data pipeline 的关键：它提供了 ground truth prior，让后续的 instruction generation 可以 verify against。

**Figure 2. Navigable area / room count / trajectory length 分布**
![](https://arxiv.org/html/2512.19021v1/x2.png)

场景规模和复杂度分布广，不是简单的 uniform spread——面积越大、房间越多，path 越长的直观关系被 scatter plot 印证。

### Task & Dataset Layer

**5 个任务涵盖 4 个维度**：

- **Granularity**：fine-grained（R2R-style 步骤指令）+ coarse-grained（REVERIE-style goal-oriented）
- **Modality**：visual-reference navigation（给 agent 一张 target object 的照片）
- **Horizon**：long-horizon（连接 2-3 个 sub-task）
- **Interactivity**：dialogue-based（可向 oracle LLM 查询）

**数据生成 pipeline 是两阶段**：

1. **Physics-aware path sampling**：在 occupancy map 上 A\* 搜索，但根据 agent 物理尺寸 **dilate** occupancy map，保证 sampled path 对物理 agent 可达
2. **3-stage instruction generation**：
   - *Prior-based init*：scene graph 给出 factually-grounded 描述种子（如 "the mirror on the sink"）
   - *Collaborative AI*：三 agent 分工——Describer（看视觉描述环境）/ Verifier（对照 scene graph 查 hallucination）/ Synthesizer（按 task 类型 + 风格合成最终指令）
   - *Human verification*：志愿者打分 clarity / naturalness / accuracy

**Table 2. Data volume（episodes per task, scenes in brackets）**

| Taxonomy | Train | Seen val | Unseen val | Test |
| --- | --- | --- | --- | --- |
| Fine-grained | 3963(177) | 423(157) | 825(33) | 1325(53) |
| Coarse-grained | 11895(177) | 1269(162) | 2505(33) | 3975(53) |
| Visual reference | 11895(177) | 1269(162) | 2505(33) | 3975(53) |
| Long horizon | 11946(177) | 1329(177) | 2475(33) | 3975(53) |
| Dialogue | 11895(177) | 1269(162) | 2505(33) | 3975(53) |

注意：coarse-grained / visual-reference / dialogue **共享同一套轨迹**（visual-reference 额外加目标图，dialogue 额外加 LLM oracle）；coarse-grained 每条轨迹生成 formal / natural / casual 三种风格指令。fine-grained 数据量显著小（因为步骤指令生成更昂贵）。

### 新 Metric: Collision Rate (CR)

在标准 SR / OSR / SPL / NE / TL / nDTW 之上，**引入 Collision Rate 量化 embodiment 违反**。这是论文实证部分最有用的工具——后文揭示 CR 是当前所有 "waypoint + LLM reasoning" 方法的最大 bottleneck。

对 long-horizon 任务引入 SR_n（到达第 n 个目标的条件成功率）+ SR_All（全程完成）。

---

## GAMA: Unified Multi-task Baseline

作者提出的统一 baseline GAMA（General-purpose Agent for Multi-task Navigation），基于 **SAME (State-Adaptive MoE)** 路由机制。

**Figure 5. GAMA framework**
![](https://arxiv.org/html/2512.19021v1/fig/GAMA-Horizontal.png)

**架构要点**：

- **Unified Transformer Backbone**：冻结的预训练 VLM（LXMERT / LLaMA-VID 等）编码 RGB/Depth frames 和 instruction，trainable fusion module Ψ 融合
- **Unified Action Decoder**：把 discrete action 视作连续空间中的 Dirac δ 先验——$p(a|\psi(V_t, L_t)) = \sum_k \alpha_k \cdot \delta(a - a_k)$，$a_k$ 是 4 个 action primitives（含 stop），$\alpha_k$ 是 MLP 输出概率；连续控制下再回归一个 residual $\Delta a$ 加在最可能的 primitive 上
- **State-Adaptive MoE（SAME, Zhou et al. 2024a）**：在**时间维度**路由而非 token 维度——每次 agent 进入新位置才重新选 experts，降低 token-wise routing 的开销，同时适配 navigation 的 sequential 结构

这个设计的核心 insight：navigation 是 multi-step rollout，linguistic cues 和 visual observation 在不同时间步的相关性不同，传统 token-wise / task-wise MoE 都与 temporal structure 不对齐。SAME 在 state level 路由，既压低计算又契合任务结构。

> ❓ GAMA 里"unified" 的程度存疑。它能 handle 多种 action space 和 observation 是因为有 unified decoder + fusion，但真正在 5 个任务上**同时**训练还是**按任务分开**训练？论文正文没有明确说明 joint training 协议，需读 Appendix 确认。

---

## Experiments

### Offline Training Baselines (Table 3)

对比 CMA / RDP / Seq2Seq / HNR / GAMA。在 fine-grained test split，GAMA 拿到 **SR 37.72 / SPL 33.85**（最高），RDP 紧随（36.38 / 32.29），CMA 和 Seq2Seq 落在 28-30%。在 coarse-grained test split GAMA SR 37.20，RDP 36.60。

**人类 baseline 为 86% SR**，机器最好 37.7%——差距巨大，benchmark 确实有挑战性。

### Foundation Model Zero-shot (Table 4)

测试 InternNav-N1 / NFM / UniNavid 的 zero-shot 迁移能力：

| Model | Fine SR | Coarse SR | CR (coarse) |
| --- | --- | --- | --- |
| InternNav-N1 | 28.95 | 17.51 | - |
| NFM | 51.59 | 38.02 | 24.22 |
| UniNavid | 45.74 | 29.68 | 15.59 |

NFM 表现最好，但 CR 24.22%——每 4 次导航就有 1 次严重碰撞。SPL 也显著下降（InternNav-N1 coarse 16.54、UniNavid 13.47），说明即使找到目标，路径效率也崩溃。

### Zero-shot MLLM Agents under Physics (Table 5) — **论文最重要的发现**

QwenVL3-4B 作 LLM backbone，对比 text-history (NavGPT-style) 和 map-history (MapGPT-style) agent，在两个 setting 下：

- **Tel-Hop**：agent 可以 teleport between waypoints，只在目标点检查 collision
- **Strict**：完全物理约束，agent 被碰撞 > 0.1m displacement 即失败

Map-history agent 在 fine-grained 任务上：Tel-Hop SR **25.53% → Strict 16.67%**，下降 ~9pp，CR 飙升到 43%+。

加 CoT 后 Tel-Hop 进一步提升到 42.19%，但 Strict 几乎没改善（16.67%）——**物理碰撞才是真正的 bottleneck，而不是 reasoning**。CoT 只能在 idealized 世界里帮助，遇到 embodiment 就失效。

这是对当前 VLN + MLLM 研究范式的**直接打脸**：过去几年"waypoint + LLM reasoning"的成功大多建立在 discrete graph 的 teleport 假设上，一旦加上物理约束就暴露出"这些 agent 根本没学 embodiment"。

### Dialogue & Long-horizon (Table 6)

**Dialogue 极有效**：map-history agent 在 coarse-grained 的 SR 从 42.2% 跃升到 67.0%——oracle LLM 提供的环境信息大幅降低歧义。这也侧面验证了 VLNVerse 场景的高质量渲染足以让 LLM 理解并提供有效指引。

**Long-horizon 严重衰减**：SR_1 = 77.1%（到第一个目标），SR_2 = 46.3%，SR_3 = 10.6%。累积误差和 state tracking 失败让多阶段规划崩盘。

### Online Fine-tuning (Table 7)

取 VLN-BERT（在 VLN-CE/Habitat 预训练）zero-shot 测试：SR 3.4%（近乎随机）。

在 VLNVerse 的 30 个场景上 online fine-tune：
- 1 epoch → SR 11.1%
- 10 epochs → SR 25.7%, CR 从 52.9% 降到 19.5%

验证了 benchmark 对 online training 的原生支持，也说明**纯 offline 预训练的 VLN 模型对新物理环境的迁移性远不如过去认为的那样好**。

---

## 关联工作

### 基于
- **Isaac Sim (NVIDIA, 2021)**：底层物理 + 渲染引擎
- **SAME (Zhou et al., 2024a)**：state-adaptive MoE 路由机制，GAMA 的核心
- **HNR (Wang et al., 2024b)**：GAMA backbone 用的 hierarchical neural radiance representation（用于 lookahead exploration）
- **Universal Scene Description (Pixar, 2021)**：263 场景的资产格式

### 对比（同用 Isaac Sim 的 benchmark）
- **[[2507-VLNPE|VLN-PE]] (Wang et al., 2025)**：最直接的对手，同样基于 Isaac + "Full" physics，但只有 1 个真正新场景（剩下的是 MP3D 格式转换）；VLNVerse 的 263 vs VLN-PE 的 ~1 是本工作最硬的差异化
- **[[2403-Behavior1K|BEHAVIOR-1K]] / Arnold / GRUtopia**：Isaac-based，但聚焦 manipulation，VLNVerse 首个聚焦 full-stack navigation

### 对比（前 Isaac 时代的 VLN benchmark）
- **R2R / R4R / RxR / REVERIE / SOON / VNLA / HANNA / CVDN**：全部 0 new scenes，在 90 MP3D 场景上重标注
- **VLN-CE / Robo-VLN**：Habitat-based 引入连续控制，但 "Specialized" 物理 + 0 new scenes
- **ALFRED / TEACh / DialFRED**：AI2-THOR-based 强交互但空间窄
- **[[2412-LHVLN|LH-VLN]] (Song et al., 2025)**：long-horizon 单任务专用 benchmark，216 scenes 但仍是 Habitat + graph action

### 方法相关
- **[[2305-NavGPT|NavGPT]] / MapGPT / NavGPT-2**：被测的 MLLM-based agents，text-history / map-history 的代表
- **[[2509-NavFoM|NFM]] / UniNavid / InternNav-N1**：被测的 navigation foundation models
- **VLN-BERT (Hong et al., 2022)**：online fine-tuning 实验的受试模型
- **Scheduled Sampling (Bengio et al., 2015)**：online training 时混合 oracle/prediction action 的策略

---

## 论文点评

### Strengths

1. **真正解决了 scene diversity 瓶颈**：Table 1 把整个 VLN 领域的数据贡献摊开看，暴露了多年"benchmark 论文"其实都在原地打转的事实。263 个 hand-crafted USD 场景（且都是 physics-aware interactive assets）是过去 5 年未见的真实增量
2. **Strict vs Tel-Hop 的对照设计非常有说服力**：把"idealized reasoning"和"physical embodiment"在同一模型上切开比较，干净地 diagnose 出 MLLM agent 的真正瓶颈。这是比单纯报告 SR 数字更有研究价值的做法
3. **CR metric 的引入务实**：给 embodiment 给了可量化的维度，让 "sim-to-real gap" 不再是 hand-wavy 的叙事
4. **Online fine-tuning 支持 + scalable data pipeline**：不只是 static dataset，而是把 "任意规模的 on-demand 数据生成 + online training" 作为一等公民。这种架构对现代 LLM 预训练思路友好
5. **三 agent 协作 + human verification 的 instruction pipeline 质量控制严格**：Describer / Verifier / Synthesizer 分工 + 对照 scene graph 查 hallucination，比纯 GPT 生成 + 人工抽检 更可信

### Weaknesses

1. **Hand-crafted USD 场景的真实性存疑**：相比 MP3D 的真实公寓扫描，263 个手工场景的视觉和布局分布可能有系统性偏差。论文没有做 sim-to-real 的真机实验来验证这一点
2. **GAMA 的"unified"程度不够清晰**：论文没有正面说明是 5 个任务 joint training 还是分开训练。如果是分开，那"统一"主要是 architectural，不是训练上的
3. **GAMA 相对 RDP 的提升边际**：fine-grained test SR 37.72 vs 36.38 的差距很小，GAMA 的架构贡献（SAME）在数值上没有展现出对 RDP (diffusion policy) 的清晰优势
4. **Long-horizon 的 sub-task 构造太简单**：只是 random sampling 2-3 个 coarse-grained 目标拼接，没有 "需要先做 A 才能 B" 的依赖结构——这与真实 lifelong learning 场景的差距较大
5. **Dialogue-based 的 oracle 是 LLM，而不是人**：oracle 的 "correct answer" 由 LLM 基于 scene graph 生成，本质是 scene graph 查询的代理。agent 学会的是"问对问题"而不是"跟人对话"——对真实 human-robot interaction 的覆盖有限
6. **没测 MLLM 的参数规模敏感度**：QwenVL3-4B 是小模型，更大的 MLLM（72B+）在 Strict 设置下的表现如何？论文没有跑 ablation

### 可信评估

#### Artifact 可获取性
- **代码**：部分公开——`william13077/IAmGoodNavigator` 是一个"Interactive Playground"demo 仓库（10 episodes 用于交互体验），提到 generation pipeline 在 `william13077/VLNTube`。主 benchmark 代码和完整评估框架是否开源未在正文明确（正文说 "All implementations, datasets, and evaluation protocols will be made publicly available" 但是 future tense）
- **模型权重**：未说明 GAMA 权重是否发布
- **训练细节**：部分公开——AdamW lr=1e-4, 4×RTX4090 GPUs, DataParallel, batch 4, 2 天收敛；但具体模型超参（layers, dim, MoE expert count 等）未在正文披露
- **数据集**：部分公开——Hugging Face datasets `Eyz/TataServices`（scenes）、`Eyz/TataMeta`（meta）、`Eyz/SceneSummary`（scene graph）已发布

#### Claim 可验证性
- ✅ **"263 new scenes, all hand-crafted USD with full physics"**：数据集已在 HF 公开，可独立核验场景数和资产属性
- ✅ **"MLLM agent Strict vs Tel-Hop drop"**：Table 5 有对齐的 baseline 对照，CR 指标提供额外证据
- ✅ **"Online fine-tuning from 3.4% → 25.7%"**：Table 7 设置清晰，可复现
- ⚠️ **"First to unify all 4 taxonomies"**：严格说 coarse / visual-ref / dialogue 共享同一套 trajectory，实际是**一套数据配三种任务头**——"unification" 的程度比 claim 显得浅
- ⚠️ **"263 unique scenes with unique geometry and visual appearance"**：作者自己定义了 "new scene" 的严格标准，但 263 个 hand-crafted scenes 彼此之间的多样性（是否只是同一模板的参数化变体？）没有量化分析
- ⚠️ **"GAMA is the best unified baseline"**：和 RDP 的差距（37.72 vs 36.38）处在统计噪声范围内，没有 multiple seeds / error bars
- ❌ **"Narrows sim-to-real gap"**：论文没有任何真机部署实验支撑这个 claim，仅靠"用 Isaac Sim 物理引擎"这个前提推断

### Notes

- **对自己研究的关联**：如果做 VLN 或 embodied navigation，这个 benchmark 值得关注并跟踪其被社区采纳的程度；特别是 Strict vs Tel-Hop 的对照设计可以移植到其他 embodied 任务的评估中
- **核心 insight 值得记住**："waypoint + LLM reasoning" 在 discrete graph 上的成功可能是一种 evaluation artifact——物理约束加上后，当前 agent 的性能 drop 在 10pp 级别。这对 VLA / embodied foundation model 的训练数据构造有启示：**纯 offline / 纯 discrete 数据可能系统性低估 embodiment 挑战**
- **开放问题**：VLNVerse 的 263 个 hand-crafted 场景的 visual realism 是否足够缩小到真实环境的 gap？这本质上是 "手工 USD 场景 vs 真实 3D 扫描" 的 trade-off——前者 physics 完整但视觉可能偏 cartoonish，后者视觉真实但物理贫瘠。未来工作应该融合两者

### Rating

**Metrics** (as of 2026-04-23): citation=1, influential=0 (0%), velocity=0.25/mo; HF upvotes=N/A（论文尚未被任何 HF model/dataset/Space README 引用）; github 88⭐ / forks=2 / 90d commits=2 / pushed 13d ago

**分数**：2 - Frontier

**理由**：
- **为什么不是 1**：论文不是 incremental——Table 1 清晰展示了 VLN 领域过去多年 0 个新场景的贡献停滞，263 个 hand-crafted USD + 五任务统一 + Strict/Tel-Hop 对照 + CR metric 同时推进了多个维度；Strict 设置下 MLLM 的暴露是真正的 research insight 而非 benchmark puffery
- **为什么还不是 3**：发布才 4 个月（2025-12-22），citation=1 + HF 未收录 + github 88⭐（对一个 Isaac-based benchmark 算中等），属于 early signal 阶段尚不足以判断是否会成为 de facto 标准；VLN-PE 等同类工作刚出来一年就进入了大家的讨论，VLNVerse 能否取代尚待观察；且目前主代码尚未完全开源（只放了 playground demo），完整 adoption 还需等待

%% 发布 < 3 个月的特例处理：citation 绝对数不作为降档依据；influential_citations=0、HF 未收录、github 88⭐ 都是中性信号——Isaac-based VLN benchmark 本身是小众方向。等 6 个月后重新看 adoption 情况。%%
