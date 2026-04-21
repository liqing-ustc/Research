---
title: "Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation"
authors: [Zipeng Fu, Tony Z. Zhao, Chelsea Finn]
institutes: [Stanford University]
date_publish: 2024-01-04
venue: CoRL 2024
tags: [mobile-manipulation, imitation-learning, manipulation]
paper: https://arxiv.org/abs/2401.02117
website: https://mobile-aloha.github.io/
github: https://github.com/MarkFzp/mobile-aloha
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation
> - **核心**: 把原 ALOHA 双臂遥操作系统装上 AgileX Tracer 移动底盘，操作员用腰部 tether 反向拖动轮子实现「全身」遥操作；用得到的少量演示与现有 static ALOHA 数据 **co-train**，在 7 个家居/厨房/公共场所任务上以 20–50 条演示拿到 80%+ 成功率。
> - **方法**: 16 维动作向量（双臂 14 维关节位置 + 底盘线/角速度 2 维）；ACT / Diffusion Policy / VINN 均可用；co-training 把 static ALOHA 的 825 条 episode 与 mobile 数据按 1:1 mini-batch 采样、对 static 数据的底盘动作 zero-pad；action chunking 还用来吸收底盘 velocity 控制的 latency（前 k-d 步只发臂、后 k-d 步只发底盘）。
> - **结果**: 总成本 32k USD（含算力与电池）；ACT + co-train 在 7 个任务上比 no-cotrain 平均 +34% 绝对成功率，对 Press Button、Turn On Faucet、Flip Shrimp 等需要精细补偿的子任务尤其关键；co-train 比 pre-train+fine-tune 显著更好（95% vs 40%）。
> - **Sources**: [paper](https://arxiv.org/abs/2401.02117) | [website](https://mobile-aloha.github.io/) | [github](https://github.com/MarkFzp/mobile-aloha)
> - **Rating**: 3 - Foundation（tether-to-base 全身遥操作 + naive cross-morphology co-training 的双 insight 已成 bimanual mobile manipulation 的事实起点，被后续 VLA / 双臂工作广泛沿用）

**Key Takeaways:**
1. **「Tether 反向拖底盘」是简洁到不可思议的全身遥操作方案**：操作员两手已被 ALOHA leader 占用，把腰绑在底盘上靠走路驱动 backdrivable 轮子，省掉了 VR/exoskeleton/foot pedal 的全部复杂度，还顺带提供 coarse haptic feedback。这是这篇论文最值得复用的 insight。
2. **跨 morphology 的 co-training 真的能 work，即使任务和臂位都不同**：static ALOHA 是黑桌面、双臂相向；Mobile ALOHA 是动态背景、双臂前向。直接 1:1 采样、不做任何 domain alignment，仍然在几乎所有任务上有正迁移。"motion prior of grasping/approaching" 似乎能跨这种差异迁移。
3. **Co-train ≫ Pre-train+fine-tune**（95% vs 40% on Wipe Wine）：fine-tune 阶段网络会忘掉 static prior。这个对比对 robotics 数据稀缺场景很有指导意义。
4. **Action chunking 不止平滑轨迹，还能吸收异构硬件 latency**：底盘 velocity control 比 position-controlled 臂慢 d 步，就把 chunk 错开发——这是 chunking 在工程层面常被忽略的优势。
5. **数据效率分水岭在 25-50 demos 量级**：35 demo + co-train 已能超过 50 demo no co-train（70% vs 50%）；这是 robot learning 在 "diverse household task" 上能否落地的关键证据点。

**Teaser. Mobile ALOHA 在真实公寓中自主完成的部分任务概览（autoplay reel）。**

<video src="https://mobile-aloha.github.io/resources/mobile-aloha.mp4" controls muted playsinline width="720"></video>

---

## 1. Motivation 与问题定义

模仿学习在 table-top 操作上已经能学到 spreading pizza sauce、slotting battery 这类细致动作，但**真正的家居任务都需要 mobility + bimanual + whole-body coordination 三者同时具备**——比如把重锅放进墙柜：要先走到柜前，再倒退着同时拉两扇门，再两手抬锅入柜。
作者指出有两个 blocker：

1. **硬件**：现有 bimanual mobile 平台 PR2/TIAGo > 200k USD；teleoperation 接口（haptic device、mocap、foot pedal、键盘/手柄）都不足以同时控双臂 + 底盘。
2. **学习**：尚无证据表明 diffusion / transformer policy 在 mobile manipulation 上同样有效；底盘新增自由度后 arm/base 的耦合更复杂，base pose 的小偏差会被放大成 EE 的大偏差。

> ❓ 但 blocker (2) 的论证薄弱——其实当时已有用 diffusion / RT-X 类方法做单臂 mobile 的工作；作者的真实贡献更接近"在自己造的硬件上做了第一个 bimanual mobile imitation learning 系统"，而不是回答了 (2)。

---

## 2. Hardware: Mobile ALOHA

### 2.1 设计目标
- **Mobile**: 与人类步速相当（~1.42 m/s）
- **Stable**: 可操纵锅、柜门等重物
- **Whole-body teleop**: 双臂 + 底盘可同时控
- **Untethered**: 板载电源 + 算力

**Figure 1. 硬件构成。** 左：完整遥操作配置，操作员通过腰部 tether 与底盘相连，双手分别操控两个 leader arm；中：自主执行模式下拆掉 leader 与 tether，仅保留两只 ViperX 300 follower（min/max 高 65/200 cm，水平延伸 100 cm）；右：技术规格。

![](https://arxiv.org/html/2401.02117v1/x1.png)

### 2.2 关键选择
- **底盘**: AgileX Tracer AGV，差速驱动，1.6 m/s 上限，100 kg 载荷，17 mm 离地——可作为低重心配重防侧翻；7k USD，比 Clearpath 同等规格 AGV 便宜 5×。可越 10 mm 障碍 / 8° 斜坡。
- **遥操作机制**: 把人腰部固定在底盘上，反向拖动 backdrivable 轮（vinyl 地面 rolling resistance ~13 N，可接受）。这个设计的**精妙在于"被动反映"**——操作员朝哪走、底盘就往哪走，不需要单独的输入设备占用第三只手；且底盘碰到东西时人能直接感知到 haptic feedback。
- **臂朝向反转**: 原 ALOHA 双臂相向，Mobile ALOHA 改为均朝前——更适合 mobile 场景的工作空间。
- **算力**: 一台 consumer laptop（i7-12800H + 3070 Ti 8GB），3 个 Logitech C922x（2 wrist + 1 top，480×640@50 Hz），1.26 kWh 电池兼配重（14 kg）。
- **总成本**: 32k USD（含 onboard power & compute），与一台 Franka Emika Panda 单臂相当。

### 2.3 可执行任务谱
作者列出了 housekeeping（浇水、用吸尘器、装/卸洗碗机、开关门、开冰箱、洗衣机、铺被子、塞枕头、挂衣服、叠裤子、台灯开关、自充电）、cooking（打蛋、剁蒜、剥菜、倒液体、煎鸡腿、焯菜、炒、装盘）、HRI（握手、递啤酒、刮胡子、铺床）—— **全部用同一套硬件遥操作完成**。这个谱的广度本身就是硬件论证。

---

## 3. Co-training Recipe

### 3.1 背景与动机
单一硬件 + 单一任务的小数据集既不鲁棒（视觉扰动差）又不数据高效。已有工作（RT-X 等）证明单臂跨平台 co-train 有效，但 **bimanual mobile manipulation 没有可用的大数据集**。作者退而求其次：用 [Zhao et al. 2023] 等累计 825 条 static ALOHA episodes（Ziploc 封口、捡叉子、糖纸、撕纸巾、打开瓶盖、乒乓、胶带、咖啡机、传铅笔、Velcro、装电池、传螺丝刀）。
这些 static 数据**和 mobile 任务在任务、背景、双臂朝向上全不一样**——但作者声称仍可正迁移。

### 3.2 训练目标

**Equation 1. Co-training loss.**

$$
\mathbb{E}_{(o^{i},a_{\text{arms}}^{i},a_{\text{base}}^{i})\sim D_{\text{mobile}}^{m}}\!\Big[L\big(a_{\text{arms}}^{i},a_{\text{base}}^{i},\pi^{m}(o^{i})\big)\Big] + \mathbb{E}_{(o^{i},a_{\text{arms}}^{i})\sim D_{\text{static}}}\!\Big[L\big(a_{\text{arms}}^{i},[0,0],\pi^{m}(o^{i})\big)\Big]
$$

**符号说明**：$a_{\text{arms}}\in\mathbb{R}^{14}$ 双臂关节位置（含两个连续 gripper）；$a_{\text{base}}\in\mathbb{R}^{2}$ 底盘线/角速度；$o$ 包含 2 个 wrist + 1 个 top RGB + 双臂关节位置；$L$ 为方法相关的 imitation loss（ACT 用 L1+KL、Diffusion Policy 用 noise prediction、VINN 用 BYOL contrastive）。
**含义**：从 $D_{\text{static}}$ 与 $D_{\text{mobile}}^{m}$ 等概率（各 50%）采样组成 batch；static 没有底盘动作就 zero-pad 成 `[0,0]`；只用 $D_{\text{mobile}}^{m}$ 的统计量做归一化（避免被 static 数据 shift）；忽略 static 的 front camera 凑齐 3 路相机。

> 这个 recipe 的**简洁性**是亮点：没有 dataset re-weighting、没有 morphology alignment、没有 contrastive auxiliary loss——就是 batch 里直接掺。"Worse is better" 的范例。

### 3.3 Action Chunking 的双重作用
所有方法都用 action chunking（k 步未来动作）。除了平滑、降推理频率外，作者用它**消化 base velocity 的延迟**：底盘指令到实际响应延迟 d 步（实测 180° 1m 半径转弯的 open-loop replay 误差 > 10 cm），就让机器人执行 chunk 的前 k-d 步臂动作 + 后 k-d 步底盘动作。这是个工程细节但对成功率影响很大。

---

## 4. Tasks（7 项）

**Figure 2. 任务定义。** 每个任务用图示给出随机化范围与子任务划分。

![](https://arxiv.org/html/2401.02117v1/x2.png)

简述：
- **Wipe Wine** (50 demos): 走去水龙头取毛巾→走回桌→单臂提酒杯、另一臂擦杯底与桌面。考验 mobility + bimanual。
- **Cook Shrimp** (20 demos): 75 秒长 horizon。一手翻虾、另一手倾锅；最后倒进白碗（白桌对比度低）。最难的一项。
- **Rinse Pan** (50 demos): 拧水龙头不锈钢小阀（4 cm × 0.7 cm）——shiny 金属 + 厘米级误差就失败，**强迫策略 visually-servo 来补 base 误差**。
- **Use Cabinet** (50 demos): 把 1.4 kg 重锅放入墙柜。最重锅超单臂 750 g 上限，需双臂协调；开柜时双手抓把手 + 底盘后退。
- **Call Elevator** (50 demos): 起点 15 m 远 + 10 m 宽随机；按 2×2 cm 按钮、绕柱、30 cm 净空入梯。强调长导航 + 精确 whole-body。
- **Push Chairs** (50 demos): 推 5 把椅子；只采集前 3 把，**测试 4/5 把的外推**。
- **High Five** (20 demos): HRI；评估 unseen 衣着与 unseen 人。

> 任务选择体现 taste：每个任务都明确指出某种"single-arm 或 static 做不了 / 做不快"的瓶颈，**任务集本身就是一种叙事**。

---

## 5. Experiments

### 5.1 Co-training 对 ACT 的提升（主结果）

**Table 1. Co-train 改善 ACT 在 7 个任务上的 success rate (%)。**

|  | Wipe Wine | Cook Shrimp | Rinse Pan | Use Cabinet | Call Elevator | Push Chairs | High Five |
|---|---|---|---|---|---|---|---|
| Co-train | **95** | 40 | **80** | 85 | **95** | **80** | 85 |
| No Co-train | 50 | 20 | 0 | 85 | 0 | 0 | 85 |
| Δ | **+45** | +20 | **+80** | 0 | **+95** | **+80** | 0 |

5 / 7 任务有显著提升，平均 +34% 绝对成功率；剩 2 个任务（Use Cabinet、High Five）持平。提升最大的子任务：**Press Button**（5%→100%）、**Turn On Faucet**（0→80%）、**Push 5th Chair OOD**（0→89%）—— 都对应"精度受限 + 误差累积"瓶颈。
作者归因于 static 数据带来的"motion prior of grasping/approaching"，再叠加 wrist camera 提供的 viewpoint invariance。

> ❓ 但 Use Cabinet（已 85%）和 High Five（已 85%）持平更可能是 **ceiling effect** 或这两类任务的精度瓶颈不在那种 prior 能补的层面（开/关柜门是大尺度物理 contact、High Five 是低精度 HRI）。作者没把这个区分讲透。

### 5.2 与 Diffusion Policy / VINN 的兼容性

**Table 2. Co-train 改善多种 IL 方法。**

|  |  | Wipe Wine 子任务（成功率 %） |  |  |  | Push Chairs |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  |  | Grasp Towel | Lift & Wipe | Place Glass | **Whole** | 1st | 2nd | 3rd | **Whole** |
| **VINN+Chunk** | Co-train | 85 | 18 | 100 | **15** | 100 | 70 | 86 | 60 |
|  | No Co-train | 50 | 40 | 100 | 20 | 90 | 72 | 62 | 40 |
| **Diffusion Policy** | Co-train | 90 | 72 | 100 | **65** | 100 | 100 | 100 | **100** |
|  | No Co-train | 75 | 47 | 100 | 35 | 100 | 80 | 100 | 80 |
| **ACT** | Co-train | 100 | 95 | 100 | **95** | 100 | 100 | 100 | **100** |
|  | No Co-train | 95 | 58 | 90 | 50 | 100 | 100 | 100 | 100 |

观察：
- **Diffusion Policy** 整体不如 ACT（Wipe Wine 65% vs 95%），作者推测 50 demo 对 diffusion 不够（它的工作通常 250+ demo）。Co-train 提升 +30% / +20%。
- **VINN+Chunking** 全面落后，Wipe Wine 仅 15%；co-train 在 Wipe Wine 上甚至 -5%。**作者解释**：VINN 仅 co-train 表征（BYOL），动作机制 NN-retrieve 没机制利用 OOD static 数据。
- **ACT** 是赢家。这是作者团队之前的方法，结果不令人意外，但 head-to-head 证据扎实。

### 5.3 Ablations

**Table 3. Co-train 对采样比例不敏感（ACT, Wipe Wine）。**

| Static 比例 | 30% | 50% (default) | 70% |
|---|---|---|---|
| Success | 95 | 95 | 90 |

**Table 4. Co-train > Pre-train（ACT, Wipe Wine）。**

|  | Co-train | Pre-train (10K + fine-tune) | Neither |
|---|---|---|---|
| Success | **95** | 40 | 50 |

Pre-train 反而比纯 in-domain 还差！作者归因 fine-tune 阶段的灾难性遗忘。这个对比对**所有 robot learning 默认"先 pre-train 后 fine-tune"的 pipeline 都是一记警示**。

**Figure 3. 数据效率：co-train 用 35 demo 即超过 no co-train 50 demo（70% vs 50%）。**

![](https://arxiv.org/html/2401.02117v1/x3.png)

### 5.4 User Study

**Figure 4. 8 名研究生用户（4 无 teleop 经验，4 有）在 5 次 trial 内逼近 expert 速度。**

![](https://arxiv.org/html/2401.02117v1/x4.png)

5 trial 内：Wipe Wine 完成时间 46s→28s（-39%），Use Cabinet 75s→36s（-52%）。佐证「腰 tether + leader arm」组合学习曲线陡峭。

### 5.5 Open-loop Replay 对照
作者强调**所有 7 任务在 open-loop replay（同 init、同动作序列）下 whole-task 成功率为 0**，证明 close-loop reactive policy 是必须的——主要 error source 是 base velocity control 的随机性（180° 1 m 半径转弯 replay 平均 > 10 cm 偏差）。

---

## 6. 自主任务展示（来自项目主页）

**Cook Shrimp 完整自主执行：**
<video src="https://mobile-aloha.github.io/resources/skills/cook_shrimp.mp4" controls muted playsinline width="720"></video>

**Wipe Wine：**
<video src="https://mobile-aloha.github.io/resources/skills/wipe_wine.mp4" controls muted playsinline width="720"></video>

**Take Elevator：**
<video src="https://mobile-aloha.github.io/resources/skills/take_elevator.mp4" controls muted playsinline width="720"></video>

**Use Cabinets：**
<video src="https://mobile-aloha.github.io/resources/skills/use_cabinets.mp4" controls muted playsinline width="720"></video>

**Robustness：Wipe Wine 9 次连续试验（8x speed）：**
<video src="https://mobile-aloha.github.io/resources/robustness/wipe_wine_9_trials_8x_speed.mp4" controls muted playsinline width="720"></video>

---

## 关联工作

### 基于
- **ALOHA** (Zhao et al. 2023) — 本工作的硬件与 ACT 算法基础；Mobile ALOHA 直接 fork 其 codebase。
- **Action Chunking with Transformers (ACT)** — Mobile ALOHA 的主力 imitation 算法；本文也展示其对底盘 latency 的处理。
- **RT-X / Open X-Embodiment** — Mobile ALOHA 取用的 825 条 static 数据来自 RT-X release；本文进一步把 cross-embodiment co-training 推到 bimanual mobile 设定。

### 对比
- **Diffusion Policy** — 作为 baseline；50 demo 数据下不如 ACT，但 co-train 同样有效。
- **VINN (+ BYOL)** — 作为 retrieval-based baseline；本文说明 retrieval 机制不易吸收 OOD co-train 数据。

### 方法相关
- **Co-training / Multi-dataset training** — RT-1、[[2307-RT2|RT-2]]、[[2405-Octo|Octo]]、Open X-Embodiment 均探索过单臂跨平台 co-train；Mobile ALOHA 是首次在 bimanual mobile manipulation 上的成功案例。
- **Whole-body teleoperation** — 与 exoskeleton suit、mocap suit、VR + haptic 路线对比，本文走"被动 tether"的极简路线。

### 后续影响（追溯）
- 本文激发了 AgileX 等公司商业化生产 Mobile ALOHA 套件，使硬件门槛进一步降低。
- 数据范式被 [[2410-Pi0|π0]]、RDT、HumanPlus、TeleVision 等后续 VLA / 双臂工作沿用：用 ALOHA 系列硬件采集大规模 bimanual 数据 + cross-embodiment co-training。

---

## 论文点评

### Strengths

1. **Hardware insight 极其简洁有效**：tether-to-base 把全身遥操作降维成"操作员走路"，绕过 VR/exoskeleton/foot-pedal 的所有复杂度。这种"被动接入"思路可推广到任何需要释放双手又要控其他自由度的场景。
2. **Co-training across morphology 的强证据**：static ALOHA（黑桌、相向双臂）和 Mobile ALOHA（动态背景、前向双臂）任务/视觉/动力学都不同，1:1 简单混合 batch 仍然 +34% 平均。这是对"robot data 是否可跨硬件复用"的有力数据点，比 RT-X 类工作的 single-arm 设定更进一步。
3. **Co-train > Pre-train 的对照实验**（95 vs 40）改变了我对 fine-tune-everything pipeline 的判断——在数据稀缺机器人场景，joint optimization 能避免灾难性遗忘。
4. **Action chunking 用于吸收硬件异构 latency** 的工程 trick 优雅且实用，对未来异构 mobile platform 通用。
5. **任务集叙事性强**：每个任务都对应明确的"单臂/static/手动控制做不到"的瓶颈，不是堆 SOTA 数字。
6. **完全开源（HW + SW + tutorial）**，复现门槛降到 32k USD。

### Weaknesses

1. **样本极少**：每个 cell 是 20 trial（Cook Shrimp 仅 5 trial），二项分布下置信区间宽（如 5% vs 0% 的差异在 n=20 时不显著）。"+95%" 这种醒目数字需要打折看。
2. **Diffusion Policy 评估不公**：作者承认 50 demo 对 diffusion 不够，**却没在更高数据量下重做 head-to-head**。"DP 不如 ACT" 的结论被这个 confound 削弱。
3. **Co-training 的迁移机制只是猜测**："motion prior of grasping/approaching" + "wrist camera invariance" 是事后归因，没有 controlled ablation（比如 mask 不同 modality 的 static 数据看哪部分贡献最大）。
4. **任务空间窄且 in-distribution**：所有评估都在数据采集环境内做（厨房、电梯、走廊），没有 test on unseen environment 的结果——除了 Push Chairs 的 OOD 4/5 椅子和 High Five 的 unseen 人。**真正的家居泛化没有被测**。
5. **7 个任务都是单任务策略**（每任务一个网络），没有 multi-task / language-conditioned 实验。作者在 limitation 中承认。
6. **VINN 失败的归因偏 surface**："只有表征 co-train" 是事实，但没有尝试用 retrieval-augmented 的 alternative（比如基于 mobile data 的 retrieval pool 与 static 表征的组合），结论略草率。
7. **"Untethered" 名不副实**：autonomous 模式确实 untethered，但 teleop 模式人必须绑在底盘上，这其实是新的 tether。论文叙述中 untethered 主要指自主模式，这点容易让读者混淆。

### 可信评估

#### Artifact 可获取性
- **代码**: `MarkFzp/mobile-aloha`（teleop + 数据采集）+ `MarkFzp/act-plus-plus`（ACT/DP/VINN 训练）开源；属于 inference + training。
- **模型权重**: 未发布预训练 checkpoint。
- **训练细节**: 论文 Appendix A.3 提供 ACT/DP/VINN 的超参；数据配比、batch size、采样比例均披露；训练步数披露不全（pre-train 写了 10K，主实验未明确 total steps）。
- **数据集**: Mobile ALOHA 数据未直接发布；static ALOHA 数据已通过 RT-X release 公开。Hardware 装配教程在项目页 Google Doc。

#### Claim 可验证性
- ✅ **硬件 32k USD 总成本可达**：BoM 在项目页公开；社区已多次复现（含 AgileX 推出的商业版）。
- ✅ **Co-train + 50 demo 在 5 个任务上达到 80%+ 成功率**：有视频证据 + 多次 trial（虽 n=20 偏小），且开源代码可复现。
- ✅ **Co-train > Pre-train 在 Wipe Wine 上 95% vs 40%**：单任务 ablation 清晰，n=20 trial。
- ⚠️ **"co-train 在跨 morphology 仍正迁移" 是普适结论**：仅在 1 套 hardware 配对（static→mobile ALOHA）上测试，且任务也都来自同一套 lab；推广到任意硬件对仍需更多证据。
- ⚠️ **VINN 弱于 ACT/DP**：仅 2 个任务，且 VINN 已知对小数据敏感，可能是实现质量而非方法本质。
- ⚠️ **Tether 遥操作"对人友好"**：仅 8 名 CS 研究生 user study；老人/小孩/不同身高人群的可用性未知。
- ⚠️ **Cook Shrimp 40% 成功** 的 n=5 trial 统计意义弱（95% CI 约 5%–85%）。

### Notes

- **真正可推广的 idea = tether-to-base + naive batch mixing + chunking-for-latency**。三者都简洁、各自都不依赖具体硬件。
- **对我研究的影响**：在 VLA 数据采集层面，Mobile ALOHA 的 hardware insight 提示——增加自由度时优先考虑"被动接入"（让人/环境物理上 backdrive 而非新增主动输入设备）。这对未来 humanoid teleop 也适用（参见 HumanPlus、TeleVision 的 retargeting 路线 vs 本文的 tether 路线对比）。
- **可借鉴 + 警示**："简单方法 + 50 demo + co-train" 这个 formula 已成 robotics field 的隐式 default。但本文 n=20 trial 的统计 noise 与 in-distribution-only 评估值得记住——下次看到类似 "+34% 平均成功率" 的工作，先问 trial 数与是否 unseen env。
- ❓ Tether 是否会限制 mobile manipulation 的 demonstration distribution？比如人不会做"快速冲刺 + 急停"，所以 policy 也学不到——这是潜在的 demonstration bias。
- **Followups to check**: HumanPlus、TeleVision、[[2410-Pi0|π0]] 是否有 head-to-head 与 Mobile ALOHA 的对比；是否有人系统研究过"co-train data ratio 在哪个数据规模拐点开始下降"。

### Rating

**分数**：3 - Foundation
**理由**：tether-to-base 全身遥操作 + naive cross-morphology co-training 两个 insight 简洁且可复用，已被 HumanPlus / TeleVision / RDT / [[2410-Pi0|π0]] 等后续 bimanual & VLA 工作沿用为 data collection 与训练范式；32k USD 开源硬件 + AgileX 商业化套件使它成为该方向的事实起点（笔记「后续影响」段已记录）。相比 Frontier(2)，它不是某代 SOTA 而是范式奠基；相比 Archived(1)，尚未被更通用的 teleop/co-training 方案取代，仍是方向必读。
