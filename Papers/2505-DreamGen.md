---
title: "DreamGen: Unlocking Generalization in Robot Learning through Video World Models"
authors: [Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xiaohui Zeng, Kaiyuan Zheng, Ruijie Zheng, Ming-Yu Liu, Luke Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, Linxi Fan]
institutes: [NVIDIA, University of Washington, KAIST, UCLA, UCSD, CalTech, NTU, University of Maryland, UT Austin]
date_publish: 2025-05-19
venue: arXiv preprint
tags: [world-model, VLA, manipulation]
paper: https://arxiv.org/abs/2505.12705
website: https://research.nvidia.com/labs/gear/dreamgen
github: https://github.com/NVIDIA/GR00T-Dreams
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] DreamGen: Unlocking Generalization in Robot Learning through Video World Models
> - **核心**: 把 SOTA 视频生成模型当作"合成数据生成器"而非实时 planner，用它产生的 photorealistic 机器人视频 + 反推的 pseudo-action 训练 visuomotor policy，在仅有单一 pick-and-place teleop 数据下解锁 22 个新动词和 10 个新环境的泛化
> - **方法**: 4 步 pipeline——(1) LoRA fine-tune video world model（默认 WAN2.1）于目标 embodiment；(2) 给 initial frame + language instruction，rollout 合成视频；(3) 用 LAPA (latent action) 或 IDM 抽 pseudo-action，得到"neural trajectory"；(4) 用 neural trajectory co-train 或单独训 visuomotor policy（GR00T N1 / π0 / Diffusion Policy）
> - **结果**: RoboCasa 上 neural trajectory 数量与策略性能呈 log-linear；纯 NT 训练即可达 20.55% / 24 task；GR1 humanoid 在新行为上 11.2% → 43.2%，新环境上 0% → 28.5%；并提出 DreamGen Bench，IF + PA 分数与下游 policy success rate 正相关
> - **Sources**: [paper](https://arxiv.org/abs/2505.12705) | [website](https://research.nvidia.com/labs/gear/dreamgen) | [github](https://github.com/NVIDIA/GR00T-Dreams)
> - **Rating**: 3 - Foundation（把 "video world model as offline data engine" 定义成一个新的 sub-paradigm，配套 DreamGen Bench + log-linear scaling 证据，是 VLA × World Model 交叉方向的 must-cite）

**Key Takeaways:**
1. **Video world model 当作 offline data engine 的范式胜过当 real-time planner**：作者明确把 video model 与 policy 解耦，让前者保留所有 internet-video prior 同时不被 latency 约束，这是与 UniSim 一类 test-time rollout 路线的核心分歧
2. **"Neural trajectory" 是关键抽象**：合成视频 + pseudo-action 的二元组同时绕开了 sim2real gap（视频本身是 photorealistic）和 manual scaling 瓶颈（视频 prompt 比 teleop 便宜得多），以 1500 张 L40 GPU × 54h 换出 240k 样本
3. **LAPA 和 IDM 在数据充足时基本等价**，作者最终选 IDM 作 default，因为它支持纯 NT 训练
4. **DreamGen Bench 给出"video model quality → robot policy success" 的可量化代理**，使视频生成研究可以 directly contribute 到 robotics，无需物理机器人；fine-tuned > zero-shot 巨大（WAN2.1 zero-shot 0% → sft 64.9% PA）

**Teaser. DreamGen 让 2D visuomotor policy 在仅采集单一 behavior（pick & place）单一 environment 的 teleop 数据下，泛化到新行为和新环境。**

![](https://arxiv.org/html/2505.12705v2/x1.png)

---

## 1. 动机：为什么不让 video world model 当 planner？

机器人 foundation model 依赖 manual teleop，每个新任务/环境都要重新采。Sim 数据便宜但 sim2real gap 大，而且对 liquid / articulated object / 工具操作这类任务几乎无法 simulate。前期工作如 UniSim、UniPi 把 video world model 当作 test-time planner，但 DreamGen 选了相反路径：

- **Video model 离线生成数据**，不参与 inference，因此可以用最大、最慢的 SOTA 模型（WAN2.1, Cosmos）
- **Policy 仍是常规小模型**（GR00T N1 / π0 / Diffusion Policy），保持实时性
- 二者解耦后，policy 完全继承 video model 的 physical prior、naturalistic motion、language grounding，而不被推理速度卡住

> ❓ 这个解耦虽然逻辑漂亮，但本质上是"用 offline compute 换 online generalization"。如果未来 video model 的 inference 提速 10×（如 distillation），real-time rollout 路线会不会反过来更优？

---

## 2. Pipeline 细节

**Figure 2. DreamGen 4 步流程概览。**

![](https://arxiv.org/html/2505.12705v2/x2.png)

### 2.1 Video World Model Fine-tuning

- **Base model**: 默认 WAN2.1（实验也覆盖 Hunyuan / CogVideoX / Cosmos）
- **Adaptation**: LoRA（rank=4, alpha=4），主要为防止 forget pretraining 的 internet-video prior
- **Multi-view 处理**: RoboCasa / DROID 这类有多视角的数据集，作者把 left / right / wrist 三视角拼成 2×2 grid，左下黑像素填空，再 fine-tune
- **训练规模**：WAN2.1 在 GR1 上 75 epoch / batch 64；RoboCasa 100 epoch / batch 32；DROID 仅 5 epoch / batch 64（数据量大）；SO-100 200 epoch / batch 8
- **判停指标**：instruction following + physics following（DreamGen Bench 里同样的两个指标）

### 2.2 Rollout

给 initial frame + language instruction（自由文本），video model 生成 video。
- Sim 任务用 simulator 截 initial frame 并 randomize 物体位置
- Real 任务手动拍 initial frame
- 环境泛化实验：从 10 个新环境拍 initial frame，但 video model **完全没在这些环境上训练过**

### 2.3 Pseudo-Action Extraction

两条路：

- **LAPA (latent action)**: 在 video 上无监督学一组 latent action token（codebook size 8, sequence length 16, batch 1024 trained 100k steps），覆盖 GR-1 Teleop / DexMG / DROID / RT-1 / Bridge-v2 / RoboCasa / Agibot-Alpha / Sth-v2 / Ego4D 共 438M 帧 / 5721 小时混合数据
- **IDM (inverse dynamics model)**: 给定 (frame_t, frame_{t+k}) 预测 action，需要少量 ground-truth (frame, action) pair 训练

实验显示二者效果相当，但 IDM 支持 only-NT 训练（LAPA 抽出的是 latent，policy 端需要额外 decode），所以 IDM 成 default。

### 2.4 Policy Training

Neural trajectory（synthetic video + pseudo-action）→ visuomotor policy。Co-train 时和 real teleop 1:1 sample。

> ❓ 论文提到 "neural trajectories have 0's as state"——即 NT 没有 proprio state，只有视觉 + 语言。GR00T N1 因为 action / decoder 参数与 backbone 解耦，gain 比 DP 和 π0 大。这暗示 NT-friendly 的架构应该刻意把 "state-conditioned" 与 "vision-conditioned" 路径分开。

---

## 3. 实验结果

### 3.1 数据增强：log-linear scaling

**Figure 4. RoboCasa 上 neural trajectory 数量与 policy 性能的 scaling，跨 low/mid/high 三种 ground-truth 数据规模。**

![](https://arxiv.org/html/2505.12705v2/x5.png)

关键观察：
- IDM 与 LAPA 两条曲线几乎重合
- NT 与 GT trajectory 数量呈 log-linear 关系——这是论文最 sellable 的 scaling claim
- 仅用 NT (240k) 训 GR00T N1 → 24 任务平均 **20.55%**，而 GT 30 traj baseline 17.44%。NT 已逼近 small-scale GT
- 加 NT 后 300 GT 从 49.59% → **57.61%**

### 3.2 Real-world：跨三种本体一致提升

**Figure 5. 9 个真实任务（GR1 humanoid + Franka + SO-100）在 low-data + neural trajectory co-training 下的成功率。**

![](https://arxiv.org/html/2505.12705v2/x6.png)

- GR1 humanoid 4 任务: 37% → **46.4%**
- Franka 3 任务: 23% → **37%**
- SO-100 2 任务: 21% → **45.5%**
- 每任务仅 10–13 真实 trajectory + 100–300 NT
- 任务覆盖 dexterous 范畴：folding towel、wiping liquid、hammer use、scooping M&Ms——全部是 simulation-infeasible 的

### 3.3 Generalization：zero-to-one 改善

GR00T N1 baseline 仅在 2,884 GR1 pick-and-place trajectory 上训。然后 DreamGen 生成 14 个新行为（pour, open/close articulated, tool use 等）+ 13 个新环境的 NT，每行为 50 NT。

| Setting | Baseline | + DreamGen NT |
|---|---|---|
| 新行为（seen env） | 11.2% | **43.2%** |
| 新环境（含未见 behavior）| 0% | **28.5%** |

这是 **真·zero-to-one**——baseline 完全做不到。

> ❓ 11.2% 的 baseline 是因为评分给了 "picking up object" 的部分分数（如 "pour water" pick 起 bottle 即给 0.5）。如果按严格 0/1 success，提升幅度会更夸张。

---

## 4. DreamGen Bench

**目的**：把 "video model 好不好" 量化成 "robot policy 会不会好"，让 video 研究者无需机器人即可贡献 robotics。

两个指标：
- **Instruction Following (IF)**: 用 Qwen-VL-2.5 / GPT4o / 人工各打 0/1，问 "video 是否完成 instruction"。模型评分与人工 Pearson > 90%
- **Physics Alignment (PA)**: 用 VideoCon-Physics（专门训练的 physics 评分 VLM）+ Qwen-VL-2.5 通用打分平均

**Table 2 (节选). DreamGen Bench 在 RoboCasa + GR1 三种泛化（Object / Behavior / Env）下的结果。**

| Model | RoboCasa Hu/PA | GR1-Object Hu/PA | GR1-Behavior Hu/PA | GR1-Env Hu/PA |
|---|---|---|---|---|
| WAN2.1-zero | 0.0 / 0.0 | 0.0 / 2.0 | 0.0 / 2.1 | 0.0 / 6.7 |
| Cosmos-zero | 22.9 / 22.9 | 32.0 / 32.0 | 31.9 / 31.9 | 24.1 / 24.1 |
| Hunyuan-sft | 81.3 / 44.8 | 52.0 / 39.0 | 14.9 / 12.8 | 43.2 / 35.4 |
| CogVideoX-sft | 79.2 / 44.8 | 72.0 / 55.0 | 21.3 / 24.7 | 61.1 / 51.3 |
| WAN2.1-sft | **91.7** / 55.3 | 80.0 / 69.0 | **74.5** / **64.9** | 67.4 / **66.5** |
| Cosmos-sft | **93.8** / **61.5** | **84.0** / **73.0** | 68.1 / 64.9 | 53.3 / 59.4 |

**Figure 6. DreamGen Bench 分数与 RoboCasa downstream policy 成功率的正相关。**

![](https://arxiv.org/html/2505.12705v2/x9.png)

要点：
- Zero-shot 几乎全军覆没（Cosmos 例外，得益于其 physics-aware pretraining）
- Fine-tuning 后 Cosmos / WAN2.1 进入 60–95 区间，且 bench 分数与 downstream RoboCasa 成功率呈正相关——这是论文给 video community 的 "请优化这个 metric" 信号

---

## 5. 局限

- **Compute 巨大**：240k RoboCasa NT 用了 **1500 张 L40 GPU × 54 小时**。即便 NVIDIA 也算重投入
- **Initial frame 仍需手动**：环境泛化要人去新场景拍图。论文承认这是 operational overhead，需要自动化的 frame proposer
- **任务复杂度低**：作者承认目前都是相对简单的 manipulation，long-horizon dexterous 还没碰
- **没和 human-video 路线 head-to-head**：限于篇幅与 framing，论文没把 DreamGen 与 LAPA-from-human-video / RT-Trajectory 等做直接对比
- **Bench 评估器会幻觉**：lightweight VLM 在 physics 评分上的可靠性是 open problem

---

## 关联工作

### 基于
- [[2501-Cosmos|Cosmos]]: NVIDIA 自家 video world foundation model；DreamGen Bench 上 Cosmos-sft 是 RoboCasa IF 最强（93.8）
- WAN 2.1: DreamGen 默认 video backbone；GR1-Behavior / Env 上最强
- [[2503-GR00TN1|GR00T N1]]: 默认 visuomotor policy backbone；NT-friendly 的解耦 action/decoder 架构使其受益最大
- [[2410-Pi0|π0]]: 实验中作为 policy 之一对比
- LoRA: video model fine-tune 的 default adaptation 方式

### 对比 / 替代路线
- [[2402-Genie|Genie]]: video model + latent action，但聚焦 game world、生成可交互环境，DreamGen 直接用做 data factory
- [[2406-IRASim|IRASim]]: 用 video diffusion 做 robot trajectory rollout 但偏 evaluation
- UniSim / UniPi (concurrent works): video model 作 real-time planner（DreamGen 明确反向选择）
- LAPA: 论文自己的前作，用 latent action 从无 action 视频学策略；DreamGen 把它作为 IDM 的替代之一

### 方法相关
- [[2405-Vista|Vista]] / [[2405-DIAMOND|DIAMOND]] / [[2408-GameNGen|GameNGen]] / [[2604-HYWorld2|HunyuanWorld 2]]: video world model 家族，DreamGen Bench 可用作它们的 robotics-utility 评测
- [[2411-WorldModelSurvey|World Model Survey]]: 综述背景
- VideoCon-Physics: 用作 physics alignment 的评分模型
- Qwen-VL-2.5 / GPT4o: instruction following 的 evaluator

---

## 论文点评

### Strengths

1. **Framing 极其干净**——"video model as offline data engine, not online planner"，一句话把自己与一大堆 concurrent 路线切开。这种 conceptual move 是高 leverage 的
2. **Scaling law 形式的 evidence**——log-linear NT-vs-success 曲线（Figure 4）是非常说服力的"这条路 scale"信号，胜过任何单点 SOTA 数字
3. **Zero-to-one 实验设计**：baseline 0% → 43.2% / 28.5% 是非常有说服力的 generalization claim，远胜过 marginal 提升
4. **配套 benchmark 自带 incentive**：DreamGen Bench 让 video model 研究有了"对 robotics 有用"的可量化目标，潜在影响超出本文实验
5. **实验覆盖广**：3 种 robot 本体（GR1 humanoid + Franka + SO-100）+ sim（RoboCasa）+ 3 种 policy 架构（DP / π0 / GR00T N1），生态级验证

### Weaknesses

1. **Compute barrier**：1500 L40-GPU-hours 的合成成本意味着这套 pipeline 在学术界基本不可复现。论文应给出 "最低可行 compute 配置" 的 ablation
2. **没和 human-video 路线对比**：LAPA / Vid2Robot / RT-Trajectory 也声称从无 action 视频学 policy。DreamGen 在 limitation 里轻轻带过 "complementary"，但读者会想知道 head-to-head
3. **Pseudo-action 质量没单独剖析**：IDM 的精度在不同 embodiment 下到底如何？哪些任务的 bottleneck 是 video 质量，哪些是 IDM 误差？仅靠 "replay in sim" 的定性观察不够
4. **行为泛化的归因模糊**：22 个 new behavior 是 video model 真的"理解"了 verb，还是 internet pretraining 里恰好见过这些动作的人类视频？没有 analysis 区分二者
5. **Bench 评分器自身的可靠性循环依赖**：用 VLM 评 physics → 训出 policy → 物理实验。如果 VLM 评分有 bias，整条 pipeline 会在那个 bias 方向上 overfit

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training pipeline 已开源（NVIDIA/GR00T-Dreams），但当前以 Cosmos-Predict2 为 default video backbone（非论文 default 的 WAN2.1）
- **模型权重**: GR00T N1 公开；Cosmos / WAN2.1 base weights 公开；fine-tuned video world model 的具体 checkpoint 未在 README 列出
- **训练细节**: 主要超参（LR=1e-4, LoRA rank=4, alpha=4, epoch 数, batch size）在 Appendix D 给出；LAPA 训练细节在 Appendix A 给出；但 240k NT 的具体 prompt list 和 initial frame 未完全披露
- **数据集**: RoboCasa / DROID / OXE 系列均开源；GR1 自采 teleop 数据未公开

#### Claim 可验证性
- ✅ **Log-linear scaling on RoboCasa**：Figure 4 的曲线 + Table 4 详细 24 任务 breakdown 给出，可在 GR00T-Dreams repo 复现
- ✅ **Real-world 9 任务一致提升**：Figure 5 给出 baseline 与 +NT 的具体数字
- ✅ **DreamGen Bench 与 downstream 正相关**：Figure 6 + Table 2 + 论文官网视频均可独立验证
- ⚠️ **"22 new behaviors"**：43.2% 的平均成功率包含 partial credit（如 "pick up bottle" 给 0.5）。严格 task-completion 成功率未单独列出
- ⚠️ **"a humanoid robot can perform 22 new behaviors using only single-task teleop data"**：技术上准确，但 video model 本身 LoRA fine-tune 仍需 2,884 条 GR1 pick-and-place trajectory，"single-task" 应理解为单一动词，不是 single-trajectory
- ⚠️ **Cross-embodiment generalization 强弱**：Franka 提升幅度 (23→37) 比 SO-100 (21→45.5) 小，但每 embodiment 的 video fine-tune 配置不同，无法干净归因
- ❌ 暂无明显 marketing-only 修辞

### Notes

- **核心 insight**：作者把"video model 好不好"和"policy 实时性"两个长期纠缠的目标完全解耦，让 SOTA video model 可以无负担地参与机器人 stack。这种 "decouple compute regimes" 的 framing 在其他 ML 子领域（如 Dreamer 系列把 world model 用在 imagination rollout）也反复出现，是值得迁移的 mental tool
- **对我（Dr. Li）的影响**：这篇是 VLA + World Model 交叉的 must-cite。它定义了一个新的 sub-paradigm（"video-as-data-engine"），而非仅 incremental SOTA。后续我自己做 VLA generalization 工作必须回答 "为什么不直接用 DreamGen-style synthesis"
- **Open question 1**：NT 没有 proprio state，但真实 policy 推理时有。GR00T N1 因架构受益，DP/π0 受影响——是否意味着未来 VLA 应该刻意设计 "state-optional" path？
- **Open question 2**：DreamGen Bench 的 IF + PA 真的 capture 了 downstream-relevant 的 video quality 维度吗？还是只是 happen-to-correlate？需要更系统的 ablation——例如在 PA 高但 contact-physics 错的样本上 policy 是否真能学
- **Action item**：考虑把 DreamGen 的 "behavior generalization via verb prompt" 和 [[2604-GEN1|GEN-1]] / [[2604-HYWorld2|HunyuanWorld 2]] 的 instruction-following video 结合，可能是 cross-embodiment manipulation 的下一步

### Rating

**分数**：3 - Foundation
**理由**：不是 2-Frontier 因为它不仅是一个 SOTA 数据点——它把 "video world model as offline data engine" 明确定义为一个与 UniSim/UniPi 路线对立的 sub-paradigm（见 Strengths 1），配套 DreamGen Bench 给 video community 提供了 "对 robotics 有用" 的可量化目标（见 Strengths 4），且 NT-vs-success 的 log-linear scaling 曲线是方向层面的奠基证据（见 Strengths 2）。不是 1-Archived 因为出自 NVIDIA GEAR + UW + KAIST 主力团队，配套 GR00T-Dreams repo 开源，已被 VLA × World Model 交叉方向作为绕不开的参照（见 Notes "对我的影响"）。
