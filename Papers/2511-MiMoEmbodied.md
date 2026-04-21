---
title: "MiMo-Embodied: X-Embodied Foundation Model Technical Report"
authors: [Xiaomi Embodied Intelligence Team]
institutes: [Xiaomi]
date_publish: 2025-11-20
venue: arXiv 2511.16518
tags: [VLM, embodied-reasoning, cross-embodiment]
paper: https://arxiv.org/abs/2511.16518
website:
github: https://github.com/XiaomiMiMo/MiMo-Embodied
rating: 1
date_added: 2026-04-21
---

## Summary

> [!summary] MiMo-Embodied: X-Embodied Foundation Model Technical Report
> - **核心**: 一个 7B VLM 同时覆盖 Embodied AI（室内机器人）与 Autonomous Driving（户外车）两个传统上各自训练的 domain，号称"first cross-embodied foundation model"，靠四阶段训练（SFT → SFT → CoT-SFT → GRPO RL）和把两个 domain 的数据 mix 进同一个模型来获得"positive transfer"
> - **方法**: 基于 MiMo-VL 7B 初始化（ViT + MLP projector + LLM）；Stage1 在 general + embodied SFT，Stage2 加入 autonomous driving SFT，Stage3 在子集上做 CoT SFT，Stage4 用 GRPO RL（reward = exact match / IoU / 格式校验）
> - **结果**: 17 个 embodied 基准（affordance / planning / spatial）+ 12 个 driving 基准（perception / prediction / planning）上声称 SOTA，超过通用 VLM（GPT-4o / Gemini / Qwen2.5-VL / InternVL3）和专用模型（[[2502-RoboBrain|RoboBrain]]、VeBrain、RoboTron-Drive、DriveLMM-o1）
> - **Sources**: [paper](https://arxiv.org/abs/2511.16518) | [github](https://github.com/XiaomiMiMo/MiMo-Embodied)
> - **Rating**: 1 - Archived（方法无原创、核心 positive-transfer claim 缺关键消融、命名即 cross-domain mix 而非真 cross-embodiment，工程 report 性质大于方向性贡献）

**Key Takeaways:**
1. **"Cross-embodiment" 的具体含义**：不是 cross robot embodiment（不同 form factor），而是把 indoor robot 和 outdoor self-driving car 视作"两类 embodiment"放进同一模型——更像是 multi-domain SFT 的命名包装
2. **正向迁移是 claim 而非实证**：Abstract 说两 domain "mutually reinforce one another"，但正文未给出 driving-only / embodied-only / mixed 的对照消融来证明这一点
3. **训练流程标准化**：4 stages = General+Embodied SFT → +Driving SFT → CoT SFT → GRPO RL，每阶段都 train all parameters，bs=512，lr=2e-6（前三阶段）/1e-6（RL）
4. **依赖 MiMo-VL 整套底座**：ViT、projector、LLM 全部从 MiMo-VL 7B 初始化，本工作核心贡献是数据 curation + 训练 recipe，不是架构创新
5. **评测覆盖广**：17 + 12 = 29 个公开 benchmark 上对比；evaluation suite 基于 lmms-eval 开源（mivllm wrapper）

**Teaser. Performance overview across 29 benchmarks.**

![](https://arxiv.org/html/2511.16518v1/x1.png)

雷达图：MiMo-Embodied 在两套 benchmark 上的 envelope 都超过 closed-source（GPT-4o、Gemini）、open-source general（Qwen2.5-VL、InternVL3）以及 specialized embodied/driving VLMs。

---

## 1. Motivation

现有的 embodied VLM 各自专精于一个 narrow 场景：
- **Indoor robotics**：[[2502-RoboBrain|RoboBrain]] / VeBrain 等，强调 task planning + spatial understanding
- **Autonomous driving**：RoboTron-Drive / DriveLMM-o1 等，强调 environmental perception + status prediction + driving planning

作者认为这种 fragmentation 导致：
1. **Lack of unified embodied VLMs**：indoor 与 outdoor 之间空间推理能力不能跨域泛化
2. **Lack of comprehensive cross-embodiment evaluation**：没有把两个 domain 的能力放一起评

> ❓ Motivation 的"cross-embodiment"用词容易和真正意义上的 cross robot embodiment（同一 policy 跨不同 robot form factor，如 [[2503-GR00TN1|GR00T N1]]、[[2406-OpenVLA|OpenVLA]] 类语义）混淆。这里其实是 cross-domain（indoor↔outdoor），是 dataset mix 的问题，不是 morphology generalization 的问题。

---

## 2. Architecture

![](https://arxiv.org/html/2511.16518v1/x3.png)

**Figure 3.** 三件套：(1) ViT 编码视觉输入（支持 single image / multi image / video）；(2) MLP projector 把视觉 token 映射到 LLM 的 latent space；(3) LLM 做文本理解与推理。ViT、projector、LLM 全部从 MiMo-VL 7B 的对应权重初始化。

> 架构本身没有任何创新——本工作的所有差异化都在数据与训练 recipe 上。

---

## 3. Training Dataset

![](https://arxiv.org/html/2511.16518v1/x4.png)

**Figure 4.** 三大类训练数据：General Dataset（基础多模态能力）、Embodied AI Dataset（affordance / planning / spatial）、Autonomous Driving Dataset（perception / prediction / planning）。

### 3.1 General Dataset

继承 MiMo-VL 训练语料，含 visual grounding、document/chart 理解、video understanding、multimodal reasoning 四类。

### 3.2 Embodied AI Dataset

| 能力 | 数据源 |
|---|---|
| Affordance Prediction | PixMo-Points, RoboAfford, RoboRefIt |
| High-level Task Planning | [[2503-CosmosReason1\|Cosmos-Reason1]], EgoPlan-IT, RoboVQA |
| Spatial Understanding | SQA3D + 自构建 3D 数据, VLM-3R, RefSpatial, EmbSpatial-SFT |

值得注意的两点：
- **3D grounding 自构建**：基于现有数据集生成大规模 3D bbox grounding 样本，每个样本是 "RGB image + spatial query → camera coordinate 下的 3D box"，用于 monocular 3D 理解
- **CoT 推理链来自 DeepSeek-R1**：Cosmos-Reason1 的 reasoning 标注是 R1 生成的 long-chain trace，本质上是 R1 知识在 embodied domain 的蒸馏

### 3.3 Autonomous Driving Dataset

| 能力 | 数据源（节选） |
|---|---|
| Environmental Perception (general scene) | CODA-LM, DriveLM, nuScenes-QA, MAPLM, MME-RealWorld, IDKB |
| Environmental Perception (regional object) | CODA-LM, DriveLM, DriveAction, MME-RealWorld, nuScenes-QA, IDKB |
| Status Prediction (intent) | DriveLM, MME-RealWorld |
| Driving Planning (action decision) | DriveLM, MME-RealWorld, IDKB |
| Driving Planning (driving reasoning) | CODA-LM, NuInstruct, LingoQA, BDD-X, DriveLM, IDKB |

> ❓ 数据章节几乎全部是 "X 数据集做 Y 任务" 的清单式陈述。**没有给出任何数据量、配比、或采样策略的数字**——这对于一份强调 "data construction" 的 technical report 是个重要 omission。

---

## 4. Training Strategy

四阶段渐进式训练，每阶段在前一阶段权重上继续：

**Table 1. Training configuration.**

| Stage | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Dataset | General + Embodied AI | + Autonomous Driving | + CoT Data | RL Data |
| Batch Size | 512 | 512 | 512 | 32 |
| Learning Rate | 2e-6 | 2e-6 | 2e-6 | 1e-6 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Weight Decay | 0.05 | 0.05 | 0.05 | 0.0 |
| LR Schedule | Cosine | Cosine | Cosine | Cosine |
| Max Sequence Length | 32768 | 32768 | 32768 | 32768 |
| Trainable | All | All | All | All |

- **Stage 1 (Embodied SFT)**：建立 affordance / planning / spatial 基本能力
- **Stage 2 (Driving SFT)**：加入驾驶数据，重点是 multi-view 空间推理、temporal consistency、safety-critical 感知
- **Stage 3 (CoT SFT)**：在前面数据子集上加 explicit reasoning chain，把"先分析再决策"的格式教给模型
- **Stage 4 (RL with GRPO)**：用 DeepSeek-R1-style GRPO，reward 设计：
  - Multi-choice：exact answer matching
  - Spatial grounding / pointing：IoU（box）或 point-in-mask
  - 所有任务：format compliance（template check）

> ❓ Stage 顺序是 Embodied 先、Driving 后；Table 1 dataset 列写"Previous + ..."提示是 mix 而非纯增量，但配比未公开。如果纯顺序 SFT 没有 replay，会有 catastrophic forgetting 的风险。

---

## 5. Evaluation

### 5.1 Embodied AI Benchmarks (17)

| 类别 | Benchmarks |
|---|---|
| Affordance | RoboRefIt, Where2Place, VABench-Point, Part-Afford, RoboAfford-Eval |
| Planning | EgoPlan2, RoboVQA, Cosmos |
| Spatial Understanding | CV-Bench, ERQA, EmbSpatial, SAT, RoboSpatial, RefSpatial, CRPE, MetaVQA, VSI-Bench |

声称在 affordance prediction 上对 VABench-Point / Part-Afford / RoboAfford-Eval 三项以"large margin"领先其他 embodied VLM。

### 5.2 Autonomous Driving Benchmarks (12)

Single-view: CODA-LM, DRAMA, MME-RealWorld, IDKB, OmniDrive, NuInstruct
Multi-view: DriveLM, MAPLM, nuScenes-QA, LingoQA, BDD-X, DriveAction

部分 reported 数字（vs. specialist / Gemini 等）：
- CODA-LM: 58.55（vs. RoboTron-Drive 58.10）
- DRAMA: 76.14（vs. specialist 68.40）
- MME-RealWorld: 60.25（vs. Gemini 67.00）
- IDKB: 43.42（vs. specialist 23.21）
- DriveLM: 57.85（vs. RoboTron-Drive 61.30，落后）
- MAPLM: 74.52（vs. RoboTron-Drive 74.34，持平）
- nuScenes-QA: 56.71（vs. specialist 53.40）
- NuInstruct: 83.58（vs. RoboTron-Drive 83.00）

> ❓ DriveLM 和 MME-RealWorld 上落后于 specialist/Gemini，但这些是 driving 领域的"招牌" benchmark。"SOTA on 12 benchmarks" 的说法值得保留——是 average win 而非 universal win。

### 5.3 Qualitative Evaluation

- **Embodied navigation**：在 global / egocentric 视图上预测 keypoints
- **Embodied manipulation**：估计 functionally-grounded 的 interaction point（即 affordance point）

### 5.4 Ablation

> 论文 Section 5.3 标题为 Ablation Study，但 defuddle 与 webfetch 都未能取到该段正文。从 outline 看应该有，但具体消融了什么（4 stages 各自贡献？driving / embodied 数据剥离？）无法核实。

---

## 关联工作

### 基于
- MiMo-VL (arXiv 2506.03569): 完全继承 ViT + projector + LLM 三件套权重；本工作可视为 MiMo-VL 在 embodied + driving domain 的 SFT/RL 续训
- [[2503-CosmosReason1|Cosmos-Reason1]]: 提供 task planning 训练数据（含 R1 生成的 reasoning trace）
- DeepSeek-R1: GRPO 算法源头；CoT 数据生成的 teacher

### 对比
- [[2502-RoboBrain|RoboBrain]] / [[2507-RoboBrain2|RoboBrain 2.0]] / [[2601-RoboBrain25|RoboBrain 2.5]]: 室内 embodied 专用 VLM 的代表线
- VeBrain (arXiv 2506.00123): 另一条 embodied VLM 路线
- RoboTron-Drive / DriveLMM-o1: 自动驾驶 VLM 的代表线
- [[2503-GeminiRobotics|Gemini Robotics]] / [[2510-GeminiRobotics15|Gemini Robotics 1.5]]: Closed-source 跨形态 robotics VLM 对照

### 方法相关
- GRPO + rule-based reward (IoU / exact match / format check): 与 [[2506-VLNR1|VLN-R1]]、[[2604-OpenSpatial|OpenSpatial]] 等 spatial RL 工作的 reward design 思路一致
- 多阶段 SFT → CoT → RL pipeline: 与 [[2506-VLNR1|VLN-R1]]、近期 reasoning VLM 的 standard recipe 同构

---

## 论文点评

### Strengths

1. **完整的 evaluation suite 开源**：基于 lmms-eval 的 mivllm wrapper 与 29 个 benchmark 的配置文件全部开源，是"reproducible eval"的实质性贡献
2. **覆盖度真的广**：17 + 12 = 29 个 benchmark 在一份 report 里同时跑通本身就是工程量，对后续工作的对比有 reference 价值
3. **Recipe 透明**：四阶段训练的超参表完整披露（bs / lr / wd / schedule），是少数把 RL stage 的 lr 和 batch size 都报全的 technical report

### Weaknesses

1. **核心 claim "positive transfer between embodied & driving" 缺证据**：abstract 与 introduction 都强调两个 domain 互相增强，但正文（至少 webfetch 可见的部分）没有给出 (a) embodied-only 训练的模型在 driving benchmark 的表现，(b) driving-only 训练的模型在 embodied benchmark 的表现。这个 ablation 是 paper 立论的关键，缺失非常显眼
2. **"Cross-embodied" 命名误导**：cross-embodiment 在文献里通常指跨机器人形态（arm vs. humanoid vs. mobile base 等，参见 [[2503-GR00TN1|GR00T N1]]、[[2406-OpenVLA|OpenVLA]]），这里实际是 cross-domain。命名 inflation
3. **数据细节不透明**：每个 sub-dataset 的样本数、四阶段的 mixing ratio、RL 样本量全都未披露
4. **VSI-Bench 等高难空间基准的具体数字**未在易获取部分给出
5. **没有真机部署**：qualitative 部分只是输出 keypoint / interaction point 的 visualization，没有接到 policy / 控制器后的成功率数字。"embodied" claim 停留在 perception+reasoning 层
6. **RL stage 的边际收益未单独报告**：GRPO 的 reward 设计写得很清楚，但没说 +RL 比 +CoT-SFT 的 delta 是多少

### 可信评估

#### Artifact 可获取性
- **代码**: inference + evaluation only（明确说明 "does **not** contain model training code"）
- **模型权重**: `XiaomiMiMo/MiMo-Embodied-7B` 在 HuggingFace 已发布
- **训练细节**: 仅高层超参（Table 1），数据配比与样本量未披露
- **数据集**: 全部为已开源的 benchmark 数据组合（PixMo-Points / RoboAfford / Cosmos-Reason1 / DriveLM / nuScenes-QA / ... 共 ~20+ 数据源），但作者新增的 "self-curated 3D grounding data" 未公开

#### Claim 可验证性
- ✅ "29 个 benchmark 上的 reported 数字"：lmms-eval 框架 + 开源 checkpoint 可独立复现
- ⚠️ "achieves SOTA on 17 embodied + 12 driving benchmarks"：average / per-task 都赢? 表格里 DriveLM、MME-RealWorld driving 部分有落后 specialist 的项
- ⚠️ "positive transfer between embodied and driving"：缺关键消融，目前只是 plausible 假说而非 evidence-based claim
- ⚠️ "first cross-embodied foundation model"：取决于"cross-embodied"的定义；按论文自己的 indoor↔outdoor 定义，可能算 first，但同一个 VLM 同时跑两个 domain 在工程上并不新颖
- ❌ "Sets a new standard for integrating diverse competencies, paving the way for more intelligent and adaptable systems"——marketing 修辞，不是技术 claim

### Notes

- **这篇 paper 的真正信号**：Xiaomi 在 push 一个统一 mobile/auto/robotics VLM 底座的 narrative。MiMo-VL → MiMo-Embodied 这条线说明他们在尝试 vertical 整合：一个 7B VLM 同时给手机端 agent、家居 robot、车端 cockpit 用
- **对我研究兴趣的相关性**：方法上没有原创性（都是 standard SFT+RL recipe），但**作为一个 multi-domain mixing 的 data point** 有参考价值——如果未来要论证 "single-VLM for embodied + driving" 是否 viable，这是一个可引用的开源对照
- **缺失的 ablation 是真问题**：positive transfer 是这篇 paper 的核心 selling point，没有 ablation 就没法知道是 transfer 真的发生了还是只是 capacity 大了/数据多了带来的提升。**如果未来要做类似 multi-domain VLM 训练，positive transfer 这一 claim 必须做 leave-one-domain-out 消融**
- **"Cross-embodied" 命名值得警惕**：未来读到带 "cross-embodiment" 字样的论文要先看其定义，避免被术语 inflation 误导

### Rating

**分数**：1 - Archived
**理由**：方法层面无原创（架构全继承 MiMo-VL、训练是标准的 SFT→CoT-SFT→GRPO pipeline，与 [[2506-VLNR1|VLN-R1]] 等同构），核心 selling point "positive transfer across embodied & driving" 缺 leave-one-domain-out 消融，连 Ablation 章节都抓不到正文——构不成方法级贡献。Benchmark 维度也不是新评测标准，而是 lmms-eval 上 17+12 个已有 benchmark 的打包；值得作为"multi-domain VLM mixing 的开源对照"引用，但不是 Frontier 档方向参考，不会主动追更相关后续。相较 [[2502-RoboBrain|RoboBrain]] 线 / [[2503-GeminiRobotics|Gemini Robotics]] 这种定义方向的工作，本文更像是 Xiaomi 把 MiMo-VL 往 embodied+driving 方向延展的 technical report，归为 Archived。
