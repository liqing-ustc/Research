---
title: "EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence"
authors: [Ding Zou, Feifan Wang, Mengyu Ge, Siyuan Fan, Zongbing Zhang, Wei Chen, Lingfeng Wang, Zhongyou Hu, Wenrui Yan, Zhengwei Gao, Hao Wang, Weizhao Jin, Yu Zhang, Hainan Zhao, Mingliang Zhang, Xianxian Xi, Yaru Zhang, Wenyuan Li, Zhengguang Gao, Yurui Zhu]
institutes: [ZTE NebulaBrain Team]
date_publish: 2025-10
venue: arXiv
tags: [task-planning, VLA, spatial-reasoning]
paper: https://arxiv.org/abs/2510.20578
website: https://zterobot.github.io/EmbodiedBrain.github.io
github:
rating: 1
date_added: 2026-04-16
---
## 速查卡片

> [!summary] EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence
> - **核心**: 基于 Qwen2.5-VL 构建面向 embodied task planning 的 VLM，通过 agent-aligned 数据结构 + Step-GRPO 强化学习实现长序列规划 SOTA
> - **方法**: 两阶段训练——rejection sampling SFT cold-start + Step-Augmented GRPO（在 planning sequence 中注入前置步骤作为 Guided Precursors）；设计 Generative Reward Model (GRM) 评估规划合理性
> - **结果**: 7B/32B 两个尺度在 spatial perception（平均 +33.8%/+11.7% vs baseline）和 task planning（VLM-PlanSim-99: 31.31%/46.46%）全面 SOTA
> - **Sources**: [paper](https://arxiv.org/abs/2510.20578) | [website](https://zterobot.github.io/EmbodiedBrain.github.io)

---
## Summary

EmbodiedBrain 是 ZTE 提出的 embodied vision-language foundation model（7B/32B），通过 agent-aligned 数据设计和 Step-GRPO 强化学习方法，大幅提升长序列任务规划能力。

**Key Takeaways:**
1. **Agent-Aligned Data Structure**: 设计 `<response>` + `<plans>` + `<actions>` 三层结构化输出格式，将高层语言回复、步骤级规划（[Navigate]/[Manipulate] 标签）和可执行动作元组解耦，直接对接 agent API
2. **Step-GRPO**: 在 GRPO 强化学习中，随机注入 planning sequence 的前置步骤作为 Guided Precursors，降低长序列规划难度，稳定训练收敛，优于标准 GRPO 和 DAPO
3. **VLM-PlanSim-99**: 提出基于 AI2-THOR 的 end-to-end 仿真 benchmark（99 个手工标注的家庭任务），验证 plan 的可执行性而非仅文本相似度

---
## Architecture

EmbodiedBrain 基于 Qwen2.5-VL 架构，包含三个核心组件：(1) 原生分辨率 ViT + 窗口注意力作为视觉编码器，(2) MLP-based vision-language merger 将视觉特征投影到 LLM embedding 空间，(3) Decoder-only LLM（Qwen2.5 初始化）。模型接收多视角图像、视频帧和自然语言指令作为统一多模态 token 序列，输出结构化三部分：自然语言回复、高层规划、可执行动作序列。

**Figure 1. The Architecture of EmbodiedBrain**
![](https://arxiv.org/html/2510.20578v1/x1.png)

---
## Training Data

数据框架覆盖四类能力：通用多模态、空间推理、任务规划、视频理解。SFT 和 RL 阶段使用不同但互补的数据集。

**Figure 2. Overview of training data**
![](https://arxiv.org/html/2510.20578v1/Figs/Sec3_datasets.png)

### Data Format Design

设计了面向规划的结构化数据格式，包含三个 XML 字段：
- `<response>`: 机器人执行任务时的自然语言反馈
- `<plans>`: 将高层指令分解为可解释的规划步骤，步骤类型严格限定为 [Navigate]（下肢移动）和 [Manipulate]（上肢操作），实现运动子系统解耦
- `<actions>`: 与规划对齐的可执行动作元组（如 `['Put', 'Bread', 'Basket']`），可直接映射到 agent API

### SFT Data

SFT 数据分为四类：
- **General MLLM data**: tulu-3-sft（10K）、UltraIF-sft（20K）、MM-IFInstruct（22K）等，强化指令跟随能力
- **Spatial data**: EmbSpatial-SFT（50K，经 rejection sampling + GPT-4o 验证）、pixmo-points（60K），覆盖 target object querying 和 inter-object relationship
- **Planning data**: ALFRED 数据集（25,743 条），从 PDDL 文件解析子任务序列，配合 AI2-THOR 环境采集全景图像和 bounding box
- **Video Understanding data**: Ego4D + Epic-Kitchens + EgoPlan-IT，生成回顾理解和前瞻规划两类 QA

**Figure 3. Overall Data Distribution in the SFT Stage**
![](https://arxiv.org/html/2510.20578v1/Figs/Sec3_datasets_sft_total_distribute_1020_w.png)

### RL Data

RL 阶段数据分为两个流：
- **Spatial Data**: EmbSpatial 子集（25K，多阶段验证）+ Orsta-Data-47k 子集（object counting 1,725 + object detection 8,000 + visual grounding 4,870）
- **Planning Data**: SelfReVision（~26K，用 Qwen3-32B 转换为结构化动作序列）+ ALFRED（~22K，转换为视觉 grounded 的可执行规划）

---
## Training Strategy

两阶段训练管线：Stage 1 SFT 建立基础能力，Stage 2 Step-GRPO 强化长序列规划。

### Stage1: Multi-modal Rejection Sampling based SFT

**Rejection Sampling 管线**：两阶段过滤——(1) 用 Qwen2.5-VL-7B 生成 k=8 个候选回答，由 Qwen3-30B-A3B 判断，全部错误的样本淘汰；(2) 用 Qwen2.5-VL-72B 作为 oracle 验证 ground truth 标签，与 GT 不符的样本过滤。样本按正确率分为 A/B/C/D 四类，D 类淘汰。

**Cold-start SFT 数据配比**：探索了 5 种数据混合配置。最终选择 General:Spatial:Planning:Video = 52:130:51.5:20 的配比，在空间推理（70.27%）和任务规划（64.64%）之间取得最佳平衡。

**Table 1. Cold-Start SFT Performance Across Data Mixing Configurations**

| Data Amount (K) | BLK | CVB | EMb | ERQ | Spatial Avg | EP1 | EP2 | EgT | Planning Avg |
|---|---|---|---|---|---|---|---|---|---|
| 30:50:70:- | 83.22 | 78.79 | 71.95 | 41.85 | 68.95 | 42.89 | 43.73 | 48.92 | 61.05 |
| 30:50:45.5:- | 86.01 | 78.77 | 72.61 | 41.60 | 69.75 | 42.95 | 42.51 | 49.83 | 61.69 |
| 30:50:51.5:- | 85.31 | 78.18 | 72.39 | 40.85 | 69.18 | 43.10 | 44.11 | 51.71 | 62.31 |
| 52:130:51.5:- | 89.51 | 79.53 | 74.64 | 39.80 | 70.87 | 42.14 | 42.21 | 51.90 | 60.87 |
| 52:130:51.5:20 | 87.41 | 80.37 | 73.43 | 39.85 | 70.27 | 46.95 | 47.91 | 53.00 | 64.64 |

**Insights**: 加入 Video Understanding 数据（20K）对空间推理略有下降但显著提升任务规划（+3.77%），说明视频理解能力对 planning 有迁移价值。

### Stage2: Multi-task post-training with Step-GRPO

#### Multimodal Rejection Sampling Strategy

训练前通过视觉遮挡实验评估样本难度：对图像施加递增的像素遮挡比例 $\Lambda=\{0.0, 0.1, ..., 0.9\}$，监测模型预测正确率的变化。容易样本（仅靠文本即可回答）被过滤，保留对视觉信息敏感的中高难度样本。

**Equation 1. Masking correctness indicator**

$$
\delta_{\lambda_{i}}=1[\mathcal{C}(A_{\lambda_{i}},A_{\text{gt}})]
$$

**Equation 2. Robust accuracy estimate**

$$
P_{c}(\lambda_{i})=\frac{1}{K}\sum_{k=1}^{K}\delta_{\lambda_{i}}^{(k)}
$$

**Equation 3. Failure threshold**

$$
\lambda_{s}^{*}=\min\{\lambda_{i}\in\Lambda\mid P_{c}(\lambda_{i})<\tau\}
$$

**含义**：$\lambda_s^*$ 越小说明样本越依赖视觉信息（越难），$\tau=0.1$，每个遮挡级别重复 $K=10$ 次独立采样。

#### Step-Augmented GRPO

**Figure 4. Detailed process of Step-GRPO**
![](https://arxiv.org/html/2510.20578v1/Figs/step-grpo.png)

Step-GRPO 的核心思想：在 planning 场景下，随机提供不同长度的前置步骤作为 hints（Guided Precursors），将长序列问题分解为更短的子问题，稳定训练并改善 reward 收敛。

RL 阶段使用四类任务特定 reward：
1. **Instruction Following**: 规则匹配 + correctness reward
2. **Visual Perception**: 自适应多任务 reward（grounding/detection 用 IoU，counting 用数值精确匹配）
3. **Spatial Perception**: 双分支评估（多选题格式匹配 + 描述题语义理解，含同义词/反义词逻辑判断）
4. **Task Planning**: 规则 reward（格式合规性，0-1）+ GRM reward（Qwen3-30B-A3B 评估规划合理性，0-1）。两个 reward 的权重调节是关键挑战——规则权重过高会限制动作空间导致功能错误的"安全"计划

异步 GRM 推理加速：将 Reward Model 推理解耦为独立多线程，端到端 RL 训练时间加速约 20%，无精度损失。

---
## Evaluation

建立三层评估体系：General（5 benchmarks）、Spatial Perception（4 benchmarks）、Task Planning（5 benchmarks，含自建 benchmark）。

VLM-PlanSim-99 是论文核心贡献之一：99 个手工标注的家庭任务实例，每个任务在 AI2-THOR 中验证可执行性。评估管线分三阶段：(A) VLM 推理生成原始计划 → (B) Unified Parsing（4 层 object resolution：LLM parsing、static mapping、context caching、smart translation）→ (C) AI2-THOR 仿真执行并验证。

**Figure 5. VLM-PlanSim-99 evaluation pipeline**
![](https://arxiv.org/html/2510.20578v1/Figs/vlm_pipeline_flow_simplified.png)

**Video 1. Toast a Slice of Bread - VLM-PlanSim-99 demo**
<video src="https://zterobot.github.io/assets/videos/cases/case_7.mp4" controls muted playsinline width="720"></video>

**Video 2. Wash the Bowl in Sink and Heat it in Microwave**
<video src="https://zterobot.github.io/assets/videos/cases/case_8.mp4" controls muted playsinline width="720"></video>

### Conclusion of Experimental Results

**Table 2. Performance of EmbodiedBrain on 14 benchmarks**

| Benchmark | Qwen2.5-VL 7B | [[2507-RoboBrain2\|RoboBrain2.0]] 7B | EmbodiedBrain 7B | Qwen2.5-VL 32B | [[2507-RoboBrain2\|RoboBrain2.0]] 32B | EmbodiedBrain 32B |
|---|---|---|---|---|---|---|
| **General Ability** | | | | | | |
| MM-IFEval | 39.56 | 30.82 | **43.61** | 46.66 | 39.75 | **46.98** |
| MMStar | **62.27** | 59.40 | 62.17 | 64.70 | **65.80** | 65.40 |
| MMMU | 51.33 | 48.67 | **52.67** | 60.00 | **60.89** | 60.44 |
| AI2D | 82.55 | 81.83 | **82.61** | **85.37** | 85.23 | 84.39 |
| OCRBench | **785** | 757 | 783 | 740 | 732 | **741** |
| **Spatial Perception** | | | | | | |
| BLINK | 58.74 | 62.94 | **88.11** | 73.43 | 68.53 | **87.41** |
| CV-Bench | 62.03 | 62.97 | **80.69** | 75.57 | 68.27 | **83.64** |
| EmbSpatial | 51.76 | 52.12 | **75.04** | 67.39 | 62.95 | **77.03** |
| ERQA | 41.00 | **42.50** | 41.75 | 44.61 | **45.11** | 43.50 |
| Spatial Avg | 53.38 | 55.13 | **71.40** | 65.25 | 61.22 | **72.90** |
| **Task Planning** | | | | | | |
| EgoPlan-Bench | 41.30 | 36.73 | **49.10** | 51.11 | 46.83 | **54.66** |
| EgoPlan-Bench2 | 38.63 | 33.54 | **49.58** | 49.81 | 49.96 | **57.11** |
| EgoThink | 52.13 | 44.92 | **53.54** | **56.75** | 49.33 | 53.92 |
| Internal Planning (F1) | 30.0 | 68.3 | **85.8** | 28.3 | 75.9 | **90.5** |
| VLM-PlanSim-99 | 23.2 | 21.21 | **31.31** | 25.25 | 24.24 | **46.46** |

**Insights**:
- **Spatial Perception** 提升最显著：EmbodiedBrain-7B 在 BLINK 上 +88.11% vs Qwen2.5-VL 的 58.74%，说明专门的空间数据 + RL 训练极为有效
- **General Ability** 基本保持：说明 embodied-specific 训练未造成灾难性遗忘
- **VLM-PlanSim-99** 32B 模型（46.46%）近乎翻倍超过 baseline（25.25%），但绝对成功率仍不到 50%，说明 embodied task planning 仍有很大提升空间
- **ERQA** 是唯一没有显著提升的 spatial benchmark，可能因为 ERQA 侧重 end-to-end 多模态推理而非纯空间感知

---
## 论文点评

### Strengths

1. **Agent-Aligned Data Format 设计务实**：response/plans/actions 三层结构直接对接 agent 系统的实际需求，比纯文本 planning 更工程友好
2. **Step-GRPO 思路简洁有效**：通过随机注入前置步骤降低长序列 RL 难度，是一个 simple and scalable 的方法，优于直接用 GRPO/DAPO 处理长序列
3. **VLM-PlanSim-99 填补评估空白**：从"文本匹配"到"仿真执行"的评估范式转变有实际意义，99 个手工验证任务的质量高于自动生成
4. **全面开源承诺**：数据、模型权重、评估方法全部开源

### Weaknesses

1. **Step-GRPO 的 ablation 不充分**：缺少 Step-GRPO vs 标准 GRPO vs DAPO 的直接对比实验，只声称"优于"但未给出具体数字
2. **Baseline 选择偏窄**：只对比 Qwen2.5-VL 和 [[2507-RoboBrain2|RoboBrain 2.0]]，缺少与其他 embodied 模型（如 Embodied-Reasoner、Robix）的对比
3. **VLM-PlanSim-99 规模小**：仅 99 个任务实例，统计显著性存疑；且限于 AI2-THOR 家庭场景，泛化性未知
4. **Spatial perception 提升来源不清**：是数据规模（130K spatial data）还是 RL 训练的贡献？缺少 ablation 解耦
5. **Real-world 验证缺失**：所有实验均在仿真环境或 offline benchmark 上完成，未展示真实机器人部署

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源（声称将在 project page 开源，但目前无 GitHub 链接）
- **模型权重**: 声称开源 7B 和 32B 两个 checkpoint，但目前 project page 无下载链接
- **训练细节**: 超参 + 数据配比 + 训练策略描述较完整，但缺少训练步数、学习率等精确超参
- **数据集**: 声称开源所有数据，但目前无实际下载链接

#### Claim 可验证性
- ✅ General Ability 保持：OpenCompass 评估，14 个 benchmark 数据完整可查
- ✅ Spatial Perception 大幅提升：BLINK/CV-Bench/EmbSpatial 提升幅度清晰，benchmark 公开可复现
- ⚠️ "most powerful Embodied vision-language foundation model among both open-sourced and closed-sourced models"：对比范围仅含 Qwen2.5-VL 和 [[2507-RoboBrain2|RoboBrain 2.0]]，未与 GPT-4o、Gemini 等闭源模型在 embodied benchmark 上全面对比
- ⚠️ Step-GRPO "outperforming both standard GRPO and DAPO"：未给出直接对比数据，仅为声明
- ❌ "the most powerful" / "unprecedented capabilities"：营销修辞，在仅对比两个 baseline 的情况下无法支撑

---
## 关联工作

### 基于
- Qwen2.5-VL: 基座模型，EmbodiedBrain 在其架构上进行 embodied-specific 训练
- GRPO: Step-GRPO 的基础 RL 算法
- ALFRED / AI2-THOR: 规划数据来源和评估环境

### 对比
- [[2507-RoboBrain2|RoboBrain 2.0]]: 主要对比 baseline，同为 embodied VLM，使用 CPT + SFT + RL 训练
- Qwen2.5-VL: 基座 baseline，7B 和 32B 两个尺度

### 方法相关
- QuestA: Step-GRPO 的灵感来源，通过问题分解简化 RL 训练
- DAPO: RL 训练的替代方法，论文声称 Step-GRPO 优于 DAPO
- SelfReVision: RL 阶段规划数据来源
- EgoPlan / Ego4D / Epic-Kitchens: 视频理解和规划数据/评估来源

---
## Notes
