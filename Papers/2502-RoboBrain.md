---
title: "RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete"
authors: [Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xiaolong Zheng, Jiaming Liu, Zhongyuan Wang, Shanghang Zhang]
institutes: [Peking University, BAAI, Institute of Automation CAS, Institute of Information Engineering CAS, HKU, UCAS]
date_publish: 2025-02
venue: CVPR 2025
tags: [VLA, task-planning, manipulation]
paper: https://arxiv.org/abs/2502.21257
website: https://superrobobrain.github.io/
github: https://github.com/FlagOpen/RoboBrain
rating: 1
date_added: 2026-04-16
---
## 速查卡片

> [!summary] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete
> - **核心**: 基于 MLLM 构建统一 robotic brain，从抽象指令到具体执行覆盖 planning、affordance perception、trajectory prediction 三层能力
> - **方法**: LLaVA 架构 (SigLIP + Qwen2.5-7B) + 多阶段训练 + ShareRobot 数据集 + A-LoRA/T-LoRA 专项模块
> - **结果**: 在 RoboVQA、OpenEQA、ShareRobot 三个 benchmark 上达到 SOTA；affordance AP 27.1%，trajectory DFD 降低 42.9%
> - **Sources**: [paper](https://arxiv.org/abs/2502.21257) | [website](https://superrobobrain.github.io/) | [github](https://github.com/FlagOpen/RoboBrain)

---
## Summary

用 MLLM 统一建模机器人操作中的 planning、affordance perception 和 trajectory prediction，配合高质量异构数据集 ShareRobot 实现从抽象指令到具体动作的完整链路。

**Key Takeaways:**
1. **三层能力统一建模**: 将 robotic manipulation 拆解为 planning（任务分解）→ affordance perception（交互区域识别）→ trajectory prediction（末端轨迹预测），用单一 MLLM 框架覆盖
2. **ShareRobot 数据集**: 从 OXE 精选 51,403 实例，标注 task planning、object affordance、end-effector trajectory 三维信息，生成 1M+ QA pairs，覆盖 102 场景 / 12 embodiment / 107 种原子任务
3. **LoRA 专项模块**: 用 A-LoRA 和 T-LoRA (rank 64) 分别在 Stage 4 微调 affordance 和 trajectory 能力，避免全量微调的灾难性遗忘

**Teaser. RoboBrain 系统概览：三层核心能力与训练数据组成**
![](https://arxiv.org/html/2502.21257v2/x1.png)

---
## Introduction

当前 MLLM 在 robotics 场景（尤其 long-horizon manipulation）中存在三个关键能力缺失：

1. **Planning Capability**: 将复杂指令分解为可执行子任务（如 "lift teapot and pour" → approach → position → tilt）
2. **Affordance Perception**: 识别交互对象的可操作区域（如茶壶的可抓取区域）
3. **Trajectory Prediction**: 预测从起点到目标的完整操作轨迹

这三层能力构成从抽象到具体的递进：planning 是语义层面的任务分解，affordance 是空间层面的目标定位，trajectory 是运动层面的路径规划。

---
## ShareRobot Dataset

**Figure 2. ShareRobot 数据生成流程**
![](https://arxiv.org/html/2502.21257v2/x2.png)

### 数据特点

ShareRobot 的核心设计思路是为 robotic brain 提供从抽象到具体的多维标注：

- **Fine-grained**: 每个数据点包含帧级低层 planning 指令（不是高层任务描述）
- **Multi-dimensional**: 同时标注 task planning、object affordance、end-effector trajectory
- **High quality**: 严格筛选标准 — 高分辨率、准确描述、成功执行、可见 affordance、清晰轨迹；3 名标注员审核
- **Large scale**: 1,027,990 QA pairs（从 51,403 实例生成）
- **Rich diversity**: 102 场景、12 embodiment、107 种原子任务

**Figure 3. ShareRobot 数据集多样性分布**
![](https://arxiv.org/html/2502.21257v2/x3.png)

### 数据选择与标注

**选择标准**（从 OXE 筛选）：
- 图像分辨率 ≥ 128px
- 有准确文字描述
- 任务执行成功
- 视频 ≥ 30 帧
- 目标物体/末端执行器未被遮挡
- 轨迹清晰完整

**标注流程**：
- **Planning**: 每个 demo 提取 30 帧 + 高层描述 → Gemini 分解为低层 planning 指令 → 3 名标注员审核 → 10 类问题 × 5 模板 → 随机选 2 模板生成 QA
- **Affordance**: 筛选 6,522 张图，标注 affordance 区域为 bounding box $\{l^{(x)},l^{(y)},r^{(x)},r^{(y)}\}$
- **Trajectory**: 筛选 6,870 张图，标注 gripper 轨迹为 ≥3 个 $\{x,y\}$ 坐标点

---
## RoboBrain Model

**Figure 4. RoboBrain 模型流水线**
![](https://arxiv.org/html/2502.21257v2/x4.png)

### Model Architecture

RoboBrain 由三个模块组成：

1. **Foundational Model for Planning**: LLaVA 架构 — SigLIP (siglip-so400m-patch14-384, 729 visual tokens) + 2-layer MLP projector + Qwen2.5-7B-Instruct
2. **A-LoRA for Affordance Perception**: LoRA (rank 64) 注入 projector 和 LLM 的 FFN 层，输出 bounding box 格式的 affordance 区域
3. **T-LoRA for Trajectory Prediction**: LoRA (rank 64) 预测 2D 轨迹 waypoints $P_{t:N}=\{(x_i,y_i)\mid i=t,\dots,N\}$

### Training Strategy

**4 阶段多阶段训练**：

- **Phase 1 (General OV Training)**:
  - Stage 1: LCS-558K 训练 Projector（视觉-语言对齐），仅 17M 可训练参数
  - Stage 1.5: 4M 高质量图文数据全模型训练
  - Stage 2: 3.2M 单图 + 1.6M 图视频数据，增强指令跟随和高分辨率/视频理解
- **Phase 2 (Robotic Training)**:
  - Stage 3: 1.3M robotic data（RoboVQA-800K + ScanView-318K + ShareRobot-200K）+ 1.7M Phase 1 replay 数据，全模型训练
  - Stage 4: A-LoRA 和 T-LoRA 分别训练（仅 28M 参数），冻结其余参数

关键设计：robotic data 与 general data 的比例约 4:6，这是实验验证的最优比例。

---
## Experiment

### Planning Task

**Figure 5. Planning benchmark 性能对比**
![](https://arxiv.org/html/2502.21257v2/x6.png)

RoboBrain 在三个 robotic benchmark 上均超越所有 baseline（GPT-4V、Claude3、LLaVA-1.5、LLaVA-OneVision-7B、Qwen2-VL-7B、RoboMamba）。在 RoboVQA 上 BLEU-4 达到 55.05，超过第二名 18.75 分。

### Affordance Prediction

**Table 2. Affordance prediction 对比**

| Model | AP |
|---|---|
| LLaVA-NeXT-7B | 9.8% |
| Qwen2-VL-7B | 12.5% |
| RoboBrain (Ours) | 27.1% (+14.6) |

### Trajectory Prediction

**Table 3. Trajectory prediction 消融实验**

| Method | DFD | HD | RMSE |
|---|---|---|---|
| RoboBrain (Base) | 0.191 | 0.171 | 0.133 |
| + Start_Points | 0.176 | 0.157 | 0.117 |
| + Max_Points | 0.185 | 0.163 | 0.125 |
| + Spec_Token | 0.109 (42.9%↓) | 0.010 (94.2%↓) | 0.091 (31.6%↓) |

加入 start points 校正了生成轨迹与末端执行器之间的平移偏移；special tokens 强调 waypoints 和起止点，带来最大改善。

### Visualization

**Figure 6. RoboBrain 多轮交互示例：从指令到 plan → affordance → trajectory**
![](https://arxiv.org/html/2502.21257v2/x7.png)

---
## 论文点评

### Strengths

1. **问题分解清晰**: 将 robotic manipulation 的认知需求显式拆解为 planning → affordance → trajectory 三层，每层有明确的输入输出格式和评估指标
2. **数据集贡献扎实**: ShareRobot 的筛选标准严格（6 条过滤准则），标注流程有人工审核，消融实验证明了数据集的有效性（Exp A vs Exp B）
3. **训练策略系统**: 多阶段训练 + data replay 缓解灾难性遗忘 + LoRA 专项模块，每个设计有对应消融支持
4. **开源完整**: 代码、三个 checkpoint（Planning / A-LoRA / T-LoRA）、数据集全部开源

### Weaknesses

1. **2D 局限**: Affordance 用 2D bounding box、trajectory 用 2D waypoints，缺乏深度信息，对 3D 操作任务的适用性有限（后续 RoboBrain 2.5 已升级到 3D）
2. **评估与真实执行脱节**: Planning benchmark 用 BLEU/GPT-4o 评分，affordance 用 AP，trajectory 用距离度量——但缺少端到端 real-robot 执行成功率评估
3. **Affordance 训练集过小**: 仅 6,000 张训练图 + 522 张测试图，且 baseline 对比只有 LLaVA-NeXT 和 Qwen2-VL，说服力有限
4. **架构创新有限**: 整体是 LLaVA + LoRA 的标准范式，核心贡献更多在数据侧

### 可信评估

#### Artifact 可获取性
- **代码**: inference+training（完整训练脚本和推理代码已开源）
- **模型权重**: Planning checkpoint (BAAI/RoboBrain)、A-LoRA (BAAI/RoboBrain-LoRA-Affordance)、T-LoRA (BAAI/RoboBrain-LoRA-Trajectory) 均在 HuggingFace 发布
- **训练细节**: 超参 + 数据配比 + 训练步数完整（Tab. 1/Tab. 4 详细列出每个 stage 的 LR / batch size / GPU 数量等）
- **数据集**: 开源 — ShareRobot (https://github.com/FlagOpen/ShareRobot)

#### Claim 可验证性
- ✅ RoboVQA SOTA: 有完整数值对比和评估代码
- ✅ ShareRobot 数据有效性: Exp A vs Exp B 消融（有/无 ShareRobot）直接证明
- ✅ 4:6 数据比例最优: Exp C-G 系统消融
- ⚠️ "state-of-the-art across various robotic tasks": OpenEQA 上实际部分子任务不如 Qwen2-VL-7B 和 LLaVA-OV-7B
- ⚠️ Affordance AP 27.1%: 绝对值仍然较低，且 baseline 仅 2 个模型

---
## 关联工作

### 基于
- LLaVA-OneVision: 模型架构和 Phase 1 训练策略直接继承
- SigLIP: 视觉编码器
- Qwen2.5-7B-Instruct: LLM backbone
- Open X-Embodiment: ShareRobot 数据集的来源

### 对比
- RoboMamba: Planning benchmark 上的主要对比对象
- GPT-4V: 闭源模型 baseline
- Qwen2-VL-7B: 开源 MLLM baseline
- LLaVA-OneVision-7B: 同架构 baseline

### 方法相关
- LoRA: 用于 Stage 4 的参数高效微调
- RoboVQA: 提供 800K 训练数据和评估 benchmark

---
## Notes

