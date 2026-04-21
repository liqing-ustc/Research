---
title: "OmniEVA: Embodied Versatile PlAnner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning"
authors: [Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan]
institutes: [Huawei Noah's Ark Lab]
date_publish: 2025-09
venue:
tags: [embodied-reasoning, spatial-reasoning, task-planning]
paper: https://arxiv.org/abs/2509.09332
website: https://omnieva.github.io/
github:
rating: 2
date_added: 2026-04-16
---

## Summary

> [!summary] OmniEVA: Embodied Versatile PlAnner
> - **核心**: 通过 Task-Adaptive 3D Grounding 和 Embodiment-Aware Reasoning 实现跨 2D/3D 的通用具身推理与可执行规划
> - **方法**: Gated Router 动态注入 3D positional encoding + TE-GRPO 强化学习引入物理约束奖励
> - **结果**: 8 个 embodied reasoning benchmark 中 7 个 SOTA，mobile manipulation 实机部署成功率显著提升
> - **Sources**: [paper](https://arxiv.org/abs/2509.09332) | [website](https://omnieva.github.io/)
> - **Rating**: 2 - Frontier（在 embodied reasoning + planning 方向给出清晰的 TAGR + TE-GRPO 范式，benchmark 覆盖广且 8B 规模超过 32B baseline，具备作为参考方法的地位；但代码/数据未开源，自定义 benchmark 无法独立验证，尚未成为社区标准）

**Key Takeaways:**
1. **Task-Adaptive Gated Router (TAGR)**：用 Gumbel-Softmax 硬门控动态决定是否注入 3D positional encoding，避免了 hard-coded 3D 融合在不需要几何推理时引入噪声的问题
2. **TE-GRPO**：在 GRPO 基础上增加 task reward 和 embodiment reward，通过 progressive curriculum 逐步强化物理可行性约束，使规划输出不仅语义正确还可在真实机器人上执行
3. **跨 embodiment 泛化**：通过在 prompt 中指定机械臂长度等物理参数，模型在未见过的机器人构型上也能保持 80.5% 成功率

**Teaser. 2D/3D Embodied Reasoning Benchmark 性能对比**
![](https://arxiv.org/html/2509.09332v3/x1.png)

---
## Introduction

当前 MLLM-based 具身系统面临两个核心挑战：

1. **Geometric Adaptability Gap**：纯 2D 模型缺乏空间推理能力，而现有 3D LLM 采用 hard-coded 的 3D 注入策略，忽视任务相关性，在 3D 输入不完整或非必要时引入噪声和计算浪费
2. **Embodiment Constraint Gap**：现有方法训练在网络数据或仿真上，忽略真实机器人的物理约束（object affordances、workspace limitations、kinematic feasibility），生成的规划理论可行但实际不可执行

OmniEVA 是第一个通过 task-conditioned feature selection 动态统一 2D/3D 输入的框架。同时提出 4 个 primitive benchmark（Where2Go、Where2Grasp、Where2Approach、Where2Fit），分别对应大空间物体搜索、抓取、接近、放置四个基本具身技能。

---
## Methodology

### Overview

OmniEVA 基于预训练 MLLM（InternVL3-8B），包含三个组件：vision transformer encoder、cross-modal projector、autoregressive text decoder。输入为自然语言指令、RGB 图像/视频帧，以及可选的 depth maps 和相机参数。核心创新是 Task-Adaptive Gated Router (TAGR) 动态融合 3D 特征，以及三阶段训练策略实现 embodiment-aware planning。

**Figure 2. Model Architecture of OmniEVA**
![](https://arxiv.org/html/2509.09332v3/x2.png)

### Task-Adaptive Gated Router

TAGR 是一个动态中介模块，根据任务需求和场景复杂度选择性注入 3D positional encoding。

**Patch-Level 3D Positional Encoding**：将 depth image 通过相机参数投影到世界坐标，按 ViT patch 大小分块取均值，再应用 sinusoidal encoding 得到 $V^{p} \in \mathbb{R}^{N \times H_p \times W_p \times d_v}$。

**Dynamic 3D Injection via Gated Routing**：TAGR 基于两个条件信号做门控决策——task condition（sentence transformer 编码的指令向量 $V^T$）和 scene condition（vision encoder 输出的全局场景描述 $V_{\text{avg}}^{I}$）。拼接后通过 MLP 得到 gate logits，再用 Gumbel-Softmax 做硬门控：

**Equation 1. Gumbel-Softmax 硬门控**

$$
g = \text{GumbelSoftmax}(V^g, \tau) \in \{0, 1\}
$$

$$
V_{\text{hybrid}}^{I} = V^{I} + g \cdot V^{p} = (1 - g)V^{I} + g(V^{I} + V^{p})
$$

**符号说明**：$g=1$ 时注入 3D 特征，$g=0$ 时仅用 2D 视觉特征。等价于纯视觉 token $V^I$ 和融合 token $(V^I + V^p)$ 之间的 MoE。
**含义**：关键设计是用 hard gating 而非 soft weighting——soft gating 会扭曲 sinusoidal encoding 的幅值，导致显著性能下降。

**Equation 2. 总损失函数**

$$
\mathcal{L}_{\psi,\theta}^{\text{total}} = \mathcal{L}_{\psi,\theta}^{\text{CE}}(o^{\text{label}}, o) + \alpha \cdot \mathcal{L}_{\psi}^{\text{KL}}(V^g || \mathcal{P}_{\text{prior}})
$$

**符号说明**：CE 为交叉熵损失，KL 项将 gate 分布正则化到 Bernoulli(0.5) 先验，防止 gate 坍缩。

### Embodiment-aware Training Strategy

三阶段级联训练：

**Stage 1: TAGR Pretraining**：在 ScanNet、Matterport3D、3RScan、ArkitScenes 等 depth-aware 数据上预训练 TAGR 模块。LLM backbone 用小学习率 $5e^{-7}$，TAGR 参数用 $1e^{-4}$。训练后 TAGR 参数冻结。

**Stage 2: SFT for General Embodied Reasoning**：混合数据集（通用 embodied reasoning + 自定义导航/操作任务），ViT 冻结，LLM 学习率 $1e^{-5}$。产出 OmniEVA-Base。

**Stage 3: TE-GRPO (Task- and Embodiment-aware GRPO)**：在 OmniEVA-Base 基础上引入两类额外奖励：

**Equation 3. Task 和 Embodiment 奖励**

$$
r_i^{\text{task}}(q, o_i) = \text{EvalTask}(q, o_i) \in [0,1], \quad r_i^{\text{embod}}(q, o_i) = \text{EvalExec}(q, o_i) \in \{0, 1\}
$$

**符号说明**：$r^{\text{task}}$ 衡量语义任务满足度（如放置点是否在目标区域内），$r^{\text{embod}}$ 通过仿真验证运动学可达性和环境约束。

**Equation 4. Progressive Embodiment Curriculum**

$$
r_{i,t}^{\text{acc}}(q, o_i) = r_i^{\text{task}}(q, o_i) \cdot \Big(\lambda_t \cdot r_i^{\text{embod}}(q, o_i) + (1 - \lambda_t)\Big)
$$

**符号说明**：$\lambda_t \in [0,1]$ 随训练递增。初期 $\lambda_t \approx 0$ 允许不满足物理约束也获得正奖励，后期 $\lambda_t \to 1$ 严格要求物理可行性。
**含义**：渐进式课程学习，从语义正确逐步过渡到物理可执行，避免一开始就施加过严的约束导致训练不稳定。

**Figure 3. Training Paradigm of OmniEVA**
![](https://arxiv.org/html/2509.09332v3/x3.png)

---
## Experimental Results

### Benchmarks for Evaluation

- **2D Embodied Reasoning**：Where2Place、VSI-bench、PACO-LVIS、RoboRefit，以及 4 个自定义 primitive benchmark（Where2Go/Fit/Approach/Grasp）
- **3D Reasoning**：SQA3D、ScanQA、Scan2Cap、ScanRefer
- **Object Navigation**：HM3D、MP3D
- **End-to-End**：Large-Space Object Seeking、Mobile Manipulation（包含 Mobile Pickup、Mobile Placement Easy/Hard）

### Task-Adaptive 3D-Grounding: Validation across Multimodal Benchmarks

**Table 1. 不同 3D 融合方法对比**

| Methods | SQA3D | ScanQA | Scan2Cap | ScanRefer | Average |
|---|---|---|---|---|---|
| Cross-Attention (separate) | 55.1 | 27.5 | 43.3 | 4.5 | 32.6 |
| Cross-Attention (interleaved) | 55.8 | 27.5 | 42.0 | 3.6 | 32.2 |
| Hard-coded 3D Integration | 61.2 | 31.5 | 95.5 | 41.2 | 57.3 |
| Without 3D Integration | 61.2 | 30.7 | 75.5 | 4.3 | 42.9 |
| Dynamic 3D, Soft Gating | 60.6 | 30.7 | 85.6 | 26.9 | 51.0 |
| **Dynamic 3D, Hard Gating (Ours)** | **62.6** | **30.8** | **97.9** | **43.1** | **58.7** |

**Insights**: Hard gating 平均优于 hard-coded 3D 融合 1.4 分，soft gating 则大幅落后（51.0 vs 58.7），验证了保持 sinusoidal encoding 幅值不失真的重要性。Cross-attention 方法在 Scan2Cap 上暴跌约 50 分，因为序列长度翻倍且需从零学习跨模态交互。

**TAGR 激活分析**：shape-related 提示词激活率最高（76.9%），动作/活动类 50.9%，遮挡类 33.0%；而 counting（9.0%）和 material/texture（19.4%）激活率很低，说明 TAGR 确实学到了只在需要几何推理时启用 3D。

**Figure 4. 3D 激活率按 prompt 语义聚类分析**
![](https://arxiv.org/html/2509.09332v3/x4.png)

**Table 2. 2D Reasoning + In-house Benchmark（部分关键结果）**

| Models | Where2Place | VSI-bench | PACO-LVIS | RoboRefit |
|---|---|---|---|---|
| GPT-4o | 20.41 | 43.60 | 2.09 | 9.96 |
| Gemini-2.5-Pro | 28.60 | 48.83 | 3.14 | 17.91 |
| [[2507-RoboBrain2\|RoboBrain2.0]]-32B | 73.59 | 42.69 | 16.23 | 69.98 |
| **OmniEVA-Base (8B)** | **74.95** | **57.17** | **21.01** | **91.19** |

**Insights**: OmniEVA-Base 仅 8B 参数，在所有 4 个 2D benchmark 上全面超过 [[2507-RoboBrain2|RoboBrain2.0]]-32B（平均 +10.45），尤其 RoboRefit 提升 +21.21。

**Table 3. 3D Reasoning Benchmark**

| Models | SQA3D (EM) | ScanQA (EM) | Scan2Cap (CIDEr) | ScanRefer (w/o.a.) |
|---|---|---|---|---|
| 3DRS | 60.6 | 30.3 | 86.1 | - |
| V-3D LLM | 58.6 | 30.1 | 83.8 | - |
| **OmniEVA-Base** | **62.9** | **30.6** | **94.6** | **55.8** |

**Insights**: 3/4 个 3D benchmark 取得 SOTA。ScanRefer 上在不使用外部检测器的纯 text I/O 模式下达到 55.8（前最佳 44.4），展现强大的端到端 3D grounding 能力。

**Table 4. Object Navigation Benchmark**

| Methods | HM3D SR | HM3D SPL | MP3D SR | MP3D SPL |
|---|---|---|---|---|
| UniNavid | 73.7 | 37.1 | - | - |
| **OmniEVA-Base** | **74.2** | **42.5** | **59.1** | **26.2** |

**Insights**: SPL 提升 +5.4，说明 OmniEVA 不仅成功率更高，路径效率也显著更好。

### Embodiment-Aware Reasoning: Performance under Physical Constraints

TE-GRPO 训练的 OmniEVA-ER 相比 OmniEVA-Base：
- Where2Approach 精度提升 +28.95%，Where2Fit +34.28%
- Mobile Placement Easy 成功率提升 +43%，Hard +50%
- Task reward 和 embodiment reward 单独都有提升，但二者联合使用效果最优

**Figure 5. TE-GRPO 消融实验**
![](https://arxiv.org/html/2509.09332v3/x5.png)

**跨 Embodiment 泛化**（Table 5 关键数据）：在 75cm/88cm/110cm 臂长上训练，OmniEVA-ER 在 seen 臂长上平均 85.08%，在 unseen 臂长（72cm-105cm）上平均 80.52%，相比 OmniEVA-Base 提升约 38 个百分点。

**真实机器人部署**：在两款不同臂长（75cm 和 70cm）的轮式双臂机器人上测试，OmniEVA-ER 在 Cluttered Place 任务达到 9/10 成功率。

**Video 1. 咖啡递送实机演示**
<video src="https://omnieva.github.io/videos/make_coffee_v2.mp4" controls muted playsinline width="720"></video>

---
## 关联工作

### 基于
- InternVL3-8B: 作为 backbone MLLM，提供视觉语言基础能力
- GRPO: TE-GRPO 的 RL 优化算法基础，在此之上增加 task 和 embodiment 奖励

### 对比
- [[2507-RoboBrain2|RoboBrain2.0]]: 2D embodied reasoning 的主要对比对象，OmniEVA-Base 在所有 2D benchmark 上超过其 32B 版本
- 3DRS: 3D reasoning 的主要对比对象，采用 hard-coded 3D 注入策略
- UniNavid: Object Navigation SOTA 对比，OmniEVA 在 SPL 上提升 +5.4

### 方法相关
- Gumbel-Softmax: TAGR 模块的核心门控机制，实现可微分的离散采样
- Mixture-of-Experts (MoE): TAGR 的概念等价——在纯 2D token 和 2D+3D 融合 token 之间选择

---
## 论文点评

### Strengths

1. **问题定义清晰**：Geometric Adaptability Gap 和 Embodiment Constraint Gap 两个 gap 的划分精准，直指当前 embodied MLLM 的核心瓶颈
2. **TAGR 设计简洁有效**：hard gating via Gumbel-Softmax 的方案既简洁又有理论依据（保护 sinusoidal encoding 幅值），消融实验充分验证了 soft gating 和 cross-attention 方案的劣势
3. **TE-GRPO 的 progressive curriculum 设计合理**：$\lambda_t$ 渐进策略避免了直接施加物理约束导致的训练不稳定，同时通过仿真验证 $r^{\text{embod}}$ 保证了奖励信号的客观性
4. **评估维度全面**：8 个公开 benchmark + 4 个自定义 primitive benchmark + 3 个 composite task + 真实机器人部署，从 2D/3D/视频/导航/操作多角度验证

### Weaknesses

1. **TAGR 粒度受限**：scene-level 门控对异质环境可能过粗，论文自己也承认部分 spatial relationship 任务出现了意外的低激活率。Patch-level gating 是明显的改进方向但未实现
2. **Embodiment 参数化不够完整**：仅考虑了臂长一个物理参数，真实机器人的自由度、安装高度、末端执行器类型等约束均未建模
3. **代码和数据未开源**：声明 upon acceptance 后发布，当前无法复现。自定义 benchmark（Where2Go/Fit/Approach/Grasp）的数据集也未公开
4. **TE-GRPO 的 EvalExec 依赖仿真器**：embodiment reward 需要在仿真中验证运动学可行性，这限制了方法的通用性——换机器人平台需要重建仿真环境

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源（声明 upon acceptance 后发布）
- **模型权重**: 未公开
- **训练细节**: 超参 + 数据配比 + 训练步数较完整（Table 7 在附录中给出）
- **数据集**: 部分公开（ScanNet、Matterport3D 等公开数据集）+ 自定义 benchmark 未公开

#### Claim 可验证性
- ✅ 2D/3D benchmark SOTA：Table 2-4 给出了与多个 baseline 的详细对比，benchmark 均为公开数据集
- ✅ TAGR hard gating 优于 soft gating / cross-attention：Table 1 消融完整，对照清晰
- ⚠️ TE-GRPO 的提升幅度：Where2Approach +28.95%、Where2Fit +34.28% 等数字仅在自定义 benchmark 上测得，该 benchmark 未公开，无法独立验证
- ⚠️ 跨 embodiment 泛化能力：仅在臂长这一个维度上测试，且训练和测试的构型差异有限（72cm-110cm），难以断言对任意物理构型的泛化性
- ⚠️ 真实机器人实验：每个配置仅 10 次试验，样本量偏小

### Notes

### Rating

**分数**：2 - Frontier
**理由**：Strengths 显示该工作在 embodied reasoning + planning 方向给出了清晰的方法范式（TAGR + TE-GRPO），且在 8 个公开 benchmark 中 7 个 SOTA、8B 规模超过 [[2507-RoboBrain2|RoboBrain2.0]]-32B，具备"必须比较的 baseline"的前沿性；但 Weaknesses 指出代码/模型/自定义 benchmark 均未开源，核心 claim 中跨 embodiment 泛化与 TE-GRPO 增益仅在私有 benchmark 上验证，社区尚未广泛采纳或复现，因此达不到 Foundation 档；又明显强于 incremental/niche 的 Archived 档。
