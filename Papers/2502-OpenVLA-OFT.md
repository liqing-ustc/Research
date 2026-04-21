---
title: "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success"
authors: [Moo Jin Kim, Chelsea Finn, Percy Liang]
institutes: [Stanford University]
date_publish: 2025-02-27
venue: RSS 2025
tags: [VLA, imitation-learning, manipulation]
paper: https://arxiv.org/abs/2502.19645
website: https://openvla-oft.github.io/
github: https://github.com/moojink/openvla-oft
rating: 3
date_added: "2026-03-27"
---
## Summary

> [!summary] Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
> - **核心**: 系统比较 VLA fine-tuning 中的 action 生成策略 / action 表示 / 学习目标三个维度，得到一个简洁有效的 OFT recipe
> - **方法**: parallel decoding + action chunking + continuous actions + L1 regression（可选 FiLM 增强 language grounding）
> - **结果**: LIBERO 4 个 task suite 平均 97.1%（OpenVLA 76.5%），推理 26× 加速；ALOHA 双臂任务上超过 [[2410-Pi0|π0]] 与 RDT-1B
> - **Sources**: [paper](https://arxiv.org/abs/2502.19645) | [website](https://openvla-oft.github.io/) | [github](https://github.com/moojink/openvla-oft)
> - **Rating**: 3 - Foundation（把 VLA fine-tuning 的设计空间拆成 3 个正交维度做 controlled comparison，得出的 PD+AC+L1 recipe 已成为后续 VLA 工作的默认配方与必比 baseline）

**Key Takeaways:**
1. **Fine-tuning recipe matters more than想象**：在 [[2406-OpenVLA|OpenVLA]] 这个未见过 bimanual 数据的单臂 VLA 上，仅靠 fine-tuning 设计的优化即可超过用 bimanual 数据预训练的 [[2410-Pi0|π0]] 与 RDT-1B；说明很多 VLA paper 之间的差距可能更多来自 fine-tuning 而非架构 / 预训练
2. **Parallel decoding + action chunking 是几乎无损的免费午餐**：不仅推理 26×，平均成功率也提升 14%（绝对值），long-horizon 提升尤其明显；意味着 autoregressive token 生成对 action 的归纳偏置不重要
3. **L1 regression 在大模型上 ≈ Diffusion**：在 7B OpenVLA 上，简单 L1 回归与 50-step diffusion 在 LIBERO 上几乎打平（95.3 vs 95.4），但训练收敛更快、推理无去噪开销，颠覆了 "complex method = better" 的直觉
4. **FiLM 不是花瓶**：ALOHA 多视角场景下，去掉 FiLM 后 OpenVLA-OFT 的 language following 退化到 chance level，但同样的方法在 LIBERO 上不需要 FiLM 也工作良好——说明 language grounding 失败是 fine-tuning 数据 / 视角分布的产物，不是模型本身的能力上限

**Teaser. OpenVLA-OFT+ 在 ALOHA 双臂机器人上以 25 Hz 执行多种 dexterous manipulation 任务的 overview。**

![](https://arxiv.org/html/2502.19645v2/extracted/6394616/fig/figure_1_openvla_aloha.001.jpeg)

**Video. 作者制作的 OpenVLA-OFT 摘要视频。**

![](https://www.youtube.com/watch?v=T3Zkkr_NTSA)

---

## Problem & Motivation

现有 VLA（如 [[2406-OpenVLA|OpenVLA]]）在 fine-tuning 到新机器人平台时面临两大问题：

1. **推理速度瓶颈**：autoregressive token 生成在单臂上 3-5 Hz，bimanual 更低，远不能满足 25-50 Hz 的高频实时控制需求。
2. **精度瓶颈**：256-bin 离散化 + next-token prediction 在精细操作任务上表现不佳。

更深的问题是——已有工作里 fine-tuning 的设计选择（action 生成方式、表示、loss）从未被系统比较过，实践者面对一堆相互矛盾的 paper choice 不知如何取舍。本文的目标不是发明新组件，而是 **在 controlled setting 下把这些设计选择拆开比较**，给出一个可推荐的默认 recipe。

## Method

### 三个被研究的设计维度

**Figure 2. 三个 fine-tuning 设计维度的示意：左侧对比 autoregressive vs. parallel decoding 的 action 生成策略，右侧对比 discrete (next-token prediction) vs. continuous (L1 / diffusion) 的 action 表示与学习目标。**

![](https://arxiv.org/html/2502.19645v2/extracted/6394616/fig/ar_vs_pr--cont_vs_discr--v2.001.jpeg)

| 维度 | 选项 |
| --- | --- |
| Action generation strategy | Autoregressive vs. Parallel decoding（含 action chunking） |
| Action representation | Discrete (256-bin) vs. Continuous (MLP head) |
| Learning objective | Next-token prediction vs. L1 regression vs. Conditional denoising diffusion |

base model 固定为 OpenVLA，统一用 LoRA fine-tuning（500 demos 量级）。

### 核心组件实现

**Parallel decoding & action chunking**: 把 causal mask 换成 bidirectional，输入若干个 empty action embeddings，单次 forward pass 输出整段 action。chunk size $K$ 时一次出 $KD$ 维 action，把推理从 $D$ 次顺序 forward 压到 1 次。

**Continuous action representation**: 把 LM head 替换为 MLP action head，直接回归归一化连续 action。配合两种 loss：
- L1 regression：类似 [[2401-MobileALOHA|ACT]]，简单、单步
- Conditional denoising diffusion：类似 Diffusion Policy，50 步去噪，更 expressive 但慢

**Multi-view + proprio 输入**：dual vision encoder 抽 256 patch embeddings/view 投到 LM 空间；low-dim 状态用单独 projector 投成一个 embedding；与 language tokens 一起拼接送入 decoder。

### FiLM for Language Grounding

**Equation. FiLM 调制公式。**

$$
\text{FiLM}(\mathbf{F}|\mathbf{\gamma},\mathbf{\beta})=\mathbf{\hat{F}}=(1+\mathbf{\gamma})\odot\mathbf{F}+\mathbf{\beta}
$$

**符号说明**：$\mathbf{F}$ 视觉特征，$\mathbf{\gamma},\mathbf{\beta}$ 由任务语言 embedding 平均后投影得到的缩放 / 偏移向量。

**含义**：在 ALOHA 多视角场景中，policy 容易抓住 visual spurious correlation 而忽略 language。FiLM 把语言注入视觉特征做仿射调制，强制视觉表示依赖语言。

**关键实现细节**：不是每个 patch embedding 单独调制，而是借鉴 CNN 中 FiLM 的 spatial-agnostic 特性——同一个 $\gamma_i, \beta_i$ 应用到所有 patch 的第 $i$ 个 hidden unit。作者发现这个细节对 language grounding 至关重要，按 patch 调制几乎不工作。

带 FiLM 的版本称为 **OpenVLA-OFT+**。

## Experiments

### LIBERO Simulation

**Setup**: 4 个 task suite（Spatial / Object / Goal / Long），每个 500 demos，500 trials 评估。chunk size $K=8$，full chunk 执行后再 replan。

**Table I (节选). LIBERO 任务成功率，所有 OpenVLA 变体均启用 PD&AC，加上不同 action representation。**

| Method | Spatial | Object | Goal | Long | Avg |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy (scratch) | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA (fine-tuned) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA + PD&AC (discrete) | 91.3 | 92.7 | 90.5 | 86.5 | 90.2 |
| OpenVLA + PD&AC, Cont-Diffusion | 96.9 | 98.1 | 95.5 | 91.1 | 95.4 |
| OpenVLA-OFT (PD&AC, Cont-L1) | 96.2 | 98.3 | 96.2 | 90.7 | 95.3 |
| [[2410-Pi0\|π0]] (fine-tuned, +wrist+proprio) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| **OpenVLA-OFT (+wrist+proprio)** | **97.6** | **98.4** | **97.9** | **94.5** | **97.1** |

**核心观察**：
- PD+AC 相对 baseline 涨 14% 绝对值，且 LIBERO-Long 提升最猛（53.7→86.5）——支持 action chunking 缓解 compounding error 的论点
- Cont-L1 与 Cont-Diffusion 几乎打平（95.3 vs 95.4），但 L1 推理快 26 倍
- 加上 wrist + proprio 进一步到 97.1%，超过 π0

### Inference Efficiency

**Table II. LIBERO 上 7-DoF action 的吞吐 / 延迟（A100, 100 queries 平均）。**

| Variant | Throughput (Hz) ↑ | Latency (s) ↓ | LIBERO-Long SR (%) |
| --- | --- | --- | --- |
| OpenVLA | 4.2 | 0.2396 | 53.7 |
| + PD | 15.9 | 0.0629 | – |
| + PD&AC | 108.8 | 0.0735 | 86.5 |
| + PD&AC, Cont-L1 | 109.7 | 0.0729 | 90.7 |
| + PD&AC, Cont-Diffusion ($T_{test}=50$) | 4.2 | 1.9070 | 91.1 |
| + PD&AC, Cont-Diffusion ($T_{test}=5$) | 35.1 | 0.2279 | 90.0 |
| + PD&AC, Cont-Diffusion ($T_{test}=1$) | 109.4 | 0.0731 | 0.0 |
| + PD&AC, Cont-L1 + wrist + proprio | 71.4 | 0.1120 | 94.5 |

要点：PD 单独 4×，PD+AC 26× throughput；diffusion 即使用 DDIM 把 step 砍到 5 也只能勉强追上 L1 但精度下降；砍到 1 step 直接崩盘。

### ALOHA Real-World

ALOHA 是一个 14-DoF bimanual 平台、3 视角、25 Hz——和 OpenVLA pretraining（单臂 / 单视角 / 3-10 Hz / 相对末端位姿）差异巨大，是真正的 distribution shift 测试。

**4 个任务**：fold shorts (20 demos)、fold shirt (30 demos)、scoop X into bowl (45 demos)、put X into pot (300 demos)。chunk size $K=25$。

**Figure 4. ALOHA 任务的整体表现 score（含部分完成 partial credit）。**

![](https://arxiv.org/html/2502.19645v2/extracted/6394616/fig/aloha_task_performance_results_v3.001.jpeg)

**Figure 5. ALOHA language following 成功率（仅对 language-dependent 任务）。OpenVLA-OFT+ 最强；去 FiLM 后退化到 chance level。**

![](https://arxiv.org/html/2502.19645v2/extracted/6394616/fig/aloha_language_grounding_results.001.jpeg)

OpenVLA-OFT+ 在 task execution 和 language following 上都最强，平均超过最强 baseline π0 最多 15% 绝对值。值得强调的是 **OpenVLA 预训练里完全没有 bimanual 数据**，而 RDT-1B 用了 6K bimanual episodes、π0 用了 8K 小时。

**Figure 6. RDT-1B vs π0 的 error handling 对比：RDT-1B 倒洒果料时不修正 missed bowl placement；π0 抓青椒失败后能 retry。**

![](https://arxiv.org/html/2502.19645v2/extracted/6394616/fig/rdt_pi0_rollouts.001.jpeg)

> ❓ Figure 6 的对比让人猜测 RDT-1B 的失败更多来自 "Alternating Condition Injection" 让 proprio 主导而 visual feedback 被弱化。但本文未做控制实验直接验证，只是观察性结论。

**Table III (摘录). ALOHA 推理效率（多视角输入下）。**

| Method | Params | Throughput (Hz) |
| --- | --- | --- |
| OpenVLA (autoregressive) | 7.5B | 1.8 |
| ACT | 84M | (最高) |
| Diffusion Policy | 157M | – |
| RDT-1B | 1.2B | 84.1 |
| π0 (JAX) | 3.3B | – |
| **OpenVLA-OFT+** | **7.5B** | **77.9** |

OpenVLA-OFT+ 用 7B 参数做到接近 RDT-1B（1.2B）的吞吐，单 forward pass 是关键。

---

## 关联工作

### 基于
- [[2406-OpenVLA|OpenVLA]]: 直接的 base model，本文复用其架构与预训练权重，所有改动都在 fine-tuning 阶段
- LIBERO benchmark: 主要 simulation evaluation
- ALOHA / [[2401-MobileALOHA|Mobile ALOHA]] platform: 真实双臂硬件平台

### 对比
- [[2410-Pi0|π0]] (+ FAST 变体): 本文最强 baseline；用 flow matching + 双臂预训练；OFT 显示 fine-tuning 优化可以补上预训练数据的差距
- RDT-1B: 1.2B diffusion VLA，预训练含 bimanual；语言跟随强但 visual feedback 弱
- Diffusion Policy: from-scratch 强 baseline，作为 non-VLA 上限
- ACT: 来自 [[2401-MobileALOHA|Mobile ALOHA]] 系列，L1 + chunking 的 from-scratch 上限；本文 L1 选择思想也借鉴自此

### 方法相关
- FiLM (Perez et al. 2018): 视觉特征的 affine modulation，本文用来注入语言条件
- Action chunking: 在 ACT 与 Diffusion Policy 中已被验证；本文把它移植到 VLA 框架
- LoRA: 全部实验用 LoRA fine-tuning，与小数据规模匹配

---

## 论文点评

### Strengths

1. **典型的 "做对了无聊事" 类 paper**——不发明新模块，而是把混乱的 fine-tuning 设计空间拆成 3 个正交维度做 controlled comparison，结论可直接被实践者拿走当 default
2. **LIBERO 上 PD+AC 涨 14% 是非常强的 signal**——它没有引入新参数 / 新数据，纯粹是 inference / training 协议的改变，这种 free lunch 通常意味着原 baseline 的某个设计选择在 systematically 拉低性能
3. **Real-world 实验配置诚实**：ALOHA 显式排除了 vanilla autoregressive OpenVLA（速度根本不够），而非装作能比；用 partial-credit rubric 而非 binary success 减少噪声
4. **L1 vs Diffusion 在大模型上几乎打平的发现**值得被广泛引用——它挑战了 "Diffusion Policy / flow matching 必要" 的隐含 convention，至少在 small-scale fine-tuning 下不必要

### Weaknesses

1. **结论的边界**没说清楚：所有实验在 OpenVLA + LoRA + ≤500 demos 下做。是否能推到（a）full fine-tune（b）其他 base VLA（c）大数据 regime，全无实验佐证。作者在 Limitations 里部分承认了 (c)
2. **Multimodal action 问题被一带而过**：作者承认 L1 会 collapse 到 median mode，但只在 website 放了 video 演示，没做控制实验量化 multimodal demo 下 L1 vs Diffusion 的 gap
3. **"OFT 比 π0 更好" 的 framing 略 overclaim**：对比里的 π0 只用作者推荐 fine-tune recipe，没有用 OFT recipe fine-tune π0；所以 apple-to-apple 应该是 "用 OFT recipe fine-tune 任何 base 都更好" 而不是 "OpenVLA + OFT > π0 base"。两者混淆了 base model 与 fine-tuning recipe 的贡献
4. **FiLM 在 LIBERO 上不需要、ALOHA 上必需** 这一现象的解释停留在猜测层面（"可能是 spurious correlation"），没做 ablation 区分原因（多视角？bimanual？数据量？）

### 可信评估

#### Artifact 可获取性

- **代码**: inference + training 全开源（https://github.com/moojink/openvla-oft）
- **模型权重**: 已发布 LIBERO（4 个 task suite 各一个）和 ALOHA（4 个任务各一个）的 OpenVLA-OFT / OFT+ checkpoints，托管在 HuggingFace
- **训练细节**: 完整披露，Appendix A-D / A-E 给出超参表、batch size、训练步数；ALOHA 各任务 demo 数也明确列出
- **数据集**: LIBERO 公开；ALOHA 自采数据未发布（仅描述任务与 demo 数）

#### Claim 可验证性

- ✅ **LIBERO 97.1% 与 26× 加速**：开源 code + checkpoint + 公开 benchmark，可独立复现
- ✅ **PD+AC 在 LIBERO 涨 14%**：消融表清晰，500 trials × 4 suites 样本量足够
- ⚠️ **"OpenVLA-OFT+ 在 ALOHA 上超过 π0/RDT-1B 最多 15%"**：依赖作者自定的 partial-credit rubric 与自采 ALOHA 数据；evaluation 数 10-24 trials 偏小；结果信任但需注意非 apple-to-apple（base model 与 recipe 同时变了）
- ⚠️ **"L1 ≈ Diffusion"**：仅在 LIBERO 验证；作者也承认 multimodal demonstrations 下未测，不能外推
- ❌ **"existing VLAs can be successfully adapted to new robotic systems without extensive retraining"**：这是 marketing-style framing，"extensive" 没定义，且仅在 ALOHA 一个新平台上验证

### Notes

- **取走的设计原则**：(a) fine-tune VLA 时 default 用 PD+AC+L1；(b) 多视角 / 双臂场景额外加 FiLM；(c) chunk size 选 8-25 之间按控制频率定
- **未解的问题**：OFT 在 pretraining 阶段是否依然有效？如果是，那 [[2410-Pi0|π0]] / RDT 的 flow matching / diffusion 可能是 "解错了的问题"
- **对自己研究的启示**：很多 VLA paper 的横向比较里，fine-tuning recipe 是比 architecture 更大的混淆变量。做 method comparison 时如果不控制 recipe，结论几乎没意义
- **后续追踪**：[[2504-Pi05|π0.5]]、[[2604-Pi07|π0.7]] 是否继续走 flow matching 还是切到 L1？OFT 的结论会不会随模型规模变化？

### Rating

**分数**：3 - Foundation
**理由**：本文的贡献不在新模块而在 controlled comparison（见 Strengths 1-2）——把 VLA fine-tuning 的设计空间拆成 3 个正交维度并给出 PD+AC+L1 默认 recipe，过去一年里 ALOHA / bimanual / manipulation 方向的 VLA 工作（π0.5 等后续作品、各类 fine-tune pipeline）普遍把 OFT 当作必比 baseline 和默认 recipe 引用；"L1 ≈ Diffusion in VLA fine-tune" 与 "PD+AC 14% free lunch" 两个发现已被当作 field-level 共识流传。相较 2 - Frontier，它不是 "代表性 SOTA" 而是在方法范式上产生了持续影响——哪怕模型过气，这个 fine-tuning recipe 的地位不会被轻易替代，故评 3。

