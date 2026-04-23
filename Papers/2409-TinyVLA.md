---
title: "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation"
authors: [Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, Jian Tang]
institutes: [East China Normal University, Midea Group AI Lab, Shanghai University, Syracuse University, Beijing Innovation Center of Humanoid Robotics]
date_publish: 2024-09-19
venue: IEEE RA-L 2025
tags: [VLA, diffusion-policy, manipulation]
paper: https://arxiv.org/abs/2409.12514
website: https://tiny-vla.github.io/
github: https://github.com/liyaxuanliyaxuan/TinyVLA
rating: 2
date_added: 2026-04-22
---

## Summary

> [!summary] TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
> - **核心**: 用 <1.4B 的小 VLM（LLaVA-Pythia）+ Diffusion Policy head 构造 VLA，去掉对 OpenX 级别大规模机器人数据预训练的依赖，同时把推理延迟从 OpenVLA 的 ~290ms 降到 ~14ms（A6000）
> - **方法**: 自研三档 Llava-Pythia VLM（0.4B/0.7B/1.3B）做 backbone → LoRA 微调（5% 可训练参数）→ 特征经 adaptive pool + MLP 作为条件送入 Diffusion Policy head 直接输出连续动作，而非 autoregressive 动作 token
> - **结果**: 单臂 Franka 5 任务平均成功率 94.0%（OpenVLA 68.3%，+25.7pp），5.5× 参数更少、20× 推理更快；双臂 UR5 上 OpenVLA 全部为 0 而 TinyVLA-H 达 44.5%
> - **Sources**: [paper](https://arxiv.org/abs/2409.12514) | [website](https://tiny-vla.github.io/) | [github](https://github.com/liyaxuanliyaxuan/TinyVLA)
> - **Rating**: 2 - Frontier（"小 VLM + diffusion head" 路线的代表性早期工作，被 SmolVLA、Pi0 等继承；但自己用的 Pythia backbone 已过时、双臂结果绝对值偏弱、代码仓库 stale）

**Key Takeaways:**
1. **VLA 不一定要 OpenX 预训练**：把泛化能力从"robot data 学出来"换成"继承自 VLM 预训练"，在 5 个单臂任务和 instruction / background / lighting / distractor 等泛化维度上仍能匹配甚至超过 OpenVLA
2. **Autoregressive action token 是延迟大头**：控制 VLM 同尺寸（OpenVLA→1B）时 diffusion head 相比 AR token 仍快 10×（140ms vs 14ms）——说明解码方式比 backbone 缩放带来的速度增益更关键
3. **VLM 规模真实影响成功率**：0.4B→1.3B 可消除 instruction 误解类 failure，但需 ≥1.3B 才开始跨越"可用"门槛，SmolVLA 级的 450M 是明显不够的；3B PaliGemma 进一步改善 localization
4. **OpenX 预训练反而拖累 bimanual**：OpenX 只含单臂数据，OpenVLA 在双臂任务上 0% 成功，TinyVLA 的"无预训练 + VLM 先验"反而迁移更好——一个关于 pretraining distribution lock-in 的 cautionary tale

**Teaser. 推理延迟 vs 平均成功率**：TinyVLA-H 以 5.5× 更少参数、20× 更低延迟压过 OpenVLA。

**Figure 1. Inference latency vs. average success rate.** y 轴为 5 个真机任务平均成功率，气泡直径代表参数量，A6000 上测得。
![](https://arxiv.org/html/2409.12514v5/x1.png)

---

## Problem & Motivation

论文把 VLA 的两个 deploy-blocking 问题并置：
1. **推理慢**：7B 级 VLM + autoregressive action token 生成（每个 DoF 一次 forward）→ 单步动作 ~290ms
2. **预训练贵**：OpenVLA 需要 OpenX 970K trajectories，机器人领域的"BigSleep"路线

作者的诊断：两者**都源于 VLA 照搬 VLM 的生成范式**——既用了 VLM 级的 backbone 尺寸，又继承了 discrete token 生成。TinyVLA 的 bet 是两者都拆：backbone 缩到 <1.4B，动作生成换成 diffusion。

> ❓ 这是一个很自然的拆分，但"VLM 预训练先验能替代 robot data 先验"这个 claim 只在 5 个相对短时程的桌面任务上验证——对 long-horizon、接触密集、需要精确位姿控制的任务是否成立，本文没有答案。

## Method

**Figure 2. Model architecture.** 左：VLM 预训练 pipeline（LLaVA recipe + Pythia）。右：robot data 微调阶段，VLM 走 LoRA，diffusion head 全参训练。
![](https://arxiv.org/html/2409.12514v5/x2.png)

### 三个核心设计

**1) 自研紧凑 VLM 系列（Llava-Pythia）**
- 语言模型后端：Pythia（70M ~ 1.4B 档位）
- 训练 recipe：LLaVA 的视觉指令微调 pipeline + 数据
- 三档尺寸：TinyVLA-S (0.4B total) / -B (0.7B) / -H (1.3B total，143M trainable)

**2) LoRA 冻结主干**
对 attention 的 $Q, K, V$ 权重施加低秩分解 $W_0 + BA$，其中 $r \ll \min(d, k)$。可训练参数 ≈ 整个 transformer 的 5%。训练完成后通过 re-parameterization 把 LoRA merge 回主干，推理零开销。

**3) Diffusion Policy head 代替 AR token 生成**
- 选型理由：论文明确反对 discrete action tokenization——引用文献 [31-34] 论证 continuous/high-dim data tokenization 训练困难、需要大量数据、易退化。
- pipeline：VLM backbone 输出多模态 embedding → adaptive pool + LayerNorm → 拼接本体感知状态向量 → 3-layer MLP 生成 conditional embedding → 标准 [[2303-DiffusionPolicy|Diffusion Policy]] (DDPM) 训练
- 动作维度：7-DoF `(x, y, z, roll, pitch, yaw, gripper_width)`

> ❓ Adaptive pool 这一步把可变长度的视觉-语言 token 序列压成固定长度 condition，听起来合理但没有消融——这其实丢掉了 token-level 的空间信息，是否是 localization 泛化偏弱的根因？

## Experiments

### Setup

- **Simulation**: MetaWorld 50 tasks（easy 28 / medium 11 / hard 6 / very hard 5），每任务 50 demos
- **Real single-arm**: Franka Panda 7-DoF，双 ZED 2 stereo camera，5 任务（CloseDrawer / StackCubes / OpenBox / PlaceTennis / FlipMug），每任务 100 trajectories
- **Real bimanual**: 双 UR5 + wrist 和 top cameras（Realsense D435i），3 任务（TransferBread / PlaceTennisBag / StackCubes）
- **Baselines**: [[2303-DiffusionPolicy|Diffusion Policy]]（加 FiLM 注入 language）、Multimodal Diffusion、[[2406-OpenVLA|OpenVLA]]（为公平比较，改造为多视角输入）

### Main Results

**Table I. MetaWorld 仿真结果**（TinyVLA-H vs Diffusion Policy，平均成功率）

| Model | Easy (28) | Medium (11) | Hard (6) | Very Hard (5) | Avg. |
|:-----|:-----|:-----|:-----|:-----|:-----|
| Diffusion Policy | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA-H | 77.6 | 21.5 | 11.4 | 15.8 | **31.6** |

在 Hard tier 上 TinyVLA 成功率是 DP 的 6×，差距随任务复杂度扩大。

**Table II. Franka 单臂真机结果**（5 任务 × 20 trials × 3 checkpoints）

| Model | Pre-trained Traj. | Total Params | Trainable | Avg. |
|:-----|:-----|:-----|:-----|:-----|
| Diffusion Policy | N/A | 111M | 111M | 35.3 |
| Multimodal Diffusion | N/A | 230M | 230M | 18.0 |
| OpenVLA | 970K | 7.2B | 195M | 68.3 |
| TinyVLA-S | N/A | 422M | 101M | 23.3 |
| TinyVLA-B | N/A | 740M | 138M | 77.4 |
| **TinyVLA-H** | **N/A** | **1.3B** | **143M** | **94.0** |

TinyVLA-S 掉得很厉害（23.3）—— 0.4B VLM 不足以撑起 VLA。关键拐点在 0.4B → 0.7B。

**Table III. 双臂 UR5 结果**（3 任务，10 trials，平均成功率）

| Model | Trainable | PlaceBread | StackCubes | PlaceTennisBag |
|:-----|:-----|:-----|:-----|:-----|
| DP | 111M | 40.3 | 31.3 | 43 |
| OpenVLA | 195M | **0** | **0** | **0** |
| TinyVLA-H | 143M | 76.7 | 36.7 | 30 |

OpenVLA 在 bimanual 上彻底失败——作者归因于 OpenX 只含单臂数据，pretraining distribution 把 policy 锁死在了单臂动作空间。这是论文最有信息量的 negative result。

### Ablations

**Table IV. 速度来源拆解**（A6000 GPU 上单步动作 latency）

| OpenVLA-7B → OpenVLA-1B | TinyVLA-1B |
|:-----|:-----|
| 292 ms → 140 ms | **14 ms** |

结论：VLM 缩放（7B→1B）只贡献 2× 加速；另外 10× 来自 diffusion head 取代 AR token 生成。这个拆解比主实验更清晰。

**Table V. Policy head 消融**（TinyVLA-H backbone 固定）

| Policy Head | PlaceTennis | FlipMug | StackCubes | CloseDrawer | OpenBox |
|:-----|:-----|:-----|:-----|:-----|:-----|
| MLP | 0 | 0 | 0 | 0 | 0 |
| ACT | 13.3 | 8.3 | 8.3 | 13.3 | 23.3 |
| Diffusion | **90** | **98.3** | **98.3** | **96.7** | **86.7** |

MLP 全军覆没说明 VLM condition + MLP head 无法优化；ACT 可以收敛但远不如 diffusion。Diffusion 在这种 condition-rich scenario 下明显优越。

### Generalization

论文做了 6 个维度的泛化实验，TinyVLA 普遍 ≥ OpenVLA：

- **Instruction（Figure 4）**：未见颜色（green mug）、Seen-object 跨任务重组（"pick the cube"）、未见物体 + 新功能（"pick car, place into box"）均成功
- **View（Figure 5）**：相机视角 ±30° 内 TinyVLA 仍鲁棒，DP 轻微视角变化即失败
- **Background（Figure 6）**：6 种不同桌面材质都成功
- **Distractor & Illumination（Figure 7）**：不加数据增强即可容忍 L1/L2 级 distractor 和低光条件；OpenVLA 在低光下失败
- **Appearance（Figure 8）**：物体颜色改变（棕 mug → 其他色）仍成功，归因于 VLM 预训练先验
- **Spatial（Figure 9）**：远离训练位置的 OOD 位置上 **OpenVLA 略好于 TinyVLA**——这里 OpenX 大规模机器人数据的优势显现，是论文少有的 TinyVLA 不占优的维度

**Figure 10. Failure mode 按 VLM 尺寸分解**：0.4B 主要死在 instruction 理解错误；升到 1.3B 后主要剩下定位不准；升到 3B（PaliGemma）后定位错误进一步减少。
![](https://arxiv.org/html/2409.12514v5/x10.png)

---

## 关联工作

### 基于
- [[2303-DiffusionPolicy|Diffusion Policy]]（DDPM-based visuomotor policy）：直接作为 action head
- LLaVA：VLM 训练 recipe
- Pythia：语言模型后端
- LoRA（Hu et al. 2022）：parameter-efficient fine-tuning

### 对比
- [[2406-OpenVLA|OpenVLA]]：7B Prismatic + OpenX 970K 预训练 + AR action token；TinyVLA 主要对标对象
- [[2307-RT2|RT-2]]：VLA 范式的源头，discrete action tokenization 的代表
- [[2405-Octo|Octo]]：cross-embodiment 预训练路线

### 方法相关
- MobileVLM v2 / LLaVA-Phi（small VLM 探索）：<3B 级 efficient multimodal
- PaliGemma：3B VLM，在 ablation 中作为更大 backbone 验证 scaling
- Action Chunking Transformer ([[2401-MobileALOHA|MobileALOHA]] 系)：policy head 消融对比项

### 后续影响
- [[2506-SmolVLA|SmolVLA]]：继承 "小 VLM + flow matching head" 思路，把规模推到 450M
- [[2502-OpenVLA-OFT|OpenVLA-OFT]]：从 OpenVLA 侧引入 continuous action decoding（parallel decoding / L1 regression），收敛到相近结论——AR action token 是 VLA 延迟的主要瓶颈

---

## 论文点评

### Strengths

1. **问题诊断清晰**：把 VLA 的 latency 拆成 "VLM 太大" 和 "AR token 生成" 两个独立源，并用 Table IV 的 OpenVLA-1B 中间点分别量化贡献——这种解耦实验比主实验更有说服力
2. **双臂结果是一个有力的 negative result**：OpenVLA 在 bimanual 上 0% 成功暴露了 OpenX 预训练的 distribution lock-in 问题，这是一个很少被其他 VLA 论文直接指出的 cautionary signal
3. **数据效率 claim 有 grounding**：每任务 100 trajectories 相比 OpenVLA 的 970K pretraining + 微调是数量级的差距，且 5 任务主实验上确实更强
4. **泛化评测覆盖面广**：instruction / view / background / distractor / illumination / appearance / spatial 7 个维度系统测试，比大多数 VLA 论文更严谨

### Weaknesses

1. **Backbone 选择已过时**：Pythia 在 2024 年已非最强 <2B 语言模型，作者自研 Llava-Pythia 而没有直接用更新的小 VLM（如 PaliGemma、Qwen2-VL）。Table V 消融中用 PaliGemma 作为 TinyVLA-3B 的提示是对的，但没有把这一档作为主线配置
2. **任务难度偏低**：5 个单臂任务都是桌面短时程 pick-and-place + 姿态调整，长时程操作、接触密集任务、需要精细力控的任务都未涉及——"VLM 先验足够替代 robot data 先验" 的 claim 需要在更难的任务上验证
3. **Spatial generalization 是弱项但被低调处理**：Figure 9 显示远距离 OOD 位置 OpenVLA 更好，但作者一笔带过。这其实是 "OpenX 大规模机器人数据的价值在哪里" 的关键证据
4. **Diffusion sampling step 未报告**：14ms 的推理延迟对应多少 denoising steps？消融里没有讨论 diffusion steps 对延迟/性能的 trade-off
5. **Adaptive pooling 黑盒**：把可变长度视觉 token 压成固定长度 condition 这个操作没有 ablation，也没有讨论对 localization 泛化的影响
6. **数据绝对量不突出**：100 trajectories × 5 任务 = 500 demos，但 OpenVLA 论文里微调阶段通常也就几百到几千 demos，所谓 "data-efficient" 主要是省了 970K pretraining；微调数据本身并没有显著更少

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training（GitHub 2025-02-17 release；但仓库已 stale，>400 天未 push，0 个 90d commits）
- **模型权重**: 三档 VLM backbone 全部公开（Llava-Pythia 400M/700M/1.3B on HuggingFace @lesjie）；VLA 训练后权重未直接发布，但脚本齐全
- **训练细节**: 高层描述 + 脚本（OUTPUT / task_name / model_name_or_path 等超参位置），LoRA 配置和 diffusion 超参需要读代码
- **数据集**: MetaWorld 开源；真机 5 任务 + 双臂 3 任务的 teleoperation 数据未公开

#### Claim 可验证性
- ✅ "20× faster than OpenVLA"：Table IV 同 GPU 同 backbone 尺寸对比，292ms/14ms，可验证
- ✅ "5.5× 参数更少 + 25.7pp 更高成功率"：Table II 数据支持（1.3B vs 7.2B = 5.54×；94.0 - 68.3 = 25.7）
- ✅ "OpenVLA bimanual 0%"：Table III 三任务全 0，grounding 很硬
- ⚠️ "No pretraining needed"：严格说 VLM 预训练仍在，只是跳过了 OpenX 级的 robot-data pretraining；措辞 "eliminating the need for pre-training stage" 容易误导
- ⚠️ "Strong generalization"：Figure 5-9 的泛化实验每个 setting 只 2-6 trials，样本量偏小；spatial generalization 其实 OpenVLA 更好
- ❌ "VLM 先验可以替代 robot 预训练"：任务难度和 horizon 都偏低，这个 claim 在 long-horizon / contact-rich 任务上没有证据

### Notes

- TinyVLA 的真正贡献是把 "小 VLM + 连续动作 head" 这条路线的可行性做实，并用解耦实验锁定 AR action token 是延迟主因。从今天（2026-04）回看，这个判断被 [[2502-OpenVLA-OFT|OpenVLA-OFT]]、π0、[[2506-SmolVLA|SmolVLA]] 全部验证，属于方向正确
- 但自己的实现（Pythia backbone、桌面任务、stale 代码仓库）没有承接这个路线的后续演化——SmolVLA、π0 在更现代的 backbone 和任务难度上把这条路线推得更远
- 对 embodied AI 研究者：读这篇主要是为了理解 "为什么小 VLM + diffusion head 的范式能 work"，而不是作为具体实现参考

### Rating

**Metrics** (as of 2026-04-22): citation=302, influential=11 (3.6%), velocity=15.9/mo; HF upvotes=0; github 67⭐ / forks=10 / 90d commits=0 / pushed 407d ago · stale

**分数**：2 - Frontier

**理由**：302 citation / 15.9 per month 表明这是被 VLA 社区广泛引用的工作，尤其作为 "不依赖 OpenX 的小型 VLA" 范式代表。但 influential citation 比例偏低（3.6%，远低于 π0 的 19%、典型的 ~10%）说明更多是被当 landmark reference 提及而非技术实质继承——后续工作如 [[2506-SmolVLA|SmolVLA]]、[[2502-OpenVLA-OFT|OpenVLA-OFT]] 重新实现了核心 insight 但没有直接继承 TinyVLA 的代码/权重。github 67 stars + 仓库 stale（>400 天未 push）也印证了这点。距离 3 - Foundation 的差距在于：核心 insight（VLA 不需要 OpenX，AR token 是延迟主因）被独立验证但具体实现未成为事实标准；距离 1 - Archived 的距离在于：这是方向里的 paradigm-shifting 早期工作，任何小型 VLA 相关研究都绕不开。
