---
title: "OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving"
authors: [Lening Wang, Wenzhao Zheng, Yilong Ren, Han Jiang, Zhiyong Cui, Haiyang Yu, Jiwen Lu]
institutes: [Beihang University, UC Berkeley, Tsinghua University]
date_publish: 2024-05-30
venue: arXiv
tags: [world-model, 3D-representation]
paper: https://arxiv.org/abs/2405.20337
website: https://wzzheng.net/OccSora
github: https://github.com/wzzheng/OccSora
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving
> - **核心**: 把自动驾驶的世界模型从 autoregressive next-token 改成 diffusion-based 4D occupancy 生成，用 trajectory 作为控制条件，一次性生成 16s 的 4D 占据视频
> - **方法**: 两阶段——(1) 4D Scene Tokenizer (3D-conv VQVAE，时空联合压缩 32×) + (2) DiT 在 latent token 上做 diffusion，trajectory MLP 嵌入作 condition
> - **结果**: nuScenes-Occ3D 上 FID=8.35（首个 4D 占据生成的可比 baseline），重建 mIoU 27.4 (压缩比 512x，是 OccWorld 16x 压缩比下 mIoU 65.7 的约 50%)
> - **Sources**: [paper](https://arxiv.org/abs/2405.20337) | [website](https://wzzheng.net/OccSora) | [github](https://github.com/wzzheng/OccSora)
> - **Rating**: 2 - Frontier（首个把 diffusion 引入 4D occupancy 生成的子方向代表，被 DynamicCity / OccLLaMA / OccTENS 作为 baseline 引用；但重建质量、trajectory control 有效性、下游验证均有明显 gap，属 Frontier 档）

**Key Takeaways:**
1. **Diffusion-as-world-model 替代 autoregressive**: OccWorld 这类 next-token 占据预测在长序列上误差累积、效率低；OccSora 一次性 denoise 一段时空 latent，回避了这两个问题。
2. **4D 联合 tokenizer**: 3D-conv VQVAE 同时在 T/H/W/D 上下采样 8×（总压缩 512×），把整段 32 帧占据压成 128×4×25×25 的 token，让 diffusion 在 manageable 的 latent 上跑。
3. **Trajectory-as-prompt**: ego 轨迹经 MLP 嵌入 + sin/cos timestep 嵌入，concat 进 DiT block；ablation 显示去掉 trajectory embedding 后 FID 8.35→17.48，去掉 timestep embedding 后 FID→87.26（trajectory 控制其实没那么 dominant，timestep 才是关键）。
4. **代价**: 极激进的时空压缩换来生成能力，但重建只剩 mIoU 27.4；小物体（bicycle 0.0、motorcycle 8.7）几乎丢失——这是把"world simulator"这个概念落到 occupancy 上的一个未解决核心矛盾。

**Teaser. Paper 概览图：和现有 autoregressive 世界模型的对比，以及 trajectory-controllable 长序列生成示例。**

![](https://arxiv.org/html/2405.20337v1/x1.png)

---
## 背景与动机

自动驾驶的世界模型最近从 image-based（GAIA-1、DriveDreamer，依赖 3D bbox 控制）发展到 occupancy-based（OccWorld，next-token 预测 occupancy）。但作者指出 occupancy world model 普遍有两个 pain point：

1. **Autoregressive 长序列效率低**：OccWorld 这类方法逐帧生成，长序列（16s+）下既慢又容易漂移。
2. **依赖历史/先验输入**：典型 setup 是 "给前几帧 + 当前 bbox/map → 预测下几帧"，这本质是 forecasting 而不是真正的 generation；缺乏从 noise 出发只靠 action 控制的能力。

作者类比 Sora 在 2D video 上做的事——把 latent diffusion + transformer 的范式直接搬到 3D occupancy 上，做"4D 占据生成"。这是首篇明确把 occupancy 长序列生成 framing 成 diffusion 任务的工作（同期 SemCity 做的是静态 3D 场景）。

> ❓ 这里的"first 4D occupancy world model"措辞偏 marketing。论文承认 OccWorld 已经是 occupancy world model，区别只是 autoregressive vs diffusion；"4D" 是 occupancy 加 time，OccWorld 也是 4D 的。所以"first generative 4D occupancy world model" 这个 claim 严格成立的边界是"first **diffusion-based** 4D occupancy generation model"。

---
## 方法

### Pipeline 总览

整体是经典的 latent diffusion 两阶段：先训 4D VQVAE 把 occupancy 序列压成 discrete tokens，再训 DiT 在 token 空间上做 conditional diffusion。

**Figure 2. OccSora 整体 pipeline。** 左侧 4D occupancy scene tokenizer 完成原始 occupancy 序列的压缩与重建；右侧 diffusion-based world model 把压缩 token 与 ego 轨迹一起作为输入，训练 denoising，推理时从随机噪声 + 任意轨迹生成 controllable token，再走 tokenizer decoder 还原为 4D occupancy。

![](https://arxiv.org/html/2405.20337v1/x2.png)

### 4D Occupancy Scene Tokenizer

输入 $R_{in} \in \mathbb{R}^{B \times D \times H \times W \times T}$（B=batch, T=32 帧, $H \times W \times D = 200 \times 200 \times 16$ voxel），输出压缩 token $R_{mi} \in \mathbb{R}^{B \times c \times h \times w \times t} = B \times 128 \times 25 \times 25 \times 4$。

总体结构是一个 **3D-conv VQVAE**：

1. **Category embedding**：每个 voxel 的 18 类语义 ID 先映射成 learnable embedding $b \in \mathbb{R}^{c'}$，沿 feature 维拼接，再 reshape 成 $R_{in}' \in \mathbb{R}^{B \times (Dc') \times T \times H \times W}$，把 D 维度 fold 进 channel——后续直接当 3D（T, H, W）卷积处理。
2. **3D Encoder**：三次 3D 下采样卷积，每次 T/H/W 同步 ÷2，最终得到 $R_{in}'' \in \mathbb{R}^{B \times (8Dc') \times T/8 \times H/8 \times W/8}$（即 4×25×25）。每个 block 后面接 dropout 做正则。下采样末尾插一个 **cross-channel attention**：把 channel 维 split 成多组，组间做 attention，再 reshape 回原 shape。
3. **Codebook 量化**：codebook $\zeta_{token} \in \mathbb{R}^{N \times D}$，每个空间位置的 feature 找最近邻 code：

$$
R_{mi}^{(ij)} = \min_{b \in \zeta_{token}} \| \widehat{R_{mi}^{(ij)}} - b \|_2
$$

4. **3D Decoder**：对称结构，3D 反卷积把 $R_{mi}$ 上采样回原始分辨率，再沿 channel split 还原 D 维，得到重建的 occupancy $R_o$。

> ❓ 论文没明确说 codebook 大小 N，也没说重建 loss 配方（CE? focal? class-weighted?），这些都是 occupancy reconstruction 的关键超参，github README 也没补全。

**Figure 3. 4D occupancy scene tokenizer 结构。** 编码端的 3D conv + cross-channel attention 把 4D 占据压缩成 latent，解码端做对称重建。

![](https://arxiv.org/html/2405.20337v1/x3.png)

### Diffusion-based World Model

#### Token & trajectory embedding

把 $R_{mi}$ flatten 成 $R_{re} \in \mathbb{R}^{B \times c \times (hwt)}$，加 sin/cos 位置编码：

$$
R_{re}^{(\mathrm{emb})} = \mathrm{emb}_i^d + R_{re}
$$

Trajectory 输入 $T_r \in \mathbb{R}^{B \times t \times 2}$（t 个时间步，每步 (x,y) 两个 coord），reshape 成 $\mathbb{R}^{B \times (t \times 2)}$ 后过 MLP $\delta(\cdot)$；和 timestep embedding $\nu(t)$ 相加得到条件向量 $g$：

$$
g = \nu(t) + \delta(T_r)
$$

#### Diffusion 训练

标准 DDPM 公式。Forward 加噪：

$$
q(R_{re}^{g} | R_{re}) = \mathcal{N}\left(R_{re}^{g}; \sqrt{\overline{\sigma^g}} R_{re}, (1 - \overline{\sigma^g}) I\right)
$$

Reverse 用 DiT 预测噪声，$L_{simple} = \frac{1}{2} (\hat{R}_{re}^{g} - R_{re}^{g})^2$，最后再用完整 KL loss finetune（沿 Dhariwal & Nichol 的训练策略）。

#### DiT 主干

模型用的是 **DiT-XL/2**（Peebles & Xie），把 token + 条件 $g$ 一起喂进 transformer block，输出 denoised token，最后过 VQVAE decoder 还原 4D occupancy。

**Figure 4. Diffusion-based world model 结构。** VQVAE 训练好的最优 codebook 把 occupancy 转成 token 序列；token + ego 轨迹 + 噪声拼成 DiT 输入，做 denoising 训练，推理时只给随机噪声和目标轨迹即可生成。

![](https://arxiv.org/html/2405.20337v1/x4.png)

---
## 实验

### 实验配置

- **数据**：nuScenes，使用 Occ3D 标注（200×200×16 voxel，18 类）
- **训练序列长度**：32 连续帧
- **硬件**：8× A100 80G
- **VQVAE**：150 epochs，约 50.6 小时，每卡 42GB 显存，batch size 2/卡
- **DiT**：1.2M steps，约 108 小时，每卡 47GB 显存
- **优化器**：AdamW, lr=1e-5, weight decay=0.01

### 4D 重建质量

**Table 1. 4D 占据重建的定量比较。** 即便压缩比 512×（OccWorld 是 16×，相当于压缩了 32 倍多），OccSora 还能保留 OccWorld 约一半的 mIoU。

| Method | Ratio | IoU | mIoU | bicycle | motorcycle | pedestrian | traffic cone | car | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OccWorld | 16× | 62.2 | 65.7 | 69.6 | 70.7 | 74.8 | 67.6 | 69.4 | 65.4 |
| OccSora | 512× | 37.0 | 27.4 | **0.0** | **8.7** | **11.5** | **3.5** | 29.0 | 29.0 |

> ❓ "保留一半 mIoU" 这个说法掩盖了一个严重问题：OccSora 对小物体的重建几乎崩溃——bicycle=0、motorcycle=8.7、pedestrian=11.5、traffic cone=3.5。这是激进时空压缩的必然代价，但论文没有正面讨论这个 failure mode 对"world simulator for decision-making"宣称的影响——下游 planning 最在意的恰恰是 vulnerable road users。

**Figure 5. 4D 占据重建可视化。**

![](https://arxiv.org/html/2405.20337v1/x5.png)

### 4D 生成 (FID)

**Table 2. OccSora 与其他生成模型的对比。** 这是 cross-modal 的对比（image vs 2D video vs 3D occupancy 静态 vs 4D occupancy video），FID 数字之间不严格可比，但作者把它当作 sanity check。

| Method | Type | Dimension | Dataset | FID |
| --- | --- | --- | --- | --- |
| DiT | Image | 2D | ImageNet | 12.03 |
| MagicDrive | Video | 3D | nuScenes | 14.46 |
| DriveDreamer | Video | 3D | nuScenes | 14.9 |
| DriveGAN | Video | 3D | nuScenes | 27.8 |
| SemCity | Occupancy | 3D | KITTI | 40.63 |
| **OccSora** | Occupancy Video | 4D | nuScenes | **8.35** |

> ❓ FID 在 occupancy domain 是怎么算的？文章没解释。FID 通常基于 Inception 在 RGB 上预训练；occupancy voxel 直接喂 Inception 不合理。这个 8.35 跟 ImageNet DiT 的 12.03 没法直接比。

**Figure 6. 训练 iteration 从 10k → 1.2M 的进度可视化。** 随着训练步数增加，生成场景逐渐清晰、coherent。

![](https://arxiv.org/html/2405.20337v1/x6.png)

### Trajectory-controllable 生成

**Figure 7. 不同输入轨迹下的 4D 占据生成。** 从上到下分别是直行、右转、静止；每个生成场景对应轨迹，保持逻辑连贯性。

![](https://arxiv.org/html/2405.20337v1/x7.png)

**Figure 8. 同一轨迹下生成 diverse 的连续场景。** 树木和道路环境随机变化，但保持原始轨迹的 logic。

![](https://arxiv.org/html/2405.20337v1/x8.png)

作者还在 project page 提供了更直观的演示视频。

**Video. Project page 上的 trajectory-aware 4D 占据生成 demo。**
<video src="https://wzzheng.net/videos/demo.mp4" controls muted playsinline width="720"></video>

### Ablation

**Table 3. 不同压缩率/组件/通道维度的 ablation。** 注意 IoU/mIoU 在前 3 行都是 37.03/27.42——这意味着 trajectory embedding 和 timestep embedding 都不影响重建（毕竟它们只在 diffusion 阶段使用），只影响 FID。

| Input Size | Token Size | Channel | Class | T embed. | Trajectory | IoU | mIoU | FID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32×200×200 | 128×4×25×25 | 8 | ✓ | ✓ | ✓ | 37.03 | 27.42 | **8.34** |
| 32×200×200 | 128×4×25×25 | 8 | ✓ | ✗ | ✓ | 37.03 | 27.42 | 87.26 |
| 32×200×200 | 128×4×25×25 | 8 | ✓ | ✓ | ✗ | 37.03 | 27.42 | 17.48 |
| 32×200×200 | 128×4×25×25 | 4 | ✓ | ✓ | ✓ | 29.67 | 23.21 | 34.24 |
| 32×200×200 | 128×8×50×50 | 8 | ✓ | ✓ | ✓ | 32.91 | 24.4 | 72.32 |
| 12×200×200 | 64×3×50×50 | 8 | ✓ | ✓ | ✓ | 26.73 | 14.12 | 187.78 |
| 12×200×200 | 64×3×25×25 | 8 | ✓ | ✓ | ✓ | 22.42 | 9.27 | 270.23 |
| 12×200×200 | 32×3×25×25 | 8 | ✓ | ✓ | ✓ | 13.60 | 3.85 | 465.18 |

**关键观察**：
- **Timestep embedding 是核心**: 去掉后 FID 8.34→87.26（10×恶化）。
- **Trajectory embedding 影响中等**: 去掉后 FID 8.34→17.48（2×恶化），相对而言比想象中小——意味着模型生成的"控制性"主要来自 timestep 隐式编码的时间结构，而非显式的 trajectory 控制。
- **更小压缩比反而更糟**: token size 128×8×50×50（比 4×25×25 大 16×）的 FID 是 72.32 vs 8.34；说明 DiT 在更长的 token 序列上没有 scale 起来，这是 capacity 不足的信号而非架构 fundamental 问题。
- **更短输入序列性能崩溃**: 12 帧输入相比 32 帧 FID 飙升一个数量级——OccSora 的"长序列建模"实际上是依赖训练序列长度的，不是真的"长序列泛化"。

> ❓ Trajectory 去掉后 FID 17.48 而不是 87.26 这个数字让人怀疑：trajectory 真的在控制生成吗？还是说模型主要依赖 token 的 spatial-temporal prior 来生成"看起来像驾驶场景"的内容？需要看 trajectory swap 的实验（用一个轨迹的噪声配上另一个轨迹的 condition，看输出是否真的跟随了新轨迹），论文没做。

### Denoising step / ratio 分析

**Table 4. 不同 denoising step 数和 denoising ratio 的影响。** Denoising rate 和 token size 比 step 数对生成质量影响大得多。

| Step | Input | Token | FID@10% | FID@50% | FID@90% | FID@100% |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 32×200×200 | 128×4×25×25 | 49863 | 17630 | 42 | 9.1 |
| 100 | 32×200×200 | 128×4×25×25 | 53297 | 19471 | 72 | 10.08 |
| 1000 | 32×200×200 | 128×4×25×25 | 32171 | 5924 | 17 | 8.94 |
| 10 | 12×200×200 | 64×3×50×50 | 71293 | 5644 | 742 | 431 |
| 100 | 12×200×200 | 64×3×50×50 | 81274 | 45346 | 456 | 446 |
| 1000 | 12×200×200 | 64×3×50×50 | 43631 | 17431 | 379 | 353 |

**Figure 9. 不同 denoising ratio / step / trajectory 下的影响可视化。**

![](https://arxiv.org/html/2405.20337v1/x9.png)

### Long-time 与场景泛化

**Figure 10. 长时序 4D 占据生成在四种轨迹（直行/右转/静止/加速）下的表现。**

![](https://arxiv.org/html/2405.20337v1/x10.png)

**Figure 11. 固定轨迹下生成不同场景的泛化能力。**

![](https://arxiv.org/html/2405.20337v1/x11.png)

---
## 关联工作

### 基于
- **OccWorld** (arXiv 2311.16038): 直接前作，autoregressive 占据世界模型；OccSora 把它的 next-token 范式换成 diffusion，并复用了 VQVAE 思路。
- **DiT** (Peebles & Xie, ICCV 2023): Diffusion Transformer 主干，OccSora 用的是 DiT-XL/2 配置。
- **VQVAE** (van den Oord, NeurIPS 2017): 离散 latent + codebook 的 tokenizer 范式。

### 对比
- **OccWorld**: 唯一同 modality 的 baseline，重建上 mIoU 65.7 (16× 压缩) vs OccSora 27.4 (512× 压缩)。
- **MagicDrive / DriveDreamer / DriveGAN**: image/video-based 自驾世界模型，做的是 RGB 视频生成，FID 14-28 区间；OccSora 把任务搬到 occupancy。
- **SemCity**: 3D 静态占据生成（KITTI），单帧；OccSora 加了时间维。

### 方法相关
- **Sora** (OpenAI, 2024): 灵感来源，把 latent diffusion 在视频生成上跑通；OccSora 是把这个 recipe 类比到 4D occupancy 的尝试。
- **GAIA-1** (Wayve, 2023): image-based 自驾世界模型，autoregressive；同期工作。
- **DriveWorld** (arXiv 2405.04390): 4D 预训练 + 世界模型用于 detection/planning，同月工作。
- **GenAD**: 生成式端到端自驾，相关方向但聚焦 trajectory generation。

### 后续方向参考
- **DynamicCity** (arXiv 2410.18084): 后续的大规模 4D occupancy 生成工作。
- **OccLLaMA** (arXiv 2409.03272): occupancy + language + action 联合的世界模型。
- **OccTENS** (arXiv 2509.03887): 用 temporal next-scale prediction 做 occupancy 世界模型。

---
## 论文点评

### Strengths

1. **范式转变到 diffusion**: 在 occupancy 长序列建模上，autoregressive 的 next-token 思路确实有 fundamental scaling/efficiency 的限制，OccSora 把 latent diffusion 套进来在概念上是 clean 的。
2. **真正的 long horizon**: 一次性生成 16s（32 帧）occupancy，比 OccWorld 这类逐帧 forecasting 的 setup 在使用场景上更接近"world simulator"——可以从 noise + trajectory 起点直接产 rollout。
3. **极激进的压缩比 ablation**: Table 3 的 ablation 在 input size / token size / channel 上做了较完整的扫描，提供了 informative 的 scaling signal，这是同类论文里不多见的细致度。
4. **代码 + 训练流程开源**: VQVAE + DiT 两阶段训练脚本都给了，包括如何从 VQVAE 生成 token 到训 DiT 的完整 pipeline。

### Weaknesses

1. **重建质量崩塌在小物体上**: bicycle=0、motorcycle=8.7、pedestrian=11.5；如果世界模型不能保留行人/骑行者，"用作 decision-making" 的 framing 就站不住——下游 planner 没法相信这个模型生成的场景。
2. **Trajectory control 的有效性存疑**: 去掉 trajectory embedding 仅让 FID 从 8.34 升到 17.48（vs timestep 去掉是 10× 恶化）；缺少 trajectory swap 实验来证明输出真的跟随条件而非仅产生 plausible 驾驶场景。
3. **"32 帧固定"严重限制了 generalization**: 12 帧训练的模型 FID 从 8.34 飙到 200+；这意味着 OccSora 实际上是 trained-at-32-frames 的特化模型，"long sequence generation" 这个标签需要打折——它生成的是定长序列，不是开放长度。
4. **FID 在 occupancy domain 的合理性没论证**: Table 2 的 cross-domain FID 比较缺乏 methodology 说明，不应直接 vs ImageNet DiT 的 FID。
5. **没有下游任务验证**: 没有任何下游指标（planner 性能、closed-loop driving、policy training 增益）证明这些生成的 4D occupancy 真的能 serve 自动驾驶决策；"world simulator" 这个 framing 完全停留在 generative quality 上。
6. **Codebook 大小 N 没披露**: 量化时的 codebook size 是 VQVAE 的核心超参，论文和代码 README 都没明确给出。
7. **Marketing 用语**: "first generative 4D occupancy world model" 中"4D"和"world model" 部分OccWorld 已经做了；严格的差异化是"first **diffusion-based**"。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training（GitHub README 提供 `train_1.py`/`step02.py`/`train_2.py`/`sample.py`/`visualize_demo.py` 完整 pipeline）
- **模型权重**: 未说明（README 引用 `/results/001-DiT-XL-2/checkpoints/1200000.pt` 但未给出下载链接）
- **训练细节**: 高层超参完整（lr、batch size、epoch 数、显存）；低层细节（codebook size N、loss 配方、cross-channel attention 头数）未披露
- **数据集**: 开源（nuScenes + Occ3D + TPVFormer 的 train/val pickle）

#### Claim 可验证性
- ✅ **可生成 16s/32 帧 4D occupancy 序列**：视频/图都展示了，长度是显式的训练 setup
- ✅ **Trajectory 影响生成**：Figure 7/8/10/11 视觉上确实不同轨迹给出不同场景；Table 3 ablation 验证 trajectory embedding 有作用
- ⚠️ **"模型理解空间-时间分布"**：FID=8.35 是 cross-domain 比较，没有标准化的 occupancy generation FID baseline 可参考；视觉效果在小物体上明显失真
- ⚠️ **"trajectory-aware control"**：缺 trajectory-swap 控制实验，不能排除 generation 主要被 spatial prior 驱动
- ⚠️ **"world simulator for decision-making"**：完全没有下游 planner / RL 验证；停留在 framing
- ❌ **"first 4D occupancy world model"**：OccWorld 已经是 4D occupancy world model；准确表述应为"first **diffusion-based** 4D occupancy generation model"

### Notes

- **核心 idea 是对的，执行有 gap**: "把 diffusion 搬到 occupancy 长序列生成" 在 2024 年 5 月是有意义的范式转换；但实际质量（小物体丢失、trajectory control 弱）说明这个方向需要 (a) 更大规模数据、(b) 更精细的 tokenizer (e.g., spatially-adaptive codebook)、(c) 更强的条件机制。
- **可能的 follow-up 思路**：
  - Trajectory + map + 其他 agent 行为联合 condition
  - 用 latent-based 而非 discrete VQVAE，避免 codebook bottleneck 在小物体上的失效
  - Closed-loop 评估：用 OccSora rollout 训 RL planner，看下游性能
- **对 world model 这个研究方向的启示**: occupancy 作为世界模型的 representation 比 RGB 更接近 driving 决策需要的信息（几何 + 语义），但代价是缺乏在大规模 web data 上预训练的 backbone；diffusion-based generation 在 occupancy 上的 scaling 行为还没被充分研究。

### Rating

**分数**：2 - Frontier
**理由**：在 occupancy-based world model 这个子方向里，OccSora 是把 diffusion 范式引入 4D occupancy 生成的首个代表性尝试，后续 DynamicCity / OccLLaMA / OccTENS 等工作在 framing 和 baseline 对比上多次引用它，属于该细分方向的必知参考。但它并非 foundational——重建质量在 VRU 上崩塌、trajectory control 有效性存疑、缺下游验证（见 Weaknesses 1/2/5），因此只能是 Frontier 档的代表 baseline，而非像 OccWorld / DiT 那种定义范式的 Foundation 工作。
