---
title: "MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based Vision-and-Language Navigation"
authors: [Lingfeng Zhang, Xiaoshuai Hao, Qinwen Xu, Qiang Zhang, Xinyao Zhang, Pengwei Wang, Jing Zhang, Zhongyuan Wang, Shanghang Zhang, Renjing Xu]
institutes: [HKUST(GZ), BAAI, Beijing Innovation Center of Humanoid Robotics, Wuhan University, Peking University]
date_publish: 2025-02-19
venue: ACL 2025
tags: [VLN, semantic-map, spatial-memory]
paper: https://arxiv.org/abs/2502.13451
website: 
github:
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] MapNav: Annotated Semantic Maps as VLN Memory
> - **核心**: 用单张可读文本注释的 top-down 语义地图（Annotated Semantic Map, ASM）替换 VLN 中的历史 RGB 帧序列，作为 VLM agent 的紧凑记忆表示
> - **方法**: 由 RGB-D + pose 投影点云 → 2D top-down 语义地图（Mask2Former 分割）→ 在每个语义连通域质心处嵌入文本标签生成 ASM，与当前 RGB + 指令一起喂入 LLaVA-OneVision (SigLIP + Qwen2-7B) 端到端预测 next action
> - **结果**: VLN-CE R2R Val-Unseen SR 36.5 / SPL 34.3（仅当前帧 + ASM），加 2 帧历史后 SR 39.7 / SPL 37.2 超越 NaVid 全帧 baseline；内存恒定 0.015 MB（vs NaVid 在 300 步时 276 MB），单步推理 0.25 s vs 1.22 s
> - **Sources**: [paper](https://arxiv.org/abs/2502.13451)
> - **Rating**: 2 - Frontier （把 "历史帧 vs 结构化地图" 作为 VLN memory 独立研究维度的代表工作，ACL 2025 长文，消融扎实；但 fair-comparison 局限且未开源 checkpoint，尚未长成该方向必读 Foundation）

**Key Takeaways:** 
1. **结构化文本注释让 VLM "看懂"地图**：比起原始 top-down 或语义掩码，把物体类别名直接写在地图上的对应位置，能利用 VLM 预训练时形成的物体-语言关联，注意力收敛到 labelled regions（attention peak >0.8 vs 语义图 <0.4）
2. **历史帧的核心价值是空间结构而非视觉细节**：在 VLN-CE 上，单 ASM + 当前帧已能 match NaVid 全历史帧（R2R SR 36.5 vs 37.4），加 2 帧历史就反超——意味着对当前主流 VLN-CE benchmark，时序视觉信息的边际价值远不如显式空间记忆
3. **VLM 不擅长 depth 模态**：RGB+Depth 反而比 RGB-only 更差（SR 23.1 vs 27.1），印证 "选择和预训练分布对齐的输入模态" 这一原则——把几何信息显式投影成 top-down 图比把 depth 当成额外通道更 work
4. **存储复杂度从 O(T) 到 O(1)**：固定大小语义地图作为 sufficient statistic，是把 VLM-based VLN 推向长 horizon 的必要条件之一

**Teaser. ASM 可视化与所含信息**

![](https://arxiv.org/html/2502.13451v2/x1.png)

ASM 在每个时刻包含：物理障碍物分布、已探索区域、agent 当前位置、历史轨迹、以及带文本标签的语义物体。通过把传统语义地图加上 "chair" / "plant" / "bed" 这样的显式文本，把空间信息桥接到 VLM 的语言理解通道。

---

## 1. Motivation：历史帧不是 VLN 的好记忆

VLN-CE 任务里，agent 要在连续 3D 环境中按自然语言指令导航。主流方法（NaVid 等）把所有历史 RGB 帧塞给 VLM 当 spatio-temporal context，问题：

- **存储线性增长**：300 步时 NaVid 累积 276 MB 视觉特征
- **缺结构化理解**：raw frame 序列里物体位置、可达区域、已走轨迹都是隐式的，VLM 需要从像素里二次推断
- **推理慢**：每步要重新处理所有历史帧

作者提出的反命题：**对 VLN 的核心是空间结构而非视觉细节**，所以一张持续更新的 top-down 语义地图（加文本注释让 VLM 能读）应该足以替代历史帧。

> ❓ 这个 "替代" 在 VLN-CE benchmark 上成立，但只覆盖 indoor、静态、object-rich 场景；在 outdoor / dynamic / 需要识别细粒度视觉线索（如 "the red door at the end of the hallway"）的 instruction 上，丢掉 RGB 历史可能就不再 work。

---

## 2. Method

### 2.1 ASM Generation Pipeline

**Figure 3. ASM 生成流程**

![](https://arxiv.org/html/2502.13451v2/x3.png)

每个 timestep：

1. RGB → Mask2Former 得到语义 mask
2. Depth 投影成 3D 点云，与 mask 对齐 → 给点云每个 voxel 打上语义标签
3. 用 pose 把点云投到全局坐标系，再投影到 2D 平面，更新多通道 semantic map $\mathbf{M}\in\mathbb{R}^{C\times W\times H}$，其中 $C = C_n + 4$：
   - 通道 1-4：物理障碍、已探索区域、当前位置、历史轨迹
   - 通道 5..C：每个物体类别一个 channel
4. 对每个 object channel 做 connected component 分析，对面积超过阈值 $\tau$ 的语义区域计算几何质心，在该质心处叠加该类别的文本标签，得到 ASM

**关键 trick**：把语义信息从 "channel index" 转化为 "可读文本"，让 VLM 能用预训练的 word→object 知识理解地图。

### 2.2 Agent Architecture

基于 LLaVA-OneVision：
- Vision encoder: SigLIP-so400m-patch14-384（共享给 RGB 观测和 ASM）
- LLM backbone: Qwen2-7B-Instruct
- 两个独立 MLP projector 分别处理 obs 和 ASM 特征：

$$
\mathbf{E}_{t}=P^{obs}_{mlp}(\mathbf{F}_t),\quad \mathbf{E}^{M}_{t}=P^{map}_{mlp}(\mathbf{F}^{M}_{t})
$$

最终输入 token 序列：

$$
\mathbf{V}_{t}=[\text{TASK};\mathbf{E}_{t};\text{OBS};\mathbf{E}^{M}_{t};\text{MAP}]
$$

### 2.3 Action Prediction

VLM 直接输出自然语言（如 "move forward"），用正则表达式匹配映射到离散动作 {FORWARD, TURN-LEFT, TURN-RIGHT, STOP}。每个动作维护一个同义词表（"proceed"/"halt"/"wait" 等），靠 pattern matching 容错 VLM 的措辞变化。

> ❓ 这种 "字符串匹配" 风格的 action decoder 在 SR 36% 量级勉强够用，但低估了 VLM 输出格式的不稳定性。更鲁棒的做法是 constrained decoding 或加一个轻量 action classification head。

### 2.4 Framework Overview

**Figure 2. MapNav 整体框架**

![](https://arxiv.org/html/2502.13451v2/x2.png)

---

## 3. Experiments

### 3.1 VLN-CE 主结果

**Table 1.（节选）R2R / RxR Val-Unseen 上 MapNav vs SOTA**

| Method | Cur. RGB | His. RGB | R2R NE↓ | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ |
|---|---|---|---|---|---|---|---|
| NaVid (All RGB Frames) | ✓ | ✓ | 5.47 | 37.4 | 35.9 | 23.8 | 21.2 |
| NaVid (Cur. RGB only) | ✓ |  | 8.10 | 13.0 | 7.8 | 8.7 | 6.8 |
| MapNav (w/o ASM + Cur. RGB) | ✓ |  | 7.26 | 27.1 | 23.5 | 15.6 | 12.2 |
| MapNav (w/ ASM + Cur. RGB) | ✓ |  | 5.22 | 36.5 | 34.3 | 22.1 | 20.2 |
| MapNav (w/ ASM + Cur. RGB + 2 His. RGB) | ✓ | ✓ | **4.93** | **39.7** | **37.2** | **32.6** | **27.7** |

关键比较：
- **MapNav (w/ ASM, no history) 几乎追平 NaVid (All Frames)** —— 36.5 vs 37.4 SR on R2R，验证 ASM 能替代历史帧
- **加仅 2 帧历史就反超** NaVid 全历史 baseline（39.7 vs 37.4）

> ⚠️ 表中其他 panoramic + waypoint predictor 方法（HNR/BEVBert/ETPNav）的 SR 在 57-61%，MapNav 单 RGB monocular 的设定不可比，这是 fair-comparison 而非全局 SOTA。

### 3.2 内存与速度

**Table 3. 内存与推理时间**

| Method | 1 Step | 10 Steps | 100 Steps | 300 Steps | Avg Time |
|---|---|---|---|---|---|
| NaVid | 0.92 MB | 9.2 MB | 92 MB | 276 MB | 1.22 s |
| MapNav | 0.015 MB | 0.015 MB | 0.015 MB | 0.015 MB | 0.25 s |

ASM 是 sufficient statistic，存储与轨迹长度无关；推理时间下降 79.5%（不需要重处理全历史帧）。

### 3.3 关键消融

**Table 4. 不同地图表示**

| Map Type | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| w/o Map | 7.26 | 41.2 | 27.3 | 23.2 |
| Original Top-Down | 8.93 | 35.1 | 26.4 | 21.9 |
| Semantic Map | 6.56 | 43.2 | 29.1 | 24.5 |
| **ASM** | **5.22** | **50.3** | **36.5** | **34.3** |

注意：**原始 top-down 地图反而比 no-map 更差**——这进一步说明 VLM 对没见过的 abstract 地图表示理解很差，需要文本 grounding。

**Table 6. 输入模态消融（反直觉发现）**

| Input | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| Only RGB | 7.26 | 41.2 | 27.1 | 23.5 |
| RGB + Depth | 8.82 | 35.6 | 23.1 | 19.9 |
| RGB + ASM | **5.22** | **50.3** | **36.5** | **34.3** |

直接给 VLM 喂 depth 反而掉性能——VLM 预训练分布里没有 depth map，相当于 OOD 输入。这印证了 ASM 路线的本质优势：**把几何信息显式 render 成 VLM 能理解的形式（带物体名的 top-down 图）**，而不是寄望 VLM 适配新模态。

**Table 8. 分割模型消融**

| Segmentation | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| YOLOv8 | 6.43 | 45.2 | 31.5 | 29.6 |
| MobileSAM | 6.02 | 48.9 | 34.8 | 31.6 |
| Mask2Former | **5.22** | **50.3** | **36.5** | **34.3** |

ASM 的 ceiling 受限于上游分割质量——5 个 SR point 的 gap 全来自分割。

### 3.4 Real-world

5 个真实场景（meeting room / office / lecture hall / tea room / living room），各 10 条指令，部署在 Unitree Go2 + RealSense D435i 上：

**Table 2.（聚合）Real-world SR**

| Method | Simple I.F. (avg) | Semantic I.F. (avg) |
|---|---|---|
| WS-MGMap | 58% | 26% |
| NaVid | 68% | 46% |
| **MapNav** | **80%** | **64%** |

semantic instruction（"move forward to the refrigerator and then turn right"）上提升尤其明显——ASM 显式 ground 了物体位置。

> ⚠️ 真实场景每个只有 10 条指令，绝对数字背后只有 3 条 episode 差异，方差巨大，"30% improvement" 的表述容易 over-claim。

### 3.5 Object Goal Navigation 零样本

在 HM3D 上 zero-shot 测试 ObjectNav：

| Method | SR↑ | SPL↑ |
|---|---|---|
| ZSON | 25.5 | 12.6 |
| ESC | **35.5** | 23.5 |
| Navid | 32.5 | 21.5 |
| MapNav | 34.6 | **25.6** |

虽然没在 HM3D 训过，泛化到 ObjectNav 接近专门方法。

### 3.6 VLM Attention 分析

**Figure 9. 不同地图表示下的 VLM 注意力可视化**

![](https://arxiv.org/html/2502.13451v2/x9.png)

ASM 输入下注意力 sharp peak (>0.8) 精确对齐到带标签的物体；语义图 peak <0.4，原始 top-down peak <0.3。VLM 描述 ASM 时会输出 "bed in the upper left part" 这样的精确空间-语义 grounding；描述原始地图只能说 "a black and white map with a blue line"。

---

## 4. Dataset & Training

- **数据规模**：~1M 训练对，含 R2R + RxR ground truth (300k) + DAgger (200k) + collision recovery (25k) + 600k 通用 VL co-training 样本
- **训练**：8×A100, 30 小时, 240 GPU-hours, 1 epoch, lr=1e-6, bf16, frozen vision encoder
- **DAgger 是最大增量**：消融显示 +DAgger 比纯 GT 增加 ~10 SR points

---

## 关联工作

### 基于
- **[[2402-NaVid|NaVid]]** (Zhang 2024): 主要 baseline 和对比对象。NaVid 把所有历史 RGB 帧塞给 VLM，MapNav 论证这种做法 unnecessary
- **LLaVA-OneVision** (Li 2024): 直接 backbone 框架（vision-language 共享 encoder + 模态-specific projector）
- **Mask2Former** (Cheng 2022): 上游语义分割
- **Qwen2-Instruct**: LLM backbone

### 对比
- **[[2304-ETPNav|ETPNav]]** (An 2024): topological waypoint predictor，panoramic + multi-sensor，SR 57%（不可直接对比）
- **VLFM** (Yokoyama 2024): value map for waypoint selection，类似的 spatial memory 思路但用 value 而非 semantic+text
- **WS-MGMap** (Chen 2022): weakly-supervised multi-granularity map，主要 map-based VLN baseline
- **GridMM** (Wang 2023): grid memory map，map-based memory 早期工作

### 方法相关
- **InstructNav** (Long 2024): 把导航分解成 subtask + value map
- **Nav-CoT** (Lin 2024): chain-of-thought for VLN
- **VoroNav** (Wu 2024): Voronoi-based topological map
- **MC-GPT** (Zhan 2024): memory map + reasoning chain
- **DAgger** (Ross 2011): interactive imitation learning，文中 +200k 训练样本来源

---

## 论文点评

### Strengths

1. **问题选得对**：把 "memory representation in VLM-based VLN" 单独拎出来作为一个研究维度，比 "再加一个 module 提点" 有意义。结论 "历史帧不必要，结构化空间记忆足够" 在 VLN-CE 上是 publishable insight
2. **ASM 这个表征很优雅**：把 "VLM 看不懂的语义掩码" 变成 "VLM 看得懂的带文字 top-down 图"，思路 simple 且可 scale。文本 grounding 的注意力分析也直接验证了机制
3. **消融做得扎实**：top-down vs semantic vs ASM 的对比清楚说明 "文本注释" 是核心，而非语义信息本身；RGB+Depth 反而下降的反直觉发现也很有价值
4. **效率优势显著**：O(1) 内存 + 79.5% 推理加速，对长 horizon 部署是实质改进

### Weaknesses

1. **Fair comparison 局限**：只跟 monocular RGB-based VLM 方法（NaVid）比，没跟 panoramic + waypoint predictor 流派（HNR/BEVBert/ETPNav, SR 57-61%）正面 PK。MapNav 的绝对 SR (39.7) 仍远低于这条线，论文没讨论是否 panoramic 信号能进一步推进
2. **依赖完美的 depth + pose**：Habitat 模拟器里 depth/pose 是 ground truth；real-world 用 D435i 也算近似 GT。在没有可靠 depth/pose 的场景（outdoor、移动设备）整个 pipeline 会崩。论文没做这方面 robustness 实验
3. **Mask2Former 训练分布限制**：semantic segmentation 的 vocabulary 决定了 ASM 能 ground 的物体范围。新场景里出现训练分布外的物体就丢失。Open-vocabulary segmentation（如 SAM2 + CLIP）应该能突破，作者未尝试
4. **Action decoder 太脆**：正则匹配字符串映射到 4 个离散动作，扩展到连续/复杂动作空间时不 work。这个设计选择把 model 锁在 low-level discrete action 框架里
5. **Real-world eval 样本量太小**：每场景 10 条指令，统计意义有限
6. **ASM 不含时序**：当前位置 + 历史轨迹是有的，但语义物体只保留 "当前观察过的"，没有 "什么时候第一次看到" 这类时序信息——对 instruction "after passing the kitchen, turn left" 这种依赖 episodic ordering 的指令有限

### 可信评估

#### Artifact 可获取性

- **代码**: 论文承诺会 release ASM 生成源码和数据集，但截至今日（2026-04-21）未发现官方 GitHub 仓库
- **模型权重**: 未说明，论文未提及 checkpoint release
- **训练细节**: 较完整——backbone 版本（SigLIP-so400m-patch14-384, Qwen2-7B-Instruct）、训练配置（8×A100, 30h, lr=1e-6, bf16, frozen vision encoder, 1 epoch）、数据组成（300k GT + 200k DAgger + 25k collision + 600k VL co-training）都披露了；collision recovery 数据的具体收集策略只在 Appendix 简述
- **数据集**: R2R/RxR 公开；论文承诺 release 1M step-wise pairs（含 RGB + ASM + instruction + action），实际未见发布

#### Claim 可验证性

- ✅ **"ASM 替代历史帧达到 NaVid 全帧水平"**：Table 1 同 backbone 对比清楚（MapNav w/ ASM 36.5 vs NaVid All Frames 37.4 SR on R2R）
- ✅ **"内存 O(1) 与 trajectory 长度无关"**：Table 3 数值 0.015 MB 是 ASM tensor 物理大小，可验证
- ✅ **"文本注释是关键"**：Table 4 ASM vs 纯语义图（36.5 vs 29.1 SR）+ Figure 9 attention 可视化双重支持
- ⚠️ **"Real-world 显著优于 NaVid"**：每场景 10 条指令样本太小，30% SR gap 背后只有 3 条 episode 差异，统计噪声可能压过 effect size
- ⚠️ **"零样本 ObjectNav 上接近 SOTA"**：MapNav 34.6 SR 略低于 ESC 的 35.5；说 "强泛化" 需要更多 baseline / multi-seed
- ⚠️ **"MapNav 适用于真实部署"**：依赖 D435i depth + 较准的 pose，没在弱 sensing 条件下 stress test
- ❌ **"will release source code and dataset to ensure reproducibility"**：截至今日未见 release，纯承诺不算技术 claim

### Notes

- **核心 take-away（reusable insight）**：VLM 处理空间信息时，**显式文本 grounding 比 channel 编码更高效**——这条不仅适用于 VLN，可能推广到其他 spatial reasoning 任务（manipulation 中的 affordance map、ObjectNav 的 frontier map 等）。把 channel 信息 "render 成图 + 写上字" 是和 VLM 预训练分布对齐的通用 trick
- **延伸方向**：ASM 目前是 2D top-down + 静态语义。可探索：
  - 3D ASM（带高度/楼层信息）→ multi-floor navigation
  - dynamic ASM（带物体最近观测时刻）→ 处理时序指令
  - open-vocabulary ASM（用 SAM + CLIP 替换 Mask2Former）→ 解锁未知物体
- **对 VLA 的启发**：当前 VLA（如 π0、OpenVLA）也面临 long-horizon memory 问题。"render 一张紧凑的 scene-state 图喂给 VLM" 这个范式可能比纯 frame-stacking 更 scalable
- **Methodological contrast with NaVid**：NaVid 把 "时序 = 多帧" 当 prior，MapNav 把 "时序 = 累积更新的空间表示" 当 prior。后者在 indoor static 场景更优，但前者对 dynamic / event-driven 任务可能更合适。两条路线本质是不同的 inductive bias
- **关于 "video → map" 的更广视角**：MapNav 隐含的是 "把视频压成一张 keyframe + state map"，与 SLAM、neural radiance field、3DGS 系列工作共享 spirit。它们的差异在于 fidelity vs interpretability 的 trade-off：MapNav 牺牲 photo-realism 换取 VLM-readable 的语义结构

### Rating

**分数**：2 - Frontier
**理由**：把 "VLN memory representation" 作为独立研究维度提出、并以 ASM 做出 ACL 2025 长文级别的 publishable insight（Strengths#1/#2），消融严谨（top-down/semantic/ASM 三档对比 + RGB+Depth 反直觉发现），是该细分方向当前必须 cite 的前沿工作；但 fair-comparison 仅限 monocular NaVid 流派、绝对 SR 仍远低于 panoramic+waypoint SOTA（Weaknesses#1），且承诺的代码/数据集/权重均未 release，没有成为 de facto baseline 的基础设施，尚不够格 Foundation；比 Archived 档次高在方法范式（"render 结构化记忆喂 VLM"）已在更广 embodied-AI / VLA 场景有延伸价值，外部信号上被 awesome-embodied-vla-vln 等综述列表收录。
