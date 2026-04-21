---
title: "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation"
authors:
  - Jiazhao Zhang
  - Kunyu Wang
  - Rongtao Xu
  - Gengze Zhou
  - Yicong Hong
  - Xiaomeng Fang
  - Qi Wu
  - Zhizheng Zhang
  - He Wang
institutes:
  - Peking University
  - Beijing Academy of Artificial Intelligence (BAAI)
  - CASIA
  - University of Adelaide
  - Australian National University
  - Galbot
date_publish: 2024-02-24
venue: RSS 2024
tags: [VLN, VLM, navigation]
paper: https://arxiv.org/abs/2402.15852
website: https://pku-epic.github.io/NaVid/
github: https://github.com/jzhzhang/NaVid-VLN-CE
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation
> - **核心**: 把一个视频 VLM (LLaMA-VID) 改造成 VLN-CE 的端到端 agent，**只用单目 RGB 视频**预测下一步低层动作（含距离/角度参数），不需要 depth、odometry 或 map
> - **方法**: 当前帧 64 个 instruction-agnostic visual tokens + Q-Former instruction-queried token；历史帧每帧仅 4 tokens；用 `<HIS>/<OBS>/<NAV>` 特殊 token 切分；510k VLN-CE 仿真数据（含 180k DAgger）+ 10k instruction reasoning + 763k web video-caption co-train Vicuna-7B
> - **结果**: VLN-CE R2R val-unseen SPL 35.9（仅 RGB，超过用 depth+odo 的 WS-MGMap 的 34.3）；R2R→RxR cross-dataset SPL 21.2（前 SOTA $A^2$Nav 仅 6.3）；Turtlebot4 真机 4 个室内场景 200 条指令简单指令 ~85% SR、复杂指令 ~47% SR
> - **Sources**: [paper](https://arxiv.org/abs/2402.15852) | [website](https://pku-epic.github.io/NaVid/) | [github](https://github.com/jzhzhang/NaVid-VLN-CE)
> - **Rating**: 3 - Foundation（首个 video-based VLM for VLN-CE，RGB-only 跑通 sim2real，Table IV 把 "video > text/map history" 钉死，已被 StreamVLN / VLN-R1 / Uni-NaVid 等后续工作作为必引 baseline）

**Key Takeaways:**
1. **首个 video-based VLM for VLN-CE**: 把历史 RGB 帧直接编码为时空视觉 token 序列，规避了之前 LLM-based VLN 把历史压缩为文本/2D map 造成的视觉信息损失（SPL 35.9 vs 文本历史 20.8 vs map+text 8.97）
2. **不对称 token 预算**: current frame 64 token、history frame 4 token——既保留当前帧的几何细节用于动作量化预测，又把历史压缩到可承受的 context（消融显示 4 tokens/frame 是 SR/latency 的甜点；16 tokens 仅 +1.6% SR 但 +122% latency）
3. **直接输出语言形式的低层动作**: "FORWARD 75 cm" / "TURN-LEFT 30 deg" 而不是 waypoint 坐标——把连续坐标改成 waypoint 预测变体在消融里直接 collapse 到 0% SR，说明 VLM 不擅长 regress 连续数值
4. **co-training 是关键**: 去掉 web video-caption co-training 后 SPL 从 35.9 掉到 23.6（-34%），说明大模型的 generalization 主要来自 anti-forgetting 而非 navigation 数据本身
5. **Sim2Real 的 RGB-only 路线**: 仅 R2R 仿真训练直接 zero-shot 到 Turtlebot4 真机，绕过了 depth/odometry 的 sim2real gap——这是把 VLM "通用知识" 用作 domain bridge 的一个 existence proof

**Teaser. NaVid 真机 demo——给定一句人类指令，仅用机器人单目 RGB 视频流预测下一步动作。**

![](https://arxiv.org/html/2402.15852v6/x1.png)

---

## 问题与动机

VLN-CE（vision-and-language navigation in continuous environments）的核心痛点是 **generalization**：训练到测试场景的迁移、以及 sim-to-real 的迁移。已有 SOTA 方法（WS-MGMap、ETPNav 等）依赖 depth + odometry + 2D/语义 map，每一类输入都引入 sim-to-real gap：

- **Depth**: 真机的深度噪声分布和 Habitat 渲染差异巨大
- **Odometry**: 累积漂移
- **Map**: 在线建图依赖 SLAM/LiDAR，pipeline 复杂

NaVid 的赌注是：**把 VLN agent 的输入收缩到只有单目 RGB 视频**，用 VLM 的通用知识弥补信息缺失，借此跨过 sim2real gap。这是一个偏 "去工程化" 的路线，对应 RT-2 在 manipulation 上做的同类型尝试。

> ❓ 这个赌注成立的前提是 VLM 的 visual prior 足够强能推断深度/几何关系。在密集障碍、低纹理场景下是否成立？论文的真机环境（meeting room/office/lab/lounge）都是 well-textured 室内，没有 stress test 极端场景。

## 方法

### 整体架构

NaVid 在 LLaMA-VID 基础上做 task-specific 改造，由 vision encoder（EVA-CLIP）、Q-Former 风格的 query generator、两个 cross-modality projector 和 Vicuna-7B LLM 组成。

**Figure 2. NaVid 整体架构。** RGB 视频帧 $\{x_0, \cdots, x_t\}$ + 人类指令 $\mathcal{I}$ 经 observation encoder 得到两类 token：橙色的 instruction-queried token（每帧 1 个）和蓝色的 instruction-agnostic token（history 帧 4 个、current 帧 64 个）。用 `<HIS>/<OBS>/<NAV>` 切分后送 Vicuna-7B 输出下一步动作。

![](https://arxiv.org/html/2402.15852v6/x2.png)

### Observation Encoding

每帧 $x_t$ 经 EVA-CLIP 得到 $\mathbf{X}_t \in \mathbb{R}^{N_x \times C}$（$N_x = 256$ patches）。然后切两路：

**Instruction-queried token**（每帧 1 个）：用 Q-Former 风格的 query generator 在视觉 patch 和 instruction embedding $\mathbf{I}$ 之间做 cross-attention，得到 instruction-aware query $\mathbf{Q}_t \in \mathbb{R}^{M \times C}$：

$$
\mathbf{Q}_t = G_Q(\mathbf{X}_t, \mathbf{I})
$$

再做 cross-attention 并 pool：

$$
\mathbf{E}^Q_t = P_Q(\text{Pool}(\text{Softmax}(\mathbf{Q}_t \mathbf{X}_t^T) \mathbf{X}_t))
$$

最终得到 $\mathbf{E}^Q_t \in \mathbb{R}^{1 \times C}$。

**Instruction-agnostic token**（每帧 $N_v$ 个）：直接对 $\mathbf{X}_t$ 做 grid pooling：

$$
\mathbf{E}^V_t = P_V(\text{GridPool}(\mathbf{X}_t))
$$

**关键设计**：$N_v$ 在 history 帧（4）和 current 帧（64）取不同值。history 提供 "我从哪来" 的语境，可以粗；current 直接决定 "下一步走多远转多少度"，需要保留几何细节。LLaMA-VID 原版每帧只用 2 个 token——直接用会丢失动作量化所需的几何信息（消融 Table VIII 验证：1 token/frame 时 SR 仅 23.9 vs 4 token 时 37.4）。

### Token 拼接与动作输出

输入格式：

```
<HIS>{history_frames}</HIS><OBS>{current_frame}</OBS><NAV>{instruction_content}
```

LLM 以语言形式输出动作，例如 `"FORWARD 75 cm"`、`"TURN-LEFT 30 degrees"`、`"STOP"`。用正则匹配解析——简单匹配在 R2R val-unseen 上达到 100% 有效动作率。

> 💡 **设计 insight**：让 LLM 输出离散 token 形式的动作（含数值短语）比直接 regress 连续坐标好很多——消融里 Waypoints prediction 变体直接 0% SR。这印证了 LLM/VLM 在分类/分桶式输出上远比 regression 鲁棒，与 [[2307-RT2|RT-2]] 的 action tokenization 结论一致。

### 训练数据

**Action planning 数据 (510k step-wise samples)**：
- **Oracle**: 从 VLN-CE R2R 训练 split（61 个 MP3D 室内场景）的 oracle trajectory 拿到 320k 样本
- **Non-oracle (DAgger-style)**: 用第一阶段训练的 NaVid 在 VLN-CE 环境里 rollout，再补 180k 样本——避免只见过 oracle trajectory 的脆弱性

**Instruction reasoning auxiliary task (10k)**：给定 video trajectory，反推对应指令（trajectory captioning）。Loss 共享同一个 prompt 模板，仅替换 instruction 槽位。

**Web video-caption (763k)**：直接复用 LLaMA-VID 的预训练数据，1 FPS 采样，防止 forgetting。

**训练配置**：24 张 A100，~28h，672 GPU·hours，1 epoch；只优化 LLaMA 和 text encoder 的 trainable parameter（vision encoder/Q-Former/BERT 冻结）。

## 实验

### VLN-CE R2R val-unseen（cross-split generalization）

**Table I. VLN-CE R2R Val-Unseen SOTA 比较。** NaVid 是唯一一个仅用 single-RGB（无 depth、无 odometry、无 panorama）的方法，但在 SPL 上达到 SOTA。

| Method | Pan. | S.RGB | Depth | Odo. | TL | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|:-:|:-:|:-:|:-:|---|---|---|---|---|
| [[2304-ETPNav\|ETPNav]] (waypoint) | ✓ | | ✓ | ✓ | 11.99 | 4.71 | 65.0 | 57.0 | 49.0 |
| GridMM (waypoint) | ✓ | | ✓ | ✓ | 13.36 | 5.11 | 61.0 | 49.0 | 41.0 |
| HAMT (waypoint, +data) | ✓ | | ✓ | ✓ | – | 4.80 | – | 55.0 | 51.0 |
| WS-MGMap (low-level) | | ✓ | ✓ | ✓ | 10.00 | 6.28 | 47.6 | 38.9 | 34.3 |
| Seq2Seq (low-level) | | ✓ | ✓ | | 9.30 | 7.77 | 37.0 | 25.0 | 22.0 |
| CMA (low-level) | | ✓ | ✓ | | 8.64 | 7.37 | 40.0 | 32.0 | 30.0 |
| RGB-Seq2Seq | | ✓ | | | 4.86 | 10.1 | 8.10 | 0.00 | 0.00 |
| RGB-CMA | | ✓ | | | 6.28 | 9.55 | 10.8 | 5.00 | 4.43 |
| **NaVid** | | ✓ | | | 7.63 | 5.47 | 49.1 | 37.4 | **35.9** |

公平的对比是 NaVid vs WS-MGMap（同样 low-level action space，最强 baseline）：NaVid 在 SPL 上 +1.6%（35.9 vs 34.3），同时去掉了 depth + odometry。注意 [[2304-ETPNav|ETPNav]]/HAMT 等用了 waypoint predictor + panorama RGB + depth，是更宽松的 setting；NaVid 的 SR 比 [[2304-ETPNav|ETPNav]] 低（37.4 vs 57.0）但用的输入信息少得多。

> ❓ **解读保留**: 论文把自己 frame 成 "SOTA"，但严格说在 SR 上仍输给用更多 modality 的 panorama+waypoint 方法。"SOTA-level with RGB-only" 是更准确的表述。

### R2R → RxR cross-dataset

**Table II. cross-dataset 泛化。** R2R 训练，RxR val-unseen 测试。

| Method | S.RGB | Depth | Odo. | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|:-:|:-:|:-:|---|---|---|---|
| WS-MGMap | ✓ | ✓ | ✓ | 9.83 | 29.8 | 15.0 | 12.1 |
| RGB-CMA | ✓ | | | 9.55 | 14.8 | 0.0 | 0.0 |
| $A^2$Nav (zero-shot, GPT planner) | ✓ | | | – | – | 16.8 | 6.3 |
| **NaVid** | ✓ | | | 8.41 | 34.5 | 23.8 | **21.2** |

cross-dataset 提升相比同 split 大得多——SPL 从 zero-shot SOTA $A^2$Nav 的 6.3 提到 21.2（+236%）。RxR 指令更长更细粒度，是更适合检验 generalization 的 setting，这里的 gap 比 R2R 内部更说服人。

### History representation 消融

**Table IV. 不同历史表示的对比。** 同样的 NaVid 框架，仅替换历史的编码方式。

| Method | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| (A) Text-based (LLaVA caption + GPT-4 summarize) | 8.82 | 0.0 | 0.0 | 0.0 |
| (B) Map-text (top-down 2D map + text) | 7.12 | 9.51 | 9.13 | 8.97 |
| (C) Ego-view-text (current image + text history) | 8.85 | 35.5 | 23.5 | 20.8 |
| **Video-based (NaVid)** | 5.47 | 49.1 | 37.4 | **35.9** |

这是 paper 最 informative 的 table：把历史压成 text 几乎无法学（0% SPL）；加 top-down map 也不行（9% SPL）；只用当前帧 + text history 能学但远不如全程 video（20.8 vs 35.9）。这给 [[2305-NavGPT|NavGPT]] 一类 LLM-as-planner 的方法泼了冷水——把视觉历史交给 LLM 用文本看，**信息瓶颈太严重**。

### 与 LLM/VLM baseline 比较

**Table III. 在 R2R val-unseen 100 episode sub-split 上对比通用大模型。**

| Method | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| GPT-4V (in-context prompt) | 11.4 | 10.00 | 5.00 | 3.11 |
| Emu | – | – | – | – |
| LLaVA | – | – | – | – |
| LLaMA-VID | – | – | – | – |
| LLaVA-Nav (co-tuned) | 7.82 | 14.0 | 10.0 | 9.43 |
| LLaMA-VID-Nav (co-tuned) | 8.73 | 40.0 | 29.0 | 27.5 |
| **NaVid** | 5.52 | 45.0 | 38.0 | **35.4** |

未微调的 Emu / LLaVA / LLaMA-VID 直接输出环境描述而非动作，无法导航。GPT-4V 配 in-context prompt 能给出有效动作但 SR 仅 5%。即使是 co-tuned 的 LLaMA-VID-Nav（同样 backbone）也比 NaVid 差 8 个 SPL，说明 NaVid 的 task-specific 设计（不对称 token 预算 + special token 分隔）是有效的。

### 真机 Sim2Real

**Table VI. 4 个真实场景的真机评测（Turtlebot4 + Kinect DK）。** 每个场景 25 条 simple + 25 条 complex 指令，共 200 个 episode。

| Method | Avg Simple SR | Avg Complex SR |
|---|---|---|
| Seq2Seq | 1.0% | 0.0% |
| CMA | 3.0% | 0.0% |
| WS-MGMap | 51.0% | 22.0% |
| **NaVid** | **85.0%** | **47.0%** |

end-to-end 的 Seq2Seq/CMA 几乎完全失败——这是已知的 sim2real failure mode（depth domain gap）。WS-MGMap 用了 LiDAR + Nav2 在线建图所以还能工作。NaVid 仅 RGB 就达到 simple 85% / complex 47% SR。**这是 paper 最强的证据**：RGB-only 的 VLM 路线在真机上是 viable 的。

> ⚠️ **caveat**: real-world 评测里 WS-MGMap 用了 LiDAR + Nav2，而 NaVid 不需要——这不是完全 apples-to-apples。但反过来说，NaVid 用更少的传感器达到更高的 SR，这个 trade-off 本身就是 message。

**Video. 简单指令真机 demo——按相似但不同的 stop 条件分别完成。**
<video src="https://pku-epic.github.io/NaVid/static/videos/teaser/simple_instruction_1_compressed.mp4" controls muted playsinline width="720"></video>

**Video. 复杂多步指令真机 demo——多个 simple instruction 顺序组合。**
<video src="https://pku-epic.github.io/NaVid/static/videos/teaser/complex_instruction_1_compressed.mp4" controls muted playsinline width="720"></video>

### Ablation 关键结论

**Table VII. 训练策略与架构 ablation。**

| Variant | TL | NE↓ | OS↑ | SR↑ | SPL↑ |
|---|---|---|---|---|---|
| No co-training (无 web video-caption) | 6.76 | 6.33 | 30.8 | 24.7 | 23.6 |
| No instruction reasoning | 9.46 | 6.51 | 46.7 | 31.1 | 29.1 |
| No non-oracle (DAgger) | 8.73 | 5.82 | 46.4 | 34.2 | 32.0 |
| No `<NAV>` | 8.71 | 5.62 | 48.1 | 35.9 | 33.5 |
| No `<HIS>` & `<OBS>` | 8.45 | 5.56 | 46.2 | 35.7 | 33.4 |
| Waypoints prediction | 13.65 | 10.8 | 11.5 | 0.00 | 0.00 |
| **Full** | 7.63 | 5.47 | 49.1 | 37.4 | **35.9** |

最大的两个杠杆：
1. **Web video-caption co-training**：去掉 -12.3 SPL（-34%）。说明 NaVid 的 generalization 大头来自 anti-forgetting + 通用 video understanding 的迁移，而不是 navigation 数据本身。这与 RT-2 的 "internet-scale web data is the secret sauce" 结论同构。
2. **动作输出形式**：waypoint regression 直接 collapse。再次印证 LLM 不适合 regress 连续值。

instruction reasoning aux task 也贡献 -6.8 SPL；DAgger non-oracle 数据贡献 -3.9 SPL。

**Table VIII. Visual token 数量 ablation。**

| Tokens per frame | NE↓ | OS↑ | SR↑ | SPL↑ | Avg. Time↓ |
|---|---|---|---|---|---|
| 1 | 8.13 | 30.4 | 23.9 | 20.5 | 0.87s |
| **4** | **5.47** | **49.1** | **37.4** | **35.9** | **1.22s** |
| 16 | 5.38 | 51.1 | 38.0 | 36.1 | 2.72s |

4 token/frame 是 SR 与 latency 的甜点：1→4 SR +56.4% 但只 +40% latency；4→16 SR 仅 +1.6% 但 +122% latency。

## 局限

- **Latency**: 1.2-1.5s/action（A100 inference）。对真机闭环控制太慢，实际只能慢速 navigate。论文提到 quantization 和 action chunk 是后续方向
- **Long-horizon**: 超过 90 步的指令性能下降，受制于 LLM 上下文长度和长视频标注数据稀缺
- **任务范围**: 只验证 VLN-CE，未涉及 mobile manipulation 或 object-goal navigation。续作 Uni-NaVid 走的就是任务统一这条路

---

## 关联工作

### 基于
- **LLaMA-VID** (Li et al. 2023): NaVid 的 backbone，提供 video-based VLM 架构与 instruction-queried + instruction-agnostic dual-token 设计
- **Vicuna-7B**: LLM backbone
- **EVA-CLIP**: 视觉编码器
- **Q-Former** (BLIP-2): instruction-queried token 的生成机制
- **VLN-CE** (Krantz et al. 2020): 把 VLN 从离散 graph 推进到连续环境的 benchmark formulation；NaVid 直接在其上评测
- **DAgger**: non-oracle 轨迹收集的 imitation learning 范式

### 对比
- **WS-MGMap**: VLN-CE 上最强的 low-level action baseline，使用 multi-granularity semantic map；NaVid 在 SPL 上 +1.6% 且无需 depth/odometry
- **[[2304-ETPNav|ETPNav]] / GridMM / HAMT**: panorama + depth + waypoint predictor 的方法，SR 更高但输入要求多得多
- **[[2305-NavGPT|NavGPT]]** / **$A^2$Nav**: 把 LLM 当 planner、用文本表示历史的方法。Table IV 直接证伪了这条路（text-based history 0% SR）
- **LM-Nav**: 离散环境 + off-the-shelf foundation model 的 baseline，NaVid 在 RxR cross-split 上大幅胜出（SR 23.5 vs 8-10）
- **[[2307-RT2|RT-2]]**: VLM 转 robot action 的同时期 manipulation 工作；NaVid 是 navigation 域的对应物，思路高度同构（pretrained VLM + action tokenization + co-training with web data）

### 后续 / 相关
- **Uni-NaVid** (RSS 2025): 同作者续作，把 NaVid 扩展到统一多种 embodied navigation 任务（VLN + ObjectNav + ImageGoal + EQA），证明 video-VLM 路线可 scale 到 task family
- **[[2507-StreamVLN|StreamVLN]]**: 用 SlowFast context modeling 解决 NaVid 的 long-context / latency 问题
- **[[2506-VLNR1|VLN-R1]]**: 把 RL 引入 video-based VLN
- **[[2502-VLNav|VL-Nav]]** / **[[2412-LHVLN|LH-VLN]]**: 后续 long-horizon VLN 方向

---

## 论文点评

### Strengths

1. **极简设计 + 强结果**：把整个 VLN pipeline 收缩到 "RGB video → VLM → action token"，去掉所有传统的 SLAM/depth/map 组件，并在 RGB-only 这个最严格的 setting 下达到 SOTA-level 性能。这是 simple, scalable 路线的成功例子
2. **History representation 消融做得透**：Table IV 同框架对比 text/map+text/current+text/video 四种历史表示，把 "为什么需要 video-based" 的问题钉死。这种控制变量的对比比单纯 SOTA 数字更有 insight
3. **Sim2Real 是真实评测**：4 个场景 200 个 episode 不是 cherry-picked，且和 WS-MGMap（用 LiDAR）做了对比，体现了 RGB-only 路线的工程价值
4. **不对称 token 预算**：current 64 + history 4 这个设计精巧但不复杂，且消融数据支持（4 是 SR/latency 甜点）。这是可以复用到其他 video-based 决策任务的 design pattern
5. **Co-training 数据配比的 ablation**：揭示了 web video data 是 generalization 的主导因素，对后续做 video-based VLA 的工作有方法论意义

### Weaknesses

1. **"SOTA" 框架过宽**：在 SR 上仍输 ETPNav/HAMT 等 panorama + depth + waypoint 的方法。论文应明确说 "SOTA among RGB-only / low-level action methods"，而不是隐含全方位 SOTA
2. **Latency 严重**：1.2-1.5s/action 对 mobile robot 是 prohibitive。Turtlebot4 demo 视频里机器人移动速度极慢，这是 dirty secret
3. **DAgger 数据收集只在 R2R train split 内做**：导致 post-DAgger 阶段提升微弱（论文自己承认）。真正能 scale 的应该是在更多样 scene/instruction 上 rollout，但这就要求构建新仿真环境
4. **真机指令分布与 R2R 不同分布**：作者自己设计了 200 条 simple/complex 指令，没有公开 instruction 列表，存在 cherry-pick 风险
5. **没有和 GPT-4V + 更强 prompt engineering 的强 baseline 充分对比**：Table III 里 GPT-4V 5% SR 用的是相对简单的 in-context prompt，没有用 chain-of-thought / scratchpad / specialized scaffolding（如 NavGPT-v2 风格）。这让 "VLM tuning > prompting" 的结论略显站不稳
6. **VLN-CE R2R/RxR 仍是 photo-realistic 但有限的 benchmark**：sim 上的 SOTA 含金量不如 R2R-CE 之外的更大规模数据（如 Habitat-Matterport）
7. **没有 failure case 分析**：复杂指令 47% 真机 SR 意味着一半失败，failure mode 是什么？指令理解失败、grounding 失败、还是 action 量化错？没有讨论

### 可信评估

#### Artifact 可获取性
- **代码**: inference + evaluation 代码已开源（jzhzhang/NaVid-VLN-CE，含 RSS 2024 NaVid 和 RSS 2025 Uni-NaVid 的 VLN-CE 评测代码）。**训练代码未在该 repo 提供**（README 只描述 evaluation）
- **模型权重**: README 提供了 NaVid checkpoint 下载（基于 Vicuna-7B + EVA-CLIP）
- **训练细节**: 仅高层描述（24 A100 × 28h、1 epoch、LLaMA + text encoder trainable）；具体的 lr / batch size / co-training data ratio 未在正文披露
- **数据集**: VLN-CE R2R/RxR 公开（基于 MP3D）；Web video-caption 数据复用 LLaMA-VID 公开数据；DAgger non-oracle 180k samples 未明确说是否随代码发布

#### Claim 可验证性
- ✅ **R2R/RxR 仿真 SOTA 数字**：可在 VLN-CE benchmark 用公开 evaluation code 复现，其他研究组（如 StreamVLN、VLN-R1）已普遍把 NaVid 作为 baseline 引用
- ✅ **video > text/map 历史表示**：Table IV 同框架对比，消融严格
- ⚠️ **"SOTA-level navigation performance"**：SR 仍输给 panorama+waypoint+depth 的方法，"SOTA" 限定条件是 "low-level action + RGB-only"，这一限定在 abstract 里没有显化
- ⚠️ **"66% real-world success rate"**（abstract 中）：这是 200 个 episode 的整体平均，simple/complex 不平衡（simple 85% / complex 47%）。"66%" 是把两类 50:50 加权得到，但用户体验上 complex 才是 challenge
- ⚠️ **Sim2Real 与 WS-MGMap 对比**：NaVid 不用 LiDAR、WS-MGMap 用 LiDAR + Nav2，传感器配置不同，对比口径需要 caveat（论文未充分强调）
- ✅ **Token 预算 ablation**：Table VIII 数据完整，4 vs 16 token 的 trade-off 量化清晰

### Notes

- **改变了我对 VLN history representation 的判断**: Table IV 是把 "video > text history" 钉死的实验。后续做任何 video-based decision making 的工作都应该把 NaVid 当 baseline 或参考点
- **不对称 token 预算 + special token 分隔**是可复用的 design pattern，可迁移到 video-based VLA、computer-use agent 等需要历史上下文的场景
- **Co-training with web video data 是 generalization 的关键**——这与 [[2307-RT2|RT-2]] 同构，是关于 "VLM-to-X 怎么不丧失 generalization" 的重要数据点

- **与 [[2307-RT2|RT-2]] 的对比值得深挖**:
  - 同：pretrained VLM + action as language tokens + co-train with web data + 真机部署
  - 异：[[2307-RT2|RT-2]] 的 action 是 7-DoF + gripper 的离散 bin，NaVid 是 navigation primitive (FORWARD/TURN-LEFT/TURN-RIGHT/STOP) + 数值参数
  - NaVid 的 token 预算管理（current 64 / history 4）比 [[2307-RT2|RT-2]] 更精细，因为 navigation 必须显式建模 history（manipulation 通常 single-step 或短 horizon）
  - **insight**: VLM-to-action 的成功路线在两个域上都 converge 到 "tokenize action + retain web pretraining"，这是一个 strong inductive prior

- **Latency 是真问题**: 1.5s/action × 100 step horizon = 150s 仅推理，加上机器人执行时间，单条指令 minutes 级。这是 NaVid 在工业部署上的硬伤。Uni-NaVid / [[2507-StreamVLN|StreamVLN]] 都在尝试解决这个问题——可以 follow 这条线追踪进展

- **可能的 follow-up 方向**:
  - 把 NaVid 的 dual-token 设计迁移到 GUI agent / computer-use agent（同样需要 long-horizon screen history + current screen）
  - 把 video-based VLM 的 grounding 能力用在 spatial reasoning 任务上（不是 action prediction，而是 spatial QA）
  - 在 NaVid 的 setup 上加 RL fine-tuning（[[2506-VLNR1|VLN-R1]] 已经走这条路）

- **❓ 待澄清**: paper 中 "Web data 763k" 具体是什么数据？是 LLaMA-VID 的 video instruction tuning 数据还是更广泛的 web video-caption？需要查 LLaMA-VID 原文确认。这个 detail 影响其他人复现 co-training 配比的可行性

### Rating

**分数**：3 - Foundation
**理由**：NaVid 是首个把 video-based VLM 作为 VLN-CE 端到端 agent 的工作，其 Table IV 的 history representation 消融（video > current+text 20.8 > map+text 8.97 > text-only 0）为整个 video-based navigation 路线提供了 foundational evidence，已被 [[2507-StreamVLN|StreamVLN]]、[[2506-VLNR1|VLN-R1]]、Uni-NaVid 等主要后续工作作为必引 baseline 和范式起点；相较 2 - Frontier，它不只是一次 SOTA，更是定义了 "RGB-only + VLM + action tokenization" 这条路线的方法论模板（与 [[2307-RT2|RT-2]] 同构），且提供了可迁移的 design pattern（不对称 token 预算、special-token 分隔、web video co-training）。
