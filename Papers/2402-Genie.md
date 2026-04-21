---
title: "Genie: Generative Interactive Environments"
authors: [Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, Tim Rocktäschel]
institutes: [Google DeepMind, University of British Columbia]
date_publish: 2024-02-23
venue: ICML 2024
tags: [world-model, video-LLM]
paper: https://arxiv.org/abs/2402.15391
website: https://sites.google.com/view/genie-2024/
github:
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] Genie: Generative Interactive Environments
> - **核心**: 第一个 *generative interactive environment* — 从 200K+ 小时无标注 Internet 视频里以**完全无监督**方式学到一个 11B 参数、**逐帧可控**的世界模型，给一张图就能"踏进去"玩
> - **方法**: 三件套全部用 ST-transformer：(1) **VQ-VAE 视频 tokenizer (ST-ViViT)** 把视频压成离散 token；(2) **Latent Action Model (LAM)** 在像素上以 VQ-VAE 目标无监督学到 8 个离散 latent action 作为帧间瓶颈；(3) **MaskGIT 动力学模型**自回归预测下一帧 token。LAM 在推理时丢掉，只留 codebook 给用户/策略选 action
> - **结果**: 11B 模型在 2D 平台游戏上生成可控视频；同方法在 RT-1 robot 视频上学到一致的 latent action（FVD 82.7）；冻结 LAM 标注 expert video → BC 训出 policy 在未见 CoinRun 上仅用 200 个 expert 样本就追平 oracle BC
> - **Sources**: [paper](https://arxiv.org/abs/2402.15391) | [website](https://sites.google.com/view/genie-2024/)
> - **Rating**: 3 - Foundation（ICML 2024 Best Paper，DeepMind 旗舰，开辟了 unsupervised + frame-level controllable 的新象限，已成为 world model / VLA 方向必引基础）

**Key Takeaways:**
1. **Latent action 是把无标注视频做成可玩世界模型的核心 trick**: 用 VQ-VAE 在帧间学一个**容量极小的离散瓶颈** (|A|=8)，强迫它编码"这一帧到下一帧的最有意义变化"。这是把"视频 ≠ 经验数据 (没有 action)"这个基本障碍以 simple 的方式绕过去
2. **ST-transformer 是 scaling 的关键**: 空间注意力在 H×W 内做、时间注意力跨 T 帧做，主导项随帧数线性而非二次增长——这让 16 帧、160×90、11B 模型在 256 TPUv5p 上跑得动
3. **方法 scale gracefully**: 40M → 2.7B 训练 loss 单调下降；batch size 128→448 同样单调改善。这是 "simple, scalable" 的标准证据
4. **Latent action 学到的是语义而非像素细节**: 同一个 latent action 在不同 prompt（platformer、robotics、未见 OOD 图像）上诱导一致的语义 (down/up/left)，且能 zero-shot 迁移到 CoinRun 做 imitation learning。这是世界模型 → generalist agent 路线最关键的 enabling 证据
5. **Pixel-input LAM > token-input LAM**: ablation 显示在 token 上学 LAM 会丢掉对动作很关键的细粒度运动信息——controllability 显著掉。tokenizer 是为生成质量设计的，不是为动作信息保留设计的
6. **明确局限**: 16 帧 memory window；~1 FPS 推理；autoregressive 累积 hallucination

**Teaser. 一张提示图 → 一段可玩的轨迹。** 顶行是 Imagen2 生成图片做 prompt，底行是手绘 sketch 做 prompt，每一帧选一个 latent action 推进。
![](https://arxiv.org/html/2402.15391v1/figures/platformer_trajectories.png)

---

## 1. Motivation: 为什么是 "Generative Interactive Environments"

作者把 Genie 定位成一类**新模型**，明确区分于已有两类：

**Table 1. 三类模型的对比**

| Model Class | Training Data | Controllability |
| --- | --- | --- |
| World Models | Video + **Actions** | Frame-level |
| Video Models | Video + **Text** | Video-level |
| Genie | **Video only** | **Frame-level** |

> Dr. Li 注：这张表是全文最 sharp 的 framing。传统 world model 卡在"必须有 action 标注"，video model 卡在"只能给整段 prompt，不能逐帧操作"。Genie 的 contribution 不在 video quality 上 SOTA（FVD 还不及 SOTA video model），而在打开了**第三象限**：unsupervised + frame-level controllable。从 research taste 角度，这是 "important > publishable" 的典型——重新定义问题本身。

核心 insight：训练时**没有任何 action 标注**，但在推理时给出一个由 LAM 自动学到的 8 维 codebook 作为"键盘"，让用户/agent 逐帧选动作并展开。

---

## 2. Methodology

### 2.1 整体架构

**Figure 2. Genie 训练流程：tokenizer + LAM + dynamics model 三件套**
![](https://arxiv.org/html/2402.15391v1/figures/genie_architecture.png)

输入 T 帧视频 → video tokenizer 出离散 token $\bm{z}$ → LAM 在帧间推断 latent action $\tilde{\bm{a}}$ → dynamics model 拿 $(\bm{z}, \tilde{\bm{a}})$ 预测下一帧 token。

训练分两阶段：(1) 先训 video tokenizer；(2) 再 co-train LAM (从像素) 和 dynamics model (从 token)。

### 2.2 ST-Transformer：所有组件的共同 backbone

**Figure 3. ST-transformer 架构**
![](https://arxiv.org/html/2402.15391v1/figures/sttransformer.png)

每个 ST block 包含：
- **Spatial layer**: 在每个时间步内的 $1 \times H \times W$ tokens 上做 self-attention
- **Temporal layer**: 在 $T \times 1 \times 1$ tokens 上跨时间做 self-attention，**带 causal mask**
- **一个共享 FFW** (而非每层一个；省下来的参数让 attention scale)

**Why it matters**: 主导计算复杂度 (spatial attention) 随帧数**线性**而非二次增长。这是把架构推到 11B + 16 帧 + 200B+ token 训练规模能跑得动的根基。

### 2.3 Latent Action Model (LAM) — 全文最关键的组件

**Figure 4. LAM 架构：从未标注视频帧中无监督学动作**
![](https://arxiv.org/html/2402.15391v1/figures/LAM_architecture.png)

机制：
- **Encoder**: 输入 $\bm{x}_{1:t}$ 和 $x_{t+1}$，输出连续 latent action $\tilde{\bm{a}}_{1:t}$
- **Decoder**: 用 $\bm{x}_{1:t}$ 和 $\tilde{\bm{a}}_{1:t}$ 重建 $\hat{x}_{t+1}$
- **VQ-VAE 目标** + 极小 codebook：$|A| = 8$
- **关键**：decoder 只能看到 history 和 latent，要重建未来必须把"帧间最有意义的变化"压进 $\tilde{a}_t$
- **推理时**：除了 VQ codebook 整个 LAM 都丢掉，由用户输入 $a_t \in [0, 8)$ 取代

> 核心 trick 解析：信息瓶颈 (|A|=8) 强迫模型只保留"action-like"的语义变化。如果 codebook 大，latent 会退化成"未来帧的所有信息"——失去可控性的语义。这种 "make the bottleneck small enough that only causal abstraction survives" 的思路在 representation learning 里很经典 (β-VAE, IB)，作者把它精确地落地到 "frame-to-frame action" 的语义层级。

### 2.4 Video Tokenizer (ST-ViViT)

**Figure 5. Video tokenizer：带 ST-transformer 的 VQ-VAE**
![](https://arxiv.org/html/2402.15391v1/figures/tokenizer_architecture.png)

VQ-VAE 把 $T \times H \times W \times C$ 视频压成 $T \times D$ 离散 token。Encoder/decoder 都用 ST-transformer，所以 token $z_t$ 因果地包含 $\bm{x}_{1:t}$ 的信息。

对比 Phenaki 的 C-ViViT：C-ViViT 用 full space-time attention，cost 随帧数二次增长。ST-ViViT 主导项线性，**而且生成质量更好**（见 ablation）。

### 2.5 Dynamics Model

**Figure 6. Dynamics model：MaskGIT 风格的 token-level 帧预测**
![](https://arxiv.org/html/2402.15391v1/figures/dynamics_architecture.png)

Decoder-only MaskGIT transformer。输入 $(\bm{z}_{1:t-1}, \tilde{\bm{a}}_{1:t-1})$，预测 $\hat{z}_t$。训练时按 Bernoulli (rate ~ U[0.5, 1]) 随机 mask token，cross-entropy loss。

**实现细节里两个被低估的 trick**：
1. **Latent action 当作 additive embedding**（而非传统的 concatenate）。作者发现这样 controllability 更好。> ❓ 论文没解释 why；猜测是 additive 让 action 可以参与所有 spatial token 的更新而不是占位置
2. **`stopgrad` latent action**：dynamics model 的梯度不回传给 LAM，避免 dynamics 通过 LAM 走捷径泄漏未来信息

### 2.6 Inference

**Figure 7. 推理流程：prompt 帧 + 用户 latent action → 自回归生成**
![](https://arxiv.org/html/2402.15391v1/figures/genie_inference.png)

给 prompt 图 $x_1$ → tokenize 成 $z_1$ → 用户给 $a_1 \in [0, 8)$ → 通过 codebook 得 $\tilde{a}_1$ → dynamics 出 $z_2$ → tokenizer decoder 解出 $\hat{x}_2$ → 重复。每帧 25 步 MaskGIT，temperature=2。

---

## 3. Experiments

### 3.1 Datasets & Setup

- **Platformers**：从 Internet 收集 55M 16s 视频片段 → filter 后 6.8M 片段 (30K 小时)，10 FPS，160×90
- **Robotics**：RT-1 ~130K demos + 209K real robot 数据 + simulation 数据，**只用视频，丢掉 action 标签**
- **Final model**: 10.1B dynamics + 200M tokenizer + 300M LAM = **10.7B 参数**, batch 512, 125K steps, **256 TPUv5p**, 942B token
- **Metrics**: FVD (video quality) + 自定义 $\Delta_t \text{PSNR}$ (controllability)

**$\Delta_t\text{PSNR}$ 的定义**：

$$
\Delta_t\text{PSNR} = \text{PSNR}(x_t, \hat{x}_t) - \text{PSNR}(x_t, \hat{x}_t')
$$

其中 $\hat{x}_t$ 是用 GT 视频反推出的 latent action 生成的帧，$\hat{x}_t'$ 是用随机 action 生成的帧。差越大 → latent action 对生成的影响越显著 → controllability 越强。

> Dr. Li 注：这个 metric 构造很聪明——用 random action 作为 baseline，把"动作真的在控制视频" vs "动作只是装饰"区分开。但有一个隐忧：它只测"action 改变 outcome"，不测"action 改变的方向是否符合人类直觉"——一个把 8 个 action 全映射到"向左移动 N 个像素 (N=1..8)"的退化解也能拿很高 $\Delta_t\text{PSNR}$。

### 3.2 Scaling

**Figure 8. Scaling results**
![](https://arxiv.org/html/2402.15391v1/x2.png)

- 模型从 40M → 2.7B：训练 loss 单调下降
- Batch size 128 → 256 → 448 (1.9M → 6.6M tokens)：同样单调改善

> 这张图是 Genie 作为 "foundation world model" claim 的核心证据。但注意：scaling 的 y-轴是**训练 loss** (cross-entropy on tokens)，不是下游 controllability 或 video quality。Loss 降不等于 latent action 学得更好——这是论文没正面回答的问题。

### 3.3 Qualitative

**Figure 9. OOD prompt 上的 playability**
![](https://arxiv.org/html/2402.15391v1/x3.png)

用 Imagen2 生成图、手绘 sketch、真实照片做 prompt，模型都能产生 "game-like behavior"。

**Figure 10. 涌现的物理：变形物体**
![](https://arxiv.org/html/2402.15391v1/figures/chips.png)

模型从视频学到了如薯片袋的形变物理。

**Figure 11. 涌现的视差 (parallax)**
![](https://arxiv.org/html/2402.15391v1/figures/parallax_new.png)

前景比中景动得快，背景几乎不动——3D-aware 的副产品，纯从 2D 视频学出来。

**Figure 12. Robotics 上 latent action 的一致性**
![](https://arxiv.org/html/2402.15391v1/figures/action_grid_robotics.png)

三个不同起始帧，每列对同一个 latent action 重复 5 次。结果：同一个 action 在不同场景下表现出**一致的语义** (down / up / left)。这是 latent action **可解释、可迁移**的核心证据。

### 3.4 Training Agents on Unseen RL Environments — 最有 implication 的实验

**Figure 13. 用 Genie 在未见 RL 环境（CoinRun）上 rollout**
![](https://arxiv.org/html/2402.15391v1/x4.png)

**Figure 14. BC results — 仅用 200 expert sample 就追平 oracle BC**
![](https://arxiv.org/html/2402.15391v1/x5.png)

流程：
1. 用 frozen LAM 给 CoinRun expert video 打 latent action 标签
2. 训一个 policy: $p(\tilde{a} | o)$
3. 用 200 个带真实 action 的 expert 样本学 latent → real 的映射

结果：在 CoinRun (LAM 几乎肯定没见过) 上达到 oracle BC 水平。

> Dr. Li 注：这是全文最有冲击力的结果。Implication：**互联网视频 → latent action → 任何环境的 policy** 这条 pipeline 在原则上 work。这正是 VPT (用 IDM 标注 Minecraft 视频) 在做的事，但 Genie 不需要任何带 action 的数据来学 IDM。这对 generalist agent / VLA 路线意义深远——把视频从"被动观看材料"变成"主动控制信号源"。但仍要注意：CoinRun 是 2D 平台游戏，跟训练分布很近；跨到 robotics 或 3D 还需要更多证据。

### 3.5 Ablations

**Table 2. LAM 输入：Pixel vs Token**

|  | Dataset | #Params | FVD ↓ | $\Delta_t\text{PSNR}$ ↑ |
| --- | --- | --- | --- | --- |
| Token-input | Platformers | 2.3B | 38.8 | 1.33 |
| **Pixel-input (Genie)** | Platformers | 2.5B | 40.1 | **1.91** |
| Token-input | Robotics | 1B | 257.8 | 1.65 |
| **Pixel-input (Genie)** | Robotics | 1B | **136.4** | **2.07** |

结论：tokenizer 是为重建质量设计的，会丢掉运动细节。LAM 在 raw pixel 上学才能保留 controllability。

> 这个 ablation 的 lesson 很 general：**representation learning 的目标决定了 latent 保留什么**。把 LAM 接在 reconstruction-tokenizer 后面 = 让动作信息流经一个没有 incentive 保留它的瓶颈。simple insight, real consequence.

**Table 3. Tokenizer 架构：ST-ViViT vs ViT vs C-ViViT**

|  | #Params | Memory | FVD ↓ | $\Delta_t\text{PSNR}$ ↑ |
| --- | --- | --- | --- | --- |
| ViT | 230M | 0.3GB | 114.5 | 1.39 |
| C-ViViT (Phenaki) | 225M | 1.6GB | 272.7 | 1.37 |
| **ST-ViViT (ours)** | 205M | 0.9GB | **81.4** | **1.66** |

C-ViViT 用 full space-time attention，参数同但内存高 1.8×，反而**显著更差**（FVD 272.7）。作者归因于过拟合。ST-ViViT 兼顾时间因果与计算效率。

---

## 4. Limitations (作者承认的)

1. **Hallucination**：autoregressive 累积，会产出物理上不合理的未来
2. **16 帧 memory window**：长 horizon 一致性差，离开视野的物体回头时会变
3. **~1 FPS 推理**：远未达到交互级帧率
4. **训练数据 bias**：只在 2D platformer + robotics 上验证

---

## 关联工作

### 基于
- **MaskGIT** (Chang et al. 2022): dynamics model 直接采用其 masked token prediction + parallel decoding
- **VQ-VAE** (van den Oord et al. 2017): tokenizer 和 LAM 都用其离散瓶颈
- **ViViT / ST-Transformer**: ST-transformer 架构灵感来源
- **Phenaki / TECO / MaskViT**: video generation 的前置工作

### 对比
- **GAIA-1** (Hu et al. 2023): 自动驾驶 world model，**需要 action+text 标注**
- **UniSim** (Yang et al. 2023): 机器人 world model，**需要 action 标注**
- **VPT** (Baker et al. 2022): 用 IDM 标注 Internet 视频再 BC——核心区别：VPT 需要小批带 action 标注的数据训 IDM，**Genie 完全无需任何 action 标注**
- **PVG** (Menapace et al. 2021/2022): 域特定的 playable video generation；Genie 把它推广到 prompt-based 跨域生成

### 方法相关
- **[[2411-WorldModelSurvey|World Model 综述]]**: Genie 是 unsupervised 路线的代表性工作
- **[[DomainMaps/WorldModel|WorldModel domain map]]**: 在 vault 的 world-model 知识地图中 Genie 占据 "video-only training" 这一支
- **后续工作**: Genie 2/3 (DeepMind 后续)、World Models for Robotics (e.g. UniPi、AVDC)

---

## 论文点评

### Strengths

1. **Reframing 而非 incremental**：定义了 "Generative Interactive Environments" 这个新象限，让"unlabeled video → controllable simulator"成为一条独立路线。这是少见的 "important > publishable" 工作
2. **核心 insight 极其 simple**：8 个 latent action 的 VQ 瓶颈。任何能复述这个 insight 的人都能解释 Genie 为什么 work——这是好方法的标志
3. **Scaling 证据完整**：40M → 2.7B 单调下降，为 11B 主模型的投入做了 justification。架构 (ST-transformer 线性 cost) 也是为 scale 设计的
4. **Generality 的实证**：Platformers + Robotics + 未见 RL 环境的 BC 实验，三个截然不同的领域用同一套方法。这是 "generalizable" 的硬证据
5. **Ablation 信息量高**：Pixel-vs-token-input LAM 的 ablation 揭示了一个关于 representation 设计的 general lesson，远超 +X% 改进的范畴
6. **Latent action 一致性的可视化**（Figure 12）做得很硬核：同一个 action 在 3 个不同 robotics 起始帧上都解释为 "down/up/left"，是 unsupervised 学到 semantic action 的最强证据

### Weaknesses

1. **不开源**：DeepMind 没放代码、没放权重、没放数据 list。Reproducibility 接近零，社区只能等 open-genie 这类第三方实现。这削弱了 Genie 作为 "foundation model" 的实际价值
2. **$\Delta_t\text{PSNR}$ 不能完全反映 controllability 质量**：random baseline 可能把"任何非平凡的 action effect"都算作 controllable，无法区分"语义化 action" 和 "退化映射"
3. **没有正面评估 latent action 的语义可解释性**：Figure 12 是 anecdotal，没有量化 "8 个 action 是不是稳定对应到 N 种语义" 这种核心 claim
4. **CoinRun BC 实验的 baseline 选择保守**：只跟 oracle BC 和 random 比，没有跟其它 zero-action-label 方法（如 BCO、ILPO、VPT-style）比。"追平 oracle" 听起来很强，但 oracle BC 在 CoinRun 上本身就不是很强的 baseline
5. **scaling 曲线只看 train loss**：没有展示 model size → controllability、generalization、agent BC performance 的 scaling，留下 "loss 改善是否真的 transfer 到下游" 的疑问
6. **1 FPS / 16 帧 / hallucination 都是结构性限制**：要做真 interactive simulator 需要根本性的架构升级 (e.g. KV cache + streaming + memory module)，不是 scaling 能解的
7. **"foundation world model" claim 偏 marketing**：foundation 通常意味着 broad transfer + 强 zero-shot。Genie 在 platformer + robotics 内有效，但跨到 3D / first-person / 真实物理的迁移没有验证

### 可信评估

#### Artifact 可获取性

- **代码**: 未开源（社区有第三方 PyTorch 实现 myscience/open-genie，非官方）
- **模型权重**: 未发布
- **训练细节**: 主要超参 + 模型规模 + TPU 配置披露，**数据 filtering pipeline 高层描述**（见 Appendix B），但具体视频源、关键词列表未公开
- **数据集**: Platformers 数据集**未发布**（filter 自公开 Internet 视频）；Robotics 用的是公开的 RT-1 数据，但未发布 Genie 用的具体切分

#### Claim 可验证性

- ✅ **架构的有效性 (ST-transformer + ST-ViViT + LAM + MaskGIT dynamics)**：Table 2/3 ablation 证据扎实
- ✅ **Scaling 行为 (40M→2.7B 训练 loss 单调改善)**：Figure 8 直接证据
- ✅ **Latent action 在 Robotics 上的一致性**：Figure 12 视觉证据 + FVD 82.7 数字
- ✅ **CoinRun BC 实验 (200 sample 追平 oracle)**：Figure 14 量化结果
- ⚠️ **"Foundation world model" 的 generality**：在 2 个领域 (2D platformer + RT-1 robotics) 验证，但论文用 "foundation" 一词暗示更广 transfer，未给出 3D、first-person、真实物理的证据
- ⚠️ **"learns physics like deformation/parallax"**：Figure 10/11 是 anecdotal，没有量化 (e.g. 物理一致性测试集、跨场景成功率)
- ⚠️ **Scaling 推论到下游能力**：只 scale 了 train loss，没 scale 下游 controllability / agent performance
- ❌ **(无明显营销话术)**：作者用词比典型 industry release 克制；"first generative interactive environment" 这个 first claim 在表 1 的定义下成立

### Notes

**对我自己研究的 implication**：
1. **VLA**: Genie 路线提示 action label 的稀缺可以用 latent action 缓解。如果能在 robot 视频上学到一致的 latent action，则可以用海量未标注 robot/human video 预训 VLA
2. **World Model**: ST-transformer 的"空间内 + 时间跨"分解是 default 选择。MaskGIT-style parallel decoding 比 pure AR 在 video 上更实用
3. **Agentic RL**: latent action space 可以作为 RL 的 abstract action space，避开 raw action 维度灾难。但需要解决 "latent → real action" 的对齐

**疑问 / 待进一步思考**：
- ❓ 8 个 latent action 在 platformer 上够，但 Robotics 上够吗？文中没有显式的 |A| ablation
- ❓ Latent action 的语义在 OOD prompt 上是否仍稳定？Figure 9 是 anecdotal，没量化
- ❓ Additive vs concat latent action embedding 的差异机制是什么？这个发现可能 generalize 到其它 conditional generation
- ❓ Genie 学到的"物理"（变形、视差）有多少是真在建模物理 vs 在做 nearest-neighbor 检索训练集中的相似 motion？没有 controlled probe

### Rating

**分数**：3 - Foundation
**理由**：Genie 拿下 ICML 2024 Best Paper，是 DeepMind 世界模型路线的旗舰，并已派生出 Genie 2/3 系列后续工作；在笔记 Strengths 中写到它"定义了 Generative Interactive Environments 这个新象限"（reframing 而非 incremental），在 "关联工作" 中被定位为 [[2411-WorldModelSurvey|World Model 综述]] unsupervised 路线的代表——做 world model / VLA / video-as-data 方向必引。与 2 - Frontier 的区别在于它不是"当前 SOTA/baseline"，而是已经**重新定义了子方向**的奠基工作；其 latent action VQ 瓶颈的 insight 在 trickery 层面跨方向复用价值高，非普通前沿工作可替代。
