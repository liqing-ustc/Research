---
title: "Efficient-VLN: A Training-Efficient Vision-Language Navigation Model"
authors: [Duo Zheng, Shijia Huang, Yanyang Li, Liwei Wang]
institutes: [The Chinese University of Hong Kong]
date_publish: 2025-12-12
venue: arXiv 2512.10310
tags: [VLN, navigation, video-LLM]
paper: https://arxiv.org/abs/2512.10310
website: https://lavi-lab.github.io/Efficient-VLN
github:
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] Efficient-VLN: A Training-Efficient Vision-Language Navigation Model
> - **核心**: 用两类高效 memory 表征 + 动态 DAgger mixing ratio，把 MLLM-based VLN 的训练成本压到 282 H800 GPU·h，同时刷到 R2R-CE / RxR-CE SOTA。
> - **方法**: (1) Progressive memory：仿"人类遗忘"，对近期帧低压缩、远期帧逐级 4× 下采样，token 总量收敛到 KS/3；(2) Recursive memory：把可学习 sentinel tokens 的 KV cache 作为 memory state 跨步传递；(3) Dynamic mixed policy：DAgger 中 oracle 概率 β=1−α^(t/T) 随步数递增，前期靠 learned policy 制造 compounding error，后期靠 oracle 兜底完成任务。
> - **结果**: R2R-CE 64.2% SR、RxR-CE 67.0% SR；训练 282 H800·h，相比 NavFoM (4032 H100·h) / StreamVLN (1500 A100·h) 大幅降本；DAgger #Train Step 从 128 降到 82（−36%）。
> - **Sources**: [paper](https://arxiv.org/abs/2512.10310) | [website](https://lavi-lab.github.io/Efficient-VLN)
> - **Rating**: 2 - Frontier（SR-vs-cost 帕累托前沿上的 MLLM-based VLN SOTA，progressive memory + dynamic DAgger 是值得借鉴的 building block，但尚未成为方向奠基工作）

**Key Takeaways:**
1. **Recency-aware memory 比 uniform compression 强**: 对所有帧统一压到 4 token 的 NaVid 式做法在 R2R/RxR 上都吃亏；保留 recent K=3 帧高分辨率、对更早帧逐级 4× 下采样后，R2R SR +2.6, RxR SR +3.0。
2. **Recursive (state-based) memory 在长 trajectory 上崩**: 64-token KV-cache 作为 state 在 R2R-CE (短 traj) 与 progressive 持平甚至更优，但 RxR-CE (长 traj) 上 SR 降了 7+。state compression 在长 horizon 上仍是开放问题。
3. **DAgger β 是个被忽略的成本旋钮**: 固定 β=0.25 比 β=0.75 SR 高 8.8 pt，但 trajectory 长度近乎翻倍（66→128）。Curriculum 式动态 β（前期 explore 后期 oracle）能拿到 β=0.25 的 SR，trajectory 只增加 16 步。
4. **3D geometry 注入 RGB-only VLN 是几乎免费的午餐**: StreamVGGT/Stream3R 提取的 latent geometry token 与 2D feature 元素相加，R2R-CE SR +3.6 / +4.8，无需 depth sensor。

**Teaser. Efficient-VLN 在 R2R-CE 上以 282 H800·h 训练成本拿到 64.2% SR，把 NavFoM/StreamVLN 等需要 1.5K–4K 卡时的方法甩在 SR-vs-cost 帕累托前沿之外。**

![](https://arxiv.org/html/2512.10310v1/x1.png)

---

## 1. 问题定义与 motivation

VLN-CE：给定语言指令 $I$ 和 RGB 观测序列 $\{v_1, \ldots, v_t\}$，模型在每步预测离散动作（forward / left / right / stop）。沿用 UniNavid / StreamVLN 的设定，模型一次预测下 4 个动作。

作者定位 MLLM-based VLN 训练贵的两个主因：

1. **Token 数二次爆炸**：StreamVLN 单步用到 196×16=3136 visual tokens，attention 二次开销在 long horizon 上不可接受。
2. **DAgger exploration-efficiency 矛盾**：低 β（多 learned policy 行动）→ 更多 error-recovery 数据 → 更长 trajectory → 训练/推理都贵；高 β → trajectory 短但 error-recovery 数据贫瘠。

> ❓ 文章 framing 是 "训练效率"，但同时也声称 SR SOTA。两者真的耦合吗？看 Table 3/4 似乎主要 efficiency gain 来自 DAgger 的 dynamic ratio，progressive memory 更像是为了在低 token budget 下保 SR——分开看会更清晰。

## 2. 方法

### 2.1 Architecture overview

三件套：visual encoder + 3D geometry encoder + MLLM backbone（Qwen2.5-VL-3B + StreamVGGT-1B）。每步 t：
1. 对 $v_t$ 抽取 geometry-enhanced visual feature $\mathbf{f}_t = \mathbf{v}_t + \mathbf{g}_t$；
2. 把 instruction embedding、$\mathbf{f}_t$ 和 memory representation 拼成 prompt，让 MLLM 生成动作序列；
3. 用 $\mathbf{f}_t$ 更新 memory。

**Figure 2. 两种 memory 范式：上方为 progressive memory（按时间近远分配 token 数），下方为 recursive memory（用 sentinel tokens 的 KV cache 作为 memory state）。**

![](https://arxiv.org/html/2512.10310v1/x2.png)

### 2.2 Geometry-enhanced visual representation

- **2D**：Qwen2.5-VL 的 patchify + 2×2 spatial merge，得到 $\mathbf{v}_t \in \mathbb{R}^{(h/2p)\times(w/2p)\times c}$。
- **3D**：用 StreamVGGT 处理 RGB stream，复用过去帧的 KV cache 输出 latent geometry tokens（无需 depth 传感器）；KV cache 超限时**随机驱逐**一帧（保留 reference frame）以控显存。geometry token 经 2-layer MLP 对齐到与 $\mathbf{v}_t$ 同形状 $\mathbf{g}_t$。
- **融合**：element-wise add $\mathbf{f}_t = \mathbf{v}_t + \mathbf{g}_t$。

> ❓ 随机驱逐一帧 KV cache——比 LRU / 重要度评分简单粗暴，但作者没消融驱逐策略对 SR 的影响。

### 2.3 Progressive memory representation

**Idea**：模仿"人脑遗忘"。最近 K 帧高保真，越早的帧空间分辨率越低。

- 给定 stride-Δ 采样后的特征序列 $\{\ldots, \mathbf{f}_{t-2\Delta}, \mathbf{f}_{t-\Delta}, \mathbf{f}_t\}$
- 最近 K 帧：2×2 下采样
- 再往前 K 帧：4×4 下采样
- 再往前 K 帧：再 4× 下采样……依此递归直到 feature map 维度无法再降

**Equation. Memory token 数上界**

$$
\frac{K}{4} S + \frac{K}{16} S + \frac{K}{64} S + \cdots = \frac{K}{4} \sum_{i=0}^{\infty} \frac{1}{4^i} S = \frac{KS}{3}
$$

其中 $S$ 为单帧 token 数。取 $K=3$ 时整段历史 token 数不超过单帧。直觉：信息几何级数衰减刚好抵消 attention 二次成本。

### 2.4 Recursive memory representation

**Idea**：固定大小的 state，靠 KV cache 跨步传递。VLN↻BERT 早就用过 recursive state token，但单 token 在深层 MLLM 里梯度传播困难。本文改成：

Prompt 结构：

$$
\{\mathbf{f}_t\}, \{\mathbf{m}_{pre}\}, \{\mathbf{w}\}, \{\mathbf{m}_{cur}\}
$$

- $\mathbf{m}_{cur}$：本步的 learnable sentinel tokens
- $\mathbf{m}_{pre}$：placeholder，每个 transformer block 内其 KV state 被替换为上一步 $\mathbf{m}_{cur}$ 在该 block 计算出的 KV cache
- $\mathbf{m}_{cur}$ 通过 attention 同时聚合当前 input 和 past memory（via $\mathbf{m}_{pre}$）

用 KV cache 而非 hidden state 作为载体，是为了缓解 BPTT 在深层 MLLM 上的长距梯度问题。

> ❓ 但这其实把"梯度长程传播"换成了"梯度只传到上一步的 sentinel KV"——本质上还是只反传一步？文章对此一笔带过，实际可能限制了 long-horizon 学习能力，恰好对应 RxR-CE 上 recursive memory 表现差的现象。

### 2.5 Dynamic mixed policy for DAgger

经典 DAgger：用混合策略 $\pi_\beta = \beta\pi^* + (1-\beta)\pi_\theta$ 收集 trajectory，再用 oracle 重标注。本文把 β 改成时间相关：

$$
\beta_t = 1 - \alpha^{t/T}
$$

- $t$：当前步；$T$：oracle GT path 长度；$\alpha$：decay rate（experiments 用 0.5）
- $\beta_t$ 从 0 增到 1：episode 开头几乎全 learned policy（最大化 explore 制造 compounding error），结尾几乎全 oracle（保证 episode 能完成不被截断）

**Algorithm 1 关键行**：

```
β_t = 1 − α^(t/T)
a_t ~ β_t · π*(·|s_t) + (1 − β_t) · π_θ(·|s_t)
a_t* = π*(s_t)         # 用 oracle 重标注当前 state
```

### 2.6 训练加速

- **两阶段训练**：Stage 1 在 R2R-CE + RxR-CE 上训练基础导航能力（去掉了 StreamVLN 用的 EnvDrop，作者说继续训练比加 EnvDrop 更划算）；Stage 2 加 ScaleVLN-150K + ScanQA + SQA3D + LLaVA-Video-178K subset + DAgger 数据。
- **Sequence packing**：把同一 trajectory 的多个连续步 (step×16) 拼成一个 flatten sequence，配 block-sparse attention。每次 backward 处理的步数从 8 翻到 16，总训练成本降 41.2%；同时也是 recursive memory 跨步反传的前提。

## 3. 实验结果

### 3.1 主结果

**Table 1（节选）. R2R-CE / RxR-CE Val-Unseen SR (%) 对比。† 表示加了 Matterport3D 之外的额外数据。**

| Method | R2R SR | R2R SPL | RxR SR | RxR SPL | nDTW |
|---|---|---|---|---|---|
| NaVid | 37.4 | 35.9 | – | – | – |
| NaVILA | 49.7 | 45.5 | – | – | – |
| StreamVLN | 52.8 | 47.2 | 48.6 | 42.5 | 60.2 |
| **Efficient-VLN** | **60.8** | **53.7** | **63.5** | **52.1** | **66.8** |
| StreamVLN † | 56.9 | 51.9 | 52.9 | 46.0 | 61.9 |
| NavFoM † | 56.2 | 51.2 | 57.4 | 49.4 | 60.2 |
| **Efficient-VLN †** | **64.2** | **55.9** | **67.0** | **54.3** | **68.4** |

**Table 2. 训练成本对比。**

|  | Training cost | #Samples | #Trajectories |
|---|---|---|---|
| UniNavid | 1400 H800·h | 3.6M | – |
| NaVILA | 576 A100·h | 1.5M | 181K |
| StreamVLN | 1500 A100·h | – | 990K |
| NavFoM | 4032 H100·h | 12.7M | – |
| **Efficient-VLN** | **282 H800·h** | 3.7M | 196K |

成本降一个数量级，SR 同时上一个台阶。需要注意 H800 vs A100/H100 的算力差异——按 FLOPs 折算 282 H800·h 大约相当于 600+ A100·h，仍比 StreamVLN 省 60%。

### 3.2 Memory representation ablation

**Table 3. 不同 memory 策略（去掉 ScaleVLN）。**

| # | Variant | R2R SR | R2R #Token | RxR SR | RxR #Token |
|---|---|---|---|---|---|
| 1 | Spatial Compression (all frames, 4 tok/帧, NaVid 风格) | 55.9 | 499 | 47.6 | 606 |
| 2 | Recursive Memory (64 sentinel tokens) | 56.1 | 586 | 54.7 | 677 |
| 3 | Prog. Compression (3 frames) | 58.5 | 661 | 50.6 | 745 |
| 4 | Prog. Compression (6 frames) | **61.3** | 692 | 51.3 | 780 |
| 5 | Prog. Compression (12 frames) | 60.8 | 701 | **63.5** | 785 |

观察：
- Recursive memory（row 2）在短 traj R2R 上能打 SOTA（56.1 vs StreamVLN 52.8），但在长 traj RxR 上明显落后于 progressive memory；作者归因为"state-based memory 难以长期保留信息"。
- Progressive memory 增大 K 在 RxR 上单调改善，R2R 在 K=6 处见顶。说明 long-horizon 任务确实吃 recent-frame 高分辨率红利。

### 3.3 DAgger ablation

**Table 4. DAgger 策略（去掉 ScaleVLN）。**

| # | DAgger 策略 | R2R SR | R2R #Train Step | R2R #Infer Step | RxR SR | RxR #Train Step | RxR #Infer Step |
|---|---|---|---|---|---|---|---|
| 1 | Baseline (no DAgger) | 45.9 | – | 83 | 49.8 | – | 137 |
| 2 | Constant β=0.75 | 50.7 | 66 | 84 | 54.1 | 98 | 126 |
| 3 | Constant β=0.5 | 54.8 | 78 | 94 | 60.7 | 112 | 142 |
| 4 | Constant β=0.25 | 59.5 | 128 | 146 | 62.6 | 160 | 186 |
| 5 | **Dynamic α=0.5** | **60.8** | **82** | **100** | **63.5** | **121** | **154** |

Dynamic ratio 比 β=0.25 SR 略高，但 #Train Step 从 128 降到 82——同时砍 36% 训练 trajectory 长度和 32% 推理长度。

**Figure 3. DAgger 生成 trajectory 的 BEV 可视化：固定 β=0.75 几乎贴着 GT 走（绿色被蓝色覆盖），β=0.25 大幅偏离绕远路，dynamic ratio 在两者之间。**

![](https://arxiv.org/html/2512.10310v1/x3.png)

### 3.4 Data composition ablation

**Figure 4. R2R-CE 上 stage-2 数据消融：R2R+RxR 基线 (45.9) → +DAgger (60.8) → +ScaleVLN (64.2)。DAgger 数据贡献最大。**

![](https://arxiv.org/html/2512.10310v1/x4.png)

### 3.5 3D geometry encoder ablation

**Table 5. R2R-CE，仅 stage-1 训练。**

|  | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
|---|---|---|---|---|
| 2D tokens only | 6.80 | 50.7 | 42.3 | 38.4 |
| + StreamVGGT | 6.41 | 54.5 | 45.9 | 41.9 |
| + Stream3R | 6.39 | 55.5 | 47.1 | 42.6 |

注入 latent geometry token 在 R2R-CE 上 +3.6~4.8 SR。Stream3R 略好但 StreamVGGT 显存友好，故主实验选 StreamVGGT。

### 3.6 3D QA 副产物

附录 Table 7/8：在 SQA3D 56.2 (vs Video-3D LLM 58.6) / ScanQA CIDEr 95.6 (vs StreamVLN 100.2)，作为 2D-VLA 模型基本与专门的 3D LLM 同档。说明 RGB+latent 3D token 的范式不只对 navigation 有用。

---

## 关联工作

### 基于
- [[2412-NaVILA|NaVILA]]: 沿用其 Habitat 配置（512×512, HFOV 90°, 500 max steps）；NaVILA 是早期 RGB-only MLLM-based VLN 代表
- [[2507-StreamVLN|StreamVLN]]: 训练 pipeline / 评估代码 / 两阶段策略都基于 StreamVLN，主要去掉了 EnvDrop 数据
- [[2402-NaVid|NaVid]]: "把每帧压到固定 token 数"的 baseline 思路来源（Spatial Compression 在 Table 3 是 NaVid 风格）
- Qwen2.5-VL: backbone（3B 版本），其 2×2 visual token merge 直接复用
- StreamVGGT: 3D geometry encoder，提供 streaming-friendly 的 latent geometry token

### 对比
- [[2509-NavFoM|NavFoM]]: 训练成本 14× 于 Efficient-VLN，本文核心对比对象之一
- UniNavid: 同样 3.6M samples 但 1400 H800·h，Efficient-VLN 用相同量级数据只要 282 H800·h
- VLN↻BERT: recursive memory 的精神前驱，本文用 KV cache 而非 single state token 改进
- JanusVLN: concurrent work，也把 latent 3D token 引入 VLN，作者主张本文的差异在于 "efficient memory + DAgger 改进"
- CorrectNav: 用 LLM 生成 error-recovery 数据；本文 dynamic DAgger 是无 LLM 依赖的替代

### 方法相关
- DAgger (Ross et al. 2011): mixed policy + dataset aggregation 的原版算法，本文把固定 β 改成时间相关的 schedule
- Stream3R: 备选 3D encoder，在 Table 5 上略优于 StreamVGGT 但更耗显存
- Block-sparse attention (magiattention): sequence packing 时限制 cross-step attention 的实现
- Video-3D LLM / LLaVA-3D / ChatScene: SQA3D / ScanQA 上的 3D LLM baseline，附录 Table 7/8 用作旁证

---

## 论文点评

### Strengths

1. **Cost-vs-SR 帕累托前沿往下移**：282 H800·h 拿到 SOTA，这对小实验室是 actionable 的——StreamVLN 1500 A100·h 已经把 VLN 推到大公司专属。Table 2 的对比图（Figure 1）很有说服力。
2. **Progressive memory 的设计直觉简单且 token 上界清晰**：$KS/3$ 这种几何级数 budget 比拍脑袋设 sliding window 优雅。Recency-bias 的 ablation（Table 3 row 4 vs row 1）clean 地证明了 uniform compression 的浪费。
3. **Dynamic DAgger 是个有 transferable 价值的 trick**：β=1−α^(t/T) 可以直接用在任何 imitation learning + DAgger 的 setup，不限于 VLN。从 cost 角度看比"暴力收更多 trajectory"漂亮。
4. **Recursive memory 的 negative finding 诚实**：作者明确承认 recursive 在长 traj 上不行，没掩饰。这种在主推方法旁边并列展示一个不 work 的方法，反倒增加可信度。

### Weaknesses

1. **Recursive memory 的"梯度传播改进"论证薄弱**：把 hidden state 换成 KV cache 是否真的解决了梯度问题？理论分析缺失，实验只能间接证明 R2R 上能打 SOTA 但 RxR 崩了——更像是 single-step memory 的 capacity 不足，而非梯度问题。
2. **Sequence packing 的 41.2% 加速没拆开归因**：到底是 GPU 利用率提升、attention pattern 变化、还是别的？没有对比 step-by-step 训练相同 step 数的精度，无法排除 packing 改变了梯度估计的可能。
3. **3D geometry encoder 的运行成本被淡化**：StreamVGGT-1B 本身不便宜，inference 时也要跑 3D encoder。Figure 1 / Table 2 只算了 MLLM 的训练成本，3D encoder 的 forward 时间和显存有多少应该单列。
4. **β decay rate α=0.5 是手调的**：α 这个超参对 final SR 的敏感度未见消融，且 α 与 trajectory 长度 T 耦合，长 traj 上等效"前期 explore 时间"更长，这种隐含 curriculum 是否最优没讨论。
5. **Conclusion 数字不一致**：Conclusion 写的是 "R2R-CE 62.3% SR / RxR-CE 64.5% SR"，与 abstract / Table 1 的 64.2 / 67.0 对不上。**编辑事故，但暴露 polish 不足**。
6. **没和 [[2509-NavFoM|NavFoM]] / JanusVLN 做 RGB-only 同设置的细颗粒对比**：NavFoM 是多机器人形态、NaviLLM 是多任务，比较粒度不一致，效率优势可能部分来自 task scope 收窄（仅 R2R/RxR/ScaleVLN 三个数据源）。

### 可信评估

#### Artifact 可获取性

- **代码**: 未说明（项目页 https://lavi-lab.github.io/Efficient-VLN 存在，但论文中未给出 code repo URL；GitHub 搜索仅能找到 LaVi-Lab.github.io 页面 repo）
- **模型权重**: 未说明
- **训练细节**: 完整——backbone (Qwen2.5-VL-3B)、3D encoder (StreamVGGT-1B)、batch 128、lr 1e-5、8×H800、stride Δ=4、window N=12、α=0.5、stage-2 数据混合 (R2R + RxR + DAgger + ScaleVLN-150K + ScanQA + SQA3D + LLaVA-Video-178K subset) 都给了
- **数据集**: 全部公开（R2R-CE / RxR-CE / ScaleVLN-150K / ScanQA / SQA3D / LLaVA-Video-178K），易于复现训练数据

#### Claim 可验证性

- ✅ **R2R-CE 64.2 SR / RxR-CE 67.0 SR**：Table 1 完整指标 (NE/OS/SR/SPL/nDTW)，配 ablation Table 3-5，数字闭环
- ✅ **Progressive > Spatial Compression**：Table 3 row 1 vs row 4 在相同 #Token (~700) 下 +5.4 SR R2R / +3.7 SR RxR，可信
- ✅ **Dynamic ratio 减少 trajectory 长度**：Table 4 row 4 vs row 5 数字明确（128→82, −36%）
- ⚠️ **"282 H800 GPU hours" 的训练效率宣称**：H800 vs A100/H100 算力不同，不同 batch / sequence packing 设置下不可直接横比；StreamVLN 用 A100，等效折算后 Efficient-VLN 真实优势是 ~3-5×（仍很可观），但论文给读者"10×+"的视觉印象（Figure 1 横轴 log scale）
- ⚠️ **Recursive memory "解决梯度传播"**：claim 缺直接证据，实验上 RxR 表现弱反倒削弱了这个 claim
- ⚠️ **Conclusion 中 R2R 62.3 / RxR 64.5 SR**：与正文 64.2 / 67.0 不一致，疑为旧版残留；以正文/表格为准
- ❌ **"providing a strong and efficient baseline for this area"**（Conclusion 末句）：典型 marketing closing，不算技术 claim

### Notes

- **Efficiency claim 的 fine print**：Figure 1 用 log scale 横轴是好看，但读者会高估"X 倍降本"。报告时建议折算到统一卡型（如 A100-equiv hours）再说倍率。
- **Progressive memory 的几何级数衰减是个 transferable building block**：任何 long-context video LLM 想做 memory compression 都可以用——core insight 是"recency-weighted token allocation 比 uniform 更高效"。在 video understanding / streaming agent 上应该普遍可用。
- **Dynamic DAgger β schedule** 对 imitation learning + RL 混合训练范式（VLA / GUI agent）也有借鉴价值：episode 早期 high learner ratio 制造 distribution shift，晚期 high oracle ratio 保 success——这种 curriculum 比固定 ratio 更划算。
- **Recursive memory 的失败是 informative**：再次印证 state-based memory（GRU / Mamba / 单 KV state）在长 horizon multimodal 任务上 capacity 不足，attention over compressed history（progressive）仍是更稳的方案。这与 VLM long-context 领域的趋势一致。
- **下一步可以追**：(1) Progressive memory + recursive memory 的 hybrid（recent K 帧 attention，distant 用 recursive state）能否在长 traj 上拿到两者优势？(2) 把 dynamic β 推广到 VLA 的 teleop+autonomy 混合数据收集；(3) 3D geometry encoder 的轻量替代——StreamVGGT-1B 仍是不小的开销。

### Rating

**分数**：2 - Frontier
**理由**：在 RGB-only MLLM-based VLN 方向是当前 SR-vs-cost 帕累托前沿（R2R-CE 64.2 / RxR-CE 67.0 在 282 H800·h 下刷过 StreamVLN / NavFoM / NaVILA），且 progressive memory（KS/3 token budget）和 dynamic DAgger（β=1−α^(t/T)）都是可迁移到其他 video LLM / imitation learning 场景的 building block——满足 Frontier 的"当前 SOTA + 必比 baseline"定义。没到 Foundation 档：作为 2025-12 新发论文尚无社区采纳信号（无 github 代码发布、未被后续工作采用），且方法上是 StreamVLN/NaVILA 脉络的高效化改进而非范式开创（NaVid/StreamVLN 才是该脉络的奠基者）；也不至于 Archived，因为 efficiency frontier 本身对小实验室是 actionable，方法尚未被更强工作取代。
