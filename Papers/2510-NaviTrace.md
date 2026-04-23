---
title: "NaviTrace: Evaluating Embodied Navigation of Vision-Language Models"
authors: [Tim Windecker, Manthan Patel, Moritz Reuss, Richard Schwarzkopf, Cesar Cadena, Rudolf Lioutikov, Marco Hutter, Jonas Frey]
institutes: [ETH Zurich (RSL), KIT (IRL), FZI, Robotics Institute Germany]
date_publish: 2025-10-30
venue: arXiv preprint
tags: [navigation, VLM, embodied-reasoning]
paper: https://arxiv.org/abs/2510.26909
website: https://leggedrobotics.github.io/navitrace_webpage/
github: https://github.com/leggedrobotics/navitrace_evaluation
rating: 2
date_added: 2026-04-23
---

## Summary

> [!summary] NaviTrace: Evaluating Embodied Navigation of Vision-Language Models
> - **核心**: 提出一个 VQA-style 的 embodied navigation benchmark——给定一张真实图像、一条自然语言指令、一个 embodiment 类型，模型需输出 image-space 的 2D 轨迹 (trace)，用 semantic-aware trace score 与人类标注对齐。
> - **方法**: 1000 张手工采集图像 + 3000+ 专家 trace，4 种 embodiment（human/legged/wheeled/bicycle）；score = DTW + FDE + Mask2Former 语义惩罚，Spearman 相关性 0.87 与 human pairwise ranking 对齐。
> - **结果**: 评估 8 个 SOTA VLM（Gemini 2.5 Pro / GPT-5 / o3 / Claude Sonnet 4 / Mistral Medium 3.1 / Qwen3-VL-235B / Qwen2.5-VL-72B / Gemma-3-27B）。最好的 Gemini 2.5 Pro 仅得 34.4（人类 75.4），主要失败模式是 **goal localization**——即便给定 goal，path shaping 仍输给人类。
> - **Sources**: [paper](https://arxiv.org/abs/2510.26909) | [website](https://leggedrobotics.github.io/navitrace_webpage/) | [github](https://github.com/leggedrobotics/navitrace_evaluation)
> - **Rating**: 2 - Frontier（首个把 embodiment-aware 2D trace 预测作为 navigation VLM 评测协议的 benchmark，填补了 VLN 和 social-nav VQA 之间的空白，但范式能否成为 de facto 标准仍待观察）

**Key Takeaways:**
1. **Trace as VLM-native navigation output**: 把 navigation answer 表达成 image-space 的 2D 点序列——这是 pointing 任务（VLM 已相对擅长）的自然延伸，比 "Forward/Left/Right" 这类离散动作更表达性强，且避开了 VLN 那种需要在特殊 action space 预测的限制。
2. **Goal localization 才是主要瓶颈**: 把 Gemini 2.5 Pro 的任务拆成 "只预测 goal" vs "只预测 path"，前者 29.65 / 完整 34.38 / Oracle-Goal Straight-Line 51.89——goal 预测的 gap 远大于 path shaping 的 gap。
3. **Embodiment awareness 未被现有 VLM 内化**: 不同 embodiment（人/四足/轮式/自行车）间的 aggregate 分差极小，说明模型并没有根据 embodiment 调整策略，而是在做 embodiment-agnostic 的大致轨迹预测。
4. **Semantic-aware score 与人类偏好对齐**: DTW(0.84) → +FDE(0.87) → +Mask2Former penalty(0.87) 的 Spearman 相关性。且 Mask2Former 自动分割与 manual segmentation 差距很小，说明 automated semantics 足够支撑一个 scalable metric。
5. **Straight Forward baseline 尴尬地接近 o3**: 一条画穿图像中央的直线作为 "zero-information" 基线就能逼近 o3 的表现，揭示当前 reasoning VLM 的 spatial grounding 远未达到可用状态。

**Teaser. NaviTrace 概览视频**：展示 benchmark 的任务设定——模型拿到一张真实世界的 first-person 图像、一个 embodiment 标签和一个自然语言指令，输出一条 2D trace；不同 embodiment 应给出不同的 trace（例如楼梯对轮式不可行）。

<video src="https://leggedrobotics.github.io/navitrace_webpage/static/videos/teaser.mp4" controls muted playsinline width="720"></video>

---

## Motivation

评估 VLM 的导航能力目前有三条路，各有硬伤：
- **Real-world closed-loop rollout**：贵、慢、不可复现、难覆盖多样场景。
- **Simulation**：可复现性好，但场景多样性受限于仿真构建，语义、地形、社会规范这些难建模；还有 sim-to-real gap。
- **VQA benchmark**：现有的 social-nav VQA（SocialNav-SUB、Social-LLaVA）用文本回答、只覆盖单一 embodiment。

NaviTrace 选了第三条路并做了两个关键扩展：(1) 输出从 text answer 升级成 image-space trace（保留 2D 空间结构）；(2) 覆盖 4 种 embodiment 以评测 embodiment-conditioned navigation。

> ❓ 用 2D image-space trace 代替 3D 路径是有损的——距离远处的微小像素偏差对应现实中的大位移误差，benchmark 里 FDE 是像素距离，无法直接映射到 3D 里的 "到没到目标"。这是刻意换取 "VLM 可直接输出 + 人类可直接标注" 的 trade-off，作者在 Limitations 里也承认了这一点。

**Figure 2. 数据多样性**。左：图像来源的地理分布（瑞士为主，辅以多国样本）；右：场景按 urban/rural、natural/structured、光照、天气分布。

![](https://arxiv.org/html/2510.26909v3/x1.png)

---

## Benchmark 设计

### Data

- **1000 scenarios + 3000+ traces**，validation/test 各 50%，test annotation 隐藏用于 public leaderboard。
- **Image**：众包采集（手机/GoPro），另加 164 张来自 GrandTour dataset；用 EgoBlur 脱敏人脸和车牌。
- **Instruction**：手写，可以是 goal-based（"Go to the red car"）或 directional（"Go forward, then turn left at the traffic light"）。
- **Trace**：每个适用 embodiment 至少 1 条，若有等价替代方案（绕障碍物从左/右）则多条 ground-truth。
- **7 类 task category** tag 标注：Geometric Terrain / Semantic Terrain / Accessibility / Visibility / Social Norms / Dynamic Obstacle / Stationary Obstacle。
- **4 种 embodiment**：Human、Legged Robot（ANYmal 类）、Wheeled Robot（wheelchair-like delivery bot，无法走楼梯）、Bicycle（遵守交规，偏好 bike lane）。**刻意排除汽车**（视角差别过大）。

### Score 函数

把三个因素线性叠加：

$$
\text{Score}(T, \mathcal{G}) = \underset{T' \in \mathcal{G}}{\min} \, \text{DTW}(T, T') + \text{FDE}(T, T') + \text{Penalty}(T)
$$

**符号说明**：
- $T$：预测的 trace；$\mathcal{G}$：该 scenario 的所有等价 ground-truth traces。
- $\text{DTW}$：Dynamic Time Warping 距离，衡量轨迹形状相似度。
- $\text{FDE}$：Final Displacement Error，衡量是否到达目标。
- $\text{Penalty}$：基于 Mask2Former + Mapillary Vistas 的 per-pixel 语义成本，按 embodiment 调整权重（见 Appendix Table IV，如轮式机器人在 "楼梯"/"楼梯类斜面" 上惩罚更大，自行车在 bike lane 惩罚为 0、在 sidewalk 上反而有惩罚）。

**可解释 scaling**：把原始 score 映射到 0–100，`Straight Forward`（画穿图像中央的竖直线）对应 0，ground-truth 对应 100（允许负分，若某些模型比 Straight Forward 还差）。

$$
\widehat{\text{Score}}(T, \mathcal{G}) = \frac{3234.75 - \text{Score}(T, \mathcal{G})}{3234.75} \cdot 100
$$

**Figure 3. Score 与人类偏好对齐实验**。左：penalty 的两种来源——Mask2Former 自动分割 vs manual segmentation，对比；右：score 和 human pairwise ranking 做 Spearman 相关性。

![](https://arxiv.org/html/2510.26909v3/x2.png)

### Score 消融（与 human preference 对齐）

**Table II. Spearman correlation between score variants and human ranking.**

| Score Variant | Spearman ↑ |
|---|---|
| RMSE | 0.8167 |
| Fréchet | 0.8310 |
| DTW | 0.8417 |
| DTW + FDE | 0.8656 |
| DTW + FDE + Manual Penalty | 0.8723 |
| DTW + FDE + Mask2Former (ours) | 0.8707 |

几个值得注意的点：
- DTW 单项就比 RMSE、Fréchet 好；说明轨迹相似度本身不是简单 pointwise 距离问题。
- 加 FDE（goal term）是最大的 jump（+0.024）。说明 "到没到目标" 是人类评判轨迹好坏的关键维度。
- 加语义惩罚再提升 ~0.005，manual vs automatic 只差 0.0016。**automated semantic penalty 几乎白嫖到了 manual segmentation 的全部收益**——对 scalable benchmark 关键。

---

## 实验

### Baselines

- **Human**: 多人合作解 test split，提供上界。
- **Straight Forward**: 图像中心竖直线（naïve baseline，作为 0 分锚点）。
- **Oracle-Goal Straight Line**: 知道真实 start/goal，直接连直线。
- **Only predict goal (Gemini 2.5 Pro)**: 只让模型预测 goal 点，再和 start 用直线连。
- **Only predict path (Gemini 2.5 Pro)**: 给 start 和 goal，让模型预测中间 path shape。

### Models

8 个 SOTA VLM——5 个 proprietary：**Gemini 2.5 Pro**、**GPT-5**、**o3**、**Claude Sonnet 4**、**Mistral Medium 3.1**；3 个 open-weight：**Qwen3-VL-235B-A22B Thinking**、**Qwen2.5-VL-72B**、**Gemma-3-27B**。其中 Gemini 2.5 Pro / GPT-5 / o3 / Qwen3-VL 是 reasoning 模型（会自动产生推理步骤）。均通过 API 调用，temperature=1.0、max_tokens=5000。输出格式约束为 normalized coordinates 的 JSON list。

### 主实验结果

**Figure 4. 排序 & per-category 拆解**。左：各 VLM、Straight Forward、Human 在 4 种 embodiment 下的 score；右：按 task category 拆。核心观察是 **VLM 之间的序 Gemini 2.5 Pro > GPT-5 > Qwen3-VL > o3 ≈ Straight Forward**，而人类远远甩开所有模型。

![](https://arxiv.org/html/2510.26909v3/x3.png)

几个关键观察：
1. **Human ≫ all VLMs**：即使最好的 Gemini 2.5 Pro 也远离 human 表现。
2. **o3 ≈ Straight Forward**：reasoning 模型 o3 竟然被画穿图像中央的竖直线逼平——"zero-information" 的空间 prior 就够它吃了。
3. **Embodiment 几乎无差**：4 种 embodiment 下的分布形状类似，说明 VLM 没真正根据 embodiment 调策略（否则 wheeled robot 在含楼梯的场景里应该显著更差）。
4. **Category 几乎无差**：同理，模型在 geometric / semantic / social 等不同 category 上表现相近——并非均衡，而是整体都差，差到 category 特异性被噪声淹没。

**Figure 5. 定性对比四个 top 模型的预测**。

![](https://arxiv.org/html/2510.26909v3/x4.png)

### 任务分解：goal vs path

**Table III. 在 Gemini 2.5 Pro 上分解 goal / path 难度**。

| Model | Score ↑ |
|---|---|
| Only goal point with Gemini 2.5 Pro | 29.65 |
| Gemini 2.5 Pro (full trace) | 34.38 |
| Oracle-Goal Straight Line | 51.89 |
| Only path with Gemini 2.5 Pro | 56.55 |
| Human Expert | 75.40 |

解读：
- **Only-goal 29.65 vs full-trace 34.38**——只差 4.7 分，说明 goal 预测错了，path 再好也救不回来；**goal localization 是 dominant failure mode**。
- **Only-path 56.55 > Oracle-Goal Straight Line 51.89**：给 Gemini 正确的 goal，它画的 path 比直线好——说明模型有 basic path shaping 能力。
- 但 only-path 的 56.55 仍 < Human 75.40——即便拿到 goal，path shape 也差于人类。
- 结论是 **dual difficulty**：goal localization 是主瓶颈，path shaping 是次瓶颈。

### Reasoning 和 grounding 脱节

**Figure 6. o3 的 reasoning vs prediction**。任务是 "go to the red car"。o3 的文字推理能正确分析 path 选项并识别出正确方向，但最终 trace 预测不 align——reasoning 是对的，trace 是错的。这反映了 **linguistic reasoning 和 spatial grounding 之间存在 gap**，尤其在定位可通行结构时。

![](https://arxiv.org/html/2510.26909v3/x5.png)

> ❓ 这个 gap 是根本性的吗？reasoning 模型理解 scene 语义但无法把 reasoning 结论投射回 2D pixel space。可能是 action head（coordinate generation）没有经过 grounding 监督训练——从 SFT 数据的角度看 CoT output 和 coordinate output 的联合训练稀少。如果 NaviTrace validation split 被用于 SFT/DPO，这个 gap 会收窄到多少？是 benchmark 设计的自然下一步问题。

---

## 关联工作

### 基于
- **Dynamic Time Warping (DTW)**（Senin, 2008）：trace 相似度主度量，benchmark 分数的核心组件。
- **Mask2Former**（Cheng et al., 2022）+ **Mapillary Vistas**（Neuhold et al., 2017）：per-pixel semantic penalty 的基础，决定了哪些语义类对哪个 embodiment 是"hazardous / irrelevant / 可通行"。
- **GrandTour dataset**（Frey et al., RSS 2025）：提供 164 张 legged robot 视角图像。

### 对比
- **VLN benchmarks**（R2R / REVERIE / RxR / OctoNav-Bench）：都依赖 MP3D/Habitat 仿真，action space 对 VLM 不友好，只评估 human 或单一 embodiment。NaviTrace 用 real images + VLM-native 2D trace + 多 embodiment。
- **Real-world nav datasets**（EgoWalk / CityWalker）：从真实 ego 视频自动提取轨迹，但没有 language-conditioned 任务（CityWalker）或指令质量受限（自动提取）。
- **Social-nav VQA**（SocialNav-SUB / Social-LLaVA）：text-based answers，单 embodiment；NaviTrace 换成 2D trace 并覆盖 4 embodiment。
- **VLM 通用 benchmark**（BLINK / Cambrian-1 / EmbSpatial-Bench / PointArena）：测 perception / spatial / pointing 但不测 sequential navigation plan。

### 方法相关
- **Pointing** 作为 grounding 评测（Molmo / PointArena / Magma）：NaviTrace 把 pointing "延伸" 成 sequential 2D trace——goal + path。
- **Trace for manipulation**（LLaRVA / Scaffolding DexManip / TraceVLA）：先例——2D trace 能帮机械臂 policy 习得 spatial-temporal awareness，NaviTrace 把 "trace as VLM I/O" 的思路从 manipulation 搬到 navigation。
- **VLA for navigation**（NaViLA / Uni-NaViD / Gemini Robotics）：应用侧工作，NaviTrace 提供的是评估而非方法。

---

## 论文点评

### Strengths

1. **Problem formulation 清晰**：把 "VLM 能不能导航" 的模糊问题精确化为 "能不能预测合理的 image-space 2D trace"，既避开了 closed-loop 和 sim-to-real 的复杂性，又保留了 embodiment-aware navigation 的核心挑战。
2. **Metric 设计经过 human-preference 对齐**：Spearman 0.87 的相关性不低，且 Mask2Former automated 版本几乎追平 manual segmentation——意味着 benchmark 可以 scale，不需要每张图都做 dense 人工分割。
3. **Ablation 诚实且信息量大**：goal vs path 分解明确揭示了 dominant failure mode；o3 的 reasoning-trace gap 例子是 reasoning VLM 社区应该警惕的信号（CoT 正确 ≠ grounding 正确）。
4. **Leaderboard + validation/test split 分离**：避免了 test set 泄漏，支持 fine-tuning experiments 用 validation split——这是 scalable benchmark 的必要基础设施。

### Weaknesses

1. **单 image, single-step**：作者在 Limitations 明确——不能评估 temporal reasoning 和 multi-step planning，而真实 navigation 恰好需要这两个。
2. **地理偏倚**：集中在瑞士，对 infrastructure 和 social norm 差异大的地区（比如东南亚城市、发展中国家农村）泛化性未验证。
3. **2D image-space 与 3D 现实的 gap**：FDE 和 DTW 用像素单位，远处像素小偏差对应实际大位移；score 不直接映射到 "任务是否完成"。
4. **Score 的 goal 定义偏硬**：ground-truth 是点而不是区域（例如 doorway），导致某些场景下 FDE 惩罚过严或过松。
5. **Reasoning 模型里 o3 的排名异常低**：Gemini 2.5 Pro / GPT-5 / Qwen3-VL 都是 reasoning 模型且表现好，o3 却接近 Straight Forward。这可能提示 o3 对坐标输出的训练或 prompt adherence 有问题，作者没深入探究是否 prompt-sensitive。
6. **没做 fine-tuning 实验**：有 validation split 却未展示在 validation 上 SFT/DPO 能把分数推多高——这会决定 benchmark 作为 "推动技术进步" 而非 "测 frozen model" 的价值。

### 可信评估

#### Artifact 可获取性
- **代码**：inference + scoring（[leggedrobotics/navitrace_evaluation](https://github.com/leggedrobotics/navitrace_evaluation)，MIT license）
- **模型权重**：N/A（不提出新模型，只评估现有 VLM）
- **训练细节**：N/A（无训练）；推理细节：temperature=1.0, max_tokens=5000, 默认 hyperparameters
- **数据集**：[HuggingFace leggedrobotics/navitrace](https://huggingface.co/datasets/leggedrobotics/navitrace)，validation split 完整公开，test split ground-truth 隐藏（通过 leaderboard 评测）

#### Claim 可验证性
- ✅ **Score 与 human preference 对齐（Spearman 0.87）**：Table II 有完整消融，manual vs automatic 对比也做了。
- ✅ **VLM << Human**：Figure 4 + Table III 数据充分，模型间排序由 public leaderboard 支持。
- ✅ **Goal localization 是 dominant failure mode**：Table III 的 29.65 / 34.38 / 51.89 / 56.55 数字差异足以支撑这一结论。
- ⚠️ **"Embodiment 间差别小 = 模型没有 embodiment-specific blind spot"**：作者自己承认另一种可能——模型整体太差导致 embodiment 差异被噪声淹没。这是并存解释，需要更强模型或 per-category per-embodiment 的细拆才能区分。
- ⚠️ **o3 的 reasoning-trace gap 是 systematic**：只展示了一个例子（Figure 6）作为 qualitative evidence，没有量化 "多少比例的 o3 样本具备正确 reasoning 但错误 trace"。

### Notes

- 对我的 spatial-reasoning / embodied reasoning 研究兴趣，这个 benchmark 最有意思的地方是 **Figure 6 揭示的 "reasoning 正确但 grounding 错误" 现象**。如果能系统量化这个现象（例如用 GPT-4o 审核 reasoning step 的正确性，再看对应 trace 误差），可能是一个 "reasoning 和 grounding 之间存在什么结构 gap" 的切入点。
- Benchmark 的一个天然扩展：给定当前的 single-image formulation 太弱，下一步要么变 multi-frame（temporal reasoning），要么变 3D-aware（从 image-space trace → BEV trace）。NaviTrace 的 trace representation 可以自然迁移到 BEV。
- 值得留意 leaderboard 的后续更新——如果 Gemini 2.5 Pro → Gemini 3 的代际更新能把 score 从 34 推到 50+，说明 benchmark 对模型能力提升敏感；如果停滞，说明存在天花板。

### Rating

**Metrics** (as of 2026-04-23): citation=8, influential=0 (0%), velocity=1.4/mo; HF upvotes=14; github 29⭐ / forks=1 / 90d commits=0 / pushed 108d ago

**分数**：2 - Frontier

**理由**：发表 < 6 个月，8 citation / 0 influential 的绝对量不能作为定档依据，需看 early signal。HF 14 upvotes、github 29⭐ 属中等热度；benchmark 方向上它填补了 "VLM-native + embodiment-aware + real-world image" 的空白，**范式上（把 navigation answer 表达为 2D trace）是社区迟早要做的事**。但能否从 Frontier 升到 Foundation 取决于：(1) 主要 VLM/VLA 团队是否在模型报告中把它当作 navigation 能力标准评测；(2) validation split 能否促成显著的 fine-tuning 收益证明 benchmark 推动了技术进步。目前两点都未定型，所以是 2 而非 3；而评测 8 个主流 VLM 的系统性、human-aligned metric 的扎实性又远高于 Archived 级别的 niche benchmark，所以不是 1。
