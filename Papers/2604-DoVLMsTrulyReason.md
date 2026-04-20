---
title: "Do Vision-Language Models Truly Perform Vision Reasoning? A Rigorous Study of the Modality Gap"
authors: [Yige Xu, Yongjie Wang, Zizhuo Wu, Kaisong Song, Jun Lin, Zhiqi Shen]
institutes: [NTU, Alibaba-NTU ANGEL, Tongyi Lab Alibaba]
date_publish: 2026-04
venue: arXiv preprint
tags: [VLM, LLM, embodied-reasoning]
paper: https://arxiv.org/abs/2604.16256
website:
github: https://github.com/xuyige/CrossMath
rating: 2
date_added: 2026-04-20
---
## Summary

> [!summary] Do VLMs Truly Perform Vision Reasoning? A Rigorous Study of the Modality Gap
> - **核心**: 构造 text-only / image-only / image+text 三种**信息等价**的同一道数学 crossword puzzle，用此对照 SOTA VLM，发现模型在 text-only 远好于 image-only，加入 image 反而**拖累** text-only baseline —— 当前 VLM 的"视觉推理"主要跑在 text backbone 上。
> - **方法**: 构建 CrossMath（5000 训 / 250 测），自动管线爬取 crossword-math 在线生成器，图像→Markdown 表（像素颜色区分单元格角色 + Qwen3-VL-Max 批量 OCR），符号求解器产出带 hop 标注的 CoT；用 LoRA SFT + GRPO（position-weighted reward）在 image-only 上 post-train Qwen3.5-9B。
> - **结果**: Qwen3.5-Plus 在 Macro Acc 上 text 92.8% vs image 12.4% vs multimodal 74.8%；SFT+GRPO 把 Qwen3.5-9B 的 image Macro 从 3.20 拉到 50.40，但与同模型 text 76.40 仍差一截；MathVerse +2.46 / MMMU +1.39。
> - **Sources**: [paper](https://arxiv.org/abs/2604.16256) | [github](https://github.com/xuyige/CrossMath)

**Key Takeaways:** 
1. **Rigorous cross-modal equivalence**: 每道题同时有 image / text / image+text 三种等价形式（人工校对 250 题），是相对前人 MathVerse/EMMA 等 "信息不对称" benchmark 的关键改进。
2. **Modality gap 远超预期**: 不只是 image < text，更反直觉的是 **image+text < text**——即使 text 已经给了完整解题信息，VLM 还是会被视觉 token 拖下水。
3. **Bottleneck 不是 perception**: OCR 可被 VLM 高精度完成；Accuracy 随 hop 数急剧衰减而非随图像复杂度衰减，定位为"把视觉 symbol 组织进 reasoning chain"的失败。
4. **Image-only post-training 可修复但无法闭合 gap**: SFT+GRPO 在 image-only 上暴涨 +47 pt Macro，text-only 也同时受益（44→76.4），但 image vs text 仍差 ~26 pt，作者归因为 visual encoder / cross-modal alignment 的底层架构瓶颈。
5. **Scale 不救 image-only**: 27B 到 397B-A17B 在 image-only 下 Macro Acc 在 12-16 间盘整无单调提升；text-only 则呈正常 scaling —— 证据指向 vision module 而非 language backbone 容量。

**Teaser. CrossMath 的三模态等价示例（image / markdown table / image+text）**

![](https://arxiv.org/html/2604.16256v1/res/math_puzzle_0217_markdown.png)

![](https://arxiv.org/html/2604.16256v1/res/math_puzzle_0004_blank.png)

---

## 1. Motivation：为什么现有 benchmark 测不出 "genuine visual reasoning"

作者指出两类失效：

1. **Surface-level / text-prior 泄漏型**（如经典 VQA）：只需粗粒度视觉识别 + 文本先验即可答对，无法逼模型做 multi-step spatial / geometric 推理。
2. **Modality-entangled 型**（MMMU / MathVerse / EMMA 等）：问题本身同时需要图和文，抽掉任一模态题目不成立——因此跨模态性能差异**混淆**了"信息缺失"和"模态特定推理能力"两个变量。

为了 cleanly 分离后者，作者提出三条评测原则：
- (i) **Vision-first**：任务本身就得靠空间/几何推理解。
- (ii) **Difficulty-stratified**：避免 saturation / floor effect。
- (iii) **Strictly equivalent** 的多模态版本：三种输入**任务相关信息完全一致**，且都独立可解。

这第三条是本文的核心主张，也是和过往 modality-gap 类论文（如 [^10] VISTA-Bench visualized-text）的关键区别：不仅等价，还要每种单模态都 self-sufficient。

## 2. CrossMath：Benchmark 构造

### 2.1 任务定义

2D 网格上纵横交错的算术方程，部分单元格缺失（标 `?`），要求同时满足所有横/竖方向方程，输出缺失值（按 top-to-bottom, left-to-right 顺序）。

VLM 解题流程形式化为两阶段：

$$
\hat{\mathcal{S}}_{j} = [\hat{s}_{1,j}, \hat{s}_{2,j}, \cdots, \hat{s}_{|\hat{\mathcal{S}}_{j}|}], \quad \hat{s}_{i,j} = \mathrm{VLM}(\mathcal{I}, \mathcal{Q}_{j}, \hat{s}_{<i,j})
$$

每个 step 包含 rationale tokens + 该步 answer span，最后按序拼成 $\hat{\mathcal{A}}_j$。

### 2.2 Data Curation 管线（4 步）

1. **Raw Collection**：用 Playwright 从某在线 arithmetic puzzle 生成器抓取，系统性变化难度（Easy/Medium/Hard）、运算符组合（`{+,-,×,÷}`）、数值范围（50–250）、方程数量（5–15）。每题同时截"未解"+"带解"两图，并解析 HTML 拿到 ground-truth。
2. **Image → Markdown**：利用 **像素颜色**辨认单元格角色（blue = 固定常数、white + red 字 = 未知变量、yellow = 运算符），把单元格切碎拼成带 index 的 mosaic，再让 Qwen3-VL-Max 批量 OCR，配合正则修正 `1/l/|` 类 OCR 歧义，最终人工 audit 250 测试样全部校对。**这一步保证了三模态信息严格对称**。
3. **Reasoning Path Extraction**：符号求解器基于 known-cells 集合做**迭代推演**——每轮找"三元里已知两元"的方程标 solvable，同一轮内求解的所有方程共同构成一个 reasoning step，推出的单元直到下一轮才并入 known-cells（保证严格因果顺序）。这给每题产出带**hop 数**的 gold CoT。
4. **Visual Style Augmentation**：4 个 style —— Original / Border-Removal / Background-Complexity / Font+Palette Variation —— 用于探模型是否仅在 exploit 特定渲染。

### 2.3 Benchmark 统计

**Table 1. CrossMath 测试集按难度与 hop 分布（N=250）**

| Difficulty | # Examples / % | Avg. Problems | 1 Hop | 2 Hops | 3 Hops | 4+ Hops |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 90 / 36.0% | 10.46 | 10.46 | 0.00 | 0.00 | 0.00 |
| Medium | 85 / 34.0% | 9.81 | 7.66 | 1.53 | 0.53 | 0.09 |
| Hard | 75 / 30.0% | 10.64 | 5.13 | 2.52 | 1.73 | 1.25 |
| Total | 250 | 10.29 | 7.91 | 1.28 | 0.70 | 0.41 |

> ❓ Hop 分布严重向 1-hop 倾斜（总体 77% 是 1-hop，4+ hop 只占 4%）。这会让 hop-scaling 曲线的尾段样本量偏低，某些数值（如 Qwen3.5-Plus 4+hop 8.82%）的置信度值得打折。

## 3. Post-Training：能否靠训练补齐

### 3.1 Data

5000 张 puzzle 图 + 解；每张原图与其背景增强版配对（增强 visual robustness）。训测严格 disjoint。

### 3.2 SFT（cold start）

- 用 Qwen3-VL-Max 把 symbolic step 转成自然语言 CoT（给 text-only + Markdown table prompt），即使偶有错答也全部保留做 behavioral cold-start。
- Qwen3.5-9B + LoRA r=16，lr=2e-5，2 epoch，cosine schedule + warmup 0.03，max len 5000，grad accum 8。
- 平均 CoT 长约 5200 token（已强调 concise）。

### 3.3 RLVR with GRPO

SFT 初始化后用 GRPO，每 instance 采 4 rollouts，LoRA r=16，lr=1e-6，200 steps，max completion 6000。

**核心改动：position-weighted reward**——给 hop 数更深的子问题更大权重，避免规则 reward 对中间结果的脆弱抽取：

$$
r_{j} = \frac{\sum_{i=1}^{|\mathcal{S}_{j}|} w_{i} \cdot \mathbb{I}[\hat{a}_{i,j} = a_{i,j}]}{\sum_{i=1}^{|\mathcal{S}_{j}|} w_{i}}
$$

$w_i$ 随逻辑深度递增。

## 4. Main Results

### 4.1 三模态对比（Table 2）

| Model | Img-Only Micro/Macro | Img+Text Micro/Macro | Text-Only Micro/Macro |
| --- | --- | --- | --- |
| Qwen3.6-Plus | 32.23 / 11.60 | 90.76 / 79.60 | 96.23 / 88.20 |
| Qwen3.5-Plus | 35.65 / 12.40 | 85.22 / 74.80 | **97.27 / 92.80** |
| Qwen3.5-397B-A17B | 39.26 / 16.00 | 87.54 / 78.40 | 96.86 / 92.00 |
| Qwen3.5-122B-A10B | 36.85 / 12.80 | 81.37 / 61.20 | 89.61 / 77.60 |
| Qwen3.5-27B | 40.74 / 12.40 | 87.04 / 66.40 | 88.67 / 75.20 |
| Qwen3.5-9B | 23.25 / 3.20 | 61.56 / 29.60 | 73.39 / 44.00 |
| Qwen3.5-9B-SFT | 59.52 / 48.50 | 67.90 / 60.00 | 82.58 / 69.60 |
| Qwen3.5-9B-SFT+GRPO | **62.33 / 50.40** | 71.21 / 62.80 | 87.36 / 76.40 |

### 4.2 三条核心发现

**(1) Image vs Text 存在巨大 gap**：信息对称的前提下，Qwen3.5-Plus Macro 从 92.8 % 掉到 12.4 %（−80 pt）。**更反常的是 image+text 几乎总是低于 text-only**——当 text 已给全信息，视觉输入非但不加分，还注入"ambiguous or poorly grounded features"干扰内部逻辑。

**(2) 失败不来自 perception**：理由三条，都相对扎实：
- 让 VLM 把 puzzle 图转成 Markdown 的 OCR 错误率极低；
- 若是 perception 瓶颈，reasoning-chain supervision 应无效——但 SFT/GRPO 大幅改善 image-only 成绩；
- 若是 perception 瓶颈，准确率应与 hop 数**独立**——但实测 accuracy 随 hop 数剧烈塌陷。

**(3) Reasoning depth 才是核心瓶颈**：Macro vs Micro 的巨大 gap 说明模型能拼对局部但守不住全局一致性。**Table 3** 的 hop 分解验证：所有模型随 hop 数单调衰减，Qwen3.5-27B image-only 从 1-hop 42.26% 崩到 4+hop 5.88%。

### 4.3 Post-training 效果（Table 3）

| Model | Micro | Macro | 1 Hop | 2 Hops | 3 Hops | 4+ Hops |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B (zero-shot) | 23.25 | 3.20 | 26.27 | 3.37 | 2.32 | 1.96 |
| Qwen3.5-9B-SFT | 59.52 | 48.50 | 57.92 | 53.96 | 52.18 | 40.93 |
| Qwen3.5-9B-SFT+GRPO | **62.33** | **50.40** | **61.02** | **58.42** | **56.98** | **41.18** |

- SFT 把 Macro 从 3.20 拉到 48.50（**+45.3**），GRPO 再加 +1.9，主要收益来自 multi-hop。
- 但和同模型 text-only（87.36 / 76.40）比，**image-only 仍差约 25 pt**——作者得到一个相当尖锐的结论：**仅对齐 visual embedding 到 text space 不够，要闭合 gap 需要更强的 vision foundation，让模型从 pixel-level recognition 升级到"内化物理/结构约束"**。

## 5. 辅助实验

### 5.1 Style Robustness（Table 4）

Post-trained 模型对 font/color/background 扰动基本不掉点（原 style 50.40 → 换色/背景 50.10 左右，字体+色 48.40）。**唯独去掉边框掉 4-6 pt**——暗示 grid border 给了重要的单元格分割结构信号，模型"理解 2D 结构"的能力比"抗色干扰"弱。

> ❓ 这其实是一个值得深挖的 finding：去掉 border 精度掉 5 pt 并不"小"，它说明模型依赖**显式视觉 delimiter** 来切分 reasoning unit。如果换到自然图像里，根本没有这种清晰 border——暗示这里的"visual reasoning"依然在"精心设计的 structured visual"的舒适区。

### 5.2 OOD Generalization（Table 5）

| Model | MathVerse (Vision-Only) | MMMU |
| --- | --- | --- |
| Qwen3.5-9B | 48.94 | 67.52 |
| Qwen3.5-9B-SFT | 50.76 | 68.05 |
| Qwen3.5-9B-SFT+GRPO | 51.40 | 68.91 |

CrossMath 训练 transfer 到两个通用多模态数学 benchmark 都是**小正收益**（MathVerse +2.46, MMMU +1.39）——作者 admit 这不等于通用 transferability，仅是"structured multi-step math"范围内的 cross-task synergy。

### 5.3 VLM 规模 vs 模态（5.7 节）

- **Image-only 不随 scale 单调提升**：27B 到 397B-A17B Macro 在 12-16 盘整。
- **Text-only scaling 正常**：Plus / 397B-A17B 明显优于 27B / 9B。
- 解读：语言 backbone 再大也救不了视觉通路；vision module 能力 / cross-modal grounding 才是真瓶颈。这是对"做大 LLM 即可"观点相当直接的反例。

## 6. 我的批判性解读

### 做对的

- **控制变量设计**是本文最硬的贡献：三模态信息等价 + 全部人工校 250 道，比 MathVerse / EMMA 这类 entangled benchmark 干净。
- **把 OCR 失败 / reasoning 失败 disentangle 的三条论证**都不仅仅靠一个数字，而是从 "OCR 本身准确" + "supervision 有用" + "hop-scaling 行为" 三路交叉验证，结构比典型"一张图一个结论"类 critique 论文强。
- **position-weighted reward**是个简单但合理的 trick，避开了从 free-form CoT 里抽中间答案的脆弱性。

### 存疑 / 没做够的

1. **"Vision-first"真的 vision-first 吗？** CrossMath 本质是 **2D 排版的符号算术**，视觉部分只是一个 color-coded grid——这更像"structured visual symbol parsing"而非真正的几何 / 物理视觉推理。去掉 border 掉 5 pt 的实验恰恰说明模型的 "visual reasoning" 依赖显式 delimiter。移到自然场景（图表、几何图、物理 diagram）结论能否保持？未验证。
2. **只做了 Qwen 系列**。所有 main table 全是 Qwen3.5 / 3.6 家族，连 GPT-4o、Gemini、Claude、InternVL、LLaVA 一个都没有。"current VLMs" 这种大词用在只测了一家的实验上，claim 的外推性被自家 vision module 的设计高度 bias。
3. **Image+Text < Text-only** 这个 headline finding 的 **intervention** 不够：作者把它解释成"visual token 注入噪声"，但没做 **attention/attribution 分析**证明模型确实"看"了 visual token。可能的 confound：multimodal prompt 的 system prefix 触发了不同的 behavior mode（如 multimodal 下模型更"话痨"更容易跑题）、或 image token 消耗 context 导致 text 部分 attention 分配改变。缺少 causal probe。
4. **Hop 分布极度不平衡**：4+hop 只占 4%（Easy 档完全没有多 hop）。"accuracy 随 hop 衰减"的曲线尾部是样本量 ~10 的点，数字波动大。
5. **OOD 增益过小** (+1.4 到 +2.5)。虽然作者诚实 disclaim 不 claim 普适 transfer，但如果 CrossMath 真揭示了"modality gap 的本质"，针对性训练在同族 math-vision benchmark 上应有更明显的迁移。
6. **Post-training 只用 image-only 数据**，那 text-only 也跟着涨（44 → 76.4）怎么解释？论文没分析。一个合理猜测：CoT trajectories 由 text 端生成，模型主要学到的是 CoT 格式和算术模式，而非"看图推理"。这会弱化"我们修复了 modality gap" 的 claim。
7. **"更强的 visual foundation 才能闭合 gap"** 是结论性断言，但论文本身没提供任何 intervention 证据——它是 narrative 而非被验证的 hypothesis。

### 和过往工作的关系

- 与 [^10] VISTA-Bench（visualized text vs pure text）共享思路，但 VISTA 关注"同一段 text 被 render 成图后变难"，CrossMath 关注的是"vision-first 任务在三模态下的对称比较"，定位不同。
- 与 PuzzleVQA、Bongard 系列的抽象视觉推理一脉相承，优势在 cross-modal equivalence；劣势在视觉形式仍然是合成的 structured grid，不是自然图。
- 针对 MMMU / MathVerse / EMMA 的"entangled"批评扎实——但严格说，它们的设计目标是 end-to-end 多模态能力，和 CrossMath 要做的 modality-specific disentanglement 是不同的 goal，不完全是替代关系。

---
## 论文点评

### Strengths

1. **三模态信息等价的 benchmark 设计**扎实，且人工校 250 道保证了 strict equivalence，这是 claim "rigorous" 的前提。
2. **"Image+Text < Text-only"** 这个 finding 很反直觉且可复现（至少在 Qwen 族内），有 headline 价值。
3. **三路证据排除 perception bottleneck**（OCR 准确 + SFT 有效 + hop-scaling）比单论据 critique 类论文严谨。
4. **Difficulty + Style + Hop 三维分解**让 failure mode 可被定位到 reasoning depth，而非笼统"VLM 不行"。
5. **Position-weighted GRPO reward** 简洁可推广，避免了抽 intermediate answer 的 regex 脆弱性。

### Weaknesses

1. **只测 Qwen 家族**是硬伤，"state-of-the-art VLMs" 的 claim 没有 GPT-4o / Gemini / Claude / InternVL 等背书。
2. **任务形式是 structured symbolic grid**，离真正的自然视觉推理（几何图、物理 diagram、图表）有距离；去 border 掉 5 pt 已暴露模型依赖显式 delimiter。
3. **没有 causal probe / attention 分析**证明 image+text 下模型"确实看了图"，"visual distractor" 的解释停留在 narrative。
4. **Post-training image-only 让 text-only 也涨** 的现象没给解释，削弱"闭合 modality gap"的 claim——更可能是学到了 CoT 格式。
5. **Hop 分布极偏**（1-hop 77%，4+hop 4%），尾部数字的置信区间没给。
6. **OOD transfer 增益小**（+1.4 / +2.5），和"真正修复 visual reasoning" 的叙事不太配。

### 可信评估

#### Artifact 可获取性

- **代码**: GitHub `xuyige/CrossMath` 声明开源（截至写稿未访问）。
- **模型权重**: 未提及是否发布 SFT / GRPO checkpoint。
- **训练细节**: 超参（LoRA rank、lr、epoch、max len、batch、GRPO rollouts/steps）基本完整，但未给随机种子、数据预处理完整配置。
- **数据集**: CrossMath 自构，声明开源 5000 train + 250 eval，4 种 style augment 总计 1000 image 测试样本。

#### Claim 可验证性

- ✅ **Image vs Text Macro Acc gap（92.8 → 12.4）**：Table 2 数据点完整、跨 6 个 Qwen 变体一致。
- ✅ **Hop-scaling 单调衰减**：Table 3 直接展示。
- ⚠️ **"Image+Text < Text-only 是因为视觉输入干扰"**：现象可复现，但归因没 attention/attribution 证据，可能被 context-length / prompt-format confound。
- ⚠️ **"Perception 不是瓶颈"**：三路论证总体有说服力，但"OCR 可被准确完成"是**同一模型在被 explicit 指令 OCR 时**的表现，不等于**在 reasoning flow 里模型隐式地把 symbol 读对了**——这是两个任务。
- ⚠️ **"Current VLMs 普遍依赖 textual shortcut"**：只测 Qwen 家族，外推到 "current VLMs" 过宽。
- ❌ **"闭合 gap 需要更强的 visual foundation"**：narrative 结论，论文本身没做 intervention。

---
## 关联工作

### 基于

- **GRPO** ([^3] DeepSeek-R1)：作为 RLVR 框架，作者替换了 reward 为 position-weighted variant。
- **LoRA** ([^6])：参数高效 fine-tuning。

### 对比 / 批判对象

- **MMMU / MathVerse / EMMA / MMReason**：作者主要的 "entangled multimodal benchmark" 批评对象。CrossMath 以 strict equivalence 区别于它们。
- **PuzzleVQA** ([^2])、**Jigsaw-Puzzles** ([^13])：抽象/空间视觉推理的 benchmark 家族。

### 方法相关

- **VISTA-Bench** ([^10])：visualized text 类论文，共享"rendering to image 会让任务变难"的观察，但 formulation 不同。
- **Bring Reason to Vision** ([^1])、**Compositional Ability Gap in VLR** ([^8])：mechanistic 视角的 perception-reasoning coupling 分析。

---
## Notes

- 🔖 最值得记住的一个数字：Qwen3.5-Plus text 92.8 → image 12.4 → multimodal 74.8。即使信息完全给在 text 里，加图也要掉 18 pt。
- 🔖 **Image-only post-training 让 text-only 也涨** 是个被论文低估的反直觉信号；值得单独做 ablation 验证"学到的到底是 visual grounding 还是 CoT 格式"。
- 🔖 本文的范式若要 generalize 到 agent / embodied 场景，关键在于：自然场景的 visual reasoning 没有显式 delimiter（border），模型对 border 的依赖暗示当前 VLM 对"自发切分视觉 reasoning unit"的能力远不如作者语气所 imply 的那样"只差架构升级"。
- 🔖 Follow-up：我的 VLA / spatial reasoning 方向里，"text hint 中掺入 visual token 是否反而干扰"这个现象值得在 [[2307-RT2|RT-2]] / [[2406-OpenVLA|OpenVLA]] / [[2410-Pi0|π0]] 这类 VLA 上自己复现一把——理论上 action policy 不像 math CoT 那样有 text shortcut，现象应该不同。
