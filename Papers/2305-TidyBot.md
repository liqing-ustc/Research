---
title: "TidyBot: Personalized Robot Assistance with Large Language Models"
authors: [Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, Thomas Funkhouser]
institutes: [Princeton, Stanford, The Nueva School, Google, Columbia]
date_publish: 2023-05-09
venue: Autonomous Robots / IROS 2023
tags: [LLM, mobile-manipulation, scene-understanding, instruction-following]
paper: https://arxiv.org/abs/2305.05658
website: https://tidybot.cs.princeton.edu/
github: https://github.com/jimmyyhwu/tidybot
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] TidyBot: Personalized Robot Assistance with Large Language Models
> - **核心**: 用 LLM 的 few-shot summarization 把少量 user-specific 物品摆放偏好抽象成通用 rule，作为家庭 cleanup 任务的 personalization 信号
> - **方法**: 用户提供 4–10 条 "object → receptacle" 示例 → text-davinci-003 用 Pythonic prompt 总结成自然语言 rule → 抽出名词作为 CLIP 的 open-vocab label set → 移动机器人按 rule 抓取并执行 pick-and-place / pick-and-toss
> - **结果**: 文本 benchmark unseen object accuracy 91.2%（baseline 78.5%）；真实世界 mobile manipulator 8 个 scenarios × 3 runs 共 240 个物品，整机成功率 85.0%
> - **Sources**: [paper](https://arxiv.org/abs/2305.05658) | [website](https://tidybot.cs.princeton.edu/) | [github](https://github.com/jimmyyhwu/tidybot)
> - **Rating**: 2 - Frontier（prompt-engineering 时代的 landmark 应用论文，IROS 2023 best paper，被 personalization / LLM-for-robotics 后续工作广泛引用为 baseline，但 method originality 有限且未成奠基框架）

**Key Takeaways:**
1. **Summarization-as-generalization**: 把 personalization 问题转成 "few-shot 示例 → 自然语言 rule"，relative to 直接 in-context inference 在 unseen object 上 +12.7pt accuracy（91.2% vs 78.5%）
2. **LLM 输出反哺 perception 词表**: rule 中的名词被自动抽取作为 CLIP 的 candidate label，把开放词表识别问题压成 2–5 类的 closed-set 分类，CLIP 准确率从 52.3% → 95.5%
3. **Pythonic prompt 设计**: 把 examples 表达成 `pick_and_place("yellow shirt", "drawer")` 形式，再让 LLM 续写 `# Summary: ...` 注释——结构化输入产生结构化、易解析的 summary，并复用 LLM 在 code 上的训练分布
4. **Bottleneck 不在 LLM**: 真机 pipeline 中 LLM 决策正确率 100%（在已识别物品上）；瓶颈在 perception（92.5% 定位 × 95.5% 分类 ≈ 88%）和执行（96.2%）

**Teaser. 任务设定示意——机器人按用户偏好把地上的物品搬到 "应该去的位置"。**

![](https://arxiv.org/html/x1.png)

---

## 1. Motivation：personalization 为什么难

家庭整理任务的核心难点在于 **"应该放哪"** 没有标准答案。同样是衬衫，有人放抽屉，有人放衣柜，有人挂衣架。先前工作分两类：

- **指定每个物体目标位置**（pointing gesture / target layout）：autonomy 不足，标注成本不可扩展
- **跨用户平均 + 学习个性化偏好**（collaborative filtering、spatial relations、latent preference vectors）：要采集大规模 crowd-sourced preference dataset，且小数据集难泛化

作者的视角：**LLM 已经在海量文本上隐式学过 "什么物品属于什么类别"、"什么属性可以聚合" 的常识**，这恰好是 personalization 需要的 generalization——不需要额外训练，只要让 LLM 把 4–10 条示例 summarize 成 rule，就能用 commonsense + summarization 把示例延展到未见物品。

> ❓ 这是一个典型的 "把 ML 问题翻译成 LLM prompt 任务" 的早期工作（2023 年）。从今天看更像是 **prompt engineering 的 application paper**，而非 method paper；但 framing—— "summarization 是 generalization 的载体"——确实是个 reusable insight。

---

## 2. Method

### 2.1 Personalized receptacle selection

**Step 1：summarize 偏好 → rule**

把用户示例编码为 Pythonic prompt，让 LLM 续写最后一行的 `# Summary:` 注释：

```
objects = ["yellow shirt", "dark purple shirt", "white socks", "black shirt"]
receptacles = ["drawer", "closet"]
pick_and_place("yellow shirt", "drawer")
pick_and_place("dark purple shirt", "closet")
pick_and_place("white socks", "drawer")
pick_and_place("black shirt", "closet")
# Summary: Put light-colored clothes in the drawer and dark-colored clothes in the closet.
```

**Step 2：用 rule 推理新物品**

把 summary 作为 prompt 前缀，让 LLM 续写未见物品的 placement：

```
# Summary: Put light-colored clothes in the drawer and dark-colored clothes in the closet.
objects = ["black socks", "white shirt", "navy socks", "beige shirt"]
receptacles = ["drawer", "closet"]
pick_and_place("black socks", "closet")
pick_and_place("white shirt", "drawer")
...
```

为什么用 Pythonic 形式？(i) LLM 在 code 上训练量大，prompt 与训练分布更近；(ii) 输出是结构化的，正则就能 parse，避免 free-form 文本歧义；(iii) function name 本身（`pick_and_place`）就是一个语义 anchor。

### 2.2 Personalized primitive selection

同样的 summarization recipe 复用到 manipulation primitive 选择上——把 example 表达成 `pick_and_place("...")` vs `pick_and_toss("...")`，让 LLM 总结成 "Pick and place clothes, pick and toss snacks" 这类规则。这种 **同构的 prompt 模板让 personalization 自然扩展到 manipulation 维度**。

### 2.3 真实机器人 pipeline

**Figure 2. System overview——overhead camera 找物 → 移动到物品 → egocentric camera 拍特写 → CLIP 分类 → LLM rule 选 receptacle 和 primitive → 执行。**

![](https://arxiv.org/html/x2.png)

伪代码核心循环：

$$
\begin{aligned}
S_{\text{recep}} &= \text{LLM.Summarize}(E_{\text{recep}}) \\
S_{\text{prim}} &= \text{LLM.Summarize}(E_{\text{prim}}) \\
C &= \text{LLM.GetCategories}(S_{\text{recep}}) \quad \text{// 抽取 rule 中的名词}
\end{aligned}
$$

之后每一轮：用 ViLD 在 overhead 视角找到最近物品 → 移动 → egocentric CLIP 用 $C$ 分类 → 查 $S_{\text{recep}}, S_{\text{prim}}$ → pick + place/toss。

**关键设计**：rule 里的名词集合 $C$ 直接做 CLIP 的 label 集。不需要人工指定细粒度词表，也不需要 CLIP 处理 open-vocab 推断——只在 2–5 个抽象类别（"light-colored clothing"）之间二选一/五选一，把 perception 问题大幅简化。

硬件配置：holonomic powered-caster 移动底盘 + Kinova Gen3 7-DoF 臂 + Robotiq 2F-85 夹爪；ArUco marker 做 base 位姿，ViLD 做物品定位（receptacle 位置硬编码）。

---

## 3. Experiments

### 3.1 Benchmark

96 scenarios，4 个 room type（living room / bedroom / kitchen / pantry），每场景 2–5 个 receptacles + 4–10 seen examples + 同等数量 unseen evaluation。总计 672 seen + 672 unseen placements，覆盖 87 unique receptacles 和 1,076 unique objects。

排序标准（一个 scenario 可同时 hit 多个）：

**Table 1. Sorting criteria 在 96 个 scenarios 中的覆盖度。**

| Category | Attribute | Function | Subcategory | Multiple |
| --- | --- | --- | --- | --- |
| 86/96 | 27/96 | 24/96 | 31/96 | 17/96 |

### 3.2 主结果与 baseline 对比

**Table 2. Unseen object placement accuracy。**

| Method | Accuracy (unseen) |
| --- | --- |
| Examples only（不 summarize 直接续写） | 78.5% |
| WordNet taxonomy（最近 seen object） | 67.5% |
| RoBERTa embeddings（cosine 最近邻） | 77.8% |
| CLIP embeddings | 83.7% |
| **Summarization (ours)** | **91.2%** |

**Table 3. 按 sorting criteria 拆分。**

| Method | Category | Attribute | Function | Subcategory | Multiple |
| --- | --- | --- | --- | --- | --- |
| Examples only | 80.1% | 72.7% | 75.7% | 77.0% | 81.5% |
| WordNet | 69.1% | 59.8% | 61.4% | 71.3% | 74.1% |
| RoBERTa | 78.6% | 75.5% | 71.8% | 71.7% | 87.5% |
| CLIP emb | 84.6% | 79.8% | 85.5% | 84.7% | 87.9% |
| **Summarization** | **91.0%** | **85.6%** | **93.9%** | **90.1%** | **93.5%** |

最关键对比是 **Examples only vs Summarization**：唯一差异就是中间是否显式产出 rule，单这一步在 unseen 上 +12.7pt。这与 chain-of-thought 同源——让 LLM 显式输出中间推理可以提高最终决策质量。WordNet 在 attribute 和 function criteria 上 fail（59.8% / 61.4%），因为 WordNet 是基于 category 的语义层级，对 "light vs dark"、"workout vs casual" 这种属性维度无能为力。

### 3.3 Ablation

**Table 4. Ablation。**

| Method | Seen | Unseen |
| --- | --- | --- |
| Commonsense（不给偏好，纯常识） | 45.0% | 45.6% |
| Summarization | 91.8% | 91.2% |
| Human summary（人写 rule，oracle） | 97.1% | 97.5% |

- Commonsense baseline 45% 说明 **任务本身需要 personalization**，否则 LLM 凭常识只能猜到一半
- Human summary 97.5% 给 LLM summary 留下了 6pt 的 headroom，说明瓶颈在 summarize 质量而非 grounding

**Table 5. 不同 LLM 的 commonsense vs summarization 表现。**

| Model | CS-seen | CS-unseen | Sum-seen | Sum-unseen |
| --- | --- | --- | --- | --- |
| text-davinci-003 | 45.0% | 45.6% | 91.8% | 91.2% |
| text-davinci-002 | 41.8% | 37.5% | 84.1% | 75.7% |
| code-davinci-002 | 41.4% | 39.4% | 88.6% | 83.2% |
| PaLM 540B | 45.5% | 49.6% | 84.6% | 75.7% |

老模型 seen-unseen gap 大（PaLM 540B: 84.6 → 75.7），原因是它们更倾向于在 summary 里 list 出 seen objects 而不是抽象成类别——summary 不够 abstract → 不泛化。

### 3.4 用户研究

40 participants，每人 24 个 scenario，960 次评估。让用户在 CLIP embeddings baseline vs Summarization 之间选 "更符合 preference 的"。

**Table 6. 用户偏好结果。**

| Method | Category | Attribute | Function | Subcategory | Multiple | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP embeddings | 19.7% | 23.7% | 11.2% | 22.6% | 21.2% | 19.1% |
| Summarization (ours) | 47.4% | 41.9% | 60.0% | 46.1% | 40.6% | 46.9% |
| Equally preferred | 32.9% | 34.4% | 28.8% | 31.3% | 38.2% | 34.1% |

paired t-test: t=9.93 (df=39), p<0.001。Function criteria 上 60% vs 11.2% 差距最大——CLIP embedding 倾向于按 category 聚合（dress pants 和 sweatpants 都归 "pants"），但用户排序逻辑可能是 formal vs casual，这种语义鸿沟 embedding 没法跨。

### 3.5 真实世界结果

8 个 scenarios × 3 runs × 10 objects = 240 次拾取，整机成功率 **85.0%**。Pipeline 各级精度：

| Stage | Success rate |
| --- | --- |
| Overhead camera localization | 92.5% |
| Object classification (CLIP) | 95.5% |
| LLM receptacle + primitive selection | 100% |
| Manipulation primitive execution | 96.2% |

平均每物品 15–20 秒。

### 3.6 VLM 对比

222 张真机 egocentric 图，三种模型 × 三种 vocabulary：

**Table 7. VLM 对比。**

|  | CLIP | ViLD | OWL-ViT |
| --- | --- | --- | --- |
| Summarized categories | **95.5%** | 76.1% | 45.9% |
| Scenario object names | 70.7% | 59.9% | 24.8% |
| All object names | 52.3% | 36.5% | 18.5% |

两个发现：

1. **Vocabulary 越粗精度越高**——把识别压成 2–5 选 1 远比 65 选 1 容易。LLM summarization 自动生成的小词表是这个收益的关键
2. **CLIP 优于 ViLD/OWL-ViT**——后两者会出现 "no detection" 的情况，且作为 detector 派生模型在 classification 上 degrade。CLIP 的 image-wide classification 在 egocentric 视角下因为前景物品占主导反而是优势

> ❓ 但 CLIP 95.5% 是建立在 "egocentric camera 已经把目标物品摆中央" 的强 prior 上的；如果换成 cluttered scene，CLIP 的弱 spatial grounding 会暴露。

### 3.7 真实场景 demos

**Video. Annotated scenario——recycle drink cans, put away other items where they belong。**
<video src="https://tidybot.cs.princeton.edu/videos/IMG_5056-4x-annotated.mp4#t=0.001" controls muted playsinline width="720"></video>

**Video. Toss drink cans into the recycling bin。**
<video src="https://tidybot.cs.princeton.edu/videos/IMG_8817-2x.mp4" controls muted playsinline width="720"></video>

---

## 4. Limitations（作者自陈）

- **Summary failure modes**: (i) 直接 list seen objects 而不抽象（不泛化）；(ii) 把不同 receptacle group 起来（如 top drawer + bottom drawer → drawers），导致选错具体目标
- **真机系统简化**: 手写 manipulation primitives；只用 top-down grasp；receptacle 位置硬编码；过度 cluttered 场景无法工作（机器人不能跨越物品）

---

## 关联工作

### 基于
- **CLIP** (Radford et al., 2021)：开放词表图像分类的 backbone
- **ViLD** (Gu et al., 2022)：overhead camera 物品定位
- **GPT-3 / text-davinci-003** (Brown et al., 2020)：summarization 与 rule 推理
- **Code-as-Policies / [[2204-SayCan|SayCan]] / Inner Monologue** (Liang, Ahn, Huang et al., 2022)：用 LLM 生成机器人控制 / 规划的并行线索

### 对比
- **Object rearrangement benchmarks**（Habitat 2.0, AI2-THOR, ALFRED）：作者强调这些任务里 target location 是预指定的，不涉及 personalization
- **Personalized placement via collaborative filtering / spatial relations / latent preference vectors**：需要大规模 user preference dataset，TidyBot 不需要

### 方法相关
- **Chain-of-thought prompting** (Wei et al., 2022)：summarization-then-act 的 framing 与 CoT 同源
- **Code-as-Policies** (Liang et al., 2022)：Pythonic prompt 形式的直接 inspiration
- **TossingBot** (Zeng et al., 2020)：pick-and-toss primitive 的引用源

---

## 论文点评

### Strengths

1. **Framing 干净**：把 personalization 翻译成 few-shot summarization，不需要数据集、不需要训练，直接用 off-the-shelf LLM。这种 reframing 本身有迁移价值
2. **Closed-loop 的 perception/LLM 协同**：rule 的名词反哺 CLIP 的词表，让 perception 从 open-vocab 退化为小 closed-set——这个设计在多个层面（precision、speed、可解释性）都受益。是一个值得抽象出来的 design pattern
3. **Ablation 严谨**：human-summary oracle 给出 6pt 上界，说明真正的 bottleneck 已经定位到 LLM summarization quality；commonsense baseline 证明 task 确实需要 personalization 而非常识
4. **真机数据完整**：分阶段拆解了 localization / classification / planning / execution 各自的精度，让读者能看清 system bottleneck 在哪

### Weaknesses

1. **Method 是 prompt engineering**：核心贡献是 prompt 模板设计。这在 2023 年是合理 framing，但今天看 "用 LLM summarize examples" 已经是 baseline 操作，方法 originality 有限
2. **真机 scope 窄**：只 8 个 scenarios，每个 scenario receptacle 位置硬编码、物品基本不重叠、地面平整。"85% success" 的语境与 BC/RL 论文里 manipulation 的 85% 完全不同——这里失败基本是 perception/grasp 错，不是 LLM 错
3. **Personalization 概念被简化**：用户偏好只在 "object → receptacle/primitive" 这一映射层。但真实 personalization 还涉及时间、上下文、隐含约束（不要把脏衣服和干净衣服放一起），这些 LLM rule 形式很难表达
4. **未对比 fine-tuning baseline**：既然 summarize 优于 in-context examples-only，那一个自然问题是：能否直接 fine-tune 一个小 LM 在 (examples, placement) pairs 上？没做这个对比

### 可信评估

#### Artifact 可获取性

- **代码**: 已开源（GitHub jimmyyhwu/tidybot），含 server + robot + benchmark + STL 3D-print 件
- **模型权重**: 不需要——使用 OpenAI text-davinci-003 API + 预训练 CLIP / ViLD
- **训练细节**: 无训练；prompt 全文在 Appendix A 给出，temperature=0
- **数据集**: Benchmark 96 scenarios 已开源，含 1076 unique objects + 87 unique receptacles

#### Claim 可验证性

- ✅ **Benchmark 91.2% unseen accuracy**：可在公开 benchmark 复现，prompt 模板在 Appendix A 完整
- ✅ **Real-world 85.0% 成功率**：240 次试验、8 scenarios，样本量足够；分阶段拆解可信
- ⚠️ **"Summarization 是 generalization 的关键"**：实验只对比了 in-context examples vs summarization，没排除 "summary 提供了额外计算 budget"（CoT 效应）的混淆。即 summary 本身可能不是 generalization 的来源，而是更长的 thinking 过程
- ⚠️ **"CLIP 95.5% 优于 ViLD/OWL-ViT"**：是基于 egocentric "目标在画面中央" 的强 prior 评测的；外推到 cluttered scene 不一定成立

### Notes

- **Reusable design pattern**: "LLM 输出反哺 perception 的词表" 这个闭环值得记到 mental model。在任何 "VLM 要识别开放类别" 的场景里，先用 LLM 把 task-relevant concept 集合压缩到 K 类，再让 VLM 在 K 类里分类，比让 VLM 直接 open-vocab 分类要稳得多
- **对 VLA 时代的启示**: TidyBot 是 modular pipeline 的代表（perception + LLM planner + 手写 primitive）。今天的 VLA 路线倾向于 end-to-end，但 TidyBot 揭示的 "personalization rule 可显式表达" 这一性质，被 end-to-end 模型隐式吃掉了——一个值得思考的 trade-off：可解释性 vs scalability
- **Summarization-as-generalization 的边界**: 这套 framing 适用于偏好可以被自然语言 compactly 描述的场景。一旦偏好需要"如果今天周三且上次穿过则放洗衣篮"这种条件复合规则，summarize 就会失败——值得验证 LLM summarize 的复杂度上限
- ❓ **后续问题**: 在多模态 LLM（GPT-4V / Gemini）成熟之后，summarize 这一步可以直接吃图像 examples 而不是 textual examples，prompt 不再需要人工把物品翻译成 "yellow shirt"。会不会因此让真机系统去掉对硬编码 textual preference 的依赖？

### Rating

**分数**：2 - Frontier

**理由**：TidyBot 属于 LLM-for-robotics personalization 方向的 representative 工作（IROS 2023 best paper，Google Scholar 500+ 引用），在 household robot / preference learning / LLM-planner 相关工作中被广泛作为 baseline 或 motivating reference 引用；笔记 Strengths 里的 summarization-as-generalization 和 LLM 输出反哺 perception 词表是可迁移的 design pattern。但没有达到 3 级的奠基/必读位置——Weaknesses 里 method 实质是 prompt engineering，且在 end-to-end VLA 主导的今天，modular LLM-planner 路线已被 [[2307-RT2|RT-2]] / [[2406-OpenVLA|OpenVLA]] / [[2504-Pi05|π0.5]] 等 end-to-end 方法的势头所压过；不足以作为方向主脉络上的奠基工作。
