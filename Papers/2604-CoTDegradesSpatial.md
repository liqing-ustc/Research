---
title: "Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs"
authors: [Sai Srinivas Kancheti, Aditya Sanjiv Kanade, Vineeth N. Balasubramanian, Tanuja Ganu]
institutes: [IIT Hyderabad, Microsoft Research India]
date_publish: 2026-04
venue: arXiv preprint
tags: [spatial-reasoning, VLM, embodied-reasoning]
paper: https://arxiv.org/abs/2604.16060
website:
github:
rating: 2
date_added: 2026-04-20
---
## Summary

> [!summary] CoT Degrades Visual Spatial Reasoning
> - **核心**: CoT prompting 在 visual spatial reasoning 上持续**掉点**；8 个 RL 训练的 MRM 里 7 个打不过它们自己的 Qwen2.5-VL-7B backbone
> - **方法**: 17 个模型 × 13 个 spatial benchmark 的大规模统一评测；创新的 *No-Image++* ablation——传入全灰图 + 加 "Cannot determine" 选项，检验模型是否靠 text prior 编造视觉细节
> - **结果**: CoT prompting 平均拉低 3% 准确率；MRM 在 No-Image++ 下仍坚定选错误选项（GThinker 5.55%、Vision-R1 7.29%），确认"幻觉视觉细节"是机制而非偶然
> - **Sources**: [paper](https://arxiv.org/abs/2604.16060)

**Key Takeaways:**
1. **MRM 没有 generalized spatial intelligence**: 8 个 RL 训练的 MRM（ViGoRL、GThinker、Vision-R1、VL-Rethinker、Vision-G1、TreeVGR、R1-Onevision、ThinkLite）中 7 个在 13 个 spatial benchmark 的平均分上低于 Qwen2.5-VL-7B 非 CoT backbone（62.68%）。连专为 spatial 训练的 ViGoRL-Spatial（-2%）和 TreeVGR（-1.57%）也不例外。
2. **CoT prompting 在 spatial 域是反向的 free lunch**: 跨 Qwen / InternVL / LLaVA 三家、3B–72B 尺度、专家 Qwen3-VL-8B-Thinking 都能观察到 CoT 掉点；Qwen3-VL-8B-Thinking non-CoT 比 CoT 高 +0.64%，13 个数据集里 8 个 non-CoT 占优。
3. **"假推理"机制——No-Image++ 揭示的 text-prior 幻觉**: 把图片替换成全灰图并加入 "Cannot determine" 选项（ground truth）后，GThinker、R1-Onevision、Vision-R1、TreeVGR 的准确率塌到 5.55%–11.35%，远不如随机，说明模型在"看不见"时仍然编造坐标、物体位置和 3D 几何描述。
4. **proprietary model 行为差异**: GPT-5、GPT-5-nano 也出现 CoT 掉点；GPT-5-mini trace 只有 ~350 字符，Qwen3-VL-8B-Thinking 有 ~3600 字符——作者假设**简洁非反思型 trace** 能缓解 hallucination。

**Teaser. Figure 1 展示 CoT vs Non-CoT 在 MRM 和 backbone MLM 上的普遍掉点。**

![](https://arxiv.org/html/2604.16060v1/x1.png)

Figure 1: (Left) 开源 MRM 的 CoT vs Non-CoT 对比，6/8 模型 non-CoT 占优；(Right) 多 backbone、多 scale 的 MLM 在 13 benchmark 上的平均 accuracy，CoT（左柱）普遍低于 non-CoT（右柱）。Qwen3-VL-8B-Thinking 作为专为增强 spatial perception 而训练的模型也未能逃脱。

---

## 1. Motivation 与问题设定

CoT / RL-trained "System 2" reasoning 在 math 和 logic 上取得了显著进展，但 spatial reasoning 对 grounding、几何直觉和精确 localization 的要求本质上和文本推理不同——是否 text-centric reasoning 能迁移到 spatial intelligence？作者指出两个 structural observation：

1. **MRM 的原论文评测几乎都在 math-heavy、非 vision-centric 数据集上**（见 Table 2）。比如 Vision-R1 的评测集是 MathVista / MMStar / ChartQA / MMEsum，Vision-G1 是 MathVista / MMMU-Pro / MMStar / ChartQA——这些数据集可能被 text reasoning 所吃掉，但对 spatial 几乎无 signal。
2. **没有一个统一的、公平的 spatial 评测框架**。不同 MRM 用不同 CoT prompt、不同 decoding、不同 judge，彼此之间不可比。

**Table 2. 现有 MRM 原论文的评测集，暴露 vision-centric 的缺位。**

| Baseline | Paper-Reported Datasets |
| --- | --- |
| GThinker-7B | MMStar, RWQA, MMMU-Pro |
| R1-Onevision-7B | MathVision, Mathvista, Mathverse |
| ViGoRL-7B-Spatial | SAT-Val, BLINK |
| VL-Rethinker | MathVision, MMMU-Pro, MEGA |
| Vision-G1 | MathVista, MMMU-Pro, MMStar, ChartQA |
| Vision-R1 | MathVista, MMStar, ChartQA, MMEsum |

---

## 2. 方法：统一评测 + No-Image++

### 2.1 Baselines（17 个模型）

- **Qwen2.5-VL-Instruct** × {3B, 7B, 72B}：MRM 的共同 backbone
- **8 个开源 MRM** 全部基于 Qwen2.5-VL-7B-Instruct：GThinker-7B、R1-Onevision-7B、ViGoRL-7B-Spatial、VL-Rethinker-7B、Vision-G1、Vision-R1-7B、TreeVGR-7B、ThinkLite-7B（其中 ViGoRL-Spatial 和 TreeVGR 显式训练 spatial reasoning）
- **Qwen3-VL-8B-Thinking**：显式增强 spatial perception 的新模型
- **其他 backbone**: InternVL3-8B、InternVL3.5-38B、LLaVA-v1.6-Mistral-7B、LLaVA-OV-Qwen2-72B
- **proprietary**: GPT-4o、GPT-4.1-mini、GPT-5、GPT-5-mini、GPT-5-nano

### 2.2 Datasets（13 个 spatial benchmark）

分两类：
- **Static 2D**（planar 关系，single-image）: BLINK, CV-Bench2D, MMVP, RealWorldQA, SpatialBench, VSR, V*Bench
- **3D / dynamic**（geometry, depth, multi-image, temporal）: 3DSRBench, CV-Bench3D, MindCube, MMSIBench, [[2604-OpenSpatial|OmniSpatial]], SAT-Real

覆盖的 spatial facet tag：REL（object-object 关系）、DEP（depth）、ORI（orientation）、LOC（localization）、SIZ（scale）、CNT（counting）、3D（3D geometry）、MV（multi-image）、TMP（motion/dynamics）、EGO（ego/allo centric）、INT（interaction）、ATT（attribute）。

### 2.3 Evaluation Protocol

- **统一 MCQ 格式**，遵循 VLMEvalKit：`Question:... Options: A. ... B. ... Please select the correct answer (letter and option text)`
- **两种 prompt**：
  - *Base / non-CoT*: `"You are a spatial-reasoning assistant. The user asks a question, and the Assistant solves it."`
  - *CoT*: base + `"First output the thinking process in <think></think> tags and then output the final answer in <answer></answer> tags."`
- **每个 MRM 都用它们自己 training 时的 custom CoT prompt**（不用简化版），避免 "prompt-mismatch" 混淆。Appendix 里作者验证 custom > simple（GThinker 62.52 vs 59.57，Vision-G1 63.26 vs 62.06）
- **Inference**: vLLM 0.10.0, 4×A100, batch 16, max_new_tokens 32768, ctx 32768, bf16, greedy (T=0), 3 seeds
- **LLM-as-judge**: Qwen3-30B-A3B-Instruct-2507，用 GPT-4o 复核得到 Cohen's κ > 0.99（MCQ 场景下 non-reasoning judge 已足够）

### 2.4 No-Image++ Ablation（关键 novelty）

两种 variant：
- **No-Image**: 把图片替换成同尺寸全灰图，其他不变。测量模型"丢掉视觉信息"时还能猜对多少——如果显著高于 random guess，说明存在 text-shortcut。
- **No-Image++**: 在 No-Image 基础上，再给选项里追加一个 "Cannot determine from the image" 选项并指定为 ground truth。这个设计巧妙——**强迫模型要么承认看不见、要么硬编造**。

---

## 3. Results

### 3.1 CoT 降低 spatial accuracy（发现 1）

- Figure 1 (left): 开源 MRM 中 6/8 在 non-CoT 下更强；GThinker 在 non-CoT 下掉 -23.14%，因为不加 `<think>` 标签时它生成退化输出（`<tool_call>\n\n\n...` 一直重复），这是 SFT 过拟合 CoT 格式的后遗症。
- Figure 1 (right): Qwen / InternVL / LLaVA 三家 backbone、3B–72B 尺度下，CoT 平均掉 3%。
- Qwen3-VL-8B-Thinking: non-CoT 比 CoT 高 +0.64%（baseline ~65%），13 个数据集里 8 个 non-CoT 占优（Table 9）。

### 3.2 MRM 打不过 backbone（发现 2）

**Table 1 (合并). 8 MRM vs Qwen2.5-VL-7B backbone 在 13 benchmark 上的 accuracy（%）。**

| Model | Avg. (13 benchmarks) | Δ vs backbone (62.68) |
| --- | ---: | ---: |
| **Qwen2.5-VL-7B (non-CoT backbone)** | **62.68** | — |
| Qwen2.5-VL-7B (CoT) | 59.68 | -3.00 |
| Vision-G1 | 63.26 | **+0.58** |
| ThinkLite-7B | 62.61 | -0.07 |
| GThinker-7B | 62.52 | -0.16 |
| TreeVGR-7B | 61.11 | -1.57 |
| VL-Rethinker-7B | 60.99 | -1.69 |
| ViGoRL-7B-Spatial | 60.68 | -2.00 |
| Vision-R1 | 58.86 | -3.82 |
| R1-Onevision-7B | 46.88 | -15.80 |

- 唯一超过 backbone 的 Vision-G1 在 No-Image++ 里表现也最差（25.28%，意味着幻觉倾向最严重），作者怀疑它的提升来自 dataset shortcut，不是 grounded reasoning。
- R1-Onevision 掉分最严重（-15.8%），应该是它用 "Modality-Bridging"（把图 caption 化 → 喂给 DeepSeek-R1 生成 cold-start 推理数据）带来的结构性问题——完全是 text-only reasoning distill，喂给一个视觉任务基本废掉。

### 3.3 MRM 靠 text prior 编造视觉细节（发现 3，最重要）

**Table 3. No-Image / No-Image++ ablation（平均准确率 %）。**

| Models | Random | Qwen2.5<sub>cot</sub> | Qwen2.5 | GThinker | R1-Ov | ViGoRL | VL-Re. | Vision-G1 | Vision-R1 | TreeVGR | ThinkLite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-Image | 38.83 | 37.45 | 38.59 | 44.17 | 28.10 | 43.18 | 41.26 | 44.46 | 41.15 | 41.91 | 42.48 |
| No-Image++ | – | 43.40 | **76.41** | 5.55 | 11.22 | 30.95 | 47.73 | 25.28 | 7.29 | 11.35 | 36.00 |

两个信号放在一起极具破坏力：

- **No-Image**: MRM 平均 40+%，都高于 random（38.83%）——说明 text-only shortcut 确实存在，即使没图也能靠 question + option + world knowledge 猜对不少。特别是 GThinker (44.17%)、Vision-G1 (44.46%)。
- **No-Image++ 是 smoking gun**: non-CoT 的 Qwen2.5-VL-7B backbone **76.41%** 会乖乖选 "Cannot determine"，但**所有 MRM 都塌得一塌糊涂**——GThinker 5.55%、Vision-R1 7.29%、TreeVGR 11.35%、R1-Onevision 11.22%。这些模型在 CoT 训练中学会了"无论如何都要给出有内容的推理链"，于是在看不见图的情况下继续**编造坐标和 spatial relation**：

> Question (CVBench2D): "Where is the cave located with respect to the trees?"
> 
> ViGoRL output (blank gray input): `<think> The cave entrance appears to be at the bottom of the image, while the trees are higher up, suggesting that the cave entrance is below the trees. (300, 450) ...The cave entrance is clearly situated at the bottom of the image, which is below the trees. (250, 550) </think> <answer> below </answer>`

这是一个 **hallucination**: 模型"看到"了根本不存在的 cave，甚至给出坐标 (300,450)、(250,550)；它的答案基于 world prior（"洞穴通常在树下方"）而非像素证据。

### 3.4 Proprietary Models（发现 4）

**Table 4. GPT 家族的 CoT vs Non-CoT。**

| Model | CoT | Non-CoT | Δ |
| --- | ---: | ---: | ---: |
| GPT-4o | 65.55 | 65.05 | +0.50 (CoT) |
| GPT-4.1-mini | 67.79 | 67.40 | +0.39 (CoT) |
| GPT-5 | 69.00 | **69.65** | +0.65 (Non-CoT) |
| GPT-5-mini | 69.86 | 69.78 | +0.08 (CoT) |
| GPT-5-nano | 60.63 | **61.86** | +1.23 (Non-CoT) |

关键定性观察：
- **trace 长度**: GPT-5-mini ~350 chars vs Qwen3-VL-8B-Thinking ~3600 chars
- **reflection 频率**: proprietary trace 里几乎没有 "wait" / "let me reconsider" 的自我反思 loop；开源 MRM trace 里这类 token 极多

作者 hypothesis：**冗长 + 反思式 trace 放大了 hallucination 的风险**；简洁、单向的 trace 反而更 grounded。这个假设和 No-Image++ 的现象一致——越愿意"生成 reasoning 文本"的模型越容易在无视觉证据下编造。

---

## 4. Takeaways 与未来方向

作者给出的两个建设性方向：
1. **Test-time visual verifier**: 每个 reasoning step 都对齐 image evidence，发现 visual hallucination 就回溯
2. **Visual process reward model**: 训练阶段奖励 "grounded, perception-first" 的推理

---

## 论文点评

### Strengths

1. **问题 formulation 锋利、反直觉**：Math/logic 上 CoT + RL 是主流 narrative，作者敢在 spatial 上亮出 "CoT degrades" 的对立论点，且 17 × 13 的 matrix 足够大到让人无法用 "样本少" 反驳。
2. **No-Image++ 是实验设计上的亮点**：这是全文最硬的 contribution。单独一个 No-Image ablation 可以被反驳为 "shortcut 不等于 hallucination"；追加 "Cannot determine" ground-truth option 是干净的 falsification test——backbone (76.41%) 和 MRM (5–11%) 之间的 gap 很难用别的假设解释。Qwen2.5-VL backbone 能拿到 76.41% 本身就证明了 "能诚实承认看不见" 是模型的 intrinsic capability，是 CoT/RL 训练**破坏**了它，而不是 base 就没有。
3. **Confounder 控制到位**：每个 MRM 用自己的 custom CoT prompt、统一 VLMEvalKit、LLM judge 与 GPT-4o κ>0.99、3 seeds 报 std。MRM 都基于 Qwen2.5-VL-7B 同一 backbone，避免 backbone 差异混进去。
4. **GThinker 的 -23.14% 失败模式解释得很细**：模型在 non-CoT 下生成 `<tool_call>\n\n\n...` 退化输出——这是 SFT 过拟合格式的直接证据，而非 "spatial 能力差"。这种诚实地拆解 failure case 的作风少见。

### Weaknesses

1. **Mechanistic explanation 浮在 behavioral level**：所有结论都来自 input-output pattern（text trace 长度、No-Image++ 准确率），**没有 attention/feature-level 分析**——CoT token 是否真的抑制了 vision encoder 的 output？cross-modal attention 是否在 CoT 模式下塌缩？没有这些，"hallucination from textual priors" 仍停在现象学，距离 mechanistic claim 还有一步。
2. **"Reasoning trace 的冗长 vs 简洁" 只是 correlational hypothesis**：GPT-5-mini trace 短 → hallucination 少，这是 observation，不是 cause。真正的实验设计应该是**控制 open-source MRM 的 max generation length 到 ~350 chars**，看掉点是否恢复——作者没做。
3. **13 benchmark 里有一半是 MCQ，MCQ 本身就会放大 text-shortcut**：spatial MCQ 里选项已经把 "possible spatial relation 的 candidate set" 给你了，模型完全可以不看图只从 options 里挑最 plausible 的——这就是 No-Image 40%+ 的根源。要让 claim 更 strong，作者应该在 **open-ended numerical/coordinate output** 任务（比如 distance estimation、box prediction）上重做同样 ablation，看 CoT 是否仍然退化。现在的结论严格来说是"在 MCQ-spatial 上 CoT 掉点"。
4. **没对 "reasoning budget" 做 sweep**：CoT 掉点 3% 可能是因为 max_new_tokens=32768 让模型跑偏。Qwen3-VL-8B-Thinking 的 -0.64% 和 proprietary 的混合结果都暗示 **budget/length 可能是真正的 confounder**。一个 budget sweep（512 / 2K / 8K / 32K）会把 causal story 讲清楚。
5. **Table 1 的"7/8 MRM 输给 backbone"可能过度 generalize**：注意 R1-Onevision 掉 -15.80% 把平均拉得很难看。它的训练方法（Modality-Bridging：image→caption→DeepSeek-R1 generates CoT→SFT）是所有 MRM 里最 text-biased 的，本来就不该指望它在 spatial 上好。剔除这个 outlier 后，真正"7/8 输"的 narrative 需要重新审视——实际上 6 个 MRM 在 Δ<-2% 以内，边际效应很小。
6. **没有 spatial-native 训练范式的 positive control**：全文结论"需要 vision-centric reasoning"完全基于 negative result。一个有说服力的 follow-up 是训一个 visual-verifier-augmented MRM（论文本身在 conclusion 提了），哪怕 10K 样本的 pilot 也好——缺了这一笔，论文停在 "complaint paper" 的水平，而不是"reformulation paper"。
7. **No-Image++ 的 ground truth 设计可能对 non-thinking MLM 过分友好**：Qwen2.5-VL-7B 76.41% 的超高分，部分原因是 non-CoT 模型倾向于快速选 "Cannot determine"（也就是 "我不知道"），本来就是 non-CoT 的 safe default。No-Image++ 更像是 "CoT 模型被迫过度自信"而不是"backbone 特别会处理缺失信息"——这个对比的 framing 需要更谨慎。

### 可信评估

#### Artifact 可获取性
- **代码**: 未说明（论文未提供 GitHub link，可能在正式发表时补）
- **模型权重**: 全部使用已发布的开源 checkpoints（Qwen2.5-VL, InternVL3/3.5, LLaVA-OV, 8 个 MRM），proprietary 通过 API
- **训练细节**: 无新训练，评测配置完整披露（vLLM 0.10.0, 4×A100, bs=16, T=0, 3 seeds, max_new_tokens=32768）
- **数据集**: 13 个 benchmark 全部开源，可按论文复现

#### Claim 可验证性
- ✅ **"CoT 平均掉 3%"**: Figure 1 + Appendix Table 7 逐 dataset 数字齐全，可复现
- ✅ **"8 个 MRM 中 7 个打不过 backbone"**: Table 1 直接读数，实验 grounded
- ✅ **"No-Image++ 下 MRM 坍塌"**: Table 3 数字 + Figure 2 定性 trace 共同支撑，是全文最强 claim
- ⚠️ **"CoT hallucination 的根源是 textual prior"**: behavioral 证据充分，但缺 mechanistic 分析；这个 claim 算 well-motivated 推测
- ⚠️ **"简洁 trace 是 proprietary model 的秘诀"**: 仅基于 character-count correlation，没有 length-intervention 实验；属于 hypothesis 而非 conclusion
- ⚠️ **"MRM 没有 generalized spatial intelligence"**: 在 13 个 MCQ benchmark 上成立；open-ended spatial task 未测，generalization claim 越界

---

## 关联工作

### 基于
- **VLMEvalKit**: 统一评测框架来源
- **Qwen2.5-VL-7B**: 所有 MRM 的共同 backbone

### 对比（被评测的 MRM）
- **ViGoRL-Spatial** (May'25): MCTS-warmstart + GRPO with coord grounding，是最 spatial-specific 的 MRM 之一
- **GThinker** (Jun'25): 显式 `<vcues>` 标签 + rethinking 机制
- **Vision-R1** (Mar'25): Modality-bridging cold-start + PTST 渐进抑制 length
- **Vision-G1** (Aug'25): Multi-round RL with IF-based data curation
- **TreeVGR** (Jul'25): Traceable grounded reasoning
- **VL-Rethinker** (Apr'25): Self-reflection via RL

### 方法相关
- **Cambrian-1** (Tong et al. 2024): "vision-centric" 论点的来源，本文立场与之一致
- **Eyes Wide Shut / MMVP** (Tong et al. 2024): MLM 视觉 shortcoming 的先声，本文在 RL-era 的继承
- **[[2604-OpenSpatial|OmniSpatial]]**: 被本文作为 13 个 benchmark 之一使用

### 对立/互补
- **R1-style RL post-training 的主流 narrative**（DeepSeek-R1, Vision-R1 等）：主张 CoT + RL 提升 reasoning；本文给出的反例限定在 spatial MCQ 域
- **"Simple, scalable, generalizable" 原则**: 本文结果支持一个读法——text-only CoT scale 到 visual 可能是不 generalizable 的方法

---

## Notes

- 这篇论文在我自己的 research agenda 上的价值：**对 "用 RL 做 spatial reasoning" 这个方向敲了一记警钟**。如果我们要做 VLA / spatial intelligence 里的 reasoning，单纯照搬 DeepSeek-R1 + GRPO 的 text-centric pipeline 很可能是个陷阱。
- 一个未来实验点：**把 No-Image++ 协议做成一个诊断 benchmark**——任何 claim 自己是 spatial MRM 的模型都应该在这上面跑分。这比 benchmark leaderboard 更能区分 "真 grounded" vs "会编故事"。
- 一个 open question：Qwen2.5-VL backbone 76.41% 的 No-Image++ accuracy 真的意味着它"懂"缺失信息吗？还是只是 non-CoT 的 refusal bias？想验证这个需要在 **有图 + 明显 unanswerable** 的场景下测 backbone 的 "Cannot determine" 选择率。
- 对 GPT-5 和 GPT-5-nano 的 CoT 掉点很意外——通常我们认为 OpenAI 模型 thinking mode 一定更好。这个 signal 很值得单独追踪。
- Figure 1 的 bar chart framing（左 CoT、右 non-CoT）本身是个 rhetorical device；对作者的立场来说正确，但读者要警惕把 3% 的 average gap 过度解读——对每个 benchmark 来说 gap 可能在噪声范围内。Std（3 seeds）大多在 0.1–0.5% 之间，所以 3% 的 mean 差确实在噪声之外。
