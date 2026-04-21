---
title: "How Good are Foundation Models in Step-by-Step Embodied Reasoning?"
authors: [Dinura Dissanayake, Ahmed Heakl, Omkar Thawakar, Noor Ahsan, Ritesh Thawkar, Ketan More, Jean Lahoud, Rao Anwer, Hisham Cholakkal, Ivan Laptev, Fahad Khan, Salman Khan]
institutes: [MBZUAI, Linköping University, Australian National University]
date_publish: 2025-09-18
venue: arXiv
tags: [embodied-reasoning, VLM, spatial-reasoning]
paper: https://arxiv.org/abs/2509.15293
website: https://mbzuai-oryx.github.io/FoMER-Bench/
github: https://github.com/mbzuai-oryx/FoMER-Bench
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] How Good are Foundation Models in Step-by-Step Embodied Reasoning?
> - **核心**: 提出 FoMER benchmark，专门评估 LMM 在具身决策场景下的 step-by-step 推理能力——既看最终动作答案，也看推理链质量
> - **方法**: 从 9 个已有数据集（Cosmos-R1 子集、Pbench、HRIBench、NYU VINN、Recon、RoboSet）curate 1112 题；用 Qwen2.5-VL-32B 自动生成 QA + reasoning trail，再人工校验剔除 ~12%；GPT-4o / Qwen3-32B 作为 judge 按 10 维 rubric 打分
> - **结果**: 9 个 SoTA 模型评测，OpenAI o4-mini 综合最强；模型在 social navigation 和 human-robot object interaction 上普遍较差；human baseline 比最强模型高 ~30%
> - **Sources**: [paper](https://arxiv.org/abs/2509.15293) | [website](https://mbzuai-oryx.github.io/FoMER-Bench/) | [github](https://github.com/mbzuai-oryx/FoMER-Bench)
> - **Rating**: 2 - Frontier（首个把 reasoning trail 作为一等公民评估的 embodied benchmark，disentangled RA/FA metric 有诊断价值，但 annotation 自循环 + 未纳入 VLA baseline 限制其成为 de facto 标准）

**Key Takeaways:**
1. **Reasoning trail evaluation 是必要的**：作者用 Gemini 2.5 Pro 和 Qwen2.5-VL 的 case 表明，两个模型可以给出同样的最终答案 ("withdraw bolt")，但底层推理路径完全不同——只评最终答案会漏掉这种"对的答案 + 错的理由"的情况
2. **Benchmark 的差异化定位**：相比 Cosmos-R1 / Robo2VLM-1 / OpenEQA / Pbench / ECBench / HRIBench，FoMER 是首个同时具备 (a) reasoning trail annotation、(b) MCQ + TF + Open-ended 三种题型、(c) 10 个任务 × 8 embodiments × 3 robot types 覆盖度的 benchmark
3. **Disentangled metric**：把 final answer accuracy (FA) 与 reasoning accuracy (RA) 分开评分，能区分"理解了但执行错"（Video-R1 高 RA 低 FA）与"猜对了但理由瞎编"（Cosmos-R1 56.83% FA 但 reasoning 维度全输）
4. **Temporal context matters**：在视频子集上，Gemini 2.5 Pro 用整段视频得 69.62%，8 帧得 62.97%，单中间帧只有 51.89%——验证 temporal context 在物理推理里不可省

**Teaser. Reasoning trail 重要性的 motivating example：**

![](https://arxiv.org/html/2509.15293v2/x1.png)

Gemini 2.5 Pro 和 Qwen2.5-VL 对同一段 "withdraw bolt" 视频都给出了正确的最终动作，但 Gemini 的推理链涉及对工具姿态/目标物体方向的几何判断，Qwen 则更多依赖 object identity 的识别——后者在更复杂的 spatial 场景下会更易失败。

---

## 1. Motivation 与定位

### 1.1 问题

具身 agent 在做决策时不只要"对"，还要"安全、空间一致、上下文相关"。当前评估 LMM embodied 能力的工作有两个 gap：

1. **General reasoning benchmark**（CLEVR、VCR、VRC-Bench）不覆盖物理交互特有的 spatial alignment / object affordance / safety awareness
2. **现有 embodied benchmark**（Cosmos-R1、Robo2VLM-1、OpenEQA、Pbench、ECBench、HRIBench）大多只评最终答案，缺少 reasoning trail 标注与评估

> Why this matters：作者用 Figure 1 展示了一个 "right answer, wrong reason" 的 case。如果 benchmark 只看最终答案，会把基于 hallucination 或 pattern matching 蒙对的模型也评高分——这在 safety-critical 的 robot deployment 里是危险的盲点。

### 1.2 Benchmark 横向对比

**Table 1. FoMER vs 现有 embodied / physical AI benchmark**

| Benchmark   | Modality              | Question Types  | Reasoning Trails | # Questions | # Tasks | # Robot Types |
| ----------- | --------------------- | --------------- | ---------------- | ----------- | ------- | ------------- |
| Cosmos-R1   | Videos + Text         | MCQ, TF         | ✗                | 510         | 5       | 2             |
| Robo2VLM-1  | Images + Text         | MCQ             | ✗                | 6676        | 5       | 1             |
| HRIBench    | Image + Text          | Open            | ✗                | 1000        | 5       | 3             |
| OpenEQA     | Videos + Text         | Open            | ✗                | 1600        | 2       | -             |
| Pbench      | Image + Text          | TF              | ✗                | 5636        | 4       | -             |
| ECBench     | Video + Text          | MCQ + Open      | ✗                | -           | 3       | -             |
| **FoMER**   | Videos/Frames + Text  | TF + MCQ + Open | **✓**            | **1112**    | **10**  | **3**         |

> FoMER 在题量上不算最大（Robo2VLM-1 6676、Pbench 5636），但卖点是 **reasoning trail 标注 + 题型 / 任务 / embodiment 多样性**。这种 trade-off 合理——大规模的 unimodal annotation 比小规模带 reasoning trail 的标注便宜得多，但前者天然评不出 reasoning quality。

---

## 2. Benchmark Curation

### 2.1 数据来源

10 个任务、8 个 embodiments（Agibot G1, Widow X, UR5e, Human Demonstrations, Jackal, Hello Stretch, Franka）、3 种 robot type，共 1112 题、758 个视频，curate 自 9 个已有数据集：Cosmos-R1（拆分为 Agibot, BridgeDataV2, HoloAssist, RoboVQA, RoboFail）、Pbench、HRIBench、NYU VINN、Recon、RoboSet。

### 2.2 QA + Reasoning Trail 生成 Pipeline

按数据集是否已有 QA 分两路：

- **无 QA pair 的数据集**（NYU VINN、Recon、RoboSet）：
  1. Prompt **Qwen2.5-VL-32B-Instruct** 列出场景中所有可见物体 + 动态元素 + 交互
  2. 再次 prompt Qwen，基于物体清单 + 视觉场景生成开放式 QA pair + chain-of-thought 推理链
  3. 题型聚焦 physical commonsense / spatial-temporal reasoning / tool use / risk assessment
- **已有 QA 的数据集**：直接 prompt Qwen 基于 (visual, Q, A) 生成 reasoning trail

### 2.3 Manual Verification

人工审核检查：
- Question 是否相关、physically plausible、与 task category 对齐
- Reasoning trail 步骤是否需要增删
- 不对齐 / 过于 trivial 的题被剔除（约 12%）

> ❓ 人工审核细节披露不充分：annotator 数量、背景、agreement rate（IAA）都没说。仅 12% 剔除率是否能保证 Qwen 自动生成的 reasoning trail 质量？尤其 Qwen2.5-VL 自己就是被评测对象之一，让它生成 GT reasoning 再用 GPT-4o 评 Qwen 自己的 reasoning，存在自评偏置风险。

### 2.4 Task Ontology

10 个任务类别：
1. Task completion verification
2. Next-action prediction
3. Action affordance
4. Physical common sense reasoning
5. Robot-centric reasoning
6. Temporal reasoning
7. Tool use and manipulation
8. Social navigation
9. Human-robot object interaction (HROI)
10. Risk assessment

**Figure 2. Dataset distribution & question type composition**

![](https://arxiv.org/html/2509.15293v2/x2.png)

Cosmos-R1 被拆解为它的 5 个 constituent sub-dataset 以暴露题型分布。

**Figure 3. 两个 benchmark 样例**

![](https://arxiv.org/html/2509.15293v2/x3.png)

左：超市 grasping 任务，agent 需根据视觉上下文 + goal 决定 next subtask。右：navigation 任务，需推理地形与障碍物来判断稳定行进路径。每个样例都包含视频帧、问题、最终答案、step-by-step reasoning。

---

## 3. Evaluation Framework

### 3.1 双重打分

每个样例同时评估：
- **Final answer accuracy (FA)**：MCQ / TF 二元打分（0 或 10）；open-ended 用 rubric 子集评 0-10
- **Reasoning accuracy (RA)**：reasoning trail 按 10 维 rubric 评分

### 3.2 10 维 Rubric

**Table 2. Reasoning trail 评估维度**

| Metric                  | Definition                                                  |
| ----------------------- | ----------------------------------------------------------- |
| Faithfulness            | 推理步骤与 ground truth reasoning 的对齐程度                          |
| Spatial Reasoning       | 空间任务（物体放置、导航、坐标系）推理质量                                       |
| Physical Causality      | 物体 / 力 / 过程之间的因果关系推理                                        |
| Safety                  | 推理过程是否考虑安全（碰撞、人机交互）                                         |
| Commonsense             | 是否覆盖 robotics 领域必要的 commonsense                              |
| Hallucination           | 是否包含与源材料无关的虚构推理步骤                                           |
| Redundancy              | 是否有不增值的冗余步骤                                                 |
| Semantic Coverage-Step  | 是否覆盖任务关键语义元素（环境、物体属性、约束）                                    |
| Reasoning Alignment     | hypothesis 与 reference reasoning chain 整体对齐                  |
| Missing Step            | 是否缺失 robotics-specific 的必要推理步骤                               |

每维 1-10 打分，输出 standardized JSON。

### 3.3 LLM-as-Judge

- 主 judge：**GPT-4o (gpt-4o-2024-05-13)**——固定版本号保证可复现
- 备选：**Qwen3-32B**——验证 framework 在不同 judge 下结果一致

### 3.4 Human Validation

**Table 5. Human evaluation 验证 GPT-4o judge 的可靠性**

|                | GPT-4o judge (%) | Human judge (%)        |
| -------------- | ---------------- | ---------------------- |
| Human Baseline | 80.93            | 84.47 ± 0.63           |
| Gemini 2.5 Pro | 60.02            | 65.00 ± 0.31           |

3 名 volunteer 答 50 题（覆盖 MCQ/TF/Open），互评 + 评 Gemini。GPT-4o 与 human judge 给的分数 closely match，且 **human 比 best model 高约 30%**——目前最强 LMM 距离人类水平仍有显著差距。

---

## 4. Benchmarking Results

### 4.1 Setup

- **9 个评测模型**：Video-R1、InternVL3-38B、Kimi-VL-A3B-Thinking、Cosmos-R1、Claude Sonnet 4、Qwen2.5-VL-32B、Grok 4、OpenAI o4-mini、Gemini 2.5 Pro
- **Frame sampling**：默认 8 帧均匀采样；Robofail 长视频（3-5 min）用 32 帧；Pbench/HRIBench 单图直接喂
- **Output limit**：4096 tokens

### 4.2 主表

**Table 3. RA / FA across all sub-datasets**

| Model             | RoboVQA RA/FA | Agibot RA/FA | Robofail RA/FA | HoloAssist RA/FA | BridgeV2 RA/FA | Pbench RA/FA  | RoboSet RA/FA | Recon RA/FA   | NYU VINN RA/FA | HRIBench RA/FA |
| ----------------- | ------------- | ------------ | -------------- | ---------------- | -------------- | ------------- | ------------- | ------------- | -------------- | -------------- |
| Video-R1          | 73.6 / 80.2   | 61.2 / 36.0  | 67.1 / 58.0    | 68.8 / 32.0      | 60.3 / 28.0    | 67.1 / 67.6   | 63.2 / 32.7   | 73.0 / 53.3   | 72.2 / 48.3    | 51.6 / 26.0    |
| InternVL3-38B     | 74.6 / 72.3   | 67.2 / 42.0  | 69.8 / 62.0    | 72.8 / 56.0      | 65.8 / 33.0    | 26.2 / 71.8   | 69.3 / 30.4   | 70.6 / 56.2   | 69.3 / 41.9    | 55.5 / 23.0    |
| Kimi-VL-A3B       | 65.9 / 66.3   | 59.3 / 30.0  | 61.9 / 52.0    | 62.6 / 39.0      | 56.4 / 29.0    | 56.9 / 63.4   | 59.9 / 58.4   | 70.5 / 71.0   | 71.0 / 68.6    | 40.2 / 21.0    |
| Cosmos-R1         | 79.1 / 85.2   | 61.1 / 45.0  | 67.2 / 57.0    | 63.6 / 46.0      | 52.5 / 27.0    | 61.8 / 60.0   | 57.0 / 53.5   | 71.8 / 66.8   | 64.7 / 62.4    | 50.2 / 27.0    |
| Claude Sonnet 4   | 78.3 / 81.2   | 68.2 / 34.0  | 70.8 / 58.0    | 73.6 / 43.0      | 70.4 / 32.0    | 74.8 / 80.3   | 62.4 / 41.3   | 72.0 / 68.8   | 73.3 / 69.2    | 62.1 / 29.0    |
| Qwen2.5-VL-32B    | 83.8 / 72.3   | 70.4 / 40.0  | 73.5 / 60.0    | 71.8 / 52.0      | 69.7 / 36.0    | 76.9 / 83.1   | 69.2 / 47.6   | 76.7 / 70.8   | 79.6 / 63.6    | 51.6 / 33.0    |
| Grok 4            | 72.3 / 63.4   | 71.3 / 34.0  | 70.6 / 59.0    | 75.8 / 36.0      | 66.5 / 36.5    | 76.8 / 71.8   | 68.7 / 53.8   | 73.7 / 69.0   | 69.1 / 55.7    | 64.1 / 35.0    |
| **OpenAI o4-mini**| **79.8 / 70.3** | **74.2 / 53.0** | **77.3 / 61.0** | **80.0 / 44.0** | **71.5 / 33.0** | **79.3 / 84.5** | **75.3 / 63.2** | **82.1 / 72.0** | **78.1 / 64.8** | **65.8 / 40.0** |
| Gemini 2.5 Pro    | 82.7 / 83.2   | 75.0 / 52.0  | 73.9 / 73.0    | 79.9 / 70.0      | 70.3 / 29.0    | 68.7 / 78.9   | 66.6 / 61.7   | 74.1 / 70.3   | 62.2 / 58.7    | 69.2 / 38.0    |

**主要发现**：
- 闭源 > 开源（除 Qwen2.5-VL-32B 与 Grok 4 / Claude Sonnet 4 comparable）
- **OpenAI o4-mini 综合最强**——RA 平均最高且各维度一致
- Video-R1 / InternVL3-38B / Kimi-VL-A3B 处于第二梯队
- **RA 与 FA 不必正相关**：Cosmos-R1 FA 56.83% 还行但 reasoning 维度全输——意味着它"猜得准但讲不清"

### 4.3 题型与任务维度

**Figure 4. Final accuracy by question type**

![](https://arxiv.org/html/2509.15293v2/x4.png)

所有模型都遵循同一规律：**TF > Open-ended > MCQ**。原因：
- TF 题多对应 task completion verification（二元判断容易蒙对）
- MCQ 干扰项设计严格，容易暴露错误推理

**Figure 5. Per-criterion reasoning accuracy breakdown**

![](https://arxiv.org/html/2509.15293v2/x5.png)

OpenAI o4-mini 各维度均匀领先；Qwen / Gemini / Claude 也较均衡。**Cosmos-R1 在所有 reasoning 维度上垫底**——尽管它 FA 还行，说明它 final answer 的"对"很可能来自 pattern matching 而非真实推理。

**Figure 6. Overall RA vs FA**

![](https://arxiv.org/html/2509.15293v2/x6.png)

闭源模型整体领先；OpenAI o4-mini 在两轴上都最强。

**Figure 7. Per-task category 表现**

![](https://arxiv.org/html/2509.15293v2/x7.png)

- Action affordance 最容易（Cosmos-R1 > 80%）
- **Social navigation 与 HROI 最难**——所有模型都掉得厉害，因为这两个需要 social norm 与 convention 的深层理解
- Gemini 2.5 Pro 在大多数任务领先，Qwen2.5-VL-32B 在 physical commonsense / risk assessment / social navigation 上最好

### 4.4 Temporal Context Ablation

**Table 4. Gemini 2.5 Pro 在 video subset 的不同视觉输入设置**

| Setting          | 8 frames | With Video | Middle Frame |
| ---------------- | -------- | ---------- | ------------ |
| Gemini 2.5 Pro   | 62.97%   | 69.62%     | 51.89%       |

**Insight**：
- 喂整段视频 > 8 帧采样 > 单中间帧
- **Single frame → 8 frames：+11 个百分点**——证明 temporal context 在物理推理中不可省
- **8 frames → full video：+6.65 个百分点**——Gemini 对完整 spatio-temporal volume 的处理优于 sparse sampling

### 4.5 Judge 一致性

**Table 6. GPT-4o vs Qwen3-32B as judge**

| Model            | GPT-4o RA / FA | Qwen3-32B RA / FA |
| ---------------- | -------------- | ----------------- |
| OpenAI o4-mini   | 76.34 / 58.14  | 76.11 / 61.62     |
| Cosmos-Reason1-7B| 62.88 / 54.49  | 64.03 / 54.66     |
| Gemini 2.5 Pro   | 72.26 / 60.58  | 71.88 / 62.39     |

两个 judge 给的 RA 差异都在 0.2-1.2 之间——表明 framework 对 judge model 的依赖较弱。

> ❓ 但只比了 3 个被评模型 + 2 个 judge，样本量小。Open-ended 题的 judge 一致性其实是这类 LLM-as-judge framework 最容易出问题的地方，更细粒度的 per-criterion / per-task agreement matrix 才能 nail down 这件事。

---

## 关联工作

### 基于
- [[2503-CosmosReason1|Cosmos-Reason1]]：FoMER 的最大数据来源（Cosmos-R1 被拆为 Agibot / BridgeV2 / HoloAssist / RoboVQA / RoboFail 5 个子集），同时也是被评模型之一
- VRC-Bench (LlamaV-o1)、DriveLMM-o1：同一团队（MBZUAI ORYX）此前的 multi-step reasoning benchmark，FoMER 的 rubric 设计直接继承

### 对比
- Robo2VLM-1 (6676 MCQ)、HRIBench (1000 open-ended)、OpenEQA (1600 open)、Pbench (5636 TF)、ECBench (MCQ + Open)：embodied / physical benchmark 但**均无 reasoning trail 评估**
- VRC-Bench：通用 multi-step reasoning benchmark（4000+ human-verified steps），FoMER 把这个思路 ported 到 embodied 场景

### 方法相关
- [[2406-OpenVLA|OpenVLA]] / RT-2：被作者列为 "applied LMM/VLM to embodied" 的 prior work，但**未被纳入 FoMER 评测**——一个明显的 omission，VLA 在 next-action prediction 上理应是相关 baseline
- Chain-of-Thought prompting (Wei et al.)：FoMER 的 reasoning trail 概念底层依据
- LLM-as-judge：Chen et al. 关于 judge bias 的工作被引为 caveat

---

## 论文点评

### Strengths

1. **Reasoning trail 的 first-class 评估**——在 embodied benchmark 里第一个把 reasoning trail 做到了 evaluation framework 的中心。10 维 rubric 中 Faithfulness / Hallucination / Missing Step 这些设计是从同团队的 LlamaV-o1 / DriveLMM-o1 借来的，但适配到 robotics 后加了 Spatial Reasoning / Physical Causality / Safety 等 domain-specific 维度
2. **Final answer 与 reasoning quality 解耦**——Cosmos-R1 的"FA 还行 RA 全输"案例展示了 disentangled metric 的诊断价值。这是 benchmark design 的真正 insight：单一 metric 会让 "guess-correctly" 的模型混过去
3. **Coverage 多样化**：10 个任务、8 个 embodiments、3 种 robot type 的覆盖，加上从 9 个已有数据集 curate 而非从头收集，避免了 distribution shift 的训练集污染问题
4. **Judge 可复现性**：固定 GPT-4o 版本号 (gpt-4o-2024-05-13)，且用 Qwen3-32B 做交叉验证——比许多用浮动 API 的工作严谨

### Weaknesses

1. **Annotation pipeline 自循环风险**：用 Qwen2.5-VL-32B 生成 GT reasoning trail，然后又把 Qwen2.5-VL-32B 作为被评模型——它在自己生成的 reasoning style 上自然占优。论文提到 Qwen2.5-VL-32B "performs comparably with Grok 4 and Claude Sonnet 4"，这个排名可能受此偏置影响。理想做法是用与被评模型完全 disjoint 的 annotator
2. **Manual verification 细节不足**：12% 剔除率，但 annotator 数量 / 背景 / IAA 都没披露。剩下 88% 的 reasoning trail 质量本质上还是 Qwen 的 trail——human 只是 filter 而非 author
3. **题量偏小（1112）**：相比 Pbench (5636)、Robo2VLM-1 (6676)，FoMER 在每个 task category 上平均只有 ~111 题，有些 sub-dataset 的统计噪声会比较大
4. **缺少 RT-2、π0、OpenVLA 等 VLA 模型评测**：所有评测对象都是通用 LMM，没有评测真正为 robot control 设计的 VLA。如果 benchmark 的初衷是评估 "embodied reasoning"，那 VLA 应该是 most relevant baseline
5. **Reasoning trail 评估的 prompt 没公开**：rubric 在 Table 2 列了，但实际 LLM-judge 的 system prompt + few-shot 示例都没在正文给出。这影响实际复现

### 可信评估

#### Artifact 可获取性

- **代码**: README 提供 GitHub repo（https://github.com/mbzuai-oryx/FoMER-Bench）但当前 README 仅含 abstract / 数据集对比表 / 结果图 + citation，**评测代码与 prompt 模板未在 README 体现**
- **模型权重**: 不适用（这是 benchmark 而非 model）
- **训练细节**: 不适用
- **数据集**: 已发布到 HuggingFace（https://huggingface.co/datasets/Dinura/FoMER）

#### Claim 可验证性

- ✅ **9 个 SoTA 模型的 RA / FA 数字**：在 Table 3 / Table 6 中给出了完整数字 + judge 模型版本
- ✅ **Temporal context 的重要性**：Table 4 的 8 frames vs video vs middle frame ablation 设计干净
- ⚠️ **"OpenAI o4-mini 是 best generalized reasoning"**：基于的是 GPT-4o 作为 judge——judge 与 OpenAI 同源，存在 self-preference bias 的可能。Qwen3-32B 作为 judge 的 cross-check 缓解了一部分但样本只有 3 个模型
- ⚠️ **"Human ~30% 高于 best model"**：human baseline 只用了 3 名 volunteer 答 50 题，统计显著性有限
- ⚠️ **Manual verification 的剔除率 12%**：未披露 IAA / annotator background / verification protocol，难以独立判断标注质量

### Notes

- **对我的研究的相关性（中等）**：作为一个 embodied reasoning benchmark，可作为评估 VLA / VLM 在 spatial / physical reasoning 上的 secondary benchmark。但作为 benchmark 论文，方法贡献本身有限——主要是数据 curation + LLM-as-judge framework 的 engineering
- **真正有意思的 finding**：Cosmos-R1 "FA 还行 RA 全输" 这个反差案例。这说明现有的 reasoning model（包括宣称做 physical reasoning 的）在 trail quality 上很弱——大概率是 RL 训练只 reward 最终答案，没有 reward reasoning faithfulness。可能的 follow-up：reward shaping 时把 reasoning trail quality 也纳入
- **Benchmark 的最大局限是没评 VLA**：如果想用这个 benchmark 来评 [[2406-OpenVLA|OpenVLA]] 或 π0 这类 action-output VLA，需要先解决 "VLA 不输出自然语言 reasoning trail" 的问题——这本身是个有趣的 research question：怎么从 action-only VLA 中诱导出可解释的 reasoning trail？
- **Annotation pipeline 自循环**值得继续追问：可以试着用一个 disjoint annotator（如 GPT-5 或 Claude Opus 4）重新生成一遍 reasoning trail，看 ranking 会不会变

### Rating

**分数**：2 - Frontier
**理由**：这是首个把 reasoning trail annotation + 10 维 rubric 评估做成一等公民的 embodied reasoning benchmark，disentangled RA/FA 的诊断价值（如 Strengths 2 所述的 Cosmos-R1 "FA 还行 RA 全输" 案例）是 benchmark design 的真 insight；但方法贡献主要停留在数据 curation + LLM-as-judge 工程，且 Weaknesses 1/4 指出的 annotation 自循环 + 未纳入 VLA baseline 两个问题使它难以成为 de facto 标准。相比相邻档，它比 Archived 有明确方法学贡献与可扩展的 rubric 框架，但距 Foundation 级（如 OpenEQA 这类已被广泛沿用的评测基线）还差社区采纳度的时间检验——发布于 2025-09，尚未显示显著的下游 baseline 采用信号。

