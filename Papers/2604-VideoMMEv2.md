---
title: "Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding"
authors: [Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, Yongkang Xie, Xiawu Zheng, Xue Yang, Haoyu Cao, Yunsheng Wu, Ziwei Liu, Xing Sun, Caifeng Shan, Ran He]
institutes: [Nanjing University, Tencent, NTU S-Lab, CASIA]
date_publish: 2026-04-06
venue: arXiv preprint
tags: [video-understanding, video-LLM, VLM]
paper: https://arxiv.org/abs/2604.05015
website: https://video-mme-v2.netlify.app/
github: https://github.com/MME-Benchmarks/Video-MME-v2
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] Video-MME-v2
> - **核心**: 接续 Video-MME 的下一代视频理解 benchmark，800 视频 + 3200 题，目标是诊断 video MLLM 的鲁棒性与推理一致性，而不是再次拼平均准确率。
> - **方法**: (1) 三级递进能力 hierarchy（信息聚合 → 时间动态 → 复杂推理）；(2) Group-based 题组（consistency / coherence）+ non-linear scoring（quadratic + first-error truncation）；(3) 12 annotators × 50 reviewers × 3300 人时质量管控。
> - **结果**: 最佳模型 Gemini-3-Pro 49.4 vs. 人类 90.7；开源最强 Qwen3.5-397B-A17B-Think 仅 39.1。Thinking 模式在带字幕时增益明显、纯视觉下经常退化，暴露当前 reasoning 过度依赖文本先验。
> - **Sources**: [paper](https://arxiv.org/abs/2604.05015) | [website](https://video-mme-v2.netlify.app/) | [github](https://github.com/MME-Benchmarks/Video-MME-v2)
> - **Rating**: 2 - Frontier（video understanding 方向近期必跑的 benchmark：group non-linear scoring 是新且锐利的方法学贡献，同时尚未成为 de facto 标准，位于前沿而非奠基层）

**Key Takeaways:**
1. **Per-question accuracy 系统性高估能力**：换成 group-based non-linear scoring 后，原本看起来差距不大的模型分化明显（如 Gemini-3-Pro 平均 acc 66.1 → Non-Lin Score 49.4，相差 17 点）。
2. **Hierarchical bottleneck**：高层推理失败往往是底层视觉聚合 / 时序建模错误的传导，而非 reasoning 模块本身不行——意味着只堆 thinking 解决不了问题。
3. **Thinking ≠ free lunch**：在 wo-subtitle 设定下 KimiVL-16B、Qwen3-VL-8B 等启用 think 反而下降（最大 -3.3 / -4.0），证明现有 reasoning chain 过度倚重语言锚点，缺少真正的视觉中心 reasoning。
4. **Scale 可以部分补偿能力缺口**：Qwen3.5-397B-A17B-Think (512 frames, 仅 C2+C3) 39.1 略胜 MiMo-v2-Omni (C1+C2+C3) 38.6；但完整能力组合仍是高分模型的共性。
5. **数据策略对 leak 很认真**：>80% 视频是 2025 年后发布，~40% 在 2025-10 后；并以 Gemini-3-Pro text-only 作为 baseline 剔除可被语言先验解掉的题。

**Teaser. Three-level hierarchy + Non-Linear leaderboard.** Video-MME-v2 把视频能力切成三层（L1 信息聚合、L2 时序动态、L3 复杂推理），右侧按 Non-Lin Score 排名，平均 acc 仅作参考，凸显 group-based 评测的鉴别力。

![](https://arxiv.org/html/2604.05015v1/x1.png)

---

## Motivation

作者的论证链条很直白：

1. **现有 video MLLM benchmark 趋于饱和**：Video-MME (CVPR'25)、MVBench、LongVideoBench 等被刷高，但模型在真实复杂视频任务上还远不够。
2. **饱和的根因有两个**：(a) 评测维度被切成孤立 task，缺乏覆盖 perception → reasoning 的渐进 taxonomy；(b) 用 per-question accuracy 评估容易被运气和 partial credit 蒙混过关，无法测 consistency 与 coherence。
3. **解法**：渐进 hierarchy + group-based 题组 + non-linear scoring。

> ❓ 关于 (a)，"现有 benchmark 缺乏综合 hierarchy" 的判断对 VideoMMMU、MMVU 这类已经显式分层的工作其实不完全成立——这里的 v2 真正的差异是 **三层设计 + group 评分** 这套组合，而不是单论 "有 hierarchy"。

---

## Benchmark Design

### Progressive Capability Hierarchy

三层、12 个子类、>30 task type，问题在三层之间均匀分布。

- **Level 1 — Visual Information Aggregation**：识别+整合特定时刻的信息。三个 aspect：Visual Recognition、Cross-Modal Consistency（如 tone-mood 对齐）、Basic Counting & Calculation。
- **Level 2 — Temporal Dynamics**：建模事件随时间演化。Action & Motion、Sequential Ordering、Causal Reasoning。
- **Level 3 — Complex Reasoning**：模拟真实认知任务。Narrative Understanding（plot twist、隐喻、非线性叙事）、Social Dynamics、Physical World Reasoning（counterfactual + 物理约束）。

这套分层的关键是 "渐进"——L3 的题目期望模型先解 L1/L2 再解决高层推理，从而暴露 hierarchical bottleneck。

### Group Type Definition

每个 group 4 道题，分两种 group 类型：

- **Consistency-Based Group**（breadth + granularity）：测同一能力的不同 facet 与不同粒度。例：spatial understanding 内同时测 object localization consistency 和 relative motion；fitness 视频里既问 global 动作顺序也问单个动作内的子动作顺序。
- **Coherence-Based Group**（reasoning chain）：把一个复杂推理拆成有依赖的 sub-question 序列。论文给的例子是 "假死推理"——依次问视觉直接线索 → 反常细节 → 行为目的 → 最终结论，模拟人类一步步推。

> ❓ 这种 coherence chain 的 "答案前后存在依赖" 的要求在数据生成阶段如何被严格保证？论文只在 high-level 描述了 annotation 流程，没有量化 "前置题错了，后续题不可能 ground-truth 推出" 的比例。

### Metrics — Group-Level Non-Linear Score

**Average Accuracy**（参考）：

$$
\mathrm{AvgAcc}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\mathbb{I}[\hat{y}_{q}=y_{q}]
$$

**Group Non-Linear Score**（主指标）：

$$
\mathrm{Overall}=\frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}S(g)
$$

- **Consistency group**：$S(g)=(\mathcal{N}/4)^2$，$\mathcal{N}$ 是 group 内 4 道题的正确数。**quadratic suppression** 惩罚孤立猜对、奖励 "全 facet 都对"——4/4 = 1.0，3/4 = 0.5625（而非 0.75），2/4 仅 0.25。
- **Coherence group**：**first-error truncation**。从 Q1 起统计**最长连续正确前缀**作为该组得分，错一步之后即使后面对了也不计——确保只有 "有支撑的推理链" 得分。

这是 v2 的方法核心：用 scoring 而不是 task 设计来消除 "统计 partial credit 拿分" 的现象。

> 个人评论：这两条 scoring 实际把 "猜对几道" 对应的期望分压得很低（8 选 1 → p=0.125，consistency group 期望分接近地板），所以 leaderboard 数字看起来普遍很低（开源全 <40），是设计的直接后果。这没什么问题——但比较跨 benchmark 的 "绝对难度" 时要意识到这是 metric 层面的差异。

---

## Dataset Construction

### Video Curation

- **800 videos**，平均时长 10.4 min，99% < 20 min，53% < 10 min。
- **Recency**：>80% 视频发布于 2025 年后，~40% 在 2025-10 之后——明确为了规避当前 MLLM 的 pretraining leakage。
- **Taxonomy**：4 个 top-level domain（Sports & Competition / Lifestyle & Entertainment / Art & Literature / Knowledge & Education）→ 31 个细类。
- **质量代理**：用 view count 过滤，84.3% > 10K views，94.4% > 1K views。
- **手工去污**：剔除经典电影 / 顶流 KOL 内容，避免训练集 "记忆" 被当成 "理解"。

**Figure 2. Video category distribution.**

![](https://arxiv.org/html/2604.05015v1/figs/imgs/video_category.png)

### Question and Option Design

- 每题 4 questions × 8 options（A–H），随机猜测概率压到 12.5%。
- Q1→Q4 题目和答案长度逐渐增加，对应 reasoning coherence 的难度递增；同一题内 8 个选项长度严格控制，避免 length bias 走捷径。
- **Adversarial distractor**：每题至少 1 个看似合理但与 ground truth 在某个 fine-grained detail 上正面冲突的强干扰项。
- **Frontier-model adversarial test**：annotation 期间用 Gemini-3-Pro 等做实时探测，识别 underspecified premise / language-only shortcut / 弱 distractor，迭代修正。

### Quality Assurance

- **12 annotators + 50 independent reviewers**，每个样本至少 2 个 QC 人审。
- **Text-only baseline**：用 Gemini-3-Pro 纯文本跑一遍，能解出来的题直接剔除——确保多模态必要性。
- **3 rounds 标注交叉 + 多轮独立盲测**。
- **Closed-loop**：任何修改后题目重跑 text-only baseline + blind test。

> 评论：这套流程对 "benchmark contamination + language shortcut" 这两类已知顽疾的应对相当扎实，是 v2 相对其他 benchmark 的硬实力差异之一。

---

## Experiments

### Setup

- **w. sub vs wo sub**：分别带字幕/音频和纯视觉两种设定。Omni 模型（MiMo-v2-Omni、Gemini-3-Pro）在 w. sub 下喂 raw audio。
- **Frame budget**：开源模型按其长上下文能力取 64 或 512 帧；商业 native video 模型按推荐采样率（如 1fps）；Gemini 系列因 API 限制压缩到 60M，GPT-5 取 50 帧。

### Main Leaderboard (节选)

**Table 1. Main Results, w. sub Non-Lin Score 排序。** 完整表见原文。

| Model | Frames | Non-Lin (w/wo sub) | L1 (w/wo) | L2 (w/wo) | L3 (w/wo) | Avg Acc (w/wo) |
| ------------------------------ | ------ | ------------------ | --------- | --------- | --------- | -------------- |
| Human Expert                   | -      | 90.7 / -           | 94.8 / -  | 91.1 / -  | 87.9 / -  | 94.9 / -       |
| Gemini-3-Pro                   | 1fps   | **49.4** / 38.2    | 64.0/43.2 | 50.0/45.4 | 40.6/30.2 | 66.1 / 56.8    |
| Doubao-Seed-2.0-Pro            | 1fps   | 43.3 / 35.2        | 54.4/41.3 | 47.0/42.6 | 34.1/26.2 | 60.5 / 53.1    |
| Gemini-3-Flash                 | 1fps   | 42.5 / 32.9        | 58.3/41.5 | 44.8/37.4 | 31.7/25.0 | 61.1 / 52.4    |
| Qwen3.5-397B-A17B-Think        | 512    | **39.1** / 30.3    | 50.3/35.4 | 41.8/36.5 | 30.7/22.9 | 55.9 / 48.8    |
| MiMo-v2-Omni                   | 1fps   | 38.6 / 29.9        | 52.6/38.7 | 43.1/36.0 | 27.4/20.4 | 56.1 / 47.1    |
| GPT-5                          | 50     | 37.0 / 26.4        | 44.5/32.2 | 39.1/28.6 | 31.1/21.3 | 55.6 / 44.7    |
| Kimi-K2.5                      | 64     | 36.1 / 27.3        | 44.3/30.5 | 40.0/32.8 | 28.5/21.6 | 54.4 / 46.0    |
| Qwen3-VL-235B-A22B-Instruct    | 64     | 25.0 / 16.5        | 30.7/20.6 | 25.2/16.8 | 21.6/13.9 | 43.3 / 33.8    |
| InternVL3.5-241B-A28B-Instruct | 64     | 23.1 / 15.8        | 28.2/18.1 | 23.7/16.3 | 19.6/14.1 | 41.4 / 32.9    |
| LLaVA-Video-72B-Qwen2          | 64     | 17.2 / 11.3        | 21.8/14.8 | 14.8/11.1 | 16.3 / 9.5 | 34.4 / 27.3   |
| LLaVA-Video-7B-Qwen2           | 64     | 9.7 / 7.2          | 15.9/12.7 | 7.4/5.6   | 7.5/5.1   | 24.0 / 19.9    |

最直观的结论：**人类与最强模型差 41 个点**（90.7 vs 49.4），这在饱和度高的 video benchmark 里相当少见。L3 上模型集体崩盘——Gemini-3-Pro w. sub 也只有 40.6。

### Capability Consistency vs Reasoning Coherence

**Figure 6. Q1–Q4 趋势与 mean–variance 分析。**

![](https://arxiv.org/html/2604.05015v1/figs/exps/q1-q4_models_acc_0406.png)

- **Consistency groups**：Q1–Q4 的 acc 大致持平，证明同 group 内难度均衡。Gemini-3-Pro / GPT-5 波动最小，stability 最强。
- **Coherence groups**：所有模型 Q1→Q4 单调下降，强模型下降平滑（说明对 incremental 难度敏感），弱模型下降不规律（更接近随机）。
- **Mean-variance 散点**：Gemini-3-Pro 同时最高 mean 最低 variance；商业模型整体优于开源，但全员距离 human expert 仍有显著 gap。

### Effect of Thinking Mode

**Figure 7. Thinking 模式开关对 Non-Lin Score 的影响。**

![](https://arxiv.org/html/2604.05015v1/figs/imgs/scientific_think_effect_v2.png)

两条主结论：

1. **Text 锚点能解锁 reasoning**：w. subtitle 下开 Think 普遍涨分，例 Qwen3.5-122B-A10B-Think (64 frames) +3.8 / +5.8 (wo / w sub)。
2. **Think 也会引入退化**：wo subtitle 下，Qwen3-VL-8B -0.6，KimiVL-16B 总体 -3.3 / -3.3，L3 上 -4.0 / -3.9。说明当前 reasoning chain 实质是 "语言-中心 reasoning"，缺乏视觉中心证据时 reasoning 会被自己的 prior 带偏。

> 这是论文最有 insight 的一段。结合 hierarchical bottleneck 看，目前 video reasoning 模型缺的是**视觉证据上的 reasoning grounding**，单纯 RL 调 chain-of-thought 效果有限。

### Capability Profiling

**Table 3.** 把模型用 C1 (omni-modal aggregation) / C2 (long-context temporal) / C3 (complex reasoning thinking) 三能力打 tag，对照 Non-Lin Score：

| Model                          | Score | C1 | C2 | C3 |
| ------------------------------ | ----- | -- | -- | -- |
| Gemini-3-Pro                   | 49.4  | ✓  | ✓  | ✓  |
| Gemini-3-Flash                 | 42.5  | ✓  | ✓  | ✓  |
| Qwen3.5-397B-A17B-Think (512)  | 39.1  |    | ✓  | ✓  |
| MiMo-v2-Omni                   | 38.6  | ✓  | ✓  | ✓  |
| Qwen3.5-397B-A17B-Think (64)   | 30.6  |    | ✓  | ✓  |
| Qwen3-VL-235B-A22B-Think (512) | 28.1  |    | ✓  | ✓  |
| Qwen3-Omni-30B-A3B-Think       | 19.5  | ✓  | ✓  | ✓  |
| Qwen3-Omni-30B-A3B-Instruct    | 17.1  | ✓  | ✓  |    |

两个观察：
- **能力组合 synergy**：C1+C2+C3 全齐的模型整体更强，C1（omni-modal）尤其稀缺。
- **Scale 部分补偿**：397B (无 C1) 39.1 ≥ Omni 38.6（有 C1）；同模型 frame 64→512 提 8.5 点，也证实 C2 (long-context) 是关键瓶颈。

### Capability Radar

**Figure 8. Capability radar across Video-MME-v2 dimensions.**

![](https://arxiv.org/html/2604.05015v1/figs/exps/radar_second_head.png)

三个观察：
- Gemini-3-Pro 在 *Frames & Audio* 维度有显著峰值，体现 vision+audio 同步对齐能力。
- 在 *Order* 与 *Video-Based Knowledge Acquisition* 等长视频时序维度上 Gemini-3-Pro 也明显领先。
- 即使 SOTA，*Action & Motion* 与 *Physical World Reasoning* 仍 <30，是接下来值得专攻的子方向。

---

## 关联工作

### 基于
- **Video-MME (CVPR 2025)**: 同团队上一代 benchmark，v2 在 hierarchy + group scoring 上做 fundamental upgrade
- **MME (image domain)**: 引入 augmented yes/no group 测 reliability，是 v2 group 评估的远祖
- **MMBench**: circular evaluation 测 answer stability

### 对比 (其他 video benchmark)
- **MVBench / MotionBench**: 偏 fine-grained action understanding，单维度
- **LongVideoBench / LVBench**: 偏长视频
- **VideoMMMU / MMVU**: 偏复杂推理但缺 group / coherence
- **VideoReasonBench**: 偏推理但仍 per-question scoring
- **Video-TT**: 第一个引入 augmented group 测 consistency 的 video benchmark，但仍是 question 级 augmentation

### 方法相关 (评测的模型)
- **Gemini-3-Pro / Gemini-3-Flash**: 商业 SOTA，omni-modal
- **GPT-5 / Doubao-Seed-2.0-Pro**
- [[2602-KimiK25|Kimi-K2.5]]
- [[2511-MiMoEmbodied|MiMo Embodied]] 系列（同家族 Omni 模型 MiMo-v2-Omni）
- **Qwen3.5 / Qwen3-VL / InternVL3.5 / LLaVA-Video**
- **Video-R1 / VideoChat-R1 / VideoChat-R1.5**: 用 GRPO 改进 video reasoning

---

## 论文点评

### Strengths

1. **Metric 是真正的贡献**：consistency 用 quadratic、coherence 用 first-error truncation 这套 group non-linear scoring 直接把 "靠 partial credit 拼平均" 这一类常见水分挤掉，让 leaderboard 重新具备 discrimination。
2. **Hierarchical bottleneck 是有用的诊断结论**：把 "高层推理弱" 拆解为 "低层 perception 误差传导"，给改进方向（视觉聚合 / 时序建模）指明优先级。
3. **数据流程认真**：3300 人时 + 12+50 人配比 + Gemini-3-Pro text-only baseline 剔除语言可解题 + recency-oriented 抗 leakage，是同类 benchmark 里少见的工程力度。
4. **Thinking 退化的发现实质**：wo-subtitle 下 reasoning 反而掉点这一现象本身比 "benchmark 数字" 更值得圈出——它揭示当前 video reasoning 的 "伪视觉性"。

### Weaknesses

1. **跨模型采样不一致**：Gemini 用 1fps + 60M 压缩，GPT-5 50 帧，开源 64 / 512 帧——虽然各自 "按推荐" 是合理的，但 frame budget 不齐让 capability comparison 在 controlled-variable 意义下不严格。512-frame Qwen vs 50-frame GPT-5 之间的差异多少来自 frame 多少来自模型，没法 clean attribution。
2. **Coherence group 的 "依赖" 靠人工保证**：first-error truncation 的合理性建立在 "前置题错则后续题不应被独立答对" 的强假设上。但论文未量化这个假设的实际成立比例——如果 ~30% 的后续题其实可独立答对，那 truncation 会系统性低估某些模型。
3. **800 videos / 3200 questions 规模偏小**：对 Non-Lin Score 这种非线性指标，per-group 4 题的小样本会让单 group 得分方差较大，跨模型差距 0.5–1 分时统计显著性存疑（论文未给置信区间）。
4. **没有 inter-rater agreement / human expert 失败案例分析**：90.7 是 human expert ceiling，但具体哪些 leaf category 上人也错（如 Physical World Reasoning），没有给出，影响判断 "模型差是因为题难还是题模糊"。
5. **Related work 对 VideoMMMU / MMVU 的差异化稍弱**：论文反复说 "现有 benchmark 缺综合 hierarchy"，但 VideoMMMU/MMVU 本身已有 discipline-level 分层；真正的 differentiator 是 group + non-linear scoring，应该明写。

### 可信评估

#### Artifact 可获取性

- **代码**：评测代码已集成到 VLMEvalKit 与 lmms-eval；GitHub repo 提供数据下载和评测脚本（inference-only，不涉及训练）。
- **模型权重**：N/A（不发布模型，是 benchmark）。
- **训练细节**：N/A。但**评测协议细节较完整**——non-linear scoring 公式、frame budget、subtitle/audio handling 都披露了。
- **数据集**：开源，HuggingFace `MME-Benchmarks/Video-MME-v2`；GitHub `MME-Benchmarks/Video-MME-v2`。

#### Claim 可验证性

- ✅ **Gemini-3-Pro 49.4 vs 人类 90.7 的 gap**：表 1 完整列出，复现路径清楚（VLMEvalKit / lmms-eval）。
- ✅ **Thinking 模式在 wo subtitle 下退化**：Figure 7 + 数值如 KimiVL-16B -3.3 / Qwen3-VL-8B -0.6 都给了具体数字，可独立复现。
- ✅ **Avg Acc 系统性高估能力**：表 1 同时列两个分数，差异显著且可计算。
- ⚠️ **"hierarchical bottleneck"**（高层推理 fail 主要由低层 perception 误差导致）：论文给的证据是 L1/L2/L3 分数关联性，但没有做 controlled ablation（如人工把 L1 错误纠正后看 L3 是否回升），属于强相关但不严格因果。
- ⚠️ **"3300 human-hours / 5 rounds QC"**：annotation 工程描述详尽但难以独立审计；接受为 best-effort 报告即可。
- ⚠️ **跨模型 frame budget 不齐下的排名**：排名本身可复现，但对 "X > Y" 这类结论的 "模型本身能力" 归因需谨慎。
- ❌ 无明显 marketing 修辞——论文明确说自己是 benchmark + 分析，不 overclaim 模型/方法贡献。

### Notes

- **对自己研究的相关性**：作为 video understanding benchmark，提供两点可以借鉴的方法学：
  1. **non-linear group scoring** 这种 "用 metric 对抗 partial-credit 噪声" 的思路可以迁移到任何评估 reasoning consistency 的场景，包括 spatial reasoning / embodied QA。
  2. **Hierarchical 评估** + **Q1→Q4 difficulty progression** 是测 reasoning depth 的好范式，对 VLA / embodied benchmark 设计有借鉴。
- **暗线**：Gemini-3-Pro 与开源模型的 gap (49.4 vs 39.1) 主要在 **L3 + omni-modal**，开源能 catch up 的最快路径不是更大的 LLM，而是 **真的 omni-modal 训练 + audio aware**——这与 MiMo-v2-Omni 的 "小但全 capability" 思路一致。
- ❓ Coherence group 用的 "first-error truncation" 是否会过度奖励 "提前放弃" 的 behavior？理论上 model 答对前几题后即使后面错也保留分数，但若模型有 self-correction 能力（先错后对），这套 metric 会低估它。值得做对比实验。
- **Action**：暂存为 reference benchmark；如果未来要 evaluate 我做的 video / spatial reasoning 模型，优先跑这个；group scoring 思路可吸收到 spatial-reasoning 的 evaluation 设计中。

### Rating

**分数**：2 - Frontier
**理由**：field-centric 看，这是 video understanding 方向当前最值得跑的 benchmark 之一——group non-linear scoring（quadratic + first-error truncation）是真正新且锐利的方法学贡献（见 Strengths #1），且已集成到 VLMEvalKit / lmms-eval 形成可复现路径。但它尚未成为 de facto 标准（Video-MME v1 仍是现役 baseline），规模（800 videos）与 frame budget 不一致（见 Weaknesses #1, #3）也限制其成为 Foundation 级的可比性基石，因此是前沿而非奠基。高于 1 - Archived 的理由：其 metric 思路对 spatial / embodied reasoning evaluation 有直接借鉴价值（见 Notes 的 Action）。
