---
title: "Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning"
authors: [Baining Zhao, Ziyou Wang, Jianjie Fang, Chen Gao, Fanhang Man, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li, Wenwu Zhu]
institutes: [Tsinghua University]
date_publish: 2025-04-17
venue: arXiv
tags: [spatial-reasoning, embodied-reasoning, video-LLM, agentic-RL]
paper: https://arxiv.org/abs/2504.12680
website: https://embodiedcity.github.io/Embodied-R/
github: https://github.com/EmbodiedCity/Embodied-R.code
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning
> - **核心**: 用一个大 VLM 做 perception、一个小 LM 做 reasoning 的解耦协作框架，靠 GRPO + 一个新设计的 logical-consistency reward 在 5K embodied video 样本上 RL 出 slow-thinking 的 embodied 空间推理能力。
> - **方法**: 关键帧抽取 → Qwen2.5-VL-72B 顺序输出 frame-level semantic representation（action / Δinfo / q-related）→ Qwen2.5-3B 用 GRPO 训 think-then-answer，三档奖励（format / accuracy / 一个用 reference model 校验"由 reasoning 反推答案是否一致"的 logical consistency reward）+ 三阶段课程。
> - **结果**: 3B LM + 72B VLM 在 VSI-Bench、UrbanVideo-Bench 8 类任务平均 acc 51.1%，超 OpenAI-o1（+13.9%）和 Gemini-2.5-Pro（+10.3%）；OOD 上 RL 训出的模型在 EgoSchema 接近 Gemini-2.5-Pro，SFT baseline 在 MVBench 反而退化。
> - **Sources**: [paper](https://arxiv.org/abs/2504.12680) | [website](https://embodiedcity.github.io/Embodied-R/) | [github](https://github.com/EmbodiedCity/Embodied-R.code)
> - **Rating**: 2 - Frontier（在 Spatial Intelligence × Agentic RL 交叉点给出"借大 VLM perception + RL 小 LM reasoning"的 cost-efficient recipe，logical-consistency reward 是可迁移的 trick；但方法 scope 偏窄、尚未成为 embodied reasoning 的 de facto 范式，属前沿参考而非奠基。）

**Key Takeaways:**
1. **Perception/Reasoning 解耦换算力**: 把 VLM 当 frozen frame describer、用文本 bottleneck 喂小 LM 做 RL，是在 8×A800-40G 的预算下"借大模型 perception、训小模型 reasoning"的可行 recipe（90 GPU-hours / run）。
2. **Logical-consistency reward 治 reward hacking**: 多选题答案空间小，只用 accuracy + format reward 时 reasoning 与 answer 经常不一致；让 reference model 仅看 question + reasoning（不看视频）复现答案，作为额外 0/1 reward，把 logical-consistency 比例从 46% 拉到 99%。这个 trick 可迁移到任何 answer space 小、易 hack 的 RLVR 任务。
3. **Embodied reasoning 上无 "aha moment"**: 与 math 上 R1 类工作不同，response length 在训练中不发散反而收敛到一个区间，作者把它解释为任务特性（embodied spatial QA 不需要长链推算）。这与 "RL 一定会出 long CoT" 的流行叙事形成反例。
4. **RL > SFT for OOD generalization**: 在 EgoSchema / MVBench 两个 OOD set 上，RL 训出的 3B LM 普遍正向迁移、SFT 在 MVBench 上掉点——再添一例 RL-vs-SFT 的 OOD 优势观察。
5. **同 scale 下 LM-on-text > VLM-on-video for RL reasoning**: 直接对 Qwen2.5-VL-3B 做 RL 收敛到 43.8%，明显低于"VLM-72B 抽语义 + LM-3B 推理"的协作方案；perception 上限制约 reasoning 上限。

**Teaser. Embodied spatial reasoning 任务示例与 slow-thinking 流程**——室内（VSI-Bench）+ 室外（UrbanVideo-Bench）两类 egocentric 视频上，模型需先 think 再 answer。

![](https://arxiv.org/html/2504.12680v1/extracted/6360624/fig/task.png)

---

## 1. Problem Setup

**输入**: 连续 video frames $\mathbf{f}=[f_0,f_1,\dots,f_T]$ + 一道 spatial reasoning 问题 $q$。
**输出**: 一个 answer $a$，以与 ground truth $g$ 语义一致为正确。
**任务范围**: 8 类多选题 spatial reasoning（VSI-Bench 选 Relative Distance / Relative Direction / Route Planning / Appearance Order；UrbanVideo-Bench 选 Landmark Position / Counterfactual / Progress Evaluation / Action Generation），共 5,415 QA / 1,492 videos。

**为什么难**（作者总结的三个 challenge）：
1. **Reasoning 依赖 perception**：连续视觉观测对感知质量要求高，幻觉直接污染推理。
2. **复杂时空关系**：跨帧物体关联 + 与任务相关语义的抽取，纯 SFT 缺乏对 reasoning 过程的监督。
3. **Egocentric 特性**：第一视角强调 observer 与环境关系；视觉观测在时间上是流式产生的；且因运动连续性帧间冗余高——直接喂全部帧给 MLLM 会 token 爆炸且丢失泛化。

---

## 2. Method

**Figure 2. Embodied-R 整体框架**——左半侧是 large-scale VLM 做 perception（Key-frame Extractor + 顺序 frame-level semantic 抽取），右半侧是 small-scale LM 做 reasoning（GRPO 训练 think-then-answer，含三类 reward）。

![](https://arxiv.org/html/2504.12680v1/extracted/6360624/fig/framework.png)

### 2.1 Key-Frame Extractor

用经典 CV pipeline 而非学到的 sampler：

1. ORB 提关键点 + Brute-Force 匹配；
2. RANSAC 估单应矩阵 $\mathbf{M}$；
3. 把 $f_{t+1}$ 四角投到 $f_t$ 坐标系，算重叠多边形面积比 $c$；
4. $c < \varepsilon$ 则 $f_{t+1}$ 入 keyframe，否则跳到 $f_{t+2}$ 继续比；旋转大时自然抽更多帧（视场变化更剧烈）。

> ❓ 阈值 $\varepsilon$ 没在正文给具体数。一个手工 CV pipeline 在 outdoor 无人机场景 / 强烈光照变化 / 模糊抖动下的 ORB 匹配稳健性是个隐忧。

**Ablation 收益**：与无抽帧相比 accuracy 几乎不变，训练时间降 8.7%、单次推理时间降约 1/3。

### 2.2 VLM-based Embodied Semantic Representation

用 Qwen2.5-VL-72B-Instruct **顺序**处理：第一帧描述场景（物体、属性、位置）；后续每帧把 $(f_{k_{j-1}}, f_{k_j}, q)$ 一起送进 VLM，输出三段式语义：

- **Action**：根据帧间变化推断 agent 动作；
- **ΔInformation**：agent 与已知物体空间关系的变化、新出现的物体；
- **q-related content**：与问题相关的物体/信息是否出现在最新视野。

这种 "differential + q-conditioned" 的 caption 设计避免一次塞所有帧 → 也让长视频可处理。最终得到文本序列 $\mathbf{s}=[s_{k_0},\dots,s_{k_n}]$，作为 LM 的输入。

**Equation 1. Frame-level semantic 抽取**

$$
s_{k_j} \sim \psi_\theta(s \mid f_{k_{j-1}}, f_{k_j}; q),\quad j=1,2,\dots,n
$$

> ❓ VLM 是 zero-shot 用的，prompt 也没在正文展开。"由 VLM 自行决定哪些是 q-related" 等于把 attention 这一步外包给一个 frozen 模型——这意味着 reasoning 模型再强，也只能在 VLM 已经留下的"残骸"上推理。72B vs 3B 的不对称在这里很关键。

### 2.3 GRPO + 三类 Reward

对每个 $(q, \mathbf{s})$，rollout $G$ 个 response，用 group-relative advantage 训小 LM $\pi_\theta$（init 自 Qwen2.5-3B-Instruct，作者推荐用 Qwen2.5-VL-3B 的 LM decoder 部分以利用多模态预训练带来的 prior）。目标函数即标准 GRPO（PPO-style clip + KL-to-ref 正则）。

**核心 reward 设计**：

- **Format reward** $r'_i$: 用正则检查输出是否严格遵循 `<think>...</think><answer>...</answer>`。
- **Accuracy reward** $r''_i$: 提取 `<answer>` 内容与 ground truth $g$ 比对，0/1。
- **Logical Consistency reward** $r'''_i$（**本文核心创新**）：

  当 $a_i = g$ 时，把 question $q$ + reasoning $p_i$（**不含视频帧**）喂给 reference model $\pi_\text{ref}$，让它产生 $a'_i$。仅当 $a'_i = a_i = g$ 才给 1 分。

  $$
  r'''_i=\begin{cases}1, & a_i = a'_i = g \\ 0, & \text{else}\end{cases}
  $$

  **意图**：embodied spatial 多选题答案集小（如"左/右/前/后"），accuracy reward 容易把"乱想但蒙对"也强化掉；用 ref model 验证"光看 reasoning 文本能否反推出同一答案"——若不能，reasoning 必然没有真正承载推理逻辑。这是个轻巧的 process supervision 替代品，不需要 process reward model。

总 reward $r_i = \omega_1 r'_i + \omega_2 r''_i + \omega_3 r'''_i$，三阶段课程切换权重：

- **Stage 1 (epoch 1-2)** $\omega = 7:3:0$：先学格式；
- **Stage 2 (epoch 3-4)** $\omega = 3:7:0$：再学正确性；
- **Stage 3 (epoch 5-12)** $\omega = 1:7:2$：最后引入 consistency。

> ❓ 为什么必须分阶段？文中没消融"一开始就给 consistency reward"会怎样。直觉上 stage-3 引入太晚也可能错过关键塑形窗口；分阶段更像工程稳定化。

---

## 3. Experiments

### 3.1 Setup

- **Models**: VLM = Qwen2.5-VL-72B-Instruct（frozen），LM = Qwen2.5-3B-Instruct。
- **Hardware**: 8×A800-SXM4-40G，每次 RL ≈ 90 GPU-hours。
- **Hyper**: lr 5e-7, temp 1.0（推理 0.5），train batch 32, rollout 8, KL coef 0.001, max response 2048, input 6144。
- **Eval**: 5-fold CV；额外做"semantic bias 过滤"——用纯文本 SFT 一个 LM，若它能不看视频就答对的题就剔除（说明题面有捷径），保证测试集真考视频空间理解。

### 3.2 Main Results (RQ1)

**Table 1 (paper Table 3 in text). 8 类任务上 acc**——Embodied-R 平均 51.1%，超 17 个 baseline（含 GPT-4o / Gemini-1.5-Pro / OpenAI-o1 / Gemini-2.5-Pro / Qwen2.5-VL-{3B,7B}-SFT）。其中 vs Qwen2.5-VL-72B 直接推理（34.9%）+1.5×。

![](https://arxiv.org/html/2504.12680v1/extracted/6360624/fig/rader.png)

要点：
- 比 OpenAI-o1 / Gemini-2.5-Pro 高 10%+，说明当前顶尖 reasoning 模型在 embodied spatial 上 generalization 有限；
- 同算力预算下能 fine-tune 的最大 VLM 是 7B 级，被 perception 上限拖累，全面输给"借 72B perception"的 Embodied-R；
- 单纯 72B VLM 也只到 34.9%，加 3B LM 推理 → 51.1%，表明 reasoning 的 RL 后训练贡献远大于"换更大 perception 模型"。

### 3.3 Slow-thinking 行为分析 (RQ2)

**Figure 3. Case Analysis**——RL 后 Embodied-R 出现的四种 human-like reasoning 行为：spatial relationship reasoning（精准描述 self-环境相对位置）、systematic analysis（part-to-whole 拆解）、contextual integration（跨帧整合语义）、严格 think-answer 格式。

![](https://arxiv.org/html/2504.12680v1/extracted/6360624/fig/case.png)

### 3.4 Ablations (RQ3)

- **Key-Frame Extractor**: acc 几乎不变，训练时间 −8.7%，单 inference 时间约 −1/3。
- **Collaboration**: 同 keyframe 输入下，Embodied-R 整体 acc 是单独 72B VLM 的 1.5×。
- **RL 训练**: 不训直接用 LM-3B reasoning 极差；RL 后 UrbanVideo-Bench +27.9%、VSI-Bench +20.6%。同时把 4 个纯文本 reasoner（o3-mini / DeepSeek-R1 / Qwen-Max / Qwen2.5-7B-Instruct）作为 baseline，accuracy 与模型 reasoning 能力正相关——但 Embodied-R 优势既来自训练 distribution，也来自与 VLM representation 的 synergy（训完的 LM 更能"读懂" VLM 的 caption 风格）。

### 3.5 Further RQs

**Figure 4 + Figure 5. RL 训练曲线 + 关键 ablation 集合**——(a) accuracy reward；(b) format reward；(c) consistency-to-accuracy 比；(d) response length on val；(e) LM vs VLM 同尺度 RL 对比；(f) 加 vs 不加 logical consistency reward；(g) RL vs SFT 在 OOD 上的对比。

![](https://arxiv.org/html/2504.12680v1/extracted/6360624/x1.png)

![](https://arxiv.org/html/2504.12680v1/extracted/6360624/fig/training_process_2.png)

#### RQ4: Aha moment & response length

Math-style RL 中 response 越训越长 + 出现 aha moment 的现象，**在 embodied spatial reasoning 中没观察到**——response length 收敛到稳定区间。作者解释为任务驱动：embodied spatial QA 不需要多步代数运算，简洁 reasoning pattern 反而 facilitate 答案。这是对"RL on LLM 必出 long CoT"叙事的有用反例。

#### RQ5: 为什么不直接 RL 训 VLM？

直接对 Qwen2.5-VL-3B 做 RL（同样训练参数和时间）→ 收敛 acc 仅 43.8%，远低于 LM-on-text 路线。结论：在算力受限时，**collaborative inference > 端到端小 VLM RL**。

#### RQ6: Accuracy + Format reward 够吗？

不够。GPT-4o 评测发现，加 logical consistency reward 后，logically consistent output 比例从 **46.01% → 99.43%**。

#### RQ7: RL vs SFT OOD generalization

EgoSchema + MVBench egocentric subset：RL 训出的 LM 两 set 都 generalize 良好，EgoSchema 上甚至接近 Gemini-2.5-Pro；SFT 在 EgoSchema 提升、MVBench 退化。再次为 "RL 比 SFT 更鲁棒地 OOD" 添一个 embodied 场景的数据点。

---

## 关联工作

### 基于
- **DeepSeek-R1-Zero** (Guo et al., 2025): rule-based reward + GRPO 启发本文 reward 设计
- **GRPO**: 直接复用其 PPO-style clip + group-relative advantage
- **Qwen2.5-VL** / **Qwen2.5** (Bai et al., 2025): 提供 VLM-72B perception 与 LM-3B reasoning 两个基座

### 对比
- **OpenAI-o1, Gemini-2.5-Pro**: 顶尖闭源 reasoner，表明它们在 embodied spatial 上不 generalize
- **Qwen-VL-Max, GPT-4o, Gemini-1.5-{Flash,Pro}**: 通用 video-capable proprietary baseline
- **Qwen2.5-VL-{3B,7B}-SFT**: 同算力下端到端小 VLM 的 SFT 上限
- **o3-mini, DeepSeek-R1, Qwen-Max, Qwen2.5-7B-Instruct**: 纯文本 reasoner，验证"reasoning 能力 vs embodied acc"正相关

### 数据集
- **VSI-Bench** (Fei-Fei Li et al., Dec 2024): 室内 first-person，Visual-Spatial Intelligence 基准；用其 4 类难任务
- **UrbanVideo-Bench** (Tsinghua, Feb 2025): 室外无人机 aerial navigation；用其 Landmark Position / Counterfactual / Progress Evaluation / Action Generation 4 类
- **EgoSchema**: long-form egocentric video QA，OOD
- **MVBench**: 多任务 video understanding，取 egocentric navigation subset 做 OOD

### 方法相关
- **Video-RL / 多模态 RL 复现 R1 系列**: 同期一类工作；本文未与这些直接对比
- **Process Reward Model (PRM)**: 本文 logical-consistency reward 是 PRM 的轻量替代——不训 verifier，直接借 ref model 的 text-only 推理做 sanity check
- **后续工作 [[2508-EmbodiedR1|Embodied-R1]]**: 同名前缀但聚焦 robotic manipulation 的 grounding，方向不同

---

## 论文点评

### Strengths

1. **Logical-consistency reward 是个清爽、便宜、可迁移的 trick**：不需要 PRM、不需要额外标注，纯靠 reference model 在 text-only 模式下复现答案做一次 0/1 检验。对所有 "answer space 小 + 容易蒙对" 的 RLVR 任务（多选 QA、yes/no 推理、短答案数学）都有借鉴意义。
2. **算力受限下的工程思路清晰**：90 GPU-hours / 8×A800-40G 内打到 SOTA reasoner 之上，且明确把"为什么不直接 RL VLM"做成 RQ5 给出对照实验，把 trade-off 摆清楚。
3. **测试集去 semantic bias 的设计认真**：先用纯文本 SFT 一个 LM 过滤"不看视频也能答对"的题，是对"video benchmark 是否真考视频"这一常被忽视问题的实际处理。
4. **OOD evaluation 给得较扎实**：引入 EgoSchema + MVBench 两个独立 benchmark，对 RL vs SFT 的 generalization 给出可观测的 delta。

### Weaknesses

1. **Perception/Reasoning 解耦的 ceiling 取决于 VLM caption 质量，但论文几乎不评估这一环**：Qwen2.5-VL-72B 在 frame-level "ΔInformation" 上的准确率、漏检率没量化。整个 pipeline 上限被这一步死死压住，却被当成 black box 处理。
2. **"slow-thinking 涌现" 的判定主观**：RQ2 完全依赖 case analysis（4 个能力的 anecdotal 例子），没有定量指标（如 reasoning step 数、entity 引用数、对照组 hallucination 率）。"Embodied-R has learned slow thinking" 这种 claim 偏强。
3. **数据规模太小且任务面狭窄**：5K QA / 8 类多选题 / 全 multiple-choice，离 open-ended embodied reasoning（自由形式问答、动作生成、长程任务规划）还有距离。OOD 的 EgoSchema 也是多选题。所谓 generalization 是"不同 multiple-choice video QA 任务间的迁移"。
4. **没有和 video-RL 同期工作（如对 Video-R1 类 baseline、对 InternVideo / LLaVA-Video 类 video reasoner）的横向对比**——只比了 17 个通用 video LLM 和 reasoner，缺少 specialized embodied/video-RL baseline。
5. **三阶段课程没有消融**：weights schedule (7:3:0 → 3:7:0 → 1:7:2) 没有"一开始就 1:7:2"或"两阶段"对照，无法判断这是必要设计还是工程稳定化。
6. **Reward hacking 的诊断证据较弱**：从 46% → 99% 的 consistency 用 GPT-4o 当 judge——judge 自身可能与 ref model 在 text-only 推理逻辑上同质，存在 evaluation contamination 嫌疑。

### 可信评估

#### Artifact 可获取性

- **代码**: inference + training（GitHub `EmbodiedCity/Embodied-R.code`，基于 ms-swift）
- **模型权重**: 已发布两版 checkpoint（"Weight 1": 后期加 consistency reward；"Weight 2": 早期加 consistency reward），托管于 HuggingFace `EmbodiedCity/Embodied-R`
- **训练细节**: 主要超参（lr/temp/batch/rollout/KL coef/max len）+ 三阶段 weight ratio + 硬件（8×A800-40G）+ epoch 数完整。但 keyframe 阈值 $\varepsilon$、reward 权重为何选 7:3:0 等具体数无消融。
- **数据集**: 全部开源——VSI-Bench（NYU/Stanford，HuggingFace `nyu-visionx/VSI-Bench`）、UrbanVideo-Bench（HuggingFace `EmbodiedCity/UrbanVideo-Bench`）、EgoSchema、MVBench

#### Claim 可验证性

- ✅ **3B LM + 72B VLM 协作 acc 51.1%，超 OpenAI-o1 +13.9%、Gemini-2.5-Pro +10.3%**：5-fold CV + 完整 baseline 列表，代码权重数据全开源，可独立复现
- ✅ **Logical consistency reward 把 consistency 从 46% → 99%**：有 before/after 数字，但 judge 是 GPT-4o（外部模型），可独立用同 prompt 复测
- ✅ **Key-frame extractor 训练时间 -8.7% / 推理时间 -1/3，acc 几乎不变**：消融数字明确
- ⚠️ **"Embodied-R 学到 slow thinking、emergent systematic analysis / contextual integration"**：纯 case study，没有定量行为指标；主观判断风险高
- ⚠️ **"RL 在 embodied spatial 上不出现 aha moment / response length 不爆"**：只在本任务、这套 prompt、这套 reward 下成立；推广到"所有 embodied reasoning"是过度概括
- ⚠️ **"RL 比 SFT 更 OOD generalize"**：仅基于两个 OOD set，且 SFT baseline 的 schedule / 数据量是否充分调优论文未交代——可能是 SFT under-trained
- ⚠️ **"perception 上限制约 reasoning 上限"**：RQ5 的对比只跑了一种 VLM-3B RL 配置，可能尚未充分调优
- ❌ **未发现明显 marketing claim**

### Notes

- **可借鉴到自己工作的 trick**: 那个 logical-consistency reward——只要任务是"结构化输出 + answer space 小"，就可以加一道 "ref model 仅看 reasoning 文本能否复现答案" 的 0/1 校验，几乎不增加训练成本。值得在 spatial-reasoning / GUI agent 的 RLVR pipeline 里试。
- **解耦 perception/reasoning 的代价**: 本文的 "VLM caption + LM reason" 路线把推理上限锁在 caption quality 上。若问题是 fine-grained visual reasoning（数像素、几何精度），caption 这层 bottleneck 会丢信息；这条路线本质适合 high-level、symbolic-friendly 的 spatial QA。
- **对"embodied 没 aha moment"的判断要谨慎**：作者只跑了 12 epoch / 2048 max len。math-domain 的 long CoT 涌现需要更长 training horizon + 更长 max len。这里观察到的"length 不发散"可能也是 reward 设计（consistency 反而抑制冗长无关推理）+ 短 max len 的产物，不一定是 embodied 任务的本质属性。
- **跟 Dr. Li 自己研究方向的关系**: Spatial Intelligence × Agentic RL 的交叉点。这个 recipe（borrow large-VLM perception + RL-train small-LM reasoning）在 embodied agent 上验证了一种 cost-efficient 路线，是 VLA 同期的另一条路——值得在思考 VLA reasoning trace / world model integration 时纳入对比 mental model。

### Rating

**分数**：2 - Frontier
**理由**：本文在 Spatial Intelligence × Agentic RL 交叉点是一个有代表性的 frontier 工作——logical-consistency reward（Strengths 1）是可迁移到 RLVR 的清爽 trick，"借大 VLM perception + RL 训小 LM reasoning" 的 cost-efficient recipe 与 RQ5 的对照实验构成明确 method contribution。但离 Foundation 档还差：方法 scope 窄（5K 样本 / 8 类多选题，Weaknesses 3）、没有和同期 video-RL baseline 横向对比（Weaknesses 4），尚未被社区采纳为 de facto embodied reasoning 范式；也未到 Archived 档——作为 embodied RL reasoning 的早期 recipe 仍在被同期工作引用参考。
