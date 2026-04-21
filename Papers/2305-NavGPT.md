---
title: "NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models"
authors: [Gengze Zhou, Yicong Hong, Qi Wu]
institutes: [The University of Adelaide, The Australian National University]
date_publish: 2023-05-26
venue: AAAI 2024
tags: [VLN, LLM, navigation]
paper: https://arxiv.org/abs/2305.16986
website: 
github: https://github.com/GengzeZhou/NavGPT
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models
> - **核心**: 用纯 LLM (GPT-4 / GPT-3.5) 做 zero-shot R2R 导航 agent，将视觉观测转成自然语言、用 ReAct 风格的 Thought+Action 循环显式输出推理过程
> - **方法**: BLIP-2 caption + Faster-RCNN object detection + Matterport3D depth → prompt manager 拼装；GPT-3.5 summarizer 压缩历史；ReAct 推理后选下一个 viewpoint
> - **结果**: R2R val-unseen SR 34% / SPL 29%，远低于 supervised SOTA (DuET 72%)，但远好于无训练 baseline DuET-LXMERT (SR 1%)
> - **Sources**: [paper](https://arxiv.org/abs/2305.16986) | [github](https://github.com/GengzeZhou/NavGPT)
> - **Rating**: 2 - Frontier（LLM-as-VLN-agent 早期可行性证明，AAAI 2024 + 后续 NavGPT-2 延续，但 caption-based pipeline 已被 multi-modal LLM 路线取代，属于方向前沿的"转折点"工作而非 building block）

**Key Takeaways:**
1. **Zero-shot LLM agent for VLN 是可行的**: 纯文本输入 + ReAct 循环就能在 R2R 拿到 SR=34，证明 LLM 内在的 commonsense 与高层规划能力可以直接用于 embodied 导航
2. **可解释的高层规划**: GPT-4 的 thought trace 显式展现 sub-goal decomposition, landmark identification, progress tracking, exception handling 五种行为，是 supervised 模型不具备的"白盒"特性
3. **Spatial / historical awareness emergent**: 仅给文本动作描述，GPT-4 能用 pyplot 画出近似的 top-down trajectory；也能从历史 action+observation 反向生成 R2R 风格 instruction
4. **瓶颈在 vision→text 的信息损失**: BLIP-2 caption 的粒度限制 + history summarizer 的压缩损失，是与 supervised 模型差距的主因；作者指出未来应做多模态 LLM 而非 caption-based pipeline

**Teaser. NavGPT 的整体架构**：LLM 通过 prompt manager 接收 VFM 的视觉描述、history buffer (含 GPT-3.5 summarizer)、navigation system principle，输出 Thought+Action。
![](https://ar5iv.labs.arxiv.org/html/2305.16986/assets/figures/NavGPT2.png)

---

## 任务设定与动机

VLN 给定自然语言指令 $\mathcal{W}$，agent 在每一步从 simulator 获取 panorama observation $\mathcal{O}_t = [\langle o_i, a_i\rangle]$ 并从可达 viewpoint $C_{t+1}$ 中选下一步。Agent 状态 $s_t = \langle v_t, \theta_t, \phi_t\rangle$ 包含 viewpoint、heading、elevation。Policy $\pi(a_t|\mathcal{W}, \mathcal{O}_t, \mathcal{O}_t^C, \mathcal{S}_t; \Theta)$ 通常需要从 VLN 数据集训练 $\Theta$。

**NavGPT 的关键 deviation**：$\Theta$ 不从 VLN 数据训练，而完全来自 LLM 预训练语料。文章想回答的问题是：**LLM 能否仅通过文字理解 interactive world、actions and consequences，从而解 navigation 问题？**

> ❓ 这个 framing 把 VLN 当成一个 "LLM reasoning probe"，而非追求 SOTA。这跟 SayCan / PaLM-E 的实用 framing 不同——前者把 LLM 当 planner 配合 affordance，后者直接做 multimodal 训练。NavGPT 走的是 "LLM-as-only-policy + VFM-as-translator" 的极简路线。

## 方法

### 系统组件

NavGPT 把决策分解为 4 个抽象组件，prompt manager $\mathcal{M}$ 把它们都翻译成纯文本喂给 LLM：

$$
\langle \mathcal{R}_{t+1}, \mathcal{A}_{t+1} \rangle = \text{LLM}(\mathcal{M}(\mathcal{P}), \mathcal{M}(\mathcal{W}), \mathcal{M}(\mathcal{F}(\mathcal{O}_t)), \mathcal{M}(\mathcal{H}_{<t+1}))
$$

- **Navigation System Principle $\mathcal{P}$**: 写死的 system prompt，定义 VLN 任务、reasoning format、规则（如 viewpoint ID 不能编造）
- **Visual Foundation Models $\mathcal{F}$**: BLIP-2 (caption) + Faster-RCNN (object box) + Matterport3D (depth)，把图像翻译成 "object class + 相对 heading + 相对距离" 的文字描述
- **Navigation History $\mathcal{H}_{<t+1}$**: 三元组序列 $\langle \mathcal{O}_i, \mathcal{R}_i, \mathcal{A}_i\rangle$，其中早期 observation 由 GPT-3.5 summarizer 压成一句话
- **Prompt Manager $\mathcal{M}$**: 把 4 类信息按当前 heading 为 "front" 顺时针拼成一个 prompt

### Visual Perceptron

每个 viewpoint 取 8 个 heading（0°/45°/.../315°）× 3 个 elevation（-30°/0°/+30°）= 24 个 egocentric view，每个 FoV=45°。BLIP-2 用 prompt "This is a scene of" 给每张图生成 caption，再用 GPT-3.5 把同 heading 的 top/middle/down 三张 caption summarize 成一句话。

Faster-RCNN 检测物体，从 Matterport3D 取 bbox 中心像素的 depth；只保留 3m 内的物体，附带相对 heading。

**Figure 2. 视觉到文字的转换流程**：从一个 viewpoint 的 8 个 direction 中取一个 direction 演示。
![](https://ar5iv.labs.arxiv.org/html/2305.16986/assets/figures/obs2.png)

### ReAct 风格的 Thought + Action

借用 ReAct 的扩展动作空间 $\tilde{\mathcal{A}} = \mathcal{A} \cup \mathcal{R}$，$\mathcal{R}$ 是任意自然语言。Thought 不与环境交互，但被注入到 history 里供后续步骤复用。两个声称的好处：
1. 思考再选择动作 → 复杂规划（sub-goal、landmark matching）
2. Thought 进 history → 长程一致性 + plan adjustment

### History 压缩

完整 history 太长会爆 context。用 GPT-3.5 summarize 每个旧 viewpoint 的 observation 为一句话，模板：
```
Given the description of a viewpoint. Summarize the scene from the viewpoint
in one concise sentence.
Description: {description}
Summarization: The scene from the viewpoint is a
```
当前 viewpoint 用完整描述，旧 viewpoint 用 summary——这是性能 vs. context length 的折中。

## 实验

### Setup

- **数据集**: R2R val-unseen split，783 trajectory，11 个 unseen indoor scene
- **VFMs**: BLIP-2 ViT-G FlanT5-XL + Faster-RCNN
- **指标**: TL（轨迹长度）/ NE（导航误差）/ OSR / SR / SPL

### 与 supervised baseline 的对比

**Table 1. R2R val-unseen 上 NavGPT 与 supervised 方法对比**

| Training Schema | Method | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Train Only | Seq2Seq | 8.39 | 7.81 | 28 | 21 | - |
| Train Only | Speaker Follower | - | 6.62 | 45 | 35 | - |
| Train Only | EnvDrop | 10.70 | 5.22 | - | 52 | 48 |
| Pretrain+Finetune | PREVALENT | 10.19 | 4.71 | - | 58 | 53 |
| Pretrain+Finetune | VLN-BERT | 12.01 | 3.93 | 69 | 63 | 57 |
| Pretrain+Finetune | HAMT | 11.46 | 2.29 | 73 | 66 | 61 |
| Pretrain+Finetune | DuET | 13.94 | 3.31 | 81 | 72 | 60 |
| No Train | DuET (Init. LXMERT) | 22.03 | 9.74 | 7 | 1 | 0 |
| No Train | **NavGPT (Ours)** | 11.45 | 6.46 | **42** | **34** | **29** |

NavGPT 离 supervised SOTA 还有 ~38 个 SR 点的 gap，但相比未训练的 LXMERT-init DuET 已是质变（SR 1 → 34）。作者归因为两点：(a) 视觉到语言的描述精度不够；(b) 物体 tracking 能力弱。

### 视觉组件 ablation

由于 budget 限制，ablation 全部用 GPT-3.5（性能上限低）。从 train + val unseen 共 72 个 scene 各采样 1 个 trajectory × 3 instructions = 216 样本。

**Table 2. 视觉描述粒度的影响**

| Granularity | # | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
| --- | --- | --- | --- | --- | --- | --- |
| FoV@60, 12 views | 1 | 12.38 | 9.07 | 14.35 | 10.19 | 6.52 |
| FoV@30, 36 views | 2 | 12.67 | 8.92 | 15.28 | 13.89 | 9.12 |
| FoV@45, 24 views | 3 | 12.18 | 8.02 | **26.39** | **16.67** | **13.00** |

FoV 太大→BLIP-2 倾向于描述房间类型，丢物体细节；FoV 太小→无法识别完整物体。FoV=45 是 sweet spot。

**Table 3. 物体检测和 depth 信息的增益**

| Agent Observation | # | TL | NE↓ | OSR↑ | SR↑ | SPL↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (caption only) | 1 | 16.11 | 9.83 | 15.28 | 11.11 | 6.92 |
| Baseline + Obj | 2 | 11.07 | 8.88 | 23.34 | 15.97 | 11.71 |
| Baseline + Obj + Dis | 3 | 12.18 | 8.02 | 26.39 | **16.67** | **13.00** |

物体 detection 把 SR 从 11 提到 16（+4.86），depth 再加 0.7。作者观察到一个具体 failure mode：agent 不知道离目标多远，看到目标就立刻停 → depth 缓解了这个问题。

### Qualitative：高层规划

**Figure 3. NavGPT 的 reasoning trace 展示**：sub-goal decomposition、commonsense reasoning、landmark identification、progress tracking、exception handling 五种行为都能被 GPT-4 显式输出。
![](https://ar5iv.labs.arxiv.org/html/2305.16986/assets/x1.png)

**Figure 4. Spatial / historical awareness probe**：仅给 GPT-4 历史 action+observation（剔除 reasoning 防止信息泄漏），它能(a) 反向生成符合 R2R 风格的 instruction，(b) 用 pyplot 画出近似的 top-down trajectory。
![](https://ar5iv.labs.arxiv.org/html/2305.16986/assets/x2.png)

> ❓ Trajectory drawing 看起来很惊艳，但这是 1 个 cherry-picked 成功 case，没有量化评估（如 trajectory IoU、平均位置误差）。GPT-4 真的能可靠地维护 metric spatial map，还是只在简单短轨迹上 work？这是 claim 与 evidence 的 gap。

### Failure modes

两类典型失败：
1. **目标物体不在 caption 中**：BLIP-2 没描述到 instruction 提到的物体 → agent 被迫盲目探索
2. **History summary 丢细节**：把 viewpoint 压成一句话后，agent 无法判断 "我有没有 turn right into the next room"，错误地以为已经完成

作者建议方向：动态生成视觉描述（类似 ChatCaptioner 的 LLM-VFM 多轮交互），避免一次性 static caption 的信息瓶颈。

---

## 关联工作

### 基于
- **ReAct** (Yao et al. 2022): NavGPT 直接借用 Thought+Action 的扩展动作空间形式
- **BLIP-2** (Li et al. 2023): 视觉描述的 caption backbone
- **R2R** (Anderson et al. 2018): VLN 任务和评测协议来源
- **Matterport3D Simulator**: 提供离散 navigation graph 和 depth

### 对比
- [[2204-SayCan|SayCan]]: 同样是 LLM + embodied，但 SayCan 是 LLM 做 high-level planning + low-level skills with affordance score；NavGPT 是 LLM 直接做 step-level decision
- [[2303-PaLME|PaLM-E]]: 多模态 LLM 路线（与 NavGPT 的 caption-based 路线对立）；NavGPT 在结论里实际上承认 PaLM-E 这条路更有前途
- **DuET / HAMT**: supervised SOTA，作为性能上界对比
- DuET (Init. LXMERT): 唯一的 "no train" baseline，但是为 supervised 设计的模型，对比不对称

### 方法相关
- **CLIP-based zero-shot VLN** (Dorbala et al.): 另一种 zero-shot VLN 路线，依赖 text-image matching 而非 LLM reasoning
- **Topological maps for VLN**: NavGPT 没有显式 map 结构，完全靠 LLM 的 history+spatial awareness
- **ChatCaptioner**: 作者建议未来用 LLM-VFM 多轮交互替代 static caption，是一个明确的 follow-up 方向

### 后续工作
- **NavGPT-2** (ECCV 2024): 同作者的后续，转向 multi-modal LLM，呼应本文 "future direction" 的判断
- [[2402-NaVid|NaVid]] / [[2412-NaVILA|NaVILA]] / [[2506-VLNR1|VLN-R1]]: 接棒的 video / multi-modal LLM 路线，验证了 caption-based pipeline 是 dead end 的判断

---

## 论文点评

### Strengths

1. **Framing 清晰且有 conceptual value**：把 LLM 当成"零训练 VLN policy"是一次干净的 probe，让我们看到 GPT-4 在 embodied 任务上能涌现哪些能力（高层规划、spatial awareness、progress tracking）——这些在 supervised black-box VLN model 里是看不到的
2. **Reasoning trace 的可解释性是真材实料**：Figure 3 中 sub-goal decomposition、exception handling 是 supervised model 几乎不可能给出的，对 VLN agent 的 debug 和 error attribution 有方法论价值
3. **Ablation 抓住关键 trade-off**：FoV granularity 和 caption vs. obj+depth 两个 ablation 直接指向了 caption-based pipeline 的核心瓶颈，对后续工作（如 NavGPT-2 转向 multi-modal LLM）有清晰的 motivation

### Weaknesses

1. **绝对性能与 supervised 模型差距巨大**：SR 34 vs. DuET 72，作为方法论 contribution 可以接受，但作为 "LLM-as-VLN-agent" 的可行性证明，这个 gap 表明 caption-based pipeline 不是正确路径
2. **Spatial awareness claim 缺乏量化**：Figure 4 的 top-down trajectory drawing 只是一个 cherry-picked case，没有任何系统性的几何精度评估，难以判断这是 reliable capability 还是 memorization artifact
3. **Ablation 仅在 GPT-3.5 上做**：所有 ablation 用 GPT-3.5（SR 上限远低于 GPT-4），主结果用 GPT-4——读者无法判断 visual component 的贡献是否在 GPT-4 上同样成立，可能存在 LLM capability × visual quality 的非线性交互
4. **依赖 oracle navigation graph**：NavGPT 选 viewpoint 的离散动作空间来自 R2R 的预定义图，不是真实场景的 continuous navigation；这意味着方法不能直接迁移到 free-space VLN，limitation 没充分讨论
5. **History summarizer 的损失没量化**：GPT-3.5 summarizer 是性能瓶颈之一，但论文没做 "no summarizer / full history" baseline（哪怕在短轨迹上）来量化这一损失

### 可信评估

#### Artifact 可获取性

- **代码**: inference-only（GitHub 提供 NavGPT.py 推理脚本，因为方法本身就 zero-shot 无训练）
- **模型权重**: 用 OpenAI 的 GPT-4 / GPT-3.5（闭源 API）+ BLIP-2 ViT-G FlanT5-XL（开源）+ Faster-RCNN（开源）；无新训练权重
- **训练细节**: N/A（zero-shot 方法），但 prompt 模板和 BLIP-2 prompt 选择有完整披露（Appendix A）
- **数据集**: R2R 公开（README 提供 Dropbox 下载链接），val-unseen split 783 trajectories

#### Claim 可验证性

- ✅ **R2R val-unseen 上 NavGPT (GPT-4) SR=34 / SPL=29**：标准 R2R 评估流程，代码 + 数据 + API 全可复现，唯一的 reproducibility 风险是 OpenAI API 版本漂移
- ✅ **VFM 组件 ablation (FoV / obj / depth) 的相对收益**：在 216 样本子集上做的，方差可能不小，但方向性结论可信
- ⚠️ **GPT-4 能"显式 sub-goal decomposition / landmark identification / exception handling"**：基于 cherry-picked qualitative case，无量化频率（多少 % 的 trajectory 出现哪种行为），且这些"能力"是 prompt-induced 还是 emergent 难以区分
- ⚠️ **GPT-4 能画出准确的 top-down trajectory**：单 case 演示，无系统精度评估；"awareness of spatial relations" 的 claim 强于 evidence
- ⚠️ **NavGPT 表现"远好于"无训练 baseline**：唯一对比是 DuET-LXMERT (SR 1)，这是一个为 supervised 训练设计的模型直接 zero-shot——比较是不对称的，更公平的 baseline 应是其他 LLM-based zero-shot 方法（如 CLIP-based zero-shot VLN）

### Notes

- **历史定位**：这是 LLM 时代第一波"把 ChatGPT 当 VLN agent"的工作之一（2023.05），与 [[2204-SayCan|SayCan]] (2022) / [[2303-PaLME|PaLM-E]] (2023) 共同定义了"LLM for embodied navigation"的早期 design space。NavGPT 选了最极端的 LLM-only + caption pipeline，作为可行性证明，但论文自己也意识到这条路的天花板低
- **方向上的 lesson**：caption→LLM 是 VLN 的 dead end（信息瓶颈不可逾越）。后续的 [[2402-NaVid|NaVid]] / [[2412-NaVILA|NaVILA]] / [[2506-VLNR1|VLN-R1]] 等都走 video / multi-modal LLM 路线，验证了这个判断
- **prompt-engineering 的工程价值**：Appendix 里 BLIP-2 prompt 选择 ("This is a scene of" vs. "Detailly describe the scene") 的对比是个有意思的细节——caption-based pipeline 中，prompt 选择直接决定上层 LLM 能否拿到正确的 grounding 信息
- **可借鉴的 component**：history summarizer 的设计（旧 viewpoint 一句话，当前 viewpoint 完整描述）是处理长 context VLN 的简单 baseline，但其信息损失是已知 weakness——值得一个量化的 ablation
- **对 spatial intelligence 的启示**：Figure 4 那个 GPT-4 画 trajectory 的实验，虽然 cherry-picked，但提示了一个有意思的问题：LLM 能否仅从语言形式的 action sequence 维护 metric spatial state？这与 [[2604-CoTDegradesSpatial|CoT degrades spatial]] 的发现可能有联系——LLM 的 spatial reasoning 是脆弱的，cherry-picked case 难以反映真实分布

### Rating

**分数**：2 - Frontier

**理由**：AAAI 2024 录用 + 同作者 NavGPT-2 延续，且作为"LLM-only + caption pipeline 做 VLN"的代表工作，后续 [[2402-NaVid|NaVid]] / [[2412-NaVILA|NaVILA]] / [[2506-VLNR1|VLN-R1]] 都把它作为 LLM-for-VLN 早期 baseline 引用——符合 Frontier 档的"方法范式代表工作"定义。没有升 3 是因为 Weakness 里写到的 caption-based pipeline 已被证明是 dead end，社区转向 multi-modal LLM 路线，NavGPT 不是这条活路的 building block；没有降 1 是因为它在 LLM-for-VLN 的 design space 讨论里仍是绕不开的"转折点"参考。
