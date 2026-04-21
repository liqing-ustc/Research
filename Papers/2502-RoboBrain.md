---
title: "RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete"
authors: [Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xiaolong Zheng, Jiaming Liu, Zhongyuan Wang, Shanghang Zhang]
institutes: [Peking University, BAAI, CASIA, IIE-CAS, HKU, UCAS]
date_publish: 2025-02-28
venue: CVPR 2025
tags: [embodied-reasoning, manipulation, task-planning]
paper: https://arxiv.org/abs/2502.21257
website: https://superrobobrain.github.io/
github: https://github.com/FlagOpen/RoboBrain
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete
> - **核心**: 把"机器人脑"拆成 planning + affordance + trajectory 三种能力，用一个 LLaVA-OneVision 风格的 7B MLLM 统一承载，通过两个 LoRA 头分别处理 affordance bbox 和 2D trajectory waypoint。
> - **方法**: 从 Open X-Embodiment (OXE) 选 51K 高质量 instances，用 Gemini + 人工标 1.03M planning QA、6.5K affordance bbox、6.9K trajectory 序列形成 ShareRobot 数据集；多阶段训练（OneVision 通用 → 1.3M robotic + 1.7M general 抗遗忘 → A/T-LoRA 分别训）。
> - **结果**: RoboVQA BLEU-4 比第二名高 18.75；AGD20K affordance AP 27.1% vs Qwen2-VL 12.5%；trajectory DFD 相比 baseline 降 42.9%（关键功臣是 `Spec_Token` + 显式起止点）。
> - **Sources**: [paper](https://arxiv.org/abs/2502.21257) | [website](https://superrobobrain.github.io/) | [github](https://github.com/FlagOpen/RoboBrain)
> - **Rating**: 2 - Frontier（BAAI "embodied brain" 系列的 1.0 版本、后续 2.0/2.5 的产品骨架来源，ShareRobot 数据集被社区作为 planning/affordance 标注范式参考，但方法本身已被 2.5 推翻升级）

**Key Takeaways:**
1. **三能力分解的产品框架**：planning（abstract）→ affordance（grounding）→ trajectory（concrete motion）。这个分解被后续 RoboBrain 2.0/2.5 沿用并演化成 BAAI "embodied brain" 系列的产品骨架。
2. **ShareRobot 是工程性贡献**：从 OXE 二次清洗 51K instances → 1.03M planning QA + 6.5K affordance + 6.9K trajectory，是当时最大规模的 OXE 派生 fine-grained planning 标注集，比 RoboVQA 在场景多样性（102 scenes / 12 embodiments / 107 atomic tasks）和 task 颗粒度上更细。
3. **Affordance / Trajectory 分别用 LoRA 头**：rank-64 LoRA 加在 projector + LLM 的 FFN 上，而非引入 action head。affordance 输出 bbox 四元组、trajectory 输出归一化到 [0, 1000) 的 2D waypoint 序列，纯文本 token 化——是 VLM-as-Output-Format 的 affordance/trace 范式。
4. **Trajectory 的实证发现：special token + 显式 start/end 点带来 94.2% HD 下降**，单纯加 start point 或限制 max points 收益小。说明 trajectory 任务对"边界条件显式化"非常敏感。

**Teaser. RoboBrain Overview——三能力 + ShareRobot 数据 composition**

![](https://arxiv.org/html/2502.21257v2/x1.png)

---

## 1. Motivation

作者把 "MLLM 应用于机器人的瓶颈" 归为三件事：

1. **Planning Capability** — 把 long-horizon 指令拆成 sub-task。
2. **Affordance Perception** — 识别可交互区域（如茶壶把手）。
3. **Trajectory Prediction** — 给出从当前位置到 affordance 的 2D 路径。

举的例子是"提起茶壶倒水进杯"——MLLM 要先拆出 "approach teapot"、"move to cup"、"tilt to pour"，再对每一步识别可抓握区域和轨迹。作者认为现有 MLLM 在这三件事上同时弱主要是**缺细颗粒度的标注数据**。

> ❓ 这个 problem framing 其实是个**产品向**的分解，不是 first-principles 的。planning / affordance / trajectory 之间的接口是什么？为什么 affordance 必须是 bbox 而不是 mask 或 keypoint？为什么 trajectory 是 2D 而不是 3D？这些选择更像 "为了让 MLLM 能用文本输出" 的工程妥协，而不是认知架构上的必然分解。RoboBrain 2.5 后来把 trajectory 升到 3D、affordance 升到 keypoint 序列，也印证了这个分解是 1.0 阶段的过渡形态。

---

## 2. ShareRobot 数据集

**Figure 2. ShareRobot 数据生成流水线**

![](https://arxiv.org/html/2502.21257v2/x2.png)

ShareRobot 是这篇论文最实在的贡献。从 OXE 选 23 个原始数据集，按 6 条 filter 保留 51,403 instances：

- 分辨率 ≥ 128 px
- 描述准确（无 vague description）
- success status = 成功
- 视频 ≥ 30 帧
- target object 和 end-effector 不被遮挡
- trajectory 清晰可见

### 2.1 三种标注

| 标注类型 | 量 | 标注方式 |
|---|---|---|
| Planning QA | 1,027,990 pairs | 每段视频抽 30 帧 → Gemini 自动拆 low-level instruction → 3 个标注员校对；用 RoboVQA 的 10 种 question type × 5 个模板生成 QA |
| Affordance | 6,522 images | 人工标 bbox `{l_x, l_y, r_x, r_y}` |
| Trajectory | 6,870 images | 人工标 ≥3 个 `(x, y)` waypoint |

**Figure 3. ShareRobot 多样性：23 个源数据集 / 12 embodiments / 107 atomic tasks**

![](https://arxiv.org/html/2502.21257v2/x3.png)

split：planning 1M 训 / 2050 测；affordance 6000 训 / 522 测；trajectory 6000 训 / 870 测。

> ❓ Planning QA 用 Gemini 自动拆再人工 review——量大但**生成质量天花板就是 Gemini 的水平**。从 51K instances → 1M QA 主要是模板扩增（10 type × 5 template × 2 抽样），实际 unique 信息量仍约束于 51K。"largest open-source dataset" 的宣传里有水分。

---

## 3. RoboBrain 模型

**Figure 4. RoboBrain pipeline——foundation + A-LoRA + T-LoRA**

![](https://arxiv.org/html/2502.21257v2/x4.png)

### 3.1 架构

标准 LLaVA 三件套：

- **Vision encoder**: SigLIP `siglip-so400m-patch14-384`，每图 729 visual tokens
- **Projector**: 2-layer MLP
- **LLM**: Qwen2.5-7B-Instruct，128K context

Affordance 和 trajectory 各一个 rank-64 LoRA，加在 projector + LLM 的 FFN 上（其余参数冻结）。**没有额外的 action head**——所有输出都是文本 token。

- Affordance：`{l_x, l_y, r_x, r_y}` bbox 四元组
- Trajectory：waypoint 序列 `{(x_i, y_i)}`，归一化到 `[0, 1000)`（沿用 Qwen2-VL 的坐标 tokenization）

### 3.2 多阶段训练

**Phase 1 — General OV Training**（沿用 LLaVA-OneVision 配方）

- Stage 1: LCS-558K 训 projector 对齐
- Stage 1.5: 4M image-text 训整模做 multimodal general knowledge
- Stage 2: 3.2M single-image + 1.6M image+video 训 instruction following

**Phase 2 — Robotic Training**

- Stage 3 (Planning): 1.3M robotic 数据（RoboVQA-800K + ScanView-318K + ShareRobot-200K subset）+ 1.7M Phase-1 高质量 image-text 防遗忘，整模微调
- Stage 4 (Affordance / Trajectory): A-LoRA 和 T-LoRA 分别训

> ❓ Stage 3 的"防遗忘"配比 1.3M robotic : 1.7M general 接近 1:1.3，比典型的 50:50 略偏 general。这个比例是搜出来的还是拍脑袋的？没有 ablation。

---

## 4. 实验

### 4.1 Planning

**Figure 5. RoboBrain 在 OpenEQA / ShareRobot / RoboVQA 上的表现**

![](https://arxiv.org/html/2502.21257v2/x6.png)

vs GPT-4V、Claude 3、LLaVA-1.5、LLaVA-OneVision-7B、Qwen2-VL-7B、RoboMamba。RoboBrain 全面领先，RoboVQA BLEU-4 比第二名高 18.75。

> ❓ RoboVQA 和 ShareRobot test set **都是 RoboBrain 训练分布内**（in-domain），ShareRobot test 还是同一管线产出的 2050 QA。把 in-distribution 的优势包装成 SOTA 不太厚道。OpenEQA 是更可信的对照点。

### 4.2 Affordance

**Table 2. AGD20K affordance AP**

| Model | AP ↑ |
|---|---|
| LLaVA-NeXT-7B | 9.8% |
| Qwen2-VL-7B | 12.5% |
| RoboBrain | 27.1% (+14.6) |

Baselines 都是 zero-shot，RoboBrain 是经过 ShareRobot affordance + AGD20K-style 数据 fine-tune 的——比较口径不完全公平，但 14.6 AP 的差距说明任务定向训练确实有效。

### 4.3 Trajectory

**Table 3. Trajectory Prediction Ablation**

| Method | DFD ↓ | HD ↓ | RMSE ↓ |
|---|---|---|---|
| RoboBrain (Base) | 0.191 | 0.171 | 0.133 |
| + Start_Points | 0.176 | 0.157 | 0.117 |
| + Max_Points | 0.185 | 0.163 | 0.125 |
| + Spec_Token & End_Points | **0.109** (-42.9%) | **0.010** (-94.2%) | **0.091** (-31.6%) |

Spec_Token + 显式 end_point 的增益异常大，HD 降 94.2% 几乎等于"消除最大偏移"。说明 trajectory 任务的瓶颈不是模型能力，而是**输出格式的归纳偏置**——给 LLM 一个 "[START] x y [WAYPOINT] ... [END] x y" 的清晰模板远比让它自由生成 waypoint 序列重要。

### 4.4 可视化

**Figure 6. 多轮交互：planning + affordance + trajectory 串联**

![](https://arxiv.org/html/2502.21257v2/x7.png)

---

## 关联工作

### 基于
- LLaVA-OneVision: foundational MLLM 训练配方（Phase 1 几乎照搬）
- SigLIP: vision encoder
- Qwen2.5-7B-Instruct: LLM backbone
- Open X-Embodiment (OXE): ShareRobot 的源数据
- LoRA: 用于 affordance / trajectory 头的 PEFT

### 系列演化
- [[2507-RoboBrain2|RoboBrain 2.0]]: 后续版本，加入 spatial reasoning 和更强的 reasoning 能力
- [[2601-RoboBrain25|RoboBrain 2.5]]: 进一步升到 3D spatial reasoning + dense temporal value estimation（GRM for VLA RL）

### 对比 / 同类
- [[2303-PaLME|PaLM-E]]: 早期把多模态映射到 language space 的 robotic MLLM 范式
- [[2403-RTH|RT-H]]: 生成 reasoning + action（带额外 policy head）
- RoboMamba: 同期 robotic MLLM，作为主要 baseline
- [[2410-Pi0|π0]] / [[2504-Pi05|π0.5]]: VLA 范式（end-to-end action），与 RoboBrain 的"输出 plan/affordance/trajectory 让下游 policy 执行"形成对照——RoboBrain 走 high-level brain 路线，π0 走 end-to-end action 路线

### 数据 / 范式
- RoboVQA: planning QA 评测；ShareRobot 直接复用其 question type taxonomy
- OpenEQA: embodied QA 评测
- AGD20K: affordance benchmark

---

## 论文点评

### Strengths

1. **三能力分解作为产品骨架很有解释力**——planning / affordance / trajectory 分别对应 abstract → grounded → concrete，是 BAAI 后续 RoboBrain 2.0/2.5 系列产品定位的基础，也是当前"embodied reasoning model"赛道的主流分解范式之一。
2. **ShareRobot 数据流水线工程扎实**：OXE 二次筛选 + Gemini-assisted planning 标注 + 人工 review，量级（1M planning QA）和多样性（23 datasets / 12 embodiments / 107 tasks）在 2025 初是开源最大。
3. **Trajectory token 化的实证 finding 有价值**：special token + 显式 start/end point 让 HD 降 94.2%，说明"VLM 输出结构化序列"对 prompt template 极度敏感。这是个可迁移的 design lesson。
4. **A/T-LoRA 解耦实用**：保留 base planning 模型的同时支持 plug-in affordance/trajectory 能力，部署友好。

### Weaknesses

1. **没有真机实验**。所有指标都是 VQA / bbox / 2D 轨迹的离线评估。"robotic manipulation" 标题下却没有 manipulation success rate，作为 robot brain 的 claim 是悬空的——它本质是个标注更细的 robotic VQA model。
2. **Planning evaluation in-distribution**。ShareRobot test 来自同一标注管线，RoboVQA 也在训练里。OpenEQA 是唯一相对独立的对照，但单一 benchmark 信号不够强。
3. **ShareRobot 的"largest"宣传含水分**。51K unique instances 通过 10 question type × 5 template × 2 抽样扩成 1M——是模板扩增不是真实 1M 信息量。
4. **Affordance / Trajectory 的本体设计偏 ad-hoc**：affordance 用单 bbox（不能表达多区域、复杂形状），trajectory 用 2D 像素点（无 depth、无 timing）。这些限制后来在 RoboBrain 2.5 都被推翻（升 3D + keypoint），说明 1.0 的设计是过渡形态。
5. **缺乏 first-principles 的能力分解论证**：为什么是这三种能力？任务规划与 affordance 之间的接口怎么定义？文章默认接受这个分解作为 axiom，没有讨论 alternative。

### 可信评估

#### Artifact 可获取性

- **代码**: inference + training 都开源（github.com/FlagOpen/RoboBrain，CVPR 2025 official repo）
- **模型权重**: 已发布 Planning checkpoint、A-LoRA、T-LoRA 三个 ckpt（HuggingFace `BAAI/RoboBrain`、`BAAI/RoboBrain-LoRA-Affordance`、`BAAI/RoboBrain-LoRA-Trajectory`）
- **训练细节**: 完整给了多阶段 batch size / lr / epoch / resolution 配置（Tab 4），数据配比和 yaml 都在 repo 里
- **数据集**: ShareRobot 已开源（github.com/FlagOpen/ShareRobot）

可复现性是这篇论文的强项——code、weights、data、训练配方都全。

#### Claim 可验证性

- ✅ "RoboVQA BLEU-4 比第二名高 18.75"：表格数据可核，但前提是 RoboVQA 在训练分布里
- ✅ "AGD20K AP 27.1% vs 12.5%"：baselines 是 zero-shot，RoboBrain 是 fine-tuned，差距来源主要是任务定向训练而非模型架构
- ⚠️ "state-of-the-art on planning task"：planning 任务的 evaluation 大多 in-distribution，泛化性未充分验证
- ⚠️ "三种核心机器人脑能力"：分解是产品视角，没有理论或行为学证据支持这是充分必要的能力集合
- ❌ "unified brain model for robotic manipulation"：没有任何真机 manipulation 评测，"manipulation" 在标题里实际指代的是 manipulation-related VQA + bbox + 2D trajectory

### Notes

- **取向判断**：这篇是"产品形态的开山之作"而非 research insight。三能力分解 + LoRA 解耦 + ShareRobot 数据 = 一个可以演化的产品骨架，但单看 1.0 没有改变我对任何根本问题的判断。归类为 Indexed——需要追踪 RoboBrain 系列演化时再回看。
- **真正有价值的实证 finding**：trajectory ablation 里 special token + 显式 end point 让 HD 降 94.2%。这是 generalizable 的——任何用 LLM 输出结构化序列的任务（trace、plan、code）都该试 explicit boundary token。
- **与 VLA 路线的对比值得思考**：RoboBrain 把"动作"分解成 affordance + trajectory + 下游 policy，π0 直接 end-to-end 出 action chunk。前者可解释、可干预、训练数据要求低；后者 closed-loop、scaling 友好。两条路线在 2025-2026 都没分出胜负，可能最终会融合（high-level brain 输出 trajectory waypoint 作为 condition 给 low-level VLA）。

### Rating

**分数**：2 - Frontier
**理由**：作为 BAAI RoboBrain 系列的 1.0 版本，方法和数据（ShareRobot）都是后续 2.0/2.5 的骨架来源，属于 "embodied brain" 赛道的一个代表工作，因此高于 Archived。但如 Weaknesses 指出——无真机、planning evaluation in-distribution、affordance/trajectory 本体在 2.5 已被推翻，方法本身的奠基性不够达到 Foundation；其定位是"产品演化链的锚点"而非独立的经典参考。
