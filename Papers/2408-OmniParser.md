---
title: "OmniParser for Pure Vision Based GUI Agent"
authors: [Yadong Lu, Jianwei Yang, Yelong Shen, Ahmed Awadallah]
institutes: [Microsoft Research, Microsoft Gen AI]
date_publish: 2024-08-01
venue: arXiv
tags: [gui-agent, computer-use, VLM]
paper: https://arxiv.org/abs/2408.00203
website: https://microsoft.github.io/OmniParser/
github: https://github.com/microsoft/OmniParser
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] OmniParser for Pure Vision Based GUI Agent
> - **核心**: 把 GUI screenshot 解析成结构化（bbox + 语义描述）DOM-like 表示，让 GPT-4V 这类通用 VLM 在没有 HTML/view-hierarchy 输入的情况下也能可靠 ground action 到 UI 元素
> - **方法**: 三件套——finetuned YOLOv8 interactable region detector（在 67k 自爬的 webpage DOM bbox 上训练）+ finetuned BLIP-2 icon caption 模型（7k GPT-4o 生成的 icon-description pair）+ OCR；输出 SoM 标注图 + 文本语义列表
> - **结果**: ScreenSpot 平均 73.0%（vs GPT-4V 16.2%）；Mind2Web 不用 HTML 也超过用 HTML 的 GPT-4V baseline；AITW 比最强 GPT-4V+history baseline 高 4.7 pts
> - **Sources**: [paper](https://arxiv.org/abs/2408.00203) | [website](https://microsoft.github.io/OmniParser/) | [github](https://github.com/microsoft/OmniParser)
> - **Rating**: 2 - Frontier（pure-vision GUI parsing 的代表工作，后续 V2/OmniTool 与多篇 computer-use agent 把它作为 perception 前置 baseline，但单步 grounding 评测 + modular 范式正被 end-to-end grounding VLM 逐步取代）

**Key Takeaways:**
1. **Pure-vision parsing 是 cross-platform GUI agent 的瓶颈**：之前 Set-of-Marks 方法都依赖 DOM/view-hierarchy 拿到 bbox ground truth，限制在 web；OmniParser 的核心论点是只要能可靠地从像素拿到 interactable region + 语义，GPT-4V 的 grounding 能力被严重低估
2. **解耦 perception 与 action prediction**：把"识别 icon 是什么"从 GPT-4V 主调用里拆出来，作为前置文本注入 prompt，显著降低 VLM 的 composite-task 负担（SeeAssign hard split: 0.620 → 0.900）
3. **DOM 自动标注是 scalable 的数据源**：从 100k popular URL 用 DOM tree 自动抽 bbox，得到 67k 训练样本，无需人工标注；YOLOv8 detector 还能 zero-shot 泛化到 mobile screen（AITW 上 work）
4. **Web-trained detector 泛化到 mobile**：interactable region detection 在 webpages 上训，但在 AITW (Android) 上替换原本专门为 Android 训的 IconNet 反而带来增益，说明 interactability 概念跨 platform 具有一定通用性

**Teaser. OmniParser 的输入输出 pipeline**：用户任务 + UI screenshot 输入，输出 SoM 标注的 screenshot + 局部语义文本（OCR text + icon description）。

![](https://microsoft.github.io/OmniParser/static/images/flow_merged0.png)

---

## 1. Problem & Motivation

GUI agent 的核心瓶颈不在"理解任务"而在 **action grounding**——把 LLM 预测的高层动作变成屏幕上的精确 click 坐标。已有路线：

- **直接预测 xy 坐标**：GPT-4V 在这上面表现极差，SoM 论文已经指出
- **Set-of-Marks (SoM) prompting**：在原图上叠 numbered bbox，GPT-4V 只需输出 ID。但**已有 SoM 用法都依赖 HTML DOM 或 Android view hierarchy 拿 ground-truth bbox**，所以局限在 web 浏览器或 Android。

OmniParser 的论点：**pure-vision 也能拿到可靠的 SoM bbox**，关键是要有针对性 finetune 过的 detector 和能输出 functional semantics 的 caption 模型。一旦补上这两块，GPT-4V 在 cross-platform GUI 上的能力被严重低估。

> ❓ 这里其实绕过了一个更根本的问题：为什么不直接训一个 end-to-end VLA 风格的 GUI 模型？作者的 implicit answer 是 GPT-4V 已经够强，瓶颈在 perception 这层；但同时期 SeeClick / CogAgent 走的就是端到端路线。这是 perception-action 分离 vs end-to-end 的典型 framing 差异。

---

## 2. Method

OmniParser = **interactable region detector + icon caption + OCR**，三路并联，输出 (SoM 图, 局部语义文本) 给下游 VLM。

### 2.1 Interactable Region Detection

#### Data Curation
- 从 ClueWeb22 取 100k popular URL，渲染网页，从 **DOM tree** 自动抽 interactable element 的 bbox
- 最终得到 **67k unique screenshot + bbox 标注**（95/5 train/val 划分: 63641/3349）
- 没有人工标注，scalable

**Figure 2. Interactable Region Detection 数据集示例**——bbox 来自 webpage DOM tree。
![](https://microsoft.github.io/OmniParser/static/images/curated_data.png)

#### Detector
- **YOLOv8**，20 epochs, batch 256, lr 1e-3, Adam, 4 GPU
- 训练完后与 OCR module 输出的文本 bbox 合并；overlap > 90% 的去重
- 每个 bbox 标 unique ID，labeling 算法尽量减小数字与其他 bbox 的重叠

> ❓ 为什么用 YOLOv8 而不是 Grounding DINO？论文 Table 2 ablation 显示 finetuned YOLOv8 (ID) 比 raw Grounding DINO (GD) 在 ScreenSpot 上高 4.3 pts。但没有对比 finetuned Grounding DINO——所以无法判断是 YOLOv8 架构本身 better 还是单纯是 finetune 数据的功劳。

### 2.2 Local Semantics: Icon Caption

观察：仅给 SoM 标注图，GPT-4V 经常在 "认 icon 是什么" + "选哪个 ID" 这两个 composite task 上失败。**前置注入语义文本**能解耦这两步。

#### Caption 模型
- **BLIP-2** finetune
- 数据集：7185 个 icon-description pair，**GPT-4o 在 ScreenSpot 截图的 detected bbox 上自动生成**——若 GPT-4o 判定是 icon，输出一句话 functionality 描述；不是 icon 也保留并标 "this is not an icon"
- 训练：1 epoch, lr 1e-5 constant, no weight decay, Adam

**Figure 4. BLIP-2 vs finetuned 对比**——原模型描述 icon 的形状颜色，finetune 后能描述功能语义。
![](https://arxiv.org/html/2408.00203v1/extracted/5767271/finetune_icon.png)

### 2.3 Pipeline 整体

最终 prompt 给 GPT-4V 的输入：
1. **图像**：原 screenshot 叠 SoM bbox + ID
2. **文本**：每个 bbox 的 (text 内容 / icon description) 列表 + 用户任务

---

## 3. Experiments

### 3.1 SeeAssign（自构造的 grounding 单步评测）

作者手工造了 112 个任务（Mobile/Desktop/Web 各取自 ScreenSpot），每个任务对应一个明确的 bbox（如 "click on settings"），按 bbox 数量分 easy (<10) / medium (10-40) / hard (>40)。

**Table 1. Local semantics 对 GPT-4V grounding 准确率的提升**

| Setting | Easy | Medium | Hard | Overall |
| ---- | ---- | ---- | ---- | ---- |
| GPT-4V w.o. local semantics | 0.913 | 0.692 | 0.620 | 0.705 |
| GPT-4V w. local semantics | 1.000 | 0.949 | 0.900 | **0.938** |

bbox 越多，semantic 文本前置的增益越大——印证 "composite task 拆分" 的假设。

### 3.2 ScreenSpot

跨 Mobile/Desktop/Web 三平台，600+ screenshot。

**Table 2. ScreenSpot 全面对比**

| Methods | Size | Mobile-T | Mobile-I | Desk-T | Desk-I | Web-T | Web-I | Avg |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Fuyu | 8B | 41.0 | 1.3 | 33.0 | 3.6 | 33.9 | 4.4 | 19.5 |
| CogAgent | 18B | 67.0 | 24.0 | 74.2 | 20.0 | 70.4 | 28.6 | 47.4 |
| SeeClick | 9.6B | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| GPT-4V | - | 22.6 | 24.5 | 20.2 | 11.8 | 9.2 | 8.8 | 16.2 |
| OmniParser (w.o. LS, w. GD) | - | 92.7 | 49.4 | 64.9 | 26.3 | 77.3 | 39.7 | 58.4 |
| OmniParser (w. LS + GD) | - | 94.8 | 53.7 | 89.3 | 44.9 | 83.0 | 45.1 | 68.7 |
| **OmniParser (w. LS + ID)** | - | 93.9 | 57.0 | 91.3 | 63.6 | 81.3 | 51.0 | **73.0** |

GPT-4V 16.2 → 73.0，碾压同期 finetuned GUI 专用模型 (SeeClick 53.4, CogAgent 47.4)。LS 和 finetuned ID 各自有显著贡献（ablation: w/o LS 掉 10 pts，GD→ID 涨 4.3 pts）。

### 3.3 Mind2Web

Web navigation，3 个 split: Cross-Website / Cross-Domain / Cross-Task。

**Table 3. Mind2Web Step Success Rate（摘 SR 指标）**

| Methods | HTML-free | image | CW-SR | CD-SR | CT-SR |
| ---- | ---- | ---- | ---- | ---- | ---- |
| SeeClick | ✓ | ✓ | 16.4 | 20.8 | 25.5 |
| MindAct (GPT-4 + HTML) | × | × | 38.9 | 39.6 | 39.6 |
| GPT-4V+SOM (HTML bbox) | × | ✓ | 32.7 | 23.7 | 20.3 |
| GPT-4V+textual choice | × | ✓ | 32.4 | 36.8 | **40.2** |
| OmniParser (w. LS + GD) | ✓ | ✓ | 36.1 | 36.8 | 38.7 |
| **OmniParser (w. LS + ID)** | ✓ | ✓ | **36.5** | **42.0** | 39.4 |

关键观察：**OmniParser 不用 HTML 也能比用 HTML 的 GPT-4 / GPT-4V baselines 强**（CW +4.1, CD +5.2 over GPT-4V+textual）。Cross-Task 上略输 0.8 pts。

### 3.4 AITW（Android in the Wild）

30k instruction, 715k trajectory 的 Android navigation benchmark。

**Table 4. AITW Overall**

| Methods | General | Install | GoogleApps | Single | WebShop | Overall |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GPT-4V image-only | 41.7 | 42.6 | 49.8 | 72.8 | 45.7 | 50.5 |
| GPT-4V + history (IconNet) | 43.0 | 46.1 | 49.2 | 78.3 | 48.2 | 53.0 |
| **OmniParser (w. LS + ID)** | 48.3 | 57.8 | 51.6 | 77.4 | 52.9 | **57.7** |

最重要的发现：**ID detector 在 webpage 上训，直接替换专为 Android 训的 IconNet 还能涨 4.7 pts**——interactability 概念跨平台泛化。

> ❓ 但这里没控制 caption 模型的影响，因为 IconNet baseline 没有 LS。所以 +4.7 里有多少来自 detector swap、多少来自 LS，分不清。

### 3.5 Plugin to other VLMs（来自 website）

OmniParser 也作为 plugin 接到 Phi-3.5-V 和 Llama-3.2-V，相同的 finetuned ID + LS 模式带来一致提升。说明前置 perception 这个 framework 对底座 VLM 不绑定。

---

## 4. Failure Modes（作者诚实讨论的部分）

1. **Repeated icons/texts**：7 个一样的 alarm 按钮，GPT-4V 选不准——SoM ID 不足以区分语义重复元素
2. **Coarse OCR bbox**：OCR 把整段文字框住，但 click point 用 box center，落在 hyperlink 之外 → 失败。作者建议把 OCR 和 interactable detection 合并到一个模型
3. **Icon misinterpretation**：三个点 icon 既可能是 "more" 也可能是 "loading"——caption 模型只看裁剪的小图，缺 global context

---

## 关联工作

### 基于
- **Set-of-Marks Prompting** [Yang et al. 2023]: OmniParser 的 visual prompting 范式直接来自 SoM，关键扩展是把 bbox 来源从 HTML/view-hierarchy 换成 finetuned 视觉 detector
- **BLIP-2** [Li et al. 2023]: icon caption 模型 backbone
- **YOLOv8**: interactable detector 架构
- **ClueWeb22** [Overwijk et al. 2022]: 数据爬取的 seed URL 来源

### 对比
- **SeeClick** [Cheng et al. 2024]: end-to-end finetuned GUI VLM，OmniParser 在 ScreenSpot 上以 modular pipeline + GPT-4V 大幅超越（73.0 vs 53.4）
- **CogAgent** [Hong et al. 2023]: 18B GUI 专用 VLM，同样被 OmniParser 超越
- **SeeAct** [Zheng et al. 2024]: 同样用 GPT-4V，但依赖 HTML 抽 element，OmniParser 在不用 HTML 的情况下平了或超过
- **MindAct** [Deng et al. 2023]: Mind2Web 原始 baseline，用 HTML，OmniParser 在 Cross-Domain/Website 上反超

### 方法相关
- **Ferret-UI**, **CogAgent**, **Fuyu**: 同期端到端 mobile/GUI VLM，代表另一条路线（不分离 perception 和 action）
- **IconNet** [Sunkara et al. 2022]: AITW 默认 detector，OmniParser 的 ID 直接替换它并涨点
- **Mobile-Agent / Mobile-Agent-v2**: 类似 modular agent framework，在 mobile domain
- **OSWorld**: cross-platform OS-level benchmark，OmniParser 没在上面评测，是一个明显的 next step

---

## 论文点评

### Strengths

1. **Framing 简洁，问题定义清楚**：把 GUI agent 的瓶颈从"VLM 不够强"重新 frame 成"perception 层不够强"，并且用一组干净的 ablation（w/o LS, w/o ID）证明了这个 framing。
2. **Cross-platform 泛化的实证发现 valuable**：webpage-trained ID detector 能直接用在 Android 上 (+4.7 pts over IconNet)，这是个 non-trivial 的正向 transfer 信号。
3. **Pipeline 简单且 scalable**：YOLOv8 + BLIP-2 + OCR，没有任何花哨架构；数据全部 DOM 自动抽 + GPT-4o 自动标注，完全可以 scale。
4. **诚实讨论 failure mode**：明确指出 repeated elements、coarse OCR、icon misinterpretation 三种典型失败，没有粉饰。

### Weaknesses

1. **Ablation 不完整**：(a) 没对比 finetuned Grounding DINO，无法 isolate "YOLOv8 architecture" vs "finetune data" 的功劳；(b) AITW 上 IconNet baseline 没有 LS，所以 +4.7 pts 的归因混淆了 detector swap 和 LS 增益。
2. **Caption 模型的 noisy supervision 没量化**：7k 训练集是 GPT-4o 生成的，不是 human label。作者没分析 GPT-4o 自己的 icon naming 准确率上限，所以 caption 质量本身有 invisible ceiling。
3. **End-to-end vs modular 的 trade-off 未讨论**：模块化 perception + frozen GPT-4V 在 latency/cost 上比端到端 GUI VLA 显然劣（每步至少 3 次模型调用 + 1 次大模型推理），但论文没提这点。
4. **Coordinate prediction 用 bbox center 是 hardcoded heuristic**，AITW failure case 1 已经暴露这个问题，但没尝试修正。
5. **Benchmark 都是 single-step 或 short-horizon evaluation**：ScreenSpot 是单步 grounding，Mind2Web 用的是 step-wise success rate（不是 trajectory-level）；OSWorld 这种长序列任务上的表现没测。

### 可信评估

#### Artifact 可获取性
- **代码**: inference 完整开源（github.com/microsoft/OmniParser，含 demo）；training 代码未在 main paper 提及
- **模型权重**: 已发布 finetuned YOLOv8 detector + BLIP-2 caption checkpoints（HuggingFace），后续 V2 checkpoints 在 2025/02 发布
- **训练细节**: 较完整——detector (20 epochs, bs 256, lr 1e-3, Adam, 4 GPU), caption (1 epoch, lr 1e-5, Adam)；数据规模 67k + 7k 都说明
- **数据集**: 67k interactable detection 数据集和 7k icon-caption 数据集是否公开 release，未明确说明（仅说明数据收集来自 ClueWeb22 + GPT-4o）

#### Claim 可验证性
- ✅ ScreenSpot / Mind2Web / AITW 数值改进：标准 benchmark + 公开 inference code，可独立复现
- ✅ ID detector 跨平台泛化（web → mobile）：AITW 的 +4.7 是直接 swap，可验证
- ⚠️ "GPT-4V 的 GUI 能力被严重低估"：定性 claim，依赖于 OmniParser 本身的 ceiling；如果 perception 模块还有缺陷（如失败案例），那 GPT-4V 的真实 ceiling 还是低估的
- ⚠️ "interactability 概念跨平台通用"：仅基于 web → Android 一个方向的实验，没在 iOS / desktop OS-level 任务上系统验证
- ⚠️ Caption 模型的 functionality 描述准确率：用 GPT-4o 自标，没有 human-annotated holdout 评测，质量上限不明

### Notes

- **核心洞察**: "把 perception bottleneck 解耦出来"是个简单但有效的 framing。后续 OmniParser V2 (2025/02) + OmniTool (Windows VM control) 验证了这条路线的工程可扩展性。
- **对我个人的启发**: 在 GUI/computer-use agent 这条线，**模块化 perception + 通用大脑** 与 **end-to-end VLA** 是两条平行路线。OmniParser 的成功说明前者短期内可能更 practical（不用大规模 GUI 数据从头训），但长期可能被 end-to-end 替代——类似 vision pipeline 的 hand-crafted feature → CNN 的转变。
- **可借鉴方法**: 用 DOM 自动抽 bbox + GPT-4o 自动生成 functionality description，这种"用现成大模型 bootstrap 标注"的 pattern 在 VLA / robotics affordance label 上也适用
- **疑问/后续可看**:
  - 这篇 single-step grounding 占大头的评测在 long-horizon 场景下是否还成立？OSWorld / Visual-WebArena trajectory-level success 才是最终的考验
  - V2 解决了 V1 的哪些 failure mode？特别是 repeated element 和 coarse OCR 这两块
  - 与 SeeClick / Aria-UI / UGround 这些后来更强的 end-to-end GUI grounding 模型的 head-to-head 比较

### Rating

**分数**：2 - Frontier
**理由**：OmniParser 是 pure-vision GUI parsing 的代表工作，被后续 computer-use agent（UFO、OmniTool、多篇 2025 agent 论文）广泛作为 perception 前置 baseline，github 影响力显著（V2 发布后社区采纳度持续扩大），符合 Frontier 的"必须比较的 Baseline / 方法范式的代表"标准。但不到 Foundation——Weakness 里指出的 modular pipeline 在长序列任务上未验证、Benchmark 仅 single-step 评测，加上外部信号显示 end-to-end grounding VLM（UGround、Aria-UI、UI-TARS 等）正在同一赛道持续提升 SOTA，把"perception + 通用大脑"范式本身推向 archived 风险中，因此不是方向脉络的奠基节点。
