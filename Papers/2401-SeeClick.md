---
title: "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents"
authors: [Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, Zhiyong Wu]
institutes: [Nanjing University, Shanghai AI Laboratory]
date_publish: 2024-01-19
venue: ACL 2024
tags: [gui-agent, computer-use, VLM]
paper: https://arxiv.org/abs/2401.10935
website: 
github: https://github.com/njucckevin/SeeClick
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents
> - **核心**: 提出"GUI grounding 是 visual GUI agent 的核心瓶颈"这一论断，并通过 grounding 预训练系统验证：grounding 提升 → 下游 agent 任务性能提升
> - **方法**: 在 Qwen-VL 上做 GUI grounding continual pre-training（1M 样本：自动爬取的 web UI + 重组 mobile UI + LLaVA general），把坐标当成自然语言数字直接生成
> - **结果**: ScreenSpot 平均 53.4%（>CogAgent 18B 47.4%，参数减半）；MiniWob 67% 仅用 0.3% Pix2Act 数据；AITW/Mind2Web 都比 Qwen-VL baseline 显著提升
> - **Sources**: [paper](https://arxiv.org/abs/2401.10935) | [github](https://github.com/njucckevin/SeeClick)
> - **Rating**: 3 - Foundation（确立"GUI grounding 是 visual CUA 核心瓶颈"这一 reframing，并产出 ScreenSpot 这一事实标准评测，后续整个 visual CUA 流派都在 build on this insight）

**Key Takeaways:**
1. **GUI grounding bottleneck 论点**：把 "visual GUI agent 性能不行" 重新 frame 为 "LVLM 不会精确点击"，这是后续整个 visual CUA 流派（[[Papers/2504-ScreenSpotPro|ScreenSpot-Pro]]、OS-Atlas、SeeClick 之后大量 follow-up）的起点。
2. **ScreenSpot 基准**：第一个跨 mobile/desktop/web 五平台的真实 GUI grounding 评测集（600+ 截图、1200+ 指令），区分 text vs icon/widget——后续成为 CUA 领域的事实标准。
3. **坐标即语言**：抛弃 1000-bin 词表方案，直接让 LVLM 生成 `(0.49, 0.40)` 这种归一化两位小数。简洁，且实验上点预测略优于 bbox 预测。
4. **数据 recipe 可复现**：Common Crawl 自动抽取 visible-text 元素 + `title` 属性元素得到 ~300K web 页面、约 380K web 样本；mobile 来自 widget captioning + RICO + UI summarization。整个 1M 数据 pipeline 可在 8×A100 24h 内完成 LoRA 持续预训练。
5. **Grounding-agent 强相关**：单独追踪 SeeClick 不同 checkpoint，ScreenSpot 分数与 MiniWob/AITW/Mind2Web 三个下游任务呈一致正相关——这是论文的灵魂图。

**Teaser. SeeClick 总览：grounding 预训练 → ScreenSpot 评测 → 下游 web agent 应用三件套**

![](https://arxiv.org/html/2401.10935v2/x3.png)

---

## Background & Motivation

现有 GUI agent 主流路线是把 HTML/DOM/Android view hierarchy 之类的 structured text 喂给 LLM。论文指出三条根本限制：
1. **结构化文本不是总能拿到**：iOS 和桌面应用尤其难
2. **冗长且丢信息**：HTML 又长又缺图标/布局
3. **每个平台一套观察空间**：HTML / DOM / Android VH 各异，不通用

→ 直接看截图、生成 click 坐标的 visual agent 是更通用的解。但实验发现 LVLM（包括 GPT-4V、Qwen-VL）在 GUI 上根本点不准——**GUI grounding 是核心瓶颈**。

**Figure 1. Text-based agent vs visual SeeClick 的对比**

![](https://arxiv.org/html/2401.10935v2/x2.png)

---

## Method

### 3.1 GUI Grounding 形式化

给定截图 $s$ 和元素集合 $\{(x_i, y_i)\}$，$x_i$ 是文本描述、$y_i$ 是位置（点或 bbox）。LVLM 学习 $p(y \mid s, x)$。

**坐标表示的关键设计**：之前工作（Pix2Struct 等）把图像分 1000 bin 加 1000 个新 token `<p0>...<p999>`。SeeClick 选择更简洁的方案——**把数字当自然语言直接生成**，沿用 Qwen-VL / Shikra 的传统。Prompt 形如 "In the UI, where should I click if I want to <instruction>?"，target 形如 `click (0.49, 0.40)`，标准 cross-entropy。

> 这个选择的代价是数字 token 化效率低、长 sequence 容易飘；好处是不需要词表扩展、跟通用 LVLM pipeline 完全兼容。从结果看是值得的。

### 3.2 数据构造

| Domain | Task | Sample Num |
|---|---|---|
| Web | text_2_point | 271K |
| Web | text_2_bbox | 54K |
| Web | point_2_text | 54K |
| Web | bbox_2_text | 54K |
| Mobile | text_2_point | 274K |
| Mobile | text_2_bbox | 56K |
| Mobile | UI summarization | 48K |
| Mobile | widget captioning | 42K |
| General | LLaVA | 145K |
| **Total** | | **1M** |

**Web data**：从 Common Crawl 抓 ~300K 网页，自动抽两类元素：(1) 有可见文本的元素；(2) 带 `title` 属性的元素（hover 时显示提示文本）。后者是关键——它能把 icon 元素的语义抽出来，而不仅是 text。除 grounding 任务外还加了反向的 web OCR 任务 $p(x|s,y)$。

**Mobile data**：widget captioning（Li et al. 数据，20K 截图 / 40K widgets / 100K descriptions）+ RICO + mobile UI summarization。grounding 数据通过 reverse widget captioning 得到。

**General data**：LLaVA 的 vision-language instruction following 数据，防止 GUI 训练把通用能力洗掉。

**Figure 3. Web 元素自动收集示例（visible text + title 属性两类）**

![](https://arxiv.org/html/2401.10935v2/x4.png)

### 3.3 训练

- Base：Qwen-VL（9.6B，448×448 输入，自带 grounding 能力）
- 持续预训练 ~10K steps（约 1 epoch）
- LoRA fine-tune 视觉编码器 + LLM
- AdamW + cosine schedule，lr=3e-5，global batch=64
- 8×A100 跑约 24h

> 一个有意思的细节：他们偏向用 **point 而非 bbox** 作为输出格式。原因是 UI 元素尺寸差异极大，bbox 学起来更难。最终评测也用 point。

---

## Experiments

### 5.1 ScreenSpot Grounding 结果

ScreenSpot：>600 截图、1200+ 指令，覆盖 iOS/Android/macOS/Windows/Web，区分 text 和 icon/widget。Web 部分故意从 [[Papers/2307-WebArena|WebArena]] 选不同类型网站（development, shopping, forum, tools）。Metric 是 click accuracy（预测点落在 GT bbox 内）。

**Table 1. ScreenSpot 评测结果**

| LVLMs | Size | GUI | Mobile-Text | Mobile-Icon | Desktop-Text | Desktop-Icon | Web-Text | Web-Icon | Avg |
|---|---|---|---|---|---|---|---|---|---|
| MiniGPT-v2 | 7B | ✗ | 8.4 | 6.6 | 6.2 | 2.9 | 6.5 | 3.4 | 5.7 |
| Qwen-VL | 9.6B | ✗ | 9.5 | 4.8 | 5.7 | 5.0 | 3.5 | 2.4 | 5.2 |
| GPT-4V | - | ✗ | 22.6 | 24.5 | 20.2 | 11.8 | 9.2 | 8.8 | 16.2 |
| Fuyu | 8B | ✓ | 41.0 | 1.3 | 33.0 | 3.6 | 33.9 | 4.4 | 19.5 |
| [[Papers/2312-CogAgent\|CogAgent]] | 18B | ✓ | 67.0 | 24.0 | **74.2** | 20.0 | **70.4** | 28.6 | 47.4 |
| **SeeClick** | 9.6B | ✓ | **78.0** | **52.0** | 72.2 | **30.0** | 55.7 | **32.5** | **53.4** |

关键观察：
- 通用 LVLM（含 GPT-4V）在 GUI 上 grounding 严重不行——GPT-4V 仅 16.2%
- SeeClick 平均超 [[Papers/2312-CogAgent|CogAgent]]（18B → 9.6B），尤其 mobile 大幅领先
- **所有模型在 icon/widget 上都很挣扎**——这是论文遗留的 open problem，也是后续 [[Papers/2504-ScreenSpotPro|ScreenSpot-Pro]] 揭示专业软件 icon 仅 4% 的伏笔
- SeeClick 在 Desktop-Text / Web-Text 落后 CogAgent，作者归因于分辨率较低（448 vs CogAgent 的 1120）和训练数据更少

### 5.2 下游 Agent 任务

#### MiniWob

仅用 2.8K 训练集（每 task 50 successful episodes）：

**Table 2. MiniWob 平均成功率**

| Methods | Modality | Dataset | Score |
|---|---|---|---|
| WebGUM | HTML+Image | 2.8K | 65.5 |
| WebGUM | HTML+Image | 347K | 86.1 |
| **SeeClick** | Image | 2.8K | **73.6** (45 tasks) |
| Pix2Act | Image | 1.3M | 64.6 |
| Qwen-VL | Image | 2.8K | 48.4 |
| **SeeClick** | Image | 2.8K | **67.0** (35 tasks) |

- 用同样 2.8K 数据下击败 HTML+Image 的 WebGUM
- 用 0.3% 的数据（2.8K vs 1.3M）击败 vision-only 的 Pix2Act
- 比同 base 模型（Qwen-VL）高 ~20pp，**直接证明 grounding 预训练的价值**

#### AITW（Android in the Wild）

作者特意指出原 split 有 train/test 严重重叠的过拟合风险（同一 instruction 的多条相似轨迹被分到 train/test），改用 **instruction-wise split**。

**Table 3. AITW 平均分**

| Methods | Modality | General | Install | GoogleApps | Single | WebShop | Overall | ClickAcc |
|---|---|---|---|---|---|---|---|---|
| ChatGPT-CoT | Text | 5.9 | 4.4 | 10.5 | 9.4 | 8.4 | 7.7 | - |
| GPT-4V | Image | 41.7 | 42.6 | 49.8 | 72.8 | 45.7 | 50.5 | - |
| Qwen-VL | Image | 49.5 | 59.9 | 46.9 | 64.7 | 50.7 | 54.3 | 57.4 |
| **SeeClick** | Image | **54.0** | **66.4** | **54.9** | 63.5 | **57.6** | **59.3** | **66.4** |

ClickAcc 指标涨 9 pp，再次印证 grounding → click 精度 → 任务成功率链条。

#### Mind2Web

把原本设计给 HTML agent 的 Mind2Web 改造成 vision-only 任务（解析 raw dump 拿到截图和 GT bbox）。

**Table 4. Mind2Web Cross-Task / Cross-Website / Cross-Domain（Step SR）**

| Methods | w/o HTML | Ele.Acc | Op.F1 | Step SR (Cross-Task) | (Cross-Web) | (Cross-Domain) |
|---|---|---|---|---|---|---|
| MindAct (HTML-based) | ✗ | 55.1 | 75.7 | 52.0 | 38.9 | 39.6 |
| GPT-4 (HTML) | ✗ | 41.6 | 60.6 | 36.2 | 30.1 | 26.4 |
| Qwen-VL | ✓ | 15.9 | 86.7 | 13.3 | 9.2 | 12.0 |
| **SeeClick** | ✓ | **28.3** | 87.0 | **25.5** | **16.4** | **20.8** |

- **vision-only Step SR 几乎翻倍**（13.3 → 25.5）
- 但仍显著低于 HTML-based MindAct（52.0），作者诚实承认：从截图预测精确点击坐标比从候选 HTML 元素中选要难得多。这条 gap 是后续 visual CUA 整个流派要解决的问题。

### 5.2.4 Grounding ↔ Agent Performance（论文灵魂图）

**Figure 6. Grounding 能力与下游任务性能的相关性**

![](https://arxiv.org/html/2401.10935v2/extracted/5426719/figures/grounding2agent.png)

把 SeeClick 训练过程中不同 checkpoint 拿出来同时评 ScreenSpot 和三个下游 agent 任务，得到强一致的正相关——这是支撑论文核心 claim 的关键证据。

### 5.2.5 统一 vs 分别训练

**Table 5. SeeClick separate vs unified**

|  | MiniWob | AITW | Mind2Web |
|---|---|---|---|
| Qwen-VL (separate) | 48.4 | 54.3 | 11.5 |
| SeeClick (separate) | 67.0 | 59.3 | 20.9 |
| SeeClick (unified) | 64.1 | 57.1 | 19.5 |

统一训练有轻微下降——三个平台的接口形态差异太大，joint training 暂时还没甜头。

### Action Space

为了统一，SeeClick 用一个混合 action space（来自 AITW + Mind2Web）：
- `click(x,y)`：归一化坐标
- `type("text")`
- `select("value")`：下拉菜单（Mind2Web 风格）
- `swipe(direction)`：上下左右
- `PRESS BACK / HOME / ENTER`
- `TASK COMPLETE / TASK IMPOSSIBLE`

每步输入：instruction + 当前截图 + 前 k=4 步 action 历史。

---

## Limitations（论文自述）

- Action 空间简化到 click + type，缺 drag、double-click 等
- 还需要在每个下游任务上做 task-specific fine-tune 才能多步执行（不是 zero-shot agent）

---

## 关联工作

### 基于
- **Qwen-VL**: SeeClick 的 base LVLM，自带 bbox grounding 能力和 448 分辨率
- **LLaVA**: 通用 vision-language instruction data，防止 GUI 训练洗掉通用能力
- **Common Crawl**: web 截图和 HTML 数据来源
- **RICO** (Deka et al.): mobile UI 数据集，提供 mobile grounding 的扩充
- **widget captioning** (Li et al.): mobile UI 元素描述数据

### 对比
- **[[Papers/2312-CogAgent|CogAgent]]** (18B, 1120 分辨率): 同期 GUI-specific LVLM，主要对比对象。SeeClick 用一半参数 + 较低分辨率仍取得更高平均分
- **Fuyu**: GUI-specific LVLM，但 icon grounding 极差（1.3-4.4%）
- **GPT-4V**: 通用 LVLM 代表，验证"通用 LVLM 在 GUI grounding 上严重不行"
- **WebGUM**: HTML+Image baseline，证明 vision-only 在同等数据量下能赢
- **Pix2Act**: vision-only baseline，证明 grounding pretrain 能用 0.3% 数据击败大规模 SL

### 方法相关
- **Pix2Struct / Shikra / Kosmos-2**: 坐标作为语言生成的传统，SeeClick 沿用并简化（直接两位小数，不用 bin token）
- **Set-of-Mark prompting**: prompt-only 的视觉 grounding 路线，不需要训练但依赖 GPT-4V

### 后续工作（截至 2026-04）
- **OS-Atlas** (2410.23218): 更大规模 grounding 数据 + 多 GUI 平台
- **OS-Genesis** (2412.19723): reverse task synthesis，自动合成 GUI agent 训练 trajectory
- **[[Papers/2504-ScreenSpotPro|ScreenSpot-Pro]]**: 把 ScreenSpot 扩展到专业软件，揭示 icon grounding 仅 4%
- **UGround / ShowUI / UI-TARS**: 都把 SeeClick 列为关键 baseline / 起点

---

## 论文点评

### Strengths

1. **Problem framing 清晰且影响深远**：把"visual GUI agent 不行"重新归因为"GUI grounding 不行"，这个 reframing 本身就是 contribution。后续整个 visual CUA 流派都在沿着这个方向走（OS-Atlas、UGround、ShowUI、ScreenSpot-Pro 等）。属于 [[DomainMaps/CUA|CUA]] DomainMap 中明确认定的 building block。
2. **证据链完整**：不是"我训了个模型 SOTA"，而是"我提出 grounding bottleneck → 建 ScreenSpot benchmark 量化它 → 训 SeeClick 在 benchmark 上验证 → 在三个下游 agent 任务上验证 grounding-agent 强相关"。这种 problem→benchmark→method→correlation 的论证结构值得学习。
3. **方法 simple, scalable, generalizable**：坐标即语言、Common Crawl 自动抽取 web grounding 数据。没有花哨的架构改动，是典型的 data-centric 路线。1M 数据 + LoRA + 24h × 8 A100，复现门槛低。
4. **诚实 Mind2Web 评测**：vision-only Step SR 25.5 远低于 HTML-based MindAct 52.0，作者没有掩饰，明确指出"choosing from HTML candidates 比 predicting click coordinates 容易得多"。
5. **AITW split 修正**：发现并指出原 split 的过拟合风险，提出 instruction-wise split——这是 healthy benchmarking 的范例。

### Weaknesses

1. **分辨率 448×448 是硬伤**：CogAgent 1120 在 desktop-text / web-text 上反超就是证据。Web 和 desktop 上小字、密集 UI 元素需要高分辨率，这点后续 work（OS-Atlas、UI-TARS）都用更高分辨率/多分辨率方案解决。
2. **Icon/widget grounding 仍然差**：即使 SeeClick 在 icon 上也只到 30-50%。后续 [[Papers/2504-ScreenSpotPro|ScreenSpot-Pro]] 揭示专业软件 icon 仅 4%——SeeClick 的训练数据（Common Crawl + RICO）覆盖的主要是消费级 web/mobile，专业软件几乎没碰。
3. **Action space 简化**：drag、scroll-to-element、long-press、double-click、keyboard shortcuts 都没有。真正的 desktop computer-use 需要这些。
4. **下游任务仍需 task-specific fine-tune**：不是 zero-shot agent，每个 benchmark 都要训一遍。这跟"unified visual GUI agent"的 narrative 有 tension。Table 5 也显示 unified 训练有性能下降。
5. **Web 数据自动抽取的 noise 没有讨论**：`title` 属性元素的描述质量参差不齐，开发者写的 alt-text/title 经常是 "image123.png" 之类。论文没有 ablation 数据质量过滤的影响。
6. **基线略弱**：CogAgent 是同期工作，但没有跟 Set-of-Mark prompting + GPT-4V 这种 prompt-only 方案严格对比（虽然报了 GPT-4V 的 raw grounding 数字）。SoM-style 方案在工程上是真实 baseline。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training（pre-training 数据处理、SFT、ScreenSpot evaluation 都开源）
- **模型权重**: SeeClick checkpoint 发布在 HuggingFace `cckevinn/SeeClick`
- **训练细节**: 完整——超参（lr=3e-5, batch=64, ~10K steps）、数据配比（Table 6 的 1M 拆解）、硬件（8×A100 24h）、LoRA 配置代码可见
- **数据集**: 完全开源——ScreenSpot benchmark（box.nju.edu.cn 和 Google Drive 双链接）、SeeClick web grounding pre-training data、爬虫脚本（独立 repo `seeclick-crawler`）

#### Claim 可验证性
- ✅ **SeeClick 在 ScreenSpot 平均 53.4%，超 CogAgent 47.4%**：表格数字齐全，ScreenSpot 已开源，第三方可独立复现
- ✅ **GUI grounding 与下游任务性能正相关**：Figure 6 的 checkpoint-wise 同步评测是直接证据，AITW ClickAcc 9pp 提升、Mind2Web Step SR 翻倍也支持这一点
- ✅ **0.3% 数据击败 Pix2Act**：2.8K vs 1.3M 数据集大小是公开事实，MiniWob 评测协议标准
- ⚠️ **"unified visual GUI agent"**：Table 5 显示 unified 训练有性能下降，"unified" 的 claim 比 narrative 暗示的弱
- ⚠️ **AITW instruction-wise split 公平性**：作者修了 split，但和原 split 的 Auto-UI / CogAgent baseline 数字不能直接比；Table 7 的原 split 数字 SeeClick 只是中等水平
- ⚠️ **数据质量贡献的归因**：1M 数据 vs 1M 通用数据 vs 不同 task mix 的 ablation 缺失，无法判断哪部分数据最重要

### Notes

- 这篇是 [[DomainMaps/CUA|CUA]] DomainMap 的核心 building block，"GUI grounding bottleneck" 论点直接来自这里
- SeeClick → ScreenSpot-Pro → GroundCUA 是 grounding 这条线的主轴。从 SeeClick 53.4% → ScreenSpot-Pro 揭示 icon 4% → GroundCUA 700K 人工标注解决 → GroundNext-3B agentic 任务超越 72B
- > ❓ 论文的"坐标即语言"方案和"bin token" 方案的实验对比只在脚注/preliminary 提了一句，没有正式 ablation。这两种 tokenization 哪个更 scaling-friendly 至今缺系统研究——后续 UI-TARS 用 1000-bin、UGround 用归一化坐标，结论分歧
- > ❓ Common Crawl 的 `title` 属性数据噪声很大（开发者经常写垃圾 alt-text），SeeClick 没有 quality filter ablation。如果做了，可能能进一步压缩数据需求
- 复用价值高的工程细节：(1) Web 元素自动抽取的 visible-text + title 双策略；(2) instruction-wise split 修复 AITW 过拟合；(3) point vs bbox 的实证选择
- 历史地位：在 2024 年初确立了"grounding 是 visual CUA 的核心问题"这一共识，后续整个 CUA 流派都在 build on this insight

### Rating

**Metrics** (as of 2026-04-23): citation=N/A (S2 unavailable, Scholar 未抓取), influential=N/A, velocity=N/A · 27.1mo old; HF upvotes=5; github 477⭐ / forks=30 / 90d commits=0 / pushed 283d ago · stale

**分数**：3 - Foundation
**理由**：problem reframing（GUI grounding 是 visual CUA 的核心瓶颈）+ ScreenSpot benchmark 两项产出都成为方向的 building block，Strengths 第 1-2 条即指出后续 OS-Atlas / UGround / ShowUI / UI-TARS / ScreenSpot-Pro 都沿着此路线 build on 或扩展。外部信号上 SeeClick 已被 visual CUA 主流工作作为事实 baseline、ScreenSpot 已是 de facto grounding 评测。复核 (2026-04-23)：metrics 显示 repo `is_stale` 且 stars=477 恰在 500 阈值下，但这只反映**模型本身**已被更大 VLM 取代（Weaknesses 1、3），Foundation 地位由 **benchmark + reframing** 承担，两者未受影响，故维持 3——符合"只读 rating=3 就能理解方向主脉络"的准入门槛。
