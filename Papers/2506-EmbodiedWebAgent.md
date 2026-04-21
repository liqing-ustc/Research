---
title: "Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence"
authors: [Yining Hong, Rui Sun, Bingxuan Li, Xingcheng Yao, Maxine Wu, Alexander Chien, Da Yin, Ying Nian Wu, Zhecan James Wang, Kai-Wei Chang]
institutes: [UCLA]
date_publish: 2025-06
venue: NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)
tags: [web-agent, navigation, task-planning]
paper: https://arxiv.org/abs/2506.15677
website: https://embodied-web-agent.github.io/
github: https://github.com/Embodied-Web-Agent/Embodied-Web-Agent
rating: 2
date_added: 2026-04-16
---
## Summary

> [!summary] Embodied Web Agents
> - **核心**: 提出跨物理-数字域的 AI agent 范式，构建统一仿真平台和 ~1.5k 任务 benchmark
> - **方法**: 集成 AI2-THOR (indoor) + Google Earth (outdoor) + 5 个功能性 web 界面，覆盖 cooking/navigation/shopping/traveling/geolocation
> - **结果**: 最佳模型 overall accuracy 远低于人类（cooking 6.4% vs 77%），66.6% 错误是 cross-domain integration failure
> - **Sources**: [paper](https://arxiv.org/abs/2506.15677) | [website](https://embodied-web-agent.github.io/) | [github](https://github.com/Embodied-Web-Agent/Embodied-Web-Agent)
> - **Rating**: 2 - Frontier（首次系统化将 web + embodied 统一为 cross-domain benchmark，NeurIPS 2025 Spotlight，但尚未成为方向 de facto 标准）

**Key Takeaways:**
1. **跨域集成是瓶颈**: 当前 LLM agent 在 web-only 和 embodied-only 子任务上表现尚可，但跨域整合时 overall accuracy 骤降（cooking 最佳模型仅 6.4% vs 人类 77%），66.6% 的 cooking 错误属于 cross-domain errors
2. **Web > Embodied**: 所有模型在 web-only accuracy 上一致高于 embodied-only accuracy，说明当前模型处理数字信息的能力远强于物理交互
3. **主动探索+Web 信息显著提升 geolocation**: 相比 passive baseline (FairLocator)，embodied agent 在 city-level 准确率提升 10+%，甚至查询本身（formulating search queries）就能作为 self-supervision 帮助推理

**Teaser. Embodied Web Agents 概念范式示意：agent 在物理环境（橙色）和 Web 环境（蓝色）之间流畅切换，完成 traveling、cooking、geolocation 等跨域任务。**
![](https://embodied-web-agent.github.io/static/imgs/teaser.png)

<video src="https://embodied-web-agent.github.io/static/imgs/demo.mp4" controls muted playsinline width="720"></video>

---

## Introduction

当前 AI agent 普遍是 siloed 的：web agent 在屏幕上检索信息和推理，embodied agent 在物理世界中感知和行动，但二者很少结合。这种分离限制了 agent 解决需要跨域智能的任务的能力——比如根据网上食谱做菜、用动态地图数据导航、利用 web 知识理解现实地标。

论文提出三个核心挑战：
1. **Perceptual grounding problem**: 如何将抽象的数字指令（如"cook potato and egg until golden brown"）与高维物理世界数据流对齐？
2. **Cross-domain planning**: agent 如何决定何时在物理行动和数字信息检索之间切换？尤其当两个域的信息矛盾时（如地图路线 vs 实际道路封闭）
3. **Persistent representation**: agent 需要维护一个连贯的跨域表示——在 web 操作时回忆物理经验，在物理行动时检索数字知识

## The Embodied Web Agent Task Environments

环境形式化为 $E = \langle S, A, O, T \rangle$，其中 $S$ 是 combined physical-digital state space，$A$ 是跨域 action space，$O$ 包含 embodied input $o_t^e$ 和 web perception $o_t^w$，$T: S \times A \to S$ 是确定性转移函数。

**Figure 2. 任务完成的示例 pipeline：agent 在 web（蓝色）和 embodied（橙色）环境间交替切换，完成从 Time Square 到 Penn Station 的导航+问答任务。**
![](https://embodied-web-agent.github.io/static/imgs/example.png)

### Outdoor Environment

基于 Google Street View 和 Google Earth API 构建，选取纽约、波士顿、费城、匹兹堡四座城市。环境建模为无向图 $G = (V, E)$，每个节点 $v \in V$ 是一个 GPS 坐标，关联东南西北四个方向的 field-of-view 图像。Agent 通过观察视觉输入、访问邻居节点集、利用朝向信息来推理空间转移。

### Indoor Environment

使用 AI2-THOR 构建逼真的 3D 室内厨房场景。物体带有属性和状态（isSliced、isCooked、parentReceptacles 等），随 agent 执行物理动作动态更新。State evaluator 对比当前厨房状态与理想目标状态来判定任务完成度。

### Web Environment

由五个功能性网站组成，使用 React.js 前端 + FastAPI 后端：
- **Homepage**: 导航 hub，链接所有 task-specific 网站
- **Recipe website**: 按食材、饮食偏好、菜系类型浏览/搜索/筛选食谱
- **Shopping website**: 购物车管理、食材查找、模拟结算
- **OpenStreetMap**: 位置搜索、地址查询、地理实体探索
- **Wikipedia**: 百科信息检索、entity linking、multi-hop reasoning

部分网站改编自 [[2307-WebArena|WebArena]] benchmark。

## The Embodied Web Agents Benchmark Construction

Benchmark 包含约 **1.5k 任务**，覆盖 5 个领域：

- **Navigation** (144 tasks): 将 web 方向指令 ground 到 embodied 导航，需要双向 web-embodied 交互
- **Shopping** (216 tasks): 线上比价+下单 → 线下导航取货，测试多源整合和序列规划
- **Traveling** (110 tasks): 导航中遇到地标时查 Wikipedia、理解建筑风格、将文字描述 ground 到视觉观察
- **Cooking** (911 tasks): 识别厨房食材 → 在线搜索匹配食谱 → 按步骤执行，需持续对齐 web 指令与物理状态
- **Geolocation** (142 tasks): 受 GeoGuessr 启发，agent 在环境中主动探索+查 web 来判断位置，改编自 FairLocator

## Experiments

### Baseline LLM Agents

评估四个 baseline：GPT-4o(-mini)、Gemini 2.0 Flash、Qwen-VL-Plus、InternVL2.5-latest。Web 观测遵循 VisualWebArena 设置。

### Evaluation Metrics

四个评估维度：
- **Overall Accuracy**: 完整跨域任务完成率
- **Web-only Accuracy**: 仅 web 子任务完成率
- **Embodied-only Accuracy**: 仅 embodied 子任务完成率
- **Overall Completion Rate**: 已满足状态条件占总目标条件的比例

### Result Analysis

**Table 2. Outdoor 任务性能（Navigation / Shopping / Traveling）**

| Task | Metric | GPT | Gemini | Qwen | Intern | Human |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Navigation | Overall Accuracy | 34.72 | 30.56 | 15.97 | 13.19 | 90.28 |
| Navigation | Web-only Acc | 69.44 | 67.36 | 57.64 | 38.89 | 92.36 |
| Navigation | Embodied-only Acc | 48.61 | 46.53 | 31.25 | 23.61 | 90.97 |
| Shopping | Overall Accuracy | 25.46 | 23.61 | 13.89 | 10.65 | 92.59 |
| Shopping | Web-only Acc | 39.35 | 37.50 | 23.15 | 17.13 | 93.06 |
| Shopping | Embodied-only Acc | 34.26 | 32.41 | 17.59 | 12.96 | 93.98 |
| Traveling | Overall Accuracy | 30.91 | 25.45 | 11.82 | 9.09 | 91.82 |
| Traveling | Web-only Acc | 57.27 | 53.64 | 41.82 | 25.45 | 94.55 |
| Traveling | Embodied-only Acc | 47.27 | 44.55 | 29.09 | 19.09 | 92.73 |

**Insights**: GPT-4o-mini 一致领先但仍远低于人类。Web-only accuracy 始终高于 embodied-only accuracy，说明模型处理数字信息的能力强于物理导航。Shopping 和 traveling 涉及更丰富的跨域交互和更长序列，overall accuracy 显著低于 navigation。

**Table 3. Cooking 任务性能（Vision vs Text modality）**

| Metric | Vision-GPT | Vision-Gemini | Vision-Qwen | Vision-Intern | Text-GPT | Text-Gemini | Text-Qwen | Text-Intern | Human |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Overall Acc | 5.4 | 4.1 | 0.6 | 0.0 | 6.4 | 5.8 | 1.5 | 0.4 | 77.08 |
| Completion Rate | 40.26 | 35.62 | 15.91 | 9.73 | 39.16 | 38.92 | 17.20 | 10.02 | 85.37 |
| Web Acc | 59.71 | 47.74 | 28.65 | 10.64 | 57.08 | 62.23 | 35.89 | 15.58 | 100 |
| Embodied Acc | 8.7 | 6.1 | 2.2 | 0.9 | 10.5 | 8.2 | 4.1 | 1.3 | 77.08 |

**Insights**: 最佳模型（text-based GPT-4o）overall accuracy 仅 6.4%，而人类达到 77.08%。Text-based 模型（使用 structured scene graph）一致优于 vision-based（使用 first-person view），说明当前模型在 cooking context 下的 visual grounding 仍然很弱。Completion rate 适中（~40%），说明模型能完成部分子任务但在 full cross-domain integration 上失败。

**Table 4. Geolocation 任务性能（FairLocator baseline vs Embodied Web Agent）**

| Setting / Model | Continent | Country | City | Street | All |
| ---- | ---- | ---- | ---- | ---- | ---- |
| FairLocator: GPT-4o-mini | 90.85 | 81.69 | 73.24 | 1.41 | 1.41 |
| FairLocator: Gemini-2.0-Flash | 93.66 | 85.92 | 78.17 | 0.70 | 0.70 |
| Embodied: GPT-4o-mini | 97.18 | 90.85 | 85.21 | 3.52 | 3.52 |
| Embodied: Gemini-2.0-Flash | 97.18 | 94.37 | 85.21 | 4.23 | 4.23 |
| Embodied: Qwen-VL-Plus | 80.28 | 69.01 | 49.30 | 0.00 | 0.00 |
| Embodied: InternVL2.5-Latest | 93.62 | 77.30 | 57.45 | 2.13 | 1.42 |

**Insights**: 主动探索+Web 访问（Embodied Web Agent）在所有模型上一致优于 passive baseline（FairLocator），尤其在 city 和 street 粒度上提升显著。有趣的是，即使 Wikipedia 搜索结果 noisy，formulating search queries 本身也能帮助 agent 更自信地推理——查询行为作为一种 self-supervision。

### Error Analysis

对 GPT-4o cooking 任务的错误类型分析：

- **Cross-domain errors (66.6%)**: 占绝对主导
  - 23.7% stuck in embodied environment（重复执行无关物理动作，不切回 web）
  - 16.7% switching without action（频繁切换环境但不执行有意义操作）
  - 13.2% stuck in web environment（无限点击"next"翻食谱页面）
  - 11.8% instruction-action misalignment（如食谱说"slice the apple"却去切 lettuce）
- **Embodied errors (14.6%)**: 5.2% 无法导航到可交互物体，4.5% 重复动作
- **Web errors (8.0%)**: 3.1% page loop，4.3% repeated web actions

关键发现：isolated domain errors 远少于 cross-domain integration failures，确认跨域集成是 embodied web agency 的核心瓶颈。

---

## 关联工作

### 基于
- [[2307-WebArena|WebArena]]: 提供了部分 web 环境（OpenStreetMap、Wikipedia）的基础，web observation 也遵循 VisualWebArena 设置
- AI2-THOR: 室内 embodied 环境的仿真基础
- FairLocator: geolocation 任务的 baseline 和数据来源

### 对比
- MiniWoB / WebShop / Mind2Web: 纯 web agent benchmark，不涉及 embodied interaction
- ALFRED / BEHAVIOR: 纯 embodied benchmark，不涉及 web interaction
- GPT-4o / Gemini 2.0 Flash / Qwen-VL-Plus / InternVL2.5: 作为 baseline 模型评估

### 方法相关
- VisualWebArena: web observation 的 Set-of-Marks (SoM) annotation 方法
- Google Street View API: outdoor environment 的视觉数据来源

---

## 论文点评

### Strengths

1. **问题定义有价值**: 将 web agent 和 embodied agent 两个长期独立发展的方向统一到一个框架下，提出了一个真实且有挑战性的研究范式。日常生活中大量任务确实需要同时利用 web 信息和物理交互
2. **Benchmark 设计系统**: 5 个领域覆盖了从简单（navigation）到复杂（cooking）的多种跨域交互模式，评估维度（overall / web-only / embodied-only / completion rate）设计合理，能精确定位 failure mode
3. **Error analysis 有洞察**: 66.6% cross-domain error 的发现直指当前 LLM agent 的核心短板——不是不会做子任务，而是不会在域间切换和协调
4. **NeurIPS Spotlight**: 数据集和代码已开源，benchmark web 环境可直接访问

### Weaknesses

1. **仿真与现实的 gap**: 论文自己也承认 simulated agents 可能无法捕捉真实物理-数字交互的复杂性和不可预测性。AI2-THOR 的厨房和 Google Street View 与真实机器人操作差距较大
2. **Web 环境较简化**: 5 个网站虽然功能完整，但与真实 web 的复杂度（广告、动态内容、认证、多跳搜索）相比仍然是 controlled setting
3. **缺少方法创新**: 论文主要是 benchmark contribution，没有提出新的 agent architecture 来改善 cross-domain integration。所有 baseline 都是 off-the-shelf LLM
4. **Geolocation 的 "All" 指标极低**: 即使有 web 和 exploration，street-level accuracy 仍然 < 5%，说明当前 benchmark 在某些维度可能过难，难以区分方法改进

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only（提供了 baseline 运行代码和环境搭建脚本）
- **模型权重**: 无自有模型，使用的均为商业 API（GPT-4o、Gemini 等）
- **训练细节**: 不适用（benchmark 论文，无训练）
- **数据集**: 开源（GitHub 仓库 + 在线 benchmark web 环境 http://98.80.38.242:1220/ ）

#### Claim 可验证性
- ✅ 性能数据：四个模型在五个任务域上的详细数值均有报告，可通过开源代码复现
- ✅ Error analysis 分类与百分比：给出了具体错误类型和占比
- ⚠️ "novel paradigm" 的 claim：虽然统一 web+embodied 确实较新，但 ALFRED 等之前的工作已经涉及 instruction following + embodied action，novelty 主要在于 web interaction 的加入而非全新范式
- ⚠️ 人类基线的具体评估方式未充分说明（如人类测试者数量、training 程度、inter-rater agreement）

### Notes

### Rating

**分数**：2 - Frontier
**理由**：这是第一个系统将 web agent + embodied agent 统一到同一 benchmark 的工作，NeurIPS 2025 D&B Spotlight，benchmark 设计系统、error analysis（66.6% 跨域错误）揭示了 LLM agent 的核心短板，具备作为 cross-domain agent 评测前沿的条件；但它本身是 benchmark contribution 无方法创新（见 Weaknesses 3），环境仍是受控仿真（Weaknesses 1-2），方向尚新也未形成 ALFRED / WebArena 级的 de facto 地位，所以不到 3 - Foundation。
