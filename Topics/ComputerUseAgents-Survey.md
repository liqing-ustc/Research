---
title: "Computer-Use Agents 领域调研"
tags: [computer-use, gui-agent, VLM, RL]
date_updated: "2026-04-02"
year_range: 2024-2026
papers_analyzed: 23
---
## Overview

Computer-use Agent（也称 GUI Agent、OS Agent）是一类能够在数字设备（桌面、手机、浏览器）上，根据自然语言指令自主完成复杂任务的 AI 系统。该领域在 2024-2025 年经历了爆发式增长，从早期的 prompt engineering + 商用模型组合，迅速演进为端到端训练的 native agent 路线。

**核心问题**：如何让 AI agent 像人类一样观察屏幕、理解界面、规划操作序列并精确执行？这涉及三个核心能力：**GUI Grounding**（将语言指令映射到屏幕坐标）、**Planning**（将复杂任务分解为可执行步骤）、**Error Recovery**（识别并从错误中恢复）。

**研究现状**：
- 领域活跃度极高，已有 3 篇系统综述（[[Papers/2501-ACUSurvey|ACU Survey]]、[[Papers/2411-GUIAgentSurvey|GUI Agent Survey]]、[[Papers/2508-OSAgentsSurvey|OS Agents Survey]]），后者被 ACL 2025 接收为 Oral
- OSWorld benchmark 上的 SOTA 从 2024 年初的 12.24% 飙升至 2025 年底的 72.6%（超人类水平 72.36%），一年半内提升了约 6 倍
- 产业界深度参与：Anthropic Claude Computer Use、OpenAI Operator/CUA、ByteDance UI-TARS、Google Project Mariner、Apple Intelligence 均已布局
- 技术路线从 framework-based（依赖 prompt + 工具链）转向 native agent（端到端训练），RL 成为核心训练范式

## 技术路线

### 路线一：End-to-End Native Agent Models

**核心思路**：将 GUI 理解、规划和执行统一到一个端到端训练的 VLM 中，仅以截图为输入，直接输出动作。

**代表工作**：
- [[Papers/2501-UITARS|UI-TARS]]（ByteDance, 2025.01）：基于 Qwen-2-VL，通过感知增强（5 项训练任务）、统一 action space、System-2 reasoning（6 种思维模式）和 DPO reflection tuning，在 10+ benchmark 上首次 SOTA
- [[Papers/2509-UITARS2|UI-TARS-2]]（ByteDance, 2025.09）：引入 data flywheel + multi-turn RL（PPO 变体含 Decoupled GAE），OSWorld 达 47.5%，扩展到游戏和 SWE 场景
- [[Papers/2508-ComputerRL|ComputerRL]]（Tsinghua/Z.AI, 2025.08）：提出 API-GUI 统一范式和 Entropulse（RL/SFT 交替解决 entropy collapse），9B 模型在 OSWorld 达 48.9% 超越 o3

**早期先驱**：
- [[Papers/2312-CogAgent|CogAgent]]（Tsinghua/Zhipu, 2023.12, CVPR 2024 Highlight）：18B VLM，首创 dual-resolution cross-attention（224+1120）解决高分辨率 GUI 理解，首个纯视觉方法在 Mind2Web 上超越 HTML-based LLM
- [[Papers/2508-OpenCUA|OpenCUA]]（HKU/XLANG Lab, 2025.08, NeurIPS 2025 Spotlight）：完整开源 pipeline（AgentNet 标注工具 + 22.6K trajectories 数据集 + reflective long CoT 训练），OpenCUA-72B 在 OSWorld-Verified 达 45.0% 开源 SOTA

**RL 训练基础设施**：
- [[Papers/2509-DARTGUI|DART-GUI]]（2025.09）：解耦异步 RL 框架，四模块非阻塞架构（环境集群/rollout 服务/数据管理器/训练器）+ 自适应数据策略（experience pool、dynamic rollout、high-entropy step selection、truncated IS），环境利用率从 12.2%→67.7%（5.5×），7B 模型 OSWorld 42.13% 超越 Claude-4-Sonnet

**优势**：知识可跨平台迁移、无需手工 prompt、可持续自我改进
**局限**：训练资源需求巨大（数千 VM），推理延迟高，safety 未充分讨论

### 路线二：Compositional Framework-based Agents

**核心思路**：将 agent 分解为多个功能模块（规划器、执行器、grounding 专家），用不同模型/工具承担不同子任务。

**代表工作**：
- [[Papers/2504-AgentS2|Agent S2]]（Simular, 2025.04）：Manager-Worker 层级 + Mixture of Grounding（视觉/文本/结构化 3 个专家），proactive planning 持续重规划，OSWorld 34.5%
- [[Papers/2510-ScalingAgents|Scaling Agents / Agent S3]]（Simular, 2025.10, ICML）：提出 Behavior Judge 框架，multi-rollout wide scaling + 结构化轨迹评估，OSWorld **72.6% 超人类水平**。核心创新是 behavior narrative——将多模态轨迹压缩为可比较的结构化文本

**优势**：模块化设计灵活、可插拔不同 grounding 专家、test-time scaling 潜力大
**局限**：依赖强力商用模型（GPT-5、Opus 4.5），多 rollout 假设独立初始状态在真实环境难以满足

### 路线三：GUI Grounding 专用模型

**核心思路**：专注于 GUI grounding 这个核心瓶颈——给定截图和语言指令，精确定位目标 UI 元素。

**代表工作**：
- [[Papers/2401-SeeClick|SeeClick]]（Nanjing U, 2024.01, ACL 2024）：首次明确 grounding 是 visual agent 核心瓶颈，通过 continual pre-training 构建 grounding 模型，创建 ScreenSpot benchmark
- [[Papers/2511-GroundCUA|GroundCUA]]（Mila/ServiceNow, 2025.11）：构建高质量 dense annotation 数据集（56K 截图，64 元素/截图），GroundNext-7B 在 ScreenSpot-Pro 达 52.9%，3B 模型在 agentic 任务超 72B 模型
- [[Papers/2504-ScreenSpotPro|ScreenSpot-Pro]]（NUS, 2025.04）：专业高分辨率 GUI grounding benchmark，揭示 icon 识别仅 4% 准确率的严峻挑战

**Foundation Models & 感知工具**：
- [[Papers/2410-OSAtlas|OS-Atlas]]（Shanghai AI Lab, 2024.10）：构建最大开源跨平台 GUI grounding 语料库（13.58M 元素，2.24M 截图），7B 模型在多个 zero-shot benchmark 超越 GPT-4o
- [[Papers/2411-ShowUI|ShowUI]]（NUS/Microsoft, 2024.11, CVPR 2025）：轻量 2B 模型，核心创新 UI-guided token selection（connected graph 减 33% token），仅 256K 数据达 75.1% zero-shot grounding
- [[Papers/2408-OmniParser|OmniParser]]（Microsoft, 2024.08）：模块化屏幕解析工具（YOLOv8 检测 + BLIP-v2 icon 描述 + OCR + Set-of-Marks），作为 plug-in 增强任意 VLM 的 GUI agent 能力，ScreenSpot 73.0%

**关键发现**：
- Grounding 能力与下游 agent 任务性能强正相关（SeeClick）
- 数据质量 > 数据规模：700K 高质量样本胜过 9M 自动采集数据（GroundCUA）
- Icon/非文字 UI 元素的 grounding 是核心未解难题（ScreenSpot-Pro）
- Token 效率关键：token selection 优于 token merging（ShowUI），positional encoding 对 grounding 至关重要
- 模块化 vs 端到端仍是 open question：OmniParser（外部检测）vs SeeClick（端到端训练）各有优劣

### 路线四：数据合成与标注

**核心思路**：高质量训练数据是 agent 性能的关键瓶颈，通过自动化方法大规模生成 agent trajectories。

**代表工作**：
- [[Papers/2412-AgentTrek|AgentTrek]]（XLANG Lab/HKU, 2024.12）：从互联网教程文本自动生成 web agent trajectories，三阶段 pipeline（教程采集→任务规范→VLM 执行+验证），成本仅 $0.55/trajectory，生成 10,398 条高质量轨迹
- [[Papers/2412-OSGenesis|OS-Genesis]]（ACL 2025）：**逆向任务合成**——先让 agent 自由探索环境，再从交互轨迹逆向生成任务描述，解决传统 "先定义任务→再执行" 范式中的 task-environment mismatch 问题
- [[Papers/2504-TongUI|TongUI]]（Shanghai AI Lab, 2025.04, AAAI 2026）：从互联网多模态教程（视频+文章）自动构建 GUI-Net 数据集（143K trajectories，5 OS，200+ apps），四阶段 pipeline（爬取→解析→匹配→标注），ScreenSpot 83.4%
- [[Papers/2508-OpenCUA|OpenCUA]] 的 AgentNet：人工标注 22.6K trajectories，配合 reflective CoT augmentation pipeline

**关键趋势**：
- 从人工标注 → 半自动 → 全自动，成本从 ~$10/trajectory 降至 $0.55/trajectory
- 三种互补范式：tutorial-guided replay（AgentTrek）、逆向任务合成（OS-Genesis）、多模态教程爬取（TongUI）
- Data flywheel（UI-TARS-2）将数据合成集成到训练循环中，实现自增强
- TongUI 的经验表明数据集可能比模型更有价值——143K trajectories 可复用于训练任何 GUI agent 架构

### 技术路线间的关系

三条路线**互补而非互斥**：
- Grounding 模型可作为 compositional agent 的专家模块（Agent S2 使用 UI-TARS 作为 visual grounding expert）
- RL training scaling（路线一）与 test-time scaling（路线二）正交且可组合
- Native agent 的 grounding 能力受益于路线三的数据和 benchmark（GroundCUA 的 dense annotation 可用于 native agent 训练）

## Datasets & Benchmarks

| Benchmark | 平台 | 规模 | 评估方式 | SOTA | 特点 |
|:----------|:-----|:-----|:---------|:-----|:-----|
| OSWorld | Ubuntu/Win/macOS | 369 tasks | Execution-based | 72.6% (Agent S3+BJudge) / Human 72.36% | 真实 OS 环境，标准评测 |
| OSWorld-Verified | Ubuntu | 369 tasks | Verified execution | ~76% (OSAgent) | OSWorld 增强版 |
| AndroidWorld | Android | — | Task completion | 73.3% (UI-TARS-2) | 移动端标准 benchmark |
| WebArena | Web | 812 tasks | Functional correctness | 14.41% GPT-4 / Human 78.24% | 4 个自托管网站，ICLR 2024 |
| ScreenSpot | Multi-platform | 1,200+ instructions | Grounding accuracy | 53.4% (SeeClick) | 首个跨平台 grounding benchmark |
| ScreenSpot-Pro | Desktop | 1,581 samples | Grounding accuracy | 52.9% (GroundNext-7B) | 专业高分辨率场景 |
| WindowsAgentArena | Windows | 154 tasks | Task completion | 56.6% (Agent S3) / Human 74.5% | Microsoft, 云端并行评测 |
| OfficeWorld | Desktop | 120 tasks | Execution-based | 43.3% (ComputerRL) | Office 软件操作 |

## Key Takeaways

1. **GUI Grounding 是核心瓶颈，但正在被快速攻克**：从 SeeClick 首次明确这一瓶颈，到 GroundNext-3B 在 agentic 任务上超越 72B 模型，grounding 正从"制约因素"转为"可解问题"。但专业软件中的 icon 识别（4% 准确率）仍是巨大挑战。

2. **RL 已成为 GUI agent 训练的 de facto 范式**：UI-TARS-2 的 data flywheel + PPO、ComputerRL 的 Entropulse + step-level GRPO、GroundCUA 的 RLOO、DART-GUI 的解耦异步训练——RL 在各种框架中均带来一致提升。核心难点是 multi-turn sparse reward 下的 credit assignment 和 entropy collapse。DART-GUI 揭示了一个被低估的 insight：**GUI agent RL 的瓶颈在系统效率而非算法**——5.5× 环境利用率提升可能比算法改进更关键。

3. **方法创新 > 模型规模**：ComputerRL 9B 超越 o3，GroundNext-3B 超越 72B agentic agent，SmolVLA 的故事在 GUI agent 领域同样成立。数据质量、训练策略和架构设计的杠杆效应远大于 scaling。

4. **Training-time scaling 与 Test-time scaling 正交互补**：UI-TARS-2/ComputerRL 代表训练时 RL scaling，Agent S3/BJudge 代表推理时 compute scaling。两者可组合——用 RL 训练的强模型作为 base，再用 test-time scaling 进一步提升可靠性。

5. **从 GUI-only 到 API-GUI 统一是效率突破口**：ComputerRL 的 API-GUI 范式和 UI-TARS-2 的 GUI-SDK 扩展都表明，单纯模拟人类 GUI 操作效率低下，让 agent 同时掌握程序化 API 调用是提升效率的关键（步数减少 3x）。

6. **数据合成正在解决 data bottleneck**：从 CogAgent 时代依赖人工标注，到 AgentTrek（$0.55/trajectory 自动合成）、OS-Genesis（逆向任务合成）和 UI-TARS-2 的 data flywheel，数据获取成本急剧下降。OpenCUA 的 reflective CoT augmentation 证明 CoT 质量比 trajectory 数量更重要（+32%）。

7. **轻量模型可行性已验证**：ShowUI 2B 接近 7B 模型（256K 数据），OS-Atlas 7B 超越 GPT-4o，ComputerRL 9B 超越 o3。Token 效率（ShowUI 的 UI-guided selection）和训练策略（Entropulse）是关键杠杆，不必追求大模型。

8. **感知 pipeline 质量是被低估的因素**：WindowsAgentArena 发现 SoM annotation 质量造成 15-57% 性能波动，OmniParser 证明 local semantics 提升 23.3%。比起 reasoning 能力，感知质量对最终性能的影响可能更大。

## Open Problems

1. **Safety & Reversibility**：Agent 自主操作桌面环境存在不可逆风险（删文件、发邮件、执行代码）。几乎所有论文都未充分讨论 safety 机制，这是部署的最大障碍。需要 undo 机制、action 确认、sandbox isolation 等系统级方案。

2. **跨应用 Workflow**：Agent S2 在 Workflow 类任务上仅 18.21%，说明需要跨应用状态追踪和长程 context 维护的任务仍是根本性挑战。这可能需要 explicit memory / world model 支持。

3. **专业领域 Icon 与非文字元素 Grounding**：ScreenSpot-Pro 揭示专业软件 icon 识别仅 4%，当前 VLM 对 domain-specific visual elements 的理解严重不足。可能需要专门的 icon 预训练数据或检索增强方法。

4. **多语言 GUI 理解**：ScreenSpot-Pro-CN 显示中文指令下性能显著下降，多语言 GUI grounding 是被忽视的方向。

5. **评估标准化与真实性**：OSWorld 已成为事实标准，但 369 个任务可能不足以覆盖长尾场景。多 rollout 方法假设独立初始状态，在真实环境中不成立。需要更大规模、更多样化、支持状态隔离的 benchmark。

6. **Compute 效率与部署**：当前 SOTA 方法需要数千 VM 训练（UI-TARS-2）或多次 rollout + GPT-5 评估（Agent S3），成本极高。如何在保持性能的同时降低部署成本是产业化的关键。

7. **Training-time vs Test-time Scaling 的 Pareto Frontier**：两种 scaling 路径的最优组合尚未被系统探索。何时投入更多训练 compute、何时投入更多推理 compute？

8. **从 Computer-use 到 Embodied Agent 的迁移**：Computer-use agent 的 grounding、planning、error recovery 能力是否可以迁移到物理世界的 embodied agent？UI-TARS 的 System-2 reasoning 和 ComputerRL 的 Entropulse 是否适用于 VLA 训练？**这是连接 computer-use agent 与我们核心研究方向（VLA/Embodied AI）的关键桥梁**（建议加入 DomainMaps）。

## Paper Comparison

| Paper | Venue | 技术路线 | 核心方法 | 关键结果 | 局限性 |
|:------|:------|:---------|:---------|:---------|:-------|
| [[Papers/2312-CogAgent\|CogAgent]] | CVPR 2024 | Native Model | Dual-resolution cross-attention (224+1120) | Mind2Web 58.2% (首个视觉>HTML) | 18B 推理成本高 |
| [[Papers/2401-SeeClick\|SeeClick]] | ACL 2024 | Grounding | Continual pre-training + ScreenSpot | ScreenSpot 53.4% | Action space 过简 |
| [[Papers/2404-OSWorld\|OSWorld]] | NeurIPS 2024 | Benchmark | 真实 OS + execution-based eval | Human 72.36% vs Model 12.24% | 369 tasks 规模有限 |
| [[Papers/2307-WebArena\|WebArena]] | ICLR 2024 | Benchmark | 4 个自托管网站 + functional eval | GPT-4 14.41% vs Human 78.24% | 仅 web 场景 |
| [[Papers/2408-OmniParser\|OmniParser]] | arXiv | Perception Tool | YOLOv8 + BLIP-v2 + SoM | ScreenSpot 73.0% | 依赖外部 VLM |
| [[Papers/2409-WindowsAgentArena\|WindowsAgentArena]] | arXiv | Benchmark | Windows VM + 154 tasks | Navi 19.5% vs Human 74.5% | 仅 Windows |
| [[Papers/2410-OSAtlas\|OS-Atlas]] | arXiv | Foundation Model | 13.58M grounding corpus | 7B 超 GPT-4o zero-shot | 桌面泛化弱 |
| [[Papers/2411-ShowUI\|ShowUI]] | CVPR 2025 | Lightweight Model | UI-guided token selection, 2B | 75.1% zero-shot, 256K 数据 | 高分辨率受限 |
| [[Papers/2411-GUIAgentSurvey\|GUI Agent Survey]] | arXiv | Survey | 8 RQ, 500+ papers | 全景知识地图 | 批判分析偏浅 |
| [[Papers/2412-AgentTrek\|AgentTrek]] | arXiv | Data Synthesis | Tutorial→trajectory, $0.55/traj | 10.4K trajectories, +230% | 仅 web |
| [[Papers/2412-OSGenesis\|OS-Genesis]] | ACL 2025 | Data Synthesis | 逆向任务合成 | AndroidWorld 17.41% | 任务覆盖窄 |
| [[Papers/2501-ACUSurvey\|ACU Survey]] | arXiv | Survey | 3 维 taxonomy, 87 agents | 6 大研究缺口 | 缺定量对比 |
| [[Papers/2501-UITARS\|UI-TARS]] | arXiv | Native Agent | Perception+System-2+Reflection | OSWorld 24.6, 10+ SOTA | 72B 延迟高 |
| [[Papers/2504-AgentS2\|Agent S2]] | arXiv | Compositional | Manager-Worker + MoG | OSWorld 34.5% | Workflow 仅 18% |
| [[Papers/2504-ScreenSpotPro\|ScreenSpot-Pro]] | arXiv | Benchmark | 专业高分辨率 grounding | Icon 仅 4% 准确率 | 规模偏小 |
| [[Papers/2508-ComputerRL\|ComputerRL]] | arXiv | RL Training | API-GUI + Entropulse | OSWorld 48.9% (9B>o3) | Rule-based reward |
| [[Papers/2508-OpenCUA\|OpenCUA]] | NeurIPS 2025 | Open Framework | AgentNet + Reflective CoT | OSWorld-V 45.0% 开源 SOTA | 与闭源差距 16% |
| [[Papers/2508-OSAgentsSurvey\|OS Agents Survey]] | ACL 2025 | Survey | 3 层框架, 33 benchmarks | 最全 OS agent 综述 | 时效受限 |
| [[Papers/2509-UITARS2\|UI-TARS-2]] | arXiv | RL Training | Data flywheel + PPO 变体 | OSWorld 47.5% | VLM verifier 噪声 |
| [[Papers/2510-ScalingAgents\|Agent S3]] | ICML | Test-time Scaling | BJudge + wide scaling | OSWorld 72.6% 超人类 | 多 rollout 假设 |
| [[Papers/2504-TongUI\|TongUI]] | AAAI 2026 | Data Synthesis | 多模态教程→trajectory, 143K | ScreenSpot 83.4% | Teacher 蒸馏上限 |
| [[Papers/2509-DARTGUI\|DART-GUI]] | arXiv | RL Infrastructure | 解耦异步 RL, 5.5× 环境利用率 | OSWorld 42.13% (7B>Claude) | 仅 OSWorld 评测 |
| [[Papers/2511-GroundCUA\|GroundCUA]] | arXiv | Grounding Data | Dense annotation, 56K screenshots | 3B > 72B agentic | 标注成本高 |

## 调研日志
- **Round 1**: 2026-04-02, 搜索 9 次，digest 12 篇（成功 12）
- **Round 2**: 2026-04-02, 搜索 4 次，digest 9 篇（成功 9），补充先驱模型、感知工具、数据合成和 benchmark
- **Round 3**: 2026-04-02, digest 2 篇：TongUI（数据合成）、DART-GUI（RL 训练基础设施）
- **论文总计**: 23 篇
- **未能获取**: 无
