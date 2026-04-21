---
title: "Gemini Robotics ER 1.6: Enhanced Embodied Reasoning"
authors: [Laura Graesser, Peng Xu]
institutes: [Google DeepMind]
date_publish: 2026-04-14
venue: Google DeepMind Blog
tags: [embodied-reasoning, spatial-reasoning, scene-understanding]
paper:
website: https://deepmind.google/blog/gemini-robotics-er-1-6/
github: https://github.com/google-deepmind/gemini-robotics-sdk
rating: 2
date_added: "2026-04-15"
---

## Summary

> [!summary] Gemini Robotics ER 1.6
> - **核心**: 面向机器人的 embodied reasoning 模型，推理与控制解耦
> - **方法**: Pointing-based spatial reasoning + multi-view success detection + agentic vision (视觉推理 + code execution)
> - **结果**: Instrument reading 93% (vs ER 1.5 23%)；ASIMOV safety +6%/+10% over Flash
> - **Sources**: [website](https://deepmind.google/blog/gemini-robotics-er-1-6/) | [github](https://github.com/google-deepmind/gemini-robotics-sdk)
> - **Rating**: 2 - Frontier（产品 blog，工程意义大但信息密度有限且缺乏跨家族横向对比）

**Key Takeaways:**
1. **Reasoning-first 架构定位**：ER 1.6 不直接输出动作，而是作为机器人系统的高层推理组件，通过 native tool-calling 调用 VLA 模型、Google Search、第三方函数，验证了 embodied reasoning 与底层控制解耦的路线
2. **Agentic vision 是核心差异化**：模型不是 one-shot 推理，而是迭代式地 zoom in → pointing → code execution → 解释，将 instrument reading 从 86%（无 agentic vision）提升到 93%
3. **工业部署信号**：与 Boston Dynamics 合作将 Spot 与 ER 1.6 结合用于设施巡检，是 embodied reasoning 进入真实工业场景的标志性案例

**Teaser. Benchmark results: ER 1.6 consistently outperforms ER 1.5 and Gemini 3.0 Flash across all four evaluation tasks.**
![](https://lh3.googleusercontent.com/_dicgE2AAgiQBrY1zvNrdLqTsE5oNi3vbp95Zo4-vp809tdsRitsV4uOQHLBJES4QFjdqrJEW0gFUvwnYVDrbqcE6yd_wuigVj2Xxi-9Q-KA1UjodQ=w2880-h1620-n-nu-rw-lo)

---

## Pointing: The foundation of spatial reasoning

Pointing 是 ER 模型的基础空间推理能力，用途覆盖四个维度：

- **Spatial reasoning**：精确物体检测与计数
- **Relational logic**：比较推理（如识别集合中最小的物品）、定义 from-to 关系（如移动 X 到位置 Y）
- **Motion reasoning**：轨迹映射与最优抓取点识别
- **Constraint compliance**：复杂约束推理（如"指出所有能放进蓝色杯子的物体"）

ER 1.6 的关键改进在于将 points 作为**中间推理步骤**——例如先用 pointing 计数，再用 pointing 定位显著点辅助数学运算，从而提升度量估计精度。

**Figure 2. Pointing 对比：ER 1.6 正确识别工具数量并避免幻觉不存在的物体。**
![](https://lh3.googleusercontent.com/wX1QYLrafPEhOPLVaFTsvztVDlTW4g7YglaDK1Ex4fO-4spBmnEYOcHFzLyDvzFQsfEbCwRqlSWCtBcCu4ou5xvIipQ-a3nnxkGzo55dhhOFJHJ0Ug=w2880-h1620-n-nu-rw-lo)

对比细节：ER 1.6 正确识别锤子 (2)、剪刀 (1)、刷子 (1)、钳子 (6)，且对图中不存在的物品（独轮车、Ryobi 电钻）不输出 point。ER 1.5 则错判锤子和刷子数量、遗漏剪刀、幻觉出独轮车。Gemini 3.0 Flash 接近 ER 1.6 但在钳子上不够精确。

## Success Detection: The engine of autonomy

Success detection 被定义为"自主性的引擎"——它使 agent 能在 retry 和 proceed to next step 之间智能决策。

挑战在于：
- 需要 sophisticated perception + reasoning + broad world knowledge
- 实际机器人部署涉及**多视角相机流**（overhead + wrist-mounted）
- 需处理遮挡、低光照、模糊指令等复杂因素

ER 1.6 的核心进展是 **multi-view reasoning**——理解不同视角如何在每个时刻和跨时间序列中组合成连贯画面。

**Video 1. Multi-view success detection：ER 1.6 综合 overhead 和 wrist camera 判断"put the blue pen into the black pen holder"任务完成。**

<video src="https://storage.googleapis.com/gdm-deepmind-com-prod-public/media/n0RLBRCstlc6TDDk/gemini-robotics_1.6__success-detection_multiview-example.webm" controls muted playsinline width="720"></video>

## Instrument reading: Real-world visual reasoning

Instrument reading 是 ER 1.6 新解锁的能力，源于与 Boston Dynamics 在设施巡检上的合作需求。工业设施中大量仪表（温度计、压力表、化学液位计等）需要持续监控，Boston Dynamics 的 Spot 机器人负责巡检并拍摄仪表图像。

**Video 2. Boston Dynamics Spot 使用 Gemini Robotics-ER 进行设施仪表巡检。**
![](https://www.youtube.com/watch?v=kBwxmlI2yHQ)

仪表读数的难点：
- 需精确感知多种输入（指针、液位、容器边界、刻度线）并理解它们的相互关系
- Sight glass 需估算液体填充比例，同时考虑相机视角的透视畸变
- Gauge 上的文字需读取并解释单位，多指针需理解不同精度并组合

ER 1.6 通过 **agentic vision** 实现高精度读数：视觉推理 + 代码执行的组合。模型执行中间步骤：先 zoom in 获取细节 → 用 pointing 和代码估算比例和间隔 → 应用世界知识解释含义。

**Figure 3. Instrument reading 各组件对性能的贡献。**
![](https://lh3.googleusercontent.com/RvYAY_w1ZJfrVeEtxg3oh6YjyQuvSgFcIammormuzrUixbvwlNjFLLFRpUULIG153bgevZaZtnEjNZNaM_U2YKXHTRbZBDYvjxadsqIMAeTcuz6X=w2880-h1620-n-nu-rw-lo)

各模型 instrument reading 成功率：ER 1.5 23% → Gemini 3.0 Flash 67% → ER 1.6 86% → ER 1.6 w/ agentic vision **93%**。Agentic vision 带来了额外 7% 的提升，但更关键的是 ER 1.6 base 本身相对 ER 1.5 的 63% 绝对提升。

**Video 3. Agentic vision demo：模型使用 pointing 和代码执行 zoom in 并导出 sub-tick 精度的 gauge 读数。**

<video src="https://storage.googleapis.com/gdm-deepmind-com-prod-public/media/n0RLBRCstlc6TDDk/gemini-robotics_1.6__instrument-reading-demo.webm" controls muted playsinline width="720"></video>

## Our safest robotics model yet

安全性集成在模型的每一层。ER 1.6 在以下维度展示了改进：

- **Adversarial spatial reasoning**：在对抗性空间推理任务上，ER 1.6 对 Gemini safety policies 的合规性优于所有前代模型
- **Physical safety constraints**：通过 pointing 等空间输出，模型在 gripper 或材料约束下做出更安全的决策（如"不处理液体"、"不拾取超过 20kg 的物体"）
- **ASIMOV benchmark**：在基于真实伤害报告的文本和视频场景中识别安全风险，ER 模型相比 Gemini 3.0 Flash 在文本上 +6%，视频上 +10%

**Figure 4. ASIMOV Safety Instruction Following：ER 1.6 在物理安全约束遵循上大幅超越 ER 1.5，在 pointing 准确率上超越 Gemini 3.0 Flash。**
![](https://lh3.googleusercontent.com/JzklnvIzHI-kFlxFia447n9ZeMHmAlqrg4sA4CL4PURVcnvMbx-DWMSWLgOR3bQ9MdeNTLNhlOc-soMWNWpZnkwx6aOYu-jBp6QyW_VLZZa18Xh6LQ=w2880-h1620-n-nu-rw-lo)

---

## 关联工作

### 基于
- [[2503-GeminiRobotics|Gemini Robotics]] (2503.20020): ER 1.6 的前代模型，首次提出 Gemini Robotics 和 Gemini Robotics-ER 架构
- Gemini Robotics-ER 1.5: 直接前代版本
- Gemini 3.0 Flash: base VLM，ER 1.6 在其基础上增强 embodied reasoning 能力

### 对比
- Gemini Robotics-ER 1.5: pointing/counting、success detection、instrument reading 全面对比
- Gemini 3.0 Flash: 作为 baseline VLM 对比

### 方法相关
- Agentic vision: 视觉推理 + 代码执行的组合范式，ER 1.6 的核心差异化技术
- ASIMOV benchmark v2: 基于真实伤害报告的机器人安全评估基准

---

## 论文点评

### Strengths

1. **Embodied reasoning 作为独立层的验证**：ER 1.6 清晰定位为 high-level reasoning module，通过 tool-calling 调用 VLA / Search / 第三方函数，验证了推理与控制解耦的架构路线。这比端到端 VLA 更模块化、更易调试和升级
2. **Agentic vision 范式**：不是 one-shot visual QA，而是让模型像人一样"看了再看、算了再算"。93% vs 23% 的仪表读数提升证明 agentic loop 在视觉推理上价值巨大，这一范式可能对其他精细视觉任务也有启发
3. **工业落地信号**：与 Boston Dynamics Spot 的合作不是 demo 级别——仪表读数是真实的工业需求，且博文提供了 API 和 Colab，说明已进入开发者可用阶段
4. **安全性不是附加项**：ASIMOV benchmark 上的系统性评估以及 physical constraint compliance 的改进，说明安全性是 training 目标之一而非 post-hoc 贴片

### Weaknesses

1. **缺乏横向对比**：所有 benchmark 只对比自家模型（ER 1.5、Gemini 3.0 Flash），没有与 GPT-4o、Claude 等在 spatial reasoning 上的对比，无法判断绝对竞争力
2. **Benchmark 不透明**：pointing/counting、success detection、instrument reading 的评估集大小和构成未披露，无法评估统计显著性
3. **Multi-view success detection 泛化存疑**：demo 场景（桌面物体操作）相对受控，对工业环境中遮挡严重、光照变化大的场景是否鲁棒未知
4. **Blog 信息密度有限**：作为产品发布 blog 而非技术报告，缺少训练细节、数据构成、模型架构变化等关键信息，难以深入分析 ER 1.6 相对 ER 1.5 的具体改进来源
5. **Agentic vision 的延迟成本未讨论**：zoom + pointing + code execution 的多步推理必然增加延迟，对实时机器人控制的影响是什么？

### 可信评估

#### Artifact 可获取性
- **代码**: inference-only（Safari SDK 需 Trusted Tester Program；公开可用的是 Gemini API + Colab notebook 示例）
- **模型权重**: gemini-robotics-er-1.6-preview（通过 Gemini API 调用，不提供权重下载）
- **训练细节**: 未披露（仅提及基于 Gemini 系列的 targeted training）
- **数据集**: 未披露

#### Claim 可验证性
- ✅ **Instrument reading 93% 成功率**：博文明确给出了四个模型的对比数字和 agentic vision 的增量贡献，bar chart 可读
- ⚠️ **"Unprecedented precision" / "new level of autonomy"**：marketing 修辞，缺乏与非 Google 模型的对比 grounding
- ⚠️ **Pointing/counting 和 success detection 的优势**：有 bar chart 但评估集规模和构成未披露，统计显著性不明
- ⚠️ **"Safest robotics model yet"**：ASIMOV benchmark 数字有据（+6%/+10%），但仅与 Gemini 3.0 Flash 比较，"yet" 的范围仅限自家模型
- ❌ **"Unprecedented precision"**：没有给出在任何公开 benchmark 上的排名，无法验证 "unprecedented"

### Notes

### Rating

**分数**：2 - Frontier
**理由**：ER 1.6 是 embodied reasoning 方向当前的重要前沿参考——agentic vision + tool-calling 的范式（Strengths 1-2）和 Boston Dynamics 工业部署信号（Strength 3）使它成为讨论 "reasoning 与 control 解耦" 路线时绕不开的 datapoint；但作为产品 blog，训练细节和数据未披露、缺乏跨家族横向对比（Weaknesses 1-2、4），信息密度不足以成为 Foundation 级别的奠基工作，也尚未形成 de facto standard。
