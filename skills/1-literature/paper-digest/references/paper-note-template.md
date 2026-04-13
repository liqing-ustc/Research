---
title:
authors:
  - author1
  - author2
  - ...
institute:
  - institute1
  - institute2
  - ...
date_publish:
venue:
tags:
  - tag1
  - tag2
  - ...
url:
code:
rating: "%% 1=不相关, 2=了解即可, 3=有参考价值, 4=重要, 5=必读 %%"
date_added: "{{today}}"
---
## Summary
%% 用一句话概括这篇论文的核心贡献，不超过50字 %%

**Key Takeaways:** 
1. **{Takeaway 1}**: 简要说明
2. **{Takeaway 2}**: 简要说明
3. ...

**Figure #.  {caption}**
%% 此处嵌入 teaser 图，通常是 Figure 1 的 overview/concept 示意图，位于 Abstract 或 Introduction，外链优先。若论文无独立 teaser（如 Figure 1 就是架构图），删除本行，不要重复嵌入。 %%

---
## Problem & Motivation
%% 问题背景与动机，2-5 句话。为什么重要？现有方法有什么局限？ %%

---
## Method
%% 核心方法/架构。可分段，列出关键组件，内嵌架构图和核心公式。 %%

**Figure #.  {caption}**
%% 此处嵌入方法架构图，通常是 Figure 1 或 2，位于 Method 或 Introduction，用于展示核心方法/架构，外链优先。 %%

**Equation #. {公式名}**

$$
{公式内容}
$$
**符号说明**：
**含义**：

... (Method 部分其他重要图表和公式)

---
## Experiments
%% 主要实验结果，包含具体数字和 benchmark 名称。Blog 文章可删除本段或改为 Key Points。 %%

### Datasets

| Dataset    | Size   | 特点   | 用途    |
| ---------- | ------ | ---- | ----- |
| {Dataset1} | {size} | {特点} | 训练/测试 |
| {Dataset2} | {size} | {特点} | 测试    |
| ...        |        |      |       |
### Implementation Details

- **Base model**: {使用的骨干网络}
- **Optimizer**: {Adam/SGD, 学习率}
- **Batch Size**: {大小}
- **Training epochs**: {epochs}
- **硬件**: {GPU 型号和数量}
- ...

### Results

**Table #.  {caption}**
%% 主 benchmark 对比表，复制为 Markdown 表格，包含主要 baseline + 论文方法的结果 %%

**Insights**: {关键发现和解释}

**Table #.  {caption}**
%% Ablation study 对比表，复制为 Markdown 表格 %%

**Insights**: {关键发现和解释}

**Figure #. {caption}**
%% 可视化结果，loss curves、examples 等等 %%

**Insights**: {关键发现和解释}

... (Experiments 部分其他重要图表)

---
## 论文点评
%% 方法亮点与局限的点评，以及对可复现性的评估。 %%

### Strengths

1. {优点1}
2. {优点2}
3. ...

### Weaknesses

1. {缺点1}
2. {缺点2}
3. ...

### 可复现性评估
%% 根据论文和 GitHub repo 实际情况勾选 %%
- [ ] 代码开源
- [ ] 模型权重开源
- [ ] 训练细节完整
- [ ] 数据集可获取

---
## 关联笔记
%% 列出相关笔记的 wikilink，无对应内容的子类直接删除 %%

### 基于
- [[{前置工作1}]]: {说明}
- [[{前置工作2}]]: {说明}

### 对比
- [[{对比方法1}]]: {为什么对比}
- [[{对比方法2}]]: {为什么对比}

### 方法相关
- [[{核心技术1}]]: 核心方法
- [[{核心技术2}]]: 重要组件

### 硬件/数据相关
- [[{硬件或数据集}]]: {说明}

---

## 速查卡片

> [!summary] {Paper Title}
> - **核心**: {一句话核心}
> - **方法**: {关键方法}
> - **结果**: {主要结果}
> - **代码**: {GitHub链接}

---
## Notes
%% 其他想法、疑问、启发。留空供后续填写。 %%
