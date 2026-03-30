---
name: literature-survey
description: >
  当 Supervisor 说"调研""survey""了解研究现状"，或需要系统了解某主题的文献全貌时，搜索外部文献、批量 digest、综合生成调研报告
argument-hint: "<topic> [scope]"
allowed-tools: Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---

## Purpose

literature-survey 是 MindFlow 的主题级文献调研技能。给定一个研究主题，它主动搜索外部文献、自动调用 paper-digest 生成结构化笔记，最终综合所有论文产出一份领域调研报告。

它与其他文献技能的分工：
- `paper-digest`：处理**单篇**论文 → Paper 笔记
- `cross-paper-analysis`：对比 vault **已有**论文 → 分析报告
- `literature-survey`：给定**主题**，向外搜索 → 批量 digest → 综合调研报告

该技能是 Level 1 Orchestration skill，会调用 paper-digest 作为子步骤。

## Steps

### Step 1：明确调研范围

解析用户输入，确定以下参数：

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `topic` | （必填） | 研究主题 |
| `year_range` | 近 3 年 | 时间范围 |
| `venue_preference` | 无 | venue 偏好 |
| `max_papers` | 8 | 最终纳入调研的论文数量上限 |

基于 topic 生成 **3-5 条搜索策略**，覆盖以下角度：

1. **核心主题**：topic 本身的直接搜索
2. **相关方法**：该领域的主流技术路线
3. **Survey / 综述**：搜索已有 survey 论文作为参考锚点
4. **Benchmark / 数据集**：该领域常用的评测基准
5. **应用场景**：下游应用或跨领域迁移

每条策略生成 1-2 个具体的搜索 query（英文），共 5-10 个 query。

### Step 2：Vault-first 检索

在执行外部搜索之前，先检查 vault 中已有的相关内容：

1. 用 Grep 在 `Papers/` 的 frontmatter 和正文中搜索 topic 相关关键词（2-3 个核心关键词）。
2. 用 Grep 在 `Topics/` 中搜索是否已有相关调研或分析。
3. 收集所有匹配的 Paper 笔记，建立"**已知论文清单**"（title 列表），后续用于去重。
4. 若发现高度相关的已有 Survey（如 `Topics/{Topic}-Survey.md` 已存在），停止执行并告知 Human，建议更新而非重建。

### Step 3：外部搜索与筛选

依次执行 Step 1 生成的搜索 query：

1. 对每个 query，用 **WebSearch** 搜索（建议加 `site:arxiv.org` 或 `"论文标题" arxiv`），提取搜索结果中的论文信息。
2. 从搜索结果中收集候选论文列表，提取：title、authors、year、venue（若可判断）、url。
3. **去重**：将每个候选论文的 title（转小写，去标点）与已知论文清单对比，跳过已在 vault 中的论文。
4. **搜索轮数上限**：最多执行 **10 次 WebSearch**。若某些 query 返回结果质量低（无相关论文），提前停止该策略。

搜索完成后，对所有候选论文进行筛选和排序：

- **相关性**：与 topic 的直接相关程度
- **影响力**：优先选择知名 venue（NeurIPS, ICML, ICLR, CoRL, RSS, CVPR 等）的论文
- **时效性**：在 year_range 内的论文优先
- **多样性**：确保覆盖不同技术路线，避免全部来自同一方向

选取 **top-N**（N = `max_papers` 减去 vault 中已有的相关论文数，最少 3 篇）作为待 digest 的论文列表。

### Step 4：批量 paper-digest

对 Step 3 筛选出的每篇论文，执行 paper-digest：

1. 读取 `skills/1-literature/paper-digest/SKILL.md`，按其 Steps 逐一处理每篇论文。
2. 输入为论文的 arXiv URL（优先）或论文标题。
3. **跳过规则**：
   - 若 paper-digest 的去重检查发现 vault 已有该笔记，跳过。
   - 若 WebFetch 无法获取论文内容（如非 arXiv 论文、付费墙），记录为"未能获取"并跳过，不阻塞流程。
4. 记录每篇论文的 digest 结果：成功（文件路径）/ 跳过（原因）/ 失败（原因）。

### Step 5：综合分析

基于 vault 中所有相关论文笔记（Step 2 已有的 + Step 4 新 digest 的），进行综合分析：

1. 用 Read 读取所有相关 Paper 笔记（重点：Summary、Method、Key Results、Strengths & Weaknesses）。
2. 读取 `DomainMaps/_index.md`（索引页）找到相关 domain，再读取对应的 `DomainMaps/{Name}.md` 了解当前认知状态。

按以下维度综合：

#### （1）领域概览
- 该主题的核心问题是什么？为什么重要？
- 近年研究的整体规模和活跃度如何？

#### （2）技术路线分类
- 将论文按方法论聚类（如 2-4 个主要技术路线）
- 每个路线的代表论文、核心思路、优劣势

#### （3）发展时间线
- 关键里程碑和突破（按时间排列）
- 技术路线的演进趋势

#### （4）对比表
构建对比表，格式：

| Paper | Year | 技术路线 | 核心方法 | 关键结果 | 局限性 |
|:------|:-----|:---------|:---------|:---------|:-------|

Paper 列使用 `[[wikilink]]` 格式。

#### （5）开放问题与研究机会
- 多篇论文共同指出的未解决问题
- 技术路线之间的矛盾或争议
- 潜在的研究空白

### Step 6：产出

#### 6a. 生成 Survey 文件

用 Write 创建 `Topics/{Topic}-Survey.md`，Topic 名称根据主题生成（CamelCase，如 `VLA-Manipulation`、`DiffusionPolicy-Robotics`）。

文件结构：

```markdown
---
title: "{topic} 文献调研"
tags: []
status: draft
date_updated: YYYY-MM-DD
scope:
  year_range: "YYYY-YYYY"
  max_papers: N
  papers_found: N
  papers_digested: N
---

## Overview

<领域概览：核心问题、研究现状、整体趋势，3-5 段>

## 技术路线

### 路线 1：<名称>

<代表论文、核心思路、优劣势>

### 路线 2：<名称>

<同上>

...

## 发展时间线

<关键里程碑，按时间排列>

## Paper Comparison

<对比表，Paper 列用 [[wikilink]]>

## Key Takeaways

<3-5 条核心结论>

## Open Problems

<开放问题与研究机会>

## 调研日志

- **调研日期**: YYYY-MM-DD
- **搜索策略**: <列出使用的 query>
- **论文统计**: vault 已有 N 篇 + 新 digest N 篇 + 跳过 N 篇
- **未能获取**: <列出未能 digest 的论文及原因>
```

所有论文引用使用 `[[wikilink]]` 格式。

#### 6b. 追加日志

用 Edit（若文件不存在则用 Write）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`：

```markdown
### [HH:MM] literature-survey
- **input**: topic: <topic> | scope: <scope 参数>
- **output**: [[Topics/{Topic}-Survey]]
- **stats**: 搜索 N 次，候选 N 篇，digest N 篇（成功 N / 跳过 N / 失败 N）
- **observation**: <一句话概括该领域的核心发现>
- **status**: success
```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Guard

- **不覆盖已有 Survey**：若 `Topics/{Topic}-Survey.md` 已存在，停止执行并告知 Human；建议在文件名加日期后缀（如 `-2026-03`）或手动更新。
- **paper-digest 失败不阻塞**：单篇论文 digest 失败时记录原因并继续处理下一篇，不中断整个 survey 流程。
- **搜索上限**：最多执行 10 次 WebSearch，避免过度消耗 token 和 API 配额。
- **不捏造论文**：所有纳入分析的论文必须来自实际搜索结果或 vault 已有笔记，不得凭记忆编造论文信息。
- **不直接修改 DomainMaps**：综合分析中如有值得纳入 DomainMaps 的发现，在 Survey 文件的 Key Takeaways 中标注"建议加入 DomainMaps"，不得直接修改 `DomainMaps/` 下的任何文件。
- **Papers/ 已有笔记只读**：不得修改 vault 中已存在的 Paper 笔记，只可读取。新论文的笔记由 paper-digest 创建。
- **语言规范**：正文用中文撰写，英文技术术语保持英文不翻译。
- **autonomy: copilot 模式**：若以 copilot 模式调用，在 Step 3 筛选完成后先输出候选论文列表给 Human 确认，确认后再执行 Step 4 批量 digest；最终 Survey 文件也先预览再写入。

## Verify

- [ ] `Topics/*-Survey.md` 已创建
- [ ] 技术路线分类 ≥2 条
- [ ] Paper Comparison 对比表论文数 ≥3
- [ ] Open Problems 节非空

## Examples

**示例 1：调研一个主题**

```
"调研一下 VLA for manipulation 的研究现状"
```

执行过程：

1. 解析：topic = "VLA for manipulation"，year_range = 2023-2026，max_papers = 8
2. 生成搜索策略：
   - "Vision-Language-Action models manipulation arxiv"
   - "VLA robot manipulation policy learning"
   - "VLA survey embodied AI"
   - "manipulation benchmark evaluation VLA"
3. Grep `Papers/` 搜索已有 VLA 相关笔记，发现 3 篇
4. WebSearch 执行 6 次搜索，收集 20 篇候选，去重后剩 14 篇
5. 筛选 top-5（8 - 3 已有 = 5）
6. 逐一 paper-digest：成功 4 篇，失败 1 篇（付费墙）
7. 综合分析 7 篇论文（3 已有 + 4 新增），生成调研报告
8. Write `Topics/VLA-Manipulation-Survey.md`
9. 追加日志到 `Workbench/logs/2026-03-27.md`

输出文件：`Topics/VLA-Manipulation-Survey.md`

---

**示例 2：带约束的调研**

```
"survey diffusion policy in robotics，只看 2024 年以后的 top venue 论文，最多 5 篇"
```

执行过程：

1. 解析：topic = "diffusion policy in robotics"，year_range = 2024-2026，venue_preference = top-tier，max_papers = 5
2. Grep `Papers/` 发现 1 篇已有
3. WebSearch 搜索，优先筛选 CoRL / RSS / ICRA / NeurIPS / ICML 论文
4. 筛选 top-4（5 - 1 = 4）
5. 逐一 paper-digest
6. 综合分析，生成报告
7. Write `Topics/DiffusionPolicy-Robotics-Survey.md`

输出文件：`Topics/DiffusionPolicy-Robotics-Survey.md`
