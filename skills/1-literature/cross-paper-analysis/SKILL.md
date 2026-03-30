---
name: cross-paper-analysis
description: >
  当需要对比多篇论文的方法/结论/实验设置，或 Supervisor 说"对比""分析这几篇"时，执行跨论文对比并识别共识、矛盾和知识空白
argument-hint: "<papers> [focus]"
allowed-tools: Read, Write, Edit, Glob, Grep
---

## Purpose

cross-paper-analysis 是 MindFlow 的核心洞察发现机制，承担"去伪存真"的职能。它读取由 paper-digest 产出的 `Papers/` 笔记，对比多篇论文的问题定义、方法设计、实验设置与核心结论，从中识别三类高价值信息：**共识**（high-confidence knowledge）、**矛盾**（open questions and active debates）、**知识空白**（potential research opportunities）。

该技能的分析结果写入 `Topics/{topic}-Analysis.md`，并将发现的 pattern 追加到 `Workbench/memory/patterns.md`，为 memory-distill 提供素材。它是 literature-sweep 等高阶工作流的核心子技能，也可单独调用用于专题文献综述。

## Steps

### Step 1：收集输入论文

根据 `papers` 参数的类型选择收集方式：

**若 `papers` 为 wikilinks 列表**（如 `[[2603-EvoScientist]], [[2501-DiffusionPolicy]]`）：

1. 用 Glob 扫描 `Papers/` 目录，匹配对应文件名。
2. 用 Read 逐一读取每篇笔记的完整内容，重点关注以下节：
   - `## Summary`（一句话概括）
   - `## Problem & Motivation`（问题定义）
   - `## Method`（核心方法）
   - `## Key Results`（关键结果）
   - `## Strengths & Weaknesses`（优劣势）

**若 `papers` 为 tag 筛选条件**（如 `tags: VLA`）：

1. 用 Grep 在 `Papers/` 目录的 frontmatter 中搜索匹配的 tag（pattern：`tags:.*VLA`，文件类型 `*.md`）。
2. 收集所有匹配文件路径，再用 Read 逐一读取（同上）。
3. 若匹配超过 15 篇，优先选取 `rating` 较高或 `status: finished` 的笔记，并告知 Human 已做筛选。

**读取背景认知**：

- 读取 `DomainMaps/_index.md`（索引页）找到相关 domain，再读取对应的 `DomainMaps/{Name}.md`，了解当前 Established Knowledge 和 Active Debates，避免重复标注已知共识。
- 读取 `Workbench/memory/patterns.md`，了解已记录的 pattern，避免重复追加。

### Step 2：构建对比表

以 Markdown 表格形式组织读取到的信息，列定义如下：

| Paper | 问题定义 | 核心方法 | 实验设置 | 关键结果 | 局限性 |
|:------|:---------|:---------|:---------|:---------|:-------|

填写规则：

- **Paper** 列：使用 `[[wikilink]]` 格式，方便 Obsidian 跳转，如 `[[2603-EvoScientist]]`。
- 每个单元格提炼 1-3 句话，保留关键数字和术语，不做过度概括。
- 对比后在关键单元格末尾添加标注：
  - `[共识]`：与其他论文结论一致
  - `[矛盾]`：与其他论文存在明显冲突
  - `[独特]`：该论文独有的设计或发现，其他论文未涉及

若 `focus` 参数被指定（如 `method comparison`），在该对比维度上加深分析，其余维度可适当精简。

### Step 3：分析发现

在对比表基础上，逐一归纳三类发现：

#### （1）共识 → 高置信度知识

收集所有标注 `[共识]` 的条目，整理为列表：

- 每条共识注明支持它的论文数量和 wikilinks（`≥2` 篇方可列为共识）。
- 评估是否已在对应 `DomainMaps/{Name}.md` 的 Established Knowledge 中记录；若未记录，标注"建议加入 DomainMaps"。

#### （2）矛盾 → 待解决的开放问题

收集所有标注 `[矛盾]` 的条目，逐条分析：

- 明确矛盾的具体内容（两篇论文各自的说法）。
- 尝试分析矛盾来源：实验设置差异？评测 benchmark 不同？问题定义范围不同？还是真正的理论分歧？
- 所有矛盾标记为"待验证"，不得在此时下确定性结论。
- 检查对应 `DomainMaps/{Name}.md` 的 Active Debates 是否已记录；若未记录，标注"建议加入 DomainMaps"。

将矛盾列表追加到 `Workbench/queue.md` 的 Questions 部分：

```markdown
### [YYYY-MM-DD] <矛盾简要描述>

- **papers**: [[PaperA]], [[PaperB]]
- **conflict**: PaperA 认为…；PaperB 认为…
- **possible_reason**: <分析可能原因>
- **status**: 待验证
```

#### （3）知识空白 → 潜在研究机会

识别以下类型的空白：

- 多篇论文均未解决但明确指出的局限性（Weaknesses 节中重复出现的条目）。
- 对比维度中出现的"无论文覆盖"区域（如某实验设置组合从未被研究）。
- 矛盾背后暗示需要更多实验才能厘清的问题。

对每个空白评估其研究价值，说明"为什么这是空白"和"填补它的潜在价值"。如有明确可行的研究方向，标注"建议生成 Idea"。

### Step 4：产出

#### 4a. 生成 Topics 分析文件

用 Write 创建 `Topics/{topic}-Analysis.md`，topic 名称根据分析主题生成（CamelCase，如 `VLA-MethodComparison`）。

文件结构如下：

```markdown
---
date: YYYY-MM-DD
papers: [[[PaperA]], [[PaperB]], ...]
focus: <focus 参数，若无则填 "general">
tags: []
---

# {Topic} Analysis

> 分析日期：YYYY-MM-DD | 论文数：N | 焦点：{focus}

## 论文概览

<对比表，来自 Step 2>

## 共识

<共识列表，每条附 wikilinks 和"建议加入 DomainMaps"标注（如适用）>

## 矛盾与争议

<矛盾列表，每条标注"待验证"，附可能原因分析>

## 知识空白

<空白列表，每条附研究价值评估和"建议生成 Idea"标注（如适用）>

## 建议

- **DomainMaps 更新建议**：<列出建议加入 Established Knowledge / Active Debates 的具体条目>
- **下一步阅读**：<基于空白和矛盾，建议进一步阅读的方向或论文类型>
- **潜在 Idea**：<若识别到明确研究机会，简述 idea 方向>
```

所有论文引用必须使用 `[[wikilink]]` 格式，确保可追溯到具体笔记。

#### 4b. 追加 patterns 到 memory

若在分析中发现跨论文的规律性现象（如"所有 VLA 论文都在 simulation-to-real 上遇到困难"），且该 pattern 尚未在 `Workbench/memory/patterns.md` 中记录，则用 Edit 追加（若文件不存在，先用 Write 创建，包含一级标题 `# Patterns`）：

```markdown
### [YYYY-MM-DD] <Pattern 简要描述>

- **observation**: <跨论文观察到的规律>
- **occurrences**: [[PaperA]], [[PaperB]], ...
- **confidence**: low / medium
- **needs_verification**: yes / no
```

仅追加新 pattern，不修改已有条目。

#### 4c. 追加日志

用 Edit（若文件不存在则用 Write）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

```markdown
### [HH:MM] cross-paper-analysis
- **input**: <papers 参数内容> | focus: <focus 参数>
- **output**: [[Topics/{topic}-Analysis]]
- **findings**: 共识 N 条，矛盾 N 条，知识空白 N 条
- **patterns**: <追加了 N 条 pattern / 无新 pattern>
- **status**: success
```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Guard

- **Papers/ 只读**：不得修改任何 `Papers/` 目录下的笔记文件，所有 Paper 笔记均为只读输入。
- **不直接修改 DomainMaps**：只在分析报告中提出"建议加入 DomainMaps"，不得直接写入或编辑 `DomainMaps/` 下的任何文件；DomainMaps 的更新由 Human 或 memory-distill 技能执行。
- **矛盾标记"待验证"**：对于识别出的矛盾，不得给出确定性结论，必须标注"待验证"，并尝试分析可能的原因而非直接判定哪篇论文正确。
- **所有引用可追溯**：分析报告中的每条结论（共识、矛盾、空白）必须附有 `[[wikilink]]`，指向具体的 Paper 笔记，不得出现无来源的泛化陈述。
- **不捏造内容**：所有分析必须来自实际读取到的笔记内容。若某笔记缺少某一节（如无 Strengths & Weaknesses），在对比表对应单元格注明"未记录"，不得推测或补全。
- **Topics 文件不覆盖**：若 `Topics/{topic}-Analysis.md` 已存在，停止执行并告知 Human；不得覆盖已有分析文件（可建议在文件名加版本后缀，如 `-v2`）。
- **language 规范**：描述与分析用中文撰写；方法名、模型名、benchmark 名、论文标题等技术术语保持英文，不做翻译。
- **autonomy: copilot 模式**：若以 copilot 模式调用，生成分析草稿后先输出给 Human 预览，确认后再执行 Write/Edit；`Workbench/` 的写入同样在确认后执行。

## Verify

- [ ] `Topics/*-Analysis.md` 已创建
- [ ] 对比表包含所有输入论文
- [ ] 共识、矛盾、知识空白三节均非空
- [ ] 所有论文引用使用 `[[wikilink]]` 格式

## Examples

**示例 1：按 tag 筛选，指定对比维度**

```
/cross-paper-analysis --papers "tags: VLA" --focus "method comparison"
```

执行过程：

1. Grep `Papers/` frontmatter，搜索 `tags:.*VLA`，收集匹配笔记列表
2. Read 每篇笔记的 Summary / Method / Key Results / Strengths & Weaknesses
3. Read `DomainMaps/_index.md`（索引）→ 相关 `DomainMaps/{Name}.md` + `Workbench/memory/patterns.md`
4. 以 method comparison 为主轴构建对比表，标注 `[共识]`/`[矛盾]`/`[独特]`
5. 归纳共识（如"VLA 普遍依赖 large-scale pretraining"）
6. 标出矛盾（如"论文 A 和论文 B 对 in-context learning 效果的结论相悖"），追加到 `Workbench/queue.md` 的 Questions 部分
7. 识别空白（如"few-shot adaptation 在真实机器人上的系统性研究缺失"）
8. Write `Topics/VLA-MethodComparison-Analysis.md`
9. 追加新 pattern 到 `Workbench/memory/patterns.md`
10. 追加日志到 `Workbench/logs/2026-03-26.md`

输出文件：`Topics/VLA-MethodComparison-Analysis.md`

---

**示例 2：指定 wikilinks 列表，通用分析**

```
/cross-paper-analysis --papers "[[2603-EvoScientist]], [[2501-DiffusionPolicy]], [[2410-Pi0]]"
```

执行过程：

1. Glob `Papers/` 定位三篇笔记，Read 内容
2. 构建通用对比表（覆盖问题定义、方法、实验、结果、局限性五列）
3. 归纳共识、矛盾和空白
4. Write `Topics/EvoScientist-DiffusionPolicy-Pi0-Analysis.md`（topic 名取三篇论文 ShortTitle 组合，或由 Human 指定）
5. 追加日志

输出文件：`Topics/EvoScientist-DiffusionPolicy-Pi0-Analysis.md`
