---
name: literature-survey
description: 当用户说"调研""survey""了解研究现状"，或需要系统了解某主题的文献全貌时，以 DomainMap 为认知基线、近期 survey paper 为文献锚点，批量 digest 后综合生成调研报告
argument-hint: <topic>
allowed-tools: Read, Write, Edit, Glob, Grep, Task, WebSearch, WebFetch
---

## Purpose

给定一个研究主题，以 `DomainMaps/` 为认知基线、以近期 survey paper 为文献锚点，批量 digest 相关论文，综合出 `Topics/{Topic}-Survey.md` —— 作为对相关 domain 当前认知的 delta 报告（无对应 DomainMap 时退化为完整领域分析）。

## Steps

### 1. 基线建立
- **认知基线**：读 `DomainMaps/_index.md` 找相关 domain；若有，读对应 `DomainMaps/{Name}.md`
- **内容基线**：检查 `Topics/{Topic}-Survey.md`；若存在，读取该文件，用 frontmatter 的 `date_updated` 作为截止日期，扫描 body 中形如 `[[YYMM-ShortTitle]]`（如 `[[2510-XVLA]]`、`[[2602-DM0]]`）的 wikilinks 作为已收录论文清单，作为增量调研的起点；若无，本次是新建

### 2. Scoping

#### 2.1 搜索相关 Survey
搜索本主题近期新出现的 survey paper（若 Step 1 存在内容基线，优先找其截止日期之后的）。先在已有 `Papers/`里寻找，再结合 WebSearch，挑 1-3 篇最相关的。若本 topic 无专门 survey，退而找**相邻 / 上位 topic 的 survey paper**。

为挑选出来的 survey paper 启动 subagent 并行调用 `paper-digest` 生成笔记，主 agent 读取这些 survey paper 的笔记文件。

#### 2.2 确定候选论文清单
与 Step 1 已收录清单去重后得到候选论文清单，分两类：
- **需 digest**：从侦察到的 survey 中提取反复引用的 canonical 论文 → 进入 Step 3
- **已有笔记**：从 `Papers/` 搜索本 topic 相关笔记 → 无需 digest，直接参与 Step 4 综合

### 3. 批量 paper-digest
为每篇待 digest 的论文启动 subagent 并行调用 `paper-digest` 生成笔记；若 paper-digest 发现论文已有笔记，则跳过（不覆盖、不询问）。等所有 subagent 完成后，进入下一步。

### 4. 综合写作

**读取笔记**：只读取候选论文清单里 rating >= 2 的论文笔记，作为新增论文（筛 rating 用 Bash 批处理 frontmatter，避免对每篇 Read）

**确定 section 结构：**  下列必备元素为骨架；各元素内部的子分类由**已有对应 DomainMap + 已有 `Topics/{Topic}-Survey.md`（增量时） + Step 2.1 新侦察的 survey paper** 综合提炼：

- **Overview**：一句话定位 + 领域活跃度（时间线 / 参与格局 / 学术产出等） + 整体趋势
- **Problem & Motivation**：本领域在解决什么核心问题，为什么重要，为什么适合现在做
- **技术路线对比**：分析各主流路线的核心思路，实际效果，优缺点等，嵌入代表性论文
- **Datasets & Benchmarks**：整理常用的 Training Datasets，Benchmarks（包含 SOTA methods 和 performance），可以用多个表格方式呈现
- **Open Problems**：真实的 open，不是教科书式列举
- **调研日志**：元数据（日期、论文统计、未能获取说明）

**写作基线**：Survey 定位为 DomainMap 的 delta 报告——已在对应 DomainMap 中 Established 的内容不重述，聚焦窗口内的新技术路线、对既有 claim 的支持/挑战、新 open problems；必要时用 `[[DomainMaps/{Name}#某节]]` 引用基线。无对应 DomainMap 时完整展开所有必备元素。

若调研中发现明显应纳入 DomainMap 的内容，在 `## 调研日志` 之前加 `## DomainMap 更新建议` 节显式列出；无则不加此节。

**Frontmatter**（新建或更新时写入/刷新）：
```yaml
---
title: <Topic 全名>
description: <一句话描述本 Survey 覆盖的范围与核心发现>
tags: [...]
date_updated: YYYY-MM-DD
year_range: YYYY-YYYY
---
```

**Tag 选择**：阅读 vault 目录下的 `{vault_root}/references/tags.md`，按照规范选择 tag。

**Obsidian syntax**：写入前参照 `{vault_root}/references/obsidian-syntax.md` Obsidian-specific quirks。

新建或增量更新 `Topics/{Topic}-Survey.md`。

### 5. 日志
追加 log entry 到 `Workbench/logs/YYYY-MM-DD.md`（文件不存在则新建并加一级标题 `# YYYY-MM-DD`）：

```markdown
### [HH:MM] literature-survey
- **input**: <topic>
- **output**: [[Topics/{Topic}-Survey]]
- **stats**: 侦察 survey N 篇，新增论文 N 篇
- **observation**: <一句话领域核心发现>
- **issues**: <未能获取的论文及原因；无则 none>
```

## Verify

### Step 1 · 基线建立
- [ ] 认知基线：已读 `DomainMaps/_index.md`；若 topic 对应某已有 domain，已读对应 DomainMap
- [ ] 内容基线：若 `Topics/{Topic}-Survey.md` 已存在，已从 frontmatter 读 `date_updated`、从 body wikilinks 重建论文清单

### Step 2 · Scoping
- [ ] 搜索 survey 的尝试路径已走过（本主题→相邻/上位）；若均无，Survey 调研日志中明示"无参考 survey"
- [ ] 候选论文清单已对 Step 1 已收录清单去重

### Step 3 · 批量 paper-digest
- [ ] "需 digest" bucket 中每篇 canonical 论文均已启动 paper-digest；issues 仅允许记录 paper-digest 实际失败（PDF 不可得 / API 错误等），不允许为控时长 / 规模而主观跳过

### Step 4 · 综合写作
- [ ] 所有新增论文（候选论文清单中 rating >= 2） 的笔记已**全部完整读取，无跳过，无截断**
- [ ] 6 条必备元素（Overview / Problem & Motivation / 技术路线对比 / Datasets & Benchmarks / Open Problems / 调研日志）全部出现
- [ ] 若 Step 2.1 有 survey 输出，本次笔记的 section 结构可追溯到其 taxonomy
- [ ] 技术路线对比有分析深度——不止罗列 method 是什么，还说清楚各自的实际效果与优缺点
- [ ] 若有对应 DomainMap，Survey 未大段重述其 Established Knowledge
- [ ] Survey 中每篇引用都用 `[[wikilink]]`，且 vault 中存在对应笔记（未能获取的论文在调研日志中明示）
- [ ] Survey 中无遗留的 `%%...%%` 注释或占位符
- [ ] 若 `{Topic}-Survey.md` 已存在：新内容与旧内容已 merge，未无故删除仍然有效的旧论据
- [ ] 有 DomainMap 更新建议时已作为独立 section 列出；未直接修改 `DomainMaps/` 下任何文件

### Step 5 · 日志
- [ ] 日志 entry 已追加到 `Workbench/logs/YYYY-MM-DD.md`，失败论文已记入 issues
