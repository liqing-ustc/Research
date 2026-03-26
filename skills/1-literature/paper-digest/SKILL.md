---
name: paper-digest
description: 消化一篇论文，生成结构化笔记到 Papers/
version: 1.0.0
intent: literature
capabilities: [search-retrieval, research-planning]
domain: general
roles: [autopilot, copilot]
autonomy: high
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - WebSearch
  - WebFetch
input:
  - name: source
    description: "论文来源：arXiv URL、PDF 路径、论文标题、或 DOI"
output:
  - file: "Papers/YYMM-ShortTitle.md"
  - memory: "Workbench/logs/YYYY-MM-DD.md (append log entry)"
---

## Purpose

paper-digest 是 MindFlow 最基础的文献技能。给定一篇论文的来源（URL、PDF 路径、标题或 DOI），它自动获取论文内容、提炼核心信息，并按照 `Templates/Paper.md` 格式生成结构化笔记保存至 `Papers/`。

该技能适用于两种工作模式：autopilot（全自动，直接写文件）和 copilot（草稿模式，Human 确认后写入）。它是 literature-sweep、reading-queue 等高阶技能的基础组件。

## Steps

### Step 1：获取论文内容

根据 `source` 的类型选择获取方式：

- **arXiv URL**（如 `https://arxiv.org/abs/2603.08127`）：用 WebFetch 抓取该页面。若需要正文，同时抓取对应的 HTML 全文页（如 `https://arxiv.org/html/2603.08127`）。
- **PDF 路径**（如 `/path/to/paper.pdf`）：用 Read 读取文件内容。
- **论文标题或关键词**：用 WebSearch 搜索（建议加上 `site:arxiv.org` 或 `filetype:pdf`），从结果中定位最可能的论文页面，再用 WebFetch 获取内容。
- **DOI**（如 `10.1145/...`）：用 WebFetch 抓取 `https://doi.org/<DOI>`，跟随重定向到出版商页面。

从获取到的内容中提取以下元数据（如果全文无法获取，至少要获取 abstract）：

| 字段 | 说明 |
|:-----|:-----|
| `title` | 完整论文标题 |
| `authors` | 作者列表（字符串数组） |
| `institute` | 作者所属机构（字符串数组，从 affiliation 提取） |
| `date_publish` | 发表日期，格式 `YYYY-MM-DD`、`YYYY-MM` 或 `YYYY` |
| `venue` | 发表场所，如 `NeurIPS 2025`、`arXiv` |
| `url` | 论文链接（优先用 arXiv abstract 页） |
| `code` | GitHub 代码链接（若论文中提及） |

### Step 2：阅读并理解

通读获取到的内容，重点提炼以下四个维度：

1. **Problem & Motivation**：作者要解决什么问题？现有方法有什么局限？为什么这个问题重要？
2. **Method**：核心方法/架构是什么？关键设计选择有哪些？用简洁的中文描述，保留必要的英文术语。
3. **Key Results**：主要实验结果是什么？在哪些 benchmark 上取得了什么指标？核心 takeaway 是什么？
4. **Strengths & Weaknesses**：方法的亮点与局限，以及对该领域的潜在影响。

如果只能获取 abstract 而非全文，在所有内容区块开头加注：`> [未获取全文，仅基于 abstract]`

### Step 3：确定文件名和 tags

**文件名格式**：`YYMM-ShortTitle.md`

- `YYMM`：取自 `date_publish` 的年份后两位 + 月份，如 `2603`（2026年3月）
- `ShortTitle`：论文标题的 CamelCase 缩写，2-4 个关键词，如 `EvoScientist`、`RoboClaw`、`DiffusionPolicy`

**去重检查**：用 Glob 扫描 `Papers/` 目录，检查是否已存在同名或同主题笔记（搜索标题关键词）。若发现重复，停止并告知 Human，不创建新文件。

**Tag 选择**：阅读 `references/Tag-Taxonomy.md`，从中选取 2-4 个最能描述该论文的 tag。必须严格使用文件中已定义的 tag，不得自创。

### Step 4：生成笔记

按照 `Templates/Paper.md` 填写所有字段，生成完整的笔记内容：

**Frontmatter**（YAML）：

```yaml
---
title: <完整论文标题>
authors: [<作者1>, <作者2>, ...]
institute: [<机构1>, <机构2>, ...]
date_publish: <YYYY-MM-DD 或 YYYY-MM 或 YYYY>
venue: <发表场所>
tags: [<tag1>, <tag2>, ...]
url: <论文链接>
code: <代码链接，无则留空>
status: unread
rating:
date_added: "<今天日期 YYYY-MM-DD>"
---
```

**正文各节**（用中文撰写，保留英文技术术语）：

- `## Summary`：一句话概括论文解决了什么问题、怎么解决的。
- `## Problem & Motivation`：问题背景与动机，2-5 句话。
- `## Method`：核心方法描述，可分段，鼓励列出关键组件。
- `## Key Results`：主要实验结果，包含具体数字和 benchmark 名称。
- `## Strengths & Weaknesses`：方法优势与局限的个人评价。
- `## Mind Map`：用 mermaid mindmap 可视化论文结构，root 节点用论文 ShortTitle，子节点覆盖 Problem / Method / Results 三个维度。
- `## Connections`：
  - 用 Grep 在 `Papers/` 中搜索方法名、任务名等关键词，找出相关论文笔记，填入 `[[wikilinks]]`。
  - 用 Grep 在 `Ideas/` 中搜索相关关键词，填入相关 idea 的链接。
  - 若无相关内容，对应子项留空。
- `## Notes`：留空，供 Human 后续填写。

### Step 5：保存并记录

1. **写入文件**：用 Write 将笔记保存到 `Papers/YYMM-ShortTitle.md`。

2. **追加日志**：用 Edit（或 Write 若文件不存在）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

   ```markdown
   ### [HH:MM] paper-digest
   - **input**: <source 的原始内容>
   - **output**: [[Papers/YYMM-ShortTitle]]
   - **observation**: <一句话描述论文核心贡献>
   - **status**: success
   ```

   若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

3. **阅读队列**：若 source 来自阅读队列（`Workbench/queue/reading.md`），用 Edit 将对应条目标记为已完成（如在行首添加 `✓` 或删除该条目）。

## Guard

- **不覆盖已有笔记**：若 `Papers/YYMM-ShortTitle.md` 已存在，停止执行并告知 Human，不得覆盖或修改已有文件。
- **不捏造信息**：所有字段必须来自论文原文。无法获取全文时，在受影响的章节开头标注 `> [未获取全文，仅基于 abstract]`，不得推测正文内容填充 Method / Key Results 等节。
- **Tags 必须来自 Tag-Taxonomy**：只能使用 `references/Tag-Taxonomy.md` 中已定义的 tag。如果发现论文需要尚未定义的 tag，在 `## Notes` 节中用注释建议新增，不得直接写入 frontmatter。
- **rating 留给 Human**：`rating` 字段始终留空，不得填写任何值。
- **语言规范**：正文用中文撰写，英文技术术语（模型名、方法名、benchmark 名）保持英文，不做翻译。
- **autonomy: copilot 模式**：若用户以 copilot 模式调用，生成笔记草稿后先输出给 Human 预览，确认后再执行 Write；日志同样在确认后追加。

## Examples

**示例 1：从 arXiv URL 消化论文**

```
/paper-digest "https://arxiv.org/abs/2603.08127"
```

执行过程：
1. WebFetch `https://arxiv.org/abs/2603.08127` 获取 abstract、作者、日期等元数据
2. WebFetch `https://arxiv.org/html/2603.08127` 获取 HTML 全文
3. 提炼 Problem / Method / Results / Strengths & Weaknesses
4. Glob `Papers/2603-*.md` 检查是否重复
5. 读取 `references/Tag-Taxonomy.md` 选取合适 tags
6. Grep `Papers/` 和 `Ideas/` 搜索相关笔记
7. 写入 `Papers/2603-EvoScientist.md`
8. 追加日志到 `Workbench/logs/2026-03-26.md`

输出文件：`Papers/2603-EvoScientist.md`

---

**示例 2：从论文标题搜索**

```
/paper-digest "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
```

执行过程：
1. WebSearch `"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" site:arxiv.org`
2. 获取 arXiv 链接，WebFetch 抓取内容
3. 后续步骤同示例 1

输出文件：`Papers/2303-DiffusionPolicy.md`

---

**示例 3：从本地 PDF 消化**

```
/paper-digest "/Users/qingli/Downloads/roboclaw_2025.pdf"
```

执行过程：
1. Read `/Users/qingli/Downloads/roboclaw_2025.pdf` 读取 PDF 内容
2. 从内容中提取元数据（title、authors、date、venue）
3. 后续步骤同示例 1
