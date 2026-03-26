# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An Obsidian-based knowledge management workspace for AI research. Content is written in mixed Chinese/English style (中英混用): Chinese prose with English technical terms.

## Repository Structure

- `Papers/` — Paper notes, named `YYMM-ShortTitle.md` (e.g., `2603-RoboClaw.md`), YYMM 取自 date_publish
- `Ideas/` — Research ideas (status: raw → developing → validated → archived)
- `Projects/` — Project tracking (status: planning → active → paused → completed)
- `Topics/` — Literature survey/comparison notes
- `Meetings/` — Meeting notes, named `YYYY-MM-DD-Description.md`
- `Daily/` — Daily research logs, named `YYYY-MM-DD.md`
- `Templates/` — Obsidian templates for each note type
- `Resources/` — Reference materials including AI prompts for paper summarization
- `Attachments/` — File attachments

## Creating Notes

All notes must follow their corresponding template in `Templates/`. Key conventions:

- **Paper notes** use YAML frontmatter with: title, authors, date_publish (YYYY-MM-DD/YYYY-MM/YYYY), venue, tags, url, code, status (`unread`/`reading`/`finished`), rating (1-5), date_added
- **Tags** 参照 `Resources/Tag-Taxonomy.md` 中定义的标准 tag，每篇笔记选取 2-4 个；如需新增 tag 请先更新该文件
- **Connections** between notes use Obsidian `[[wikilinks]]`
- Paper notes include a `mermaid mindmap` section summarizing the paper structure

## AI Workflow for Paper Notes

### Claude Code（主要方式）
1. User 提供论文名称、关键词或链接
2. Claude Code 根据链接抓取论文内容，或者根据名称和关键词搜索内容
3. 按 `Templates/Paper.md` 格式生成笔记
4. 保存到 `Papers/YYMM-ShortTitle.md`
5. User 在 Obsidian 中阅读、修改

### 外部 AI 工具
1. 复制 `Resources/AI-Prompts.md` 中的 prompt，附上论文信息发给 AI
2. AI 输出笔记内容，User 粘贴到 Obsidian 并补充 frontmatter 和链接

## MindFlow Skill System

MindFlow 使用标准化的 Markdown skill 来自动化科研工作流。Skills 定义在 `skills/` 目录中，协议文档在 `references/` 中。

### 核心概念
- **Skills**: 定义在 `skills/<category>/<name>/SKILL.md` 中的可执行能力单元
- **Workbench/**: AI 的工作状态（agenda、memory、queue、logs），Human 可随时查看和编辑
- **Topics/Domain-Map.md**: Human-AI 共同维护的核心认知地图

### 自然语言触发

用户不需要记住 skill 名称，用自然语言即可触发：

| 用户说 | 触发 Skill | 执行 |
|--------|-----------|------|
| "阅读论文 xxx" / "读一下这篇 xxx" / "digest this paper" | paper-digest | 读取 `.claude/skills/1-literature/paper-digest/SKILL.md` 并严格按其 Steps 执行 |
| "对比这几篇论文" / "分析一下 VLA 相关论文" | cross-paper-analysis | 读取 `.claude/skills/1-literature/cross-paper-analysis/SKILL.md` 并执行 |
| "整理一下最近的记忆" / "蒸馏记忆" | memory-distill | 读取 `.claude/skills/5-evolution/memory-distill/SKILL.md` 并执行 |

**重要**：触发 skill 时，必须先 Read 对应的 SKILL.md 文件，然后严格按照其中定义的 Steps 和 Guard 执行。不要凭记忆执行——每次都重新读取 SKILL.md。

### 已有 Skills
- `paper-digest`: 消化论文生成笔记（替代上述 AI Workflow 中 Claude Code 方式）
- `cross-paper-analysis`: 跨论文对比分析
- `memory-distill`: 从日志蒸馏记忆

### 协议文档
- `references/skill-protocol.md`: Skill 编写规范
- `references/memory-protocol.md`: 记忆格式和晋升规则
- `references/agenda-protocol.md`: 研究议程管理规则

## Key Conventions

- No build system, tests, or linting — this is a pure Markdown/Obsidian vault
- `.obsidian/plugins/` is gitignored (plugin binaries); core Obsidian config files are tracked
- The Claudian plugin is used for in-Obsidian AI interaction
