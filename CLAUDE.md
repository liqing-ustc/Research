# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An Obsidian-based knowledge management workspace for AI research. Content is written in mixed Chinese/English style (中英混用): Chinese prose with English technical terms.

## Repository Structure

- `Papers/` — Paper notes, named `AuthorYear-ShortTitle.md` (e.g., `Li2026-RoboClaw.md`)
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

- **Paper notes** use YAML frontmatter with: title, authors, year, venue, tags, url, code, status (`unread`/`reading`/`finished`), rating (1-5), date_added
- **Tags** are flat, using standard English terms: `LLM`, `CV`, `RL`, `transformer`, `RLHF`, `NeurIPS`, etc.
- **Connections** between notes use Obsidian `[[wikilinks]]`
- Paper notes include a `mermaid mindmap` section summarizing the paper structure

## AI Workflow for Paper Notes

### Claude Code（主要方式）
1. User 提供论文名称、关键词或链接
2. Claude Code 根据链接抓取论文内容，或者根据名称和关键词搜索内容
3. 按 `Templates/Paper.md` 格式生成笔记
4. 保存到 `Papers/AuthorYear-ShortTitle.md`
5. User 在 Obsidian 中阅读、修改

### 外部 AI 工具
1. 复制 `Resources/AI-Prompts.md` 中的 prompt，附上论文信息发给 AI
2. AI 输出笔记内容，User 粘贴到 Obsidian 并补充 frontmatter 和链接

## Key Conventions

- No build system, tests, or linting — this is a pure Markdown/Obsidian vault
- `.obsidian/plugins/` is gitignored (plugin binaries); core Obsidian config files are tracked
- The Claudian plugin is used for in-Obsidian AI interaction
