# MindFlow Phase 1: Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the MindFlow repository structure, protocol documents, 3 core skills (paper-digest, cross-paper-analysis, memory-distill), vault templates, and a working demo vault — so that a user can `git clone`, copy skills to their agent, and start using MindFlow immediately.

**Architecture:** Pure Markdown skill protocol (Layer 1 only). No backend, no dependencies. Skills are SKILL.md files following the MindFlow format (ARIS frontmatter + Dr. Claw taxonomy + uditgoenka references/ separation). Vault state lives in `Workbench/` directory. All output follows existing MindFlow conventions (Papers/, Topics/, Ideas/).

**Tech Stack:** Markdown, YAML frontmatter, JSON (taxonomy schema + stage-skill-map), Mermaid (diagrams in docs)

**Spec:** `docs/superpowers/specs/2026-03-26-mindflow-design.md`

**Scope notes:**
- `npx mindflow install` deferred to a later phase; Phase 1 is manual `git clone` + copy only.
- `examples/demo-vault/` simplified to `examples/demo-walkthrough.md` (tutorial document, more useful for Phase 1 than empty vault scaffold).
- `Experiments/<id>/` subdirectory structure (per spec) simplified to single-file template for Phase 1; subdirectory structure deferred to Phase 3 (experiment skills).

**Important context:**
- This is a pure Markdown/Obsidian vault. No build system, no test runner, no linting.
- "Testing" means: manually invoke the skill in Claude Code and verify it produces correct output files.
- The vault already has `Papers/`, `Topics/`, `Ideas/`, `Templates/`, `Resources/`, `Daily/`, `Meetings/`, `Projects/`, `Attachments/`.
- Existing templates (`Templates/Paper.md`, `Templates/Idea.md`) must not be broken.
- Content is mixed Chinese/English (Chinese prose, English technical terms). New templates follow this convention.
- Tags follow `references/Tag-Taxonomy.md`.

---

## File Structure (Phase 1 deliverables)

```
MindFlow/                              # Existing vault root
│
├── skills/                            # NEW: skill definitions
│   ├── taxonomy.schema.json           # Skill classification standard
│   ├── stage-skill-map.json           # Stage × Task → Skill routing (Phase 1 subset)
│   ├── 1-literature/
│   │   ├── paper-digest/
│   │   │   └── SKILL.md              # Atomic: digest one paper into a note
│   │   └── cross-paper-analysis/
│   │       └── SKILL.md              # Orchestration: compare multiple papers
│   └── 5-evolution/
│       └── memory-distill/
│           └── SKILL.md              # Atomic: extract patterns from logs
│
├── references/                        # NEW: protocol documents
│   ├── skill-protocol.md             # How to write a SKILL.md
│   ├── memory-protocol.md            # Memory format + insight promotion rules
│   └── agenda-protocol.md            # Research agenda management rules
│
├── Templates/                         # EXISTING dir, add new templates
│   ├── Paper.md                       # EXISTING — do not modify
│   ├── Idea.md                        # EXISTING — do not modify
│   ├── Experiment.md                  # NEW: experiment record template
│   ├── Report.md                      # NEW: AI report template
│   └── Domain-Map.md                  # NEW: domain map template
│
├── Workbench/                         # NEW: AI working state (init)
│   ├── agenda.md                      # Research agenda (initialized empty)
│   ├── identity.md                    # AI self-awareness + Autopilot rules
│   ├── memory/
│   │   ├── insights.md
│   │   ├── failed-directions.md
│   │   ├── effective-methods.md
│   │   └── patterns.md
│   ├── queue/
│   │   ├── reading.md
│   │   ├── experiments.md
│   │   ├── questions.md
│   │   └── review.md
│   ├── logs/                          # Empty dir (logs created at runtime)
│   └── evolution/
│       └── changelog.md
│
├── Experiments/                       # NEW: empty dir, ready for experiment records
├── Reports/                           # NEW: empty dir, ready for AI reports
│
├── Topics/
│   └── Domain-Map.md                  # NEW: initialized from template
│
└── examples/                          # NEW: demo vault walkthrough
    └── demo-walkthrough.md            # Shows how to use the 3 skills end-to-end
```

---

## Task 1: Protocol Documents

Write the foundational protocol docs that all skills reference.

**Files:**
- Create: `references/skill-protocol.md`
- Create: `references/memory-protocol.md`
- Create: `references/agenda-protocol.md`

- [ ] **Step 1: Write skill-protocol.md**

Define the SKILL.md format specification:

```markdown
# Skill Protocol

> This document defines the standard format for MindFlow skills.
> All skills MUST follow this format to ensure cross-agent portability.

## SKILL.md Format

Every skill is a single Markdown file with YAML frontmatter + instructions body.

### Frontmatter (Required Fields)

| Field | Type | Description |
|-------|------|-------------|
| name | string | Skill identifier (kebab-case) |
| description | string | One-line purpose |
| version | semver | e.g. 1.0.0 |
| intent | enum | literature / ideation / experiment / analysis / writing / evolution / orchestration / utility |
| capabilities | string[] | From: research-planning, cross-validation, search-retrieval, data-processing, training-tuning, evaluation-benchmarking, prompt-structured-output, visualization-reporting, agent-workflow |
| domain | enum | general / cs-ai / bioinformatics / medical / vision / nlp / data-engineering |
| roles | enum[] | Subset of: autopilot, sparring, copilot |
| autonomy | enum | high (fully autonomous) / medium (autonomous but Human review recommended) / low (Human must be present) |
| allowed-tools | string[] | Tools this skill may use (e.g. Read, Write, Edit, Glob, Grep, Bash, WebSearch, WebFetch) |

### Frontmatter (Optional Fields)

| Field | Type | Description |
|-------|------|-------------|
| input | object[] | Each with name + description |
| output | object[] | Each with file pattern + memory file if applicable |
| related-skills | string[] | Other skills this one may invoke |

### Body Sections

| Section | Required | Purpose |
|---------|----------|---------|
| Purpose | Yes | What this skill does and why |
| Steps | Yes | Numbered execution instructions |
| Guard | Yes | Safety constraints (what NOT to do) |
| Examples | Recommended | Usage examples |

### Complex Skills

Skills with >200 lines of instructions SHOULD split detailed protocols into a `references/` subdirectory:

```
skills/<category>/<skill-name>/
├── SKILL.md              # Entry point: frontmatter + overview
└── references/
    ├── detailed-protocol.md
    └── ...
```

The SKILL.md body references these with: "See references/detailed-protocol.md for full protocol."

### Skill Levels

- **Level 0 (Atomic)**: One specific task. No skill dependencies.
- **Level 1 (Orchestration)**: Chains multiple Level 0 skills.
- **Level 2 (Global)**: Manages research agenda and dispatches Level 1 skills.

## Naming Conventions

- Skill directories: kebab-case (e.g. `paper-digest`, `cross-paper-analysis`)
- Category directories: numbered prefix + kebab-case (e.g. `1-literature`, `5-evolution`)
- SKILL.md is always uppercase
```

- [ ] **Step 2: Write memory-protocol.md**

```markdown
# Memory Protocol

> Defines how AI working memory is structured, updated, and promoted to shared knowledge.

## Directory Structure

All memory lives in `Workbench/memory/` as Markdown files.

| File | Purpose | Write Frequency |
|------|---------|-----------------|
| insights.md | Discovered insights (provisional → validated) | Per analysis cycle |
| failed-directions.md | Abandoned research directions + reasons | When direction fails |
| effective-methods.md | Proven experiment strategies | After experiment analysis |
| patterns.md | Observed cross-paper/cross-experiment patterns | Per analysis cycle |

## Entry Format

### insights.md

Each entry is an H3 heading:

```markdown
### [YYYY-MM-DD] Insight title
- **claim**: One-sentence statement
- **evidence**: [[Papers/xxx]], [[Experiments/xxx]]
- **confidence**: high / medium / low
- **source**: literature-analysis / experiment / cross-validation
- **impact**: Which research directions affected
- **status**: provisional / validated / integrated
```

### failed-directions.md

```markdown
### [YYYY-MM-DD] Direction name
- **original_hypothesis**: What was believed
- **evidence_against**: [[Papers/xxx]], [[Experiments/xxx]]
- **lesson**: One-sentence takeaway for future reference
- **related_directions**: Which similar directions should be cautious
```

### effective-methods.md

```markdown
### [YYYY-MM-DD] Method name
- **context**: When this method is effective
- **method**: What to do
- **evidence**: [[Experiments/xxx]]
- **pitfalls**: What to watch out for
```

### patterns.md

```markdown
### [YYYY-MM-DD] Pattern description
- **observation**: What was noticed
- **occurrences**: [[Papers/xxx]], [[Papers/yyy]], ...
- **confidence**: low / medium
- **needs_verification**: yes / no
```

## Insight Promotion Hierarchy

```
Level 0: Raw Log (Workbench/logs/YYYY-MM-DD.md)
    ↓  memory-distill skill
Level 1: Pattern (Workbench/memory/patterns.md)
    ↓  multiple independent observations
Level 2: Provisional Insight (Workbench/memory/insights.md, status: provisional)
    ↓  cross-validation (experiment or literature)
Level 3: Validated Insight (Workbench/memory/insights.md, status: validated)
    ↓  ≥2 independent evidence sources + confidence > 0.8
Level 4: Domain Map (Topics/Domain-Map.md)
```

### Promotion Rules

- Level 0 → 1: memory-distill skill extracts patterns automatically
- Level 1 → 2: Pattern observed ≥3 times independently → becomes provisional insight
- Level 2 → 3: Supported by ≥2 independent sources (different papers OR paper + experiment)
- Level 3 → 4: AI may auto-promote if confidence > 0.8 and ≥2 independent evidence sources. Otherwise write to Workbench/queue/review.md for Human review.

### Update Rules

- All memory files are **append-only** (never delete entries, mark as superseded if outdated)
- Every update to Domain-Map by AI must be logged in Workbench/evolution/changelog.md
- Human can write directly to any memory file or Domain-Map without restrictions
```

- [ ] **Step 3: Write agenda-protocol.md**

```markdown
# Agenda Protocol

> Defines how the research agenda (Workbench/agenda.md) is structured and managed.

## File Format

```markdown
---
last_updated: YYYY-MM-DD
updated_by: ai / human / both
---
## Mission
[Human-defined long-term research mission. AI does not modify this section.]

## Active Directions
### 1. [Direction name]
- **priority**: high / medium / low
- **status**: exploring / validating / consolidating
- **origin**: human-assigned / ai-discovered / paper-inspired
- **hypothesis**: Current core hypothesis
- **evidence**: [[Papers/xxx]], [[Experiments/xxx]]
- **next_action**: Next step plan
- **confidence**: 0.0-1.0

## Paused Directions
### [Direction name]
- **reason**: Why paused
- **resume_condition**: What would trigger resumption

## Abandoned Directions
### [Direction name]
- **reason**: Why abandoned (linked to Workbench/memory/failed-directions.md)
- **date_abandoned**: YYYY-MM-DD

## Pending Decisions
- [Question requiring Human judgment]
```

## Management Rules

### AI Permissions

| Action | Autopilot Allowed? | Condition |
|--------|--------------------|-----------|
| Add exploring direction (medium/low priority) | Yes | Supported by evidence |
| Add exploring direction (high priority) | No | Write to Pending Decisions |
| Change status exploring → validating | Yes | When evidence supports hypothesis |
| Change status → abandoned | No | Write to Pending Decisions |
| Update next_action | Yes | After completing previous action |
| Update confidence | Yes | Based on new evidence |
| Modify Mission | Never | Human-only section |

### Human Overrides

Human can edit agenda.md freely at any time. Changes made by Human take effect immediately — AI reads the file at the start of each cycle and respects the current state.

### Sync with Memory

When a direction is abandoned:
1. AI writes failure analysis to Workbench/memory/failed-directions.md (IVE)
2. AI moves the direction to Abandoned Directions section
3. AI logs the change in Workbench/evolution/changelog.md
```

- [ ] **Step 4: Commit protocol documents**

```bash
git add references/
git commit -m "Add core protocol documents: skill, memory, agenda"
```

---

## Task 2: Taxonomy Schema and Stage-Skill Map

**Files:**
- Create: `skills/taxonomy.schema.json`
- Create: `skills/stage-skill-map.json`

- [ ] **Step 1: Write taxonomy.schema.json**

Adapted from Dr. Claw's `skills-taxonomy-v2.schema.json`, extended with MindFlow's `roles` and `autonomy` fields:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MindFlow Skill Taxonomy",
  "type": "object",
  "required": ["name", "description", "version", "intent", "capabilities", "domain", "roles", "autonomy"],
  "properties": {
    "name": { "type": "string", "minLength": 1 },
    "description": { "type": "string", "minLength": 1 },
    "version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "intent": {
      "type": "string",
      "enum": ["literature", "ideation", "experiment", "analysis", "writing", "evolution", "orchestration", "utility"]
    },
    "capabilities": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["search-retrieval", "research-planning", "cross-validation", "data-processing", "training-tuning", "evaluation-benchmarking", "prompt-structured-output", "visualization-reporting", "agent-workflow"]
      },
      "minItems": 1, "uniqueItems": true
    },
    "domain": {
      "type": "string",
      "enum": ["general", "cs-ai", "bioinformatics", "medical", "vision", "nlp", "data-engineering"]
    },
    "roles": {
      "type": "array",
      "items": { "type": "string", "enum": ["autopilot", "copilot", "sparring"] },
      "minItems": 1, "uniqueItems": true
    },
    "autonomy": {
      "type": "string",
      "enum": ["high", "medium", "low"]
    },
    "status": {
      "type": "string",
      "enum": ["candidate", "verified", "experimental", "deprecated"]
    },
    "status": {
      "type": "string",
      "enum": ["candidate", "verified", "experimental", "deprecated"]
    },
    "summary": { "type": "string", "minLength": 1 },
    "related-skills": {
      "type": "array",
      "items": { "type": "string" },
      "uniqueItems": true
    }
  }
}
```

- [ ] **Step 2: Write stage-skill-map.json (Phase 1 subset)**

Only include the 3 skills being built in Phase 1. Placeholder entries for future skills.

```json
{
  "_comment": "MindFlow Stage × Task → Skill routing. Phase 1 subset.",
  "literature": {
    "discovery": [],
    "analysis": ["paper-digest", "cross-paper-analysis"],
    "synthesis": []
  },
  "ideation": {
    "generation": [],
    "evaluation": []
  },
  "experiment": {
    "design": [],
    "execution": [],
    "analysis": []
  },
  "writing": {
    "planning": [],
    "drafting": [],
    "review": []
  },
  "evolution": {
    "distill": ["memory-distill"],
    "retrieve": [],
    "agenda": []
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add skills/taxonomy.schema.json skills/stage-skill-map.json
git commit -m "Add skill taxonomy schema and stage-skill routing map"
```

---

## Task 3: Vault Templates

New templates for Experiment, Report, Domain-Map. Plus initialize `Workbench/` directory.

**Files:**
- Create: `Templates/Experiment.md`
- Create: `Templates/Report.md`
- Create: `Templates/Domain-Map.md`

- [ ] **Step 1: Write Templates/Experiment.md**

```markdown
---
title:
idea: ""           # [[Ideas/xxx]] 关联的 idea
tags: []
status: planned    # planned / running / completed / failed
date_created: "{{date}}"
date_completed:
---
## Objective
这个实验要验证什么假设？

## Setup
- **代码**: [repo 链接或本地路径]
- **数据**: [数据集描述]
- **环境**: [硬件/软件环境]
- **关键参数**: [实验参数]

## Method
实验步骤描述。

## Results
主要结果（数据、图表、关键指标）。

## Analysis
结果说明了什么？假设是否被验证？

## Insights
从这个实验中获得的 insight（可链接到 Domain-Map）。

## Next Steps
后续实验或研究方向建议。
```

- [ ] **Step 2: Write Templates/Report.md**

```markdown
---
type: weekly       # weekly / discovery / decision-needed
period:            # YYYY-MM-DD ~ YYYY-MM-DD
date_created: "{{date}}"
---
## Highlights
最重要的 1-3 个发现。

## Progress by Direction

### [方向名称]
- **本周做了什么**:
- **关键发现**:
- **下一步**:
- **需要 Human 决策**: [是/否]

## New Discoveries
意外发现的 pattern / 值得关注的新论文。

## Experiments Summary

| Experiment | Status | Key Result |
|-----------|--------|-----------|
| [[Experiments/xxx]] | | |

## Questions for Human
1.

## Resource Usage
- Papers read:
- Experiments run:
- API tokens consumed: ~
```

- [ ] **Step 3: Write Templates/Domain-Map.md**

```markdown
---
last_updated: "{{date}}"
contributors: []
---
## [领域/子领域名称]

### Established Knowledge
高置信度的领域共识。
- [claim] — evidence: [[Papers/xxx]]

### Active Debates
存在矛盾的观点。
- [claim A] vs [claim B]
  - 支持 A: [[Papers/xxx]]
  - 支持 B: [[Papers/yyy]]
  - **status**: unresolved / leaning-A / leaning-B

### Open Questions
尚未回答的问题。
- [question] — related: [[Ideas/xxx]]

### Known Dead Ends
已证伪的方向。
- [direction] — why: [原因]
```

- [ ] **Step 4: Commit templates**

```bash
git add Templates/Experiment.md Templates/Report.md Templates/Domain-Map.md
git commit -m "Add templates: Experiment, Report, Domain-Map"
```

---

## Task 4: Initialize Workbench/ Directory

Set up the AI working state directory with starter content.

**Files:**
- Create: `Workbench/agenda.md`
- Create: `Workbench/identity.md`
- Create: `Workbench/memory/insights.md`
- Create: `Workbench/memory/failed-directions.md`
- Create: `Workbench/memory/effective-methods.md`
- Create: `Workbench/memory/patterns.md`
- Create: `Workbench/queue/reading.md`
- Create: `Workbench/queue/experiments.md`
- Create: `Workbench/queue/questions.md`
- Create: `Workbench/queue/review.md`
- Create: `Workbench/evolution/changelog.md`
- Create: `Workbench/logs/.gitkeep`

- [ ] **Step 1: Write Workbench/agenda.md**

```markdown
---
last_updated: 2026-03-26
updated_by: human
---
## Mission

[在此定义你的长期研究使命。例如："探索 Embodied AI 中 VLA 模型在 mobile manipulation 场景下的 scaling law 与 sample efficiency"]

## Active Directions

_尚无活跃方向。在此添加你的第一个研究方向，或让 AI 基于你的 Papers/ 和 Ideas/ 自动发现。_

## Paused Directions

_无_

## Abandoned Directions

_无_

## Pending Decisions

_无_
```

- [ ] **Step 2: Write Workbench/identity.md**

```markdown
## Domain

[描述你的研究领域，如 "Embodied AI, focusing on VLA + mobile manipulation"]

## Expertise

_AI 的能力自评会随使用积累自动更新。初始状态为空。_

## Collaboration Preferences

- **autonomy_level**: moderate
- **report_frequency**: weekly
- **human_review_required**: [abandon direction, start long experiments, modify Domain-Map established knowledge]

## Autopilot Rules

- CAN: read papers, update memory, generate reports, discover new papers, explore new directions based on agenda
- CAN: auto-promote validated insight to Domain-Map (per Domain-Map update rules in memory-protocol.md)
- NEED APPROVAL: start experiments >2h, abandon a research direction, exceed daily API budget
- CANNOT: delete existing notes, modify Human-written content, publish externally
- MUST: log all operations to Workbench/logs/, trigger Reporter mode for major discoveries

## Budget

- **daily_token_limit**: 500000
- **per_cycle_limit**: 50000
- **expensive_action_threshold**: 100000
```

- [ ] **Step 3: Write memory files (empty initialized)**

`Workbench/memory/insights.md`:
```markdown
# Insights

> 已发现的 insight 集合。格式参见 references/memory-protocol.md。

_尚无 insight。AI 运行后会自动积累。_
```

`Workbench/memory/failed-directions.md`:
```markdown
# Failed Directions

> 已放弃的研究方向和失败教训。格式参见 references/memory-protocol.md。

_尚无记录。_
```

`Workbench/memory/effective-methods.md`:
```markdown
# Effective Methods

> 经验证有效的实验策略。格式参见 references/memory-protocol.md。

_尚无记录。_
```

`Workbench/memory/patterns.md`:
```markdown
# Patterns

> 跨论文/跨实验观察到的 pattern。格式参见 references/memory-protocol.md。

_尚无记录。_
```

- [ ] **Step 4: Write queue files (empty initialized)**

`Workbench/queue/reading.md`:
```markdown
# Reading Queue

> 待读论文。Human 和 AI 都可以往这里添加。

_队列为空。_
```

`Workbench/queue/experiments.md`:
```markdown
# Experiment Queue

> 待跑实验。Human 和 AI 都可以往这里添加。

_队列为空。_
```

`Workbench/queue/questions.md`:
```markdown
# Open Questions

> 待探索的问题。Human 和 AI 都可以往这里添加。

_队列为空。_
```

`Workbench/queue/review.md`:
```markdown
# Review Queue

> 待 Human review 的 AI 产出。

_队列为空。_
```

- [ ] **Step 5: Write evolution/changelog.md**

```markdown
# Evolution Changelog

> 记录 AI 对 agenda/memory/Domain-Map 的重要变更。

### 2026-03-26 [System] Initialized
- **action**: MindFlow Workbench initialized
- **content**: Empty state, ready for first research cycle
```

- [ ] **Step 6: Create logs/.gitkeep and Experiments/ and Reports/ directories**

```bash
mkdir -p Workbench/logs && touch Workbench/logs/.gitkeep
mkdir -p Experiments && touch Experiments/.gitkeep
mkdir -p Reports && touch Reports/.gitkeep
```

- [ ] **Step 7: Commit Workbench and new directories**

```bash
git add Workbench/ Experiments/ Reports/
git commit -m "Initialize Workbench/ (AI state), Experiments/, Reports/ directories"
```

---

## Task 5: Initialize Topics/Domain-Map.md

Create the shared cognition file from template.

**Files:**
- Create: `Topics/Domain-Map.md`

- [ ] **Step 1: Write Topics/Domain-Map.md**

Based on `Templates/Domain-Map.md` but with initial content seeded from the user's existing vault (the user works in Embodied AI / VLA / mobile manipulation):

```markdown
---
last_updated: "2026-03-26"
contributors: [human]
---

# Domain Map

> Human-AI 共同维护的核心认知地图。这是整个 vault 中最重要的文件。
> 所有 Papers/Ideas/Experiments 的精华汇聚于此。

_这是一个新初始化的 Domain Map。随着你和 AI 一起阅读论文、跑实验、讨论 idea，这里会逐渐充实。_

_你可以直接编辑这个文件写入你已有的领域认知，也可以让 AI 从你现有的 Papers/ 和 Topics/ 中自动提取。_

## [你的研究领域]

### Established Knowledge
_高置信度的领域共识。_

### Active Debates
_存在矛盾的观点。_

### Open Questions
_尚未回答的问题。_

### Known Dead Ends
_已证伪的方向。_
```

- [ ] **Step 2: Commit**

```bash
git add Topics/Domain-Map.md
git commit -m "Initialize shared Domain-Map"
```

---

## Task 6: Skill — paper-digest

The first and most essential skill: digest a paper into a structured note following `Templates/Paper.md`.

**Files:**
- Create: `skills/1-literature/paper-digest/SKILL.md`

- [ ] **Step 1: Write the SKILL.md**

```markdown
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

# Paper Digest

## Purpose

给定一篇论文（URL、标题或 PDF），生成结构化笔记并保存到 `Papers/`。笔记格式严格遵循 `Templates/Paper.md`。这是 MindFlow 最基础的技能，将人工读论文 + 记笔记的过程半自动化。

## Steps

### Step 1: 获取论文内容

- 如果输入是 arXiv URL → 用 WebFetch 抓取论文页面，提取 abstract、正文
- 如果输入是论文标题 → 用 WebSearch 搜索，找到论文链接后用 WebFetch 抓取
- 如果输入是本地 PDF → 用 Read 读取
- 提取：title, authors, institute, date_publish, venue, url, code (GitHub link if available)

### Step 2: 阅读并理解

通读论文，重点关注：
- Problem & Motivation: 解决什么问题？为什么重要？
- Method: 核心方法/架构
- Key Results: 主要实验结果
- Strengths & Weaknesses: 方法的优缺点

### Step 3: 确定文件名和 tags

- 文件名：`YYMM-ShortTitle.md`，YYMM 取自 date_publish
- 从 `references/Tag-Taxonomy.md` 中选取 2-4 个 tags
- 用 Glob 检查 `Papers/` 下是否已有同名文件，避免覆盖

### Step 4: 生成笔记

按 `Templates/Paper.md` 格式填写所有字段：
- frontmatter: title, authors, institute, date_publish, venue, tags, url, code, status: unread, rating（留空让 Human 填）, date_added
- Summary: 一句话概括
- Problem & Motivation
- Method
- Key Results
- Strengths & Weaknesses
- Mind Map: 生成 mermaid mindmap
- Connections: 用 Grep 搜索 Papers/ 和 Ideas/ 中与该论文相关的笔记，填入 Related papers/ideas
- Notes: 留空

### Step 5: 保存并记录

- 写入 `Papers/YYMM-ShortTitle.md`
- 追加日志到 `Workbench/logs/YYYY-MM-DD.md`:
  ```
  ### [HH:MM] paper-digest
  - **input**: [论文标题]
  - **output**: [[Papers/YYMM-ShortTitle.md]]
  - **observation**: [一句话：这篇论文的核心贡献是什么]
  - **status**: success
  ```
- 如果论文来自 `Workbench/queue/reading.md`，将对应条目标记为已完成

## Guard

- 不修改任何已存在的 Papers/ 笔记
- 不捏造论文内容——如果无法获取全文，在笔记中标注 "[未获取全文，仅基于 abstract]"
- tags 必须来自 `references/Tag-Taxonomy.md`；如果需要新 tag，在 Notes 部分建议而非自行添加
- rating 字段留空，由 Human 填写
- 笔记语言：中文描述 + 英文术语（遵循 vault 惯例）

## Examples

**输入**: `/paper-digest "https://arxiv.org/abs/2603.08127"`

**预期输出**: `Papers/2603-EvoScientist.md`，包含完整的论文笔记，mermaid mindmap，以及与 vault 中已有笔记的 wikilink 关联。
```

- [ ] **Step 2: Verify skill by reading it back**

Read `skills/1-literature/paper-digest/SKILL.md` and confirm:
- Frontmatter has all required fields per `references/skill-protocol.md`
- Steps reference correct paths (`Templates/Paper.md`, `references/Tag-Taxonomy.md`, `Workbench/logs/`)
- Guard section is present and meaningful

- [ ] **Step 3: Commit**

```bash
git add skills/1-literature/paper-digest/
git commit -m "Add paper-digest skill: digest paper into structured note"
```

---

## Task 7: Skill — cross-paper-analysis

Level 1 orchestration skill: compare multiple papers to find consensus, contradictions, and gaps.

**Files:**
- Create: `skills/1-literature/cross-paper-analysis/SKILL.md`

- [ ] **Step 1: Write the SKILL.md**

```markdown
---
name: cross-paper-analysis
description: 跨论文对比分析，识别共识、矛盾和知识空白
version: 1.0.0

intent: literature
capabilities: [research-planning, cross-validation]
domain: general

roles: [autopilot, sparring, copilot]
autonomy: medium
related-skills: [paper-digest, memory-distill]

allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep

input:
  - name: papers
    description: "要对比的论文列表：[[wikilinks]] 或 tag 筛选条件（如 'tags: VLA'）"
  - name: focus
    description: "（可选）对比维度，如 'method comparison'、'scaling behavior'"
output:
  - file: "Topics/{topic}-Analysis.md"
  - memory: "Workbench/memory/patterns.md (append if patterns found)"
  - memory: "Workbench/logs/YYYY-MM-DD.md (append log entry)"
---

# Cross-Paper Analysis

## Purpose

给定一组论文笔记，从多个维度做系统对比分析。核心目标是**去伪存真**——识别领域共识（多篇论文一致）、矛盾点（论文之间冲突）、以及知识空白（无论文覆盖）。这是 MindFlow insight 发现的关键技能。

## Steps

### Step 1: 收集输入论文

- 如果输入是 wikilinks 列表 → 直接读取对应 Papers/*.md
- 如果输入是 tag 筛选 → 用 Grep 在 Papers/ 中搜索 frontmatter 中包含指定 tag 的文件
- 读取每篇论文的 Summary、Method、Key Results、Strengths & Weaknesses 部分
- 同时读取 `Topics/Domain-Map.md` 获取当前领域认知
- 读取 `Workbench/memory/patterns.md` 检查已知 pattern

### Step 2: 构建对比表

为每篇论文提取关键维度，构建 Markdown 表格：

| Paper | 问题定义 | 核心方法 | 实验设置 | 关键结果 | 局限性 |
|-------|---------|---------|---------|---------|-------|

每个单元格标注：
- [共识] — 与多数论文一致
- [矛盾] — 与某篇论文冲突
- [独特] — 仅此论文提出

### Step 3: 分析发现

从对比表中提取三类发现：

1. **共识** → 高置信度领域知识
   - 如果 Domain-Map 中尚未记录 → 建议添加到 Established Knowledge
2. **矛盾** → 需要验证的开放问题
   - 写入 `Workbench/queue/questions.md`
   - 如果 Domain-Map 中未记录此矛盾 → 建议添加到 Active Debates
3. **知识空白** → 潜在研究机会
   - 考虑是否值得生成 `Ideas/` 条目

### Step 4: 产出

- 生成 `Topics/{topic}-Analysis.md`（完整分析报告）
  - 包含：对比表、共识列表、矛盾列表、知识空白、建议
  - 所有 claim 附带 [[wikilink]] 指向具体论文
- 如果发现新 pattern → 追加到 `Workbench/memory/patterns.md`
- 追加日志到 `Workbench/logs/YYYY-MM-DD.md`:
  ```
  ### [HH:MM] cross-paper-analysis
  - **input**: [N 篇论文, focus: xxx]
  - **output**: [[Topics/xxx-Analysis.md]]
  - **observation**: [最重要的发现]
  - **status**: success
  ```

## Guard

- 不修改原始 Papers/ 笔记（只读取）
- 不修改 Domain-Map（只建议更新，由 Human 或后续 skill 执行）
- 矛盾点标注为"待验证"，不直接下结论
- 所有引用必须可追溯到具体论文的具体部分
- 笔记语言：中文描述 + 英文术语

## Examples

**输入**: `/cross-paper-analysis --tags VLA --focus "method comparison"`

**预期输出**: `Topics/VLA-Method-Comparison-Analysis.md`，包含所有 VLA 标签论文的方法对比表、识别出的共识和矛盾。
```

- [ ] **Step 2: Verify skill structure**

Read back and confirm frontmatter completeness, correct path references, guard section present.

- [ ] **Step 3: Commit**

```bash
git add skills/1-literature/cross-paper-analysis/
git commit -m "Add cross-paper-analysis skill: compare papers for consensus/contradictions"
```

---

## Task 8: Skill — memory-distill

Level 0 atomic skill: extract patterns from daily logs into memory files.

**Files:**
- Create: `skills/5-evolution/memory-distill/SKILL.md`

- [ ] **Step 1: Write the SKILL.md**

```markdown
---
name: memory-distill
description: 从工作日志中蒸馏 pattern 和 insight 到记忆库
version: 1.0.0

intent: evolution
capabilities: [research-planning, cross-validation]
domain: general

roles: [autopilot]
autonomy: high

allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep

input:
  - name: period
    description: "（可选）要蒸馏的时间范围，默认最近 7 天"
output:
  - memory: "Workbench/memory/patterns.md (append new patterns)"
  - memory: "Workbench/memory/insights.md (promote patterns to insights if qualified)"
  - memory: "Workbench/evolution/changelog.md (log changes)"
---

# Memory Distill

## Purpose

定期从 `Workbench/logs/` 的原始工作日志中提取有价值的 pattern 和 insight，写入结构化的记忆库。这是 MindFlow 记忆进化系统的基础——把「日常积累的零散观察」蒸馏为「可复用的经验」。

## Steps

### Step 1: 收集日志

- 用 Glob 获取 `Workbench/logs/YYYY-MM-DD.md` 文件列表
- 筛选指定时间范围内的日志（默认最近 7 天）
- 读取所有匹配的日志文件

### Step 2: 提取候选 pattern

扫描日志中的 observation 字段，寻找：
- **重复出现的观察**（不同日期出现类似观察 → pattern 候选）
- **意外发现**（与已有 Domain-Map 知识不一致的观察）
- **关联线索**（不同论文/实验之间的潜在联系）

### Step 3: 检查已有记忆

- 读取 `Workbench/memory/patterns.md` — 新发现是否已记录？
- 读取 `Workbench/memory/insights.md` — 是否有 provisional insight 被本次发现进一步支持？

### Step 4: 更新记忆

**新 pattern**（从未记录过）：
- 追加到 `Workbench/memory/patterns.md`，格式遵循 `references/memory-protocol.md`

**已有 pattern 获得新证据**：
- 更新 patterns.md 中对应条目的 occurrences 列表
- 如果 occurrences ≥ 3 且来自独立来源 → 晋升为 provisional insight：
  - 追加到 `Workbench/memory/insights.md`，status: provisional
  - 在 patterns.md 中标注 "→ promoted to insight"

**已有 provisional insight 获得新证据**：
- 更新 insights.md 中对应条目的 evidence 列表
- 如果 evidence 来自 ≥2 独立来源 → 标记 status: validated
- 如果 validated 且 confidence > 0.8 → 写入 `Workbench/queue/review.md` 建议晋升到 Domain-Map

### Step 5: 记录变更

追加到 `Workbench/evolution/changelog.md`:
```
### [YYYY-MM-DD] [AI] memory-distill
- **period**: [蒸馏的时间范围]
- **new_patterns**: N
- **promoted_to_insight**: N
- **validated_insights**: N
- **logs_processed**: N files
```

## Guard

- 只追加记忆，不修改或删除已有条目
- 不直接修改 Domain-Map（通过 queue/review.md 建议）
- Pattern 和 insight 必须有明确的日志来源引用
- 不捏造 pattern——观察必须实际出现在日志中

## Examples

**输入**: `/memory-distill --period "2026-03-20 ~ 2026-03-26"`

**预期输出**: patterns.md 新增 2 条 pattern，insights.md 中 1 条 provisional insight 获得新证据升级为 validated。
```

- [ ] **Step 2: Verify skill structure**

Read back and confirm frontmatter, path references, guard section.

- [ ] **Step 3: Commit**

```bash
git add skills/5-evolution/memory-distill/
git commit -m "Add memory-distill skill: extract patterns from logs into memory"
```

---

## Task 9: Demo Walkthrough

A concrete example showing how to use the 3 skills end-to-end.

**Files:**
- Create: `examples/demo-walkthrough.md`

- [ ] **Step 1: Write the walkthrough**

```markdown
# MindFlow Demo Walkthrough

> 这个文档演示如何使用 MindFlow Phase 1 的 3 个核心 skill 完成一个完整的研究循环。

## 前提条件

1. 一个 coding agent（Claude Code / Codex / Cursor 等）
2. MindFlow skills 已安装到 agent 的 skills 目录
3. 当前工作目录是你的 Obsidian vault

## Step 1: 消化论文 (paper-digest)

假设你想了解 EvoScientist 这篇论文：

```
/paper-digest "https://arxiv.org/abs/2603.08127"
```

AI 会：
1. 抓取论文内容
2. 生成 `Papers/2603-EvoScientist.md`，包含完整笔记
3. 自动关联 vault 中已有的相关论文
4. 记录到 `Workbench/logs/` 今日日志

打开 Obsidian 查看 Papers/2603-EvoScientist.md，确认笔记质量，填写 rating。

## Step 2: 重复几次

用同样方法消化几篇相关论文：

```
/paper-digest "The AI Scientist Sakana AI"
/paper-digest "https://github.com/karpathy/autoresearch"
```

## Step 3: 跨论文分析 (cross-paper-analysis)

现在有了多篇论文笔记，做对比分析：

```
/cross-paper-analysis --tags AutoResearch --focus "memory and evolution mechanisms"
```

AI 会：
1. 读取所有 AutoResearch 标签的论文
2. 构建对比表
3. 识别共识、矛盾、知识空白
4. 生成 `Topics/AutoResearch-Memory-Evolution-Analysis.md`
5. 如果发现 pattern → 写入 `Workbench/memory/patterns.md`

## Step 4: 蒸馏记忆 (memory-distill)

经过几天的论文阅读和分析后：

```
/memory-distill
```

AI 会：
1. 扫描最近 7 天的 `Workbench/logs/`
2. 提取重复出现的 pattern
3. 够格的 pattern 晋升为 provisional insight
4. 有足够证据的 insight 标记为 validated
5. 建议将高置信度 insight 添加到 `Topics/Domain-Map.md`

## 查看 AI 状态

随时可以在 Obsidian 中查看：
- `Workbench/agenda.md` — AI 的研究议程
- `Workbench/memory/` — AI 积累的经验
- `Workbench/queue/` — 待办事项
- `Workbench/logs/` — 每日工作日志
- `Topics/Domain-Map.md` — 你和 AI 的共同认知地图

所有文件都是 Markdown，你可以直接阅读和编辑。
```

- [ ] **Step 2: Commit**

```bash
git add examples/
git commit -m "Add demo walkthrough for Phase 1 skills"
```

---

## Task 10: Update Existing Files

Update CLAUDE.md to reference the new skill system, and add Idea template fields.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `Templates/Idea.md`

- [ ] **Step 1: Update CLAUDE.md**

In the "AI Workflow for Paper Notes" section, update step 4 from `Papers/AuthorYear-ShortTitle.md` to `Papers/YYMM-ShortTitle.md` (matching the convention in the repository structure section and the paper-digest skill).

Then add a new section after "AI Workflow for Paper Notes":

```markdown
## MindFlow Skill System

MindFlow 使用标准化的 Markdown skill 来自动化科研工作流。Skills 定义在 `skills/` 目录中，协议文档在 `references/` 中。

### 核心概念
- **Skills**: 定义在 `skills/<category>/<name>/SKILL.md` 中的可执行能力单元
- **Workbench/**: AI 的工作状态（agenda、memory、queue、logs），Human 可随时查看和编辑
- **Topics/Domain-Map.md**: Human-AI 共同维护的核心认知地图

### 已有 Skills
- `paper-digest`: 消化论文生成笔记（替代上述 AI Workflow）
- `cross-paper-analysis`: 跨论文对比分析
- `memory-distill`: 从日志蒸馏记忆

### 协议文档
- `references/skill-protocol.md`: Skill 编写规范
- `references/memory-protocol.md`: 记忆格式和晋升规则
- `references/agenda-protocol.md`: 研究议程管理规则
```

- [ ] **Step 2: Update Templates/Idea.md**

Add `linked_project`, `linked_experiment`, `feasibility` fields to frontmatter (per spec):

Add after `status: raw` line:
```yaml
linked_project:      # [[Projects/xxx]] 关联项目
linked_experiment:   # [[Experiments/xxx]] 关联实验
feasibility: unverified  # unverified / promising / confirmed / rejected
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md Templates/Idea.md
git commit -m "Update CLAUDE.md with skill system docs, extend Idea template"
```

---

## Task 11: Final Verification

Verify the entire Phase 1 deliverable is consistent and complete.

- [ ] **Step 1: Verify directory structure**

Run `find` to confirm all expected files exist:
```bash
ls skills/taxonomy.schema.json skills/stage-skill-map.json
ls skills/1-literature/paper-digest/SKILL.md
ls skills/1-literature/cross-paper-analysis/SKILL.md
ls skills/5-evolution/memory-distill/SKILL.md
ls references/skill-protocol.md references/memory-protocol.md references/agenda-protocol.md
ls Templates/Experiment.md Templates/Report.md Templates/Domain-Map.md
ls Workbench/agenda.md Workbench/identity.md
ls Workbench/memory/insights.md Workbench/memory/patterns.md
ls Workbench/queue/reading.md Workbench/queue/review.md
ls Topics/Domain-Map.md
ls examples/demo-walkthrough.md
```

- [ ] **Step 2: Cross-reference check**

Verify internal consistency:
- All SKILL.md files reference correct paths (Templates/, Workbench/, Resources/)
- All SKILL.md frontmatter fields match `taxonomy.schema.json` enums
- All memory format references match `references/memory-protocol.md`
- `stage-skill-map.json` lists exactly the 3 implemented skills
- `CLAUDE.md` accurately lists the 3 skills

- [ ] **Step 3: Final commit with tag**

```bash
git add -A
git commit -m "MindFlow Phase 1 complete: skeleton with 3 core skills"
```
