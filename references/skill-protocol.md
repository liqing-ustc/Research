# Skill Protocol

This document defines the SKILL.md format used by MindFlow skills. Every skill — whether atomic or orchestrating — must include a SKILL.md at its root. This file serves as the single source of truth for what a skill does, how it behaves, what it requires, and what it produces.

---

## Frontmatter Specification

Each SKILL.md begins with YAML frontmatter enclosed by `---` delimiters. MindFlow frontmatter 遵循 [Claude Code 官方 skill 规范](https://code.claude.com/docs/en/skills)，仅使用框架实际解析的字段。

### Fields

| Field | Required | Type | Description |
|---|---|---|---|
| `name` | Yes | string (kebab-case) | Unique identifier, must match directory name. Lowercase letters, numbers, hyphens only (max 64 chars). |
| `description` | Yes | string | Pushy trigger description (see below). Front-load key use case; truncated at 250 chars in skill listing. |
| `allowed-tools` | Yes | string (comma-separated) | Claude Code tools this skill may invoke, e.g. `Read, Write, Glob` |
| `argument-hint` | Recommended | string | Hint shown during autocomplete, e.g. `[source] [constraints]`. Use `<required>` and `[optional]` notation. |
| `disable-model-invocation` | No | boolean | Set `true` to prevent Claude from auto-triggering. Use for high-impact skills (e.g. agenda-evolve). Default: `false`. |
| `user-invocable` | No | boolean | Set `false` to hide from `/` menu. Use for internal-only skills (e.g. memory-retrieve). Default: `true`. |

**`description` — Pushy Trigger Principle:**

Description 应积极触发（"pushy"）——明确描述**触发场景**，降低 under-triggering 风险。不要只写"做什么"，要写"什么情况下该用我"。

| Bad | Good |
|-----|------|
| `消化一篇论文，生成结构化笔记` | `当 Supervisor 给出论文 URL/标题/PDF/DOI，或阅读队列中有待处理论文时，消化论文并生成结构化笔记到 Papers/` |

**`allowed-tools` values** (Claude Code tool names):

`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, `WebSearch`, `WebFetch`

### Example Frontmatter

```yaml
---
name: paper-digest
description: >
  当 Supervisor 给出论文 URL/标题/PDF/DOI，或阅读队列中有待处理论文时，消化论文并生成结构化笔记到 Papers/
argument-hint: "[arXiv URL / PDF path / title / DOI]"
allowed-tools: Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---
```

## Body Sections

Every SKILL.md body must include the following sections. Sections are written as Markdown headings (`##` or `###`).

| Section | Required | Purpose |
|---|---|---|
| `## Purpose` | Yes | What problem this skill solves and why it exists |
| `## Steps` | Yes | Numbered, step-by-step execution instructions for the Researcher |
| `## Guard` | Yes | Preconditions, invariants, and prohibited actions (during execution) |
| `## Verify` | Recommended | Post-execution quality checklist — mechanical checks on output quality |
| `## Examples` | Recommended | Concrete input/output examples or usage scenarios |

### Purpose

Explain the skill's goal in 2-5 sentences. Include:
- The research workflow it fits into
- What inputs it consumes
- What outputs or effects it produces

### Steps

A numbered list of executable instructions. Each step should be unambiguous enough for the Researcher to follow without clarification. Use sub-steps where needed. Reference other skills using their kebab-case names.

```markdown
## Steps

1. Receive `paper_url` or `paper_title` from the user.
2. If `paper_url` is provided, fetch the page with WebFetch.
3. Extract title, authors, abstract, and publication date.
4. Look up any existing note in `Papers/` using Glob to avoid duplicates.
5. Populate the `Templates/Paper.md` template with extracted metadata.
6. Save the completed note to `Papers/YYMM-ShortTitle.md`.
7. Append a provisional insight entry to `Workbench/memory/insights.md` if a novel claim is found.
```

### Guard

A bulleted list of rules the skill must never violate. These act as hard constraints during execution.

```markdown
## Guard

- Never overwrite an existing paper note without explicit user confirmation.
- Never modify `agenda.md` Mission section.
- Do not mark an insight as `validated` without ≥2 independent evidence sources.
- Copilot mode: produce a draft only — output to conversation for Supervisor review before writing files.
```

### Verify

A checklist of mechanical assertions to run after execution completes. Each item must be objectively verifiable (no subjective judgments).

**Verify vs Guard**: Guard constrains behavior _during_ execution ("don't do X"). Verify checks output quality _after_ execution ("did the output meet the bar?").

```markdown
## Verify

- [ ] Output file exists and is non-empty
- [ ] Frontmatter required fields are all populated
- [ ] No `[TODO]` or `[TBD]` placeholders remain
- [ ] All paper references use `[[wikilink]]` format
```

### Examples

Optional but strongly recommended. Show representative invocations and expected outputs. Use fenced code blocks or Markdown tables.

---

## Complex Skills: The `references/` Subdirectory Pattern

When a skill's SKILL.md exceeds approximately 200 lines, or when it requires supporting reference material (schemas, lookup tables, extended examples), split the content using a `references/` subdirectory inside the skill directory.

**Structure:**

```
skills/
  paper-ingest/
    SKILL.md              # Frontmatter + concise Steps + Guard
    references/
      field-mapping.md    # Detailed mapping of paper fields to frontmatter keys
      venue-list.md       # Canonical venue abbreviations
      example-note.md     # Full worked example of an output note
```

**Rules for the `references/` pattern:**

- SKILL.md remains the authoritative entry point. It must be self-contained enough to execute; `references/` files are supplementary detail.
- Reference files are linked from SKILL.md using relative Markdown links, e.g. `[field mapping](references/field-mapping.md)`.
- Reference files do not have their own frontmatter — they are plain Markdown.
- The `references/` directory may not contain nested skill directories (no sub-skills here; use orchestration instead).

---

## Skill Levels

Skills are organized into three levels based on scope and composition.

### Level 0 — Atomic

A single-purpose skill that calls Claude Code tools directly. It does not invoke other skills.

- Typically < 150 lines
- Lives in a leaf category directory (e.g. `1-literature/`, `2-ideation/`)
- Example: `paper-digest`, `idea-evaluate`, `experiment-track`

### Level 1 — Orchestration

A skill that sequences or conditions other skills to accomplish a compound goal. It calls Level 0 skills by name.

- Lives in `6-orchestration/` or similar
- Steps reference other skills explicitly: "Run `paper-digest` for each URL."
- Example: `literature-survey`, `autoresearch`

### Level 2 — Global

A skill that operates across the entire vault or research state. It may invoke multiple Level 1 skills, modify `agenda.md`, update memory files, or trigger cross-cutting evolution.

- Typically run on a schedule or by explicit Supervisor command
- Broad `allowed-tools` and strict `## Guard` section
- Example: `memory-distill`, `agenda-evolve`

---

## Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Skill directory name | kebab-case | `paper-digest/` |
| Category directory name | numbered prefix + kebab-case | `1-literature/` |
| Skill entrypoint filename | Always uppercase | `SKILL.md` |
| `name` field in frontmatter | Must match directory name exactly | `paper-digest` |

**Rationale for numbered category prefixes:** Obsidian and most file explorers sort directories alphabetically. Numbered prefixes (`01-`, `02-`, ...) enforce a logical reading order that mirrors the research workflow (literature → ideation → experiment → analysis → writing → evolution).

**Example directory layout:**

```
skills/
  1-literature/
    paper-digest/
      SKILL.md
    cross-paper-analysis/
      SKILL.md
    literature-survey/
      SKILL.md
  2-ideation/
    idea-generate/
      SKILL.md
    idea-evaluate/
      SKILL.md
  3-experiment/
    experiment-design/
      SKILL.md
    experiment-track/
      SKILL.md
    result-analysis/
      SKILL.md
  4-writing/
    draft-section/
      SKILL.md
    writing-refine/
      SKILL.md
  5-evolution/
    memory-distill/
      SKILL.md
    agenda-evolve/
      SKILL.md
    memory-retrieve/
      SKILL.md
  6-orchestration/
    autoresearch/
      SKILL.md
      references/
        judge-heuristics.md
```
