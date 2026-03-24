# VLN-VLA Unification Survey — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a literature survey in the Obsidian vault exploring the convergence of VLN and VLA foundation models for indoor home robotics, with SLAM as a spatial memory bridge.

**Architecture:** A single Topic note (`Topics/VLN-VLA-Unification.md`) serves as the survey backbone with 6 sections. Paper notes in `Papers/` provide deep dives on representative works, linked via wikilinks. An Idea note in `Ideas/` captures the final research direction.

**Tech Stack:** Obsidian Markdown, YAML frontmatter, Mermaid diagrams, wikilinks

**Spec:** `docs/superpowers/specs/2026-03-24-vln-vla-unification-survey-design.md`

---

## File Structure

**Create:**
- `Topics/VLN-VLA-Unification.md` — Core survey note (6 sections)
- `Papers/<VLN papers>` — 3-4 new VLN paper notes
- `Papers/<VLA additions>` — 2-3 new VLA paper notes
- `Papers/<SLAM papers>` — 2-3 new semantic SLAM paper notes
- `Papers/<Nav+Manip papers>` — 2-3 new Nav+Manip system paper notes
- `Ideas/<TBD>.md` — Research idea extracted from survey (Task 8)

**Modify:**
- `Papers/Black2024-Pi0.md` — Add wikilink to Topic note in Connections section
- `Papers/Black2025-Pi05.md` — Add wikilink to Topic note in Connections section
- `Papers/Torne2026-MEM.md` — Add wikilink to Topic note in Connections section
- `Papers/Li2026-RoboClaw.md` — Add wikilink to Topic note in Connections section

---

### Task 1: Create Topic Note Skeleton

**Files:**
- Create: `Topics/VLN-VLA-Unification.md`

- [ ] **Step 1: Create the Topic note with frontmatter and all 6 section headers**

```markdown
---
title: "VLN-VLA Unification: Foundation Models for Indoor Robot Navigation and Manipulation"
tags: [VLN, VLA, SLAM, embodied-AI, foundation-model, indoor-scene, navigation, manipulation]
status: draft
date_updated: "2026-03-24"
---

## Overview
本 Topic 从 foundation model 视角，梳理 VLN（Vision-and-Language Navigation）和 VLA（Vision-Language-Action）两个领域的交汇与趋同。核心问题：VLN 和 VLA 在架构上正在趋同（都使用 VLM backbone + action prediction），能否统一？SLAM-based 空间表示在其中扮演什么角色？

## 1. VLA 基础模型现状
<!-- Survey of VLA foundation models for robot manipulation -->

## 2. VLN 基础模型现状
<!-- Survey of VLN models leveraging foundation models -->

## 3. 语义 SLAM 与空间表示
<!-- Semantic SLAM and spatial representations for VLMs -->

## 4. 架构趋同分析
<!-- Cross-cutting comparison of VLN and VLA architectures -->

## 5. 现有 Nav+Manip 系统
<!-- Systems combining navigation and manipulation -->

## 6. Gap 分析与潜在方向
<!-- Research gaps, benchmarks, and future directions -->
```

- [ ] **Step 2: Verify the file renders correctly**

Open `Topics/VLN-VLA-Unification.md` and confirm frontmatter parses and sections display.

- [ ] **Step 3: Commit**

```bash
git add Topics/VLN-VLA-Unification.md
git commit -m "feat: create Topic note skeleton for VLN-VLA unification survey"
```

---

### Task 2: Section 1 — VLA 基础模型现状

**Files:**
- Modify: `Topics/VLN-VLA-Unification.md` (Section 1)
- Modify: `Papers/Black2024-Pi0.md`, `Papers/Black2025-Pi05.md`, `Papers/Torne2026-MEM.md`, `Papers/Li2026-RoboClaw.md` (add Topic wikilink)
- Create: 2-3 new Paper notes for VLA additions (RT-2, Octo, OpenVLA)

- [ ] **Step 1: Read the 4 existing VLA paper notes**

Read `Papers/Black2024-Pi0.md`, `Papers/Black2025-Pi05.md`, `Papers/Torne2026-MEM.md`, `Papers/Li2026-RoboClaw.md` to extract key architectural details.

- [ ] **Step 2: Search for RT-2, Octo, OpenVLA, and navigation-capable VLA papers**

Use WebSearch to find these papers. For each, determine: full title, authors, year, venue, URL, key architecture details (VLM backbone, action space, training paradigm). Also explicitly search for VLA models that incorporate navigation / mobile base control (e.g., RT-2 with mobile base, or newer models that blur VLA/VLN boundaries). This cross-cutting category is especially relevant to the unification thesis.

- [ ] **Step 3: Create Paper notes for the most representative VLA additions**

Create 2-3 Paper notes following `Templates/Paper.md`. Priority order:
1. RT-2 (Google DeepMind) — VLM → robot actions paradigm
2. OpenVLA (Stanford) — open-source VLA
3. Octo (UC Berkeley) — transformer-based generalist policy

Each note must include: YAML frontmatter, Summary, Method, Key Results, Mind Map, and Connections (with `[[VLN-VLA-Unification]]` link).

- [ ] **Step 4: Add wikilinks to existing paper notes**

In each of the 4 existing paper notes, add `[[VLN-VLA-Unification]]` to the Connections section under "Related topics."

- [ ] **Step 5: Populate Section 1 of the Topic note**

Write Section 1 content in `Topics/VLN-VLA-Unification.md`:
- One-paragraph overview of VLA landscape
- Table of key VLA models (linking to Paper notes where available):

```markdown
| Model | Year | VLM Backbone | Action Space | Training | Key Innovation |
|-------|------|-------------|-------------|----------|---------------|
| [[Black2024-Pi0\|π0]] | 2024 | PaliGemma 3B | Continuous (flow matching) | Cross-embodiment | Action Expert + Flow Matching |
| ... | ... | ... | ... | ... | ... |
```

- Key takeaway: architectural patterns emerging in VLA

- [ ] **Step 6: Commit**

```bash
git add Topics/VLN-VLA-Unification.md Papers/
git commit -m "feat: populate Section 1 (VLA models) with existing + new paper notes"
```

---

### Task 3: Section 2 — VLN 基础模型现状

**Files:**
- Modify: `Topics/VLN-VLA-Unification.md` (Section 2)
- Create: 3-4 new Paper notes for VLN works

- [ ] **Step 1: Search for key VLN foundation model papers**

Use WebSearch for each: NaVILA, NavGPT, VLN-DUET, VLN-CE, ETPNav. Find: full title, authors, year, venue, URL, architecture details. Prioritize papers that use VLM/LLM backbones.

- [ ] **Step 2: Fetch and read the most important papers**

Use WebFetch on paper URLs (arXiv, project pages) to get detailed architecture and method information for the top 3-4 papers.

- [ ] **Step 3: Create Paper notes for representative VLN works**

Create 3-4 Paper notes following `Templates/Paper.md`. Priority:
1. NaVILA — VLM-based navigation (most relevant to unification thesis)
2. NavGPT — LLM as navigation reasoning engine
3. VLN-DUET or ETPNav — strong task-specific baselines for comparison
4. (Optional) One additional recent VLN work using foundation models

Each must include Connections linking to `[[VLN-VLA-Unification]]` and relevant VLA papers.

- [ ] **Step 4: Populate Section 2 of the Topic note**

Write Section 2 in `Topics/VLN-VLA-Unification.md`:
- Overview of VLN evolution: task-specific → VLM-based
- Table of key VLN models (same column structure as Section 1 where applicable)
- Key differences from VLA: action space (discrete waypoints vs. continuous), primary environment (simulation vs. real), evaluation benchmarks (R2R, REVERIE, etc.)
- Note on sim-to-real gap

- [ ] **Step 5: Commit**

```bash
git add Topics/VLN-VLA-Unification.md Papers/
git commit -m "feat: populate Section 2 (VLN models) with new paper notes"
```

---

### Task 4: Section 3 — 语义 SLAM 与空间表示

**Files:**
- Modify: `Topics/VLN-VLA-Unification.md` (Section 3)
- Create: 2-3 new Paper notes for semantic SLAM / spatial representation works

- [ ] **Step 1: Search for key semantic SLAM papers**

Use WebSearch for: ConceptGraphs, SplaTAM, CLIP-Fields, VLMaps, OpenScene. Find: full title, authors, year, venue, URL, what spatial representation they produce and how it interfaces with language.

- [ ] **Step 2: Fetch and read the most important papers**

Use WebFetch to get detailed method information for the top 2-3 papers. Focus on: representation format (3D scene graph, neural field, point cloud, etc.), how language/semantics are integrated, downstream use cases.

- [ ] **Step 3: Create Paper notes for representative works**

Create 2-3 Paper notes following `Templates/Paper.md`. Priority:
1. ConceptGraphs — 3D scene graphs with open-vocabulary semantics
2. VLMaps — spatial map queryable by language
3. (Optional) SplaTAM or CLIP-Fields — neural implicit representations

Each must link to `[[VLN-VLA-Unification]]` in Connections.

- [ ] **Step 4: Populate Section 3 of the Topic note**

Write Section 3 in `Topics/VLN-VLA-Unification.md`:
- Overview: why spatial representations matter for unifying navigation and manipulation
- Table of methods: representation type, semantic grounding method, downstream tasks supported
- Analysis: which representations can serve both VLN (navigation planning) and VLA (manipulation grounding)?
- SLAM as "spatial memory": how continuous mapping enables long-horizon tasks

- [ ] **Step 5: Commit**

```bash
git add Topics/VLN-VLA-Unification.md Papers/
git commit -m "feat: populate Section 3 (semantic SLAM) with new paper notes"
```

---

### Task 5: Section 5 — 现有 Nav+Manip 系统

**Files:**
- Modify: `Topics/VLN-VLA-Unification.md` (Section 5)
- Create: 2-3 new Paper notes for Nav+Manip systems

Note: Section 5 is done before Section 4 because the cross-cutting analysis in Section 4 draws on all survey sections.

- [ ] **Step 1: Search for key Nav+Manip system papers**

Use WebSearch for: OK-Robot, SayCan, HomeRobot Challenge, Mobile ALOHA, TidyBot. Find: full title, authors, year, venue, URL, system architecture (how navigation and manipulation are connected).

- [ ] **Step 2: Fetch and read the most important papers**

Use WebFetch on top 2-3 papers. Focus on: system pipeline (modular vs. end-to-end), how navigation module hands off to manipulation, role of scene understanding/SLAM, task success rates.

- [ ] **Step 3: Create Paper notes for representative systems**

Create 2-3 Paper notes following `Templates/Paper.md`. Priority:
1. OK-Robot — open-knowledge robot for nav+manip in homes
2. Mobile ALOHA — mobile manipulation with whole-body control
3. SayCan or TidyBot — LLM-grounded task planning with physical execution

Each must link to `[[VLN-VLA-Unification]]` and relevant VLA/VLN papers in Connections.

- [ ] **Step 4: Populate Section 5 of the Topic note**

Write Section 5 in `Topics/VLN-VLA-Unification.md`:
- Overview: current approaches to combining nav and manip
- System architecture comparison: modular pipeline vs. end-to-end
- Common bottlenecks: handoff between nav/manip, lack of shared spatial representation, separate training
- Analysis: what would an ideal unified system look like?

- [ ] **Step 5: Commit**

```bash
git add Topics/VLN-VLA-Unification.md Papers/
git commit -m "feat: populate Section 5 (Nav+Manip systems) with new paper notes"
```

---

### Task 6: Section 4 — 架构趋同分析

**Files:**
- Modify: `Topics/VLN-VLA-Unification.md` (Section 4)

No new Paper notes — this section synthesizes from Sections 1-3 and 5.

- [ ] **Step 1: Re-read all Paper notes created so far**

Read all Paper notes in `Papers/` to extract comparison data for the table.

- [ ] **Step 2: Build the comparison table**

Create a comprehensive Markdown table in Section 4:

```markdown
| Model | Domain | VLM Backbone | Action Space | Scene/Memory | Training Data | Paradigm | Horizon | Sim/Real |
|-------|--------|-------------|-------------|-------------|--------------|---------|---------|----------|
| π0 | VLA | PaliGemma 3B | Continuous | ... | ... | ... | ... | Real |
| NaVILA | VLN | ... | Discrete | ... | ... | ... | ... | Sim→Real |
| ... |
```

- [ ] **Step 3: Write the convergence analysis**

Below the table, write analysis covering:
- **共性 (Commonalities):** VLM backbone, instruction-conditioned action prediction, pre-training on web data
- **差异 (Differences):** action space granularity, environment representation, sim vs. real training
- **趋同点 (Convergence points):** where unification is natural (e.g., shared VLM backbone, language-conditioned planning)
- **分歧点 (Divergence points):** where unification is challenging (e.g., action space mismatch, different control frequencies)
- **SLAM 的角色:** how SLAM-based spatial memory could bridge the gap
- **Simulation 与 sim-to-real:** how simulation environments (Habitat, AI2-THOR, Matterport3D) affect both domains

- [ ] **Step 4: Commit**

```bash
git add Topics/VLN-VLA-Unification.md
git commit -m "feat: populate Section 4 (architecture convergence analysis)"
```

---

### Task 7: Section 6 — Gap 分析与潜在方向

**Files:**
- Modify: `Topics/VLN-VLA-Unification.md` (Section 6)

- [ ] **Step 1: Synthesize gaps from Sections 1-5**

Re-read the completed Topic note sections. Identify:
- What problems remain unsolved?
- What combinations haven't been tried?
- Where do existing systems fail?

- [ ] **Step 2: Search for relevant benchmarks**

Use WebSearch for ALFRED, TEACh, HomeRobot Challenge benchmarks. Note: what tasks they evaluate, what capabilities they require (nav, manip, or both), and current SOTA performance.

- [ ] **Step 3: Write Section 6**

Populate Section 6 with:
- **Research Gaps:** Numbered list of identified gaps (e.g., "No end-to-end model handles both continuous navigation and dexterous manipulation", "SLAM representations are not optimized for VLM consumption")
- **Benchmarks:** Brief overview of ALFRED, TEACh, HomeRobot and what they test
- **潜在方向:** 2-3 concrete research directions, each with a one-paragraph description
- **Open Questions:** Unresolved questions that need further investigation
- **Idea Seeds:** Mark the most promising direction(s) for extraction into an Idea note

- [ ] **Step 4: Revise the Topic note Overview**

Now that all 6 sections are complete, update the Overview paragraph at the top of `Topics/VLN-VLA-Unification.md` to serve as an executive summary reflecting the actual survey findings, key conclusions, and the most promising research directions.

- [ ] **Step 5: Verify paper count**

Count all new Paper notes created across Tasks 2-5. Target: 10-15. If below 10, identify which sections could benefit from additional papers and note for follow-up.

- [ ] **Step 6: Commit**

```bash
git add Topics/VLN-VLA-Unification.md
git commit -m "feat: populate Section 6 (gap analysis and future directions), revise Overview"
```

---

### Task 8: Extract Idea Note

**Files:**
- Create: `Ideas/<TBD-based-on-idea>.md`
- Modify: `Topics/VLN-VLA-Unification.md` (add link to Idea note)

This task is conditional — only proceed if the gap analysis in Task 7 produces a sufficiently concrete research direction. If not, mark this as deferred and discuss with user.

- [ ] **Step 1: Identify the strongest idea seed from Section 6**

Select the most promising research direction based on: novelty, feasibility, alignment with user's VLA expertise.

- [ ] **Step 2: Create the Idea note**

Create `Ideas/<descriptive-name>.md` following `Templates/Idea.md`:

```markdown
---
title: "<Idea title>"
tags: [VLN, VLA, SLAM, <additional relevant tags>]
status: raw
date_created: "2026-03-24"
---

## Core Idea
一句话描述。

## Motivation
从 survey 中发现的 gap 和机会。Link to [[VLN-VLA-Unification]].

## Related Work
- [[Paper1]] — ...
- [[Paper2]] — ...
(wikilinks to the most relevant Paper notes)

## Rough Plan
初步思路。

## Open Questions
待解决的问题。
```

- [ ] **Step 3: Add Idea link to Topic note**

Add a link to the new Idea note at the end of Section 6 in `Topics/VLN-VLA-Unification.md`.

- [ ] **Step 4: Commit**

```bash
git add Ideas/ Topics/VLN-VLA-Unification.md
git commit -m "feat: extract research idea from VLN-VLA unification survey"
```
