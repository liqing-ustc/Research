# VLN-VLA Unification Survey — Design Spec

**Date:** 2026-03-24
**Status:** Draft
**Author:** Qing Li + Claudian

## Goal

Build a literature survey in the Obsidian vault exploring the convergence of VLN (Vision-and-Language Navigation) and VLA (Vision-Language-Action) foundation models for indoor home robotics, with SLAM as a spatial memory bridge. The survey aims to identify research gaps and ultimately produce a concrete research idea.

## Perspective

**Foundation model viewpoint:** Both VLN and VLA are converging toward VLM backbone + action prediction architectures. The core question is whether and how they can be unified, and what role SLAM-based spatial representations play in enabling that unification.

## Deliverables

All output lives inside the Obsidian vault:

1. **Core Topic Note:** `Topics/VLN-VLA-Unification.md`
2. **New Paper Notes:** 8-12 papers in `Papers/`, following `AuthorYear-ShortTitle.md` naming
3. **Idea Note:** `Ideas/` (when survey matures), following `Ideas/` template

## Topic Note Structure

`Topics/VLN-VLA-Unification.md` is organized into 6 sections:

### Section 1: VLA 基础模型现状

Survey of Vision-Language-Action foundation models for robot manipulation.

- **Existing notes to link:** π0 (Black2024), π0.5 (Black2025), MEM (Torne2026), RoboClaw (Li2026)
- **Additional papers to survey:** RT-2, Octo, OpenVLA, and other representative VLA works
- **Focus:** Architecture patterns (VLM backbone + action decoder), training paradigms (co-training, flow matching), action spaces, generalization strategies

### Section 2: VLN 基础模型现状

Survey of Vision-and-Language Navigation models, especially those leveraging foundation models.

- **Key papers to survey:** NaVILA, NavGPT, VLN-DUET, VLN-CE, ETPNav, and LLM/VLM-based navigation agents
- **Focus:** How VLN is shifting from task-specific architectures to VLM-based approaches; action space differences vs. VLA (discrete waypoints vs. continuous control); sim-to-real gap

### Section 3: 语义 SLAM 与空间表示

Survey of semantic SLAM and spatial representations that can interface with foundation models.

- **Key papers to survey:** ConceptGraphs, SplaTAM, CLIP-Fields, VLMaps, OpenScene, and related 3D scene understanding works
- **Focus:** Scene representations that are both geometrically accurate (for navigation/manipulation) and semantically rich (queryable by VLMs); how SLAM can serve as shared spatial memory for both VLN and VLA

### Section 4: 架构趋同分析

Cross-cutting analysis comparing VLN and VLA architectures.

- **Comparison dimensions:**
  - VLM backbone (PaliGemma, CLIP, SigLIP, etc.)
  - Action space (discrete waypoints vs. continuous joint control vs. hybrid)
  - Scene/memory representation (history tokens, video encoding, 3D maps)
  - Training data (simulation vs. real, scale, diversity)
  - Task horizon (single-step vs. long-horizon)
- **Key question:** What are the fundamental architectural similarities and differences? Where is unification natural vs. forced?

### Section 5: 现有 Nav+Manip 系统

Survey of existing systems that combine navigation and manipulation.

- **Key papers to survey:** OK-Robot, SayCan, HomeRobot Challenge, Mobile ALOHA, TidyBot, and similar integrated systems
- **Focus:** How these systems connect navigation and manipulation modules; what are the integration bottlenecks; are they modular pipelines or end-to-end?

### Section 6: Gap 分析与潜在方向

Synthesized from Sections 1-5.

- Identified research gaps
- Potential research directions
- Open questions
- Seeds for the eventual Idea note

## Paper Notes Strategy

### When to create an independent Paper note:

- Representative work in its sub-field (e.g., NaVILA for VLN, ConceptGraphs for semantic SLAM)
- Directly relevant to the unification thesis
- Requires detailed architecture analysis

### When to only mention in Topic note:

- Earlier/background works cited for context
- Papers with similar methods where only conclusions differ

### Estimated new Paper notes: 8-12

- VLN foundation models: ~3-4 papers
- Semantic SLAM / spatial representation: ~2-3 papers
- Nav+Manip systems: ~2-3 papers

### Existing Paper notes to integrate: 4

- `[[Black2024-Pi0]]` — π0 VLA foundation model
- `[[Black2025-Pi05]]` — π0.5 open-world generalization
- `[[Torne2026-MEM]]` — MEM multi-scale memory
- `[[Li2026-RoboClaw]]` — RoboClaw agentic framework

All Paper notes follow `Templates/Paper.md` format with YAML frontmatter, mermaid mindmap, and wikilinks.

## Workflow

### Phase 1: 文献收集与 Topic Note 搭建

- Search and fetch key papers for each section
- Build Topic note skeleton with one-line summaries and key insights per paper
- Create Paper notes for representative works

### Phase 2: 分析与交叉比较

- Fill in Section 4 (架构趋同分析) with structured comparison
- Build a comparison table across dimensions listed above
- Annotate commonalities and differences between VLN and VLA

### Phase 3: Gap 分析 → Idea 提炼

- Populate Section 6 with research gaps discovered during survey
- When sufficient insight accumulates, extract into an Idea note in `Ideas/`
- Idea note links back to Topic note and relevant Paper notes via wikilinks

## Execution

- Each session focuses on one section at a time
- Claudian searches papers, generates notes; user reviews and supplements in Obsidian
- Topic note is a living document, updated incrementally

## Conventions

- Language: 中英混用 (Chinese prose, English technical terms)
- Tags: flat English tags (`VLN`, `VLA`, `SLAM`, `embodied-AI`, `navigation`, `manipulation`, `foundation-model`, `indoor-scene`)
- Links: Obsidian `[[wikilinks]]` for all cross-references
- Paper naming: `AuthorYear-ShortTitle.md`
