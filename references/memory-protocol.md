# Memory Protocol

This document defines how MindFlow stores, organizes, and promotes research knowledge. The memory system transforms raw observations — logs, paper readings, experiment results — into structured, reusable insights that inform future research decisions.

---

## Directory Structure

All persistent memory lives under `Workbench/memory/` as plain Markdown files. This keeps memory inside the Obsidian vault, making it searchable, linkable, and human-readable.

```
Workbench/
  memory/
    insights.md            # Discovered insights (provisional → validated)
    failed-directions.md   # Abandoned research directions and the reasons why
    effective-methods.md   # Proven experiment and analysis strategies
    patterns.md            # Cross-paper or cross-experiment patterns (not yet validated)
  logs/                    # Raw session logs (Level 0 input to memory)
  evolution/
    changelog.md           # Record of every Domain-Map update
  queue/
    review.md              # Pending decisions requiring human approval
```

### Memory Files

| File | Purpose |
|---|---|
| `insights.md` | Curated claims with evidence and confidence tracking; entries move from `provisional` to `validated` over time |
| `failed-directions.md` | Documents what was tried and why it was abandoned; prevents re-exploring dead ends |
| `effective-methods.md` | Captures strategies that worked, with context and caveats; accelerates future experiment design |
| `patterns.md` | Early-stage observations that recur across sources; feeds the promotion pipeline but not yet validated |

---

## Entry Formats

Each file uses a consistent heading-and-bullet format. Entries are **append-only** — never edit past entries; add a new entry if an update is needed.

### insights.md

```markdown
### [YYYY-MM-DD] Insight title

- **claim**: One-sentence statement of the insight
- **evidence**: [[Papers/xxx]], [[Experiments/xxx]]
- **confidence**: high / medium / low
- **source**: literature-analysis / experiment / cross-validation
- **impact**: Which research directions are affected by this insight
- **status**: provisional / validated / integrated
```

Field notes:
- `claim` must be falsifiable and specific — avoid vague generalizations.
- `evidence` uses Obsidian wikilinks to the exact note(s) that support the claim.
- `confidence` reflects current certainty; it may increase as more evidence accumulates.
- `source` indicates how the insight was discovered.
- `impact` is free text describing which directions in `agenda.md` this insight affects.
- `status` follows the promotion hierarchy (see below).

### failed-directions.md

```markdown
### [YYYY-MM-DD] Direction name

- **original_hypothesis**: What was believed when this direction was started
- **evidence_against**: [[Papers/xxx]], [[Experiments/xxx]]
- **lesson**: One-sentence takeaway that generalizes from this failure
- **related_directions**: Which similar directions should be approached with caution
```

Field notes:
- `original_hypothesis` preserves the reasoning at the time, not a post-hoc rationalization.
- `lesson` should be actionable — what would a future researcher do differently?
- `related_directions` helps the AI avoid recommending structurally similar dead ends.

### effective-methods.md

```markdown
### [YYYY-MM-DD] Method name

- **context**: When this method is effective (conditions, dataset types, model scales, etc.)
- **method**: What to do — step-by-step if needed
- **evidence**: [[Experiments/xxx]]
- **pitfalls**: What to watch out for; known failure modes
```

Field notes:
- `context` is critical — a method that works in one setting may fail in another.
- `pitfalls` should reflect actual encountered problems, not theoretical concerns.

### patterns.md

```markdown
### [YYYY-MM-DD] Pattern description

- **observation**: What was noticed across sources
- **occurrences**: [[Papers/xxx]], [[Papers/yyy]], ...
- **confidence**: low / medium
- **needs_verification**: yes / no
```

Field notes:
- Patterns here are not yet validated insights. They are hypotheses about regularities.
- `confidence` is capped at `medium` — a pattern that clears full validation belongs in `insights.md`.
- `needs_verification` signals whether human review or additional experiments are needed before promotion.

---

## Insight Promotion Hierarchy

Knowledge is promoted upward through five levels as evidence accumulates. Promotion is never automatic below Level 3; each level has a defined trigger.

```
Level 4  ┌─────────────────────────────────┐
         │  Domain Map                     │  Domain-Map/_index.md + Domain-Map/{Name}.md
         │  Stable, integrated knowledge   │
         └────────────────┬────────────────┘
                          │ AI promotes when it judges evidence sufficient
Level 3  ┌────────────────▼────────────────┐
         │  Validated Insight              │  insights.md, status: validated
         │  ≥2 independent sources         │
         └────────────────┬────────────────┘
                          │ ≥2 independent sources confirm the claim
Level 2  ┌────────────────▼────────────────┐
         │  Provisional Insight            │  insights.md, status: provisional
         │  First formulation of claim     │
         └────────────────┬────────────────┘
                          │ Pattern observed ≥3 times independently
Level 1  ┌────────────────▼────────────────┐
         │  Pattern                        │  patterns.md
         │  Recurring observation          │
         └────────────────┬────────────────┘
                          │ memory-distill extracts from logs
Level 0  ┌────────────────▼────────────────┐
         │  Raw Log                        │  Workbench/logs/
         │  Unstructured session output    │
         └─────────────────────────────────┘
```

### Promotion Rules

| Transition | Trigger | Who |
|---|---|---|
| L0 → L1 | `memory-distill` skill processes session logs and extracts recurring observations | AI (via skill) |
| L1 → L2 | Pattern appears in ≥3 independent sources (different papers, experiments, or sessions) | AI (via skill) |
| L2 → L3 | Provisional insight is supported by ≥2 independent evidence sources | AI (via skill) |
| L3 → L4 | AI judges evidence sufficient for Domain Map integration → promotes directly. No numeric threshold — AI uses judgment. Logged to `Workbench/evolution/changelog.md` | AI |

---

## Update Rules

1. **Append-only**: Never edit or delete existing entries in memory files. To supersede an entry, append a new one with an updated date and reference the old entry if needed.

2. **Domain-Map logging**: Every time an entry is added to any `Domain-Map/{Name}.md` file (at Level 4), a corresponding log entry must be written to `Workbench/evolution/changelog.md`. Format:

   ```markdown
   ### [YYYY-MM-DD] Domain Map updated: <map name>
   - **added**: <insight title>
   - **source**: [[Workbench/memory/insights.md#<heading>]]
   - **promoted_by**: <skill name or "human">
   ```

3. **Human writes**: Humans may write directly to any memory file at any time without restrictions. Human entries are treated as authoritative and do not require evidence thresholds to be promoted manually.

4. **Conflict handling**: If a new insight contradicts an existing validated one, create a new `provisional` entry noting the contradiction. Do not modify the old entry. Flag the pair in `Workbench/queue/review.md` for human resolution.
