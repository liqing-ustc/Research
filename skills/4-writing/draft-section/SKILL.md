---
name: draft-section
description: >
  当 Supervisor 说"写一下 introduction""起草 related work"，
  或 autoresearch 判断某 direction 已积累足够素材需要成文时，
  起草论文或报告的指定章节
argument-hint: "<target> <section> <sources>"
allowed-tools: Read, Write, Edit, Glob, Grep
---

## Purpose

draft-section 是 MindFlow 的基础写作技能。给定目标文件、章节名和素材引用列表，它从 vault 中读取所有相关笔记，按照学术写作规范起草指定章节，并写入目标文件。

该技能适用于两种工作模式：autopilot（直接写入文件）和 copilot（生成草稿后 Human 确认再写入）。它是 autoresearch 高阶工作流的写作子技能，也可由 Supervisor 直接触发用于快速成文。

所有 claim 必须有 `[[wikilink]]` 来源，正文用中文撰写，英文技术术语保持英文。

## Steps

### Step 1：读取素材

根据 `sources` 参数读取所有引用笔记：

- 用 Glob 确认文件存在，再用 Read 逐一读取每个引用文件的完整内容。
- 涵盖范围：`[[Papers/...]]`（论文笔记）、`[[Experiments/...]]`（实验记录）、`[[Ideas/...]]`（研究想法）、`[[Topics/...]]`（跨论文分析）。
- 对于每份素材，重点提取：核心论点、关键数据、方法细节、引用文献（如 frontmatter 中的 `title`/`authors`/`venue`）。

若某个 `[[wikilink]]` 指向的文件不存在，在执行日志中记录"未找到：{path}"，跳过该文件，继续处理其余素材。

### Step 2：读取目标文件上下文（若已存在）

若 `target` 文件已存在：

1. 用 Read 读取完整内容，了解：
   - 已有章节结构（标题层级）
   - 已有内容风格（语言风格、术语习惯、引用格式）
   - `section` 参数指定的章节是否已有占位符或草稿
2. 确定插入位置：若目标章节已有标题但无内容，插入标题之后；若无对应标题，在文件末尾追加新章节。

若 `target` 不存在，则计划在 Step 5 用 Write 创建新文件。

### Step 3：读取 DomainMaps

用 Read 读取 `DomainMaps/_index.md`，找到与 `section` 和 `sources` 相关的 domain，再读取对应的 `DomainMaps/{Name}.md`。

重点关注：
- **Established Knowledge**：用于在 Related Work / Introduction 中准确描述领域背景，避免描述过时或已被推翻的"共识"。
- **Active Debates**：用于在 Introduction / Discussion 中定位本工作的贡献，或在 Related Work 中点出未解决问题。
- **Open Questions**：可作为 Introduction 的 motivation 依据。

若无明显相关 domain，跳过此步骤。

### Step 4：起草章节

综合 Step 1-3 收集的所有信息，按如下规范起草指定章节：

**语言规范**：
- 正文用中文撰写；模型名、方法名、benchmark 名、论文标题等技术术语保持英文，不做翻译。
- 学术写作风格：逻辑严谨，表述简洁，避免口语化。

**结构规范**（根据 `section` 类型参考以下框架，不必机械套用）：

| Section | 典型结构 |
|:--------|:---------|
| Introduction | 背景 → 问题陈述 → 现有方法局限 → 本文贡献 → 章节组织 |
| Related Work | 按主题分组 → 每组综述 → 与本文关系 |
| Method | 整体框架 → 核心模块 → 关键设计选择 |
| Experiments | 设置 → 指标 → 结果分析 → 消融实验 |
| Discussion | 主要发现 → 局限性 → 未来方向 |

**引用规范**：
- 每个 claim（事实性陈述、数据引用、方法描述）必须附 `[[wikilink]]`，指向对应的 Papers 或 Topics 笔记。
- 引用格式示例：`…如 Diffusion Policy [[2303-DiffusionPolicy]] 所示…`
- 不捏造任何引用：若某 claim 无法在素材中找到来源，明确标注 `[需引用]` 而非编造。

**禁止占位符**：起草内容必须基于素材完整成文，不得留下 `[TODO]`、`[待补充]` 等占位符。

### Step 5：写入目标文件

根据 `target` 是否已存在选择操作：

**若 `target` 不存在**：
用 Write 创建新文件，文件内容为：
```markdown
---
date: YYYY-MM-DD
title: <从 target 路径推断的标题>
tags: []
---

## {section}

<起草的章节内容>
```

**若 `target` 已存在**：
用 Edit 将起草内容插入目标位置（Step 2 确定的位置）：
- 若目标章节标题已存在但内容为空，在标题之后插入。
- 若目标章节标题不存在，在文件末尾追加：
  ```markdown

  ## {section}

  <起草的章节内容>
  ```
- 只操作指定章节，不修改文件中的其他任何内容。

### Step 6：追加日志

用 Edit（若文件不存在则用 Write）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

```markdown
### [HH:MM] draft-section
- **input**: target: {target} | section: {section} | sources: {sources 列表}
- **output**: [[{target}]] § {section}
- **word_count**: 约 N 字
- **missing_sources**: <未找到的 wikilinks，若无则填"无">
- **status**: success
```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] 指定章节已写入 `target` 且正文 >200 字
- [ ] 每个事实性陈述附有 `[[wikilink]]` 来源
- [ ] 不含 `[TODO]`、`[待补充]` 等占位符
- [ ] 未修改 `target` 中其他已有章节的内容
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`

## Guard

- **不修改其他章节**：Edit 操作只针对指定的 `section`，严禁修改目标文件中其他章节的任何内容。如不确定插入边界，停止并告知 Human。
- **不捏造引用**：所有 `[[wikilink]]` 必须指向 vault 中实际存在的文件。若 claim 无来源，标注 `[需引用]`，不得编造笔记文件名或论文信息。
- **copilot 模式先预览**：若以 copilot 模式调用，生成章节草稿后先完整输出给 Human 预览，明确确认后再执行 Write/Edit；日志同样在确认后追加。
- **语言规范**：正文用中文撰写；技术术语（模型名、方法名、benchmark 名）保持英文，不做翻译；不使用非正式口语。
- **不覆盖完整章节**：若目标章节已有实质性内容（>100 字），不直接覆盖，停止执行并告知 Human，建议以 writing-refine 技能进行修订。

## Examples

**示例 1：起草 Related Work 章节**

```
/draft-section
  --target "Reports/VLN-Survey-Draft.md"
  --section "Related Work"
  --sources "[[Papers/2301-EAI-Survey]], [[Papers/2410-NavGPT]], [[Papers/2501-MapNav]], [[Topics/VLN-MethodComparison-Analysis]]"
```

执行过程：

1. Glob + Read 读取四份素材：
   - `Papers/2301-EAI-Survey.md`：了解 Embodied AI 整体背景和近年进展
   - `Papers/2410-NavGPT.md`：了解 LLM-based navigation 方法
   - `Papers/2501-MapNav.md`：了解 map-augmented navigation 方法
   - `Topics/VLN-MethodComparison-Analysis.md`：读取已有跨论文分析，获取共识与争议
2. Read `Reports/VLN-Survey-Draft.md`，确认 Related Work 节已有标题但无内容，确定插入位置
3. Read `DomainMaps/_index.md` → `DomainMaps/VLN.md`，了解 Established Knowledge 和 Active Debates
4. 起草 Related Work，分三个子主题组织：
   - **经典 VLN 方法**：基于 Seq2Seq 和 cross-modal attention 的早期工作 [[2301-EAI-Survey]]
   - **LLM-based Navigation**：利用大语言模型作为 zero-shot 规划器的新范式 [[2410-NavGPT]]
   - **Map-augmented 方法**：结合拓扑地图或语义地图提升长程导航能力 [[2501-MapNav]]
   - 每组末尾说明与本综述的关系，点出 [[Topics/VLN-MethodComparison-Analysis]] 揭示的知识空白
5. Edit `Reports/VLN-Survey-Draft.md`，将起草内容插入 `## Related Work` 标题之后；不触碰其他章节
6. 追加日志到 `Workbench/logs/2026-03-28.md`

输出：`Reports/VLN-Survey-Draft.md` 中的 Related Work 节已完成，约 600 字，所有 claim 附 `[[wikilink]]`。
