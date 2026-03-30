---
name: memory-distill
description: >
  当积累了多天工作日志、或 Supervisor 说"整理记忆""蒸馏"时，从日志中提取 pattern 和 insight 到记忆库。也可被 autoresearch 在合适时机自动调用
argument-hint: "[period]"
allowed-tools: Read, Write, Edit, Glob, Grep
---

## Purpose

memory-distill 是 MindFlow 记忆演化体系的基础技能。它定期扫描 `Workbench/logs/` 中的原始工作日志，从中提取跨日期重复出现的 pattern 和意外发现，并将有价值的 observation 提升为结构化的记忆条目，写入 `Workbench/memory/patterns.md` 和 `Workbench/memory/insights.md`。

该技能实现了 `references/memory-protocol.md` 中定义的 Insight Promotion Hierarchy 的底层两级跃迁：从 Level 0（Raw Log）提升至 Level 1（Pattern），再进一步触发 Level 1 → Level 2（Provisional Insight）和 Level 2 → Level 3（Validated Insight）的升级。它将"散落的日常观察"转化为"可复用的研究经验"，是 MindFlow 自我进化机制的核心入口。

输入为时间范围内的日志文件（`Workbench/logs/YYYY-MM-DD.md`），输出为更新后的记忆文件和 changelog 条目。

## Steps

### Step 1：收集日志

1. 解析 `period` 参数，确定起止日期。若未提供，默认为今天往前 7 天（含今天）。
2. 用 Glob 列出 `Workbench/logs/YYYY-MM-DD.md` 格式的所有日志文件。
3. 根据文件名中的日期过滤，保留落在 `period` 范围内的文件。
4. 用 Read 逐一读取全部匹配的日志文件，记录每个文件的日期和内容。
5. 若范围内无任何日志文件，输出提示"指定时间段内无日志，跳过蒸馏"，终止执行。

### Step 2：提取候选 Pattern

通读所有收集到的日志内容，重点扫描每条 log entry 的 `observation` 字段及其他叙述性文字，寻找以下三类候选 pattern：

1. **跨日期重复观察**：同一现象、规律或结论在两个或更多不同日期的日志中均有提及。即使措辞不同，只要语义相似，均视为同一 pattern 的多次出现。
2. **意外发现（anomaly）**：某个结果或行为与 `DomainMaps/` 中记录的已有知识相悖，或日志中明确标注为"出乎意料"、"与预期不符"的观察。
3. **关联线索（correlation clue）**：日志中提示两篇论文、两个实验或两个概念之间存在潜在联系，且该联系尚未在任何记忆文件中被明确记录。

对每个候选 pattern，记录：
- 核心 observation 的一句话概括
- 来源日志文件列表（`Workbench/logs/YYYY-MM-DD.md`）

### Step 3：检查已有记忆

1. 用 Read 读取 `Workbench/memory/patterns.md`，逐条对比 Step 2 提取的候选 pattern 与已记录条目的语义相似度。判断每个候选是：
   - **全新 pattern**：记忆库中无对应条目
   - **已有 pattern 的新证据**：与某条已有 pattern 高度相似，新日志提供了额外的 occurrence

2. 用 Read 读取 `Workbench/memory/insights.md`，检查是否存在与候选 pattern 相关的 `provisional` insight。若有，候选的新证据将用于支持该 insight 的升级。

### Step 4：更新记忆

根据 Step 3 的分类结果，分三种情况处理：

**情况 A：全新 pattern**

用 Edit 将以下格式的条目 append 到 `Workbench/memory/patterns.md` 末尾：

```markdown
### [YYYY-MM-DD] <Pattern 描述>

- **observation**: <一句话描述跨源观察到的规律或现象>
- **occurrences**: [[Workbench/logs/YYYY-MM-DD]], [[Workbench/logs/YYYY-MM-DD]], ...
- **confidence**: low
- **needs_verification**: yes
```

日期填写今天（执行蒸馏的日期）。

**情况 B：已有 pattern 获得新证据**

1. 用 Read 再次确认目标 pattern 条目的当前 `occurrences` 列表。
2. 用 Edit 在该条目的 `occurrences` 行末 append 新的日志来源链接。
3. 统计更新后的 `occurrences` 总数：
   - 若 occurrences 数量 **< 3**：仅更新 occurrences，不晋升。
   - 若 occurrences 数量 **≥ 3 且来自独立来源**（不同日期或不同论文/实验）：触发 L1 → L2 晋升。

   晋升时：
   - 用 Edit 在该 pattern 条目的 `needs_verification` 行后追加一行：`- **status**: → promoted to insight ([YYYY-MM-DD])`
   - 用 Edit 将以下格式的条目 append 到 `Workbench/memory/insights.md` 末尾：

     ```markdown
     ### [YYYY-MM-DD] <Insight 标题（与 pattern 描述一致）>

     - **claim**: <从 pattern observation 提炼的可证伪的一句话断言>
     - **evidence**: [[Workbench/logs/YYYY-MM-DD]], [[Workbench/logs/YYYY-MM-DD]], ...
     - **confidence**: low
     - **source**: cross-validation
     - **impact**: <该 insight 可能影响的研究方向，若暂不明确可填"待评估">
     - **status**: provisional
     ```

**情况 C：已有 provisional insight 获得新证据**

1. 用 Read 确认目标 insight 条目当前的 `evidence` 列表和 `confidence`。
2. 用 Edit 在该 insight 条目的 `evidence` 行末 append 新的来源链接。
3. 统计更新后独立证据来源数量：
   - 若独立来源 **< 2**：仅更新 evidence，保持 `status: provisional`。
   - 若独立来源 **≥ 2**：触发 L2 → L3 晋升，用 Edit 将 `status: provisional` 所在行改为 `status: validated`，并将 `confidence` 提升为 `medium` 或 `high`（根据证据强度判断）。

   若晋升后 `status: validated` 且 `confidence > 0.8`（即 `high`）：
   - 用 Edit 将以下条目 append 到 `Workbench/queue.md` 的 Review 部分，建议晋升至 DomainMaps：

     ```markdown
     ### [YYYY-MM-DD] 建议晋升至 DomainMaps

     - **insight**: [[Workbench/memory/insights.md#<heading>]]
     - **claim**: <insight 的 claim 原文>
     - **confidence**: high
     - **reason**: validated insight，≥2 独立来源，confidence > 0.8，符合 L3 → L4 晋升条件
     - **suggested_map**: <建议写入的 DomainMaps/{Name}.md 文件名>
     ```

### Step 5：记录变更

用 Edit 将以下格式的条目 append 到 `Workbench/evolution/changelog.md` 末尾（若文件不存在，先用 Write 创建并加一级标题 `# Evolution Changelog`）：

```markdown
### [YYYY-MM-DD] memory-distill

- **period**: <YYYY-MM-DD ~ YYYY-MM-DD>
- **logs_processed**: <数量>
- **new_patterns**: <数量>
- **promoted_to_insight**: <数量>（L1 → L2）
- **validated_insights**: <数量>（L2 → L3）
- **queued_for_review**: <数量>（L3 → L4 候选）
```

## Guard

- **仅追加，不修改**：永远不修改或删除记忆文件中的已有条目。若需更新，只能在对应条目的现有字段行末追加内容，或在条目末尾追加新字段行，不得改动原始文字。
- **不直接修改 DomainMaps**：memory-distill 无权写入 `DomainMaps/` 下的任何文件，只能通过 `Workbench/queue.md` 的 Review 部分提出建议，由 Human 或上层技能决策。
- **来源引用必须明确**：patterns.md 中每条 pattern 的 `occurrences`，以及 insights.md 中每条 insight 的 `evidence`，都必须包含指向具体日志文件的 Obsidian wikilink，不得仅凭印象记录"多次观察到"。
- **不捏造 pattern**：只有在日志中确实出现的 observation 才能被提取为候选 pattern，不得基于推断或联想凭空生成。若某规律听起来合理但日志中找不到明确依据，不记录。
- **晋升需引用具体证据**：将 pattern 晋升为 provisional insight，或将 provisional insight 标记为 validated 时，必须在 insight 的 `evidence` 字段中列出支撑该结论的所有具体日志来源。
- **独立来源的判断**：同一天的多条日志条目不算作独立来源；独立来源需来自不同日期，或来自不同论文/实验的观察。

## Verify

- [ ] `Workbench/evolution/changelog.md` 已追加本次蒸馏记录
- [ ] 蒸馏结果已记录（新增 pattern 数 + 晋升 insight 数，允许为 0 但须明确记录）

## Examples

**示例：指定时间段蒸馏**

```
/memory-distill --period "2026-03-20 ~ 2026-03-26"
```

执行过程：

1. 解析 period，确定范围为 2026-03-20 至 2026-03-26
2. Glob `Workbench/logs/` 找到 `2026-03-20.md`、`2026-03-22.md`、`2026-03-24.md`、`2026-03-26.md` 共 4 个文件，逐一读取
3. 扫描 observation 字段，发现：
   - "reward shaping 在 sparse-reward 环境中显著提升收敛速度" 在 03-22 和 03-24 均有记录 → 候选 pattern（2次出现，全新）
   - "基于 diffusion 的策略对 action chunk size 高度敏感" 在 03-20、03-22、03-26 均有记录 → 候选 pattern（3次出现，全新但已达晋升阈值）
   - "cross-attention 替代 concatenation 在多模态融合中更有效" 在 03-24 出现，且 patterns.md 已有一条相似 pattern（来自 03-15、03-18），本次为第3次出现 → 已有 pattern 获得新证据
4. 读取 patterns.md：reward shaping pattern 为全新；diffusion 敏感性 pattern 为全新但需立即晋升；cross-attention pattern 已存在，本次更新后 occurrences = 3，触发晋升
5. 读取 insights.md：无与新候选直接相关的 provisional insight
6. 写入：
   - patterns.md 新增 2 条 pattern（reward shaping、diffusion chunk size）
   - patterns.md 中 cross-attention pattern 更新 occurrences，追加晋升标记
   - insights.md 新增 1 条 provisional insight（cross-attention 融合有效性）
   - diffusion chunk size pattern 因同次蒸馏即达 3 次，直接同步写入 insights.md 为 provisional insight
7. 追加 changelog 条目

最终输出摘要：

```
memory-distill 完成（2026-03-20 ~ 2026-03-26）
- 处理日志：4 个文件
- patterns.md 新增 2 条 pattern
- insights.md 中 1 条 provisional insight 获得新证据升级为 validated（cross-attention 融合，第3次出现触发晋升后立即在 insights.md 确认）
- changelog 已更新
```
