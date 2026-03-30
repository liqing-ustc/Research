---
name: agenda-evolve
description: >
  当积累了新的 validated insight、实验结果改变了方向判断、
  或 Supervisor 说"更新研究方向""复盘 agenda"时，
  根据记忆和发现演化研究议程
argument-hint: "[trigger]"
allowed-tools: Read, Edit, Glob, Grep
---

## Purpose

agenda-evolve 是 MindFlow 研究议程的演化引擎。它定期审视 `Workbench/` 中积累的 validated insight、实验结果和 idea 评估，判断现有研究方向是否需要调整——新增方向、更新方向状态、暂停低优先级方向、或废弃已被证伪的方向——并相应地更新 `Workbench/agenda.md`。

该技能实现了 `references/agenda-protocol.md` 中定义的 Researcher 完全自主的议程演化权限。它将"分散在记忆和队列中的证据积累"转化为"研究方向层面的有据决策"，是 MindFlow 自我进化机制中连接 insight 与 strategy 的关键环节。

触发时机：每次 memory-distill 产出新的 validated insight 之后；收到 Supervisor 的方向性反馈之后；实验结果与当前 hypothesis 出现显著偏差时；或定期复盘（建议每两周一次）。

## Steps

### Step 1：读取当前 Agenda

用 Read 读取 `Workbench/agenda.md` 的完整内容，记录：

- 所有 Active Directions 的名称、priority、status、hypothesis、evidence、next_action、confidence
- 所有 Paused Directions 的名称及 resume_condition
- 所有 Abandoned Directions 的名称（用于避免重复引入已被废弃的方向）
- Mission 节的完整文本（后续步骤中不得修改 Mission）

若文件不存在，用 Edit 按照 `references/agenda-protocol.md` 的模板创建初始 agenda.md，填写 Mission 留空占位符，Active/Paused/Abandoned Directions 留空，frontmatter 中 `last_updated` 填今天日期，`updated_by: agenda-evolve`。然后继续执行后续步骤。

### Step 2：读取近期 Validated Insights

用 Read 读取 `Workbench/memory/insights.md`，筛选出满足以下任一条件的 insight 条目：

1. `status: validated`，且条目日期在今天往前 30 天以内
2. `status: validated`，且 `confidence: high`（即使超过 30 天，高置信度 insight 仍有价值）

对每条符合条件的 insight，记录：
- 标题和 `claim` 字段内容
- `impact` 字段（该 insight 涉及哪个研究方向）
- `confidence` 和 `evidence` 来源

若 `insights.md` 不存在或内容为空，记录"无可用 validated insight"，继续执行后续步骤（agenda 仍可基于其他输入演化）。

### Step 3：读取 Failed Directions

用 Read 读取 `Workbench/memory/failed-directions.md`（若存在）。记录其中每条已废弃方向的核心 reason 和 lesson，用于：

- 防止在后续步骤中重新引入已被证明无效的方向
- 为新增方向提供"前车之鉴"参照

若文件不存在，跳过本步骤。

### Step 4：读取 Queue 中的 Questions 和 Review 条目

用 Read 读取 `Workbench/queue.md`，提取两类条目：

1. **Questions 部分**：Researcher 自问的悬而未决的研究问题。若某个问题已经积累了足够多的答案线索（在 insights 或论文中出现相关回答），该 question 可能触发一个新方向的诞生。
2. **Review 部分**：已完成某类积累、等待 Researcher 决策的条目。特别关注"建议晋升至 DomainMaps"类条目，这类条目说明某个 validated insight 已足够成熟，可能值得围绕其开辟或强化一个研究方向。

对每条相关条目，记录其内容摘要，供后续判断变更时参考。

### Step 5：扫描高置信度 Ideas

用 Grep 在 `Ideas/` 目录中搜索包含 `status: developing` 或 `status: validated` 的 Idea 文件，获取匹配文件路径列表。

对每个匹配文件，用 Read 读取其内容，提取：
- `hypothesis` 字段（该 Idea 的核心假设）
- 评估结论（支持或否定 hypothesis 的证据摘要）
- `confidence` 字段（若有）

若 `Ideas/` 不存在或无匹配文件，跳过本步骤。

### Step 6：判断并执行 Agenda 变更

综合 Step 1-5 收集的全部信息，逐条评估 agenda 中每个 Active Direction 和 Paused Direction 的当前状态，同时评估是否需要新增方向。

每种变更类型的判断标准和执行方式如下：

---

**A. 新增 Direction**

触发条件（满足其一即可）：
- Step 2 中发现一条 validated insight，其 `impact` 指向一个 agenda 中尚不存在的研究方向
- Step 4 中有一个 question 已在 insights 或论文中找到足够的答案线索，形成可证伪的 hypothesis
- Step 5 中有一个 `status: validated` 的 Idea，其 hypothesis 尚未被现有 direction 覆盖

执行：用 Edit 在 `Workbench/agenda.md` 的 `## Active Directions` 节末尾追加新方向，格式严格遵循 `references/agenda-protocol.md`：

```markdown
### [Direction Name]

- **priority**: high / medium / low
- **status**: exploring
- **origin**: researcher-discovered
- **hypothesis**: <来自 insight claim 或 idea hypothesis 的一句话可证伪断言>
- **evidence**: [[Workbench/memory/insights.md#<heading>]]
- **next_action**: <基于现有证据，最合理的下一步具体行动>
- **confidence**: <根据 insight confidence 对应赋值，通常 0.3-0.5>
```

priority 依据：高度相关 Mission 且有强证据支持 → high；与 Mission 相关但证据初步 → medium；探索性方向 → low。

---

**B. 更新现有 Direction**

触发条件：某个 Active Direction 的 evidence、confidence、status 或 next_action 因新的 insight 或实验结果而需要更新。

常见更新场景：
- 新 insight 为现有 direction 提供了额外证据支撑 → 更新 `evidence` 和 `confidence`
- 该 direction 的 hypothesis 已被部分验证 → 将 `status` 从 `exploring` 升为 `validating`
- 多项独立证据指向同一结论 → 将 `status` 升为 `consolidating`，confidence 可升至 0.7-0.9
- 原有 `next_action` 已完成或已过时 → 更新为更具体的下一步

执行：用 Edit 修改 `Workbench/agenda.md` 中对应 direction 条目的相关字段行。每次只修改有变更的字段，保留其余字段原文不变。

---

**C. 暂停 Direction**

触发条件（满足其一即可）：
- 该 direction 的 `next_action` 已连续两次无法推进（因缺乏资源、前置依赖未完成等）
- 有更高优先级的新方向需要占用同等精力，当前方向需要让位
- 该 direction 的 hypothesis 在短期内无法被验证（如需等待实验数据、等待 Supervisor 反馈）

执行：
1. 用 Edit 将该 direction 从 `## Active Directions` 节剪切，移入 `## Paused Directions` 节
2. 在条目中添加 `pause_reason` 和 `resume_condition` 字段，格式：
   ```
   - **pause_reason**: <一句话说明暂停原因>
   - **resume_condition**: <明确的可判断的恢复条件>
   ```
3. 将 `status` 字段值改为 `paused`
4. 保留原有的 `hypothesis`、`evidence`、`confidence` 字段

注意：不得删除 Supervisor 手动添加的 direction，只能暂停并在 `pause_reason` 中注明来源；如确需删除，须先在 `## Discussion Topics` 节提出讨论。

---

**D. 废弃 Direction**

触发条件（需同时满足）：
- 该 direction 的 hypothesis 已被明确证伪（有具体证据，非猜测）
- 且 `resume_condition` 不存在或在可预期时间内无法满足

执行：
1. 用 Edit 将该 direction 从 `## Active Directions` 或 `## Paused Directions` 节移入 `## Abandoned Directions` 节，格式：
   ```markdown
   ### [Direction Name]

   - **abandoned_on**: <今天日期>
   - **original_hypothesis**: <原 hypothesis 字段内容>
   - **reason**: <一句话说明废弃原因，必须引用具体证据>
   - **lesson**: <一句话可复用的教训>
   - **memory_ref**: [[Workbench/memory/failed-directions.md#<heading>]]
   ```
2. 同步写入 `Workbench/memory/failed-directions.md`，追加以下格式条目（若文件不存在，先用 Edit 创建并加一级标题 `# Failed Directions`）：
   ```markdown
   ### [YYYY-MM-DD] <Direction Name>

   - **hypothesis**: <原 hypothesis>
   - **reason**: <废弃原因>
   - **lesson**: <可复用教训>
   - **evidence_against**: <指向证伪证据的具体来源>
   ```

---

**E. 新增 Discussion Topic**

触发条件：演化过程中发现需要 Supervisor 判断的议题，如：
- 两个方向的 hypothesis 存在逻辑冲突，需要 Supervisor 决定优先级
- 某个方向按现有证据应废弃，但它是 supervisor-assigned 来源，需确认

执行：用 Edit 在 `Workbench/agenda.md` 的 `## Discussion Topics` 节末尾追加：

```markdown
### [Topic title] — [今天日期]

- **raised_by**: agenda-evolve
- **context**: <触发该话题的具体情境>
- **question**: <需要 Supervisor 回答的具体问题>
- **related_direction**: <相关的 direction 名称>
```

---

若评估后无任何变更需要执行，在日志中记录"本次 agenda-evolve 扫描未发现需变更项"，仍需更新 frontmatter 并完成后续步骤。

### Step 7：更新 Agenda Frontmatter

用 Edit 修改 `Workbench/agenda.md` 开头的 frontmatter，将：
- `last_updated` 更新为今天日期（格式 `YYYY-MM-DD`）
- `updated_by` 更新为 `agenda-evolve`

### Step 8：追加 Changelog 条目

用 Edit 将以下格式的条目 append 到 `Workbench/evolution/changelog.md` 末尾（若文件不存在，先用 Edit 创建并加一级标题 `# Evolution Changelog`）：

```markdown
### [YYYY-MM-DD] agenda-evolve

- **trigger**: <触发原因，来自 input.trigger 或自动描述>
- **insights_reviewed**: <本次审视的 validated insight 数量>
- **directions_added**: <新增 direction 数量>
- **directions_updated**: <更新 direction 数量>
- **directions_paused**: <暂停 direction 数量>
- **directions_abandoned**: <废弃 direction 数量>
- **reasoning**: <2-3 句话说明本次变更的核心逻辑，便于日后回溯>
```

### Step 9：追加工作日志

用 Edit（或 Write 若文件不存在）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

```markdown
### [HH:MM] agenda-evolve

- **trigger**: <触发原因>
- **changes**: 新增 <N> 个方向，更新 <N> 个，暂停 <N> 个，废弃 <N> 个
- **observation**: <一句话描述本次演化的核心判断>
- **status**: success
```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] `Workbench/agenda.md` 的 `last_updated` 已更新为今天日期
- [ ] `Workbench/agenda.md` 的 `updated_by` 已更新为 `agenda-evolve`
- [ ] 每个 Active Direction 都有非空的 `next_action`
- [ ] 新增或变更（含新增、更新、暂停、废弃）在 `Workbench/evolution/changelog.md` 中均有记录
- [ ] 废弃的 direction 已同步写入 `Workbench/memory/failed-directions.md`
- [ ] 工作日志已追加到 `Workbench/logs/YYYY-MM-DD.md`

## Guard

- **严格遵循格式**：所有 agenda 条目的字段和格式必须符合 `references/agenda-protocol.md` 的规范，不得自定义字段名或省略必填字段。
- **不删除 Supervisor 方向**：`origin: supervisor-assigned` 的 direction 不得被废弃，只可暂停，并在 `pause_reason` 中注明"supervisor-assigned，需 Supervisor 确认后方可废弃"，同时在 Discussion Topics 中提出讨论。
- **变更必须有推理记录**：每次变更在 `evolution/changelog.md` 的 `reasoning` 字段中必须说明决策依据，不得只记录"变更了 X 方向"而不说明为何。
- **不修改 Mission 节**：`## Mission` 节的内容只有 Supervisor 可以直接修改，Researcher 可在 Discussion Topics 中提出 Mission 演化建议，但不得直接改写 Mission 正文。
- **废弃需有明确证据**：废弃 direction 时，`reason` 字段必须引用具体证据来源（insight 链接、实验结果等），不得基于"感觉方向不对"等主观判断执行废弃。
- **confidence 需有依据**：所有 confidence 值必须遵循 `references/agenda-protocol.md` 的定量规范：exploring 无证据 0.1-0.2，多篇论文支持 0.7-0.8，实验确认才可达 0.9+。
- **语言规范**：中文正文，英文技术术语（方向名、模型名、方法名、benchmark 名）保持英文，不做翻译。

## Examples

**示例：新 validated insight 触发方向演化**

背景：memory-distill 刚产出一条新的 validated insight，Researcher 触发 agenda-evolve 进行复盘。

```
/agenda-evolve --trigger "new validated insight from memory-distill"
```

执行过程：

1. Read `Workbench/agenda.md`，记录当前 3 个 Active Directions：
   - "VLN Generalization"（validating，confidence: 0.6）
   - "Reward Shaping for Sparse Environments"（exploring，confidence: 0.3）
   - "Cross-Modal Attention Fusion"（exploring，confidence: 0.2）

2. Read `Workbench/memory/insights.md`，发现 2 条近期 validated insight：
   - Insight A（validated，high confidence）：claim 为"waypoint prediction 辅助监督显著提升 VLN agent 在 unseen environment 中的泛化能力"，impact 指向 "VLN Generalization"
   - Insight B（validated，medium confidence）：claim 为"基于 diffusion 的策略对 action chunk size 高度敏感，chunk size=8 在连续控制任务中最优"，impact 指向"尚无对应方向"

3. Read `Workbench/memory/failed-directions.md`，发现历史上"End-to-End Waypoint Regression"曾被废弃（原因：训练不稳定，与 Insight A 无直接冲突）。

4. Read `Workbench/queue.md` Review 部分，发现一条"建议晋升至 DomainMaps"的条目，指向 Insight A，suggested_map 为 `DomainMaps/VLN.md`。

5. Glob + Read `Ideas/`，发现一个 `status: developing` 的 Idea "DiffusionPolicyTuning"，hypothesis 为"调整 action chunk size 可在操控任务上提升 10% 性能"，与 Insight B 高度吻合。

6. 判断变更：
   - **更新** "VLN Generalization"：Insight A 为其提供强证据，status 从 `validating` 升为 `consolidating`，confidence 从 0.6 升为 0.8，evidence 追加 Insight A 的链接，next_action 更新为"整理 waypoint prediction 方法对比，准备写综述段落"
   - **新增** "Diffusion Policy Chunk Size Optimization"：基于 Insight B + Idea "DiffusionPolicyTuning"，origin: researcher-discovered，status: exploring，confidence: 0.45，next_action: "复现 chunk size ablation 实验，验证 size=8 结论在 RoboMimic 数据集上是否成立"
   - "Reward Shaping for Sparse Environments" 和 "Cross-Modal Attention Fusion" 无新证据，保持不变

7. Edit `Workbench/agenda.md`：更新 "VLN Generalization" 的 status/confidence/evidence/next_action，追加新方向 "Diffusion Policy Chunk Size Optimization"，更新 frontmatter。

8. Edit `Workbench/evolution/changelog.md` 追加变更记录，reasoning 说明：Insight A 将 VLN Generalization 推进至 consolidating 阶段；Insight B 结合 developing idea 开辟新方向。

9. Edit `Workbench/logs/2026-03-28.md` 追加日志条目。

最终输出摘要：

```
agenda-evolve 完成（trigger: new validated insight from memory-distill）
- 审视 validated insights：2 条
- 更新 direction：1 个（VLN Generalization → consolidating，confidence 0.6 → 0.8）
- 新增 direction：1 个（Diffusion Policy Chunk Size Optimization）
- changelog 和日志已更新
```
