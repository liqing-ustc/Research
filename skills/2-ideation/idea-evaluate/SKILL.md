---
name: idea-evaluate
description: >
  当 Supervisor 说"评估一下这个 idea""这个可行吗"，
  或 autoresearch 需要筛选 idea 优先级时，
  从 novelty/feasibility/impact/risk/evidence 五维评估研究 idea
argument-hint: "<idea> [criteria]"
allowed-tools: Read, Edit, Glob, Grep
---

## Purpose

idea-evaluate 是 MindFlow 的 idea 质量守门人。给定一个研究 idea（以 `[[Ideas/xxx.md]]` 形式引用），它系统性地从五个维度进行评估：

| 维度 | 含义 |
| :--- | :--- |
| **Novelty** | 与已有工作的差异化程度——这个 idea 是否真的新？ |
| **Feasibility** | 当前资源条件下能否执行——有没有致命工程障碍？ |
| **Impact** | 若成功，对领域的贡献——有多少人会在意这个结果？ |
| **Risk** | 主要失败模式和概率——分越高代表风险越低 |
| **Evidence** | 当前支撑假设的证据强度——有多少先验支撑这个方向？ |

评估结束后产出三种结论之一：**recommend**（建议推进）/ **revise**（有潜力但需修正）/ **shelve**（搁置）。结论会同步写回 Idea 文件的 `status` 字段，确保状态一致性。

该技能适用于三种模式：autopilot（全自动评估写回）、copilot（草稿预览后 Human 确认写回）、sparring（仅输出评估结论，不修改文件）。

## Steps

### Step 1：读取目标 Idea 笔记

用 Read 打开 `idea` 参数指向的 `Ideas/xxx.md` 文件。重点提取以下字段：

- `title`：idea 标题
- `hypothesis`：核心假设（评估围绕此展开）
- `related_papers`：引用的 Paper 笔记列表（下一步使用）
- `status`：当前状态（评估前记录，评估后更新）
- `tags`：所属 domain/领域关键词
- idea 正文中已有的 Motivation、背景描述、初步方案

若文件不存在，停止并告知 Human，不继续执行。

### Step 2：读取相关 Paper 笔记

逐一 Read `related_papers` 中引用的每篇 Paper 笔记（通常在 `Papers/` 目录下）。从每篇笔记中提取：

- **Summary**：论文核心贡献一句话
- **Method**：方法要点（判断与 idea 的差异化）
- **Key Results**：主要实验结果（判断 idea 的 Novelty 和 Evidence）
- **Strengths & Weaknesses**：先前工作的局限（判断 idea 的 Feasibility 和 Impact）

若 `related_papers` 为空，用 Grep 在 `Papers/` 中搜索与 idea `tags` 相关的关键词，找出最相关的 2-4 篇笔记并读取。

### Step 3：读取 DomainMaps 上下文

1. 用 Read 打开 `DomainMaps/_index.md`，定位与该 idea 相关的 domain（依据 tags 或 hypothesis 中的关键词）。
2. 用 Read 打开对应的 `DomainMaps/{Name}.md`，重点阅读：
   - **Active Debates**：领域当前争议（判断 Novelty 和 Impact）
   - **Open Questions**：未解决问题（判断 Impact 和 Evidence）
   - **Key Methods / Baselines**：主流方法（判断 Feasibility）

若找不到对应 domain 文件，跳过此步，并在评估记录中注明"DomainMaps 无对应条目，评估基于 Paper 笔记"。

### Step 4：五维评估打分

综合前三步收集的信息，对以下五个维度分别给出 **1-5 分**和简要说明（1-3 句话）：

#### Novelty（1-5）
评估 idea 与已有工作的差异化程度：
- **5**：核心方法/框架在 vault 中无先例，与相关论文有明确区分
- **3**：有新角度，但与已有工作重叠较多，需要进一步差异化
- **1**：与已有论文高度重复，几乎无新贡献

#### Feasibility（1-5）
评估当前资源条件（算力、数据、工程能力）下能否在合理时间内完成：
- **5**：可用已有工具/数据直接实现，技术路径清晰
- **3**：需要较大工程投入，但无不可逾越的障碍
- **1**：存在致命依赖（专有数据/硬件/外部合作），短期无法启动

#### Impact（1-5）
评估若实验成功，对领域的贡献和影响力：
- **5**：直接解决 DomainMaps Open Questions 或 Active Debates，有顶会潜力
- **3**：对领域有增量贡献，但非核心问题
- **1**：结论意义有限，受众极小

#### Risk（1-5，分越高风险越低）
列举 1-3 个主要失败模式，评估综合风险水平：
- **5**：假设有充分先验支持，失败模式已知且可控
- **3**：核心假设未经验证，存在中等概率的负面结果
- **1**：假设高度投机，现有证据薄弱，失败概率极高

#### Evidence（1-5）
评估当前支撑 hypothesis 的证据强度：
- **5**：相关论文或 ablation 直接佐证了核心假设
- **3**：有间接证据或类比支持，但无直接实验验证
- **1**：纯理论推测，vault 中无任何支撑证据

**总分** = Novelty + Feasibility + Impact + Risk + Evidence（满分 25）

### Step 5：生成结论

根据评分规则确定 Verdict：

| 条件 | Verdict |
| :--- | :--- |
| 总分 ≥ 18，且无任何单维度 ≤ 1 | **recommend** |
| 总分 < 12，或 Feasibility = 1 | **shelve** |
| 其余情况（有潜力但某维度 ≤ 2） | **revise** |

同时生成：
- **Reasoning**（2-4 句话）：解释为什么得出此结论，指出最关键的决定性因素
- **Suggestions**（可选，仅 revise 时必须提供）：具体的修改方向——如"缩小任务范围以提升 Feasibility"或"补充 XYZ 实验以加强 Evidence"

### Step 6：追加评估记录到 Idea 文件

用 Edit 在目标 Idea 文件（`Ideas/xxx.md`）末尾追加以下格式的评估节：

```markdown
## Evaluation — YYYY-MM-DD

| 维度 | 分数 | 说明 |
| :--- | :---: | :--- |
| Novelty | X/5 | <简要说明> |
| Feasibility | X/5 | <简要说明> |
| Impact | X/5 | <简要说明> |
| Risk | X/5 | <简要说明> |
| Evidence | X/5 | <简要说明> |
| **Total** | **XX/25** | |

**Verdict**：recommend / revise / shelve

**Reasoning**：<2-4 句话解释结论>

**Suggestions**：<若 verdict=revise，列出具体修改方向；否则可省略>
```

日期填写今天（`YYYY-MM-DD`）。不得修改 Idea 文件中已有的 `hypothesis` 字段或正文内容，仅追加 `## Evaluation` 节。

### Step 7：同步更新 Idea frontmatter status

用 Edit 修改 Idea 文件 frontmatter 中的 `status` 字段，按如下规则映射：

| Verdict | 新 status |
| :--- | :--- |
| recommend | `developing` |
| revise | 保持原值（通常为 `raw`） |
| shelve | `archived` |

仅修改 `status` 字段，frontmatter 其他字段保持不变。

### Step 8：追加日志

用 Edit（若文件不存在则先 Write）将以下 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`：

```markdown
### [HH:MM] idea-evaluate
- **input**: [[Ideas/xxx]]
- **verdict**: recommend / revise / shelve
- **scores**: Novelty X, Feasibility X, Impact X, Risk X, Evidence X（Total XX/25）
- **key-factor**: <决定 verdict 的最关键因素一句话>
- **status**: success
```

若日志文件不存在，先创建文件（一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] Idea 文件已追加 `## Evaluation — YYYY-MM-DD` 节
- [ ] 5 个维度均有评分（1-5）和简要说明
- [ ] 结论（Verdict）为 recommend / revise / shelve 之一
- [ ] Idea 的 frontmatter `status` 已按规则同步更新
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`

## Guard

- **不修改 hypothesis 字段**：`hypothesis` 是 idea-generate 的职责范围，idea-evaluate 只读取不修改。
- **评估必须基于可追溯证据**：每个维度的说明必须能对应到 vault 中的具体笔记（Paper、DomainMaps、已有 Idea）。禁止凭空判断——若无足够证据支撑某维度评分，在说明中注明"证据不足，保守估计"。
- **不覆盖已有 Evaluation 节**：若 Idea 文件中已有 `## Evaluation` 节，追加新节（带新日期），不得删除历史评估记录。
- **sparring 模式不写文件**：若以 sparring 模式调用，仅输出评估结论到对话，不执行任何 Edit 操作。
- **语言规范**：中文正文 + 英文技术术语（模型名、方法名、benchmark 名保持英文，不翻译）。

## Examples

**示例：评估一个 VLA 相关 idea**

假设 Idea 文件为 `Ideas/VLA-GoalConditionedRecovery.md`，其 hypothesis 为：
> "在 VLA 策略中引入 goal-conditioned recovery module，可以在 distribution shift 情境下显著减少 task failure。"

执行过程：

1. Read `Ideas/VLA-GoalConditionedRecovery.md` — 提取 hypothesis、related_papers（`[[Papers/2403-OpenVLA]]`、`[[Papers/2309-RT2]]`）、tags（`VLA`、`robot-learning`、`recovery`）

2. Read `Papers/2403-OpenVLA.md` 和 `Papers/2309-RT2.md` — 提取 Method 和 Key Results，确认两者均未涉及 recovery mechanism

3. Read `DomainMaps/_index.md` → 定位 `VLA` → Read `DomainMaps/VLA.md` — Active Debates 中有"如何提升 VLA 在 out-of-distribution 场景的鲁棒性"

4. 五维评估结果：

   | 维度 | 分数 | 说明 |
   | :--- | :---: | :--- |
   | Novelty | 4/5 | OpenVLA 和 RT-2 均无 recovery module，差异化明显；与 manipulation recovery 文献有部分重叠 |
   | Feasibility | 3/5 | 需要构建 failure detection dataset，工程量较大，但无不可逾越的障碍 |
   | Impact | 4/5 | 直接对应 VLA.md 中 Active Debates 的 OOD 鲁棒性问题，受众广 |
   | Risk | 3/5 | goal-conditioned 模块的训练稳定性未知，存在中等风险 |
   | Evidence | 3/5 | 有间接证据（recovery 在 classical robotics 中有效），但 VLA 场景无直接验证 |
   | **Total** | **17/25** | |

5. 总分 17，Risk 和 Evidence 均为 3（无致命短板，但未达 ≥18 阈值） → **Verdict: revise**

   **Reasoning**：idea 方向有价值，Novelty 和 Impact 均强，直接对应领域核心痛点。主要短板在 Feasibility（failure detection dataset 的构建成本）和 Evidence（缺乏 VLA 场景的直接先验）。

   **Suggestions**：（1）先用现有 VLA benchmark 的 failure case 做 pilot experiment，验证 recovery module 的基本可行性；（2）调研 `manipulation recovery` 文献，补充 Evidence；（3）考虑简化为 lightweight recovery probe 降低工程成本，提升 Feasibility 评分。

6. Edit `Ideas/VLA-GoalConditionedRecovery.md` — 追加 `## Evaluation — 2026-03-28` 节（含上表、Verdict、Reasoning、Suggestions）

7. Edit frontmatter `status: raw`（verdict=revise，保持原值不变）

8. Edit `Workbench/logs/2026-03-28.md` — 追加 log entry
