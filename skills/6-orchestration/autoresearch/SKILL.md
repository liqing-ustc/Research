---
name: autoresearch
description: >
  MindFlow 的核心研究循环。当 Supervisor 说"自己干活吧""开始研究"，
  或系统需要自主推进研究时启动。
  持续运行：读取当前状态 → 判断最高价值行动 → 调用卫星 skill → 记录 → 循环
argument-hint: "[focus]"
allowed-tools: Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---

## Purpose

autoresearch 是 MindFlow 的核心研究循环，实现了 PhD 导师制中 Researcher 的"自主研究日常"。它持续运行，每轮读取 Workbench 状态（agenda、queue、memory、logs），判断当前研究进展的最大瓶颈，调用对应的卫星 skill 执行最高价值行动，记录后进入下一轮。

与 Karpathy autoresearch 的关键区别：Karpathy 版优化单一 metric（val_bpb），MindFlow 版驱动知识积累和研究方向演化——目标不是某个指标下降，而是"我们理解了什么新东西"。

## Steps

每轮执行 4 步，然后循环：

### Step 1：READ STATE

读取以下文件了解当前状态：

1. `Workbench/agenda.md` — 当前研究方向、优先级、各 direction 的 next_action
2. `Workbench/queue.md` — 待处理任务（Reading / Review / Questions / Experiments 四个部分）
3. `Workbench/memory/insights.md` — 近期 insight（关注 status: validated 且近 30 天内的条目）
4. `Workbench/memory/patterns.md` — 近期 pattern
5. 用 Glob 列出最近 3 天的 `Workbench/logs/YYYY-MM-DD.md`，用 Read 读取，了解近期执行了什么（避免重复行动）

若 `focus` 参数指定了某个 direction，重点关注该 direction 相关的信息。

### Step 2：JUDGE

基于 Step 1 读取的状态，判断下一个最高价值行动。以下为参考启发式（非硬编码规则，由 LLM 综合判断——真实科研不是 if-else）：

| 状态信号 | 可能的行动 |
|:--------|:----------|
| queue 的 Reading 部分有待处理论文 | 读取 `skills/1-literature/paper-digest/SKILL.md` 并执行 |
| agenda 中某 direction 缺乏文献支撑（evidence 稀疏） | 读取 `skills/1-literature/literature-survey/SKILL.md` 并执行 |
| vault 中有多篇相关论文但未做对比分析 | 读取 `skills/1-literature/cross-paper-analysis/SKILL.md` 并执行 |
| 近期有 Topics/*-Analysis.md 标注了知识空白 | 读取 `skills/2-ideation/idea-generate/SKILL.md` 并执行 |
| Ideas/ 中有 status: raw 的 idea 待评估 | 读取 `skills/2-ideation/idea-evaluate/SKILL.md` 并执行 |
| Ideas/ 中有 status: developing 的 idea 缺实验方案 | 读取 `skills/3-experiment/experiment-design/SKILL.md` 并执行 |
| Experiments/ 中有 status: completed 且无 ## Analysis 节 | 读取 `skills/3-experiment/result-analysis/SKILL.md` 并执行 |
| 最近一次 memory-distill 距今 >5 天（检查 logs） | 读取 `skills/5-evolution/memory-distill/SKILL.md` 并执行 |
| 近期有新 validated insight 但 agenda 未反映 | 读取 `skills/5-evolution/agenda-evolve/SKILL.md` 并执行 |
| 某 direction 已有充足论文+实验+idea，需要成文 | 读取 `skills/4-writing/draft-section/SKILL.md` 并执行 |

判断时优先考虑：

- agenda 中 priority: high 的 direction 的 next_action
- queue 中积压的待处理项
- 最近未被处理的完成态工作（如 completed experiment 未分析）

### Step 3：ACT

读取判断出的目标 skill 的 SKILL.md 文件，**严格按其 Steps、Guard、Verify 执行**。

**关键**：一轮只调一个 skill。做一件事，做完记录，再想下一件。

### Step 4：LOG

用 Edit 追加本轮行动到 `Workbench/logs/YYYY-MM-DD.md`（若文件不存在，先用 Write 创建，包含一级标题 `# YYYY-MM-DD`）：

```markdown
### [HH:MM] autoresearch — round N
- **state_summary**: <读到了什么关键状态（1-2 句）>
- **judgment**: <为什么选这个行动（一句话推理）>
- **action**: <调了哪个 skill，传了什么参数>
- **outcome**: <产出了什么文件/更新>
```

然后回到 Step 1 开始下一轮。

## Verify

每轮结束时检查：

- [ ] 本轮有明确的 skill 调用（不允许"思考了一圈但什么都没做"）
- [ ] 日志已追加本轮记录（包含 state_summary / judgment / action / outcome 四个字段）

## Guard

- **原子性**：一轮只调一个卫星 skill——做一件事，做完记录，再想下一件
- **状态刷新**：每轮必须重新读取最新状态（不跳过 READ STATE），因为上一轮的行动可能改变了 vault 状态
- **Mission 只读**：不修改 `agenda.md` 的 Mission 节（Mission 的演化由 Supervisor 决定或通过 agenda-evolve 提议）
- **卡住检测**：若连续 3 轮的 JUDGE 判断结果指向同一个 skill 且同一个目标（如连续 3 轮都想对同一个 idea 做 evaluate），说明被卡住了——暂停循环，在 `agenda.md` 的 Discussion Topics 中添加一条问题，等待 Supervisor 输入
- **不对外发布**：不投稿论文、不发送外部通讯——PhD 导师制的唯一硬约束
- **Skill 执行规范**：调用卫星 skill 时，必须先 Read 对应的 SKILL.md 文件，严格按其 Steps 和 Guard 执行。不要凭记忆执行——每次都重新读取 SKILL.md
- **日志完整**：每轮的 LOG 步骤不可跳过，即使 skill 执行失败也要记录失败原因

## Examples

**示例：一次 autoresearch 会话的前 3 轮**

```
Supervisor: "自己干活吧"

--- Round 1 ---
READ STATE:
  agenda: 1 active direction "VLA few-shot adaptation"（status: exploring, next_action: "需要更多文献支撑"）
  queue: Reading 部分有 2 篇待读论文
  memory: 无近期 insight
  logs: 昨天执行了 1 次 paper-digest

JUDGE: queue 有待读论文 → 优先消化（清理积压）
ACT: 读取 skills/1-literature/paper-digest/SKILL.md，执行 digest 第一篇论文
LOG: round 1, action: paper-digest, outcome: Papers/2603-FewShotVLA.md

--- Round 2 ---
READ STATE:
  agenda: 同上（next_action 仍为"需要更多文献支撑"）
  queue: Reading 还剩 1 篇待读论文
  ...

JUDGE: queue 仍有待读论文 → 继续消化
ACT: paper-digest 第二篇
LOG: round 2, action: paper-digest, outcome: Papers/2601-AdaptiveVLA.md

--- Round 3 ---
READ STATE:
  agenda: next_action "需要更多文献支撑"
  queue: Reading 为空
  vault: Papers/ 中有 5 篇 VLA 相关论文但未做对比

JUDGE: queue 已清空，agenda direction 缺文献支撑 + vault 有多篇未对比论文 → 做跨论文分析
ACT: 读取 skills/1-literature/cross-paper-analysis/SKILL.md，对比 VLA 相关论文
LOG: round 3, action: cross-paper-analysis, outcome: Topics/VLA-FewShot-Analysis.md
```
