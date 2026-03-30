---
name: experiment-track
description: >
  当 Supervisor 说"记录一下实验结果""实验跑完了"，
  或 Researcher 完成一轮实验需要记录时，
  在 Experiment 笔记中追加 Run Entry 并更新状态
argument-hint: "<experiment> <result>"
allowed-tools: Read, Edit, Glob
---

## Purpose

experiment-track 是 MindFlow 实验执行循环的记录层。每当一轮实验（training run、ablation、baseline 对比等）产出结果，该 skill 负责以 **append-only** 的方式将 Run Entry 写入目标 Experiment 笔记，并根据当前进展同步更新 frontmatter `status`。

核心设计原则：**只追加，不修改**。已有 Run Entry 和 Variables/Baseline/Metrics 等设计字段一律保持不变，确保实验记录的完整性和可追溯性。

## Steps

### Step 1：读取目标 Experiment 笔记

用 Read 打开 `experiment` 参数指向的 `Experiments/xxx.md` 文件。重点提取以下字段：

- `status`：当前实验状态（`planning` / `running` / `completed` / `failed`）
- `variables`：实验变量（用于核对 Run Entry 中的 config 是否合理）
- `metrics`：评估指标（用于核对 result 中上报的指标是否对齐）
- 正文中已有的 `### Run [N]` 标题列表（用于计算新 Run 编号）
- `## Analysis` 节是否存在（用于确定追加位置）

若文件不存在，停止执行并告知 Supervisor，不继续后续步骤。

### Step 2：计算新 Run 编号

扫描已有的 `### Run [N]` 标题（N 为正整数）：

- 若存在已有 Run Entry，新 Run 编号 = max(N) + 1
- 若尚无任何 Run Entry，新 Run 编号 = 1

记录新编号供下一步使用。

### Step 3：追加 Run Entry

用 Edit 将 Run Entry 插入 Experiment 笔记，位置规则：

- 若存在 `## Analysis` 节，将 Run Entry 追加在 `## Analysis` **之前**
- 若不存在 `## Analysis` 节，追加到文件末尾

Run Entry 格式如下：

```markdown
### Run [N] — YYYY-MM-DD

- **config**: <本次运行的关键超参数或配置差异，对照 variables 字段逐项说明>
- **result**: <量化结果，务必包含 metrics 中定义的指标；可附路径指向日志或结果文件>
- **observation**: <对结果的直接观察：是否符合预期？异常现象？与 baseline 的差距？>
- **next**: <基于本次结果，下一步计划（调整 config、新 ablation、停止实验等）>
```

日期填写今天（`YYYY-MM-DD`）。`result` 字段必须非空——若 Supervisor 未提供具体数值，在字段中注明"待补充"并给出已知信息，不得留空。

### Step 4：更新 frontmatter status

用 Edit 修改 Experiment 笔记 frontmatter 中的 `status` 字段，按如下规则映射：

| 当前 status | next 字段内容 | 新 status |
| :--- | :--- | :--- |
| `planning` | 任意 | `running` |
| `running` | 非停止性描述（继续调参、新 ablation 等） | `running`（保持不变） |
| `running` | 停止性描述（实验达标、终止实验、结论确认等） | `completed` 或 `failed` |
| `completed` / `failed` | 任意 | 保持不变（不得回退已终止状态） |

停止性描述的判断标准：`next` 字段中含有"终止""停止""结论""完成""达标""failed""abort""done""conclude"等语义。若无法判断，保持 `running` 不变，在日志中注明"status 未变更，待 Supervisor 确认"。

仅修改 `status` 字段，frontmatter 其他字段保持不变。

### Step 5：追加日志

用 Edit（若文件不存在则先 Write）将以下 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`：

```markdown
### [HH:MM] experiment-track
- **experiment**: [[Experiments/xxx]]
- **run**: Run [N]
- **result-summary**: <result 字段的一句话摘要>
- **status-change**: <旧 status> → <新 status>
- **status**: success
```

若日志文件不存在，先创建文件（一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] Run Entry 已追加到 Experiment 笔记，编号连续且正确
- [ ] `result` 字段非空（有具体数值或"待补充"说明）
- [ ] `config` 字段对照 variables 有实质内容
- [ ] `observation` 和 `next` 字段均已填写
- [ ] frontmatter `status` 已按规则同步更新
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`

## Guard

- **append-only**：不得修改或删除已有的任何 `### Run [N]` Entry。已记录的实验结果是不可变的历史记录。
- **不修改设计字段**：`variables`、`baseline`、`metrics` 等由 experiment-design 写入的字段一律不得修改。
- **result 必须非空**：若 Supervisor 未提供量化结果，在 `result` 字段注明"待补充"，但不得完全省略该字段。
- **不回退终止状态**：`status` 为 `completed` 或 `failed` 时，不得改回 `running` 或 `planning`。
- **语言规范**：中文正文 + 英文技术术语（模型名、方法名、benchmark 名、超参数名保持英文，不翻译）。

## Examples

**示例：记录一次 VLN 模型的 training run 结果**

假设 Experiment 笔记为 `Experiments/VLN-BEVFusion-Ablation.md`，当前 status 为 `running`，已有 Run 1 和 Run 2，metrics 定义为 `SR`（Success Rate）和 `SPL`。

Supervisor 说："实验跑完了，Run 3 用 lr=1e-4、BEV resolution=0.5m，SR 61.2%，SPL 0.54，比 baseline 高了 3 个点，感觉 resolution 影响比较大，下一步试试 0.3m。"

执行过程：

1. Read `Experiments/VLN-BEVFusion-Ablation.md` — 确认文件存在，status=`running`，已有 Run 1、Run 2，metrics 含 SR 和 SPL，存在 `## Analysis` 节。

2. 计算新 Run 编号：max(1, 2) + 1 = **Run 3**。

3. Edit `Experiments/VLN-BEVFusion-Ablation.md`，在 `## Analysis` 之前追加：

   ```markdown
   ### Run 3 — 2026-03-28

   - **config**: lr=1e-4，BEV resolution=0.5m；其余超参与 Run 2 保持一致
   - **result**: SR 61.2%，SPL 0.54；较 baseline 分别提升 +3.0pp 和 +0.04
   - **observation**: 结果符合预期，BEV resolution 从 1.0m 降至 0.5m 带来了明显增益；SR 提升幅度大于 SPL，推测 resolution 对路径规划精度的影响比路径效率更显著
   - **next**: 进一步将 BEV resolution 降至 0.3m，验证 resolution 收益是否持续递增，同时观察推理速度变化
   ```

4. Edit frontmatter：`next` 描述为"进一步调参"，非停止性 → status 保持 `running` 不变。

5. Edit `Workbench/logs/2026-03-28.md`，追加：

   ```markdown
   ### [14:32] experiment-track
   - **experiment**: [[Experiments/VLN-BEVFusion-Ablation]]
   - **run**: Run 3
   - **result-summary**: lr=1e-4 + BEV 0.5m，SR 61.2%，SPL 0.54，较 baseline +3pp
   - **status-change**: running → running
   - **status**: success
   ```
