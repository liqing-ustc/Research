---
name: experiment-design
description: >
  当 Supervisor 说"设计个实验""怎么验证这个 idea"，
  或 autoresearch 判断某个 developing/validated idea 需要实验验证时，
  为 idea 设计完整实验方案
argument-hint: "<idea> [constraints]"
allowed-tools: Read, Write, Edit, Glob, Grep
---

## Purpose

experiment-design 是 MindFlow 从 idea 到实验的桥接技能。给定一个 status 为 `developing` 或 `validated` 的研究 idea，它综合 idea 的 hypothesis、历史记忆中的有效方法与失败方向，为其设计完整的实验方案，并按 `Templates/Experiment.md` 格式写入 `Experiments/` 目录。

实验方案覆盖六个核心模块：

| 模块 | 内容 |
| :--- | :--- |
| **Variables** | 自变量 / 因变量 / 控制变量 |
| **Baseline** | 对比方法与来源 |
| **Metrics** | 可量化的评估指标 |
| **Steps** | 具体执行步骤 |
| **Expected Outcome** | 假设成立与不成立两种情况 |
| **Risk & Mitigation** | 主要风险点与应对方案 |

该技能适用于两种模式：autopilot（全自动设计写入）、copilot（草稿预览后 Supervisor 确认写入）。

## Steps

### Step 1：读取目标 Idea 笔记，验证 status

用 Read 打开 `idea` 参数指向的 `Ideas/xxx.md` 文件。重点提取以下字段：

- `title`：idea 标题
- `hypothesis`：核心假设（实验围绕此展开）
- `status`：当前状态——**必须为 `developing` 或 `validated`，否则立即停止执行并告知 Supervisor**
- `tags`：所属 domain / 领域关键词
- `related_papers`：引用的 Paper 笔记列表
- idea 正文中的 Motivation、approach sketch（有助于确定 Variables 和 Steps）

若文件不存在，停止并告知 Supervisor，不继续执行。

### Step 2：检索历史记忆

读取 `skills/5-evolution/memory-retrieve/SKILL.md`，按其 Steps 执行，scope 设为 `effective-methods` + `failed-directions`，query 使用 idea 的 hypothesis 和 tags 关键词。

从检索结果中识别：
- **可复用的有效方法**：可直接借鉴到本实验的 baseline 选取、metric 设计、实验步骤
- **需要规避的失败方向**：历史上已被证明无效或高风险的路径，列入 Risk & Mitigation

若记忆文件不存在或无相关条目，跳过此步，在实验方案中注明"无历史记忆参考"。

### Step 3：设计实验方案六部分

综合 Step 1 的 idea 内容、Step 2 的历史记忆、以及可选的 `constraints` 参数，逐一设计以下六个模块：

#### Variables（变量设计）
- **自变量（Independent Variable）**：实验中主动改变的因素，直接对应 hypothesis 中的核心机制
- **因变量（Dependent Variable）**：被测量的结果变量，必须对应具体 Metric
- **控制变量（Controlled Variable）**：需要保持不变的条件（数据集、随机种子、训练轮数等），避免混淆变量

#### Baseline（对比方法）
列举 2-4 个 baseline，每个 baseline 注明：
- 方法名称和来源论文（引用 `[[Papers/xxx]]`）
- 选取理由（为何是公平的对比）

若 `constraints` 中含"纯理论推导"，可省略实验 baseline，改为理论分析对比。

#### Metrics（评估指标）
列举 2-4 个指标，每个指标必须满足：
- **可量化**：有明确的数值定义，可从实验结果直接计算
- **与 hypothesis 直接相关**：能判断假设是否成立
- 注明主指标（primary metric）和辅助指标（secondary metrics）

**禁止**使用无法量化的描述性指标（如"效果更好"、"更鲁棒"）——必须转化为具体数值（如"success rate ↑ X%"、"BLEU-4 score"）。

#### Steps（执行步骤）
按时间顺序列出 4-8 个可执行步骤，每步包含：
- 具体操作（做什么、用什么工具/数据集/框架）
- 预期产出（checkpoint、日志、中间结果）

若有 `constraints`（如"单卡 A100"），在 Steps 中体现对应的资源配置。

#### Expected Outcome（预期结果）
必须包含**两种情况**：

- **假设成立（Hypothesis Confirmed）**：若 hypothesis 正确，主指标应呈现什么数值趋势？给出定量预期（如"success rate 比 baseline 提升 ≥5%"）
- **假设不成立（Hypothesis Refuted）**：若 hypothesis 错误，结果会呈现什么模式？此时应如何解读——是放弃 idea、修正 hypothesis、还是设计后续实验？

#### Risk & Mitigation（风险与应对）
列举 2-3 个主要风险，每条注明：
- 风险描述（什么可能出错）
- 发生概率（high / medium / low）
- 应对方案（如何规避或补救）

优先纳入 Step 2 中识别的历史失败方向作为风险条目。

### Step 4：按模板创建实验文件

读取 `Templates/Experiment.md`，以其格式为基础，在 `Experiments/` 目录下创建新文件。

文件命名规则：`Experiments/YYYY-MM-DD-<IdeaTitle>.md`（日期为今天，IdeaTitle 取 idea 的 `title` 字段，去除空格和特殊字符）。

Frontmatter 字段填写：
- `title`：实验标题（可与 idea title 相同或更具体）
- `idea`：`"[[Ideas/xxx]]"`（关联源 idea）
- `tags`：从 idea 继承
- `status`：设为 `planned`
- `date_created`：今天日期

正文按 Step 3 的六模块内容填充各 section，具体对应：
- `## Objective`：填写来自 hypothesis 的实验目标（1-3 句话）
- `## Setup`：从 Variables（控制变量）和 Steps（环境要求）提取
- `## Method`：填写 Steps（执行步骤）和 Baseline
- `## Results`：留空（实验完成后由 experiment-track 填写）
- `## Analysis`：留空
- `## Insights`：留空
- `## Next Steps`：填写 Expected Outcome 中"假设不成立"情况下的后续路径

在文件末尾额外追加一个 `## Design Notes` 节，包含 Variables 完整设计、Metrics、Expected Outcome（两种情况）、Risk & Mitigation，以保留完整实验设计细节。

### Step 5：在 Idea 笔记中追加实验链接

用 Edit 在源 Idea 文件（`Ideas/xxx.md`）末尾追加以下格式的链接节：

```markdown
## Experiments

- [[Experiments/YYYY-MM-DD-IdeaTitle]] — <一句话说明这个实验验证什么>
```

若 Idea 文件中已有 `## Experiments` 节，在该节末尾追加新条目，不重建整个节。不修改 idea 的 `hypothesis` 字段或任何其他已有内容。

### Step 6：追加日志

用 Edit（若文件不存在则先 Write）将以下 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`：

```markdown
### [HH:MM] experiment-design
- **input**: [[Ideas/xxx]]
- **output**: [[Experiments/YYYY-MM-DD-IdeaTitle]]
- **hypothesis**: <一句话复述 idea 的 hypothesis>
- **primary-metric**: <主指标名称>
- **key-risk**: <最高优先级风险一句话>
- **status**: success
```

若日志文件不存在，先创建文件（一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] `Experiments/YYYY-MM-DD-<IdeaTitle>.md` 已创建，frontmatter status 为 `planned`
- [ ] Variables（自变量 / 因变量 / 控制变量）均非空
- [ ] Baseline 列举了至少 2 个对比方法
- [ ] Metrics 均为可量化指标（有具体数值定义）
- [ ] Expected Outcome 包含假设成立和不成立两种情况
- [ ] 源 Idea 文件已追加实验链接
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`

## Guard

- **Metrics 必须可量化**：任何无法直接计算数值的指标必须被拒绝或转化。若无法为某个 hypothesis 找到可量化的 metric，停止执行并告知 Supervisor。
- **不修改源 Idea hypothesis**：`hypothesis` 字段属于 idea-generate / idea-evaluate 的职责范围，experiment-design 只读取，绝不修改。
- **status 校验为硬性前置条件**：若 idea 的 status 不是 `developing` 或 `validated`，立即停止，不执行任何写入操作。告知 Supervisor 当前 status 及建议（如先执行 idea-evaluate）。
- **不伪造历史记忆**：Step 2 的记忆检索若无结果，如实说明，不编造相关历史经验。
- **语言规范**：中文正文 + 英文技术术语（模型名、数据集名、benchmark 名、metric 名保持英文，不翻译）。
- **copilot 模式**：若以 copilot 模式调用，在写入文件前先输出完整实验方案草稿，等待 Supervisor 确认后再执行 Step 4-6。

## Examples

**示例：为 VLA 相关 idea 设计实验**

假设 Idea 文件为 `Ideas/VLA-GoalConditionedRecovery.md`，其关键字段为：
- `status: developing`
- `hypothesis`: "在 VLA 策略中引入 goal-conditioned recovery module，可以在 distribution shift 情境下显著减少 task failure"
- `tags`: [VLA, robot-learning, recovery]

执行过程：

1. Read `Ideas/VLA-GoalConditionedRecovery.md` — 确认 status=`developing`，提取 hypothesis 和 approach sketch（在 OpenVLA 基础上增加一个轻量 recovery head，输入 current state + goal image，输出 recovery action sequence）

2. 执行 memory-retrieve（scope: effective-methods + failed-directions）— 检索到一条 effective-methods 条目："reward shaping 在 manipulation task 上比直接 BC 收敛更稳定"；检索到一条 failed-directions 条目："直接 fine-tune 整个 VLA 做 recovery 导致 catastrophic forgetting"

3. 设计六模块方案：

   **Variables**
   - 自变量：是否附加 goal-conditioned recovery module（w/ recovery vs. w/o recovery）
   - 因变量：task success rate、recovery success rate（从 failure state 恢复后完成任务的比例）
   - 控制变量：base VLA 权重（OpenVLA-7B frozen）、训练数据集（Bridge v2）、随机种子（固定 3 个种子取均值）

   **Baseline**
   - `OpenVLA`（`[[Papers/2403-OpenVLA]]`）：无 recovery 机制的原始模型，直接对比
   - `SayCan + replanning`（`[[Papers/2204-SayCan]]`）：基于 LLM replanning 的 failure recovery，代表现有主流方案

   **Metrics**
   - 主指标：task success rate on out-of-distribution perturbation test set（扰动后成功率，%）
   - 辅助指标：recovery trigger precision（正确识别 failure state 的精度）、extra inference latency（ms/step）

   **Steps**
   1. 在 OpenVLA 基础上添加轻量 recovery head（2 层 MLP），仅 head 参数可训练，base VLA frozen（规避历史失败方向：catastrophic forgetting）
   2. 构建 recovery training set：从 Bridge v2 中提取 failure trajectory，标注 recovery action
   3. 训练 recovery head，单卡 A100，训练 10 epoch，保存 checkpoint
   4. 在 Bridge v2 OOD test split（加入随机扰动）上评估 task success rate
   5. 与两个 baseline 在相同 test split 上对比

   **Expected Outcome**
   - 假设成立：w/ recovery 的 task success rate 比 OpenVLA baseline 在 OOD test set 上提升 ≥5%（绝对值），recovery trigger precision ≥ 80%
   - 假设不成立：若提升 < 2% 或 recovery trigger precision < 60%，说明轻量 head 表达能力不足；下一步应扩大 recovery module（考虑增加 cross-attention 层）或重新审视 failure detection 的 feature 设计

   **Risk & Mitigation**
   - 风险 1：recovery training set 规模不足，head 过拟合（medium）→ 先做数据增强（random crop + color jitter），不足时考虑 synthetic failure generation
   - 风险 2：OOD test set 与 training distribution 差距过大，base VLA 本身失败率过高掩盖 recovery 效果（medium）→ 分层分析：分别报告 mild / severe OOD 扰动下的指标
   - 风险 3：recovery head 增加推理延迟超出实时控制要求（low）→ 提前 profile latency；若超标改用 distillation 压缩 head

4. Read `Templates/Experiment.md` → Write `Experiments/2026-03-28-VLA-GoalConditionedRecovery.md`，填入上述方案，status=`planned`

5. Edit `Ideas/VLA-GoalConditionedRecovery.md` — 追加：
   ```markdown
   ## Experiments
   - [[Experiments/2026-03-28-VLA-GoalConditionedRecovery]] — 验证 goal-conditioned recovery module 在 OOD 扰动下的 task success rate 提升
   ```

6. Edit `Workbench/logs/2026-03-28.md` — 追加 log entry，primary-metric: task success rate on OOD test set，key-risk: recovery training set 规模不足导致过拟合
