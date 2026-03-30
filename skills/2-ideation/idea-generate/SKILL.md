---
name: idea-generate
description: >
  当 cross-paper-analysis 发现知识空白、memory 中有 validated insight 待探索、
  或 Supervisor 说"想个 idea""有什么研究机会"时，
  从知识空白和已有洞察中生成可证伪的研究 idea
argument-hint: "<source> [constraints]"
allowed-tools: Read, Write, Edit, Glob, Grep
---

## Purpose

idea-generate 是 MindFlow 的创意引擎。给定知识空白、validated insight、或 Supervisor 的方向性指令，它生成 2-3 个可证伪的研究 idea，按 `Templates/Idea.md` 格式写入 `Ideas/`。它是从"发现问题"到"提出假设"的桥梁——将文献分析的输出（空白、矛盾、未解决问题）转化为可操作的研究假设。

与 cross-paper-analysis 不同，idea-generate 不做文献梳理；与 idea-evaluate 不同，它不评估可行性。它只做一件事：给出清晰、可证伪、值得深入的研究假设，供 Researcher 和 Supervisor 讨论与筛选。

## Steps

### Step 1：读取来源材料

根据 `source` 参数的类型，选择对应读取路径：

**若 source 指向 Topics 分析文件中的知识空白节**（如 `[[Topics/VLA-MethodComparison-Analysis#知识空白]]`）：

1. 用 Read 读取对应的 `Topics/*-Analysis.md` 文件，定位 `## 知识空白` 节。
2. 提取所有标注了"建议生成 Idea"的空白条目，记录其描述、相关论文 wikilinks、以及研究价值评估。
3. 若知识空白节未明确标注但内容丰富，自行判断哪些空白最具研究潜力（优先选择：多篇论文均指出但无人解决的局限、实验设置组合空白、方法适用性边界未探索等）。

**若 source 指向 insights.md 中的 validated insight**（如 `[[Workbench/memory/insights.md]]`）：

1. 用 Read 读取 `Workbench/memory/insights.md`，筛选 `confidence: high` 或 `status: validated` 的条目。
2. 识别哪些 insight 暗示了可探索的研究方向（如"发现 X 在 Y 场景下失效"暗示"设计专门针对 Y 的 X 变体"）。
3. 记录每个 insight 的原始来源 wikilinks，后续 idea 文件中需引用。

**若 source 为 Supervisor 的直接指令**（如"想个 idea：怎么让 VLA 在 few-shot 场景下更好"）：

1. 解析 Supervisor 的核心意图：涉及什么任务（VLA/VLN/其他）、什么约束（few-shot / no-GPU / 理论向等）、期望的改进方向（性能提升、效率、泛化……）。
2. 用 Grep 在 `Papers/` 目录搜索相关关键词（如 `few-shot`、`VLA`），了解 vault 中已有哪些相关论文笔记，避免生成已被充分研究的方向。
3. 将 Supervisor 指令转化为结构化的"问题-约束"对，作为后续 idea 生成的输入框架。

**若 source 指向 agenda.md 中某 direction 的 next_action**：

1. 用 Read 读取 `Workbench/agenda.md`，找到对应 direction，提取其 `goal`、`context` 和 `next_action` 字段。
2. 将 `next_action` 的描述作为 idea 生成的核心方向。

### Step 2：读取废弃方向，规避已走过的弯路

1. 用 Glob 检查 `Workbench/memory/failed-directions.md` 是否存在。
2. 若存在，用 Read 读取全文，提取所有已废弃方向的核心描述（通常是每个条目的 `direction` 或标题字段）。
3. 将废弃方向列表记录在工作内存中，后续生成每个 candidate idea 时逐一比对：
   - 若候选 idea 与某废弃方向实质相同（即使表述不同），跳过该 idea，改换角度。
   - 若候选 idea 在废弃方向基础上有明确的新突破口，可保留，但需在 idea 文件的 `## Open Questions` 节中注明"与废弃方向 X 的区别在于……"。

### Step 3：读取 DomainMaps，锚定领域已知边界

1. 用 Read 读取 `DomainMaps/_index.md`，找到与 source 相关的 domain（如 VLA、VLN、RL 等）。
2. 用 Read 读取对应的 `DomainMaps/{Name}.md`，重点关注：
   - `## Established Knowledge`：已有共识，生成 idea 时可以利用，但不应作为 contribution 点（除非有明确突破）。
   - `## Active Debates`：领域内仍有争议的问题，是生成 idea 的高价值区域——可以针对某个 debate 设计决定性实验或提出新框架。
   - `## Open Questions`：领域显式列出的未解决问题，可直接对应生成 idea。
3. 将 DomainMaps 的内容整合进 idea 生成时的"背景约束"：生成的 idea 要建立在 Established Knowledge 之上，要么解决 Open Questions，要么推进 Active Debates。

### Step 4：生成 2-3 个候选 idea

综合 Step 1-3 的材料，生成 **2-3 个** 候选 idea。若 `constraints` 参数被指定（如"不需要 GPU""偏理论"），在此步骤中对每个候选 idea 检查约束满足情况，不满足约束的方向直接排除。

每个候选 idea 须包含以下五个要素（在工作内存中组织，下一步写入文件）：

- **hypothesis**：一句话的可证伪断言，格式建议为"若……则……"或"我们假设……"。判断标准：这句话能否在合理资源范围内被实验证伪？
- **motivation**：为什么这个方向值得研究？参考三个维度：（1）它解决了哪个知识空白或 Active Debate；（2）成功的话对领域有什么影响；（3）为什么现在是做这个的好时机（相关工具/数据已成熟？）。
- **approach sketch**：初步的方法思路，不需要完整，但要有足够细节让 Researcher 能判断可行性。说明核心技术路径（例如：用 LoRA fine-tuning + synthetic data augmentation 解决 few-shot adaptation）。
- **expected outcome**：若 hypothesis 成立，实验应该观察到什么？用具体的 benchmark 或 metric 描述（如"在 LIBERO-Spatial 上 few-shot success rate 提升 >10%"）。
- **risk**：最可能的失败模式是什么？（如"合成数据 domain gap 过大导致迁移失败"）。

多个候选 idea 之间应有足够差异度，避免生成本质相同的变体：
- 可以在方法层面求变（不同技术路径解决同一问题）；
- 可以在问题层面求变（同一方法应用到不同开放问题）；
- 避免仅在超参或实现细节上差异的"伪多样性"。

### Step 5：按模板写入 Ideas/ 文件

1. 用 Read 读取 `Templates/Idea.md`，了解模板字段结构（frontmatter + sections）。
2. 为每个候选 idea 创建独立的 `Ideas/` 文件：
   - **文件名**：用描述性 CamelCase 名称，2-4 个关键词，体现 idea 的核心主张，如 `Cross-Domain-VLA-Transfer.md`、`FewShot-VLA-LoRA.md`。避免用 `Idea1.md` 此类无意义命名。
   - **去重检查**：用 Glob 扫描 `Ideas/` 目录，若已存在内容相近的文件（文件名或标题高度重叠），停止创建该 idea 并告知 Human，不覆盖已有文件。
3. 按模板格式填写每个 idea 文件：
   - `title`：与文件名对应的简洁标题
   - `status`：固定填 `raw`（新生成的 idea 均为未评估状态）
   - `date_created`：今天日期，格式 `YYYY-MM-DD`
   - `feasibility`：固定填 `unverified`
   - `## Core Idea`：填写 Step 4 中的 hypothesis（一句话断言）
   - `## Motivation`：填写 Step 4 中的 motivation（为什么值得研究）
   - `## Related Work`：填写相关论文的 `[[wikilinks]]`，仅引用 vault 中 `Papers/` 目录下确实存在的笔记，不引用未经 paper-digest 消化的外部论文
   - `## Rough Plan`：填写 Step 4 中的 approach sketch + expected outcome（合并为初步实验计划）
   - `## Open Questions`：填写 Step 4 中的 risk，以及 Researcher 尚未想清楚的关键问题（鼓励诚实记录不确定性）
4. 用 Write 将每个 idea 文件保存到 `Ideas/` 目录。

### Step 6：追加日志

用 Edit（若文件不存在则用 Write）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

```markdown
### [HH:MM] idea-generate
- **input**: <source 参数内容>
- **output**: [[Ideas/Idea1Name]], [[Ideas/Idea2Name]], ...
- **observation**: <一句话描述本次生成的 idea 共同指向的研究方向>
- **status**: success
```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] `Ideas/*.md` 已创建（至少 1 个文件）
- [ ] 每个 idea 文件的 `## Core Idea` 节包含可证伪的一句话断言
- [ ] 所有 `## Related Work` 中的引用均指向 vault 中已有的 `Papers/` 笔记（`[[Papers/...]]` 格式）
- [ ] 未与 `Workbench/memory/failed-directions.md` 中已废弃方向实质重复
- [ ] 每个 idea 文件的 frontmatter `status` 字段为 `raw`，`feasibility` 为 `unverified`
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`

## Guard

- **不修改任何已有 Idea 文件**：若 `Ideas/` 下已存在内容相近的文件，停止执行并告知 Human；不得覆盖或追加已有 Idea 文件的内容。
- **不捏造文献支持**：`## Related Work` 中所有引用必须指向 vault 中 `Papers/` 目录下已存在的笔记（`[[Papers/YYMM-ShortTitle]]`）；若尚无相关 Paper 笔记，该字段留空或写"暂无相关笔记，建议先 paper-digest X"，不得引用不存在的链接。
- **不直接修改 agenda.md**：idea-generate 的产出只写入 `Ideas/`；是否将某个 idea 纳入 agenda 的 direction，由 agenda-evolve skill 或 Supervisor 决策，idea-generate 不擅自修改 `Workbench/agenda.md`。
- **hypothesis 必须可证伪**：若某个候选 idea 的假设无法在合理实验条件下被证伪（如过于宽泛的"改进 VLA 的泛化能力"），不得写入文件；须拒绝该候选，并在输出中说明原因，改换更具体的方向。
- **语言规范**：正文用中文撰写，英文技术术语（模型名、方法名、benchmark 名、任务名）保持英文，不做翻译。
- **autonomy: copilot 模式**：若以 copilot 模式调用，先将所有候选 idea 的摘要（hypothesis + motivation）输出给 Human 预览和筛选，Human 确认选择哪些后，再执行 Write 写入文件；日志同样在确认后追加。

## Examples

**示例 1：从 Analysis 中的知识空白生成 idea**

```
/idea-generate --source "[[Topics/VLA-MethodComparison-Analysis#知识空白]]"
```

执行过程：

1. Read `Topics/VLA-MethodComparison-Analysis.md`，定位 `## 知识空白` 节，提取标注了"建议生成 Idea"的条目（例如："few-shot adaptation 在真实机器人上的系统性研究缺失"）
2. Glob 检查 `Workbench/memory/failed-directions.md` 是否存在；若存在，Read 并记录废弃方向
3. Read `DomainMaps/_index.md` → Read `DomainMaps/VLA.md`，了解 Established Knowledge（如"VLA 普遍依赖大规模预训练"）和 Active Debates（如"in-context learning 对 VLA 的实际效果存在争议"）
4. 生成 3 个候选 idea：
   - **Idea A**：假设"通过 LoRA adapter + 少量真实机器人数据，VLA 可在 5-shot 内适应新任务"（针对 few-shot 空白）
   - **Idea B**：假设"合成数据增强可显著降低 few-shot VLA 所需真实数据量"（从数据角度切入）
   - **Idea C**：假设"prompt-based in-context learning 在 VLA 上比 gradient-based fine-tuning 更 sample-efficient"（针对 Active Debate）
5. Glob `Ideas/` 检查无重复
6. Read `Templates/Idea.md`，为每个候选 idea 写入文件：
   - Write `Ideas/FewShot-VLA-LoRA.md`
   - Write `Ideas/SyntheticData-VLA-Adaptation.md`
   - Write `Ideas/InContextLearning-VLA-Efficiency.md`
7. 追加日志到 `Workbench/logs/2026-03-28.md`

输出文件：`Ideas/FewShot-VLA-LoRA.md`、`Ideas/SyntheticData-VLA-Adaptation.md`、`Ideas/InContextLearning-VLA-Efficiency.md`

---

**示例 2：从 Supervisor 直接指令生成 idea**

```
想个 idea：怎么让 VLA 在 few-shot 场景下更好
```

执行过程：

1. 解析 Supervisor 意图：任务 = VLA，场景约束 = few-shot，方向 = 性能改进（success rate / sample efficiency）
2. Grep `Papers/` 搜索 `few-shot`、`VLA`，了解 vault 中已有哪些相关笔记
3. Glob 检查 `Workbench/memory/failed-directions.md` 是否存在；Read 并记录废弃方向
4. Read `DomainMaps/_index.md` → Read `DomainMaps/VLA.md`，提取 Active Debates 和 Open Questions
5. 生成 2-3 个候选 idea，每个聚焦不同技术路径（如 adapter fine-tuning、retrieval augmentation、meta-learning）
6. Glob `Ideas/` 检查无重复
7. Read `Templates/Idea.md`，Write 每个 idea 文件到 `Ideas/`
8. 追加日志到 `Workbench/logs/2026-03-28.md`

输出文件：`Ideas/` 下新增 2-3 个 idea 文件
