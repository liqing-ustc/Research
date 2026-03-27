# MindFlow Specification

> 本文件是 MindFlow 的 **single source of truth**，记录系统的当前设计和约定。
> CLAUDE.md 引用本文件提供结构信息，自身聚焦 AI 操作指令。

**Last updated**: 2026-03-27

---

## 1. What is MindFlow

MindFlow 是一个基于 Obsidian 的 **Human-AI 协作科研知识管理系统**。它将论文阅读、idea 孵化、实验追踪、记忆蒸馏等科研工作流编码为可执行的 Markdown skill，通过 Claude Code 在 vault 内直接执行，实现 Human 和 AI 在同一知识库上的无缝协作。

### 设计哲学

```
Insight  — 目标不是论文数量或 metric 提升，而是 "我们理解了什么新东西？"
Trust    — 透明 → 可审计 → 信任 → 更大自主权。AI 通过持续产出高质量 insight 赢得信任。
Markdown — 一切皆文件，一切可读，一切有版本控制。
```

### 核心差异化

| 现有 AutoResearch 的假设 | MindFlow 的假设 |
|:-------------------------|:----------------|
| 目标是**论文产出** | 目标是 **insight 发现与认知成长**（论文是副产物） |
| AI 是**执行者**，Human 下命令 | AI 有**自己的研究议程**，自主探索 |
| 单次运行，无跨项目学习 | **持久记忆 + 进化**，经验跨项目传递 |
| 固定角色（AI 做 X，Human 做 Y） | **角色流动** — AI 在不同阶段扮演不同角色 |
| 状态存储在数据库 / 向量库 | **一切在 Markdown**，Human 随时可审计 AI 的任何状态 |

### 范围边界

- **In scope**：文献发现/消化、idea 孵化、实验设计/执行、写作（作为副产物）、记忆/进化、自主研究议程
- **Out of scope**：训练自定义 LLM（仅 API 调用）、GPU 集群管理（实验在用户自己的环境运行）、实时多人协作（Google Docs 模式）

## 2. Architecture

### 双层设计

```
┌─────────────────────────────────────────────────┐
│  Layer 2: Orchestrator (optional) 🔮 Planned     │
│  Scheduler · Memory Index · Notifier ·           │
│  Agent Bridge                                    │
├─────────────────────────────────────────────────┤
│  Layer 1: Skill Protocol (core) ✅ Implemented   │
│  skills/*.md · Workbench/ · Templates/           │
│  Zero dependency, any agent can execute          │
├─────────────────────────────────────────────────┤
│  Obsidian Vault (Markdown)                       │
│  Papers/ Topics/ Ideas/ Domain-Map/              │
│  Workbench/ (AI working state)                   │
└─────────────────────────────────────────────────┘
```

**Layer 1（核心）**：纯 Markdown skill + vault 模板 + 协议文档。零依赖，任何支持文件读写的 AI agent 均可执行。当前已实现。

**Layer 2（可选）** 🔮：当需要 AI 完全自主运行时启用。包括：
- `scheduler/` — Cron / 事件驱动的任务调度
- `memory_index/` — 基于向量的记忆检索（从 Markdown 可重建）
- `notifier/` — 推送通知（Telegram / Email 等）
- `agent_bridge/` — 统一 agent 抽象（Claude Code / Codex / Gemini CLI）

**接口契约**：Layer 2 只读写 vault Markdown 文件，不引入 Layer 1 不知道的状态。同一个 skill 无论由 Human 手动触发还是 Layer 2 调度，行为完全一致。

```
Layer 2 写入：
  Workbench/queue/reading.md      ← scheduler 发现新论文
  Workbench/logs/YYYY-MM-DD.md    ← 执行日志
  Reports/YYYY-MM-DD-*.md         ← 定期报告

Layer 2 读取：
  Workbench/agenda.md             ← 决定下一步做什么
  Workbench/memory/*.md           ← 检索相关经验
  Workbench/queue/*.md            ← 获取待办任务
```

### 四种角色模式

| 模式 | AI 角色 | Human 角色 | 触发条件 | 状态 |
|:-----|:--------|:-----------|:---------|:-----|
| **Copilot** | 执行具体任务 | 在线，给指令 | Human 发起 + 明确命令 | ✅ |
| **Autopilot** | 自主探索 | 离线，事后审阅 | Human 离线 + agenda 有活跃方向 | ✅ 部分 |
| **Sparring** | 辩论伙伴 | 在线，讨论 idea | Human 发起 + 开放性问题 | 🔮 |
| **Reporter** | 结构化汇报 | 离线，异步审阅 | 定期 / 重大发现 / 需要决策 | 🔮 |

**模式切换逻辑**：模式是**隐式**的——由交互上下文决定，不需要显式配置。

- Human 在线 + 明确命令 → **Copilot**
- Human 在线 + 开放性问题/讨论 → **Sparring**
- Human 离线 + agenda 有活跃方向 → **Autopilot**
- Autopilot 过程中发现重要结果 → 切换到 **Reporter**，生成报告

例外：Autopilot 的权限边界在 `identity.md` 中显式定义（涉及信任边界）。

**Report 格式** 🔮：

```markdown
# Reports/YYYY-MM-DD-{type}.md
---
type: weekly / discovery / decision-needed
period: YYYY-MM-DD ~ YYYY-MM-DD
---
## Highlights
[Top 1-3 findings with evidence links]

## Progress by Direction
### Direction A
- **Actions taken**: ...
- **Key findings**: ...
- **Needs Human decision**: [yes/no]

## New Discoveries
[Unexpected patterns / notable new papers]

## Questions for Human
1. [Questions requiring Human judgment]

## Resource Usage
- Papers read: N / Experiments run: N / API tokens: ~N
```

## 3. Directory Structure

### Knowledge Assets vs AI Working State

Vault 中的内容分为两类：

- **Knowledge Assets**（`Papers/`、`Topics/`、`Ideas/`、`Experiments/`、`Reports/`）：完成的产出物，Human-AI 共享，vault 的持久价值
- **AI Working State**（`Workbench/`）：AI 的过程产物——议程、记忆、队列、日志。Human 可随时查看/编辑，但这些是过程性的，不是精炼的知识

**规则**：AI 的完成产出放入 Knowledge Asset 目录，中间工作放入 Workbench。

### 目录布局

```
MindFlow/
│
├── Papers/              # 论文笔记（YYMM-ShortTitle.md）
├── Ideas/               # 研究 idea（status: raw → developing → validated → archived）
├── Projects/            # 项目追踪（status: planning → active → paused → completed）
├── Topics/              # 文献调研 / 跨论文分析报告
├── Experiments/         # 实验记录
├── Reports/             # AI 生成的报告
├── Meetings/            # 会议记录（YYYY-MM-DD-Description.md）
├── Daily/               # 每日研究日志（YYYY-MM-DD.md）
│
├── Domain-Map/          # 核心认知地图（按 domain 拆分）
│   ├── _index.md        #   索引页：domain 列表 + cross-domain insights
│   ├── VLA.md           #   各 domain 的四象限认知地图
│   ├── VLN.md
│   └── SpatialRep.md
│
├── Templates/           # Obsidian 模板（Paper.md, Idea.md, ...）
├── Attachments/         # 文件附件
│
├── skills/              # Skill 定义
│   ├── 1-literature/    #   文献类（paper-digest, cross-paper-analysis, literature-survey）
│   └── 5-evolution/     #   进化类（memory-distill）
│
├── references/          # 协议文档
│   ├── skill-protocol.md
│   ├── memory-protocol.md
│   ├── agenda-protocol.md
│   └── tag-taxonomy.md
│
├── Workbench/           # AI 工作状态（Human 可随时查看和编辑）
│   ├── agenda.md        #   研究议程
│   ├── identity.md      #   AI 身份与权限配置
│   ├── memory/          #   蒸馏后的记忆（patterns, insights, ...）
│   ├── queue/           #   待办队列（reading, review, questions）
│   ├── logs/            #   每日操作日志（YYYY-MM-DD.md）
│   └── evolution/       #   演化记录（changelog.md）
│
├── SPEC.md              # ★ 本文件：系统设计 single source of truth
├── CLAUDE.md            # AI 操作指令（引用 SPEC.md）
└── .obsidian/           # Obsidian 配置
```

### 信息流

```
                    Human input
                    (read papers, write ideas, edit Domain-Map, ask questions)
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│              Knowledge Assets (Human-AI shared)       │
│                                                       │
│  ★ Domain-Map/  ← 核心共享认知                        │
│  Papers/ Topics/ Ideas/ Experiments/ Reports/         │
└────────────────────┬─────────────────────────────────┘
                     │
          knowledge ←→ AI state 双向流动
                     │
┌────────────────────┴─────────────────────────────────┐
│              Workbench/ (AI working state)             │
│  agenda.md → 驱动 AI 下一步行动                        │
│  memory/   → 积累的经验                               │
│  queue/    → 任务队列（Human 和 AI 均可写入）           │
│  logs/     → 原始操作日志                              │
│  evolution/→ 变更追踪                                  │
└──────────────────────────────────────────────────────┘
```

## 4. Key Concepts

### 4.1 Notes

所有笔记遵循 `Templates/` 中对应的模板。共通约定：
- YAML frontmatter 存储结构化元数据
- 正文用中文撰写，英文技术术语保持英文不翻译
- 笔记之间通过 `[[wikilinks]]` 建立连接

| 类型 | 目录 | 命名规则 | 模板 |
|:-----|:-----|:---------|:-----|
| 论文笔记 | `Papers/` | `YYMM-ShortTitle.md`（YYMM 取自 date_publish） | `Templates/Paper.md` |
| 研究 Idea | `Ideas/` | 描述性名称 | `Templates/Idea.md` |
| 项目 | `Projects/` | 描述性名称 | `Templates/Project.md` |
| 文献调研 | `Topics/` | `{Topic}-Survey.md` | `Templates/Topic.md` |
| 实验 | `Experiments/` | 描述性名称 | `Templates/Experiment.md` |
| 会议 | `Meetings/` | `YYYY-MM-DD-Description.md` | `Templates/Meeting.md` |
| 每日日志 | `Daily/` | `YYYY-MM-DD.md` | `Templates/Daily.md` |

### 4.2 Domain Map

Domain Map 是 vault 中**层级最高的知识**——从所有 Papers/Topics/Ideas/Experiments 中蒸馏而来的核心认知。

**结构**：`Domain-Map/` 目录，每个研究 domain 一个文件，包含四个象限：

| 象限 | 含义 |
|:-----|:-----|
| **Established Knowledge** | 高置信度的领域共识，附来源论文 |
| **Active Debates** | 存在矛盾或未定论的观点 |
| **Open Questions** | 尚未回答的问题 |
| **Known Dead Ends** | 已证伪或不推荐的方向 |

**治理原则**：Domain-Map 是 AI 的认知地图。AI 自由维护，Human 随时可编辑。Git 是安全网。

- AI 可自由添加、修改、晋升条目（任何象限）
- AI 可标记条目为 deprecated（~~删除线~~ + 原因），但不可删除
- 每次更新记录到 `Workbench/evolution/changelog.md`（由执行 skill 负责）
- Human 可随时编辑、重排、覆盖任何 AI 的修改

**新增 domain**：创建 `Domain-Map/{Name}.md`，复制四象限结构，在 `_index.md` 表格中添加一行。

### 4.3 Skills

Skill 是 MindFlow 的自动化核心——定义在 `skills/<category>/<name>/SKILL.md` 中的可执行能力单元。

**格式**：YAML frontmatter（元数据）+ Purpose + Steps + Guard + Examples。详见 → `references/skill-protocol.md`

**三级层次**：

| Level | 名称 | 说明 | 示例 |
|:------|:-----|:-----|:-----|
| Level 0 | Atomic | 单一任务，直接调用工具，不调用其他 skill | `paper-digest`, `memory-distill` |
| Level 1 | Orchestration | 组合多个 Level 0 skill 完成复合目标 | `cross-paper-analysis`, `literature-survey` |
| Level 2 | Global | 跨 vault 级别操作，可调用 Level 1 skill，管理 agenda/memory | `insight-loop` 🔮 |

**Stage-Skill 路由**：`skills/stage-skill-map.json` 定义了研究阶段 × 任务类型 → skill 的映射关系。

**当前 skills**：

| Skill | 目录 | 功能 |
|:------|:-----|:-----|
| `paper-digest` | `skills/1-literature/` | 消化单篇论文 → Paper 笔记 |
| `cross-paper-analysis` | `skills/1-literature/` | 跨论文对比分析 → 分析报告 |
| `literature-survey` | `skills/1-literature/` | 主题级文献调研（搜索 + 批量 digest + 综合） |
| `memory-distill` | `skills/5-evolution/` | 从日志蒸馏记忆（patterns → insights） |

**触发方式**：自然语言触发，AI 识别意图后读取对应 SKILL.md 并严格执行。

### 4.4 Memory System

AI 的经验通过五级层级逐步蒸馏：

```
L0: Raw Log        Workbench/logs/YYYY-MM-DD.md     每次操作自动记录
     ↓ memory-distill 提取
L1: Pattern         Workbench/memory/patterns.md     跨日期重复出现的观察
     ↓ ≥2 独立来源
L2: Provisional     Workbench/memory/insights.md     初步洞察（待验证）
     ↓ 实验/文献验证
L3: Validated       Workbench/memory/insights.md     已验证洞察
     ↓ AI 判断证据充分时自主晋升
L4: Domain Map      Domain-Map/{Name}.md             持久领域知识
```

**核心规则**：
- **Append-only**：记忆文件只追加不修改
- **来源引用必须明确**：每条 pattern/insight 必须包含指向具体日志的 wikilink
- **Domain-Map logging**：每次更新 Domain Map 必须在 `Workbench/evolution/changelog.md` 中记录

**记忆检索** 🔮：
- Layer 1（文件检索）：Grep + Glob + LLM 判断。记忆条目 < 100 条时足够。
- Layer 2（向量加速）：mxbai-embed-large（Ollama 本地）→ ChromaDB → cosine similarity top-k。索引可从 Markdown 完全重建。索引不可用时回退到 Layer 1。

详见 → `references/memory-protocol.md`

### 4.5 Evolution Mechanisms 🔮 Planned

Memory System 的 L0→L4 晋升由三种进化机制驱动（改编自 EvoScientist）：

| 机制 | 全称 | 触发时机 | 输入 | 输出 |
|:-----|:-----|:---------|:-----|:-----|
| **IDE** | Insight Direction Evolution | cross-paper-analysis 或 knowledge-synthesis 完成后 | Topics/ 分析 + Papers/ | 新方向 → `memory/insights.md`（provisional） |
| **IVE** | Insight Validation Evolution | 研究方向被放弃时 | agenda.md 废弃方向 + 关联实验/论文 | 失败教训 → `memory/failed-directions.md` |
| **ESE** | Experiment Strategy Evolution | 实验分析完成后 | `Experiments/<id>/` 全套文件 | 有效方法 → `memory/effective-methods.md` |

### 4.6 Autopilot Core Loop (insight-loop) 🔮 Planned

Autopilot 模式下 AI 的核心执行循环，每次触发执行一个完整 cycle：

```
insight-loop (one cycle)
    │
    ├── Phase 1: Orient — 读取 agenda + queue + memory + Domain Map，决定下一步
    │   优先级：queue/review 紧急项 > Human 新增任务 > 待验证 insight
    │         > agenda 最高优先级方向 > Domain Map Open Questions > paper-discovery
    │
    ├── Phase 2: Act — 查询 stage-skill-map.json，调用对应 skill（每 cycle 一个 action）
    │   执行前检查 identity.md 权限边界；需审批的写入 queue/review.md 并跳过
    │
    ├── Phase 3: Learn — 记录日志 + 条件触发进化（IDE/IVE/ESE）+ 检查 insight 晋升
    │
    └── Phase 4: Report — 满足条件时生成 Report
        条件：高置信度 insight 晋升 / 重大矛盾 / 实验结果异常 / 需要 Human 决策
```

**触发方式**：
- Layer 1：Human 手动执行 `/insight-loop`
- Layer 2：Daemon scheduler 自动触发（可配置频率）

**错误处理**：
- Skill 失败 → 记录日志，跳过本 cycle，不重试相同输入
- API 超时 → 重试一次（60s 后），仍失败则跳过并写入 queue/review
- 部分状态 → 利用 git commit/revert 实现原子性
- 资源耗尽 → 进入 COMPACT mode（读摘要而非全文），仍失败则暂停 Autopilot 并触发 Reporter

**API 预算**（定义在 `Workbench/identity.md`）：

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `daily_token_limit` | 500k | 日 token 上限 |
| `per_cycle_limit` | 50k | 单 cycle 上限 |
| `expensive_action_threshold` | 100k | 超过此值需审批 |

接近日上限时进入"节约模式"（仅处理 Human 队列，跳过自主探索）。

**并发与冲突**（Layer 2 场景）：
- AI 写共享文件前先读取当前版本，写后检查 git diff
- 若文件在读写间被他人修改，AI 的版本写入 `.conflict` 文件并通知 Human
- Append-only 文件（logs、queue、memory）冲突极少
- Domain Map：AI 自由维护，Human 亦可自由编辑；冲突时以 git diff 检测，AI 版本写入 `.conflict` 文件

### 4.7 Workbench

`Workbench/` 是 AI 的工作状态目录，Human 可随时查看和编辑：

| 文件/目录 | 职责 |
|:----------|:-----|
| `agenda.md` | 研究议程（active/paused/abandoned directions） |
| `identity.md` | AI 身份、权限、预算配置 |
| `memory/` | 蒸馏后的记忆文件 |
| `queue/` | 待办队列（reading、review、questions） |
| `logs/` | 每日操作日志 |
| `evolution/` | 系统演化记录 |

详见 → `references/agenda-protocol.md`

## 5. Conventions

### 语言
- 正文用**中文**撰写
- 英文技术术语（模型名、方法名、benchmark 名）保持英文不翻译
- Frontmatter 字段名用英文

### 文件命名
- Papers: `YYMM-ShortTitle.md`（CamelCase，2-4 关键词）
- Domain Map: `Domain-Map/{Name}.md`（CamelCase）
- Meetings: `YYYY-MM-DD-Description.md`
- Logs: `YYYY-MM-DD.md`
- Skills: `skills/{N}-{category}/{kebab-case-name}/SKILL.md`

### Wikilinks
- 笔记间引用使用 `[[wikilinks]]`
- 带 alias：`[[2410-Pi0|π₀]]`
- 在 Markdown `*` 可能被误解析时转义：`π\*₀.₆`

### Tags
- 每篇笔记 1-3 个 tag
- 从 `references/tag-taxonomy.md` 中选取
- 需要新 tag 时遵循设计原则（粒度适中、正交、稳定）并更新 taxonomy

## 6. Protocols

| 协议 | 文件 | 管辖范围 |
|:-----|:-----|:---------|
| Skill Protocol | `references/skill-protocol.md` | SKILL.md 格式、frontmatter 字段、skill levels |
| Memory Protocol | `references/memory-protocol.md` | 记忆文件格式、L0-L4 晋升规则、更新规则 |
| Agenda Protocol | `references/agenda-protocol.md` | agenda.md 格式、AI 权限矩阵、Human override |
| Tag Taxonomy | `references/tag-taxonomy.md` | Tag 列表、选择原则、更新记录 |

## 7. AI Behavior

AI（Claude Code）通过 `CLAUDE.md` 接收操作指令。关键约束：

**权限边界**：
- **CAN**：读论文、更新记忆、生成报告、发现新论文、按 agenda 探索新方向、自由维护 Domain Map
- **NEED APPROVAL**：启动 >2h 实验、放弃研究方向
- **CANNOT**：删除已有笔记、修改 Human 撰写的内容、对外发布

**Skill 执行规则**：
- 每次执行 skill 必须先 Read 对应 SKILL.md，不凭记忆执行
- 严格遵守 SKILL.md 中的 Steps 和 Guard
- 所有操作记录到 `Workbench/logs/`

## 8. Technology Choices 🔮

Layer 2 Orchestrator 的技术选型：

| 组件 | 选择 | 理由 |
|:-----|:-----|:-----|
| Layer 1 skills | Pure Markdown | 零依赖，跨 agent 可移植 |
| Install CLI | Node.js (npx) | 最广泛的前端生态 |
| Layer 2 daemon | Python | 最成熟的 AI/ML 生态 |
| Vector embedding | mxbai-embed-large via Ollama | 本地运行，无 API 依赖 |
| Vector storage | ChromaDB | 轻量本地存储，可从 Markdown 完全重建 |
| Notifications | Apprise (Python) | 一个库覆盖 Telegram/Email/Feishu/DingTalk/Slack |
| Scheduling | APScheduler | Python 原生，支持 cron 和 interval |
| Agent invocation | Agent CLIs | Claude Code / Codex / Gemini CLI + 统一 wrapper |
| Config | YAML (orchestrator) + Markdown (vault) | YAML 给 daemon，Markdown 给 AI 和 Human |
| License | MIT | 最大化开源研究工具的采用 |

## 9. Repository Structure 🔮

开源发布时的仓库结构：

```
github.com/xxx/mindflow
├── README.md
├── LICENSE (MIT)
├── CONTRIBUTING.md
│
├── skills/                        # Layer 1
│   ├── taxonomy.schema.json
│   ├── stage-skill-map.json
│   ├── 0-orchestration/
│   │   └── insight-loop/
│   ├── 1-literature/
│   │   ├── paper-digest/
│   │   ├── cross-paper-analysis/
│   │   ├── literature-survey/
│   │   └── knowledge-synthesis/
│   ├── 2-ideation/
│   │   ├── idea-generation/
│   │   ├── idea-tournament/
│   │   └── literature-grounding/
│   ├── 3-experiment/
│   │   ├── experiment-design/
│   │   ├── experiment-iterate/
│   │   └── result-analysis/
│   ├── 4-writing/
│   │   ├── paper-outline/
│   │   ├── paper-draft/
│   │   └── paper-review/
│   └── 5-evolution/
│       ├── memory-distill/
│       ├── memory-retrieve/
│       └── agenda-evolve/
│
├── templates/
│   ├── paper.md
│   ├── idea.md
│   ├── experiment.md
│   ├── report.md
│   ├── domain-map.md
│   └── Workbench/ (init template)
│
├── references/
│   ├── skill-protocol.md
│   ├── memory-protocol.md
│   ├── agenda-protocol.md
│   ├── role-protocol.md
│   └── report-protocol.md
│
├── packages/
│   └── mindflow-cli/
│       ├── package.json
│       ├── bin/install.js
│       └── lib/
│
├── orchestrator/              # Layer 2
│   ├── requirements.txt
│   ├── daemon.py
│   ├── scheduler/
│   ├── memory_index/
│   ├── notifier/
│   ├── agent_bridge/
│   └── config.example.yaml
│
├── docs/
│   ├── getting-started.md
│   ├── skill-authoring.md
│   ├── orchestrator-setup.md
│   └── architecture.md
│
└── examples/
    └── demo-vault/
```

## 10. Roadmap

### Phase 1 — Skeleton ✅ Done

- 仓库结构 + 协议文档
- 3 核心 skills：paper-digest, cross-paper-analysis, memory-distill
- Templates 和 `Workbench/` 初始化
- `literature-survey` skill（额外完成）

### Phase 2 — Core Loop 🔮

- insight-loop orchestration skill
- agenda-evolve + memory-retrieve
- idea-generation + idea-tournament
- 完整 IDE/IVE/ESE evolution skills
- Domain-Map 更新协议实现

### Phase 3 — Experiment 🔮

- experiment-design + experiment-iterate
- result-analysis
- Guard mechanism
- Cross-model review（ARIS MCP 方式）

### Phase 4 — Orchestrator 🔮

- daemon + scheduler
- memory-index（向量检索）
- notifier（Telegram + Email）
- agent-bridge（Claude + Codex）
- Reporter mode 自动报告

### Phase 5 — Polish 🔮

- Writing skills（paper-outline / draft / review）
- 完整文档
- 社区贡献指南
- Release v0.1.0

## 11. Design Provenance

MindFlow 的设计吸收了多个开源框架的优秀实践：

| 组件 | 来源 | 如何采纳 |
|:-----|:-----|:---------|
| SKILL.md 格式 | ARIS | 采纳 frontmatter + allowed-tools 标准 |
| Skill + references/ 分离 | uditgoenka/autoresearch | 采纳用于复杂 skill |
| Atomic → Orchestration → Global 分层 | ARIS | 采纳三级层次 |
| NPX 一键安装 | Orchestra AI-Research-SKILLs | 采纳分发机制 |
| Taxonomy schema | Dr. Claw | 采纳 + 扩展 roles/autonomy 字段 |
| Stage × Task → Skill 路由 | Dr. Claw | 采纳 stage-skill-map.json 模式 |
| IDE/IVE/ESE 进化机制 | EvoScientist | 采纳，但使用 Markdown 存储而非向量库 |
| Idea Elo tournament | EvoScientist | 采纳用于 idea 排名 |
| 8 迭代原则 + Guard | uditgoenka/autoresearch | 采纳用于 experiment-iterate skill |
| Cross-model adversarial review | ARIS | 采纳 executor + reviewer 模式（via MCP） |
| Multi-agent backend 抽象 | Dr. Claw | 采纳 claude-sdk/gemini-cli/codex 模式 |

**MindFlow 原创贡献**（未在任何现有框架中发现）：
- AI 自管理研究议程（Research Agenda）
- Human-AI 角色流动（Autopilot / Copilot / Sparring / Reporter）
- 四级 Insight 晋升层级（log → pattern → insight → Domain-Map）
- 共享 Domain-Map 作为核心 Human-AI 认知
- Orient → Act → Learn → Report 自主循环
- 完全透明：所有 AI 状态存储在可审计的 Markdown 中

## 12. Changelog

| 日期 | 变更 | 影响范围 |
|:-----|:-----|:---------|
| 2026-03-27 | SPEC.md 合并 design spec 内容：新增设计哲学/差异化/范围边界（§1）、接口契约示例/模式切换逻辑（§2）、Knowledge Assets vs Working State/信息流（§3）、Domain-Map 治理简化为"AI 自由维护"（§4.2）、skill 三级层次（§4.3）、记忆检索策略（§4.4）、Technology Choices（§8）、Repository Structure（§9）、Roadmap（§10）、Design Provenance（§11） | 全局 |
| 2026-03-27 | Domain Map 从 `Topics/` 迁移到 vault 根目录 `Domain-Map/`，按 domain 拆分为独立文件 | CLAUDE.md, skills, references |
| 2026-03-27 | `Templates/Paper.md` 扩充为 single source of truth（含 `%%` 注释指导），`paper-digest` SKILL.md Step 4 精简为引用模板 | Templates/, skills/ |
| 2026-03-27 | 新增 SPEC.md 作为系统设计的 single source of truth；补充 Layer 2、Role Fluidity、insight-loop、Evolution Mechanisms 等未实现设计 | vault root |
| 2026-03-26 | 初始 vault 结构搭建，skill 系统、记忆协议、议程协议就位 | 全局 |
