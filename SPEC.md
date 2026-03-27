# MindFlow Specification

> 本文件是 MindFlow 的 **single source of truth**，记录系统的当前设计和约定。
> CLAUDE.md 引用本文件提供结构信息，自身聚焦 Researcher 操作指令。

**Last updated**: 2026-03-27

---

## 1. What is MindFlow

MindFlow 是一个基于 Obsidian 的 **Supervisor-Researcher 协作科研知识管理系统**。它将论文阅读、idea 孵化、实验追踪、记忆蒸馏等科研工作流编码为可执行的 Markdown skill，在 vault 内直接执行。

### 角色定位

MindFlow 采用 **PhD 导师制**模型：

- **Researcher（AI）**= PhD 学生：有自己的研究议程，独立驱动日常工作——读论文、跑实验、写初稿、调整方向。大多数决策自主完成。
- **Supervisor（Human）**= 导师：设定高层研究方向，定期 check-in，给战略性建议。可以 redirect、veto、suggest——但不微观管理。

### 设计哲学

```
Insight  — 目标不是论文数量或 metric 提升，而是 "我们理解了什么新东西？"
Trust    — 透明 → 可审计 → 信任。
Markdown — 一切皆文件，一切可读，一切有版本控制。
```

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
│  Workbench/ (Researcher working state)           │
└─────────────────────────────────────────────────┘
```

**Layer 1（核心）**：纯 Markdown skill + vault 模板 + 协议文档。零依赖，任何支持文件读写的 AI agent 均可执行。当前已实现。

**Layer 2（可选）** 🔮：当需要 Researcher 完全自主运行时启用。包括 scheduler、向量记忆检索、推送通知、统一 agent 抽象。Layer 2 只读写 vault Markdown 文件，不引入 Layer 1 不知道的状态。

### 交互方式

Researcher 默认独立工作。Supervisor 随时可以 drop in 给指令或讨论。Researcher 在有重要发现时主动汇报。

## 3. Directory Structure

Vault 中的内容分为两类：

- **Knowledge Assets**（`Papers/`、`Topics/`、`Ideas/`、`Experiments/`、`Reports/`）：完成的产出物，Supervisor-Researcher 共享
- **Researcher Working State**（`Workbench/`）：过程产物——议程、记忆、队列、日志

```
MindFlow/
│
├── Papers/              # 论文笔记（YYMM-ShortTitle.md）
├── Ideas/               # 研究 idea（status: raw → developing → validated → archived）
├── Projects/            # 项目追踪（status: planning → active → paused → completed）
├── Topics/              # 文献调研 / 跨论文分析报告
├── Experiments/         # 实验记录
├── Reports/             # Researcher 生成的报告
├── Meetings/            # 会议记录（YYYY-MM-DD-Description.md）
├── Daily/               # 每日研究日志（YYYY-MM-DD.md）
│
├── Domain-Map/          # 核心认知地图（按 domain 拆分）
│   ├── _index.md        #   索引页：domain 列表 + cross-domain insights
│   ├── {Name}.md #   各 domain 的四象限认知地图
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
├── Workbench/           # Researcher 工作状态（Supervisor 可随时查看和编辑）
│   ├── agenda.md        #   研究议程
│   ├── identity.md      #   Researcher 身份与配置
│   ├── memory/          #   蒸馏后的记忆（patterns, insights, ...）
│   ├── queue/           #   待办队列（reading, review, questions）
│   ├── logs/            #   每日操作日志（YYYY-MM-DD.md）
│   └── evolution/       #   演化记录（changelog.md）
│
├── SPEC.md              # ★ 本文件
├── CLAUDE.md            # Researcher 操作指令
└── .obsidian/           # Obsidian 配置
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

Researcher 自由维护 Domain Map。

### 4.3 Skills

Skill 是 MindFlow 的自动化核心——定义在 `skills/<category>/<name>/SKILL.md` 中的可执行能力单元。

**格式**：YAML frontmatter（元数据）+ Purpose + Steps + Guard + Examples。详见 → `references/skill-protocol.md`

**当前 skills**：

| Skill | 目录 | 功能 |
|:------|:-----|:-----|
| `paper-digest` | `skills/1-literature/` | 消化单篇论文 → Paper 笔记 |
| `cross-paper-analysis` | `skills/1-literature/` | 跨论文对比分析 → 分析报告 |
| `literature-survey` | `skills/1-literature/` | 主题级文献调研（搜索 + 批量 digest + 综合） |
| `memory-distill` | `skills/5-evolution/` | 从日志蒸馏记忆（patterns → insights） |

### 4.4 Memory System

Researcher 的经验通过五级层级逐步蒸馏：

```
L0: Raw Log        Workbench/logs/YYYY-MM-DD.md     每次操作自动记录
     ↓ memory-distill 提取
L1: Pattern         Workbench/memory/patterns.md     跨日期重复出现的观察
     ↓ 多次独立观察
L2: Provisional     Workbench/memory/insights.md     初步洞察（待验证）
     ↓ 实验/文献验证
L3: Validated       Workbench/memory/insights.md     已验证洞察
     ↓ Researcher 判断证据充分时自主晋升
L4: Domain Map      Domain-Map/{Name}.md             持久领域知识
```

详见 → `references/memory-protocol.md`

### 4.5 Workbench

`Workbench/` 是 Researcher 的工作状态目录，Supervisor 可随时查看和编辑：

| 文件/目录 | 职责 |
|:----------|:-----|
| `agenda.md` | 研究议程（active/paused/abandoned directions） |
| `identity.md` | Researcher 身份与配置 |
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
- 详见 → `references/tag-taxonomy.md` 

## 6. Protocols

| 协议 | 文件 | 管辖范围 |
|:-----|:-----|:---------|
| Skill Protocol | `references/skill-protocol.md` | SKILL.md 格式、frontmatter 字段、skill levels |
| Memory Protocol | `references/memory-protocol.md` | 记忆文件格式、L0-L4 晋升规则、更新规则 |
| Agenda Protocol | `references/agenda-protocol.md` | agenda.md 格式、Researcher 权限、Supervisor override |
| Tag Taxonomy | `references/tag-taxonomy.md` | Tag 列表、选择原则、更新记录 |

## 7. Roadmap

| Phase           | 内容                                                                                 | 状态     |
| :-------------- | :--------------------------------------------------------------------------------- | :----- |
| 1. Skeleton     | 仓库结构、协议文档、paper-digest / cross-paper-analysis / memory-distill / literature-survey | ✅ Done |
| 2. Core Loop    | insight-loop、agenda-evolve、memory-retrieve、idea-generation、IDE/IVE/ESE             | 🔮     |
| 3. Experiment   | experiment-design、experiment-iterate、result-analysis                               | 🔮     |
| 4. Orchestrator | daemon、scheduler、向量检索、notifier、agent-bridge                                        | 🔮     |
| 5. Polish       | writing skills、完整文档、社区贡献指南、v0.1.0                                                  | 🔮     |

## 8. Changelog

| 日期 | 变更 |
|:-----|:-----|
| 2026-03-27 | Spec 精简：从 ~600 行删减至 ~250 行。移除过度形式化的交互模式、Evolution 机制命名、insight-loop 详细实现、未实现的技术选型/仓库结构/Design Provenance。核心内容保留在协议文档和 design spec archive 中。 |
| 2026-03-27 | 角色模型重定位：Human→Supervisor, AI→Researcher, PhD 导师制。移除所有 NEED APPROVAL 限制。 |
| 2026-03-27 | 合并 design spec 内容、Domain Map 迁移至 `Domain-Map/`、新增 SPEC.md |
| 2026-03-26 | 初始 vault 结构搭建 |
