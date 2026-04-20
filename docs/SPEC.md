# MindFlow Specification

> 本文件是 MindFlow 的 **single source of truth**，记录系统的当前设计和约定。

**Last updated**: 2026-04-01

---

## 1. What is MindFlow

MindFlow 是一个基于 Obsidian 的 **Supervisor-Researcher 协作科研知识管理系统**。它将论文阅读、idea 孵化、实验追踪、记忆蒸馏等科研工作流编码为可执行的 Markdown skill，在 vault 内直接执行。

### 角色定位

MindFlow 采用 **Researcher - Supervisor**模型：

- **Researcher（AI）**：有自己的研究议程，独立驱动日常工作——读论文、跑实验、写初稿、调整方向。大多数决策自主完成。
- **Supervisor（Human）**：设定高层研究方向，定期 check-in，给战略性建议。可以 redirect、veto、suggest——但不微观管理。

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
│  Papers/ Topics/ Ideas/ DomainMaps/              │
│  Workbench/ (Researcher working state)           │
└─────────────────────────────────────────────────┘
```

**Layer 1（核心）**：纯 Markdown skill + vault 模板 + 协议文档。零依赖，任何支持文件读写的 AI agent 均可执行。当前已实现。

**Layer 2（可选）** 🔮：当需要 Researcher 完全自主运行时启用。包括 scheduler、向量记忆检索、推送通知、统一 agent 抽象。Layer 2 只读写 vault Markdown 文件，不引入 Layer 1 不知道的状态。

### 交互方式

Researcher 默认独立工作。Supervisor 随时可以 drop in 给指令或讨论。Researcher 在有重要发现时主动汇报。

## 3. Directory Structure

Vault 中的内容分为两类：

- **Knowledge Assets**（`Papers/`、`Topics/`、`Ideas/`、`Projects/`、`Reports/`）：完成的产出物，Supervisor-Researcher 共享
- **Researcher Working State**（`Workbench/`）：过程产物——议程、记忆、队列、日志

```
MindFlow/
│
├── Papers/              # 论文笔记（YYMM-ShortTitle.md）
├── Ideas/               # 研究 idea（status: raw → developing → validated → archived）
├── Projects/            # 项目追踪（status: planning → active → paused → completed）
├── Topics/              # 文献调研 / 跨论文分析报告
├── Reports/             # Researcher 生成的报告
├── Meetings/            # 会议记录（YYYY-MM-DD-Description.md）
│
├── DomainMaps/          # 核心认知地图（按 domain 拆分）
│   ├── _index.md        #   索引页：domain 列表 + cross-domain insights
│   ├── {CamelCase}.md   #   各 domain 的认知地图
│
├── Templates/           # Obsidian 模板（Paper.md, Idea.md, ...）
│
├── skills/              # Skill 定义（详见 §4.3）
│   ├── 1-literature/    #   文献技能
│   ├── 2-ideation/      #   创意技能
│   ├── 3-experiment/    #   实验技能
│   ├── 4-writing/       #   写作技能
│   ├── 5-evolution/     #   进化技能
│   └── 6-orchestration/ #   编排技能
│
├── references/          # 协议文档
│   ├── skill-protocol.md
│   ├── memory-protocol.md
│   ├── agenda-protocol.md
│   └── tags.md
│
├── Workbench/           # Researcher 工作状态（Supervisor 可随时查看和编辑）
│   ├── agenda.md        #   研究议程
│   ├── memory/          #   蒸馏后的记忆（patterns, insights, ...）
│   ├── queue.md         #   待办队列（Reading, Review, Questions, Experiments）
│   ├── logs/            #   每日操作日志（YYYY-MM-DD.md）
│   └── evolution/       #   演化记录（changelog.md）
│
├── examples/            # 示例文件
│
├── docs/SPEC.md         # ★ 本文件
├── AGENTS.md            # Researcher 身份与操作指令（CLAUDE.md 为其 symlink）
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
| 文献调研 | `Topics/` | `{Topic}-Survey.md` | `Templates/Survey.md` |
| 会议 | `Meetings/` | `YYYY-MM-DD-Description.md` | `Templates/Meeting.md` |
| 报告 | `Reports/` | 描述性名称 | `Templates/Report.md` |

### 4.2 Domain Map

Domain Map 是 vault 中**层级最高的知识**——从所有 Papers/Topics/Ideas/Experiments 中蒸馏而来的核心认知。

**结构**：`DomainMaps/` 目录，每个研究 domain 一个文件，包含四个象限：

| 象限 | 含义 |
|:-----|:-----|
| **Established Knowledge** | 高置信度的领域共识，附来源论文 |
| **Active Debates** | 存在矛盾或未定论的观点 |
| **Open Questions** | 尚未回答的问题 |
| **Known Dead Ends** | 已证伪或不推荐的方向 |

Researcher 自由维护 Domain Map。

### 4.3 Skills

Skill 是 MindFlow 的自动化核心——定义在 `skills/<category>/<name>/SKILL.md` 中的可执行能力单元。

**格式**：YAML frontmatter（元数据）+ Purpose + Steps + Guard + Verify + Examples。

创建新 skill 或了解格式详情 → `references/skill-protocol.md`

**架构**：核心循环 + 卫星 skill。`autoresearch`（L2 编排）读取 agenda/memory/queue 状态，判断下一个最高价值行动，调用对应卫星 skill。每个卫星 skill 也可被 Supervisor 自然语言直接触发。

```
               ┌─── autoresearch (L2) ───┐
               │   读状态 → 判断 → 执行    │
               └────┬──┬──┬──┬──┬────────┘
                    │  │  │  │  │
        ┌───────────┘  │  │  │  └───────────┐
        ▼              ▼  ▼  ▼              ▼
   1-literature   2-ideation  3-experiment  4-writing
                       │         │
                       ▼         ▼
                  5-evolution → 更新 Workbench/ → 下一轮
```

**完整 skill 清单**（13 个）：

| Category | Skill | Level | 功能 | 状态 |
|:---------|:------|:------|:-----|:-----|
| `1-literature` | `paper-digest` | L0 | 消化单篇论文 → Paper 笔记 | ✅ |
| | `literature-survey` | L1 | 主题级调研（搜索 + 批量 digest + 综合） | ✅ |
| `2-ideation` | `idea-generate` | L0 | 从知识空白生成研究 idea | ✅ |
| | `idea-evaluate` | L0 | 评估 idea 可行性和新颖性 | ✅ |
| `3-experiment` | `experiment-design` | L0 | 设计实验方案 | ✅ |
| | `experiment-track` | L0 | 记录实验进展和结果 | ✅ |
| | `result-analysis` | L0 | 分析实验结果，提取 insight | ✅ |
| `4-writing` | `draft-section` | L0 | 起草论文/报告章节 | ✅ |
| | `writing-refine` | L0 | 打磨已有文稿 | ✅ |
| `5-evolution` | `memory-distill` | L2 | 从日志蒸馏记忆 | ✅ |
| | `agenda-evolve` | L2 | 演化研究议程 | ✅ |
| | `memory-retrieve` | L0 | 从记忆库检索相关经验 | ✅ |
| `6-orchestration` | `autoresearch` | L2 | 核心研究循环（读状态→判断→执行→积累） | ✅ |

详细设计 → `docs/superpowers/specs/2026-03-28-skill-system-design.md`

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
L4: Domain Map      DomainMaps/{Name}.md             持久领域知识
```

详见 → `references/memory-protocol.md`

### 4.5 Workbench

`Workbench/` 是 Researcher 的工作状态目录，Supervisor 可随时查看和编辑：

| 文件/目录 | 职责 |
|:----------|:-----|
| `agenda.md` | 研究议程（active/paused/abandoned directions） |
| `memory/` | 蒸馏后的记忆文件 |
| `queue.md` | 待办队列（Reading、Review、Questions、Experiments） |
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
- Domain Map: `DomainMaps/{Name}.md`（CamelCase）
- Meetings: `YYYY-MM-DD-Description.md`
- Logs: `YYYY-MM-DD.md`
- Skills: `skills/{N}-{category}/{kebab-case-name}/SKILL.md`

### Wikilinks
- 笔记间引用使用 `[[wikilinks]]`
- 带 alias：`[[2410-Pi0|π₀]]`
- 在 Markdown `*` 可能被误解析时转义：`π\*₀.₆`

### Tags
- 详见 → `references/tags.md` 

## 6. Protocols

| 协议 | 文件 | 管辖范围 |
|:-----|:-----|:---------|
| Skill Protocol | `references/skill-protocol.md` | SKILL.md 格式、frontmatter 字段、skill levels |
| Memory Protocol | `references/memory-protocol.md` | 记忆文件格式、L0-L4 晋升规则、更新规则 |
| Agenda Protocol | `references/agenda-protocol.md` | agenda.md 格式、Researcher 权限、Supervisor override |
| Tag Taxonomy | `references/tags.md` | Tag 列表、选择原则、更新记录 |

## 7. Roadmap

| Phase | 内容 | 状态 |
|:------|:-----|:-----|
| 1. Skeleton | 仓库结构、协议文档、paper-digest / memory-distill / literature-survey | ✅ Done |
| 1.5 Protocol | skill-protocol 改造（pushy description、Verify 节、budget 字段）+ 现有 skill 改造 | 🔮 |
| 2. Core Loop | memory-retrieve、idea-generate、idea-evaluate、agenda-evolve | ✅ Done |
| 3. Experiment | experiment-design、experiment-track、result-analysis | ✅ Done |
| 4. Writing | draft-section、writing-refine | ✅ Done |
| 5. Orchestration | autoresearch（依赖 Phase 2-4 所有卫星 skill） | ✅ Done |
| 6. Orchestrator | daemon、scheduler、向量检索、notifier、agent-bridge（Layer 2） | 🔮 |

## 8. Changelog

| 日期 | 变更 |
|:-----|:-----|
| 2026-04-01 | 移除 Daily/ 和 Experiments/ 目录；移除 cross-paper-analysis skill；CLAUDE.md 改为 AGENTS.md 的 symlink，身份信息集中到 AGENTS.md；精简 paper-digest 和 literature-survey；Paper 模板移除 Connections 章节 |
| 2026-03-28 | Skill System Design：14 skill 全景图（核心循环 + 卫星架构）、protocol 改造（Verify/pushy desc/budget）、Roadmap 重排 Phase 1.5-6 |
| 2026-03-27 | Spec 精简至 ~250 行；补全目录结构（dist/docs/examples/website）、笔记类型表（Report）；明确 skill 编号约定；移除未定义缩写 |
| 2026-03-27 | Spec 精简：从 ~600 行删减至 ~250 行。移除过度形式化的交互模式、Evolution 机制命名、insight-loop 详细实现、未实现的技术选型/仓库结构/Design Provenance。核心内容保留在协议文档和 design spec archive 中。 |
| 2026-03-27 | 角色模型重定位：Human→Supervisor, AI→Researcher, PhD 导师制。移除所有 NEED APPROVAL 限制。 |
| 2026-03-27 | 合并 design spec 内容、Domain Map 迁移至 `DomainMaps/`、新增 SPEC.md |
| 2026-03-26 | 初始 vault 结构搭建 |
