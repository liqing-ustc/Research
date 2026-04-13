---
title: AutoResearch 生态 Skill 设计深度分析
tags:
  - auto-research
status: active
date_updated: 2026-03-28
---

## Overview

本文对 AutoResearch.md 中覆盖的 **9 个开源项目** 的 skill/能力单元设计进行深度拆解，从 **定义格式、组织方式、执行协议、状态管理、质量守卫** 五个维度横向对比，提炼出可复用的设计 pattern。

---

## 一、逐项目 Skill 设计深度分析

### 1. Karpathy autoresearch — 极简三文件协议

**Skill 载体**：不叫 "skill"，但 `program.md` 本质就是一个 skill 文件——自然语言指令，定义 agent 的行为边界。

**三文件架构即协议**：

| 文件 | 角色 | 可变性 |
|------|------|--------|
| `prepare.py` | 数据准备 + 评估逻辑 | **不可变**（冻结评估标准） |
| `train.py` | 模型 + 训练循环 | **Agent 唯一可改文件** |
| `program.md` | 自然语言指令 | **人类唯一编辑入口** |

**核心设计 pattern**：
- **沙盒隔离**：评估代码（prepare.py）与实验代码（train.py）物理隔离，杜绝 agent 通过修改评估逻辑"作弊"
- **固定时间窗口**：每次实验严格 5 分钟，不同架构在同一标尺下公平对比
- **单指标棘轮**：val_bpb（bits per byte），词表无关，方向单一（越低越好）
- **协议极简性**：整个 "skill" 就是一个 Markdown 文件 + 两个 Python 文件，~630 行总代码

**状态管理**：实验日志（results log），无 git-as-memory，无持久化记忆

**Guard Rail**：
- 文件系统级约束（只能改 train.py）
- 固定 5 分钟超时
- 评估代码冻结

**设计哲学**：**最小可行协议**——用文件系统的读写权限实现 agent 行为约束，零框架开销。

---

### 2. AutoResearchClaw — 23 阶段流水线协议

**Skill 载体**：23 个 stage 即 23 个隐式 skill，每个 stage 有专用 prompt（定义在 `prompts.default.yaml`）。

**架构层次**：

```
config.arc.yaml          ← 全局配置（模型、预算、审批门）
prompts.default.yaml     ← 23 stage-specific system prompts
RESEARCHCLAW_AGENTS.md   ← Agent 编排逻辑
RESEARCHCLAW_CLAUDE.md   ← Claude Code 集成入口
~/.metaclaw/skills/      ← MetaClaw 学习型 skill（运行时生成）
```

**23 阶段 = 8 相位**：

| 相位 | 阶段 | 核心能力 |
|------|------|---------|
| A: Scoping | 1-2 | 问题树分解 |
| B: Literature | 3-6 | 多源检索 + 相关性筛选 |
| C: Synthesis | 7-8 | 聚类 + 假设生成（辩论） |
| D: Design | 9-11 | 实验设计 + 硬件感知代码生成 |
| E: Execution | 12-13 | 沙盒执行 + NaN 检测 + 自修复 |
| F: Decision | 14-15 | 多 Agent 分析 → PROCEED/REFINE/PIVOT |
| G: Writing | 16-19 | 大纲→草稿→评审→修订 |
| H: Finalize | 20-23 | 质量门 + 知识归档 + LaTeX 导出 + 引用校验 |

**Multi-Agent 辩论协议**：
- 假设生成、结果分析、同行评审三环节均用 structured multi-perspective debate
- 不是自我博弈，而是多个 LLM 视角对抗

**6 类知识库**：decisions / experiments / findings / literature / questions / reviews，30 天时间衰减

**MetaClaw 自学习 Skill**：
- 每次运行提取最多 3 个 skill（从失败中学习）
- 存储格式：`Severity → Stage → Pattern → Fix → Time-Decay`
- 通过 `build_overlay()` 注入所有 23 个 stage 的 prompt

**Gate 机制**：Stage 5、9、20 为审批门，可人工/自动；Stage 15 自主决策 REFINE→13 或 PIVOT→8

**质量守卫**：
- NaN/Inf 快速失败 + 自修复循环（最多 10 次）
- 4 层引用校验（arXiv ID → CrossRef DOI → Semantic Scholar 标题匹配 → LLM 相关性打分）
- Anti-fabrication：VerifiedRegistry 确保论文数据与实验一致
- AI-slop 检测 + 7 维 NeurIPS 标准评审

**设计哲学**：**工业级流水线**——将科研拆解为可审计的 23 个原子步骤，每步有独立 prompt、gate、知识库读写。

---

### 3. EvoScientist — 双层架构 + Dual Memory + EvoSkills 生态

**双层架构**：论文与代码实现存在对应关系但粒度不同。

| 层次 | 架构 | 说明 |
|------|------|------|
| 论文 | 3 Agent（RA/EA/EMA） | Researcher Agent（创意）+ Engineer Agent（实验）+ Evolution Manager Agent（蒸馏） |
| 代码 | 6 Sub-agent（LangGraph） | Planner / Researcher / Coder / Debugger / Analyst / Writer，更细粒度分工 |

**Dual Memory 系统**（核心创新）：

| 记忆模块 | 内容 | 消费者 | 更新者 |
|---------|------|--------|--------|
| **Ideation Memory (M_I)** | 可行研究方向 + 失败方向 | Researcher Agent | EMA (IDE + IVE) |
| **Experimentation Memory (M_E)** | 有效数据处理策略 + 模型训练策略 + 最佳代码 | Engineer Agent | EMA (ESE) |

关键设计："what to research" 与 "how to implement" 是根本不同的知识类型，分开存储和检索。

**记忆检索**：mxbai-embed-large（1024 维）via Ollama 语义搜索，本质是轻量 RAG。

**三大自进化机制**（EMA 管理）：

| 机制 | 缩写 | 蒸馏内容 | 更新记忆 |
|------|------|---------|---------|
| Idea Direction Evolution | IDE | 从 top-ranked ideas 提取可行方向 | M_I |
| Idea Validation Evolution | IVE | 从失败验证中提取 anti-patterns | M_I |
| Experiment Strategy Evolution | ESE | 从成功实验提取可复用 patterns | M_E |

**实测效果**：执行成功率从 34.39%（无进化）→ 44.56%（有进化），相对提升约 30%。

**EvoSkills 生态**（独立仓库 github.com/EvoScientist/EvoSkills）：

10 个可安装 skill，组织为流水线：**ideation → experimentation → writing → memory**

| Skill | 功能 |
|-------|------|
| idea-tournament | 生成并排名竞争性研究创意；读取 M_I |
| proposal-extension | 将获胜创意扩展为 5+1 节研究提案 |
| pre-writing-planning | 故事设计 + 实验规划 + 图表设计 + 时间线 |
| experiment-pipeline | 结构化执行，有 attempt 预算和 gate 条件：Initial(≤20) → Tuning(≤12) → Method(≤12) → Ablation(≤18) |
| evo-memory | 学习层——维护 M_I 和 M_E，跨 cycle 积累知识 |

**自进化闭环**：idea-tournament + experiment-pipeline + evo-memory 形成学习循环——evo-memory 从 cycle 结果中蒸馏策略，写回记忆库供下一 cycle 检索。

**Skill 定义格式**（skill-creator/SKILL.md 标准）：

```
skill-name/
├── SKILL.md          ← 必须：YAML frontmatter + Markdown 指令
├── scripts/          ← 可选：自动化脚本
├── references/       ← 可选：参考文档
└── assets/           ← 可选：资源文件
```

**渐进式披露（3 级）**：
- Level 1 = Trigger（何时激活，~100 words，始终加载）
- Level 2 = Runbook（详细步骤，触发时加载，<500 行理想）
- Level 3 = Reference Library（支撑材料，按需加载）

**自由度控制**：
- 高自由度：文本指导（多种有效路径）
- 中自由度：伪代码 + 可配参数
- 低自由度：具体脚本（一致性关键时）

**Description 设计原则**："a little pushy to combat undertriggering"——积极触发防止遗漏。

**记忆中间件**（`middleware/memory.py`）：
- 持久化到 `MEMORY.md`（filesystem backend）
- 4 类记忆：User Profile / Research Preferences / Experiment History / Learned Preferences
- **Injection**：每次 LLM 调用前注入系统提示
- **Extraction**：每 20 条对话自动提取结构化事实（Pydantic model 强类型校验）
- 线程安全，跨会话持久

**Agent 通信**：file-based handoffs（非 message-passing），保真度高，可并行，可审计。

**设计哲学**：**进化式科研**——不只完成任务，而是在每个 cycle 中积累经验，逐步提升成功率。Dual Memory 是从「工具」到「科研伙伴」的关键跳跃。

---

### 4. ARIS — Markdown-as-Protocol 先锋

**Skill 载体**：纯 Markdown SKILL.md，agent-agnostic，任何 LLM 均可读取执行。

**Skill 组织**（31 Core + 12 Community）：

| 类别 | 代表 Skill |
|------|-----------|
| 核心工作流 | idea-discovery, experiment-bridge, auto-review-loop, paper-writing, research-pipeline, rebuttal |
| 文献 & 创意 | literature-survey, novelty-check, research-refine, experiment-plan |
| 实验 & 分析 | experiment-bridge-gpu, training-check, result-to-claim, ablation-planner, monitor-experiment |
| 论文 & 展示 | paper-slides, paper-poster, paper-illustration, grant-proposal, proof-writer |
| 专项 | semantic-scholar, comm-lit-review, dse-loop, formula-derivation, paper-review |

**跨模型对抗协议**（核心创新）：
- **Executor**（快速流畅）：Claude Code 写代码 / 生成想法 / 起草论文
- **Reviewer**（审慎严谨）：GPT-5.4 xhigh 批评 / 压力测试 / 评分（1-10）
- 循环直到收敛，打破 self-play 盲区

**触发机制**：
- Slash 命令：`/idea-discovery "topic"`, `/research-pipeline "topic"`
- 参数透传：`— key: value` 内联覆盖
- 组合链路：单 skill 独立运行 / 手动串联 / `/research-pipeline` 全自动

**Guard 机制**：
- Rebuttal 三门：No fabrication / No overpromise / Full coverage
- Anti-hallucination 引用：从 DBLP/CrossRef 获取真实 BibTeX
- GPU 部署前代码审查（GPT-5.4 xhigh review）
- Checkpoint 门控：`AUTO_PROCEED: true/false`, `human_checkpoint: true`

**多文献源**：Zotero / Obsidian vault / Local PDF / arXiv / Semantic Scholar

**跨 Agent 兼容**：Claude Code / Codex CLI / OpenClaw / Cursor / Trae / Windsurf / Antigravity

**Executor 变体**：
- `skills-codex/` → Codex 原生版
- `skills-codex-claude-review/` → Codex 执行 + Claude 评审
- `skills-codex-gemini-review/` → Codex 执行 + Gemini 评审

**设计哲学**：**方法论优先**——skill 是可组合的 Markdown 指令，整个栈基于纯文本，不绑定任何框架。

---

### 5. Dr. Claw — GUI 平台 + Skill 符号链接模式

**Skill 载体**：标准 SKILL.md，通过 symlink 挂载到项目的 `.claude/skills/` 目录。

**Skill 库架构**：
- 100+ 研究 skill，跨 6 个研究阶段
- 全局库 `dr-claw/skills/` → 项目级 symlink `.claude/skills/<skill-id>/`
- 运行时自动发现，无需注册

**6 阶段映射**：

| 阶段 | Skill 类别 | 输出目录 |
|------|-----------|---------|
| Survey | 文献检索 skill | `Survey/reports/` |
| Ideation | 创意生成 skill | `Ideation/ideas/` |
| Experiment | 实验实现 skill | `Experiment/core_code/` |
| Analysis | 分析统计 skill | `Experiment/analysis/` |
| Publication | 论文撰写 skill | `Publication/paper/` |
| Promotion | 展示传播 skill | `Promotion/slides/` |

**调用方式**：
- GUI：Skill Library Dashboard 浏览 → 选择 → 执行
- Chat：`Read .claude/skills/<skill-id>/SKILL.md and follow it`
- CLI：`drclaw --json chat reply --project <id> -m "message"`

**多后端支持**：Claude Code CLI / Gemini CLI / Codex CLI，用户在 Settings 中切换

**Pipeline Planner**：`inno-pipeline-planner` skill 允许对话式生成研究计划

**远程执行**：`openclaw_drclaw_turn.sh` 串行化执行防止 session-lock 冲突

**设计哲学**：**GUI 降门槛 + Symlink 即插即用**——skill 存为版本化目录，新项目自动继承全局 skill 库。

---

### 6. Orchestra AI-Research-SKILLs — 超大规模 Skill 库

**Skill 载体**：标准 SKILL.md，22 分类 87 个 skill。

**分类体系**（22 categories）：

| 编号 | 分类 | Skill 数 |
|------|------|---------|
| 00 | Autoresearch（编排层）| 1 |
| 01-04 | 模型开发（架构/Tokenization/Fine-Tuning）| 11 |
| 05-07 | 高级训练（可解释性/数据处理/Post-Training）| 14 |
| 08-10 | 基础设施（分布式训练/优化/Infra）| 15 |
| 11-12 | 部署（推理/评估）| 7 |
| 13-16 | AI 能力（多模态/Agent/RAG/Prompt Eng）| 20 |
| 17-18 | 生产（MLOps/Observability）| 5 |
| 19-21 | 新兴（Techniques/Paper Writing/Ideation）| 10 |

**安装协议**：
- CLI：`npx @orchestra-research/ai-research-skills`（交互式，自动检测已装 agent）
- Marketplace：`/plugin install fine-tuning@ai-research-skills`
- 安装到 `~/.orchestra/skills/`，agent symlink 自动生成

**编排层**：`00-autoresearch-skill/` 是中枢，双循环架构（inner optimization + outer synthesis），路由到领域 skill

**Prompt Guard**：Meta 86M 参数模型，99%+ TPR，<2ms GPU，injection/jailbreak 检测

**跨 Agent 兼容**：Claude Code / OpenCode / Cursor / Codex / Gemini CLI / Qwen Code

**设计哲学**：**大规模 skill 库 + NPM 分发**——用包管理器思维做 skill 发现和安装，skill 数量 > 质量深度。

---

### 7. Research-Claw — 5 阶段任务流水线 + 定时自动化

**Skill 载体**：标准 SKILL.md（YAML frontmatter），自动发现无需注册。

**Skill 目录结构**：
```
config/.skills/{skill-name}/
├── SKILL.md          ← 必须：YAML frontmatter 定义
└── templates/        ← 可选：模板文件
```

**Sub-Agent 沙盒架构**：
- Main Agent 编排 → Sub-Agent 在隔离 overlay 目录中执行 → 合并输出
- 防止子任务意外覆盖主项目文件

**5 阶段任务流水线**：

| 阶段 | 功能 | 命令 |
|------|------|------|
| UNDERSTAND | 读取项目上下文 | 自动 |
| PROPOSE | 生成范围 | `task_propose` |
| PLAN | 构建任务 DAG | `task_build` |
| EXECUTE | 并行批处理 | `task_execute` |
| FINALIZE | 合并提交 | `task_commit` |

**项目记忆**：`project.yaml` 跨会话持久化 + 自动 context 摘要（token 限制内）

**Overleaf 双向同步**：`/sync pull` + `/sync push`，AI 编辑自动 commit 到 Git

**定时自动化**（APScheduler + cron）：
- Daily Scan：搜索新论文
- Direction Drift：检测研究方向偏移
- Deadline Watch：会议截止日期追踪
- Weekly Digest：周报汇编
- Profile Refresh：更新研究画像

**多渠道通知**：Telegram / 飞书 / 钉钉 / Email / Apprise

**设计哲学**：**自托管科研助手**——重点不在 skill 深度，而在生活方式集成（定时任务 + 通知 + 协作同步）。

---

### 8. AI Scientist v2 — 目标驱动阶段式 Agent 管理

**Skill 载体**：不使用 SKILL.md，而是**代码级模块化**——每个 Python 模块就是一个能力单元。

**模块架构**：

```
ai_scientist/
├── perform_ideation_temp_free.py   ← 创意生成（Semantic Scholar 检索）
├── perform_writeup.py              ← 论文撰写
├── perform_plotting.py             ← 图表生成
├── perform_llm_review.py           ← LLM 评审
├── perform_vlm_review.py           ← VLM 视觉评审
├── tools/
│   ├── base_tool.py                ← 工具基类
│   └── semantic_scholar.py         ← 文献搜索工具
├── treesearch/
│   ├── agent_manager.py            ← 阶段式 Agent 编排
│   ├── parallel_agent.py           ← 并行 Agent 执行
│   ├── journal.py                  ← 实验日志（节点树）
│   ├── bfts_utils.py               ← Best-First Tree Search
│   ├── interpreter.py              ← 代码解释器
│   └── journal2report.py           ← 日志→报告转化
└── fewshot_examples/               ← Few-shot 论文示例
```

**4 阶段 Agent 管理器**（AgentManager）：

| Stage | 目标 | 最大迭代 |
|-------|------|---------|
| 1 | 获得至少 1 个可运行实现 | 20 |
| 2 | 跨 ≥2 数据集收敛 | 12 |
| 3 | 实验计划执行 | 12 |
| 4 | 风险因子处理 | 18 |

**Progressive Agentic Tree Search**（BFTS）：
- `num_workers: 4` 并行探索路径
- `max_debug_depth: 3` 调试递归深度
- `debug_prob: 0.5` 调试触发概率
- `num_drafts: 3` 每次生成多个候选方案
- Journal 追踪每个节点的指标、父子关系、分析反馈

**LLM 驱动的阶段推进**：
- `_check_stage_completion()` 检查完成条件
- `_evaluate_stage_progression()` LLM 判断是否推进
- `_identify_issues()` 扫描持续错误模式
- 每次转换记录 `StageTransition` 推理过程

**多模型分工**：
- 代码生成：Claude 3.5 Sonnet（12,000 token）
- 反馈分析：GPT-4o（8,192 token）
- 报告合成：GPT-4o（temperature 1.0）

**Checkpoint 持久化**：pickle 序列化 journal + stage_history + config + workspace

**创意生成协议**：
- 迭代精炼（5 轮 reflection）
- Semantic Scholar 文献验证（推荐但不强制）
- 结构化输出（ACTION + ARGUMENTS regex 解析）
- 跨运行复用（`reload_ideas` 参数）

**设计哲学**：**判断驱动流水线**——不是刚性状态机，而是 LLM 在每个决策点做推理判断；tree search 探索解空间而非线性推进。

---

### 9. uditgoenka autoresearch — 8 铁律 + 9 命令 Claude Code Skill

**Skill 载体**：标准 SKILL.md + 10 个 reference 协议文档 + 9 个 command 定义。

**文件结构**：
```
claude-plugin/
├── skills/autoresearch/
│   ├── SKILL.md                        ← 主 skill 定义
│   └── references/
│       ├── autonomous-loop-protocol.md ← 8 阶段循环
│       ├── core-principles.md          ← 7 通用原则
│       ├── plan-workflow.md
│       ├── security-workflow.md
│       ├── ship-workflow.md
│       ├── debug-workflow.md
│       ├── fix-workflow.md
│       ├── scenario-workflow.md
│       ├── predict-workflow.md
│       ├── learn-workflow.md
│       └── results-logging.md          ← TSV 追踪
└── commands/autoresearch/              ← 9 个命令注册
```

**8 条铁律协议**：

| # | 铁律 | 实现机制 |
|---|------|---------|
| 1 | Loop until done | 无界循环或 N 轮 |
| 2 | Read before write | 每轮先读全部上下文 |
| 3 | One change per iteration | 原子修改，故障可诊断 |
| 4 | Mechanical verification only | 只看指标，禁止主观判断 |
| 5 | Auto rollback | `git revert` 失败即回滚 |
| 6 | Simplicity preferred | 同效果更少代码 = 保留 |
| 7 | Git as memory | `experiment:` 前缀 commit |
| 8 | Think harder when stuck | 重读上下文 + 尝试激进方案 |

**9 命令集**：

| 命令 | 对应协议文档 | 核心能力 |
|------|------------|---------|
| `/autoresearch` | autonomous-loop-protocol.md | 主循环 |
| `/autoresearch:plan` | plan-workflow.md | 交互式 setup |
| `/autoresearch:security` | security-workflow.md | STRIDE/OWASP 审计 |
| `/autoresearch:ship` | ship-workflow.md | 8 阶段发布 |
| `/autoresearch:debug` | debug-workflow.md | 7 技法 bug 猎手 |
| `/autoresearch:fix` | fix-workflow.md | 迭代修复 |
| `/autoresearch:scenario` | scenario-workflow.md | 12 维 edge case |
| `/autoresearch:predict` | predict-workflow.md | 5 角色辩论 |
| `/autoresearch:learn` | learn-workflow.md | 文档引擎 |

**Verify + Guard 双轨**：
- Verify = "指标改善了吗？"（主目标）
- Guard = "其他东西没坏吧？"（回归防护）
- Guard 失败 → 最多 2 次返工

**Crash Recovery**：

| 故障类型 | 处理 |
|---------|------|
| 语法错误 | 即修不算轮次 |
| 运行时错误 | 最多 3 次重试 |
| 资源耗尽 | 回滚 + 降级变体 |
| 无限循环 | 超时自杀 + 回滚 |
| 外部依赖故障 | 跳过 + 记录 |

**Git-as-Memory**：
- `experiment:` 前缀 commit
- 失败实验通过 `git revert` 保留（不删除）
- 每轮读 `git log` + `git diff` 学习历史
- TSV 日志：iteration / commit hash / metric / delta / status / description

**设计哲学**：**Karpathy Loop 的工程化泛化**——从 ML 训练泛化到任意可量化任务，用 8 铁律 + Git 协议实现自主迭代。

---

## 二、Skill 设计维度横向对比

### 2.1 Skill 定义格式

| 项目 | 格式 | 载体 | Agent-Agnostic? |
|------|------|------|----------------|
| Karpathy | 自然语言 Markdown（program.md）| 单文件 | 是 |
| AutoResearchClaw | YAML prompt 配置 + MetaClaw SKILL.md | prompts.default.yaml | 部分 |
| EvoScientist | YAML frontmatter SKILL.md + 3 级渐进披露 | 目录结构（EvoSkills） | 是 |
| ARIS | 纯 Markdown SKILL.md | 目录结构 | **最强**（6+ agent 验证） |
| Dr. Claw | SKILL.md + symlink | 目录结构 | 是 |
| Orchestra | SKILL.md（编号分类）| 目录结构 | 是（6+ agent） |
| Research-Claw | SKILL.md + YAML frontmatter | 目录结构 | 部分 |
| AI Scientist v2 | **Python 模块**（非 Markdown）| .py 文件 | 否（代码绑定） |
| uditgoenka | SKILL.md + 10 reference docs + commands/ | 目录结构 | 否（Claude Code 绑定） |

### 2.2 Skill 发现与安装

| 项目 | 发现机制 | 安装方式 |
|------|---------|---------|
| Karpathy | 手动指定 program.md | 无需安装 |
| AutoResearchClaw | 23 阶段内置 | pip install + config |
| EvoScientist | `npx -y skills find` + skill_manager | skills.sh 安装 |
| ARIS | Slash 命令触发 | 手动复制到 skills/ |
| Dr. Claw | GUI Skill Library Dashboard | symlink 自动生成 |
| Orchestra | `npx @orchestra-research/ai-research-skills` | NPM 交互式安装 |
| Research-Claw | 启动时自动发现 | 放入 config/.skills/ 即可 |
| AI Scientist v2 | 无（硬编码模块）| pip install |
| uditgoenka | `/plugin marketplace add` | Claude Code plugin |

### 2.3 状态管理与记忆

| 项目 | 记忆机制 | 跨会话 | 跨项目 |
|------|---------|--------|--------|
| Karpathy | results log | 否 | 否 |
| AutoResearchClaw | 6 类 KB + 30 天衰减 + MetaClaw skill | 是 | 是（MetaClaw） |
| EvoScientist | Dual Memory（M_I + M_E）+ MEMORY.md middleware + mxbai-embed 语义检索 | **是**（middleware + 向量 DB） | **是**（EvoSkills evo-memory 跨 cycle） |
| ARIS | 无内建记忆 | 否 | 否 |
| Dr. Claw | Session-scoped | 部分 | 否 |
| Orchestra | 无内建记忆 | 否 | 否 |
| Research-Claw | project.yaml + context 摘要 | 是 | 否 |
| AI Scientist v2 | Journal + pickle checkpoint | 是（checkpoint） | 否 |
| uditgoenka | Git history + TSV | 是（Git） | 否 |

### 2.4 质量守卫

| 项目 | Guard Rail 设计 | 强度 |
|------|----------------|------|
| Karpathy | 文件权限 + 5min 超时 + 评估冻结 | ★★★★★（物理隔离） |
| AutoResearchClaw | NaN 检测 + 4 层引用校验 + 7 维评审 + AI-slop | ★★★★★（多层校验） |
| EvoScientist | Pydantic schema + attempt 预算 gate + file-based 可审计 | ★★★☆ |
| ARIS | 三门（no fabrication/no overpromise/full coverage）+ DBLP 引用 | ★★★★ |
| Dr. Claw | GUI 审批 | ★★ |
| Orchestra | Prompt Guard（Meta 86M）| ★★★ |
| Research-Claw | Sub-agent 隔离目录 | ★★★ |
| AI Scientist v2 | 阶段完成检查 + LLM 推进判断 + issue 扫描 | ★★★★ |
| uditgoenka | Verify + Guard 双轨 + auto rollback + crash recovery | ★★★★★ |

### 2.5 组合性与可扩展性

| 项目 | Skill 组合方式 | 可扩展性 |
|------|--------------|---------|
| Karpathy | 无（单一循环）| 低（仅 ML 训练） |
| AutoResearchClaw | 固定流水线（可 REFINE/PIVOT 跳转）| 中（MetaClaw 学习） |
| EvoScientist | EvoSkills 流水线（ideation→exp→writing→memory）+ 自进化闭环 | 高（开放生态 + 自学习） |
| ARIS | Slash 命令串联 / pipeline 全自动 | **最高**（43 skill 自由组合） |
| Dr. Claw | 6 阶段 × skill 矩阵 | 高（symlink 即用） |
| Orchestra | 编排层路由到领域 skill | 高（87 skill） |
| Research-Claw | 5 阶段 DAG + cron 自动化 | 中 |
| AI Scientist v2 | 4 阶段树搜索 | 低（代码级扩展） |
| uditgoenka | 9 命令独立 + 可组合 | 中（泛化但无组合协议） |

---

## 三、5 大 Skill 设计 Pattern 提炼

### Pattern 1: Markdown-as-Protocol（ARIS / Orchestra / EvoScientist / Dr. Claw）

**核心**：Skill = YAML frontmatter + Markdown 指令，纯文本可移植

**优势**：
- Agent-agnostic：同一 skill 可被 Claude / GPT / Gemini / Codex 执行
- 版本控制友好（Git diff 可读）
- 零框架开销，渐进式加载

**标准结构**：
```
skill-name/
├── SKILL.md          ← 必须
├── references/       ← 可选
├── scripts/          ← 可选
└── templates/        ← 可选
```

**关键设计决策**：
- Description 要 "pushy"（防止 under-triggering）
- 自由度按脆弱性分级（高→文本，低→脚本）
- 渐进加载（metadata ~100 words always → body on trigger → resources on demand）

### Pattern 2: Immutable Evaluation + Mutable Sandbox（Karpathy / uditgoenka）

**核心**：冻结评估标准，只允许 agent 改实验代码

**优势**：
- 杜绝 agent "作弊"（改评估逻辑让指标虚高）
- 公平对比（同一标尺）
- 可审计（每次变更只有 train.py diff）

**泛化**：uditgoenka 将此 pattern 从固定 val_bpb 泛化为用户定义的 Verify + Guard 双轨

### Pattern 3: Multi-Agent Adversarial Review（ARIS / AutoResearchClaw）

**核心**：不同 LLM 扮演 executor 和 reviewer，对抗式协作

**优势**：
- 打破单模型 self-play 盲区
- 两个模型比一个或多个的 ROI 最高
- 迫使 executor 应对 reviewer 未预见的弱点

**实现变体**：
- ARIS：Claude 执行 + GPT-5.4 评审（或 Codex+Claude / Codex+Gemini）
- AutoResearchClaw：多视角结构化辩论（同模型多角色）

### Pattern 4: Knowledge Base with Time Decay（AutoResearchClaw / EvoScientist）

**核心**：运行时积累知识，带时间衰减防止记忆污染

**优势**：
- 不从零开始——复用历史经验
- 衰减机制避免过时知识误导
- 分类存储（6 类 KB / 4 类 Memory）提高检索精度

**实现变体**：
- AutoResearchClaw：6 类 KB + 30 天衰减 + MetaClaw 从失败中学习 skill
- EvoScientist：Dual Memory（M_I 管"研究什么" / M_E 管"怎么实现"）+ mxbai-embed 语义检索 + EMA 三机制蒸馏（IDE/IVE/ESE）+ MEMORY.md middleware 每 20 条对话自动提取

### Pattern 5: Goal-Driven Stage Progression（AI Scientist v2 / AutoResearchClaw）

**核心**：不是刚性状态机，而是 LLM 判断是否推进到下一阶段

**优势**：
- 适应性强——可在某阶段多停留或快速跳过
- 每次转换有推理记录，可审计
- 自动检测并处理系统性问题

**实现变体**：
- AI Scientist v2：4 阶段 + BFTS 树搜索 + LLM 推进评估
- AutoResearchClaw：23 阶段 + 3 个 gate + PROCEED/REFINE/PIVOT 自主决策

---

## 四、对 MindFlow Skill 系统的启示

基于以上分析，以下是可直接借鉴的设计决策：

### 已做对的
1. **Markdown-as-Protocol**：MindFlow 的 SKILL.md 格式与生态共识一致
2. **渐进式加载**：先读 frontmatter 再读 body 的模式与 EvoScientist 标准吻合
3. **Skill 目录结构**：`skills/<category>/<name>/SKILL.md` 与主流结构一致

### 可借鉴的
1. **Verify + Guard 双轨**（from uditgoenka）：paper-digest 等 skill 可加入 Guard 检查（如"生成的笔记是否遗漏了论文的 key contribution"）
2. **记忆自动提取**（from EvoScientist）：Workbench/memory 可加入对话触发的自动蒸馏（每 N 轮提取结构化事实），而非只依赖手动 `/distill`
3. **跨模型评审**（from ARIS）：paper-digest 生成后可用第二模型做 quality check
4. **Description 要 "pushy"**（from EvoScientist skill-creator）：SKILL.md 的 description 应积极触发，防止 under-triggering
5. **MetaClaw 自学习 Skill**（from AutoResearchClaw）：从执行失败中自动提取 pattern → fix → 注入未来执行
6. **Dual Memory 分离**（from EvoScientist）：将 "研究什么"（ideation memory）与 "怎么实现"（experimentation memory）分开存储和检索，可启发 MindFlow 将 DomainMaps（方向性知识）与 Workbench/memory（操作性知识）更明确地分离
7. **Attempt 预算 + Gate 条件**（from EvoScientist experiment-pipeline）：给 skill 的每个阶段设 attempt 上限（如初始实现 ≤20 次），防止无限循环

### 不适用的
1. **NPM/Plugin 分发**（Orchestra / uditgoenka）：MindFlow 是个人知识库，不需要包管理器
2. **23 阶段流水线**（AutoResearchClaw）：过重，MindFlow 偏轻量人机协同
3. **BFTS 树搜索**（AI Scientist v2）：适合高算力实验室，不适合个人 Researcher

---

## Sources

- 各项目 GitHub repo README 及源码（2026-03-28 访问）
- 分析基于 repo 公开代码和文档，非论文描述（部分项目实际代码与论文有差异，如 EvoScientist）
