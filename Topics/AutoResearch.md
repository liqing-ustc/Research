---
title: AutoResearch 项目全景调研
tags:
  - AutoResearch
status: active
date_updated: 2026-03-26
---
## Overview

2026 年 3 月前后，以 Karpathy 的 **autoresearch** 发布为标志，"AI 自主做科研"从概念走向可落地的开源工具链。本文梳理 **8 款代表性项目**，从核心定位、技术架构、全流程能力、落地成本、适用场景做深度横向对比。

---

## 一、核心参数全景对比

| 项目 | 核心定位 | 技术架构 | 全流程覆盖 | 开源 | 部署门槛 |
|------|---------|---------|-----------|------|---------|
| **Karpathy autoresearch** | LLM 训练专属极简实验迭代 | 三文件架构（固定评估 + 可编辑沙盒 + 指令层），单指标棘轮 | 仅实验迭代与评估 | MIT | 极低 |
| **AutoResearchClaw** | 端到端「一句话出论文」 | 8 相位 23 阶段流水线，多 Agent 辩论，Docker 沙箱，4 层引用校验 | 文献→假设→实验→论文→评审 | ✓ | 中 |
| **EvoScientist** | 带长期记忆的自进化 AI 科研团队 | 三 Agent（RA/EA/EMA）+ 双持久化记忆库 | 全流程，核心强化跨项目经验沉淀 | 部分 | 较高 |
| **ARIS** | 轻量夜间科研自动化 | 双模型对抗（executor + reviewer），Markdown 技能集 | 文献综述、论文修改、审稿回复、轻量实验脚本 | ✓ | 极低 |
| **OpenLAIR Dr. Claw** | GUI 科研平台，科研版 NotebookLM | Web UI + 模块化技能插件 + 多项目管理 | 文献管理、笔记、实验、论文，主打人机协同 | AGPL-3.0 | 低 |
| **Orchestra SKILLs + Research-Claw** | 科研技能库 + 日常管理 | 70+ 预封装 skill + 自托管助手 + Overleaf 同步 | 实验工程化 + 文献/deadline/写作协同 | ✓ | 低-中 |
| **The AI Scientist** | 端到端科学发现标杆 | 多 Agent 闭环 + 贝叶斯优化实验 + 学术规范校验 | 全流程，从开放式发现到完整论文 | ✓ | 极高 |
| **uditgoenka autoresearch** | Karpathy Loop 的泛化版 Claude Code 技能 | 8 条铁律协议 + 9 命令集 + Git-as-memory + Guard 机制 | 任意可量化任务的自主迭代优化 | MIT | 极低 |

---

## 二、技术路线与设计哲学

五大技术流派，分化源于对「AI 在科研中的角色」的不同定位：

### 1. 极简单任务派：Karpathy autoresearch

- **哲学**：AI 收敛到科研中最机械的实验迭代环节，人类把控方向与评估标准
- **技术核心**：冻结评估标准（`prepare.py` 不可改），Agent 仅改 `train.py`，每次 5 分钟固定时长，同一标尺公平对比，杜绝 AI 作弊
- **范式转变**：从「人写代码→跑实验→改代码」到「人定规则→AI 做循环」

### 2. 泛化迭代派：uditgoenka autoresearch

- **哲学**：Karpathy Loop 的核心不在 ML，而是「改→验→留/弃」循环本身——泛化到任何可量化任务
- **技术核心**：8 条铁律（原子修改、机械验证、自动回滚、Git-as-memory）+ Guard 回归防护 + 9 个专用命令覆盖代码/安全/文档/营销
- **范式转变**：从「ML 专属实验循环」到「通用自主优化引擎」

### 3. 全流程端到端派：AutoResearchClaw & The AI Scientist

- **哲学**：AI 是完整的「虚拟研究员」，人仅提供 idea，全程零干预
- **技术核心**：科研工作流标准化拆解 → 多 Agent 分工 → 多层校验解决幻觉/虚假引用/不可复现三大痛点
- **范式转变**：数周科研压缩到数小时

### 4. 进化记忆派：EvoScientist

- **哲学**：核心瓶颈不是单次任务完成度，而是「无法像人类一样积累经验」
- **技术核心**：双记忆库持久化 + EMA 持续复盘 → 主动规避死胡同、复用有效路径
- **范式转变**：从「单次执行」到「持续进化」

### 5. 工具链赋能派：ARIS / Dr. Claw / Orchestra

- **哲学**：AI 不替代人，做「科研副驾驶」，解决特定痛点
- **技术核心**：可插拔模块，按需组合，不强制全流程自动化
- **范式转变**：平衡自动化与可控性

---

## 三、项目深度拆解

### 3.1 Karpathy — autoresearch

> AI 夜间自主改代码、跑实验、看指标，极简轻量，适合自主迭代实验。

| 维度 | 内容 |
|------|------|
| **GitHub** | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) |
| **规模** | ~630 行 Python，MIT 协议 |
| **核心三文件** | `prepare.py`（不可变；数据准备 + metric）、`train.py`（Agent 沙盒）、`program.md`（自然语言指令） |
| **指标** | `val_bpb`（validation bits per byte），词表无关，公平比较不同架构 |
| **循环** | 读源码 → 提假设 → 改 `train.py` → 训练 5 min → 评估 → `results.tsv` → 保留/回滚 → 下一轮。~12 exp/h，一夜 ~100 exp |
| **实绩** | 2 天 700 实验，发现 20 个可叠加优化，Time-to-GPT-2 从 2.02h → 1.80h（↓11%）；Shopify CEO 37 exp 一夜 ↓19% |
| **核心优势** | 极简零冗余；评估与实验代码完全隔离杜绝作弊；单卡 GPU 即可 |
| **短板** | 场景极度单一（仅 LLM 训练）；无文献/论文能力；无多 Agent 协作 |

```
Human ──→ program.md (自然语言指令)
              │
              ▼
         AI Agent (Claude / Codex)
              │
         读 + 改 train.py
              │
              ▼
         GPU 训练 5 min → eval val_bpb
              │
         ┌────┴────┐
       improve?   no → revert
         │
       commit → results.tsv → 下一轮
```

---

### 3.2 AutoResearchClaw

> 文献→假设→实验→论文全流程自动化，多 Agent 辩论，「Chat an Idea, Get a Paper」。

| 维度 | 内容 |
|------|------|
| **GitHub** | [aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) |
| **流水线** | 8 相位 23 阶段：研究定义 → 文献检索 → 知识综合 → 假设生成（辩论）→ 实验设计执行 → 分析决策 → 论文撰写 → 引用验证 |
| **多视角辩论** | 假设生成、结果分析、同行评审三环节均用 structured multi-perspective debate |
| **知识库** | 6 类 KB（decisions / experiments / findings / literature / questions / reviews）+ 30 天时间衰减 |
| **质量守卫** | NaN/Inf 检测、论文-证据一致性、引用相关性打分、anti-fabrication guard |
| **输出** | NeurIPS / ICML 模板 LaTeX 论文 |
| **文献源** | arXiv、Semantic Scholar |
| **核心优势** | 全流程零干预覆盖 8 大学科；解决引用幻觉；自动硬件检测与代码自修复 |
| **短板** | 算力要求较高（8G+ 显存）；创新深度依赖基座模型；学术诚信需人工审核 |

---

### 3.3 EvoScientist

> 具备记忆 + 进化能力，失败策略存入记忆库，跨项目积累经验。

| 维度 | 内容 |
|------|------|
| **论文** | [arXiv 2603.08127](https://arxiv.org/abs/2603.08127)（2026-03-09） |
| **GitHub** | [EvoScientist/EvoScientist](https://github.com/EvoScientist/EvoScientist) |
| **Agent 组成** | **RA**（Researcher，idea 生成）+ **EA**（Engineer，实验执行）+ **EMA**（Evolution Manager，经验蒸馏） |
| **双记忆模块** | ① **Ideation Memory**：高质量方向 + 失败方向 ② **Experimentation Memory**：有效策略 + 最佳代码 |
| **记忆检索** | `mxbai-embed-large` embedding via Ollama 语义检索 |
| **评估** | 6 篇论文投 ICAIS 2025，AI + 人类双重评审；novelty / feasibility / relevance / clarity 四维超 7 个 SOTA |
| **核心优势** | 唯一具备长期进化能力；多 Agent 贴合真实科研协作；6 投 6 中 |
| **短板** | 架构复杂学习曲线陡；算力存储要求高；轻量任务启动成本高 |

```
        ┌───────────────────────────────────┐
        │        Evolution Manager (EMA)     │
        │  ideation memory + exp memory      │
        └──────┬────────────────┬────────────┘
               │ 读取历史经验    │ 写入新经验
               ▼                │
         Researcher (RA)        │
           idea 生成             │
               │                │
               ▼                │
         Engineer (EA) ─────────┘
           实验 + 代码
```

---

### 3.4 ARIS（Auto-Research-In-Sleep）

> 极简 Markdown 技能集，即插即用，专注文献综述 + 论文修改 + 轻量实验。

| 维度 | 内容 |
|------|------|
| **GitHub** | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) |
| **核心理念** | 每个 skill 就是一个 `SKILL.md`，任何 LLM agent 可读取执行 |
| **兼容** | Claude Code / Codex CLI / OpenClaw / Cursor / Trae / Windsurf |
| **Cross-model Review** | executor + reviewer 对抗式协作，克服单模型盲区 |
| **文献综述** | `/research-lit`：IEEE/ACM/ScienceDirect 优先级、venue 分级、自动用本地 PDF + web 搜索 |
| **Obsidian 集成** | 可搜索 vault 中的 paper summary、tag reference、user insights |
| **核心优势** | 即插即用零配置；双模型对抗降幻觉；普通笔电可运行 |
| **短板** | 无自主实验执行能力；强依赖 Claude API；全流程自动化弱 |

---

### 3.5 OpenLAIR — Dr. Claw

> GUI 科研平台，类科研版 NotebookLM/Cursor，多项目管理，代码新手友好。

| 维度 | 内容 |
|------|------|
| **GitHub** | [OpenLAIR/dr-claw](https://github.com/OpenLAIR/dr-claw) |
| **定位** | "A Super AI Lab with massive AI Doctors as Assistants" |
| **可视化** | 文献图谱、笔记、任务看板、进度追踪、生成物管理 |
| **功能** | ① 文献管理（paper review / literature graph）② Skills Explorer + 全局技能库 ③ 计算资源管理 ④ Research Lab |
| **多后端** | Codex / Gemini workflows |
| **v1.0.0** | 桌面 + 移动端，skill discovery redesign、taxonomy browsing |
| **License** | AGPL-3.0（核心开源，商业版闭源） |
| **核心优势** | 图形化零代码上手；适配国内工具链；本地模型 + 商用 API 均可 |
| **短板** | 完全自主能力弱需人工引导；高阶功能付费；复杂实验工程化不足 |

---

### 3.6 Orchestra SKILLs & Research-Claw

> 科研技能库 + 日常管理工具链，解决环境配置痛点。

#### 6a AI-Research-SKILLs

| 维度 | 内容 |
|------|------|
| **GitHub** | [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs) |
| **定位** | 70+ 预封装科研 skill，给任意 coding agent 赋能 |
| **安装** | `npx @orchestra-research/ai-research-skills` |
| **兼容** | Claude Code / OpenCode / Cursor / Codex / Gemini CLI / Qwen Code |
| **v0.15.0** | Prompt Guard（Meta 86M，99%+ TPR, <1% FPR）+ 8 语种 |
| **覆盖** | experiment tracking / hyperparameter sweep / model registry / profiling / paper writing 等 |

#### 6b Research-Claw（nanoAgentTeam）

| 维度 | 内容 |
|------|------|
| **GitHub** | [nanoAgentTeam/research-claw](https://github.com/nanoAgentTeam/research-claw) |
| **定位** | 自托管科研助手：论文 + 文献 + deadline + 多渠道通知 |
| **Overleaf** | 双向 sync（AI → Overleaf ↔ 协作者） |
| **Sub-Agent** | 自动拆分 researcher + writer，隔离目录，合并后写入主项目 |
| **项目记忆** | 跨 session 记住主题、偏好、历史 |
| **定时任务** | daily scan + weekly digest + drift detection |
| **多渠道** | Telegram / 飞书 / 钉钉 / Email / Apprise |
| **API** | 任何 OpenAI-compatible（GPT / DeepSeek / Qwen / Claude） |

---

### 3.7 The AI Scientist（Sakana AI）

> 端到端科学发现标杆，首个通过顶会同行评审的 AI 科研系统。

| 维度 | 内容 |
|------|------|
| **GitHub** | [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)（v1）/ [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)（v2） |
| **论文** | 发表于 **Nature**：[The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://sakana.ai/ai-scientist-nature/) |
| **里程碑** | v2 生成的论文通过 ICLR 2025 workshop 同行评审，平均分 6.33，超过 55% 人类论文（团队出于学术伦理主动撤稿） |
| **v2 改进** | 移除对人类模板的依赖；泛化至多 ML 领域；引入 **progressive agentic tree search** + experiment manager agent |
| **技术** | 多 Agent 闭环 + 贝叶斯优化实验设计 + 完整学术规范校验 |
| **核心优势** | 学术认可度最高（Nature 发表）；具备开放式科学发现能力；全流程可审计 |
| **短板** | 部署极高（高性能 GPU 集群 + 64G 内存）；算力成本极高；迭代速度慢 |

---

### 3.8 uditgoenka/autoresearch — Karpathy Loop 泛化版

> 把 Karpathy 的实验循环从 ML 训练泛化为 Claude Code 通用自主迭代技能，覆盖代码/安全/文档/营销等一切可量化任务。

| 维度 | 内容 |
|------|------|
| **GitHub** | [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch)（⭐ 2.3k, MIT） |
| **定位** | "Turn Claude Code into a relentless improvement engine" |
| **版本** | v1.8.2，125 commits，活跃维护 |
| **载体** | Claude Code 插件（SKILL.md + commands），非独立脚本 |
| **安装** | `/plugin marketplace add uditgoenka/autoresearch` 一行命令 |

**8 条铁律协议**：

| # | 规则 | 说明 |
|---|------|------|
| 1 | Loop until done | 无界循环或 N 轮 |
| 2 | Read before write | 先理解上下文再改 |
| 3 | One change per iteration | 原子修改 |
| 4 | Mechanical verification only | 只看指标，禁止主观判断 |
| 5 | Auto rollback | 失败即回滚 |
| 6 | Simplicity preferred | 更少代码 + 同等效果 = 保留 |
| 7 | Git as memory | `experiment:` 前缀 commit，每轮读 git history |
| 8 | Think harder when stuck | 卡住时重分析 + 尝试激进方案 |

**9 命令集**：

| 命令 | 功能 |
|------|------|
| `/autoresearch` | 主循环，无界自主迭代 |
| `/autoresearch:plan` | 交互式 setup（目标/范围/指标） |
| `/autoresearch:security` | STRIDE/OWASP 安全审计 + 自动修复 |
| `/autoresearch:ship` | 多阶段发布（代码/内容/营销/研究） |
| `/autoresearch:debug` | 假设驱动自主 bug 猎手 |
| `/autoresearch:fix` | 迭代修复（tests/types/lint/build） |
| `/autoresearch:scenario` | 12 维度 edge case 探索 |
| `/autoresearch:predict` | 5 角色 multi-persona 辩论分析 |
| `/autoresearch:learn` | 文档引擎（init/update/check/summarize） |

**Crash Recovery**：语法错误即修不算轮次 → 运行时最多 3 次修复 → 资源耗尽回滚降级 → 无限循环超时自杀

**与 Karpathy 原版的核心区别**：

| 维度 | Karpathy 原版 | uditgoenka 泛化版 |
|------|-------------|-----------------|
| 载体 | 独立 Python 脚本 | Claude Code 插件 |
| 场景 | 仅 LLM 训练 | 任意可量化任务 |
| 指标 | 固定 `val_bpb` | 用户自定义 |
| 记忆 | `results.tsv` | Git history + TSV |
| 安全 | 无 | Guard 回归防护 + auto rollback |
| 命令 | 单一循环 | 9 个专用命令 |

| **核心优势** | Karpathy Loop 最忠实的工程化泛化；安装极简；Guard + rollback 更健壮；multi-persona 辩论独创 |
|------|------|
| **短板** | 强绑定 Claude Code 不跨 agent；无持久化记忆（仅 git history）；无科研专属能力（文献/论文） |

```
Karpathy autoresearch (原始灵感，ML only)
    │
    ├─→ uditgoenka/autoresearch (泛化为通用 Claude Code skill)
    │
    ├─→ AutoResearchClaw (泛化为端到端论文生成)
    │
    └─→ ARIS (泛化为 Markdown 科研技能集)
```

---

## 四、全流程能力量化测评

按科研 6 大环节 10 分制评分：

| 项目 | 文献调研 | 假设生成 | 实验执行 | 结果分析 | 论文撰写 | 同行评审 |
|------|---------|---------|---------|---------|---------|---------|
| autoresearch | 0 | 3 | **10** | 8 | 0 | 0 |
| AutoResearchClaw | 9 | 8 | 8 | 8 | 9 | 8 |
| EvoScientist | 9 | **10** | 9 | 9 | 8 | 9 |
| ARIS | 7 | 5 | 4 | 6 | 8 | 9 |
| Dr. Claw | 8 | 6 | 5 | 6 | 7 | 5 |
| Orchestra + Research-Claw | 0 | 4 | 9 | 8 | 0 | 0 |
| **The AI Scientist** | **10** | **10** | 9 | **10** | **10** | **10** |
| uditgoenka autoresearch | 0 | 5 | 8 | 7 | 0 | 0 |

---

## 五、部署门槛与落地成本

| 项目 | 最低硬件 | 依赖 | 商用 API | 学习成本 | 综合成本 |
|------|---------|------|---------|---------|---------|
| autoresearch | 单卡 GPU（4G+） | Python + PyTorch | 可选 | 极低（10 min） | 极低 |
| AutoResearchClaw | 8G+ 显存，16G 内存 | Python + Docker + 学术 API | 强依赖 | 中（30 min） | 中 |
| EvoScientist | 多卡 GPU，32G+ 内存 | Python + 向量 DB + Docker | 强依赖多 API | 极高（需工程团队） | 极高 |
| ARIS | 普通笔电（8G+ 内存） | Claude Code 环境 | 强依赖 Claude | 极低（即插即用） | 低 |
| Dr. Claw | 普通笔电（8G+ 内存） | 一键安装 | 可选 | 极低（GUI） | 低 |
| Orchestra + Research-Claw | 单卡 GPU（6G+） | Python + 对应框架 | 可选 | 中 | 低 |
| The AI Scientist | GPU 集群，64G+ 内存 | 复杂分布式环境 | 强依赖顶级模型 | 极高 | 极高 |
| uditgoenka autoresearch | 普通笔电 | Claude Code + Git | 强依赖 Claude | 极低（一行安装） | 极低 |

---

## 六、综合对比一览

| 项目 | 类型 | Agent 数 | 记忆/进化 | UI | 最佳场景 |
|------|------|---------|----------|----|----|
| **autoresearch** | 实验循环 | 1 | ✗ | CLI | 有 GPU 的 ML 调参迭代 |
| **AutoResearchClaw** | 端到端论文 | 多（辩论） | KB + 30 天衰减 | CLI | 快速产出论文初稿 |
| **EvoScientist** | 自进化科学家 | 3（RA/EA/EMA） | ✓ 双记忆 | CLI | 长期深耕单领域的团队 |
| **ARIS** | 技能集 | 2（executor + reviewer） | ✗ | Markdown | 给现有 agent 快速加技能 |
| **Dr. Claw** | GUI 平台 | 多 | ✓ | Desktop + Mobile | 无代码基础 / 多项目管理 |
| **Orchestra + R-Claw** | 技能库 + 管理 | 多（sub-agent） | ✓ 项目记忆 | CLI + Web + IM | ML 团队工程化 + 日常管理 |
| **The AI Scientist** | 端到端发现 | 多（闭环） | ✓ | CLI | 顶级机构高严谨性研究 |
| **uditgoenka autoresearch** | 泛化迭代 skill | 1 | Git history | CLI（Claude Code） | 任意可量化任务的自主优化 |

---

## 七、选型决策指南

### 个人研究者 / 学生

| 画像                | 推荐                   | 理由                |
| ----------------- | -------------------- | ----------------- |
| DL/LLM 方向，高频做模型实验 | **autoresearch**     | 极简零成本，通宵跑实验效率最高   |
| 硕博投稿期，需改论文/回审稿意见  | **ARIS**             | 轻量无门槛，双模型对抗提升质量   |
| 无代码基础 / 文科社科      | **Dr. Claw**         | GUI 操作，覆盖文献到论文全流程 |
| 需快速验证跨学科想法出论文     | **AutoResearchClaw** | 全流程自动化，一句话搞定      |
| 用 Claude Code 做日常代码/内容优化 | **uditgoenka autoresearch** | 泛化迭代，9 命令覆盖代码/安全/文档 |

### 科研团队 / 实验室

| 画像 | 推荐 | 理由 |
|------|------|------|
| ML 工程化团队，解决环境/部署痛点 | **Orchestra SKILLs** | 海量预封装技能，提升团队工程效率 |
| 长期深耕单领域，需沉淀经验 | **EvoScientist** | 长期进化，避免重复踩坑 |
| 顶级机构，需高创新高严谨 | **The AI Scientist** | 学术认可度最高，可支撑顶刊研究 |
| 多人协作，需日常管理 | **Research-Claw** | Overleaf 同步 + deadline 追踪 + 多渠道 |

### 组合使用建议

这些项目**不互斥**，可按需组合：
- **autoresearch**（跑实验）+ **ARIS**（文献综述）+ **Research-Claw**（日常管理）
- **Orchestra SKILLs** 作为底层技能层，被 Dr. Claw 或 ARIS 调用
- **EvoScientist**（核心研发）+ **AutoResearchClaw**（快速出初稿）覆盖不同阶段
- **uditgoenka autoresearch**（代码迭代优化）+ **ARIS**（科研特定能力）互补短板

---

## Key Takeaways

1. **三条路线分化**：指标驱动（autoresearch）→ ML 训练调优；泛化迭代（uditgoenka autoresearch）→ 任意可量化任务；流程驱动（AutoResearchClaw / The AI Scientist）→ 论文产出

2. **记忆与进化是关键差异**：EvoScientist 双记忆 + AutoResearchClaw KB 衰减 → agent 不再每次从零开始，是从「工具」到「科研伙伴」的跳跃

3. **Markdown-as-Protocol 成为共识**：ARIS 和 Orchestra SKILLs 用纯 Markdown 定义 skill → 跨 agent 可移植，不绑定框架

4. **GUI vs CLI**：Dr. Claw 走 GUI 降门槛；autoresearch / ARIS 极简 CLI——看用户画像

5. **The AI Scientist 树立标杆**：Nature 发表 + ICLR 评审通过证明 AI 端到端科研的上限，但门槛极高，短期内普通团队难以复制

## Open Problems

1. **实验可复现性**：autoresearch 5-min 固定窗口是聪明设计，但多 GPU / 长训练场景如何公平比较？
2. **学术诚信边界**：The AI Scientist 主动撤稿展示了负责任态度，但 AI 生成论文的伦理规范仍需社区共识
3. **记忆污染**：EvoScientist 如果积累错误经验如何清洗？遗忘机制需更好设计
4. **互操作标准**：Markdown-as-Protocol 方向正确，但各家 skill 格式仍不统一
5. **评估基准**：如何客观评估 AI 科研助手？需要系统化 benchmark
6. **Agent 锁定风险**：uditgoenka autoresearch 强绑定 Claude Code，ARIS 依赖 Claude API——跨 agent 可移植性仍是未解难题

---

## Sources

- [karpathy/autoresearch (GitHub)](https://github.com/karpathy/autoresearch)
- ['The Karpathy Loop': 700 experiments, 2 days (Fortune)](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)
- [Karpathy's autoresearch (VentureBeat)](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
- [Karpathy's 630-line script (The New Stack)](https://thenewstack.io/karpathy-autonomous-experiment-loop/)
- [aiming-lab/AutoResearchClaw (GitHub)](https://github.com/aiming-lab/AutoResearchClaw)
- [Auto Research Claw: 23-Stage Pipeline](https://juliangoldie.com/auto-research-claw/)
- [EvoScientist: arXiv 2603.08127](https://arxiv.org/abs/2603.08127)
- [EvoScientist (GitHub)](https://github.com/EvoScientist/EvoScientist)
- [ARIS (GitHub)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)
- [ARIS (DeepWiki)](https://deepwiki.com/wanshuiyin/Auto-claude-code-research-in-sleep)
- [OpenLAIR/dr-claw (GitHub)](https://github.com/OpenLAIR/dr-claw)
- [Orchestra-Research/AI-Research-SKILLs (GitHub)](https://github.com/Orchestra-Research/AI-Research-SKILLs)
- [nanoAgentTeam/research-claw (GitHub)](https://github.com/nanoAgentTeam/research-claw)
- [The AI Scientist — Nature publication (Sakana AI)](https://sakana.ai/ai-scientist-nature/)
- [SakanaAI/AI-Scientist-v2 (GitHub)](https://github.com/SakanaAI/AI-Scientist-v2)
- [AI Scientist first peer-reviewed publication (Sakana AI)](https://sakana.ai/ai-scientist-first-publication/)
- [How to build an AI Scientist (Nature News)](https://www.nature.com/articles/d41586-026-00899-w)
- [Autoresearch Explained (DataScienceDojo)](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/)
- [Guide to AutoResearch (DataCamp)](https://www.datacamp.com/tutorial/guide-to-autoresearch)
- [uditgoenka/autoresearch (GitHub)](https://github.com/uditgoenka/autoresearch)
- [Autoresearch — Autonomous Goal-Directed Iteration (udit.co)](https://udit.co/projects/autoresearch)
- [Karpathy's Pattern Spawns Generalized Claude Code Skill (SimpleNews.ai)](https://www.simplenews.ai/news/karpathys-autoresearch-pattern-spawns-generalized-claude-code-skill-for-autonomous-iteration-srsk)
