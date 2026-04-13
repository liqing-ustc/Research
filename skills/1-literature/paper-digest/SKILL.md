---
name: paper-digest
description: Use when user asks to "read paper", "analyze paper", "summarize paper", "读论文", "分析文献", "帮我看一下这篇paper", "论文笔记", or provides a PDF file that appears to be an academic paper.
argument-hint: "[arXiv URL / blog URL / PDF path / title]"
allowed-tools: Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---

## Purpose

给定一篇论文或技术 blog 的来源（URL、PDF 路径、论文标题或者关键词），它自动获取内容、提炼核心信息，生成结构化笔记。

支持两种内容类型：
- **论文**：arXiv、PDF 等学术论文
- **Blog**：技术博客文章（如 Google Research Blog、Lilian Weng、公司技术博客等）

## Steps

### Step 1：获取论文内容

根据 `source` 的类型选择获取方式：

| 类型        | 示例                                 | 获取方式                                                                |
| :-------- | :--------------------------------- | :------------------------------------------------------------------ |
| arXiv URL | `https://arxiv.org/abs/{arxiv_id}` | WebFetch 抓取arxiv HTML: `https://arxiv.org/html/{arxiv_id}`          |
| PDF 路径    | `/path/to/paper.pdf`               | Read 直接读取                                                           |
| 论文标题/关键词  | `"Diffusion Policy"`               | WebSearch（建议加 `site:arxiv.org`），定位论文arxiv id后，WebFetch 抓取arxiv HTML |
| Blog URL  | `https://lilianweng.github.io/...` | WebFetch 抓取页面                                                       |

### Step 2：笔记生成

**模板**: 遵循 `references/paper-note-template.md`，不可自行简化。

#### 核心质量规则
1. **主要图表和公式数量下限（硬性）**：
   - Figures：**架构/方法图**（嵌入 Method，必须）+ **主结果图**（嵌入 Experiments → Results 子节，必须）+ **teaser 图**（嵌入 Summary 之后，若论文有则必须，没有则跳过）。判断 teaser：看 Figure 1 是否为 "overview / concept / 动机示意"类整体图示；若 Figure 1 本身就是架构图，则不存在独立 teaser，只需嵌入到 Method 即可，不要重复嵌入
   - Tables ≥ 1：主 benchmark 对比表，包含主要 baseline 和论文方法的结果，复制为 Markdown 表格嵌入 Experiments → Results
   - Equations ≥ 1：核心 loss / 更新规则 / 目标函数，嵌入 Method 段
   - 图表公式必须**内嵌在对应论证段落**（Method / Experiments）中，不单独开章节
2. **模板填写提示的处理**: 模板中形如 `%% ... %%` 的块是 Obsidian 隐藏注释，用于提示如何填写。生成笔记时必须**替换为实际内容**，或在无内容时保留为空。**禁止**把提示文本当正文写出来。
3. **内联概念链接**: 笔记中首次出现的技术术语必须用 `[[Concept]]` wikilink 标注，例如 `[[Flow Matching]]`、`[[LoRA]]`、`[[Action Chunking]]`
4. **严禁 ASCII 流程图**: 用结构化 Markdown 列表 + `$数学符号$` 描述架构
5. **公式格式**: 遵循模板的 `**Equation #. {公式名}**` + `$$` 块 + `**符号说明**` + `**含义**` 扁平结构。`$$` 块前后**必须有空行**否则 Obsidian 不渲染。超长公式用 `aligned` 拆分。公式名无需 wikilink；若该公式对应某个已有概念笔记，可在正文首次提及该概念时用 `[[Concept]]` 标注。
6. **图片外链优先**: arXiv HTML / 项目主页 / GitHub，找不到再本地下载
7. **Blog 适配**: 若 source 为 blog 而非论文，删除或简化 `Experiments` 段（`Datasets` / `Implementation Details` 通常不适用；`Results` 改为 "Key Points"），`可复现性评估` 整段删除。`关联笔记` 里无内容的子类也直接删除。

> 公式/图片/表格的详细质量规范见 `references/quality-standards.md`

### Step 3：笔记文件保存

**文件名格式**：`YYMM-ShortTitle.md`

- `YYMM`：取自 `date_publish` 的年份后两位 + 月份，如 `2603`（2026年3月）
- `ShortTitle`：标题的 CamelCase 缩写，2-4 个关键词，如 `EvoScientist`、`RoboClaw`、`DiffusionPolicy`
- Blog 同理，如一篇 2026 年 2 月的 blog 关于 scaling laws → `2602-ScalingLaws.md`

**去重检查**：用 Glob 扫描 `Papers/` 目录，检查是否已存在同名或同主题笔记（搜索标题关键词）。若发现重复，停止并告知 Human，不创建新文件。

**Tag 选择**：阅读 `references/tag-taxonomy.md`，按照规范选择 tag。

**写入文件**：用 Write 将笔记保存到 `Papers/{笔记文件名}.md`。

### Step 4：日志记录

用 Edit（或 Write 若文件不存在）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

   ```markdown
   ### [HH:MM] paper-digest
   - **input**: <source 的原始内容>
   - **output**: [[Papers/{笔记文件名}]]
   - **observation**: <一句话描述论文核心贡献>
   - **status**: success
   ```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

- [ ] `Papers/{笔记文件名}.md` 已创建
- [ ] 关键 concept 已用 `[[concept]]` wikilink 标注
- [ ] 架构/方法图已嵌入 Method 段（必须）
- [ ] 主结果图已嵌入 Experiments → Results 子节（必须）
- [ ] teaser 图已嵌入 Summary 之后（若论文有独立 teaser；若 Figure 1 即架构图则跳过此项）
- [ ] Tables ≥ 1（主 benchmark 表已复制为 Markdown，包含主要 baseline 和论文方法的结果）
- [ ] Equations ≥ 1（核心公式用 `$$` 块，前后空行，含 `符号说明` 与 `含义`）
- [ ] 模板中的 `%% ... %%` 提示已全部替换为实际内容或留空，无提示文本外泄到正文
- [ ] `可复现性评估` checkbox 已根据论文与 GitHub repo 实际情况勾选（论文适用时）
- [ ] `关联笔记` 中无内容的子类已删除
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`
