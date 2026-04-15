---
name: paper-digest
description: Use when user asks to "read paper", "analyze paper", "summarize paper", "读论文", "分析文献", "帮我看一下这篇paper", "论文笔记", or provides a PDF file that appears to be an academic paper.
argument-hint: "[arXiv URL / blog URL / PDF path / title]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, WebSearch, WebFetch
---

## Purpose

给定一篇论文或技术 blog 的来源（URL、PDF 路径、论文标题或者关键词），它自动获取内容、提炼核心信息，生成结构化笔记。

支持两种内容类型（在 Step 2.2 的 extraction 草稿中标记；**不**用于决定笔记结构——body 结构统一由 `source_sections` 驱动）：
- **paper**：arXiv、PDF 等学术论文（图/公式/benchmark/ablation 完整）
- **blog**：所有 blog 类来源（Lilian Weng 教程、Google Research Blog、公司发布会、Generalist AI / OpenAI / Anthropic 的 announcement 等）

## Steps

### Step 1：Dedup 预检（在 fetch 之前必做）

任何 fetch 之前先扫一遍 `Papers/`，避免对一篇已有笔记重复 digest 浪费 5 分钟和几十次 tool call。

1. **从 source 猜 ShortTitle / 关键标识**：
   - arXiv URL → 用 arxiv id 反查：`Grep "arxiv.org/abs/<id>" Papers/`（frontmatter `url:` 字段）
   - 论文标题/关键词 → 用关键词 CamelCase 形式 Glob
   - Blog URL → 从 URL slug 提取关键词（如 `apr-02-2026-GEN-1` → `GEN1`）
2. **多模式扫描**：
   ```
   Glob Papers/*<ShortTitleCamelCase>*.md
   Glob Papers/*<acronym>*.md         # 若有缩写
   Grep "<source_url>" Papers/        # frontmatter url 字段反查
   ```
3. **命中处理**：
   - 找到对应笔记 → **立刻停止**，告知 Human "已有笔记 [[Papers/xxx]]，是否需要 force-refresh / 增量更新 / 取消"
   - 没命中 → 进 Step 2 fetch + extract

### Step 2：Fetch + Extract（写笔记之前必做）

**目的**：拉取源材料 → 结构化抽取所有"事实性元素"（章节大纲、图、表、公式、视频）。Step 3 的笔记 compose **只允许**使用这一步抽取的内容，不允许在写笔记时凭印象引入任何新 URL、新数字、新公式。

**抽取和 compose 必须分开**：fabrication（捏造图链接、目测图表数字、错引公式）几乎永远发生在"边读边写"的模式里。把"抽取"和"compose"切开，事实层和叙事层各自负责，捏造路径就被堵死了。抽取阶段 **诚实地只记录源里真实存在的元素**——不存在的就跳过，不要凭印象补。

#### Step 2.1：fetch 源材料并落盘到本地

必须有一个**本地可读的副本**——使用 [defuddle](https://github.com/kepano/defuddle) CLI 抓取网页并转换为干净的 Markdown 和 meta data，不用 WebFetch（WebFetch 的 LLM summarizer 会丢图/视频/数字 URL）。

> **若 source 是论文标题或关键词**：先 WebSearch（建议加 `site:arxiv.org`）定位 arxiv id，再按 arXiv URL 处理。

**文件名约定**（避免并行 / 重复 digest 时互相覆盖）：源材料文件名必须**包含 source 唯一标识**，禁用 `/tmp/paper.html`、`/tmp/page.html` 这种通用名。

##### Step 2.1.a：fetch 主源

| 来源                                     | 落盘方式                                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| arXiv URL `https://arxiv.org/abs/{id}` | Bash `npx defuddle parse "https://arxiv.org/html/{id}" --json --markdown -o /tmp/arxiv_{id}.json`（如 `/tmp/arxiv_2512.04601.json`） |
| Blog URL                               | Bash `npx defuddle parse "<url>" --json --markdown -o /tmp/blog_{last-path-segment}.json`（如 `/tmp/blog_apr-02-2026-GEN-1.json`） |
| PDF 路径                                 | 已经是本地文件，直接用路径                                                                                     |

**记住这个路径**，Step 2.2 抽取时会读取。

##### Step 2.1.b：官方次源自动发现（仅 `content_type: paper`，blog 跳过 b / c 子步）

一篇论文常常有多个官方呈现：arXiv 是文本权威来源，project page 有 demo 视频和高清图，GitHub 有代码和模型 release 信息，次源让笔记的媒体层和 artifact 层更完整。

**发现流程**：

1. **主通道（主源抽取）**：Grep 主源文件的 abstract + 开头区域，找：
   - `github.com/<org>/<repo>` 形式的链接
   - 紧邻 "project page" / "website" / "code" / "demo" 等 label 的 URL
2. **Fallback（WebSearch）**：主通道找不到时，用 WebSearch 兜底：
   ```
   WebSearch "<paper title>" project page
   WebSearch "<paper title>" github
   ```
   只看前几条结果，挑候选 URL 进 officiality check；找不到就略过，不强求。
3. **Officiality check**：任何候选 URL 进入 fetch 之前，综合判断它是否来源于官方——有无任何证据把它和论文作者 / 机构绑定（org 名、域名、arxiv 里的反向链接、README / 页面文案、作者名 subdomain 等都算）。

##### Step 2.1.c：fetch 次源（仅通过 officiality 的候选）

| 次源            | 落盘方式                                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Website       | Bash `npx defuddle parse "<website_url>" --json --markdown -o /tmp/website_{slug}.json`（slug 取 URL 最后一段）                              |
| GitHub README | Bash `curl -sL https://raw.githubusercontent.com/<org>/<repo>/main/README.md -o /tmp/readme_<org>_<repo>.md`（若 main 分支不存在，试 master） |


记下 `secondary_sources.website.local_path` 和 `secondary_sources.github.local_path`，Step 2.2 抽取次源内容时会读取。

#### Step 2.2：输出结构化 extraction 草稿

读取所有源文件，在对话里产出一个 YAML/Markdown 结构化草稿（不写文件，只在工作记忆里），字段如下：

```yaml
content_type: paper | blog
primary_source:
  url:
  local_path:
secondary_sources:                    # 任一子源可用则保留子键；两个都没有则整块省略
  website:                            # 若无则删除 website 子键
    url:
    local_path:
  github:                             # 若无则删除 github 子键
    url:
    local_path:
bibliographic:
  title: ...
  authors: [...]
  institutes: [...]                 # 作者机构，找不到就留空，不强制 grep 验证
  date_publish: YYYY-MM-DD or YYYY-MM  # arXiv 源可从 id 直接推断：`2504.16054` → 2025-04，`2602.12684` → 2026-02
  venue: ...
code_assessment:                    # 仅当 GitHub 次源可用时填写，否则整块省略
  scope: inference-only | inference+train | train-only | unclear   # 从 README 判断
  released_models:                  # README 里明确列出的 checkpoint / weight
    - name: model1
      description:                  # 参数量、用途、训练数据等一句话描述
    - name: model2
      description:
source_sections:                    # 源文档的大纲，顺序保留
  - id: sec1
    heading: "Introduction"
  - id: sec2
    heading: "Method"
  - id: sec3
    heading: "Experiments"
figures:
  - id: fig1
    caption: "..."
    url: "https://..."              # 或 null（canvas / 无静态 URL，只记 caption）
    section_id: sec2                # 挂载到哪个 source_section；compose 时就近嵌入
tables:
  - id: tab1
    caption: "..."
    content: |                        # 从源复制的 Markdown 表体（表头 + 数据行），compose 时直接粘贴到笔记，不在 compose 阶段重新组装
      | Col1 | Col2 | Col3 |
      |---|---|---|
      | ... | ... | ... |
    section_id: sec3
equations:
  - id: eq1
    name: "Flow matching loss"
    latex: "\\mathcal{L}_{FM} = ..."
    section_id: sec2
videos:
  - id: vid1
    caption: "..."
    url: "https://.../clip.mp4"      # 或 youtube.com/watch?v=...
    section_id: sec3
```

**抽取原则**：
- 只抽取**源材料中实际存在**的元素。若某一项不存在，跳过，**禁止捏造**。
- **图片 / 视频 URL**：defuddle 已自动将相对路径转为绝对 URL（如 `https://arxiv.org/html/2504.16054v1/x1.png`），直接从 Markdown 中提取即可。若遇到仍为相对路径的 URL，手动拼接 origin + base path。
- 对 PDF：图通常无外链 URL，先标记 `url: null` + caption；后续 compose 阶段决定要不要本地下载。
- **`source_sections`** 按源文档的章节顺序记录大纲（paper 用 `\section{}` / `\subsection{}` 或 `<h2>` / `<h3>`；blog 用 `<h2>` / `<h3>` / 明显的段落标题）。层级和粒度根据源结构判断。**排除非核心章节**：Related Work、Conclusions、Acknowledgments、Contributions / Author List、References / Bibliography、Appendix——这些章节不进 `source_sections[]`，也不承载 figures / equations / tables / videos。每个 `figure` / `equation` / `table` / `video` 必须挂 `section_id` 指向它在源里所属的章节——这决定了 Step 3 compose 时它出现在笔记的哪一段。
  - **Website**（若可用）：抽取 figures + videos。
  - **GitHub README**（若可用）：(1) **scope** ——  有无 training 代码？只有 inference？两者都有？都没有 → `unclear`。(2) **released_models** —— README 里明确的 checkpoint / weight 列表。

**抽取的硬性禁区**（不靠 grep，靠诚实自律）：
- ❌ 把 WebFetch summarizer 用文字描述的图当成"存在的图"——必须在本地源文件里看到可用的图片 URL
- ❌ 把 bar/line chart 目测的数字写成 Markdown 表——`tables[]` 只能从源里真实存在的数字表格复制
- ❌ 把"论文领域常见公式"凭印象补进 Method 段——`equations[]` 只收源里真实列出的公式
- ❌ 捏造次源 URL、checkpoint 名字、benchmark 结果

抽取完成，草稿里的每条 figure / equation / table / video / code_assessment 条目都应该能在对应的 local_path 文件里找到实锤——你 **不**被强制跑 grep 证明，但写的时候就得能指出来它从哪来。

---

### Step 3：笔记生成与保存

**Compose 的硬约束**：所有出现在笔记里的 URL、数字、公式、表格，**必须**来自 Step 2 extraction 草稿。compose 阶段不允许引入任何新的事实性元素。

**两个参考文件**（都在本 skill 目录内的 `references/`，即 `{skill_dir}/references/...`，**不是** vault 根目录的 `references/`）：
- **模板**：`{skill_dir}/references/paper-note-template.md` —— frontmatter + 固定 shell 骨架（top + outro）
- **Syntax 参考**：`{skill_dir}/references/obsidian-syntax.md` —— 公式 / 图 / 视频 / 表格的 Obsidian-specific 语法 quirks

#### 结构模型：固定 shell + 源结构 body

**固定 shell**（每篇笔记不变）：
- **Top**: frontmatter → `## Summary`
- **Outro**: `## 论文点评`→ `## 关联工作` → `## 速查卡片` → `## Notes`

**Body**（镜像源结构）：
- 逐个展开 `source_sections[]`：用每个 section 的 `heading`（源标题原文）作为笔记 body 的标题。层级（`##` / `###`）根据源结构和阅读顺畅度判断，不强制
- 在每个 section 下，根据需要嵌入合适的 figures / equations / tables / videos，**就近放置**

**Teaser 处理**：若 extraction 草稿里存在一个明确是 overview / concept / motivation 用途的 figure 或 video，嵌入到笔记里。若论文唯一的 high-level 视觉就是架构图，跳过 teaser——架构图自己会出现在对应的 body section 里。

#### 不变的硬规（所有 content_type）

- **装饰性图片**：OG image / social card / banner / favicon 一律不嵌入。
- **模板注释**：模板里的 `%% ... %%` 提示必须全部替换或整块删除，禁止外泄到正文。
- **媒体语法**：YouTube / 外链 mp4 / 外链图 / 本地图 / 公式的写法全部按 `obsidian-syntax.md`。
- **不写 wikilink**：所有技术术语和相关工作（`π0`、`Flow Matching`、`OpenVLA`等）都保持**纯文本**。

#### 文件名 + Tag + Write

**文件名格式**：`YYMM-ShortTitle.md`

- `YYMM`：取自 `date_publish` 的年份后两位 + 月份，如 `2603`（2026年3月）
- `ShortTitle`：标题的 CamelCase 缩写，2-4 个关键词，如 `EvoScientist`、`RoboClaw`、`DiffusionPolicy`
- Blog 同理，如一篇 2026 年 2 月的 blog 关于 scaling laws → `2602-ScalingLaws.md`

**Tag 选择**：阅读 vault 根目录下的 `references/tags.md`（即 `{vault_root}/references/tags.md`，**不是**本 skill 内的 `references/`），按照规范选择 tag。

**写入文件**：用 Write 将笔记保存到 `Papers/{笔记文件名}.md`。

---

### Step 4：Wikilink 注入（独立的 post-compose pass）

**目的**：扫描 Step 3 写出的笔记正文，识别技术术语，把能在 vault 找到对应笔记的 term 就地升级成 `[[...|term]]` wikilink。Compose 和 graph building 解耦，术语识别和 Concept Glob 协议集中在这一步处理。

**输入**：Step 3 刚写入的 `Papers/{笔记文件名}.md`（extraction 草稿已用完，不再参与）

**流程**：

1. **Read 笔记正文**，跳过 frontmatter、代码块、表格分隔线
2. **识别候选术语**（LLM 判断，不是机械正则）：
   - 论文名（`π0`、`RT-2`、`OpenVLA`、`Diffusion Policy`）
   - 核心方法 / 算法名（`Flow Matching`、`FAST tokenizer`、`Action Chunking`）
   - 数据集 / benchmark 名（`OXE`、`DROID`、`Bridge`）
   - 具体模型家族（`PaliGemma`、`Gemma-2B`、`SigLIP`）
   - **排除**：宽泛类别词（`VLA`、`world model`、`RL`、`transformer`）、通用英文词（`method`、`results`）、论文自己的术语
3. 对每个候选 term 执行 **Concept Glob 协议**
4. 命中 → 用**上下文锚定的 per-occurrence Edit**（**不是** `replace_all`）把笔记里出现的 term 包成 `[[{文件名}|{term}]]`（详见下方 *注入写法*）
5. 未命中 → 记入 dead-link inventory，term 保持纯文本

#### Concept Glob 协议

vault 的笔记命名约定是 `YYMM-CamelCase.md`（如 `2411-WorldModelSurvey.md`、`2410-Pi0.md`、`2604-GEN1.md`），**不是** `2411-World Model Survey.md`。所以朴素的 `Glob Papers/*<term>*.md` 对多词术语**几乎一定漏匹配**。

**匹配规则**：把 term 折叠成 CamelCase（去掉空格和 `-`），然后 Glob 三个可能的目录：

```
Glob Papers/*<TermCamelCase>*.md
Glob Concepts/*<TermCamelCase>*.md     # 若目录存在
Glob DomainMaps/*<TermCamelCase>*.md   # 若目录存在
```

例：
- `"World Model Survey"` → `*WorldModelSurvey*.md`（命中 `Papers/2411-WorldModelSurvey.md`）
- `"Flow Matching"` → `*FlowMatching*.md`
- `"GEN-1"` → `*GEN1*.md`
- `"π0.5"` → `*Pi05*.md`（Unicode / 特殊字符按读音或常用 ASCII 折叠）

任一目录命中即视为存在。命中后 Read 文件 frontmatter 的 `title:` 字段**确认主题相关**再注入——Glob 可能有无关的子串 false positive（如 `*Flow*` 匹配 `SnapFlow`）。

#### 注入写法

**绝对禁止使用 `replace_all=true`**。即使 term 看起来只出现一次，也必须用**上下文锚定的 per-occurrence Edit**。原因：技术术语互为前缀 / 子串是常态——

| 长 term | 短 term（被污染风险） |
|---|---|
| `π0.5`, `π0-FAST` | `π0` |
| `OpenVLA-OFT` | `OpenVLA` |
| `GEN-1`, `GEN-01` | `GEN-0` |
| `RT-2` | `RT-1` |
| `GPT-4o` | `GPT-4` |
| `Papers/2405-Octo.md`（路径） | `Octo` |

`replace_all` 对以上场景会：
- **污染已注入的 alias 文本**：先注入 `[[2504-Pi05|π0.5]]`，再 `replace_all("π0", ...)` → `[[2504-Pi05|[[2410-Pi0|π0]].5]]`（真实事故，见 2026-04-14 subagent 测试）
- **污染文件路径字符串**：`关联工作` 段如果出现 `Papers/2405-Octo.md`，`replace_all("Octo", ...)` → `Papers/2405-[[2405-Octo|Octo]].md`

**注入流程**（对每个命中 term，按顺序执行）：

1. **按长度降序处理 term**。先把 `π0.5` 全部注入完，再处理 `π0`。这样短 term 注入时，长 term 已经在 alias 里被保护了（alias 的前后会有 `|` 和 `]]`，可用作上下文锚点过滤）。
2. **Grep 枚举**：`Grep -n "<term>" <笔记路径>` 列出所有 occurrence 的行号和周围文本。
3. **过滤排除区**（必跳过）：
   - Frontmatter 区（`---` 之间）
   - 代码块（fenced ``` 或缩进 4 空格）
   - 已 wikilink：occurrence 左右窗口包含 `[[` / `|` / `]]`（说明该字符序列是已注入的 alias 或已有 wikilink 的一部分）
   - **更长术语的子串**：occurrence 右侧紧跟 `.5` / `-FAST` / `-OFT` / `-1` / `-2` / `o` 等后缀，或左侧被字母数字字符相邻（说明这是 `π0.5` / `OpenVLA-OFT` 等长 term 的一部分，不是 term 本身）
   - **文件路径**：occurrence 左侧包含 `Papers/` 或 `.md` 紧邻（说明是路径字符串的一部分）
4. **Per-occurrence Edit**：对每个合法 occurrence，用**包含周围字符的 old_string** 让匹配唯一，new_string 只把 term 本身包成 `[[文件|term]]`：

   ```
   # 例：注入 π0（π0.5 已注入完）
   # Grep 出两处：
   #   L42: "... baseline 是 π0 (Black et al.)"
   #   L58: "... π0.5 surpasses π0 in all tasks"  ← 右边是空格，合法
   
   Edit(old="baseline 是 π0 (Black", new="baseline 是 [[2410-Pi0|π0]] (Black")
   Edit(old="surpasses π0 in",       new="surpasses [[2410-Pi0|π0]] in")
   ```

   **表格 cell 内的 occurrence 必须用 `\|` 转义 alias 分隔符**（详见 `obsidian-syntax.md` §4.1）——否则 Markdown 会把 alias 的 `|` 当成列分隔符，wikilink 断裂。例：

   ```
   # 表格行：| Flow matching | π0 (Black et al.) | ...
   Edit(old="| π0 (Black", new="| [[2410-Pi0\\|π0]] (Black")
   ```

**为什么不用 replace_all**：省下的几次 Edit 调用代价是**静默损坏笔记**。Per-occurrence Edit 多几次工具调用，但每次替换都有上下文锚点，不会污染长 term 或文件路径。

**泛化类别词反模式**：宽泛类别词（`VLA` / `world model` / `scaling law` / `transformer`）在**候选识别阶段**就应被排除；即使漏网到 Glob 阶段并命中，也**不**自动 wikilink——它们在 vault 里可能有匹配文件，但论文正文用的是作为范畴词，不是指那篇具体 reference。

#### Dead-link Inventory

注入流程结束后，把所有**未命中**的 term 汇总成一个 dead-link inventory，写入 Step 5 的 log entry（见下一节）。这是 vault 长期建设的 TODO 信号，不写入笔记本身。

---

### Step 5：日志记录

用 Edit（或 Write 若文件不存在）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

   ```markdown
   ### [HH:MM] paper-digest
   - **input**: <source 的原始内容>
   - **output**: [[Papers/{笔记文件名}]]
   - **observation**: <一句话描述论文核心贡献>
   - **dead_links**: [term1, term2, ...]  # Step 4 未命中的 term，vault curation TODO
   - **status**: success
   ```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

### Step 2 (Fetch & Extract) 自检
- [ ] 源材料已落盘到 `/tmp/...`，路径已记录
- [ ] extraction 草稿已产出
- [ ] **`source_sections`** 已按源顺序记录大纲；每个 figure / equation / table / video 都挂了 `section_id`
- [ ] extraction 草稿里的每条 figure / table / equation / video 都能指向源的具体出处，没有凭印象补的条目

### Step 3 (Compose & Save) 自检
- [ ] `Papers/{笔记文件名}.md` 已创建
- [ ] 笔记中所有 URL / 数字 / 公式 / 表格 **均来自 Step 2 extraction 草稿**，无新增
- [ ] **Body 章节结构来自 `source_sections`**（标题用每个 section 的 `heading` 原文；层级根据源结构判断）
- [ ] 每个 figure / equation / table / video 嵌在 `section_id` 对应的 section，**就近放置**
- [ ] Teaser 媒体（若存在）嵌在 `## Summary` 段内
- [ ] 未嵌入 OG image / social card / banner 等装饰性图片

### Step 4 (Wikilink 注入) 自检 

**Read 最终写入的笔记文件**，肉眼逐个检查每个 `[[...]]` occurrence 是否合法。对每个 wikilink 过一遍下面的问题，任一 "否" 即需修复：

- [ ] **结构合法**：是 `[[文件名|显示文本]]` 形式，`[[` 和 `]]` 成对，`|` 恰好一个（表格 cell 内的 alias 分隔符已用 `\|` 转义）
- [ ] **无嵌套**：`|` 之后到 `]]` 之间是纯文本，不含另一个 `[[` —— 没有 `[[...|[[...|term]]]]` 这种双重嵌套
- [ ] **未污染路径**：wikilink 没有出现在 `Papers/xxxx-xxx.md` 这类文件路径字符串的中间（即路径里的 term 没有被错误包装成 `Papers/2405-[[2405-Octo|Octo]].md`）
- [ ] **未污染更长术语**：短 term（如 `π0`）的注入没有破坏更长术语的 alias（如 `[[2504-Pi05|π0.5]]` 里的 `π0` 没有被再次包装）

### Step 5 (Log) 自检
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`
- [ ] Log entry 包含 `dead_links` 列表
