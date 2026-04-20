---
name: paper-digest
description: Use when user asks to "read paper", "analyze paper", "summarize paper", "读论文", "分析文献", "帮我看一下这篇paper", "论文笔记", or provides a PDF file that appears to be an academic paper.
argument-hint: "[URL / PDF path / title]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, WebSearch, WebFetch
---

## Purpose

给定一篇论文或技术 blog 的来源（URL、PDF 路径、论文标题或者关键词），它自动获取内容、提炼核心信息，生成结构化笔记。

支持两种内容类型：
- **paper**：学术论文
- **blog**：所有 blog 类文章（tech blog，product blog 等）

## Steps

### Step 1：Dedup 预检

先扫一遍 `Papers/`，避免对一篇已有笔记重复 digest。若找到对应笔记 → **立刻停止**，告知用户已有笔记 [[xxx]]，询问是否需要 覆盖 / 更新 / 取消

### Step 2：Fetch

#### Step 2.1：寻找官方源

一篇内容常有**三类官方源**：
- **paper**：arXiv 或者 PDF URL，论文的权威文本来源
- **website**：project page、官方博客文章等
- **github**：代码仓库的 README

根据输入信息寻找这三类官方源
**用 WebSearch 兜底，不可跳过**
综合判断找到的源是否来自于官方，若不是，则丢弃

#### Step 2.2：抓取源、生成源文件

对上一步发现的每类源，按下表抓取到 **`/tmp/{ShortTitle}/`** 目录下。先创建目录。

| 源       | 抓取方式                                                                                                                                                                                                                                                                                                                                                                                               |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| paper   | **arxiv**: Bash `npx defuddle parse <url> --json --md -o /tmp/{ShortTitle}/paper.json`。先试 `"https://arxiv.org/html/{arxiv_id}"`，失败则 `"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"`。<br>**PDF**: Bash `curl -sL <url> -o /tmp/{ShortTitle}/paper.pdf && uv run {skill_dir}/scripts/pdf_to_md.py /tmp/{ShortTitle}/paper.pdf` → 输出 `paper.md` + `images/`。 |
| website | Bash `npx defuddle parse "<url>" --json --md -o /tmp/{ShortTitle}/website.json`                                                                                                                                                                                                                                                                                                       |
| github  | Bash `curl -sL https://raw.githubusercontent.com/<org>/<repo>/main/README.md -o /tmp/{ShortTitle}/github.md`。若 main 分支不存在，试 master。                                                                                                                                                                                                                                                   |

**记下每类 source 的 `local_path`。**

#### Step 2.3：检查生成的源文件

读取生成的源文件，检查是否出现以下问题：

| 问题                           | 应用源     | 失败处理                                  |
| ---------------------------- | ------- | ------------------------------------- |
| 文件不完整                        | 全部      | 返回上一步，重新抓取                            |
| video 和 figure 的 URL 丢失或者不完整 | website | 用 `curl` 抓取 raw HTML，找到完整 URL 后补充到源文件 |

---

### Step 3：笔记生成与保存

仔细阅读生成的所有源文件，生成笔记。

**读取两个参考文件**（都在本 skill 目录内，`{skill_dir}/references/...`）：
- **模板**：`paper-note-template.md` 
- **Syntax 参考**：`obsidian-syntax.md` —— 公式 / 图 / 视频 / 表格的 Obsidian-specific 语法 quirks

#### 笔记结构

`## Summary`→ **Body**（内容解读） →`## 论文点评` → `## 关联工作` → `## Notes`

**撰写顺序**：先写 Body 等其他内容，最后回头写 `## Summary`——它是对全文的压缩，依赖对内容的完整理解。

**Body**（内容解读）：
- **批判性阅读**：综合源文件内容和你自己的理解，生成笔记，不要机械翻译
- 综合参考源文件的结构确定层级结构，**要保留源内容的核心结构和要素**，不要过于简略和压缩内容
- 从所有源文件中，按需挑选 figures / equations / tables / videos 嵌入合适位置
- 加入你自己的理解，标出有疑问的地方：`> ❓ ...`

**Teaser 处理**：若某类源文件里存在明确是 overview / concept / motivation 用途的 figure 或 video（通常在 intro / abstract 段），嵌入到笔记 `## Summary` 下。

#### 不变的硬规

- **装饰性图片**：OG image / social card / banner / favicon 一律不嵌入。
- **模板注释**：模板里的 `%% ... %%` 提示必须全部替换或整块删除，禁止外泄到正文。
- **PDF 源的 images**：把需要嵌入的图从 `/tmp/{ShortTitle}/images/` 拷到 `Papers/assets/{ShortTitle}/`，再用 `![](assets/{ShortTitle}/fig_N.png)` 引用。

#### 文件名 + Tag + Write

**文件名格式**：`YYMM-ShortTitle.md`

- `YYMM`：取自 `date_publish` 的年份后两位 + 月份，如 `2603`（2026年3月）
- `ShortTitle`：标题的 CamelCase 缩写，2-4 个关键词，如 `EvoScientist`、`RoboClaw`、`DiffusionPolicy`
- Blog 同理，如一篇 2026 年 2 月的 blog 关于 scaling laws → `2602-ScalingLaws.md`

**Tag 选择**：阅读 vault 目录下的 `{vault_root}/references/tags.md`，按照规范选择 tag。

**写入文件**：将笔记保存到 `Papers/{笔记文件名}.md`。

---

### Step 4：Wikilink 注入

扫描写出的笔记正文，识别技术术语，把能在 vault 找到对应笔记的 term 就地升级成 `[[...|term]]` wikilink。

**⚠️注意**：
1. **只处理笔记正文**，跳过 frontmatter、代码块
2. **识别候选术语**：论文（`π0`、`RT-2`），核心方法 / 算法（`Flow Matching`、`FAST tokenizer`、`Action Chunking`）， 数据集（`DROID`），模型（`PaliGemma`、`Gemma-2B`、`SigLIP`）。 **排除**：宽泛类别词（`VLA`、`world model`、`RL`、`transformer`）、论文自己的术语
3. 对每个候选 term 执行 **Concept Glob 协议**
4. 把所有**未命中**的 term 汇总成一个 dead-link inventory
5. **（必须做！！！）检查所有插入的 wikilink，修正所有不合理的情况** 

#### Concept Glob 协议

vault 的笔记命名约定是 `YYMM-CamelCase.md`（如 `2411-WorldModelSurvey.md`、`2410-Pi0.md`、`2604-GEN1.md`），**不是** `2411-World Model Survey.md`。所以朴素的 `Glob Papers/*<term>*.md` 对多词术语**几乎一定漏匹配**。

**匹配规则**：把 term 折叠成 CamelCase（去掉空格和 `-`），然后 Glob 三个可能的目录：

```
Glob Papers/*<TermCamelCase>*.md
Glob Concepts/*<TermCamelCase>*.md     # 若目录存在
Glob DomainMaps/*<TermCamelCase>*.md   # 若目录存在
```

例：
- `"Flow Matching"` → `*FlowMatching*.md`
- `"GEN-1"` → `*GEN1*.md`
- `"π0.5"` → `*Pi05*.md`（Unicode / 特殊字符按读音或常用 ASCII 折叠）

任一目录命中即视为存在。命中后 Read 文件 frontmatter 的 `title:` 字段**确认主题相关**再注入——Glob 可能有无关的子串 false positive（如 `*Flow*` 匹配 `SnapFlow`）。

---

### Step 5：日志记录

用 Edit（或 Write 若文件不存在）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

   ```markdown
   ### [HH:MM] paper-digest
   - **input**: <source 的原始内容>
   - **output**: [[Papers/{笔记文件名}]]
   - **observation**: <一句话描述论文核心贡献>
   - **extracted**: figures=<N>, tables=<N>, equations=<N>, videos=<N>  # Step 2 所有源文件里各类元素的总数（跨源去重后的 ceiling，即笔记最多能嵌多少）
   - **embedded**: figures=<N>, tables=<N>, equations=<N>, videos=<N>, wikilinks=<N>  # 最终笔记里实际出现的各类元素条目数
   - **dead_links**: term1, term2, ...  # Step 4 未命中的 term
   - **issues**: <执行过程中遇到的问题，如源获取失败、defuddle 解析失败；无问题则写 "none">
   ```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

### Step 2 (Fetch) 自检
- [ ] 三类源（paper / website / github）均已尝试发现；WebSearch 兜底已对每一类跑过（不可跳过）
- [ ] paper / website 源已通过 defuddle 抓到 `/tmp/{ShortTitle}/paper.json` 或 `website.json`；github 源已通过 `curl` 抓到同目录下的 `github.md`
- [ ] Step 2.3 两项检查（文件完整性；website 的 video/figure URL 完整性）均已过；未过的已记入 Step 5 log 的 `issues`

### Step 3 (Compose & Save) 自检
- [ ] `Papers/{笔记文件名}.md` 已创建
- [ ] 笔记中所有 URL / 数字 / 公式 / 表格 **均可在 Step 2 的某类源文件里 grep 到**，无新增
- [ ] 笔记 body 的 prose（方法名、技术 claim、数值）**均可追溯到主文本源文件对应 section 的原文**，无凭印象补写
- [ ] 每个 figure / equation / table / video **就近放置**——放在相关讨论的段落附近
- [ ] Teaser 媒体（若存在）嵌在 `## Summary` 段内
- [ ] 未嵌入 OG image / social card / banner 等装饰性图片
- [ ] **每个嵌入元素都有紧邻的描述**，说明这张图/表/视频在展示什么，而不是光秃秃嵌入或重复 section 标题
- [ ] **每个元素的嵌入位置与其描述在语境上一致**：元素不应被放在会误导读者的上下文（例：一段讲 "zero-shot 失败" 的文字下方不能紧接一个展示成功的视频）。若描述和周围文字冲突，更换嵌入的位置

### Step 4 (Wikilink 注入) 自检 

**Read 最终写入的笔记文件**，肉眼逐个检查每个 `[[...]]` occurrence 是否合法。对每个 wikilink 过一遍下面的问题，任一 "否" 即需修复：

- [ ] **结构合法**：是 `[[文件名|显示文本]]` 形式，`[[` 和 `]]` 成对，`|` 恰好一个（表格 cell 内的 alias 分隔符已用 `\|` 转义）
- [ ] **无嵌套**：`|` 之后到 `]]` 之间是纯文本，不含另一个 `[[` —— 没有 `[[...|[[...|term]]]]` 这种双重嵌套
- [ ] **未污染路径**：wikilink 没有出现在 `Papers/xxxx-xxx.md` 这类文件路径字符串的中间（即路径里的 term 没有被错误包装成 `Papers/2405-[[2405-Octo|Octo]].md`）
- [ ] **未污染更长术语**：短 term（如 `π0`）的注入没有破坏更长术语的 alias（如 `[[2504-Pi05|π0.5]]` 里的 `π0` 没有被再次包装）

### Step 5 (Log) 自检
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`
