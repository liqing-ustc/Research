---
title:
authors:            # [author1, author2, ...]
institutes:         # [institute1, institute2, ...]
date_publish:
venue:
tags:               # [tag1, tag2, ...]
url:                # 主源 URL（arxiv / blog / PDF）
website:            # 官方 project page URL；无则留空
code:               # 官方 GitHub repo URL；无官方 repo 则留空，禁止填 community fork
rating:              # 1=有参考价值, 2=重要, 3=必读
date_added:          # 填写今天的日期 YYYY-MM-DD
---
## Summary
%% 用一句话概括这篇论文的核心贡献，不超过50字 %%

**Key Takeaways:** 
1. **{Takeaway 1}**: 简要说明
2. **{Takeaway 2}**: 简要说明
3. ...

**Teaser. {caption}** 
%% 若源有独立 teaser 图/视频（ Abstract / Intro 里的 overview / concept / motivation 示意），在这里直接嵌入；若论文唯一的 high-level 视觉就是架构图，跳过不写，不留 placeholder %%

**Sources**: [arxiv](arxiv_url) | [blog](blog_url) | [website](website_url) | [github](github_url)
%% 保留存在的 slot，缺失的整段删除（含 `|` 分隔符）。label 文本固定不改，只替换括号内 URL。主源是 arxiv/PDF 用 `arxiv` slot，主源是博客/发布页用 `blog` slot——两者互斥，通常只出现一个。 %%

---
<!-- ═══ Body：章节结构镜像源文档，不是固定模板 ═══ -->
%% 
Body 区使用源文档的 section 结构。原则：
- 标题取自源的章节名原文（`heading` 字段），层级（`##` / `###`）根据源结构和阅读顺畅度判断，不强制
- 源里没有的 section 不写，不留 placeholder
- 源里有但 extraction 草稿没挖出任何内容的 section 也不写
- 图 / 公式 / 表 / 视频就近嵌入对应段落（section_id 匹配）
%%

%% 以下只是"可用构件"的展示，**不是**必须按序出现的章节。实际章节数量、顺序、标题、层级都由 `source_sections[]` 决定。每种构件（图 / 公式 / 表 / 视频）在自己的 section_id 对应的 section 里就近嵌入，不需要集中。%%

<!-- 可用构件 1：嵌入图 -->
**Figure #. {caption}**
![](https://.../figure.png)

<!-- 可用构件 2：嵌入公式 -->
**Equation #. {公式名}**
$$
{公式内容}
$$

**符号说明**：
**含义**：

<!-- 可用构件 3：嵌入数字表（仅当 extraction 草稿存在正式数字表时写；禁止从 bar/line chart 目测编造） -->
**Table #. {caption}**
| Col1 | Col2 |
| ---- | ---- |
| ...  | ...  |

**Insights**: {关键发现和解释}

<!-- 可用构件 4：嵌入视频（外链 mp4 用 <video>，YouTube 用 ![](watch?v=...)，详见 obsidian-syntax.md §3）-->
**Video #. {caption}**
<video src="https://.../clip.mp4" controls muted playsinline width="720"></video>

---
<!-- ═══ 固定 outro：vault 编辑 overlay，所有笔记一致 ═══ -->
## 论文点评
%% 方法亮点与局限的点评，以及对可复现性的评估。 %%

### Strengths

1. {优点1}
2. {优点2}
3. ...

### Weaknesses

1. {缺点1}
2. {缺点2}
3. ...

### 可信评估
%% 两个互不替代的子轴：artifact 层（有没有开源）+ claim 层（每个 claim 有没有 grounding）。任何 content_type 都填两栏。 %%

#### Artifact 可获取性
%% 结合Github README和正文内容分析，未知就写 "未说明"，严禁推测。 %%
- **代码**: {inference-only / inference+training / 未开源}
- **模型权重**: {已发布的 checkpoint 名字与描述}
- **训练细节**: {超参 + 数据配比 + 训练步数完整 / 仅超参 / 仅高层描述 / 未披露}
- **数据集**: {开源（名字 + 链接）/ 私有 / 部分公开}

#### Claim 可验证性
%% 用 ✅/⚠️/❌ 三档对核心 claim 分类。若全部 ✅ 也要写出来，明确 "无 ⚠️/❌"，避免漏过潜在的 marketing 修辞 %%
- ✅ {可验证 claim}：{grounding——论文实验/视频/独立复现}
- ⚠️ {半可信 claim}：{打折原因——归因不严、样本量不明、定义模糊}
- ❌ {营销话术}：{为什么不算技术 claim——如 "first to cross commercial viability"}

---
## 关联工作
%% 列出相关工作。无对应内容的子类直接删除；子类标签可自定义。 %%
### 基于
- {前置工作}: {说明}
- ...

### 对比
- {对比方法}: {为什么对比}
- ...

### 方法相关
- {核心技术}: {说明}
- ...

---

## 速查卡片

> [!summary] {Paper Title}
> - **核心**: {一句话核心}
> - **方法**: {关键方法}
> - **结果**: {主要结果}
> - **代码**: {GitHub链接}

---
## Notes
%% 其他想法、疑问、启发。留空供后续填写。 %%
