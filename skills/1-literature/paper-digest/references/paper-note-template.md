---
title:
authors:            # [author1, author2, ...]
institutes:         # [institute1, institute2, ...]
date_publish:
venue:
tags:               # [tag1, tag2, ...]
paper:
website: 
github:
rating:              # {3 / 2 / 1}（在正文 ### Rating 段决定后回填此字段）
date_added:          # 填写今天的日期 YYYY-MM-DD
---

## Summary

> [!summary] {Paper Title}
> - **核心**: {一句话核心}
> - **方法**: {关键方法}
> - **结果**: {主要结果}
> - **Sources**: [paper](paper_url) | [website](website_url) | [github](github_url) %% 三类 slot 对应 frontmatter 的 paper / website / github 字段，不互斥——有几类就保留几类，缺失的整段删除（含 `|` 分隔符）。label 文本固定不改，只替换括号内 URL。 %%
> - **Rating**: {3 - Foundation / 2 - Frontier / 1 - Archived} （一句话理由） %% 从下文 `### Rating` 压缩而来：`### Rating` 先写完整 justification，这里再回写成一句话。 %%

**Key Takeaways:** 
1. **{Takeaway 1}**: 简要说明
2. **{Takeaway 2}**: 简要说明
3. ...

**Teaser. {描述}** 
%% 若有独立 teaser 图/视频，在这里直接嵌入；若无，则跳过 %%

---
%% ═══ Body （内容解读）  ═══  %%

%% 可用构件 1：嵌入图 %%
**Figure #. {描述}**
![](https://.../figure.png)

%% 可用构件 2：嵌入公式 %%
**Equation #. {公式名}**

$$
{公式内容}
$$

**符号说明**：
**含义**：

%% 可用构件 3：嵌入数字表 %%
**Table #. {描述}**

| Col1 | Col2 |
| ---- | ---- |
| ...  | ...  |

%% 可用构件 4：嵌入视频 %%
**Video #. {描述}**
<video src="https://.../clip.mp4" controls muted playsinline width="720"></video>

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

#### Artifact 可获取性
%% 结合Github README和正文内容分析,未知就写 "未说明"，严禁推测。 %%
- **代码**: {inference-only / inference+training / 未开源}
- **模型权重**: {已发布的 checkpoint 名字与描述}
- **训练细节**: {超参 + 数据配比 + 训练步数完整 / 仅超参 / 仅高层描述 / 未披露}
- **数据集**: {开源（名字 + 链接）/ 私有 / 部分公开}

#### Claim 可验证性
%% 用 ✅/⚠️/❌ 三档对核心 claim 分类。若全部 ✅ 也要写出来，明确 "无 ⚠️/❌"，避免漏过潜在的 marketing 修辞 %%
- ✅ {可验证 claim}：{grounding——论文实验/视频/独立复现}
- ⚠️ {半可信 claim}：{打折原因——归因不严、样本量不明、定义模糊}
- ❌ {营销话术}：{为什么不算技术 claim——如 "first to cross commercial viability"}

### Notes
%% 其他想法、疑问、启发。留空供后续填写。 %%

### Rating

%% Rating Rubric — 语义：这篇论文（方法、benchmark/dataset）在我关心的研究方向里的位置。

- **3 — Foundation**：方向必读、必引的奠基工作。
  - 方法：具备开创性 / 奠基性 / 颠覆性，社区影响力大（citation 高，github repo star 多等）。
  - Benchmark / Dataset：已成为方向的 de facto 标准评测或基础数据（如 ImageNet、DROID 级别）。
  - 准入门槛：只读 rating=3 的论文就能理解这个方向的主要脉络和发展过程。

- **2 — Frontier**：方向的研究前沿和重要参考。
  - 方法：当前或最近的 SOTA、必须比较的 Baseline、方法范式的代表工作。
  - Benchmark / Dataset：有一定使用量、正被主要工作采用，可能成长为标准但尚未定型。

- **1 — Archived**：读过但不在方向主脉络和前沿。
  - 方法：incremental / niche / 为某个具体问题查的一次性参考 / 被后续工作取代。
  - Benchmark / Dataset：使用量低 / 范围过窄 / 被更通用的 benchmark 取代。
  - 预期不再主动翻。

注：rating 是动态的——SOTA 会过气降为 1；被时间验证的重要工作可能从 2 升为 3；benchmark 也会随社区采纳情况升降；方向 pivot 或新证据出现后需要重新评估。 %%

**分数**：{3 - Foundation / 2 - Frontier / 1 - Archived}
**理由**：{2-3 句，必须 grounded 在本笔记已写内容上，要引用已写内容的具体判断，而非凭印象。说清楚"为什么是这个档，而不是相邻档"。}

%% 决定分数后，回填：(1) Summary 里的 Rating 一句话，(2) frontmatter 的 rating 字段。三处保持一致。 %%
