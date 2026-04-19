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
rating:              # 1=有参考价值, 2=重要, 3=必读
date_added:          # 填写今天的日期 YYYY-MM-DD
---
## Summary

> [!summary] {Paper Title}
> - **核心**: {一句话核心}
> - **方法**: {关键方法}
> - **结果**: {主要结果}
> - **Sources**: [paper](paper_url) | [website](website_url) | [github](github_url)
> %% 三类 slot 对应 frontmatter 的 paper / website / github 字段，不互斥——有几类就保留几类，缺失的整段删除（含 `|` 分隔符）。label 文本固定不改，只替换括号内 URL。 %%

**Key Takeaways:** 
1. **{Takeaway 1}**: 简要说明
2. **{Takeaway 2}**: 简要说明
3. ...

**Teaser. {描述}** 
%% 若有独立 teaser 图/视频，在这里直接嵌入；若无，则跳过 %%

---
<!-- ═══ Body （内容解读）  ═══ -->

<!-- 可用构件 1：嵌入图 -->
**Figure #. {描述}**
![](https://.../figure.png)

<!-- 可用构件 2：嵌入公式 -->
**Equation #. {公式名}**

$$
{公式内容}
$$

**符号说明**：
**含义**：

<!-- 可用构件 3：嵌入数字表 -->
**Table #. {描述}**

| Col1 | Col2 |
| ---- | ---- |
| ...  | ...  |

<!-- 可用构件 4：嵌入视频 -->
**Video #. {描述}**
<video src="https://.../clip.mp4" controls muted playsinline width="720"></video>

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
## Notes
%% 其他想法、疑问、启发。留空供后续填写。 %%
