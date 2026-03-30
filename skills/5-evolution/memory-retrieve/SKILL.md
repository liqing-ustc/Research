---
name: memory-retrieve
description: >
  被其他 skill 内部调用，从 Workbench/memory/ 中检索与当前任务相关的历史经验。
  当 idea-generate 需要查失败方向、experiment-design 需要查有效方法、
  或任何 skill 需要历史上下文时调用
argument-hint: "<query> [scope]"
user-invocable: false
allowed-tools: Read, Glob, Grep
---

## Purpose

memory-retrieve 是 MindFlow 记忆系统的检索接口。它被其他 skill 内部调用（如 idea-generate 查失败方向、experiment-design 查有效方法），从 `Workbench/memory/` 的结构化记忆文件中检索与当前任务语义相关的历史经验。

这是 Layer 1 的轻量实现——纯 Markdown 文件读取 + LLM 语义匹配，零外部依赖。未来 Layer 2 可用向量检索替换，接口不变。

## Steps

### Step 1：确定检索范围

根据 `scope` 参数确定要读取的记忆文件：

| scope | 目标文件 |
|:------|:---------|
| `patterns` | `Workbench/memory/patterns.md` |
| `insights` | `Workbench/memory/insights.md` |
| `effective-methods` | `Workbench/memory/effective-methods.md` |
| `failed-directions` | `Workbench/memory/failed-directions.md` |
| `all` | 以上全部 |

### Step 2：读取记忆文件

用 Read 读取 scope 对应的记忆文件。若文件不存在或为空，返回空结果并告知调用方"该记忆类别暂无条目"。

### Step 3：语义匹配

逐条扫描记忆文件中的条目（每个 `### [YYYY-MM-DD]` 标题为一条），判断其与 `query` 的语义相关性。判断标准：

- 条目的 observation/claim/method 是否与 query 描述的任务/问题直接相关
- 条目中引用的论文/实验是否与 query 涉及的领域重叠
- 条目的 lesson/pitfall 是否对 query 的任务有参考价值

### Step 4：返回结果

返回 top-k 相关条目（默认 k=5），每条包含：

- 条目原文（完整保留，不做概括改写）
- 来源文件路径（如 `[[Workbench/memory/patterns.md#条目标题]]`）

若相关条目不足 k 条，返回全部匹配条目，不补凑。

## Guard

- **只读**：不修改任何记忆文件，不修改任何 vault 文件
- **原文保真**：返回结果必须包含条目原文和 `[[wikilink]]` 来源引用，不做概括性改写
- **诚实返回**：若无相关条目，返回空结果，不编造记忆内容
