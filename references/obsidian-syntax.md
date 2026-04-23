# Obsidian Syntax 参考

---
## 0. Markdown 特殊字符

术语名称中的 `*` 在 Markdown 中会触发斜体/加粗。写入笔记时必须转义为 `\*`：

```markdown
<!-- ✗ 错误：*会被解析为斜体标记 -->
π*₀.₆

<!-- ✓ 正确 -->
π\*₀.₆
```

适用于正文、列表、表格 cell 等**纯文本上下文**中的 `*` 术语（如 `π*₀.₆`、`RL*` 等）。

**例外：wikilink alias 内不转义**。`[[...|...]]` 的 alias 段不走 Markdown 解析，`*` 保持字面：

```markdown
<!-- ✗ 错误：alias 里的 \* 会被字面渲染成反斜杠 -->
[[2511-PiStar06|π\*0.6]]

<!-- ✓ 正确 -->
[[2511-PiStar06|π*0.6]]
```

---

## 1. 公式

### 1.1 公式格式

````
$$
{公式内容}
$$
````

- `$$` 块**前后必须有空行**——否则 Obsidian 不渲染
- 超长公式（> 80 字符）用 `\begin{aligned}` 拆分

### 1.2 Obsidian MathJax 安全命令

1. **安全命令**：`\operatorname{}`、`\text{}`、`\begin{aligned}`、`\underbrace{}`
2. **括号处理**：`\Big` / `\bigg` 优于 `\left` / `\right`（后者在某些环境会失败）

---

## 2. 图片

### 2.1 外链优先

arXiv HTML / 项目主页 / GitHub 的图都用外链直接嵌入。**找不到外链 URL 才本地下载**到 `assets/` 用 `![[]]` 引用——本地下载的成本是 vault 体积膨胀和 git diff 噪声。

### 2.2 嵌入语法

| 来源 | 写法 |
|---|---|
| 外链 | `![](https://.../figure.png)` |
| 本地（vault 内） | `![[clip.png]]` 或 `![](assets/clip.png)` |

---

## 3. 视频

| 来源                | 写法                                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **YouTube**       | `![](https://www.youtube.com/watch?v=VIDEO_ID)` —— Obsidian 原生支持，markdown 图片语法直接渲染播放器（`https://youtu.be/VIDEO_ID` 同样有效） |
| **外链 mp4**        | `<video src="..." controls muted playsinline width="720"></video>` —— **width 固定为 720**，attributes 顺序固定                 |
| **本地文件（vault 内）** | `![[clip.mp4]]` 或 `![](clip.mp4)`                                                                                       |

**Obsidian 的 `![](url.mp4)` 对外链 mp4 完全失效**（只显示 broken media icon），**禁止用此写法**。`<video>` 必带 `controls` 才有播放条；`muted playsinline` 避免移动端干扰。如果某些 CDN 没开 CORS / Range request，Obsidian 拉不到会黑屏，这种情况只能下载到 vault 用 `![[]]` 嵌。

---

## 4. Wikilink

**alias 形式**: `[[文件名|术语原文]]`，链接靠真实文件名解析，显示文本保留术语自然写法。

- 例：`[[2411-WorldModelSurvey|World Model Survey]]`
- 例：`[[2410-Pi0|π0]]`
- 例：`[[2604-GEN1|GEN-1]]`

这样既保证 Obsidian 的 wikilink 能正确解析（靠文件名），又不破坏正文的可读性（显示术语原文而非笔记文件名）。

### 4.1 Markdown 表格里的 alias —— pipe 必须转义

**坑**：alias 形式的 `|` 和 Markdown 表格的列分隔符冲突。`| [[2410-Pi0|π0]] | ...` 会被解析成 4 列：`[[2410-Pi0`、`π0]]`、`...`——wikilink 断裂。

**写法**：在表格 cell 里用 `\|` 转义 alias 的分隔符：

```markdown
| Method | Paper |
|---|---|
| Flow matching | [[2410-Pi0\|π0]] |
| Diffusion  | [[2405-DiffusionPolicy\|Diffusion Policy]] |
```

渲染时 Obsidian 把 `\|` 还原成 `|`，wikilink 正常显示 alias 文本。

**适用范围**：只在表格 cell 里需要转义。正文、列表、callout 里的 `[[...|...]]` 不必改。Step 4 注入时如果 occurrence 在表格 cell 内，new_string 必须用 `\|` 形式。

---

## 5. 表格

Markdown 表格**前后必须有空行**——否则 Obsidian 不渲染为表格，而是显示为纯文本管道符。与 `$$` 公式块同理。

```markdown
<!-- ✗ 错误：表格紧贴上方文本，不渲染 -->
推理时间如下：
| 模型部分 | 推理时间 |
|---|---|
| Image encoders | 14 ms |

<!-- ✓ 正确：表格前后各有一个空行 -->
推理时间如下：

| 模型部分 | 推理时间 |
|---|---|
| Image encoders | 14 ms |

...
```
