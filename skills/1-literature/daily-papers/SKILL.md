---
name: daily-papers
description: 每日论文总结。抓取 HuggingFace Daily/Trending + arXiv 最新论文，按研究方向打分筛选， 生成论文笔记后基于深度阅读写出有态度的总结锐评。 触发词："今日论文总结""过去3天论文总结""过去一周论文总结""看看最近有什么论文"
argument-hint: "[今日 / 过去N天 / 过去一周]"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

## Purpose

自动发现与研究方向相关的最新论文，生成候选列表。分三步：

1. **Python 脚本**抓取 + 打分（零 token）
2. **快速分流** → 确定必读论文
3. **每篇必读论文**：独立 subagent 生成笔记 + 基于笔记写单篇点评 → 主 agent 汇总锐评和分流表，保存文件

> **关键设计**：单篇点评由生成笔记的 subagent 在同一会话内完成——它此时刚读完论文、有完整上下文；让主 agent 读回笔记重写点评是对 context 的浪费，且摘要驱动的重写容易丢失笔记里的 ablation / caveat 细节。

## Steps

### Step 0：解析时间范围

从用户输入中解析天数：
- "今日论文总结"、"今日论文"、"每日总结" → 当天（`--days 1`）
- "过去3天"、"最近三天" → `--days 3`
- "过去一周"、"最近7天" → `--days 7`
- "过去两周" → `--days 14`
- 无特殊指定 → 默认当天

将解析出的天数存为 `DAYS` 变量。

### Step 1：抓取 + 打分（Python 脚本，零 token）

运行 `fetch_and_score.py`，输出到 `Workbench/daily/.candidates.json`：

```bash
python3 skills/1-literature/daily-papers/fetch_and_score.py \
  --days {DAYS} \
  --output Workbench/daily/.candidates.json
```

**检查输出**：确认文件存在且包含有效 JSON 数组。如果为空数组，检查 stderr 诊断问题（可能是周末 arXiv 无更新、网络问题等），告知用户原因后停止。

**历史去重**：脚本在 output 同目录维护 `.history.json`，单天模式自动过滤已总结过的论文（30 天窗口），多天模式跳过去重。每次运行后自动更新历史。

### Step 2：快速分流

读取 `Workbench/daily/.candidates.json`，做**轻量级分类**，不写详细点评。

#### 2a：兜底过滤

参照研究兴趣判断论文相关性，如果发现某篇论文与所有研究兴趣均无关，而且 score 不高，直接跳过。

#### 2b：分流

基于摘要和 score，将论文分流为：
- **要精读**：强相关 + 方法有新意或结果显著
- **可跳过**：其他论文（弱相关，limited novelty，marginal improvement 等）

每篇论文只需**一句话分流理由**，不写详细点评。

### Step 3：每篇要精读论文 → 笔记 + 单篇点评

**每篇要精读论文派发一个 subagent**，指示它：
1. 调用 `paper-digest` skill 生成笔记；若笔记已存在，则直接读取已有笔记
2. 基于笔记正文（而非摘要），依照点评原则生成点评，并按点评模版返回点评

**范围控制**：仅对"要精读" 论文执行，"可跳过"不派发 subagent。

**并行执行**：要精读论文全部并行派发（background agents）。等待所有返回后再进入下一步。

#### 点评模板

```markdown
### {短标题}
- **Title**: {完整标题}
- **Institutes**: {institutes}
- **Source**: [link]({url})  {来源徽章：📰 HF Daily ⬆️ N / 🔥 HF Trending ⬆️ N / 📄 arXiv}   **📒 论文笔记**: [[{笔记文件名}]]
- **核心**: 3-5 句，核心 idea + 主要结果，避免黑话
- **锐评**: 方法有没有硬伤？claim 和证据匹配吗？跟已有工作本质区别在哪？哪些数字亮眼、哪些
  暴露问题？
- **Rating**: `3` 🔥 / `2` 👀 / `1` 💤 ，{一句话理由}
```

#### 点评原则

- **点评人设**: 毒舌但眼光极准的 AI 论文审稿人，见多识广、对灌水零容忍的 senior researcher。
- **语气要求**：毒舌、尖锐、精炼、有态度。不和稀泥，不说"总体还行"。明确判断好/坏。
- **内容具体：** 夸要具体（哪个数字、哪个设计），骂要更具体（哪个假设不成立、哪个实验缺了、哪个 claim 站不住脚）
- **来源格式**：
  - `hf-daily` → `📰 HF Daily ⬆️ {hf_upvotes}`
  - `hf-trending` → `🔥 HF Trending ⬆️ {hf_upvotes}`
  - `arxiv` → `📄 arXiv`

> **重要‼️**：Subagent prompt 必须包含以上完整的`点评模板` 和 `点评原则`，**不要省略**

### Step 4：汇总点评 + 生成总结文件

汇总论文点评，生成总结文件，保存到 `Workbench/daily/YYYY-MM-DD.md`（日期为目标日期）。若文件已存在，覆盖写入。

**总结文件模版：**

```markdown
---
date: YYYY-MM-DD
tags: [daily-papers, tag1, tag2, ...]
---
# 🔪 今日总结

{2-3句总结}

## 评分表

| Rating | 论文 |
|--------|------|
| 🔥 `3 - Foundation` | [[#{短标题}]]（理由）· [[#{短标题}]]（理由） |
| 👀 `2 - Frontier` | [[#{短标题}]]（理由）· ... |
| 💤 `1 - Archived` | [[#{短标题}]]（理由） |

## 论文点评

{按评分表顺序，原样拼接 Step 3 返回的各单篇点评块}

## 已跳过论文

| 论文 | 跳过原因 |
|------|----------|
| ... | ... |
```

**Tag 选择**：阅读 vault 目录下的 `{vault_root}/references/tags.md`，按照规范选择 tag。

### Step 5：追加工作日志

将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

```markdown
### [HH:MM] daily-papers
- **input**: {DAYS} 天
- **output**: [[Workbench/daily/YYYY-MM-DD]]
- **observation**: 抓取 K 篇，精读 N 篇（rating 3: X / 2: Y / 1: Z），跳过 M 篇
```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

**告知用户**：抓取 K 篇，精读 N 篇（rating 3: X / 2: Y / 1: Z），跳过 M 篇

## Verify

- [ ] `Workbench/daily/.candidates.json` 存在且非空
- [ ] `Workbench/daily/YYYY-MM-DD.md` 已创建
- [ ] 评分表中的 `[[#{短标题}]]` 链接与论文点评的 `### {短标题}` 完全匹配
- [ ] 要精读论文笔记已生成，论文点评基于笔记内容
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`
