---
title: "CogAgent: A Visual Language Model for GUI Agents"
authors: [Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, Jie Tang]
institutes: [Tsinghua University, Zhipu AI]
date_publish: 2023-12-14
venue: CVPR 2024
tags: [gui-agent, VLM, computer-use]
paper: https://arxiv.org/abs/2312.08914
website: 
github: https://github.com/THUDM/CogVLM
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] CogAgent: A Visual Language Model for GUI Agents
> - **核心**: 一个 18B 的 VLM，用 dual-resolution（224 + 1120）image encoder 在 screenshot-only 输入下做 GUI 理解和导航，证明纯视觉 agent 可以超过吃 HTML 文本的 LLM agent
> - **方法**: 在 CogVLM-17B 基础上加一个 0.30B 的 high-resolution cross-module（EVA2-CLIP-L），通过 cross-attention 把 1120×1120 的 features 注入 decoder 每一层；构造 CCS400K 数据集做 GUI grounding 预训练
> - **结果**: Mind2Web 上 step SR 58.2%（超过 LLaMA2-70B+HTML 11.6%），AITW 76.88%；同时在 9 个 VQA benchmark 上拿到 generalist SOTA（DocVQA 81.6, TextVQA 76.1）
> - **Sources**: [paper](https://arxiv.org/abs/2312.08914) | [github](https://github.com/THUDM/CogVLM)
> - **Rating**: 3 - Foundation（pure-vision GUI agent 路线的奠基工作，CCS400K pipeline 与 screenshot-only paradigm 被 SeeClick / OS-Atlas / ShowUI / UI-TARS 等主流后续工作显式继承或作为 baseline）

**Key Takeaways:** 
1. **Resolution 是 GUI agent 的瓶颈**: 224×224 看不清 icon 和小字，但直接放大到 1120 在 self-attention 里是 6400 tokens（quadratic 爆炸）。CogAgent 的解决方案是 asymmetric design——保留原 ViT 处理低分辨率主干，新增小 hidden size 的 cross-attention 分支处理高分辨率细节
2. **Text-related features 不需要大 hidden size**: 论文核心 insight 之一，cross-module 的 hidden size 仅 1024（vs decoder 的 4096），却能高效捕获 OCR / GUI 元素信息。这个观察直接 motivate 了 architecture choice
3. **Pure-vision GUI agent 可行**: Mind2Web 上 screenshot-only 的 CogAgent 超过用 cleansed HTML 的 LLaMA2-70B（差不多 4× 参数量）。当时 first time 视觉 agent 在结构化文本 baseline 上反超
4. **CCS400K 数据集**: 从 Common Crawl 抓 400k 网页截图，用 Playwright 渲染拿到 DOM 元素和 bbox，构造 140M REC/REG QA 对。这种 web-scale 的 GUI grounding 数据是后续 GUI agent 的重要 building block
5. **限制**: 单图输入（不支持 multi-image history），坐标输出精度有限，没有 RL / online 反馈环节

**Teaser. CogAgent 在多种 GUI 场景下的 demo（PPT、地图、网页、IDE 等）。**

![](https://arxiv.org/html/2312.08914v3/extracted/6097505/figures/main_demo.jpg)

---

## Method

### 整体架构

CogAgent 基于 CogVLM-17B，在原有 low-resolution 分支（EVA2-CLIP-E，224×224，4.4B 参数）之上新增一条 high-resolution cross-module 分支（EVA2-CLIP-L，1120×1120，0.30B 参数）。两个分支并行编码同一张图，然后 high-res features 通过 cross-attention 注入 decoder 每一层。

**Figure 2. CogAgent 模型架构。右侧是原 CogVLM 的 low-resolution 主干（visual expert + 文本 token 自回归），左侧是新加的 high-resolution cross-module 分支。**

![](https://arxiv.org/html/2312.08914v3/x1.png)

### High-Resolution Cross-Module 设计动机

两条核心观察：
1. 224×224 已经能描述 object 和 layout，但**渲染不清文字**——而 GUI 的本质是 text-rich
2. 通用 VLM 用大 hidden size（PALI-X / CogVLM 4096，LLaVA 5120），但 OCR-centric 的 VLM 用小 hidden size 就够（Kosmos-2.5 / Pix2Struct 1536）→ **text features 可以用小维度承载**

这两点直接推出 architecture：用一个小的 vision encoder 加小 hidden size 的 cross-attention，把高分辨率信息作为 low-res 的"补丁"注入。

### 计算复杂度对比

设 $L_{I_{\text{lo}}}$、$L_{I_{\text{hi}}}$、$L_T$ 为低分图、高分图、文本序列长度。原始做法（直接把 low-res 替换为 high-res）的 attention 复杂度：

$$
\text{T}_{\text{original}} = \mathbf{O}\bigl((L_{I_{\text{hi}}} + L_T)^2 H_{\text{dec}} d_{\text{dec}}\bigr)
$$

CogAgent 的 cross-module 方案：

$$
\text{T}_{\text{improved}} = \mathbf{O}\bigl((L_{I_{\text{lo}}} + L_T) L_{I_{\text{hi}}} H_{\text{cross}} d_{\text{cross}} + (L_{I_{\text{lo}}} + L_T)^2 H_{\text{dec}} d_{\text{dec}}\bigr)
$$

实现配置：$d_{\text{cross}}=32$、$H_{\text{cross}}=32$、$d_{\text{dec}}=128$、$H_{\text{dec}}=32$（继承自 CogVLM-17B），patch size 14×14 → $L_{I_{\text{hi}}}=6400$、$L_{I_{\text{lo}}}=256$。理论加速下界 $\frac{6400+L_T}{256+L_T} \times$。

> ❓ 这个 derivation 假设 cross-attention 的 KV 长度（6400）和 self-attention 的 query 长度（256+L_T）解耦，所以 cross-attention 部分是线性而非平方——本质上是把 high-res 当成 "external memory" 而不是 sequence 的一部分。这种 "看图但不进 sequence" 的思路在后来的 PaliGemma / SigLIP 路线里没占主流，主流是直接 tokenize 进 sequence + 各种 token compression。值得追问：1120 输入下二者的 quality 真的差不多吗？

### Architecture 配置

| 模块 | 配置 |
|---|---|
| VLM decoder | Vicuna-1.5-7B + visual expert，32 layers，hidden 4096，32 heads |
| Low-res visual encoder | EVA2-CLIP-E，224×224 输入，patch 14×14 |
| High-res visual encoder | EVA2-CLIP-L，1120×1120 输入，patch 14×14 |
| Cross-attention | hidden 1024，32 heads |

整体 18B 参数（17B CogVLM + 0.30B 高分编码器 + cross-attention 矩阵）。

### Pre-training 数据

三类，全部 publicly available：

1. **Text recognition**：合成渲染（80M，从 LAION-2B 取背景，随机 font/size/orientation）+ 自然图 OCR（18M，COYO + LAION-2B，Paddle-OCR 抽 bbox）+ 学术文档（9M，arXiv LaTeX 源码 → image-text，遵循 Nougat 流程）
2. **Visual grounding**：40M 图，沿用 CogVLM 从 LAION-115M 构造，bbox 格式 $[[x_0, y_0, x_1, y_1]]$，归一化到 $[000, 999]$
3. **GUI imagery**：自建 **CCS400K**（Common Crawl Screenshot 400K）——从 Common Crawl 抓 URL，Playwright 渲染 400k 网页截图，配合可见 DOM 元素 + 渲染 bbox，生成 **140M** REC（Referring Expression Comprehension，给 DOM 元素出 bbox）和 REG（Referring Expression Generation，给 bbox 出 HTML）QA 对。多种屏幕分辨率随机采样防 overfit；DOM 属性精简防 token 爆炸

### Training Schedule

- **Pre-training**: 60k steps, batch 4608, lr 2e-5。前 20k steps 只解冻 cross-module（646M trainable，3.5%），后 40k steps 额外解冻 visual expert
- **Curriculum**: 先 easy text recognition + caption → 加 academic doc → 加 grounding → 加 web page，作者说收敛更快更稳
- **Multi-task fine-tuning**: 10k steps, batch 1024, lr 2e-5, 全参数解冻。数据 = 自标 2k 截图 QA + Mind2Web/AITW（用 GPT-4 转成 QA 格式）+ 公开 VQA

---

## Experiments

### VQA Benchmarks

**Table 1. VQA benchmark 结果（generalist setting）。CogAgent 在 5/6 文本密集 VQA 上 SOTA，在 TextVQA / STVQA / DocVQA 上甚至超越 task-specific 模型。**

| Method | VQAv2 | OKVQA | OCRVQA | TextVQA | STVQA | ChartQA | InfoVQA | DocVQA |
|---|---|---|---|---|---|---|---|---|
| PALI-X-55B (task-specific) | 86.0 | 66.1 | 75.0 | 71.4 | 79.9 | 70.9 | 49.2 | 80.0 |
| Qwen-VL (generalist) | 79.5 | 58.6 | 75.7 | 63.8 | - | 65.7 | - | 65.1 |
| LLaVA-1.5 (generalist) | 80.0 | - | - | 61.5 | - | - | - | - |
| CogVLM (generalist) | 83.4 | 58.9 | 74.1 | 68.1 | - | - | - | - |
| **CogAgent (Ours)** | **83.7** | **61.2** | **75.0** | **76.1** | **80.5** | **68.4** | 44.5 | **81.6** |

generalist 类别下 CogAgent 在 TextVQA +8.0、DocVQA +16.2 拉开很大差距，证明 high-res + OCR 数据的组合对 text-rich 任务很有效。InfoVQA 是唯一未拿 SOTA（infographic 的 layout 复杂度可能超过模型 grounding 能力）。

**Table 2. MM-Vet 和 POPE-adversarial 上的零样本评测。**

| Method | LLM | MM-Vet | POPE_adv |
|---|---|---|---|
| LLaVA-1.5 | Vicuna-13B | 36.3 | 84.5 |
| Emu | LLaMA-13B | 36.3 | - |
| **CogAgent** | Vicuna-7B | **52.8** | **85.9** |

MM-Vet +16.5 的提升非常显著，作者把 7B 的 base 干过 13B baseline，说明 high-res + OCR 对 conversational complex reasoning 也有帮助。

### Mind2Web（PC 端 GUI Agent）

**Table 3. Mind2Web 上的 step success rate。CogAgent 是唯一只用 screenshot 的 SOTA 方法。**

| Method | cross-task | cross-website | cross-domain | overall |
|---|---|---|---|---|
| **HTML 输入** | | | | |
| GPT-4 (few-shot) | 36.2 | 30.1 | 26.4 | 30.9 |
| LLaMA2-7B | 52.7 | 47.1 | 50.3 | 50.1 |
| LLaMA2-70B | 55.8 | 51.6 | 55.7 | 54.4 |
| **图像输入** | | | | |
| Qwen-VL | 12.6 | 10.1 | 8.0 | 10.2 |
| CogVLM | 37.1 | 23.4 | 26.3 | 23.9 |
| **CogAgent** | **62.3** | **54.0** | **59.4** | **58.2** |

核心 finding：仅用 screenshot 的 CogAgent（18B）在三个 OOD subset 上分别比 LLaMA2-70B + 清洗 HTML 高 11.6 / 4.7 / 6.6 个点。同时也吊打 CogVLM baseline（+34.3 overall），说明 cross-module + GUI 数据有显著贡献。

> ❓ Top-50 candidate 的设置实际上是把 element selection 退化为 50-way classification——这意味着 "找元素" 的难度被外部 candidate generator 帮 CogAgent 兜了一道。在没有 candidate 的纯 grounding 场景（比如真实 web automation）性能可能要打不少折扣。

### AITW（Android）

**Table 4. AITW 上的 matching score。CogAgent 是 unified 模型（一套权重跑全部 subset）。**

| Method | GoogleApp | Install | WebShop | General | Single | Overall |
|---|---|---|---|---|---|---|
| GPT-3.5 (OCR+icon) | 10.47 | 4.38 | 8.42 | 5.93 | 9.39 | 7.72 |
| LLaMA2-7B (per-subset FT) | 30.99 | 35.18 | 19.92 | 28.56 | 27.35 | 28.40 |
| Auto-UI (image, unified) | 71.37 | 76.89 | 70.26 | 68.24 | 84.58 | 74.27 |
| **CogAgent (image, unified)** | **74.95** | **78.86** | **71.73** | 65.38 | **93.49** | **76.88** |

vs Auto-UI +2.61 overall。General subset 反而略低（-2.86）—— 作者抽样发现 40%+ 的 "incorrect" 实际上是合法替代路径，AITW 的 single-ground-truth 评测太严苛。

### Ablation: Architecture

**Table 5. 架构消融——直接增大 base resolution vs 用 cross-module。**

| cross-module | base res | cross res | STVQA | OCRVQA | DocVQA | Mind2Web | TFLOPs |
|---|---|---|---|---|---|---|---|
| ✗ | 224 | — | 48.0 | 70.2 | 28.6 | 34.6 | 7.77 |
| ✗ | 490 | — | 68.1 | 74.5 | 57.6 | 40.7 | 29.14 |
| ✓ | 224 | 756 | 73.6 | 74.2 | 62.3 | 40.7 | **10.08** |
| ✓ | 224 | **1120** | **78.2** | **75.9** | **74.1** | **41.4** | 12.56 |

最后一行 vs 第二行：cross res 1120 比 base res 490 在 DocVQA 上 +16.5，FLOPs 反而降了一半多（12.56 vs 29.14）。

**Figure 3. 不同分辨率下 FLOPs 对比。原 CogVLM 架构在高分辨率下 FLOPs 几乎是 cross-module 的 10 倍以上。**

![](https://arxiv.org/html/2312.08914v3/x2.png)

### Ablation: Pre-train Data

**Table 6. 数据消融，依次叠加 caption、OCR、GUI+grounding。**

| Pre-train data | base res | cross res | STVQA | OCRVQA | DocVQA | Mind2Web |
|---|---|---|---|---|---|---|
| Cap | 490 | — | 68.1 | 74.5 | 57.6 | 38.6 |
| Cap+OCR | 490 | — | 72.5 | 75.0 | 59.8 | 40.7 |
| Cap+OCR | 224 | 1120 | 78.2 | 75.9 | 74.1 | 41.4 |
| All | 224 | 1120 | 79.4 | 75.6 | 76.4 | **54.2** |

Mind2Web 从 41.4 → 54.2（+12.8）几乎完全来自 GUI / grounding 数据，证明 web 截图预训练对下游 web agent 是 critical。VQA 任务的提升则均匀分布在 OCR 和 GUI 数据上。

---

## 关联工作

### 基于
- **CogVLM** (Wang et al. 2023): 提供 17B base VLM 和 visual expert 设计；CogAgent 直接 freeze 它然后加 cross-module
- **EVA2-CLIP** (Sun et al.): 提供 E（4.4B）和 L（0.30B）两个 visual encoder
- **Vicuna-1.5-7B**: decoder 的语言模型基座

### 同期 / 对比
- **Qwen-VL** (Bai et al. 2023): 用 position-aware adapter 压缩 image features，最高 448×448；CogAgent 在 Mind2Web 上 +48 个点超过它
- **Auto-UI** (Zhan et al. 2023): AITW 上的视觉 agent SOTA baseline，被 CogAgent 超过 +2.61
- **Kosmos-2.5**: 用 Perceiver Resampler 处理 OCR / document，但仍 restricted；CogAgent 借鉴了 "text features 用小 hidden size" 的观察
- **Fuyu-8B**: 同期试图原生处理任意分辨率的方案

### 数据 / 任务
- [[2411-GUIAgentSurvey|GUI Agent Survey]]: CogAgent 是这篇 survey 里 vision-only agent 的代表性早期工作
- **Mind2Web** (Deng et al. 2023): 137 网站 / 31 domain / 2k+ tasks 的 web agent benchmark
- **AITW** (Rawles et al. 2023): 715k Android episode 的 mobile agent benchmark

### 后续影响
- [[2401-SeeClick|SeeClick]]: 同方向的 vision-only GUI agent，更聚焦 element grounding；明确把 CogAgent 当 baseline
- [[2408-OmniParser|OmniParser]]: 走另一条路——先 parse 出 structured screen representation 再喂 LLM，可以看作对 CogAgent 端到端方案的反向探索
- [[2410-OSAtlas|OS-Atlas]]: 大规模 GUI grounding 模型，CCS400K 的精神继承者
- [[2411-ShowUI|ShowUI]]: Visual GUI agent 的轻量化版本

---

## 论文点评

### Strengths

1. **架构 insight 清晰且有 first-principle 推理**：把 "high-res 用于 text" 和 "text features 不需要大 hidden size" 两个观察连起来，得到 asymmetric dual-encoder 设计。这种 "用约束推出架构" 的方式比纯 ablation-driven 调参更有说服力
2. **Compute 优势是真实的**：cross-module 1120 的 FLOPs（12.56T）比纯 ViT 490（29.14T）还低一半，效果反而更好。这是 architectural Pareto improvement 而不是 scaling 替代
3. **CCS400K 数据集是 contribution**：Playwright 渲染 + DOM bbox 配对的 pipeline 后续被很多 GUI agent 工作复用（OS-Atlas、ShowUI 等）
4. **Pure-vision agent 击败 HTML+LLM 的 milestone 价值**：在 2023.12 这是有节点意义的结果——它证明视觉 GUI agent 不只是 nice-to-have，而是 effective alternative。改变了后续 community 对 modality choice 的判断
5. **诚实评估 AITW failure**：作者主动指出 40%+ 的错误是合法替代路径，没有掩盖 ground-truth 评测的局限

### Weaknesses

1. **Cross-module 思路在后续主流路线中被边缘化**：现在的主流（Qwen2-VL / Pixtral / InternVL）都走 native dynamic resolution + token compression，而不是 dual-encoder + cross-attention。CogAgent 这条路的 generality 没有被广泛验证
2. **单图输入是硬伤**：Conclusion 里也承认了 "incapability of processing multiple images"。GUI agent 本质需要 history 对比（前一帧 vs 当前帧才能判断动作是否成功），单图严重限制了 long-horizon 任务
3. **没有 online / RL feedback**：完全 supervised，依赖 Mind2Web/AITW 的标注序列。这意味着 agent 不能从执行错误中纠正，泛化到 unseen workflow 时容易 cascade error
4. **Mind2Web 的 top-K candidate setting 有水分**：candidate generator 提前帮你 prune 到 50 个，纯 grounding 难度被显著降低。真实场景下 element selection 是 from scratch
5. **Coordinate 精度受 [000,999] 离散化限制**：bbox 用三位数字表示，screenshot 上 1px 的精度都拿不到。后续 SeeClick / OS-Atlas 转向连续坐标或更精细的 grounding head

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training（github.com/THUDM/CogVLM 提供完整 SAT 训练框架）；后续 zai-org/CogAgent 仓库还放了 9B 升级版 CogAgent-9B-20241220
- **模型权重**: CogAgent-18B（HuggingFace `THUDM/CogAgent`），后续有 CogAgent-9B-20241220 升级版
- **训练细节**: 完整披露——pre-train 60k steps / batch 4608 / lr 2e-5，multi-task FT 10k steps / batch 1024，curriculum 顺序也写明了；数据来源都是公开数据集
- **数据集**: CCS400K 论文里描述了构造方法（Common Crawl + Playwright），但**未直接发布数据**（只能复现 pipeline）；其他都是公开数据集（LAION/COYO/Mind2Web/AITW）

#### Claim 可验证性
- ✅ **Mind2Web overall 58.2% 超过 LLaMA2-70B+HTML 的 54.4%**：Tab. 3 数据齐全，对方的 baseline 也是作者自己用同样 cleansing 过程 fine-tune 的，公平对比
- ✅ **High-res cross-module 在 1120 分辨率下 FLOPs 比直接用 ViT 490 少且效果更好**：Tab. 5 + Fig. 3 ablation 完整
- ✅ **9 个 VQA benchmark 上 generalist SOTA**：Tab. 1+2 数据明确，可验证
- ⚠️ **"Outperforms LLM-based methods that consume HTML"**：仅在 Mind2Web 一个 benchmark 上验证，且对方用的 cleansed HTML 已经丢失部分信息（dynamic content / canvas / iframe）；说成 "advancing SOTA on this benchmark" 更准确
- ⚠️ **AITW General subset 上输给 Auto-UI（65.38 vs 68.24）**：作者用 "40% 的错误其实是合法替代" 解释，但没给 systematic 重评；结论方向正确但量化不严
- ⚠️ **CCS400K "ensures comprehensive training and understanding of GUI elements"**：dataset 描述详细但未发布；complementary REG 任务（出 HTML code）的实际下游收益没单独 ablate

### Notes

CogAgent 在 GUI agent 历史上是 vision-only 路线 viable 的关键证据。它的具体架构（dual-encoder + cross-attention）后来没成为主流，主流走了 native dynamic resolution（Qwen2-VL 路线），但它把 "high-res 处理可以 architecturally 解耦" 这个 idea 留下来了——后续 LLaVA-NeXT 的 anyres、PaliGemma 的 high-res 路线都能看到类似的 spirit。

对我研究方向的相关性：
- **GUI agent / computer-use**: 直接相关，是 building block 之一（rating 3 的核心理由）。理解 CogAgent 的 design choice 对评估后续 GUI agent 论文（OS-Atlas、ShowUI、UI-TARS 系列）的 architectural 贡献是必要 baseline
- **VLM**: high-resolution handling 的早期 architectural 探索，对理解 vision encoder 设计 trade-off 有参考价值

值得追问的问题：
- 如果今天重做，dual-encoder + cross-attention 的方案在 native dynamic resolution 已经 mature 的情况下还有 niche 吗？也许在 long-screenshot / 文档理解（输入 token 极长）场景下还有 compute advantage
- CCS400K 的 REG 任务（给 bbox 出 HTML）后续被验证是否有用？还是 REC（grounding）才是主要贡献？论文没单独 ablate
- CogAgent 的 grounding 用 [000,999] 离散坐标，这种 tokenization 后来被很多论文沿用——但在小 element / 高分辨率屏幕上误差实际多大？需要查一下 SeeClick 的 detailed comparison

### Rating

**分数**：3 - Foundation
**理由**：CogAgent 是 pure-vision GUI agent 路线的奠基工作——Strengths 里已点明它首次在 Mind2Web 上让 screenshot-only 方案反超 HTML+LLaMA2-70B，改变了社区对 modality choice 的判断；CCS400K 的 Playwright + DOM bbox pipeline 也被 SeeClick / OS-Atlas / ShowUI 等主流后续工作显式继承。相比 2 档（Frontier/SOTA），它已经过了"待验证的前沿"阶段，进入 GUI agent 方向的必读 baseline 序列；虽然具体 dual-encoder 架构被 native dynamic resolution（Qwen2-VL 路线）边缘化（见 Weaknesses 1），但其 paradigm-level 贡献（vision-only 可行、web-scale GUI grounding 数据范式）仍是该方向主脉络的 building block。
