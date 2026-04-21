---
title: "Grounding Computer Use Agents on Human Demonstrations"
authors: [Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, Sai Rajeswar]
institutes: [Mila, McGill, Université de Montréal, ServiceNow Research, University of Waterloo, University of Oxford, NUS, Polytechnique Montréal, ÉTS]
date_publish: 2025-11-10
venue: arXiv 2025
tags: [gui-agent, computer-use, agentic-RL]
paper: https://arxiv.org/abs/2511.07332
website: https://groundcua.github.io/
github: https://github.com/ServiceNow/GroundCUA
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] Grounding Computer Use Agents on Human Demonstrations
> - **核心**: 用 expert human demonstration 构建 desktop grounding 数据集 GroundCUA（87 apps、56K screenshots、3.56M elements），证明高质量数据 + 700K SFT 即可达到 9M 数据训练的 SOTA。
> - **方法**: 真人录制 desktop 任务 → 提取 keyframe → 标注每个可见元素的 bbox + 文本标签 + 类别（50% 元素）→ MLLM 合成 Direct/Functional/Spatial 三类指令；Qwen2.5-VL-3B/7B 上 SFT (700K) + RLOO RL (10K) 训练 GroundNext。
> - **结果**: GroundNext-7B 在 5 个 grounding benchmark 平均 70.5（vs JEDI-7B 56.1）；GroundNext-3B + o3 planner 在 OSWorld-Verified 拿 50.6，与 7 倍大的 OpenCUA-72B（46.1）和 JEDI-7B（51.0）相当。
> - **Sources**: [paper](https://arxiv.org/abs/2511.07332) | [website](https://groundcua.github.io/) | [github](https://github.com/ServiceNow/GroundCUA)
> - **Rating**: 2 - Frontier（高质量 desktop grounding 数据集 + 当前 SOTA 开源模型，GUI grounding 方向的必比 baseline；尚未定型为 de facto 标准，故不到 Foundation）

**Key Takeaways:** 
1. **数据质量 >> 数据规模**：在 desktop grounding 上，700K 高质量 expert 标注样本可击败 9M+ 自动化构建样本。这是个对 GUI/CUA 数据策略有方向性影响的 negative result against scale-only 路线。
2. **Dense annotation 是关键**：每张图平均 64 个元素（3× OS-Atlas、6× UGround）、平均 element 仅占图 0.13%，覆盖到通常被自动化 pipeline 漏掉的 icon/toolbar/control，这才是 desktop 比 web/mobile 更难的核心点。
3. **RL 边际效益依赖 SFT 质量**：在 GroundCUA 上做完 SFT 后 RL 只涨 ~1-2 点，但用其它数据集 SFT 后 RL 涨幅更大。说明 RL 主要在 "修补 SFT 漏掉的 case"，不是独立的能力来源。
4. **Desktop → mobile/web 跨域泛化非对称**：仅在 desktop 训练，mobile 上接近 SOTA，web 上落后，提示 desktop 可能比 web/mobile 包含更多 transferable grounding skill。

**Teaser. GroundCUA 数据集与 GroundNext 训练 pipeline 的整体示意（FreeCAD "open the color picker" 任务为例，screenshot + UI metadata → instruction → SFT/RL 两阶段训练）。**

![](https://arxiv.org/html/2511.07332v1/x1.png)

---

## 1. Motivation

CUA 的主要 failure mode 不是 planning 而是 grounding——planner 决定 "click Save" 后，agent 必须在密集、视觉相似的 desktop 元素中定位到正确像素。论文给的 motivating example 是 FreeCAD 的 "open the color picker"，需要从一堆几乎一样的小调色板 icon 里选对那一个。

为什么 desktop 特别难（vs web/mobile）：
- **高分辨率 + 高密度**：单屏可有 542 元素，element 中位面积 0.13% 远小于 web 数据集
- **应用多样性**：办公、CAD、视频编辑、IDE 等差异巨大，每个 app 有自己的 icon convention
- **自动化标注 pipeline 不可靠**：accessibility tree 经常漏元素或标错；纯合成数据（如 JEDI）则失去真实复杂度

> ❓ 论文把 "grounding" 定为单步 click，但真实 CUA 经常涉及拖拽、组合手势——这部分被 task formulation 略过了。

## 2. GroundCUA Dataset

### 2.1 数据收集 pipeline

三步：(1) 选 87 个 open-source desktop app（覆盖 12 个 category，主要从 UI-Vision 来 + 4 个 finance/scientific 补充），(2) 让训练过的 annotator 设计并执行真实任务（共 10K+ task demonstrations），(3) 从 demonstration 抽取 "用户动作触发界面变化前一刻" 的 keyframe，对每个可见元素标 bbox + 文本标签，约 50% 元素额外标 8 类高层 category（Button/Menu/Sidebar/Input/Navigation/Visual/InfoDisplay/Other）。长文本（如代码段）用 PaddleOCR 抽 raw text。

> 关键设计选择：用真实任务驱动 vs OS-Atlas 用 BFS/DFS 随机点击。前者得到的 screenshot 分布更接近真实使用场景——比如 "正在编辑的文档" 而非 "全空界面"。

### 2.2 Instruction 构造

利用 dense annotation 提示 MLLM 合成三类 instruction：
- **Direct**: 描述元素属性、位置、上下文（"Click the magnifying-glass icon next to the search bar"）
- **Functional**: 描述意图而非元素（"Open a new tab" 而非 "Click '+' button"）
- **Spatial**: 用相对位置描述（"Click the element to the left of 'Files'"）

最终从 3.56M annotations 采样出 700K SFT instruction + 10K RL instruction（disjoint）。

### 2.3 与现有数据集对比

**Table 1. Grounding 数据集对比（节选）。GroundCUA 是唯一同时满足 human label + desktop + 高密度 + 高分辨率范围的数据集。**

| Dataset | Human | Desktop | Elements | Screenshots | Res Range (MP) | EleArea | #AvgE |
| ---- | :----: | :----: | ---- | ---- | ---- | ---- | ---- |
| UGround | ✗ | ✗ | 9M | 773k | 0.4–1.9 | — | 11.6 |
| JEDI | ✗ | ✓ | 4M | 575k | 0.9–2.1 | — | 7.0 |
| Aguvis-G | ✗ | ✗ | 3.8M | 452k | 0.5–2.1 | — | 8.5 |
| [[2410-OSAtlas\|OS-Atlas]] | ✗ | ✓ | 14.5M | 1.85M | 0.5–5.2 | 0.53% | 7.8 |
| **GroundCUA** | ✓ | ✓ | 3.56M | 55k | **0.4–7.0** | **0.13%** | **64.1** |

值得注意：GroundCUA 的总 element 数（3.56M）反而**小于** OS-Atlas 的 14.5M，但每张图密度（64 vs 7.8）和 element 平均占比（0.13% vs 0.53%）都说明它在 fine-grained 元素覆盖上质量更高。这正是论文的核心 thesis——expert dense labeling > 大规模浅标注。

## 3. GroundNext 模型

### 3.1 SFT 阶段

Base model: Qwen2.5-VL-Instruct（3B / 7B），同时 fine-tune vision encoder 和 LM。Global batch 128，单 8×H100 节点，700K 样本。

### 3.2 RL Post-training (RLOO)

用 Relative Leave-One-Out（RLOO）做 policy optimization，避免训 critic：

**Equation. RLOO gradient.**

$$
\nabla_{\theta}J(\pi_{\theta}) = \frac{1}{n}\sum_{i=1}^{n}\Big(R(y_i,x) - \frac{1}{n-1}\sum_{j\neq i}R(y_j,x)\Big)\nabla_{\theta}\log\pi_{\theta}(y_i|x)
$$

**符号**：$y_i$ 是预测坐标 token 序列，$x$ 是 (instruction, image)，$n=8$ rollouts/group。
**含义**：每个 rollout 的 advantage = 自己 reward − 同组其它 rollout 的平均 reward；group-relative baseline 等价于 GRPO 的简化版。

**Reward function.** 离散 6 档 reward，基于预测点到 GT bbox 的归一化距离 $\mathcal{D}_{norm} = \mathcal{D}(\hat{p}, B) / \mathcal{D}_{max}(B, I)$：

$$
R_{score}(\hat{p},B,I)=\begin{cases}
-1.0 & \mathcal{D}_{norm} < -0.5 \\
-0.5 & -0.5 \leq \mathcal{D}_{norm} < -0.1 \\
-0.1 & -0.1 \leq \mathcal{D}_{norm} < 0 \\
+0.1 & 0 \leq \mathcal{D}_{norm} < 0.1 \\
+0.5 & 0.1 \leq \mathcal{D}_{norm} < 0.5 \\
+1.0 & \mathcal{D}_{norm} \geq 0.5
\end{cases}
$$

> 设计直觉：bbox 内部 reward 仍鼓励向中心靠（细化 reward），bbox 外部按距离梯度惩罚。论文显式拒绝 reward model 路线，理由是当前 judge 不可靠。

## 4. 实验结果

### 4.1 SFT-only 对比

**Table 2. SFT-only 在 5 个 benchmark 上的结果（SSPro / OSW-G / MMB-GUI / SSv2 / UI-V，节选 7B 段）。**

| Model | SSPro | OSW-G | MMB-GUI | SSv2 | UI-V | Avg |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: |
| Qwen2.5-VL-7B (Agent mode) | 29.7 | 42.7 | 67.7 | 86.4 | 16.5 | 48.6 |
| OS-Atlas-7B | 18.9 | 27.7 | 41.4 | 85.1 | 9.0 | 36.4 |
| UGround-V1-7B | 16.5 | 36.4 | 65.7 | 87.6 | 12.9 | 43.8 |
| Aguvis-7B | 39.5 | 38.7 | 45.7 | 86.0 | 13.7 | 44.7 |
| GUI-Actor-7B | 44.6 | 47.0 | 70.9 | 92.1 | 21.9 | 55.3 |
| JEDI-7B | 39.5 | 54.1 | 70.4 | 91.7 | 24.8 | 56.1 |
| **GroundNext-7B (SFT)** | **50.2** | **67.2** | **80.4** | 89.3 | **58.7** | **69.2** |

GroundNext-7B SFT 比 JEDI-7B 高 +13.1 average，比 GUI-Actor-7B 高 +13.9。在 in-domain 的 UI-V 上 +33.9 是预期的，但即使排除 UI-V，仍领先 +7.9。

**Apples-to-apples 数据质量对比**：用同一 base + 100K 样本分别训 Aguvis/UGround/OS-Atlas/JEDI/GroundCUA，GroundCUA 显著领先所有其它来源（Figure 3）——这是排除 model size 后对数据质量本身的验证，是论文最 honest 的实验之一。

### 4.2 RL 后训练

**Table 3. RL-tuned 7B 段对比。**

| Model | SSPro | OSW-G | MMB-GUI | SSv2 | UI-V | Avg |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: |
| UI-TARS-1.5-7B | 49.6 | 64.2 | 64.3 | 90.3 | 20.8 | 57.8 |
| GUI-G²-7B | 47.5 | 61.9 | 79.5 | 93.3 | 25.6 | 61.7 |
| InfiGUI-G1-7B | 51.9 | 59.9 | 80.8 | 93.5 | 26.1 | 62.4 |
| GTA1-7B | 50.1 | 67.7 | 79.4 | 92.4 | 25.7 | 63.1 |
| **GroundNext-7B (SFT)** | 50.2 | 67.2 | 80.4 | 89.3 | 58.7 | 69.2 |
| **GroundNext-7B (RL)** | **52.9** | **67.7** | **81.1** | 90.4 | **60.3** | **70.5** |

RL 在 7B 上从 69.2 → 70.5（+1.3），3B 上从 66.4 → 68.4（+2.0）。

**Analyzing RL gains（Figure 3）**：从不同数据集 SFT 出发再用 GroundCUA 的 10K 做 RL，发现 SFT 用 GroundCUA 的模型 RL 增益最小，用其它数据集 SFT 的反而获益更大。结论："SFT with high-quality data captures most of the performance, RL is incremental refinement"。

> 这个 framing 我部分认同：它意味着如果你已经有好数据，不需要复杂 RL。但反过来也可解读为——GroundCUA 不够大/不够 verifiable 以让 RL 真正发力。论文的 reward 也是简单距离 reward，没用 [[2509-UITARS2|UI-TARS-2]] 这种带组合 reward 的方案。

### 4.3 Agentic 评估（OSWorld）

**Table 4. OSWorld-Verified 上的 agentic 表现（GroundNext-3B 用 o3 做 planner）。**

| Model | OS | Office | Daily | Pro | Workflow | Overall |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: |
| OpenAI o3 | 62.5 | 14.5 | 21.4 | 38.8 | 16.5 | 23.0 |
| Claude-4-Sonnet | 45.8 | 39.3 | 48.1 | 59.2 | 27.9 | 41.4 |
| UI-TARS-250705 | 41.7 | 50.4 | 55.7 | 51.0 | 14.7 | 41.8 |
| Claude-4.5-Sonnet | 70.8 | 72.6 | 61.4 | 63.3 | 49.0 | **62.9** |
| [[2508-OpenCUA\|OpenCUA-72B]] | 58.3 | 47.0 | 53.8 | 73.5 | 20.4 | 46.1 |
| JEDI-7B w/ o3 | 50.0 | 46.1 | **61.9** | **75.5** | 35.3 | **51.0** |
| **GroundNext-3B w/ o3** | **62.5** | **47.0** | 55.0 | 73.5 | **36.5** | 50.6 |

3B 模型与 7B 的 JEDI、72B 的 OpenCUA 相当，且超过 Claude-4-Sonnet。但仍远低于 Claude-4.5-Sonnet（62.9）——闭源前沿模型在 agentic 任务上仍领先 ~12 点。

### 4.4 跨平台泛化

只在 desktop 训练，但在 mobile/web 也表现 OK：MMBench-GUI 7B (RL) 在 mobile 89.2%、web 81.9%（vs InfiGUI-G1-7B 的 90.9% / 85.3%）；SSv2 mobile 接近 SOTA、web 落后。Desktop training 对 icon recognition 提升尤大（SSPro icon 类别 +10.7% 平均）——open-source app 多样性带来的 icon 知识被泛化吸收了。

## 关联工作

### 基于
- [[2410-OSAtlas|OS-Atlas]]: 主要 baseline，desktop grounding 的前 SOTA 数据策略代表（accessibility tree + BFS/DFS 自动化）
- JEDI: 用 9M 合成 desktop 数据训练 7B 模型，论文的核心对比对象——证明合成 ≠ real
- Qwen2.5-VL: base model

### 对比
- [[2508-OpenCUA|OpenCUA]]: 7B/72B 对照，验证 GroundNext-3B 用更小模型达到可比性能
- [[2501-UITARS|UI-TARS]] / UI-TARS-1.5: agentic CUA 的强基线
- GUI-Actor / InfiGUI-G1 / GUI-G² / GTA1: RL-tuned 同期工作，主要做复杂 reward 设计

### 方法相关
- RLOO: 来自 Ahmadian et al., 用作 critic-free policy optimization；与 GRPO 思路接近
- [[2401-SeeClick|SeeClick]]: web grounding 的早期代表
- [[2312-CogAgent|CogAgent]] / [[2411-ShowUI|ShowUI]]: 早期 GUI agent，代表 "vision+language+action" 的范式起点
- [[2404-OSWorld|OSWorld]]: agentic 评估 benchmark
- [[2504-ScreenSpotPro|ScreenSpot-Pro]]: 主要 desktop benchmark，用于 SSPro 和 icon-level 分析

---

## 论文点评

### Strengths

1. **方向判断正确**：押注 "高质量 expert annotation > 自动化大规模"，并用 100K samples 同条件对比直接证伪 scale-only 路线。这是 GUI grounding 数据策略的一个 important rather than publishable insight。
2. **数据集 dense 程度真的领先**：64 elements/screen、0.13% avg area 是别的数据集做不到的，且这正是 desktop CUA 的核心瓶颈（小 icon、密集 toolbar）。
3. **完整的 ablation**：100K 同条件数据对比、RL gain decomposition、cross-platform breakdown——比大多数同类工作 honest 得多。
4. **3B 模型实用**：GroundNext-3B + o3 在 OSWorld 接近 OpenCUA-72B，对 resource-constrained 部署很有意义。
5. **完全开源**：dataset + 模型权重 + 代码全公开，permissive license。

### Weaknesses

1. **"700K vs 9M" 的对比有点 unfair**：JEDI 等基线没有在同 700K 体量下重新训练；论文的 100K head-to-head 才是真正干净的对比，但只跑了 3B + 10 个 benchmark 的 average，没有 full table。
2. **Grounding ≠ CUA**：论文标题用 "Computer Use Agents"，但核心贡献仅在 single-step grounding。Agentic 评估里 planner 全是 o3，没 ablate "如果换 Claude/Qwen3 planner，gain 是否还在"。
3. **RL 收益弱且解释 self-serving**：作者把 RL 增益小解释为 "SFT 已经很好"，但也可能是 reward function 太简单（仅距离）+ 10K 数据太小。没和 GUI-G²/InfiGUI-G1 的复杂 reward 在同 SFT 起点上比 RL 算法本身。
4. **UI-V 是 in-domain**：论文承认 UI-V 与训练数据有 platform overlap，所以 +33.9 的提升不能算泛化能力。Average 数应该报告 w/ 和 w/o UI-V 两版（论文部分地这么做了）。
5. **没有 fail case 分析**：appendix E 是 "error analysis" 但正文没体现失败模式（特别是 web 上落后的具体原因）。

### 可信评估

#### Artifact 可获取性
- **代码**: training + inference 全开源（github.com/ServiceNow/GroundCUA）
- **模型权重**: GroundNext-3B 和 GroundNext-7B 两个 checkpoint，HuggingFace 上发布（README 链接）
- **训练细节**: 完整披露（hyperparams 在 Appendix C.2 / C.4，数据配比和 SFT subset 构建在 C.1，RLOO group size、batch、epoch 都给了）
- **数据集**: 完整开源（HuggingFace `ServiceNow/GroundCUA`，permissive license）

#### Claim 可验证性
- ✅ "700K SFT 击败 9M JEDI"：5 个 benchmark + 100K 同条件 head-to-head 数据均可独立复现
- ✅ "GroundNext-3B 接近 JEDI-7B / OpenCUA-72B 在 OSWorld"：50.6 vs 51.0 vs 46.1 数字明确，benchmark setup 详细（361 tasks、Azure、10 Docker）
- ⚠️ "RL post-training 提供 incremental refinement"：增益仅 1-2 点，且未排除 RL 算法本身设计不足的可能。"high-quality SFT captures most of performance" 是正确归因还是事后合理化，需要更强对照（同 reward 不同 SFT，或同 SFT 不同 reward 的 2×2 ablation）才能区分
- ⚠️ "Cross-platform generalization"：SSv2 web 落后、MMBench-GUI web 也落后，"strong generalization" 的描述对 web 来说不成立，论文措辞偏宽
- ⚠️ "Expert annotation 是 quality 的来源"：annotator 训练流程在 A.2，但 inter-annotator agreement、QA pipeline、错误率等量化质量指标缺失
- ❌ 无明显营销性 claim

### Notes

- **对我的研究方向**: 对 GUI agent / CUA 方向是 important data resource，对 VLA / world model / spatial reasoning 没直接关联。Rating 2（最新 SOTA in 我直接关心的 grounding 子方向，但不改变我对 CUA 整体方向的判断）。
- **可拿走的 transferable insight**: "Dense expert annotation 在 fine-grained perception 任务上 vs 大规模浅标注" 这个权衡可能也适用于 manipulation 的 affordance grounding 或 VLA 的 keypoint annotation——是否值得花 expert 时间标 dense 而非便宜地标 sparse？这是一个跨 domain 的开放问题。
- **可借鉴方法**: RLOO + 6 档离散距离 reward 的简单设计，对未来 grounding-style 任务（包括 spatial reasoning）的 RL 后训练可能是个好起点，比复杂 reward 容易 debug。
- **疑问**: 
  - ❓ Annotator 用了多少人时？10K demonstration × 64 element/frame × multiple frames = 数千万级 annotation event。expert quality 的成本是否 justify？这个数据集是 ServiceNow 出钱才可能的，对学术界 reproducibility 有 implication。
  - ❓ 如果把 GroundCUA 加到现有大数据集（如 OS-Atlas 的 14.5M）做 mix training，是否 +1+1>2？论文没做这个 obvious experiment，可能因为 ablation 成本太高，但缺这一项就难说 "expert data 是必需" vs "expert data 是 useful complement"。

### Rating

**分数**：2 - Frontier
**理由**：GroundCUA 是当前开源 desktop grounding 数据集中密度与 expert 质量最高的（64 elements/screen、0.13% avg area），GroundNext-7B (RL) 在 5 个 benchmark 上平均 70.5 超越 JEDI/UI-TARS-1.5/GTA1 等同期工作，已构成 GUI grounding 方向必比的 SOTA baseline；但发布仅 2025-11，尚未被后续主要工作普遍采纳为 de facto 标准，且核心贡献仍在数据策略而非范式开创，故不到 Foundation 而低于相邻的 3 档。
