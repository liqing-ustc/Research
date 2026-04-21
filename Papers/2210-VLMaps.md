---
title: Visual Language Maps for Robot Navigation
authors: [Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard]
institutes: [University of Freiburg, Google Research, University of Technology Nuremberg]
date_publish: 2022-10-11
venue: ICRA 2023
tags: [semantic-map, navigation, scene-understanding]
paper: https://arxiv.org/abs/2210.05714
website: https://vlmaps.github.io/
github: https://github.com/vlmaps/vlmaps
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] Visual Language Maps for Robot Navigation
> - **核心**: 把 LSeg 的 dense pixel-level visual-language features 反投影到 3D 重建并平均到 top-down grid map，得到一张可被自然语言索引的开放词汇语义地图
> - **方法**: RGB-D + 已知位姿 → back-project LSeg embeddings 到 grid → 用 CLIP text encoder 做 cosine 相似度索引 landmark；配合 LLM 生成 Python 调用 navigation primitives
> - **结果**: 在 Habitat/Matterport3D 多目标导航 SR 59% (vs CoW 36%, LM-Nav 26%)，在 21 条空间目标语言任务上 SR 62% vs CoW 33%；同一张地图按障碍类目列表为 LoCoBot/drone 分别生成 obstacle map
> - **Sources**: [paper](https://arxiv.org/abs/2210.05714) | [website](https://vlmaps.github.io/) | [github](https://github.com/vlmaps/vlmaps)
> - **Rating**: 3 - Foundation（spatial-grounded VLM representation 路线的早期奠基工作，方法简单可复现，related-work 必引，且 cross-embodiment 等 use case 是 representation choice 的独特产物）

**Key Takeaways:**
1. **Spatial-grounded VLM features**: 与 LM-Nav / CoW 把 VLM 当作 image-goal critic 不同，VLMaps 把 VLM features 直接绑在 3D 几何上——这才是支持 "in between"、"to the right of" 这类空间引用的关键，几何与开放词汇语义不再脱节
2. **Cross-embodiment via category list**: 单张 VLMap + 一个 embodiment-specific 的 obstacle category 列表 → 不同形态机器人的 obstacle map（drone 可飞过 table，LoCoBot 不行）。相同 SR 下 drone 的 SPL 提升
3. **Code-as-policy 用于 spatial 子目标**: 用 LLM 把指令解析成 Python 程序，调用 `move_to_left`/`move_in_between` 等 17 个 primitives；这些 primitives 内部去查 VLMap 拿坐标。绕开了 "语言→affordance value" 那条链路
4. **关键边界**: 平均特征会丢辨识力（"sofa" vs "chair" 同材质时混淆）、对 odometry 漂移敏感、LSeg 训练词表外的类（如 "seating"）embedding 会塌缩到训练过的相近类——VLM 的训练分布是硬上限

**Teaser. 给出一条复合空间指令，机器人解析为子目标序列并在 top-down map 上规划。**

![](https://ar5iv.labs.arxiv.org/html/2210.05714/assets/x1.png)

<video src="https://vlmaps.github.io/static/images/VLMaps_v24_lt.mp4" controls muted playsinline width="720"></video>

---

## 背景与动机

经典 SLAM/navigation 给出几何精度但词表封闭；近期 zero-shot 方法（[[2204-SayCan|SayCan]] 风格的 LLM planner、LM-Nav、CoW）借助 VLM 把语言落到图像上，但用法是 "VLM as critic" —— 拿 image observation 与 object goal description 做匹配，**与建图过程脱节**。后果是：

- 同一物体的多视角观测无法互相关联
- 无法定位 "in between the sofa and the TV" 这种空间引用
- 持久化表征难以在多 embodiment 间共享

VLMaps 的判断是：要让 "open-vocabulary" 与 "spatial precision" 同时成立，就要把 VLM features 与 3D geometry 绑定，而不是只用 VLM 输出 score 来选 image。

> ❓ 这其实是个 "表征 vs 检索" 的取舍——VLMaps 选的是 representation 路线，代价是地图要预先建好；CoW/LM-Nav 是 query-time retrieval，更轻但失去空间结构。两者并不互斥，是否可以 hybrid？

## 方法

### Pipeline 总览

**Figure. System overview —— RGB-D 视频 + 视觉 odometry → 反投影 LSeg pixel embeddings 到 top-down grid → 用 CLIP text encoder 索引 landmark / 生成 obstacle mask → LLM 生成 Python 调用 navigation primitives.**

![](https://ar5iv.labs.arxiv.org/html/2210.05714/assets/x3.png)

### A. Building the VLMap

VLMap 定义为 $\mathcal{M}\in\mathbb{R}^{\bar{H}\times\bar{W}\times C}$，scale $s$ 决定每个 grid cell 对应 $s$ 米。流程：

1. 对每帧 RGB-D，用相机内参 $K$ 把深度反投影成 3D 点云：$\mathbf{P}_k = D(\mathbf{u}) K^{-1} \tilde{\mathbf{u}}$，再用 odometry $T_{Wk}$ 转到世界系
2. 把每个 3D 点正交投影到 top-down grid，公式：

$$
p^{x}_{map} = \left\lfloor \frac{\bar{H}}{2} + \frac{P^{x}_{W}}{s} + 0.5 \right\rfloor, \quad p^{y}_{map} = \left\lfloor \frac{\bar{W}}{2} - \frac{P^{z}_{W}}{s} + 0.5 \right\rfloor
$$

3. 对 RGB 帧 $\mathcal{I}_k$ 用 LSeg 视觉编码器 $f$ 得到 dense pixel embedding $\mathcal{F}_k\in\mathbb{R}^{H\times W\times C}$（每个像素一个 CLIP-aligned 向量）
4. 每个像素的 embedding 累加到对应 grid cell，最后**对落到同一格的所有点（含跨帧）做平均**：

$$\mathcal{M}(p^{x}_{map}, p^{y}_{map}) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{q}_i$$

> ❓ 平均聚合是设计空间里最弱的一种聚合——丢辨识力且对 outlier 敏感（论文也承认这是一个 noise 来源）。后续工作用 attention / max / 概率加权做改进很自然，但作者选 mean 大概率是为了简洁与可复现。

**为什么用 LSeg 而不是直接用 CLIP**：CLIP 是 image-level alignment，pixel-level 信息靠 GradCAM 等启发式提取，噪声大（CoW 那条 baseline 的失败模式正是这个）；LSeg 在 segmentation 数据集上微调过，pixel embedding 与 CLIP text space 共享。这个选择决定了 VLMaps 的 ceiling 受限于 LSeg 训练时的 close-set 词表（appendix 里 "seating" 类 IOU 为 0 就是直接证据）。

### B. Open-Vocabulary Landmark Indexing

输入 category 列表 $\mathcal{L}=[\mathbf{l}_0,\ldots,\mathbf{l}_M]$（如 `["chair","sofa","table","other"]`），用 CLIP text encoder 得 embedding 矩阵 $E\in\mathbb{R}^{M\times C}$。把地图特征 flatten 成 $Q\in\mathbb{R}^{\bar{H}\bar{W}\times C}$，相似度 $S = Q\cdot E^T$，沿 category 维 argmax → reshape 回 $\bar{H}\times\bar{W}$ 的分割图。

要点：每张地图可以用**不同的 category 列表**重新索引，无需重训。"other" 这类 catch-all 类别充当 negative anchor，避免任意像素被强行归类。

### C. Open-Vocabulary Obstacle Map

先做 height-filter 拿到 base obstacle mask（去掉地板/天花板的点）：

$$
\mathcal{O}_{ij} = \begin{cases} 1, & t_1 \leq P^{y}_{W} \leq t_2 \text{ 且该点投到 (i,j)} \\ 0, & \text{otherwise} \end{cases}
$$

然后给 embodiment $k$ 提供一个 obstacle category 子集（如 LoCoBot 包含 "table"，drone 不包含），union 它们的 segmentation mask 与 $\mathcal{O}$ 取交集得到 $\mathcal{O}_{em_k}$。

**这是 cross-embodiment 的核心 trick**：地图建一次，per-embodiment 的成本只是写一个 obstacle list。

### D. Zero-Shot Spatial Goal Navigation via Code-Writing LLM

LLM（GPT-3 / Codex）做 few-shot prompting，把 NL 指令翻译成 Python 程序，调用一组 navigation primitives（共 17 个，见 Appendix Table IV，包括 `move_to`、`move_to_left`、`move_in_between`、`move_north`、`face`、`turn_absolute` 等）。

例子（完整 prompt 在 Appendix）：

```python
# move first to the left side of the counter, then move between
# the sink and the oven, then move back and forth to the sofa
# and the table twice
robot.move_to_left('counter')
robot.move_in_between('sink', 'oven')
pos1 = robot.get_pos('sofa')
pos2 = robot.get_pos('table')
for i in range(2):
    robot.move_to(pos1)
    robot.move_to(pos2)
```

Primitives 内部调用 §III-B 的索引拿到 landmark 坐标，加上脚本化的方位偏移（"left" 是 hardcoded 的）；下层导航走 off-the-shelf navigation stack 加 §III-C 的 obstacle map。

> ❓ "to the right of" 的语义是 hardcoded 的脚本偏移——这意味着所有 spatial relation 的灵活度受限于 primitives 表。系统把空间推理外包给了 LLM 的 program synthesis + primitives 的预定义语义，而不是让模型真正学会空间推理。

## 实验

仿真用 Habitat + Matterport3D（12,096 frames / 10 scenes），cross-embodiment 用 AI2THOR（1,826 frames / 10 rooms）。Baselines：LM-Nav、CoW、CLIP Map（ablation：把 LSeg 换成 CLIP visual features 走同样 pipeline），上界是 GT semantic map。

### Multi-Object Navigation

91 sequences × 4 subgoals，从 30 个 category 里随机挑。SR 的统计：连续完成 1-4 个子目标的成功率，以及独立 subgoal SR。

**Table I. Multi-object navigation success rate [%]. VLMaps 在所有 horizon 上稳定优于 baselines；GT Map 是上界。**

| Method         | 1 SG | 2 SG | 3 SG | 4 SG | Independent |
| -------------- | ---: | ---: | ---: | ---: | ----------: |
| LM-Nav         |   26 |    4 |    1 |    1 |          26 |
| CoW            |   42 |   15 |    7 |    3 |          36 |
| CLIP Map       |    8 |    2 |    0 |    0 |          30 |
| **VLMaps**     |   59 |   34 |   22 |   15 |          59 |
| GT Map (upper) |   91 |   78 |   71 |   67 |          85 |

CLIP Map 是 ablation——证明把 LSeg 换成 CLIP 后 mask 噪声极大——印证 LSeg 这个选择不是可有可无。

定性分析（论文 Fig. 4）：CoW 的 GradCAM saliency map 与 CLIP Map 都有大量 false positive，把规划带去错误目标；VLMaps 的 mask 显著更干净。

### Zero-Shot Spatial Goal Navigation

21 trajectories × 4 spatial subgoals（如 "east of the table"、"in between the chair and the sofa"、"with the counter on your right"、"move forward 3 meters"）。所有 map-based 方法用 §III-D 的 code generation；LM-Nav 沿用原文的 NL parsing。

**Table II. Spatial goal navigation success rate [%]. VLMaps 在每个 horizon 上 2-3× 优于次佳 baseline；与 GT Map 仍有 gap，主要差距在 spatial localization 精度。**

| Method     |   1 |   2 |   3 |   4 |
| ---------- | --: | --: | --: | --: |
| LM-Nav     |   5 |   0 |   0 |   0 |
| CoW        |  33 |   5 |   0 |   0 |
| CLIP Map   |   0 |   0 |   0 |   0 |
| **VLMaps** |  62 |  33 |  14 |  10 |
| GT Map     |  76 |  48 |  33 |  29 |

> 这张表是这篇论文最有说服力的部分——空间目标导航需要对 landmark 精确定位（不是 "接近就行"），CoW/CLIP Map 的 mask 噪声直接放大成定位失败。

### Cross-Embodiment

AI2THOR，>100 sequences，三个配置：LoCoBot+ground map、drone+ground map、drone+drone map。Drone+drone map 的 SR 持平 LoCoBot+ground map，**SPL 显著优于 drone+ground map**（同样 SR 但路径更短，drone 直接飞过 ground obstacle）。这是 "single map, multiple embodiments" claim 的直接证据。

**Figure. Cross-embodiment navigation —— 同一张 VLMap 为 LoCoBot 与 drone 生成不同 obstacle map，drone 的 drone map 中 sofa 不再是 obstacle。**

![](https://ar5iv.labs.arxiv.org/html/2210.05714/assets/x9.png)

### 真机：HSR mobile robot

374 帧建图，用 RTAB-Map 做 RGB-D SLAM 估位姿。20 条 spatial 指令成功 10 条，其中 6 条是真正的 spatial goal（"between chair and wooden box"、"south of table"），3 条相对位移、1 条带循环（"between keyboard and laptop twice"）。失败主要源于 depth noise 与 action noise——属于 VLMap 之外的传统 navigation stack 问题。

**Demo videos（成功的 spatial 指令执行）：**

<video src="https://vlmaps.github.io/static/images/back_and_forth_x4_hres_caption.mp4" controls muted playsinline width="640"></video>

*"move back and forth between the box and the keyboard"*

<video src="https://vlmaps.github.io/static/images/move_in_between_x4_hres_caption.mp4" controls muted playsinline width="640"></video>

*"move in between the wooden box and the chair"*

### Top-Down Semantic Segmentation Ablation（Appendix）

**Table V. Top-down semantic segmentation on Matterport3D categories. VLMaps 全面优于 CoW Map，频率加权 mIOU 从 42.9 → 85.9。**

| Metric                  | CoW Map | **VLMaps** |
| ----------------------- | ------: | ---------: |
| pixel accuracy          |    66.1 |   **92.3** |
| mean accuracy           |     9.6 |   **27.7** |
| mIOU                    |     5.7 |   **19.0** |
| frequency weighted mIOU |    42.9 |   **85.9** |

但 mean accuracy / mIOU 仅 27.7 / 19.0——开放词汇分割本身离 "好用" 还很远，spatial navigation 之所以能 work 是因为它**对 mask 精确度的要求低于纯 segmentation 评测**（找对大致区域就够导航）。

---

## 关联工作

### 基于
- **LSeg** (Li et al., ICLR 2022): VLMaps 的视觉 backbone，提供 CLIP-aligned dense pixel embedding。VLMaps 的 ceiling 直接受 LSeg pretraining 限制
- **CLIP** (Radford et al., 2021): 提供 text encoder 用于开放词汇索引
- **RTAB-Map** (Labbé & Michaud, 2019): 真机实验的 RGB-D SLAM 解决方案
- **Codex / GPT-3**: 用于 NL → Python program 的 code generation；这条思路与 [[2204-SayCan|SayCan]] 系列同期

### 对比
- **LM-Nav** (Shah et al., CoRL 2022): 把 image observation 存成 graph node，CLIP+GPT-3 在 graph 上规划。受限于 graph nodes 表达的位置，无法处理空间引用
- **CLIP on Wheels (CoW)** (Gadre et al., 2022): CLIP+GradCAM 生成 saliency map；GradCAM 是 image-level → pixel 的弱启发式，导致 mask 噪声大
- **CLIP Map (ablation)**: 把 VLMaps 里的 LSeg 换成直接的 CLIP visual features → mask 极差，证明 LSeg 不可替代
- **NLMap** (Chen et al., 2022, concurrent): 也是 VLM-based queryable scene representation

### 方法相关
- **Code-as-Policies** (Liang et al., ICRA 2023): code-writing LLM 用作机器人 policy 的一般框架，VLMaps 把它落到 navigation primitives
- **Socratic Models** (Zeng et al., 2022): 多模态 zero-shot 组合 paradigm，VLMaps 是其中 LLM+VLM 协作的一个具体实例
- **OpenScene / ConceptFusion / LERF (后续)**: 都把 CLIP/VLM features 与 3D representation 融合，VLMaps 是这条思路在 2D top-down map 上的早期代表，后续工作扩展到 3D NeRF/voxel/Gaussian
- **AVLMaps** (后续，同作者): 加入音频模态扩展 VLMaps 思路

---

## 论文点评

### Strengths

1. **抓住了正确的 abstraction**：把 VLM features 与 3D geometry 绑定，而非把 VLM 当 image-level critic。这是后续一大批 open-vocab semantic mapping / 3D scene representation 工作（OpenScene、ConceptFusion、LERF 等系列）的共同思路源头之一，VLMaps 是这条路线最早最简洁的代表
2. **简单到可复现**：LSeg + 反投影 + 平均 + cosine similarity，没有任何训练、没有可学参数，pipeline 可以一句话讲清楚。这是 "simple, scalable, generalizable" 的好示例
3. **Cross-embodiment 这个 use case 设计得漂亮**：用同一张地图 + 不同 obstacle list 生成不同 embodiment 的 obstacle map——这是 spatial-grounded representation 才能做到的事，纯 image-level retrieval 做不到。AI2THOR drone vs LoCoBot 的对比是 representation choice 推出的 unique capability，不是性能数字
4. **Code-as-policy 与 spatial map 的耦合**：navigation primitives 把 "spatial reasoning" 外包给了 LLM 的 program synthesis + 预定义函数，而 grounding 留给 VLMap。这种解耦让 LLM 不必懂坐标，VLMap 不必懂指令——分工干净

### Weaknesses

1. **平均聚合丢辨识力**：mean pooling 是最朴素的多视角 fusion，材质相近的 sofa/chair 混淆是直接后果（appendix Fig. 7 ab）。设计空间里有 max / attention / 概率融合等更好的选择
2. **闭集词表是隐藏 ceiling**：LSeg 是在 segmentation 数据集上微调的，"open-vocabulary" 实际上受限于 LSeg 训练分布——论文自己用 "seating" 这个词验证 IOU = 0。这把 open-vocab claim 的范围打了折
3. **空间语义是 hardcoded primitives**："left/right/north/between" 的语义在 17 个函数里被写死，模型并没有真正理解空间关系。复杂的 3D 空间关系（"on top of"、"behind"、"closest to me"）需要扩 primitives 表，scale 路径不清
4. **2D top-down 的天花板**：丢失垂直信息（"the cup on the table" vs "the cup under the table" 无法区分），3D 体素或多层 grid 能扩展但论文没探索
5. **Spatial nav benchmark 自建且小（21 trajectories）**：缺乏外部比较基准，21 条轨迹不足以做 statistically significant 的方法比较。这是 2022 年 open-vocab navigation 普遍问题
6. **真机实验 SR 50% (10/20)**：作者归因于 depth/action noise，但缺乏 ablation 证明 VLMap 本身在真实场景下的 indexing 精度。有可能 indexing 也退化但被混在 nav stack 噪声里

### 可信评估

#### Artifact 可获取性

- **代码**: inference + map building 全开源（github.com/vlmaps/vlmaps，MIT license），含 Habitat-Sim 数据生成脚本、map 创建/索引/导航评测脚本、Colab demo
- **模型权重**: 不需要——全 pipeline 用 off-the-shelf 模型（LSeg、CLIP、GPT-3/Codex）
- **训练细节**: N/A（无训练）；超参（cell size `cs`、grid size `gs`、`depth_sample_rate`、`skip_frame`）在 config 文件中暴露
- **数据集**: Matterport3D（需签 ToS 申请）+ Habitat-Sim 自采集 RGB-D 序列（10 scenes、12,096 frames），生成脚本与 pose meta data 都开源；AI2THOR 公开

#### Claim 可验证性

- ✅ **VLMaps 在 multi-object & spatial navigation 上优于 CoW/LM-Nav/CLIP Map**：Table I/II 有完整数字，code 开源可复现
- ✅ **Cross-embodiment via obstacle list works**：Table III drone+drone map vs drone+ground map 在 SPL 上的 gap 是 representation choice 的直接证据
- ✅ **LSeg 选择 > CLIP-only**：CLIP Map ablation 给出直接对比
- ⚠️ **"open-vocabulary" 的范围**：依赖 LSeg pretraining distribution，appendix 里 "seating" IOU=0 自己暴露了边界。claim 严格说应该是 "open-vocab within LSeg's pretrain coverage"
- ⚠️ **Spatial relation 的 generalization**：21 trajectories 的样本量偏小；spatial primitives 是预定义的 17 个函数，"unseen instructions" 的实际 novelty 受限于 prompt 中已经枚举过的模式
- ⚠️ **真机 SR 10/20**：归因为 depth/action noise，缺独立 ablation 隔离 VLMap indexing 在真实场景下的精度

### Notes

- **Open question — VLM features 的多视角融合**：mean pooling 是最简单的，是否能用 attention / 信任度加权 / probabilistic fusion 提升辨识力？这是 VLMaps 留下的一个明显改进空间
- **Open question — primitives 的 scale 路径**：17 个 hardcoded primitives 是 spatial relation 的瓶颈。能否让 LLM 直接生成 occupancy/cost field 操作的代码（即 spatial relation 也由 LLM 表达），而不是依赖预定义函数？
- **Open question — 2D top-down 的局限**：垂直语义信息丢失（"on/under the table"），后续 3D 工作（OpenScene 等）解决了这个，但 2D 的简洁性也丢了。是否存在 multi-layer grid 这样的折中？
- **可作为 baseline / 教学案例**：方法极简、code 开源、Colab 可跑，适合作为 open-vocab navigation 的入门 baseline

### Rating

**分数**：3 - Foundation

**理由**：VLMaps 是 "VLM features 绑定 3D geometry" 这条路线最早最简洁的代表（见 Strengths 1），后续 OpenScene/ConceptFusion/LERF 等 open-vocab 3D scene representation 工作都承袭这个 abstraction，引用数超 800（ICRA 2023）、Google Scholar 上被 open-vocab navigation / semantic mapping 领域主要后续工作（CLIP-Fields、OK-Robot、VLFM 等）持续作为 baseline 引用。方法极简（LSeg+反投影+平均）使其成为可复用 building block，而非仅是 SOTA 数字——这让它稳在 3 档而非 2 档：不是 "当前 SOTA"（2D top-down 已被后续 3D 表征超越），而是方向的奠基 reference。
