---
title: "OpenSpatial: A Principled Data Engine for Empowering Spatial Intelligence"
authors: [Jianhui Liu, Haoze Sun, Wenbo Li, Yanbing Zhang, Rui Yang, Zhiliang Zhu, Yijun Yang, Shenghe Zheng, Nan Jiang, Jiaxiu Jiang, Haoyang Huang, Tien-Tsin Wong, Nan Duan, Xiaojuan Qi]
institutes: [JD, HKU]
date_publish: 2026-04-08
venue: arXiv
tags: [spatial-reasoning, VLM, 3D-representation]
paper: https://arxiv.org/abs/2604.07296
website: 
github: https://github.com/VINHYU/OpenSpatial
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] OpenSpatial: A Principled Data Engine for Empowering Spatial Intelligence
> - **核心**: 把"空间数据生产"从静态数据集升级成开源、可控、可扩展的 data engine——以 **3D Oriented Bounding Box (OBB)** 作为唯一原语，通过 manual annotation + automated 3D lifting 两条路径，跨 5 类任务程序化合成 3M 条 spatial QA。
> - **方法**: OBB-centric scene graph → frame-level attribute mapping (projection + visibility 过滤 + SAM mask 精修) → single-view / multi-view QA 合成；automated lifting pipeline 用 Gemini 做 per-view recognition、SAM 做 mask、3D 关联 + convex hull 拟合 OBB，把 in-the-wild web video 也纳入数据源。
> - **结果**: SFT on Qwen3-VL-8B / Qwen2.5-VL-7B / InternVL3-8B / InternVL2.5-8B，3D-Avg 平均提升 ~14%，最大单基座增益 19% (Qwen2.5-VL-7B)；在 BLINK / AllAngles / MMSI 上超过 SenseNova-SI 与 [[2401-SpatialVLM|SpatialVLM]]。
> - **Sources**: [paper](https://arxiv.org/abs/2604.07296) | [github](https://github.com/VINHYU/OpenSpatial)
> - **Rating**: 2 - Frontier（开放整套 spatial data engine 的立场到位、OBB-centric 设计清晰、multi-backbone 验证扎实，但核心归因 ablation 缺失、3D lifting 定量质量未评估、关键 artifact 仍在 roadmap，不足以成为 Foundation）

**Key Takeaways:**
1. **3D box-centric supervision 是这套 engine 的"杠杆"**：OBB 同时提供 viewpoint-invariance（解决跨视图对应）+ metric scale（支持距离/尺寸 QA）+ canonical anchor（支撑 mask 精修和遮挡过滤），是 "weak 2D label" 与 "expensive dense 3D recon" 之间的中间表示。
2. **可扩展性来自 automated 3D lifting**：让数据脱离 EmbodiedScan 等 curated 室内集，能从 in-the-wild web video 生成 3D 标注，缓解 spatial data 的"室内偏置"。
3. **Task taxonomy 是 "spatial myopia" 的解药**：5 类任务 (SM/SR/CP/MC/SAR) 跨 19 子任务，ablation 显示任务多样性比单纯堆数据更能拉动 3D-Avg。
4. **Data scaling 已有 diminishing return**：20%→100% 数据量，3D-Avg 从 57.6 → 59.7，曲线趋平——3M 已接近 single-engine 的上限，再涨需要换源或换 backbone。

**Teaser. OpenSpatial pipeline overview——左：从 2D web data 到 3D spatial supervision 的整体流；右：OpenSpatial-3M SFT 在多 backbone × 多 benchmark 上的提升。**

![](https://arxiv.org/html/2604.07296v2/x1.png)

---

## 1. Motivation：为什么 spatial intelligence 卡在数据上

作者把当前 spatialized VLM 的瓶颈归因到**数据基础设施**而非模型架构：
- **Spatial myopia**：现有 spatial 数据集（VST、SenseNova-SI、Cambrian-S）在某些 benchmark 上 SOTA，但任务覆盖窄，泛化差。
- **黑箱 pipeline**：多数工作只 release 预处理后的 fixed dataset，**生成引擎闭源**，无法做受控 ablation、无法判断到底是哪个 design choice 在 work——这才是阻碍累积式进步的根本原因。

OpenSpatial 的立场：**放弃 "再发一个 dataset" 的范式，开源整套生成 infrastructure**。三个 design pillar：
1. **3D box-centric grounding**——quality 来自 OBB 的 viewpoint-invariance；
2. **3D lifting**——scalability 来自能把 unlabeled web data 变成可用监督；
3. **scene-graph driven synthesis**——diversity 来自程序化枚举 object × attribute × relation。

> ❓ 第 (3) 点的 "programmatic enumeration" 在论文里没有给出 prompt 模板或具体规则，只能等 GitHub 后续放 evaluation suite + 3D lifting 模块（Roadmap 仍未完成）才能完整复现。

---

## 2. 数据 Pipeline

**Figure 2. Data engine 全貌——左：标注 pipeline（scene-level OBB → frame-level attributes → single/multi-view QA）；右：数据源分布与任务分布统计。**

![](https://arxiv.org/html/2604.07296v2/datapipline_cropped_v1_compressed.png)

### 2.1 3D Box-Centric 表示

每个物体参数化为 OBB:

$$
(x, y, z, x_l, y_l, z_l, r, p, y)
$$

**符号说明**：$(x,y,z)$ 物体在 world coordinate 的中心；$(x_l,y_l,z_l)$ 沿 X/Y/Z 轴的边长；$(r,p,y)$ Roll/Pitch/Yaw。所有 box 都在 Z-up 全局坐标系下定义。

**含义**：OBB 是**最小够用**的 3D 表示——比 2D box 多了 depth/extent/orientation（够支撑 metric reasoning），又比 dense 3D mesh 便宜得多（够 scale）。这个选择决定了整套 engine 能 work：太弱的表示无法支撑跨视图对应，太重的表示无法 scale。

### 2.2 Scene-level OBB 标注：两条路

| 模式 | 实现 | 优劣 |
|---|---|---|
| **Manual annotation** | 沿用 EmbodiedScan 的标注协议 | 高精度，但慢且贵 |
| **Automated 3D lifting** | per-view object recognition (Gemini) → instance mask (SAM) → 3D 空间关联与合并 → convex hull 拟合 OBB | 可 scale 到 web data / 开源 asset，无需 fine-grained label |

> ❓ 论文没披露 lifting pipeline 的精度（仅有 Fig. 4 的 qualitative outdoor 可视化）。Tab. 6 验证了 web-only 数据的 downstream 效果，但没有直接的 box IoU / recall 指标。这是复现时的硬骨头。

### 2.3 Attribute-Centric Object–Frame Mapping

把 scene-level box 投影到每帧后做两道过滤：
1. **Frustum 过滤**：camera frustum 外的 box 直接弃。
2. **Depth-based occlusion 过滤**：投影 2D box 内像素经 depth map 反投影到 world，算 3D box 内的体素占用率，低于阈值视为遮挡剔除。

通过过滤的 box 用其反投影点云作为 coarse mask，再喂 SAM 精修得到 fine-grained 2D instance mask。每个 object 被打上 **metric flag**：若为 False，跳过测量类 QA 生成——这是避免 noisy supervision 的关键 trick。

### 2.4 Scene-Graph Driven QA Synthesis

- **Single-view QA**：基于 per-frame scene graph，对多实例同语义场景渲染 marked image 作为视觉 anchor 防止 referential ambiguity；生成 relational / attribute comparison / context reasoning 三类 QA。
- **Multi-view QA**：用 OBB 作为 viewpoint-invariant key 采样**有共享 box 的视图对**（既保证 overlap 又保留视角差异），合并构造 unified multi-view scene graph，生成 re-identification / camera change / consistency 类 QA。

---

## 3. OpenSpatial-3M Dataset

**Figure 3. OpenSpatial-3M 的 5 类 capability，每类附 representative QA 样例。**

![](https://arxiv.org/html/2604.07296v2/x2.png)

**数据源**: 沿用 VST 的策略，以 EmbodiedScan 标注的 3D box 为基础（聚合 ScanNet / Matterport3D / ARKitScenes / SUN-RGBD，但**剔除了 SUN-RGBD**因为标注质量差），补充 ScanNet++、Hypersim 增加环境多样性，再加自采 web data。

**Task taxonomy** (5 大类 / 19 子任务):
- **SM (Spatial Measurement)**: length / width / height / distance 等 metric 量化
- **SR (Spatial Relationship)**: 相对方位、inter-object dependency
- **CP (Camera Perception)**: camera pose、object-camera 相对关系
- **MC (Multi-view Consistency)**: 跨视角同物体识别、共享物体的几何一致性
- **SAR (Scene-Aware Reasoning)**: 场景级布局、planning / navigation 可行性

> 这套 taxonomy 实质上是"perception → relation → ego-aware → cross-view → scene-level"的渐进 curriculum。SAR 是其中最 ambitious 的——但 paper 里 SAR 的 QA 实例和评测占比都偏低，还像是占位。

---

## 4. Experiments

### 4.1 训练设置

- 单 epoch SFT，32 张 NVIDIA GPU，global batch size 128，AdamW，base lr $5\times 10^{-5}$，vision encoder lr 解耦为 $5\times 10^{-6}$。
- **训练数据混合**: OpenSpatial-3M + SenseNova-800K (补充未覆盖维度) + LLaVA-OneVision general data (1:1 ratio 与 spatial data) 防 catastrophic forgetting。

> ❓ "OpenSpatial 单独 vs OpenSpatial+SenseNova 混合" 的 ablation 没有给——主表里所有 "Ours" 的提升都是混合后的结果，无法干净归因到 OpenSpatial 自身。Tab. 2 的 "subset 500K vs SenseNova 800K" 的对比是数据质量层面的，而非 SFT 的最终效果对比。

### 4.2 主结果

**Table 1. 主要结果（节选）——4 个 baseline + OpenSpatial SFT 后的提升幅度（带 ± 表示相对 baseline 的变化）。**

| Model | 3D-Avg | BLINK | AllAngles | VSI | MMSI | CV-3D |
|---|---|---|---|---|---|---|
| InternVL2.5-8B (baseline) | 51.6 | 54.9 | 48.9 | 39.3 | 28.6 | 79.9 |
| OpenSpatial-InternVL2.5-8B | **59.3 (+7.7)** | 63.5(+8.6) | 58.3(+9.4) | 56.7(+17.6) | 38.7(+10.1) | 93.8(+13.9) |
| InternVL3-8B (baseline) | 53.2 | 55.7 | 50.5 | 38.7 | 30.9 | 86.0 |
| OpenSpatial-InternVL3-8B | **59.8 (+6.6)** | 66.0(+10.3) | 58.3(+7.8) | 57.4(+18.7) | 38.6(+7.7) | 93.7(+7.7) |
| Qwen2.5-VL-7B (baseline) | 50.0 | 55.3 | 50.1 | 36.0 | 26.5 | 73.8 |
| OpenSpatial-Qwen2.5-VL-7B | **59.5 (+9.5)** | 65.9(+10.6) | 58.4(+8.3) | 56.7(+20.7) | 39.6(+13.1) | 92.5(+18.4) |
| Qwen3-VL-8B (baseline) | 56.7 | 66.1 | 49.5 | 55.6 | 28.1 | 90.8 |
| OpenSpatial-Qwen3-VL-8B | **62.1 (+5.4)** | 68.2(+2.1) | 59.8(+10.3) | 61.6(+6.0) | 41.9(+13.8) | 94.0(+3.2) |

**观察**：
- 在 BLINK / AllAngles / MMSI 三个 benchmark 上提升最大（>10 pt），这些都是要求 cross-view consistency 与精细 spatial relation 的题。
- Qwen3-VL-8B 增益最小（+5.4），作者归因于其 SigLIP encoder 已经较强；但也说明 base model 越强，data engine 的边际收益越小。
- 在 RealWorldQA 上某些模型甚至 -3.9 / -0.8——存在轻微 domain shift。

### 4.3 Module Ablation：哪些设计真在 work

**Table 3. 把 box-centric 换成 point-cloud-centric 之后的性能下降——partial point cloud 无法表达完整几何，对 SM 任务伤害最大。同时验证了 occlusion filtering 的必要性。**

![](https://arxiv.org/html/2604.07296v2/module_resonable_compressed.png)

两个核心结论：
1. **Box-centric > point-cloud-centric**：partial point cloud 几何残缺，导致 SM 类标注噪声大。
2. **Visibility filtering 必须做**：不过滤会因为 occluded 物体的伪标签产生 hallucination。

### 4.4 Data Comparison：数据质量 vs 其他开源 spatial 数据集

**Table 2. Scale-matched 子集对比（500K-800K，统一 Qwen2.5-VL backbone）。MAD = Mean Absolute Deviation from best, 越接近 0 越好。**

| Data source | Size | MAD | Std. Dev. | BLINK | AllAngles | VSI | MMSI | CV-3D |
|---|---|---|---|---|---|---|---|---|
| Cambrian-S | 590k | -6.0 | 5.4 | 54.1 | 48.6 | **57.0** | 29.2 | 75.3 |
| SenseNova-SI | 800k | -6.5 | 7.0 | 59.7 | 47.5 | 55.0 | **36.3** | 69.0 |
| VST | 500k | -2.8 | 3.9 | 61.4 | 50.7 | 44.6 | 32.4 | **93.2** |
| OpenSpatial (subset) | 500k | **-2.5** | 4.4 | **64.2** | **53.9** | 43.0 | 34.7 | 91.8 |

**Insight**: OpenSpatial 与 VST 都是"全能型"数据集（MAD 最小），但 Cambrian-S 在 VSI、SenseNova 在 MMSI 上有 niche 优势——所以作者最终在 full SFT 里**混入 SenseNova-800K**来取互补。

### 4.5 Scaling

**Table 4. Data scaling——20% → 100%，3D-Avg 单调上升但收益递减（57.6 → 59.7）。**

| Data Size | 3D-Avg | BLINK | AllAngles | VSI | MMSI |
|---|---|---|---|---|---|
| 20% | 57.6 | 64.9 | 55.0 | 51.8 | 34.7 |
| 40% | 58.5 | 66.2 | 56.8 | 52.4 | 35.8 |
| 60% | 58.6 | 66.9 | 56.7 | 53.2 | 36.0 |
| 80% | 59.2 | 66.2 | 59.7 | 53.5 | 37.6 |
| Full | **59.7** | 65.9 | 58.4 | 56.7 | **39.6** |

**Table 5. Model scaling——3B → 7B → 32B 单调改善，证明 engine 能持续转化模型容量为 spatial 能力。**

| Model Size | 3D-Avg | BLINK | AllAngles | VSI | MMSI | CV-3D |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 56.1 | 61.0 | 53.0 | 55.2 | 32.0 | 94.0 |
| Qwen2.5-VL-7B | 59.7 | 65.9 | 58.4 | 56.7 | 39.6 | 92.5 |
| Qwen2.5-VL-32B | **61.3** | **68.2** | **63.3** | 57.3 | 39.8 | 93.4 |

### 4.6 In-the-Wild 3D Lifting

**Figure 4. 3D lifting 在未经精选的 outdoor web video 上的可视化——能恢复点云几何 + 输出 tag 与 3D box。**

![](https://arxiv.org/html/2604.07296v2/x3.png)

Tab. 6 进一步验证了**仅用 web-sourced data** 走 3D lifting 走出来的 dataset 也能带来 spatial 能力提升（虽然论文里 Tab. 6 的具体数字未在主文展示完整）。

### 4.7 Task Diversity Ablation

**Figure 5. 任务多样性的影响——左：每类任务在不同 benchmark 上的贡献热力图，证明任务库非冗余；右：incremental task integration 后 3D-Avg 单调上升。**

![](https://arxiv.org/html/2604.07296v2/x4.png)

---

## 关联工作

### 基于
- **EmbodiedScan**: 提供 manual 3D OBB 标注协议，OpenSpatial 的 ground truth box 来源
- **VST**: 训练 protocol（1 epoch / 32 GPU）和数据源策略（聚合 ScanNet/Matterport3D/ARKitScenes）都直接沿用
- **SAM**: instance mask 提取（automated lifting + frame-level mask 精修）
- Gemini-2.5: automated lifting 中 per-view object recognition

### 对比
- **VST**: 同样 500K spatial dataset，OpenSpatial subset 的 MAD 略好（-2.5 vs -2.8）；OpenSpatial 的卖点是开放整个 engine 而非只开放数据
- **SenseNova-SI**: 800K，在 MMSI 上是 niche 强者；OpenSpatial 选择**互补混合**而非替代
- **Cambrian-S**: 590K，VSI 上的 niche 强者；同样是互补关系
- [[2401-SpatialVLM|SpatialVLM]] / SpatialRGPT: 早期大规模 spatial VQA 合成工作，OpenSpatial 的 task taxonomy 比它们更系统
- **3DThinker / VLM-3R / Spatial-MLLM**: 走"加 3D encoder"的架构路线，OpenSpatial 走纯数据路线，证明无需改架构也能拿到 SOTA

### 方法相关
- **EmbodiedScan-style OBB 标注**: 9 维 (x,y,z,xl,yl,zl,r,p,y) Z-up world coordinate
- **Scene graph driven QA synthesis**: programmatic enumeration of objects × attributes × relations
- **Marked image rendering**: 解决多实例同语义场景的 referential ambiguity（在 query 物体上加 visual marker）

---

## 论文点评

### Strengths

1. **Problem framing 准确**：把"空间数据是黑箱"作为根因点出来，立场清晰。这是社区当下确实需要的——VST / Cambrian-S / SenseNova-SI 都只 release 数据不 release 引擎，做 controlled comparison 极难。
2. **OBB 这个表示选得好**：viewpoint-invariance + metric scale + canonical anchor，三件事一个 representation 全占了，是这套 pipeline 能 scale 的关键。Tab. 3 的 box vs point cloud ablation 是有说服力的支持证据。
3. **5 类任务 taxonomy 比 prior work 更系统**：SM/SR/CP/MC/SAR 覆盖了 perception → relation → ego → cross-view → scene 的完整 spectrum，且 task complementarity 的 heatmap 让人信服这不是事后凑分类。
4. **Multi-backbone 验证**：4 个不同 base model 都受益，说明改进归因于数据而非某个 model-specific trick。

### Weaknesses

1. **核心 ablation 缺失**：主表所有 "Ours" 都是 OpenSpatial-3M + SenseNova-800K + LLaVA-OneVision 的混合 SFT，**纯 OpenSpatial-3M 单独效果在主表里没有**。Tab. 2 的 500K subset 对比是数据质量层面的，但 full-scale 的清洁归因被刻意省略了。
2. **3D lifting pipeline 缺少定量评估**：只有 Fig. 4 的 qualitative outdoor 可视化和 Tab. 6 的下游效果。box IoU / recall / scale accuracy 这些直接质量指标没给——而这恰恰是整个 scalability claim 的根基。
3. **Roadmap 多数未完成**：截至 release，3D lifting module、evaluation suite、trained model 都未开源（仅数据 + engine 框架）。这削弱了 "open-source data engine" 的实际可用性。
4. **Outdoor / desktop 仍是 weak spot**：作者自己承认 RealWorldQA 上有些 model 出现 -3.9 / -0.8 的轻微下降，归因于数据分布偏 indoor。
5. **Data scaling 收益递减明显**：20%→100% 只涨 2.1 pt，说明 single-engine 已接近边际。再涨需要换数据源（更多 outdoor）或换更强 backbone，纯堆量已不划算。

### 可信评估

#### Artifact 可获取性

- **代码**: 部分开源——core 3D data engine 已 release（2026-04-08），但 evaluation suite、3D lifting module、trained model 均在 roadmap 标记为未完成（截至 2026-04-15）
- **模型权重**: 未发布（roadmap 中"Model Release"未勾选）
- **训练细节**: 仅高层描述（lr / batch / GPU 数 / 数据混合比，但没有 step / warm-up / 具体 SFT 数据 schedule）
- **数据集**: 部分公开——OpenSpatial-3M 的 open-source subset 已上 HF (`jdopensource/JoyAI-Image-OpenSpatial`)，是否完整 3M 不明

#### Claim 可验证性

- ✅ "3D-Avg 提升 14% 平均 / 19% 最大"：Tab. 1 完整数据可查，多 backbone 一致性良好
- ✅ "Box-centric > point-cloud-centric"：Tab. 3 ablation 有 quantitative + qualitative 双重证据
- ⚠️ "OpenSpatial-3M 是性能提升来源"：归因被混合训练（+SenseNova-800K +LLaVA-OneVision）稀释，缺少纯 OpenSpatial-3M 的清洁对照
- ⚠️ "3D lifting 能 scale 到 in-the-wild"：只有 Fig. 4 qualitative + Tab. 6 间接下游证据，无 lifting 质量的定量评估
- ⚠️ "Versatile / generalizable"：Outdoor / desktop / RealWorldQA 上存在 regression，"versatile" 主要适用于 indoor scene
- ❌ "Democratize the creation of high-quality data"：在 evaluation suite 与 lifting module 都未开源前，这更像愿景而非已交付的能力

### Notes

- 这篇 paper 在 framing 上比方法本身更有价值——"open the engine, not just the dataset" 应该成为后续 spatial data 工作的 default 立场。
- 5 类任务 taxonomy 可作为后续 spatial benchmark 设计的 reference scaffold。
- 关心的 follow-up:
  - 3D lifting 的精度边界在哪？outdoor / cluttered / dynamic scene 的 box recall 是多少？
  - 当 SenseNova-SI 整体被 OpenSpatial 完全覆盖后（即 OpenSpatial 自己产生 MMSI 强项数据），是否还需要混合？
  - 把这套数据用于 [[2604-GeminiRoboticsER16|Gemini Robotics ER]] 这类 embodied reasoning 模型，是否对真实 manipulation 决策有 transfer？目前所有验证都停留在 VLM benchmark 层面，离 embodied 还有一道桥。
- 训练 protocol 与 [[2401-SpatialVLM|SpatialVLM]] 一脉相承，但 task taxonomy 更宽（SpatialVLM 主要 SM+SR，OpenSpatial 多 CP/MC/SAR 三类）。

### Rating

**分数**：2 - Frontier
**理由**：按 field-centric rubric，OpenSpatial 是 spatial VLM 数据工程方向当前必须参考的 Frontier 工作——OBB-centric engine + 5 类任务 taxonomy + multi-backbone 一致提升（3D-Avg +5.4~+9.5）让它成为后续 spatial data pipeline 的 default baseline；但距离 Foundation 仍差两步：(a) 核心归因 ablation 被混合训练稀释、3D lifting 缺定量质量指标，可信度未拉满；(b) evaluation suite / lifting module / model weights 都还在 roadmap，尚未成为社区 de facto 标准。若后续 roadmap 全部落地且被广泛采用，可升至 3。
