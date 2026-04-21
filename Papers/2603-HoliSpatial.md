---
title: "Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence"
authors: [Yuanyuan Gao, Hao Li, Yifei Liu, Xinhao Ji, Yuning Gong, Yuanjun Liao, Fangfu Liu, Manyuan Zhang, Yuchen Yang, Dan Xu, Xue Yang, Huaxi Huang, Hongjie Zhang, Ziwei Liu, Xiao Sun, Dingwen Zhang, Zhihang Zhong]
institutes: [Shanghai AI Lab, Northwestern Polytechnical University, Shanghai Jiao Tong University, Peking University, Nanyang Technological University, Beihang University, Sichuan University, Tsinghua University, CUHK, Fudan University, HKUST]
date_publish: 2026-03
venue: arXiv
tags: [3D-representation, scene-understanding, spatial-reasoning]
paper: https://arxiv.org/abs/2603.07660
website: https://visionary-laboratory.github.io/holi-spatial/
github: https://github.com/Visionary-Laboratory/holi-spatial
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence
> - **核心**: 把 raw video 自动 lift 成 3DGS 场景 + 全套 3D 语义标注（mask / OBB / caption / grounding / spatial QA），完全无需人工或 3D sensor，绕开手工 3D 数据这个 spatial intelligence 的瓶颈。
> - **方法**: 三阶段 pipeline——(i) SfM + 3DGS 几何优化（用 Depth-Anything-V3 初始化、几何正则消 floater）；(ii) 关键帧上 VLM 给 open-vocab 类别 + SAM3 出 mask + 用 GS depth back-project 到 3D OBB；(iii) 多视角 IoU merge + confidence 三档 + VLM agent verify。在此 pipeline 上构建 Holi-Spatial-4M，再用它 fine-tune Qwen3-VL。
> - **结果**: ScanNet++ 上 Depth F1 0.89（M3-Spatial 0.39）、3D Det AP25 81.06（LLaVA-3D 12.2）；Qwen3-VL-8B + Holi-Spatial-4M 在 MMSI-Bench +1.5、MindCube +19.7、ScanNet++ 3D Grounding AP50 +14.48。
> - **Sources**: [paper](https://arxiv.org/abs/2603.07660) | [website](https://visionary-laboratory.github.io/holi-spatial/) | [github](https://github.com/Visionary-Laboratory/holi-spatial)
> - **Rating**: 2 - Frontier（问题 framing 准 + pipeline 工程完整且 artifact 已 release，是当前 "data engine for spatial intelligence" 方向的重要参考；但 raw-video misnomer + 同分布 eval + VLM fine-tune 提升需打折，尚未到 Foundation 档。）

**Key Takeaways:**
1. **数据 flywheel 而非新模型**：作者的 thesis 是 spatial intelligence 的瓶颈不在 architecture 而在数据——manual scan + 50-class label 的 ScanNet 类数据集 vs. LAION-5B 的 billion-scale 图像，scale 差好几个数量级。Holi-Spatial 把 "需要手工 3D 数据" 这个隐含假设拆掉，改用 web video + 现成 foundation model 自动生成。
2. **三阶段 coarse-to-fine 是关键**：feed-forward depth (DA3) 单独不够（P25 仅 0.13），3DGS per-scene 优化把 P25 推到 0.81——表明 multi-view 几何一致性是 3D OBB 质量的瓶颈，而非 2D 感知。
3. **Confidence + Agent 互补**：单纯 confidence filter 提 precision 但伤 recall（P 0.35→0.67，R 0.74→0.69），加 VLM agent re-verify 把 borderline case 拉回来，最终 P/R 都到 0.81/0.89。
4. **Fine-tune 收益不均**：MMSI-Bench 提升只 1.5pt（绝对值小），MindCube 提升 19.7pt。值得追问哪些 spatial reasoning sub-skill 真的被 transfer——QA 模板覆盖的 camera/object-centric 任务可能恰好对齐 MindCube 而非 MMSI。

**Teaser. Holi-Spatial 数据 pipeline + 下游 VLM fine-tune 总览。**

![](https://arxiv.org/html/2603.07660v1/x1.png)

---

## 1. Motivation：Spatial Intelligence 的数据 bottleneck

作者的 framing 直接、犀利：spatial intelligence 落后于 2D 视觉理解的根因不是模型架构，而是 raw 3D data 的 scale & diversity 远不够。

- **现状对照**：LAION-5B 提供 billion-scale 2D 图像；spatial domain 即使是号称 million-level 的 SenseNova-SI-800K / VST-4M，底层 3D scene pool 也只有几千个静态 scan（ScanNet 体量）。生成 1.2M QA 但只 grounding 在 1k 场景上 → 严重 over-sample 同一 distribution。
- **三种现有 paradigm 的 scalability 瓶颈**（Sec 2.2）：
  1. 3D-native LMM（SpatialLM、LLaVA-3D）：依赖手工标注的 mesh / point cloud；
  2. 2D-centric spatial LMM（VST、Cambrian-S）：scale up SFT/RL 数据但底层 scene 还是手工标的；
  3. 3DGS-based（M3-Spatial、LangSplat）：per-scene optimization，不能批量跑。

> ❓ 这个 framing 我大体认同，但有个 caveat：scaling 2D 用 LAION 也是经过 CLIP score filter 的——纯 raw web 数据噪声很大。Holi-Spatial 自动化但仍需 ScanNet/ScanNet++/DL3DV 这种**已经 captured 的视频**作为输入，并不是真正从随便的 web video 跑出来。表面上是 "raw video"，实际上还是 indoor scan video distribution。真正的 generalization 测试得拿 YouTube vlog 一类 in-the-wild egocentric video 来跑。

**Table 1. Pipeline capability 对比。** Holi-Spatial 是唯一同时输出 depth / 2D seg / 3D det / grounding / spatial QA 的框架（仅以 image 为输入）。

| Method | Img | PC | Depth | 2D Seg | 3D Det | Grounding | Spatial QA |
|---|---|---|---|---|---|---|---|
| SAM3 | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| SA2VA | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| SpatialLM | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| LLaVA-3D | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| M3-Spatial | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| Holi-Spatial | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 2. Method：三阶段 Pipeline

整体 pipeline 见下图，三阶段串行：Geometric Optimization → Image-level Perception → Scene-level Refinement。

**Figure 3. Holi-Spatial 数据 curation pipeline 总览。**

![](https://arxiv.org/html/2603.07660v1/x3.png)

### 2.1 Stage I：Geometric Optimization

目的是从 raw video 拿到高保真 3D geometry，作为后续 2D-to-3D lift 的基础。

- **Step 1**: SfM 估计 camera intrinsics + extrinsics；
- **Step 2**: 用 Depth-Anything-V3（"spatial foundation model"）初始化 dense point cloud；
- **Step 3**: 3DGS per-scene 优化，加 surface reconstruction 风格的几何正则（multi-view depth consistency）压制 floater。

> 这里不算方法 novelty——基本是 surface-aware 3DGS 的现有配方（refs 44/41/48/51/52/50）。novelty 体现在 "把它放在 pipeline 里给后续阶段提供干净 depth"。

### 2.2 Stage II：Image-level Perception + 2D→3D OBB Lifting

关键帧均匀采样 → Gemini3-Pro 生成 caption → 维护一个 cumulative class-label memory $\mathcal{M}_t = \mathcal{M}_{t-1} \cup \text{Extract}(I_t)$ 强制 cross-frame 标签一致性 → SAM3 用这个 vocabulary 做 open-vocab segmentation。

每个 mask pixel 用 GS-rendered depth 反投影到 3D：

$$
\mathbf{P} = D_t(\mathbf{u}) \cdot \mathbf{K}^{-1} \tilde{\mathbf{u}}
$$

但直接 OBB-fit 这些 3D 点会被 depth edge floater 污染。作者用四步 filter（图 4）解决：

**Figure 4. 2D-to-3D OBB 生成流程。** (1) GS depth ⊕ SAM3 mask 得到 object depth；(2) mask erosion 去 SAM3 边界误差；(3) mesh-guided depth filter 去 3D outlier；(4) 从 refined point cloud fit OBB。

![](https://arxiv.org/html/2603.07660v1/x4.png)

OBB 还要再做 floor-aligned post-processing：估全局 up-axis（用 floor / 主 planar structure），yaw-lock + PCA fallback re-align 每个 instance。

**Figure 5. Floor-aligned OBB post-processing。**

![](https://arxiv.org/html/2603.07660v1/x5.png)

### 2.3 Stage III：Scene-level Refinement

三个动作：

**(1) Multi-View Merge** — 同类别 + 3D IoU > 0.2 就 merge。merge 后保留最大 confidence 的 source frame 作为 "canonical view"，下游 captioning 都从这个 view 出。

**(2) Confidence-Based Tri-Level Filter**：

$$
\operatorname{Action}(p_k) = \begin{cases}
\text{keep}, & s_k \geq \tau_{\text{high}} = 0.9 \\
\text{discard}, & s_k < \tau_{\text{low}} = 0.8 \\
\text{verify}, & 0.8 \leq s_k < 0.9
\end{cases}
$$

中间 verify band 走 VLM agent，agent 配 image zoom-in tool + SAM3 re-segmentation tool 重新打分。

**(3) Caption + QA 生成**：每个 final instance 用 Qwen3-VL-30B 在 canonical view 上写 fine-grained caption，用 template 程式化生成 spatial QA。

> 这个 tri-level + agent recall 是整个 pipeline 里我觉得最 elegant 的设计。Pure threshold 总是 P-R trade-off；引入 LLM agent 把 borderline 决策从 "拍一个阈值" 升级为 "做一次 visual reasoning"，而且把昂贵的 VLM call 限制在 borderline 比例上，cost 可控。

---

## 3. Holi-Spatial-4M Dataset

源：ScanNet + ScanNet++ + DL3DV-10K → 12K 优化过的 3DGS 场景。annotation 数量：

| 类别 | 数量 |
|---|---|
| 优化 3DGS 场景 | 12K |
| 2D instance mask | 1.3M |
| 3D bounding box | 320K |
| Instance caption | 320K |
| 3D grounding pair | 1.2M |
| Spatial QA pair | 1.25M |

**Figure 6. Holi-Spatial-4M 统计：(1) word cloud 显示 long-tail open-vocab 类别；(2) 数据来源 + annotation breakdown；(3) QA taxonomy（camera-centric vs. object-centric）。**

![](https://arxiv.org/html/2603.07660v1/x6.png)

QA 分两类：
- **Camera-centric**：camera rotation / movement direction / movement distance；
- **Object-centric**：object-object distance / global+local direction / object measurement / camera-object distance。

> ❓ "1.2M 3D grounding instances 来自 12K 场景" → 平均每场景 100 个 grounding query，对单 scan 来说已经是 over-saturate。再考虑 ScanNet/ScanNet++/DL3DV 本身 distribution overlap，等效 "unique scene-instance" 数远小于 1.2M。这点和 paper 自己批 "VST-4M 来自 a few thousand scenes" 的论调有点 self-irony——Holi-Spatial-4M 也不是 truly diverse。

---

## 4. Experiments

### 4.1 Pipeline Quality（Annotation 质量评估）

设置：每个 dataset 抽 10 scenes 手工标 GT 2D mask + 3D box，depth 用官方 GT。

**Table 2. 3D spatial understanding 在 ScanNet / ScanNet++ / DL3DV 上的对比。**

| Method | ScanNet Depth/2D/AP25/AP50 | ScanNet++ | DL3DV |
|---|---|---|---|
| SAM3 (2D-VLM) | – / 0.63 / – / – | – / 0.50 / – / – | – / 0.66 / – / – |
| SA2VA (2D-VLM) | – / 0.64 / – / – | – / 0.25 / – / – | – / 0.44 / – / – |
| SpatialLM (3D-VLM) | – / – / 11.42 / 8.19 | – / – / 9.11 / 6.23 | – / – / 7.05 / 4.38 |
| LLaVA-3D | – / – / 9.13 / 6.86 | – / – / 12.2 / 4.80 | – / – / 6.83 / 4.11 |
| SceneScript | – / – / 8.97 / 3.54 | – / – / 9.86 / 4.42 | – / – / 5.65 / 3.98 |
| M3-Spatial (3DGS) | 0.32 / 0.22 / – / – | 0.39 / 0.11 / – / – | 0.23 / 0.13 / – / – |
| LangSplat (3DGS) | 0.19 / 0.36 / – / – | 0.21 / 0.06 / – / – | 0.18 / 0.24 / – / – |
| **Holi-Spatial** | **0.98 / 0.66 / 76.60 / 67.00** | **0.89 / 0.64 / 81.06 / 70.05** | **0.78 / 0.71 / 62.89 / 52.67** |

3D detection 的 AP25 从 baseline 的 ~10 跳到 80+，差了一个数量级。这种 gap 当然部分是因为 3D-VLM 是 zero-shot 推理 vs. Holi-Spatial 是 per-scene 优化；但作者的 framing 是 "数据生成 vs. 模型推理"，比较的对象本就不同 task。

**Figure 7. ScanNet++ 上 multi-view depth 对比。** 右侧 multi-view back-projection 的 point cloud 显示 Holi-Spatial 几乎无 ghosting。

![](https://arxiv.org/html/2603.07660v1/x7.png)

**Figure 8. Open-vocab 2D segmentation 对比。**

![](https://arxiv.org/html/2603.07660v1/x8.png)

**Figure 9. ScanNet++ 上 3D detection 对比。**

![](https://arxiv.org/html/2603.07660v1/x9.png)

### 4.2 VLM Fine-tuning Evaluation

Qwen3-VL（2B & 8B）用 1.2M spatial QA fine-tune 1 epoch，bs=1024，32× H800。

**Table 3. Spatial Reasoning QA。**

| Model | MMSI-Bench | MindCube |
|---|---|---|
| VST-SFT-3B | 30.2 | 35.9 |
| Cambrian-S-3B | 25.2 | 32.5 |
| VST-SFT-7B | 32.0 | 39.7 |
| Cambrian-S-7B | 25.8 | 39.6 |
| SpaceR-SFT-7B | 27.4 | 37.9 |
| Intern3-VL-8B | 28.0 | 41.5 |
| Spatial-MLLM | 27.0 | 32.1 |
| Qwen3-VL-2B | 26.1 | 33.5 |
| Qwen3-VL-2B + Ours | 27.6 | **44.0** |
| Qwen3-VL-8B | 31.1 | 29.4 |
| Qwen3-VL-8B + Ours | **32.6** | **49.1** |

> Qwen3-VL-8B base 在 MindCube 仅 29.4 异常低（连 2B base 33.5 都不如），fine-tune 后跳到 49.1（+19.7）—— 大概率是 base model 在该 benchmark 上有某种 prompt format mismatch，fine-tune 顺便修了。要谨慎解读这个 "巨大提升"。MMSI-Bench 上 +1.5 (8B) / +1.5 (2B) 的提升才是更可信的真实信号。

**Table 4. ScanNet++ 上 3D Grounding。**

| Method | AP15 | AP25 | AP50 |
|---|---|---|---|
| VST-7B-SFT | 17.29 | 14.50 | 11.20 |
| Qwen3-VL-8B | 19.82 | 16.80 | 13.50 |
| Qwen3-VL-8B + Ours | **35.52** | **31.94** | **27.98** |

3D grounding +14.48 AP50 是这篇的硬证据——但要注意 evaluation 也在 ScanNet++ 上，和 training data 同分布，不是 OOD test。

**Figure 11. Grounding 定性对比。**

![](https://arxiv.org/html/2603.07660v1/x11.png)

### 4.3 Ablation

**Table 5. Depth refinement / confidence filter / agent recall 的消融（ScanNet++）。**

| ID | DA3 Depth | 3DGS Train | Conf. Filter | Agent Recall | P25 | R25 |
|---|---|---|---|---|---|---|
| 1 | ✓ | ✗ | ✓ | ✓ | 0.13 | 0.31 |
| 2 | ✓ | ✓ | ✓ | ✓ | **0.81** | **0.89** |
| 3 | ✓ | ✓ | ✗ | ✗ | 0.35 | 0.74 |
| 4 | ✓ | ✓ | ✓ | ✗ | 0.67 | 0.69 |
| 5 | ✓ | ✓ | ✓ | ✓ | **0.81** | **0.89** |

读法：
- **ID 1 vs 2**：3DGS 优化把 P25 从 0.13 推到 0.81（6×）— 几何精度是 OBB 质量的最大瓶颈；
- **ID 3 vs 4**：confidence filter 提 P 0.35→0.67 但伤 R 0.74→0.69；
- **ID 4 vs 5**：agent recall 把 R 拉回 0.89 同时 P 涨到 0.81 — confidence + agent 真的是互补设计，不是冗余。

**Figure 10. Scene-level refinement 各阶段可视化。**

![](https://arxiv.org/html/2603.07660v1/x10.png)

**Figure 12. Multi-view merging 消融**：(a) 3D 几何 clustering 修复 SAM3 image-level 的 over-segmentation（一张床被切成两张）；(b) GS-refined depth 避免不同物体被 false-merge。

![](https://arxiv.org/html/2603.07660v1/x12.png)

---

## 关联工作

### 基于
- **3D Gaussian Splatting**（Kerbl 2023）：场景表示底座
- **Depth-Anything-V3**：单目 depth prior 初始化
- **SAM3**：open-vocab 2D 分割
- **Qwen3-VL / Gemini3-Pro**：caption 生成 + agent verification

### 对比
- **VST / Cambrian-S / Spatial-MLLM**：2D-centric spatial LMM 的 SOTA baseline
- **SpatialLM / LLaVA-3D / SceneScript**：3D-VLM（point cloud 输入）
- **M3-Spatial / LangSplat**：3DGS-based 场景理解

### Benchmark
- **MMSI-Bench**、**MindCube**：spatial reasoning eval
- **ScanNet / ScanNet++ / DL3DV**：source data + eval

---

## 论文点评

### Strengths

1. **问题 framing 准** — 直击 spatial intelligence 数据稀缺这个 root cause，而不是又调一遍 attention 或 RL 配方。诚实地承认 "我们做的是 data engine 而不是新模型"。
2. **Pipeline 的工程完整度高** — 三阶段每一步都有清楚的失败模式分析（depth floater、SAM3 boundary error、over-segmentation）和对应解。Tri-level confidence + agent recall 这个设计简洁有效。
3. **Ablation 充分** — Table 5 直接量化每个组件对 P/R 的边际贡献，不是光秀 SOTA 数字。
4. **数据集真开源** — README 显示已 release 2K+ Gaussian 模型 + HoliSpatial-QA-2M + 模型 checkpoint，承诺 4 月 1 日前完整 release。降低复现门槛。

### Weaknesses

1. **"Raw video" 是 misnomer** — 实际输入是 ScanNet / ScanNet++ / DL3DV 这种已经精心 capture 过的 indoor scan video，不是 in-the-wild YouTube vlog。最关键的 distribution gap 测试缺失：能不能在没有相机 calib & 控制 trajectory 的 raw web video 上跑？
2. **Pipeline 重度依赖闭源 / 大模型 API** — Gemini3-Pro 做 captioning + Qwen3-VL-30B 做 instance caption + agent verify 都要 VLM call。12K 场景 × N keyframes × M instances 的 inference cost 没披露，"automation" 在 cost 上的真实代价不透明。
3. **MMSI-Bench 提升微弱** — +1.5pt 的 spatial reasoning gain 不能强支撑 "4M 数据真的提升 spatial intelligence"。MindCube +19.7 受 base model 异常低分影响，benchmark-level gain 解读要打折。
4. **Eval 同分布** — 3D grounding 评估的 ScanNet++ 也是 training scene 的来源 — in-distribution 的 +14 AP50 不能等同于 transfer 收益。缺 OOD benchmark（如 MultiScan、HM3D）。
5. **"vs SOTA" 的 unfair comparison** — 表 2 把 per-scene 优化的 Holi-Spatial 和 zero-shot inference 的 SpatialLM/LLaVA-3D 直接比 detection AP，order-of-magnitude gap 的根因部分来自 paradigm 差异而不是方法 superiority。

### 可信评估

#### Artifact 可获取性

- **代码**：仓库已开（github.com/Visionary-Laboratory/holi-spatial），README 显示已 release 模型 checkpoint；training code 状态未明确披露
- **模型权重**：HoliSpatial-2M-QA-Qwen3-VL-8B（HuggingFace 已发布）
- **训练细节**：QA fine-tune 1 epoch / bs 1024 / 32× H800 给出；data curation 各阶段 hyperparam（τ_high=0.9 / τ_low=0.8 / τ_iou=0.2）披露；但 VLM API 调用频次、3DGS 优化时长等 cost 数据未披露
- **数据集**：Holi-Spatial-4M 部分发布（2K+ Gaussian 模型 + HoliSpatial-QA-2M），README 承诺 4 月 1 日前 full release；底层 ScanNet/ScanNet++/DL3DV 都是公开数据

#### Claim 可验证性

- ✅ **"Annotation 质量超过 baseline"**：表 2 + 表 5 + 多张定性图，数据完整。
- ✅ **"Confidence + Agent 互补"**：表 5 ID 3/4/5 直接量化。
- ⚠️ **"Fine-tune 提升 spatial reasoning"**：MMSI-Bench +1.5 太小；MindCube +19.7 受 base model 异常低分污染，需要更多 benchmark 才稳。
- ⚠️ **"3D Grounding +14.48 AP50"**：grounding eval set 与训练数据同源（ScanNet++），不算真正 transfer。
- ⚠️ **"Fully automated"**：流程自动化 ✓，但依赖闭源 VLM API；如果某天 Gemini3-Pro 改 quota 或 Qwen3-VL-30B 不可用，pipeline 复现性立刻断。
- ❌ **"raw video streams"**：实际输入是已 calibrated 的室内 scan，并非真正的 raw web video。

### Notes

- 这篇可以作为 "data engine 是 spatial intelligence 短期最大杠杆" 这个 thesis 的 anchor。
- **可借鉴的设计模式**：`tri-level confidence + LLM-agent borderline arbitration`。同样的 pattern 可以套到 AutoLabel 的任何场景：自动标注 + 人/agent 仅复核中间灰区。
- **可追问的实验**：拿真正 in-the-wild 的 egocentric video（如 Ego4D、EpicKitchens）跑一遍 pipeline，看 SfM 失败率、3DGS 收敛失败率、最终 instance recall 各是多少。这是判断方法 generalization 的真正 test。
- **和我兴趣的连接**：Spatial Intelligence + VLA 方向上，"自动从 video 生成 3D-grounded instruction data" 是个明显 direction。Holi-Spatial 给出了感知侧的 recipe；可以延伸到 action 侧——能不能从 manipulation video 自动生成 OBB-grounded action label？

### Rating

**分数**：2 - Frontier
**理由**：这篇在 "data engine for spatial intelligence" 方向上是当前重要参考——问题 framing 准（Sec 1 直击数据而非架构 bottleneck）、pipeline 工程完整（tri-level confidence + agent recall 的互补设计有 Table 5 消融支撑）、artifact 已开源（2K+ Gaussian + HoliSpatial-QA-2M + checkpoint）。之所以不升到 Foundation，是因为 Weaknesses 里列出的硬伤未解：raw-video 是 misnomer（输入仍是已 calibrated 的 indoor scan）、3D Grounding eval 与训练同分布、MMSI-Bench 真实提升仅 +1.5pt。论文于 2026-03 发布、尚无大量后续 adoption 的外部信号，定位为 "值得跟进的前沿参考" 比 "方向必读" 更诚实。
