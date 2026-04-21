---
title: "Generative World Renderer"
authors: [Zheng-Hui Huang, Zhixiang Wang, Jiaming Tan, Ruihan Yu, Yidan Zhang, Bo Zheng, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang]
institutes: [Alaya Studio (Shanda AI Research Tokyo), National Taiwan University, The University of Tokyo, National Yang Ming Chiao Tung University]
date_publish: 2026-04-02
venue: arXiv
tags: [world-model, 3D-representation]
paper: https://arxiv.org/abs/2604.02329
website: https://alaya-studio.github.io/renderer/
github: https://github.com/ShandaAI/AlayaRenderer
rating: 2
date_added: 2026-04-21
---

## Summary

> [!summary] Generative World Renderer
> - **核心**: 用 ReShade + RenderDoc 在图形 API 层"非侵入式"截取 AAA 游戏（Cyberpunk 2077、Black Myth: Wukong）的 G-buffer，构建 4M 帧 720p/30FPS 的连续视频 + 5 通道 G-buffer 同步数据集，用于 fine-tune 双向 generative renderer。
> - **方法**: 双屏 mosaic 拼接录制保留高分辨率；RIFE 合成 motion blur 作为"clean + blurred"双版本；fine-tune Cosmos 版 DiffusionRenderer（inverse）和 Wan 2.1-T2V-1.3B（forward / 游戏编辑）；提出 VLM-based 排序协议（Gemini 3 Pro 作 judge）评估 in-the-wild metallic/roughness。
> - **结果**: Black Myth held-out 上 depth/normal/material 全面超过 DiffusionRenderer 与 DNF-Intrinsic；MPI-Sintel 跨数据集仍领先；VLM judge 与 25 名 CG expert 偏好一致率 60-85%；下游 G-buffer→relighting / game style editing 显著优于 ControlNet/SDEdit/DR baseline。
> - **Sources**: [paper](https://arxiv.org/abs/2604.02329) | [website](https://alaya-studio.github.io/renderer/) | [github](https://github.com/ShandaAI/AlayaRenderer)
> - **Rating**: 2 - Frontier（data-centric 证明 "AAA 游戏 G-buffer + 纯 fine-tune" 路线可行，但尚未确立为必引范式，且数据/模型获取门槛尚存）

**Key Takeaways:** 
1. **数据 > 算法**：作者明确把这视作 data-centric paper——不动 DiffusionRenderer 架构，仅 fine-tune 即取得显著增益，验证了 inverse rendering 的瓶颈是 scale + 真实感而非 architecture。
2. **AAA 游戏 = 高保真 G-buffer 工厂**：通过 ReShade hook + RenderDoc 离线分析的"非侵入"管线绕过反编译/资产抽取，规避 EULA 风险，证明可大规模、合法地从商业游戏获取 photoreal supervision。
3. **VLM-as-Judge for materials**：metallic/roughness 这类带强语义先验、传统指标无 GT 的属性，VLM (Gemini 3 Pro) 配合 grid 同步播放可作为 scalable evaluator——与 CG expert 一致率达 75-85%。

**Teaser. 数据集示意——长时序 RGB + 5 通道 G-buffer，跨场景、跨天气、跨动态。** 

![](https://arxiv.org/html/2604.02329v1/x1.png)

---

## Pipeline 与数据集构建

**Figure 3. 三阶段数据采集 pipeline。** Stage I：ReShade hook 图形 API → 用 RenderDoc 离线锁定目标 render target → deferred shading 重渲 RGB 做 pixel-level 一致性自检；Stage II：Qwen3-VL-235B 自动标注 + 质量过滤；Stage III：RIFE 合成 motion blur。

![](https://arxiv.org/html/2604.02329v1/x3.png)

### G-buffer 截取（Sec 3.1）

核心难点：**现代 AAA 游戏的 G-buffer packing 因引擎/作品而异，没有标准 layout**。作者的解法是工程组合拳：

- **离线分析 + 运行时 hook**：先用 RenderDoc 抓一帧，离线识别候选 render pass 与其 attachment（format、extent、sample count），再写 game-specific ReShade add-on hook 图形 API callback，运行时只 GPU-copy 满足"格式稳定 + extent 稳定 + 反复绑定"等不变量的 render target。
- **Camera-space normal 重建**：游戏吐出的是 world-space normal，但缺 view matrix 转换 → 改从 depth 反投影 + 有限差分得到 view-space position $\mathbf{P}$，再叉乘求法向：

$$
\mathbf{n}=\operatorname{normalize}\!\left(\frac{\partial\mathbf{P}}{\partial x}\times\frac{\partial\mathbf{P}}{\partial y}\right)
$$

- **材质通道解耦**：metallic / roughness 常打包在同一 render target 的不同通道，screen-capture 视频压缩会引起 inter-channel bleeding → 把每个通道渲到屏幕上**空间分离的不同区域**避免串扰。

### 双屏 mosaic 同步录制（Sec 3.2）

直接 dump 多通道 G-buffer 每帧的 GPU→CPU readback 会卡爆。改成把所有 6 个通道（RGB + 5 G-buffer）shade 到屏幕，OBS 近无损录制——为了不被单屏分辨率限制，**拼两块 2K 屏**做 mosaic，每个通道获得有效 720p。游戏因显示扩大会自动加大 FOV，作者对 source buffer 先做 center-crop 再 tile，保留原始视野比例。

> ❓ 把 G-buffer 渲到屏幕用视频编码"代替"原始 buffer dump，本质是 lossy 压缩。作者只做了"deferred shading 反推 RGB 一致性"的总体校验，但材质通道的精度损失对下游 fine-tune 的 ceiling 影响有多大，没有定量分析。

### Motion blur 合成（Sec 3.5）

RGB 用引擎自带 motion blur 关闭后录制（保 sharp），离线再用 RIFE 插 8 个 sub-frame 在线性域平均得到 blurred 版本：

$$
I^{\text{blur}}_{t}=\mathrm{RGB}\!\Big(\tfrac{1}{K}\sum_{i=1}^{K}\mathrm{Lin}\!\big(\tilde{I}_{t,i}\big)\Big)
$$

发布"clean + blurred"双版本，对应"实验室干净观测"与"真实手持模糊"两种下游域。

### 数据集统计

- 4M+ 帧、720p、30 FPS、6 同步通道（RGB + depth + normal + albedo + metallic + roughness）
- 40 hours 来自 Cyberpunk 2077 + Black Myth: Wukong；平均单段 8 min，最长 53 min 连续
- 自动标注 4 个 categorical 属性：texture（材质类别）、weather（晴/阴/雾/雨/雪）、scene（室内/室外）、motion（相机/场景动静四宫格）
- 分布特征：Cyberpunk 富金属高 metallic 像素，Wukong 富自然高 roughness；Wukong 整体亮度更低（户外遮挡多）

## VLM-based 真实场景评估（Sec 4）

**问题**：真实视频没有 G-buffer GT，传统 PSNR/LPIPS 不适用；用户调研对 metallic/roughness 这类需要 CG 专家判断的属性不可扩展。

**方法**：VLM 编码了大量材质常识，可作 pairwise comparison 的 scalable judge。具体只评 metallic 和 roughness（这两者有强语义先验：金属表面、特征性高光），用 Gemini 3 Pro 看一个 RGB reference + 各方法输出的 fixed-layout 同步播放 grid，输出三维分数：

- **Sem.** — 语义合理性
- **App.** — 空间外观质量
- **Temp.** — 时序一致性

**Table 3. VLM 评估（数值越低越好，即 ranking 更靠前）：**

| Channel | Method                | Sem. ↓ | App. ↓ | Temp. ↓ |
| ------- | --------------------- | ------ | ------ | ------- |
| R       | DiffusionRenderer     | 2.45   | 2.40   | 2.10    |
| R       | Ours                  | 1.78   | 1.78   | 2.08    |
| R       | Ours (w/ motion blur) | 1.78   | 1.83   | **1.83**    |
| M       | DiffusionRenderer     | 2.35   | 2.28   | 2.00    |
| M       | Ours                  | 1.90   | 2.13   | 2.15    |
| M       | Ours (w/ motion blur) | **1.75**   | **1.60**   | **1.85**    |

motion blur 增强让所有 6 个 cell 的 Temp. 都改善，且 metallic 的 Sem./App. 也最佳——说明训练数据加入 blur variant 不仅提鲁棒性还反过来帮材质判别。

**用户研究**：25 名 CG expert 做 pairwise；VLM 偏好 our model 的样本上人类同意率 75% (R) / 85% (M)；VLM 偏好 DiffusionRenderer 的样本上 61% (R) / 70% (M)。整体 VLM↔人类判断一致，roughness 方向因线索更模糊一致率略低。

## 实验

### 实验设置（Sec 5.1）

- **Inverse renderer**：用 Cosmos 版 DiffusionRenderer 的 pre-trained weight 全量 fine-tune，57 帧 / 24 FPS / 1280×720 clip。Cyberpunk 训练，Black Myth 测试。最终选 motion-augmented 变体；额外训了 113 帧的 long-clip 变体，长视频推理显著更稳。
- **Forward renderer / Game editing**：基于 Wan 2.1-T2V-1.3B 改造，把 G-buffer 加入 conditional input；用 Qwen3-VL 给每段视频生成"只描述光照与环境效果"的 caption（因为 G-buffer 已经管几何/材质），让用户文本控制 lighting/style。832×480 / 16 FPS / 81 帧训练。

### Inverse rendering 定量结果

**Table 1. Black Myth held-out 39 段 × 57 帧。** "Ours" 在 depth 全四项、normal 全两项、albedo 的 si-PSNR/si-LPIPS、metallic 与 roughness 的 RMSE/MAE 上都是最佳；albedo 的非尺度不变 PSNR/LPIPS 略逊于 DR（被全局 intensity scaling 影响）。

| Method | Depth Abs Rel ↓ | Depth δ<1.25 ↑ | Normal Acc@11.25° ↑ | Albedo si-PSNR ↑ | Metallic RMSE ↓ | Roughness RMSE ↓ |
| ------ | --------------- | -------------- | ------------------- | ---------------- | --------------- | ---------------- |
| RGB↔X  | -               | -              | 0.035               | 20.11            | 0.510           | 0.349            |
| DNF    | 0.862           | 0.361          | 0.065               | 15.59            | 0.245           | 0.566            |
| DR     | 1.118           | 0.267          | 0.110               | 19.90            | 0.230           | 0.281            |
| **Ours**   | **0.697**           | **0.609**          | **0.150**               | **21.44**            | **0.104**           | **0.266**            |

metallic RMSE 从 0.230 → 0.104（−55%）是单项最大增益，呼应"我们数据 metallic-rich 像素覆盖好"的论点。

**Table 2. Sintel 跨数据集 final pass。** 即使迁到带 motion blur / DOF 的合成动画，仍在 depth (RMSE 0.220 vs DR 0.268) 和 albedo (PSNR 15.40 vs 14.87) 上领先——data 增益跨域可迁移，不是 cherry-picked。

### Ablation：motion blur

**Table 5. Sintel 上的 motion blur 消融。** 加 blur 增强让 RMSE log 从 0.773 → 0.745，δ<1.25³ 从 0.756 → 0.776，si-PSNR 从 17.37 → 17.80。Albedo PSNR 略降 (15.73 → 15.40)，但 LPIPS / si-LPIPS 都改善——motion blur 的训练信号让模型学到 blur-invariant 的几何/材质表示。

### 定性结果

**Figure 4. Real-world 视频上 inverse rendering 对比（top→bottom: albedo, normal, depth, metallic, roughness）。** "Ours" albedo 更干净（delighting 更彻底）、几何更无 artifact、metallic/roughness 在烟雾等大气干扰下仍稳定。

![](https://arxiv.org/html/2604.02329v1/x4.png)

### Relighting（Sec 5.5）

冻结 DiffusionRenderer 原 forward renderer，只换 G-buffer 输入：用 baseline DR 抽的 G-buffer vs 用 our fine-tuned DR 抽的 G-buffer，喂同样的 environment map。结论：天空区域 baseline 经常翻车，our G-buffer 给出与目标光照一致的合理结果——证明**单纯升级 inverse renderer 的输入质量，无需重训 forward renderer 就能改善下游**。

**Figure 6. Relighting 应用。**

![](https://arxiv.org/html/2604.02329v1/x6.png)

### Game editing（Sec 5.6）

把 G-buffer 当做条件输入到 Wan 2.1，用文本调风格/天气/光效。对比三类 baseline：(i) 用 RGB 边缘的 ControlNet——边缘从 RGB 抽噪声大，时序闪烁；(ii) SDEdit——偏离原图过多，关键小物体经常消失；(iii) DiffusionRenderer + DiffusionLight 抽 env-map——对激进风格变换无能为力，且 env-map 工作流不友好。

**Video. Inverse rendering 真实视频结果——albedo 通道方法对比。**

<video src="https://alaya-studio.github.io/renderer/static/videos/inverse/outdoor_albedo_merged.mp4" controls muted playsinline width="720"></video>

**Figure 7. Game editing 应用：用 G-buffer 作条件，用文本控制 lighting/weather/visual effect。**

![](https://arxiv.org/html/2604.02329v1/x7.png)

---

## 关联工作

### 基于
- Cosmos：Cosmos-Transfer1-DiffusionRenderer 7B 是 inverse renderer 的 base model，本文不改架构，只换 fine-tune 数据。
- DiffusionRenderer (Liang et al. 2025)：本文的核心 baseline 与 architectural backbone，inverse + forward 双向架构均沿用。
- Wan 2.1-T2V-1.3B：forward renderer (game editing) 的 base，加 G-buffer 作为 conditional input fine-tune。
- Qwen3-VL-235B-A22B-Instruct：用于 dataset annotation（4 categorical attributes）+ caption generation。
- RIFE：用于合成 motion blur 的帧插值器。

### 对比
- DNF-Intrinsic：image-based diffusion inverse renderer，作 baseline 验证 video-based 优势。
- DiffusionRenderer (Cosmos / SVD 变体)：currently the only public video inverse renderer，主要对照。
- ControlNet (edge-conditioned)、SDEdit、DR + DiffusionLight：作 game editing 的三个 baseline 范式。

### 方法相关
- ReShade + RenderDoc：图形 API hook + 离线 frame 分析的"非侵入"游戏数据采集 toolchain。
- GTA-V dataset / VIPER：早期"从游戏抽 CV 数据"先例，本文将其框架扩展到长视频 + 多通道 G-buffer。
- DiffusionLight：从 RGB 估 environment map，作 forward renderer 的对照用光照源。
- Gemini 3 Pro：作 VLM judge model（video understanding + temporal reasoning 强）。

---

## 论文点评

### Strengths

1. **诚实的 data-centric 立场**：架构不动、纯换数据 fine-tune，把"the bottleneck is data"这个 hypothesis 用 controlled experiment 直接证明——这是这类 paper 应有的科学态度。
2. **工程难度被严肃对待**：dual-screen mosaic、ReShade hook + RenderDoc 反向工程定位 G-buffer、material 通道空间解耦——任何一个细节做不好整个 pipeline 都崩。EULA-aware 的 API-level 截取设计也很专业。
3. **VLM-as-judge 在 metallic/roughness 这种"无 GT 但有强语义先验"任务上找到了合适落点**：用户研究的 60-85% agreement 说明它在该 niche 是可信的。
4. **跨数据集泛化（Sintel）+ 下游任务（relighting / editing）双重验证**：增益不是 in-domain overfit。

### Weaknesses

1. **数据 lossy 的 ceiling 未量化**：把 G-buffer shade 到屏幕用 OBS 录制的链路是 lossy 的，对最终模型精度的上限影响没有 ablation。理论上 GPU readback 一帧 GT 与 mosaic 录制的同帧之间应可做对比。
2. **只测 metallic/roughness 的 VLM judge**：作者承认其他通道（normal、depth）VLM 先验弱所以不评——但这恰好让 VLM 评估的适用边界很窄，远未达到"通用 inverse rendering judge"。
3. **训练数据只来自 2 个 game**：Cyberpunk 偏未来都市、Wukong 偏自然奇幻，对真实物理世界的"日常"分布（家居、办公室、街景行人车流）覆盖仍有缺口。Real-world 40 段评估也是从网上抓的。
4. **没有公开下载链路**：dataset 是 gated access + ToU，复现门槛高；model checkpoint 已上 HuggingFace 部分缓解。
5. **VLM judge 的 reliability 隐患**：当 VLM 倾向 DiffusionRenderer 的样本上人类一致率显著下降（61% R / 70% M）——表明 VLM 在"反方向"判断时不那么可靠，可能存在"偏好新模型/某类风格"的 bias，paper 没深究。

### 可信评估

#### Artifact 可获取性

- **代码**：inference + 数据 curation toolkit 开源（[github](https://github.com/ShandaAI/AlayaRenderer)）；training 代码未明确说明。
- **模型权重**：已发布 — `Brian9999/world_inverse_renderer` (基于 Cosmos-Transfer1-DiffusionRenderer 7B 的 inverse renderer) 和 `Brian9999/stylerenderer` (基于 Wan 2.1 1.3B 的 game editing)；HuggingFace Space `Brian9999/game-editing` 提供 live demo。
- **训练细节**：仅高层描述（57/113 帧 clip、24 FPS、1280×720、Wan 训练 832×480/16 FPS/81 帧）；具体超参、batch size、learning rate、训练步数未披露。
- **数据集**：gated access + CC BY-NC-SA 4.0，需签 ToU 申请；toolkit 开源以便用户从其他游戏自行采集。

#### Claim 可验证性

- ✅ "Fine-tune on our data improves DR on Black Myth held-out"：Table 1 实验，metallic RMSE −55% 显著。
- ✅ "Sintel 跨数据集仍领先"：Table 2，外部 benchmark + 公开数据。
- ✅ "VLM judge 与 CG expert 高一致率"：25 人用户研究 + 报出的 disagreement case，方法学透明。
- ⚠️ "Forward renderer 实现 game style editing 优于 baseline"：仅 qualitative + 内部 baseline，无量化用户研究。
- ⚠️ "Long-clip variant 在长视频显著更好"：只在 Figure 2 定性展示，无 long-video 量化指标。
- ⚠️ "VLM-based eval 对 metallic/roughness 之外的通道也可推广"：作者明确不评 normal/depth/albedo，但 abstract 与 contribution 里"semantic, spatial, temporal" 三轴的措辞会让人误以为通用——实际只对两通道验证过。

### Notes

- 这篇 paper 给我的核心 take-away 是 **"用 AAA 游戏作为 photorealistic supervision 工厂"** 的范式正在变可行——之前 GTA-V dataset 只能拿到 RGB + 简单 segmentation，现在通过 ReShade hook 能拿到完整 G-buffer。如果未来能拓展到第一人称 motion 数据（比如 Black Myth 的角色控制 trajectory），对 spatial reasoning / world model 训练都会很有意思。
- World model 圈的"用游戏 frame 做 photorealistic action-conditioned 数据"路线和这篇的"用游戏 G-buffer 做 inverse rendering 监督"是互补的——前者关心 action→frame 的因果，后者关心 frame→intrinsic decomposition。两者结合可能产生 action-conditioned material-aware world model。
- VLM-as-judge 的设计很值得借鉴到我们自己的 spatial reasoning eval：我们也面临"GT 难拿但有语义先验"的问题。但要警惕作者发现的 disagreement bias——VLM 在偏好"新方法"时可能更准、偏好"旧方法"时一致率掉，需要 controlled audit。
- 数据集 gated access 是合理的法律 / 道德选择，但意味着真复现仍要走 toolkit 自采。toolkit 是否会维护到能跑通其他游戏（不仅是 Cyberpunk + Wukong），决定这条路的长期可达性。
- ❓ Cosmos 系的 DiffusionRenderer 7B fine-tune 一次的 compute 成本？paper 没披露 — 这是判断"data-centric paradigm 普及门槛"的关键变量。

### Rating

**分数**：2 - Frontier
**理由**：作为 data-centric 研究路线的代表工作，它用严格的 controlled experiment（架构不动、纯换数据）证明了 "AAA 游戏 G-buffer 作为 photorealistic inverse rendering 监督" 这条路线可行，且在 Black Myth held-out (metallic RMSE −55%) 与 Sintel 跨数据集上都显著领先 DiffusionRenderer/DNF-Intrinsic——足以成为当前 video inverse rendering 的 frontier 参考与必比 baseline。但不到 Foundation：数据 gated access、只覆盖 2 个 game、VLM judge 适用边界仅限 metallic/roughness，尚未成为方向的 de facto 奠基范式；而相较 Archived，它的数据范式 + VLM-as-judge 设计对下一步工作（包括我们自己的 spatial reasoning eval）有直接可借鉴性，不是一次性参考。
