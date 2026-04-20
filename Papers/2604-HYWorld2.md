---
title: "HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds"
authors: [Tencent Hunyuan]
institutes: [Tencent Hunyuan]
date_publish: 2026-04
venue: arXiv
tags: [world-model, 3D-representation]
paper: https://arxiv.org/abs/2604.14268
website: https://3d-models.hunyuan.tencent.com/world/
github: https://github.com/Tencent-Hunyuan/HY-World-2.0
rating: 2
date_added: 2026-04-20
---
## Summary

> [!summary] HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds
> - **核心**: 开源的多模态 offline 3D world model，统一"生成"（text/single-view → navigable 3DGS）和"重建"（multi-view/video → 3D 几何），四阶段 pipeline: Panorama → Trajectory → Expansion → Composition
> - **方法**: HY-Pano 2.0（MMDiT panorama）+ WorldNav（5 种 heuristic trajectory）+ WorldStereo 2.0（keyframe-latent VDM + GGM/SSM++ memory + DMD 4-step 蒸馏）+ WorldMirror 2.0（feed-forward 几何预测，normalized RoPE + depth-to-normal）+ 3DGS with MaskGaussian
> - **结果**: 开源 SOTA，一张场景端到端 ~10 分钟（712s on H20）；claim 和闭源 Marble 相当；WorldMirror 2.0 在 7-Scenes/NRGBD/DTU 全面超 VGGT / π³；单场景 ~1.38M Gaussian
> - **Sources**: [paper](https://arxiv.org/abs/2604.14268) | [website](https://3d-models.hunyuan.tencent.com/world/) | [github](https://github.com/Tencent-Hunyuan/HY-World-2.0)

**Key Takeaways:**
1. **"生成"和"重建"的统一靠 WorldMirror 2.0 这座桥**: 生成 pipeline 在最后一步用重建模型把生成的 keyframes 转回 3D 点云来初始化 3DGS，两个任务共享一个 feed-forward 几何 backbone
2. **Keyframe-latent VDM 是核心新意**: 放弃 Video-VAE 的时空压缩，改成稀疏 keyframe 上的 spatial-only VAE，用更大采样间隔换同等视角覆盖，但保留高频细节（尤其快速相机运动下）
3. **Memory 机制是工程重点**: GGM（全局点云作 3D prior）+ SSM++（retrieval-based 空间拼接 + RoPE 修改），把原 WorldStereo 的 two-branch 简化进 DiT 主干
4. **WorldMirror 2.0 的 normalized RoPE 解决 multi-resolution**: 把位置编码从 `[0, H_p-1]` 改成 `[-1,1]`，高分辨率推理从 extrapolation 变成 interpolation，AUC@30 在 high-res 从 66.29 拉回 86.89
5. **和 Marble 的对比主要靠"fidelity to input"**: 论文的核心卖点是生成结果严格 adhere to 输入 panorama/image，Marble 倾向 drift；但只有定性图、没有定量与 Marble 的对比

**Teaser. 框架总览：从 text/image/multi-view 输入统一到 3DGS 世界。**

![](https://arxiv.org/html/2604.14268v1/x1.png)

---

## 1. Motivation 与 Positioning

HY-World 2.0 定位是 HY-World 1.0（offline 3D）+ HY-World 1.5（online video-based）之后的一次 paradigm consolidation。作者的 framing：

- 当前 3D world modeling 生态 bifurcated——生成类（FlashWorld、Lyra、WonderJourney 等）擅长 hallucinate 未见区域，但几何精度不够；重建类（VGGT、π³、DepthAnything3、MapAnything 等）擅长 recover 几何，但无生成 prior
- 闭源的 Marble (World Labs) 已经展示统一范式，但开源社区缺一个 comprehensive multi-modal foundational world model
- HY-World 2.0 明确选择 **offline 3DGS 路线**（区别于 Genie 3 这类 online/autoregressive video generation），以便兼容标准 CG pipeline

> ❓ "offline 3DGS 路线" vs "online interactive video" 是两条根本不同的路径：前者把世界冻结为 static 3DGS，交互只是 camera control；后者（Genie 3）支持事件级干预和动态响应。HY-World 2.0 选择前者，**本质上不是真正的"world model"**（没有 dynamics），而是"navigable 3D scene generator"。后面点评会重点讨论。

---

## 2. 四阶段 Pipeline 总览

![](https://arxiv.org/html/2604.14268v1/x2.png)
**Figure 2. 四阶段架构：Panorama Generation → Trajectory Planning → World Expansion → World Composition。**

流水线：
1. **Stage I — Panorama Generation (HY-Pano 2.0)**: 文本或单图 → 360° ERP panorama
2. **Stage II — Trajectory Planning (WorldNav)**: 解析 panorama 的几何+语义，生成 5 类 collision-free 相机轨迹
3. **Stage III — World Expansion (WorldStereo 2.0)**: 沿轨迹用 camera-guided VDM 生成 novel-view keyframes
4. **Stage IV — World Composition**: WorldMirror 2.0 把 keyframes 重建为点云，对齐到 panorama 全局坐标系，训 3DGS

---

## 3. HY-Pano 2.0：Panorama Generation

### 关键设计
- **放弃显式几何 warping**（HY-World 1.0 需要 camera intrinsics），改用 MMDiT，把 perspective input latent 和 panorama noise latent 拼成统一 token 序列，让 self-attention 自己学 perspective→ERP 映射
- **边界 seam 处理**：latent 级 circular padding + 像素级 linear blending，消除 ERP 左右边界不连续

![](https://arxiv.org/html/2604.14268v1/x3.png)
**Figure 3. HY-Pano 2.0 架构：MMDiT + circular padding + pixel blending。**

### 数据
- Real-world panorama（scale 提升）+ 合成（UE 渲染）。严格过滤 stitching artifacts 和设备露出的样本

### 结果
**Table 4. T2P / I2P 对比（摘要）：HY-Pano 2.0 几乎全面领先 DiT360 / Matrix3D / HY-World 1.0 / CubeDiff / GenEx。**

| 指标 | T2P HY-World 1.0 | T2P HY-Pano 2.0 | I2P HY-World 1.0 | I2P HY-Pano 2.0 |
| --- | --- | --- | --- | --- |
| CLIP-T/I ↑ | 0.250 | **0.258** | 0.831 | **0.844** |
| Q-Align Qual (Persp) ↑ | 3.992 | **4.103** | 3.317 | **4.026** |
| Q-Align Aes (Persp) ↑ | **3.404** | 3.376 | 2.638 | **3.208** |

> ❓ T2P 上 Q-Align Qual (Equi) 是 4.403 < HY-World 1.0 的 4.493——被 1.0 反超了。作者没 highlight 这点。

---

## 4. WorldNav：Trajectory Planning

纯 heuristic，**没有 learned planner**。先做 scene parsing：
- MoGe2 多视角深度对齐（12 → 42 views, GPU-accelerated LSMR）得到 panoramic point cloud
- Qwen3-VL + SAM3 做语义 grounding
- Recast Navigation 构建 NavMesh

![](https://arxiv.org/html/2604.14268v1/x5.png)
**Figure 5. 五类 heuristic trajectory: Regular / Surrounding / Reconstruct-Aware / Wandering / Aerial。**

**Table 1. 最多 35 条轨迹 per scene。**

| | Regular | Surrounding | Recon-Aware | Wandering | Aerial | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Max Number | 9 | 5 | 10 | 3 | 8 | 35 |

**Reconstruct-Aware** 值得注意：检测 panoramic mesh 中 aspect ratio 过大的"拉伸面"作为欠观测区域，专门规划轨迹去补这些 hole。是几何驱动的 active view selection。

> ❓ 全部轨迹是 heuristic 启发式，这部分天花板很明显——把 "trajectory planning" 做成可学习的策略（结合场景先验和 reconstruction uncertainty）应该能显著更好。现在这个版本像传统 CV pipeline 的模块化组合。

---

## 5. WorldStereo 2.0：World Expansion

核心 IP 所在。训练分三阶段：domain-adaption → middle-training (memory) → post-distillation。

![](https://arxiv.org/html/2604.14268v1/x7.png)
**Figure 7. WorldStereo 2.0 pipeline：主 DiT 分支 + SSM++ retrieval stitching，控制分支以 panoramic point cloud 作 GGM。**

### 5.1 Keyframe-VAE：最关键的设计决策

**问题**：camera-guided VDM 用标准 Video-VAE 做时空压缩，快 camera motion 下产生 motion blur / geometric distortion。

**方案**：不做时间压缩，每帧独立按 image 处理（spatial-only VAE）。采样间隔加大以保持相同 viewpoint coverage。

![](https://arxiv.org/html/2604.14268v1/x9.png)
**Figure 9. Keyframe-VAE vs Video-VAE：前者保留高频细节。**

**Table 7. VAE + Frozen Parts 的 ablation（user study 占主导）：**

| Frozen | VAE | RotErr ↓ | TransErr ↓ | Camera 偏好 ↑ | Quality 偏好 ↑ |
| --- | --- | --- | --- | --- | --- |
| Main DiT | Video-VAE (baseline) | 0.762 | 1.245 | 84.85% | 46.46% |
| None | Keyframe-VAE | 0.578 | 1.115 | 93.81% | 60.61% |
| **Cross-Attn + FFN** | **Keyframe-VAE (final)** | **0.492** | **0.968** | 92.44% | **64.39%** |

冻结 cross-attn + FFN 最佳 trade-off：既吸收 keyframe latent 又防 style drift。

### 5.2 Memory Mechanisms

**GGM (Global-Geometric Memory)**：额外从 $T_g=2$ novel view 采样点云扩展为 $\mathbf{P}^{glo}=[\mathbf{P}^{ref}, \hat{\mathbf{P}}]$，提供全局 3D prior。强迫 VDM 真正"看"点云结构而不是把它当 soft hint。

**SSM++ (Improved Spatial-Stereo Memory)**：相比 WorldStereo 1.0，
1. 丢掉独立 memory 分支，retrieved keyframes 直接进主 DiT
2. 修改 RoPE 让 retrieved view 继承 target frame 的 temporal index（横向拼接 2W）
3. 选择性检索（不是每帧都 retrieve）
4. 显式 pointmap guidance → 隐式 camera embedding (7-dim quaternion+translation, 3-layer MLP, zero-init)

![](https://arxiv.org/html/2604.14268v1/x11.png)
**Figure 11. RoPE 修改：retrieved view 继承 target frame 的 temporal index。**

**数据配对**：
- Multi-view 真实数据 → temporally misaligned retrieval（30%-90% 时序重叠）
- UE 合成数据 → multi-trajectory retrieval（跨轨迹按 3D FoV similarity 选）

**Table 8. Memory + Distillation ablation：**

| Config | PSNR ↑ | PSNR_m ↑ | RotErr ↓ | ATE ↓ |
| --- | --- | --- | --- | --- |
| Baseline (camera control only) | 16.13 | 28.81 | 0.396 | 0.071 |
| + GGM + SSM++ | 20.94 | 30.27 | 0.407 | 0.046 |
| + Trainable FFN | 21.56 | 30.44 | 0.351 | 0.035 |
| Temporal-concat SSM (ablation ✗) | 19.83 | 29.77 | 0.545 | 0.114 |
| **Final (A-F)** | **21.63** | **30.76** | 0.296 | 0.041 |
| **After DMD distillation (G)** | **21.84** | **30.93** | 0.316 | 0.072 |

**关键发现**：SSM 的 spatial 拼接 vs 时间拼接，差距极大（PSNR 21.63 vs 19.83，RotErr 0.296 vs 0.545）——spatial stereo 拼接是核心设计。

### 5.3 DMD Post-Distillation

基于 Distribution Matching Distillation，4-step 学生。跟 WorldStereo 1.0 只蒸馏 camera control 不同，这里 memory 也一起蒸馏（得益于 SSM++ 不需要显式 depth 对齐）。

### 5.4 Scene Reconstruction 结果

**Table 5. Single-view scene reconstruction (Tanks-and-Temples + MipNeRF360):**

| Method | T&T F1 ↑ | T&T AUC ↑ | MipNeRF360 F1 ↑ | MipNeRF360 AUC ↑ |
| --- | --- | --- | --- | --- |
| SEVA | 36.73 | 51.03 | 28.75 | 46.81 |
| Gen3C | 31.24 | 42.44 | 35.26 | 52.10 |
| Lyra | 32.54 | 43.05 | 36.05 | 49.89 |
| FlashWorld | 22.29 | 30.45 | 42.60 | 53.86 |
| **WorldStereo 2.0** | **41.43** | **58.19** | **51.27** | **65.79** |
| WorldStereo 2.0 (DMD) | **43.16** | **60.09** | 50.52 | 65.64 |

显著超过所有 open-source 竞品。

---

## 6. WorldMirror 2.0：Feed-Forward Reconstruction

扮演"桥梁"角色——既做独立 reconstruction benchmark，又作为 Stage IV 的几何 extractor。

![](https://arxiv.org/html/2604.14268v1/x12.png)
**Figure 12. WorldMirror 2.0 架构：Any-Modal Tokenization + 共享 Transformer + 多 DPT decoder heads。**

### 6.1 三大 Model 改进

**1. Normalized RoPE**：把 patch grid 从 `[0, H_p-1]` 整数索引 → `[-1,1]` 归一化坐标。

$$
\hat{x}_{i}=\frac{2i+1}{H_{p}}-1,\quad \hat{y}_{j}=\frac{2j+1}{W_{p}}-1
$$

把 resolution extrapolation 变成 interpolation。Figure 13 显示 cross-resolution cosine similarity > 0.95（标准 RoPE 大幅 degrade）。

**2. Depth-to-Normal Loss**：显式耦合 depth 和 normal。

$$
\tilde{\mathbf{N}}_{i}(x)=\operatorname{normalize}\left(\frac{\partial\mathbf{P}_{i}}{\partial u}\times\frac{\partial\mathbf{P}_{i}}{\partial v}\right)
$$

$$
\mathcal{L}_{\text{d2n}}=\frac{1}{|\mathcal{V}|}\sum_{x\in\mathcal{V}}\arccos\left(\frac{\tilde{\mathbf{N}}_{i}(x)\cdot\hat{\mathbf{N}}_{i}(x)}{\|\tilde{\mathbf{N}}_{i}(x)\|\|\hat{\mathbf{N}}_{i}(x)\|}\right)
$$

对 real-world dataset 用 monocular normal teacher 作 pseudo target，绕过 depth pseudo-label 的 multi-view inconsistency（pointcloud layering artifacts）。

**3. Depth Mask Prediction Head**：显式 per-pixel validity（BCE loss），替代原 1.0 只用 confidence weighting。

### 6.2 Data / Training / Inference 改进

- Data：加入 UE 合成 + normal pseudo labels（而非 depth pseudo labels，后者 multi-view 不一致）
- Training：Token-budget dynamic batch（固定 $T_{\max}=25000$ tokens/GPU，views 反算）+ 三阶段 curriculum
- Inference：Token/Frame SP + BF16 + FSDP，256-view inference 在 4 GPU 17.52s（baseline OOM）

### 6.3 重建 Benchmarks

**Table 11/12/13 摘要（点云、相机、深度、normal、NVS）：**

| Metric | Best Baseline | WM 1.0 (M) | WM 2.0 (H) |
| --- | --- | --- | --- |
| 7-Scenes Acc ↓ | π³ 0.048 | 0.043 | **0.037** |
| 7-Scenes + all priors Acc ↓ | - | 0.018 (M) | **0.012** |
| RealEstate10K Cam AUC@30 ↑ | π³ 85.90 | 86.13 | **86.89** |
| WM 1.0 (H) Cam AUC@30 | - | **66.29** ⚠️ | - |
| ScanNet Normal mean ↓ | DSine 16.2 | 13.8 | **12.5** |

**关键**：WorldMirror 1.0 在 high-res 严重崩溃（AUC 86.13 → 66.29，normal error 13.8 → 17.6），2.0 彻底解决。

---

## 7. World Composition (Stage IV)

把 generated keyframes 转成 3DGS 的最后一步。

### 7.1 Depth Alignment

WorldMirror 2.0 预测的 depth 有 scale ambiguity，需对齐到 panoramic point cloud。方案：

1. 渲染 $\mathbf{P}^{pan}$ 到每个 view 作 sparse guidance
2. 交集 mask $\mathbf{M}_i = \mathbf{M}^m \cap \mathbf{M}^g \cap \mathbf{M}^n \cap \mathbf{M}^p \cap \overline{\mathbf{M}}^{sky}$（confidence、normal consistency、percentile outlier、sky mask 全部 AND）
3. RANSAC 估 per-frame linear scale+shift $\gamma_i, \beta_i$
4. 全局 outlier detection：9 个 anchor depth 算 transformed value 的 median，偏差超 90th percentile 的 coefficient 被替换成最近 inlier

![](https://arxiv.org/html/2604.14268v1/x14.png)
**Figure 14. Depth alignment pipeline。**

### 7.2 3DGS with MaskGaussian

**Growth Strategy 两难**：不densify → sky 区 Gaussian 过多拖慢渲染；densify → 天空区产生 floater。

**解法**：
1. 把点云分成 sky / scene 子集，**只对 scene 子集 apply densification**
2. 引入 MaskGaussian：每个 Gaussian 有 Gumbel-Softmax 采样的 binary mask $M_k$，rendering 时 $\mathbf{c}(\mathbf{x})=\sum_{k}M_{k}\mathbf{c}_{k}\sigma_{k}T_{k}$，配合 squared loss 稀疏化

**Table 9. 3DGS ablation:**

| Voxel | Densify | MaskGauss | GS # | PSNR ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- |
| | | | 6.000M | **25.176** | **0.209** |
| ✓ | | | 1.000M | 24.504 | 0.276 |
| ✓ | ✓ | | 5.254M | 25.158 | 0.210 |
| ✓ | ✓ | ✓ | 1.383M | 25.017 | 0.216 |
| **✓** | **✓†** | **✓** | **1.381M** | **25.023** | **0.215** |

以 77% Gaussian 数量换 -0.15 dB PSNR——很合理的 efficiency/quality trade-off。

### 7.3 Optimization

Photometric (L1 + SSIM + LPIPS) + Depth L1 (稀疏) + Normal cosine (密集，MoGe2 提供的 alignment-free normal) + scale regularization + mask loss。

弃用 Spherical Harmonics 改 view-independent RGB（generated scenes 没有明显 view-dependent effects）。

---

## 8. 整体结果与 Marble 对比

![](https://arxiv.org/html/2604.14268v1/x21.png)
**Figure 21. 整体 pipeline 输出：panorama、点云、3DGS、mesh、novel views。**

**Table 10. Runtime (H20 GPU)：**

| Stage | Panorama | Trajectory | Expansion | Recon+Align | 3DGS | **Total** |
| --- | --- | --- | --- | --- | --- | --- |
| Time | 15s | 182s | 286s | 102s | 127s | **712s** |

端到端 ~12 分钟 per scene。

**与 Marble 的对比（Figure 23/24）**：**纯定性**。作者 claim：
- 更严格 adhere to 输入 panorama / image（Marble drift）
- 更好的 detail preservation 和 geometric consistency

![](https://arxiv.org/html/2604.14268v1/x22.png)
**Figure 22. 交互探索：用 extracted mesh 作 collision proxy，支持 real-time 物理反馈和 character navigation（楼梯、室内布局）。**

---

## 论文点评

### Strengths

1. **系统级的完整度**：一套从 text/image 到可交互 3DGS 的端到端开源 pipeline，代码 + 权重 + 技术细节全部释放，对开源社区是实打实的贡献。runtime 712s 已经到 practical 门槛
2. **Keyframe-VAE 是真 insight**：快 camera motion 下 Video-VAE 的时空压缩造成重建灾难，改成 spatial-only + 稀疏采样是干净的修复。不是新模型堆料，是对现有 paradigm 的反思
3. **WorldMirror 2.0 的 normalized RoPE 干净利落**：识别到 multi-resolution extrapolation → interpolation 的本质，一行公式修复。cross-resolution cosine sim > 0.95 的验证曲线很漂亮。此外 7-Scenes / NRGBD / DTU / 深度 / normal benchmark 基本全面刷新 open-source SOTA，实打实的贡献
4. **Ablation 扎实**：Tab 7/8/9/11 都给了足够细的 configuration 对比，SSM 的 spatial vs temporal concatenation 差距 PSNR 1.8dB，立即说服
5. **MaskGaussian 的工程价值**：77% Gaussian 数量削减只掉 0.15 dB PSNR，配合 sky/scene 分离 densification，解决 generated 数据训 3DGS 的 floater 顽疾

### Weaknesses

1. **"World model" 的命名严重虚胖**。这不是 world model，是 **panorama-conditioned 3D scene generator**。没有 dynamics、没有 action-conditioning、没有 physics simulation，场景一旦生成就冻结成静态 3DGS。把 "collision detection + character navigation" 当 "Simulating 3D Worlds" 写进 title，和 Genie 3 / WorldLabs Marble 的 interactive world modeling 定义完全不在一个层次
2. **和 Marble 的对比没定量数据**。整个 Sec 8.1.5 就两张定性对比图（Fig 23, 24），选 cherry-picked scenes 说"我们更 adhere to input"。Marble 不开放 API 是客观约束，但至少应该做 user study 或者在 WorldScore 这类共同 benchmark 上跑——作者自己引用了 WorldScore [^15] 但没用它对比 Marble
3. **Trajectory planning 是 hand-crafted heuristic**。五种固定 mode + 硬编码的 45° pitch / 120° FoV / 72 candidate nodes——典型的"规则驱动，刚好 work"。每个常数的选择没有 ablation。真正 scalable 的 active view planning 应该是 learned policy（with uncertainty estimation），这部分是下一代该做的事
4. **Panorama → 3DGS 的 information bottleneck 没量化**。整个生成的"世界"被一张 panorama 初始化，然后 35 条轨迹内的 novel view 扩展。这意味着生成世界的 semantic scope 完全被 panorama 锁死，超出 panorama 覆盖的部分都是视频模型 hallucinate 的。**可探索空间的真实边界有多大？** 论文没给。看 Fig 22 演示的 character navigation 范围，仍然是单房间/单街角尺度
5. **Panorama 的 T2P Q-Align Equi 被 HY-World 1.0 反超**（4.493 → 4.403），作者在 Tab 4 旁只说 "best on majority of metrics"，没解释这个回归
6. **Memory 机制的 retrieval 策略是 3D FoV similarity**，但 how robust is this to drift？长轨迹下 memory bank 持续积累生成 keyframes，generation artifacts 也会进 memory——没有针对 error accumulation 的长轨迹 stress test
7. **System/engineering 占比极高**。论文 70% 篇幅是 VAE 选型、RoPE 修改、memory 机制、数据过滤、3DGS tricks、inference 并行、mesh 抽取。science contribution 相对薄——主要就是 keyframe-latent VDM 这一个 insight，其余是 competent 的工程优化和已有技术的巧妙组合。对社区有用，但本质上是 HY-World 1.0/1.5 + Uni3C + WorldMirror 1.0 + FlashWorld 的 incremental integration
8. **"Comparable to Marble" 的 claim 过度**。读者只能看到挑选过的 panoramic input 下的定性对比，Marble 的 text-to-world 模式、较大空间尺度、interactive event 都没比

### 可信评估

#### Artifact 可获取性
- **代码**: 已开源（GitHub: Tencent-Hunyuan/HY-World-2.0，README 已存在）
- **模型权重**: HuggingFace `tencent/HY-World-2.0`，包含 HY-Pano 2.0、WorldStereo 2.0、WorldMirror 2.0
- **训练细节**: 论文给出 batch size、三阶段 curriculum、augmentation 比例、DMD 蒸馏步数，但无具体 hyper-param / step count 全量披露
- **数据集**: 部分公开——real-world panorama 未详述来源，UE 合成 asset 未开放；训练用的 multi-view 数据集（Map-Free、RealEstate10K 等）引用了公共来源

#### Claim 可验证性
- ✅ **WorldMirror 2.0 在 7-Scenes/NRGBD/DTU/ScanNet/NYUv2 SOTA**：Tab 11/12/13 数据齐全，各 resolution 对比清晰，开源 checkpoint 可复现
- ✅ **Keyframe-VAE 优于 Video-VAE**：Tab 7 user study（92.44% camera + 64.39% quality）+ camera metric 一致支持
- ✅ **端到端 712s on H20**：Tab 10 给出分解，可复现
- ⚠️ **"Open-source SOTA" in world generation**：对比 SEVA/Gen3C/Lyra/FlashWorld 确实领先（Tab 5），但没对比 Scene Splatter、DimensionX、WonderWorld 等同期工作
- ⚠️ **"Results comparable to Marble"**：只有定性图，scene 是 cherry-picked，没 user study，没共同 benchmark。对闭源 baseline 的定性对比几乎不构成有效 claim
- ⚠️ **WorldNav 的 5 类 trajectory 是最优配置**：Fig 19 定性消融展示各类互补性，但没定量证明 35 是合适的总数
- ❌ **"Simulating 3D Worlds"**：title 里的 "Simulating" 在论文中对应的是 mesh-based collision detection + character locomotion（WorldLens）——这是 real-time rendering + physics primitive，不是任何意义上的 dynamics simulation。严重的 marketing overreach
- ❌ **"World Model" 的定位**：完整 pipeline 没有任何 temporal dynamics 或 action-conditioned prediction，不符合 Ha & Schmidhuber 意义上的 world model 定义。属于 "3D scene generator" 蹭 world model 概念的热度

---

## 关联工作

### 基于
- **HY-World 1.0**: 前置 offline 3D world model 基线，HY-Pano 2.0、WorldNav 的 scene parsing 都继承 1.0 的工作
- **WorldStereo 1.0 / Uni3C**: camera-conditioned VDM 和 memory 机制的前身；WorldStereo 2.0 的 GGM + SSM 思想来自此
- **WorldMirror 1.0**: feed-forward 3D 重建 foundation，Any-Modal Tokenization
- **3DGS [Kerbl et al. 2023]**: 场景表示
- **MoGe2**: panoramic point cloud 初始化和 normal supervision 都靠它

### 对比
- **Marble (World Labs)**: 唯一的闭源标杆，但对比纯定性
- **SEVA / Gen3C / Lyra / FlashWorld**: open-source 生成类 baseline（Tab 5/6）
- **[[2402-Genie|Genie]] 3 / WorldLabs Marble**: 同期 world model，但范式不同（Genie 3 是 online video，这里是 offline 3DGS）
- **VGGT / π³ / DepthAnything3 / MapAnything / CUT3R / Fast3R / FLARE**: WorldMirror 2.0 的 reconstruction baseline

### 方法相关
- **DMD (Distribution Matching Distillation)**: 4-step diffusion 蒸馏
- **DINOv3**: normalized RoPE 的灵感来源
- **MaskGaussian**: 概率 mask 的 3DGS pruning
- **Recast Navigation**: NavMesh 构建
- **SAM3 / Qwen3-VL**: 语义 grounding
- **[[2411-WorldModelSurvey|World Model Survey]]**: 本文定位背景

---

## Notes

- **核心方法论判断**：HY-World 2.0 的真正贡献是**证明了 open-source 社区可以通过 competent 的工程整合逼近闭源 world generation 效果**——这本身有价值。但它没有 push 科学边界。真正的 insight 是 keyframe-latent VDM；其余是该做的 integration
- **对我研究的启发**：
    1. Spatial-only VAE vs 时空 VAE 的 trade-off——在 VLA 的 action prediction 里，action sequence 短而快变化，是否该用类似的思想（按 keyframe 独立 encode 而非统一时空 latent）？
    2. Normalized RoPE 对 resolution generalization 的根本性修复——可以迁移到任何有 multi-scale 输入的 embodied 模型
    3. Memory 机制的 spatial concatenation vs temporal concatenation——SSM++ 用 retrieval-stitched target-reference 对的设计，对 long-horizon video prediction 里的 memory bank 很有启发
- **要追问**：为什么 "offline 3DGS" 路线和 "online video world model" 路线在 2026 年仍然平行发展？本质是 **static scene generation** vs **dynamic event simulation** 的根本分野。HY-World 2.0 选前者，因为可以对接 CG pipeline、支持 standard rendering；Genie / Marble 选后者，因为更接近"真正的 world model"定义。对 Embodied AI 来说，前者用于 perception/nav prior，后者用于 policy rollout——**需要的是后者**
- **可能的 follow-up 攻击面**：把 WorldNav 替换成 learned planner（with uncertainty-driven active view selection）；在 WorldStereo 2.0 上加入动作条件（从 camera-conditioned → action-conditioned），让它变成真正的 interactive world model；针对长轨迹 error accumulation 设计 benchmark
