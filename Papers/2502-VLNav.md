---
title: "VL-Nav: A Neuro-Symbolic Approach for Reasoning-based Vision-Language Navigation"
authors: [Yi Du, Taimeng Fu, Zhipeng Zhao, Shaoshu Su, Zitong Zhan, Qiwei Du, Zhuoqun Chen, Bowen Li, Chen Wang]
institutes: [University at Buffalo, Carnegie Mellon University]
date_publish: 2025-02-02
venue: arXiv
tags: [VLN, navigation, task-planning]
paper: https://arxiv.org/abs/2502.00931
website: https://sairlab.org/vlnav/
github:
rating: 2
date_added: 2026-04-20
---

## Summary

> [!summary] VL-Nav: A Neuro-Symbolic Approach for Reasoning-based Vision-Language Navigation
> - **核心**: 把 reasoning-based VLN 拆成 "VLM 做语义/任务推理 + 经典 frontier exploration 做几何探索"，由一个统一的 3D scene graph + 最佳视角图像记忆做桥梁，实现实时、可在 Jetson 上跑的多目标抽象指令导航
> - **方法**: NeSy Task Planner（Qwen3-VL + 3D scene graph + image memory，把 "下雨→雨衣" 这种抽象 prompt 拆成 explore / go-to 原子子任务）+ NeSy Exploration System（YOLO-World/FastSAM 提供 instance-based target points，与 frontier-based target points 一起被 Gaussian-mixture VL score + distance + unknown-area 三项加权打分）
> - **结果**: DARPA TIAMAT Phase 1 indoor 83.4% / outdoor 75% SR；真实世界 4 环境平均 86.3% SR、SPL 显著高于 VLFM；483 m 长程跑 + Go2 多楼层 demo
> - **Sources**: [paper](https://arxiv.org/abs/2502.00931) | [website](https://sairlab.org/vlnav/)
> - **Rating**: 2 - Frontier（reasoning-based VLN 范式下的扎实工程示范 + 干净消融，但未开源 + baseline 阵容偏弱 + 任务集小，尚未具备奠基性）

**Key Takeaways:**
1. **抽象指令导航需要显式的任务分解层**：直接把 VLM 塞进每一帧的 frontier 评分（VLFM/SG-Nav 路线）解决不了 "找雨衣 + 雨鞋 + 伞" 这种多目标 + 隐式语义任务，VLM 必须先做 task decomposition 才能避免 aimless wandering，且 SG-Nav 的 per-step CoT 几乎必然超时（MTUR ≈ 1.0）
2. **3D scene graph + 最佳视角图像 = VLM 的 working memory**：每个 object node 存 centroid + 最高置信度检测分 + 该检测时机器人 pose + 最佳视角 RGB，使 VLM 在 "go to" 阶段可以基于符号过滤 top-k 候选 + 在保存的高质量图像上做细粒度 verification（coarse-to-fine），而不是反复重新观察
3. **Instance-Based Target Points (IBTP) 是 verification 的关键 short-cut**：当 OVD 给出 confidence > τ 的 candidate 时，机器人主动绕过去近距离确认（"glimpse → approach → verify"），消融显示去掉 IBTP 后语义复杂场景（Apartment）SR 从 88.9% → 70.2%，是显著大幅退化
4. **工程上是 "VLM 远程 + lightweight detector 本地" 的解耦**：Qwen3-VL-8B 跑在 RTX 4090 笔记本，YOLO-World/FastSAM + 全部规划/控制跑在 Jetson Orin NX；task planner 异步只在子任务切换时触发，避免实时 loop 被大模型卡住
5. **Curiosity 项（distance + unknown-area weighting）专治大场景的反复横跳**：Outdoor 真实环境去掉 curiosity 后 SR 从 77.8% → 55.6%，说明只用 VL score 在大尺度地图里会被远处弱信号反复吸引

**Teaser. 大学楼内多楼层导航 + humanoid demo（VL-Nav 主推视频）**

<video src="https://github.com/Inoriros/VL-NAV-Video/raw/refs/heads/main/vlnav_website_vidoes/multi-floor%20demo.mp4" controls muted playsinline width="720"></video>

<video src="https://github.com/Inoriros/VL-NAV-Video/raw/refs/heads/main/vlnav_website_vidoes/humanoid%20demo.mp4" controls muted playsinline width="720"></video>

---

## Problem 与定位

VL-Nav 把 reasoning-based VLN 与传统 ObjectNav / 命令式 VLN 显式对立：传统任务给定明确目标（"go to L-shape sofa"），而 reasoning-based VLN 给的是隐式、抽象、多目标指令——例如 DARPA TIAMAT 的 "Today's weather report indicates rain. Help Rob find an umbrella, an appropriate jacket, and shoes."。机器人必须：

1. 推理 "rain → 防水装备"，且不能拿成普通夹克或运动鞋（object disambiguation）
2. 在大尺度未知环境里高效搜索分散在不同位置的多个目标
3. 在边缘设备实时跑

作者把现有方法分三类并指出失效模式：
- **Classical SLAM + frontier**：没有语义理解
- **End-to-end RL/VLA**：data-hungry、sim2real 差、不可解释
- **Foundation model modular（VLFM / SG-Nav / ApexNav）**：把 target verification 与 exploration 紧耦合在一起，VLM 在每一帧打分既贵又不擅长跨帧的长程任务记忆，频繁找错物体或超时

> ❓ 这个 framing 里 "reasoning-based VLN" 本质上是 "ObjectNav with abstract goal description + multi-target + long-horizon"。这个区分主要在工程意义上成立（任务分解必要性），但作为研究 framing 与 SG-Nav / ApexNav 类工作的边界并不锋利——它们也支持 open-vocabulary instruction，只是没有显式的 task planner。

---

## 方法

**Figure 1. 整体架构示意：复杂指令经 VLM 推理 + 统一 memory 拆成原子子任务，第一个子任务进入 NeSy Exploration**

![](https://arxiv.org/html/2502.00931v6/x2.png)

系统两个核心模块：NeSy Task Planner（高层任务推理）+ NeSy Exploration System（底层探索）。

### NeSy Task Planner

#### 统一符号 memory

借鉴 Hydra 的 3D scene graph 思路，构建两类 node：
- **Object node**：由 open-vocabulary detector（YOLO-World + FastSAM）生成，存 4 个属性——centroid、最高置信度检测分、该最佳检测时机器人 pose、对应最佳视角 RGB 图像
- **Room node**：用形态学操作做房间分割得到 mask，再用 LLM 根据房间内 object 推断房间标签
- **Edge**：object centroid 落在 room mask 内则连边

> ❓ 把 "best-viewpoint image" 和 scene graph node 绑定是关键设计——它让后续 VLM verification 不需要回到现场重新观察，可以离线 reasoning。这个思路与 mobile manipulation 里 "memory-of-best-view" 的工作（如部分 spatial memory 文献）有共鸣。

#### Task decomposition + replanning

VLM 后端是 **Qwen3-VL-8B**。Planner 把复杂任务拆成两类原子子任务：
- `exploration`：去某个未充分覆盖区域收集信息
- `go to`：去到某个被识别出的目标物体并报告

每个子任务完成后触发 replanning，基于新的 memory 状态生成下一批子任务。

**Coarse-to-fine target acquisition**：找 "rain jacket" 这类抽象目标时
1. **Symbolic Filtering**：在 3D scene graph 里按 detection confidence 取 top-k（例如 k=3）候选
2. **Neural Verification**：VLM 对这些候选的 best-view 图像 + 邻居场景图节点做细粒度推理，挑出语义最匹配的那个

选定后 publish 该 best-view pose 作为导航目标。

### NeSy Exploration System

**Figure 3. 探索系统总览：VL 模块给 instance points + map 模块给 frontier points → candidate pool → NeSy scoring 选目标**

![](https://arxiv.org/html/2502.00931v6/x3.png)

#### Frontier-Based Target Points

cell $(m_x, m_y)$ 是 frontier 当且仅当它本身 free 且至少一个邻居 unknown。每次更新只测试在前向 FoV wedge 内的 cell：

$$
\begin{split}\Bigl|\mathrm{Angle}\bigl((m_{x},m_{y}),(x_{r},y_{r})\bigr)-\theta_{r}\Bigr|\;&\leq\;\frac{\mathrm{hfov}}{2},\\
\text{and}\quad\|(m_{x},m_{y})-(x_{r},y_{r})\|\;&\leq\;R,\end{split}
$$

BFS 聚类后每个 cluster 用 centroid（小 cluster）或多个采样点（大 cluster）表示。

#### Instance-Based Target Points (IBTP)

VL 模块周期性输出 $(q_x, q_y, \text{confidence})$，confidence > $\tau_{\det}$ 的保留并做 voxel-grid 下采样。这模仿人在搜索时 "瞥见疑似目标→走近确认" 的行为，是相对 VLFM 的关键差异——VLFM 的 frontier-only 设计没有这种 "短路" 机制。

#### NeSy Scoring

**VL Score**：把开放词检测结果转成 FoV 上的 Gaussian mixture 分布。$K$ 个 likely direction，每个由 $(\mu_k, \sigma_k, \alpha_k)$ 参数化（$\sigma_k$ 实现里固定 0.1）：

$$
S_{\mathrm{VL}}(\mathbf{g})\;=\;\sum_{k=1}^{K}\alpha_{k}\;\exp\!\Bigl(-\tfrac{1}{2}\Bigl(\tfrac{\Delta\theta-\mu_{k}}{\sigma_{k}}\Bigr)^{2}\Bigr)\;\cdot\;C\bigl(\Delta\theta\bigr),
$$

$$
C(\Delta\theta)=\cos^{2}\!\biggl(\frac{\Delta\theta}{(\theta_{\mathrm{fov}}/2)}\cdot\frac{\pi}{2}\biggr).
$$

$C(\Delta\theta)$ 是借鉴 VLFM 的 "view confidence" 项，FoV 边缘的检测被降权。最终 clip 到 $[0,1]$。

**Curiosity Cues**：

$$
S_{\mathrm{dist}}(\mathbf{g})\;=\;\frac{1}{1+d(\mathbf{x}_{r},\mathbf{g})}, \qquad
S_{\mathrm{unknown}}(\mathbf{g})\;=\;1\;-\;\exp\!\bigl(-\,k\,\mathrm{ratio}(\mathbf{g})\bigr)
$$

其中 $\mathrm{ratio}(\mathbf{g}) = \#(\text{unknown cells}) / \#(\text{total visited cells})$，由从 $\mathbf{g}$ 出发的 local BFS 统计。

**Combined NeSy Score**（仅 frontier-based goal 用）：

$$
S_{\mathrm{NeSy}}(\mathbf{g})\;=\;w_{\mathrm{dist}}\;S_{\mathrm{dist}}(\mathbf{g})\;+\;w_{\mathrm{VL}}\;S_{\mathrm{VL}}(\mathbf{g})\;\cdot\;S_{\mathrm{unknown}}(\mathbf{g})
$$

instance-based goal 不用 curiosity 项，因为它们的目的是 verify 而非 explore。

### Goal Selection

优先选高语义信心的 instance target——若有任何 instance target 距离 > $\delta_{\text{reached}}$，取 $S_{\mathrm{VL}}$ 最高那个；否则 fallback 到 $S_{\mathrm{NeSy}}$ 最高的 frontier。路径用 **FAR Planner** 生成。

> ❓ 这个 hierarchy 表面上让 verification 优先于 exploration——但如果 OVD 在错误物体上给了 high confidence（典型失败模式），会发生 "走过去发现错了 → 回头继续 frontier"，浪费时间。论文没量化这种 false-positive verification 的开销。

---

## 实验

### 仿真：DARPA TIAMAT Phase 1

**环境**：HabitatSim 上 Apartment 1 / 2（室内）+ IsaacSim 上 Camping Site / Factory（室外），机器人是 Boston Dynamics Spot + 5 个 RGB-D 相机环视。每环境 8 个抽象任务 × 3 trials。

**Table 1. DARPA TIAMAT 仿真结果（SR ↑ / MTUR ↓）**

| Method | Apt 1 SR | Apt 2 SR | Camping SR | Factory SR | Apt 1 MTUR | Apt 2 MTUR | Camping MTUR | Factory MTUR |
|---|---|---|---|---|---|---|---|---|
| Frontier Exploration | 8.3 | 8.3 | 0.0 | 0.0 | 0.958 | 0.884 | 1.000 | 1.000 |
| VLFM | 8.3 | 8.3 | 4.2 | 8.3 | 0.931 | 0.864 | 0.953 | 0.859 |
| SG-Nav | 0.0 | 4.2 | 0.0 | 8.3 | 1.000 | 0.973 | 1.000 | 0.901 |
| ApexNav | 25.0 | 25.0 | 20.8 | 12.5 | 0.817 | 0.795 | 0.828 | 0.861 |
| VL-Nav w/o IBTP | 70.8 | 62.5 | 62.5 | 58.3 | 0.680 | 0.724 | 0.731 | 0.762 |
| VL-Nav w/o Curiosity | 79.1 | 75.0 | 58.3 | 66.7 | 0.612 | 0.635 | 0.793 | 0.735 |
| **VL-Nav** | **87.5** | **79.2** | **75.0** | **75.0** | **0.562** | **0.591** | **0.647** | **0.679** |

baseline 几乎全军覆没——VLFM、Frontier 在 8% 量级；SG-Nav 因每步 CoT 几乎全部超时（MTUR ≈ 1.0）。VL-Nav 把 SR 拉到 75–87.5%，相对最强 baseline ApexNav 也有 50+ 绝对点的提升。

**Figure 5/6. 定性结果：indoor 隐式语义推理（weather→rain jacket）+ outdoor unstructured 多目标搜索（laptop / outfit / truck），projected value map 显示探索如何被引导**

![](https://arxiv.org/html/2502.00931v6/x5.png)

![](https://arxiv.org/html/2502.00931v6/x6.png)

### 真实世界：4 环境 9 任务

**Table 2. 真实世界结果（SR / SPL）**

| Method | Hall SR | Office SR | Apt SR | Outdoor SR | Hall SPL | Office SPL | Apt SPL | Outdoor SPL |
|---|---|---|---|---|---|---|---|---|
| Frontier Exploration | 40.0 | 41.7 | 55.6 | 33.3 | 0.239 | 0.317 | 0.363 | 0.189 |
| VLFM | 53.3 | 75.0 | 66.7 | 44.4 | 0.366 | 0.556 | 0.412 | 0.308 |
| VL-Nav w/o IBTP | 66.7 | 83.3 | 70.2 | 55.6 | 0.593 | 0.738 | 0.615 | 0.573 |
| VL-Nav w/o curiosity | 73.3 | 86.3 | 66.7 | 55.6 | 0.612 | 0.743 | 0.631 | 0.498 |
| **VL-Nav** | **86.7** | **91.7** | **88.9** | **77.8** | **0.672** | **0.812** | **0.733** | **0.637** |

硬件：四轮 Rover（Jetson Orin NX，本地跑实时 stack）+ Unitree Go2（多楼层 demo，加 D1 arm + D435 做抓取），都用 Livox Mid-360 LiDAR + RealSense D455。状态估计用 Super Odometry。Qwen3-VL-8B 在远端 RTX 4090 笔记本异步跑——只在子任务切换时触发，所以延迟不影响实时 loop。SG-Nav / ApexNav 因为太重无法在边缘部署，没法对比。

**消融的两条干净结论**：
- 去掉 **IBTP**：Apartment（语义复杂）SR 88.9 → 70.2，验证机制最关键的是杂乱场景
- 去掉 **Curiosity**：Outdoor（大场景）SR 77.8 → 55.6，distance + unknown-area 项主要解决大场景的反复横跳

**Figure 7. 仿真 4 环境（上）+ 真实 4 环境（下）**

![](https://arxiv.org/html/2502.00931v6/x7.png)

**Real-world deployment demos：indoor + outdoor 真机视频**

<video src="https://github.com/Inoriros/VL-NAV-Video/raw/refs/heads/main/vlnav_website_vidoes/indoor%20vln%20real.mp4" controls muted playsinline width="720"></video>

<video src="https://github.com/Inoriros/VL-NAV-Video/raw/refs/heads/main/vlnav_website_vidoes/outdoor%20demo%20real.mp4" controls muted playsinline width="720"></video>

---

## 关联工作

### 基于
- **Hydra (Hughes et al.)**：3D scene graph 的 object/room node 设计直接借鉴
- **VLFM**：view-confidence cosine 项 $C(\Delta\theta)$ 与 frontier-based 的整体框架沿用
- **FAR Planner**：路径规划组件
- **Super Odometry**：状态估计组件
- **YOLO-World + FastSAM**：开放词汇检测与分割
- **Qwen3-VL-8B**：VLM 推理 backbone

### 对比
- [[2402-NaVid|NaVid]]: 视频 LLM 直接 plan 下一步动作，end-to-end 路线，与本文模块化 NeSy 路线对立
- **VLFM** (Yokoyama et al., 2024): 最直接的 baseline，frontier + VL score，但只支持 single explicit target，无 task decomposition、无 IBTP
- **SG-Nav** (Yin et al.): scene graph + 每步 LLM CoT，因 latency 在大尺度任务里几乎全超时（DARPA TIAMAT MTUR≈1.0）
- **ApexNav**: 开放词汇 ObjectNav，缺 long-horizon symbolic memory，多目标场景下 SR 仅 12-25%

### 方法相关
- [[2412-NaVILA|NaVILA]]、[[2506-VLNR1|VLN-R1]]、[[2507-StreamVLN|StreamVLN]]: 同期 VLN 工作，对比 modular vs. end-to-end 路线
- [[VLN]] DomainMap: 整体路线图

---

## 论文点评

### Strengths

1. **正确诊断了现有 foundation-model VLN 的瓶颈**——把 VLM 塞进每一帧打分既贵又不擅长 long-horizon 任务记忆。把 task decomposition 与 per-step exploration scoring 解耦，是合理且 scalable 的架构选择。
2. **统一符号 memory（3D scene graph + best-view image）是好设计**：让 VLM 的 verification 可以离线在保存的高质量图像上做，不需要重访目标。这个思路对 mobile manipulation / lifelong robot 都是可复用 building block。
3. **工程落地扎实**：Jetson Orin NX 实时 + 远端 VLM 异步、483 m 长程跑、多楼层 humanoid demo、跨四种真实环境，证明不是只在 Habitat 里能跑。Curiosity 消融在 outdoor 上的 22 点跌幅说明几何项不是 add-on 而是真的 load-bearing。
4. **消融干净、解释清晰**：IBTP 与 curiosity 各自影响哪类场景，在数据上对应得很整齐，不是把所有部件混在一起 claim "全都重要"。

### Weaknesses

1. **VLM backbone 的具体作用没消融**：Qwen3-VL-8B 换成更小或更弱的 VLM 性能怎么变？task planner 失败模式（subtask 拆错、replanning 死循环）的案例分析缺失。
2. **真实世界 baseline 阵容偏弱**：因为 SG-Nav / ApexNav 跑不了边缘设备就只对比 Frontier + VLFM。但这两条 baseline 都是 2024 之前的水平，2025–2026 的 modular VLN 工作（如各种 zero-shot ObjectNav 改进、stream-VLN 路线）没有比较。"超过 baseline 50 点" 的强 claim 在仿真里 holds，在真实里因 baseline 弱而打折。
3. **IBTP 的 false-positive 开销没量化**：当 OVD 给错误物体高 confidence 时，机器人会被 "钓" 过去再回头。这种 verification waste 可能在某些场景反而拖累 SPL，论文没分析。
4. **任务集偏小**：每仿真环境 8 任务 × 3 trials = 24 trials，real-world 9 任务 × 3-5 trials。统计意义有限，置信区间也没报。
5. **"reasoning-based VLN" 的概念边界模糊**：和 "open-vocabulary ObjectNav with abstract instruction" 的差别本质上是工程上的 task planner 模块，不是任务定义层面的根本差异。把这点包装成一个新范式略显 oversell。
6. **场景图本身的质量是否瓶颈未讨论**：Hydra-style scene graph 在 cluttered / dynamic 场景下检测错误率怎么样？scene graph 错了 task planner 直接误判。

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源（截至当前 v6，paper 与 project page 均未给出 GitHub 代码仓库链接，video 仓库 `Inoriros/VL-NAV-Video` 只放了 demo 视频）
- **模型权重**: 不适用（系统组装现成模型 Qwen3-VL-8B + YOLO-World + FastSAM，无新训练 checkpoint）
- **训练细节**: 不适用（无训练，全 zero-shot 组装）；但 NeSy scoring 的 $w_{\mathrm{VL}}, w_{\mathrm{dist}}, k, \tau_{\det}, \delta_{\text{reached}}$ 等关键超参数值未在正文给出
- **数据集**: DARPA TIAMAT Phase 1 是受限竞赛环境（HabitatSim Apartment + IsaacSim Camping/Factory，9 个真实环境为作者自建），完整任务 prompt 列表未公开

#### Claim 可验证性
- ✅ **DARPA TIAMAT 仿真 SR 75-87.5%**：Table I 完整列出，仿真环境受限但可重现
- ✅ **真实世界 SR 86.3%、SPL 显著高于 VLFM**：Table II 数据 + 项目页 demo 视频可见
- ✅ **483 m 长程跑 + 多楼层 demo**：项目页有视频佐证
- ⚠️ **"intertwines neural reasoning with symbolic guidance"** 中 "symbolic" 实际指 "scene graph + heuristic scoring"，和 LogiCity 等 NeSy 工作中 "符号逻辑/规则引擎" 的强意义不一样——更接近 "structured memory + classical frontier"，NeSy framing 略 oversell
- ⚠️ **"vastly superior path efficiency vs. classical frontier and VLFM"**：vastly 在某些指标上成立（Office SPL 0.812 vs 0.317），但因 baseline 阵容弱（缺 SG-Nav/ApexNav 真机数）打折
- ⚠️ **"sim-to-real transfer 强"**：仿真和真机用的不是同一套环境/任务，所以严格意义上不是 transfer 测试，是各自独立部署的鲁棒性
- ❌ 无明显营销话术

### Notes

- **复用价值**：VL-Nav 的 "VLM remote async + lightweight detector local" 解耦工程模式可以直接迁移到 mobile manipulation 任务。big-VLM 不应该在控制 loop 里同步跑，这点论文给了很干净的工程样例。
- **Best-view image memory** 这个 abstraction 值得追踪——它把 "什么时候 query VLM" 与 "用什么图像 query VLM" 解耦，比 frame-by-frame 输入要稳定且便宜得多。可以和 [[SpatialRep]] 里 spatial memory 的工作连起来读。
- **Reasoning-based VLN** 作为问题 framing：DARPA TIAMAT 是个有意思的 benchmark（多目标 + 抽象 + 大场景），但目前公开的任务量还小。如果未来扩展到 100+ 任务且开源，会成为 VLN 领域的 standard testbed。
- **NeSy 的命名**：本文 NeSy 中的 "Symbolic" 是 "scene graph + heuristic scoring 项"，比 LogiCity 这种带规则推理引擎的 NeSy 弱一档。读者要警惕 "NeSy" 这个标签在 VLM 时代被泛化成 "structured memory + neural"。
- **可考察的 follow-up**：(1) IBTP 在 false-positive detection 下的开销曲线；(2) 把 task planner 替换为更便宜的 LLM（不需要 vision）能不能保持性能；(3) 把 best-view image memory 扩展为可编辑/可遗忘的 lifelong memory。

### Rating

**分数**：2 - Frontier
**理由**：属于 reasoning-based VLN 方向的代表性工程工作，Strengths 指出的 "task decomposition + per-step exploration 解耦" 架构 + best-view image memory 是可复用 building block，DARPA TIAMAT 上相对 VLFM/SG-Nav/ApexNav 有大幅 SR 提升且消融干净。但够不到 Foundation：2025-02 发布、未开源、baseline 阵容偏弱（缺 SG-Nav/ApexNav 真机数）、任务集小（24 trials/仿真环境），尚未被主流 VLN 后续工作作为标准 baseline 采纳；VLFM / NaVILA / StreamVLN 仍是更常见的比较对象。相对 Archived 档，本文的问题 framing（reasoning-based VLN）和 NeSy 探索打分设计在同期工作里仍是清晰的 frontier 参考，而非可被替代的 incremental 工作。
