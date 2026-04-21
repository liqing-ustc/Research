---
title: "DAM-VLA: A Dynamic Action Model-Based Vision-Language-Action Framework for Robot Manipulation"
authors: [Xiongfeng Peng, Jiaqian Yu, Dingzhe Li, Yixiang Jin, Lu Xu, Yamin Mao, Chao Zhang, Weiming Li, Sujin Jang, Dongwook Lee, Daehyun Ji]
institutes: [Samsung R&D Institute China-Beijing (SRCB), Samsung AI Center DS Division, Hanyang University ERICA]
date_publish: 2026-03-01
venue: ICRA 2026
tags: [VLA, manipulation, diffusion-policy]
paper: https://arxiv.org/abs/2603.00926
website: https://research.samsung.com/blog/DAM-VLA-A-Dynamic-Action-Model-Based-Vision-Language-Action-Framework-for-Robot-Manipulation
github:
rating: 1
date_added: 2026-04-21
---

## Summary

> [!summary] DAM-VLA: A Dynamic Action Model-Based VLA Framework for Robot Manipulation
> - **核心**: 把单一 diffusion action head 拆成两个专家头——arm-movement vs gripper-manipulation——由 VLM reasoning latent 学习的 router 在 timestep 级动态选择，并用 dual-scale (trajectory + action chunk) weighting 监督训练。
> - **方法**: VLM (DINOv2 + SigLIP + LLaMA-2) 输出 cognition / reasoning latent；reasoning latent 进 router 预测权重 $w$；arm head 取全局 class token，gripper head 取 register token；trajectory-level asymmetric Gaussian + action-chunk exponential decay 共同加权两个 diffusion loss。
> - **结果**: SIMPLER (Google VM 83% / VA 81%, WidowX VM 71%)、FurnitureBench One-Leg 56% (vs CogACT 42%)、real-world Franka pick-and-place ID 91.4% / OOD 82.2% (vs CogACT 62.9% Avg)。
> - **Sources**: [paper](https://arxiv.org/abs/2603.00926) | [website](https://research.samsung.com/blog/DAM-VLA-A-Dynamic-Action-Model-Based-Vision-Language-Action-Framework-for-Robot-Manipulation)
> - **Rating**: 1 - Archived（ablation 显示核心 dual-head 架构单独贡献有限，主要增益来自 dual-scale weighting；代码未开源、routing 信号依赖 gripper 翻转难以 scale，预期不会成为后续工作主 baseline。）

**Key Takeaways:**
1. **"Arm movement vs gripper manipulation" 的分离假设**：作者列出三条经验观察作为 motivation——(i) path constraints（arm 路径自由 / gripper 必须精准 grasp pose）、(ii) visual attention（arm 全局 / gripper 局部）、(iii) dataset representation（arm episode 远多于 gripper，但 gripper 关键且复杂）。这是一个 well-defined 但仍待验证泛化性的二分。
2. **Token-level 任务分配**：用 DINOv2 的 class token 喂 arm head（全局），register token 喂 gripper head（局部）。把"同一编码器不同 token 表征不同 attention scale"的设计直接挂到 action expert 上，是 cute 的工程决策，但论文没有用 attention map 等可视化证据来支撑这一选择。
3. **Routing 监督信号来自 gripper 状态翻转**：$\hat w$ 的 ground truth 取自 dataset 中 gripper open/close 的二值翻转点——简单可批量生成的弱监督，但只能区分"接近 grasp 时刻 vs 其他"，无法处理 contact-rich 但无 gripper 状态变化的过程（如 screw、push、wipe）。
4. **Dual-scale 加权 = 时间维 prior 注入**：trajectory 用非对称 Gaussian（leading $\sigma_l=6$ > trailing $\sigma_r=2$，强调"接触前需要更精准"），action chunk 用 $\gamma^j = 0.8^j$ 指数衰减（远期预测置信度低）。两者外积调制 per-timestep diffusion loss。
5. **Ablation 显示 dual head 单独贡献有限**：仅加 DAM 不加 dual latent (DL) 平均只到 66%（vs 完整版 78%）；移除 VT + ACW + TW 后只剩 DAM+DL 跌到 60%。**核心增益来自 dual-scale weighting + dual latent 协同**，而非 dual head 架构本身。

---

## 1. Motivation：为什么要把 action head 拆开

VLA 主流路线两条：
- **CoT-style**（RT-H、RT-Affordance、ECoT）：用语言 / affordance 中间表示丰富 reasoning，但额外 token 让 inference 显著变慢（OpenVLA 7 个 action token vs ECoT 350 个）。
- **Diffusion-head VLA**（π0、TinyVLA、RDT-1B、CogACT、HybridVLA）：在 VLM 后挂 diffusion head 做连续 action，精度好但 head 只看 VLM-extracted feature，缺低层视觉细节。

作者认为单一 diffusion head 没法同时满足"arm 大尺度移动"和"gripper 接触级精度"两类截然不同的需求，提出三条区分（论文 Section I 总结）：

1. **Path Constraints**：arm 多解，gripper 单解（必须正确 grasp pose）。
2. **Visual Attention**：arm 需要 global scene，gripper 需要 local fine-grained。
3. **Dataset Representation**：arm step 多但好学，gripper step 少但是 critical bottleneck。

**Figure 1.** 在 "place carrot on plate" 任务上对比 arm movement 与 gripper manipulation 的三条区分

![](https://arxiv.org/html/2603.00926v1/x1.png)

> ❓ 这三条观察是定性 framing，论文没有给出 quantitative evidence（如 attention 热图统计、数据集中 gripper 状态切换点占比）。把"task-level 二分"直接落到"两个独立 diffusion head"是相当强的归纳偏置。

## 2. 整体架构

**Figure 2.** DAM-VLA 整体结构：VLM 出 cognition / reasoning latent；router 由 reasoning latent 预测 $w$；dynamic action model 根据 $w$ 选择 arm 或 gripper head，前者吃 class token，后者吃 register token

![](https://arxiv.org/html/2603.00926v1/x2.png)

输入：RGB observation $o_t$ + language instruction $l$。输出：action chunk $[a_t, ..., a_{t+N}]$，每个 action 7-DoF（$\delta x$、$\delta \theta$、gripper state $s^{grip} \in \{0,1\}$）。

三个组件：

1. **VLM**：vision tower = DINOv2 + SigLIP（双编码器拼通道再投影），vision token 拼 language token 喂 LLaMA-2，输出 cognition latent $f^{cog}$（最后一层 hidden）和 reasoning latent $f^{rea}$（第二层 hidden）。
2. **Action Routing**：$f^{rea}$ → FC → norm → 标量 $w \in [0,1]$。
3. **Dynamic Action Model**：两个 DiT-based diffusion head（arm / gripper），arm head 条件 = $f^{cog} + f^{cls}$，gripper head 条件 = $f^{cog} + f^{reg}$。inference 时 $w<0.5$ 跑 arm，否则跑 gripper。

> ❓ "reasoning 用浅层 hidden、cognition 用深层 hidden" 这种 dual-latent 划分没有可解释性证据。Ablation Table VI 显示去掉 DL 性能从 78% 掉到 66%，但论文没解释为什么浅层就是 reasoning 而深层就是 cognition——这可能只是经验最优 layer 的事后解读。

## 3. Action Routing Mechanism

routing 的 label $\hat w \in \{0, 1\}$ 从数据中**机械取自 gripper 状态翻转**：
- gripper open ↔ closed 切换点附近 → $\hat w = 1$（gripper manipulation phase）
- 其他 → $\hat w = 0$（arm movement phase）

预测 $w$ 由 cross-entropy 监督：

$$
L_{class}=\big\| -\big(\hat{w}\log(w)+(1-\hat{w})\log(1-w)\big) \big\|_1
$$

> ❓ Routing 信号纯靠 gripper open/close 翻转——这意味着对于不需要 gripper 切换的接触阶段（screw、wipe、push），整段都被标成 arm movement。FurnitureBench 的 "screw leg" 步骤恰好属于这类，论文显示 DAM-VLA 在该步骤 56% vs CogACT 42% 仍提升，但绝对值远低于前三步（100%）——routing 假设的边界情况。

## 4. Dynamic Action Model 训练

每个 head 是 DiT (Peebles & Xie 2023)，按 CogACT 的 block 设计实现。两个 head 各自有 noise prediction loss，用 Mahalanobis 距离形式按权重加权：

$$
L_{move}=\big\|n^{move}_{i}-\hat{n}^{move}\big\|^{2}_{\sum \hat{w}^{move}}
$$

$$
L_{mani}=\big\|n^{mani}_{i}-\hat{n}^{mani}\big\|^{2}_{\sum \hat{w}^{mani}}
$$

总 loss：

$$
L = w_1 L_{move} + w_2 L_{mani} + w_3 L_{class}, \quad w_1=w_2=1.0,\ w_3=10^{-4}
$$

注意 $w_3$ 极小——routing 监督只是辅助，不主导训练。

## 5. Dual-Scale Action Weighting

**核心 trick**：在 timestep × action-chunk index 二维上构造权重，调制每个样本对两个 head 的 loss 贡献。

**Figure 3.** Dual-scale action weighting：trajectory 维用非对称高斯标记 critical manipulation phase，chunk 维用指数衰减强调近期 step

![](https://arxiv.org/html/2603.00926v1/x3.png)

### Trajectory-level $w^t$
对每个 gripper manipulation 段 $k$，用以 grip 状态切换时刻 $u$ 为均值的 **非对称高斯**：

$$
\big\{\mathcal{N}(u, \sigma_l^2),\ \mathcal{N}(u, \sigma_r^2)\big\},\quad \sigma_l=6,\ \sigma_r=2
$$

leading edge variance > trailing edge variance —— prior 是"接触发生前的姿态准备远比接触后阶段重要"。聚合：$w^t = \text{Norm}\big(\sum_k w_k^t\big)$。

### Action-chunk-level $w^a$
chunk 内第 $j$ 个未来 step 加权：

$$
w_j^a = \gamma^j,\quad \gamma=0.8
$$

未来越远，loss 权重越小——补偿 prediction confidence 的 temporal decay。

### Multi-scale 整合

$$
w^{move} = (1-w^t)\odot w^a,\quad w^{mani} = w^t \odot w^a
$$

最后由 trajectory 权重二值化得到 $\hat w$（$w^t > 0.5$ → $\hat w = 1$）作为 router 的 supervision label。

## 6. Experiments

### 6.1 训练配置
- **Pre-training**：Open X-Embodiment 中 Fractal + BridgeDataV2 子集；lr $2\times 10^{-5}$，batch 256，8× H100，约 2 天。
- **Fine-tuning**：FurnitureBench One-Leg 用 500 expert traj；real-world Franka pick-and-place 用 50 demo。

### 6.2 SIMPLER 评测

**Table I.** Google robot, Variant Aggregation (VA) 设置（lighting / background 等扰动）

| Method | PCC | MN | OCD | ODPA | Avg |
| ---- | ---- | ---- | ---- | ---- | ---- |
| RT-1 | 90% | 46% | 35% | 3% | 44% |
| RT-2-X | 82% | 79% | 35% | 21% | 54% |
| Octo-Base | 1% | 4% | 1% | 0% | 1% |
| RoboVLM | 76% | 60% | 11% | - | - |
| SpatialVLA | 88% | 73% | 42% | - | - |
| OpenVLA | 64% | 64% | 19% | 1% | 37% |
| CogACT | 96% | 84% | 29% | 40% | 62% |
| **DAM-VLA** | **98%** | 74% | **68%** | **84%** | **81%** |

最大增益在 ODPA (Open Drawer and Place Apple)：84% vs CogACT 40% —— 这正是需要 arm 大幅运动 + gripper 精准 manipulation 的 long-horizon 任务，与方法 motivation 高度吻合。MN (Move Near) 反而比 CogACT 差 10%，论文未解释。

**Table II.** Google robot, Visual Matching (VM)

| Method | PCC | MN | OCD | ODPA | Avg |
| ---- | ---- | ---- | ---- | ---- | ---- |
| CogACT | 92% | 82% | 75% | 39% | 72% |
| π₀\* (open-pi-zero) | 89% | 81% | 55% | 53% | 70% |
| **DAM-VLA** | **96%** | **84%** | **75%** | **78%** | **83%** |

**Table III.** WidowX robot, VM

| Method | SoT | CoP | GoY | EiB | Avg |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Octo-Small | 47% | 8% | 1% | 51% | 27% |
| OpenVLA | 4% | 0% | 0% | 4% | 2% |
| SpatialVLA | 17% | 25% | 29% | 100% | 43% |
| CogACT | 63% | 50% | 25% | 71% | 52% |
| π₀\* | 62% | 59% | 24% | 81% | 57% |
| **DAM-VLA** | **88%** | **71%** | 25% | **100%** | **71%** |

> ❓ "Stack Green Block on Yellow Block" (GoY) 卡在 25%，与 SpatialVLA 持平，远低于其他三个任务——这是堆叠任务，gripper release 后 block 是否稳定与 routing 设计无关。论文 Conclusion 也承认这是局限。

### 6.3 FurnitureBench One-Leg 装配

**Figure 4.** FurnitureBench One-Leg assembly task 全过程：grasp tabletop → place tabletop → pick up leg → insert leg → screw leg

![](https://arxiv.org/html/2603.00926v1/x4.png)

**Table IV.** 各 step 累积成功率

| Method | Step1 | Step2 | Step3 | Step4 | Step5 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| OpenVLA | 96% | 94% | 78% | 53% | 29% |
| CogACT | 98% | 96% | 96% | 56% | 42% |
| **DAM-VLA** | **100%** | **100%** | **100%** | **62%** | **56%** |

前三步（grasp/place/pick）100% 完成，contact-rich 的 step 4 (insert) 和 step 5 (screw) 仍然是瓶颈，但相对 CogACT 的 +14% 增益（56% vs 42%）显示 dual head 对 contact-rich 阶段确有帮助。

### 6.4 Real-world Franka pick-and-place

**Figure 5.** Real-world 评测设置：ID（已见 lighting / object pose）vs OOD（新背景、新物体、distractor）

![](https://arxiv.org/html/2603.00926v1/x5.png)

**Table V.** ID / OOD 成功率（80 trials）

| Method | ID | OOD | Avg |
| ---- | ---- | ---- | ---- |
| CogACT | 65.7% | 60.0% | 62.9% |
| **DAM-VLA** | **91.4%** | **82.2%** | **86.8%** |

OOD 仅比 ID 掉 9.2pp（CogACT 5.7pp）——绝对值占优但 OOD-vs-ID 相对掉幅略大于 baseline。

### 6.5 Ablation

**Table VI.** 五个组件的 ablation（VT=visual tokens, ACW=action chunk weight, TW=trajectory weight, DAM=dual action model, DL=dual latent）

| VT | ACW | TW | DAM | DL | Google VM | Google VA | WidowX VM | Avg |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| - | - | - | - | - | 64% | 61% | 50% | 58% |
| ✓ | ✓ | - | - | - | 78% | 68% | 53% | 66% |
| ✓ | ✓ | ✓ | - | - | 76% | 63% | 51% | 63% |
| ✓ | ✓ | ✓ | ✓ | - | 82% | 72% | 43% | 66% |
| ✓ | ✓ | ✓ | ✓ | ✓ | **83%** | **81%** | **71%** | **78%** |
| - | ✓ | ✓ | ✓ | ✓ | 84% | 75% | 58% | 73% |
| - | - | - | ✓ | ✓ | 66% | 64% | 49% | 60% |

读这张表的关键观察：
- 单加 VT + ACW（基本只是单 head + chunk weight）就从 58→66，**绝大部分单组件增益其实来自 chunk-weighted diffusion supervision**。
- 加 TW 不加 DAM 反而掉 3pp（66→63）——TW 是为 dual head 设计的，单 head 模型用了反而干扰。
- 加 DAM 但不加 DL 只能恢复到 66——**dual head 架构单独无收益**。
- 完整版 78% vs 只有 DAM+DL（无 VT/ACW/TW）的 60% —— **18pp 增益主要来自 dual-scale weighting + visual token 双输入**，dual head 本身只是必要条件。

> ❓ 这一 ablation 暗示 contribution 排序更可能是：dual-scale weighting > dual latent ≈ visual token 选择 > dual action head 架构。论文 framing 把 dual head 当主卖点，但数据更支持 weighting scheme 才是主因。

---

## 关联工作

### 基于
- [[2410-Pi0|π0]]：在 VLM 后挂 diffusion / flow matching head 做连续 action 的 paradigm 起点
- [[2406-OpenVLA|OpenVLA]]：discretized action token 路线，DAM-VLA 作为 diffusion head 路线的对照
- CogACT：直接前作，DAM-VLA 的 DiT block 实现复用 CogACT 的设计；real-world 主要 baseline
- DINOv2：register token 概念来自此，作者用其作为 gripper head 的 local 视觉条件
- Diffusion Policy：diffusion-based action generation 的源头

### 对比
- [[2406-OpenVLA|OpenVLA]] / [[2307-RT2|RT-2]] / [[2405-Octo|Octo]] / SpatialVLA / RoboVLM：SIMPLER 主对比表
- ECoT / RT-H / RT-Affordance：CoT-style VLA，DAM-VLA 用 routing 替代显式 reasoning token，号称避免 inference 慢
- [[2410-Pi0|π0]] (open-pi-zero) / HybridVLA / TinyVLA / RDT-1B：diffusion-head VLA，被 DAM-VLA 定位为"未充分利用 VLM 多 token 信息"的 baseline

### 方法相关
- DiT (Peebles & Xie)：两个 action head 的 backbone block
- LLaMA-2：作为 LLM backbone
- SigLIP：vision tower 的另一半（与 DINOv2 拼接）
- Open X-Embodiment + Fractal + BridgeDataV2：pre-training 数据

---

## 论文点评

### Strengths

1. **Motivation 清晰**：arm vs gripper 的三条区分（path / attention / dataset）是直观且 testable 的 framing，把 long-horizon contact-rich 任务的 bottleneck 显式建模。
2. **Routing 监督低成本**：从 gripper open/close 翻转自动生成 $\hat w$，无需额外标注，对 scaling 友好。
3. **Dual-scale weighting 设计有 prior 含义**：非对称高斯（接触前比接触后更重要）+ chunk 维指数衰减（远期 horizon 不可信）都是 grounded prior，不是凑数 trick。
4. **Real-world 实验完整**：80 trial、ID/OOD 二分、对 CogACT 公平 fine-tune，比许多只跑 SIMPLER 的工作扎实。
5. **Ablation 完整且诚实**：Table VI 拆出五个组件单独/组合的 contribution，没有藏起反直觉的现象（如 TW 单独加掉性能）。

### Weaknesses

1. **核心架构 contribution 被 ablation 削弱**：Table VI 显示 dual head 架构本身贡献有限，主要增益来自 dual-scale weighting。论文 framing 与数据 evidence 不完全 aligned。
2. **Routing label 假设过窄**：仅依赖 gripper open/close 翻转，对 wipe / push / screw 等 contact-rich 但 gripper 状态不变的子任务无法区分；这也解释了为什么 FurnitureBench step 5 (screw) 提升有限。
3. **缺少可视化证据**：声称 class token 全局、register token 局部，但没给出 attention map / 特征聚类等支撑——这是 DINOv2 文献里的 prior，搬到 VLA 的有效性应独立 verify。
4. **基线选择不均衡**：SIMPLER table 把 OpenVLA、CogACT、π₀ 都列了，但 FurnitureBench 只对比 OpenVLA + CogACT，real-world 只对比 CogACT。π₀ / π0.5 / OpenVLA-OFT 这些更近的 diffusion-VLA baseline 在最有挑战性的两个 setting 缺席。
5. **"Move Near" 任务退步未解释**：VA 表上 DAM-VLA 74% vs CogACT 84%，论文未讨论。这正是"主要是 arm movement 不需要 gripper precision"的任务，dual-head 架构反而拖累——值得追问的负面信号。
6. **代码 / 模型未开源**：截至发稿无 GitHub 仓库，复现门槛高；Samsung 内部实验受限可理解，但限制独立验证。

### 可信评估

#### Artifact 可获取性
- **代码**: 未开源（截至 2026-04，无 GitHub 仓库；论文与 Samsung Research blog 均未提供链接）
- **模型权重**: 未发布
- **训练细节**: 仅高层描述（lr / batch / GPU / 数据子集名 + DiT 引用 CogACT），但 router FC 维度、DiT block 数、各 head 参数量、loss weight schedule、fine-tune step 数等关键 detail **未披露**
- **数据集**: 部分公开（pre-train: Open X-Embodiment 的 Fractal + BridgeDataV2，公开；fine-tune: FurnitureBench One-Leg 公开 + 50 traj 自采 Franka pick-and-place 私有）

#### Claim 可验证性
- ✅ **SIMPLER / FurnitureBench / real-world success rate**：跑了完整 baseline 对比，real-world 80 trial 数量充分
- ⚠️ **"dual action model 架构是核心贡献"**：Table VI 数据更支持 dual-scale weighting 是主增益来源，dual head 单独贡献有限
- ⚠️ **"class token 全局 / register token 局部所以分别喂 arm / gripper head"**：缺乏 attention 可视化 / 特征分析支撑，是 architecture 选择的事后合理化
- ⚠️ **"VLM 浅层 reasoning / 深层 cognition"**：layer 选择的解释更像经验最优，缺独立证据
- ⚠️ **OOD 泛化**：real-world OOD 仅扩展到 unseen background + distractor + novel object，相对 CogACT 提升明显但 absolute drop（91→82）也并不小，泛化边界未充分探索

### Notes

- **"两个 expert head + router" 是个值得 generalize 的模板**：把 timestep 级控制问题按"sub-skill"分解、用 cheap label 训 router、再用 weighting scheme 调和 expert——这个 pipeline 可以推广到 mobile manipulation（locomotion vs manipulation）、bimanual（左手 vs 右手 vs 协调）。但需要更通用的 routing 信号，gripper open/close 这种 task-specific signal 不 scalable。
- **Anti-pattern 警惕**：作者把 dual head + dual latent + dual-scale weighting + dual visual token 一锅端推成 contribution，但 ablation 显示主增益是 weighting。如果只取 weighting 单独发 paper 可能更 clean，更利于 community 复用。这是 method engineering 思路下"堆组件冲 SOTA" 的典型样本。
- **未来值得读**：(1) HybridVLA—它把 diffusion + autoregressive 在一个 LLM 里做了，与 DAM-VLA 双 head 是另一种解法；(2) RoboDual—dual-system 架构与 dual-head 思路相关。

### Rating

**分数**：1 - Archived
**理由**：论文定位为 CogACT 上的 component engineering——Weakness 1 与 Ablation 分析都显示 dual-head 架构（主卖点）单独贡献有限，真正增益来自 dual-scale weighting；加上代码 / 权重未开源（Weakness 6）与 routing 信号仅依赖 gripper 翻转（Weakness 2）的 scale 瓶颈，难以作为后续 diffusion-VLA 工作的 building block 或必比 baseline。不到 Frontier（2）因为没有范式级 framing，也未被主流 follow-up 采纳；相对 π0 / OpenVLA-OFT / HybridVLA 等同档位方法没有独占的 insight，归档备查即可，FurnitureBench / SIMPLER long-horizon 场景时可回查其 dual-scale weighting 设计。
