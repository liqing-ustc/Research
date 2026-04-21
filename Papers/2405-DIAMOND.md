---
title: "Diffusion for World Modeling: Visual Details Matter in Atari"
authors: [Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, François Fleuret]
institutes: [University of Geneva, University of Edinburgh, Microsoft Research]
date_publish: 2024-05-20
venue: NeurIPS 2024 (Spotlight)
tags: [world-model, RL, diffusion-policy]
paper: https://arxiv.org/abs/2405.12399
website: https://diamond-wm.github.io/
github: https://github.com/eloialonso/diamond
rating: 3
date_added: 2026-04-20
---

## Summary

> [!summary] Diffusion for World Modeling: Visual Details Matter in Atari
> - **核心**: 用 EDM-flavored diffusion 直接在像素空间建模环境动力学，避免 discrete-latent world model 丢失对 RL 至关重要的视觉细节
> - **方法**: EDM 预条件 + U-Net 在 channel 上拼接历史帧、通过 AdaGN 注入 action，3 个 denoising steps，配单独的 reward/termination CNN-LSTM，agent 用 actor-critic 在 imagination 内训练
> - **结果**: Atari 100k mean HNS 1.46（agents-trained-entirely-in-WM 的新 SOTA，11 项 superhuman）；同一架构 scale 到 CS:GO Dust II，得到一个可用键鼠交互的 10Hz neural game engine
> - **Sources**: [paper](https://arxiv.org/abs/2405.12399) | [website](https://diamond-wm.github.io/) | [github](https://github.com/eloialonso/diamond)
> - **Rating**: 3 - Foundation（pixel-space diffusion WM 的奠基工作，EDM-vs-DDPM 的 c_skip 稳定性 insight 对长时序自回归生成普适，NeurIPS 2024 Spotlight + 完整开源 + CS:GO neural game engine 跨数量级 scale 证据）

**Key Takeaways:**
1. **Pixel-space diffusion 是 discrete-latent WM 的可行替代**: DreamerV3/IRIS/STORM 等主流路线都把观测压成 discrete tokens，DIAMOND 证明保留像素能在 100k 样本制约下反而提分，关键来自对小尺寸 visual detail（如 Asterix 里的奖励/敌人区分）的保真。
2. **EDM 而非 DDPM 是稳定性的来源**: 在 ≤10 NFE 的低预算下 DDPM 自回归 rollout 几步就 OOD，EDM 即便单步也稳定数百步。原因是 EDM 的 c_skip 加权使高噪声时模型直接预测 clean image（而非 DDPM 的 noise），低 NFE 下 score 估计不退化。
3. **3 NFE 已足够**: 单步 denoising 在 deterministic 游戏（Breakout）即可，但部分可观测时（Boxing）单步会插值多模式产生模糊，多步采样把生成驱向单一 mode。Sweet spot n=3，比 IRIS 的 16 NFE 还低。
4. **WM 同架构可作 neural game engine**: 把 4M 参 U-Net scale 到 381M（含 51M upsampler），仅靠 87h 静态 CS:GO 录像训练，就得到 10Hz 可玩的 Dust II simulator——证明该路线超出 Atari 的 toy regime。
5. **Frame-stacking 是当前 memory bottleneck**: 没有时间维 transformer，绕墙/失视容易 forget state（出现新武器或新地图）。作者明确指出向 DiT-style 时序 attention 升级是下一步。

**Teaser. DIAMOND 在 imagination 中 unroll 的示意：横轴是环境时间 t，纵轴是 denoising 时间 τ，policy 在世界模型预测的下一帧上继续选择 action 形成自回归。**

![](https://arxiv.org/html/2405.12399v2/extracted/5965885/images/noise/noise_0_level_1.0.png)

---

## 1. Motivation：为什么 pixel-space diffusion

近年 world model 路线（DreamerV2/V3、IRIS、TWM、STORM）几乎一致选择 **discrete latent**（VQ-VAE / RSSM 的 categorical state），理由是离散化能减弱多步预测中的 compounding error。代价是 **lossy compression**：对 pixel 细节敏感的任务（自动驾驶里的远处行人/红绿灯，Atari 里几像素大的奖励 vs 敌人）会把决策依据丢掉。

Diffusion 同期在图像生成上击败了 token-based 方法，且天然支持灵活 conditioning 与 multi-modal 分布建模——这两条性质对 WM 都是关键：action 应该真正 condition 出环境响应（credit assignment 才靠谱），而 partially observable 环境本身就是多模态。DIAMOND 就是把这套搬到 WM。

---

## 2. Method

### 2.1 Score-based diffusion 的简化训练目标

把环境动力学写成条件 diffusion $p(\mathbf{x}_{t+1}\mid\mathbf{x}_{\le t}, a_{\le t})$。沿用 score-based 的连续时间 SDE 视角，使用 affine drift 让 perturbation kernel 是 Gaussian，目标坍缩为 $L_2$ 重建：

$$
\mathcal{L}(\theta)=\mathbb{E}\left[\|\mathbf{D}_{\theta}(\mathbf{x}_{t+1}^{\tau},\tau,\mathbf{x}_{\le t}^{0},a_{\le t})-\mathbf{x}_{t+1}^{0}\|^{2}\right]
$$

**符号说明**：$\tau$ 是 diffusion time（与环境时间 $t$ 区分），$\mathbf{D}_\theta$ 是 denoiser，$\mathbf{x}_{t+1}^\tau$ 是对真实下一帧 $\mathbf{x}_{t+1}^0$ 加 $\tau$ 级噪声后的样本。

### 2.2 关键决策：选 EDM 而非 DDPM

DIAMOND 在 EDM 的预条件框架下定义 $\mathbf{D}_\theta$：

$$
\mathbf{D}_{\theta}(\mathbf{x}_{t+1}^{\tau},y_{t}^{\tau})=c_{\text{skip}}^{\tau}\,\mathbf{x}_{t+1}^{\tau}+c_{\text{out}}^{\tau}\,\mathbf{F}_{\theta}(c_{\text{in}}^{\tau}\,\mathbf{x}_{t+1}^{\tau},y_{t}^{\tau})
$$

其中 $c_{\text{skip}}^\tau=\sigma_{\text{data}}^2/(\sigma_{\text{data}}^2+\sigma^2(\tau))$。当噪声主导（$\sigma\gg\sigma_{\text{data}}$）时 $c_{\text{skip}}\to 0$，目标退化为预测 **clean image**；当噪声很小时 $c_{\text{skip}}\to 1$，目标变为预测 **noise**。这种自适应 mixing 是后续稳定性的根因。

DDPM 全程预测 noise，在高噪声时模型容易学成恒等映射（$\xi_\theta(\mathbf{x}^\tau)\to \mathbf{x}^\tau$），起步 step 的 score 估计极差，autoregressive rollout 中误差迅速积累。这是 5.1 节实验证实的核心 finding。

### 2.3 架构与采样

- **Backbone**：标准 2D U-Net；过去 $L$ 帧观测沿 channel 维与当前噪声帧拼接，actions 通过残差块的 **adaptive group norm** 注入。
- **Sampler**：Euler 一阶足够，不需要 Heun 的额外 NFE 也不需要 stochastic 注入。**$n=3$ denoising steps** 是默认。
- **Reward/Termination**：单独 CNN-LSTM head $R_\psi$，处理 partial observability；不与 diffusion 合并，作者显式说明合并 representation extraction "并非 trivial"。
- **Policy/Value**：CNN-LSTM 共享 backbone + actor/critic head；REINFORCE + value baseline；value 用 $\lambda$-returns Bellman error。
- **训练循环**：collect → 更新 WM（用累积所有数据）→ imagination 内训 RL → 重复（同 SimPLe / IRIS）。

---

## 3. Atari 100k 实验

100k 行动 ≈ 2 小时人类 gameplay。每游戏 5 个 seed，单 RTX 4090 约 2.9 天/run，总计 ~1.03 GPU-year。

**Table 1. 26 个游戏的 returns 及聚合 HNS（节选关键行；完整对比见原文）**

| Game            | IRIS    | DreamerV3 | STORM   | DIAMOND     |
| --------------- | ------- | --------- | ------- | ----------- |
| Asterix         | 853.6   | 932.0     | 1028.0  | **3698.5**  |
| Boxing          | 70.1    | 78.0      | 79.7    | **86.9**    |
| Breakout        | 83.7    | 31.0      | 15.9    | **132.5**   |
| CrazyClimber    | 59324.2 | 97190.0   | 66776.0 | **99167.8** |
| Pong            | 14.6    | 18.0      | 11.3    | **20.4**    |
| RoadRunner      | 9614.6  | 15565.0   | 17564.0 | **20673.2** |
| #Superhuman (↑) | 10      | 9         | 10      | **11**      |
| Mean HNS (↑)    | 1.046   | 1.097     | 1.266   | **1.459**   |
| IQM (↑)         | 0.501   | 0.497     | 0.636   | **0.641**   |

DIAMOND 在 11 个游戏 superhuman，mean HNS 1.46 是 "agents trained entirely within a WM" 的新 SOTA。注意 IQM 与 STORM 接近，说明 mean 主要由几个 visual-detail-sensitive 游戏（Asterix、Breakout、RoadRunner）拉起来——这与作者关于 visual detail 的 narrative 一致。

> ❓ Mean 受 Asterix/Breakout 极值显著拉动，IQM 与 STORM 的差距远没 mean 那么大。"agents trained in a WM 的新 SOTA" 这个 headline 在 IQM 视角下并不显著——更准确的 claim 是 "在依赖 visual detail 的游戏上显著领先"。

---

## 4. Analysis

### 4.1 EDM vs DDPM 的 rollout 稳定性

固定网络结构、共享 100k Breakout expert frames 训两个 variant，autoregressive rollout 至 $t=1000$，扫 $n\le 10$。

**Figure 3a. DDPM rollout 在低 denoising step 下严重 compounding error，世界模型很快漂移出分布。**

![](https://arxiv.org/html/2405.12399v2/extracted/5965885/images/figure__karras_vs_ddpm__ddpm.png)

EDM 即使 single-step 也保持稳定数百步——这是 5.1 节的核心证据，也是为什么 DIAMOND 用 EDM 而非更"主流"的 DDPM。

### 4.2 Denoising steps 的下界由多模态决定

Breakout 是确定性转移，单步 OK。Boxing 里黑方动作不可知 → 观测分布多模态 → 单步 denoising 取期望产生模糊插值；多步 sampling 把生成驱向单一 mode 得到锐利图像。

**Figure 4. Boxing 上 single-step（顶）vs multi-step（底）。注意白方动作已知，单步多步对白方位置预测都对；只在不可知的黑方上出现 blurry interpolation。**

![](https://arxiv.org/html/2405.12399v2/extracted/5965885/images/figure__blurry_boxing.png)

最终选 $n=3$：覆盖多模态情况，又远低于 IRIS 的 16 NFE。

### 4.3 与 IRIS 的视觉一致性对比

同样 100k expert frames 训 DIAMOND vs IRIS。IRIS 的 imagined trajectory 中"奖励变敌人 / 敌人变奖励"等 inconsistency 时有发生（仅几像素差异，但对 RL 灾难性）。DIAMOND 在 64×64 同分辨率下更 faithful，且参数更少、训练更快、NFE 3 vs 16。

**Figure 5 (IRIS portion). IRIS 的 token-based rollout 容易在小尺寸语义元素上产生跨帧不一致。**

![](https://arxiv.org/html/2405.12399v2/extracted/5965885/images/figure__iris_vs_diamond__iris.png)

---

## 5. Scale 到 CS:GO：从 toy 到 neural game engine

NeurIPS 接收后追加的 §6。在 5M 帧 / 87h Dust II 人类 gameplay（Pearce et al. CS:GO 数据集）上训 WM only，**无 RL**。

- 分辨率：world model 在 56×30 上做，再用一个 51M 参数的 diffusion **upsampler** 还原到 280×150
- 参数量：U-Net 通道 scale 到 381M（含 upsampler）
- Compute：12 days on RTX 4090
- Sampling：动力学模型仍 3 步；upsampler 引入 **stochastic sampling + 10 步**，权衡画质与延迟，整体 **10 Hz on RTX 3090**

**Figure 6. 玩家用键鼠在 DIAMOND 的 CS:GO 世界模型里实时游玩的截屏（项目页有完整视频，比静态图更有说服力）。**

![](https://arxiv.org/html/2405.12399v2/extracted/5965885/images/csgo_grid.png)

**Video. CS:GO Dust II 实时交互的 demo grid（项目页主视频）。**

<video src="https://diamond-wm.github.io/static/videos/grid.mp4" controls muted playsinline width="720"></video>

观察到的有趣 emergence 与 failure：
- 不常去的区域更容易 OOD
- 接近墙面/失去视野 → 模型遗忘当前 state，可能"重新生成"一把武器或一片地图（**memory bottleneck**：frame stacking 没有长程记忆）
- 模型把"jump 改变几何"这一一次性变化 generalize 成连续多次 jump（训练数据里多连跳几乎不出现）——既算 generalization 又算 hallucination

---

## 6. Limitations（作者自陈）

1. 评测集中在离散控制；连续控制未做。
2. Frame stacking 是最 minimal 的 memory，向时间维 transformer（DiT 风格）升级是显然的下一步——但作者初步实验里 cross-attention 不如 frame stacking。
3. Reward/termination 没集成进 diffusion；要做需要从 diffusion model 提 representation，作者认为不 trivial 且会过度复杂化。

---

## 关联工作

### 基于
- **EDM (Karras et al., 2022)**: 预条件 + log-normal 噪声采样的 diffusion training framework，是 DIAMOND 稳定性的根因。
- **Score-based SDE (Song et al., 2021)**: 连续时间 score-based diffusion 的统一框架。

### 对比（同代 WM-based RL agents on Atari 100k）
- **IRIS**: discrete autoencoder + autoregressive transformer；DIAMOND 直接对标，Fig 5 的 inconsistency 证据主要冲它。
- **DreamerV3**: RSSM + categorical latents；fixed hyperparams 的强基线。
- **STORM**: DreamerV3 的 transformer 化变体，IQM 与 DIAMOND 接近。
- **TWM / SimPLe**: 早期 / 中期对比 baseline。
- [[2411-WorldModelSurvey|World Model Survey]]: 系统综述同期 WM-based RL 路线，DIAMOND 是 pixel-space diffusion 那一支的代表。

### 方法相关 / 同时期
- [[2408-GameNGen|GameNGen]]: concurrent work，diffusion 做 DOOM 的 neural simulator；与 DIAMOND CS:GO 部分思路高度重合。
- [[2402-Genie|Genie]]: 从图像 prompt 生成 playable 2D platformer；与 DIAMOND 都属"playable WM as foundation engine"路线。
- **DiT / Sora**: transformer-based diffusion，DIAMOND 在 §8 明确指出向这类时序架构升级是下一步。

---

## 论文点评

### Strengths

1. **Problem framing 很到位**: "discrete latent 丢小尺寸细节" 不是新观察，但用 Atari 100k 的 mean HNS 把它做成可量化的 head-to-head 对比，再用 IRIS 跨帧 inconsistency 的可视化把因果链补完，是教科书级别的"motivation → method → evidence"闭环。
2. **EDM vs DDPM 的 ablation 是真 insight**: §5.1 不只是 "我们换了个 sampler"，而是从 c_skip 的训练目标推出"低 NFE 下 DDPM 的 score 估计为何必然崩"。这条结论对所有想用 diffusion 做长时序自回归生成的人（video gen、agent rollout）都通用。
3. **Scale 到 CS:GO 的 §6 把 toy 标签摘掉**: Atari 单独看会被诟病为"小玩具"，CS:GO 的 10Hz neural game engine 直接证明同架构能跨数量级 scale，且与 GameNGen / Genie 形成同代竞争。
4. **完整 release**: code + 5 seeds × 26 games 的 checkpoint + playable WM。这种级别的 reproducibility 在 NeurIPS WM 论文里少见。

### Weaknesses

1. **"agents trained entirely in WM 的 SOTA" 是修饰过的口径**: 排除了 BBF / EfficientZero（model-free + tree search 等），这些方法在 100k 上的绝对分数远高于 DIAMOND。论文承认了这一点但 headline 里没体现，对快速浏览 abstract 的读者会形成 SOTA 错觉。
2. **mean 与 IQM 的 gap 没有充分讨论**: mean 1.46 vs STORM 1.27 看起来 ~15% 提升，但 IQM 0.641 vs 0.636 几乎打平。说明 DIAMOND 的优势集中在少数几个游戏，"普适提升"的叙事比实际证据强。
3. **没和 GameNGen 直接对比**: §7 只一句 "concurrent work"。两者方法非常接近（都是 diffusion 做 game engine），定量对比的缺失让"diffusion as game engine"路线的内部 trade-off 不清晰。
4. **Memory 是真硬伤但 punt 给未来**: CS:GO 里"绕墙忘记 state" 是任何想用此类 WM 做 long-horizon planning 的致命问题，frame stacking 的根本性局限在 §8 只用一段带过。

### 可信评估

#### Artifact 可获取性
- **代码**: inference + training，开源仓库，含 Atari 主分支与 CS:GO 分支
- **模型权重**: HF Hub `eloialonso/diamond` 提供 Atari 100k pretrained world models + policies；CS:GO WM 在 `csgo` 分支可下载 playable
- **训练细节**: 完整 — 超参、架构、RL 目标分别在 Appendices D/E/F；明确 1.03 GPU-year 总计算
- **数据集**: Atari 100k 标准 benchmark；CS:GO 用 Pearce et al. 公开的 5.5M frame Dust II 数据集（链接已给）

#### Claim 可验证性
- ✅ **Atari 100k mean HNS 1.46 (5 seeds)**：表 1 给出每游戏分数 + bootstrap CI（Fig 2），可与 IRIS/STORM/DreamerV3 公开复现对齐。
- ✅ **EDM 比 DDPM 在低 NFE 下稳定**：Fig 3 + Appendix K 的定量 compounding error 分析，公开 code 可复现。
- ✅ **CS:GO WM 10Hz 可玩**：playable demo 已 release，可独立验证。
- ⚠️ **"Visual detail 的改善是 Asterix/Breakout 提分的原因"**：相关性 + 定性观察（Fig 5 IRIS inconsistency），但没有 controlled ablation——比如在 IRIS 的 token 表示上 mock 出"无 inconsistency"再测 RL 性能。归因偏向 narrative。
- ⚠️ **"agents trained entirely within a WM 的 SOTA"**：技术正确但口径限制掉了真正的 100k SOTA（BBF/EfficientZero）。
- 无明显营销话术 ❌。

### Notes

- §5.1 的 EDM-vs-DDPM 解释是这篇最有 transfer value 的部分。它说明的不是"diffusion 适合 WM"，而是"低 NFE 自回归生成必须用 EDM 风格的 c_skip mixing 才稳定"。这个原理对任何想把 image diffusion 当 dynamics model 用的工作（VLA 的 future prediction、video generation 的长 horizon rollout）都适用。
- **memory bottleneck 是这条 line 后续工作的真正题眼**：DIAMOND 已经证明 spatial 保真足够好，但 frame stacking 无法支撑 long-horizon planning。要做 useful 的 agent-relevant WM，下一步必然是 temporal attention + 更长 context（或 explicit memory token）。值得跟踪 DiT-style 时序架构在 WM 上的尝试。
- **跟 GameNGen 的 missing 对比是个机会**：两者都是 diffusion-based game simulator，sample efficiency / inference cost / horizon stability 的 head-to-head 还没有人做。是个轻量但有信息量的 follow-up 实验。
- ❓ 把 reward/termination 集成进 diffusion 真的"过度复杂"吗？现在的 separate CNN-LSTM 意味着 reward 不享受 diffusion 学到的视觉表征，可能在 sparse reward 任务上吃亏。值得验证。

### Rating

**分数**：3 - Foundation
**理由**：DIAMOND 是 pixel-space diffusion world model 这一支的奠基工作，NeurIPS 2024 Spotlight，且 §5.1 的 EDM-c_skip 稳定性 insight（Strengths 2）具备跨方向的可复用性——任何长时序 image-space 自回归生成（video gen、VLA future prediction）都能直接借鉴。CS:GO 10Hz neural game engine 的跨数量级 scale 证据与完整开源 release 使其成为 world-model / diffusion-for-RL 路线的 de facto 必引；不是 2 - Frontier 是因为它已改变了"discrete latent 是 WM 必由之路"的社区共识判断，而非仅是一个 SOTA baseline。
