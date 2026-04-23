---
title: "A Survey on Vision-Language-Action Models: An Action Tokenization Perspective"
authors: [Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, Zhiquan Qi, Yitao Liang, Yuanpei Chen, Yaodong Yang]
institutes: [Peking University Institute for AI, PKU-PsiBot Joint Lab]
date_publish: 2025-07-02
venue: arXiv preprint
tags: [VLA, embodied-reasoning, manipulation]
paper: https://arxiv.org/abs/2507.01925
website:
github: https://github.com/Psi-Robot/Awesome-VLA-Papers
rating: 2
date_added: 2026-04-22
---

## Summary

> [!summary] A Survey on Vision-Language-Action Models: An Action Tokenization Perspective
> - **核心**: 把所有 VLA 模型抽象为同一框架——vision/language 输入经过一串 VLA modules，逐步产生越来越 grounded 的 **action tokens**，最终落到可执行动作；论文的主要区分变成 "用了哪种 action token"。
> - **方法**: 把 action tokens 按具体形式归成 8 类（language description / code / affordance / trajectory / goal state / latent / raw action / reasoning），对每一类综述其代表工作、优缺点、未来方向；再谈 data pyramid 和从 IL→RL、VLA model→VLA agent 的范式演进。
> - **结果**: 提出 "未来 VLA 不会是单一 token 主导，而是 hierarchical 组合"——顶层 language plan + code 做长时 planning 和 logic control，中间层 affordance/trajectory/goal video 提供精细运动表征，底层学习 subtask→raw action 的 end-to-end policy；reasoning 作为 meta-token 贯穿各层。
> - **Sources**: [paper](https://arxiv.org/abs/2507.01925) | [github](https://github.com/Psi-Robot/Awesome-VLA-Papers)
> - **Rating**: 2 - Frontier（对 VLA 领域 landscape 梳理系统且提出有辨识度的 tokenization framework，但 influential_citations=3/65 且 github 已 stale，未见其他工作以本文的 taxonomy 作为 reference frame，分类框架是否会成为社区共识仍待观察）

**Key Takeaways:**
1. **统一框架**：VLA 模型可以被 re-cast 为 "vision+language → 一串 VLA module 产生的 action tokens chain → executable action"，其中 "action token" = VLA module 的可微/功能性输出 + 语义上有意义的中间表示。这与 LLM 里 "language token" 的地位 analogous。
2. **Action Token 八分类**：language description、code、affordance、trajectory、goal state、latent representation、raw action、reasoning，每类是本文 Section 4–11 的一章。
3. **不会有单一 winner**：不同 token 互补，语言适合 long-horizon planning、affordance/trajectory/goal video 适合中间表征、raw action 适合 scale-up 的 end-to-end 学习；建议 **hierarchical** 组合。
4. **Reasoning 是 meta-token**：作者认为未来的 reasoning 不应该只在 language space 做，而应该直接在 action token space 做 chain-of-thought（"人不是只靠语言规划，也靠物理想象"）；加上 adaptive test-time compute。
5. **Data Pyramid**：bottom = web/human video、middle = synthetic/simulation、top = real robot（最稀缺但最 embodiment-aligned）；这三层共同支撑 token 学习的 scalability。
6. **未走完的路**：from IL to RL（real-world reset 成本 + VLM auto reward），from gripper to dexterous hand + 多模态（触觉/听觉），from capability-centric 到 safety-aware。

**Teaser. 同一任务、八种 action token。** 给定同一个 language instruction 和 visual scene，不同 VLA 方案会生成形态各异的 action token——从 "pick up the cup"（language plan）到一段 Python（code）、一个 heatmap（affordance）、一组 waypoints（trajectory）、一张未来图（goal state）、一列 codebook index（latent）、一个 6-DoF pose 向量（raw action）、甚至一段 CoT（reasoning）。这张图是全文 organizing principle。

![](https://arxiv.org/html/2507.01925v1/x2.png)

---

## 统一框架：VLA Module 与 Action Token

**定义**：
- **VLA module**：VLA 里的 "maximal differentiable subnetwork that supports end-to-end gradient flow"，或者 non-differentiable 的 functional unit（如 motion planning）。多个组件若联合优化则算同一 module。
- **Action token**：VLA module 的输出，以及 module 内 "semantically meaningful intermediate representations"（如 latent action、goal image）。

**抽象框架**（Figure 1 的核心）：
$$
\text{Vision+Language inputs} \;\to\; \text{VLA module}_1 \;\to\; \text{token}_1 \;\to\; \text{VLA module}_2 \;\to\; \text{token}_2 \;\to\; \cdots \;\to\; \text{raw action}
$$

每一步 token "grounded and actionable" 程度递增。作者举例：
- **Hi Robot**：fine-tuned PaliGemma 输出 language plan（token 1）→ $\pi_0$-like 模型输出 raw action（token 2）。
- **VoxPoser**：LLM 产生 language plan → LLM+VLM 产生 affordance map → motion planning → raw action，共三类 token。

> ❓ 这个定义的边界有点宽——"功能性 non-differentiable unit" 允许 motion planner 也算 VLA module，加上 "semantically meaningful intermediate representations" 允许非 module 输出（如 VQ-VAE codebook 里的中间状态）也算 action token。这让 taxonomy 能涵盖所有现有工作，但代价是分类门槛低、标准不锐利：几乎任何有中间表示的 pipeline 都能被归进来。好处是统一叙事，坏处是 "action token" 作为概念的 predictive power 不强。

---

## Action Token 八分类（Section 4–11）

### 1. Language Description (§4)

两个子类：**language plan**（subtask 级，如 "pick up the cup"）和 **language motion**（motor 级，如 "move arm forward 75cm"）。

**代表脉络**：
- Zero-shot LLM planning 起步：[[2204-SayCan|SayCan]]、Socratic Models、Language Planner。用 LLM 分解指令，缺陷是无 perceptual grounding。
- 外挂 grounding：SayCan 用 affordance function 重加权、Inner Monologue 加 feedback loop、DoReMi 生成 constraints。
- VLM-based planner：[[2303-PaLME|PaLM-E]] 把 vision/language/robot state 编码到同一多模态输入、EmbodiedGPT 轻量 VLM、ViLa 用 GPT-4V zero-shot。
- 结合 generalist low-level policy：[[2502-HiRobot|Hi Robot]] 和 $\pi_{0.5}$（[[2504-Pi05|Pi0.5]]）—— high-level VLM 输出 free-form language，low-level 是通用 policy，不再局限于预设 skill set。
- Language motion 线：RT-H（[[2403-RTH|RTH]]）引入 "move arm forward" 这种中间层帮助 multi-task data sharing；[[2412-NaVILA|NaVILA]] 用 "move forward 75 cm" 指挥 locomotion。

**优势**：零样本调用 foundation model capability；和 VLM 输出空间天然对齐；可利用 co-training data（PaLM-E、$\pi_{0.5}$ 都验证了 world knowledge transfer）；可解释、好 human-in-the-loop 干预。

**限制**：自然语言本身的 expressiveness 不够做 contact-rich 或 deformable 操作；大模型推理延迟。作者建议 language description 主要做 high-level planning，细粒度交给 affordance/trajectory/goal state。

### 2. Code (§5)

Code as Policies、ProgPrompt、Text2Motion、RoboScript、RoboCodeX 等。LLM 生成带 control flow 的可执行代码片段，调预定义 API 库。

**优势**：有 logical structure、可嵌第三方库、透明可审计。

**Brittleness**：表达力被 API 库边界死锁；符号 grounding 问题——即便代码语法对，也会在 API 预设前提（"surface is dry"）被环境违反时失败；一旦失败后果严重（硬件/对象损坏）。

**未来**：打造综合 API function library（多模态感知 + 推理 + action primitives）；对 API 库做 formal verification；利用代码的可解释性做 interactive debugging。

### 3. Affordance (§6)

四种表示：**keypoints**（精确锚点）、**bounding boxes**（粗 localization）、**segmentation masks**（pixel 级轮廓）、**affordance maps**（dense scene-level heatmap，如 VoxPoser）。

**优势**：spatially grounded、跨 embodiment 泛化好、object-centric；VLM 的 spatial reasoning 能被直接利用。

**限制**：大多数方法仍是 2D，缺 3D 几何和遮挡处理；静态 affordance，不建模时间演化（lift lid 之后从 "openable" 变 "pourable"）；对视觉扰动敏感。

**未来方向**：学 true 3D affordance（NeRF/3DGS/mesh 作为 backbone）；时间动态 affordance；增强鲁棒性和 uncertainty-awareness（输出概率 affordance）。

### 4. Trajectory (§7)

三种形式：**Point Trajectory** $\mathbf{P}\in\mathbb{R}^{T\times K\times 2}$（BEV waypoints 或 end-effector path）、**Visual Trajectory** $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$（路径直接画在像素上，如 RT-Trajectory、HAMSTER）、**Optical Flow** $\mathbf{V}\in\mathbb{R}^{H\times W\times 2}$（像素级 motion field，如 AVDC）。

**动机**：trajectory 可从 off-domain video 自动抽出，缓解 action-labeled data 稀缺；而且 trajectory-conditioned policy 比 language-conditioned policy 在 "wipe table → slide block" 这种 semantically 不同但 motion pattern 相似的任务间泛化好（RT-Trajectory 验证）。

**代表**：ATM（Any-point Trajectory Modeling）用少量 action-labeled data + 大量 action-free video；Im2Flow2Act 不用真机数据，用 human demo video + simulation；LLARVA/ARM4R/Magma 做 trajectory-centric pretraining（与 [[2402-Genie|Genie]] 用 latent action 做 world model 的路数不同，trajectory 是 explicit 可解释的）。

**限制**：2D 缺 3D 信息（用 depth 补救）；point trajectory 只有位置、缺方向，不好做 dexterous；生成模型计算贵、VLM 输出 waypoint 频率低。

### 5. Goal State (§8)

Hierarchical：上层生成模型（DiT/CVAE）合成未来 image/video，下层 policy 条件于 goal 执行。

**单帧**（如 3D-VLA 生成单张 goal image）vs **多帧**（如 FLIP、VPP、Gen2Act 生成 goal video）。

**优势**：数据 scalable（hindsight goal relabeling 自动生成训练对）；可以用 action-free video 预训练生成器；cross-embodiment 友好（训练用人类 goal state）；白盒可解释；可以用 language-image alignment model 做自动评估（FLIP）。

**限制**：overspecification（生成了不必要的细节，policy 过拟合到伪 detail）；inaccuracies（物理不合理/时序不一致）；生成延迟高——AVDC 8-frame video 要 10s，Gen2Act 才 3 Hz，VPP 单步去噪也才 7-10 Hz。随着 Veo 3 等 world model 进步这些可能被缓解。

### 6. Latent Representation (§9)

三阶段 pipeline：**Latent Construction**（无监督建 latent action space）→ **Latent Pretraining**（VLM 学预测 latent）→ **Action Fine-tuning**（latent→executable command）。

三个子类：
- **Vision-based**（[[2402-Genie|Genie]] / LAPA / GO-1 / UniVLA）：VQ-VAE 式建模前后帧视觉变化。
- **Action-based**（QueST）：直接 encode-reconstruct 长度为 $H$ 的 action chunk；依赖 action-label，scalability 受限。
- **Goal-based**（GROOT/GROOT-2/OmniJARVIS）：整条 trajectory 编码成 latent goal vector；主要在 Minecraft 这类虚拟 open-world 验证。

**优势**：能用 internet-scale action-free video + cross-embodiment data；比 raw action 更简单的 VLM pretraining target（UniVLA 用 OpenVLA 4.45% 训练时间达到 comparable 性能）；能融合触觉/听觉等非视觉非语言模态。

**限制**：不可解释——无法像 RT-H 那样 human intervention；granularity、comprehensiveness、task-centric alignment 都是未解。作者因此在未来架构推荐里**暂不纳入 latent**。

### 7. Raw Action (§10)

直接 VLM → 6-DoF pose / gripper state。这一章篇幅最大，按技术路线分六支：

1. **Vision-language feature fusion**：LangLfP、BC-Z（FiLM 调制）。
2. **Transformer-based generalists**：Gato、VIMA、[[2311-LEO|LEO]]、JARVIS-VLA。
3. **Autoregressive Robot VLA**：RT-1、[[2307-RT2|RT-2]]、RT-X（OXE）；开源线 RoboFlamingo → [[2406-OpenVLA|OpenVLA]] → [[2502-OpenVLA-OFT|OpenVLA-OFT]] / TinyVLA / HiRT / [[2503-MoManipVLA|MoManipVLA]]。
4. **Video pretraining + robot fine-tuning**：GR-1、GR-2（video 预训练 → 条件于 video 生成的 plan 来出 action）。
5. **Diffusion-based action chunking**：[[2405-Octo|Octo]]、[[2410-Pi0|π0]]、RDT、CogACT、HybridVLA；动机是 autoregressive 难建模 multi-modal 连续分布，且单步推理限制频率。
6. **Heterogeneous dataset / unified action space**：UniAct（codebook + embodiment-specific MLP head）、[[2503-GR00TN1|GR00T N1]]（embodiment-specific action space + data pyramid）。
7. **最近进展**：Real-Time Chunking（用 flow matching inpainting 解决 chunk 边界不连续）、$\pi_0$-FAST（对 action chunk 做 DCT tokenization，13.2× 压缩且平滑）、$\pi_{0.5}$ with knowledge insulation（backbone 在离散 action + VL data 上预训练，action expert 独立训，梯度不回传防 catastrophic forgetting）。

**核心 trade-off**：raw action 最直接、最少 prior，符合 bitter lesson；但 real-world action data 比 web data 稀少 200,000 倍（作者估算 OXE tokens ≈ LLM 语料的 1/200000），cross-embodiment 迁移差，fine-tuning 容易 catastrophic forgetting。

### 8. Reasoning (§11)

**定义**：explicitly externalized 自然语言 deliberative process，不直接表示动作，而是**增强其他 action token 的生成**——是个 **meta-token**。RAD 用 reasoning 增强 raw action，DriveVLM 用 reasoning 增强 trajectory，ECoT 用 fixed CoT reasoning steps 增强 raw action。

**演化**：CoT prompting → LLM + 外挂感知（Inner Monologue）→ VLM-based（[[2407-ECoT|ECoT]]、RAD）→ 专为 embodied reasoning 训练的 VLM（Cosmos-Reason1 用 GRPO+SFT on physical common sense）。

**优势**：桥接 instruction-action gap；可解释；cross-embodiment（高层 plan 跨机器人一致）。

**限制**：显著拖慢推理；reasoning steps 经常是手工固定的；高质量 reasoning 数据稀缺。

**未来**：更快的 foundation model；更好的 reasoning 数据合成（ECoT 和 RAD 已经尝试自动 pipeline）；**adaptive test-time compute**；**action-token-based reasoning**（不只在语言空间思考）。这是本文的一个重要 vision。

---

## Data Pyramid（§12）

三层：
- **Bottom**：web-scale vision-language + human video（Something-Something V2、Ego4D、Ego-Exo4D、EPIC-KITCHENS-100），最充足但最不 embodiment-aligned。
- **Middle**：合成/仿真数据（MimicGen、RoboCasa、AgiBot Digital World），可扩但 sim-to-real gap。
- **Top**：real robot data（OXE、DROID、$\pi$ dataset、AgiBot World），最稀缺但最 actionable。

OXE 累计 token 估算约是 LLM 语料的 1/200,000，印证了 real action data 是当下最硬的瓶颈。

---

## 推荐架构与趋势（§13）

**Figure 4 的 take-away**：Venn diagram 把 AGI = Embodied AI ⊃ VLA ∪ Hardware ∪ Robotics，把 VLA 放在 Digital AI/Hardware/Robotics 三者交集、且是 AGI 必经之路。

**推荐的 hierarchical 架构**（§13.1 最核心的 normative 立场）：
- **顶层**：language description + code，做长时 planning 和 logic control。
- **中间层**：3D affordance + trajectory + goal video，提供 precise motion 表征。
- **底层**：policy 把中间表征映到 raw action；短期靠 hierarchical 结构 bootstrap data flywheel，长期走 end-to-end subtask→raw action。
- **Reasoning** 贯穿各层，未来从 language-based 变成 action-token-based + adaptive test-time compute。
- **Latent representation** 暂缓——granularity / comprehensiveness / alignment 三道坎没过前，不如 explicit token 好控制。

**其他 normative claims**：
- **VLA model → VLA agent**：加 memory、exploration、reflection；架构从 linear 变 graph-structured；关注 multi-agent + human-agent co-existence。
- **IL → RL**：IL 天花板是人类 demonstrator，RL 能自我探索超越；但 real-world reset 成本高，需要 sample-efficient RL（如 in-context RL）+ VLM 自动生成 dense reward。
- **Hardware**：gripper 必须让位给 dexterous hand；tactile/audition/olfaction 等模态不可再缺。
- **Safety**：embodied AI 错了是 physical harm，不是 chatbot 失言，safety 必须 first-class。
- **Data scarcity**：四个维度不足——总量、模态覆盖、embodiment 兼容、质量（尤其 dexterous 场景）。两条路：更好用 simulation + web data；开发更靠谱的 in-the-wild 数据采集系统。

> ❓ 这个 hierarchical 推荐其实和 [[2504-Pi05|π0.5]] 的架构高度吻合（unified 模型先预测 language subtask 再 continuous action），也符合 [[2503-GR00TN1|GR00T N1]] 的 System 1 / System 2 设计。作者在推结论时似乎部分是在 rationalize 自家体系（PsiBot Joint Lab）和近期 $\pi$-series 的 design choice。但这个方向判断和社区的 consensus 接近，不算 over-fitting。

---

## 关联工作

### 基于
- **Foundation model evolution**（§2）：Transformer → BERT/T5/GPT → VLM 族（BLIP/LLaVA/Qwen-VL/PaliGemma）。PaliGemma 作为 $\pi_0$-series backbone 被多次强调。
- **Action token 的概念溯源**：把 "action token" 类比 "language token" 是本文最核心的 conceptual move，显然是向 LLM 叙事致敬。

### 对比

其他 VLA survey（参考 HuggingFace 同期也出现 arxiv 2508.13073 "Large VLM-based VLAs for Robotic Manipulation" 和 arxiv 2505.04769）。本文的 differentiator 是 **action tokenization** 作为 organizing principle，而非 "按 task 分" 或 "按 backbone 分"。这是个有辨识度的 lens。

### 方法相关

覆盖的代表作包括 [[2204-SayCan|SayCan]]、[[2303-PaLME|PaLM-E]]、[[2307-RT2|RT-2]]、[[2403-RTH|RT-H]]、[[2406-OpenVLA|OpenVLA]]、[[2407-ECoT|ECoT]]、[[2410-Pi0|π0]]、[[2412-NaVILA|NaVILA]]、[[2412-RoboVLMs|RoboVLMs]]、[[2502-HiRobot|Hi Robot]]、[[2502-OpenVLA-OFT|OpenVLA-OFT]]、[[2503-GR00TN1|GR00T N1]]、[[2503-MoManipVLA|MoManipVLA]]、[[2504-Pi05|π0.5]]、GO-1、Genie、LAPA、UniVLA 等；以及 VoxPoser、Code as Policies 这类 code-based 工作。

---

## 论文点评

### Strengths

1. **Organizing principle 有辨识度**：用 "action token 形态" 做分类维度比 "按任务/按 backbone" 更本质，因为它直接刻画了每个方法的 inductive bias 和 data requirement。
2. **分类覆盖面广**：8 类 token + 每类一个 section（带 Table），几乎把 2022-2025 有代表性的 VLA 都收了。做 literature survey entry point 很方便。
3. **有 opinionated 立场**：§13.1 直接给出 hierarchical 架构推荐，并明确暂缓 latent、reasoning 要走 action-token 化、raw action 是长期方向。不是纯 cataloging。
4. **Data pyramid + hardware/safety/RL** 的跨章讨论把 VLA 的 systems-level 瓶颈讲清了——特别是 OXE/LLM 1:200000 的粗略量级对比很有冲击力。
5. **GitHub repo 作为 companion paper list** 方便追踪。

### Weaknesses

1. **"Action token" 定义过宽**（前文 ❓）：既包含可微 module 输出，又包含 module 内中间表示，又包含 non-differentiable unit 的输出——几乎任何 pipeline 都能装进来，辨识度被稀释。
2. **缺对比式实验/ meta-analysis**：全书几乎不做"在同一 benchmark 下不同 token 谁赢" 的定量比较。token 类别之间的 trade-off 都是 qualitative，读者很难判断"如果我只能选一类 token，该选谁"。
3. **Reasoning 章节偏薄**：RAD、ECoT、DriveVLM、Cosmos-Reason1、AlphaDrive 几例说完就结束；而 "action-token-based reasoning"（本文 normative claim 里最新颖的概念）没有具体实例 grounding。
4. **Latent 的暂缓判断**理由薄：UniVLA 已经用 4.45% 训练时间达到 OpenVLA comparable 性能，是个强信号，作者却主要以 "interpretability 差" 为由暂缓推荐，这个权衡没算细账。
5. **Hierarchical 架构推荐 suspiciously 贴合 $\pi_{0.5}$**：作者来自 PKU-PsiBot Joint Lab，这个架构观跟近年 $\pi$-series（Hi Robot、$\pi_{0.5}$）靠得很近，可能有 "合理化近期工作" 的嫌疑，虽然方向判断本身合理。

### 可信评估

#### Artifact 可获取性
- **代码**：N/A（survey）
- **模型权重**：N/A
- **训练细节**：N/A
- **数据集**：N/A；但 github repo 列了完整的 VLA paper list 和所述 dataset 的引用。

#### Claim 可验证性
- ✅ "VLA 可被统一到 module + action token 框架"：定义足够宽可容纳所有现有工作；更像 analytical lens 而非可证伪 claim。
- ✅ "8 种 action token 覆盖现有 VLA"：可在 Table 2-9 每表一个 token 类别的代表作里核对，分类是 exhaustive 的。
- ⚠️ "未来最好的 VLA 是 hierarchical 组合"：作者的 normative 立场，证据主要来自 $\pi_{0.5}$、GR00T N1、RT-H 这几条线的实证；但没排除 "大一统 end-to-end raw-action" 走通的可能（§10.8 作者自己也提到这个对照）。
- ⚠️ "latent representation 因 interpretability 差而不该纳入"：UniVLA 的训练效率是反证，作者处理 superficial。
- ⚠️ "OXE tokens ≈ LLM 语料的 1/200,000"：具体引用 [[2503-GR00TN1|GR00T N1]]，数量级对比意义大于精确值。

### Notes

- 把 "action token" 看作 "language token 的 embodied counterpart"是本文最 generative 的概念——它让 VLA 的很多设计选择（tokenization scheme、auto-regressive vs diffusion、chunk size、DCT 压缩）都可以 map 到 LLM 里的对应技术史。这种类比可以 predictively 用来想下一步：e.g. 如果 raw action 要做 MoE / sparse expert、要做 speculative decoding、要做 distillation，都有 LLM 侧的 prior。
- 但类比也有天花板：language token 的 vocab 是 discrete 且 finite，action token 尤其 raw action 是 continuous + embodiment-specific，$\pi_0$-FAST 的 DCT 表示、action chunking、flow matching 都在处理这个 mismatch。这一点本文没 surface 得足够重。
- 未来可验证的 downstream prediction：若本文 hierarchical 架构判断正确，2026-2027 的主要 VLA SOTA 会明显向 [[2504-Pi05|π0.5]] / GR00T 这类分层式靠近；若 raw-action end-to-end 继续突破（如更大 dataset 出现后 OpenVLA 类 scale up），则作者的 normative 判断需要修正。

### Rating

**Metrics** (as of 2026-04-22): citation=65, influential=3 (4.6%), velocity=6.73/mo; HF upvotes=39; github 520⭐ / forks=18 / 90d commits=0 / pushed 293d ago · **stale**

**分数**：2 - Frontier
**理由**：总 citation 65、velocity ~6.7/mo、HF 39 upvotes，作为 survey 流量不错但不爆；**influential_citations 只有 3/65 = 4.6%，显著低于典型 ~10%**，说明多数引用是 "landmark reference"（当一本 VLA 教科书式引用），而本文提出的 "action tokenization" 分类框架还没被后续工作实质继承作为 reference frame——这是它够不上 3-Foundation 的主要信号。另一方面，github repo 在 2025-07 之后**已 stale**（293d 无 push，90d commits=0），survey 的 paper list 更新停滞，会逐步降低长期价值。但作为 2025 年 VLA 领域最系统的调研之一、且 opinionated 给出了 hierarchical 架构 recommendation，对研究者理解 landscape 价值高，配 2-Frontier 适当。
