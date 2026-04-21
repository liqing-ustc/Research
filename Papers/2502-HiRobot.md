---
title: "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"
authors: [Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn]
institutes: [Physical Intelligence]
date_publish: 2025-02
venue: arXiv
tags: [VLA, instruction-following, task-planning, manipulation, embodied-reasoning]
paper: https://arxiv.org/abs/2502.19417
website: https://www.pi.website/research/hirobot
github:
rating: 2
date_added: "2026-03-24"
---
## Summary

> [!summary] Hi Robot: Open-Ended Instruction Following with Hierarchical VLAs
> - **核心**: 把 [[2410-Pi0|π0]] 这类 VLA 套进一个 System1/System2 的两层 VLM 结构，由 high-level VLM "对自己说话" 把 open-ended prompt 拆成原子 skill，喂给 low-level VLA 执行
> - **方法**: 同一个 [[2410-Pi0|π0]] backbone（PaliGemma-3B）训两个 policy；high-level 的训练数据靠 generator VLM 在 teleop 切片上合成多轮 user-robot 对话补足
> - **结果**: 在 table bussing / sandwich making / grocery shopping 三个真机任务上 instruction accuracy 平均比 GPT-4o high-level + π0 baseline 高 40%+；synthetic data 与 hierarchy 各自 ablation 都显著
> - **Sources**: [paper](https://arxiv.org/abs/2502.19417) | [website](https://www.pi.website/research/hirobot)
> - **Rating**: 2 - Frontier（hierarchical VLA + synthetic interaction data 的代表工作，但代码/权重未放出，且已被后续 [[2504-Pi05|π0.5]] 整合覆盖，尚未到 foundation 级别）

**Key Takeaways:**
1. **Hierarchy 是 prompt-VLA 解耦的工程化答案**：low-level VLA 不必学会理解 "I'm allergic to pickles"，high-level VLM 把它翻成 "pick up cheese / pick up lettuce / ..." 这种 atomic command，VLA 只要听得懂自己训过的指令就够了。
2. **Synthetic interaction data 是 hierarchy 真正能 work 的关键**：直接用 demo label 训 high-level 学不到 multi-turn correction、constraint satisfaction、verbal response；ablation 证明去掉合成数据，模型完全无法响应 mid-task 反馈。
3. **High-level 和 low-level 共享同一个 PaliGemma-3B backbone 但独立训练**——high-level 用 cross-entropy 预测下一个 skill label + 可选 verbal utterance，low-level 沿用 [[2410-Pi0|π0]] 的 flow matching action expert。两层在 inference 时串联（1 Hz / 10-50 Hz）。
4. **Hi Robot 在 instruction-following 上吊打 GPT-4o**：fine-tuned 3B PaliGemma 在 situated grounding（"that's not trash" 指代当前夹爪里的物体）上比 GPT-4o 强得多，说明 generic 大 VLM 缺的是 robot-context 的对齐，而非 raw reasoning 能力。

---

## Problem & Motivation

**目标**：让通用机器人能处理 open-ended、多轮、带约束的自然语言指令，例如 *"Could you make me a vegetarian sandwich? I'm allergic to pickles"*、*"that's not trash"*、*"I also want a KitKat"*。

现有 VLA（包括 π0）的 prompt 接口被训成 atomic command 形式（"pick up the cup"），有三类失效：

1. **复杂 open-ended 指令**：含约束、偏好、否定，VLA 没有训练数据覆盖
2. **mid-task situated correction**：用户 *在执行过程中* 给反馈，需要把语言 ground 到当前 image 里的具体物体
3. **multi-step deliberation**：long-horizon 任务需要 System 2 慢速推理，VLA 的反应式行为做不到

**核心 insight**：把 policy 拆成两层——high-level VLM 做 "thinking"（说出下一步该干啥），low-level VLA 做 "acting"（怎么干）。论文的比喻是脑子里那个 "little voice"。

**Figure 1. 三种 open-ended 指令场景**——multi-stage instruction、real-time correction、unseen long-horizon、verbal response。
![](https://arxiv.org/html/2502.19417v2/x1.png)

---

## Method

### 形式化

设 observation $\mathbf{o}_t = [\mathbf{I}^1_t, ..., \mathbf{I}^n_t, \ell_t, \mathbf{q}_t]$，包含多摄像头图像、prompt、robot configuration。policy 输出 action chunk $\mathbf{A}_t = [\mathbf{a}_t, ..., \mathbf{a}_{t+H-1}]$。

Hi Robot 把这个 $p(\mathbf{A}_t | \mathbf{o}_t)$ 拆成：

$$
p^{\text{hi}}(\hat{\ell}_t \mid \mathbf{I}^1_t,...,\mathbf{I}^n_t, \ell_t)
$$

$$
p^{\text{lo}}(\mathbf{A}_t \mid \mathbf{I}^1_t,...,\mathbf{I}^n_t, \hat{\ell}_t, \mathbf{q}_t)
$$

**含义**：high-level $p^{\text{hi}}$ 把 open-ended prompt $\ell_t$ 翻译成 intermediate skill label $\hat{\ell}_t$（如 *pick up one piece of lettuce*）。low-level $p^{\text{lo}}$ 接口与标准 VLA 完全一致，只是 prompt 换成了 $\hat{\ell}_t$。当任务足够简单时可以直接 $\hat{\ell}_t = \ell_t$ 退化回普通 VLA。

> 这种解耦的代价是 high-level 与 low-level 不共享梯度，但好处明显：low-level 训练目标稳定，high-level 可以独立用语言数据迭代。

### 两层架构总览

**Figure 2. 整体架构**——high-level VLM 处理 prompt + 多视角图像，输出 intermediate command（可附带 verbal utterance）；low-level π0 接收该 command + 图像 + state，输出 action chunk。
![](https://arxiv.org/html/2502.19417v2/x2.png)

- **High-level policy**：fine-tuned PaliGemma-3B；约 1 Hz 或在用户输入时立即触发；可选地输出 utterance $u_t$，经 TTS（Cartesia API）说出来后从 $\hat{\ell}_t$ 中剥离再喂给 low-level
- **Low-level policy**：[[2410-Pi0|π0]] VLA（PaliGemma-3B + flow-matching action expert）；10-50 Hz 控制
- **触发策略**：高层 1 秒一次重新推理，或用户输入立即触发——简单但 reactive

### Synthetic Data Generation（核心创新）

**痛点**：teleop demo 只能给 *(image, atomic skill label)* 这种 tuple。但 high-level 真正要学的是 *(image, open-ended user prompt)* → *(verbal response, atomic skill label)* 这种映射，这种数据在 demo 里**根本不存在**。

**方法**：用一个 generator VLM $p^{\text{gen}}$ 从 demo 切片自动合成多轮对话。

1. 收集 teleop demo $\mathcal{D}_{demo}$，并按时间切成 1-3 秒的 short skill，得到 labeled tuples $\mathcal{D}_{labeled} = \{(\hat{\ell}_t, \mathbf{I}^1_t, ..., \mathbf{I}^n_t)\}$
2. 用 $p^{\text{gen}}$ 在视觉 context 与 skill label 条件下 *反向* 生成可能导致这个 skill 的 user prompt $\ell_t$ 和 robot utterance $u_t$（例如对 "pick up KitKat" 反推出 "Can you get me something sweet?"）
3. 覆盖多种 scenario：constraint、preference、negation、mid-task interjection、clarification request
4. 产出 $\mathcal{D}_{syn}$，与 $\mathcal{D}_{labeled}$ 联合用 next-token cross-entropy 训 high-level
5. Low-level 用 $\mathcal{D}_{labeled} \cup \mathcal{D}_{demo}$ 配 flow-matching loss 训，沿用 [[2410-Pi0|π0]]

**Figure 3. 数据生成 pipeline**——左侧 teleop demo 切片成短 skill，中间 generator VLM 在视觉 + skill label 条件下合成多轮对话，右侧 high-level 在合成 + 真实数据上联合训练。
![](https://arxiv.org/html/2502.19417v2/x3.png)

> ❓ Generator VLM 用的是哪个？合成数据的质量如何评估？论文没有量化合成数据本身的多样性或正确率，只通过下游 ablation 间接证明它有用。这是 synthetic data pipeline 通常缺的环节。

### 用户交互

- 文本或语音（STT 转文本）输入
- 用户介入立即触发 high-level 重推理
- High-level 可输出 verbal $u_t$（"Got it, skipping tomatoes"）经 TTS 反馈，*同时* 把命令传给 low-level
- 用户也可以信号 "继续" 让 high-level 切回上一个 outstanding command（论文中是手动信号，没有自动 detection）

### 训练超参

- 都从 PaliGemma-3B fine-tune，全 unfreeze
- AdamW, $\beta_1=0.9$, $\beta_2=0.95$, no weight decay, grad clip 1.0, EMA 0.999
- High-level 训练 ~2h on 8×H100
- Low-level 沿用 π0 的 flow matching pipeline

---

## Experiments

### 任务

**Figure 4. 三个评测域**——Table Bussing（UR5e 单臂）、Sandwich Making（双臂 ARX）、Grocery Shopping（双臂 mobile ARX）。每个任务都设计了复杂 prompt + mid-task feedback + interjection。
![](https://arxiv.org/html/2502.19417v2/x4.png)

| 任务 | 平台 | 关键 prompt 类型 |
|---|---|---|
| Table bussing | UR5e 单臂 | "bus only yellowish things"、"this is not trash"、"leave it alone" |
| Sandwich making | Bimanual ARX | "vegetarian, allergic to pickles"、"that's all, no more" |
| Grocery shopping | Mobile bimanual ARX | "something sweet"、"prepping for a movie night"、"I also want KitKat" |

### Baseline

- **GPT-4o high-level + π0 low-level**：把 high-level 换成现成大 VLM
- **Flat VLA (π0)**：直接把 open-ended prompt 喂给 π0
- **Flat VLA + synthetic data**：flat 但用同样合成数据训
- **Hi Robot w/o synthetic**：去掉合成数据
- **Expert human guidance**：人在回路给 high-level command（上界）

### 主结果

**Figure 5. 与 baseline 的定量对比**——Hi Robot 在三个任务上 instruction accuracy 平均比 GPT-4o 高 40%+，并接近 expert human guidance 的上界。
![](https://arxiv.org/html/2502.19417v2/x5.png)

**Figure 6. 定性对比**——GPT-4o 的失败模式：(a) 误识物体（"pick up bermuda triangle"）、(b) 跳过 subtask、(c) 忽略 user intent。Hi Robot 一致输出与机器人当前行为对齐的指令。
![](https://arxiv.org/html/2502.19417v2/x6.png)

论文总结的四个发现：

1. **Open-ended instruction following**：Hi Robot 能正确执行 "I'm allergic to pickles"；GPT-4o 在物理交互开始后丢失 context，开始胡乱命名（everything 变成 "plate" 或 "spoon"）
2. **Situated reasoning + real-time feedback**：mid-task "leave the rest"、"I also want KitKat" Hi Robot 能更新；GPT-4o 状态混乱（夹爪还占着就让它去拿别的）
3. **跨任务跨平台**：Hi Robot 在三种 embodiment 上都能处理 dynamic constraint
4. **Expert guidance gap**：Hi Robot 接近但仍低于 expert human guidance——主要 gap 来自 low-level 失败（OOD 物体配置、drop 后 recovery）

### Ablation

**Figure 7. 合成数据 ablation**——去掉合成数据后，模型忽略 clarification（"this is not trash"）、不遵守 constraint（仍放 pickle），instruction accuracy 大幅下降。
![](https://arxiv.org/html/2502.19417v2/x7.png)

**Figure 8. Hierarchy ablation**——同样的合成数据下，flat policy 仍退化到 default behavior（清掉所有物体、忽略 partial instruction）。证明 hierarchy 的好处不只是来自数据，而是 reasoning step 的显式分离。
![](https://arxiv.org/html/2502.19417v2/x8.png)

> 这两个 ablation 互相印证：synthetic data 提供 prompt 多样性，hierarchy 提供 reasoning 解耦。**两者都不可或缺**。

### 推理延迟（Appendix B.3，RTX 4090）

**Low-level Per-Step**

| Component | Time (ms) |
| --- | --- |
| Image encoding | 14 |
| Observation processing | 32 |
| Action prediction (×10) | 27 |
| Total (on-board) | 73 |
| Total (off-board + WiFi) | 86 |

**High-level（单次 decode）**

- RTX 4090: 47 ms prefill + 13.2 ms decode
- H100: 17.3 ms prefill + 5.7 ms decode

Action chunking 让有效控制频率到 50 Hz；high-level 1 Hz 完全够用。

### Failure Cases（Appendix C.4）

- High-level：long-context reasoning 弱（无 memory）
- Low-level：proximity bias（夹爪靠近 cheese 时即使用户说 lactose intolerance 仍可能拿）；drop 后 OOD recovery 差

---

## 关联工作

### 基于
- [[2410-Pi0|π0]]：直接作为 low-level policy；Hi Robot 不修改 π0 内部，只换 prompt 来源
- PaliGemma-3B：两层共用的 VLM backbone，open-source，3B 参数

### 相关
- [[2504-Pi05|π0.5]]：Physical Intelligence 后续工作，把 hierarchy 思路 + open-world generalization 整合进单个模型；Hi Robot 是这条路线的中间产物
- [[2511-PiStar06|π*0.6]]：Physical Intelligence 最新工作，用 RL 给 VLA 加 self-improvement，正好补 Hi Robot 缺的 low-level 失败 recovery
- [[2401-MobileALOHA|Mobile ALOHA]]：Hi Robot 的 mobile manipulation 平台基础

### 方法相关
- System 1 / System 2（Kahneman）：hierarchy 设计的认知科学动机
- LLM-based task planning（[[2204-SayCan|SayCan]], ProgPrompt 等）：Hi Robot 的 high-level 在概念上类似 task planner，但用真机 grounding 而非 affordance scoring
- VLM 数据合成（self-instruct 系列）：Hi Robot 的 synthetic data 是 self-instruct 在 robot 场景的实例化

---

## 论文点评

### Strengths

1. **方法 simple 且 scalable**：System1/System2 hierarchy + synthetic data 两个 idea 都 minimal 但 effective，不依赖花哨结构。两层共享 PaliGemma-3B backbone 进一步降低工程复杂度。
2. **Synthetic data pipeline 的 framing 很值得借鉴**：用 generator VLM 从 *(image, skill label)* 反向合成 *(prompt, response)*——这是把 VLA 数据扩展到 instruction-following 维度的通用配方，可迁移到 navigation 等其他领域
3. **Ablation 清晰**：data ablation 与 architecture ablation 各自独立做，互相印证；不像有些工作把 confound 揉一起
4. **跨 embodiment 验证**：UR5e、bimanual ARX、mobile ARX 三平台，避免单平台 cherry-pick 怀疑
5. **Beats GPT-4o by a wide margin**：fine-tuned 3B 模型在 situated grounding 上压过通用大模型，说明 robot-specific data alignment 比模型规模重要

### Weaknesses

1. **Navigation 能力受限**：虽然用了 mobile platform，但 navigation 距离很短，不涉及 building-scale，spatial memory 缺失（论文自己也承认）
2. **High-level 与 low-level 完全独立训练**：没有 end-to-end joint optimization；high-level 不感知 low-level 的失败（drop 物体、grasp 失败时仍发下一条命令）
3. **Synthetic data 的质量没有量化**：generator VLM 引入的 bias / hallucination 没有评估，只能靠下游间接验证
4. **High-level 的 "memory" 完全靠 prompt context**：每次重新推理都从头做，long-horizon 任务里跨多分钟的 context 维护不知道怎么处理
5. **Trigger 策略 ad hoc**：1 秒固定 + 用户介入触发是个工程决策，缺少对 *什么时候应该重新规划* 的原则性研究
6. **Verbal response 的训练监督不清楚**：synthetic 的 utterance $u_t$ 如何评估"质量"？论文没有 ablation

### 可信评估

#### Artifact 可获取性

- **代码**: 未开源（Physical Intelligence 没有为本文单独发 repo；openpi 仓库只发了 [[2410-Pi0|π0]] / π0-FAST / [[2504-Pi05|π0.5]]，不含 Hi Robot 的 high-level）
- **模型权重**: 未发布
- **训练细节**: 仅高层描述（optimizer、batch、超参写了，但 synthetic data 的生成 prompt、generator VLM 选择、数据规模、配比等关键细节没披露）
- **数据集**: 私有（teleop demo + synthetic 都未公开）

#### Claim 可验证性

- ✅ **Hi Robot 在三个任务上 instruction accuracy 高于 GPT-4o + π0 baseline**：定量数字 + 视频 + ablation；项目页有完整 rollout 视频可直观验证
- ✅ **Hierarchy + synthetic data 各自不可或缺**：两个 ablation 独立做，因果归因清晰
- ✅ **Real-time feasibility (10-50 Hz)**：Appendix B.3 给了组件级 latency breakdown
- ⚠️ **"approaches expert human guidance"**：Figure 5 显示有 gap，"approaches" 是定性描述；具体差距取决于任务，部分任务上还有 10-20% IA 的差异
- ⚠️ **"超越 GPT-4o 40%+"**：average over three tasks；单看 sandwich making gap 更大，看 grocery shopping gap 更小，average 数字会平均掉单任务差异
- ⚠️ **Synthetic data 是 "essential"**：仅在论文 evaluator 定义的指标下成立。实际部署时 synthetic data 引入的 bias 可能在论文 setting 之外暴露
- ❌ 无明显的 marketing 修辞——论文整体表述克制

### Notes

#### 对 VLN-VLA Unification 的启示

1. **Hi Robot 是我们 idea "上两层" 的直接验证**：VLM planner（Hi Robot 的 high-level）+ VLA executor（Hi Robot 的 low-level [[2410-Pi0|π0]]）。它证明了 hierarchy 在 real-world manipulation 上 work，且 fine-tuned 小模型胜过 GPT-4o。我们的 idea 只需在中间加 spatial memory 层并扩展 navigation。

2. **Synthetic data generation 可迁移到 Nav+Manip**：用 generator VLM 从少量 demo 自动生成多样化指令的方法，可以用来生成 Nav+Manip 的 composite 指令（如 "go to the kitchen and make me coffee"），解决 Nav+Manip 数据稀缺问题。

3. **Hi Robot 的局限 = 我们的机会**：
   - 无 spatial memory → 加入 ConceptGraphs / MTU3D 式 spatial representation
   - Navigation 范围有限 → 扩展到 building-scale（需要 SLAM）
   - High-level 不感知 low-level 失败 → 加入 value function（[[2511-PiStar06|π*0.6]] 的 Recap）做 failure detection

4. **Hi Robot + [[2511-PiStar06|π*0.6]] + spatial memory ≈ 我们 idea 的完整系统**：
   - Hi Robot 提供 hierarchical VLM-VLA 架构 + synthetic data pipeline
   - [[2511-PiStar06|π*0.6]] 提供 manipulation 的 RL self-improvement
   - Spatial memory 提供 nav + manip 的 shared representation

#### 一些追问

> ❓ generator VLM 的选择对下游性能影响多大？如果换成 open-source VLM，能否复现？这关系到方法的 reproducibility。

> ❓ High-level 的 1 Hz 触发策略在 long-horizon task 上是否会拖慢响应？例如机器人正在 grasp 一个 fragile 物体时被打断，会不会 mid-grasp 切换 command 导致掉落？论文没讨论这种边缘情况。

> ❓ Hi Robot 能不能"主动提问"（如 "你是要 ham 还是 turkey 三明治？"）？合成数据里似乎主要是 user→robot 的 reactive 模式，缺乏 robot 主动 clarify 的训练样本。

### Rating

**分数**：2 - Frontier
**理由**：方法上 hierarchical VLM + VLA 的拆分和 synthetic interaction data pipeline 是 instruction-following 方向目前的代表做法（Strengths 1-3），且跨三种 embodiment 验证（Strengths 4），在 situated grounding 上压 GPT-4o（Strengths 5），属于 Physical Intelligence 路线里必须比较的 baseline。但尚未达到 Foundation 档：代码/权重/合成数据 pipeline 细节均未公开（Artifact 可获取性全部未发布），且该工作已被同团队后续 [[2504-Pi05|π0.5]] 整合覆盖，社区更多被 π0 / π0.5 作为 baseline 引用而非 Hi Robot 本身——Hi Robot 更像 PI 技术路线的中间产物，未形成独立标准。
